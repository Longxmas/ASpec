# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""DSpark training model: DFlash training wrapper + Markov / L1 / confidence losses.

Reuses :class:`DFlashModel`'s anchor sampling, block-causal FlexAttention mask,
and MASK-token noise construction verbatim, then layers on the DSpark training
objective:

  - Markov-biased draft logits (teacher-forced previous token).
  - Cross-entropy against the ground-truth next tokens (hard labels).
  - L1 distribution distillation: ``|softmax(draft) - softmax(target)|`` where the
    target distribution is the frozen LM head applied to the *target's* final
    hidden state at the aligned position (requires ``last_hidden_states``).
  - Confidence head BCE against the empirical per-token accept rate.

Combined: ``ce_alpha*ce + l1_alpha*l1 + confidence_alpha*confidence``.

Loss formulation adapted from DeepSeek's DeepSpec
(deepspec/modeling/dspark/loss.py, MIT). Additive loss terms are returned to
the trainer so normalization covers every rank and gradient-accumulation step.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from aspec.models.dflash import DFlashModel
from aspec.models.ops.acceptance import acceptance_components, acceptance_statistics
from aspec.models.ops.loss import compiled_l1_from_logits, use_sm90_tiled_l1
from aspec.models.ops.output_blocks import (
    empty_prediction_terms,
    output_block_counts,
    output_block_indices,
    prepare_block_layout,
    select_blocks,
)


def _cached_projection_chunks(counts, row_budget, block_size, *, copies, limit=None):
    """Fill a projection-row budget without regrouping original head chunks."""
    selected = set()
    for index in sorted(range(len(counts)), key=counts.__getitem__, reverse=True):
        cost = copies * counts[index] * block_size
        if cost > 0 and cost <= row_budget:
            selected.add(index)
            row_budget -= cost
            if limit is not None and len(selected) == limit:
                break
    return selected


class DSparkModel(DFlashModel):
    """DSpark training wrapper (DFlash backbone + Markov/L1/confidence heads)."""

    def __init__(
        self,
        draft_model,
        block_size: int = 7,
        num_anchors: int = 512,
        loss_decay_gamma: float = 4.0,
        ce_loss_alpha: float = 0.1,
        l1_loss_alpha: float = 0.9,
        confidence_head_alpha: float = 1.0,
        attention_backend: str = "flex_attention",
        draft_context_parallel_size: int = 1,
        deterministic_anchor_sampling: bool = False,
        anchor_seed: int = 0,
        anchor_sampling_scope: str = "row",
        l1_target_sampling: tuple[float, float, int] = (1.0, 1.0, -1),
        l1_draft_sampling: tuple[float, float, int] = (1.0, 1.0, -1),
    ):
        # Reuse DFlash anchor/mask/noise machinery. The "decay" objective drives
        # the per-within-block position weighting shared by all DSpark terms.
        super().__init__(
            draft_model=draft_model,
            block_size=block_size,
            num_anchors=num_anchors,
            loss_objective="decay",
            dpace_alpha=0.5,
            loss_decay_gamma=loss_decay_gamma,
            attention_backend=attention_backend,
            draft_context_parallel_size=draft_context_parallel_size,
            deterministic_anchor_sampling=deterministic_anchor_sampling,
            anchor_seed=anchor_seed,
            anchor_sampling_scope=anchor_sampling_scope,
        )
        self.ce_loss_alpha = float(ce_loss_alpha)
        self.l1_loss_alpha = float(l1_loss_alpha)
        self.confidence_head_alpha = float(confidence_head_alpha)
        self.l1_sampling = (l1_target_sampling, l1_draft_sampling)
        self.l1_sampling_enabled = self.l1_sampling != ((1.0, 1.0, -1), (1.0, 1.0, -1))
        self.l1_loss_backend = None
        self.l1_kernel_invocations = {}
        self._use_l1_kernel_optimizations = True
        # Internal dense oracle for correctness checks, not a sampling policy.
        self._compact_output_blocks = True
        self._compact_backbone_blocks = True
        self._reuse_head_projections = True
        self._reuse_context_teacher_projection = True
        self._reuse_context_draft_projection = True
        # Execution-only byte bound: four largest-chunk projection tensors.
        # Smaller original chunks may fill spare space without raising it.
        self._head_projection_cache_chunks = 2
        self._fill_head_projection_cache = True
        self._reuse_l1_statistics = True
        self._prepared_output_counts = None

    def _select_backbone_blocks(self, keep_mask, owners, rank):
        if not self._compact_output_blocks:
            return super()._select_backbone_blocks(keep_mask, owners, rank)
        selection, counts = prepare_block_layout(keep_mask, owners, rank, self.num_anchors)
        self._prepared_output_counts = counts
        return selection

    def _is_csa_indexer_warmup(self) -> bool:
        config = getattr(self.draft_model, "config", None)
        return getattr(config, "csa_training_stage", None) == "indexer_warmup"

    def _indexer_warmup_kind(self) -> str | None:
        config = getattr(self.draft_model, "config", None)
        if getattr(config, "csa_training_stage", None) == "indexer_warmup":
            return "csa"
        if getattr(config, "qsa_training_stage", None) == "indexer_warmup":
            return "qsa"
        if getattr(config, "dsa_training_stage", None) == "indexer_warmup":
            return "dsa"
        return None

    @property
    def uses_target_hidden_states(self) -> bool:
        if self._indexer_warmup_kind() is not None:
            # The warm-up phase optimizes only the detached indexer KL.  Loading
            # verifier hidden states would retain and transfer target activations
            # that cannot affect any trainable parameter.
            return False
        # The confidence head regresses onto an accept rate derived from the
        # verifier distribution, so the L1 weight alone does not cover it.
        return super().uses_target_hidden_states or (
            self.draft_model.confidence_head is not None and self.confidence_head_alpha > 0
        )

    def _decay_weights(self, device: torch.device) -> torch.Tensor:
        """exp(-k/gamma) over within-block position k (DeepSpec convention).

        Every slot 0..B-1 is a real prediction here, so slot 0 (the first
        predicted token) gets weight 1.0 and later slots decay.
        """
        k = torch.arange(self.block_size, device=device).view(1, 1, -1)
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            return torch.exp(-k.float() / self.loss_decay_gamma)
        return torch.ones_like(k, dtype=torch.float32)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor] | torch.Tensor,
        loss_mask: torch.Tensor,
        lm_head_weight: torch.Tensor,
        last_hidden_states: Optional[torch.Tensor] = None,
        ctx_doc_ids: Optional[torch.Tensor] = None,
        base_position_ids: Optional[torch.Tensor] = None,
        packed_context_positions: Optional[torch.Tensor] = None,
        injected_anchors: Optional[torch.Tensor] = None,
        injected_keep_mask: Optional[torch.Tensor] = None,
        anchor_seed: Optional[int] = None,
        packed_document_lengths: Optional[tuple[int, ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """DSpark training forward.

        Returns DFlashModel.forward's metrics plus a dict of detached per-component
        ``(numerator, denominator)`` pairs for the loss components and analytical
        acceptance observations, plus the additive loss terms. ``loss`` is the
        combined ce+l1+confidence objective. Per-position cross-entropy/accuracy
        and probability-overlap observations are separate metrics. DSparkTrainer
        unpacks the extra elements; DFlash's aggregation is reused unchanged.
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        self._prepared_output_counts = None

        router_logits = []
        csa_losses = []
        qsa_losses = []
        dsa_losses = []
        # ---- Shared DFlash backbone (steps 1-6) → draft hidden states ----
        draft_hidden, anchor_positions, block_keep_mask, n_blocks = self._draft_backbone(
            input_ids,
            hidden_states_list,
            loss_mask,
            ctx_doc_ids=ctx_doc_ids,
            base_position_ids=base_position_ids,
            packed_context_positions=packed_context_positions,
            injected_anchors=injected_anchors,
            injected_keep_mask=injected_keep_mask,
            anchor_seed=anchor_seed,
            router_logits_out=router_logits,
            csa_losses_out=csa_losses,
            qsa_losses_out=qsa_losses,
            dsa_losses_out=dsa_losses,
            packed_document_lengths=packed_document_lengths,
            attention_mask=attention_mask,
        )
        # ---- Labels + eval mask (DeepSpec convention) ----
        # Slot j predicts the token at anchor+j+1 (the real anchor token seeds
        # slot 0). All block_size slots are supervised — there is no masked
        # anchor slot, unlike DFlash.
        label_offsets = torch.arange(1, self.block_size + 1, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets  # [B, nb, bs]
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        if ctx_doc_ids is not None:
            anchor_docs = ctx_doc_ids.gather(1, anchor_positions)
            label_docs = torch.gather(
                ctx_doc_ids.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
            )
            valid_label_mask = valid_label_mask & (label_docs == anchor_docs.unsqueeze(-1))
        safe_label_indices = torch.where(
            block_keep_mask.unsqueeze(-1),
            safe_label_indices,
            torch.zeros_like(safe_label_indices),
        )

        # eval mask = contiguous supervised prefix per block (DeepSpec
        # build_eval_mask): block kept, label in-bounds, target token supervised,
        # then cumprod so a gap truncates the rest of the block.
        target_loss_mask = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
        )
        eval_bool = block_keep_mask.unsqueeze(-1) & valid_label_mask & (target_loss_mask > 0.5)
        eval_bool = eval_bool.to(torch.int32).cumprod(dim=-1).bool()
        eval_mask = eval_bool.float()  # [B, nb, bs]

        decay_weight_mask = eval_mask * self._decay_weights(device)
        local_den = decay_weight_mask.sum()

        warmup_kind = self._indexer_warmup_kind()
        if warmup_kind is not None:
            if warmup_kind == "csa":
                numerator, loss_components = self._csa_training_loss(
                    csa_losses, decay_weight_mask, local_den,
                )
                key = "csa_indexer_loss"
            elif warmup_kind == "qsa":
                numerator, loss_components = self._qsa_training_loss(
                    qsa_losses, decay_weight_mask, local_den,
                )
                key = "qsa_indexer_loss"
            else:
                numerator, loss_components = self._dsa_training_loss(
                    dsa_losses, decay_weight_mask, local_den,
                )
                key = "dsa_indexer_loss"
            if key not in loss_components:
                raise RuntimeError(
                    f"{warmup_kind.upper()} indexer warm-up produced no indexer KL objective"
                )
            loss = numerator / local_den.clamp(min=1e-6)
            zero = draft_hidden.new_zeros((), dtype=torch.float32)
            zero_per_position = draft_hidden.new_zeros(
                self.block_size, dtype=torch.float32
            )
            return (
                loss,
                zero,
                zero_per_position,
                zero_per_position.clone(),
                zero_per_position.clone(),
                loss_components,
                (numerator, local_den.detach()),
            )

        hidden_4d = draft_hidden.view(bsz, n_blocks, self.block_size, -1)
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
        )  # [B, nb, bs]
        anchor_token_ids = torch.gather(input_ids, 1, anchor_positions)  # [B, nb]
        prev_token_ids = torch.cat([anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]], dim=-1)
        aligned_hidden = None
        if self.uses_target_hidden_states:
            if last_hidden_states is None:
                raise ValueError(
                    "DSpark L1/confidence losses require target last_hidden_states; set "
                    "inference.store_last_hidden_states=true in the run config."
                )
            # target distribution for the token at label_indices = target LM head
            # applied to the target hidden one position earlier (anchor+j).
            tgt_idx = (safe_label_indices - 1).clamp(min=0)  # [B, nb, bs]
            aligned_hidden = self._align_target_hidden_states(
                last_hidden_states,
                tgt_idx,
                global_sequence_length=seq_len,
                packed_context_positions=packed_context_positions,
            )
            aligned_hidden = aligned_hidden.view(bsz, n_blocks, self.block_size, -1)

        # Keep the owner-sharded teacher gather above collective-compatible even
        # when one CP rank has no valid queries. Only local output work shrinks;
        # router/indexer losses below still see the original complete query set.
        self.loss_head_original_blocks_observed = n_blocks

        # Bound vocabulary-sized intermediates by original anchor-sized chunks.
        # Checkpoint only the head/loss for larger packed query sets;
        # attention/backbone execute once and additive normalization is unchanged.
        from torch.utils.checkpoint import checkpoint

        chunk_size = self.num_anchors
        # Never regroup valid blocks across the original head-chunk boundaries:
        # that changes the grouping/order of BF16 parameter-gradient accumulation.
        # One shape synchronization covers every chunk, not one per document.
        counts = self._prepared_output_counts
        if counts is None:
            counts = (output_block_counts(block_keep_mask, chunk_size)
                      if self._compact_output_blocks else
                      [min(chunk_size, n_blocks - start) for start in range(0, n_blocks, chunk_size)])
        self.loss_head_blocks_observed = sum(counts)
        self.loss_head_chunks_observed = sum(count > 0 for count in counts)
        # Retain frozen LM-head projections within the existing byte budget.
        # These are already BF16 intermediates in the normal path;
        # reuse avoids backward recomputation without saving logits for the whole
        # packed row or changing where trainable BF16 gradients are scaled.
        checkpoint_heads = n_blocks > chunk_size and torch.is_grad_enabled()
        largest_chunk = (
            max(range(len(counts)), key=counts.__getitem__)
            if checkpoint_heads and aligned_hidden is not None
            and not lm_head_weight.requires_grad and not aligned_hidden.requires_grad
            and self._reuse_head_projections
            and any(counts) else None
        )
        # Short rows often predict the same frozen Target position repeatedly.
        cache_limit = self._head_projection_cache_chunks
        cache_tokens = (2 * cache_limit * counts[largest_chunk] * self.block_size
                        if largest_chunk is not None else 0)
        # Spend the retained-logit budget on one context projection instead
        # of a Draft/Target chunk pair when that removes more projection work.
        # Gather INSIDE checkpoint: retaining each expanded query tensor outside
        # it would accidentally cache the entire packed row's vocabulary logits.
        # Owner-sharded CP teachers keep the existing collective-compatible path;
        # never gather a full context merely to enable this local optimization.
        chunk_limit = None if self._fill_head_projection_cache else cache_limit
        regular_chunks = _cached_projection_chunks(
            counts, cache_tokens, self.block_size, copies=2, limit=chunk_limit,
        ) if largest_chunk is not None else set()
        use_context = (
            largest_chunk is not None and self._reuse_context_teacher_projection
            and self.draft_context_parallel_size == 1
            and last_hidden_states.shape[1] == seq_len
            and seq_len <= cache_tokens
            and seq_len < 2 * (sum(counts) - counts[largest_chunk]) * self.block_size
        )
        # Choose which projections to retain, never change execution order or
        # merge chunks. Context logits consume the same budget as chunk logits.
        context_chunks = set()
        if use_context and self._reuse_context_draft_projection:
            context_chunks = _cached_projection_chunks(
                counts, cache_tokens - seq_len, self.block_size, copies=1, limit=chunk_limit,
            )
        if use_context and cache_limit > 1:
            # Compare total projected rows, including checkpoint replay. A
            # larger budget alone does not make full-context reuse profitable.
            regular_rows = (4 * sum(counts) - 2 * sum(counts[i] for i in regular_chunks)) * self.block_size
            context_rows = seq_len + (2 * sum(counts) - sum(counts[i] for i in context_chunks)) * self.block_size
            use_context = context_rows < regular_rows
        if not checkpoint_heads:
            # A native BSHD row has one anchor chunk even at B>1. Frozen
            # Target-position reuse can still avoid repeated vocabulary work;
            # Draft logits already execute once and need no additional cache.
            # Count actual B*S rows against a batch-independent retained-logit
            # bound, rather than multiplying the cache allowance by MBS.
            context_budget = 2 * cache_limit * max(counts, default=0) * self.block_size
            use_context = (
                self._reuse_head_projections and self._reuse_context_teacher_projection
                and aligned_hidden is not None and not aligned_hidden.requires_grad
                and not lm_head_weight.requires_grad
                and self.draft_context_parallel_size == 1
                and last_hidden_states.shape[1] == seq_len
                and bsz * seq_len <= context_budget
                and seq_len < sum(counts) * self.block_size
            )
        cached_chunks = context_chunks if use_context else regular_chunks
        context_teacher = None
        if use_context:
            context_teacher = F.linear(
                last_hidden_states.reshape(-1, last_hidden_states.shape[-1]), lm_head_weight
            ).view(bsz, seq_len, lm_head_weight.shape[0])
        self.loss_head_cached_context_tokens_observed = (
            bsz * seq_len if context_teacher is not None else 0
        )
        self.loss_head_cached_teacher_blocks_observed = 0
        self.loss_head_cached_draft_blocks_observed = 0
        head_terms = []
        for chunk, start in enumerate(range(0, n_blocks, chunk_size)):
            stop = min(start + chunk_size, n_blocks)
            if not counts[chunk]:
                empty_terms = empty_prediction_terms(
                    hidden_4d[:, start:stop],
                    (self.draft_model.markov_head,
                     self.draft_model.confidence_head if self.confidence_head_alpha > 0 else None),
                )
                head_terms.append((*empty_terms, hidden_4d.new_zeros(
                    (2, self.block_size + 1), dtype=torch.float32,
                )))
                continue
            values = (hidden_4d[:, start:stop], target_ids[:, start:stop],
                      prev_token_ids[:, start:stop], lm_head_weight,
                      aligned_hidden[:, start:stop] if aligned_hidden is not None else None,
                      eval_mask[:, start:stop], decay_weight_mask[:, start:stop])
            selection = output_block_indices(block_keep_mask[:, start:stop], counts[chunk])
            if selection is not None:
                values = tuple(value if value is None or index == 3 else select_blocks(value, selection)
                               for index, value in enumerate(values))
            cached_teacher = None
            cached_draft = None
            cached_positions = None
            if context_teacher is not None:
                cached_teacher = context_teacher
                cached_positions = tgt_idx[:, start:stop]
                if selection is not None:
                    cached_positions = select_blocks(cached_positions, selection)
                self.loss_head_cached_teacher_blocks_observed += counts[chunk]
            if chunk in cached_chunks:
                if context_teacher is None:
                    teacher = values[4]
                    cached_teacher = F.linear(
                        teacher.reshape(-1, teacher.shape[-1]), lm_head_weight
                    ).view(*teacher.shape[:-1], lm_head_weight.shape[0])
                    self.loss_head_cached_teacher_blocks_observed += counts[chunk]
                hidden = values[0]
                cached_draft = F.linear(
                    hidden.reshape(-1, hidden.shape[-1]), lm_head_weight
                ).view(*hidden.shape[:-1], lm_head_weight.shape[0])
                self.loss_head_cached_draft_blocks_observed += counts[chunk]
            # One cache per original chunk and forward, never per model/step.
            # Keep only small FP32 L1 statistics across checkpoint replay; full
            # vocabulary logits still follow the projection budget above.
            loss_cache = {} if checkpoint_heads and self._reuse_l1_statistics else None
            values = (*values, cached_teacher, cached_draft, cached_positions, loss_cache)
            if checkpoint_heads:
                terms = checkpoint(self._prediction_terms, *values,
                                   use_reentrant=False, preserve_rng_state=False)
            else:
                terms = self._prediction_terms(*values)
            head_terms.append(terms)
        ce_num, l1_num, conf_num, ce_position_sum, correct_position_sum, acceptance_totals = (
            sum(parts[1:], parts[0]) for parts in zip(*head_terms, strict=True)
        )

        # The trainer pools these additive terms over the entire optimizer window
        # before compensating for FSDP's mean gradient reduction.
        loss_numerator = (
            self.ce_loss_alpha * ce_num
            + self.l1_loss_alpha * l1_num
            + self.confidence_head_alpha * conf_num
        )
        moe_numerator, moe_components = self._moe_training_loss(
            router_logits, decay_weight_mask, local_den,
        )
        csa_numerator, csa_components = self._csa_training_loss(
            csa_losses, decay_weight_mask, local_den,
        )
        qsa_numerator, qsa_components = self._qsa_training_loss(
            qsa_losses, decay_weight_mask, local_den,
        )
        dsa_numerator, dsa_components = self._dsa_training_loss(
            dsa_losses, decay_weight_mask, local_den,
        )
        loss_numerator = (
            loss_numerator + moe_numerator + csa_numerator + qsa_numerator + dsa_numerator
        )
        loss = loss_numerator / local_den.clamp(min=1e-6)

        # Per-component terms for logging only — lets you watch L1 fall while the
        # greedy-CE proxy plateaus. Reported as numerator/denominator rather than a
        # local mean so the trainer can pool them over the same window as the
        # objective; averaging per-micro-batch means would let these curves drift
        # away from the loss actually optimized.
        component_denominator = local_den.detach()
        loss_components = {
            "ce_loss": (ce_num.detach(), component_denominator),
            "l1_loss": (l1_num.detach(), component_denominator),
            "confidence_loss": (conf_num.detach(), component_denominator),
            **moe_components,
            **csa_components,
            **qsa_components,
            **dsa_components,
        }
        if aligned_hidden is not None:
            loss_components.update(acceptance_components(acceptance_totals))

        # ---- Metrics (cross-entropy based; all block_size slots are productive) ----
        with torch.no_grad():
            accuracy = correct_position_sum.sum() / eval_mask.sum().clamp(min=1e-6)
            count_per_position = eval_mask.sum(dim=(0, 1))
            count_pp = count_per_position.clamp(min=1.0)
            loss_per_position = ce_position_sum / count_pp
            acc_per_position = correct_position_sum / count_pp

        return (
            loss,
            accuracy,
            loss_per_position,
            acc_per_position,
            count_per_position,
            loss_components,
            (loss_numerator, local_den.detach()),
        )

    def _prediction_terms(self, hidden, target_ids, prev_ids, head_weight,
                          aligned_target_hidden, eval_mask, weights, cached_target_logits=None,
                          cached_draft_logits=None, cached_target_positions=None, loss_cache=None):
        """Additive DSpark head terms for a bounded block slice."""
        # Explicit token-major GEMM: a block slice can be non-contiguous even
        # though the complete hidden stream is contiguous. Do not broadcast a
        # vocabulary projection over many tiny [block_size, hidden] matrices.
        logits = cached_draft_logits if cached_draft_logits is not None else F.linear(
            hidden.reshape(-1, hidden.shape[-1]), head_weight
        ).view(
            *hidden.shape[:-1], head_weight.shape[0]
        )
        if self.draft_model.markov_head is not None:
            logits = self.draft_model.markov_head.apply_block_logits(logits, token_ids=prev_ids)
        vocab = logits.shape[-1]
        ce = F.cross_entropy(logits.reshape(-1, vocab), target_ids.reshape(-1),
                             reduction="none").view_as(weights)
        ce_num = (ce * weights).sum()
        l1_num = logits.new_zeros((), dtype=torch.float32)
        accept_rate = None
        if aligned_target_hidden is not None:
            indexed_teacher = (
                not self.l1_sampling_enabled
                and self._use_l1_kernel_optimizations and cached_target_positions is not None
                and use_sm90_tiled_l1(logits, cached_target_logits, True)
            )
            target_indices = None
            if indexed_teacher:
                batch, context, _ = cached_target_logits.shape
                target_indices = cached_target_positions.reshape(-1)
                if batch > 1:
                    offsets = torch.arange(batch, device=logits.device, dtype=torch.int64) * context
                    target_indices = (cached_target_positions.flatten(1) + offsets[:, None]).reshape(-1)
                target_logits = cached_target_logits.view(-1, vocab)
            elif cached_target_positions is not None:
                positions = cached_target_positions.flatten(1).unsqueeze(-1)
                target_logits = cached_target_logits.gather(
                    1, positions.expand(-1, -1, vocab)
                ).view_as(logits)
            else:
                target_logits = cached_target_logits if cached_target_logits is not None else F.linear(
                    aligned_target_hidden.reshape(-1, aligned_target_hidden.shape[-1]), head_weight
                ).view_as(logits)
            if not self.l1_sampling_enabled and (
                indexed_teacher or use_sm90_tiled_l1(logits, target_logits)
            ):
                from aspec.models.ops.tiled_l1 import tiled_l1_from_logits

                l1 = tiled_l1_from_logits(logits, target_logits, target_indices=target_indices,
                                         _replay_cache=loss_cache)
                self.l1_loss_backend = "tiled_fp32"
                kernel = ("indexed_" if indexed_teacher else "") + self.l1_loss_backend
                self.l1_kernel_invocations[kernel] = self.l1_kernel_invocations.get(kernel, 0) + 1
            else:
                l1 = (compiled_l1_from_logits(logits, target_logits, *self.l1_sampling)
                      if self.l1_sampling_enabled else compiled_l1_from_logits(logits, target_logits))
                self.l1_loss_backend = "torch_compiled"
            if self.l1_loss_alpha > 0:
                l1_num = (l1 * weights).sum()
            accept_rate = (1.0 - 0.5 * l1).clamp(0.0, 1.0)
        conf_num = logits.new_zeros((), dtype=torch.float32)
        if self.draft_model.confidence_head is not None and self.confidence_head_alpha > 0:
            if self.draft_model.confidence_head_with_markov:
                prev_emb = self.draft_model.markov_head.get_prev_embeddings(prev_ids).to(hidden.dtype)
                conf_features = torch.cat([hidden, prev_emb], dim=-1)
            else:
                conf_features = hidden
            prediction = self.draft_model.confidence_head(conf_features).float()
            conf_num = (F.binary_cross_entropy_with_logits(
                prediction, accept_rate.detach(), reduction="none"
            ) * weights).sum()
        with torch.no_grad():
            correct = (logits.argmax(-1) == target_ids) & (eval_mask > 0.5)
            position_loss = (ce * eval_mask).sum((0, 1))
            position_correct = correct.float().sum((0, 1))
            acceptance_totals = (
                acceptance_statistics(accept_rate, eval_mask)
                if accept_rate is not None else logits.new_zeros(
                    (2, self.block_size + 1), dtype=torch.float32,
                )
            )
        return ce_num, l1_num, conf_num, position_loss, position_correct, acceptance_totals
