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

"""DFlash training model: wraps the DFlash draft model with training-specific logic.

Handles anchor sampling, block-causal mask generation, noise input construction,
and cross-entropy loss with exponential decay weighting.

Matches SpecForge's OnlineDFlashModel (specforge/core/dflash.py).
"""

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from aspec.config.draft_attention import (
    resolve_draft_attention,
    resolve_draft_head_layout,
)
from aspec.config.draft_execution import (
    is_cudnn_flex_backend,
    is_fa4_backend,
    resolve_draft_execution_capability,
)
from aspec.models.ops.anchors import (
    document_anchor_slots,
    sample_anchor_positions,
    sample_document_anchor_positions,
    validate_injected_anchors,
)
from aspec.models.ops.flex_attention import compile_friendly_create_block_mask

_VALID_DFLASH_LOSS_OBJECTIVES = {"decay", "dpace"}


def _dpace_position_weights(confidences: torch.Tensor, alpha: float) -> torch.Tensor:
    """Compute detached D-PACE weights from per-position draft confidences."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"dflash_dpace_alpha must be in [0, 1], got {alpha}")

    with torch.no_grad():
        smoothed = (1.0 - alpha) * confidences.float() + alpha
        prefix_products = torch.cumprod(smoothed, dim=-1)
        weights = torch.flip(
            torch.cumsum(torch.flip(prefix_products, dims=[-1]), dim=-1),
            dims=[-1],
        )
        return weights.to(dtype=confidences.dtype)


def _create_dflash_mask_mod(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    ctx_len: int,
    block_size: int,
    is_causal: bool = False,
    sliding_window: int | None = None,
    ctx_doc_ids: Optional[torch.Tensor] = None,
    context_window: int | None = None,
):
    """Create a mask_mod function for DFlash block-causal attention.

    KV: [Context (ctx_len tokens) | Block_0 | Block_1 | ... | Block_{n-1}]
    Q:  [Block_0 | Block_1 | ... | Block_{n-1}]

    Rules:
      1. Each block sees context strictly before its anchor (kv_idx < anchor_pos)
      2. Intra-block attention follows is_causal and sliding_window
      3. Different blocks are invisible to each other
      4. Invalid blocks (block_keep_mask=False) see nothing
    """
    num_anchors = anchor_positions.shape[1]
    packed = ctx_doc_ids is not None

    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        anchor_pos = anchor_positions[b, q_block_id]

        is_context = kv_idx < ctx_len
        mask_context = is_context & (kv_idx < anchor_pos)
        if packed:
            anchor_doc = ctx_doc_ids[b, anchor_pos]
            context_index = torch.where(is_context, kv_idx, torch.zeros_like(kv_idx))
            context_doc = ctx_doc_ids[b, context_index]
            mask_context = mask_context & (anchor_doc >= 0) & (context_doc == anchor_doc)

        is_draft = kv_idx >= ctx_len
        kv_block_id = (kv_idx - ctx_len) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)

        q_offset = q_idx % block_size
        kv_offset = (kv_idx - ctx_len) % block_size
        if is_causal:
            mask_draft = mask_draft & (kv_offset <= q_offset)
        if context_window is not None:
            mask_context = mask_context & (kv_idx >= anchor_pos - context_window)
        if sliding_window is not None:
            query_position = anchor_pos + q_offset
            mask_context = mask_context & (query_position - kv_idx < sliding_window)
            mask_draft = mask_draft & (
                abs(query_position - (anchor_pos + kv_offset)) < sliding_window
            )

        is_valid_block = block_keep_mask[b, q_block_id]
        return (mask_context | mask_draft) & is_valid_block

    suffix = "_packed" if packed else ""
    dflash_mask_mod.__name__ = (
        f"dflash_mask_A{num_anchors}_B{block_size}_C{ctx_len}{suffix}"
    )
    return dflash_mask_mod


class DFlashModel(nn.Module):
    """DFlash training wrapper.

    Wraps the DFlash draft model with training-specific logic:
      - Random anchor sampling with block_keep_mask
      - Block-causal attention mask via FlexAttention
      - Noise input construction (anchor + MASK)
      - Cross-entropy loss with configurable position weighting
      - Per-position loss_mask application
    """

    prediction_start = 1

    def __init__(
        self,
        draft_model,
        block_size: int = 16,
        num_anchors: int = 512,
        loss_objective: str = "decay",
        dpace_alpha: float = 0.5,
        loss_decay_gamma: float = 7.0,
        ce_loss_alpha: float = 1.0,
        l1_loss_alpha: float = 0.0,
        attention_backend: str = "flex_attention",
        draft_context_parallel_size: int = 1,
        deterministic_anchor_sampling: bool = False,
        anchor_seed: int = 0,
        anchor_sampling_scope: str = "row",
    ):
        super().__init__()
        loss_objective = loss_objective.lower()
        if loss_objective not in _VALID_DFLASH_LOSS_OBJECTIVES:
            valid = ", ".join(sorted(_VALID_DFLASH_LOSS_OBJECTIVES))
            raise ValueError(
                f"Unknown DFlash loss objective {loss_objective!r}; expected one of {valid}"
            )
        if not 0.0 <= dpace_alpha <= 1.0:
            raise ValueError(f"dflash_dpace_alpha must be in [0, 1], got {dpace_alpha}")

        self.draft_model = draft_model
        self.mtp_conditioning = getattr(getattr(draft_model, "config", None), "mtp_conditioning", None)
        self.prediction_start = type(self).prediction_start + int(self.mtp_conditioning is not None)
        self.block_size = block_size
        self.num_anchors = num_anchors
        if anchor_sampling_scope not in {"row", "sample"}:
            raise ValueError("anchor_sampling_scope must be row or sample")
        self.anchor_sampling_scope = anchor_sampling_scope
        self.deterministic_anchor_sampling = deterministic_anchor_sampling
        self.anchor_seed = int(anchor_seed)
        self.loss_objective = loss_objective
        self.dpace_alpha = dpace_alpha
        self.loss_decay_gamma = loss_decay_gamma
        self.ce_loss_alpha = float(ce_loss_alpha)
        self.l1_loss_alpha = float(l1_loss_alpha)
        if self.mtp_conditioning and (block_size <= self.prediction_start
                                     or self.loss_objective != "decay"):
            raise ValueError("MTP conditioning requires decay loss and an unknown suffix slot")
        execution = resolve_draft_execution_capability(
            attention_backend, draft_context_parallel_size
        )
        self.attention_backend = execution.attention_backend
        self.draft_context_parallel_size = execution.context_parallel_size
        if self.mtp_conditioning and (self.draft_context_parallel_size != 1
                                     or anchor_sampling_scope != "row"):
            raise ValueError("MTP conditioning currently requires CP1 row anchors")
        self.direct_cp_backend = execution.direct_cp_backend
        from aspec.config.draft_attention import (
            draft_context_shard_alignment,
            resolve_draft_attention,
        )

        draft_config = getattr(self.draft_model, "config", None)
        gqa_mode = getattr(draft_config, "cudnn_gqa_mode", "auto")
        if gqa_mode != "auto" and not is_cudnn_flex_backend(self.attention_backend):
            raise ValueError("Explicit cudnn_gqa_mode requires attention_backend=cudnn_flex_attention")
        if gqa_mode != "auto" and (
            getattr(draft_config, "model_type", None) == "k3_dspark"
            or getattr(draft_config, "attention_mode", "gqa") == "mla"
        ):
            raise NotImplementedError("Explicit cudnn_gqa_mode is not supported for MLA")
        if gqa_mode == "internal" and self.draft_context_parallel_size != 1:
            raise NotImplementedError("internal GQA backward currently requires CP=1")
        self.context_shard_alignment = (
            draft_context_shard_alignment(resolve_draft_attention(draft_config.to_dict()))
            if draft_config is not None
            else 1
        )
        if is_cudnn_flex_backend(self.attention_backend):
            configure_backend = getattr(self.draft_model, "set_attention_backend", None)
            if configure_backend is None:
                raise NotImplementedError(
                    "the selected draft model does not implement "
                    "attention_backend=cudnn_flex_attention"
                )
            configure_backend(self.attention_backend)
        if self.direct_cp_backend:
            if self.num_anchors % self.draft_context_parallel_size:
                raise ValueError(
                    f"num_anchors ({self.num_anchors}) must be divisible by draft "
                    f"CP size ({self.draft_context_parallel_size})"
                )
            from aspec.utils.distributed import get_direct_cp_group

            self.direct_cp_group = get_direct_cp_group()
            if (
                self.draft_context_parallel_size > 1
                and self.direct_cp_group is None
            ):
                raise RuntimeError("Direct CP group has not been initialized")
        else:
            self.direct_cp_group = None

    @property
    def uses_target_distribution(self) -> bool:
        config = getattr(self.draft_model, "config", None)
        if getattr(config, "dsa_training_stage", None) == "indexer_warmup":
            return False
        return self.l1_loss_alpha > 0

    @property
    def uses_target_hidden_states(self) -> bool:
        return self.uses_target_distribution or bool(
            self.mtp_conditioning and self.mtp_conditioning["mode"] == "hidden"
            and self.mtp_conditioning.get("feature_source", "draft") == "draft"
        )

    def _prepare_draft_prefix(self, input_ids, last_hidden_states, attention_mask, draft_prefix,
                              *packed_metadata):
        if not self.mtp_conditioning:
            if draft_prefix is not None:
                raise ValueError("Unconditioned Draft does not accept an MTP prefix")
            return None
        if any(value is not None for value in packed_metadata):
            raise ValueError("MTP conditioning does not accept packed or sharded documents")
        if input_ids.shape[0] != 1:
            raise ValueError("MTP conditioning currently requires microbatch size one")
        if draft_prefix is None:
            from aspec.models.dmtp import training_mtp_prefix

            return training_mtp_prefix(self.draft_model, input_ids, last_hidden_states, attention_mask)
        return draft_prefix

    def _sample_anchor_positions(
        self,
        seq_len: int,
        loss_mask: torch.Tensor,
        device: torch.device,
        ctx_doc_ids: Optional[torch.Tensor] = None,
        injected_anchors: Optional[torch.Tensor] = None,
        injected_keep_mask: Optional[torch.Tensor] = None,
        anchor_seed: Optional[int] = None,
        packed_document_lengths: Optional[tuple[int, ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        per_document = self.anchor_sampling_scope == "sample" and ctx_doc_ids is not None
        if per_document and packed_document_lengths is None:
            raise ValueError("Sample anchor sampling requires packed_document_lengths from the collator")
        if (injected_anchors is None) != (injected_keep_mask is None):
            raise ValueError("Anchor replay requires both injected_anchors and injected_keep_mask")
        if self.mtp_conditioning:
            # There is no verifier h[-1] for the first document token. Keep
            # this exclusion in sampling/replay, not a stochastic runtime error.
            loss_mask = loss_mask.clone()
            loss_mask[:, 0] = 0
        if injected_anchors is not None:
            return validate_injected_anchors(
                injected_anchors, injected_keep_mask,
                loss_mask=loss_mask, num_anchors=(document_anchor_slots(
                    packed_document_lengths, self.num_anchors, self.block_size,
                    self.draft_context_parallel_size,
                )[2] if per_document else self.num_anchors),
                block_size=self.block_size, ctx_doc_ids=ctx_doc_ids,
                allow_partial_last_target=per_document,
            )
        generator = None
        if self.deterministic_anchor_sampling:
            generator = torch.Generator(device=device).manual_seed(
                self.anchor_seed if anchor_seed is None else int(anchor_seed)
            )
        if per_document:
            return sample_document_anchor_positions(
                loss_mask, ctx_doc_ids, packed_document_lengths,
                self.num_anchors, self.block_size, generator,
                alignment=self.draft_context_parallel_size,
            )
        if ctx_doc_ids is not None:
            if ctx_doc_ids.shape != loss_mask.shape:
                raise ValueError("ctx_doc_ids and loss_mask must have identical shapes")
            effective_mask = torch.zeros_like(loss_mask)
            usable = max(seq_len - self.block_size, 0)
            if usable:
                head = ctx_doc_ids[:, :usable]
                tail = ctx_doc_ids[:, self.block_size : self.block_size + usable]
                same_document = (head == tail) & (head >= 0)
                effective_mask[:, :usable] = loss_mask[:, :usable] * same_document
            loss_mask = effective_mask
        return sample_anchor_positions(
            seq_len, loss_mask, self.num_anchors, self.block_size, device,
            generator=generator, attention_mask=attention_mask,
        )

    def _create_position_ids(
        self,
        anchor_positions: torch.Tensor,
        seq_len: int,
        base_position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create position IDs for context and draft tokens."""
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device

        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        if base_position_ids is not None:
            context_position_ids = base_position_ids.to(device=device, dtype=torch.long).contiguous()
            anchor_base = torch.gather(context_position_ids, 1, anchor_positions)
            draft_position_ids = anchor_base.unsqueeze(-1) + offsets
            return context_position_ids, draft_position_ids.view(bsz, -1)

        # Materialize the small position metadata once per forward, shared by
        # all layers. The fused Q/K adapter flattens contiguous positions; an
        # expanded batch has stride zero when B>1 and cannot be viewed flat.
        context_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1).contiguous()
        draft_position_ids = anchor_positions.unsqueeze(-1) + offsets
        draft_position_ids = draft_position_ids.view(bsz, -1)

        return context_position_ids, draft_position_ids

    def _create_noise_embed(
        self,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Create noise embeddings: anchor token at block starts, MASK elsewhere.

        Matches SpecForge's OnlineDFlashModel._create_noise_embed().
        """
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        noise_ids = torch.full(
            (bsz, n * bs), self.draft_model.mask_token_id, dtype=torch.long, device=device
        )

        block_starts = torch.arange(n, device=device) * bs
        block_starts = block_starts.unsqueeze(0).expand(bsz, -1)

        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)

        flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)
        noise_ids[flat_batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.draft_model.mask_token_id, dtype=torch.long, device=device),
        )

        return self.draft_model.embed_tokens(noise_ids)

    def _condition_noise_embed(self, embeddings, input_ids, anchors, keep, draft_prefix):
        if self.mtp_conditioning:
            from aspec.models.dmtp import condition_noise_embeddings

            return condition_noise_embeddings(self.draft_model, embeddings, input_ids,
                                                anchors, keep, draft_prefix, self.block_size)
        if draft_prefix is not None:
            raise ValueError("This training wrapper does not accept a conditioned draft prefix")
        return embeddings

    def _block_mask_options(self) -> dict:
        return {}

    def _block_mask_options_for_layer(self, layer_id: int) -> dict:
        """Return the attention-mask options for one draft layer.

        DFlash historically used one mask for every layer.  Keeping this
        method separate from ``_block_mask_options`` lets model variants use
        per-layer attention schedules without changing the public training
        wrapper API.
        """
        options = dict(self._block_mask_options())
        config = getattr(self.draft_model, "config", None)
        if config is not None and getattr(config, "layer_types", None):
            if config.layer_types[layer_id] == "sliding_attention":
                options["context_window"] = config.sliding_window
        return options

    def _compute_logits(
        self,
        draft_hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(self.draft_model, "lm_head"):
            return self.draft_model.lm_head(draft_hidden)
        return F.linear(draft_hidden, lm_head_weight)

    def _extra_training_loss(
        self,
        draft_hidden: torch.Tensor,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        objective_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        return logits.new_zeros(()), {}

    def _align_target_hidden_states(
        self,
        last_hidden_states: torch.Tensor,
        target_positions: torch.Tensor,
        *,
        global_sequence_length: int,
        packed_context_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Gather teacher rows from either a full sequence or a CP-owned shard."""

        bsz = target_positions.shape[0]
        hidden_size = last_hidden_states.size(-1)
        if not self.direct_cp_backend:
            gather_idx = target_positions.reshape(bsz, -1, 1).expand(
                -1, -1, hidden_size
            )
            return torch.gather(last_hidden_states, 1, gather_idx)

        from aspec.models.ops.direct_cp import (
            build_context_positions,
            gather_owner_sharded_rows,
        )

        cp_rank = (
            dist.get_rank(self.direct_cp_group)
            if self.direct_cp_group is not None
            else 0
        )
        owned_positions = (
            packed_context_positions[0].to(last_hidden_states.device)
            if packed_context_positions is not None
            else build_context_positions(
                    global_sequence_length,
                    cp_rank,
                    self.draft_context_parallel_size,
                    layout="contiguous",
                    alignment=self.context_shard_alignment,
            ).to(last_hidden_states.device)
        )
        if last_hidden_states.shape[1] == global_sequence_length and (
            packed_context_positions is None or self.draft_context_parallel_size == 1
        ):
            gather_idx = target_positions.reshape(bsz, -1, 1).expand(
                -1, -1, hidden_size
            )
            return torch.gather(last_hidden_states, 1, gather_idx)
        if last_hidden_states.shape[1] != owned_positions.numel():
            raise ValueError(
                "Direct CP last-hidden-state shard has an unexpected length: "
                f"got {last_hidden_states.shape[1]}, expected {owned_positions.numel()}"
            )
        return gather_owner_sharded_rows(
            last_hidden_states,
            owned_positions.unsqueeze(0).expand(bsz, -1),
            target_positions.flatten(1),
            global_sequence_length=global_sequence_length,
            group=self.direct_cp_group,
        )

    def _select_backbone_blocks(self, keep_mask, owners, rank):
        from aspec.models.ops.output_blocks import owner_block_indices

        return owner_block_indices(keep_mask, owners)

    def _draft_backbone(
        self,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor] | torch.Tensor,
        loss_mask: torch.Tensor,
        ctx_doc_ids: Optional[torch.Tensor] = None,
        base_position_ids: Optional[torch.Tensor] = None,
        packed_context_positions: Optional[torch.Tensor] = None,
        injected_anchors: Optional[torch.Tensor] = None,
        injected_keep_mask: Optional[torch.Tensor] = None,
        anchor_seed: Optional[int] = None,
        router_logits_out: Optional[list] = None,
        csa_losses_out: Optional[list] = None,
        qsa_losses_out: Optional[list] = None,
        dsa_losses_out: Optional[list] = None,
        packed_document_lengths: Optional[tuple[int, ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        draft_prefix=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Shared DFlash backbone (context features → anchor sampling → noise
        embedding → position ids → block-causal mask → draft model forward).

        Both ``DFlashModel.forward`` and ``DSparkModel.forward`` build the draft
        hidden states this exact way; only the label/loss tail differs. Keeping
        the attention/mask/anchor wiring here gives it a single source of truth.

        Returns:
            draft_hidden: [B, n_blocks*block_size, D] pre-loss draft hidden states
            anchor_positions: [B, n_blocks] sampled anchor positions
            block_keep_mask: [B, n_blocks] bool validity of each anchor slot
            n_blocks: number of anchor slots (== num_anchors)
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # 1. Sample anchor positions with validity mask
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device, ctx_doc_ids=ctx_doc_ids,
            injected_anchors=injected_anchors, injected_keep_mask=injected_keep_mask,
            anchor_seed=anchor_seed,
            packed_document_lengths=packed_document_lengths,
            attention_mask=attention_mask,
        )
        direct_cp_metadata = None
        # Full/SWA dense blocks are independent. Routing/indexer objectives can
        # depend on the complete query population, so retain their existing path.
        attention_policy = resolve_draft_attention(self.draft_model.config.to_dict())
        attention_policy["head_layout"] = resolve_draft_head_layout(
            self.draft_model.config.to_dict()
        )
        compact_queries = (
            getattr(self, "_compact_backbone_blocks", False)
            and getattr(self.draft_model.config, "ffn_type", "dense") == "dense"
            and set(attention_policy["layer_types"]) <= {"full_attention", "sliding_attention"}
        )
        original_anchors, original_keep = anchor_positions, block_keep_mask
        query_selection = None
        cp_size, cp_rank = 1, 0
        use_direct_cp = self.direct_cp_backend and (
            self.draft_context_parallel_size > 1 or ctx_doc_ids is not None
            or is_fa4_backend(self.attention_backend)
        )
        if use_direct_cp:
            from aspec.models.ops.direct_cp import (
                DirectCPMetadata,
                build_context_positions,
            )

            cp_size = self.draft_context_parallel_size
            cp_rank = (
                dist.get_rank(self.direct_cp_group)
                if self.direct_cp_group is not None
                else 0
            )
            if self.direct_cp_group is not None:
                group_root = dist.get_global_rank(self.direct_cp_group, 0)
                dist.broadcast(
                    anchor_positions, src=group_root, group=self.direct_cp_group
                )
                dist.broadcast(
                    block_keep_mask, src=group_root, group=self.direct_cp_group
                )

            if compact_queries:
                from aspec.models.ops.output_blocks import select_blocks

                query_selection = self._select_backbone_blocks(block_keep_mask, cp_size, cp_rank)
                if query_selection is not None:
                    anchor_positions = select_blocks(anchor_positions, query_selection)
                    block_keep_mask = select_blocks(block_keep_mask, query_selection)

            global_anchor_positions = anchor_positions
            global_block_keep_mask = block_keep_mask
            local_block_ids = build_context_positions(
                global_anchor_positions.shape[1],
                cp_rank,
                cp_size,
                layout="contiguous",
            ).to(device)
            anchor_positions = global_anchor_positions.index_select(1, local_block_ids)
            block_keep_mask = global_block_keep_mask.index_select(1, local_block_ids)
            context_positions = (
                packed_context_positions[0].to(device=device, dtype=torch.long)
                if packed_context_positions is not None
                else build_context_positions(
                    seq_len,
                    cp_rank,
                    cp_size,
                    layout="contiguous",
                    alignment=self.context_shard_alignment,
                ).to(device)
            )
            if packed_context_positions is not None and (
                packed_context_positions.shape != (bsz, context_positions.numel())
                or not torch.equal(packed_context_positions.to(device), context_positions[None].expand(bsz, -1))
                or (context_positions < 0).any() or (context_positions >= seq_len).any()
                or context_positions.unique().numel() != context_positions.numel()
            ):
                raise ValueError("Invalid packed context ownership positions")
            direct_cp_metadata = DirectCPMetadata(
                anchor_positions=global_anchor_positions,
                block_keep_mask=global_block_keep_mask,
                context_position_ids=context_positions.unsqueeze(0).expand(bsz, -1),
                block_size=self.block_size,
                cp_rank=cp_rank,
                cp_size=cp_size,
                group=self.direct_cp_group,
                sequence_length=seq_len,
                context_doc_ids=(
                    ctx_doc_ids.index_select(1, context_positions)
                    if ctx_doc_ids is not None else None
                ),
                anchor_doc_ids=(
                    torch.gather(ctx_doc_ids, 1, global_anchor_positions)
                    if ctx_doc_ids is not None else None
                ),
                context_rope_position_ids=(
                    base_position_ids.index_select(1, context_positions)
                    if base_position_ids is not None else None
                ),
                backend=self.attention_backend,
                context_window=self._block_mask_options_for_layer(0).get("context_window"),
                layout="contiguous",
            )
        elif compact_queries and is_cudnn_flex_backend(self.attention_backend):
            # Native BSHD has the same independent Dense query blocks as THD.
            # Reuse owner-local selection with one owner, then restore original
            # slots before the head. One preparation transfer serves all layers
            # and original head chunks; sampling and masks remain unchanged.
            from aspec.models.ops.output_blocks import select_blocks

            query_selection = self._select_backbone_blocks(block_keep_mask, 1, 0)
            if query_selection is not None:
                anchor_positions = select_blocks(anchor_positions, query_selection)
                block_keep_mask = select_blocks(block_keep_mask, query_selection)
        # Resolve shape/ownership decisions before queuing the large context
        # projection, so their host transfers do not fence its device work.
        context_feature = self.draft_model.extract_context_feature(hidden_states_list)
        if use_direct_cp:
            if context_feature.shape[1] == seq_len:
                context_feature = context_feature.index_select(1, context_positions)
            elif context_feature.shape[1] != context_positions.numel():
                raise ValueError(
                    "Direct CP context feature has an unexpected local length: "
                    f"got {context_feature.shape[1]}, expected {context_positions.numel()}"
                )
        # Other attention paths keep the original layout; this is an execution
        # optimization, never a new capability requirement or sampling policy.
        n_blocks = anchor_positions.shape[1]
        self.backbone_original_blocks_observed = original_anchors.shape[1] // (
            self.draft_context_parallel_size if use_direct_cp else 1
        )
        self.backbone_blocks_observed = n_blocks

        # 3. Create noise embeddings (anchor token + MASK tokens)
        noise_embedding = self._create_noise_embed(input_ids, anchor_positions, block_keep_mask)
        noise_embedding = self._condition_noise_embed(
            noise_embedding, input_ids, anchor_positions, block_keep_mask, draft_prefix
        )

        # 4. Create position IDs
        context_position_ids, draft_position_ids = self._create_position_ids(
            anchor_positions, seq_len, base_position_ids=base_position_ids
        )
        if direct_cp_metadata is not None:
            context_position_ids = (
                direct_cp_metadata.context_rope_position_ids
                if direct_cp_metadata.context_rope_position_ids is not None
                else direct_cp_metadata.context_position_ids
            ).contiguous()

        # 5. Create block-causal attention mask
        draft_len = n_blocks * self.block_size
        kv_len = seq_len + draft_len
        layer_types = attention_policy["layer_types"]
        has_csa_layers = "compressed_sparse_attention" in layer_types
        has_sparse_qsa_layers = (
            attention_policy["qsa"] is not None
            and attention_policy["qsa"]["training_stage"] == "sparse"
        )
        has_sparse_dsa_layers = (
            attention_policy.get("dsa") is not None
            and attention_policy["dsa"]["training_stage"] == "sparse"
        )
        has_dynamic_sparse_layers = has_csa_layers or has_sparse_qsa_layers or has_sparse_dsa_layers

        block_mask = None
        if device.type == "cuda" and direct_cp_metadata is None:
            # A mixed full/SWA schedule needs a distinct BlockMask for each
            # attention policy.  Cache equal policies so the common uniform
            # schedule still creates exactly one mask.
            mask_cache = {}
            block_masks = []
            for layer_id in range(self.draft_model.num_layers):
                # CSA has a different KV layout (raw KV followed by arbitrary
                # compressed entries) and must always construct its exact local
                # plus compressed-entry mask after selection.  A generic mask is
                # both wasted work and incorrect when there are zero groups.
                if layer_types[layer_id] == "compressed_sparse_attention" or (
                    layer_types[layer_id] == "qwen_sparse_attention"
                    and has_sparse_qsa_layers
                ) or (
                    layer_types[layer_id] == "deepseek_sparse_attention"
                    and has_sparse_dsa_layers
                ):
                    block_masks.append(None)
                    continue
                options = self._block_mask_options_for_layer(layer_id)
                cache_key = tuple(sorted(options.items()))
                if cache_key not in mask_cache:
                    if is_cudnn_flex_backend(self.attention_backend):
                        from aspec.models.ops.cudnn_flex_attention import (
                            build_dspark_mla_mask,
                        )

                        mask_cache[cache_key] = build_dspark_mla_mask(
                            anchor_positions=anchor_positions,
                            block_keep_mask=block_keep_mask,
                            context_length=seq_len,
                            block_size=self.block_size,
                            ctx_doc_ids=ctx_doc_ids,
                            **options,
                        )
                    else:
                        mask_mod = _create_dflash_mask_mod(
                            anchor_positions=anchor_positions,
                            block_keep_mask=block_keep_mask,
                            ctx_len=seq_len,
                            block_size=self.block_size,
                            ctx_doc_ids=ctx_doc_ids,
                            **options,
                        )
                        mask_cache[cache_key] = compile_friendly_create_block_mask(
                            mask_mod=mask_mod,
                            B=bsz,
                            H=None,
                            Q_LEN=draft_len,
                            KV_LEN=kv_len,
                            device=device,
                        )
                block_masks.append(mask_cache[cache_key])
            block_mask = (
                block_masks[0]
                if not has_dynamic_sparse_layers and len(mask_cache) == 1
                else block_masks
            )

        if direct_cp_metadata is not None:
            from dataclasses import replace

            metadata_cache = {}
            per_layer_metadata = []
            for layer_id in range(self.draft_model.num_layers):
                options = self._block_mask_options_for_layer(layer_id)
                if set(options) - {"context_window", "sliding_window", "is_causal"}:
                    raise ValueError("Direct CP does not implement these Draft mask options")
                policy = (options.get("context_window"), options.get("sliding_window"),
                          options.get("is_causal", False))
                if policy not in metadata_cache:
                    original = (direct_cp_metadata.context_window, direct_cp_metadata.query_window,
                                direct_cp_metadata.draft_is_causal)
                    metadata_cache[policy] = (
                        direct_cp_metadata if policy == original else replace(
                            direct_cp_metadata, context_window=policy[0], query_window=policy[1],
                            draft_is_causal=policy[2], attention_observation={},
                        )
                    )
                per_layer_metadata.append(metadata_cache[policy])
            direct_cp_metadata = (per_layer_metadata[0] if len(metadata_cache) == 1
                                  else per_layer_metadata)
        elif device.type != "cuda" and any(
            self._block_mask_options_for_layer(i).get("context_window") is not None
            for i in range(self.draft_model.num_layers)
        ):
            raise ValueError("SWA training requires a CUDA attention backend; CPU cannot omit the window")

        # Only populated on the actual forward path, not by reading requested config.
        self.attention_windows_observed = [
            self._block_mask_options_for_layer(i).get("context_window")
            for i in range(self.draft_model.num_layers)
        ]
        sparse_direct_metadata = (
            direct_cp_metadata[0]
            if isinstance(direct_cp_metadata, (list, tuple))
            else direct_cp_metadata
        )

        csa_metadata = None
        if attention_policy["csa"] is not None:
            from aspec.models.draft.csa import CSAForwardMetadata

            csa_metadata = CSAForwardMetadata(
                anchor_positions=(
                    sparse_direct_metadata.anchor_positions
                    if sparse_direct_metadata is not None else anchor_positions
                ),
                block_keep_mask=(
                    sparse_direct_metadata.block_keep_mask
                    if sparse_direct_metadata is not None else block_keep_mask
                ),
                block_size=self.block_size,
                context_window=attention_policy["sliding_window"],
                context_doc_ids=(
                    sparse_direct_metadata.context_doc_ids
                    if sparse_direct_metadata is not None else ctx_doc_ids
                ),
                anchor_doc_ids=(
                    sparse_direct_metadata.anchor_doc_ids
                    if sparse_direct_metadata is not None else (
                        torch.gather(ctx_doc_ids, 1, anchor_positions)
                        if ctx_doc_ids is not None else None
                    )
                ),
                context_physical_position_ids=(
                    sparse_direct_metadata.context_position_ids
                    if sparse_direct_metadata is not None else None
                ),
            )
        qsa_metadata = None
        dsa_metadata = None
        if attention_policy["qsa"] is not None or attention_policy.get("dsa") is not None:
            from aspec.models.draft.qsa import QSAForwardMetadata

            qsa_metadata = QSAForwardMetadata(
                anchor_positions=(
                    sparse_direct_metadata.anchor_positions
                    if sparse_direct_metadata is not None else anchor_positions
                ),
                block_keep_mask=(
                    sparse_direct_metadata.block_keep_mask
                    if sparse_direct_metadata is not None else block_keep_mask
                ),
                block_size=self.block_size,
                context_doc_ids=(
                    sparse_direct_metadata.context_doc_ids
                    if sparse_direct_metadata is not None else ctx_doc_ids
                ),
                anchor_doc_ids=(
                    sparse_direct_metadata.anchor_doc_ids
                    if sparse_direct_metadata is not None else (
                        torch.gather(ctx_doc_ids, 1, anchor_positions)
                        if ctx_doc_ids is not None else None
                    )
                ),
                context_physical_position_ids=(
                    sparse_direct_metadata.context_position_ids
                    if sparse_direct_metadata is not None else None
                ),
                anchor_rope_position_ids=(
                    torch.gather(
                        base_position_ids, 1, sparse_direct_metadata.anchor_positions
                    )
                    if sparse_direct_metadata is not None
                    and base_position_ids is not None
                    else (
                        sparse_direct_metadata.anchor_positions
                        if sparse_direct_metadata is not None else None
                    )
                ),
            )

            if attention_policy.get("dsa") is not None:
                dsa_metadata, qsa_metadata = qsa_metadata, None

        # 6. Draft model forward — router/indexer activations live only in this call.
        moe = getattr(getattr(self.draft_model, "config", None), "ffn_type", "dense") != "dense"
        model_kwargs = {"return_router_logits": True} if moe else {}
        draft_hidden = self.draft_model(
            draft_input_ids=None,
            context_feature=context_feature,
            draft_position_ids=draft_position_ids,
            context_position_ids=context_position_ids,
            block_mask=block_mask,
            noise_embedding=noise_embedding,
            direct_cp_metadata=direct_cp_metadata,
            csa_metadata=csa_metadata,
            qsa_metadata=qsa_metadata,
            dsa_metadata=dsa_metadata,
            **model_kwargs,
        )
        if moe:
            draft_hidden, router_logits = draft_hidden
            if router_logits_out is not None:
                router_logits_out.extend(router_logits)
        if csa_losses_out is not None:
            csa_losses_out.extend(
                loss for loss in self.draft_model.last_csa_indexer_losses if loss is not None
            )
        if qsa_losses_out is not None:
            qsa_losses_out.extend(
                loss for loss in self.draft_model.last_qsa_indexer_losses if loss is not None
            )
        if dsa_losses_out is not None:
            dsa_losses_out.extend(
                loss for loss in self.draft_model.last_dsa_indexer_losses if loss is not None
            )
        self.attention_policy_observed = {
            "head_layout": attention_policy["head_layout"],
            "context_windows": list(attention_policy["context_windows"]),
            "window_semantics": attention_policy["window_semantics"],
            "draft_block_attention": attention_policy["draft_block_attention"],
            "csa": attention_policy["csa"],
            "qsa": attention_policy["qsa"],
            "dsa": attention_policy.get("dsa"),
            "layers": list(
                getattr(self.draft_model, "last_attention_observations", ())
            ),
        }

        if query_selection is not None:
            from aspec.models.ops.output_blocks import restore_blocks

            original_width = original_anchors.shape[1] // cp_size
            local_selection = query_selection.unflatten(1, (cp_size, -1))[:, cp_rank]
            local_selection = local_selection - cp_rank * original_width
            draft_hidden = restore_blocks(
                draft_hidden.unflatten(1, (n_blocks, self.block_size)),
                local_selection, original_width,
            ).flatten(1, 2)
            anchor_positions = original_anchors[:, cp_rank * original_width:(cp_rank + 1) * original_width]
            block_keep_mask = original_keep[:, cp_rank * original_width:(cp_rank + 1) * original_width]
            n_blocks = original_width
        return draft_hidden, anchor_positions, block_keep_mask, n_blocks

    def _moe_training_loss(self, router_logits, objective_weights, denominator):
        if not router_logits:
            return denominator.new_zeros(()), {}
        if self.draft_model.config.ffn_type == "ling_moe":
            from aspec.models.draft.ling_moe import ling_router_terms

            return ling_router_terms(router_logits, objective_weights, denominator,
                                     self.draft_model.config.router_z_loss_coef,
                                     cp_group=self.direct_cp_group)
        from aspec.models.draft.moe import router_loss_terms

        numerator, components = router_loss_terms(
            router_logits, objective_weights, denominator,
            self.draft_model.config.num_experts_per_tok,
            cp_group=self.direct_cp_group,
        )
        return self.draft_model.config.router_aux_loss_coef * numerator, components

    def _csa_training_loss(self, per_query_losses, objective_weights, denominator):
        if not per_query_losses:
            return denominator.new_zeros(()), {}
        flat_weights = objective_weights.reshape(objective_weights.shape[0], -1)
        numerator = sum(
            (loss.to(flat_weights.dtype) * flat_weights).sum()
            for loss in per_query_losses
        )
        components = {
            "csa_indexer_loss": (numerator.detach(), denominator.detach())
        }
        return self.draft_model.config.csa_indexer_loss_coef * numerator, components

    def _qsa_training_loss(self, per_query_losses, objective_weights, denominator):
        if not per_query_losses:
            return denominator.new_zeros(()), {}
        flat_weights = objective_weights.reshape(objective_weights.shape[0], -1)
        numerator = sum(
            (loss.to(flat_weights.dtype) * flat_weights).sum()
            for loss in per_query_losses
        )
        components = {
            "qsa_indexer_loss": (numerator.detach(), denominator.detach())
        }
        return self.draft_model.config.qsa_indexer_loss_coef * numerator, components

    def _dsa_training_loss(self, per_query_losses, objective_weights, denominator):
        if not per_query_losses:
            return objective_weights.new_zeros(()), {}
        flat_weights = objective_weights.flatten(1)
        numerator = sum((loss * flat_weights).sum() for loss in per_query_losses)
        return self.draft_model.config.dsa_indexer_loss_coef * numerator, {
            "dsa_indexer_loss": (numerator.detach(), denominator.detach())
        }

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
        draft_prefix=None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """
        Full DFlash training forward pass.

        Returns:
            loss: scalar training loss (objective-weighted)
            accuracy: scalar accuracy (binary mask, no decay)
            loss_per_position: [block_size] mean loss at each within-block position
                (slots before prediction_start are known inputs and always 0)
            acc_per_position: [block_size] mean accuracy at each within-block position
            count_per_position: [block_size] valid label count at each within-block
                position before loss decay is applied
            loss_components: dict of extra per-component ``(numerator, denominator)``
                pairs for logging, pooled by the trainer just like ``loss_terms``
                (empty for the base DFlash objective; populated by subclasses).
            loss_terms: additive objective numerator and denominator. The trainer
                pools these over the full accumulation window and data-parallel group.
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        router_logits = []
        csa_losses = []
        qsa_losses = []
        dsa_losses = []
        # 1-6. Shared backbone → draft hidden states + anchor bookkeeping.
        draft_prefix = self._prepare_draft_prefix(
            input_ids, last_hidden_states, attention_mask, draft_prefix,
            ctx_doc_ids, base_position_ids, packed_context_positions, packed_document_lengths,
        )
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
            draft_prefix=draft_prefix,
        )

        if getattr(self.draft_model.config, "dsa_training_stage", None) == "indexer_warmup":
            offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
            labels = anchor_positions.unsqueeze(-1) + offsets
            weights = torch.gather(
                loss_mask.unsqueeze(1).expand(-1, n_blocks, -1), 2,
                labels.clamp(max=seq_len - 1),
            ) * (labels < seq_len) * block_keep_mask.unsqueeze(-1) * (offsets > 0)
            if self.loss_objective == "decay" and self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
                weights = weights * torch.exp(-(offsets - 1).clamp_min(0).float() / self.loss_decay_gamma)
            denominator = weights.sum()
            numerator, components = self._dsa_training_loss(dsa_losses, weights, denominator)
            zero = draft_hidden.new_zeros((), dtype=torch.float32)
            positions = draft_hidden.new_zeros(self.block_size, dtype=torch.float32)
            return (numerator / denominator.clamp_min(1e-6), zero, positions,
                    positions.clone(), positions.clone(), components,
                    (numerator, denominator.detach()))

        # 7. Compute logits via frozen LM head
        logits = self._compute_logits(draft_hidden, lm_head_weight)

        # 8. Compute labels and weight mask (SpecForge pattern)
        # Labels: same-position prediction (position k predicts token at anchor+k)
        label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets  # [B, n_blocks, block_size]
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n_blocks, -1),
            2,
            safe_label_indices,
        )  # [B, n_blocks, block_size]

        # Weight mask: block validity × bounds × exclude anchor (pos 0) × loss_mask
        weight_mask = block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        weight_mask = weight_mask * valid_label_mask.float()

        pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block >= self.prediction_start).float()

        # Gather original loss_mask at label positions
        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, n_blocks, -1),
            2,
            safe_label_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered

        # Capture binary mask BEFORE applying objective weights. Accuracy measures
        # "did we predict correctly?" uniformly across positions, while weighting
        # only shapes gradient contribution. SpecForge uses no decay at all;
        # our objective weighting is an addition to the training signal, not the metric.
        binary_eval_mask = weight_mask.view(-1)

        # 9. Per-token loss: ce_loss_alpha*CE + l1_loss_alpha*L1.
        vocab_size = logits.size(-1)
        flat_logits = logits.view(-1, vocab_size)
        flat_targets = target_ids.view(-1)
        ce_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")

        loss_per_token = self.ce_loss_alpha * ce_per_token
        if self.uses_target_distribution:
            if last_hidden_states is None:
                raise ValueError(
                    "DFlash L1 distillation (l1_loss_alpha > 0) requires target "
                    "last_hidden_states; set inference.store_last_hidden_states=true in the "
                    "run config."
                )
            tgt_idx = (safe_label_indices - 1).clamp(min=0)
            aligned_hidden = self._align_target_hidden_states(
                last_hidden_states,
                tgt_idx,
                global_sequence_length=seq_len,
                packed_context_positions=packed_context_positions,
            )
            target_logits = F.linear(aligned_hidden, lm_head_weight).view(-1, vocab_size)
            target_probs = torch.softmax(target_logits.float(), dim=-1)
            draft_probs = torch.softmax(flat_logits.float(), dim=-1)
            l1_per_token = (draft_probs - target_probs).abs().sum(dim=-1)
            loss_per_token = loss_per_token + self.l1_loss_alpha * l1_per_token

        loss_per_token_by_position = ce_per_token.view(bsz, n_blocks, self.block_size)

        objective_weights = weight_mask
        if (
            self.loss_objective == "decay"
            and self.loss_decay_gamma is not None
            and self.loss_decay_gamma > 0
        ):
            # Loss decay: exp(-(k-1)/γ) so k=1 (1st prediction) gets weight 1.0
            k = torch.arange(self.block_size, device=device).view(1, 1, -1)
            decay_weights = torch.exp(
                -(k - self.prediction_start).clamp(min=0).float() / self.loss_decay_gamma
            )
            objective_weights = weight_mask * decay_weights
        elif self.loss_objective == "dpace":
            dpace_weights = torch.ones_like(weight_mask)
            if self.block_size > self.prediction_start:
                with torch.no_grad():
                    target_confidences = torch.exp(
                        -loss_per_token_by_position[..., self.prediction_start:].float()
                    )
                    dpace_pred_weights = _dpace_position_weights(
                        target_confidences,
                        self.dpace_alpha,
                    ).to(dtype=weight_mask.dtype)
                dpace_weights[..., self.prediction_start:] = dpace_pred_weights
            objective_weights = weight_mask * dpace_weights

        flat_weights = objective_weights.view(-1)
        loss_numerator = (loss_per_token * flat_weights).sum()
        loss_denominator = flat_weights.sum()
        extra_numerator, loss_components = self._extra_training_loss(
            draft_hidden=draft_hidden,
            logits=logits,
            target_ids=target_ids,
            objective_weights=objective_weights,
        )
        moe_numerator, moe_components = self._moe_training_loss(
            router_logits, objective_weights, loss_denominator,
        )
        csa_numerator, csa_components = self._csa_training_loss(
            csa_losses, objective_weights, loss_denominator,
        )
        qsa_numerator, qsa_components = self._qsa_training_loss(
            qsa_losses, objective_weights, loss_denominator,
        )
        dsa_numerator, dsa_components = self._dsa_training_loss(
            dsa_losses, objective_weights, loss_denominator,
        )
        loss_components.update(moe_components)
        loss_components.update(csa_components)
        loss_components.update(qsa_components)
        loss_components.update(dsa_components)
        loss_numerator = (
            loss_numerator
            + extra_numerator
            + moe_numerator
            + csa_numerator
            + qsa_numerator
            + dsa_numerator
        )
        loss = loss_numerator / loss_denominator.clamp(min=1e-6)

        # 10. Accuracy (using binary mask without decay)
        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            actual_token_count = binary_eval_mask.sum().clamp(min=1e-6)
            accuracy = correct.sum().float() / actual_token_count

            # Per-position-within-block metrics (index 0 = anchor, masked out;
            # indices 1..block_size-1 correspond to 1..B-1 tokens past the anchor).
            # Matches Eagle3's per-TTT-position breakdown semantically.
            binary_weights = binary_eval_mask.view(bsz, n_blocks, self.block_size)
            count_per_position = binary_weights.sum(dim=(0, 1))
            count_per_pos = count_per_position.clamp(min=1.0)

            loss_per_position = (loss_per_token_by_position * binary_weights).sum(
                dim=(0, 1)
            ) / count_per_pos
            acc_per_position = (correct.view(bsz, n_blocks, self.block_size).float()).sum(
                dim=(0, 1)
            ) / count_per_pos

        return (
            loss,
            accuracy,
            loss_per_position,
            acc_per_position,
            count_per_position,
            loss_components,
            (loss_numerator, loss_denominator.detach()),
        )
