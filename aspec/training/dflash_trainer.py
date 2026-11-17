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

"""DFlash trainer — extends Trainer with DFlash-specific model init, forward, and metrics."""

from argparse import Namespace
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from aspec.models.dflash import DFlashModel
from aspec.models.draft.dflash import DFlashConfig, DFlashDraftModel
from aspec.models.ops.anchors import anchor_sampling_seed
from aspec.training import checkpoint
from aspec.training.fsdp import apply_fsdp2, fsdp2_load_full_state_dict
from aspec.training.optimizer import BF16Optimizer
from aspec.training.trainer import Trainer
from aspec.utils.distributed import get_gloo_group
from aspec.utils.logging import logger


class DFlashTrainer(Trainer):
    """DFlash-specific trainer.

    Extends ``Trainer`` with DFlash model initialisation (dual-source KV draft model),
    forward/backward with anchor sampling + block-causal mask, and metric aggregation.

    DSparkTrainer is a subclass that overrides the build hooks:
    - `_draft_config_class`
    - `_build_draft_model`
    - `_build_training_wrapper`
    """

    _draft_config_class = DFlashConfig
    _anchor_slot_offset = 1
    _extra_loss_component_keys: list[str] = []

    def _build_draft_model(self, config):
        """Instantiate the draft network. Overridden by subclasses."""
        return DFlashDraftModel(config)

    def _build_training_wrapper(self, draft_model):
        """Wrap the draft network with the training-objective module."""
        return DFlashModel(
            draft_model=draft_model,
            block_size=self.block_size,
            num_anchors=self.num_anchors,
            loss_objective=self.loss_objective,
            dpace_alpha=self.dpace_alpha,
            loss_decay_gamma=self.loss_decay_gamma,
            ce_loss_alpha=self.ce_loss_alpha,
            l1_loss_alpha=self.l1_loss_alpha,
            deterministic_anchor_sampling=getattr(
                self.args, "deterministic_anchor_sampling", False
            ),
            anchor_seed=getattr(self.args, "seed", 0),
            anchor_sampling_scope=getattr(self.args, "anchor_sampling_scope", "row"),
            attention_backend=getattr(self.args, "attention_backend", "flex_attention"),
            draft_context_parallel_size=getattr(
                self.args, "draft_context_parallel_size", 1
            ),
        )

    def __init__(self, args: Namespace):
        super().__init__(args)
        self.target_lm_head: Optional[torch.nn.Module] = None
        self.num_target_layers = getattr(args, "dflash_num_target_layers", 5)
        self.block_size = getattr(args, "dflash_block_size", 16)
        self.num_anchors = getattr(args, "dflash_num_anchors", 512)
        self.loss_objective = getattr(args, "dflash_loss_objective", "decay")
        self.dpace_alpha = getattr(args, "dflash_dpace_alpha", 0.5)
        self.loss_decay_gamma = getattr(args, "dflash_loss_decay_gamma", 7.0)
        self.ce_loss_alpha = getattr(args, "dflash_ce_loss_alpha", 1.0)
        self.l1_loss_alpha = getattr(args, "dflash_l1_loss_alpha", 0.0)
        self._last_hs_prenorm = getattr(args, "last_hidden_states_prenorm", False)
        self.verifier_norm: Optional[torch.nn.Module] = None
        self._mtp_binding_validated = False

    def init_model(
        self,
        draft_model_config,
        target_model_path: str,
        mooncake_config=None,
    ) -> int:
        if mooncake_config is not None:
            from aspec.transfer.mooncake.utils import (
                check_mooncake_master_available,
            )

            check_mooncake_master_available(
                mooncake_config.master_server_address, mooncake_config.metadata_server
            )

        init_context = self._get_init_weight_context_manager()

        with init_context():
            cfg_cls = self._draft_config_class

            if isinstance(draft_model_config, str):
                config = cfg_cls.from_pretrained(draft_model_config)
            elif isinstance(draft_model_config, dict):
                config = cfg_cls(**draft_model_config)
            elif isinstance(draft_model_config, DFlashConfig):
                config = draft_model_config
            else:
                raise TypeError(
                    f"Unsupported draft_model_config type: {type(draft_model_config).__name__}. "
                    f"Expected str, dict, or {cfg_cls.__name__}."
                )

            if not hasattr(config, "num_target_layers") or config.num_target_layers is None:
                config.num_target_layers = self.num_target_layers
            if not hasattr(config, "target_hidden_size") or config.target_hidden_size is None:
                config.target_hidden_size = config.hidden_size
            if (
                not hasattr(config, "target_num_hidden_layers")
                or config.target_num_hidden_layers is None
            ):
                from transformers import AutoConfig

                target_config = AutoConfig.from_pretrained(
                    target_model_path,
                    trust_remote_code=getattr(self.args, "trust_remote_code", False),
                )
                config.target_num_hidden_layers = target_config.num_hidden_layers

            draft_model = self._build_draft_model(config)

        conditioning = getattr(config, "mtp_conditioning", None)
        if dist.get_rank() == 0:
            if conditioning:
                draft_model.initialize_mtp_conditioning(target_model_path)
            initial_path = getattr(self.args, "initial_draft_model_path", None)
            if initial_path:
                checkpoint.load_initial_draft_weights(
                    draft_model, initial_path, allow_target_embedding=True
                )
            draft_model.load_embedding(
                target_model_path,
                embedding_key=getattr(self.args, "embedding_key", "model.embed_tokens.weight"),
            )

        draft_model.freeze_embedding()
        if getattr(self.args, "gradient_checkpointing", False):
            draft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        draft_model = draft_model.to(torch.bfloat16)
        draft_model.keep_csa_projection_precision()

        dist.barrier(group=get_gloo_group())

        frozen_count = sum(p.numel() for p in draft_model.parameters() if not p.requires_grad)
        trainable_count = sum(p.numel() for p in draft_model.parameters() if p.requires_grad)
        logger.info(
            f"[Rank {self.dp_rank}] DFlash draft model: {trainable_count:,} trainable, "
            f"{frozen_count:,} frozen (embedding) parameters"
        )

        dflash_model = self._build_training_wrapper(draft_model)

        full_state = dflash_model.state_dict() if dist.get_rank() == 0 else {}

        dflash_model = apply_fsdp2(
            dflash_model,
            mesh=self.dp_mesh,
            cpu_offload=self.fsdp_cpu_offload,
            args=self.args,
            modules_to_shard=list(draft_model.layers),
        )

        dflash_model = fsdp2_load_full_state_dict(
            dflash_model,
            full_state,
            self.dp_mesh,
            cpu_offload=True if self.fsdp_cpu_offload else None,
        )

        if getattr(self.args, "compile_model", False):
            logger.info("Compiling DFlash model with torch.compile (inductor backend)")
            dflash_model = torch.compile(dflash_model)

        self.model = dflash_model
        # Unwrap torch.compile and/or DDP module wrappers to access underlying DFlashModel
        _unwrapped = getattr(self.model, "_orig_mod", self.model)  # torch.compile
        self.dflash = getattr(_unwrapped, "module", _unwrapped)  # DDP/replicate
        self.draft_model = self.dflash.draft_model
        if conditioning:
            self._anchor_slot_offset = self.dflash.prediction_start
            if hasattr(self, "confidence_head_alpha"):
                self._extra_loss_component_keys = [
                    "ce_loss", "l1_loss", "confidence_loss",
                    *(f"suffix_accept_rate@{index}" for index in range(
                        self.block_size - self.dflash.prediction_start)),
                    "suffix_tau_probabilistic",
                ]

        total_steps = self.args.lr_total_steps
        decay_style = getattr(self.args, "lr_decay_style", "cosine")
        warmup_ratio = getattr(self.args, "warmup_ratio", 0.1)

        self.optimizer = BF16Optimizer(
            self.draft_model,
            lr=self.args.learning_rate,
            weight_decay=getattr(self.args, "weight_decay", 0.0),
            max_grad_norm=self.args.max_grad_norm,
            warmup_ratio=warmup_ratio,
            total_steps=total_steps,
            decay_style=decay_style if decay_style != "WSD" else "cosine",
            min_lr=getattr(self.args, "min_lr", 0.0),
        )

        if decay_style == "WSD" and total_steps:
            from aspec.training.lr_scheduler import LRSchedulerWithWarmup

            wsd_ratio = getattr(self.args, "wsd_decay_ratio", None)
            if wsd_ratio is None:
                wsd_ratio = getattr(self.args, "lr_wsd_decay_ratio", 0.2)
            self.optimizer.scheduler = LRSchedulerWithWarmup(
                self.optimizer.optimizer,
                max_lr=self.args.learning_rate,
                total_steps=total_steps,
                warmup_steps=int(warmup_ratio * total_steps),
                decay_style="WSD",
                min_lr=getattr(self.args, "min_lr", 0.0),
                wsd_decay_steps=max(1, int(wsd_ratio * total_steps)),
                wsd_decay_style=(getattr(self.args, "wsd_decay_style", None)
                                 or getattr(self.args, "lr_wsd_decay_style", "cosine")),
            )

        self.lr_scheduler = self.optimizer.lr_scheduler

        checkpoint_payload = checkpoint.load(self)
        if conditioning:
            self.draft_model.validate_mtp_binding(
                target_model_path,
                embedding_key=getattr(self.args, "embedding_key", "model.embed_tokens.weight"),
            )

        self._init_target_lm_head(target_model_path)

        dsa_warmup = (
            getattr(getattr(self.draft_model, "config", None), "dsa_training_stage", None)
            == "indexer_warmup"
        )
        if self.target_lm_head is None and not dsa_warmup:
            raise ValueError(
                "target_lm_head is required but was None. Ensure _init_target_lm_head succeeded."
            )
        self.target_lm_head_weight = (
            None if self.target_lm_head is None else self.target_lm_head.lm_head.weight
        )
        self._mtp_binding_validated = bool(conditioning)

        # Frozen Target helpers can initialize temporary random parameters.
        # Restore the training RNG only after all model initialization ends.
        checkpoint.finalize_load(self, checkpoint_payload)

        self.prof.on_init_end()

        logger.info(f"[Rank {self.dp_rank}] DFlash model initialized with FSDP2")

        return 0

    # ------------------------------------------------------------------
    # Target LM head (same as Eagle3Trainer)
    # ------------------------------------------------------------------

    def _init_target_lm_head(self, target_model_path: str) -> None:
        config = getattr(getattr(self, "draft_model", None), "config", None)
        if getattr(config, "dsa_training_stage", None) == "indexer_warmup":
            # Attention KL does not consume a vocabulary head or verifier norm.
            self.target_lm_head = self.target_lm_head_weight = self.verifier_norm = None
            return
        from aspec.models.target.target_utils import load_synced_target_lm_head

        # The norm exists to make the target's pre-norm hidden states usable, so a
        # run whose objective never reads them must not fail on loading it.
        load_norm = self._last_hs_prenorm and self.dflash.uses_target_hidden_states
        if self._last_hs_prenorm and not load_norm:
            logger.info(
                "[Rank %s] last_hidden_states_prenorm is set, but the objective does not read "
                "the target's last hidden states; skipping the verifier norm",
                self.dp_rank,
            )

        self.target_lm_head = load_synced_target_lm_head(
            target_model_path,
            load_norm=load_norm,
            lm_head_key=getattr(self.args, "lm_head_key", "lm_head.weight"),
            norm_key=getattr(self.args, "norm_key", "model.norm.weight"),
            trust_remote_code=getattr(self.args, "trust_remote_code", False),
        )
        self.verifier_norm = self.target_lm_head.norm

    def get_feature_contract_status(self) -> dict:
        status = super().get_feature_contract_status()
        from aspec.models.draft.ling_moe import LingDraftMoE
        from aspec.models.draft.moe import QwenDraftMoE

        draft = getattr(self, "draft_model", None)
        if getattr(getattr(draft, "config", None), "mtp_conditioning", None):
            import inspect
            import sys

            from aspec.config.mtp_conditioning import mtp_conditioning_semantics

            status = dict(status)
            status["mtp_conditioning"] = {
                **mtp_conditioning_semantics(draft.config.to_dict()),
                "prediction_start": self.dflash.prediction_start,
                "frozen_source_parameters": (sum(p.numel() for p in draft.mtp_source.parameters())
                                             if draft.mtp_source is not None else 0),
                "fusion_parameters": (sum(p.numel() for p in draft.mtp_fusion.parameters())
                                      if draft.mtp_fusion is not None else 0),
                "target_binding_validated": self._mtp_binding_validated,
                "runtime_origin": {
                    "python": sys.executable,
                    "trainer": inspect.getfile(type(self)),
                    "draft": inspect.getfile(type(draft)),
                    "native_features": (inspect.getfile(type(draft.mtp_source))
                                        if draft.mtp_source is not None else None),
                },
            }
        layers = list(getattr(draft, "layers", []))
        if layers and any(isinstance(layer.mlp, (QwenDraftMoE, LingDraftMoE)) for layer in layers):
            status = dict(status)
            status["draft_structure"] = {
                "ffn_modules": [type(layer.mlp).__name__ for layer in layers],
                "expert_counts": [len(getattr(layer.mlp, "experts", [])) for layer in layers],
                "top_k": [getattr(layer.mlp, "top_k", None) for layer in layers],
                "shared_expert_parameters": [sum(p.numel() for p in layer.mlp.shared_experts.parameters())
                                             if getattr(layer.mlp, "shared_experts", None) is not None else 0
                                             for layer in layers],
                "trainable_parameters": sum(p.numel() for p in draft.parameters() if p.requires_grad),
                "frozen_parameters": sum(p.numel() for p in draft.parameters() if not p.requires_grad),
            }
        observed = getattr(getattr(self, "dflash", None), "attention_policy_observed", None)
        if observed is not None:
            status = dict(status)
            execution = dict(status.get("execution_metadata") or {})
            execution.update(
                {
                    "draft_attention_backend": [
                        getattr(self.args, "attention_backend", "flex_attention")
                    ],
                    "draft_context_parallel_size": [
                        int(getattr(self.args, "draft_context_parallel_size", 1))
                    ],
                }
            )
            status["execution_metadata"] = execution
            status["draft_attention"] = {**observed, "forward_observed": True}
            if self._indexer_warmup_contract() is not None:
                main_frozen = not any(
                    parameter.requires_grad for name, parameter in self.draft_model.named_parameters()
                    if not any(f".{kind}_layer.indexer." in name for kind in ("csa", "qsa", "dsa"))
                )
                execution.update(draft_body_frozen=main_frozen,
                                 draft_attention_backward_required=not main_frozen)
                if main_frozen:
                    execution["draft_attention_backward"] = ["not_required_frozen_backbone"]
        return status

    def save_draft_model_for_serving(self, output_dir: str) -> None:
        if getattr(self.draft_model.config, "mtp_conditioning", None):
            raise NotImplementedError("MTP-conditioned Draft requires native --format aspec export")
        if getattr(self.draft_model.config, "ffn_type", "dense") != "dense":
            raise NotImplementedError(
                "MoE Draft serving export is not yet supported; retain the complete training checkpoint"
            )
        from aspec.config.draft_attention import resolve_draft_attention

        attention = resolve_draft_attention(self.draft_model.config.to_dict())
        if attention["csa"] is not None:
            raise NotImplementedError(
                "CSA Draft serving export requires a compressed-entry/indexer-aware loader; "
                "retain the complete training checkpoint"
            )
        if attention["qsa"] is not None:
            raise NotImplementedError(
                "QSA Draft serving export requires an indexer/sparse-attention-aware "
                "loader; retain the complete training checkpoint"
            )
        if attention.get("dsa") is not None:
            raise NotImplementedError("DSA Draft requires native --format aspec export")
        if any(w is not None for w in attention["context_windows"]):
            raise NotImplementedError(
                "SWA Draft serving export requires an anchor-window-aware loader; "
                "retain the complete training checkpoint"
            )
        return super().save_draft_model_for_serving(output_dir)

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def _forward(
        self, batch: dict, *, anchor_seed: Optional[int] = None
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        device = torch.device("cuda")
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        hidden_states = batch["hidden_states"].to(device, non_blocking=True)

        loss_mask = batch["loss_mask"]
        if loss_mask.dim() == 3:
            loss_mask = loss_mask.squeeze(-1)
        loss_mask = loss_mask.to(device, non_blocking=True)
        ctx_doc_ids = batch.get("ctx_doc_ids")
        if ctx_doc_ids is not None:
            ctx_doc_ids = ctx_doc_ids.to(device, non_blocking=True)
        attention_mask = batch.get("attention_mask")
        if ctx_doc_ids is None and attention_mask is not None:
            attention_mask = attention_mask.to(device, non_blocking=True)
        else:
            attention_mask = None
        base_position_ids = batch.get("base_position_ids")
        if base_position_ids is not None:
            base_position_ids = base_position_ids.to(device, non_blocking=True)

        packed_context_positions = batch.get("packed_context_positions")
        if packed_context_positions is not None:
            packed_context_positions = packed_context_positions.to(device, non_blocking=True)

        # Gated on the same predicate as the norm load: feeding the loss an
        # unnormalised pre-norm tensor is silently wrong, so a run that skipped
        # the norm must not consume these at all.
        last_hidden_states = None
        if self.dflash.uses_target_hidden_states and batch.get("last_hidden_states") is not None:
            last_hidden_states = batch["last_hidden_states"].to(device, non_blocking=True)
            if self.verifier_norm is not None:
                with torch.no_grad():
                    last_hidden_states = self.verifier_norm(last_hidden_states)

        prefix_kwargs = {}
        from aspec.config.mtp_conditioning import target_mtp_features

        if target_mtp_features(getattr(self.dflash, "mtp_conditioning", None)):
            from aspec.models.dmtp import MTPPrefix

            mtp = batch.get("mtp_hidden_states")
            if mtp is None or mtp.shape[:2] != input_ids.shape:
                raise ValueError("Target-side MTP conditioning requires aligned mtp_hidden_states")
            prefix_kwargs["draft_prefix"] = MTPPrefix(mtp[:, 1:].to(device, non_blocking=True).detach())

        (
            loss,
            accuracy,
            loss_per_position,
            acc_per_position,
            count_per_position,
            loss_components,
            loss_terms,
        ) = self.model(
            input_ids=input_ids,
            hidden_states_list=hidden_states,
            loss_mask=loss_mask,
            lm_head_weight=self.target_lm_head_weight,
            last_hidden_states=last_hidden_states,
            ctx_doc_ids=ctx_doc_ids,
            base_position_ids=base_position_ids,
            packed_context_positions=packed_context_positions,
            anchor_seed=anchor_seed,
            packed_document_lengths=batch.get("packed_document_lengths"),
            attention_mask=attention_mask,
            **prefix_kwargs,
        )

        return (
            loss,
            accuracy,
            loss_per_position,
            acc_per_position,
            count_per_position,
            loss_components,
            loss_terms,
        )

    def _backward(self, loss: torch.Tensor, accumulation_steps: int = 1) -> torch.Tensor:
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()
        return loss

    def _normalize_accumulated_gradients(
        self,
        local_denominator: torch.Tensor,
        accumulation_steps: int,
    ) -> torch.Tensor:
        """Rescale accumulated gradients into a globally token-pooled mean.

        Backward ran on raw loss numerators, so gradients currently carry
        ``sum(numerator) / accumulation_steps``, and FSDP additionally *means*
        them over the shard group — hence the ``world_size * accumulation_steps``
        factor over the pooled denominator.

        The empty-window case goes through ``torch.where`` rather than a Python
        branch: a fully masked accumulation window is legal here (see the
        all-zero ``loss_mask`` fallback in ``DFlashModel``) and must not cost a
        device-to-host sync on every optimizer step.
        """
        denominator = local_denominator.detach().clone()
        group = getattr(self, "dp_group", None)
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size(group=group)
            if world_size > 1:
                dist.all_reduce(denominator, op=dist.ReduceOp.SUM, group=group)

        scale = torch.where(
            denominator > 0,
            (world_size * accumulation_steps) / denominator.clamp(min=1e-6),
            torch.zeros_like(denominator),
        )
        with torch.no_grad():
            gradients = [p.grad for p in self.optimizer.model_params if p.grad is not None]
            if scale.is_cuda:
                from aspec.models.ops.gradient_scaling import scale_gradients_

                scale_gradients_(gradients, scale)
            else:
                for gradient in gradients:
                    gradient.mul_(scale)
        return denominator

    # ------------------------------------------------------------------
    # Eval (no-grad forward on CPU-cached data) — mirrors Eagle3Trainer
    # ------------------------------------------------------------------

    def eval_forward(self, batch: dict) -> dict:
        """Single forward pass without backward — returns per-position tensors."""
        with torch.no_grad():
            (
                _,
                _,
                loss_per_position,
                acc_per_position,
                count_per_position,
                loss_components,
                _,
            ) = self._forward(batch)
        return {
            "loss_pp": loss_per_position.detach(),
            "acc_pp": acc_per_position.detach(),
            "count_pp": count_per_position.detach(),
            **loss_components,
        }

    def _reduce_position_metrics(
        self,
        all_step_metrics: list[dict],
        *,
        loss_key: str,
        acc_key: str,
        count_key: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_sum_pp = torch.stack([m[loss_key] * m[count_key] for m in all_step_metrics]).sum(dim=0)
        correct_sum_pp = torch.stack([m[acc_key] * m[count_key] for m in all_step_metrics]).sum(
            dim=0
        )
        count_pp = torch.stack([m[count_key] for m in all_step_metrics]).sum(dim=0)

        dist.all_reduce(loss_sum_pp, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_sum_pp, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_pp, op=dist.ReduceOp.SUM)

        safe_count_pp = count_pp.clamp(min=1.0)
        avg_loss_pp = loss_sum_pp / safe_count_pp
        avg_acc_pp = correct_sum_pp / safe_count_pp
        return avg_loss_pp, avg_acc_pp, count_pp

    def _compute_scalar_metrics(
        self,
        pred_loss_pp: torch.Tensor,
        pred_acc_pp: torch.Tensor,
        pred_count_pp: torch.Tensor,
    ) -> Tuple[float, float]:
        safe_total_count = pred_count_pp.sum().clamp(min=1.0)
        avg_acc = ((pred_acc_pp * pred_count_pp).sum() / safe_total_count).item()

        gamma = self.loss_decay_gamma
        if gamma is not None and gamma > 0:
            k = torch.arange(pred_loss_pp.shape[0], device=pred_loss_pp.device)
            weights = torch.exp(-k.float() / gamma)
        else:
            weights = torch.ones_like(pred_loss_pp)

        weighted_counts = pred_count_pp * weights
        safe_weighted_count = weighted_counts.sum().clamp(min=1.0)
        avg_loss = ((pred_loss_pp * weighted_counts).sum() / safe_weighted_count).item()
        return avg_loss, avg_acc

    def _reduce_loss_components(self, all_step_metrics: list[dict], prefix: str) -> dict:
        """Reduce extra loss components into ``{prefix}{key}`` token-pooled means.

        Components arrive as ``(numerator, denominator)`` pairs and are pooled the
        same way the optimized objective is: sum both sides over the accumulation
        window and the data-parallel group, then divide once. Averaging pre-divided
        per-micro-batch means instead would weight a sparsely supervised
        micro-batch the same as a dense one, so the component curves could diverge
        from the loss actually being optimized.

        All keys share one all-reduce and one host sync.
        """
        keys: list[str] = []
        totals: list[torch.Tensor] = []
        for key in (
            *self._extra_loss_component_keys,
            "router_aux_loss",
            "router_z_loss",
            "router_max_load",
            "router_used_experts",
            "csa_indexer_loss",
            "qsa_indexer_loss",
            "dsa_indexer_loss",
        ):
            pairs = [m[key] for m in all_step_metrics if key in m]
            if not pairs:
                continue
            keys.append(key)
            totals.append(
                torch.stack(
                    [
                        torch.stack(
                            (
                                numerator.detach().float().reshape(()),
                                denominator.detach().float().reshape(()),
                            )
                        )
                        for numerator, denominator in pairs
                    ]
                ).sum(dim=0)
            )
        if not keys:
            return {}

        packed = torch.stack(totals)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        pooled = (packed[:, 0] / packed[:, 1].clamp(min=1e-6)).tolist()
        return {f"{prefix}{key}": value for key, value in zip(keys, pooled, strict=True)}

    def eval_from_cache(self) -> dict:
        """Run forward-only eval over all CPU-cached eval samples.

        Samples are stored individually (no padding). Re-collate into batches
        of ``eval_micro_batch_size`` (or ``micro_batch_size``) so the eval
        forward batch size is independent of cache generation throughput.
        """
        if not getattr(self, "_eval_cache", None):
            return {}

        eval_mbs = getattr(self.args, "eval_micro_batch_size", None) or self.args.micro_batch_size

        self.model.eval()
        all_metrics: list[dict] = []
        for i in range(0, len(self._eval_cache), eval_mbs):
            chunk = self._eval_cache[i : i + eval_mbs]
            batch = self._eval_collator(chunk)
            gpu_batch = {
                k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }
            all_metrics.append(self.eval_forward(gpu_batch))

        self.model.train()

        return self._aggregate_eval_metrics(all_metrics)

    def _prediction_metric_prefix(self) -> str:
        """MTP observations cover the unknown suffix conditional on visible T1."""
        return "suffix_" if getattr(getattr(self, "dflash", None), "mtp_conditioning", None) else ""

    def _aggregate_eval_metrics(self, all_step_metrics: list[dict]) -> dict:
        if not all_step_metrics:
            return {}

        loss_components = self._reduce_loss_components(all_step_metrics, "eval/")
        warmup = self._indexer_warmup_contract()
        if warmup is not None:
            key, coefficient = warmup
            metric_key = f"eval/{key}"
            if metric_key not in loss_components:
                raise RuntimeError("attention indexer warm-up metrics are missing indexer KL")
            metrics = {
                "eval/avg_loss": coefficient * loss_components[metric_key],
                **loss_components,
            }
            if dist.get_rank() == 0:
                logger.info(f"eval: indexer_kl={metrics['eval/avg_loss']:.4f}")
            return metrics

        avg_loss_pp, avg_acc_pp, count_pp = self._reduce_position_metrics(
            all_step_metrics,
            loss_key="loss_pp",
            acc_key="acc_pp",
            count_key="count_pp",
        )

        # Observe only unknown predictions, excluding visible T1 when conditioned.
        pred_loss_pp = avg_loss_pp[self._anchor_slot_offset :]
        pred_acc_pp = avg_acc_pp[self._anchor_slot_offset :]
        pred_count_pp = count_pp[self._anchor_slot_offset :]

        cumulative = 1.0
        simulated_acc_len = 0.0
        for i in range(pred_acc_pp.shape[0]):
            cumulative *= pred_acc_pp[i].item()
            simulated_acc_len += cumulative

        weighted_avg_loss, avg_acc = self._compute_scalar_metrics(
            pred_loss_pp, pred_acc_pp, pred_count_pp
        )
        prefix = self._prediction_metric_prefix()

        metrics: dict = {
            "eval/avg_loss": weighted_avg_loss,
            "eval/avg_acc": avg_acc,
            f"eval/{prefix}simulated_acc_len": simulated_acc_len,
        }
        for i in range(pred_loss_pp.shape[0]):
            metrics[f"eval/{prefix}ploss_{i}"] = pred_loss_pp[i].item()
            metrics[f"eval/{prefix}acc_{i}"] = pred_acc_pp[i].item()

        metrics.update(loss_components)

        if dist.get_rank() == 0:
            logger.info(
                f"eval: loss={weighted_avg_loss:.4f}, acc={avg_acc:.4f}, "
                f"{prefix}sim_acc_len={simulated_acc_len:.2f}"
            )

        return metrics

    # ------------------------------------------------------------------
    # Subclass contract implementations
    # ------------------------------------------------------------------

    def _train_step(
        self,
        batch: dict,
        accumulation_steps: int,
        step: int,
        batch_idx: int,
        num_batches: int,
    ) -> dict:
        evt_fwd_s = torch.cuda.Event(enable_timing=True)
        evt_fwd_e = torch.cuda.Event(enable_timing=True)
        evt_bwd_s = torch.cuda.Event(enable_timing=True)
        evt_bwd_e = torch.cuda.Event(enable_timing=True)

        evt_fwd_s.record()
        (
            loss,
            accuracy,
            loss_per_position,
            acc_per_position,
            count_per_position,
            loss_components,
            loss_terms,
        ) = self._forward(
            batch,
            anchor_seed=(
                anchor_sampling_seed(
                    getattr(self.args, "seed", 0), self.global_step, batch_idx, self.dp_rank
                )
                if getattr(self.args, "deterministic_anchor_sampling", False)
                else None
            ),
        )
        evt_fwd_e.record()

        loss_numerator, loss_denominator = loss_terms
        if batch_idx == 0:
            self._accumulated_loss_denominator = loss_denominator.detach().clone()
        else:
            self._accumulated_loss_denominator.add_(loss_denominator.detach())

        evt_bwd_s.record()
        self._backward(loss_numerator, accumulation_steps=accumulation_steps)
        if batch_idx == num_batches - 1:
            self._normalize_accumulated_gradients(
                self._accumulated_loss_denominator,
                accumulation_steps,
            )
        evt_bwd_e.record()

        return {
            "loss": loss.detach(),
            "accuracy": accuracy.detach(),
            "loss_per_position": loss_per_position.detach(),
            "acc_per_position": acc_per_position.detach(),
            "count_per_position": count_per_position.detach(),
            "total_loss": loss.detach(),
            "_fwd_events": (evt_fwd_s, evt_fwd_e),
            "_bwd_events": (evt_bwd_s, evt_bwd_e),
            **loss_components,
        }

    def _after_optimizer_step(self, all_step_metrics, grad_norm):
        self._update_ling_router_bias(all_step_metrics, grad_norm)

    def _is_csa_indexer_warmup(self) -> bool:
        config = getattr(getattr(self, "draft_model", None), "config", None)
        return (
            getattr(config, "csa_training_stage", None) == "indexer_warmup"
            or getattr(config, "qsa_training_stage", None) == "indexer_warmup"
            or getattr(config, "dsa_training_stage", None) == "indexer_warmup"
        )

    def _indexer_warmup_contract(self) -> tuple[str, float] | None:
        config = getattr(getattr(self, "draft_model", None), "config", None)
        if getattr(config, "csa_training_stage", None) == "indexer_warmup":
            return "csa_indexer_loss", float(config.csa_indexer_loss_coef)
        if getattr(config, "qsa_training_stage", None) == "indexer_warmup":
            return "qsa_indexer_loss", float(config.qsa_indexer_loss_coef)
        if getattr(config, "dsa_training_stage", None) == "indexer_warmup":
            return "dsa_indexer_loss", float(config.dsa_indexer_loss_coef)
        return None

    def _update_ling_router_bias(self, all_step_metrics, grad_norm):
        if self._indexer_warmup_contract() is not None:
            # A frozen main model includes its non-gradient router control state.
            return
        draft = getattr(self, "draft_model", None)
        config = getattr(draft, "config", None)
        if getattr(config, "ffn_type", "dense") != "ling_moe":
            return
        from aspec.models.draft.ling_moe import update_ling_bias

        if not all("_ling_router_counts" in m for m in all_step_metrics):
            raise RuntimeError("Ling training step is missing router load observations")
        counts = torch.stack([m["_ling_router_counts"] for m in all_step_metrics]).sum(0)
        update_ling_bias([layer.mlp for layer in draft.layers], counts,
                         config.moe_router_bias_update_rate, config.bias_zero_mean_update,
                         group=getattr(self, "dp_group", None),
                         apply_update=torch.isfinite(grad_norm) if grad_norm is not None else False)

    def _aggregate_metrics(
        self, all_step_metrics: list[dict], step: int, *, grad_norm: torch.Tensor = None
    ) -> dict:
        if not all_step_metrics:
            return {}

        loss_components = self._reduce_loss_components(all_step_metrics, "train/")
        warmup = self._indexer_warmup_contract()
        if warmup is not None:
            key, coefficient = warmup
            metric_key = f"train/{key}"
            if metric_key not in loss_components:
                raise RuntimeError("attention indexer warm-up metrics are missing indexer KL")
            metrics = {
                "train/avg_loss": coefficient * loss_components[metric_key],
                "train/grad_norm": grad_norm.item() if grad_norm is not None else 0.0,
                "train/global_step": self.global_step,
                "train/lr": self.optimizer.get_learning_rate(),
                "train/step": step,
                **loss_components,
            }
            fwd_ms = sum(
                m["_fwd_events"][0].elapsed_time(m["_fwd_events"][1])
                for m in all_step_metrics
                if "_fwd_events" in m
            )
            bwd_ms = sum(
                m["_bwd_events"][0].elapsed_time(m["_bwd_events"][1])
                for m in all_step_metrics
                if "_bwd_events" in m
            )
            metrics["perf/forward_time"] = fwd_ms / 1000.0
            metrics["perf/backward_time"] = bwd_ms / 1000.0
            if dist.get_rank() == 0 and (step % 5 == 0 or step <= 5):
                logger.info(
                    f"COMPUTE_BREAKDOWN step={step}: "
                    f"forward={fwd_ms:.1f}ms backward={bwd_ms:.1f}ms"
                )
            return metrics

        avg_loss_pp, avg_acc_pp, count_pp = self._reduce_position_metrics(
            all_step_metrics,
            loss_key="loss_per_position",
            acc_key="acc_per_position",
            count_key="count_per_position",
        )

        # Re-index unknown predictions from zero; conditioned suffix_0 denotes T2.
        pred_loss_pp = avg_loss_pp[self._anchor_slot_offset :]
        pred_acc_pp = avg_acc_pp[self._anchor_slot_offset :]
        pred_count_pp = count_pp[self._anchor_slot_offset :]

        # Simulated accepted length: acc_0 + acc_0*acc_1 + ... + prod(acc_0..acc_{B-2})
        # Models the expected number of consecutively accepted draft tokens.
        cumulative = 1.0
        simulated_acc_len = 0.0
        for i in range(pred_acc_pp.shape[0]):
            cumulative *= pred_acc_pp[i].item()
            simulated_acc_len += cumulative

        avg_loss, avg_acc = self._compute_scalar_metrics(pred_loss_pp, pred_acc_pp, pred_count_pp)
        prefix = self._prediction_metric_prefix()

        metrics = {
            "train/avg_loss": avg_loss,
            "train/avg_acc": avg_acc,
            f"train/{prefix}simulated_acc_len": simulated_acc_len,
            "train/grad_norm": grad_norm.item() if grad_norm is not None else 0.0,
            "train/global_step": self.global_step,
            "train/lr": self.optimizer.get_learning_rate(),
            "train/step": step,
        }

        for i in range(pred_loss_pp.shape[0]):
            metrics[f"train/{prefix}ploss_{i}"] = pred_loss_pp[i].item()
            metrics[f"train/{prefix}acc_{i}"] = pred_acc_pp[i].item()

        metrics.update(loss_components)

        # Sub-timing breakdown (forward vs backward)
        fwd_ms = sum(
            m["_fwd_events"][0].elapsed_time(m["_fwd_events"][1])
            for m in all_step_metrics
            if "_fwd_events" in m
        )
        bwd_ms = sum(
            m["_bwd_events"][0].elapsed_time(m["_bwd_events"][1])
            for m in all_step_metrics
            if "_bwd_events" in m
        )
        metrics["perf/forward_time"] = fwd_ms / 1000.0
        metrics["perf/backward_time"] = bwd_ms / 1000.0

        if dist.get_rank() == 0 and (step % 5 == 0 or step <= 5):
            logger.info(
                f"COMPUTE_BREAKDOWN step={step}: forward={fwd_ms:.1f}ms backward={bwd_ms:.1f}ms"
            )

        if dist.get_rank() == 0:
            logger.debug(f"step {step}: {metrics}")

        return metrics
