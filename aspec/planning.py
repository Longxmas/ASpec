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

"""Compile supported TorchSpec-style input into ASpec's canonical run plan."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from omegaconf import DictConfig, ListConfig, OmegaConf

from aspec.config.draft_algorithm import DRAFT_ALGORITHMS
from aspec.config.draft_attention import (
    draft_thd_contract,
    resolve_draft_attention,
    resolve_draft_head_layout,
)
from aspec.config.draft_execution import (
    is_cudnn_flex_backend,
    is_direct_cp_backend,
    is_fa4_backend,
    resolve_draft_execution_capability,
)
from aspec.config.draft_structure import resolve_draft_structure
from aspec.config.mooncake_config import MooncakeConfig, resolve_rdma_routes
from aspec.config.train_config import config_to_flat_args
from aspec.feature_contract import FeatureContractError, build_target_feature_contract
from aspec.inference.concurrency import direct_output_sample_capacity
from aspec.inference.context import (
    ContextContractError,
    compile_offline_context_contract,
    compile_sglang_context_contract,
)
from aspec.locality import allocate_node_ready_watermarks

TORCHSPEC_COMPAT_REVISION = "6c042a87140a84d13839e341ece2c5c3ada918bc"
PLAN_SCHEMA_VERSION = "aspec.execution-plan/v0alpha10"


_PROTECTED_SGLANG_EXTRA_ARGS = frozenset(
    {
        "allow_auto_truncate",
        "aux_hidden_state_layer_ids",
        "base_gpu_id",
        "chunked_prefill_size",
        "disable_cuda_graph",
        "disable_radix_cache",
        "context_length",
        "dist_init_addr",
        "dist_timeout",
        "enable_aux_hidden_states",
        "enable_multimodal",
        "enable_return_hidden_states",
        "enable_spec_training_mooncake",
        "gpu_id_step",
        "mem_fraction_static",
        "model_path",
        "nccl_port",
        "nnodes",
        "node_rank",
        "port",
        "pp_size",
        "spec_training_store_last_hidden_states",
        "tp_size",
        "trust_remote_code",
    }
)


class PlanCompileError(ValueError):
    """The requested run cannot be represented by the current ASpec path."""


@dataclass(frozen=True)
class WorkloadContract:
    """Canonical description of what is trained, independent of host topology."""

    models: Mapping[str, Any]
    data: Mapping[str, Any]
    objective: Mapping[str, Any]
    features: Mapping[str, Any]
    duration: Mapping[str, Any]


@dataclass(frozen=True)
class ExecutionPolicy:
    """Canonical execution requirements and preferences for a workload."""

    backends: Mapping[str, Any]
    resources: Mapping[str, Any]
    parallelism: Mapping[str, Any]
    pipeline: Mapping[str, Any]
    capacity: Mapping[str, Any]
    measurement: Mapping[str, Any]
    artifacts: Mapping[str, Any]


@dataclass(frozen=True)
class CapabilityDecision:
    """One requested capability and its compile-time and runtime states."""

    name: str
    requirement: str
    requested: Any
    resolved: Any
    effective: Any = None
    effective_state: str = "pending_runtime_evidence"


@dataclass(frozen=True)
class ExecutionPlan:
    """Stable, inspectable output of the plan compiler.

    ``_runtime_values`` is the temporary adapter boundary for the retained
    TorchSpec-backed runtime. It is deliberately excluded from serialized plan
    output: compatibility input is compiled once, then a fresh Namespace is
    created from these resolved implementation values.
    """

    source: Mapping[str, Any]
    workload: WorkloadContract
    policy: ExecutionPolicy
    capabilities: tuple[CapabilityDecision, ...]
    fallbacks: tuple[Mapping[str, Any], ...]
    omissions: tuple[Mapping[str, Any], ...]
    diagnostics: tuple[Mapping[str, Any], ...]
    _runtime_values: Mapping[str, Any] = field(repr=False, compare=False)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": PLAN_SCHEMA_VERSION,
            "source": copy.deepcopy(dict(self.source)),
            "workload": asdict(self.workload),
            "policy": asdict(self.policy),
            "capabilities": [asdict(item) for item in self.capabilities],
            "fallbacks": [copy.deepcopy(dict(item)) for item in self.fallbacks],
            "omissions": [copy.deepcopy(dict(item)) for item in self.omissions],
            "diagnostics": [copy.deepcopy(dict(item)) for item in self.diagnostics],
        }

    @property
    def plan_id(self) -> str:
        payload = json.dumps(
            self._payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["plan_id"] = self.plan_id
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)

    def to_runtime_args(self) -> argparse.Namespace:
        values = copy.deepcopy(dict(self._runtime_values))
        values["aspec_plan_id"] = self.plan_id
        values["aspec_feature_contract"] = copy.deepcopy(dict(self.workload.features))
        values["aspec_locality_domains"] = copy.deepcopy(
            list(self.policy.resources.get("locality_domains", ()))
        )
        if values.get("timeline_enabled"):
            if not values.get("timeline_output_dir"):
                values["timeline_output_dir"] = str(
                    Path(values.get("output_dir") or ".") / "traces"
                )
            if not values.get("timeline_run_id"):
                values["timeline_run_id"] = self.plan_id.replace(":", "-")
        return argparse.Namespace(**values)


def _plain(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return copy.deepcopy(value)


def _select(config: DictConfig, key: str, default: Any = None) -> Any:
    return _plain(OmegaConf.select(config, key, default=default))


def _source_record(config_path: str | None, cli_args: Sequence[str] | None) -> dict[str, Any]:
    source_path = None
    source_sha256 = None
    if config_path is not None:
        path = Path(config_path).expanduser().resolve()
        source_path = str(path)
        source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    override_keys = []
    for item in cli_args or ():
        if "=" not in item:
            raise PlanCompileError(
                f"Unsupported CLI argument {item!r}; overrides must use dotted key=value syntax"
            )
        override_keys.append(item.split("=", 1)[0])

    return {
        "format": "torchspec_yaml",
        "reference_revision": TORCHSPEC_COMPAT_REVISION,
        "conversion": "deterministic",
        "path": source_path,
        "sha256": source_sha256,
        "cli_override_keys": sorted(override_keys),
        "compatibility": {
            "unsupported_fields": [],
            "ambiguous_fields": [],
        },
    }


def _resolved_config_sha256(config: DictConfig) -> str:
    serialized = json.dumps(
        _plain(config), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _load_draft_descriptor(config: DictConfig) -> dict[str, Any]:
    configured_path = _select(config, "model.draft_model_config")
    if not configured_path:
        raise PlanCompileError(
            "ASpec requires model.draft_model_config so the drafter family and feature "
            "contract can be resolved before backend startup"
        )

    path = Path(configured_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.is_file():
        raise PlanCompileError(f"Draft model config does not exist: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanCompileError(f"Cannot read draft model config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise PlanCompileError(f"Draft model config must contain a JSON object: {path}")

    architectures = raw.get("architectures")
    if not isinstance(architectures, list) or len(architectures) != 1:
        raise PlanCompileError(
            "Draft model config must declare exactly one architecture; "
            f"got {architectures!r} in {path}"
        )
    architecture = architectures[0]
    family = DRAFT_ALGORITHMS.get(architecture)
    if family is None:
        raise PlanCompileError(
            f"Draft architecture {architecture!r} is outside the current DSpark/DFlash/DFlash2/EAGLE3 scope"
        )

    if family == "mtp":
        from aspec.models.draft.mtp import NativeMTPConfig

        try:
            raw = NativeMTPConfig.from_dict(raw).to_dict()
        except (ValueError, TypeError) as exc:
            raise PlanCompileError(str(exc)) from exc
        steps = _select(config, "training.mtp_steps")
        beta = _select(config, "training.mtp_step_weight_beta")
        if type(steps) is not int or steps <= 0:
            raise PlanCompileError("Native MTP steps must be a positive integer")
        if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(beta) or beta < 0:
            raise PlanCompileError("Native MTP step_weight_beta must be finite and non-negative")
        if _select(config, "training.attention_backend") not in {"sdpa", "fa4"} or (
            _select(config, "training.attention_backend") != "fa4" and (
            _select(config, "training.dflash_packing", False)
            or _select(config, "training.draft_context_parallel_size", 1) != 1
        )):
            raise PlanCompileError("Native MTP packing/CP requires FA4; SDPA remains the CP1 reference")
        if _select(config, "training.compile_model", False):
            raise PlanCompileError("Native MTP whole-model compile has not been enabled")
        if _select(config, "inference.store_last_hidden_states"):
            raise PlanCompileError("Native MTP publishes final post-norm features once; set store_last_hidden_states=false")

    if family == "eagle4":
        from aspec.config.eagle4 import Eagle4Config, eagle4_policy

        try:
            raw = Eagle4Config.from_dict(raw).to_dict()
            eagle4 = eagle4_policy(raw)
        except (ValueError, TypeError) as exc:
            raise PlanCompileError(str(exc)) from exc
    else:
        eagle4 = None

    if family == "dflash2":
        from aspec.models.draft.dflash2 import DFlash2Config

        try:
            raw = DFlash2Config.from_dict(raw).to_dict()
        except (ValueError, TypeError) as exc:
            raise PlanCompileError(str(exc)) from exc
        if raw["block_size"] != _select(config, "training.dflash_block_size"):
            raise PlanCompileError("DFlash2 convolution block_size must match training.dflash_block_size")
        alpha = _select(config, "training.dflash2_selector_loss_alpha")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha < 0:
            raise PlanCompileError("DFlash2 selector_loss_alpha must be finite and non-negative")

    target_layer_ids = raw.get("target_layer_ids")
    if not isinstance(target_layer_ids, list) or not target_layer_ids:
        raise PlanCompileError(
            "ASpec requires non-empty target_layer_ids in model.draft_model_config "
            "so the target feature contract is explicit at plan time"
        )
    if not all(isinstance(layer_id, int) and layer_id >= 0 for layer_id in target_layer_ids):
        raise PlanCompileError(f"target_layer_ids must be non-negative integers: {target_layer_ids!r}")

    try:
        structure = resolve_draft_structure(raw)
        attention = resolve_draft_attention(raw)
        attention["head_layout"] = resolve_draft_head_layout(raw)
    except ValueError as exc:
        raise PlanCompileError(str(exc)) from exc
    backend = _select(config, "training.attention_backend")
    l1_sampling_enabled = False
    for side in ("target", "draft"):
        prefix = f"training.dspark_l1_{side}_"
        temperature = _select(config, prefix + "temperature", 1.0)
        top_p = _select(config, prefix + "top_p", 1.0)
        top_k = _select(config, prefix + "top_k", -1)
        if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
                or not math.isfinite(temperature) or temperature <= 0):
            raise PlanCompileError(f"{prefix}temperature must be finite and positive")
        if (isinstance(top_p, bool) or not isinstance(top_p, (int, float))
                or not math.isfinite(top_p) or not 0 < top_p <= 1):
            raise PlanCompileError(f"{prefix}top_p must be in (0, 1]")
        if type(top_k) is not int or (top_k != -1 and top_k < 1):
            raise PlanCompileError(f"{prefix}top_k must be -1 or a positive integer")
        l1_sampling_enabled |= (temperature, top_p, top_k) != (1.0, 1.0, -1)
    if l1_sampling_enabled:
        indexer_warmup = any((attention.get(kind) or {}).get("training_stage") == "indexer_warmup"
                            for kind in ("csa", "qsa", "dsa"))
        if family != "dspark" or _select(config, "training.dspark_l1_loss_alpha") <= 0 or indexer_warmup:
            raise PlanCompileError("DSpark L1 sampling requires DSpark with positive L1 weight outside indexer warmup")
    eagle3_target_cache = _select(config, "training.eagle3_target_cache", "lazy")
    eagle3_loss_checkpointing = _select(
        config, "training.eagle3_loss_gradient_checkpointing"
    )
    if eagle3_target_cache not in {"lazy", "logits"}:
        raise PlanCompileError("training.eagle3_target_cache must be 'lazy' or 'logits'")
    if eagle3_loss_checkpointing is not None and type(eagle3_loss_checkpointing) is not bool:
        raise PlanCompileError(
            "training.eagle3_loss_gradient_checkpointing must be a boolean or null"
        )
    if family != "eagle3" and eagle3_target_cache != "lazy":
        raise PlanCompileError(
            "training.eagle3_target_cache=logits is implemented only for EAGLE3"
        )
    if family != "eagle3" and eagle3_loss_checkpointing is False:
        raise PlanCompileError(
            "training.eagle3_loss_gradient_checkpointing=false is implemented only for EAGLE3"
        )
    if backend == "cudnn_flex_attention" and family not in {
        "dflash",
        "dflash2",
        "dspark",
    }:
        raise PlanCompileError(
            "cuDNN FlexAttention is implemented for the DSpark/DFlash family; "
            f"{family} must use one of its supported attention backends"
        )
    if family == "dflash2" and (raw.get("is_causal", False) or any(
        window is not None for window in attention["context_windows"]
    )) and backend not in {"flex_attention", "cudnn_flex_attention", "fa4"}:
        raise PlanCompileError(
            "Causal/SWA DFlash2 requires flex_attention, cudnn_flex_attention or fa4"
        )
    if family in {"dflash", "dflash2"}:
        for key in ("dflash_ce_loss_alpha", "dflash_l1_loss_alpha"):
            value = _select(config, "training." + key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise PlanCompileError(f"{key} must be finite and non-negative")
        dsa_warmup = (attention.get("dsa") or {}).get("training_stage") == "indexer_warmup"
        if (_select(config, "training.dflash_l1_loss_alpha") > 0
                and not _select(config, "inference.store_last_hidden_states") and not dsa_warmup):
            raise PlanCompileError("DFlash L1 distillation requires inference.store_last_hidden_states=true")
    if family in {"eagle3", "eagle4"}:
        ttt_length = _select(config, "training.ttt_length")
        if type(ttt_length) is not int or ttt_length < 1:
            raise PlanCompileError("EAGLE3 ttt_length must be a positive integer")
        if _select(config, "training.loss_type") not in {"forward_kl", "lk_alpha", "lk_lambda"}:
            raise PlanCompileError("Unsupported EAGLE3 loss_type")
        eta = _select(config, "training.lk_eta")
        if isinstance(eta, bool) or not isinstance(eta, (int, float)) or not math.isfinite(eta) or eta < 0:
            raise PlanCompileError("EAGLE3 lk_eta must be finite and non-negative")
        weights = _select(config, "training.ploss_weights")
        if weights is not None and (
            len(weights) != ttt_length or any(
                isinstance(w, bool) or not isinstance(w, (int, float))
                or not math.isfinite(w) or w < 0 for w in weights
            ) or sum(weights) <= 0
        ):
            raise PlanCompileError("EAGLE3 ploss_weights must match ttt_length, be finite/non-negative and have positive sum")
        unsupported = []
        if backend not in {"sdpa", "flex_attention", "fa4"}:
            unsupported.append("attention backend (use sdpa, flex_attention or fa4)")
        if _select(config, "training.draft_context_parallel_size", 1) != 1 and backend != "fa4":
            unsupported.append("context parallel (use fa4)")
        if _select(config, "training.dflash_packing", False) and backend != "fa4":
            unsupported.append("packing (use fa4 for document-isolated THD)")
        if _select(config, "training.eagle3_num_anchors", 0):
            unsupported.append("anchored EAGLE3")
        if _select(config, "training.compile_model", False):
            unsupported.append("whole-model compile (loss kernels are compiled independently)")
        if family == "eagle3" and (structure["ffn_type"] != "dense" or any(
            kind != "full_attention" for kind in attention["layer_types"]
        )):
            unsupported.append("non-Dense or sparse attention structure")
        if raw.get("num_hidden_layers", 1) != 1:
            unsupported.append("multiple decoder layers")
        if any(raw.get(key, False) for key in ("bias", "attention_bias", "mlp_bias", "norm_before_fc")) or (
            raw.get("eagle_config") or {}
        ).get("norm_before_fc", False):
            unsupported.append("projection bias or norm_before_fc")
        draft_vocab = int(raw.get("draft_vocab_size") or raw["vocab_size"])
        if not 0 < draft_vocab <= raw["vocab_size"]:
            raise PlanCompileError("EAGLE3 draft_vocab_size must be positive and no larger than vocab_size")
        if eagle3_target_cache == "logits":
            if backend not in {"sdpa", "flex_attention"}:
                unsupported.append("logits target cache (use padded sdpa or flex_attention)")
            if draft_vocab != raw["vocab_size"]:
                unsupported.append("logits target cache with vocabulary pruning")
        if eagle3_loss_checkpointing is False:
            if backend not in {"sdpa", "flex_attention"}:
                unsupported.append(
                    "disabled loss checkpointing (use padded sdpa or flex_attention)"
                )
            if draft_vocab != raw["vocab_size"]:
                unsupported.append("disabled loss checkpointing with vocabulary pruning")
        if draft_vocab != raw["vocab_size"] and _select(config, "model.freeze_lm_head", False) and not (
            _select(config, "model.initial_draft_model_path") or _select(config, "training.load_path")
        ):
            raise PlanCompileError("A frozen pruned EAGLE3 head requires initialized weights and mapping")
        if raw.get("num_aux_hidden_states", len(target_layer_ids)) != len(target_layer_ids):
            raise PlanCompileError("EAGLE3 num_aux_hidden_states must match target_layer_ids")
        if raw.get("use_aux_norm") and not raw.get("fc_norm"):
            raise PlanCompileError("EAGLE3 use_aux_norm requires fc_norm=true; the trainer does not interpret the serving alias")
        eagle_layers = (raw.get("eagle_config") or {}).get("eagle_aux_hidden_state_layer_ids")
        if eagle_layers is not None and eagle_layers != target_layer_ids:
            raise PlanCompileError("EAGLE3 eagle_config layers must match target_layer_ids")
        if unsupported:
            raise PlanCompileError("EAGLE3 does not yet support: " + ", ".join(unsupported))
        if eagle4 is not None:
            if any(kind not in {"full_attention", "sliding_attention"} for kind in attention["layer_types"]):
                raise PlanCompileError("EAGLE4 supports full/SWA attention, not CSA/QSA")
            if backend == "flex_attention" and attention["sliding_window"] is not None:
                raise PlanCompileError("EAGLE4 SWA uses fa4 THD/CP or sdpa reference")
            if eagle4["stage"] == "candidate_head":
                if _select(config, "training.loss_type") != "forward_kl":
                    raise PlanCompileError("Candidate stage uses frozen-Draft retrieval forward-KL")
                if not (_select(config, "model.initial_draft_model_path") or _select(config, "training.load_path")):
                    raise PlanCompileError("Candidate stage requires pretrained Draft weights or exact resume")
                if _select(config, "training.continual_training", False) and _select(config, "training.load_path"):
                    raise PlanCompileError("Use initial_draft_model_path for candidate stage initialization")
                if _select(config, "inference.store_last_hidden_states"):
                    raise PlanCompileError("Candidate teacher is frozen Draft; set store_last_hidden_states=false")
        attention["draft_block_attention"] = "causal_ttt_cache"
    materialized_mla = attention["head_layout"]["type"] == "mla_materialized"
    if materialized_mla:
        if backend not in ({"sdpa", "flex_attention", "fa4"} if family in {"eagle3", "eagle4"}
                           else {"flex_attention", "cudnn_flex_attention", "fa4"}):
            raise PlanCompileError(
                "Materialized MLA Draft requires flex_attention, "
                "cudnn_flex_attention or fa4"
            )
        if raw.get("mla_use_output_gate", False):
            raise PlanCompileError(
                "Materialized MLA Draft does not support mla_use_output_gate=True; "
                "the retained K3 checkpoint contract has no output-gate weights"
            )
        if any(attention.get(kind) is not None for kind in ("csa", "qsa", "dsa")):
            if attention.get("dsa") is not None:
                raise PlanCompileError("Materialized MLA Draft has not completed DSA integration")
            raise PlanCompileError(
                "Materialized MLA Draft has not completed CSA/QSA integration; "
                "use full or sliding attention"
            )
        head_layout = attention["head_layout"]
        if backend == "cudnn_flex_attention" and (
            head_layout["head_dim"], head_layout["value_head_dim"]
        ) != (192, 128):
            raise PlanCompileError(
                "Materialized MLA with cudnn_flex_attention requires "
                "(qk_head_dim, value_head_dim)=(192, 128); got "
                f"({head_layout['head_dim']}, {head_layout['value_head_dim']})"
            )
    csa_requested = attention["csa"] is not None
    if csa_requested:
        if backend not in {"flex_attention", "fa4"}:
            raise PlanCompileError(
                "CSA Draft requires training.attention_backend=flex_attention or fa4"
            )
        if _select(config, "training.compile_model", False):
            raise PlanCompileError(
                "CSA Draft dynamic selected-entry masks do not yet support compile_model"
            )
        if _select(config, "training.dflash_packing", False) and backend != "fa4":
            raise PlanCompileError(
                "CSA Draft packing requires training.attention_backend=fa4"
            )
    qsa_requested = attention["qsa"] is not None
    if qsa_requested:
        if backend not in {"flex_attention", "fa4"}:
            raise PlanCompileError(
                "QSA Draft requires training.attention_backend=flex_attention or fa4"
            )
        if _select(config, "training.compile_model", False):
            raise PlanCompileError(
                "QSA Draft dynamic selection does not yet support compile_model"
            )
        if _select(config, "training.dflash_packing", False) and backend != "fa4":
            raise PlanCompileError(
                "QSA Draft packing requires training.attention_backend=fa4"
            )
    if attention.get("dsa") is not None:
        if family not in {"dflash", "dspark"}:
            raise PlanCompileError("DSA is implemented for the DSpark/DFlash GQA Draft path")
        if backend not in {"flex_attention", "fa4"}:
            raise PlanCompileError("DSA Draft requires training.attention_backend=flex_attention or fa4")
        if _select(config, "training.compile_model", False):
            raise PlanCompileError("DSA dynamic selection does not yet support compile_model")
        if _select(config, "training.dflash_packing", False) and backend != "fa4":
            raise PlanCompileError("DSA packing requires training.attention_backend=fa4")
    if any(w is not None for w in attention["context_windows"]) and not (
        (family == "eagle4" and backend == "sdpa") or
        is_fa4_backend(backend) or is_cudnn_flex_backend(backend) or backend == "flex_attention"
    ):
        raise PlanCompileError("SWA Draft requires flex_attention, cudnn_flex_attention or fa4")
    if structure["ffn_type"] != "dense" and _select(config, "training.compile_model", False):
        raise PlanCompileError("MoE Draft torch.compile is not yet supported; disable compile_model")

    return {
        "structure": structure,
        "eagle4": eagle4,
        "attention": attention,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "architecture": architecture,
        "family": family,
        "model_type": raw.get("model_type"),
        "num_target_layers": raw.get("num_target_layers"),
        "target_hidden_size": raw.get("target_hidden_size"),
        "target_num_hidden_layers": raw.get("target_num_hidden_layers"),
        "target_layer_ids": target_layer_ids,
    }


def _read_local_target_config(target_path: str) -> Mapping[str, Any] | None:
    path = Path(target_path).expanduser()
    if path.is_dir():
        path = path / "config.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanCompileError(f"Cannot read target model config {path}: {exc}") from exc
    return value if isinstance(value, dict) else None


def _target_config_with_sglang_overrides(
    config: DictConfig, target_path: str
) -> Mapping[str, Any] | None:
    """Return the effective model config used for context admission.

    SGLang applies ``json_model_override_args`` to the loaded Hugging Face
    config before constructing the engine.  The plan compiler must inspect the
    same explicit model declaration; otherwise a valid YaRN extension is
    rejected even though the runtime receives it.  Backend-only context
    overrides remain separate and therefore still cannot manufacture a model
    context extension.
    """

    model_config = copy.deepcopy(_read_local_target_config(target_path))
    raw_overrides = _select(
        config,
        "inference.sglang.extra_args.json_model_override_args",
    )
    if raw_overrides is None:
        return model_config
    if isinstance(raw_overrides, str):
        try:
            raw_overrides = json.loads(raw_overrides)
        except json.JSONDecodeError as exc:
            raise PlanCompileError(
                "inference.sglang.extra_args.json_model_override_args must be valid JSON: "
                f"{exc}"
            ) from exc
    if not isinstance(raw_overrides, Mapping):
        raise PlanCompileError(
            "inference.sglang.extra_args.json_model_override_args must contain a JSON object"
        )

    effective = dict(model_config or {})
    effective.update(copy.deepcopy(dict(raw_overrides)))
    return effective


def _resolve_target_family(target_path: str) -> str:
    local_config = _read_local_target_config(target_path)
    markers: list[str] = [target_path]
    if local_config is not None:
        markers.append(str(local_config.get("model_type", "")))
        architectures = local_config.get("architectures", [])
        if isinstance(architectures, list):
            markers.extend(str(item) for item in architectures)
        if (
            local_config.get("model_type") == "bailing_hybrid"
            and isinstance(architectures, list)
            and "BailingMoeV3ForCausalLM" in architectures
        ):
            return "ling3"
    elif target_path.rstrip("/").split("/")[-1].lower() in {
        "ling-3.0-tiny", "ling-3.0-flash", "ling-3.1-flash",
    }:
        return "ling3"
    if any("qwen3" in marker.lower().replace("_", "") for marker in markers):
        return "qwen3"
    raise PlanCompileError(
        "The supported target scope is Qwen3 and Ling-3.x, but the target model family "
        f"could not be established from {target_path!r}"
    )


def _resolve_batch_size(args: argparse.Namespace) -> None:
    """Derive the retained runtime's batch and data-parallel values."""

    world_size = args.training_num_nodes * args.training_num_gpus_per_node
    execution = resolve_draft_execution_capability(
        getattr(args, "attention_backend", None),
        getattr(args, "draft_context_parallel_size", 1),
    )
    args.attention_backend = execution.attention_backend
    args.draft_context_parallel_size = execution.context_parallel_size
    if execution.direct_cp_backend:
        cp_size = execution.context_parallel_size
        if world_size % cp_size != 0:
            raise PlanCompileError(
                f"world_size ({world_size}) must be divisible by draft CP size ({cp_size})"
            )
        dp_size = getattr(args, "dp_size", None) or (world_size // cp_size)
        if dp_size * cp_size != world_size:
            raise PlanCompileError(
                f"dp_size ({dp_size}) * draft CP size ({cp_size}) must equal "
                f"world_size ({world_size})"
            )
        args.dp_size = dp_size
        args.sp_size = 1
        args.per_dp_rank_batch_size = args.micro_batch_size
        sp_size = 1
    elif execution.attention_backend == "usp":
        sp_size = getattr(args, "sp_ulysses_size", 1) * getattr(args, "sp_ring_size", 1)
        if sp_size <= 0:
            raise PlanCompileError(f"USP requires positive sp_size, got {sp_size}")
        if world_size % sp_size != 0:
            raise PlanCompileError(
                f"world_size ({world_size}) must be divisible by USP sp_size ({sp_size})"
            )
        dp_size = getattr(args, "dp_size", None) or (world_size // sp_size)
        if dp_size * sp_size != world_size:
            raise PlanCompileError(
                f"dp_size ({dp_size}) * sp_size ({sp_size}) must equal world_size ({world_size})"
            )
        args.dp_size = dp_size
        args.sp_size = sp_size
        args.per_dp_rank_batch_size = 1
    else:
        dp_size = getattr(args, "dp_size", None) or world_size
        args.dp_size = dp_size
        sp_size = getattr(args, "sp_size", None)
        if sp_size is not None and sp_size != 1:
            raise PlanCompileError(f"Sequence parallel is not yet supported (got sp_size={sp_size})")
        sp_size = sp_size or 1
        args.sp_size = sp_size
        args.per_dp_rank_batch_size = args.micro_batch_size * sp_size

    accumulation_steps = getattr(args, "draft_accumulation_steps", 1)
    args.global_batch_size = args.per_dp_rank_batch_size * dp_size * accumulation_steps

    requested_budget = getattr(args, "source_work_budget", None)
    args.requested_source_work_budget = requested_budget
    args.source_work_alignment = None
    args.planned_dropped_source_work = 0
    args.source_work_tail_policy = None
    if requested_budget is None:
        return

    requested_budget = int(requested_budget)
    packing_enabled = bool(getattr(args, "dflash_packing", False))
    alignment = dp_size if packing_enabled else args.global_batch_size
    effective_budget = requested_budget - requested_budget % alignment
    minimum_executable_work = (
        dp_size * accumulation_steps if packing_enabled else args.global_batch_size
    )
    if effective_budget < minimum_executable_work:
        unit = "packed optimizer step" if packing_enabled else "optimizer batch"
        raise PlanCompileError(
            "source_work_budget cannot form one complete executable "
            f"{unit}: requested={requested_budget}, alignment={alignment}, "
            f"minimum={minimum_executable_work}"
        )
    args.source_work_budget = effective_budget
    args.source_work_alignment = alignment
    args.planned_dropped_source_work = requested_budget - effective_budget
    args.source_work_tail_policy = (
        "drop_last_complete_packed_optimizer_step"
        if packing_enabled
        else "drop_last_complete_optimizer_batch"
    )


def _validate_usp_args(args: argparse.Namespace) -> None:
    if getattr(args, "attention_backend", None) != "usp":
        return

    sp_size = getattr(args, "sp_size", None)
    if sp_size is None:
        sp_size = getattr(args, "sp_ulysses_size", 1) * getattr(args, "sp_ring_size", 1)
    if sp_size <= 1:
        raise PlanCompileError(f"USP requires sp_size > 1, got {sp_size}")
    if getattr(args, "inference_engine_type", "sgl") != "sgl":
        raise PlanCompileError("USP currently requires inference_engine_type=sgl")
    if getattr(args, "fsdp_strategy", "REPLICATE").upper() != "REPLICATE":
        raise PlanCompileError("USP currently requires fsdp_strategy=REPLICATE")
    if getattr(args, "micro_batch_size", 1) != 1:
        raise PlanCompileError("USP currently requires micro_batch_size=1")


def _validate_scope(args: argparse.Namespace, draft: Mapping[str, Any]) -> None:
    if getattr(args, "train_backend", None) != "fsdp":
        raise PlanCompileError(
            f"The current training backend is fsdp; got {getattr(args, 'train_backend', None)!r}"
        )
    cluster_mode = getattr(args, "cluster_mode", "single_node")
    if cluster_mode not in {"single_node", "static_multi_node"}:
        raise PlanCompileError(f"Unsupported cluster.mode {cluster_mode!r}")
    configured_node_ips = set(getattr(args, "training_node_ips", None) or ()) | set(
        getattr(args, "inference_node_ips", None) or ()
    )
    multi_node_requested = (
        getattr(args, "training_num_nodes", 0) != 1
        or getattr(args, "sglang_nnodes", 0) != 1
        or len(configured_node_ips) > 1
    )
    if cluster_mode == "single_node" and multi_node_requested:
        raise PlanCompileError(
            "Multi-node execution requires cluster.mode=static_multi_node and the "
            "production cluster contract"
        )
    if getattr(args, "training_num_gpus_per_node", 0) <= 0:
        raise PlanCompileError("training_num_gpus_per_node must be positive")
    if getattr(args, "colocate", False):
        raise PlanCompileError("Colocated training and inference is unsupported")
    if getattr(args, "train_with_decode", False):
        raise PlanCompileError("training.train_with_decode is unsupported")
    if getattr(args, "sglang_enable_multimodal", False):
        raise PlanCompileError("Multimodal target inference is unsupported")
    if getattr(args, "sglang_pp_size", 1) != 1:
        raise PlanCompileError("The current SGLang target-feature path requires pp_size=1")
    if getattr(args, "last_hidden_states_prenorm", False):
        raise PlanCompileError(
            "SGLang target features are post-norm; inference.last_hidden_states_prenorm "
            "cannot be true"
        )

    total_inference_gpus = getattr(args, "inference_num_gpus", None)
    gpus_per_engine = getattr(args, "inference_num_gpus_per_engine", 0)
    offline = getattr(args, "inference_engine_type", None) == "offline"
    if not isinstance(total_inference_gpus, int) or (total_inference_gpus != 0 if offline else total_inference_gpus <= 0):
        raise PlanCompileError("inference.inference_num_gpus must be a positive integer")
    if not isinstance(gpus_per_engine, int) or gpus_per_engine <= 0:
        raise PlanCompileError("inference.inference_num_gpus_per_engine must be positive")
    if total_inference_gpus % gpus_per_engine:
        raise PlanCompileError(
            "inference_num_gpus must be divisible by inference_num_gpus_per_engine; "
            f"got {total_inference_gpus} and {gpus_per_engine}"
        )

    if cluster_mode == "static_multi_node":
        if getattr(args, "placement_strategy", None) != "custom":
            raise PlanCompileError(
                "static_multi_node requires training.placement_strategy=custom"
            )
        if getattr(args, "colocate", False):
            raise PlanCompileError("static_multi_node does not yet support colocate=true")
        if getattr(args, "training_node_selectors", None) or getattr(
            args, "inference_node_selectors", None
        ):
            raise PlanCompileError(
                "static_multi_node requires explicit role node IPs, not label selectors"
            )
        training_node_ips = list(getattr(args, "training_node_ips", None) or ())
        inference_node_ips = list(getattr(args, "inference_node_ips", None) or ())
        if len(training_node_ips) != getattr(args, "training_num_nodes", 0):
            raise PlanCompileError(
                "training_node_ips must contain exactly training_num_nodes entries"
            )
        inference_gpus_per_node = getattr(args, "inference_num_gpus_per_node", 0)
        if not isinstance(inference_gpus_per_node, int) or inference_gpus_per_node <= 0:
            raise PlanCompileError("inference_num_gpus_per_node must be positive")
        expected_inference_nodes = math.ceil(
            total_inference_gpus / inference_gpus_per_node
        )
        if len(inference_node_ips) != expected_inference_nodes:
            raise PlanCompileError(
                "inference_node_ips must match the planned inference node count: "
                f"expected {expected_inference_nodes}, got {len(inference_node_ips)}"
            )
        remaining_inference_gpus = total_inference_gpus
        for node_ip in inference_node_ips:
            allocated = min(inference_gpus_per_node, remaining_inference_gpus)
            if allocated % gpus_per_engine:
                raise PlanCompileError(
                    "every node-local Target allocation must contain complete TP groups: "
                    f"node={node_ip!r}, allocated_gpus={allocated}, "
                    f"gpus_per_engine={gpus_per_engine}"
                )
            remaining_inference_gpus -= allocated
        if len(set(training_node_ips) | set(inference_node_ips)) < 2:
            raise PlanCompileError(
                "static_multi_node requires at least two distinct node IPs"
            )
        if getattr(args, "sglang_nnodes", 1) != 1:
            raise PlanCompileError(
                "static_multi_node first release keeps Target TP groups node-local; "
                "sglang.nnodes must be 1"
            )
        if gpus_per_engine > inference_gpus_per_node:
            raise PlanCompileError(
                "node-local Target TP cannot exceed inference_num_gpus_per_node"
            )
        if getattr(args, "cluster_storage_mode", "shared") == "shared" and not getattr(args, "cluster_shared_root", None):
            raise PlanCompileError("static_multi_node requires cluster.shared_root")

    expected_tp = getattr(args, "sglang_nnodes", 1) * gpus_per_engine
    if not offline and getattr(args, "sglang_tp_size", None) != expected_tp:
        raise PlanCompileError(
            "inference.sglang.tp_size must equal sglang.nnodes * "
            f"inference_num_gpus_per_engine ({expected_tp})"
        )

    extra_args = _plain(getattr(args, "sglang_extra_args", {})) or {}
    protected = _PROTECTED_SGLANG_EXTRA_ARGS.intersection(extra_args)
    if protected:
        raise PlanCompileError(
            "inference.sglang.extra_args cannot override ASpec-managed keys: "
            f"{sorted(protected)}"
        )

    protocol = str(getattr(args, "mooncake_protocol", "tcp")).lower()
    data_plane = str(getattr(args, "mooncake_data_plane", "store")).lower()
    gpu_direct = bool(getattr(args, "mooncake_enable_gpu_direct", False))
    allow_host_fallback = bool(
        getattr(args, "mooncake_allow_host_fallback", True)
    )
    require_nvlink = bool(getattr(args, "mooncake_require_nvlink", False))
    feature_affinity = str(
        getattr(args, "mooncake_feature_affinity", "any")
    ).lower()
    if feature_affinity not in {"any", "node_local_required"}:
        raise PlanCompileError(
            "mooncake.feature_affinity must be 'any' or 'node_local_required'"
        )
    node_local_features = feature_affinity == "node_local_required"
    if node_local_features and cluster_mode != "static_multi_node":
        raise PlanCompileError(
            "mooncake.feature_affinity=node_local_required is a static multi-node "
            "placement contract"
        )
    if protocol not in {"tcp", "rdma", "nvlink_intra"}:
        raise PlanCompileError(
            f"Unsupported Mooncake protocol {protocol!r}"
        )
    if data_plane not in {"store", "transfer_engine"}:
        raise PlanCompileError(
            f"Unsupported Mooncake data plane {data_plane!r}"
        )
    if data_plane == "transfer_engine" and not gpu_direct:
        raise PlanCompileError(
            "mooncake.data_plane=transfer_engine requires enable_gpu_direct=true"
        )
    if data_plane == "transfer_engine" and allow_host_fallback:
        raise PlanCompileError(
            "mooncake.data_plane=transfer_engine requires allow_host_fallback=false"
        )
    if gpu_direct and data_plane != "transfer_engine":
        raise PlanCompileError(
            "ASpec currently admits GPU-direct transport only through the "
            "generation-checked transfer_engine data plane"
        )
    if protocol == "nvlink_intra" and data_plane != "transfer_engine":
        raise PlanCompileError(
            "mooncake.protocol=nvlink_intra requires data_plane=transfer_engine"
        )
    if require_nvlink and (
        protocol != "nvlink_intra" or data_plane != "transfer_engine"
    ):
        raise PlanCompileError(
            "mooncake.require_nvlink=true requires protocol=nvlink_intra and "
            "data_plane=transfer_engine"
        )
    if cluster_mode == "static_multi_node":
        if node_local_features:
            if set(training_node_ips) != set(inference_node_ips):
                raise PlanCompileError(
                    "node_local_required feature affinity requires training_node_ips "
                    "and inference_node_ips to select the same nodes"
                )
            cp_size = int(getattr(args, "draft_context_parallel_size", 1))
            if getattr(args, "training_num_gpus_per_node", 0) % cp_size:
                raise PlanCompileError(
                    "node_local_required feature affinity requires every Draft CP "
                    "group to remain within one node"
                )
            ready_watermark = int(getattr(args, "max_sample_pool_size", 0))
            if 0 < ready_watermark < len(training_node_ips):
                raise PlanCompileError(
                    "node_local_required bounded ready pool needs at least one "
                    "credit per locality domain"
                )
        registered_cuda_transport = (
            data_plane == "transfer_engine"
            and protocol in {"tcp", "rdma"}
            and gpu_direct
            and not allow_host_fallback
        )
        host_staged_tcp = (
            data_plane == "store"
            and protocol == "tcp"
            and not gpu_direct
            and not allow_host_fallback
        )
        node_local_nvlink = (
            node_local_features
            and data_plane == "transfer_engine"
            and protocol == "nvlink_intra"
            and gpu_direct
            and not allow_host_fallback
            and require_nvlink
        )
        if node_local_features and not node_local_nvlink:
            raise PlanCompileError(
                "node_local_required feature affinity requires strict "
                "transfer_engine/nvlink_intra with require_nvlink=true"
            )
        if not (registered_cuda_transport or host_staged_tcp or node_local_nvlink):
            raise PlanCompileError(
                "static_multi_node requires either a generation-checked TCP/RDMA "
                "TransferEngine, the explicit host-staged TCP Store path, or strict "
                "node-local-required NVLink; all require allow_host_fallback=false"
            )
        if require_nvlink and not node_local_nvlink:
            raise PlanCompileError(
                "static_multi_node may require intra-node NVLink only with "
                "feature_affinity=node_local_required"
            )
        if host_staged_tcp:
            pool_size = getattr(
                args, "mooncake_async_put_pool_size", args.inference_batch_size
            )
            if pool_size is not None and (type(pool_size) is not int or pool_size < 1):
                raise PlanCompileError(
                    "host-staged Store requires async_put_pool_size >= 1"
                )
        device_name = getattr(args, "mooncake_device_name", "")
        rdma_port = getattr(args, "mooncake_rdma_port", 1)
        rdma_gid_index = getattr(args, "mooncake_rdma_gid_index", None)
        rdma_routes = _plain(getattr(args, "mooncake_rdma_routes", None))
        if protocol == "rdma":
            try:
                resolve_rdma_routes(
                    sorted(configured_node_ips),
                    device_name=device_name,
                    port=rdma_port,
                    gid_index=rdma_gid_index,
                    overrides=rdma_routes,
                )
            except ValueError as exc:
                raise PlanCompileError(str(exc)) from exc
        if protocol == "tcp" and (device_name or rdma_routes):
            raise PlanCompileError(
                "static_multi_node TCP requires empty RDMA route configuration"
            )

    direct_target_output = bool(
        getattr(args, "target_prefill_memory_optimized", False)
    )
    if direct_target_output:
        if getattr(args, "inference_engine_type", "sgl") != "sgl":
            raise PlanCompileError(
                "target_prefill_memory_optimized currently requires SGLang"
            )
        if getattr(args, "train_with_decode", False):
            raise PlanCompileError(
                "target_prefill_memory_optimized requires prefill-only Target execution"
            )
        if gpus_per_engine != 1 or getattr(args, "sglang_tp_size", 1) != 1:
            raise PlanCompileError(
                "target_prefill_memory_optimized currently requires Target TP1"
            )
        if getattr(args, "sglang_pp_size", 1) != 1:
            raise PlanCompileError(
                "target_prefill_memory_optimized currently requires Target PP1"
            )
        if getattr(args, "attention_backend", None) == "usp":
            raise PlanCompileError(
                "target_prefill_memory_optimized is not compatible with the USP "
                "feature layout"
            )
        transport_valid = (
            data_plane == "transfer_engine"
            and gpu_direct
            and not allow_host_fallback
            and (
                (
                    cluster_mode == "static_multi_node"
                    and protocol in {"tcp", "rdma"}
                    and not require_nvlink
                )
                or (
                    cluster_mode == "static_multi_node"
                    and node_local_features
                    and protocol == "nvlink_intra"
                    and require_nvlink
                )
                or (
                    cluster_mode != "static_multi_node"
                    and protocol == "nvlink_intra"
                    and require_nvlink
                )
            )
        )
        if not transport_valid:
            raise PlanCompileError(
                "target_prefill_memory_optimized requires a strict generation-checked "
                "TransferEngine data plane: NVLink for single-node or TCP/RDMA for "
                "static multi-node, without host fallback"
            )
        if not getattr(args, "streaming_dispatch_with_training", False):
            raise PlanCompileError(
                "target_prefill_memory_optimized requires streaming dispatch"
            )
    # Native backend capture owns the same finite TransferEngine source rings.
    # Admission must cover it as well as the direct-prefill specialization.
    if direct_target_output or (data_plane == "transfer_engine" and gpu_direct and not allow_host_fallback):
        slots_per_sample = (
            int(getattr(args, "source_context_parallel_size", args.draft_context_parallel_size))
            if is_direct_cp_backend(getattr(args, "attention_backend", None))
            else 1
        )
        source_slots = getattr(args, "mooncake_source_gpu_ring_slots", None)
        if source_slots is None:
            source_slots = getattr(args, "mooncake_gpu_ring_slots", 2)
        if node_local_features:
            engine_count = int(getattr(args, "inference_num_gpus_per_node", 0)) // gpus_per_engine
            dispatch_batch_size = (
                int(getattr(args, "training_num_gpus_per_node", 0))
                // int(getattr(args, "draft_context_parallel_size", 1))
                * args.per_dp_rank_batch_size
            )
            capacity_scope = "per node"
        else:
            engine_count = total_inference_gpus // gpus_per_engine
            dispatch_batch_size = args.per_dp_rank_batch_size * args.dp_size
            capacity_scope = "across the run"
        resident_sample_capacity = direct_output_sample_capacity(
            engine_count=engine_count,
            source_slots=int(source_slots),
            slots_per_sample=slots_per_sample,
        )
        packing_supply_threshold = 0
        if getattr(args, "dflash_packing", False):
            packing_supply_threshold = int(
                getattr(args, "packing_min_samples_per_dispatch", 0) or 0
            )
            max_pool_size = int(getattr(args, "max_sample_pool_size", 0) or 0)
            if packing_supply_threshold > 0 and max_pool_size > 0:
                # The controller forces progress at the ready watermark even when a
                # larger explicit packing threshold was requested.
                packing_supply_threshold = min(
                    packing_supply_threshold, max_pool_size
                )
        if resident_sample_capacity < dispatch_batch_size:
            raise PlanCompileError(
                "TransferEngine source rings cannot retain one "
                f"complete Draft dispatch {capacity_scope}: "
                f"capacity={resident_sample_capacity}, "
                f"dispatch_batch_size={dispatch_batch_size}, engines={engine_count}, "
                f"source_gpu_ring_slots={source_slots}, "
                f"slots_per_sample={slots_per_sample}"
            )
        if resident_sample_capacity < packing_supply_threshold:
            raise PlanCompileError(
                "TransferEngine source rings cannot retain enough "
                f"samples to satisfy the packing supply threshold {capacity_scope}: "
                f"capacity={resident_sample_capacity}, "
                f"required_samples={packing_supply_threshold}, "
                f"packing_min_samples_per_dispatch={packing_supply_threshold}, "
                f"engines={engine_count}, source_gpu_ring_slots={source_slots}, "
                f"slots_per_sample={slots_per_sample}"
            )
    if getattr(args, "defer_tokenization", False):
        raise PlanCompileError(f"{draft['family']} does not support defer_tokenization=True")

    target_hidden_size = draft.get("target_hidden_size")
    if type(target_hidden_size) is not int or target_hidden_size <= 0:
        raise PlanCompileError(
            "draft target_hidden_size must be a positive integer; "
            f"got {target_hidden_size!r}"
        )
    transport_hidden_dim = getattr(args, "mooncake_hidden_dim", 4096)
    if type(transport_hidden_dim) is not int or transport_hidden_dim <= 0:
        raise PlanCompileError(
            "mooncake.hidden_dim must be a positive integer; "
            f"got {transport_hidden_dim!r}"
        )
    if transport_hidden_dim != target_hidden_size:
        raise PlanCompileError(
            "mooncake.hidden_dim must match draft target_hidden_size so transport "
            "capacity covers the target feature payload; "
            f"got {transport_hidden_dim} and {target_hidden_size}"
        )

    configured_aux_layers = getattr(args, "aux_hidden_states_layers", None)
    if configured_aux_layers is not None and list(configured_aux_layers) != list(
        draft["target_layer_ids"]
    ):
        raise PlanCompileError(
            "inference.aux_hidden_states_layers must match draft target_layer_ids; "
            f"got {list(configured_aux_layers)!r} and {draft['target_layer_ids']!r}"
        )

    expected_target_layers = len(draft["target_layer_ids"])
    declared_target_layers = draft.get("num_target_layers")
    if declared_target_layers is not None and declared_target_layers != expected_target_layers:
        raise PlanCompileError(
            "draft num_target_layers must match the number of target_layer_ids; "
            f"got {declared_target_layers} and {expected_target_layers}"
        )
    training_target_layers = (
        expected_target_layers if draft["family"] in {"eagle3", "eagle4", "mtp"} else getattr(
            args,
            "dspark_num_target_layers" if draft["family"] == "dspark" else "dflash_num_target_layers",
        )
    )
    if training_target_layers != expected_target_layers:
        raise PlanCompileError(
            f"{draft['family']} training num_target_layers must match target_layer_ids; "
            f"got {training_target_layers} and {expected_target_layers}"
        )

    block_size = getattr(args, "dflash_block_size", 16)
    min_loss_tokens = getattr(args, "min_loss_tokens", 0)
    if draft["family"] not in {"eagle3", "eagle4", "mtp"} and min_loss_tokens < 2 * block_size:
        raise PlanCompileError(
            f"{draft['family']} requires dataset.min_loss_tokens >= "
            f"2 * training.dflash_block_size ({min_loss_tokens} < {2 * block_size})"
        )
    indexer_warmup = (draft["attention"].get("dsa") or {}).get("training_stage") == "indexer_warmup"
    needs_teacher = not indexer_warmup and (
        draft["family"] in {"dspark", "eagle3"}
        or (draft["family"] == "eagle4" and draft["eagle4"]["stage"] == "draft")
    )
    if needs_teacher and not getattr(args, "store_last_hidden_states", False):
        raise PlanCompileError(
            f"{'DSpark' if draft['family'] == 'dspark' else 'EAGLE3'} requires "
            "inference.store_last_hidden_states=true"
        )


def _runtime_values(config: DictConfig, draft: Mapping[str, Any], *, source_context_parallel_size=None) -> dict[str, Any]:
    args = config_to_flat_args(config)
    args.draft_algorithm = draft["family"]
    if args.draft_algorithm == "mtp":
        from aspec.models.mtp import mtp_step_weights

        args.ttt_length = args.mtp_steps  # common recurrent trainer's metric slots
        args.ploss_weights = mtp_step_weights(args.mtp_steps, args.mtp_step_weight_beta)
        args.freeze_lm_head = True
    args.source_context_parallel_size = (args.draft_context_parallel_size
                                        if source_context_parallel_size is None else source_context_parallel_size)
    if args.draft_algorithm in {"eagle3", "eagle4"} and args.ploss_weights is None:
        args.ploss_weights = [0.8 ** i for i in range(args.ttt_length)]
    from aspec.config.draft_attention import draft_context_shard_alignment

    args.draft_context_shard_alignment = draft_context_shard_alignment(
        draft["attention"]
    )
    # ``mooncake`` remains an open mapping at the compatibility boundary, so
    # omitted fields are not materialized by OmegaConf's structured schema.
    # The Plan resolves the RDMA port to 1; retain that same effective value in
    # the runtime adapter so terminal Evidence validates against the Plan.
    if not hasattr(args, "mooncake_rdma_port"):
        args.mooncake_rdma_port = _select(config, "mooncake.rdma_port", 1)
    args.rank = 0
    args.world_size = args.training_num_nodes * args.training_num_gpus_per_node
    defaults = {"colocate": False, "dp_size": None, "save_debug_train_data": None}
    for key, value in defaults.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
    _resolve_batch_size(args)
    _validate_usp_args(args)
    _validate_scope(args, draft)
    if not getattr(args, "aux_hidden_states_layers", None):
        args.aux_hidden_states_layers = list(draft["target_layer_ids"])
    return copy.deepcopy(vars(args))


def _workload(
    config: DictConfig,
    draft: Mapping[str, Any],
    target_family: str,
    runtime: Mapping[str, Any],
) -> WorkloadContract:
    family = draft["family"]
    anchor_scope = _select(config, "training.anchor_sampling_scope", "row")
    if anchor_scope != "row" and family not in {"dspark", "dflash", "dflash2"}:
        raise PlanCompileError("anchor_sampling_scope=sample requires DSpark or DFlash")
    if family == "mtp":
        from aspec.models.mtp import mtp_step_weights

        steps = _select(config, "training.mtp_steps")
        beta = _select(config, "training.mtp_step_weight_beta")
        objective = {
            "family": "mtp", "loss": "teacher_forced_native_mtp_ce", "steps": steps,
            "step_weight_beta": beta, "step_weights": mtp_step_weights(steps, beta),
            "frozen_shared": ["embedding", "lm_head"],
            "verifier_feature_stage": "post_norm",
        }
    elif family in {"eagle3", "eagle4"}:
        objective = {
            "family": family,
            "ttt_length": _select(config, "training.ttt_length"),
            "ploss_weights": (
                _plain(_select(config, "training.ploss_weights"))
                if _select(config, "training.ploss_weights") is not None
                else [0.8 ** i for i in range(_select(config, "training.ttt_length"))]
            ),
            "loss_type": _select(config, "training.loss_type"),
            "lk_eta": _select(config, "training.lk_eta"),
            "gradient_checkpointing": _select(config, "training.gradient_checkpointing"),
        }
    elif family == "dspark":
        objective = {
            "family": family,
            "block_size": _select(config, "training.dflash_block_size"),
            "num_anchors": _select(config, "training.dspark_num_anchors"),
            "num_target_layers": _select(config, "training.dspark_num_target_layers"),
            "loss_decay_gamma": _select(config, "training.dspark_loss_decay_gamma"),
            "ce_loss_alpha": _select(config, "training.dspark_ce_loss_alpha"),
            "l1_loss_alpha": _select(config, "training.dspark_l1_loss_alpha"),
            "confidence_head_alpha": _select(
                config, "training.dspark_confidence_head_alpha"
            ),
        }
        for side in ("target", "draft"):
            for name, default in (("temperature", 1.0), ("top_p", 1.0), ("top_k", -1)):
                objective[f"l1_{side}_{name}"] = _select(config, f"training.dspark_l1_{side}_{name}", default)
    else:
        objective = {
            "family": family,
            "block_size": _select(config, "training.dflash_block_size"),
            "num_anchors": _select(config, "training.dflash_num_anchors"),
            "num_target_layers": _select(config, "training.dflash_num_target_layers"),
            "loss_objective": _select(config, "training.dflash_loss_objective"),
            "dpace_alpha": _select(config, "training.dflash_dpace_alpha"),
            "loss_decay_gamma": _select(config, "training.dflash_loss_decay_gamma"),
            "ce_loss_alpha": _select(config, "training.dflash_ce_loss_alpha"),
            "l1_loss_alpha": _select(config, "training.dflash_l1_loss_alpha"),
        }
        if family == "dflash2":
            objective["selector_loss_alpha"] = _select(config, "training.dflash2_selector_loss_alpha")
            objective["selector_objective"] = "experimental_teacher_forced_topk_ce"
    if family == "eagle4":
        objective["eagle4"] = draft["eagle4"]
        objective["frozen_parts"] = (["draft_body", "embedding", "lm_head"]
                                     if draft["eagle4"]["stage"] == "candidate_head" else
                                     ["embedding", "candidate_head"] + (["lm_head"] if _select(config, "model.freeze_lm_head", False) else []))
    if family in {"dspark", "dflash", "dflash2"}:
        objective["anchor_sampling_scope"] = anchor_scope
    objective.update(
        {
            "learning_rate": _select(config, "training.learning_rate"),
            "max_grad_norm": _select(config, "training.max_grad_norm"),
            "weight_decay": _select(config, "training.weight_decay"),
            "max_sequence_length": _select(config, "training.max_seq_length"),
            "seed": _select(config, "training.seed"),
            "deterministic_anchor_sampling": bool(
                _select(config, "training.deterministic_anchor_sampling")
            ),
        }
    )

    store_last = bool(_select(config, "inference.store_last_hidden_states"))
    try:
        features = build_target_feature_contract(
            drafter_family=family,
            target_hidden_size=draft["target_hidden_size"],
            target_num_hidden_layers=draft["target_num_hidden_layers"],
            aux_hidden_state_layers=list(draft["target_layer_ids"]),
            store_last_hidden_states=store_last,
        )
    except FeatureContractError as exc:
        raise PlanCompileError(str(exc)) from exc
    return WorkloadContract(
        models={
            "target": {
                "family": target_family,
                "path": _select(config, "model.target_model_path"),
                "native_context_length": _select(
                    config, "model.native_context_length"
                ),
            },
            "draft": {
                "family": family,
                "algorithm": family,
                "architecture": draft["architecture"],
                "config_path": draft["path"],
                "config_sha256": draft["sha256"],
                "initial_checkpoint": _select(config, "model.initial_draft_model_path"),
                "structure": draft["structure"],
                "attention": draft["attention"],
            },
        },
        data={
            "train_source": _select(config, "dataset.train_data_path"),
            "eval_source": _select(config, "dataset.eval_data_path"),
            "prompt_key": _select(config, "dataset.prompt_key"),
            "renderer": _select(config, "dataset.renderer"),
            "chat_template": _select(config, "dataset.chat_template"),
            "last_turn_loss_only": _select(config, "dataset.last_turn_loss_only"),
            "min_loss_tokens": _select(config, "dataset.min_loss_tokens"),
            "scheduling": {
                "packing": bool(_select(config, "training.dflash_packing")),
                "strategy": (
                    "oldest_seed_anchor_density_best_fit"
                    if _select(config, "training.dflash_packing")
                    else "length_grouped_padding"
                ),
                "packing_min_samples_per_dispatch": _select(
                    config, "training.packing_min_samples_per_dispatch"
                ),
            },
        },
        objective=objective,
        features=features,
        duration={
            "num_train_steps": _select(config, "training.num_train_steps"),
            "num_epochs": _select(config, "training.num_epochs"),
            "source_work": {
                "requested": runtime["requested_source_work_budget"],
                "scheduled": runtime["source_work_budget"],
                "planned_dropped": runtime["planned_dropped_source_work"],
                "alignment": runtime["source_work_alignment"],
                "tail_policy": runtime["source_work_tail_policy"],
            },
        },
    )


def _policy(
    config: DictConfig,
    runtime: Mapping[str, Any],
    context_contract: Mapping[str, Any],
) -> ExecutionPolicy:
    cluster_mode = _select(config, "cluster.mode", "single_node")
    data_plane = _select(config, "mooncake.data_plane", "store")
    transport_protocol = _select(config, "mooncake.protocol", "tcp")
    feature_affinity = _select(config, "mooncake.feature_affinity", "any")
    eagle3_loss_checkpointing = _select(config, "training.eagle3_loss_gradient_checkpointing")
    if eagle3_loss_checkpointing is None:
        eagle3_loss_checkpointing = _select(config, "training.gradient_checkpointing", False)
    gpu_direct_enabled = bool(
        _select(config, "mooncake.enable_gpu_direct", False)
    )
    if data_plane == "store":
        transport_stage = f"host_staged_{transport_protocol}"
        staging_mode = "pinned_host_buffers"
    elif transport_protocol == "rdma":
        transport_stage = "rdma_gpu_direct"
        staging_mode = "registered_cuda_ring"
    elif transport_protocol == "nvlink_intra":
        transport_stage = (
            "node_local_nvlink_registered_cuda_ring"
            if cluster_mode == "static_multi_node"
            and feature_affinity == "node_local_required"
            else "nvlink_registered_cuda_ring"
        )
        staging_mode = "registered_cuda_ring"
    else:
        transport_stage = "generation_checked_tcp"
        staging_mode = "registered_cuda_ring"
    host_capacity = MooncakeConfig(
        max_seq_len=_select(
            config, "mooncake.max_seq_len", _select(config, "training.max_seq_length")
        ),
        hidden_dim=_select(config, "mooncake.hidden_dim", 4096),
        num_aux_layers=len(runtime["aux_hidden_states_layers"]),
        host_buffer_size=_select(config, "mooncake.host_buffer_size"),
        async_put_pool_size=_select(
            config,
            "mooncake.async_put_pool_size",
            _select(config, "inference.inference_batch_size"),
        ),
    )
    cluster_nodes: list[dict[str, Any]] = []
    locality_domains: list[dict[str, Any]] = []
    ready_watermark_by_domain: dict[str, int] = {}
    resolved_rdma_routes: dict[str, dict[str, Any]] = {}
    if cluster_mode == "static_multi_node":
        allocations: dict[str, dict[str, Any]] = {}
        training_node_ips = list(_select(config, "training.training_node_ips") or ())
        for node_ip in training_node_ips:
            allocation = allocations.setdefault(
                node_ip,
                {"address": node_ip, "roles": [], "training_gpus": 0, "inference_gpus": 0},
            )
            allocation["roles"].append("training")
            allocation["training_gpus"] += int(
                _select(config, "training.training_num_gpus_per_node")
            )

        remaining_inference_gpus = int(
            _select(config, "inference.inference_num_gpus")
        )
        inference_gpus_per_node = int(
            _select(config, "inference.inference_num_gpus_per_node")
        )
        for node_ip in list(_select(config, "inference.inference_node_ips") or ()):
            allocation = allocations.setdefault(
                node_ip,
                {"address": node_ip, "roles": [], "training_gpus": 0, "inference_gpus": 0},
            )
            if "inference" not in allocation["roles"]:
                allocation["roles"].append("inference")
            gpus_on_node = min(inference_gpus_per_node, remaining_inference_gpus)
            allocation["inference_gpus"] += gpus_on_node
            remaining_inference_gpus -= gpus_on_node
        cluster_nodes = [
            {**allocation, "roles": sorted(allocation["roles"])}
            for allocation in allocations.values()
        ]
        training_world_ranks: dict[str, list[int]] = {}
        next_training_rank = 0
        for node_ip in training_node_ips:
            count = int(_select(config, "training.training_num_gpus_per_node"))
            training_world_ranks[node_ip] = list(
                range(next_training_rank, next_training_rank + count)
            )
            next_training_rank += count

        parallel_group_size = runtime["world_size"] // runtime["dp_size"]
        draft_dp_ranks: dict[str, list[int]] = {}
        next_dp_rank = 0
        for node_ip in training_node_ips:
            count = (
                int(_select(config, "training.training_num_gpus_per_node"))
                // parallel_group_size
            )
            draft_dp_ranks[node_ip] = list(range(next_dp_rank, next_dp_rank + count))
            next_dp_rank += count

        target_engine_ranks: dict[str, list[int]] = {}
        next_engine_rank = 0
        gpus_per_engine = int(
            _select(config, "inference.inference_num_gpus_per_engine")
        )
        for node_ip in list(_select(config, "inference.inference_node_ips") or ()):
            engine_count = allocations[node_ip]["inference_gpus"] // gpus_per_engine
            target_engine_ranks[node_ip] = list(
                range(next_engine_rank, next_engine_rank + engine_count)
            )
            next_engine_rank += engine_count

        for ordinal, allocation in enumerate(cluster_nodes):
            node_ip = str(allocation["address"])
            dp_ranks = draft_dp_ranks.get(node_ip, [])
            locality_domains.append(
                {
                    "domain_id": f"physical-node-{ordinal}",
                    "kind": "physical_node",
                    "node_address": node_ip,
                    "training_world_ranks": training_world_ranks.get(node_ip, []),
                    "draft_dp_ranks": dp_ranks,
                    "draft_parallel_group_size": parallel_group_size,
                    "target_engine_ranks": target_engine_ranks.get(node_ip, []),
                    "target_tensor_parallel_size": gpus_per_engine,
                    "dispatch_samples_per_microstep": (
                        len(dp_ranks) * int(runtime["per_dp_rank_batch_size"])
                    ),
                }
            )
        if feature_affinity == "node_local_required":
            by_node = allocate_node_ready_watermarks(
                int(_select(config, "inference.max_sample_pool_size")),
                {
                    str(domain["node_address"]): int(
                        domain["dispatch_samples_per_microstep"]
                    )
                    for domain in locality_domains
                },
            )
            ready_watermark_by_domain = {
                str(domain["domain_id"]): by_node[str(domain["node_address"])]
                for domain in locality_domains
            }
            for domain in locality_domains:
                domain["ready_watermark"] = ready_watermark_by_domain[
                    str(domain["domain_id"])
                ]
        if _select(config, "mooncake.protocol", "tcp") == "rdma":
            resolved_rdma_routes = resolve_rdma_routes(
                sorted(allocations),
                device_name=_select(config, "mooncake.device_name", ""),
                port=_select(config, "mooncake.rdma_port", 1),
                gid_index=_select(config, "mooncake.rdma_gid_index", None),
                overrides=_select(config, "mooncake.rdma_routes", None),
            )

    return ExecutionPolicy(
        backends={
            "target_features": "offline_replay" if runtime["inference_engine_type"] == "offline" else "sglang",
            "training": _select(config, "training.train_backend"),
            "transport": "mooncake",
        },
        resources={
            "cluster": {
                "mode": cluster_mode,
                "membership": (
                    "static_explicit" if cluster_mode == "static_multi_node" else "local"
                ),
                "nodes": cluster_nodes,
                "shared_root": _select(config, "cluster.shared_root"),
                "storage_mode": _select(config, "cluster.storage_mode", "shared"),
                "homogeneous_gpus_required": bool(
                    _select(config, "cluster.require_homogeneous_gpus", True)
                ),
                "failure_policy": (
                    "fail_stop_attempt_checkpoint_restart"
                    if cluster_mode == "static_multi_node"
                    else "local_attempt"
                ),
            },
            "training": {
                "nodes": _select(config, "training.training_num_nodes"),
                "gpus_per_node": _select(config, "training.training_num_gpus_per_node"),
                "world_size": runtime["world_size"],
            },
            "inference": {
                "nodes": (
                    len(_select(config, "inference.inference_node_ips") or ())
                    if cluster_mode == "static_multi_node"
                    else 1
                ),
                "gpus": _select(config, "inference.inference_num_gpus"),
                "gpus_per_node": (0 if runtime["inference_engine_type"] == "offline"
                                  else _select(config, "inference.inference_num_gpus_per_node")),
                "gpus_per_engine": (0 if runtime["inference_engine_type"] == "offline"
                                    else _select(config, "inference.inference_num_gpus_per_engine")),
                "engines": (runtime["offline_num_engines"] if runtime["inference_engine_type"] == "offline"
                            else runtime["inference_num_gpus"] // runtime["inference_num_gpus_per_engine"]),
                "target_group_locality": "node_local",
            },
            "locality_domains": locality_domains,
        },
        parallelism={
            "training": {
                "data_parallel": runtime["dp_size"],
                "sequence_parallel": runtime["sp_size"],
                "context_parallel": _select(
                    config, "training.draft_context_parallel_size", 1
                ),
                "attention_backend": _select(config, "training.attention_backend"),
                "fsdp_strategy": _select(config, "training.fsdp_strategy"),
            },
            "inference": {
                "tensor_parallel": _select(config, "inference.sglang.tp_size"),
                "pipeline_parallel": _select(config, "inference.sglang.pp_size"),
                "source_context_parallel_size": runtime["source_context_parallel_size"],
            },
        },
        pipeline={
            "request_batch_size": _select(config, "inference.inference_batch_size"),
            "train_micro_batch_size": _select(config, "training.micro_batch_size"),
            "accumulation_steps": _select(config, "training.draft_accumulation_steps"),
            "global_batch_size": runtime["global_batch_size"],
            "optimizer_step_timeout_seconds": _select(
                config, "training.optimizer_step_timeout_seconds"
            ),
            "max_concurrent_batches": _select(
                config, "training.max_concurrent_batches"
            ),
            "prefetch_depth": _select(config, "training.prefetch_depth"),
            "streaming_dispatch_with_training": bool(
                _select(config, "training.streaming_dispatch_with_training", False)
            ),
            "pipeline_dispatch_with_training": bool(
                _select(config, "training.pipeline_dispatch_with_training", False)
            ),
            "packed_rows": bool(_select(config, "training.dflash_packing")),
            "packing_min_samples_per_dispatch": _select(
                config, "training.packing_min_samples_per_dispatch"
            ),
            "target_feature_output": (
                "offline_replay" if runtime["inference_engine_type"] == "offline" else "direct_to_transport"
                if _select(
                    config,
                    "inference.target_prefill_memory_optimized",
                    False,
                )
                else "buffered_then_transport"
            ),
            "feature_affinity": feature_affinity,
            "eagle3_target_cache": _select(
                config, "training.eagle3_target_cache", "lazy"
            ),
            "eagle3_loss_gradient_checkpointing": eagle3_loss_checkpointing,
        },
        capacity={
            "device_memory": {
                "sglang_static_fraction": (None if runtime["inference_engine_type"] == "offline"
                                           else _select(config, "inference.sglang.mem_fraction_static"))
            },
            "backend_tokens": {
                **copy.deepcopy(dict(context_contract)),
                "profiled_max_total_tokens": None,
                "state": "not_applicable_offline" if runtime["inference_engine_type"] == "offline" else "pending_backend_probe",
            },
            "transfer_storage": {
                "data_plane": data_plane,
                "protocol": transport_protocol,
                "feature_affinity": feature_affinity,
                "transport_stage": transport_stage,
                "staging_mode": staging_mode,
                "registered_cuda_buffers_required": (
                    data_plane == "transfer_engine" and gpu_direct_enabled
                ),
                "network_gpu_direct_required": (
                    transport_protocol == "rdma" and gpu_direct_enabled
                ),
                "host_staging_selected": data_plane == "store",
                "host_buffer_size": (
                    host_capacity.host_buffer_size if data_plane == "store" else 0
                ),
                "async_put_pool_size": (
                    host_capacity.async_put_pool_size if data_plane == "store" else 0
                ),
                "producer_pinned_host_bytes_per_endpoint": (
                    host_capacity.host_buffer_size * host_capacity.async_put_pool_size
                    if data_plane == "store"
                    else 0
                ),
                "device_name": _select(config, "mooncake.device_name", ""),
                "rdma_port": _select(config, "mooncake.rdma_port", 1),
                "rdma_gid_index": _select(
                    config, "mooncake.rdma_gid_index", None
                ),
                "resolved_rdma_routes": resolved_rdma_routes,
                "gpu_buffer_size": _select(
                    config, "mooncake.gpu_buffer_size", None
                ),
                "gpu_ring_slots": _select(config, "mooncake.gpu_ring_slots", 2),
                "source_gpu_ring_slots": _select(
                    config, "mooncake.source_gpu_ring_slots", None
                ),
                "destination_gpu_ring_slots": _select(
                    config, "mooncake.destination_gpu_ring_slots", None
                ),
                "host_fallback_allowed": bool(
                    _select(config, "mooncake.allow_host_fallback", True)
                ),
                "global_segment_size": _select(config, "mooncake.global_segment_size"),
                "local_buffer_size": _select(config, "mooncake.local_buffer_size"),
                "transfer_timeout_seconds": _select(
                    config, "mooncake.transfer_timeout_seconds", 300
                ),
                "tcp_progress_timeout_seconds": _select(
                    config, "mooncake.tcp_progress_timeout_seconds", 300
                ),
            },
            "host_orchestrator": {
                "max_sample_pool_size": _select(
                    config, "inference.max_sample_pool_size"
                ),
                "sample_pool_semantics": "ready_watermark",
                "ready_watermark_scope": (
                    "per_locality_domain"
                    if feature_affinity == "node_local_required"
                    else "global"
                ),
                "ready_watermark_by_domain": ready_watermark_by_domain,
                "inflight_sample_limit": (
                    _select(config, "inference.inference_batch_size")
                    * _select(config, "training.max_concurrent_batches")
                    * (runtime["offline_num_engines"] if runtime["inference_engine_type"] == "offline"
                       else runtime["inference_num_gpus"] // runtime["inference_num_gpus_per_engine"])
                ),
                "inference_buffer_threshold": _select(
                    config, "inference.inference_buffer_threshold"
                ),
            },
        },
        measurement={
            "performance": {
                "required": bool(
                    _select(config, "debug.performance_measurement_required")
                ),
                "warmup_steps": _select(config, "debug.performance_warmup_steps"),
                "drain_steps": _select(config, "debug.performance_drain_steps"),
                "metric": "committed_work_per_second",
                "window": "continuous_optimizer_steps",
            },
            "service_profiles": {
                "producer": {
                    "samples": _select(
                        config, "debug.topology_producer_profile_samples"
                    ),
                    "warmup_samples": _select(
                        config,
                        "debug.topology_producer_profile_warmup_samples",
                    ),
                    "sink": "initialized_trainer_transfer_clients",
                },
                "consumer": {
                    "metric": "committed_work_per_trainer_compute_second",
                    "window": "performance_optimizer_steps",
                },
            },
        },
        artifacts={
            "output_dir": _select(config, "output_dir"),
            "cache_dir": _select(config, "cache_dir"),
        },
    )


def compile_execution_plan(
    config: DictConfig,
    *,
    config_path: str | None = None,
    cli_args: Sequence[str] | None = None,
    source_context_parallel_size: int | None = None,
    source_feature_contract: Mapping[str, Any] | None = None,
) -> ExecutionPlan:
    """Compile a validated structured config without probing or starting a backend."""

    target_path = _select(config, "model.target_model_path")
    if not target_path:
        raise PlanCompileError("model.target_model_path is required")
    offline = _select(config, "inference.inference_engine_type") == "offline"
    train_source = _select(config, "inference.offline.data_path" if offline else "dataset.train_data_path")
    if not train_source:
        raise PlanCompileError("dataset.train_data_path is required")

    source = _source_record(config_path, cli_args)
    source["resolved_config_sha256"] = _resolved_config_sha256(config)
    draft = _load_draft_descriptor(config)
    target_family = _resolve_target_family(target_path)
    runtime = _runtime_values(config, draft, source_context_parallel_size=source_context_parallel_size)
    try:
        context_compiler = compile_offline_context_contract if offline else compile_sglang_context_contract
        context_contract = context_compiler(
            max_sequence_length=_select(config, "training.max_seq_length"),
            context_parallel_size=int(
                _select(config, "training.draft_context_parallel_size", 1)
            ),
            model_config=_target_config_with_sglang_overrides(config, target_path),
            native_context_length=_select(
                config, "model.native_context_length"
            ),
            allow_context_length_override=bool(
                _select(
                    config,
                    "inference.sglang.allow_context_length_override",
                    False,
                )
            ),
            context_shard_alignment=int(
                runtime["draft_context_shard_alignment"]
            ),
        )
    except ContextContractError as exc:
        raise PlanCompileError(str(exc)) from exc
    runtime["aspec_context_contract"] = copy.deepcopy(context_contract)
    workload = _workload(config, draft, target_family, runtime)
    if source_feature_contract is not None:
        from aspec.feature_contract import shared_feature_union

        # Internal producer-only plan. Validate through the same payload rules;
        # this does not relax any consumer's configured model/feature contract.
        source_features = shared_feature_union([source_feature_contract])
        workload = replace(workload, features=source_features)
        runtime["aux_hidden_states_layers"] = list(source_features["aux_hidden_state_layers"])
        runtime["store_last_hidden_states"] = "last_hidden_states" in source_features["tensors"]
    if offline:
        from aspec.offline.dataset import OfflineDataset, bind_offline_dataset

        dataset = OfflineDataset(train_source)
        offline_args = argparse.Namespace(**runtime)
        bind_offline_dataset(dataset, offline_args, workload.features)
        runtime.update(vars(offline_args))
        workload = replace(workload, data={
            **workload.data, "train_source": train_source,
            "eval_source": train_source if dataset.count("eval") else None,
            "offline_dataset_identity": dataset.identity})
    policy = _policy(config, runtime, context_contract)
    max_pool_size = policy.capacity["host_orchestrator"]["max_sample_pool_size"]
    if not isinstance(max_pool_size, int) or max_pool_size < 0:
        raise PlanCompileError("inference.max_sample_pool_size must be a non-negative integer")

    requested_features = sorted(workload.features["tensors"])
    data_plane = _select(config, "mooncake.data_plane", "store")
    transport_protocol = _select(config, "mooncake.protocol", "tcp")
    feature_affinity = _select(config, "mooncake.feature_affinity", "any")
    gpu_direct_requested = bool(
        _select(config, "mooncake.enable_gpu_direct", False)
    ) and transport_protocol in {"rdma", "nvlink_intra"}
    streaming_requested = bool(
        _select(config, "training.streaming_dispatch_with_training", False)
    )
    pipeline_requested = bool(
        _select(config, "training.pipeline_dispatch_with_training", False)
    )
    attention_backend = _select(config, "training.attention_backend")
    context_parallel_size = int(
        _select(config, "training.draft_context_parallel_size", 1)
    )
    draft_attention_requested = (
        attention_backend == "fa4"
        or is_cudnn_flex_backend(attention_backend)
        or context_parallel_size > 1
        or any(w is not None for w in draft["attention"]["context_windows"])
        or draft["attention"]["csa"] is not None
        or draft["attention"]["qsa"] is not None
        or draft["attention"].get("dsa") is not None
    )
    direct_target_output = bool(
        _select(config, "inference.target_prefill_memory_optimized", False)
    )
    cluster_mode = _select(config, "cluster.mode", "single_node")
    multi_node_requested = cluster_mode == "static_multi_node"
    capabilities = [
        CapabilityDecision(
            name="target_feature_engine",
            requirement="required",
            requested=_select(config, "inference.inference_engine_type"),
            resolved="offline_replay" if offline else "sglang",
        ),
        CapabilityDecision(
            name="draft_trainer",
            requirement="required",
            requested=draft["architecture"],
            resolved=draft["family"],
        ),
        CapabilityDecision(
            name="training_backend",
            requirement="required",
            requested=_select(config, "training.train_backend"),
            resolved="fsdp",
        ),
        CapabilityDecision(
            name="static_multi_node_execution",
            requirement="required" if multi_node_requested else "optional",
            requested=multi_node_requested,
            resolved=(
                {
                    "membership": "static_explicit",
                    "target_group_locality": "node_local",
                    "transport_stage": (
                        "host_staged_tcp"
                        if data_plane == "store"
                        else (
                            "rdma_gpu_direct"
                            if transport_protocol == "rdma"
                            else (
                                "node_local_nvlink_registered_cuda_ring"
                                if transport_protocol == "nvlink_intra"
                                else "generation_checked_tcp"
                            )
                        )
                    ),
                    "feature_affinity": feature_affinity,
                    "failure_policy": "fail_stop_attempt_checkpoint_restart",
                }
                if multi_node_requested
                else None
            ),
            effective_state=(
                "pending_runtime_evidence" if multi_node_requested else "omitted"
            ),
        ),
        CapabilityDecision(
            name="target_features",
            requirement="required",
            requested=requested_features,
            resolved=requested_features,
        ),
        CapabilityDecision(
            name="node_local_feature_affinity",
            requirement=(
                "required" if feature_affinity == "node_local_required" else "optional"
            ),
            requested=feature_affinity,
            resolved=(
                {
                    "producer_consumer_locality": "same_physical_node",
                    "remote_feature_fallback": False,
                    "gradient_collective_scope": "global_draft_data_parallel",
                }
                if feature_affinity == "node_local_required"
                else None
            ),
            effective_state=(
                "pending_runtime_evidence"
                if feature_affinity == "node_local_required"
                else "omitted"
            ),
        ),
        CapabilityDecision(
            name="tensor_transport",
            requirement="required",
            requested={"data_plane": data_plane, "protocol": transport_protocol},
            resolved=f"mooncake:{data_plane}:{transport_protocol}",
        ),
        CapabilityDecision(
            name="gpu_direct_transport",
            requirement="required" if gpu_direct_requested else "optional",
            requested=gpu_direct_requested,
            resolved=(
                "generation_checked_cuda_ring"
                if data_plane == "transfer_engine" and gpu_direct_requested
                else None
            ),
            effective_state=(
                "pending_runtime_evidence" if gpu_direct_requested else "omitted"
            ),
        ),
        CapabilityDecision(
            name="target_prefill_output",
            requirement="required" if direct_target_output else "optional",
            requested=direct_target_output,
            resolved=(
                "prefill_direct_to_generation_checked_transport"
                if direct_target_output
                else None
            ),
            effective_state=(
                "pending_runtime_evidence" if direct_target_output else "omitted"
            ),
        ),
        CapabilityDecision(
            name="producer_consumer_pipeline",
            requirement=(
                "required" if streaming_requested or pipeline_requested else "optional"
            ),
            requested={
                "streaming": streaming_requested,
                "next_step_prefix": pipeline_requested,
            },
            resolved=(
                "bounded_queue_streaming_with_optional_next_step_prefix"
                if streaming_requested or pipeline_requested
                else None
            ),
            effective_state=(
                "pending_runtime_evidence"
                if streaming_requested or pipeline_requested
                else "omitted"
            ),
        ),
        CapabilityDecision(
            name="draft_attention_execution",
            requirement="required" if draft_attention_requested else "optional",
            requested={
                "attention_backend": attention_backend,
                "context_parallel_size": context_parallel_size,
                "head_layout": draft["attention"]["head_layout"],
                **(
                    {"attention_policy": draft["attention"]}
                    if (
                        draft["attention"]["csa"] is not None
                        or draft["attention"]["qsa"] is not None
                        or draft["attention"].get("dsa") is not None
                    )
                    else {}
                ),
            },
            resolved=(
                {
                    "attention_backend": attention_backend,
                    "context_parallel_size": context_parallel_size,
                    "feature_layout": (
                        "context_sharded" if context_parallel_size > 1 else "full"
                    ),
                    "head_layout": draft["attention"]["head_layout"],
                    **(
                        {"cp_strategy": (
                            "local" if context_parallel_size == 1 else
                            "head_sequence_all_to_all"
                            if draft["attention"]["head_layout"]["query_heads"] % context_parallel_size == 0
                            else "query_partials"
                        )}
                        if draft["family"] in {"eagle3", "eagle4"} and attention_backend == "fa4"
                        else {}
                    ),
                    **(
                        {"cp_strategy": (
                            "local" if context_parallel_size == 1 else "query_partials"
                        )}
                        if draft["family"] in {"dflash", "dflash2", "dspark"}
                        and is_cudnn_flex_backend(attention_backend)
                        else {}
                    ),
                    **(
                        {"attention_policy": draft["attention"]}
                        if (
                            draft["attention"]["csa"] is not None
                            or draft["attention"]["qsa"] is not None
                            or draft["attention"].get("dsa") is not None
                        )
                        else {}
                    ),
                }
                if draft_attention_requested
                else None
            ),
            effective_state=(
                "pending_runtime_evidence" if draft_attention_requested else "omitted"
            ),
        ),
        CapabilityDecision(
            name="work_scheduler",
            requirement="required",
            requested="optimizer_step_exactly_once",
            resolved="stable_work_identity_with_step_commit",
        ),
        CapabilityDecision(
            name="data_lifecycle",
            requirement="required",
            requested="observable_terminal_ownership",
            resolved="produce_dispatch_release_or_discard",
        ),
        CapabilityDecision(
            name="capacity_control",
            requirement="required" if max_pool_size > 0 else "optional",
            requested=policy.capacity["host_orchestrator"]["max_sample_pool_size"],
            resolved=(
                "ready_pool_watermark_with_bounded_inflight" if max_pool_size > 0 else None
            ),
            effective_state=(
                "pending_runtime_evidence" if max_pool_size > 0 else "omitted"
            ),
        ),
    ]
    packing_requested = bool(_select(config, "training.dflash_packing"))
    capabilities.append(
        CapabilityDecision(
            name="mixed_length_packing",
            requirement="required" if packing_requested else "optional",
            requested=packing_requested,
            resolved=(
                ("controller_owned_document_aware_thd_rows"
                 if is_direct_cp_backend(attention_backend) else
                 "controller_owned_document_aware_fixed_rows")
                if packing_requested
                else None
            ),
            effective_state=("pending_runtime_evidence" if packing_requested else "omitted"),
        )
    )
    if packing_requested and is_direct_cp_backend(attention_backend):
        capabilities.append(CapabilityDecision(
            name="draft_thd_execution",
            requirement="required",
            requested=True,
            resolved=draft_thd_contract(
                draft["attention"], backend=attention_backend, frozen_backbone=(
                    draft["family"] == "eagle4" and draft["eagle4"]["stage"] == "candidate_head"
                ),
            ),
            effective_state="pending_runtime_evidence",
        ))
    long_context_requested = int(_select(config, "training.max_seq_length")) >= 131072
    capabilities.append(
        CapabilityDecision(
            name="long_context_execution",
            requirement="required" if long_context_requested else "optional",
            requested=_select(config, "training.max_seq_length"),
            resolved=(
                {
                    "effective_training_max_sequence_length": context_contract[
                        "effective_training_max_sequence_length"
                    ],
                    "backend_context_length": context_contract[
                        "backend_context_length"
                    ],
                    "model_native_context_limit": context_contract[
                        "model_native_context_limit"
                    ],
                    "model_context_extension": copy.deepcopy(
                        context_contract["model_context_extension"]
                    ),
                    "admitted_training_context_limit": context_contract[
                        "admitted_training_context_limit"
                    ],
                    "context_override": copy.deepcopy(
                        context_contract["context_override"]
                    ),
                    "training_sharding": copy.deepcopy(
                        context_contract["training_sharding"]
                    ),
                }
                if long_context_requested
                else None
            ),
            effective_state=(
                "pending_runtime_evidence" if long_context_requested else "omitted"
            ),
        )
    )
    performance_requested = bool(
        _select(config, "debug.performance_measurement_required")
    )
    capabilities.append(
        CapabilityDecision(
            name="performance_measurement",
            requirement="required" if performance_requested else "optional",
            requested=performance_requested,
            resolved=(
                "continuous_committed_work_window" if performance_requested else None
            ),
            effective_state=(
                "pending_runtime_evidence" if performance_requested else "omitted"
            ),
        )
    )
    capabilities.append(
        CapabilityDecision(
            name="training_numerics",
            requirement="required" if performance_requested else "optional",
            requested=("finite_loss_and_gradient_norm" if performance_requested else False),
            resolved=("finite_loss_and_gradient_norm" if performance_requested else None),
            effective_state=(
                "pending_runtime_evidence" if performance_requested else "omitted"
            ),
        )
    )
    capabilities = tuple(capabilities)

    omissions: list[Mapping[str, Any]] = []
    if not _select(config, "training.train_with_decode"):
        omissions.append({"capability": "decode_training", "reason": "not_requested"})
    if not multi_node_requested:
        omissions.append(
            {"capability": "static_multi_node_execution", "reason": "not_requested"}
        )
    if feature_affinity != "node_local_required":
        omissions.append(
            {"capability": "node_local_feature_affinity", "reason": "not_requested"}
        )
    if not gpu_direct_requested:
        omissions.append({"capability": "gpu_direct_transport", "reason": "not_requested"})
    if not direct_target_output:
        omissions.append(
            {"capability": "target_prefill_output", "reason": "not_requested"}
        )
    if not streaming_requested and not pipeline_requested:
        omissions.append(
            {"capability": "producer_consumer_pipeline", "reason": "not_requested"}
        )
    if not draft_attention_requested:
        omissions.append(
            {"capability": "draft_attention_execution", "reason": "not_requested"}
        )
    if max_pool_size == 0:
        omissions.append(
            {
                "capability": "capacity_control",
                "reason": "legacy_unbounded_pool_requested",
            }
        )
    if not packing_requested:
        omissions.append({"capability": "mixed_length_packing", "reason": "not_requested"})
    if not long_context_requested:
        omissions.append(
            {"capability": "long_context_execution", "reason": "sequence_limit_below_128k"}
        )
    if not performance_requested:
        omissions.append(
            {"capability": "performance_measurement", "reason": "not_requested"}
        )
        omissions.append(
            {"capability": "training_numerics", "reason": "not_requested"}
        )
    if "last_hidden_states" not in workload.features["tensors"]:
        omissions.append(
            {
                "capability": "target_last_hidden_state",
                "reason": "not_required_by_drafter",
            }
        )
    omissions.extend(
        {
            "capability": f"target_{item['tensor']}",
            "reason": item["reason"],
        }
        for item in workload.features["omissions"]
    )

    return ExecutionPlan(
        source=source,
        workload=workload,
        policy=policy,
        capabilities=capabilities,
        fallbacks=(),
        omissions=tuple(omissions),
        diagnostics=(
            {
                "code": "effective_state_pending",
                "message": (
                    "Dry-run proves static resolution only; effective capabilities require "
                    "runtime evidence"
                ),
            },
        ),
        _runtime_values=runtime,
    )


def save_execution_plan(plan: ExecutionPlan, path: str | os.PathLike[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(plan.to_json() + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
