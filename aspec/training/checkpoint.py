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

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful

from aspec.models.draft.keymap import to_internal_keys
from aspec.utils.logging import logger

WORK_STATE_SCHEMA_VERSION = "aspec.checkpoint-work-state/v1"
CHECKPOINT_COMPLETION_SCHEMA_VERSION = "aspec.checkpoint-completion/v1"
RNG_STATE_SCHEMA_VERSION = "aspec.rng-state/v1"


def _logical_training_topology(args: Any, world_size: int) -> dict[str, Any]:
    """Return resume-relevant logical topology without physical host identity."""

    topology: dict[str, Any] = {"world_size": int(world_size)}
    for field in (
        "training_num_nodes",
        "training_num_gpus_per_node",
        "attention_backend",
        "draft_context_parallel_size",
        "sp_ulysses_size",
        "sp_ring_size",
        "fsdp_strategy",
        "draft_algorithm",
    ):
        value = getattr(args, field, None)
        if value is not None:
            topology[field] = value
    if getattr(args, "draft_algorithm", None) == "mtp":
        topology["training_objective"] = {
            "steps": args.mtp_steps, "step_weight_beta": args.mtp_step_weight_beta,
            "step_weights": args.ploss_weights,
            "loss": "teacher_forced_native_mtp_ce", "frozen_shared": ["embedding", "lm_head"],
        }
    elif getattr(args, "draft_algorithm", None) in {"eagle3", "eagle4"}:
        topology["training_objective"] = {
            key: getattr(args, key, None)
            for key in ("ttt_length", "loss_type", "lk_eta", "ploss_weights")
        }
    elif getattr(args, "draft_algorithm", None) in {"dflash", "dflash2", "dspark"}:
        topology["training_objective"] = {
            "block_size": int(args.dflash_block_size),
            "anchor_sampling_scope": getattr(args, "anchor_sampling_scope", "row"),
        }
        prefix = "dspark" if args.draft_algorithm == "dspark" else "dflash"
        fields = ("num_anchors", "num_target_layers", "loss_decay_gamma",
                  "ce_loss_alpha", "l1_loss_alpha")
        if prefix == "dspark":
            fields += ("confidence_head_alpha",)
            for side in ("target", "draft"):
                for name, default in (("temperature", 1.0), ("top_p", 1.0), ("top_k", -1)):
                    key = f"l1_{side}_{name}"
                    topology["training_objective"][key] = getattr(args, f"dspark_{key}", default)
        else:
            fields += ("loss_objective", "dpace_alpha")
        for name in fields:
            key = f"{prefix}_{name}"
            if hasattr(args, key):
                topology["training_objective"][name] = getattr(args, key)
        if args.draft_algorithm == "dflash2":
            topology["training_objective"]["selector_loss_alpha"] = args.dflash2_selector_loss_alpha
            topology["training_objective"]["selector_objective"] = "experimental_teacher_forced_topk_ce"
    if "training_objective" in topology:
        fields = ("max_seq_length", "micro_batch_size", "draft_accumulation_steps",
                  "max_grad_norm", "seed")
        fields += (("freeze_lm_head",) if args.draft_algorithm in {"eagle3", "eagle4", "mtp"}
                   else ("deterministic_anchor_sampling",))
        for key in fields:
            if hasattr(args, key):
                topology["training_objective"][key] = getattr(args, key)
    config = getattr(args, "draft_model_config_obj", None)
    if config is not None:
        from aspec.config.draft_algorithm import draft_model_semantics
        from aspec.config.draft_attention import resolve_draft_attention

        raw = config if isinstance(config, dict) else config.to_dict()
        attention = resolve_draft_attention(raw)
        topology["draft_attention_windows"] = attention["context_windows"]
        topology["draft_attention_policy"] = attention
        topology["draft_model_semantics"] = draft_model_semantics(config)
        from aspec.config.draft_structure import resolve_draft_structure

        topology["draft_ffn_policy"] = resolve_draft_structure(raw)
        if getattr(args, "draft_algorithm", None) == "eagle4":
            from aspec.config.eagle4 import eagle4_training_policy
            topology["training_objective"]["eagle4"] = eagle4_training_policy(raw)
    return topology


def _validate_exact_resume_topology(args: Any, metadata: dict[str, Any], *, checkpoint_dir: Path | None = None) -> None:
    saved_world_size = metadata.get("world_size")
    if not isinstance(saved_world_size, int) or saved_world_size < 1:
        raise RuntimeError(
            "Exact resume requires checkpoint metadata with a valid world_size"
        )
    current_world_size = (
        dist.get_world_size()
        if dist.is_available() and dist.is_initialized()
        else getattr(args, "world_size", None)
    )
    if not isinstance(current_world_size, int) or current_world_size < 1:
        raise RuntimeError("Exact resume cannot determine the current training world_size")
    if current_world_size != saved_world_size:
        raise RuntimeError(
            "Exact resume does not support world-size changes: "
            f"checkpoint={saved_world_size}, current={current_world_size}. "
            "Use continual_training=true for a weights-only new stage."
        )

    saved_topology = metadata.get("training_topology")
    if saved_topology is not None and not isinstance(saved_topology, dict):
        raise RuntimeError("Exact resume checkpoint has invalid training_topology metadata")
    current_topology = _logical_training_topology(args, current_world_size)
    if getattr(args, "draft_algorithm", None) == "dspark":
        saved_objective = (saved_topology or {}).get("training_objective") or {}
        for side in ("target", "draft"):
            for name, default in (("temperature", 1.0), ("top_p", 1.0), ("top_k", -1)):
                key = f"l1_{side}_{name}"
                if saved_objective.get(key, default) != getattr(args, f"dspark_{key}", default):
                    raise RuntimeError(f"Exact resume cannot change dspark_{key}; use weights-only initialization")
    # Checkpoints predating this option always sampled per physical row.
    # Missing semantic identity must not authorize a switch to per-sample training.
    if (getattr(args, "draft_algorithm", None) in {"dspark", "dflash", "dflash2"}
            and ((saved_topology or {}).get("training_objective") or {}).get(
                "anchor_sampling_scope", "row"
            ) != getattr(args, "anchor_sampling_scope", "row")):
        raise RuntimeError("Exact resume cannot change anchor_sampling_scope; use weights-only initialization")
    from aspec.config.draft_structure import checkpoint_ffn_policy

    saved_ffn = checkpoint_ffn_policy(checkpoint_dir, saved_topology or {})
    current_ffn = current_topology.get("draft_ffn_policy", {"ffn_type": "dense"})
    if saved_ffn is None and current_ffn["ffn_type"] != "dense":
        raise RuntimeError("Exact resume cannot verify legacy MoE routing without draft_config.json")
    if saved_ffn is not None and saved_ffn != current_ffn:
        raise RuntimeError("Exact resume cannot change the checkpoint Draft FFN/routing policy")
    current_windows = current_topology.get("draft_attention_windows", [])
    current_policy = current_topology.get("draft_attention_policy") or {}
    if (
        any(current_policy.get(kind) is not None for kind in ("csa", "dsa"))
        and (saved_topology is None or "draft_attention_policy" not in saved_topology
             or (current_policy.get("dsa") is not None
                 and not (saved_topology.get("draft_attention_policy") or {}).get("dsa")))
    ):
        name = "DSA" if current_policy.get("dsa") is not None else "CSA"
        raise RuntimeError(
            f"Exact resume cannot reinterpret a checkpoint without a {name} policy; "
            f"use continual_training=true for a new {name} phase"
        )
    # Legacy checkpoints predate SWA and therefore mean full attention.
    if (saved_topology is None or "draft_attention_windows" not in saved_topology) and any(
        window is not None for window in current_windows
    ):
        raise RuntimeError("Exact resume cannot change a legacy full-attention checkpoint to SWA")
    if saved_topology is None:
        # Older complete checkpoints recorded world_size only.  Preserve that
        # compatibility while every newly published checkpoint records the
        # full logical parallelism contract below.
        return
    if current_policy.get("dsa") is not None:
        # Chunk workspace is an execution bound, not the DSA objective. Preserve
        # every recorded semantic field while allowing a different memory budget.
        def without_dsa_workspace(topology):
            policy = topology.get("draft_attention_policy") or {}
            dsa = policy.get("dsa")
            if dsa is None:
                return topology
            return {**topology, "draft_attention_policy": {
                **policy, "dsa": {key: value for key, value in dsa.items() if key != "workspace_bytes"}
            }}
        saved_topology = without_dsa_workspace(saved_topology)
        current_topology = without_dsa_workspace(current_topology)
    # Older checkpoints recorded only part of the objective (e.g. block_size).
    # Keep that recovery path, while checking every field that was recorded;
    # newly published checkpoints protect the complete current objective.
    def matches_saved(value, current):
        if isinstance(value, dict) and isinstance(current, dict):
            return all(key in current and matches_saved(item, current[key])
                       for key, item in value.items())
        return value == current

    if (saved_topology.get("draft_algorithm") == current_topology.get("draft_algorithm") == "eagle4"
            and "eagle4" in (saved_topology.get("training_objective") or {})):
        # Early EAGLE4 metadata included token chunking, which changes workspace
        # rather than the retrieval objective. Normalize at the recovery boundary.
        saved_topology = {**saved_topology, "training_objective": {
            **saved_topology["training_objective"], "eagle4": {
                key: value for key, value in saved_topology["training_objective"]["eagle4"].items()
                if key != "loss_chunk_size"
            },
        }}
    if "draft_model_semantics" in saved_topology:
        from aspec.config.draft_algorithm import draft_model_semantics

        saved_topology = dict(saved_topology)
        saved_topology["draft_model_semantics"] = draft_model_semantics(saved_topology["draft_model_semantics"])
    # Early native MTP checkpoints used the generic Draft policy label even
    # though their only decoder path was official causal, teacher-forced MTP.
    # Normalize that known serialization error at this boundary; never accept
    # a genuinely different attention policy or rewrite the stored checkpoint.
    saved_policy = saved_topology.get("draft_attention_policy") or {}
    saved_semantics = saved_topology.get("draft_model_semantics") or {}
    objective = saved_topology.get("training_objective") or {}
    if (saved_topology.get("draft_algorithm") == current_topology.get("draft_algorithm") == "mtp"
            and saved_semantics.get("architectures") == ["NativeMTPDraftModel"]
            and saved_semantics.get("is_causal") is True
            and objective.get("loss") == "teacher_forced_native_mtp_ce"
            and objective.get("frozen_shared") == ["embedding", "lm_head"]
            and saved_policy.get("draft_block_attention") == "bidirectional"
            and current_policy.get("draft_block_attention") == "causal_native_mtp"):
        saved_topology = {**saved_topology, "draft_attention_policy": {
            **saved_policy, "draft_block_attention": "causal_native_mtp",
        }}
        logger.warning("Normalized legacy native MTP attention label; causal execution is unchanged")
    mismatches = {
        key: {"checkpoint": value, "current": current_topology.get(key)}
        for key, value in saved_topology.items()
        if not (matches_saved(value, current_topology.get(key))
                if key == "training_objective" else current_topology.get(key) == value)
    }
    if mismatches:
        raise RuntimeError(
            "Exact resume does not support logical training-topology changes: "
            f"{mismatches}. Use continual_training=true for a weights-only new stage."
        )


def _checkpoint_paths(
    checkpoint_dir: str | Path, completed_step: int
) -> tuple[Path, Path, Path]:
    base_dir = Path(checkpoint_dir).expanduser()
    step_id = int(completed_step) + 1
    staging_dir = base_dir / f".iter_{step_id:07d}.incomplete"
    final_dir = base_dir / f"iter_{step_id:07d}"
    return base_dir, staging_dir, final_dir


def _atomic_write_text(path: Path, value: str) -> None:
    """Durably replace a small control file without exposing partial contents."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _rng_state_path(checkpoint_dir: Path, rank: int) -> Path:
    return checkpoint_dir / "rng" / f"rank-{int(rank):05d}.pt"


def _save_rank_rng_state(checkpoint_dir: Path, *, rank: int, world_size: int) -> None:
    path = _rng_state_path(checkpoint_dir, rank)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    state = {
        "schema_version": RNG_STATE_SCHEMA_VERSION,
        "rank": int(rank),
        "world_size": int(world_size),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_rng_state(state: Any, *, legacy: bool = False) -> None:
    """Reject incomplete random state before an exact resume can continue."""

    if not isinstance(state, dict):
        raise RuntimeError("Exact resume requires complete RNG state")

    def valid_tensor(value):
        return (
            isinstance(value, torch.Tensor)
            and value.device.type == "cpu"
            and value.dtype == torch.uint8
            and value.ndim == 1
            and value.numel() > 0
        )

    if not valid_tensor(state.get("torch")):
        raise RuntimeError("Invalid or missing CPU RNG state in exact-resume checkpoint")
    cuda_state = state.get("cuda")
    if cuda_state is None:
        if torch.cuda.is_available():
            raise RuntimeError("Exact CUDA resume requires saved CUDA RNG state")
    elif legacy and isinstance(cuda_state, (list, tuple)):
        if not cuda_state or not all(valid_tensor(value) for value in cuda_state):
            raise RuntimeError("Invalid legacy CUDA RNG state in exact-resume checkpoint")
    elif not valid_tensor(cuda_state):
        raise RuntimeError("Invalid CUDA RNG state in exact-resume checkpoint")


def _load_rank_rng_state(
    checkpoint_dir: Path,
    *,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    path = _rng_state_path(checkpoint_dir, rank)
    if path.is_file():
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to load RNG checkpoint from {path}: {exc}") from exc
        if (
            not isinstance(state, dict)
            or state.get("schema_version") != RNG_STATE_SCHEMA_VERSION
            or state.get("rank") != rank
            or state.get("world_size") != world_size
        ):
            raise RuntimeError(
                "RNG checkpoint identity does not match the current rank/topology: "
                f"path={path}, rank={rank}, world_size={world_size}"
            )
        _validate_rng_state(state)
        return state

    legacy_path = checkpoint_dir / "rng.pt"
    if legacy_path.is_file() and world_size == 1:
        try:
            state = torch.load(legacy_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load legacy RNG checkpoint from {legacy_path}: {exc}"
            ) from exc
        _validate_rng_state(state, legacy=True)
        return state
    if legacy_path.is_file():
        raise RuntimeError(
            "Exact multi-rank resume requires one RNG checkpoint per rank; the checkpoint "
            f"only contains legacy rank-0 state at {legacy_path}"
        )
    raise RuntimeError(
        f"No RNG checkpoint for rank {rank} at {path}. Exact resume requires "
        "deterministic per-rank RNG state."
    )


def prepare_checkpoint_save(checkpoint_dir: str, completed_step: int) -> Path:
    """Reserve a fresh staging directory before distributed checkpoint I/O starts."""

    base_dir, staging_dir, final_dir = _checkpoint_paths(checkpoint_dir, completed_step)
    base_dir.mkdir(parents=True, exist_ok=True)
    if final_dir.exists():
        raise FileExistsError(f"final checkpoint already exists: {final_dir}")
    try:
        staging_dir.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(
            f"staging checkpoint already exists: {staging_dir}; inspect or remove the "
            "incomplete prior attempt before retrying"
        ) from exc
    return staging_dir


class ModelState(Stateful):
    """Wrapper for model state only."""

    def __init__(self, model):
        self.model = model

    def state_dict(self):
        model_state_dict, _ = get_state_dict(self.model, optimizers=[])
        return {"model": model_state_dict}

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model, optimizers=[], model_state_dict=state_dict["model"], optim_state_dict=None
        )


def _fp32_param_names(optimizer) -> list[str]:
    """Names for a ``BF16Optimizer``'s fp32 master copies, in ``fp32_params`` order."""
    return [name for name, param in optimizer.model.named_parameters() if param.requires_grad]


class OptimizerState(Stateful):
    """Wrapper for optimizer + fp32 master params."""

    def __init__(self, model, optimizer, *, load_path=None):
        self.model = model
        self.optimizer = optimizer
        self._is_bf16_optimizer = hasattr(optimizer, "sync_fp32_params_from_model")
        self._load_path = load_path
        self._load_template = None

    def _optimizer_load_template(self):
        # DCP reads only destination keys. A fresh Adam has no lazy moments;
        # passing its empty state silently omits every saved moment and step.
        # Materialize only slots actually present in this checkpoint, including
        # legitimately empty/unused parameters, without performing an update.
        if self._load_template is None:
            metadata = dcp.FileSystemReader(str(self._load_path)).read_metadata()
            template = self.optimizer.state_dict()
            states = {}
            prefix = "optim_state.optim.state."
            for key, value in metadata.state_dict_metadata.items():
                if not key.startswith(prefix):
                    continue
                parts = key[len(prefix) :].split(".")
                index = int(parts[0])
                if not 0 <= index < len(self.optimizer.fp32_params):
                    raise ValueError("Optimizer checkpoint references an unknown master parameter")
                state = states.setdefault(index, {})
                if len(parts) == 1:
                    continue  # An explicitly saved empty lazy state.
                field = parts[1]
                master = self.optimizer.fp32_params[index]
                if len(parts) != 2 or field not in {"step", "exp_avg", "exp_avg_sq"}:
                    raise ValueError(f"Unsupported BF16 Adam checkpoint state: {key}")
                expected_shape = () if field == "step" else tuple(master.shape)
                dtype = value.properties.dtype
                valid_dtype = (
                    dtype in {torch.float32, torch.float64}
                    if field == "step"
                    else dtype == master.dtype
                )
                if tuple(value.size) != expected_shape or not valid_dtype:
                    raise ValueError(f"Optimizer checkpoint state shape/dtype differs: {key}")
                state[field] = (
                    torch.empty((), device=master.device, dtype=dtype)
                    if field == "step"
                    else torch.empty_like(master)
                )
            self._load_template = {**template, "state": states}
        return self._load_template

    def state_dict(self):
        if self._is_bf16_optimizer:
            names = _fp32_param_names(self.optimizer)
            return {
                "optim": (
                    self._optimizer_load_template()
                    if self._load_path is not None
                    else self.optimizer.state_dict()
                ),
                # Keyed by name: which parameters are trainable can differ between the run
                # that saved and the run that loads.
                "fp32_params": {
                    name: param.data
                    for name, param in zip(names, self.optimizer.fp32_params, strict=True)
                },
            }
        _, optimizer_state_dict = get_state_dict(self.model, optimizers=[self.optimizer])
        return {"optim": optimizer_state_dict}

    def load_state_dict(self, state_dict):
        if self._is_bf16_optimizer:
            self.optimizer.load_state_dict(state_dict["optim"])
            saved = state_dict.get("fp32_params")
            if saved:
                names = _fp32_param_names(self.optimizer)
                missing = [name for name in names if name not in saved]
                if missing:
                    raise ValueError(
                        "Optimizer checkpoint does not cover every trainable parameter: "
                        f"{missing[:5]}{'...' if len(missing) > 5 else ''}. The saved run trained "
                        "a different parameter set (for example it froze the lm_head and this one "
                        "does not). Use training.continual_training=true to start a fresh "
                        "optimizer from those weights instead of resuming this one."
                    )
                with torch.no_grad():
                    for name, master in zip(names, self.optimizer.fp32_params, strict=True):
                        master.data.copy_(saved[name])
                logger.info("Restored fp32 master params from checkpoint")
            return
        set_state_dict(
            self.model,
            optimizers=[self.optimizer],
            model_state_dict=None,
            optim_state_dict=state_dict["optim"],
        )


class LRSchedulerState(Stateful):
    """Wrapper for LR scheduler state only."""

    def __init__(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def state_dict(self):
        return {"lr_scheduler": self.lr_scheduler.state_dict()}

    def load_state_dict(self, state_dict):
        self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])


def _read_checkpoint_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse checkpoint metadata at {path}")
        return {}


def _write_checkpoint_metadata(path: Path, metadata: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def _committed_work_sha256(work_ids: list[str]) -> str:
    payload = json.dumps(
        sorted(work_ids), separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_checkpoint_work_state(
    checkpoint_dir: str,
    completed_step: int,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Persist controller-owned committed work next to a completed checkpoint.

    The distributed checkpoint captures model/optimizer/RNG state.  This sidecar
    captures the semantic source work that reached optimizer commit so a new
    controller can regenerate only uncommitted work after restart.
    """

    work_ids = sorted(str(value) for value in state.get("committed_work_ids", []))
    if len(work_ids) != len(set(work_ids)):
        raise ValueError("checkpoint work state contains duplicate committed work IDs")
    dataset_fingerprint = state.get("dataset_fingerprint")
    if not isinstance(dataset_fingerprint, str) or not dataset_fingerprint.startswith(
        "sha256:"
    ):
        raise ValueError("checkpoint work state requires a sha256 dataset fingerprint")

    step_id = int(completed_step) + 1
    base_dir, artifact_dir, final_dir = _checkpoint_paths(checkpoint_dir, completed_step)
    if not (artifact_dir / "model").is_dir():
        raise FileNotFoundError(
            f"cannot publish work state before staged model checkpoint exists: {artifact_dir}"
        )

    payload = {
        "schema_version": WORK_STATE_SCHEMA_VERSION,
        "optimizer_step": int(completed_step),
        "dataset_epoch": int(state.get("dataset_epoch", 0)),
        "dataset_size": int(state.get("dataset_size", 0)),
        "dataset_fingerprint": dataset_fingerprint,
        "committed_work_count": len(work_ids),
        "committed_work_sha256": _committed_work_sha256(work_ids),
        "committed_work_ids": work_ids,
    }
    if "dataset_order" in state:
        payload["dataset_order"] = dict(state["dataset_order"])
    path = artifact_dir / "work_state.json"
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )

    metadata_path = artifact_dir / "meta.json"
    metadata = _read_checkpoint_metadata(metadata_path)
    if int(metadata.get("global_step", -1)) != int(completed_step):
        raise RuntimeError(
            "staged checkpoint metadata does not match the optimizer commit: "
            f"path={metadata_path}, metadata_step={metadata.get('global_step')!r}, "
            f"completed_step={completed_step}"
        )
    metadata["work_state"] = {
        "schema_version": WORK_STATE_SCHEMA_VERSION,
        "committed_work_count": payload["committed_work_count"],
        "committed_work_sha256": payload["committed_work_sha256"],
        "dataset_fingerprint": dataset_fingerprint,
    }
    if "dataset_order" in payload:
        metadata["work_state"]["dataset_order"] = payload["dataset_order"]
    _write_checkpoint_metadata(metadata_path, metadata)

    world_size = int(metadata.get("world_size", 0))
    if world_size < 1:
        raise RuntimeError("checkpoint publication requires valid world_size metadata")
    required_files = (
        artifact_dir / "model" / ".metadata",
        artifact_dir / "optimizer" / ".metadata",
        artifact_dir / "lr_scheduler" / ".metadata",
        metadata_path,
        path,
        *(_rng_state_path(artifact_dir, rank) for rank in range(world_size)),
    )
    missing = [str(item) for item in required_files if not item.is_file()]
    if missing:
        raise RuntimeError(
            "checkpoint publication requires complete model/optimizer/scheduler/RNG/work "
            f"state; missing={missing}"
        )

    completion = {
        "schema_version": CHECKPOINT_COMPLETION_SCHEMA_VERSION,
        "iteration": step_id,
        "optimizer_step": int(completed_step),
        "committed_work_count": payload["committed_work_count"],
        "committed_work_sha256": payload["committed_work_sha256"],
        "dataset_fingerprint": dataset_fingerprint,
        "rng_world_size": world_size,
    }
    _atomic_write_text(
        artifact_dir / "complete.json",
        json.dumps(completion, indent=2, sort_keys=True) + "\n",
    )

    if final_dir.exists():
        raise FileExistsError(f"refusing to replace existing final checkpoint: {final_dir}")
    os.replace(artifact_dir, final_dir)
    _atomic_write_text(base_dir / "latest_checkpointed_iteration.txt", str(step_id))
    logger.info(
        "Saved checkpoint work state to %s (%d committed work items)",
        final_dir / "work_state.json",
        len(work_ids),
    )
    return {key: value for key, value in payload.items() if key != "committed_work_ids"}


def load_checkpoint_work_state(args: Any) -> dict[str, Any] | None:
    """Load and validate committed source work for resume or phase continuity.

    Exact resume restores this state together with optimizer/LR/RNG state. A
    continual stage may explicitly opt into only the committed source frontier
    via ``continue_source_cursor``; model loading remains weights-only.
    """

    if not getattr(args, "load_path", None):
        return None
    if getattr(args, "continual_training", False) and not getattr(
        args, "continue_source_cursor", False
    ):
        return None
    model_dir = resolve_resume_model_dir(args)
    if model_dir is None:
        raise FileNotFoundError(
            f"training.load_path does not resolve to a checkpoint: {args.load_path}"
        )

    artifact_dir = model_dir.parent
    path = artifact_dir / "work_state.json"
    if not path.is_file():
        raise RuntimeError(
            f"Checkpoint {artifact_dir} has no work_state.json; exact resume cannot prove that "
            "already committed source work will not be replayed. Use continual_training=true "
            "for a weights-only new stage instead."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid checkpoint work state at {path}: {exc}") from exc

    if payload.get("schema_version") != WORK_STATE_SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported checkpoint work-state schema at {path}: "
            f"{payload.get('schema_version')!r}"
        )
    work_ids = payload.get("committed_work_ids")
    if not isinstance(work_ids, list) or not all(
        isinstance(value, str) and value for value in work_ids
    ):
        raise RuntimeError(f"Invalid committed_work_ids in {path}")
    if len(work_ids) != len(set(work_ids)):
        raise RuntimeError(f"Duplicate committed work IDs in {path}")
    if payload.get("committed_work_count") != len(work_ids):
        raise RuntimeError(f"Committed work count mismatch in {path}")
    if payload.get("committed_work_sha256") != _committed_work_sha256(work_ids):
        raise RuntimeError(f"Committed work digest mismatch in {path}")

    metadata = _read_checkpoint_metadata(artifact_dir / "meta.json")
    optimizer_step = metadata.get("global_step")
    if optimizer_step is not None and int(payload.get("optimizer_step", -1)) != int(
        optimizer_step
    ):
        raise RuntimeError(
            f"Checkpoint work state step does not match model metadata in {artifact_dir}"
        )
    recorded = metadata.get("work_state") or {}
    if "dataset_order" in payload:
        dataset_seed = getattr(args, "dataset_seed", None)
        expected_order = {
            "seed": getattr(args, "seed", 42) if dataset_seed is None else dataset_seed,
            "shuffle": getattr(args, "shuffle_dataset", True),
            "length_group_size": getattr(args, "length_group_size", 32),
            "packing": bool(getattr(args, "dflash_packing", False)),
        }
        if payload["dataset_order"] != recorded.get("dataset_order"):
            raise RuntimeError("Checkpoint metadata disagrees with dataset ordering")
        if payload["dataset_order"] != expected_order:
            raise RuntimeError(
                "Exact resume cannot change dataset ordering; "
                "use continual_training=true for a new stage"
            )
    for key in (
        "schema_version",
        "committed_work_count",
        "committed_work_sha256",
        "dataset_fingerprint",
    ):
        if recorded.get(key) != payload.get(key):
            raise RuntimeError(
                f"Checkpoint metadata and work_state.json disagree on {key} in {artifact_dir}"
            )
    completion_path = artifact_dir / "complete.json"
    if completion_path.is_file():
        try:
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid checkpoint completion marker at {completion_path}: {exc}"
            ) from exc
        expected_completion = {
            "schema_version": CHECKPOINT_COMPLETION_SCHEMA_VERSION,
            "iteration": int(artifact_dir.name.removeprefix("iter_")),
            "optimizer_step": int(payload["optimizer_step"]),
            "committed_work_count": int(payload["committed_work_count"]),
            "committed_work_sha256": payload["committed_work_sha256"],
            "dataset_fingerprint": payload["dataset_fingerprint"],
        }
        if "rng_world_size" in completion:
            expected_completion["rng_world_size"] = int(metadata.get("world_size", 0))
        elif not (
            int(metadata.get("world_size", 0)) == 1
            and (artifact_dir / "rng.pt").is_file()
        ):
            raise RuntimeError(
                "Checkpoint completion marker does not prove complete per-rank RNG state "
                f"in {artifact_dir}"
            )
        if completion != expected_completion:
            raise RuntimeError(
                "Checkpoint completion marker disagrees with committed work state in "
                f"{artifact_dir}"
            )
    return payload


def resolve_initial_draft_checkpoint(path: str) -> Path:
    """Resolve ``model.initial_draft_model_path`` to the ``model.safetensors`` file it names.

    Either the file itself or the directory holding it is accepted; anything else is rejected
    rather than ignored, since a typo here would start a full training run from random weights.
    """
    checkpoint_path = Path(path).expanduser()
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "model.safetensors"
    if checkpoint_path.suffix != ".safetensors" or not checkpoint_path.is_file():
        raise ValueError(
            "initial_draft_model_path must be a serving-format model.safetensors file "
            f"(or the directory holding it), got {path!r}"
        )
    return checkpoint_path


def load_initial_draft_weights(
    draft_model: torch.nn.Module, path: str, *, allow_target_embedding: bool = False
) -> Path:
    """Load a published draft checkpoint into a freshly constructed draft model.

    Unrelated to ``load()`` below, which resumes a training checkpoint with its optimizer and
    scheduler state. Published drafts carry the serving key names ``tools/convert_to_hf.py``
    writes, so keys are mapped back first, and the load is strict in both directions: a mismatch
    would otherwise leave part of the model randomly initialised for a whole run.
    """
    checkpoint_path = resolve_initial_draft_checkpoint(path)
    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        tensors = {key: f.get_tensor(key) for key in f.keys()}

    manifest_path = checkpoint_path.parent / "export_manifest.json"
    native = manifest_path.is_file() and json.loads(manifest_path.read_text()).get("storage_format") == "aspec_native"
    if native:
        from aspec.draft_artifact import validate_native_asset

        if hasattr(draft_model, "prepare_initial_native_state"):
            saved_config, manifest = validate_native_asset(checkpoint_path.parent)
            tensors = draft_model.prepare_initial_native_state(tensors, saved_config, manifest=manifest)
        else:
            # This entry is weights-only initialization, not exact recovery.
            # Preserve model bindings and strict tensors while allowing the new
            # run to choose its indexer stage/objective policy.
            validate_native_asset(checkpoint_path.parent, expected_config=draft_model.config, weights_only=True)
        if hasattr(draft_model, "keep_csa_projection_precision"):
            draft_model.keep_csa_projection_precision()
        state = tensors
    else:
        state = (draft_model.state_dict_from_serving(tensors)
                 if hasattr(draft_model, "state_dict_from_serving")
                 else to_internal_keys(tensors, draft_model.state_dict().keys()))
    if allow_target_embedding and "embed_tokens.weight" not in state:
        # DFlash-family serving assets normally borrow this frozen tensor from
        # the Target. The caller loads it from that Target immediately afterward.
        state["embed_tokens.weight"] = draft_model.state_dict()["embed_tokens.weight"]
    draft_model.load_state_dict(state, strict=True)
    if hasattr(draft_model, "t2d"):
        from aspec.models.vocab import validate_vocab_mapping

        validate_vocab_mapping(draft_model.d2t, draft_model.t2d,
                               draft_model.vocab_size, draft_model.target_vocab_size)
    return checkpoint_path


def resolve_resume_model_dir(args: Any) -> Path | None:
    """Model directory ``training.load_path`` resolves to, or None if there is nothing to resume.

    Resolution is deliberately soft so a run can point at an output directory that has no
    checkpoint yet, but callers that need weights to actually arrive must check the result
    rather than the path string.
    """
    load_root = getattr(args, "load_path", None)
    if load_root is None:
        return None

    root_path = Path(load_root).expanduser()
    if not root_path.exists():
        logger.info(f"Checkpoint directory {root_path} not found; skipping load.")
        return None

    target_step = getattr(args, "ckpt_step", None)
    # A published checkpoint is an explicit immutable recovery frontier. Keep
    # parent-directory/latest resolution, but do not force users to rewrite a
    # tracker or copy a checkpoint when recovering an earlier experiment step.
    if root_path.name.startswith("iter_") and root_path.name[5:].isdigit():
        selected_step = int(root_path.name[5:])
        if target_step is not None and int(target_step) != selected_step:
            raise ValueError("Explicit checkpoint path disagrees with ckpt_step")
        model_dir = root_path / "model"
        return model_dir if (model_dir / ".metadata").is_file() else None
    if target_step is None:
        tracker_file = root_path / "latest_checkpointed_iteration.txt"
        if not tracker_file.exists():
            logger.info(f"No tracker file at {tracker_file}; skipping load.")
            return None
        target_step = int(tracker_file.read_text().strip())

    model_dir = root_path / f"iter_{target_step:07d}" / "model"
    if not model_dir.exists():
        logger.info(f"Model checkpoint {model_dir} not found; skipping load.")
        return None
    return model_dir


def load(actor: Any) -> dict[str, Any] | None:
    """Load checkpoint from disk.

    Model weights and optimizer state live in separate directories. A resume requires both:
    restoring weights without the matching optimizer state would leave the fp32 masters stale.
    Use training.continual_training=true to start from the weights alone.
    """
    model_dir = resolve_resume_model_dir(actor.args)
    if model_dir is None:
        if getattr(actor.args, "load_path", None):
            raise FileNotFoundError(
                f"training.load_path does not resolve to a checkpoint: "
                f"{actor.args.load_path}"
            )
        return None

    checkpoint_dir = model_dir.parent
    target_step = int(checkpoint_dir.name.removeprefix("iter_"))
    optimizer_dir = checkpoint_dir / "optimizer"
    lr_scheduler_dir = checkpoint_dir / "lr_scheduler"
    metadata = _read_checkpoint_metadata(checkpoint_dir / "meta.json")
    continual_training = getattr(actor.args, "continual_training", False)
    if not continual_training:
        if not metadata:
            raise RuntimeError(
                f"Exact resume requires valid checkpoint metadata in {checkpoint_dir}"
            )
        _validate_exact_resume_topology(actor.args, metadata, checkpoint_dir=checkpoint_dir)

    # Load model weights (always)
    model_state = ModelState(actor.model)
    state_dict = {"model_state": model_state}

    try:
        dcp.load(state_dict=state_dict, checkpoint_id=str(model_dir))
        logger.info(f"Loaded model from {model_dir}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_dir}: {e}") from e

    # Keep optimizer/LR state out of continual training so it starts fresh.
    if not continual_training and getattr(actor, "optimizer", None) is None:
        raise RuntimeError("Exact resume requires an initialized optimizer")
    load_optimizer = not continual_training
    if load_optimizer and (optimizer_dir / ".metadata").is_file():
        optimizer_state = OptimizerState(actor.model, actor.optimizer, load_path=optimizer_dir)
        optim_state_dict = {"optim_state": optimizer_state}
        try:
            dcp.load(state_dict=optim_state_dict, checkpoint_id=str(optimizer_dir))
            logger.info(f"Loaded optimizer from {optimizer_dir}")
        except Exception as e:
            # The model is already restored but the fp32 masters still hold their pre-load
            # values, and the first step would copy those back over the loaded weights.
            raise RuntimeError(
                f"Failed to load optimizer from {optimizer_dir}: {e}. The model weights were "
                "restored, so continuing would let stale fp32 master params overwrite them on "
                "the first optimizer step. Set training.continual_training=true to start a "
                "fresh optimizer from these weights."
            ) from e
    elif load_optimizer:
        raise RuntimeError(
            f"No optimizer checkpoint at {optimizer_dir}. Resuming without it would leave the "
            "fp32 master params holding their pre-load values, which the first optimizer step "
            "would copy over the restored weights. Set training.continual_training=true to "
            "start a fresh optimizer from these weights."
        )

    # Exact resume requires the scheduler state; continual training starts fresh.
    if not continual_training and getattr(actor, "lr_scheduler", None) is None:
        raise RuntimeError("Exact resume requires an initialized LR scheduler")
    load_lr_scheduler = not continual_training
    if load_lr_scheduler:
        if not (lr_scheduler_dir / ".metadata").is_file():
            raise RuntimeError(
                f"No LR scheduler checkpoint at {lr_scheduler_dir}. Exact resume requires "
                "model, optimizer, LR scheduler, RNG, and committed-work state."
            )
        lr_scheduler_state = LRSchedulerState(actor.lr_scheduler)
        lr_scheduler_state_dict = {"lr_scheduler_state": lr_scheduler_state}
        try:
            dcp.load(state_dict=lr_scheduler_state_dict, checkpoint_id=str(lr_scheduler_dir))
            logger.info(f"Loaded LR scheduler from {lr_scheduler_dir}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load LR scheduler from {lr_scheduler_dir}: {e}"
            ) from e

    rng_state = None
    if not continual_training:
        world_size = int(metadata["world_size"])
        rank = (
            dist.get_rank()
            if dist.is_available() and dist.is_initialized()
            else int(getattr(actor.args, "rank", 0))
        )
        rng_state = _load_rank_rng_state(
            checkpoint_dir,
            rank=rank,
            world_size=world_size,
        )

    return {
        "rng": rng_state,
        "metadata": metadata,
        "iteration": target_step,
        "optimizer_dir": optimizer_dir,
    }


def _sync_continual_training_master_params(actor: Any) -> None:
    """Initialize a new stage's FP32 masters from its loaded model weights.

    Continual training is intentionally a weights-only stage transition.  Its
    trainable parameter set may differ from the checkpoint that produced those
    weights (for example CSA dense -> indexer warmup -> sparse training), so the
    previous optimizer checkpoint is neither required nor safe to reinterpret.
    """
    optimizer = getattr(actor, "optimizer", None)
    if optimizer is None:
        return
    sync = getattr(optimizer, "sync_fp32_params_from_model", None)
    if sync is not None:
        sync()
        logger.info("Initialized continual-training fp32 masters from loaded model weights")


def finalize_load(actor: Any, checkpoint_payload: dict[str, Any] | None) -> None:
    if checkpoint_payload is None:
        dist.barrier()
        return

    continual_training = getattr(actor.args, "continual_training", False)

    if not continual_training:
        rng_state = checkpoint_payload.get("rng")
        _validate_rng_state(
            rng_state,
            legacy=isinstance(rng_state, dict) and "schema_version" not in rng_state,
        )
        torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and rng_state.get("cuda") is not None:
            cuda_state = rng_state["cuda"]
            if isinstance(cuda_state, (list, tuple)):
                # Single-rank legacy checkpoints used get_rng_state_all().
                torch.cuda.set_rng_state_all(cuda_state)
            else:
                torch.cuda.set_rng_state(cuda_state)

    metadata = checkpoint_payload.get("metadata") or {}
    iteration = checkpoint_payload.get("iteration")
    if metadata and not continual_training:
        actor.global_step = int(metadata.get("global_step", actor.global_step))
        next_step = metadata.get("next_step") or metadata.get("next_inference_id")
        if next_step is not None:
            actor.args.start_step = next_step
    elif iteration is not None and not continual_training:
        if getattr(actor.args, "start_step", None) is None:
            actor.args.start_step = iteration

    if continual_training:
        _sync_continual_training_master_params(actor)

    torch.cuda.synchronize()
    dist.barrier()


def save(actor: Any, step: int) -> None:
    """Save checkpoint to disk.

    Saves model weights and optimizer state to separate directories.
    This allows loading weights without optimizer or deleting optimizer before loading.
    """
    torch.cuda.synchronize()

    base_dir = Path(actor.args.checkpoint_dir).expanduser()
    step_id = step + 1
    checkpoint_dir = base_dir / f".iter_{step_id:07d}.incomplete"
    model_dir = checkpoint_dir / "model"
    optimizer_dir = checkpoint_dir / "optimizer"
    lr_scheduler_dir = checkpoint_dir / "lr_scheduler"

    if dist.get_rank() == 0 or getattr(actor.args, "cluster_storage_mode", "shared") == "node_local":
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        optimizer_dir.mkdir(parents=True, exist_ok=True)
        lr_scheduler_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "rng").mkdir(parents=True, exist_ok=True)
    dist.barrier()

    # Save model weights
    model_state = ModelState(actor.model)
    state_dict = {"model_state": model_state}
    dcp.save(state_dict, checkpoint_id=str(model_dir))

    if hasattr(actor, "optimizer") and actor.optimizer is not None:
        optimizer_state = OptimizerState(actor.model, actor.optimizer)
        optim_state_dict = {"optim_state": optimizer_state}
        dcp.save(optim_state_dict, checkpoint_id=str(optimizer_dir))

    if hasattr(actor, "lr_scheduler") and actor.lr_scheduler is not None:
        lr_scheduler_state = LRSchedulerState(actor.lr_scheduler)
        lr_scheduler_state_dict = {"lr_scheduler_state": lr_scheduler_state}
        dcp.save(lr_scheduler_state_dict, checkpoint_id=str(lr_scheduler_dir))

    _save_rank_rng_state(
        checkpoint_dir,
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
    )

    if dist.get_rank() == 0:
        metadata = {
            "iteration": step_id,
            "step": step,
            "inference_id": step,  # compat: old checkpoints use this key
            "next_step": step + 1,
            "next_inference_id": step + 1,  # compat: old checkpoints use this key
            "global_step": actor.global_step,
            "world_size": dist.get_world_size(),
            "training_topology": _logical_training_topology(
                actor.args, dist.get_world_size()
            ),
            "timestamp": time.time(),
        }
        _write_checkpoint_metadata(checkpoint_dir / "meta.json", metadata)
        config = getattr(actor.args, "draft_model_config_obj", None)
        if config is not None:
            raw = config if isinstance(config, dict) else config.to_dict()
            _atomic_write_text(checkpoint_dir / "draft_config.json", json.dumps(raw, indent=2) + "\n")

        logger.info(f"Staged distributed checkpoint at {checkpoint_dir}")

    dist.barrier()
