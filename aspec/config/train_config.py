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

import argparse
import copy
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf

from aspec.config.draft_execution import (
    is_direct_cp_backend,
    resolve_draft_execution_capability,
)
from aspec.config.inference_config import InferenceConfig, _validate_inference_batch_config
from aspec.config.mooncake_config import resolve_rdma_routes
from aspec.inference.concurrency import direct_output_sample_capacity
from aspec.utils.logging import logger


@dataclass
class DatasetConfig:
    chat_template: Optional[str] = "llama3"
    defer_tokenization: bool = False
    eval_data_path: Optional[str] = None
    eval_interval: int = 50
    eval_cache_idle_timeout_seconds: float = 300.0
    eval_micro_batch_size: Optional[int] = None
    eval_prompt_key: Optional[str] = None
    last_turn_loss_only: Any = "auto"  # bool or "auto"
    length_group_size: int = 32
    min_loss_tokens: int = 0  # DFlash: skip sequences with < N supervised tokens (use 2*block_size)
    prompt_key: str = "conversations"
    renderer: Optional[str] = None  # registered ConversationRenderer; overrides chat_template
    shuffle_dataset: bool = True
    dataset_seed: Optional[int] = None  # Defaults to the training seed for single-Draft runs.
    train_data_path: Optional[str] = ""


@dataclass
class DebugConfig:
    enable_perf_metrics: bool = True
    max_dump_steps: int = 5
    memory_recorder: str = "torch"
    memory_snapshot_dir: str = "."
    memory_snapshot_num_steps: Optional[int] = None
    memory_snapshot_path: str = ""
    profile_dir_name: Optional[str] = "/tmp/aspec_profiles"
    profile_step_end: int = 0
    profile_step_start: int = 0
    profile_target: list = field(default_factory=lambda: ["train_overall"])
    performance_drain_steps: int = 0
    performance_measurement_required: bool = False
    performance_warmup_steps: int = 0
    topology_producer_profile_samples: int = 0
    topology_producer_profile_warmup_samples: int = 0
    record_memory_history: bool = False
    save_debug_train_data: Optional[str] = None
    timeline_enabled: bool = False
    timeline_flush_every: int = 128
    timeline_output_dir: str = ""
    timeline_run_id: str = ""
    use_pytorch_profiler: bool = False


@dataclass
class LoggingConfig:
    report_to: str = "none"
    use_tensorboard: bool = False
    use_wandb: bool = False
    wandb_dir: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_host: Optional[str] = None
    wandb_key: Optional[str] = None
    wandb_mode: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_random_suffix: bool = True
    wandb_run_id: Optional[str] = None
    wandb_team: Optional[str] = None


@dataclass
class ClusterConfig:
    """Execution-cluster contract for production runs.

    ``static_multi_node`` is deliberately narrow: membership is expressed by
    the existing role node IP lists, hardware must be homogeneous, and a
    storage mode declares shared storage or explicit local checkpoint exchange.
    """

    mode: str = "single_node"
    shared_root: Optional[str] = None
    storage_mode: str = "shared"
    file_transfer_timeout_seconds: float = 600.0
    require_homogeneous_gpus: bool = True
    preflight_timeout_seconds: float = 120.0


@dataclass
class ModelConfig:
    draft_model_config: Optional[str] = None
    initial_draft_model_path: Optional[str] = None
    # Product-supported native training context, which can be narrower than a
    # backend's max_position_embeddings allocation.
    native_context_length: Optional[int] = None
    keep_initial_vocab_mapping: bool = False
    embedding_key: str = "model.embed_tokens.weight"
    freeze_lm_head: bool = False
    lm_head_key: str = "lm_head.weight"
    norm_key: str = "model.norm.weight"
    target_model_backend: str = "sglang"
    target_model_path: str = ""
    trust_remote_code: bool = False


@dataclass
class TrainingConfig:
    attention_backend: str = "sdpa"
    # Attention backend and context parallelism are independent execution
    # dimensions. CP>1 currently requires the FA4 backend.
    draft_context_parallel_size: int = 1
    colocate: bool = False
    continual_training: bool = False
    # A weights-only phase transition normally starts a fresh data stream. Set
    # this only when the new phase must continue the predecessor's committed
    # source frontier while still resetting optimizer/LR/RNG state.
    continue_source_cursor: bool = False
    distributed_backend: str = "nccl"
    distributed_timeout_minutes: int = 10
    draft_accumulation_steps: int = 1
    mtp_steps: int = 3
    mtp_step_weight_beta: float = 0.6
    eagle3_num_anchors: int = 0
    eagle3_anchor_max_gap: Optional[int] = None
    fsdp_reduce_dtype: str = "float32"  # "float32" or "bfloat16"
    fsdp_strategy: str = "REPLICATE"
    # Controls which workload claims head-node GPUs first under PACK strategy.
    # "training_first" (default), "inference_first", or "custom".
    placement_strategy: str = "training_first"
    training_node_ips: Optional[list[str]] = None
    training_node_selectors: Optional[list[dict[str, str]]] = None
    compile_model: bool = False  # torch.compile the full training model
    sp_ring_size: int = 1
    sp_ulysses_size: int = 1

    gradient_checkpointing: bool = False
    # Checkpoint EAGLE3's full-vocabulary loss independently from the backbone.
    # None preserves the existing backbone policy. Disabling this spends
    # activation memory to avoid loss recomputation.
    eagle3_loss_gradient_checkpointing: Optional[bool] = None
    learning_rate: float = 1e-4
    lk_eta: float = 3.0
    load_path: Optional[str] = None
    loss_type: str = "forward_kl"  # "forward_kl", "lk_alpha", or "lk_lambda" (Eagle3 only)
    # EAGLE3 Target distribution materialization. ``logits`` projects the
    # frozen full-vocabulary teacher once per microbatch and reuses its BF16
    # logits across recurrent depths; FP32 softmax/loss computation is unchanged.
    eagle3_target_cache: str = "lazy"  # "lazy" or "logits"
    lr_decay_style: str = "cosine"
    lr_wsd_decay_ratio: float = 0.2
    lr_wsd_decay_style: str = "cosine"
    lr_total_steps: Optional[int] = None
    max_concurrent_batches: int = 1
    max_grad_norm: float = 0.5
    max_seq_length: int = 8192
    min_lr: float = 0.0
    weight_decay: float = 0.0
    num_epochs: int = 10
    num_train_steps: Optional[int] = None
    # Optional semantic-work limit for a finite source pass. Unlike
    # num_train_steps, this counts source documents before packing. The Plan
    # deterministically drops an incomplete fixed-row optimizer batch. Packing
    # first schedules a DP-aligned upper bound, then reports any additional
    # document tail that cannot form a complete optimizer step at runtime.
    source_work_budget: Optional[int] = None
    micro_batch_size: int = 2
    # Bounds one optimizer step, including source supply and native tensor transfer.
    # A timeout terminates the owning trainer processes rather than trusting a
    # blocked native call to return to Python.
    optimizer_step_timeout_seconds: float = 1800.0
    # Stream this step's producer/consumer work, and optionally dispatch a
    # bounded prefix of the following step while the trainer is computing.
    streaming_dispatch_with_training: bool = False
    pipeline_dispatch_with_training: bool = False
    prefetch_depth: int = 2  # 0 = disabled, >0 = async pre-fetch N batches ahead
    save_interval: int = 5000
    save_per_epoch: bool = False
    save_final_checkpoint: bool = True
    max_checkpoints: int = 0  # 0 = keep all, N > 0 = rotate and keep only N most recent
    seed: int = 0
    # Debug/replay only: separate stateless anchor RNG; normal sampling is unchanged.
    deterministic_anchor_sampling: bool = False
    # Anchor budget per physical row (legacy) or original source sample.
    anchor_sampling_scope: str = "row"
    train_backend: str = "fsdp"
    train_env_vars: str = "{}"
    train_with_decode: bool = False
    training_num_gpus_per_node: int = 1
    training_num_nodes: int = 1
    ttt_length: int = 7
    # Per-position TTT loss weights. If unset, defaults to [0.8**i for i in range(ttt_length)].
    # Length must equal ttt_length when supplied.
    ploss_weights: Optional[list[float]] = None
    warmup_ratio: float = 0.015

    # WSD LR schedule parameters (used by DFlash trainer only)
    wsd_decay_ratio: Optional[float] = None
    wsd_decay_style: Optional[str] = None

    # DFlash-specific parameters (ignored for Eagle3 training)
    dflash_block_size: int = 16
    dflash_dpace_alpha: float = 0.5
    dflash_loss_decay_gamma: float = 7.0
    dflash_loss_objective: str = "decay"  # "decay" or "dpace"
    dflash_ce_loss_alpha: float = 1.0
    dflash_l1_loss_alpha: float = 0.0
    dflash_num_anchors: int = 512
    dflash_num_target_layers: int = 5
    # Pack multiple complete documents into one fixed row. The controller owns
    # grouping; the trainer preserves document boundaries for attention/RoPE.
    dflash_packing: bool = False
    packing_min_samples_per_dispatch: int = 0  # 0 = one seed per DP rank

    # DFlash2-specific parameters (used by DFlash2 trainer only)
    dflash2_selector_loss_alpha: float = 1.0

    # DSpark-specific parameters (used by DSpark trainer only)
    dspark_num_anchors: int = 512
    dspark_num_target_layers: int = 5
    dspark_loss_decay_gamma: float = 4.0
    dspark_ce_loss_alpha: float = 0.1
    dspark_l1_loss_alpha: float = 0.9
    # Independent sampling transforms for the DSpark L1 distributions.
    dspark_l1_target_temperature: float = 1.0
    dspark_l1_target_top_p: float = 1.0
    dspark_l1_target_top_k: int = -1
    dspark_l1_draft_temperature: float = 1.0
    dspark_l1_draft_top_p: float = 1.0
    dspark_l1_draft_top_k: int = -1
    dspark_confidence_head_alpha: float = 1.0


@dataclass
class DecodeConfig:
    """Config for train-with-decode mode (speculative decoding during training)."""

    cuda_graph_max_bs: Optional[int] = None
    max_new_tokens: int = 512
    min_new_tokens: int = 2
    stop_token_ids: Optional[list[int]] = None
    max_running_requests: Optional[int] = None
    speculative_algorithm: Optional[str] = None
    speculative_draft_model_path: Optional[str] = None
    speculative_eagle_topk: Optional[int] = None
    speculative_num_draft_tokens: Optional[int] = None
    speculative_num_steps: Optional[int] = None
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    weight_sync_enabled: bool = False
    weight_sync_interval: int = 500


@dataclass
class Config:
    experiments: list[dict[str, Any]] = field(default_factory=list)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mooncake: dict[str, Any] = field(default_factory=dict)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    cache_dir: str = "./cache"
    cache_key: Optional[str] = None
    model_download_dir: Optional[str] = None
    output_dir: str = ""


_ALWAYS_LOCAL_PATH_KEYS = (
    "cluster.shared_root",
    "output_dir",
    "cache_dir",
    "model_download_dir",
    "inference.offline.data_path",
    "model.draft_model_config",
    "model.initial_draft_model_path",
)
_DATA_PATH_KEYS = ("dataset.train_data_path", "dataset.eval_data_path")


def _resolve_relative_paths(
    config: DictConfig,
    base_dir: str,
    *,
    skip_keys: frozenset[str] = frozenset(),
) -> None:
    """Resolve local relative paths in *config* against *base_dir* (in-place).

    Always-local keys (output_dir, cache_dir, …) are absolutized unconditionally.
    Data-path keys are only absolutized when ``is_local_data_path`` says they look
    like filesystem paths (as opposed to HF Hub dataset IDs).

    Keys listed in *skip_keys* are left untouched (useful for deferring
    CWD-relative keys when resolving a file-level config).
    """
    # Keep schema import independent of runtime data/worker imports.
    from aspec.data.utils import is_local_data_path

    for dotted_key in (*_ALWAYS_LOCAL_PATH_KEYS, *_DATA_PATH_KEYS):
        if dotted_key in skip_keys:
            continue
        val = OmegaConf.select(config, dotted_key, default=None)
        if not (isinstance(val, str) and val):
            continue

        expanded = os.path.expanduser(val)
        if os.path.isabs(expanded):
            if expanded != val:
                OmegaConf.update(config, dotted_key, expanded)
            continue

        if dotted_key in _ALWAYS_LOCAL_PATH_KEYS or is_local_data_path(expanded, base_dir=base_dir):
            OmegaConf.update(config, dotted_key, os.path.abspath(os.path.join(base_dir, expanded)))


def _validate_vllm_config(config: DictConfig) -> None:
    """Raise if the vllm backend is selected with unsupported feature flags."""
    if config.model.target_model_backend != "vllm":
        return
    unsupported_flags = {
        "inference.vllm.enable_multimodal": "enable_multimodal",
        "training.train_with_decode": "train_with_decode",
    }
    for key, label in unsupported_flags.items():
        if OmegaConf.select(config, key):
            raise NotImplementedError(f"{label} is not yet supported with the vllm backend!")


def _validate_current_scope(config: DictConfig) -> None:
    """Reject backend selections outside ASpec's current executable path."""
    engine_type = config.inference.inference_engine_type
    target_backend = config.model.target_model_backend
    if engine_type not in {"sgl", "offline"} or target_backend != "sglang":
        raise NotImplementedError(
            "ASpec currently supports inference.inference_engine_type=sgl or offline with "
            "model.target_model_backend=sglang; "
            f"got engine_type={engine_type!r}, target_backend={target_backend!r}"
        )


def _validate_cluster_config(config: DictConfig) -> None:
    mode = config.cluster.mode
    if mode not in {"single_node", "static_multi_node"}:
        raise ValueError(
            "cluster.mode must be 'single_node' or 'static_multi_node', "
            f"got {mode!r}"
        )
    for name in ("preflight_timeout_seconds", "file_transfer_timeout_seconds"):
        value = getattr(config.cluster, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"cluster.{name} must be finite and positive")
    feature_affinity = str(
        OmegaConf.select(config, "mooncake.feature_affinity", default="any")
    ).lower()
    if feature_affinity not in {"any", "node_local_required"}:
        raise ValueError(
            "mooncake.feature_affinity must be 'any' or 'node_local_required'"
        )
    if mode == "single_node":
        if feature_affinity == "node_local_required":
            raise ValueError(
                "mooncake.feature_affinity=node_local_required is a static multi-node "
                "placement contract; single-node strict NVLink does not require it"
            )
        return

    if config.training.placement_strategy != "custom":
        raise ValueError(
            "cluster.mode=static_multi_node requires "
            "training.placement_strategy=custom"
        )
    if config.training.colocate:
        raise ValueError("static_multi_node does not yet support colocate=true")
    if config.cluster.storage_mode not in {"shared", "node_local"}:
        raise ValueError("cluster.storage_mode must be shared or node_local")
    shared_root = None
    if config.cluster.storage_mode == "node_local" and config.training.train_with_decode:
        raise ValueError("node_local storage currently supports prefill-only training, not scratch serving assets")
    if config.cluster.storage_mode == "shared":
        if not config.cluster.shared_root:
            raise ValueError("shared storage requires cluster.shared_root")
        if not os.path.isabs(str(config.cluster.shared_root)):
            raise ValueError("cluster.shared_root must resolve to an absolute path")
        shared_root = Path(str(config.cluster.shared_root)).resolve()
    shared_path_fields = {
        "output_dir": config.output_dir,
        "model.target_model_path": config.model.target_model_path,
        "model.draft_model_config": config.model.draft_model_config,
        "dataset.train_data_path": config.dataset.train_data_path,
    }
    if config.model.initial_draft_model_path:
        shared_path_fields["model.initial_draft_model_path"] = config.model.initial_draft_model_path
    if config.dataset.eval_data_path:
        shared_path_fields["dataset.eval_data_path"] = config.dataset.eval_data_path
    if config.training.load_path:
        shared_path_fields["training.load_path"] = config.training.load_path
    for field_name, value in shared_path_fields.items():
        if not isinstance(value, str) or not value or not os.path.isabs(value):
            raise ValueError(
                f"static_multi_node requires {field_name} to be an absolute path available on every node"
            )
        if shared_root is None:
            continue
        try:
            Path(value).resolve().relative_to(shared_root)
        except ValueError as exc:
            raise ValueError(
                f"static_multi_node requires {field_name} under cluster.shared_root "
                f"({shared_root})"
            ) from exc

    role_ips = {
        "training": config.training.training_node_ips,
        "inference": config.inference.inference_node_ips,
    }
    if config.training.training_node_selectors or config.inference.inference_node_selectors:
        raise ValueError(
            "static_multi_node initially requires explicit training_node_ips and "
            "inference_node_ips; label selectors are not deterministic enough for preflight"
        )
    for role, values in role_ips.items():
        if not isinstance(values, (list, tuple, ListConfig)) or not values:
            raise ValueError(f"static_multi_node requires explicit {role}_node_ips")
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"{role}_node_ips entries must be non-empty strings")
        if len(set(values)) != len(values):
            raise ValueError(f"{role}_node_ips must not contain duplicates")
    if len(set(role_ips["training"]) | set(role_ips["inference"])) < 2:
        raise ValueError("static_multi_node requires at least two distinct node IPs")

    if config.inference.sglang.nnodes != 1:
        raise ValueError(
            "static_multi_node first release keeps every Target TP group node-local; "
            "inference.sglang.nnodes must be 1"
        )
    if (
        config.inference.inference_num_gpus_per_engine
        > config.inference.inference_num_gpus_per_node
    ):
        raise ValueError(
            "node-local Target TP requires inference_num_gpus_per_engine <= "
            "inference_num_gpus_per_node"
        )
    remaining_inference_gpus = int(config.inference.inference_num_gpus)
    for node_ip in role_ips["inference"]:
        allocated = min(
            int(config.inference.inference_num_gpus_per_node),
            remaining_inference_gpus,
        )
        if allocated % int(config.inference.inference_num_gpus_per_engine):
            raise ValueError(
                "every node-local Target allocation must contain complete TP groups: "
                f"node={node_ip!r}, allocated_gpus={allocated}, "
                "gpus_per_engine="
                f"{config.inference.inference_num_gpus_per_engine}"
            )
        remaining_inference_gpus -= allocated

    mooncake = config.mooncake
    data_plane = OmegaConf.select(mooncake, "data_plane", default="store")
    enable_gpu_direct = bool(
        OmegaConf.select(mooncake, "enable_gpu_direct", default=False)
    )
    allow_host_fallback = bool(
        OmegaConf.select(mooncake, "allow_host_fallback", default=True)
    )
    protocol = OmegaConf.select(mooncake, "protocol", default="tcp")
    node_local_features = feature_affinity == "node_local_required"
    if node_local_features:
        if set(role_ips["training"]) != set(role_ips["inference"]):
            raise ValueError(
                "node_local_required feature affinity requires training_node_ips and "
                "inference_node_ips to select the same nodes"
            )
        cp_size = int(config.training.draft_context_parallel_size)
        if config.training.training_num_gpus_per_node % cp_size:
            raise ValueError(
                "node_local_required feature affinity requires every Draft CP group "
                "to remain within one node"
            )
        ready_watermark = int(config.inference.max_sample_pool_size)
        if 0 < ready_watermark < len(role_ips["training"]):
            raise ValueError(
                "node_local_required bounded ready pool needs at least one credit "
                "per locality domain"
            )
        if (
            not config.training.dflash_packing
            and config.training.num_train_steps is not None
        ):
            logical_dp_size = (
                config.training.training_num_nodes
                * config.training.training_num_gpus_per_node
                // cp_size
            )
            global_batch_size = (
                config.training.micro_batch_size
                * logical_dp_size
                * config.training.draft_accumulation_steps
            )
            total_samples = config.training.num_train_steps * global_batch_size
            node_batch_width = (
                len(role_ips["inference"])
                * config.inference.inference_batch_size
            )
            if total_samples % node_batch_width:
                raise ValueError(
                    "node_local_required fixed-row sample budget must divide into "
                    "complete Target batches on every node: "
                    f"total_samples={total_samples}, nodes={len(role_ips['inference'])}, "
                    f"inference_batch_size={config.inference.inference_batch_size}"
                )
    if data_plane == "transfer_engine":
        if not enable_gpu_direct or allow_host_fallback:
            raise ValueError(
                "static_multi_node TransferEngine requires enable_gpu_direct=true "
                "and allow_host_fallback=false"
            )
        allowed_protocols = (
            {"tcp", "rdma", "nvlink_intra"}
            if node_local_features
            else {"tcp", "rdma"}
        )
        if protocol not in allowed_protocols:
            raise ValueError(
                "static_multi_node TransferEngine protocol is incompatible with "
                f"feature_affinity={feature_affinity!r}: {protocol!r}"
            )
    elif data_plane == "store":
        if protocol != "tcp" or enable_gpu_direct:
            raise ValueError(
                "static_multi_node host-staged Store requires protocol='tcp' and "
                "enable_gpu_direct=false"
            )
        if allow_host_fallback:
            raise ValueError(
                "the explicit static_multi_node host-staged Store path is selected, "
                "not a fallback; mooncake.allow_host_fallback must be false"
            )
        pool_size = OmegaConf.select(
            mooncake,
            "async_put_pool_size",
            default=config.inference.inference_batch_size,
        )
        if pool_size is not None and (type(pool_size) is not int or pool_size < 1):
            raise ValueError("host-staged Store requires async_put_pool_size >= 1")
    else:
        raise ValueError(
            "static_multi_node requires mooncake.data_plane='store' or "
            f"'transfer_engine', got {data_plane!r}"
        )
    require_nvlink = bool(
        OmegaConf.select(mooncake, "require_nvlink", default=False)
    )
    if node_local_features:
        if (
            data_plane != "transfer_engine"
            or protocol != "nvlink_intra"
            or not require_nvlink
        ):
            raise ValueError(
                "node_local_required feature affinity requires strict "
                "transfer_engine/nvlink_intra with require_nvlink=true"
            )
    elif require_nvlink:
        raise ValueError(
            "static_multi_node may require intra-node NVLink only with "
            "feature_affinity=node_local_required"
        )
    device_name = OmegaConf.select(mooncake, "device_name", default="")
    rdma_port = OmegaConf.select(mooncake, "rdma_port", default=1)
    rdma_gid_index = OmegaConf.select(mooncake, "rdma_gid_index", default=None)
    rdma_routes = OmegaConf.select(mooncake, "rdma_routes", default=None)
    if isinstance(rdma_routes, DictConfig):
        rdma_routes = OmegaConf.to_container(rdma_routes, resolve=True)
    if protocol == "rdma":
        resolve_rdma_routes(
            sorted(set(role_ips["training"]) | set(role_ips["inference"])),
            device_name=device_name,
            port=rdma_port,
            gid_index=rdma_gid_index,
            overrides=rdma_routes,
        )
    if protocol == "tcp" and (device_name or rdma_routes):
        raise ValueError(
            "static_multi_node TCP requires mooncake.device_name and rdma_routes "
            "to be empty"
        )


def _validate_vocab_mapping_config(config: DictConfig) -> None:
    """Reusing a loaded vocabulary mapping requires weights that carry one."""
    if not OmegaConf.select(config, "model.keep_initial_vocab_mapping"):
        return
    if not (
        OmegaConf.select(config, "model.initial_draft_model_path")
        or OmegaConf.select(config, "training.load_path")
    ):
        raise ValueError(
            "model.keep_initial_vocab_mapping requires model.initial_draft_model_path or "
            "training.load_path — without loaded weights there is no mapping to keep, and the "
            "draft lm_head would be trained against an all-pass t2d."
        )


def _validate_offline_config(config: DictConfig) -> None:
    if config.inference.inference_engine_type == "offline":
        if config.dataset.renderer:
            raise ValueError("Offline replay uses recorded loss masks; dataset.renderer must be null")
        if config.cluster.mode != "single_node" or config.experiments:
            raise ValueError("Offline replay currently supports single-node, single-Draft training")
        if config.inference.inference_num_gpus != 0:
            raise ValueError("Offline replay requires inference.inference_num_gpus=0")
        if (OmegaConf.select(config, "mooncake.data_plane", default="store") != "store"
                or OmegaConf.select(config, "mooncake.protocol", default="tcp") != "tcp"
                or OmegaConf.select(config, "mooncake.enable_gpu_direct", default=False)):
            raise ValueError("Offline replay requires Mooncake CPU store/tcp transport")
        if config.inference.target_prefill_memory_optimized:
            raise ValueError("Offline replay requires target_prefill_memory_optimized=false")
        if config.training.draft_context_parallel_size != 1:
            raise ValueError("Offline replay currently requires draft_context_parallel_size=1")
        if not config.inference.offline.data_path:
            raise ValueError(
                "inference.offline.data_path is required when "
                "inference.inference_engine_type=offline"
            )
        if config.training.train_with_decode:
            raise ValueError("training.train_with_decode is not supported in offline mode")
        if config.training.attention_backend == "usp":
            raise ValueError("training.attention_backend=usp is not supported offline")
        if config.inference.offline.num_engines <= 0:
            raise ValueError("inference.offline.num_engines must be positive")


def _validate_training_batch_config(config: DictConfig) -> None:
    if config.training.anchor_sampling_scope not in {"row", "sample"}:
        raise ValueError("anchor_sampling_scope must be row or sample")
    if config.training.micro_batch_size < 1:
        raise ValueError(
            "training.micro_batch_size must be positive (>= 1): a value of 0 yields "
            "dispatch_batch_size=0, so try_dispatch_batch no-op-dispatches and every rank "
            "blocks on the data queue, surfacing as an NCCL all-gather timeout"
        )
    if config.training.dflash_packing:
        if config.training.micro_batch_size != 1:
            raise ValueError("dflash_packing requires training.micro_batch_size=1")
        if config.training.attention_backend == "usp":
            raise ValueError("dflash_packing is not supported with USP")
        if config.training.max_seq_length <= 0:
            raise ValueError("dflash_packing requires training.max_seq_length > 0")
        if config.training.packing_min_samples_per_dispatch < 0:
            raise ValueError("packing_min_samples_per_dispatch must be non-negative")
    if (
        config.training.source_work_budget is not None
        and config.training.source_work_budget <= 0
    ):
        raise ValueError("source_work_budget must be positive when configured")


def _validate_training_numeric_config(config: DictConfig) -> None:
    """Reject non-positive values that would otherwise fail silently or crash late.

    These fields share the fail-closed-at-load principle of #171: reject a
    configuration that would produce flat loss, sign-flipped gradients, or an
    opaque post-init crash before expensive Ray/mooncake initialization.
    """
    if config.training.draft_accumulation_steps <= 0:
        raise ValueError(
            f"draft_accumulation_steps must be > 0 (got {config.training.draft_accumulation_steps}); "
            f"<=0 propagates into global_batch_size/lr_total_steps and crashes post-init "
            f"(ZeroDivisionError or invalid scheduler total_steps)"
        )
    if not math.isfinite(config.training.learning_rate) or config.training.learning_rate <= 0:
        raise ValueError(
            f"learning_rate must be finite and > 0 (got {config.training.learning_rate}); "
            f"0 yields silent flat loss (untrained checkpoint), <0 is invalid for the scheduler"
        )
    if not math.isfinite(config.training.min_lr) or not 0 <= config.training.min_lr <= config.training.learning_rate:
        raise ValueError(
            "min_lr must be in [0, learning_rate], got "
            f"min_lr={config.training.min_lr}, learning_rate={config.training.learning_rate}"
        )
    if not 0 <= config.training.warmup_ratio < 1:
        raise ValueError(
            f"warmup_ratio must be in [0, 1), got {config.training.warmup_ratio}"
        )
    for name in ("num_train_steps", "lr_total_steps"):
        value = getattr(config.training, name)
        if value is not None and value <= 0:
            raise ValueError(f"training.{name} must be positive when configured, got {value}")
    if not math.isfinite(config.training.max_grad_norm) or config.training.max_grad_norm <= 0:
        raise ValueError(
            f"max_grad_norm must be finite and > 0 (got {config.training.max_grad_norm}); "
            f"0 zeroes all grads (silent flat loss), <0 sign-flips grads (silent gradient ascent)"
        )
    if (not math.isfinite(config.dataset.eval_cache_idle_timeout_seconds)
            or config.dataset.eval_cache_idle_timeout_seconds <= 0):
        raise ValueError("eval_cache_idle_timeout_seconds must be finite and > 0")
    if not math.isfinite(config.training.optimizer_step_timeout_seconds) or config.training.optimizer_step_timeout_seconds <= 0:
        raise ValueError(
            "optimizer_step_timeout_seconds must be finite and > 0 so a blocked trainer/native "
            "transfer has a bounded owner-process lifetime"
        )

    if config.training.lr_decay_style == "WSD":
        for name in ("lr_wsd_decay_style", "wsd_decay_style"):
            value = getattr(config.training, name)
            if value is not None and value not in {"linear", "cosine", "exponential", "minus_sqrt"}:
                raise ValueError(f"training.{name} has an unsupported WSD decay style: {value}")
        for name in ("lr_wsd_decay_ratio", "wsd_decay_ratio"):
            value = getattr(config.training, name)
            if value is not None and not 0 < value <= 1:
                raise ValueError(f"training.{name} must be in (0, 1]")

    warmup_steps = config.debug.performance_warmup_steps
    drain_steps = config.debug.performance_drain_steps
    if warmup_steps < 0 or drain_steps < 0:
        raise ValueError(
            "debug.performance_warmup_steps and performance_drain_steps must be non-negative"
        )
    if config.debug.timeline_flush_every < 1:
        raise ValueError("debug.timeline_flush_every must be positive")
    producer_samples = config.debug.topology_producer_profile_samples
    producer_warmup = config.debug.topology_producer_profile_warmup_samples
    if producer_samples < 0 or producer_warmup < 0:
        raise ValueError(
            "debug.topology_producer_profile_samples and "
            "topology_producer_profile_warmup_samples must be non-negative"
        )
    if producer_samples == 0 and producer_warmup:
        raise ValueError(
            "debug.topology_producer_profile_warmup_samples requires a non-zero "
            "topology_producer_profile_samples"
        )
    if producer_samples and producer_warmup >= producer_samples:
        raise ValueError(
            "debug.topology_producer_profile_warmup_samples must be smaller than "
            "topology_producer_profile_samples"
        )
    if config.debug.performance_measurement_required:
        num_steps = config.training.num_train_steps
        if not isinstance(num_steps, int) or num_steps <= warmup_steps + drain_steps:
            raise ValueError(
                "required performance measurement needs explicit training.num_train_steps greater "
                "than debug.performance_warmup_steps + performance_drain_steps"
            )


def _validate_execution_capabilities(config: DictConfig, *, source_context_parallel_size=None) -> None:
    """Resolve optional execution dimensions before expensive runtime setup."""

    draft_execution = resolve_draft_execution_capability(
        config.training.attention_backend,
        config.training.draft_context_parallel_size,
    )
    if source_context_parallel_size is None:
        # Shared sessions stage one full source, even when CP is inherited
        # from the common config. Children pass the same producer layout.
        source_context_parallel_size = (
            1 if config.experiments else draft_execution.context_parallel_size
        )
    data_plane = OmegaConf.select(config, "mooncake.data_plane", default="store")
    if data_plane == "transfer_engine" and not config.training.streaming_dispatch_with_training:
        raise ValueError(
            "mooncake.data_plane=transfer_engine requires "
            "training.streaming_dispatch_with_training=true so bounded GPU rings "
            "have a live consumer while inference publishes"
        )

    if data_plane == "transfer_engine":
        common_slots = int(
            OmegaConf.select(config, "mooncake.gpu_ring_slots", default=2)
        )
        source_slots = OmegaConf.select(
            config, "mooncake.source_gpu_ring_slots", default=None
        )
        destination_slots = OmegaConf.select(
            config, "mooncake.destination_gpu_ring_slots", default=None
        )
        source_slots = common_slots if source_slots is None else int(source_slots)
        destination_slots = (
            common_slots if destination_slots is None else int(destination_slots)
        )
        context_parallel_size = draft_execution.context_parallel_size
        required_source_slots = config.inference.inference_batch_size * source_context_parallel_size
        required_destination_slots = (
            config.training.micro_batch_size if context_parallel_size > 1 else 1
        )
        if source_slots < required_source_slots:
            raise ValueError(
                "transfer_engine source GPU ring is too shallow: "
                f"source_gpu_ring_slots={source_slots}, require at least "
                f"{required_source_slots}"
            )
        if destination_slots < required_destination_slots:
            raise ValueError(
                "transfer_engine destination GPU ring is too shallow: "
                f"destination_gpu_ring_slots={destination_slots}, require at least "
                f"{required_destination_slots}"
            )

    if draft_execution.direct_cp_backend:
        # Anchor divisibility belongs to block-parallel algorithms, not EAGLE.
        path = Path(config.model.draft_model_config or "")
        algorithm = None
        if path.is_file():
            from aspec.config.draft_algorithm import DRAFT_ALGORITHMS

            raw = json.loads(path.read_text())
            architectures = raw.get("architectures") or []
            algorithm = DRAFT_ALGORITHMS.get(architectures[0]) if architectures else None
        if algorithm not in {"eagle3", "eagle4", "mtp"}:
            field = "dspark_num_anchors" if algorithm in (None, "dspark") else "dflash_num_anchors"
            if config.training[field] % draft_execution.context_parallel_size:
                raise ValueError(f"training.{field} must be divisible by training.draft_context_parallel_size")
    if not config.inference.target_prefill_memory_optimized:
        return

    if config.inference.inference_engine_type != "sgl":
        raise NotImplementedError(
            "target_prefill_memory_optimized currently supports SGLang only"
        )
    if config.training.train_with_decode:
        raise NotImplementedError(
            "target_prefill_memory_optimized requires prefill-only Target execution"
        )
    if config.inference.inference_num_gpus_per_engine != 1:
        raise NotImplementedError(
            "target_prefill_memory_optimized currently requires one GPU per Target engine"
        )
    if config.inference.sglang.tp_size != 1 or config.inference.sglang.pp_size != 1:
        raise NotImplementedError(
            "target_prefill_memory_optimized currently requires SGLang TP1/PP1"
        )
    if config.inference.sglang.enable_multimodal:
        raise NotImplementedError(
            "target_prefill_memory_optimized does not yet support multimodal requests"
        )
    if config.training.attention_backend == "usp":
        raise NotImplementedError(
            "target_prefill_memory_optimized is not yet compatible with USP feature layout"
        )

    mooncake = config.mooncake
    if OmegaConf.select(mooncake, "data_plane", default="store") != "transfer_engine":
        raise ValueError(
            "target_prefill_memory_optimized requires mooncake.data_plane=transfer_engine"
        )
    protocol = OmegaConf.select(mooncake, "protocol", default="tcp")
    require_nvlink = bool(OmegaConf.select(mooncake, "require_nvlink", default=False))
    node_local_features = (
        str(OmegaConf.select(mooncake, "feature_affinity", default="any")).lower()
        == "node_local_required"
    )
    if config.cluster.mode == "static_multi_node":
        valid_multi_node = (
            protocol in {"tcp", "rdma"} and not require_nvlink
        ) or (
            node_local_features and protocol == "nvlink_intra" and require_nvlink
        )
        if not valid_multi_node:
            raise ValueError(
                "multi-node target_prefill_memory_optimized requires strict "
                "cross-node TCP/RDMA or node-local-required NVLink transport"
            )
    elif protocol != "nvlink_intra" or not require_nvlink:
        raise ValueError(
            "single-node target_prefill_memory_optimized requires strict "
            "mooncake.protocol=nvlink_intra with mooncake.require_nvlink=true"
        )
    if not OmegaConf.select(mooncake, "enable_gpu_direct", default=False):
        raise ValueError(
            "target_prefill_memory_optimized requires mooncake.enable_gpu_direct=true"
        )
    if OmegaConf.select(mooncake, "allow_host_fallback", default=True):
        raise ValueError(
            "target_prefill_memory_optimized requires mooncake.allow_host_fallback=false"
        )

    cp_size = int(config.training.draft_context_parallel_size)
    slots_per_sample = source_context_parallel_size if is_direct_cp_backend(config.training.attention_backend) else 1
    common_slots = int(OmegaConf.select(mooncake, "gpu_ring_slots", default=2))
    source_slots = OmegaConf.select(mooncake, "source_gpu_ring_slots", default=None)
    source_slots = common_slots if source_slots is None else int(source_slots)
    minimum_source_slots = config.inference.inference_batch_size * slots_per_sample
    if source_slots < minimum_source_slots:
        raise ValueError(
            "target_prefill_memory_optimized source ring cannot hold one Target batch: "
            f"source_gpu_ring_slots={source_slots}, require at least "
            f"{minimum_source_slots}"
        )

    world_size = (
        config.training.training_num_nodes
        * config.training.training_num_gpus_per_node
    )
    logical_dp_size = world_size // cp_size if is_direct_cp_backend(
        config.training.attention_backend
    ) else world_size
    dispatch_batch_size = config.training.micro_batch_size * logical_dp_size
    inference_gpus = config.inference.inference_num_gpus
    if not isinstance(inference_gpus, int) or inference_gpus < 1:
        raise ValueError(
            "target_prefill_memory_optimized requires a positive inference_num_gpus"
        )
    if node_local_features:
        engine_count = config.inference.inference_num_gpus_per_node
        dispatch_batch_size = (
            config.training.micro_batch_size
            * config.training.training_num_gpus_per_node
            // cp_size
        )
        capacity_scope = "per node"
    else:
        engine_count = inference_gpus
        capacity_scope = "across the run"
    resident_sample_capacity = direct_output_sample_capacity(
        engine_count=engine_count,
        source_slots=source_slots,
        slots_per_sample=slots_per_sample,
    )
    if resident_sample_capacity < dispatch_batch_size:
        raise ValueError(
            "target_prefill_memory_optimized source rings cannot retain one complete "
            f"Draft dispatch {capacity_scope}: capacity={resident_sample_capacity} "
            f"samples, dispatch_batch_size={dispatch_batch_size}, engines={engine_count}, "
            f"source_gpu_ring_slots={source_slots}, slots_per_sample={slots_per_sample}"
        )


def _save_config_snapshot(config: DictConfig) -> None:
    """Atomically save the resolved config for a real run."""
    output_dir = OmegaConf.select(config, "output_dir", default=None)
    if not output_dir:
        raise ValueError("output_dir is required for a real ASpec run")
    dest = Path(output_dir) / "config.yaml"
    temporary = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        save_config(config, str(temporary))
        os.replace(temporary, dest)
        logger.info(f"Saved resolved config to {dest}")
    finally:
        temporary.unlink(missing_ok=True)


def load_config(
    config_path: Optional[str] = None,
    cli_args: Optional[list] = None,
    base_config: Optional[DictConfig] = None,
    save_snapshot: bool = False,
    source_context_parallel_size: int | None = None,
) -> DictConfig:
    schema = OmegaConf.structured(Config)

    configs_to_merge = [schema]

    if base_config is not None:
        configs_to_merge.append(base_config)

    if config_path is not None:
        file_config = OmegaConf.load(config_path)
        _resolve_relative_paths(
            file_config,
            os.path.dirname(os.path.abspath(config_path)),
            skip_keys=frozenset(_ALWAYS_LOCAL_PATH_KEYS),
        )
        configs_to_merge.append(file_config)

    if cli_args:
        cli_config = OmegaConf.from_dotlist(cli_args)
        configs_to_merge.append(cli_config)

    config = OmegaConf.merge(*configs_to_merge)
    _resolve_relative_paths(config, os.getcwd())

    _validate_current_scope(config)
    _validate_cluster_config(config)
    _validate_vllm_config(config)
    _validate_offline_config(config)
    _validate_vocab_mapping_config(config)
    _validate_training_batch_config(config)
    _validate_training_numeric_config(config)
    _validate_inference_batch_config(config)
    _validate_execution_capabilities(config, source_context_parallel_size=source_context_parallel_size)

    if save_snapshot:
        _save_config_snapshot(config)

    return config


# Sub-sections whose fields receive a name prefix when flattened.
_PREFIXED_SECTIONS = {
    "cluster": "cluster_",
    "decode": "decode_",
    "mooncake": "mooncake_",
    "offline": "offline_",
    "sglang": "sglang_",
    "vllm": "vllm_",
    "trtllm": "trtllm_",
    "tokenspeed": "tokenspeed_",
}


def config_to_flat_args(config: DictConfig) -> argparse.Namespace:
    flat: dict[str, Any] = {}

    def _add(key: str, val: Any, origin: str) -> None:
        if key in flat:
            raise ValueError(f"Duplicate config key '{key}' (from '{origin}')")
        # Ray actors must receive canonical values, not lazy OmegaConf nodes.
        # In particular, a ListConfig can retain ${oc.env:...} references that
        # would otherwise be resolved in the actor's environment rather than
        # once in the submitting driver.
        if isinstance(val, (DictConfig, ListConfig)):
            flat[key] = OmegaConf.to_container(val, resolve=True)
        else:
            flat[key] = copy.deepcopy(val)

    for section_name, section in config.items():
        if not isinstance(section, DictConfig):
            _add(section_name, section, section_name)
            continue

        prefix = _PREFIXED_SECTIONS.get(section_name, "")
        for key, val in section.items():
            # Nested sub-config (e.g. inference.sglang) — flatten with its
            # own prefix so consumers keep seeing ``sglang_tp_size`` etc.
            if isinstance(val, DictConfig) and key in _PREFIXED_SECTIONS:
                sub_prefix = _PREFIXED_SECTIONS[key]
                for sub_key, sub_val in val.items():
                    _add(
                        f"{sub_prefix}{sub_key}",
                        sub_val,
                        f"{section_name}.{key}.{sub_key}",
                    )
            else:
                _add(f"{prefix}{key}", val, f"{section_name}.{key}")

    # --- Computed / alias fields ---
    for key in ("wsd_decay_ratio", "wsd_decay_style"):
        flat[key] = flat[key] if flat.get(key) is not None else flat[f"lr_{key}"]
    flat["world_size"] = flat["training_num_nodes"] * flat["training_num_gpus_per_node"]
    flat["rank"] = 0
    if flat.get("inference_engine_type") == "offline":
        # Replay records are already tokenized and carry their loss masks.
        flat["defer_tokenization"] = False
    # Deferred tokenization always needs the training-time matcher, since only
    # the engine knows the final token IDs. Renderers need it too: a renderer
    # emits its mask before the engine expands media placeholders, so any
    # multimodal sample's mask has to be recomputed after expansion.
    flat["dynamic_loss_mask"] = (
        flat.get("inference_engine_type") != "offline"
        and (flat["defer_tokenization"] or bool(flat.get("renderer")))
        and not flat["train_with_decode"]
    )
    flat["use_wandb"] = flat.get("use_wandb", False) or flat.get("report_to") == "wandb"
    flat["use_tensorboard"] = (
        flat.get("use_tensorboard", False) or flat.get("report_to") == "tensorboard"
    )
    flat["checkpoint_dir"] = (
        str(Path(flat["output_dir"]) / "checkpoints") if flat.get("output_dir") else None
    )
    if flat.get("continual_training") and not flat.get("load_path"):
        logger.warning("continual_training=True but no training.load_path was provided")
    if flat.get("continue_source_cursor"):
        if not flat.get("continual_training"):
            raise ValueError(
                "training.continue_source_cursor requires continual_training=true"
            )
        if not flat.get("load_path"):
            raise ValueError(
                "training.continue_source_cursor requires training.load_path"
            )

    if (
        "last_hidden_states_prenorm" not in flat or flat["last_hidden_states_prenorm"] is None
    ) and flat.get("inference_engine_type") != "offline":
        flat["last_hidden_states_prenorm"] = flat.get("inference_engine_type") == "vllm"

    return argparse.Namespace(**flat)


def save_config(config: DictConfig, path: str) -> None:
    resolved = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    OmegaConf.save(resolved, path)


def print_config(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config, resolve=True))

def validate_local_data_paths(config: DictConfig) -> None:
    """Check explicit local inputs at the CLI boundary without fetching Hub data."""
    from aspec.data.utils import is_local_data_path

    if config.inference.inference_engine_type == "offline":
        return
    for key in _DATA_PATH_KEYS:
        value = OmegaConf.select(config, key)
        if value and is_local_data_path(str(value)) and not Path(value).exists():
            raise FileNotFoundError(f"{key}: local dataset path not found: {value}")
