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

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from aspec.config.train_config import Config, load_config
from aspec.planning import PlanCompileError, compile_execution_plan
from aspec.topology import current_capability_bundle

_REPO_ROOT = Path(__file__).parents[2]
_REFERENCE_CONFIG = _REPO_ROOT / "configs" / "examples" / "sglang_qwen3_8b_dspark.yaml"
_DRAFT_CONFIG = _REPO_ROOT / "aspec" / "config" / "dspark_draft_config.json"


def _config(**dot_overrides):
    overrides = [f"{key}={value}" for key, value in dot_overrides.items()]
    return load_config(config_path=str(_REFERENCE_CONFIG), cli_args=overrides)


def _local_qwen_target(tmp_path, *, max_position_embeddings=262144):
    target = tmp_path / "Qwen3.5-test"
    target.mkdir()
    (target / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "max_position_embeddings": max_position_embeddings,
                },
            }
        ),
        encoding="utf-8",
    )
    return target


@pytest.mark.parametrize("precision", ["bf16", "fp8"])
def test_dspark_head_precision_reaches_trainer_and_resume(precision):
    from aspec.training.checkpoint import (
        _logical_training_topology,
        _validate_exact_resume_topology,
    )

    plan = compile_execution_plan(_config(**{"training.dspark_lm_head_precision": precision}))
    args = plan.to_runtime_args()
    assert args.dspark_lm_head_precision == plan.workload.objective["lm_head_precision"] == precision
    topology = _logical_training_topology(args, args.world_size)
    metadata = {"world_size": args.world_size, "training_topology": topology}
    assert topology["training_objective"]["lm_head_precision"] == precision
    _validate_exact_resume_topology(args, metadata)
    args.dspark_lm_head_precision = "bf16" if precision == "fp8" else "fp8"
    with pytest.raises(RuntimeError, match="dspark_lm_head_precision"):
        _validate_exact_resume_topology(args, metadata)
    del topology["training_objective"]["lm_head_precision"]
    args.dspark_lm_head_precision = "bf16"
    _validate_exact_resume_topology(args, metadata)
    args.dspark_lm_head_precision = "fp8"
    with pytest.raises(RuntimeError, match="dspark_lm_head_precision"):
        _validate_exact_resume_topology(args, metadata)


@pytest.mark.parametrize("options", [
    {"training.dspark_lm_head_precision": "int8"},
    {"training.dspark_lm_head_precision": "fp8", "training.draft_context_parallel_size": 2},
    {"training.dspark_lm_head_precision": "fp8", "training.compile_model": True},
    {"training.dspark_lm_head_precision": "fp8", "model.draft_model_config": str(_DRAFT_CONFIG.with_name("dflash_draft_config.json"))},
])
def test_invalid_dspark_head_precision_is_rejected(options):
    with pytest.raises(PlanCompileError, match="LM head|lm_head_precision"):
        compile_execution_plan(_config(**options))


def test_reference_config_compiles_deterministically():
    config = _config()

    first = compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG))
    second = compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG))

    assert first.to_json() == second.to_json()
    assert first.plan_id == second.plan_id


def test_reference_plan_separates_contract_policy_and_capability_states():
    config = _config(**{"training.num_train_steps": 1})
    plan = compile_execution_plan(
        config,
        config_path=str(_REFERENCE_CONFIG),
        cli_args=["training.num_train_steps=1"],
    ).to_dict()

    assert plan["schema_version"] == "aspec.execution-plan/v0alpha10"
    assert plan["source"]["format"] == "torchspec_yaml"
    assert plan["source"]["conversion"] == "deterministic"
    assert len(plan["source"]["resolved_config_sha256"]) == 64
    assert plan["source"]["cli_override_keys"] == ["training.num_train_steps"]
    assert plan["workload"]["models"]["target"]["family"] == "qwen3"
    assert plan["workload"]["models"]["draft"]["family"] == "dspark"
    assert plan["workload"]["features"]["aux_hidden_state_layers"] == [1, 9, 17, 25, 33]
    assert plan["workload"]["features"]["tensors"]["hidden_states"]["shape"] == [
        "sequence",
        20480,
    ]
    assert plan["workload"]["features"]["tensors"]["last_hidden_states"] == {
        "semantic": "target_final_hidden_state",
        "dtype": "bfloat16",
        "shape": ["sequence", 4096],
        "layout": "sequence_hidden",
        "value_stage": "post_norm",
    }
    assert len(plan["workload"]["features"]["contract_id"]) == 71
    assert plan["workload"]["duration"]["num_train_steps"] == 1
    assert plan["workload"]["duration"]["source_work"] == {
        "requested": None,
        "scheduled": None,
        "planned_dropped": 0,
        "alignment": None,
        "tail_policy": None,
    }
    assert plan["policy"]["pipeline"]["global_batch_size"] == 8
    assert plan["policy"]["capacity"].keys() == {
        "device_memory",
        "backend_tokens",
        "transfer_storage",
        "host_orchestrator",
    }
    assert plan["policy"]["measurement"]["performance"] == {
        "required": False,
        "warmup_steps": 0,
        "drain_steps": 0,
        "metric": "committed_work_per_second",
        "window": "continuous_optimizer_steps",
    }

    decisions = {decision["name"]: decision for decision in plan["capabilities"]}
    assert decisions["target_feature_engine"] == {
        "name": "target_feature_engine",
        "requirement": "required",
        "requested": "sgl",
        "resolved": "sglang",
        "effective": None,
        "effective_state": "pending_runtime_evidence",
    }
    assert decisions["work_scheduler"]["resolved"] == "stable_work_identity_with_step_commit"
    assert decisions["data_lifecycle"]["resolved"] == "produce_dispatch_release_or_discard"
    assert decisions["capacity_control"]["requested"] == 64
    assert decisions["capacity_control"]["resolved"] == "ready_pool_watermark_with_bounded_inflight"
    attention = decisions["draft_attention_execution"]
    assert attention["requirement"] == "required"
    assert attention["requested"]["attention_backend"] == "cudnn_flex_attention"
    assert attention["resolved"]["attention_backend"] == "cudnn_flex_attention"
    assert attention["effective"] is None
    assert attention["effective_state"] == "pending_runtime_evidence"
    capacity = plan["policy"]["capacity"]["host_orchestrator"]
    assert capacity["sample_pool_semantics"] == "ready_watermark"
    assert capacity["inflight_sample_limit"] == 32
    assert plan["fallbacks"] == []
    assert {item["capability"] for item in plan["omissions"]} == {
        "decode_training",
        "static_multi_node_execution",
        "node_local_feature_affinity",
        "gpu_direct_transport",
        "target_prefill_output",
        "producer_consumer_pipeline",
        "long_context_execution",
        "mixed_length_packing",
        "performance_measurement",
        "target_logits",
        "training_numerics",
    }


def test_static_multi_node_plan_records_explicit_role_topology(tmp_path):
    target = _local_qwen_target(tmp_path)
    draft_config = tmp_path / "dspark_draft_config.json"
    draft_config.write_bytes(_DRAFT_CONFIG.read_bytes())
    train_data = tmp_path / "train.jsonl"
    train_data.write_text("{}\n", encoding="utf-8")
    output_dir = tmp_path / "output"
    config = _config(
        **{
            "model.target_model_path": target,
            "model.draft_model_config": draft_config,
            "dataset.train_data_path": train_data,
            "output_dir": output_dir,
        }
    )
    config.cluster.mode = "static_multi_node"
    config.cluster.shared_root = str(tmp_path)
    config.training.placement_strategy = "custom"
    config.training.training_node_ips = ["10.0.0.1"]
    config.inference.inference_node_ips = ["10.0.0.2"]
    config.training.streaming_dispatch_with_training = True
    config.mooncake.data_plane = "transfer_engine"
    config.mooncake.protocol = "rdma"
    config.mooncake.device_name = "mlx5_bond_0"
    config.mooncake.rdma_gid_index = 3
    config.mooncake.enable_gpu_direct = True
    config.mooncake.allow_host_fallback = False
    config.mooncake.require_nvlink = False

    compiled = compile_execution_plan(config)
    plan = compiled.to_dict()

    assert plan["policy"]["resources"]["cluster"] == {
        "mode": "static_multi_node",
        "membership": "static_explicit",
        "nodes": [
            {
                "address": "10.0.0.1",
                "roles": ["training"],
                "training_gpus": 4,
                "inference_gpus": 0,
            },
            {
                "address": "10.0.0.2",
                "roles": ["inference"],
                "training_gpus": 0,
                "inference_gpus": 4,
            },
        ],
        "shared_root": str(tmp_path),
        "storage_mode": "shared",
        "homogeneous_gpus_required": True,
        "failure_policy": "fail_stop_attempt_checkpoint_restart",
    }
    decisions = {item["name"]: item for item in plan["capabilities"]}
    assert decisions["static_multi_node_execution"]["requirement"] == "required"
    assert decisions["static_multi_node_execution"]["effective_state"] == (
        "pending_runtime_evidence"
    )
    assert plan["policy"]["resources"]["inference"]["target_group_locality"] == (
        "node_local"
    )
    assert plan["policy"]["capacity"]["transfer_storage"]["rdma_port"] == 1
    assert compiled.to_runtime_args().mooncake_rdma_port == 1
    assert plan["policy"]["capacity"]["transfer_storage"]["rdma_gid_index"] == 3
    assert plan["policy"]["capacity"]["transfer_storage"][
        "resolved_rdma_routes"
    ] == {
        "10.0.0.1": {
            "device_name": "mlx5_bond_0",
            "port": 1,
            "gid_index": 3,
        },
        "10.0.0.2": {
            "device_name": "mlx5_bond_0",
            "port": 1,
            "gid_index": 3,
        },
    }


def test_static_multi_node_plan_accepts_generation_checked_tcp(tmp_path):
    target = _local_qwen_target(tmp_path)
    draft_config = tmp_path / "dspark_draft_config.json"
    draft_config.write_bytes(_DRAFT_CONFIG.read_bytes())
    train_data = tmp_path / "train.jsonl"
    train_data.write_text("{}\n", encoding="utf-8")
    config = _config(
        **{
            "model.target_model_path": target,
            "model.draft_model_config": draft_config,
            "dataset.train_data_path": train_data,
            "output_dir": tmp_path / "output",
        }
    )
    config.cluster.mode = "static_multi_node"
    config.cluster.shared_root = str(tmp_path)
    config.training.placement_strategy = "custom"
    config.training.training_node_ips = ["10.0.0.1"]
    config.inference.inference_node_ips = ["10.0.0.2"]
    config.training.streaming_dispatch_with_training = True
    config.mooncake.data_plane = "transfer_engine"
    config.mooncake.protocol = "tcp"
    config.mooncake.device_name = ""
    config.mooncake.enable_gpu_direct = True
    config.mooncake.allow_host_fallback = False
    config.mooncake.require_nvlink = False

    plan = compile_execution_plan(config).to_dict()

    decisions = {item["name"]: item for item in plan["capabilities"]}
    assert decisions["static_multi_node_execution"]["resolved"][
        "transport_stage"
    ] == "generation_checked_tcp"
    assert decisions["gpu_direct_transport"]["effective_state"] == "omitted"
    transfer = plan["policy"]["capacity"]["transfer_storage"]
    assert transfer["protocol"] == "tcp"
    assert transfer["transfer_timeout_seconds"] == 300
    assert transfer["tcp_progress_timeout_seconds"] == 300


def test_static_multi_node_plan_records_node_local_nvlink_affinity(tmp_path):
    target = _local_qwen_target(tmp_path)
    draft_config = tmp_path / "dspark_draft_config.json"
    draft_config.write_bytes(_DRAFT_CONFIG.read_bytes())
    train_data = tmp_path / "train.jsonl"
    train_data.write_text("{}\n", encoding="utf-8")
    config = _config(
        **{
            "model.target_model_path": target,
            "model.draft_model_config": draft_config,
            "dataset.train_data_path": train_data,
            "output_dir": tmp_path / "output",
        }
    )
    config.cluster.mode = "static_multi_node"
    config.cluster.shared_root = str(tmp_path)
    config.training.placement_strategy = "custom"
    config.training.training_num_nodes = 2
    config.training.training_num_gpus_per_node = 4
    config.training.training_node_ips = ["10.0.0.1", "10.0.0.2"]
    config.inference.inference_num_gpus = 8
    config.inference.inference_num_gpus_per_node = 4
    config.inference.inference_node_ips = ["10.0.0.1", "10.0.0.2"]
    config.training.streaming_dispatch_with_training = True
    config.mooncake.data_plane = "transfer_engine"
    config.mooncake.protocol = "nvlink_intra"
    config.mooncake.feature_affinity = "node_local_required"
    config.mooncake.enable_gpu_direct = True
    config.mooncake.allow_host_fallback = False
    config.mooncake.require_nvlink = True

    plan = compile_execution_plan(config).to_dict()
    decisions = {item["name"]: item for item in plan["capabilities"]}

    assert decisions["node_local_feature_affinity"]["requirement"] == "required"
    assert decisions["node_local_feature_affinity"]["resolved"] == {
        "producer_consumer_locality": "same_physical_node",
        "remote_feature_fallback": False,
        "gradient_collective_scope": "global_draft_data_parallel",
    }
    assert decisions["static_multi_node_execution"]["resolved"][
        "transport_stage"
    ] == "node_local_nvlink_registered_cuda_ring"
    assert plan["policy"]["pipeline"]["feature_affinity"] == (
        "node_local_required"
    )
    assert plan["policy"]["resources"]["cluster"]["nodes"] == [
        {
            "address": "10.0.0.1",
            "roles": ["inference", "training"],
            "training_gpus": 4,
            "inference_gpus": 4,
        },
        {
            "address": "10.0.0.2",
            "roles": ["inference", "training"],
            "training_gpus": 4,
            "inference_gpus": 4,
        },
    ]
    assert plan["policy"]["resources"]["locality_domains"] == [
        {
            "domain_id": "physical-node-0",
            "kind": "physical_node",
            "node_address": "10.0.0.1",
            "training_world_ranks": [0, 1, 2, 3],
            "draft_dp_ranks": [0, 1, 2, 3],
            "draft_parallel_group_size": 1,
            "target_engine_ranks": [0, 1, 2, 3],
            "target_tensor_parallel_size": 1,
            "dispatch_samples_per_microstep": 4,
            "ready_watermark": 32,
        },
        {
            "domain_id": "physical-node-1",
            "kind": "physical_node",
            "node_address": "10.0.0.2",
            "training_world_ranks": [4, 5, 6, 7],
            "draft_dp_ranks": [4, 5, 6, 7],
            "draft_parallel_group_size": 1,
            "target_engine_ranks": [4, 5, 6, 7],
            "target_tensor_parallel_size": 1,
            "dispatch_samples_per_microstep": 4,
            "ready_watermark": 32,
        },
    ]
    capacity = plan["policy"]["capacity"]["host_orchestrator"]
    assert capacity["ready_watermark_scope"] == "per_locality_domain"
    assert capacity["ready_watermark_by_domain"] == {
        "physical-node-0": 32,
        "physical-node-1": 32,
    }

    configured_watermark = config.inference.max_sample_pool_size
    config.inference.max_sample_pool_size = 1
    with pytest.raises(
        PlanCompileError,
        match="at least one credit per locality domain",
    ):
        compile_execution_plan(config)
    config.inference.max_sample_pool_size = configured_watermark

    config.mooncake.protocol = "tcp"
    config.mooncake.require_nvlink = False
    with pytest.raises(
        PlanCompileError,
        match="requires strict transfer_engine/nvlink_intra",
    ):
        compile_execution_plan(config)


def test_static_multi_node_plan_records_explicit_host_staged_tcp(tmp_path):
    target = _local_qwen_target(tmp_path)
    draft_config = tmp_path / "dspark_draft_config.json"
    draft_config.write_bytes(_DRAFT_CONFIG.read_bytes())
    train_data = tmp_path / "train.jsonl"
    train_data.write_text("{}\n", encoding="utf-8")
    config = _config(
        **{
            "model.target_model_path": target,
            "model.draft_model_config": draft_config,
            "dataset.train_data_path": train_data,
            "output_dir": tmp_path / "output",
        }
    )
    config.cluster.mode = "static_multi_node"
    config.cluster.shared_root = str(tmp_path)
    config.training.placement_strategy = "custom"
    config.training.training_node_ips = ["10.0.0.1"]
    config.inference.inference_node_ips = ["10.0.0.2"]
    config.training.streaming_dispatch_with_training = True
    config.mooncake.data_plane = "store"
    config.mooncake.protocol = "tcp"
    config.mooncake.device_name = ""
    config.mooncake.enable_gpu_direct = False
    config.mooncake.allow_host_fallback = False
    config.mooncake.require_nvlink = False

    plan = compile_execution_plan(config).to_dict()

    decisions = {item["name"]: item for item in plan["capabilities"]}
    assert decisions["static_multi_node_execution"]["resolved"][
        "transport_stage"
    ] == "host_staged_tcp"
    assert decisions["tensor_transport"]["resolved"] == "mooncake:store:tcp"
    assert decisions["gpu_direct_transport"]["effective_state"] == "omitted"
    assert plan["policy"]["capacity"]["transfer_storage"]["staging_mode"] == (
        "pinned_host_buffers"
    )
    assert plan["policy"]["capacity"]["transfer_storage"][
        "host_fallback_allowed"
    ] is False
    transfer = plan["policy"]["capacity"]["transfer_storage"]
    assert transfer["producer_pinned_host_bytes_per_endpoint"] == (
        transfer["host_buffer_size"] * transfer["async_put_pool_size"]
    )
    assert transfer["producer_pinned_host_bytes_per_endpoint"] > 0

    bundle = current_capability_bundle(compile_execution_plan(config))
    assert bundle.data_plane == "store"
    assert bundle.gpu_direct is False
    assert bundle.host_fallback_allowed is False


@pytest.mark.parametrize("protocol", ["tcp", "rdma"])
def test_static_multi_node_plan_accepts_direct_target_output(tmp_path, protocol):
    target = _local_qwen_target(tmp_path)
    draft_config = tmp_path / "dspark_draft_config.json"
    draft_config.write_bytes(_DRAFT_CONFIG.read_bytes())
    train_data = tmp_path / "train.jsonl"
    train_data.write_text("{}\n", encoding="utf-8")
    config = _config(
        **{
            "model.target_model_path": target,
            "model.draft_model_config": draft_config,
            "dataset.train_data_path": train_data,
            "output_dir": tmp_path / "output",
        }
    )
    config.cluster.mode = "static_multi_node"
    config.cluster.shared_root = str(tmp_path)
    config.training.placement_strategy = "custom"
    config.training.training_node_ips = ["10.0.0.1"]
    config.inference.inference_node_ips = ["10.0.0.2"]
    config.training.streaming_dispatch_with_training = True
    config.inference.target_prefill_memory_optimized = True
    config.inference.inference_batch_size = 1
    config.mooncake.data_plane = "transfer_engine"
    config.mooncake.protocol = protocol
    config.mooncake.device_name = "mlx5_bond_0" if protocol == "rdma" else ""
    config.mooncake.rdma_gid_index = 3 if protocol == "rdma" else None
    config.mooncake.enable_gpu_direct = True
    config.mooncake.allow_host_fallback = False
    config.mooncake.require_nvlink = False

    plan = compile_execution_plan(config).to_dict()
    decisions = {item["name"]: item for item in plan["capabilities"]}

    assert decisions["target_prefill_output"]["requirement"] == "required"
    assert decisions["target_prefill_output"]["resolved"] == (
        "prefill_direct_to_generation_checked_transport"
    )
    assert plan["policy"]["pipeline"]["target_feature_output"] == (
        "direct_to_transport"
    )
    assert plan["policy"]["capacity"]["transfer_storage"]["protocol"] == protocol
    assert decisions["gpu_direct_transport"]["requirement"] == (
        "optional" if protocol == "tcp" else "required"
    )


def test_required_performance_window_is_explicit_in_plan():
    overrides = {
        "training.num_train_steps": 6,
        "debug.performance_measurement_required": "true",
        "debug.performance_warmup_steps": 2,
        "debug.performance_drain_steps": 1,
    }
    plan = compile_execution_plan(
        _config(**overrides),
        config_path=str(_REFERENCE_CONFIG),
        cli_args=[f"{key}={value}" for key, value in overrides.items()],
    ).to_dict()

    decisions = {item["name"]: item for item in plan["capabilities"]}
    assert decisions["performance_measurement"]["requirement"] == "required"
    assert decisions["training_numerics"]["requirement"] == "required"
    assert plan["policy"]["measurement"]["performance"]["warmup_steps"] == 2
    assert plan["policy"]["measurement"]["performance"]["drain_steps"] == 1


def test_context_contract_fails_closed_without_declared_override(tmp_path):
    target = _local_qwen_target(tmp_path)
    config = _config(
        **{
            "model.target_model_path": target,
            "model.native_context_length": 262144,
            "training.max_seq_length": 262144,
        }
    )

    with pytest.raises(PlanCompileError, match="exceeds the backend-declared limit"):
        compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG))


def test_context_contract_makes_override_and_cp_shards_explicit(tmp_path):
    target = _local_qwen_target(tmp_path)
    overrides = {
        "model.target_model_path": target,
        "model.native_context_length": 262144,
        "training.max_seq_length": 262144,
        "training.attention_backend": "fa4",
        "training.draft_context_parallel_size": 2,
        "inference.sglang.allow_context_length_override": "true",
    }
    plan = compile_execution_plan(
        _config(**overrides),
        config_path=str(_REFERENCE_CONFIG),
        cli_args=[f"{key}={value}" for key, value in overrides.items()],
    )

    capacity = plan.to_dict()["policy"]["capacity"]["backend_tokens"]
    assert capacity["requested_max_sequence_length"] == 262144
    assert capacity["effective_training_max_sequence_length"] == 262144
    assert capacity["target_input_reserve_tokens"] == 6
    assert capacity["backend_context_length"] == 262150
    assert capacity["model_native_context_limit"] == 262144
    assert capacity["model_native_context_limit_source"] == "model.native_context_length"
    assert capacity["model_config_context_limit"] == 262144
    assert capacity["model_context_extension"] is None
    assert capacity["admitted_training_context_limit"] == 262144
    assert capacity["backend_declared_context_limit"] == 262144
    assert capacity["context_override"] == {
        "allowed": True,
        "required": True,
        "effective": True,
        "mechanism": "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN",
    }
    assert capacity["training_sharding"] == {
        "context_parallel_size": 2,
        "context_shard_alignment": 1,
        "shard_lengths_at_max": [131072, 131072],
        "sequence_padding_tokens_at_max": 0,
    }
    assert plan.to_runtime_args().aspec_context_contract == {
        key: capacity[key]
        for key in (
            "requested_max_sequence_length",
            "effective_training_max_sequence_length",
            "target_input_reserve_tokens",
            "backend_context_length",
            "model_native_context_limit",
            "model_native_context_limit_source",
            "model_config_context_limit",
            "model_config_context_limit_source",
            "model_context_extension",
            "admitted_training_context_limit",
            "backend_declared_context_limit",
            "model_limit_state",
            "context_override",
            "training_sharding",
        )
    }


def test_backend_override_cannot_replace_model_context_extension(tmp_path):
    target = _local_qwen_target(tmp_path, max_position_embeddings=40960)
    config = _config(
        **{
            "model.target_model_path": target,
            "model.native_context_length": 32768,
            "training.max_seq_length": 131072,
            "inference.sglang.allow_context_length_override": "true",
        }
    )

    with pytest.raises(PlanCompileError, match="cannot replace a declared RoPE"):
        compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG))


def test_sglang_model_override_declares_yarn_context_extension(tmp_path):
    target = _local_qwen_target(tmp_path, max_position_embeddings=32768)
    config = _config(
        **{
            "model.target_model_path": target,
            "model.native_context_length": 32768,
            "training.max_seq_length": 131072,
            "inference.sglang.allow_context_length_override": "true",
            "inference.sglang.extra_args.json_model_override_args": json.dumps(
                {
                    "text_config": {
                        "max_position_embeddings": 131072,
                        "rope_scaling": {
                            "rope_type": "yarn",
                            "factor": 4.0,
                            "original_max_position_embeddings": 32768,
                        },
                    }
                }
            ),
        }
    )

    plan = compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG))

    capacity = plan.to_dict()["policy"]["capacity"]["backend_tokens"]
    assert capacity["model_config_context_limit"] == 131072
    assert capacity["model_context_extension"] == {
        "type": "yarn",
        "factor": 4.0,
        "original_context_length": 32768,
        "declared_context_limit": 131072,
        "source": "model rope_scaling",
    }
    assert capacity["admitted_training_context_limit"] == 131072
    assert capacity["context_override"]["required"] is True
    assert capacity["context_override"]["effective"] is True


@pytest.mark.parametrize("value", ["not-json", "[]"])
def test_sglang_model_override_must_be_a_json_object(tmp_path, value):
    target = _local_qwen_target(tmp_path)
    config = _config(
        **{
            "model.target_model_path": target,
            "inference.sglang.extra_args.json_model_override_args": value,
        }
    )

    with pytest.raises(
        PlanCompileError,
        match="json_model_override_args must (?:be valid JSON|contain a JSON object)",
    ):
        compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG))


def test_runtime_namespace_is_generated_from_plan():
    plan = compile_execution_plan(_config(), config_path=str(_REFERENCE_CONFIG))

    args = plan.to_runtime_args()

    assert args.aspec_plan_id == plan.plan_id
    assert args.dp_size == 4
    assert args.sp_size == 1
    assert args.global_batch_size == 8
    assert args.aux_hidden_states_layers == [1, 9, 17, 25, 33]
    assert args.draft_model_config == str(_DRAFT_CONFIG)
    assert args.aspec_feature_contract == plan.workload.features


@pytest.mark.parametrize(
    "dp_size,requested,effective",
    [(2, 5, 4), (3, 8, 6), (4, 11, 8)],
)
def test_packed_source_budget_drops_only_incomplete_dp_tail(
    dp_size, requested, effective
):
    inference_gpus = 8 - dp_size
    config = _config(
        **{
            "training.training_num_gpus_per_node": dp_size,
            "training.dflash_packing": "true",
            "training.source_work_budget": requested,
            "inference.inference_num_gpus": inference_gpus,
            "inference.inference_num_gpus_per_node": inference_gpus,
        }
    )

    plan = compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG))
    source_work = plan.workload.duration["source_work"]
    args = plan.to_runtime_args()

    assert source_work == {
        "requested": requested,
        "scheduled": effective,
        "planned_dropped": requested - effective,
        "alignment": dp_size,
        "tail_policy": "drop_last_complete_packed_optimizer_step",
    }
    assert args.requested_source_work_budget == requested
    assert args.source_work_budget == effective
    assert args.planned_dropped_source_work == requested - effective


def test_fixed_row_source_budget_drops_incomplete_optimizer_batch():
    config = _config(
        **{
            "training.training_num_gpus_per_node": 3,
            "training.source_work_budget": 10,
            "inference.inference_num_gpus": 5,
            "inference.inference_num_gpus_per_node": 5,
        }
    )

    plan = compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG))

    assert plan.workload.duration["source_work"] == {
        "requested": 10,
        "scheduled": 6,
        "planned_dropped": 4,
        "alignment": 6,
        "tail_policy": "drop_last_complete_optimizer_batch",
    }


def test_source_budget_rejects_less_than_one_executable_unit():
    config = _config(
        **{
            "training.training_num_gpus_per_node": 3,
            "training.dflash_packing": "true",
            "training.source_work_budget": 5,
            "inference.inference_num_gpus": 5,
            "inference.inference_num_gpus_per_node": 5,
        }
    )

    with pytest.raises(PlanCompileError, match="complete executable packed optimizer step"):
        compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG))


def test_runtime_namespace_eagerly_resolves_environment_lists(monkeypatch):
    monkeypatch.setenv("ASPEC_TEST_PROFILE_0", "train_overall")
    monkeypatch.setenv("ASPEC_TEST_PROFILE_1", "feature_transport")
    config = _config()
    config.debug.profile_target = [
        "${oc.env:ASPEC_TEST_PROFILE_0}",
        "${oc.env:ASPEC_TEST_PROFILE_1}",
    ]

    plan = compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG))
    monkeypatch.delenv("ASPEC_TEST_PROFILE_0")
    monkeypatch.delenv("ASPEC_TEST_PROFILE_1")

    assert plan.to_runtime_args().profile_target == [
        "train_overall",
        "feature_transport",
    ]


def test_plan_identity_covers_resolved_config_not_only_rendered_fields():
    first = compile_execution_plan(_config(), config_path=str(_REFERENCE_CONFIG))
    second = compile_execution_plan(
        _config(**{"debug.max_dump_steps": 17}), config_path=str(_REFERENCE_CONFIG)
    )

    assert first.plan_id != second.plan_id


def test_required_dspark_last_hidden_state_fails_closed():
    config = _config(**{"inference.store_last_hidden_states": "false"})

    with pytest.raises(PlanCompileError, match="DSpark requires"):
        compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG))


def test_ready_watermark_need_not_hold_an_entire_inflight_batch():
    config = _config(**{"inference.max_sample_pool_size": 1})
    plan = compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG))
    assert plan.policy.pipeline["request_batch_size"] == 8
    assert plan.policy.capacity["host_orchestrator"]["max_sample_pool_size"] == 1
    assert plan.policy.capacity["host_orchestrator"]["inflight_sample_limit"] == 32


def test_unbounded_legacy_output_pool_is_explicitly_omitted():
    config = _config(**{"inference.max_sample_pool_size": 0})

    plan = compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG)).to_dict()

    capacity = next(item for item in plan["capabilities"] if item["name"] == "capacity_control")
    assert capacity["requirement"] == "optional"
    assert capacity["effective_state"] == "omitted"
    assert {item["capability"] for item in plan["omissions"]} >= {"capacity_control"}


def test_nvlink_pipeline_and_draft_execution_are_explicit_capabilities():
    config = _config(
        **{
            "training.attention_backend": "fa4",
            "training.draft_context_parallel_size": 2,
            "training.streaming_dispatch_with_training": "true",
            "training.pipeline_dispatch_with_training": "true",
            "mooncake.data_plane": "transfer_engine",
            "mooncake.protocol": "nvlink_intra",
            "mooncake.enable_gpu_direct": "true",
            "mooncake.allow_host_fallback": "false",
            "mooncake.require_nvlink": "true",
            "mooncake.source_gpu_ring_slots": 16,
        }
    )

    plan = compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG)).to_dict()
    decisions = {item["name"]: item for item in plan["capabilities"]}

    assert decisions["tensor_transport"]["resolved"] == (
        "mooncake:transfer_engine:nvlink_intra"
    )
    assert decisions["gpu_direct_transport"]["requirement"] == "required"
    assert decisions["producer_consumer_pipeline"]["requested"] == {
        "streaming": True,
        "next_step_prefix": True,
    }
    assert decisions["draft_attention_execution"]["resolved"] == {
        "attention_backend": "fa4",
        "context_parallel_size": 2,
        "feature_layout": "context_sharded",
        "head_layout": {
            "type": "gqa",
            "query_heads": 32,
            "key_value_heads": 8,
            "query_heads_per_key_value_head": 4,
            "head_dim": 128,
            "head_dim_source": "hidden_size/num_attention_heads",
        },
    }
    assert plan["policy"]["capacity"]["transfer_storage"]["host_fallback_allowed"] is False
    assert plan["policy"]["capacity"]["transfer_storage"]["source_gpu_ring_slots"] == 16


def test_sdpa_at_cp1_omits_specialized_draft_attention_execution():
    config = _config(**{
        "training.attention_backend": "sdpa",
        "training.draft_context_parallel_size": 1,
    })

    plan = compile_execution_plan(config).to_dict()
    decision = next(
        item for item in plan["capabilities"]
        if item["name"] == "draft_attention_execution"
    )

    assert decision["requirement"] == "optional"
    assert decision["requested"]["attention_backend"] == "sdpa"
    assert decision["resolved"] is None
    assert decision["effective"] is None
    assert decision["effective_state"] == "omitted"
    assert {"capability": "draft_attention_execution", "reason": "not_requested"} in plan["omissions"]


def test_cudnn_flex_attention_is_a_required_draft_execution_capability():
    config = _config(
        **{
            "training.attention_backend": "cudnn_flex_attention",
            "training.draft_context_parallel_size": 1,
        }
    )

    plan = compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG)).to_dict()
    decision = next(
        item
        for item in plan["capabilities"]
        if item["name"] == "draft_attention_execution"
    )

    assert decision["requirement"] == "required"
    assert decision["effective_state"] == "pending_runtime_evidence"
    assert decision["resolved"] == {
        "attention_backend": "cudnn_flex_attention",
        "context_parallel_size": 1,
        "feature_layout": "full",
        "cp_strategy": "local",
        "head_layout": {
            "type": "gqa",
            "query_heads": 32,
            "key_value_heads": 8,
            "query_heads_per_key_value_head": 4,
            "head_dim": 128,
            "head_dim_source": "hidden_size/num_attention_heads",
        },
    }
    assert "draft_attention_execution" not in {
        item["capability"] for item in plan["omissions"]
    }


def test_cudnn_packing_plan_uses_native_thd_at_cp1():
    config = _config(**{
        "training.attention_backend": "cudnn_flex_attention",
        "training.dflash_packing": True,
        "training.draft_context_parallel_size": 1,
        "training.micro_batch_size": 1,
    })
    plan = compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG)).to_dict()
    decisions = {item["name"]: item for item in plan["capabilities"]}
    assert decisions["mixed_length_packing"]["resolved"] == "controller_owned_document_aware_thd_rows"
    assert decisions["draft_thd_execution"]["resolved"] == {
        "qkv_format": "thd",
        "kernel_backend": "cudnn_flex_attention",
        "kernel_mode": "varlen_interval",
        "backward_strategy": "cudnn_flex_varlen",
    }


@pytest.mark.parametrize("cp_size", [2, 4])
def test_cudnn_context_parallel_plan_uses_query_partials(cp_size):
    config = _config(**{
        "training.attention_backend": "cudnn_flex_attention",
        "training.draft_context_parallel_size": cp_size,
        "training.training_num_gpus_per_node": 4,
        "training.dspark_num_anchors": 512,
    })
    plan = compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG)).to_dict()
    decision = next(
        item for item in plan["capabilities"]
        if item["name"] == "draft_attention_execution"
    )
    assert decision["resolved"]["context_parallel_size"] == cp_size
    assert decision["resolved"]["cp_strategy"] == "query_partials"


def test_cudnn_context_parallel_fails_closed_for_non_dflash_family():
    config = _config(**{
        "model.draft_model_config": "aspec/config/eagle3_qwen3_8b_draft_config.json",
        "training.attention_backend": "cudnn_flex_attention",
        "training.dflash_packing": True,
    })

    with pytest.raises(PlanCompileError, match="implemented for the DSpark/DFlash family"):
        compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG))


def test_direct_target_output_is_a_separate_required_capability():
    config = _config(
        **{
            "inference.inference_batch_size": 1,
            "inference.target_prefill_memory_optimized": "true",
            "training.streaming_dispatch_with_training": "true",
            "mooncake.data_plane": "transfer_engine",
            "mooncake.protocol": "nvlink_intra",
            "mooncake.enable_gpu_direct": "true",
            "mooncake.allow_host_fallback": "false",
            "mooncake.require_nvlink": "true",
        }
    )

    plan = compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG)).to_dict()
    decision = next(
        item for item in plan["capabilities"]
        if item["name"] == "target_prefill_output"
    )

    assert decision["requirement"] == "required"
    assert decision["resolved"] == "prefill_direct_to_generation_checked_transport"
    assert plan["policy"]["pipeline"]["target_feature_output"] == "direct_to_transport"
    assert "target_prefill_output" not in {
        item["capability"] for item in plan["omissions"]
    }


def test_direct_target_output_uses_source_ring_capacity_not_distinct_engines():
    config = _config(
        **{
            "training.training_num_gpus_per_node": 2,
            "inference.inference_num_gpus": 1,
            "inference.inference_batch_size": 1,
            "inference.target_prefill_memory_optimized": "true",
            "training.streaming_dispatch_with_training": "true",
            "mooncake.data_plane": "transfer_engine",
            "mooncake.protocol": "nvlink_intra",
            "mooncake.enable_gpu_direct": "true",
            "mooncake.allow_host_fallback": "false",
            "mooncake.require_nvlink": "true",
            "mooncake.source_gpu_ring_slots": 2,
        }
    )

    plan = compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG))
    assert plan.policy.pipeline["target_feature_output"] == "direct_to_transport"

    config.mooncake.source_gpu_ring_slots = 1
    with pytest.raises(ValueError, match="cannot retain one complete Draft dispatch"):
        load_config(base_config=config)


def test_direct_target_output_rejects_packing_threshold_above_source_capacity():
    config = _config(
        **{
            "training.training_num_gpus_per_node": 4,
            "training.attention_backend": "fa4",
            "training.draft_context_parallel_size": 4,
            "training.dflash_packing": "true",
            "training.packing_min_samples_per_dispatch": 4,
            "inference.inference_num_gpus": 1,
            "inference.inference_batch_size": 1,
            "inference.target_prefill_memory_optimized": "true",
            "training.streaming_dispatch_with_training": "true",
            "mooncake.data_plane": "transfer_engine",
            "mooncake.protocol": "nvlink_intra",
            "mooncake.enable_gpu_direct": "true",
            "mooncake.allow_host_fallback": "false",
            "mooncake.require_nvlink": "true",
            # Direct CP consumes four descriptors per sample, so this can retain
            # two samples: enough for a DP1 dispatch, but not the packing gate.
            "mooncake.source_gpu_ring_slots": 8,
        }
    )

    with pytest.raises(PlanCompileError, match="packing supply threshold"):
        compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG))


def test_packing_threshold_uses_ready_watermark_force_progress_capacity():
    config = _config(
        **{
            "training.training_num_gpus_per_node": 4,
            "training.attention_backend": "fa4",
            "training.draft_context_parallel_size": 4,
            "training.dflash_packing": "true",
            "training.packing_min_samples_per_dispatch": 16,
            "inference.max_sample_pool_size": 2,
            "inference.inference_num_gpus": 1,
            "inference.inference_batch_size": 1,
            "inference.target_prefill_memory_optimized": "true",
            "training.streaming_dispatch_with_training": "true",
            "mooncake.data_plane": "transfer_engine",
            "mooncake.protocol": "nvlink_intra",
            "mooncake.enable_gpu_direct": "true",
            "mooncake.allow_host_fallback": "false",
            "mooncake.require_nvlink": "true",
            "mooncake.source_gpu_ring_slots": 8,
        }
    )

    plan = compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG))
    assert plan.policy.capacity["transfer_storage"]["source_gpu_ring_slots"] == 8


def test_topology_mismatch_fails_before_backend_start():
    config = _config(**{"inference.sglang.tp_size": 2})

    with pytest.raises(PlanCompileError, match="tp_size must equal"):
        compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG))


def test_feature_contract_mismatch_fails_before_backend_start():
    config = _config(**{"inference.aux_hidden_states_layers": "[1,9,17]"})

    with pytest.raises(PlanCompileError, match="must match draft target_layer_ids"):
        compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG))


def test_invalid_draft_dimensions_fail_before_backend_start(tmp_path):
    invalid = tmp_path / "draft.json"
    descriptor = json.loads(_DRAFT_CONFIG.read_text(encoding="utf-8"))
    descriptor["target_hidden_size"] = None
    invalid.write_text(json.dumps(descriptor), encoding="utf-8")
    config = _config(**{"model.draft_model_config": str(invalid)})

    with pytest.raises(PlanCompileError, match="target_hidden_size"):
        compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG))


def test_invalid_draft_head_layout_fails_before_backend_start(tmp_path):
    invalid = tmp_path / "draft.json"
    descriptor = json.loads(_DRAFT_CONFIG.read_text(encoding="utf-8"))
    descriptor["num_attention_heads"] = 30
    descriptor["num_key_value_heads"] = 8
    invalid.write_text(json.dumps(descriptor), encoding="utf-8")
    config = _config(**{"model.draft_model_config": str(invalid)})

    with pytest.raises(PlanCompileError, match="must be divisible"):
        compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG))


def test_transport_hidden_dim_mismatch_fails_before_backend_start():
    config = _config(**{"mooncake.hidden_dim": 3584})

    with pytest.raises(
        PlanCompileError,
        match="mooncake.hidden_dim must match draft target_hidden_size",
    ):
        compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG))


def test_sglang_prenorm_claim_fails_closed():
    config = _config(**{"inference.last_hidden_states_prenorm": "true"})

    with pytest.raises(PlanCompileError, match="post-norm"):
        compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG))


def test_unsupported_drafter_fails_closed(tmp_path):
    unsupported = tmp_path / "draft.json"
    unsupported.write_text(
        json.dumps(
            {
                "architectures": ["UnknownDraftModel"],
                "model_type": "dflash2",
                "target_layer_ids": [1, 9, 17, 25, 33],
            }
        )
    )
    config = _config(**{"model.draft_model_config": str(unsupported)})

    with pytest.raises(PlanCompileError, match="outside the current DSpark/DFlash/DFlash2/EAGLE3 scope"):
        compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG))


def test_protected_sglang_passthrough_fails_closed():
    config = _config(**{"inference.sglang.extra_args.tp_size": 4})

    with pytest.raises(PlanCompileError, match="ASpec-managed keys"):
        compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG))


def test_compile_requires_explicit_draft_contract():
    config = OmegaConf.structured(Config)
    config.model.target_model_path = "Qwen/Qwen3-8B"
    config.dataset.train_data_path = "dataset"
    config.inference.inference_num_gpus = 1
    config.inference.sglang.tp_size = 1

    with pytest.raises(PlanCompileError, match="requires model.draft_model_config"):
        compile_execution_plan(config)


def test_dry_run_prints_plan_without_writing_artifacts(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "aspec",
            "--config",
            str(_REFERENCE_CONFIG),
            "--dry-run",
            f"output_dir={output_dir}",
            "training.num_train_steps=1",
        ],
    )

    from aspec.train_entry import parse_config

    with pytest.raises(SystemExit) as exc_info:
        parse_config()

    assert exc_info.value.code == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["workload"]["duration"]["num_train_steps"] == 1
    assert not output_dir.exists()
