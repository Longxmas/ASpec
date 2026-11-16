# Copyright 2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import subprocess
from pathlib import Path

import pytest

from aspec import evidence
from aspec.config.train_config import load_config
from aspec.evidence import RunEvidenceRecorder
from aspec.planning import compile_execution_plan, save_execution_plan

_REPO_ROOT = Path(__file__).parents[1]
_REFERENCE_CONFIG = _REPO_ROOT / "configs" / "examples" / "sglang_qwen3_8b_dspark.yaml"
_DRAFT_CONFIG = _REPO_ROOT / "aspec" / "config" / "dspark_draft_config.json"


def _plan():
    config = load_config(config_path=str(_REFERENCE_CONFIG))
    return compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG))


def _recorder(tmp_path):
    plan = _plan()
    return plan, _recorder_for_plan(tmp_path, plan)


def _recorder_for_plan(tmp_path, plan):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "config.yaml").write_text("resolved: true\n", encoding="utf-8")
    save_execution_plan(plan, output_dir / "plan.json")
    return RunEvidenceRecorder(plan, output_dir)


def test_evidence_recorder_refuses_to_overwrite_existing_attempt(tmp_path):
    plan, recorder = _recorder(tmp_path)

    with pytest.raises(FileExistsError, match="existing ASpec evidence"):
        RunEvidenceRecorder(plan, recorder.path.parent)


def test_dataset_admission_is_persisted_before_actor_initialization(tmp_path):
    _plan_value, recorder = _recorder(tmp_path)
    admission = {
        "train": {
            "state": "admitted",
            "source_samples": 3,
            "processed_samples": 2,
            "admitted_samples": 2,
            "skipped_samples": 1,
        },
        "eval": {
            "state": "not_configured",
            "source_samples": 0,
            "processed_samples": 0,
            "admitted_samples": 0,
            "skipped_samples": 0,
            "truncated_samples": 0,
        },
    }

    recorder.record_dataset_admission(
        dataset_size=2,
        eval_dataset_size=0,
        dataset_admission=admission,
    )

    persisted = json.loads(recorder.path.read_text(encoding="utf-8"))
    assert persisted["observations"]["runtime"]["phase"] == "dataset_loaded"
    assert persisted["observations"]["runtime"]["dataset_admission"] == admission


def _static_multi_node_plan(tmp_path, *, data_plane="transfer_engine"):
    target = tmp_path / "Qwen3-test"
    target.mkdir()
    (target / "config.json").write_text(
        json.dumps(
            {"model_type": "qwen3", "max_position_embeddings": 32768}
        ),
        encoding="utf-8",
    )
    draft = tmp_path / "draft.json"
    draft.write_bytes(_DRAFT_CONFIG.read_bytes())
    train_data = tmp_path / "train.jsonl"
    train_data.write_text("{}\n", encoding="utf-8")
    overrides = [
        f"model.target_model_path={target}",
        f"model.draft_model_config={draft}",
        f"dataset.train_data_path={train_data}",
        f"output_dir={tmp_path / 'plan-output'}",
    ]
    config = load_config(config_path=str(_REFERENCE_CONFIG), cli_args=overrides)
    config.cluster.mode = "static_multi_node"
    config.cluster.shared_root = str(tmp_path)
    config.training.placement_strategy = "custom"
    config.training.training_node_ips = ["10.0.0.1"]
    config.inference.inference_node_ips = ["10.0.0.2"]
    config.mooncake.data_plane = data_plane
    config.mooncake.protocol = "rdma" if data_plane == "transfer_engine" else "tcp"
    config.mooncake.device_name = "mlx5_bond_0" if data_plane == "transfer_engine" else ""
    config.mooncake.rdma_port = 1
    config.mooncake.rdma_gid_index = 3 if data_plane == "transfer_engine" else None
    config.mooncake.enable_gpu_direct = data_plane == "transfer_engine"
    config.mooncake.allow_host_fallback = False
    config.mooncake.require_nvlink = False
    return compile_execution_plan(config)


def _node_local_multi_node_plan(tmp_path):
    target = tmp_path / "Qwen3-node-local-test"
    target.mkdir()
    (target / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "max_position_embeddings": 32768}),
        encoding="utf-8",
    )
    draft = tmp_path / "node-local-draft.json"
    draft.write_bytes(_DRAFT_CONFIG.read_bytes())
    train_data = tmp_path / "node-local-train.jsonl"
    train_data.write_text("{}\n", encoding="utf-8")
    overrides = [
        f"model.target_model_path={target}",
        f"model.draft_model_config={draft}",
        f"dataset.train_data_path={train_data}",
        f"output_dir={tmp_path / 'node-local-plan-output'}",
    ]
    config = load_config(config_path=str(_REFERENCE_CONFIG), cli_args=overrides)
    config.cluster.mode = "static_multi_node"
    config.cluster.shared_root = str(tmp_path)
    config.training.placement_strategy = "custom"
    config.training.training_num_nodes = 2
    config.training.training_num_gpus_per_node = 4
    config.training.training_node_ips = ["10.0.0.1", "10.0.0.2"]
    config.training.fsdp_strategy = "REPLICATE"
    config.training.streaming_dispatch_with_training = True
    config.inference.inference_num_gpus = 8
    config.inference.inference_num_gpus_per_node = 4
    config.inference.inference_node_ips = ["10.0.0.1", "10.0.0.2"]
    config.mooncake.data_plane = "transfer_engine"
    config.mooncake.protocol = "nvlink_intra"
    config.mooncake.feature_affinity = "node_local_required"
    config.mooncake.enable_gpu_direct = True
    config.mooncake.allow_host_fallback = False
    config.mooncake.require_nvlink = True
    return compile_execution_plan(config)


def _observed_locality_domains(plan):
    return [
        {
            **copy.deepcopy(domain),
            "ray_node_id": f"ray-{domain['domain_id']}",
            "hostname": domain["domain_id"],
            "gpu_uuids": [],
        }
        for domain in plan.policy.resources["locality_domains"]
    ]


@pytest.mark.parametrize("observed_data_plane", ["store", "transfer_engine", None])
def test_host_staged_transport_needs_matching_runtime_metadata(
    tmp_path, observed_data_plane
):
    plan = _static_multi_node_plan(tmp_path, data_plane="store")
    recorder = _recorder_for_plan(tmp_path, plan)
    observation = _feature_observation(plan)
    observation["participants"] = [
        {
            "role": role,
            "execution_metadata": {
                "transport_data_plane": [observed_data_plane],
                "transport_protocol": ["tcp"],
            },
        }
        for role in ("producer", "consumer")
    ]

    recorder.record_training(
        {"optimizer_steps_observed": 1, "feature_contract": observation}
    )

    decisions = {item["name"]: item for item in recorder.data["capabilities"]}
    transport = decisions["tensor_transport"]
    if observed_data_plane == "store":
        assert transport["effective_state"] == "active"
        observation = transport["evidence"][-1]
        assert observation["staging_mode"] == "pinned_host_buffers"
        assert observation["host_staging_selected"] is True
    else:
        assert transport["effective_state"] == "pending_runtime_evidence"
    assert recorder.data["fallbacks"] == []


def test_host_staged_transport_does_not_hide_endpoint_without_metadata(tmp_path):
    plan = _static_multi_node_plan(tmp_path, data_plane="store")
    recorder = _recorder_for_plan(tmp_path, plan)
    observation = _feature_observation(plan)
    observation["participants"] = [
        {
            "role": role,
            "execution_metadata": {
                "transport_data_plane": ["store"],
                "transport_protocol": ["tcp"],
            },
        }
        for role in ("producer", "consumer")
    ] + [{"role": "consumer", "execution_metadata": {}}]

    recorder.record_training(
        {"optimizer_steps_observed": 1, "feature_contract": observation}
    )

    transport = next(
        item for item in recorder.data["capabilities"] if item["name"] == "tensor_transport"
    )
    assert transport["effective_state"] == "pending_runtime_evidence"


def _feature_observation(plan):
    policy = plan.workload.models["draft"]["attention"]
    parallel = plan.policy.parallelism["training"]
    return {
        "state": "validated",
        "contract_id": plan.workload.features["contract_id"],
        "producer_count": 4,
        "consumer_count": 4,
        "producer_metadata_validations": 8,
        "consumer_metadata_validations": 8,
        "consumer_tensor_validations": 8,
        "participants": [{
            "role": "consumer",
            "execution_metadata": {
                "draft_attention_backend": [parallel["attention_backend"]],
                "draft_context_parallel_size": [parallel["context_parallel"]],
            },
            "draft_attention": {
                "head_layout": policy["head_layout"],
                "context_windows": policy["context_windows"],
                "window_semantics": policy.get("window_semantics", "anchor_context"),
                "draft_block_attention": policy.get("draft_block_attention", "bidirectional"),
                "forward_observed": True,
            },
        }],
    }


def test_accelerator_inventory_records_comparable_facts_without_uuid(monkeypatch):
    monkeypatch.delenv("ASPEC_HARDWARE_LABEL", raising=False)
    monkeypatch.setattr(
        evidence.subprocess,
        "run",
        lambda *args, **kwargs: type(
            "Completed",
            (),
            {
                "returncode": 0,
                "stdout": "0, NVIDIA H200, 143771, 9.0, 570.1\n",
                "stderr": "",
            },
        )(),
    )

    inventory = evidence._accelerator_inventory()

    assert inventory == {
        "state": "observed",
        "visible_gpu_count": 1,
        "gpus": [
            {
                "index": 0,
                "reported_name": "NVIDIA H200",
                "memory_mib": 143771,
                "compute_capability": "9.0",
                "driver_version": "570.1",
            }
        ],
    }

    monkeypatch.setenv("ASPEC_HARDWARE_LABEL", "H200")
    assert evidence._accelerator_inventory()["hardware_label"] == "H200"


def _completed_cleanup():
    return {
        "state": "completed",
        "operations": [],
        "inference_manager": {
            "inflight_samples": 0,
            "inflight_sample_limit": 32,
            "peak_inflight_samples": 8,
            "peak_admitted_ready_samples": 0,
            "peak_reserved_samples": 8,
        },
        "work_lifecycle": {
            "state": "validated",
            "produced_outputs": 8,
            "committed_work": 8,
            "discarded_outputs": 0,
            "resident_outputs": 0,
            "resident_bytes": 0,
            "peak_resident_outputs": 8,
            "peak_resident_bytes": 1024,
            "optimizer_steps_committed": 1,
            "optimizer_steps_contiguous": True,
        },
    }

@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"peak_admitted_ready_samples": 63, "peak_inflight_samples": 32, "peak_reserved_samples": 95}, "active"),
        ({"peak_admitted_ready_samples": 64}, "pending_runtime_evidence"),
        ({"peak_reserved_samples": 96}, "pending_runtime_evidence"),
        ({"peak_inflight_samples": 33}, "pending_runtime_evidence"),
        ({"inflight_sample_limit": 33}, "pending_runtime_evidence"),
        ({"inflight_sample_limit": 0}, "pending_runtime_evidence"),
        ({"peak_inflight_samples": -1}, "pending_runtime_evidence"),
        ({"peak_admitted_ready_samples": None}, "pending_runtime_evidence"),
        ({"inflight_samples": 1}, "pending_runtime_evidence"),
    ],
)
def test_capacity_evidence_validates_watermark_and_independent_inflight_bound(
    tmp_path, updates, expected
):
    _plan_value, recorder = _recorder(tmp_path)
    cleanup = _completed_cleanup()
    cleanup["inference_manager"].update(updates)
    if "peak_admitted_ready_samples" in updates and updates["peak_admitted_ready_samples"] is None:
        cleanup["inference_manager"].pop("peak_admitted_ready_samples")
    recorder.record_cleanup(cleanup)
    recorder.record_training({"optimizer_steps_observed": 1})
    recorder.complete()
    decision = next(
        item for item in recorder.data["capabilities"] if item["name"] == "capacity_control"
    )
    assert decision["effective_state"] == expected


def test_node_local_capacity_evidence_validates_each_domain_independently(tmp_path):
    plan = _node_local_multi_node_plan(tmp_path)
    recorder = _recorder_for_plan(tmp_path, plan)
    nodes = ["10.0.0.1", "10.0.0.2"]
    cleanup = _completed_cleanup()
    expected_committed = plan.policy.pipeline["global_batch_size"]
    cleanup["work_lifecycle"].update(
        {
            "produced_outputs": expected_committed,
            "committed_work": expected_committed,
            "peak_resident_outputs": expected_committed,
        }
    )
    cleanup["inference_manager"].update(
        {
            "ready_watermark_by_node": {node: 32 for node in nodes},
            "inflight_samples_by_node": {node: 0 for node in nodes},
            "inflight_sample_limit_by_node": {node: 8 for node in nodes},
            "peak_inflight_samples_by_node": {node: 8 for node in nodes},
            "peak_admitted_ready_by_node": {node: 31 for node in nodes},
            "peak_reserved_by_node": {node: 39 for node in nodes},
        }
    )
    recorder.record_cleanup(cleanup)
    recorder.record_training(
        {"optimizer_steps_observed": 1, "scheduler": {"packing_enabled": False}}
    )
    recorder.complete()

    decision = next(
        item
        for item in recorder.data["capabilities"]
        if item["name"] == "capacity_control"
    )
    assert decision["effective_state"] == "active"
    assert decision["evidence"][-1]["predicate"] == (
        "per_locality_ready_credits_gate_admission_with_bounded_inflight"
    )



def _inference_engine_statuses(plan):
    backend_tokens = plan.policy.capacity["backend_tokens"]
    context_keys = (
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
    context = {key: backend_tokens[key] for key in context_keys}
    return [
        {
            "rank": rank,
            "token_capacity": {
                "profiled_max_total_tokens": 262400,
                "required_max_total_tokens": None,
                "max_prefill_tokens": (1 << 31) - 1,
                "requested_max_running_requests": None,
                "context_length": backend_tokens["backend_context_length"],
                "context": context,
            },
        }
        for rank in range(plan.policy.resources["inference"]["engines"])
    ]


@pytest.mark.parametrize("different", [False, True])
def test_backend_source_identity_is_reported_by_target_workers(tmp_path, different):
    plan, recorder = _recorder(tmp_path)
    statuses = _inference_engine_statuses(plan)
    backend = {"path": "/target/site-packages/sglang", "git_revision": None,
               "git_dirty": None, "effective_tree_sha256": "a" * 64,
               "source_file_count": 3, "missing_files": []}
    for status in statuses:
        status["runtime"] = {"backend_identity": dict(backend), "sglang_version": "0.5.21"}
    if different:
        statuses[-1]["runtime"]["backend_identity"]["effective_tree_sha256"] = "b" * 64
    arguments = dict(
        training_actor_count=plan.policy.resources["training"]["world_size"],
        inference_engine_count=len(statuses), inference_engine_statuses=statuses,
        dataset_size=128, eval_dataset_size=0, cluster_resources={"CPU": 16, "GPU": 8},
        alive_node_count=1,
    )
    if different:
        with pytest.raises(RuntimeError, match="different SGLang identities"):
            recorder.record_actor_initialization(**arguments)
    else:
        recorder.record_actor_initialization(**arguments)
        assert recorder.data["source"]["backends"]["sglang"] == backend
        assert recorder.data["source"]["package_scope"] == "draft_driver"
        assert len(recorder.data["source"]["target_runtimes"]) == len(statuses)


def test_static_multi_node_evidence_validates_actual_actor_placement(tmp_path):
    plan = _static_multi_node_plan(tmp_path)
    recorder = _recorder_for_plan(tmp_path, plan)
    training = [
        {
            "rank": rank,
            "world_size": 4,
            "node_ip": "10.0.0.1",
            "assigned_gpu_ids": [rank],
            "local_gpu_id": rank,
            "initialized": True,
        }
        for rank in range(4)
    ]
    inference = _inference_engine_statuses(plan)
    for rank, status in enumerate(inference):
        status.update(
            {
                "node_ip": "10.0.0.2",
                "engine_gpu_ids": [rank],
                "num_gpus_per_engine": 1,
                "local_gpu_id": rank,
                "initialized": True,
            }
        )

    recorder.record_actor_initialization(
        training_actor_count=4,
        training_actor_statuses=training,
        inference_engine_count=4,
        inference_engine_statuses=inference,
        dataset_size=8,
        eval_dataset_size=0,
        cluster_resources={"CPU": 32.0, "GPU": 16.0},
        alive_node_count=2,
    )

    placement = recorder.data["observations"]["runtime"]["placement"]
    assert placement["state"] == "validated"
    assert {item["node_ip"] for item in placement["training"]} == {"10.0.0.1"}
    assert {item["node_ip"] for item in placement["inference"]} == {"10.0.0.2"}


def test_static_multi_node_evidence_rejects_role_placement_drift(tmp_path):
    plan = _static_multi_node_plan(tmp_path)
    recorder = _recorder_for_plan(tmp_path, plan)
    training = [
        {
            "rank": rank,
            "node_ip": "10.0.0.1",
            "assigned_gpu_ids": [rank],
        }
        for rank in range(4)
    ]
    inference = _inference_engine_statuses(plan)
    for rank, status in enumerate(inference):
        status.update(
            {
                "node_ip": "10.0.0.1" if rank == 0 else "10.0.0.2",
                "engine_gpu_ids": [rank + 4],
                "num_gpus_per_engine": 1,
            }
        )

    with pytest.raises(RuntimeError, match="inference placement does not match"):
        recorder.record_actor_initialization(
            training_actor_count=4,
            training_actor_statuses=training,
            inference_engine_count=4,
            inference_engine_statuses=inference,
            dataset_size=8,
            eval_dataset_size=0,
            cluster_resources={"CPU": 32.0, "GPU": 16.0},
            alive_node_count=2,
        )


def test_cluster_preflight_routes_must_match_resolved_plan(tmp_path):
    plan = _static_multi_node_plan(tmp_path)
    transfer = plan.policy.capacity["transfer_storage"]
    planned_routes = transfer["resolved_rdma_routes"]
    report = {
        "state": "validated",
        "expected_node_ips": ["10.0.0.1", "10.0.0.2"],
        "locality_domains": _observed_locality_domains(plan),
        "configured_rdma_routes": planned_routes,
        "transport": {
            key: transfer[key]
            for key in (
                "data_plane",
                "protocol",
                "feature_affinity",
                "transport_stage",
                "staging_mode",
                "host_staging_selected",
                "registered_cuda_buffers_required",
                "network_gpu_direct_required",
                "host_fallback_allowed",
            )
        },
    }
    recorder = RunEvidenceRecorder(plan, tmp_path / "valid")

    recorder.record_cluster_preflight(report)

    mismatched = copy.deepcopy(report)
    mismatched["configured_rdma_routes"]["10.0.0.2"]["gid_index"] = 7
    rejecting_recorder = RunEvidenceRecorder(plan, tmp_path / "mismatch")
    with pytest.raises(RuntimeError, match="RDMA routes do not match"):
        rejecting_recorder.record_cluster_preflight(mismatched)

    transport_drift = copy.deepcopy(report)
    transport_drift["transport"]["transport_stage"] = "host_staged_tcp"
    transport_recorder = RunEvidenceRecorder(plan, tmp_path / "transport-mismatch")
    with pytest.raises(RuntimeError, match="transport does not match the Plan"):
        transport_recorder.record_cluster_preflight(transport_drift)


def test_node_local_affinity_activates_only_with_complete_runtime_evidence(tmp_path):
    plan = _node_local_multi_node_plan(tmp_path)
    recorder = _recorder_for_plan(tmp_path, plan)
    transfer = plan.policy.capacity["transfer_storage"]
    node_ips = ["10.0.0.1", "10.0.0.2"]
    recorder.record_cluster_preflight(
        {
            "state": "validated",
            "expected_node_ips": node_ips,
            "locality_domains": _observed_locality_domains(plan),
            "configured_rdma_routes": transfer["resolved_rdma_routes"],
            "transport": {
                key: transfer[key]
                for key in (
                    "data_plane",
                    "protocol",
                    "feature_affinity",
                    "transport_stage",
                    "staging_mode",
                    "host_staging_selected",
                    "registered_cuda_buffers_required",
                    "network_gpu_direct_required",
                    "host_fallback_allowed",
                )
            },
            "node_local_nvlink_probes": [
                {
                    "node_ip": node_ip,
                    "source_drained": True,
                    "transfer": {"matched": True},
                }
                for node_ip in node_ips
            ],
        }
    )
    expected_committed = plan.policy.pipeline["global_batch_size"]
    cleanup = _completed_cleanup()
    cleanup["work_lifecycle"]["committed_work"] = expected_committed
    recorder.record_cleanup(cleanup)
    feature_observation = _feature_observation(plan)
    feature_observation["consumer_count"] = 8
    feature_observation["participants"] = [
        {
            "role": "consumer",
            "feature_affinity": {"state": "validated", "violations": 0},
        }
        for _rank in range(8)
    ]
    recorder.record_training(
        {
            "optimizer_steps_observed": 1,
            "feature_contract": feature_observation,
            "scheduler": {
                "packing_enabled": False,
                "feature_affinity": "node_local_required",
                "node_local_dispatch_samples": expected_committed,
                "cross_node_dispatch_samples": 0,
                "node_local_dispatch_by_node": {
                    node_ip: expected_committed // 2 for node_ip in node_ips
                },
            },
        }
    )

    capabilities = {item["name"]: item for item in recorder.data["capabilities"]}
    affinity = capabilities["node_local_feature_affinity"]
    assert affinity["effective_state"] == "active"
    assert affinity["evidence"][-1]["cross_node_dispatch_samples"] == 0


def test_evidence_separates_initialized_and_end_to_end_capabilities(tmp_path):
    plan, recorder = _recorder(tmp_path)

    initial = json.loads(recorder.path.read_text(encoding="utf-8"))
    assert initial["schema_version"] == "aspec.run-evidence/v0alpha8"
    assert initial["state"] == "running"
    assert initial["plan_id"] == plan.plan_id
    assert len(initial["source"]["aspec"]["source_tree_sha256"]) == 64
    assert initial["source"]["aspec"]["source_file_count"] > 0
    assert initial["source"]["backends"]["sglang"].keys() == {
        "path",
        "git_revision",
        "git_dirty",
        "effective_tree_sha256",
        "source_file_count",
        "missing_files",
    }
    assert initial["observations"]["artifacts"]["plan"]["exists"] is True
    assert all(
        item["effective_state"] == "pending_runtime_evidence"
        for item in initial["capabilities"]
        if item["requirement"] == "required"
    )

    recorder.record_actor_initialization(
        training_actor_count=plan.policy.resources["training"]["world_size"],
        inference_engine_count=plan.policy.resources["inference"]["engines"],
        inference_engine_statuses=_inference_engine_statuses(plan),
        dataset_size=128,
        eval_dataset_size=0,
        dataset_admission={
            "train": {
                "state": "admitted",
                "source_samples": 130,
                "processed_samples": 128,
                "admitted_samples": 128,
                "skipped_samples": 2,
            },
            "eval": {
                "state": "not_configured",
                "source_samples": 0,
                "processed_samples": 0,
                "admitted_samples": 0,
                "skipped_samples": 0,
                "truncated_samples": 0,
            },
        },
        cluster_resources={"CPU": 16.0, "GPU": 8.0},
        alive_node_count=1,
    )

    initialized = {item["name"]: item for item in recorder.data["capabilities"]}
    assert initialized["target_feature_engine"]["effective_state"] == "active"
    assert initialized["draft_trainer"]["effective_state"] == "active"
    assert initialized["training_backend"]["effective_state"] == "active"
    assert initialized["target_features"]["effective_state"] == "pending_runtime_evidence"
    assert initialized["tensor_transport"]["effective_state"] == "pending_runtime_evidence"
    assert recorder.data["observations"]["runtime"]["dataset_admission"]["train"][
        "skipped_samples"
    ] == 2

    recorder.record_cleanup(_completed_cleanup())
    recorder.record_training(
        {
            "start_step": 0,
            "completed_step": 1,
            "optimizer_steps_observed": 1,
            "final_metrics": {"train/avg_loss": 1.25},
            "checkpoint": {"enabled": False, "exists": False},
            "feature_contract": _feature_observation(plan),
            "execution": {
                "attention_backend": plan.policy.parallelism["training"]["attention_backend"],
                "context_parallel_size": plan.policy.parallelism["training"]["context_parallel"],
            },
        }
    )
    recorder.complete()

    completed = json.loads(recorder.path.read_text(encoding="utf-8"))
    assert completed["state"] == "completed"
    assert completed["health"] == "healthy"
    assert completed["required_capabilities_pending"] == []
    assert all(
        item["effective_state"] == "active"
        for item in completed["capabilities"]
        if item["requirement"] == "required"
    )
    assert completed["fallbacks"] == []
    assert {item["capability"] for item in completed["omissions"]} == {
        "decode_training",
        "static_multi_node_execution",
        "node_local_feature_affinity",
        "gpu_direct_transport",
        "target_prefill_output",
        "producer_consumer_pipeline",
        "target_logits",
        "mixed_length_packing",
        "long_context_execution",
        "performance_measurement",
        "training_numerics",
    }


def test_required_performance_and_numerics_need_runtime_evidence(tmp_path):
    overrides = [
        "training.num_train_steps=3",
        "debug.performance_measurement_required=true",
        "debug.performance_warmup_steps=1",
        "debug.performance_drain_steps=0",
    ]
    config = load_config(config_path=str(_REFERENCE_CONFIG), cli_args=overrides)
    plan = compile_execution_plan(
        config,
        config_path=str(_REFERENCE_CONFIG),
        cli_args=overrides,
    )
    recorder = _recorder_for_plan(tmp_path, plan)
    recorder.record_actor_initialization(
        training_actor_count=plan.policy.resources["training"]["world_size"],
        inference_engine_count=plan.policy.resources["inference"]["engines"],
        inference_engine_statuses=_inference_engine_statuses(plan),
        dataset_size=128,
        eval_dataset_size=0,
        cluster_resources={"CPU": 16.0, "GPU": 8.0},
        alive_node_count=1,
    )
    cleanup = _completed_cleanup()
    cleanup["work_lifecycle"]["committed_work"] = 24
    cleanup["work_lifecycle"]["optimizer_steps_committed"] = 3
    recorder.record_cleanup(cleanup)
    recorder.record_training(
        {
            "optimizer_steps_observed": 3,
            "checkpoint": {"enabled": False, "exists": False},
            "feature_contract": _feature_observation(plan),
            "execution": {
                "attention_backend": plan.policy.parallelism["training"]["attention_backend"],
                "context_parallel_size": plan.policy.parallelism["training"]["context_parallel"],
            },
            "final_metrics": {"train/avg_loss": 1.25, "train/grad_norm": 2.5},
            "performance": {
                "state": "measured",
                "optimizer_steps": 2,
                "committed_work": 16,
                "window_wall_seconds": 4.0,
                "committed_work_per_second": 4.0,
            },
        }
    )
    recorder.complete()

    capabilities = {item["name"]: item for item in recorder.data["capabilities"]}
    assert capabilities["performance_measurement"]["effective_state"] == "active"
    assert capabilities["training_numerics"]["effective_state"] == "active"
    assert recorder.data["health"] == "healthy"


def test_cudnn_flex_attention_execution_activates_from_runtime_evidence(tmp_path):
    overrides = ["training.attention_backend=cudnn_flex_attention"]
    config = load_config(config_path=str(_REFERENCE_CONFIG), cli_args=overrides)
    plan = compile_execution_plan(
        config,
        config_path=str(_REFERENCE_CONFIG),
        cli_args=overrides,
    )
    recorder = _recorder_for_plan(tmp_path, plan)
    feature_observation = _feature_observation(plan)
    policy = plan.workload.models["draft"]["attention"]
    feature_observation["participants"] = [
        {
            "role": "consumer",
            "execution_metadata": {
                "draft_attention_backend": ["cudnn_flex_attention"],
                "draft_context_parallel_size": [1],
            },
            "draft_attention": {
                "head_layout": policy["head_layout"],
                "context_windows": policy["context_windows"],
                "window_semantics": "anchor_context",
                "draft_block_attention": "bidirectional",
                "forward_observed": True,
            },
        }
    ]

    recorder.record_training(
        {
            "optimizer_steps_observed": 1,
            "checkpoint": {"enabled": False, "exists": False},
            "feature_contract": feature_observation,
            "execution": {
                "attention_backend": "cudnn_flex_attention",
                "context_parallel_size": 1,
            },
        }
    )

    capability = {
        item["name"]: item for item in recorder.data["capabilities"]
    }["draft_attention_execution"]
    assert capability["effective_state"] == "active"
    assert capability["evidence"] == [
        {
            "predicate": "optimizer_steps_consumed_planned_attention_backend_at_planned_cp",
            "optimizer_steps": 1,
            "head_layout": policy["head_layout"],
            "attention_backend": "cudnn_flex_attention",
            "context_parallel_size": 1,
        }
    ]


@pytest.mark.parametrize("observation", ["missing", "wrong_head_layout"])
def test_full_attention_execution_requires_actual_head_layout(tmp_path, observation):
    overrides = ["training.attention_backend=cudnn_flex_attention"]
    config = load_config(config_path=str(_REFERENCE_CONFIG), cli_args=overrides)
    plan = compile_execution_plan(
        config,
        config_path=str(_REFERENCE_CONFIG),
        cli_args=overrides,
    )
    recorder = _recorder_for_plan(tmp_path, plan)
    feature_observation = _feature_observation(plan)
    policy = plan.workload.models["draft"]["attention"]
    participant = {
        "role": "consumer",
        "execution_metadata": {
            "draft_attention_backend": ["cudnn_flex_attention"],
            "draft_context_parallel_size": [1],
        },
        "draft_attention": {
            "head_layout": policy["head_layout"],
            "context_windows": policy["context_windows"],
            "window_semantics": "anchor_context",
            "draft_block_attention": "bidirectional",
            "forward_observed": True,
        },
    }
    if observation == "missing":
        del participant["draft_attention"]
    else:
        participant["draft_attention"]["head_layout"] = {
            **policy["head_layout"],
            "type": "mha",
        }
    feature_observation["participants"] = [participant]

    recorder.record_training(
        {
            "optimizer_steps_observed": 1,
            "checkpoint": {"enabled": False, "exists": False},
            "feature_contract": feature_observation,
            "execution": {
                "attention_backend": "cudnn_flex_attention",
                "context_parallel_size": 1,
            },
        }
    )

    capability = next(
        item
        for item in recorder.data["capabilities"]
        if item["name"] == "draft_attention_execution"
    )
    assert capability["requirement"] == "required"
    assert capability["effective_state"] == "pending_runtime_evidence"


def test_optimizer_step_without_contract_observation_is_incomplete(tmp_path):
    plan, recorder = _recorder(tmp_path)
    recorder.record_actor_initialization(
        training_actor_count=plan.policy.resources["training"]["world_size"],
        inference_engine_count=plan.policy.resources["inference"]["engines"],
        dataset_size=128,
        eval_dataset_size=0,
        cluster_resources={"CPU": 16.0, "GPU": 8.0},
        alive_node_count=1,
    )
    recorder.record_cleanup({"state": "completed", "operations": []})
    recorder.record_training(
        {
            "optimizer_steps_observed": 1,
            "checkpoint": {"enabled": False, "exists": False},
            "feature_contract": None,
        }
    )
    recorder.complete()

    assert recorder.data["health"] == "incomplete_evidence"
    assert recorder.data["required_capabilities_pending"] == [
        "target_features",
        "tensor_transport",
        "draft_attention_execution",
        "work_scheduler",
        "data_lifecycle",
        "capacity_control",
    ]


@pytest.mark.parametrize(
    ("requested_length", "observed_length", "expected_state"),
    [
        (131072, 129996, "active"),
        (131072, 129000, "pending_runtime_evidence"),
        (262144, 197474, "active"),
        (262144, 129000, "pending_runtime_evidence"),
    ],
)
def test_long_context_evidence_records_declared_backend_contract(
    tmp_path, requested_length, observed_length, expected_state
):
    target = tmp_path / "Qwen3.5-test"
    target.mkdir()
    (target / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "text_config": {"max_position_embeddings": 262144},
            }
        ),
        encoding="utf-8",
    )
    overrides = [
        f"model.target_model_path={target}",
        "model.native_context_length=262144",
        f"training.max_seq_length={requested_length}",
    ]
    if requested_length == 262144:
        # The backend reserves six input tokens beyond the training limit.
        # Keep that independently validated override explicit when the test
        # exercises the model's full declared 256K training capacity.
        overrides.append("inference.sglang.allow_context_length_override=true")
    config = load_config(config_path=str(_REFERENCE_CONFIG), cli_args=overrides)
    plan = compile_execution_plan(
        config,
        config_path=str(_REFERENCE_CONFIG),
        cli_args=overrides,
    )
    recorder = _recorder_for_plan(tmp_path, plan)
    recorder.record_actor_initialization(
        training_actor_count=plan.policy.resources["training"]["world_size"],
        inference_engine_count=plan.policy.resources["inference"]["engines"],
        inference_engine_statuses=_inference_engine_statuses(plan),
        dataset_size=128,
        eval_dataset_size=0,
        cluster_resources={"CPU": 16.0, "GPU": 8.0},
        alive_node_count=1,
    )
    recorder.record_cleanup(_completed_cleanup())
    recorder.record_training(
        {
            "optimizer_steps_observed": 1,
            "checkpoint": {"enabled": False, "exists": False},
            "feature_contract": _feature_observation(plan),
            "scheduler": {
                "packing_enabled": False,
                "observed_sequence_max": observed_length,
                "observed_sequence_count": 8,
            },
        }
    )
    recorder.complete()

    capability = {
        item["name"]: item for item in recorder.data["capabilities"]
    }["long_context_execution"]
    assert capability["effective_state"] == expected_state
    if expected_state != "active":
        assert capability["evidence"] == []
        return
    assert (
        capability["evidence"][0]["requested_max_sequence_length"]
        == requested_length
    )
    assert capability["evidence"][0]["target_input_reserve_tokens"] == 6
    assert capability["evidence"][0]["backend_context_length"] == requested_length + 6
    assert capability["evidence"][0]["observed_context_coverage"] == pytest.approx(
        observed_length / requested_length
    )
    assert capability["evidence"][0]["long_context_execution_threshold"] == 131072
    assert capability["evidence"][0][
        "observed_long_context_threshold_coverage"
    ] == pytest.approx(observed_length / 131072)
    assert capability["evidence"][0]["observed_training_shard_lengths"] == [
        observed_length
    ]


def test_packing_evidence_counts_documents_separately_from_rows(tmp_path):
    overrides = ["training.dflash_packing=true"]
    config = load_config(config_path=str(_REFERENCE_CONFIG), cli_args=overrides)
    plan = compile_execution_plan(
        config,
        config_path=str(_REFERENCE_CONFIG),
        cli_args=overrides,
    )
    recorder = _recorder_for_plan(tmp_path, plan)
    recorder.record_actor_initialization(
        training_actor_count=plan.policy.resources["training"]["world_size"],
        inference_engine_count=plan.policy.resources["inference"]["engines"],
        dataset_size=128,
        eval_dataset_size=0,
        cluster_resources={"CPU": 16.0, "GPU": 8.0},
        alive_node_count=1,
    )
    cleanup = _completed_cleanup()
    cleanup["work_lifecycle"]["committed_work"] = 12
    recorder.record_cleanup(cleanup)
    recorder.record_training(
        {
            "optimizer_steps_observed": 1,
            "checkpoint": {"enabled": False, "exists": False},
            "feature_contract": _feature_observation(plan),
            "scheduler": {
                "packing_enabled": True,
                "rows_dispatched": 8,
                "documents_dispatched": 12,
                "maximum_documents_per_row": 2,
                "packed_tokens": 16000,
                "useful_anchor_starts": 12000,
                "observed_sequence_max": 2042,
                "observed_sequence_count": 12,
            },
        }
    )
    recorder.complete()

    capabilities = {item["name"]: item for item in recorder.data["capabilities"]}
    assert capabilities["work_scheduler"]["evidence"][0]["committed_work"] == 12
    assert capabilities["mixed_length_packing"]["effective_state"] == "active"


def test_finite_source_tail_preserves_pipeline_and_packing_evidence(tmp_path):
    overrides = [
        "training.dflash_packing=true",
        "training.streaming_dispatch_with_training=true",
        "training.pipeline_dispatch_with_training=true",
    ]
    config = load_config(config_path=str(_REFERENCE_CONFIG), cli_args=overrides)
    plan = compile_execution_plan(
        config,
        config_path=str(_REFERENCE_CONFIG),
        cli_args=overrides,
    )
    recorder = _recorder_for_plan(tmp_path, plan)
    cleanup = _completed_cleanup()
    cleanup["work_lifecycle"].update(
        committed_work=40,
        optimizer_steps_committed=4,
    )
    recorder.record_cleanup(cleanup)
    recorder.record_training(
        {
            "optimizer_steps_observed": 4,
            "requested_source_work_budget": 40,
            "source_work_budget": 40,
            "effective_source_work": 40,
            "dropped_source_work": 0,
            "committed_source_work": 40,
            "checkpoint": {"enabled": False, "exists": False},
            "feature_contract": _feature_observation(plan),
            "execution": {
                "streaming_dispatch": True,
                "streaming_optimizer_steps": 3,
                "finite_source_tail_optimizer_steps": 1,
                "optimizer_microsteps": 7,
                "pipelined_microsteps": 1,
            },
            "scheduler": {
                "packing_enabled": True,
                "rows_dispatched": 28,
                "documents_dispatched": 40,
                "maximum_documents_per_row": 2,
                "packed_tokens": 16000,
                "useful_anchor_starts": 12000,
            },
        }
    )

    capabilities = {item["name"]: item for item in recorder.data["capabilities"]}
    pipeline = capabilities["producer_consumer_pipeline"]
    assert pipeline["effective_state"] == "active"
    assert pipeline["evidence"][0]["finite_source_tail_optimizer_steps"] == 1
    assert capabilities["mixed_length_packing"]["effective_state"] == "active"


def test_long_context_evidence_uses_consumed_packed_row_length(tmp_path):
    target = tmp_path / "Qwen3.5-test"
    target.mkdir()
    (target / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "text_config": {"max_position_embeddings": 262144},
            }
        ),
        encoding="utf-8",
    )
    overrides = [
        f"model.target_model_path={target}",
        "model.native_context_length=262144",
        "training.max_seq_length=131072",
        "training.dflash_packing=true",
    ]
    config = load_config(config_path=str(_REFERENCE_CONFIG), cli_args=overrides)
    plan = compile_execution_plan(
        config,
        config_path=str(_REFERENCE_CONFIG),
        cli_args=overrides,
    )
    recorder = _recorder_for_plan(tmp_path, plan)
    recorder.record_actor_initialization(
        training_actor_count=plan.policy.resources["training"]["world_size"],
        inference_engine_count=plan.policy.resources["inference"]["engines"],
        inference_engine_statuses=_inference_engine_statuses(plan),
        dataset_size=16,
        eval_dataset_size=0,
        cluster_resources={"CPU": 16.0, "GPU": 8.0},
        alive_node_count=1,
    )
    cleanup = _completed_cleanup()
    cleanup["work_lifecycle"]["committed_work"] = 16
    recorder.record_cleanup(cleanup)
    recorder.record_training(
        {
            "optimizer_steps_observed": 1,
            "checkpoint": {"enabled": False, "exists": False},
            "feature_contract": _feature_observation(plan),
            "scheduler": {
                "packing_enabled": True,
                "rows_dispatched": 8,
                "documents_dispatched": 16,
                "maximum_documents_per_row": 2,
                "packed_tokens": 8 * 131072,
                "useful_anchor_starts": 1,
                "observed_sequence_max": 65536,
                "observed_sequence_count": 16,
                "observed_packed_row_max": 131072,
                "observed_packed_row_count": 8,
            },
        }
    )
    recorder.complete()

    capability = {
        item["name"]: item for item in recorder.data["capabilities"]
    }["long_context_execution"]
    assert capability["effective_state"] == "active"
    observed = capability["evidence"][0]
    assert observed["predicate"] == "optimizer_step_consumed_packed_long_context_row"
    assert observed["observed_sequence_max"] == 131072
    assert observed["observed_sequence_count"] == 8
    assert observed["observed_input_sequence_max"] == 65536


def test_evidence_preserves_failure_as_terminal_state(tmp_path):
    _plan_value, recorder = _recorder(tmp_path)

    recorder.fail(RuntimeError("backend init failed"))

    failed = json.loads(recorder.path.read_text(encoding="utf-8"))
    assert failed["state"] == "failed"
    assert failed["health"] == "failed"
    assert failed["error"] == {"type": "RuntimeError", "message": "backend init failed"}
    assert failed["finished_at"] is not None
    assert failed["duration_seconds"] >= 0


def test_evidence_preserves_keyboard_interrupt_as_cancellation(tmp_path):
    _plan_value, recorder = _recorder(tmp_path)

    recorder.fail(KeyboardInterrupt())

    cancelled = json.loads(recorder.path.read_text(encoding="utf-8"))
    assert cancelled["state"] == "cancelled"
    assert cancelled["health"] == "failed"


def test_main_persists_a_terminal_record_when_runtime_fails(tmp_path, monkeypatch):
    plan = _plan()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "config.yaml").write_text("resolved: true\n", encoding="utf-8")
    save_execution_plan(plan, output_dir / "plan.json")
    args = plan.to_runtime_args()
    args.output_dir = str(output_dir)

    def fail_runtime(_args, evidence=None):
        assert evidence is not None
        raise RuntimeError("actor unavailable")

    monkeypatch.setattr("aspec.train_entry._prepare_config", lambda: (args, plan))
    monkeypatch.setattr("aspec.train_entry.ensure_ray_initialized", lambda: None)
    monkeypatch.setattr("aspec.train_entry.train_async_no_generation", fail_runtime)

    from aspec.train_entry import main

    with pytest.raises(RuntimeError, match="actor unavailable"):
        main()

    failed = json.loads((output_dir / "evidence.json").read_text(encoding="utf-8"))
    assert failed["state"] == "failed"
    assert failed["error"]["message"] == "actor unavailable"


@pytest.mark.parametrize("documents_per_row", [1, 3])
@pytest.mark.parametrize("kind", ["full", "eagle4_candidate", "dsa_sparse", "dsa_warmup"])
def test_thd_packing_evidence_uses_logical_dp_rows_and_all_consumers(tmp_path, documents_per_row, kind):
    overrides = [
        "training.dflash_packing=true", "training.attention_backend=fa4",
        "training.draft_context_parallel_size=3",
        "training.training_num_gpus_per_node=6", "training.dspark_num_anchors=510",
    ]
    frozen_candidate = kind in {"eagle4_candidate", "dsa_warmup"}
    if kind == "eagle4_candidate":
        overrides += [
            "model.draft_model_config=aspec/config/eagle4_qwen3_8b_candidate_config.json",
            "model.initial_draft_model_path=draft-asset", "inference.store_last_hidden_states=false",
        ]
    elif kind.startswith("dsa_"):
        phase = "indexer" if kind == "dsa_warmup" else "sparse"
        overrides += [f"model.draft_model_config=aspec/config/dspark_qwen3_8b_dsa_{phase}_draft_config.json"]
        if kind == "dsa_warmup":
            overrides += ["inference.store_last_hidden_states=false"]
    config = load_config(config_path=str(_REFERENCE_CONFIG), cli_args=overrides)
    plan = compile_execution_plan(config)
    thd_contract = next(item.resolved for item in plan.capabilities if item.name == "draft_thd_execution")
    recorder = _recorder_for_plan(tmp_path, plan)
    cleanup = _completed_cleanup()
    cleanup["work_lifecycle"]["committed_work"] = 4 * documents_per_row
    recorder.record_cleanup(cleanup)
    observation = _feature_observation(plan)
    consumer = {
        "role": "consumer", "execution_metadata": {
            "draft_qkv_format": ["thd"], "draft_thd_fa4_mode": [thd_contract["fa4_mode"]],
            "draft_attention_backward": [thd_contract["backward_strategy"]],
            "draft_thd_backward_validated": [True],
        },
    }
    if frozen_candidate:
        consumer["execution_metadata"].update(
            draft_attention_backward=["not_required_frozen_backbone"],
            draft_attention_backward_required=False, draft_body_frozen=True,
            draft_thd_backward_validated=[False],
        )
    observation["participants"] = [consumer, {"role": "consumer", "execution_metadata": {}}]
    summary = {
        "optimizer_steps_observed": 1,
        "execution": {"attention_backend": "fa4", "context_parallel_size": 3},
        "feature_contract": observation,
        "scheduler": {
            "packing_enabled": True, "rows_dispatched": 4, "documents_dispatched": 4 * documents_per_row,
            "maximum_documents_per_row": documents_per_row, "packed_tokens": 8000,
            "useful_anchor_starts": 6000,
        },
    }
    recorder.record_training(summary)
    states = {c["name"]: c["effective_state"] for c in recorder.data["capabilities"]}
    assert states["mixed_length_packing"] == "active"
    assert states["draft_thd_execution"] == "pending_runtime_evidence"
    observation["participants"] = [consumer, consumer]
    if frozen_candidate:
        consumer["execution_metadata"]["draft_body_frozen"] = False
        recorder.record_training(summary)
        states = {c["name"]: c["effective_state"] for c in recorder.data["capabilities"]}
        assert states["draft_thd_execution"] == "pending_runtime_evidence"
        consumer["execution_metadata"]["draft_body_frozen"] = True
    recorder.record_training(summary)
    states = {c["name"]: c["effective_state"] for c in recorder.data["capabilities"]}
    assert states["draft_thd_execution"] == "active"


def test_cudnn_thd_packing_evidence_uses_generic_kernel_contract(tmp_path):
    overrides = [
        "training.dflash_packing=true",
        "training.attention_backend=cudnn_flex_attention",
        "training.draft_context_parallel_size=2",
        "training.training_num_gpus_per_node=4",
        "training.dspark_num_anchors=512",
    ]
    config = load_config(config_path=str(_REFERENCE_CONFIG), cli_args=overrides)
    plan = compile_execution_plan(config)
    contract = next(
        item.resolved
        for item in plan.capabilities
        if item.name == "draft_thd_execution"
    )
    assert contract == {
        "qkv_format": "thd",
        "kernel_backend": "cudnn_flex_attention",
        "kernel_mode": "varlen_interval",
        "backward_strategy": "cudnn_flex_varlen",
    }

    recorder = _recorder_for_plan(tmp_path, plan)
    rows = (
        plan.policy.pipeline["accumulation_steps"]
        * plan.policy.resources["training"]["world_size"]
        // 2
    )
    cleanup = _completed_cleanup()
    cleanup["work_lifecycle"].update(
        committed_work=rows,
        optimizer_steps_committed=1,
    )
    recorder.record_cleanup(cleanup)
    observation = _feature_observation(plan)
    observation["participants"] = [
        {
            "role": "consumer",
            "execution_metadata": {
                "draft_qkv_format": ["thd"],
                "draft_thd_kernel_backend": ["cudnn_flex_attention"],
                "draft_thd_kernel_mode": ["varlen_interval"],
                "draft_attention_backward": ["cudnn_flex_varlen"],
                "draft_thd_backward_validated": [True],
            },
        }
    ]
    recorder.record_training(
        {
            "optimizer_steps_observed": 1,
            "execution": {
                "attention_backend": "cudnn_flex_attention",
                "context_parallel_size": 2,
            },
            "feature_contract": observation,
            "scheduler": {
                "packing_enabled": True,
                "rows_dispatched": rows,
                "documents_dispatched": rows,
                "maximum_documents_per_row": 1,
                "packed_tokens": 8192,
                "useful_training_positions": 4096,
            },
        }
    )

    decisions = {item["name"]: item for item in recorder.data["capabilities"]}
    assert decisions["mixed_length_packing"]["effective_state"] == "active"
    assert decisions["draft_thd_execution"]["effective_state"] == "active"
    assert decisions["draft_thd_execution"]["evidence"][-1]["kernel_backend"] == (
        "cudnn_flex_attention"
    )


@pytest.mark.parametrize("backend", ["fa4", "cudnn_flex_attention", "flex_attention"])
@pytest.mark.parametrize("observation", ["correct", "missing", "wrong_window", "wrong_semantics"])
def test_swa_execution_requires_actual_window_observation(tmp_path, backend, observation):
    raw = json.loads(_DRAFT_CONFIG.read_text())
    raw["sliding_window"] = 128
    draft = tmp_path / "swa.json"
    draft.write_text(json.dumps(raw))
    overrides = [f"model.draft_model_config={draft}", f"training.attention_backend={backend}"]
    config = load_config(config_path=str(_REFERENCE_CONFIG), cli_args=overrides)
    plan = compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG), cli_args=overrides)
    recorder = _recorder_for_plan(tmp_path, plan)
    observed = _feature_observation(plan)
    policy = plan.workload.models["draft"]["attention"]
    participant = {
        "role": "consumer",
        "execution_metadata": {"draft_attention_backend": [backend],
                               "draft_context_parallel_size": [1]},
        "draft_attention": {
            "head_layout": policy["head_layout"],
            "context_windows": policy["context_windows"],
            "window_semantics": "anchor_context",
            "draft_block_attention": "bidirectional", "forward_observed": True,
        },
    }
    if observation == "missing":
        del participant["draft_attention"]
    elif observation == "wrong_window":
        participant["draft_attention"]["context_windows"] = [None]*raw["num_hidden_layers"]
    elif observation == "wrong_semantics":
        participant["draft_attention"]["window_semantics"] = "query_relative"
    observed["participants"] = [participant]
    recorder.record_training({
        "optimizer_steps_observed": 1, "checkpoint": {"enabled": False, "exists": False},
        "feature_contract": observed,
        "execution": {"attention_backend": backend, "context_parallel_size": 1},
    })
    decision = next(c for c in recorder.data["capabilities"] if c["name"] == "draft_attention_execution")
    assert decision["requirement"] == "required"
    assert (decision["effective_state"] == "active") == (observation == "correct")


@pytest.mark.parametrize("observation", ["correct", "missing", "wrong_stage", "missing_layer"])
def test_csa_execution_requires_canonical_policy_and_layer_observation(tmp_path, observation):
    raw = json.loads(_DRAFT_CONFIG.read_text())
    raw.update(
        sliding_window=128,
        layer_types=["compressed_sparse_attention"] * raw["num_hidden_layers"],
        compress_ratios=[4] * raw["num_hidden_layers"],
        csa_training_stage="sparse",
        index_topk=64,
    )
    draft = tmp_path / "csa.json"
    draft.write_text(json.dumps(raw))
    overrides = [
        f"model.draft_model_config={draft}",
        "training.attention_backend=flex_attention",
    ]
    config = load_config(config_path=str(_REFERENCE_CONFIG), cli_args=overrides)
    plan = compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG), cli_args=overrides)
    recorder = _recorder_for_plan(tmp_path, plan)
    observed = _feature_observation(plan)
    policy = plan.workload.models["draft"]["attention"]
    layers = [
        {
            "layer": layer,
            "type": "compressed_sparse_attention",
            "variant": "csa",
            "training_stage": "sparse",
            "compress_ratio": 4,
            "compressed_group_capacity": 32,
            "index_topk": 64,
            "indexer_input_detached": True,
            "cross_layer_reuse": False,
        }
        for layer in range(raw["num_hidden_layers"])
    ]
    participant = {
        "role": "consumer",
        "execution_metadata": {
            "draft_attention_backend": ["flex_attention"],
            "draft_context_parallel_size": [1],
        },
        "draft_attention": {
            "head_layout": policy["head_layout"],
            "context_windows": policy["context_windows"],
            "window_semantics": "anchor_context",
            "draft_block_attention": "bidirectional",
            "csa": policy["csa"],
            "layers": layers,
            "forward_observed": True,
        },
    }
    if observation == "missing":
        del participant["draft_attention"]
    elif observation == "wrong_stage":
        participant["draft_attention"]["layers"][0]["training_stage"] = "dense"
    elif observation == "missing_layer":
        participant["draft_attention"]["layers"].pop()
    observed["participants"] = [participant]
    recorder.record_training(
        {
            "optimizer_steps_observed": 1,
            "checkpoint": {"enabled": False, "exists": False},
            "feature_contract": observed,
            "execution": {"attention_backend": "flex_attention", "context_parallel_size": 1},
        }
    )
    decision = next(
        item for item in recorder.data["capabilities"]
        if item["name"] == "draft_attention_execution"
    )
    assert decision["requirement"] == "required"
    assert (decision["effective_state"] == "active") == (observation == "correct")


@pytest.mark.parametrize("observation", ["correct", "missing", "wrong_stage", "missing_layer"])
def test_qsa_execution_requires_canonical_policy_and_layer_observation(tmp_path, observation):
    raw = json.loads(_DRAFT_CONFIG.read_text())
    raw.update(
        layer_types=["qwen_sparse_attention"] * raw["num_hidden_layers"],
        qsa_training_stage="sparse",
        qsa_token_budget=2048,
        qsa_compress_ratio=4,
        qsa_index_n_heads=4,
        qsa_index_head_dim=128,
        qsa_rope_dim=64,
        qsa_workspace_bytes=128 * 1024 * 1024,
    )
    draft = tmp_path / "qsa.json"
    draft.write_text(json.dumps(raw))
    overrides = [
        f"model.draft_model_config={draft}",
        "training.attention_backend=flex_attention",
    ]
    config = load_config(config_path=str(_REFERENCE_CONFIG), cli_args=overrides)
    plan = compile_execution_plan(
        config, config_path=str(_REFERENCE_CONFIG), cli_args=overrides
    )
    recorder = _recorder_for_plan(tmp_path, plan)
    observed = _feature_observation(plan)
    policy = plan.workload.models["draft"]["attention"]
    qsa = policy["qsa"]
    layers = [
        {
            "layer": layer,
            "type": "qwen_sparse_attention",
            "variant": "qsa",
            "training_stage": "sparse",
            "compress_ratio": 4,
            "token_budget": 2048,
            "block_topk": 512,
            "index_n_heads": 4,
            "index_head_dim": 128,
            "rope_dim": 64,
            "indexer_input_detached": False,
            "cross_layer_reuse": False,
            "draft_block_index_reuse": True,
            "sparse_core_backend": "flash_attn_varlen_gather",
        }
        for layer in range(raw["num_hidden_layers"])
    ]
    participant = {
        "role": "consumer",
        "execution_metadata": {
            "draft_attention_backend": ["flex_attention"],
            "draft_context_parallel_size": [1],
        },
        "draft_attention": {
            "head_layout": policy["head_layout"],
            "context_windows": policy["context_windows"],
            "window_semantics": "anchor_context",
            "draft_block_attention": "bidirectional",
            "csa": None,
            "qsa": qsa,
            "layers": layers,
            "forward_observed": True,
        },
    }
    if observation == "missing":
        del participant["draft_attention"]
    elif observation == "wrong_stage":
        participant["draft_attention"]["layers"][0]["training_stage"] = "dense"
    elif observation == "missing_layer":
        participant["draft_attention"]["layers"].pop()
    observed["participants"] = [participant]
    recorder.record_training(
        {
            "optimizer_steps_observed": 1,
            "checkpoint": {"enabled": False, "exists": False},
            "feature_contract": observed,
            "execution": {
                "attention_backend": "flex_attention",
                "context_parallel_size": 1,
            },
        }
    )
    decision = next(
        item
        for item in recorder.data["capabilities"]
        if item["name"] == "draft_attention_execution"
    )
    assert decision["requirement"] == "required"
    assert (decision["effective_state"] == "active") == (observation == "correct")


@pytest.mark.parametrize("observation", ["correct", "mixed_correct", "wrong_stage", "block_shared", "missing_policy"])
def test_dsa_activation_requires_actual_token_granular_stage_observation(tmp_path, observation):
    raw = json.loads(_DRAFT_CONFIG.read_text())
    raw.update(layer_types=["deepseek_sparse_attention"] * raw["num_hidden_layers"],
               dsa_training_stage="sparse")
    if observation == "mixed_correct":
        raw["layer_types"][0] = "full_attention"
    draft = tmp_path / "dsa.json"
    draft.write_text(json.dumps(raw))
    config = load_config(config_path=str(_REFERENCE_CONFIG), cli_args=[
        f"model.draft_model_config={draft}", "training.attention_backend=flex_attention",
    ])
    plan = compile_execution_plan(config)
    recorder = _recorder_for_plan(tmp_path, plan)
    observed = _feature_observation(plan)
    policy = plan.workload.models["draft"]["attention"]
    dsa = policy["dsa"]
    layers = [
        {"layer": layer, "type": "deepseek_sparse_attention", "variant": "dsa",
         "training_stage": "sparse", "token_budget": dsa["token_budget"],
         "index_n_heads": dsa["index_n_heads"], "index_head_dim": dsa["index_head_dim"],
         "q_lora_rank": dsa["q_lora_rank"], "indexer_input_detached": True,
         "cross_layer_reuse": False, "draft_block_index_reuse": False,
         "sparse_core_backend": "flash_attn_varlen_gather"}
        for layer in range(raw["num_hidden_layers"])
        if raw["layer_types"][layer] == "deepseek_sparse_attention"
    ]
    attention = {
        "head_layout": policy["head_layout"], "context_windows": policy["context_windows"],
        "window_semantics": "anchor_context", "draft_block_attention": "bidirectional",
        "csa": None, "qsa": None, "dsa": dsa, "layers": layers, "forward_observed": True,
    }
    if observation == "wrong_stage":
        layers[0]["training_stage"] = "dense"
    elif observation == "block_shared":
        layers[0]["draft_block_index_reuse"] = True
    elif observation == "missing_policy":
        attention.pop("dsa")
    elif observation == "mixed_correct":
        layers.insert(0, {"layer": 0, "type": "full_attention", "qkv_format": "thd"})
    observed["participants"] = [{
        "role": "consumer", "draft_attention": attention,
        "execution_metadata": {"draft_attention_backend": ["flex_attention"],
                               "draft_context_parallel_size": [1]},
    }]
    recorder.record_training({
        "optimizer_steps_observed": 1, "checkpoint": {"enabled": False, "exists": False},
        "feature_contract": observed,
        "execution": {"attention_backend": "flex_attention", "context_parallel_size": 1},
    })
    decision = next(item for item in recorder.data["capabilities"]
                    if item["name"] == "draft_attention_execution")
    assert (decision["effective_state"] == "active") == (observation in {"correct", "mixed_correct"})


def test_shared_full_source_context_is_not_confused_with_child_cp(tmp_path):
    config = load_config(config_path=str(_REFERENCE_CONFIG), save_snapshot=False)
    config.training.attention_backend = "fa4"
    config.training.draft_context_parallel_size = 2
    child = compile_execution_plan(config, source_context_parallel_size=1)
    full = copy.deepcopy(config)
    full.training.draft_context_parallel_size = 1
    source = compile_execution_plan(full)
    context = source.to_runtime_args().aspec_context_contract
    recorder = _recorder_for_plan(tmp_path, child)
    statuses = _inference_engine_statuses(child)
    for status in statuses:
        status["token_capacity"]["context"] = copy.deepcopy(context)
    kwargs = dict(
        training_actor_count=child.policy.resources["training"]["world_size"],
        inference_engine_count=child.policy.resources["inference"]["engines"],
        inference_engine_statuses=statuses, dataset_size=128, eval_dataset_size=0,
        cluster_resources={"GPU": 8.0}, alive_node_count=1,
    )
    recorder.record_actor_initialization(**kwargs, source_context_contract=context)
    observed = recorder.data["observations"]["runtime"]["backend_tokens"]
    assert observed["context"]["training_sharding"]["context_parallel_size"] == 1
    assert observed["training_context"]["training_sharding"]["context_parallel_size"] == 2
    assert observed["state"] == "validated"
    altered = copy.deepcopy(context)
    altered["backend_context_length"] += 1
    with pytest.raises(RuntimeError, match="differs from the child Plan"):
        recorder.record_actor_initialization(**kwargs, source_context_contract=altered)
    altered = copy.deepcopy(context)
    altered["training_sharding"]["context_parallel_size"] = 3
    with pytest.raises(RuntimeError, match="CP staging differs"):
        recorder.record_actor_initialization(**kwargs, source_context_contract=altered)


def test_failed_actor_evidence_preserves_backend_log_excerpt(tmp_path):
    _, recorder = _recorder(tmp_path)
    from aspec.reporting import build_run_report, render_run_report
    directory = recorder.path.parent / "backend_logs"
    directory.mkdir()
    (directory / "sglang-0-0-0.log").write_text("RuntimeError: port 47966 EADDRINUSE\n")
    recorder.fail(RuntimeError("actor unavailable"))
    assert "EADDRINUSE" in recorder.data["error"]["backend_logs"][0]["tail"]
    rendered = render_run_report(build_run_report(recorder.path.parent))
    assert "EADDRINUSE" in rendered


@pytest.mark.parametrize("tracked", [False, True])
def test_module_git_identity_requires_tracked_module_source(tmp_path, monkeypatch, tracked):
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text(".venv/\n", encoding="utf-8")
    origin = root / ("sglang/__init__.py" if tracked else ".venv/lib/site-packages/sglang/__init__.py")
    origin.parent.mkdir(parents=True)
    origin.write_text("# test package\n", encoding="utf-8")
    if tracked:
        subprocess.run(["git", "-C", str(root), "add", "sglang/__init__.py"], check=True)
    runtime = {
        "path": str(origin),
        "effective_tree_sha256": "source-hash",
        "file_count": 1,
        "missing_files": [],
    }
    monkeypatch.setattr(evidence, "module_runtime_identity", lambda *a, **kw: runtime)

    observed = evidence._module_source_identity("sglang")

    # An uncommitted editable checkout still has useful Git status; a wheel
    # nested in the application's venv does not belong to the application's Git.
    assert observed["git_revision"] is None
    assert observed["git_dirty"] is (True if tracked else None)
    assert observed["effective_tree_sha256"] == "source-hash"


def test_packed_evidence_with_missing_feature_validation_remains_pending(tmp_path):
    config = load_config(
        config_path=str(_REFERENCE_CONFIG),
        cli_args=["training.dflash_packing=true"],
    )
    plan = compile_execution_plan(config)
    recorder = _recorder_for_plan(tmp_path, plan)
    recorder.record_cleanup(_completed_cleanup())
    recorder.record_training({
        "optimizer_steps_observed": 1,
        "scheduler": {
            "packing_enabled": True,
            "rows_dispatched": 8,
            "documents_dispatched": 8,
            "maximum_documents_per_row": 1,
            "packed_tokens": 8 * 4096,
            "useful_anchor_starts": 1,
        },
    })
    capabilities = {item["name"]: item for item in recorder.data["capabilities"]}
    assert capabilities["mixed_length_packing"]["effective_state"] == "pending_runtime_evidence"
