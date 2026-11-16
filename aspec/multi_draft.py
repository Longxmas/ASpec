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

"""Compile and execute independent drafts over one bounded Target stream."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import ray
from omegaconf import OmegaConf

from aspec.config.train_config import _save_config_snapshot, load_config
from aspec.controller.eval import EVAL_CACHE_IDLE_TIMEOUT, setup_eval
from aspec.controller.inference_manager import AsyncInferenceManager
from aspec.controller.loop import _collect_feature_contract_observation, training_loop
from aspec.controller.setup import auto_calculate_training_steps, build_mooncake_config
from aspec.controller.shared_source import ExperimentController, SharedSourceController
from aspec.evidence import RunEvidenceRecorder
from aspec.inference.factory import prepare_inference_engines
from aspec.planning import ExecutionPlan, compile_execution_plan, save_execution_plan
from aspec.ray.owner_fence import _OwnerFenceProbe
from aspec.ray.placement_group import allocate_train_group, create_placement_groups
from aspec.reporting import _is_safe_experiment_name
from aspec.training.trainer_actor import TrainerActor
from aspec.transfer.mooncake.utils import launch_mooncake_master
from aspec.utils.env import get_aspec_env_vars
from aspec.utils.logging import ExperimentTracking, logger


@dataclass
class MultiDraftPlan:
    configs: dict
    plans: dict[str, ExecutionPlan]
    source_seed: int
    source_plan: ExecutionPlan | None = None

    def to_dict(self):
        payload = {
            "schema_version": "aspec.multi-draft-plan/v1",
            "experiments": {name: plan.to_dict() for name, plan in self.plans.items()},
            "shared_source": {
                "dataset_seed": self.source_seed,
                "features": (self.source_plan or next(iter(self.plans.values()))).workload.features,
                "feature_views": {
                    name: {"contract_id": child.workload.features["contract_id"],
                           "aux_hidden_state_layers": child.workload.features["aux_hidden_state_layers"]}
                    for name, child in self.plans.items()
                },
                "progress": "independent",
                "read_ownership": "registered_generation_checked_readers",
            },
            "resources": {
                "target_gpus": next(iter(self.plans.values())).to_runtime_args().inference_num_gpus,
                "draft_gpus": {
                    name: plan.to_runtime_args().world_size for name, plan in self.plans.items()
                },
            },
            "tracking": {
                "scope": "per_experiment_driver",
                "requested_backends": {
                    name: {"tensorboard": plan.to_runtime_args().use_tensorboard,
                           "wandb": plan.to_runtime_args().use_wandb}
                    for name, plan in self.plans.items()
                },
            },
        }
        if self.source_plan is not None:
            payload["shared_source"]["source_plan"] = {
                "scope": "source_layout_and_capacity_only",
                **self.source_plan.to_dict(),
            }
            payload["shared_source"]["context_views"] = {
                name: plan.to_runtime_args().draft_context_parallel_size
                for name, plan in self.plans.items()
            }
        payload["plan_id"] = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(
                    payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ).encode()
            ).hexdigest()
        )
        return payload

    @property
    def plan_id(self):
        return self.to_dict()["plan_id"]

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def to_runtime_args(self):
        args = (self.source_plan or next(iter(self.plans.values()))).to_runtime_args()
        args.output_dir = str(Path(args.output_dir).parent)
        args.seed = self.source_seed
        return args


def compile_multi_draft_plan(config) -> MultiDraftPlan:
    base = OmegaConf.to_container(config, resolve=True)
    experiments = base.pop("experiments")
    source_seed = config.dataset.dataset_seed
    if source_seed is None:
        source_seed = config.training.seed
    base["dataset"]["dataset_seed"] = int(source_seed)
    configs, plans = {}, {}
    for experiment in experiments:
        if not isinstance(experiment, dict) or set(experiment) - {"name", "model", "training", "logging"}:
            raise ValueError("each experiment accepts name, model, training and logging overrides")
        name = experiment.get("name")
        if not _is_safe_experiment_name(name):
            raise ValueError("experiment name must be a non-empty path-safe identity")
        if name in plans:
            raise ValueError(f"duplicate experiment name: {name}")
        overrides = {key: value for key, value in experiment.items() if key != "name"}
        child = load_config(base_config=OmegaConf.merge(base, overrides), save_snapshot=False,
                            source_context_parallel_size=1)
        child.output_dir = str(Path(config.output_dir) / name)
        args = compile_execution_plan(child, source_context_parallel_size=1).to_runtime_args()
        if args.report_to not in {"none", "tensorboard", "wandb"}:
            raise ValueError("logging.report_to must be none, tensorboard or wandb")
        unsupported = {
            "multi-node": args.cluster_mode != "single_node" or args.training_num_nodes != 1,
            "colocated GPU roles": args.colocate,
            "legacy sequence parallel": args.attention_backend == "usp",
            "decode": args.train_with_decode,
            "service profiling": bool(getattr(args, "topology_producer_profile_samples", 0)),
            "Store data plane": args.mooncake_data_plane != "transfer_engine",
        }
        rejected = [key for key, value in unsupported.items() if value]
        if rejected:
            raise NotImplementedError(f"shared Target training does not support: {rejected}")
        if child.training.max_seq_length != config.training.max_seq_length:
            raise ValueError("shared experiments require compatible source sequence lengths")
        for key in ("placement_strategy", "training_node_ips", "training_node_selectors"):
            if child.training[key] != config.training[key]:
                raise ValueError(f"experiment {name} changes the shared placement policy: {key}")
        for key in (
            "target_model_path",
            "target_model_backend",
            "embedding_key",
            "lm_head_key",
            "norm_key",
        ):
            if child.model[key] != config.model[key]:
                raise ValueError(f"experiment {name} changes the shared Target binding: {key}")
        configs[name] = child
        plans[name] = compile_execution_plan(child, source_context_parallel_size=1)
    if not plans:
        raise ValueError("shared source requires at least one experiment")
    from aspec.feature_contract import shared_feature_union

    features = shared_feature_union(plan.workload.features for plan in plans.values())
    heterogeneous = len({p.workload.features["contract_id"] for p in plans.values()}) > 1
    source_plan = None
    if heterogeneous or any(plan.to_runtime_args().draft_context_parallel_size > 1 for plan in plans.values()):
        source_config = load_config(base_config=OmegaConf.merge(
            next(iter(configs.values())),
            {"training": {"draft_context_parallel_size": 1, "training_num_gpus_per_node": 1}},
        ), save_snapshot=False)
        # This plan describes producer staging, not a phantom CP1 trainer
        # with the first experiment\'s GPU count and DP dispatch demand.
        source_plan = compile_execution_plan(source_config, source_feature_contract=features)
    source_context = (source_plan or next(iter(plans.values()))).to_runtime_args().aspec_context_contract
    for child in plans.values():
        context = child.to_runtime_args().aspec_context_contract
        if any(context[key] != source_context[key] for key in context if key != "training_sharding"):
            raise ValueError("shared experiments require compatible source context contracts")
    return MultiDraftPlan(configs, plans, int(source_seed), source_plan)


class _ConsumerObservations:
    def __init__(self, statuses):
        self.statuses = statuses

    def get_feature_contract_status(self):
        return self.statuses


def prepare_shared_eval(hub, manager, groups, args):
    """Capture compatible held-out features once; every child keeps its own cache."""
    states = {}
    for name, group in groups.items():
        controller = ExperimentController(hub, name)
        size = ray.get(controller.get_eval_dataset_size.remote(), timeout=10)
        states[name] = setup_eval(controller, group, args[name], size)
    missing = [n for n, state in states.items() if state.eval_enabled and not state.eval_cache_loaded]
    if not missing:
        return states
    budget = max(states[n].eval_dataset_size for n in missing)
    # First RPC also waits for a cold Ray actor's imports/constructor. Keep a
    # bounded startup budget, not a ten-second deployment-specific deadline.
    ray.get(manager.start_eval_budget.remote(budget), timeout=EVAL_CACHE_IDLE_TIMEOUT)
    if ray.get(hub.begin_eval.remote(missing), timeout=10) != budget:
        raise RuntimeError("shared eval source and consumer budgets disagree")
    progress = dict.fromkeys(missing, 0)
    last_progress = time.monotonic()
    while any(progress[n] < states[n].eval_dataset_size for n in missing):
        moved = False
        for name in missing:
            state = states[name]
            if progress[name] >= state.eval_dataset_size:
                continue
            controller = ExperimentController(hub, name)
            remaining = EVAL_CACHE_IDLE_TIMEOUT - (time.monotonic() - last_progress)
            if remaining <= 0:
                raise TimeoutError(f"shared held-out cache stopped progressing: {progress}")
            if ray.get(controller.try_dispatch_eval_batch.remote(), timeout=remaining):
                groups[name].cache_eval_samples(1)
                progress[name] += state.eval_dispatch_bs
                moved = True
        if moved:
            last_progress = time.monotonic()
        elif time.monotonic() - last_progress >= EVAL_CACHE_IDLE_TIMEOUT:
            raise TimeoutError(f"shared held-out cache stopped progressing: {progress}")
        else:
            time.sleep(0.01)
    ray.get(manager.finish_eval_budget.remote(), timeout=EVAL_CACHE_IDLE_TIMEOUT)
    ray.get(hub.end_eval.remote(), timeout=10)
    for name in missing:
        groups[name].async_save_eval_cache(states[name].eval_cache_path)
        states[name].eval_cache_loaded = True
    return states


def require_single_gpu_node(pgs):
    """Validate actual GPU placements, not the number of nodes requested in YAML."""
    nodes = set()
    for role in ("training", "inference"):
        if pgs[role] is None:
            continue
        group, bundles, _gpu_ids = pgs[role]
        locations = ray.util.placement_group_table(group).get("bundles_to_node_id", {})
        for bundle in bundles:
            node = locations.get(bundle)
            if not node:
                raise RuntimeError("shared-source GPU placement has no observed node identity")
            nodes.add(node)
    if len(nodes) != 1:
        raise NotImplementedError(
            "shared Target training requires all GPU roles on one physical node"
        )
    return next(iter(nodes))


def experiment_contract(contract, name):
    """Attribute consumer observations, retaining explicitly session-scoped source facts."""
    result = copy.deepcopy(contract)
    consumers = [
        item
        for item in result["participants"]
        if item.get("role") == "consumer" and item["participant"].startswith(name + "/")
    ]
    producers = [item for item in result["participants"] if item.get("role") == "producer"]
    result["participants"] = producers + consumers
    result["consumer_count"] = len(consumers)
    result["source_contract_id"] = result["contract_id"]
    if consumers:
        identities = {item["contract_id"] for item in consumers}
        if len(identities) != 1:
            raise RuntimeError("Experiment consumers disagree on their feature view")
        result["contract_id"] = identities.pop()
    result["producer_scope"] = "shared_session_not_experiment_work"
    for key in ("metadata_validations", "tensor_validations"):
        result["consumer_" + key] = sum(item[key] for item in consumers)
    result["data_admission"] = {
        "neutralized_rank_samples": sum(
            (item.get("data_admission") or {}).get("neutralized_rank_samples", 0)
            for item in consumers
        ),
        "participants": [
            item
            for item in result["data_admission"]["participants"]
            if item["participant"].startswith(name + "/")
        ],
    }
    transport = result["transport"]
    transport["producer_scope"] = "shared_session_not_experiment_work"
    transport["consumer_endpoints"] = len(consumers)
    transport["participants"] = [item["transport"] for item in producers + consumers]
    transport["destination_max_transfer_seconds"] = max(
        (
            item["transport"]["metrics"].get("destination_max_transfer_seconds", 0)
            for item in consumers
        ),
        default=0,
    )
    for key in (
        "destination_reads",
        "destination_releases",
        "destination_payload_bytes",
        "destination_transfer_seconds",
    ):
        transport[key] = sum(item["transport"]["metrics"][key] for item in consumers)
    transport["expected_destination_reads"] = transport["destination_reads"]
    seconds = transport["destination_transfer_seconds"]
    transport["destination_payload_bytes_per_second"] = (
        transport["destination_payload_bytes"] / seconds if seconds else None
    )
    return result



def _fence_group(group, statuses, timeout=30):
    # Single-node scope: the probe is pinned to the trainer's actual node, not
    # the driver's placement or a hostname copied from a configuration.
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    node_ip = statuses[0]["node_ip"]
    node_id = next(
        node["NodeID"]
        for node in ray.nodes()
        if node["Alive"] and node["NodeManagerAddress"] == node_ip
    )
    probe = _OwnerFenceProbe.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
    ).remote()
    deadline = time.monotonic() + timeout
    try:
        for actor in group._actor_handlers:
            ray.kill(actor, no_restart=True)
        while (remaining := deadline - time.monotonic()) > 0:
            if ray.get(
                probe.exited.remote([item["pid"] for item in statuses]),
                timeout=min(5, remaining),
            ):
                return
            time.sleep(0.05)
        raise TimeoutError("experiment read owners did not exit; source withdrawal is unsafe")
    finally:
        try:
            ray.kill(probe, no_restart=True)
        except Exception:
            logger.exception("Failed to close experiment owner probe")


def _finish_experiment(group, statuses, hub, name, *, graceful):
    """A failed graceful close must still fence owners before reader withdrawal."""
    close_error = None
    if graceful:
        try:
            ray.get([actor.close.remote() for actor in group._actor_handlers], timeout=10)
        except Exception as exc:
            close_error = f"{type(exc).__name__}: {exc}"
    _fence_group(group, statuses)
    ray.get(hub.retire.remote(name, owners_fenced=True), timeout=30)
    return close_error


def run_multi_draft(args, plan: MultiDraftPlan):
    from aspec.train_entry import (
        _get_draft_model_config,
        _prepare_run_directory,
        _prepare_vocab_mapping,
        _validate_and_configure_dflash,
    )

    root = Path(args.output_dir)
    run_id, attempt_id = f"uuid:{uuid.uuid4()}", f"uuid:{uuid.uuid4()}"
    environment = {
        "ASPEC_RUN_ID": run_id,
        "ASPEC_ATTEMPT_ID": attempt_id,
        "ASPEC_SHARED_SOURCE": "1",
        "ASPEC_SHARED_FEATURE_VIEWS": str(int(len({p.workload.features["contract_id"] for p in plan.plans.values()}) > 1)),
    }
    previous_environment = {key: os.environ.get(key) for key in environment}
    experiment_args = {}
    destination_configs = {}
    recorders = {}
    trackers = {}
    groups, actor_statuses, consumer_statuses = {}, {}, {}
    results = {}
    terminal_recorders = set()
    fenced_groups = set()
    pgs = {}
    allocation_in_progress = False
    training_started = False
    engines, manager, hub, manager_future = [], None, None, None
    source_args = copy.copy(args)
    source_args.aspec_run_id, source_args.aspec_attempt_id = run_id, attempt_id
    source_args.aspec_shared_consumer_contracts = {
        name: child.workload.features["contract_id"] for name, child in plan.plans.items()
    }
    started = time.monotonic()
    allocated_gpus = (
        sum(p.to_runtime_args().world_size for p in plan.plans.values()) + args.inference_num_gpus
    )
    session = {
        "schema_version": "aspec.multi-draft-evidence/v1",
        "plan_id": plan.plan_id,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "state": "running",
        "experiments": {},
    }

    def persist_session():
        destination = root / "evidence.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(session, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, destination)

    persist_session()
    try:
        os.environ.update(environment)
        for name, child_plan in plan.plans.items():
            child = child_plan.to_runtime_args()
            child.aspec_run_id, child.aspec_attempt_id = run_id, attempt_id
            child.aspec_shared_source = True
            child.aspec_experiment_name = name
            _prepare_run_directory(child.output_dir)
            _save_config_snapshot(plan.configs[name])
            save_execution_plan(child_plan, Path(child.output_dir) / "plan.json")
            child.draft_model_config_obj = _get_draft_model_config(child)
            _validate_and_configure_dflash(child, child.draft_model_config_obj)
            experiment_args[name] = child
            recorders[name] = RunEvidenceRecorder(
                child_plan, child.output_dir, run_id=run_id, attempt_id=attempt_id
            )

        # One immutable full-layout source; each experiment chooses CP views.
        # No duplicate Target forward or full-feature destination copy.
        if any(getattr(child, "draft_context_parallel_size", 1) > 1 for child in experiment_args.values()):
            source_args.draft_context_parallel_size = 1
            source_args.sp_size = 1
        source_args.training_num_gpus_per_node = sum(
            child.world_size for child in experiment_args.values()
        )
        source_args.world_size = source_args.training_num_gpus_per_node
        source_args.num_train_steps = None  # demand comes from the per-experiment budgets
        source_args.aspec_shared_source = True
        hub = SharedSourceController.options(
            runtime_env={"env_vars": get_aspec_env_vars()},
            name=f"shared-source-{run_id.removeprefix('uuid:')}",
            namespace="aspec",
        ).remote(source_args, experiment_args)
        session["control"] = {
            "actor_name": f"shared-source-{run_id.removeprefix('uuid:')}",
            "namespace": "aspec",
        }
        persist_session()
        # The shared coordinator loads once and resolves its retained states;
        # the driver resolves the same horizons for trainer initialization.
        dataset_size = ray.get(hub.load_dataset.remote(source_args))
        for child in experiment_args.values():
            auto_calculate_training_steps(child, dataset_size)
        pgs = create_placement_groups(source_args)
        session["gpu_node_id"] = require_single_gpu_node(pgs)
        launch_mooncake_master(source_args)
        mooncake = build_mooncake_config(source_args)
        source_args.draft_model_config_obj = next(
            iter(experiment_args.values())
        ).draft_model_config_obj
        _validate_and_configure_dflash(source_args, source_args.draft_model_config_obj)
        training_pg, bundles, gpu_ids = pgs["training"]
        offset = 0
        init_refs = {}
        for name, child in experiment_args.items():
            width = child.world_size
            allocation_in_progress = True
            group = allocate_train_group(
                child,
                1,
                width,
                (training_pg, bundles[offset : offset + width], gpu_ids[offset : offset + width]),
                TrainerActor,
            )
            groups[name] = group
            allocation_in_progress = False
            actor_statuses[name] = group.get_status()
            offset += width
            destination = copy.deepcopy(mooncake)
            destination.gpu_buffer_size = build_mooncake_config(child).gpu_buffer_size
            destination_configs[name] = destination
            init_refs[name] = group.async_init(child, "training", mooncake_config=destination)
        source_args.aspec_read_owners = [
            owner for statuses in actor_statuses.values() for owner in statuses
        ]
        engines, engine_refs = prepare_inference_engines(source_args, pgs["inference"], mooncake)
        ray.get(engine_refs)  # Target initialization failure is a shared fault.
        producer_placements = ray.get([engine.get_status.remote() for engine in engines])
        for name, child in experiment_args.items():
            group = groups[name]
            controller = ExperimentController(hub, name)
            try:
                mapping = _prepare_vocab_mapping(
                    child, child.draft_model_config_obj, controller,
                )
                ray.get(init_refs[name])
                if mapping is not None:
                    group.set_vocab_buffers(*mapping)
                queues = ray.get(controller.get_train_queues.remote())
                group.set_train_queues(
                    queues, destination_configs[name], child.per_dp_rank_batch_size, lifecycle_observer=controller
                )
                if getattr(child, "eval_data_path", None):
                    group.set_eval_queues(
                        ray.get(controller.get_eval_queues.remote()), destination_configs[name],
                        1, lifecycle_observer=controller,
                    )
                actor_statuses[name] = group.get_status()
                # Initialize SDKs serially; the concurrent loops only use their
                # own explicit handles. Prepared but unused runs close below.
                trackers[name] = ExperimentTracking(child)
            except Exception as exc:
                recorders[name].fail(exc)
                terminal_recorders.add(name)
                _finish_experiment(group, actor_statuses[name], hub, name, graceful=False)
                fenced_groups.add(name)
                recorders[name].record_cleanup(
                    {
                        "state": "completed",
                        "read_owners_fenced": True,
                        "work_lifecycle": ray.get(
                            controller.get_work_lifecycle_status.remote(), timeout=10
                        ),
                    }
                )
                results[name] = {
                    "state": "failed",
                    "phase": "initialization",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                session["experiments"][name] = results[name]
                persist_session()
                continue
            recorders[name].record_actor_initialization(
                training_actor_count=child.world_size,
                training_actor_statuses=actor_statuses[name],
                inference_engine_count=len(engines),
                inference_engine_statuses=producer_placements,
                dataset_size=dataset_size,
                eval_dataset_size=(ray.get(controller.get_eval_dataset_size.remote())
                                   if getattr(child, "eval_data_path", None) else 0),
                dataset_admission=ray.get(controller.get_dataset_admission_evidence.remote()),
                cluster_resources=ray.cluster_resources(),
                alive_node_count=sum(node["Alive"] for node in ray.nodes()),
                source_context_contract=source_args.aspec_context_contract,
            )
        running_names = [name for name in groups if name not in results]
        if not running_names:
            raise RuntimeError("all Draft experiments failed to initialize")
        manager = AsyncInferenceManager.options(
            runtime_env={"env_vars": get_aspec_env_vars()}
        ).remote(
            source_args,
            hub,
            inference_engines=engines,
            max_concurrent_batches=source_args.max_concurrent_batches,
        )
        manager_future = manager.run.remote()
        training_started = True  # Eval capture can already publish registered source readers.
        eval_states = prepare_shared_eval(
            hub, manager, {n: groups[n] for n in running_names}, experiment_args,
        ) if any(getattr(experiment_args[n], "eval_data_path", None) for n in running_names) else {}

        def train_experiment(name):
            child, group = experiment_args[name], groups[name]
            controller = ExperimentController(hub, name)
            failure = None
            summary = None
            try:
                summary = training_loop(
                    child,
                    controller,
                    manager,
                    group,
                    dataset_size=dataset_size,
                    eval_dataset_size=eval_states[name].eval_dataset_size if name in eval_states else 0,
                    inference_future=manager_future,
                    eval_state=eval_states.get(name),
                    tracking=trackers[name],
                )
                consumer_statuses[name] = group.get_feature_contract_status()
                summary["state"] = "completed"
                summary["no_new_work"] = summary["optimizer_steps_observed"] == 0
                return summary
            except Exception as exc:
                failure = exc
                try:
                    cancellation = ray.get(hub.cancellation_reason.remote(name), timeout=5)
                except Exception:
                    cancellation = None  # Preserve the original failure if the hub died.
                recorders[name].fail(
                    KeyboardInterrupt(cancellation) if cancellation is not None else exc
                )
                terminal_recorders.add(name)
                return {
                    "error": f"{type(exc).__name__}: {exc}",
                    "state": "cancelled" if cancellation is not None else "failed",
                }
            finally:
                try:
                    close_error = _finish_experiment(
                        group, actor_statuses[name], hub, name, graceful=failure is None
                    )
                    fenced_groups.add(name)
                    cleanup = {
                        "state": "completed",
                        "read_owners_fenced": True,
                        "graceful_close_error": close_error,
                        "work_lifecycle": ray.get(
                            controller.get_work_lifecycle_status.remote(), timeout=10
                        ),
                    }
                    try:
                        trackers[name].close(exit_code=int(failure is not None))
                    except Exception as tracking_error:
                        # A logging failure is not an unsafe source-owner fence.
                        # Keep healthy siblings running and retain a prior cancel/error.
                        cleanup["tracking_close_error"] = f"{type(tracking_error).__name__}: {tracking_error}"
                        if failure is None:
                            failure = tracking_error
                            summary.update(state="failed", error=cleanup["tracking_close_error"])
                            recorders[name].fail(tracking_error)
                            terminal_recorders.add(name)
                    recorders[name].record_cleanup(cleanup)
                    if failure is None:
                        summary["cleanup"] = cleanup
                        recorders[name].record_training(summary)
                        recorders[name].complete()
                        terminal_recorders.add(name)
                except BaseException as cleanup_error:
                    # Failed fencing is a shared fault, not an isolated Draft failure.
                    try:
                        ray.get(
                            hub.set_inference_error.remote(
                                f"unsafe experiment cleanup ({name}): {cleanup_error}"
                            ),
                            timeout=5,
                        )
                    except Exception:
                        pass
                    raise

        executor = ThreadPoolExecutor(max_workers=len(running_names))
        try:
            training_started = True
            futures = {executor.submit(train_experiment, name): name for name in running_names}
            for future in as_completed(futures):
                name = futures[future]
                results[name] = future.result()
                session["experiments"][name] = results[name]
                persist_session()
        except BaseException:
            # Executor context-manager exit waits for workers. Fence their Ray
            # tasks first so cancellation cannot wait for the entire horizon.
            for name in groups:
                try:
                    ray.get(hub.request_cancel.remote(name, "session interrupted"), timeout=10)
                except Exception:
                    pass  # The failed coordinator must not prevent process fencing.
                for actor in groups[name]._actor_handlers:
                    try:
                        ray.kill(actor, no_restart=True)
                    except Exception:
                        pass
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
        ray.get(manager.stop.remote(), timeout=10)
        producer_error = None
        try:
            ray.get(
                manager_future,
                timeout=getattr(source_args, "inference_shutdown_timeout_seconds", 120),
            )
        except Exception as exc:
            producer_error = exc
        manager_status = ray.get(manager.get_status.remote(), timeout=10)
        session["inference_manager"] = manager_status
        session["inference_metrics"] = ray.get(manager.flush_metrics.remote(), timeout=10)
        source_status = ray.get(
            hub.finish_production.remote(inflight_samples=manager_status["inflight_samples"]),
            timeout=30,
        )
        session["shared_source"] = source_status
        if producer_error is not None:
            raise producer_error
        if source_status["outstanding_observations"] or source_status["pending_production"]:
            raise RuntimeError("shared source did not reach a storage-free terminal state")
        failed = any("error" in summary for summary in results.values())
        working_names = [
            name
            for name, summary in results.items()
            if "error" not in summary and not summary["no_new_work"]
        ]
        if not failed and working_names:
            statuses = [
                dict(status, participant=f"{name}/{status['participant']}")
                for name, items in consumer_statuses.items()
                if name in working_names
                for status in items
            ]
            observed_args = copy.copy(source_args)
            observed_args.world_size = sum(
                experiment_args[name].world_size for name in working_names
            )
            contract = _collect_feature_contract_observation(
                observed_args, _ConsumerObservations(statuses), engines
            )
            session["feature_contract"] = contract
            session["feature_contract_consumer_scope"] = "experiments_with_new_work"
        for name, summary in results.items():
            if "error" not in summary:
                cleanup = {
                    **summary["cleanup"],
                    "state": "completed",
                    "inference_manager": manager_status,
                    "inference_manager_scope": "shared_session",
                    "work_lifecycle": ray.get(
                        ExperimentController(hub, name).get_work_lifecycle_status.remote()
                    ),
                }
                if summary["no_new_work"]:
                    pass  # No new reads means no fresh capability/throughput claim.
                elif not failed:
                    summary["feature_contract"] = experiment_contract(contract, name)
                else:
                    child = copy.copy(source_args)
                    child.world_size = experiment_args[name].world_size
                    child.aspec_shared_reader = name
                    statuses = [
                        dict(item, participant=f"{name}/{item['participant']}")
                        for item in consumer_statuses[name]
                    ]
                    scoped = _collect_feature_contract_observation(
                        child, _ConsumerObservations(statuses), engines
                    )
                    summary["feature_contract"] = experiment_contract(scoped, name)
                summary["cleanup"] = cleanup
                recorders[name].record_cleanup(cleanup)
                recorders[name].record_training(summary)
                recorders[name].complete()
                terminal_recorders.add(name)
        session["state"] = "completed_with_failed_experiments" if failed else "completed"
    except BaseException as exc:
        session["state"] = "failed"
        session["error"] = f"{type(exc).__name__}: {exc}"
        for name, recorder in recorders.items():
            if name not in terminal_recorders:
                recorder.fail(exc)
        raise
    finally:
        # Killing only this driver's actors fences reads before registered source
        # allocations can be destroyed, including initialization or SIGTERM failures.
        cleanup_errors = []

        def cleanup(action):
            try:
                action()
            except BaseException as exc:
                cleanup_errors.append(f"{type(exc).__name__}: {exc}")

        for name, group in groups.items():
            if name in fenced_groups:
                continue
            if name in actor_statuses:

                def fence(name=name, group=group):
                    _fence_group(group, actor_statuses[name])
                    fenced_groups.add(name)

                cleanup(fence)
            else:
                for actor in group._actor_handlers:
                    cleanup(lambda: ray.kill(actor, no_restart=True))
        if manager is not None:
            cleanup(lambda: ray.kill(manager, no_restart=True))
        owners_fenced = not allocation_in_progress and len(fenced_groups) == len(groups)
        session["read_owners_fenced"] = owners_fenced
        # Before the source starts, no experiment can own a published reader.
        # Eval cache generation also starts the source, before optimizer work.
        if owners_fenced or not training_started:
            if hub is not None:
                # Withdraw registered eval/train rights while source endpoints
                # are still alive; killing engines first loses the ACK boundary.
                for name in experiment_args:
                    cleanup(lambda name=name: ray.get(
                        hub.retire.remote(name, owners_fenced=True), timeout=30
                    ))
            for engine in engines:
                force_source = (owners_fenced and session["state"] != "completed"
                                and getattr(source_args, "inference_engine_type", None) == "sgl")
                shutdown_kwargs = {"force": True} if force_source else {}
                cleanup(lambda: ray.get(engine.shutdown.remote(**shutdown_kwargs), timeout=30))
                cleanup(lambda: ray.kill(engine, no_restart=True))
            master = getattr(source_args, "_mooncake_master_actor", None)
            if master is not None:
                cleanup(lambda: ray.get(master.shutdown.remote(), timeout=10))
                cleanup(lambda: ray.kill(master, no_restart=True))
            if hub is not None:
                cleanup(lambda: ray.get(hub.close_queues.remote(), timeout=10))
                cleanup(lambda: ray.kill(hub, no_restart=True))
            unique_pgs = {item[0].id: item[0] for item in pgs.values() if item is not None}
            for pg in unique_pgs.values():
                cleanup(lambda pg=pg: ray.util.remove_placement_group(pg))
        else:
            cleanup_errors.append("read owners are not fenced; source teardown was withheld")
        for tracking in trackers.values():
            cleanup(lambda tracking=tracking: tracking.close(exit_code=int(session["state"] != "completed")))
        session["cleanup_errors"] = cleanup_errors
        if cleanup_errors and session["state"] == "completed":
            session["state"] = "failed"
        session["wall_seconds"] = time.monotonic() - started
        # Fixed-budget experiment cost, including startup/cleanup; not a claim
        # about GPU utilization or the scheduler's actual billed reservation.
        session["budget_gpu_hours"] = allocated_gpus * session["wall_seconds"] / 3600
        try:
            persist_session()
        finally:
            for key, value in previous_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    if session["state"] != "completed":
        raise RuntimeError(
            f"multi-Draft session ended in {session['state']}; see {root / 'evidence.json'}"
        )
    return session
