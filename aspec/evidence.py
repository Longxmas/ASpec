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

"""Machine-readable runtime evidence for one ASpec execution plan."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from aspec.planning import ExecutionPlan
from aspec.runtime_control import is_termination_exception
from aspec.runtime_identity import (
    SGLANG_RUNTIME_RELATIVE_PATHS,
    module_runtime_identity,
)

EVIDENCE_SCHEMA_VERSION = "aspec.run-evidence/v0alpha8"
_PACKAGE_NAMES = (
    "ling-aspec",
    "ling-aspec-runtime",
    "ling-aspec-target",
    "torch",
    "ray",
    "sglang",
    "mooncake-transfer-engine",
    "flash-attn-4",
    "nvidia-cutlass-dsl",
    "quack-kernels",
    "transformers",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    result: dict[str, Any] = {
        "path": str(resolved),
        "exists": resolved.exists(),
    }
    if resolved.is_file():
        result["sha256"] = _sha256_file(resolved)
        result["size_bytes"] = resolved.stat().st_size
    return result


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in _PACKAGE_NAMES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _accelerator_inventory() -> dict[str, Any]:
    """Collect stable GPU facts without making Evidence creation depend on NVML."""

    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"state": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        return {
            "state": "unavailable",
            "error": f"nvidia-smi exited with {completed.returncode}: {detail}",
        }
    gpus = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            return {
                "state": "unavailable",
                "error": f"unexpected nvidia-smi GPU row: {line!r}",
            }
        index, name, memory_mib, compute_capability, driver = fields
        try:
            parsed_memory_mib = int(memory_mib)
            parsed_index = int(index)
        except ValueError:
            return {
                "state": "unavailable",
                "error": f"invalid nvidia-smi GPU row: {line!r}",
            }
        gpus.append(
            {
                "index": parsed_index,
                "reported_name": name,
                "memory_mib": parsed_memory_mib,
                "compute_capability": compute_capability,
                "driver_version": driver,
            }
        )
    result = {
        "state": "observed" if gpus else "unavailable",
        "visible_gpu_count": len(gpus),
        "gpus": gpus,
    }
    hardware_label = os.environ.get("ASPEC_HARDWARE_LABEL")
    if hardware_label:
        result["hardware_label"] = hardware_label
    return result


def _source_identity(package_root: Path | None = None) -> dict[str, Any]:
    package = package_root or Path(__file__).resolve().parent
    root = package.parent

    def _git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )

    try:
        # An installed wheel may live inside somebody else's Git checkout.
        # Only report live Git facts when this exact package file is tracked.
        tracked = _git("ls-files", "--error-unmatch", "--", f"{package.name}/evidence.py")
        if tracked.returncode == 0:
            revision = _git("rev-parse", "HEAD")
            status = _git("status", "--porcelain=v1")
            identity = {
                "revision": revision.stdout.strip() if revision.returncode == 0 else None,
                "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
            }
        else:
            identity = {"revision": None, "dirty": None}
    except (OSError, subprocess.SubprocessError):
        identity = {"revision": None, "dirty": None}

    build_info = package / "_build_info.json"
    identity["build"] = json.loads(build_info.read_text()) if build_info.is_file() else None
    try:
        identity["version"] = importlib.metadata.version("ling-aspec-runtime")
    except importlib.metadata.PackageNotFoundError:
        identity["version"] = None
    source_files = [path for path in package.rglob("*") if path.suffix in
                    {".py", ".json", ".patch", ".yaml", ".txt", ".jsonl"}]
    if identity["revision"] is not None:
        source_files.append(root / "pyproject.toml")
        source_files.extend((root / "requirements").glob("*.txt"))
        source_files.extend((root / "configs/recipes").glob("*.yaml"))
        source_files.extend((root / "examples/data").glob("*.jsonl"))
        source_files.append(root / "tools/recipe_catalog.json")
    source_files = sorted(path for path in source_files if path.is_file())
    digest = hashlib.sha256()
    for path in source_files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
    identity["source_tree_sha256"] = digest.hexdigest()
    identity["source_file_count"] = len(source_files)
    return identity


def _module_source_identity(module: str) -> dict[str, Any]:
    runtime_identity = module_runtime_identity(
        module,
        relative_paths=(
            SGLANG_RUNTIME_RELATIVE_PATHS if module == "sglang" else None
        ),
    )
    identity: dict[str, Any] = {
        "path": runtime_identity["path"],
        "git_revision": None,
        "git_dirty": None,
        "effective_tree_sha256": runtime_identity["effective_tree_sha256"],
        "source_file_count": runtime_identity["file_count"],
        "missing_files": runtime_identity["missing_files"],
    }
    if runtime_identity["path"] is None:
        return identity
    path = Path(runtime_identity["path"])
    try:
        root = subprocess.run(
            ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
        if root.returncode != 0:
            return identity
        root_path = root.stdout.strip()
        # A wheel installed in this project's .venv is inside our Git worktree,
        # but its source and revision still belong to the installed backend.
        # Only attach Git facts when that repository actually tracks the module.
        relative_path = path.resolve().relative_to(Path(root_path).resolve())
        tracked = subprocess.run(
            ["git", "-C", root_path, "ls-files", "--error-unmatch", "--",
             relative_path.as_posix()],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
        if tracked.returncode != 0:
            return identity
        revision = subprocess.run(
            ["git", "-C", root_path, "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
        status = subprocess.run(
            ["git", "-C", root_path, "status", "--porcelain=v1"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return identity
    identity["git_revision"] = (
        revision.stdout.strip() if revision.returncode == 0 else None
    )
    identity["git_dirty"] = bool(status.stdout.strip()) if status.returncode == 0 else None
    return identity


class RunEvidenceRecorder:
    """Persist facts without turning the evidence artifact into runtime control state."""

    def __init__(
        self,
        plan: ExecutionPlan,
        output_dir: str | os.PathLike[str],
        *,
        run_id: str | None = None,
        attempt_id: str | None = None,
    ) -> None:
        self._plan = plan
        self._started_monotonic = time.monotonic()
        self.path = Path(output_dir).expanduser() / "evidence.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing ASpec evidence: {self.path}"
            )

        plan_dict = plan.to_dict()
        self._data: dict[str, Any] = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "run_id": run_id or f"uuid:{uuid.uuid4()}",
            "attempt_id": attempt_id or f"uuid:{uuid.uuid4()}",
            "plan_id": plan.plan_id,
            "state": "running",
            "health": "pending",
            "started_at": _utc_now(),
            "finished_at": None,
            "duration_seconds": None,
            "source": {
                "plan": plan_dict["source"],
                "aspec": _source_identity(),
                "packages": _package_versions(),
                "package_scope": "draft_driver",
                "backends": {"sglang": _module_source_identity("sglang")},
                "host": {
                    "hostname": platform.node(),
                    "system": platform.system(),
                    "release": platform.release(),
                    "machine": platform.machine(),
                    "python": sys.version.split()[0],
                },
            },
            "capabilities": [dict(item) for item in plan_dict["capabilities"]],
            "fallbacks": list(plan_dict["fallbacks"]),
            "omissions": list(plan_dict["omissions"]),
            "observations": {
                "artifacts": {
                    "config": _artifact(self.path.parent / "config.yaml"),
                    "plan": _artifact(self.path.parent / "plan.json"),
                },
                "runtime": {"accelerators": _accelerator_inventory()},
                "training": None,
                "cleanup": None,
            },
            "error": None,
        }
        for capability in self._data["capabilities"]:
            capability["evidence"] = []
        self._write()

    @property
    def data(self) -> Mapping[str, Any]:
        return self._data

    def _write(self) -> None:
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(self._data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)

    def _activate(self, name: str, observed: Mapping[str, Any]) -> None:
        capability = next(item for item in self._data["capabilities"] if item["name"] == name)
        capability["effective"] = capability["resolved"]
        capability["effective_state"] = "active"
        capability["evidence"].append(dict(observed))

    @staticmethod
    def _validate_dataset_admission(
        dataset_admission: Mapping[str, Any],
        *,
        dataset_size: int,
        eval_dataset_size: int,
    ) -> dict[str, dict[str, Any]]:
        admitted: dict[str, dict[str, Any]] = {}
        for split, expected in (("train", dataset_size), ("eval", eval_dataset_size)):
            split_observation = dataset_admission.get(split)
            if not isinstance(split_observation, Mapping):
                raise RuntimeError(
                    f"Dataset admission Evidence is missing the {split!r} split"
                )
            if split_observation.get("admitted_samples") != expected:
                raise RuntimeError(
                    "Dataset admission Evidence disagrees with runtime dataset size: "
                    f"split={split}, observed={split_observation.get('admitted_samples')}, "
                    f"runtime={expected}"
                )
            admitted[split] = dict(split_observation)
        return admitted

    def record_dataset_admission(
        self,
        *,
        dataset_size: int,
        eval_dataset_size: int,
        dataset_admission: Mapping[str, Any],
    ) -> None:
        """Persist source-to-runtime accounting before expensive actor startup."""

        admitted = self._validate_dataset_admission(
            dataset_admission,
            dataset_size=dataset_size,
            eval_dataset_size=eval_dataset_size,
        )
        runtime = self._data["observations"]["runtime"]
        runtime.update(
            {
                "phase": "dataset_loaded",
                "dataset_size": dataset_size,
                "eval_dataset_size": eval_dataset_size,
                "dataset_admission": admitted,
            }
        )
        self._write()

    def record_actor_initialization(
        self,
        *,
        training_actor_count: int,
        training_actor_statuses: Sequence[Mapping[str, Any]] | None = None,
        inference_engine_count: int,
        inference_engine_statuses: Sequence[Mapping[str, Any]] | None = None,
        dataset_size: int,
        eval_dataset_size: int,
        dataset_admission: Mapping[str, Any] | None = None,
        cluster_resources: Mapping[str, float],
        alive_node_count: int,
        source_context_contract: Mapping[str, Any] | None = None,
    ) -> None:
        expected_training = self._plan.policy.resources["training"]["world_size"]
        expected_inference = self._plan.policy.resources["inference"]["engines"]
        if training_actor_count != expected_training:
            raise RuntimeError(
                "Initialized training actor count does not match the plan: "
                f"{training_actor_count} != {expected_training}"
            )
        if inference_engine_count != expected_inference:
            raise RuntimeError(
                "Initialized inference engine count does not match the plan: "
                f"{inference_engine_count} != {expected_inference}"
            )

        placement = self._validate_actor_placement(
            training_actor_statuses=training_actor_statuses,
            inference_engine_statuses=inference_engine_statuses,
        )
        target_runtimes = [dict(status["runtime"]) for status in (inference_engine_statuses or ())
                           if isinstance(status.get("runtime"), Mapping)]
        if target_runtimes:
            backend_identities = [item.get("backend_identity") for item in target_runtimes]
            if (len(target_runtimes) != inference_engine_count
                    or any(not isinstance(item, Mapping) or not item.get("effective_tree_sha256")
                           or item.get("missing_files") for item in backend_identities)
                    or len({item["effective_tree_sha256"] for item in backend_identities}) != 1):
                raise RuntimeError("Target workers reported incomplete or different SGLang identities")
            self._data["source"]["backends"]["sglang"] = dict(backend_identities[0])
            self._data["source"]["target_runtimes"] = target_runtimes

        prior_runtime = self._data["observations"]["runtime"]
        if dataset_admission is not None:
            admitted = self._validate_dataset_admission(
                dataset_admission,
                dataset_size=dataset_size,
                eval_dataset_size=eval_dataset_size,
            )
        else:
            admitted = prior_runtime.get("dataset_admission")

        backend_tokens = self._plan.policy.capacity["backend_tokens"]
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
        training_context = {key: backend_tokens[key] for key in context_keys}
        planned_context = training_context
        if source_context_contract is not None:
            # The shared source is full-layout; child CP describes destination
            # ownership, not the producer's physical staging. All other
            # context semantics must remain identical to the child Plan.
            planned_context = dict(source_context_contract)
            if any(planned_context.get(key) != training_context[key]
                   for key in context_keys if key != "training_sharding"):
                raise RuntimeError("Shared source context differs from the child Plan")
            source_cp = self._plan.policy.parallelism["inference"]["source_context_parallel_size"]
            if planned_context.get("training_sharding", {}).get("context_parallel_size") != source_cp:
                raise RuntimeError("Shared source CP staging differs from the Plan")
        observed_token_capacity: list[dict[str, Any]] = []
        context_state = "not_collected"
        if inference_engine_statuses is not None:
            if len(inference_engine_statuses) != inference_engine_count:
                raise RuntimeError(
                    "Collected inference engine status count does not match initialized "
                    f"engines: {len(inference_engine_statuses)} != {inference_engine_count}"
                )
            offline = self._plan.policy.backends["target_features"] == "offline_replay"
            for status in inference_engine_statuses:
                if offline:
                    if (status.get("engine_type") != "offline" or status.get("engine_gpu_ids")
                            or status.get("dataset_identity") != self._plan.workload.data["offline_dataset_identity"]):
                        raise RuntimeError("Offline replay actor does not match the planned dataset/CPU placement")
                    continue
                capacity = status.get("token_capacity")
                if not isinstance(capacity, Mapping):
                    raise RuntimeError(
                        f"SGLang engine {status.get('rank')} did not report token capacity"
                    )
                observed_context = capacity.get("context")
                if observed_context != planned_context:
                    raise RuntimeError(
                        "Observed SGLang context contract does not match the Plan: "
                        f"rank={status.get('rank')}, planned={planned_context}, "
                        f"observed={observed_context}"
                    )
                if capacity.get("context_length") != backend_tokens[
                    "backend_context_length"
                ]:
                    raise RuntimeError(
                        "Observed SGLang context length does not match the Plan: "
                        f"rank={status.get('rank')}, "
                        f"planned={backend_tokens['backend_context_length']}, "
                        f"observed={capacity.get('context_length')}"
                    )
                observed_token_capacity.append(
                    {
                        "rank": status.get("rank"),
                        "profiled_max_total_tokens": capacity.get(
                            "profiled_max_total_tokens"
                        ),
                        "required_max_total_tokens": capacity.get(
                            "required_max_total_tokens"
                        ),
                        "max_prefill_tokens": capacity.get("max_prefill_tokens"),
                        "requested_max_running_requests": capacity.get(
                            "requested_max_running_requests"
                        ),
                        "context_length": capacity.get("context_length"),
                    }
                )
            context_state = "not_applicable_offline" if offline else "validated"

        accelerators = prior_runtime.get("accelerators")
        cluster_preflight = prior_runtime.get("cluster_preflight")
        self._data["observations"]["runtime"] = {
            "phase": "actors_initialized",
            "accelerators": accelerators,
            "cluster_preflight": cluster_preflight,
            "training_actor_count": training_actor_count,
            "inference_engine_count": inference_engine_count,
            "dataset_size": dataset_size,
            "eval_dataset_size": eval_dataset_size,
            "dataset_admission": admitted or None,
            "placement": placement,
            "backend_tokens": {
                "state": context_state,
                "context": planned_context,
                **({"training_context": training_context, "context_scope": "shared_source"}
                   if source_context_contract is not None else {}),
                "engines": observed_token_capacity,
            },
            "ray": {
                "alive_node_count": alive_node_count,
                "cluster_resources": {
                    key: value
                    for key, value in sorted(cluster_resources.items())
                    if key in {"CPU", "GPU", "memory", "object_store_memory"}
                    or key.startswith("accelerator_type:")
                },
            },
        }
        self._activate(
            "target_feature_engine",
            {
                 "predicate": ("all_planned_offline_replay_initializers_completed"
                              if self._plan.policy.backends["target_features"] == "offline_replay"
                              else "all_planned_sglang_engine_initializers_completed"),
                "observed_engine_count": inference_engine_count,
            },
        )
        trainer_observation = {
            "predicate": "all_planned_training_actor_initializers_completed",
            "observed_actor_count": training_actor_count,
        }
        self._activate("draft_trainer", trainer_observation)
        self._activate("training_backend", trainer_observation)
        self._write()

    def _validate_actor_placement(
        self,
        *,
        training_actor_statuses: Sequence[Mapping[str, Any]] | None,
        inference_engine_statuses: Sequence[Mapping[str, Any]] | None,
    ) -> dict[str, Any]:
        """Validate actual rank/node/GPU placement against a static cluster Plan."""

        cluster = self._plan.policy.resources.get("cluster", {})
        static_multi_node = cluster.get("mode") == "static_multi_node"
        if static_multi_node and training_actor_statuses is None:
            raise RuntimeError("static multi-node Evidence requires trainer placement status")
        if static_multi_node and inference_engine_statuses is None:
            raise RuntimeError("static multi-node Evidence requires inference placement status")

        training = [dict(item) for item in (training_actor_statuses or ())]
        inference = [dict(item) for item in (inference_engine_statuses or ())]
        if training and len(training) != self._plan.policy.resources["training"]["world_size"]:
            raise RuntimeError("trainer placement status count does not match the Plan")
        if inference and len(inference) != self._plan.policy.resources["inference"]["engines"]:
            raise RuntimeError("inference placement status count does not match the Plan")

        detailed_training = bool(training) and all(
            item.get("node_ip") and "assigned_gpu_ids" in item for item in training
        )
        detailed_inference = bool(inference) and all(
            item.get("node_ip")
            and "engine_gpu_ids" in item
            and "num_gpus_per_engine" in item
            for item in inference
        )
        if static_multi_node and not detailed_training:
            raise RuntimeError("static multi-node trainer placement status is incomplete")
        if static_multi_node and not detailed_inference:
            raise RuntimeError("static multi-node inference placement status is incomplete")

        ranks = sorted(int(item.get("rank", -1)) for item in training)
        if training and ranks != list(range(len(training))):
            raise RuntimeError(f"trainer ranks are not contiguous: {ranks}")
        inference_ranks = sorted(int(item.get("rank", -1)) for item in inference)
        if inference and inference_ranks != list(range(len(inference))):
            raise RuntimeError(
                f"inference engine ranks are not contiguous: {inference_ranks}"
            )

        training_allocations: list[tuple[str, int]] = []
        for status in training if detailed_training else ():
            assigned = [int(value) for value in status.get("assigned_gpu_ids") or ()]
            if len(assigned) != 1:
                raise RuntimeError(
                    f"trainer rank {status.get('rank')} must own exactly one Ray GPU, "
                    f"observed={assigned}"
                )
            training_allocations.append((str(status.get("node_ip")), assigned[0]))

        inference_allocations: list[tuple[str, int]] = []
        for status in inference if detailed_inference else ():
            assigned = [int(value) for value in status.get("engine_gpu_ids") or ()]
            expected = int(status.get("num_gpus_per_engine") or 0)
            if len(assigned) != expected or expected <= 0:
                raise RuntimeError(
                    f"inference rank {status.get('rank')} GPU placement is incomplete: "
                    f"expected={expected}, observed={assigned}"
                )
            inference_allocations.extend(
                (str(status.get("node_ip")), gpu_id) for gpu_id in assigned
            )

        all_allocations = training_allocations + inference_allocations
        if len(all_allocations) != len(set(all_allocations)):
            raise RuntimeError(
                "actual actor placement assigns the same node-local GPU more than once"
            )

        if static_multi_node:
            expected_training: dict[str, int] = {}
            expected_inference: dict[str, int] = {}
            for node in cluster.get("nodes", ()):
                node_ip = str(node["address"])
                expected_training[node_ip] = int(node.get("training_gpus", 0))
                expected_inference[node_ip] = int(node.get("inference_gpus", 0))

            def allocation_counts(values: Sequence[tuple[str, int]]) -> dict[str, int]:
                return {
                    node_ip: sum(1 for observed_ip, _gpu_id in values if observed_ip == node_ip)
                    for node_ip in sorted(set(expected_training) | set(expected_inference))
                }

            observed_training = allocation_counts(training_allocations)
            observed_inference = allocation_counts(inference_allocations)
            if observed_training != expected_training:
                raise RuntimeError(
                    "actual trainer placement does not match the Plan: "
                    f"expected={expected_training}, observed={observed_training}"
                )
            if observed_inference != expected_inference:
                raise RuntimeError(
                    "actual inference placement does not match the Plan: "
                    f"expected={expected_inference}, observed={observed_inference}"
                )

            training_node_by_rank = {
                int(item["rank"]): str(item["node_ip"]) for item in training
            }
            inference_node_by_rank = {
                int(item["rank"]): str(item["node_ip"]) for item in inference
            }
            for domain in self._plan.policy.resources.get("locality_domains", ()):
                node_ip = str(domain["node_address"])
                invalid_training = [
                    rank
                    for rank in domain.get("training_world_ranks", ())
                    if training_node_by_rank.get(int(rank)) != node_ip
                ]
                invalid_inference = [
                    rank
                    for rank in domain.get("target_engine_ranks", ())
                    if inference_node_by_rank.get(int(rank)) != node_ip
                ]
                if invalid_training or invalid_inference:
                    raise RuntimeError(
                        "actual rank placement does not match the Plan locality domain: "
                        f"domain={domain.get('domain_id')!r}, node={node_ip!r}, "
                        f"trainer_ranks={invalid_training}, "
                        f"target_engine_ranks={invalid_inference}"
                    )

        return {
            "state": (
                "validated"
                if detailed_training and detailed_inference
                else "partially_collected"
                if training or inference
                else "not_collected"
            ),
            "training": sorted(training, key=lambda item: int(item.get("rank", -1))),
            "inference": sorted(inference, key=lambda item: int(item.get("rank", -1))),
            "locality_domains": copy.deepcopy(
                list(self._plan.policy.resources.get("locality_domains", ()))
            ),
        }

    def record_cluster_preflight(self, report: Mapping[str, Any]) -> None:
        planned = self._plan.policy.resources.get("cluster", {})
        if planned.get("mode") != "static_multi_node":
            raise RuntimeError(
                "cluster preflight evidence is only valid for static_multi_node plans"
            )
        if report.get("state") != "validated":
            raise RuntimeError("cluster preflight report is not validated")
        planned_ips = sorted(
            str(node["address"]) for node in planned.get("nodes", ())
        )
        if sorted(report.get("expected_node_ips", ())) != planned_ips:
            raise RuntimeError(
                "cluster preflight membership does not match the Plan: "
                f"planned={planned_ips}, observed={report.get('expected_node_ips')}"
            )
        planned_domains = list(
            self._plan.policy.resources.get("locality_domains", ())
        )
        observed_domains = list(report.get("locality_domains", ()))
        domain_keys = (
            "domain_id",
            "kind",
            "node_address",
            "training_world_ranks",
            "draft_dp_ranks",
            "draft_parallel_group_size",
            "target_engine_ranks",
            "target_tensor_parallel_size",
            "dispatch_samples_per_microstep",
            "ready_watermark",
        )
        planned_geometry = [
            {key: domain.get(key) for key in domain_keys} for domain in planned_domains
        ]
        observed_geometry = [
            {key: domain.get(key) for key in domain_keys} for domain in observed_domains
        ]
        if observed_geometry != planned_geometry or any(
            not domain.get("ray_node_id") for domain in observed_domains
        ):
            raise RuntimeError(
                "cluster preflight locality geometry does not match the Plan"
            )
        planned_routes = self._plan.policy.capacity.get("transfer_storage", {}).get(
            "resolved_rdma_routes", {}
        )
        if dict(report.get("configured_rdma_routes") or {}) != dict(planned_routes):
            raise RuntimeError(
                "cluster preflight RDMA routes do not match the Plan: "
                f"planned={planned_routes}, "
                f"observed={report.get('configured_rdma_routes')}"
            )
        planned_transport = self._plan.policy.capacity.get("transfer_storage", {})
        observed_transport = dict(report.get("transport") or {})
        transport_keys = (
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
        transport_mismatches = {
            key: {
                "planned": planned_transport.get(key),
                "observed": observed_transport.get(key),
            }
            for key in transport_keys
            if planned_transport.get(key) != observed_transport.get(key)
        }
        if transport_mismatches:
            raise RuntimeError(
                "cluster preflight transport does not match the Plan: "
                f"{transport_mismatches}"
            )
        self._data["observations"]["runtime"]["cluster_preflight"] = dict(report)
        self._activate(
            "static_multi_node_execution",
            {
                "predicate": "all_static_nodes_passed_production_preflight",
                "node_ips": planned_ips,
                "transport": dict(report.get("transport") or {}),
            },
        )
        self._write()

    def record_cleanup(self, cleanup: Mapping[str, Any]) -> None:
        self._data["observations"]["cleanup"] = dict(cleanup)
        self._write()

    def record_training(self, summary: Mapping[str, Any]) -> None:
        training = dict(summary)
        training.pop("cleanup", None)
        self._data["observations"]["training"] = training
        feature_contract = training.get("feature_contract") or {}
        execution = training.get("execution") or {}
        planned_contract_id = self._plan.workload.features.get("contract_id")
        contract_validated = (
            feature_contract.get("state") == "validated"
            and feature_contract.get("contract_id") == planned_contract_id
        )
        if training.get("optimizer_steps_observed", 0) > 0 and contract_validated:
            transfer_observation = {
                 "predicate": ("optimizer_step_consumed_validated_offline_features_via_mooncake"
                              if self._plan.policy.backends["target_features"] == "offline_replay"
                              else "optimizer_step_consumed_validated_sglang_features_via_mooncake"),
                "optimizer_steps_observed": training["optimizer_steps_observed"],
                "feature_contract_id": planned_contract_id,
                "producer_count": feature_contract["producer_count"],
                "consumer_count": feature_contract["consumer_count"],
                "producer_metadata_validations": feature_contract[
                    "producer_metadata_validations"
                ],
                "consumer_metadata_validations": feature_contract[
                    "consumer_metadata_validations"
                ],
                "consumer_tensor_validations": feature_contract[
                    "consumer_tensor_validations"
                ],
            }
            self._activate("target_features", transfer_observation)

            participants = feature_contract.get("participants", [])

            def _observed_values(role: str, key: str) -> set[Any]:
                values: set[Any] = set()
                matching = [item for item in participants if item.get("role") == role]
                for item in matching:
                    values.update(
                        (item.get("execution_metadata") or {}).get(key, [])
                    )
                return values if matching else set()

            transfer = self._plan.policy.capacity["transfer_storage"]
            selected_host_staging = (
                self._plan.policy.resources["cluster"]["mode"] == "static_multi_node"
                and transfer["host_staging_selected"]
            )
            host_staging_observed = True
            for role in ("producer", "consumer"):
                matching = [item for item in participants if item.get("role") == role]
                if not matching or not all(
                    set((item.get("execution_metadata") or {}).get(
                        "transport_data_plane", []
                    )) == {"store"}
                    and set((item.get("execution_metadata") or {}).get(
                        "transport_protocol", []
                    )) == {"tcp"}
                    for item in matching
                ):
                    host_staging_observed = False
                    break
            if not selected_host_staging or host_staging_observed:
                self._activate(
                    "tensor_transport",
                    {
                        **transfer_observation,
                        "data_plane": transfer["data_plane"],
                        "protocol": transfer["protocol"],
                        "staging_mode": transfer["staging_mode"],
                        "host_staging_selected": transfer["host_staging_selected"],
                    },
                )

            direct_output = self._plan.policy.pipeline["target_feature_output"]
            if (
                direct_output == "direct_to_transport"
                and _observed_values("producer", "target_feature_output")
                == {"direct_to_transport"}
                and _observed_values("consumer", "target_feature_output")
                == {"direct_to_transport"}
            ):
                self._activate(
                    "target_prefill_output",
                    {
                        "predicate": "producer_and_consumer_observed_direct_target_output",
                        "producer_metadata_validations": feature_contract[
                            "producer_metadata_validations"
                        ],
                        "consumer_metadata_validations": feature_contract[
                            "consumer_metadata_validations"
                        ],
                    },
                )

            transport_observation = feature_contract.get("transport") or {}
            if (
                transfer["data_plane"] == "transfer_engine"
                and transfer["protocol"] in {"nvlink_intra", "rdma"}
                and transfer["host_fallback_allowed"] is False
                and _observed_values("consumer", "transport_data_plane")
                == {"transfer_engine"}
                and _observed_values("consumer", "transport_protocol")
                == {transfer["protocol"]}
                and transport_observation.get("state") == "validated"
                and transport_observation.get("strict_transport") is True
                and transport_observation.get("destination_reads", 0) > 0
                and transport_observation.get("destination_payload_bytes", 0) > 0
                and transport_observation.get("transfer_failures", 0) == 0
            ):
                self._activate(
                    "gpu_direct_transport",
                    {
                        "predicate": (
                            "validated_features_crossed_strict_"
                            f"{transfer['protocol']}_transfer_engine"
                        ),
                        "protocol": transfer["protocol"],
                        "destination_reads": transport_observation[
                            "destination_reads"
                        ],
                        "destination_payload_bytes": transport_observation[
                            "destination_payload_bytes"
                        ],
                        "destination_transfer_seconds": transport_observation[
                            "destination_transfer_seconds"
                        ],
                        "destination_payload_bytes_per_second": transport_observation[
                            "destination_payload_bytes_per_second"
                        ],
                        "ring_wait_seconds": transport_observation[
                            "ring_wait_seconds"
                        ],
                        "consumer_tensor_validations": feature_contract[
                            "consumer_tensor_validations"
                        ],
                    },
                )

            planned_attention = self._plan.policy.parallelism["training"]
            planned_attention_backend = planned_attention["attention_backend"]
            policy = self._plan.workload.models["draft"].get("attention") or {}
            planned_head_layout = policy.get("head_layout")
            planned_windows = policy.get("context_windows", [])
            swa_requested = any(window is not None for window in planned_windows)
            planned_csa = policy.get("csa")
            planned_qsa = policy.get("qsa")
            planned_dsa = policy.get("dsa")
            expected_csa_layers = [
                {
                    "layer": layer_id,
                    "type": "compressed_sparse_attention",
                    "variant": "csa",
                    "training_stage": planned_csa["training_stage"],
                    "compress_ratio": policy["compress_ratios"][layer_id],
                    "index_topk": planned_csa["index_topk"],
                    "indexer_input_detached": True,
                    "cross_layer_reuse": False,
                }
                for layer_id, layer_type in enumerate(policy.get("layer_types", []))
                if planned_csa is not None
                and layer_type == "compressed_sparse_attention"
            ]
            expected_qsa_layers = [
                {
                    "layer": layer_id,
                    "type": "qwen_sparse_attention",
                    "variant": "qsa",
                    "training_stage": planned_qsa["training_stage"],
                    "compress_ratio": planned_qsa["compress_ratio"],
                    "token_budget": planned_qsa["token_budget"],
                    "block_topk": planned_qsa["block_topk"],
                    "index_n_heads": planned_qsa["index_n_heads"],
                    "index_head_dim": planned_qsa["index_head_dim"],
                    "rope_dim": planned_qsa["rope_dim"],
                    "indexer_input_detached": False,
                    "cross_layer_reuse": False,
                    "draft_block_index_reuse": True,
                    "sparse_core_backend": "flash_attn_varlen_gather",
                }
                for layer_id, layer_type in enumerate(policy.get("layer_types", []))
                if planned_qsa is not None and layer_type == "qwen_sparse_attention"
            ]
            consumers = [item for item in participants if item.get("role") == "consumer"]
            expected_dsa_layers = [
                {"layer": layer_id, "type": "deepseek_sparse_attention", "variant": "dsa",
                 "training_stage": planned_dsa["training_stage"],
                 "token_budget": planned_dsa["token_budget"],
                 "index_n_heads": planned_dsa["index_n_heads"],
                 "index_head_dim": planned_dsa["index_head_dim"],
                 "q_lora_rank": planned_dsa["q_lora_rank"],
                 "indexer_input_detached": True, "cross_layer_reuse": False,
                 "draft_block_index_reuse": False,
                 "sparse_core_backend": "flash_attn_varlen_gather"}
                for layer_id, layer_type in enumerate(policy.get("layer_types", []))
                if planned_dsa is not None and layer_type == "deepseek_sparse_attention"
            ]
            def attention_observed(item):
                observed = item.get("draft_attention") or {}
                if (
                    observed.get("head_layout") != planned_head_layout
                    or observed.get("context_windows") != planned_windows
                    or observed.get("window_semantics") != policy.get("window_semantics", "anchor_context")
                    or observed.get("draft_block_attention") != policy.get("draft_block_attention", "bidirectional")
                    or observed.get("forward_observed") is not True
                ):
                    return False
                if planned_csa is None and planned_qsa is None and planned_dsa is None:
                    return True
                if observed.get("csa") != planned_csa or observed.get("qsa") != planned_qsa:
                    return False
                if observed.get("dsa") != planned_dsa:
                    return False
                actual_layers = observed.get("layers") or []
                if planned_dsa is not None:
                    # A legal mixed full/SWA+DSA schedule also reports ordinary
                    # attention layers. Only indexed layers belong to this
                    # per-indexer activation check; do not discard other sparse
                    # families, which must still make a DSA-only plan fail.
                    actual_layers = [layer for layer in actual_layers if layer.get("type") in {
                        "compressed_sparse_attention", "qwen_sparse_attention", "deepseek_sparse_attention"
                    }]
                expected_layers = expected_csa_layers + expected_qsa_layers + expected_dsa_layers
                return len(actual_layers) == len(expected_layers) and all(
                    all(actual.get(key) == value for key, value in expected.items())
                    for actual, expected in zip(actual_layers, expected_layers, strict=True)
                )

            selected_policy_requested = (
                swa_requested or planned_csa is not None or planned_qsa is not None or planned_dsa is not None
            )
            # Activation is an execution claim, so every consumer must report the
            # physical head layout even for full attention.  Plan-time validation
            # alone cannot prove which Q/KV topology actually reached a forward.
            attention_policy_observed = bool(consumers) and all(
                attention_observed(item) for item in consumers
            )
            if (
                (
                    planned_attention_backend in {"fa4", "cudnn_flex_attention"}
                    or selected_policy_requested
                )
                and attention_policy_observed
                and execution.get("attention_backend") == planned_attention_backend
                and execution.get("context_parallel_size")
                == planned_attention["context_parallel"]
                and _observed_values("consumer", "draft_attention_backend")
                == {planned_attention_backend}
                and _observed_values("consumer", "draft_context_parallel_size")
                == {planned_attention["context_parallel"]}
            ):
                predicate = (
                    "optimizer_steps_consumed_fa4_features_at_planned_cp"
                    if planned_attention_backend == "fa4"
                    else "optimizer_steps_consumed_selected_draft_attention_at_planned_cp"
                    if selected_policy_requested
                    else "optimizer_steps_consumed_planned_attention_backend_at_planned_cp"
                )
                self._activate(
                    "draft_attention_execution",
                    {
                        "predicate": predicate,
                        **({"context_windows": planned_windows, "window_semantics": policy.get("window_semantics", "anchor_context")}
                           if swa_requested else {}),
                        **({"csa": planned_csa, "layers": expected_csa_layers}
                           if planned_csa is not None else {}),
                        **({"qsa": planned_qsa, "layers": expected_qsa_layers}
                           if planned_qsa is not None else {}),
                        **({"dsa": planned_dsa, "layers": expected_dsa_layers}
                           if planned_dsa is not None else {}),
                        "optimizer_steps": training["optimizer_steps_observed"],
                        "head_layout": planned_head_layout,
                        "attention_backend": planned_attention_backend,
                        "context_parallel_size": planned_attention[
                            "context_parallel"
                        ],
                    },
                )

            pipeline = self._plan.policy.pipeline
            pipeline_requested = pipeline["pipeline_dispatch_with_training"]
            streaming_steps = execution.get("streaming_optimizer_steps", 0)
            finite_tail_steps = execution.get(
                "finite_source_tail_optimizer_steps", 0
            )
            finite_tail_validated = (
                finite_tail_steps >= 0
                and training.get("source_work_budget") is not None
                and training.get("committed_source_work")
                == training.get("effective_source_work")
                and training.get("requested_source_work_budget")
                == training.get("effective_source_work")
                + training.get("dropped_source_work", 0)
            )
            streaming_validated = (
                pipeline["streaming_dispatch_with_training"]
                and execution.get("streaming_dispatch") is True
                and streaming_steps > 0
                and (
                    streaming_steps == training["optimizer_steps_observed"]
                    or (
                        finite_tail_validated
                        and streaming_steps + finite_tail_steps
                        == training["optimizer_steps_observed"]
                    )
                )
            )
            prefix_validated = (
                not pipeline_requested
                or execution.get("pipelined_microsteps", 0) > 0
            )
            if streaming_validated and prefix_validated:
                self._activate(
                    "producer_consumer_pipeline",
                    {
                        "predicate": "trainers_waited_while_bounded_target_work_was_dispatched",
                        "streaming_optimizer_steps": execution[
                            "streaming_optimizer_steps"
                        ],
                        "finite_source_tail_optimizer_steps": finite_tail_steps,
                        "pipelined_microsteps": execution.get(
                            "pipelined_microsteps", 0
                        ),
                    },
                )

        cleanup = self._data["observations"]["cleanup"] or {}
        lifecycle = cleanup.get("work_lifecycle") or {}
        observed_steps = training.get("optimizer_steps_observed", 0)
        scheduler = training.get("scheduler") or {}
        expected_committed = (
            scheduler.get("documents_dispatched")
            if scheduler.get("packing_enabled")
            else observed_steps * self._plan.policy.pipeline["global_batch_size"]
        )
        scheduler_validated = (
            lifecycle.get("state") == "validated"
            and lifecycle.get("committed_work") == expected_committed
            and lifecycle.get("optimizer_steps_committed") == observed_steps
            and lifecycle.get("optimizer_steps_contiguous") is True
        )
        if scheduler_validated:
            resume = training.get("resume") or {}
            scheduler_observation = {
                "predicate": "each_successful_optimizer_step_committed_exact_unique_work",
                "optimizer_steps": observed_steps,
                "committed_work": expected_committed,
            }
            if resume.get("mode") == "exact":
                scheduler_observation.update(
                    {
                        "resume_mode": "exact",
                        "restored_optimizer_step": resume.get(
                            "restored_optimizer_step"
                        ),
                        "restored_committed_work": resume.get(
                            "restored_committed_work"
                        ),
                        "restored_work_sha256": resume.get(
                            "committed_work_sha256"
                        ),
                        "dataset_fingerprint": resume.get("dataset_fingerprint"),
                    }
                )
            self._activate(
                "work_scheduler",
                scheduler_observation,
            )
            planned_affinity = self._plan.policy.pipeline.get(
                "feature_affinity", "any"
            )
            if planned_affinity == "node_local_required":
                consumers = [
                    item
                    for item in feature_contract.get("participants", [])
                    if item.get("role") == "consumer"
                ]
                consumer_affinity_validated = bool(consumers) and all(
                    (item.get("feature_affinity") or {}).get("state")
                    == "validated"
                    and (item.get("feature_affinity") or {}).get("violations") == 0
                    for item in consumers
                )
                planned_nodes = {
                    str(node["address"])
                    for node in self._plan.policy.resources["cluster"].get(
                        "nodes", []
                    )
                    if int(node.get("training_gpus", 0)) > 0
                }
                dispatch_by_node = scheduler.get("node_local_dispatch_by_node") or {}
                preflight = self._data["observations"]["runtime"].get(
                    "cluster_preflight", {}
                )
                probed_nodes = {
                    str(item.get("node_ip"))
                    for item in preflight.get("node_local_nvlink_probes", [])
                    if item.get("source_drained") is True
                    and (item.get("transfer") or {}).get("matched") is True
                }
                if (
                    scheduler.get("feature_affinity") == "node_local_required"
                    and scheduler.get("node_local_dispatch_samples")
                    == expected_committed
                    and scheduler.get("cross_node_dispatch_samples") == 0
                    and set(dispatch_by_node) == planned_nodes
                    and all(int(dispatch_by_node[node]) > 0 for node in planned_nodes)
                    and consumer_affinity_validated
                    and probed_nodes == planned_nodes
                ):
                    self._activate(
                        "node_local_feature_affinity",
                        {
                            "predicate": (
                                "all_features_consumed_on_source_node_over_strict_nvlink"
                            ),
                            "node_ips": sorted(planned_nodes),
                            "dispatched_samples": expected_committed,
                            "dispatch_by_node": dict(dispatch_by_node),
                            "cross_node_dispatch_samples": 0,
                        },
                    )
            packed_rows_validated = False
            if scheduler.get("packing_enabled"):
                optimizer_microsteps = execution.get("optimizer_microsteps")
                expected_rows = (
                    optimizer_microsteps
                    if isinstance(optimizer_microsteps, int)
                    and optimizer_microsteps > 0
                    else observed_steps
                    * self._plan.policy.pipeline["accumulation_steps"]
                ) * (
                    self._plan.policy.resources["training"]["world_size"]
                    // self._plan.policy.capacity["backend_tokens"][
                        "training_sharding"
                    ]["context_parallel_size"]
                )
                packed_rows_validated = (
                    contract_validated
                    and scheduler.get("rows_dispatched") == expected_rows
                    and scheduler.get("documents_dispatched") == expected_committed
                    # A full-length document can fill a THD row by itself;
                    # execution evidence is not a multi-document packing gain.
                    and scheduler.get("maximum_documents_per_row", 0) >= 1
                    and scheduler.get("packed_tokens", 0) > 0
                    and scheduler.get("useful_training_positions", scheduler.get("useful_anchor_starts", 0)) > 0
                )
                if packed_rows_validated:
                    consumers = [
                        item for item in feature_contract.get("participants", [])
                        if item.get("role") == "consumer"
                    ]
                    frozen_candidate = (
                        self._plan.workload.objective.get("family") == "eagle4"
                        and self._plan.workload.objective.get("eagle4", {}).get("stage") == "candidate_head"
                    )
                    thd_decision = next((item for item in self._plan.capabilities if item.name == "draft_thd_execution"), None)
                    thd_contract = (thd_decision.resolved if thd_decision is not None else None) or {
                        "qkv_format": "thd", "kernel_backend": "fa4",
                        "kernel_mode": "fixed", "fa4_mode": "fixed",
                        "backward_strategy": "fa4_fixed_sparse"
                    }
                    expected_backend = thd_contract.get("kernel_backend", "fa4")
                    expected_modes = set(thd_contract.get(
                        "kernel_modes",
                        thd_contract.get(
                            "fa4_modes",
                            [thd_contract.get("kernel_mode", thd_contract.get("fa4_mode"))],
                        ),
                    ))
                    expected_backward = set(thd_contract.get("backward_strategies", [thd_contract["backward_strategy"]]))
                    frozen_candidate = frozen_candidate or expected_backward == {"not_required_frozen_backbone"}

                    def thd_consumer_observed(item):
                        metadata = item.get("execution_metadata") or {}
                        observed_modes = metadata.get(
                            "draft_thd_kernel_mode",
                            metadata.get("draft_thd_fa4_mode", []),
                        )
                        observed_backends = metadata.get("draft_thd_kernel_backend")
                        if observed_backends is None:
                            observed_backends = (
                                ["fa4"] if metadata.get("draft_thd_fa4_mode") else []
                            )
                        backward_valid = (
                            (
                                metadata.get("draft_attention_backward_required") is False
                                and metadata.get("draft_body_frozen") is True
                                and set(metadata.get("draft_attention_backward", []))
                                == {"not_required_frozen_backbone"}
                            )
                            if frozen_candidate
                            else (
                                set(metadata.get("draft_attention_backward", []))
                                == expected_backward
                                and metadata.get("draft_thd_backward_validated")
                                and all(metadata["draft_thd_backward_validated"])
                            )
                        )
                        return (
                            set(metadata.get("draft_qkv_format", [])) == {"thd"}
                            and set(observed_backends) == {expected_backend}
                            and set(observed_modes) == expected_modes
                            and backward_valid
                        )

                    thd_observed = bool(consumers) and all(
                        thd_consumer_observed(item) for item in consumers
                    )
                    if contract_validated and thd_observed:
                        self._activate("draft_thd_execution", {
                            "predicate": ("packed_optimizer_steps_consumed_thd_forward_with_frozen_backbone"
                                          if frozen_candidate else "packed_optimizer_steps_consumed_thd_forward_and_backward"),
                            **thd_contract,
                        })
                    self._activate(
                        "mixed_length_packing",
                        {
                            "predicate": "controller_rows_preserved_document_work_and_identity",
                            **scheduler,
                        },
                    )

            backend_tokens = self._plan.policy.capacity["backend_tokens"]
            requested_length = backend_tokens["requested_max_sequence_length"]
            effective_length = backend_tokens[
                "effective_training_max_sequence_length"
            ]
            reserve_tokens = backend_tokens["target_input_reserve_tokens"]
            observed_input_length = scheduler.get("observed_sequence_max", 0)
            if scheduler.get("packing_enabled") and packed_rows_validated:
                observed_length = scheduler.get("observed_packed_row_max", 0)
                observed_count = scheduler.get("observed_packed_row_count", 0)
                predicate = "optimizer_step_consumed_packed_long_context_row"
            else:
                observed_length = observed_input_length
                observed_count = scheduler.get("observed_sequence_count", 0)
                predicate = "optimizer_step_consumed_declared_long_context_lane"
            runtime_tokens = self._data["observations"]["runtime"].get(
                "backend_tokens", {}
            )
            # Runtime execution and backend capacity are distinct claims.  A
            # natural 256K workload can exercise genuinely long sequences
            # without containing a document close to the configured ceiling.
            # Require a near-128K real sample to establish long-context
            # execution, while preserving the exact configured-capacity and
            # observed-coverage facts below.  This avoids padding or synthetic
            # repetition merely to turn Evidence green.
            context_coverage = (
                observed_length / effective_length if effective_length else 0.0
            )
            long_context_threshold = min(effective_length, 131072)
            long_context_threshold_coverage = (
                observed_length / long_context_threshold
                if long_context_threshold
                else 0.0
            )
            if (
                requested_length >= 131072
                and contract_validated
                and runtime_tokens.get("state") == (
                    "not_applicable_offline"
                    if self._plan.policy.backends["target_features"] == "offline_replay"
                    else "validated"
                )
                and observed_length <= effective_length
                and long_context_threshold_coverage >= 0.99
            ):
                cp_size = backend_tokens["training_sharding"][
                    "context_parallel_size"
                ]
                shard_base, shard_extra = divmod(observed_length, cp_size)
                observed_shards = [
                    shard_base + (1 if rank < shard_extra else 0)
                    for rank in range(cp_size)
                ]
                self._activate(
                    "long_context_execution",
                    {
                        "predicate": predicate,
                        "backend_context_state": runtime_tokens.get("state"),
                        "requested_max_sequence_length": requested_length,
                        "effective_training_max_sequence_length": effective_length,
                        "target_input_reserve_tokens": reserve_tokens,
                        "backend_context_length": backend_tokens[
                            "backend_context_length"
                        ],
                        "model_native_context_limit": backend_tokens[
                            "model_native_context_limit"
                        ],
                        "model_context_extension": backend_tokens[
                            "model_context_extension"
                        ],
                        "admitted_training_context_limit": backend_tokens[
                            "admitted_training_context_limit"
                        ],
                        "context_override": backend_tokens["context_override"],
                        "observed_sequence_max": observed_length,
                        "observed_sequence_count": observed_count,
                        "observed_input_sequence_max": observed_input_length,
                        "observed_context_coverage": context_coverage,
                        "long_context_execution_threshold": long_context_threshold,
                        "observed_long_context_threshold_coverage": (
                            long_context_threshold_coverage
                        ),
                        "observed_training_shard_lengths": observed_shards,
                        "observed_sequence_padding_tokens": 0,
                    },
                )
            self._activate(
                "data_lifecycle",
                {
                    "predicate": "all_generated_outputs_reached_observable_terminal_state",
                    "produced_outputs": lifecycle["produced_outputs"],
                    "discarded_outputs": lifecycle["discarded_outputs"],
                    "resident_outputs": lifecycle["resident_outputs"],
                    "resident_bytes": lifecycle["resident_bytes"],
                },
            )

        manager = cleanup.get("inference_manager") or {}
        host_capacity = self._plan.policy.capacity["host_orchestrator"]
        max_pool_size = host_capacity["max_sample_pool_size"]
        planned_inflight_limit = host_capacity["inflight_sample_limit"]
        observed_inflight_limit = manager.get("inflight_sample_limit")
        peak_inflight = manager.get("peak_inflight_samples")
        peak_admitted_ready = manager.get("peak_admitted_ready_samples")
        peak_reserved = manager.get("peak_reserved_samples")
        capacity_facts = (
            observed_inflight_limit, peak_inflight, peak_admitted_ready, peak_reserved
        )
        global_capacity_validated = (
            max_pool_size > 0
            and all(type(value) is int and value >= 0 for value in capacity_facts)
            and scheduler_validated
            and manager.get("inflight_samples") == 0
            and 0 < observed_inflight_limit <= planned_inflight_limit
            and peak_inflight <= observed_inflight_limit
            and peak_admitted_ready < max_pool_size
            and peak_reserved <= max_pool_size - 1 + observed_inflight_limit
        )
        node_capacity_observation = None
        if host_capacity.get("ready_watermark_scope") == "per_locality_domain":
            planned_by_node = {
                str(domain["node_address"]): int(domain["ready_watermark"])
                for domain in self._plan.policy.resources.get("locality_domains", ())
            }
            observed_watermarks = manager.get("ready_watermark_by_node")
            inflight_by_node = manager.get("inflight_samples_by_node")
            inflight_limits_by_node = manager.get("inflight_sample_limit_by_node")
            peak_inflight_by_node = manager.get("peak_inflight_samples_by_node")
            peak_admitted_by_node = manager.get("peak_admitted_ready_by_node")
            peak_reserved_by_node = manager.get("peak_reserved_by_node")
            node_maps = (
                observed_watermarks,
                inflight_by_node,
                inflight_limits_by_node,
                peak_inflight_by_node,
                peak_admitted_by_node,
                peak_reserved_by_node,
            )
            capacity_validated = (
                max_pool_size > 0
                and scheduler_validated
                and all(isinstance(value, Mapping) for value in node_maps)
                and dict(observed_watermarks or {}) == planned_by_node
                and set().union(*(set(value or {}) for value in node_maps))
                == set(planned_by_node)
                and all(
                    type((inflight_by_node or {}).get(node)) is int
                    and (inflight_by_node or {})[node] == 0
                    and type((inflight_limits_by_node or {}).get(node)) is int
                    and (inflight_limits_by_node or {})[node] > 0
                    and type((peak_inflight_by_node or {}).get(node)) is int
                    and 0 <= (peak_inflight_by_node or {})[node]
                    <= (inflight_limits_by_node or {})[node]
                    and type((peak_admitted_by_node or {}).get(node)) is int
                    and 0 <= (peak_admitted_by_node or {})[node] < watermark
                    and type((peak_reserved_by_node or {}).get(node)) is int
                    and (peak_reserved_by_node or {})[node]
                    <= watermark - 1 + (inflight_limits_by_node or {})[node]
                    for node, watermark in planned_by_node.items()
                )
            )
            node_capacity_observation = {
                "ready_watermark_by_node": dict(observed_watermarks or {}),
                "inflight_sample_limit_by_node": dict(inflight_limits_by_node or {}),
                "peak_inflight_samples_by_node": dict(peak_inflight_by_node or {}),
                "peak_admitted_ready_by_node": dict(peak_admitted_by_node or {}),
                "peak_reserved_by_node": dict(peak_reserved_by_node or {}),
            }
        else:
            capacity_validated = global_capacity_validated
        if capacity_validated:
            self._activate(
                "capacity_control",
                {
                    "predicate": (
                        "per_locality_ready_credits_gate_admission_with_bounded_inflight"
                        if node_capacity_observation is not None
                        else "ready_watermark_gates_admission_with_bounded_inflight"
                    ),
                    "max_sample_pool_size": max_pool_size,
                    "planned_inflight_sample_limit": planned_inflight_limit,
                    "observed_inflight_sample_limit": observed_inflight_limit,
                    "peak_inflight_samples": peak_inflight,
                    "peak_admitted_ready_samples": peak_admitted_ready,
                    "queued_plus_inflight_sample_limit": (
                        max_pool_size - 1 + observed_inflight_limit
                    ),
                    "peak_reserved_samples": peak_reserved,
                    "resident_capacity": "tracked_separately_by_transfer_storage",
                    "peak_resident_outputs": lifecycle["peak_resident_outputs"],
                    "peak_resident_bytes": lifecycle["peak_resident_bytes"],
                    **(node_capacity_observation or {}),
                },
            )
        performance_policy = self._plan.policy.measurement["performance"]
        performance = training.get("performance") or {}
        expected_window_steps = (
            training.get("optimizer_steps_observed", 0)
            - performance_policy["warmup_steps"]
            - performance_policy["drain_steps"]
        )
        performance_validated = (
            performance_policy["required"]
            and performance.get("state") == "measured"
            and performance.get("optimizer_steps") == expected_window_steps
            and performance.get("committed_work", 0) > 0
            and performance.get("window_wall_seconds", 0) > 0
            and performance.get("committed_work_per_second", 0) > 0
            and math.isfinite(performance.get("window_wall_seconds", math.nan))
            and math.isfinite(
                performance.get("committed_work_per_second", math.nan)
            )
        )
        if performance_validated:
            self._activate(
                "performance_measurement",
                {
                    "predicate": "continuous_optimizer_window_committed_finite_work",
                    "warmup_steps": performance_policy["warmup_steps"],
                    "drain_steps": performance_policy["drain_steps"],
                    "optimizer_steps": performance["optimizer_steps"],
                    "committed_work": performance["committed_work"],
                    "window_wall_seconds": performance["window_wall_seconds"],
                    "committed_work_per_second": performance[
                        "committed_work_per_second"
                    ],
                },
            )
        final_metrics = training.get("final_metrics") or {}
        avg_loss = final_metrics.get("train/avg_loss")
        grad_norm = final_metrics.get("train/grad_norm")
        if (
            performance_policy["required"]
            and isinstance(avg_loss, (int, float))
            and isinstance(grad_norm, (int, float))
            and math.isfinite(avg_loss)
            and math.isfinite(grad_norm)
        ):
            self._activate(
                "training_numerics",
                {
                    "predicate": "final_training_loss_and_gradient_norm_are_finite",
                    "train/avg_loss": avg_loss,
                    "train/grad_norm": grad_norm,
                },
            )
        self._write()

    def complete(self) -> None:
        pending = [
            item["name"]
            for item in self._data["capabilities"]
            if item["requirement"] == "required" and item["effective_state"] != "active"
        ]
        cleanup = self._data["observations"]["cleanup"] or {}
        training = self._data["observations"]["training"] or {}
        checkpoint = training.get("checkpoint", {})
        checkpoint_ok = not checkpoint.get("enabled") or checkpoint.get("exists")
        artifacts = self._data["observations"]["artifacts"]
        artifacts_ok = artifacts["config"]["exists"] and artifacts["plan"]["exists"]
        healthy = (
            not pending
            and training.get("optimizer_steps_observed", 0) > 0
            and cleanup.get("state") == "completed"
            and checkpoint_ok
            and artifacts_ok
        )
        self._data["state"] = "completed"
        self._data["health"] = "healthy" if healthy else "incomplete_evidence"
        self._data["required_capabilities_pending"] = pending
        self._finish()

    def fail(self, error: BaseException) -> None:
        self._data["state"] = (
            "cancelled"
            if isinstance(error, KeyboardInterrupt)
            or is_termination_exception(error)
            else "failed"
        )
        self._data["health"] = "failed"
        self._data["error"] = {
            "type": type(error).__name__,
            "message": str(error)[:2000],
        }
        from aspec.inference.diagnostics import backend_error_logs

        backend_logs = backend_error_logs(self.path.parent)
        if backend_logs:
            self._data["error"]["backend_logs"] = backend_logs
        self._finish()

    def _finish(self) -> None:
        self._data["finished_at"] = _utc_now()
        self._data["duration_seconds"] = round(time.monotonic() - self._started_monotonic, 6)
        self._write()
