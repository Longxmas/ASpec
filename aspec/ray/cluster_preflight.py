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

"""Fail-closed inventory and readiness checks for static multi-node runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from aspec.config.mooncake_config import MooncakeConfig, resolve_rdma_routes
from aspec.ray.owner_fence import SourceTeardownBlockedError, wait_for_owner_exit
from aspec.runtime_identity import (
    SGLANG_RUNTIME_RELATIVE_PATHS,
    module_runtime_identity,
)
from aspec.utils.env import get_aspec_env_vars
from aspec.utils.logging import logger

CLUSTER_PREFLIGHT_SCHEMA_VERSION = "aspec.cluster-preflight/v3"


class ClusterPreflightError(RuntimeError):
    """The live cluster does not satisfy the compiled static membership."""


def _source_tree_sha256() -> str:
    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for source_file in sorted(package_root.rglob("*.py")):
        relative = source_file.relative_to(package_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        payload = source_file.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in (
        "torch",
        "ray",
        "sglang",
        "mooncake-transfer-engine",
        "flash-attn-4",
        "nvidia-cutlass-dsl",
        "quack-kernels",
        "transformers",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _gpu_inventory() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ClusterPreflightError(f"cannot execute nvidia-smi: {exc}") from exc
    if result.returncode != 0:
        raise ClusterPreflightError(
            f"nvidia-smi inventory failed: {result.stderr.strip() or result.returncode}"
        )

    gpus: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            raise ClusterPreflightError(f"unexpected nvidia-smi row: {line!r}")
        gpus.append(
            {
                "index": int(fields[0]),
                "reported_name": fields[1],
                "uuid": fields[2],
                "memory_mib": int(fields[3]),
                "driver_version": fields[4],
                "compute_capability": fields[5],
            }
        )
    if not gpus:
        raise ClusterPreflightError("nvidia-smi reported no GPUs")
    return gpus


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _rdma_inventory() -> list[dict[str, Any]]:
    root = Path("/sys/class/infiniband")
    if not root.is_dir():
        return []
    devices: list[dict[str, Any]] = []
    for device_path in sorted(root.iterdir()):
        verbs_names: list[str] = []
        verbs_root = device_path / "device" / "infiniband_verbs"
        if verbs_root.is_dir():
            verbs_names = sorted(entry.name for entry in verbs_root.iterdir())
        userspace_paths = [Path("/dev/infiniband") / name for name in verbs_names]
        ports: list[dict[str, Any]] = []
        ports_root = device_path / "ports"
        if ports_root.is_dir():
            for port_path in sorted(ports_root.iterdir(), key=lambda item: int(item.name)):
                netdevs: set[str] = set()
                ndev_root = port_path / "gid_attrs" / "ndevs"
                if ndev_root.is_dir():
                    for entry in ndev_root.iterdir():
                        value = _read_text(entry)
                        if value:
                            netdevs.add(value)
                gids: list[dict[str, Any]] = []
                gids_root = port_path / "gids"
                if gids_root.is_dir():
                    for gid_path in sorted(
                        gids_root.iterdir(), key=lambda item: int(item.name)
                    ):
                        index = int(gid_path.name)
                        gid_value = _read_text(gid_path)
                        # Empty table entries dominate the inventory on RoCE
                        # hosts and do not describe a selectable route.
                        if not gid_value or set(gid_value) <= {"0", ":"}:
                            continue
                        gids.append(
                            {
                                "index": index,
                                "gid": gid_value,
                                "type": _read_text(
                                    port_path / "gid_attrs" / "types" / str(index)
                                ),
                                "netdev": _read_text(
                                    port_path / "gid_attrs" / "ndevs" / str(index)
                                ),
                            }
                        )
                state = _read_text(port_path / "state")
                ports.append(
                    {
                        "port": int(port_path.name),
                        "state": state,
                        "physical_state": _read_text(port_path / "phys_state"),
                        "rate": _read_text(port_path / "rate"),
                        "lid": _read_text(port_path / "lid"),
                        "gids": gids,
                        "netdevs": sorted(netdevs),
                        "netdev_facts": {
                            netdev: {
                                "operstate": _read_text(
                                    Path("/sys/class/net") / netdev / "operstate"
                                ),
                                "mtu": _read_text(
                                    Path("/sys/class/net") / netdev / "mtu"
                                ),
                                "speed_mbps": _read_text(
                                    Path("/sys/class/net") / netdev / "speed"
                                ),
                                "address": _read_text(
                                    Path("/sys/class/net") / netdev / "address"
                                ),
                            }
                            for netdev in sorted(netdevs)
                        },
                        "active": bool(state and "ACTIVE" in state.upper()),
                    }
                )
        try:
            pci_device = device_path.joinpath("device").resolve(strict=True).name
        except OSError:
            pci_device = None
        devices.append(
            {
                "name": device_path.name,
                "node_guid": _read_text(device_path / "node_guid"),
                "pci_device": pci_device,
                "numa_node": _read_text(device_path / "device" / "numa_node"),
                "ports": ports,
                "active": any(port["active"] for port in ports),
                "verbs_devices": verbs_names,
                "userspace_paths": [str(path) for path in userspace_paths],
                "userspace_access": bool(userspace_paths)
                and all(
                    path.exists() and os.access(path, os.R_OK | os.W_OK)
                    for path in userspace_paths
                ),
            }
        )
    return devices


def _path_inventory(paths: Sequence[str]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        try:
            stat = path.stat()
            filesystem = _filesystem_capacity(path)
        except OSError as exc:
            inventory.append(
                {"path": str(path), "state": "unavailable", "error": str(exc)}
            )
            continue
        kind = "directory" if path.is_dir() else "file"
        entry = {
            "path": str(path),
            "state": "available",
            "kind": kind,
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "size": stat.st_size,
            "filesystem": filesystem,
        }
        # Hash ordinary configuration/data files when doing so is bounded.
        # Large model/data artifacts stay identified by their shared-root
        # contract and manifest shape rather than adding unbounded startup I/O.
        if kind == "file" and stat.st_size <= 64 * 1024 * 1024:
            try:
                entry["content_sha256"] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
            except OSError as exc:
                entry["state"] = "unavailable"
                entry["error"] = str(exc)
        inventory.append(entry)
    return inventory


def _filesystem_capacity(path: Path) -> dict[str, int]:
    """Return byte capacity without inventing a workload-specific threshold."""

    stat = os.statvfs(path)
    block_size = stat.f_frsize
    return {
        "capacity_bytes": block_size * stat.f_blocks,
        "free_bytes": block_size * stat.f_bfree,
        "available_bytes": block_size * stat.f_bavail,
    }


def _collect_local_inventory(
    required_paths: Sequence[str], marker_path: str, marker_sha256: str,
    storage_mode: str = "shared", output_dir: str | None = None,
    target_address: str | None = None,
    roles: Sequence[str] | None = None,
) -> dict[str, Any]:
    marker = Path(marker_path)
    if storage_mode == "node_local":
        # This tests local writability only; it does not claim cross-node visibility.
        marker = Path(output_dir) / f".local-probe-{uuid.uuid4().hex}"
        marker.write_text(marker_sha256)
        marker_observed_sha256 = marker.read_text()
        marker.unlink()
    else:
        try:
            marker_observed_sha256 = hashlib.sha256(marker.read_bytes()).hexdigest()
        except OSError as exc:
            raise ClusterPreflightError(f"shared marker is not readable at {marker}: {exc}") from exc
    paths = _path_inventory(required_paths)
    if storage_mode == "node_local":
        from aspec.ray.node_local_files import file_digest

        for item in paths:
            path = Path(item["path"])
            if item["state"] != "available" or str(path) == output_dir:
                continue
            files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
            identity = [(p.relative_to(path).as_posix() if path.is_dir() else p.name,
                         p.stat().st_size, file_digest(p)) for p in files]
            item["content_sha256"] = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()
    memlock_soft, memlock_hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    target_runtime = None
    if target_address:
        from aspec.inference.service_protocol import request
        from aspec.inference.target_runtime import release_version

        info = request(target_address, "/v1/info", timeout=30)
        target_runtime = info["runtime"]
        if info["version"] != release_version():
            raise ClusterPreflightError("Target and Draft ASpec release versions differ")
        if info["node_ip"] != ray.util.get_node_ip_address():
            raise ClusterPreflightError("Target service is not on the planned node")
    return {
        "node_id": ray.get_runtime_context().get_node_id(),
        "node_ip": ray.util.get_node_ip_address(),
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "source_tree_sha256": _source_tree_sha256(),
        "packages": _package_versions(),
        "roles": list(roles) if roles is not None else None,
        "target_runtime": target_runtime,
        "runtime_artifacts": {
            "sglang": target_runtime["sglang"] if target_runtime else module_runtime_identity(
                "sglang", relative_paths=SGLANG_RUNTIME_RELATIVE_PATHS
            ),
            "mooncake": module_runtime_identity("mooncake.engine"),
        },
        "gpus": _gpu_inventory(),
        "rdma_devices": _rdma_inventory(),
        "memlock_limit_bytes": {
            "soft": "unlimited" if memlock_soft == resource.RLIM_INFINITY else memlock_soft,
            "hard": "unlimited" if memlock_hard == resource.RLIM_INFINITY else memlock_hard,
        },
        "paths": paths,
        "storage_mode": storage_mode,
        ("local_writable_probe" if storage_mode == "node_local" else "shared_marker"): {
            "path": str(marker),
            "expected_sha256": marker_sha256,
            "observed_sha256": marker_observed_sha256,
            "matched": marker_observed_sha256 == marker_sha256,
        },
    }


def _gpu_fingerprint(node: Mapping[str, Any]) -> tuple[Any, ...]:
    gpus = list(node.get("gpus") or ())
    return (
        len(gpus),
        tuple(
            sorted(
                (
                    gpu.get("reported_name"),
                    gpu.get("memory_mib"),
                    gpu.get("driver_version"),
                    gpu.get("compute_capability"),
                )
                for gpu in gpus
            )
        ),
    )


def validate_cluster_inventories(
    inventories: Sequence[Mapping[str, Any]],
    *,
    expected_gpu_counts: Mapping[str, int],
    required_rdma_routes: Mapping[str, Mapping[str, Any]] | None = None,
    required_memlock_bytes: int = 0,
    require_homogeneous_gpus: bool,
) -> None:
    expected_ips = set(expected_gpu_counts)
    observed_ips = {str(node.get("node_ip")) for node in inventories}
    if observed_ips != expected_ips:
        raise ClusterPreflightError(
            "inventory membership does not match the Plan: "
            f"expected={sorted(expected_ips)}, observed={sorted(observed_ips)}"
        )

    if len({node.get("storage_mode", "shared") for node in inventories}) != 1:
        raise ClusterPreflightError("Storage modes differ across nodes")
    source_hashes = {node.get("source_tree_sha256") for node in inventories}
    if len(source_hashes) != 1 or None in source_hashes:
        raise ClusterPreflightError("ASpec source identity differs across nodes")
    package_sets = defaultdict(set)
    for node in inventories:
        roles = node.get("roles")
        if roles is not None and (not roles or set(roles) - {"draft", "target"}):
            raise ClusterPreflightError("Unknown or empty node runtime roles")
        group = "legacy" if roles is None else "draft" if "draft" in roles else "target"
        package_sets[group].add(json.dumps(node.get("packages"), sort_keys=True))
        if roles and "target" in roles and not node.get("target_runtime"):
            raise ClusterPreflightError("Target node did not report its isolated runtime")
    if any(len(versions) != 1 for versions in package_sets.values()):
        raise ClusterPreflightError("required package versions differ across nodes")
    if len({(node.get("packages") or {}).get("ray") for node in inventories}) != 1:
        raise ClusterPreflightError("Ray versions differ across runtime roles")
    target_sets = {json.dumps({key: value for key, value in node["target_runtime"].items()
                              if key not in {"python", "prefix", "sglang"}}, sort_keys=True)
                   for node in inventories if node.get("target_runtime")}
    if len(target_sets) > 1:
        raise ClusterPreflightError("Target runtime identities differ across nodes")
    runtime_artifact_sets = defaultdict(set)
    for node in inventories:
        artifacts = node.get("runtime_artifacts") or {}
        names = ("sglang", "mooncake") if node.get("roles") is None or "target" in node["roles"] else ("mooncake",)
        incomplete = {
            name: identity
            for name in names
            if not isinstance((identity := artifacts.get(name)), Mapping)
            or not identity.get("effective_tree_sha256")
            or identity.get("missing_files")
        }
        if incomplete:
            raise ClusterPreflightError(
                "required runtime artifact identity is incomplete on node "
                f"{node.get('node_ip')}: {incomplete}"
            )
        for name in names:
            identity = artifacts[name]
            runtime_artifact_sets[name].add((identity["effective_tree_sha256"], identity.get("file_count")))
    if any(len(identities) != 1 for identities in runtime_artifact_sets.values()):
        raise ClusterPreflightError(
            "effective SGLang/Mooncake runtime artifacts differ across nodes"
        )

    if require_homogeneous_gpus:
        fingerprints = {_gpu_fingerprint(node) for node in inventories}
        if len(fingerprints) != 1:
            raise ClusterPreflightError("GPU inventory is not homogeneous across nodes")

    path_identities: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    path_sets: set[tuple[str, ...]] = set()
    selected_gid_types: set[str] = set()
    routes = dict(required_rdma_routes or {})
    if routes and set(routes) != expected_ips:
        raise ClusterPreflightError(
            "RDMA route membership does not match the Plan: "
            f"expected={sorted(expected_ips)}, routes={sorted(routes)}"
        )
    for node in inventories:
        node_ip = str(node["node_ip"])
        gpus = list(node.get("gpus") or ())
        required_gpus = int(expected_gpu_counts[node_ip])
        if len(gpus) < required_gpus:
            raise ClusterPreflightError(
                f"node {node_ip} has {len(gpus)} GPUs but Plan requires {required_gpus}"
            )
        marker_key = "local_writable_probe" if node.get("storage_mode", "shared") == "node_local" else "shared_marker"
        if not (node.get(marker_key) or {}).get("matched"):
            raise ClusterPreflightError(f"shared marker mismatch on node {node_ip}")

        rdma = {device["name"]: device for device in node.get("rdma_devices") or ()}
        route = routes.get(node_ip)
        if route is not None:
            device_name = str(route["device_name"])
            required_rdma_port = int(route["port"])
            required_rdma_gid_index = int(route["gid_index"])
            device = rdma.get(device_name)
            if device is None or not device.get("active"):
                raise ClusterPreflightError(
                    f"required RDMA device {device_name!r} is not active on node {node_ip}"
                )
            if not device.get("userspace_access"):
                raise ClusterPreflightError(
                    f"required RDMA device {device_name!r} has no usable verbs device "
                    f"on node {node_ip}: {device.get('userspace_paths') or []}"
                )
            port = next(
                (
                    item
                    for item in device.get("ports") or ()
                    if int(item.get("port", -1)) == required_rdma_port
                ),
                None,
            )
            if port is None or not port.get("active"):
                raise ClusterPreflightError(
                    f"required RDMA port {device_name}:{required_rdma_port} is not "
                    f"active on node {node_ip}"
                )
            gid = next(
                (
                    item
                    for item in port.get("gids") or ()
                    if int(item.get("index", -1)) == required_rdma_gid_index
                ),
                None,
            )
            if gid is None or not gid.get("gid") or set(str(gid["gid"])) <= {"0", ":"}:
                raise ClusterPreflightError(
                    f"required RDMA GID {device_name}:{required_rdma_port}/"
                    f"{required_rdma_gid_index} is unavailable on node {node_ip}"
                )
            gid_type = str(gid.get("type") or "")
            if not gid_type:
                raise ClusterPreflightError(
                    f"required RDMA GID type is unavailable on node {node_ip}"
                )
            selected_gid_types.add(gid_type)
            netdev = str(gid.get("netdev") or "")
            netdev_facts = (port.get("netdev_facts") or {}).get(netdev) or {}
            if not netdev or str(netdev_facts.get("operstate")).lower() != "up":
                raise ClusterPreflightError(
                    f"required RDMA GID netdev is not up on node {node_ip}: "
                    f"netdev={netdev!r}, facts={netdev_facts}"
                )
            try:
                _ = int(netdev_facts["mtu"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ClusterPreflightError(
                    f"required RDMA netdev MTU is unavailable on node {node_ip}"
                ) from exc
            # MTU is recorded per node for performance cohort analysis.  A
            # lower MTU is not itself illegal when the real pairwise transfer
            # probe succeeds, so it is not required to match cluster-wide.

        if route is not None:
            memlock = node.get("memlock_limit_bytes") or {}
            soft_limit = memlock.get("soft")
            if soft_limit != "unlimited" and (
                not isinstance(soft_limit, int)
                or soft_limit < required_memlock_bytes
            ):
                raise ClusterPreflightError(
                    f"RDMA memlock soft limit is below the planned registered-memory "
                    f"requirement on node {node_ip}: required={required_memlock_bytes}, "
                    f"observed={soft_limit!r}"
                )

        node_paths = list(node.get("paths") or ())
        path_sets.add(tuple(sorted(str(path.get("path")) for path in node_paths)))
        for path in node_paths:
            if path.get("state") != "available":
                raise ClusterPreflightError(
                    f"required path unavailable on node {node_ip}: "
                    f"{path.get('path')} ({path.get('error')})"
                )
            # A marker proves cross-node visibility for the writable root.
            # Client-local inode/device numbers are not a portable shared-FS
            # identity.  For files, shape plus the optional content digest is
            # still useful without rejecting valid NFS/FUSE mounts.
            path_identities[str(path["path"])].add(
                (
                    path.get("kind"),
                    path.get("size") if path.get("kind") == "file" else None,
                    path.get("content_sha256"),
                )
            )

    if len(path_sets) != 1:
        raise ClusterPreflightError("required path inventory differs across nodes")
    divergent_paths = sorted(
        path for path, identities in path_identities.items() if len(identities) != 1
    )
    if divergent_paths:
        raise ClusterPreflightError(
            "configured input files do not have consistent identities "
            f"on every node: {divergent_paths}"
        )
    if len(selected_gid_types) > 1:
        raise ClusterPreflightError(
            "selected RDMA GID type differs across nodes: "
            f"{sorted(selected_gid_types)}"
        )


def _required_paths(args) -> list[str]:
    values = [
        getattr(args, "cluster_shared_root", None) if getattr(args, "cluster_storage_mode", "shared") == "shared" else None,
        getattr(args, "output_dir", None),
        getattr(args, "target_model_path", None),
        getattr(args, "draft_model_config", None),
        getattr(args, "initial_draft_model_path", None),
        getattr(args, "train_data_path", None),
        getattr(args, "eval_data_path", None),
        getattr(args, "load_path", None),
    ]
    return sorted({str(value) for value in values if value})


def _expected_gpu_counts(args) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for node_ip in getattr(args, "training_node_ips", None) or ():
        counts[str(node_ip)] += int(args.training_num_gpus_per_node)

    remaining = int(args.inference_num_gpus)
    per_node = int(args.inference_num_gpus_per_node)
    for node_ip in getattr(args, "inference_node_ips", None) or ():
        allocated = min(per_node, remaining)
        counts[str(node_ip)] += allocated
        remaining -= allocated
    if remaining:
        raise ClusterPreflightError(
            f"inference node list leaves {remaining} planned GPU(s) unassigned"
        )
    return dict(counts)


def _validate_and_resolve_locality_domains(
    args,
    inventories: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Prove the compiled rank geometry against live Ray node identities."""

    domains = [dict(item) for item in getattr(args, "aspec_locality_domains", ())]
    if not domains:
        raise ClusterPreflightError(
            "static multi-node Plan does not contain locality-domain geometry"
        )
    inventory_by_ip = {str(item["node_ip"]): item for item in inventories}
    training_expected = {
        str(node): int(args.training_num_gpus_per_node)
        for node in getattr(args, "training_node_ips", ())
    }
    remaining_inference = int(args.inference_num_gpus)
    inference_expected: dict[str, int] = {}
    for node in getattr(args, "inference_node_ips", ()):
        count = min(int(args.inference_num_gpus_per_node), remaining_inference)
        inference_expected[str(node)] = count
        remaining_inference -= count

    addresses = [str(domain.get("node_address", "")) for domain in domains]
    if len(addresses) != len(set(addresses)) or set(addresses) != set(inventory_by_ip):
        raise ClusterPreflightError(
            "locality-domain membership does not match live cluster membership: "
            f"planned={addresses}, observed={sorted(inventory_by_ip)}"
        )

    all_training_ranks: list[int] = []
    all_dp_ranks: list[int] = []
    all_engine_ranks: list[int] = []
    resolved: list[dict[str, Any]] = []
    for domain in domains:
        node = str(domain["node_address"])
        training_ranks = [int(value) for value in domain.get("training_world_ranks", ())]
        dp_ranks = [int(value) for value in domain.get("draft_dp_ranks", ())]
        engine_ranks = [int(value) for value in domain.get("target_engine_ranks", ())]
        parallel_group_size = int(domain.get("draft_parallel_group_size", 0))
        target_tp = int(domain.get("target_tensor_parallel_size", 0))
        if len(training_ranks) != training_expected.get(node, 0):
            raise ClusterPreflightError(
                f"locality domain {node} has invalid trainer-rank geometry"
            )
        if len(dp_ranks) * parallel_group_size != training_expected.get(node, 0):
            raise ClusterPreflightError(
                f"locality domain {node} has invalid Draft parallel geometry"
            )
        if len(engine_ranks) * target_tp != inference_expected.get(node, 0):
            raise ClusterPreflightError(
                f"locality domain {node} has invalid Target TP geometry"
            )
        all_training_ranks.extend(training_ranks)
        all_dp_ranks.extend(dp_ranks)
        all_engine_ranks.extend(engine_ranks)
        inventory = inventory_by_ip[node]
        resolved.append(
            {
                **domain,
                "ray_node_id": str(inventory["node_id"]),
                "hostname": str(inventory["hostname"]),
                "gpu_uuids": [
                    str(gpu["uuid"]) for gpu in inventory.get("gpus", ())
                ],
            }
        )

    rank_sets = {
        "training_world_ranks": all_training_ranks,
        "draft_dp_ranks": all_dp_ranks,
        "target_engine_ranks": all_engine_ranks,
    }
    for name, values in rank_sets.items():
        if sorted(values) != list(range(len(values))):
            raise ClusterPreflightError(
                f"locality domains do not define contiguous unique {name}: {values}"
            )
    return resolved


def _required_rdma_memlock_bytes(args) -> int:
    """Bound the largest registered CUDA ring owned by one endpoint process."""

    config = MooncakeConfig.from_flat_args(args)
    slot_bytes = int(config.gpu_buffer_size or 0)
    largest_ring = max(
        config.ring_slots_for_role("source"),
        config.ring_slots_for_role("destination"),
    )
    registered_bytes = slot_bytes * largest_ring
    return int(registered_bytes * 1.10) + 64 * 1024 * 1024


def _selected_rdma_routes(
    inventories: Sequence[Mapping[str, Any]],
    *,
    configured_routes: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    for node in inventories:
        node_ip = str(node["node_ip"])
        configured = configured_routes[node_ip]
        device_name = str(configured["device_name"])
        port_number = int(configured["port"])
        gid_index = int(configured["gid_index"])
        device = next(
            item
            for item in node.get("rdma_devices") or ()
            if item.get("name") == device_name
        )
        port = next(
            item
            for item in device.get("ports") or ()
            if int(item.get("port", -1)) == port_number
        )
        gid = next(
            item
            for item in port.get("gids") or ()
            if int(item.get("index", -1)) == gid_index
        )
        netdev = str(gid.get("netdev") or "")
        routes.append(
            {
                "node_ip": node_ip,
                "device_name": device_name,
                "pci_device": device.get("pci_device"),
                "numa_node": device.get("numa_node"),
                "port": port_number,
                "port_state": port.get("state"),
                "port_rate": port.get("rate"),
                "gid_index": gid_index,
                "gid": gid.get("gid"),
                "gid_type": gid.get("type"),
                "netdev": netdev,
                "netdev_facts": (port.get("netdev_facts") or {}).get(netdev),
            }
        )
    return sorted(routes, key=lambda item: item["node_ip"])


class _GpuDirectProbeActor:
    """One-GPU endpoint for fail-closed RDMA or node-local NVLink probes."""

    def __init__(
        self,
        *,
        role: str,
        protocol: str,
        device_name: str,
        rdma_port: int,
        rdma_gid_index: int,
        run_id: str,
        attempt_id: str,
        timeout_seconds: float,
    ) -> None:
        import torch

        from aspec.config.mooncake_config import MooncakeConfig
        from aspec.transfer.mooncake.p2p import MooncakeP2PTransfer

        os.environ["ASPEC_RUN_ID"] = run_id
        os.environ["ASPEC_ATTEMPT_ID"] = attempt_id
        visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
        assigned_ids = ray.get_gpu_ids()
        device_index = 0 if visible_devices else int(assigned_ids[0])
        self.device = torch.device("cuda", device_index)
        torch.cuda.set_device(self.device)
        self.role = role
        self.config = MooncakeConfig(
            local_hostname=ray.util.get_node_ip_address(),
            protocol=protocol,
            device_name=device_name,
            rdma_port=rdma_port,
            rdma_gid_index=rdma_gid_index,
            data_plane="transfer_engine",
            gpu_buffer_size=1024 * 1024,
            gpu_ring_slots=1,
            enable_gpu_direct=True,
            allow_host_fallback=False,
            require_nvlink=protocol == "nvlink_intra",
            get_retry_max_wait_seconds=timeout_seconds,
        )
        self.config.export_env()
        self.transfer = MooncakeP2PTransfer(self.config)
        self.transfer.setup(self.device, role=role)

    def stage(self, *, run_id: str, attempt_id: str, pair_id: str) -> str:
        import torch

        if self.role != "source":
            raise RuntimeError("only the RDMA probe source can stage a payload")
        hidden = torch.arange(
            4096, dtype=torch.float32, device=self.device
        ).reshape(128, 32)
        tokens = torch.arange(128, dtype=torch.int64, device=self.device)
        descriptor = self.transfer.stage(
            store_key=f"cluster-preflight-{pair_id}",
            tensors=(("hidden_states", hidden), ("input_ids", tokens)),
            run_id=run_id,
            attempt_id=attempt_id,
            data_id=pair_id,
            request_id=pair_id,
            source_engine="cluster-preflight",
        )
        return descriptor.to_key()

    def read_and_validate(self, descriptor_key: str) -> dict[str, Any]:
        import torch

        from aspec.transfer.mooncake.p2p import MooncakeP2PDescriptor

        if self.role != "destination":
            raise RuntimeError("only the RDMA probe destination can read a payload")
        descriptor = MooncakeP2PDescriptor.from_key(descriptor_key)
        started = time.monotonic()
        result = self.transfer.read(descriptor)
        expected_hidden = torch.arange(
            4096, dtype=torch.float32, device=self.device
        ).reshape(128, 32)
        expected_tokens = torch.arange(128, dtype=torch.int64, device=self.device)
        torch.cuda.synchronize(self.device)
        matched = bool(
            torch.equal(result.tensors["hidden_states"], expected_hidden)
            and torch.equal(result.tensors["input_ids"], expected_tokens)
        )
        result.lease.release()
        return {
            "matched": matched,
            "elapsed_seconds": time.monotonic() - started,
            "payload_bytes": descriptor.payload_bytes,
            "source_endpoint": descriptor.source_endpoint,
            "destination_ip": ray.util.get_node_ip_address(),
        }

    def wait_for_release(self, timeout_seconds: float) -> bool:
        return self.transfer.wait_for_source_releases(timeout=timeout_seconds)

    def status(self) -> dict[str, Any]:
        status = self.transfer.status()
        status["probe_placement"] = {
            "ray_node_id": str(ray.get_runtime_context().get_node_id()),
            "node_ip": ray.util.get_node_ip_address(),
            "assigned_gpu_ids": [int(float(value)) for value in ray.get_gpu_ids()],
        }
        return status

    def get_owner_status(self) -> dict[str, Any]:
        return {
            "node_id": str(ray.get_runtime_context().get_node_id()),
            "node_ip": ray.util.get_node_ip_address(),
            "pid": os.getpid(),
        }

    def close(self) -> None:
        self.transfer.close()



def _terminate_probe_pair(source, destination, reader_owner, *, timeout):
    """Stop native readers before invalidating their registered source memory."""
    if destination is not None:
        try:
            ray.kill(destination, no_restart=True)
            if reader_owner is not None:
                wait_for_owner_exit([reader_owner], timeout=timeout)
        except BaseException as exc:
            if reader_owner is not None:
                raise SourceTeardownBlockedError([reader_owner]) from exc
            raise
    # An absent owner means initialization failed before reads were dispatched.
    ray.kill(source, no_restart=True)


def _cleanup_probe_pair(source, destination, reader_owner, *, timeout):
    failure = sys.exc_info()[1]
    try:
        _terminate_probe_pair(source, destination, reader_owner, timeout=timeout)
    except SourceTeardownBlockedError as blocked:
        # This safety decision must reach the CLI even while another probe
        # error is already unwinding; generic driver shutdown is not safe.
        if failure is not None:
            raise blocked from failure
        raise
    except Exception:
        if failure is None:
            raise
        # Preserve the probe's original error; the failed fence intentionally
        # leaves the source alive rather than reclaiming possibly active memory.
        logger.exception("Failed to fence preflight reader; source retained")

def _run_pairwise_rdma_probes(
    args,
    *,
    ray_nodes_by_ip: Mapping[str, Mapping[str, Any]],
    configured_routes: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    source_ips = sorted(set(str(ip) for ip in args.inference_node_ips))
    destination_ips = sorted(set(str(ip) for ip in args.training_node_ips))
    pairs = [
        (source_ip, destination_ip)
        for source_ip in source_ips
        for destination_ip in destination_ips
        if source_ip != destination_ip
    ]
    if not pairs:
        raise ClusterPreflightError(
            "static multi-node RDMA requires at least one cross-node inference-to-training pair"
        )

    timeout_seconds = float(args.cluster_preflight_timeout_seconds)
    run_id = f"cluster-preflight-{uuid.uuid4().hex}"
    actor = ray.remote(num_cpus=0, num_gpus=1, max_restarts=0)(_GpuDirectProbeActor)
    results: list[dict[str, Any]] = []
    for pair_index, (source_ip, destination_ip) in enumerate(pairs):
        pair_id = f"{pair_index}-{source_ip}-to-{destination_ip}"
        attempt_id = f"{run_id}-{pair_index}"
        common = {
            "run_id": run_id,
            "attempt_id": attempt_id,
            "timeout_seconds": timeout_seconds,
        }
        source_route = configured_routes[source_ip]
        destination_route = configured_routes[destination_ip]
        source = actor.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ray_nodes_by_ip[source_ip]["NodeID"], soft=False
            )
        ).remote(
            role="source",
            protocol="rdma",
            device_name=source_route["device_name"],
            rdma_port=int(source_route["port"]),
            rdma_gid_index=int(source_route["gid_index"]),
            **common,
        )
        destination = None
        reader_owner = None
        try:
            destination = actor.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray_nodes_by_ip[destination_ip]["NodeID"], soft=False
                )
            ).remote(
                role="destination",
                protocol="rdma",
                device_name=destination_route["device_name"],
                rdma_port=int(destination_route["port"]),
                rdma_gid_index=int(destination_route["gid_index"]),
                **common,
            )
            reader_owner = ray.get(
                destination.get_owner_status.remote(), timeout=timeout_seconds
            )
            descriptor_key = ray.get(
                source.stage.remote(
                    run_id=run_id,
                    attempt_id=attempt_id,
                    pair_id=pair_id,
                ),
                timeout=timeout_seconds,
            )
            transfer = ray.get(
                destination.read_and_validate.remote(descriptor_key),
                timeout=timeout_seconds,
            )
            drained = ray.get(
                source.wait_for_release.remote(timeout_seconds),
                timeout=timeout_seconds,
            )
            source_status, destination_status = ray.get(
                [source.status.remote(), destination.status.remote()],
                timeout=timeout_seconds,
            )
            if not transfer["matched"] or not drained:
                raise ClusterPreflightError(
                    f"RDMA data-plane probe failed for {source_ip} -> {destination_ip}: "
                    f"matched={transfer['matched']}, source_drained={drained}"
                )
            if source_status["ring"]["in_use"] or destination_status["ring"]["in_use"]:
                raise ClusterPreflightError(
                    f"RDMA data-plane probe leaked a slot for {source_ip} -> {destination_ip}"
                )
            ray.get(
                [source.close.remote(), destination.close.remote()],
                timeout=timeout_seconds,
            )
            results.append(
                {
                    "source_ip": source_ip,
                    "destination_ip": destination_ip,
                    "state": "validated",
                    "transfer": transfer,
                    "source_transport": source_status,
                    "destination_transport": destination_status,
                }
            )
        except ClusterPreflightError:
            raise
        except BaseException as exc:
            raise ClusterPreflightError(
                f"RDMA data-plane probe failed for {source_ip} -> {destination_ip}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        finally:
            _cleanup_probe_pair(
                source, destination, reader_owner, timeout=timeout_seconds
            )
    return results


def _run_node_local_nvlink_probes(
    args,
    *,
    ray_nodes_by_ip: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Move one registered-CUDA payload between two GPUs on every planned node."""

    timeout_seconds = float(args.cluster_preflight_timeout_seconds)
    run_id = f"cluster-preflight-{uuid.uuid4().hex}"
    actor = ray.remote(num_cpus=0, num_gpus=1, max_restarts=0)(_GpuDirectProbeActor)
    results: list[dict[str, Any]] = []
    for pair_index, node_ip in enumerate(sorted(ray_nodes_by_ip)):
        node_id = ray_nodes_by_ip[node_ip]["NodeID"]
        pair_id = f"{pair_index}-{node_ip}-local"
        attempt_id = f"{run_id}-{pair_index}"
        common = {
            "protocol": "nvlink_intra",
            "device_name": "",
            "rdma_port": 1,
            "rdma_gid_index": 0,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "timeout_seconds": timeout_seconds,
        }
        strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        source = actor.options(scheduling_strategy=strategy).remote(
            role="source", **common
        )
        destination = None
        reader_owner = None
        try:
            destination = actor.options(scheduling_strategy=strategy).remote(
                role="destination", **common
            )
            reader_owner = ray.get(
                destination.get_owner_status.remote(), timeout=timeout_seconds
            )
            descriptor_key = ray.get(
                source.stage.remote(
                    run_id=run_id,
                    attempt_id=attempt_id,
                    pair_id=pair_id,
                ),
                timeout=timeout_seconds,
            )
            transfer = ray.get(
                destination.read_and_validate.remote(descriptor_key),
                timeout=timeout_seconds,
            )
            drained = ray.get(
                source.wait_for_release.remote(timeout_seconds),
                timeout=timeout_seconds,
            )
            source_status, destination_status = ray.get(
                [source.status.remote(), destination.status.remote()],
                timeout=timeout_seconds,
            )
            if not transfer["matched"] or not drained:
                raise ClusterPreflightError(
                    f"NVLink data-plane probe failed on {node_ip}: "
                    f"matched={transfer['matched']}, source_drained={drained}"
                )
            if source_status["ring"]["in_use"] or destination_status["ring"]["in_use"]:
                raise ClusterPreflightError(
                    f"NVLink data-plane probe leaked a slot on {node_ip}"
                )
            source_placement = source_status.get("probe_placement") or {}
            destination_placement = destination_status.get("probe_placement") or {}
            source_gpus = set(source_placement.get("assigned_gpu_ids") or ())
            destination_gpus = set(
                destination_placement.get("assigned_gpu_ids") or ()
            )
            if (
                source_placement.get("ray_node_id") != node_id
                or destination_placement.get("ray_node_id") != node_id
                or len(source_gpus) != 1
                or len(destination_gpus) != 1
                or source_gpus & destination_gpus
            ):
                raise ClusterPreflightError(
                    f"NVLink probe placement is not a distinct GPU pair on {node_ip}: "
                    f"source={source_placement}, destination={destination_placement}"
                )
            ray.get(
                [source.close.remote(), destination.close.remote()],
                timeout=timeout_seconds,
            )
            results.append(
                {
                    "node_ip": node_ip,
                    "transfer": transfer,
                    "source": source_status,
                    "destination": destination_status,
                    "source_drained": drained,
                }
            )
        except ClusterPreflightError:
            raise
        except BaseException as exc:
            raise ClusterPreflightError(
                f"NVLink data-plane probe failed on {node_ip}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        finally:
            _cleanup_probe_pair(
                source, destination, reader_owner, timeout=timeout_seconds
            )
    return results


def run_cluster_preflight(args) -> dict[str, Any] | None:
    """Probe every planned node before model loading or placement allocation."""

    if getattr(args, "cluster_mode", "single_node") != "static_multi_node":
        return None
    if not ray.is_initialized():
        raise ClusterPreflightError("Ray must be initialized before cluster preflight")

    expected_gpu_counts = _expected_gpu_counts(args)
    expected_ips = set(expected_gpu_counts)
    alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
    ray_nodes_by_ip = {str(node.get("NodeManagerAddress")): node for node in alive_nodes}
    observed_ips = set(ray_nodes_by_ip)
    missing_ips = expected_ips - observed_ips
    if missing_ips:
        raise ClusterPreflightError(
            "planned nodes are missing from live Ray membership: "
            f"missing={sorted(missing_ips)}, observed={sorted(observed_ips)}"
        )

    from aspec.ray.node_local_files import prepare_resume

    prepare_resume(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / f".cluster-preflight-{uuid.uuid4().hex}.json"
    marker_payload = json.dumps(
        {"plan_id": args.aspec_plan_id, "nonce": uuid.uuid4().hex},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    marker.write_bytes(marker_payload)
    marker_sha256 = hashlib.sha256(marker_payload).hexdigest()

    probe = ray.remote(num_cpus=0, runtime_env={"env_vars": get_aspec_env_vars(args)})(
        _collect_local_inventory
    )
    refs = []
    try:
        for node_ip in sorted(expected_ips):
            node_id = ray_nodes_by_ip[node_ip]["NodeID"]
            roles = (["draft"] if node_ip in (getattr(args, "training_node_ips", None) or ()) else [])
            if node_ip in (getattr(args, "inference_node_ips", None) or ()):
                roles.append("target")
            service = getattr(args, "target_service", None) or {}
            target_address = ((service.get("addresses") or {}).get(node_ip, service.get(
                "address", "http://127.0.0.1:30000")) if "target" in roles else None)
            refs.append(
                probe.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=False
                    )
                ).remote(_required_paths(args), str(marker), marker_sha256,
                         getattr(args, "cluster_storage_mode", "shared"), str(output_dir),
                         target_address, roles)
            )
        inventories = ray.get(
            refs,
            timeout=float(getattr(args, "cluster_preflight_timeout_seconds", 120.0)),
        )
    finally:
        try:
            marker.unlink()
        except FileNotFoundError:
            pass

    configured_rdma_routes: dict[str, dict[str, Any]] = {}
    if getattr(args, "mooncake_protocol", "tcp") == "rdma":
        configured_rdma_routes = resolve_rdma_routes(
            sorted(expected_ips),
            device_name=getattr(args, "mooncake_device_name", ""),
            port=getattr(args, "mooncake_rdma_port", 1),
            gid_index=getattr(args, "mooncake_rdma_gid_index", None),
            overrides=getattr(args, "mooncake_rdma_routes", None),
        )
    required_memlock_bytes = (
        _required_rdma_memlock_bytes(args) if configured_rdma_routes else 0
    )
    validate_cluster_inventories(
        inventories,
        expected_gpu_counts=expected_gpu_counts,
        required_rdma_routes=configured_rdma_routes,
        required_memlock_bytes=required_memlock_bytes,
        require_homogeneous_gpus=bool(
            getattr(args, "cluster_require_homogeneous_gpus", True)
        ),
    )
    locality_domains = _validate_and_resolve_locality_domains(args, inventories)
    selected_rdma_routes: list[dict[str, Any]] = []
    rdma_pairwise_probes: list[dict[str, Any]] = []
    node_local_nvlink_probes: list[dict[str, Any]] = []
    if configured_rdma_routes:
        selected_rdma_routes = _selected_rdma_routes(
            inventories,
            configured_routes=configured_rdma_routes,
        )
        rdma_pairwise_probes = _run_pairwise_rdma_probes(
            args,
            ray_nodes_by_ip=ray_nodes_by_ip,
            configured_routes=configured_rdma_routes,
        )
    if (
        getattr(args, "mooncake_protocol", None) == "nvlink_intra"
        and getattr(args, "mooncake_feature_affinity", "any")
        == "node_local_required"
    ):
        node_local_nvlink_probes = _run_node_local_nvlink_probes(
            args,
            ray_nodes_by_ip={ip: ray_nodes_by_ip[ip] for ip in sorted(expected_ips)},
        )
    return {
        "schema_version": CLUSTER_PREFLIGHT_SCHEMA_VERSION,
        "state": "validated",
        "mode": "static_multi_node",
        "expected_node_ips": sorted(expected_ips),
        "unplanned_live_node_ips": sorted(observed_ips - expected_ips),
        "expected_gpu_counts": expected_gpu_counts,
        "locality_domains": locality_domains,
        "configured_rdma_routes": configured_rdma_routes,
        "required_memlock_bytes_per_endpoint": required_memlock_bytes,
        "selected_rdma_routes": selected_rdma_routes,
        "rdma_pairwise_probes": rdma_pairwise_probes,
        "node_local_nvlink_probes": node_local_nvlink_probes,
        "transport": {
            "data_plane": getattr(args, "mooncake_data_plane", None),
            "protocol": getattr(args, "mooncake_protocol", None),
            "feature_affinity": getattr(
                args, "mooncake_feature_affinity", "any"
            ),
            "transport_stage": (
                "host_staged_tcp"
                if getattr(args, "mooncake_data_plane", None) == "store"
                else (
                    "rdma_gpu_direct"
                    if getattr(args, "mooncake_protocol", None) == "rdma"
                    else (
                        "node_local_nvlink_registered_cuda_ring"
                        if getattr(args, "mooncake_protocol", None) == "nvlink_intra"
                        else "generation_checked_tcp"
                    )
                )
            ),
            "staging_mode": (
                "pinned_host_buffers"
                if getattr(args, "mooncake_data_plane", None) == "store"
                else "registered_cuda_ring"
            ),
            "host_staging_selected": (
                getattr(args, "mooncake_data_plane", None) == "store"
            ),
            "registered_cuda_buffers_required": bool(
                getattr(args, "mooncake_enable_gpu_direct", False)
            ),
            "network_gpu_direct_required": (
                getattr(args, "mooncake_protocol", None) == "rdma"
                and bool(getattr(args, "mooncake_enable_gpu_direct", False))
            ),
            "host_fallback_allowed": bool(
                getattr(args, "mooncake_allow_host_fallback", True)
            ),
        },
        "ray_cluster_resources": {
            key: value
            for key, value in sorted(ray.cluster_resources().items())
            if key in {"CPU", "GPU", "memory", "object_store_memory"}
            or key.startswith("accelerator_type:")
            or key.startswith("node:")
        },
        "nodes": sorted(inventories, key=lambda node: node["node_ip"]),
    }
