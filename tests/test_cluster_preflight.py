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

from types import SimpleNamespace

import pytest

from aspec.ray.cluster_preflight import (
    ClusterPreflightError,
    _path_inventory,
    _validate_and_resolve_locality_domains,
    validate_cluster_inventories,
)


def _inventory(
    node_ip: str,
    *,
    gpu_name: str = "NVIDIA H200",
    rdma_active=True,
    rdma_userspace_access=True,
    device_name="mlx5_bond_0",
    gid_index=3,
    gid_type="RoCE v2",
    mtu="9000",
    netdev_state="up",
    memlock="unlimited",
):
    return {
        "node_id": f"ray-{node_ip}",
        "node_ip": node_ip,
        "hostname": f"host-{node_ip}",
        "source_tree_sha256": "source-sha",
        "packages": {"torch": "2.9.1", "ray": "2.58.0"},
        "runtime_artifacts": {
            "sglang": {"effective_tree_sha256": "sglang-sha", "file_count": 10},
            "mooncake": {"effective_tree_sha256": "mooncake-sha", "file_count": 2},
        },
        "gpus": [
            {
                "index": index,
                "reported_name": gpu_name,
                "uuid": f"GPU-{node_ip}-{index}",
                "memory_mib": 143771,
                "driver_version": "570.148.08",
                "compute_capability": "9.0",
            }
            for index in range(8)
        ],
        "rdma_devices": [
            {
                "name": device_name,
                "active": rdma_active,
                "ports": [
                    {
                        "port": 1,
                        "active": rdma_active,
                        "gids": [
                            {
                                "index": gid_index,
                                "gid": f"0000:0000:0000:0000:0000:ffff:{node_ip}",
                                "type": gid_type,
                                "netdev": "bond0",
                            }
                        ],
                        "netdev_facts": {
                            "bond0": {"operstate": netdev_state, "mtu": mtu}
                        },
                    }
                ],
                "verbs_devices": ["uverbs0"],
                "userspace_paths": ["/dev/infiniband/uverbs0"],
                "userspace_access": rdma_userspace_access,
            }
        ],
        "memlock_limit_bytes": {"soft": memlock, "hard": "unlimited"},
        "paths": [
            {
                "path": "/example/ASpec",
                "state": "available",
                "kind": "directory",
                "device": 17,
                "inode": 42,
            }
        ],
        "shared_marker": {"matched": True},
    }


def _validate(inventories):
    validate_cluster_inventories(
        inventories,
        expected_gpu_counts={"10.0.0.1": 4, "10.0.0.2": 4},
        required_rdma_routes={
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
        },
        required_memlock_bytes=1024 * 1024,
        require_homogeneous_gpus=True,
    )


def test_static_cluster_inventory_accepts_homogeneous_shared_nodes():
    _validate([_inventory("10.0.0.1"), _inventory("10.0.0.2")])


def test_separate_runtime_roles_do_not_require_sglang_on_draft_nodes():
    draft, target = _inventory("10.0.0.1"), _inventory("10.0.0.2")
    draft["roles"], target["roles"] = ["draft"], ["target"]
    draft["runtime_artifacts"]["sglang"] = {"path": None}
    draft["packages"].update({"sglang": None, "nvidia-cutlass-dsl": "4.7.1"})
    target["packages"].update({"sglang": "0.5.21", "nvidia-cutlass-dsl": "4.6.2"})
    target["target_runtime"] = {"version": "0.1.0b0", "code_sha256": "same"}
    _validate([draft, target])


def test_split_target_node_cannot_omit_its_environment_observation():
    draft, target = _inventory("10.0.0.1"), _inventory("10.0.0.2")
    draft["roles"], target["roles"] = ["draft"], ["target"]
    with pytest.raises(ClusterPreflightError, match="isolated runtime"):
        _validate([draft, target])


def test_locality_domain_geometry_resolves_live_node_identity():
    args = SimpleNamespace(
        training_node_ips=["10.0.0.1", "10.0.0.2"],
        training_num_gpus_per_node=4,
        inference_node_ips=["10.0.0.1", "10.0.0.2"],
        inference_num_gpus=8,
        inference_num_gpus_per_node=4,
        aspec_locality_domains=[
            {
                "domain_id": f"physical-node-{index}",
                "node_address": node,
                "training_world_ranks": list(range(index * 4, index * 4 + 4)),
                "draft_dp_ranks": list(range(index * 4, index * 4 + 4)),
                "draft_parallel_group_size": 1,
                "target_engine_ranks": list(range(index * 4, index * 4 + 4)),
                "target_tensor_parallel_size": 1,
                "dispatch_samples_per_microstep": 4,
            }
            for index, node in enumerate(("10.0.0.1", "10.0.0.2"))
        ],
    )

    resolved = _validate_and_resolve_locality_domains(
        args,
        [_inventory("10.0.0.1"), _inventory("10.0.0.2")],
    )

    assert [item["ray_node_id"] for item in resolved] == [
        "ray-10.0.0.1",
        "ray-10.0.0.2",
    ]


def test_locality_domain_geometry_rejects_rank_drift():
    args = SimpleNamespace(
        training_node_ips=["10.0.0.1"],
        training_num_gpus_per_node=4,
        inference_node_ips=["10.0.0.2"],
        inference_num_gpus=4,
        inference_num_gpus_per_node=4,
        aspec_locality_domains=[
            {
                "domain_id": "training",
                "node_address": "10.0.0.1",
                "training_world_ranks": [0, 1, 2, 4],
                "draft_dp_ranks": [0, 1, 2, 3],
                "draft_parallel_group_size": 1,
                "target_engine_ranks": [],
                "target_tensor_parallel_size": 1,
            },
            {
                "domain_id": "inference",
                "node_address": "10.0.0.2",
                "training_world_ranks": [],
                "draft_dp_ranks": [],
                "draft_parallel_group_size": 1,
                "target_engine_ranks": [0, 1, 2, 3],
                "target_tensor_parallel_size": 1,
            },
        ],
    )

    with pytest.raises(ClusterPreflightError, match="contiguous unique"):
        _validate_and_resolve_locality_domains(
            args,
            [_inventory("10.0.0.1"), _inventory("10.0.0.2")],
        )


def test_static_cluster_inventory_accepts_distinct_per_node_single_rail_routes():
    inventories = [
        _inventory("10.0.0.1"),
        _inventory("10.0.0.2", device_name="mlx5_7", gid_index=5),
    ]

    validate_cluster_inventories(
        inventories,
        expected_gpu_counts={"10.0.0.1": 4, "10.0.0.2": 4},
        required_rdma_routes={
            "10.0.0.1": {
                "device_name": "mlx5_bond_0",
                "port": 1,
                "gid_index": 3,
            },
            "10.0.0.2": {
                "device_name": "mlx5_7",
                "port": 1,
                "gid_index": 5,
            },
        },
        required_memlock_bytes=1024 * 1024,
        require_homogeneous_gpus=True,
    )


def test_static_cluster_inventory_accepts_nfs_client_device_identity_drift():
    first = _inventory("10.0.0.1")
    second = _inventory("10.0.0.2")
    second["paths"][0]["device"] = 1048780

    _validate([first, second])


def test_static_cluster_inventory_rejects_membership_drift():
    with pytest.raises(ClusterPreflightError, match="membership"):
        _validate([_inventory("10.0.0.1"), _inventory("10.0.0.3")])


def test_static_cluster_inventory_rejects_heterogeneous_gpus():
    with pytest.raises(ClusterPreflightError, match="not homogeneous"):
        _validate(
            [_inventory("10.0.0.1"), _inventory("10.0.0.2", gpu_name="NVIDIA H100")]
        )


def test_static_cluster_inventory_rejects_inactive_rdma():
    with pytest.raises(ClusterPreflightError, match="not active"):
        _validate(
            [_inventory("10.0.0.1"), _inventory("10.0.0.2", rdma_active=False)]
        )


def test_static_cluster_inventory_rejects_inaccessible_rdma_userspace_device():
    with pytest.raises(ClusterPreflightError, match="no usable verbs device"):
        _validate(
            [
                _inventory("10.0.0.1"),
                _inventory("10.0.0.2", rdma_userspace_access=False),
            ]
        )


def test_static_cluster_inventory_rejects_gid_type_drift():
    with pytest.raises(ClusterPreflightError, match="GID type differs"):
        _validate(
            [_inventory("10.0.0.1"), _inventory("10.0.0.2", gid_type="RoCE v1")]
        )


def test_static_cluster_inventory_accepts_mtu_drift_when_routes_are_valid():
    _validate([_inventory("10.0.0.1"), _inventory("10.0.0.2", mtu="1500")])


def test_static_cluster_inventory_rejects_down_gid_netdev():
    with pytest.raises(ClusterPreflightError, match="netdev is not up"):
        _validate(
            [_inventory("10.0.0.1"), _inventory("10.0.0.2", netdev_state="down")]
        )


def test_static_cluster_inventory_accepts_sufficient_bounded_memlock():
    _validate(
        [_inventory("10.0.0.1"), _inventory("10.0.0.2", memlock=2 * 1024 * 1024)]
    )


def test_static_cluster_inventory_rejects_insufficient_bounded_memlock():
    with pytest.raises(ClusterPreflightError, match="below the planned"):
        _validate([_inventory("10.0.0.1"), _inventory("10.0.0.2", memlock=65536)])


def test_static_cluster_inventory_rejects_shared_marker_mismatch():
    first = _inventory("10.0.0.1")
    second = _inventory("10.0.0.2")
    second["shared_marker"]["matched"] = False

    with pytest.raises(ClusterPreflightError, match="shared marker mismatch"):
        _validate([first, second])


def test_static_cluster_inventory_accepts_client_local_inode_drift():
    first = _inventory("10.0.0.1")
    second = _inventory("10.0.0.2")
    second["paths"][0]["inode"] = 99

    _validate([first, second])


def test_static_cluster_inventory_rejects_shared_file_content_drift():
    first = _inventory("10.0.0.1")
    second = _inventory("10.0.0.2")
    for node, digest in ((first, "a" * 64), (second, "b" * 64)):
        node["paths"][0].update(
            {"kind": "file", "size": 8, "content_sha256": digest}
        )

    with pytest.raises(ClusterPreflightError, match="file.*identities"):
        _validate([first, second])


def test_static_cluster_inventory_rejects_effective_backend_drift():
    first = _inventory("10.0.0.1")
    second = _inventory("10.0.0.2")
    second["runtime_artifacts"]["sglang"]["effective_tree_sha256"] = "other"

    with pytest.raises(ClusterPreflightError, match="runtime artifacts differ"):
        _validate([first, second])


def test_path_inventory_records_filesystem_capacity(tmp_path):
    [entry] = _path_inventory([str(tmp_path)])

    assert entry["state"] == "available"
    assert entry["filesystem"]["capacity_bytes"] > 0
    assert entry["filesystem"]["free_bytes"] >= 0
    assert entry["filesystem"]["available_bytes"] >= 0
