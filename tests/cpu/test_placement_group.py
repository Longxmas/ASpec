# MIT License
#
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Adapted from TorchSpec 6c042a87140a84d13839e341ece2c5c3ada918bc:
# tests/test_placement_group.py. Namespace bindings and ASpec regression coverage adapted.

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from aspec.ray import placement_group as placement
from aspec.ray.placement_group import (
    _build_custom_bundles,
    _NodeConstraint,
    _sort_probed_bundle_infos,
    create_placement_groups,
)


def _make_args(**overrides):
    defaults = dict(
        placement_strategy="training_first",
        colocate=False,
        training_num_nodes=1,
        training_num_gpus_per_node=2,
        inference_num_gpus=2,
        inference_num_gpus_per_node=2,
        training_node_ips=None,
        inference_node_ips=None,
        training_node_selectors=None,
        inference_node_selectors=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_custom_ip_bundles_add_node_resources():
    bundles, selectors, node_groups = _build_custom_bundles(
        "training",
        [_NodeConstraint(ip="10.0.0.1"), _NodeConstraint(ip="10.0.0.2")],
        total_gpus=3,
        gpus_per_node=2,
    )

    assert bundles == [
        {"GPU": 1, "CPU": 1, "node:10.0.0.1": 0.001},
        {"GPU": 1, "CPU": 1, "node:10.0.0.1": 0.001},
        {"GPU": 1, "CPU": 1, "node:10.0.0.2": 0.001},
    ]
    assert selectors == [{}, {}, {}]
    assert node_groups == [0, 0, 1]


def test_external_target_does_not_require_gpus_in_draft_ray():
    args = _make_args(inference_engine_type="sgl", inference_num_gpus=8)
    reservation = (MagicMock(name="draft_pg"), [0, 1], [4, 5])
    with (
        patch.object(placement, "_ensure_ray_initialized"),
        patch.object(placement, "_wait_for_gpu_resources") as wait,
        patch.object(placement, "_create_placement_group", return_value=reservation) as create,
    ):
        result = create_placement_groups(args)
    wait.assert_called_once_with(2)
    create.assert_called_once_with(2, strategy="PACK", name="training_pg")
    assert result == {"training": reservation, "inference": None}


def test_target_only_generation_needs_no_draft_gpu_reservation():
    args = _make_args(inference_engine_type="sgl")
    with (
        patch.object(placement, "_ensure_ray_initialized"),
        patch.object(placement, "_wait_for_gpu_resources") as wait,
    ):
        assert create_placement_groups(args, roles={"inference"}) == {"training": None, "inference": None}
    wait.assert_not_called()


def test_custom_label_bundles_add_bundle_label_selectors():
    bundles, selectors, node_groups = _build_custom_bundles(
        "inference",
        [
            _NodeConstraint(label_selector=(("aspec/node", "infer-0"),)),
            _NodeConstraint(label_selector=(("aspec/node", "infer-1"),)),
        ],
        total_gpus=3,
        gpus_per_node=2,
    )

    assert bundles == [
        {"GPU": 1, "CPU": 1},
        {"GPU": 1, "CPU": 1},
        {"GPU": 1, "CPU": 1},
    ]
    assert selectors == [
        {"aspec/node": "infer-0"},
        {"aspec/node": "infer-0"},
        {"aspec/node": "infer-1"},
    ]
    assert node_groups == [0, 0, 1]


def test_custom_bundles_validate_node_count():
    with pytest.raises(ValueError, match="expected 2 node"):
        _build_custom_bundles(
            "training",
            [_NodeConstraint(ip="10.0.0.1")],
            total_gpus=3,
            gpus_per_node=2,
        )


def test_custom_bundle_sort_preserves_user_node_order_then_gpu_id():
    gpu_ids = [
        ("10.0.0.2", 7),
        ("10.0.0.1", 1),
        ("10.0.0.2", 0),
        ("10.0.0.1", 0),
    ]

    sorted_infos = _sort_probed_bundle_infos(gpu_ids, node_group_indices=[1, 0, 1, 0])

    assert [info[0] for info in sorted_infos] == [3, 1, 2, 0]


def test_create_placement_groups_requires_custom_strategy_for_custom_fields():
    args = _make_args(training_node_ips=["10.0.0.1"])

    with (
        patch("aspec.ray.placement_group._ensure_ray_initialized"),
        patch("aspec.ray.placement_group._wait_for_gpu_resources") as wait_for_gpus,
        pytest.raises(ValueError, match="placement_strategy=custom"),
    ):
        create_placement_groups(args)

    wait_for_gpus.assert_not_called()


def test_create_placement_groups_validates_custom_constraints_before_waiting_for_gpus():
    args = _make_args(
        placement_strategy="custom",
        training_num_nodes=2,
        training_num_gpus_per_node=2,
        training_node_ips=["10.0.0.1"],
        inference_node_ips=["10.0.0.2"],
    )

    with (
        patch("aspec.ray.placement_group._ensure_ray_initialized"),
        patch("aspec.ray.placement_group._wait_for_gpu_resources") as wait_for_gpus,
        pytest.raises(ValueError, match="training custom placement expected 2 node"),
    ):
        create_placement_groups(args)

    wait_for_gpus.assert_not_called()


def test_create_placement_groups_custom_unified_uses_role_node_order():
    args = _make_args(
        placement_strategy="custom",
        training_num_nodes=1,
        training_num_gpus_per_node=2,
        inference_num_gpus=2,
        inference_num_gpus_per_node=2,
        training_node_ips=["10.0.0.1"],
        inference_node_ips=["10.0.0.2"],
    )
    fake_pg = MagicMock(name="pg")

    with (
        patch("aspec.ray.placement_group._ensure_ray_initialized"),
        patch("aspec.ray.placement_group._wait_for_gpu_resources"),
        patch(
            "aspec.ray.placement_group._create_placement_group",
            return_value=(fake_pg, [0, 1, 2, 3], [0, 1, 0, 1]),
        ) as create_pg,
    ):
        result = create_placement_groups(args)

    assert create_pg.call_count == 1
    kwargs = create_pg.call_args.kwargs
    assert kwargs["bundles"][0]["node:10.0.0.1"] == 0.001
    assert kwargs["bundles"][2]["node:10.0.0.2"] == 0.001
    assert kwargs["node_group_indices"] == [0, 0, 1, 1]
    assert result["training"] == (fake_pg, [0, 1], [0, 1])
    assert result["inference"] == (fake_pg, [2, 3], [0, 1])


def test_create_placement_groups_custom_unified_allows_zero_inference_gpus():
    args = _make_args(
        placement_strategy="custom",
        training_num_nodes=1,
        training_num_gpus_per_node=2,
        inference_num_gpus=0,
        inference_num_gpus_per_node=2,
        training_node_ips=["10.0.0.1"],
        inference_node_ips=None,
    )
    fake_pg = MagicMock(name="pg")

    with (
        patch("aspec.ray.placement_group._ensure_ray_initialized"),
        patch("aspec.ray.placement_group._wait_for_gpu_resources"),
        patch(
            "aspec.ray.placement_group._create_placement_group",
            return_value=(fake_pg, [0, 1], [0, 1]),
        ) as create_pg,
    ):
        result = create_placement_groups(args)

    kwargs = create_pg.call_args.kwargs
    assert kwargs["bundles"] == [
        {"GPU": 1, "CPU": 1, "node:10.0.0.1": 0.001},
        {"GPU": 1, "CPU": 1, "node:10.0.0.1": 0.001},
    ]
    assert kwargs["node_group_indices"] == [0, 0]
    assert result["training"] == (fake_pg, [0, 1], [0, 1])
    assert result["inference"] == (fake_pg, [], [])


def test_custom_colocate_uses_training_topology_for_inference_constraints():
    args = _make_args(
        placement_strategy="custom",
        colocate=True,
        training_num_nodes=2,
        training_num_gpus_per_node=4,
        inference_num_gpus_per_node=8,
        training_node_ips=None,
        inference_node_ips=["10.0.0.1", "10.0.0.2"],
    )
    fake_pg = MagicMock(name="pg")

    with (
        patch("aspec.ray.placement_group._ensure_ray_initialized"),
        patch("aspec.ray.placement_group._wait_for_gpu_resources"),
        patch(
            "aspec.ray.placement_group._create_placement_group",
            return_value=(fake_pg, list(range(8)), list(range(8))),
        ) as create_pg,
    ):
        result = create_placement_groups(args)

    kwargs = create_pg.call_args.kwargs
    assert len(kwargs["bundles"]) == 8
    assert kwargs["bundles"][0]["node:10.0.0.1"] == 0.001
    assert kwargs["bundles"][4]["node:10.0.0.2"] == 0.001
    assert kwargs["node_group_indices"] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert result["training"] == (fake_pg, list(range(8)), list(range(8)))
    assert result["inference"] == (fake_pg, list(range(8)), list(range(8)))


def test_inference_only_placement_reserves_only_inference_gpus():
    args = _make_args(inference_num_gpus=2)
    fake_pg = MagicMock(name="pg")

    with (
        patch("aspec.ray.placement_group._ensure_ray_initialized"),
        patch("aspec.ray.placement_group._wait_for_gpu_resources") as wait_for_gpus,
        patch(
            "aspec.ray.placement_group._create_placement_group",
            return_value=(fake_pg, [0, 1], [0, 1]),
        ) as create_pg,
    ):
        result = create_placement_groups(args, roles={"inference"})

    wait_for_gpus.assert_called_once_with(2)
    create_pg.assert_called_once_with(2, strategy="PACK", name="inference_pg")
    assert result["training"] == (fake_pg, [], [])
    assert result["inference"] == (fake_pg, [0, 1], [0, 1])


def test_training_only_placement_ignores_role_ordering_strategy():
    args = _make_args(
        inference_engine_type="offline",
        placement_strategy="inference_first",
        inference_num_gpus=8,
    )
    fake_pg = MagicMock(name="pg")

    with (
        patch("aspec.ray.placement_group._ensure_ray_initialized"),
        patch("aspec.ray.placement_group._wait_for_gpu_resources") as wait_for_gpus,
        patch(
            "aspec.ray.placement_group._create_placement_group",
            return_value=(fake_pg, [0, 1], [0, 1]),
        ) as create_pg,
    ):
        result = create_placement_groups(args, roles={"training"})

    wait_for_gpus.assert_called_once_with(2)
    create_pg.assert_called_once_with(2, strategy="PACK", name="training_pg")
    assert result["training"] == (fake_pg, [0, 1], [0, 1])
    assert result["inference"] == (fake_pg, [], [])


def test_single_role_placement_rejects_custom_strategy():
    args = _make_args(
        inference_engine_type="offline",
        placement_strategy="custom",
        training_node_ips=["10.0.0.1"],
    )

    with (
        patch("aspec.ray.placement_group._ensure_ray_initialized"),
        patch("aspec.ray.placement_group._wait_for_gpu_resources") as wait_for_gpus,
        pytest.raises(ValueError, match="only supported when training and inference"),
    ):
        create_placement_groups(args, roles={"training"})

    wait_for_gpus.assert_not_called()


@pytest.mark.parametrize("phase", ["ready", "probe", "actor_creation"])
def test_failed_placement_releases_its_reservation_and_probes(phase):
    pg, probe = MagicMock(), MagicMock()
    ready_ref = pg.ready.return_value
    info = MagicMock()
    info.options.return_value.remote.side_effect = (
        RuntimeError("probe creation failed") if phase == "actor_creation" else None
    )
    info.options.return_value.remote.return_value = probe
    failure = RuntimeError("placement probe failed")

    def get(ref, **_kwargs):
        if ref is ready_ref and phase != "ready":
            return None
        raise failure

    with (
        patch.object(placement, "_placement_group", return_value=pg),
        patch.object(placement, "InfoActor", info),
        patch.object(placement, "PlacementGroupSchedulingStrategy"),
        patch.object(placement.ray, "get", side_effect=get),
        patch.object(placement.ray, "kill") as kill,
        patch.object(placement.ray.util, "remove_placement_group") as remove,
        pytest.raises(RuntimeError, match="failed"),
    ):
        placement._create_placement_group(1)
    remove.assert_called_once_with(pg)
    if phase == "probe":
        kill.assert_called_once_with(probe, no_restart=True)
    else:
        kill.assert_not_called()


def test_placement_ready_and_probe_share_a_bounded_budget():
    pg, probe = MagicMock(), MagicMock()
    waits = []

    def get(ref, **kwargs):
        timeout = kwargs.get("timeout")
        assert timeout is not None and 0 < timeout <= 300
        waits.append(timeout)
        return None if ref is pg.ready.return_value else [("127.0.0.1", 0)]

    with (
        patch.object(placement, "_placement_group", return_value=pg),
        patch.object(placement.InfoActor, "options") as options,
        patch.object(placement, "PlacementGroupSchedulingStrategy"),
        patch.object(placement.ray, "get", side_effect=get),
        patch.object(placement.ray, "kill") as kill,
        patch.object(placement.ray.util, "remove_placement_group") as remove,
    ):
        options.return_value.remote.return_value = probe
        result = placement._create_placement_group(1)
    assert result == (pg, [0], [0])
    assert len(waits) == 2 and waits[1] <= waits[0]
    kill.assert_called_once_with(probe, no_restart=True)
    remove.assert_not_called()


@pytest.mark.parametrize("nodes,assigned,expected_error", [
    (["here", "other"], [4, 6], "coordinator node only"),
    (["here"], [0, 1], "GPU split"),
    (["here"], [6, 4], None),
])
def test_managed_target_checks_actual_draft_allocation(monkeypatch, nodes, assigned, expected_error):
    args = _make_args(inference_engine_type="sgl", target_service={"mode": "managed-local"})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,6")
    reservation = (object(), [0, 1], assigned)
    with (
        patch.object(placement, "_ensure_ray_initialized"),
        patch.object(placement, "_wait_for_gpu_resources"),
        patch.object(placement, "_create_placement_group", return_value=reservation) as reserve,
        patch.object(placement, "ray") as ray,
    ):
        ray.nodes.return_value = [{"NodeID": node, "Alive": True} for node in nodes]
        ray.get_runtime_context.return_value.get_node_id.return_value = "here"
        if expected_error:
            with pytest.raises(ValueError, match=expected_error):
                create_placement_groups(args)
        else:
            assert create_placement_groups(args)["training"] == reservation
        if len(nodes) > 1:
            reserve.assert_not_called()
        elif expected_error:
            ray.util.remove_placement_group.assert_called_once_with(reservation[0])
