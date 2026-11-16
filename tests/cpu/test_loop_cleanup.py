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
# tests/test_loop_cleanup.py. Namespace bindings and ASpec regression coverage adapted.

from types import SimpleNamespace
from unittest import mock

import pytest
from omegaconf import OmegaConf

from aspec.controller import eval as eval_utils
from aspec.controller import loop


def test_write_training_metrics_logs_each_step_loss():
    with (
        mock.patch("aspec.controller.loop.logger.info") as mock_log,
        mock.patch("aspec.controller.loop.get_tb_writer", return_value=None),
    ):
        loop._write_training_metrics({"train/avg_loss": 1.23456789}, train_step=2, inference_step=3)

    mock_log.assert_called_once_with("TRAIN_STEP step=2 loss=1.234568")


def test_performance_observation_uses_one_contiguous_window():
    records = [
        {
            "optimizer_step": 1,
            "committed_work": 2,
            "started_seconds": 0.0,
            "finished_seconds": 2.0,
            "step_wall_seconds": 2.0,
            "dispatch_wait_seconds": 1.0,
        },
        {
            "optimizer_step": 2,
            "committed_work": 2,
            "started_seconds": 3.0,
            "finished_seconds": 5.0,
            "step_wall_seconds": 2.0,
            "dispatch_wait_seconds": 0.5,
        },
        {
            "optimizer_step": 3,
            "committed_work": 2,
            "started_seconds": 6.0,
            "finished_seconds": 9.0,
            "step_wall_seconds": 3.0,
            "dispatch_wait_seconds": 0.5,
        },
        {
            "optimizer_step": 4,
            "committed_work": 2,
            "started_seconds": 10.0,
            "finished_seconds": 14.0,
            "step_wall_seconds": 4.0,
            "dispatch_wait_seconds": 0.5,
        },
    ]

    result = loop._build_performance_observation(
        records, warmup_steps=1, drain_steps=1
    )

    assert result["optimizer_steps"] == 2
    assert result["committed_work"] == 4
    assert result["window_wall_seconds"] == 6.0
    assert result["committed_work_per_second"] == pytest.approx(2 / 3)
    assert result["step_wall_seconds"]["median"] == 2.5


def test_feature_contract_observation_joins_producer_and_consumers():
    args = SimpleNamespace(
        aspec_feature_contract={"contract_id": "sha256:contract"},
        world_size=2,
    )
    engines = [mock.MagicMock(), mock.MagicMock()]
    producers = [
        {
            "participant": f"engine_{rank}",
            "contract_id": "sha256:contract",
            "binding_validations": 1,
            "metadata_validations": rank,
            "tensor_validations": 0,
        }
        for rank in range(2)
    ]
    consumers = [
        {
            "participant": f"trainer_{rank}",
            "contract_id": "sha256:contract",
            "binding_validations": 0,
            "metadata_validations": 2,
            "tensor_validations": 2,
            "data_admission": {"neutralized_rank_samples": rank},
        }
        for rank in range(2)
    ]
    train_group = mock.MagicMock()
    train_group.get_feature_contract_status.return_value = consumers

    with mock.patch("aspec.controller.loop.ray.get", return_value=producers):
        observation = loop._collect_feature_contract_observation(args, train_group, engines)

    assert observation["state"] == "validated"
    assert observation["producer_count"] == 2
    assert observation["consumer_count"] == 2
    assert observation["producer_metadata_validations"] == 1
    assert observation["consumer_tensor_validations"] == 4
    assert observation["data_admission"] == {
        "neutralized_rank_samples": 1,
        "participants": [
            {"participant": "trainer_0", "neutralized_rank_samples": 0},
            {"participant": "trainer_1", "neutralized_rank_samples": 1},
        ],
    }


def test_feature_contract_observation_rejects_unvalidated_consumer():
    args = SimpleNamespace(
        aspec_feature_contract={"contract_id": "sha256:contract"},
        world_size=1,
    )
    producer = {
        "participant": "engine_0",
        "contract_id": "sha256:contract",
        "binding_validations": 1,
        "metadata_validations": 1,
    }
    train_group = mock.MagicMock()
    train_group.get_feature_contract_status.return_value = [
        {
            "participant": "trainer_0",
            "contract_id": "sha256:contract",
            "metadata_validations": 1,
            "tensor_validations": 0,
        }
    ]

    with (
        mock.patch("aspec.controller.loop.ray.get", return_value=[producer]),
        pytest.raises(RuntimeError, match="did not validate transported tensors"),
    ):
        loop._collect_feature_contract_observation(args, train_group, [mock.MagicMock()])


@pytest.mark.parametrize("shared_reader", [None, "draft_a"])
def test_transfer_observation_accepts_descriptor_scoped_producer_metrics(shared_reader):
    args = SimpleNamespace(
        aspec_feature_contract={"contract_id": "sha256:contract"},
        aspec_run_id="run-1",
        aspec_attempt_id="attempt-1",
        mooncake_data_plane="transfer_engine",
        mooncake_protocol="tcp",
        world_size=1,
        aspec_shared_source=shared_reader is not None,
        aspec_shared_reader=shared_reader,
    )
    identity = {
        "state": "active",
        "protocol": "tcp",
        "strict_transport": True,
        "run_id": "run-1",
        "attempt_id": "attempt-1",
    }
    producer = {
        "participant": "engine_0",
        "role": "producer",
        "contract_id": "sha256:contract",
        "binding_validations": 1,
        "metadata_validations": 1,
        "transport": {
            **identity,
            "activation": "descriptor_observed",
            "ring": {"in_use": 0},
            "notification_worker_error": None,
            "metrics": {
                "source_descriptors": 1,
                "source_releases": 1,
                "discarded_source_descriptors": 0,
                "source_payload_bytes": 128,
                "discarded_source_payload_bytes": 0,
                "transfer_failures": 0,
                "ring_wait_seconds": None,
            },
        },
    }
    consumer = {
        "participant": "trainer_0",
        "role": "consumer",
        "contract_id": "sha256:contract",
        "metadata_validations": 1,
        "tensor_validations": 1,
        "transport": {
            **identity,
            "activation": "initialized",
            "ring": {"in_use": 0},
            "metrics": {
                "destination_reads": 1,
                "destination_releases": 1,
                "destination_payload_bytes": 128,
                "destination_transfer_seconds": 0.5,
                "destination_max_transfer_seconds": 0.4,
                "transfer_failures": 0,
                "ring_wait_seconds": 0.25,
            },
        },
    }
    train_group = mock.MagicMock()
    train_group.get_feature_contract_status.return_value = [consumer]

    if shared_reader:
        producer["transport"]["metrics"].update({
            "shared_expected_reads": 2, "shared_expected_payload_bytes": 256,
            "shared_completed_readers": {"draft_a": {"reads": 1, "payload_bytes": 128},
                                         "failed_b": {"reads": 1, "payload_bytes": 128}},
        })
    with mock.patch("aspec.controller.loop.ray.get", return_value=[producer]):
        observation = loop._collect_feature_contract_observation(
            args, train_group, [mock.MagicMock()]
        )

    transport = observation["transport"]
    assert transport["source_descriptors"] == 1
    assert transport["destination_reads"] == 1
    assert transport["destination_max_transfer_seconds"] == pytest.approx(0.4)
    assert transport["ring_wait_seconds"] == pytest.approx(0.25)
    assert transport["ring_wait_participants_observed"] == 1
    assert transport["ring_wait_participants_total"] == 2


def test_transfer_observation_allows_planned_but_idle_producer():
    args = SimpleNamespace(
        aspec_feature_contract={"contract_id": "sha256:contract"},
        aspec_run_id="run-1",
        aspec_attempt_id="attempt-1",
        mooncake_data_plane="transfer_engine",
        mooncake_protocol="rdma",
        mooncake_device_name="mlx5_bond_0",
        mooncake_rdma_port=1,
        mooncake_rdma_gid_index=3,
        world_size=1,
    )
    identity = {
        "state": "active",
        "protocol": "rdma",
        "strict_transport": True,
        "run_id": "run-1",
        "attempt_id": "attempt-1",
        "configured_rdma_route": {
            "device_name": "mlx5_bond_0",
            "port": 1,
            "gid_index": 3,
        },
    }
    active_producer = {
        "participant": "engine_0",
        "role": "producer",
        "contract_id": "sha256:contract",
        "binding_validations": 1,
        "metadata_validations": 1,
        "transport": {
            **identity,
            "activation": "descriptor_observed",
            "ring": {"in_use": 0},
            "notification_worker_error": None,
            "metrics": {
                "source_descriptors": 1,
                "source_releases": 1,
                "discarded_source_descriptors": 0,
                "source_payload_bytes": 128,
                "discarded_source_payload_bytes": 0,
                "transfer_failures": 0,
                "ring_wait_seconds": None,
            },
        },
    }
    idle_producer = {
        "participant": "engine_1",
        "role": "producer",
        "contract_id": "sha256:contract",
        "binding_validations": 1,
        "metadata_validations": 0,
        "transport": {
            **identity,
            "activation": "strict_rdma_initialized",
            "ring": {"in_use": 0},
            "notification_worker_error": None,
            "metrics": {
                "source_descriptors": 0,
                "source_releases": 0,
                "discarded_source_descriptors": 0,
                "source_payload_bytes": 0,
                "discarded_source_payload_bytes": 0,
                "transfer_failures": 0,
                "ring_wait_seconds": 0.0,
            },
        },
    }
    consumer = {
        "participant": "trainer_0",
        "role": "consumer",
        "contract_id": "sha256:contract",
        "metadata_validations": 1,
        "tensor_validations": 1,
        "transport": {
            **identity,
            "activation": "strict_rdma_initialized",
            "ring": {"in_use": 0},
            "metrics": {
                "destination_reads": 1,
                "destination_releases": 1,
                "destination_payload_bytes": 128,
                "destination_transfer_seconds": 0.5,
                "destination_max_transfer_seconds": 0.4,
                "transfer_failures": 0,
                "ring_wait_seconds": 0.0,
            },
        },
    }
    train_group = mock.MagicMock()
    train_group.get_feature_contract_status.return_value = [consumer]

    with mock.patch(
        "aspec.controller.loop.ray.get",
        return_value=[active_producer, idle_producer],
    ):
        observation = loop._collect_feature_contract_observation(
            args, train_group, [mock.MagicMock(), mock.MagicMock()]
        )

    transport = observation["transport"]
    assert transport["state"] == "validated"
    assert transport["planned_producer_endpoints"] == 2
    assert transport["producer_endpoints"] == 2
    assert transport["producer_endpoints_with_data"] == 1
    assert transport["inactive_producer_participants"] == ["engine_1"]
    assert transport["consumer_endpoints"] == 1


def test_transfer_observation_rejects_active_producer_without_transport():
    args = SimpleNamespace(
        aspec_feature_contract={"contract_id": "sha256:contract"},
        aspec_run_id="run-1",
        aspec_attempt_id="attempt-1",
        mooncake_data_plane="transfer_engine",
        mooncake_protocol="rdma",
        mooncake_device_name="mlx5_bond_0",
        mooncake_rdma_port=1,
        mooncake_rdma_gid_index=3,
        world_size=1,
    )
    producer = {
        "participant": "engine_0",
        "role": "producer",
        "contract_id": "sha256:contract",
        "binding_validations": 1,
        "metadata_validations": 1,
    }
    consumer = {
        "participant": "trainer_0",
        "role": "consumer",
        "contract_id": "sha256:contract",
        "metadata_validations": 1,
        "tensor_validations": 1,
        "transport": {},
    }
    train_group = mock.MagicMock()
    train_group.get_feature_contract_status.return_value = [consumer]

    with (
        mock.patch("aspec.controller.loop.ray.get", return_value=[producer]),
        pytest.raises(
            RuntimeError,
            match="did not report terminal data-plane Evidence.*engine_0",
        ),
    ):
        loop._collect_feature_contract_observation(
            args, train_group, [mock.MagicMock()]
        )


class TestTrainingLoopCleanup:
    def test_unused_output_drain_confirms_storage_release(self):
        output = SimpleNamespace(metadata={"aspec_execution_id": "work"})
        controller = mock.MagicMock()
        controller.pop_unused_inference_results.remote.side_effect = [
            "pop-with-output",
            "pop-empty",
            "pop-final",
        ]
        controller.confirm_discarded_inference_results.remote.return_value = "confirmed"
        actor = mock.MagicMock()
        actor.discard_inference_outputs.remote.return_value = "discarded"
        train_group = SimpleNamespace(_actor_handlers=[actor])
        future = object()

        def get(ref, timeout=None):
            if ref == "pop-with-output":
                return [output]
            if ref in {"pop-empty", "pop-final"}:
                return []
            if ref == "discarded":
                return 1
            return None

        with (
            mock.patch("aspec.controller.loop.ray.get", side_effect=get),
            mock.patch("aspec.controller.loop.ray.wait", return_value=([future], [])),
        ):
            released = loop._drain_unused_inference_outputs(
                SimpleNamespace(inference_shutdown_timeout_seconds=1),
                controller,
                future,
                train_group,
            )

        assert released == 1
        actor.discard_inference_outputs.remote.assert_called_once_with([output])
        controller.confirm_discarded_inference_results.remote.assert_called_once_with([output])

    def test_safe_cleanup_stops_inference_and_shutdowns_mooncake_actor(self):
        args = SimpleNamespace(_mooncake_master_actor=mock.MagicMock())
        inference_manager = mock.MagicMock()
        inference_future = object()
        train_group = mock.MagicMock()
        controller = mock.MagicMock()

        stop_ref = inference_manager.stop.remote.return_value
        shutdown_ref = args._mooncake_master_actor.shutdown.remote.return_value
        finalize_ref = object()
        lifecycle_ref = object()
        pop_ref = object()
        controller.finalize_uncommitted_work.remote.return_value = finalize_ref
        controller.get_work_lifecycle_status.remote.return_value = lifecycle_ref
        controller.pop_unused_inference_results.remote.return_value = pop_ref

        def get(ref, timeout=None):
            if ref is finalize_ref:
                return {"discarded": 0}
            if ref is lifecycle_ref:
                return {"state": "validated"}
            if ref is pop_ref:
                return []
            return mock.MagicMock()

        with (
            mock.patch(
                "aspec.controller.loop.ray.get", side_effect=get
            ) as mock_ray_get,
            mock.patch(
                "aspec.controller.loop.ray.wait",
                return_value=([inference_future], []),
            ),
        ):
            result = loop._safe_training_cleanup(
                args,
                inference_manager,
                inference_future,
                controller=controller,
                train_group=train_group,
            )

        mock_ray_get.assert_any_call(stop_ref, timeout=10)
        mock_ray_get.assert_any_call(inference_future)
        mock_ray_get.assert_any_call(shutdown_ref, timeout=10)
        mock_ray_get.assert_any_call(
            controller.finalize_uncommitted_work.remote.return_value, timeout=10
        )
        train_group.close.assert_called_once_with()
        assert result["state"] == "completed"
        assert {item["component"] for item in result["operations"]} == {
            "inference_manager_stop",
            "training_queues",
            "unused_inference_outputs",
            "inference_manager_run_loop",
            "inference_manager_status",
            "training_actors",
            "uncommitted_work",
            "mooncake_master",
            "work_lifecycle",
        }

    def test_safe_cleanup_reports_errors_without_hiding_them(self):
        args = SimpleNamespace()
        inference_manager = mock.MagicMock()
        inference_future = object()

        with (
            mock.patch(
                "aspec.controller.loop.ray.get",
                side_effect=RuntimeError("stop failed"),
            ),
            mock.patch("aspec.controller.loop.ray.cancel"),
        ):
            result = loop._safe_training_cleanup(args, inference_manager, inference_future)

        assert result["state"] == "completed_with_errors"
        failed = [item for item in result["operations"] if "failed" in item["state"]]
        assert failed
        assert all("error" in item for item in failed)

    def test_safe_cleanup_skips_mooncake_shutdown_when_actor_missing(self):
        args = SimpleNamespace()
        inference_manager = mock.MagicMock()
        inference_future = object()

        with mock.patch("aspec.controller.loop.ray.get"):
            loop._safe_training_cleanup(args, inference_manager, inference_future)

        inference_manager.stop.remote.assert_called_once()

    def test_forced_cleanup_kills_owner_actors_without_graceful_waits(self):
        mooncake = mock.MagicMock()
        args = SimpleNamespace(_mooncake_master_actor=mooncake)
        inference_manager = mock.MagicMock()
        controller = mock.MagicMock()
        trainers = [mock.MagicMock(), mock.MagicMock()]
        engines = [mock.MagicMock(), mock.MagicMock()]
        train_group = SimpleNamespace(_actor_handlers=trainers, _owner_statuses=[{"pid": 1}])

        lifecycle = {"state": "incomplete", "resident_outputs": 1}
        finalize_ref = controller.finalize_uncommitted_work.remote.return_value
        lifecycle_ref = controller.get_work_lifecycle_status.remote.return_value
        fenced_ref = controller.finalize_fenced_work.remote.return_value
        terminal = {"state": "validated", "resident_outputs": 0, "resident_bytes": 0}

        def get(ref, timeout=None):
            if ref is finalize_ref:
                return {"discarded": 2}
            if ref is lifecycle_ref:
                return lifecycle
            if ref is fenced_ref:
                return {"discarded": 1, "lifecycle": terminal}
            return None

        with (
            mock.patch("aspec.controller.loop.ray.get", side_effect=get),
            mock.patch("aspec.controller.loop.ray.kill") as kill,
            mock.patch("aspec.controller.loop.wait_for_owner_exit", return_value={"state": "verified"}),
        ):
            result = loop._safe_training_cleanup(
                args,
                inference_manager,
                object(),
                inference_engines=engines,
                controller=controller,
                train_group=train_group,
                force_kill_trainers=True,
            )

        inference_manager.stop.remote.assert_not_called()
        controller.shutdown.remote.assert_called_once_with()
        for actor in [inference_manager, *trainers, *engines, controller]:
            kill.assert_any_call(actor, no_restart=True)
        assert result["state"] == "completed"
        assert result["mode"] == "forced_owner_fence"
        assert result["work_lifecycle"] == terminal
        controller.finalize_fenced_work.remote.assert_called_once_with(owners_fenced=True)
        assert {
            (item["component"], item["state"])
            for item in result["operations"]
        } >= {
            ("training_queues", "stop_signaled"),
            ("uncommitted_work", "discarded_before_force"),
            ("work_lifecycle", "captured_before_force"),
        }

    def test_forced_cleanup_treats_missing_queue_node_as_observed_fence(self):
        args = SimpleNamespace()
        inference_manager = mock.MagicMock()
        controller = mock.MagicMock()
        shutdown_ref = controller.shutdown.remote.return_value
        finalize_ref = controller.finalize_uncommitted_work.remote.return_value
        lifecycle_ref = controller.get_work_lifecycle_status.remote.return_value

        def get(ref, timeout=None):
            if ref is shutdown_ref:
                raise RuntimeError("queue owner node unavailable")
            if ref is finalize_ref:
                return {"discarded": 0}
            if ref is lifecycle_ref:
                return {"state": "incomplete", "resident_outputs": 1}
            return None

        with (
            mock.patch("aspec.controller.loop.ray.get", side_effect=get),
            mock.patch("aspec.controller.loop.ray.kill"),
        ):
            result = loop._safe_training_cleanup(
                args,
                inference_manager,
                object(),
                controller=controller,
                force_kill_trainers=True,
            )

        queue_operation = next(
            item
            for item in result["operations"]
            if item["component"] == "training_queues"
        )
        assert queue_operation["state"] == "stop_unavailable_before_force"
        assert "queue owner node unavailable" in queue_operation["error"]
        assert result["state"] == "completed_with_errors"
        assert result["work_lifecycle"]["resident_outputs"] == 1
        controller.finalize_fenced_work.remote.assert_not_called()

    def test_run_training_loop_finally_runs_cleanup_on_success(self):
        args = SimpleNamespace(_mooncake_master_actor=mock.MagicMock())
        inference_manager = mock.MagicMock()
        inference_future = inference_manager.run.remote.return_value

        with (
            mock.patch("aspec.controller.loop.training_loop", return_value="ok") as mock_impl,
            mock.patch(
                "aspec.controller.loop._safe_training_cleanup",
                return_value={"feature_contract_error": None},
            ) as mock_cleanup,
        ):
            result = loop.run_training_loop(args, object(), inference_manager, object())

        assert result == "ok"
        mock_impl.assert_called_once()
        mock_cleanup.assert_called_once_with(
            args=args,
            inference_manager=inference_manager,
            inference_future=inference_future,
            inference_engines=None,
            controller=mock.ANY,
            train_group=mock.ANY,
        )

    def test_run_training_loop_finally_runs_cleanup_on_exception(self):
        args = SimpleNamespace(_mooncake_master_actor=mock.MagicMock())
        inference_manager = mock.MagicMock()
        inference_future = inference_manager.run.remote.return_value

        with (
            mock.patch(
                "aspec.controller.loop.training_loop",
                side_effect=RuntimeError("training failed"),
            ),
            mock.patch("aspec.controller.loop._safe_training_cleanup") as mock_cleanup,
        ):
            with pytest.raises(RuntimeError, match="training failed"):
                loop.run_training_loop(args, object(), inference_manager, object())

        mock_cleanup.assert_called_once_with(
            args=args,
            inference_manager=inference_manager,
            inference_future=inference_future,
            inference_engines=None,
            controller=mock.ANY,
            train_group=mock.ANY,
            force_kill_trainers=True,
        )

def test_run_training_loop_publishes_cleanup_observation():
    args = SimpleNamespace(_mooncake_master_actor=mock.MagicMock())
    inference_manager = mock.MagicMock()
    cleanup = {"state": "completed", "operations": []}
    observer = mock.Mock()

    with (
        mock.patch("aspec.controller.loop.training_loop", return_value={"completed_step": 1}),
        mock.patch("aspec.controller.loop._safe_training_cleanup", return_value=cleanup),
    ):
        result = loop.run_training_loop(
            args,
            object(),
            inference_manager,
            object(),
            cleanup_observer=observer,
        )

    observer.assert_called_once_with(cleanup)
    assert result["cleanup"] == cleanup


def test_producer_profile_restarts_source_before_training_budget():
    args = SimpleNamespace(
        topology_producer_profile_samples=1,
        topology_producer_profile_warmup_samples=0,
        optimizer_step_timeout_seconds=10,
        mooncake_data_plane="transfer_engine",
    )
    output = object()
    events = []

    controller = mock.MagicMock()
    controller.pop_unused_training_results.remote.return_value = [output]
    controller.confirm_discarded_inference_results.remote.return_value = None
    controller.restart_training_source_after_profile.remote.side_effect = (
        lambda: events.append("restart")
        or {
            "state": "reloaded",
            "discarded_prompts": 2,
            "reloaded_samples": 3,
            "dataset_epoch": 1,
        }
    )

    inference_manager = mock.MagicMock()
    inference_manager.get_status.remote.return_value = {
        "sample_budget_drained": True,
        "current_pool_size": 0,
        "scheduled_samples": 1,
    }
    inference_manager.start_training_budget.remote.side_effect = (
        lambda: events.append("start_budget") or {"phase": "training"}
    )

    actor = mock.MagicMock()
    actor.discard_inference_outputs.remote.return_value = 1
    train_group = mock.MagicMock()
    train_group._actor_handlers = [actor]

    with (
        mock.patch("aspec.controller.loop.ray.get", side_effect=lambda value, **_kw: value),
        mock.patch(
            "aspec.controller.loop.time.monotonic",
            side_effect=iter(i * 0.01 for i in range(100)),
        ),
    ):
        observation = loop._run_producer_service_profile(
            args, controller, inference_manager, train_group
        )

    assert events == ["restart", "start_budget"]
    assert observation["source_restart"]["reloaded_samples"] == 3
    assert observation["next_budget_phase"] == "training"


def _run_mock_training_loop(
    *,
    prefetch_depth: int,
    num_steps: int,
    steps_per_epoch: int,
    dispatch_results=None,
    shared_source=False,
    start_step=0,
    tracking=None,
    train_metrics=None,
    final_metrics=None,
    source_work_budget=None,
    accumulation_steps=1,
    streaming_dispatch=False,
    max_sample_pool_size=0,
    committed_work_totals=None,
    inference_statuses=None,
    packing_enabled=None,
    micro_batch_size=1,
    rpc_get=None,
    resume_state=None,
    continual_training=False,
    submitted_samples=None,
    require_initial_reload=False,
):
    args = SimpleNamespace(
        training_num_nodes=1,
        training_num_gpus_per_node=1,
        num_train_steps=num_steps,
        draft_accumulation_steps=accumulation_steps,
        steps_per_epoch=steps_per_epoch,
        num_epochs=(num_steps + steps_per_epoch - 1) // steps_per_epoch,
        global_batch_size=accumulation_steps * micro_batch_size,
        per_dp_rank_batch_size=micro_batch_size,
        prefetch_depth=prefetch_depth,
        enable_perf_metrics=False,
        save_interval=0,
        checkpoint_dir=None,
        save_per_epoch=False,
        train_with_decode=False,
        aspec_shared_source=shared_source,
        source_work_budget=source_work_budget,
        requested_source_work_budget=source_work_budget,
        source_work_alignment=1 if source_work_budget is not None else None,
        planned_dropped_source_work=0,
        source_work_tail_policy=(
            "drop_last_complete_packed_optimizer_step"
            if source_work_budget is not None
            else None
        ),
        dflash_packing=(
            source_work_budget is not None if packing_enabled is None else packing_enabled
        ),
        max_sample_pool_size=max_sample_pool_size,
        streaming_dispatch_with_training=streaming_dispatch,
        continual_training=continual_training,
    )
    events = []
    results = iter(dispatch_results) if dispatch_results is not None else None

    controller = mock.MagicMock()
    controller.submit_training_dataset.remote.return_value = submitted_samples
    controller.reload_dataset.remote.return_value = None

    def dispatch(
        *, optimizer_step=None, microstep=None, force_packing_progress=False,
        final_source_drained=False, max_packing_documents=None,
        return_document_count=False,
    ):
        assert optimizer_step is not None
        assert microstep is not None
        if require_initial_reload and not controller.reload_dataset.remote.call_count:
            raise RuntimeError("Restored epoch has no uncommitted rows; source must advance before dispatch")
        result = next(results) if results is not None else True
        events.append(f"dispatch:{result}")
        return result

    controller.try_dispatch_batch.remote.side_effect = dispatch
    committed_totals = iter(committed_work_totals) if committed_work_totals else None
    controller.commit_optimizer_step.remote.side_effect = (
        lambda optimizer_step, execution_ids, expected_unique: {
            "committed_work": (
                next(committed_totals)
                if committed_totals is not None
                else source_work_budget
                if source_work_budget is not None
                else optimizer_step - start_step
            )
        }
    )
    controller.get_full_status.remote.return_value = {
        "inference_speed": 0.0,
        "sample_pool_size": 0,
        "dispatch_batch_size": 1,
        "prompt_buffer_size": 0,
        "elapsed_seconds": 0.0,
        "avg_inference_speed": 0.0,
        "avg_training_speed": 0.0,
    }

    actor = mock.MagicMock()
    actor.get_global_step.remote.return_value = start_step
    metric_results = iter(train_metrics) if train_metrics is not None else None

    def train(*, step, num_batches):
        events.append(
            f"train:{step}:{num_batches}"
            if source_work_budget is not None
            else f"train:{step}"
        )
        return next(metric_results) if metric_results is not None else {}

    actor.train_from_queue.remote.side_effect = train
    train_group = mock.MagicMock()
    train_group._actor_handlers = [actor]
    train_group.pop_committed_work.side_effect = lambda step, **_kw: [[f"execution-{step}"]]
    train_group.flush_pending_train_metrics.return_value = [final_metrics] if final_metrics else []

    eval_state = eval_utils.EvalSetupState(
        eval_interval=0,
        eval_enabled=False,
        eval_cache_loaded=False,
        eval_cache_path=None,
        best_eval_score=0.0,
        eval_dispatch_bs=0,
        eval_dataset_size=0,
        dp_size=1,
    )

    with (
        mock.patch("aspec.controller.loop.setup_eval", return_value=eval_state),
        mock.patch("aspec.controller.loop.load_checkpoint_work_state", return_value=resume_state),
        mock.patch("aspec.controller.loop.ray.get", side_effect=rpc_get or (lambda value, **_: value)),
        mock.patch(
            "aspec.controller.loop._wait_for_trainer_group",
            side_effect=lambda values, **_: values,
        ),
        mock.patch("aspec.controller.loop.time.sleep") as mock_sleep,
        mock.patch("aspec.controller.loop.tqdm"),
    ):
        inference_manager = mock.MagicMock()
        if inference_statuses is None:
            inference_manager.get_status.remote.return_value = {
                "sample_budget_drained": True,
            }
        else:
            inference_manager.get_status.remote.side_effect = iter(inference_statuses)
        loop.training_loop(
            args,
            controller,
            inference_manager,
            train_group,
            dataset_size=max(num_steps, source_work_budget or 0, 1),
            eval_dataset_size=0,
            tracking=tracking,
        )

    return events, controller, mock_sleep


def test_finite_packing_source_rejects_when_no_optimizer_step_fits():
    with pytest.raises(ValueError, match="one complete optimizer step"):
        _run_mock_training_loop(
            prefetch_depth=0,
            num_steps=4,
            steps_per_epoch=4,
            source_work_budget=1,
            accumulation_steps=2,
        )


@pytest.mark.parametrize(
    "dp_size,requested,scheduled",
    [(2, 7, 6), (3, 8, 6), (4, 11, 8)],
)
def test_finite_source_budget_accepts_resolved_drop_last_values(
    dp_size, requested, scheduled
):
    args = SimpleNamespace(
        requested_source_work_budget=requested,
        source_work_budget=scheduled,
        source_work_alignment=dp_size,
        per_dp_rank_batch_size=1,
        dflash_packing=True,
        global_batch_size=dp_size * 2,
    )

    assert loop._validate_source_work_budget(
        args, dataset_size=requested, dp_size=dp_size
    ) == scheduled


def test_finite_source_budget_rejects_unresolved_alignment():
    args = SimpleNamespace(
        source_work_budget=10,
        dflash_packing=False,
        global_batch_size=6,
    )

    with pytest.raises(ValueError, match="complete executable work"):
        loop._validate_source_work_budget(args, dataset_size=10, dp_size=3)


def test_finite_source_budget_validates_requested_count_against_dataset():
    args = SimpleNamespace(
        requested_source_work_budget=8,
        source_work_budget=6,
        source_work_alignment=3,
        dflash_packing=True,
        global_batch_size=6,
    )

    with pytest.raises(ValueError, match="requested source_work_budget"):
        loop._validate_source_work_budget(args, dataset_size=7, dp_size=3)


def test_finite_packing_budget_rejects_zero_optimizer_steps():
    args = SimpleNamespace(
        requested_source_work_budget=5,
        source_work_budget=3,
        source_work_alignment=3,
        per_dp_rank_batch_size=1,
        draft_accumulation_steps=2,
        dflash_packing=True,
        global_batch_size=6,
    )

    with pytest.raises(ValueError, match="one complete optimizer step"):
        loop._validate_source_work_budget(args, dataset_size=5, dp_size=3)


def test_finite_source_drained_mid_streaming_step_forces_full_accumulation():
    events, controller, _ = _run_mock_training_loop(
        prefetch_depth=0,
        num_steps=4,
        steps_per_epoch=4,
        dispatch_results=[True, False, True],
        source_work_budget=68,
        accumulation_steps=2,
        streaming_dispatch=True,
        max_sample_pool_size=64,
    )

    assert events == [
        "train:0:2",
        "dispatch:True",
        "dispatch:False",
        "dispatch:True",
    ]
    assert controller.try_dispatch_batch.remote.call_args_list[-1].kwargs[
        "force_packing_progress"
    ] is True
    controller.reload_dataset.remote.assert_not_called()


@pytest.mark.parametrize("micro_batch_size", [1, 3])
def test_finite_nonpacking_tail_streams_a_complete_optimizer_batch(micro_batch_size):
    budget = 2 * micro_batch_size
    events, controller, _ = _run_mock_training_loop(
        prefetch_depth=0,
        num_steps=1,
        steps_per_epoch=1,
        dispatch_results=[True, True],
        source_work_budget=budget,
        accumulation_steps=2,
        streaming_dispatch=True,
        max_sample_pool_size=budget,
        committed_work_totals=[budget],
        packing_enabled=False,
        micro_batch_size=micro_batch_size,
    )

    # Trainers must consume before the whole batch is staged. Otherwise bounded
    # source slots can fill at the final step and prevent further production.
    assert events == ["train:0:2", "dispatch:True", "dispatch:True"]
    controller.commit_optimizer_step.remote.assert_called_once()
    assert controller.commit_optimizer_step.remote.call_args.kwargs["expected_unique"] == budget
    controller.reload_dataset.remote.assert_not_called()


def test_finite_source_tail_streams_a_full_step_with_bounded_packing():
    events, controller, _ = _run_mock_training_loop(
        prefetch_depth=0,
        num_steps=2,
        steps_per_epoch=2,
        dispatch_results=[3, 3],
        source_work_budget=6,
        accumulation_steps=2,
        streaming_dispatch=True,
        max_sample_pool_size=6,
        committed_work_totals=[6],
    )

    assert events == [
        "train:0:2",
        "dispatch:3",
        "dispatch:3",
    ]
    calls = controller.try_dispatch_batch.remote.call_args_list
    assert [call.kwargs["max_packing_documents"] for call in calls] == [3, 3]
    assert all(call.kwargs["return_document_count"] for call in calls)
    controller.reload_dataset.remote.assert_not_called()


def test_finite_source_scheduled_budget_forces_packing_progress():
    events, controller, _ = _run_mock_training_loop(
        prefetch_depth=0,
        num_steps=1,
        steps_per_epoch=1,
        dispatch_results=[False, 3, 3],
        source_work_budget=6,
        accumulation_steps=2,
        streaming_dispatch=True,
        max_sample_pool_size=6,
        committed_work_totals=[6],
        inference_statuses=[
            {
                "sample_budget_drained": False,
                "remaining_sample_budget": 0,
            },
            {
                "sample_budget_drained": False,
                "remaining_sample_budget": 0,
            },
            {
                "sample_budget_drained": False,
                "remaining_sample_budget": 0,
            },
        ],
    )

    assert events == [
        "train:0:2",
        "dispatch:False",
        "dispatch:3",
        "dispatch:3",
    ]
    assert controller.try_dispatch_batch.remote.call_args_list[0].kwargs == {
        "optimizer_step": 0,
        "microstep": 0,
        "force_packing_progress": True,
        "final_source_drained": False,
        "max_packing_documents": 3,
        "return_document_count": True,
    }
    controller.reload_dataset.remote.assert_not_called()


def test_shared_resumed_deferred_metrics_and_final_flush_keep_own_step_axis(monkeypatch):
    tracker = mock.Mock()
    monkeypatch.setattr(loop.wandb, "log", mock.Mock(side_effect=AssertionError("global W&B used")))
    monkeypatch.setattr(loop, "get_tb_writer", mock.Mock(side_effect=AssertionError("global TensorBoard used")))
    _, controller, _ = _run_mock_training_loop(
        prefetch_depth=1, num_steps=8, steps_per_epoch=10, shared_source=True,
        start_step=6, tracking=tracker,
        train_metrics=[{}, {"_metrics_step": 6, "train/avg_loss": 1.5}],
        final_metrics={"_metrics_step": 7, "train/avg_loss": 2.5},
    )
    assert controller.submit_training_dataset.remote.call_args.kwargs["completed_steps"] == 6
    assert [call.kwargs["train_step"] for call in tracker.log.call_args_list] == [6, 7]
    assert [call.kwargs["inference_step"] for call in tracker.log.call_args_list] == [8, 8]
    assert [call.args[0]["train/step"] for call in tracker.log.call_args_list] == [6, 7]
    assert [call.args[0]["train/avg_loss"] for call in tracker.log.call_args_list] == [1.5, 2.5]


def test_wait_for_trainer_group_times_out_with_pending_ranks():
    future = object()
    with (
        mock.patch("aspec.controller.loop.ray.wait", return_value=([], [future])),
        mock.patch(
            "aspec.controller.loop.time.monotonic",
            side_effect=[10.0, 10.0, 10.2],
        ),
        pytest.raises(loop.TrainerStepTimeoutError, match=r"pending_ranks=\[0\]"),
    ):
        loop._wait_for_trainer_group([future], timeout_seconds=0.1)


def test_wait_for_trainer_group_checks_cancellation_before_waiting_for_native_compute():
    check = mock.Mock(side_effect=RuntimeError("experiment cancelled"))
    with mock.patch.object(loop.ray, "wait") as wait, pytest.raises(RuntimeError, match="cancelled"):
        loop._wait_for_trainer_group([object()], timeout_seconds=10, check_progress=check)
    check.assert_called_once_with()
    wait.assert_not_called()


@pytest.mark.parametrize("shared", [False, True])
def test_shared_source_wait_does_not_advance_experiment_epoch(shared):
    _, controller, _ = _run_mock_training_loop(
        prefetch_depth=0, num_steps=2, steps_per_epoch=10,
        dispatch_results=[True] + [False] * 500 + [True], shared_source=shared,
    )
    if shared:
        controller.reload_dataset.remote.assert_not_called()
    else:
        controller.reload_dataset.remote.assert_called_once_with()


def test_optimizer_step_timeout_includes_source_supply_phase():
    with (
        mock.patch("aspec.controller.loop.time.monotonic", return_value=10.1),
        pytest.raises(
            loop.TrainerStepTimeoutError,
            match="phase=source_supply_and_dispatch",
        ),
    ):
        loop._remaining_optimizer_step_time(
            deadline=10.0,
            timeout_seconds=0.1,
            phase="source_supply_and_dispatch",
        )


def test_source_wait_propagates_inference_failure():
    future = object()
    failure = RuntimeError("inference failed")

    with (
        mock.patch("aspec.controller.loop.ray.wait", return_value=([future], [])),
        mock.patch("aspec.controller.loop.ray.get", side_effect=failure),
        pytest.raises(RuntimeError, match="inference failed"),
    ):
        loop._raise_if_inference_finished(future)


def test_source_wait_rejects_unexpected_clean_inference_exit():
    future = object()

    with (
        mock.patch("aspec.controller.loop.ray.wait", return_value=([future], [])),
        mock.patch("aspec.controller.loop.ray.get", return_value=None),
        pytest.raises(RuntimeError, match="stopped before training completed"),
    ):
        loop._raise_if_inference_finished(future)


def test_trainer_timeout_forces_owner_process_cleanup():
    args = SimpleNamespace(_mooncake_master_actor=mock.MagicMock())
    inference_manager = mock.MagicMock()
    inference_future = inference_manager.run.remote.return_value

    with (
        mock.patch(
            "aspec.controller.loop.training_loop",
            side_effect=loop.TrainerStepTimeoutError("stuck transfer"),
        ),
        mock.patch("aspec.controller.loop._safe_training_cleanup") as cleanup,
        pytest.raises(loop.TrainerStepTimeoutError, match="stuck transfer"),
    ):
        loop.run_training_loop(args, object(), inference_manager, object())

    cleanup.assert_called_once_with(
        args=args,
        inference_manager=inference_manager,
        inference_future=inference_future,
        inference_engines=None,
        controller=mock.ANY,
        train_group=mock.ANY,
        force_kill_trainers=True,
    )


def test_training_loop_prefills_and_refills_controller_queue():
    events, _, _ = _run_mock_training_loop(
        prefetch_depth=2,
        num_steps=3,
        steps_per_epoch=3,
    )

    assert events == [
        "dispatch:True",
        "dispatch:True",
        "train:0",
        "dispatch:True",
        "train:1",
        "train:2",
    ]


def test_training_loop_prefetch_zero_preserves_one_step_dispatching():
    events, _, _ = _run_mock_training_loop(
        prefetch_depth=0,
        num_steps=3,
        steps_per_epoch=3,
    )

    assert events == [
        "dispatch:True",
        "train:0",
        "dispatch:True",
        "train:1",
        "dispatch:True",
        "train:2",
    ]


def test_training_loop_runs_ready_step_when_lookahead_cannot_be_filled():
    events, _, mock_sleep = _run_mock_training_loop(
        prefetch_depth=2,
        num_steps=3,
        steps_per_epoch=3,
        dispatch_results=[True, False, True, True],
    )

    assert events == [
        "dispatch:True",
        "dispatch:False",
        "train:0",
        "dispatch:True",
        "dispatch:True",
        "train:1",
        "train:2",
    ]
    mock_sleep.assert_not_called()


def test_training_loop_does_not_prefetch_across_epoch_boundary():
    events, controller, _ = _run_mock_training_loop(
        prefetch_depth=3,
        num_steps=4,
        steps_per_epoch=2,
    )

    assert events == [
        "dispatch:True",
        "dispatch:True",
        "train:0",
        "train:1",
        "dispatch:True",
        "dispatch:True",
        "train:2",
        "train:3",
    ]
    controller.reload_dataset.remote.assert_called_once_with()


def test_consumer_service_profile_uses_trainer_compute_not_step_wall_time():
    result = loop._build_consumer_service_profile(
        {1: 2.0, 2: 4.0, 3: 8.0},
        [
            {"optimizer_step": 1, "committed_work": 8, "step_wall_seconds": 20.0},
            {"optimizer_step": 2, "committed_work": 8, "step_wall_seconds": 30.0},
            {"optimizer_step": 3, "committed_work": 8, "step_wall_seconds": 40.0},
        ],
        warmup_steps=1,
        drain_steps=0,
    )

    assert result["state"] == "measured"
    assert result["committed_work"] == 16
    assert result["trainer_compute_seconds"] == 12.0
    assert result["work_per_second"] == pytest.approx(4 / 3)


def test_continual_source_cursor_does_not_restore_optimizer_frontier():
    loop._validate_resume_optimizer_frontier(
        start_step=0,
        resume_state={"optimizer_step": 12},
        continual_training=True,
    )


def test_exact_resume_requires_matching_optimizer_frontier():
    with pytest.raises(RuntimeError, match="trainer=0, work_state=12"):
        loop._validate_resume_optimizer_frontier(
            start_step=0,
            resume_state={"optimizer_step": 12},
            continual_training=False,
        )


@pytest.mark.parametrize("continual_training,start_step", [(True, 0), (False, 8)])
def test_empty_committed_epoch_advances_before_training(continual_training, start_step):
    restored = {
        "optimizer_step": 8,
        "dataset_epoch": 1,
        "committed_work_ids": [f"work-{i}" for i in range(8)],
        "committed_work_count": 8,
        "committed_work_sha256": "restored-work-digest",
        "dataset_fingerprint": "unchanged-corpus",
    }
    events, controller, _ = _run_mock_training_loop(
        prefetch_depth=0,
        num_steps=start_step + 1,
        steps_per_epoch=4,
        start_step=start_step,
        continual_training=continual_training,
        resume_state=restored,
        submitted_samples=0,
        require_initial_reload=True,
    )

    controller.submit_training_dataset.remote.assert_called_once_with(
        epoch=1,
        committed_work_ids=restored["committed_work_ids"],
        expected_dataset_fingerprint="unchanged-corpus",
    )
    controller.reload_dataset.remote.assert_called_once_with()
    assert events == ["dispatch:True", f"train:{start_step}"]


@pytest.mark.parametrize(
    "submitted_samples,shared_source,source_work_budget",
    [(2, False, None), (0, True, None), (0, False, 1)],
)
def test_restored_source_preserves_remaining_rows_and_source_ownership(
    submitted_samples, shared_source, source_work_budget
):
    _, controller, _ = _run_mock_training_loop(
        prefetch_depth=0,
        num_steps=1,
        steps_per_epoch=4,
        continual_training=True,
        resume_state={
            "optimizer_step": 8,
            "dataset_epoch": 1,
            "committed_work_ids": [f"work-{i}" for i in range(8)],
            "committed_work_count": 8,
            "committed_work_sha256": "restored-work-digest",
            "dataset_fingerprint": "unchanged-corpus",
        },
        submitted_samples=submitted_samples,
        shared_source=shared_source,
        source_work_budget=source_work_budget,
    )

    controller.reload_dataset.remote.assert_not_called()


def test_generate_eval_cache_dispatches_and_finalizes():
    controller = mock.MagicMock()
    inference_manager = mock.MagicMock()
    train_group = mock.MagicMock()

    controller.try_dispatch_eval_batch.remote.side_effect = [True, True, True, True]
    inference_manager.start_eval_budget.remote.return_value = {
        "phase": "eval",
        "sample_budget": 8,
    }
    inference_manager.finish_eval_budget.remote.return_value = {
        "phase": "training",
        "sample_budget": 4,
    }

    state = eval_utils.EvalSetupState(
        eval_interval=0,
        eval_enabled=True,
        eval_cache_loaded=False,
        eval_cache_path=None,
        best_eval_score=0.0,
        eval_dispatch_bs=2,
        eval_dataset_size=8,
        dp_size=2,
        initial_eval_submit_count=8,
    )

    with (
        mock.patch("aspec.controller.eval.ray.get", side_effect=lambda x: x),
        mock.patch("aspec.controller.eval.time.sleep"),
    ):
        eval_utils.generate_eval_cache(controller, inference_manager, train_group, state)

    inference_manager.start_eval_budget.remote.assert_called_once_with(8)
    controller.submit_eval_chunk.remote.assert_called_once_with(0, 8)
    assert train_group.cache_eval_samples.call_count == 4
    train_group.cache_eval_samples.assert_called_with(1)
    controller.finalize_eval_dispatch.remote.assert_called_once()
    inference_manager.finish_eval_budget.remote.assert_called_once_with()


def test_generate_eval_cache_interleaves_refill_and_drain_without_stalling():
    controller = mock.MagicMock()
    inference_manager = mock.MagicMock()
    train_group = mock.MagicMock()

    controller.try_dispatch_eval_batch.remote.side_effect = [
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
    ]
    inference_manager.start_eval_budget.remote.return_value = {
        "phase": "eval",
        "sample_budget": 8,
    }
    inference_manager.finish_eval_budget.remote.return_value = {
        "phase": "training",
        "sample_budget": 4,
    }

    state = eval_utils.EvalSetupState(
        eval_interval=0,
        eval_enabled=True,
        eval_cache_loaded=False,
        eval_cache_path=None,
        best_eval_score=0.0,
        eval_dispatch_bs=2,
        eval_dataset_size=8,
        dp_size=2,
        initial_eval_submit_count=4,
    )

    with (
        mock.patch("aspec.controller.eval.ray.get", side_effect=lambda x: x),
        mock.patch("aspec.controller.eval.time.sleep") as mock_sleep,
    ):
        eval_utils.generate_eval_cache(controller, inference_manager, train_group, state)

    assert train_group.cache_eval_samples.call_count == 4
    assert controller.submit_eval_chunk.remote.call_args_list == [
        mock.call(0, 4),
        mock.call(4, 6),
        mock.call(6, 8),
    ]
    controller.finalize_eval_dispatch.remote.assert_called_once()
    assert mock_sleep.call_count == 4


def test_generate_eval_cache_times_out_when_eval_never_arrives():
    controller = mock.MagicMock()
    inference_manager = mock.MagicMock()
    train_group = mock.MagicMock()

    state = eval_utils.EvalSetupState(
        eval_interval=0,
        eval_enabled=True,
        eval_cache_loaded=False,
        eval_cache_path=None,
        best_eval_score=0.0,
        eval_dispatch_bs=2,
        eval_dataset_size=8,
        dp_size=2,
        initial_eval_submit_count=4,
    )

    controller.try_dispatch_eval_batch.remote.return_value = False
    inference_manager.start_eval_budget.remote.return_value = {
        "phase": "eval",
        "sample_budget": 8,
    }
    with (
        mock.patch("aspec.controller.eval.ray.get", side_effect=lambda x: x),
        mock.patch("aspec.controller.eval.time.sleep") as mock_sleep,
        mock.patch("aspec.controller.eval.EVAL_CACHE_IDLE_TIMEOUT", 0.0),
    ):
        with pytest.raises(TimeoutError, match="Timed out while waiting for eval cache generation"):
            eval_utils.generate_eval_cache(
                controller, inference_manager, train_group, state
            )

    train_group.cache_eval_samples.assert_not_called()
    controller.finalize_eval_dispatch.remote.assert_not_called()
    mock_sleep.assert_not_called()


def test_setup_eval_dispatch_bs_is_dp_size(tmp_path):
    controller = mock.MagicMock()
    controller.get_dataset_admission_evidence.remote.return_value = {"eval": {"dataset_fingerprint": "sha256:eval"}}
    train_group = mock.MagicMock()
    train_group.load_eval_cache.return_value = [0]
    controller.submit_eval_chunk.remote.return_value = 4

    args = SimpleNamespace(
        eval_interval=50,
        dp_size=2,
        inference_batch_size=4,
        max_sample_pool_size=64,
        checkpoint_dir=None,
        cache_dir="./cache",
        eval_data_path="eval.jsonl",
        target_model_path=str(tmp_path),
        max_seq_length=4096,
    )

    with mock.patch("aspec.controller.eval.ray.get", side_effect=lambda x: x):
        state = eval_utils.setup_eval(
            controller=controller,
            train_group=train_group,
            args=args,
            eval_dataset_size=16,
        )

    assert state.eval_enabled is True
    assert state.eval_dispatch_bs == 2
    assert state.eval_dataset_size == 16
    assert state.initial_eval_submit_count == 4
    controller.submit_eval_chunk.remote.assert_not_called()


def test_setup_eval_dispatch_bs_caps_at_dataset_size(tmp_path):
    controller = mock.MagicMock()
    controller.get_dataset_admission_evidence.remote.return_value = {"eval": {"dataset_fingerprint": "sha256:eval"}}
    train_group = mock.MagicMock()
    train_group.load_eval_cache.return_value = [0]
    controller.submit_eval_chunk.remote.return_value = 4

    args = SimpleNamespace(
        eval_interval=50,
        dp_size=8,
        inference_batch_size=4,
        max_sample_pool_size=64,
        checkpoint_dir=None,
        cache_dir="./cache",
        eval_data_path="eval.jsonl",
        target_model_path=str(tmp_path),
        max_seq_length=4096,
    )

    with mock.patch("aspec.controller.eval.ray.get", side_effect=lambda x: x):
        state = eval_utils.setup_eval(
            controller=controller,
            train_group=train_group,
            args=args,
            eval_dataset_size=4,
        )

    assert state.eval_dispatch_bs == 4
    assert state.initial_eval_submit_count == 4
    controller.submit_eval_chunk.remote.assert_not_called()


def test_setup_eval_cache_key_includes_aux_hidden_state_layers(tmp_path):
    controller = mock.MagicMock()
    controller.get_dataset_admission_evidence.remote.return_value = {"eval": {"dataset_fingerprint": "sha256:eval"}}
    train_group = mock.MagicMock()
    train_group.load_eval_cache.return_value = [0]
    controller.submit_eval_chunk.remote.return_value = 1

    args = SimpleNamespace(
        eval_interval=50,
        dp_size=1,
        inference_batch_size=1,
        checkpoint_dir=None,
        cache_dir="./cache",
        eval_data_path="eval.jsonl",
        target_model_path=str(tmp_path),
        max_seq_length=4096,
        aux_hidden_states_layers=[1, 17, 32],
    )

    with mock.patch("aspec.controller.eval.ray.get", side_effect=lambda x: x):
        first = eval_utils.setup_eval(
            controller=controller,
            train_group=train_group,
            args=args,
            eval_dataset_size=1,
        )
        args.aux_hidden_states_layers = [1, 8, 16, 24, 32]
        second = eval_utils.setup_eval(
            controller=controller,
            train_group=train_group,
            args=args,
            eval_dataset_size=1,
        )

    assert first.eval_cache_path != second.eval_cache_path


def test_setup_eval_cache_key_accepts_omegaconf_list(tmp_path):
    controller = mock.MagicMock()
    controller.get_dataset_admission_evidence.remote.return_value = {"eval": {"dataset_fingerprint": "sha256:eval"}}
    train_group = mock.MagicMock()
    train_group.load_eval_cache.return_value = [0]
    controller.submit_eval_chunk.remote.return_value = 1

    args = SimpleNamespace(
        eval_interval=50,
        dp_size=1,
        inference_batch_size=1,
        checkpoint_dir=None,
        cache_dir="./cache",
        eval_data_path="eval.jsonl",
        target_model_path=str(tmp_path),
        max_seq_length=4096,
        aux_hidden_states_layers=OmegaConf.create([2, 46, 90, 93]),
    )

    with mock.patch("aspec.controller.eval.ray.get", side_effect=lambda x: x):
        state = eval_utils.setup_eval(
            controller=controller,
            train_group=train_group,
            args=args,
            eval_dataset_size=1,
        )

    assert state.eval_enabled is True
    train_group.load_eval_cache.assert_called_once()


def test_discard_routes_by_observed_producer_node():
    from aspec.ray.train_group import discard_actor_batches

    actors = [mock.MagicMock(), mock.MagicMock(), mock.MagicMock()]
    for actor, node in zip(actors, ["node-b", "node-a", "node-b"], strict=True):
        actor.get_status.remote.return_value = {"node_ip": node}
    group = SimpleNamespace(_actor_handlers=actors,
                            args=SimpleNamespace(mooncake_feature_affinity="node_local_required"))
    outputs = [SimpleNamespace(metadata={"feature_affinity": "node_local_required",
                                        "source_node_ip": node})
               for node in ["node-a", "node-b", "node-b", "node-a"]]
    batches = discard_actor_batches(group, outputs, resolve=lambda value: value)
    assert batches == [(actors[0], [outputs[1]]),
                       (actors[1], [outputs[0], outputs[3]]),
                       (actors[2], [outputs[2]])]
    outputs[0].metadata["source_node_ip"] = "unavailable-node"
    with pytest.raises(RuntimeError, match="matching source node"):
        discard_actor_batches(group, outputs, resolve=lambda value: value)
    for actor in actors:
        actor.discard_inference_outputs.remote.assert_not_called()


def test_discard_rejects_missing_locality_metadata():
    from aspec.ray.train_group import discard_actor_batches

    actor = mock.MagicMock()
    actor.get_status.remote.return_value = {"node_ip": "node-a"}
    group = SimpleNamespace(_actor_handlers=[actor],
                            args=SimpleNamespace(mooncake_feature_affinity="node_local_required"))
    with pytest.raises(RuntimeError, match="matching source node"):
        discard_actor_batches(group, [SimpleNamespace(metadata={})], resolve=lambda value: value)


def test_eval_cache_surfaces_failed_producer_before_waiting_for_idle_timeout():
    controller = mock.MagicMock()
    manager = mock.MagicMock()
    manager.start_eval_budget.remote.return_value = {"phase": "eval"}
    state = eval_utils.EvalSetupState(
        eval_interval=1, eval_enabled=True, eval_cache_loaded=False,
        eval_cache_path=None, best_eval_score=0.0, eval_dispatch_bs=1,
        eval_dataset_size=1, dp_size=1, initial_eval_submit_count=1,
    )
    producer = mock.Mock(side_effect=RuntimeError("Target engine lost"))
    with mock.patch("aspec.controller.eval.ray.get", side_effect=lambda value: value):
        with pytest.raises(RuntimeError, match="Target engine lost"):
            eval_utils.generate_eval_cache(controller, manager, mock.MagicMock(), state,
                                           check_producer=producer)
    controller.try_dispatch_eval_batch.remote.assert_not_called()


def test_blocked_eval_read_observes_producer_failure():
    from aspec.ray.train_group import RayTrainGroup

    group = RayTrainGroup.__new__(RayTrainGroup)
    group.args = SimpleNamespace(optimizer_step_timeout_seconds=30)
    actor = mock.MagicMock()
    ref = object()
    actor.cache_eval_samples.remote.return_value = ref
    group._actor_handlers = [actor]
    check = mock.Mock(side_effect=[None, RuntimeError("producer died during read")])
    with mock.patch("aspec.ray.train_group.ray.wait", return_value=([], [ref])), \
         mock.patch("aspec.ray.train_group.ray.get", return_value=[]):
        with pytest.raises(RuntimeError, match="producer died during read"):
            group.cache_eval_samples(1, check_producer=check)
    assert check.call_count == 2


def test_async_actor_initialization_honors_configured_timeout():
    from aspec.train_entry import _InitTimer

    timer = _InitTimer()
    timer.begin_async("actors")
    refs = [object()]
    with mock.patch("aspec.train_entry.ray.get", return_value=["ready"]) as get:
        assert timer.wait("actors", refs, timeout=17) == ["ready"]
    get.assert_called_once_with(refs, timeout=17)


def test_async_actor_initialization_surfaces_timeout():
    from aspec.train_entry import _InitTimer

    timer = _InitTimer()
    timer.begin_async("actors")
    with mock.patch("aspec.train_entry.ray.get", side_effect=TimeoutError("init stalled")):
        with pytest.raises(TimeoutError, match="init stalled"):
            timer.wait("actors", [object()], timeout=1)


def test_eval_cache_idle_budget_can_cover_cold_kernel_compilation():
    with mock.patch("aspec.controller.eval.time.monotonic", return_value=600):
        eval_utils._check_idle_timeout(0, 0, 32, timeout_seconds=1200)
        with pytest.raises(TimeoutError, match="idle_timeout=300.0s"):
            eval_utils._check_idle_timeout(0, 0, 32)


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_eval_cache_idle_budget_rejects_unbounded_or_invalid_values(value):
    from aspec.config.train_config import load_config

    with pytest.raises(ValueError, match="eval_cache_idle_timeout_seconds"):
        load_config("configs/recipes/quickstart_qwen3_8b.yaml",
                    cli_args=[f"dataset.eval_cache_idle_timeout_seconds={value}"],
                    save_snapshot=False)

def test_forced_cleanup_preserves_residency_if_process_fence_fails():
    group = SimpleNamespace(_actor_handlers=[mock.MagicMock()], _owner_statuses=[{"pid": 1}])
    controller = mock.MagicMock()
    controller.finalize_uncommitted_work.remote.return_value = {"discarded": 0}
    controller.get_work_lifecycle_status.remote.return_value = {
        "state": "incomplete", "resident_outputs": 1, "resident_bytes": 100}
    source = mock.MagicMock()
    with mock.patch("aspec.controller.loop.ray.get", side_effect=lambda ref, **kw: ref), \
         mock.patch("aspec.controller.loop.ray.kill") as kill, \
         mock.patch("aspec.controller.loop.wait_for_owner_exit", side_effect=TimeoutError("owner alive")):
        result = loop._safe_training_cleanup(SimpleNamespace(), None, None, [source],
                                            controller, group, force_kill_trainers=True)
    assert result["state"] == "completed_with_errors"
    assert result["work_lifecycle"]["resident_outputs"] == 1
    controller.finalize_fenced_work.remote.assert_not_called()
    assert mock.call(source, no_restart=True) not in kill.call_args_list



@pytest.mark.parametrize("snapshot", ["unavailable", None, {}, {"resident_outputs": 0},
                                     {"resident_outputs": 0, "resident_bytes": 100}])
def test_forced_cleanup_requires_owner_fence_when_lifecycle_is_unknown(snapshot):
    group = SimpleNamespace(_actor_handlers=[mock.MagicMock()], _owner_statuses=[{"pid": 1}])
    controller = mock.MagicMock()
    lifecycle_ref = controller.get_work_lifecycle_status.remote.return_value
    controller.finalize_uncommitted_work.remote.return_value = {"discarded": 0}
    source, master = mock.MagicMock(), mock.MagicMock()

    def get(ref, **kwargs):
        if ref is lifecycle_ref:
            if snapshot == "unavailable":
                raise RuntimeError("controller unavailable")
            return snapshot
        return ref

    with (
        mock.patch("aspec.controller.loop.ray.get", side_effect=get),
        mock.patch("aspec.controller.loop.ray.kill") as kill,
        mock.patch("aspec.controller.loop.wait_for_owner_exit",
                   side_effect=TimeoutError("reader still alive")) as fence,
    ):
        result = loop._safe_training_cleanup(
            SimpleNamespace(_mooncake_master_actor=master), None, None, [source],
            controller, group, force_kill_trainers=True)
    fence.assert_called_once()
    assert result["state"] == "completed_with_errors"
    controller.finalize_fenced_work.remote.assert_not_called()
    master.shutdown.remote.assert_not_called()
    assert mock.call(source, no_restart=True) not in kill.call_args_list


@pytest.mark.parametrize("fence_fails", [False, True])
@pytest.mark.parametrize("resident_outputs", [0, 1])
def test_failed_trainer_close_fences_readers_before_source_teardown(fence_fails, resident_outputs):
    trainer, source, master, controller = [mock.MagicMock() for _ in range(4)]
    group = SimpleNamespace(
        _actor_handlers=[trainer], _owner_statuses=[{"pid": 1}],
        close=mock.Mock(side_effect=RuntimeError("prefetch worker still reading")),
    )
    controller.finalize_uncommitted_work.remote.return_value = {"discarded": 0}
    controller.get_work_lifecycle_status.remote.return_value = {
        "state": "incomplete", "resident_outputs": resident_outputs,
        "resident_bytes": 100 * resident_outputs,
    }
    controller.finalize_fenced_work.remote.return_value = {
        "discarded": resident_outputs,
        "lifecycle": {"state": "validated", "resident_outputs": 0, "resident_bytes": 0},
    }
    events = []

    def fence(*_args, **_kwargs):
        events.append("fence")
        if fence_fails:
            raise TimeoutError("reader still alive")
        return {"state": "verified"}

    def kill(actor, **_kwargs):
        if actor is trainer:
            events.append("kill_reader")
        elif actor is source:
            events.append("kill_source")

    with (
        mock.patch.object(loop.ray, "get", side_effect=lambda ref, **_kw: ref),
        mock.patch.object(loop.ray, "kill", side_effect=kill),
        mock.patch.object(loop, "wait_for_owner_exit", side_effect=fence),
        mock.patch.object(loop, "_collect_feature_contract_observation", return_value={}),
    ):
        result = loop._safe_training_cleanup(
            SimpleNamespace(_mooncake_master_actor=master), None, None,
            [source], controller, group,
        )
    assert result["state"] == "completed_with_errors"
    assert result["mode"] == "forced_owner_fence"
    assert events == (["kill_reader", "fence"] if fence_fails
                      else ["kill_reader", "fence", "kill_source"])
    source.shutdown.remote.assert_not_called()
    if fence_fails:
        master.shutdown.remote.assert_not_called()
        controller.finalize_fenced_work.remote.assert_not_called()
    else:
        master.shutdown.remote.assert_called_once_with()
        controller.finalize_fenced_work.remote.assert_called_once_with(owners_fenced=True)


def test_train_group_close_is_bounded_by_cleanup_budget():
    from aspec.ray.train_group import RayTrainGroup

    group = RayTrainGroup.__new__(RayTrainGroup)
    group.args = SimpleNamespace(inference_shutdown_timeout_seconds=17)
    actor = mock.MagicMock()
    group._actor_handlers = [actor]
    with mock.patch("aspec.ray.train_group.ray.get") as get:
        group.close()
    get.assert_called_once_with([actor.close.remote.return_value], timeout=17.0)


def test_pipelined_dispatch_uses_remaining_optimizer_deadline():
    import threading

    controller = mock.MagicMock()
    with (
        mock.patch.object(loop.time, "monotonic", return_value=11.0),
        mock.patch.object(loop.ray, "get", return_value=True) as get,
    ):
        count, _ = loop._dispatch_ahead(
            controller, optimizer_step=1, first_microstep=0, target_microsteps=1,
            stop_event=threading.Event(), deadline=13.0, timeout_seconds=10.0,
        )
    assert count == 1
    get.assert_called_once_with(controller.try_dispatch_batch.remote.return_value, timeout=2.0)


def test_pipelined_dispatch_rpc_timeout_reaches_owner_fencing():
    import threading

    with (
        mock.patch.object(loop.time, "monotonic", return_value=11.0),
        mock.patch.object(loop.ray, "get", side_effect=loop.ray.exceptions.GetTimeoutError()),
        pytest.raises(loop.TrainerStepTimeoutError, match="pipelined dispatch"),
    ):
        loop._dispatch_ahead(
            mock.MagicMock(), optimizer_step=1, first_microstep=0, target_microsteps=1,
            stop_event=threading.Event(), deadline=13.0, timeout_seconds=10.0,
        )


def test_pipelined_dispatch_status_rpc_shares_the_deadline():
    import threading

    controller = mock.MagicMock()
    statuses = iter([False] * 100 + [loop.ray.exceptions.GetTimeoutError()])
    calls = []

    def get(_ref, *, timeout):
        calls.append(timeout)
        result = next(statuses)
        if isinstance(result, BaseException):
            raise result
        return result

    with (
        mock.patch.object(loop.time, "monotonic", return_value=11.0),
        mock.patch.object(loop.ray, "get", side_effect=get),
        mock.patch.object(threading.Event, "wait"),
        pytest.raises(loop.TrainerStepTimeoutError, match="pipelined dispatch"),
    ):
        loop._dispatch_ahead(
            controller, optimizer_step=1, first_microstep=0, target_microsteps=1,
            stop_event=threading.Event(), deadline=13.0, timeout_seconds=10.0,
        )
    controller.get_full_status.remote.assert_called_once_with()
    assert calls == [2.0] * 101


@pytest.mark.parametrize("reply_kind", ["dispatch", "source_status", "commit"])
def test_training_step_rpcs_have_a_bounded_wait(reply_kind):
    observed = []

    def get(value, **kwargs):
        selected = (
            value is True if reply_kind == "dispatch"
            else isinstance(value, dict) and (
                "sample_budget_drained" in value if reply_kind == "source_status"
                else "committed_work" in value
            )
        )
        if selected:
            assert 0 < kwargs.get("timeout", 0) <= 1800
            observed.append(value)
        return value

    _run_mock_training_loop(
        prefetch_depth=0, num_steps=1, steps_per_epoch=1,
        source_work_budget=1, packing_enabled=False, rpc_get=get,
    )
    assert observed


def test_producer_profile_rpc_obeys_profile_deadline():
    args = SimpleNamespace(
        topology_producer_profile_samples=1,
        topology_producer_profile_warmup_samples=0,
        optimizer_step_timeout_seconds=10,
    )

    def get(_ref, **kwargs):
        assert 0 < kwargs.get("timeout", 0) <= 10
        raise loop.ray.exceptions.GetTimeoutError()

    with (
        mock.patch.object(loop.ray, "get", side_effect=get),
        pytest.raises(loop.TrainerStepTimeoutError, match="producer_profile"),
    ):
        loop._run_producer_service_profile(
            args, mock.MagicMock(), mock.MagicMock(), mock.MagicMock()
        )


def test_terminal_producer_observation_has_cleanup_timeout():
    args = SimpleNamespace(
        aspec_feature_contract={"contract_id": "test"},
        inference_shutdown_timeout_seconds=17,
    )

    def get(_ref, **kwargs):
        assert kwargs.get("timeout") == 17
        raise loop.ray.exceptions.GetTimeoutError()

    with (
        mock.patch.object(loop.ray, "get", side_effect=get),
        pytest.raises(loop.ray.exceptions.GetTimeoutError),
    ):
        loop._collect_feature_contract_observation(
            args, mock.MagicMock(), [mock.MagicMock()]
        )


def test_terminal_consumer_observation_has_cleanup_timeout():
    from aspec.ray.train_group import RayTrainGroup

    group = RayTrainGroup.__new__(RayTrainGroup)
    group.args = SimpleNamespace(inference_shutdown_timeout_seconds=17)
    group._actor_handlers = [mock.MagicMock()]
    with mock.patch("aspec.ray.train_group.ray.get") as get:
        group.get_feature_contract_status()
    assert get.call_args.kwargs.get("timeout") == 17


def test_closed_external_reader_allows_forced_source_shutdown(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from aspec.controller import loop

    manager, controller, engine = Mock(), Mock(), Mock()
    future = object()
    controller.finalize_uncommitted_work.remote.return_value = {"discarded": 0}
    controller.get_work_lifecycle_status.remote.return_value = {
        "resident_outputs": 1, "resident_bytes": 8,
    }
    controller.finalize_fenced_work.remote.return_value = {
        "discarded": 1, "lifecycle": {"state": "validated", "resident_outputs": 0, "resident_bytes": 0},
    }
    def get(ref, **kwargs):
        if ref is future:
            raise TimeoutError("producer blocked")
        return ref
    monkeypatch.setattr(loop.ray, "get", get)
    monkeypatch.setattr(loop.ray, "kill", Mock())
    monkeypatch.setattr(loop.ray, "cancel", Mock())
    probe = Mock(side_effect=AssertionError("reader is already closed"))
    monkeypatch.setattr(loop, "wait_for_owner_exit", probe)
    fence = {"state": "verified", "scope": "closed external reader"}
    result = loop._safe_training_cleanup(
        SimpleNamespace(), manager, future, [engine], controller=controller,
        closed_reader_fence=fence,
    )
    probe.assert_not_called()
    assert result["read_owner_fence"] == fence
    assert any(item["component"] == "inference_engine" and item["state"] == "force_killed" for item in result["operations"])
    controller.finalize_fenced_work.remote.assert_called_once_with(owners_fenced=True)


def test_training_reporting_and_reload_rpcs_have_bounded_waits():
    calls = []

    def get(value, **kwargs):
        assert 0 < kwargs.get("timeout", 0) <= 1800
        calls.append(value)
        return value

    _run_mock_training_loop(
        prefetch_depth=0, num_steps=2, steps_per_epoch=1,
        train_metrics=[{"train/avg_loss": 1.0}, {"train/avg_loss": 0.9}],
        rpc_get=get,
    )
    assert len(calls) > 8


def test_final_training_metrics_have_bounded_wait():
    from aspec.ray.train_group import RayTrainGroup

    group = RayTrainGroup.__new__(RayTrainGroup)
    group.args = SimpleNamespace(optimizer_step_timeout_seconds=19)
    group._actor_handlers = [mock.MagicMock()]
    with mock.patch.object(loop.ray, "get", return_value=[]) as get:
        assert group.flush_pending_train_metrics() == []
    assert get.call_args.kwargs["timeout"] == 19


@pytest.mark.parametrize("parent_budget", [None, 0.15])
def test_profile_sink_placement_and_discard_share_one_deadline(monkeypatch, parent_budget):
    actors = [mock.MagicMock(), mock.MagicMock()]
    group = SimpleNamespace(
        _actor_handlers=actors, args=SimpleNamespace(mooncake_feature_affinity="node_local_required"),
    )
    outputs = [SimpleNamespace(metadata={
        "feature_affinity": "node_local_required", "source_node_ip": "node",
    }) for _ in actors]
    for index, actor in enumerate(actors):
        actor.get_status.remote.return_value = f"status-{index}"
        actor.discard_inference_outputs.remote.return_value = 1
    now = [10.0]
    monkeypatch.setattr(loop.time, "monotonic", lambda: now[0])
    budgets = []

    def get(ref, **kwargs):
        budgets.append(kwargs.get("timeout"))
        if isinstance(ref, str):
            now[0] += 0.06
            return {"node_ip": "node"}
        return ref

    monkeypatch.setattr(loop.ray, "get", get)
    deadline = None if parent_budget is None else now[0] + parent_budget
    assert loop._discard_profile_outputs(group, outputs, timeout_seconds=0.25, deadline=deadline) == 2
    budget = 0.25 if parent_budget is None else parent_budget
    assert budgets == pytest.approx([budget, budget - 0.06, budget - 0.12])


def test_profile_sink_placement_timeout_stops_before_discard(monkeypatch):
    actor = mock.MagicMock()
    group = SimpleNamespace(
        _actor_handlers=[actor], args=SimpleNamespace(mooncake_feature_affinity="node_local_required"),
    )
    output = SimpleNamespace(metadata={
        "feature_affinity": "node_local_required", "source_node_ip": "node",
    })

    def stalled(ref, **kwargs):
        assert 0 < kwargs.get("timeout", 0) <= 0.25
        raise loop.ray.exceptions.GetTimeoutError()

    monkeypatch.setattr(loop.ray, "get", stalled)
    with pytest.raises(loop.TrainerStepTimeoutError, match="producer_profile_sink"):
        loop._discard_profile_outputs(group, [output], timeout_seconds=0.25)
    actor.discard_inference_outputs.remote.assert_not_called()


def test_profile_sink_does_not_dispatch_after_placement_exhausts_budget(monkeypatch):
    actor = mock.MagicMock()
    group = SimpleNamespace(
        _actor_handlers=[actor], args=SimpleNamespace(mooncake_feature_affinity="node_local_required"),
    )
    output = SimpleNamespace(metadata={
        "feature_affinity": "node_local_required", "source_node_ip": "node",
    })
    now = [10.0]
    monkeypatch.setattr(loop.time, "monotonic", lambda: now[0])

    def get(ref, **kwargs):
        now[0] += 0.3
        return {"node_ip": "node"}

    monkeypatch.setattr(loop.ray, "get", get)
    with pytest.raises(loop.TrainerStepTimeoutError, match="producer_profile_sink"):
        loop._discard_profile_outputs(group, [output], timeout_seconds=0.25)
    actor.discard_inference_outputs.remote.assert_not_called()


def test_interrupted_cleanup_cannot_allow_blanket_shutdown(monkeypatch):
    from aspec.ray.owner_fence import SourceTeardownBlockedError

    group = SimpleNamespace(_owner_statuses=[{"node_id": "node", "node_ip": "ip", "pid": 123}])
    interrupted = KeyboardInterrupt("cleanup interrupted")
    monkeypatch.setattr(loop, "training_loop", mock.Mock(return_value={}))
    monkeypatch.setattr(loop, "_safe_training_cleanup", mock.Mock(side_effect=interrupted))
    observer = mock.Mock()
    with pytest.raises(SourceTeardownBlockedError) as captured:
        loop.run_training_loop(
            SimpleNamespace(), object(), mock.MagicMock(), group, cleanup_observer=observer,
        )
    assert captured.value.__cause__ is interrupted
    assert captured.value.owner_statuses == group._owner_statuses
    assert observer.call_args.args[0]["source_teardown"] == "blocked"


def test_profile_sink_stops_dispatch_when_budget_expires_between_sinks(monkeypatch):
    actors = [mock.MagicMock(), mock.MagicMock()]
    group = SimpleNamespace(
        _actor_handlers=actors, args=SimpleNamespace(mooncake_feature_affinity="any"),
    )
    outputs = [SimpleNamespace(metadata={}), SimpleNamespace(metadata={})]
    now = [10.0]
    monkeypatch.setattr(loop.time, "monotonic", lambda: now[0])

    def submit_first(chunk):
        now[0] += 0.3
        return "pending-first-discard"

    actors[0].discard_inference_outputs.remote.side_effect = submit_first
    get = mock.Mock()
    monkeypatch.setattr(loop.ray, "get", get)
    with pytest.raises(loop.TrainerStepTimeoutError, match="producer_profile_sink"):
        loop._discard_profile_outputs(group, outputs, timeout_seconds=0.25)
    actors[0].discard_inference_outputs.remote.assert_called_once()
    actors[1].discard_inference_outputs.remote.assert_not_called()
    get.assert_not_called()


@pytest.mark.parametrize("engine_type", ["sgl", "hf"])
@pytest.mark.parametrize("fence_fails", [False, True])
@pytest.mark.parametrize("resident_outputs", [0, 1])
def test_forced_target_cleanup_uses_source_termination_only_after_reader_fence(
    monkeypatch, engine_type, fence_fails, resident_outputs,
):
    trainer, source, controller = mock.MagicMock(), mock.MagicMock(), mock.MagicMock()
    group = SimpleNamespace(_actor_handlers=[trainer], _owner_statuses=[{"pid": 1}])
    controller.finalize_uncommitted_work.remote.return_value = {"discarded": 0}
    controller.get_work_lifecycle_status.remote.return_value = {
        "resident_outputs": resident_outputs, "resident_bytes": 8 * resident_outputs,
    }
    controller.finalize_fenced_work.remote.return_value = {
        "discarded": 1, "lifecycle": {"state": "validated", "resident_outputs": 0, "resident_bytes": 0},
    }
    events = []

    def fence(*_args, **_kwargs):
        events.append("reader_fence")
        if fence_fails:
            raise TimeoutError("reader alive")
        return {"state": "verified"}

    source.shutdown.remote.side_effect = lambda **kwargs: events.append(("shutdown", kwargs))
    monkeypatch.setattr(loop, "wait_for_owner_exit", fence)
    monkeypatch.setattr(loop.ray, "get", lambda ref, **_kwargs: ref)
    monkeypatch.setattr(loop.ray, "kill", mock.Mock())
    result = loop._safe_training_cleanup(
        SimpleNamespace(inference_engine_type=engine_type, target_service={"mode": "external"}),
        None, None, [source], controller, group, force_kill_trainers=True,
    )
    if engine_type == "hf" and not resident_outputs:
        assert not events and result["state"] == "completed"
    elif fence_fails:
        source.shutdown.remote.assert_not_called()
        assert result["source_teardown"] == "blocked"
    elif engine_type == "sgl":
        assert events == ["reader_fence", ("shutdown", {"force": True})]
        assert result["state"] == "completed"
    else:
        source.shutdown.remote.assert_not_called()
        assert result["state"] == "completed"


def test_lost_source_domain_is_recorded_without_claiming_source_process_exit(monkeypatch):
    trainer, source, controller = mock.MagicMock(), mock.MagicMock(), mock.MagicMock()
    group = SimpleNamespace(_actor_handlers=[trainer], _owner_statuses=[{"pid": 1}])
    controller.finalize_uncommitted_work.remote.return_value = {"discarded": 0}
    controller.get_work_lifecycle_status.remote.return_value = {"resident_outputs": 1, "resident_bytes": 8}
    controller.finalize_fenced_work.remote.return_value = {
        "discarded": 1, "lifecycle": {"state": "validated", "resident_outputs": 0, "resident_bytes": 0},
    }
    source.shutdown.remote.return_value = {
        "state": "node_local_domain_lost", "node_id": "lost-node", "physical_memory_unobserved": True,
    }
    monkeypatch.setattr(loop, "wait_for_owner_exit", lambda *_args, **_kwargs: {"state": "verified"})
    monkeypatch.setattr(loop.ray, "get", lambda ref, **_kwargs: ref)
    monkeypatch.setattr(loop.ray, "kill", mock.Mock())
    result = loop._safe_training_cleanup(
        SimpleNamespace(inference_engine_type="sgl", target_service={"mode": "external"}),
        None, None, [source], controller, group, force_kill_trainers=True,
    )
    assert result["state"] == "completed"
    operation = next(x for x in result["operations"] if x["component"] == "inference_engine")
    assert operation == {"component": "inference_engine", "state": "node_local_domain_lost",
                         "node_id": "lost-node", "physical_memory_unobserved": True}
