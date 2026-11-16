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

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from aspec import multi_draft, train_entry


def test_graceful_close_timeout_still_fences_before_withdrawing_reader(monkeypatch):
    events = []
    group, hub = Mock(), Mock()
    group._actor_handlers = [Mock()]
    monkeypatch.setattr(multi_draft.ray, "get", Mock(side_effect=[TimeoutError("close"), None]))
    monkeypatch.setattr(multi_draft, "_fence_group", lambda *_: events.append("fenced"))
    hub.retire.remote.side_effect = lambda *args, **kwargs: events.append("withdrawn")
    assert (
        multi_draft._finish_experiment(group, [], hub, "a", graceful=True) == "TimeoutError: close"
    )
    assert events == ["fenced", "withdrawn"]
    assert multi_draft.ray.get.call_args_list[0].kwargs == {"timeout": 10}


def test_failed_fence_does_not_withdraw_reader(monkeypatch):
    group, hub = Mock(), Mock()
    monkeypatch.setattr(multi_draft, "_fence_group", Mock(side_effect=TimeoutError("owner alive")))
    with pytest.raises(TimeoutError, match="owner alive"):
        multi_draft._finish_experiment(group, [], hub, "a", graceful=False)
    hub.retire.remote.assert_not_called()


def test_fence_probe_uses_live_node_not_stale_record_for_same_ip(monkeypatch):
    group, probe_actor = Mock(), Mock()
    group._actor_handlers = [Mock()]
    monkeypatch.setattr(multi_draft, "_OwnerFenceProbe", probe_actor)
    monkeypatch.setattr(
        multi_draft.ray,
        "nodes",
        lambda: [
            {"NodeID": "0" * 56, "NodeManagerAddress": "host", "Alive": False},
            {"NodeID": "1" * 56, "NodeManagerAddress": "host", "Alive": True},
        ],
    )
    monkeypatch.setattr(multi_draft.ray, "get", Mock(return_value=True))
    monkeypatch.setattr(multi_draft.ray, "kill", Mock())
    multi_draft._fence_group(group, [{"node_ip": "host", "pid": 1}])
    placement = probe_actor.options.call_args.kwargs["scheduling_strategy"]
    assert placement.node_id == "1" * 56


def _session(
    monkeypatch,
    tmp_path,
    *,
    source_error=False,
    fence_error=False,
    initialization_error=False,
    no_new_work=(),
):
    """Exercise the real session orchestrator without GPU jobs or sleeping."""
    args = SimpleNamespace(output_dir=str(tmp_path), inference_num_gpus=1,
                           inference_engine_type="sgl",
                           aspec_context_contract={},
                           aspec_feature_contract={"contract_id": "union"})
    children = {
        name: SimpleNamespace(
            output_dir=str(tmp_path / name),
            world_size=1,
            per_dp_rank_batch_size=1,
            max_concurrent_batches=1,
            aspec_feature_contract={"contract_id": name},
        )
        for name in ("a", "b")
    }
    plans = {
        name: SimpleNamespace(to_runtime_args=lambda child=child: child,
                              workload=SimpleNamespace(features=child.aspec_feature_contract))
        for name, child in children.items()
    }
    plan = SimpleNamespace(plan_id="plan", plans=plans, configs={"a": {}, "b": {}})
    args.max_concurrent_batches = 1
    hub, manager, engine = (Mock() for _ in range(3))
    hub.load_dataset.remote.return_value = 4
    producer_future = object()
    manager.run.remote.return_value = producer_future
    manager.get_status.remote.return_value = {"inflight_samples": 0}
    manager.flush_metrics.remote.return_value = {}
    hub.finish_production.remote.return_value = {
        "outstanding_observations": 0,
        "pending_production": 0,
    }
    hub.cancellation_reason.remote.return_value = None
    hub.invoke.remote.return_value = {}
    recorders = {name: Mock() for name in children}
    groups = {name: Mock() for name in children}
    for group in groups.values():
        group._actor_handlers = [Mock()]
        group.async_init.return_value = []
        group.get_status.return_value = [{"node_ip": "node", "pid": 1}]
        group.get_feature_contract_status.return_value = []
    init_failure = object()
    if initialization_error:
        groups["b"].async_init.return_value = [init_failure]

    def get(ref, **kwargs):
        if ref is init_failure:
            raise RuntimeError("Draft B initialization failed")
        if ref is producer_future and source_error:
            raise RuntimeError("shared source failed later")
        return [get(item) for item in ref] if isinstance(ref, list) else ref

    monkeypatch.setattr(multi_draft.ray, "get", get)
    monkeypatch.setattr(multi_draft.ray, "kill", Mock())
    monkeypatch.setattr(multi_draft.ray, "nodes", lambda: [{"Alive": True}])
    monkeypatch.setattr(multi_draft.ray, "cluster_resources", lambda: {"GPU": 3})
    monkeypatch.setattr(multi_draft.ray.util, "remove_placement_group", Mock())
    for target, result in (("SharedSourceController", hub), ("AsyncInferenceManager", manager)):
        actor = Mock()
        actor.options.return_value.remote.return_value = result
        monkeypatch.setattr(multi_draft, target, actor)
    monkeypatch.setattr(train_entry, "_prepare_run_directory", lambda *_: None)
    monkeypatch.setattr(train_entry, "_get_draft_model_config", lambda *_: {})
    monkeypatch.setattr(train_entry, "_prepare_vocab_mapping", lambda *_: None)
    monkeypatch.setattr(train_entry, "_validate_and_configure_dflash", lambda *_: None)
    for target in (
        "_save_config_snapshot",
        "save_execution_plan",
        "launch_mooncake_master",
        "auto_calculate_training_steps",
    ):
        monkeypatch.setattr(multi_draft, target, lambda *_: None)
    monkeypatch.setattr(multi_draft, "build_mooncake_config", lambda *_: SimpleNamespace(gpu_buffer_size=1024))
    pg = SimpleNamespace(id="owned-pg")
    monkeypatch.setattr(
        multi_draft,
        "create_placement_groups",
        lambda *_: {"training": (pg, [0, 1], [0, 1]), "inference": None},
    )
    monkeypatch.setattr(multi_draft, "require_single_gpu_node", lambda *_: "node")
    monkeypatch.setattr(
        multi_draft,
        "allocate_train_group",
        lambda child, *_: groups[next(name for name in children if child is children[name])],
    )
    monkeypatch.setattr(multi_draft, "prepare_inference_engines", lambda *_: ([engine], []))
    monkeypatch.setattr(
        multi_draft,
        "RunEvidenceRecorder",
        lambda child_plan, *_args, **_kwargs: recorders[
            next(name for name in plans if child_plan is plans[name])
        ],
    )
    monkeypatch.setattr(multi_draft, "_collect_feature_contract_observation", Mock(return_value={}))
    monkeypatch.setattr(multi_draft, "experiment_contract", lambda *_: {})
    monkeypatch.setattr(
        multi_draft,
        "training_loop",
        lambda child, *_args, **_kwargs: {
            "optimizer_steps_observed": 0 if child.aspec_experiment_name in no_new_work else 1
        },
    )
    fence = Mock(side_effect=TimeoutError("owner alive") if fence_error else None)
    monkeypatch.setattr(multi_draft, "_fence_group", fence)
    return args, plan, recorders, engine


def test_external_target_cleanup_persists_terminal_session_without_an_inference_pg(monkeypatch, tmp_path):
    args, plan, recorders, engine = _session(monkeypatch, tmp_path)
    multi_draft.run_multi_draft(args, plan)
    evidence = json.loads((tmp_path / "evidence.json").read_text())
    assert evidence["state"] == "completed"
    assert evidence["read_owners_fenced"] and not evidence["cleanup_errors"]
    engine.shutdown.remote.assert_called_once_with()
    multi_draft.ray.util.remove_placement_group.assert_called_once()
    assert multi_draft.ray.util.remove_placement_group.call_args.args[0].id == "owned-pg"
    for recorder in recorders.values():
        recorder.complete.assert_called()
        recorder.fail.assert_not_called()


def test_completed_draft_is_not_relabelled_failed_by_later_shared_fault(monkeypatch, tmp_path):
    args, plan, recorders, engine = _session(monkeypatch, tmp_path, source_error=True)
    with pytest.raises(RuntimeError, match="shared source failed later"):
        multi_draft.run_multi_draft(args, plan)
    for recorder in recorders.values():
        recorder.complete.assert_called_once()
        recorder.fail.assert_not_called()
    engine.shutdown.remote.assert_called_once_with(force=True)
    multi_draft.ray.util.remove_placement_group.assert_called_once()
    evidence = json.loads((tmp_path / "evidence.json").read_text())
    assert evidence["state"] == "failed"
    assert evidence["read_owners_fenced"]
    for name in ("a", "b"):
        assert json.loads((tmp_path / name / "tracking.json").read_text())["exit_code"] == 0


@pytest.mark.parametrize("error", [RuntimeError("shared eval failed"), KeyboardInterrupt("session interrupted")])
def test_prepared_tracking_closes_when_session_stops_before_training(monkeypatch, tmp_path, error):
    args, plan, recorders, _ = _session(monkeypatch, tmp_path)
    for child_plan in plan.plans.values():
        child_plan.to_runtime_args().eval_data_path = "eval.jsonl"
    monkeypatch.setattr(multi_draft, "prepare_shared_eval", Mock(side_effect=error))
    with pytest.raises(type(error), match=str(error)):
        multi_draft.run_multi_draft(args, plan)
    for name in ("a", "b"):
        tracking = json.loads((tmp_path / name / "tracking.json").read_text())
        assert tracking["state"] == "closed" and tracking["exit_code"] == 1
        assert tracking["metric_calls"] == 0
        recorders[name].fail.assert_called_once()
    evidence = json.loads((tmp_path / "evidence.json").read_text())
    assert evidence["read_owners_fenced"] and evidence["cleanup_errors"] == []


def test_tracking_initialization_failure_isolated_from_healthy_draft(monkeypatch, tmp_path):
    args, plan, recorders, _ = _session(monkeypatch, tmp_path)
    original = multi_draft.ExperimentTracking

    def tracking(child):
        if child.aspec_experiment_name == "b":
            raise RuntimeError("Draft B tracking initialization failed")
        return original(child)

    monkeypatch.setattr(multi_draft, "ExperimentTracking", tracking)
    with pytest.raises(RuntimeError, match="completed_with_failed_experiments"):
        multi_draft.run_multi_draft(args, plan)
    recorders["a"].fail.assert_not_called()
    recorders["b"].fail.assert_called_once()
    assert json.loads((tmp_path / "a/tracking.json").read_text())["exit_code"] == 0
    evidence = json.loads((tmp_path / "evidence.json").read_text())
    assert evidence["experiments"]["a"]["state"] == "completed"
    assert evidence["experiments"]["b"]["phase"] == "initialization"
    assert evidence["cleanup_errors"] == []


def test_driver_failure_finishes_only_failed_draft_with_nonzero_code(monkeypatch, tmp_path):
    args, plan, recorders, _ = _session(monkeypatch, tmp_path)

    def train(child, *args, **kwargs):
        if child.aspec_experiment_name == "b":
            raise OSError("Draft B log write failed")
        return {"optimizer_steps_observed": 1}

    monkeypatch.setattr(multi_draft, "training_loop", train)
    with pytest.raises(RuntimeError, match="completed_with_failed_experiments"):
        multi_draft.run_multi_draft(args, plan)
    for name, exit_code in (("a", 0), ("b", 1)):
        tracking = json.loads((tmp_path / name / "tracking.json").read_text())
        assert tracking["state"] == "closed" and tracking["exit_code"] == exit_code
    recorders["a"].fail.assert_not_called()
    recorders["b"].fail.assert_called_once()


def test_tracking_close_failure_is_not_misclassified_as_unsafe_source_fence(monkeypatch, tmp_path):
    args, plan, recorders, _ = _session(monkeypatch, tmp_path)
    original = multi_draft.ExperimentTracking

    def tracking(child):
        result = original(child)
        if child.aspec_experiment_name == "b":
            close = result.close

            def fail_close(**kwargs):
                if result._closed:
                    return
                close(**kwargs)
                raise OSError("SDK finish failed")

            result.close = fail_close
        return result

    monkeypatch.setattr(multi_draft, "ExperimentTracking", tracking)
    with pytest.raises(RuntimeError, match="completed_with_failed_experiments"):
        multi_draft.run_multi_draft(args, plan)
    recorders["a"].fail.assert_not_called()
    recorders["b"].fail.assert_called_once()
    evidence = json.loads((tmp_path / "evidence.json").read_text())
    assert evidence["experiments"]["a"]["state"] == "completed"
    assert evidence["experiments"]["b"]["state"] == "failed"
    assert evidence["cleanup_errors"] == []


def test_unsafe_fence_withholds_source_teardown_and_placement_release(monkeypatch, tmp_path):
    args, plan, recorders, engine = _session(monkeypatch, tmp_path, fence_error=True)
    with pytest.raises(TimeoutError, match="owner alive"):
        multi_draft.run_multi_draft(args, plan)
    engine.shutdown.remote.assert_not_called()
    multi_draft.ray.util.remove_placement_group.assert_not_called()
    evidence = json.loads((tmp_path / "evidence.json").read_text())
    assert evidence["state"] == "failed"
    assert not evidence["read_owners_fenced"]
    assert any("teardown was withheld" in error for error in evidence["cleanup_errors"])


def test_independent_draft_initialization_failure_does_not_stop_healthy_draft(
    monkeypatch, tmp_path
):
    args, plan, recorders, engine = _session(monkeypatch, tmp_path, initialization_error=True)
    with pytest.raises(RuntimeError, match="completed_with_failed_experiments"):
        multi_draft.run_multi_draft(args, plan)
    recorders["a"].fail.assert_not_called()
    assert recorders["a"].complete.call_count == 2
    recorders["b"].fail.assert_called_once()
    recorders["b"].complete.assert_not_called()
    evidence = json.loads((tmp_path / "evidence.json").read_text())
    assert evidence["experiments"]["a"]["state"] == "completed"
    assert evidence["experiments"]["b"]["phase"] == "initialization"
    assert evidence["read_owners_fenced"]
    assert evidence["cleanup_errors"] == []


def test_failed_initialization_can_release_placement_before_any_read_subscription(
    monkeypatch, tmp_path
):
    args, plan, recorders, engine = _session(
        monkeypatch, tmp_path, initialization_error=True, fence_error=True
    )
    with pytest.raises(TimeoutError, match="owner alive"):
        multi_draft.run_multi_draft(args, plan)
    engine.shutdown.remote.assert_called_once()
    multi_draft.ray.util.remove_placement_group.assert_called_once()


@pytest.mark.parametrize("idle", [("a",), ("a", "b")])
def test_finished_checkpoint_does_not_fail_other_drafts_for_having_no_new_reads(
    monkeypatch, tmp_path, idle
):
    args, plan, recorders, engine = _session(monkeypatch, tmp_path, no_new_work=idle)
    session = multi_draft.run_multi_draft(args, plan)
    assert session["state"] == "completed"
    for name in idle:
        assert session["experiments"][name]["no_new_work"]
        recorders[name].fail.assert_not_called()
    if len(idle) == 1:
        multi_draft._collect_feature_contract_observation.assert_called_once()
        assert multi_draft._collect_feature_contract_observation.call_args.args[0].world_size == 1
    else:
        multi_draft._collect_feature_contract_observation.assert_not_called()


def test_cp_lifecycle_harness_keeps_picklable_parent_and_resume_list(monkeypatch, tmp_path):
    import sys
    from pathlib import Path

    from tools.validation import recipe_gpu_smoke, shared_cp_lifecycle

    class Captured(Exception):
        pass

    parent = recipe_gpu_smoke.AuditedTrainerActor
    config = Path(__file__).parents[2] / "configs/examples/qwen3_8b_multi_algorithm_feature_views.yaml"
    monkeypatch.setattr(sys, "argv", [
        "lifecycle", "--config", str(config), "--output", str(tmp_path / "new"),
        "--resume-from", str(tmp_path / "old"), "--gpus", "8",
    ])

    def capture(*, trainer_class):
        assert recipe_gpu_smoke.AuditedTrainerActor is parent
        assert trainer_class.__mro__[1] is parent
        from ray import cloudpickle

        copy = cloudpickle.loads(cloudpickle.dumps(trainer_class))
        assert copy.__mro__[1] is parent
        override = next(item for item in sys.argv if item.startswith("experiments="))
        experiments = json.loads(override.split("=", 1)[1])
        assert [item["name"] for item in experiments] == ["dspark_cp1", "eagle3_cp2", "dflash_cp4"]
        for item in experiments:
            assert item["training"]["load_path"] == str(
                tmp_path / "old" / item["name"] / "checkpoints" / "iter_0000017"
            )
        assert experiments[1]["training"]["draft_context_parallel_size"] == 2
        assert experiments[2]["training"]["draft_context_parallel_size"] == 4
        raise Captured

    monkeypatch.setattr(recipe_gpu_smoke, "main", capture)
    with pytest.raises(Captured):
        shared_cp_lifecycle.main()

def test_failed_sibling_keeps_union_producer_and_independent_consumer_contract(monkeypatch, tmp_path):
    args, plan, recorders, _engine = _session(monkeypatch, tmp_path, initialization_error=True)

    def observe(observed_args, _group, _engines):
        assert observed_args.aspec_feature_contract == {"contract_id": "union"}
        assert observed_args.aspec_shared_consumer_contracts == {"a": "a", "b": "b"}
        assert observed_args.world_size == 1
        assert observed_args.aspec_shared_reader == "a"
        return {}

    monkeypatch.setattr(multi_draft, "_collect_feature_contract_observation", observe)
    with pytest.raises(RuntimeError, match="completed_with_failed_experiments"):
        multi_draft.run_multi_draft(args, plan)
    evidence = json.loads((tmp_path / "evidence.json").read_text())
    assert evidence["state"] == "completed_with_failed_experiments"
    assert evidence["cleanup_errors"] == []
    recorders["a"].fail.assert_not_called()
