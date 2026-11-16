# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Isolation regressions; real Ray/venv and GPU checks run on Linux separately."""
import importlib
import importlib.metadata
import subprocess
import sys
from argparse import Namespace
from unittest.mock import patch

import pytest


@pytest.fixture(params=["wheel", "source"])
def target_environment(monkeypatch, tmp_path, request):
    version = importlib.metadata.version

    def installed(name):
        if name == "ling-aspec-target":
            if request.param == "source":
                raise importlib.metadata.PackageNotFoundError(name)
            return "1.0"
        return version(name)

    monkeypatch.setattr(importlib.metadata, "version", installed)
    prefix = tmp_path / "target-venv"
    prefix.mkdir()
    if request.param == "source":
        (prefix / ".aspec-role").write_text("target\n")
    monkeypatch.setattr(sys, "prefix", str(prefix))


@pytest.mark.parametrize("arguments", [
    ["train", "--config", "unused.yaml"],
    ["--config", "unused.yaml"],
    ["recipes", "train", "quickstart", "--output", "unused"],
    ["recipes", "accept", "quickstart", "--output", "unused"],
    ["quickstart", "qwen3-8b", "--output", "unused"],
])
def test_target_environment_rejects_training_launchers(target_environment, monkeypatch, tmp_path, arguments):
    from aspec import cli, train_entry
    from aspec.commands import quickstart, recipes

    def dispatch(argv, *_):
        monkeypatch.setattr(sys, "argv", list(map(str, argv[2:])))
        importlib.import_module(argv[2]).main()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "execute", dispatch)
    monkeypatch.setattr(recipes, "execute", dispatch)
    monkeypatch.setattr(quickstart, "execute", dispatch)
    monkeypatch.setattr(train_entry, "load_config", lambda **_: pytest.fail("Target must reject before loading training config"))
    monkeypatch.setattr(sys, "argv", ["aspec", *arguments])
    with pytest.raises(SystemExit, match="Draft venv"):
        cli.main()
    assert not (tmp_path / "unused").exists()


def test_python_training_entry_rejects_target_before_config_or_runtime_start(target_environment, monkeypatch):
    from aspec import train_entry

    monkeypatch.setattr(sys, "argv", ["aspec.train_entry", "--config", "unused.yaml"])
    monkeypatch.setattr(train_entry, "load_config", lambda **_: pytest.fail("must reject before config loading"))
    with pytest.raises(SystemExit, match="Draft venv"):
        train_entry.main()


def test_acceptance_api_rejects_target_before_gpu_checks_or_output_creation(target_environment, tmp_path):
    from aspec.commands.recipes import run_acceptance

    output = tmp_path / "run"
    with pytest.raises(SystemExit, match="Draft venv"):
        run_acceptance("quickstart", {}, output, 8, inputs={})
    assert not output.exists()


@pytest.mark.parametrize("arguments", [
    ["recipes", "list"],
    ["recipes", "plan", "quickstart"],
    ["quickstart", "qwen3-8b", "--plan", "--output", "unused"],
    ["train", "--config", "profile.yaml", "--dry-run"],
    ["train", "--config", "profile.yaml", "--print-config-only"],
    ["--report", "unused", "--report-format", "json"],
])
def test_target_environment_keeps_read_only_commands(target_environment, monkeypatch, tmp_path, arguments):
    from omegaconf import OmegaConf

    from aspec import cli, train_entry
    from aspec.commands import quickstart, recipes

    def dispatch(argv, *_):
        monkeypatch.setattr(sys, "argv", list(map(str, argv[2:])))
        importlib.import_module(argv[2]).main()

    monkeypatch.chdir(tmp_path)
    OmegaConf.save(quickstart.build_profile("qwen3-8b"), tmp_path / "profile.yaml")
    monkeypatch.setattr(cli, "execute", dispatch)
    monkeypatch.setattr(recipes, "execute", dispatch)
    monkeypatch.setattr(quickstart, "execute", dispatch)
    monkeypatch.setattr(train_entry, "compile_execution_plan", lambda *_, **__: Namespace(to_json=lambda: "{}"))
    monkeypatch.setattr(train_entry, "build_run_report", lambda _: {})
    monkeypatch.setattr(sys, "argv", ["aspec", *arguments])
    try:
        cli.main()
    except SystemExit as error:
        assert error.code == 0
    assert not (tmp_path / "unused").exists()


def test_multinode_control_facades_stay_on_the_coordinator_without_gpu_reservations():
    from aspec.inference.factory import _prepare_sgl_engines

    args = Namespace(
        cluster_mode="static_multi_node", inference_num_gpus=4,
        inference_num_gpus_per_engine=1, inference_num_gpus_per_node=2,
        inference_node_ips=["target-a", "target-b"],
    )
    with patch("aspec.inference.factory.ray") as ray:
        ray.get_runtime_context.return_value.get_node_id.return_value = "ab" * 28
        ray.get.side_effect = [30001, 30003, 30005, 30007]
        actor = ray.remote.return_value
        _prepare_sgl_engines(args, None, None)
    assert actor.options.call_count == 4
    for invocation in actor.options.call_args_list:
        assert invocation.kwargs["num_gpus"] == 0
        strategy = invocation.kwargs["scheduling_strategy"]
        assert strategy.node_id == "ab" * 28 and not strategy.soft
    calls = actor.options.return_value.remote.call_args_list
    assert [invocation.kwargs["node_ip"] for invocation in calls] == ["target-a", "target-a", "target-b", "target-b"]


def test_factory_import_does_not_import_sglang():
    result = subprocess.run([sys.executable, "-c", """
import sys
class Reject:
    def find_spec(self, fullname, *args):
        if fullname == 'sglang' or fullname.startswith('sglang.'):
            raise AssertionError('Draft imported SGLang')
sys.meta_path.insert(0, Reject())
import aspec.inference.factory
import aspec.inference.engine.target_actor
"""], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_target_bootstrap_rebuilds_environment_before_starting_worker(monkeypatch):
    import os

    from aspec import _target_worker

    for name in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LIBRARY_PATH"):
        monkeypatch.setenv(name, "/draft/foreign")
    monkeypatch.setenv("VIRTUAL_ENV", "/draft")
    monkeypatch.setenv("PATH", "/draft/bin:/usr/bin")
    monkeypatch.setenv("ASPEC_CACHE_HOME", "/tmp/aspec-cache")
    monkeypatch.setattr(sys, "argv", ["bootstrap", "ray-worker.py", "--node-id", "123"])
    calls = []
    monkeypatch.setattr(_target_worker, "execute", calls.append)
    _target_worker.main()
    assert calls == [[sys.executable, "ray-worker.py", "--node-id", "123"]]
    assert os.environ["VIRTUAL_ENV"] == sys.prefix
    assert os.environ["PATH"].startswith(os.path.dirname(sys.executable) + os.pathsep)
    assert os.environ["ASPEC_CACHE_HOME"] == "/tmp/aspec-cache/target"
    assert not any(name in os.environ for name in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LIBRARY_PATH"))


def test_target_worker_uses_startup_validation_without_repeating_doctor(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from aspec.inference.engine.target_actor import TargetActor

    engine = MagicMock()
    monkeypatch.setitem(sys.modules, "aspec.inference.engine.sgl_engine", SimpleNamespace(SglEngine=engine))
    actor = TargetActor(gpu_ids=[4, 6], base_gpu_id=4)
    assert actor.gpu_ids == [4, 6]
    engine.assert_called_once_with(base_gpu_id=4)


@pytest.mark.parametrize("command", ["evaluate", "prepare-target-data", "patch-sglang"])
def test_target_cli_commands_run_in_the_active_environment(monkeypatch, command):
    from aspec import cli

    def missing(_):
        raise importlib.metadata.PackageNotFoundError("ling-aspec-target")

    monkeypatch.setattr(importlib.metadata, "version", missing)
    monkeypatch.setenv("ASPEC_TARGET_PYTHON", "/target/bin/python")
    monkeypatch.setattr(sys, "argv", ["aspec", command])
    calls = []
    monkeypatch.setattr(cli, "execute", calls.append)
    cli.main()
    assert calls[0][:2] == [sys.executable, "-m"]


def test_target_address_is_recorded_in_plan_without_opposing_interpreter(monkeypatch):
    from aspec.commands.quickstart import build_profile
    from aspec.config.train_config import load_config
    from aspec.planning import compile_execution_plan

    monkeypatch.setenv("ASPEC_TARGET_PYTHON", "/environment-target/bin/python")
    config = build_profile("qwen3-8b")
    config.inference.target_service = {"address": "http://target:30000"}
    plan = compile_execution_plan(load_config(base_config=config))
    assert plan.to_runtime_args().target_service["address"] == "http://target:30000"
    inference = plan.to_dict()["policy"]["resources"]["inference"]
    assert "python_executable" not in inference
    assert inference["target_service"]["mode"] == "external"


def test_partial_target_preparation_rolls_back_every_created_facade():
    from unittest.mock import MagicMock, call

    from aspec.inference.factory import _prepare_sgl_engines

    args = Namespace(inference_num_gpus=2, inference_num_gpus_per_engine=1)
    first, second = MagicMock(), MagicMock()
    with patch("aspec.inference.factory.ray") as ray:
        ray.get_runtime_context.return_value.get_node_id.return_value = "ab" * 28
        actor = ray.remote.return_value
        actor.options.return_value.remote.side_effect = [first, second]
        ray.get.side_effect = [30001, RuntimeError("port probe failed"), None, None]
        with pytest.raises(RuntimeError, match="port probe failed"):
            _prepare_sgl_engines(args, None)
    for engine in (first, second):
        engine.init.remote.assert_not_called()
        engine.shutdown.remote.assert_called_once_with(force=True)
    assert ray.kill.call_args_list == [call(first, no_restart=True), call(second, no_restart=True)]
    assert actor.options.call_args.kwargs["scheduling_strategy"].node_id == "ab" * 28


def test_cross_node_target_tp_is_rejected_before_creating_facades():
    from aspec.inference.factory import _prepare_sgl_engines

    with patch("aspec.inference.factory.ray") as ray:
        with pytest.raises(ValueError, match="node-local"):
            _prepare_sgl_engines(Namespace(sglang_nnodes=2), None)
        ray.remote.assert_not_called()
