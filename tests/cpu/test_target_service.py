# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Control protocol, source ownership and independent launch regressions."""
import os
import sys
import threading
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock

import psutil
import pytest
import torch

from aspec.inference.service import TargetService, http_server, readers_exited
from aspec.inference.service_protocol import (
    decode,
    encode,
    engine_arguments,
    request,
    validate_address,
)


def test_control_payload_roundtrip_and_feature_tensor_rejection():
    payload = {"input_ids_ref": [torch.tensor([1, 3, 5])],
               "result": {"tensor_dtypes": {"hidden_states": torch.bfloat16}}}
    result = decode(encode(payload))
    assert result["input_ids_ref"][0].tolist() == [1, 3, 5]
    assert result["result"]["tensor_dtypes"]["hidden_states"] == torch.bfloat16
    with pytest.raises(TypeError, match="Only CPU token IDs"):
        encode(torch.ones(3, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="dtype"):
        decode({"$dtype": "not_a_dtype"})


def test_target_shutdown_keeps_polling_interval_until_sources_are_idle(monkeypatch, tmp_path):
    import signal

    from aspec.inference import service as module

    events = []
    handlers = {}
    instance = MagicMock(draining=False)
    instance.idle.side_effect = lambda: events.count("reap") == 2
    instance.reap_disconnected.side_effect = lambda: events.append("reap")
    server = MagicMock()
    worker = MagicMock()

    def register(sig, handler):
        handlers[sig] = handler
        return signal.SIG_DFL

    worker.start.side_effect = lambda: handlers[signal.SIGTERM](signal.SIGTERM, None)
    monkeypatch.setattr(module.signal, "signal", register)
    monkeypatch.setattr(module.threading, "Thread", lambda **_: worker)
    monkeypatch.setattr(module, "TargetService", lambda **_: instance)
    monkeypatch.setattr(module, "http_server", lambda *_: server)
    monkeypatch.setattr(module, "visible_gpu_ids", lambda: [0])
    monkeypatch.setattr(module, "inventory", lambda: {"python": sys.executable})
    monkeypatch.setattr(module, "ray", MagicMock())
    monkeypatch.setattr(module.tempfile, "mkdtemp", lambda **_: str(tmp_path))
    monkeypatch.setattr(module.time, "sleep", lambda seconds: events.append(("wait", seconds)))
    monkeypatch.setenv("RAY_USAGE_STATS_ENABLED", "0")
    monkeypatch.setattr(sys, "argv", ["aspec", "--address", "http://127.0.0.1:30000"])

    module.main()

    assert instance.draining
    assert events == [("wait", 1), "reap", ("wait", 1), "reap"]
    server.shutdown.assert_called_once()
    server.server_close.assert_called_once()
    module.ray.shutdown.assert_called_once()


@pytest.mark.parametrize("address", ["file:///tmp/a", "https://target:30000", "http://target",
                                    "http://user:secret@target:30000", "http://target:30000/run"])
def test_invalid_target_address_is_rejected(address):
    with pytest.raises(ValueError):
        validate_address(address)


@pytest.mark.parametrize("host", ["[::1]", "[2001:db8::1]"])
def test_target_rejects_ipv6_literals(host):
    with pytest.raises(ValueError, match="IPv6"):
        validate_address(f"http://{host}:30000")


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "target.example"])
def test_target_accepts_ipv4_and_hostnames(host):
    address = f"http://{host}:30000"
    assert validate_address(address) == address


def test_no_driver_objects_cross_control_boundary():
    args = Namespace(target_model_path="qwen", draft_model_config_obj=object(),
                     wandb_key="private", learning_rate=0.1, _mooncake_master_actor=object())
    assert engine_arguments(args) == {"target_model_path": "qwen"}
    args.mystery = object()
    assert engine_arguments(args) == {"target_model_path": "qwen"}


def test_complete_training_plan_has_an_explicit_data_only_target_boundary():
    import json

    from aspec.commands.quickstart import build_profile
    from aspec.config.train_config import load_config
    from aspec.planning import compile_execution_plan

    args = compile_execution_plan(load_config(base_config=build_profile("qwen3-8b"))).to_runtime_args()
    payload = engine_arguments(args)
    json.dumps(payload, allow_nan=False)
    assert payload["aspec_feature_contract"]["contract_id"]
    assert "target_service" not in payload
    assert "wandb_key" not in payload


@pytest.mark.parametrize("algorithm,hidden_size", [("dspark", 5120), ("mtp", 4096)])
def test_target_capture_policy_survives_the_control_boundary(algorithm, hidden_size):
    import json

    args = Namespace(
        mooncake_hidden_dim=hidden_size, draft_algorithm=algorithm,
        draft_context_parallel_size=2, max_seq_length=261888,
        store_last_hidden_states=True, target_prefill_memory_optimized=False,
        attention_backend="cudnn_flex_attention",
    )
    target_args = Namespace(**decode(json.loads(json.dumps(engine_arguments(args)))))
    assert target_args.mooncake_hidden_dim == hidden_size
    assert target_args.draft_algorithm == algorithm


@pytest.fixture
def service(monkeypatch):
    import aspec.inference.service as module

    handle = MagicMock()
    handle.get_status.remote.return_value = {"initialized": True, "pid": 999999999, "created": 1.0}
    monkeypatch.setattr(module.os, "getpgid", lambda pid: pid)
    handle.shutdown.remote.return_value = None
    actor = MagicMock()
    actor.options.return_value.remote.return_value = handle
    import ray

    monkeypatch.setattr(module, "ray", SimpleNamespace(remote=lambda _: actor,
                                                     get=lambda x, **_: x, kill=MagicMock(),
                                                     exceptions=ray.exceptions))
    instance = TargetService(model="qwen", devices=[0, 1], runtime={})
    instance.test_handle = handle
    return instance


def creation(service, key="engine", count=1):
    from aspec.inference.target_runtime import release_version

    return {"key": key, "version": release_version(),
            "arguments": {"target_model_path": "qwen", "aspec_attempt_id": "attempt-1",
                          "aspec_run_id": "run-1"},
            "rank": 0, "num_gpus_per_engine": count,
            "node_rank": 0, "engine_group": 0}


def test_gpu_ownership_and_closed_engine_idempotence(service):
    service.create(creation(service, count=2))
    with pytest.raises(RuntimeError, match="Insufficient free Target GPUs"):
        service.create(creation(service, "other"))
    service.call("engine", "shutdown", [], {})
    service.call("engine", "shutdown", [], {})
    assert service.test_handle.shutdown.remote.call_count == 1
    assert service.idle()
    service.create(creation(service, "next"))


def test_model_identity_and_allowed_gpu_fail_before_creation(service):
    wrong = creation(service)
    wrong["version"] = "different"
    with pytest.raises(ValueError, match="versions differ"):
        service.create(wrong)
    wrong = creation(service)
    wrong["arguments"]["target_model_path"] = "other-model"
    with pytest.raises(ValueError, match="model"):
        service.create(wrong)
    with pytest.raises(RuntimeError, match="Insufficient free Target GPUs"):
        service.create(creation(service, count=3))
    assert service.idle()


def test_pending_creation_is_not_expired_before_heartbeat_can_start(service, monkeypatch):
    import aspec.inference.service as module

    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])

    def initializing(value, **_):
        clock[0] = 45.0
        service.reap_disconnected()
        assert service.engines["engine"]["state"] == "creating"
        return value

    monkeypatch.setattr(module.ray, "get", initializing)
    service.create(creation(service))
    assert service.engines["engine"]["active"] == 0
    # Creation took longer than the heartbeat expiry. The first watchdog tick
    # after returning must still give the new client its full heartbeat window.
    service.reap_disconnected()
    assert service.engines["engine"]["state"] == "created"


def test_target_allocates_its_own_gpu_pool_and_excludes_local_readers(service):
    value = creation(service)
    value["arguments"]["aspec_read_owners"] = [
        {"node_ip": service.node_ip, "assigned_gpu_ids": [0]},
    ]
    assert service.create(value)["gpu_ids"] == [1]
    assert set(service.claims) == {1}


def test_remote_readers_do_not_reserve_same_numbered_target_gpu(service):
    value = creation(service)
    value["arguments"]["aspec_read_owners"] = [
        {"node_ip": "another-node", "assigned_gpu_ids": [0, 1]},
    ]
    assert service.create(value)["gpu_ids"] == [0]


def test_wrong_service_node_and_unknown_target_fields_are_rejected(service):
    value = creation(service)
    value["node_ip"] = "another-node"
    with pytest.raises(ValueError, match="planned node"):
        service.create(value)
    value.pop("node_ip")
    value["arguments"]["wandb_key"] = "must-not-cross"
    with pytest.raises(ValueError, match="Unknown Target configuration"):
        service.create(value)
    assert service.idle()


def test_handshake_uses_release_not_source_hash_or_python_minor(monkeypatch):
    from aspec.inference import service_client
    from aspec.inference.target_runtime import release_version

    payloads = []

    def control(_address, path, payload=None, **_):
        if path == "/v1/info":
            return {"version": release_version(), "identity": {
                "code_sha256": "different-source", "python_minor": [3, 11]}, "node_ip": "target"}
        payloads.append(payload)
        return {"key": payload["key"], "gpu_ids": [5]}

    monkeypatch.setattr(service_client, "request", control)
    monkeypatch.setattr(service_client.threading, "Thread", MagicMock())
    service_client.TargetServiceClient(address="http://target:30000", args=Namespace(),
                                       rank=0, num_gpus_per_engine=1)
    assert payloads[0]["version"] == release_version()
    assert "identity" not in payloads[0]
    assert "base_gpu_id" not in payloads[0]


def test_native_scheduler_uses_the_callers_run_identity(service, monkeypatch):
    import aspec.inference.service as module

    monkeypatch.setenv("ASPEC_RUN_ID", "stale-service-run")
    monkeypatch.setenv("ASPEC_ATTEMPT_ID", "stale-service-attempt")
    actor = module.ray.remote(None)
    service.create(creation(service))
    environment = actor.options.call_args.kwargs["runtime_env"]["env_vars"]
    assert environment["ASPEC_RUN_ID"] == "run-1"
    assert environment["ASPEC_ATTEMPT_ID"] == "attempt-1"
    assert environment["ASPEC_SHARED_SOURCE"] == "0"


@pytest.mark.parametrize("second,views", [("contract-a", "0"), ("contract-b", "1")])
def test_shared_source_view_flags_follow_the_consumer_contract_ids(second, views):
    from aspec.inference.service import session_environment

    environment = session_environment(Namespace(
        aspec_run_id="run", aspec_attempt_id="attempt", aspec_shared_source=True,
        aspec_shared_consumer_contracts={"first": "contract-a", "second": second},
    ))
    assert environment["ASPEC_SHARED_SOURCE"] == "1"
    assert environment["ASPEC_SHARED_FEATURE_VIEWS"] == views


def test_disconnect_retains_unverified_reader_storage(service, monkeypatch):
    service.create(creation(service))
    service.call("engine", "generate", [], {})
    service.engines["engine"]["processes"] = {999999999: 1.0}
    service.reap_disconnected(expiry=-1)
    assert service.engines["engine"]["state"] == "retained"
    assert not service.idle()
    service.test_handle.shutdown.remote.assert_not_called()
    monkeypatch.setattr("aspec.inference.service.readers_exited", lambda *_: True)
    service.reap_disconnected(expiry=-1)
    assert service.idle()
    service.test_handle.shutdown.remote.assert_not_called()


def test_failed_source_actor_is_retired_only_after_reader_fence(service, monkeypatch):
    import ray

    service.create(creation(service))
    service.call("engine", "generate", [], {})
    item = service.engines["engine"]
    item["processes"] = {999999999: 1.0}
    service.test_handle.shutdown.remote.side_effect = ray.exceptions.RayActorError("actor exited")
    monkeypatch.setattr("aspec.inference.service.readers_exited", lambda *_: False)
    stop = MagicMock()
    monkeypatch.setattr("aspec.inference.service.stop_owned_processes", stop)
    service.reap_disconnected(expiry=-1)
    stop.assert_not_called()
    assert not service.idle()
    monkeypatch.setattr("aspec.inference.service.readers_exited", lambda *_: True)
    service.reap_disconnected(expiry=-1)
    stop.assert_called_once_with(item["processes"])
    assert service.idle()
    assert item["terminal_reason"] == "readers_fenced_sources_exited"
    service.test_handle.shutdown.remote.assert_not_called()


def test_disconnected_fenced_source_bypasses_a_stalled_native_drain(service, monkeypatch):
    import ray

    service.create(creation(service))
    service.call("engine", "generate", [], {})
    item = service.engines["engine"]
    item["processes"] = {999999999: 1.0}
    service.test_handle.shutdown.remote.side_effect = ray.exceptions.GetTimeoutError("native drain")
    stop = MagicMock()
    monkeypatch.setattr("aspec.inference.service.readers_exited", lambda *_: True)
    monkeypatch.setattr("aspec.inference.service.stop_owned_processes", stop)
    service.reap_disconnected(expiry=-1)
    stop.assert_called_once_with(item["processes"])
    service.test_handle.shutdown.remote.assert_not_called()
    assert service.idle() and item["state"] == "closed"


def test_disconnected_fenced_source_is_retired_with_an_active_generation(service, monkeypatch):
    service.create(creation(service))
    service.call("engine", "generate", [], {})
    item = service.engines["engine"]
    item.update(processes={999999999: 1.0}, active=1, state="ready")
    monkeypatch.setattr("aspec.inference.service.readers_exited", lambda *_: False)
    stop = MagicMock()
    monkeypatch.setattr("aspec.inference.service.stop_owned_processes", stop)
    service.reap_disconnected(expiry=-1)
    stop.assert_not_called()
    assert not service.idle()
    monkeypatch.setattr("aspec.inference.service.readers_exited", lambda *_: True)
    service.reap_disconnected(expiry=-1)
    stop.assert_called_once_with(item["processes"])
    assert item["state"] == "closed" and service.idle()


@pytest.mark.parametrize("failed_reply", [False, True])
def test_late_graceful_shutdown_cannot_reopen_or_release_a_new_gpu_lease(service, monkeypatch, failed_reply):
    service.create(creation(service))
    item = service.engines["engine"]
    item["processes"] = {999999999: 1.0}
    started, finish = threading.Event(), threading.Event()
    errors = []
    ref = service.test_handle.shutdown.remote.return_value

    def get(value, **_kwargs):
        if value is ref:
            started.set()
            assert finish.wait(timeout=5)
            if failed_reply:
                raise RuntimeError("late native shutdown failure")
        return value

    def shutdown():
        try:
            service.call("engine", "shutdown", [], {})
        except RuntimeError as error:
            errors.append(str(error))

    monkeypatch.setattr("aspec.inference.service.ray.get", get)
    monkeypatch.setattr("aspec.inference.service.stop_owned_processes", MagicMock())
    worker = threading.Thread(target=shutdown)
    worker.start()
    try:
        assert started.wait(timeout=5)
        assert item["state"] == "closing"
        service.heartbeat("engine")  # A live caller may still be draining.
        service.call("engine", "shutdown", [], {"force": True})
        service.create(creation(service, key="next"))
    finally:
        finish.set()
        worker.join(timeout=5)
    assert not worker.is_alive() and item["state"] == "closed"
    assert service.claims == {0: "next"}
    assert errors == (["late native shutdown failure"] if failed_reply else [])


def test_shutdown_does_not_race_active_generation(service):
    service.create(creation(service))
    service.engines["engine"]["active"] = 1
    with pytest.raises(RuntimeError, match="active calls"):
        service.call("engine", "shutdown", [], {})
    service.test_handle.shutdown.remote.assert_not_called()


def test_fenced_force_shutdown_stops_owned_processes_without_a_native_drain(service, monkeypatch):
    import aspec.inference.service as module

    service.create(creation(service))
    service.call("engine", "generate", [], {})
    item = service.engines["engine"]
    item.update(processes={999999999: 1.0}, active=1)
    observe, stop = MagicMock(), MagicMock()
    monkeypatch.setattr(module, "observe_process_tree", observe)
    monkeypatch.setattr(module, "stop_owned_processes", stop)
    service.call("engine", "shutdown", [], {"force": True})
    observe.assert_called_once_with(item["processes"])
    module.ray.kill.assert_called_once_with(service.test_handle, no_restart=True)
    stop.assert_called_once_with(item["processes"])
    service.test_handle.shutdown.remote.assert_not_called()
    assert service.idle() and not service.claims
    assert item["terminal_reason"] == "readers_fenced_sources_exited"
    service.call("engine", "shutdown", [], {"force": True})
    stop.assert_called_once()


@pytest.mark.parametrize("known_processes", [False, True])
def test_force_shutdown_retains_lease_until_source_exit_is_observed(service, monkeypatch, known_processes):
    service.create(creation(service))
    item = service.engines["engine"]
    if known_processes:
        item["processes"] = {999999999: 1.0}
        monkeypatch.setattr("aspec.inference.service.stop_owned_processes",
                            MagicMock(side_effect=RuntimeError("source alive")))
    else:
        item["processes"].clear()
    with pytest.raises(RuntimeError, match="source alive|identities are unknown"):
        service.call("engine", "shutdown", [], {"force": True})
    assert item["state"] == "retained" and service.claims
    assert not service.idle()


def test_late_initialization_reply_cannot_reopen_a_forcibly_closed_source(service, monkeypatch):
    import threading

    service.create(creation(service))
    item = service.engines["engine"]
    item["processes"] = {999999999: 1.0}
    started, finish = threading.Event(), threading.Event()
    ref = service.test_handle.init.remote.return_value

    def get(value, **_kwargs):
        if value is ref:
            started.set()
            assert finish.wait(timeout=5)
            return None
        return value

    monkeypatch.setattr("aspec.inference.service.ray.get", get)
    monkeypatch.setattr("aspec.inference.service.observe_process_tree", MagicMock())
    monkeypatch.setattr("aspec.inference.service.stop_owned_processes", MagicMock())
    worker = threading.Thread(target=lambda: service.call("engine", "init", [], {}))
    worker.start()
    try:
        assert started.wait(timeout=5)
        service.call("engine", "shutdown", [], {"force": True})
    finally:
        finish.set()
        worker.join(timeout=5)
    assert not worker.is_alive() and item["state"] == "closed"
    assert service.idle()


def test_reader_fence_requires_process_identity_and_local_observation():
    owner = {"pid": os.getpid(), "created": psutil.Process().create_time(), "node_ip": "local"}
    assert not readers_exited([owner], "local")
    assert not readers_exited([owner], "another-host")
    assert not readers_exited([], "local")
    assert readers_exited([{**owner, "created": owner["created"] - 100}], "local")


def test_owned_source_exit_does_not_require_pidfd_waiting(monkeypatch):
    import subprocess
    import sys

    from aspec.inference.service import stop_owned_processes

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        created = psutil.Process(child.pid).create_time()
        monkeypatch.setattr(psutil, "wait_procs", MagicMock(side_effect=OSError(22, "Invalid argument")))
        stop_owned_processes({child.pid: created})
        assert child.poll() is not None
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)


def test_owned_source_cleanup_does_not_signal_a_reused_pid():
    import subprocess
    import sys

    from aspec.inference.service import stop_owned_processes

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        stop_owned_processes({child.pid: psutil.Process(child.pid).create_time() - 1})
        assert child.poll() is None
    finally:
        child.kill()
        child.wait(timeout=10)


@pytest.mark.parametrize("strict,force,node_state,api_error,retired", [
    (True, True, "dead", False, True),
    (False, True, "dead", False, False),
    (True, False, "dead", False, False),
    (True, True, "alive", False, False),
    (True, True, "missing", False, False),
    (True, True, "replaced", False, False),
    (True, True, "dead", True, False),
])
def test_target_shutdown_distinguishes_lost_local_domain_from_unverified_failure(
    monkeypatch, strict, force, node_state, api_error, retired,
):
    import asyncio

    from aspec.inference import service_client
    from aspec.inference.target_runtime import release_version

    def control(_address, path, payload=None, **_kwargs):
        if path == "/v1/info":
            return {"version": release_version(), "node_ip": "target"}
        return {"key": payload["key"], "gpu_ids": [0]}

    nodes = [{"NodeID": "old-source-node", "NodeManagerAddress": "target", "Alive": True}]
    monkeypatch.setattr(service_client, "request", control)
    monkeypatch.setattr(service_client.ray, "nodes", lambda: nodes)
    monkeypatch.setattr(service_client.TargetServiceClient, "_keepalive", lambda _self: None)
    client = service_client.TargetServiceClient(
        address="http://target:30000", rank=0, num_gpus_per_engine=1,
        args=Namespace(cluster_mode="static_multi_node",
                       mooncake_feature_affinity="node_local_required" if strict else "any"),
    )
    if node_state == "dead":
        nodes[0]["Alive"] = False
    elif node_state == "missing":
        nodes.clear()
    elif node_state == "replaced":
        nodes[0]["NodeID"] = "another-node"
    error = RuntimeError("Target refused cleanup") if api_error else OSError("Target unreachable")
    client._call = MagicMock(side_effect=error)
    if retired:
        result = asyncio.run(client.shutdown(force=force))
        assert result == {"state": "node_local_domain_lost", "node_id": "old-source-node",
                          "physical_memory_unobserved": True}
        assert client._closed.is_set()
    else:
        with pytest.raises(type(error), match=str(error)):
            asyncio.run(client.shutdown(force=force))
        assert not client._closed.is_set()


@pytest.mark.parametrize("method", ["__dict__", "generate_with_decode", "update_weights_from_disk"])
def test_real_http_control_roundtrip_and_unknown_method(service, method):
    server = http_server(service, "127.0.0.1", 0)
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    address = f"http://127.0.0.1:{server.server_port}"
    try:
        assert request(address, "/v1/info")["runtime"] == service.runtime
        request(address, "/v1/engines", creation(service))
        assert request(address, "/v1/call", {"key": "engine", "method": "get_status",
                                             "args": [], "kwargs": {}}) == service.test_handle.get_status.remote.return_value
        with pytest.raises(RuntimeError, match="Unknown Target control method"):
            request(address, "/v1/call", {"key": "engine", "method": method,
                                          "args": [], "kwargs": {}})
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_managed_local_keeps_resume_configuration(tmp_path):
    from aspec.commands.quickstart import build_profile
    from aspec.config.train_config import load_config

    config = build_profile("qwen3-8b")
    config.inference.target_service = {"mode": "managed-local", "python_executable": "/target/bin/python"}
    config.training.load_path = str(tmp_path / "checkpoint")
    loaded = load_config(base_config=config)
    assert loaded.training.load_path == str(tmp_path / "checkpoint")


def test_managed_supervisor_rejects_ipv6_before_probing(monkeypatch):
    from aspec.inference import managed

    probe = MagicMock()
    monkeypatch.setattr(managed, "request", probe)
    args = Namespace(target_service={"mode": "managed-local", "address": "http://[::1]:30000"})
    with pytest.raises(ValueError, match="loopback"):
        with managed.managed_target(args):
            pytest.fail("accepted IPv6 loopback")
    probe.assert_not_called()


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_managed_supervisor_never_stops_an_existing_service(monkeypatch, host):
    from aspec.inference import managed

    monkeypatch.setattr(managed, "request", lambda *_a, **_k: {})
    popen = MagicMock()
    monkeypatch.setattr(managed.subprocess, "Popen", popen)
    with pytest.raises(RuntimeError, match="existing Target"):
        with managed.managed_target(Namespace(target_service={"mode": "managed-local",
                                                              "address": f"http://{host}:30000"})):
            pytest.fail("existing service was adopted")
    popen.assert_not_called()


def test_managed_supervisor_splits_gpu_visibility_and_restores_parent(monkeypatch, tmp_path):
    from aspec.inference import managed

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    control = MagicMock(side_effect=[OSError("not started"), {"pid": 123}])
    process = MagicMock(pid=123, returncode=0)
    process.poll.return_value = None
    popen = MagicMock(return_value=process)
    monkeypatch.setattr(managed, "request", control)
    monkeypatch.setattr(managed.subprocess, "Popen", popen)
    args = Namespace(target_service={"mode": "managed-local", "address": "http://127.0.0.1:30000",
                                     "python_executable": "/target/bin/python"},
                     output_dir=str(tmp_path), target_model_path="qwen",
                     training_num_nodes=1, training_num_gpus_per_node=1, inference_num_gpus=1)
    with managed.managed_target(args):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "4"
        assert popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "5"
        process.poll.return_value = 0
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4,5"


@pytest.mark.parametrize("method", ["init", "generate"])
def test_draining_service_rejects_new_work_and_accepts_cleanup(service, method):
    service.create(creation(service))
    service.draining = True
    with pytest.raises(RuntimeError, match="draining"):
        service.call("engine", method, [], {})
    getattr(service.test_handle, method).remote.assert_not_called()
    service.call("engine", "get_status", [], {})
    service.call("engine", "shutdown", [], {})
    assert service.idle()


def test_generation_timeout_requires_fenced_shutdown(service, monkeypatch):
    from aspec.inference import service as module

    service.create(creation(service))
    monkeypatch.setattr(module.ray, "get", MagicMock(side_effect=module.ray.exceptions.GetTimeoutError()))
    with pytest.raises(module.ray.exceptions.GetTimeoutError):
        service.call("engine", "generate", [], {})
    with pytest.raises(RuntimeError, match="retained"):
        service.call("engine", "shutdown", [], {})
    service.reap_disconnected(expiry=-1)
    assert not service.idle()
    service.test_handle.shutdown.remote.assert_not_called()
    service.call("engine", "shutdown", [], {"force": True})
    assert service.idle()


@pytest.mark.parametrize("initialized", [False, True])
def test_disconnected_startup_without_generation_releases_remote_readers_lease(service, initialized):
    value = creation(service)
    value["arguments"]["aspec_read_owners"] = [{"node_ip": "remote"}]
    service.create(value)
    if initialized:
        service.call("engine", "init", [], {})
    service.reap_disconnected(expiry=-1)
    assert service.idle()
    assert service.engines["engine"]["terminal_reason"] == "unused_sources_exited"


def test_force_retains_claim_until_late_worker_group_member_exits(service, monkeypatch):
    from aspec.inference import service as module

    service.create(creation(service))
    probe = MagicMock()
    monkeypatch.setattr(module.os, "killpg", probe)
    monkeypatch.setattr(module.time, "monotonic", iter(range(0, 1000, 11)).__next__)
    with pytest.raises(RuntimeError, match="process group exit"):
        service.call("engine", "shutdown", [], {"force": True})
    assert service.claims == {0: "engine"} and service.engines["engine"]["state"] == "retained"
    probe.assert_called_once_with(999999999, 0)
    probe.side_effect = ProcessLookupError()
    service.call("engine", "shutdown", [], {"force": True})
    assert service.idle()


def test_control_status_does_not_block_other_facade_requests(monkeypatch):
    import asyncio

    from aspec.inference import service_client

    started, release = threading.Event(), threading.Event()

    def request(_address, _path, payload, **_):
        if payload["method"] == "get_status":
            started.set()
            assert release.wait(5)
            return {}
        return True

    monkeypatch.setattr(service_client, "request", request)
    client = service_client.TargetServiceClient.__new__(service_client.TargetServiceClient)
    client.address, client.key, client.timeout = "http://target:30000", "key", 5

    async def run():
        status = asyncio.create_task(client.get_status())
        try:
            assert await asyncio.to_thread(started.wait, 3)
            assert await asyncio.wait_for(client.health_check(), timeout=2)
            assert not status.done()
        finally:
            release.set()
            await status

    asyncio.run(run())


def test_sigterm_during_managed_readiness_stops_only_owned_target(monkeypatch, tmp_path):
    import signal

    from aspec import train_entry
    from aspec.inference import managed
    from aspec.runtime_control import TerminationRequested

    args = Namespace(target_service={"mode": "managed-local", "address": "http://127.0.0.1:30000",
                                     "python_executable": "/target/bin/python"},
                     output_dir=str(tmp_path), target_model_path="qwen",
                     training_num_nodes=1, training_num_gpus_per_node=1, inference_num_gpus=1)
    process = MagicMock()
    process.poll.return_value = None
    process.send_signal.side_effect = lambda sig: setattr(process.poll, "return_value", -sig)
    monkeypatch.setattr(train_entry, "_prepare_config", lambda: (args, object()))
    monkeypatch.setattr(managed.subprocess, "Popen", lambda *_a, **_kw: process)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,6")
    def request(*_args, **_kwargs):
        if not process.poll.called:
            raise OSError("no existing service")
        signal.raise_signal(signal.SIGTERM)

    monkeypatch.setattr(managed, "request", request)
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(TerminationRequested):
        train_entry.main()
    process.send_signal.assert_called_once_with(signal.SIGTERM)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4,6"
    assert signal.getsignal(signal.SIGTERM) == previous


def test_draining_reclaims_idle_sessions_despite_continuing_heartbeats(service, monkeypatch):
    from aspec.inference import service as module

    clock = 0
    monkeypatch.setattr(module.time, "monotonic", lambda: clock)
    exited = False
    monkeypatch.setattr(module, "readers_exited", lambda *_: exited)
    service.create(creation(service, "unused"))
    service.create(creation(service, "used"))
    service.call("used", "generate", [], {})
    service.draining = True

    for clock in range(5, 51, 5):
        service.heartbeat("used")
        service.call("used", "get_status", [], {})
        if clock == 5:
            service.heartbeat("unused")
        service.reap_disconnected()
        assert service.engines["unused"]["state"] == "closed"
        assert service.claims == {1: "used"}

    exited = True
    service.heartbeat("used")
    service.reap_disconnected()
    assert service.idle()


def test_drain_finishes_an_accepted_call_and_stops_new_polls(service, monkeypatch):
    from aspec.inference import service as module

    service.create(creation(service))

    def finish_status(result, **_):
        service.draining = True
        service.heartbeat("engine")
        service.reap_disconnected()
        assert service.claims == {0: "engine"}
        with pytest.raises(RuntimeError, match="retained"):
            service.call("engine", "get_status", [], {})
        return result

    monkeypatch.setattr(module.ray, "get", finish_status)
    service.call("engine", "get_status", [], {})
    service.reap_disconnected()
    assert service.idle()


@pytest.mark.parametrize("nnodes,node_rank", [(2, 0), (1, 1)])
def test_cross_node_tp_is_rejected_before_target_allocation(service, nnodes, node_rank):
    value = creation(service)
    value["arguments"]["sglang_nnodes"] = nnodes
    value["node_rank"] = node_rank
    with pytest.raises(ValueError, match="node-local"):
        service.create(value)
    assert not service.engines and service.idle()
