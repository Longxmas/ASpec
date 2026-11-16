# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Standalone Target control service around the retained SglEngine.

Inspired by SpecForge f8cc3dc external capture. JSON control messages replace
cross-environment Ray calls; ASpec's GPU leases, feature contracts and tensor
transport remain in use. No SpecForge server code is copied.
"""
import argparse
import json
import logging
import os
import signal
import tempfile
import threading
import time
import uuid
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import ray

from aspec.inference.service_protocol import (
    MAX_BODY,
    METHODS,
    PROTOCOL,
    TARGET_ARGUMENTS,
    decode,
    encode,
    validate_address,
    validate_target_tp,
)
from aspec.inference.target_runtime import inventory, release_version
from aspec.utils.env import get_aspec_env_vars
from aspec.utils.misc import get_current_node_ip, visible_gpu_ids

logger = logging.getLogger(__name__)


def readers_exited(owners, local_ip):
    """Only observe local processes; unknown/remote owners retain source storage."""
    import psutil

    if not owners:
        return False
    for owner in owners:
        if (owner.get("node_ip") != local_ip or not isinstance(owner.get("pid"), int)
                or not isinstance(owner.get("created"), (int, float))):
            return False
        try:
            process = psutil.Process(owner["pid"])
            if (process.create_time() == owner["created"]
                    and process.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)):
                return False
        except psutil.NoSuchProcess:
            pass
    return True


class TargetService:
    def __init__(self, *, model=None, devices=None, runtime=None):
        self.runtime = runtime if runtime is not None else inventory()
        self.model = model
        self.devices = tuple(visible_gpu_ids() if devices is None else devices)
        self.node_ip = get_current_node_ip()
        self.server_id = uuid.uuid4().hex
        self.lock = threading.RLock()
        self.engines = {}
        self.claims = {}
        self.draining = False

    def info(self):
        with self.lock:
            return {"protocol": PROTOCOL, "version": release_version(),
                    "runtime": self.runtime, "node_ip": self.node_ip,
                    "pid": os.getpid(),
                    "server_id": self.server_id, "model": self.model,
                    "devices": list(self.devices),
                    "draining": self.draining,
                    "engines": {key: {k: item[k] for k in ("state", "session", "gpus")}
                                for key, item in self.engines.items()}}

    def create(self, value):
        from aspec.inference.engine.target_actor import TargetActor

        if value["version"] != release_version():
            raise ValueError("Draft/Target ASpec release versions differ")
        if value.get("node_ip") and value["node_ip"] != self.node_ip:
            raise ValueError("Target service is not on the planned node")
        arguments = decode(value["arguments"])
        if set(arguments) - TARGET_ARGUMENTS:
            raise ValueError("Unknown Target configuration fields")
        args = Namespace(**arguments)
        validate_target_tp(getattr(args, "sglang_nnodes", 1), value["node_rank"])
        if self.model and args.target_model_path != self.model:
            raise ValueError("Target model differs from the service configuration")
        session = getattr(args, "aspec_attempt_id", None)
        run_id = getattr(args, "aspec_run_id", None)
        if not isinstance(session, str) or not session or not isinstance(run_id, str) or not run_id:
            raise ValueError("Target requires explicit run and attempt identities")
        count = value["num_gpus_per_engine"]
        if type(count) is not int or count <= 0:
            raise ValueError("Invalid Target GPU allocation")
        key = value["key"]
        with self.lock:
            if self.draining:
                raise RuntimeError("Target service is draining")
            if key in self.engines:
                raise ValueError("Target engine identity already exists; creation is not replayed")
            # Allocate only from this service's CUDA-visible devices. Same-host
            # reader placements also fence Draft GPUs when visibility overlaps.
            readers = getattr(args, "aspec_read_owners", None) or ()
            unavailable = set(self.claims)
            for owner in readers:
                if owner.get("node_ip") == self.node_ip:
                    unavailable.update(owner.get("assigned_gpu_ids") or ())
            gpus = next((list(self.devices[start:start + count])
                         for start in range(len(self.devices) - count + 1)
                         if not unavailable.intersection(self.devices[start:start + count])), None)
            if gpus is None:
                raise RuntimeError("Insufficient free Target GPUs; select disjoint Target/Draft CUDA_VISIBLE_DEVICES")
            for old_key in list(self.engines):
                if len(self.engines) < 128:
                    break
                if self.engines[old_key]["state"] == "closed":
                    del self.engines[old_key]
            environment = get_aspec_env_vars(args)
            environment.update(session_environment(args))
            actor = ray.remote(TargetActor).options(
                num_cpus=0.2, num_gpus=0, runtime_env={"env_vars": environment},
            ).remote(args=args, rank=value["rank"], gpu_ids=gpus,
                     base_gpu_id=gpus[0], num_gpus_per_engine=count,
                     node_rank=value["node_rank"], engine_group=value["engine_group"])
            self.engines[key] = {"actor": actor, "state": "creating", "session": session,
                                 "gpus": gpus, "heartbeat": time.monotonic(),
                                 "owners": getattr(args, "aspec_read_owners", None),
                                 "status": {}, "features": {}, "active": 1, "processes": {},
                                 "generation_admitted": False}
            self.claims.update({gpu: key for gpu in gpus})
        try:
            status = ray.get(actor.get_status.remote(), timeout=120)
            pid = status["pid"]
            if os.getpgid(pid) != pid:
                raise RuntimeError("Target worker requires its own Ray process group")
            with self.lock:
                self.engines[key]["pgid"] = pid
                self.engines[key]["processes"][pid] = status["created"]
                self.engines[key]["status"] = status
                self.engines[key]["heartbeat"] = time.monotonic()
                if self.engines[key]["state"] == "creating":
                    self.engines[key]["state"] = "created"
        except BaseException:
            # SglEngine's constructor loads no model or source buffers. Failure
            # here precedes init/publication, so this creation can be rolled back.
            ray.kill(actor, no_restart=True)
            with self.lock:
                self.engines[key]["state"] = "closed"
                for gpu in gpus:
                    self.claims.pop(gpu)
            raise
        finally:
            with self.lock:
                self.engines[key]["active"] -= 1
        return {"key": key, "gpu_ids": gpus}

    def heartbeat(self, key):
        with self.lock:
            item = self.engines[key]
            if item["state"] in ("retained", "terminating"):
                raise RuntimeError("Target session has stopped admitting work")
            item["heartbeat"] = time.monotonic()
        return None

    def call(self, key, method, args, kwargs):
        if method not in METHODS:
            raise ValueError("Unknown Target control method")
        if method == "shutdown":
            force = kwargs.pop("force", False)
            if type(force) is not bool:
                raise ValueError("Target shutdown force must be boolean")
            if force:
                if args or kwargs:
                    raise ValueError("Unknown forced Target shutdown arguments")
                return self._force_shutdown(key)
        with self.lock:
            item = self.engines[key]
            if item["state"] == "closed":
                if method == "shutdown":
                    return None
                if method == "get_status":
                    return {**item["status"], "initialized": False}
                if method in ("get_feature_contract_status", "prepare_feature_contract_status"):
                    return item["features"]
                raise RuntimeError("Target engine is closed")
            if item["state"] in ("creating", "retained", "closing", "terminating"):
                raise RuntimeError("Target engine is retained pending safe cleanup")
            if self.draining and method in ("init", "generate"):
                raise RuntimeError("Target service is draining")
            if method == "generate":
                # From admission onward a descriptor may have reached a reader,
                # even if the caller never received its reply.
                item["generation_admitted"] = True
            if method == "shutdown":
                if item["active"]:
                    raise RuntimeError("Target still has active calls; drain before shutdown")
                item["state"] = "closing"
            item["heartbeat"] = time.monotonic()
            item["active"] += 1
        try:
            if method == "init" and kwargs.get("mooncake_config") is not None:
                from aspec.config.mooncake_config import MooncakeConfig

                kwargs["mooncake_config"] = MooncakeConfig(**kwargs["mooncake_config"])
            result = ray.get(getattr(item["actor"], method).remote(*args, **kwargs), timeout=1800)
            with self.lock:
                if method == "get_status":
                    item["status"] = result
                if method in ("get_feature_contract_status", "prepare_feature_contract_status"):
                    item["features"] = result
                if method == "init" and item["state"] == "created":
                    item["state"] = "ready"
                if method == "shutdown" and item["state"] == "closing":
                    ray.kill(item["actor"], no_restart=True)
                    item["state"] = "closed"
                    for gpu in item["gpus"]:
                        self.claims.pop(gpu)
                    logger.info("Target session %s engine %s closed", item["session"], key)
            return result
        except BaseException as error:
            with self.lock:
                # Timing out a ray.get leaves the submitted operation in flight.
                if ((isinstance(error, ray.exceptions.GetTimeoutError)
                     and item["state"] in ("created", "ready", "closing"))
                        or (method == "shutdown" and item["state"] == "closing")):
                    item["state"] = "retained"
            raise
        finally:
            with self.lock:
                item["active"] -= 1
                item["heartbeat"] = time.monotonic()

    def _force_shutdown(self, key):
        """The caller has fenced readers; acknowledge only observed source exit."""
        with self.lock:
            item = self.engines[key]
            if item["state"] == "closed":
                return None
            if item["state"] == "terminating":
                raise RuntimeError("Target shutdown is already in progress")
            if not item["processes"]:
                item["state"] = "retained"
                raise RuntimeError("Target source process identities are unknown")
            observe_process_tree(item["processes"])
            processes = dict(item["processes"])
            item["state"] = "terminating"
            item["active"] += 1
        try:
            ray.kill(item["actor"], no_restart=True)
            stop_owned_processes(processes)
            wait_for_process_group_exit(item["pgid"])
            with self.lock:
                item["state"] = "closed"
                item["terminal_reason"] = ("readers_fenced_sources_exited" if item["generation_admitted"]
                                           else "unused_sources_exited")
                for gpu in item["gpus"]:
                    self.claims.pop(gpu, None)
        except BaseException:
            with self.lock:
                item["state"] = "retained"
            raise
        finally:
            with self.lock:
                item["active"] -= 1
                item["heartbeat"] = time.monotonic()
        return None

    def reap_disconnected(self, *, expiry=30):
        """Reclaim unused or reader-fenced sources after disconnect or during drain."""
        with self.lock:
            candidates = []
            for key, item in self.engines.items():
                if item["state"] == "closed":
                    continue
                observe_process_tree(item["processes"])
                if item["state"] in ("creating", "terminating"):
                    continue
                expired = time.monotonic() - item["heartbeat"] > expiry
                if expired:
                    # Stop new admissions as soon as the caller disconnects.
                    item["state"] = "retained"
                elif not self.draining or item["state"] == "closing":
                    continue
                # A live client may still finish shutdown while its readers exit.
                # Once safe, stop admission so polling cannot prolong the drain.
                if not item["generation_admitted"] or readers_exited(item["owners"], self.node_ip):
                    item["state"] = "retained"
                    if expired or not item["active"]:
                        candidates.append(key)
        for key in candidates:
            try:
                self._force_shutdown(key)
            except Exception:
                logger.exception("Could not reclaim Target %s; retaining its lease", key)

    def idle(self):
        with self.lock:
            return not self.claims


def observe_process_tree(identities):
    import psutil

    for pid, created in list(identities.items()):
        try:
            process = psutil.Process(pid)
            if process.create_time() == created:
                for child in process.children(recursive=True):
                    try:
                        identities[child.pid] = child.create_time()
                    except psutil.NoSuchProcess:
                        pass
        except psutil.NoSuchProcess:
            pass


def session_environment(args):
    """Native scheduler children inherit this session, never the service's job."""
    contracts = getattr(args, "aspec_shared_consumer_contracts", None) or {}
    return {
        "ASPEC_RUN_ID": args.aspec_run_id,
        "ASPEC_ATTEMPT_ID": args.aspec_attempt_id,
        "ASPEC_SHARED_SOURCE": str(int(bool(getattr(args, "aspec_shared_source", False)))),
        "ASPEC_SHARED_FEATURE_VIEWS": str(int(len(set(contracts.values())) > 1)),
    }


def stop_owned_processes(identities):
    """Called only after the read fence; never discover or stop unrelated PIDs."""
    import psutil

    def alive():
        processes = []
        for pid, created in identities.items():
            try:
                process = psutil.Process(pid)
                if (process.create_time() == created
                        and process.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)):
                    processes.append(process)
            except psutil.NoSuchProcess:
                pass
        return processes

    # These processes belong to Ray/SGLang, not this service's child-reaping
    # API. Observe exit directly, including containers without usable pidfds.
    for terminate in (True, False):
        for process in alive():
            try:
                if terminate:
                    process.terminate()
                else:
                    process.kill()
            except psutil.NoSuchProcess:
                pass
        deadline = time.monotonic() + 5
        while alive() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not alive():
            return
    if alive():
        raise RuntimeError("Owned Target source process exit remains unverified")


def wait_for_process_group_exit(pgid):
    """Ray kills the worker group on exit, including scheduler children forked during teardown."""
    deadline = time.monotonic() + 10
    while True:
        try:
            os.killpg(pgid, 0)  # Probe only. PID reuse can delay release, never signal another job.
        except ProcessLookupError:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("Target worker process group exit remains unverified")
        time.sleep(0.05)


def http_server(service, host, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, status, result):
            data = json.dumps({"protocol": PROTOCOL, "result": encode(result)}, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                logger.debug("Target caller disconnected before reading its response")

        def do_GET(self):
            if self.path != "/v1/info":
                self.respond(404, "Unknown Target endpoint")
                return
            self.respond(200, service.info())

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    raise ValueError("Invalid Target request size")
                value = decode(json.loads(self.rfile.read(length)))
                if self.path == "/v1/engines":
                    result = service.create(value)
                elif self.path == "/v1/heartbeat":
                    result = service.heartbeat(value["key"])
                elif self.path == "/v1/call":
                    result = service.call(value["key"], value["method"], value["args"], value["kwargs"])
                else:
                    raise ValueError("Unknown Target endpoint")
                self.respond(200, result)
            except Exception as error:
                logger.exception("Target control request failed")
                self.respond(409, f"{type(error).__name__}: {error}")

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main():
    from omegaconf import OmegaConf

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run Target in this environment; Draft connects by address.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--address", help="Listening http://host:port; defaults to the config or localhost:30000")
    args = parser.parse_args()
    config = OmegaConf.load(args.config) if args.config else OmegaConf.create({})
    address = validate_address(args.address or OmegaConf.select(
        config, "inference.target_service.address", default="http://127.0.0.1:30000"))
    model = OmegaConf.select(config, "model.target_model_path")
    devices = visible_gpu_ids()
    if not devices:
        parser.error("Select allocated Target GPUs with CUDA_VISIBLE_DEVICES")
    runtime = inventory()  # Check this environment and patch before starting processes.
    service = TargetService(model=model, devices=devices, runtime=runtime)
    parsed = urlsplit(address)
    server = http_server(service, parsed.hostname, parsed.port)
    stop = threading.Event()

    def stopping(*_):
        service.draining = True
        stop.set()

    previous = {sig: signal.signal(sig, stopping) for sig in (signal.SIGINT, signal.SIGTERM)}
    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
    temporary = tempfile.mkdtemp(prefix="aspec-t-", dir="/tmp")
    try:
        ray.init(address="local", num_gpus=0, num_cpus=min(8, os.cpu_count() or 1),
                 include_dashboard=False, object_store_memory=256 * 1024 * 1024,
                 _temp_dir=temporary)
        # Ray may replace the process SIGTERM handler during initialization.
        for sig in previous:
            signal.signal(sig, stopping)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        print(json.dumps({"state": "ready", "address": address, "protocol": PROTOCOL,
                          "python": runtime["python"], "pid": os.getpid()}), flush=True)
        while not stop.is_set() or not service.idle():
            time.sleep(1)
            service.reap_disconnected()
        server.shutdown()
        worker.join()
    finally:
        # Active sources must survive until readers are fenced, including during
        # signal handling. Never let Ray's implicit driver exit reclaim them.
        while not service.idle():
            try:
                service.draining = True
                service.reap_disconnected()
                time.sleep(1)
            except BaseException:
                logger.exception("Target cleanup is incomplete; retaining the service")
        server.server_close()
        ray.shutdown()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
