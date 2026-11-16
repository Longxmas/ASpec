# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Optional local supervision of the same external Target service.

Inspired by SpecForge f8cc3dc managed-local readiness and process ownership.
ASpec keeps its full checkpoint/resume path and its existing reader fence.
"""
import json
import logging
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

from aspec.inference.service_protocol import request
from aspec.runtime_control import termination_signal_handlers
from aspec.utils.misc import visible_gpu_ids

logger = logging.getLogger(__name__)


@contextmanager
def managed_target(args, *, draft_gpu_count=None):
    settings = getattr(args, "target_service", None) or {}
    if settings.get("mode", "external") != "managed-local":
        yield
        return
    address = settings["address"]
    if urlsplit(address).hostname not in ("localhost", "127.0.0.1"):
        raise ValueError("managed-local Target must listen on IPv4 loopback (127.0.0.1 or localhost)")
    try:
        request(address, "/v1/info", timeout=2)
    except OSError:
        pass
    else:
        raise RuntimeError("managed-local refuses to replace an existing Target service")
    devices = visible_gpu_ids()
    draft_count = draft_gpu_count or args.training_num_nodes * args.training_num_gpus_per_node
    target_count = args.inference_num_gpus
    if len(devices) != draft_count + target_count:
        raise ValueError("managed-local requires CUDA_VISIBLE_DEVICES to select exactly the Draft and Target GPUs")
    if getattr(args, "placement_strategy", "training_first") == "inference_first":
        target_devices, draft_devices = devices[:target_count], devices[target_count:]
    else:
        draft_devices, target_devices = devices[:draft_count], devices[draft_count:]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, target_devices))
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config = root / "target-service.json"
    config.write_text(json.dumps({"model": {"target_model_path": args.target_model_path}}))
    command = [settings["python_executable"], "-m", "aspec._target_worker", "-m",
               "aspec.inference.service", "--config", str(config), "--address", address]
    log_path = root / "target-service.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=environment)
        ready = False
        prior_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            deadline = time.monotonic() + float(getattr(args, "sglang_init_timeout", 300))
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Target exited before readiness; see {log_path}")
                try:
                    info = request(address, "/v1/info", timeout=2)
                    if info.get("draining"):
                        raise RuntimeError("Target is draining during startup")
                    if info.get("pid") != process.pid:
                        raise RuntimeError("Target address belongs to another process")
                    ready = True
                    break
                except OSError:
                    time.sleep(0.25)
            if not ready:
                raise TimeoutError(f"Target readiness timed out; see {log_path}")
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, draft_devices))
            yield
        finally:
            if prior_devices is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = prior_devices
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                with termination_signal_handlers(defer=True):
                    # Target retains disconnected source storage until its
                    # registered read processes have observably exited. The
                    # supervisor must not bypass that guard with SIGKILL.
                    while process.poll() is None:
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            logger.info("Waiting for owned Target cleanup; see %s", log_path)
            if ready and process.returncode != 0:
                logger.error("Owned Target exited with %s; see %s", process.returncode, log_path)
