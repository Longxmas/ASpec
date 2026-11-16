# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Draft-owned Ray facade for an independently launched Target service."""
import asyncio
import threading
import uuid

import ray

from aspec.inference.service_protocol import PROTOCOL, engine_arguments, request
from aspec.inference.target_runtime import release_version


class TargetServiceClient:
    def __init__(self, *, address, args, rank, num_gpus_per_engine,
                 node_ip=None, node_rank=0, engine_group=0):
        self.address = address
        self.key = uuid.uuid4().hex
        self.timeout = float(getattr(args, "sglang_init_timeout", 1200)) + 60
        self._closed = threading.Event()
        info = request(address, "/v1/info", timeout=30)
        if info["version"] != release_version():
            raise RuntimeError("Target service has a different ASpec release version")
        if node_ip and info["node_ip"] != node_ip:
            raise RuntimeError("Target service is not on the planned node")
        self._fail_stop_node = None
        if (getattr(args, "cluster_mode", "single_node") == "static_multi_node"
                and getattr(args, "mooncake_feature_affinity", "any") == "node_local_required"):
            node = next((n for n in ray.nodes()
                         if n.get("Alive") and n["NodeManagerAddress"] == info["node_ip"]), None)
            if node is None:
                raise RuntimeError("Target node is missing from the planned Draft cluster")
            self._fail_stop_node = node["NodeID"]
        request(address, "/v1/engines", {
            "key": self.key, "version": release_version(), "arguments": engine_arguments(args),
            "rank": rank, "node_ip": node_ip,
            "num_gpus_per_engine": num_gpus_per_engine, "node_rank": node_rank,
            "engine_group": engine_group,
        }, timeout=self.timeout)
        self._heartbeat = threading.Thread(target=self._keepalive, daemon=True)
        self._heartbeat.start()

    def _keepalive(self):
        while not self._closed.wait(5):
            try:
                request(self.address, "/v1/heartbeat", {"key": self.key}, timeout=5)
            except Exception:
                # Calls fail visibly; a heartbeat never retries a generation request.
                pass

    async def _call(self, method, *args, **kwargs):
        return await asyncio.to_thread(request, self.address, "/v1/call", {
            "key": self.key, "method": method, "args": args, "kwargs": kwargs,
        }, timeout=self.timeout)

    async def init(self, **kwargs):
        return await self._call("init", **kwargs)

    async def generate(self, *args, **kwargs):
        # ObjectRefs are resolved inside Draft's Ray job, never serialized to Target.
        if isinstance(kwargs.get("input_ids_ref"), ray.ObjectRef):
            kwargs["input_ids_ref"] = await kwargs["input_ids_ref"]
        return await self._call("generate", *args, **kwargs)

    async def get_node_ip(self):
        return await self._call("get_node_ip")

    async def find_free_port(self, *args, **kwargs):
        return await self._call("find_free_port", *args, **kwargs)

    async def get_status(self):
        status = await self._call("get_status")
        status["target_service"] = {"address": self.address, "protocol": PROTOCOL}
        return status

    async def get_mooncake_config(self):
        from aspec.config.mooncake_config import MooncakeConfig

        value = await self._call("get_mooncake_config")
        return MooncakeConfig(**value) if value else None

    async def get_feature_contract_status(self):
        return await self._call("get_feature_contract_status")

    async def prepare_feature_contract_status(self):
        return await self._call("prepare_feature_contract_status")

    async def health_check(self, **kwargs):
        return await self._call("health_check", **kwargs)

    async def shutdown(self, *, force=False):
        kwargs = {"force": True} if force else {}
        try:
            result = await self._call("shutdown", **kwargs)
        except OSError:
            if not force or self._fail_stop_node is None:
                raise
            node = next((n for n in ray.nodes() if n["NodeID"] == self._fail_stop_node), None)
            if node is None or node.get("Alive"):
                raise
            # Only the declared node-local fail-stop domain may be retired;
            # source process exit on an unreachable node remains unobserved.
            result = {"state": "node_local_domain_lost", "node_id": self._fail_stop_node,
                      "physical_memory_unobserved": True}
        self._closed.set()
        return result
