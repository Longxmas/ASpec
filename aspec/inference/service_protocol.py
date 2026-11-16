# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Versioned JSON control plane; feature tensors stay in Mooncake.

Inspired by SpecForge f8cc3dc's external capture adapter. ASpec retains its
SglEngine, feature contracts and transport rather than copying that adapter.
"""
import json
from dataclasses import asdict, is_dataclass
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

PROTOCOL = "aspec.target-service/v2"
MAX_BODY = 64 * 1024 * 1024
METHODS = frozenset({
    "init", "generate",
    "get_node_ip", "find_free_port", "get_status", "get_mooncake_config",
    "get_feature_contract_status", "prepare_feature_contract_status", "health_check", "shutdown",
})

# Only settings consumed by SglEngine and its source/transport contracts belong
# to this interface. Optimizers, datasets, logging credentials and driver
# objects remain in Draft. Unknown Draft settings do not change this protocol.
TARGET_ARGUMENTS = frozenset("""
target_model_path trust_remote_code aux_hidden_states_layers store_last_hidden_states native_context_length
draft_algorithm mooncake_hidden_dim
sglang_mem_fraction_static sglang_pp_size sglang_nnodes sglang_tp_size
sglang_extra_args sglang_allow_context_length_override sglang_enable_multimodal
sglang_dist_init_addr sglang_dist_timeout
target_prefill_memory_optimized max_seq_length max_concurrent_batches inference_batch_size
attention_backend sp_ring_size sp_ulysses_size ttt_length draft_context_parallel_size
draft_context_shard_alignment mooncake_gpu_ring_slots mooncake_source_gpu_ring_slots
mooncake_feature_affinity mooncake_protocol
mooncake_allow_host_fallback mooncake_data_plane mooncake_get_retry_max_wait_seconds
aspec_feature_contract aspec_context_contract aspec_run_id aspec_attempt_id
aspec_read_owners aspec_shared_source aspec_shared_consumer_contracts
timeline_enabled timeline_output_dir timeline_run_id timeline_flush_every output_dir cluster_mode
""".split())


def validate_target_tp(nnodes, node_rank=0):
    if nnodes != 1 or node_rank != 0:
        raise ValueError("Target TP groups must be node-local (inference.sglang.nnodes=1, node_rank=0)")


def validate_address(value):
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
        raise ValueError("Target address must be an http://host:port origin")
    if ":" in parsed.hostname:
        raise ValueError("Target address does not support IPv6 literals; use an IPv4 address or hostname")
    if parsed.port is None:
        raise ValueError("Target address must include a port")
    return value.rstrip("/")


def encode(value):
    """Encode only explicit data types, never Python objects or feature tensors."""
    from omegaconf import OmegaConf

    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if is_dataclass(value):
        return encode(asdict(value))
    # Importing torch is confined to actual control payload conversion.
    import torch

    if isinstance(value, torch.dtype):
        return {"$dtype": str(value).removeprefix("torch.")}
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.dtype not in (torch.int64, torch.int32):
            raise TypeError("Only CPU token IDs may cross the Target control plane")
        return {"$tokens": value.tolist(), "dtype": str(value.dtype).removeprefix("torch.")}
    raise TypeError(f"Unsupported Target control value: {type(value).__name__}")


def decode(value):
    if isinstance(value, list):
        return [decode(v) for v in value]
    if isinstance(value, dict):
        if set(value) == {"$dtype"}:
            import torch

            dtype = getattr(torch, value["$dtype"], None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError("Invalid wire dtype")
            return dtype
        if set(value) == {"$tokens", "dtype"}:
            import torch

            if value["dtype"] not in ("int64", "int32"):
                raise ValueError("Invalid token dtype")
            return torch.tensor(value["$tokens"], dtype=getattr(torch, value["dtype"]))
        return {k: decode(v) for k, v in value.items()}
    return value


def request(address, path, payload=None, *, timeout=1200):
    address = validate_address(address)
    body = None if payload is None else json.dumps(encode(payload), allow_nan=False).encode()
    if body is not None and len(body) > MAX_BODY:
        raise ValueError("Target control request exceeds the size limit")
    req = Request(address + path, data=body, headers={"Content-Type": "application/json"})
    # Cluster-internal control endpoints must not be sent through HTTP proxies.
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as response:
            raw = response.read(MAX_BODY + 1)
    except HTTPError as error:
        detail = error.read(8192).decode(errors="replace")
        raise RuntimeError(f"Target {address}{path}: {detail}") from error
    if len(raw) > MAX_BODY:
        raise ValueError("Target control response exceeds the size limit")
    result = json.loads(raw)
    if result.get("protocol") != PROTOCOL:
        raise RuntimeError("Target service protocol differs")
    return decode(result["result"])


def engine_arguments(args):
    return {key: encode(value) for key, value in vars(args).items() if key in TARGET_ARGUMENTS}
