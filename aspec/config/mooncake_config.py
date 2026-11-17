# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Tuple
from urllib.parse import urlparse

from aspec.config.draft_execution import is_direct_cp_backend
from aspec.transfer.mooncake.helpers import (
    calculate_eagle3_buffer_size,
    calculate_eagle3_context_shard_buffer_size,
)


def resolve_rdma_routes(
    node_addresses,
    *,
    device_name: str | None,
    port: int | None,
    gid_index: int | None,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Resolve one explicit physical RDMA route per planned node.

    The top-level values are defaults for homogeneous deployments.  Per-node
    entries may override any field, so the single-rail capability does not
    depend on identical Linux device names or GID indexes across machines.
    """

    nodes = [str(node) for node in node_addresses]
    if overrides is not None and not isinstance(overrides, Mapping):
        raise ValueError("mooncake.rdma_routes must map node addresses to routes")
    override_map = dict(overrides or {})
    unknown = sorted(set(override_map) - set(nodes))
    if unknown:
        raise ValueError(
            "mooncake.rdma_routes contains nodes outside the planned membership: "
            f"{unknown}"
        )

    resolved: dict[str, dict[str, Any]] = {}
    defaults = {
        "device_name": device_name,
        "port": port,
        "gid_index": gid_index,
    }
    for node in nodes:
        configured = override_map.get(node, {})
        if not isinstance(configured, Mapping):
            raise ValueError(f"mooncake.rdma_routes[{node!r}] must be a mapping")
        unknown_fields = sorted(set(configured) - set(defaults))
        if unknown_fields:
            raise ValueError(
                f"mooncake.rdma_routes[{node!r}] contains unknown fields: "
                f"{unknown_fields}"
            )
        route = {
            key: configured.get(key, default)
            for key, default in defaults.items()
        }
        route_device = route["device_name"]
        if not isinstance(route_device, str) or not route_device.strip():
            raise ValueError(f"RDMA route for node {node} requires device_name")
        if "," in route_device:
            raise ValueError(
                f"RDMA route for node {node} must select exactly one device"
            )
        if type(route["port"]) is not int or route["port"] < 1:
            raise ValueError(f"RDMA route for node {node} requires port >= 1")
        if type(route["gid_index"]) is not int or route["gid_index"] < 0:
            raise ValueError(
                f"RDMA route for node {node} requires a non-negative gid_index"
            )
        resolved[node] = {
            "device_name": route_device.strip(),
            "port": route["port"],
            "gid_index": route["gid_index"],
        }
    return resolved


@dataclass
class MooncakeConfig:
    """
    Configuration for Mooncake Store client.

    Memory parameters (accept ``str`` like ``"4GB"`` or ``int`` in bytes):
        global_segment_size: Memory this client CONTRIBUTES to the distributed pool.
            Other clients can store objects in this space. Inference workers should
            set this large enough to hold multiple outputs (~2-4GB per batch
            for large models).

        local_buffer_size: Memory for RECEIVING data via Get() operations.
            This is the buffer for reading objects from other clients. Trainers
            should set this large enough to receive hidden states (~512MB-2GB).

        gpu_buffer_size: Memory for accepting tensors from store directly into GPU memory.
            This is auto determined based on the training config.

    Note: The metadata_server defaults to the Mooncake Master's built-in HTTP
    metadata server (same host as master, port 8090).
    """

    local_hostname: str = "localhost"
    metadata_server: str = "http://localhost:8090/metadata"
    master_server_address: str = "localhost:50051"
    global_segment_size: str | int = 4 * 1024 * 1024 * 1024
    local_buffer_size: str | int = 512 * 1024 * 1024
    protocol: str = "tcp"
    device_name: str = ""
    # A production RDMA route is explicit rather than inherited from a worker
    # shell.  Mooncake consumes ``MC_GID_INDEX``; keeping the selected port and
    # GID in canonical ASpec configuration lets preflight and Evidence
    # validate the same route that TransferEngine will initialize.
    rdma_port: int = 1
    rdma_gid_index: int | None = None
    # Optional per-node overrides keep the logical single-rail contract from
    # requiring identical Linux HCA names or GID table indexes on every host.
    # Each actor resolves its own entry before initializing Mooncake.
    rdma_routes: dict[str, dict[str, Any]] | None = None
    # ``store`` is the host-backed object path with asynchronous pinned-buffer
    # staging. ``transfer_engine``
    # exchanges descriptors through the controller and moves registered CUDA
    # payloads directly with Mooncake.
    data_plane: str = "store"
    # ``node_local_required`` binds every produced feature to a Draft consumer
    # on the same physical node.  It is a scheduler/placement contract rather
    # than a best-effort transport hint.
    feature_affinity: str = "any"
    gpu_buffer_size: str | int | None = None
    gpu_ring_slots: int = 2
    source_gpu_ring_slots: int | None = None
    destination_gpu_ring_slots: int | None = None
    require_nvlink: bool = False
    # Historical configuration name. For ``transfer_engine`` this enables
    # registered CUDA source/destination rings. Network GPUDirect is claimed
    # only when the selected protocol is RDMA (or intra-node NVLink).
    enable_gpu_direct: bool = False
    # This governs fallback from a requested GPU-direct path. Selecting Store
    # with enable_gpu_direct=false uses host staging directly, including when
    # allow_host_fallback=false.
    allow_host_fallback: bool = True
    replica_num: int = 1
    enable_hard_pin: bool = False
    host_buffer_size: str | int | None = None
    get_batch_size: int = 1
    max_seq_len: int = 8192
    hidden_dim: int = 4096
    num_aux_layers: int = 3
    include_mtp_hidden_states: bool = False
    async_put_pool_size: int | None = None
    store_full_error_codes: Tuple[int, ...] = (-200,)
    store_full_wait_seconds: float = 0.5
    store_full_log_interval_seconds: float = 5.0
    store_full_max_wait_seconds: float = 0.0
    get_retry_wait_seconds: float = 0.5
    get_retry_log_interval_seconds: float = 10.0
    get_retry_max_wait_seconds: float = 60.0
    # Mooncake defaults both limits to 30 seconds.  That is too short for a
    # valid long-context feature over a contended multi-node TCP path (a
    # single TransferEngine slice can legitimately take longer), so ASpec
    # makes the effective native limits explicit and observable.
    transfer_timeout_seconds: int = 300
    tcp_progress_timeout_seconds: int = 300
    kv_lease_ttl_s: float = 5.0  # Mooncake master lease TTL only; not used for deletion timing

    def __post_init__(self):
        self.data_plane = self.data_plane.lower()
        self.protocol = self.protocol.lower()
        self.feature_affinity = self.feature_affinity.lower()
        # OmegaConf dotlist input represents an explicitly empty value as
        # ``None``.  For TCP that is equivalent to the field being omitted;
        # normalize it at the configuration boundary before exporting env.
        if self.device_name is None:
            self.device_name = ""
        if self.rdma_routes is not None:
            if not isinstance(self.rdma_routes, Mapping):
                raise ValueError("mooncake.rdma_routes must map node addresses to routes")
            normalized_routes = {}
            for node, route in self.rdma_routes.items():
                if not isinstance(route, Mapping):
                    raise ValueError(
                        f"mooncake.rdma_routes[{str(node)!r}] must be a mapping"
                    )
                normalized_routes[str(node)] = dict(route)
            self.rdma_routes = resolve_rdma_routes(
                normalized_routes,
                device_name=self.device_name,
                port=self.rdma_port,
                gid_index=self.rdma_gid_index,
                overrides=normalized_routes,
            )
            route = self.rdma_routes.get(self.local_hostname)
            if route is not None:
                self.device_name = route["device_name"]
                self.rdma_port = route["port"]
                self.rdma_gid_index = route["gid_index"]
        if self.data_plane not in ("store", "transfer_engine"):
            raise ValueError(
                "mooncake.data_plane must be 'store' or 'transfer_engine', "
                f"got {self.data_plane!r}"
            )
        if self.feature_affinity not in ("any", "node_local_required"):
            raise ValueError(
                "mooncake.feature_affinity must be 'any' or "
                f"'node_local_required', got {self.feature_affinity!r}"
            )
        if not isinstance(self.rdma_port, int) or self.rdma_port < 1:
            raise ValueError("mooncake.rdma_port must be at least 1")
        if self.rdma_gid_index is not None and (
            not isinstance(self.rdma_gid_index, int) or self.rdma_gid_index < 0
        ):
            raise ValueError("mooncake.rdma_gid_index must be non-negative when set")
        if self.gpu_ring_slots < 1:
            raise ValueError("mooncake.gpu_ring_slots must be at least 1")
        if not isinstance(self.transfer_timeout_seconds, int) or isinstance(
            self.transfer_timeout_seconds, bool
        ) or self.transfer_timeout_seconds < 1:
            raise ValueError("mooncake.transfer_timeout_seconds must be a positive integer")
        if not isinstance(self.tcp_progress_timeout_seconds, int) or isinstance(
            self.tcp_progress_timeout_seconds, bool
        ) or self.tcp_progress_timeout_seconds < 1:
            raise ValueError(
                "mooncake.tcp_progress_timeout_seconds must be a positive integer"
            )
        for role in ("source", "destination"):
            slots = getattr(self, f"{role}_gpu_ring_slots")
            if slots is not None and slots < 1:
                raise ValueError(
                    f"mooncake.{role}_gpu_ring_slots must be at least 1 when set"
                )
        if self.data_plane == "transfer_engine":
            if not self.enable_gpu_direct:
                raise ValueError("transfer_engine data_plane requires enable_gpu_direct=true")
            if self.allow_host_fallback:
                raise ValueError("transfer_engine data_plane requires allow_host_fallback=false")
        if self.require_nvlink and (
            self.data_plane != "transfer_engine" or self.protocol != "nvlink_intra"
        ):
            raise ValueError(
                "require_nvlink=true requires data_plane=transfer_engine and "
                "protocol=nvlink_intra"
            )

        # Coerce size fields: accept str ("4GB") or int
        for field_name in (
            "global_segment_size",
            "local_buffer_size",
            "host_buffer_size",
            "gpu_buffer_size",
        ):
            val = getattr(self, field_name)
            if isinstance(val, str):
                setattr(self, field_name, self.parse_size(val))

        if self.host_buffer_size is None:
            # put() stages one sample at a time, so batch_size=1
            self.host_buffer_size = calculate_eagle3_buffer_size(
                max_seq_len=self.max_seq_len,
                batch_size=1,
                hidden_dim=self.hidden_dim,
                num_aux_layers=self.num_aux_layers,
                safety_margin=2.0,
                include_mtp_hidden_states=self.include_mtp_hidden_states,
            )

        if self.async_put_pool_size is None:
            self.async_put_pool_size = 1

        if self.gpu_buffer_size is None and self.enable_gpu_direct:
            # TransferEngine publishes one descriptor per logical sample. MBS
            # is assembled by fetching and releasing descriptors individually.
            buffer_batch_size = 1 if self.data_plane == "transfer_engine" else self.get_batch_size
            self.gpu_buffer_size = calculate_eagle3_buffer_size(
                max_seq_len=self.max_seq_len,
                batch_size=buffer_batch_size,
                hidden_dim=self.hidden_dim,
                num_aux_layers=self.num_aux_layers,
                safety_margin=1.0 if self.data_plane == "transfer_engine" else 1.1,
                include_mtp_hidden_states=self.include_mtp_hidden_states,
            )

    @staticmethod
    def _infer_num_aux_layers(args) -> int:
        contract = getattr(args, "aspec_feature_contract", None)
        if contract is not None:
            return len(contract["aux_hidden_state_layers"])
        draft_cfg = getattr(args, "draft_model_config_obj", None)
        if draft_cfg is not None:
            target_layer_ids = getattr(draft_cfg, "target_layer_ids", None)
            if target_layer_ids:
                return len(target_layer_ids)
            num_target_layers = getattr(draft_cfg, "num_target_layers", None)
            if num_target_layers:
                return int(num_target_layers)
        aux_layers = getattr(args, "aux_hidden_states_layers", None)
        if aux_layers:
            return len(aux_layers)
        return 3

    @classmethod
    def from_flat_args(cls, args) -> "MooncakeConfig":
        """Create config from a flat args namespace (mooncake_* prefixed fields).

        Handles:
        - metadata_server URL construction from metadata_port if metadata_server is not set
        - local_hostname auto-resolution via RayActor.get_node_ip()
        - Size string parsing (handled automatically by __post_init__)
        """
        from aspec.ray.ray_actor import RayActor

        master_server_address = getattr(args, "mooncake_master_server_address", None)
        metadata_port = getattr(args, "mooncake_metadata_port", None)
        metadata_server = getattr(args, "mooncake_metadata_server", None)

        if metadata_server is None and master_server_address is not None:
            master_host = master_server_address.split(":")[0]
            port = metadata_port if metadata_port is not None else 8090
            metadata_server = f"http://{master_host}:{port}/metadata"

        local_hostname = getattr(args, "mooncake_local_hostname", None)
        if local_hostname is None or local_hostname == "localhost":
            local_hostname = RayActor.get_node_ip()

        kwargs = {
            "local_hostname": local_hostname,
            "master_server_address": master_server_address or "localhost:50051",
            "global_segment_size": getattr(args, "mooncake_global_segment_size", "4GB"),
            "local_buffer_size": getattr(args, "mooncake_local_buffer_size", "512MB"),
            "host_buffer_size": getattr(args, "mooncake_host_buffer_size", None),
            "protocol": getattr(args, "mooncake_protocol", "tcp"),
            "device_name": getattr(args, "mooncake_device_name", ""),
            "rdma_port": getattr(args, "mooncake_rdma_port", 1),
            "rdma_gid_index": getattr(args, "mooncake_rdma_gid_index", None),
            "rdma_routes": getattr(args, "mooncake_rdma_routes", None),
            "data_plane": getattr(args, "mooncake_data_plane", "store"),
            "feature_affinity": getattr(args, "mooncake_feature_affinity", "any"),
            "gpu_buffer_size": getattr(args, "mooncake_gpu_buffer_size", None),
            "gpu_ring_slots": getattr(args, "mooncake_gpu_ring_slots", 2),
            "source_gpu_ring_slots": getattr(args, "mooncake_source_gpu_ring_slots", None),
            "destination_gpu_ring_slots": getattr(
                args, "mooncake_destination_gpu_ring_slots", None
            ),
            "require_nvlink": getattr(args, "mooncake_require_nvlink", False),
            "enable_gpu_direct": getattr(args, "mooncake_enable_gpu_direct", False),
            "allow_host_fallback": getattr(args, "mooncake_allow_host_fallback", True),
            "async_put_pool_size": getattr(
                args, "mooncake_async_put_pool_size", getattr(args, "inference_batch_size", 1)
            ),
            "get_batch_size": getattr(
                args, "mooncake_get_batch_size", getattr(args, "per_dp_rank_batch_size", 1)
            ),
            "kv_lease_ttl_s": getattr(args, "mooncake_kv_lease_ttl_s", 5.0),
            "enable_hard_pin": getattr(args, "mooncake_enable_hard_pin", False),
            "max_seq_len": getattr(
                args,
                "mooncake_max_seq_len",
                getattr(args, "max_seq_length", 8192),
            ),
            "hidden_dim": getattr(args, "mooncake_hidden_dim", 4096),
            "num_aux_layers": cls._infer_num_aux_layers(args),
            "include_mtp_hidden_states": "mtp_hidden_states" in (
                (getattr(args, "aspec_feature_contract", None) or {}).get("tensors", {})),
            "store_full_error_codes": tuple(
                getattr(args, "mooncake_store_full_error_codes", (-200,))
            ),
            "store_full_wait_seconds": getattr(
                args, "mooncake_store_full_wait_seconds", 0.5
            ),
            "store_full_log_interval_seconds": getattr(
                args, "mooncake_store_full_log_interval_seconds", 5.0
            ),
            "store_full_max_wait_seconds": getattr(
                args, "mooncake_store_full_max_wait_seconds", 0.0
            ),
            "get_retry_wait_seconds": getattr(
                args, "mooncake_get_retry_wait_seconds", 0.5
            ),
            "get_retry_log_interval_seconds": getattr(
                args, "mooncake_get_retry_log_interval_seconds", 10.0
            ),
            "get_retry_max_wait_seconds": getattr(
                args, "mooncake_get_retry_max_wait_seconds", 60.0
            ),
            "transfer_timeout_seconds": getattr(
                args, "mooncake_transfer_timeout_seconds", 300
            ),
            "tcp_progress_timeout_seconds": getattr(
                args, "mooncake_tcp_progress_timeout_seconds", 300
            ),
        }

        cp_size = int(getattr(args, "draft_context_parallel_size", 1))
        if (
            kwargs["gpu_buffer_size"] is None
            and str(kwargs["data_plane"]).lower() == "transfer_engine"
            and is_direct_cp_backend(getattr(args, "attention_backend", None))
            and cp_size > 1
        ):
            kwargs["gpu_buffer_size"] = calculate_eagle3_context_shard_buffer_size(
                max_seq_len=kwargs["max_seq_len"],
                hidden_dim=kwargs["hidden_dim"],
                context_parallel_size=cp_size,
                context_shard_alignment=int(
                    getattr(args, "draft_context_shard_alignment", 1)
                ),
                num_aux_layers=kwargs["num_aux_layers"],
            )

        if metadata_server is not None:
            kwargs["metadata_server"] = metadata_server

        return cls(**kwargs)

    def ring_slots_for_role(self, role: str) -> int:
        """Return the configured slot count for one P2P endpoint role."""

        if role not in ("source", "destination"):
            raise ValueError("Mooncake P2P role must be 'source' or 'destination'")
        configured = getattr(self, f"{role}_gpu_ring_slots")
        return self.gpu_ring_slots if configured is None else configured

    def export_env(self) -> None:
        """Export mooncake configuration as environment variables.

        Sets the environment variables that sglang and other mooncake clients
        read during initialization. The caller is responsible for setting
        ``local_hostname`` to the correct value beforehand.
        """
        os.environ["MOONCAKE_LOCAL_HOSTNAME"] = self.local_hostname
        os.environ["MOONCAKE_METADATA_SERVER"] = self.metadata_server
        os.environ["MOONCAKE_MASTER_SERVER"] = self.master_server_address
        os.environ["MOONCAKE_GLOBAL_SEGMENT_SIZE"] = str(self.global_segment_size)
        os.environ["MOONCAKE_LOCAL_BUFFER_SIZE"] = str(self.local_buffer_size)
        os.environ["MOONCAKE_HOST_BUFFER_SIZE"] = str(self.host_buffer_size)
        os.environ["MOONCAKE_NUM_AUX_LAYERS"] = str(self.num_aux_layers)
        os.environ["MOONCAKE_INCLUDE_MTP_HIDDEN_STATES"] = "1" if self.include_mtp_hidden_states else "0"
        os.environ["MOONCAKE_PROTOCOL"] = self.protocol
        os.environ["MOONCAKE_DEVICE_NAME"] = self.device_name
        os.environ["MOONCAKE_RDMA_PORT"] = str(self.rdma_port)
        if self.rdma_gid_index is not None:
            os.environ["MOONCAKE_RDMA_GID_INDEX"] = str(self.rdma_gid_index)
        else:
            os.environ.pop("MOONCAKE_RDMA_GID_INDEX", None)
        os.environ["MOONCAKE_DATA_PLANE"] = self.data_plane
        os.environ["MOONCAKE_FEATURE_AFFINITY"] = self.feature_affinity
        os.environ["MOONCAKE_GPU_RING_SLOTS"] = str(self.gpu_ring_slots)
        os.environ["MOONCAKE_SOURCE_GPU_RING_SLOTS"] = str(
            self.ring_slots_for_role("source")
        )
        os.environ["MOONCAKE_DESTINATION_GPU_RING_SLOTS"] = str(
            self.ring_slots_for_role("destination")
        )
        os.environ["MOONCAKE_REQUIRE_NVLINK"] = "1" if self.require_nvlink else "0"
        if self.gpu_buffer_size is not None:
            os.environ["MOONCAKE_GPU_BUFFER_SIZE"] = str(self.gpu_buffer_size)
        os.environ["MOONCAKE_ENABLE_GPU_DIRECT"] = "1" if self.enable_gpu_direct else "0"
        os.environ["MOONCAKE_ALLOW_HOST_FALLBACK"] = (
            "1" if self.allow_host_fallback else "0"
        )
        self.apply_env_defaults()
        if self.async_put_pool_size is not None:
            os.environ["MOONCAKE_ASYNC_PUT_POOL_SIZE"] = str(self.async_put_pool_size)
        os.environ["MOONCAKE_STORE_FULL_WAIT_SECONDS"] = str(self.store_full_wait_seconds)
        os.environ["MOONCAKE_STORE_FULL_LOG_INTERVAL_SECONDS"] = str(
            self.store_full_log_interval_seconds
        )
        os.environ["MOONCAKE_STORE_FULL_MAX_WAIT_SECONDS"] = str(self.store_full_max_wait_seconds)
        os.environ["MOONCAKE_GET_RETRY_WAIT_SECONDS"] = str(self.get_retry_wait_seconds)
        os.environ["MOONCAKE_GET_RETRY_LOG_INTERVAL_SECONDS"] = str(
            self.get_retry_log_interval_seconds
        )
        os.environ["MOONCAKE_GET_RETRY_MAX_WAIT_SECONDS"] = str(self.get_retry_max_wait_seconds)
        os.environ["MOONCAKE_TRANSFER_TIMEOUT_SECONDS"] = str(
            self.transfer_timeout_seconds
        )
        os.environ["MOONCAKE_TCP_PROGRESS_TIMEOUT_SECONDS"] = str(
            self.tcp_progress_timeout_seconds
        )
        os.environ["MOONCAKE_ENABLE_HARD_PIN"] = "1" if self.enable_hard_pin else "0"

    def apply_env_defaults(self) -> None:
        """Apply Mooncake process defaults that are needed before client setup."""
        os.environ["MC_TRANSFER_TIMEOUT"] = str(self.transfer_timeout_seconds)
        if self.protocol == "rdma":
            # Mooncake's native RDMA implementation consumes MC_IB_PORT and
            # MC_GID_INDEX.  The MOONCAKE_* names preserve the canonical route
            # when ASpec reconstructs its Python config in child processes.
            os.environ["MC_IB_PORT"] = str(self.rdma_port)
            if self.rdma_gid_index is not None:
                os.environ["MC_GID_INDEX"] = str(self.rdma_gid_index)
            else:
                # Do not let a stale worker-shell value silently become part
                # of a supposedly canonical RDMA route.
                os.environ.pop("MC_GID_INDEX", None)
        else:
            os.environ.pop("MC_IB_PORT", None)
            os.environ.pop("MC_GID_INDEX", None)
        if (
            self.protocol == "rdma"
            and self.data_plane == "transfer_engine"
            and not self.allow_host_fallback
        ):
            # Mooncake otherwise treats an RDMA request as a preference and
            # silently installs TCP when no usable HCA is visible.  ASpec's
            # transfer_engine contract is strict: a requested zero-copy RDMA
            # path must either activate or fail during setup.
            os.environ["MC_FORCE_HCA"] = "1"
        else:
            os.environ.pop("MC_FORCE_HCA", None)
        if self.protocol == "tcp":
            # TransferEngine otherwise auto-discovers HCAs and may select its
            # RDMA transport even though the canonical ASpec Plan says TCP.
            # Keep the native route aligned with the claimed protocol.
            os.environ["MC_FORCE_TCP"] = "1"
            os.environ["MC_TCP_PROGRESS_TIMEOUT_SEC"] = str(
                self.tcp_progress_timeout_seconds
            )
        else:
            # Ray workers are reused across cases, so a previous TCP run must
            # not silently force a later RDMA or NVLink run onto TCP.
            os.environ.pop("MC_FORCE_TCP", None)
            os.environ.pop("MC_TCP_PROGRESS_TIMEOUT_SEC", None)
        if self.protocol == "nvlink_intra":
            os.environ["MC_INTRANODE_NVLINK"] = "true"
            os.environ["MC_INTRA_NVLINK"] = "1"
            os.environ["MC_USE_NVLINK_IPC"] = "1"
            os.environ.pop("MC_FORCE_MNNVL", None)
        else:
            os.environ.pop("MC_INTRANODE_NVLINK", None)
            os.environ.pop("MC_INTRA_NVLINK", None)
            os.environ.pop("MC_USE_NVLINK_IPC", None)
        # Fix: https://github.com/kvcache-ai/Mooncake/issues/1986
        if self.protocol.lower() == "tcp" and "MC_STORE_MEMCPY" not in os.environ:
            # Mooncake's TCP-only memcpy fast path can segfault in same-host
            # multi-process get paths. Preserve an explicit user override.
            os.environ["MC_STORE_MEMCPY"] = "0"

    @classmethod
    def from_env(cls) -> "MooncakeConfig":
        """Create config from environment variables."""
        master_host = os.getenv("MOONCAKE_MASTER_HOST", "localhost")
        master_port = cls._env_port("MOONCAKE_MASTER_PORT", "50051")
        metadata_port = cls._env_port("MOONCAKE_METADATA_PORT", "8090")
        store_full_wait_seconds = float(os.getenv("MOONCAKE_STORE_FULL_WAIT_SECONDS", "0.5"))
        store_full_log_interval_seconds = float(
            os.getenv("MOONCAKE_STORE_FULL_LOG_INTERVAL_SECONDS", "5.0")
        )
        store_full_max_wait_seconds = float(
            os.getenv("MOONCAKE_STORE_FULL_MAX_WAIT_SECONDS", "0.0")
        )
        get_retry_wait_seconds = float(os.getenv("MOONCAKE_GET_RETRY_WAIT_SECONDS", "0.2"))
        get_retry_log_interval_seconds = float(
            os.getenv("MOONCAKE_GET_RETRY_LOG_INTERVAL_SECONDS", "5.0")
        )
        get_retry_max_wait_seconds = float(os.getenv("MOONCAKE_GET_RETRY_MAX_WAIT_SECONDS", "5.0"))
        transfer_timeout_seconds = int(
            os.getenv(
                "MOONCAKE_TRANSFER_TIMEOUT_SECONDS",
                os.getenv("MC_TRANSFER_TIMEOUT", "300"),
            )
        )
        tcp_progress_timeout_seconds = int(
            os.getenv(
                "MOONCAKE_TCP_PROGRESS_TIMEOUT_SECONDS",
                os.getenv("MC_TCP_PROGRESS_TIMEOUT_SEC", "300"),
            )
        )

        host_buffer_env = os.getenv("MOONCAKE_HOST_BUFFER_SIZE")
        host_buffer_size = int(host_buffer_env) if host_buffer_env is not None else None

        pool_size_env = os.getenv("MOONCAKE_ASYNC_PUT_POOL_SIZE")
        async_put_pool_size = int(pool_size_env) if pool_size_env is not None else None

        gpu_buffer_env = os.getenv("MOONCAKE_GPU_BUFFER_SIZE")
        gpu_buffer_size = int(gpu_buffer_env) if gpu_buffer_env is not None else None
        source_slots_env = os.getenv("MOONCAKE_SOURCE_GPU_RING_SLOTS")
        destination_slots_env = os.getenv("MOONCAKE_DESTINATION_GPU_RING_SLOTS")

        num_aux_layers = int(os.getenv("MOONCAKE_NUM_AUX_LAYERS", "3"))

        return cls(
            local_hostname=os.getenv("MOONCAKE_LOCAL_HOSTNAME", "localhost"),
            metadata_server=os.getenv(
                "MOONCAKE_METADATA_SERVER",
                f"http://{master_host}:{metadata_port}/metadata",
            ),
            master_server_address=os.getenv(
                "MOONCAKE_MASTER_SERVER", f"{master_host}:{master_port}"
            ),
            global_segment_size=int(
                os.getenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", str(4 * 1024 * 1024 * 1024))
            ),
            local_buffer_size=int(os.getenv("MOONCAKE_LOCAL_BUFFER_SIZE", str(512 * 1024 * 1024))),
            host_buffer_size=host_buffer_size,
            num_aux_layers=num_aux_layers,
            include_mtp_hidden_states=os.getenv("MOONCAKE_INCLUDE_MTP_HIDDEN_STATES", "0") == "1",
            async_put_pool_size=async_put_pool_size,
            protocol=os.getenv("MOONCAKE_PROTOCOL", "tcp"),
            device_name=os.getenv("MOONCAKE_DEVICE_NAME", ""),
            rdma_port=int(os.getenv("MOONCAKE_RDMA_PORT", "1")),
            rdma_gid_index=(
                int(os.environ["MOONCAKE_RDMA_GID_INDEX"])
                if "MOONCAKE_RDMA_GID_INDEX" in os.environ
                else None
            ),
            data_plane=os.getenv("MOONCAKE_DATA_PLANE", "store"),
            feature_affinity=os.getenv("MOONCAKE_FEATURE_AFFINITY", "any"),
            gpu_buffer_size=gpu_buffer_size,
            gpu_ring_slots=int(os.getenv("MOONCAKE_GPU_RING_SLOTS", "2")),
            source_gpu_ring_slots=(
                int(source_slots_env) if source_slots_env is not None else None
            ),
            destination_gpu_ring_slots=(
                int(destination_slots_env) if destination_slots_env is not None else None
            ),
            require_nvlink=os.getenv("MOONCAKE_REQUIRE_NVLINK", "0") == "1",
            enable_gpu_direct=os.getenv("MOONCAKE_ENABLE_GPU_DIRECT", "0") == "1",
            allow_host_fallback=os.getenv("MOONCAKE_ALLOW_HOST_FALLBACK", "1") == "1",
            store_full_wait_seconds=store_full_wait_seconds,
            store_full_log_interval_seconds=store_full_log_interval_seconds,
            store_full_max_wait_seconds=store_full_max_wait_seconds,
            get_retry_wait_seconds=get_retry_wait_seconds,
            get_retry_log_interval_seconds=get_retry_log_interval_seconds,
            get_retry_max_wait_seconds=get_retry_max_wait_seconds,
            transfer_timeout_seconds=transfer_timeout_seconds,
            tcp_progress_timeout_seconds=tcp_progress_timeout_seconds,
            enable_hard_pin=os.getenv("MOONCAKE_ENABLE_HARD_PIN", "0") == "1",
        )

    @staticmethod
    def _env_port(name: str, default: str) -> str:
        value = os.getenv(name, default)
        parsed = urlparse(value)
        if parsed.scheme and parsed.port is not None:
            return str(parsed.port)
        return value.rsplit(":", 1)[-1] if ":" in value else value

    @classmethod
    def from_master_address(
        cls,
        master_host: str,
        master_port: int = 50051,
        metadata_port: int = 8090,
        **kwargs,
    ) -> "MooncakeConfig":
        """
        Create config from master address.

        Assumes the master is running with built-in HTTP metadata server enabled.
        """
        return cls(
            metadata_server=f"http://{master_host}:{metadata_port}/metadata",
            master_server_address=f"{master_host}:{master_port}",
            **kwargs,
        )

    @staticmethod
    def parse_size(size_str: str) -> int:
        """Parse size string like '4GB', '512MB', or '4G' to bytes."""
        size_str = size_str.upper().strip()
        # Ordered by suffix length (longest first) to avoid "4GB" matching "B".
        multipliers = [
            ("TB", 1024 * 1024 * 1024 * 1024),
            ("GB", 1024 * 1024 * 1024),
            ("MB", 1024 * 1024),
            ("KB", 1024),
            ("T", 1024 * 1024 * 1024 * 1024),
            ("G", 1024 * 1024 * 1024),
            ("M", 1024 * 1024),
            ("K", 1024),
            ("B", 1),
        ]
        for suffix, multiplier in multipliers:
            if size_str.endswith(suffix):
                return int(float(size_str[: -len(suffix)]) * multiplier)
        return int(size_str)
