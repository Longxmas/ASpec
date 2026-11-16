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

"""Inference engine creation and initialization with Ray placement groups."""

import secrets

import ray
from ray.util.scheduling_strategies import (
    NodeAffinitySchedulingStrategy,
    PlacementGroupSchedulingStrategy,
)

from aspec.utils.env import get_aspec_env_vars
from aspec.utils.logging import logger

# Multi-node TP worker engines must stay alive to participate in NCCL
# operations but are never called directly. Store refs here to prevent GC.
_alive_worker_engines: list = []


def _require_supported_engine(engine_type: str) -> None:
    if engine_type not in {"sgl", "offline"}:
        raise NotImplementedError(
            "ASpec currently supports the SGLang or offline replay path "
            f"(inference_engine_type='sgl' or 'offline'); got {engine_type!r}"
        )


def create_inference_engines(args, inference_pg, mooncake_config, engine_group: int = 0):
    """Create inference engines based on configured engine type (blocking).

    Supports live inference engines and the CPU-only ``offline`` replay engine.

    Returns:
        List of initialized engines used for dispatching requests.
    """
    engine_type = getattr(args, "inference_engine_type", "sgl")
    _require_supported_engine(engine_type)
    if engine_type == "offline":
        engines, init_refs = _prepare_offline_replay_engines(args, mooncake_config, engine_group)
        _wait_for_init(init_refs, "OfflineReplay", timeout=300)
        return engines

    if engine_type not in ("hf", "sgl", "vllm", "trtllm", "tokenspeed"):
        raise ValueError(f"Unknown inference_engine_type: {engine_type}")

    logger.info(f"Using {engine_type} engine for inference")

    engines = init_engines(
        args,
        inference_pg,
        engine_type=engine_type,
        mooncake_config=mooncake_config,
        engine_group=engine_group,
    )

    logger.info(
        f"Created {len(engines)} distributed {engine_type} engines from {args.target_model_path}"
    )
    return engines


def prepare_inference_engines(args, inference_pg, mooncake_config, engine_group: int = 0):
    """Create inference engines and fire init calls without waiting.

    Use this to parallelize engine initialization with other setup work
    (e.g., training actor initialization). Call ray.get() on the returned
    init_refs before using the engines.

    Returns:
        Tuple of (head_engines, init_refs) where head_engines are the engines
        for dispatching requests, and init_refs are ObjectRefs to wait on.
    """
    engine_type = getattr(args, "inference_engine_type", "sgl")
    _require_supported_engine(engine_type)
    if engine_type == "offline":
        return _prepare_offline_replay_engines(args, mooncake_config, engine_group)

    if engine_type not in ("hf", "sgl", "vllm", "trtllm", "tokenspeed"):
        raise ValueError(f"Unknown inference_engine_type: {engine_type}")

    logger.info(f"Preparing {engine_type} inference engines...")

    if engine_type == "hf":
        engines, init_refs = _prepare_hf_engines(args, inference_pg, mooncake_config, engine_group)
    elif engine_type == "sgl":
        engines, init_refs = _prepare_sgl_engines(args, inference_pg, mooncake_config, engine_group)
    elif engine_type == "trtllm":
        engines, init_refs = _prepare_trtllm_engines(
            args, inference_pg, mooncake_config, engine_group
        )
    elif engine_type == "tokenspeed":
        engines, init_refs = _prepare_tokenspeed_engines(
            args, inference_pg, mooncake_config, engine_group
        )
    else:
        engines, init_refs = _prepare_vllm_engines(
            args, inference_pg, mooncake_config, engine_group
        )

    return engines, init_refs


def _prepare_offline_replay_engines(
    args, mooncake_config, engine_group: int = 0
) -> tuple[list, list]:
    """Create CPU replay actors; they do not consume inference placement GPUs."""
    from aspec.inference.engine.offline_replay_engine import OfflineReplayEngine

    num_engines = getattr(args, "offline_num_engines", 1)
    if num_engines <= 0:
        raise ValueError("inference.offline.num_engines must be positive")
    ReplayRayActor = ray.remote(OfflineReplayEngine)
    env_vars = get_aspec_env_vars(args)
    env_vars["CUDA_VISIBLE_DEVICES"] = ""
    engines = [
        ReplayRayActor.options(
            num_cpus=1,
            num_gpus=0,
            runtime_env={"env_vars": env_vars},
        ).remote(args=args, rank=rank, engine_group=engine_group)
        for rank in range(num_engines)
    ]
    init_refs = [engine.init.remote(mooncake_config=mooncake_config) for engine in engines]
    logger.info("Preparing %d CPU offline replay engine(s)", num_engines)
    return engines, init_refs


def init_engines(args, pg, engine_type: str, mooncake_config=None, engine_group: int = 0) -> list:
    """Initialize inference engines with Ray placement groups.

    Args:
        args: Configuration arguments.
        pg: Placement group tuple (pg, reordered_bundle_indices, reordered_gpu_ids).
        engine_type: Engine type ("hf", "sgl", "vllm", or "trtllm").
        mooncake_config: MooncakeConfig object.

    Returns:
        List of head engines for dispatching requests.
    """
    _require_supported_engine(engine_type)
    if engine_type == "hf":
        return _init_hf_engines(args, pg, mooncake_config, engine_group)
    elif engine_type == "sgl":
        return _init_sgl_engines(args, pg, mooncake_config, engine_group)
    elif engine_type == "vllm":
        return _init_vllm_engines(args, pg, mooncake_config, engine_group)
    elif engine_type == "trtllm":
        return _init_trtllm_engines(args, pg, mooncake_config, engine_group)
    elif engine_type == "tokenspeed":
        return _init_tokenspeed_engines(args, pg, mooncake_config, engine_group)
    else:
        raise ValueError(f"Unknown engine_type: {engine_type}")


def _prepare_hf_engines(args, pg, mooncake_config=None, engine_group: int = 0) -> tuple[list, list]:
    """Create HF engine actors and fire init calls without waiting.

    Returns:
        Tuple of (engines, init_handles).
    """
    num_gpus_total = getattr(args, "inference_num_gpus", 1)
    num_gpus_per_engine = getattr(args, "inference_num_gpus_per_engine", 1)
    num_gpus_per_node = getattr(args, "inference_num_gpus_per_node", 8)

    num_gpus_per_engine = min(num_gpus_per_engine, num_gpus_per_node)
    num_engines = num_gpus_total // num_gpus_per_engine

    logger.info(f"Initializing {num_engines} HF engines ({num_gpus_per_engine} GPU(s) each)")

    from aspec.inference.engine.hf_engine import HFEngine

    HFRayActor = ray.remote(HFEngine)
    return _create_and_init_actors(
        args,
        pg,
        num_engines,
        num_gpus_per_engine,
        HFRayActor,
        mooncake_config,
        engine_group=engine_group,
    )


def _init_hf_engines(args, pg, mooncake_config=None, engine_group: int = 0) -> list:
    """Initialize HF engines with Ray placement groups."""
    engines, init_handles = _prepare_hf_engines(args, pg, mooncake_config, engine_group)
    _wait_for_init(init_handles, "HF", timeout=1200)
    return engines


def _prepare_sgl_engines(
    args, pg, mooncake_config=None, engine_group: int = 0
) -> tuple[list, list]:
    """Start one complete, node-local TP group per Target engine."""
    from aspec.inference.service_client import TargetServiceClient
    from aspec.inference.service_protocol import validate_target_tp

    validate_target_tp(getattr(args, "sglang_nnodes", 1))
    num_gpus_total = getattr(args, "inference_num_gpus", 1)
    gpus_per_engine = getattr(args, "inference_num_gpus_per_engine", 1)
    gpus_per_node = getattr(args, "inference_num_gpus_per_node", num_gpus_total)
    num_engines = num_gpus_total // gpus_per_engine
    logger.info("Initializing %s Target engines (%s GPUs each)", num_engines, gpus_per_engine)

    SglRayActor = ray.remote(TargetServiceClient)
    env_vars = get_aspec_env_vars(args)
    runtime_env = {"env_vars": env_vars}
    service = getattr(args, "target_service", None) or {}
    address = service.get("address", "http://127.0.0.1:30000")
    addresses = service.get("addresses") or {}
    node_ips = getattr(args, "inference_node_ips", None) or []
    shared_node = None
    if getattr(args, "aspec_shared_source", False):
        reader_nodes = {owner["node_ip"] for owner in args.aspec_read_owners}
        if len(reader_nodes) != 1:
            raise ValueError("Shared Target readers must be on one physical node")
        shared_node = next(iter(reader_nodes))

    proxy_options = {
        "num_cpus": 0.2, "num_gpus": 0, "runtime_env": runtime_env,
        # Loopback addresses refer to the coordinator. Keeping the facade here
        # also preserves control when a remote Target node is lost.
        "scheduling_strategy": NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(), soft=False,
        ),
    }

    # Step 1: Create all engine actors (without calling init yet)
    engines = []
    try:
        for i in range(num_engines):
            node_index = i * gpus_per_engine // gpus_per_node
            node_ip = node_ips[node_index] if node_ips else shared_node

            engine = SglRayActor.options(**proxy_options).remote(
                address=addresses.get(node_ip, address),
                args=args,
                rank=i,
                node_ip=node_ip,
                num_gpus_per_engine=gpus_per_engine,
                engine_group=engine_group,
            )
            engines.append(engine)

        # Each engine needs two consecutive ports. Spread independent sessions over
        # the port space instead of all probing 10000 before slow backend startup.
        # Sequential probes keep this session's engines distinct. This is a free
        # port probe, not a reservation: a later external bind still fails closed.
        pre_allocated_ports: dict[int, int] = {}
        next_start = 10000 + secrets.randbelow(40000)
        for i in range(num_engines):
            port = ray.get(
                engines[i].find_free_port.remote(start_port=next_start, consecutive=2),
                timeout=120,
            )
            pre_allocated_ports[i] = port
            next_start = port + 2
            logger.info(f"Engine {i}: pre-allocated ports {port}, {port + 1}")

        # Dispatch initialization only after every port probe succeeds.
        init_handles = []
        for i, engine in enumerate(engines):
            init_handles.append(
                engine.init.remote(
                    mooncake_config=mooncake_config,
                    pre_allocated_port=pre_allocated_ports.get(i),
                )
            )

    except BaseException:
        rollback_unused_targets(engines)
        raise

    return engines, init_handles


def rollback_unused_targets(engines):
    """No generation has been dispatched. Retire every acquired Target, including partial init."""
    for engine in engines:
        try:
            ray.get(engine.shutdown.remote(force=True), timeout=30)
        except Exception:
            # A lost create/close response is covered by Target's expiry of sessions
            # with no generation. Dropping the facade stops its heartbeat.
            logger.exception("Target startup rollback was not acknowledged")
        finally:
            try:
                ray.kill(engine, no_restart=True)
            except Exception:
                logger.exception("Could not retire Target control facade")


def _init_sgl_engines(args, pg, mooncake_config=None, engine_group: int = 0) -> list:
    """Initialize SGLang engines with Ray placement groups (blocking)."""
    engines, init_handles = _prepare_sgl_engines(args, pg, mooncake_config, engine_group)
    init_timeout = getattr(args, "sglang_init_timeout", 300)
    try:
        _wait_for_init(init_handles, "Sgl", timeout=init_timeout)
    except BaseException:
        rollback_unused_targets(engines)
        raise
    return engines


def _prepare_vllm_engines(
    args, pg, mooncake_config=None, engine_group: int = 0
) -> tuple[list, list]:
    """Create vLLM engine actors and fire init calls without waiting.

    Handles three cases:
      - Single-node, multiple engines: one engine per group of GPUs
      - Multi-node, single replica: one engine per node, all forming one TP group
      - Multi-node, multiple replicas: N independent TP groups, each spanning nnodes

    For multi-node, worker engines are stored in a module-level list to prevent GC.

    Returns:
        Tuple of (head_engines, init_handles). head_engines are the engines that
        accept generate() calls. init_handles are ObjectRefs for ALL engines
        (head + worker) that must be waited on before use.
    """
    nnodes = getattr(args, "vllm_nnodes", 1)
    num_gpus_total = getattr(args, "inference_num_gpus", 1)

    if nnodes > 1:
        gpus_per_node = getattr(args, "inference_num_gpus_per_node", 8)
        gpus_per_replica = nnodes * gpus_per_node
        num_replicas = num_gpus_total // gpus_per_replica
        num_engines = num_replicas * nnodes
        gpus_per_engine = gpus_per_node
    else:
        gpus_per_engine = getattr(args, "inference_num_gpus_per_engine", 1)
        num_replicas = num_gpus_total // gpus_per_engine
        num_engines = num_replicas

    logger.info(
        f"Initializing {num_engines} vLLM engines "
        f"({gpus_per_engine} GPU(s) each, nnodes={nnodes}, replicas={num_replicas})"
    )

    from aspec.inference.engine.vllm_engine import VllmEngine

    pg_obj, reordered_bundle_indices, reordered_gpu_ids = pg

    VllmRayActor = ray.remote(VllmEngine)
    env_vars = get_aspec_env_vars(args)

    engines = []
    for i in range(num_engines):
        node_rank = i % nnodes if nnodes > 1 else 0

        bundle_offset = i * gpus_per_engine
        base_gpu_id = int(reordered_gpu_ids[bundle_offset])

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg_obj,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[bundle_offset],
        )

        engine = VllmRayActor.options(
            num_cpus=0.2,
            num_gpus=0.2,
            scheduling_strategy=scheduling_strategy,
            runtime_env={"env_vars": env_vars},
        ).remote(
            args=args,
            rank=i,
            base_gpu_id=base_gpu_id,
            num_gpus_per_engine=gpus_per_engine,
            node_rank=node_rank,
            engine_group=engine_group,
        )
        engines.append(engine)

    # Pre-allocate ports to avoid TOCTOU races between parallel engines.
    # Each engine needs 2 consecutive ports (dist_init + nccl).
    pre_allocated_ports: dict[int, int] = {}
    next_start = 10000
    for i in range(num_engines):
        port = ray.get(
            engines[i].find_free_port.remote(start_port=next_start, consecutive=2),
            timeout=30,
        )
        pre_allocated_ports[i] = port
        next_start = port + 2
        logger.info(f"vLLM Engine {i}: pre-allocated ports {port}, {port + 1}")

    dist_init_addrs: dict[int, str] = {}
    if nnodes > 1:
        configured_addr = getattr(args, "vllm_dist_init_addr", None)
        for replica_idx in range(num_replicas):
            if configured_addr and num_replicas == 1:
                dist_init_addrs[replica_idx] = configured_addr
                logger.info(
                    f"Replica {replica_idx}: using configured dist_init_addr: {configured_addr}"
                )
            else:
                head_idx = replica_idx * nnodes
                head_engine = engines[head_idx]
                ip = ray.get(head_engine.get_node_ip.remote(), timeout=30)
                addr = f"{ip}:{pre_allocated_ports[head_idx]}"
                dist_init_addrs[replica_idx] = addr
                logger.info(
                    f"Replica {replica_idx}: dist_init_addr from pre-allocated port: {addr}"
                )

    init_handles = []
    for i, engine in enumerate(engines):
        replica_idx = i // nnodes if nnodes > 1 else i
        init_handles.append(
            engine.init.remote(
                mooncake_config=mooncake_config,
                dist_init_addr=dist_init_addrs.get(replica_idx),
                pre_allocated_port=pre_allocated_ports.get(i),
            )
        )

    if nnodes > 1:
        head_engines = [engines[i] for i in range(num_engines) if i % nnodes == 0]
        worker_engines = [engines[i] for i in range(num_engines) if i % nnodes != 0]
        _alive_worker_engines.extend(worker_engines)
        logger.info(
            f"Prepared multi-node vLLM engines: {len(head_engines)} heads + "
            f"{len(worker_engines)} workers ({num_replicas} replicas)"
        )
        return head_engines, init_handles

    return engines, init_handles


def _init_vllm_engines(args, pg, mooncake_config=None, engine_group: int = 0) -> list:
    """Initialize vLLM engines with Ray placement groups (blocking)."""
    head_engines, init_handles = _prepare_vllm_engines(args, pg, mooncake_config, engine_group)
    nnodes = getattr(args, "vllm_nnodes", 1)
    init_timeout = getattr(args, "vllm_init_timeout", 300 if nnodes == 1 else 600)
    _wait_for_init(init_handles, "Vllm", timeout=init_timeout)
    return head_engines


def _prepare_trtllm_engines(
    args, pg, mooncake_config=None, engine_group: int = 0
) -> tuple[list, list]:
    """Create TensorRT-LLM engine actors and fire init calls without waiting.

    Single-node only: each engine owns a contiguous block of
    ``inference_num_gpus_per_engine`` GPUs and runs TRT-LLM's own MPI workers
    across them (TP degree = num_gpus_per_engine). Unlike sgl/vllm there is no
    cross-node dist-init negotiation or port pre-allocation — TRT-LLM manages
    worker bring-up internally.

    Returns:
        Tuple of (engines, init_handles).
    """
    nnodes = getattr(args, "trtllm_nnodes", 1)
    if nnodes > 1:
        raise NotImplementedError(
            f"trtllm backend supports single-node TP only (trtllm_nnodes={nnodes})."
        )

    num_gpus_total = getattr(args, "inference_num_gpus", 1)
    gpus_per_engine = getattr(args, "inference_num_gpus_per_engine", 1)
    num_engines = num_gpus_total // gpus_per_engine

    logger.info(
        f"Initializing {num_engines} TensorRT-LLM engines "
        f"({gpus_per_engine} GPU(s) each, tp_size={gpus_per_engine})"
    )

    from aspec.inference.engine.trtllm_engine import TrtllmEngine

    pg_obj, reordered_bundle_indices, reordered_gpu_ids = pg

    TrtllmRayActor = ray.remote(TrtllmEngine)
    env_vars = get_aspec_env_vars(args)

    # TRT-LLM's MPI workers bind GPUs themselves, so assignment differs by TP degree:
    # tp==1 drops NOSET and reserves a whole GPU so Ray scopes it as cuda:0 per engine;
    # tp>1 keeps NOSET, pins a contiguous block in init(), and reserves a 0.2 placeholder.
    single_gpu = gpus_per_engine == 1
    if single_gpu:
        env_vars.pop("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", None)
    actor_num_gpus = float(gpus_per_engine) if single_gpu else 0.2

    engines = []
    for i in range(num_engines):
        bundle_offset = i * gpus_per_engine
        base_gpu_id = int(reordered_gpu_ids[bundle_offset])

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg_obj,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[bundle_offset],
        )

        engine = TrtllmRayActor.options(
            num_cpus=0.2,
            num_gpus=actor_num_gpus,
            scheduling_strategy=scheduling_strategy,
            runtime_env={"env_vars": env_vars},
        ).remote(
            args=args,
            rank=i,
            base_gpu_id=base_gpu_id,
            num_gpus_per_engine=gpus_per_engine,
            node_rank=0,
            engine_group=engine_group,
        )
        engines.append(engine)

    init_handles = [engine.init.remote(mooncake_config=mooncake_config) for engine in engines]
    return engines, init_handles


def _init_trtllm_engines(args, pg, mooncake_config=None, engine_group: int = 0) -> list:
    """Initialize TensorRT-LLM engines with Ray placement groups (blocking)."""
    engines, init_handles = _prepare_trtllm_engines(args, pg, mooncake_config, engine_group)
    init_timeout = getattr(args, "trtllm_init_timeout", 600)
    _wait_for_init(init_handles, "Trtllm", timeout=init_timeout)
    return engines


def _prepare_tokenspeed_engines(
    args, pg, mooncake_config=None, engine_group: int = 0
) -> tuple[list, list]:
    """Create single-node TokenSpeed engine actors."""
    nnodes = getattr(args, "tokenspeed_nnodes", 1)
    if nnodes != 1:
        raise NotImplementedError(
            "The initial TokenSpeed backend supports single-node inference only"
        )

    num_gpus_total = getattr(args, "inference_num_gpus", 1)
    gpus_per_engine = getattr(args, "inference_num_gpus_per_engine", 1)
    num_engines = num_gpus_total // gpus_per_engine
    if num_engines <= 0:
        raise ValueError("TokenSpeed requires at least one inference engine")

    from aspec.inference.engine.tokenspeed_engine import TokenSpeedEngine

    pg_obj, reordered_bundle_indices, reordered_gpu_ids = pg
    TokenSpeedRayActor = ray.remote(TokenSpeedEngine)
    env_vars = get_aspec_env_vars(args)
    engines = []
    for index in range(num_engines):
        bundle_offset = index * gpus_per_engine
        base_gpu_id = int(reordered_gpu_ids[bundle_offset])
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg_obj,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[bundle_offset],
        )
        engines.append(
            TokenSpeedRayActor.options(
                num_cpus=0.2,
                num_gpus=0.2,
                scheduling_strategy=scheduling_strategy,
                runtime_env={"env_vars": env_vars},
            ).remote(
                args=args,
                rank=index,
                base_gpu_id=base_gpu_id,
                num_gpus_per_engine=gpus_per_engine,
                node_rank=0,
                engine_group=engine_group,
            )
        )

    init_refs = [engine.init.remote(mooncake_config=mooncake_config) for engine in engines]
    logger.info(
        "Preparing %d TokenSpeed engine(s), %d GPU(s) each",
        num_engines,
        gpus_per_engine,
    )
    return engines, init_refs


def _init_tokenspeed_engines(args, pg, mooncake_config=None, engine_group: int = 0) -> list:
    engines, init_refs = _prepare_tokenspeed_engines(args, pg, mooncake_config, engine_group)
    timeout = getattr(args, "tokenspeed_init_timeout", 600)
    _wait_for_init(init_refs, "TokenSpeed", timeout=timeout)
    return engines


def _create_and_init_actors(
    args,
    pg,
    num_engines,
    num_gpus_per_engine,
    ray_actor_cls,
    mooncake_config,
    extra_kwargs=None,
    engine_group: int = 0,
) -> tuple[list, list]:
    """Create Ray actors and start their init calls.

    Args:
        args: Configuration arguments.
        pg: Placement group tuple.
        num_engines: Number of engines to create.
        num_gpus_per_engine: GPUs per engine.
        ray_actor_cls: The ray.remote-wrapped actor class.
        mooncake_config: Mooncake configuration dict.
        extra_kwargs: Extra kwargs to pass to actor constructor (e.g. num_gpus_per_engine for Sgl).
        engine_group: Group index for log file disambiguation.

    Returns:
        Tuple of (engines list, init_handles list).
    """
    pg_obj, reordered_bundle_indices, reordered_gpu_ids = pg

    engines = []
    init_handles = []

    for i in range(num_engines):
        num_gpus = 0.2
        num_cpus = num_gpus

        base_gpu_id = int(reordered_gpu_ids[i * num_gpus_per_engine])

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg_obj,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[i * num_gpus_per_engine],
        )

        env_vars = get_aspec_env_vars(args)

        constructor_kwargs = {
            "args": args,
            "rank": i,
            "base_gpu_id": base_gpu_id,
            "engine_group": engine_group,
        }
        if extra_kwargs:
            constructor_kwargs.update(extra_kwargs)

        engine = ray_actor_cls.options(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            scheduling_strategy=scheduling_strategy,
            runtime_env={"env_vars": env_vars},
        ).remote(**constructor_kwargs)

        engines.append(engine)
        init_handles.append(engine.init.remote(mooncake_config=mooncake_config))

    return engines, init_handles


def _wait_for_init(init_handles: list, engine_label: str, timeout: int = 300):
    """Wait for engine init calls to complete with error handling.

    Args:
        init_handles: List of init() call ObjectRefs.
        engine_label: Label for log messages (e.g. "HF", "Sgl").
        timeout: Timeout in seconds.
    """
    try:
        logger.info(
            f"Waiting for {len(init_handles)} {engine_label} engines to initialize "
            f"(timeout={timeout}s)..."
        )
        ray.get(init_handles, timeout=timeout)
    except ray.exceptions.GetTimeoutError:
        logger.error(f"Timeout waiting for {engine_label} engines to initialize after {timeout}s")
        ready, not_ready = ray.wait(init_handles, num_returns=len(init_handles), timeout=0)
        logger.error(f"Initialized: {len(ready)}/{len(init_handles)}, Failed: {len(not_ready)}")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize {engine_label} engines: {e}")
        raise

    logger.info(f"Successfully initialized {len(init_handles)} {engine_label} engines")
