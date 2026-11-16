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

from __future__ import annotations

import math
import os
import socket
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from aspec.ray.train_group import RayTrainGroup
from aspec.utils.logging import logger

# Ray exposes a tiny "node:<ip>" resource on each node. Requiring a fractional
# amount pins a bundle to that node without consuming a full logical resource.
_NODE_RESOURCE_EPSILON = 0.001
_CUSTOM_PLACEMENT_FIELDS = (
    "training_node_ips",
    "inference_node_ips",
    "training_node_selectors",
    "inference_node_selectors",
)


@dataclass(frozen=True)
class _NodeConstraint:
    ip: str | None = None
    label_selector: tuple[tuple[str, str], ...] = ()

    @property
    def selector_for_log(self) -> str:
        if self.ip is not None:
            return self.ip
        return str(dict(self.label_selector))

    def to_bundle_resource(self) -> dict[str, float]:
        if self.ip is None:
            return {}
        return {_node_ip_resource(self.ip): _NODE_RESOURCE_EPSILON}

    def to_label_selector(self) -> dict[str, str]:
        return dict(self.label_selector)


@ray.remote(num_gpus=1)
class InfoActor:
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]


def sort_key(x):
    _index, node_identifier, gpu_id = x
    # Sort by node IP number and then by GPU ID
    try:
        # try to parse it as an IP address.
        ip_address = node_identifier
        node_ip_parts = list(map(int, ip_address.split(".")))
    except ValueError:
        # Try to resolve the hostname to an IP address.
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # Instead, we convert each character of the original identifier string
            # to its ASCII value. This provides a stable and consistent numerical
            # representation that allows for sorting.
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, gpu_id)


def _has_value(value) -> bool:
    return value is not None and value != [] and value != {}


def _has_custom_placement_fields(args) -> bool:
    return any(_has_value(getattr(args, field, None)) for field in _CUSTOM_PLACEMENT_FIELDS)


def _validate_custom_strategy_usage(args) -> None:
    placement_strategy = getattr(args, "placement_strategy", "training_first")
    if placement_strategy != "custom" and _has_custom_placement_fields(args):
        raise ValueError(
            "Custom placement fields require training.placement_strategy=custom. "
            f"Got placement_strategy={placement_strategy!r}."
        )


def _as_list(value, field_name: str) -> list:
    if value is None:
        return []
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a list, got {type(value).__name__}")
    return list(value)


def _normalize_node_constraints(
    args, role: str, *, required: bool = False
) -> list[_NodeConstraint]:
    ip_field = f"{role}_node_ips"
    selector_field = f"{role}_node_selectors"
    node_ips = _as_list(getattr(args, ip_field, None), ip_field)
    node_selectors = _as_list(getattr(args, selector_field, None), selector_field)

    if node_ips and node_selectors:
        raise ValueError(f"Set only one of {ip_field} or {selector_field}, not both")
    if not node_ips and not node_selectors:
        if required:
            raise ValueError(
                f"training.placement_strategy=custom requires {ip_field} or {selector_field}"
            )
        return []

    if node_ips:
        constraints = []
        for ip in node_ips:
            if not isinstance(ip, str) or not ip:
                raise ValueError(f"{ip_field} entries must be non-empty strings")
            constraints.append(_NodeConstraint(ip=ip))
        return constraints

    constraints = []
    for selector in node_selectors:
        if not isinstance(selector, Mapping) or not selector:
            raise ValueError(f"{selector_field} entries must be non-empty mappings")
        normalized = tuple(sorted((str(key), str(value)) for key, value in selector.items()))
        constraints.append(_NodeConstraint(label_selector=normalized))
    return constraints


def _node_ip_resource(ip: str) -> str:
    return f"node:{ip}"


def _expected_node_count(total_gpus: int, gpus_per_node: int, role: str) -> int:
    if total_gpus < 0:
        raise ValueError(f"{role} total GPUs must be non-negative, got {total_gpus}")
    if total_gpus == 0:
        return 0
    if gpus_per_node <= 0:
        raise ValueError(f"{role} GPUs per node must be positive, got {gpus_per_node}")
    return math.ceil(total_gpus / gpus_per_node)


def _build_custom_bundles(
    role: str,
    constraints: list[_NodeConstraint],
    total_gpus: int,
    gpus_per_node: int,
) -> tuple[list[dict[str, float]], list[dict[str, str]], list[int]]:
    """Build Ray bundles for nodes in user-provided order.

    ``node_group_indices`` records the configured node ordinal for each bundle.
    After Ray schedules the placement group, InfoActor probes actual nodes and
    GPU ids; the ordinal lets us restore user node order while still sorting
    GPUs within a node by physical GPU id.
    """
    expected_nodes = _expected_node_count(total_gpus, gpus_per_node, role)
    if len(constraints) != expected_nodes:
        raise ValueError(
            f"{role} custom placement expected {expected_nodes} node(s) for "
            f"{total_gpus} GPU(s) with {gpus_per_node} GPU(s) per node, "
            f"got {len(constraints)}"
        )

    bundles: list[dict[str, float]] = []
    bundle_label_selectors: list[dict[str, str]] = []
    node_group_indices: list[int] = []
    remaining = total_gpus

    for node_index, constraint in enumerate(constraints):
        gpus_on_node = min(gpus_per_node, remaining)
        remaining -= gpus_on_node

        for _ in range(gpus_on_node):
            bundle: dict[str, float] = {"GPU": 1, "CPU": 1, **constraint.to_bundle_resource()}
            bundles.append(bundle)
            bundle_label_selectors.append(constraint.to_label_selector())
            node_group_indices.append(node_index)

    return bundles, bundle_label_selectors, node_group_indices


def _merge_bundle_label_selectors(
    selectors: list[dict[str, str]],
) -> list[dict[str, str]] | None:
    return selectors if any(selector for selector in selectors) else None


def _placement_group(
    bundles: list[dict[str, float]],
    *,
    strategy: str,
    name: str | None,
    bundle_label_selector: list[dict[str, str]] | None = None,
):
    kwargs = {"bundles": bundles, "strategy": strategy, "name": name}
    if bundle_label_selector is not None:
        kwargs["bundle_label_selector"] = bundle_label_selector

    try:
        return placement_group(**kwargs)
    except TypeError as e:
        if bundle_label_selector is not None and "bundle_label_selector" in str(e):
            raise RuntimeError(
                "Ray bundle_label_selector is not supported by the installed Ray version. "
                "Use training_node_ips/inference_node_ips or upgrade Ray."
            ) from e
        raise


def _sort_probed_bundle_infos(gpu_ids, node_group_indices: list[int] | None = None):
    """Sort probed bundles by default topology or explicit user node order."""
    bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(len(gpu_ids))]
    if node_group_indices is None:
        return sorted(bundle_infos, key=sort_key)
    if len(node_group_indices) != len(gpu_ids):
        raise ValueError(
            f"node_group_indices length ({len(node_group_indices)}) must match "
            f"bundle count ({len(gpu_ids)})"
        )
    return sorted(bundle_infos, key=lambda info: (node_group_indices[info[0]], info[2]))


def _create_placement_group(
    num_gpus,
    strategy="PACK",
    name=None,
    *,
    bundles: list[dict[str, float]] | None = None,
    bundle_label_selector: list[dict[str, str]] | None = None,
    node_group_indices: list[int] | None = None,
    timeout_seconds: float = 300,
):
    """Create and probe one GPU reservation, rolling back failed setup."""
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("placement timeout must be finite and positive")
    if bundles is None:
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    elif len(bundles) != num_gpus:
        raise ValueError(f"num_gpus={num_gpus} does not match bundle count={len(bundles)}")

    pg = _placement_group(
        bundles,
        strategy=strategy,
        name=name,
        bundle_label_selector=bundle_label_selector,
    )
    num_bundles = len(bundles)

    deadline = time.monotonic() + timeout_seconds
    info_actors = []

    def remaining_timeout():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("placement group setup exceeded its time budget")
        return remaining

    try:
        ray.get(pg.ready(), timeout=remaining_timeout())
        for i in range(num_bundles):
            info_actors.append(
                InfoActor.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=i,
                    )
                ).remote()
            )
        gpu_ids = ray.get(
            [actor.get_ip_and_gpu_id.remote() for actor in info_actors],
            timeout=remaining_timeout(),
        )
        # Probe actors belong only to this reservation. Stop them before
        # returning the bundles to the caller for real trainer/backend actors.
        while info_actors:
            ray.kill(info_actors[-1], no_restart=True)
            info_actors.pop()

        sorted_bundle_infos = _sort_probed_bundle_infos(gpu_ids, node_group_indices)
        pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
        # Map from logical index -> physical GPU ID
        pg_reordered_gpu_ids = [gpu_ids[info[0]][1] for info in sorted_bundle_infos]

        for i in range(num_bundles):
            actual_bundle_index = pg_reordered_bundle_indices[i]
            logger.info(
                f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
                f"node: {gpu_ids[actual_bundle_index][0]}, gpu: {gpu_ids[actual_bundle_index][1]}"
            )

        return pg, pg_reordered_bundle_indices, pg_reordered_gpu_ids
    except BaseException:
        # Retain the original setup error, but attempt every cleanup even if
        # one probe is already unreachable or Ray is shutting down.
        for actor in info_actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                logger.exception("Failed to stop placement probe during rollback")
        try:
            ray.util.remove_placement_group(pg)
        except Exception:
            logger.exception("Failed to remove placement group during rollback")
        raise


def _ensure_ray_initialized():
    """Connect to an existing Ray cluster, or start a local instance as fallback."""
    if ray.is_initialized():
        return

    # Target owns a separate Ray runtime. Do not auto-attach Draft to it (or
    # another user's local cluster) merely because it was started last.
    ray_address = os.environ.get("RAY_ADDRESS", "local")
    if ray_address == "local":
        ray.init(address="local", ignore_reinit_error=True, include_dashboard=False,
                 _temp_dir=tempfile.mkdtemp(prefix="aspec-d-", dir="/tmp"))
        return
    try:
        ray.init(address=ray_address, ignore_reinit_error=True)
        logger.info(f"Connected to Ray cluster at {ray_address}")
    except ConnectionError:
        if ray_address == "auto":
            logger.warning("No existing Ray cluster found, starting a local instance")
            ray.init(ignore_reinit_error=True)
            return
        raise RuntimeError(
            f"Failed to connect to Ray cluster at {ray_address}. "
            "Refusing to fall back to a local Ray instance when RAY_ADDRESS is explicitly set."
        ) from None


def ensure_ray_initialized() -> None:
    """Initialize Ray before process-level signal handlers are finalized."""

    _ensure_ray_initialized()


def _get_expected_gpu_count(args) -> int:
    training_gpus = args.training_num_nodes * args.training_num_gpus_per_node
    inference_gpus = getattr(args, "inference_num_gpus", 0)
    if getattr(args, "colocate", False):
        return max(training_gpus, inference_gpus)
    return training_gpus + inference_gpus


def _wait_for_gpu_resources(expected_gpus: int, timeout: int = 300, poll_interval: int = 5):
    """Block until the Ray cluster has at least ``expected_gpus`` GPUs."""
    available = int(ray.cluster_resources().get("GPU", 0))
    if available >= expected_gpus:
        logger.info(f"Ray cluster has {available} GPUs (need {expected_gpus})")
        return

    logger.info(f"Waiting for {expected_gpus} GPUs (currently {available}), timeout={timeout}s...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(poll_interval)
        available = int(ray.cluster_resources().get("GPU", 0))
        logger.info(f"Ray cluster GPUs: {available}/{expected_gpus}")
        if available >= expected_gpus:
            logger.info(f"All {expected_gpus} GPUs available")
            return

    raise RuntimeError(
        f"Timed out waiting for GPUs: {available}/{expected_gpus} after {timeout}s. "
        f"Check that all Ray worker nodes have joined the cluster."
    )


def _create_custom_unified_placement_group(args, num_training_gpus: int, num_inference_gpus: int):
    training_constraints = _normalize_node_constraints(
        args, "training", required=num_training_gpus > 0
    )
    inference_constraints = _normalize_node_constraints(
        args, "inference", required=num_inference_gpus > 0
    )

    training_bundles, training_selectors, training_groups = _build_custom_bundles(
        "training",
        training_constraints,
        num_training_gpus,
        args.training_num_gpus_per_node,
    )
    inference_bundles, inference_selectors, inference_groups = _build_custom_bundles(
        "inference",
        inference_constraints,
        num_inference_gpus,
        args.inference_num_gpus_per_node,
    )

    node_group_offset = len(training_constraints)
    node_group_indices = training_groups + [
        node_group_offset + node_group_index for node_group_index in inference_groups
    ]
    bundles = training_bundles + inference_bundles
    bundle_label_selector = _merge_bundle_label_selectors(training_selectors + inference_selectors)

    total_gpus = num_training_gpus + num_inference_gpus
    logger.info(
        "Creating custom unified placement group with "
        f"{total_gpus} GPUs ({num_training_gpus} training + {num_inference_gpus} inference); "
        f"training nodes={[constraint.selector_for_log for constraint in training_constraints]}, "
        f"inference nodes={[constraint.selector_for_log for constraint in inference_constraints]}"
    )

    pg, sorted_bundle_indices, sorted_gpu_ids = _create_placement_group(
        total_gpus,
        strategy="PACK",
        name="custom_unified_pg",
        bundles=bundles,
        bundle_label_selector=bundle_label_selector,
        node_group_indices=node_group_indices,
    )

    training_bundle_indices = sorted_bundle_indices[:num_training_gpus]
    training_gpu_ids = sorted_gpu_ids[:num_training_gpus]
    inference_bundle_indices = sorted_bundle_indices[num_training_gpus:]
    inference_gpu_ids = sorted_gpu_ids[num_training_gpus:]

    logger.info(
        f"Placement (strategy=custom): "
        f"training bundles={training_bundle_indices}, "
        f"inference bundles={inference_bundle_indices}"
    )

    return {
        "training": (pg, training_bundle_indices, training_gpu_ids),
        "inference": (pg, inference_bundle_indices, inference_gpu_ids),
    }


def _get_custom_colocated_constraints(args) -> tuple[str, list[_NodeConstraint]]:
    training_constraints = _normalize_node_constraints(args, "training")
    inference_constraints = _normalize_node_constraints(args, "inference")
    if (
        training_constraints
        and inference_constraints
        and training_constraints != inference_constraints
    ):
        raise ValueError(
            "custom colocate placement requires training and inference node constraints "
            "to match, or only one role's constraints to be set"
        )

    if training_constraints:
        role = "training"
        constraints = training_constraints
    elif inference_constraints:
        role = "inference"
        constraints = inference_constraints
    else:
        raise ValueError(
            "training.placement_strategy=custom with colocate=True requires training_node_* "
            "or inference_node_* constraints"
        )

    return role, constraints


def _validate_custom_placement_constraints(args) -> None:
    if getattr(args, "placement_strategy", "training_first") != "custom":
        return

    if args.colocate:
        num_gpus = args.training_num_nodes * args.training_num_gpus_per_node
        _role, constraints = _get_custom_colocated_constraints(args)
        _build_custom_bundles(
            "colocate",
            constraints,
            num_gpus,
            args.training_num_gpus_per_node,
        )
        return

    num_training_gpus = args.training_num_nodes * args.training_num_gpus_per_node
    num_inference_gpus = args.inference_num_gpus

    training_constraints = _normalize_node_constraints(
        args, "training", required=num_training_gpus > 0
    )
    _build_custom_bundles(
        "training",
        training_constraints,
        num_training_gpus,
        args.training_num_gpus_per_node,
    )

    inference_constraints = _normalize_node_constraints(
        args, "inference", required=num_inference_gpus > 0
    )
    _build_custom_bundles(
        "inference",
        inference_constraints,
        num_inference_gpus,
        args.inference_num_gpus_per_node,
    )


def _create_custom_colocated_placement_group(args, num_gpus: int):
    role, constraints = _get_custom_colocated_constraints(args)
    # Colocate creates one shared placement group using the training topology.
    # Either role's node constraints may select the nodes, but node count
    # validation must use the topology that determines ``num_gpus``.
    bundles, bundle_label_selectors, node_group_indices = _build_custom_bundles(
        "colocate",
        constraints,
        num_gpus,
        args.training_num_gpus_per_node,
    )
    logger.info(
        f"Creating custom colocated placement group with {num_gpus} GPU(s) "
        f"using {role} constraints on "
        f"{[constraint.selector_for_log for constraint in constraints]}"
    )
    return _create_placement_group(
        num_gpus,
        strategy="PACK",
        name="custom_colocate_pg",
        bundles=bundles,
        bundle_label_selector=_merge_bundle_label_selectors(bundle_label_selectors),
        node_group_indices=node_group_indices,
    )


def create_placement_groups(args, roles: set[str] | None = None):
    """Create placement groups for the requested training/inference roles."""
    if roles is None:
        roles = {"training", "inference"}
    roles = frozenset(roles)
    unknown = roles - {"training", "inference"}
    if not roles or unknown:
        raise ValueError(f"Invalid placement roles: {sorted(roles)}")

    _ensure_ray_initialized()

    if getattr(args, "inference_engine_type", None) == "sgl":
        managed_local = (getattr(args, "target_service", None) or {}).get("mode") == "managed-local"
        if managed_local:
            live_nodes = [node for node in ray.nodes() if node.get("Alive")]
            if len(live_nodes) != 1 or live_nodes[0]["NodeID"] != ray.get_runtime_context().get_node_id():
                raise ValueError("managed-local requires Draft Ray on the coordinator node only")
        # Target owns its own CUDA-visible pool. A control proxy has no GPU
        # reservation in Draft's Ray cluster, including for remote services.
        _validate_custom_strategy_usage(args)
        _validate_custom_placement_constraints(args)
        result = {"training": None, "inference": None}
        if "training" not in roles:
            return result
        count = args.training_num_nodes * args.training_num_gpus_per_node
        _wait_for_gpu_resources(count)
        if getattr(args, "placement_strategy", "training_first") == "custom":
            constraints = _normalize_node_constraints(args, "training", required=True)
            bundles, selectors, node_groups = _build_custom_bundles(
                "training", constraints, count, args.training_num_gpus_per_node,
            )
            result["training"] = _create_placement_group(
                count, strategy="PACK", name="training_pg", bundles=bundles,
                bundle_label_selector=_merge_bundle_label_selectors(selectors),
                node_group_indices=node_groups,
            )
        else:
            result["training"] = _create_placement_group(count, strategy="PACK", name="training_pg")
        if managed_local:
            from aspec.utils.misc import visible_gpu_ids

            allocated = {int(gpu) for gpu in result["training"][2]}
            if allocated != set(visible_gpu_ids()):
                ray.util.remove_placement_group(result["training"][0])
                raise ValueError("Draft Ray GPU allocation differs from the managed-local GPU split")
        return result

    if len(roles) == 1:
        if getattr(
            args, "placement_strategy", "training_first"
        ) == "custom" or _has_custom_placement_fields(args):
            raise ValueError(
                "Custom placement is only supported when training and inference "
                "roles are created together"
            )

        role = next(iter(roles))
        if role == "training":
            num_gpus = args.training_num_nodes * args.training_num_gpus_per_node
        else:
            num_gpus = args.inference_num_gpus
        if not isinstance(num_gpus, int) or num_gpus <= 0:
            raise ValueError(f"{role.capitalize()} placement requires a positive GPU count")

        _wait_for_gpu_resources(num_gpus)
        logger.info("Creating %s-only placement with %d GPUs...", role, num_gpus)
        pg, bundle_indices, gpu_ids = _create_placement_group(
            num_gpus, strategy="PACK", name=f"{role}_pg"
        )
        empty = (pg, [], [])
        result = {"training": empty, "inference": empty}
        result[role] = (pg, bundle_indices, gpu_ids)
        return result

    _validate_custom_strategy_usage(args)
    _validate_custom_placement_constraints(args)
    _wait_for_gpu_resources(_get_expected_gpu_count(args))
    placement_strategy = getattr(args, "placement_strategy", "training_first")

    if args.colocate:
        num_gpus = args.training_num_nodes * args.training_num_gpus_per_node
        logger.info(f"Creating colocated placement group with {num_gpus} GPUs...")
        if placement_strategy == "custom":
            pg, bundle_indices, gpu_ids = _create_custom_colocated_placement_group(args, num_gpus)
        else:
            pg, bundle_indices, gpu_ids = _create_placement_group(
                num_gpus, strategy="PACK", name="colocate_pg"
            )
        return {
            "training": (pg, bundle_indices, gpu_ids),
            "inference": (pg, bundle_indices, gpu_ids),
        }

    num_training_gpus = args.training_num_nodes * args.training_num_gpus_per_node
    num_inference_gpus = args.inference_num_gpus
    total_gpus = num_training_gpus + num_inference_gpus

    if placement_strategy == "custom":
        return _create_custom_unified_placement_group(args, num_training_gpus, num_inference_gpus)

    # Single PG ensures deterministic node-to-role assignment across restarts,
    # avoiding kernel/weight cache misses from random GPU shuffling.
    logger.info(
        f"Creating unified placement group with {total_gpus} GPUs "
        f"({num_training_gpus} training + {num_inference_gpus} inference)..."
    )

    pg, sorted_bundle_indices, sorted_gpu_ids = _create_placement_group(
        total_gpus, strategy="PACK", name="unified_pg"
    )

    if placement_strategy == "training_first":
        training_bundle_indices = sorted_bundle_indices[:num_training_gpus]
        training_gpu_ids = sorted_gpu_ids[:num_training_gpus]
        inference_bundle_indices = sorted_bundle_indices[num_training_gpus:]
        inference_gpu_ids = sorted_gpu_ids[num_training_gpus:]
    else:
        inference_bundle_indices = sorted_bundle_indices[:num_inference_gpus]
        inference_gpu_ids = sorted_gpu_ids[:num_inference_gpus]
        training_bundle_indices = sorted_bundle_indices[num_inference_gpus:]
        training_gpu_ids = sorted_gpu_ids[num_inference_gpus:]

    logger.info(
        f"Placement (strategy={placement_strategy}): "
        f"training bundles={training_bundle_indices}, "
        f"inference bundles={inference_bundle_indices}"
    )

    return {
        "training": (pg, training_bundle_indices, training_gpu_ids),
        "inference": (pg, inference_bundle_indices, inference_gpu_ids),
    }


def allocate_train_group(args, num_nodes, num_gpus_per_node, pg, training_class=None):
    return RayTrainGroup(
        args=args,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        pg=pg,
        num_gpus_per_actor=0.4,
        training_class=training_class,
    )


def create_train_group(args, training_pg, training_class=None, mooncake_config=None):
    train_group = allocate_train_group(
        args=args,
        num_nodes=args.training_num_nodes,
        num_gpus_per_node=args.training_num_gpus_per_node,
        pg=training_pg,
        training_class=training_class,
    )

    some_ids = ray.get(
        train_group.async_init(
            args, role="training", mooncake_config=mooncake_config, with_ref=False
        )
    )

    if len(set(some_ids)) != 1:
        raise RuntimeError(
            "Training actors returned inconsistent initialization identifiers: "
            f"{some_ids!r}"
        )

    return train_group
