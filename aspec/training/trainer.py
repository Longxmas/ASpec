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

import abc
import concurrent.futures
import dataclasses
import itertools
import json
import logging
import os
import tempfile
import time
from argparse import Namespace
from contextlib import nullcontext
from typing import TYPE_CHECKING, Callable, Optional

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh

from aspec.config.draft_execution import (
    is_cudnn_flex_backend,
    is_direct_cp_backend,
    is_fa4_backend,
)
from aspec.config.mooncake_config import MooncakeConfig
from aspec.data.utils import DataCollatorWithPadding, DFlashPackingCollator
from aspec.training import checkpoint
from aspec.training.data_fetcher import (
    MooncakeDataFetcher,
    PrefetchedDataFetcher,
    _take_p2p_receive_leases,
)
from aspec.training.fsdp import init_empty_weights
from aspec.training.optimizer import BF16Optimizer
from aspec.utils.distributed import (
    get_direct_cp_group,
    get_gloo_group,
    get_usp_device_mesh,
    get_usp_grad_sync_mesh,
)
from aspec.utils.logging import logger
from aspec.utils.processing import get_assistant_token_ids
from aspec.utils.profiling import TrainProfiler
from aspec.utils.train_dump import extract_gradients, extract_model_weights

if TYPE_CHECKING:
    from aspec.transfer.mooncake.eagle_store import EagleMooncakeStore


def _consumer_participant(
    role: str,
    *,
    dp_rank: int,
    sequence_rank: int | None = None,
) -> str:
    """Build a stable, unique identity for one feature/storage consumer."""

    participant = f"draft_{role}_dp_rank_{dp_rank}"
    if sequence_rank is not None:
        participant = f"{participant}_sequence_rank_{sequence_rank}"
    return participant


@dataclasses.dataclass
class DeferredTrainMetrics:
    """One-step-delayed GPU metric transfer.

    The current step enqueues a non-blocking copy into pinned host memory. The
    next step is submitted before these values are materialized, allowing CPU
    launch work to overlap the previous step's device-to-host dependency.
    """

    cpu_values: torch.Tensor
    source_values: torch.Tensor
    ready_event: torch.cuda.Event
    metric_step: int
    finalize_values: Callable[[list[float]], dict]
    data_time: float = 0.0
    compute_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = dataclasses.field(
        default_factory=list
    )
    optimizer_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = dataclasses.field(
        default_factory=list
    )
    step_events: Optional[tuple[torch.cuda.Event, torch.cuda.Event]] = None

    def finalize(self) -> dict:
        if self.step_events is not None:
            self.step_events[1].synchronize()
        else:
            self.ready_event.synchronize()

        metrics = self.finalize_values(self.cpu_values.tolist())
        metrics["_metrics_step"] = self.metric_step

        if self.step_events is not None:
            metrics["perf/step_time"] = (
                self.step_events[0].elapsed_time(self.step_events[1]) / 1000.0
            )
            metrics["perf/data_time"] = self.data_time
            metrics["perf/compute_time"] = (
                sum(start.elapsed_time(end) for start, end in self.compute_events) / 1000.0
            )
            metrics["perf/optimizer_time"] = (
                sum(start.elapsed_time(end) for start, end in self.optimizer_events) / 1000.0
            )

        # The copy and timing events are complete; release the CUDA source now.
        self.source_values = torch.empty(0)
        return metrics


class Trainer(abc.ABC):
    """Base trainer for async training pipeline.

    Provides shared infrastructure: device mesh, data fetching, training loop
    skeleton, checkpointing, and profiling. Subclasses implement model-specific
    logic via ``init_model``, ``_train_step``, and ``_aggregate_metrics``.
    """

    def __init__(self, args: Namespace):
        self.args = args

        self._setup_device_mesh()
        torch.manual_seed(getattr(args, "seed", 42))

        self.fsdp_cpu_offload = getattr(args, "fsdp_cpu_offload", False)

        self.global_step = 0
        self.model = None
        self.draft_model = None
        self.optimizer: Optional[BF16Optimizer] = None
        self.lr_scheduler = None
        self.data_fetcher: Optional[MooncakeDataFetcher] = None
        self.train_queue = None
        self.mooncake_store: Optional["EagleMooncakeStore"] = None
        self._eval_cache: list[dict] = []
        self._committed_work_by_step: dict[int, list[str]] = {}
        self._work_lifecycle_enabled = False

        self.prof = TrainProfiler(args)

        self.dynamic_loss_mask = getattr(args, "dynamic_loss_mask", False)
        self.last_turn_loss_only = getattr(args, "last_turn_loss_only", False)
        self.assistant_header_ids, self.end_token_ids, self.skip_after_header = (
            get_assistant_token_ids(self.args)
        )

        self.save_debug_train_data = getattr(args, "save_debug_train_data", None)
        self.max_dump_steps = getattr(args, "max_dump_steps", 5)

        self._enable_perf_metrics = getattr(args, "enable_perf_metrics", True)
        self._pending_train_metrics: Optional[DeferredTrainMetrics] = None

        self._io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._eval_cache_save_future: Optional[concurrent.futures.Future] = None

    # ------------------------------------------------------------------
    # Device mesh
    # ------------------------------------------------------------------

    def _setup_device_mesh(self) -> None:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        self.cache_rank = rank

        if is_direct_cp_backend(getattr(self.args, "attention_backend", None)):
            cp_size = int(getattr(self.args, "draft_context_parallel_size", 1))
            if world_size % cp_size:
                raise ValueError(
                    f"world_size ({world_size}) must be divisible by draft CP size ({cp_size})"
                )
            self.cp_group = get_direct_cp_group()
            self.cp_size = cp_size
            self.cp_rank = (
                dist.get_rank(self.cp_group) if self.cp_group is not None else 0
            )
            self.dp_size = world_size // cp_size
            self.dp_rank = rank // cp_size
            self.mesh = init_device_mesh(
                "cuda", mesh_shape=(world_size,), mesh_dim_names=("dp_with_cp",)
            )
            self.dp_group = self.mesh.get_group("dp_with_cp")
            self.dp_mesh = self.mesh
            self.grad_sync_mesh = self.mesh
            logger.info(
                f"[Rank {rank}] Device mesh (Direct CP): world_size={world_size}, "
                f"dp_size={self.dp_size}, dp_rank={self.dp_rank}, "
                f"cp_size={self.cp_size}, cp_rank={self.cp_rank}"
            )
            return

        usp_mesh = None
        if getattr(self.args, "attention_backend", None) == "usp":
            usp_mesh = get_usp_device_mesh()

        if usp_mesh is not None:
            self.mesh = usp_mesh
            self.dp_size = getattr(self.args, "dp_size", world_size)
            self.dp_mesh = usp_mesh["draft_dp"]
            self.grad_sync_mesh = get_usp_grad_sync_mesh()
            if self.grad_sync_mesh is None:
                raise RuntimeError("USP grad sync mesh has not been initialized")
            self.dp_group = usp_mesh.get_group("draft_dp")
            self.dp_rank = dist.get_rank(self.dp_group)
            logger.info(
                f"[Rank {rank}] Device mesh (USP): world_size={world_size}, dp_size={self.dp_size}, "
                f"dp_rank={self.dp_rank}, grad_sync_size={world_size}"
            )
            return

        self.dp_size = world_size
        self.dp_rank = rank

        self.mesh = init_device_mesh("cuda", mesh_shape=(self.dp_size,), mesh_dim_names=("dp",))
        self.dp_group = self.mesh.get_group("dp")
        self.dp_mesh = self.mesh
        self.grad_sync_mesh = self.dp_mesh

        logger.info(
            f"[Rank {rank}] Device mesh (1D): world_size={world_size}, dp_size={self.dp_size}"
        )

    def _get_init_weight_context_manager(self):
        """Meta-device context for non-rank-0 processes to save memory."""

        def cpu_init_weights():
            return torch.device("cpu")

        if dist.get_rank() != 0:
            return init_empty_weights
        return cpu_init_weights

    # ------------------------------------------------------------------
    # Mooncake store
    # ------------------------------------------------------------------

    def init_mooncake_store(
        self,
        mooncake_config: Optional[MooncakeConfig] = None,
    ) -> "EagleMooncakeStore":
        from aspec.transfer.mooncake.eagle_store import EagleMooncakeStore

        if mooncake_config is None:
            mooncake_config = MooncakeConfig.from_flat_args(self.args)

        mooncake_config = dataclasses.replace(
            mooncake_config,
            global_segment_size=0,
            async_put_pool_size=0,
        )

        store = EagleMooncakeStore(mooncake_config)
        store.setup(device=torch.cuda.current_device(), p2p_role="destination")
        self.mooncake_store = store
        logger.info(f"[Rank {self.dp_rank}] EagleMooncakeStore initialized")
        return store

    # ------------------------------------------------------------------
    # Data queue
    # ------------------------------------------------------------------

    def set_train_queue(
        self,
        queue,
        mooncake_config: Optional[MooncakeConfig] = None,
        per_dp_rank_batch_size: int = 1,
        lifecycle_observer=None,
    ) -> None:
        self.train_queue = queue
        self.per_dp_rank_batch_size = per_dp_rank_batch_size
        self._work_lifecycle_enabled = lifecycle_observer is not None
        usp_enabled = getattr(self.args, "attention_backend", None) == "usp"
        direct_cp_enabled = is_direct_cp_backend(
            getattr(self.args, "attention_backend", None)
        ) and int(getattr(self.args, "draft_context_parallel_size", 1)) > 1
        sequence_rank = self.cache_rank if usp_enabled else None
        if direct_cp_enabled:
            sequence_rank = self.cp_rank
        if usp_enabled and per_dp_rank_batch_size != 1:
            raise ValueError("USP requires per_dp_rank_batch_size=1")
        if mooncake_config is not None and self.mooncake_store is None:
            self.init_mooncake_store(mooncake_config)

        packing_enabled = bool(getattr(self.args, "dflash_packing", False))
        if packing_enabled and usp_enabled:
            raise NotImplementedError("dflash_packing is incompatible with USP")
        collator = (
            DFlashPackingCollator(
                max_seq_length=self.args.max_seq_length,
                pad_to_max_length=not (
                    is_fa4_backend(self.args.attention_backend)
                    or is_cudnn_flex_backend(self.args.attention_backend)
                ),
                context_parallel_rank=self.cp_rank if direct_cp_enabled else 0,
                context_parallel_size=int(getattr(self.args, "draft_context_parallel_size", 1)),
                context_shard_alignment=int(
                    getattr(self.args, "draft_context_shard_alignment", 1)
                ),
            )
            if packing_enabled
            else DataCollatorWithPadding(
                usp_enabled=usp_enabled,
                direct_cp_enabled=direct_cp_enabled,
            )
        )

        prefetch_depth = getattr(self.args, "prefetch_depth", 0)
        gpu_device = torch.cuda.current_device()

        # When prefetching, stage data on CPU to avoid GPU contention between
        # background Mooncake TCP transfers and forward/backward compute.
        fetch_device = "cpu" if prefetch_depth > 0 else gpu_device

        inner_fetcher = MooncakeDataFetcher(
            queue=self.train_queue,
            mooncake_store=self.mooncake_store,
            collator=collator,
            device=fetch_device,
            batch_size=per_dp_rank_batch_size,
            assistant_header_ids=self.assistant_header_ids,
            end_token_ids=self.end_token_ids,
            dynamic_loss_mask=self.dynamic_loss_mask,
            last_turn_loss_only=self.last_turn_loss_only,
            skip_after_header=self.skip_after_header,
            min_loss_tokens=getattr(self.args, "min_loss_tokens", 0),
            usp_enabled=usp_enabled,
            direct_cp_enabled=direct_cp_enabled,
            ttt_length=getattr(self.args, "ttt_length", 1),
            max_seq_length=getattr(self.args, "max_seq_length", None),
            feature_contract=getattr(self.args, "aspec_feature_contract", None),
            feature_contract_participant=_consumer_participant(
                "trainer",
                dp_rank=self.dp_rank,
                sequence_rank=sequence_rank,
            ),
            lifecycle_observer=lifecycle_observer,
            packing_enabled=packing_enabled,
            feature_affinity=getattr(
                self.args, "mooncake_feature_affinity", "any"
            ),
        )

        if prefetch_depth > 0:
            self.data_fetcher = PrefetchedDataFetcher(
                inner_fetcher,
                prefetch_depth=prefetch_depth,
                target_device=gpu_device,
            )
            logger.info(
                f"[Rank {self.dp_rank}] Prefetched data fetcher initialized "
                f"(batch_size={per_dp_rank_batch_size}, prefetch_depth={prefetch_depth}, "
                f"staging=CPU, target={gpu_device})"
            )
        else:
            self.data_fetcher = inner_fetcher
            logger.info(
                f"[Rank {self.dp_rank}] Data fetcher initialized with batch_size={per_dp_rank_batch_size}"
            )

    # ------------------------------------------------------------------
    # Eval queue & CPU cache
    # ------------------------------------------------------------------

    def set_eval_queue(
        self,
        queue,
        mooncake_config: Optional[MooncakeConfig] = None,
        per_dp_rank_batch_size: int = 1,
        lifecycle_observer=None,
    ) -> None:
        usp_enabled = getattr(self.args, "attention_backend", None) == "usp"
        direct_cp_enabled = is_direct_cp_backend(
            getattr(self.args, "attention_backend", None)
        ) and int(getattr(self.args, "draft_context_parallel_size", 1)) > 1
        sequence_rank = self.cache_rank if usp_enabled else None
        if direct_cp_enabled:
            sequence_rank = self.cp_rank
        if mooncake_config is not None and self.mooncake_store is None:
            self.init_mooncake_store(mooncake_config)

        collator = DataCollatorWithPadding(
            usp_enabled=usp_enabled,
            direct_cp_enabled=direct_cp_enabled,
        )

        self._eval_data_fetcher = MooncakeDataFetcher(
            queue=queue,
            mooncake_store=self.mooncake_store,
            collator=collator,
            device=torch.cuda.current_device(),
            batch_size=per_dp_rank_batch_size,
            assistant_header_ids=self.assistant_header_ids,
            end_token_ids=self.end_token_ids,
            dynamic_loss_mask=self.dynamic_loss_mask,
            last_turn_loss_only=self.last_turn_loss_only,
            skip_after_header=self.skip_after_header,
            min_loss_tokens=getattr(self.args, "min_loss_tokens", 0),
            usp_enabled=usp_enabled,
            direct_cp_enabled=direct_cp_enabled,
            ttt_length=getattr(self.args, "ttt_length", 1),
            max_seq_length=getattr(self.args, "max_seq_length", None),
            feature_contract=getattr(self.args, "aspec_feature_contract", None),
            feature_contract_participant=_consumer_participant(
                "eval",
                dp_rank=self.dp_rank,
                sequence_rank=sequence_rank,
            ),
            lifecycle_observer=lifecycle_observer,
        )
        self._eval_collator = collator
        self._eval_cache: list[dict] = []
        logger.info(
            f"[Rank {self.dp_rank}] Eval data fetcher initialized "
            f"with batch_size={per_dp_rank_batch_size}"
        )

    def cache_eval_samples(self, count: int) -> int:
        for sample in itertools.islice(self._eval_data_fetcher, count):
            receive_leases = _take_p2p_receive_leases(sample)
            try:
                # The fetcher has already consumed and terminally released the eval
                # work.  Its identity is lifecycle metadata for that transfer, not a
                # model input, and the fetched value is already a batch of size one.
                # Keeping the collator-produced list here would make re-collation
                # treat a list[work] as one work item.
                cpu_sample = {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in sample.items()
                    if k != "_aspec_work"
                }
            finally:
                for lease in receive_leases:
                    lease.release()
            self._eval_cache.append(cpu_sample)
        return len(self._eval_cache)

    def save_eval_cache(self, cache_dir: str) -> None:
        if not getattr(self, "_eval_cache", None):
            return
        self._wait_for_eval_cache_save()

        cache_snapshot = list(self._eval_cache)
        rank = self.cache_rank

        def _save() -> None:
            os.makedirs(cache_dir, exist_ok=True)
            path = os.path.join(cache_dir, f"eval_rank_{rank}.pt")
            temporary = None
            try:
                # Identical feature consumers intentionally share a cache key.
                # Their writers must not share a staging filename; either
                # complete, identity-bound snapshot may atomically win.
                with tempfile.NamedTemporaryFile(
                    dir=cache_dir, prefix=f".eval_rank_{rank}.", suffix=".tmp", delete=False,
                ) as stream:
                    temporary = stream.name
                    torch.save({"identity": getattr(self, "_eval_cache_identity", None),
                                "samples": cache_snapshot}, stream)
                os.replace(temporary, path)
            finally:
                if temporary and os.path.exists(temporary):
                    os.unlink(temporary)
            logger.info(f"[Rank {rank}] Saved {len(cache_snapshot)} eval batches to {path}")

        self._eval_cache_save_future = self._io_executor.submit(_save)

    def _wait_for_eval_cache_save(self) -> None:
        fut = self._eval_cache_save_future
        if fut is not None:
            fut.result()
            self._eval_cache_save_future = None

    def load_eval_cache(self, cache_dir: str, identity=None) -> int:
        # Safe guard to wait for eval cache save to complete.
        self._wait_for_eval_cache_save()
        self._eval_cache_identity = identity
        self._eval_cache = []
        path = os.path.join(cache_dir, f"eval_rank_{self.cache_rank}.pt")
        if not os.path.exists(path):
            return 0
        try:
            payload = torch.load(path, weights_only=False, mmap=True)
            if isinstance(payload, dict) and "identity" in payload:
                if payload["identity"] != identity:
                    self._eval_cache = []
                    return 0
                payload = payload["samples"]
            elif identity is not None:
                # Unbound legacy tensors are not safe for current production eval.
                self._eval_cache = []
                return 0
            self._eval_cache = payload
        except Exception as e:
            logger.warning(f"[Rank {self.dp_rank}] Corrupt eval cache at {path}, ignoring: {e}")
            return 0
        if not isinstance(self._eval_cache, list) or not all(
            isinstance(sample, dict) for sample in self._eval_cache
        ):
            logger.warning(f"[Rank {self.dp_rank}] Invalid eval cache at {path}, ignoring")
            self._eval_cache = []
            return 0
        # Caches created before eval work identity was separated from model inputs
        # may contain a collated list under this private key.  It is safe to drop:
        # transfer ownership reached its terminal state before the sample was cached.
        for sample in self._eval_cache:
            sample.pop("_aspec_work", None)
        logger.info(
            f"[Rank {self.dp_rank}] Loaded {len(self._eval_cache)} eval batches from {path}"
        )
        return len(self._eval_cache)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train_from_queue(self, step: int, num_batches: int) -> dict:
        if self.data_fetcher is None:
            raise RuntimeError("Data fetcher not initialized. Call set_train_queue first.")
        perf = self._enable_perf_metrics
        if perf:
            t0 = time.time()
            step_start = torch.cuda.Event(enable_timing=True)
            step_end = torch.cuda.Event(enable_timing=True)
            step_start.record()
        metrics = self._train_core_from_queue(step=step, num_batches=num_batches)
        if isinstance(metrics, DeferredTrainMetrics):
            if perf:
                step_end.record()
                metrics.step_events = (step_start, step_end)
            previous_metrics = self._pending_train_metrics
            self._pending_train_metrics = metrics
            metrics = previous_metrics.finalize() if previous_metrics is not None else {}
        elif perf:
            # Immediate metric aggregation has already synchronized the step.
            metrics["perf/step_time"] = time.time() - t0
        self.prof.step(step=step)
        return metrics

    def flush_pending_metrics(self) -> dict:
        """Materialize the final deferred training metrics, if any."""
        pending = self._pending_train_metrics
        result = pending.finalize() if pending is not None else {}
        self._pending_train_metrics = None
        return result

    def get_feature_contract_status(self) -> dict:
        """Return consumer-side target-feature observations for Evidence."""
        if self.data_fetcher is None:
            direct_cp_enabled = is_direct_cp_backend(
                getattr(self.args, "attention_backend", None)
            ) and int(getattr(self.args, "draft_context_parallel_size", 1)) > 1
            sequence_rank = self.cache_rank if getattr(
                self.args, "attention_backend", None
            ) == "usp" else None
            if direct_cp_enabled:
                sequence_rank = self.cp_rank
            return {
                "participant": _consumer_participant(
                    "trainer",
                    dp_rank=self.dp_rank,
                    sequence_rank=sequence_rank,
                ),
                "role": "consumer",
                "contract_id": None,
                "state": "not_ready",
            }
        status = self.data_fetcher.get_feature_contract_status()
        draft = getattr(self, "draft_model", None)
        layers = getattr(draft, "layers", None)
        if layers is None:
            layers = [draft.midlayer] if hasattr(draft, "midlayer") else []
        observations = [
            getattr(getattr(layer, "self_attn", None), "attention_observation", None)
            for layer in layers
        ]
        if observations and all(
            isinstance(item, dict) and item.get("qkv_format") for item in observations
        ):
            status = dict(status)
            execution = dict(status.get("execution_metadata") or {})
            execution.update({
                "draft_qkv_format": sorted({item["qkv_format"] for item in observations}),
                "draft_attention_backward": sorted({
                    item["backward_strategy"] for item in observations
                }),
                "draft_thd_backward_validated": [
                    item["backward_observed"] or item["empty_local_partition"]
                    for item in observations
                ],
                "draft_thd_kernel_backend": sorted({
                    item.get("kernel_backend", "fa4") for item in observations
                }),
                "draft_thd_kernel_mode": sorted({
                    item.get("kernel_mode", item.get("fa4_mode"))
                    for item in observations
                }),
                "draft_thd_fa4_mode": sorted({
                    item["fa4_mode"] for item in observations if "fa4_mode" in item
                }),
                "draft_thd_empty_partitions": [
                    item["empty_local_partition"] for item in observations
                ],
            })
            status["execution_metadata"] = execution
            status["draft_attention_observations"] = [dict(item) for item in observations]
        return status

    def pop_committed_work(self, optimizer_step: int) -> list[str]:
        """Return exactly the work consumed by one completed optimizer step."""

        if optimizer_step not in self._committed_work_by_step:
            raise RuntimeError(
                f"No committed work recorded for optimizer step {optimizer_step} "
                f"on DP rank {self.dp_rank}"
            )
        return self._committed_work_by_step.pop(optimizer_step)

    def _train_core_from_queue(self, step: int, num_batches: int) -> dict:
        """Training loop skeleton.

        Calls ``_train_step`` for each micro-batch, wrapped with debug logging
        and optional dump extraction.  One optimizer step is performed after
        all micro-batches.  ``global_step`` counts optimizer steps.
        """
        self.model.train()
        accumulation_steps = num_batches

        all_step_metrics: list[dict] = []
        step_execution_ids: list[str] = []
        grad_norm = None

        perf = self._enable_perf_metrics
        if perf:
            data_time = 0.0
            compute_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
            t_data_start = time.time()

        # Gradient sync control for micro-batch accumulation.
        # FSDP2 fully_shard and composable replicate both expose
        # set_requires_gradient_sync(bool). Keep no_sync() as a fallback for a
        # conventional DDP wrapper.
        _model = getattr(self.model, "_orig_mod", self.model)
        _set_grad_sync = getattr(_model, "set_requires_gradient_sync", None)
        _no_sync = getattr(_model, "no_sync", None) if _set_grad_sync is None else None

        batches = self.prof.iterate_train_actor(self._iter_batches_from_queue(num_batches))
        for batch_idx, batch in enumerate(batches):
            is_last = batch_idx == num_batches - 1
            work_items = batch.pop("_aspec_work", None)
            if work_items is None and self._work_lifecycle_enabled:
                raise RuntimeError(
                    f"Training batch {batch_idx} at step {step} has no ASpec work identity"
                )
            if work_items is not None:
                step_execution_ids.extend(item["execution_id"] for item in work_items)

            if perf:
                data_time += time.time() - t_data_start
                evt_start = torch.cuda.Event(enable_timing=True)
                evt_end = torch.cuda.Event(enable_timing=True)
                evt_start.record()

            if logger.isEnabledFor(logging.DEBUG):
                self._log_batch_debug(batch, step, batch_idx, num_batches)

            if _set_grad_sync is not None:
                _set_grad_sync(is_last)
                ctx = nullcontext()
            else:
                ctx = _no_sync() if (_no_sync is not None and not is_last) else nullcontext()

            receive_leases = _take_p2p_receive_leases(batch)
            try:
                with ctx:
                    step_metrics = self._train_step(
                        batch=batch,
                        accumulation_steps=accumulation_steps,
                        step=step,
                        batch_idx=batch_idx,
                        num_batches=num_batches,
                    )
            finally:
                for receive_lease in receive_leases:
                    receive_lease.release()

            if is_last:
                self._maybe_dump(batch, step_metrics, step, batch_idx)
                _evt_opt_s = torch.cuda.Event(enable_timing=True)
                _evt_opt_e = torch.cuda.Event(enable_timing=True)
                _evt_opt_s.record()
                grad_norm = self.optimizer.step()
                self._after_optimizer_step(all_step_metrics + [step_metrics], grad_norm)
                _evt_opt_e.record()
                step_metrics["_opt_events"] = (_evt_opt_s, _evt_opt_e)

            if perf:
                evt_end.record()
                compute_events.append((evt_start, evt_end))

            all_step_metrics.append(step_metrics)

            if perf:
                t_data_start = time.time()

        if len(all_step_metrics) != num_batches:
            raise RuntimeError(
                f"Training step {step} received {len(all_step_metrics)} batches; "
                f"expected {num_batches}"
            )
        self.global_step += 1
        if self._work_lifecycle_enabled:
            if self.global_step in self._committed_work_by_step:
                raise RuntimeError(f"Duplicate committed-work record for step {self.global_step}")
            self._committed_work_by_step[self.global_step] = step_execution_ids

        metrics = self._aggregate_metrics(all_step_metrics, step, grad_norm=grad_norm)
        if perf:
            optimizer_events = []
            for m in all_step_metrics:
                if "_opt_events" in m:
                    optimizer_events.append(m["_opt_events"])

            if isinstance(metrics, DeferredTrainMetrics):
                metrics.data_time = data_time
                metrics.compute_events = compute_events
                metrics.optimizer_events = optimizer_events
            else:
                # Immediate aggregation synchronized CUDA, so event queries are ready.
                metrics["perf/data_time"] = data_time
                metrics["perf/compute_time"] = (
                    sum(start.elapsed_time(end) for start, end in compute_events) / 1000.0
                )
                metrics["perf/optimizer_time"] = (
                    sum(start.elapsed_time(end) for start, end in optimizer_events) / 1000.0
                )

        return metrics

    def _iter_batches_from_queue(self, num_batches: int):
        yield from itertools.islice(self.data_fetcher, num_batches)

    # ------------------------------------------------------------------
    # Checkpointing & persistence
    # ------------------------------------------------------------------

    def save_model(self, step: int, force_sync: bool = False) -> None:
        if not self.args.checkpoint_dir:
            return
        checkpoint.save(self, step=step)

    def discard_inference_outputs(self, outputs: list) -> int:
        """Release generated outputs that were never assigned to training."""

        if not outputs:
            return 0
        if self.mooncake_store is None:
            raise RuntimeError("cannot discard inference outputs before Mooncake is initialized")
        for output in outputs:
            shapes = output.tensor_shapes or {}
            self.mooncake_store.discard_unconsumed_eagle3_tensors(
                output.mooncake_key,
                has_last_hidden_states="last_hidden_states" in shapes,
                has_target="target" in shapes,
                **({"has_mtp_hidden_states": True} if "mtp_hidden_states" in shapes else {}),
            )
        return len(outputs)

    def _write_serving_artifacts(self, model, state_dict: dict, output_dir: str) -> None:
        if hasattr(model, "config"):
            config = (
                model.config_for_serving() if hasattr(model, "config_for_serving") else model.config
            )
            if isinstance(config, dict):
                with open(os.path.join(output_dir, "config.json"), "w") as config_file:
                    json.dump(config, config_file, indent=2)
            else:
                config.save_pretrained(output_dir)
        if hasattr(model, "state_dict_for_serving"):
            state_dict = model.state_dict_for_serving(state_dict)
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

    def _synchronize_serving_error(self, error: Exception | None) -> str | None:
        if not dist.is_initialized():
            return None if error is None else f"{type(error).__name__}: {error}"

        group = get_gloo_group()
        errors = [None] * dist.get_world_size(group)
        local_error = None if error is None else f"{type(error).__name__}: {error}"
        dist.all_gather_object(errors, local_error, group=group)
        return next((message for message in errors if message is not None), None)

    def save_draft_model_for_serving(self, output_dir: str) -> None:
        """Save draft model in HuggingFace format for serving update."""
        os.makedirs(output_dir, exist_ok=True)

        model = self.draft_model
        if hasattr(model, "module"):
            model = model.module
        custom_schema = hasattr(model, "config_for_serving") or hasattr(
            model, "state_dict_for_serving"
        )

        save_error = None
        try:
            state_dict = get_model_state_dict(
                self.draft_model,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )

            if self.dp_rank == 0:
                if hasattr(model, "save_pretrained"):
                    if custom_schema:
                        self._write_serving_artifacts(model, state_dict, output_dir)
                    else:
                        if hasattr(model, "config"):
                            model.config.save_pretrained(output_dir)
                        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
                    logger.info(f"[Rank {self.dp_rank}] Saved draft model to {output_dir}")
                else:
                    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
                    logger.info(
                        f"[Rank {self.dp_rank}] Saved draft model state dict to {output_dir}"
                    )

        except Exception as error:
            save_error = error

        if custom_schema:
            synchronized_error = self._synchronize_serving_error(save_error)
            if synchronized_error is None:
                return

            strategy = getattr(getattr(self, "args", None), "fsdp_strategy", "REPLICATE")
            if strategy.upper() == "FULL_SHARD":
                if save_error is not None:
                    raise save_error
                raise RuntimeError(
                    f"Failed to save a full serving state dict on another rank: "
                    f"{synchronized_error}"
                )

            logger.warning(
                f"[Rank {self.dp_rank}] Failed to save with FSDP2 state dict, trying fallback: "
                f"{synchronized_error}"
            )
            fallback_error = None
            try:
                if self.dp_rank == 0:
                    self._write_serving_artifacts(model, model.state_dict(), output_dir)
                    logger.info(
                        f"[Rank {self.dp_rank}] Saved draft model using save_pretrained to {output_dir}"
                    )
            except Exception as error:
                fallback_error = error

            synchronized_error = self._synchronize_serving_error(fallback_error)
            if synchronized_error is not None:
                if fallback_error is not None:
                    raise fallback_error
                raise RuntimeError(
                    f"Failed to save serving artifacts on another rank: {synchronized_error}"
                )
            return

        if save_error is not None:
            logger.warning(
                f"[Rank {self.dp_rank}] Failed to save with FSDP2 state dict, trying fallback: "
                f"{save_error}"
            )
            if self.dp_rank == 0:
                if hasattr(model, "save_pretrained"):
                    model.save_pretrained(output_dir)
                    logger.info(
                        f"[Rank {self.dp_rank}] Saved draft model using save_pretrained to {output_dir}"
                    )
                else:
                    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
                    logger.info(
                        f"[Rank {self.dp_rank}] Saved draft model state dict to {output_dir}"
                    )

        if dist.is_initialized():
            dist.barrier()

    def load_checkpoint(self) -> dict | None:
        return checkpoint.load(self)

    def close(self) -> None:
        try:
            self._wait_for_eval_cache_save()
            if self.data_fetcher is not None and hasattr(self.data_fetcher, "close"):
                self.data_fetcher.close()
            self._io_executor.shutdown(wait=True)
            if self.mooncake_store is not None and hasattr(self.mooncake_store, "close"):
                self.mooncake_store.close()
                logger.info(f"[Rank {self.dp_rank}] EagleMooncakeStore closed")
        finally:
            try:
                self.prof.close()
            except Exception:
                logger.exception("Failed to close training profiler")

    # ------------------------------------------------------------------
    # Debug logging & dump helpers
    # ------------------------------------------------------------------

    def _log_batch_debug(self, batch: dict, step: int, batch_idx: int, num_batches: int) -> None:
        batch_size = batch["input_ids"].shape[0]
        seq_len = batch["input_ids"].shape[1]
        hs_shape = (
            tuple(batch["hidden_states"].shape) if batch.get("hidden_states") is not None else None
        )
        logger.debug(
            f"step={step} batch={batch_idx}/{num_batches} | "
            f"batch_size={batch_size}, seq_len={seq_len}, "
            f"input_ids={tuple(batch['input_ids'].shape)}, hidden_states={hs_shape}"
        )

    def _should_dump_step(self) -> bool:
        return bool(self.save_debug_train_data and (self.global_step + 1) <= self.max_dump_steps)

    def _maybe_dump(self, batch: dict, step_metrics: dict, step: int, batch_idx: int) -> None:
        if not self._should_dump_step():
            return

        self._save_dump_data(
            batch=batch,
            step_metrics=step_metrics,
            gradients=extract_gradients(self.model),
            model_weights=extract_model_weights(self.model),
            step=step,
            batch_idx=batch_idx,
        )

    def _save_dump_data(
        self,
        *,
        batch: dict,
        step_metrics: dict,
        gradients: dict,
        model_weights: dict,
        step: int,
        batch_idx: int,
    ) -> None:
        """Save debug dump data. Override in subclass for model-specific dumps."""

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def init_model(self, *args, **kwargs) -> int:
        """Initialize model, optimizer, and load checkpoint.

        Returns:
            Start step.
        """
        ...

    @abc.abstractmethod
    def _train_step(
        self,
        batch: dict,
        accumulation_steps: int,
        step: int,
        batch_idx: int,
        num_batches: int,
    ) -> dict:
        """Run forward + backward for a single micro-batch.

        The optimizer step is handled by the base class after the last
        micro-batch — do NOT call ``self.optimizer.step()`` here.

        Returns:
            Dict of step-level metrics for later aggregation.
        """
        ...

    def _after_optimizer_step(self, all_step_metrics: list[dict], grad_norm: torch.Tensor) -> None:
        """Update model control state once, after the optimizer and before checkpoint/commit."""

    @abc.abstractmethod
    def _aggregate_metrics(
        self, all_step_metrics: list[dict], step: int, *, grad_norm: torch.Tensor = None
    ) -> dict:
        """Aggregate per-step metrics into a single metrics dict.

        Called once per optimizer step after all micro-batches.
        """
        ...
