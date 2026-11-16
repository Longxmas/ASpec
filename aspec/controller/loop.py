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

"""Pipeline training loop: main training loop with sync training and async inference."""

import math
import re
import shutil
import statistics
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Mapping

import ray
import wandb
from tqdm import tqdm

from aspec.config.mooncake_config import resolve_rdma_routes
from aspec.controller.eval import (
    eval_selection_metric,
    generate_eval_cache,
    run_eval,
    setup_eval,
    update_checkpoint_eval_meta,
)
from aspec.ray.owner_fence import SourceTeardownBlockedError, wait_for_owner_exit
from aspec.training.checkpoint import (
    load_checkpoint_work_state,
    prepare_checkpoint_save,
    save_checkpoint_work_state,
)
from aspec.utils.logging import get_tb_writer, logger


def _write_training_metrics(metrics: dict, train_step: int, inference_step: int, *, tracking=None) -> None:
    loss = metrics.get("train/avg_loss")
    if isinstance(loss, (int, float)):
        logger.info(f"TRAIN_STEP step={train_step} loss={loss:.6f}")

    if tracking is not None:
        tracking.log(metrics, train_step=train_step, inference_step=inference_step)
        return
    if getattr(wandb, "run", None) is not None:
        wandb.log(metrics)

    tb_writer = get_tb_writer()
    if tb_writer is not None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                scalar_step = inference_step if key.startswith("inference/") else train_step
                tb_writer.add_scalar(key, value, scalar_step)


def _maybe_sync_draft_weights(args, completed_steps, train_group, inference_engines):
    """Sync draft model weights to inference engines (decode mode only)."""
    weight_sync_enabled = getattr(args, "decode_weight_sync_enabled", False)
    weight_sync_interval = getattr(args, "decode_weight_sync_interval", 500)
    if not (
        getattr(args, "train_with_decode", False)
        and weight_sync_enabled
        and inference_engines
        and weight_sync_interval > 0
        and completed_steps > 0
        and completed_steps % weight_sync_interval == 0
    ):
        return

    # NOTE: uses local tmp dir; for multi-node, ensure output_dir is on shared filesystem.
    tmp_dir = tempfile.mkdtemp(prefix="draft_weight_sync_")
    try:
        logger.info(f"Step {completed_steps}: saving draft model to {tmp_dir}")
        train_group.save_draft_model_for_serving(tmp_dir)

        logger.info(f"Step {completed_steps}: updating {len(inference_engines)} engine(s)")
        update_results = ray.get(
            [engine.update_weights_from_disk.remote(tmp_dir) for engine in inference_engines]
        )
        for i, res in enumerate(update_results):
            log_fn = logger.info if res.get("success") else logger.warning
            log_fn(f"Engine {i}: success={res.get('success')}, message={res.get('message')}")
    except Exception:
        logger.exception(f"Step {completed_steps}: weight sync failed, skipping")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info(f"Cleaned up temp dir {tmp_dir}")


def _is_save_interval_step(step: int, interval: int) -> bool:
    return interval > 0 and step % interval == 0


def _validate_resume_optimizer_frontier(
    *, start_step: int, resume_state: Mapping[str, Any] | None, continual_training: bool
) -> None:
    """Keep optimizer-step equality exclusive to an exact resume."""

    if (
        resume_state is not None
        and not continual_training
        and int(resume_state["optimizer_step"]) != int(start_step)
    ):
        raise RuntimeError(
            "Loaded trainer step does not match checkpoint work state: "
            f"trainer={start_step}, work_state={resume_state['optimizer_step']}"
        )


def _save_training_checkpoint(args, train_group, controller, completed_steps: int) -> dict:
    """Save distributed state, then atomically attach controller commit state."""

    prepare_checkpoint_save(args.checkpoint_dir, completed_steps)
    train_group.save_model(completed_steps, force_sync=True)
    from aspec.ray.node_local_files import clear_staging, collect_checkpoint

    collect_checkpoint(args, completed_steps)
    work_state = ray.get(
        controller.get_checkpoint_work_state.remote(),
        timeout=float(getattr(args, "optimizer_step_timeout_seconds", 1800.0)),
    )
    result = save_checkpoint_work_state(
        args.checkpoint_dir,
        completed_steps,
        work_state,
    )
    clear_staging(args, completed_steps)
    return result


class TrainerStepTimeoutError(TimeoutError):
    """One or more trainer owner processes failed to finish an optimizer step."""


def _remaining_optimizer_step_time(
    *, deadline: float, timeout_seconds: float, phase: str
) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TrainerStepTimeoutError(
            "Timed out waiting for optimizer step "
            f"after {timeout_seconds:.1f}s; phase={phase}"
        )
    return remaining



def _get_before_deadline(ref, *, deadline: float, timeout_seconds: float, phase: str):
    """Bound a blocking RPC by the enclosing operation's remaining budget."""
    remaining = _remaining_optimizer_step_time(
        deadline=deadline, timeout_seconds=timeout_seconds, phase=phase,
    )
    try:
        return ray.get(ref, timeout=remaining)
    except ray.exceptions.GetTimeoutError as exc:
        raise TrainerStepTimeoutError(
            f"Timed out waiting for {phase} after {timeout_seconds:.1f}s"
        ) from exc


def _raise_if_inference_finished(inference_future) -> None:
    """Surface a stopped inference loop while training still needs source work."""

    if inference_future is None:
        return
    ready, _ = ray.wait([inference_future], num_returns=1, timeout=0)
    if not ready:
        return
    ray.get(ready[0])
    raise RuntimeError("inference manager stopped before training completed")


def _wait_for_trainer_group(train_futures, *, timeout_seconds: float, check_progress=None):
    """Return rank-ordered results while surfacing the first failed rank."""

    if timeout_seconds <= 0:
        raise ValueError("optimizer step timeout must be positive")
    results = [None] * len(train_futures)
    pending = {ref: rank for rank, ref in enumerate(train_futures)}
    deadline = time.monotonic() + timeout_seconds
    while pending:
        if check_progress is not None:
            check_progress()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TrainerStepTimeoutError(
                "Timed out waiting for trainer optimizer step "
                f"after {timeout_seconds:.1f}s; pending_ranks={sorted(pending.values())}"
            )
        ready, _ = ray.wait(
            list(pending),
            num_returns=1,
            timeout=min(1.0, remaining),
        )
        if not ready:
            continue
        for ref in ready:
            rank = pending.pop(ref)
            results[rank] = ray.get(ref)
    return results


def _dispatch_ahead(
    controller,
    *,
    optimizer_step: int,
    first_microstep: int,
    target_microsteps: int,
    stop_event: threading.Event,
    deadline: float,
    timeout_seconds: float,
) -> tuple[int, float]:
    """Dispatch a bounded next-step prefix while the current step computes."""

    get_before_deadline = partial(
        _get_before_deadline, deadline=deadline, timeout_seconds=timeout_seconds,
        phase="pipelined dispatch",
    )

    dispatched = first_microstep
    started = time.monotonic()
    attempts = 0
    while dispatched < target_microsteps and not stop_event.is_set():
        attempts += 1
        if get_before_deadline(
            controller.try_dispatch_batch.remote(
                optimizer_step=optimizer_step,
                microstep=dispatched,
            )
        ):
            dispatched += 1
            continue
        if attempts % 100 == 0:
            status = get_before_deadline(controller.get_full_status.remote())
            logger.debug(
                "Pipelined dispatch step %d: %d/%d batches, pool_size=%d, need=%d",
                optimizer_step,
                dispatched,
                target_microsteps,
                status["sample_pool_size"],
                status["dispatch_batch_size"],
            )
        stop_event.wait(0.01)
    return dispatched - first_microstep, time.monotonic() - started


def _validate_source_work_budget(args, *, dataset_size: int, dp_size: int) -> int | None:
    """Validate the already-resolved finite source pass against loaded data."""

    source_work_budget = getattr(args, "source_work_budget", None)
    if source_work_budget is None:
        return None
    source_work_budget = int(source_work_budget)
    requested = int(
        getattr(args, "requested_source_work_budget", source_work_budget)
    )
    if requested <= 0 or requested > dataset_size:
        raise ValueError(
            "requested source_work_budget must be in [1, dataset_size]: "
            f"requested={requested}, scheduled={source_work_budget}, "
            f"dataset_size={dataset_size}"
        )
    packing_enabled = bool(getattr(args, "dflash_packing", False))
    alignment = int(
        getattr(
            args,
            "source_work_alignment",
            dp_size if packing_enabled else args.global_batch_size,
        )
    )
    if source_work_budget < alignment or source_work_budget % alignment:
        raise ValueError(
            "scheduled source_work_budget must contain complete executable work: "
            f"budget={source_work_budget}, alignment={alignment}"
        )
    minimum_executable_work = (
        dp_size * int(getattr(args, "draft_accumulation_steps", 1))
        if packing_enabled
        else int(args.global_batch_size)
    )
    if source_work_budget < minimum_executable_work:
        raise ValueError(
            "scheduled source_work_budget cannot form one complete optimizer step: "
            f"budget={source_work_budget}, minimum={minimum_executable_work}"
        )
    return source_work_budget


def _cleanup_old_checkpoints(checkpoint_dir: str | None, max_checkpoints: int) -> None:
    """Delete old checkpoints, keeping only the most recent `max_checkpoints`.

    Checkpoint directories are named ``iter_NNNNNNN`` where N is the step number.
    The ``latest_checkpointed_iteration.txt`` and ``best_*`` files are preserved.
    """
    if not checkpoint_dir or max_checkpoints <= 0:
        return

    base_dir = Path(checkpoint_dir).expanduser()
    if not base_dir.exists():
        return

    # Find all iter_* directories, sorted by step number
    iter_dirs = sorted(
        (d for d in base_dir.iterdir() if d.is_dir() and re.fullmatch(r"iter_\d+", d.name)),
        key=lambda d: int(re.search(r"\d+", d.name).group()),
    )

    if len(iter_dirs) <= max_checkpoints:
        return

    # Best is an additional protected asset, not merely a preserved text pointer.
    best = base_dir / "best_checkpointed_iteration.txt"
    protected = None
    if best.is_file():
        try:
            protected = base_dir / f"iter_{int(best.read_text().strip()):07d}"
        except ValueError as exc:
            raise RuntimeError(f"Invalid best checkpoint pointer: {best}") from exc
        if not protected.is_dir():
            raise RuntimeError(f"Best checkpoint is missing: {protected}")
    to_delete = [d for d in iter_dirs[:-max_checkpoints] if d != protected]
    for old_dir in to_delete:
        logger.info(f"Removing old checkpoint: {old_dir}")
        try:
            shutil.rmtree(old_dir)
        except OSError as e:
            logger.warning(f"Failed to remove old checkpoint {old_dir}: {e}")


def _drain_unused_inference_outputs(
    args,
    controller,
    inference_future,
    train_group,
) -> int:
    """Release unassigned outputs while the inference manager finishes draining."""

    timeout_seconds = float(
        getattr(args, "inference_shutdown_timeout_seconds", 120.0)
    )
    deadline = time.monotonic() + timeout_seconds
    released = 0
    manager_done = False

    def get_before_deadline(ref):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("timed out draining unused inference outputs")
        return ray.get(ref, timeout=remaining)

    def discard(outputs):
        from aspec.ray.train_group import discard_actor_batches

        batches = discard_actor_batches(train_group, outputs, resolve=get_before_deadline)
        refs = [actor.discard_inference_outputs.remote(chunk) for actor, chunk in batches]
        return sum(get_before_deadline(ref) for ref in refs)

    while time.monotonic() < deadline:
        outputs = get_before_deadline(
            controller.pop_unused_inference_results.remote(32)
        )
        if outputs:
            discarded = discard(outputs)
            if discarded != len(outputs):
                raise RuntimeError(
                    f"Trainer discarded {discarded} of {len(outputs)} claimed outputs"
                )
            get_before_deadline(
                controller.confirm_discarded_inference_results.remote(outputs)
            )
            released += discarded
            continue

        ready, _ = ray.wait(
            [inference_future],
            num_returns=1,
            timeout=min(0.05, max(0.0, deadline - time.monotonic())),
        )
        if ready:
            manager_done = True
            # The manager may forward its last result in the same turn it exits.
            outputs = get_before_deadline(
                controller.pop_unused_inference_results.remote(32)
            )
            if outputs:
                discarded = discard(outputs)
                if discarded != len(outputs):
                    raise RuntimeError(
                        f"Trainer discarded {discarded} of {len(outputs)} claimed outputs"
                    )
                get_before_deadline(
                    controller.confirm_discarded_inference_results.remote(outputs)
                )
                released += discarded
                continue
            break
    if not manager_done:
        raise TimeoutError("timed out draining unused inference outputs")
    ray.get(inference_future)
    return released


def _safe_training_cleanup(
    args,
    inference_manager,
    inference_future,
    inference_engines=None,
    controller=None,
    train_group=None,
    force_kill_trainers: bool = False,
    closed_reader_fence: dict | None = None,
) -> dict:
    """Bounded teardown with explicit data-plane ownership finalization."""
    operations = []
    lifecycle = None
    feature_contract = None
    feature_contract_error = None
    read_owner_fence = closed_reader_fence if train_group is None else None
    source_teardown_safe = True
    external_target = (getattr(args, "inference_engine_type", None) == "sgl"
                       and bool(getattr(args, "target_service", None)))

    def record(component: str, state: str, error: Exception | None = None) -> None:
        item = {"component": component, "state": state}
        if error is not None:
            item["error"] = f"{type(error).__name__}: {error}"
        operations.append(item)

    if inference_manager is not None and force_kill_trainers:
        try:
            ray.kill(inference_manager, no_restart=True)
            record("inference_manager", "force_killed")
        except Exception as exc:
            logger.warning(f"Failed to terminate inference manager actor: {exc}")
            record("inference_manager", "kill_failed", exc)
    elif inference_manager is not None:
        try:
            ray.get(inference_manager.stop.remote(), timeout=10)
            record("inference_manager_stop", "acknowledged")
        except Exception as exc:
            logger.warning(f"Failed to stop inference manager: {exc}")
            record("inference_manager_stop", "failed", exc)
    if controller is not None:
        try:
            ray.get(controller.shutdown.remote(), timeout=10)
            record("training_queues", "stop_signaled")
        except Exception as exc:
            logger.warning(f"Failed to signal training queues to stop: {exc}")
            record(
                "training_queues",
                "stop_unavailable_before_force"
                if force_kill_trainers
                else "stop_failed",
                exc,
            )

    if (
        inference_manager is not None
        and inference_future is not None
        and not force_kill_trainers
    ):
        try:
            if controller is not None and train_group is not None:
                released = _drain_unused_inference_outputs(
                    args,
                    controller,
                    inference_future,
                    train_group,
                )
                record("unused_inference_outputs", "released")
                operations[-1]["count"] = released
            else:
                ray.get(
                    inference_future,
                    timeout=float(
                        getattr(args, "inference_shutdown_timeout_seconds", 120.0)
                    ),
                )
            record("inference_manager_run_loop", "joined")
        except Exception as exc:
            logger.warning(
                f"Inference manager run loop exited with error during cleanup: {exc}"
            )
            record("inference_manager_run_loop", "failed", exc)
            force_kill_trainers = isinstance(exc, TimeoutError) or (
                exc.__class__.__name__ == "GetTimeoutError"
            )
            try:
                ray.cancel(inference_future, force=True)
                record("inference_manager_run_loop", "cancelled")
            except Exception as cancel_exc:
                logger.warning(f"Failed to cancel inference manager run loop: {cancel_exc}")
                record("inference_manager_cancel", "failed", cancel_exc)
    manager_status = None
    if inference_manager is not None and not force_kill_trainers:
        try:
            manager_status = ray.get(inference_manager.get_status.remote(), timeout=10)
            record("inference_manager_status", "captured")
        except Exception as exc:
            logger.warning(f"Failed to capture inference manager status: {exc}")
            record("inference_manager_status", "failed", exc)

    # Terminal transport Evidence must be captured after inference admission
    # has stopped and every unused output has been consumed/discarded, but
    # before trainer/source actors close their registered buffers.
    if (
        not force_kill_trainers
        and train_group is not None
        and inference_engines
    ):
        try:
            feature_contract = _collect_feature_contract_observation(
                args, train_group, inference_engines
            )
            record("feature_contract", "terminal_state_validated")
        except Exception as exc:
            feature_contract_error = exc
            logger.warning(
                f"Failed to validate terminal feature transport state: {exc}"
            )
            record("feature_contract", "terminal_state_failed", exc)

    trainer_close_failed = False
    if train_group is not None and not force_kill_trainers:
        try:
            train_group.close()
            record("training_actors", "closed")
        except Exception as exc:
            logger.warning(f"Failed to close trainer resources: {exc}")
            record("training_actors", "close_failed", exc)
            # A failed close can leave native reads running. Reuse the owner
            # fence before any source/client teardown, even with an empty ledger.
            trainer_close_failed = True
            force_kill_trainers = True

    if controller is not None and force_kill_trainers:
        try:
            finalized = ray.get(
                controller.finalize_uncommitted_work.remote(), timeout=10
            )
            record("uncommitted_work", "discarded_before_force")
            operations[-1]["count"] = int(finalized["discarded"])
        except Exception as exc:
            logger.warning(f"Failed to finalize uncommitted work before fencing: {exc}")
            record("uncommitted_work", "discard_failed", exc)
        try:
            lifecycle = ray.get(
                controller.get_work_lifecycle_status.remote(), timeout=10
            )
            record("work_lifecycle", "captured_before_force")
        except Exception as exc:
            logger.warning(f"Failed to capture lifecycle before owner fencing: {exc}")
            record("work_lifecycle", "capture_failed", exc)

    if force_kill_trainers and train_group is not None:
        for actor in train_group._actor_handlers:
            try:
                ray.kill(actor, no_restart=True)
                record("training_actor", "force_killed")
            except Exception as exc:
                logger.warning(f"Failed to terminate unresponsive trainer actor: {exc}")
                record("training_actor", "kill_failed", exc)

    if controller is not None and not force_kill_trainers:
        try:
            finalized = ray.get(
                controller.finalize_uncommitted_work.remote(), timeout=10
            )
            record("uncommitted_work", "discarded")
            operations[-1]["count"] = int(finalized["discarded"])
        except Exception as exc:
            logger.warning(f"Failed to finalize uncommitted work: {exc}")
            record("uncommitted_work", "discard_failed", exc)

    ownership_empty = (
        isinstance(lifecycle, dict)
        and lifecycle.get("resident_outputs") == 0
        and lifecycle.get("resident_bytes") == 0
    )
    if (
        force_kill_trainers
        and read_owner_fence is None
        and (trainer_close_failed or not ownership_empty or external_target)
    ):
        try:
            read_owner_fence = wait_for_owner_exit(
                getattr(train_group, "_owner_statuses", None),
                allow_node_loss=(
                    getattr(args, "cluster_mode", "single_node") == "static_multi_node"
                    and getattr(args, "mooncake_feature_affinity", "any") == "node_local_required"
                ),
            )
            record("read_owner_fence", "verified")
        except Exception as exc:
            source_teardown_safe = False
            record("read_owner_fence", "failed", exc)

    if inference_engines and force_kill_trainers and source_teardown_safe:
        for engine in inference_engines:
            try:
                source_state = None
                if external_target:
                    # The facade is Draft-owned; the actual source is owned by
                    # Target. After fencing readers, terminate its source process
                    # tree rather than waiting for an interrupted native drain.
                    source_state = ray.get(engine.shutdown.remote(force=True), timeout=float(
                        getattr(args, "inference_shutdown_timeout_seconds", 120)))
                ray.kill(engine, no_restart=True)
                if isinstance(source_state, Mapping) and source_state.get("state") == "node_local_domain_lost":
                    record("inference_engine", "node_local_domain_lost")
                    operations[-1].update(node_id=source_state["node_id"], physical_memory_unobserved=True)
                else:
                    record("inference_engine", "force_killed")
            except Exception as exc:
                logger.warning(f"Failed to terminate inference engine actor: {exc}")
                record("inference_engine", "kill_failed", exc)
    elif inference_engines and not force_kill_trainers:
        logger.info(f"Shutting down {len(inference_engines)} inference engine(s)...")
        shutdown_refs = []
        for engine in inference_engines:
            try:
                shutdown_refs.append(engine.shutdown.remote())
            except Exception as exc:
                logger.warning(f"Failed to initiate engine shutdown: {exc}")
                record("inference_engine", "shutdown_dispatch_failed", exc)
        for ref in shutdown_refs:
            try:
                ray.get(ref, timeout=30)
                record("inference_engine", "shutdown_acknowledged")
            except Exception as exc:
                logger.warning(f"Engine shutdown timed out or failed: {exc}")
                record("inference_engine", "shutdown_failed", exc)

    if controller is not None and read_owner_fence is not None:
        try:
            finalized = ray.get(
                controller.finalize_fenced_work.remote(owners_fenced=True), timeout=10
            )
            lifecycle = finalized["lifecycle"]
            record("work_lifecycle", "retired_after_owner_fence")
            operations[-1]["count"] = int(finalized["discarded"])
        except Exception as exc:
            record("work_lifecycle", "retire_failed", exc)

    if controller is not None and force_kill_trainers:
        try:
            ray.kill(controller, no_restart=True)
            record("training_controller", "force_killed")
        except Exception as exc:
            logger.warning(f"Failed to terminate training controller actor: {exc}")
            record("training_controller", "kill_failed", exc)

    mooncake_master_actor = getattr(args, "_mooncake_master_actor", None)
    if mooncake_master_actor is not None and source_teardown_safe:
        try:
            ray.get(mooncake_master_actor.shutdown.remote(), timeout=10)
            record("mooncake_master", "shutdown_acknowledged")
        except Exception as exc:
            logger.warning(f"Failed to shutdown mooncake master actor: {exc}")
            record("mooncake_master", "shutdown_failed", exc)

    if controller is not None and not force_kill_trainers:
        try:
            lifecycle = ray.get(controller.get_work_lifecycle_status.remote(), timeout=10)
            record(
                "work_lifecycle",
                "validated" if lifecycle.get("state") == "validated" else "incomplete",
            )
        except Exception as exc:
            logger.warning(f"Failed to capture work lifecycle status: {exc}")
            record("work_lifecycle", "failed", exc)

    failed = [
        item
        for item in operations
        if "failed" in item["state"] or item["state"] == "incomplete"
    ]
    return {
        "state": "completed_with_errors" if failed else "completed",
        "mode": "forced_owner_fence" if force_kill_trainers else "graceful",
        "operations": operations,
        "inference_manager": manager_status,
        "work_lifecycle": lifecycle,
        "read_owner_fence": read_owner_fence,
        "source_teardown": "allowed" if source_teardown_safe else "blocked",
        "feature_contract": feature_contract,
        "feature_contract_error": (
            f"{type(feature_contract_error).__name__}: {feature_contract_error}"
            if feature_contract_error is not None
            else None
        ),
    }


def _checkpoint_observation(checkpoint_dir: str | None) -> dict:
    if not checkpoint_dir:
        return {"enabled": False, "exists": False}

    base_dir = Path(checkpoint_dir).expanduser().resolve()
    tracker = base_dir / "latest_checkpointed_iteration.txt"
    observation = {
        "enabled": True,
        "checkpoint_dir": str(base_dir),
        "tracker_exists": tracker.is_file(),
        "exists": False,
    }
    if not tracker.is_file():
        return observation
    try:
        iteration = int(tracker.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return observation

    checkpoint = base_dir / f"iter_{iteration:07d}"
    complete_marker = checkpoint / "complete.json"
    work_state = checkpoint / "work_state.json"
    observation.update(
        {
            "iteration": iteration,
            "path": str(checkpoint),
            "exists": (
                checkpoint.is_dir()
                and complete_marker.is_file()
                and work_state.is_file()
            ),
            "metadata_exists": (checkpoint / "meta.json").is_file(),
            "completion_marker_exists": complete_marker.is_file(),
            "work_state_exists": work_state.is_file(),
        }
    )
    return observation


def _numeric_metrics(metrics: dict) -> dict[str, int | float]:
    return {key: value for key, value in metrics.items() if isinstance(value, (int, float))}


def _build_performance_observation(
    step_records: list[dict], *, warmup_steps: int, drain_steps: int
) -> dict:
    """Summarize one contiguous optimizer-step measurement window."""
    if warmup_steps < 0 or drain_steps < 0:
        raise ValueError("performance warmup/drain steps must be non-negative")
    stop = len(step_records) - drain_steps if drain_steps else len(step_records)
    window = step_records[warmup_steps:stop]
    if not window:
        return {
            "state": "unavailable",
            "reason": "no_optimizer_steps_remain_after_warmup_and_drain",
            "warmup_steps": warmup_steps,
            "drain_steps": drain_steps,
            "observed_steps": len(step_records),
        }

    wall_seconds = window[-1]["finished_seconds"] - window[0]["started_seconds"]
    committed_work = sum(record["committed_work"] for record in window)
    step_walls = [record["step_wall_seconds"] for record in window]
    if wall_seconds <= 0 or committed_work <= 0:
        return {
            "state": "unavailable",
            "reason": "non_positive_wall_or_committed_work",
            "warmup_steps": warmup_steps,
            "drain_steps": drain_steps,
            "observed_steps": len(step_records),
        }

    return {
        "state": "measured",
        "metric": "committed_work_per_second",
        "window": "continuous_optimizer_steps",
        "warmup_steps": warmup_steps,
        "drain_steps": drain_steps,
        "observed_steps": len(step_records),
        "optimizer_steps": len(window),
        "first_optimizer_step": window[0]["optimizer_step"],
        "last_optimizer_step": window[-1]["optimizer_step"],
        "committed_work": committed_work,
        "window_wall_seconds": wall_seconds,
        "committed_work_per_second": committed_work / wall_seconds,
        "step_wall_seconds": {
            "minimum": min(step_walls),
            "median": statistics.median(step_walls),
            "maximum": max(step_walls),
        },
        "steps": step_records,
    }


def _discard_profile_outputs(
    train_group, outputs: list, *, timeout_seconds: float, deadline: float | None = None,
) -> int:
    """Read and release producer outputs across initialized transfer clients."""

    from aspec.ray.train_group import discard_actor_batches

    if deadline is None:
        deadline = time.monotonic() + timeout_seconds
    get_sink_result = partial(
        _get_before_deadline, deadline=deadline, timeout_seconds=timeout_seconds,
        phase="producer_profile_sink",
    )
    batches = discard_actor_batches(train_group, outputs, resolve=get_sink_result)
    # Placement discovery consumed part of this same budget. Do not dispatch
    # native reads after it is exhausted, or reset it for the final join.
    futures = []
    for actor, chunk in batches:
        _remaining_optimizer_step_time(
            deadline=deadline, timeout_seconds=timeout_seconds, phase="producer_profile_sink",
        )
        futures.append(actor.discard_inference_outputs.remote(chunk))
    discarded = sum(int(count) for count in get_sink_result(futures))
    if discarded != len(outputs):
        raise RuntimeError(
            f"producer profile sinks discarded {discarded} of {len(outputs)} outputs"
        )
    return discarded


def _run_producer_service_profile(args, controller, inference_manager, train_group) -> dict | None:
    """Measure target production with real transfer sinks and no draft compute."""

    total_samples = int(getattr(args, "topology_producer_profile_samples", 0))
    if total_samples <= 0:
        return None
    warmup_samples = int(
        getattr(args, "topology_producer_profile_warmup_samples", 0)
    )
    timeout_seconds = float(
        getattr(args, "optimizer_step_timeout_seconds", 1800.0)
    )
    deadline = time.monotonic() + timeout_seconds
    get_profile_result = partial(
        _get_before_deadline, deadline=deadline, timeout_seconds=timeout_seconds,
        phase="producer_profile",
    )
    discarded = 0
    measured_started = time.monotonic() if warmup_samples == 0 else None

    logger.info(
        "Starting isolated producer service profile: samples=%d, warmup=%d, sinks=%d",
        total_samples,
        warmup_samples,
        len(train_group._actor_handlers),
    )
    while discarded < total_samples:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "timed out waiting for isolated producer service profile"
            )
        limit = min(32, total_samples - discarded)
        if discarded < warmup_samples:
            limit = min(limit, warmup_samples - discarded)
        outputs = get_profile_result(
            controller.pop_unused_training_results.remote(limit)
        )
        if outputs:
            released = _discard_profile_outputs(
                train_group, outputs, deadline=deadline, timeout_seconds=timeout_seconds,
            )
            get_profile_result(controller.confirm_discarded_inference_results.remote(outputs))
            discarded += released
            if discarded == warmup_samples:
                measured_started = time.monotonic()
            continue

        status = get_profile_result(inference_manager.get_status.remote())
        if not status.get("running", False) and not status.get(
            "sample_budget_drained", False
        ):
            raise RuntimeError(
                "inference manager stopped before the producer profile completed"
            )
        if status.get("source_drained") and not status.get(
            "sample_budget_drained", False
        ):
            raise RuntimeError(
                "training source exhausted before the producer profile sample budget"
            )
        time.sleep(0.01)

    status = get_profile_result(inference_manager.get_status.remote())
    if not status.get("sample_budget_drained", False):
        raise RuntimeError(
            "producer outputs were released before the inference budget fully drained"
        )
    if status.get("current_pool_size") != 0:
        raise RuntimeError(
            "producer profile left ready outputs in the controller pool: "
            f"{status.get('current_pool_size')}"
        )
    if status.get("scheduled_samples") != total_samples:
        raise RuntimeError(
            "producer profile scheduled an unexpected sample count: "
            f"{status.get('scheduled_samples')} != {total_samples}"
        )

    measured_finished = time.monotonic()
    if measured_started is None:
        raise RuntimeError("producer profile warmup boundary was not observed")
    measured_samples = total_samples - warmup_samples
    wall_seconds = measured_finished - measured_started
    if measured_samples <= 0 or wall_seconds <= 0:
        raise RuntimeError("producer profile produced a non-positive measurement window")

    source_restart = get_profile_result(
        controller.restart_training_source_after_profile.remote()
    )
    transition = get_profile_result(inference_manager.start_training_budget.remote())
    observation = {
        "state": "measured",
        "metric": "work_per_second",
        "mode": "target_generation_with_real_transfer_sink",
        "total_samples": total_samples,
        "warmup_samples": warmup_samples,
        "measured_work": measured_samples,
        "window_wall_seconds": wall_seconds,
        "work_per_second": measured_samples / wall_seconds,
        "sink_actor_count": len(train_group._actor_handlers),
        "data_plane": getattr(args, "mooncake_data_plane", "store"),
        "source_restart": source_restart,
        "next_budget_phase": transition["phase"],
    }
    logger.info(
        "Producer service profile completed: %.3f work/s (%d samples in %.3fs)",
        observation["work_per_second"],
        measured_samples,
        wall_seconds,
    )
    return observation


def _build_consumer_service_profile(
    metric_records: dict[int, float],
    step_records: list[dict],
    *,
    warmup_steps: int,
    drain_steps: int,
) -> dict:
    """Aggregate trainer-only compute service independently of source waits."""

    stop = len(step_records) - drain_steps if drain_steps else len(step_records)
    window = step_records[warmup_steps:stop]
    selected = [
        (record, metric_records.get(int(record["optimizer_step"])))
        for record in window
    ]
    selected = [(record, seconds) for record, seconds in selected if seconds]
    if not selected:
        return {
            "state": "unavailable",
            "reason": "no_trainer_compute_timings_in_performance_window",
            "observed_steps": len(metric_records),
        }
    compute_seconds = sum(float(seconds) for _, seconds in selected)
    committed_work = sum(int(record["committed_work"]) for record, _ in selected)
    if compute_seconds <= 0 or committed_work <= 0:
        return {
            "state": "unavailable",
            "reason": "non_positive_compute_time_or_committed_work",
            "observed_steps": len(metric_records),
        }
    return {
        "state": "measured",
        "metric": "committed_work_per_trainer_compute_second",
        "optimizer_steps": len(selected),
        "first_optimizer_step": selected[0][0]["optimizer_step"],
        "last_optimizer_step": selected[-1][0]["optimizer_step"],
        "committed_work": committed_work,
        "trainer_compute_seconds": compute_seconds,
        "work_per_second": committed_work / compute_seconds,
        "step_compute_seconds": {
            "minimum": min(float(seconds) for _, seconds in selected),
            "median": statistics.median(float(seconds) for _, seconds in selected),
            "maximum": max(float(seconds) for _, seconds in selected),
        },
    }


def _collect_feature_contract_observation(args, train_group, inference_engines) -> dict | None:
    """Join producer and consumer observations for the contract carried by the Plan."""
    contract = getattr(args, "aspec_feature_contract", None)
    if contract is None:
        return None
    if not inference_engines:
        raise RuntimeError("Target feature contract requires inference engine observations")

    contract_id = contract.get("contract_id")
    terminal_transport = (
        getattr(args, "mooncake_data_plane", "store") == "transfer_engine"
    )
    producer_statuses = ray.get(
        [
            (
                engine.prepare_feature_contract_status.remote()
                if terminal_transport
                else engine.get_feature_contract_status.remote()
            )
            for engine in inference_engines
        ],
        timeout=float(getattr(args, "inference_shutdown_timeout_seconds", 120.0)),
    )
    consumer_statuses = train_group.get_feature_contract_status()
    expected_consumers = getattr(args, "world_size", len(consumer_statuses))
    if len(consumer_statuses) != expected_consumers:
        raise RuntimeError(
            "Target feature contract consumer count does not match the plan: "
            f"{len(consumer_statuses)} != {expected_consumers}"
        )

    participants = producer_statuses + consumer_statuses
    views = getattr(args, "aspec_shared_consumer_contracts", {})
    def expected_contract(item):
        if item.get("role") == "producer" or not views:
            return contract_id
        experiment = item.get("participant", "").split("/", 1)[0]
        return views.get(experiment)
    mismatched = [
        item.get("participant", "unknown")
        for item in participants
        if item.get("contract_id") != expected_contract(item)
    ]
    if mismatched:
        raise RuntimeError(
            "Target feature contract identity mismatch for participants: "
            f"{sorted(mismatched)!r}"
        )
    unbound_producers = [
        item.get("participant", "unknown")
        for item in producer_statuses
        if item.get("binding_validations", 0) <= 0
    ]
    if unbound_producers:
        raise RuntimeError(
            "Target feature producers did not validate backend bindings: "
            f"{sorted(unbound_producers)!r}"
        )
    producer_samples = sum(item.get("metadata_validations", 0) for item in producer_statuses)
    if producer_samples <= 0:
        raise RuntimeError("No SGLang output was validated against the target feature contract")
    unvalidated_consumers = [
        item.get("participant", "unknown")
        for item in consumer_statuses
        if item.get("metadata_validations", 0) <= 0 or item.get("tensor_validations", 0) <= 0
    ]
    if unvalidated_consumers:
        raise RuntimeError(
            "Target feature consumers did not validate transported tensors: "
            f"{sorted(unvalidated_consumers)!r}"
        )

    neutralization_by_participant = [
        {
            "participant": item.get("participant", "unknown"),
            "neutralized_rank_samples": int(
                (item.get("data_admission") or {}).get("neutralized_rank_samples", 0)
            ),
        }
        for item in consumer_statuses
    ]

    transport_observation = None
    if terminal_transport:
        active_producers = [
            item for item in producer_statuses if item.get("metadata_validations", 0) > 0
        ]
        transport_participants = producer_statuses + consumer_statuses
        missing_transport = [
            item.get("participant", "unknown")
            for item in transport_participants
            if not isinstance(item.get("transport"), dict)
        ]
        if missing_transport:
            raise RuntimeError(
                "TransferEngine participant did not report terminal data-plane Evidence: "
                f"{sorted(missing_transport)!r}"
            )
        transport_statuses = [item["transport"] for item in transport_participants]
        expected_protocol = getattr(args, "mooncake_protocol", None)
        invalid = [
            item
            for item in transport_statuses
            if item.get("state") != "active"
            or item.get("protocol") != expected_protocol
            or item.get("strict_transport") is not True
            or item.get("run_id") != getattr(args, "aspec_run_id", None)
            or item.get("attempt_id") != getattr(args, "aspec_attempt_id", None)
        ]
        if invalid:
            raise RuntimeError(
                "TransferEngine runtime identity or strict transport does not match the Plan"
            )
        if expected_protocol == "rdma":
            planned_nodes = sorted(
                set(getattr(args, "training_node_ips", None) or ())
                | set(getattr(args, "inference_node_ips", None) or ())
            )
            if planned_nodes:
                expected_routes = resolve_rdma_routes(
                    planned_nodes,
                    device_name=getattr(args, "mooncake_device_name", None),
                    port=getattr(args, "mooncake_rdma_port", None),
                    gid_index=getattr(args, "mooncake_rdma_gid_index", None),
                    overrides=getattr(args, "mooncake_rdma_routes", None),
                )
            else:
                expected_routes = {}
            default_route = {
                "device_name": getattr(args, "mooncake_device_name", None),
                "port": getattr(args, "mooncake_rdma_port", None),
                "gid_index": getattr(args, "mooncake_rdma_gid_index", None),
            }
            invalid_routes = []
            for item in transport_statuses:
                node_address = item.get("node_address")
                if expected_routes and node_address not in expected_routes:
                    invalid_routes.append(item.get("endpoint", "unknown"))
                    continue
                expected_route = expected_routes.get(node_address, default_route)
                if item.get("configured_rdma_route") != expected_route:
                    invalid_routes.append(item.get("endpoint", "unknown"))
            if invalid_routes:
                raise RuntimeError(
                    "TransferEngine configured RDMA route does not match the Plan: "
                    f"{invalid_routes!r}"
                )
        producer_transports = [status["transport"] for status in producer_statuses]
        consumer_transports = [
            status["transport"] for status in consumer_statuses
        ]
        terminal_errors = []
        for participant, transport in zip(
            producer_statuses, producer_transports, strict=True
        ):
            metrics = transport.get("metrics") or {}
            descriptors = int(metrics.get("source_descriptors", -1))
            releases = int(metrics.get("source_releases", -1))
            discarded = int(metrics.get("discarded_source_descriptors", -1))
            if descriptors != releases + discarded:
                terminal_errors.append(
                    f"{participant.get('participant', 'unknown')}: "
                    f"source descriptors={descriptors}, releases={releases}, "
                    f"discarded={discarded}"
                )
            if (transport.get("ring") or {}).get("in_use") != 0:
                terminal_errors.append(
                    f"{participant.get('participant', 'unknown')}: source ring not empty"
                )
            if transport.get("notification_worker_error"):
                terminal_errors.append(
                    f"{participant.get('participant', 'unknown')}: "
                    f"{transport['notification_worker_error']}"
                )
        for participant, transport in zip(
            consumer_statuses, consumer_transports, strict=True
        ):
            metrics = transport.get("metrics") or {}
            reads = int(metrics.get("destination_reads", -1))
            releases = int(metrics.get("destination_releases", -1))
            if reads != releases:
                terminal_errors.append(
                    f"{participant.get('participant', 'unknown')}: "
                    f"destination reads={reads}, releases={releases}"
                )
            if (transport.get("ring") or {}).get("in_use") != 0:
                terminal_errors.append(
                    f"{participant.get('participant', 'unknown')}: destination ring not empty"
                )

        source_descriptors = sum(
            int(item["metrics"]["source_descriptors"])
            for item in producer_transports
        )
        source_releases = sum(
            int(item["metrics"]["source_releases"])
            for item in producer_transports
        )
        discarded_source_descriptors = sum(
            int(item["metrics"]["discarded_source_descriptors"])
            for item in producer_transports
        )
        source_payload_bytes = sum(
            int(item["metrics"]["source_payload_bytes"])
            for item in producer_transports
        )
        discarded_source_payload_bytes = sum(
            int(item["metrics"]["discarded_source_payload_bytes"])
            for item in producer_transports
        )
        destination_reads = sum(
            int(item["metrics"]["destination_reads"])
            for item in consumer_transports
        )
        destination_releases = sum(
            int(item["metrics"]["destination_releases"])
            for item in consumer_transports
        )
        transferred_bytes = sum(
            int(item["metrics"]["destination_payload_bytes"])
            for item in consumer_transports
        )
        shared_source = getattr(args, "aspec_shared_source", False) is True
        shared_reader = getattr(args, "aspec_shared_reader", None)
        expected_reads = (
            sum(int(item["metrics"].get("shared_expected_reads", 0))
                - int(item["metrics"].get("shared_withdrawn_reads", 0)) for item in producer_transports)
            if shared_source else source_releases
        )
        expected_payload_bytes = (
            sum(int(item["metrics"].get("shared_expected_payload_bytes", 0))
                - int(item["metrics"].get("shared_withdrawn_payload_bytes", 0)) for item in producer_transports)
            if shared_source else source_payload_bytes - discarded_source_payload_bytes
        )
        if shared_source and shared_reader is not None:
            completed = [
                result for item in producer_transports
                for reader, result in item["metrics"].get("shared_completed_readers", {}).items()
                if reader == shared_reader or reader.startswith(shared_reader + "/cp/")
            ]
            expected_reads = sum(item.get("reads", 0) for item in completed)
            expected_payload_bytes = sum(item.get("payload_bytes", 0) for item in completed)
        if expected_reads != destination_reads:
            terminal_errors.append(
                "source release and destination read counts differ: "
                f"{expected_reads} != {destination_reads}"
            )
        if expected_payload_bytes != transferred_bytes:
            terminal_errors.append(
                "source and terminal destination/discarded byte counts differ: "
                f"{expected_payload_bytes} != {transferred_bytes}"
            )
        transfer_failures = sum(
            int(item["metrics"]["transfer_failures"])
            for item in transport_statuses
        )
        if transfer_failures:
            terminal_errors.append(
                f"participants reported {transfer_failures} transfer failure(s)"
            )
        if terminal_errors:
            raise RuntimeError(
                "TransferEngine ownership did not reach a clean terminal state: "
                + "; ".join(terminal_errors)
            )
        transfer_seconds = sum(
            float(item["metrics"]["destination_transfer_seconds"])
            for item in consumer_transports
        )
        max_transfer_seconds = max(
            (
                float(item["metrics"].get("destination_max_transfer_seconds", 0.0))
                for item in consumer_transports
            ),
            default=0.0,
        )
        observed_ring_waits = [
            float(value)
            for item in transport_statuses
            if (value := item["metrics"].get("ring_wait_seconds")) is not None
        ]
        transport_observation = {
            "state": "validated",
            "protocol": expected_protocol,
            "strict_transport": True,
            "activation": sorted(
                {str(item.get("activation")) for item in transport_statuses}
            ),
            "planned_producer_endpoints": len(producer_statuses),
            "producer_endpoints": len(producer_transports),
            "producer_endpoints_with_data": len(active_producers),
            "inactive_producer_participants": sorted(
                item.get("participant", "unknown")
                for item in producer_statuses
                if item.get("metadata_validations", 0) <= 0
            ),
            "consumer_endpoints": len(consumer_transports),
            "source_descriptors": source_descriptors,
            "shared_source": shared_source,
            "shared_reader": shared_reader,
            "expected_destination_reads": expected_reads,
            "source_releases": source_releases,
            "discarded_source_descriptors": discarded_source_descriptors,
            "source_payload_bytes": source_payload_bytes,
            "discarded_source_payload_bytes": discarded_source_payload_bytes,
            "destination_reads": destination_reads,
            "destination_releases": destination_releases,
            "destination_payload_bytes": transferred_bytes,
            "destination_transfer_seconds": transfer_seconds,
            "destination_max_transfer_seconds": max_transfer_seconds,
            "destination_payload_bytes_per_second": (
                transferred_bytes / transfer_seconds if transfer_seconds > 0 else None
            ),
            "transfer_failures": transfer_failures,
            "ring_wait_seconds": sum(observed_ring_waits),
            "ring_wait_participants_observed": len(observed_ring_waits),
            "ring_wait_participants_total": len(transport_statuses),
            "participants": transport_statuses,
        }

    return {
        "state": "validated",
        "contract_id": contract_id,
        "producer_count": len(producer_statuses),
        "consumer_count": len(consumer_statuses),
        "producer_metadata_validations": producer_samples,
        "consumer_metadata_validations": sum(
            item["metadata_validations"] for item in consumer_statuses
        ),
        "consumer_tensor_validations": sum(
            item["tensor_validations"] for item in consumer_statuses
        ),
        "data_admission": {
            "neutralized_rank_samples": sum(
                item["neutralized_rank_samples"]
                for item in neutralization_by_participant
            ),
            "participants": neutralization_by_participant,
        },
        "transport": transport_observation,
        "participants": participants,
    }


def training_loop(
    args,
    controller,
    inference_manager,
    train_group,
    inference_engines=None,
    dataset_size=None,
    eval_dataset_size=None,
    inference_future=None,
    eval_state=None,
    tracking=None,
):
    """Run the training loop with sync training and async inference.

    Training is synchronous - waits for each step to complete.
    Inference runs in background, continuously producing data.

    Each optimizer step (with draft_accumulation_steps dispatches):
      1. Controller dispatches per_dp_rank_batch_size * dp_size samples, accumulation_steps times
      2. Each DP rank receives per_dp_rank_batch_size * accumulation_steps samples total
      3. train_from_queue(num_batches=accumulation_steps) processes all micro-batches
      4. Optimizer steps after the last micro-batch

    completed_steps counts optimizer steps (consistent with lr_total_steps).

    Args:
        args: Configuration arguments.
        controller: AsyncTrainingController ray actor handle (dataset already loaded).
        inference_manager: AsyncInferenceManager ray actor handle.
        train_group: Training group with set_train_queues method.
        dataset_size: Number of training samples. If None, queried from controller.
        eval_dataset_size: Number of eval samples. If None, queried from controller. 0 means no eval.
    """
    rpc_timeout = float(getattr(args, "optimizer_step_timeout_seconds", 1800.0))
    if dataset_size is None:
        dataset_size = ray.get(controller.get_dataset_size.remote(), timeout=rpc_timeout)
    if dataset_size <= 0:
        raise ValueError(
            f"Training dataset size is {dataset_size}. "
            f"Ensure controller.load_dataset() was called before run_training_loop()."
        )
    if eval_dataset_size is None:
        eval_dataset_size = ray.get(controller.get_eval_dataset_size.remote(), timeout=rpc_timeout)

    # ── Eval setup ──────────────────────────────────────────────
    eval_state = eval_state or setup_eval(controller, train_group, args, eval_dataset_size)

    # Inference engine is alive. Run eval hs generation first.
    if eval_state.eval_enabled and not eval_state.eval_cache_loaded:
        generate_eval_cache(
            controller, inference_manager, train_group, eval_state,
            check_producer=lambda: _raise_if_inference_finished(inference_future),
        )

    eval_interval = eval_state.eval_interval
    eval_enabled = eval_state.eval_enabled
    best_eval_score = eval_state.best_eval_score

    def evaluate(step):
        return run_eval(step, train_group, eval_enabled, state=eval_state, args=args, tracking=tracking)

    dp_size = (
        getattr(args, "dp_size", None) or args.training_num_nodes * args.training_num_gpus_per_node
    )
    num_steps = args.num_train_steps
    accumulation_steps = getattr(args, "draft_accumulation_steps", 1)
    # steps_per_epoch in optimizer steps, pre-computed in auto_calculate_training_steps
    steps_per_epoch = getattr(args, "steps_per_epoch", dataset_size // args.global_batch_size)
    if steps_per_epoch == 0:
        steps_per_epoch = 1
    num_epochs = getattr(args, "num_epochs", 1)
    source_work_budget = _validate_source_work_budget(
        args,
        dataset_size=dataset_size,
        dp_size=dp_size,
    )
    requested_source_work_budget = getattr(
        args, "requested_source_work_budget", source_work_budget
    )
    planned_dropped_source_work = int(
        getattr(args, "planned_dropped_source_work", 0) or 0
    )
    if planned_dropped_source_work:
        logger.info(
            "Finite source pass uses deterministic drop-last: requested=%d, "
            "scheduled=%d, planned_dropped=%d, alignment=%d",
            requested_source_work_budget,
            source_work_budget,
            planned_dropped_source_work,
            int(getattr(args, "source_work_alignment")),
        )

    # Submit training data AFTER eval hs generation so that training prompts don't
    # leak into the inference pipeline during eval. A true resume restores the
    # checkpoint's exact committed semantic-work set; only uncommitted work is
    # regenerated after process-local buffers disappear.
    start_step = ray.get(train_group._actor_handlers[0].get_global_step.remote(), timeout=rpc_timeout)
    if source_work_budget is not None and start_step:
        raise ValueError(
            "source_work_budget currently requires a fresh optimizer state; "
            f"got start_step={start_step}"
        )
    resume_state = load_checkpoint_work_state(args)
    resume_epoch = start_step // steps_per_epoch if steps_per_epoch > 0 else 0
    submit_kwargs = {"epoch": resume_epoch}
    if resume_state is not None:
        _validate_resume_optimizer_frontier(
            start_step=start_step,
            resume_state=resume_state,
            continual_training=bool(getattr(args, "continual_training", False)),
        )
        resume_epoch = int(resume_state["dataset_epoch"])
        submit_kwargs = {
            "epoch": resume_epoch,
            "committed_work_ids": resume_state["committed_work_ids"],
            "expected_dataset_fingerprint": resume_state["dataset_fingerprint"],
        }
    if getattr(args, "aspec_shared_source", False) is True:
        submit_kwargs["completed_steps"] = start_step
    submitted_samples = ray.get(
        controller.submit_training_dataset.remote(**submit_kwargs), timeout=rpc_timeout
    )
    if (
        resume_state is not None
        and int(resume_state["committed_work_count"]) > 0
        and submitted_samples == 0
        and source_work_budget is None
        and getattr(args, "aspec_shared_source", False) is not True
    ):
        # A weights-only phase resets optimizer steps, not the semantic source
        # frontier. Do not wait for optimizer progress to advance an empty epoch.
        logger.info("Restored source epoch %d is fully committed; advancing", resume_epoch)
        ray.get(controller.reload_dataset.remote(), timeout=rpc_timeout)
        resume_epoch += 1
    producer_service_profile = _run_producer_service_profile(
        args,
        controller,
        inference_manager,
        train_group,
    )

    logger.info(
        f"Starting: num_steps={num_steps}, num_epochs={num_epochs}, "
        f"steps_per_epoch={steps_per_epoch}, global_batch_size={args.global_batch_size}, "
        f"accumulation_steps={accumulation_steps}, "
        f"dp_size={dp_size}, per_dp_rank_batch_size={args.per_dp_rank_batch_size}"
    )

    enable_perf = getattr(args, "enable_perf_metrics", True)
    prefetch_batches = max(accumulation_steps, getattr(args, "prefetch_depth", 0))
    streaming_dispatch = bool(
        getattr(args, "streaming_dispatch_with_training", False)
    )
    pipeline_dispatch = bool(
        getattr(args, "pipeline_dispatch_with_training", False)
    )
    optimizer_step_timeout = float(
        getattr(args, "optimizer_step_timeout_seconds", 1800.0)
    )

    completed_steps = start_step
    current_epoch = resume_epoch + 1
    steps_in_current_epoch = completed_steps % steps_per_epoch
    if start_step > 0:
        logger.info(f"Resuming from step {start_step} (epoch {current_epoch})")
    dispatch_attempts = 0
    consecutive_failures = 0
    queued_batches = 0
    last_saved_step: int | None = None
    previous_dispatch_wait: float | None = None
    last_observed_metrics: dict[str, int | float] = {}
    training_clock_start = time.monotonic()
    step_records: list[dict] = []
    trainer_compute_times: dict[int, float] = {}
    streaming_steps = 0
    finite_source_tail_steps = 0
    optimizer_microsteps = 0
    pipelined_microsteps = 0
    previous_committed_work = 0
    peak_committed_step_work = 0
    experiment_name = getattr(args, "aspec_experiment_name", "")
    check_progress = None
    if getattr(args, "aspec_shared_source", False) is True:
        def check_progress():
            ray.get(controller.check_running.remote(), timeout=5)
    progress = tqdm(total=num_steps, desc=f"Training {experiment_name}".rstrip(), unit="step", initial=start_step)
    while completed_steps < num_steps:
        if check_progress is not None:
            check_progress()
        optimizer_step_started = time.monotonic()
        optimizer_step_deadline = optimizer_step_started + optimizer_step_timeout
        get_step_result = partial(
            _get_before_deadline, deadline=optimizer_step_deadline,
            timeout_seconds=optimizer_step_timeout, phase="optimizer_step",
        )
        remaining_steps = min(
            num_steps - completed_steps,
            steps_per_epoch - steps_in_current_epoch,
        )
        target_queued_batches = min(
            prefetch_batches,
            remaining_steps * accumulation_steps,
        )
        remaining_source_work = (
            source_work_budget - previous_committed_work
            if source_work_budget is not None
            else None
        )
        tail_mode = bool(
            remaining_source_work is not None
            and remaining_source_work
            <= max(
                int(getattr(args, "max_sample_pool_size", 0) or 0),
                dp_size * accumulation_steps,
                math.ceil(peak_committed_step_work * 1.25),
            )
        )
        finite_drain_check = bool(
            remaining_source_work is not None
            and remaining_source_work
            <= max(
                int(getattr(args, "max_sample_pool_size", 0) or 0)
                * max(1, accumulation_steps),
                dp_size * accumulation_steps,
                math.ceil(peak_committed_step_work * 1.25),
            )
        )
        packing_finite_source = bool(
            source_work_budget is not None
            and getattr(args, "dflash_packing", False)
        )
        minimum_packed_optimizer_work = dp_size * accumulation_steps
        if (
            packing_finite_source
            and remaining_source_work is not None
            and remaining_source_work < minimum_packed_optimizer_work
        ):
            if queued_batches:
                raise RuntimeError(
                    "Finite packing tail reached with prefetched work from an "
                    "incomplete optimizer step"
                )
            logger.info(
                "Finite packing source drops an incomplete optimizer tail: "
                "committed=%d, scheduled=%d, dropped_tail=%d, minimum_step_work=%d",
                previous_committed_work,
                source_work_budget,
                remaining_source_work,
                minimum_packed_optimizer_work,
            )
            break
        if tail_mode:
            # A finite tail owns only its current optimizer step.  This keeps
            # next-step prefetch from creating work that drop-last cannot commit.
            target_queued_batches = min(target_queued_batches, accumulation_steps)
        step_accumulation_steps = accumulation_steps
        if enable_perf:
            t_dispatch = time.time()
        train_futures = None
        stream_this_step = streaming_dispatch and (
            not tail_mode
            or (
                remaining_source_work is not None
                and remaining_source_work >= (
                    minimum_packed_optimizer_work
                    if packing_finite_source else args.global_batch_size
                )
            )
        )
        if stream_this_step:
            train_futures = [
                actor.train_from_queue.remote(
                    step=completed_steps,
                    num_batches=step_accumulation_steps,
                )
                for actor in train_group._actor_handlers
            ]
        status = None
        source_budget_scheduled = False
        source_drained = False
        queued_source_work = 0
        while queued_batches < target_queued_batches:
            _remaining_optimizer_step_time(
                deadline=optimizer_step_deadline,
                timeout_seconds=optimizer_step_timeout,
                phase="source_supply_and_dispatch",
            )
            dispatch_attempts += 1

            dispatch_step = completed_steps + queued_batches // accumulation_steps
            dispatch_microstep = queued_batches % accumulation_steps
            if finite_drain_check and source_work_budget is not None:
                inference_status = get_step_result(inference_manager.get_status.remote())
                source_budget_scheduled = bool(
                    inference_status.get("sample_budget_drained", False)
                    or inference_status.get("remaining_sample_budget") == 0
                )
                source_drained = bool(
                    inference_status.get("sample_budget_drained", False)
                )
            max_packing_documents = None
            return_document_count = False
            if packing_finite_source and tail_mode:
                remaining_microsteps = target_queued_batches - queued_batches
                unallocated_work = remaining_source_work - queued_source_work
                max_packing_documents = max(
                    dp_size,
                    math.ceil(unallocated_work / remaining_microsteps),
                )
                return_document_count = True
            dispatch_result = get_step_result(
                controller.try_dispatch_batch.remote(
                    optimizer_step=dispatch_step,
                    microstep=dispatch_microstep,
                    # Once every source item has been scheduled, waiting for
                    # all in-flight Target outputs before flushing packing can
                    # deadlock against ready-pool backpressure.  Permit
                    # progress here, but reserve final_source_drained for the
                    # stronger state where those outputs have also returned.
                    force_packing_progress=(tail_mode or source_budget_scheduled),
                    final_source_drained=source_drained,
                    max_packing_documents=max_packing_documents,
                    return_document_count=return_document_count,
                )
            )
            if dispatch_result:
                queued_batches += 1
                if type(dispatch_result) is int:
                    queued_source_work += dispatch_result
                consecutive_failures = 0
            else:
                _raise_if_inference_finished(inference_future)
                consecutive_failures += 1
                if finite_drain_check and source_work_budget is not None:
                    if not source_budget_scheduled or not source_drained:
                        inference_status = get_step_result(inference_manager.get_status.remote())
                        source_budget_scheduled = bool(
                            inference_status.get("sample_budget_drained", False)
                            or inference_status.get("remaining_sample_budget") == 0
                        )
                        source_drained = bool(
                            inference_status.get("sample_budget_drained", False)
                        )
                    if source_budget_scheduled:
                        forced = get_step_result(
                            controller.try_dispatch_batch.remote(
                                optimizer_step=dispatch_step,
                                microstep=dispatch_microstep,
                                force_packing_progress=True,
                                final_source_drained=source_drained,
                                max_packing_documents=max_packing_documents,
                                return_document_count=return_document_count,
                            )
                        )
                        if forced:
                            queued_batches += 1
                            if type(forced) is int:
                                queued_source_work += forced
                            consecutive_failures = 0
                            continue
                    if source_drained:
                        if queued_batches > 0:
                            if stream_this_step:
                                raise RuntimeError(
                                    "Finite source drained after streaming trainers "
                                    "started but before the configured accumulation "
                                    "was filled"
                                )
                            step_accumulation_steps = queued_batches
                            target_queued_batches = queued_batches
                            consecutive_failures = 0
                            continue
                        raise RuntimeError(
                            "Finite source drained before a complete DP dispatch; "
                            f"committed={previous_committed_work}, "
                            f"budget={source_work_budget}, status={status}"
                        )
                if queued_batches >= accumulation_steps:
                    # The current step can run while inference refills the pool.
                    target_queued_batches = queued_batches
                    consecutive_failures = 0
                    continue

                # Only fetch status when needed for logging or reload decision
                if dispatch_attempts % 100 == 0 or consecutive_failures >= 500:
                    status = get_step_result(controller.get_full_status.remote())

                if dispatch_attempts % 100 == 0 and status is not None:
                    log_tail_wait = logger.info if tail_mode else logger.debug
                    inference_tail = (
                        {
                            key: inference_status.get(key)
                            for key in (
                                "scheduled_samples",
                                "remaining_sample_budget",
                                "pending_tasks",
                                "inflight_samples",
                                "prompt_buffer_size",
                                "current_pool_size",
                                "resident_outputs",
                                "sample_budget_drained",
                                "source_drained",
                            )
                        }
                        if tail_mode and finite_drain_check
                        else None
                    )
                    log_tail_wait(
                        f"Epoch {current_epoch}, Step {completed_steps}: "
                        f"dispatch failed {dispatch_attempts} times "
                        f"(consecutive={consecutive_failures}), "
                        f"pool_size={status['sample_pool_size']}, "
                        f"need={status['dispatch_batch_size']}"
                        + (
                            f", queued={queued_batches}/{target_queued_batches}, "
                            f"remaining_work={remaining_source_work}, "
                            f"inference={inference_tail}"
                            if inference_tail is not None
                            else ""
                        )
                    )

                if stream_this_step and consecutive_failures % 100 == 0:
                    ready, _ = ray.wait(train_futures, num_returns=1, timeout=0)
                    if ready:
                        # Trainers cannot complete normally before every
                        # microstep is queued; surface the original failure.
                        get_step_result(ready)

                should_reload = False
                if steps_in_current_epoch >= steps_per_epoch:
                    should_reload = True
                elif (
                    # The shared producer owns source reloads. An experiment's
                    # deliberately empty prompt queue is not source exhaustion.
                    getattr(args, "aspec_shared_source", False) is not True
                    and consecutive_failures >= 500
                    and (completed_steps > 0 or queued_batches > 0)
                    and status is not None
                    and status["sample_pool_size"] < status["dispatch_batch_size"]
                    and status.get("prompt_buffer_size", 0) == 0
                ):
                    logger.warning(
                        f"Pool insufficient for dispatch "
                        f"(pool_size={status['sample_pool_size']}, "
                        f"need={status['dispatch_batch_size']}, "
                        f"{queued_batches}/{accumulation_steps} batches queued, "
                        f"{steps_in_current_epoch}/{steps_per_epoch} steps in epoch). "
                        f"Reloading dataset."
                    )
                    should_reload = True

                if should_reload:
                    if completed_steps < num_steps:
                        current_epoch += 1
                        steps_in_current_epoch = 0
                        consecutive_failures = 0
                        logger.info(f"Dataset exhausted, reloading (epoch {current_epoch})...")
                        get_step_result(controller.reload_dataset.remote())
                    else:
                        logger.info("Max steps reached, stopping")
                        break

                time.sleep(0.01)
        else:
            # The current optimizer step is fully queued.
            if enable_perf:
                dispatch_wait = time.time() - t_dispatch
            else:
                dispatch_wait = 0.0

            if train_futures is None:
                train_futures = [
                    actor.train_from_queue.remote(
                        step=completed_steps,
                        num_batches=step_accumulation_steps,
                    )
                    for actor in train_group._actor_handlers
                ]

            added_ahead = 0
            ahead_wait = 0.0
            existing_ahead = max(0, queued_batches - accumulation_steps)
            can_pipeline = (
                pipeline_dispatch
                and not tail_mode
                and not finite_drain_check
                and remaining_steps > 1
                and existing_ahead < accumulation_steps
            )
            if can_pipeline:
                target_ahead = max(existing_ahead, max(1, accumulation_steps // 2))
                stop_dispatch = threading.Event()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    next_dispatch = executor.submit(
                        _dispatch_ahead,
                        controller,
                        optimizer_step=completed_steps + 1,
                        first_microstep=existing_ahead,
                        target_microsteps=target_ahead,
                        stop_event=stop_dispatch,
                        deadline=optimizer_step_deadline,
                        timeout_seconds=optimizer_step_timeout,
                    )
                    try:
                        train_results = _wait_for_trainer_group(
                            train_futures,
                            check_progress=check_progress,
                            timeout_seconds=_remaining_optimizer_step_time(
                                deadline=optimizer_step_deadline,
                                timeout_seconds=optimizer_step_timeout,
                                phase="trainer_completion",
                            ),
                        )
                    finally:
                        stop_dispatch.set()
                    added_ahead, ahead_wait = next_dispatch.result()
            else:
                train_results = _wait_for_trainer_group(
                    train_futures,
                    check_progress=check_progress,
                    timeout_seconds=_remaining_optimizer_step_time(
                        deadline=optimizer_step_deadline,
                        timeout_seconds=optimizer_step_timeout,
                        phase="trainer_completion",
                    ),
                )
            queued_batches += added_ahead
            queued_batches -= step_accumulation_steps
            completed_steps += 1
            if stream_this_step:
                streaming_steps += 1
            if tail_mode and source_work_budget is not None:
                finite_source_tail_steps += 1
            optimizer_microsteps += step_accumulation_steps
            pipelined_microsteps += added_ahead
            committed_by_rank = train_group.pop_committed_work(
                completed_steps,
                timeout_seconds=_remaining_optimizer_step_time(
                    deadline=optimizer_step_deadline,
                    timeout_seconds=optimizer_step_timeout, phase="commit_work",
                ),
            )
            committed_execution_ids = [
                execution_id
                for rank_items in committed_by_rank
                for execution_id in rank_items
            ]
            commit_status = get_step_result(
                controller.commit_optimizer_step.remote(
                    optimizer_step=completed_steps,
                    execution_ids=committed_execution_ids,
                    expected_unique=(
                        len(set(committed_execution_ids))
                        if getattr(args, "dflash_packing", False)
                        else args.global_batch_size
                    ),
                )
            )
            committed_total = int(commit_status["committed_work"])
            if source_work_budget is not None and committed_total > source_work_budget:
                raise RuntimeError(
                    "Committed source work exceeded the configured finite budget: "
                    f"committed={committed_total}, budget={source_work_budget}"
                )
            committed_work = committed_total - previous_committed_work
            if committed_work <= 0:
                raise RuntimeError(
                    "Optimizer step did not advance committed source work: "
                    f"step={completed_steps}, before={previous_committed_work}, "
                    f"after={committed_total}"
                )
            previous_committed_work = committed_total
            peak_committed_step_work = max(
                peak_committed_step_work, committed_work
            )
            optimizer_step_finished = time.monotonic()
            step_records.append(
                {
                    "optimizer_step": completed_steps,
                    "microsteps": step_accumulation_steps,
                    "committed_work": committed_work,
                    "started_seconds": optimizer_step_started - training_clock_start,
                    "finished_seconds": optimizer_step_finished - training_clock_start,
                    "step_wall_seconds": optimizer_step_finished - optimizer_step_started,
                    "dispatch_wait_seconds": dispatch_wait,
                    "pipelined_dispatch_wait_seconds": ahead_wait,
                }
            )

            # Log metrics from training (use rank 0's metrics - they're already all-reduced)
            metrics = train_results[0] if train_results and train_results[0] else {}
            deferred_metric_step = metrics.pop("_metrics_step", None)
            metric_step = int(
                deferred_metric_step
                if deferred_metric_step is not None
                else completed_steps
            )
            metric_optimizer_step = (
                int(deferred_metric_step) + 1
                if deferred_metric_step is not None
                else completed_steps
            )
            metric_step_time = metrics.get("perf/step_time")
            if (
                isinstance(metric_step_time, (int, float))
                and metric_step_time > 0
            ):
                trainer_compute_times[metric_optimizer_step] = float(metric_step_time)
            metric_dispatch_wait = (
                previous_dispatch_wait
                if metric_step != completed_steps and previous_dispatch_wait is not None
                else dispatch_wait
            )
            previous_dispatch_wait = dispatch_wait
            if metrics:
                # Add step counters for wandb x-axis (required in shared mode)
                metrics["train/step"] = metric_step
                metrics["inference/step"] = completed_steps

                # Add inference metrics (e2e_latency, spec metrics, etc.)
                if getattr(args, "aspec_shared_source", False) is not True:
                    inference_metrics = ray.get(inference_manager.flush_metrics.remote(), timeout=rpc_timeout)
                    metrics.update(inference_metrics)

                if enable_perf:
                    metrics["perf/dispatch_wait"] = metric_dispatch_wait
                    step_time = metrics.get("perf/step_time", 0)
                    if step_time > 0:
                        # This is Draft compute throughput for the current
                        # end-to-end step, not an isolated service capacity.
                        metrics["perf/draft_compute_rate"] = (
                            args.global_batch_size / step_time
                        )

                    if metric_step % 5 == 0 or metric_step <= 5:
                        data_t = metrics.get("perf/data_time", 0)
                        compute_t = metrics.get("perf/compute_time", 0)
                        fwd_t = metrics.get("perf/forward_time", 0)
                        bwd_t = metrics.get("perf/backward_time", 0)
                        opt_t = metrics.get("perf/optimizer_time", 0)
                        logger.info(
                            f"TIMING step={metric_step}: "
                            f"step={step_time:.3f}s "
                            f"data={data_t:.3f}s "
                            f"compute={compute_t:.3f}s "
                            f"[fwd={fwd_t:.3f}s bwd={bwd_t:.3f}s opt={opt_t:.3f}s] "
                            f"dispatch={metric_dispatch_wait:.3f}s"
                        )

                _write_training_metrics(metrics, metric_step, completed_steps, tracking=tracking)
                last_observed_metrics = _numeric_metrics(metrics)

            # ── Eval at explicit interval (if configured) ─────────
            # Skip if a checkpoint save is about to run (it will eval anyway)
            save_due = _is_save_interval_step(completed_steps, args.save_interval)
            if eval_interval > 0 and completed_steps % eval_interval == 0 and not save_due:
                evaluate(completed_steps)

            steps_in_current_epoch += 1
            dispatch_attempts = 0

            status = ray.get(controller.get_full_status.remote(), timeout=rpc_timeout)
            postfix = {
                "loss": f"{metrics.get('train/avg_loss', 0):.3f}",
                "thru": f"{status['inference_speed']:.1f}",
            }
            if "train/avg_acc" in metrics:
                postfix["acc"] = f"{metrics['train/avg_acc']:.3f}"
            for key, label in (("train/simulated_acc_len", "acc_len"),
                               ("train/suffix_simulated_acc_len", "suffix_acc_len")):
                if key in metrics:
                    postfix[label] = f"{metrics[key]:.2f}"
            if enable_perf:
                postfix["P~"] = (
                    f"{metrics.get('perf/producer_batch_rate_estimate', 0):.1f}"
                )
                postfix["Dcmp"] = f"{metrics.get('perf/draft_compute_rate', 0):.1f}"
                postfix["wait"] = f"{dispatch_wait:.1f}s"
                postfix["pool"] = status["sample_pool_size"]
            postfix["epoch"] = f"{current_epoch}/{num_epochs}"
            # Set postfix before update so tqdm emits only one line
            progress.set_postfix(postfix, refresh=False)
            progress.update(1)

            if _is_save_interval_step(completed_steps, args.save_interval):
                eval_metrics = evaluate(completed_steps)
                logger.info(f"Saving checkpoint at step {completed_steps}...")
                _save_training_checkpoint(
                    args,
                    train_group,
                    controller,
                    completed_steps,
                )
                last_saved_step = completed_steps
                best_eval_score = update_checkpoint_eval_meta(
                    args.checkpoint_dir, completed_steps, eval_metrics, best_eval_score
                )
                max_ckpts = getattr(args, "max_checkpoints", 0)
                if max_ckpts > 0:
                    _cleanup_old_checkpoints(args.checkpoint_dir, max_ckpts)

            _maybe_sync_draft_weights(args, completed_steps, train_group, inference_engines)

            if (
                source_work_budget is not None
                and previous_committed_work == source_work_budget
            ):
                logger.info(
                    "Finite source pass completed: committed_work=%s, optimizer_steps=%s",
                    previous_committed_work,
                    completed_steps,
                )
                break

            # Check if epoch completed
            if steps_in_current_epoch >= steps_per_epoch:
                logger.info(
                    f"Epoch {current_epoch} completed ({steps_in_current_epoch} steps). "
                    f"Total steps: {completed_steps}/{num_steps}"
                )

                if (
                    args.save_per_epoch
                    and args.checkpoint_dir
                    and last_saved_step != completed_steps
                ):
                    eval_metrics = evaluate(completed_steps)
                    logger.info(
                        f"Saving checkpoint at end of epoch {current_epoch} "
                        f"(step {completed_steps})..."
                    )
                    _save_training_checkpoint(
                        args,
                        train_group,
                        controller,
                        completed_steps,
                    )
                    last_saved_step = completed_steps
                    best_eval_score = update_checkpoint_eval_meta(
                        args.checkpoint_dir, completed_steps, eval_metrics, best_eval_score
                    )
                    max_ckpts = getattr(args, "max_checkpoints", 0)
                    if max_ckpts > 0:
                        _cleanup_old_checkpoints(args.checkpoint_dir, max_ckpts)

                if completed_steps < num_steps:
                    current_epoch += 1
                    steps_in_current_epoch = 0
                    logger.info(f"Dataset exhausted, reloading (epoch {current_epoch})...")
                    ray.get(controller.reload_dataset.remote(), timeout=rpc_timeout)
                else:
                    logger.info("Max steps reached")
                    break
            continue

        # Inner while broke (max steps reached during reload), break outer loop
        break

    if source_work_budget is not None and previous_committed_work != source_work_budget:
        remaining = source_work_budget - previous_committed_work
        valid_packed_drop_last = bool(
            getattr(args, "dflash_packing", False)
            and 0 < remaining < dp_size * accumulation_steps
        )
        if not valid_packed_drop_last:
            raise RuntimeError(
                "Training stopped before the finite source budget was committed: "
                f"committed={previous_committed_work}, budget={source_work_budget}, "
                f"completed_steps={completed_steps}, max_steps={num_steps}"
            )

    effective_source_work = (
        previous_committed_work if source_work_budget is not None else None
    )
    dropped_source_work = (
        int(requested_source_work_budget) - previous_committed_work
        if requested_source_work_budget is not None
        else 0
    )

    final_train_results = train_group.flush_pending_train_metrics()
    if not isinstance(final_train_results, (list, tuple)):
        final_train_results = []
    final_metrics = final_train_results[0] if final_train_results and final_train_results[0] else {}
    if final_metrics:
        deferred_final_step = final_metrics.pop("_metrics_step", None)
        final_metric_step = int(
            deferred_final_step
            if deferred_final_step is not None
            else completed_steps
        )
        final_optimizer_step = (
            int(deferred_final_step) + 1
            if deferred_final_step is not None
            else completed_steps
        )
        final_step_time = final_metrics.get("perf/step_time")
        if isinstance(final_step_time, (int, float)) and final_step_time > 0:
            trainer_compute_times[final_optimizer_step] = float(final_step_time)
        final_metrics["train/step"] = final_metric_step
        final_metrics["inference/step"] = completed_steps
        if enable_perf and previous_dispatch_wait is not None:
            final_metrics["perf/dispatch_wait"] = previous_dispatch_wait
            step_time = final_metrics.get("perf/step_time", 0)
            if step_time > 0:
                final_metrics["perf/draft_compute_rate"] = (
                    args.global_batch_size / step_time
                )
        _write_training_metrics(final_metrics, final_metric_step, completed_steps, tracking=tracking)
        last_observed_metrics = _numeric_metrics(final_metrics)

    progress.close()

    # Always save a final checkpoint unless saved.
    if (
        getattr(args, "save_final_checkpoint", True)
        and args.checkpoint_dir
        and last_saved_step != completed_steps
    ):
        eval_metrics = evaluate(completed_steps)
        logger.info(f"Saving final checkpoint at step {completed_steps}...")
        _save_training_checkpoint(
            args,
            train_group,
            controller,
            completed_steps,
        )
        best_eval_score = update_checkpoint_eval_meta(
            args.checkpoint_dir, completed_steps, eval_metrics, best_eval_score
        )
        max_ckpts = getattr(args, "max_checkpoints", 0)
        if max_ckpts > 0:
            _cleanup_old_checkpoints(args.checkpoint_dir, max_ckpts)

    final_status = ray.get(controller.get_full_status.remote(), timeout=rpc_timeout)
    scheduler_evidence = ray.get(controller.get_scheduler_evidence.remote(), timeout=rpc_timeout)
    performance = _build_performance_observation(
        step_records,
        warmup_steps=int(getattr(args, "performance_warmup_steps", 0)),
        drain_steps=int(getattr(args, "performance_drain_steps", 0)),
    )
    consumer_service_profile = _build_consumer_service_profile(
        trainer_compute_times,
        step_records,
        warmup_steps=int(getattr(args, "performance_warmup_steps", 0)),
        drain_steps=int(getattr(args, "performance_drain_steps", 0)),
    )
    logger.info(
        f"Training completed: {completed_steps} steps in {final_status['elapsed_seconds']:.1f}s | "
        f"avg inference={final_status['avg_inference_speed']:.1f} entries/s | "
        f"avg training={final_status['avg_training_speed']:.1f} entries/s"
    )
    return {
        "start_step": start_step,
        "completed_step": completed_steps,
        "optimizer_steps_observed": completed_steps - start_step,
        "dataset_size": dataset_size,
        "requested_source_work_budget": requested_source_work_budget,
        "source_work_budget": source_work_budget,
        "effective_source_work": effective_source_work,
        "dropped_source_work": dropped_source_work,
        "planned_dropped_source_work": planned_dropped_source_work,
        "source_work_alignment": getattr(args, "source_work_alignment", None),
        "source_work_tail_policy": getattr(args, "source_work_tail_policy", None),
        "committed_source_work": previous_committed_work,
        "eval_dataset_size": eval_dataset_size,
        "evaluation": {
            "enabled": eval_enabled,
            "cache_identity": eval_state.cache_identity,
            "selection_metric": eval_selection_metric(eval_state.history[-1]["metrics"] if eval_state.history else {}),
            "proxy_only": True,
            "history": eval_state.history,
            "best_checkpoint": max(
                (r for r in eval_state.history if r.get("checkpoint") and Path(r["checkpoint"]).is_dir()
                 and eval_selection_metric(r["metrics"]) in r["metrics"]),
                key=lambda r: r["metrics"].get(eval_selection_metric(r["metrics"]), -math.inf), default=None,
            ),
        },
        "final_metrics": last_observed_metrics,
        "final_controller_status": _numeric_metrics(final_status),
        "checkpoint": _checkpoint_observation(
            args.checkpoint_dir if getattr(args, "save_final_checkpoint", True) else None
        ),
        "resume": {
            "requested": bool(getattr(args, "load_path", None)),
            "mode": (
                "continual_training_with_source_cursor"
                if getattr(args, "continual_training", False)
                and resume_state is not None
                else "continual_training"
                if getattr(args, "continual_training", False)
                else "exact"
                if resume_state is not None
                else "none"
            ),
            "restored_optimizer_step": (
                int(resume_state["optimizer_step"])
                if resume_state is not None
                and not getattr(args, "continual_training", False)
                else 0
            ),
            "source_cursor_checkpoint_step": (
                int(resume_state["optimizer_step"])
                if resume_state is not None
                else 0
            ),
            "restored_committed_work": (
                int(resume_state["committed_work_count"])
                if resume_state is not None
                else 0
            ),
            "committed_work_sha256": (
                resume_state["committed_work_sha256"]
                if resume_state is not None
                else None
            ),
            "dataset_fingerprint": (
                resume_state["dataset_fingerprint"]
                if resume_state is not None
                else None
            ),
        },
        "feature_contract": None,
        "scheduler": scheduler_evidence,
        "performance": performance,
        "service_profiles": {
            "producer": producer_service_profile,
            "transport": (
                None
            ),
            "consumer": consumer_service_profile,
        },
        "execution": {
            "streaming_dispatch": streaming_dispatch,
            "streaming_optimizer_steps": streaming_steps,
            "finite_source_tail_optimizer_steps": finite_source_tail_steps,
            "optimizer_microsteps": optimizer_microsteps,
            "next_step_prefix_dispatch": pipeline_dispatch,
            "pipelined_microsteps": pipelined_microsteps,
            "attention_backend": getattr(args, "attention_backend", None),
            "context_parallel_size": int(
                getattr(args, "draft_context_parallel_size", 1)
            ),
        },
    }


def run_training_loop(
    args,
    controller,
    inference_manager,
    train_group,
    inference_engines=None,
    dataset_size=None,
    eval_dataset_size=None,
    cleanup_observer=None,
):
    inference_future = inference_manager.run.remote()
    result = None
    failure = None
    try:
        result = training_loop(
            args,
            controller,
            inference_manager,
            train_group,
            inference_engines=inference_engines,
            dataset_size=dataset_size,
            eval_dataset_size=eval_dataset_size,
            inference_future=inference_future,
        )
    except BaseException as exc:
        failure = exc
        raise
    finally:
        cleanup_args = {
            "args": args,
            "inference_manager": inference_manager,
            "inference_future": inference_future,
            "inference_engines": inference_engines,
            "controller": controller,
            "train_group": train_group,
        }
        if failure is not None:
            cleanup_args["force_kill_trainers"] = True
        cleanup_error = None
        try:
            cleanup = _safe_training_cleanup(**cleanup_args)
        except BaseException as exc:
            # An interrupted cleanup has no proof that native readers exited.
            cleanup_error = exc
            cleanup = {
                "state": "completed_with_errors", "source_teardown": "blocked",
                "operations": [{"component": "cleanup", "state": "failed",
                                "error": f"{type(exc).__name__}: {exc}"}],
            }
        try:
            if cleanup_observer is not None:
                cleanup_observer(cleanup)
        finally:
            # This safety decision must survive an Evidence write failure too.
            if cleanup.get("source_teardown") == "blocked":
                raise SourceTeardownBlockedError(
                    getattr(train_group, "_owner_statuses", None),
                    allow_node_loss=(
                        getattr(args, "cluster_mode", "single_node") == "static_multi_node"
                        and getattr(args, "mooncake_feature_affinity", "any") == "node_local_required"
                    ),
                    cleanup=cleanup,
                ) from (cleanup_error or failure)
    if isinstance(result, dict):
        result["cleanup"] = cleanup
        result["feature_contract"] = cleanup.get("feature_contract")
        result.setdefault("service_profiles", {})["transport"] = (
            result["feature_contract"].get("transport")
            if result["feature_contract"]
            else None
        )
    if failure is None and cleanup.get("feature_contract_error"):
        raise RuntimeError(cleanup["feature_contract_error"])
    return result
