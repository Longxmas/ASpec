# Adapted from TorchSpec 6c042a87140a84d13839e341ece2c5c3ada918bc,
# torchspec/offline/generate.py. Adds ASpec payload validation and run evidence.
#
# MIT License
#
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Materialize live target-model outputs for offline training."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path

import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from aspec.config.train_config import load_config
from aspec.controller.inference_manager import AsyncInferenceManager
from aspec.controller.loop import _safe_training_cleanup
from aspec.controller.setup import build_mooncake_config
from aspec.controller.training_controller import AsyncTrainingController
from aspec.inference.factory import create_inference_engines
from aspec.offline.dataset import OfflineDataset, target_identity
from aspec.offline.saving_actor import OfflineSavingActor
from aspec.ray.owner_fence import wait_for_owner_exit
from aspec.ray.placement_group import _ensure_ray_initialized, create_placement_groups
from aspec.train_entry import (
    _get_draft_model_config,
    _validate_and_configure_dflash,
)
from aspec.transfer.mooncake.utils import launch_mooncake_master
from aspec.utils.env import get_aspec_env_vars
from aspec.utils.logging import logger


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--output", "-o", required=True)
    cli, unknown = parser.parse_known_args()
    from aspec.config.train_config import validate_local_data_paths
    from aspec.planning import compile_execution_plan

    config = load_config(cli.config, cli_args=unknown or None, save_snapshot=False)
    validate_local_data_paths(config)
    if config.experiments:
        raise ValueError("Materialize one feature contract at a time")
    args = compile_execution_plan(config).to_runtime_args()
    args.rank = 0
    args.world_size = args.training_num_nodes * args.training_num_gpus_per_node
    if getattr(args, "attention_backend", None) == "usp":
        raise ValueError("Offline materialization does not support USP")
    return cli, args


def _save_vocab_mapping(controller, output_dir: str, draft_config) -> None:
    draft_vocab_size = getattr(draft_config, "draft_vocab_size", None)
    if draft_vocab_size is None or draft_vocab_size == draft_config.vocab_size:
        return
    path = Path(output_dir) / "vocab_mapping.pt"
    d2t, t2d = ray.get(
        controller.compute_vocab_mapping.remote(draft_config.vocab_size, draft_vocab_size),
        timeout=1800,
    )
    temporary = path.with_suffix(".pt.tmp")
    torch.save({"d2t": d2t.cpu(), "t2d": t2d.cpu()}, temporary)
    os.replace(temporary, path)


def _cleanup_materialization(
    args, controller, manager, manager_future, engines, saver, saver_owner, inference_pg,
):
    """Fence a failed saver before releasing buffers it may still be reading."""
    errors = []
    readers_safe = True
    if saver is not None:
        try:
            ray.get(saver.close.remote(), timeout=30)
        except Exception as exc:
            errors.append(f"offline_saver.close: {type(exc).__name__}: {exc}")
            try:
                ray.kill(saver, no_restart=True)
                if saver_owner is not None:
                    wait_for_owner_exit([saver_owner], timeout=30)
            except Exception as fence_error:
                # No reads are dispatched until the owner status has returned.
                readers_safe = saver_owner is None
                errors.append(f"offline_saver.fence: {type(fence_error).__name__}: {fence_error}")

    if readers_safe:
        try:
            cleanup = _safe_training_cleanup(
                args, manager, manager_future, engines, controller=controller,
                closed_reader_fence={
                    "state": "verified",
                    "scope": "offline reader closed or exited; no further reads dispatched",
                },
            )
        except Exception as exc:
            errors.append(f"source_cleanup: {type(exc).__name__}: {exc}")
            cleanup = {"state": "completed_with_errors"}
    else:
        # Source buffers and their placement group must stay alive when process
        # exit is unproven. Stop admission without invalidating registered memory.
        cleanup = {"state": "completed_with_errors", "source_teardown": "blocked"}
        for actor in (manager, controller):
            if actor is not None:
                try:
                    ray.kill(actor, no_restart=True)
                except Exception as exc:
                    errors.append(f"stop_admission: {type(exc).__name__}: {exc}")

    if readers_safe:
        for actor in (saver, manager, controller):
            if actor is not None:
                try:
                    ray.kill(actor, no_restart=True)
                except Exception as exc:
                    errors.append(f"actor_cleanup: {type(exc).__name__}: {exc}")
        # Only release the reservation after source teardown has acknowledged.
        if inference_pg is not None and cleanup.get("state") == "completed":
            try:
                pg, _, _ = inference_pg
                ray.util.remove_placement_group(pg)
            except Exception as exc:
                errors.append(f"placement_cleanup: {type(exc).__name__}: {exc}")
    if errors:
        cleanup = {**cleanup, "state": "completed_with_errors", "errors": errors}
        logger.error("Materialization cleanup errors: %s", errors)
    return cleanup


def materialize(args, output_dir: str) -> dict[str, int]:
    if getattr(args, "inference_engine_type", None) == "offline":
        raise ValueError("Materialization requires a live inference engine")

    if (
        getattr(args, "mooncake_data_plane", "store") != "store"
        or getattr(args, "mooncake_protocol", "tcp") != "tcp"
    ):
        raise ValueError("Materialization uses mooncake.data_plane=store and protocol=tcp")
    if args.draft_context_parallel_size != 1 or args.dflash_packing:
        raise ValueError("Materialization requires CP1 and dflash_packing=false")
    args.num_epochs = 1
    args.aspec_run_id = f"uuid:{uuid.uuid4()}"
    args.aspec_attempt_id = f"uuid:{uuid.uuid4()}"
    # Materialize the complete admitted source once, independently of the
    # optimizer-step and producer-profiling budgets in a reused training YAML.
    args.num_train_steps = None
    args.topology_producer_profile_samples = 0
    draft_config = _get_draft_model_config(args)
    args.draft_model_config_obj = draft_config
    _validate_and_configure_dflash(args, draft_config)
    # A one-item dispatch lets the saving actor persist every source sample,
    # including a final partial inference batch.
    args.per_dp_rank_batch_size = 1
    OfflineDataset(
        output_dir,
        create=True,
        last_hidden_states_prenorm=args.last_hidden_states_prenorm,
        binding={
            "feature_contract": args.aspec_feature_contract,
            "target_identity": target_identity(args.target_model_path),
        },
    )

    _ensure_ray_initialized()
    controller = manager = manager_future = saver = inference_pg = None
    saver_owner = None
    engines = []
    counts = None
    rpc_timeout = float(getattr(args, "optimizer_step_timeout_seconds", 1800.0))

    def get(ref):
        return ray.get(ref, timeout=rpc_timeout)

    try:
        driver_node_id = ray.get_runtime_context().get_node_id()
        controller = AsyncTrainingController.options(
            runtime_env={"env_vars": get_aspec_env_vars()},
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=driver_node_id, soft=False),
        ).remote(args, 1)
        train_size, eval_size = get(
            [controller.load_dataset.remote(args), controller.load_eval_dataset.remote(args)]
        )
        _save_vocab_mapping(controller, output_dir, draft_config)

        inference_pg = create_placement_groups(args, roles={"inference"})["inference"]
        launch_mooncake_master(args)
        mooncake_config = build_mooncake_config(args)
        saver = OfflineSavingActor.remote(output_dir, mooncake_config, controller)
        saver_owner = get(saver.get_owner_status.remote())
        args.aspec_read_owners = [saver_owner]
        engines = create_inference_engines(args, inference_pg, mooncake_config)
        train_queues, eval_queues = get(
            [controller.get_train_queues.remote(), controller.get_eval_queues.remote()]
        )
        get(saver.set_queues.remote(train_queues[0], eval_queues[0]))

        if getattr(args, "max_sample_pool_size", 0) <= 0:
            args.max_sample_pool_size = max(
                8,
                getattr(args, "inference_batch_size", 1)
                * getattr(args, "max_concurrent_batches", 1)
                * 4,
            )
        manager = AsyncInferenceManager.remote(
            args,
            controller,
            inference_engines=engines,
            max_concurrent_batches=getattr(args, "max_concurrent_batches", 1),
        )
        if eval_size:
            get(controller.submit_eval_chunk.remote(0, eval_size))
        get(controller.submit_training_dataset.remote())

        manager_future = manager.run.remote()
        expected = {"train": train_size, "eval": eval_size}
        processed = {"train": 0, "eval": 0}
        last_progress = time.monotonic()
        while processed != expected:
            made_progress = False
            for split, method in (
                ("eval", controller.try_dispatch_eval_batch),
                ("train", controller.try_dispatch_batch),
            ):
                if processed[split] < expected[split] and get(method.remote()):
                    get(saver.save_from_queue.remote(split))
                    processed[split] += 1
                    made_progress = True

            if made_progress:
                last_progress = time.monotonic()
                logger.info("Materialization progress: %s / %s", processed, expected)
                continue

            manager_status, controller_status = get(
                [manager.get_status.remote(), controller.get_full_status.remote()]
            )
            drained = (
                manager_status["prompt_buffer_size"] == 0
                and manager_status["pending_tasks"] == 0
                and controller_status["prompt_buffer_size"] == 0
                and controller_status["sample_pool_size"] == 0
                and get(controller.get_eval_pool_size.remote()) == 0
            )
            finished, _ = ray.wait([manager_future], timeout=0)
            if finished:
                get(manager_future)
                raise RuntimeError("Inference manager stopped before materialization completed")
            if drained or manager_status.get("sample_budget_drained"):
                raise RuntimeError(
                    f"Inference drained early: processed={processed}, expected={expected}"
                )
            if time.monotonic() - last_progress >= rpc_timeout:
                raise TimeoutError(f"Materialization made no progress for {rpc_timeout}s")
            time.sleep(0.05)

        if eval_size:
            get(controller.finalize_eval_dispatch.remote())
        get(saver.finalize.remote(expected))
        counts = get(saver.counts.remote())
    finally:
        cleanup = _cleanup_materialization(
            args, controller, manager, manager_future, engines, saver, saver_owner,
            inference_pg,
        )
        try:
            (Path(output_dir) / "materialization.json").write_text(
                json.dumps({"counts": counts, "cleanup": cleanup}, indent=2) + "\n"
            )
        except OSError:
            logger.exception("Failed to write materialization cleanup report")

    if cleanup["state"] != "completed":
        raise RuntimeError("Materialization cleanup failed; see materialization.json")
    logger.info("Offline dataset ready at %s: %s", output_dir, counts)
    return counts


def main() -> None:
    cli, args = _parse_args()
    materialize(args, cli.output)


if __name__ == "__main__":
    main()
