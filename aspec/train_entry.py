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

"""ASpec training entry point for DSpark, DFlash and EAGLE3."""

import argparse
import json
import os
import sys
import time
import uuid

# Keep ATen and Triton GEMM candidates available for compiled attention backward.
# Preserve an explicitly supplied backend selection.
os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS", "ATEN,TRITON")
from collections import namedtuple
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import ray
import torch
from omegaconf import OmegaConf
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from aspec import AutoDraftModelConfig
from aspec._environment import require_draft_environment
from aspec.config.train_config import _save_config_snapshot, load_config
from aspec.config.utils import generate_draft_model_config
from aspec.controller import (
    AsyncTrainingController,
    auto_calculate_training_steps,
    build_mooncake_config,
    run_training_loop,
    setup_async_training_with_engines,
)
from aspec.evidence import RunEvidenceRecorder
from aspec.inference.factory import prepare_inference_engines, rollback_unused_targets
from aspec.planning import ExecutionPlan, compile_execution_plan, save_execution_plan
from aspec.ray.cluster_preflight import run_cluster_preflight
from aspec.ray.owner_fence import SourceTeardownBlockedError, run_with_source_fencing
from aspec.ray.placement_group import (
    allocate_train_group,
    create_placement_groups,
    ensure_ray_initialized,
)
from aspec.reporting import RunReportError, build_run_report, render_run_report
from aspec.runtime_control import (
    install_termination_signal_handler,
    termination_signal_handlers,
)
from aspec.training.trainer_actor import TrainerActor
from aspec.transfer.mooncake.utils import launch_mooncake_master
from aspec.utils.env import get_aspec_env_vars
from aspec.utils.logging import init_tracking, logger

_Phase = namedtuple("_Phase", ["name", "duration", "is_async", "blocked"])

_MANAGED_RUN_ARTIFACTS = (
    "evaluation.json",
    "tracking.json",
    "config.yaml",
    "plan.json",
    "evidence.json",
    "cluster-preflight.json",
    "checkpoints",
    "scratch_draft_model",
    "runs",
    "traces",
)


def _prepare_run_directory(output_dir: str) -> Path:
    """Claim a run directory without rejecting unrelated user files."""

    if not output_dir:
        raise ValueError("output_dir is required for a real ASpec run")
    path = Path(output_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    conflicts = [name for name in _MANAGED_RUN_ARTIFACTS if (path / name).exists()]
    if conflicts:
        raise FileExistsError(
            "output_dir already contains ASpec-managed run artifacts; choose a fresh "
            f"run directory: output_dir={path}, conflicts={conflicts}"
        )
    return path


class _InitTimer:
    """Lightweight segmented timer for initialization phases."""

    def __init__(self) -> None:
        self._t0 = time.time()
        self._phases: list[_Phase] = []
        self._pending: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str) -> Generator[None, None, None]:
        """Time a synchronous phase."""
        start = time.time()
        yield
        self._phases.append(_Phase(name, time.time() - start, is_async=False, blocked=0.0))

    def begin_async(self, name: str) -> None:
        """Mark the start of an async operation (e.g., ray.remote dispatch)."""
        self._pending[name] = time.time()

    def wait(self, name: str, refs, *, timeout: float | None = None) -> Any:
        """Wrap ray.get for async phases. Returns the result."""
        if name not in self._pending:
            raise ValueError(f"No async phase '{name}' was started via begin_async()")
        t_before = time.time()
        result = ray.get(refs) if timeout is None else ray.get(refs, timeout=timeout)
        t_after = time.time()
        dispatch_time = self._pending.pop(name)
        total = t_after - dispatch_time
        blocked = t_after - t_before
        self._phases.append(_Phase(name, total, is_async=True, blocked=blocked))
        return result

    def log_summary(self) -> None:
        total = time.time() - self._t0
        lines = ["Initialization timing:"]
        for p in self._phases:
            suffix = f"  (blocked {p.blocked:.2f}s)" if p.is_async else ""
            lines.append(f"  {p.name:<48s} {p.duration:>8.2f}s{suffix}")
        lines.append(f"  {'─' * 57}")
        lines.append(f"  {'Total':<48s} {total:>8.2f}s")
        logger.info("\n".join(lines))


def _prepare_config() -> tuple[argparse.Namespace, ExecutionPlan]:
    """Compile CLI input and return both the retained runtime adapter and its plan.

    Supports configs with sections matching the Config dataclass:
    model, dataset, training, debug, inference, logging, mooncake, decode.

    TorchSpec-style fields stop at the planner boundary. The returned flat
    namespace is generated from the compiled plan for the retained upstream
    runtime and is not the user-facing configuration contract.
    """

    parser = argparse.ArgumentParser(description="ASpec speculative decoding training")
    parser.add_argument("--config", "-c", type=str, help="Path to YAML config")
    parser.add_argument(
        "--report",
        metavar="RUN_DIR",
        help="Validate plan/evidence identity and summarize an existing run",
    )
    parser.add_argument(
        "--report-format",
        choices=("text", "json"),
        default="text",
        help="Output format used with --report (default: text)",
    )
    parser.add_argument(
        "--print-config-only", action="store_true", help="Print resolved config and exit"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compile and print the canonical execution plan without starting Ray or SGLang",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Connect to the explicit static cluster, validate production inventory, "
            "write cluster-preflight.json, and exit before model loading"
        ),
    )

    args, unknown = parser.parse_known_args()

    if args.report:
        if args.config or args.print_config_only or args.dry_run or args.preflight_only or unknown:
            parser.error("--report cannot be combined with training or config arguments")
        try:
            report = build_run_report(args.report)
        except RunReportError as exc:
            parser.error(str(exc))
        if args.report_format == "json":
            print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
        else:
            print(render_run_report(report))
        raise SystemExit(0)
    if not args.config:
        parser.error("--config is required unless --report is used")
    if args.report_format != "text":
        parser.error("--report-format requires --report")
    if args.preflight_only and (args.dry_run or args.print_config_only):
        parser.error("--preflight-only cannot be combined with --dry-run or --print-config-only")

    if not (args.dry_run or args.print_config_only):
        require_draft_environment()

    config = load_config(
        config_path=args.config,
        cli_args=unknown if unknown else None,
        save_snapshot=False,
    )

    if args.print_config_only:
        logger.info("Resolved config:\n%s", OmegaConf.to_yaml(config, resolve=True))
        sys.exit(0)

    from aspec.config.train_config import validate_local_data_paths

    validate_local_data_paths(config)
    if config.experiments:
        from aspec.multi_draft import compile_multi_draft_plan

        if args.preflight_only:
            parser.error("shared Target training does not support --preflight-only")
        plan = compile_multi_draft_plan(config)
    else:
        plan = compile_execution_plan(
            config,
            config_path=args.config,
            cli_args=unknown if unknown else None,
        )
    if args.dry_run:
        print(plan.to_json())
        sys.exit(0)

    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(config, resolve=True))
    logger.info("Compiled execution plan %s", plan.plan_id)
    _prepare_run_directory(str(config.output_dir or ""))
    _save_config_snapshot(config)
    save_execution_plan(plan, os.path.join(config.output_dir, "plan.json"))

    runtime_args = plan.to_runtime_args()
    runtime_args.preflight_only = args.preflight_only
    return runtime_args, plan


def parse_config() -> argparse.Namespace:
    """Return the retained runtime adapter for callers using the historical entry helper."""
    args, _plan = _prepare_config()
    return args


def _maybe_create_scratch_draft(args, train_group):
    """Auto-create scratch draft checkpoint for inference engine if not provided."""
    if (
        getattr(args, "train_with_decode", False)
        and getattr(args, "decode_speculative_algorithm", None)
        and getattr(args, "decode_speculative_draft_model_path", None) is None
    ):
        scratch_dir = os.path.join(getattr(args, "output_dir", "./outputs"), "scratch_draft_model")
        os.makedirs(scratch_dir, exist_ok=True)
        logger.info(f"Auto-creating scratch draft checkpoint at {scratch_dir}")
        train_group.save_draft_model_for_serving(scratch_dir)
        args.decode_speculative_draft_model_path = scratch_dir
        logger.info(f"Set decode_speculative_draft_model_path = {scratch_dir}")


def _get_draft_model_config(args):
    """Resolve draft model config from args or auto-generate from target model."""

    draft_config_path = getattr(args, "draft_model_config", None)
    if draft_config_path is not None:
        return AutoDraftModelConfig.from_file(draft_config_path)

    config_dict = generate_draft_model_config(
        target_model_path=args.target_model_path,
        cache_dir=getattr(args, "model_download_dir", None),
        trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
    )
    return AutoDraftModelConfig.from_dict(config_dict)


def _validate_and_configure_dflash(args, draft_model_config) -> None:
    """Validate DFlash-specific config and auto-set aux layer IDs.

    Called before dataset loading to fail fast on misconfigurations.
    """
    from aspec.models.draft.dflash import DFlashConfig
    from aspec.models.draft.dspark import DSparkConfig

    if not isinstance(draft_model_config, DFlashConfig):
        return

    is_dspark = isinstance(draft_model_config, DSparkConfig)
    algo = "DSpark" if is_dspark else "DFlash"

    engine_type = getattr(args, "inference_engine_type", "sgl")
    if engine_type not in {"sgl", "offline"}:
        raise NotImplementedError(
            f"{algo} currently supports inference_engine_type='sgl' or 'offline', got '{engine_type}'."
        )
    if getattr(args, "defer_tokenization", False):
        raise NotImplementedError(f"{algo} does not support defer_tokenization=True.")
    block_size = getattr(args, "dflash_block_size", 16)
    min_loss = getattr(args, "min_loss_tokens", 0)
    if min_loss < 2 * block_size:
        raise ValueError(
            f"{algo} requires dataset.min_loss_tokens >= 2 * training.dflash_block_size "
            f"({min_loss} < {2 * block_size}). Set dataset.min_loss_tokens={2 * block_size}."
        )

    target_layer_ids = getattr(draft_model_config, "target_layer_ids", None)
    if not getattr(args, "aux_hidden_states_layers", None):
        from aspec.models.draft.dflash import build_target_layer_ids

        if target_layer_ids is None:
            num_target = getattr(draft_model_config, "num_target_layers", 5)
            target_num_hidden = getattr(draft_model_config, "target_num_hidden_layers", 36)
            target_layer_ids = build_target_layer_ids(num_target, target_num_hidden)
        args.aux_hidden_states_layers = target_layer_ids
        logger.info(f"{algo}: set aux_hidden_states_layers = {target_layer_ids}")


def train_async_no_generation(args, evidence: RunEvidenceRecorder | None = None):
    """Entry point for asynchronous DSpark/DFlash training.

    Supports prefill-only mode (default) and decode mode (train_with_decode=True)
    with speculative decoding. Uses distributed Ray actors with placement groups.
    Engines store tensors in mooncake and return keys to AsyncInferenceManager.
    """
    if (
        getattr(args, "train_with_decode", False)
        and getattr(args, "inference_engine_type", "sgl") != "sgl"
    ):
        raise ValueError("train_with_decode=True requires inference_engine_type=sgl")

    if getattr(args, "inference_engine_type", None) == "offline":
        from aspec.offline.dataset import (
            OfflineDataset,
            configure_offline_args,
        )

        offline_dataset = OfflineDataset(args.offline_data_path)
        configure_offline_args(offline_dataset, args)

    init_tracking(args)
    timer = _InitTimer()

    # [1] Create controller early (lightweight: only needs args + dp_size)
    with timer.phase("Create controller"):
        driver_node_id = ray.get_runtime_context().get_node_id()
        controller = AsyncTrainingController.options(
            runtime_env={"env_vars": get_aspec_env_vars(args)},
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=driver_node_id, soft=False),
        ).remote(args, args.dp_size)

    # [1.5] Parse draft config + DFlash validation (before any async work)
    with timer.phase("Parse draft model config"):
        draft_model_config = _get_draft_model_config(args)
        args.draft_model_config_obj = draft_model_config

        _validate_and_configure_dflash(args, draft_model_config)

    # [2] Kick off dataset loading on controller (async — runs on actor while driver continues)
    timer.begin_async("Dataset loading")
    dataset_size_ref = controller.load_dataset.remote(args)
    eval_dataset_size_ref = controller.load_eval_dataset.remote(args)

    # [3] Do initialization that doesn't depend on dataset in parallel
    with timer.phase("Driver-side init"):
        roles = (
            {"training"}
            if getattr(args, "inference_engine_type", None) == "offline"
            else {"training", "inference"}
        )
        pgs = create_placement_groups(args, roles=roles)
        launch_mooncake_master(args)
        mooncake_config = build_mooncake_config(args)

    # [4] Wait for dataset sizes (small ints, unlike the old ray.put of the full dataset)
    dataset_size, eval_dataset_size = timer.wait(
        "Dataset loading", [dataset_size_ref, eval_dataset_size_ref]
    )
    dataset_admission = ray.get(controller.get_dataset_admission_evidence.remote())
    logger.info(f"Dataset loaded on controller: {dataset_size} train, {eval_dataset_size} eval")
    if evidence is not None:
        evidence.record_dataset_admission(
            dataset_size=dataset_size,
            eval_dataset_size=eval_dataset_size,
            dataset_admission=dataset_admission,
        )

    # [5] Auto-calculate training steps (needs dataset_size)
    with timer.phase("Auto-calculate training steps"):
        auto_calculate_training_steps(args, dataset_size)

    # [6] Bind each pruned head to its immutable vocabulary.
    with timer.phase("Vocab mapping"):
        vocab_mapping = _prepare_vocab_mapping(args, draft_model_config, controller)

    # [7] Create training actors + inference engines (args now has num_train_steps)
    timer.begin_async("Actor initialization")
    with timer.phase("Allocate actors + dispatch init"):
        train_group = allocate_train_group(
            args=args,
            num_nodes=args.training_num_nodes,
            num_gpus_per_node=args.training_num_gpus_per_node,
            pg=pgs["training"],
            training_class=TrainerActor,
        )
        args.aspec_read_owners = ray.get([
            actor.get_owner_identity.remote() for actor in train_group._actor_handlers
        ])
        train_init_refs = train_group.async_init(
            args, role="training", mooncake_config=mooncake_config, with_ref=False
        )

        # Decode mode: create scratch draft checkpoint before inference engines
        # are prepared, since they need decode_speculative_draft_model_path on args.
        # This blocks on train actor init (FSDP gather), so inference engines are
        # dispatched after to maximize parallelism with the wait below.
        _maybe_create_scratch_draft(args, train_group)

        inference_engines, engine_init_refs = prepare_inference_engines(
            args, pgs.get("inference"), mooncake_config
        )

    try:
        # [8] Wait for all actor init to complete concurrently
        n_train = len(train_init_refs)
        logger.info(
            f"Waiting for {n_train} training actors and {len(engine_init_refs)} "
            f"inference engines to initialize in parallel..."
        )
        backend_name = {"sgl": "sglang"}.get(args.inference_engine_type, args.inference_engine_type)
        init_timeout = float(getattr(args, f"{backend_name}_init_timeout", 1200))
        all_results = timer.wait(
            "Actor initialization", train_init_refs + engine_init_refs, timeout=init_timeout
        )

        train_results = all_results[:n_train]
        if len(set(train_results)) != 1:
            raise RuntimeError(
                "Training actors returned inconsistent initialization results: "
                f"{train_results!r}"
            )
        logger.info(
            f"All {n_train} training actors and {len(engine_init_refs)} inference engines initialized"
        )
        training_actor_statuses = train_group.get_status()
        inference_engine_statuses = ray.get(
            [engine.get_status.remote() for engine in inference_engines]
        )
        if evidence is not None:
            evidence.record_actor_initialization(
                training_actor_count=n_train,
                training_actor_statuses=training_actor_statuses,
                inference_engine_count=len(engine_init_refs),
                inference_engine_statuses=inference_engine_statuses,
                dataset_size=dataset_size,
                eval_dataset_size=eval_dataset_size,
                dataset_admission=dataset_admission,
                cluster_resources=ray.cluster_resources(),
                alive_node_count=sum(node.get("Alive", False) for node in ray.nodes()),
            )

        if vocab_mapping is not None:
            train_group.set_vocab_buffers(*vocab_mapping)
            logger.info("Loaded vocab mapping into training actors")

        # [9] Setup async training with pre-created controller
        with timer.phase("Setup async training"):
            controller, inference_manager = setup_async_training_with_engines(
                args,
                train_group,
                mooncake_config,
                inference_engines,
                controller=controller,
                inference_engine_statuses=inference_engine_statuses,
            )

        timer.log_summary()

    except BaseException:
        if args.inference_engine_type == "sgl":
            rollback_unused_targets(inference_engines)
        raise

    # [10] Run training loop (no ray.put needed — dataset lives on controller)
    summary = run_training_loop(
        args,
        controller,
        inference_manager,
        train_group,
        inference_engines=inference_engines,
        dataset_size=dataset_size,
        eval_dataset_size=eval_dataset_size,
        cleanup_observer=evidence.record_cleanup if evidence is not None else None,
    )
    if evidence is not None:
        evidence.record_training(summary)
    return summary


def _prepare_vocab_mapping(args, draft_model_config, controller):
    """Fresh heads get a dataset mapping; loaded heads keep their own rows."""
    from aspec.models.vocab import validate_vocab_mapping
    from aspec.training.checkpoint import resolve_resume_model_dir

    target_size = draft_model_config.vocab_size
    draft_size = getattr(draft_model_config, "draft_vocab_size", None) or target_size
    if draft_size == target_size:
        return None
    if resolve_resume_model_dir(args) is not None or getattr(args, "initial_draft_model_path", None):
        logger.info("Using the vocabulary mapping bound to the loaded draft weights")
        return None
    if getattr(args, "keep_initial_vocab_mapping", False):
        raise ValueError("keep_initial_vocab_mapping requires an existing draft checkpoint")
    if getattr(args, "inference_engine_type", None) == "offline":
        mapping_path = os.path.join(args.offline_data_path, "vocab_mapping.pt")
        saved = torch.load(mapping_path, map_location="cpu", weights_only=True)
        mapping = (saved["d2t"], saved["t2d"])
    else:
        mapping = ray.get(controller.compute_vocab_mapping.remote(target_size, draft_size))
    validate_vocab_mapping(*mapping, draft_size, target_size)
    return mapping


def main() -> None:
    # Programmatic launchers use main() too; protect their finally/atexit
    # teardown just as the console entry point's interpreter exit.
    run_with_source_fencing(_main)


def _main() -> None:
    args, plan = _prepare_config()
    from aspec.inference.managed import managed_target
    from aspec.multi_draft import MultiDraftPlan

    draft_gpu_count = (sum(child.to_runtime_args().world_size for child in plan.plans.values())
                       if isinstance(plan, MultiDraftPlan) else None)
    with termination_signal_handlers(), managed_target(args, draft_gpu_count=draft_gpu_count):
        _run_prepared(args, plan)


def _run_prepared(args, plan):
    from aspec.multi_draft import MultiDraftPlan, run_multi_draft

    if isinstance(plan, MultiDraftPlan):
        try:
            ensure_ray_initialized()
            install_termination_signal_handler()
            run_multi_draft(args, plan)
        finally:
            if ray.is_initialized():
                ray.shutdown()
        return
    # ``preflight_only`` is an argparse boolean.  Requiring the literal value
    # avoids treating proxy objects with truthy missing attributes as an
    # explicit request to skip the workload.
    preflight_only = getattr(args, "preflight_only", False) is True
    run_id = f"uuid:{uuid.uuid4()}"
    attempt_id = f"uuid:{uuid.uuid4()}"
    args.aspec_run_id = run_id
    args.aspec_attempt_id = attempt_id
    os.environ["ASPEC_RUN_ID"] = run_id
    os.environ["ASPEC_ATTEMPT_ID"] = attempt_id
    evidence = None
    if not preflight_only:
        evidence = RunEvidenceRecorder(
            plan,
            args.output_dir,
            run_id=run_id,
            attempt_id=attempt_id,
        )
    try:
        # Ray installs its own SIGTERM handler during initialization. Start it
        # explicitly, then restore ASpec's cancellable handler for the run.
        ensure_ray_initialized()
        install_termination_signal_handler()
        cluster_preflight = run_cluster_preflight(args)
        if preflight_only:
            if cluster_preflight is None:
                raise RuntimeError(
                    "--preflight-only requires cluster.mode=static_multi_node"
                )
            output_path = Path(args.output_dir) / "cluster-preflight.json"
            temporary = output_path.with_name(
                f".{output_path.name}.{os.getpid()}.tmp"
            )
            temporary.write_text(
                json.dumps(
                    cluster_preflight,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, output_path)
            print(json.dumps(cluster_preflight, indent=2, sort_keys=True))
            ray.shutdown()
            return
        if cluster_preflight is not None and evidence is not None:
            evidence.record_cluster_preflight(cluster_preflight)
        train_async_no_generation(args, evidence=evidence)
    except BaseException as exc:
        try:
            if evidence is not None:
                try:
                    evidence.fail(exc)
                except Exception:
                    logger.exception("Failed to record training failure")
        finally:
            # Even cancellation during failure reporting must preserve the
            # pending fence instead of reaching Ray's implicit exit cleanup.
            if isinstance(exc, SourceTeardownBlockedError):
                raise exc
        if ray.is_initialized():
            # Initialization failures also precede the training-loop cleanup
            # registry. A blocked read-owner fence instead retains its sources;
            # disconnecting here would bypass that lifetime guarantee.
            try:
                ray.shutdown()
            except Exception:
                logger.exception("Failed to disconnect Ray after training failure")
        raise
    else:
        if evidence is not None:
            evidence.complete()
    finally:
        from aspec.ray.node_local_files import close_stores

        if not isinstance(sys.exception(), SourceTeardownBlockedError):
            close_stores(args)


if __name__ == "__main__":
    main()
