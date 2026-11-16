# Copyright 2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Release recipe launcher and bounded functional acceptance (not quality certification)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from aspec._environment import command_environment, require_draft_environment
from aspec.assets import asset_path, asset_root, recipe_config

ROOT = asset_root()
CATALOG = ROOT / "tools/recipe_catalog.json"


def catalog():
    return json.loads(CATALOG.read_text())


def execute(argv, log=None):
    command = list(map(str, argv))
    print("+ " + shlex.join(command), flush=True)
    if log is None:
        subprocess.run(command, env=command_environment(), check=True)
    else:
        with Path(log).open("w") as stream:
            subprocess.run(command, env=command_environment(), stdout=stream, stderr=subprocess.STDOUT, check=True)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(source, destination):
    """Validate/copy conversation JSONL or convert already-tokenized JSONL. No re-tokenization."""
    from aspec.commands.convert_pretokenized_jsonl import convert

    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError(f"Refusing existing destination: {destination}")
    if source.suffix == ".parquet":
        raise ValueError("Use existing typed Parquet directly; prepare accepts JSONL inputs")
    with source.open() as f:
        first = json.loads(next(f))
    if "input_ids" in first:
        if destination.suffix != ".parquet":
            raise ValueError("Tokenized JSONL requires a .parquet destination")
        convert(source, destination)
    else:
        if destination.suffix != ".jsonl":
            raise ValueError("Conversation input requires a .jsonl destination")
        rows = 0
        with source.open() as f:
            for line in f:
                row = json.loads(line)
                turns = row.get("conversations")
                if (
                    not isinstance(turns, list)
                    or not turns
                    or not any(t.get("role") == "assistant" for t in turns)
                ):
                    raise ValueError(
                        f"Row {rows + 1}: conversations with assistant supervision required"
                    )
                if any(
                    t.get("role") not in {"system", "user", "assistant", "tool"}
                    or not isinstance(t.get("content"), str)
                    for t in turns
                ):
                    raise ValueError(f"Row {rows + 1}: unsupported conversation schema")
                rows += 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as src, destination.open("xb") as dst:
            __import__("shutil").copyfileobj(src, dst)
    return {"source": str(source), "sha256": digest(source), "output": str(destination)}


def check_inputs(recipe, *, acceptance=False):
    if recipe == "quickstart":
        return {}
    paths = {}
    for key in ("ASPEC_TRAIN_DATA", "ASPEC_EVAL_DATA"):
        value = os.environ.get(key)
        if not value or not Path(value).is_absolute() or not Path(value).is_file():
            raise ValueError(f"{key} must name an existing absolute train/eval file")
        paths[key] = {"path": value, "sha256": digest(value)}
    if paths["ASPEC_TRAIN_DATA"]["sha256"] == paths["ASPEC_EVAL_DATA"]["sha256"]:
        raise ValueError("Train and eval files must differ; provide disjoint data")
    minimum = catalog()[recipe].get("minimum_input_tokens") if acceptance else None
    if minimum:
        import pyarrow.parquet as pq

        # Require every training row to exercise the long-context lane, so a short
        # bounded run cannot accidentally consume only short rows from a mixed corpus.
        count = 0
        for batch in pq.ParquetFile(paths["ASPEC_TRAIN_DATA"]["path"]).iter_batches(
            columns=["input_ids"], batch_size=8
        ):
            for ids in batch.column(0).to_pylist():
                if len(ids) < minimum:
                    raise ValueError(
                        f"Long-context acceptance needs every input >= {minimum} tokens"
                    )
                count += 1
        if not count:
            raise ValueError("Empty long-context training data")
        paths["long_context_rows"] = count
    return paths


def assert_report(report, step):
    if report.get("state") != "completed" or report.get("health") != "healthy":
        raise ValueError("Run must be completed/healthy; incomplete evidence is not a pass")
    if "experiments" in report:
        for child in report["experiments"].values():
            assert_report(child, step)
        return
    if report.get("fallbacks") or report["required_capabilities"]["pending"]:
        raise ValueError("Unexpected fallback or unobserved required capability")
    training = report["training"]
    if training["completed_step"] != step or training.get("no_new_work"):
        raise ValueError(f"Expected completed step {step}, got {training['completed_step']}")
    metrics = training["final_metrics"]
    losses = [v for k, v in metrics.items() if "loss" in k and isinstance(v, (int, float))]
    if not losses or any(not math.isfinite(v) for v in losses):
        raise ValueError("Missing/nonfinite loss metrics")
    for key, value in metrics.items():
        if isinstance(value, (float, int)) and not math.isfinite(value):
            raise ValueError(f"Nonfinite metric {key}")
    lifecycle = report["cleanup"]["work_lifecycle"]
    if lifecycle.get("resident_outputs") != 0 or lifecycle.get("resident_bytes") != 0:
        raise ValueError("Missing or nonzero terminal ownership counters")


def audit_run(directory, step, resumed=False, require_eval=False, expected_ranks=None):
    from aspec.reporting import build_run_report

    directory = Path(directory)
    report = build_run_report(directory)
    assert_report(report, step)
    children = list(report.get("experiments", {})) or [""]
    for child in children:
        run = directory / child
        child_report = report.get("experiments", {}).get(child, report)
        if require_eval:
            evaluation = child_report["training"].get("evaluation") or {}
            if not evaluation.get("enabled") or not evaluation.get("history"):
                raise ValueError("Held-out evaluation was not observed")
            for observation in evaluation["history"]:
                metrics = observation.get("metrics", {})
                if not metrics or any(
                    not math.isfinite(v) for v in metrics.values() if isinstance(v, (float, int))
                ):
                    raise ValueError("Missing/nonfinite held-out metrics")
        work_file = run / "checkpoints" / f"iter_{step + 1:07d}" / "work_state.json"
        work = json.loads(work_file.read_text())
        ids = work["committed_work_ids"]
        if not ids or len(ids) != len(set(ids)) or work["committed_work_count"] != len(ids):
            raise ValueError("Missing or repeated committed work")
        if resumed:
            old = json.loads(
                (
                    directory.parent
                    / "first"
                    / child
                    / "checkpoints"
                    / f"iter_{step // 2 + 1:07d}"
                    / "work_state.json"
                ).read_text()
            )
            if not set(old["committed_work_ids"]) < set(ids):
                raise ValueError("Resume lost committed work or made no progress")
        initial = sorted(run.glob("recipe-audit-initial*.json"))
        if expected_ranks is not None and len(initial) != expected_ranks:
            raise ValueError("Missing rank audits; every trainer rank must be observed")
        if not initial:
            raise ValueError(f"No trainer state audit in {run}")
        for path in initial:
            suffix = path.stem.removeprefix("recipe-audit-initial")
            before = json.loads(path.read_text())
            final = run / f"recipe-audit-checkpoint-{step:07d}{suffix}.json"
            after = json.loads(final.read_text())
            if before["state"]["model"] == after["state"]["model"]:
                raise ValueError("Training did not update model parameters")
            if resumed:
                saved_audit = directory.parent / "first" / child / (
                    f"recipe-audit-checkpoint-{step // 2:07d}{suffix}.json"
                )
                saved = json.loads(saved_audit.read_text())
                if before["scheduler"] != saved["scheduler"]:
                    raise ValueError("Checkpoint LR scheduler state was not restored exactly")
                diff = json.loads((run / f"recipe-resume-diff{suffix}.json").read_text())
                states = diff.get("state", {})
                required_states = {"model", "master", "exp_avg", "exp_avg_sq"}
                if (not diff.get("rng_exact") or not required_states <= states.keys() or any(
                    value.get("coordinates", 0) <= 0
                    or value["relative_l2"] != 0 or value["max_absolute"] != 0
                    for value in states.values()
                )):
                    raise ValueError("Checkpoint state/RNG was not restored exactly")
    (directory / "recipe-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return children


def export_asset(checkpoint, output, *, log_prefix=None):
    # Native strict roundtrip works for every declared structure. Serving-format
    # acceptance is a separate boundary; this does not certify SGLang decoding.
    execute(
        [
            sys.executable,
            "-m",
            "aspec.commands.convert_to_hf",
            "--input-dir",
            checkpoint,
            "--output-dir",
            output,
            "--format",
            "aspec",
        ],
        Path(str(log_prefix) + "-export.log") if log_prefix is not None else None,
    )
    execute(
        [
            sys.executable,
            "-c",
            "from aspec.draft_artifact import load_native_draft; import sys; load_native_draft(sys.argv[1]); print('Strict native load passed')",
            output,
        ],
        Path(str(log_prefix) + "-load.log") if log_prefix is not None else None,
    )


def stop_memory_monitor(process, stream):
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    stream.close()


def acceptance(recipe, output, steps, overrides=()):
    return run_acceptance(recipe, catalog()[recipe], output, steps,
                          inputs=check_inputs(recipe, acceptance=True), overrides=overrides)


def run_acceptance(recipe, entry, output, steps, *, inputs, prepared_config=None, progress=None, overrides=()):
    require_draft_environment()
    if steps < 2 or steps % 2:
        raise ValueError("--steps must be a positive even number >= 2")
    if entry["nodes"] == 1:
        from omegaconf import OmegaConf

        config = prepared_config if prepared_config is not None else recipe_config(entry["configs"][0])
        if overrides:
            config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
        expected_gpus = entry["draft_gpus"]
        if OmegaConf.select(config, "inference.target_service.mode", default="external") == "managed-local":
            expected_gpus = entry["gpus"]
        devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if (
            len(devices) != expected_gpus
            or any(not d.strip() for d in devices)
            or len(set(devices)) != len(devices)
        ):
            raise ValueError(
                f"Select exactly {expected_gpus} Draft GPUs with CUDA_VISIBLE_DEVICES "
                "(include Target GPUs only for managed-local)"
            )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if progress is not None:
        progress("Preparing the run directory")
    if prepared_config is not None:
        from omegaconf import OmegaConf

        config_path = output / "quickstart-profile.yaml"
        OmegaConf.save(prepared_config, config_path)
        entry = {**entry, "configs": [str(config_path)]}
    if entry["nodes"] > 1:
        os.environ["ASPEC_OUTPUT_DIR"] = str(output)
        os.environ["ASPEC_CACHE_DIR"] = str(output / "cache")
        os.environ.setdefault(
            "ASPEC_DRAFT_CONFIG", str(asset_path("aspec/config/dspark_draft_config.json"))
        )
    memory_monitor = None
    memory_stream = None
    result = {
        "recipe": recipe,
        "scope": "bounded training/resume/native export",
        "status": "running",
        "inputs": inputs,
        "steps": steps,
        "catalog": entry,
        "launcher_environment": {key: os.environ.get(key) for key in entry.get("environment", {})},
        "stages": [],
        "additional_gates": [
            "held-out quality",
            "serving decode",
            "fault injection",
            "CP numerical parity where applicable",
        ],
    }
    announce = progress or (lambda message: None)
    try:
        announce("Checking GPU environment")
        if recipe == "qwen27b-long":
            from aspec.commands._recipe_capacity import memory_gpu_ids

            memory_devices = memory_gpu_ids(config, entry)
            memory_stream = (output / "gpu-memory.csv").open("w")
            memory_monitor = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--id=" + ",".join(map(str, memory_devices)),
                    "--query-gpu=timestamp,index,uuid,memory.used,memory.total",
                    "--format=csv",
                    "--loop=1",
                ],
                stdout=memory_stream,
                stderr=subprocess.STDOUT,
            )
        execute([sys.executable, "-m", "aspec.commands.doctor", "--gpu"], output / "doctor.log")
        for index, config in enumerate(entry["configs"]):
            stage = output / f"stage-{index + 1}"
            stage.mkdir()
            if not Path(config).is_absolute():
                from omegaconf import OmegaConf

                path = stage / "recipe.yaml"
                OmegaConf.save(recipe_config(config), path)
                config = str(path)
            if index:
                os.environ["ASPEC_INITIAL_DRAFT"] = str(output / f"stage-{index}" / "asset")
            base = [
                f"training.lr_total_steps={steps}",
                "training.num_epochs=100",
                "training.draft_accumulation_steps=1",
                f"training.save_interval={steps // 2}",
                "training.save_final_checkpoint=true",
                "logging.report_to=none",
                "logging.use_tensorboard=false",
                "logging.use_wandb=false",
                f"dataset.eval_interval={steps // 2}",
                f"cache_dir={output / 'cache'}",
                *overrides,
            ]
            announce("Inspecting the execution Plan")
            execute(
                [
                    sys.executable,
                    "-m",
                    "aspec.train_entry",
                    "--config",
                    config,
                    "--dry-run",
                    *base,
                ],
                stage / "plan.log",
            )
            if entry["nodes"] > 1:
                address = os.environ.get("RAY_ADDRESS")
                if not address or address in {"auto", "local"}:
                    raise ValueError(
                        "Multi-node acceptance requires explicit RAY_ADDRESS for an owned cluster"
                    )
                execute(
                    [
                        sys.executable,
                        "-m",
                        "aspec.train_entry",
                        "--config",
                        config,
                        "--preflight-only",
                        *base,
                    ],
                    stage / "preflight.log",
                )
            for label, limit in (("first", steps // 2), ("resumed", steps)):
                run = stage / label
                stage_overrides = [*base, f"training.num_train_steps={limit}"]
                if label == "resumed":
                    if recipe == "shared-target":
                        from omegaconf import OmegaConf

                        first = OmegaConf.load(stage / "first" / "config.yaml")
                        experiments = OmegaConf.to_container(first.experiments, resolve=True)
                        for experiment in experiments:
                            experiment.setdefault("training", {})["load_path"] = str(
                                stage / "first" / experiment["name"] / "checkpoints"
                            )
                        stage_overrides.append("experiments=" + json.dumps(experiments))
                    else:
                        stage_overrides.append(f"training.load_path={stage / 'first' / 'checkpoints'}")
                command = [
                    sys.executable,
                    "-m",
                    "aspec.commands._recipe_smoke",
                    "--config",
                    config,
                    "--output",
                    run,
                    "--gpus",
                    str(entry["gpus"]),
                ]
                if entry["nodes"] > 1:
                    command += ["--cluster-address", os.environ["RAY_ADDRESS"]]
                announce(f"Training the first {limit} steps" if label == "first"
                         else f"Restoring the checkpoint and continuing to step {limit}")
                execute([*command, *stage_overrides], stage / f"{label}.log")
                children = audit_run(
                    run,
                    limit,
                    label == "resumed",
                    require_eval=recipe != "quickstart",
                    expected_ranks=entry["draft_gpus"] // (2 if recipe == "shared-target" else 1),
                )
                if recipe == "qwen27b-long":
                    from aspec.commands._recipe_capacity import audit_long_run

                    audit_long_run(run)
            announce("Exporting the Draft and checking strict loading")
            for child in children:
                export_asset(
                    stage / "resumed" / child / "checkpoints" / f"iter_{steps + 1:07d}",
                    stage / (f"asset-{child}" if child else "asset"),
                    log_prefix=(stage / (f"asset-{child}" if child else "asset"))
                    if prepared_config is not None else None,
                )
            result["stages"].append({"config": config, "status": "passed"})
        if memory_monitor is not None:
            stop_memory_monitor(memory_monitor, memory_stream)
            memory_monitor = None
            from aspec.commands._recipe_capacity import summarize_memory

            result["capacity"] = summarize_memory(output / "gpu-memory.csv", expected_gpus=entry["gpus"])
        result["status"] = "passed"
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = str(error)
        raise
    finally:
        if memory_monitor is not None:
            stop_memory_monitor(memory_monitor, memory_stream)
        (output / "acceptance.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="List, copy, plan and run bundled training recipes.")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list")
    copy = sub.add_parser("copy", help="Copy editable YAML templates to a new directory")
    copy.add_argument("recipe", choices=catalog())
    copy.add_argument("--output", type=Path, required=True)
    export = sub.add_parser("export")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--input", required=True)
    prep.add_argument("--output", required=True)
    for action in ("plan", "train", "accept"):
        command = sub.add_parser(action)
        command.add_argument("recipe", choices=catalog())
        command.add_argument("--output", required=action != "plan")
        command.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        if action == "accept":
            command.add_argument("--steps", type=int, default=8)
        else:
            command.add_argument("--stage", type=int, default=1)
    args = parser.parse_args()
    if args.action in {"train", "accept"}:
        require_draft_environment()
    if args.action == "list":
        for name, entry in catalog().items():
            print(f"{name:16} {entry['model']:18} {entry['nodes']} node(s), {entry['gpus']} GPUs")
        return
    if args.action == "export":
        export_asset(Path(args.checkpoint).resolve(), Path(args.output).resolve())
        return
    if args.action == "prepare":
        print(json.dumps(prepare(args.input, args.output), indent=2))
        return
    entry = catalog()[args.recipe]
    for key, value in entry.get("environment", {}).items():
        os.environ.setdefault(key, value)
    if args.action == "copy":
        args.output.mkdir(parents=True, exist_ok=False)
        for relative in entry["configs"]:
            path = args.output / Path(relative).name
            OmegaConf.save(recipe_config(relative), path)
            print(path.resolve())
        return
    if args.action == "accept":
        acceptance(args.recipe, args.output, args.steps, args.set)
        return
    if not 1 <= args.stage <= len(entry["configs"]):
        parser.error("Invalid stage for recipe")
    if args.action == "train":
        check_inputs(args.recipe)
        if Path(args.output).exists():
            parser.error("Use a fresh output directory")
    if entry["nodes"] > 1 and args.output:
        os.environ["ASPEC_OUTPUT_DIR"] = str(Path(args.output).resolve())
        os.environ.setdefault("ASPEC_DRAFT_CONFIG", str(asset_path("aspec/config/dspark_draft_config.json")))
    options = list(args.set)
    if args.output:
        options.append(f"output_dir={Path(args.output).resolve()}")
    if args.action == "plan":
        options.append("--dry-run")
    elif entry["nodes"] == 1:
        os.environ["RAY_ADDRESS"] = "local"
    with tempfile.TemporaryDirectory(prefix="aspec-recipe-") as directory:
        path = Path(directory) / "recipe.yaml"
        OmegaConf.save(recipe_config(entry["configs"][args.stage - 1]), path)
        execute([sys.executable, "-m", "aspec.train_entry", "--config", path, *options])


if __name__ == "__main__":
    main()
