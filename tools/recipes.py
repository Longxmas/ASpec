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

"""Source recipe maintenance and fault/CP acceptance commands."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from aspec.commands.recipes import CATALOG as CATALOG
from aspec.commands.recipes import ROOT as ROOT
from aspec.commands.recipes import assert_report as assert_report
from aspec.commands.recipes import audit_run as audit_run
from aspec.commands.recipes import catalog as catalog
from aspec.commands.recipes import check_inputs as check_inputs
from aspec.commands.recipes import digest as digest
from aspec.commands.recipes import execute as execute
from aspec.commands.recipes import export_asset as export_asset
from aspec.commands.recipes import prepare as prepare
from aspec.commands.recipes import run_acceptance as run_acceptance
from aspec.commands.recipes import stop_memory_monitor as stop_memory_monitor


def acceptance(recipe, output, steps):
    return run_acceptance(recipe, catalog()[recipe], output, steps, inputs=check_inputs(recipe, acceptance=True))


def fault_check(recipe, action, output, overrides=()):
    from omegaconf import OmegaConf

    from aspec.assets import recipe_config
    from aspec.reporting import build_run_report

    entry = catalog()[recipe]
    check_inputs(recipe)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if entry["nodes"] > 1:
        os.environ["ASPEC_OUTPUT_DIR"] = str(output)
        os.environ["ASPEC_CACHE_DIR"] = str(output / "cache")
        os.environ.setdefault(
            "ASPEC_DRAFT_CONFIG", str(ROOT / "aspec/config/dspark_draft_config.json")
        )
    result = {
        "scope": "bounded trainer exception" if action == "fail" else "slow consumer",
        "recipe": recipe,
        "action": action,
        "status": "running",
    }
    try:
        if entry["nodes"] == 1:
            config = OmegaConf.merge(recipe_config(entry["configs"][0]), OmegaConf.from_dotlist(overrides))
            expected_gpus = entry["draft_gpus"]
            if OmegaConf.select(config, "inference.target_service.mode", default="external") == "managed-local":
                expected_gpus = entry["gpus"]
            devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if (
                len(devices) != expected_gpus
                or any(not d for d in devices)
                or len(set(devices)) != len(devices)
            ):
                raise ValueError(f"Select exactly {expected_gpus} role-allocated GPUs explicitly")
        overrides = [
            "training.num_train_steps=8",
            "training.lr_total_steps=8",
            "training.num_epochs=100",
            "training.draft_accumulation_steps=1",
            "training.save_interval=4",
            "training.save_final_checkpoint=true",
            "dataset.eval_interval=4",
            f"cache_dir={output / 'cache'}",
            *overrides,
        ]
        command = [
            sys.executable,
            "-m",
            "tools.validation.recipe_faults",
            "--action",
            action,
            "--config",
            entry["configs"][0],
            "--output",
            output / "run",
            "--gpus",
            str(entry["gpus"]),
        ]
        if recipe == "shared-target":
            command += ["--shared"]
        if entry["nodes"] > 1:
            address = os.environ.get("RAY_ADDRESS")
            if not address or address in {"auto", "local"}:
                raise ValueError("Use an explicit task-owned cluster address")
            command += ["--cluster-address", address]
            execute(
                [
                    sys.executable,
                    "-m",
                    "aspec.train_entry",
                    "--config",
                    entry["configs"][0],
                    "--preflight-only",
                    *overrides,
                ],
                output / "preflight.log",
            )
        failed = False
        try:
            execute([*command, *overrides], output / "fault.log")
        except subprocess.CalledProcessError:
            failed = True
        if failed != (action == "fail"):
            raise ValueError("Unexpected exit status for injected scenario")
        markers = list((output / "run").rglob("recipe-fault.json"))
        report_path = output / "run" / "evidence.json"
        if not report_path.is_file():
            raise ValueError("Injection was not observed; startup failure is not fault acceptance")
        report = build_run_report(output / "run")
        marker_observed = bool(markers) and all(
            json.loads(p.read_text())["action"] == action for p in markers
        )
        # A failing remote rank may never reach checkpoint/audit collection. Its
        # exception must still be present in the coordinator's validated Report.
        exception_observed = (
            action == "fail"
            and "Recipe acceptance: injected trainer exception" in json.dumps(report)
        )
        if not marker_observed and not exception_observed:
            raise ValueError("Injection was not observed; startup failure is not fault acceptance")
        if action == "slow":
            assert_report(report, 8)
        else:
            if report["state"] not in {"failed", "completed_with_failed_experiments"}:
                raise ValueError("Failure did not reach an explicit terminal state")
            if report.get("cleanup_errors") or report.get("read_owners_fenced") is False:
                raise ValueError("Failure cleanup/fencing failed")
            for child in report.get("experiments", {"single": report}).values():
                cleanup = child.get("cleanup", {})
                if cleanup.get("state") != "completed":
                    raise ValueError("Failure cleanup did not complete")
                lifecycle = cleanup.get("work_lifecycle", {})
                if lifecycle.get("resident_outputs") != 0 or lifecycle.get("resident_bytes") != 0:
                    raise ValueError("Terminal ownership was not proven empty")
        result.update(status="passed", report=report)
    except BaseException as error:
        result.update(status="failed", error=str(error))
        raise
    finally:
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list")
    sub.add_parser("contracts")
    fault = sub.add_parser("fault-check")
    fault.add_argument("recipe", choices=("shared-target", "multinode"))
    fault.add_argument("--action", dest="fault_action", choices=("fail", "slow"), default="fail")
    fault.add_argument("--output", required=True)
    fault.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    cp = sub.add_parser("cp-check")
    cp.add_argument("--output", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("--checkpoint", required=True)
    exp.add_argument("--output", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--input", required=True)
    prep.add_argument("--output", required=True)
    for action in ("plan", "train", "accept"):
        p = sub.add_parser(action)
        p.add_argument("recipe", choices=catalog())
        p.add_argument("--output", required=action != "plan")
        if action == "accept":
            p.add_argument("--steps", type=int, default=8)
        else:
            p.add_argument("--stage", type=int, default=1)
            p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    if getattr(args, "recipe", None):
        for key, value in catalog()[args.recipe].get("environment", {}).items():
            os.environ.setdefault(key, value)
    os.chdir(ROOT)
    if args.action == "list":
        for name, item in catalog().items():
            print(f"{name:16} {item['model']:18} {item['nodes']} node(s), {item['gpus']} GPUs")
    elif args.action == "contracts":
        execute(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/cpu/test_release_recipes.py",
                "tests/cpu/test_node_local_files.py",
                "tests/cpu/test_recipe_node_loss.py",
                "tests/cpu/test_owner_fence.py",
                "tests/cpu/test_recipe_capacity.py",
                "tests/cpu/test_loop_cleanup.py",
                "tests/cpu/test_stage_recipe_inputs.py",
                "tests/test_packing.py",
                "tests/test_direct_cp.py",
                "tests/test_eagle3_trainer.py",
                "tests/cpu/test_eagle4.py",
                "tests/test_native_mtp.py",
                "tests/cpu/test_dmtp.py",
                "tests/cpu/test_training_resume_rng.py",
                "tests/cpu/test_evaluation_contract.py",
                "tests/test_ling_moe.py",
                "tests/test_multi_draft_plan.py",
                "tests/test_multinode_tcp_reference.py",
            ]
        )
    elif args.action == "fault-check":
        fault_check(args.recipe, args.fault_action, args.output, overrides=args.set)
    elif args.action == "cp-check":
        import xml.etree.ElementTree as ET

        devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if len(devices) != 2 or any(not d.strip() for d in devices) or len(set(devices)) != 2:
            parser.error("CP check requires exactly two explicitly allocated GPUs")
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=False)
        os.environ["ASPEC_RUN_GPU_INTEGRATION"] = "1"
        execute(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                "--module",
                "pytest",
                "-q",
                "tests/test_direct_cp_gpu.py",
                f"--junitxml={output / 'junit.xml'}",
            ],
            output / "cp.log",
        )
        suites = ET.parse(output / "junit.xml").getroot().iter("testsuite")
        counts = [dict(s.attrib) for s in suites]
        if not counts or any(
            int(c.get("skipped", 0)) or int(c.get("tests", 0)) == 0 for c in counts
        ):
            raise ValueError("Skipped/missing CP tests do not pass acceptance")
        (output / "result.json").write_text(
            json.dumps(
                {
                    "scope": "CP2 FA4 operator forward/backward parity",
                    "status": "passed",
                    "suites": counts,
                },
                indent=2,
            )
        )
    elif args.action == "export":
        export_asset(Path(args.checkpoint).resolve(), Path(args.output).resolve())
    elif args.action == "prepare":
        print(json.dumps(prepare(args.input, args.output), indent=2))
    elif args.action == "accept":
        acceptance(args.recipe, args.output, args.steps)
    else:
        entry = catalog()[args.recipe]
        if not 1 <= args.stage <= len(entry["configs"]):
            parser.error("Invalid stage for recipe")
        if args.action == "train":
            check_inputs(args.recipe)
            if Path(args.output).exists():
                parser.error("Use a fresh output directory")
        if entry["nodes"] > 1:
            if args.output:
                os.environ["ASPEC_OUTPUT_DIR"] = str(Path(args.output).resolve())
            os.environ.setdefault(
                "ASPEC_DRAFT_CONFIG", str(ROOT / "aspec/config/dspark_draft_config.json")
            )
        options = list(args.set)
        if args.output:
            options.append(f"output_dir={Path(args.output).resolve()}")
        if args.action == "plan":
            options.append("--dry-run")
        elif entry["nodes"] == 1:
            os.environ["RAY_ADDRESS"] = "local"
        execute(
            [
                sys.executable,
                "-m",
                "aspec.train_entry",
                "--config",
                entry["configs"][args.stage - 1],
                *options,
            ]
        )


if __name__ == "__main__":
    main()
