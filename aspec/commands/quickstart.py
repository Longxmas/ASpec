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

"""The same two-GPU train/resume/native-export workflow for Qwen and Ling."""
import argparse
import contextlib
import copy
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

from omegaconf import OmegaConf

from aspec.assets import asset_path, recipe_config
from aspec.commands.recipes import catalog, digest, execute, run_acceptance

MODELS = {"qwen3-8b": "Qwen/Qwen3-8B", "ling-tiny": "inclusionAI/Ling-3.0-tiny"}


def build_profile(name, target=None):
    config = recipe_config("configs/recipes/quickstart_qwen3_8b.yaml")
    if name == "ling-tiny":
        ling = recipe_config("configs/recipes/ling3_tiny_dspark.yaml")
        config.model = ling.model
        config.dataset.chat_template = ling.dataset.chat_template
        config.inference.target_prefill_memory_optimized = False
        config.inference.sglang.mem_fraction_static = ling.inference.sglang.mem_fraction_static
        config.mooncake.hidden_dim = ling.mooncake.hidden_dim
    elif name != "qwen3-8b":
        raise ValueError("Unknown quickstart model")
    config.model.target_model_path = target or MODELS[name]
    config.dataset.train_data_path = str(asset_path("examples/data/sample_conversations.jsonl"))
    config.output_dir = "outputs/quickstart-" + name
    config.cache_dir = "cache/quickstart-" + name
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--target", help="Optional compatible local Target directory or Hub ID")
    parser.add_argument("--output", required=True, type=Path, help="Fresh run directory")
    parser.add_argument("--target-address", help="Address of an already running Target service")
    parser.add_argument("--managed-target", help="Optionally start local Target using this absolute Python path")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--plan", action="store_true", help="Inspect CPU Plan only; no GPU services")
    args = parser.parse_args()
    config = build_profile(args.model, args.target)
    if args.target_address:
        config.inference.target_service = {"address": args.target_address}
    if args.managed_target:
        config.inference.target_service = OmegaConf.merge(
            config.inference.get("target_service", {}),
            {"mode": "managed-local", "python_executable": args.managed_target},
        )
    if args.set:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.set))
    output = args.output.resolve()
    if args.plan:
        with tempfile.TemporaryDirectory(prefix="aspec-plan-") as folder:
            path = Path(folder) / "quickstart.yaml"
            OmegaConf.save(config, path)
            execute([sys.executable, "-m", "aspec.train_entry", "--config", path, "--dry-run",
                     f"output_dir={output}"])
        return
    if output.exists() or output.is_symlink():
        parser.error("Use a fresh output directory; existing runs are never changed.")
    entry = copy.deepcopy(catalog()["quickstart"])
    entry.update(model=MODELS[args.model], profile=args.model)
    data = Path(config.dataset.train_data_path)
    inputs = {"bundled_conversations": {"path": str(data), "sha256": digest(data)}}
    terminal = sys.stdout
    created = False

    def announce(message):
        nonlocal created
        created = True
        print(message, file=terminal, flush=True)

    with tempfile.TemporaryFile(mode="w+") as log:
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                run_acceptance(
                    "quickstart", entry, output, args.steps, inputs=inputs, prepared_config=config,
                    progress=announce,
                )
        except Exception as error:
            traceback.print_exc(file=log)
            detail = str(error) if isinstance(error, ValueError) else type(error).__name__
            location = f" See {output}/quickstart.log and stage logs." if created and output.is_dir() else ""
            parser.exit(1, f"Quickstart failed: {detail}.{location}\n")
        finally:
            if created and output.is_dir():
                log.flush()
                log.seek(0)
                with (output / "quickstart.log").open("w") as saved:
                    shutil.copyfileobj(log, saved)
    print(json.dumps({"output": str(output), "profile": args.model,
                      "asset": str(output / "stage-1/asset"),
                      "scope": "synthetic functional train/resume/native export; not quality"}, indent=2))


if __name__ == "__main__":
    main()
