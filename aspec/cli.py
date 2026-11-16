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

"""Workflow shortcuts and native-library bootstrap for the public CLI."""
import sys

from aspec._environment import execute, is_target_environment


def main():
    arguments = sys.argv[1:]
    modules = {
        "train": "aspec.train_entry",
        "target": "aspec.inference.service",
        "quickstart": "aspec.commands.quickstart",
        "recipes": "aspec.commands.recipes",
        "doctor": "aspec.commands.doctor",
        "patch-sglang": "aspec.compat.sglang_patch",
        "export": "aspec.commands.convert_to_hf",
        "stitch-mtp": "aspec.commands.stitch_native_mtp",
        "evaluate": "aspec.commands.evaluate_draft",
        "prepare-data": "aspec.commands.prepare_quality_dataset",
        "prepare-target-data": "aspec.commands.prepare_target_dataset",
        "select-answers": "aspec.commands.prepare_selected_answers",
        "convert-data": "aspec.commands.convert_pretokenized_jsonl",
    }
    if not arguments or arguments in (["--help"], ["-h"]):
        print("usage: aspec COMMAND [OPTIONS]\n"
              "       aspec --config PATH [KEY=VALUE ...]\n"
              "       aspec --report RUN_DIRECTORY [--report-format json]\n\n"
              "Commands:\n  " + "\n  ".join(modules) +
              "\n\nUse aspec COMMAND --help for command options. "
              "Use python -m aspec.train_entry --help for training options.")
        return
    if arguments and arguments[0] in modules:
        module = modules[arguments.pop(0)]
    else:
        module = "aspec.train_entry"
    help_requested = arguments in (["--help"], ["-h"])
    if module == "aspec.inference.service" and not help_requested and not is_target_environment():
        raise SystemExit("Start aspec target in the Target environment (pip install ling-aspec-target)")
    try:
        execute([sys.executable, "-m", module, *arguments])
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
