# Copyright 2026 Ant Group Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check an installed wheel in an isolated CPU fixture, outside every source checkout.

Requires the CPU CI dependencies and official SGLang wheel (without its GPU
dependencies). Does not download models, start Ray or exercise training/GPUs.
"""

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def check_install():
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Run the wheel check in a dedicated venv")
    env = dict(os.environ, VIRTUAL_ENV=sys.prefix, CUDA_VISIBLE_DEVICES="",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for name in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
    cli = str(Path(sys.prefix) / "bin/aspec")
    with tempfile.TemporaryDirectory(prefix="aspec-wheel-check-") as directory:
        work = Path(directory)

        def run(arguments, expected=0, contains=None):
            result = subprocess.run(arguments, cwd=work, env=env, capture_output=True, text=True)
            if result.returncode != expected or (contains and contains not in result.stdout + result.stderr):
                raise RuntimeError(f"{arguments}: exit {result.returncode}\n{result.stdout}\n{result.stderr}")
            print(f"PASS {arguments[1:]} (exit {expected})", flush=True)
            return result.stdout

        observed = json.loads(run([sys.executable, "-I", "-c", """
import importlib.metadata,importlib.util,json,sys
from pathlib import Path
import aspec
from aspec.assets import asset_path
from aspec.evidence import _source_identity
from aspec.compat.sglang_patch import PATCH
from aspec.commands.convert_to_hf import _aspec_version
prefix=Path(sys.prefix)
assert Path(aspec.__file__).is_relative_to(prefix),aspec.__file__
assert importlib.util.find_spec('tools') is None
assert PATCH.is_file() and PATCH.is_relative_to(prefix)
assert asset_path('requirements/sglang-v0.5.21.txt').is_relative_to(prefix)
identity=_source_identity()
assert identity['revision'] is None and identity['dirty'] is None
assert identity['source_file_count'] > 0
assert identity['version'] == importlib.metadata.version('ling-aspec-runtime')
assert _aspec_version() == identity['version']
assert identity['build'] is not None
spec=importlib.util.find_spec('sglang')
assert Path(spec.origin).is_relative_to(prefix)
print(json.dumps({'sglang':str(Path(spec.origin).parent),'identity':identity}))
"""]))
        run([cli, "--help"], contains="patch-sglang")
        for command in ("doctor", "target", "patch-sglang", "recipes", "quickstart", "export",
                        "stitch-mtp", "evaluate", "prepare-data", "prepare-target-data",
                        "select-answers", "convert-data"):
            run([cli, command, "--help"], contains="usage:")
        run([cli, "recipes", "list"], contains="qwen27b-long")
        copied = work / "recipe"
        run([cli, "recipes", "copy", "quickstart", "--output", str(copied)])
        config = copied / "quickstart_qwen3_8b.yaml"
        before = config.read_bytes()
        run([cli, "--config", str(config), "--dry-run"])
        for model in ("qwen3-8b", "ling-tiny"):
            run([cli, "quickstart", model, "--plan", "--output", str(work / model)])
        run([cli, "recipes", "copy", "quickstart", "--output", str(copied)], expected=1)
        if config.read_bytes() != before:
            raise RuntimeError("Recipe copy overwrote an existing file")

        doctor = json.loads(run([cli, "doctor", "--json"], expected=1))
        if doctor["ok"] or not doctor["checks"] or doctor["gpu_requested"]:
            raise RuntimeError("CPU fixture must not claim full runtime readiness")
        target_doctor = json.loads(run([cli, "doctor", "--target", "--json"], expected=1))
        if not target_doctor["target_requested"]:
            raise RuntimeError("Target doctor must require the Target runtime")
        package = Path(observed["sglang"])

        def hashes():
            return {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in package.rglob("*") if path.is_file() and "__pycache__" not in path.parts}

        clean = hashes()
        run([cli, "patch-sglang", "--check"], expected=1, contains="not applied")
        if hashes() != clean:
            raise RuntimeError("Unpatched --check modified SGLang")
        run([cli, "patch-sglang"], contains="Applied ASpec")
        patched = hashes()
        if patched == clean:
            raise RuntimeError("Patch application changed no files")
        run([cli, "patch-sglang"], contains="already present")
        run([cli, "patch-sglang", "--check"], contains="already present")
        if hashes() != patched:
            raise RuntimeError("Repeat apply/check modified SGLang")
        conflict = package / "srt/model_executor/hook_manager.py"
        original = conflict.read_bytes()
        try:
            conflict.write_bytes(original.replace(b"SUPPORTS_MODEL_HOOK_SETUP = True",
                                                  b"SUPPORTS_MODEL_HOOK_SETUP = 'conflict'"))
            unknown = hashes()
            run([cli, "patch-sglang"], expected=1, contains="match neither")
            if hashes() != unknown:
                raise RuntimeError("Conflicting files were overwritten")
        finally:
            conflict.write_bytes(original)
        distribution = importlib.metadata.distribution("sglang")
        metadata = next(path for path in distribution.files if path.name == "METADATA")
        metadata_path = Path(distribution.locate_file(metadata))
        original_metadata = metadata_path.read_bytes()
        if b"Version: 0.5.21\n" not in original_metadata:
            raise RuntimeError("Wheel fixture requires official sglang==0.5.21")
        try:
            metadata_path.write_bytes(original_metadata.replace(b"Version: 0.5.21\n", b"Version: 0.5.20\n", 1))
            run([cli, "patch-sglang"], expected=1, contains="Expected sglang==0.5.21, found 0.5.20")
            if hashes() != patched:
                raise RuntimeError("Wrong-version rejection modified SGLang")
        finally:
            metadata_path.write_bytes(original_metadata)
        run([cli, "patch-sglang", "--check"], contains="already present")
        print(json.dumps({"ok": True, "scope": "installed wheel CPU commands and SGLang patch; no GPU/training",
                          "identity": observed["identity"]}, indent=2))


if __name__ == "__main__":
    check_install()
