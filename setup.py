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

"""Collect canonical user assets at build time; never patch an installed runtime."""

import json
import shutil
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

ROOT = Path(__file__).resolve().parent


def build_identity():
    existing = ROOT / "aspec/_build_info.json"
    if existing.is_file():
        return existing.read_text()
    result = {"revision": None, "dirty": None}
    try:
        tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch",
                                  "--", "pyproject.toml"], capture_output=True, check=True)
        if tracked.stdout:
            result["revision"] = subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
            result["dirty"] = bool(subprocess.check_output(
                ["git", "-C", str(ROOT), "status", "--porcelain=v1"], text=True).strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return json.dumps(result, sort_keys=True) + "\n"


class BuildPackage(build_py):
    def run(self):
        super().run()
        package = Path(self.build_lib) / "aspec"
        assets = [*sorted((ROOT / "configs/recipes").glob("*.yaml")),
                  ROOT / "examples/data/sample_conversations.jsonl",
                  ROOT / "tools/recipe_catalog.json",
                  ROOT / "requirements/sglang-v0.5.21.txt",
                  ROOT / "requirements/draft.txt"]
        for source in assets:
            destination = package / "resources" / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        (package / "_build_info.json").write_text(build_identity())


class SourcePackage(sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        (Path(base_dir) / "aspec/_build_info.json").write_text(build_identity())


setup(cmdclass={"build_py": BuildPackage, "sdist": SourcePackage})
