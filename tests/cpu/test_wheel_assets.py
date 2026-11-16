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

import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from aspec.assets import asset_path, recipe_config
from aspec.commands import recipes
from aspec.config.train_config import load_config
from aspec.evidence import _package_versions, _source_identity
from aspec.planning import compile_execution_plan


def test_bundled_recipe_plans_from_an_arbitrary_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "config.yaml"
    OmegaConf.save(recipe_config("configs/recipes/quickstart_qwen3_8b.yaml"), path)
    config = load_config(str(path))
    compile_execution_plan(config)
    assert Path(config.model.draft_model_config).is_file()
    assert Path(config.dataset.train_data_path).is_file()
    assert Path(config.output_dir).is_relative_to(tmp_path)
    assert Path(config.cache_dir).is_relative_to(tmp_path)


def test_recipe_copy_is_editable_and_never_overwrites(tmp_path, monkeypatch):
    output = tmp_path / "recipes"
    monkeypatch.setattr(sys, "argv", ["recipes", "copy", "quickstart", "--output", str(output)])
    recipes.main()
    path = output / "quickstart_qwen3_8b.yaml"
    path.write_text(path.read_text() + "\n# User modification\n")
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        recipes.main()
    assert path.read_bytes() == original


def test_installed_command_runner_preserves_cwd_and_arguments(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log = tmp_path / "command.log"
    value = "spaces; $(literal)"
    recipes.execute([sys.executable, "-c", "import os,sys; print(os.getcwd()); print(sys.argv[1])", value], log)
    assert log.read_text().splitlines() == [str(tmp_path), value]


def test_package_identity_hashes_bundled_patch_and_does_not_borrow_parent_git(tmp_path, monkeypatch):
    versions = {"ling-aspec-runtime": "0.1.0b0", "aspec": "0.0.16"}

    def version(name):
        if name not in versions:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(importlib.metadata, "version", version)
    observed = _package_versions()
    assert observed["ling-aspec-runtime"] == "0.1.0b0"
    assert "aspec" not in observed
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    package = tmp_path / "venv/lib/site-packages/aspec"
    package.mkdir(parents=True)
    (package / "evidence.py").write_text("# Installed package\n")
    (package / "_build_info.json").write_text(json.dumps({"revision": "a" * 40, "dirty": False}))
    patch = package / "compat/patches/sglang.patch"
    patch.parent.mkdir(parents=True)
    patch.write_text("first patch\n")
    first = _source_identity(package)
    assert first["version"] == "0.1.0b0"
    assert first["revision"] is None and first["dirty"] is None
    assert first["build"] == {"revision": "a" * 40, "dirty": False}
    patch.write_text("second patch\n")
    assert _source_identity(package)["source_tree_sha256"] != first["source_tree_sha256"]


@pytest.mark.parametrize("relative", ["../outside", "/absolute"])
def test_asset_paths_cannot_escape_the_package(relative):
    with pytest.raises(ValueError):
        asset_path(relative)


def test_export_requires_metadata_even_with_optimized_python():
    from aspec.commands.convert_to_hf import _EmptyStateDictLoadPlanner

    with pytest.raises(ValueError, match="metadata is required"):
        _EmptyStateDictLoadPlanner().set_up_planner({}, metadata=None)
