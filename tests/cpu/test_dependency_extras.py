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

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHARED_CUTE_PACKAGES = ("nvidia-cutlass-dsl", "quack-kernels")


def _shared_cute_requirements(requirements: list[str]) -> set[str]:
    return {
        requirement
        for requirement in requirements
        if requirement.startswith(SHARED_CUTE_PACKAGES)
    }


def test_dspark_qk_uses_the_fa_cutlass_runtime_without_requiring_quack() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extras = project["project"]["optional-dependencies"]

    fa_runtime = _shared_cute_requirements(extras["fa"])
    dspark_qk_runtime = _shared_cute_requirements(extras["dspark-qk"])

    assert dspark_qk_runtime == {
        requirement
        for requirement in fa_runtime
        if requirement.startswith("nvidia-cutlass-dsl")
    }
    assert sum(
        requirement.startswith("nvidia-cutlass-dsl")
        for requirement in dspark_qk_runtime
    ) == 1
    assert any(requirement.startswith("quack-kernels") for requirement in fa_runtime)


def test_draft_and_target_dependencies_are_separate():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["name"] == "ling-aspec-runtime"
    core = project["dependencies"]
    assert not any(x.startswith(("sglang", "torch==", "nvidia-cutlass", "quack", "flash-attn")) for x in core)
    draft = (ROOT / "requirements/draft.txt").read_text()
    target = (ROOT / "requirements/sglang-v0.5.21.txt").read_text()
    assert "nvidia-cutlass-dsl[cu13]==4.7.1" in draft
    assert "sglang==0.5.21" in target
    assert "nvidia-cutlass" not in target and "quack-kernels" not in target
    assert not (ROOT / "requirements/cute-overrides.txt").exists()
