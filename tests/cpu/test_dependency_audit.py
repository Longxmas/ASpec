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

from types import SimpleNamespace

import pytest

from tools.doctor import check_dependency_metadata


def distribution(name, version, requires=()):
    return SimpleNamespace(metadata={"Name": name}, version=version, requires=requires)


def test_previous_overrides_are_now_rejected():
    result = check_dependency_metadata([
        distribution("sglang", "0.5.21", ["nvidia-cutlass-dsl[cu13]==4.6.2", "quack-kernels==0.6.4"]),
        distribution("nvidia-cutlass-dsl", "4.7.1"), distribution("quack-kernels", "0.6.5"),
    ])
    assert len(result["conflicts"]) == 2
    assert not result["overrides"]


@pytest.mark.parametrize("owner,version,requires,installed", [
    ("sglang", "0.5.22", "quack-kernels==0.6.4", "0.6.5"),
    ("sglang", "0.5.21", "quack-kernels==0.6.3", "0.6.5"),
    ("sglang", "0.5.21", "quack-kernels==0.6.4", "0.6.6"),
    ("another-package", "0.5.21", "quack-kernels==0.6.4", "0.6.5"),
])
def test_all_dependency_conflicts_are_rejected(owner, version, requires, installed):
    result = check_dependency_metadata([
        distribution(owner, version, [requires]), distribution("quack-kernels", installed),
    ])
    assert len(result["conflicts"]) == 1
    assert not result["overrides"]


def test_missing_dependency_fails_and_inactive_extra_is_ignored():
    result = check_dependency_metadata([
        distribution("example", "1.0", ["missing>=1", 'optional; extra == "unused"']),
    ])
    assert len(result["conflicts"]) == 1
    assert "missing" in result["conflicts"][0]


@pytest.mark.parametrize("target", [False, True])
def test_doctor_requires_sglang_only_for_target_nodes(monkeypatch, target):
    from tools import doctor

    monkeypatch.setattr(doctor, "check_dependency_metadata", lambda: {"conflicts": [], "overrides": []})
    monkeypatch.setattr(doctor, "venv_error", lambda: None)
    monkeypatch.setattr(doctor, "expected_versions", lambda target=False: {"sglang": "0.5.21"})

    def missing(_=None):
        raise doctor.importlib.metadata.PackageNotFoundError("sglang")

    monkeypatch.setattr(doctor.importlib.metadata, "version", missing)
    monkeypatch.setattr(doctor, "installed_site", missing)
    result = doctor.inspect_environment(target=target)
    assert result["ok"] == (not target)
    assert result["patch"] == ("unavailable" if target else "not_required")
    assert result["target_requested"] == target


@pytest.mark.parametrize("version", [(3, 10), (3, 11), (3, 12), (3, 13), (3, 14)])
def test_doctor_enforces_only_the_python_minimum(monkeypatch, tmp_path, version):
    from tools import doctor

    monkeypatch.setattr(doctor, "check_dependency_metadata", lambda: {"conflicts": [], "overrides": []})
    monkeypatch.setattr(doctor, "venv_error", lambda: None)
    monkeypatch.setattr(doctor, "expected_versions", lambda target=False: {})
    monkeypatch.setattr(doctor, "installed_site", lambda: tmp_path)
    monkeypatch.setattr(doctor, "patch_state", lambda _: "patched")
    monkeypatch.setattr(doctor.sys, "version_info", (*version, *doctor.sys.version_info[2:]))
    result = doctor.inspect_environment()
    assert result["ok"] == (version >= (3, 11))
    assert result["errors"] == ([] if version >= (3, 11) else ["Python >=3.11 is required"])
