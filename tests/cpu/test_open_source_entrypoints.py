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

import importlib.metadata
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from aspec.compat import sglang_patch as patcher
from tools.check_docs import check_links


@pytest.fixture(params=[False, True], ids=["standalone", "inside_git"])
def wheel(tmp_path, request):
    if request.param:
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    site = tmp_path / "site"
    package = site / "sglang"
    package.mkdir(parents=True)
    file = package / "runtime.py"
    file.write_text("value = 1\n")
    patch = tmp_path / "training.patch"
    patch.write_text("diff --git a/python/sglang/runtime.py b/python/sglang/runtime.py\n"
                     "--- a/python/sglang/runtime.py\n+++ b/python/sglang/runtime.py\n"
                     "@@ -1 +1 @@\n-value = 1\n+value = 2\n")
    return site, file, patch


def test_patch_helper_is_idempotent_and_check_is_read_only(wheel, monkeypatch):
    site, file, patch = wheel
    original = patcher.patch_state
    monkeypatch.setattr(patcher, "PATCH", patch)
    monkeypatch.setattr(patcher, "patch_state", lambda root: original(root, patch))
    monkeypatch.setattr(patcher, "installed_site", lambda **_: site)
    monkeypatch.setattr(sys, "argv", ["patch", "--check"])
    with pytest.raises(SystemExit) as error:
        patcher.main()
    assert error.value.code == 1
    assert file.read_text() == "value = 1\n"
    monkeypatch.setattr(sys, "argv", ["patch"])
    patcher.main()
    patcher.main()
    assert file.read_text() == "value = 2\n"
    monkeypatch.setattr(sys, "argv", ["patch", "--check"])
    patcher.main()


def test_unknown_wheel_changes_are_not_overwritten(wheel):
    site, file, patch = wheel
    file.write_text("value = 99\n")
    with pytest.raises(RuntimeError, match="neither"):
        patcher.patch_state(site, patch)
    assert file.read_text() == "value = 99\n"


@pytest.mark.parametrize("check", [False, True])
def test_missing_target_runtime_never_triggers_dependency_install(monkeypatch, check):
    def missing(**kwargs):
        raise importlib.metadata.PackageNotFoundError("sglang")

    monkeypatch.setattr(patcher, "installed_site", missing)
    monkeypatch.setattr(patcher.subprocess, "run", lambda *a, **k: pytest.fail("must not install"))
    monkeypatch.setattr(sys, "argv", ["patch", *(["--check"] if check else [])])
    with pytest.raises(SystemExit) as error:
        patcher.main()
    assert error.value.code == 1


def test_patch_check_works_inside_an_unrelated_git_checkout(wheel):
    site, file, patch = wheel
    subprocess.run(["git", "init", "-q", str(site.parent)], check=True)
    assert patcher.patch_state(site, patch) == "unpatched"
    assert file.read_text() == "value = 1\n"


def test_wrong_sglang_version_rejected_before_locating_files(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setattr(patcher.importlib.metadata, "version", lambda _: "0.5.20")
    with pytest.raises(RuntimeError, match="Expected sglang"):
        patcher.installed_site(writable=True)


def test_shared_editable_sglang_is_not_modified(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv"))
    monkeypatch.setattr(patcher.importlib.metadata, "version", lambda _: "0.5.21")
    monkeypatch.setattr(patcher.importlib.util, "find_spec", lambda _: SimpleNamespace(origin=str(tmp_path / "source/sglang/__init__.py")))
    with pytest.raises(RuntimeError, match="outside the active venv"):
        patcher.installed_site(writable=True)
    assert patcher.installed_site() == tmp_path / "source"


def test_doctor_reports_missing_dependencies_without_importing_cuda(monkeypatch):
    from tools import doctor

    def missing(_):
        raise importlib.metadata.PackageNotFoundError("missing")

    monkeypatch.setattr(doctor.importlib.metadata, "version", missing)
    result = doctor.inspect_environment()
    assert not result["ok"]
    assert not result["gpu_requested"]
    assert result["checks"] and all(not item["ok"] for item in result["checks"])


def test_document_links_ignore_snapshots_and_fenced_examples(tmp_path):
    (tmp_path / "archive").mkdir(parents=True)
    (tmp_path / "archive/old.md").write_text("[old](missing.md)\n")
    (tmp_path / "README.md").write_text("[good](archive/old.md#section)\n```md\n[x](missing)\n```\n[x](docs/missing.md)\n")
    assert check_links(tmp_path) == [("README.md", 5, "docs/missing.md")]


def test_maintained_documentation_has_no_missing_local_targets():
    assert not check_links(Path(__file__).resolve().parents[2])


def test_quickstart_has_two_gpu_roles_and_fixed_resume_horizon():
    from aspec.config.train_config import load_config

    root = Path(__file__).resolve().parents[2]
    config = load_config(str(root / "configs/recipes/quickstart_qwen3_8b.yaml"))
    assert config.training.training_num_gpus_per_node == 1
    assert config.inference.inference_num_gpus == 1
    assert config.training.num_train_steps == config.training.lr_total_steps == 8
    assert config.training.fsdp_strategy == "REPLICATE"
    assert Path(config.dataset.train_data_path).is_file()
    assert config.mooncake.require_nvlink


def test_sdist_includes_runtime_install_requirements():
    root = Path(__file__).resolve().parents[2]
    assert "recursive-include requirements *.txt" in (root / "MANIFEST.in").read_text()


def test_report_rejects_preflight_instead_of_silently_ignoring_it(monkeypatch, capsys):
    from aspec import train_entry

    monkeypatch.setattr(sys, "argv", ["aspec", "--report", "unused", "--preflight-only"])
    monkeypatch.setattr(train_entry, "build_run_report", lambda _: pytest.fail("must reject conflicting modes first"))
    with pytest.raises(SystemExit) as error:
        train_entry.parse_config()
    assert error.value.code == 2
    assert "--report cannot be combined" in capsys.readouterr().err


@pytest.mark.parametrize("shared", ["true", " TrUe ", "false"])
def test_patch_helper_rejects_explicit_system_site_packages(tmp_path, monkeypatch, shared):
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    (tmp_path / "pyvenv.cfg").write_text(f"include-system-site-packages = {shared}\n")
    monkeypatch.setattr(patcher.importlib.metadata, "version", lambda _: patcher.VERSION)
    monkeypatch.setattr(patcher.importlib.util, "find_spec",
                        lambda _: SimpleNamespace(origin=str(tmp_path / "lib/sglang/__init__.py")))
    if shared.strip().lower() == "true":
        with pytest.raises(RuntimeError, match="include-system-site-packages=false"):
            patcher.installed_site(writable=True)
    else:
        assert patcher.installed_site(writable=True) == tmp_path / "lib"


def test_doctor_reports_system_site_packages_even_when_versions_match(tmp_path, monkeypatch):
    from tools import doctor

    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    (tmp_path / "pyvenv.cfg").write_text("include-system-site-packages = true\n")
    monkeypatch.setattr(doctor, "expected_versions", lambda target=False: {})
    monkeypatch.setattr(doctor, "installed_site", lambda: tmp_path)
    monkeypatch.setattr(doctor, "patch_state", lambda _: "patched")
    result = doctor.inspect_environment()
    assert not result["ok"]
    assert any("include-system-site-packages=false" in error for error in result["errors"])


def test_document_links_check_nested_public_tools_and_configs(tmp_path):
    for relative in ("docs/reference/current.md", "configs/benchmarks/README.md", "tools/README.md", "patches/README.md"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[missing](missing.md)\n")
    assert {row[0] for row in check_links(tmp_path)} == {
        "docs/reference/current.md", "configs/benchmarks/README.md", "tools/README.md", "patches/README.md",
    }
