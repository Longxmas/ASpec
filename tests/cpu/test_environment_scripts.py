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

"""Subprocess contracts for the public environment wrapper; no GPU or install required."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "tools/run_in_env.sh"


def environment():
    env = dict(os.environ, VIRTUAL_ENV=sys.prefix)
    for name in ("PYTHONPATH", "PYTHONHOME", "ASPEC_CUDA_COMPAT_DIR", "ASPEC_CACHE_HOME"):
        env.pop(name, None)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    return env


def test_environment_wrapper_preserves_arguments_exit_status_and_cwd(tmp_path):
    env = environment()
    value = "argument with spaces; $(never executed)"
    result = subprocess.run(["bash", str(WRAPPER), sys.executable, "-c",
                             "import os,sys; print(os.getcwd()); print(sys.argv[1]); sys.exit(7)",
                             value], env=env, cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 7, result.stderr
    assert result.stdout.splitlines() == [str(tmp_path), value]


@pytest.mark.parametrize("name,value", [("PYTHONPATH", "/foreign/imports"),
                                        ("VIRTUAL_ENV", "/another/venv"),
                                        ("ASPEC_CUDA_COMPAT_DIR", "/missing/compat")])
def test_environment_wrapper_rejects_invalid_inputs_before_command(name, value):
    env = environment()
    env[name] = value
    result = subprocess.run(["bash", str(WRAPPER), "echo", "command-executed"],
                            env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert "command-executed" not in result.stdout


def test_environment_wrapper_sets_absolute_cache_paths_without_empty_loader_entry(tmp_path):
    env = environment()
    env["ASPEC_CACHE_HOME"] = "cache with spaces"
    env.pop("LD_LIBRARY_PATH", None)
    result = subprocess.run(["bash", str(WRAPPER), sys.executable, "-c",
                             "import json,os; print(json.dumps(dict(os.environ)))"],
                            env=env, cwd=tmp_path, text=True, capture_output=True, check=True)
    actual = json.loads(result.stdout)
    cache = str(tmp_path / "cache with spaces")
    assert actual["XDG_CACHE_HOME"] == cache
    assert actual["TRITON_CACHE_DIR"] == cache + "/triton"
    assert "" not in actual.get("LD_LIBRARY_PATH", "absent").split(":")


def test_installer_rejects_unknown_options_without_installing():
    result = subprocess.run(["bash", str(ROOT / "tools/install.sh"), "--not-supported"],
                            text=True, capture_output=True)
    assert result.returncode == 2
    assert "Unknown argument" in result.stderr


def test_pip_cuda_toolkit_links_like_a_development_install_without_changing_wheels(tmp_path, monkeypatch):
    from aspec import _environment

    purelib = tmp_path / "site-packages"
    toolkit = purelib / "nvidia/cu13"
    (toolkit / "bin").mkdir(parents=True)
    (toolkit / "bin/nvcc").write_text("compiler fixture")
    (toolkit / "lib").mkdir()
    library = toolkit / "lib/libcudart.so.13"
    library.write_bytes(b"runtime fixture")
    monkeypatch.setattr(_environment.sysconfig, "get_paths", lambda: {"purelib": str(purelib)})
    monkeypatch.setenv("VIRTUAL_ENV", sys.prefix)
    monkeypatch.setenv("ASPEC_CACHE_HOME", str(tmp_path / "cache"))
    for key in ("CUDA_HOME", "CUDA_PATH", "PYTHONPATH", "PYTHONHOME", "ASPEC_CUDA_COMPAT_DIR"):
        monkeypatch.delenv(key, raising=False)
    environment = _environment.command_environment()
    view = Path(environment["CUDA_HOME"])
    assert (view / "lib64/libcudart.so").resolve() == library
    assert (view / "bin/nvcc").resolve() == toolkit / "bin/nvcc"
    assert environment["LIBRARY_PATH"].split(":")[0] == str(view / "lib64")
    assert not (toolkit / "lib64").exists()
    assert not (toolkit / "lib/libcudart.so").exists()
    assert library.read_bytes() == b"runtime fixture"
    assert _environment.command_environment()["CUDA_HOME"] == str(view)


@pytest.mark.parametrize("version", [(3, 10), (3, 11), (3, 12), (3, 13), (3, 14)])
def test_environment_wrapper_enforces_only_the_python_minimum(monkeypatch, version):
    from aspec import _environment

    for name in ("PYTHONPATH", "PYTHONHOME", "ASPEC_CUDA_COMPAT_DIR", "ASPEC_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VIRTUAL_ENV", sys.prefix)
    monkeypatch.setattr(_environment.sys, "version_info", (*version, *sys.version_info[2:]))
    if version < (3, 11):
        with pytest.raises(ValueError, match="Python >=3.11"):
            _environment.command_environment()
    else:
        assert _environment.command_environment()["VIRTUAL_ENV"] == sys.prefix
