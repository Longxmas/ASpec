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

"""Prepare this venv's native libraries before executing a fresh process."""
import hashlib
import importlib.metadata
import os
import sys
import sysconfig
from pathlib import Path


def is_target_environment():
    marker = Path(sys.prefix) / ".aspec-role"
    if marker.is_file() and marker.read_text().strip() == "target":
        return True
    try:
        importlib.metadata.version("ling-aspec-target")
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def require_draft_environment():
    if is_target_environment():
        raise SystemExit("Run training from the Draft venv (pip install ling-aspec); "
                         "this interpreter belongs to Target")


def _toolkit_view(toolkit, env):
    """Expose pip's CUDA layout to JITs expecting a development toolkit.

    NVIDIA wheels ship lib/libcudart.so.13, while SGLang links -lcudart from
    CUDA_HOME/lib64. Keep the development aliases in ASpec's private cache;
    the installed wheels and host toolkit remain untouched.
    """
    cache = Path(env.get("ASPEC_CACHE_HOME") or
                 Path(env.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "aspec")
    identity = hashlib.sha256(str(toolkit).encode()).hexdigest()[:16]
    view = cache.expanduser().resolve() / "cuda-toolkit" / identity
    libraries = view / "lib64"
    libraries.mkdir(parents=True, exist_ok=True)

    def link(destination, source):
        try:
            destination.symlink_to(source, target_is_directory=source.is_dir())
        except FileExistsError:
            if not destination.is_symlink() or destination.readlink() != source:
                raise ValueError(f"Unexpected file in private CUDA toolkit view: {destination}")

    for source in toolkit.iterdir():
        if source.name != "lib64":
            link(view / source.name, source)
    for source in sorted((toolkit / "lib").iterdir(), key=lambda p: (len(p.name), p.name)):
        link(libraries / source.name, source)
        if ".so." in source.name:
            alias = libraries / (source.name.split(".so.", 1)[0] + ".so")
            if not alias.is_symlink():
                link(alias, source)
    return view


def command_environment():
    if sys.platform != "linux" or sys.version_info[:2] < (3, 11):
        raise ValueError("Use Linux with Python >=3.11.")
    active = os.environ.get("VIRTUAL_ENV")
    if not active or sys.prefix == sys.base_prefix or Path(active).resolve() != Path(sys.prefix).resolve():
        raise ValueError("Activate the intended venv; python and VIRTUAL_ENV must agree.")
    if os.environ.get("PYTHONPATH") or os.environ.get("PYTHONHOME"):
        raise ValueError("Unset PYTHONPATH and PYTHONHOME; use the active isolated venv.")
    settings = dict(line.lower().replace(" ", "").split("=", 1)
                    for line in (Path(sys.prefix) / "pyvenv.cfg").read_text().splitlines()
                    if "=" in line)
    if settings.get("include-system-site-packages") != "false":
        raise ValueError("Use include-system-site-packages=false.")
    env = dict(os.environ)
    paths = []
    if directory := env.get("ASPEC_CUDA_COMPAT_DIR"):
        path = Path(directory).resolve()
        if not (path / "libcuda.so.1").is_file():
            raise ValueError("ASPEC_CUDA_COMPAT_DIR must contain libcuda.so.1.")
        paths.append(str(path))
    purelib = Path(sysconfig.get_paths()["purelib"])
    toolkit = purelib / "nvidia/cu13"
    if not env.get("CUDA_HOME") and not env.get("CUDA_PATH") and (toolkit / "bin/nvcc").is_file():
        view = _toolkit_view(toolkit, env)
        env["CUDA_HOME"] = str(view)
        env["ASPEC_AUTO_CUDA_HOME"] = str(view)
        env["PATH"] = str(view / "bin") + os.pathsep + env.get("PATH", "")
        previous = env.get("LIBRARY_PATH", "")
        env["LIBRARY_PATH"] = str(view / "lib64") + (os.pathsep + previous if previous else "")
        paths.append(str(view / "lib64"))
    paths.extend(str(p) for p in (purelib / "nvidia/cuda_runtime/lib", purelib / "torch/lib")
                 if p.is_dir())
    if (toolkit / "lib").is_dir():
        paths.append(str(toolkit / "lib"))
    if paths:
        previous = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(paths + ([previous] if previous else []))
    if directory := env.get("ASPEC_CACHE_HOME"):
        root = Path(directory).resolve()
        root.mkdir(parents=True, exist_ok=True)
        for key, suffix in {
            "XDG_CACHE_HOME": "", "TRITON_CACHE_DIR": "triton",
            "TORCHINDUCTOR_CACHE_DIR": "torchinductor", "TORCH_EXTENSIONS_DIR": "torch-extensions",
            "CUDA_CACHE_PATH": "cuda", "CUTE_DSL_CACHE_DIR": "cute",
            "FLASHINFER_WORKSPACE_BASE": "flashinfer", "SGLANG_JIT_CACHE_DIR": "sglang",
        }.items():
            env[key] = str(root / suffix)
    return env


def execute(argv):
    if not argv:
        raise ValueError("Expected a command.")
    os.execvpe(argv[0], argv, command_environment())


if __name__ == "__main__":
    try:
        execute(sys.argv[1:])
    except (ValueError, OSError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(2) from error
