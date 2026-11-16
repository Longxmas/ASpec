# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Identity shared across the independently installed Draft and Target roles."""
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path


def release_version():
    """Public compatibility identity; neither Python nor source bytes are wire types."""
    return importlib.metadata.version("ling-aspec-runtime")


def code_identity():
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*")):
        if path.suffix in {".py", ".patch"}:
            digest.update(path.relative_to(package).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return {"version": release_version(), "code_sha256": digest.hexdigest(),
            "python_minor": list(sys.version_info[:2])}


def dependency_versions(*names):
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def inventory():
    """Read Target identity in its own interpreter, without starting GPU services."""
    from aspec.commands.doctor import inspect_environment
    from aspec.runtime_identity import SGLANG_RUNTIME_RELATIVE_PATHS, module_runtime_identity

    doctor = inspect_environment(target=True)
    if not doctor["ok"]:
        raise RuntimeError("Target environment is not ready: " + "; ".join(doctor["errors"]))
    return {**code_identity(), "python": sys.executable, "prefix": sys.prefix,
            "sglang": module_runtime_identity("sglang", relative_paths=SGLANG_RUNTIME_RELATIVE_PATHS),
            "dependencies": dependency_versions("torch", "ray", "nvidia-cutlass-dsl", "quack-kernels")}


if __name__ == "__main__":
    print(json.dumps(inventory()))
