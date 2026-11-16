# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Bootstrap the optional managed-local Target with its own native libraries."""
import os
import sys
from pathlib import Path

from aspec._environment import execute


def main():
    os.environ["VIRTUAL_ENV"] = sys.prefix
    automatic_toolkit = os.environ.pop("ASPEC_AUTO_CUDA_HOME", None)
    if automatic_toolkit and os.environ.get("CUDA_HOME") == automatic_toolkit:
        os.environ.pop("CUDA_HOME", None)
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    if cache := os.environ.get("ASPEC_CACHE_HOME"):
        if Path(cache).name != "target":
            os.environ["ASPEC_CACHE_HOME"] = str(Path(cache) / "target")
    # The supervisor belongs to Draft. Never load its CUDA libraries in Target.
    os.environ.pop("LD_LIBRARY_PATH", None)
    os.environ.pop("LIBRARY_PATH", None)
    os.environ.pop("PYTHONPATH", None)
    os.environ.pop("PYTHONHOME", None)
    execute([sys.executable, *sys.argv[1:]])


if __name__ == "__main__":
    main()
