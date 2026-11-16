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

"""Apply/check the training patch in the active Target environment."""

import argparse
import importlib.metadata
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

VERSION = "0.5.21"
PATCH = Path(__file__).with_name("patches") / "sglang-v0.5.21.patch"


def _patch_environment(site):
    # --no-index still honors a containing worktree's subdirectory filter.
    # A project-local .venv must be patched relative to site-packages itself.
    env = os.environ.copy()
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE"):
        env.pop(name, None)
    env["GIT_CEILING_DIRECTORIES"] = str(Path(site).resolve().parent)
    return env


def patch_state(site, patch=PATCH):
    """Never partially apply a patch to an unknown/modified installation."""
    if not patch.is_file():
        raise RuntimeError("Training patch is missing; reinstall the ASpec wheel or source distribution")
    command = ["git", "apply", "--no-index", "-p2", "--check"]
    env = _patch_environment(site)
    forward = subprocess.run([*command, str(patch)], cwd=site, env=env, capture_output=True, text=True)
    if forward.returncode == 0:
        return "unpatched"
    backward = subprocess.run([*command, "--reverse", str(patch)], cwd=site, env=env, capture_output=True, text=True)
    if backward.returncode == 0:
        return "patched"
    raise RuntimeError("SGLang files match neither the clean wheel nor this complete patch; "
                       "use a fresh venv rather than overwriting unknown changes. " + forward.stderr[-500:])


def venv_error():
    """Diagnose explicit non-isolation without importing installed packages."""
    if sys.prefix == sys.base_prefix:
        return "Activate a dedicated venv; system installations are not modified"
    config = Path(sys.prefix) / "pyvenv.cfg"
    try:
        lines = config.read_text().splitlines()
    except FileNotFoundError:
        # Keep support for isolated prefix layouts that do not use pyvenv.cfg.
        return None
    except OSError as exc:
        return f"Cannot read venv configuration: {exc}"
    for line in lines:
        key, _, value = line.partition("=")
        if (key.strip().casefold() == "include-system-site-packages"
                and value.strip().casefold() == "true"):
            return "Use a dedicated venv with include-system-site-packages=false"
    return None


def installed_site(*, writable=False):
    if error := venv_error():
        raise RuntimeError(error)
    actual = importlib.metadata.version("sglang")
    if actual != VERSION:
        raise RuntimeError(f"Expected sglang=={VERSION}, found {actual}; install the pinned requirements")
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.origin:
        raise RuntimeError("Cannot locate the installed SGLang package")
    package = Path(spec.origin).resolve().parent
    if writable and not package.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("SGLang is outside the active venv (editable/shared source); "
                           "this helper only patches an isolated wheel installation")
    return package.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Require the complete patch; do not change files")
    args = parser.parse_args()
    try:
        site = installed_site(writable=not args.check)
        state = patch_state(site)
        if state == "unpatched":
            if args.check:
                raise RuntimeError("Training patch is not applied; run aspec patch-sglang")
            subprocess.run(["git", "apply", "--no-index", "-p2", str(PATCH)], cwd=site, env=_patch_environment(site), check=True)
            if patch_state(site) != "patched":
                raise RuntimeError("Post-apply verification failed")
            print(f"Applied ASpec training patch to {site / 'sglang'}")
        else:
            print(f"ASpec training patch already present: {site / 'sglang'}")
    except importlib.metadata.PackageNotFoundError:
        parser.exit(1, "Activate the Target venv and install ling-aspec-target before patching SGLang\n")
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"ASpec patch check failed: {exc}\n")


if __name__ == "__main__":
    main()
