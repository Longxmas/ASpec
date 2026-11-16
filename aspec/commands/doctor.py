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

"""Read-only checks for the documented runtime; no model downloads or services."""

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from aspec.assets import asset_path
from aspec.compat.sglang_patch import installed_site, patch_state, venv_error


def check_dependency_metadata(distributions=None):
    """Audit installed active requirements; no metadata conflicts are exempted."""
    if distributions is None:
        distributions = importlib.metadata.distributions()
    installed = {canonicalize_name(dist.metadata["Name"]): dist for dist in distributions}
    result = {"overrides": [], "conflicts": []}
    for name, dist in sorted(installed.items()):
        for raw in dist.requires or ():
            try:
                requirement = Requirement(raw)
            except InvalidRequirement:
                result["conflicts"].append(f"{name}: invalid requirement {raw!r}")
                continue
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            dependency = installed.get(canonicalize_name(requirement.name))
            actual = dependency.version if dependency is not None else None
            if actual is not None and requirement.specifier.contains(actual, prereleases=True):
                continue
            result["conflicts"].append(
                f"{name} {dist.version} requires {requirement}; installed: {actual}"
            )
    return result


def expected_versions(target=False):
    source = asset_path("requirements/sglang-v0.5.21.txt" if target else "requirements/draft.txt")
    requirements = [Requirement(line) for line in source.read_text().splitlines()
                    if line.strip() and not line.startswith("#")]
    return {requirement.name: next((spec.version for spec in requirement.specifier
                                   if spec.operator == "=="), None)
            for requirement in requirements}


def inspect_environment(*, gpu=False, target=False):
    result = {"python": sys.executable, "prefix": sys.prefix, "platform": platform.platform(),
              "gpu_requested": gpu, "target_requested": target, "checks": [], "errors": []}
    result["dependencies"] = check_dependency_metadata()
    result["errors"].extend(result["dependencies"]["conflicts"])
    if error := venv_error():
        result["errors"].append(error)
    if sys.version_info[:2] < (3, 11):
        result["errors"].append("Python >=3.11 is required")
    if sys.platform != "linux":
        result["errors"].append("The GPU runtime requires Linux; macOS/Windows training is not validated")
    installed = {canonicalize_name(dist.metadata["Name"]) for dist in importlib.metadata.distributions()}
    marker = Path(sys.prefix) / ".aspec-role"
    target = target or "ling-aspec-target" in installed or (marker.is_file() and marker.read_text().strip() == "target")
    result["role"] = "target" if target else "draft"
    if {"ling-aspec", "ling-aspec-target"} <= installed:
        result["errors"].append("Install Draft and Target in separate venvs")
    has_sglang = False
    for name, expected in expected_versions(target).items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        if name == "sglang":
            has_sglang = actual is not None
            if not target and not has_sglang:
                continue
        ok = actual is not None and (expected is None or actual.split("+", 1)[0] == expected)
        result["checks"].append({"package": name, "expected": expected, "installed": actual, "ok": ok})
        if not ok:
            result["errors"].append(f"{name}: expected {expected}, found {actual}; use the pinned requirements")
    if not target:
        try:
            importlib.metadata.version("sglang")
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            result["errors"].append("SGLang belongs in the isolated Target environment")
    result["patch"] = "unavailable" if target or has_sglang else "not_required"
    if target or has_sglang:
        try:
            site = installed_site()
            result["sglang_path"] = str(site / "sglang")
            result["patch"] = patch_state(site)
            if result["patch"] != "patched":
                result["errors"].append("Apply the training patch: aspec patch-sglang")
        except (RuntimeError, OSError, importlib.metadata.PackageNotFoundError) as exc:
            if str(exc) not in result["errors"]:
                result["errors"].append(str(exc))
    if gpu:
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; check driver/CUDA compatibility and process library paths")
            value = torch.ones((16, 16), dtype=torch.bfloat16, device="cuda")
            product = value @ value
            torch.cuda.synchronize()
            if not torch.all(product == 16).item():
                raise RuntimeError("CUDA BF16 matmul failed")
            from mooncake.engine import TransferEngine

            result["cuda"] = {"torch": torch.__version__, "runtime": torch.version.cuda,
                              "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                              "bf16_matmul": True, "mooncake_native_import": TransferEngine is not None}
        except Exception as exc:
            result["errors"].append(f"GPU/native-library check failed: {exc}; see docs/user-guide/troubleshooting.md")
    result["ok"] = not result["errors"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", help="Also allocate a small CUDA tensor and import Mooncake")
    parser.add_argument("--target", action="store_true", help="Also require SGLang and its training patch")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = inspect_environment(gpu=args.gpu, target=args.target)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Python: {result['python']}")
        for check in result["checks"]:
            print(f"{'OK' if check['ok'] else 'FAIL'} {check['package']}: {check['installed']}")
        for override in result["dependencies"]["overrides"]:
            print(f"OVERRIDE {override['package']} {override['version']}: "
                  f"{override['requires']} -> {override['installed']}")
        for error in result["errors"]:
            print(f"FAIL {error}")
        print("PASS" if result["ok"] else "FAIL")
        if not args.gpu:
            print("GPU execution/transport not checked; run with --gpu, then the quickstart")
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
