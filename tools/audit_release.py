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

"""Audit built ASpec source and wheel artifacts without extracting them."""

from __future__ import annotations

import argparse
import json
import stat
import tarfile
import zipfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path, PurePosixPath

FORBIDDEN_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "archive",
    "artifacts",
    "checkpoints",
    "outputs",
}
FORBIDDEN_SUFFIXES = {
    ".key",
    ".log",
    ".nsys-rep",
    ".pem",
    ".pt",
    ".pth",
    ".pyc",
    ".pyo",
    ".safetensors",
    ".trace",
}
SDIST_REQUIRED = {
    "AGENTS.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "PKG-INFO",
    "README.md",
    "ROADMAP.md",
    "SECURITY.md",
    "licenses/AngelSpec-LICENSE",
    "licenses/DFlash-MIT.txt",
    "licenses/DeepSpec-LICENSE",
    "licenses/EAGLE-LICENSE",
    "licenses/FlashAttention-AUTHORS",
    "licenses/FlashAttention-BSD-3-Clause.txt",
    "licenses/FlashLinearAttention-MIT.txt",
    "licenses/Ling-V2-MIT.txt",
    "licenses/NVIDIA-Megatron-LM-BSD-3-Clause.txt",
    "licenses/NVIDIA-cuDNN-Frontend-MIT.txt",
    "licenses/SGLang-LICENSE",
    "licenses/SpecForge-MIT.txt",
    "licenses/Speculators-LICENSE",
    "licenses/TorchSpec-LICENSE",
    "licenses/TorchSpec-THIRDPARTYNOTICES",
    "licenses/Transformers-LICENSE",
    "aspec/compat/fa4_cutlass.py",
    "aspec/models/target/target_utils.py",
    "aspec/compat/patches/sglang-v0.5.21.patch",
    "requirements/sglang-v0.5.21.txt",
    "requirements/draft.txt",
    "tools/audit_release.py",
    "tools/recipe_catalog.json",
    "tools/build_release.py",
    "tools/check_wheel_install.py",
    "setup.py",
}
WHEEL_REQUIRED = {
    "aspec/_target_worker.py",
    "aspec/inference/target_runtime.py",
    "aspec/inference/engine/target_actor.py",
    "aspec/inference/service.py",
    "aspec/inference/service_client.py",
    "aspec/inference/service_protocol.py",
    "aspec/inference/managed.py",
    "aspec/compat/fa4_cutlass.py",
    "aspec/compat/sglang_patch.py",
    "aspec/compat/patches/sglang-v0.5.21.patch",
    "aspec/models/target/target_utils.py",
    "aspec/assets.py",
    "aspec/_build_info.json",
    "aspec/commands/doctor.py",
    "aspec/commands/quickstart.py",
    "aspec/commands/recipes.py",
    "aspec/commands/convert_to_hf.py",
    "aspec/commands/stitch_native_mtp.py",
    "aspec/commands/evaluate_draft.py",
    "aspec/commands/prepare_quality_dataset.py",
    "aspec/commands/prepare_target_dataset.py",
    "aspec/commands/prepare_selected_answers.py",
    "aspec/commands/convert_pretokenized_jsonl.py",
    "aspec/commands/_recipe_smoke.py",
    "aspec/commands/_checkpoint_audit.py",
    "aspec/commands/_recipe_capacity.py",
    "aspec/resources/requirements/sglang-v0.5.21.txt",
    "aspec/resources/requirements/draft.txt",
    "aspec/resources/tools/recipe_catalog.json",
    "aspec/resources/examples/data/sample_conversations.jsonl",
}
WHEEL_REQUIRED.update(
    "aspec/resources/" + config
    for entry in json.loads((Path(__file__).with_name("recipe_catalog.json")).read_text()).values()
    for config in entry["configs"]
)


def _validate_name(name: str, artifact: Path) -> PurePosixPath:
    if not name or "\x00" in name or "\\" in name:
        raise ValueError(f"{artifact.name}: unsafe member path {name!r}")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or (path.parts and path.parts[0].endswith(":"))
        or path.as_posix() != name.rstrip("/")
    ):
        raise ValueError(f"{artifact.name}: unsafe member path {name!r}")
    if FORBIDDEN_PARTS.intersection(path.parts):
        raise ValueError(f"{artifact.name}: forbidden release member {name!r}")
    if path.name == ".DS_Store" or path.name.startswith("._"):
        raise ValueError(f"{artifact.name}: forbidden platform metadata {name!r}")
    if any(name.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
        raise ValueError(f"{artifact.name}: forbidden generated/sensitive member {name!r}")
    return path


def _identity(raw: bytes, artifact: Path) -> tuple[str, str]:
    metadata = BytesParser(policy=default).parsebytes(raw)
    name = metadata.get("Name")
    version = metadata.get("Version")
    if not name or not version:
        raise ValueError(f"{artifact.name}: package metadata lacks Name or Version")
    if name not in {"ling-aspec", "ling-aspec-target", "ling-aspec-runtime"}:
        raise ValueError(f"{artifact.name}: unexpected distribution name {name!r}")
    if name != "ling-aspec-runtime":
        requirements = {value.replace(" ", "") for value in metadata.get_all("Requires-Dist", [])}
        if f"ling-aspec-runtime=={version}" not in requirements:
            raise ValueError(f"{artifact.name}: role must require the matching runtime version")
    return name, version


def _wheel_members(path: Path) -> tuple[set[str], tuple[str, str]]:
    with zipfile.ZipFile(path) as wheel:
        items = wheel.infolist()
        names = {_validate_name(item.filename, path).as_posix() for item in items}
        if len(names) != len(items):
            raise ValueError(f"{path.name}: duplicate wheel member")
        if any(stat.S_ISLNK(item.external_attr >> 16) for item in items):
            raise ValueError(f"{path.name}: wheel contains a symbolic link")
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise ValueError(f"{path.name}: expected one dist-info/METADATA")
        identity = _identity(wheel.read(metadata_names[0]), path)
    missing = WHEEL_REQUIRED - names if identity[0] == "ling-aspec-runtime" else set()
    if identity[0] != "ling-aspec-runtime" and any(name.startswith("aspec/") for name in names):
        raise ValueError(f"{path.name}: role packages must not own runtime files")
    if missing:
        raise ValueError(f"{path.name}: missing wheel members {sorted(missing)}")
    return names, identity


def _sdist_members(path: Path) -> tuple[set[str], tuple[str, str]]:
    with tarfile.open(path, mode="r:gz") as source:
        members = source.getmembers()
        if len({member.name for member in members}) != len(members):
            raise ValueError(f"{path.name}: duplicate source member")
        for member in members:
            _validate_name(member.name, path)
            if not (member.isfile() or member.isdir()):
                raise ValueError(
                    f"{path.name}: unsupported source member type for {member.name!r}"
                )
    roots = {PurePosixPath(member.name).parts[0] for member in members if member.name}
    if len(roots) != 1:
        raise ValueError(f"{path.name}: expected one source root, got {sorted(roots)}")
    root = next(iter(roots))
    names = {
        PurePosixPath(*PurePosixPath(member.name).parts[1:]).as_posix()
        for member in members
        if len(PurePosixPath(member.name).parts) > 1
    }
    missing = SDIST_REQUIRED - names if root.startswith("ling_aspec_runtime-") else {"PKG-INFO", "LICENSE", "README.md", "pyproject.toml"} - names
    if missing:
        raise ValueError(f"{path.name}: missing source members {sorted(missing)}")
    if not root.startswith(("ling_aspec-", "ling_aspec_target-", "ling_aspec_runtime-")):
        raise ValueError(f"{path.name}: unexpected source root {root!r}")
    with tarfile.open(path, mode="r:gz") as source:
        metadata_member = source.getmember(f"{root}/PKG-INFO")
        source_metadata = source.extractfile(metadata_member)
        if source_metadata is None:
            raise ValueError(f"{path.name}: PKG-INFO is not a regular file")
        identity = _identity(source_metadata.read(), path)
    return names, identity


def audit(directory: Path) -> dict[str, int]:
    wheels = sorted(directory.glob("*.whl"))
    sdists = sorted(directory.glob("*.tar.gz"))
    if len(wheels) != 3 or len(sdists) != 3:
        raise ValueError(f"{directory}: expected exactly three wheels and three sdists")
    result, identities = {}, {}
    for artifact in [*wheels, *sdists]:
        members, identity = (_wheel_members(artifact) if artifact.suffix == ".whl"
                             else _sdist_members(artifact))
        result[artifact.name] = len(members)
        identities.setdefault(identity, []).append(artifact.suffix == ".whl")
    expected = {"ling-aspec", "ling-aspec-target", "ling-aspec-runtime"}
    if ({name for name, version in identities} != expected
            or len({version for name, version in identities}) != 1
            or any(sorted(kinds) != [False, True] for kinds in identities.values())):
        raise ValueError("wheel identity does not match the three version-aligned source identities")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("dist"))
    args = parser.parse_args()
    for artifact, members in audit(args.directory).items():
        print(f"PASS {artifact}: {members} reviewed members")


if __name__ == "__main__":
    main()
