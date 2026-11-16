# Copyright 2026 Ant Group Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build and audit wheel/sdist, then collect runtime pins and SHA256 checksums."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

from tools.audit_release import audit


def build_release(directory):
    root = Path(__file__).resolve().parents[1]
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("Use an empty release output directory")
    subprocess.run([sys.executable, "-m", "build", "--outdir", str(directory)],
                   cwd=root, check=True)
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    for role, manifest in (("draft", "draft.txt"), ("target", "sglang-v0.5.21.txt")):
        with tempfile.TemporaryDirectory(prefix="aspec-role-") as staging:
            stage = Path(staging)
            name = "ling-aspec" if role == "draft" else "ling-aspec-target"
            dependencies = [f"ling-aspec-runtime=={project['version']}"]
            dependencies += [line.strip() for line in (root / "requirements" / manifest).read_text().splitlines()
                             if line.strip() and not line.lstrip().startswith("#")]
            metadata = ("[build-system]\nrequires = [\"setuptools>=77\", \"wheel\"]\n"
                        "build-backend = \"setuptools.build_meta\"\n\n[project]\n"
                        f"name = {json.dumps(name)}\nversion = {json.dumps(project['version'])}\n"
                        f"description = {json.dumps('ASpec ' + role + ' environment')}\n"
                        "readme = \"README.md\"\nlicense = \"Apache-2.0\"\n"
                        "license-files = [\"LICENSE\"]\nrequires-python = \">=3.11\"\n"
                        f"dependencies = {json.dumps(dependencies)}\n\n"
                        "[project.urls]\nHomepage = \"https://github.com/inclusionAI/ASpec\"\n"
                        "\n[tool.setuptools]\npackages = []\n")
            (stage / "pyproject.toml").write_text(metadata)
            (stage / "README.md").write_text(f"# ASpec {role} environment\n\n"
                                            "See https://github.com/inclusionAI/ASpec for installation and usage.\n")
            shutil.copyfile(root / "LICENSE", stage / "LICENSE")
            subprocess.run([sys.executable, "-m", "build", "--outdir", str(directory)],
                           cwd=stage, check=True)
    for artifact, count in audit(directory).items():
        print(f"PASS {artifact}: {count} reviewed members", flush=True)
    runtime = directory / "runtime"
    runtime.mkdir()
    for name in ("sglang-v0.5.21.txt", "draft.txt"):
        shutil.copyfile(root / "requirements" / name, runtime / name)
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    (directory / "SHA256SUMS").write_text("".join(
        hashlib.sha256(path.read_bytes()).hexdigest() + "  "
        + path.relative_to(directory).as_posix() + "\n" for path in files))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    build_release(args.outdir)


if __name__ == "__main__":
    main()
