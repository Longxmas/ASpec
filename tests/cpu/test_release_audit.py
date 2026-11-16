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

import io
import stat
import tarfile
import zipfile

import pytest

from tools.audit_release import SDIST_REQUIRED, WHEEL_REQUIRED, audit


def _artifacts(
    directory,
    *,
    wheel_extra=(),
    wheel_omit=(),
    sdist_extra=(),
    wheel_version="0.1.0b0",
    sdist_version="0.1.0b0",
    sdist_link=False,
    distribution="ling-aspec-runtime",
):
    archive_name = distribution.replace("-", "_")
    wheel = directory / f"{archive_name}-0.1.0b0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as output:
        for name in sorted(WHEEL_REQUIRED - set(wheel_omit)):
            output.writestr(name, "")
        output.writestr(
            f"{archive_name}-0.1.0b0.dist-info/METADATA",
            f"Name: {distribution}\nVersion: {wheel_version}\n",
        )
        for name in wheel_extra:
            output.writestr(name, "")

    source = directory / f"{archive_name}-0.1.0b0.tar.gz"
    with tarfile.open(source, "w:gz") as output:
        for relative in sorted(SDIST_REQUIRED | set(sdist_extra)):
            data = (
                f"Name: {distribution}\nVersion: {sdist_version}\n".encode()
                if relative == "PKG-INFO"
                else relative.encode()
            )
            info = tarfile.TarInfo(f"{archive_name}-0.1.0b0/{relative}")
            info.size = len(data)
            output.addfile(info, io.BytesIO(data))
        if sdist_link:
            link = tarfile.TarInfo(f"{archive_name}-0.1.0b0/aspec/link")
            link.type = tarfile.SYMTYPE
            link.linkname = "target"
            output.addfile(link)
    for role in ("ling-aspec", "ling-aspec-target"):
        normalized = role.replace("-", "_")
        metadata = (f"Name: {role}\nVersion: {wheel_version}\n"
                    f"Requires-Dist: ling-aspec-runtime=={wheel_version}\n")
        with zipfile.ZipFile(directory / f"{normalized}-0.1.0b0-py3-none-any.whl", "w") as output:
            output.writestr(f"{normalized}-0.1.0b0.dist-info/METADATA", metadata)
        with tarfile.open(directory / f"{normalized}-0.1.0b0.tar.gz", "w:gz") as output:
            for relative in ("PKG-INFO", "LICENSE", "README.md", "pyproject.toml"):
                data = metadata.encode() if relative == "PKG-INFO" else b"fixture"
                info = tarfile.TarInfo(f"{normalized}-0.1.0b0/{relative}")
                info.size = len(data)
                output.addfile(info, io.BytesIO(data))
    return wheel, source


def test_release_audit_accepts_complete_artifact_pair(tmp_path):
    wheel, source = _artifacts(tmp_path)
    result = audit(tmp_path)
    assert result[wheel.name] == len(WHEEL_REQUIRED) + 1
    assert result[source.name] == len(SDIST_REQUIRED)
    assert len(result) == 6


@pytest.mark.parametrize("member", ["aspec/compat/sglang_patch.py",
                                  "aspec/compat/patches/sglang-v0.5.21.patch",
                                  "aspec/_target_worker.py",
                                  "aspec/inference/engine/target_actor.py",
                                  "aspec/inference/service.py",
                                  "aspec/inference/service_protocol.py",
                                  "aspec/inference/service_client.py",
                                  "aspec/inference/managed.py"])
def test_release_audit_requires_bundled_patch_command_and_resource(tmp_path, member):
    _artifacts(tmp_path, wheel_omit=(member,))
    with pytest.raises(ValueError, match="missing wheel members"):
        audit(tmp_path)


@pytest.mark.parametrize(
    "member",
    [
        "archive/old.md",
        "outputs/run.log",
        "weights/model.safetensors",
        "aspec/__pycache__/module.pyc",
        "../escape",
        r"..\escape",
        "./aspec/module.py",
        "aspec//module.py",
        "C:/escape.py",
    ],
)
def test_release_audit_rejects_non_source_members(tmp_path, member):
    _artifacts(tmp_path, wheel_extra=(member,))
    with pytest.raises(ValueError, match="forbidden|unsafe"):
        audit(tmp_path)


def test_release_audit_requires_one_wheel_and_one_sdist(tmp_path):
    with pytest.raises(ValueError, match="exactly three wheels and three sdists"):
        audit(tmp_path)


def test_release_audit_rejects_mismatched_package_identity(tmp_path):
    _artifacts(tmp_path, sdist_version="0.2.0b0")
    with pytest.raises(ValueError, match="does not match"):
        audit(tmp_path)


def test_release_audit_rejects_the_unrelated_aspec_distribution(tmp_path):
    _artifacts(tmp_path, distribution="aspec")
    with pytest.raises(ValueError, match="unexpected distribution name"):
        audit(tmp_path)


def test_release_audit_rejects_duplicate_wheel_member(tmp_path):
    wheel, _ = _artifacts(tmp_path)
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(wheel, "a") as output:
            output.writestr(next(iter(WHEEL_REQUIRED)), "duplicate")
    with pytest.raises(ValueError, match="duplicate wheel member"):
        audit(tmp_path)


def test_release_audit_rejects_wheel_symlink(tmp_path):
    wheel, _ = _artifacts(tmp_path)
    link = zipfile.ZipInfo("aspec/link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(wheel, "a") as output:
        output.writestr(link, "target")
    with pytest.raises(ValueError, match="symbolic link"):
        audit(tmp_path)


def test_release_audit_rejects_sdist_symlink(tmp_path):
    _artifacts(tmp_path, sdist_link=True)
    with pytest.raises(ValueError, match="unsupported source member type"):
        audit(tmp_path)
