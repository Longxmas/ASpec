# Adapted from TorchSpec 6c042a87140a84d13839e341ece2c5c3ada918bc,
# torchspec/offline/dataset.py. Adds ASpec payload validation and run evidence.
#
# MIT License
#
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Small on-disk dataset used by offline replay."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

OFFLINE_SCHEMA_VERSION = 2
_TENSOR_NAMES = ("input_ids", "hidden_states", "target", "last_hidden_states", "mtp_hidden_states")


class OfflineDataset:
    """Read and append replay records.

    The format intentionally has only three parts: ``dataset.json``, one
    ``manifest.jsonl``, and self-describing ``samples/*.pt`` files.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        create: bool = False,
        last_hidden_states_prenorm: bool | None = None,
        binding: dict | None = None,
        allow_incomplete: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if create and self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f"Offline output must be a new or empty directory: {self.root}")
        if create and not binding:
            raise ValueError("Materialization requires a Target feature binding")

        metadata_path = self.root / "dataset.json"
        if create and not metadata_path.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(
                json.dumps(
                    {
                        "version": OFFLINE_SCHEMA_VERSION,
                        "binding": binding,
                        "complete": False,
                        "last_hidden_states_prenorm": last_hidden_states_prenorm,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Offline dataset not found: {metadata_path}")

        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("version") != OFFLINE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported offline dataset version {self.metadata.get('version')!r}"
            )
        if (
            create
            and last_hidden_states_prenorm is not None
            and self.metadata.get("last_hidden_states_prenorm") != last_hidden_states_prenorm
        ):
            raise ValueError("Offline dataset uses a different hidden-state representation")

        if not (create or allow_incomplete) and not self.metadata.get("complete"):
            raise ValueError("Offline dataset is incomplete; finish materialization first")
        from aspec.feature_contract import FeatureContractRuntime

        self._validation = FeatureContractRuntime(
            self.metadata["binding"]["feature_contract"],
            role="consumer",
            participant="offline_dataset",
        )
        self._rows: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
        self._by_id: dict[str, dict[str, Any]] = {}
        manifest = self.root / "manifest.jsonl"
        if manifest.exists():
            with manifest.open(encoding="utf-8") as stream:
                for lineno, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    split = row.get("split")
                    data_id = str(row.get("data_id"))
                    if split not in self._rows:
                        raise ValueError(f"Invalid split in {manifest}:{lineno}")
                    if data_id in self._by_id:
                        raise ValueError(f"Duplicate data_id {data_id!r} in {manifest}")
                    path = (self.root / row["file"]).resolve()
                    if self.root not in path.parents or not path.is_file():
                        raise FileNotFoundError(f"Offline sample not found: {path}")
                    item = {
                        "split": split,
                        "data_id": data_id,
                        "file": row["file"],
                        "seq_len": row["seq_len"],
                        "sha256": row["sha256"],
                    }
                    self._rows[split].append(item)
                    self._by_id[data_id] = item

        if self.metadata.get("complete"):
            actual = {split: self.count(split) for split in ("train", "eval")}
            if self.metadata.get("counts") != actual:
                raise ValueError("Offline manifest counts differ from completed dataset")

    def _validate_record(self, record, *, expected_length):
        from aspec.data.utils import unpack_loss_mask

        tensors = {name: record[name] for name in _TENSOR_NAMES if name in record}
        shapes = {name: tuple(value.shape) for name, value in tensors.items()}
        dtypes = {name: value.dtype for name, value in tensors.items()}
        self._validation.validate_metadata(
            tensor_shapes=shapes, tensor_dtypes=dtypes, metadata=record.get("metadata") or {}
        )
        self._validation.validate_tensors(tensors, expected_shapes=shapes, expected_dtypes=dtypes)
        if tensors["input_ids"].numel() != expected_length:
            raise ValueError("Offline sample length differs from its manifest")
        mask = record.get("packed_loss_mask")
        if mask is None or unpack_loss_mask(mask).numel() != expected_length:
            raise ValueError("Offline loss mask length differs from input_ids")

    @property
    def identity(self):
        manifest = self.root / "manifest.jsonl"
        return hashlib.sha256(
            json.dumps(
                {"metadata": self.metadata, "manifest": file_digest(manifest)}, sort_keys=True
            ).encode()
        ).hexdigest()

    def finalize(self, expected_counts):
        actual = {split: self.count(split) for split in ("train", "eval")}
        if actual != expected_counts or not actual["train"]:
            raise ValueError(f"Incomplete offline dataset: {actual} != {expected_counts}")
        self.metadata.update(complete=True, counts=actual)
        temporary = self.root / ".dataset.json.tmp"
        temporary.write_text(json.dumps(self.metadata, indent=2) + "\n")
        os.replace(temporary, self.root / "dataset.json")

    def rows(self, split: str) -> list[dict[str, Any]]:
        return list(self._rows[split])

    def ids(self, split: str) -> set[str]:
        return {row["data_id"] for row in self._rows[split]}

    def count(self, split: str) -> int:
        return len(self._rows[split])

    def load(self, row_or_id: dict[str, Any] | str) -> dict[str, Any]:
        row = self._by_id[str(row_or_id)] if isinstance(row_or_id, str) else row_or_id
        path = self.root / row["file"]
        if file_digest(path) != row["sha256"]:
            raise ValueError(f"Offline sample checksum mismatch: {path}")
        record = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(record, dict):
            raise ValueError(f"Offline sample must contain a dict: {path}")
        if str(record.get("data_id")) != row["data_id"]:
            raise ValueError(f"Offline sample data_id does not match manifest: {path}")
        if not all(
            isinstance(record.get(name), torch.Tensor) for name in ("input_ids", "hidden_states")
        ):
            raise ValueError(f"Offline sample has missing tensors: {path}")
        self._validate_record(record, expected_length=row["seq_len"])
        return record

    def append(
        self,
        split: str,
        *,
        data_id: str,
        tensors: dict[str, torch.Tensor | None],
        packed_loss_mask: list[int] | str | None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        if self.metadata.get("complete"):
            raise ValueError("Completed offline datasets are immutable; use a new output directory")
        if split not in self._rows:
            raise ValueError(f"Unknown offline split: {split!r}")
        data_id = str(data_id)
        if data_id in self._by_id:
            if self._by_id[data_id]["split"] != split:
                raise ValueError(f"data_id {data_id!r} already exists in another split")
            return False

        saved = {
            name: value.detach().cpu().contiguous()
            for name, value in tensors.items()
            if name in _TENSOR_NAMES and isinstance(value, torch.Tensor)
        }
        if not {"input_ids", "hidden_states"}.issubset(saved):
            raise ValueError(f"Offline sample {data_id!r} has missing tensors")

        relative = Path("samples") / f"{hashlib.sha1(data_id.encode()).hexdigest()}.pt"
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        from aspec.work_lifecycle import WORK_IDENTITY_KEYS

        payload_metadata = {
            key: value for key, value in (metadata or {}).items() if key not in WORK_IDENTITY_KEYS
        }
        record = {
            **saved,
            "data_id": data_id,
            "packed_loss_mask": packed_loss_mask,
            "metadata": payload_metadata,
        }
        self._validate_record(record, expected_length=int(saved["input_ids"].numel()))
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        os.close(fd)
        try:
            torch.save(record, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

        row = {
            "split": split,
            "data_id": data_id,
            "file": relative.as_posix(),
            "sha256": file_digest(path),
            "seq_len": int(saved["input_ids"].numel()),
        }
        with (self.root / "manifest.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._rows[split].append(row)
        self._by_id[data_id] = row
        return True


def configure_offline_args(dataset: OfflineDataset, args) -> None:
    """Use the representation detail recorded with the tensors."""
    value = dataset.metadata.get("last_hidden_states_prenorm")
    if value is None:
        raise ValueError("Offline dataset does not declare last_hidden_states_prenorm")
    args.last_hidden_states_prenorm = bool(value)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def target_identity(model_path):
    """Content identity, independent of the local directory used for the copy."""
    root = Path(model_path)
    if not (root / "config.json").is_file():
        raise ValueError("Offline materialization/replay requires a local Target checkpoint")
    files = sorted(
        {
            *root.glob("*.json"),
            *root.glob("*.safetensors"),
            *root.glob("pytorch_model*.bin"),
            *root.glob("tokenizer.model"),
        }
    )
    if not any(path.suffix in {".safetensors", ".bin"} for path in files):
        raise ValueError("Local Target checkpoint has no model weights")
    return {path.name: file_digest(path) for path in files}


def bind_offline_dataset(dataset, args, feature_contract):
    binding = dataset.metadata["binding"]
    if binding["feature_contract"] != feature_contract:
        raise ValueError("Offline Target feature contract differs from the requested Draft")
    expected = getattr(args, "offline_target_identity", None)
    if expected is None:
        expected = target_identity(args.target_model_path)
    if binding["target_identity"] != expected:
        raise ValueError("Offline dataset Target checkpoint identity does not match")
    if not dataset.count("train"):
        raise ValueError("Offline dataset contains no training samples")
    for split in ("train", "eval"):
        for row in dataset.rows(split):
            if row["seq_len"] > args.max_seq_length:
                raise ValueError(
                    "Offline sample exceeds training.max_seq_length; regenerate with the requested limit"
                )
    args.offline_target_identity = expected
    args.offline_dataset_identity = dataset.identity
    configure_offline_args(dataset, args)
