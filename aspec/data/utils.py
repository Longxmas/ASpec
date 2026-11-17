# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import IterableDataset, load_dataset
from huggingface_hub import hf_hub_download, list_repo_files

from aspec.models.ops.loss_mask import compute_assistant_loss_mask

_LOCAL_DATA_EXTS = frozenset({".json", ".jsonl", ".parquet", ".arrow", ".csv", ".tsv", ".txt"})


def is_local_data_path(path: str, base_dir: str | None = None) -> bool:
    """True if *path* looks like a local file/directory rather than a HF Hub dataset ID.

    When *base_dir* is given, relative paths are probed against it instead of
    the process CWD.
    """
    if path.startswith((".", "/", "~")):
        return True
    if os.path.splitext(path)[1].lower() in _LOCAL_DATA_EXTS:
        return True
    probe = os.path.join(base_dir, path) if base_dir is not None else path
    return os.path.exists(probe)


class DataCollatorWithPadding:
    def __init__(
        self, usp_enabled: bool = False, direct_cp_enabled: bool = False
    ):
        self.sp_degree = 1
        self.usp_enabled = usp_enabled
        self.direct_cp_enabled = direct_cp_enabled

    def paddingtensor(self, intensors: torch.Tensor, N: int) -> torch.Tensor:
        B, n, S = intensors.shape
        # Truncate if longer than target (can happen when loss_mask/hidden_states
        # length differs from input_ids after unpacking).
        if n > N:
            return intensors[:, :N, :]
        if n == N:
            return intensors
        padding_tensor = torch.zeros(B, N - n, S, dtype=intensors.dtype, device=intensors.device)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors: torch.Tensor, N: int) -> torch.Tensor:
        B, n = intensors.shape
        # Truncate if longer than target (prevents negative padding dimension
        # when loss_mask length differs from input_ids after collation).
        if n > N:
            return intensors[:, :N]
        if n == N:
            return intensors
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype, device=intensors.device)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def _get_loss_mask(self, item: Dict[str, Any]) -> torch.Tensor:
        """Read the materialized loss_mask tensor from the item.

        Callers (e.g. MooncakeDataset) are responsible for computing and
        attaching loss_mask before items reach the collator.
        """
        if "loss_mask" in item and isinstance(item["loss_mask"], torch.Tensor):
            return item["loss_mask"]
        raise KeyError(f"loss_mask not found in item: {item}")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item["input_ids"].shape[1] for item in features)
        max_length = ((max_length + self.sp_degree - 1) // self.sp_degree) * self.sp_degree

        if self.usp_enabled:
            attention_masks = [item["attention_mask"].long() for item in features]
        else:
            # Cached eval samples have already been padded once. Retain their
            # valid-token mask when re-batching; treating padding as a document
            # changes both supervision and per-document CP ownership.
            attention_masks = [
                (item["attention_mask"] if "attention_mask" in item
                 else torch.ones_like(item["input_ids"])).long()
                for item in features
            ]

        batch_input_ids = torch.cat(
            [self.paddingtensor2D(item["input_ids"], max_length) for item in features]
        )
        batch_attention_mask = torch.cat(
            [self.paddingtensor2D(mask, max_length) for mask in attention_masks]
        )
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(self._get_loss_mask(item), max_length) for item in features]
        )
        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "hidden_states": None,
            "target": None,
            "last_hidden_states": None,
        }
        work_items = [item.get("_aspec_work") for item in features]
        if any(item is not None for item in work_items):
            if not all(
                isinstance(item, dict)
                and isinstance(item.get("execution_id"), str)
                and item.get("execution_id")
                for item in work_items
            ):
                raise ValueError("Batch contains incomplete ASpec work identity metadata")
            batch["_aspec_work"] = work_items
        if self.usp_enabled:
            max_position_length = max(item["position_ids"].shape[1] for item in features)
            batch["position_ids"] = torch.cat(
                [
                    self.paddingtensor2D(item["position_ids"], max_position_length)
                    for item in features
                ]
            )
        if all("hidden_states" in item for item in features):
            hidden_max_length = (
                max(item["hidden_states"].shape[1] for item in features)
                if self.direct_cp_enabled
                else max_length
            )
            batch["hidden_states"] = torch.cat(
                [
                    self.paddingtensor(item["hidden_states"], hidden_max_length)
                    for item in features
                ]
            )
            has_target = all(item.get("target") is not None for item in features)
            has_last_hs = all(item.get("last_hidden_states") is not None for item in features)
            if not has_target and not has_last_hs:
                pass
            if has_target:
                batch["target"] = torch.cat(
                    [
                        self.paddingtensor(item["target"], hidden_max_length)
                        for item in features
                    ]
                )
            if has_last_hs:
                last_hidden_max_length = (
                    max(item["last_hidden_states"].shape[1] for item in features)
                    if self.direct_cp_enabled
                    else max_length
                )
                batch["last_hidden_states"] = torch.cat(
                    [
                        self.paddingtensor(
                            item["last_hidden_states"], last_hidden_max_length
                        )
                        for item in features
                    ]
                )
        mtp = [item.get("mtp_hidden_states") for item in features]
        if any(value is not None for value in mtp):
            if not all(value is not None for value in mtp):
                raise ValueError("MTP features must be present for every document in the batch")
            if any(value.shape[:2] != item["input_ids"].shape for item, value in zip(features, mtp, strict=True)):
                raise ValueError("MTP features must align with each document's input token IDs")
            batch["mtp_hidden_states"] = torch.cat([self.paddingtensor(value, max_length) for value in mtp])
        return batch


def _packing_tensor(item: Dict[str, Any], name: str, sample_len: int) -> torch.Tensor:
    value = item.get(name)
    if not isinstance(value, torch.Tensor):
        raise KeyError(f"DFlashPackingCollator: {name} tensor not found")
    expected_dim = 2 if name in {"input_ids", "loss_mask"} else 3
    if value.dim() != expected_dim or value.shape[0] != 1:
        raise ValueError(
            f"DFlashPackingCollator: invalid {name} shape {tuple(value.shape)}"
        )
    if value.shape[1] < sample_len:
        raise ValueError(
            f"DFlashPackingCollator: {name} length {value.shape[1]} "
            f"is shorter than input_ids length {sample_len}"
        )
    return value[:, :sample_len]


class DFlashPackingCollator:
    """Concatenate complete documents into one fixed, document-aware row.

    Adapted from AngelSpec. Overflow is a contract error here because the live
    controller is the sole row owner; silently dropping a transferred document
    would violate ASpec work identity and commit semantics.
    """

    def __init__(
        self, max_seq_length: int, *, pad_to_max_length: bool = True,
        context_parallel_rank: int = 0, context_parallel_size: int = 1,
        context_shard_alignment: int = 1,
    ):
        if max_seq_length <= 0:
            raise ValueError("DFlashPackingCollator requires max_seq_length > 0")
        self.max_seq_length = int(max_seq_length)
        self.pad_to_max_length = bool(pad_to_max_length)
        self.cp_rank = int(context_parallel_rank)
        self.cp_size = int(context_parallel_size)
        self.cp_alignment = int(context_shard_alignment)
        if self.cp_size <= 0 or not 0 <= self.cp_rank < self.cp_size:
            raise ValueError("Invalid packed context-parallel ownership")
        if self.cp_alignment <= 0:
            raise ValueError("Packed context shard alignment must be positive")
        if self.cp_size > 1 and self.pad_to_max_length:
            raise ValueError("Context-sharded packing requires unpadded THD rows")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not features:
            raise ValueError("DFlashPackingCollator received an empty row")
        if any(item.get("mtp_hidden_states") is not None for item in features):
            raise ValueError("MTP-conditioned documents do not support packing")
        lengths = [int(item["input_ids"].shape[1]) for item in features]
        used = sum(lengths)
        if used <= 0 or used > self.max_seq_length:
            raise ValueError(
                f"Packed row has {used} tokens; expected 1..{self.max_seq_length}"
            )

        have_hidden = [item.get("hidden_states") is not None for item in features]
        have_last_hidden = [item.get("last_hidden_states") is not None for item in features]
        if any(have_hidden) and not all(have_hidden):
            raise ValueError("hidden_states must be present for every packed document")
        if any(have_last_hidden) and not all(have_last_hidden):
            raise ValueError("last_hidden_states must be present for every packed document")

        ids = [_packing_tensor(item, "input_ids", length) for item, length in zip(features, lengths)]
        masks = [_packing_tensor(item, "loss_mask", length) for item, length in zip(features, lengths)]
        device = ids[0].device
        pad = self.max_seq_length - used if self.pad_to_max_length else 0

        def pad2d(parts: List[torch.Tensor], fill: int) -> torch.Tensor:
            value = torch.cat(parts, dim=1)
            if pad:
                value = torch.cat(
                    [
                        value,
                        torch.full(
                            (1, pad), fill, dtype=value.dtype, device=value.device
                        ),
                    ],
                    dim=1,
                )
            return value

        doc_ids = [
            torch.full((1, length), doc_id, dtype=torch.long, device=device)
            for doc_id, length in enumerate(lengths)
        ]
        positions = [
            torch.arange(length, dtype=torch.long, device=device).unsqueeze(0)
            for length in lengths
        ]
        batch: Dict[str, Any] = {
            "packed_document_lengths": tuple(lengths),
            "input_ids": pad2d(ids, 0),
            "attention_mask": torch.cat(
                [
                    torch.ones((1, used), dtype=torch.long, device=device),
                    torch.zeros((1, pad), dtype=torch.long, device=device),
                ],
                dim=1,
            ),
            "loss_mask": pad2d(masks, 0),
            "ctx_doc_ids": pad2d(doc_ids, -1),
            "base_position_ids": pad2d(positions, 0),
            "hidden_states": None,
            "target": None,
            "last_hidden_states": None,
        }

        for name, presence in (
            ("hidden_states", have_hidden),
            ("last_hidden_states", have_last_hidden),
        ):
            if not all(presence):
                continue
            from aspec.inference.context import context_shard_bounds

            bounds = [
                context_shard_bounds(
                    length,
                    self.cp_rank,
                    self.cp_size,
                    alignment=self.cp_alignment,
                )
                for length in lengths
            ]
            parts = [
                _packing_tensor(item, name, end - start)
                for item, (start, end) in zip(features, bounds, strict=True)
            ]
            if self.cp_size > 1:
                expected = [end - start for start, end in bounds]
                if any(item[name].shape[1] != count for item, count in zip(features, expected)):
                    raise ValueError(f"Packed {name} does not match per-document CP ownership")
            value = torch.cat(parts, dim=1)
            if pad:
                value = torch.cat(
                    [
                        value,
                        torch.zeros(
                            (1, pad, value.shape[-1]),
                            dtype=value.dtype,
                            device=value.device,
                        ),
                    ],
                    dim=1,
                )
            batch[name] = value

        work = [item.get("_aspec_work") for item in features]
        if not all(
            isinstance(item, dict)
            and isinstance(item.get("execution_id"), str)
            and item["execution_id"]
            for item in work
        ):
            raise ValueError("Packed row contains incomplete ASpec work identity")
        batch["_aspec_work"] = work
        if self.cp_size > 1:
            offset = 0
            owned_positions = []
            for length in lengths:
                from aspec.inference.context import context_shard_bounds

                start, end = context_shard_bounds(
                    length,
                    self.cp_rank,
                    self.cp_size,
                    alignment=self.cp_alignment,
                )
                owned_positions.append(torch.arange(
                    offset + start,
                    offset + end,
                    dtype=torch.long, device=device,
                ))
                offset += length
            batch["packed_context_positions"] = torch.cat(owned_positions).unsqueeze(0)
        return batch


def pack_loss_mask(loss_mask: torch.Tensor) -> List[int]:
    """
    Pack a loss_mask tensor into interleaved segment lengths.

    The returned list alternates between prompt and response lengths,
    always starting with prompt (even if length 0).

    Args:
        loss_mask: 1D tensor of 0s and 1s indicating which tokens contribute to loss.

    Returns:
        List of segment lengths: [prompt_len, response_len, prompt_len, response_len, ...]

    Example:
        loss_mask = [0, 0, 1, 1, 1, 0, 0, 1, 1, 0]
        returns: [2, 3, 2, 2, 1]  # 2 prompt, 3 response, 2 prompt, 2 response, 1 prompt
    """
    if loss_mask.dim() > 1:
        loss_mask = loss_mask.squeeze()

    if len(loss_mask) == 0:
        return []

    lengths = []
    mask_list = loss_mask.tolist()
    current_val = 0
    current_len = 0

    for val in mask_list:
        if val == current_val:
            current_len += 1
        else:
            lengths.append(current_len)
            current_val = val
            current_len = 1

    lengths.append(current_len)
    return lengths


def unpack_loss_mask(packed: Union[List[int], str]) -> torch.Tensor:
    """
    Unpack segment lengths back into a loss_mask tensor.

    Args:
        packed: List of segment lengths [prompt_len, response_len, prompt_len, ...],
                or a serialized packed_loss_mask string (e.g. "2,3,2,2,1").

    Returns:
        1D tensor of 0s and 1s.

    Example:
        packed = [2, 3, 2, 2, 1]
        returns: tensor([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
    """
    if isinstance(packed, str):
        packed = deserialize_packed_loss_mask(packed)
    if not packed:
        return torch.tensor([], dtype=torch.long)

    total = sum(packed)
    loss_mask = torch.zeros(total, dtype=torch.long)
    pos = 0

    for i, length in enumerate(packed):
        if i % 2 == 1:
            loss_mask[pos : pos + length] = 1
        pos += length

    return loss_mask


def resolve_loss_mask(
    data: Dict[str, Any],
    *,
    dynamic_loss_mask: bool = False,
    assistant_header_ids: Optional[List[int]] = None,
    end_token_ids: Optional[List[int]] = None,
    last_turn_loss_only: bool = False,
    skip_after_header: int = 0,
) -> torch.Tensor | None:
    """
    Two strategies, tried in order:
    1. ``packed_loss_mask`` key present → unpack it.
    2. ``dynamic_loss_mask`` enabled with valid header/end ids → compute from
       ``input_ids`` via :func:`compute_assistant_loss_mask`.
    """
    packed = data.get("packed_loss_mask")
    if packed is not None:
        mask = unpack_loss_mask(packed)
        input_ids = data.get("input_ids")
        if input_ids is not None:
            if input_ids.dim() == 2:
                input_ids = input_ids.squeeze(0)
            expected_len = input_ids.shape[-1]
            if mask.shape[0] > expected_len:
                mask = mask[:expected_len]
            elif mask.shape[0] < expected_len:
                mask = torch.nn.functional.pad(mask, (0, expected_len - mask.shape[0]))
        if not mask.any():
            return None
        data["loss_mask"] = mask
        return mask

    if dynamic_loss_mask and assistant_header_ids and end_token_ids:
        input_ids = data.get("input_ids")
        if input_ids is None:
            return None
        if input_ids.dim() == 2:
            input_ids = input_ids.squeeze(0)
        per_sample = data.get("last_turn_loss_only")
        last_turn_only = per_sample if per_sample is not None else last_turn_loss_only
        if not isinstance(last_turn_only, bool):
            last_turn_only = False
        mask = compute_assistant_loss_mask(
            input_ids,
            assistant_header_ids,
            end_token_ids,
            last_turn_only=last_turn_only,
            skip_after_header=skip_after_header,
        )
        if not mask.any():
            return None
        data["loss_mask"] = mask
        return mask

    return torch.ones(1)


def serialize_packed_loss_mask(packed: List[int]) -> str:
    """
    Serialize packed loss_mask to a comma-separated string.

    Args:
        packed: List of segment lengths from pack_loss_mask().

    Returns:
        Comma-separated string of integers.

    Example:
        packed = [2, 3, 2, 2, 1]
        returns: "2,3,2,2,1"
    """
    return ",".join(str(x) for x in packed)


def deserialize_packed_loss_mask(s: str) -> List[int]:
    """
    Deserialize a comma-separated string back to packed loss_mask.

    Args:
        s: Comma-separated string of integers.

    Returns:
        List of segment lengths.

    Example:
        s = "2,3,2,2,1"
        returns: [2, 3, 2, 2, 1]
    """
    if not s:
        return []
    return [int(x) for x in s.split(",")]


def extract_media_urls(messages: list) -> dict | None:
    """Extract image/video URLs from structured messages without loading them."""
    images = []
    videos = []

    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "image":
                if "image" in item:
                    images.append(item["image"])
                else:
                    images.append(None)
            elif item_type == "image_url":
                image_url = item.get("image_url")
                if isinstance(image_url, dict) and "url" in image_url:
                    images.append(image_url["url"])
                else:
                    images.append(None)
            elif item_type == "video":
                if "video" in item:
                    videos.append(item["video"])
                else:
                    videos.append(None)

    if not images and not videos:
        return None
    return {"images": images or None, "videos": videos or None}


def flatten_multimodal_content(messages, image_placeholder="<image>"):
    """Convert list-type content parts to plain text strings.

    Transforms the standard HF multimodal format:
      [{"type":"image"}, {"type":"text","text":"Describe"}]
    into a single string:
      "<image>\\nDescribe"

    Messages with string content are left unchanged.
    Must be called AFTER extract_media_urls so structured info is captured first.
    """
    for msg in messages:
        content = msg.get("content")
        if content is None:
            msg["content"] = ""
            continue
        if not isinstance(content, (str, list)):
            raise ValueError(
                f"Message content must be a str or list, got {type(content).__name__}: {repr(content)[:100]}"
            )
        if not isinstance(content, list):
            continue
        text_parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type", "")
            if part_type == "text":
                text_parts.append(part.get("text", ""))
            elif part_type in ("image", "image_url"):
                text_parts.append(image_placeholder)
            elif part_type == "video":
                text_parts.append("<video>")
        msg["content"] = "\n".join(text_parts)
    return messages


def sample_length_hint(sample: Dict[str, Any]) -> int:
    """Sequence length of a stored dataset entry, for length grouping.

    Exactly one of these three is present: ``seq_len`` on offline replay rows,
    the tokenized ``input_ids``, or — under ``defer_tokenization`` — only the
    formatted text, whose character count is monotone in token count and free.
    """
    if "seq_len" in sample:
        return sample["seq_len"]
    if "input_ids" in sample:
        return sample["input_ids"].numel()
    return len(sample["formatted_prompt"])


def length_grouped_order(samples: List[Any], group_size: int) -> List[Any]:
    """Sort *samples* longest-first within fixed chunks of *group_size*.

    Chunks are positional, so only the order inside each one changes. That keeps
    a dispatch's sequences similar in length without turning length into a
    curriculum the way a global sort would.
    """
    if group_size <= 1:
        return list(samples)

    ordered = []
    for start in range(0, len(samples), group_size):
        chunk = samples[start : start + group_size]
        chunk.sort(key=sample_length_hint, reverse=True)
        ordered.extend(chunk)
    return ordered


def estimate_row_count(data_path):
    if not os.path.isfile(data_path):
        return None
    if data_path.endswith(".jsonl"):
        with open(data_path, "rb") as f:
            return sum(1 for _ in f)
    if data_path.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq

            return pq.ParquetFile(data_path).metadata.num_rows
        except Exception:
            return None
    if data_path.endswith(".json"):
        return None
    return None


def load_local_json(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            yield from json.load(f)
        else:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _list_hub_data_files(data_path: str, suffixes: tuple[str, ...]) -> list[str]:
    """List repo files matching ``suffixes``, preferring those under ``data/``."""
    files = list_repo_files(data_path, repo_type="dataset")
    matching = sorted(f for f in files if f.endswith(suffixes))
    preferred = [f for f in matching if f.startswith("data/")]
    return preferred or matching


def _load_hub_json_files(data_path):
    """Download JSON/JSONL files from a HF Hub dataset and yield rows.

    Uses raw json module instead of load_dataset to avoid PyArrow schema
    inference failures on datasets with mixed-type columns.
    """
    data_files = _list_hub_data_files(data_path, (".jsonl", ".json"))
    if not data_files:
        raise ValueError(f"No JSON/JSONL files found in HF Hub dataset '{data_path}'.")

    for filename in data_files:
        local_path = hf_hub_download(repo_id=data_path, filename=filename, repo_type="dataset")
        yield from load_local_json(local_path)


def _load_hub_parquet_dataset(data_path: str):
    """Stream parquet files from a HF Hub dataset.

    Used when ``load_dataset(repo_id, ...)`` fails due to schema/card parsing
    issues but the repo's parquet shards themselves are valid.
    """
    parquet_files = _list_hub_data_files(data_path, (".parquet",))
    if not parquet_files:
        raise ValueError(f"No parquet files found in HF Hub dataset '{data_path}'.")
    urls = [f"https://huggingface.co/datasets/{data_path}/resolve/main/{f}" for f in parquet_files]
    return load_dataset("parquet", data_files=urls, split="train", streaming=True)


def local_dataset_shards(data_path: str | Path):
    """Resolve the same ordered directory inputs for loading and cache identity."""
    patterns = {"json": ("*.json", "*.jsonl"), "parquet": ("*.parquet",), "arrow": ("*.arrow",)}
    for fmt, globs in patterns.items():
        files = sorted(p for pattern in globs for p in Path(data_path).rglob(pattern) if p.is_file())
        if files:
            return fmt, files
    raise ValueError(f"No supported dataset files found in local directory: {data_path}")


def load_hf_dataset(data_path: str):
    """Load dataset as a streaming IterableDataset.

    Local paths are loaded directly; everything else goes to HF Hub.
    """
    data_path = os.path.expanduser(data_path)

    if is_local_data_path(data_path):
        if os.path.isfile(data_path):
            if data_path.endswith((".json", ".jsonl")):
                return IterableDataset.from_generator(
                    load_local_json, gen_kwargs={"data_path": data_path}
                )
            ext = os.path.splitext(data_path)[1].lower()
            fmt = {".parquet": "parquet", ".arrow": "arrow"}.get(ext, "json")
            return load_dataset(fmt, data_files=data_path, split="train", streaming=True)

        if os.path.isdir(data_path):
            fmt, files = local_dataset_shards(data_path)
            return load_dataset(fmt, data_files=[str(p) for p in files], split="train", streaming=True)

        raise FileNotFoundError(f"Local dataset path not found: {data_path}")

    # hub path — try native load_dataset first (handles Arrow, Parquet, etc.),
    # fall back to manual JSON download for repos with mixed-type columns
    _KEEP_COLUMNS = frozenset(
        {
            "id",
            "data_id",
            "conversations",
            "text",
            "messages",
            "tools",
            "generation_config",
        }
    )
    try:
        ds = load_dataset(data_path, split="train", streaming=True)
        columns = ds.column_names or []
        # A pretokenized repo's provenance columns cannot be enumerated ahead of
        # time, and dropping input_ids would silently demote it to raw text.
        if "input_ids" in columns:
            return ds
        drop_cols = [c for c in columns if c not in _KEEP_COLUMNS]
        if drop_cols:
            ds = ds.remove_columns(drop_cols)
        return ds
    except (ValueError, TypeError, ArithmeticError, KeyError) as e:
        # Schema inference / card-parsing failures (e.g., mixed-type columns in
        # Arrow/Parquet, malformed dataset_info YAML). Fall back to direct file
        # streaming — parquet first, then JSON/JSONL.
        import logging

        log = logging.getLogger(__name__)
        log.info(f"load_dataset failed for '{data_path}' ({e}), trying direct file streaming")
        try:
            return _load_hub_parquet_dataset(data_path)
        except ValueError as parquet_err:
            log.info(f"parquet streaming failed ({parquet_err}), falling back to JSON download")
            return IterableDataset.from_generator(
                _load_hub_json_files, gen_kwargs={"data_path": data_path}
            )
