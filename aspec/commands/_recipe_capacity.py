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

"""Validate actual 256K-class packed work and sampled device memory."""

import argparse
import csv
import json
from pathlib import Path

from omegaconf import OmegaConf


def memory_gpu_ids(config, entry):
    """Observe the complete single-node allocation across both environments."""
    from aspec.inference.service_protocol import request
    from aspec.utils.misc import get_current_node_ip, visible_gpu_ids

    draft = visible_gpu_ids()
    service = OmegaConf.select(config, "inference.target_service", default={})
    if service.get("mode", "external") == "managed-local":
        return draft  # The supervisor has not split the combined visibility yet.
    node_ip = get_current_node_ip()
    address = (service.get("addresses") or {}).get(
        node_ip, service.get("address", "http://127.0.0.1:30000"),
    )
    info = request(address, "/v1/info", timeout=30)
    if info["node_ip"] != node_ip:
        raise ValueError("Single-node capacity acceptance requires Target on the Draft host")
    if info["draining"] or any(item["state"] != "closed" for item in info["engines"].values()):
        raise ValueError("Capacity acceptance requires an idle Target GPU pool")
    target = [gpu for gpu in info["devices"] if gpu not in draft]
    if len(target) != entry["target_gpus"]:
        raise ValueError("Select the recipe's allocated Target GPUs before capacity acceptance")
    return draft + target


def audit_long_run(directory):
    config = OmegaConf.load(Path(directory) / "config.yaml")
    backend = OmegaConf.select(config, "training.attention_backend")
    if backend not in {"fa4", "cudnn_flex_attention"}:
        raise ValueError(f"Unsupported configured attention backend for long-context CP2: {backend!r}")
    evidence = json.loads((Path(directory) / "evidence.json").read_text())
    training = evidence.get("observations", {}).get("training", {})
    scheduler = training.get("scheduler", {})
    execution = training.get("execution", {})
    if (
        scheduler.get("observed_sequence_min") != 261888
        or scheduler.get("observed_packed_row_min") != 261888
        or scheduler.get("observed_packed_row_max") != 261888
        or scheduler.get("rows_dispatched", 0) < 1
        or scheduler.get("packed_tokens") != 261888 * scheduler.get("rows_dispatched", 0)
        or execution.get("context_parallel_size") != 2
        or execution.get("attention_backend") != backend
    ):
        raise ValueError(f"Missing actual 261888-token packed work with {backend} CP2")
    return {
        "tokens_per_row": 261888,
        "rows": scheduler["rows_dispatched"],
        "packed_tokens": scheduler["packed_tokens"],
        "context_parallel_size": 2,
        "attention_backend": backend,
    }


def summarize_memory(path, expected_gpus=8):
    devices = {}
    with Path(path).open() as stream:
        for raw in csv.DictReader(stream):
            row = {key.strip(): value.strip() for key, value in raw.items()}
            key = row.get("uuid", row["index"])
            used = int(row["memory.used [MiB]"].split()[0])
            total = int(row["memory.total [MiB]"].split()[0])
            if not 0 <= used <= total or total <= 0:
                raise ValueError("Invalid GPU memory sample")
            device = devices.setdefault(
                key, {"index": row["index"], "total_mib": total, "peak_used_mib": 0, "samples": 0}
            )
            if device["total_mib"] != total:
                raise ValueError("GPU capacity changed during the run")
            device["samples"] += 1
            device["peak_used_mib"] = max(device["peak_used_mib"], used)
    if len(devices) != expected_gpus or any(d["samples"] < 2 for d in devices.values()):
        raise ValueError("Missing memory observations for the allocated GPUs")
    return {
        "scope": "one-second sampled device-wide memory, not allocator peak or quality",
        "devices": devices,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-csv", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads((args.output / "acceptance.json").read_text())
    if receipt.get("status") != "passed" or receipt.get("recipe") != "qwen27b-long":
        raise ValueError("A passed 27B train/resume/export receipt is required")
    result = {
        "status": "passed",
        "training": {
            phase: audit_long_run(args.output / "stage-1" / phase) for phase in ("first", "resumed")
        },
        "memory": summarize_memory(args.memory_csv),
    }
    (args.output / "capacity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
