# Adapted from TorchSpec 6c042a87140a84d13839e341ece2c5c3ada918bc,
# torchspec/inference/engine/offline_replay_engine.py. Adds ASpec payload validation and run evidence.
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

"""Inference-engine adapter that replays materialized target tensors from disk."""

from __future__ import annotations

import dataclasses
import os
import uuid
from typing import Any

import ray
import torch

from aspec.inference.engine.base import InferenceEngine
from aspec.offline.dataset import OfflineDataset, bind_offline_dataset
from aspec.ray.ray_actor import RayActor
from aspec.transfer.mooncake.eagle_store import EagleMooncakeStore
from aspec.utils.logging import logger, setup_file_logging


class OfflineReplayEngine(InferenceEngine, RayActor):
    """Serve recorded target-model outputs through the normal Mooncake contract.

    The async inference manager treats this actor like any other inference
    engine. Instead of running a target model, ``generate`` looks records up by
    data ID, writes their tensors to Mooncake, and returns the same metadata as
    a live engine.
    """

    def __init__(self, args, rank: int, engine_group: int = 0, **_kwargs) -> None:
        self.args = args
        self.rank = rank
        self._mooncake_config = None
        self._mooncake_store: EagleMooncakeStore | None = None
        self._dataset: OfflineDataset | None = None
        self._rows_by_id: dict[str, dict[str, Any]] = {}
        self._unpublished_keys: set[str] = set()
        setup_file_logging("offline_replay", rank, group=engine_group)

    def init(self, mooncake_config=None) -> None:
        if mooncake_config is None:
            raise ValueError("OfflineReplayEngine requires a Mooncake configuration")

        from aspec.transfer.mooncake.utils import check_mooncake_master_available

        mooncake_config = dataclasses.replace(
            mooncake_config,
            local_hostname=self.get_node_ip(),
            enable_gpu_direct=False,
        )
        check_mooncake_master_available(
            mooncake_config.master_server_address,
            mooncake_config.metadata_server,
        )
        self._mooncake_config = mooncake_config
        self._dataset = OfflineDataset(self.args.offline_data_path)
        bind_offline_dataset(self._dataset, self.args, self.args.aspec_feature_contract)
        from aspec.feature_contract import FeatureContractRuntime

        contract = self.args.aspec_feature_contract
        self._validation = FeatureContractRuntime(
            contract, role="producer", participant=f"offline_replay_rank_{self.rank}"
        )
        self._validation.validate_producer_binding(
            hidden_size=contract["target_hidden_size"],
            num_hidden_layers=contract["target_num_hidden_layers"],
            aux_hidden_state_layers=contract["aux_hidden_state_layers"],
            store_last_hidden_states="last_hidden_states" in contract["tensors"],
        )
        rows = self._dataset.rows("train") + self._dataset.rows("eval")
        self._rows_by_id = {str(row["data_id"]): row for row in rows}
        if len(self._rows_by_id) != len(rows):
            raise ValueError("Offline dataset contains duplicate data IDs across splits")

        self._mooncake_store = EagleMooncakeStore(mooncake_config)
        self._mooncake_store.setup(device=torch.device("cpu"))
        logger.info(
            "OfflineReplayEngine rank %d initialized with %d records from %s",
            self.rank,
            len(rows),
            self.args.offline_data_path,
        )

    def _replay_one(self, data_id: str) -> dict[str, Any]:
        if self._dataset is None or self._mooncake_store is None:
            raise RuntimeError("OfflineReplayEngine not initialized. Call init() first.")
        row = self._rows_by_id.get(str(data_id))
        if row is None:
            raise KeyError(f"Offline dataset has no record for data_id={data_id!r}")
        record = self._dataset.load(row)
        key = str(uuid.uuid4())
        tensors = {
            name: record.get(name)
            for name in ("input_ids", "hidden_states", "target", "last_hidden_states", "mtp_hidden_states")
        }

        # Dataset.load validated the recorded producer contract. Physical
        # placement and Draft execution flags belong to this replay attempt.
        metadata = {
            "aspec_feature_contract_id": self._validation.contract_id,
            "request_id": key,
            "engine_rank": self.rank,
            "feature_affinity": getattr(self.args, "mooncake_feature_affinity", "any"),
        }
        metadata.update(
            offline_replay=True,
            target_feature_output="offline_replay",
            transport_data_plane="store",
            transport_protocol="tcp",
            source_node_ip=self.get_node_ip(),
        )
        self._validation.validate_metadata(
            tensor_shapes={
                name: tuple(value.shape) for name, value in tensors.items() if value is not None
            },
            tensor_dtypes={
                name: value.dtype for name, value in tensors.items() if value is not None
            },
            metadata=metadata,
        )

        # Record ownership before put: a failed call may already have queued
        # writes without returning metadata for this record.
        self._unpublished_keys.update((f"{key}_hs", f"{key}_ids"))
        if tensors["target"] is not None:
            self._unpublished_keys.add(f"{key}_tgt")
        if tensors["last_hidden_states"] is not None:
            self._unpublished_keys.add(f"{key}_lhs")
        if tensors["mtp_hidden_states"] is not None:
            self._unpublished_keys.add(f"{key}_mhs")
        store_meta = self._mooncake_store.put(
            key=key,
            hidden_states=tensors["hidden_states"],
            input_ids=tensors["input_ids"],
            last_hidden_states=tensors["last_hidden_states"],
            target=tensors["target"],
            **({"mtp_hidden_states": tensors["mtp_hidden_states"]}
               if tensors["mtp_hidden_states"] is not None else {}),
        )
        return {
            "mooncake_key": key,
            "tensor_shapes": store_meta["shapes"],
            "tensor_dtypes": store_meta["dtypes"],
            "packed_loss_mask": record.get("packed_loss_mask"),
            "metadata": metadata,
        }

    def generate(
        self,
        data_id: str | list[str],
        input_ids_ref: ray.ObjectRef | list[torch.Tensor] | None = None,
        packed_loss_mask_list: list[str] | None = None,
        formatted_prompts: list[str] | None = None,
        return_last_hidden_states: bool = False,
        return_logits: bool = True,
        multimodal_inputs: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        del (
            input_ids_ref,
            packed_loss_mask_list,
            formatted_prompts,
            return_last_hidden_states,
            return_logits,
            multimodal_inputs,
        )
        data_ids = data_id if isinstance(data_id, list) else [data_id]
        self._discard_unpublished_outputs()
        outputs = []
        try:
            for item in data_ids:
                outputs.append(self._replay_one(str(item)))
            self._mooncake_store.flush()
        except Exception:
            # No keys escaped to the manager. Drain all writes before deletion;
            # a failed deletion leaves the ownership set intact for retry.
            self._discard_unpublished_outputs()
            raise
        self._unpublished_keys.clear()
        return outputs

    def _discard_unpublished_outputs(self) -> None:
        if self._unpublished_keys:
            self._mooncake_store.discard_unpublished_objects(sorted(self._unpublished_keys))
            self._unpublished_keys.clear()

    def health_check(self, timeout: float = 5.0) -> bool:
        del timeout
        return (
            self._dataset is not None
            and self._mooncake_store is not None
            and not self._unpublished_keys
        )

    def shutdown(self) -> None:
        if self._mooncake_store is not None:
            self._discard_unpublished_outputs()
            self._mooncake_store.close()
            self._mooncake_store = None
        self._dataset = None
        self._rows_by_id.clear()
        logger.info("OfflineReplayEngine rank %d shutdown complete", self.rank)

    def get_status(self) -> dict:
        return {
            "rank": self.rank,
            "engine_type": "offline",
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_initialized": torch.cuda.is_initialized(),
            "engine_gpu_ids": [],
            "num_gpus_per_engine": 0,
            "dataset_identity": getattr(self.args, "offline_dataset_identity", None),
            "initialized": self._dataset is not None,
            "records": len(self._rows_by_id),
            "unpublished_objects": len(self._unpublished_keys),
            "data_path": getattr(self.args, "offline_data_path", None),
        }

    def get_feature_contract_status(self):
        return self._validation.status()
