# Adapted from TorchSpec 6c042a87140a84d13839e341ece2c5c3ada918bc,
# torchspec/offline/saving_actor.py. Adds ASpec payload validation and run evidence.
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

"""Queue consumer that saves inference outputs instead of training on them."""

from __future__ import annotations

import dataclasses
import os

import ray
import torch

from aspec.offline.dataset import OfflineDataset
from aspec.ray.ray_actor import RayActor
from aspec.transfer.mooncake.eagle_store import EagleMooncakeStore
from aspec.work_lifecycle import EXECUTION_ID_KEY


@ray.remote(num_cpus=1, num_gpus=0)
class OfflineSavingActor:
    """Consume the normal training queues and persist their tensors."""

    def __init__(self, output_dir, mooncake_config, controller) -> None:
        self.controller = controller
        self.dataset = OfflineDataset(output_dir, allow_incomplete=True)
        config = dataclasses.replace(
            mooncake_config,
            local_hostname=RayActor.get_node_ip(),
            global_segment_size=0,
            async_put_pool_size=0,
            enable_gpu_direct=False,
        )
        self.store = EagleMooncakeStore(config)
        self.store.setup(device=torch.device("cpu"))
        self.queues = {}

    def get_owner_status(self):
        import psutil

        return {
            "node_id": ray.get_runtime_context().get_node_id(),
            "node_ip": RayActor.get_node_ip(),
            "pid": os.getpid(),
            "created": psutil.Process().create_time(),
        }

    def set_queues(self, train_queue, eval_queue) -> None:
        self.queues = {"train": train_queue, "eval": eval_queue}

    def save_from_queue(self, split: str, count: int = 1) -> int:
        """Save exactly ``count`` items and return the number newly written."""
        written = 0
        for _ in range(count):
            sample = self.queues[split].get()
            if not sample.data_id:
                raise ValueError("Offline saving requires TrainSample.data_id")
            if (sample.metadata or {}).get("vllm_pp_layer_manifest") is not None:
                # Under vLLM pipeline parallelism each stage writes its own
                # per-layer fragments and the base key is never written, so a
                # plain get() here would miss the data and then "clean up" a
                # key that does not exist while every fragment leaks.
                raise NotImplementedError(
                    "Offline saving does not support vLLM pipeline parallelism yet; "
                    f"sample {sample.data_id} carries a per-layer PP manifest. "
                    "Generate offline data with vllm_pp_size=1."
                )
            dtypes = {
                key: getattr(torch, value.replace("torch.", ""))
                if isinstance(value, str)
                else value
                for key, value in (sample.tensor_dtypes or {}).items()
            }
            outcome = "discarded_after_error"
            try:
                output = self.store.get(
                    key=sample.mooncake_key,
                    shapes=sample.tensor_shapes,
                    dtypes=dtypes,
                    device=torch.device("cpu"),
                )
                written += self.dataset.append(
                    split,
                    data_id=sample.data_id,
                    tensors=output.to_tensor_dict(),
                    packed_loss_mask=sample.packed_loss_mask,
                    metadata=sample.metadata,
                )
                outcome = "consumed"
            finally:
                self.store.remove_eagle3_tensors(
                    sample.mooncake_key,
                    has_last_hidden_states="last_hidden_states" in sample.tensor_shapes,
                    has_target="target" in sample.tensor_shapes,
                )
                ray.get(
                    self.controller.release_inference_output.remote(
                        execution_id=sample.metadata[EXECUTION_ID_KEY],
                        data_id=str(sample.data_id),
                        mooncake_key=sample.mooncake_key,
                        participant="offline_saver",
                        outcome=outcome,
                    )
                )
        return written

    def finalize(self, expected_counts):
        self.dataset.finalize(expected_counts)

    def counts(self) -> dict[str, int]:
        return {split: self.dataset.count(split) for split in ("train", "eval")}

    def close(self) -> None:
        self.store.close()
