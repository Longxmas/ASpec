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

import os
from argparse import Namespace
from datetime import timedelta

import ray
import torch.distributed as dist

from aspec import AutoDraftModelConfig
from aspec.config.draft_execution import is_direct_cp_backend
from aspec.models.draft.dflash import DFlashConfig
from aspec.models.draft.dflash2 import DFlash2Config
from aspec.models.draft.dspark import DSparkConfig
from aspec.ray.ray_actor import RayActor
from aspec.utils.distributed import (
    init_direct_cp_groups,
    init_gloo_group,
    init_usp_groups,
)
from aspec.utils.logging import setup_file_logging


def _trainer_class_for_config(draft_model_config):
    if getattr(draft_model_config, "architectures", None) == ["LlamaForCausalLMEagle4"]:
        from aspec.training.eagle4_trainer import Eagle4Trainer

        return Eagle4Trainer
    if getattr(draft_model_config, "architectures", None) == ["NativeMTPDraftModel"]:
        from aspec.training.mtp_trainer import MTPTrainer

        return MTPTrainer
    if isinstance(draft_model_config, DFlash2Config):
        from aspec.training.dflash2_trainer import DFlash2Trainer

        return DFlash2Trainer
    if isinstance(draft_model_config, DSparkConfig):
        from aspec.training.dspark_trainer import DSparkTrainer

        return DSparkTrainer
    if isinstance(draft_model_config, DFlashConfig):
        from aspec.training.dflash_trainer import DFlashTrainer

        return DFlashTrainer
    if getattr(draft_model_config, "architectures", None) in (
        ["LlamaForCausalLMEagle3"], ["Eagle3DeepseekV2ForCausalLM"],
    ):
        from aspec.training.eagle3_trainer import Eagle3Trainer

        return Eagle3Trainer
    raise ValueError(
        "ASpec supports DSpark, DFlash and Llama EAGLE3 draft model configs; "
        f"got {type(draft_model_config).__name__}"
    )


class TrainerActor(RayActor):
    def __init__(self, world_size: int, rank: int, master_addr: str, master_port: int):
        self._world_size = world_size
        self._rank = rank

        self.setup_master(master_addr, master_port, port_range=(20000, 21000))

        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)

        self.setup_gpu()
        setup_file_logging("training", self._rank)

    def init(self, args: Namespace, role: str, mooncake_config=None, with_ref: bool = False) -> int:
        self.args = args

        backend = getattr(args, "distributed_backend", "nccl")
        if getattr(args, "fsdp_cpu_offload", False) and getattr(args, "fsdp_cpu_backend", None):
            cpu_backend = args.fsdp_cpu_backend
            backend = f"cpu:{cpu_backend},cuda:{backend}"

        dist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=getattr(args, "distributed_timeout_minutes", 30)),
        )

        if getattr(args, "attention_backend", None) == "usp":
            init_usp_groups(
                sp_ulysses_size=getattr(args, "sp_ulysses_size", 1),
                sp_ring_size=getattr(args, "sp_ring_size", 1),
            )
        elif is_direct_cp_backend(getattr(args, "attention_backend", None)):
            init_direct_cp_groups(
                getattr(args, "draft_context_parallel_size", 1)
            )

        init_gloo_group()

        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()

        draft_model_config = getattr(args, "draft_model_config_obj", None)
        if draft_model_config is None and getattr(args, "draft_model_config", None):
            draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)

        trainer_class = _trainer_class_for_config(draft_model_config)
        self._trainer = trainer_class(args)

        target_model_path = getattr(args, "target_model_path", None)

        if draft_model_config is not None:
            self._trainer.init_model(
                draft_model_config=draft_model_config,
                target_model_path=target_model_path,
                mooncake_config=mooncake_config,
            )

        return 0

    def train_from_queue(self, step: int, num_batches: int) -> dict:
        return self._trainer.train_from_queue(step, num_batches)

    def flush_pending_train_metrics(self) -> dict:
        return self._trainer.flush_pending_metrics()

    def get_feature_contract_status(self) -> dict:
        return self._trainer.get_feature_contract_status()

    def pop_committed_work(self, optimizer_step: int) -> list[str]:
        return self._trainer.pop_committed_work(optimizer_step)

    def set_train_queue(
        self,
        queue,
        mooncake_config=None,
        per_dp_rank_batch_size: int = 1,
        lifecycle_observer=None,
    ):
        return self._trainer.set_train_queue(
            queue,
            mooncake_config=mooncake_config,
            per_dp_rank_batch_size=per_dp_rank_batch_size,
            lifecycle_observer=lifecycle_observer,
        )

    def discard_inference_outputs(self, outputs: list) -> int:
        return self._trainer.discard_inference_outputs(outputs)

    def close(self) -> None:
        """Release trainer-owned storage and compiler worker pools."""

        self._trainer.close()
        try:
            from torch._inductor.async_compile import shutdown_compile_workers

            shutdown_compile_workers()
        except ImportError:
            pass

    def get_global_step(self) -> int:
        return self._trainer.global_step

    def get_status(self) -> dict:
        """Return the rank placement actually selected by Ray and CUDA."""
        import sys

        import psutil
        import torch

        import aspec
        from aspec.inference.target_runtime import code_identity, dependency_versions

        return {
            "rank": self._rank,
            "world_size": self._world_size,
            "node_ip": self.get_node_ip(),
            "node_id": ray.get_runtime_context().get_node_id(),
            "pid": os.getpid(),
            "created": psutil.Process().create_time(),
            "assigned_gpu_ids": [int(float(value)) for value in ray.get_gpu_ids()],
            "local_gpu_id": int(os.environ["LOCAL_RANK"]),
            "initialized": hasattr(self, "_trainer"),
            "runtime": {**code_identity(), "python": sys.executable, "prefix": sys.prefix,
                        "aspec": aspec.__file__, "ray": ray.__file__,
                        "torch": torch.__version__, "cuda": torch.version.cuda,
                        "dependencies": dependency_versions("nvidia-cutlass-dsl", "quack-kernels")},
        }

    def save_model(self, step: int, force_sync: bool = False) -> None:
        self._trainer.save_model(step, force_sync)

    def save_draft_model_for_serving(self, output_dir: str) -> None:
        self._trainer.save_draft_model_for_serving(output_dir)

    def set_vocab_buffers(self, d2t, t2d) -> None:
        if hasattr(self._trainer, "draft_model") and hasattr(
            self._trainer.draft_model, "set_vocab_buffers"
        ):
            self._trainer.draft_model.set_vocab_buffers(d2t, t2d)
        else:
            raise AttributeError(
                "set_vocab_buffers called but draft model does not support vocab pruning. "
                "DFlash training should not use vocab pruning — check train_entry config."
            )

    def set_eval_queue(
        self,
        queue,
        mooncake_config=None,
        per_dp_rank_batch_size: int = 1,
        lifecycle_observer=None,
    ):
        return self._trainer.set_eval_queue(
            queue,
            mooncake_config=mooncake_config,
            per_dp_rank_batch_size=per_dp_rank_batch_size,
            lifecycle_observer=lifecycle_observer,
        )

    def cache_eval_samples(self, count: int) -> int:
        return self._trainer.cache_eval_samples(count)

    def clear_eval_cache(self):
        self._trainer._eval_cache = []

    def save_eval_cache(self, cache_dir: str) -> None:
        return self._trainer.save_eval_cache(cache_dir)

    def load_eval_cache(self, cache_dir: str, identity=None) -> int:
        return self._trainer.load_eval_cache(cache_dir, identity=identity)

    def eval_from_cache(self) -> dict:
        from aspec.training.evaluation import evaluation_rng, training_attention_evidence

        was_training = self._trainer.model.training
        try:
            with (evaluation_rng(self._trainer.args.seed + self._trainer.dp_rank),
                  training_attention_evidence(self._trainer.draft_model)):
                return self._trainer.eval_from_cache()
        finally:
            self._trainer.model.train(was_training)
