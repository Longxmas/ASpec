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

"""
Sgl Ray actor engine for distributed deployment across multiple nodes.

This wraps sgl.Engine (patched sglang) as a Ray actor with placement group support,
parallel to HFEngine. Uses sglang's spec_training mode with mooncake storage.

Accepts pre-tokenized input_ids and loss_mask from batch preprocessing.
"""

import json
import os
from typing import Any

import ray
import sglang as sgl
import torch
from omegaconf import DictConfig, OmegaConf

from aspec.config.draft_execution import (
    is_direct_cp_backend,
)
from aspec.feature_contract import FeatureContractRuntime
from aspec.inference.concurrency import (
    effective_engine_concurrency,
    required_target_prefill_tokens,
)
from aspec.inference.context import (
    compile_sglang_context_contract,
    configure_sglang_context_override,
)
from aspec.inference.engine.base import InferenceEngine
from aspec.inference.engine.sgl_engine_decode import SglDecodeEngineMixin
from aspec.inference.service_protocol import validate_target_tp
from aspec.ray.ray_actor import RayActor
from aspec.transfer.mooncake.eagle_store import HIDDEN_STATES_STORAGE_DTYPE
from aspec.utils.logging import logger, setup_file_logging
from aspec.utils.misc import get_default_eagle3_aux_layer_ids
from aspec.utils.timeline import export_timeline_env, get_timeline

# Keys that users might plausibly put in extra_args but are managed by
# ASpec.  Used only to emit a warning — the actual protection comes
# from the .update() ordering in init() which overwrites extra_args.
_PROTECTED_ENGINE_KEYS = frozenset(
    {
        "model_path",
        "tp_size",
        "mem_fraction_static",
        "nnodes",
        "port",
        "nccl_port",
        "dist_init_addr",
        "dist_timeout",
        "enable_multimodal",
        "allow_auto_truncate",
        "context_length",
    }
)


_SGLANG_UNBOUNDED_PREFILL_TOKENS = (1 << 31) - 1


def sglang_training_max_prefill_tokens(configured: int | None) -> int:
    """Translate an unset/-1 prefill budget without disabling the adder."""
    if configured is None or configured == -1:
        return _SGLANG_UNBOUNDED_PREFILL_TOKENS
    if configured < 1:
        raise ValueError("SGLang max_prefill_tokens must be positive or -1")
    return configured


def validate_sglang_profiled_capacity(
    profiled_max_total_tokens: int | None,
    required_tokens: int | None,
) -> None:
    """Reject a profiled token pool smaller than the admitted workload."""
    if profiled_max_total_tokens is None or required_tokens is None:
        return
    if int(profiled_max_total_tokens) < required_tokens:
        raise RuntimeError(
            "SGLang profiled max_total_tokens cannot satisfy the requested "
            "Target-prefill concurrency: "
            f"profiled={profiled_max_total_tokens}, required={required_tokens}"
        )


_FEATURE_TRANSPORT_ENV_KEYS = (
    "ASPEC_USP_SHARDED_MOONCAKE",
    "ASPEC_USP_SP_SIZE",
    "ASPEC_USP_RING_SIZE",
    "ASPEC_USP_TTT_LENGTH",
    "ASPEC_USP_MAX_SEQ_LENGTH",
    "ASPEC_DIRECT_CP_SHARDED_MOONCAKE",
    "ASPEC_DIRECT_CP_SIZE",
    "ASPEC_DIRECT_CP_MAX_SEQ_LENGTH",
    "ASPEC_DIRECT_CP_ALIGNMENT",
    "ASPEC_TARGET_PREFILL_MEMORY_OPTIMIZED",
)


def _configure_usp_sharded_mooncake_env(args: Any, max_seq_length: int | None) -> None:
    values: dict[str, str] = {}
    if getattr(args, "attention_backend", None) == "usp":
        sp_ring_size = getattr(args, "sp_ring_size", 1)
        values = {
            "ASPEC_USP_SHARDED_MOONCAKE": "1",
            "ASPEC_USP_SP_SIZE": str(getattr(args, "sp_ulysses_size", 1) * sp_ring_size),
            "ASPEC_USP_RING_SIZE": str(sp_ring_size),
            "ASPEC_USP_TTT_LENGTH": str(getattr(args, "ttt_length", 1)),
        }
        if max_seq_length is not None:
            values["ASPEC_USP_MAX_SEQ_LENGTH"] = str(max_seq_length)
    elif is_direct_cp_backend(getattr(args, "attention_backend", None)):
        values = {
            "ASPEC_DIRECT_CP_SHARDED_MOONCAKE": "1",
            "ASPEC_DIRECT_CP_SIZE": str(
                getattr(args, "draft_context_parallel_size", 1)
            ),
            "ASPEC_DIRECT_CP_ALIGNMENT": str(
                getattr(args, "draft_context_shard_alignment", 1)
            ),
        }
        if max_seq_length is not None:
            values["ASPEC_DIRECT_CP_MAX_SEQ_LENGTH"] = str(max_seq_length)

    if getattr(args, "target_prefill_memory_optimized", False):
        values["ASPEC_TARGET_PREFILL_MEMORY_OPTIMIZED"] = "1"

    for name in _FEATURE_TRANSPORT_ENV_KEYS:
        if name in values:
            os.environ[name] = values[name]
        else:
            os.environ.pop(name, None)


class SglEngine(SglDecodeEngineMixin, InferenceEngine, RayActor):
    """Ray actor wrapper for sgl.Engine with distributed deployment support.

    Uses patched sglang's spec_training mode to generate training data and store
    it in mooncake. Returns mooncake keys instead of tensors for efficient
    distributed training.

    Accepts pre-tokenized input_ids and loss_mask instead of raw prompts.
    """

    def __init__(
        self,
        args,
        rank: int,
        base_gpu_id: int | None = None,
        num_gpus_per_engine: int = 1,
        node_rank: int = 0,
        engine_group: int = 0,
    ):
        """Store configuration but don't load model yet.

        Args:
            args: Configuration arguments.
            rank: Engine rank for distributed setup.
            base_gpu_id: Base GPU ID from placement group.
            num_gpus_per_engine: Number of GPUs this engine uses (for TP).
            node_rank: Must be zero for a node-local TP group.
            engine_group: Group index to disambiguate multiple engine groups on the same node.
        """
        validate_target_tp(getattr(args, "sglang_nnodes", 1), node_rank)
        self.args = args
        self.rank = rank
        self.base_gpu_id = base_gpu_id
        self.num_gpus_per_engine = num_gpus_per_engine
        self.node_rank = node_rank
        self._engine = None
        self._mooncake_config = None
        self._mooncake_store = None
        self._hidden_size = None
        self._num_hidden_layers = None
        self._capacity_status: dict[str, Any] = {}
        self._producer_transport_observation = {
            "endpoints": set(),
            "source_descriptors": 0,
            "source_payload_bytes": 0,
        }
        self.timeline = get_timeline(args, role="sglang_engine", rank=rank)
        self._store_last_hidden_states = True
        feature_contract = getattr(args, "aspec_feature_contract", None)
        self._feature_contract_runtime = (
            FeatureContractRuntime(
                feature_contract,
                role="producer",
                participant=f"sglang_engine_rank_{rank}",
            )
            if feature_contract is not None
            else None
        )
        self.local_gpu_id = None
        self.engine_group = engine_group
        setup_file_logging("inference", self.rank, group=engine_group)

    def init(
        self,
        mooncake_config=None,
        dist_init_addr: str | None = None,
        pre_allocated_port: int | None = None,
    ) -> None:
        """Initialize the sgl.Engine on the allocated GPU.

        This is called after the Ray actor is scheduled on a node.

        Args:
            mooncake_config: MooncakeConfig object for distributed storage.
            dist_init_addr: Legacy control clients may pass None for a node-local group.
            pre_allocated_port: Base port pre-allocated by the factory.
        """
        if dist_init_addr is not None:
            raise ValueError("A node-local Target TP group does not take a dist_init_addr")
        if self.base_gpu_id is not None:
            self.local_gpu_id = self.setup_gpu(self.base_gpu_id)
            logger.info(
                f"SglEngine rank {self.rank}: base_gpu_id={self.base_gpu_id}, "
                f"using local GPU {self.local_gpu_id}"
            )

        self._mooncake_config = mooncake_config
        if mooncake_config is not None:
            logger.info(f"SglEngine rank {self.rank}: received mooncake_config={mooncake_config}")

            local_ip = self.get_node_ip()

            mooncake_config.local_hostname = local_ip
            mooncake_config.export_env()

            logger.info(
                f"SglEngine rank {self.rank}: mooncake env vars set - "
                f"local_hostname={local_ip}, "
                f"master_server={mooncake_config.master_server_address}, "
                f"metadata_server={mooncake_config.metadata_server}, "
                f"protocol={mooncake_config.protocol}, "
                f"device_name={mooncake_config.device_name}"
            )

            from aspec.transfer.mooncake.utils import (
                check_mooncake_master_available,
            )

            check_mooncake_master_available(
                mooncake_config.master_server_address,
                mooncake_config.metadata_server,
            )

        self._store_last_hidden_states = getattr(self.args, "store_last_hidden_states", True)
        export_timeline_env(self.args)
        os.environ["ASPEC_ENGINE_RANK"] = str(self.rank)
        # Shared SGLang checkouts may also serve AngelSpec. Bind the producer
        # to this framework's Store rather than inheriting a legacy import.
        os.environ["SGLANG_SPEC_TRAINING_PACKAGE"] = "aspec"

        # Get configuration
        mem_fraction = getattr(self.args, "sglang_mem_fraction_static", 0.8)
        pp_size = getattr(self.args, "sglang_pp_size", 1)
        if self.args.aux_hidden_states_layers is not None:
            self.aux_hidden_state_layer_ids = self.args.aux_hidden_states_layers
        else:
            self.aux_hidden_state_layer_ids = get_default_eagle3_aux_layer_ids(
                self.args.target_model_path,
                trust_remote_code=getattr(self.args, "trust_remote_code", False),
            )
            if self.rank == 0:
                logger.info(
                    f"Using default aux hidden state layer ids: {self.aux_hidden_state_layer_ids}"
                )

        if pp_size != 1:
            raise ValueError(f"pp_size must be 1, got {pp_size}")

        tp_size = self.num_gpus_per_engine
        configured_tp_size = getattr(self.args, "sglang_tp_size", None)
        if configured_tp_size is not None and configured_tp_size != tp_size:
            raise ValueError(
                f"sglang_tp_size ({configured_tp_size}) must equal "
                f"num_gpus_per_engine ({tp_size})"
            )

        logger.info(
            f"SglEngine rank {self.rank}: BEFORE init - "
            f"base_gpu_id={self.base_gpu_id}, num_gpus={self.num_gpus_per_engine}, "
            f"tp_size={tp_size}, pp_size={pp_size}, "
            f"aux_hidden_state_layer_ids={self.aux_hidden_state_layer_ids}"
        )

        # Build engine kwargs - base config for spec_training mode.
        # Overridable defaults (e.g. log_level) are set first so that
        # extra_args can override them; protected keys are set after
        # extra_args and cannot be overridden.
        engine_kwargs = {
            "log_level": "warning",
        }

        # Apply extra_args (can override defaults above, but not protected keys)
        extra_args = getattr(self.args, "sglang_extra_args", None)
        if extra_args:
            if isinstance(extra_args, DictConfig):
                extra = OmegaConf.to_container(extra_args, resolve=True)
            else:
                extra = dict(extra_args) if not isinstance(extra_args, dict) else extra_args
            blocked = extra.keys() & _PROTECTED_ENGINE_KEYS
            if blocked:
                logger.warning(
                    f"sglang extra_args contains protected keys that will be ignored: "
                    f"{sorted(blocked)}. These are managed internally by ASpec."
                )
                extra = {k: v for k, v in extra.items() if k not in _PROTECTED_ENGINE_KEYS}
            engine_kwargs.update(extra)

        engine_kwargs["max_prefill_tokens"] = sglang_training_max_prefill_tokens(
            engine_kwargs.get("max_prefill_tokens")
        )

        required_tokens = None
        if getattr(self.args, "target_prefill_memory_optimized", False):
            max_sequence_length = getattr(self.args, "max_seq_length", None)
            if max_sequence_length is None:
                raise ValueError(
                    "target_prefill_memory_optimized requires max_seq_length"
                )
            source_slots = getattr(
                self.args, "mooncake_source_gpu_ring_slots", None
            )
            if source_slots is None:
                source_slots = getattr(self.args, "mooncake_gpu_ring_slots", 2)
            cp_size = int(getattr(self.args, "draft_context_parallel_size", 1))
            slots_per_sample = (
                cp_size
                if is_direct_cp_backend(getattr(self.args, "attention_backend", None))
                else 1
            )
            effective_concurrency = effective_engine_concurrency(
                configured=int(getattr(self.args, "max_concurrent_batches", 1)),
                batch_size=int(getattr(self.args, "inference_batch_size", 1)),
                source_slots=int(source_slots),
                slots_per_sample=slots_per_sample,
            )
            required_tokens = required_target_prefill_tokens(
                max_sequence_length=int(max_sequence_length),
                batch_size=int(getattr(self.args, "inference_batch_size", 1)),
                max_concurrent_batches=effective_concurrency,
            )
            configured_tokens = engine_kwargs.get("max_total_tokens")
            if configured_tokens is None:
                engine_kwargs["max_total_tokens"] = required_tokens
            elif int(configured_tokens) < required_tokens:
                raise ValueError(
                    "sglang.extra_args.max_total_tokens is too small for "
                    "target_prefill_memory_optimized: "
                    f"configured={configured_tokens}, require at least {required_tokens}"
                )

        # SGLang's patched scheduler reads these process env vars when writing
        # Mooncake training tensors.
        max_seq_length = getattr(self.args, "max_seq_length", None)
        context_contract = getattr(self.args, "aspec_context_contract", None)
        if context_contract is None and max_seq_length is not None:
            context_contract = compile_sglang_context_contract(
                max_sequence_length=int(max_seq_length),
                context_parallel_size=int(
                    getattr(self.args, "draft_context_parallel_size", 1)
                ),
                model_config=None,
                native_context_length=getattr(
                    self.args, "native_context_length", None
                ),
                allow_context_length_override=bool(
                    getattr(self.args, "sglang_allow_context_length_override", False)
                ),
            )
        if context_contract is None:
            context_length = None
            context_override_effective = configure_sglang_context_override({})
        else:
            context_length = int(context_contract["backend_context_length"])
            context_override_effective = configure_sglang_context_override(
                context_contract
            )
        _configure_usp_sharded_mooncake_env(self.args, max_seq_length)

        engine_kwargs.update(
            {
                "model_path": self.args.target_model_path,
                "disable_radix_cache": True,
                "enable_return_hidden_states": True,
                "enable_aux_hidden_states": True,
                "aux_hidden_state_layer_ids": self.aux_hidden_state_layer_ids,
                "enable_spec_training_mooncake": True,
                "tp_size": tp_size,
                "pp_size": pp_size,
                "base_gpu_id": self.local_gpu_id,
                "gpu_id_step": 1,
                "mem_fraction_static": mem_fraction,
                "enable_multimodal": getattr(self.args, "sglang_enable_multimodal", False),
                "trust_remote_code": getattr(self.args, "trust_remote_code", False),
                "chunked_prefill_size": -1,
                "allow_auto_truncate": True,
                **({"context_length": context_length} if context_length else {}),
                **(
                    {"spec_training_store_last_hidden_states": False}
                    if not self._store_last_hidden_states
                    else {}
                ),
            }
        )

        # Decode mode: add speculative decoding and performance tuning params
        self._train_with_decode = getattr(self.args, "train_with_decode", False)
        if self._train_with_decode:
            self._build_decode_engine_kwargs(engine_kwargs)
        else:
            engine_kwargs["disable_cuda_graph"] = True

        if pre_allocated_port is None:
            raise ValueError(
                f"SglEngine rank {self.rank}: pre_allocated_port is required "
                "(ports must be pre-allocated by the factory)"
            )
        engine_kwargs["port"] = pre_allocated_port
        engine_kwargs["nccl_port"] = pre_allocated_port + 1

        engine_kwargs["nnodes"] = 1
        engine_kwargs["node_rank"] = 0

        from aspec.inference.sglang_v0521 import TrainingEngine, training_engine_kwargs

        engine_kwargs = training_engine_kwargs(engine_kwargs, self.args)
        from pathlib import Path

        from aspec.inference.diagnostics import capture_backend_output

        output_dir = getattr(self.args, "output_dir", None)
        backend_log = (Path(output_dir) / "backend_logs" /
                       f"sglang-{self.engine_group}-{self.node_rank}-{self.rank}.log") if output_dir else None
        if backend_log is not None:
            logger.info("SGLang backend output: %s", backend_log)
        with capture_backend_output(backend_log):
            self._engine = TrainingEngine(sgl.Engine(**engine_kwargs))

        scheduler_info = None
        init_result = getattr(self._engine, "_scheduler_init_result", None)
        scheduler_infos = getattr(init_result, "scheduler_infos", None)
        if scheduler_infos:
            scheduler_info = scheduler_infos[0]
        profiled_max_total_tokens = (
            scheduler_info.get("max_total_num_tokens")
            if isinstance(scheduler_info, dict)
            else None
        )
        validate_sglang_profiled_capacity(
            profiled_max_total_tokens, required_tokens
        )
        self._capacity_status = {
            "profiled_max_total_tokens": profiled_max_total_tokens,
            "required_max_total_tokens": required_tokens,
            "max_prefill_tokens": engine_kwargs["max_prefill_tokens"],
            "requested_max_running_requests": engine_kwargs.get(
                "max_running_requests"
            ),
            "context_length": context_length,
            "context": (
                {
                    **dict(context_contract),
                    "context_override": {
                        **dict(context_contract["context_override"]),
                        "effective": context_override_effective,
                    },
                }
                if context_contract is not None
                else None
            ),
        }
        logger.info(
            f"SglEngine rank {self.rank}: token capacity {self._capacity_status}"
        )

        # Get hidden size from model config
        self._hidden_size = self._get_hidden_size_from_engine()
        if self._feature_contract_runtime is not None:
            self._feature_contract_runtime.validate_producer_binding(
                hidden_size=self._hidden_size,
                num_hidden_layers=self._num_hidden_layers,
                aux_hidden_state_layers=list(self.aux_hidden_state_layer_ids),
                store_last_hidden_states=self._store_last_hidden_states,
            )

        if hasattr(self._engine, "model_config"):
            model_config = self._engine.model_config
            logger.info(
                f"SglEngine rank {self.rank}: model_config attributes: "
                f"{[attr for attr in dir(model_config) if not attr.startswith('_')]}"
            )
            logger.info(
                f"SglEngine rank {self.rank}: model_config.hidden_size = "
                f"{getattr(model_config, 'hidden_size', 'NOT_FOUND')}"
            )
            logger.info(
                f"SglEngine rank {self.rank}: model_config.num_hidden_layers = "
                f"{getattr(model_config, 'num_hidden_layers', 'NOT_FOUND')}"
            )

        logger.info(
            f"SglEngine rank {self.rank}: initialized from {self.args.target_model_path} "
            f"(tp_size={tp_size}, aux_layers={self.aux_hidden_state_layer_ids}, "
            f"hidden_size={self._hidden_size})"
        )

    @staticmethod
    def _extract_image_data(multimodal_inputs: list[dict] | None) -> list | None:
        """Extract image_data list from multimodal_inputs for sgl.Engine.

        Returns a list-of-lists so that SGLang's _normalize_image_data always
        takes the "already a list of lists" branch.  Using [] instead of None
        for imageless requests prevents the normalizer from misdetecting the
        format when the first request in the batch has no images.
        """
        if not multimodal_inputs:
            return None
        image_data = []
        has_images = False
        for mm_input in multimodal_inputs:
            if mm_input and mm_input.get("images"):
                image_data.append(mm_input["images"])
                has_images = True
            else:
                image_data.append([])
        return image_data if has_images else None

    async def generate(
        self,
        data_id: str | list[str],
        input_ids_ref: ray.ObjectRef | list[torch.Tensor] | None = None,
        packed_loss_mask_list: list[str] | None = None,
        formatted_prompts: list[str] | None = None,
        return_last_hidden_states: bool = False,
        return_logits: bool = True,
        multimodal_inputs: list[dict] | None = None,
        request_ids: str | list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate training data using spec_training mode.

        Accepts either pre-tokenized input_ids or formatted prompt strings.
        Exactly one of input_ids_ref or formatted_prompts must be set.

        Args:
            data_id: Data ID(s) for the batch.
            input_ids_ref: Ray ObjectRef or list of input_ids tensors.
            packed_loss_mask_list: List of packed loss_mask strings (e.g. "2,3,2,2,1").
            formatted_prompts: List of already chat-template-formatted prompt strings.
            return_last_hidden_states: Whether to return last hidden states (ignored, always in mooncake).
            return_logits: Whether to return target logits (ignored, always in mooncake).

        Returns:
            List of dicts with mooncake_key and tensor metadata.
        """
        if self._engine is None:
            raise RuntimeError("SglEngine not initialized. Call init() first.")

        if (input_ids_ref is None) == (formatted_prompts is None):
            raise ValueError("Exactly one of input_ids_ref or formatted_prompts must be set")

        use_prompts = formatted_prompts is not None

        if use_prompts:
            batch_size = len(formatted_prompts)
        else:
            if isinstance(input_ids_ref, ray.ObjectRef):
                input_ids_list = ray.get(input_ids_ref)
            else:
                input_ids_list = input_ids_ref
            batch_size = len(input_ids_list)

        if isinstance(data_id, str):
            data_ids = [f"{data_id}_{i}" for i in range(batch_size)]
        elif len(data_id) == batch_size:
            data_ids = data_id
        else:
            raise ValueError(
                f"data_id length {len(data_id)} does not match batch size {batch_size}"
            )

        if request_ids is None:
            resolved_request_ids = [None] * batch_size
        elif isinstance(request_ids, str):
            resolved_request_ids = [
                f"{request_ids}_{index}" for index in range(batch_size)
            ]
        elif len(request_ids) == batch_size:
            resolved_request_ids = request_ids
        else:
            raise ValueError(
                f"request_ids length {len(request_ids)} does not match "
                f"batch size {batch_size}"
            )

        if use_prompts:
            logger.debug(
                f"SglEngine rank {self.rank}: prompt mode processing data_ids={data_ids}, "
                f"num_prompts={len(formatted_prompts)}"
            )
            # loss mask is computed during training time based on input_ids.
            engine_kwargs = {
                "prompt": formatted_prompts,
                "spec_training_data_id": data_ids,
                "sampling_params": {"max_new_tokens": 0},
                "return_hidden_states": True,
            }
        else:
            input_ids_list_of_lists = []
            for ids in input_ids_list:
                if ids.dim() == 2 and ids.shape[0] == 1:
                    ids = ids.squeeze(0)
                elif ids.dim() > 2:
                    raise ValueError(f"Unexpected input_ids shape: {ids.shape}")
                input_ids_list_of_lists.append(ids.tolist())

            logger.debug(
                f"SglEngine rank {self.rank}: processing data_ids={data_ids}, "
                f"seq_lengths={[len(ids) for ids in input_ids_list_of_lists]}"
            )
            engine_kwargs = {
                "input_ids": input_ids_list_of_lists,
                "spec_training_data_id": data_ids,
                "packed_loss_mask": packed_loss_mask_list,
                "sampling_params": {"max_new_tokens": 0},
                "return_hidden_states": True,
            }

        image_data = self._extract_image_data(multimodal_inputs)
        if image_data is not None:
            engine_kwargs["image_data"] = image_data
        if any(request_id is not None for request_id in resolved_request_ids):
            engine_kwargs["rid"] = resolved_request_ids

        async_generate = getattr(self._engine, "async_generate", None)
        if async_generate is None:
            raise RuntimeError(
                "The selected SGLang runtime does not expose async_generate; "
                "ASpec's concurrent Target pipeline requires the admitted "
                "SGLang revision"
            )
        with self.timeline.span(
            "sglang_generate",
            nvtx=True,
            request_ids=resolved_request_ids,
            engine_rank=self.rank,
            batch_size=batch_size,
            data_ids=data_ids,
        ):
            results = await async_generate(**engine_kwargs)

        # Extract mooncake keys and construct shapes based on actual sequence length
        outputs = []
        for i, result in enumerate(results):
            store_keys = result["meta_info"].get("spec_training_mooncake_store_keys", [])
            if not store_keys:
                logger.error(
                    f"SglEngine rank {self.rank}: ERROR: No mooncake keys returned for "
                    f"data_id={data_ids[i]}. Training may be corrupted."
                )
                continue

            self._observe_transfer_descriptors(store_keys)

            logger.debug(
                f"SglEngine rank {self.rank}: result meta_info keys: {list(result['meta_info'].keys())}"
            )

            for key in store_keys:
                seq_len = result["meta_info"].get("prompt_tokens")
                if seq_len is None:
                    if use_prompts:
                        raise RuntimeError(
                            f"SglEngine rank {self.rank}: 'prompt_tokens' missing from "
                            f"meta_info for data_id={data_ids[i]}. The engine must report "
                            f"prompt_tokens when using formatted_prompts (defer_tokenization mode)."
                        )
                    else:
                        seq_len = len(input_ids_list_of_lists[i])

                tensor_shapes = self._get_tensor_shapes(seq_len)
                logger.debug(
                    f"SglEngine rank {self.rank}: mooncake_key={key}, seq_len={seq_len}, "
                    f"tensor_shapes={tensor_shapes}"
                )

                tensor_dtypes = self._get_tensor_dtypes()
                metadata = {
                    "request_id": resolved_request_ids[i],
                    "engine_rank": self.rank,
                    "source_node_ip": self.get_node_ip(),
                    "feature_affinity": getattr(
                        self.args, "mooncake_feature_affinity", "any"
                    ),
                }
                if self._feature_contract_runtime is not None:
                    metadata["aspec_feature_contract_id"] = (
                        self._feature_contract_runtime.contract_id
                    )
                if getattr(self.args, "attention_backend", None) == "usp":
                    metadata["usp_sharded"] = True
                elif is_direct_cp_backend(
                    getattr(self.args, "attention_backend", None)
                ):
                    cp_size = int(
                        getattr(
                            self.args,
                            "draft_context_parallel_size",
                            1,
                        )
                    )
                    feature_layout = "context_sharded" if cp_size > 1 else "full"
                    metadata.update(
                        {
                            "direct_cp_sharded": True,
                            "direct_cp_size": cp_size,
                            "feature_layout": feature_layout,
                            "draft_attention_backend": getattr(
                                self.args, "attention_backend", None
                            ),
                            "draft_context_parallel_size": cp_size,
                            "draft_feature_layout": feature_layout,
                        }
                    )
                metadata["target_feature_output"] = (
                    "direct_to_transport"
                    if getattr(
                        self.args, "target_prefill_memory_optimized", False
                    )
                    else "buffered_then_transport"
                )
                metadata["transport_data_plane"] = getattr(
                    self.args, "mooncake_data_plane", "store"
                )
                metadata["transport_protocol"] = getattr(
                    self.args, "mooncake_protocol", "tcp"
                )
                if self._feature_contract_runtime is not None:
                    self._feature_contract_runtime.validate_metadata(
                        tensor_shapes=tensor_shapes,
                        tensor_dtypes=tensor_dtypes,
                        metadata=metadata,
                    )

                output = {
                    "mooncake_key": key,
                    "tensor_shapes": tensor_shapes,
                    "tensor_dtypes": tensor_dtypes,
                }
                output["metadata"] = metadata
                outputs.append(output)
                self.timeline.event(
                    "target_features_ready",
                    data_id=data_ids[i],
                    request_id=resolved_request_ids[i],
                    mooncake_key=key,
                    engine_rank=self.rank,
                    sequence_length=seq_len,
                    target_feature_output=metadata["target_feature_output"],
                    tensor_shapes=tensor_shapes,
                )

        logger.debug(
            f"SglEngine rank {self.rank}: generated {len(outputs)} mooncake keys "
            f"for data_ids={data_ids}"
        )
        return outputs

    def health_check(self, timeout: float = 5.0) -> bool:
        """Check if the engine is healthy."""
        return self._engine is not None

    async def shutdown(self) -> None:
        """Drain Target-prefill writes before closing their backing storage."""
        engine = self._engine
        try:
            if engine is not None:
                if getattr(self.args, "target_prefill_memory_optimized", False):
                    try:
                        result = await engine.tokenizer_manager.flush_cache(
                            timeout_s=10.0
                        )
                        if not result.success:
                            logger.warning(
                                "SglEngine rank %s: Target-prefill source drain failed: %s",
                                self.rank,
                                result.message,
                            )
                    except Exception:
                        logger.warning(
                            "SglEngine rank %s: Target-prefill source drain failed",
                            self.rank,
                            exc_info=True,
                        )
                await engine.async_shutdown()
        finally:
            self._engine = None
            if self._mooncake_store is not None:
                try:
                    self._mooncake_store.close()
                finally:
                    self._mooncake_store = None
        logger.info(f"SglEngine rank {self.rank}: shutdown complete")

    def get_status(self) -> dict:
        """Get engine status."""
        return {
            "rank": self.rank,
            "initialized": self._engine is not None,
            "node_ip": self.get_node_ip(),
            "pid": os.getpid(),
            "base_gpu_id": self.base_gpu_id,
            "engine_gpu_ids": (
                list(range(self.base_gpu_id, self.base_gpu_id + self.num_gpus_per_engine))
                if self.base_gpu_id is not None
                else []
            ),
            "local_gpu_id": self.local_gpu_id,
            "num_gpus_per_engine": self.num_gpus_per_engine,
            "hidden_size": self._hidden_size,
            "token_capacity": dict(self._capacity_status),
            "target_prefill_memory_optimized": bool(
                getattr(self.args, "target_prefill_memory_optimized", False)
            ),
        }

    def get_feature_contract_status(self) -> dict:
        """Return producer-side contract observations for Evidence."""
        if self._feature_contract_runtime is None:
            return {
                "participant": f"sglang_engine_rank_{self.rank}",
                "role": "producer",
                "contract_id": None,
                "state": "not_configured",
            }
        status = self._feature_contract_runtime.status()
        if self._mooncake_store is not None:
            transport = self._mooncake_store.get_p2p_status()
            if transport is not None:
                status["transport"] = transport
        elif self._producer_transport_observation["source_descriptors"] > 0:
            observation = self._producer_transport_observation
            status["transport"] = {
                "state": "active",
                "activation": "descriptor_observed",
                "data_plane": "transfer_engine",
                "protocol": getattr(self.args, "mooncake_protocol", None),
                "strict_transport": not bool(
                    getattr(self.args, "mooncake_allow_host_fallback", True)
                ),
                "role": "source",
                "endpoint": sorted(observation["endpoints"]),
                "run_id": getattr(self.args, "aspec_run_id", None),
                "attempt_id": getattr(self.args, "aspec_attempt_id", None),
                "metrics_scope": "emitted_generation_checked_descriptors",
                "metrics": {
                    "source_descriptors": observation["source_descriptors"],
                    "source_payload_bytes": observation["source_payload_bytes"],
                    "transfer_failures": 0,
                    # Scheduler-internal ring wait is not exported across the
                    # SGLang process boundary. Keep it unknown rather than
                    # reporting a false zero.
                    "ring_wait_seconds": None,
                },
            }
        return status

    async def prepare_feature_contract_status(self) -> dict:
        """Drain scheduler-owned source rings and return their real status.

        The Mooncake source endpoint lives in SGLang's scheduler subprocess,
        not in this Ray actor.  The ASpec SGLang patch returns a structured
        source status in ``FlushCacheReqOutput.message`` after the scheduler is
        idle and every published source lease has reached a terminal state.
        Descriptor-derived counters remain useful during a run, but are not a
        sufficient terminal ownership observation.
        """

        status = self.get_feature_contract_status()
        if (
            self._engine is None
            or getattr(self.args, "mooncake_data_plane", "store")
            != "transfer_engine"
        ):
            return status

        drain_seconds = max(
            float(
                getattr(
                    self.args,
                    "mooncake_get_retry_max_wait_seconds",
                    60.0,
                )
            ),
            5.0,
        )
        result = await self._engine.tokenizer_manager.flush_cache(
            timeout_s=drain_seconds + 5.0
        )
        if not result.success:
            raise RuntimeError(
                "SGLang scheduler could not drain ASpec source transport: "
                f"{result.message or 'no status returned'}"
            )
        try:
            transport = json.loads(result.message)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "SGLang scheduler did not return structured ASpec source "
                "transport status"
            ) from exc
        if not isinstance(transport, dict) or transport.get("role") != "source":
            raise RuntimeError(
                "SGLang scheduler returned invalid ASpec source transport status"
            )
        status["transport"] = transport
        return status

    def _observe_transfer_descriptors(self, store_keys: list[str]) -> None:
        """Record source-side facts carried across the SGLang process boundary."""

        if getattr(self.args, "mooncake_data_plane", "store") != "transfer_engine":
            return

        from aspec.transfer.mooncake.p2p import p2p_descriptors_from_key

        expected_run_id = getattr(self.args, "aspec_run_id", None)
        expected_attempt_id = getattr(self.args, "aspec_attempt_id", None)
        for key in store_keys:
            try:
                descriptors = p2p_descriptors_from_key(key)
            except ValueError as exc:
                key_preview = repr(key)
                if len(key_preview) > 160:
                    key_preview = f"{key_preview[:157]}..."
                raise ValueError(
                    "SGLang returned an invalid TransferEngine key: "
                    f"engine_rank={self.rank}, key_type={type(key).__name__}, "
                    f"key={key_preview}"
                ) from exc
            for descriptor in descriptors:
                if (
                    descriptor.run_id != expected_run_id
                    or descriptor.attempt_id != expected_attempt_id
                ):
                    raise RuntimeError(
                        "SGLang returned a TransferEngine descriptor from a stale "
                        "or foreign run attempt"
                    )
                self._producer_transport_observation["endpoints"].add(
                    descriptor.source_endpoint
                )
                self._producer_transport_observation["source_descriptors"] += 1
                self._producer_transport_observation["source_payload_bytes"] += int(
                    descriptor.payload_bytes
                )

    def _get_hidden_size_from_engine(self) -> int:
        """Get hidden size from the model config using AutoConfig.

        Returns:
            Hidden size dimension.
        """
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            self.args.target_model_path,
            trust_remote_code=getattr(self.args, "trust_remote_code", False),
        )
        model_config = getattr(config, "text_config", config)
        hidden_size = getattr(model_config, "hidden_size", None)
        if hidden_size is None:
            text_config = getattr(config, "text_config", None)
            if text_config is not None:
                hidden_size = getattr(text_config, "hidden_size", None)
        self._num_hidden_layers = getattr(model_config, "num_hidden_layers", None)
        return hidden_size

    def _get_tensor_shapes(self, seq_len: int) -> dict:
        """Get tensor shapes for mooncake metadata.

        Args:
            seq_len: Sequence length for this sample (prompt length in prefill mode,
                or prompt + completion - 1 in decode mode).

        Returns:
            Dict mapping tensor names to shapes.
        """
        aux_hidden_state_layer_ids = self.aux_hidden_state_layer_ids
        num_aux_layers = len(aux_hidden_state_layer_ids)
        if self._hidden_size is None:
            raise ValueError(
                f"SglEngine rank {self.rank}: hidden_size not initialized. Call init() first."
            )
        hidden_size = self._hidden_size

        # Concatenated hidden states from all aux layers
        # Sglang concatenates hidden states from all specified layers along the last dimension
        concat_hidden_size = num_aux_layers * hidden_size

        # IMPORTANT: Sglang stores tensors WITHOUT batch dimension in mooncake
        # We must request the SAME shapes that sglang stored, otherwise we get size mismatch
        # The collator will add batch dimension when needed
        shapes = {
            "hidden_states": (seq_len, concat_hidden_size),  # 2D without batch dim
            "input_ids": (seq_len,),  # 1D without batch dim
        }
        if self._store_last_hidden_states:
            shapes["last_hidden_states"] = (seq_len, hidden_size)
        return shapes

    def _get_tensor_dtypes(self) -> dict:
        """Get tensor dtypes for mooncake metadata."""
        dtypes = {
            "hidden_states": HIDDEN_STATES_STORAGE_DTYPE,
            "input_ids": torch.long,
        }
        if self._store_last_hidden_states:
            dtypes["last_hidden_states"] = HIDDEN_STATES_STORAGE_DTYPE
        return dtypes
