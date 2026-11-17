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

"""Queue-based data fetching with mooncake store.
Data flow:
  TrainActor -> MooncakeDataFetcher -> MooncakeDataset -> MooncakeStore -> Collator
                     |                      |                  |               |
                iter(fetcher)          queue.get()      store.get(key)     pad & batch
"""

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

import ray
import torch
import torch.distributed as dist
import torch.nn.functional as F
from ray.util.queue import Queue as RayQueue
from torch.utils.data import DataLoader, IterableDataset

from aspec.data.utils import deserialize_packed_loss_mask, resolve_loss_mask, unpack_loss_mask
from aspec.feature_contract import FeatureContractError, FeatureContractRuntime
from aspec.utils.distributed import (
    get_direct_cp_group,
    get_draft_sp_group,
    get_sp_ring_group,
    get_usp_rank_coords,
)
from aspec.utils.logging import logger
from aspec.utils.misc import get_current_node_ip
from aspec.work_lifecycle import EXECUTION_ID_KEY, WORK_ID_KEY


def _take_p2p_receive_leases(value: Any) -> list[Any]:
    """Remove private destination-ring leases from a sample or batch."""

    leases: list[Any] = []
    if isinstance(value, dict):
        lease = value.pop("_mooncake_p2p_receive_lease", None)
        if lease is not None:
            leases.append(lease)
        leases.extend(value.pop("_mooncake_p2p_receive_leases", []))
    elif isinstance(value, (list, tuple)):
        for item in value:
            leases.extend(_take_p2p_receive_leases(item))
    return leases


def _release_p2p_receive_leases(value: Any) -> None:
    for lease in _take_p2p_receive_leases(value):
        lease.release()


def _synchronize_cuda_tensor_devices(values: Any) -> None:
    """Wait for clones on their CUDA devices before a P2P slot is reused."""

    devices = {value.device for value in values if getattr(value, "is_cuda", False)}
    for device in devices:
        torch.cuda.current_stream(device).synchronize()


def _clone_fragment_before_receive_buffer_reuse(
    tensors: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Clone Mooncake receive-buffer views before the next GPUDirect GET.

    CUDA clones are asynchronous, while Mooncake writes the registered receive
    buffer outside CUDA stream ordering. Synchronize the clone stream before a
    subsequent GET can reuse and overwrite that buffer.
    """
    cloned = {name: tensor.clone() for name, tensor in tensors.items()}
    cuda_tensor = next((tensor for tensor in cloned.values() if tensor.is_cuda), None)
    if cuda_tensor is not None:
        torch.cuda.current_stream(cuda_tensor.device).synchronize()
    return cloned


@dataclass
class TrainSample:
    mooncake_key: str
    tensor_shapes: Dict[str, Tuple[int, ...]]
    tensor_dtypes: Optional[Dict[str, torch.dtype]] = None
    packed_loss_mask: Optional[str] = None
    last_turn_loss_only: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None
    data_id: Optional[str] = None
    row_id: Optional[int] = None
    last_in_row: bool = False


class MooncakeDataset(IterableDataset):
    """IterableDataset that loads from mooncake via queue.

    Each DP rank waits on its queue for TrainSample items sent by the
    centralized controller. Data is loaded from mooncake.
    """

    def __init__(
        self,
        ray_queue: RayQueue,
        mooncake_store,
        device: torch.device,
        prefetch_factor: int = 2,
        timeout: Optional[float] = None,
        assistant_header_ids: Optional[List[int]] = None,
        end_token_ids: Optional[List[int]] = None,
        dynamic_loss_mask: bool = False,
        last_turn_loss_only: bool = False,
        skip_after_header: int = 0,
        batch_size: int = 1,
        min_loss_tokens: int = 0,
        usp_enabled: bool = False,
        direct_cp_enabled: bool = False,
        ttt_length: int = 1,
        max_seq_length: Optional[int] = None,
        feature_contract: Optional[Dict[str, Any]] = None,
        feature_contract_participant: str = "draft_trainer",
        lifecycle_observer=None,
        packing_enabled: bool = False,
        feature_affinity: str = "any",
        consumer_node_ip: str | None = None,
    ):
        self.ray_queue = ray_queue
        self.mooncake_store = mooncake_store
        self.device = device
        self.prefetch_factor = prefetch_factor
        self.timeout = timeout
        self._stop_requested = threading.Event()
        self.assistant_header_ids = assistant_header_ids
        self.end_token_ids = end_token_ids
        self.dynamic_loss_mask = dynamic_loss_mask
        self.last_turn_loss_only = last_turn_loss_only
        self.skip_after_header = skip_after_header
        self._batch_size = batch_size
        self._min_loss_tokens = min_loss_tokens
        self.usp_enabled = usp_enabled
        self.direct_cp_enabled = direct_cp_enabled
        self.ttt_length = ttt_length
        self.max_seq_length = max_seq_length
        self._feature_contract_runtime = (
            FeatureContractRuntime(
                feature_contract,
                role="consumer",
                participant=feature_contract_participant,
            )
            if feature_contract is not None
            else None
        )
        self._feature_contract_participant = feature_contract_participant
        self._neutralized_rank_samples = 0
        self._lifecycle_observer = lifecycle_observer
        self._lifecycle_participant = feature_contract_participant
        self.packing_enabled = packing_enabled
        self.feature_affinity = str(feature_affinity).lower()
        if self.feature_affinity not in {"any", "node_local_required"}:
            raise ValueError(f"unsupported feature affinity {self.feature_affinity!r}")
        self.consumer_node_ip = (
            str(consumer_node_ip or get_current_node_ip())
            if self.feature_affinity == "node_local_required"
            else None
        )
        self._node_local_affinity_validations = 0
        self._node_local_affinity_violations = 0
        if packing_enabled and (
            usp_enabled or max_seq_length is None or max_seq_length <= 0
        ):
            raise ValueError("packing requires non-USP training and positive max_seq_length")
        self._init_sp_context()

    def request_stop(self) -> None:
        """Cancel future queue reads without interrupting a native transfer."""
        self._stop_requested.set()

    def _next_queue_item(self):
        # Poll only the queue wait, not Mooncake itself: close must be able to
        # join an idle prefetch worker before its registered buffers are freed.
        deadline = None if self.timeout is None else time.monotonic() + self.timeout
        while not self._stop_requested.is_set():
            timeout = 0.1 if deadline is None else min(
                0.1, max(0.0, deadline - time.monotonic())
            )
            try:
                return self.ray_queue.get(block=True, timeout=timeout)
            except queue.Empty:
                if deadline is not None and time.monotonic() >= deadline:
                    raise
        return None

    def _init_sp_context(self) -> None:
        self._sp_group = None
        self._sp_world_size = 1
        self._sp_rank = 0
        self._sp_ring_size = 1
        self._sp_ring_rank = 0
        if self.direct_cp_enabled:
            self._sp_group = get_direct_cp_group()
            if self._sp_group is not None:
                self._sp_world_size = dist.get_world_size(self._sp_group)
                self._sp_rank = dist.get_rank(self._sp_group)
            return
        if not self.usp_enabled:
            return

        sp_group = get_draft_sp_group()
        if sp_group is None:
            return

        self._sp_group = sp_group
        self._sp_world_size = dist.get_world_size(sp_group)
        self._sp_rank = dist.get_rank(sp_group)

        ring_group = get_sp_ring_group()
        if ring_group is not None:
            self._sp_ring_size = dist.get_world_size(ring_group)
            self._sp_ring_rank = dist.get_rank(ring_group)

    def _load_from_mooncake(self, sample: TrainSample) -> Dict[str, Any]:
        """Load tensors from mooncake key into device memory."""
        dtypes_raw = sample.tensor_dtypes or {}

        # Convert string dtypes to torch.dtype objects
        dtypes = {}
        for key, dtype_val in dtypes_raw.items():
            if isinstance(dtype_val, str):
                # Handle "bfloat16" or "torch.bfloat16" format
                dtype_str = dtype_val.replace("torch.", "")
                dtypes[key] = getattr(torch, dtype_str)
            else:
                dtypes[key] = dtype_val

        # DEBUG: Print the shapes we're requesting
        logger.debug(
            f"_load_from_mooncake: key={sample.mooncake_key}, "
            f"requesting shapes={sample.tensor_shapes}"
        )

        metadata = sample.metadata or {}
        self._validate_node_local_affinity(metadata, sample.data_id)
        if self._feature_contract_runtime is not None:
            self._feature_contract_runtime.validate_metadata(
                tensor_shapes=sample.tensor_shapes,
                tensor_dtypes=dtypes,
                metadata=metadata,
            )
        pp_manifest = metadata.get("vllm_pp_layer_manifest")
        storage_released = False
        p2p_receive_lease = None
        try:
            if pp_manifest is not None:
                result = self._load_vllm_pp_layers(sample, pp_manifest, dtypes)
                storage_released = True
            else:
                tensors = self.mooncake_store.get(
                    key=sample.mooncake_key,
                    shapes=sample.tensor_shapes,
                    dtypes=dtypes,
                    device=self.device,
                )

                p2p_receive_lease = getattr(tensors, "p2p_receive_lease", None)
                tensor_dict = tensors.to_tensor_dict()
                if self._batch_size > 1 or self.packing_enabled:
                    # Clone to prevent use-after-free: collator holds sample N while
                    # fetching N+1, but cleanup frees the Mooncake buffer (Issue 31).
                    # Note: clone() converts pinned → unpinned, breaking non_blocking
                    # H2D transfers. Only do this when actually needed.
                    result = _clone_fragment_before_receive_buffer_reuse(tensor_dict)
                    if p2p_receive_lease is not None:
                        p2p_receive_lease.release()
                        p2p_receive_lease = None
                else:
                    # batch_size=1: safe to use pinned views — consumed immediately.
                    # Preserves pinned memory for async H2D via non_blocking=True.
                    result = dict(tensor_dict)

                self._cleanup_mooncake_data(sample)
                storage_released = True
            if self._feature_contract_runtime is not None:
                self._validate_received_feature_tensors(
                    result,
                    expected_shapes=sample.tensor_shapes,
                    expected_dtypes=dtypes,
                )
            if p2p_receive_lease is not None:
                result["_mooncake_p2p_receive_lease"] = p2p_receive_lease
        except BaseException:
            if p2p_receive_lease is not None:
                p2p_receive_lease.release()
            if storage_released:
                try:
                    self._notify_release(sample, outcome="discarded_after_error")
                except Exception as notify_error:
                    logger.error(
                        "Failed to report discarded output %s: %s",
                        sample.data_id,
                        notify_error,
                    )
            raise
        self._notify_release(sample, outcome="consumed")
        if self._lifecycle_observer is not None:
            metadata = sample.metadata or {}
            result["_aspec_work"] = {
                "work_id": metadata.get(WORK_ID_KEY),
                "execution_id": metadata.get(EXECUTION_ID_KEY),
            }
        if sample.packed_loss_mask is not None:
            result["packed_loss_mask"] = sample.packed_loss_mask
        if sample.last_turn_loss_only is not None:
            result["last_turn_loss_only"] = sample.last_turn_loss_only
        return result

    def _notify_release(self, sample: TrainSample, *, outcome: str) -> None:
        if self._lifecycle_observer is None:
            return
        metadata = sample.metadata or {}
        execution_id = metadata.get(EXECUTION_ID_KEY)
        if not isinstance(execution_id, str) or not execution_id:
            raise RuntimeError(
                f"Transferred sample {sample.data_id!r} has no {EXECUTION_ID_KEY}"
            )
        ray.get(
            self._lifecycle_observer.release_inference_output.remote(
                execution_id=execution_id,
                data_id=str(sample.data_id),
                mooncake_key=sample.mooncake_key,
                participant=self._lifecycle_participant,
                outcome=outcome,
            )
        )

    def _validate_received_feature_tensors(
        self,
        tensors: Dict[str, Any],
        *,
        expected_shapes: Dict[str, Any],
        expected_dtypes: Dict[str, Any],
    ) -> None:
        """Validate transported features without treating local views as payload."""

        if self._feature_contract_runtime is None:
            raise FeatureContractError(
                "Received target feature tensors before the feature contract was initialized"
            )
        contract_names = set(self._feature_contract_runtime.contract["tensors"])
        local_derivatives = {"input_ids_cpu"}
        unexpected = set(tensors) - contract_names - local_derivatives
        if unexpected:
            raise FeatureContractError(
                "Received undeclared target feature tensors: "
                f"{sorted(unexpected)!r}"
            )
        self._feature_contract_runtime.validate_tensors(
            {name: tensors[name] for name in contract_names if name in tensors},
            expected_shapes=expected_shapes,
            expected_dtypes=expected_dtypes,
        )

    def _load_vllm_pp_layers(
        self,
        sample: TrainSample,
        manifest: list[dict[str, Any]],
        dtypes: dict[str, torch.dtype],
    ) -> Dict[str, Any]:
        metadata = sample.metadata or {}
        if not metadata.get("vllm_pp_complete", False):
            raise RuntimeError(f"Incomplete vLLM PP sample: mooncake_key={sample.mooncake_key}")

        seq_len = sample.tensor_shapes["input_ids"][0]
        hidden_size = sample.tensor_shapes["last_hidden_states"][-1]
        fragment_shapes = {
            "hidden_states": (seq_len, hidden_size),
            "input_ids": (seq_len,),
        }
        fragment_dtypes = {
            "hidden_states": dtypes.get("hidden_states", torch.bfloat16),
            "input_ids": torch.int64,
        }

        hidden_states = []
        last_hidden_states = None
        input_ids = None
        fragment_keys = []
        for entry in manifest:
            key = entry["mooncake_key"]
            tensors = self.mooncake_store.get(
                key=key,
                shapes=fragment_shapes,
                dtypes=fragment_dtypes,
                device=self.device,
            ).to_tensor_dict()

            cloned = _clone_fragment_before_receive_buffer_reuse(tensors)
            layer_hidden = cloned["hidden_states"]
            layer_input_ids = cloned["input_ids"]
            if input_ids is None:
                input_ids = layer_input_ids
            elif not torch.equal(input_ids, layer_input_ids):
                raise RuntimeError(f"vLLM PP input_ids mismatch for fragment {key}")

            if entry["role"] == "last_hidden_states":
                last_hidden_states = layer_hidden
            else:
                hidden_states.append(layer_hidden)

            fragment_keys.append(key)

        if not hidden_states or last_hidden_states is None or input_ids is None:
            raise RuntimeError(f"Invalid vLLM PP layer manifest for {sample.mooncake_key}")

        for key in fragment_keys:
            self.mooncake_store.remove_eagle3_tensors(
                key,
                has_last_hidden_states=False,
                has_target=False,
            )

        return {
            "hidden_states": torch.cat(hidden_states, dim=-1),
            "last_hidden_states": last_hidden_states,
            "input_ids": input_ids,
        }

    def _cleanup_mooncake_data(self, sample: TrainSample) -> None:
        """Remove data from mooncake store to release buffer space."""
        shapes = sample.tensor_shapes or {}
        has_lhs = "last_hidden_states" in shapes
        has_target = "target" in shapes

        self.mooncake_store.remove_eagle3_tensors(
            sample.mooncake_key,
            has_last_hidden_states=has_lhs,
            has_target=has_target,
            **({"has_mtp_hidden_states": True} if "mtp_hidden_states" in shapes else {}),
        )

    def _compute_loss_mask(self, data: Dict[str, Any]) -> torch.Tensor | None:
        return resolve_loss_mask(
            data,
            dynamic_loss_mask=self.dynamic_loss_mask,
            assistant_header_ids=self.assistant_header_ids,
            end_token_ids=self.end_token_ids,
            last_turn_loss_only=self.last_turn_loss_only,
            skip_after_header=self.skip_after_header,
        )

    @staticmethod
    def _fallback_mask_len(data: Dict[str, Any]) -> int:
        """Sequence length for a neutralized sample's zero loss mask."""
        ids = data.get("input_ids")
        if isinstance(ids, torch.Tensor) and ids.dim() >= 1:
            return ids.shape[-1]
        for key in ("hidden_states", "last_hidden_states", "target"):
            t = data.get(key)
            if isinstance(t, torch.Tensor) and t.dim() >= 2:
                return t.shape[-2]
        return 1

    def _resolve_and_neutralize_loss_mask(
        self, data: Dict[str, Any], mooncake_key: str, neutralized_count: int
    ) -> int:
        """Zero an empty / sub-min_loss_tokens loss mask in place instead of
        dropping the sample; a per-rank drop desyncs FSDP collectives."""
        mask = self._compute_loss_mask(data)  # None == all-zero
        neutralize = mask is None or (
            self._min_loss_tokens > 0
            and isinstance(mask, torch.Tensor)
            and mask.sum() < self._min_loss_tokens
        )
        if not neutralize:
            return neutralized_count

        if isinstance(mask, torch.Tensor):
            n_tokens = int(mask.sum())
            data["loss_mask"] = torch.zeros_like(mask)
        else:
            # resolve_loss_mask returns None without setting data["loss_mask"]
            n_tokens = 0
            data["loss_mask"] = torch.zeros(self._fallback_mask_len(data), dtype=torch.long)

        neutralized_count += 1
        self._neutralized_rank_samples += 1
        logger.warning(
            f"Neutralized loss mask ({n_tokens} < min_loss_tokens="
            f"{self._min_loss_tokens}, mooncake_key={mooncake_key}, "
            f"total_neutralized={neutralized_count})"
        )
        return neutralized_count

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over samples synchronously.

        Blocks waiting for each item from the queue and loads from mooncake.
        Empty / sub-min_loss_tokens masks are neutralized, not dropped
        (see _resolve_and_neutralize_loss_mask).
        """
        if self.packing_enabled:
            yield from self._iter_packed_rows()
            return

        yield_count = 0
        neutralized_count = 0
        while True:
            if self.direct_cp_enabled:
                data, neutralized_count = self._direct_cp_get_sharded_item(
                    neutralized_count
                )
                if data is None:
                    break
                yield_count += 1
                yield data
                continue
            if self.usp_enabled:
                data, neutralized = self._usp_get_sharded_item(neutralized_count=neutralized_count)
                neutralized_count += neutralized
                if data is None:
                    break
                yield_count += 1
                yield data
                continue

            logger.debug(f"__iter__: waiting for item from ray_queue (yield_count={yield_count})")
            try:
                item = self._next_queue_item()
            except Exception as e:
                logger.warning(f"__iter__: Exception waiting for data: {e}, timeout={self.timeout}")
                break

            if item is None:
                logger.debug("__iter__: received None sentinel, stopping iteration")
                break

            logger.debug(f"__iter__: got item, mooncake_key={item.mooncake_key}")
            data = self._load_from_mooncake(item)

            neutralized_count = self._resolve_and_neutralize_loss_mask(
                data, item.mooncake_key, neutralized_count
            )

            # Note: target is computed in the collator from last_hidden_states for sglang mode

            # Add batch dimension if missing (sglang stores without batch dim)
            for key, tensor in data.items():
                if tensor is not None and isinstance(tensor, torch.Tensor):
                    # Check if tensor is missing batch dimension
                    # 1D tensors (loss_mask, input_ids) should be 2D: (1, seq_len)
                    # 2D tensors (hidden_states, last_hidden_states) should be 3D: (1, seq_len, dim)
                    if tensor.dim() == 1:
                        data[key] = tensor.unsqueeze(0)  # (seq_len,) -> (1, seq_len)
                    elif tensor.dim() == 2 and key in [
                        "hidden_states",
                        "last_hidden_states",
                        "mtp_hidden_states",
                        "target",
                    ]:
                        data[key] = tensor.unsqueeze(0)  # (seq_len, dim) -> (1, seq_len, dim)

            # Debug: log all tensor shapes after adding batch dim
            if data:
                shapes_str = {
                    k: v.shape if hasattr(v, "shape") else type(v) for k, v in data.items()
                }
                logger.debug(f"final shapes (with batch dim): {shapes_str}")
            yield_count += 1
            logger.debug(f"__iter__: yielding batch {yield_count}, keys={list(data.keys())}")
            yield data

    def _direct_cp_get_sharded_item(
        self, neutralized_count: int
    ) -> tuple[Dict[str, torch.Tensor] | None, int]:
        """Read this rank's feature shard while retaining complete token semantics."""

        try:
            item = self._next_queue_item()
        except Exception as exc:
            logger.warning(
                "Direct CP fetch failed while waiting for data: %s, timeout=%s",
                exc,
                self.timeout,
            )
            return None, neutralized_count
        if item is None:
            return None, neutralized_count

        return self._load_direct_cp_sharded_item(item, neutralized_count)

    def _load_direct_cp_sharded_item(
        self, item: TrainSample, neutralized_count: int,
    ) -> tuple[Dict[str, torch.Tensor], int]:
        metadata = item.metadata or {}
        self._validate_node_local_affinity(metadata, item.data_id)
        if not metadata.get("direct_cp_sharded", False):
            raise RuntimeError(
                "Direct CP data fetcher received a non-sharded sample: "
                f"{item.mooncake_key}"
            )

        dtypes = {
            name: (
                getattr(torch, value.replace("torch.", ""))
                if isinstance(value, str)
                else value
            )
            for name, value in (item.tensor_dtypes or {}).items()
        }
        if self._feature_contract_runtime is not None:
            self._feature_contract_runtime.validate_metadata(
                tensor_shapes=item.tensor_shapes,
                tensor_dtypes=dtypes,
                metadata=metadata,
            )

        global_len = int(item.tensor_shapes["input_ids"][-1])
        if self.max_seq_length is not None:
            global_len = min(global_len, self.max_seq_length)
        from aspec.inference.context import context_shard_bounds

        start, end = context_shard_bounds(
            global_len, self._sp_rank, self._sp_world_size,
            alignment=int(metadata.get("context_shard_alignment", 1)),
        )
        local_shapes: dict[str, tuple[int, ...]] = {}
        for name, original in item.tensor_shapes.items():
            shape = list(original)
            if name == "input_ids":
                shape[-1] = global_len
            else:
                shape[-2] = end - start
            local_shapes[name] = tuple(shape)

        from aspec.transfer.mooncake.p2p import direct_cp_p2p_key_for_rank

        shard_key = direct_cp_p2p_key_for_rank(
            item.mooncake_key,
            cp_rank=self._sp_rank,
            cp_size=self._sp_world_size,
        )
        storage_released = False
        receive_lease = None
        try:
            output = self.mooncake_store.get(
                key=shard_key,
                shapes=local_shapes,
                dtypes=dtypes,
                device=self.device,
            )
            receive_lease = getattr(output, "p2p_receive_lease", None)
            tensor_dict = output.to_tensor_dict()
            if self._batch_size > 1 or self.packing_enabled:
                data = _clone_fragment_before_receive_buffer_reuse(tensor_dict)
                if receive_lease is not None:
                    receive_lease.release()
                    receive_lease = None
            else:
                data = dict(tensor_dict)
            self.mooncake_store.remove_eagle3_tensors(
                shard_key,
                has_last_hidden_states="last_hidden_states" in local_shapes,
                has_target="target" in local_shapes,
                **({"has_mtp_hidden_states": True} if "mtp_hidden_states" in local_shapes else {}),
            )
            storage_released = True
            if self._feature_contract_runtime is not None:
                self._validate_received_feature_tensors(
                    data,
                    expected_shapes=local_shapes,
                    expected_dtypes=dtypes,
                )
            if receive_lease is not None:
                data["_mooncake_p2p_receive_lease"] = receive_lease
            if item.packed_loss_mask is not None:
                data["packed_loss_mask"] = item.packed_loss_mask
            if item.last_turn_loss_only is not None:
                data["last_turn_loss_only"] = item.last_turn_loss_only
            neutralized_count = self._resolve_and_neutralize_loss_mask(
                data, item.mooncake_key, neutralized_count
            )
            for name, tensor in list(data.items()):
                if not isinstance(tensor, torch.Tensor):
                    continue
                if tensor.dim() == 1:
                    data[name] = tensor.unsqueeze(0)
                elif tensor.dim() == 2 and name in {
                    "hidden_states",
                    "last_hidden_states",
                    "mtp_hidden_states",
                    "target",
                }:
                    data[name] = tensor.unsqueeze(0)
        except BaseException:
            if receive_lease is not None:
                receive_lease.release()
            if storage_released:
                try:
                    self._notify_release(item, outcome="discarded_after_error")
                except Exception as notify_error:
                    logger.error(
                        "Failed to report discarded Direct CP output %s: %s",
                        item.data_id,
                        notify_error,
                    )
            raise

        self._notify_release(item, outcome="consumed")
        if self._lifecycle_observer is not None:
            data["_aspec_work"] = {
                "work_id": metadata.get(WORK_ID_KEY),
                "execution_id": metadata.get(EXECUTION_ID_KEY),
            }
        return data, neutralized_count

    def _iter_packed_rows(self) -> Iterator[List[Dict[str, torch.Tensor]]]:
        """Assemble the exact row stamped by the controller, without look-ahead."""
        neutralized_count = 0
        current_row: List[Dict[str, torch.Tensor]] = []
        current_id: int | None = None
        while True:
            item = self._next_queue_item()
            if item is None:
                if current_row and not self._stop_requested.is_set():
                    raise RuntimeError("Packing stream ended in the middle of a row")
                return
            if item.row_id is None:
                raise RuntimeError("Packing sample is missing controller-owned row_id")
            if current_id is None:
                current_id = item.row_id
            elif current_id != item.row_id:
                raise RuntimeError("Packing row_id changed before last_in_row")

            if self.direct_cp_enabled:
                data, neutralized_count = self._load_direct_cp_sharded_item(
                    item, neutralized_count
                )
            else:
                data = self._load_from_mooncake(item)
                neutralized_count = self._resolve_and_neutralize_loss_mask(
                    data, item.mooncake_key, neutralized_count
                )
            for key, tensor in list(data.items()):
                if tensor is None or not isinstance(tensor, torch.Tensor):
                    continue
                if tensor.dim() == 1:
                    data[key] = tensor.unsqueeze(0)
                elif tensor.dim() == 2 and key in {
                    "hidden_states",
                    "last_hidden_states",
                    "mtp_hidden_states",
                    "target",
                }:
                    data[key] = tensor.unsqueeze(0)
            current_row.append(data)
            if item.last_in_row:
                yield current_row
                current_row = []
                current_id = None

    def _usp_global_len(self, sample: TrainSample) -> int:
        global_len = sample.tensor_shapes["input_ids"][-1]
        if self.max_seq_length is not None:
            global_len = min(global_len, self.max_seq_length)
        return global_len

    def _usp_chunk_size(self, global_len: int) -> int:
        return (global_len + self._sp_world_size - 1) // self._sp_world_size

    def _usp_loss_mask(self, sample: TrainSample, global_len: int) -> torch.Tensor:
        if sample.packed_loss_mask is None:
            raise RuntimeError("USP sharded Mooncake reads require packed_loss_mask metadata")
        loss_mask = unpack_loss_mask(deserialize_packed_loss_mask(sample.packed_loss_mask))
        loss_mask = loss_mask[:global_len]
        if loss_mask.shape[0] < global_len:
            loss_mask = F.pad(loss_mask, (0, global_len - loss_mask.shape[0]))
        return loss_mask

    def _local_usp_shapes(self, sample: TrainSample) -> dict[str, tuple[int, ...]]:
        local_len = self._usp_chunk_size(self._usp_global_len(sample)) + self.ttt_length
        shapes: dict[str, tuple[int, ...]] = {
            "input_ids": (1, local_len),
            "hidden_states": (1, local_len, sample.tensor_shapes["hidden_states"][-1]),
        }
        if "last_hidden_states" in sample.tensor_shapes:
            shapes["last_hidden_states"] = (
                1,
                local_len,
                sample.tensor_shapes["last_hidden_states"][-1],
            )
        if "target" in sample.tensor_shapes:
            shapes["target"] = (1, local_len, sample.tensor_shapes["target"][-1])
        return shapes

    def _local_usp_loss_and_position(
        self,
        sample: TrainSample,
        local_len: int,
    ) -> dict[str, torch.Tensor]:
        sp_ulysses_size = max(1, self._sp_world_size // self._sp_ring_size)
        global_len = self._usp_global_len(sample)
        chunk_size = self._usp_chunk_size(global_len)
        start = self._sp_rank * chunk_size
        end = min(start + local_len, global_len)
        valid_len = max(0, end - start)

        loss_mask = self._usp_loss_mask(sample, global_len)[start:end].unsqueeze(0)
        if loss_mask.shape[-1] < local_len:
            loss_mask = F.pad(loss_mask, (0, local_len - loss_mask.shape[-1]))

        attention_mask = torch.zeros((1, local_len), dtype=torch.long)
        attention_mask[:, :valid_len] = 1

        usp_chunk_size = max(local_len - self.ttt_length, 0)
        ring_chunk = usp_chunk_size * sp_ulysses_size
        _, ring_rank = get_usp_rank_coords(
            sp_rank=self._sp_rank,
            sp_ulysses_size=sp_ulysses_size,
            sp_ring_size=self._sp_ring_size,
        )
        ring_start = ring_rank * ring_chunk
        position_ids = torch.arange(
            ring_start,
            ring_start + ring_chunk,
            dtype=torch.long,
        ).unsqueeze(0)

        return {
            "loss_mask": loss_mask.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "position_ids": position_ids.to(self.device),
        }

    def _should_skip_usp_sharded_sample(self, sample: TrainSample) -> bool:
        """Return the SP-consistent skip decision for a pre-sharded USP sample."""
        full_loss_mask = self._usp_loss_mask(sample, self._usp_global_len(sample))
        min_tokens = max(1, self._min_loss_tokens)
        return int(full_loss_mask.sum().item()) < min_tokens

    def _usp_get_sharded_item(
        self, neutralized_count: int
    ) -> tuple[Dict[str, torch.Tensor] | None, int]:
        neutralized = 0
        while True:
            try:
                item = self._next_queue_item()
            except Exception as e:
                logger.warning(
                    f"_usp_get_sharded_item: Exception waiting for data: {e}, "
                    f"timeout={self.timeout}"
                )
                return None, neutralized
            if item is None:
                return None, neutralized

            metadata = item.metadata or {}
            self._validate_node_local_affinity(metadata, item.data_id)
            if not metadata.get("usp_sharded", False):
                raise RuntimeError(
                    "USP sharded data fetcher received a non-sharded Mooncake sample. "
                    f"mooncake_key={item.mooncake_key}"
                )

            if self._feature_contract_runtime is not None:
                self._feature_contract_runtime.validate_metadata(
                    tensor_shapes=item.tensor_shapes,
                    tensor_dtypes=item.tensor_dtypes or {},
                    metadata=metadata,
                )

            shapes = self._local_usp_shapes(item)
            dtypes_raw = item.tensor_dtypes or {}
            dtypes = {}
            for key, dtype_val in dtypes_raw.items():
                if isinstance(dtype_val, str):
                    dtypes[key] = getattr(torch, dtype_val.replace("torch.", ""))
                else:
                    dtypes[key] = dtype_val

            should_skip = self._should_skip_usp_sharded_sample(item)
            shard_key = f"{item.mooncake_key}_usp{self._sp_rank}"
            tensors = self.mooncake_store.get(
                key=shard_key,
                shapes=shapes,
                dtypes=dtypes,
                device=self.device,
            ).to_tensor_dict()
            if self._feature_contract_runtime is not None:
                self._validate_received_feature_tensors(
                    tensors,
                    expected_shapes=shapes,
                    expected_dtypes=dtypes,
                )
            tensors.update(self._local_usp_loss_and_position(item, shapes["input_ids"][-1]))

            self.mooncake_store.remove_eagle3_tensors(
                shard_key,
                has_last_hidden_states="last_hidden_states" in shapes,
                has_target="target" in shapes,
            )

            if should_skip:
                # Neutralize, don't drop: a per-DP-group drop desyncs FSDP.
                neutralized += 1
                self._neutralized_rank_samples += 1
                tensors["loss_mask"] = torch.zeros_like(tensors["loss_mask"])
                logger.warning(
                    f"Neutralized USP sharded sample (loss mask zeroed): "
                    f"mooncake_key={item.mooncake_key}, sp_rank={self._sp_rank}, "
                    f"total_neutralized={neutralized_count + neutralized}"
                )

            return tensors, neutralized

    def get_feature_contract_status(self) -> dict[str, Any]:
        if self._feature_contract_runtime is None:
            status = {
                "participant": self._feature_contract_participant,
                "role": "consumer",
                "contract_id": None,
                "state": "not_configured",
                "data_admission": {
                    "neutralized_rank_samples": self._neutralized_rank_samples,
                },
            }
        else:
            status = self._feature_contract_runtime.status()
        if self.feature_affinity == "node_local_required":
            status["feature_affinity"] = {
                "mode": self.feature_affinity,
                "consumer_node_ip": self.consumer_node_ip,
                "validations": self._node_local_affinity_validations,
                "violations": self._node_local_affinity_violations,
                "state": (
                    "validated"
                    if self._node_local_affinity_validations > 0
                    and self._node_local_affinity_violations == 0
                    else "pending_observation"
                ),
            }
        if self._feature_contract_runtime is not None:
            status["data_admission"] = {
                "neutralized_rank_samples": self._neutralized_rank_samples,
            }
        get_transport_status = getattr(self.mooncake_store, "get_p2p_status", None)
        transport = (
            get_transport_status() if callable(get_transport_status) else None
        )
        if transport is not None:
            status["transport"] = transport
        return status

    def _validate_node_local_affinity(
        self, metadata: Mapping[str, Any], data_id: str | None
    ) -> None:
        if self.feature_affinity != "node_local_required":
            return
        source_node = str(metadata.get("source_node_ip", ""))
        assigned_node = str(metadata.get("assigned_draft_node_ip", ""))
        if (
            metadata.get("feature_affinity") != "node_local_required"
            or not source_node
            or source_node != assigned_node
            or assigned_node != self.consumer_node_ip
        ):
            self._node_local_affinity_violations += 1
            raise RuntimeError(
                "node-local feature affinity violation at Draft consumer: "
                f"data_id={data_id!r}, source={source_node!r}, "
                f"assigned={assigned_node!r}, consumer={self.consumer_node_ip!r}"
            )
        self._node_local_affinity_validations += 1


def create_mooncake_dataloader(
    ray_queue: RayQueue,
    mooncake_store,
    collator: Callable[[List[Dict]], Dict[str, torch.Tensor]],
    device: torch.device,
    batch_size: int = 1,
    prefetch_factor: int = 2,
    timeout: Optional[float] = None,
    assistant_header_ids: Optional[List[int]] = None,
    end_token_ids: Optional[List[int]] = None,
    dynamic_loss_mask: bool = False,
    last_turn_loss_only: bool = False,
    skip_after_header: int = 0,
    min_loss_tokens: int = 0,
    usp_enabled: bool = False,
    direct_cp_enabled: bool = False,
    ttt_length: int = 1,
    max_seq_length: Optional[int] = None,
    feature_contract: Optional[Dict[str, Any]] = None,
    feature_contract_participant: str = "draft_trainer",
    lifecycle_observer=None,
    packing_enabled: bool = False,
    feature_affinity: str = "any",
    consumer_node_ip: str | None = None,
) -> DataLoader:
    """Create a DataLoader that fetches from mooncake via queue.

    Data flow:
      Controller (dispatches dispatch_batch_size samples) ->
      Ray Queue (per_dp_rank_batch_size samples per rank) ->
      DataLoader (batches per_dp_rank_batch_size samples together with padding) ->
      Training loop (one iteration per step)

    The collator pads sequences within the batch to the same length.

    Args:
        ray_queue: Ray Queue to receive TrainSample from controller.
        mooncake_store: Mooncake store client for loading tensors.
        collator: Collator for padding and batching samples.
        device: Target device for tensors.
        batch_size: Number of samples per batch (= per_dp_rank_batch_size).
        prefetch_factor: Unused, kept for API compatibility.
        timeout: Timeout in seconds for waiting on queue. None means wait forever.
        assistant_header_ids: Token IDs for assistant header (for loss mask skip check).
        end_token_ids: Token IDs for end of turn (for loss mask skip check).
        dynamic_loss_mask: Whether loss mask is computed dynamically from input_ids.
        last_turn_loss_only: Global fallback for last-turn-only loss masking.

    Returns:
        DataLoader instance.
    """
    dataset = MooncakeDataset(
        ray_queue,
        mooncake_store,
        device,
        prefetch_factor,
        timeout,
        assistant_header_ids=assistant_header_ids,
        end_token_ids=end_token_ids,
        dynamic_loss_mask=dynamic_loss_mask,
        last_turn_loss_only=last_turn_loss_only,
        skip_after_header=skip_after_header,
        batch_size=batch_size,
        min_loss_tokens=min_loss_tokens,
        usp_enabled=usp_enabled,
        direct_cp_enabled=direct_cp_enabled,
        ttt_length=ttt_length,
        max_seq_length=max_seq_length,
        feature_contract=feature_contract,
        feature_contract_participant=feature_contract_participant,
        lifecycle_observer=lifecycle_observer,
        packing_enabled=packing_enabled,
        feature_affinity=feature_affinity,
        consumer_node_ip=consumer_node_ip,
    )

    def dataloader_collator(samples):
        receive_leases = _take_p2p_receive_leases(samples)
        try:
            batch = collator(samples)
        except BaseException:
            for lease in receive_leases:
                lease.release()
            raise
        if receive_leases:
            batch["_mooncake_p2p_receive_leases"] = receive_leases
        return batch

    return DataLoader(
        dataset,
        batch_size=None if packing_enabled else batch_size,
        collate_fn=dataloader_collator,
        num_workers=0,
        # Iterator construction draws a worker base seed even with no workers.
        # A resumed queue creates a fresh iterator; this infrastructure draw
        # must not advance the checkpoint-restored training RNG.
        generator=torch.Generator().manual_seed(0),
    )


class MooncakeDataFetcher:
    """Queue-based data fetcher for mooncake with DataLoader backend.

    Provides iteration over training samples that are pushed to a Ray queue
    by the AsyncTrainingController and loaded from mooncake.

    Batch size design:
      - micro_batch_size: Samples per GPU per training step (user config)
      - per_dp_rank_batch_size = micro_batch_size * sp_size (derived)
      - dispatch_batch_size = per_dp_rank_batch_size * dp_size (derived)
      - DataLoader batch_size = per_dp_rank_batch_size (all samples batched together)
      - Training loop does ONE iteration per step

    The collator pads sequences within the batch to the max length.
    """

    def __init__(
        self,
        queue: RayQueue,
        mooncake_store,
        collator: Callable[[List[Dict]], Dict[str, torch.Tensor]],
        device: torch.device,
        batch_size: int = 1,
        prefetch_factor: int = 2,
        timeout: Optional[float] = None,
        assistant_header_ids: Optional[List[int]] = None,
        end_token_ids: Optional[List[int]] = None,
        dynamic_loss_mask: bool = False,
        last_turn_loss_only: bool = False,
        skip_after_header: int = 0,
        min_loss_tokens: int = 0,
        usp_enabled: bool = False,
        direct_cp_enabled: bool = False,
        ttt_length: int = 1,
        max_seq_length: Optional[int] = None,
        feature_contract: Optional[Dict[str, Any]] = None,
        feature_contract_participant: str = "draft_trainer",
        lifecycle_observer=None,
        packing_enabled: bool = False,
        feature_affinity: str = "any",
        consumer_node_ip: str | None = None,
    ):
        self.batch_size = batch_size
        self._dataloader = create_mooncake_dataloader(
            ray_queue=queue,
            mooncake_store=mooncake_store,
            collator=collator,
            device=device,
            batch_size=batch_size,
            prefetch_factor=prefetch_factor,
            timeout=timeout,
            assistant_header_ids=assistant_header_ids,
            end_token_ids=end_token_ids,
            dynamic_loss_mask=dynamic_loss_mask,
            last_turn_loss_only=last_turn_loss_only,
            skip_after_header=skip_after_header,
            min_loss_tokens=min_loss_tokens,
            usp_enabled=usp_enabled,
            direct_cp_enabled=direct_cp_enabled,
            ttt_length=ttt_length,
            max_seq_length=max_seq_length,
            feature_contract=feature_contract,
            feature_contract_participant=feature_contract_participant,
            lifecycle_observer=lifecycle_observer,
            packing_enabled=packing_enabled,
            feature_affinity=feature_affinity,
            consumer_node_ip=consumer_node_ip,
        )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        return iter(self._dataloader)

    def get_feature_contract_status(self) -> dict[str, Any]:
        return self._dataloader.dataset.get_feature_contract_status()

    def request_stop(self) -> None:
        self._dataloader.dataset.request_stop()


class PrefetchedDataFetcher:
    """Wraps MooncakeDataFetcher with async pre-fetching.

    A background thread continuously fetches batches from the underlying
    MooncakeDataFetcher (which blocks on Mooncake TCP), staging them in a
    thread-safe queue.  The training loop reads from this queue, overlapping
    data transfer with GPU compute.

    Without prefetch: [data] → [compute] → [data] → [compute]  (sequential)
    With prefetch:    [compute] → [compute] → [compute]         (overlapped)
                      [data]      [data]      [data]

    The background thread starts lazily on the first ``__iter__`` call and
    keeps running across multiple ``itertools.islice`` invocations (one per
    training step).  The training loop simply reads from the shared queue.
    """

    _SENTINEL = object()

    def __init__(
        self,
        inner: MooncakeDataFetcher,
        prefetch_depth: int = 2,
        target_device: Optional[torch.device] = None,
    ):
        self.inner = inner
        self.prefetch_depth = prefetch_depth
        self.target_device = target_device
        self._queue: queue.Queue = queue.Queue(maxsize=prefetch_depth)
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._error: Optional[BaseException] = None
        self._stop = False
        self._exhausted = False

    def _put_with_stop(self, item: Any) -> bool:
        while not self._stop:
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _prefetch_loop(self) -> None:
        try:
            for batch in self.inner:
                if not self._put_with_stop(batch):
                    _release_p2p_receive_leases(batch)
                    return
        except Exception as e:
            # Preserve the original traceback so re-raise in __next__
            # points to the actual failure site, not to __next__ itself.
            import sys

            self._error = e.with_traceback(sys.exc_info()[2])
        finally:
            self._put_with_stop(self._SENTINEL)

    def _ensure_started(self) -> None:
        if not self._started and not self._stop:
            self._started = True
            self._thread = threading.Thread(target=self._prefetch_loop, daemon=True)
            self._thread.start()

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        self._ensure_started()
        return self

    def close(self) -> None:
        """Join reads before the trainer closes the backing Mooncake client."""
        self._stop = True
        request_stop = getattr(self.inner, "request_stop", None)
        if callable(request_stop):
            request_stop()
        if self._thread is not None:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                # A native transfer may be stuck. Retain all backing storage;
                # the caller's owner-fencing path must handle that failure.
                raise RuntimeError("prefetch worker did not stop; backing storage must remain open")
        try:
            while True:
                _release_p2p_receive_leases(self._queue.get_nowait())
        except queue.Empty:
            pass
        self._exhausted = True

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move a batch of tensors to the target device (GPU)."""
        if self.target_device is None:
            return batch
        return {
            k: v.to(self.target_device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def __next__(self) -> Dict[str, torch.Tensor]:
        if self._stop or self._exhausted:
            raise StopIteration
        if self._error is not None:
            raise self._error
        item = self._queue.get()
        if item is self._SENTINEL:
            self._exhausted = True
            if self._error is not None:
                raise self._error
            raise StopIteration
        try:
            return self._to_device(item)
        except BaseException:
            _release_p2p_receive_leases(item)
            raise

    def get_feature_contract_status(self) -> dict[str, Any]:
        return self.inner.get_feature_contract_status()
