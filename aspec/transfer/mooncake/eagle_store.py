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

import ctypes
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

from aspec.transfer.mooncake.helpers import _format_bytes
from aspec.transfer.mooncake.store import MooncakeHiddenStateStore
from aspec.utils.distributed import get_usp_rank_coords
from aspec.utils.logging import logger
from aspec.utils.usp import split_usp_batch

if TYPE_CHECKING:
    from aspec.models.target.eagle3_target_model import Eagle3TargetOutput
    from aspec.transfer.mooncake.p2p import MooncakeP2PReservation

# Static lookup for dtype → element size in bytes (avoids creating a tensor
# on every call to _compute_tensor_size).
_DTYPE_ELEMENT_SIZES = {
    torch.float64: 8,
    torch.float32: 4,
    torch.bfloat16: 2,
    torch.float16: 2,
    torch.int64: 8,
    torch.int32: 4,
    torch.int16: 2,
    torch.int8: 1,
    torch.uint8: 1,
    torch.bool: 1,
}

# Canonical dtype for hidden-state tensors written to / read from Mooncake.
HIDDEN_STATES_STORAGE_DTYPE = torch.bfloat16


class EagleMooncakeStore(MooncakeHiddenStateStore):
    """
    Mooncake Store wrapper specialized for Eagle3 hidden states.

    Uses RDMA-registered host buffers and put_from for zero-copy transfers.
    Each Eagle3 output is stored as multiple tensors with key suffixes:
    - {key}_hs: hidden_states
    - {key}_tgt: target
    - {key}_ids: input_ids
    - {key}_lhs: last_hidden_states (if present)

    Deletions use ``batch_remove(force=True)`` for immediate cleanup.
    """

    TENSOR_SUFFIXES = ["_hs", "_tgt", "_ids", "_lhs", "_mhs"]

    def reserve_p2p_output(
        self,
        *,
        key: str,
        sequence_length: int,
        feature_sequence_length: int | None = None,
        hidden_size: int,
        num_aux_layers: int,
        store_last_hidden_states: bool,
        data_id: str = "",
        request_id: str = "",
        source_engine: str = "",
        store_mtp_hidden_states: bool = False,
    ) -> "MooncakeP2PReservation":
        """Reserve the final P2P layout for direct Target-prefill writes."""

        self._ensure_initialized()
        if self.config.data_plane != "transfer_engine" or self._p2p_transfer is None:
            raise RuntimeError(
                "direct Target-prefill output requires Mooncake transfer_engine"
            )
        if feature_sequence_length is None:
            feature_sequence_length = sequence_length
        if (
            sequence_length <= 0
            or feature_sequence_length <= 0
            or hidden_size <= 0
            or num_aux_layers <= 0
        ):
            raise ValueError(
                "direct Target-prefill output requires positive token/feature "
                "sequence lengths, hidden size, and auxiliary layer count"
            )
        tensors = [
            (
                "hidden_states",
                (feature_sequence_length, hidden_size * num_aux_layers),
                HIDDEN_STATES_STORAGE_DTYPE,
            ),
            ("input_ids", (sequence_length,), torch.int64),
        ]
        if store_last_hidden_states:
            tensors.append(
                (
                    "last_hidden_states",
                    (feature_sequence_length, hidden_size),
                    HIDDEN_STATES_STORAGE_DTYPE,
                )
            )
        if store_mtp_hidden_states:
            tensors.append(("mtp_hidden_states", (feature_sequence_length, hidden_size), HIDDEN_STATES_STORAGE_DTYPE))
        return self._p2p_transfer.reserve(
            feature_layer_width=(hidden_size if os.getenv("ASPEC_SHARED_FEATURE_VIEWS") == "1" else None),
            store_key=key,
            tensors=tensors,
            run_id=os.getenv("ASPEC_RUN_ID", ""),
            attempt_id=os.getenv("ASPEC_ATTEMPT_ID", ""),
            data_id=data_id,
            request_id=request_id,
            source_engine=source_engine,
        )

    def put_raw_tensors(self, keys: List[str], tensors: List[torch.Tensor]) -> None:
        """Publish an already-normalized group of tensors in one batch.

        The vLLM completed-request connector uses this to pack fragments from
        multiple requests into each registered host buffer. This avoids one
        serialized ``batch_put_from`` call per layer while retaining the same
        per-layer Mooncake keys consumed by ``MooncakeDataset``.
        """
        self._ensure_initialized()
        if not keys or len(keys) != len(tensors):
            raise ValueError(
                f"Expected equal non-empty keys/tensors, got {len(keys)}/{len(tensors)}"
            )
        self._put_raw_tensors(keys, tensors)

    def _put_raw_tensors(self, keys: List[str], tensors: List[torch.Tensor]) -> None:
        if self._gpu_direct_available and self._gpu_send_buffer is not None:
            buf = self._gpu_send_buffer
            buffer_ptrs, sizes = self._stage_tensors_into_buffer(buf, tensors)
            # copy_from_tensor() enqueues device-to-device copies on the
            # current stream. Mooncake does not participate in CUDA stream
            # ordering, so make those writes visible before the NIC reads the
            # registered send buffer.
            torch.cuda.current_stream(buf.device).synchronize()
            self._do_sync_batch_put(keys, buffer_ptrs, sizes)
        elif self._host_buffer_pool is None or self._async_put_manager is None:
            raise RuntimeError(
                "put() requires either GPU Direct (enable_gpu_direct=True) or "
                "async host-buffer puts (async_put_pool_size > 0). "
                "Current config has async_put_pool_size=0 and GPU Direct is "
                f"{'enabled but gpu_send_buffer failed to initialize' if self._gpu_direct_available else 'disabled'}. "
                "Set async_put_pool_size >= 1 or enable GPU Direct."
            )
        else:
            buf = self._host_buffer_pool.get_buffer()
            self._async_put_manager.check_last_error()
            self._async_put_manager.wait_for_buffer(buf.ptr)

            # Offline replay tensors are already on the host, so CUDA event and
            # stream synchronization is both unnecessary and unavailable.
            if not any(tensor.is_cuda for tensor in tensors):
                buffer_ptrs, sizes = self._stage_tensors_into_buffer(buf, tensors)
                self._async_put_manager.submit(keys, buffer_ptrs, sizes, buf.ptr)
                return

            compute_event = torch.cuda.Event()
            compute_event.record()

            with torch.cuda.stream(self._copy_stream):
                self._copy_stream.wait_event(compute_event)
                buffer_ptrs, sizes = self._stage_tensors_into_buffer(buf, tensors)
                copy_done = torch.cuda.Event()
                copy_done.record()

            for t in tensors:
                if t.is_cuda:
                    t.record_stream(self._copy_stream)

            self._async_put_manager.submit(
                keys,
                buffer_ptrs,
                sizes,
                buf.ptr,
                wait_event=copy_done,
                device_index=self._copy_stream.device.index,
            )

    def put_usp_shards(
        self,
        key: str,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        last_hidden_states: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
        *,
        sp_size: int,
        sp_ring_size: int,
        ttt_length: int,
        max_seq_length: Optional[int],
    ) -> Dict[str, Any]:
        self._ensure_initialized()
        if self.config.data_plane == "transfer_engine":
            raise NotImplementedError(
                "transfer_engine data_plane does not support USP-sharded payloads"
            )
        logger.debug("put_usp_shards: starting for key=%s, sp_size=%s", key, sp_size)

        if hidden_states.dtype != HIDDEN_STATES_STORAGE_DTYPE:
            hidden_states = hidden_states.to(HIDDEN_STATES_STORAGE_DTYPE)
        if (
            last_hidden_states is not None
            and last_hidden_states.dtype != HIDDEN_STATES_STORAGE_DTYPE
        ):
            last_hidden_states = last_hidden_states.to(HIDDEN_STATES_STORAGE_DTYPE)
        if target is not None and target.dtype != HIDDEN_STATES_STORAGE_DTYPE:
            target = target.to(HIDDEN_STATES_STORAGE_DTYPE)

        target_hs = last_hidden_states if last_hidden_states is not None else target
        if target_hs is None:
            raise ValueError("USP sharded Mooncake storage requires last_hidden_states or target")

        if input_ids.dim() == 2:
            loss_mask_len = input_ids.shape[-1]
        else:
            loss_mask_len = input_ids.shape[0]
        dummy_loss_mask = torch.ones((1, loss_mask_len), dtype=torch.long, device=input_ids.device)
        sp_ulysses_size = max(1, sp_size // sp_ring_size)

        for sp_rank in range(sp_size):
            _, ring_rank = get_usp_rank_coords(
                sp_rank=sp_rank,
                sp_ulysses_size=sp_ulysses_size,
                sp_ring_size=sp_ring_size,
            )
            (
                shard_input_ids,
                _attention_mask,
                _loss_mask,
                shard_hidden_states,
                shard_target_hs,
                _position_ids,
            ) = split_usp_batch(
                input_ids=input_ids,
                loss_mask=dummy_loss_mask,
                hidden_states=hidden_states,
                target_hidden_states=target_hs,
                ttt_length=ttt_length,
                sp_rank=sp_rank,
                sp_size=sp_size,
                ring_rank=ring_rank,
                sp_ring_size=sp_ring_size,
                max_len=max_seq_length,
            )

            shard_key = f"{key}_usp{sp_rank}"
            keys = [f"{shard_key}_hs", f"{shard_key}_ids"]
            tensors = [shard_hidden_states, shard_input_ids]
            if last_hidden_states is not None:
                keys.append(f"{shard_key}_lhs")
                tensors.append(shard_target_hs)
            else:
                keys.append(f"{shard_key}_tgt")
                tensors.append(shard_target_hs)
            self._put_raw_tensors(keys, tensors)

        shapes = {
            "hidden_states": tuple(hidden_states.shape),
            "input_ids": tuple(input_ids.shape),
        }
        dtypes = {
            "hidden_states": hidden_states.dtype,
            "input_ids": input_ids.dtype,
        }
        if target is not None:
            shapes["target"] = tuple(target.shape)
            dtypes["target"] = target.dtype
        if last_hidden_states is not None:
            shapes["last_hidden_states"] = tuple(last_hidden_states.shape)
            dtypes["last_hidden_states"] = last_hidden_states.dtype

        logger.debug("put_usp_shards: completed key=%s, shapes=%s", key, shapes)
        return {"shapes": shapes, "dtypes": dtypes}

    def put_direct_cp_shards(
        self,
        key: str,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        last_hidden_states: Optional[torch.Tensor],
        target: Optional[torch.Tensor] = None,
        *,
        cp_size: int,
        max_seq_length: Optional[int],
        context_shard_alignment: int = 1,
        data_id: str = "",
        request_id: str = "",
        source_engine: str = "",
        mtp_hidden_states: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Store contiguous context shards while replicating the small token row."""

        self._ensure_initialized()
        if cp_size <= 0:
            raise ValueError(f"Direct CP size must be positive, got {cp_size}")

        if hidden_states.dtype != HIDDEN_STATES_STORAGE_DTYPE:
            hidden_states = hidden_states.to(HIDDEN_STATES_STORAGE_DTYPE)
        if last_hidden_states is not None and last_hidden_states.dtype != HIDDEN_STATES_STORAGE_DTYPE:
            last_hidden_states = last_hidden_states.to(HIDDEN_STATES_STORAGE_DTYPE)
        if target is not None and target.dtype != HIDDEN_STATES_STORAGE_DTYPE:
            target = target.to(HIDDEN_STATES_STORAGE_DTYPE)

        sequence_dim = 1 if hidden_states.dim() == 3 else 0
        sequence_length = min(
            input_ids.shape[-1],
            max_seq_length if max_seq_length is not None else input_ids.shape[-1],
        )
        if hidden_states.shape[sequence_dim] < sequence_length:
            raise ValueError(
                "hidden_states is shorter than the Direct CP sequence: "
                f"{hidden_states.shape[sequence_dim]} < {sequence_length}"
            )
        replicated_ids = input_ids[..., :sequence_length].contiguous()
        if cp_size == 1:
            # CP1 uses the ordinary (unsharded) consumer. Do not publish a
            # synthetic _cp0 namespace or descriptor bundle it cannot read.
            def unsharded_slice(tensor):
                if tensor is None:
                    return None
                dim = 1 if tensor.dim() == 3 else 0
                return tensor.narrow(dim, 0, sequence_length).contiguous()

            return self.put(
                key,
                unsharded_slice(hidden_states),
                replicated_ids,
                unsharded_slice(last_hidden_states),
                unsharded_slice(target),
                data_id=data_id,
                request_id=request_id,
                source_engine=source_engine,
                **({"mtp_hidden_states": unsharded_slice(mtp_hidden_states)}
                   if mtp_hidden_states is not None else {}),
            )
        if mtp_hidden_states is not None:
            raise ValueError("MTP features require an unsharded CP1 consumer")
        p2p_descriptors = []
        try:
            for cp_rank in range(cp_size):
                from aspec.inference.context import context_shard_bounds

                start, end = context_shard_bounds(
                    sequence_length,
                    cp_rank,
                    cp_size,
                    alignment=context_shard_alignment,
                )

                def context_slice(tensor: torch.Tensor) -> torch.Tensor:
                    dim = 1 if tensor.dim() == 3 else 0
                    return tensor.narrow(dim, start, end - start).contiguous()

                shard_key = f"{key}_cp{cp_rank}"
                keys = [f"{shard_key}_hs", f"{shard_key}_ids"]
                names = ["hidden_states", "input_ids"]
                tensors = [context_slice(hidden_states), replicated_ids]
                if target is not None:
                    keys.append(f"{shard_key}_tgt")
                    names.append("target")
                    tensors.append(context_slice(target))
                if last_hidden_states is not None:
                    keys.append(f"{shard_key}_lhs")
                    names.append("last_hidden_states")
                    tensors.append(context_slice(last_hidden_states))
                if self.config.data_plane == "transfer_engine":
                    if self._p2p_transfer is None:
                        raise RuntimeError("Mooncake TransferEngine source is not initialized")
                    p2p_descriptors.append(
                        self._p2p_transfer.stage(
                            store_key=shard_key,
                            tensors=zip(names, tensors),
                            run_id=os.getenv("ASPEC_RUN_ID", ""),
                            attempt_id=os.getenv("ASPEC_ATTEMPT_ID", ""),
                            data_id=data_id,
                            request_id=request_id,
                            source_engine=source_engine,
                        )
                    )
                else:
                    self._put_raw_tensors(keys, tensors)
        except BaseException:
            for descriptor in reversed(p2p_descriptors):
                self._p2p_transfer.discard_staged(descriptor)
            raise

        shapes = {
            "hidden_states": tuple(hidden_states.shape),
            "input_ids": tuple(replicated_ids.shape),
        }
        dtypes = {
            "hidden_states": hidden_states.dtype,
            "input_ids": replicated_ids.dtype,
        }
        if target is not None:
            shapes["target"] = tuple(target.shape)
            dtypes["target"] = target.dtype
        if last_hidden_states is not None:
            shapes["last_hidden_states"] = tuple(last_hidden_states.shape)
            dtypes["last_hidden_states"] = last_hidden_states.dtype
        descriptor_key = key
        if p2p_descriptors:
            from aspec.transfer.mooncake.p2p import pack_direct_cp_p2p_keys

            descriptor_key = pack_direct_cp_p2p_keys(
                descriptor.to_key() for descriptor in p2p_descriptors
            )
        return {"shapes": shapes, "dtypes": dtypes, "mooncake_key": descriptor_key}

    def put(
        self,
        key: str,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        last_hidden_states: Optional[torch.Tensor],
        target: Optional[torch.Tensor] = None,
        *,
        data_id: str = "",
        request_id: str = "",
        source_engine: str = "",
        mtp_hidden_states: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Store Eagle3 output tensors via async batch_put_from.

        DtoH staging runs on ``_copy_stream`` so the caller's compute stream
        is never blocked.  The RDMA transfer runs on a background thread via
        ``AsyncPutManager``.  With *pool_size* host buffers the caller almost
        never waits — ``wait_for_buffer`` only blocks when every buffer is
        still in-flight.

        For GPU Direct send the path is synchronous (no DtoH needed).

        Returns a dict with ``"shapes"`` and ``"dtypes"`` sub-dicts that
        reflect the *actually stored* tensors (post dtype-cast).  Callers
        should forward these to consumers so metadata always matches bytes.
        """
        self._ensure_initialized()
        logger.debug("put: starting for key=%s", key)

        if hidden_states.dtype != HIDDEN_STATES_STORAGE_DTYPE:
            hidden_states = hidden_states.to(HIDDEN_STATES_STORAGE_DTYPE)
        if (
            last_hidden_states is not None
            and last_hidden_states.dtype != HIDDEN_STATES_STORAGE_DTYPE
        ):
            last_hidden_states = last_hidden_states.to(HIDDEN_STATES_STORAGE_DTYPE)
        if target is not None and target.dtype != HIDDEN_STATES_STORAGE_DTYPE:
            target = target.to(HIDDEN_STATES_STORAGE_DTYPE)

        if mtp_hidden_states is not None:
            if tuple(mtp_hidden_states.shape) != (input_ids.numel(), self.config.hidden_dim):
                raise ValueError("MTP hidden states must align with input tokens and hidden width")
            mtp_hidden_states = mtp_hidden_states.to(HIDDEN_STATES_STORAGE_DTYPE)

        keys = [f"{key}_hs", f"{key}_ids"]
        tensors = [hidden_states, input_ids]

        if target is not None:
            keys.append(f"{key}_tgt")
            tensors.append(target)

        if last_hidden_states is not None:
            keys.append(f"{key}_lhs")
            tensors.append(last_hidden_states)

        if mtp_hidden_states is not None:
            keys.append(f"{key}_mhs")
            tensors.append(mtp_hidden_states)

        descriptor_key = key
        if self.config.data_plane == "transfer_engine":
            if self._p2p_transfer is None:
                raise RuntimeError("Mooncake TransferEngine source is not initialized")
            names = ["hidden_states", "input_ids"]
            if target is not None:
                names.append("target")
            if last_hidden_states is not None:
                names.append("last_hidden_states")
            if mtp_hidden_states is not None:
                names.append("mtp_hidden_states")
            descriptor = self._p2p_transfer.stage(
                store_key=key,
                tensors=zip(names, tensors),
                run_id=os.getenv("ASPEC_RUN_ID", ""),
                attempt_id=os.getenv("ASPEC_ATTEMPT_ID", ""),
                data_id=data_id,
                request_id=request_id,
                source_engine=source_engine,
            )
            descriptor_key = descriptor.to_key()
        else:
            self._put_raw_tensors(keys, tensors)

        shapes = {
            "hidden_states": tuple(hidden_states.shape),
            "input_ids": tuple(input_ids.shape),
        }
        dtypes = {
            "hidden_states": hidden_states.dtype,
            "input_ids": input_ids.dtype,
        }
        if target is not None:
            shapes["target"] = tuple(target.shape)
            dtypes["target"] = target.dtype
        if last_hidden_states is not None:
            shapes["last_hidden_states"] = tuple(last_hidden_states.shape)
            dtypes["last_hidden_states"] = last_hidden_states.dtype

        if mtp_hidden_states is not None:
            shapes["mtp_hidden_states"] = tuple(mtp_hidden_states.shape)
            dtypes["mtp_hidden_states"] = mtp_hidden_states.dtype

        logger.debug("put: completed key=%s, shapes=%s", key, shapes)
        return {"shapes": shapes, "dtypes": dtypes, "mooncake_key": descriptor_key}

    def flush(self) -> None:
        """Block until all in-flight async puts have completed.

        Called before returning mooncake keys to the controller so that
        consumers can GET immediately.  Because ``put()`` stages DtoH on
        ``_copy_stream``, the copies are typically finished by the time
        this is called — the wait is only for the (fast) RDMA transfer.
        """
        self._ensure_initialized()
        if self.config.data_plane == "transfer_engine":
            self._p2p_transfer.check_last_error()
            return
        if self._async_put_manager is None:
            return
        self._async_put_manager.check_last_error()
        self._async_put_manager.drain()
        self._async_put_manager.check_last_error()

    def wait_for_p2p_source_releases(self, timeout: float = 5.0) -> bool:
        """Keep source allocations alive until every consumer returns its lease."""

        self._ensure_initialized()
        if self._p2p_transfer is None:
            return True
        return self._p2p_transfer.wait_for_source_releases(timeout)

    def _do_sync_batch_put(
        self,
        keys: List[str],
        buffer_ptrs: List[int],
        sizes: List[int],
    ) -> None:
        """Synchronous batch_put_from with error handling."""
        total_bytes = sum(sizes)
        if self._replicate_config is not None:
            results = self._store.batch_put_from(
                keys, buffer_ptrs, sizes, config=self._replicate_config
            )
        else:
            results = self._store.batch_put_from(keys, buffer_ptrs, sizes)
        failures = [(k, r) for k, r in zip(keys, results) if r != 0]
        if failures:
            try:
                self._store.batch_remove(keys, force=True)
            except Exception:
                logger.warning(
                    "Failed to cleanup keys after batch_put_from failure: %s",
                    keys,
                    exc_info=True,
                )
            failure_details = ", ".join(f"{k} (code={r})" for k, r in failures)
            config_details = (
                f"total_bytes={_format_bytes(total_bytes)}, "
                f"global_segment_size={_format_bytes(self.config.global_segment_size)}, "
                f"local_buffer_size={_format_bytes(self.config.local_buffer_size)}, "
                f"host_buffer_size={_format_bytes(self.config.host_buffer_size)}"
            )
            raise RuntimeError(
                f"batch_put_from failed for keys: {failure_details}. "
                f"{config_details}. Consider increasing Mooncake segment/buffer sizes "
                "or reducing batch/sequence length/prefetch depth."
            )

    @staticmethod
    def _stage_tensors_into_buffer(buf, tensors: List[torch.Tensor]) -> Tuple[List[int], List[int]]:
        """Copy tensors into a buffer and return (pointers, sizes) for batch_put_from."""
        buffer_ptrs = []
        sizes = []
        offset = 0
        for tensor in tensors:
            nbytes = buf.copy_from_tensor(tensor, offset=offset)
            buffer_ptrs.append(buf.ptr + offset)
            sizes.append(nbytes)
            offset += nbytes
        return buffer_ptrs, sizes

    def get(
        self,
        key: str,
        shapes: Dict[str, Tuple[int, ...]],
        dtypes: Dict[str, torch.dtype],
        device: torch.device,
    ) -> "Eagle3TargetOutput":
        """
        Retrieve Eagle3 tensors into GPU memory.

        For RDMA/InfiniBand: Uses GPUDirect RDMA (batch_get_into directly into GPU).
        For TCP: Uses batch_get_buffer to host buffer, then copies to GPU.

        The Store/TCP bundle uses host buffers as its selected data path. A
        GPU-direct Store configuration may separately allow fallback to it.

        Returns:
            Eagle3TargetOutput with the retrieved tensors.
        """
        self._ensure_initialized()

        from aspec.models.target.eagle3_target_model import Eagle3TargetOutput

        if self.config.data_plane == "transfer_engine":
            from aspec.transfer.mooncake.p2p import MooncakeP2PDescriptor

            descriptor = MooncakeP2PDescriptor.from_key(key)
            read_result = self._p2p_transfer.read(descriptor)
            tensor_map = read_result.tensors
            descriptor_shapes = {
                tensor.name: tensor.shape for tensor in descriptor.tensors
            }
            expected_shapes = {name: tuple(shape) for name, shape in shapes.items()}
            if descriptor_shapes != expected_shapes:
                read_result.lease.release()
                raise RuntimeError(
                    "Mooncake P2P descriptor shape metadata does not match controller "
                    f"metadata: descriptor={descriptor_shapes}, controller={expected_shapes}"
                )
            return Eagle3TargetOutput(
                hidden_states=tensor_map["hidden_states"],
                target=tensor_map.get("target"),
                input_ids=tensor_map["input_ids"],
                last_hidden_states=tensor_map.get("last_hidden_states"),
                input_ids_cpu=None,
                p2p_receive_lease=read_result.lease,
                mtp_hidden_states=tensor_map.get("mtp_hidden_states"),
            )

        keys = [f"{key}_hs", f"{key}_ids"]
        tensor_specs = [
            (
                "hidden_states",
                shapes["hidden_states"],
                dtypes.get("hidden_states", HIDDEN_STATES_STORAGE_DTYPE),
            ),
            ("input_ids", shapes["input_ids"], torch.int64),
        ]

        if "target" in shapes:
            keys.append(f"{key}_tgt")
            tensor_specs.append(
                ("target", shapes["target"], dtypes.get("target", HIDDEN_STATES_STORAGE_DTYPE))
            )

        if "last_hidden_states" in shapes:
            keys.append(f"{key}_lhs")
            tensor_specs.append(
                (
                    "last_hidden_states",
                    shapes["last_hidden_states"],
                    dtypes.get("hidden_states", HIDDEN_STATES_STORAGE_DTYPE),
                )
            )

        if "mtp_hidden_states" in shapes:
            keys.append(f"{key}_mhs")
            tensor_specs.append(("mtp_hidden_states", shapes["mtp_hidden_states"],
                                 dtypes.get("mtp_hidden_states", HIDDEN_STATES_STORAGE_DTYPE)))

        # Publication is asynchronous on inference workers.  Wait for a
        # complete metadata census before attempting either byte-transfer
        # path; this also gives GPUDirect the retry/fail-closed behavior that
        # previously existed only in the host-buffer fallback.
        self.wait_for_keys(keys)

        tensor_map = None
        if self._gpu_direct_available and self._gpu_receive_buffer is not None:
            tensor_map = self._get_tensors_gpu_direct(keys, tensor_specs, device)
            if tensor_map is None:
                if not self.config.allow_host_fallback:
                    raise RuntimeError(
                        "GPU-direct batch_get_into failed and Mooncake host fallback is "
                        "disabled."
                    )
                logger.warning(
                    "GPUDirect batch_get_into failed; falling back to host buffer path."
                )

        if tensor_map is None:
            host_staging_selected = not self.config.enable_gpu_direct
            if not host_staging_selected and not self.config.allow_host_fallback:
                raise RuntimeError(
                    "Mooncake GPU-direct receive path is unavailable and host fallback is "
                    "disabled."
                )
            tensor_map = self._get_tensors_via_host_buffer(keys, tensor_specs, device)
            logger.debug(
                "Using %s host buffer path (TCP)",
                "selected" if host_staging_selected else "fallback",
            )
        logger.debug("Retrieved Eagle3 tensors with base key: %s", key)

        return Eagle3TargetOutput(
            hidden_states=tensor_map["hidden_states"],
            target=tensor_map.get("target"),
            input_ids=tensor_map["input_ids"],
            last_hidden_states=tensor_map.get("last_hidden_states"),
            input_ids_cpu=tensor_map.get("input_ids_cpu"),
            mtp_hidden_states=tensor_map.get("mtp_hidden_states"),
        )

    def _get_tensors_gpu_direct(
        self,
        keys: List[str],
        tensor_specs: List[Tuple[str, Tuple[int, ...], torch.dtype]],
        device: torch.device,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Transfer directly into GPU memory using batch_get_into (GPUDirect RDMA).

        Pre-allocates individual destination tensors and transfers directly into
        their memory, avoiding an extra GPU→GPU copy.  Falls back to the
        registered GPU receive buffer when a tensor is too small to be
        page-aligned (RDMA typically requires registered memory).

        Returns None if transfer fails, allowing caller to fall back to host
        buffer path.
        """
        total_size = sum(
            self._compute_tensor_size(shape, dtype) for _, shape, dtype in tensor_specs
        )

        if total_size > self._gpu_receive_buffer.size:
            logger.warning(
                "GPU buffer too small: need %.1fMB, have %.1fMB. Increase gpu_buffer_size in config.",
                total_size / (1024**2),
                self._gpu_receive_buffer.size / (1024**2),
            )
            return None

        # Compute per-tensor sizes and buffer offsets up-front.
        buffer_ptrs: List[int] = []
        sizes: List[int] = []
        offsets: List[int] = []
        offset = 0

        for _, shape, dtype in tensor_specs:
            size = self._compute_tensor_size(shape, dtype)
            buffer_ptrs.append(self._gpu_receive_buffer.ptr + offset)
            sizes.append(size)
            offsets.append(offset)
            offset += size

        try:
            results = self._store.batch_get_into(keys, buffer_ptrs, sizes)
            for i, (k, r) in enumerate(zip(keys, results)):
                if r < 0:
                    logger.warning("batch_get_into failed for %s with error code: %s", k, r)
                    return None
                if r != 0 and r != sizes[i]:
                    logger.warning(
                        "batch_get_into for %s: unexpected return %s (expected 0 or %s)",
                        k,
                        r,
                        sizes[i],
                    )
        except Exception as e:
            logger.warning("batch_get_into exception: %s", e)
            return None

        tensor_map = {}
        for i, (name, shape, dtype) in enumerate(tensor_specs):
            numel = 1
            for dim in shape:
                numel *= dim
            buf_slice = self._gpu_receive_buffer.get_slice(offsets[i], sizes[i])
            # View into the registered buffer; valid until the next call.
            tensor_map[name] = buf_slice.view(dtype)[:numel].reshape(shape)

        logger.debug("GPU Direct RDMA transfer successful for %s tensors", len(keys))
        return tensor_map

    @staticmethod
    def _compute_tensor_size(shape: Tuple[int, ...], dtype: torch.dtype) -> int:
        """Compute the byte size of a tensor with given shape and dtype."""
        numel = 1
        for dim in shape:
            numel *= dim
        return numel * _DTYPE_ELEMENT_SIZES[dtype]

    def _get_tensors_via_host_buffer(
        self,
        keys: List[str],
        tensor_specs: List[Tuple[str, Tuple[int, ...], torch.dtype]],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Transfer via Mooncake's registered host buffer, then copy to device."""
        wait_seconds = max(self.config.get_retry_wait_seconds, 0.05)
        log_interval = max(self.config.get_retry_log_interval_seconds, wait_seconds)
        max_wait = max(self.config.get_retry_max_wait_seconds, 0.0)
        start_time = time.time()
        last_log = 0.0

        while True:
            buffers = self._store.batch_get_buffer(keys)
            missing = [i for i, buf in enumerate(buffers) if buf is None]
            if not missing:
                break

            elapsed = time.time() - start_time
            if max_wait > 0 and elapsed >= max_wait:
                missing_keys = ", ".join(keys[i] for i in missing)
                raise RuntimeError(
                    f"batch_get_buffer returned None for keys: {missing_keys}. "
                    f"Waited {elapsed:.1f}s; aborting."
                )

            now = time.time()
            if last_log == 0.0 or (now - last_log) >= log_interval:
                missing_keys = ", ".join(keys[i] for i in missing)
                logger.warning(
                    "batch_get_buffer missing keys (%s); sleeping %.2fs.",
                    missing_keys,
                    wait_seconds,
                )
                last_log = now
            time.sleep(wait_seconds)

        tensor_map = {}
        for i, ((name, shape, dtype), buf) in enumerate(zip(tensor_specs, buffers)):
            if buf is None:
                raise RuntimeError(
                    f"batch_get_buffer returned None for key '{keys[i]}' (tensor: {name}). "
                    "This may indicate the key doesn't exist or RDMA transfer failed."
                )

            numel = 1
            for dim in shape:
                numel *= dim
            element_size = _DTYPE_ELEMENT_SIZES[dtype]
            expected_size = numel * element_size

            buf_size = buf.size()
            if buf_size != expected_size:
                actual_elements = buf_size // element_size if element_size > 0 else 0
                logger.error(
                    f"Size mismatch for {name} (key={keys[i]}): "
                    f"got {buf_size} bytes ({actual_elements} elements), "
                    f"expected {expected_size} bytes ({numel} elements). "
                    f"Expected shape: {shape}, dtype: {dtype}, element_size: {element_size}"
                )
                raise RuntimeError(
                    f"Size mismatch for {name}: got {buf_size} bytes, expected {expected_size} bytes"
                )

            c_array = (ctypes.c_byte * buf_size).from_address(buf.ptr())
            host_tensor = torch.frombuffer(c_array, dtype=dtype, count=numel).reshape(shape)

            # A CPU .to() otherwise aliases memory owned by the temporary
            # Mooncake buffer. Materialization must own the bytes after return.
            tensor_map[name] = host_tensor.to(device, copy=True)

            if name == "input_ids":
                tensor_map["input_ids_cpu"] = host_tensor.clone()

        return tensor_map

    def discard_unpublished_objects(self, keys: List[str]) -> None:
        """Drain writes and force-delete only producer-owned, unsent objects.

        Unlike consumer best-effort deletion, failure must keep the publisher's
        ownership record live. Exact keys include partially written CP/USP
        shards, which may never have produced a descriptor.
        """
        self._ensure_initialized()
        if self.config.data_plane != "store":
            raise RuntimeError("Unpublished object deletion requires the Store data plane")
        if self._async_put_manager is not None:
            # flush() may surface an earlier error before draining. No queued
            # put may finish after deletion and resurrect an abandoned object.
            self._async_put_manager.drain()
        remaining = list(dict.fromkeys(keys))
        if not remaining:
            return
        last_error = None
        for attempt in range(3):
            try:
                results = list(self._store.batch_remove(remaining, force=True))
                if len(results) != len(remaining):
                    raise RuntimeError("batch_remove returned an incomplete result")
                remaining = [key for key, code in zip(remaining, results, strict=True)
                             if code not in (None, 0, -704)]
                if not remaining:
                    return
                last_error = RuntimeError(f"batch_remove failed: {results}")
            except Exception as exc:
                last_error = exc
            if attempt < 2:
                time.sleep(0.5)
        raise RuntimeError(f"Failed to delete unpublished Store objects: {remaining}") from last_error

    def remove_eagle3_tensors(
        self,
        key: str,
        has_last_hidden_states: bool = False,
        has_target: bool = False,
        has_mtp_hidden_states: bool = False,
    ) -> None:
        """Force-delete all tensors associated with an Eagle3 output.

        Uses ``batch_remove(force=True)`` to bypass lease TTL and delete
        immediately after consumption. Retries up to 3 times on failure.
        Never raises — deletion is best-effort to avoid breaking the
        training fetch path.

        Args:
            key: Base key used when storing
            has_last_hidden_states: Whether last_hidden_states was stored
            has_target: Whether target (logits) was stored
        """
        if self.config.data_plane == "transfer_engine":
            # The successful READ already notified the source and there is no
            # Store object. Parsing rejects accidental data-plane mixing.
            from aspec.transfer.mooncake.p2p import MooncakeP2PDescriptor

            MooncakeP2PDescriptor.from_key(key)
            return

        keys = [f"{key}_hs", f"{key}_ids"]
        if has_target:
            keys.append(f"{key}_tgt")
        if has_last_hidden_states:
            keys.append(f"{key}_lhs")
        if has_mtp_hidden_states:
            keys.append(f"{key}_mhs")

        for attempt in range(1, 4):
            try:
                results = self._store.batch_remove(keys, force=True)
            except Exception:
                if attempt < 3:
                    logger.warning(
                        "batch_remove raised for %s (attempt %d/3)", key, attempt, exc_info=True
                    )
                    time.sleep(0.5)
                else:
                    logger.error(
                        "Force delete abandoned for %s after 3 exceptions", key, exc_info=True
                    )
                continue
            failed = [(k, r) for k, r in zip(keys, results) if r not in (None, 0, -704)]
            if not failed:
                logger.debug("Force-deleted %s (%d keys)", key, len(results))
                return
            if attempt < 3:
                time.sleep(0.5)
                logger.warning(
                    "Retrying force delete for %s: %d keys failed (attempt %d/3)",
                    key,
                    len(failed),
                    attempt,
                )
            else:
                logger.error("Force delete abandoned for %s: %s", key, failed)
                return
            keys = [k for k, _ in failed]

    def discard_unconsumed_eagle3_tensors(
        self,
        key: str,
        has_last_hidden_states: bool = False,
        has_target: bool = False,
        has_mtp_hidden_states: bool = False,
    ) -> None:
        """Release storage for an output that no trainer consumed.

        TransferEngine source slots are released only by the descriptor's
        generation-checked read acknowledgement. During bounded shutdown we
        therefore read each descriptor into the destination ring and release
        it immediately. The Store data plane retains its force-delete path.
        """

        if self.config.data_plane != "transfer_engine":
            self.remove_eagle3_tensors(
                key,
                has_last_hidden_states=has_last_hidden_states,
                has_target=has_target,
                **({"has_mtp_hidden_states": True} if has_mtp_hidden_states else {}),
            )
            return
        if self._p2p_transfer is None:
            raise RuntimeError("Mooncake TransferEngine destination is not initialized")

        from aspec.transfer.mooncake.p2p import p2p_descriptors_from_key

        for descriptor in p2p_descriptors_from_key(key):
            read_result = self._p2p_transfer.read(descriptor)
            read_result.lease.release()
