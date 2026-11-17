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

"""Bounded output lifecycle for no-decode Target prefill."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class TargetPrefillRequest:
    request_id: str
    data_id: str
    sequence_length: int


class TargetPrefillDirectSink:
    """Writes requested Target features directly into registered P2P slots."""

    is_target_prefill_direct_sink = True

    def __init__(
        self,
        *,
        store,
        requests: Sequence[TargetPrefillRequest],
        capture_layers: Sequence[int],
        hidden_size: int,
        store_last_hidden_states: bool,
        final_hidden_as_aux: bool = False,
        source_engine: str = "",
        context_parallel_size: int | None = None,
        context_shard_alignment: int | None = None,
        mtp_feature_producer=None,
    ) -> None:
        if not requests:
            raise ValueError("Target prefill direct output requires at least one request")
        if not capture_layers or len(set(capture_layers)) != len(capture_layers):
            raise ValueError("Target prefill capture layers must be non-empty and unique")
        if hidden_size <= 0:
            raise ValueError("Target prefill hidden size must be positive")
        if any(request.sequence_length <= 0 for request in requests):
            raise ValueError("Target prefill request lengths must be positive")
        if context_parallel_size is None:
            context_parallel_size = int(os.getenv("ASPEC_DIRECT_CP_SIZE", "1"))
        if context_shard_alignment is None:
            context_shard_alignment = int(
                os.getenv("ASPEC_DIRECT_CP_ALIGNMENT", "1")
            )
        if context_parallel_size <= 0:
            raise ValueError(
                "Target prefill context parallel size must be positive, "
                f"got {context_parallel_size}"
            )
        if context_shard_alignment <= 0:
            raise ValueError("Target prefill context shard alignment must be positive")
        if any(request.sequence_length < context_parallel_size for request in requests):
            raise ValueError(
                "Target prefill request lengths must be at least the context parallel size"
            )
        request_ids = [request.request_id for request in requests]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("Target prefill request ids must be unique within a batch")

        self.store = store
        self.requests = tuple(requests)
        self.capture_layers = tuple(int(layer) for layer in capture_layers)
        self.hidden_size = int(hidden_size)
        if final_hidden_as_aux and (len(capture_layers) != 1 or store_last_hidden_states):
            raise ValueError("Post-norm feature output requires one auxiliary slot and no duplicate teacher")
        self.final_hidden_as_aux = final_hidden_as_aux
        self.store_last_hidden_states = bool(store_last_hidden_states)
        self.source_engine = source_engine
        self.mtp_feature_producer = mtp_feature_producer
        self._input_ids = None
        self.context_parallel_size = int(context_parallel_size)
        self.context_shard_alignment = int(context_shard_alignment)
        self._layer_columns = {
            layer: column for column, layer in enumerate(self.capture_layers)
        }
        self._reservations: dict[str, tuple] = {}
        self._slices: dict[str, tuple[slice, ...]] = {}
        self._captured_layers: set[int] = set()
        self._final_written = False
        self._begun = False
        self._published = False
        self._finished_requests: set[str] = set()

    def begin(
        self,
        *,
        input_ids: torch.Tensor,
        request_ids: Sequence[str],
        sequence_lengths: Sequence[int],
        prefix_lengths: Sequence[int],
    ) -> None:
        if self._begun:
            raise RuntimeError("Target prefill direct sink was started more than once")
        if list(request_ids) != [request.request_id for request in self.requests]:
            raise ValueError("Target prefill request order changed before model execution")
        expected_lengths = [request.sequence_length for request in self.requests]
        if list(sequence_lengths) != expected_lengths:
            raise ValueError(
                "Target prefill sequence lengths changed before model execution: "
                f"expected={expected_lengths}, actual={list(sequence_lengths)}"
            )
        if any(int(length) != 0 for length in prefix_lengths):
            raise ValueError("Target prefill direct output requires prefix-free requests")
        if input_ids.dim() != 1 or input_ids.numel() != sum(expected_lengths):
            raise ValueError(
                "Target prefill input_ids must be the flat, complete prefill batch"
            )

        if self.mtp_feature_producer is not None:
            self._input_ids = input_ids
        offset = 0
        try:
            for request in self.requests:
                end = offset + request.sequence_length
                key = f"{request.data_id}_{uuid.uuid4().hex[:8]}"
                reservations = []
                slices = []
                self._reservations[request.request_id] = ()
                for cp_rank in range(self.context_parallel_size):
                    from aspec.inference.context import context_shard_bounds

                    local_start, local_end = context_shard_bounds(
                        request.sequence_length,
                        cp_rank,
                        self.context_parallel_size,
                        alignment=self.context_shard_alignment,
                    )
                    shard_key = (
                        key
                        if self.context_parallel_size == 1
                        else f"{key}_cp{cp_rank}"
                    )
                    reservation = self.store.reserve_p2p_output(
                        key=shard_key,
                        sequence_length=request.sequence_length,
                        feature_sequence_length=local_end - local_start,
                        hidden_size=self.hidden_size,
                        num_aux_layers=len(self.capture_layers),
                        store_last_hidden_states=self.store_last_hidden_states,
                        data_id=request.data_id,
                        request_id=request.request_id,
                        source_engine=self.source_engine,
                        **({"store_mtp_hidden_states": True} if self.mtp_feature_producer is not None else {}),
                    )
                    reservations.append(reservation)
                    # Register ownership before a copy or later CP reservation
                    # can fail, so abort sees every acquired slot.
                    self._reservations[request.request_id] = tuple(reservations)
                    reservation.tensors["input_ids"].copy_(input_ids[offset:end])
                    slices.append(slice(offset + local_start, offset + local_end))
                self._reservations[request.request_id] = tuple(reservations)
                self._slices[request.request_id] = tuple(slices)
                offset = end
        except BaseException:
            self.abort()
            raise
        self._begun = True

    def capture(
        self,
        layer_index: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> None:
        self._require_active()
        if layer_index not in self._layer_columns:
            raise ValueError(f"unexpected Target prefill capture layer {layer_index}")
        if layer_index in self._captured_layers:
            raise RuntimeError(f"Target prefill layer {layer_index} was captured twice")
        self._validate_hidden_batch(hidden_states)
        if residual is not None and residual.shape != hidden_states.shape:
            raise ValueError("Target prefill residual shape does not match hidden states")

        if self.final_hidden_as_aux:
            self._captured_layers.add(layer_index)
            return
        column = self._layer_columns[layer_index]
        start = column * self.hidden_size
        end = start + self.hidden_size
        for request in self.requests:
            for reservation, part in zip(
                self._reservations[request.request_id],
                self._slices[request.request_id],
            ):
                storage = reservation.tensors["hidden_states"]
                target = storage[column] if storage.ndim == 3 else storage[:, start:end]
                if residual is None:
                    target.copy_(hidden_states[part])
                else:
                    torch.add(hidden_states[part], residual[part], out=target)
        self._captured_layers.add(layer_index)

    def finish(self, hidden_states: torch.Tensor) -> None:
        self._require_active()
        self._validate_hidden_batch(hidden_states)
        if not self.final_hidden_as_aux and self._captured_layers != set(self.capture_layers):
            missing = sorted(set(self.capture_layers) - self._captured_layers)
            raise RuntimeError(f"Target prefill did not capture layers {missing}")
        if self._final_written:
            raise RuntimeError("Target prefill final hidden states were written twice")
        if self.final_hidden_as_aux:
            for request in self.requests:
                for reservation, part in zip(self._reservations[request.request_id],
                                              self._slices[request.request_id], strict=True):
                    storage = reservation.tensors["hidden_states"]
                    (storage[0] if storage.ndim == 3 else storage).copy_(hidden_states[part])
        if self.store_last_hidden_states:
            for request in self.requests:
                for reservation, part in zip(
                    self._reservations[request.request_id],
                    self._slices[request.request_id],
                ):
                    reservation.tensors["last_hidden_states"].copy_(
                        hidden_states[part]
                    )
        if self.mtp_feature_producer is not None:
            offset = 0
            for request in self.requests:
                end = offset + request.sequence_length
                features = self.mtp_feature_producer(self._input_ids[offset:end], hidden_states[offset:end])
                for reservation, part in zip(self._reservations[request.request_id],
                                              self._slices[request.request_id], strict=True):
                    reservation.tensors["mtp_hidden_states"].copy_(features[part.start - offset:part.stop - offset])
                offset = end
            self._input_ids = None
        self._final_written = True

    def commit_request(self, request_id: str) -> str:
        self._require_complete()
        try:
            reservations = self._reservations[request_id]
        except KeyError as exc:
            raise KeyError(f"unknown Target prefill request {request_id!r}") from exc
        keys = tuple(reservation.commit().to_key() for reservation in reservations)
        if self.context_parallel_size == 1:
            return keys[0]
        from aspec.transfer.mooncake.p2p import pack_direct_cp_p2p_keys

        return pack_direct_cp_p2p_keys(keys)

    def payload_bytes(self, request_id: str) -> int:
        try:
            reservations = self._reservations[request_id]
        except KeyError as exc:
            raise KeyError(f"unknown Target prefill request {request_id!r}") from exc
        return sum(int(item.descriptor.payload_bytes) for item in reservations)

    def has_request(self, request_id: str) -> bool:
        return request_id in self._reservations

    def publish(self) -> None:
        self.validate_publishable()
        for request in self.requests:
            self.finalize_request(request.request_id, emitted=True)

    def finalize_request(self, request_id: str, *, emitted: bool) -> None:
        """Transfer only emitted output; discard canceled/failed response slots."""
        self._require_complete()
        if request_id in self._finished_requests:
            raise RuntimeError(f"Target prefill request {request_id!r} finalized twice")
        reservations = self._reservations[request_id]
        if any(reservation.state != "committed" for reservation in reservations):
            raise RuntimeError("Target prefill response was finalized before commit")
        for reservation in reservations:
            if emitted:
                reservation.publish()
            else:
                reservation.discard()
        self._finished_requests.add(request_id)
        self._published = len(self._finished_requests) == len(self.requests)

    def validate_publishable(self) -> None:
        """Check the whole batch before its descriptors leave the scheduler."""

        self._require_complete()
        for reservations in self._reservations.values():
            if any(reservation.state != "committed" for reservation in reservations):
                raise RuntimeError(
                    "Target prefill batch was published before every request committed"
                )

    def abort(self) -> None:
        if self._published:
            return
        errors = []
        for request_id, reservations in reversed(tuple(self._reservations.items())):
            if request_id in self._finished_requests:
                continue
            for reservation in reversed(reservations):
                try:
                    reservation.discard()
                except BaseException as exc:
                    errors.append(exc)
        self._reservations.clear()
        self._slices.clear()
        self._input_ids = None
        if errors:
            raise RuntimeError("failed to release a Target prefill P2P reservation") from errors[0]

    def _require_active(self) -> None:
        if not self._begun or self._published:
            raise RuntimeError("Target prefill direct sink is not active")

    def _require_complete(self) -> None:
        self._require_active()
        if not self._final_written:
            raise RuntimeError("Target prefill final hidden states are incomplete")

    def _validate_hidden_batch(self, hidden_states: torch.Tensor) -> None:
        expected = (sum(request.sequence_length for request in self.requests), self.hidden_size)
        if tuple(hidden_states.shape) != expected:
            raise ValueError(
                f"Target prefill hidden shape mismatch: expected={expected}, "
                f"actual={tuple(hidden_states.shape)}"
            )
