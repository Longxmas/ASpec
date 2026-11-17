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

"""Official SGLang request hooks over the retained ASpec feature data plane.

Request metadata and prefill-only completion use upstream APIs. Direct sink,
feature ownership and the consumer format remain the existing ASpec path.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import os
import time
import uuid
from functools import wraps

import torch

from aspec.config.mooncake_config import MooncakeConfig
from aspec.inference.target_prefill import TargetPrefillDirectSink, TargetPrefillRequest
from aspec.transfer.mooncake import EagleMooncakeStore


class StreamingLayerAccumulator:
    """Native layer-output callbacks write straight into existing sink columns."""

    def __init__(self, sink, capture_order):
        self.sink = sink
        self.capture_order = tuple(capture_order)
        self.index = 0

    def capture(self, hidden, *, owned=False):
        if self.index >= len(self.capture_order):
            raise RuntimeError("Target returned an extra captured layer")
        self.sink.capture(self.capture_order[self.index], hidden, None)
        self.index += 1

    def __len__(self):
        return self.index


class _StoreOutputBatch:
    """Own exact Store objects until response handoff or confirmed deletion."""

    def __init__(self, store):
        self.store = store
        self.requests = {}
        self.ready = False

    def validate_publishable(self):
        if not self.ready:
            raise RuntimeError("Store output batch was staged before all puts completed")

    def finalize_request(self, request_id, *, emitted):
        objects = self.requests[request_id]
        if not emitted:
            # Drains every async write before removing these objects, and
            # raises on unconfirmed deletion so ownership remains tracked.
            self.store.discard_unpublished_objects(objects)
        del self.requests[request_id]

    def abort(self):
        errors = []
        for rid in tuple(self.requests):
            try:
                self.finalize_request(rid, emitted=False)
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(f"Store output cleanup failed: {errors}") from errors[0]


class TrainingFeaturePublisher:
    """Publish GPU features, never a Python/host hidden-state response."""

    always_call = True

    def __init__(self, config):
        self.layers = list(config["layers"])
        self.hidden_size = int(config["hidden_size"])
        self.cp_size = int(config["context_parallel_size"])
        self.cp_alignment = int(config.get("context_shard_alignment", 1))
        self.max_sequence_length = int(config["max_sequence_length"])
        self.store_last = bool(config["store_last_hidden_states"])
        self.direct = bool(config["direct_prefill"])
        self.final_hidden_as_aux = bool(config.get("final_hidden_as_aux", False))
        self.attention_backend = config["attention_backend"]
        self.store = None
        self.sink = None
        self.backbone = None
        self.capture_layers = ()
        self.is_publisher = True
        self.forward_started_ns = None
        self.pending_outputs = {}
        self.mtp_policy = config.get("mtp")
        self.target_model_path = config.get("target_model_path")
        self.mtp = None
        self.target_tensor_parallel_size = config.get("target_tensor_parallel_size", 1)

    def configure_model(self, model):
        from sglang.srt.runtime_context import get_parallel

        self.is_publisher = get_parallel().attn_tp_rank == 0
        if getattr(model, "aspec_feature_publisher", None) is not None:
            raise RuntimeError("ASpec training hooks were registered twice")
        if not hasattr(model, "logits_processor"):
            raise RuntimeError("Target does not expose its verifier logits processor")
        # Each native setter owns its coordinate conversion. In particular,
        # Qwen captures before k+1 while Ling captures after k: do not apply
        # an additional offset based on the method name.
        self.backbone = model.model
        if self.final_hidden_as_aux:
            if len(self.layers) != 1 or self.store_last:
                raise ValueError("Native MTP requires one post-norm feature, without a duplicate teacher")
            # The verifier LM-head boundary already supplies the final normalized
            # state. Do not ask an auxiliary setter to translate a last-layer ID
            # into the out-of-range layer after it.
            self.capture_layers = tuple(self.layers)
        else:
            if hasattr(model, "set_dspark_layers_to_capture"):
                model.set_dspark_layers_to_capture(self.layers)
            elif hasattr(model, "set_dflash_layers_to_capture"):
                model.set_dflash_layers_to_capture(self.layers)
            elif hasattr(model, "set_eagle3_layers_to_capture"):
                model.set_eagle3_layers_to_capture(self.layers)
            else:
                raise RuntimeError("Target lacks a supported auxiliary feature capture API")
            self.backbone = model.model
            actual = tuple(self.backbone.layers_to_capture)
            if len(actual) != len(self.layers):
                raise RuntimeError("Target capture configuration dropped requested layers")
            self.capture_layers = actual
        if self.mtp_policy is not None and self.is_publisher:
            from aspec.inference.mtp_features import FrozenTargetMTP

            embedding = getattr(self.backbone, "embed_tokens", None)
            weight = getattr(embedding, "weight", None)
            parameter = next(model.parameters())
            self.mtp = FrozenTargetMTP.from_target(
                self.mtp_policy, self.target_model_path, weight,
                device=parameter.device, dtype=parameter.dtype,
                allow_replicated_embedding=self.target_tensor_parallel_size > 1,
            )
        model.aspec_feature_publisher = self
        original = model.logits_processor.forward

        def forward(input_ids, hidden_states, lm_head, batch, aux_hidden_states=None,
                    hidden_states_before_norm=None):
            params = getattr(batch, "request_custom_params", None)
            identified = params and any(
                item and "aspec_data_id" in item for item in params
            )
            if not identified:
                if self.direct:
                    raise RuntimeError("Direct training Target received ordinary generation")
                return original(input_ids, hidden_states, lm_head, batch,
                                aux_hidden_states, hidden_states_before_norm)
            try:
                if self.final_hidden_as_aux and isinstance(hidden_states, tuple):
                    # Qwen's outer multimodal wrapper only unpacks its model
                    # return when auxiliary capture is enabled. Native MTP has
                    # no such capture; preserve the direct sink at our boundary.
                    if len(hidden_states) != 2 or hidden_states[1] is not self.sink:
                        raise RuntimeError("Native MTP received an unexpected verifier output tuple")
                    hidden_states, aux_hidden_states = hidden_states
                # The native LM-head boundary restores sequence-sharded rows.
                # Every TP rank must participate even though only rank zero
                # owns the feature publication.
                from sglang.srt.layers.layernorm_sp import (
                    maybe_exit_gather,
                    runs_sp,
                    sp_exit_gather,
                )

                if runs_sp(batch.forward_mode):
                    # Native Qwen3 snapshots are on the same SP rows as the
                    # pre-LM-head teacher. Restore every selected layer too.
                    if isinstance(aux_hidden_states, torch.Tensor):
                        aux_hidden_states = sp_exit_gather(aux_hidden_states, input_ids.shape[0])
                    elif aux_hidden_states is not None:
                        aux_hidden_states = [sp_exit_gather(item, input_ids.shape[0])
                                             for item in aux_hidden_states]
                hidden_states, _ = maybe_exit_gather(
                    hidden_states=hidden_states,
                    hidden_states_before_norm=hidden_states_before_norm,
                    input_ids=input_ids, forward_mode=batch.forward_mode,
                )
                if not self.is_publisher:
                    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

                    self.validate_batch(batch)
                    return LogitsProcessorOutput(next_token_logits=None)
                keys = self.publish(input_ids, hidden_states, batch, aux_hidden_states)
                from sglang.srt.layers.logits_processor import LogitsProcessorOutput

                return LogitsProcessorOutput(
                    next_token_logits=None,
                    # Scheduler appends one element per request/forward, then
                    # tokenizer metadata extends the per-request list.
                    customized_info={"spec_training_mooncake_store_keys": keys},
                )
            except BaseException:
                self.abort()
                raise

        model.logits_processor.forward = forward
        original_forward = model.forward

        @wraps(original_forward)
        def training_forward(*args, **kwargs):
            # Upstream's eager runner calls model.forward directly, bypassing
            # nn.Module pre/post hooks. Cover that path and ordinary __call__.
            try:
                self.begin(model, args, kwargs)
                return original_forward(*args, **kwargs)
            finally:
                self.__call__(model, args, None)

        model.forward = training_forward
        if self.direct:
            family = getattr(self.backbone.config, "model_type", None)
            if family not in {"qwen3", "qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text"}:
                raise RuntimeError("Target does not implement the migrated direct-prefill path")
            model.lm_head = None
            if hasattr(model, "visual"):
                model.visual = None
            torch.cuda.empty_cache()

    def _store(self, device):
        if self.store is None:
            self.store = EagleMooncakeStore(MooncakeConfig.from_env())
            self.store.setup(device=device, p2p_role="source")
            from aspec.utils.timeline import get_timeline_from_env

            if self.store._p2p_transfer is not None:
                self.store._p2p_transfer.timeline = get_timeline_from_env(
                    "sglang", os.getenv("ASPEC_ENGINE_RANK", "0")
                )
        return self.store

    def validate_batch(self, batch):
        if not batch.forward_mode.is_extend() or not batch.is_prefill_only:
            raise RuntimeError("Feature publication requires complete prefill-only requests")
        lengths = list(batch.extend_seq_lens_cpu)
        if any(batch.extend_prefix_lens_cpu):
            raise RuntimeError("Training features require prefix-free, unchunked input")
        identities = batch.request_custom_params
        if not identities or len(identities) != len(lengths):
            raise RuntimeError("Target feature requests lack work identity")
        if len(batch.rids) != len(lengths):
            raise RuntimeError("Target request identity count does not match rows")
        for rid, identity, length in zip(batch.rids, identities, lengths, strict=True):
            if not identity or not identity.get("aspec_data_id"):
                raise RuntimeError("Target mixed ordinary and training requests")
            if identity.get("aspec_request_id") != rid:
                raise RuntimeError("Target request identity changed before feature production")
            if length < self.cp_size or length > self.max_sequence_length:
                raise RuntimeError("Target feature length is outside the admitted workload")
        return lengths, identities

    def begin(self, model, args, kwargs):
        batch = kwargs.get("forward_batch", args[2] if len(args) > 2 else None)
        if batch.forward_mode.is_idle():
            return
        self.forward_started_ns = time.monotonic_ns()
        if not self.direct:
            return
        lengths, identities = self.validate_batch(batch)
        if self.sink is not None:
            raise RuntimeError("Target started another forward before closing its sink")
        ids = kwargs.get("input_ids", args[0] if args else None)
        self.sink = TargetPrefillDirectSink(
            store=self._store(ids.device),
            requests=[TargetPrefillRequest(rid, identity["aspec_data_id"], length)
                      for rid, identity, length in zip(batch.rids, identities, lengths, strict=True)],
            capture_layers=self.capture_layers,
            hidden_size=self.hidden_size,
            store_last_hidden_states=self.store_last,
            final_hidden_as_aux=self.final_hidden_as_aux,
            source_engine=os.getenv("ASPEC_ENGINE_RANK", ""),
            context_parallel_size=self.cp_size,
            context_shard_alignment=self.cp_alignment,
            **({"mtp_feature_producer": self.mtp} if self.mtp is not None else {}),
        )
        self.sink.aux_accumulator = StreamingLayerAccumulator(self.sink, sorted(self.capture_layers))
        self.backbone.target_prefill_direct_sink = self.sink
        self.sink.begin(input_ids=ids, request_ids=batch.rids,
                        sequence_lengths=lengths, prefix_lengths=batch.extend_prefix_lens_cpu)

    def publish(self, ids, teacher, batch, aux):
        lengths, identities = self.validate_batch(batch)
        total = sum(lengths)
        if tuple(ids.shape) != (total,) or tuple(teacher.shape) != (total, self.hidden_size):
            raise RuntimeError("Target input/teacher rows do not match the complete batch")
        if self.direct:
            if aux is not self.sink:
                raise RuntimeError("Target did not forward its registered direct sink")
            keys = [self.sink.commit_request(rid) for rid in batch.rids]
            self.stage_outputs(batch.rids, keys)
            return keys
        if self.final_hidden_as_aux:
            hidden = teacher
        elif isinstance(aux, torch.Tensor):
            if tuple(aux.shape) != (total, self.hidden_size * len(self.layers)):
                raise RuntimeError("Target returned an incomplete packed auxiliary capture")
            hidden = aux
        elif isinstance(aux, (tuple, list)) and len(aux) == len(self.layers):
            if any(tuple(item.shape) != tuple(teacher.shape) for item in aux):
                raise RuntimeError("Target auxiliary capture has inconsistent rows")
            hidden = torch.cat(aux, dim=-1)
        else:
            raise RuntimeError("Target returned an incomplete auxiliary capture")
        # Native loops visit layers in physical order, even for a reordered request.
        order = sorted(self.capture_layers)
        columns = [order.index(layer) for layer in self.capture_layers]
        if columns != list(range(len(columns))):
            hidden = hidden.reshape(total, len(columns), self.hidden_size)[:, columns].reshape(total, -1)
        store = self._store(teacher.device)
        if store.config.data_plane == "transfer_engine" and self.attention_backend != "usp":
            # Keep the whole batch unpublished until every reservation commits.
            # Reuse the direct sink lifecycle, without changing buffered model
            # capture or the consumer's sharded/unsharded layout.
            cp_size = self.cp_size if os.getenv("ASPEC_DIRECT_CP_SHARDED_MOONCAKE") == "1" else 1
            self.sink = TargetPrefillDirectSink(
                store=store,
                requests=[TargetPrefillRequest(rid, identity["aspec_data_id"], length)
                          for rid, identity, length in zip(batch.rids, identities, lengths, strict=True)],
                capture_layers=self.capture_layers, hidden_size=self.hidden_size,
                store_last_hidden_states=self.store_last,
                final_hidden_as_aux=self.final_hidden_as_aux,
                source_engine=os.getenv("ASPEC_ENGINE_RANK", ""),
                context_parallel_size=cp_size,
                context_shard_alignment=self.cp_alignment,
                **({"mtp_feature_producer": self.mtp} if self.mtp is not None else {}),
            )
            try:
                self.sink.begin(input_ids=ids, request_ids=batch.rids,
                                sequence_lengths=lengths, prefix_lengths=batch.extend_prefix_lens_cpu)
                for column, layer in enumerate(self.capture_layers):
                    self.sink.capture(layer, hidden[:, column * self.hidden_size:(column + 1) * self.hidden_size], None)
                self.sink.finish(teacher)
                keys = [self.sink.commit_request(rid) for rid in batch.rids]
                self.stage_outputs(batch.rids, keys)
                return keys
            except BaseException:
                self.abort()
                raise
        owner = _StoreOutputBatch(store)
        self.sink = owner
        keys = []
        offset = 0
        try:
            for rid, identity, length in zip(batch.rids, identities, lengths, strict=True):
                end = offset + length
                payload = dict(
                    key=f"{identity['aspec_data_id']}_{uuid.uuid4().hex}",
                    hidden_states=hidden[offset:end], input_ids=ids[offset:end],
                    last_hidden_states=teacher[offset:end] if self.store_last else None,
                    data_id=identity["aspec_data_id"], request_id=rid,
                    source_engine=os.getenv("ASPEC_ENGINE_RANK", ""),
                    **({"mtp_hidden_states": self.mtp(ids[offset:end], teacher[offset:end])}
                       if self.mtp is not None else {}),
                )
                if os.getenv("ASPEC_DIRECT_CP_SHARDED_MOONCAKE") == "1" and self.cp_size > 1:
                    bases = [f"{payload['key']}_cp{rank}" for rank in range(self.cp_size)]
                elif self.attention_backend == "usp":
                    bases = [f"{payload['key']}_usp{rank}"
                             for rank in range(int(os.environ["ASPEC_USP_SP_SIZE"]))]
                else:
                    bases = [payload["key"]]
                suffixes = ("hs", "ids", "lhs") if self.store_last else ("hs", "ids")
                if self.mtp is not None:
                    suffixes += ("mhs",)
                if rid in owner.requests:
                    raise RuntimeError("Target Store batch contains duplicate request IDs")
                owner.requests[rid] = tuple(f"{base}_{suffix}" for base in bases for suffix in suffixes)
                if os.getenv("ASPEC_DIRECT_CP_SHARDED_MOONCAKE") == "1":
                    result = store.put_direct_cp_shards(
                        **payload,
                        cp_size=self.cp_size,
                        max_seq_length=self.max_sequence_length,
                        context_shard_alignment=self.cp_alignment,
                    )
                elif self.attention_backend == "usp":
                    for name in ("data_id", "request_id", "source_engine"):
                        payload.pop(name)
                    result = store.put_usp_shards(
                        **payload, target=None,
                        sp_size=int(os.environ["ASPEC_USP_SP_SIZE"]),
                        sp_ring_size=int(os.environ["ASPEC_USP_RING_SIZE"]),
                        ttt_length=int(os.environ["ASPEC_USP_TTT_LENGTH"]),
                        max_seq_length=self.max_sequence_length,
                    )
                    result.setdefault("mooncake_key", payload["key"])
                else:
                    result = store.put(**payload)
                keys.append(result["mooncake_key"])
                offset = end
            store.flush()
            owner.ready = True
            self.stage_outputs(batch.rids, keys)
            return keys
        except BaseException:
            self.abort()
            raise

    def stage_outputs(self, request_ids, keys):
        self.sink.validate_publishable()
        if any(rid in self.pending_outputs for rid in request_ids):
            raise RuntimeError("Target staged the same active request twice")
        for rid, key in zip(request_ids, keys, strict=True):
            self.pending_outputs[rid] = (self.sink, key)
        self.sink = None
        if self.backbone is not None:
            self.backbone.target_prefill_direct_sink = None

    def resolve_outputs(self, request_ids, emitted_keys):
        emitted_keys = set(emitted_keys)
        # Handoff is irreversible. Finalize emitted siblings before attempting
        # deletion: the streamer's failure callback must not reclaim their data.
        errors = []
        ordered = sorted(request_ids, key=lambda rid: (
            self.pending_outputs.get(rid, (None, None))[1] not in emitted_keys
        ))
        for rid in ordered:
            pending = self.pending_outputs.get(rid)
            if pending is None:
                continue
            sink, key = pending
            try:
                sink.finalize_request(rid, emitted=key in emitted_keys)
            except Exception as exc:
                errors.append(exc)
            else:
                del self.pending_outputs[rid]
        if errors:
            raise RuntimeError(f"Target output cleanup failed: {errors}") from errors[0]

    def abort(self):
        if self.sink is not None:
            self.sink.abort()
            self.sink = None
        if self.backbone is not None:
            self.backbone.target_prefill_direct_sink = None

    def __call__(self, model, args, output):
        # always_call also releases unpublished reservations after a layer error.
        self.abort()
        if self.forward_started_ns is not None:
            from aspec.utils.timeline import get_timeline_from_env

            get_timeline_from_env("sglang", os.getenv("ASPEC_ENGINE_RANK", "0")).complete(
                "target_model_forward", start_ns=self.forward_started_ns,
            )
            self.forward_started_ns = None
        return output

    def drain(self):
        # flush_cache reaches here only when the native scheduler is idle.
        # Any staged response still lacking an IPC handoff is canceled work.
        self.abort()
        self.resolve_outputs(tuple(self.pending_outputs), ())
        if self.store is None:
            return True
        self.store.flush()
        return self.store.wait_for_p2p_source_releases(
            timeout=max(float(self.store.config.get_retry_max_wait_seconds), 5.0)
        )

    def transport_status_message(self):
        status = self.store.get_p2p_status() if self.store is not None else None
        if status is not None:
            status["pending_response_outputs"] = len(self.pending_outputs)
            if self.mtp is not None:
                status["mtp_features"] = self.mtp.status()
        return json.dumps(status, sort_keys=True, separators=(",", ":")) if status is not None else ""


def create_feature_publisher(config):
    return TrainingFeaturePublisher(config)


class TrainingEngine:
    """Translate retained calls into upstream request metadata at the boundary."""

    def __init__(self, engine):
        self.engine = engine
        self._closed = False

    def __getattr__(self, name):
        return getattr(self.engine, name)

    @staticmethod
    def prepare(kwargs):
        kwargs = dict(kwargs)
        data_ids = kwargs.pop("spec_training_data_id")
        kwargs.pop("packed_loss_mask", None)  # The consumer retains the exact mask.
        sampling = kwargs.pop("sampling_params")
        if sampling.get("max_new_tokens") != 0:
            raise ValueError("Migrated feature API currently admits prefill-only requests")
        request_ids = kwargs.get("rid")
        if request_ids is None:
            request_ids = [uuid.uuid4().hex for _ in data_ids]
            kwargs["rid"] = request_ids
        if len(request_ids) != len(data_ids):
            raise ValueError("Training requires a request ID for every work item")
        kwargs["sampling_params"] = [
            {**sampling, "custom_params": {"aspec_data_id": data_id, "aspec_request_id": rid}}
            for data_id, rid in zip(data_ids, request_ids, strict=True)
        ]
        return kwargs

    async def async_generate(self, **kwargs):
        return await self.engine.async_generate(**self.prepare(kwargs))

    def generate(self, **kwargs):
        return self.engine.generate(**self.prepare(kwargs))

    async def _close_background_tasks(self):
        tasks = [task for task in self.engine.tokenizer_manager.asyncio_tasks
                 if not task.done() and task is not asyncio.current_task()]
        current_loop = asyncio.get_running_loop()
        if any(task.get_loop() is not current_loop for task in tasks):
            raise RuntimeError("Close the Target on the event loop owning its tokenizer tasks")
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def async_shutdown(self):
        if self._closed:
            return
        try:
            await self._close_background_tasks()
        finally:
            self.engine.shutdown()
            self._closed = True
            atexit.unregister(self.engine.shutdown)

    def shutdown(self):
        self.engine.loop.run_until_complete(self.async_shutdown())


def training_engine_kwargs(kwargs, args):
    from sglang.srt.model_executor import hook_manager

    if not getattr(hook_manager, "SUPPORTS_MODEL_HOOK_SETUP", False):
        raise RuntimeError("Apply the SGLang v0.5.21 training patch with aspec patch-sglang")
    if getattr(args, "train_with_decode", False):
        raise ValueError("training.train_with_decode is outside the current ASpec path")
    result = dict(kwargs)
    # Admission already chose the exact document; the backend may not truncate it.
    result["allow_auto_truncate"] = False
    graph_config = result.get("cuda_graph_config") or {}
    if isinstance(graph_config, str):
        graph_config = json.loads(graph_config)
    graph_config = {phase: dict(settings) for phase, settings in graph_config.items()}
    for phase in ("prefill", "decode"):
        requested = result.get(f"cuda_graph_backend_{phase}")
        settings = graph_config.setdefault(phase, {})
        if requested not in (None, "disabled") or settings.get("backend") not in (None, "disabled"):
            raise ValueError("Training feature hooks require eager execution, not CUDA graph replay")
        settings["backend"] = "disabled"
    result["cuda_graph_config"] = graph_config
    for name in ("enable_aux_hidden_states", "aux_hidden_state_layer_ids",
                 "enable_spec_training_mooncake", "spec_training_store_last_hidden_states"):
        result.pop(name, None)
    hooks = list(result.get("forward_hooks") or [])
    hooks.append({
        "name": "aspec_training_features", "target_modules": [""],
        "hook_factory": "aspec.inference.sglang_v0521:create_feature_publisher",
        "config": {
            "layers": list(kwargs["aux_hidden_state_layer_ids"]),
            "hidden_size": getattr(args, "mooncake_hidden_dim", 4096),
            "context_parallel_size": args.draft_context_parallel_size,
            "context_shard_alignment": getattr(
                args, "draft_context_shard_alignment", 1
            ),
            "max_sequence_length": args.max_seq_length,
            "store_last_hidden_states": args.store_last_hidden_states,
            "direct_prefill": args.target_prefill_memory_optimized,
            "final_hidden_as_aux": getattr(args, "draft_algorithm", None) == "mtp",
            "attention_backend": args.attention_backend,
            **({"mtp": args.aspec_feature_contract["mtp"], "target_model_path": args.target_model_path}
               if (getattr(args, "aspec_feature_contract", None) or {}).get("mtp") is not None else {}),
            "target_tensor_parallel_size": getattr(args, "sglang_tp_size", 1),
        },
    })
    result["forward_hooks"] = hooks
    return result
