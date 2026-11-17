# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Experimental SM90 Draft-only FP8 LM head for native SGLang DSpark decode.

Reuses DSpark's native training-time rowwise FP8 projection and SGLang's native quant_method
dispatch. The Target module and its canonical BF16 parameter are never changed.
This is an explicit per-process Qwen3/TP1 benchmark adapter, not a serving flag.
No SGLang implementation is copied; source hashes bind the adapter to the tested
interfaces before installing it. Runtime FP8 caches are not checkpoint weights.
"""
from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from aspec.models.ops.dspark_fp8_lm_head import _project, quantize_activation


def _version(weight):
    try:
        return weight._version
    except RuntimeError as error:
        if "Inference tensors do not track version counter" not in str(error):
            raise
        return None


class DraftRowwiseFP8Method:
    """Inference-only E4M3 GEMM; forward weight scale axis matches training B."""

    @torch.no_grad()
    def __init__(self, weight):
        if weight.ndim != 2 or not weight.is_cuda or weight.dtype != torch.bfloat16:
            raise ValueError("Draft FP8 inference requires a CUDA BF16 [vocab, hidden] weight")
        if torch.cuda.get_device_capability(weight.device) != (9, 0):
            raise ValueError("This inference prototype requires SM90")
        if min(weight.shape) == 0 or any(size % 16 for size in weight.shape):
            raise ValueError("Head dimensions must be positive multiples of 16")
        if not bool(torch.isfinite(weight).all()):
            raise ValueError("Head weight contains nonfinite values")
        self.weight = weight
        self.weight_version = _version(weight)
        self.weight_pointer = weight.data_ptr()
        bits, scales = quantize_activation(weight.detach())
        self.weight_t = bits.t()
        self.scale_t = scales.t()
        self.capture_shapes = set()
        self.eager_shapes = set()

    @property
    def cache_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.weight_t, self.scale_t))

    def apply(self, layer, hidden, bias=None):
        if bias is not None or layer.weight is not self.weight:
            raise ValueError("FP8 head view received a different weight or bias")
        if self.weight.dtype != torch.bfloat16 or self.weight.data_ptr() != self.weight_pointer:
            raise ValueError("Canonical Target head changed")
        if self.weight_version is not None and _version(self.weight) != self.weight_version:
            raise ValueError("Canonical head changed; rebuild before graph capture")
        if hidden.ndim < 2 or hidden.shape[-1] != self.weight.shape[1]:
            raise ValueError("Draft hidden dimension does not match the head")
        if hidden.dtype != torch.bfloat16 or hidden.device != self.weight.device:
            raise ValueError("Draft hidden must remain BF16 on the head device")
        if torch.is_grad_enabled() and (hidden.requires_grad or self.weight.requires_grad):
            raise ValueError("This projection is for inference, not training")
        matrix = hidden.reshape(-1, hidden.shape[-1])
        shapes = self.capture_shapes if torch.cuda.is_current_stream_capturing() else self.eager_shapes
        shapes.add(tuple(matrix.shape))
        if matrix.shape[0] == 0:
            return hidden.new_empty((*hidden.shape[:-1], self.weight.shape[0]))
        return _project(matrix, self.weight_t, self.scale_t).view(
            *hidden.shape[:-1], self.weight.shape[0]
        )


class DraftFP8HeadView(nn.Module):
    """Separate quant_method owner with a read-only alias of the Target weight."""

    def __init__(self, target_head):
        super().__init__()
        self.weight = target_head.weight
        self.org_vocab_size = int(target_head.org_vocab_size)
        if self.weight.shape[0] < self.org_vocab_size:
            raise ValueError("This experiment supports TP1 only")
        self.quant_method = DraftRowwiseFP8Method(self.weight)


def install_sglang_draft_head(*, precision, evidence_dir, source_hashes):
    """Install in every scheduler process before shared-head attachment/capture."""
    if precision not in {"bf16", "fp8"}:
        raise ValueError("Unknown Draft head inference precision")
    if importlib.metadata.version("sglang") != "0.5.21":
        raise ValueError("The inference adapter is validated for SGLang 0.5.21")
    from sglang.srt.layers import logits_processor
    from sglang.srt.models import dspark
    from sglang.srt.speculative.dspark_components import dspark_worker_v2

    modules = {"dspark.py": dspark, "logits_processor.py": logits_processor,
               "dspark_worker_v2.py": dspark_worker_v2}
    actual = {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
              for name, module in modules.items()}
    if actual != source_hashes:
        raise ValueError("SGLang serving sources differ from the reviewed binding")
    if getattr(dspark, "_aspec_draft_head_precision", None) is not None:
        raise ValueError("Draft head adapter is already installed in this process")
    dspark._aspec_draft_head_precision = precision
    destination = Path(evidence_dir)
    original_attach = dspark.DSparkDraftMixin.attach_shared_modules
    original_graph = dspark_worker_v2.DSparkWorkerV2.init_cuda_graphs

    def write(stage, data):
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f"{stage}-{os.getpid()}.json").write_text(
            json.dumps(data, indent=2) + "\n")

    @functools.wraps(original_attach)
    def attach(self, *, embed_tokens, lm_head):
        if type(self).__name__ != "Qwen3DSparkModel":
            raise ValueError("Draft-only FP8 adapter is scoped to dense Qwen3 DSpark")
        if logits_processor.should_apply_lm_head_quant_method(lm_head, lm_head.quant_method):
            raise ValueError("Target head must use its native unquantized BF16 path")
        original_attach(self, embed_tokens=embed_tokens, lm_head=lm_head)
        if any(p.dtype != torch.bfloat16 for p in self.parameters() if p.is_floating_point()):
            raise ValueError("Draft parameters must remain BF16")
        before = (lm_head.weight.data_ptr(), _version(lm_head.weight), lm_head.quant_method)
        if precision == "fp8":
            self.lm_head = DraftFP8HeadView(lm_head)
        if before != (lm_head.weight.data_ptr(), _version(lm_head.weight), lm_head.quant_method):
            raise ValueError("Adapter modified the Target head")
        data = {
            "precision": precision, "model_class": type(self).__name__,
            "canonical_head_dtype": str(lm_head.weight.dtype),
            "draft_parameter_dtype": "torch.bfloat16",
            "shared_weight_pointer": lm_head.weight.data_ptr(),
            "target_quant_method": type(lm_head.quant_method).__name__,
            "target_head_unchanged": True,
            "draft_head_is_target_head": self.lm_head is lm_head,
            "cache_bytes": self.lm_head.quant_method.cache_bytes if precision == "fp8" else 0,
            "source_hashes": actual,
            "kernel": "torch._scaled_mm E4M3 rowwise power-of-two, fast_accum=False" if precision == "fp8" else "native BF16",
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        }
        self._aspec_head_audit = SimpleNamespace(target_head=lm_head, before=before, data=data)
        write("attached", data)

    @functools.wraps(original_graph)
    def graph(self, *args, **kwargs):
        result = original_graph(self, *args, **kwargs)
        audit = getattr(self.draft_model, "_aspec_head_audit", None)
        if audit is None:
            raise ValueError("Draft head adapter was not activated before graph capture")
        target_head = self.target_worker.model_runner.model.lm_head
        if target_head is not audit.target_head:
            raise ValueError("Target no longer owns the original BF16 head module")
        if audit.before != (target_head.weight.data_ptr(), _version(target_head.weight), target_head.quant_method):
            raise ValueError("Target head changed during graph capture")
        sampler = self._draft_sampler
        data = {**audit.data,
                "target_parameter_dtypes": sorted({str(p.dtype) for p in
                    self.target_worker.model_runner.model.parameters() if p.is_floating_point()}),
                "draft_graph_available": self.draft_model_runner.decode_cuda_graph_runner is not None,
                "folded_proposal": sampler is not None,
                "folded_sampling": sampler.folded_sampling if sampler is not None else None}
        if precision == "fp8":
            method = self.draft_model.lm_head.quant_method
            data.update(fp8_cache_dtype=str(method.weight_t.dtype),
                        capture_shapes=sorted(method.capture_shapes),
                        eager_shapes=sorted(method.eager_shapes))
        write("graph-ready", data)
        return result

    dspark.DSparkDraftMixin.attach_shared_modules = attach
    dspark_worker_v2.DSparkWorkerV2.init_cuda_graphs = graph
