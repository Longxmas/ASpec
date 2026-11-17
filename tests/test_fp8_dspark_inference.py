# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Draft FP8 dispatch must preserve the shared Target and CUDA graph results."""
import hashlib
import os
from pathlib import Path

import pytest
import torch

from aspec.inference.dspark_fp8_head import DraftFP8HeadView, install_sglang_draft_head
from aspec.models.ops.dspark_fp8_lm_head import quantize_rows

pytestmark = pytest.mark.skipif(
    os.environ.get("ASPEC_RUN_GPU_INTEGRATION") != "1" or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0),
    reason="requires an explicit SM90 inference check",
)


def head():
    value = torch.nn.Linear(128, 256, bias=False).to("cuda", dtype=torch.bfloat16)
    value.org_vocab_size = 256
    value.quant_method = None
    return value


@pytest.mark.parametrize("rows", [0, 1, 7, 14, 28, 56])
@torch.no_grad()
def test_native_dispatch_changes_only_the_draft_view(rows):
    from sglang.srt.models.dspark import project_through_lm_head

    torch.manual_seed(148)
    target = head()
    canonical = target.weight.clone()
    hidden = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
    target_logits = project_through_lm_head(hidden, target)
    draft = DraftFP8HeadView(target)
    actual = project_through_lm_head(hidden, draft)
    bits, scale = quantize_rows(canonical)
    method = draft.quant_method
    # The serving weight recipe must be identical to the training forward cache.
    assert torch.equal(method.weight_t.t().view(torch.uint8), bits.view(torch.uint8))
    assert torch.equal(method.scale_t.t(), scale)
    if rows:
        x, xs = quantize_rows(hidden)
        expected = (x.float() * xs) @ (bits.float() * scale).t()
        torch.testing.assert_close(actual, expected.bfloat16(), atol=1e-3, rtol=1e-2)
    assert actual.shape == (rows, 256) and actual.dtype == torch.bfloat16
    assert draft.weight is target.weight
    assert target.quant_method is None
    assert torch.equal(target.weight, canonical)
    assert torch.equal(project_through_lm_head(hidden, target), target_logits)
    assert method.cache_bytes == 256 * 128 + 256 * 4


@torch.no_grad()
def test_native_fp8_head_replays_in_cuda_graph_and_rejects_stale_weights():
    from sglang.srt.models.dspark import project_through_lm_head

    target = head()
    draft = DraftFP8HeadView(target)
    hidden = torch.randn(7, 128, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            project_through_lm_head(hidden, draft)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = project_through_lm_head(hidden, draft)
    for _ in range(3):
        hidden.copy_(torch.randn_like(hidden))
        graph.replay()
        expected = project_through_lm_head(hidden, draft)
        assert torch.equal(captured, expected)
    assert (7, 128) in draft.quant_method.capture_shapes
    target.weight.add_(1)
    with pytest.raises(ValueError, match="Canonical head changed"):
        project_through_lm_head(hidden, draft)


@torch.no_grad()
def test_unsupported_head_dtype_and_partition_fail_closed():
    target = head()
    target.org_vocab_size = 512
    with pytest.raises(ValueError, match="TP1"):
        DraftFP8HeadView(target)
    target.org_vocab_size = 256
    target.to(dtype=torch.float32)
    with pytest.raises(ValueError, match="BF16"):
        DraftFP8HeadView(target)


def test_process_installer_keeps_native_target_projection(monkeypatch, tmp_path):
    from sglang.srt.layers import logits_processor
    from sglang.srt.models import dspark
    from sglang.srt.speculative.dspark_components import dspark_worker_v2

    modules = {"dspark.py": dspark, "logits_processor.py": logits_processor,
               "dspark_worker_v2.py": dspark_worker_v2}
    binding = {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
               for name, module in modules.items()}
    projection = dspark.project_through_lm_head
    dispatch = logits_processor.should_apply_lm_head_quant_method
    attach = dspark.DSparkDraftMixin.attach_shared_modules
    graph = dspark_worker_v2.DSparkWorkerV2.init_cuda_graphs
    monkeypatch.setattr(dspark, "_aspec_draft_head_precision", None, raising=False)
    monkeypatch.setattr(dspark.DSparkDraftMixin, "attach_shared_modules", attach)
    monkeypatch.setattr(dspark_worker_v2.DSparkWorkerV2, "init_cuda_graphs", graph)
    with pytest.raises(ValueError, match="sources differ"):
        install_sglang_draft_head(precision="fp8", evidence_dir=tmp_path, source_hashes={})
    install_sglang_draft_head(precision="fp8", evidence_dir=tmp_path, source_hashes=binding)
    assert dspark.project_through_lm_head is projection
    assert logits_processor.should_apply_lm_head_quant_method is dispatch
    assert dspark.DSparkDraftMixin.attach_shared_modules is not attach
    assert dspark_worker_v2.DSparkWorkerV2.init_cuda_graphs is not graph
