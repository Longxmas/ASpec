# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""CUDA loss/gradient parity when moving native MTP capture to Target."""

import copy
import os

import pytest
import torch

from aspec.inference.mtp_features import FrozenTargetMTP
from aspec.models.dmtp import MTPPrefix
from tests.cpu.test_dmtp import _fusion_inputs, _nonzero_model
from tests.cpu.test_target_mtp_features import capture_policy, target_model
from tests.cpu.test_target_mtp_features import native_weights as native_weights

pytestmark = pytest.mark.skipif(
    os.getenv("ASPEC_RUN_GPU_INTEGRATION") != "1",
    reason="set ASPEC_RUN_GPU_INTEGRATION=1 on an allocated CUDA device",
)


@pytest.mark.parametrize("algorithm", ["dflash", "dspark"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_target_capture_cuda_outputs_and_gradients(native_weights, algorithm, dtype):
    template = _nonzero_model(algorithm)
    config = copy.deepcopy(template.draft_model.config)
    config.head_dim = 16  # FlexAttention's CUDA head-width lower bound.
    draft = type(template.draft_model)(config)
    draft.initialize_mtp_conditioning("checkpoint")
    torch.nn.init.normal_(draft.mtp_fusion.proj.weight, std=0.1)
    reference = type(template)(draft, block_size=5, num_anchors=2, attention_backend="sdpa")
    actual = target_model(reference).to(device="cuda", dtype=dtype)
    reference = reference.to(device="cuda", dtype=dtype)
    producer = FrozenTargetMTP.from_target(capture_policy(), "checkpoint", draft.embed_tokens.weight,
                                           device=draft.embed_tokens.weight.device, dtype=dtype)
    inputs = {name: value.to(device="cuda", dtype=dtype if value.is_floating_point() else value.dtype)
              for name, value in _fusion_inputs().items() if name != "draft_prefix"}
    captured = producer(inputs["input_ids"][0], inputs["last_hidden_states"][0])[None]
    captured.requires_grad_()
    output = actual(**inputs, draft_prefix=MTPPrefix(captured[:, 1:]))
    expected = reference(**inputs)
    tolerance = dict(rtol=0.01, atol=0.001) if dtype == torch.bfloat16 else dict(rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(output, expected, **tolerance)
    output[0].backward()
    expected[0].backward()
    previous = dict(reference.named_parameters())
    for name, parameter in actual.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            torch.testing.assert_close(parameter.grad, previous[name].grad, msg=name, **tolerance)
    assert captured.grad is None
    assert all(p.grad is None and not p.requires_grad for p in producer.parameters())
