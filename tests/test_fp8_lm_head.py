# Copyright 2026 Ant Group Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""SM90 checks for the standalone FP8 head experiment; explicit GPU opt-in."""

import os

import pytest
import torch
import torch.nn.functional as F

from aspec.models.ops.dspark_fp8_lm_head import FrozenFP8Head, quantize_activation

pytestmark = pytest.mark.skipif(
    os.environ.get("ASPEC_RUN_GPU_INTEGRATION") != "1" or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0),
    reason="requires ASPEC_RUN_GPU_INTEGRATION=1 and SM90 CUDA",
)


def reference_rows(x):
    # Independent FP64 scaling arithmetic, with explicit dequantization.
    source = x.double()
    amax = source.abs().max(dim=1, keepdim=True).values.clamp_min(1e-12)
    # ldexp constructs exact powers of two; generic CUDA pow can add FP64
    # rounding noise even when the mathematical result is exactly binary.
    exponent = torch.ceil(torch.log2(amax / 448.0)).to(torch.int32)
    scale = torch.ldexp(torch.ones_like(amax), exponent)
    bits = (source / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return bits.double() * scale


@pytest.mark.parametrize("rows", [0, 1, 15, 16, 17, 63])
@pytest.mark.parametrize("strided", [False, True])
def test_both_gemms_match_independent_quantized_reference(rows, strided):
    torch.manual_seed(901)
    weight = torch.randn(48, 32, device="cuda", dtype=torch.bfloat16) * 0.1
    source = torch.randn(rows, 64 if strided else 32, device="cuda", dtype=torch.bfloat16)
    hidden = (source[:, ::2] if strided else source).detach().requires_grad_()
    head = FrozenFP8Head(weight)
    output = head(hidden)
    grad = torch.randn(rows, 48, device="cuda", dtype=torch.bfloat16) * 0.001
    output.backward(grad)
    if rows:
        expected = reference_rows(hidden.detach()) @ reference_rows(weight).t()
        expected_grad = reference_rows(grad) @ reference_rows(weight.t()).t()
        torch.testing.assert_close(output, expected.bfloat16(), atol=1e-3, rtol=1e-2)
        torch.testing.assert_close(hidden.grad, expected_grad.bfloat16(), atol=1e-5, rtol=1e-2)
    else:
        assert output.shape == (0, 48)
        assert hidden.grad.shape == hidden.shape
    assert weight.grad is None


def test_zero_rows_and_stale_weight_cache():
    weight = torch.zeros(48, 32, device="cuda", dtype=torch.bfloat16)
    head = FrozenFP8Head(weight)
    hidden = torch.zeros(2, 17, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    head(hidden).sum().backward()
    assert torch.count_nonzero(hidden.grad) == 0
    assert torch.isfinite(head(hidden)).all()
    weight.add_(1)
    with pytest.raises(ValueError, match="changed"):
        head(hidden)


def test_trainable_head_is_rejected():
    weight = torch.zeros(48, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with pytest.raises(ValueError, match="frozen"):
        FrozenFP8Head(weight)


def test_gradient_quality_against_bf16_and_checkpoint_recomputation():
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(73)
    weight = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16) * 0.02
    hidden = torch.randn(33, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    head = FrozenFP8Head(weight)
    labels = torch.arange(33, device="cuda") % 256
    expected = torch.autograd.grad(F.cross_entropy(F.linear(hidden, weight).float(), labels), hidden)[0]
    actual = torch.autograd.grad(F.cross_entropy(head(hidden).float(), labels), hidden)[0]
    replay = torch.autograd.grad(
        checkpoint(lambda x: F.cross_entropy(head(x).float(), labels), hidden, use_reentrant=False), hidden
    )[0]
    # A bounded random-fixture smoke, not a real-model quality threshold.
    assert (actual.float() - expected.float()).norm() / expected.float().norm() < 0.08
    torch.testing.assert_close(actual, replay, rtol=0, atol=0)


def test_production_vocabulary_backward_reduction():
    # The backward reduction dimension is the entire vocabulary, much larger
    # than the forward hidden width. Exercise the actual Qwen3-8B vocabulary.
    torch.manual_seed(19)
    weight = torch.randn(151936, 128, device="cuda", dtype=torch.bfloat16) * 0.02
    hidden = torch.randn(17, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    upstream = torch.randn(17, 151936, device="cuda", dtype=torch.bfloat16) * 1e-5
    head = FrozenFP8Head(weight)
    output = head(hidden)
    actual, = torch.autograd.grad(output, hidden, upstream)
    expected = upstream.float() @ weight.float()
    assert torch.isfinite(actual).all()
    assert (actual.float() - expected).norm() / expected.norm() < 0.08


@pytest.mark.parametrize("width", [32, 4096, 151936])
def test_tiled_quantization_and_padding_match_independent_reference(width):
    torch.manual_seed(51)
    values = torch.randn(17, width, device="cuda", dtype=torch.bfloat16)
    values *= torch.logspace(-8, 2, 17, device="cuda")[:, None]
    values[0].zero_()
    values[3, -1] = 448
    bits, scale = quantize_activation(values)
    actual = bits[:17].double() * scale[:17].double()
    torch.testing.assert_close(actual, reference_rows(values), rtol=0, atol=0)
    assert bits.shape == (32, width)
    assert torch.count_nonzero(bits[17:].float()) == 0
    assert torch.isfinite(scale).all()


@pytest.mark.parametrize("width", [32, 8208])
def test_quantization_uses_input_device(width):
    if torch.cuda.device_count() < 2 or torch.cuda.get_device_capability(1) != (9, 0):
        pytest.skip("requires a second visible SM90 CUDA device")

    with torch.cuda.device(1):
        values = torch.zeros((17, width), device="cuda:1", dtype=torch.bfloat16)
        values[:, 0] = 0.5
        values[:, -1] = -1
    with torch.cuda.device(0):
        bits, scale = quantize_activation(values)
        assert torch.cuda.current_device() == 0

    assert bits.device == scale.device == values.device
    with torch.cuda.device(1):
        actual = bits[:17].float() * scale[:17]
        torch.testing.assert_close(actual, values.float(), rtol=0, atol=0)


@pytest.mark.parametrize("width", [32, 151936])
def test_quantization_preserves_nan_rows_and_padding(width):
    torch.manual_seed(53)
    values = torch.randn(17, width, device="cuda", dtype=torch.bfloat16)
    values[0].zero_()
    # Exercise the first, middle and final reduction tiles, including a tail row.
    values[1, 0] = values[4, width // 2] = values[16, -1] = float("nan")
    bits, scale = quantize_activation(values)
    invalid = values.isnan().any(dim=1)
    assert scale[:17][invalid].isnan().all()
    assert bits[:17][invalid].float().isnan().all()
    actual = bits[:17][~invalid].double() * scale[:17][~invalid].double()
    torch.testing.assert_close(actual, reference_rows(values[~invalid]), rtol=0, atol=0)
    assert torch.count_nonzero(bits[17:].float()) == 0
    assert scale[17:].isfinite().all()


def test_nan_propagates_through_forward_and_input_gradient():
    weight = torch.ones(8208, 32, device="cuda", dtype=torch.bfloat16)
    hidden = torch.ones(3, 32, device="cuda", dtype=torch.bfloat16)
    hidden[1, -1] = float("nan")
    hidden.requires_grad_()
    output = FrozenFP8Head(weight)(hidden)
    assert output[1].isnan().all()
    assert output[[0, 2]].isfinite().all()
    upstream = torch.ones_like(output)
    upstream[2, -1] = float("nan")
    output.backward(upstream)
    assert hidden.grad[2].isnan().all()
    assert hidden.grad[:2].isfinite().all()


def test_native_dspark_losses_keep_teacher_and_validation_bf16(monkeypatch):
    from types import SimpleNamespace

    from aspec.models.draft.dspark import AcceptRatePredictor, VanillaMarkov
    from aspec.models.dspark import DSparkModel
    from aspec.training.dflash_trainer import DFlashTrainer
    from aspec.training.dspark_trainer import DSparkTrainer

    torch.manual_seed(82)
    draft = torch.nn.Module()
    draft.markov_head = VanillaMarkov(vocab_size=256, markov_rank=16)
    draft.confidence_head = AcceptRatePredictor(144)
    draft.confidence_head_with_markov = True
    model = DSparkModel(draft).cuda().to(torch.bfloat16)
    hidden = torch.randn(1, 3, 7, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    teacher = torch.randn_like(hidden, requires_grad=False)
    weight = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16) * 0.02
    ids = torch.randint(0, 256, (1, 3, 7), device="cuda")
    mask = torch.ones_like(ids, dtype=torch.float32)
    weights = mask * torch.exp(-torch.arange(7, device="cuda").float() / 4)
    arguments = (hidden, ids, ids.roll(1, -1), weight, teacher, mask, weights)
    model.eval()
    with torch.no_grad():
        baseline = model._prediction_terms(*arguments)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    trainer = DSparkTrainer.__new__(DSparkTrainer)
    trainer.args = SimpleNamespace(dspark_lm_head_precision="fp8")
    trainer.dflash = model
    trainer.target_lm_head = SimpleNamespace(lm_head=SimpleNamespace(weight=weight))
    monkeypatch.setattr(DFlashTrainer, "_init_target_lm_head", lambda self, path: None)
    trainer._init_target_lm_head("test-model")
    assert model.fp8_lm_head.cache_bytes > 0
    assert model.state_dict().keys() == before.keys()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    with torch.no_grad():
        validation = model._prediction_terms(*arguments)
    for actual, expected in zip(validation, baseline, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    native_linear = F.linear
    teacher_inputs = []

    def observe(source, projection, bias=None):
        if projection is weight:
            teacher_inputs.append(source.detach().clone())
        return native_linear(source, projection, bias)

    monkeypatch.setattr(F, "linear", observe)
    model.train()
    terms = model._prediction_terms(*arguments)
    (0.1 * terms[0] + 0.9 * terms[1] + terms[2]).backward()
    assert len(teacher_inputs) == 1
    torch.testing.assert_close(teacher_inputs[0], teacher.reshape(-1, 128), rtol=0, atol=0)
    assert torch.isfinite(hidden.grad).all() and hidden.grad.norm() > 0
    assert weight.grad is None
    assert all(p.dtype == torch.bfloat16 and p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())
    with pytest.raises(ValueError, match="different frozen weight"):
        model._project_draft(hidden, weight.clone())
