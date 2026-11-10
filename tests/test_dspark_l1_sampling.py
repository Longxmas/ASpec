# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""DSpark-only sampling: independent distribution oracle, gradients and recovery."""

import copy
import math
import os
from argparse import Namespace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from aspec.config import config_to_flat_args, load_config
from aspec.models import dspark as dspark_module
from aspec.models.ops.loss import compiled_l1_from_logits, sampling_logits
from aspec.planning import PlanCompileError, compile_execution_plan
from aspec.training.checkpoint import (
    _logical_training_topology,
    _validate_exact_resume_topology,
)
from aspec.training.dspark_trainer import DSparkTrainer

DEFAULT = (1.0, 1.0, -1)
TARGET = (0.8, 0.85, 5)
DRAFT = (1.2, 0.9, 7)
CASES = [(DEFAULT, DEFAULT), (TARGET, DEFAULT), (DEFAULT, DRAFT), (TARGET, DRAFT)]
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/examples/sglang_qwen3_8b_dspark.yaml"
EAGER_L1 = compiled_l1_from_logits._torchdynamo_orig_callable
GPU = pytest.mark.skipif(os.getenv("ASPEC_RUN_LOSS_SAMPLING_GPU") != "1", reason="explicit GPU integration")


def reference_probs(logits, sampling):
    """Select support with scalar Python arithmetic, retaining autograd only for probabilities."""
    temperature, top_p, top_k = sampling
    keep = []
    for row in logits.detach().float().reshape(-1, logits.shape[-1]).cpu().tolist():
        maximum = max(row)
        values = [math.exp((value - maximum) / temperature) for value in row]
        total = sum(values)
        probabilities = [value / total for value in values]
        order = sorted(range(len(row)), key=lambda index: (-row[index], index))
        support = [False] * len(row)
        cumulative = 0.0
        for rank, index in enumerate(order):
            support[index] = (top_k == -1 or rank < top_k) and (top_p == 1.0 or cumulative <= top_p)
            cumulative += probabilities[index]
        keep.append(support)
    keep = torch.tensor(keep, device=logits.device).reshape(logits.shape)
    values = logits.float()
    values = (values - values.amax(-1, keepdim=True)) / temperature
    return values.masked_fill(~keep, float("-inf")).softmax(-1)


@pytest.mark.parametrize("sampling", [DEFAULT, (0.4, 1.0, -1), (1.0, 1.0, 1),
                                     (1.0, 1.0, 100), (1.0, 0.5, 10**30), (1.0, 0.5, -1), TARGET, DRAFT])
def test_distribution_matches_independent_reference(sampling):
    logits = torch.tensor([[3.0, -2.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]], requires_grad=True)
    before = logits.detach().clone()
    actual = sampling_logits(logits, sampling).softmax(-1)
    expected = reference_probs(logits, sampling)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.sum(-1), torch.ones(2))
    torch.testing.assert_close(logits, before, rtol=0, atol=0)
    assert torch.isfinite(actual).all()
    if sampling == DEFAULT:
        assert sampling_logits(logits, sampling) is logits


def test_joint_filter_uses_full_distribution_and_keeps_boundary_token():
    logits = torch.tensor([[0.4, 0.3, 0.2, 0.1]]).log()
    # Joint top-2/top-p=0.5 keeps A,B. Renormalizing top-2 first would keep only A.
    actual = sampling_logits(logits, (1.0, 0.5, 2)).softmax(-1)
    torch.testing.assert_close(actual, torch.tensor([[4 / 7, 3 / 7, 0.0, 0.0]]))
    boundary = sampling_logits(torch.zeros(1, 4), (1.0, 0.5, -1)).softmax(-1)
    torch.testing.assert_close(boundary, torch.tensor([[1 / 3, 1 / 3, 1 / 3, 0.0]]))
    tiny = sampling_logits(torch.tensor([[1e30, 0.0, -1e30]]), (1e-10, 0.1, 1)).softmax(-1)
    torch.testing.assert_close(tiny, torch.tensor([[1.0, 0.0, 0.0]]))


@pytest.mark.parametrize("target_sampling,draft_sampling", CASES)
def test_l1_outputs_and_gradients(target_sampling, draft_sampling):
    torch.manual_seed(15)
    logits = torch.randn(2, 3, 17, requires_grad=True)
    teacher = torch.randn_like(logits, requires_grad=True)
    weights = torch.tensor([[0.0, 0.25, 1.0], [1.5, 0.0, 2.0]])
    expected = (reference_probs(logits, draft_sampling) - reference_probs(teacher, target_sampling)).abs().sum(-1)
    actual = EAGER_L1(logits, teacher, target_sampling, draft_sampling)
    expected_grad = torch.autograd.grad((expected * weights).sum(), (logits, teacher))
    actual_grad = torch.autograd.grad((actual * weights).sum(), (logits, teacher))
    torch.testing.assert_close(actual, expected)
    for value, reference in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(value, reference, atol=1e-6, rtol=1e-5)
    if (target_sampling, draft_sampling) == (DEFAULT, DEFAULT):
        baseline = (logits.softmax(-1) - teacher.softmax(-1)).abs().sum(-1)
        torch.testing.assert_close(actual, baseline, rtol=0, atol=0)


class ConfidenceHead(torch.nn.Linear):
    def forward(self, value):
        return super().forward(value).squeeze(-1)


def head_model(target_sampling, draft_sampling, device="cpu", dtype=torch.float32):
    draft = torch.nn.Module()
    draft.markov_head = None
    draft.confidence_head = ConfidenceHead(4, 1).to(device=device, dtype=dtype)
    draft.confidence_head_with_markov = False
    return dspark_module.DSparkModel(draft, block_size=3, num_anchors=2,
                                    l1_target_sampling=target_sampling, l1_draft_sampling=draft_sampling)


@pytest.mark.parametrize("target_sampling,draft_sampling", CASES)
@pytest.mark.parametrize("cached", [False, True])
def test_dspark_ce_l1_confidence_and_cache_wiring(target_sampling, draft_sampling, cached, monkeypatch):
    monkeypatch.setattr(dspark_module, "compiled_l1_from_logits", EAGER_L1)
    torch.manual_seed(34)
    model = head_model(target_sampling, draft_sampling)
    hidden = torch.randn(1, 2, 3, 4, requires_grad=True)
    teacher = torch.randn(1, 7, 4)
    head = torch.randn(17, 4)
    positions = torch.tensor([[[0, 1, 2], [4, 5, 6]]])
    aligned = teacher[:, positions.flatten()].reshape_as(hidden)
    logits = F.linear(hidden, head)
    teacher_logits = F.linear(aligned, head)
    ids = torch.tensor([[[0, 5, 16], [1, 6, 13]]])
    weights = torch.tensor([[[1.0, 0.5, 0.0], [0.0, 0.25, 1.0]]])
    kwargs = {"cached_target_logits": F.linear(teacher, head), "cached_target_positions": positions} if cached else {}
    terms = model._prediction_terms(hidden, ids, ids, head, aligned, weights > 0, weights, **kwargs)
    ce = F.cross_entropy(logits.flatten(0, 2), ids.flatten(), reduction="none").reshape_as(weights)
    l1 = (reference_probs(logits, draft_sampling) - reference_probs(teacher_logits, target_sampling)).abs().sum(-1)
    alpha = (1 - l1 / 2).clamp(0, 1)
    confidence = F.binary_cross_entropy_with_logits(model.draft_model.confidence_head(hidden), alpha.detach(), reduction="none")
    expected = [(ce * weights).sum(), (l1 * weights).sum(), (confidence * weights).sum()]
    for value, reference in zip(terms[:3], expected, strict=True):
        torch.testing.assert_close(value, reference)
    variables = (hidden, *model.draft_model.confidence_head.parameters())
    actual_grad = torch.autograd.grad(sum(terms[:3]), variables, retain_graph=True)
    expected_grad = torch.autograd.grad(sum(expected), variables)
    for value, reference in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(value, reference, atol=1e-6, rtol=1e-5)
    assert model.l1_loss_backend == "torch_compiled"
    assert not model.l1_kernel_invocations


def test_disjoint_support_keeps_ce_finite_and_trainable(monkeypatch):
    monkeypatch.setattr(dspark_module, "compiled_l1_from_logits", EAGER_L1)
    model = head_model((1.0, 1.0, 1), (1.0, 1.0, 1))
    hidden = torch.tensor([[[[0.0, 4.0, 1.0, 0.0]]]]).expand(1, 2, 3, 4).clone().requires_grad_()
    teacher = torch.tensor([[[[4.0, 0.0, 1.0, 0.0]]]]).expand_as(hidden)
    ids = torch.zeros(1, 2, 3, dtype=torch.long)
    weights = torch.tensor([[[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]])
    terms = model._prediction_terms(hidden, ids, ids, torch.eye(4), teacher, weights > 0, weights)
    assert all(torch.isfinite(value).all() for value in terms)
    torch.testing.assert_close(terms[1], 2 * weights.sum())
    l1_gradient = torch.autograd.grad(terms[1], hidden, retain_graph=True)[0]
    ce_gradient = torch.autograd.grad(terms[0], hidden)[0]
    assert l1_gradient.count_nonzero() == 0
    assert (ce_gradient[..., 0][weights > 0] < 0).all()
    empty = model._prediction_terms(hidden, ids, ids, torch.eye(4), teacher, weights * 0, weights * 0)
    assert all(value.count_nonzero() == 0 for value in empty)


def config_with(**overrides):
    return load_config(config_path=str(CONFIG), cli_args=[f"{key}={value}" for key, value in overrides.items()])


def test_plan_runtime_trainer_and_checkpoint_keep_both_policies():
    overrides = {f"training.dspark_l1_{side}_{name}": value
                 for side, values in (("target", TARGET), ("draft", DRAFT))
                 for name, value in zip(("temperature", "top_p", "top_k"), values, strict=True)}
    config = config_with(**overrides, **{"training.attention_backend": "flex_attention"})
    plan = compile_execution_plan(config, config_path=str(CONFIG)).to_dict()
    args = config_to_flat_args(config)
    args.draft_algorithm = "dspark"
    args.world_size = 1
    trainer = DSparkTrainer.__new__(DSparkTrainer)
    trainer.args = args
    trainer.block_size = args.dflash_block_size
    for name in ("num_anchors", "loss_decay_gamma", "ce_loss_alpha", "l1_loss_alpha", "confidence_head_alpha"):
        setattr(trainer, name, getattr(args, f"dspark_{name}"))
    wrapper = trainer._build_training_wrapper(head_model(DEFAULT, DEFAULT).draft_model)
    assert wrapper.l1_sampling == (TARGET, DRAFT)
    topology = _logical_training_topology(args, 1)
    for side, values in (("target", TARGET), ("draft", DRAFT)):
        for name, value in zip(("temperature", "top_p", "top_k"), values, strict=True):
            assert plan["workload"]["objective"][f"l1_{side}_{name}"] == value
            assert topology["training_objective"][f"l1_{side}_{name}"] == value
    _validate_exact_resume_topology(args, {"world_size": 1, "training_topology": topology})
    for key in overrides:
        changed = copy.copy(args)
        field = key.removeprefix("training.")
        setattr(changed, field, -1 if field.endswith("top_k") else 1.0)
        with pytest.raises(RuntimeError, match=field):
            _validate_exact_resume_topology(changed, {"world_size": 1, "training_topology": topology})


@pytest.mark.parametrize("side", ["target", "draft"])
@pytest.mark.parametrize("name,value", [("temperature", 0), ("temperature", -1), ("temperature", "nan"),
                                        ("temperature", "inf"),
                                        ("top_p", 0), ("top_p", 1.1), ("top_p", "nan"),
                                        ("top_k", 0), ("top_k", -2)])
def test_plan_rejects_invalid_sampling(side, name, value):
    config = config_with(**{f"training.dspark_l1_{side}_{name}": value})
    with pytest.raises(PlanCompileError, match=f"dspark_l1_{side}_{name}"):
        compile_execution_plan(config, config_path=str(CONFIG))


@pytest.mark.parametrize("draft_path", ["dflash_draft_config.json", "eagle3_qwen3_8b_draft_config.json"])
def test_sampling_is_rejected_for_other_models(draft_path):
    config = config_with(**{"model.draft_model_config": str(ROOT / "aspec/config" / draft_path),
                            "training.dspark_l1_target_top_k": 2})
    with pytest.raises(PlanCompileError, match="DSpark L1 sampling requires"):
        compile_execution_plan(config, config_path=str(CONFIG))


def test_sampling_requires_an_active_l1_objective():
    config = config_with(**{"training.dspark_l1_loss_alpha": 0, "training.dspark_l1_draft_top_p": 0.8})
    with pytest.raises(PlanCompileError, match="DSpark L1 sampling requires"):
        compile_execution_plan(config, config_path=str(CONFIG))


def test_legacy_checkpoint_means_default_sampling():
    args = Namespace(world_size=1, draft_algorithm="dspark", dflash_block_size=7)
    legacy = {"world_size": 1, "training_topology": {"world_size": 1, "draft_algorithm": "dspark",
                                                     "training_objective": {"block_size": 7}}}
    _validate_exact_resume_topology(args, legacy)
    args.dspark_l1_draft_top_k = 2
    with pytest.raises(RuntimeError, match="dspark_l1_draft_top_k"):
        _validate_exact_resume_topology(args, legacy)


@GPU
@pytest.mark.parametrize("target_sampling,draft_sampling", CASES + [
    ((0.8, 0.7, -1), DEFAULT),
    (DEFAULT, (1.2, 0.8, -1)),
    ((0.8, 0.7, -1), (1.2, 0.8, -1)),
])
def test_cuda_compiled_l1_large_vocabulary(target_sampling, draft_sampling):
    torch.manual_seed(19)
    logits = torch.randn(1, 2, 3, 8192, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    teacher = torch.randn_like(logits)
    before = logits.detach().clone()
    actual = compiled_l1_from_logits(logits, teacher, target_sampling, draft_sampling)
    expected = (reference_probs(logits, draft_sampling) - reference_probs(teacher, target_sampling)).abs().sum(-1)
    actual_gradient = torch.autograd.grad(actual.sum(), logits)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), logits)[0]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-5, rtol=0.02)
    torch.testing.assert_close(logits, before, atol=0, rtol=0)


@GPU
@pytest.mark.parametrize("target_sampling,draft_sampling", CASES)
def test_cuda_dspark_training_and_resume(target_sampling, draft_sampling, tmp_path):
    from aspec.models.draft.dspark import DSparkConfig, DSparkDraftModel

    def make_model():
        config = DSparkConfig(
            hidden_size=64, intermediate_size=128, head_dim=16, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
            rms_norm_eps=1e-6, max_position_embeddings=64, rope_theta=10000.0,
            num_target_layers=2, target_hidden_size=64, target_num_hidden_layers=12,
            mask_token_id=127, markov_rank=8, markov_head_type="vanilla",
            enable_confidence_head=True, confidence_head_with_markov=True,
        )
        draft = DSparkDraftModel(config).to(device="cuda", dtype=torch.bfloat16)
        draft.freeze_embedding()
        return dspark_module.DSparkModel(
            draft, block_size=4, num_anchors=4, attention_backend="flex_attention",
            l1_target_sampling=target_sampling, l1_draft_sampling=draft_sampling,
        )

    torch.manual_seed(62)
    model = make_model()
    batch = {
        "input_ids": torch.randint(0, 128, (1, 32), device="cuda"),
        "hidden_states_list": [torch.randn(1, 32, 64, device="cuda", dtype=torch.bfloat16) for _ in range(2)],
        "last_hidden_states": torch.randn(1, 32, 64, device="cuda", dtype=torch.bfloat16),
        "lm_head_weight": torch.randn(128, 64, device="cuda", dtype=torch.bfloat16) * 0.1,
        "loss_mask": torch.ones(1, 32, device="cuda"),
        "injected_anchors": torch.tensor([[2, 9, 17, 25]], device="cuda"),
        "injected_keep_mask": torch.ones(1, 4, device="cuda", dtype=torch.bool),
    }
    batch["loss_mask"][:, :2] = 0
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.002)

    def step(candidate, opt):
        opt.zero_grad(set_to_none=True)
        output = candidate(**batch)
        loss = output[0]
        assert torch.isfinite(loss)
        loss.backward()
        gradients = [p.grad for p in candidate.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(gradient.count_nonzero() > 0 for gradient in gradients)
        opt.step()
        assert all(torch.isfinite(p).all() for p in candidate.parameters())
        return loss.detach()

    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    step(model, optimizer)
    step(model, optimizer)
    assert any(not torch.equal(before[name], value) for name, value in model.named_parameters())
    checkpoint = tmp_path / "sampling.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
    expected = step(model, optimizer)
    restored = make_model()
    restored_optimizer = torch.optim.AdamW((p for p in restored.parameters() if p.requires_grad), lr=0.002)
    saved = torch.load(checkpoint, map_location="cuda", weights_only=True)
    restored.load_state_dict(saved["model"])
    restored_optimizer.load_state_dict(saved["optimizer"])
    actual = step(restored, restored_optimizer)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for value, reference in zip(restored.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
