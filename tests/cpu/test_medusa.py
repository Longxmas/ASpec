# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Medusa heads-only reference parity, supervision offsets and fail-closed config.

Reference: FasterDecoding/Medusa e2a5d20c048a9b0a4092e6933c34313687422518
(medusa/model/medusa_model.py and medusa/train/train_legacy.py).
"""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from aspec.models.draft.medusa import MedusaConfig, MedusaDraftModel, MedusaResBlock
from aspec.models.medusa import MedusaModel


def make_config(num_heads=3, num_layers=1, hidden=16, vocab=32, target_layers=4):
    return MedusaConfig(hidden_size=hidden, vocab_size=vocab, target_hidden_size=hidden,
                        target_num_hidden_layers=target_layers,
                        medusa_num_heads=num_heads, medusa_num_layers=num_layers)


def make_model(num_heads=3, num_layers=1, seed=7, checkpointing=False):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        draft = MedusaDraftModel(make_config(num_heads, num_layers))
        # Exercise non-trivial residual mathematics instead of the zero-weight init.
        for head in draft.medusa_head:
            for module in head:
                if isinstance(module, MedusaResBlock):
                    nn.init.normal_(module.linear.weight, std=0.05)
    return MedusaModel(draft, gradient_checkpointing=checkpointing)


def make_inputs(batch=2, length=12, hidden=16, vocab=32, seed=31):
    generator = torch.Generator().manual_seed(seed)
    ids = torch.arange(batch * length).reshape(batch, length) % vocab
    return dict(input_ids=ids,
                hidden_states=torch.randn(batch, length, hidden, generator=generator),
                loss_mask=torch.ones(batch, length, dtype=torch.long),
                attention_mask=torch.ones(batch, length, dtype=torch.long))


def reference_head_logits(head, features):
    """Dense residual/projection equations independent of the module forwards."""
    value = features
    for module in head:
        if isinstance(module, MedusaResBlock):
            value = value + F.silu(F.linear(value, module.linear.weight, module.linear.bias))
        else:
            value = F.linear(value, module.weight, module.bias)
    return value


def reference_objective(model, inputs):
    """Per-head masked mean CE at the reference t+2+i offset, with counts/accuracy."""
    input_ids, hidden_states = inputs["input_ids"], inputs["hidden_states"]
    loss_mask = inputs["loss_mask"]
    lengths = inputs["attention_mask"].sum(-1).long().tolist()
    losses, counts, accuracies = [], [], []
    for head in range(model.num_heads):
        offset = head + 2
        total = hidden_states.new_zeros((), dtype=torch.float32)
        count = hits = 0
        for row, length in enumerate(lengths):
            valid = max(0, length - offset)
            if not valid:
                continue
            labels = input_ids[row, offset:offset + valid]
            keep = loss_mask[row, offset:offset + valid].bool()
            logits = reference_head_logits(model.draft_model.medusa_head[head],
                                           hidden_states[row, :valid])
            ce = F.cross_entropy(logits.float(), labels, reduction="none")
            total = total + ce[keep].sum()
            count += int(keep.sum())
            hits += int((logits.argmax(-1)[keep] == labels[keep]).sum())
        losses.append(total / max(count, 1))
        counts.append(count)
        accuracies.append(hits / max(count, 1))
    return losses, counts, accuracies


def _assert_parameter_gradients(actual, expected):
    actual_parameters = dict(actual.named_parameters())
    expected_parameters = dict(expected.named_parameters())
    assert actual_parameters.keys() == expected_parameters.keys()
    for name, parameter in actual_parameters.items():
        reference = expected_parameters[name]
        assert (parameter.grad is None) == (reference.grad is None), name
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(parameter.grad, reference.grad,
                                       rtol=2e-5, atol=2e-6, msg=name)


def test_config_derives_final_state_contract():
    config = make_config(target_layers=8)
    assert config.architectures == ["MedusaDraftModel"]
    assert config.model_type == "aspec_medusa"
    assert config.target_layer_ids == [7]
    assert config.num_target_layers == 1
    assert config.num_aux_hidden_states == 1
    assert config.draft_vocab_size == config.vocab_size
    assert config.tie_word_embeddings is False
    restored = MedusaConfig.from_dict(config.to_dict())
    assert restored.medusa_num_heads == config.medusa_num_heads
    assert restored.target_layer_ids == config.target_layer_ids


@pytest.mark.parametrize("override", [
    {"hidden_size": 0},
    {"hidden_size": None},
    {"vocab_size": 0},
    {"target_hidden_size": 8},
    {"target_num_hidden_layers": 0},
    {"medusa_num_heads": 0},
    {"medusa_num_layers": 0},
    {"medusa_num_heads": 2.5},
    {"draft_vocab_size": 16},
    {"ffn_type": "moe"},
    {"lm_head_rank_divisor": 2},
    {"num_hidden_layers": 1},
    {"target_layer_ids": [0]},
    {"num_aux_hidden_states": 3},
    {"architectures": ["DFlashDraftModel"]},
])
def test_config_fail_closed(override):
    kwargs = dict(hidden_size=16, vocab_size=32, target_hidden_size=16,
                  target_num_hidden_layers=4)
    kwargs.update(override)
    with pytest.raises(ValueError):
        MedusaConfig(**kwargs)


def test_from_dict_overrides_stay_validated():
    config = make_config()
    # The inherited loader applies overrides after construction; these must still fail.
    with pytest.raises(ValueError):
        MedusaConfig.from_dict(config.to_dict(), target_layer_ids=[0])
    with pytest.raises(ValueError):
        MedusaConfig.from_dict(config.to_dict(), medusa_num_heads=0)
    restored = MedusaConfig.from_dict(config.to_dict(), medusa_num_heads=2)
    assert restored.medusa_num_heads == 2


def test_from_dict_preserves_unused_arguments():
    config = make_config()
    restored, unused = MedusaConfig.from_dict(
        config.to_dict(), gradient_checkpointing=True, return_unused_kwargs=True)
    assert unused == {"gradient_checkpointing": True}
    assert not hasattr(restored, "gradient_checkpointing")


@pytest.mark.parametrize("override", [{"dtype": "auto"}, {"torch_dtype": "auto"}])
def test_from_dict_defers_auto_dtype(override):
    config = make_config()
    restored = MedusaConfig.from_dict(config.to_dict(), **override)
    assert restored.dtype == "auto"


def test_resblock_zero_weight_initialization():
    block = MedusaResBlock(16)
    assert torch.count_nonzero(block.linear.weight) == 0
    # The reference zeroes only the weight; the default bias init survives, so
    # the initial block is a constant SiLU(bias) offset rather than exact identity.
    assert torch.count_nonzero(block.linear.bias) > 0
    generator = torch.Generator().manual_seed(3)
    x1 = torch.randn(4, 16, generator=generator)
    x2 = torch.randn(4, 16, generator=generator)
    # (x + c) - x is constant only up to float rounding.
    torch.testing.assert_close(block(x1) - x1, block(x2) - x2, rtol=0, atol=1e-6)
    with torch.no_grad():
        block.linear.bias.zero_()
    torch.testing.assert_close(block(x1), x1, rtol=0, atol=0)


def test_head_structure_matches_reference():
    draft = MedusaDraftModel(make_config(num_heads=4, num_layers=2))
    assert len(draft.medusa_head) == 4
    for head in draft.medusa_head:
        assert len(head) == 3
        # The reference repeats one shared ResBlock instance per head depth.
        assert head[0] is head[1]
        assert isinstance(head[0], MedusaResBlock)
        assert isinstance(head[-1], nn.Linear)
        assert head[-1].bias is None
        assert head[-1].out_features == 32
    # The repeated depth shares one parameter tensor.
    state = draft.state_dict(keep_vars=True)
    assert state["medusa_head.0.0.linear.weight"] is state["medusa_head.0.1.linear.weight"]


@pytest.mark.parametrize("num_heads", [1, 4])
def test_forward_logits_match_reference_equations(num_heads):
    model = make_model(num_heads=num_heads, num_layers=2)
    features = torch.randn(2, 10, 16, generator=torch.Generator().manual_seed(5))
    logits = model.draft_model(features)
    assert logits.shape == (num_heads, 2, 10, 32)
    for head in range(num_heads):
        expected = reference_head_logits(model.draft_model.medusa_head[head], features)
        torch.testing.assert_close(logits[head], expected, rtol=0, atol=0)


@pytest.mark.parametrize("num_layers", [1, 2])
def test_loss_and_gradients_match_independent_reference(num_layers):
    actual = make_model(num_layers=num_layers)
    reference = copy.deepcopy(actual)
    inputs = make_inputs()
    inputs["loss_mask"][0, :4] = 0
    inputs["loss_mask"][1, :2] = 0
    inputs["loss_mask"][1, 7] = 0
    losses, detached, accuracies, counts, pad = actual(**inputs)
    assert len(losses) == actual.num_heads
    ref_losses, ref_counts, ref_accuracies = reference_objective(reference, inputs)
    for head in range(actual.num_heads):
        torch.testing.assert_close(losses[head], ref_losses[head], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(detached[head], losses[head].detach(), rtol=0, atol=0)
        assert int(counts[head]) == ref_counts[head]
        torch.testing.assert_close(accuracies[head],
                                   torch.tensor(ref_accuracies[head]), rtol=2e-5, atol=2e-6)
        assert float(pad[head]) == 0.0
    sum(losses).backward()
    sum(reference_objective(reference, inputs)[0]).backward()
    _assert_parameter_gradients(actual, reference)


@pytest.mark.parametrize("label_position", [3, 7])
def test_supervision_offset_pins_head_i_to_token_t_plus_2_i(label_position):
    model = make_model()
    inputs = make_inputs(batch=1)
    inputs["loss_mask"] = torch.zeros(1, 12, dtype=torch.long)
    inputs["loss_mask"][0, label_position] = 1
    losses, _, _, counts, _ = model(**inputs)
    for head in range(model.num_heads):
        feature_position = label_position - (head + 2)
        if feature_position < 0:
            assert int(counts[head]) == 0
            assert float(losses[head].detach()) == 0.0
            continue
        assert int(counts[head]) == 1
        logits = reference_head_logits(
            model.draft_model.medusa_head[head],
            inputs["hidden_states"][0, feature_position:feature_position + 1])
        expected = F.cross_entropy(
            logits.float(), inputs["input_ids"][0, label_position:label_position + 1])
        torch.testing.assert_close(losses[head], expected, rtol=2e-5, atol=2e-6)


def test_prompt_mask_and_padding_exclude_supervision():
    model = make_model()
    inputs = make_inputs()
    inputs["loss_mask"][0, :5] = 0
    inputs["attention_mask"][1, 8:] = 0
    # Adversarial: padding positions stay marked supervised in loss_mask.
    losses, _, _, counts, _ = model(**inputs)
    assert [int(count) for count in counts] == [13, 12, 11]
    ref_losses, ref_counts, _ = reference_objective(model, inputs)
    for head in range(model.num_heads):
        assert int(counts[head]) == ref_counts[head]
        torch.testing.assert_close(losses[head], ref_losses[head], rtol=2e-5, atol=2e-6)


def test_left_padded_rows_match_right_padded_documents():
    model = make_model()
    generator = torch.Generator().manual_seed(9)
    real_length, pad = 6, 2
    tokens = torch.arange(real_length)[None]
    features = torch.randn(1, real_length, 16, generator=generator)
    loss_mask = torch.ones(1, real_length + pad, dtype=torch.long)
    right = dict(input_ids=torch.cat((tokens, torch.zeros(1, pad, dtype=torch.long)), dim=1),
                 hidden_states=torch.cat(
                     (features, torch.randn(1, pad, 16, generator=generator)), dim=1),
                 attention_mask=torch.cat((torch.ones(1, real_length),
                                           torch.zeros(1, pad)), dim=1).long(),
                 loss_mask=loss_mask)
    # Adversarial: padding positions stay marked supervised in loss_mask.
    left = dict(input_ids=torch.cat((torch.zeros(1, pad, dtype=torch.long), tokens), dim=1),
                hidden_states=torch.cat(
                    (torch.randn(1, pad, 16, generator=generator), features), dim=1),
                attention_mask=torch.cat((torch.zeros(1, pad),
                                          torch.ones(1, real_length)), dim=1).long(),
                loss_mask=loss_mask)
    right_result = model(**right)
    left_result = model(**left)
    for head in range(model.num_heads):
        assert int(right_result[3][head]) == max(0, real_length - (head + 2))
        assert int(left_result[3][head]) == int(right_result[3][head])
        torch.testing.assert_close(left_result[0][head], right_result[0][head],
                                   rtol=0, atol=0)


def test_non_contiguous_positions_fail_closed():
    model = make_model()
    inputs = make_inputs(batch=1)
    inputs["attention_mask"][0, 4] = 0
    with pytest.raises(ValueError):
        model(**inputs)


def test_rows_are_independent_documents():
    model = make_model()
    inputs = make_inputs()
    inputs["attention_mask"][1] = 0
    partial_losses, _, _, partial_counts, _ = model(**inputs)
    solo_losses, _, _, solo_counts, _ = model(**make_inputs(batch=1))
    for head in range(model.num_heads):
        torch.testing.assert_close(partial_losses[head], solo_losses[head], rtol=0, atol=0)
        assert int(partial_counts[head]) == int(solo_counts[head])


def test_heads_have_no_cross_position_or_future_leakage():
    model = make_model()
    inputs = make_inputs()
    logits = model.draft_model(inputs["hidden_states"])
    perturbed = inputs["hidden_states"].clone()
    perturbed[:, 6:] = torch.randn_like(perturbed[:, 6:])
    moved = model.draft_model(perturbed)
    torch.testing.assert_close(logits[:, :, :6], moved[:, :, :6], rtol=0, atol=0)
    # Masked label tokens never enter the objective.
    inputs["loss_mask"][0, :5] = 0
    base_losses, _, _, _, _ = model(**inputs)
    changed_ids = inputs["input_ids"].clone()
    changed_ids[0, :5] = (changed_ids[0, :5] + 1) % 32
    new_losses, _, _, _, _ = model(**dict(inputs, input_ids=changed_ids))
    for head in range(model.num_heads):
        torch.testing.assert_close(base_losses[head], new_losses[head], rtol=0, atol=0)


@pytest.mark.parametrize("length", [1, 2, 3, 4])
def test_short_sequences_keep_losses_finite(length):
    model = make_model()
    inputs = make_inputs(batch=1, length=length)
    losses, _, _, counts, _ = model(**inputs)
    for head in range(model.num_heads):
        assert torch.isfinite(losses[head])
        assert int(counts[head]) == max(0, length - (head + 2))
        if int(counts[head]) == 0:
            assert float(losses[head].detach()) == 0.0
    sum(losses).backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_empty_loss_mask_is_neutral():
    model = make_model()
    inputs = make_inputs()
    inputs["loss_mask"] = torch.zeros_like(inputs["loss_mask"])
    losses, _, _, counts, _ = model(**inputs)
    assert all(int(count) == 0 for count in counts)
    assert all(float(loss.detach()) == 0.0 for loss in losses)
    sum(losses).backward()
    for parameter in model.parameters():
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0


def test_many_supervised_positions_match_reference():
    model = make_model()
    inputs = make_inputs(batch=2, length=300)
    losses, _, _, counts, _ = model(**inputs)
    ref_losses, ref_counts, _ = reference_objective(model, inputs)
    for head in range(model.num_heads):
        assert int(counts[head]) == ref_counts[head] > 128
        torch.testing.assert_close(losses[head], ref_losses[head], rtol=2e-5, atol=2e-6)


def test_gradient_checkpointing_preserves_loss_and_gradients():
    plain = make_model(seed=11)
    checked = make_model(seed=11, checkpointing=True)
    inputs = make_inputs(seed=41)
    inputs["loss_mask"][0, :3] = 0
    actual = plain(**inputs)
    expected = checked(**inputs)
    for head in range(plain.num_heads):
        torch.testing.assert_close(actual[0][head], expected[0][head], rtol=0, atol=0)
    sum(actual[0]).backward()
    sum(expected[0]).backward()
    _assert_parameter_gradients(plain, checked)


def test_gradient_checkpointing_dispatches_on_the_flag(monkeypatch):
    import aspec.models.medusa as medusa_module

    original = medusa_module.checkpoint
    calls = []

    def counting_checkpoint(function, *args, **kwargs):
        calls.append(function.__name__)
        return original(function, *args, **kwargs)

    monkeypatch.setattr(medusa_module, "checkpoint", counting_checkpoint)
    plain = make_model(checkpointing=False)
    plain.train()
    plain(**make_inputs())
    assert calls == []
    checked = make_model(checkpointing=True)
    checked.train()
    checked(**make_inputs())
    assert calls and set(calls) == {"loss_chunk"}


def test_eval_mode_without_grad():
    model = make_model()
    model.eval()
    with torch.no_grad():
        losses, _, _, _, _ = model(**make_inputs())
    assert all(torch.isfinite(loss) for loss in losses)


def test_fully_frozen_model_keeps_zero_accumulator_on_device():
    model = make_model()
    model.requires_grad_(False)
    losses, _, _, counts, _ = model(**make_inputs())
    assert all(torch.isfinite(loss) for loss in losses)
    assert all(int(count) > 0 for count in counts)


def test_bfloat16_features_produce_finite_losses():
    model = make_model()
    model.to(torch.bfloat16)
    inputs = make_inputs()
    inputs["hidden_states"] = inputs["hidden_states"].to(torch.bfloat16)
    losses, _, _, _, _ = model(**inputs)
    assert all(torch.isfinite(loss) for loss in losses)


def test_bfloat16_counts_stay_exact_above_256_positions():
    model = make_model().to(torch.bfloat16)
    inputs = make_inputs(batch=1, length=299)
    inputs["hidden_states"] = inputs["hidden_states"].to(torch.bfloat16)
    losses, _, _, counts, _ = model(**inputs)
    reference_losses, reference_counts, _ = reference_objective(model, inputs)
    for head in range(model.num_heads):
        assert int(counts[head]) == reference_counts[head] == 299 - (head + 2)
        torch.testing.assert_close(losses[head].float(), reference_losses[head],
                                   rtol=2e-2, atol=2e-2)


def test_thd_execution_fails_closed():
    model = make_model()
    with pytest.raises(NotImplementedError):
        model(**make_inputs(), thd_layout=object())


def test_objective_does_not_backpropagate_into_features():
    model = make_model()
    inputs = make_inputs()
    inputs["hidden_states"].requires_grad_(True)
    losses, _, _, _, _ = model(**inputs)
    sum(losses).backward()
    assert inputs["hidden_states"].grad is None
