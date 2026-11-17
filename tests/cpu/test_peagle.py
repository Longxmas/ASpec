# SPDX-License-Identifier: Apache-2.0
"""Independent scalar/linear-algebra checks of the bounded P-EAGLE core."""

import copy
import math

import pytest
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, Qwen3Config

from aspec.models.peagle import (
    PEagleDraftModel,
    peagle_attention_mask,
    peagle_dense_layout,
    peagle_kl_loss,
)


def model(*, layers=2, depths=4, implementation="sdpa", **changes):
    config = Qwen3Config(
        vocab_size=23, hidden_size=16, intermediate_size=32,
        num_hidden_layers=layers, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=64, rms_norm_eps=1e-6,
        rope_theta=10000.0, attention_dropout=0.0, pad_token_id=0,
        layer_types=["full_attention"] * layers,
    )
    config._attn_implementation = implementation
    for name, value in changes.items():
        setattr(config, name, value)
    torch.manual_seed(7)
    return PEagleDraftModel(config, mask_token_id=0, num_aux_hidden_states=3, num_depths=depths)


def example():
    generator = torch.Generator().manual_seed(9)
    return dict(
        input_ids=torch.tensor([[2, 4, 6, 8, 7, 5, 3, 1]]),
        hidden_states=torch.randn(1, 8, 48, generator=generator, requires_grad=True),
        document_ids=torch.tensor([[0, 0, 0, 1, 1, 1, -1, -1]]),
        position_ids=torch.tensor([[1, 2, 3, 1, 2, 3, 0, 0]]),
    )


def scalar_layout(docs, depths):
    slots = [(i, 0) for i in range(len(docs))]
    for d in range(1, depths):
        for a in range(len(docs) - d):
            if docs[a] >= 0 and all(v == docs[a] for v in docs[a:a + d + 1]):
                slots.append((a, d))
    return slots


def scalar_mask(slots, docs):
    result = torch.zeros(len(slots), len(slots), dtype=torch.bool)
    for qi, (a, d) in enumerate(slots):
        for ki, (b, e) in enumerate(slots):
            if qi == ki:
                result[qi, ki] = True
            elif docs[a] >= 0 and docs[b] >= 0:
                connected = all(v == docs[a] for v in docs[min(a, b):max(a, b) + 1])
                result[qi, ki] = connected and ((e == 0 and b <= a) or (a == b and e <= d))
    return result


def rms(x, norm):
    dtype = x.dtype
    x = x.float()
    return norm.weight * (x / torch.sqrt(x.square().mean(-1, keepdim=True) + norm.variance_epsilon)).to(dtype)


def linear(x, layer):
    return F.linear(x, layer.weight, layer.bias)


def reference_forward(m, batch):
    """No production layout, attention, rotary, decoder, norm or loss calls."""
    docs = batch["document_ids"][0].tolist()
    slots = scalar_layout(docs, m.num_depths)
    embeds = torch.stack([
        m.embed_tokens.weight[batch["input_ids"][0, a] if d == 0 else m.mask_token_id]
        for a, d in slots
    ])[None]
    features = torch.stack([
        batch["hidden_states"].detach()[0, a] if d == 0 else m.mask_hidden[0, 0]
        for a, d in slots
    ])[None]
    states = linear(features, m.fc)
    mask = scalar_mask(slots, docs)
    positions = torch.tensor([batch["position_ids"][0, a + d] for a, d in slots])
    dim = m.config.head_dim
    theta = m.config.rope_parameters["rope_theta"]
    inv_freq = theta ** (-torch.arange(0, dim, 2).float() / dim)
    angles = positions[:, None] * inv_freq[None]
    angles = torch.cat((angles, angles), -1)[None, None]

    def rotate(x):
        first, second = x.chunk(2, -1)
        return x * angles.cos() + torch.cat((-second, first), -1) * angles.sin()

    for i, layer in enumerate(m.layers):
        residual = states
        if i == 0:
            attention_input = torch.cat((
                rms(embeds, layer.input_layernorm), rms(states, layer.hidden_norm),
            ), -1)
        else:
            attention_input = rms(states, layer.input_layernorm)
        attn = layer.self_attn
        n = len(slots)
        q = linear(attention_input, attn.q_proj).reshape(1, n, m.config.num_attention_heads, dim)
        k = linear(attention_input, attn.k_proj).reshape(1, n, m.config.num_key_value_heads, dim)
        v = linear(attention_input, attn.v_proj).reshape(1, n, m.config.num_key_value_heads, dim)
        q = rotate(rms(q, attn.q_norm).transpose(1, 2))
        k = rotate(rms(k, attn.k_norm).transpose(1, 2))
        groups = m.config.num_attention_heads // m.config.num_key_value_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.transpose(1, 2).repeat_interleave(groups, dim=1)
        scores = q @ k.transpose(-1, -2) / math.sqrt(dim)
        probabilities = scores.masked_fill(~mask, -torch.inf).softmax(-1)
        attended = (probabilities @ v).transpose(1, 2).reshape(1, n, -1)
        states = residual + linear(attended, attn.o_proj)
        normed = rms(states, layer.post_attention_layernorm)
        mlp = layer.mlp
        states = states + linear(F.silu(linear(normed, mlp.gate_proj)) * linear(normed, mlp.up_proj), mlp.down_proj)
    return linear(rms(states, m.norm), m.lm_head), slots


def reference_loss(logits, slots, docs, teacher, loss_mask):
    terms = []
    for i, (a, d) in enumerate(slots):
        if docs[a] >= 0 and loss_mask[0, a + d]:
            t = teacher.detach()[0, a + d].float()
            logp = t - t.logsumexp(-1)
            q = logits[0, i].float()
            logq = q - q.logsumexp(-1)
            terms.append((logp.exp() * (logp - logq)).sum())
    return torch.stack(terms).sum() / (len(terms) + 1e-5)


@pytest.mark.parametrize("docs", [
    [0], [-1], [0, 0, 0, 0], [0, 0, 1, 1],
    [0, 0, -1, -1], [-1, 0, 0, -1], [0, 0, 1, 0, 0], [-1, -1, -1],
])
@pytest.mark.parametrize("depths", [1, 2, 5])
def test_layout_and_attention_match_independent_scalar_rules(docs, depths):
    anchors, depth = peagle_dense_layout(torch.tensor(docs), depths)
    slots = scalar_layout(docs, depths)
    assert list(zip(anchors.tolist(), depth.tolist())) == slots
    actual = peagle_attention_mask(anchors, depth, torch.tensor(docs))
    assert torch.equal(actual, scalar_mask(slots, docs))
    assert actual.any(-1).all()
    for i, (a, _) in enumerate(slots):
        if docs[a] == -1:
            assert actual[i].sum() == 1


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("layers", [1, 2])
@pytest.mark.parametrize("depths", [1, 4])
def test_forward_loss_and_all_parameter_gradients_match_reference(implementation, layers, depths):
    actual = model(implementation=implementation, layers=layers, depths=depths)
    expected = copy.deepcopy(actual)
    batch = example()
    logits, slots = reference_forward(expected, batch)
    output = actual(**batch)
    torch.testing.assert_close(output.logits, logits, rtol=2e-5, atol=2e-6)
    torch.manual_seed(13)
    teacher = torch.randn(1, 8, 23, requires_grad=True)
    loss_mask = torch.tensor([[0, 1, 1, 0, 1, 1, 1, 1]])
    loss, metrics = peagle_kl_loss(output, teacher, loss_mask)
    expected_loss = reference_loss(logits, slots, batch["document_ids"][0].tolist(), teacher, loss_mask)
    torch.testing.assert_close(loss, expected_loss, rtol=2e-5, atol=2e-6)
    loss.backward()
    expected_loss.backward()
    for (name, param), (other_name, other) in zip(actual.named_parameters(), expected.named_parameters()):
        assert name == other_name
        assert param.requires_grad
        assert param.grad is not None, name
        if other.grad is None:
            assert not param.grad.any(), name
        else:
            torch.testing.assert_close(param.grad, other.grad, rtol=3e-4, atol=2e-6, msg=name)
    assert teacher.grad is None
    assert batch["hidden_states"].grad is None
    counts = torch.tensor([
        sum(int(loss_mask[0, a + d]) for a, d in slots
            if d == wanted and batch["document_ids"][0, a] >= 0)
        for wanted in range(depths)
    ]).float()
    assert torch.equal(metrics["count"], counts)
    expected_sums, expected_correct = torch.zeros(depths), torch.zeros(depths)
    for i, (a, d) in enumerate(slots):
        if batch["document_ids"][0, a] >= 0 and loss_mask[0, a + d]:
            t = teacher.detach()[0, a + d].float()
            logp = t - t.logsumexp(-1)
            q = logits.detach()[0, i].float()
            logq = q - q.logsumexp(-1)
            expected_sums[d] += (logp.exp() * (logp - logq)).sum()
            expected_correct[d] += (t.argmax() == q.argmax()).float()
    torch.testing.assert_close(metrics["kl_sum"], expected_sums, rtol=2e-5, atol=2e-6)
    assert torch.equal(metrics["correct"], expected_correct)
    assert not metrics["kl_sum"].requires_grad


def test_future_tokens_other_documents_and_padding_do_not_leak():
    m = model()
    batch = example()
    before = m(**batch)
    changed = {k: v.detach().clone() for k, v in batch.items()}
    changed["input_ids"][0, 1:] = torch.tensor([1, 2, 3, 4, 6, 8, 9])
    changed["hidden_states"][0, 1:] = 100 * torch.randn(7, 48)
    after = m(**changed)
    chain_zero = before.anchor_pos == 0
    torch.testing.assert_close(before.logits[:, chain_zero], after.logits[:, chain_zero], rtol=0, atol=0)

    changed = {k: v.detach().clone() for k, v in batch.items()}
    changed["input_ids"][0, :3] = 1
    changed["hidden_states"][0, :3] = 100
    after = m(**changed)
    second_document = (before.anchor_pos >= 3) & (before.anchor_pos <= 5)
    torch.testing.assert_close(before.logits[:, second_document], after.logits[:, second_document], rtol=0, atol=0)

    changed = {k: v.detach().clone() for k, v in batch.items()}
    changed["input_ids"][0, 6:] = 22
    changed["hidden_states"][0, 6:] = -100
    after = m(**changed)
    torch.testing.assert_close(before.logits[:, before.valid], after.logits[:, after.valid], rtol=0, atol=0)


def test_mask_token_and_hidden_parameters_update_even_when_mask_id_is_padding_id():
    m = model()
    batch = example()
    output = m(**batch)
    torch.manual_seed(4)
    loss, _ = peagle_kl_loss(output, torch.randn(1, 8, 23), torch.ones(1, 8))
    mask_before = m.mask_hidden.detach().clone()
    embed_before = m.embed_tokens.weight[0].detach().clone()
    optimizer = torch.optim.SGD(m.parameters(), lr=0.1)
    loss.backward()
    assert m.mask_hidden.grad.abs().sum() > 0
    assert m.embed_tokens.weight.grad[0].abs().sum() > 0
    optimizer.step()
    assert not torch.equal(mask_before, m.mask_hidden)
    assert not torch.equal(embed_before, m.embed_tokens.weight[0])


@pytest.mark.parametrize("padding", [False, True])
def test_empty_supervision_has_finite_differentiable_zero(padding):
    m = model()
    batch = example()
    if padding:
        batch["document_ids"].fill_(-1)
    output = m(**batch)
    teacher = torch.full((1, 8, 23), torch.nan, requires_grad=True)
    loss, metrics = peagle_kl_loss(output, teacher, torch.zeros(1, 8))
    assert loss.item() == 0 and loss.requires_grad
    assert torch.isfinite(output.logits).all()
    assert not metrics["count"].any()
    loss.backward()
    assert teacher.grad is None
    for param in m.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all() and not param.grad.any()


def test_unsupervised_nonfinite_teacher_values_are_excluded_before_softmax():
    m = model()
    output = m(**example())
    teacher = torch.randn(1, 8, 23)
    mask = torch.tensor([[0, 1, 1, 0, 1, 1, 0, 0]])
    before, _ = peagle_kl_loss(output, teacher, mask)
    teacher[:, ~mask[0].bool()] = torch.nan
    after, _ = peagle_kl_loss(output, teacher, mask)
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_reference_next_token_shift_and_depth_target_alignment():
    m = model(layers=1)
    torch.manual_seed(21)
    raw_ids = torch.tensor([[2, 4, 6, 8, 10]])
    raw_features = torch.randn(1, 5, 48)
    raw_teacher = torch.randn(1, 5, 23)
    output = m(raw_ids[:, 1:], raw_features[:, :-1], torch.zeros(1, 4, dtype=torch.long))
    loss, _ = peagle_kl_loss(output, raw_teacher[:, 1:], torch.ones(1, 4))
    terms = []
    for i, (a, d) in enumerate(zip(output.anchor_pos.tolist(), output.depth.tolist())):
        source_index = a + d + 1
        target = raw_teacher[0, source_index]
        p = target.softmax(-1)
        terms.append((p * (target.log_softmax(-1) - output.logits[0, i].log_softmax(-1))).sum())
    torch.testing.assert_close(loss, torch.stack(terms).sum() / (len(terms) + 1e-5))


def test_positions_reset_per_document_and_support_explicit_offsets():
    m = model()
    batch = example()
    explicit = m(**batch)
    batch.pop("position_ids")
    automatic = m(**batch)
    torch.testing.assert_close(explicit.logits[:, explicit.valid], automatic.logits[:, automatic.valid], rtol=0, atol=0)
    batch["position_ids"] = torch.tensor([[25, 26, 27, 50, 51, 52, 0, 0]])
    expected, _ = reference_forward(m, batch)
    torch.testing.assert_close(m(**batch).logits, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("missing_key", ["mask_hidden", "embed_tokens.weight"])
def test_strict_state_dict_requires_both_learned_masks(missing_key):
    m = model()
    batch = example()
    state = m.state_dict()
    restored = model()
    restored.load_state_dict(state, strict=True)
    torch.testing.assert_close(m(**batch).logits, restored(**batch).logits, rtol=0, atol=0)
    missing = dict(state)
    missing.pop(missing_key)
    with pytest.raises(RuntimeError, match=missing_key):
        restored.load_state_dict(missing, strict=True)


def test_bfloat16_cpu_forward_backward_is_finite():
    m = model().to(torch.bfloat16)
    batch = example()
    batch["hidden_states"] = batch["hidden_states"].detach().to(torch.bfloat16)
    output = m(**batch)
    loss, _ = peagle_kl_loss(output, torch.randn(1, 8, 23), torch.ones(1, 8))
    assert output.logits.dtype == torch.bfloat16 and torch.isfinite(loss)
    loss.backward()
    for param in m.parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all()


@pytest.mark.parametrize("changes", [
    {"attention_dropout": 0.1}, {"layer_types": ["sliding_attention", "full_attention"]},
    {"hidden_act": "relu"}, {"num_hidden_layers": 0},
    {"draft_vocab_size": 12}, {"tie_word_embeddings": True},
    {"norm_before_residual": True}, {"norm_before_fc": True}, {"fc_norm": True},
    {"num_key_value_heads": 3}, {"head_dim": 7},
])
def test_unsupported_model_recipe_fails_closed(changes):
    with pytest.raises(ValueError, match="P-EAGLE"):
        model(**changes)


def test_other_target_and_attention_backend_fail_closed():
    with pytest.raises(ValueError, match="Qwen3"):
        PEagleDraftModel(LlamaConfig(), mask_token_id=0)
    with pytest.raises(ValueError, match="eager or sdpa"):
        model(implementation="flash_attention_2")


@pytest.mark.parametrize("depths", [0, -1, 1.2, True])
def test_invalid_depths_fail_closed(depths):
    with pytest.raises(ValueError, match="positive integer"):
        peagle_dense_layout(torch.tensor([0, 0]), depths)


@pytest.mark.parametrize("kind", ["batch", "feature", "token", "position", "document"])
def test_malformed_inputs_fail_closed(kind):
    m = model()
    batch = example()
    if kind == "batch":
        batch["input_ids"] = batch["input_ids"].repeat(2, 1)
    elif kind == "feature":
        batch["hidden_states"] = batch["hidden_states"][..., :-1]
    elif kind == "token":
        batch["input_ids"][0, 0] = 23
    elif kind == "position":
        batch["position_ids"][0, 0] = -1
    else:
        batch["document_ids"][0, 0] = -2
    with pytest.raises(ValueError, match="P-EAGLE"):
        m(**batch)


def test_wrong_loss_mask_and_vocabulary_fail_closed():
    output = model()(**example())
    with pytest.raises(ValueError, match="binary"):
        peagle_kl_loss(output, torch.randn(1, 8, 23), torch.full((1, 8), 0.5))
    with pytest.raises(ValueError, match="full vocabulary"):
        peagle_kl_loss(output, torch.randn(1, 8, 22), torch.ones(1, 8))
