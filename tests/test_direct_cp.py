# Copyright (C) 2026 Tencent. All rights reserved.
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

# Adapted from the retained AngelSpec source snapshot 0966eb139d1c4054767e4a37bfc8a33db5989a72
# tests/test_direct_cp.py. ASpec adjusts namespace bindings and regression coverage.

# Adapted from AngelSpec tests/test_direct_cp.py (Apache-2.0).

import inspect
import math
import os

import pytest
import torch

from aspec.models.ops.direct_cp import (
    DirectCPMetadata,
    build_context_positions,
    direct_cp_attention,
)
from aspec.models.ops.fa4_sparse import _direct_cp_tile_masks


@pytest.mark.parametrize("batch", [1, 2])
def test_cudnn_output_keeps_token_major_storage_through_the_consumer(monkeypatch, batch):
    """Layout-only oracle; actual cuDNN/collectives have separate GPU checks."""
    import aspec.models.ops.cudnn_flex_attention as adapter
    import aspec.models.ops.direct_cp as cp_module

    emitted = []

    def backend(q, k, v, **kwargs):
        result = q * 2
        emitted.append(result)
        return result, torch.zeros(q.shape[1], q.shape[0])

    monkeypatch.setattr(adapter, "cudnn_flex_attention", backend)
    monkeypatch.setattr(cp_module, "_build_cudnn_thd_mask", lambda *args, **kwargs: object())
    source = torch.randn(batch, 4, 4, 8, requires_grad=True)  # BQHD
    query = source.transpose(1, 2)
    key = torch.randn(batch, 2, 7, 8)
    value = torch.randn_like(key)
    keep = torch.ones(batch, 2, dtype=torch.bool)
    keep[-1, -1] = False
    metadata = DirectCPMetadata(
        anchor_positions=torch.tensor([[0, 2]]).expand(batch, -1), block_keep_mask=keep,
        context_position_ids=torch.arange(3).expand(batch, -1), block_size=2,
        cp_rank=0, cp_size=1, group=None, sequence_length=3,
        backend="cudnn_flex_attention",
    )
    partial, _ = cp_module._cudnn_partial(query, key, value, metadata, 1.0)
    assert partial.data_ptr() == emitted[-1].data_ptr()
    assert partial.transpose(1, 2).is_contiguous()
    output, _ = cp_module.direct_cp_attention_from_kv(query, key, value, metadata, 1.0)
    token_major = output.transpose(1, 2).contiguous()
    assert token_major.data_ptr() == output.data_ptr()
    positions = (metadata.anchor_positions[..., None] + torch.arange(2)).flatten(1)
    token_keep = keep.repeat_interleave(2, dim=1) & (positions < 3)
    expected = torch.where(token_keep[..., None, None], source * 2, 0)
    torch.testing.assert_close(token_major, expected, rtol=0, atol=0)
    token_major.square().sum().backward()
    torch.testing.assert_close(source.grad, expected.detach() * 4, rtol=0, atol=0)


@pytest.mark.parametrize(("context_window", "query_window", "selection_keys"), [
    (None, None, 260), (64, None, 68), (None, 32, 36), (64, 32, 36), (1, 2, 5),
])
def test_cudnn_gqa_selection_respects_window_visibility(monkeypatch, context_window, query_window, selection_keys):
    import aspec.models.ops.cudnn_flex_attention as adapter
    import aspec.models.ops.direct_cp as cp_module

    monkeypatch.setattr(adapter, "cudnn_flex_attention",
                        lambda q, k, v, **kwargs: (q, torch.zeros(q.shape[1], q.shape[0])))
    metadata = DirectCPMetadata(anchor_positions=torch.tensor([[12, 74]]),
                                block_keep_mask=torch.ones(1, 2, dtype=torch.bool),
                                context_position_ids=torch.arange(256)[None],
                                block_size=2, cp_rank=0, cp_size=1, group=None,
                                backend="cudnn_flex_attention", context_window=context_window,
                                query_window=query_window)
    cp_module._cudnn_partial(torch.randn(1, 4, 4, 128), torch.randn(1, 2, 260, 128),
                             torch.randn(1, 2, 260, 128), metadata, 128 ** -0.5)
    assert metadata.attention_observation["gqa_selection_key_tokens"] == selection_keys
    assert metadata.attention_observation["gqa_layout"] == "native"


def test_context_positions_cover_sequence_once_and_balance_causal_work():
    total = 32
    contiguous = [
        build_context_positions(total, rank, 4, layout="contiguous")
        for rank in range(4)
    ]
    zigzag = [
        build_context_positions(total, rank, 4, layout="zigzag")
        for rank in range(4)
    ]

    assert torch.equal(torch.cat(contiguous).sort().values, torch.arange(total))
    assert torch.equal(torch.cat(zigzag).sort().values, torch.arange(total))
    assert torch.equal(zigzag[0], torch.tensor([0, 1, 2, 3, 28, 29, 30, 31]))
    assert torch.equal(zigzag[3], torch.tensor([12, 13, 14, 15, 16, 17, 18, 19]))

    contiguous_work = [int(positions.sum()) for positions in contiguous]
    zigzag_work = [int(positions.sum()) for positions in zigzag]
    assert max(zigzag_work) - min(zigzag_work) < max(contiguous_work) - min(
        contiguous_work
    )


def _dense_reference(
    query,
    context_key,
    context_value,
    draft_key,
    draft_value,
    anchors,
    keep,
    block_size,
    context_doc_ids=None,
    anchor_doc_ids=None,
):
    key = torch.cat((context_key, draft_key), dim=2).repeat_interleave(
        query.shape[1] // context_key.shape[1], dim=1
    )
    value = torch.cat((context_value, draft_value), dim=2).repeat_interleave(
        query.shape[1] // context_value.shape[1], dim=1
    )
    query_length = query.shape[2]
    context_length = context_key.shape[2]
    query_blocks = torch.arange(query_length) // block_size
    context_visible = (
        torch.arange(context_length)[None, None, :] < anchors[:, query_blocks, None]
    )
    if context_doc_ids is not None:
        query_docs = anchor_doc_ids[:, query_blocks]
        context_visible &= (query_docs[:, :, None] >= 0) & (
            context_doc_ids[:, None, :] == query_docs[:, :, None]
        )
    draft_visible = query_blocks[:, None] == query_blocks[None, :]
    visible = torch.cat(
        (context_visible, draft_visible[None].expand(query.shape[0], -1, -1)),
        dim=-1,
    )
    visible &= keep[:, query_blocks, None]
    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1])
    scores = scores.masked_fill(~visible[:, None], -1.0e30)
    lse = torch.logsumexp(scores, dim=-1)
    return torch.matmul(torch.softmax(scores, dim=-1), value), lse


@pytest.mark.parametrize("layout", ["contiguous", "zigzag"])
def test_cp1_math_matches_dense_gqa_forward_and_backward(layout):
    torch.manual_seed(17)
    dtype = torch.float64
    tensors = [
        torch.randn(*shape, dtype=dtype, requires_grad=True)
        for shape in (
            (1, 4, 8, 16),
            (1, 2, 16, 16),
            (1, 2, 16, 16),
            (1, 2, 8, 16),
            (1, 2, 8, 16),
        )
    ]
    query, context_key, context_value, draft_key, draft_value = tensors
    anchors = torch.tensor([[2, 6, 10, 14]])
    keep = torch.tensor([[True, True, False, True]])
    metadata = DirectCPMetadata(
        anchor_positions=anchors,
        block_keep_mask=keep,
        context_position_ids=torch.arange(16).view(1, -1),
        block_size=2,
        cp_rank=0,
        cp_size=1,
        group=None,
        sequence_length=16,
        backend="math",
        layout=layout,
    )

    actual, actual_lse = direct_cp_attention(*tensors, metadata)
    reference_inputs = [
        tensor.detach().clone().requires_grad_(True) for tensor in tensors
    ]
    expected, expected_lse = _dense_reference(
        *reference_inputs, anchors, keep, metadata.block_size
    )
    token_keep = keep.repeat_interleave(metadata.block_size, dim=1)
    expected = torch.where(
        token_keep[:, None, :, None], expected, torch.zeros_like(expected)
    )
    expected_lse = torch.where(
        token_keep[:, None], expected_lse, torch.full_like(expected_lse, -1.0e30)
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_lse, expected_lse)

    output_grad = torch.randn_like(actual)
    (actual * output_grad).sum().backward()
    (expected * output_grad).sum().backward()
    for tensor, reference in zip(tensors, reference_inputs, strict=True):
        torch.testing.assert_close(tensor.grad, reference.grad)


def test_cp1_math_packed_documents_match_dense_forward_and_backward():
    torch.manual_seed(29)
    dtype = torch.float64
    tensors = [
        torch.randn(*shape, dtype=dtype, requires_grad=True)
        for shape in (
            (1, 4, 8, 16),
            (1, 2, 16, 16),
            (1, 2, 16, 16),
            (1, 2, 8, 16),
            (1, 2, 8, 16),
        )
    ]
    anchors = torch.tensor([[2, 6, 10, 14]])
    keep = torch.ones_like(anchors, dtype=torch.bool)
    context_doc_ids = torch.tensor([[0] * 8 + [1] * 8])
    anchor_doc_ids = torch.tensor([[0, 0, 1, 1]])
    metadata = DirectCPMetadata(
        anchor_positions=anchors,
        block_keep_mask=keep,
        context_position_ids=torch.arange(16).view(1, -1),
        block_size=2,
        cp_rank=0,
        cp_size=1,
        group=None,
        context_doc_ids=context_doc_ids,
        anchor_doc_ids=anchor_doc_ids,
        sequence_length=16,
        backend="math",
    )

    actual, actual_lse = direct_cp_attention(*tensors, metadata)
    reference_inputs = [
        tensor.detach().clone().requires_grad_(True) for tensor in tensors
    ]
    expected, expected_lse = _dense_reference(
        *reference_inputs,
        anchors,
        keep,
        metadata.block_size,
        context_doc_ids,
        anchor_doc_ids,
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_lse, expected_lse)

    output_grad = torch.randn_like(actual)
    (actual * output_grad).sum().backward()
    (expected * output_grad).sum().backward()
    for tensor, reference in zip(tensors, reference_inputs, strict=True):
        torch.testing.assert_close(tensor.grad, reference.grad)


@pytest.mark.skipif(
    os.getenv("ASPEC_RUN_GPU_INTEGRATION") != "1",
    reason="set ASPEC_RUN_GPU_INTEGRATION=1 to validate the real FA4 kernel",
)
def test_cp1_fa4_matches_math_forward_and_backward_on_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("FA4 parity requires an SM100+ GPU")
    pytest.importorskip("flash_attn.cute.interface")

    torch.manual_seed(31)
    device = torch.device("cuda")
    shapes = (
        (1, 4, 28, 128),
        (1, 2, 512, 128),
        (1, 2, 512, 128),
        (1, 2, 28, 128),
        (1, 2, 28, 128),
    )
    fa4_inputs = [
        torch.randn(*shape, device=device, dtype=torch.bfloat16, requires_grad=True)
        for shape in shapes
    ]
    math_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in fa4_inputs]
    anchors = torch.tensor([[64, 192, 320, 448]], device=device)
    keep = torch.ones_like(anchors, dtype=torch.bool)
    context_positions = torch.arange(512, device=device).view(1, -1)

    def metadata(backend):
        return DirectCPMetadata(
            anchor_positions=anchors,
            block_keep_mask=keep,
            context_position_ids=context_positions,
            block_size=7,
            cp_rank=0,
            cp_size=1,
            group=None,
            sequence_length=512,
            backend=backend,
        )

    fa4_output, fa4_lse = direct_cp_attention(*fa4_inputs, metadata("fa4"))
    math_output, math_lse = direct_cp_attention(*math_inputs, metadata("math"))
    torch.testing.assert_close(fa4_output, math_output, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(fa4_lse, math_lse, rtol=5e-2, atol=5e-2)

    output_grad = torch.randn_like(fa4_output)
    (fa4_output * output_grad).sum().backward()
    (math_output * output_grad).sum().backward()
    for fa4_tensor, math_tensor in zip(fa4_inputs, math_inputs, strict=True):
        torch.testing.assert_close(
            fa4_tensor.grad,
            math_tensor.grad,
            rtol=8e-2,
            atol=8e-2,
        )


def test_direct_cp_sparse_tiles_cover_visible_tokens_and_only_mark_full_tiles():
    anchors = torch.tensor([[2, 6, 10, 14]], dtype=torch.int32)
    keep = torch.tensor([[True, True, False, True]])
    query_blocks = torch.arange(4, dtype=torch.int32).repeat_interleave(2).view(1, -1)
    context_positions = torch.arange(16, dtype=torch.int32).view(1, -1)
    draft_blocks = query_blocks.clone()
    query_keep = keep.repeat_interleave(2, dim=1)
    key_keep = torch.cat((torch.ones(1, 16, dtype=torch.bool), query_keep), dim=1)

    partial, full = _direct_cp_tile_masks(
        anchor_positions=anchors,
        query_blocks=query_blocks,
        context_positions=context_positions,
        draft_blocks=draft_blocks,
        query_token_keep=query_keep,
        key_token_keep=key_keep,
        tile_m=4,
        tile_n=4,
    )

    query_anchor = torch.gather(anchors, 1, query_blocks.long())
    visible_context = context_positions[:, None, :] < query_anchor[:, :, None]
    visible_draft = query_blocks[:, :, None] == draft_blocks[:, None, :]
    visible = torch.cat((visible_context, visible_draft), dim=-1)
    visible &= query_keep[:, :, None] & key_keep[:, None, :]
    visible_tiles = visible.view(1, 2, 4, 6, 4).permute(0, 1, 3, 2, 4)
    expected_any = visible_tiles.any(dim=(-1, -2)).unsqueeze(1)
    expected_all = visible_tiles.all(dim=(-1, -2)).unsqueeze(1)

    assert torch.all(expected_any <= (partial | full))
    assert torch.all(full <= expected_all)


@pytest.fixture(scope="module")
def cold_dispatch_caches():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("requires SM90+ CUDA")
    from aspec.compat.fa4_cutlass import (
        configure_fa4_cutlass_compilation,
        install_ampere_helpers_shim,
    )

    install_ampere_helpers_shim()
    configure_fa4_cutlass_compilation()
    import cutlass.cute as cute
    from flash_attn.cute import cache_utils, interface

    # Start cold, then retain entries across semantic variants as well as
    # lengths. A bad hash that aliases two different masks must fail numerics.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(interface._flash_attn_fwd, "compile_cache", cache_utils.JITCache())
        patch.setattr(interface._flash_attn_bwd, "compile_cache", cache_utils.JITCache())
        yield cute


@pytest.mark.skipif(
    os.getenv("ASPEC_RUN_GPU_INTEGRATION") != "1",
    reason="set ASPEC_RUN_GPU_INTEGRATION=1 to exercise real FA4 compilation",
)
@pytest.mark.parametrize("context_window,query_window,causal", [
    (None, None, False), (97, None, False), (None, 3, False), (None, 3, True),
])
def test_variable_lengths_reuse_mask_kernels_and_match_math(monkeypatch, cold_dispatch_caches, context_window, query_window, causal):
    cute = cold_dispatch_caches
    original_compile = cute.compile
    compiled_families = set()

    def observe_compile(*args, **kwargs):
        name = type(args[0]).__name__
        if name.startswith(("FlashAttentionForwardSm", "FlashAttentionBackwardSm")):
            caller = inspect.currentframe().f_back.f_locals
            # FA4 deliberately separates singleton block strides in backward.
            # Preserve its ABI/tile families, but never create another kernel
            # merely because the modifier captured a different row length.
            mask_hash = caller["mask_mod_hash"]
            family = (name, tuple("<mask>" if isinstance(value, str) and value == mask_hash else value
                                  for value in caller["compile_key"]))
            assert family not in compiled_families, "row geometry recompiled an existing FA4 dispatch family"
            compiled_families.add(family)
        return original_compile(*args, **kwargs)

    monkeypatch.setattr(cute, "compile", observe_compile)
    for index, (context, blocks) in enumerate([(129, 4), (257, 9), (641, 13), (257, 9)]):
        torch.manual_seed(31 + index)
        queries = blocks * 7
        shapes = [(1, 4, queries, 128), (1, 1, context, 128), (1, 1, context, 128),
                  (1, 1, queries, 128), (1, 1, queries, 128)]
        actual_inputs = [torch.randn(*shape, device="cuda", dtype=torch.bfloat16, requires_grad=True) for shape in shapes]
        reference_inputs = [value.detach().clone().requires_grad_() for value in actual_inputs]
        anchors = torch.randint(context, (1, blocks), device="cuda").sort().values
        anchors[0, 0], anchors[0, -1] = 0, context - 1
        keep = torch.ones_like(anchors, dtype=torch.bool)
        keep[0, -2] = False

        def metadata(backend):
            return DirectCPMetadata(
                anchor_positions=anchors, block_keep_mask=keep,
                context_position_ids=torch.arange(context, device="cuda")[None],
                block_size=7, cp_rank=0, cp_size=1, group=None, sequence_length=context,
                backend=backend, fa4_sparse=True, context_window=context_window,
                query_window=query_window, draft_is_causal=causal,
            )

        actual, actual_lse = direct_cp_attention(*actual_inputs, metadata("fa4"))
        reference, reference_lse = direct_cp_attention(*reference_inputs, metadata("math"))
        torch.testing.assert_close(actual, reference, rtol=5e-2, atol=5e-2)
        torch.testing.assert_close(actual_lse, reference_lse, rtol=5e-2, atol=5e-2)
        upstream = torch.randn_like(actual)
        (actual * upstream).sum().backward()
        (reference * upstream).sum().backward()
        for value, expected in zip(actual_inputs, reference_inputs, strict=True):
            torch.testing.assert_close(value.grad, expected.grad, rtol=8e-2, atol=8e-2)
