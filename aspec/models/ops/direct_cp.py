# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Adapted for ASpec from Megatron-LM's DSpark context-parallel attention.
# Redistribution is governed by licenses/NVIDIA-Megatron-LM-BSD-3-Clause.txt.

"""Direct context-parallel attention for DFlash/DSpark batches."""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor

MASK_VALUE = -1.0e30


def build_context_positions(
    total_positions: int,
    cp_rank: int,
    cp_size: int,
    *,
    layout: str = "contiguous",
    alignment: int = 1,
) -> Tensor:
    """Return the global positions owned by one context-parallel rank."""

    if total_positions < 0 or cp_size <= 0 or not 0 <= cp_rank < cp_size:
        raise ValueError("invalid context-parallel position layout")
    if layout not in {"contiguous", "zigzag"}:
        raise ValueError(f"unsupported context-parallel layout: {layout}")
    if layout == "contiguous" or cp_size == 1:
        from aspec.inference.context import context_shard_bounds

        start, end = context_shard_bounds(
            total_positions, cp_rank, cp_size, alignment=alignment
        )
        return torch.arange(start, end, dtype=torch.long)
    if alignment != 1:
        raise ValueError("zigzag context ownership does not support aligned groups")

    chunks = 2 * cp_size
    owned = []
    for chunk_rank in (cp_rank, chunks - 1 - cp_rank):
        start = total_positions * chunk_rank // chunks
        end = total_positions * (chunk_rank + 1) // chunks
        owned.append(torch.arange(start, end, dtype=torch.long))
    return torch.cat(owned)


def gather_owner_sharded_rows(
    hidden_states: Tensor,
    context_position_ids: Tensor,
    query_positions: Tensor,
    *,
    global_sequence_length: int,
    group: dist.ProcessGroup,
) -> Tensor:
    """Gather short requested rows while keeping the long sequence CP-sharded."""

    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must be [batch, local_sequence, hidden]")
    if context_position_ids.ndim != 2 or query_positions.ndim != 2:
        raise ValueError("context_position_ids and query_positions must be [batch, sequence]")
    if context_position_ids.shape[0] != query_positions.shape[0]:
        raise ValueError("context and query batches must match")
    if hidden_states.shape[:2] != context_position_ids.shape:
        raise ValueError("hidden_states and context_position_ids shapes do not match")
    if global_sequence_length <= 0:
        raise ValueError("global_sequence_length must be positive")
    if context_position_ids.shape[1] and not torch.equal(
        context_position_ids, context_position_ids[:1].expand_as(context_position_ids)
    ):
        raise ValueError("all samples must use the same context ownership")

    context_positions = context_position_ids[0].to(device=hidden_states.device, dtype=torch.long)
    global_to_local = torch.full(
        (global_sequence_length,), -1, dtype=torch.long, device=hidden_states.device
    )
    global_to_local[context_positions] = torch.arange(
        context_positions.numel(), device=hidden_states.device
    )
    # CP ranks supervise different anchors. Reduce the SAME requested rows on
    # every owner, then return this rank's slice, not unrelated labels.
    cp_size = dist.get_world_size(group) if dist.is_initialized() else 1
    cp_rank = dist.get_rank(group) if cp_size > 1 else 0
    local_query_count = query_positions.shape[1]
    if cp_size > 1:
        requests = [torch.empty_like(query_positions) for _ in range(cp_size)]
        dist.all_gather(requests, query_positions.contiguous(), group=group)
        query_positions = torch.cat(requests, dim=1)
    valid = (query_positions >= 0) & (query_positions < global_sequence_length)
    safe_global = query_positions.clamp(0, global_sequence_length - 1)
    local_indices = global_to_local.index_select(0, safe_global.reshape(-1)).view_as(
        query_positions
    )
    owned = valid & (local_indices >= 0)
    if hidden_states.shape[1]:
        partial = torch.gather(
            hidden_states, 1,
            local_indices.clamp_min(0).unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)),
        )
    else:
        partial = hidden_states.new_zeros(
            hidden_states.shape[0], query_positions.shape[1], hidden_states.shape[-1]
        ) + hidden_states.sum()
    partial = torch.where(owned.unsqueeze(-1), partial, torch.zeros_like(partial))
    if cp_size > 1:
        # The same bounded row gather is used by CSA's cross-shard compression
        # halo.  Keep it differentiable so an owner receives gradients for the
        # few rows consumed by its neighbour; a plain in-place all_reduce would
        # silently sever that path.
        from torch.distributed.nn.functional import all_reduce

        partial = all_reduce(partial, group=group)
    return partial[:, cp_rank * local_query_count : (cp_rank + 1) * local_query_count].contiguous()


class _ForwardSparseCache:
    """One forward's integer sparse metadata, shared across FSDP layer copies.

    FSDP recursively copies dataclasses and containers when casting inputs.
    This opaque owner survives that traversal; it holds no model activations.
    A fresh DirectCPMetadata owns a fresh cache, never a cross-step mask.
    """

    def __init__(self):
        self._entries = {}
        # Integer mask inputs share exactly the same per-forward lifetime.
        self.thd_inputs = {}

    def get(self, key):
        return self._entries.get(key)

    def __setitem__(self, key, value):
        self._entries[key] = value

    def values(self):
        return self._entries.values()


@dataclass
class DirectCPMetadata:
    """Replicated mask metadata plus this rank's context and query ownership."""

    anchor_positions: Tensor
    block_keep_mask: Tensor
    context_position_ids: Tensor
    block_size: int
    cp_rank: int
    cp_size: int
    group: Optional[dist.ProcessGroup]
    context_doc_ids: Optional[Tensor] = None
    anchor_doc_ids: Optional[Tensor] = None
    context_rope_position_ids: Optional[Tensor] = None
    sequence_length: Optional[int] = None
    backend: str = "fa4"
    layout: str = "contiguous"
    fa4_sparse: bool = True
    context_window: Optional[int] = None
    query_window: Optional[int] = None  # DFlash2: strict distance from each query
    draft_is_causal: bool = False
    # FSDP2 rewrites tensor-bearing kwargs with dataclasses.replace(), which
    # requires every dataclass field it sees to be accepted by __init__.
    _fa4_cache: _ForwardSparseCache = field(default_factory=_ForwardSparseCache, repr=False)
    attention_observation: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.anchor_positions.ndim != 2 or self.block_keep_mask.ndim != 2:
            raise ValueError("anchor_positions and block_keep_mask must be [batch, blocks]")
        if self.anchor_positions.shape != self.block_keep_mask.shape:
            raise ValueError("anchor_positions and block_keep_mask shapes must match")
        if self.context_position_ids.ndim != 2:
            raise ValueError("context_position_ids must be [batch, local_context]")
        if self.context_position_ids.shape[0] != self.anchor_positions.shape[0]:
            raise ValueError("context and anchor batches must match")
        if (self.context_doc_ids is None) != (self.anchor_doc_ids is None):
            raise ValueError("context_doc_ids and anchor_doc_ids must be provided together")
        if self.context_doc_ids is not None:
            if self.context_doc_ids.shape != self.context_position_ids.shape:
                raise ValueError("context_doc_ids must match context_position_ids")
            if self.anchor_doc_ids.shape != self.anchor_positions.shape:
                raise ValueError("anchor_doc_ids must match anchor_positions")
        if (
            self.context_rope_position_ids is not None
            and self.context_rope_position_ids.shape != self.context_position_ids.shape
        ):
            raise ValueError(
                "context_rope_position_ids must match context_position_ids"
            )
        if self.context_window is not None and (
            isinstance(self.context_window, bool) or not isinstance(self.context_window, int)
            or self.context_window <= 0
        ):
            raise ValueError("context_window must be a positive integer or null")
        if self.query_window is not None and (
            isinstance(self.query_window, bool) or not isinstance(self.query_window, int)
            or self.query_window <= 0
        ):
            raise ValueError("query_window must be a positive integer or null")
        if not isinstance(self.draft_is_causal, bool):
            raise ValueError("draft_is_causal must be boolean")
        if self.block_size <= 0 or self.cp_size <= 0:
            raise ValueError("block_size and cp_size must be positive")
        if self.sequence_length is not None and self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if not 0 <= self.cp_rank < self.cp_size:
            raise ValueError("invalid CP rank")
        if self.layout not in {"contiguous", "zigzag"}:
            raise ValueError(f"unsupported context-parallel layout: {self.layout}")
        layout_divisor = self.cp_size if self.layout == "contiguous" else 2 * self.cp_size
        if self.cp_size == 1:
            layout_divisor = 1
        if self.num_blocks % layout_divisor != 0:
            raise ValueError(
                f"direct CP requires num_blocks divisible by {layout_divisor} for {self.layout}"
            )
        if self.backend not in {"auto", "fa4", "cudnn_flex_attention", "math"}:
            raise ValueError(f"unsupported direct CP backend: {self.backend}")
        if self.cp_size > 1 and self.group is None:
            raise ValueError("a process group is required when cp_size > 1")
        if self.context_position_ids.shape[1] > 0 and not torch.equal(
            self.context_position_ids,
            self.context_position_ids[:1].expand_as(self.context_position_ids),
        ):
            raise ValueError("all samples must use the same context ownership")

    @property
    def num_blocks(self) -> int:
        return self.anchor_positions.shape[1]

    @property
    def blocks_per_rank(self) -> int:
        return self.num_blocks // self.cp_size

    @property
    def queries_per_rank(self) -> int:
        return self.blocks_per_rank * self.block_size

    @property
    def context_length(self) -> int:
        return self.context_position_ids.shape[1]

    @property
    def local_block_ids(self) -> Tensor:
        return build_context_positions(
            self.num_blocks, self.cp_rank, self.cp_size, layout=self.layout
        )

    @property
    def collective_block_ids(self) -> Tensor:
        return torch.cat(
            [
                build_context_positions(self.num_blocks, rank, self.cp_size, layout=self.layout)
                for rank in range(self.cp_size)
            ]
        )


def direct_cp_attention_observation(metadata: DirectCPMetadata) -> dict:
    """Return one forward-scoped observation shared across metadata copies.

    FSDP2 may replace tensor-bearing dataclasses before invoking a layer.  The
    opaque sparse cache deliberately survives that traversal, so it is also the
    stable owner for kernel observations that Plan/Evidence reads afterwards.
    """

    key = (
        "attention_observation",
        metadata.backend,
        metadata.context_window,
        metadata.query_window,
        metadata.draft_is_causal,
    )
    observation = metadata._fa4_cache.get(key)
    if observation is None:
        observation = metadata.attention_observation
        metadata._fa4_cache[key] = observation
    return observation


class _AllGatherQuery(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: Tensor, group) -> Tensor:
        world_size = dist.get_world_size(group)
        query_first = tensor.movedim(2, 0).contiguous()
        gathered = torch.empty(
            (query_first.shape[0] * world_size, *query_first.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        dist.all_gather_into_tensor(gathered, query_first, group=group)
        ctx.group = group
        ctx.local_queries = query_first.shape[0]
        return gathered.movedim(0, 2).contiguous()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        grad_query_first = grad_output.movedim(2, 0).contiguous()
        grad_local = torch.empty(
            (ctx.local_queries, *grad_query_first.shape[1:]),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        dist.reduce_scatter_tensor(
            grad_local,
            grad_query_first,
            op=dist.ReduceOp.SUM,
            group=ctx.group,
        )
        return grad_local.movedim(0, 2).contiguous(), None


class _AllToAllQueryParts(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: Tensor, group) -> Tensor:
        world_size = dist.get_world_size(group)
        if tensor.shape[2] % world_size:
            raise ValueError("query rows must be divisible by CP size")
        queries_per_rank = tensor.shape[2] // world_size
        send = tensor.unflatten(2, (world_size, queries_per_rank)).movedim(2, 0).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=group)
        ctx.group = group
        return recv

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        # empty_like preserves grad_output's strides. Residual connections can
        # produce a non-contiguous gradient view, while NCCL all_to_all requires
        # both buffers to be contiguous.
        grad_send = torch.empty(
            grad_output.shape,
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        dist.all_to_all_single(grad_send, grad_output.contiguous(), group=ctx.group)
        return grad_send.movedim(0, 2).flatten(2, 3).contiguous(), None


def all_gather_query_rows(tensor: Tensor, *, group) -> Tensor:
    """Differentiably gather token-major query rows inside one CP group."""

    if tensor.ndim != 4:
        raise ValueError("CP query rows must be [batch, query, heads, dim]")
    return _AllGatherQuery.apply(tensor.permute(0, 2, 1, 3), group).permute(
        0, 2, 1, 3
    )


def all_to_all_query_partials(tensor: Tensor, *, group) -> Tensor:
    """Route global-query partials back to their query-owning CP rank."""

    return _AllToAllQueryParts.apply(tensor, group)


def merge_attention_partials(
    outputs: Tensor, lse_parts: Tensor
) -> tuple[Tensor, Tensor]:
    """Merge independently normalized attention partials exactly."""

    return _merge_partials(outputs, lse_parts)


def _token_keep(metadata: DirectCPMetadata, block_ids: Tensor, device) -> Tensor:
    anchor_keep = metadata.block_keep_mask.to(device).index_select(1, block_ids)
    if metadata.sequence_length is None:
        return anchor_keep.repeat_interleave(metadata.block_size, dim=1)
    anchors = metadata.anchor_positions.to(device).index_select(1, block_ids)
    offsets = torch.arange(metadata.block_size, device=device).view(1, 1, -1)
    return (
        anchor_keep.unsqueeze(-1)
        & (anchors.unsqueeze(-1) + offsets < metadata.sequence_length)
    ).flatten(1)


def _visibility(metadata: DirectCPMetadata, device) -> Tensor:
    collective_block_ids = metadata.collective_block_ids.to(device)
    query_blocks = collective_block_ids.repeat_interleave(metadata.block_size)
    anchors = metadata.anchor_positions.to(device)[:, query_blocks]
    query_keep = _token_keep(metadata, collective_block_ids, device)
    context_positions = metadata.context_position_ids.to(device)
    visible_context = context_positions[:, None, :] < anchors[:, :, None]
    if metadata.context_window is not None:
        visible_context &= context_positions[:, None, :] >= anchors[:, :, None] - metadata.context_window
    if metadata.context_doc_ids is not None:
        anchor_docs = metadata.anchor_doc_ids.to(device)[:, query_blocks]
        context_docs = metadata.context_doc_ids.to(device)
        visible_context &= (
            (anchor_docs[:, :, None] >= 0)
            & (context_docs[:, None, :] == anchor_docs[:, :, None])
        )
    offsets = torch.arange(metadata.block_size, device=device)
    query_offsets = offsets.repeat(collective_block_ids.numel())
    query_positions = anchors + query_offsets
    if metadata.query_window is not None:
        visible_context &= query_positions[:, :, None] - context_positions[:, None, :] < metadata.query_window
    local_block_ids = metadata.local_block_ids.to(device)
    draft_blocks = local_block_ids.repeat_interleave(metadata.block_size)
    visible_draft = (
        draft_blocks[None, None, :] == query_blocks[None, :, None]
    ).expand(metadata.anchor_positions.shape[0], -1, -1)
    draft_offsets = offsets.repeat(local_block_ids.numel())
    if metadata.draft_is_causal:
        visible_draft = visible_draft & (draft_offsets[None, None, :] <= query_offsets[None, :, None])
    if metadata.query_window is not None:
        visible_draft = visible_draft & ((query_offsets[:, None] - draft_offsets[None, :]).abs()
                                         < metadata.query_window)[None]
    key_keep = torch.cat(
        (
            torch.ones_like(context_positions, dtype=torch.bool),
            _token_keep(metadata, local_block_ids, device),
        ),
        dim=1,
    )
    return (
        torch.cat((visible_context, visible_draft), dim=-1)
        & query_keep[:, :, None]
        & key_keep[:, None, :]
    )


def _expand_kv_heads(query: Tensor, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
    query_heads = query.shape[1]
    key_heads = key.shape[1]
    if value.shape[1] != key_heads or query_heads % key_heads:
        raise ValueError("query heads must be divisible by the matching KV head count")
    heads_per_group = query_heads // key_heads
    if heads_per_group == 1:
        return key, value
    return (
        key.repeat_interleave(heads_per_group, dim=1),
        value.repeat_interleave(heads_per_group, dim=1),
    )


def _math_partial(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    metadata: DirectCPMetadata,
    scale: float,
) -> tuple[Tensor, Tensor]:
    key, value = _expand_kv_heads(query, key, value)
    stat_dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    scores = torch.matmul(query.to(stat_dtype), key.to(stat_dtype).transpose(-1, -2)) * scale
    visibility = _visibility(metadata, query.device)
    scores = torch.where(visibility[:, None], scores, torch.full_like(scores, MASK_VALUE))
    lse = torch.logsumexp(scores, dim=-1)
    probabilities = torch.exp(scores - lse.unsqueeze(-1))
    output = torch.matmul(probabilities, value.to(stat_dtype))
    return output, lse


def _merge_partials(outputs: Tensor, lse_parts: Tensor) -> tuple[Tensor, Tensor]:
    has_keys = torch.isfinite(lse_parts).any(dim=0)
    safe_parts = torch.where(has_keys.unsqueeze(0), lse_parts, torch.zeros_like(lse_parts))
    safe_lse = torch.logsumexp(safe_parts, dim=0)
    lse = torch.where(has_keys, safe_lse, torch.full_like(safe_lse, MASK_VALUE))
    weights = torch.exp(lse_parts - safe_lse.unsqueeze(0))
    weights = torch.where(has_keys.unsqueeze(0), weights, torch.zeros_like(weights))
    return (weights.unsqueeze(-1) * outputs).sum(dim=0), lse


def _apply_block_keep(
    output: Tensor, lse: Tensor, metadata: DirectCPMetadata
) -> tuple[Tensor, Tensor]:
    local_block_ids = metadata.local_block_ids.to(metadata.block_keep_mask.device)
    keep = _token_keep(metadata, local_block_ids, metadata.block_keep_mask.device)
    output = torch.where(keep[:, None, :, None], output, torch.zeros_like(output))
    lse = torch.where(keep[:, None, :], lse, torch.full_like(lse, MASK_VALUE))
    return output, lse


@lru_cache(maxsize=None)
def _build_fa4_mask(
    block_size: int,
    packed: bool,
    context_window: Optional[int] = None,
    query_window: Optional[int] = None,
    draft_is_causal: bool = False,
):
    """Specialize only on mask semantics; row geometry stays on the device.

    FA4 hashes modifier closures. Capturing concrete row lengths creates fresh
    forward/backward kernels throughout a variable-length corpus. Auxiliary
    tensor dimensions are dynamic in FA4, so they also provide safe bounds.
    """
    import cutlass
    import cutlass.cute as cute
    from flash_attn.cute import utils

    @cute.jit
    def read_block_aux(aux, batch, index: "cute.TensorSSA") -> "cute.TensorSSA":
        index_fragment = cute.make_rmem_tensor(1, cutlass.Int32)
        index_fragment.store(index)
        value_fragment = cute.make_rmem_tensor(1, cutlass.Int32)
        value_fragment[0] = aux[batch[0], cutlass.min(index_fragment[0], aux.shape[1] - 1)]
        return value_fragment.load()

    @cute.jit
    def read_query_aux(aux, batch, index: "cute.TensorSSA") -> "cute.TensorSSA":
        index_fragment = cute.make_rmem_tensor(1, cutlass.Int32)
        index_fragment.store(index)
        value_fragment = cute.make_rmem_tensor(1, cutlass.Int32)
        value_fragment[0] = aux[batch[0], cutlass.min(index_fragment[0], aux.shape[1] - 1)]
        return value_fragment.load()

    @cute.jit
    def read_key_aux(aux, batch, index: "cute.TensorSSA") -> "cute.TensorSSA":
        index_fragment = cute.make_rmem_tensor(1, cutlass.Int32)
        index_fragment.store(index)
        value_fragment = cute.make_rmem_tensor(1, cutlass.Int32)
        value_fragment[0] = aux[batch[0], cutlass.min(index_fragment[0], aux.shape[1] - 1)]
        return value_fragment.load()

    @cute.jit
    def mask_mod(batch, head, query_index, key_index, seqlen_info, aux_tensors):
        del head, seqlen_info
        (
            anchors,
            keep_mask,
            query_blocks,
            context_positions,
            draft_blocks,
            query_token_keep,
            key_token_keep,
            anchor_doc_ids,
            key_context_doc_ids,
            context_length,
        ) = aux_tensors
        context_len = utils.scalar_to_ssa(context_length[0], cutlass.Int32)
        zero = utils.scalar_to_ssa(0, cutlass.Int32)
        query_block = read_query_aux(query_blocks, batch, query_index)
        anchor = read_block_aux(anchors, batch, query_block)
        keep = read_block_aux(keep_mask, batch, query_block)
        is_context = key_index < context_len
        context_position = read_key_aux(context_positions, batch, key_index)
        draft_block = read_key_aux(draft_blocks, batch, key_index)
        visible_context = is_context & (context_position < anchor)
        if cutlass.const_expr(context_window is not None):
            window = utils.scalar_to_ssa(context_window, cutlass.Int32)
            visible_context = visible_context & (context_position >= anchor - window)
        if packed:
            anchor_doc = read_block_aux(anchor_doc_ids, batch, query_block)
            context_doc = read_key_aux(key_context_doc_ids, batch, key_index)
            visible_context = (
                visible_context & (anchor_doc >= zero) & (context_doc == anchor_doc)
            )
        visible_draft = (key_index >= context_len) & (draft_block == query_block)
        query_offset = query_index % block_size
        draft_offset = (key_index - context_len) % block_size
        if cutlass.const_expr(query_window is not None):
            window = utils.scalar_to_ssa(query_window, cutlass.Int32)
            visible_context = visible_context & (anchor + query_offset - context_position < window)
            distance = query_offset - draft_offset
            visible_draft = visible_draft & (distance < window) & (distance > -window)
        if cutlass.const_expr(draft_is_causal):
            visible_draft = visible_draft & (draft_offset <= query_offset)
        query_valid = read_query_aux(query_token_keep, batch, query_index)
        key_valid = read_key_aux(key_token_keep, batch, key_index)
        return (
            (visible_context | visible_draft)
            & (keep > zero)
            & (query_valid > zero)
            & (key_valid > zero)
        )

    # Nested cute.jit helpers otherwise contribute process-specific function
    # addresses to FA4's closure hash. Bind every remaining specialization and
    # the complete factory source so disk artifacts are reusable across ranks
    # while code or semantics changes still invalidate the key.
    identity = (block_size, packed, context_window, query_window, draft_is_causal)
    mask_mod.__cute_hash__ = hashlib.sha256(
        (inspect.getsource(_build_fa4_mask) + repr(identity)).encode()
    ).hexdigest()
    return mask_mod


def _fa4_partial(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    metadata: DirectCPMetadata,
    scale: float,
) -> tuple[Tensor, Tensor]:
    if metadata.context_doc_ids is not None:
        from aspec.models.ops.thd_attention import thd_fa4_attention

        return thd_fa4_attention(query, key, value, metadata, scale)
    from flash_attn.cute.interface import flash_attn_func

    try:
        from aspec.models.ops.fa4_sparse import build_direct_cp_sparse_pair
    except ModuleNotFoundError:
        # The E18 dependency-matrix harness can load this module without importing
        # ASpec's full runtime dependency graph.
        from fa4_sparse import build_direct_cp_sparse_pair

    batch, heads, queries, _ = query.shape
    mask_mod = _build_fa4_mask(
        metadata.block_size,
        metadata.context_doc_ids is not None,
        metadata.context_window,
        metadata.query_window,
        metadata.draft_is_causal,
    )
    query_blocks = metadata.collective_block_ids.to(query.device).repeat_interleave(
        metadata.block_size
    )
    draft_blocks = metadata.local_block_ids.to(query.device).repeat_interleave(
        metadata.block_size
    )
    query_blocks = query_blocks.unsqueeze(0).expand(batch, -1).to(torch.int32)
    draft_blocks = draft_blocks.unsqueeze(0).expand(batch, -1).to(torch.int32)
    draft_length = draft_blocks.shape[1]
    key_context_positions = torch.cat(
        (
            metadata.context_position_ids.to(query.device, dtype=torch.int32),
            torch.zeros(batch, draft_length, device=query.device, dtype=torch.int32),
        ),
        dim=1,
    ).contiguous()
    key_draft_blocks = torch.cat(
        (
            torch.full(
                (batch, metadata.context_length),
                -1,
                device=query.device,
                dtype=torch.int32,
            ),
            draft_blocks,
        ),
        dim=1,
    ).contiguous()
    query_token_keep = _token_keep(
        metadata, metadata.collective_block_ids.to(query.device), query.device
    ).to(torch.int32)
    key_token_keep = torch.cat(
        (
            torch.ones(batch, metadata.context_length, device=query.device, dtype=torch.bool),
            _token_keep(metadata, metadata.local_block_ids.to(query.device), query.device),
        ),
        dim=1,
    ).to(torch.int32)
    if metadata.context_doc_ids is None:
        anchor_doc_ids = torch.zeros_like(
            metadata.anchor_positions,
            dtype=torch.int32,
            device=query.device,
        )
        context_doc_ids = torch.zeros(
            batch,
            metadata.context_length,
            device=query.device,
            dtype=torch.int32,
        )
    else:
        anchor_doc_ids = metadata.anchor_doc_ids.to(
            query.device, dtype=torch.int32
        )
        context_doc_ids = metadata.context_doc_ids.to(
            query.device, dtype=torch.int32
        )
    key_context_doc_ids = torch.cat(
        (
            context_doc_ids,
            torch.full(
                (batch, draft_length),
                -1,
                device=query.device,
                dtype=torch.int32,
            ),
        ),
        dim=1,
    ).contiguous()
    geometry_key = ("mask_context_length", query.device, metadata.context_length)
    context_length_tensor = metadata._fa4_cache.get(geometry_key)
    if context_length_tensor is None:
        context_length_tensor = torch.full(
            (1,), metadata.context_length, dtype=torch.int32, device=query.device
        )
        metadata._fa4_cache[geometry_key] = context_length_tensor
    aux_tensors = [
        metadata.anchor_positions.to(torch.int32).contiguous(),
        metadata.block_keep_mask.to(torch.int32).contiguous(),
        query_blocks.contiguous(),
        key_context_positions,
        key_draft_blocks,
        query_token_keep.contiguous(),
        key_token_keep.contiguous(),
        anchor_doc_ids.contiguous(),
        key_context_doc_ids,
        context_length_tensor,
    ]
    from aspec.models.ops.fa4_sparse import fixed_mask_aux

    aux_tensors = [fixed_mask_aux(tensor) for tensor in aux_tensors]
    sparse_pair = None
    if metadata.fa4_sparse:
        cache_key = (query.device, queries, key.shape[2], query.shape[-1], value.shape[-1],
                     metadata.context_window, metadata.query_window, metadata.draft_is_causal)
        sparse_pair = metadata._fa4_cache.get(cache_key)
        if sparse_pair is None:
            sparse_pair = build_direct_cp_sparse_pair(
                anchor_positions=metadata.anchor_positions.to(torch.int32).contiguous(),
                query_blocks=query_blocks,
                context_positions=metadata.context_position_ids.to(
                    query.device, dtype=torch.int32
                ).contiguous(),
                draft_blocks=draft_blocks,
                query_token_keep=query_token_keep,
                key_token_keep=key_token_keep,
                context_window=metadata.context_window,
                query_window=metadata.query_window,
                draft_block_size=metadata.block_size,
                query_head_dim=query.shape[-1],
                value_head_dim=value.shape[-1],
                anchor_doc_ids=(
                    anchor_doc_ids if metadata.context_doc_ids is not None else None
                ),
                context_doc_ids=(
                    context_doc_ids if metadata.context_doc_ids is not None else None
                ),
            )
            metadata._fa4_cache[cache_key] = sparse_pair
    partial, lse, *_ = flash_attn_func(
        query.permute(0, 2, 1, 3),
        key.permute(0, 2, 1, 3),
        value.permute(0, 2, 1, 3),
        softmax_scale=scale,
        causal=False,
        mask_mod=mask_mod,
        aux_tensors=aux_tensors,
        block_sparse_tensors=None if sparse_pair is None else sparse_pair.forward,
        block_sparse_tensors_bwd=None if sparse_pair is None else sparse_pair.backward,
        return_lse=True,
    )
    partial = partial.permute(0, 2, 1, 3)
    if lse.shape != (batch, heads, queries):
        lse = lse.reshape(batch, heads, queries)
    return partial, lse


def _build_cudnn_thd_mask(
    metadata: DirectCPMetadata,
    *,
    device: torch.device,
    query_length: int,
    key_length: int,
):
    """Build O(Q+K) interval metadata for one native cuDNN THD partial.

    Q and K remain in their existing batch rows. Packed-document isolation is
    represented by interval endpoints instead of regrouping and copying Q/K/V
    per document on every layer.
    """

    from aspec.models.ops.cudnn_flex_attention import CudnnFlexMask

    batch = metadata.anchor_positions.shape[0]
    expected_queries = metadata.num_blocks * metadata.block_size
    if query_length != expected_queries:
        raise ValueError(
            f"cuDNN THD expected {expected_queries} gathered queries, got {query_length}"
        )
    if key_length != metadata.context_length + metadata.queries_per_rank:
        raise ValueError("cuDNN THD key length does not match local CP ownership")

    context_positions = metadata.context_position_ids.to(device=device, dtype=torch.long)
    if context_positions.shape[1] > 1 and bool(
        (context_positions[:, 1:] < context_positions[:, :-1]).any()
    ):
        raise ValueError("cuDNN THD requires monotonically ordered context positions")

    collective_blocks = metadata.collective_block_ids.to(device=device)
    query_blocks = collective_blocks.repeat_interleave(metadata.block_size)
    offsets = torch.arange(metadata.block_size, device=device, dtype=torch.long).repeat(
        collective_blocks.numel()
    )
    anchors = metadata.anchor_positions.to(device=device, dtype=torch.long).index_select(
        1, query_blocks
    )
    query_keep = _token_keep(metadata, collective_blocks, device)

    context_lower_position = torch.zeros_like(anchors)
    if metadata.context_window is not None:
        context_lower_position = torch.maximum(
            context_lower_position, anchors - metadata.context_window
        )
    if metadata.query_window is not None:
        context_lower_position = torch.maximum(
            context_lower_position,
            anchors + offsets.unsqueeze(0) - metadata.query_window + 1,
        )
    context_begin = torch.searchsorted(
        context_positions.contiguous(), context_lower_position.contiguous(), right=False
    )
    context_end = torch.searchsorted(
        context_positions.contiguous(), anchors.contiguous(), right=False
    )

    if metadata.context_doc_ids is not None:
        context_docs = metadata.context_doc_ids.to(device=device, dtype=torch.long)
        anchor_docs = metadata.anchor_doc_ids.to(device=device, dtype=torch.long).index_select(
            1, query_blocks
        )
        # Packed document ids are monotonic except for the negative padding
        # sentinel at the tail. Move that sentinel after every real document so
        # batched searchsorted can recover document-local intervals.
        sortable_docs = torch.where(
            context_docs >= 0,
            context_docs,
            torch.full_like(context_docs, torch.iinfo(torch.long).max),
        ).contiguous()
        doc_begin = torch.searchsorted(sortable_docs, anchor_docs.contiguous(), right=False)
        doc_end = torch.searchsorted(sortable_docs, anchor_docs.contiguous(), right=True)
        context_begin = torch.maximum(context_begin, doc_begin)
        context_end = torch.minimum(context_end, doc_end)
        query_keep = query_keep & (anchor_docs >= 0)

    context_begin = torch.minimum(context_begin, context_end)

    local_blocks = metadata.local_block_ids.to(device=device)
    block_to_slot = torch.full(
        (metadata.num_blocks,), -1, device=device, dtype=torch.long
    )
    block_to_slot[local_blocks] = torch.arange(
        local_blocks.numel(), device=device, dtype=torch.long
    )
    local_slots = block_to_slot.index_select(0, query_blocks)
    owns_block = local_slots >= 0
    local_base = (
        metadata.context_length + local_slots.clamp_min(0) * metadata.block_size
    ).unsqueeze(0).expand(batch, -1)
    draft_begin = local_base
    valid_draft = torch.full_like(local_base, metadata.block_size)
    if metadata.sequence_length is not None:
        valid_draft = torch.clamp(
            metadata.sequence_length - anchors, min=0, max=metadata.block_size
        )
    draft_end = local_base + valid_draft
    if metadata.draft_is_causal:
        draft_end = torch.minimum(draft_end, local_base + offsets.unsqueeze(0) + 1)
    if metadata.query_window is not None:
        draft_begin = local_base + torch.clamp(
            offsets.unsqueeze(0) - metadata.query_window + 1, min=0
        )
        draft_end = torch.minimum(
            draft_end,
            local_base
            + torch.clamp(
                offsets.unsqueeze(0) + metadata.query_window,
                max=metadata.block_size,
            ),
        )
    draft_begin = torch.minimum(draft_begin, draft_end)
    absent = torch.full_like(draft_begin, key_length)
    draft_begin = torch.where(owns_block.unsqueeze(0), draft_begin, absent)
    draft_end = torch.where(owns_block.unsqueeze(0), draft_end, absent)

    zeros = torch.zeros_like(context_begin)
    endpoints = torch.stack(
        (zeros, context_begin, context_end, draft_begin, draft_end), dim=1
    )
    endpoints = torch.where(
        query_keep.unsqueeze(1), endpoints, torch.zeros_like(endpoints)
    )
    mask_func = (
        endpoints.permute(1, 0, 2)
        .reshape(1, endpoints.shape[1], batch * query_length)
        .to(torch.int32)
        .contiguous()
    )
    cu_seqlens_q = torch.arange(
        0,
        (batch + 1) * query_length,
        query_length,
        device=device,
        dtype=torch.int32,
    )
    cu_seqlens_k = torch.arange(
        0,
        (batch + 1) * key_length,
        key_length,
        device=device,
        dtype=torch.int32,
    )
    return CudnnFlexMask(
        mask_func=mask_func,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=query_length,
        max_seqlen_k=key_length,
    )


def _cudnn_partial(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    metadata: DirectCPMetadata,
    scale: float,
    gqa_mode: str = "auto",
) -> tuple[Tensor, Tensor]:
    from aspec.models.ops.cudnn_flex_attention import (
        _prefer_thd_physical_gqa,
        cudnn_flex_attention,
        resolve_cudnn_gqa_mode,
    )

    batch, heads, queries, query_dim = query.shape
    key_length = key.shape[2]
    cache_key = (
        "cudnn_thd",
        query.device,
        batch,
        queries,
        key_length,
        query_dim,
        value.shape[-1],
        metadata.context_window,
        metadata.query_window,
        metadata.draft_is_causal,
    )
    mask = metadata._fa4_cache.get(cache_key)
    if mask is None:
        mask = _build_cudnn_thd_mask(
            metadata,
            device=query.device,
            query_length=queries,
            key_length=key_length,
        )
        metadata._fa4_cache[cache_key] = mask

    query_thd = query.permute(0, 2, 1, 3).contiguous().view(
        batch * queries, heads, query_dim
    )
    key_thd = key.permute(0, 2, 1, 3).contiguous().view(
        batch * key_length, key.shape[1], key.shape[-1]
    )
    value_thd = value.permute(0, 2, 1, 3).contiguous().view(
        batch * key_length, value.shape[1], value.shape[-1]
    )
    # Windowed attention must not count invisible long-context keys as useful
    # reuse. Use a CPU-known upper bound; no mask scan or device synchronization.
    selection_context = metadata.context_length
    for window in (metadata.context_window, metadata.query_window):
        if window is not None:
            selection_context = min(selection_context, window)
    selection_keys = key_length - metadata.context_length + selection_context
    physical_gqa = (
        gqa_mode == "auto"
        and query.is_cuda
        and _prefer_thd_physical_gqa(queries, selection_keys, heads, key.shape[1])
        and torch.cuda.get_device_capability(query.device) == (9, 0)
    )
    effective_mode = resolve_cudnn_gqa_mode(gqa_mode, physical_gqa=physical_gqa)
    if effective_mode == "internal" and metadata.cp_size != 1:
        raise NotImplementedError("internal GQA backward currently requires CP=1")
    physical_gqa = effective_mode == "external"
    output_thd, lse_thd = cudnn_flex_attention(
        query_thd,
        key_thd,
        value_thd,
        mask=mask,
        scale=scale,
        pack_gqa=None,
        return_lse=True,
        gqa_mode=effective_mode,
    )
    output = output_thd.view(batch, queries, heads, value.shape[-1]).permute(
        0, 2, 1, 3
    )
    # Keep cuDNN's token-major storage. The Draft consumer transposes back to
    # [B,Q,H,D]; materializing BHQD here forces two otherwise redundant copies.
    # CP all-to-all explicitly packs its send buffers and accepts this view.
    if lse_thd.shape != (heads, batch * queries):
        raise RuntimeError(
            "cuDNN THD returned an unexpected LSE shape: "
            f"{tuple(lse_thd.shape)}"
        )
    lse = lse_thd.view(heads, batch, queries).permute(1, 0, 2).contiguous()

    observation = direct_cp_attention_observation(metadata)
    observation.update(
        qkv_format="thd",
        kernel_backend="cudnn_flex_attention",
        kernel_mode="varlen_interval",
        backward_strategy=(
            "cudnn_internal_gqa" if effective_mode == "internal" else "cudnn_flex_varlen"
        ),
        gqa_layout=(
            "internal_joint"
            if effective_mode == "internal"
            else ("query_head_fold" if physical_gqa else "native")
        ),
        gqa_selection_key_tokens=selection_keys,
        physical_query_tokens=queries * (heads // key.shape[1]) if physical_gqa else queries,
        backward_observed=False,
        context_parallel_size=metadata.cp_size,
        context_parallel_rank=metadata.cp_rank,
        empty_local_partition=False,
        query_tokens=queries,
        key_tokens=key_length,
        context_tokens=metadata.context_length,
        context_window=metadata.context_window,
        query_window=metadata.query_window,
        draft_is_causal=metadata.draft_is_causal,
    )

    def observe_backward(gradient):
        observation["backward_observed"] = True
        return gradient

    for tensor in (output, lse):
        if tensor.requires_grad:
            tensor.register_hook(observe_backward)
    return output, lse


def direct_cp_attention(
    query_own: Tensor,
    context_key: Tensor,
    context_value: Tensor,
    draft_key_own: Tensor,
    draft_value_own: Tensor,
    metadata: DirectCPMetadata,
    scale: Optional[float] = None,
) -> tuple[Tensor, Tensor]:
    """Return this rank's query rows after an exact cross-rank softmax merge."""

    return direct_cp_attention_from_kv(
        query_own,
        torch.cat((context_key, draft_key_own), dim=2),
        torch.cat((context_value, draft_value_own), dim=2),
        metadata,
        scale,
    )


def direct_cp_attention_from_kv(
    query_own: Tensor,
    key: Tensor,
    value: Tensor,
    metadata: DirectCPMetadata,
    scale: Optional[float] = None,
    *,
    cudnn_gqa_mode: str = "auto",
) -> tuple[Tensor, Tensor]:
    """Use already assembled context+draft KV without another long-row copy."""
    length = metadata.context_length + metadata.queries_per_rank
    if (key.ndim != 4 or value.ndim != 4 or key.shape[:3] != value.shape[:3]
            or key.shape[0] != query_own.shape[0] or key.shape[2] != length):
        raise ValueError("Combined KV must match local context and draft ownership")
    scale = scale if scale is not None else query_own.shape[-1] ** -0.5
    backend = metadata.backend
    if backend == "auto":
        backend = (
            "fa4"
            if query_own.is_cuda and query_own.dtype in (torch.float16, torch.bfloat16)
            else "math"
        )
    if cudnn_gqa_mode != "auto":
        from aspec.models.ops.cudnn_flex_attention import resolve_cudnn_gqa_mode

        resolve_cudnn_gqa_mode(cudnn_gqa_mode)
        if backend != "cudnn_flex_attention":
            raise ValueError("Explicit cudnn_gqa_mode requires the cuDNN backend")
        if cudnn_gqa_mode == "internal" and metadata.cp_size != 1:
            raise NotImplementedError("internal GQA backward currently requires CP=1")
    query_full = (
        query_own
        if metadata.cp_size == 1
        else _AllGatherQuery.apply(query_own, metadata.group)
    )
    if backend == "fa4":
        partial_output, partial_lse = _fa4_partial(query_full, key, value, metadata, scale)
    elif backend == "cudnn_flex_attention":
        partial_output, partial_lse = _cudnn_partial(
            query_full, key, value, metadata, scale, gqa_mode=cudnn_gqa_mode
        )
    elif backend == "math":
        partial_output, partial_lse = _math_partial(query_full, key, value, metadata, scale)
    else:
        raise ValueError(f"unsupported direct CP backend: {backend}")

    if metadata.cp_size == 1:
        output, lse = partial_output, partial_lse
    else:
        output_parts = _AllToAllQueryParts.apply(partial_output, metadata.group)
        lse_parts = _AllToAllQueryParts.apply(partial_lse, metadata.group)
        output, lse = _merge_partials(output_parts, lse_parts)
    output, lse = _apply_block_keep(output, lse, metadata)
    return output.to(query_own.dtype), lse


__all__ = [
    "DirectCPMetadata",
    "build_context_positions",
    "all_gather_query_rows",
    "all_to_all_query_partials",
    "direct_cp_attention_observation",
    "direct_cp_attention",
    "gather_owner_sharded_rows",
    "merge_attention_partials",
]
