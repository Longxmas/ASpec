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

"""Experimental frozen-head FP8 forward AND input-gradient projection on SM90.

Independent implementation of row-scaled GEMM and a straight-through backward.
Inspired by SGLang PR 39622 (d3cacb55, cached frozen-head projection) and
torchao 4ab9d61d (different rowwise weight axes for output and input gradient).
No upstream implementation is copied. Uses PyTorch's existing scaled-mm kernel.

DSpark constructs both disposable FP8 caches from the frozen Target weight
after initialization or checkpoint restore. Exports retain canonical BF16 weights.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


def quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """E4M3 with a power-of-two dequantization scale per GEMM output row.

    Scaling is reduced in FP32. An absolute floor bounds tiny/zero-row scales.
    Quantization rounding is not differentiated; the custom backward implements
    the selected approximate linear gradient, as in float8 training recipes.
    """
    values = x.float()
    scale = torch.exp2(torch.ceil(torch.log2(values.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448.0)))
    bits = (values / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return bits, scale


@triton.jit
def _maximum_with_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit(do_not_specialize=["ROWS"])
def _small_rows(X, Q, S, ROWS, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    value = tl.load(X + row * WIDTH + col, (row < ROWS) & (col < WIDTH), other=0).to(tl.float32)
    maximum = _maximum_with_nan(tl.reduce(tl.abs(value), 0, _maximum_with_nan), 1e-12)
    scale = tl.exp2(tl.ceil(tl.log2(maximum / 448.0)))
    quantized = tl.clamp(value / scale, -448.0, 448.0, propagate_nan=tl.PropagateNan.ALL)
    tl.store(Q + row * WIDTH + col, quantized, col < WIDTH)
    tl.store(S + row, scale)


@triton.jit
def _row_partials(X, P, WIDTH: tl.constexpr, TILES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + row * WIDTH + col, col < WIDTH, other=0).to(tl.float32)
    maximum = tl.reduce(tl.abs(value), 0, _maximum_with_nan)
    tl.store(P + row * TILES + tile, maximum)


@triton.jit(do_not_specialize=["ROWS"])
def _quantize_tiles(X, P, Q, S, ROWS, WIDTH: tl.constexpr,
                    TILES: tl.constexpr, REDUCE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    indices = tl.arange(0, REDUCE)
    partial = tl.load(P + row * TILES + indices, (row < ROWS) & (indices < TILES), other=0)
    maximum = _maximum_with_nan(tl.reduce(partial, 0, _maximum_with_nan), 1e-12)
    scale = tl.exp2(tl.ceil(tl.log2(maximum / 448.0)))
    col = tile * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + row * WIDTH + col, (row < ROWS) & (col < WIDTH), other=0).to(tl.float32)
    quantized = tl.clamp(value / scale, -448.0, 448.0, propagate_nan=tl.PropagateNan.ALL)
    tl.store(Q + row * WIDTH + col, quantized, col < WIDTH)
    if tile == 0:
        tl.store(S + row, scale)


def quantize_activation(x):
    """Fuse zero padding with quantization; tile long vocabulary reductions.

    The first compiled-PyTorch prototype spent ~3 ms reducing a 151936-wide
    gradient and materialized a padded BF16 copy. This two-pass reduction uses
    bounded per-program state and emits the padded FP8 buffer directly.
    """
    x = x.contiguous()
    rows, width = x.shape
    padded = triton.cdiv(rows, 16) * 16
    bits = torch.empty((padded, width), dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty((padded, 1), dtype=torch.float32, device=x.device)
    if width <= 8192:
        _small_rows[(padded,)](x, bits, scale, rows, width, triton.next_power_of_2(width))
    else:
        block = 4096
        tiles = triton.cdiv(width, block)
        partials = torch.empty((rows, tiles), dtype=torch.float32, device=x.device)
        _row_partials[(rows, tiles)](x, partials, width, tiles, block)
        _quantize_tiles[(padded, tiles)](
            x, partials, bits, scale, rows, width, tiles, triton.next_power_of_2(tiles), block,
        )
    return bits, scale


def _project(x, weight_t, scale_t):
    rows = x.shape[0]
    bits, scale = quantize_activation(x)
    output = torch._scaled_mm(
        bits, weight_t, scale_a=scale, scale_b=scale_t,
        out_dtype=torch.bfloat16, use_fast_accum=False,
    )
    return output[:rows]


class _FrozenHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, forward_weight_t, forward_scale_t, backward_weight_t, backward_scale_t):
        ctx.input_shape = hidden.shape
        ctx.save_for_backward(backward_weight_t, backward_scale_t)
        matrix = hidden.reshape(-1, hidden.shape[-1])
        if matrix.shape[0] == 0:
            return hidden.new_empty((*hidden.shape[:-1], forward_weight_t.shape[1]))
        return _project(matrix, forward_weight_t, forward_scale_t).view(
            *hidden.shape[:-1], forward_weight_t.shape[1]
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        weight_t, scale_t = ctx.saved_tensors
        if grad_output.numel() == 0:
            grad_hidden = grad_output.new_empty(ctx.input_shape)
        else:
            matrix = grad_output.reshape(-1, grad_output.shape[-1])
            grad_hidden = _project(matrix, weight_t, scale_t).view(ctx.input_shape)
        return grad_hidden, None, None, None, None


class FrozenFP8Head:
    """Explicitly initialized, version-checked cache for one frozen BF16 head.

    Shape: canonical weight [vocabulary, hidden]. Both dimensions must divide 16.
    The caller must rebuild after replacing/mutating the canonical weight. Cache
    construction, validation and allocated bytes are visible to the benchmark.
    """

    def __init__(self, weight: torch.Tensor):
        if weight.ndim != 2 or not weight.is_cuda or weight.dtype != torch.bfloat16:
            raise ValueError("FP8 head requires a CUDA BF16 [vocabulary, hidden] weight")
        if weight.requires_grad:
            raise ValueError("FP8 head requires frozen weights")
        if min(weight.shape) == 0 or any(size % 16 for size in weight.shape):
            raise ValueError("FP8 head weight dimensions must be positive multiples of 16")
        if torch.cuda.get_device_capability(weight.device) != (9, 0):
            raise ValueError("This prototype is scoped to SM90")
        if not bool(torch.isfinite(weight).all()):
            raise ValueError("FP8 head weight contains nonfinite values")
        self.weight = weight
        self.version = weight._version
        with torch.no_grad():
            forward_bits, forward_scale = quantize_rows(weight.contiguous())
            backward_bits, backward_scale = quantize_rows(weight.t().contiguous())
        self.forward_weight_t = forward_bits.t()
        self.forward_scale_t = forward_scale.t()
        self.backward_weight_t = backward_bits.t()
        self.backward_scale_t = backward_scale.t()

    @property
    def cache_bytes(self):
        return sum(t.numel() * t.element_size() for t in (
            self.forward_weight_t, self.forward_scale_t,
            self.backward_weight_t, self.backward_scale_t,
        ))

    def __call__(self, hidden: torch.Tensor):
        if self.weight._version != self.version or self.weight.requires_grad:
            raise ValueError("Frozen head weight changed; rebuild the FP8 cache")
        if hidden.ndim < 2 or hidden.shape[-1] != self.weight.shape[1]:
            raise ValueError("Hidden input must end in the head hidden dimension")
        if hidden.device != self.weight.device or hidden.dtype != torch.bfloat16:
            raise ValueError("Hidden input must use the same CUDA device and BF16 dtype")
        return _FrozenHead.apply(
            hidden, self.forward_weight_t, self.forward_scale_t,
            self.backward_weight_t, self.backward_scale_t,
        )
