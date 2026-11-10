# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Optional

import torch
import torch.nn.functional as F


def use_sm90_tiled_l1(logits: torch.Tensor, target: torch.Tensor, indexed: bool = False) -> bool:
    """Keep the established path outside compact SM90 BF16 distillation."""
    return (
        not torch.compiler.is_compiling()
        and logits.is_cuda
        and logits.dtype == torch.bfloat16
        and target.dtype == logits.dtype
        and target.device == logits.device
        and not target.requires_grad
        and (logits.shape == target.shape if not indexed else (
            target.ndim == 3 and target.shape[0] == logits.shape[0]
            and target.shape[-1] == logits.shape[-1]
        ))
        and logits.numel() > 0
        and len(logits.shape) > 0
        and logits.shape[-1] > 4096
        and logits.is_contiguous()
        and target.is_contiguous()
        and torch.cuda.get_device_capability(logits.device) == (9, 0)
    )


def _forward_kl_from_logits(logits: torch.Tensor, target_p: torch.Tensor) -> torch.Tensor:
    logits_f32 = logits.float()
    return torch.logsumexp(logits_f32, dim=-1) - (target_p * logits_f32).sum(-1)


def _softmax_from_logits(logits: torch.Tensor) -> torch.Tensor:
    logits_f32 = logits.float()
    return torch.exp(logits_f32 - torch.logsumexp(logits_f32, dim=-1, keepdim=True))


def sampling_logits(logits: torch.Tensor, sampling=(1.0, 1.0, -1)) -> torch.Tensor:
    """Differentiable logits on joint top-k/top-p support; ties use token-ID order.

    Both filters use the full temperature-scaled distribution, matching the
    pinned SGLang joint-filter convention. The discrete support has no gradient.
    """
    if sampling == (1.0, 1.0, -1):
        return logits
    temperature, top_p, top_k = sampling
    scaled = logits.float()
    if temperature != 1.0:
        scaled = (scaled - scaled.amax(dim=-1, keepdim=True)) / temperature
    vocab = scaled.shape[-1]
    if top_p == 1.0 and (top_k == -1 or top_k >= vocab):
        return scaled
    sorted_logits, indices = scaled.sort(dim=-1, descending=True, stable=True)
    remove = torch.zeros_like(sorted_logits, dtype=torch.bool)
    if top_p < 1.0:
        probs = sorted_logits.softmax(dim=-1)
        remove = probs.cumsum(dim=-1) - probs > top_p
    if 0 < top_k < vocab:
        remove = remove | (torch.arange(vocab, device=logits.device) >= top_k)
    return scaled.masked_fill(remove.scatter(-1, indices, remove), float("-inf"))


@torch.compile(dynamic=None)
def compiled_l1_from_logits(logits: torch.Tensor, target_logits: torch.Tensor,
                            target_sampling=(1.0, 1.0, -1),
                            draft_sampling=(1.0, 1.0, -1)) -> torch.Tensor:
    """Per-token L1 distance between optional sampling-transformed FP32 distributions."""
    draft_probs = torch.softmax(sampling_logits(logits, draft_sampling).float(), dim=-1)
    target_probs = torch.softmax(sampling_logits(target_logits, target_sampling).float(), dim=-1)
    return (draft_probs - target_probs).abs().sum(dim=-1)


def _project(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_up: Optional[torch.Tensor],
) -> torch.Tensor:
    """Draft logits. ``lm_head_up`` is None for a dense head, the up projection otherwise."""
    logits = F.linear(hidden, lm_head_weight)
    if lm_head_up is not None:
        logits = F.linear(logits, lm_head_up)
    return logits


def _rmsnorm_logits(
    hs_flat: torch.Tensor,
    norm_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_up: Optional[torch.Tensor],
    norm_eps: float,
) -> torch.Tensor:
    hs_f32 = hs_flat.float()
    variance = hs_f32.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + norm_eps)
    norm_hs = (hs_f32 * rstd).to(hs_flat.dtype) * norm_weight
    return _project(norm_hs, lm_head_weight, lm_head_up)


@torch.compile(dynamic=None)
def compiled_sum_forward_kl_loss(
    prenorm_hidden_states_flat,
    target_p_flat,
    valid_idx,
    norm_weight,
    lm_head_weight,
    lm_head_up,
    norm_eps,
):
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    tp = target_p_flat.index_select(0, valid_idx)

    hs_f32 = hs.float()
    variance = hs_f32.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + norm_eps)
    norm_hs = (hs_f32 * rstd).to(hs.dtype) * norm_weight

    logits = _project(norm_hs, lm_head_weight, lm_head_up)
    token_loss = _forward_kl_from_logits(logits, tp)
    correct = (logits.argmax(-1) == tp.argmax(-1)).float()
    count = torch.ones_like(token_loss, dtype=torch.float32).sum()
    return token_loss.sum(), correct.sum(), count


@torch.compile(dynamic=None)
def compiled_forward_kl_loss(
    prenorm_hidden_states_flat,
    target_p_flat,
    valid_idx,
    norm_weight,
    lm_head_weight,
    lm_head_up,
    norm_eps,
):
    """torch.compile'd index_select + RMSNorm + lm_head + Forward KL loss.

    Takes full (B*T, ...) flat tensors and performs index_select inside the
    compiled graph so the compiler can fuse the gather with subsequent ops.

    Args:
        prenorm_hidden_states_flat: (B*T, H) — flattened draft hidden states
        target_p_flat: (B*T, V_out) — flattened target probs (detached)
        valid_idx: (N,) int64 — indices of non-masked positions
        norm_weight: (H,)
        lm_head_weight: (V_out, H) — draft lm_head weight
        norm_eps: float
    """
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    tp = target_p_flat.index_select(0, valid_idx)

    # RMSNorm
    hs_f32 = hs.float()
    variance = hs_f32.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + norm_eps)
    norm_hs = (hs_f32 * rstd).to(hs.dtype) * norm_weight

    logits = _project(norm_hs, lm_head_weight, lm_head_up)  # (N, V_out)

    token_loss = _forward_kl_from_logits(logits, tp)
    correct = (logits.argmax(-1) == tp.argmax(-1)).float()
    count = torch.ones_like(token_loss, dtype=torch.float32).sum()
    return token_loss.sum(), correct.sum(), count


@torch.compile(dynamic=None)
def compiled_lk_alpha_loss(
    prenorm_hidden_states_flat,
    target_p_flat,
    valid_idx,
    norm_weight,
    lm_head_weight,
    lm_head_up,
    norm_eps,
    coverage_flat,
):
    """LK^alpha loss: -log(acceptance_rate), summed over valid positions.

    Directly optimizes the (log) acceptance rate alpha_i = sum_x min(p_i(x), q_i(x))
    from "LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding"
    (arXiv:2602.23881). Returns (loss_sum, correct_sum, count, alpha_sum) — the same
    sum/count convention as ``compiled_forward_kl_loss`` plus a trailing alpha sum.

    ``target_p_flat`` is the target distribution renormalized over the (possibly
    pruned) draft vocab; ``coverage_flat`` rescales it back to the true,
    un-renormalized target probabilities before computing alpha, per §4.4 of the
    paper (out-of-draft-vocab mass must contribute 0, not be redistributed).
    ``coverage_flat`` is 1 when there is no vocab pruning, leaving ``tp`` unchanged.
    """
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    tp_tilde = target_p_flat.index_select(0, valid_idx)
    coverage = coverage_flat.index_select(0, valid_idx)

    logits = _rmsnorm_logits(hs, norm_weight, lm_head_weight, lm_head_up, norm_eps)
    q = _softmax_from_logits(logits)

    tp = tp_tilde * coverage.unsqueeze(-1)
    alpha = torch.min(tp, q).sum(-1)
    token_loss = -torch.log(alpha.clamp(min=1e-8))
    correct = (logits.argmax(-1) == tp_tilde.argmax(-1)).float()
    count = torch.ones_like(token_loss, dtype=torch.float32).sum()
    return token_loss.sum(), correct.sum(), count, alpha.sum()


@torch.compile(dynamic=None)
def compiled_lk_lambda_loss(
    prenorm_hidden_states_flat,
    target_p_flat,
    valid_idx,
    norm_weight,
    lm_head_weight,
    lm_head_up,
    norm_eps,
    coverage_flat,
    external_mean_alpha,
    eta,
):
    """LK^lambda loss: lambda*KL(p||q) + (1-lambda)*TV(p,q), lambda = exp(-eta*sg[alpha]).

    Returns (loss_sum, correct_sum, count, alpha_sum), matching
    ``compiled_lk_alpha_loss``'s convention. ``lambda`` is computed once per call
    (i.e. once per EAGLE-3 depth) from the mean acceptance rate across the batch
    and sequence, matching the paper's schedule — not per token. See
    ``compiled_lk_alpha_loss`` for ``coverage_flat``'s role in vocab pruning: the
    KL term still uses the renormalized ``tp_tilde`` (raw ``p`` would make KL
    diverge for out-of-draft-vocab mass), but alpha/TV use the true ``tp``.

    ``external_mean_alpha``, if not None, overrides the locally-computed mean
    alpha used to form lambda. This is required under sequence parallelism
    (USP): a single call only sees this rank's shard of the depth's sequence,
    but the schedule is defined over the *full* batch and sequence for the
    depth (arXiv:2602.23881), so the caller must all-reduce alpha across the
    SP group before forming lambda and pass the result in here. Pass None
    when there is no sequence sharding, in which case the local mean already
    covers the whole depth.
    """
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    tp_tilde = target_p_flat.index_select(0, valid_idx)
    coverage = coverage_flat.index_select(0, valid_idx)

    logits = _rmsnorm_logits(hs, norm_weight, lm_head_weight, lm_head_up, norm_eps)
    q = _softmax_from_logits(logits)
    log_q = F.log_softmax(logits.float(), dim=-1)

    tp = tp_tilde * coverage.unsqueeze(-1)
    alpha = torch.min(tp, q).sum(-1)
    mean_alpha = external_mean_alpha if external_mean_alpha is not None else alpha.mean()
    lam = torch.exp(-eta * mean_alpha.detach())

    kl = F.kl_div(log_q, tp_tilde, reduction="none").sum(-1)
    tv = 0.5 * (tp - q).abs().sum(-1)
    token_loss = lam * kl + (1.0 - lam) * tv

    correct = (logits.argmax(-1) == tp_tilde.argmax(-1)).float()
    count = torch.ones_like(token_loss, dtype=torch.float32).sum()
    return token_loss.sum(), correct.sum(), count, alpha.sum()


@torch.compile(dynamic=None)
def compiled_sum_forward_kl_loss_from_hs(
    prenorm_hidden_states_flat,
    target_hidden_states_flat,
    valid_idx,
    norm_weight,
    lm_head_weight,
    lm_head_up,
    target_lm_head_weight,
    norm_eps,
):
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    ths = target_hidden_states_flat.index_select(0, valid_idx)

    target_logits = F.linear(ths, target_lm_head_weight)
    tp = _softmax_from_logits(target_logits)

    hs_f32 = hs.float()
    variance = hs_f32.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + norm_eps)
    norm_hs = (hs_f32 * rstd).to(hs.dtype) * norm_weight

    logits = _project(norm_hs, lm_head_weight, lm_head_up)
    token_loss = _forward_kl_from_logits(logits, tp)
    correct = (logits.argmax(-1) == target_logits.argmax(-1)).float()
    count = torch.ones_like(token_loss, dtype=torch.float32).sum()
    return token_loss.sum(), correct.sum(), count


@torch.compile(dynamic=None)
def compiled_lk_alpha_loss_from_hs(
    prenorm_hidden_states_flat,
    target_hidden_states_flat,
    valid_idx,
    norm_weight,
    lm_head_weight,
    lm_head_up,
    target_lm_head_weight,
    norm_eps,
):
    """LK^alpha loss from hidden states (LazyTarget path). See ``compiled_lk_alpha_loss``."""
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    ths = target_hidden_states_flat.index_select(0, valid_idx)

    target_logits = F.linear(ths, target_lm_head_weight)
    tp = _softmax_from_logits(target_logits)

    logits = _rmsnorm_logits(hs, norm_weight, lm_head_weight, lm_head_up, norm_eps)
    q = _softmax_from_logits(logits)

    alpha = torch.min(tp, q).sum(-1)
    token_loss = -torch.log(alpha.clamp(min=1e-8))
    correct = (logits.argmax(-1) == target_logits.argmax(-1)).float()
    count = torch.ones_like(token_loss, dtype=torch.float32).sum()
    return token_loss.sum(), correct.sum(), count, alpha.sum()


@torch.compile(dynamic=None)
def compiled_lk_alpha_loss_from_logits(
    prenorm_hidden_states_flat,
    target_logits_flat,
    valid_idx,
    norm_weight,
    lm_head_weight,
    lm_head_up,
    norm_eps,
):
    """LK^alpha from cached full-vocabulary Target logits."""
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    target_logits = target_logits_flat.index_select(0, valid_idx)
    tp = _softmax_from_logits(target_logits)

    logits = _rmsnorm_logits(hs, norm_weight, lm_head_weight, lm_head_up, norm_eps)
    q = _softmax_from_logits(logits)

    alpha = torch.min(tp, q).sum(-1)
    token_loss = -torch.log(alpha.clamp(min=1e-8))
    correct = (logits.argmax(-1) == target_logits.argmax(-1)).float()
    count = torch.ones_like(token_loss, dtype=torch.float32).sum()
    return token_loss.sum(), correct.sum(), count, alpha.sum()


@torch.compile(dynamic=None)
def compiled_lk_lambda_loss_from_hs(
    prenorm_hidden_states_flat,
    target_hidden_states_flat,
    valid_idx,
    norm_weight,
    lm_head_weight,
    lm_head_up,
    target_lm_head_weight,
    norm_eps,
    external_mean_alpha,
    eta,
):
    """LK^lambda loss from hidden states (LazyTarget path). See
    ``compiled_lk_lambda_loss`` for ``external_mean_alpha``'s role under USP."""
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    ths = target_hidden_states_flat.index_select(0, valid_idx)

    target_logits = F.linear(ths, target_lm_head_weight)
    tp = _softmax_from_logits(target_logits)

    logits = _rmsnorm_logits(hs, norm_weight, lm_head_weight, lm_head_up, norm_eps)
    q = _softmax_from_logits(logits)
    log_q = F.log_softmax(logits.float(), dim=-1)

    alpha = torch.min(tp, q).sum(-1)
    # lambda is computed once per call (i.e. once per EAGLE-3 depth) from the
    # mean acceptance rate across batch and sequence, matching the paper's
    # schedule — not per token (and, under USP, from external_mean_alpha —
    # the SP-group-reduced mean — rather than this rank's local shard alone).
    mean_alpha = external_mean_alpha if external_mean_alpha is not None else alpha.mean()
    lam = torch.exp(-eta * mean_alpha.detach())

    kl = F.kl_div(log_q, tp, reduction="none").sum(-1)
    tv = 0.5 * (tp - q).abs().sum(-1)
    token_loss = lam * kl + (1.0 - lam) * tv

    correct = (logits.argmax(-1) == target_logits.argmax(-1)).float()
    count = torch.ones_like(token_loss, dtype=torch.float32).sum()
    return token_loss.sum(), correct.sum(), count, alpha.sum()


@torch.compile(dynamic=None)
def compiled_lk_lambda_loss_from_logits(
    prenorm_hidden_states_flat,
    target_logits_flat,
    valid_idx,
    norm_weight,
    lm_head_weight,
    lm_head_up,
    norm_eps,
    external_mean_alpha,
    eta,
):
    """LK^lambda from cached full-vocabulary Target logits."""
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    target_logits = target_logits_flat.index_select(0, valid_idx)
    tp = _softmax_from_logits(target_logits)

    logits = _rmsnorm_logits(hs, norm_weight, lm_head_weight, lm_head_up, norm_eps)
    q = _softmax_from_logits(logits)
    log_q = F.log_softmax(logits.float(), dim=-1)

    alpha = torch.min(tp, q).sum(-1)
    mean_alpha = external_mean_alpha if external_mean_alpha is not None else alpha.mean()
    lam = torch.exp(-eta * mean_alpha.detach())

    kl = F.kl_div(log_q, tp, reduction="none").sum(-1)
    tv = 0.5 * (tp - q).abs().sum(-1)
    token_loss = lam * kl + (1.0 - lam) * tv

    correct = (logits.argmax(-1) == target_logits.argmax(-1)).float()
    count = torch.ones_like(token_loss, dtype=torch.float32).sum()
    return token_loss.sum(), correct.sum(), count, alpha.sum()


@torch.compile(dynamic=None)
def compiled_forward_kl_loss_from_hs(
    prenorm_hidden_states_flat,
    target_hidden_states_flat,
    valid_idx,
    norm_weight,
    lm_head_weight,
    lm_head_up,
    target_lm_head_weight,
    norm_eps,
):
    """torch.compile'd index_select + target softmax + RMSNorm + lm_head + Forward KL loss.

    Like compiled_forward_kl_loss but takes full (B*T, ...) flat tensors and
    performs index_select inside the compiled graph.  This lets the compiler
    fuse the gather with subsequent ops, avoiding a separate (N, V_full) copy
    outside the compiled region.

    Used for the non-pruning (LazyTarget) path where V_full is large.
    """
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    ths = target_hidden_states_flat.index_select(0, valid_idx)

    # Target probs (detached weights → no grad flows through target)
    target_logits = F.linear(ths, target_lm_head_weight)
    tp = _softmax_from_logits(target_logits)

    # RMSNorm
    hs_f32 = hs.float()
    variance = hs_f32.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + norm_eps)
    norm_hs = (hs_f32 * rstd).to(hs.dtype) * norm_weight

    logits = _project(norm_hs, lm_head_weight, lm_head_up)

    token_loss = _forward_kl_from_logits(logits, tp)
    correct = (logits.argmax(-1) == target_logits.argmax(-1)).float()
    count = torch.ones_like(token_loss, dtype=torch.float32).sum()
    return token_loss.sum(), correct.sum(), count


@torch.compile(dynamic=None)
def compiled_forward_kl_loss_from_logits(
    prenorm_hidden_states_flat,
    target_logits_flat,
    valid_idx,
    norm_weight,
    lm_head_weight,
    lm_head_up,
    norm_eps,
):
    """Forward-KL objective from cached full-vocabulary Target logits."""
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    target_logits = target_logits_flat.index_select(0, valid_idx)
    tp = _softmax_from_logits(target_logits)

    hs_f32 = hs.float()
    variance = hs_f32.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + norm_eps)
    norm_hs = (hs_f32 * rstd).to(hs.dtype) * norm_weight

    logits = _project(norm_hs, lm_head_weight, lm_head_up)
    token_loss = _forward_kl_from_logits(logits, tp)
    correct = (logits.argmax(-1) == target_logits.argmax(-1)).float()
    count = torch.ones_like(token_loss, dtype=torch.float32).sum()
    return token_loss.sum(), correct.sum(), count
