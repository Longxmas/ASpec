# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-project/speculators at
# 3702320434bbdaa853f0846a66e6b6b8f1d97aec:
# models/peagle/{core,attention,metrics}.py and
# models/eagle3/model_definitions.py. See licenses/Speculators-LICENSE.
# Changes: Qwen3-only model/loss prerequisite, deterministic dense depth layout,
# contiguous-document isolation, explicit aligned inputs and detached Target data.
"""Parallel P-EAGLE core; no trainer, sampler, checkpoint/export or serving adapter."""

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)


def _segments(document_ids: torch.Tensor) -> torch.Tensor:
    changes = torch.ones_like(document_ids)
    changes[1:] = (document_ids[1:] != document_ids[:-1]).long()
    return changes.cumsum(0)


def peagle_dense_layout(
    document_ids: torch.Tensor, num_depths: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """All base positions, then every in-document (anchor, depth) pair.

    This deterministic unsampled layout is for the model/loss prerequisite;
    it does not implement the reference's stochastic COD training policy.
    """
    if document_ids.ndim != 1 or document_ids.dtype != torch.long or not document_ids.numel():
        raise ValueError("P-EAGLE document_ids must be a nonempty int64 vector")
    if (document_ids < -1).any():
        raise ValueError("P-EAGLE uses -1 for padding and nonnegative document IDs")
    if isinstance(num_depths, bool) or not isinstance(num_depths, int) or num_depths < 1:
        raise ValueError("P-EAGLE num_depths must be a positive integer")
    segments = _segments(document_ids)
    base = torch.arange(document_ids.numel(), device=document_ids.device)
    anchors, depths = [base], [torch.zeros_like(base)]
    for d in range(1, min(num_depths, document_ids.numel())):
        candidates = base[:-d]
        valid = (document_ids[:-d] >= 0) & (segments[:-d] == segments[d:])
        candidates = candidates[valid]
        anchors.append(candidates)
        depths.append(torch.full_like(candidates, d))
    return torch.cat(anchors), torch.cat(depths)


def peagle_attention_mask(
    anchor_pos: torch.Tensor, depth: torch.Tensor, document_ids: torch.Tensor
) -> torch.Tensor:
    """Boolean [N,N] mask: causal base context and earlier depths of the same chain.

    Padding/cross-document slots attend only to themselves. Repeated document
    labels in disjoint segments never share attention.
    """
    if anchor_pos.ndim != 1 or depth.shape != anchor_pos.shape:
        raise ValueError("P-EAGLE anchor/depth vectors must have the same shape")
    if anchor_pos.dtype != torch.long or depth.dtype != torch.long:
        raise ValueError("P-EAGLE anchor/depth vectors must use int64")
    if document_ids.ndim != 1 or document_ids.dtype != torch.long:
        raise ValueError("P-EAGLE document_ids must be an int64 vector")
    if not (anchor_pos.device == depth.device == document_ids.device):
        raise ValueError("P-EAGLE layout tensors must share a device")
    positions = anchor_pos + depth
    if ((anchor_pos < 0) | (depth < 0) | (positions >= document_ids.numel())).any():
        raise ValueError("P-EAGLE layout is outside the sequence")
    segments = _segments(document_ids)
    valid = (document_ids[anchor_pos] >= 0) & (segments[anchor_pos] == segments[positions])
    same_document = segments[anchor_pos, None] == segments[anchor_pos][None, :]
    base_context = (depth[None, :] == 0) & (anchor_pos[:, None] >= anchor_pos[None, :])
    same_chain = (anchor_pos[:, None] == anchor_pos[None, :]) & (
        depth[:, None] >= depth[None, :]
    )
    allowed = valid[:, None] & valid[None, :] & same_document & (base_context | same_chain)
    return allowed | torch.eye(anchor_pos.numel(), dtype=torch.bool, device=anchor_pos.device)


class _PEagleFirstLayer(Qwen3DecoderLayer):
    """Speculators' EAGLE first-layer fusion, retaining Qwen3 Q/K normalization."""

    def __init__(self, config):
        super().__init__(config, layer_idx=0)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        for name in ("q_proj", "k_proj", "v_proj"):
            previous = getattr(self.self_attn, name)
            setattr(
                self.self_attn, name,
                nn.Linear(2 * config.hidden_size, previous.out_features,
                          bias=config.attention_bias),
            )

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None):
        embeds, hidden = hidden_states.chunk(2, dim=-1)
        fused = torch.cat((self.input_layernorm(embeds), self.hidden_norm(hidden)), dim=-1)
        attended, _ = self.self_attn(
            hidden_states=fused,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden = hidden + attended  # norm_before_residual=False in the selected recipe
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


@dataclass
class PEagleOutput:
    logits: torch.Tensor
    anchor_pos: torch.Tensor
    depth: torch.Tensor
    valid: torch.Tensor
    num_depths: int

    @property
    def positions(self) -> torch.Tensor:
        return self.anchor_pos + self.depth


class PEagleDraftModel(nn.Module):
    """One packed row of aligned Qwen3 Target features, all depths in one pass.

    At aligned index i: input_ids=x[i+1], hidden_states=Target auxiliary h[i],
    teacher_logits=Target logits[i+1], loss_mask=reference mask[i+1].
    Each document must be shifted separately before packing. Future depth d
    at anchor a uses learned token/hidden masks and targets aligned index a+d.
    See docs/user-guide/peagle.md for the scope and alignment contract.
    """

    def __init__(
        self, config: Qwen3Config, *, mask_token_id: int,
        num_aux_hidden_states: int = 3, num_depths: int = 4,
    ):
        super().__init__()
        if config.model_type != "qwen3":
            raise ValueError("P-EAGLE core currently supports only Qwen3")
        if config._attn_implementation not in (None, "eager", "sdpa"):
            raise ValueError("P-EAGLE core supports only eager or sdpa attention")
        if config.attention_dropout != 0 or any(
            kind != "full_attention" for kind in config.layer_types
        ):
            raise ValueError("P-EAGLE core requires full attention and zero dropout")
        if (getattr(config, "draft_vocab_size", config.vocab_size) != config.vocab_size
                or config.tie_word_embeddings
                or any(getattr(config, field, False) for field in (
                    "norm_before_residual", "norm_before_fc", "fc_norm",
                ))):
            raise ValueError("P-EAGLE core requires the selected full-vocabulary untied-head recipe")
        if (config.num_attention_heads < 1 or config.num_key_value_heads < 1
                or config.num_attention_heads % config.num_key_value_heads
                or config.head_dim < 1 or config.head_dim % 2):
            raise ValueError("P-EAGLE core requires valid GQA geometry and an even head dimension")
        if config.num_hidden_layers < 1 or config.hidden_act != "silu":
            raise ValueError("P-EAGLE core requires at least one layer and SiLU")
        for name, value in (("num_aux_hidden_states", num_aux_hidden_states),
                            ("num_depths", num_depths)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"P-EAGLE {name} must be a positive integer")
        if isinstance(mask_token_id, bool) or not isinstance(mask_token_id, int) or not (
            0 <= mask_token_id < config.vocab_size
        ):
            raise ValueError("P-EAGLE mask_token_id must be in the Target vocabulary")
        self.config = copy.deepcopy(config)
        self.config._attn_implementation = config._attn_implementation or "sdpa"
        self.mask_token_id = mask_token_id
        self.num_aux_hidden_states = num_aux_hidden_states
        self.num_depths = num_depths
        h = config.hidden_size
        # No padding_idx: the learned mask-token row must receive gradients.
        self.embed_tokens = nn.Embedding(config.vocab_size, h)
        self.mask_hidden = nn.Parameter(torch.randn(1, 1, num_aux_hidden_states * h))
        self.fc = nn.Linear(num_aux_hidden_states * h, h, bias=False)
        self.layers = nn.ModuleList([_PEagleFirstLayer(self.config)] + [
            Qwen3DecoderLayer(self.config, layer_idx=i)
            for i in range(1, config.num_hidden_layers)
        ])
        self.rotary_emb = Qwen3RotaryEmbedding(self.config)
        self.norm = Qwen3RMSNorm(h, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(h, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, hidden_states, document_ids, position_ids=None) -> PEagleOutput:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or not input_ids.shape[1]:
            raise ValueError("P-EAGLE core requires one nonempty packed row [1,S]")
        if input_ids.dtype != torch.long or document_ids.shape != input_ids.shape:
            raise ValueError("P-EAGLE requires int64 IDs and document_ids shaped [1,S]")
        expected = (*input_ids.shape, self.fc.in_features)
        if hidden_states.shape != expected or not hidden_states.is_floating_point():
            raise ValueError(f"P-EAGLE hidden_states must be floating point with shape {expected}")
        if not (input_ids.device == hidden_states.device == document_ids.device
                == self.embed_tokens.weight.device):
            raise ValueError("P-EAGLE inputs and model must share a device")
        if ((input_ids < 0) | (input_ids >= self.config.vocab_size)).any():
            raise ValueError("P-EAGLE input token is outside the Target vocabulary")
        docs = document_ids[0]
        anchors, depth = peagle_dense_layout(docs, self.num_depths)
        positions = anchors + depth
        if position_ids is None:
            base = torch.arange(docs.numel(), device=docs.device)
            starts = torch.cat((torch.ones_like(docs[:1]), (docs[1:] != docs[:-1]).long()))
            position_ids = (base - torch.where(starts.bool(), base, 0).cummax(0).values)[None] + 1
        if (position_ids.shape != input_ids.shape or position_ids.dtype != torch.long
                or position_ids.device != input_ids.device or (position_ids < 0).any()):
            raise ValueError("P-EAGLE position_ids must be nonnegative int64 [1,S] on the input device")
        is_base = (depth == 0)[None]
        token_ids = torch.where(is_base, input_ids[:, positions], self.mask_token_id)
        sampled_hidden = torch.where(
            is_base[..., None], hidden_states.detach()[:, positions],
            self.mask_hidden.to(hidden_states.dtype),
        )
        projected = self.fc(sampled_hidden.to(self.fc.weight.dtype))
        states = torch.cat((self.embed_tokens(token_ids), projected), dim=-1)
        allowed = peagle_attention_mask(anchors, depth, docs)
        mask = states.new_zeros(allowed.shape).masked_fill(~allowed, -torch.inf)[None, None]
        rope = self.rotary_emb(states, position_ids[:, positions])
        for layer in self.layers:
            states = layer(states, attention_mask=mask, position_embeddings=rope)
        return PEagleOutput(
            self.lm_head(self.norm(states)), anchors, depth,
            docs[positions] >= 0, self.num_depths,
        )


def peagle_kl_loss(
    output: PEagleOutput, teacher_logits: torch.Tensor, loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Forward KL(Target || Draft), mean over all supervised sampled positions.

    Targets are detached. Per-depth sums/counts are observations of independent
    parallel predictions, not decoder acceptance or a joint accepted prefix.
    The 1e-5 denominator matches the pinned Speculators loss reduction.
    """
    if (teacher_logits.ndim != 3 or teacher_logits.shape[0] != 1
            or teacher_logits.shape[2] != output.logits.shape[2]
            or loss_mask.shape != teacher_logits.shape[:2]
            or teacher_logits.shape[1] != int((output.depth == 0).sum())):
        raise ValueError("P-EAGLE teacher/mask must match the aligned row and full vocabulary")
    if not (teacher_logits.device == loss_mask.device == output.logits.device):
        raise ValueError("P-EAGLE loss tensors must share a device")
    if not ((loss_mask == 0) | (loss_mask == 1)).all():
        raise ValueError("P-EAGLE loss_mask must be binary")
    selected = output.valid & loss_mask[0, output.positions].bool()
    logits = output.logits[0, selected].float()
    targets = teacher_logits.detach()[0, output.positions[selected]].float()
    logq = F.log_softmax(logits, dim=-1)
    p = F.softmax(targets, dim=-1)
    terms = F.kl_div(logq, p, reduction="none").sum(-1)
    loss = terms.sum() / (selected.sum() + 1e-5)
    with torch.no_grad():
        d = output.depth[selected]
        counts = torch.zeros(output.num_depths, device=logits.device)
        sums, correct = torch.zeros_like(counts), torch.zeros_like(counts)
        counts.scatter_add_(0, d, torch.ones_like(terms))
        sums.scatter_add_(0, d, terms.detach())
        correct.scatter_add_(0, d, (logits.argmax(-1) == targets.argmax(-1)).float())
    return loss, {"kl_sum": sums, "count": counts, "correct": correct}
