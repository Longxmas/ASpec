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

"""DFlash draft model: block-diffusion drafter with dual-source KV injection.

Architecture overview:
  - W_proj projects concatenated multi-layer target hidden states into context features
  - Each decoder layer uses dual-source KV: context KV (from target) + draft KV (from draft)
  - Bidirectional attention within each block; no inter-block attention
  - Shared embedding and LM head from target model (frozen)
"""

import json
import os
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import PretrainedConfig, PreTrainedModel

from aspec.config.draft_attention import (
    resolve_draft_attention,
    resolve_draft_head_layout,
)
from aspec.config.draft_structure import resolve_draft_structure
from aspec.config.utils import resolve_dflash_rope_parameters, resolve_rope_theta

try:
    from quack import rmsnorm as _quack_rmsnorm
except ImportError:  # Optional pointwise-kernel dependency.
    _quack_rmsnorm = None


class DFlashConfig(PretrainedConfig):
    """Configuration for DFlash draft model."""

    model_type = "dflash"

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 1,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        vocab_size: int = 152064,
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 32768,
        rope_theta: float = 10000.0,
        num_target_layers: int = 5,
        target_hidden_size: int = 4096,
        target_num_hidden_layers: int = 36,
        target_layer_ids: Optional[List[int]] = None,
        fc_norm: bool = False,
        mask_token_id: int = 151669,
        tie_word_embeddings: bool = False,
        ffn_type: str = "dense",
        num_experts: int = 0,
        num_experts_per_tok: Optional[int] = None,
        moe_intermediate_size: Optional[int] = None,
        norm_topk_prob: bool = True,
        router_aux_loss_coef: Optional[float] = None,
        cudnn_gqa_mode: str = "auto",
        mtp_conditioning: Optional[dict] = None,
        **kwargs,
    ):
        # Transformers 5 validates YaRN parameters inside PretrainedConfig's
        # initializer and consults this field before returning to subclasses.
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        if cudnn_gqa_mode not in {"auto", "native", "external", "internal"}:
            raise ValueError(f"Unsupported cuDNN GQA mode: {cudnn_gqa_mode!r}")
        self.cudnn_gqa_mode = cudnn_gqa_mode
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.num_target_layers = num_target_layers
        self.target_hidden_size = target_hidden_size
        self.target_num_hidden_layers = target_num_hidden_layers
        self.target_layer_ids = target_layer_ids
        self.fc_norm = bool(fc_norm)
        self.mask_token_id = mask_token_id
        # K3 replaces this physical GQA/MHA layout with MLA-specific q/k/v
        # dimensions. Public DFlash/DSpark configs must otherwise fail while
        # loading, before a backend or any parameters are created.
        if getattr(type(self), "model_type", None) != "k3_dspark":
            resolve_draft_head_layout(
                {
                    "hidden_size": hidden_size,
                    "num_attention_heads": num_attention_heads,
                    "num_key_value_heads": num_key_value_heads,
                    "head_dim": kwargs.get("head_dim"),
                }
            )
        attention = resolve_draft_attention(
            {
                **kwargs,
                "num_hidden_layers": num_hidden_layers,
                "rope_theta": rope_theta,
            }
        )
        self.sliding_window = attention["sliding_window"]
        self.layer_types = attention["layer_types"]
        self.compress_ratios = attention["compress_ratios"]
        if attention["csa"] is not None:
            csa = attention["csa"]
            self.csa_training_stage = csa["training_stage"]
            self.index_topk = csa["index_topk"]
            self.index_n_heads = csa["index_n_heads"]
            self.index_head_dim = csa["index_head_dim"]
            self.csa_indexer_loss_coef = csa["indexer_loss_coef"]
            self.csa_workspace_bytes = csa["workspace_bytes"]
            self.compress_rope_theta = csa["compress_rope_theta"]
        if attention["qsa"] is not None:
            qsa = attention["qsa"]
            self.qsa_training_stage = qsa["training_stage"]
            self.qsa_token_budget = qsa["token_budget"]
            self.qsa_compress_ratio = qsa["compress_ratio"]
            self.qsa_index_n_heads = qsa["index_n_heads"]
            self.qsa_index_head_dim = qsa["index_head_dim"]
            self.qsa_rope_dim = qsa["rope_dim"]
            self.qsa_rope_theta = qsa["rope_theta"]
            self.qsa_indexer_loss_coef = qsa["indexer_loss_coef"]
            self.qsa_workspace_bytes = qsa["workspace_bytes"]
        if attention.get("dsa") is not None:
            if self.model_type in {"k3_dspark", "qwen3_dflash2"}:
                raise NotImplementedError("DSA is not yet integrated with materialized MLA or DFlash2")
            dsa = attention["dsa"]
            for field in ("training_stage", "token_budget", "index_n_heads", "index_head_dim",
                          "q_lora_rank", "rope_dim", "rope_theta", "indexer_loss_coef", "workspace_bytes"):
                setattr(self, f"dsa_{field}", dsa[field])
        structure = resolve_draft_structure({
            **kwargs, "ffn_type": ffn_type, "num_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "moe_intermediate_size": moe_intermediate_size, "norm_topk_prob": norm_topk_prob,
            "router_aux_loss_coef": router_aux_loss_coef, "num_hidden_layers": num_hidden_layers,
        })
        for key, value in structure.items():
            setattr(self, key, value)
        if self.model_type != "k3_dspark" and getattr(self, "attention_mode", "gqa") != "mla":
            resolve_dflash_rope_parameters(self)
        from aspec.config.mtp_conditioning import resolve_mtp_conditioning

        self.mtp_conditioning = resolve_mtp_conditioning({
            **self.to_dict(), "mtp_conditioning": mtp_conditioning,
        })


class DFlashRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if (
            _quack_rmsnorm is not None
            and hidden_states.is_cuda
            and not torch.compiler.is_compiling()
        ):
            return _quack_rmsnorm(hidden_states, self.weight, eps=self.variance_epsilon)
        return _dflash_rms_norm(hidden_states, self.weight, self.variance_epsilon)


class DFlashRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: float = 10000.0,
                 rope_parameters: dict | None = None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.rope_parameters = dict(rope_parameters or {"rope_type": "default", "rope_theta": base})
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Build on first use instead of allocating the native context length in
        # every layer.  Ling advertises 262K positions while the current recipe
        # commonly trains at 4K; eager per-layer caches otherwise waste 640 MiB.
        self.max_seq_len_cached = 0
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        # ``module.to(bfloat16)`` also casts non-persistent buffers. Recreate the
        # frequencies from the scalar contract so lazy cache construction keeps
        # the same FP32 phase accuracy as the former eager-before-cast path.
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim)
        )
        attention_factor = 1.0
        kind = self.rope_parameters["rope_type"]
        if kind != "default":
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

            # Call the fixed Transformers implementation with this layer's physical head width.
            config = PretrainedConfig(rope_parameters=dict(self.rope_parameters), max_position_embeddings=self.max_position_embeddings)
            config.head_dim = self.dim
            config.hidden_size = self.dim
            config.num_attention_heads = 1
            config.max_position_embeddings = self.max_position_embeddings
            inv_freq, attention_factor = ROPE_INIT_FUNCTIONS[kind](config, device=device)
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", (emb.cos() * attention_factor)[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * attention_factor)[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = self.max_position_embeddings
        if seq_len > self.max_seq_len_cached:
            # A small margin avoids rebuilding for adjacent shapes without
            # returning positions outside the requested sequence.
            self._set_cos_sin_cache(seq_len=seq_len + 20, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# NOTE: Not called in DFlashAttention.forward() (RoPE is applied inline there),
# but kept as a standard utility matching SpecForge's apply_rotary_pos_emb.
def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _dflash_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> torch.Tensor:
    """Reference RMSNorm math shared by eager and fused execution."""

    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return weight * hidden_states.to(input_dtype)


def _prepare_dflash_qk(
    query: torch.Tensor,
    context_key: torch.Tensor,
    draft_key: torch.Tensor,
    query_norm_weight: torch.Tensor,
    key_norm_weight: torch.Tensor,
    variance_epsilon: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query_position_ids: torch.Tensor,
    key_position_ids: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Join, normalize, and rotate Q/K with the original FP32 reduction math."""

    query = query.view(query.shape[0], query.shape[1], num_heads, head_dim)
    query = _dflash_rms_norm(query, query_norm_weight, variance_epsilon)
    key = torch.cat([context_key, draft_key], dim=1)
    key = key.view(key.shape[0], key.shape[1], num_kv_heads, head_dim)
    key = _dflash_rms_norm(key, key_norm_weight, variance_epsilon)

    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos_q = cos[query_position_ids].unsqueeze(2)
    sin_q = sin[query_position_ids].unsqueeze(2)
    query = (query * cos_q) + (_rotate_half(query) * sin_q)
    cos_k = cos[key_position_ids].unsqueeze(2)
    sin_k = sin[key_position_ids].unsqueeze(2)
    key = (key * cos_k) + (_rotate_half(key) * sin_k)
    return query, key


class DFlashAttention(nn.Module):
    """Dual-source KV attention for DFlash.

    K/V come from two sources concatenated along the sequence dimension:
      1. Context KV: projected from target model's context features (via shared W_k/W_v)
      2. Draft KV: projected from draft model's own hidden states (via same W_k/W_v)

    Q comes only from the draft model's hidden states.
    """

    def __init__(self, config: PretrainedConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        head_layout = resolve_draft_head_layout(
            {
                "hidden_size": self.hidden_size,
                "num_attention_heads": config.num_attention_heads,
                "num_key_value_heads": config.num_key_value_heads,
                "head_dim": getattr(config, "head_dim", None),
            }
        )
        self.num_heads = head_layout["query_heads"]
        self.head_dim = head_layout["head_dim"]
        self.num_kv_heads = head_layout["key_value_heads"]
        self.num_kv_groups = head_layout["query_heads_per_key_value_head"]
        self.head_layout = head_layout["type"]
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 32768)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Q-norm and K-norm (Qwen3 architecture requirement)
        self.q_norm = DFlashRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = DFlashRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.attention_type = config.layer_types[layer_idx]
        rope_base = (
            config.compress_rope_theta
            if self.attention_type == "compressed_sparse_attention"
            else resolve_rope_theta(config)
        )
        self.rotary_emb = DFlashRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=rope_base,
            rope_parameters=(None if self.attention_type == "compressed_sparse_attention"
                             else resolve_dflash_rope_parameters(config)),
        )
        self.csa_layer = None
        if self.attention_type == "compressed_sparse_attention":
            from aspec.models.draft.csa import CSALayer

            self.csa_layer = CSALayer(config, layer_idx)
        self.qsa_layer = None
        if self.attention_type == "qwen_sparse_attention":
            from aspec.models.draft.qsa import QSALayer

            self.qsa_layer = QSALayer(config, layer_idx)
        self.dsa_layer = None
        if self.attention_type == "deepseek_sparse_attention":
            from aspec.models.draft.dsa import DSALayer

            self.dsa_layer = DSALayer(config, layer_idx)
        self.csa_indexer_loss_per_query = None
        self.qsa_indexer_loss_per_query = None
        self.dsa_indexer_loss_per_query = None
        self.attention_observation = None
        self.cudnn_gqa_mode = getattr(config, "cudnn_gqa_mode", "auto")
        self.attention_backend = "flex_attention"
        self.qk_prepare_backend = "eager"

    def set_attention_backend(self, attention_backend: str) -> None:
        if self.dsa_layer is not None and attention_backend != "flex_attention":
            raise NotImplementedError("DSA uses flex_attention dense execution or the fa4 THD path")
        if self.csa_layer is not None and attention_backend != "flex_attention":
            raise NotImplementedError(
                "CSA Draft currently requires attention_backend=flex_attention; "
                "cuDNN/FA4 and Direct CP need separate arbitrary-index validation"
            )
        if (
            self.qsa_layer is not None
            and self.qsa_layer.stage == "sparse"
            and attention_backend != "flex_attention"
        ):
            raise NotImplementedError(
                "sparse QSA Draft currently requires attention_backend=flex_attention"
            )
        if attention_backend not in {"flex_attention", "cudnn_flex_attention"}:
            raise ValueError(
                "DFlash GQA attention supports flex_attention and "
                f"cudnn_flex_attention, got {attention_backend!r}"
            )
        if attention_backend == "cudnn_flex_attention" and self.head_dim != 128:
            raise NotImplementedError(
                "DFlash cuDNN FlexAttention currently requires head_dim=128, "
                f"got {self.head_dim}"
            )
        self.attention_backend = attention_backend

    def forward(
        self,
        draft_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
        direct_cp_metadata=None,
        csa_metadata=None,
        csa_prepared=None,
        csa_block_mask=None,
        qsa_metadata=None,
        qsa_layout=None,
        dsa_metadata=None,
        dsa_layout=None,
    ) -> torch.Tensor:
        """Forward pass with dual-source KV.

        Args:
            draft_hidden: [B, draft_len, D] — hidden states of draft tokens
            context_hidden: [B, ctx_len, D] — context features from target
            draft_position_ids: [B, draft_len] — position IDs for draft tokens
            context_position_ids: [B, ctx_len] — position IDs for context tokens
            block_mask: FlexAttention BlockMask for block-causal attention
        """
        bsz, draft_len, _ = draft_hidden.shape
        ctx_len = context_hidden.shape[1]
        if (
            self.attention_backend == "cudnn_flex_attention"
            and block_mask is None
            and direct_cp_metadata is None
        ):
            raise RuntimeError(
                "attention_backend=cudnn_flex_attention requires a CUDA cuDNN mask; "
                "the backend cannot fall back to unmasked SDPA"
            )

        # Q only from draft
        q = self.q_proj(draft_hidden)

        # K/V from both context and draft (shared projections)
        k_ctx = self.k_proj(context_hidden)
        v_ctx = self.v_proj(context_hidden)
        k_draft = self.k_proj(draft_hidden)
        v_draft = self.v_proj(draft_hidden)

        # V is consumed directly. The regular GQA path defers the K join and
        # Q/K norms to one preparation region below. CSA/QSA need normalized
        # Q/K earlier and retain the original eager sequence.
        v = torch.cat([v_ctx, v_draft], dim=1)

        total_len = ctx_len + draft_len
        rotary_len = total_len
        if direct_cp_metadata is not None:
            # Direct CP keeps only an owner-local context/query shard, while
            # RoPE ids remain global (or document-local under packing).  The
            # local tensor length is therefore not a valid cache bound.  Use
            # the known global sequence bound plus one complete Draft block;
            # this avoids a device sync to inspect position-id maxima.
            sequence_length = direct_cp_metadata.sequence_length
            if sequence_length is None:
                raise ValueError("Direct CP sparse attention requires sequence_length")
            rotary_len = max(
                rotary_len,
                sequence_length + direct_cp_metadata.block_size,
            )
        v = v.view(bsz, total_len, self.num_kv_heads, self.head_dim)

        if self.csa_layer is not None or self.qsa_layer is not None or self.dsa_layer is not None:
            q = q.view(bsz, draft_len, self.num_heads, self.head_dim)
            q = self.q_norm(q)
            # Preserve the SpecForge order: concatenate context/draft K, then normalize.
            k = torch.cat([k_ctx, k_draft], dim=1)
            k = k.view(bsz, total_len, self.num_kv_heads, self.head_dim)
            k = self.k_norm(k)

        if self.csa_layer is not None:
            if csa_metadata is None:
                raise ValueError("CSA Draft requires anchor/document metadata")
            if direct_cp_metadata is None and self.attention_backend != "flex_attention":
                raise RuntimeError("CSA Draft reached an unsupported attention backend")

            # A CSA layer uses its compression RoPE for the layer-local SWA path
            # and for compressed entries, matching the official architecture.
            full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)
            cos, sin = self.rotary_emb(q, seq_len=rotary_len)
            cos = cos.to(q.device)
            sin = sin.to(q.device)
            cos_q = cos.squeeze(1).squeeze(0)[draft_position_ids].unsqueeze(2)
            sin_q = sin.squeeze(1).squeeze(0)[draft_position_ids].unsqueeze(2)
            q = (q * cos_q) + (_rotate_half(q) * sin_q)
            cos_k = cos.squeeze(1).squeeze(0)[full_position_ids].unsqueeze(2)
            sin_k = sin.squeeze(1).squeeze(0)[full_position_ids].unsqueeze(2)
            k = (k * cos_k) + (_rotate_half(k) * sin_k)

            (
                compressed_key,
                compressed_value,
                compressed_visible,
                indexer_loss,
                observation,
            ) = self.csa_layer(
                draft_hidden=draft_hidden,
                context_hidden=context_hidden,
                draft_position_ids=draft_position_ids,
                context_position_ids=context_position_ids,
                metadata=csa_metadata,
                main_query=q,
                scale=self.head_dim**-0.5,
                direct_cp_metadata=direct_cp_metadata,
                **(
                    {
                        "layout": csa_prepared[self.csa_layer.ratio][0],
                        "eligible": csa_prepared[self.csa_layer.ratio][1],
                    }
                    if csa_prepared is not None else {}
                ),
            )
            if direct_cp_metadata is not None:
                if self.csa_layer.stage == "sparse":
                    from aspec.models.draft.csa import (
                        csa_distributed_sparse_attention,
                    )

                    attn_output = csa_distributed_sparse_attention(
                        query_own=q,
                        raw_key=k,
                        raw_value=v,
                        compressed_key=compressed_key,
                        compressed_value=compressed_value,
                        local_selected=compressed_visible,
                        metadata=csa_metadata,
                        layout=csa_prepared[self.csa_layer.ratio][0],
                        direct_cp_metadata=direct_cp_metadata,
                        scale=self.head_dim**-0.5,
                        workspace_bytes=self.csa_layer.workspace_bytes,
                    )
                else:
                    from aspec.models.draft.csa import (
                        csa_direct_dense_attention,
                    )

                    attn_output = csa_direct_dense_attention(
                        query_own=q,
                        raw_key=k,
                        raw_value=v,
                        compressed_key=compressed_key,
                        compressed_value=compressed_value,
                        metadata=csa_metadata,
                        layout=csa_prepared[self.csa_layer.ratio][0],
                        direct_cp_metadata=direct_cp_metadata,
                        scale=self.head_dim**-0.5,
                    )
                observation.update(
                    qkv_format="thd",
                    fa4_mode=(
                        "varlen_selected" if self.csa_layer.stage == "sparse" else "fixed"
                    ),
                    backward_strategy=(
                        "fa4_varlen_selected"
                        if self.csa_layer.stage == "sparse"
                        else "fa4_fixed_sparse"
                    ),
                    empty_local_partition=observation.get("empty_local_partition", False),
                    context_parallel_size=direct_cp_metadata.cp_size,
                    context_parallel_rank=direct_cp_metadata.cp_rank,
                    backward_observed=False,
                )

                def observe_backward(gradient):
                    observation["backward_observed"] = True
                    return gradient

                if attn_output.requires_grad:
                    attn_output.register_hook(observe_backward)
                self.csa_indexer_loss_per_query = indexer_loss
                self.attention_observation = observation
                return self.o_proj(
                    attn_output.contiguous().reshape(
                        bsz, draft_len, self.num_heads * self.head_dim
                    )
                )
            if compressed_key.shape[1]:
                k = torch.cat((k, compressed_key), dim=1)
                v = torch.cat((v, compressed_value), dim=1)
            if csa_block_mask is None:
                from aspec.models.draft.csa import build_csa_block_mask

                block_mask = build_csa_block_mask(
                    metadata=csa_metadata,
                    compressed_visible=compressed_visible,
                    context_length=ctx_len,
                    draft_length=draft_len,
                    device=q.device,
                )
            else:
                block_mask = csa_block_mask
            if block_mask is None:
                raise RuntimeError("CSA Draft requires an effective local/compressed mask")
            from aspec.models.ops.flex_attention import compile_friendly_flex_attention

            attn_output = compile_friendly_flex_attention(
                query=q.transpose(1, 2),
                key=k.transpose(1, 2),
                value=v.transpose(1, 2),
                block_mask=block_mask,
                enable_gqa=True,
            )
            self.csa_indexer_loss_per_query = indexer_loss
            self.attention_observation = observation
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(
                bsz, draft_len, self.num_heads * self.head_dim
            )
            return self.o_proj(attn_output)

        if self.qsa_layer is not None or self.dsa_layer is not None:
            selector_layer = self.dsa_layer if self.dsa_layer is not None else self.qsa_layer
            selector_metadata = dsa_metadata if self.dsa_layer is not None else qsa_metadata
            selector_layout = dsa_layout if self.dsa_layer is not None else qsa_layout
            selector_name = "DSA" if self.dsa_layer is not None else "QSA"
            if selector_metadata is None:
                raise ValueError(f"{selector_name} Draft requires anchor/document metadata")
            if selector_layer.stage != "dense" and selector_layout is None:
                raise ValueError(f"{selector_name} indexer stages require a token layout")

            full_position_ids = torch.cat(
                [context_position_ids, draft_position_ids], dim=1
            )
            cos, sin = self.rotary_emb(q, seq_len=rotary_len)
            cos = cos.to(q.device)
            sin = sin.to(q.device)
            cos_q = cos.squeeze(1).squeeze(0)[draft_position_ids].unsqueeze(2)
            sin_q = sin.squeeze(1).squeeze(0)[draft_position_ids].unsqueeze(2)
            q = (q * cos_q) + (_rotate_half(q) * sin_q)
            cos_k = cos.squeeze(1).squeeze(0)[full_position_ids].unsqueeze(2)
            sin_k = sin.squeeze(1).squeeze(0)[full_position_ids].unsqueeze(2)
            k = (k * cos_k) + (_rotate_half(k) * sin_k)
            selected, indexer_loss, observation = selector_layer(
                draft_hidden=draft_hidden,
                context_hidden=context_hidden,
                draft_position_ids=draft_position_ids,
                metadata=selector_metadata,
                layout=selector_layout,
                main_query=q,
                main_key=k,
                scale=self.head_dim**-0.5,
                direct_cp_metadata=direct_cp_metadata,
            )
            if selector_layer.stage == "sparse":
                if direct_cp_metadata is None and self.attention_backend != "flex_attention":
                    raise RuntimeError("sparse QSA reached an unsupported attention backend")
                if self.dsa_layer is not None:
                    from aspec.models.draft.dsa import dsa_sparse_attention

                    attn_output = dsa_sparse_attention(
                        query=q, key=k, value=v, metadata=selector_metadata,
                        layout=selector_layout, selected=selected,
                        scale=self.head_dim**-0.5,
                        workspace_bytes=selector_layer.indexer.workspace_bytes,
                        direct_cp_metadata=direct_cp_metadata,
                        observation=observation,
                    )
                elif direct_cp_metadata is not None:
                    from aspec.models.draft.qsa import (
                        qsa_distributed_sparse_attention,
                    )

                    attn_output = qsa_distributed_sparse_attention(
                        query_own=q,
                        key=k,
                        value=v,
                        metadata=qsa_metadata,
                        layout=qsa_layout,
                        local_selected=selected,
                        direct_cp_metadata=direct_cp_metadata,
                        scale=self.head_dim**-0.5,
                        workspace_bytes=self.qsa_layer.indexer.workspace_bytes,
                    )
                else:
                    from aspec.models.draft.qsa import qsa_sparse_attention

                    attn_output = qsa_sparse_attention(
                        query=q,
                        key=k,
                        value=v,
                        metadata=qsa_metadata,
                        layout=qsa_layout,
                        selected=selected,
                        scale=self.head_dim**-0.5,
                        workspace_bytes=self.qsa_layer.indexer.workspace_bytes,
                    )
                if direct_cp_metadata is not None:
                    observation.update(
                        qkv_format="thd",
                        fa4_mode="varlen_selected",
                        backward_strategy="fa4_varlen_selected",
                        empty_local_partition=observation.get("empty_local_partition", False),
                        context_parallel_size=direct_cp_metadata.cp_size,
                        context_parallel_rank=direct_cp_metadata.cp_rank,
                        backward_observed=False,
                    )

                    def observe_backward(gradient):
                        observation["backward_observed"] = True
                        return gradient

                    if attn_output.requires_grad:
                        attn_output.register_hook(observe_backward)
                if self.dsa_layer is not None:
                    self.dsa_indexer_loss_per_query = indexer_loss
                else:
                    self.qsa_indexer_loss_per_query = indexer_loss
                self.attention_observation = observation
                return self.o_proj(
                    attn_output.contiguous().reshape(
                        bsz, draft_len, self.num_heads * self.head_dim
                    )
                )
            if direct_cp_metadata is not None:
                if block_mask is not None:
                    raise ValueError(
                        "QSA Direct CP attention does not accept a FlexAttention mask"
                    )
                from aspec.models.ops.direct_cp import direct_cp_attention_from_kv

                attn_output, _ = direct_cp_attention_from_kv(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    direct_cp_metadata,
                    scale=self.head_dim**-0.5,
                    cudnn_gqa_mode=self.cudnn_gqa_mode,
                )
                attn_output = attn_output.transpose(1, 2)
                from aspec.models.ops.direct_cp import (
                    direct_cp_attention_observation,
                )

                observation.update(
                    direct_cp_attention_observation(direct_cp_metadata)
                )
                # This is a merged layer observation, not the metadata's
                # original dict. Observe the real tensor backward here too.
                if attn_output.requires_grad:
                    def observe_fixed_backward(gradient):
                        observation["backward_observed"] = True
                        return gradient

                    attn_output.register_hook(observe_fixed_backward)
            elif block_mask is None:
                raise RuntimeError("QSA Draft requires an effective attention mask")
            elif self.attention_backend == "cudnn_flex_attention":
                from aspec.models.ops.cudnn_flex_attention import cudnn_flex_attention

                attn_output = cudnn_flex_attention(
                    q,
                    k,
                    v,
                    mask=block_mask,
                    scale=self.head_dim**-0.5,
                    pack_gqa=False,
                    physical_gqa=True,
                    gqa_mode=self.cudnn_gqa_mode,
                )
            else:
                from aspec.models.ops.flex_attention import compile_friendly_flex_attention

                attn_output = compile_friendly_flex_attention(
                    query=q.transpose(1, 2),
                    key=k.transpose(1, 2),
                    value=v.transpose(1, 2),
                    block_mask=block_mask,
                    enable_gqa=True,
                ).transpose(1, 2)
            if self.dsa_layer is not None:
                self.dsa_indexer_loss_per_query = indexer_loss
            else:
                self.qsa_indexer_loss_per_query = indexer_loss
            self.attention_observation = observation
            return self.o_proj(
                attn_output.contiguous().reshape(
                    bsz, draft_len, self.num_heads * self.head_dim
                )
            )

        use_cudnn_flex = (
            block_mask is not None
            and self.attention_backend == "cudnn_flex_attention"
        )
        rope_len = total_len
        if direct_cp_metadata is not None:
            rope_len = max(total_len, direct_cp_metadata.sequence_length or total_len)
        cos, sin = self.rotary_emb(q, seq_len=rope_len)
        cos = cos.to(q.device)
        sin = sin.to(q.device)

        use_dspark_qk_kernel = False
        if (
            (use_cudnn_flex or (
                direct_cp_metadata is not None
                and direct_cp_metadata.backend == "cudnn_flex_attention"
            ))
            and q.is_cuda
            and q.dtype in (torch.bfloat16, torch.float16)
            and self.q_norm.weight.dtype == q.dtype
            and self.k_norm.weight.dtype == q.dtype
            and not torch.compiler.is_compiling()
        ):
            from aspec.models.ops.dspark_qk_norm_rope import (
                dspark_qk_norm_rope_available,
            )

            use_dspark_qk_kernel = dspark_qk_norm_rope_available()

        if use_dspark_qk_kernel:
            from aspec.models.ops.dspark_qk_norm_rope import (
                dspark_qk_norm_rope,
            )

            q, k = dspark_qk_norm_rope(
                q,
                k_ctx,
                k_draft,
                self.q_norm.weight,
                self.k_norm.weight,
                self.q_norm.variance_epsilon,
                cos,
                sin,
                draft_position_ids,
                context_position_ids,
                draft_position_ids,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
            )
            self.qk_prepare_backend = "aspec_cutedsl"
        else:
            full_position_ids = torch.cat(
                [context_position_ids, draft_position_ids], dim=1
            )
            q, k = _prepare_dflash_qk(
                q,
                k_ctx,
                k_draft,
                self.q_norm.weight,
                self.k_norm.weight,
                self.q_norm.variance_epsilon,
                cos,
                sin,
                draft_position_ids,
                full_position_ids,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
            )
            self.qk_prepare_backend = "eager"

        if not use_cudnn_flex:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

        if direct_cp_metadata is not None:
            if block_mask is not None:
                raise ValueError(
                    "Direct CP attention does not accept a FlexAttention block mask"
                )
            expected_context_positions = (
                direct_cp_metadata.context_rope_position_ids
                if direct_cp_metadata.context_rope_position_ids is not None
                else direct_cp_metadata.context_position_ids
            )
            if not torch.equal(
                context_position_ids,
                expected_context_positions.to(context_position_ids.device),
            ):
                raise ValueError(
                    "context_position_ids must match Direct CP RoPE metadata"
                )
            from aspec.models.ops.direct_cp import direct_cp_attention_from_kv

            # KV is already concatenated, normalized and rotated above.
            attn_output, _ = direct_cp_attention_from_kv(
                q,
                k,
                v,
                direct_cp_metadata,
                scale=self.head_dim**-0.5,
                cudnn_gqa_mode=self.cudnn_gqa_mode,
            )
            # Keep the actual kernel observation on the module: FSDP may replace
            # tensor-bearing metadata kwargs, so mutating the caller's object
            # is not a reliable cross-layer observation channel.
            from aspec.models.ops.direct_cp import direct_cp_attention_observation

            self.attention_observation = direct_cp_attention_observation(
                direct_cp_metadata
            )
        elif use_cudnn_flex:
            from aspec.models.ops.cudnn_flex_attention import cudnn_flex_attention

            attn_output = cudnn_flex_attention(
                q,
                k,
                v,
                mask=block_mask,
                scale=self.head_dim**-0.5,
                pack_gqa=False,
                physical_gqa=True,
                gqa_mode=self.cudnn_gqa_mode,
            )
        elif block_mask is not None:
            from aspec.models.ops.flex_attention import compile_friendly_flex_attention

            # Use enable_gqa=True to let FlexAttention handle GQA internally
            # instead of materializing expanded KV via _repeat_kv
            attn_output = compile_friendly_flex_attention(
                query=q,
                key=k,
                value=v,
                block_mask=block_mask,
                enable_gqa=True,
            )
        else:
            # Fallback: bidirectional attention (no mask) — expand KV for SDPA
            k = _repeat_kv(k, self.num_kv_groups)
            v = _repeat_kv(v, self.num_kv_groups)
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=0.0)

        if not use_cudnn_flex:
            attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, draft_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class DFlashMLP(nn.Module):
    def __init__(self, config: PretrainedConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        from transformers.activations import ACT2FN

        self.act_fn = ACT2FN[getattr(config, "hidden_act", "silu")]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DFlashDecoderLayer(nn.Module):
    """Single transformer decoder layer for DFlash draft model."""

    # Subclasses (e.g. K3 MLA) swap the attention implementation only.
    attention_class = DFlashAttention

    def __init__(self, config: PretrainedConfig, layer_idx: int = 0):
        super().__init__()
        self.self_attn = self.attention_class(config, layer_idx=layer_idx)
        self.is_moe = getattr(config, "ffn_type", "dense") != "dense"
        if getattr(config, "ffn_type", "dense") == "ling_moe":
            from aspec.models.draft.ling_moe import LingDraftMoE

            self.mlp = LingDraftMoE(config, layer_idx)
        elif self.is_moe:
            from aspec.models.draft.moe import QwenDraftMoE

            self.mlp = QwenDraftMoE(config)
        else:
            self.mlp = DFlashMLP(config)
        self.input_layernorm = DFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        draft_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
        direct_cp_metadata=None,
        csa_metadata=None,
        csa_prepared=None,
        csa_block_mask=None,
        qsa_metadata=None,
        qsa_layout=None,
        dsa_metadata=None,
        dsa_layout=None,
    ) -> torch.Tensor:
        residual = draft_hidden
        draft_hidden = self.input_layernorm(draft_hidden)

        csa_kwargs = {}
        if hasattr(self.self_attn, "csa_layer"):
            csa_kwargs = {
                "csa_prepared": csa_prepared,
                "csa_block_mask": csa_block_mask,
            }
        qsa_kwargs = {}
        if hasattr(self.self_attn, "qsa_layer"):
            qsa_kwargs = {
                "qsa_metadata": qsa_metadata,
                "qsa_layout": qsa_layout,
            }
        if hasattr(self.self_attn, "dsa_layer"):
            qsa_kwargs.update(dsa_metadata=dsa_metadata, dsa_layout=dsa_layout)
        draft_hidden = self.self_attn(
            draft_hidden=draft_hidden,
            context_hidden=context_hidden,
            draft_position_ids=draft_position_ids,
            context_position_ids=context_position_ids,
            block_mask=block_mask,
            direct_cp_metadata=direct_cp_metadata,
            csa_metadata=csa_metadata,
            **csa_kwargs,
            **qsa_kwargs,
        )
        draft_hidden = residual + draft_hidden

        residual = draft_hidden
        draft_hidden = self.post_attention_layernorm(draft_hidden)
        router_logits = None
        if self.is_moe:
            draft_hidden, router_logits = self.mlp(draft_hidden)
        else:
            draft_hidden = self.mlp(draft_hidden)
        draft_hidden = residual + draft_hidden

        return (draft_hidden, router_logits) if self.is_moe else draft_hidden


def build_target_layer_ids(num_target_layers: int, num_hidden_layers: int) -> List[int]:
    """Compute uniformly spaced layer IDs from the target model.

    Matches SpecForge's build_target_layer_ids() exactly:
      start = 1, end = num_hidden_layers - 3, span = end - start
      For num_target_layers=5 and num_hidden_layers=36:
        start=1, end=33, span=32
        → [1, 9, 17, 25, 33]

    Note: SpecForge convention — num_target_layers here is the number of layers
    to capture (called num_draft_layers in SpecForge). num_hidden_layers is the
    total number of target model decoder layers.
    """
    if num_target_layers == 1:
        return [num_hidden_layers // 2]
    start = 1
    end = num_hidden_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_target_layers - 1))) for i in range(num_target_layers)
    ]


class DFlashDraftModel(PreTrainedModel):
    """DFlash draft model with dual-source KV injection.

    Trainable parameters:
      - W_proj: Linear(num_target_layers * target_hidden_size, hidden_size)
      - proj_norm: RMSNorm after projection
      - N decoder layers (each with attention + FFN)

    Frozen (from target):
      - embed_tokens: token embedding
      - LM head is external (loaded separately in trainer)
    """

    config_class = DFlashConfig
    supports_gradient_checkpointing = True
    decoder_layer_class = DFlashDecoderLayer
    # See Eagle3DraftModel.all_tied_weights_keys.
    all_tied_weights_keys = {}

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers

        target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)
        num_target_layers = getattr(config, "num_target_layers", 5)
        self.num_target_layers = num_target_layers
        self.mask_token_id = getattr(config, "mask_token_id", 151669)

        # Target layer IDs for hidden state extraction
        target_num_hidden = getattr(config, "target_num_hidden_layers", 36)
        self.target_layer_ids = getattr(config, "target_layer_ids", None)
        if self.target_layer_ids is None:
            self.target_layer_ids = build_target_layer_ids(num_target_layers, target_num_hidden)

        # Context feature projection: concat(multi-layer hidden) → hidden_size
        proj_input_dim = num_target_layers * target_hidden_size
        if getattr(config, "fc_norm", False):
            self.fc_norm = nn.ModuleList(
                [
                    DFlashRMSNorm(target_hidden_size, eps=config.rms_norm_eps)
                    for _ in range(num_target_layers)
                ]
            )
        else:
            self.fc_norm = None
        self.context_proj = nn.Linear(proj_input_dim, self.hidden_size, bias=False)
        self.context_norm = DFlashRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        # Token embedding (shared from target, frozen)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Decoder layers
        self.layers = nn.ModuleList(
            [self.decoder_layer_class(config, layer_idx=i) for i in range(self.num_layers)]
        )

        # Final norm
        self.final_norm = DFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mtp_fusion = self.mtp_source = self.mtp_binding = None
        if getattr(config, "mtp_conditioning", None):
            from aspec.models.dmtp import FrozenMTPFeatures, MTPAnchorFusion

            self.freeze_embedding()
            if config.mtp_conditioning.get("feature_source", "draft") == "draft":
                self.mtp_source = FrozenMTPFeatures(config.mtp_conditioning["native_config"])
            else:
                from aspec.models.dmtp import FrozenMTPBinding

                self.mtp_binding = FrozenMTPBinding(config.mtp_conditioning["native_config"])
            if config.mtp_conditioning["mode"] == "hidden":
                self.mtp_fusion = MTPAnchorFusion(config.hidden_size, target_hidden_size,
                                                  eps=config.rms_norm_eps)
        self.last_csa_indexer_losses = ()
        self.last_qsa_indexer_losses = ()
        self.last_dsa_indexer_losses = ()
        self.last_attention_observations = ()
        self.configure_csa_trainable_parameters()
        self.configure_qsa_trainable_parameters()
        self.configure_dsa_trainable_parameters()

    def initialize_mtp_conditioning(self, target_model_path):
        if self.mtp_source is not None:
            self.mtp_source.initialize_native_weights(target_model_path)
        if self.mtp_binding is not None:
            self.mtp_binding.initialize_native_weights(target_model_path)

    @torch.no_grad()
    def validate_mtp_binding(self, target_model_path, embedding_key):
        if not getattr(self.config, "mtp_conditioning", None):
            return
        from transformers import AutoConfig

        from aspec.config.mtp_conditioning import validate_mtp_target_config
        from aspec.models.draft.base import load_tensor_from_pretrained

        validate_mtp_target_config(
            self.config.mtp_conditioning["native_config"],
            AutoConfig.from_pretrained(target_model_path).to_dict(),
        )

        actual = self.embed_tokens.weight
        if hasattr(actual, "full_tensor"):
            actual = actual.full_tensor()
        expected = load_tensor_from_pretrained(target_model_path, embedding_key)
        if actual.shape != expected.shape:
            raise ValueError("MTP conditioning embedding shape differs from its Target")
        for start in range(0, expected.shape[0], 1024):
            if not torch.equal(actual[start:start + 1024].cpu(),
                               expected[start:start + 1024].to(dtype=actual.dtype)):
                raise ValueError("MTP conditioning frozen embedding differs from its Target")
        if self.mtp_source is not None:
            self.mtp_source.validate_native_weights(target_model_path)
        if self.mtp_binding is not None:
            self.mtp_binding.validate_native_weights(target_model_path)

    def configure_dsa_trainable_parameters(self) -> None:
        layers = [layer.self_attn.dsa_layer for layer in self.layers
                  if getattr(layer.self_attn, "dsa_layer", None) is not None]
        if not layers:
            return
        if self.config.dsa_training_stage == "indexer_warmup":
            for parameter in self.parameters():
                parameter.requires_grad = False
        for layer in layers:
            for parameter in layer.indexer.parameters():
                parameter.requires_grad = (
                    self.config.dsa_training_stage != "dense" and self.config.dsa_indexer_loss_coef > 0
                )

    def configure_csa_trainable_parameters(self) -> None:
        """Apply the phase-specific CSA freeze policy to the complete model."""

        csa_layers = [
            layer.self_attn.csa_layer
            for layer in self.layers
            if getattr(layer.self_attn, "csa_layer", None) is not None
        ]
        if csa_layers and self.config.csa_training_stage == "dense":
            for csa_layer in csa_layers:
                for parameter in csa_layer.indexer.parameters():
                    parameter.requires_grad = False
        elif csa_layers and self.config.csa_training_stage == "indexer_warmup":
            for parameter in self.parameters():
                parameter.requires_grad = False
            for csa_layer in csa_layers:
                csa_layer.configure_trainable_parameters()

    def configure_qsa_trainable_parameters(self) -> None:
        """Apply Qwen's dense/warm-up/sparse phase trainability boundary."""

        qsa_layers = [
            layer.self_attn.qsa_layer
            for layer in self.layers
            if getattr(layer.self_attn, "qsa_layer", None) is not None
        ]
        if qsa_layers and self.config.qsa_training_stage == "dense":
            for qsa_layer in qsa_layers:
                for parameter in qsa_layer.indexer.parameters():
                    parameter.requires_grad = False
        elif qsa_layers and self.config.qsa_training_stage == "indexer_warmup":
            for parameter in self.parameters():
                parameter.requires_grad = False
            for qsa_layer in qsa_layers:
                qsa_layer.configure_trainable_parameters()

    def set_attention_backend(self, attention_backend: str) -> None:
        for layer in self.layers:
            layer.self_attn.set_attention_backend(attention_backend)

    def keep_csa_projection_precision(self) -> None:
        """Restore the FP32 compressor projections used by official CSA training."""

        for layer in self.layers:
            csa_layer = getattr(layer.self_attn, "csa_layer", None)
            if csa_layer is not None:
                csa_layer.keep_projection_fp32()

    def extract_context_feature(
        self, all_hidden_states: List[torch.Tensor] | torch.Tensor
    ) -> torch.Tensor:
        """Extract and project context features from target hidden states.

        Args:
            all_hidden_states: concatenated [B, seq_len, K*D] target features
                or a list of K [B, seq_len, D] tensors.

        Returns:
            context_feature: [B, seq_len, hidden_size]
        """
        if isinstance(all_hidden_states, torch.Tensor) and self.fc_norm is None:
            concatenated = all_hidden_states
        else:
            if isinstance(all_hidden_states, torch.Tensor):
                per_layer_dim = all_hidden_states.shape[-1] // self.num_target_layers
                all_hidden_states = all_hidden_states.split(per_layer_dim, dim=-1)
            if self.fc_norm is not None:
                all_hidden_states = [
                    norm(hidden_states)
                    for norm, hidden_states in zip(self.fc_norm, all_hidden_states, strict=True)
                ]
            concatenated = torch.cat(all_hidden_states, dim=-1)
        concatenated = concatenated.to(self.context_proj.weight.dtype)
        projected = self.context_proj(concatenated)
        return self.context_norm(projected)

    def forward(
        self,
        draft_input_ids: Optional[torch.Tensor],
        context_feature: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
        noise_embedding: Optional[torch.Tensor] = None,
        direct_cp_metadata=None,
        csa_metadata=None,
        qsa_metadata=None,
        dsa_metadata=None,
        return_router_logits: bool = False,
    ) -> torch.Tensor:
        """Forward pass through draft model.

        Args:
            draft_input_ids: [B, draft_len] — token IDs (anchor + MASK tokens).
                Ignored if noise_embedding is provided.
            context_feature: [B, ctx_len, D] — projected context from target
            draft_position_ids: [B, draft_len]
            context_position_ids: [B, ctx_len]
            block_mask: FlexAttention BlockMask, or a per-layer sequence of
                BlockMasks for mixed attention schedules
            noise_embedding: [B, draft_len, D] — pre-computed embeddings (from training wrapper)

        Returns:
            hidden_states: [B, draft_len, D] — pre-norm hidden states
        """
        if noise_embedding is not None:
            draft_hidden = noise_embedding.to(context_feature.dtype)
        else:
            draft_hidden = self.embed_tokens(draft_input_ids).to(context_feature.dtype)

        router_logits = []
        csa_indexer_losses = []
        qsa_indexer_losses = []
        dsa_indexer_losses = []
        attention_observations = []
        csa_prepared = None
        csa_shared_block_masks = {}
        if csa_metadata is not None:
            from aspec.models.draft.csa import (
                build_csa_block_mask,
                build_csa_compression_layout,
                csa_eligible_groups,
            )

            csa_prepared = {}
            ratios = {
                layer.self_attn.csa_layer.ratio
                for layer in self.layers
                if layer.self_attn.csa_layer is not None
            }
            for ratio in ratios:
                layout = build_csa_compression_layout(
                    csa_metadata, context_position_ids, ratio
                )
                eligible = (
                    None
                    if direct_cp_metadata is not None
                    else csa_eligible_groups(csa_metadata, layout)
                )
                csa_prepared[ratio] = (layout, eligible)
                if (
                    self.config.csa_training_stage != "sparse"
                    and direct_cp_metadata is None
                ):
                    csa_shared_block_masks[ratio] = build_csa_block_mask(
                        metadata=csa_metadata,
                        compressed_visible=eligible,
                        context_length=context_feature.shape[1],
                        draft_length=draft_hidden.shape[1],
                        device=draft_hidden.device,
                    )
        qsa_layout = None
        if (
            qsa_metadata is not None
            and self.config.qsa_training_stage != "dense"
        ):
            from aspec.models.draft.qsa import build_qsa_compression_layout

            qsa_layout = build_qsa_compression_layout(
                qsa_metadata,
                context_position_ids,
                self.config.qsa_compress_ratio,
            )
        dsa_layout = None
        if dsa_metadata is not None and self.config.dsa_training_stage != "dense":
            from aspec.models.draft.dsa import build_dsa_token_layout

            dsa_layout = build_dsa_token_layout(
                dsa_metadata, context_position_ids, draft_position_ids,
                rope_dim=self.config.dsa_rope_dim, rope_theta=self.config.dsa_rope_theta,
            )
        for layer_id, layer in enumerate(self.layers):
            layer_block_mask = block_mask
            if isinstance(block_mask, (list, tuple)):
                layer_block_mask = block_mask[layer_id]
            layer_metadata = (
                direct_cp_metadata[layer_id]
                if isinstance(direct_cp_metadata, (list, tuple)) else direct_cp_metadata
            )
            window = (self.config.sliding_window
                      if self.config.layer_types[layer_id] == "sliding_attention" else None)
            if (
                window is not None
                and layer_block_mask is None
                and layer_metadata is None
                and csa_metadata is None
            ):
                raise ValueError("SWA Draft requires an effective attention mask or Direct CP metadata")
            if layer_metadata is not None:
                if self.config.model_type == "qwen3_dflash2":
                    causal = getattr(self.config, "is_causal", None)
                    expected_causal = window is not None if causal is None else bool(causal)
                    if (layer_metadata.context_window is not None
                            or layer_metadata.query_window != window
                            or layer_metadata.draft_is_causal != expected_causal):
                        raise ValueError("Direct CP query visibility disagrees with the DFlash2 layer policy")
                elif layer_metadata.context_window != window:
                    raise ValueError("Direct CP context window disagrees with the Draft layer policy")
            csa_layer = getattr(layer.self_attn, "csa_layer", None)
            layer_forward = (partial(self._gradient_checkpointing_func, layer)
                             if self.gradient_checkpointing and self.training else layer)
            draft_hidden = layer_forward(
                draft_hidden=draft_hidden,
                context_hidden=context_feature,
                draft_position_ids=draft_position_ids,
                context_position_ids=context_position_ids,
                block_mask=layer_block_mask,
                direct_cp_metadata=layer_metadata,
                csa_metadata=csa_metadata,
                csa_prepared=csa_prepared,
                csa_block_mask=(
                    csa_shared_block_masks.get(csa_layer.ratio)
                    if csa_layer is not None else None
                ),
                qsa_metadata=qsa_metadata,
                qsa_layout=qsa_layout,
                dsa_metadata=dsa_metadata,
                dsa_layout=dsa_layout,
            )
            if csa_layer is not None:
                csa_indexer_losses.append(layer.self_attn.csa_indexer_loss_per_query)
            qsa_layer = getattr(layer.self_attn, "qsa_layer", None)
            if qsa_layer is not None:
                qsa_indexer_losses.append(layer.self_attn.qsa_indexer_loss_per_query)
            if getattr(layer.self_attn, "dsa_layer", None) is not None:
                dsa_indexer_losses.append(layer.self_attn.dsa_indexer_loss_per_query)
            module_observation = getattr(
                layer.self_attn, "attention_observation", None
            )
            # Sparse-family layers report their executed policy even when the
            # non-packed kernel has no THD execution metadata (e.g. evaluation).
            if isinstance(module_observation, dict) and (
                module_observation.get("qkv_format")
                or module_observation.get("type") in {
                    "compressed_sparse_attention", "qwen_sparse_attention",
                    "deepseek_sparse_attention",
                }
            ):
                attention_observation = module_observation
            elif layer_metadata is not None:
                from aspec.models.ops.direct_cp import (
                    direct_cp_attention_observation,
                )

                attention_observation = direct_cp_attention_observation(
                    layer_metadata
                )
                if not attention_observation.get("qkv_format"):
                    attention_observation = None
            else:
                attention_observation = module_observation
            if attention_observation is not None:
                attention_observations.append(attention_observation)
            if getattr(self.config, "ffn_type", "dense") != "dense":
                draft_hidden, logits = draft_hidden
                router_logits.append(logits)

        self.last_csa_indexer_losses = tuple(csa_indexer_losses)
        self.last_qsa_indexer_losses = tuple(qsa_indexer_losses)
        self.last_dsa_indexer_losses = tuple(dsa_indexer_losses)
        self.last_attention_observations = tuple(attention_observations)
        result = self.final_norm(draft_hidden)
        return (result, tuple(router_logits)) if return_router_logits else result

    def freeze_embedding(self) -> None:
        self.embed_tokens.weight.requires_grad = False

    @torch.no_grad()
    def load_embedding(
        self, model_path: str, embedding_key: str = "model.embed_tokens.weight"
    ) -> None:
        """Load embedding weights from target model checkpoint."""
        if os.path.exists(model_path):
            glob_path = os.path.join(model_path, "*.index.json")
            import glob as glob_mod

            index_json_path = glob_mod.glob(glob_path)

            if len(index_json_path) == 0:
                safetensors_path = os.path.join(model_path, "model.safetensors")
                if os.path.exists(safetensors_path):
                    with safe_open(safetensors_path, framework="pt") as f:
                        self.embed_tokens.weight.copy_(f.get_tensor(embedding_key))
                    return
                pytorch_model_path = os.path.join(model_path, "pytorch_model.bin")
                if os.path.exists(pytorch_model_path):
                    state_dict = torch.load(
                        pytorch_model_path, map_location="cpu", weights_only=True
                    )
                    self.embed_tokens.weight.copy_(state_dict[embedding_key])
                    return
                raise FileNotFoundError(
                    f"No index.json, model.safetensors or pytorch_model.bin found in {model_path}"
                )
            index_json_path = index_json_path[0]
            with open(index_json_path, "r") as f:
                index_json = json.load(f)
            ckpt_file = index_json["weight_map"][embedding_key]
            if ckpt_file.endswith(".safetensors"):
                with safe_open(os.path.join(model_path, ckpt_file), framework="pt") as f:
                    self.embed_tokens.weight.copy_(f.get_tensor(embedding_key))
            else:
                state_dict = torch.load(
                    os.path.join(model_path, ckpt_file), map_location="cpu", weights_only=True
                )
                self.embed_tokens.weight.copy_(state_dict[embedding_key])
        else:
            local_cache_path = snapshot_download(repo_id=model_path)
            self.load_embedding(local_cache_path, embedding_key)
