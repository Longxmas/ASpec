# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Adapted from ASpec's retained native MTP implementation (Speculators
# 7ec7aba9450b533334e8f7992baf4e93be3c1a57 ancestry). The Target provider reuses
# the shared predictor equations and official Transformers modules, without
# importing Draft-only execution dependencies.
"""Frozen, document-local native MTP features in the Target process."""

import torch
from torch import nn
from transformers import AutoConfig
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
)

from aspec.config.mtp_conditioning import validate_mtp_target_config
from aspec.models.checkpoint import load_tensor_from_pretrained
from aspec.models.mtp_features import predict_mtp_embeddings


class FrozenTargetMTP(nn.Module):
    """Produce row a predicting x[a+1]; row zero has no predecessor state.

    TP1 borrows the loaded Target embedding. A TP publisher with sharded
    vocabulary weights owns an explicit replicated embedding instead; this
    is reported separately from the native predictor body.
    """

    def __init__(self, native_config):
        super().__init__()
        self.native = Qwen3_5TextConfig(**native_config)
        self.native._attn_implementation = "sdpa"
        full = [i for i, kind in enumerate(self.native.layer_types) if kind == "full_attention"]
        if not full:
            raise ValueError("Native MTP requires a full-attention predictor")
        width, eps = self.native.hidden_size, self.native.rms_norm_eps
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(width, eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(width, eps)
        self.fc = nn.Linear(2 * width, width, bias=False)
        self.midlayer = Qwen3_5DecoderLayer(self.native, full[-1])
        self.norm = Qwen3_5RMSNorm(width, eps)
        # RoPE buffers are nonpersistent and must be materialized independently
        # of the meta construction used for checkpoint-backed parameters.
        with torch.device("cpu"):
            self.rotary_emb = Qwen3_5TextRotaryEmbedding(self.native, device="cpu")
        self.requires_grad_(False)
        self.eval()
        self.embedding_storage = None
        self.embedding_bytes = 0

    def train(self, mode=True):
        return super().train(False)

    @classmethod
    def from_target(cls, policy, target_model_path, embedding_weight, *, device, dtype,
                    allow_replicated_embedding=False):
        native = policy["native_config"]
        target = AutoConfig.from_pretrained(target_model_path).to_dict()
        if target.get("quantization_config") or target.get("text_config", {}).get("quantization_config"):
            raise ValueError("Target MTP capture requires unquantized native weights")
        validate_mtp_target_config(native, target)
        with torch.device("meta"):
            model = cls(native)
        state = {}
        for name, destination in model.state_dict().items():
            key = "mtp.layers.0." + name.removeprefix("midlayer.") if name.startswith("midlayer.") else "mtp." + name
            tensor = load_tensor_from_pretrained(target_model_path, key)
            if tensor.shape != destination.shape:
                raise ValueError("Native MTP tensor shape mismatch: " + key)
            state[name] = tensor.to(device=device, dtype=dtype)
        model.load_state_dict(state, strict=True, assign=True)
        model.rotary_emb = Qwen3_5TextRotaryEmbedding(model.native, device=device)
        shape = (model.native.vocab_size, model.native.hidden_size)
        if (embedding_weight is not None and embedding_weight.ndim == 2
                and embedding_weight.shape[0] >= shape[0] and embedding_weight.shape[1] == shape[1]):
            embedding = embedding_weight[:shape[0]].detach()
            if embedding.device != torch.device(device) or embedding.dtype != dtype:
                raise ValueError("Borrowed Target embedding has a different device or dtype")
            model.embedding_storage = "borrowed_target"
        else:
            if not allow_replicated_embedding:
                raise ValueError("TP1 MTP capture requires the loaded Target embedding")
            embedding = load_tensor_from_pretrained(target_model_path, policy["embedding_key"])
            if tuple(embedding.shape) != shape:
                raise ValueError("Native MTP Target embedding shape mismatch")
            embedding = embedding.to(device=device, dtype=dtype)
            model.embedding_storage = "replicated_target"
            model.embedding_bytes = embedding.numel() * embedding.element_size()
        # The Target owns a borrowed Parameter; do not register a second module
        # owner or a vocabulary-sized tensor in the predictor body's state.
        object.__setattr__(model, "embedding_weight", embedding)
        model.requires_grad_(False)
        return model

    @torch.no_grad()
    def forward(self, input_ids, final_hidden_states):
        if (input_ids.ndim != 1 or final_hidden_states.ndim != 2
                or final_hidden_states.shape != (input_ids.shape[0], self.native.hidden_size)):
            raise ValueError("MTP capture requires one complete document with aligned Target final states")
        result = final_hidden_states.new_zeros(final_hidden_states.shape)
        if input_ids.numel() > 1:
            positions = torch.arange(input_ids.numel() - 1, device=input_ids.device)[None]
            embeddings = torch.nn.functional.embedding(input_ids[None, 1:], self.embedding_weight)
            result[1:] = predict_mtp_embeddings(
                self, final_hidden_states[None, :-1], embeddings, positions
            )[0]
        return result

    def status(self):
        return {"state": "ready", "alignment": "anchor_post_norm_v1",
                "frozen": True, "embedding_storage": self.embedding_storage,
                "embedding_bytes": self.embedding_bytes,
                "predictor_parameters": sum(p.numel() for p in self.parameters()),
                "predictor_weight_bytes": sum(p.numel() * p.element_size() for p in self.parameters())}
