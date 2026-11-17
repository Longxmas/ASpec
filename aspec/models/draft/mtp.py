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

# Adapted from Speculators MTP (7ec7aba9450b533334e8f7992baf4e93be3c1a57).
# Native decoder, normalization and rotary mathematics use official Transformers.
"""Native Qwen3.5 Dense MTP assets, not an independently initialized EAGLE."""

import copy

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
)

from aspec.models.draft.base import Eagle3DraftModel, load_tensor_from_pretrained


class NativeMTPConfig(PretrainedConfig):
    model_type = "aspec_native_mtp"
    has_no_defaults_at_init = True

    def __init__(self, native_config=None, **kwargs):
        if native_config is None or native_config.get("model_type") != "qwen3_5_text":
            raise ValueError("Native MTP currently requires a Qwen3.5 Dense text config")
        native = copy.deepcopy(native_config)
        if native.get("mtp_use_dedicated_embeddings", False):
            raise ValueError("Native MTP dedicated embeddings are not the shared-embedding recipe")
        if kwargs.get("ffn_type", "dense") != "dense" or kwargs.get("lm_head_rank_divisor"):
            raise ValueError("Native Qwen3.5 Dense MTP cannot replace its native FFN or frozen output head")
        if native.get("mtp_num_hidden_layers", 1) != 1:
            raise ValueError("Native Qwen3.5 MTP checkpoint must contain its single native prediction layer")
        # RoPE belongs to the native text config. Accept an identical alias,
        # but do not duplicate MRoPE fields into generic HF rope validation.
        if "rope_parameters" in kwargs and kwargs.pop("rope_parameters") != native.get("rope_parameters"):
            raise ValueError("Native MTP RoPE disagrees with native_config")
        for key in ("hidden_size", "vocab_size", "intermediate_size", "num_attention_heads",
                    "num_key_value_heads", "head_dim", "rms_norm_eps", "max_position_embeddings",
                    "attention_bias", "attention_dropout", "hidden_act"):
            if key in kwargs and key in native and kwargs[key] != native[key]:
                raise ValueError("Native MTP shape/semantics disagree with native_config: " + key)
            if key in native:
                kwargs[key] = native[key]
        vocab = native["vocab_size"]
        if kwargs.get("draft_vocab_size", vocab) != vocab:
            raise ValueError("Native MTP uses the frozen Target vocabulary/head")
        fixed = dict(architectures=["NativeMTPDraftModel"], num_hidden_layers=1,
                      layer_types=["full_attention"], draft_vocab_size=vocab,
                      target_hidden_size=native["hidden_size"],
                      target_num_hidden_layers=native["num_hidden_layers"],
                      target_layer_ids=[native["num_hidden_layers"] - 1],
                      num_target_layers=1, num_aux_hidden_states=1,
                      tie_word_embeddings=False, norm_output=False, fc_norm=False,
                      is_causal=True)
        for key, value in fixed.items():
            if key in kwargs and kwargs[key] != value:
                raise ValueError("Native MTP derived field disagrees with native_config: " + key)
        kwargs.update(fixed)
        super().__init__(**kwargs)
        self.native_config = native


class NativeMTPDraftModel(Eagle3DraftModel):
    config_class = NativeMTPConfig

    def __init__(self, config, attention_backend="sdpa"):
        super().__init__(config)
        if attention_backend not in {"sdpa", "fa4"}:
            raise NotImplementedError("Native MTP requires official SDPA or FA4 THD execution")
        native = Qwen3_5TextConfig(**config.native_config)
        native._attn_implementation = "sdpa"
        self.attention_backend = attention_backend
        if attention_backend == "fa4":
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

            if native.attention_dropout:
                raise ValueError("Native MTP FA4 does not support attention dropout")
            ALL_ATTENTION_FUNCTIONS.register("aspec_mtp_fa4", native_mtp_fa4_attention)
            native._attn_implementation = "aspec_mtp_fa4"
        full_layers = [i for i, kind in enumerate(native.layer_types) if kind == "full_attention"]
        if not full_layers:
            raise ValueError("Native Qwen3.5 MTP requires a full-attention decoder definition")
        self.native = native
        self.target_vocab_size = self.vocab_size = native.vocab_size
        self.embed_tokens = nn.Embedding(native.vocab_size, native.hidden_size)
        self.lm_head = nn.Linear(native.hidden_size, native.vocab_size, bias=False)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(native.hidden_size, native.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(native.hidden_size, native.rms_norm_eps)
        self.fc = nn.Linear(native.hidden_size * 2, native.hidden_size, bias=False)
        self.midlayer = Qwen3_5DecoderLayer(native, full_layers[-1])
        self.norm = Qwen3_5RMSNorm(native.hidden_size, native.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(native)
        self.norm_output = False
        self.freeze_embedding()
        self.freeze_lm_head()

    @staticmethod
    def native_key(name):
        if name.startswith("midlayer."):
            return "mtp.layers.0." + name.removeprefix("midlayer.")
        return "mtp." + name

    @torch.no_grad()
    def initialize_native_weights(self, target_model_path):
        """Load only native MTP tensors, never the full verifier transformer."""
        state = self.state_dict()
        for name, destination in state.items():
            if name in {"embed_tokens.weight", "lm_head.weight"}:
                continue
            source = load_tensor_from_pretrained(target_model_path, self.native_key(name))
            if source.shape != destination.shape:
                raise ValueError("Native MTP tensor shape mismatch: " + name)
            destination.copy_(source)

    @torch.no_grad()
    def validate_target_binding(self, target_model_path, *, embedding_key, lm_head_key):
        """Frozen shared tensors must be the verifier's, including after resume."""
        for name, key in (("embed_tokens.weight", embedding_key), ("lm_head.weight", lm_head_key)):
            actual = self.state_dict()[name]
            if hasattr(actual, "full_tensor"):
                actual = actual.full_tensor()  # All training ranks validate loaded bindings.
            expected = load_tensor_from_pretrained(target_model_path, key)
            if tuple(actual.shape) != tuple(expected.shape):
                raise ValueError("Native MTP shared Target tensor shape mismatch: " + name)
            for start in range(0, expected.shape[0], 1024):
                reference = expected[start:start + 1024].to(dtype=actual.dtype)
                if not torch.equal(actual[start:start + 1024].cpu(), reference):
                    raise ValueError("Native MTP frozen tensor differs from its Target: " + name)

    def predict(self, hidden_states, input_ids, position_ids, layout=None):
        return self.predict_embeddings(hidden_states, self.embed_tokens(input_ids), position_ids, layout)

    def predict_embeddings(self, hidden_states, input_embeds, position_ids, layout=None):
        """Shared native mathematics for trainable MTP and frozen feature extraction."""
        from aspec.models.mtp_features import predict_mtp_embeddings

        return predict_mtp_embeddings(self, hidden_states, input_embeds, position_ids, layout)

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)

    def project_hidden_states(self, hidden_states):
        return hidden_states

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)

    def backbone(self, input_embeds, hidden_states, attention_mask, position_ids,
                 cache_keys=None, cache_values=None, use_cache=True):
        raise NotImplementedError("Native MTP uses predict(), not EAGLE suffix attention/cache")

    def state_dict_for_serving(self, state_dict=None):
        state = self.state_dict() if state_dict is None else state_dict
        return {("model.embed_tokens.weight" if name == "embed_tokens.weight" else
                 name if name == "lm_head.weight" else self.native_key(name)): value
                for name, value in state.items()}

    def state_dict_from_serving(self, tensors):
        names = {("model.embed_tokens.weight" if name == "embed_tokens.weight" else
                  name if name == "lm_head.weight" else self.native_key(name)): name
                 for name in self.state_dict()}
        unknown = set(tensors) - set(names)
        if unknown:
            raise ValueError("Unknown native MTP serving tensors: " + str(sorted(unknown)))
        return {names[name]: value for name, value in tensors.items()}


def native_mtp_fa4_attention(module, query, key, value, attention_mask, *,
                             mtp_layout, scaling=None, dropout=0.0, **kwargs):
    """Keep official Qwen projection/RoPE/gating; replace only the causal core."""
    from aspec.models.ops.eagle_thd import eagle_fa4_attention

    if attention_mask is not None or dropout:
        raise ValueError("MTP FA4 requires its document-causal metadata and zero dropout")
    if not hasattr(module, "attention_observation"):
        module.attention_observation = {}
    output = eagle_fa4_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
        mtp_layout, 0, module.attention_observation, softmax_scale=scaling,
    )
    return output, None
