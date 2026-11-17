# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Adapted from FasterDecoding/Medusa e2a5d20c048a9b0a4092e6933c34313687422518
# (medusa/model/medusa_model.py, Apache-2.0). Head structure, zero-weight
# residual initialization and the shared-block repetition follow the reference;
# configuration validation uses ASpec's fail-closed draft contract.
"""Heads-only Medusa draft: parallel future-token heads on frozen Target features."""

import torch
from torch import nn
from transformers import PretrainedConfig


class MedusaConfig(PretrainedConfig):
    model_type = "aspec_medusa"
    has_no_defaults_at_init = True

    def __init__(self, hidden_size=None, vocab_size=None, target_hidden_size=None,
                 target_num_hidden_layers=None, medusa_num_heads=5, medusa_num_layers=1,
                 **kwargs):
        for name, value in (("hidden_size", hidden_size), ("vocab_size", vocab_size),
                            ("target_hidden_size", target_hidden_size),
                            ("target_num_hidden_layers", target_num_hidden_layers)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"Medusa requires a positive integer {name}")
        if hidden_size != target_hidden_size:
            raise ValueError(
                "Medusa heads read Target features directly: "
                "hidden_size must equal target_hidden_size")
        for name, value in (("medusa_num_heads", medusa_num_heads),
                            ("medusa_num_layers", medusa_num_layers)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"Medusa requires a positive integer {name}")
        if kwargs.get("draft_vocab_size", vocab_size) != vocab_size:
            raise ValueError("Medusa uses the full Target vocabulary")
        if kwargs.get("ffn_type", "dense") != "dense" or kwargs.get("lm_head_rank_divisor"):
            raise ValueError("Medusa heads have no FFN or replaceable output head")
        if "num_hidden_layers" in kwargs:
            raise ValueError("Medusa is heads-only; use medusa_num_layers for head depth")
        # The single post-norm final-state slot is the native-MTP feature convention.
        fixed = dict(architectures=["MedusaDraftModel"],
                     target_layer_ids=[target_num_hidden_layers - 1],
                     num_target_layers=1, num_aux_hidden_states=1,
                     draft_vocab_size=vocab_size,
                     tie_word_embeddings=False, norm_output=False, fc_norm=False)
        for key, value in fixed.items():
            if key in kwargs and kwargs[key] != value:
                raise ValueError(
                    "Medusa derived field disagrees with the heads-only contract: " + key)
        kwargs.update(fixed)
        super().__init__(hidden_size=hidden_size, vocab_size=vocab_size,
                         target_hidden_size=target_hidden_size,
                         target_num_hidden_layers=target_num_hidden_layers,
                         medusa_num_heads=medusa_num_heads, medusa_num_layers=medusa_num_layers,
                         **kwargs)

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        # The base class applies keyword overrides with setattr after construction,
        # so recognized overrides are merged through __init__ instead; unknown
        # arguments stay unused, preserving the inherited loader contract.
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        config = cls(**dict(config_dict))
        recognized = {key: value for key, value in kwargs.items() if hasattr(config, key)}
        # An "auto" dtype is a loader value, not a constructor input; the
        # inherited loader applies it after construction.
        deferred = {key: value for key, value in recognized.items()
                    if key in ("dtype", "torch_dtype") and value == "auto"}
        merged_overrides = {key: value for key, value in recognized.items()
                            if key not in deferred}
        if merged_overrides:
            merged = dict(config_dict)
            merged.update(merged_overrides)
            config = cls(**merged)
        for key, value in deferred.items():
            setattr(config, key, value)
        unused = {key: value for key, value in kwargs.items() if key not in recognized}
        return (config, unused) if return_unused_kwargs else config


class MedusaResBlock(nn.Module):
    """Reference residual block: x + SiLU(Linear(x)) with a zero-initialized weight."""

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))


class MedusaDraftModel(nn.Module):
    config_class = MedusaConfig

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.medusa_num_heads = config.medusa_num_heads
        self.medusa_num_layers = config.medusa_num_layers
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.medusa_head = nn.ModuleList([
            nn.Sequential(
                # The reference repeats one shared ResBlock instance per head depth.
                *([MedusaResBlock(config.hidden_size)] * config.medusa_num_layers),
                nn.Linear(config.hidden_size, config.vocab_size, bias=False),
            )
            for _ in range(config.medusa_num_heads)
        ])

    def forward(self, hidden_states):
        """[B, S, H] frozen Target features -> [K, B, S, V] per-head logits."""
        return torch.stack([head(hidden_states) for head in self.medusa_head], dim=0)
