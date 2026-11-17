# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Explicit MTP-prefix semantics, independent of execution and Target payloads."""

import copy
from collections.abc import Mapping


def resolve_mtp_conditioning(raw):
    policy = raw.get("mtp_conditioning")
    if policy is None:
        return None
    if (not isinstance(policy, Mapping)
            or not {"mode", "native_config"} <= set(policy)
            or set(policy) - {"mode", "native_config", "feature_source"}):
        raise ValueError("mtp_conditioning requires mode and native_config, with optional feature_source")
    if policy.get("feature_source", "draft") not in {"draft", "target"}:
        raise ValueError("MTP feature_source must be draft or target")
    if policy["mode"] not in {"token", "hidden"}:
        raise ValueError("MTP conditioning mode must be token or hidden")
    architectures = raw.get("architectures")
    if ((architectures and architectures[0] not in {
            "DFlashDraftModel", "DSparkDraftModel", "Qwen3DSparkModel"})
            or raw.get("model_type") in {"dflash2", "k3_dspark"}):
        raise ValueError("MTP conditioning currently requires DFlash or DSpark full GQA")
    if not isinstance(policy["native_config"], Mapping):
        raise ValueError("MTP conditioning native_config must be a native Qwen3.5 config mapping")
    from aspec.models.draft.mtp import NativeMTPConfig

    native = NativeMTPConfig(native_config=policy["native_config"]).native_config
    if (raw.get("ffn_type", "dense") != "dense"
            or raw.get("attention_mode", "gqa") != "gqa"
            or raw.get("is_causal", False)
            or any(kind != "full_attention" for kind in raw.get("layer_types", []))):
        raise ValueError("MTP conditioning currently requires Dense bidirectional full GQA Drafts")
    if native.get("attention_dropout", 0):
        raise ValueError("Frozen MTP conditioning requires zero attention dropout")
    for name, expected in (
        ("hidden_size", raw.get("target_hidden_size")),
        ("vocab_size", raw.get("vocab_size")),
        ("num_hidden_layers", raw.get("target_num_hidden_layers")),
    ):
        if native.get(name) != expected:
            raise ValueError(f"MTP native {name} disagrees with its Target/Draft binding")
    if raw.get("hidden_size") != native["hidden_size"]:
        raise ValueError("MTP conditioning requires the same frozen Target embedding width")
    if "full_attention" not in native.get("layer_types", []):
        raise ValueError("MTP native config must identify its full-attention predictor")
    return {"mode": policy["mode"], "native_config": copy.deepcopy(native),
            **({"feature_source": policy["feature_source"]} if "feature_source" in policy else {})}


def target_mtp_features(policy):
    """Old assets retain Draft-side execution; new policies can request Target capture."""
    return bool(policy and policy["mode"] == "hidden" and policy.get("feature_source", "draft") == "target")


def mtp_conditioning_semantics(raw):
    policy = resolve_mtp_conditioning(raw)
    if policy is None:
        return None
    return {
        **policy,
        "prefix_layout": "dual_anchor",
        "first_candidate": "teacher_forced_in_training_native_mtp_in_proposals",
        "target_context": "strictly_before_original_anchor",
        "prediction_state": "target_h[a-1]_and_token_x[a]_predict_x[a+1]",
        "feature_computation": ("frozen_target_side_native_mtp"
                                if policy.get("feature_source", "draft") == "target"
                                else "frozen_draft_side_native_mtp"),
        "feature_stage": "native_predictor_post_norm",
        "fusion": "zero_initialized_embedding_residual" if policy["mode"] == "hidden" else None,
        "frozen": ["native_mtp", "target_embedding", "target_lm_head"],
    }


def validate_mtp_target_config(native, target):
    """Check the native predictor's mathematical binding, beyond tensor shapes."""
    target = target.get("text_config", target)
    if target.get("model_type") != "qwen3_5_text":
        raise ValueError("MTP conditioning requires the native Qwen3.5 Dense Target")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    target = Qwen3_5TextConfig(**target).to_dict()
    keys = ("model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size",
            "rms_norm_eps", "hidden_act", "attention_bias", "attention_dropout",
            "layer_types", "rope_parameters", "attn_output_gate", "max_position_embeddings")
    for key in keys:
        if key in native and native[key] != target.get(key):
            raise ValueError("MTP conditioning native config differs from its Target: " + key)


def mtp_conditioning_memory(policy, *, target_tensor_parallel_size=1):
    """Additional BF16 model/optimizer bytes; neither total fit nor activation peak."""
    if policy is None:
        return None
    from aspec.models.dmtp import FrozenMTPFeatures

    source = FrozenMTPFeatures(policy["native_config"])
    frozen = sum(p.numel() for p in source.parameters())
    width = policy["native_config"]["hidden_size"]
    fusion = 2 * width * width + 2 * width if policy["mode"] == "hidden" else 0
    result = {
        "scope": "additional Draft-side components; no duplicated vocabulary embedding/head",
        "frozen_weight_bytes": 2 * frozen,
        "trainable_weight_bytes": 2 * fusion,
        "gradient_bytes": 2 * fusion,
        "fp32_master_bytes": 4 * fusion,
        "fp32_gradient_buffer_bytes": 4 * fusion,
        "adam_moment_bytes": 8 * fusion,
        "derived_feature_bytes_per_token": 2 * width if fusion else 0,
        "activation_peak_bytes": None,
    }
    if policy.get("feature_source", "draft") == "target":
        result.update(
            scope="Draft fusion and Target publisher predictor are separate allocations",
            frozen_weight_bytes=0, derived_feature_bytes_per_token=0,
            frozen_binding_bytes=32,
            target={
                "publisher_rank_only": True,
                "frozen_weight_bytes": 2 * frozen if policy["mode"] == "hidden" else 0,
                "replicated_embedding_bytes": (2 * width * policy["native_config"]["vocab_size"]
                                               if policy["mode"] == "hidden" and target_tensor_parallel_size > 1 else 0),
                "activation_peak_bytes": None,
            },
            transported_feature_bytes_per_token=2 * width if policy["mode"] == "hidden" else 0,
        )
    return result
