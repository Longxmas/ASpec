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

import importlib.metadata
import json

import pytest
import torch
from safetensors.torch import load_file

from aspec.models.draft import AutoDraftModelConfig, AutoEagle3DraftModel
from aspec.models.draft.keymap import to_export_keys
from tools import convert_to_hf

_CONFIG = {
    "architectures": ["Qwen3DSparkModel"],
    "model_type": "qwen3_dspark",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 128,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 256,
    "rope_theta": 10000.0,
    "num_target_layers": 2,
    "target_hidden_size": 64,
    "target_num_hidden_layers": 12,
    "target_layer_ids": [1, 9],
    "mask_token_id": 127,
    "markov_rank": 16,
    "markov_head_type": "vanilla",
    "enable_confidence_head": True,
    "confidence_head_with_markov": True,
    "tie_word_embeddings": False,
}


def _model():
    config = AutoDraftModelConfig.from_dict(_CONFIG)
    return AutoEagle3DraftModel.from_config(config, torch_dtype=torch.float32)


def test_convert_checkpoint_writes_strict_loadable_serving_artifact(tmp_path, monkeypatch):
    versions = {"ling-aspec-runtime": "0.1.0b0", "aspec": "0.0.16"}
    original_version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version",
                        lambda name: versions[name] if name in versions else original_version(name))
    torch.manual_seed(7)
    source = _model()
    checkpoint = tmp_path / "iter_0000002" / "model"
    checkpoint.mkdir(parents=True)
    (checkpoint / ".metadata").write_bytes(b"stub")
    config_path = tmp_path / "draft.json"
    config_path.write_text(json.dumps(_CONFIG), encoding="utf-8")
    wrapped = {
        f"model_state.model.draft_model.{key}": value.detach().clone()
        for key, value in source.state_dict().items()
    }
    monkeypatch.setattr(convert_to_hf, "_load_fsdp_state_dict", lambda _: wrapped)

    output = tmp_path / "published"
    manifest = convert_to_hf.convert_checkpoint(
        input_dir=checkpoint.parent,
        output_dir=output,
        config_path=config_path,
    )

    exported = load_file(output / "model.safetensors")
    expected = to_export_keys(source.state_dict())
    assert set(exported) == set(expected)
    for key, value in expected.items():
        torch.testing.assert_close(exported[key], value)
    exported_config = json.loads((output / "config.json").read_text())
    assert exported_config["_aspec_version"] == "0.1.0b0"
    assert exported_config["model_type"] == "qwen3"
    assert exported_config["num_target_layers"] == 2
    assert exported_config["dflash_config"] == {
        "num_target_layers": 12, "target_layer_ids": [1, 9],
    }
    assert manifest["tensor_count"] == len(expected)
    assert manifest["model_sha256"]


@pytest.mark.parametrize("change", [
    {"target_num_hidden_layers": None},
    {"target_layer_ids": [1, 12]},
    {"num_target_layers": 3},
    {"dflash_config": {"num_target_layers": 2}},
])
def test_serving_target_feature_contract_fails_closed(change):
    with pytest.raises(ValueError, match="Serving export"):
        convert_to_hf._serving_config({**_CONFIG, **change})


@pytest.mark.parametrize("model_type", ["qwen3_dspark", "dflash"])
def test_dense_draft_export_uses_registered_upstream_backbone_config(model_type):
    raw = {**_CONFIG, "model_type": model_type}
    if model_type == "dflash":
        raw["architectures"] = ["DFlashDraftModel"]
    exported = convert_to_hf._serving_config(raw)
    assert exported["model_type"] == "qwen3"
    assert exported["architectures"] == raw["architectures"]
    assert raw["model_type"] == model_type


def test_dspark_export_recovers_trained_gamma_and_rejects_a_conflicting_config(tmp_path):
    checkpoint = tmp_path / "iter_0000032"
    model_dir = checkpoint / "model"
    model_dir.mkdir(parents=True)
    (checkpoint / "meta.json").write_text(json.dumps({"training_topology": {
        "draft_algorithm": "dspark", "training_objective": {"block_size": 7}}}))
    assert convert_to_hf._checkpoint_training_contract(model_dir, _CONFIG)["block_size"] == 7
    with pytest.raises(ValueError, match="block_size"):
        convert_to_hf._checkpoint_training_contract(model_dir, {**_CONFIG, "block_size": 8})


def test_swa_export_preserves_official_loader_fields_and_manifest_contract():
    raw = {
        **_CONFIG,
        "sliding_window": 128,
        "layer_types": ["full_attention", "sliding_attention"],
    }
    exported = convert_to_hf._serving_config(raw)
    assert exported["sliding_window"] == 128
    assert exported["layer_types"] == ["full_attention", "sliding_attention"]
    contract = convert_to_hf._draft_structure_manifest(raw)
    assert contract["attention"] == {
        "head_layout": {
            "type": "gqa",
            "query_heads": 4,
            "key_value_heads": 2,
            "query_heads_per_key_value_head": 2,
            "head_dim": 16,
            "head_dim_source": "hidden_size/num_attention_heads",
        },
        "layer_types": ["full_attention", "sliding_attention"],
        "sliding_window": 128,
        "window_semantics": "anchor_context",
        "draft_block_attention": "bidirectional",
    }
    assert contract["loader_contract"] == {
        "runtime": "sglang",
        "validated_version": "0.5.21",
        "speculative_algorithm": "DSPARK",
        "required_patch_features": [],
        "validation": "required_per_artifact",
    }


@pytest.mark.parametrize("ffn_type", ["qwen3_moe", "ling_moe"])
def test_moe_export_declares_required_loader_patch(ffn_type):
    raw = {
        **_CONFIG,
        "ffn_type": ffn_type,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
    }
    if ffn_type == "ling_moe":
        raw.update(
            num_shared_experts=1,
            moe_shared_expert_intermediate_size=32,
            n_group=2,
            topk_group=1,
            norm_topk_prob=True,
            topk_method="noaux_tc",
            score_function="sigmoid",
            scoring_func="sigmoid",
            routed_scaling_factor=2.5,
            router_dtype="fp32",
            moe_router_enable_expert_bias=True,
            moe_router_bias_update_rate=0.001,
            router_z_loss_coef=3.5e-6,
            expert_swiglu_limit_list=[0, 4],
            share_expert_swiglu_limit_list=[0, 5],
        )

    exported = convert_to_hf._serving_config(raw)
    assert exported["ffn_type"] == ffn_type
    assert convert_to_hf._draft_structure_manifest(raw)["loader_contract"][
        "required_patch_features"
    ] == ["dflash_moe_v1"]


def test_convert_checkpoint_rejects_incompatible_state(tmp_path, monkeypatch):
    checkpoint = tmp_path / "iter_0000002" / "model"
    checkpoint.mkdir(parents=True)
    (checkpoint / ".metadata").write_bytes(b"stub")
    config_path = tmp_path / "draft.json"
    config_path.write_text(json.dumps(_CONFIG), encoding="utf-8")
    monkeypatch.setattr(
        convert_to_hf,
        "_load_fsdp_state_dict",
        lambda _: {"model.draft_model.unknown.weight": torch.zeros(1)},
    )

    try:
        convert_to_hf.convert_checkpoint(
            input_dir=checkpoint.parent,
            output_dir=tmp_path / "published",
            config_path=config_path,
        )
    except RuntimeError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("incompatible checkpoint was accepted")


@pytest.mark.parametrize("algorithm,architecture", [
    ("dflash", "DFlashDraftModel"),
    ("eagle3", "LlamaForCausalLMEagle3"),
])
def test_algorithm_exports_keep_identity_and_all_weights(tmp_path, monkeypatch, algorithm, architecture):
    from aspec.models.draft.keymap import to_internal_keys

    raw = {
        **_CONFIG,
        "architectures": [architecture],
        "model_type": "llama" if algorithm == "eagle3" else "dflash",
        "num_hidden_layers": 1,
        "num_aux_hidden_states": 2,
    }
    config = AutoDraftModelConfig.from_dict(raw)
    source = AutoEagle3DraftModel.from_config(config, torch_dtype=torch.float32)
    checkpoint = tmp_path / "iter_0000002" / "model"
    checkpoint.mkdir(parents=True)
    (checkpoint / ".metadata").write_bytes(b"stub")
    config_path = tmp_path / "draft.json"
    config_path.write_text(json.dumps(raw))
    wrapped = {f"model_state.model.draft_model.{key}": value.detach().clone()
               for key, value in source.state_dict().items()}
    monkeypatch.setattr(convert_to_hf, "_load_fsdp_state_dict", lambda _: wrapped)
    output = tmp_path / "published"
    manifest = convert_to_hf.convert_checkpoint(
        input_dir=checkpoint.parent, output_dir=output, config_path=config_path,
    )
    assert manifest["algorithm"] == algorithm
    assert manifest["draft_structure"]["loader_contract"]["speculative_algorithm"] == algorithm.upper()
    reloaded = AutoEagle3DraftModel.from_config(
        AutoDraftModelConfig.from_file(str(output / "config.json")), torch_dtype=torch.float32,
    )
    reloaded.load_state_dict(to_internal_keys(load_file(output / "model.safetensors"),
                                             reloaded.state_dict()), strict=True)
    for key, value in source.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[key], value)


def test_dflash2_export_roundtrip_keeps_selector_and_block_local_convolutions(tmp_path, monkeypatch):
    from aspec.models.draft.keymap import to_internal_keys

    raw = {**_CONFIG, "architectures": ["DFlash2DraftModel"], "model_type": "qwen3_dflash2",
           "block_size": 4, "conv_kernel_size": 2, "conv_group_size": 4,
           "selector_rank": 8, "selector_top_k": 4}
    for key in ("markov_rank", "markov_head_type", "enable_confidence_head", "confidence_head_with_markov"):
        raw.pop(key)
    source = AutoEagle3DraftModel.from_config(AutoDraftModelConfig.from_dict(raw))
    checkpoint = tmp_path / "iter_0000002" / "model"
    checkpoint.mkdir(parents=True)
    (checkpoint / ".metadata").write_bytes(b"stub")
    config_path = tmp_path / "draft.json"
    config_path.write_text(json.dumps(raw))
    wrapped = {f"model_state.model.draft_model.{key}": value.detach().clone()
               for key, value in source.state_dict().items()}
    monkeypatch.setattr(convert_to_hf, "_load_fsdp_state_dict", lambda _: wrapped)
    output = tmp_path / "published"
    manifest = convert_to_hf.convert_checkpoint(input_dir=checkpoint.parent, output_dir=output,
                                              config_path=config_path)
    assert manifest["algorithm"] == "dflash2"
    assert manifest["draft_structure"]["loader_contract"]["speculative_algorithm"] == "DFLASH"
    saved = json.loads((output / "config.json").read_text())
    assert saved["dflash_config"]["block_size"] == 4
    assert saved["dflash_config"]["selector_rank"] == 8
    assert "block_size" not in saved
    tensors = load_file(output / "model.safetensors")
    assert "embed_tokens.weight" not in tensors
    reloaded = AutoEagle3DraftModel.from_config(AutoDraftModelConfig.from_dict(saved))
    mapped = to_internal_keys(tensors, reloaded.state_dict())
    missing, unexpected = reloaded.load_state_dict(mapped, strict=False)
    assert missing == ["embed_tokens.weight"] and not unexpected
    for key, value in source.state_dict().items():
        if key != "embed_tokens.weight":
            torch.testing.assert_close(reloaded.state_dict()[key], value)

@pytest.mark.parametrize("destination", ["same", "ancestor", "nested", "symlink", "config"])
def test_export_rejects_overlapping_paths_before_loading_or_deleting(tmp_path, monkeypatch, destination):
    checkpoint = tmp_path / "checkpoints" / "iter_0000001"
    model = checkpoint / "model"
    model.mkdir(parents=True)
    (model / ".metadata").write_bytes(b"preserve checkpoint")
    config = checkpoint / "draft_config.json"
    config.write_text(json.dumps(_CONFIG))
    if destination == "same":
        output = checkpoint
    elif destination == "ancestor":
        output = checkpoint.parent
    elif destination == "nested":
        output = checkpoint / "export"
    elif destination == "symlink":
        output = tmp_path / "alias"
        output.symlink_to(checkpoint, target_is_directory=True)
    else:
        output = tmp_path / "config_output"
        output.mkdir()
        config = output / "draft.json"
        config.write_text(json.dumps(_CONFIG))
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    def unexpected_load(_):
        pytest.fail("overlapping export reached checkpoint loading")
    monkeypatch.setattr(convert_to_hf, "_load_fsdp_state_dict", unexpected_load)
    with pytest.raises(ValueError, match="overlap"):
        convert_to_hf.convert_checkpoint(input_dir=checkpoint, output_dir=output,
                                         config_path=config, force=True)
    assert {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
