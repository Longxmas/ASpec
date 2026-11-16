#!/usr/bin/env python3
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

"""Export a ASpec FSDP checkpoint as a current-scope serving artifact."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import pickle
import shutil
from collections import Counter
from pathlib import Path

import torch
import torch.distributed.checkpoint as dist_cp
from safetensors.torch import save_file
from typing_extensions import override

from aspec.config.draft_algorithm import draft_model_semantics, resolve_draft_algorithm
from aspec.models.draft import AutoDraftModelConfig, AutoEagle3DraftModel
from aspec.models.draft.keymap import to_export_keys

logger = logging.getLogger(__name__)


class _UnpicklerWrapper(pickle.Unpickler):
    @override
    def find_class(self, mod_name, name):
        class DummyClass:
            def __init__(self, *args, **kwargs):
                pass

        if mod_name.startswith(("megatron", "glm")):
            return DummyClass
        return super().find_class(mod_name, name)


class _WrappedStorageReader(dist_cp.FileSystemReader):
    @override
    def read_metadata(self):
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = _UnpicklerWrapper(metadata_file).load()
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = dist_cp.StorageMeta()
        metadata.storage_meta.load_id = self.load_id
        if metadata.planner_data is None:
            metadata.planner_data = {}
        return metadata


class _EmptyStateDictLoadPlanner(dist_cp.default_planner.DefaultLoadPlanner):
    @override
    def set_up_planner(
        self,
        state_dict: dist_cp.metadata.STATE_DICT_TYPE,
        metadata: dist_cp.metadata.Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        if metadata is None:
            raise ValueError("Checkpoint metadata is required for export")
        for key, value in metadata.state_dict_metadata.items():
            if "optimizer" in key:
                continue
            if isinstance(value, dist_cp.metadata.TensorStorageMetadata):
                value = torch.empty(value.size, dtype=value.properties.dtype)
            state_dict[key] = value
        super().set_up_planner(state_dict, metadata, is_coordinator)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aspec_version() -> str:
    try:
        return importlib.metadata.version("ling-aspec-runtime")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _detect_model_dir(input_dir: Path) -> Path:
    model_dir = input_dir / "model"
    return model_dir if model_dir.is_dir() else input_dir


def _load_fsdp_state_dict(input_dir: Path) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=_WrappedStorageReader(str(input_dir)),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    return state_dict


def _extract_model_weights(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    model_state = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor) or "draft_model." not in key:
            continue
        export_key = key.split("draft_model.", 1)[1]
        if export_key in model_state:
            raise ValueError(f"duplicate draft tensor after checkpoint extraction: {export_key}")
        model_state[export_key] = value
    if not model_state:
        raise ValueError(
            "No draft model weights found. Pass an iter_xxxxxxx directory or its model/ child."
        )
    return model_state


def _resolve_config_path(input_dir: Path, configured: Path | None) -> Path:
    checkpoint_dir = _detect_model_dir(input_dir).parent
    saved_config = checkpoint_dir / "draft_config.json"
    path = configured or (saved_config if saved_config.is_file() else input_dir / "config.json")
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Draft config not found at {path}; pass --config with the training draft config"
        )
    return path


def _prepare_output_directory(path: Path, *, force: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not force:
            raise FileExistsError(f"output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _serving_config(raw_config: dict) -> dict:
    from aspec.config.draft_attention import resolve_draft_attention
    from aspec.config.draft_structure import resolve_draft_structure

    if raw_config.get("mtp_conditioning"):
        raise NotImplementedError("MTP-conditioned Drafts require native --format aspec export")

    if raw_config.get("architectures") == ["NativeMTPDraftModel"]:
        native = json.loads(json.dumps(raw_config["native_config"]))
        native["architectures"] = ["Qwen3_5ForCausalLMMTP"]
        native["mtp_num_hidden_layers"] = 1
        return native
    attention = resolve_draft_attention(raw_config)
    if attention["csa"] is not None:
        raise NotImplementedError(
            "CSA Draft serving export requires a compressed-entry/indexer-aware loader"
        )
    if attention["qsa"] is not None:
        raise NotImplementedError(
            "QSA Draft serving export requires an indexer/sparse-attention-aware loader"
        )
    if attention.get("dsa") is not None:
        raise NotImplementedError("DSA Draft requires native --format aspec export")
    resolve_draft_structure(raw_config)
    config = json.loads(json.dumps(raw_config))
    if config.get("architectures") == ["DFlash2DraftModel"]:
        from aspec.models.draft.dflash2 import DFlash2Config, dflash2_config_for_serving

        return dflash2_config_for_serving(DFlash2Config.from_dict(config).to_dict())
    if config.get("architectures") == ["K3DSparkModel"]:
        raise NotImplementedError("Materialized K3 MLA requires --format aspec; official SGLang has no matching DSpark loader")
    if config.get("architectures") == ["Eagle3DeepseekV2ForCausalLM"]:
        raise NotImplementedError("MLA EAGLE requires --format aspec; SGLang's retained loader has Target-MLA cache assumptions not verified by this recipe")
    if config.get("architectures") == ["LlamaForCausalLMEagle4"]:
        raise NotImplementedError("EAGLE4 requires --format aspec; SGLang0.5.21 serving is outside this contract")
    if config.get("architectures") == ["LlamaForCausalLMEagle3"]:
        if config.get("lm_head_rank_divisor"):
            raise NotImplementedError("SGLang EAGLE3 serving export does not support a factorized lm_head")
        layers = config.get("target_layer_ids")
        if not layers or config.get("num_aux_hidden_states", len(layers)) != len(layers):
            raise ValueError("Serving export has inconsistent EAGLE3 feature layers")
        if len(layers) * config.get("target_hidden_size", config.get("hidden_size")) == config.get("hidden_size"):
            raise NotImplementedError("SGLang EAGLE3 cannot distinguish a single auxiliary feature from recurrent hidden states; its loader would skip the trained FC")
        if config.get("use_aux_norm") and not config.get("fc_norm"):
            raise ValueError("EAGLE3 use_aux_norm requires fc_norm=true in the training config; do not add serving-only normalization")
        config["eagle_config"] = {
            **dict(config.get("eagle_config") or {}),
            "eagle_aux_hidden_state_layer_ids": list(layers),
        }
        return config
    if config.get("model_type") in {"qwen3_dspark", "dflash"}:
        config["model_type"] = "qwen3"
    if config.get("model_type") == "qwen3" and "sliding_attention" in (config.get("layer_types") or []):
        config["use_sliding_window"] = True
    # ASpec's flat num_target_layers is the feature count; upstream's
    # dflash_config.num_target_layers is the Target decoder depth. Translate at
    # the export boundary, retaining the flat fields for ASpec reloads.
    layers = config.get("target_layer_ids")
    if layers is not None:
        depth = config.get("target_num_hidden_layers")
        if not isinstance(depth, int) or depth <= 0:
            raise ValueError("Serving export requires the Target decoder depth")
        if len(layers) != config.get("num_target_layers") or any(
            not isinstance(layer, int) or not 0 <= layer < depth for layer in layers
        ):
            raise ValueError("Serving export has inconsistent Target feature layers")
        upstream = dict(config.get("dflash_config") or {})
        expected = {"num_target_layers": depth, "target_layer_ids": layers}
        for key, value in expected.items():
            if key in upstream and upstream[key] != value:
                raise ValueError(f"Serving export has conflicting dflash_config.{key}")
        config["dflash_config"] = {**upstream, **expected}
    return config


def _checkpoint_training_contract(model_dir: Path, raw_config: dict) -> dict:
    """Recover authoritative training identity from the checkpoint/run artifacts."""
    algorithm = resolve_draft_algorithm(raw_config)
    family = "DSpark" if algorithm == "dspark" else "DFlash"
    if algorithm == "dflash2":
        raw_config = AutoDraftModelConfig.from_dict(raw_config).to_dict()
    contract = {"algorithm": algorithm}
    metadata_path = model_dir.parent / "meta.json"
    if metadata_path.is_file():
        topology = json.loads(metadata_path.read_text()).get("training_topology") or {}
        saved_algorithm = topology.get("draft_algorithm")
        if saved_algorithm is not None and saved_algorithm != algorithm:
            raise ValueError("Export algorithm conflicts with the checkpoint algorithm")
        if topology.get("training_objective") is not None:
            contract["objective"] = topology["training_objective"]
            if algorithm in {"dspark", "dflash", "dflash2"} and "block_size" in contract["objective"]:
                contract["block_size"] = contract["objective"]["block_size"]
        if algorithm == "eagle4":
            from aspec.config.eagle4 import eagle4_training_policy
            saved_policy = (topology.get("training_objective") or {}).get("eagle4")
            if saved_policy is None or {k: v for k, v in saved_policy.items() if k != "loss_chunk_size"} != eagle4_training_policy(raw_config):
                raise ValueError("EAGLE4 export stage/retrieval objective conflicts with checkpoint")
        saved_semantics = topology.get("draft_model_semantics")
        if saved_semantics is not None:
            effective = AutoDraftModelConfig.from_dict(raw_config)
            if draft_model_semantics(effective) != draft_model_semantics(saved_semantics):
                raise ValueError("Export model semantics conflict with the checkpoint; use its saved draft_config.json")
    from aspec.config.draft_structure import checkpoint_ffn_policy, resolve_draft_structure

    saved_ffn = checkpoint_ffn_policy(model_dir.parent, topology if metadata_path.is_file() else {})
    current_ffn = resolve_draft_structure(raw_config)
    if saved_ffn is None and current_ffn["ffn_type"] != "dense":
        raise ValueError("Export cannot verify legacy MoE routing without draft_config.json")
    if saved_ffn is not None and saved_ffn != current_ffn:
        raise ValueError("Export FFN/routing policy conflicts with the checkpoint")
    # A real run's snapshot records block size even for legacy checkpoints whose
    # standalone model config did not persist the training-only field.
    checkpoint_dir = model_dir.parent
    if algorithm in {"dspark", "dflash", "dflash2"} and checkpoint_dir.parent.name == "checkpoints":
        snapshot = checkpoint_dir.parent.parent / "config.yaml"
        if snapshot.is_file():
            from omegaconf import OmegaConf

            size = OmegaConf.select(OmegaConf.load(snapshot), "training.dflash_block_size")
            if type(size) is not int or size < 2:
                raise ValueError(f"Invalid {family} block size in the training snapshot")
            saved_size = contract.get("block_size")
            if saved_size is not None and saved_size != size:
                raise ValueError("Export block_size conflicts with the checkpoint objective")
            declared = raw_config.get("block_size")
            if declared is not None and declared != size:
                raise ValueError("Export block_size conflicts with the training snapshot")
            contract["block_size"] = size
    if algorithm in {"dspark", "dflash", "dflash2"}:
        size = contract.get("block_size", raw_config.get("block_size"))
        if size is not None:
            if type(size) is not int or size < 2:
                raise ValueError(f"{family} export requires a valid trained block_size")
            if raw_config.get("block_size", size) != size:
                raise ValueError("Export block_size conflicts with the checkpoint objective")
            contract["block_size"] = size
        elif metadata_path.is_file():
            raise ValueError(f"{family} export requires trained block_size in checkpoint, run snapshot or explicit config")
    return contract


def _draft_structure_manifest(raw_config: dict) -> dict:
    """Describe the exported structure without claiming per-artifact loader validation."""
    from aspec.config.draft_attention import (
        resolve_draft_attention,
        resolve_draft_head_layout,
    )
    from aspec.config.draft_structure import resolve_draft_structure

    algorithm = resolve_draft_algorithm(raw_config)
    attention = resolve_draft_attention(raw_config)
    structure = resolve_draft_structure(raw_config)
    return {
        "ffn": structure,
        "attention": {
            "head_layout": resolve_draft_head_layout(raw_config),
            "layer_types": attention["layer_types"],
            "sliding_window": attention["sliding_window"],
            "window_semantics": attention["window_semantics"],
            "draft_block_attention": attention["draft_block_attention"],
        },
        "loader_contract": {
            "runtime": "sglang",
            "validated_version": "0.5.21",
            "speculative_algorithm": ("DFLASH" if algorithm == "dflash2" else
                                      "EAGLE" if algorithm == "mtp" else algorithm.upper()),
            "required_patch_features": (
                ["dflash_moe_v1"] if structure["ffn_type"] != "dense" else []
            ),
            "validation": "required_per_artifact",
        },
    }


def convert_checkpoint(
    *,
    input_dir: Path,
    output_dir: Path,
    config_path: Path | None = None,
    target_model_path: str | None = None,
    embedding_key: str = "model.embed_tokens.weight",
    output_dtype: torch.dtype | None = None,
    force: bool = False,
    artifact_format: str = "sglang",
) -> dict:
    input_dir = input_dir.expanduser().resolve()
    model_dir = _detect_model_dir(input_dir)
    if not (model_dir / ".metadata").is_file():
        raise FileNotFoundError(f"FSDP model checkpoint not found at {model_dir}")
    config_path = _resolve_config_path(input_dir, config_path)
    output_dir = output_dir.expanduser().resolve()
    checkpoint_dir = model_dir.parent
    if (output_dir == checkpoint_dir or output_dir.is_relative_to(checkpoint_dir)
            or checkpoint_dir.is_relative_to(output_dir) or config_path.is_relative_to(output_dir)):
        raise ValueError("Export output directory must not overlap the source checkpoint or config")

    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    if artifact_format not in {"sglang", "aspec"}:
        raise ValueError("artifact_format must be sglang or aspec")
    if resolve_draft_algorithm(raw_config) == "mtp":
        raw_config = AutoDraftModelConfig.from_dict(raw_config).to_dict()
    training_contract = _checkpoint_training_contract(model_dir, raw_config)
    export_config = (_serving_config(raw_config) if artifact_format == "sglang"
                     else json.loads(json.dumps(raw_config)))
    if "block_size" in training_contract:
        if training_contract["algorithm"] == "dflash2" and artifact_format == "sglang":
            export_config["dflash_config"]["block_size"] = training_contract["block_size"]
        else:
            export_config["block_size"] = training_contract["block_size"]
    vocab_size = int(raw_config["vocab_size"])
    draft_vocab_size = int(raw_config.get("draft_vocab_size") or vocab_size)

    config = AutoDraftModelConfig.from_file(str(config_path))
    state_dict = _extract_model_weights(_load_fsdp_state_dict(model_dir))
    vocabulary = None
    if draft_vocab_size != vocab_size:
        from aspec.models.vocab import validate_vocab_mapping, vocab_mapping_manifest

        validate_vocab_mapping(state_dict.get("d2t"), state_dict.get("t2d"), draft_vocab_size, vocab_size)
        vocabulary = vocab_mapping_manifest(state_dict["d2t"], state_dict["t2d"])
    else:
        # Old full-vocabulary checkpoints may contain identity mapping buffers.
        state_dict.pop("d2t", None)
        state_dict.pop("t2d", None)
    checkpoint_dtype = Counter(value.dtype for value in state_dict.values() if value.is_floating_point()).most_common(1)[0][0]
    if artifact_format == "aspec" and "embed_tokens.weight" in state_dict:
        checkpoint_dtype = state_dict["embed_tokens.weight"].dtype
    final_dtype = output_dtype or checkpoint_dtype
    model = AutoEagle3DraftModel.from_config(config, torch_dtype=final_dtype if artifact_format == "aspec" else checkpoint_dtype)
    if artifact_format == "aspec" and hasattr(model, "keep_csa_projection_precision"):
        model.keep_csa_projection_precision()
    if artifact_format == "aspec" and output_dtype is not None:
        expected = model.state_dict()
        state_dict = {key: value.to(expected[key].dtype) if value.is_floating_point() else value
                      for key, value in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=artifact_format == "aspec")
    allowed_missing = {"embed_tokens.weight"} if target_model_path else set()
    invalid_missing = sorted(set(missing) - allowed_missing)
    if invalid_missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not match the configured draft model: "
            f"missing={invalid_missing}, unexpected={sorted(unexpected)}"
        )
    if missing:
        try:
            model.load_embedding(target_model_path, embedding_key=embedding_key)
        except (AttributeError, KeyError) as exc:
            raise RuntimeError(
                f"Cannot restore missing {sorted(missing)} from target model "
                f"{target_model_path!r} using {embedding_key!r}"
            ) from exc

    final_dtype = output_dtype or checkpoint_dtype
    if final_dtype != checkpoint_dtype and artifact_format != "aspec":
        model = model.to(dtype=final_dtype)
    serving_state = model.state_dict() if artifact_format == "aspec" else (
        model.state_dict_for_serving() if hasattr(model, "state_dict_for_serving")
        else to_export_keys(model.state_dict())
    )
    tensors = {
        key: value.detach().cpu().contiguous()
        for key, value in serving_state.items()
    }

    eagle4_asset = None
    if resolve_draft_algorithm(raw_config) == "eagle4":
        from aspec.config.eagle4 import eagle4_asset_metadata
        eagle4_asset = eagle4_asset_metadata(raw_config, int(model.candidate_head_steps))
    _prepare_output_directory(output_dir, force=force)
    version = _aspec_version()
    model_path = output_dir / "model.safetensors"
    save_file(
        tensors,
        str(model_path),
        metadata={"format": "pt", "aspec_version": version},
    )

    export_config["draft_vocab_size"] = draft_vocab_size
    export_config["_aspec_version"] = version
    export_config["torch_dtype"] = str(final_dtype).removeprefix("torch.")
    exported_config_path = output_dir / "config.json"
    exported_config_path.write_text(
        json.dumps(export_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": "aspec.serving-export/v1",
        "storage_format": "aspec_native" if artifact_format == "aspec" else "sglang",
        "source_checkpoint": str(model_dir),
        "source_config": str(config_path),
        "source_config_sha256": _sha256_file(config_path),
        "model_sha256": _sha256_file(model_path),
        "config_sha256": _sha256_file(exported_config_path),
        "tensor_count": len(tensors),
        "dtype": str(final_dtype),
        "algorithm": resolve_draft_algorithm(raw_config),
        "training_contract": training_contract,
        "architecture": export_config.get("architectures"),
        "model_type": export_config.get("model_type"),
        "draft_structure": _draft_structure_manifest(raw_config),
        "vocabulary": vocabulary,
    }
    if eagle4_asset is not None:
        manifest["eagle4"] = eagle4_asset
    if raw_config.get("mtp_conditioning"):
        from aspec.config.mtp_conditioning import mtp_conditioning_semantics

        manifest["mtp_conditioning"] = mtp_conditioning_semantics(raw_config)
    if artifact_format == "aspec":
        manifest["compute_dtype"] = str(model.embed_tokens.weight.dtype)
        manifest["tensor_dtypes"] = dict(Counter(str(t.dtype) for t in tensors.values()))
        if len(manifest["tensor_dtypes"]) > 1:
            manifest["dtype"] = "mixed"
        from aspec.config.draft_attention import resolve_draft_attention

        policy = resolve_draft_attention(raw_config)
        manifest["draft_structure"]["attention"].update(csa=policy["csa"], qsa=policy["qsa"])
        if policy.get("dsa") is not None:
            manifest["draft_structure"]["attention"]["dsa"] = policy["dsa"]
        algorithm = training_contract["algorithm"]
        block_proposal = algorithm in {"dspark", "dflash"} and not raw_config.get("mtp_conditioning")
        mla_proposal = raw_config.get("architectures") == ["Eagle3DeepseekV2ForCausalLM"]
        proposal = ("aspec.draft_artifact.MLAEagleConsumer" if mla_proposal else
                    "aspec.draft_artifact.BlockDraftConsumer" if block_proposal else None)
        manifest["draft_structure"]["loader_contract"] = {
            "runtime": "aspec", "entrypoint": "aspec.draft_artifact.load_native_draft",
            "proposal_entrypoint": proposal,
            "validation": "required_per_artifact",
            "scope": ("asset and proposal; caller owns verifier features/cache/verification"
                      if proposal else "asset loading and weights-only warmstart; no native proposal component"),
            "borrowed_tensors": ["Target output head"] if algorithm in {"dspark", "dflash"} else [],
        }
    (output_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--target-model-path")
    parser.add_argument("--format", choices=("sglang", "aspec"), default="sglang", dest="artifact_format")
    parser.add_argument("--embedding-key", default="model.embed_tokens.weight")
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default=None
    )
    parser.add_argument("-f", "--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir or Path(str(args.input_dir).rstrip("/\\") + "_hf")
    dtype = getattr(torch, args.dtype) if args.dtype else None
    manifest = convert_checkpoint(
        input_dir=args.input_dir,
        output_dir=output_dir,
        config_path=args.config,
        target_model_path=args.target_model_path,
        embedding_key=args.embedding_key,
        output_dtype=dtype,
        force=args.force,
        artifact_format=args.artifact_format,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
