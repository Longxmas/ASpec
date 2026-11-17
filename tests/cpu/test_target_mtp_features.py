# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Moving frozen MTP capture must preserve the Draft's numerical contract."""

import copy
import json
import queue
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from aspec.config.mooncake_config import MooncakeConfig
from aspec.feature_contract import (
    FeatureContractError,
    build_target_feature_contract,
    shared_feature_union,
)
from aspec.inference.mtp_features import FrozenTargetMTP
from aspec.inference.target_prefill import TargetPrefillDirectSink, TargetPrefillRequest
from aspec.models.dmtp import MTPPrefix
from aspec.models.draft.mtp import NativeMTPConfig, NativeMTPDraftModel
from tests.cpu.test_dmtp import _fusion_inputs, _nonzero_model, native_config
from tests.test_target_prefill import _Store


@pytest.fixture
def native_weights(monkeypatch):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(29)
        source = NativeMTPDraftModel(NativeMTPConfig(native_config=native_config()))
    state = {source.native_key(name): tensor.detach().clone()
             for name, tensor in source.state_dict().items()}
    state["model.language_model.embed_tokens.weight"] = source.embed_tokens.weight.detach().clone()
    monkeypatch.setattr("aspec.models.draft.base.load_tensor_from_pretrained", lambda _, key: state[key])
    monkeypatch.setattr("aspec.inference.mtp_features.load_tensor_from_pretrained", lambda _, key: state[key])
    monkeypatch.setattr("aspec.inference.mtp_features.AutoConfig.from_pretrained",
                        lambda _: SimpleNamespace(to_dict=lambda: native_config()))
    return state


def capture_policy():
    return {"native_config": native_config(), "embedding_key": "model.language_model.embed_tokens.weight"}


def target_model(reference):
    config = copy.deepcopy(reference.draft_model.config)
    config.mtp_conditioning = {**config.mtp_conditioning, "feature_source": "target"}
    draft = type(reference.draft_model)(config)
    draft.initialize_mtp_conditioning("checkpoint")
    state = draft.state_dict()
    state.update({name: value for name, value in reference.draft_model.state_dict().items()
                  if not name.startswith("mtp_source.")})
    draft.load_state_dict(state, strict=True)
    return type(reference)(draft, block_size=5, num_anchors=2, attention_backend="sdpa")


@pytest.mark.parametrize("algorithm", ["dflash", "dspark"])
def test_target_capture_preserves_loss_outputs_and_trainable_gradients(native_weights, algorithm):
    reference = _nonzero_model(algorithm)
    reference.draft_model.initialize_mtp_conditioning("checkpoint")
    actual = target_model(reference)
    producer = FrozenTargetMTP.from_target(capture_policy(), "checkpoint",
                                           reference.draft_model.embed_tokens.weight,
                                           device="cpu", dtype=torch.float32)
    inputs, other = _fusion_inputs(), _fusion_inputs()
    inputs.pop("draft_prefix")
    other.pop("draft_prefix")
    captured = producer(other["input_ids"][0], other["last_hidden_states"][0])[None]
    captured.requires_grad_()
    other["draft_prefix"] = MTPPrefix(captured[:, 1:])
    output, expected = actual(**other), reference(**inputs)
    torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-6)
    output[0].backward()
    expected[0].backward()
    old = dict(reference.named_parameters())
    for name, parameter in actual.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            torch.testing.assert_close(parameter.grad, old[name].grad, rtol=2e-5, atol=2e-6, msg=name)
    assert captured.grad is None
    assert actual.draft_model.mtp_source is None
    assert not any(name.startswith("mtp_source.") for name in actual.draft_model.state_dict())
    assert actual.draft_model.mtp_binding.fingerprint.numel() == 32
    assert all(p.grad is None and not p.requires_grad for p in producer.parameters())


def test_capture_uses_complete_document_history_and_keeps_the_target_embedding(native_weights):
    embedding = native_weights["model.language_model.embed_tokens.weight"]
    producer = FrozenTargetMTP.from_target(capture_policy(), "checkpoint", embedding,
                                           device="cpu", dtype=torch.float32)
    assert producer.embedding_weight.data_ptr() == embedding.data_ptr()
    assert producer.status()["embedding_bytes"] == 0
    ids, hidden = torch.arange(7), torch.randn(7, 16)
    full = producer(ids, hidden)
    assert not full.requires_grad and not full[0].any()
    for length in range(2, 8):
        torch.testing.assert_close(producer(ids[:length], hidden[:length])[-1], full[length - 1])
    changed = hidden.clone()
    changed[4:] += 100
    torch.testing.assert_close(producer(ids, changed)[:5], full[:5])
    assert not producer.train().training


def test_replication_is_explicit_and_native_binding_survives_checkpoints(native_weights):
    with pytest.raises(ValueError, match="loaded Target embedding"):
        FrozenTargetMTP.from_target(capture_policy(), "checkpoint", None, device="cpu", dtype=torch.float32)
    producer = FrozenTargetMTP.from_target(capture_policy(), "checkpoint", None, device="cpu",
                                           dtype=torch.float32, allow_replicated_embedding=True)
    assert producer.status()["embedding_bytes"] == 32 * 16 * 4
    reference = _nonzero_model("dflash")
    actual = target_model(reference)
    binding = actual.draft_model.mtp_binding
    original = binding.state_dict()
    restored = copy.deepcopy(binding)
    restored.load_state_dict(original, strict=True)
    restored.validate_native_weights("checkpoint")
    native_weights["mtp.fc.weight"][0, 0] += 1
    with pytest.raises(ValueError, match="fingerprint"):
        restored.validate_native_weights("checkpoint")


@pytest.mark.parametrize("nested", [False, True])
def test_quantized_native_checkpoints_fail_before_capture(monkeypatch, nested):
    target = {"text_config": native_config()}
    (target["text_config"] if nested else target)["quantization_config"] = {"quant_method": "fp8"}
    monkeypatch.setattr("aspec.inference.mtp_features.AutoConfig.from_pretrained",
                        lambda _: SimpleNamespace(to_dict=lambda: target))
    with pytest.raises(ValueError, match="unquantized"):
        FrozenTargetMTP.from_target(capture_policy(), "checkpoint", None, device="cpu", dtype=torch.float32)


def test_target_conditioning_requires_captured_features(native_weights):
    model = target_model(_nonzero_model("dflash"))
    inputs = _fusion_inputs()
    inputs.pop("draft_prefix")
    with pytest.raises(ValueError, match="captured mtp_hidden_states"):
        model(**inputs)


@pytest.mark.parametrize("algorithm", ["dflash", "dspark"])
def test_target_conditioned_native_export_roundtrip(native_weights, tmp_path, monkeypatch, algorithm):
    from aspec.commands import convert_to_hf
    from aspec.draft_artifact import load_native_draft

    model = target_model(_nonzero_model(algorithm)).eval()
    draft = model.draft_model
    checkpoint = tmp_path / "iter_0000003"
    (checkpoint / "model").mkdir(parents=True)
    (checkpoint / "model/.metadata").touch()
    (checkpoint / "draft_config.json").write_text(json.dumps(draft.config.to_dict()))
    monkeypatch.setattr(convert_to_hf, "_load_fsdp_state_dict", lambda _: {
        "draft_model." + key: value for key, value in draft.state_dict().items()})
    output = tmp_path / "native"
    convert_to_hf.convert_checkpoint(input_dir=checkpoint, output_dir=output, artifact_format="aspec")
    restored = load_native_draft(output)
    assert restored.mtp_source is None
    restored.mtp_binding.validate_native_weights("checkpoint")
    for name, tensor in draft.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
    reloaded = type(model)(restored, block_size=5, num_anchors=2, attention_backend="sdpa").eval()
    inputs = _fusion_inputs()
    with torch.no_grad():
        torch.testing.assert_close(reloaded(**inputs), model(**inputs), rtol=0, atol=0)


def test_target_provider_does_not_import_draft_execution_dependencies():
    result = subprocess.run([sys.executable, "-c", """
import sys
for name in ('aspec.models.draft', 'flash_attn', 'quack', 'quack_kernels', 'cutlass'):
    sys.modules[name] = None
from aspec.inference.mtp_features import FrozenTargetMTP
assert FrozenTargetMTP.__module__ == 'aspec.inference.mtp_features'
"""], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_target_feature_memory_and_buffer_capacity_are_separate():
    from aspec.config.mtp_conditioning import mtp_conditioning_memory

    policy = {"mode": "hidden", "native_config": native_config()}
    legacy = mtp_conditioning_memory(policy)
    local = mtp_conditioning_memory({**policy, "feature_source": "target"})
    tp = mtp_conditioning_memory({**policy, "feature_source": "target"}, target_tensor_parallel_size=2)
    assert local["frozen_weight_bytes"] == 0
    assert local["target"]["frozen_weight_bytes"] == legacy["frozen_weight_bytes"]
    assert local["trainable_weight_bytes"] == legacy["trainable_weight_bytes"]
    assert local["target"]["replicated_embedding_bytes"] == 0
    assert tp["target"]["replicated_embedding_bytes"] == 32 * 16 * 2
    plain = MooncakeConfig(hidden_dim=16, num_aux_layers=1, max_seq_len=128, enable_gpu_direct=True)
    captured = MooncakeConfig(hidden_dim=16, num_aux_layers=1, max_seq_len=128,
                              include_mtp_hidden_states=True, enable_gpu_direct=True)
    assert captured.host_buffer_size > plain.host_buffer_size
    assert captured.gpu_buffer_size > plain.gpu_buffer_size


def test_public_plan_requests_target_features_and_rejects_unsupported_capture(monkeypatch, tmp_path):
    from aspec.config.train_config import load_config
    from aspec.planning import compile_execution_plan

    monkeypatch.setenv("ASPEC_TRAIN_DATA", "examples/data/sample_conversations.jsonl")
    monkeypatch.setenv("ASPEC_EVAL_DATA", "examples/data/sample_conversations.jsonl")
    path = Path(__file__).parents[2] / "configs/recipes/qwen3_5_9b_dmtp.yaml"
    config = load_config(str(path), save_snapshot=False)
    plan = compile_execution_plan(config)
    assert "mtp_hidden_states" in plan.workload.features["tensors"]
    assert plan.workload.objective["mtp_conditioning"]["feature_computation"] == "frozen_target_side_native_mtp"
    args = plan.to_runtime_args()
    assert MooncakeConfig.from_flat_args(args).include_mtp_hidden_states
    assert plan.policy.capacity["device_memory"]["target_mtp_features"]["frozen_weight_bytes"] > 0
    plain = json.loads(Path(config.model.draft_model_config).read_text())
    plain.pop("mtp_conditioning")
    plain_path = tmp_path / "plain.json"
    plain_path.write_text(json.dumps(plain))
    source = copy.deepcopy(config)
    source.model.draft_model_config = str(plain_path)
    shared = compile_execution_plan(source, source_feature_contract=plan.workload.features)
    assert shared.policy.capacity["device_memory"]["target_mtp_features"] == plan.policy.capacity["device_memory"]["target_mtp_features"]
    assert "mtp_conditioning" not in shared.policy.capacity["device_memory"]
    config.inference.sglang.pp_size = 2
    with pytest.raises(ValueError, match="PP1"):
        compile_execution_plan(config)
    config.inference.sglang.pp_size = 1
    target = tmp_path / "quantized-target"
    target.mkdir()
    (target / "config.json").write_text(json.dumps({**plan.workload.features["mtp"]["native_config"],
        "quantization_config": {"quant_method": "fp8"}}))
    config.model.target_model_path = str(target)
    with pytest.raises(ValueError, match="unquantized"):
        compile_execution_plan(config)


def feature_contract(*, mtp=False, last=False):
    return build_target_feature_contract(drafter_family="dflash", target_hidden_size=16,
                                         target_num_hidden_layers=4, aux_hidden_state_layers=[1],
                                         store_last_hidden_states=last,
                                         **({"mtp": capture_policy()} if mtp else {}))


def test_capture_contract_union_keeps_optional_mtp_and_rejects_other_bindings():
    plain, conditioned = feature_contract(), feature_contract(mtp=True)
    union = shared_feature_union([plain, conditioned])
    assert union == conditioned
    assert set(plain["tensors"]) == {"input_ids", "hidden_states"}
    assert conditioned["tensors"]["mtp_hidden_states"]["alignment"] == "anchor_post_norm_v1"
    other = build_target_feature_contract(drafter_family="dflash", target_hidden_size=16,
        target_num_hidden_layers=4, aux_hidden_state_layers=[1], store_last_hidden_states=False,
        mtp={**capture_policy(), "embedding_key": "other.weight"})
    with pytest.raises(FeatureContractError, match="incompatible"):
        shared_feature_union([conditioned, other])


def test_received_mtp_features_keep_the_batch_dimension_through_collation():
    from aspec.data.utils import DataCollatorWithPadding, pack_loss_mask, serialize_packed_loss_mask
    from aspec.models.target.eagle3_target_model import Eagle3TargetOutput
    from aspec.training.data_fetcher import MooncakeDataset, TrainSample

    contract = feature_contract(mtp=True)
    output = Eagle3TargetOutput(hidden_states=torch.randn(7, 16).bfloat16(), input_ids=torch.arange(7),
                               mtp_hidden_states=torch.randn(7, 16).bfloat16())
    tensors = output.to_tensor_dict()
    store = SimpleNamespace(get=Mock(return_value=output), remove_eagle3_tensors=Mock())
    samples = queue.Queue()
    samples.put(TrainSample("sample", {name: tuple(value.shape) for name, value in tensors.items()},
        {name: value.dtype for name, value in tensors.items()},
        packed_loss_mask=serialize_packed_loss_mask(pack_loss_mask(torch.ones(7))),
        metadata={"aspec_feature_contract_id": contract["contract_id"]}))
    samples.put(None)
    received = list(MooncakeDataset(samples, store, torch.device("cpu"), feature_contract=contract))
    assert received[0]["mtp_hidden_states"].shape == (1, 7, 16)
    batch = DataCollatorWithPadding()(received)
    torch.testing.assert_close(batch["mtp_hidden_states"][0], output.mtp_hidden_states, rtol=0, atol=0)
    assert store.remove_eagle3_tensors.call_args.kwargs["has_mtp_hidden_states"]


def test_offline_dataset_keeps_captured_mtp_features(tmp_path, monkeypatch):
    from aspec.config.train_config import load_config
    from aspec.offline.dataset import OfflineDataset, target_identity
    from aspec.planning import compile_execution_plan

    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text(json.dumps(native_config()))
    (target / "model.safetensors").write_bytes(b"identity-only")
    contract = feature_contract(mtp=True)
    dataset = OfflineDataset(tmp_path / "data", create=True,
        binding={"target_identity": target_identity(target), "feature_contract": contract},
        last_hidden_states_prenorm=False)
    ids, hidden, mtp = torch.arange(7), torch.randn(7, 16).bfloat16(), torch.randn(7, 16).bfloat16()
    for split in ["train", "eval"]:
        dataset.append(split, data_id=split, tensors={"input_ids": ids, "hidden_states": hidden,
                       "mtp_hidden_states": mtp}, packed_loss_mask=[2, 5],
                       metadata={"aspec_feature_contract_id": contract["contract_id"]})
    dataset.finalize({"train": 1, "eval": 1})
    replay = OfflineDataset(dataset.root)
    torch.testing.assert_close(replay.load("train")["mtp_hidden_states"], mtp, rtol=0, atol=0)
    monkeypatch.setenv("ASPEC_TRAIN_DATA", "unused")
    monkeypatch.setenv("ASPEC_EVAL_DATA", "unused")
    root = Path(__file__).parents[2]
    config = load_config(str(root / "configs/recipes/qwen3_5_9b_dmtp.yaml"), save_snapshot=False)
    raw = _nonzero_model("dflash").draft_model.config.to_dict()
    raw["mtp_conditioning"]["feature_source"] = "target"
    draft_path = tmp_path / "draft.json"
    draft_path.write_text(json.dumps(raw))
    config.model.target_model_path = str(target)
    config.model.native_context_length = 128
    config.model.draft_model_config = str(draft_path)
    config.training.max_seq_length = 128
    config.training.training_num_gpus_per_node = 1
    config.training.draft_accumulation_steps = 1
    config.training.dflash_num_target_layers = 1
    config.training.dflash_block_size = config.training.ttt_length = 5
    config.dataset.train_data_path = config.dataset.eval_data_path = None
    config.inference.inference_engine_type = "offline"
    config.inference.offline.data_path = str(dataset.root)
    config.inference.inference_num_gpus = 0
    config.inference.store_last_hidden_states = False
    config.mooncake.update(data_plane="store", protocol="tcp", enable_gpu_direct=False,
                           require_nvlink=False, allow_host_fallback=True, hidden_dim=16)
    plan = compile_execution_plan(config)
    assert "mtp_hidden_states" in plan.workload.features["tensors"]
    assert "target_mtp_features" not in plan.policy.capacity["device_memory"]


@pytest.mark.parametrize("fail_second", [False, True])
def test_direct_capture_is_document_local_and_keeps_atomic_publication(fail_second):
    class MTPStore(_Store):
        def reserve_p2p_output(self, **kwargs):
            reservation = super().reserve_p2p_output(**kwargs)
            if kwargs.get("store_mtp_hidden_states"):
                reservation.tensors["mtp_hidden_states"] = torch.empty(
                    kwargs["feature_sequence_length"], kwargs["hidden_size"])
            return reservation

    calls = []
    def produce(ids, hidden):
        calls.append(ids.clone())
        if fail_second and len(calls) == 2:
            raise RuntimeError("MTP failed")
        output = hidden + ids[:, None]
        output[0] = 0
        return output

    store = MTPStore()
    sink = TargetPrefillDirectSink(store=store,
        requests=[TargetPrefillRequest("a", "a", 3), TargetPrefillRequest("b", "b", 4)],
        capture_layers=[1], hidden_size=16, store_last_hidden_states=False,
        context_parallel_size=1, mtp_feature_producer=produce)
    ids, hidden = torch.arange(7), torch.randn(7, 16)
    sink.begin(input_ids=ids, request_ids=["a", "b"], sequence_lengths=[3, 4], prefix_lengths=[0, 0])
    sink.capture(1, hidden, None)
    if fail_second:
        with pytest.raises(RuntimeError, match="MTP failed"):
            sink.finish(hidden)
        sink.abort()
        assert all(r.state == "discarded" for r in store.reservations)
    else:
        sink.finish(hidden)
        for rid in ["a", "b"]:
            sink.commit_request(rid)
        sink.publish()
        assert all(r.state == "published" for r in store.reservations)
        torch.testing.assert_close(store.reservations[0].tensors["mtp_hidden_states"][1:], hidden[1:3] + ids[1:3, None])
        torch.testing.assert_close(store.reservations[1].tensors["mtp_hidden_states"][1:], hidden[4:] + ids[4:, None])
        assert not store.reservations[1].tensors["mtp_hidden_states"][0].any()
    assert [x.tolist() for x in calls] == [[0, 1, 2], [3, 4, 5, 6]]
