# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Target MTP transport and ownership checks for the full runtime."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from aspec.config.mooncake_config import MooncakeConfig
from aspec.inference.sglang_v0521 import TrainingFeaturePublisher
from aspec.transfer.mooncake.eagle_store import EagleMooncakeStore


def test_store_roundtrip_and_cleanup_include_mtp_features():
    store = EagleMooncakeStore(MooncakeConfig(hidden_dim=16, num_aux_layers=1, max_seq_len=7,
                                               include_mtp_hidden_states=True))
    store._initialized = True
    store._init_event.set()
    values = {}
    store._put_raw_tensors = lambda keys, tensors: values.update(zip(keys, tensors, strict=True))
    store.wait_for_keys = Mock()
    store._get_tensors_via_host_buffer = lambda keys, specs, device: {
        name: values[key].clone() for key, (name, _, _) in zip(keys, specs, strict=True)}
    hidden, ids, mtp = torch.randn(7, 16), torch.arange(7), torch.randn(7, 16)
    metadata = store.put("record", hidden, ids, None, mtp_hidden_states=mtp)
    output = store.get("record", metadata["shapes"], metadata["dtypes"], "cpu")
    torch.testing.assert_close(output.mtp_hidden_states, mtp.bfloat16(), rtol=0, atol=0)
    assert "mtp_hidden_states" in output.to_tensor_dict()
    store._store = SimpleNamespace(batch_remove=Mock(return_value=[0, 0, 0]))
    store.remove_eagle3_tensors("record", has_mtp_hidden_states=True)
    assert store._store.batch_remove.call_args.args[0] == ["record_hs", "record_ids", "record_mhs"]

def test_cp1_store_path_preserves_mtp_features():
    store = EagleMooncakeStore(MooncakeConfig(hidden_dim=16, num_aux_layers=1, max_seq_len=7))
    store._initialized = True
    store._init_event.set()
    store.put = Mock(return_value={"mooncake_key": "record"})
    mtp = torch.randn(7, 16)
    store.put_direct_cp_shards("record", torch.randn(7, 16), torch.arange(7), None,
                               cp_size=1, max_seq_length=5, mtp_hidden_states=mtp)
    torch.testing.assert_close(store.put.call_args.kwargs["mtp_hidden_states"], mtp[:5])
    with pytest.raises(ValueError, match="CP1"):
        store.put_direct_cp_shards("record", torch.randn(7, 16), torch.arange(7), None,
                                   cp_size=2, max_seq_length=5, mtp_hidden_states=mtp)

def test_buffered_capture_and_cancellation_own_the_mtp_store_objects():
    store = EagleMooncakeStore(MooncakeConfig(hidden_dim=16, num_aux_layers=1, max_seq_len=7,
                                               include_mtp_hidden_states=True))
    store._initialized = True
    store._init_event.set()
    store._put_raw_tensors = Mock()
    store.flush = Mock()
    store.discard_unpublished_objects = Mock()
    publisher = TrainingFeaturePublisher(dict(layers=[1], hidden_size=16,
        context_parallel_size=1, max_sequence_length=7, store_last_hidden_states=False,
        direct_prefill=False, attention_backend="flex_attention"))
    publisher.capture_layers = (1,)
    publisher.store = store
    publisher.mtp = Mock(side_effect=lambda ids, hidden: hidden + ids[:, None])
    batch = SimpleNamespace(forward_mode=SimpleNamespace(is_extend=lambda: True),
        is_prefill_only=True, extend_seq_lens_cpu=[3, 4], extend_prefix_lens_cpu=[0, 0],
        request_custom_params=[{"aspec_data_id": rid, "aspec_request_id": rid} for rid in ["a", "b"]],
        rids=["a", "b"])
    ids, hidden = torch.arange(7), torch.randn(7, 16)
    keys = publisher.publish(ids, hidden, batch, hidden)
    assert [call.args[0].tolist() for call in publisher.mtp.call_args_list] == [[0, 1, 2], [3, 4, 5, 6]]
    publisher.resolve_outputs(["a", "b"], [keys[0]])
    removed = store.discard_unpublished_objects.call_args.args[0]
    assert set(removed) == {keys[1] + suffix for suffix in ["_hs", "_ids", "_mhs"]}
    assert not publisher.pending_outputs
