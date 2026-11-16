# MIT License
#
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Adapted from TorchSpec 6c042a87140a84d13839e341ece2c5c3ada918bc:
# tests/test_sgl_engine_factory.py. Namespace bindings and ASpec regression coverage adapted.

"""Tests for SGLang engine factory: node-local engine allocation and tp_size validation."""

import asyncio
import os
from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
import torch

from aspec.feature_contract import build_target_feature_contract
from aspec.transfer.mooncake.p2p import (
    MooncakeP2PDescriptor,
    MooncakeP2PTensorDescriptor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(**overrides):
    """Build a minimal args Namespace for _init_sgl_engines."""
    defaults = dict(
        target_model_path="/fake/model",
        inference_num_gpus=8,
        inference_num_gpus_per_engine=8,
        inference_num_gpus_per_node=8,
        sglang_nnodes=1,
        sglang_init_timeout=10,
        sglang_tp_size=None,
        sglang_pp_size=1,
        aux_hidden_states_layers=None,
        sglang_mem_fraction_static=0.8,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _make_pg(num_bundles):
    """Create a fake placement group tuple with *num_bundles* bundles.

    Returns (pg_obj, reordered_bundle_indices, reordered_gpu_ids) where
    indices and gpu_ids are simple identity lists.
    """
    pg_obj = MagicMock(name="pg_obj")
    indices = list(range(num_bundles))
    gpu_ids = list(range(num_bundles))
    return (pg_obj, indices, gpu_ids)


def _build_mock_ray_infrastructure():
    """Set up mock Ray so _init_sgl_engines can run without a real cluster.

    Returns (engines, init_calls, mock_ray_remote) where:
      - engines: list collecting every actor created via .remote()
      - init_calls: list of (engine, kwargs) for every engine.init.remote() call
    """
    engines = []
    init_calls = []

    def make_engine(**constructor_kwargs):
        engine = MagicMock(name=f"engine_{len(engines)}")
        engine._constructor_kwargs = constructor_kwargs

        # .init.remote() → record call, return a sentinel ObjectRef
        def _init_remote(**kw):
            ref = MagicMock(name=f"init_ref_{len(init_calls)}")
            init_calls.append((engine, kw))
            return ref

        engine.init.remote.side_effect = _init_remote

        # get_node_ip / find_free_port → return ObjectRefs resolved by ray.get
        engine.get_node_ip.remote.return_value = MagicMock(name="ip_ref")
        engine.find_free_port.remote.return_value = MagicMock(name="port_ref")

        engines.append(engine)
        return engine

    # ray.remote(SglEngine) → SglRayActor
    mock_sgl_ray_actor = MagicMock(name="SglRayActor")

    # SglRayActor.options(...).remote(**kwargs) → make_engine(**kwargs)
    mock_sgl_ray_actor.options.return_value.remote.side_effect = make_engine

    def mock_ray_remote(cls):
        return mock_sgl_ray_actor

    return engines, init_calls, mock_ray_remote


# ---------------------------------------------------------------------------
# Topology tests for _init_sgl_engines
# ---------------------------------------------------------------------------


class TestSglEngineTopology:
    """Test engine count, replica count, and node_rank assignment."""

    def _call(self, args, pg, mooncake_config=None):
        engines, init_calls, mock_ray_remote = _build_mock_ray_infrastructure()

        with (
            patch("aspec.inference.factory.ray") as mock_ray,
            patch("aspec.inference.factory.get_aspec_env_vars", return_value={}),
        ):
            mock_ray.get_runtime_context.return_value.get_node_id.return_value = "ab" * 28
            mock_ray.remote.side_effect = mock_ray_remote
            next_port = 10000

            def mock_ray_get(refs, **kwargs):
                nonlocal next_port
                if isinstance(refs, list):
                    return None
                result = next_port
                next_port += 2
                return result

            mock_ray.get.side_effect = mock_ray_get

            from aspec.inference.factory import _init_sgl_engines

            result = _init_sgl_engines(args, pg, mooncake_config)

        return result, engines, init_calls

    # -- Single-node cases --------------------------------------------------

    def test_independent_sessions_do_not_always_probe_same_initial_ports(self):
        args = _make_args()
        with patch("aspec.inference.factory.secrets.randbelow", side_effect=[5000, 17000]):
            _, first, *_ = self._call(args, _make_pg(8))
            _, second, *_ = self._call(args, _make_pg(8))
        first[0].find_free_port.remote.assert_called_once_with(start_port=15000, consecutive=2)
        second[0].find_free_port.remote.assert_called_once_with(start_port=27000, consecutive=2)

    def test_single_node_4_engines(self):
        """8 GPUs, 2 GPUs/engine → 4 engines, all returned."""
        args = _make_args(inference_num_gpus=8, inference_num_gpus_per_engine=2)
        pg = _make_pg(8)

        result, engines, init_calls = self._call(args, pg)

        assert len(engines) == 4
        assert len(result) == 4
        assert [e for e, _ in init_calls] == result == engines

    def test_single_node_1_engine(self):
        """8 GPUs, 8 GPUs/engine → 1 engine."""
        args = _make_args(inference_num_gpus=8, inference_num_gpus_per_engine=8)
        pg = _make_pg(8)

        result, engines, *_ = self._call(args, pg)

        assert len(engines) == 1
        assert len(result) == 1

    def test_single_node_gpu_ids(self):
        """A remote service owns allocation; Draft needs no inference PG."""
        args = _make_args(inference_num_gpus=8, inference_num_gpus_per_engine=2)
        _, engines, *_ = self._call(args, None)

        for i, e in enumerate(engines):
            assert "base_gpu_id" not in e._constructor_kwargs
            assert e._constructor_kwargs["num_gpus_per_engine"] == 2

    def test_single_node_no_dist_init_addr(self):
        """Single-node engines should NOT receive dist_init_addr."""
        args = _make_args(inference_num_gpus=8, inference_num_gpus_per_engine=2)
        pg = _make_pg(8)

        _, _, init_calls = self._call(args, pg)

        for _, kw in init_calls:
            assert "dist_init_addr" not in kw


class TestSglEngineTpSizeValidation:
    """Test fail-closed tp_size and pp_size validation in SglEngine.init()."""

    def _make_engine(self, num_gpus_per_engine=8, node_rank=0, **args_overrides):
        from aspec.inference.engine.sgl_engine import SglEngine

        args = _make_args(**args_overrides)
        return SglEngine(
            args=args,
            rank=0,
            base_gpu_id=0,
            num_gpus_per_engine=num_gpus_per_engine,
            node_rank=node_rank,
        )

    def _init_engine(self, engine):
        """Call engine.init() with all heavy deps mocked out."""
        mock_sgl = MagicMock()
        mock_sgl.Engine.return_value = MagicMock(model_config=MagicMock(hidden_size=4096))
        with (
            patch.object(engine, "setup_gpu", return_value=0),
            patch("aspec.inference.engine.sgl_engine.sgl", mock_sgl),
            patch(
                "aspec.inference.sglang_v0521.training_engine_kwargs",
                side_effect=lambda kwargs, args: kwargs,
            ),
            patch(
                "aspec.inference.engine.sgl_engine.get_default_eagle3_aux_layer_ids",
                return_value=[0, 1, 2],
            ),
            patch.object(engine, "_get_hidden_size_from_engine", return_value=4096),
        ):
            engine.init(mooncake_config=None, pre_allocated_port=10000)
        return mock_sgl

    def test_pp_size_must_be_1(self):
        """pp_size != 1 should fail before engine initialization."""
        engine = self._make_engine(sglang_pp_size=2)

        with pytest.raises(ValueError, match="pp_size must be 1"):
            self._init_engine(engine)

    def test_tp_size_correct_single_node(self):
        """sglang_tp_size=8, nnodes=1, gpus_per_engine=8 → OK, tp_size=8 passed to sgl.Engine."""
        engine = self._make_engine(
            num_gpus_per_engine=8,
            sglang_tp_size=8,
            sglang_nnodes=1,
        )
        mock_sgl = self._init_engine(engine)
        kwargs = mock_sgl.Engine.call_args[1]
        assert kwargs["tp_size"] == 8
        assert kwargs["nnodes"] == 1

    def test_producer_store_is_bound_to_aspec(self, monkeypatch):
        monkeypatch.setenv("SGLANG_SPEC_TRAINING_PACKAGE", "angelspec")
        engine = self._make_engine(num_gpus_per_engine=1, sglang_tp_size=1)
        self._init_engine(engine)
        assert os.environ["SGLANG_SPEC_TRAINING_PACKAGE"] == "aspec"

    @pytest.mark.parametrize("nnodes,node_rank", [(2, 0), (1, 1)])
    def test_engine_rejects_cross_node_tp_before_init(self, nnodes, node_rank):
        with pytest.raises(ValueError, match="node-local"):
            self._make_engine(sglang_nnodes=nnodes, node_rank=node_rank)

    def test_tp_size_wrong_single_node(self):
        """sglang_tp_size=4 with gpus_per_engine=8 must fail closed."""
        engine = self._make_engine(
            num_gpus_per_engine=8,
            sglang_tp_size=4,
            sglang_nnodes=1,
        )
        with pytest.raises(ValueError, match="sglang_tp_size"):
            self._init_engine(engine)

    def test_tp_size_defaults_when_not_configured(self):
        engine = self._make_engine(num_gpus_per_engine=8, sglang_tp_size=None)
        mock_sgl = self._init_engine(engine)
        assert mock_sgl.Engine.call_args.kwargs["tp_size"] == 8

    def test_output_metadata_is_validated_and_carries_contract_identity(self):
        contract = build_target_feature_contract(
            drafter_family="dspark",
            target_hidden_size=4096,
            target_num_hidden_layers=36,
            aux_hidden_state_layers=[0, 1, 2],
            store_last_hidden_states=True,
        )
        engine = self._make_engine(
            num_gpus_per_engine=1,
            sglang_tp_size=1,
            aux_hidden_states_layers=[0, 1, 2],
            store_last_hidden_states=True,
            aspec_feature_contract=contract,
        )
        mock_sgl = self._init_engine(engine)
        mock_sgl.Engine.return_value.async_generate = AsyncMock(return_value=[
            {
                "meta_info": {
                    "spec_training_mooncake_store_keys": ["sample-key"],
                    "prompt_tokens": 3,
                }
            }
        ])

        output = asyncio.run(
            engine.generate(
                data_id=["sample"],
                input_ids_ref=[torch.tensor([1, 2, 3])],
            )
        )

        assert output[0]["metadata"]["aspec_feature_contract_id"] == contract["contract_id"]
        assert output[0]["tensor_shapes"]["hidden_states"] == (3, 12288)
        status = engine.get_feature_contract_status()
        assert status["state"] == "validated"
        assert status["binding_validations"] == 1
        assert status["metadata_validations"] == 1

    def test_cudnn_flex_output_metadata_declares_full_draft_layout(self):
        contract = build_target_feature_contract(
            drafter_family="dspark",
            target_hidden_size=4096,
            target_num_hidden_layers=36,
            aux_hidden_state_layers=[0, 1, 2],
            store_last_hidden_states=True,
        )
        engine = self._make_engine(
            num_gpus_per_engine=1,
            sglang_tp_size=1,
            aux_hidden_states_layers=[0, 1, 2],
            store_last_hidden_states=True,
            aspec_feature_contract=contract,
            attention_backend="cudnn_flex_attention",
            draft_context_parallel_size=1,
        )
        mock_sgl = self._init_engine(engine)
        mock_sgl.Engine.return_value.async_generate = AsyncMock(
            return_value=[
                {
                    "meta_info": {
                        "spec_training_mooncake_store_keys": ["sample-key"],
                        "prompt_tokens": 3,
                    }
                }
            ]
        )

        output = asyncio.run(
            engine.generate(
                data_id=["sample"],
                input_ids_ref=[torch.tensor([1, 2, 3])],
            )
        )

        metadata = output[0]["metadata"]
        assert metadata["draft_attention_backend"] == "cudnn_flex_attention"
        assert metadata["draft_context_parallel_size"] == 1
        assert metadata["draft_feature_layout"] == "full"

    def test_transfer_engine_descriptor_reports_source_transport_evidence(self):
        contract = build_target_feature_contract(
            drafter_family="dspark",
            target_hidden_size=4096,
            target_num_hidden_layers=36,
            aux_hidden_state_layers=[0, 1, 2],
            store_last_hidden_states=True,
        )
        engine = self._make_engine(
            num_gpus_per_engine=1,
            sglang_tp_size=1,
            aux_hidden_states_layers=[0, 1, 2],
            store_last_hidden_states=True,
            aspec_feature_contract=contract,
            aspec_run_id="run-1",
            aspec_attempt_id="attempt-1",
            mooncake_data_plane="transfer_engine",
            mooncake_protocol="tcp",
            mooncake_allow_host_fallback=False,
        )
        mock_sgl = self._init_engine(engine)
        descriptor = MooncakeP2PDescriptor(
            version=3,
            run_id="run-1",
            attempt_id="attempt-1",
            store_key="sample-key",
            data_id="sample",
            request_id="request-1",
            source_engine="0",
            source_endpoint="192.0.2.32:15000",
            source_release_endpoint="192.0.2.32:16000",
            slot_base_ptr=0x100000,
            slot_id=0,
            generation=1,
            payload_bytes=16,
            tensors=(
                MooncakeP2PTensorDescriptor(
                    "hidden_states", (8,), "uint8", 0, 8
                ),
            ),
        )
        mock_sgl.Engine.return_value.async_generate = AsyncMock(
            return_value=[
                {
                    "meta_info": {
                        "spec_training_mooncake_store_keys": [descriptor.to_key()],
                        "prompt_tokens": 3,
                    }
                }
            ]
        )

        asyncio.run(
            engine.generate(
                data_id=["sample"],
                input_ids_ref=[torch.tensor([1, 2, 3])],
            )
        )

        transport = engine.get_feature_contract_status()["transport"]
        assert transport["activation"] == "descriptor_observed"
        assert transport["endpoint"] == ["192.0.2.32:15000"]
        assert transport["run_id"] == "run-1"
        assert transport["attempt_id"] == "attempt-1"
        assert transport["metrics"]["source_descriptors"] == 1
        assert transport["metrics"]["source_payload_bytes"] == 16
        assert transport["metrics"]["ring_wait_seconds"] is None

    def test_invalid_transfer_engine_key_reports_engine_and_key_shape(self):
        engine = self._make_engine(
            num_gpus_per_engine=1,
            sglang_tp_size=1,
            mooncake_data_plane="transfer_engine",
        )

        with pytest.raises(
            ValueError,
            match=(
                r"invalid TransferEngine key: engine_rank=0, "
                r"key_type=list, key=\['ordinary-store-key'\]"
            ),
        ):
            engine._observe_transfer_descriptors([["ordinary-store-key"]])

    def test_terminal_feature_status_comes_from_scheduler_owned_source_ring(self):
        contract = build_target_feature_contract(
            drafter_family="dspark",
            target_hidden_size=4096,
            target_num_hidden_layers=36,
            aux_hidden_state_layers=[0, 1, 2],
            store_last_hidden_states=True,
        )
        engine = self._make_engine(
            num_gpus_per_engine=1,
            sglang_tp_size=1,
            aspec_feature_contract=contract,
            mooncake_data_plane="transfer_engine",
            mooncake_get_retry_max_wait_seconds=12,
        )
        self._init_engine(engine)
        transport = {
            "state": "active",
            "role": "source",
            "metrics": {"source_descriptors": 1, "source_releases": 1},
        }
        flush = AsyncMock(
            return_value=Namespace(success=True, message='{"state":"active",'
            '"role":"source","metrics":{"source_descriptors":1,'
            '"source_releases":1}}')
        )
        engine._engine = Namespace(
            tokenizer_manager=Namespace(flush_cache=flush)
        )

        status = asyncio.run(engine.prepare_feature_contract_status())

        assert status["transport"] == transport
        flush.assert_awaited_once_with(timeout_s=17.0)

    def test_terminal_feature_status_rejects_scheduler_drain_failure(self):
        contract = build_target_feature_contract(
            drafter_family="dspark",
            target_hidden_size=4096,
            target_num_hidden_layers=36,
            aux_hidden_state_layers=[0, 1, 2],
            store_last_hidden_states=True,
        )
        engine = self._make_engine(
            num_gpus_per_engine=1,
            sglang_tp_size=1,
            aspec_feature_contract=contract,
            mooncake_data_plane="transfer_engine",
        )
        self._init_engine(engine)
        engine._engine = Namespace(
            tokenizer_manager=Namespace(
                flush_cache=AsyncMock(
                    return_value=Namespace(success=False, message="source ring busy")
                )
            )
        )

        with pytest.raises(RuntimeError, match="source ring busy"):
            asyncio.run(engine.prepare_feature_contract_status())

    def test_shutdown_drains_target_prefill_before_closing_store(self):
        engine = self._make_engine(
            num_gpus_per_engine=1,
            sglang_tp_size=1,
            target_prefill_memory_optimized=True,
        )
        calls = MagicMock()
        runtime = MagicMock()

        async def drain(**_):
            calls.source_drain()
            return Namespace(success=True, message="")

        runtime.tokenizer_manager.flush_cache = AsyncMock(side_effect=drain)
        runtime.async_shutdown = AsyncMock(side_effect=lambda: calls.engine_shutdown())
        store = MagicMock()
        store.close.side_effect = lambda: calls.store_close()
        engine._engine = runtime
        engine._mooncake_store = store

        asyncio.run(engine.shutdown())

        assert calls.mock_calls == [
            call.source_drain(),
            call.engine_shutdown(),
            call.store_close(),
        ]
        assert engine._engine is None
        assert engine._mooncake_store is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
