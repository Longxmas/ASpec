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
# tests/test_training_config_validation.py. Namespace bindings and ASpec regression coverage adapted.

"""Regression tests for training-config validation at config load.

Guards the empty-dispatch NCCL-hang root cause behind the issue #126 surface:
a ``training.micro_batch_size`` of 0 makes the derived ``dispatch_batch_size`` 0,
so ``try_dispatch_batch`` no-op-dispatches (returns True while queuing nothing)
and every rank blocks on the data-fetcher queue, surfacing as an NCCL
all-gather timeout. The validator rejects non-positive sizes at config load,
before any training process starts.

Extends the same fail-closed-at-load idiom (PR #171) to four more numeric
fields whose non-positive values currently fail *silently* (flat loss from
``learning_rate=0`` / ``max_grad_norm=0``, sign-flipped gradients from
``max_grad_norm<0``, inference-pool starvation from ``inference_batch_size=0``)
or crash *opaquely and late* (post-init ``ZeroDivisionError`` / invalid
``total_steps`` from ``draft_accumulation_steps<=0``). Rejecting at
load surfaces the misconfig before Ray/mooncake/vLLM init.
"""

import pytest
from omegaconf import OmegaConf

from aspec.config.train_config import Config, load_config


def _resolved_config(micro_batch_size: int):
    """Build a fully-resolved config from the schema defaults + a batch-size override."""
    config = OmegaConf.structured(Config)
    config.training.micro_batch_size = micro_batch_size
    return config


def test_load_config_rejects_zero_micro_batch_size():
    """micro_batch_size=0 must fail at load: empty dispatch -> NCCL hang."""
    base = _resolved_config(micro_batch_size=0)
    with pytest.raises(ValueError, match="micro_batch_size"):
        load_config(base_config=base)


def test_load_config_rejects_negative_micro_batch_size():
    """Negative sizes are rejected for the same reason."""
    base = _resolved_config(micro_batch_size=-4)
    with pytest.raises(ValueError, match="micro_batch_size"):
        load_config(base_config=base)


def test_load_config_accepts_positive_micro_batch_size():
    """Positive control: a valid size loads without raising."""
    base = _resolved_config(micro_batch_size=2)
    config = load_config(base_config=base)
    assert config.training.micro_batch_size == 2


def test_single_node_config_rejects_cross_node_target_tp():
    config = _resolved_config(micro_batch_size=2)
    config.inference.sglang.nnodes = 2
    with pytest.raises(ValueError, match="node-local"):
        load_config(base_config=config)


def _static_multi_node_config(protocol="rdma"):
    config = OmegaConf.structured(Config)
    config.cluster.mode = "static_multi_node"
    config.cluster.shared_root = "/example"
    config.output_dir = "/example/aspec-runs/test"
    config.model.target_model_path = "/example/models/Qwen3-8B"
    config.model.draft_model_config = "/example/ASpec/aspec/config/dspark_draft_config.json"
    config.dataset.train_data_path = "/example/data/train.jsonl"
    config.training.placement_strategy = "custom"
    config.training.training_node_ips = ["10.0.0.1"]
    config.inference.inference_node_ips = ["10.0.0.2"]
    config.inference.inference_num_gpus = 4
    config.inference.inference_num_gpus_per_node = 4
    config.inference.inference_num_gpus_per_engine = 1
    config.inference.inference_batch_size = 8
    config.inference.sglang.tp_size = 1
    config.training.streaming_dispatch_with_training = True
    config.mooncake = {
        "data_plane": "transfer_engine",
        "protocol": protocol,
        "device_name": "mlx5_bond_0" if protocol == "rdma" else "",
        "rdma_port": 1,
        "rdma_gid_index": 3 if protocol == "rdma" else None,
        "source_gpu_ring_slots": 8,
        "destination_gpu_ring_slots": 2,
        "enable_gpu_direct": True,
        "allow_host_fallback": False,
        "require_nvlink": False,
    }
    return config


def test_static_multi_node_config_requires_strict_explicit_contract():
    config = load_config(base_config=_static_multi_node_config())

    assert config.cluster.mode == "static_multi_node"
    assert config.training.placement_strategy == "custom"
    assert config.mooncake.protocol == "rdma"


def test_static_multi_node_config_accepts_generation_checked_tcp():
    config = load_config(base_config=_static_multi_node_config(protocol="tcp"))

    assert config.mooncake.protocol == "tcp"
    assert config.mooncake.device_name == ""
    assert config.mooncake.data_plane == "transfer_engine"


def _node_local_nvlink_config():
    config = _static_multi_node_config(protocol="nvlink_intra")
    config.training.training_num_nodes = 2
    config.training.training_num_gpus_per_node = 4
    config.training.training_node_ips = ["10.0.0.1", "10.0.0.2"]
    config.inference.inference_num_gpus = 8
    config.inference.inference_num_gpus_per_node = 4
    config.inference.inference_node_ips = ["10.0.0.1", "10.0.0.2"]
    config.mooncake.update(
        {
            "feature_affinity": "node_local_required",
            "require_nvlink": True,
        }
    )
    return config


def test_static_multi_node_accepts_strict_node_local_nvlink():
    loaded = load_config(base_config=_node_local_nvlink_config())

    assert loaded.mooncake.feature_affinity == "node_local_required"
    assert loaded.mooncake.protocol == "nvlink_intra"
    assert loaded.training.training_num_nodes == 2


def test_node_local_nvlink_requires_same_role_nodes():
    config = _node_local_nvlink_config()
    config.inference.inference_node_ips = ["10.0.0.1", "10.0.0.3"]

    with pytest.raises(ValueError, match="same nodes"):
        load_config(base_config=config)


def test_node_local_nvlink_keeps_each_cp_group_inside_one_node():
    config = _node_local_nvlink_config()
    config.training.draft_context_parallel_size = 3

    with pytest.raises(ValueError, match="CP group"):
        load_config(base_config=config)


def test_node_local_nvlink_requires_one_ready_credit_per_node():
    config = _node_local_nvlink_config()
    config.inference.max_sample_pool_size = 1

    with pytest.raises(ValueError, match="one credit per locality domain"):
        load_config(base_config=config)


def test_static_multi_node_rejects_partial_target_tp_group_on_last_node():
    config = _node_local_nvlink_config()
    config.inference.inference_num_gpus = 6
    config.inference.inference_num_gpus_per_node = 4
    config.inference.inference_num_gpus_per_engine = 4
    config.inference.sglang.tp_size = 4

    with pytest.raises(ValueError, match="complete TP groups"):
        load_config(base_config=config)


def test_node_local_nvlink_requires_balanced_finite_target_batches():
    config = _node_local_nvlink_config()
    config.training.num_train_steps = 1
    config.training.draft_accumulation_steps = 1
    config.training.micro_batch_size = 1
    config.inference.inference_batch_size = 3

    with pytest.raises(ValueError, match="complete Target batches on every node"):
        load_config(base_config=config)


def test_static_multi_node_config_accepts_explicit_host_staged_tcp():
    config = _static_multi_node_config(protocol="tcp")
    config.mooncake.update(
        {
            "data_plane": "store",
            "enable_gpu_direct": False,
            "allow_host_fallback": False,
        }
    )

    loaded = load_config(base_config=config)

    assert loaded.mooncake.data_plane == "store"
    assert loaded.mooncake.protocol == "tcp"
    assert loaded.mooncake.enable_gpu_direct is False
    assert loaded.mooncake.allow_host_fallback is False


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda config: config.mooncake.update({"protocol": "rdma"}), "host-staged"),
        (
            lambda config: config.mooncake.update({"enable_gpu_direct": True}),
            "host-staged",
        ),
        (
            lambda config: config.mooncake.update({"allow_host_fallback": True}),
            "explicit.*fallback",
        ),
        (
            lambda config: config.mooncake.update({"async_put_pool_size": 0}),
            "async_put_pool_size",
        ),
    ],
)
def test_static_multi_node_host_staged_tcp_fails_closed(mutation, match):
    config = _static_multi_node_config(protocol="tcp")
    config.mooncake.update(
        {
            "data_plane": "store",
            "enable_gpu_direct": False,
            "allow_host_fallback": False,
        }
    )
    mutation(config)

    with pytest.raises(ValueError, match=match):
        load_config(base_config=config)


@pytest.mark.parametrize("protocol", ["tcp", "rdma"])
def test_static_multi_node_config_accepts_direct_target_output(protocol):
    config = _static_multi_node_config(protocol=protocol)
    config.inference.target_prefill_memory_optimized = True
    config.inference.inference_batch_size = 1
    config.training.micro_batch_size = 1
    config.training.training_num_gpus_per_node = 4

    loaded = load_config(base_config=config)

    assert loaded.inference.target_prefill_memory_optimized is True
    assert loaded.mooncake.protocol == protocol
    assert loaded.mooncake.require_nvlink is False


def test_static_multi_node_config_accepts_per_node_single_rail_rdma_routes():
    config = _static_multi_node_config()
    config.mooncake.device_name = ""
    config.mooncake.rdma_gid_index = None
    config.mooncake.rdma_routes = {
        "10.0.0.1": {"device_name": "mlx5_0", "port": 1, "gid_index": 3},
        "10.0.0.2": {"device_name": "mlx5_7", "port": 1, "gid_index": 5},
    }

    loaded = load_config(base_config=config)

    assert loaded.mooncake.rdma_routes["10.0.0.2"].gid_index == 5


def test_transfer_engine_rejects_source_ring_smaller_than_target_batch():
    config = _static_multi_node_config()
    config.mooncake.source_gpu_ring_slots = 2

    with pytest.raises(ValueError, match="source GPU ring is too shallow"):
        load_config(base_config=config)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda config: setattr(config.training, "placement_strategy", "training_first"), "custom"),
        (lambda config: setattr(config.inference, "inference_node_ips", ["10.0.0.1"]), "distinct"),
        (lambda config: setattr(config.inference.sglang, "nnodes", 2), "node-local"),
        (lambda config: config.mooncake.update({"protocol": "nvlink_intra"}), "protocol"),
        (lambda config: config.mooncake.update({"device_name": ""}), "device_name"),
        (lambda config: config.mooncake.update({"rdma_gid_index": None}), "gid_index"),
        (lambda config: config.mooncake.update({"device_name": "mlx5_0,mlx5_1"}), "exactly one device"),
    ],
)
def test_static_multi_node_config_fails_closed(mutation, match):
    config = _static_multi_node_config()
    mutation(config)

    with pytest.raises(ValueError, match=match):
        load_config(base_config=config)


def test_validate_training_batch_config_raises_on_zero():
    """Directly exercise the module-level validator, not just the load_config wiring."""
    from aspec.config.train_config import _validate_training_batch_config

    config = _resolved_config(micro_batch_size=0)
    with pytest.raises(ValueError, match="micro_batch_size"):
        _validate_training_batch_config(config)


def test_validate_training_batch_config_accepts_positive():
    """Positive control for the direct validator call."""
    from aspec.config.train_config import _validate_training_batch_config

    config = _resolved_config(micro_batch_size=8)
    _validate_training_batch_config(config)  # must not raise


def test_load_config_rejects_backend_outside_current_scope():
    base = OmegaConf.structured(Config)
    base.inference.inference_engine_type = "vllm"
    base.model.target_model_backend = "vllm"

    with pytest.raises(NotImplementedError, match="currently supports"):
        load_config(base_config=base)


def test_load_config_accepts_r4_packing_contract():
    base = _resolved_training_config(
        dflash_packing=True,
        micro_batch_size=1,
        packing_min_samples_per_dispatch=8,
    )

    config = load_config(base_config=base)

    assert config.training.dflash_packing is True
    assert config.training.packing_min_samples_per_dispatch == 8


def test_load_config_rejects_packing_with_multi_sample_micro_batch():
    base = _resolved_training_config(dflash_packing=True, micro_batch_size=2)

    with pytest.raises(ValueError, match="micro_batch_size=1"):
        load_config(base_config=base)


def test_load_config_rejects_packing_with_usp():
    base = _resolved_training_config(
        dflash_packing=True,
        micro_batch_size=1,
        attention_backend="usp",
    )

    with pytest.raises(ValueError, match="not supported with USP"):
        load_config(base_config=base)


def test_load_config_rejects_negative_packing_supply_threshold():
    base = _resolved_training_config(
        dflash_packing=True,
        micro_batch_size=1,
        packing_min_samples_per_dispatch=-1,
    )

    with pytest.raises(ValueError, match="packing_min_samples_per_dispatch"):
        load_config(base_config=base)


def test_load_config_rejects_nonpositive_source_work_budget():
    base = _resolved_training_config(source_work_budget=0)

    with pytest.raises(ValueError, match="source_work_budget"):
        load_config(base_config=base)


def test_final_checkpoint_can_be_disabled_explicitly():
    base = _resolved_training_config(save_final_checkpoint=False)

    config = load_config(base_config=base)

    assert config.training.save_final_checkpoint is False


def test_required_performance_window_must_leave_a_measured_step():
    base = _resolved_training_config(num_train_steps=3)
    base.debug.performance_measurement_required = True
    base.debug.performance_warmup_steps = 2
    base.debug.performance_drain_steps = 1

    with pytest.raises(ValueError, match="required performance measurement"):
        load_config(base_config=base)


def test_required_performance_window_accepts_bounded_measurement():
    base = _resolved_training_config(num_train_steps=4)
    base.debug.performance_measurement_required = True
    base.debug.performance_warmup_steps = 2
    base.debug.performance_drain_steps = 1

    config = load_config(base_config=base)

    assert config.debug.performance_warmup_steps == 2
    assert config.debug.performance_drain_steps == 1


def test_producer_profile_requires_a_strictly_smaller_warmup():
    base = _resolved_training_config()
    base.debug.topology_producer_profile_samples = 16
    base.debug.topology_producer_profile_warmup_samples = 16

    with pytest.raises(ValueError, match="warmup_samples must be smaller"):
        load_config(base_config=base)


# --- Numeric training-config fields (draft_accumulation_steps, learning_rate,
#     max_grad_norm) and inference_batch_size — PR #171 idiom extended. ---


def _resolved_training_config(**overrides):
    """Build a fully-resolved config from the schema defaults + training overrides."""
    config = OmegaConf.structured(Config)
    for key, value in overrides.items():
        setattr(config.training, key, value)
    return config


def _resolved_inference_config(inference_batch_size: int):
    """Build a fully-resolved config from the schema defaults + an inference-batch override."""
    config = OmegaConf.structured(Config)
    config.inference.inference_batch_size = inference_batch_size
    return config


def test_context_parallel_requires_positive_size_and_direct_cp_backend():
    with pytest.raises(NotImplementedError, match="requires attention_backend=fa4 or"):
        load_config(
            base_config=_resolved_training_config(
                attention_backend="flex_attention",
                draft_context_parallel_size=2,
            )
        )

    with pytest.raises(ValueError, match="positive integer"):
        load_config(
            base_config=_resolved_training_config(
                attention_backend="fa4",
                draft_context_parallel_size=0,
            )
        )


def test_transfer_engine_requires_live_streaming_consumer():
    base = _resolved_training_config(streaming_dispatch_with_training=False)
    base.mooncake = {"data_plane": "transfer_engine"}

    with pytest.raises(ValueError, match="streaming_dispatch_with_training"):
        load_config(base_config=base)


def test_fa4_context_parallel_and_streaming_load_together():
    base = _resolved_training_config(
        attention_backend="fa4",
        draft_context_parallel_size=2,
        streaming_dispatch_with_training=True,
    )
    base.mooncake = {"data_plane": "transfer_engine"}

    config = load_config(base_config=base)

    assert config.training.draft_context_parallel_size == 2


def test_cudnn_context_parallel_and_streaming_load_together():
    base = _resolved_training_config(
        attention_backend="cudnn_flex_attention",
        draft_context_parallel_size=2,
        streaming_dispatch_with_training=True,
    )
    base.mooncake = {"data_plane": "transfer_engine"}

    config = load_config(base_config=base)

    assert config.training.draft_context_parallel_size == 2


def test_fa4_context_parallel_rejects_late_ring_and_anchor_failures():
    base = _resolved_training_config(
        attention_backend="fa4",
        draft_context_parallel_size=2,
        streaming_dispatch_with_training=True,
        micro_batch_size=2,
    )
    base.mooncake = {
        "data_plane": "transfer_engine",
        "gpu_ring_slots": 2,
        "source_gpu_ring_slots": 1,
        "destination_gpu_ring_slots": 2,
    }
    with pytest.raises(ValueError, match="source GPU ring"):
        load_config(base_config=base)

    base.mooncake.source_gpu_ring_slots = 2
    base.mooncake.destination_gpu_ring_slots = 1
    with pytest.raises(ValueError, match="destination GPU ring"):
        load_config(base_config=base)

    base.mooncake.destination_gpu_ring_slots = 2
    base.training.dspark_num_anchors = 511
    with pytest.raises(ValueError, match="dspark_num_anchors"):
        load_config(base_config=base)


@pytest.mark.parametrize("backend", ["fa4", "cudnn_flex_attention"])
@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_direct_context_parallel_accepts_packed_rows(backend, cp_size):
    base = _resolved_training_config(
        attention_backend=backend,
        draft_context_parallel_size=cp_size,
        streaming_dispatch_with_training=True,
        dflash_packing=True,
        micro_batch_size=1,
    )
    base.mooncake = {"data_plane": "transfer_engine", "gpu_ring_slots": max(2, cp_size)}

    config = load_config(base_config=base)
    assert config.training.draft_context_parallel_size == cp_size
    assert config.training.dflash_packing is True


def test_direct_target_output_requires_strict_transport_and_source_topology():
    base = _resolved_training_config(
        streaming_dispatch_with_training=True,
        micro_batch_size=1,
    )
    base.inference.target_prefill_memory_optimized = True
    base.inference.inference_num_gpus = 1
    base.inference.inference_batch_size = 1
    base.inference.sglang.tp_size = 1

    with pytest.raises(ValueError, match="data_plane=transfer_engine"):
        load_config(base_config=base)

    base.mooncake = {
        "data_plane": "transfer_engine",
        "protocol": "tcp",
        "enable_gpu_direct": True,
        "allow_host_fallback": False,
        "require_nvlink": False,
    }
    with pytest.raises(ValueError, match="single-node.*nvlink_intra"):
        load_config(base_config=base)

    base.mooncake = {
        "data_plane": "transfer_engine",
        "protocol": "nvlink_intra",
        "enable_gpu_direct": True,
        "allow_host_fallback": False,
        "require_nvlink": True,
    }
    config = load_config(base_config=base)
    assert config.inference.target_prefill_memory_optimized is True

    # One Target can feed two Draft ranks when its generation-checked source
    # ring can retain the complete dispatch; source diversity is a preference,
    # not an ownership requirement.
    base.training.training_num_gpus_per_node = 2
    base.mooncake.source_gpu_ring_slots = 2
    config = load_config(base_config=base)
    assert config.training.training_num_gpus_per_node == 2

    base.mooncake.source_gpu_ring_slots = 1
    with pytest.raises(ValueError, match="cannot retain one complete Draft dispatch"):
        load_config(base_config=base)


def test_load_config_rejects_zero_draft_accumulation_steps():
    """draft_accumulation_steps=0 must fail at load: propagates to global_batch_size=0
    and crashes post-init (ZeroDivisionError or invalid total_steps)."""
    base = _resolved_training_config(draft_accumulation_steps=0)
    with pytest.raises(ValueError, match="draft_accumulation_steps"):
        load_config(base_config=base)


def test_load_config_rejects_negative_draft_accumulation_steps():
    """Negative values propagate the same way and must be rejected at load."""
    base = _resolved_training_config(draft_accumulation_steps=-2)
    with pytest.raises(ValueError, match="draft_accumulation_steps"):
        load_config(base_config=base)


def test_load_config_rejects_zero_learning_rate():
    """learning_rate=0 yields silent flat loss (AdamW+scheduler at lr=0) and an
    untrained checkpoint; reject at load."""
    base = _resolved_training_config(learning_rate=0)
    with pytest.raises(ValueError, match="learning_rate"):
        load_config(base_config=base)


def test_load_config_rejects_negative_learning_rate():
    """Negative learning rates hit late scheduler validation; reject at load."""
    base = _resolved_training_config(learning_rate=-1e-4)
    with pytest.raises(ValueError, match="learning_rate"):
        load_config(base_config=base)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"min_lr": -1e-5}, "min_lr"),
        ({"learning_rate": 1e-4, "min_lr": 2e-4}, "min_lr"),
        ({"warmup_ratio": -0.1}, "warmup_ratio"),
        ({"warmup_ratio": 1.0}, "warmup_ratio"),
        ({"num_train_steps": 0}, "num_train_steps"),
        ({"lr_total_steps": -1}, "lr_total_steps"),
    ],
)
def test_load_config_rejects_invalid_scheduler_inputs(overrides, match):
    base = _resolved_training_config(**overrides)
    with pytest.raises(ValueError, match=match):
        load_config(base_config=base)


def test_load_config_rejects_zero_max_grad_norm():
    """max_grad_norm=0 zeroes all grads via clip_grad_norm_ (silent flat loss);
    reject at load."""
    base = _resolved_training_config(max_grad_norm=0)
    with pytest.raises(ValueError, match="max_grad_norm"):
        load_config(base_config=base)


def test_load_config_rejects_negative_max_grad_norm():
    """Negative max_grad_norm sign-flips grads (silent gradient ascent); reject at load."""
    base = _resolved_training_config(max_grad_norm=-0.5)
    with pytest.raises(ValueError, match="max_grad_norm"):
        load_config(base_config=base)


def test_load_config_rejects_non_positive_optimizer_step_timeout():
    base = _resolved_training_config(optimizer_step_timeout_seconds=0)
    with pytest.raises(ValueError, match="optimizer_step_timeout_seconds"):
        load_config(base_config=base)


def test_load_config_rejects_zero_inference_batch_size():
    """inference_batch_size=0 starves the inference pool (silent dispatch spin) and
    vLLM rejects max_num_seqs=0 late after init; reject at load."""
    base = _resolved_inference_config(inference_batch_size=0)
    with pytest.raises(ValueError, match="inference_batch_size"):
        load_config(base_config=base)


def test_load_config_rejects_negative_inference_batch_size():
    """Negative inference batch sizes starve the pool the same way; reject at load."""
    base = _resolved_inference_config(inference_batch_size=-1)
    with pytest.raises(ValueError, match="inference_batch_size"):
        load_config(base_config=base)

@pytest.mark.parametrize("cp_size", [3, 8, 16])
def test_fa4_cp_degree_has_no_size_whitelist(cp_size):
    from aspec.config.draft_execution import resolve_draft_execution_capability
    capability = resolve_draft_execution_capability("fa4", cp_size)
    assert capability.context_parallel_size == cp_size
    assert capability.feature_layout == "context_sharded"


def test_fa4_cp3_packing_loads_with_real_world_and_anchor_divisibility():
    config = _resolved_training_config(
        attention_backend="fa4", draft_context_parallel_size=3,
        training_num_gpus_per_node=6, dspark_num_anchors=510,
        streaming_dispatch_with_training=True, dflash_packing=True,
        micro_batch_size=1,
    )
    loaded = load_config(base_config=config)
    assert loaded.training.draft_context_parallel_size == 3

@pytest.mark.parametrize("overrides,style,ratio", [([], "cosine", 0.2),
    (["training.lr_wsd_decay_style=linear", "training.lr_wsd_decay_ratio=0.4"], "linear", 0.4),
    (["training.wsd_decay_style=exponential", "training.wsd_decay_ratio=0.6"], "exponential", 0.6)])
def test_wsd_aliases_resolve_to_a_working_decay_curve(overrides, style, ratio):
    import torch

    from aspec.config.train_config import config_to_flat_args
    from aspec.training.lr_scheduler import LRSchedulerWithWarmup
    args = config_to_flat_args(load_config(cli_args=["training.lr_decay_style=WSD", *overrides], save_snapshot=False))
    assert args.wsd_decay_style == style
    assert args.wsd_decay_ratio == ratio
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=1.0)
    scheduler = LRSchedulerWithWarmup(optimizer, max_lr=1.0, total_steps=5, decay_style="WSD",
                                     wsd_decay_steps=max(1, int(5*ratio)), wsd_decay_style=args.wsd_decay_style)
    rates = []
    for _ in range(5):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    assert rates[-1] < rates[0]


@pytest.mark.parametrize("override", ["training.wsd_decay_style=unknown", "training.lr_wsd_decay_style=unknown",
                                      "training.wsd_decay_ratio=0", "training.lr_wsd_decay_ratio=1.1"])
def test_wsd_invalid_parameters_fail_during_config_load(override):
    with pytest.raises(ValueError, match="wsd_decay"):
        load_config(cli_args=["training.lr_decay_style=WSD", override], save_snapshot=False)

@pytest.mark.parametrize("field", ["train_data_path", "eval_data_path"])
def test_cli_local_data_preflight_checks_missing_files_without_fetching(tmp_path, field):
    from aspec.config.train_config import validate_local_data_paths
    config = OmegaConf.structured(Config)
    path = tmp_path / "missing.jsonl"
    setattr(config.dataset, field, str(path))
    with pytest.raises(FileNotFoundError, match=field):
        validate_local_data_paths(config)
    path.write_text("{}\n")
    validate_local_data_paths(config)
    setattr(config.dataset, field, "organization/hub-dataset")
    validate_local_data_paths(config)


def test_wsd_snapshot_with_omitted_aliases_can_be_reloaded(tmp_path):
    from aspec.config.train_config import config_to_flat_args

    original = load_config(cli_args=["training.lr_decay_style=WSD"], save_snapshot=False)
    path = tmp_path / "config.yaml"
    OmegaConf.save(original, path)
    restored = load_config(str(path), save_snapshot=False)
    args = config_to_flat_args(restored)
    assert args.wsd_decay_ratio == 0.2
    assert args.wsd_decay_style == "cosine"


@pytest.mark.parametrize("field", [
    "training.learning_rate", "training.min_lr", "training.max_grad_norm",
    "training.optimizer_step_timeout_seconds", "cluster.preflight_timeout_seconds",
    "cluster.file_transfer_timeout_seconds",
])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_config_rejected(field, value):
    config = OmegaConf.structured(Config)
    OmegaConf.update(config, field, value)
    with pytest.raises(ValueError, match=field.split(".")[-1]):
        load_config(base_config=config, save_snapshot=False)
