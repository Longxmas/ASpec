# Copyright 2026 Tencent. Adapted for ASpec.
# Licensed under the Apache License, Version 2.0.
# Thread defaults adapted from AngelSpec d3412be.

import os

# Environment variables that should be forwarded to all Ray actors.
# NOTE: TORCHINDUCTOR_CACHE_DIR is intentionally excluded — each node should
# use its own node-local default (/tmp/torchinductor_$USER/) to avoid
# cross-node triton kernel cache corruption over NFS.
_ASPEC_ENV_KEYS = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "CUDA_LAUNCH_BLOCKING",
    "GLOO_SOCKET_IFNAME",
    "HF_HOME",
    "HF_TOKEN",
    # CUDA wheels may need the host-driver path ordered ahead of the toolkit's
    # compatibility directory. Ray runtime_env workers do not reliably retain
    # the driver's process loader path, so forward the resolved value exactly.
    "LD_LIBRARY_PATH",
    # Source snapshots and explicitly selected backend adapters must resolve to
    # the same code in the driver and every Ray actor.
    "PYTHONPATH",
    "MC_LOG_LEVEL",
    "MODELOPT_MAX_TOKENS_PER_EXPERT",
    "NCCL_DEBUG",
    "NCCL_SOCKET_IFNAME",
    "SGLANG_DISABLE_CUDNN_CHECK",
    "SGLANG_VLM_CACHE_SIZE_MB",
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC",
    "TORCHINDUCTOR_FX_GRAPH_CACHE",
    "ASPEC_LOG_DIR",
    "ASPEC_LOG_LEVEL",
    "TP_SOCKET_IFNAME",
    "CUTE_DSL_CACHE_DIR",
    "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED",
    "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR",
    "ASPEC_FLASH_ATTN_OPT_LEVEL",
    "ASPEC_RUN_ID",
    "ASPEC_ATTEMPT_ID",
    "ASPEC_SHARED_SOURCE",
    "ASPEC_SHARED_FEATURE_VIEWS",
]

# Preserve AngelSpec's normal defaults; explicit caller overrides win.
_ASPEC_DEFAULT_ENV_VARS: dict[str, str] = {
    "OMP_NUM_THREADS": "8",
    "MKL_NUM_THREADS": "8",
    "OPENBLAS_NUM_THREADS": "8",
}

# Prevent Ray from overriding VISIBLE_DEVICES so actors manage GPU assignment themselves.
# Reference: https://github.com/ray-project/ray/blob/161849364/python/ray/_private/accelerators/
_RAY_NOSET_VISIBLE_DEVICES_KEYS = [
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
    "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
    "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
    "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
]


def get_aspec_env_vars(args=None) -> dict[str, str]:
    """Return common environment variables for all Ray actors.

    Includes:
    - ASPEC_* variables (e.g. log level) from the current process
    - RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES = "1" to prevent Ray from
      overriding device visibility

    Intended for use with ``ray.remote(runtime_env={"env_vars": ...})``.
    Call-site env vars merged after this dict take higher priority.
    """
    env = {
        **_ASPEC_DEFAULT_ENV_VARS,
        **{k: "1" for k in _RAY_NOSET_VISIBLE_DEVICES_KEYS},
    }
    env.update({k: os.environ[k] for k in _ASPEC_ENV_KEYS if k in os.environ})
    if getattr(args, "cluster_mode", "single_node") == "static_multi_node":
        # Ray is started from each node's own venv/environment. A driver's
        # library/cache paths must not overwrite another node's local layout.
        for key in ("LD_LIBRARY_PATH", "PYTHONPATH", "HF_HOME", "CUTE_DSL_CACHE_DIR",
                    "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR", "ASPEC_LOG_DIR"):
            env.pop(key, None)
    return env
