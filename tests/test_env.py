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
# tests/test_env.py. Namespace bindings and ASpec regression coverage adapted.

"""Tests for the default environment forwarded to ASpec Ray actors."""

from aspec.utils.env import get_aspec_env_vars


def test_aspec_log_level_is_forwarded(monkeypatch):
    monkeypatch.setenv("ASPEC_LOG_LEVEL", "DEBUG")

    env = get_aspec_env_vars()

    assert env["ASPEC_LOG_LEVEL"] == "DEBUG"


def test_vllm_model_runner_setting_is_not_forwarded(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")

    env = get_aspec_env_vars()

    assert "VLLM_USE_V2_MODEL_RUNNER" not in env


def test_run_and_attempt_identity_are_forwarded(monkeypatch):
    monkeypatch.setenv("ASPEC_RUN_ID", "run-1")
    monkeypatch.setenv("ASPEC_ATTEMPT_ID", "attempt-2")

    env = get_aspec_env_vars()

    assert env["ASPEC_RUN_ID"] == "run-1"
    assert env["ASPEC_ATTEMPT_ID"] == "attempt-2"


def test_aspec_actor_env_forwards_dynamic_loader_path(monkeypatch):
    loader_path = "/driver/lib:/cuda/runtime/lib:/cuda/compat"
    monkeypatch.setenv("LD_LIBRARY_PATH", loader_path)

    assert get_aspec_env_vars()["LD_LIBRARY_PATH"] == loader_path


def test_aspec_actor_env_forwards_explicit_pythonpath(monkeypatch):
    python_path = "/benchmark/ASpec:/benchmark/sglang/python"
    monkeypatch.setenv("PYTHONPATH", python_path)

    assert get_aspec_env_vars()["PYTHONPATH"] == python_path


def test_fa4_disk_cache_policy_reaches_actor_processes(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "1")
    monkeypatch.setenv("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR", "/task-local/compiled-fa4")
    env = get_aspec_env_vars()
    assert env["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] == "1"
    assert env["FLASH_ATTENTION_CUTE_DSL_CACHE_DIR"] == "/task-local/compiled-fa4"
    node_local = get_aspec_env_vars(SimpleNamespace(cluster_mode="static_multi_node"))
    assert node_local["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] == "1"
    assert "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR" not in node_local
