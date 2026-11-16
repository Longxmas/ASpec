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

import json

import pytest

from tools.validation.recipe_capacity import audit_long_run, summarize_memory


@pytest.mark.parametrize("fault", ["remote", "busy", "incomplete"])
def test_capacity_rejects_unobservable_or_unavailable_target_allocation(monkeypatch, fault):
    from omegaconf import OmegaConf

    from aspec.commands._recipe_capacity import memory_gpu_ids

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6,7")
    monkeypatch.setattr("aspec.utils.misc.get_current_node_ip", lambda: "local")
    info = {"node_ip": "local", "devices": list(range(6)), "engines": {}, "draining": False}
    if fault == "remote":
        info["node_ip"] = "another-host"
    elif fault == "busy":
        info["engines"] = {"another-session": {"state": "ready"}}
    else:
        info["devices"] = [0, 1]
    monkeypatch.setattr("aspec.inference.service_protocol.request", lambda *_a, **_k: info)
    with pytest.raises(ValueError, match="Draft host|idle Target|allocated Target"):
        memory_gpu_ids(OmegaConf.create({}), {"target_gpus": 6})


@pytest.mark.parametrize("backend", ["fa4", "cudnn_flex_attention"])
@pytest.mark.parametrize("broken", [None, "short", "empty", "cp1", "padding", "backend"])
def test_long_capacity_requires_observed_full_context(tmp_path, broken, backend):
    (tmp_path / "config.yaml").write_text(f"training:\n  attention_backend: {backend}\n")
    training = {
        "scheduler": {
            "observed_sequence_min": 261888,
            "observed_packed_row_min": 261888,
            "observed_packed_row_max": 261888,
            "rows_dispatched": 4,
            "packed_tokens": 4 * 261888,
        },
        "execution": {"context_parallel_size": 2, "attention_backend": backend},
    }
    if broken == "short":
        training["scheduler"]["observed_sequence_min"] = 128
    elif broken == "empty":
        training["scheduler"]["rows_dispatched"] = 0
    elif broken == "cp1":
        training["execution"]["context_parallel_size"] = 1
    elif broken == "padding":
        training["scheduler"]["packed_tokens"] -= 1
    elif broken == "backend":
        training["execution"]["attention_backend"] = (
            "fa4" if backend == "cudnn_flex_attention" else "cudnn_flex_attention"
        )
    (tmp_path / "evidence.json").write_text(json.dumps({"observations": {"training": training}}))
    if broken:
        with pytest.raises(ValueError, match="actual"):
            audit_long_run(tmp_path)
    else:
        result = audit_long_run(tmp_path)
        assert result["tokens_per_row"] == 261888
        assert result["attention_backend"] == backend


def test_long_capacity_rejects_unsupported_configured_backend(tmp_path):
    (tmp_path / "config.yaml").write_text("training:\n  attention_backend: math\n")
    (tmp_path / "evidence.json").write_text("{}")
    with pytest.raises(ValueError, match="configured attention backend"):
        audit_long_run(tmp_path)


def test_sampled_memory_requires_all_devices_and_repeated_samples(tmp_path):
    p = tmp_path / "gpu.csv"
    header = "index, memory.used [MiB], memory.total [MiB]\n"
    p.write_text(header + "0, 10 MiB, 100 MiB\n")
    with pytest.raises(ValueError, match="Missing"):
        summarize_memory(p, expected_gpus=1)
    p.write_text(p.read_text() + "0, 20 MiB, 100 MiB\n")
    assert summarize_memory(p, expected_gpus=1)["devices"]["0"]["peak_used_mib"] == 20
    with pytest.raises(ValueError, match="Missing"):
        summarize_memory(p, expected_gpus=2)
