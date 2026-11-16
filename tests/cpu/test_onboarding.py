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
import os
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from aspec.config.train_config import load_config
from aspec.planning import compile_execution_plan
from tools.quickstart import MODELS, build_profile

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", MODELS)
def test_quickstart_profiles_preserve_two_gpu_workload_and_model_binding(name, tmp_path):
    profile = build_profile(name)
    path = tmp_path / "profile.yaml"
    OmegaConf.save(profile, path)
    config = load_config(str(path), save_snapshot=False)
    plan = compile_execution_plan(config)
    args = plan.to_runtime_args()
    assert config.training.num_train_steps == config.training.lr_total_steps == 8
    assert config.training.training_num_gpus_per_node == config.inference.inference_num_gpus == 1
    assert config.training.fsdp_strategy == "REPLICATE"
    assert Path(config.dataset.train_data_path).is_file()
    assert not config.dataset.eval_data_path
    assert args.max_seq_length == 1024
    if name == "ling-tiny":
        assert config.dataset.chat_template == "ling3"
        assert config.model.embedding_key == "model.word_embeddings.weight"
        assert config.mooncake.hidden_dim == 1536
        assert not config.inference.target_prefill_memory_optimized
    else:
        assert config.dataset.chat_template == "qwen3-instruct"
        assert config.inference.target_prefill_memory_optimized


def test_package_import_does_not_initialize_the_gpu_stack():
    result = subprocess.run([sys.executable, "-c",
                             "import sys,aspec; assert 'torch' not in sys.modules; "
                             "assert 'DSparkModel' in dir(aspec); "
                             "assert set(aspec.__all__)"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("command", ["doctor", "quickstart", "recipes", "patch-sglang"])
def test_cli_shortcuts_execute_in_the_selected_venv(command):
    env = dict(os.environ, VIRTUAL_ENV=sys.prefix)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    result = subprocess.run([sys.executable, "-m", "aspec.cli", command, "--help"],
                            env=env, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_setup_owns_only_its_new_isolated_venv_and_preserves_failed_install(tmp_path):
    root = tmp_path / "source with spaces"
    (root / "tools").mkdir(parents=True)
    (root / "setup.sh").write_bytes((ROOT / "setup.sh").read_bytes())
    installer = root / "tools/install.sh"
    installer.write_text(
        "#!/usr/bin/env bash\n"
        "python -c 'import json,sys,os;print(json.dumps({"
        "\"isolated\":sys.prefix!=sys.base_prefix,\"prefix\":sys.prefix,"
        "\"active\":os.environ[\"VIRTUAL_ENV\"]}))'\n"
        "printf '%s\\n' \"$@\"\nexit 7\n")
    venv = tmp_path / "owned environment"
    command = ["bash", str(root / "setup.sh"), "--python", sys.executable,
               "--venv", str(venv), "--dev"]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 7, result.stderr
    status = json.loads(result.stdout.splitlines()[0])
    assert status == {"isolated": True, "prefix": str(venv), "active": str(venv)}
    assert "--dev" in result.stdout and "--with-qk" not in result.stdout
    assert (venv / ".aspec-source").read_text().strip() == str(root)
    installer.write_text("#!/usr/bin/env bash\nexit 0\n")
    assert subprocess.run(command, env=env, capture_output=True).returncode == 0
    unrelated = tmp_path / "unowned"
    unrelated.mkdir()
    before = list(unrelated.iterdir())
    refused = subprocess.run(["bash", str(root / "setup.sh"), "--venv", str(unrelated),
                              "--python", sys.executable], env=env, capture_output=True)
    assert refused.returncode != 0
    assert list(unrelated.iterdir()) == before


def test_container_reuses_source_setup_and_excludes_local_artifacts():
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "bash setup.sh --venv /opt/aspec-venv" in dockerfile
    assert "libibverbs1" in dockerfile and "libmlx5-1" in dockerfile
    assert "--gpu" not in dockerfile
    excluded = (ROOT / ".dockerignore").read_text().splitlines()
    assert {".git", ".venv", "models", "outputs", "cache"} <= set(excluded)


def test_ray_queue_payload_import_works_in_a_fresh_worker():
    result = subprocess.run([sys.executable, "-c",
                             "import aspec.training.data_fetcher; "
                             "from aspec.config import Config; "
                             "from aspec import DSparkModel; assert Config and DSparkModel"],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("managed,expected_gpus", [(False, 1), (True, 2)])
def test_quickstart_rejects_unallocated_gpus_without_a_traceback(model, managed, expected_gpus, tmp_path, monkeypatch, capsys):
    from tools import quickstart

    output = tmp_path / "run"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    command = ["quickstart", model, "--output", str(output)]
    if managed:
        command += ["--managed-target", "/target/bin/python"]
    monkeypatch.setattr(sys, "argv", command)
    with pytest.raises(SystemExit) as error:
        quickstart.main()
    assert error.value.code == 1
    message = capsys.readouterr().err
    assert f"Select exactly {expected_gpus} " in message
    assert "Traceback" not in message
    assert not output.exists()


@pytest.mark.parametrize("fail", [False, True])
def test_quickstart_saves_diagnostics_and_shows_progress(tmp_path, monkeypatch, capsys, fail):
    from tools import quickstart

    output = tmp_path / "run"

    def run(*args, **kwargs):
        output.mkdir()
        kwargs["progress"]("Training the first four steps")
        print("internal command diagnostics")
        if fail:
            raise ValueError("bounded test failure")

    monkeypatch.setattr(quickstart, "run_acceptance", run)
    monkeypatch.setattr(sys, "argv", ["quickstart", "ling-tiny", "--output", str(output)])
    if fail:
        with pytest.raises(SystemExit) as error:
            quickstart.main()
        assert error.value.code == 1
    else:
        quickstart.main()
    captured = capsys.readouterr()
    assert "Training the first four steps" in captured.out
    assert "internal command diagnostics" not in captured.out
    assert "internal command diagnostics" in (output / "quickstart.log").read_text()
    if fail:
        assert "ValueError" in (output / "quickstart.log").read_text()
        assert str(output / "quickstart.log") in captured.err

@pytest.mark.parametrize("race", [False, True])
def test_quickstart_never_changes_an_existing_or_concurrent_output(tmp_path, monkeypatch, race):
    from tools import quickstart

    output = tmp_path / "existing"
    if not race:
        output.mkdir()
        (output / "quickstart.log").write_text("original diagnostics")

    def concurrent(*args, **kwargs):
        output.mkdir()
        (output / "quickstart.log").write_text("original diagnostics")
        raise FileExistsError("another run created this output")

    monkeypatch.setattr(quickstart, "run_acceptance", concurrent)
    monkeypatch.setattr(sys, "argv", ["quickstart", "ling-tiny", "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        quickstart.main()
    assert error.value.code != 0
    assert (output / "quickstart.log").read_text() == "original diagnostics"
    assert list(output.iterdir()) == [output / "quickstart.log"]
