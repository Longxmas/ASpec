# Contributing to ASpec

Read the [README](README.md), [architecture](docs/framework/ARCHITECTURE.md), accepted
[decisions](docs/framework/DECISIONS.md), [roadmap](ROADMAP.md), and the relevant
code/tests. [AGENTS.md](AGENTS.md) gives AI collaborators the same invariants; it is not a user CLI tutorial.

## Scope and attribution

- Preserve unrelated work; keep changes small and verifiable.
- Review the pinned TorchSpec path before replacing overlapping code; reuse correct working code by default.
- Classify upstream use as copied/adapted/reimplemented/inspired. Preserve applicable copyright,
  license and notice text, including inherited notices. Identify sources and modifications in file
  headers or adjacent documentation, and distribute the required license texts in `licenses/`.
- Separate training semantics, execution policy and observed facts. Required capabilities fail closed;
  do not hide fallback, lose committed work, or reclaim storage before owners are fenced.
- Update user guides when public behavior changes, not just a development diary.

## Checks

Use a dedicated Linux venv and the [pinned runtime](docs/user-guide/installation.md):

```bash
bash setup.sh --dev
bash setup.sh --target --dev
source .venv/bin/activate
python -m ruff check aspec tests tools setup.py
python -m tools.check_docs
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 bash tools/run_in_env.sh python -m pytest -q --ignore=tests/test_sgl_engine_factory.py --ignore=tests/test_inference_concurrency.py
python -m tools.build_release --outdir /absolute/path/to/new-release-dist
source .venv-target/bin/activate
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 aspec doctor --target
python -m pytest -q tests/test_sgl_engine_factory.py tests/test_inference_concurrency.py tests/test_sgl_engine_decode.py
```

Default tests are fast/offline. GPU, tokenizer, training and cuDNN gates need explicit switches
declared in their test files. A skip is not GPU acceptance. Public CI checks lint, local doc links,
compilation, plus a curated CPU/offline contract suite installed from `requirements/ci-cpu.txt`
on pull requests and configured branch pushes. CI uses pytest's directory discovery: put new
dependency-light repository checks in `tests/source/test_*.py` and offline CPU contracts supported by
the CI runtime in `tests/cpu/test_*.py`. New tests in either directory run automatically without
editing the workflow. Keep tests requiring GPUs, model downloads or the full pinned runtime outside
these directories. Run the same suites locally with:

```bash
python -m pytest -q tests/source
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q tests/cpu
```

The separate [release build](.github/workflows/build-release.yml) workflow builds and audits
sdist/wheel artifacts and checks wheel installation on manual runs and `v*` tag pushes.
Manual runs only validate the artifacts and their installation. In `inclusionAI/ASpec`, tag pushes
publish through the `pypi` environment; see [release instructions](docs/releasing.md).
Neither workflow certifies GPU health or
replaces the complete pinned-runtime suite above. Equivalent remote checks are not proof that hosted
GitHub Actions has already run.

For runtime/attention/data-plane changes, use the relevant [GPU and fault checks](docs/contributing/validation.md).
Do not clean shared hosts with host-wide Ray/process commands. Follow the current task's remote environment
instructions rather than connecting to historical machine examples.

## Verification and issues

Describe changed behavior, checks actually run, environment/revision and unverified scope. Performance
changes require comparable GPU model, workload, statistics and correctness/ownership evidence;
disclose actual configs and effective settings. Loss, throughput, capacity, health and quality are
separate claims. Do not tune semantics/LR or rewrite a failed gate to manufacture agreement.

Include a minimal redacted reproduction, config, Plan/Evidence and doctor output. Do not attach
weights, credentials or private data. Security-sensitive reports follow [SECURITY.md](SECURITY.md).

## Repository layout

- `aspec/`: runtime, model assets and canonical contracts.
- `configs/`, `examples/`: maintained user and benchmark entry points.
- `tools/`, `tests/`: source utilities, reusable verification and regression tests.
- `docs/`: current user guides, support, architecture and release instructions.
- `requirements/`, `patches/`, `.github/`: runtime and engineering configuration.

Run new work in an isolated checkout when other jobs are using the repository. Keep local deployment
helpers, credentials, machine paths and experiment outputs outside the maintained source. Preserve
reproducible public conclusions in the maintained docs and keep raw evidence with its run manifest.
See [source release checks](docs/releasing.md).
