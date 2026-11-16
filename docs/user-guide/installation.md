# Installation

Use Linux x86-64, Python >=3.11 and an isolated virtual environment. Training requires NVIDIA
GPUs and a CUDA 13-compatible driver. The Target package supplies its matching CUDA compiler;
the host still needs the C++20 compiler and native libraries listed below.
See [native prerequisites](installation-advanced.md#requirements) for the compiler, glibc and RDMA
library requirements, or use the [container](containers.md).

## Install from PyPI

```bash
python -m venv .venv
source .venv/bin/activate
pip install ling-aspec
```

This installs ASpec and its Draft training dependencies. The Python package and command are named
`aspec`. Keep the unrelated PyPI package `aspec` out of this environment.

## Target nodes

In a separate virtual environment on nodes that run the Target model:

```bash
python -m venv .venv-target
source .venv-target/bin/activate
pip install ling-aspec-target
aspec patch-sglang
CUDA_VISIBLE_DEVICES=0 aspec target
```

Target dependencies follow SGLang's requirements; Draft acceleration packages stay in the Draft
venv. The patch command only changes the supported SGLang source files, never dependencies.

Keep this service running. In the Draft environment, start training:

```bash
CUDA_VISIBLE_DEVICES=1 aspec train --config /path/to/run.yaml
```

The default Target address is `http://127.0.0.1:30000`. Set `inference.target_service.address`
for a remote service, or pass `--target-address` to Quickstart. Use an IPv4 address or a hostname
that resolves to IPv4. Use the same ASpec release in both environments.
Each role allocates its own GPUs; select disjoint GPU sets for a same-host run.
The example uses one GPU per role; adjust both selections to match your config.
`aspec target --config /path/to/run.yaml` reads its address and Target model from the same config.
The service remains available after a training session ends; each session releases its own workers.

Recheck the patch after reinstalling SGLang:

```bash
aspec patch-sglang --check
```

Applying the patch again verifies an already patched installation. Conflicting changes or an
unsupported SGLang version are rejected. `--check` never installs dependencies or modifies SGLang files.

## Check the environment

Optionally run `aspec doctor` in Draft, or `aspec doctor --target` in Target to
check its runtime and patch. Add `--gpu` to check CUDA and native libraries on an
allocated GPU. These checks do not run training.

## First run

Quickstart requires two NVLink-connected GPUs:

```bash
CUDA_VISIBLE_DEVICES=1 aspec quickstart ling-tiny --output outputs/ling-tiny-001
# Or:
CUDA_VISIBLE_DEVICES=1 aspec quickstart qwen3-8b --output outputs/qwen3-8b-001
```

Add `--target /absolute/path/to/model` to use a local checkpoint.
Continue with [Quickstart](quickstart.md) or choose a [training recipe](recipes.md).

## Install from source

From a source checkout:

```bash
bash setup.sh
source .venv/bin/activate
```

Setup installs the Draft environment. Run `bash setup.sh --target` to prepare `.venv-target`,
then activate it and start Target on its allocated GPUs as above. Each setup runs the appropriate doctor. Contributors can use
`bash setup.sh --dev`. Run `bash setup.sh --help` for environment and interpreter options.

## Install a release wheel

Use `pip install --find-links /path/to/release ling-aspec` in Draft, and
`pip install --find-links /path/to/release ling-aspec-target` in Target. Each role installs
the same `ling-aspec-runtime` wheel, which owns the `aspec` code and CLI. Then prepare Target nodes and optionally check the environment as above. The installed commands,
recipes and Quickstart data work outside the source checkout.

See [advanced installation](installation-advanced.md) for native-library paths and manual
dependency management, or [troubleshooting](troubleshooting.md) if a check fails.

## Optional local supervision

To start and stop Target automatically for a single-node run, add:

```yaml
inference:
  target_service:
    mode: managed-local
    python_executable: /absolute/path/to/target-venv/bin/python
```

Then run the same `aspec train --config ...` command. This optional supervisor needs the
Target environment location; independent startup only needs a service address. Both forms use
the same Target interface and support checkpoint restoration. Logs go to `target-service.log`
inside the run directory. Existing services are never adopted or stopped by the supervisor.
Quickstart also accepts `--managed-target /absolute/path/to/target-venv/bin/python`.
For managed-local, use a single-node Draft Ray runtime and select the combined Draft
and Target GPUs. The supervisor splits them according to the configured counts and placement order.
