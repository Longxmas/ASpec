# Container setup

Source setup is the primary entry. This optional image supplies native prerequisites and runs the
same `setup.sh`; it is not a separate runtime/dependency lane. Build from the repository root:

```bash
docker build --platform linux/amd64 -t aspec:local .
```

The [Dockerfile](../../Dockerfile) pins NVIDIA's CUDA 13.0.2 development image on Ubuntu 24.04 by
[manifest digest](https://hub.docker.com/layers/nvidia/cuda/13.0.2-devel-ubuntu24.04/images/sha256-0eee3094c71518ad31d011a594ae6ed6de72959ee07e318cb31cffe71690e90c).
It installs Python 3.12, C++, Git and RDMA runtime libraries, then runs source setup into
`/opt/aspec-venv`, including the pinned Q/K specialization. Building needs network access and
substantial disk space; models, tokens, the default local venv and documented caches/outputs are
excluded from the build context. The image does not contain model weights or cluster credentials.

Use a Docker host with NVIDIA Container Toolkit and a compatible driver. For example, select the
two allocated GPUs and use explicitly prepared host cache/output directories:

```bash
mkdir -p container-cache container-outputs
docker run --rm -it --gpus '"device=0,1"' --shm-size=16g \
  -v "$PWD/container-cache:/cache" -v "$PWD/container-outputs:/outputs" \
  -e ASPEC_CACHE_HOME=/cache aspec:local
# Inside the container, choose either model:
# First start Target in another terminal:
# source /opt/aspec-target/bin/activate
# CUDA_VISIBLE_DEVICES=0 aspec target
CUDA_VISIBLE_DEVICES=1 aspec quickstart ling-tiny --output /outputs/ling-tiny-001
# Or use qwen3-8b with a different fresh output.
```

The image activates Draft through `PATH`/`VIRTUAL_ENV`. In another container terminal, activate
`/opt/aspec-target/bin/activate` and run `aspec target` before training. Both environments are
already installed. Optional [local supervision](installation.md#optional-local-supervision) can
use `/opt/aspec-target/bin/python` instead of starting Target manually.
The selected GPUs are numbered within the container. Mount an existing model explicitly and pass
`--target /container/path/to/model` if downloading is undesirable. The image preserves NVIDIA's
entrypoint and does not install a host driver, start a cluster, or assume SSH/shared storage.
Multi-node execution still needs the [per-node cluster setup](multinode.md).

Container build/run verification is separate from native source/GPU checks; do not infer one from
the other. [Advanced installation](installation-advanced.md) covers older drivers and native libraries.
