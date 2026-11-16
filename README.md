# ASpec

[中文概览](README_zh-CN.md) · [Documentation](docs/README.md) · [Roadmap](ROADMAP.md) · [Quickstart](docs/user-guide/quickstart.md)

**Faster draft training. More room to experiment.**

ASpec is a high-throughput training framework for speculative decoding draft models. It optimizes
Draft forward, backward and optimizer execution, overlaps training with Target feature generation,
and shares compatible Target features across independent experiments.

ASpec is developed from the [TorchSpec](https://github.com/lightseekorg/TorchSpec) codebase,
building on its training infrastructure with further execution and workflow optimizations.

Use one workflow to explore algorithms and model structures, train at long context, compare
independent experiments, and take a checkpoint through evaluation and export.

![ASpec architecture: asynchronous Target feature generation, GPU tensor transport, and independent Draft experiments sharing compatible features.](docs/assets/architecture.svg)

## Why ASpec?

- **Speed up Draft training itself.** Optimized attention, fused kernels and selective head/loss
  computation reduce the work and memory traffic in each forward and backward pass.
- **Keep the training pipeline moving.** Overlap Target feature generation with Draft training,
  move features through a GPU tensor data path, and schedule training by useful work.
- **One Target, multiple concurrent Draft experiments.** Compare algorithms, structures or
  hyperparameters while generating compatible Target features once. Each Draft keeps its own
  weights, optimizer, progress and checkpoints, without a shared optimizer-step barrier.
  See [shared Target training](docs/user-guide/sharing.md) and the
  [comparison and recovery walkthrough](examples/qwen3-8b-shared-target/README.md).
- **Make long-context training practical.** Combine sequence packing, context parallelism and
  optimized attention execution to reduce padding waste and distribute long sequences across GPUs.
- **Explore without rebuilding the workflow.** Switch between algorithm and model recipes with
  consistent configuration, reporting, recovery, evaluation and export.

## Choose your Draft

Choose the algorithm and model structure for your workload. Draft and Target models can have
different architectures:

| Dimension | Implemented choices |
| --- | --- |
| Draft algorithm | DSpark, DFlash, EAGLE3, EAGLE4, native MTP; experimental DFlash2 |
| FFN and routing | Dense, Qwen MoE, Ling MoE |
| Attention structure | MHA/GQA, materialized MLA, sliding-window attention (SWA), CSA, QSA, DSA |
| Execution | Sequence packing, context parallelism, shared Target features, static multi-node training |

See [algorithm details](docs/user-guide/algorithms.md) and [structure stages](docs/user-guide/structures.md)
for the supported combinations, training phases and export formats.
DSpark/DFlash with Qwen and Ling are the main optimization and recipe priorities; other implemented
paths have explicit support boundaries, and DFlash2 remains experimental.

Scale from a first multi-GPU run to static multi-node training, with NVLink, TCP and RDMA data paths.
Check [supported paths and limitations](docs/support.md), then browse the [recipes](docs/user-guide/recipes.md) for starting points.

## Installation

In an activated Linux virtual environment:

```bash
pip install ling-aspec
```

This installs the Draft training environment. In a **separate Target virtual environment**, install
the SGLang integration and apply its patch:

```bash
pip install ling-aspec-target
aspec patch-sglang
CUDA_VISIBLE_DEVICES=0 aspec target
```

Keep the Target service running, then start training in the Draft environment below.
The default connection is `http://127.0.0.1:30000`; set `inference.target_service.address`
for a remote Target. Each command uses its own environment.
Select the allocated GPUs separately for each role; this example uses GPU 0 for Target and GPU 1 for Draft.

See the [installation guide](docs/user-guide/installation.md) for prerequisites, source installation
and release wheels, or use the [container setup](docs/user-guide/containers.md).
Optionally run `aspec doctor` to check the environment.

## Quickstart

With two NVLink-connected GPUs:

```bash
CUDA_VISIBLE_DEVICES=1 aspec quickstart ling-tiny --output outputs/ling-tiny-001
# Or choose Qwen3-8B:
CUDA_VISIBLE_DEVICES=1 aspec quickstart qwen3-8b --output outputs/qwen3-8b-001
```

Both profiles use bundled raw conversations, their own tokenizer/template and an eight-step
train/resume/native-export check. Add `--target /absolute/path/to/model` to use a local checkpoint.
See [Quickstart](docs/user-guide/quickstart.md) for the full workflow, then choose a
[training recipe](docs/user-guide/recipes.md) and [prepare your data](docs/user-guide/data.md).
For training with Target-generated data and held-out evaluation, continue with the
[Qwen3-8B DSpark recipe](docs/user-guide/recipes.md#qwen3-8b-dspark-quality-recipe).

Additional workflows cover [offline feature generation and replay](examples/qwen3-8b-offline/README.md)
and [DSA dense → indexer → sparse training](examples/qwen3-8b-dsa/README.md).

## Performance

ASpec improves both Draft execution and the complete online training pipeline. The
[DSpark comparison](docs/benchmarks/results.md) measures Qwen3-8B, Qwen3.5-9B and Qwen3.8-27B
on Hopper GPUs at short and long context. A representative Qwen3-8B / 4K result:

| Measurement | ASpec input tokens/s | / TorchSpec | / SpecForge | / AngelSpec |
| --- | ---: | ---: | ---: | ---: |
| Draft-only, 1 GPU | 15,256.44 | 1.96× | 2.07× | 2.48× |
| Online steady window, 8 GPUs | 77,721.37 | 2.28× | 2.50× | 2.68× |

The comparison uses one original document per Draft rank and microstep (MBS1), aligned per-document
anchor budgets and logical batch sizes, normal interconnects and sufficient buffers. Draft-only
excludes Target execution, transport and feature loading; the online steady window includes them
after warmup update eight. These are training input rates, not inference speedups or time to a given
model quality. The full matrix reports software stacks, GPU splits and complete-loop rates separately.

A separate [fixed-sequence Qwen3-8B / CP1 study](docs/benchmarks/dspark-alignment.md) compares
training quality: BSHD validation CE stays close to TorchSpec; measured mean accepted length
(tokens per verification round, including the Target bonus token) is
3.665717 for TorchSpec, 3.691665 for ASpec BSHD and 3.746300 for ASpec THD. With deterministic
inference enabled, all three match Target-only greedy output token IDs on 256 questions across
three repeats. These findings cover that training sequence and evaluation configuration.

For your own training run, see [DSpark performance settings](docs/user-guide/recipes.md#dspark-performance-settings)
for cuDNN attention and direct-prefill options. Resource splits depend on the workload.

See [benchmark results and methodology](docs/user-guide/benchmarks.md) for measured workloads and
reproducible comparisons. To assess the drafts you train, use the
[evaluation and export guide](docs/user-guide/evaluation.md) for held-out metrics, acceptance and
downstream validation.

## Learn more

- [Configuration](docs/user-guide/configuration.md) · [Training and recovery](docs/user-guide/training.md)
- [Evaluation and export](docs/user-guide/evaluation.md) · [Troubleshooting](docs/user-guide/troubleshooting.md)
- [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)
- [Architecture](docs/framework/ARCHITECTURE.md) · [Project roadmap and status](ROADMAP.md)
- [Supported paths](docs/support.md) · [Contributor validation](docs/contributing/validation.md)

## Citation

If you find ASpec useful, please consider citing it:

```bibtex
@misc{aspec2026,
  author = {Zhou, Aihui and Long, Lingkun and Yu, Chaofan and
            Yang, Zhe and Peng, Cunyin and Zhang, Zhiqiang and Zhou, Jun},
  title = {{ASpec}: A High-Throughput Training Framework for Speculative Decoding},
  year = {2026},
  url = {https://github.com/inclusionAI/ASpec}
}
```

Citation metadata is available in [CITATION.cff](CITATION.cff).

## License

ASpec is licensed under [Apache 2.0](LICENSE). Third-party material retains its own licenses
and notices; see the source headers and [license texts](licenses/). Model and dataset licenses
are separate.

## Acknowledgements

We thank [TorchSpec](https://github.com/lightseekorg/TorchSpec),
[AngelSpec](https://github.com/Tencent/AngelSpec), and
[SpecForge](https://github.com/sgl-project/SpecForge) for their contributions to the speculative
decoding training ecosystem.
