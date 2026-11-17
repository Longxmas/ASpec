# Sources and attribution

Source headers identify copied or adapted paths, revisions and material changes. This page records
the retained source identities; applicable license and notice texts are distributed in
[`licenses/`](../../licenses/).

| Source | Role |
| --- | --- |
| [TorchSpec](https://github.com/lightseekorg/TorchSpec) `6c042a87140a84d13839e341ece2c5c3ada918bc` | Initial runnable training, data, transport and model implementation |
| [TorchSpec](https://github.com/lightseekorg/TorchSpec) `a4e5c1085bfc5990ddcae6cf5832bb26380b38d3` | EAGLE4 backbone adaptation; individual newer paths retain their headers |
| [AngelSpec](https://github.com/Tencent/AngelSpec) | Selected lifecycle, packing, pipeline, NVLink, calibration and performance adaptations, retaining TorchSpec ancestry |
| [AngelSpec](https://github.com/Tencent/AngelSpec) `d3412bed231025ea85d35dbf5e0af44f1ac62a5b` | Thread-default reference retained in `aspec/utils/env.py` |
| Retained [AngelSpec](https://github.com/Tencent/AngelSpec) snapshot `0966eb139d1c4054767e4a37bfc8a33db5989a72` | Concurrency, Direct-CP, Target-prefill, model-configuration and tensor-transfer regression cases; original source/license ancestry remains applicable |
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [FlashAttention](https://github.com/Dao-AILab/flash-attention) | Declared Direct CP, sparse metadata, output compaction and attention bridge derivations |
| Official [SGLang](https://github.com/sgl-project/sglang) v0.5.21 `e00930c5489053f26d86b179cee0d087f846acbb` | Current Target runtime plus `aspec/compat/patches/sglang-v0.5.21.patch` |
| [cuDNN Frontend](https://github.com/NVIDIA/cudnn-frontend) `cb8cec8a9884f77fa514767237502bef7b7ad9f0` | Source of the adapted ASpec-owned DSpark Q/K CuTeDSL specialization and selected Frost primitives |
| cuDNN Frontend **1.31.0 wheel**, with the FlashAttention/CuTe source chain retained | Adapted SM90/SM100 joint GQA backward, pre/postprocess, layouts and host dispatch; [source paths, hashes and licenses](../../aspec/kernels/cutedsl/cudnn_gqa/README.md) |
| [Ling-V2](https://github.com/inclusionAI/Ling-V2) and [DeepSpec](https://github.com/deepseek-ai/DeepSpec) | Declared model, routing and DSpark implementation derivations |
| [DFlash](https://github.com/z-lab/dflash) `07ebd93db9f472af339b644bb70221ad8428328a` | DFlash2 model adaptation inherited through TorchSpec; Z Lab MIT notices retained |
| [Transformers](https://github.com/huggingface/transformers) v4.57.1 `8cb5963cc22174954e7dca2c0a3320b7dc2f4edc` | Adapted `Qwen3MoeSparseMoeBlock` and load-balancing objective in `aspec/models/draft/moe.py` |
| [DMTP](https://sylvan820.github.io/D-MTP/) | Inspiration for dual-anchor MTP hidden conditioning and zero-initialized embedding residual; independent ASpec integration reuses the retained native predictor, without copying DMTP code |
| [Speculators](https://github.com/vllm-project/speculators) `7ec7aba9450b533334e8f7992baf4e93be3c1a57` | Native MTP recurrent objective and checkpoint bindings; ASpec supplies its own training/layout integration |
| [Speculators](https://github.com/vllm-project/speculators) `3702320434bbdaa853f0846a66e6b6b8f1d97aec` | Adapted P-EAGLE model/loss and Qwen3 first-layer fusion; deterministic dense layout and ASpec input/observation boundary |
| [Ling-3.0-flash](https://huggingface.co/inclusionAI/Ling-3.0-flash) `fcc60a52b34c69c6271d0ac8a8cb0815020b6a43` | Adapted MoE routing and independently retained reference class bodies; Apache-2.0 source headers preserved |
| [EAGLE](https://github.com/SafeAILab/EAGLE), [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) and Transformers through TorchSpec | Model/data helpers in `aspec/models/draft/base.py` and `aspec/data/preprocessing.py`; original source headers retain the transitive attribution |

`aspec/utils/usp.py` and the inherited test modules retain TorchSpec's MIT notice even when the
upstream file had no per-file header. ASpec's namespace, contract and regression changes do not
remove that origin. JSON model assets cannot carry comment headers; their retained configuration
ancestry is covered by the distributed notices and license texts.

Use the exact revisions in source headers, requirements and patch headers when reproducing behavior.
The pinned SGLang v0.5.21 runtime retains the source and license chain of its training patch.
Do not infer a source identity from a current upstream branch name. An independent implementation
of public behavior must still state its comparison and must not claim an unpublished official recipe.

Benchmark pages record the public identities required to interpret maintained results.
