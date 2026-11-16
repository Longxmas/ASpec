# Native MTP conditions a DSpark or DFlash Draft

This example adds a frozen Qwen3.5-9B native MTP predictor to the ordinary ASpec
training workflow. It trains the Draft and, in hidden mode, a zero-initialized
residual fusion. The Target, its embedding/output head and the native predictor
stay frozen. It does not define a separate training engine or a serving runtime.

## Configuration and training

Activate the [isolated Linux environment](../../docs/user-guide/installation.md).
Set disjoint training/evaluation inputs using the [supported data formats](../../docs/user-guide/data.md):

```bash
# In the Target environment:
CUDA_VISIBLE_DEVICES=4,5,6,7 aspec target
# In the Draft environment:
export CUDA_VISIBLE_DEVICES=0,1,2,3
export ASPEC_TARGET_MODEL=/path/to/Qwen3.5-9B
export ASPEC_TRAIN_DATA=/path/to/train.parquet
export ASPEC_EVAL_DATA=/path/to/heldout.parquet
export RAY_ADDRESS=local

aspec --config configs/recipes/qwen3_5_9b_dmtp.yaml --dry-run
aspec --config configs/recipes/qwen3_5_9b_dmtp.yaml
aspec --report outputs/qwen3_5_9b_dmtp
```

The starting recipe uses DSpark, five full-attention GQA layers, four Draft GPUs
and four TP1 Target GPUs. `mtp_conditioning` in the Draft JSON has exactly two
fields: `mode` (`token` or `hidden`) and `native_config` (the native Qwen3.5 text
configuration). Hidden mode requires `inference.store_last_hidden_states=true`.
Ordinary selected-layer and final Target features are transported unchanged;
the Draft derives the native MTP state locally. No second Target transformer or
vocabulary-sized embedding/head is allocated for the MTP body.

To use DFlash, select `aspec/config/dflash_qwen3_5_9b_dmtp_config.json`, set
`training.dflash_num_target_layers=5`, and choose the DFlash block/anchor/loss
settings. The current objective is `decay`. To compare token-only conditioning,
copy the JSON to a new file and change `mtp_conditioning.mode` to `token`.
A plain Draft omits `mtp_conditioning` entirely. These are separate workloads.

The first supported scope is Dense full bidirectional GQA, one complete document
per microbatch, CP1, REPLICATE, one node, no packing or whole-model compilation.
Plan rejects unsupported combinations. The native config and all frozen tensors
must match the same Target; loading and recovery validate that binding.

## Timing and supervision

For original anchor `a`, Target context ends strictly before `a`. Native MTP takes
post-norm Target `h[a-1]` and known token `x[a]`, producing the state that predicts
`T1=x[a+1]`. The Draft input is `[A,T1,MASK,...]`. Training teacher-forces T1;
hidden mode adds a normalized, zero-initialized embedding residual at its input
slot. With identical Draft weights, this initialization has the same forward output
as token-only conditioning.

DFlash supervises input slots `2..B-1`; DSpark supervises prediction slots `1..B-1`.
The visible T1 is excluded from loss, accuracy and decay position zero. DSpark
reports `suffix_accept_rate@k` and `suffix_tau_probabilistic` conditional on T1.
Both algorithms report `train/suffix_simulated_acc_len` and
`eval/suffix_simulated_acc_len`; their per-position `suffix_acc_0` and
`suffix_ploss_0` start at T2. Checkpoint selection and evaluation metadata use the
suffix proxy explicitly. Plain DSpark/DFlash runs retain their ordinary metric names.
These observations omit acceptance of the native MTP proposal and cannot establish
full-block accepted length or serving speed.

## Recovery and assets

Use a fresh output directory for exact resume, preserving the model policy,
source identity and full learning-rate horizon:

```bash
aspec --config configs/recipes/qwen3_5_9b_dmtp.yaml \
  training.load_path=outputs/qwen3_5_9b_dmtp/checkpoints \
  output_dir=outputs/qwen3_5_9b_dmtp-resumed
aspec export --format aspec --input-dir /path/to/complete/iter_checkpoint \
  --output-dir outputs/qwen3_5_9b_dmtp-asset
```

Native assets retain conditioning config, fusion and frozen predictor weights.
`aspec.draft_artifact.load_native_draft` strictly reloads them. Generic SGLang
export and the ordinary block-proposal consumer reject conditioned assets;
a downstream MTP-aware consumer must implement the actual proposed T1 and its
verification/cache timing. Export does not certify that consumer.

`model.initial_draft_model_path` is weights-only initialization. A plain Draft
can start a new conditioned phase with initialized native tensors and zero
fusion; undeclared or partial MTP tensors fail strict loading. Exact resume
cannot change the conditioning mode or native policy.

## Framework validation

Feature regressions cover configuration rejection, native predictor timing,
future-token isolation, visible-token loss masking, frozen parameter ownership,
weights-only initialization, exact recovery and strict native asset loading.
The recipe checks exercise the ordinary training, recovery and export entry points.
See [contributor validation](../../docs/contributing/validation.md) for the
additional GPU and recipe checks required for this path.
