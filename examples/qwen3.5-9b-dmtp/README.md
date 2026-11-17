# Native MTP conditions a DSpark or DFlash Draft

This example uses Target-side frozen Qwen3.5-9B native MTP features in the ordinary ASpec
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
and four TP1 Target GPUs. `mtp_conditioning` in the Draft JSON specifies
`mode` (`token` or `hidden`), `native_config` (the native Qwen3.5 text configuration),
and `feature_source: target`. The Target publisher computes the frozen predictor
once per document and publishes `mtp_hidden_states` alongside ordinary features.
The Draft owns the trainable residual fusion, without allocating the predictor.
DSpark still requires `inference.store_last_hidden_states=true` for its own
distillation/confidence objectives; DFlash CE-only conditioning does not.

The feature contract identifies the native configuration, embedding binding and
`anchor_post_norm_v1` alignment. Row `a` is the post-norm state predicting `x[a+1]`,
computed from `x[a]` and Target `h[a-1]`; row zero is padding. Shared consumers can
reuse one physical capture, and offline materialization retains the same tensor
and contract. No MTP vocabulary projection or token sampling is needed in training.

TP1 borrows the loaded Target embedding. For Target TP greater than one, the
publisher rank holds a replicated embedding and predictor; Plan budgets these
separately from Draft parameters and transfer buffers. Target capture requires
PP1 and unquantized native weights. No second Target transformer or MTP output
head is loaded.

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

Native assets retain the conditioning config, fusion and a fingerprint binding
the frozen predictor to its Target checkpoint. Target-side assets contain no
predictor weights; the matching Target checkpoint supplies them.
`aspec.draft_artifact.load_native_draft` strictly reloads them. Generic SGLang
export and the ordinary block-proposal consumer reject conditioned assets;
a downstream MTP-aware consumer must implement the actual proposed T1 and its
verification/cache timing. Export does not certify that consumer.

`model.initial_draft_model_path` is weights-only initialization. A plain Draft
can start a new conditioned phase with initialized native tensors and zero
fusion; undeclared or partial MTP tensors fail strict loading. Exact resume
cannot change the conditioning mode or native policy.

Existing configs and assets without `feature_source` retain Draft-side predictor
execution and their original checkpoint identity. Set the source explicitly for
a new Target-side run; changing it is not an exact-resume transition.

## Framework validation

Feature regressions cover configuration rejection, native predictor timing,
future-token isolation, visible-token loss masking, frozen parameter ownership,
weights-only initialization, exact recovery and strict native asset loading.
The recipe checks exercise the ordinary training, recovery and export entry points.
See [contributor validation](../../docs/contributing/validation.md) for the
additional GPU and recipe checks required for this path.
