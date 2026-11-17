# Algorithm recipes

Training recipes under `configs/recipes/` require absolute `ASPEC_TRAIN_DATA` and
`ASPEC_EVAL_DATA` inputs before loading (Quickstart is the exception). See the
[recipe launcher](recipes.md) and [runtime validation](../contributing/validation.md).

Choose algorithm/objective separately from Draft structure and execution. All commands run from
repository root in the [installed environment](installation.md). Bundled conversations and short
step budgets are functional examples; supply licensed real data and a full LR horizon for training.
Use [support](../support.md) for unimplemented combinations.

## DSpark, DFlash and EAGLE3

Select `configs/recipes/qwen3_8b_dflash.yaml` or `configs/recipes/qwen3_8b_eagle3.yaml` and replace the Target path:

```bash
aspec --config configs/recipes/qwen3_8b_eagle3.yaml --dry-run \
  model.target_model_path=/path/to/Qwen3-8B
RAY_ADDRESS=local aspec --config configs/recipes/qwen3_8b_eagle3.yaml \
  model.target_model_path=/path/to/Qwen3-8B
```

DSpark retains CE/L1/Markov/confidence semantics; DFlash does not use those auxiliary targets.
EAGLE3 uses recurrent TTT and its selected distillation objective. Its pruned vocabulary preserves
the checkpoint's `d2t/t2d` mapping across recovery/export. Teacher argmax outside the selected
vocabulary is filtered; omitted probability mass is not reassigned to the reduced head.

The full-vocabulary recipe sets `training.eagle3_target_cache=logits`. This projects the frozen
Target head once per microbatch and reuses the resulting BF16 logits across TTT depths and
checkpoint recomputation; each depth still performs the same FP32 softmax and distillation loss.
The optimization is supported for padded `sdpa` and `flex_attention`. Use `lazy` for pruned-vocabulary
or FA4/THD execution; unsupported combinations fail during Plan compilation rather than falling back.

Backbone and loss checkpointing are separate. The full-vocabulary recipe keeps
`training.gradient_checkpointing=true` for the Draft backbone but sets
`training.eagle3_loss_gradient_checkpointing=false`, retaining the loss graph so backward does not
repeat the full-vocabulary projection and reductions. This changes activation-memory use, not dtype
or loss mathematics. The optimized setting is supported on the same padded full-vocabulary path;
use loss checkpointing for pruned-vocabulary or FA4/THD execution.
When omitted or `null`, the loss-checkpoint setting inherits `gradient_checkpointing`, preserving
the behavior of existing configurations. Plan and runtime observations report the resolved boolean.

For a pruned EAGLE3 packed CP example, change the original recipe as follows:

```bash
aspec --config configs/recipes/qwen3_8b_eagle3.yaml --dry-run \
  model.draft_model_config=aspec/config/eagle3_qwen3_8b_pruned_draft_config.json \
  training.attention_backend=fa4 training.dflash_packing=true \
  training.eagle3_target_cache=lazy \
  training.eagle3_loss_gradient_checkpointing=true \
  training.draft_context_parallel_size=2 output_dir=outputs/eagle3-pruned-cp2
```

`dflash_packing` is the shared configuration field name; it does not change EAGLE into DFlash.
TTT shifts, teacher halos and supervision remain document-local. Unsupported factorized/ambiguous
heads fail export rather than producing an unfaithful serving asset.

## P-EAGLE core prerequisite

The [P-EAGLE model/loss API](peagle.md) provides a dense unsampled Qwen3 parallel
Draft and forward-KL objective with independent numerical checks. Public training,
COD sampling, recovery, export and consumer validation remain pending.

## EAGLE4: two stages

The Draft stage trains the full recurrent model. The candidate stage freezes the entire Draft,
including its full-vocabulary head, and trains two low-rank retrieval projections. The candidate
objective uses frozen-head Top-K soft targets and forward-KL with full-vocabulary normalization;
SVD is initialization only. This is not claimed as an unpublished official training recipe.

The default Dense example uses three NVLink GPUs. Train and export the Draft before the candidate stage:

```bash
RAY_ADDRESS=local aspec --config configs/recipes/qwen3_8b_eagle4.yaml \
  model.target_model_path=/path/to/Qwen3-8B
python -m tools.convert_to_hf --format aspec \
  --input-dir /path/to/published-draft-checkpoint \
  --output-dir outputs/qwen3-8b-eagle4-asset
RAY_ADDRESS=local aspec --config configs/recipes/qwen3_8b_eagle4_candidate.yaml \
  model.target_model_path=/path/to/Qwen3-8B \
  model.initial_draft_model_path=outputs/qwen3-8b-eagle4-asset
```

Use the complete checkpoint published by the first stage; normal training defaults to three data passes,
so the iteration number depends on the admitted data and batch schedule.
Export the candidate checkpoint with `--format aspec` too. Native assets retain the full head,
projections, feature/norm/RoPE/SWA settings and MoE control state. Candidate scoring exactly rescores
retrieved rows with the full head; low-rank logits are not final scores.

Cross-stage initialization is weights-only. Within a stage, exact resume preserves stage, rank,
supervision strategy, frozen weights and all optimizer/RNG/work state. Set
`model.initial_draft_model_path=null` with `training.load_path`, retain `lr_total_steps`, and choose
a fresh output directory. Candidate loss chunk size controls workspace and can change; candidate
rank and stage semantics cannot. Candidate recall/teacher mass are not decode acceptance metrics.

## DFlash2

`configs/examples/qwen3_8b_dflash2.yaml` illustrates the experimental objective. Select
`aspec/config/dflash2_qwen3_8b_swa_draft_config.json` through `model.draft_model_config` for SWA;
packing and CP use the normal training overrides. Full/causal/SWA semantics differ from DSpark's
anchor-shared window. FA4 preserves
convolution, selector, document boundaries and CP ownership. Functional results do not establish
official-recipe reproduction or long-term quality.

## Native MTP conditioning for DSpark/DFlash

`configs/recipes/qwen3_5_9b_dmtp.yaml` uses the native Qwen3.5 predictor as a frozen
condition for a DSpark Draft. DFlash uses the corresponding DMTP Draft JSON.
The ordinary trainer owns fusion parameters, optimizer, recovery and native assets;
Target selected-layer/final-state transport stays unchanged. Dense full GQA, CP1,
MBS1, REPLICATE, one node and non-packing are the initial scope. Other combinations
fail Plan validation. Follow the [workflow and timing contract](../../examples/qwen3.5-9b-dmtp/README.md).
Given teacher-forced T1, only unknown suffix positions receive supervision. Suffix
acceptance observations are conditional; native export requires an MTP-aware downstream
consumer and does not certify generic SGLang serving.

## Native Qwen3.5 MTP

`configs/recipes/qwen3_5_9b_mtp.yaml` initializes the actual native predictor and binds frozen embeddings/head
from the same Target. Final post-norm feature at position `t` plus token `t+1` predicts token `t+2`;
later rollout depths reuse normalized predictor state. It is not an independently randomized EAGLE model.

Use the normal CLI and [recovery workflow](training.md). SDPA is CP1; the FA4 packed path can use CP
with sufficient Draft GPUs and compatible geometry. Export/stitch preserves the shared Target bindings:

```bash
python -m tools.convert_to_hf --input-dir /path/to/complete/iter_checkpoint \
  --output-dir outputs/native-mtp-export
python -m tools.stitch_native_mtp --target /path/to/Qwen3.5-9B \
  --draft outputs/native-mtp-export --output outputs/verifier-with-trained-mtp
```

The original Target is not overwritten. Official SGLang consumption selects `EAGLE`, not a `MTP`
flag. A format manifest does not replace testing the particular exported asset with its consumer.
