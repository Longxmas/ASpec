# Training and recovery

Start with the [quickstart](quickstart.md), then a workload-appropriate [recipe](recipes.md).
Keep config, Plan and Evidence alongside checkpoints; logs alone are not a recovery contract.
For an isolated single-node run, use `RAY_ADDRESS=local` and select idle GPUs explicitly. Only connect
to an existing Ray cluster when you intend to use its resources; multi-node recipes have separate setup.

```bash
aspec --report outputs/my-run
aspec --report outputs/my-run --report-format json
```

`completed/healthy` means runtime predicates passed, not convergence, useful acceptance or universal
performance. `incomplete_evidence` identifies an unobserved requirement; short windows can miss a
requested pipeline event. Do not relabel them healthy.

## Exact resume

Use a fresh output directory and an explicit complete checkpoint:

```bash
aspec --config configs/recipes/quickstart_qwen3_8b.yaml \
  model.target_model_path=/path/to/Qwen3-8B \
  training.load_path=outputs/old-run/checkpoints output_dir=outputs/resumed-run
```

`load_path` accepts the checkpoint parent (latest tracker) or a published `iter_*` directory (explicit
frontier). Model, optimizer/FP32 master, scheduler, metadata, per-rank RNG and committed work must all
load. Missing/incompatible required state fails rather than quietly becoming weights-only.
CPU RNG state must be a valid saved byte tensor; a CUDA resume also requires saved CUDA RNG
state. An incomplete legacy checkpoint cannot be used for exact resume.

For a preplanned split, keep `training.lr_total_steps` equal to the **full** horizon in both segments,
even when the first `num_train_steps` is smaller. Do not silently change data/objective/topology/model
semantics across exact resume. Iteration names are one ahead of completed steps. Hidden `.incomplete`
staging is not published recovery state. Uncommitted features may be recomputed.
With asynchronous packing, those samples may form different rows after restart;
checkpoint-state restoration does not guarantee the same document count or
numerical trajectory at a later optimizer step as an uninterrupted run.

## New phases and initial weights

`model.initial_draft_model_path` loads an asset into a new run without optimizer/scheduler/RNG/progress.
`training.continual_training=true` with `load_path` is likewise an explicit weights-only phase, not
exact resume. CSA/QSA stage transitions retain their documented gradient/asset semantics;
`continue_source_cursor` is explicit when that new phase must retain source progress.
For repeating sources, a fully committed restored epoch advances before the next update, without
replaying those documents, even when the new phase's optimizer starts at step zero.

## Shared Drafts

Each `experiments` entry owns its model, optimizer, LR, RNG, work, output and checkpoints. Set each
experiment's `training.load_path` independently. Compatible Target capture can be shared; optimizer
state and progress cannot. The root report describes the session; child reports describe experiments.
Slow consumers apply bounded backpressure. Failed/cancelled readers are fenced before rights are
reclaimed, while safe healthy siblings continue. Failed/cancelled experiments make session exit nonzero.

## TensorBoard and W&B

Install TensorBoard with `python -m pip install -e '.[tracking]'`. Enable either backend with
`logging.report_to=tensorboard` / `logging.report_to=wandb`, or both with
`logging.use_tensorboard=true logging.use_wandb=true`. The same switches work for single and shared
Target runs; each `experiments` entry may override its own `logging` section.

```bash
export ASPEC_TARGET_MODEL=/path/to/Qwen3-8B
export ASPEC_TRAIN_DATA=/path/to/train.parquet
export ASPEC_EVAL_DATA=/path/to/eval.parquet
RAY_ADDRESS=local aspec --config configs/recipes/qwen3_8b_4k_multi_draft_nvlink.yaml \
  logging.use_tensorboard=true logging.use_wandb=true logging.wandb_mode=offline
tensorboard --logdir outputs/qwen3-8b-multi-draft
```

Shared Target mode owns one TensorBoard writer at `output_dir/<name>/runs` and one W&B run per Draft.
W&B runs use the experiment name and a common session group (`wandb_group`, with a session suffix by
default). `wandb_dir`, when set, is further separated by experiment name. Use `wandb_mode=online`
and your normal credentials for cloud upload; offline mode writes local records only.
Keep W&B credentials out of training YAML and command-line overrides. Use
`WANDB_API_KEY` or an existing W&B login in the submitting environment instead;
explicit config values can be included in resolved config snapshots and logs.
For multi-node runs, configure credentials in each environment that uses W&B.

EAGLE3 CUDA metrics are reported one step behind execution and retain their
originating step and learning rate. The final report is flushed at normal
completion. DSpark and DFlash report metrics immediately.

Training curves use each Draft's `train/step`; evaluation uses its `eval/step`, not W&B's internal
history counter `_step`. Deferred metrics, including the final flush, retain the trainer's original
computation step. Independent resume
starts a **new logging run** in the fresh output directory, retaining restored step coordinates rather
than resetting them or appending to an unrelated cloud run. Compare/overlay attempts using these axes.
`tracking.json` and the child Report expose actual backend locations, W&B identity and close status.
Each Draft flushes/closes its own handles on completion, cancellation or failure; initialization or
logging failure is reported, not silently disabled.

Shared producer counts and process-wide console/system statistics are not duplicated as per-Draft
measurements. The shared session retains source evidence; W&B's automatic process-stat/console capture
is disabled for these concurrent runs. This does not change training, checkpoint or evaluation semantics.

## Stop safely

Use SIGINT/SIGTERM on your own run. Cancellation records/fences owners but does not promise a new
checkpoint on every stop; resume from the last complete one. The optimizer-step deadline defaults
to 1,800 seconds and covers supply/dispatch/native reads/training. Do not extend it to hide deadlock.
The same configured budget bounds controller status, dataset reload and final metric RPCs;
these calls must not hang after computation has finished. Failed readers are stopped and their
process exit is verified before registered source memory is reclaimed. If that fence cannot be
verified, cleanup reports an error and retains the source rather than claiming successful teardown.
This blocked state propagates through the entry point, including preflight failures. The CLI
keeps its Ray driver alive and retries bounded process-exit probes, because normal Python exit
would trigger Ray's automatic actor cleanup. SIGINT/SIGTERM during this wait does not bypass
the fence. Fence the outstanding readers externally before terminating the driver or cluster;
after their exit is verified, the CLI disconnects and still exits with the original failure chain.
Forced driver/host loss is outside this retention guarantee. Programmatic `main()` launchers
use the same guard. Callers of lower-level training APIs receiving `SourceTeardownBlockedError`
must likewise keep the driver alive until readers are fenced.
Never use host-wide `ray stop --force` or kill every GPU process on shared machines.

Isolated producer profiling uses one optimizer-step deadline for placement discovery, native
output disposal and its acknowledgement. A slow placement lookup consumes that budget rather
than receiving an additional full timeout; no disposal reads start once it is exhausted.

## Activation checkpointing and routing identity

`training.gradient_checkpointing=true` enables non-reentrant layer recomputation for DSpark and DFlash during training, including Dense and MoE FFNs. Evaluation runs without recomputation. This trades extra computation for fewer saved activations.

Exact resume and export check the Draft FFN/routing policy, including expert count, top-k, routing normalization and auxiliary-loss settings. Older checkpoints use their saved `draft_config.json` when routing metadata is absent. A legacy MoE checkpoint whose routing cannot be verified is rejected. To change model semantics intentionally, start a separate training configuration under the existing continual-training rules.

## Draft activation and RoPE configuration

DSpark/DFlash Dense FFNs use the Draft's `hidden_act` through the pinned Transformers activation registry, including `silu` and `gelu`. MoE retains its existing SiLU contract. Unknown Dense activation names are rejected during configuration.

The GQA path executes `default`, `linear` and `yarn` RoPE from `rope_scaling` or `rope_parameters`; linear/YaRN use the pinned Transformers formulas. Unsupported scaling types and partial-head rotary settings are rejected. The effective scaling and activation remain part of the model configuration used for checkpoint recovery and export.

## Offline Target features

The [complete offline example](../../examples/qwen3-8b-offline/README.md) provides
paired materialization/replay configs and commands for evaluation, exact resume and
native export on one GPU used sequentially.

Materialize Target features once, then train Drafts from disk with zero Target GPUs. Draft architectures can share the dataset when they request the same Target feature contract. The current path uses single-node, single-Draft, CP1 training and CPU Store/TCP transport.

Use a single-Draft SGLang config with local Target weights and train/eval data. Set `inference.target_prefill_memory_optimized=false`, `training.draft_context_parallel_size=1`, `training.dflash_packing=false`, and the `mooncake` settings shown below. Generate into a new or empty directory:

```bash
python -m aspec.offline.generate --config materialize.yaml --output /data/qwen3-features
```

The generator saves every admitted train/eval sample; training step limits do not truncate generation. The output contains `dataset.json`, `manifest.jsonl` and `samples/*.pt`; training accepts the dataset after generation completes. Interrupted outputs remain incomplete and must be regenerated into a new directory.
Generation RPCs and periods without saved-sample progress use
`training.optimizer_step_timeout_seconds`. `materialization.json` records counts and cleanup,
including startup failures. A cleanup failure produces a nonzero exit even if all samples were saved.

For training, keep the same local Target checkpoint and feature layers, and apply these settings to the normal training config:

```yaml
inference:
  inference_engine_type: offline
  inference_num_gpus: 0
  target_prefill_memory_optimized: false
  offline:
    data_path: /data/qwen3-features
    num_engines: 2
training:
  training_num_nodes: 1
  training_num_gpus_per_node: 8
  draft_context_parallel_size: 1
dataset:
  train_data_path: null
  eval_data_path: null
  renderer: null
mooncake:
  data_plane: store
  protocol: tcp
  enable_gpu_direct: false
  require_nvlink: false
```

Run `aspec --config offline.yaml --dry-run`, then `aspec --config offline.yaml`. CPU actors replay the features while the Target checkpoint supplies frozen embeddings and the teacher head. Evaluation uses the saved eval split; checkpoint and exact-resume commands are unchanged.

Dataset metadata binds the Target configuration, tokenizer and weights by SHA256, plus the feature layers, dtype, layout and normalization convention. Replay validates sample checksums and tensor shapes. Set `training.max_seq_length` to cover all recorded sequences; changing preprocessing requires regenerating the dataset. The dataset identity participates in exact-resume validation.

## DSpark L1 sampling distributions

DSpark L1 supports independent Target and Draft sampling parameters. Both sides default to `temperature=1.0`, `top_p=1.0` and `top_k=-1` (disabled). Configure either side or both:

```yaml
training:
  dspark_l1_target_temperature: 0.8
  dspark_l1_target_top_p: 0.95
  dspark_l1_target_top_k: 50
  dspark_l1_draft_temperature: 1.0
  dspark_l1_draft_top_p: 0.9
  dspark_l1_draft_top_k: 40
```

Each side computes $p=\operatorname{softmax}(z/T)$ in FP32. Top-k and top-p select from this full distribution, then renormalize their intersection, following the pinned SGLang joint-filter convention. Ties use increasing token ID. Top-p retains tokens whose preceding cumulative mass is at most top-p. Temperature must be finite and positive, top-p in $(0,1]$, and top-k `-1` or a positive integer; top-k at least the vocabulary size keeps every token.

The transformed distributions define $L_{\mathrm{L1}}=\sum_i|P_s(i)-Q_s(i)|$ and the confidence label $\alpha=\operatorname{stopgrad}(1-L_{\mathrm{L1}}/2)$. Acceptance metrics use the same distributions. CE uses the original Draft logits, including Markov-head correction. Loss coefficients and masks are unchanged, with no $T^2$ multiplier. These parameters are independent of `decode.*` and apply only to DSpark with positive L1 weight outside indexer warmup.

Draft gradients flow through retained logits and their normalization. Discrete selection has no gradient: top-k one, or disjoint Target/Draft supports, gives zero L1 gradient while the supports stay fixed. CE retains its own gradients.

Default settings retain the existing kernels; non-default settings use the `torch_compiled` L1 path. LM-head computation is unchanged. Exact resume requires the same sampling settings, treating missing fields in older checkpoints as defaults; use weights-only initialization to change them.
