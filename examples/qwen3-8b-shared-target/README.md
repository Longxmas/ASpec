# Compare Draft experiments and resume them independently

This source-checkout walkthrough reuses the existing
[shared-target recipe](../../configs/recipes/qwen3_8b_4k_multi_draft_nvlink.yaml).
Two DSpark Drafts share compatible Qwen3-8B Target features, use different learning
rates, and own separate training state, evaluation curves and checkpoints.
The short run demonstrates experiment management; it does not establish which
learning rate converges better.

Use the [isolated Linux environment](../../docs/user-guide/installation.md), with
the [TensorBoard dependency](../../docs/user-guide/training.md#tensorboard-and-wb)
installed, and run from the repository root. The recipe allocates eight GPUs on
one NVLink node: four Target GPUs and two GPUs for each Draft. It uses CP1, MBS1,
four accumulation steps and no packing, giving eight documents per optimizer
update per Draft.

In the Target environment, keep `CUDA_VISIBLE_DEVICES=4,5,6,7 aspec target` running.
The commands below run in Draft's environment on GPUs 0–3.

Provide a local Qwen3-8B checkpoint and disjoint train/eval conversation files
following the [data format guide](../../docs/user-guide/data.md). Samples must fit
4K context and contain at least 32 supervised tokens. For a first run, a small
subset such as 128 training and 16 evaluation samples is sufficient.

```bash
export ASPEC_TARGET_MODEL=/path/to/Qwen3-8B
export ASPEC_TRAIN_DATA=/path/to/train.jsonl
export ASPEC_EVAL_DATA=/path/to/eval.jsonl
export ASPEC_SHARED_OUTPUT=outputs/qwen3-shared-compare
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RAY_ADDRESS=local
```

## Prepare and run the comparison

Create a new directory and derive a resolved configuration from the recipe.
Keep both complete experiment entries. This example gives both Drafts seed 42 and
a shared source-order seed, so the intended hyperparameter difference is the
learning-rate schedule: A uses `6e-4 → 6e-5`, B uses `3e-4 → 3e-5`. Both schedules
have the same eight-update horizon. B stops at update four to provide a different
recovery frontier.

```bash
python - <<'PY'
import os
from pathlib import Path
from omegaconf import OmegaConf
from aspec.config.train_config import load_config

root = Path(os.environ["ASPEC_SHARED_OUTPUT"]).expanduser().resolve()
root.mkdir(parents=True, exist_ok=False)
config = load_config(
    "configs/recipes/qwen3_8b_4k_multi_draft_nvlink.yaml",
    cli_args=[
        "dataset.chat_template=qwen3-instruct",
        "dataset.dataset_seed=42",
        "dataset.eval_interval=4",
        "training.seed=42",
        "training.num_train_steps=8",
        "training.lr_total_steps=8",
        "training.save_interval=4",
        "training.save_per_epoch=false",
        "logging.report_to=tensorboard",
        f"output_dir={root / 'first'}",
        f"cache_dir={root / 'cache'}",
    ],
    save_snapshot=False,
)
for experiment in config.experiments:
    experiment.training.seed = 42
    experiment.training.num_train_steps = 8 if experiment.name == "draft_a" else 4
OmegaConf.save(config, root / "first.yaml", resolve=True)
PY
```

Inspect the shared Plan and run it:

```bash
aspec --config "$ASPEC_SHARED_OUTPUT/first.yaml" --dry-run
aspec --config "$ASPEC_SHARED_OUTPUT/first.yaml"
aspec --report "$ASPEC_SHARED_OUTPUT/first"
aspec --report "$ASPEC_SHARED_OUTPUT/first/draft_a"
aspec --report "$ASPEC_SHARED_OUTPUT/first/draft_b"
```

Expected optimizer progress is A `0 → 8` and B `0 → 4`. Each child directory has
its own `config.yaml`, `plan.json`, `evidence.json`, `tracking.json`, `runs/` and
`checkpoints/`. Verify healthy completion, evaluation and cleanup in the reports.
`tracking.json` records each writer's location and closure state.

Open local curves with:

```bash
tensorboard --logdir "$ASPEC_SHARED_OUTPUT/first"
```

Select `draft_a/runs` and `draft_b/runs` and compare the same metric at matching
optimizer steps, especially held-out evaluation at step four. In this configuration,
training scalar steps, evaluation and checkpoint frontiers count completed updates.
A's step-eight result cannot be compared with B's step-four
result as an equal-budget quality comparison. Common seeds do not establish
identical asynchronous execution or a statistically reliable quality ranking.
Shared Target production and per-Draft consumption have different scopes; do not
sum child consumption and call it unique Target work or isolated throughput.

## Resume B while A is already complete

Use the **saved root configuration**, which contains the full experiment list.
Point each child at its own explicit checkpoint: A at eight completed updates,
B at four. Checkpoint iteration names are one ahead of completed updates.
Preserve all model/data/order, execution, optimizer and LR-horizon settings.
Clear both phase-transition flags within every experiment.

```bash
python - <<'PY'
import os
from pathlib import Path
from omegaconf import OmegaConf

root = Path(os.environ["ASPEC_SHARED_OUTPUT"]).expanduser().resolve()
config = OmegaConf.load(root / "first/config.yaml")
frontiers = {"draft_a": 9, "draft_b": 5}
assert {experiment.name for experiment in config.experiments} == set(frontiers)
for experiment in config.experiments:
    checkpoint = root / "first" / experiment.name / "checkpoints" / f"iter_{frontiers[experiment.name]:07d}"
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    experiment.training.load_path = str(checkpoint)
    experiment.training.num_train_steps = 8
    experiment.training.continual_training = False
    experiment.training.continue_source_cursor = False
config.output_dir = str(root / "resumed")
with (root / "resume.yaml").open("x") as stream:
    stream.write(OmegaConf.to_yaml(config, resolve=True))
PY
```

```bash
aspec --config "$ASPEC_SHARED_OUTPUT/resume.yaml" --dry-run
aspec --config "$ASPEC_SHARED_OUTPUT/resume.yaml"
aspec --report "$ASPEC_SHARED_OUTPUT/resumed"
aspec --report "$ASPEC_SHARED_OUTPUT/resumed/draft_a"
aspec --report "$ASPEC_SHARED_OUTPUT/resumed/draft_b"
```

Expected progress is A `8 → 8` with `no_new_work`, and B `4 → 8`. A's committed
work must remain intact; B restores its own model, optimizer, scheduler, RNG and
source frontier. This demonstrates that finishing one experiment does not advance
another experiment's progress. The session retains the original eight-GPU
allocation and initializes both children; completed A is not a GPU-rescheduling
mechanism. A failed or cancelled child makes the shared session exit nonzero.

A's report is `completed / incomplete_evidence` because it performs no new
training or feature transfer; the root report is therefore `incomplete_or_failed`.
Check A's `no_new_work`, unchanged committed frontier, closed tracking and completed
cleanup. B should report `completed / healthy`.

Keep the full experiment list even when only B has new work. Individual list-index
dotlist overrides are unsupported, and replacing `experiments` with partial
entries can discard the original configuration. To resume both children from
different frontiers, choose checkpoints below their intended total horizons.
See [recovery semantics](../../docs/user-guide/training.md#exact-resume); this
workflow restores each child independently; it does not guarantee equality with
an uninterrupted training trajectory.

TensorBoard records the resumed attempt separately, preserving restored metric
steps rather than appending to the old event files. To view both attempts, point
`tensorboard --logdir` at `$ASPEC_SHARED_OUTPUT`; B's training steps continue at
five and its next evaluation is at eight. Use A's first attempt for its completed
curves, since its resumed attempt performs no new optimizer updates.

For optional native export, A's completed asset comes from the first run and B's
from the resumed run:

```bash
aspec export --format aspec \
  --input-dir "$ASPEC_SHARED_OUTPUT/first/draft_a/checkpoints/iter_0000009" \
  --output-dir "$ASPEC_SHARED_OUTPUT/asset-a"
aspec export --format aspec \
  --input-dir "$ASPEC_SHARED_OUTPUT/resumed/draft_b/checkpoints/iter_0000009" \
  --output-dir "$ASPEC_SHARED_OUTPUT/asset-b"
```

Use a fresh output root for another attempt. See [shared Target support](../../docs/user-guide/sharing.md)
and [export boundaries](../../docs/user-guide/evaluation.md).
