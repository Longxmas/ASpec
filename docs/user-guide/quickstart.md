# First run: Qwen3-8B or Ling tiny

Choose **Qwen3-8B** or **Ling-3.0-tiny**. Both use one Target and one Dense DSpark Draft on two
NVLink-connected GPUs, bundled raw conversations and an eight-step functional workflow. Each model
uses its own tokenizer/template and feature bindings. These synthetic examples do not train a useful
Draft or establish quality/performance.

## 1. Install and choose a model

Install the separate Draft and Target environments using the [installation guide](installation.md).
For source setup, from the repository root:

```bash
bash setup.sh
bash setup.sh --target
```

In a Target terminal, start the service on one allocated GPU:

```bash
source .venv-target/bin/activate
CUDA_VISIBLE_DEVICES=0 aspec target
```

In a Draft terminal:

```bash
source .venv/bin/activate
```

| Profile | Default Target | GPU allocation |
| --- | --- | --- |
| `ling-tiny` | `inclusionAI/Ling-3.0-tiny` | 1 Target + 1 Draft |
| `qwen3-8b` | `Qwen/Qwen3-8B` | 1 Target + 1 Draft |

The selected public model is resolved through its provider. To reuse a local copy, add
`--target /absolute/path/to/model` to either profile. The model's own license applies.

## 2. Inspect the Plan

```bash
aspec quickstart ling-tiny --plan --output outputs/ling-tiny-001
aspec quickstart qwen3-8b --plan --output outputs/qwen3-8b-001
```

Plan inspection starts no GPU service, but may resolve model metadata. Expect two roles,
`REPLICATE`, CP1 and required NVLink. Ling retains its native Target capture path; Qwen uses its
configured direct-prefill path. Changing a model path is not an architecture substitution switch.

## 3. Run either profile

With Target on GPU 0, select GPU 1 for Draft. Each command needs a new output directory:

```bash
CUDA_VISIBLE_DEVICES=1 aspec quickstart ling-tiny --output outputs/ling-tiny-001
# Or:
CUDA_VISIBLE_DEVICES=1 aspec quickstart qwen3-8b --output outputs/qwen3-8b-001
```

The command performs GPU doctor, Plan inspection, four training steps with an eight-step LR horizon,
checkpoint publication, exact resume into a fresh directory through step eight, native export and
strict CPU loading. The result includes reports for training, restored checkpoint state and exported
assets. Cold model loading and JIT can take time before the first step.

No external train/eval files are required. For useful training, continue with a recipe and your own
independently prepared train/eval inputs.
`--steps` accepts an even horizon of at least two; the default is eight.

## 4. Inspect the result

For Ling (substitute the Qwen output directory for the other profile):

```bash
aspec --report outputs/ling-tiny-001/stage-1/resumed
aspec --report outputs/ling-tiny-001/stage-1/resumed --report-format json
```

| Output | Purpose |
| --- | --- |
| `quickstart-profile.yaml` | Exact model-specific smoke configuration |
| `stage-1/first/` | First half, Plan, Evidence, rank audits and complete checkpoint |
| `stage-1/resumed/` | Exact recovery, final checkpoint and audited Report |
| `stage-1/asset/` | Native Draft asset, already checked by the strict loader |
| `acceptance.json` | Profile, input identity, stages, success or failure |

An incomplete or failed result is not a pass. Preserve its logs and choose a fresh output for the
next attempt; the launcher does not overwrite prior checkpoints or clean other Ray clusters.

## Continue training

Use [Ling recipes](ling.md) or the [training recipes](recipes.md) with your own disjoint inputs.
[Training/recovery](training.md) explains normal training and segmented schedules. The separate
[evaluation/export guide](evaluation.md) covers serving formats, held-out quality and decode checks;
a native load alone does not verify downstream decoding. Check
[decode reproducibility](../support.md#decode-reproducibility) when evaluating a trained Draft.
