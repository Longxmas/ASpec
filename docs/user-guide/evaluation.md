# Evaluation, selection and export

Training health, held-out learning, asset compatibility and decode usefulness are separate questions.

Run serving-based `aspec evaluate` in the Target environment.
Checkpoint export runs in Draft. Held-out training evaluation uses the configured Target workers.

## Held-out evaluation

Set `dataset.eval_data_path` and `dataset.eval_interval`. Target observations are cached with
data/model/preprocessing/software/layout identity; mismatched or partial caches are not reused.
Eval does not advance training RNG or commit training work.

`evaluation.json` stores the curve. JSON Report exposes `training.evaluation` and the best checkpoint
**actually saved and still available**; shared sessions expose this per experiment. Retention protects
best assets in addition to the configured latest checkpoints. Do not assume the latest is best.

`eval/avg_loss` monitors **cross-entropy loss (CE)** against reference tokens.
For DSpark, **total loss** is `ce_alpha × CE + l1_alpha × L1 + confidence_alpha × confidence BCE`
(default coefficients: 0.1, 0.9, 1.0). L1 matches Draft/Target probability distributions;
confidence BCE trains the confidence head against analytical acceptance probabilities.
CE is one component of total loss; `avg_loss` does not include the other two components.
The L1 term is the raw distance, not half the distance used in the acceptance formula below.
`eval/simulated_acc_len` measures proxy **extra Draft tokens**, not real verifier acceptance.
Different auxiliary objectives are not directly sortable by one loss number.

DSpark also reports `train/accept_rate@0` through the last block position and
`train/tau_probabilistic`, with corresponding `eval/` metrics. `accept_rate@i` is the
**analytical acceptance rate (distribution overlap)**
`alpha = sum_v min(draft_probs(v), target_probs(v)) = 1 - L1(draft_probs, target_probs) / 2`.
It ranges from zero to one (or 0–100% when displayed as a percentage); higher is better.
It is neither a measured decode acceptance rate, top-1 accuracy nor a predicted confidence score.
Position zero is the first Draft prediction; the Target bonus appears only in tau.
`tau_probabilistic` is **estimated accepted length**. For each supervised block, it is
`1 + sum(cumprod(alpha * eval_mask))`; prefix products are computed before averaging blocks.
The extra one is a Target token. Counts are pooled across microbatches and ranks before
division; entirely masked blocks are excluded. These metrics describe prediction on reference
contexts. Use real decoding below to measure generation behavior.

`acc_i` measures hard-label prediction accuracy. `simulated_acc_len` multiplies average
position accuracies to estimate extra Draft tokens; it excludes the Target token and differs
from `tau_probabilistic`, which pools per-block probability products.

## Export

In the active ASpec venv, use a new directory:

```bash
aspec export --input-dir /path/to/best-checkpoint \
  --output-dir outputs/exported-draft
```

Default export targets the declared SGLang format, with weights/config and `export_manifest.json`
binding algorithm, vocabulary, structure, source and hashes. Missing/unexpected weights fail.
Frozen Target tensor bindings remain part of the contract; partial loader success is not compatibility.

CSA/QSA and other native paths that cannot faithfully use official loaders require `--format aspec`
and their strict loader/proposal component, not a renamed SGLang model. Native MTP can be stitched
back with `aspec stitch-mtp`; see [algorithm contracts](algorithms.md).
These components are not a turnkey serving runtime or arbitrary Target-cache compatibility guarantee.

## Real decode

```bash
CUDA_VISIBLE_DEVICES=0 aspec evaluate \
  --target /path/to/Qwen3-8B --draft outputs/exported-draft \
  --data /path/to/target-generated/eval.jsonl --output outputs/decode.json
```

Defaults: 32 held-out prefixes, greedy, at most 256 new tokens, three complete repeats; one warmup
per engine and startup excluded. Normal SGLang settings including radix cache remain enabled.
For a Target checkpoint that requires custom code, explicitly add `--trust-remote-code` after
reviewing that checkpoint's code. Decode and selected-answer preparation default to no remote-code trust.
**Mean accepted length (τ_decode)** is total completion tokens / real verifier calls,
**including Target tokens**, in **tokens/verification**. It is measured during generation;
`tau_probabilistic` instead estimates length from reference-context distributions.
An acceptance **rate** needs its own token-attempt denominator. For example, a measured
position-wise conditional rate counts accepted tokens at that position over attempts where
all preceding Draft tokens were accepted. `accept_rate@i` from training does not measure this.
These names follow [DSpark](https://arxiv.org/html/2607.05147v1). Match sampling settings,
data and token-count conventions when comparing results with another evaluator.
Timing is bounded serial decode, not serving throughput.
DSpark's trained `block_size` is its Draft prediction count (gamma); the evaluator requests
`gamma + 1` SGLang verification tokens, including the Target token. Export preserves the trained
block size from checkpoint metadata; legacy checkpoints without it need an explicit training config.

Use exact saved prompt IDs. Rendering non-thinking raw conversations requires
`--disable-thinking`. Nonidentical greedy outputs cause nonzero exit even when a JSON result
was saved; check the verdict and exit status.
For a controlled quality comparison, `--deterministic-inference` enables the official SGLang option
on both engines while retaining normal radix cache. It is opt-in, not a changed serving default;
the result records effective settings and separately checks cross-repeat stability.

Default-inference repeatability for the Qwen3-8B recipe is not established; see
[decode reproducibility](../support.md#decode-reproducibility). Check all repeats before comparing timing.
The [DSpark quality comparison](../benchmarks/dspark-alignment.md) uses one fixed sequence for
TorchSpec, ASpec BSHD and ASpec THD/CP1, with matching optimizer documents,
per-document anchors, initialization and LR. A corpus and seed alone do not fix an asynchronous
optimizer sequence. The report presents the training curves and measured mean accepted length
under these matched inputs.
