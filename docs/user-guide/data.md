# Training data

Target generation, training rows and decode prefixes have different boundaries. Bundled synthetic
data is a smoke fixture, not a quality recipe.

## Conversations

One JSON object per JSONL line:

```json
{"conversations":[{"role":"user","content":"Explain binary search."},{"role":"assistant","content":"Binary search repeatedly halves a sorted search interval until the requested element is found or the interval is empty."}]}
```

Set `dataset.prompt_key=conversations` and the correct Target template. Do not silently switch
thinking mode between generation, training and decode. Loss masks supervise intended assistant tokens;
short unusable rows are excluded and counted. DSpark block 7 needs at least 14 supervised tokens.

Use disjoint train/eval data and preserve identity across resume. Evidence distinguishes raw, processed,
admitted, skipped and DP-truncated counts. Epoch-scoped work count is not unique-document count.

## Target-generated data

The Qwen3-8B quality recipe generates answers from the actual Target, saving exact non-thinking prompt
IDs, generated IDs, supervision and a resumable manifest. Original assistant answers are not input.

Prepare the input rows in Draft.

```bash
source .venv/bin/activate
aspec prepare-data --output-dir data/raw
```

Generate responses in Target.

```bash
source .venv-target/bin/activate
aspec prepare-target-data --target /path/to/Qwen3-8B \
  --train-data data/raw/open_perfectblend_train_65536.jsonl \
  --eval-data data/raw/open_perfectblend_eval_1024.jsonl \
  --output-dir data/target-generated --dp 8
```

Convert the generated rows in Draft.

```bash
source .venv/bin/activate
aspec convert-data \
  data/target-generated/train.jsonl data/target-generated/train.parquet --batch-rows 256
aspec convert-data \
  data/target-generated/eval.jsonl data/target-generated/eval.parquet --batch-rows 256
```

This is an expensive preparation job, not required for quickstart. Review model/dataset licenses and
access first. `--dp` is the generation GPU budget, separate from training's Target/Draft split.
Resume with the same command plus `--resume`; input/model/software/settings and completed journal are
validated. A seed is not bitwise reproduction across batch shapes, versions or GPUs.

Use typed Parquet for pretokenized training/eval. Raw generation JSONL is an audit journal and decode
prefix source, not automatically a recognized training schema. Conversion preserves masks and real EOS;
it does not re-tokenize or append fake EOS to a length-truncated answer.

Decode reads `prompt_input_ids`; conversation input is rendered only up to the user prefix. Held-out
answers must not be prefilled. Training tokens without a prompt boundary are rejected for decode.
Packing preserves document boundaries, coordinates, supervision and work identity while changing grouping.


Local directory cache identities include the ordered shard paths and file contents. Editing a shard or adding/removing an admitted JSON, JSONL, Parquet or Arrow shard invalidates the tokenized cache. For saved hidden states, use the [offline feature workflow](training.md#offline-target-features).
