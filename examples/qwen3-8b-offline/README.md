# Generate features once, train from disk

This source-checkout example uses one GPU in two sequential phases: Qwen3-8B
materializes DSpark features, then one Draft trains without a live Target engine.
It uses single-node CP1, no packing, and CPU Store/TCP transport. The eight-update
training window demonstrates evaluation, exact recovery and native export; it is
not a quality or performance recipe.

Activate the [isolated Linux environment](../../docs/user-guide/installation.md)
and run from the repository root. Set absolute paths to the **local** Qwen3-8B
checkpoint, disjoint train/eval conversations or prepared Parquet, and a new feature
directory. A small input subset is useful for a first run: generation saves **all**
admitted samples, regardless of the training step limit. Features can occupy much
more disk space than the source text.

For feature generation, keep `CUDA_VISIBLE_DEVICES=0 aspec target` running in the
separate Target environment. Run the commands below in Draft. Generation releases
its Target workers before the same GPU is reused for offline training.

```bash
export ASPEC_TARGET_MODEL=/path/to/Qwen3-8B
export ASPEC_TRAIN_DATA=/path/to/train.jsonl
export ASPEC_EVAL_DATA=/path/to/eval.jsonl
export ASPEC_OFFLINE_DATA=/path/to/new-qwen3-features
export CUDA_VISIBLE_DEVICES=0
export RAY_ADDRESS=local

bash tools/run_in_env.sh python -m aspec.offline.generate \
  --config configs/examples/qwen3_8b_offline_materialize.yaml \
  --output "$ASPEC_OFFLINE_DATA"
```

Use the [documented data formats](../../docs/user-guide/data.md). Each usable
training sample needs at least 14 supervised tokens, and sequences must fit the
4K limit. `materialization.json` records saved train/eval counts and cleanup;
`dataset.json`, `manifest.jsonl` and `samples/` form the replay dataset. An interrupted
generation is incomplete: preserve it and generate into a fresh directory.

## Train, then resume

Inspect the Plan after generation completes. It should select offline replay,
zero Target GPUs and one Draft GPU. The same GPU used for generation can now train.

```bash
aspec --config configs/examples/qwen3_8b_offline_replay.yaml --dry-run

aspec --config configs/examples/qwen3_8b_offline_replay.yaml \
  training.num_train_steps=4 output_dir=outputs/offline-first
aspec --report outputs/offline-first

aspec --config configs/examples/qwen3_8b_offline_replay.yaml \
  training.load_path=outputs/offline-first/checkpoints \
  training.continual_training=false training.continue_source_cursor=false \
  output_dir=outputs/offline-resumed
aspec --report outputs/offline-resumed

aspec export --format aspec \
  --input-dir outputs/offline-resumed/checkpoints/iter_0000009 \
  --output-dir outputs/offline-native-asset
```

Both segments retain `lr_total_steps=8`; the resumed run continues to total step 8.
Checkpoint iteration names are one ahead of completed optimizer steps. Choose fresh
output/export directories on subsequent attempts. Evaluation reads the recorded
eval split rather than starting a Target engine.

Keep the Target checkpoint available: replay still loads frozen embedding/head
weights. Target weights, tokenizer, capture layers, normalization, dtype and dataset
identity must match generation. Changing preprocessing requires new features.
Architectures with compatible feature contracts may reuse this dataset in separate
runs; changing the feature contract requires materializing it again.

See [training](../../docs/user-guide/training.md#offline-target-features) and
[export boundaries](../../docs/user-guide/evaluation.md).
