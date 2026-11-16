# Qwen3-8B DSA: three dependent training stages

Run the existing full DSA Draft configuration through dense training, indexer
warmup and sparse training. Each stage uses a new optimizer/scheduler and output
directory, initializes from the preceding complete checkpoint, and retains source
progress.

Use a source checkout with the [isolated Linux environment](../../docs/user-guide/installation.md)
activated. The default allocation is one Target GPU and one Draft GPU on a node
with NVLink, using FA4 packing at CP1. Supply disjoint train/eval inputs prepared
for Qwen3-8B, with sequences up to 4K and at least 32 supervised tokens per usable
sample. See [data preparation](../../docs/user-guide/data.md).

In the Target environment, keep `CUDA_VISIBLE_DEVICES=0 aspec target` running.
Run the following commands in Draft's environment on GPU 1:

```bash
export ASPEC_TARGET_MODEL=/path/to/Qwen3-8B
export ASPEC_TRAIN_DATA=/path/to/train.parquet
export ASPEC_EVAL_DATA=/path/to/eval.parquet
export CUDA_VISIBLE_DEVICES=1
export RAY_ADDRESS=local

bash examples/qwen3-8b-dsa/run.sh outputs/dsa-workflow
```

The script uses [the existing DSA YAML](../../configs/examples/qwen3_8b_dsa_dspark.yaml)
and three model JSONs. It checks each Plan, trains 16 updates per stage, evaluates
every four updates, saves checkpoints and exports the final sparse Draft as a native
asset. These short defaults demonstrate the workflow, not convergence or sparse
attention speed. Pass ordinary overrides after the output root, for example
`training.num_train_steps=8 training.lr_total_steps=8` for eight updates per stage.
The same overrides apply to all stages; the script fixes stage model, checkpoint,
source-cursor policy, capture requirements and output paths afterward.

The output root must be new. Failed attempts remain available for diagnosis; the
script stops on the first failed command and never skips an existing stage.
The generated tree contains `dense/`, `indexer/`, `sparse/` and `native-asset/`.

| Stage | Trainable state | Initialization |
| --- | --- | --- |
| Dense | Main Draft; indexer frozen | Fresh model |
| Indexer warmup | Indexer; main Draft frozen | Dense checkpoint, weights-only |
| Sparse | Main Draft and indexer | Indexer checkpoint, weights-only |

Warmup does not request last hidden states needed by the normal DSpark objective;
the sparse stage requests them again. Warmup evaluation reports indexer KL rather than
a directly comparable language-model CE. Stage changes are intentional new phases,
not exact checkpoint recovery.

## Resume within a stage

If training was interrupted after a complete checkpoint, use that stage's saved
`config.yaml`. Clear **both** phase-transition flags for exact resume and select a
fresh output directory. Keep the original total step limit and LR horizon.

Recovery restores checkpoint state and the committed source frontier. Asynchronous
packing can regroup uncommitted samples after restart, so a later optimizer-step
count can correspond to a different document count and numerical trajectory than
an uninterrupted run. Check restored state and source continuity separately from
later trajectory equivalence.

```bash
aspec --config outputs/dsa-workflow/indexer/config.yaml \
  training.load_path=outputs/dsa-workflow/indexer/checkpoints \
  training.continual_training=false training.continue_source_cursor=false \
  output_dir=outputs/dsa-indexer-resumed
aspec --report outputs/dsa-indexer-resumed
```

After that stage finishes, start the sparse phase from its recovered checkpoint:

```bash
aspec --config outputs/dsa-indexer-resumed/config.yaml \
  model.draft_model_config=aspec/config/dspark_qwen3_8b_dsa_sparse_draft_config.json \
  training.load_path=outputs/dsa-indexer-resumed/checkpoints \
  training.continual_training=true training.continue_source_cursor=true \
  inference.store_last_hidden_states=true output_dir=outputs/dsa-sparse-after-resume
```

This uses the same configured horizon for the new stage. Review each Report's
actual progress, required capabilities and lifecycle. DSA assets use
`aspec export --format aspec`; strict native loading is separate from downstream
serving integration. See [DSA semantics](../../docs/user-guide/structures.md#dsa-token-level-sparse-attention)
and [recovery](../../docs/user-guide/training.md).
