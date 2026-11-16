#!/usr/bin/env bash
# Copyright 2026 Ant Group Co., Ltd. Licensed under Apache-2.0.
# Run dense, indexer warmup and sparse training with native checkpoint export.
set -euo pipefail

if [[ $# -lt 1 || ${1:-} == --help || ${1:-} == -h ]]; then
  echo 'Usage: bash examples/qwen3-8b-dsa/run.sh NEW_OUTPUT_ROOT [KEY=VALUE ...]'
  echo 'Set ASPEC_TARGET_MODEL, ASPEC_TRAIN_DATA, ASPEC_EVAL_DATA and CUDA_VISIBLE_DEVICES.'
  echo 'Runs dense -> indexer warmup -> sparse, then exports a native asset.'
  exit 0
fi
: "${ASPEC_TARGET_MODEL:?Set an absolute local Qwen3-8B checkpoint path}"
: "${ASPEC_TRAIN_DATA:?Set an absolute training data path}"
: "${ASPEC_EVAL_DATA:?Set an absolute disjoint evaluation data path}"
: "${CUDA_VISIBLE_DEVICES:?Select the allocated Draft GPU explicitly}"

# Resolve the output before changing directory so paths are relative to the caller.
output_root=$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$1")
shift
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
mkdir -p -- "$(dirname -- "$output_root")"
mkdir -- "$output_root"
export RAY_ADDRESS="${RAY_ADDRESS:-local}"
config=configs/examples/qwen3_8b_dsa_dspark.yaml
previous=

for stage in dense indexer sparse; do
  continuation=false
  last_hidden=true
  if [[ $stage != dense ]]; then
    continuation=true
  fi
  if [[ $stage == indexer ]]; then
    last_hidden=false
  fi
  args=(--config "$config"
    "model.target_model_path=$ASPEC_TARGET_MODEL"
    "dataset.train_data_path=$ASPEC_TRAIN_DATA"
    "dataset.eval_data_path=$ASPEC_EVAL_DATA"
    dataset.eval_interval=4 "$@"
    "model.draft_model_config=aspec/config/dspark_qwen3_8b_dsa_${stage}_draft_config.json"
    "training.continual_training=$continuation"
    "training.continue_source_cursor=$continuation"
    "training.load_path=${previous:-null}"
    training.save_final_checkpoint=true
    "inference.store_last_hidden_states=$last_hidden"
    "output_dir=$output_root/$stage"
    "cache_dir=$output_root/cache")
  aspec "${args[@]}" --dry-run
  aspec "${args[@]}"
  aspec --report "$output_root/$stage"
  previous="$output_root/$stage/checkpoints"
done

# Export the final published checkpoint selected by the existing tracker.
iteration=$(cat "$previous/latest_checkpointed_iteration.txt")
printf -v checkpoint '%s/iter_%07d' "$previous" "$iteration"
aspec export --format aspec --input-dir "$checkpoint" --output-dir "$output_root/native-asset"
