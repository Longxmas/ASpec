# Validating recipes and runtime changes

Use the [recipe catalog](../user-guide/recipes.md) and an isolated checkout/venv. These contributor
checks cover configuration, training, recovery, assets and faults. Choose the checks relevant to
your change and retain the commands, inputs, reports and failure logs.
The launcher never starts a host-wide cleanup or stops another task's Ray cluster.

## Offline configuration and regression checks

```bash
bash tools/install.sh --dev
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  bash tools/run_in_env.sh python -m tools.recipes contracts
python -m tools.check_docs
python -m ruff check aspec tests tools
```

`contracts` checks all 14 configs/13 workflows, resource budgets and training defaults, negative
acceptance cases, and existing algorithm/packing/CP/sharing/multi-node contracts. Model names, data
paths and node IPs in CPU tests are fixtures, never real GPU inputs. No models are downloaded by this
command. CPU contracts do not establish GPU kernel parity or available capacity.

The CLI source-retention regression starts its own CPU-only Ray instance and checks real actor
processes and interpreter exit. Run it separately from the default offline suite:

```bash
CUDA_VISIBLE_DEVICES='' ASPEC_RUN_RAY_INTEGRATION=1 \
  bash tools/run_in_env.sh python -m pytest -q tests/test_source_fence_cli.py
```

This verifies driver retention until reader-process fencing, including Ray's automatic exit
cleanup. It does not certify unreturning CUDA/RDMA operations, forced driver loss or host partitions.

## Bounded GPU training, recovery and native assets

Activate the installed venv, use the documented environment wrapper, and select only GPUs allocated
to this task. Start Target separately as described in [installation](../user-guide/installation.md)
and select only the catalog's Draft GPUs in the training shell. Managed-local instead takes the
combined Target/Draft list. For example, with Target already running on GPU 0:

```bash
export ASPEC_TARGET_MODEL=/models/Qwen3-8B
CUDA_VISIBLE_DEVICES=1 bash tools/run_in_env.sh python -m tools.recipes accept quickstart \
  --output outputs/acceptance/quickstart-001
```

For every other entry, first set absolute `ASPEC_TRAIN_DATA` and `ASPEC_EVAL_DATA` to disjoint,
model-compatible inputs. Select its catalog GPU allocation, then substitute the recipe ID. The output
must be new. `--steps` defaults to 8 and must be even; increase it if required pipeline/evaluation
observations cannot be established. Incomplete evidence fails rather than becoming a healthy run.

The command performs:

1. Input identity checks and GPU/dependency/patch doctor; log failures.
2. CPU Plan, plus actual cluster preflight for multi-node.
3. Half-horizon training with a fixed full LR horizon, accumulation 1, periodic evaluation and checkpoint.
4. Exact resume into a fresh directory to the full horizon. Shared Drafts each restore their own checkpoint.
5. Canonical Report checks: completed/healthy, expected step, finite observed metrics, no fallback,
   no pending required capability and zero resident output/byte ownership.
6. Per-rank trainer audits: parameters changed; resumed model/master/optimizer state, LR scheduler
   and RNG match the saved checkpoint. Model, master weights and both optimizer moments must each
   have a nonempty exact comparison; missing state or zero compared coordinates fails acceptance.
7. Native export and strict CPU loading of each resumed Draft. EAGLE4 automatically passes the first
   stage's native asset into the candidate stage and repeats the workflow; its audit also checks frozen
   backbone weights.

`acceptance.json` records this **bounded functional scope**, stages, inputs and failures. Each stage
retains Plan, Evidence, reports, trainer audits and subprocess logs. A passed native roundtrip is not
SGLang serving-format/decode certification, full uninterrupted-versus-resumed trajectory parity, or
proof that every OS shared-memory resource was reclaimed.

For normal-training checkpoints, the same strict native export is available separately:

```bash
bash tools/run_in_env.sh python -m tools.recipes export \
  --checkpoint outputs/my-run/checkpoints/iter_0000101 --output outputs/my-native-asset
```

## Algorithm, capacity and fault checks

| Scope | Required additional evidence |
| --- | --- |
| DFlash / EAGLE3 / Native MTP | Algorithm contract tests plus the actual recipe's train/resume/assets; Native MTP initialization must use the native predictor |
| MTP-conditioned DSpark / DFlash | Native timing/loss/frozen-state tests, real train/resume/native assets and conditional-suffix observations; no full serving acceptance claim |
| EAGLE4 | Both stage reports and candidate frozen-body audit; stage-local resume; no official candidate-recipe claim |
| Ling | Correct model tokenizer/template and feature layers; use Ling-prepared data; MoE routing/bias state is retained |
| 27B CP2 | Genuine long supervised inputs, actual long work in runtime metrics, GPU CP loss/gradient parity and measured memory fit |
| Shared Target | Independent child reports/checkpoints plus slow-reader, cancellation/failure/fencing acceptance |
| Multi-node | Real two-node preflight, node-local transport evidence, cross-node synchronization, bounded fault exit and restart |
| Quality / serving | Held-out learning and repeated decode checks; retain the [known default-inference limitation](../support.md#decode-reproducibility) |

The 27B bounded acceptance check (`accept qwen27b-long`) rejects a training Parquet containing
any row shorter than 261,888 tokens. It also requires observed 261,888-token sequences and packed rows, positive dispatched work, and actual
CP2 execution with the configured attention backend in both phases (cuDNN by default).
A different observed backend fails the check.
It records device memory every second in `gpu-memory.csv` and
per-device peaks/capacities in `acceptance.json`; missing observations fail acceptance. Sampled
device-wide memory is distinct from an allocator peak and can include other processes, so use
exclusive allocated GPUs. Synthetic inputs can establish capacity, but cannot establish quality.
Ordinary `train qwen27b-long` runs may use shorter valid samples; the minimum length is an
acceptance requirement, not a training input constraint.

To audit an existing train/resume/export receipt and its saved memory samples:

```bash
bash tools/run_in_env.sh python -m tools.validation.recipe_capacity \
  --output outputs/acceptance/qwen27b-long-001 \
  --memory-csv outputs/acceptance/qwen27b-long-001/gpu-memory.csv
```

GPU CP numerical tests can be invoked independently on allocated GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash tools/run_in_env.sh python -m tools.recipes cp-check \
  --output outputs/acceptance/cp2-001
```

A skipped CUDA test is unverified; `cp-check` launches the two ranks and rejects skipped/empty tests.
It checks the CP2 FA4 operator, not full 256K model quality or capacity.

Shared-reader and multi-node trainer exceptions have a scripted entry:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/run_in_env.sh \
  python -m tools.recipes fault-check shared-target --action fail --output outputs/acceptance/shared-fail-001
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/run_in_env.sh \
  python -m tools.recipes fault-check shared-target --action slow --output outputs/acceptance/shared-slow-001
# With the dedicated cluster prepared, select a fresh node-local output path:
bash tools/run_in_env.sh python -m tools.recipes fault-check multinode --action fail \
  --output "$ASPEC_OUTPUT_DIR"
```

The scripts verify that injection actually occurred, the expected exit status, canonical terminal
state and empty terminal ownership. These examples use an independently started Target;
`fault-check --set inference.target_service.address=http://host:port` selects its address.
Slow-reader mode requires healthy completion. A startup failure
cannot pass as fault acceptance. This scoped injection is a Python trainer exception/slow consumer,
not loss of a physical node or an unreturning native/CUDA call. Test those mechanisms separately.
Use the last complete checkpoint to resume into a fresh output after failure, then run the
same report/checkpoint audit; an ordinary healthy resume does not close node-loss acceptance.

## Multi-node

Prepare the dedicated cluster and replicated inputs using [multi-node setup](../user-guide/multinode.md).
No shared filesystem is required. Configure `ASPEC_NODE_0_IP`, `ASPEC_NODE_1_IP`, model/data/Draft
paths and an explicit `RAY_ADDRESS`. Run with a fresh absolute output path:

```bash
bash tools/run_in_env.sh python -m tools.recipes accept multinode \
  --output /data/aspec/acceptance/multinode-001
```

The launcher attaches to the explicit address, preflights each node, and disconnects its client on
exit; it does not stop the cluster. Input replicas are hashed, checkpoint shards are explicitly
collected, and resume state is explicitly distributed. This test needs **two compatible nodes,
eight GPUs each**; it cannot be replaced by single-node file-exchange tests.

## Whole worker-node loss and recovery

Use a dedicated cluster and an external infrastructure controller. Keep the coordinator and its
checkpoint disk available. Run the fault phase on the coordinator with the same node-local input
environment as the healthy multi-node check:

```bash
export NODE_LOSS_OUTPUT=/data/aspec/acceptance/node-loss-001
bash tools/run_in_env.sh python -m tools.validation.recipe_node_loss fault \
  --output "$NODE_LOSS_OUTPUT" --worker-ip "$WORKER_IP" --cluster-address "$RAY_ADDRESS"
```

Wait for `first/node-loss-armed.json` under that output. It appears only after the complete step-4
checkpoint is published. Then remove the selected whole worker through your controller: for a
process-level node-loss test, abruptly stop that task's complete worker Ray/GPU process tree; for a
power-loss test, use the infrastructure's host power controls. Never target another task's cluster.
The script does not execute a host shutdown or assume SSH credentials, shared mounts or host aliases.

The fault audit requires the old Ray node to be observed dead, a failed old attempt with completed
owner fencing, zero resident ownership, a valid checkpoint, and exit within 300 seconds of arming.
Cleanup records the live-node read-process exit checks and lost node IDs in `read_owner_fence`.
Missing or incomplete lifecycle counters still require read-owner fencing before source teardown;
an unreadable ownership snapshot cannot be treated as empty ownership.
Lost-domain retirement is limited to the declared node-local fail-stop topology. Its
`physical_memory_unobserved_nodes` remains explicit: terminal ledger counters do not certify memory
reclamation on an unreachable host, a network partition or a stuck CUDA/driver operation.
Also verify GPU processes and memory on each surviving/reachable host externally; metadata fencing
alone does not prove physical CUDA-resource reclamation. Preserve the controller's injection receipt.

After the worker returns, start it again with the same isolated installation and input replicas, then:

```bash
bash tools/run_in_env.sh python -m tools.validation.recipe_node_loss resume \
  --output "$NODE_LOSS_OUTPUT" --worker-ip "$WORKER_IP" --cluster-address "$RAY_ADDRESS"
```

Recovery requires a new worker node identity and a new run/attempt identity, then audits all eight
trainer ranks, committed-work preservation, exact model/optimizer/RNG restore, evaluation and native
export/loading through step 8. Receipts are `node-loss-fault.json` and `node-loss-resume.json`.
Whole-process node loss, physical power loss and an unreturning native/driver call are distinct
fault mechanisms; record which one was actually exercised.
