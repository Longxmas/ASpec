# Static multi-node training

The node-local recipe uses **independent local storage**, not a cross-node shared filesystem. It places
4 Target + 4 Draft GPUs on each of two compatible eight-GPU nodes. Features remain on their production
node over NVLink; Draft gradients synchronize across nodes. Use a dedicated Ray cluster on a trusted
private network. Shared multi-Draft execution remains single-node.
Unused-output disposal follows the same locality rule: the runtime selects a consumer on the
producer's node and rejects missing or unmatched placement metadata.

## Prepare each node independently

Install the same source revision, isolated runtime and backend patch on each node. Activate that node's
venv, enter its checkout and use `tools/run_in_env.sh` when starting Ray. Source and venv installation
locations can differ: multi-node actor environment forwarding does not replace worker-local library,
Python import, model-download cache or compilation-cache paths with the driver's paths.

The current single-config interface requires model, data and Draft JSON inputs at the **same absolute
logical paths on every node**. These may be independent local disks. It also uses one absolute output
path interpreted separately on each node. It does not yet map different per-node input path strings;
use explicitly prepared directories or container mounts to provide a common logical layout.
This constraint is separate from sharing file contents or mounts.

You can copy and checksum inputs without provisioning NFS:

```bash
# Run from an installed checkout with SSH access. Repeat for model, train/eval and Draft JSON.
bash tools/run_in_env.sh python -m tools.stage_recipe_inputs \
  --source /downloads/Qwen3-8B --destination /data/aspec/Qwen3-8B \
  --host node-a --host node-b
# Existing copies can be checked without changing them:
bash tools/run_in_env.sh python -m tools.stage_recipe_inputs \
  --source /downloads/Qwen3-8B --destination /data/aspec/Qwen3-8B \
  --host node-a --host node-b --verify-only
```

Host aliases and paths above are examples. The helper uses SSH/tar and SHA256, refuses existing copy
destinations, and checks subprocess exit codes. `--ssh-config` selects a custom SSH configuration.
It neither installs software nor starts clusters. It dereferences model-cache symlinks when copying.
Source files must remain unchanged during transfer. An interrupted copy leaves a named staging
folder for inspection; it is never silently reported as successful.

## Start an owned cluster

Choose routable node addresses and a free Ray port; do not assume `eth0`, a particular RDMA device or
port 6379. Keep the head port outside Ray's configured worker-port range (the pinned runtime defaults
to 10002–19999). Run each command from that node's activated venv and checkout in a persistent terminal
or supervised process; `--block` keeps the service attached to that session:

```bash
# Head node
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/start_recipe_node.sh --head --node-ip-address="$HEAD_IP" --port="$RAY_PORT"
# Worker node
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/start_recipe_node.sh --address="$HEAD_IP:$RAY_PORT" --node-ip-address="$WORKER_IP"
```

In each node's separate Target environment, start
`CUDA_VISIBLE_DEVICES=4,5,6,7 aspec target --address http://NODE_IP:30000`.
Set `inference.target_service.addresses` in the training config to map each planned
Target node IP to that service URL. Draft's Ray cluster reserves only GPUs 0–3;
each Target service leases its own GPUs 4–7.

Before starting either node, provision the same private Ray authentication token in a local file on
each node and set `RAY_AUTH_MODE=token` and `RAY_AUTH_TOKEN_PATH` there. Set both in the driver's shell
as well: attaching clients must explicitly enable token mode, not only select a token file. Token paths can
differ across nodes. Keep token contents out of source control and restrict endpoints to trusted
members; see [security](../../SECURITY.md).
The startup helper checks the token file, enables token mode and runs Ray in the foreground through
the environment wrapper. It does not generate/distribute secrets or stop existing clusters.
Native library/compat paths and local JIT caches are set on **each node before starting Ray**. The
runtime verifies package/source/backend identities; copying a driver's venv path is not installation.
GPU indices must refer to allocated devices. Do not attach to another task's cluster or run host-wide cleanup.

## Configure, preflight and train

```bash
export ASPEC_NODE_0_IP="$HEAD_IP" ASPEC_NODE_1_IP="$WORKER_IP"
export ASPEC_TARGET_MODEL=/data/aspec/Qwen3-8B
export ASPEC_DRAFT_CONFIG=/data/aspec/dspark_draft_config.json
export ASPEC_TRAIN_DATA=/data/aspec/train.parquet
export ASPEC_EVAL_DATA=/data/aspec/eval.parquet
export ASPEC_OUTPUT_DIR=/data/aspec/runs/run-001
export ASPEC_CACHE_DIR=/data/aspec/cache
export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"
bash tools/run_in_env.sh python -m tools.recipes plan multinode
bash tools/run_in_env.sh python -m tools.recipes train multinode --output "$ASPEC_OUTPUT_DIR"
```

The training entry runs actual cluster preflight before model initialization. For a separate
`aspec --preflight-only` invocation, choose a separate output directory; its report is a run artifact.
Preflight verifies node membership, source/runtime/input content identities, local writability,
placement and node-local CUDA-to-CUDA NVLink transfer. Large local model replicas are content-hashed;
allow appropriate I/O time with `cluster.preflight_timeout_seconds`.

NCCL/Gloo select interfaces from the live environment by default. If explicit selection is needed,
configure each node before Ray startup, or use `training.train_env_vars` for an intentional cluster-wide
override. NCCL's `=INTERFACE` syntax requests an exact interface match; the interface name itself is
user-provided. Do not paste a machine-specific interface into a public default.

## Checkpoints and recovery without a shared mount

Each rank writes DCP shards and its RNG state locally. After all ranks finish, ASpec collects the
shards to the coordinator in bounded chunks and verifies SHA256. The coordinator publishes the complete
checkpoint only after attaching committed-work state. Missing, truncated or conflicting shards fail;
unpublished staging is not a recovery point. Successful publication permits remote staging cleanup.

For exact resume, the coordinator must have the complete checkpoint. ASpec explicitly replicates
that checkpoint into the new run's local `resume-input` on each node before loading it. It preserves
the logical topology and committed-work frontier; no live reshaping or elastic recovery is implied.
The coordinator's output disk is therefore the durable source for checkpoints, Plan, Evidence and
reports. Back it up as needed; the framework does not claim disaster recovery if that disk is lost.
Worker caches and logs are local. Validation rank audits are returned to the coordinator explicitly.

Use [runtime validation](../contributing/validation.md) for train/resume/native export, slow-consumer checks
and bounded trainer-exception checks. The [whole-node loss procedure](../contributing/validation.md#whole-worker-node-loss-and-recovery)
arms an external fault after a complete checkpoint and audits fencing and exact recovery after the
worker rejoins. Run these checks on the intended cluster; single-host file exchange does not
exercise cross-node failure or recovery.

Forced cleanup confirms read-process exit on live nodes before retiring old-attempt output ownership.
For a lost node-local fail-stop domain, Evidence records domain loss separately from live-process
exit and leaves that node's physical memory reclamation unobserved. Verify infrastructure/process
fencing externally before rejoining it. Cross-node memory access and arbitrary partitions/driver
hangs do not inherit this node-local loss assumption.

## Optional shared storage and other topologies

The [role-separated TCP example](../../configs/examples/qwen3_8b_4k_multinode_tcp.yaml) retains `cluster.storage_mode=shared`
(the compatibility default) and requires `cluster.shared_root`. That optional mode checks shared marker
visibility. The formal node-local recipe explicitly selects `cluster.storage_mode=node_local` and does
not require or use a shared root. Host-staged TCP, registered-CUDA TCP and RDMA/GPU Direct are distinct
capabilities; do not silently exchange them to rescue a failed run.
