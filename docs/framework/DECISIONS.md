# Current engineering decisions

These decisions describe the maintained implementation and its public behavior.

- **Product scope.** SGLang with DSpark/DFlash and Qwen3 is the primary multi-GPU performance path.
  Other implemented algorithms and structures have explicit [support boundaries](../support.md).
  A source release does not promise all combinations or production fault certification.
- **Naming and licensing.** The project uses ASpec; the Python namespace and CLI use `aspec`.
  The distribution name on PyPI is `ling-aspec`; the unrelated `aspec` distribution must not be
  installed alongside it because their import namespaces can overlap. ASpec uses Apache-2.0;
  inherited material retains its own licenses and notices. PyPI publication requires the
  [release checks and publishing configuration](../releasing.md).
- **Release versions.** Maintainers choose the major/minor/patch version and currently retain
  the `b0` beta suffix for public releases.
  CI validates the matching `v<version>` tag without incrementing or promoting the version.
  See [versioning](../releasing.md#versioning).
- **Reuse.** Review the pinned upstream implementation before adding overlapping functionality.
  Adapt working code by default; explain a rewrite and verify it against the original behavior.
  Preserve transitive provenance. See [retention boundaries](RETENTION_BOUNDARY.md).
- **Runtime baseline.** Use the [pinned installation lane](../user-guide/installation.md): official
  SGLang v0.5.21, commit `e00930c5489053f26d86b179cee0d087f846acbb`, with the retained training
  patch in Target. Draft uses FA4 b34, CUTLASS DSL 4.7.1 and Quack 0.6.5 independently. Upstream
  migration references do not constrain ASpec to their older runtime versions.
- **Installation.** `ling-aspec` and `ling-aspec-target` install separate Draft/Target dependency
  environments, sharing one version of `ling-aspec-runtime`. Target follows upstream SGLang pins;
  no metadata overrides are accepted. `aspec patch-sglang` only patches files. `aspec target` and
  `aspec train` start in their own environments and connect through a versioned JSON control
  protocol; feature tensors retain Mooncake transport. Target's private Ray runtime orchestrates
  its source processes and leases its own CUDA-visible GPUs; Draft's Ray reserves only training GPUs.
  Compatibility requires the same release and control protocol; source hashes and Python versions
  are Evidence rather than handshake gates. Optional single-node
  managed-local supervision starts the same service, with an explicit Target interpreter only for
  that supervisor. Both forms retain exact resume. This boundary is inspired by SpecForge
  f8cc3dc external/managed-local capture; its newer SGLang plugin is not copied.
- **Semantics and execution.** Workload, capability, Plan and observed Evidence are distinct.
  Required capability failures are errors; fallback/omission is explicit. No legacy fields leak
  across the canonical boundary. [Architecture](ARCHITECTURE.md) defines the work/ownership invariants.
- **Capacity and topology.** Do not collapse memory or buffers into one number. Small Drafts use
  `REPLICATE` by default; sharding requires a capacity reason. Target TP is an ordered capacity
  fallback. Automated selection uses expert-defined capabilities, measured service profiles and
  end-to-end confirmation, without changing the workload.
- **Packing.** Preserve complete documents, local positions, supervision and committed identity.
  FA4 packed execution retains fixed THD for full attention and selected-key varlen for sparse
  attention. DSpark/DFlash full and sliding attention can instead use native cuDNN THD at CP1 and
  above; its interval metadata preserves documents without dense masks or per-layer regrouping.
  Plan/Evidence records the effective backend and mode. CP degrees follow actual resource/geometry
  constraints rather than a 1/2/4 whitelist. Finite-source tails are dropped explicitly.
- **Runs and recovery.** Real runs require a writable output directory and reject conflicting
  ASpec-managed artifacts. Exact resume uses a new run directory and a complete checkpoint.
  Unrelated files do not force rejection; implicit overwrite is unsupported.
- **Sharing and clusters.** Compatible Target capture can serve independently progressing Drafts.
  Shared mode is single-node. Multi-node single-Draft mode uses static homogeneous membership,
  node-local model parallelism and fail-stop attempt recovery; node-local feature affinity is a
  required capability, with no remote feature substitution.
- **Algorithms and assets.** EAGLE3/4, DFlash2 and native MTP retain their own objectives and
  state identities. CSA/QSA/DSA stages use explicit weights-only transitions. DSA uses per-query raw-token
  selection and a detached, separately supervised indexer, not QSA block-shared indexing. EAGLE4 candidate training
  freezes the complete Draft and head, uses the documented Top-K forward-KL objective and delivers
  native assets; it does not claim an unpublished official training recipe. Native MTP preserves
  the original Qwen3.5 predictor and same-Target frozen bindings.
- **Serving boundary.** Maintain faithful export formats, strict loaders and proposal components.
  SGLang serving scheduling, deployment and performance belong to the downstream runtime.
- **Measurement.** Compare normally enabled complete workflows with reasonable buffers and
  interconnects. The public DSpark matrix compares TorchSpec, SpecForge and AngelSpec;
  AngelSpec Full is a migration/regression reference. Other studies identify their own boundaries. Compare
  matching GPU model/workload/statistics, disclose hosts and software differences, and never infer
  quality or serving speed from training input throughput.

Public behavior changes require corresponding guide/config updates and executable verification.

- **Training recipes.** Maintain the [recipe catalog](../user-guide/recipes.md), including
  Ling tiny/flash 3.0/flash 3.1, Native MTP, EAGLE3/4, one Qwen3.8-27B long-context CP2 lane and one
  two-node node-local lane. Separate full training defaults, bounded validation overrides and
  supplementary examples. Validate training, recovery and asset behavior when changing a recipe.

- **Multi-node portability.** The release recipe requires no shared filesystem. `cluster.storage_mode=node_local`
  checks input replicas and exchanges checkpoint shards explicitly; `shared` retains the older optional
  shared-root contract. Logical input/output path strings currently match across nodes, while source/venv
  locations may differ. Worker-local library/cache environments are not replaced by driver paths.

- **Failed-attempt ownership.** Forced single-Draft cleanup observes read-process exit on live nodes
  before retiring remaining output ownership. Declared node-local fail-stop domain loss is recorded
  separately, including unobserved physical memory on the lost node. Committed work remains intact;
  late output admission is rejected after fencing. External infrastructure/process fencing and a new
  node/attempt identity precede recovery. This does not certify arbitrary partitions or driver hangs.
