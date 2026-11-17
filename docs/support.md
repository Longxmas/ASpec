# Supported paths and limitations

ASpec is a source release for trusted Linux GPU training environments. The primary performance
path is SGLang + DSpark/DFlash + Qwen3 on multi-GPU systems. The
[installation guide](user-guide/installation.md) defines the tested dependency lane. CPU planning
and tests do not imply CPU training support.

## Algorithms and structures

| Path | Implemented behavior | Boundary |
| --- | --- | --- |
| DSpark / DFlash | Training, exact recovery, evaluation and export; Dense and selected MoE/attention structures | Objectives, feature bindings and assets remain algorithm-specific |
| EAGLE3 | Recurrent TTT, full/pruned vocabulary, GQA/MHA or materialized MLA; FA4 packing/CP | No EAGLE3 MoE, anchored mode or whole-model compile |
| P-EAGLE core | Qwen3 full-vocabulary parallel model/loss, deterministic dense slots, eager/SDPA | Model/loss prerequisite only; no public trainer, COD sampler, resume, export or consumer validation; see [scope](user-guide/peagle.md) |
| EAGLE4 | Dense and selected upstream MoE; Draft/candidate stages; packing/CP and native assets | Candidate Top-K forward-KL is explicitly implemented from public semantics, not an unpublished official recipe; no serving integration |
| DFlash2 | Experimental full/causal/SWA training with Flex, cuDNN THD or FA4 packing/CP | Experimental objective; no claim to reproduce an official training recipe |
| MTP-conditioned DSpark / DFlash | Frozen native Qwen3.5 predictor, token/hidden dual-anchor training, exact recovery and strict native assets | Dense full GQA, CP1, MBS1, REPLICATE, one node, non-packing, decay objective; no generic serving consumer |
| Native MTP | Qwen3.5-9B native single predictor, same-Target frozen embedding/head, multi-step training, recovery and export/stitch | Text-only Dense MTP; SDPA CP1 or independently validated FA4 packing/CP; no generic multi-layer/MoE MTP |
| Dense / Qwen MoE / Ling MoE | Structure-specific FFNs, routing state, checkpointing and strict assets | No promise of every algorithm × structure combination |
| MHA/GQA / materialized MLA | Explicit head geometry and tensor bindings | MLA materializes K/V; no absorbed-MLA claim |
| SWA / CSA / QSA / DSA | Explicit window or staged compressed/indexed structures; native assets where required | DSA uses per-query raw-token selection on DSpark/DFlash GQA/MHA; no CSA2 or arbitrary structure combinations; short contexts may be slower |

Use [recipes](user-guide/recipes.md), [algorithm details](user-guide/algorithms.md) and
[structure details](user-guide/structures.md) to choose a concrete configuration. A configuration
passing Plan validation establishes legality, not performance, convergence or downstream quality.

## Execution and assets

- Single-node online training supports required NVLink and explicitly selected Store/transport paths.
  With packing enabled, DSpark/DFlash full and sliding attention preserve document semantics with
  FA4 or native cuDNN flattened-token execution at CP1 and above. Selected-entry CSA/QSA/DSA
  packing/CP remains FA4-only.
- Shared Target capture supports independent single-node Drafts, progress, packing/CP views,
  evaluation, recovery and tracking. Shared multi-node execution and online weight synchronization
  are outside scope.
- Static homogeneous multi-node single-Draft training supports node-local model parallelism,
  explicit transports and required node-local feature affinity. Elastic membership and cross-node
  Target TP are outside scope. Preflight and throughput checks do not certify all failure modes.
- Offline feature replay is single-node, single-Draft, CP1 with CPU Store/TCP transport.
- SGLang export is accepted only when the actual structure can be faithfully represented. CSA/QSA/DSA,
  materialized MLA and EAGLE4 use native assets where required; strict loading/proposal components
  do not provide a serving scheduler or arbitrary Target-cache compatibility.

## Known limitations

Measured training-loss and verifier-acceptance results are available in the
[Qwen3-8B DSpark comparison](benchmarks/dspark-alignment.md) and
[CP comparison](benchmarks/dspark-cp-alignment.md). Their configurations and
validation scope are recorded in those reports.

- Short functional training and exact-recovery tests do not certify long-run model quality, every
  hardware/backend combination, or CUDA driver/DMA/RDMA hardware-hang recovery.
- Static node-local multi-node acceptance covers whole worker Ray/GPU process loss, fencing and
  exact restart with a new node/attempt identity. Host power loss, arbitrary partitions and
  unreturning native/driver operations remain unverified; see [fault validation](contributing/validation.md#whole-worker-node-loss-and-recovery).
- Some short runs do not observe a requested pipeline event and correctly report
  `incomplete_evidence`; absence must not be relabeled successful activation.
- Repeated SGLang engine startup/shutdown has produced upstream `resource_tracker` warnings.
  No GPU-process residue in a bounded check does not certify every OS shared-memory resource.
- Original benchmark environments remain attached to their results. They do not automatically
  certify the current dependency lane. [Benchmark guidance](user-guide/benchmarks.md) distinguishes
  input capacity, sampled supervision and quality.

### Decode reproducibility

Repeated greedy evaluation of the Qwen3-8B DSpark recipe has produced different output token IDs
under default SGLang inference settings with radix cache enabled. Default-inference repeatability
is therefore not established.

For controlled comparisons, use `--deterministic-inference` on the evaluator. The
[framework comparison](benchmarks/dspark-alignment.md) and separate
[CP quality comparison](benchmarks/dspark-cp-alignment.md) use this option and normal
radix cache: output token IDs match across Target-only decoding, the reported Drafts
and all repeats. That result applies to its stated
configuration, not to the default inference mode. See [evaluation](user-guide/evaluation.md) for
the command and failure reporting.
