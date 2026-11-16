# Changelog

## Unreleased

- Install independent Draft (`ling-aspec`) and Target (`ling-aspec-target`) environments,
  sharing the version-aligned `ling-aspec-runtime` code package. Target retains SGLang's own
  dependency requirements; no resolver overrides are used. Start `aspec target` and `aspec train`
  in their respective environments and connect by service address. Optional managed-local
  supervision uses the same service interface and retains full checkpoint/resume.
- Keep `aspec patch-sglang` limited to patching files. Report role-specific worker provenance,
  and expose pip CUDA toolkit libraries through a private JIT development view.
- Set the first PyPI version to `0.1.0b0`; developers maintain release versions with the `b0`
  suffix while CI checks the matching tag without changing the version.
- Use `ling-aspec` as the distribution name, retaining the `aspec` namespace and CLI. Publish
  audited tag artifacts to PyPI after wheel installation checks through the GitHub `pypi` environment.
- Allow repeatable manual build/install validation without publication; reserve PyPI uploads
  for version tag pushes.
- Consolidate supplementary examples around distinct workflows; use recipe/config overrides for
  alternative lengths, backends and structure combinations.
- Advance fully committed restored epochs for repeating training sources, including new
  weights-only phases, without replaying committed documents.
- Allow 1200 seconds for Ling Flash Target initialization in the recipes and shared, indexed
  attention and 128K SWA examples to accommodate cold checkpoint loading.
- Bundle the pinned SGLang training patch in the wheel and expose explicit
  `aspec patch-sglang [--check]` application and verification outside a source checkout.
- Install doctor, Quickstart, formal recipes, export/stitching, data preparation and decode evaluation
  with package-local input resources; retain source compatibility commands and Quickstart state audits.
- Build audited release bundles with runtime requirement files and checksums, and verify installed
  wheel commands outside the checkout in GitHub Actions. Evidence includes package assets and build provenance.
- Decode evaluation and selected-answer preparation require explicit `--trust-remote-code` for
  custom Target code. Export rejects missing checkpoint metadata even under optimized Python.

## 0.1.0rc1 - 2026-11-05

ASpec's initial source release provides asynchronous Target-to-Draft training, explicit
Plan/Evidence, work/ownership tracking, exact checkpoint recovery, evaluation and strict export.

- DSpark/DFlash, EAGLE3/EAGLE4, experimental DFlash2 and native Qwen3.5 MTP paths.
- Dense and selected MoE/MLA/windowed/indexed structures, packing and context parallelism.
- Native cuDNN flattened-token packing and context parallelism for DSpark/DFlash full and
  sliding-window attention; selected-entry sparse attention continues to use FA4.
- Single-node shared Target capture for independent Drafts and static single-Draft multi-node paths.
- Pinned SGLang training patch, environment diagnostics, examples and source tools.
- ASpec-owned DSpark Q/K CuTeDSL forward/backward kernels and an isolated, tested FA4/CUTLASS
  compatibility boundary.
- Source setup, container instructions, training recipes and benchmark reproduction tools.

See [supported paths and limitations](docs/support.md) before choosing a combination. Short
functional validation does not establish model quality, universal speedup or all-fault production
certification. Final benchmark claims remain tied to their documented source and environment identity.
