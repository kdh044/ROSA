# C32 exact homogeneous inference-reuse CPU preflight

Status: `AUTHOR_CPU_PREFLIGHT_NOT_ADOPTED`.

The cumulative representation/runtime branch now exposes a release-safe
fallback that keeps the historical homogeneous 4x4 OCam camera transform and
its reduction order exactly, while enabling the previously tested owned
intermediate-storage reuse during ordinary no-grad evaluation.  The module is
eval-only and rejects grad-enabled calls.  Training and the existing public
OCam module remain untouched.

The resolved candidate differs from its cumulative runtime parent only in the
registered deformable-aggregation type plus provenance fields.  It adds no
parameter, buffer, checkpoint key, inference arithmetic, or numeric tuning
dimension.  Its purpose is operational: if the faster affine/compiled arm
fails an official numerical or metric gate, this exact path remains usable.

CPU tests compared the registered module against the ordinary homogeneous
projection in FP16, FP32, and FP64 with raw-bit and stride equality.  The
broader inherited projection suite covers BF16, both point layouts, both
projection contracts, mixed dtypes, non-contiguous/empty inputs, and invalid
rays.  A fresh seeded real-model build preserved 595 state keys, 29,667,109
parameters, 10,097 buffer elements, construction RNG, every state byte, and
strict cross-loading.

The checked-in homogeneous projection probe counts cumulative distinct
operator-output storage while retaining every emitted tensor and excluding
aliases of pre-existing input storages.  Under the production B=4, C=4,
N=300, P=13 FP32 shape, the unreused homogeneous parent materialized
24,336,280 bytes in 95 owned output storages; the exact-reuse candidate
materialized 14,851,464 bytes in 51.  This is 9,484,816 fewer bytes
(38.97398%) and 44 fewer storages, with raw-bit and stride equality.  This is
not peak allocator memory, RSS, CUDA memory, latency, or throughput.

Reproduce the receipt from the repository root with:

```text
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' python \
  results/raw_metrics/diagnostics/v4_c32_exact_homogeneous_inference_reuse_cpu_preflight_20260802/PROBE.py
```

`RAW.json` is the exact probe output.  Earlier shared-host timing aggregates
had no sealed raw samples or complete protocol and are withdrawn.  No CPU or
CUDA speed claim is retained here; full-model latency remains a later gate.

- Worktree: `/home/airlab/user/kdh/ROSA_wt_c32_inference_valid_incidence_reuse`
- Parent cumulative preflight HEAD: `e4f533aaaa01d2293f1e70e13ea7fdfcc31cf60d`
- Code/config commit: `8d29476437f3f1aaaec0138ee2ff90dfb3d0e4ff`
- Tree: `2e7cdce1857837bbf8528732808948b4e7e432dd`
- Focused tests: 63 passed.
- Complete CPU suite: 300 passed, 5 optional-host skips.
- Homogeneous component storage: 24,336,280 B / 95 storages ->
  14,851,464 B / 51 storages (-9,484,816 B, -38.97398%, -44 storages).
- Storage scope: cumulative distinct retained operator-output storage,
  excluding input aliases; not allocator peak or RSS.
- GPU/CUDA/official metric/training actions: none.

No speed claim follows from this component storage result.  Adoption still
requires an independent CPU audit and an idle-GPU same-checkpoint check of raw
predictions, D0--D5 state, official metrics, full `test_step` median/p95, and
allocated and reserved memory.
