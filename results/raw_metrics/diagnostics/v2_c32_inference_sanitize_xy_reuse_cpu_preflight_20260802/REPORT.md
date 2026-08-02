# C32 owned-geometry inference reuse CPU preflight

## Decision

The source semantics and CPU retained-storage reduction are approved as a
preflight only.  The candidate reuses the freshly allocated camera-transform
XYZ tensor while sanitizing invalid rays and, when dtype promotion is not
required, for the projected x/y products.  It never aliases a caller input.
Training and grad-enabled evaluation remain fail-closed through the inherited
inference-reuse class.

This is not an adoption, CUDA, FPS, end-to-end latency, or official metric
claim.  In particular, independent shared-host runs did not establish a
worst-p95 tail guarantee.

## Exactness and state

The semantic parent is `e0d539d`; the candidate code is `05f541d`.  Independent
CPU checks compared affine and homogeneous 4x4 transforms across 1,024 mixed
dtype, layout, NaN, Inf, and camera-origin cases.  They also covered 16
noncontiguous cases, 32 empty shapes, and four CPU autocast cases.  Output
dtype, shape, stride, and raw bits were equal, while caller bytes, version
counters, and storage aliases remained unchanged.

The real parent and candidate models have equal state and initialization:

- 595 state keys;
- 535 parameter tensors and 29,667,109 parameter elements;
- 99 buffer tensors and 10,097 buffer elements;
- equal state bytes, Python/NumPy/Torch RNG states, and strict loading.

The full CPU suite passed 294 tests with five optional-host skips; the focused
geometry suite passed 106 tests.  CUDA was hidden and never queried.

## Storage result

A TorchDispatch inventory retained every operator output and counted unique
non-input storages for `B=4,C=4,N=300,P=13`.  The immediate reuse parent used
12,106,448 bytes across 64 storages.  The candidate used 10,047,240 bytes
across 57 storages: 2,059,208 bytes (`17.0092%`) less.  Relative to the
unreused affine projection, the cumulative retained-output reduction is
8,798,416 bytes (`46.6867%`).  These are allocation inventories, not process
or CUDA peak-memory measurements.

## CPU component observation

The author run used one thread, 60 warmups per arm, six alternating PC/CP
rounds, and 300 measurements per arm per round.  The median candidate/parent
ratio was `0.98459`; the median round-p95 ratio was `0.98504`; the observed
worst round-p95 ratio was `0.98753`.

Independent repetitions reproduced the central improvement: median ratios
were `0.98480` and `0.98582`, with median-p95 ratios `0.98841` and `0.98619`.
Their worst round-p95 ratios were `1.06651` and `1.57298`, including a
shared-host outlier.  Therefore this evidence approves neither worst-tail
non-regression nor any whole-detector speedup.  It records only a small CPU
component tendency plus a reproducible storage reduction.

## Pending gates

Adoption still requires CUDA FP32/BF16 exactness, real D0--D5 mask and decoded
prediction checks, fresh official Main126 primary/replay metric closure, and
paired projection plus full-test-step CUDA latency and allocated/reserved peak
memory.  Tail latency must be rerun in an isolated protocol.
