# C32 incidence and validity storage reuse CPU preflight

## Decision

`AUTHOR_CPU_PREFLIGHT_NOT_ADOPTED`

The no-grad OCam projection now reuses two tensors that are owned by the
projection call and dead at the reuse point.  After Horner evaluation,
`theta_ocam` becomes the output buffer for incidence.  The owned
`geometry_valid` mask then accumulates the lens, finite-pixel, and image-bound
predicates in place.  Caller inputs, the training path, the grad-enabled path,
the projection formula, parameters, buffers, checkpoint keys, and public
configuration are unchanged.

This is a CPU source and allocation preflight only.  It is not an official
metric, CUDA, FPS, full-detector, tail-latency, or deployment result.

## Isolation

- Worktree: `/home/airlab/user/kdh/ROSA_wt_c32_inference_valid_incidence_reuse`
- Branch: `optimize/c32-inference-valid-incidence-reuse`
- Semantic parent: `d3e7982b72fa42f6cc7dbfaa8e1733f4fd49d146`
- Code commit: `fc3eef9b47a22c3b74e9ee1a05933fa9a472ba1a`
- Code tree: `137b07e966fe88637077daaf534f11e3c6e20f06`
- Changed code paths: `src/rosa/geometry.py` and its dedicated storage-reuse
  test only.

The implementation is reached only through the already fail-closed
`inference_storage_reuse=True` path, which rejects ordinary grad-enabled
execution.  The eager parent path is untouched.

## Exactness and model identity

A deterministic 512-case CPU comparison covered FP16, FP32, FP64, and BF16;
shared and camera-conditioned points; homogeneous and affine transforms;
legacy-front-only and full-FOV contracts; non-contiguous and empty inputs;
mixed polynomial/image-extent dtypes; camera origins, NaN, Inf, and boundedness
rejection.  Output dtype, shape, stride, and raw bits matched the immediate
parent.  Caller values, version counters, and storage identities did not
change.

The focused geometry suite passed 108 tests.  The complete CPU suite passed
296 tests with five optional-host skips.  A fresh seeded build of the real
affine parent and inference-reuse C32 models produced identical Python,
NumPy, and Torch RNG states, 595 state keys, 535 parameter tensors containing
29,667,109 elements, 99 buffer tensors containing 10,097 elements, state
digest `56a5e11a119922ed4946501cd13ded1f92e4d138a72ab9aafc7ada738cbe78de`,
and strict cross-loading.

The operator-contract test observes two ordinary `atan2` calls in the parent.
The candidate has one ordinary call and one `atan2.out` call.  It also records
seven in-place validity conjunctions for full-FOV and eight for the legacy
front-plane contract, while retaining exact outputs.

## CPU allocation inventory

For `B=4,C=4,N=300,P=13`, a TorchDispatch mode retained every operator output
and counted unique non-input storages:

| Arm | Retained bytes | Storages |
|---|---:|---:|
| Immediate parent | 10,047,240 | 57 |
| Candidate | 9,360,840 | 49 |

The incremental reduction is 686,400 bytes and eight storages, or 6.8317% of
the immediate parent's retained-output inventory.  Relative to the earlier
unreused affine inventory of 18,845,656 bytes, the cumulative reduction is
9,484,816 bytes, or 50.3289%.  This inventory is not allocator peak, RSS, or
CUDA memory.

## CPU component observation

One CPU thread, 60 warmups per arm, six alternating PC/CP rounds, and 300
measurements per arm per round gave:

- median of round median ratios: `0.9948737857`;
- median of round p95 ratios: `0.9898214567`;
- worst observed round p95 ratio: `0.9989283401`.

The median component tendency is about 0.51% faster.  The effect is small and
was measured under shared-host load, so it is not promoted to a speed claim.

## Pending gates

Before adoption, an independent reviewer must reproduce the source semantics,
mixed-dtype and invalid-input exactness, allocation inventory, state identity,
and evidence hashes.  An idle GPU must then pass FP32/BF16 raw-bit projection,
real D0--D5 and decoded-output parity, official Main126 primary/replay metric,
and paired full-`test_step` latency-tail plus allocated/reserved-memory gates.

