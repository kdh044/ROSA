# C32 single-frame CPU metadata optimization

Status: **source/CPU contract approved; CUDA parity and latency pending**.

## Change

The exact C32 single-frame inference arm keeps `T_global`, `T_global_inv`, and
`timestamp` on CPU while the remaining prediction inputs follow the unchanged
Sparse4D device preprocessor. Sparse4D's unchanged `_build_metas` therefore
constructs the same NumPy pose matrices and Python timestamps without first
copying these three tensors to CUDA and immediately reading them back.

The optimization is fail-closed to the exact
`ROSAOfficialResidualNativeBaselineHead`, an exact integral
`num_temp_instances=0`, and an operation order without `temp_gnn`. Prediction
TTA/list input is rejected. Training uses the historical preprocessor path.

## Proven CPU/source contracts

- The candidate config differs from C32 only in `model.data_preprocessor.type`.
- Both models contain 29,667,109 parameters and identical state-dict keys and
  values, buffers, construction RNG, and strict checkpoint schema.
- Synthetic prediction inputs and internal `img_metas` are bitwise equal,
  including mapping order, dtype, shape, and NumPy bytes.
- A real Main126 validation batch has contiguous CPU tensors with shapes
  `T_global=(4,4,4)`, `T_global_inv=(4,4,4)`, and `timestamp=(4,)`.
- Focused tests: 9 passed.
- Full worktree CPU suite: 166 passed, 5 skipped.
- Independent read-only review: APPROVE.

For batch size four, source tracing predicts removal of three small H2D input
transfers, eight 4x4 D2H pose copies, and four CUDA scalar timestamp reads per
test step. This is an operation-count result, not a measured speedup.

## Remaining release gate

Do not claim or deploy a speed improvement until an idle GPU is available and
the frozen C32 checkpoint passes both requirements:

1. raw-bit prediction, internal state, and RNG parity on repeated real batches;
2. an alternating same-batch CUDA benchmark and profiler trace confirming the
   expected metadata transfers are absent, with no latency-tail or peak-memory
   regression.
