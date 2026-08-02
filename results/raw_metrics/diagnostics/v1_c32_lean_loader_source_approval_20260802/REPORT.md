# C32 lean Fisheye3DOD loader

Status: **correctness/parity approved; end-to-end speed pending**.

## Change

The exact C32 train, validation, and test pipelines select a dedicated lean
loader that skips `cam2ego = inverse(lidar2cam)`.  None of those exact
pipelines consumes or packs `cam2ego`; all model inputs, labels, instance IDs,
and evaluator annotations remain unchanged.  The historical loader still emits
`cam2ego`, so the optional ego D4/C4 augmentation branches retain their public
contract.

This is a loader-only optimization.  It adds no learned parameter, buffer, or
numeric user hyperparameter and does not alter the model or checkpoint schema.

## Proven contracts

- The resolved candidate train/val/test configs differ from the CPU-metadata
  C32 parent only by the loader type, work directory, and experiment metadata.
- All 12,726 released Main126 train/validation camera records were audited;
  none supplies an input `cam2ego` field.
- Real first train, validation, and test samples produced bitwise-identical
  packed inputs and data samples, including nonempty ground truth and evaluator
  annotations.
- Synthetic tests forbid the inverse in the lean loader, verify the default
  loader's inverse contract, registry identity, and bitwise packed parity.
- Independent read-only review: APPROVE for correctness/parity.
- Focused suite: 11 passed.
- Full CPU suite with the released evaluator root: 172 passed, 5 skipped.
- `git diff --check`: passed.

## Performance boundary

An alternating mocked four-camera 800×800 component benchmark observed parent
median 3.9199215 ms and candidate median 3.8716255 ms (about 48.296 µs, ratio
0.987679).  This only demonstrates the cost of the removed component and is not
an end-to-end dataloader or inference-speed result.

Do not claim or deploy a speed improvement from this evidence alone.  Promotion
requires a real dataloader-inclusive benchmark, followed by the preregistered
same-batch CUDA parity, profiler, latency-tail, and memory gates for the
cumulative candidate.
