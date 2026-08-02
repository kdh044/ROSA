# C32 strict OCam homogeneous-pad CPU preflight

## Decision

The candidate is prepared, but not adopted and not authorized for a CUDA run.
It replaces the separate `ones_like` allocation followed by `cat` with one
constant `F.pad`.  It deliberately retains the exact historical homogeneous
`[x,y,z,1]` bytes, broadcast 4x4 `matmul` operand shapes, output slice, detector
graph, checkpoint state, and all numeric settings.

This is distinct from the faster 3x3 affine rewrite in `45310fc`.  That rewrite
changes floating-point reduction order and is therefore not part of this
strict candidate.

## CPU correctness

Raw-bit comparisons covered float32 and float64, shared and camera-conditioned
keypoints, and seeds 3407, 7101080, and 918031.  Forward camera coordinates,
point gradients, and extrinsic gradients were exactly equal.  The padded and
legacy homogeneous tensors also had equal bytes and strides, and the recorded
`matmul` operand shapes were unchanged.

The focused runtime/geometry stack passed 45 tests with one optional-host skip.
The complete CPU suite passed 194 tests with five optional-host skips.  CUDA was
hidden for both runs.

## CPU observation

For only the camera transform at B=4, C=4, N=300, P=13, float32 and one CPU
thread, three blocked medians were approximately 0.8183 ms for the old path and
0.8052 ms for the candidate, a ratio of 0.9840.  This small isolated CPU result
is neither a CUDA result nor an end-to-end detector speed claim.

## Required acceptance

When both training jobs have finished and the GPUs are explicitly idle, the
candidate still must pass raw-bit CUDA checks through camera XYZ, normalized
UV, validity, D0-D5 intermediates, and final predictions.  It must then pass the
preregistered paired model-latency and peak-memory gate.  An inconclusive or
negative result retains the parent implementation.
