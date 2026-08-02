# C32 strict OCam temporary-allocation CPU preflight

## Decision boundary

This opt-in candidate preserves the historical 4x4 homogeneous camera
transform and every projection arithmetic operation.  It removes only full
value-shaped fallback inputs that are immediately consumed by `where`, and it
defers the Horner leading-coefficient materialization until the first multiply
that already creates a full output.  Invalid rays still evaluate at
`(0,0,-1)` and return the exact `(2,2)` sentinel.

No parameter, buffer, checkpoint key, configuration field, or user numeric
setting is added.  The candidate is CPU-prepared only; it is not the default
and carries no CUDA or end-to-end speed claim.

## Exactness evidence

The focused contracts compare the prior implementation and candidate for
FP32 and FP64, three fixed seeds, shared and camera-conditioned point layouts,
and both `legacy_front_only` and `full_fov` projection contracts.  Complete
projection outputs and gradients with respect to points, `lidar2cam`, OCam
intrinsics, and OCam polynomial coefficients are raw-bit equal.  Dedicated
Horner checks cover one, two, five, and eight coefficients, including output
strides and unused degree-zero gradients.  NaN, Inf, and camera-origin rays
retain the exact sentinel behavior.

The repository-wide CPU suite passed 242 tests with five optional-host skips
and CUDA hidden.

## CPU component observation

For the production projection shape `B=4,C=4,N=300,P=13`, one Torch thread,
60 warmups per arm, and six alternating physical rounds with 300 measurements
per arm per round, the candidate/parent median ratios were `0.99481`,
`0.99583`, `0.99614`, `0.99751`, `0.99620`, and `0.99576`; their median was
`0.99599`.

The sum of positive self CPU operator allocations fell from 24,024,260 to
21,840,260 bytes, a reduction of 2,184,000 bytes per projection call.  The
three full `zeros_like` inputs, one full `full_like` sentinel, and one
redundant Horner clone disappear.  This allocation inventory is not a peak
memory measurement.  CPU latency changed only slightly, so the evidence does
not support a material detector speedup claim.

The profiler emitted a failed device-count probe despite `use_cuda=False` and
`CUDA_VISIBLE_DEVICES` being empty; no GPU context, kernel, or allocation was
used.  This is recorded rather than described as a zero-query run.

## Pending acceptance gates

An idle-GPU check must prove raw-bit forward and first-order-gradient equality
for FP32/BF16 where supported, identical D0--D5 projection masks and decoded
predictions on real Main126 batches, and non-regressing CUDA latency/peak
memory under the preregistered paired harness.  Any failure retains the
parent.
