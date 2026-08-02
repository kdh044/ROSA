# C32 affine OCam no-grad storage-reuse CPU preflight

## Decision boundary

This opt-in evaluation class inherits the already isolated affine 3x3 OCam
candidate.  It changes no arithmetic expression relative to that immediate
parent.  During `torch.no_grad()` or `torch.inference_mode()` only, it reuses
owned Horner, pixel-coordinate, normalization, and final projection buffers
instead of allocating a new tensor after each operation.  Training and
grad-enabled evaluation fail closed.

The historical homogeneous 4x4 default and the existing affine parent remain
unchanged.  No learned parameter, buffer, checkpoint key, or user numeric
setting is added.  This is CPU preflight evidence only; the candidate is not
adopted and carries no CUDA, detector FPS, or official-metric claim.

## Exactness and state evidence

Parent and candidate projection outputs are raw-bit equal for FP16, FP32,
FP64, and BF16; shared and camera-conditioned point layouts; one, two, five,
and ten Horner coefficients; NaN, Inf, camera-origin, and boundedness-rejected
rays.  Output strides and all caller inputs are unchanged.  The strict default
path continues to pass its original contracts.

Seeded full-model builds have identical 595 state keys, all state tensor
bytes, 29,667,109 parameters, 10,097 buffers, and final RNG state.  The only
resolved-config change from the affine parent is the registered deformable
aggregation type, applied to all six decoder stages.  The repository-wide CPU
suite passed 278 tests with five optional-host skips while CUDA was hidden.

## CPU component observation

For `B=4,C=4,N=300,P=13`, one Torch thread, 60 warmups per arm, and six
alternating rounds with 300 paired measurements per arm, the median
candidate/parent projection-time ratio was `0.87497`.  The median round p95
ratio was `0.86418`, and the worst round p95 ratio was `0.87126`.

A Torch-dispatch storage inventory retained every operator output to prevent
allocator address reuse.  Unique non-input output storage fell from
18,845,656 to 12,106,448 bytes, a reduction of 6,739,208 bytes (`35.76%`) and
93 to 64 storages for one projection call.  This inventory is not a measured
process or CUDA peak.

## Pending acceptance gates

The inherited affine transform still requires a newly sealed, fresh
immediate-parent and candidate Main126 primary/replay evaluation with full
validation payload closure.  The existing affine evidence does not satisfy
that adoption gate.  An idle-GPU run must additionally prove FP32/BF16
raw-bit equality for the storage-reuse operations, identical D0--D5 masks and
predictions, and non-regressing paired CUDA latency and allocated/reserved
peak memory.  Any failure retains the strict parent.
