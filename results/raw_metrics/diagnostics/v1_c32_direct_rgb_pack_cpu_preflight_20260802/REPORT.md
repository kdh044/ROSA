# C32 direct released-RGB packing CPU preflight

## Decision boundary

The released validation loader supplies four independent 800x800 uint8 HWC
arrays.  The parent RGB-fused packer first stacked those arrays into one HWC
batch and then allocated a second contiguous RGB-CHW batch.  This candidate
allocates the final RGB-CHW array once and asks NumPy to stack four
zero-copy channel-reversed CHW views directly into that output.

The ndarray fallback retains the parent implementation.  No model operation,
checkpoint state, input value, learned parameter, buffer, or user numeric
setting changes.  The candidate remains opt-in and carries no CUDA or
end-to-end detector speed claim.

## Exactness evidence

Synthetic even and odd image shapes, two fixed seeds, ndarray and released
list ingress, caller-input immutability, mixed dtype/shape rejection, and the
paired preprocessor boundary all pass with raw-bit equality.  The direct-list
test intercepts `numpy.stack` and requires the final output buffer to be
provided.

Real Main126 validation frames 0, 1953, and 3905 were loaded from all four
cameras.  The packed RGB tensors equal an exact channel flip of the parent
BGR tensors, caller image bytes remain unchanged, and the normalized FP32
model inputs are raw-bit equal.  The repository-wide CPU suite passed 248
tests with five optional-host skips while CUDA was hidden.

## CPU component observation

At the released four-camera 800x800 shape, the mandatory packed output is
7,680,000 bytes per sample.  The parent list path additionally materializes a
7,680,000-byte HWC stack; the candidate removes that payload allocation.
This is an arithmetic payload inventory, not a measured process peak.

With 60 warmups per arm and six alternating physical rounds of 300 paired
measurements, the median candidate/parent packing-time ratio was `0.84414`.
The median round p95 ratio was `0.74878`, and the worst round p95 ratio was
`0.82052`.  This supports the CPU packing optimization itself, but not a
whole-pipeline or GPU FPS claim.

## Pending acceptance gates

An idle-GPU run must jointly seal this packer with `bgr_to_rgb=False`, prove
the same common-boundary model input and prediction semantics, and pass the
representation-aware loader/collate/transfer and full-test-step latency and
memory gates.  Any failure retains the parent packer.
