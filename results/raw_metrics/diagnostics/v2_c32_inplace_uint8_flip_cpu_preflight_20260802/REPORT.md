# C32 exact uint8 channel-flip CPU preflight

## Decision

The prediction-only preprocessor now uses `torch.flip` on the sealed
three-channel axis instead of the generic advanced-index expression
`[..., [2,1,0], :, :]`.  Both perform the same exact BGR-to-RGB permutation
and allocate fresh uint8 storage before the FP32 cast.  Inputs outside
`[B,camera,3,H,W]` fail closed.

This is still a prepared candidate, not the default and not a CUDA speed claim.

## CPU evidence

For B=4 and four 800x800 RGB images, the isolated channel permutation was
raw-bit and stride equal.  Three blocked medians gave approximately 10.87 ms
for advanced indexing and 1.41 ms for `flip` with one CPU thread.

The complete uint8-to-normalized-FP32 preprocessing region remained raw-bit
equal to the parent.  Its observed blocked median was approximately 102.77 ms
for the original preprocessor and 42.29 ms for the in-place/flip candidate.
The positive self-allocation inventory was 399,360,024 versus 153,600,000
bytes.  These are CPU-region observations only.

The focused suite passed 22 tests, and the full CPU suite passed 202 tests with
five optional-host skips.  CUDA was hidden.

## Required acceptance

The idle-GPU phase must verify raw-bit equality for the channel permutation,
post-normalization model inputs, and final predictions.  Transfer,
preprocessing, full `test_step`, and peak memory must be measured and reported
separately; failure or an inconclusive strict gate retains the parent.
