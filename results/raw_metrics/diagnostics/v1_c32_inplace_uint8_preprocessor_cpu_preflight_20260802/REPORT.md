# C32 in-place uint8 preprocessor CPU preflight

## Decision

The candidate is implemented as an opt-in, prediction-only preprocessor and is
not adopted yet.  It accepts only the released uint8 BGR layout with the fixed
BGR-to-RGB channel selection.  That selection and the following `float()` own
fresh storage, so `sub_` and `div_` normalize the model input without mutating
the caller's batch.  Training, non-uint8 input, and disabled channel conversion
fail closed.

The model graph, checkpoint, mean/std values, operation order, parameters,
buffers, and user numeric settings are unchanged.

## CPU correctness

Direct preprocessing was raw-bit equal to the parent for two padded/unpadded
shapes and two seeds.  A complete prediction-preprocessor call including the
CPU temporal metadata path was also raw-bit equal for every returned input
field.  The source uint8 tensor remained unchanged and the candidate FP32
output owned different storage.

The focused stack passed 24 tests.  The complete CPU suite passed 202 tests
with five optional-host skips.  CUDA was hidden.

## CPU allocation and timing observation

At B=4, four 800x800 RGB views, the normalized FP32 image is 122,880,000
bytes.  A CPU operator-allocation inventory observed 399,360,024 positive
self-allocation bytes for the parent and 153,600,024 for the candidate, a
245,760,000-byte difference.  This inventory is not a peak-memory claim.

Single-thread blocked medians for this preprocessing region were approximately
133.20 ms and 51.31 ms.  These are CPU-only preprocessing observations under
the current machine load, not CUDA transfer results and not end-to-end model
speed claims.

## Required acceptance

After the active training queue is idle, both arms must use semantically paired
batches and pass raw-bit parity at the common GPU model boundary and final
prediction.  Transfer, preprocessing, full `test_step`, and allocated/reserved
peak memory must be reported separately.  The parent remains the default if
the preregistered strict gate is not met.
