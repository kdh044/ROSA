# C32 compact uint8 image transfer

Status: **correctness/parity approved; CUDA speed and memory pending**.

## Change

The opt-in exact C32 runtime pipeline keeps released 8-bit camera images as
uint8 through loading, packing, collation, and host-to-device movement.  The
unchanged device preprocessor then executes its existing `img.float()` before
normalization.  The immediate parent instead expands those same integer pixels
to FP32 on the CPU before transfer.

Existing loaders, the public `PackSparse4DInputs`, and all prior configs remain
unchanged.  The candidate is selected only by its derived config and adds no
learned parameter, buffer, checkpoint key, or numeric model setting.

## Proven contracts

- Resolved train/val/test configs differ from the lean-loader parent only by
  `to_float32=False`, the opt-in uint8 packer, its custom import, work
  directory, and experiment metadata.
- Real Main126 train, validation, and test sample 0 produced byte-identical
  non-image packed inputs and data samples.  This included five nonempty train
  boxes and validation/test evaluator annotations.
- After the actual C32 preprocessor, every model input and data sample was
  raw-byte identical in train and evaluation modes.  Integer pixels in the
  released range 0--255 are exactly representable in FP32.
- The resolved model, state schema, buffers, construction RNG, and numeric
  model knobs are unchanged.
- Independent read-only review: APPROVE.
- Independent CPU-safe focused checks: 25 passed.
- Candidate-specific focused file: 5 passed.
- Full CPU suite: 177 passed, 5 skipped.
- Formatting and `git diff --check`: passed.

## Byte calculation and performance boundary

For batch 4, four RGB cameras, and native 800×800 images:

- FP32 parent: `4 * 4 * 3 * 800 * 800 * 4 = 122,880,000` bytes;
- uint8 candidate: `4 * 4 * 3 * 800 * 800 = 30,720,000` bytes;
- theoretical host image payload reduction: 92,160,000 bytes (75%).

This arithmetic is not a latency result.  The candidate introduces a device
dtype-conversion operation and may change allocator-dependent transient peak
memory.  Do not make a speed or deployment claim until same-sample CUDA
profiling measures transfer bytes, cast cost, latency tails, and peak memory,
and raw-bit output parity passes on CUDA.
