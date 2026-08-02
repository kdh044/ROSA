# C32 representation-aware paired runtime harness

Status: **CPU implementation and negative-test seal complete; CUDA execution
not performed**.

## Implemented contract

The existing disarmed paired harness now has an opt-in
`lean-loader-uint8-pinned` protocol. It compares two representations of the
same semantic C32 batch:

- parent: lean loader, contiguous CPU FP32 images, unpinned blocking transfer;
- candidate: lean loader, contiguous CPU uint8 images, pinned non-blocking
  transfer;
- both: identical checkpoint, model graph, sample order, batch size,
  preprocessor type, precision settings, and default CUDA stream policy.

The config validator normalizes only the audited loader/packer dtype and
pinning/non-blocking surface. Any other executable delta fails closed.

Each captured pair records the ordered sample identities, a canonical semantic
SHA-256, and separate raw representation fingerprints. FP32 parent pixels must
be exact expanded uint8 values. Pair swaps, semantic mismatches, representation
mutation, dtype/shape/layout changes, pinning drift, and custom stream settings
are rejected.

Before the live input-pipeline timing pass, the harness independently loads and
fingerprints the complete 360-pair warmup-plus-measurement sequence. The timed
pass rebuilds both dataloaders and must replay the same ordered identities and
canonical semantic digests; missing, swapped, reused, or divergent pairs fail.
This deliberately produces a prevalidated warm-cache measurement; the harness
does not flush operating-system page caches.

The harness validates raw-bit parity at the common post-preprocessor FP32
boundary separately from model output/state/RNG parity.

## Separate measurement scopes

Both scopes retain the physical six-round order
`PCCP, CPPC, PCCP, CPPC, PCCP, CPPC`, with 60 excluded warmups and 300 measured
pairs per arm.

1. `model-only-common-preprocessor-boundary` times only
   `model._run_forward(mode="predict")` from an already-proven common boundary.
   Its decision is a non-regression gate and cannot authorize a speed claim.
2. `end-to-end-loader-collate-transfer` times dataset loading/transforms,
   collate/pinning, and preprocessing/H2D transfer. It excludes model execution
   and has its own paired-bootstrap gate.

The harness does not synthesize or report a combined full-inference latency.
Its summary explicitly sets `full_inference_speed_claim_authorized=false`.

Execution additionally requires `--execute`, `--execution-state ARMED`, and the
existing exact idle/reserved-GPU confirmation. The default path remains a
CPU-only dry run and creates no output directory.

## CPU verification

- New representation-aware negative suite: 16 passed.
- Full relevant guarded suite: 68 passed.
- Repository-wide guarded CPU suite: 218 passed, 5 skipped.
- Source compilation, pyflakes, and diff checks passed.
- The real uint8 packer fixture produced one shared semantic digest with
  30,720,000 FP32 parent image bytes versus 7,680,000 uint8 candidate bytes for
  a one-sample, four-camera batch. This is a representation-size contract, not
  runtime evidence.

All CUDA availability calls were replaced before test collection. No real CUDA
query, GPU inference, profiler run, timing run, job signal, queue mutation, or
job restart occurred.

## Evidence boundary

There are no actual GPU numbers and no speed, adoption, or near-native claim.
Pinning effectiveness, asynchronous-transfer behavior, CUDA stream identity,
latency distributions, memory, and profiler evidence remain pending a separate
authorized idle-GPU execution. A C32-versus-C0 near-native statement still
requires a separately sealed true C0/C32 arm pair.
