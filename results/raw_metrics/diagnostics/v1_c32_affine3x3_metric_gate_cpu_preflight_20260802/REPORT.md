# C32 affine 3x3 OCam official-metric-gate CPU preflight

## Decision boundary

The candidate is an explicit evaluation-only deformable-aggregation class. It
replaces only the camera-frame homogeneous 4x4 multiply with the equivalent
3x3 rotation followed by translation. The default strict class and its
historical reduction order remain unchanged. Attempting to use the affine
class while training raises before projection.

The source is CPU-prepared only. It is not adopted and carries no CUDA,
end-to-end, or official metric-parity claim.

## Model and configuration isolation

Parent and candidate configs differ only in the six native deformable-layer
type strings plus non-executable work-directory/experiment metadata. Building
both models from seed 3407 produced raw-bit equal values for all 29,667,109
parameters, 10,097 buffers, and 595 state entries, as well as the same
post-build RNG state. The parent contains six strict OCam layers and the
candidate contains six affine evaluation-only layers. There are no new
parameters, buffers, checkpoint keys, or numeric user settings.

The repository-wide CPU suite passed 250 tests with five optional-host skips.
Focused tests cover shared/camera-conditioned points, gradients, both
projection contracts, full-FOV boundary masks, default strict-path retention,
registration, config isolation, and fail-closed training/unknown modes.

## Numerical and CPU component evidence

At the production projection shape `B=4,C=4,N=300,P=13`, the strict and affine
paths had zero validity-mask mismatches over 62,400 camera-points. Complete
FP32 tensors were not raw-bit equal: 140 scalar elements differed, with a
maximum common-valid absolute difference of `1.1920928955078125e-7`.

With one Torch thread, 60 warmups per arm, and six alternating rounds of 300
measurements per arm, the affine/strict median ratios were `0.38838`,
`0.38798`, `0.38784`, `0.38762`, `0.38764`, and `0.38669`; their median was
`0.38774`. Positive self CPU operator allocations fell from 21,840,260 to
16,349,636 bytes per projection. These are isolated CPU component results and
must not be extrapolated to detector FPS or CUDA.

The profiler emitted a failed device-count probe despite `use_cuda=False` and
CUDA being hidden; no GPU context, kernel, or allocation was used.

## Preregistered official gate

The comparison checkpoint is C32 epoch 20, SHA-256
`cb5b0956ab8357d64bbcde919d8aa10aa772d03c73252c137709529e6299e85d`.
Its established strict global:300 reference is mAP
`0.5649954849343621` and NDS `0.6447228330516904`. The affine arm must run a
fresh standalone Main126 evaluation with that exact checkpoint and frozen
validation payloads. Acceptance requires exact evaluator sample/prediction
counts, no NaN/Inf, primary/replay official-analysis equality, and an absolute
delta of at most `1e-9` for mAP, NDS, mAOE, mATE, mASE, every per-class
AP/AOE/ATE/ASE, and every AP distance threshold. Raw D0--D5 validity-mask and
decoded-output deltas must be reported, not hidden.

Only after that metric gate may the affine arm enter the paired CUDA latency
and memory harness. Failure at either gate retains the strict path.
