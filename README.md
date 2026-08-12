# ROSA: Re-Observation for Native-Fisheye 3D Detection

ROSA refines detector proposals by re-observing their 3D keypoints in
calibrated native-fisheye feature maps. The method code in `src/rosa` is
detector-agnostic; MMDetection3D, Sparse4D, and Fisheye3DOD bindings are kept
under `src/rosa/integrations`.

## Result

Fisheye3DOD Main126, ResNet-18 FPN, temporal off, 20 epochs, global batch 8,
seed 3407:

| Method | mAP | NDS/FDS | mATE ↓ | mASE ↓ | mAOE ↓ |
|---|---:|---:|---:|---:|---:|
| Detector baseline | 0.5145 | 0.5936 | 0.5090 | 0.1003 | 0.3725 |
| ROSA, epoch 20 | **0.5256** | **0.6029** | **0.4860** | **0.0979** | 0.3753 |

The best validation checkpoint is epoch 16 (mAP 0.5291, NDS 0.6035). The
authoritative 11-variant fixed-epoch table is
[`results/metrics/ablation_epoch20.csv`](results/metrics/ablation_epoch20.csv);
best-epoch values, histories, raw evaluator outputs, logs, configs, analyses,
and figures are all consolidated under [`results/`](results/). Every reported
ablation uses one seed, so these results do not estimate cross-seed variance.

## Method

For each selected object query, ROSA:

1. projects 13 stable 3D keypoints into four native fisheye cameras;
2. applies degree-2 spherical ray feature lifting;
3. performs query-conditioned camera fusion independently per keypoint;
4. aggregates the 13 point observations;
5. scores five adaptive log-range hypotheses and predicts a reliability-gated
   center, size, and yaw update;
6. reprojects the updated box for a shared second refinement pass.

The released experiment refines the top 100 of 300 detector queries while
leaving unselected boxes and native detector confidence unchanged.

## Fisheye sparse-ray decoder V2 (research arm)

V2 is not named or presented as Sparse4D. It reuses the released image
backbone, FPN, 300-query budget, six-decoder schedule, matching, losses, and
evaluator as experimental infrastructure, but replaces Sparse4D's defining
box-conditioned 13-keypoint deformable aggregation and Cartesian refinement.
The corrected Sparse4D C0 remains the architecture baseline.
Its registered detector type is therefore `FisheyeSparseRayHead`; the host
`Sparse4DHead` superclass is used only as a matching/loss/decode compatibility
mixin and its constructor and detection architecture are not instantiated.

Each V2 decoder stage is:

```text
self-attention
  -> one native-fisheye observation per query/camera/FPN level
  -> RFL value lifting + ROSA camera-level joint attention
  -> FFN
  -> spherical ray-state refinement
```

D0 starts from a learned direction and has no range, size, yaw, or box. It
uses rotation-only projection and predicts a continuous log-range
distribution plus the first box state. D1--D5 propagate that query and state,
project the current 3D centre with camera translation, and refine direction on
the sphere's tangent plane together with log-range, size, relative yaw, and
velocity. This decoder-to-decoder query propagation is retained. Temporal
cross-frame propagation was already disabled in the C0/Full experiments and
remains disabled.

The fisheye-specific intervention is not an OCam-only rule. A generic lens
backend supplies native projection and a pixel-per-radian Jacobian; the
released dataset selects its OCam implementation. The Jacobian converts FPN
cell size to angular footprint, a query tangent frame describes camera
relation, and one joint softmax selects across all four cameras and four FPN
levels. There is no BEV conversion, 2D detector, dense depth head, fixed image
patch, metric range-candidate list, post-D5 sidecar, or duplicate observation.

Training retains C0's five denoising groups. A noisy DN box is reduced to its
centre direction before D0, so the training query receives no privileged
range, shape, or yaw. Free and DN queries are isolated by the native attention
mask and all six native DN losses remain active.

The evaluated V2 head config contains no `InstanceBank`, anchor encoder,
graph/deformable aggregation, Cartesian refiner, or legacy ROSA sidecar entry.
It names only the C0 matcher/DN sampler, losses, output decoder, and the anchor
file used to initialize directions. On a real batch, all 380 trainable tensors
receive finite gradients, which seals the two-process DDP contract with
`find_unused_parameters=False`.

The first implementation is
[`configs/fisheye_ray_decoder_v2_raw_uniform_20e.py`](configs/fisheye_ray_decoder_v2_raw_uniform_20e.py).
Its core and host integration currently pass 26 focused tests, including a
real Sparse4D matcher/loss/backward test and the isotropic-Jacobian gradient
boundary. The broader repository regression passes 998 tests after excluding
the historical near-native harness whose sealed external snapshot path is not
present in this workspace. A real Fisheye3DOD batch also passes a 200-iteration
two-GPU stability gate through the image backbone, matcher, all 54 losses,
deterministic backward, gradient clipping, and parameter update. This gate
specifically covers the learned-ray trajectory that exposed the original
`acos` axis-singularity after roughly 140 iterations.

The accuracy baseline is the separately trained fresh 20-epoch C0, not an
epoch-20 checkpoint from a 30-epoch cosine schedule. It obtained mAP 0.5644
and NDS 0.6342. The checkpoint hashes, per-class AP, and matched protocol are
recorded in
[`c0_raw_uniform_20e_reference.json`](results/raw_metrics/evaluations/c0_raw_uniform_20e_reference.json).
Against the queued V2 run, dataset, raw-uniform sampling, global batch 8,
optimizer, 20-epoch scheduler, seed, backbone/FPN, preprocessing, and official
evaluator are identical; the detector head is the intended intervention.

The completed epoch-20 comparison is:

| Detector head | mAP | NDS/FDS | mATE ↓ | mASE ↓ | mAOE ↓ |
|---|---:|---:|---:|---:|---:|
| corrected Sparse4D C0 | **0.5644** | **0.6342** | **0.4623** | **0.0858** | **0.3402** |
| Fisheye Sparse-Ray Detector V2 | 0.5288 | 0.5913 | 0.5216 | 0.1448 | 0.3723 |

V2 therefore does not beat C0: its deltas are -0.0356 mAP and -0.0429 NDS.
The largest class-AP regression is Bus (-0.1044), followed by Truck (-0.0446),
while Pedestrian (-0.0041) and Cyclist (-0.0039) nearly match C0. The best V2
mAP is 0.5307 at epoch 16; epoch 20 is the best NDS checkpoint. These results
point to metric centre/size estimation for large vehicles, rather than sparse
angular evidence itself, as the first refinement target.

On the same GPU and native batch, the reproducible harness at
[`tools/benchmark_fisheye_ray_decoder_v2.py`](tools/benchmark_fisheye_ray_decoder_v2.py)
measured:

| Complete detector path | C0 | V2 | V2/C0 |
|---|---:|---:|---:|
| deterministic train, batch 4, forward + loss + backward | 267.00 ms | 361.25 ms | 1.353x |
| deterministic train peak allocation | 8,692 MiB | 9,872 MiB | 1.136x |
| compiled inference, batch 1 | 11.87 ms | 9.92 ms | 0.835x |
| compiled inference peak allocation | 741 MiB | 699 MiB | 0.943x |
| parameters | 29.67 M | 18.58 M | 0.626x |

Both inference arms compile the complete head with the same
`reduce-overhead` setting. V2's maximum eager-to-compiled differences were
`3.3e-5` for classification, `1.9e-6` for boxes, and `2.2e-5` for quality.
Uncompiled V2 inference is still slower (about 27.9 ms versus 19.6 ms), so the
compiled head is the intended latency path. A cold compile is excluded from
steady-state latency; the final V2 graph's uncached warmup took about 84
seconds. Training uses an explicit four-gather sampler
because CUDA `grid_sample` backward violates the sealed deterministic runner
contract. That makes training 35.9% slower even though the intended compiled
inference path is 16.5% faster; these are separate engineering results. The
post-training benchmark records are
[`fisheye_ray_decoder_v2_deterministic_train_post20e_exact.json`](results/raw_metrics/benchmarks/fisheye_ray_decoder_v2_deterministic_train_post20e_exact.json)
and
[`fisheye_ray_decoder_v2_compiled_inference_post20e_exact.json`](results/raw_metrics/benchmarks/fisheye_ray_decoder_v2_compiled_inference_post20e_exact.json).

Focused verification and the two fair benchmarks are:

```bash
pytest -q \
  tests/test_fisheye_ray_sparse4d_integration.py \
  tests/test_fisheye_decoder.py tests/test_fisheye_evidence.py \
  tests/test_fisheye_projection.py tests/test_ray_state.py \
  tests/test_ray_losses.py tests/test_summarize_fisheye_ray_decoder_v2.py

CUDA_VISIBLE_DEVICES=1 python tools/benchmark_fisheye_ray_decoder_v2.py \
  --mode train --batch-size 4 --warmup 5 --iterations 15

CUDA_VISIBLE_DEVICES=1 python tools/benchmark_fisheye_ray_decoder_v2.py \
  --mode inference --batch-size 1 --warmup 5 --iterations 30 --compile-head
```

The queued experiment runs a 200-iteration two-GPU DDP stability preflight
before training.
After epoch 20, `tools/summarize_fisheye_ray_decoder_v2.py` requires the final
official validation, checkpoints, and execution-source snapshot, then writes
`final_metrics_vs_c0.json` with validation history, C0 deltas, and SHA256
records. Missing completion evidence is an error rather than a partial result.

### Post-training deterministic sampler optimization

The sealed epoch-20 result above predates a sampler-only optimization. The
deterministic bilinear sampler now fetches all four neighbours through one
vectorized gather and prepares normalized query/camera coordinates once per
FPN pyramid. It preserves deterministic CUDA output, feature gradients, and
UV gradients. On the same complete batch-4 training benchmark, V2 changed
from 361.25 ms/iteration to 290.90 ms/iteration. The matched rerun measured
270.32 ms for C0, making optimized V2 1.076x C0 instead of 1.353x. Peak
allocation remains 1.124x C0 and requires separate work. This code has not
been retrained for an accuracy claim.

The machine-readable benchmark is
[`fisheye_ray_decoder_v2_one_gather_train_post20e.json`](results/raw_metrics/benchmarks/fisheye_ray_decoder_v2_one_gather_train_post20e.json).
The sampler oracle, focused/integration suite, and broad sealed-host regression
pass 999 tests in total (the unavailable historical near-native harness remains
excluded).

## Fisheye box decoder V3 (implemented research arm)

V2 showed that a direction-only query observed at one image point is
insufficient for large-object centre and extent estimation. V3 therefore
returns to corrected Sparse4D C0 as its host instead of deleting the box-query
pipeline:

```text
300 learned query features + anchor boxes
  -> native-fisheye centre-ray box initializer
  -> D0: current box -> 13 points -> fisheye ROSA -> RFL -> refine
  -> D1: current box -> 13 points -> fisheye ROSA -> RFL -> refine
  -> D2: current box -> 13 points -> fisheye ROSA -> RFL -> refine
  -> D3: current box -> 13 points -> fisheye ROSA -> RFL -> refine
  -> D4: current box -> 13 points -> fisheye ROSA -> RFL -> refine
  -> D5: current box -> 13 points -> fisheye ROSA -> RFL -> final box
```

The initializer uses one centre ray across four cameras and four FPN levels.
RFL supplies spherical-ray appearance lifting; ROSA uses the calibrated lens
Jacobian, lens margin, camera overlap, and image evidence to gate a residual to
the existing anchor box. It does not introduce a probabilistic tracking decoder
or a 2D/depth prompt.

Each D0--D5 stage derives seven fixed object-frame points from the current box
(centre and positive/negative face centres on all three axes) and six
query-learned in-box points. Every point is projected into the four native
fisheye views. Query-conditioned ROSA logits combine the analytic fisheye
Jacobian, lens margin, overlap, and expected angular footprint to select among
camera x FPN-level observations independently for each point. Per-point RFL
then couples the fused appearance with that point's calibrated ego ray, and a
separate query-conditioned attention fuses all 13 points. The result refines
spherical direction, log-range, log-size, ray-relative yaw, and velocity before
propagation to the next stage. Setting `num_learnable_points=0` is the sealed
seven-point ablation.

The production implementation presents each box point to Sparse4D's fused DAF
CUDA kernel as an independent one-keypoint sparse anchor. DAF is only the
bilinear weighted feature reader: V3 owns the fisheye projection, calibrated
ROSA weights, point rays, RFL, 13-point attention, and box update. This removes
the initial materialized `[B,Q,13,4,4,256]` carrier without reverting to C0's
query-only aggregation weights.

Temporal cross-frame propagation remains disabled for the first matched C0
comparison. Decoder-to-decoder query and box propagation is retained. ROSA and
RFL are decoder-internal observation operators rather than a post-D5 sidecar.

The implemented config is
[`configs/fisheye_box_decoder_v3_raw_uniform_20e.py`](configs/fisheye_box_decoder_v3_raw_uniform_20e.py).
It retains C0's 300 free boxes, five DN groups (up to 320 additional training
queries), six decoder stages, raw-uniform sampling, seed 3407, and global batch
8. It has 18.91 M parameters versus C0's 29.67 M. On the matched pre-training
benchmark, batch-4 forward/loss/backward is 349.34 ms versus C0's 266.41 ms
(1.311x), while batch-1 inference is 39.64 ms versus 21.90 ms (1.810x, 25.2
FPS). Inference peak allocation is slightly lower than C0; training peak is
12.34 GiB versus 8.69 GiB. The machine-readable record is
[`fisheye_box_decoder_v3_pretrain_benchmark.json`](results/raw_metrics/benchmarks/fisheye_box_decoder_v3_pretrain_benchmark.json).

The V3 unit/integration tests and complete sealed-host repository regression
pass 1037 tests. A real 200-iteration stability preflight completed with finite
losses and no unused trainable parameters. The two-GPU launch helper waits for
both GPUs to be genuinely free, reruns the 200-step DDP gate, snapshots the
source, and then starts the matched 20-epoch experiment.

The matched 20-epoch run completed successfully. At epoch 20, V3 reached mAP
0.5924 and NDS 0.6333 versus C0's 0.5644 and 0.6342. Translation improved
(mATE 0.4497 versus 0.4623), while scale and orientation regressed (mASE
0.1212 versus 0.0858; mAOE 0.4066 versus 0.3402). The best checkpoint was
epoch 16 at mAP 0.5950 and NDS 0.6337. The compact result, per-class AP,
checkpoint hashes, source-snapshot hash, and exact C0 deltas are recorded in
[`fisheye_box_decoder_v3_final_20e.json`](results/raw_metrics/evaluations/fisheye_box_decoder_v3_final_20e.json).

The detector host, official evaluator, and dataset environment variables in
the setup table below must be exported for these integration commands.

## Repository

```text
ROSA/
├── assets/          released 300-query anchor
├── configs/         one reproducible Main126 training config
├── results/         all retained experiment records and figures
├── src/rosa/        ROSA core and optional framework integrations
├── tests/           CPU regression and release-integrity tests
├── tools/           MMEngine train/evaluate runner
└── weights/         one ignored local release checkpoint
```

`README.md` is the repository's only documentation file. Historical
experiment configs live in `results/configs`; they are evidence, not the
recommended entry point.

## Setup

Python 3.10 or newer and PyTorch 2.1 or newer are required. Core-only
development needs no MMDetection3D installation:

```bash
python -m pip install -e ".[dev]"
pytest -q
```

Training additionally needs the MMEngine-based Sparse4D host and the released
Fisheye3DOD evaluator. The default local layout is:

```text
ROSA/
└── external/
    ├── detector_host/  tested Sparse4D detector host
    └── fisheye3dod/    weiyangdaren/Fisheye3DOD
../datasets/
└── Fisheye3DODdataset/
    └── ImageSets-main126/
        ├── fisheye3dod_infos_train.pkl
        └── fisheye3dod_infos_val.pkl
```

For a fresh checkout, clone the two tested revisions:

```bash
git clone https://github.com/kdh044/sphere_stream_sparse4d.git external/detector_host
git clone https://github.com/weiyangdaren/Fisheye3DOD.git external/fisheye3dod
git -C external/detector_host checkout e0817a30b90ec22f9a9d02952d45ad734de2468c
git -C external/fisheye3dod checkout a28726ed002a02cb1328960848c0f4f2473748de
python -m pip install -e ".[integration,dev]"
(
  cd external/detector_host/projects/mmdet3d_plugin/ops
  python setup.py build_ext --inplace
)
```

The deformable aggregation extension requires a compatible CUDA toolkit and
C++ compiler. The ROSA adapter accepts the tested upstream host API directly;
no dirty Sparse4D working tree is required.

The config accepts these overrides:

| Variable | Purpose |
|---|---|
| `ROSA_DETECTOR_HOST_ROOT` | detector-host repository |
| `SPARSE4D_ROOT` | legacy alias for `ROSA_DETECTOR_HOST_ROOT` |
| `FISHEYE3DOD_OFFICIAL_ROOT` | released evaluator repository |
| `FISHEYE3DOD_DATA_ROOT` | dataset root |
| `FISHEYE3DOD_ANN_ROOT` | Main126 annotation directory |
| `FISHEYE3DOD_ANCHOR_ROOT` | directory containing the anchor file |
| `FISHEYE3DOD_ANCHOR` | anchor `.npy` file |
| `ROSA_WORK_DIR` | logs and checkpoints |

## Train and evaluate

From the ROSA repository root:

```bash
python tools/run.py train configs/rosa_r18_main126.py
```

For the released two-GPU global batch of 8:

```bash
torchrun --nproc-per-node=2 tools/run.py train \
  configs/rosa_r18_main126.py --launcher pytorch
```

Evaluate the retained checkpoint:

```bash
python tools/run.py evaluate \
  configs/rosa_r18_main126.py \
  weights/rosa_r18_main126_best_nds_epoch16.pth
```

The standalone core API is:

```python
from rosa import ROSAConfig, ROSARefiner

config = ROSAConfig(
    num_refinement_steps=2,
    refine_topk=100,
    projection_contract="legacy_front_only",
)
refiner = ROSARefiner(config)
```

See [`tests/test_refiner.py`](tests/test_refiner.py) for an executable tensor
example.

## Compact RFL and checkpoint compatibility

The `balanced` and `fast` profiles use a signed rank-16 factorization of the
appearance-by-degree-2-SH interaction. This reduces the RFL from 40,960 to
8,336 learned weights while leaving the `published` and `legacy_release`
profiles on the released dense outer-product topology. The compact output
projection starts at zero. At the isolated RFL module boundary, a supplied
upstream gradient trains that projection on the first optimizer step and
unlocks the appearance and ray factors on the second backward. The complete
refiner also starts its task readouts at zero to preserve the native box: its
first backward trains those readouts, its second trains the RFL output
projection, and its third unlocks the appearance and ray factors. Separate
module-local and production-loss regressions cover both intentional sequences.

Factorized compact checkpoints use contract v3 and
`rosa.rfl.signed_gated_cp16.v3`. Dense compact-v2 checkpoints are not
shape-compatible and must be evaluated with their pinned v2 source or
retrained; they are never silently converted. Evaluation, benchmarking,
stage-route analysis, and explicit or automatic training resume validate the
complete config-owned ROSA refiner key/shape schema before MMEngine loads the
checkpoint. The mismatch override may relax audited projection metadata, but
cannot bypass topology or state integrity failures. Historical
`canonical_sixth_block` prefixes remain accepted through the released remap.
Training resume and stage-route dumping validate and load the same unlinked,
private exact-byte snapshot rather than reopening the source pathname. Under
torchrun, every DDP process owns, validates, and cleans up its own descriptor;
descriptor paths are process-local and are never shared between ranks.

The paired 30-epoch C32 control and fast-CP16 candidate add a stricter resume
contract. It binds the exact head, resolved config, source revisions, package
runtime, complete native-plus-sidecar state schema, optimizer groups, and the
converted 30-epoch scheduler before MMEngine's permissive loader runs. Their
checkpoint policy is identical: retain the latest five resumable epochs,
hard-link epoch 8 as a protected resume milestone, and retain separate best
`fisheye_bev/mAP` and best `fisheye_bev/NDS` weights. Best-only weights do not
contain optimizer/scheduler state and are therefore rejected for resume.

The speed harness leaves the raw-exact `fast` compile route unchanged and adds
a benchmark-only `fast_compiled_bounded` arm. It uses
`max-autotune-no-cudagraphs` and matches outputs by query ID. Query membership,
labels, and instance IDs stay exact; probability errors are capped at
`3.5e-6`/`4e-6` and decoded boxes at 3 mm, 0.003 rad, or 3 mm/s. Rank inversions
may only cross numerical ties within `7e-6`. The gate requires an eager-repeat
baseline and checks the compiled output against both eager references.
Promotion still requires official Main126 full-metric parity; these are fixed
protocol constants, not user knobs.

## Checkpoint and results

Only the best Full checkpoint is retained locally:

| File | Bytes | SHA-256 |
|---|---:|---|
| `weights/rosa_r18_main126_best_nds_epoch16.pth` | 122,122,979 | `9c78d86c7f0b6bc4b5a9b0ad1a7d49fcbeed7113c5fd4df7afea0addca68830c` |

It exceeds GitHub's regular 100 MB file limit and is ignored by Git. Publish
it as a GitHub Release asset or in external object storage, and document the
download URL in the release notes. The local and source hashes remain in
[`results/checkpoints/MANIFEST.csv`](results/checkpoints/MANIFEST.csv).
The release file contains only the unchanged 657-tensor `state_dict` and
primitive metadata, so it loads with PyTorch's default safe loader. The
original full-training checkpoint SHA is retained in the manifest.

```text
results/
├── metrics/       canonical tables and histories
├── analysis/      condition, matching, attention, and trajectory CSVs
├── configs/       frozen canonical and expanded experiment configs
├── logs/          run logs and JSONL scalar histories
├── raw_metrics/   evaluator outputs, benchmarks, and historical audits
├── figures/       selected PNG summaries
└── checkpoints/   retained-checkpoint provenance
```

Frozen logs and historical configs may contain their original absolute paths.
Curated tables use relative provenance paths.

## Projection contract

The released cameras declare a 220° field of view. The reported experiments
also required camera-frame `Z < 0`, clipping rays behind the optical plane. A
deterministic 200,000-ray audit found 133,613 valid in-image rays; this
historical gate kept 99,973 and clipped 33,640 (25.1772%).

- `legacy_front_only` exactly reproduces the reported experiments and is the
  default.
- `full_fov` follows the declared lens cone and requires retraining and
  reevaluation.

Never evaluate a published checkpoint with `full_fov` without clearly
labeling the protocol change.

## Local dataset organization

Large datasets are kept outside Git under a single sibling `../datasets`
directory. In the cleaned workspace this includes `Fisheye3DODdataset`,
`kitti-360`, `snapspace_captures`, `new_dataset`, and
`lidar_fusion_export`. Only the Fisheye3DOD and real-world compatibility
symlinks remain at the workspace root because active legacy configs still use
those paths.

The LiDAR export contains all 1,611 declared LiDAR frames, but its manifest
also names 6,444 camera files that are not present. Treat it as a
LiDAR-complete, camera-incomplete export.

## Attribution and license

No separate license has been selected for ROSA-authored code. The notice below
covers the inherited Sparse4D portions only; choose a project license before a
public release.

The integration is derived from
[HorizonRobotics/Sparse4D](https://github.com/HorizonRobotics/Sparse4D) and
the [kdh044/sphere_stream_sparse4d](https://github.com/kdh044/sphere_stream_sparse4d)
lineage. Its upstream MIT notice is included here because this repository
intentionally keeps one documentation file:

> Copyright (c) 2024 Horizon Robotics
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in
> all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.
