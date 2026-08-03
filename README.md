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
