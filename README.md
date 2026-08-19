# ROSA: Re-Observation for Native-Fisheye 3D Detection

ROSA is a temporal-free 3D detector that repeatedly re-observes current 3D box
keypoints in calibrated native-fisheye feature maps. It retains Sparse4D's
300-query, six-stage recurrence while replacing each stage's image observation
and box update with native-fisheye geometry, camera-wise ray feature lifting,
and task-isolated refinement.

## Main result

Fisheye3DOD Main126 validation, 3,906 frames, official center-distance metric:

| Method | Backbone | Temporal | mAP | FDS | mATE ↓ | mASE ↓ | mAOE ↓ |
|---|---|---|---:|---:|---:|---:|---:|
| FisheyePETR | R18 | Off | 0.3883 | 0.4842 | 0.7110 | 0.1413 | 0.4075 |
| FisheyeBEVDet | R18 | Off | 0.3987 | 0.5000 | 0.5821 | 0.1616 | 0.4523 |
| DAPETR | VoVNet-99 | Off | 0.4396 | 0.5047 | 0.7319 | 0.1598 | 0.3987 |
| Sparse4D | R18 | Off | 0.5644 | 0.6342 | 0.4623 | 0.0858 | **0.3402** |
| **ROSA (Ours)** | R18 | Off | **0.6143** | **0.6647** | **0.4289** | **0.0788** | 0.3469 |

Here, Sparse4D is the paired temporal-off baseline adapted to the same four
native fisheye cameras, R18-FPN, 300 queries, raw-uniform training, and official
evaluator. It contains no ROSA observation or refinement module. FisheyePETR
and FisheyeBEVDet use their authors' released checkpoints; all rows use the
same Main126 validation annotation and evaluator.

ROSA epoch 16 is selected because it has the highest validation mAP and FDS
among epochs 4, 8, 12, 16, and 20. The experiment uses one seed (3407), so it
does not estimate cross-seed variance. Main126 follows the dataset's within-
sequence split and is not a scene-disjoint held-out claim. Exact values and
artifact hashes are in [results/metrics/main126_final.json](results/metrics/main126_final.json).

## Runtime

Single-GPU inference on an NVIDIA RTX PRO 6000 Blackwell Server Edition,
batch size 1, 50 warm-up iterations, and 300 measured iterations:

| Method | Latency (ms) ↓ | P95 (ms) ↓ | FPS ↑ | Peak allocated GPU memory (MiB) ↓ |
|---|---:|---:|---:|---:|
| FisheyeBEVDet | **16.16** | **16.61** | **61.88** | 1086.9 |
| FisheyePETR | 22.54 | 22.79 | 44.37 | 800.2 |
| Sparse4D | 27.42 | 27.78 | 36.48 | 776.8 |
| ROSA (Ours) | 48.20 | 49.20 | 20.75 | **749.8** |

The timed region is the complete MMEngine `model.test_step`, synchronized
before and after each iteration; fetching the next dataloader batch is outside
the timer. Peak memory is CUDA allocated memory with the loaded model resident,
not reserved memory. Exact outputs, artifact hashes, environment, and the known
detector-host worktree condition are recorded in
[results/raw_metrics/benchmarks/20260819_main126_rtxpro6000_b1/protocol.json](results/raw_metrics/benchmarks/20260819_main126_rtxpro6000_b1/protocol.json).

## Method

Each D0-D5 stage performs one recurrent update:

```text
Sparse4D query + current Cartesian box
  -> query self-attention
  -> 7 semantic + 6 learned current-box points
  -> native OCam projection into 4 cameras and 4 FPN levels
  -> camera-level evidence aggregation
  -> camera-wise appearance-by-ray RFL
  -> ROSA cross-camera fusion
  -> point-wise RFL + query-conditioned 13-point attention
  -> FFN
  -> gated center, size, yaw, and velocity refinement
```

Center is represented internally as unit ray direction plus log range. Size is
updated in log space, yaw by rotation composition, and velocity additively in
the global frame. Each decoder stage has an independent full-capacity
task-isolated log-size expert. Camera-wise RFL is applied before ROSA camera
fusion; point-wise RFL remains after fusion. D5 supplies inference outputs.

The model uses native 800x800 fisheye images and the calibrated OCam lens cone.
It has no temporal cache, pre-D0 image observation, post-D5 sidecar, seventh
decoder stage, 2D detector, dense depth head, or BEV conversion. The released
configuration adds no hand-set update scale, physical clamp, class prior, or
extra image read.

## Repository

```text
ROSA/
├── assets/          released 300-query anchor
├── configs/         one flattened final Main126 configuration
├── results/         metrics, analyses, figures, and checkpoint manifest
├── src/rosa/        ROSA core and framework integrations
├── tests/           CPU and integration regression tests
├── tools/           MMEngine train/evaluate runner
└── weights/         ignored local checkpoint location
```

Historical result files remain as evidence, but
`configs/rosa_r18_main126.py` and `results/metrics/main126_final.json` are
the authoritative final entries.

## Setup

Python 3.10 or newer and PyTorch 2.1 or newer are required. Core-only
development does not require MMDetection3D:

```bash
python -m pip install -e ".[dev]"
pytest -q
```

Training additionally needs the MMEngine Sparse4D host and the official
Fisheye3DOD evaluator:

```text
ROSA/
└── external/
    ├── detector_host/
    └── fisheye3dod/
../datasets/
└── Fisheye3DODdataset/
    └── ImageSets-main126/
        ├── fisheye3dod_infos_train.pkl
        └── fisheye3dod_infos_val.pkl
```

The official evaluator revision is public:

```bash
git clone https://github.com/weiyangdaren/Fisheye3DOD.git external/fisheye3dod
git -C external/fisheye3dod checkout a28726ed002a02cb1328960848c0f4f2473748de
python -m pip install -e ".[integration,dev]"
(
  cd external/detector_host/projects/mmdet3d_plugin/ops
  python setup.py build_ext --inplace
)
```

The former detector-host URL, `kdh044/sphere_stream_sparse4d`, now redirects
to this ROSA repository and must not be used as a host clone. The final fused
model was validated from detector-host base commit
`e0817a30b90ec22f9a9d02952d45ad734de2468c` plus a four-file per-camera
deformable-aggregation patch. That minimal patch builds and runs the released
checkpoint, but it still needs a stable public host repository before the
integration can be called fully reproducible. Its exact file list and diff
hash are recorded in the runtime protocol linked above.

The config supports `ROSA_DETECTOR_HOST_ROOT`, `SPARSE4D_ROOT`,
`FISHEYE3DOD_OFFICIAL_ROOT`, `FISHEYE3DOD_DATA_ROOT`,
`FISHEYE3DOD_ANN_ROOT`, `FISHEYE3DOD_ANCHOR_ROOT`,
`FISHEYE3DOD_ANCHOR`, and `ROSA_WORK_DIR`.

## Train and evaluate

Single GPU:

```bash
python tools/run.py train configs/rosa_r18_main126.py
```

Released two-GPU global batch of 8:

```bash
torchrun --nproc-per-node=2 tools/run.py train \
  configs/rosa_r18_main126.py --launcher pytorch
```

Evaluate the selected checkpoint:

```bash
python tools/run.py evaluate \
  configs/rosa_r18_main126.py \
  weights/rosa_r18_main126_best_fds_epoch16.pth
```

Reproduce the ROSA runtime measurement with batch size 1:

```bash
python tools/benchmark.py \
  configs/rosa_r18_main126.py \
  weights/rosa_r18_main126_best_fds_epoch16.pth \
  --output results/raw_metrics/benchmarks/local/rosa.json \
  --warmup 50 --iterations 300 \
  --cfg-options test_dataloader.batch_size=1 \
    test_dataloader.num_workers=0 \
    test_dataloader.persistent_workers=False
```

## Checkpoint

The selected best-FDS checkpoint is not stored in Git because it exceeds
GitHub's regular file limit.

| File | Bytes | SHA-256 |
|---|---:|---|
| `weights/rosa_r18_main126_best_fds_epoch16.pth` | 108,760,817 | `98580a5f9a0e50d7921e9d718361da4f98980e530e153507082a2f1899809afc` |

The complete archival checkpoint and alternate best-mAP artifact hashes are
recorded in [results/metrics/main126_final.json](results/metrics/main126_final.json)
and [results/checkpoints/MANIFEST.csv](results/checkpoints/MANIFEST.csv).

## Evaluation protocol

The official metric averages class AP at center-distance thresholds
0.5, 1, 2, and 4 meters. FDS combines mAP with mATE, mASE, and mAOE. The
evaluation range is X/Y in [-48, 48] meters and Z in [-5, 5] meters.

The released cameras declare a 220-degree field of view. The final ROSA
projector uses the calibrated native OCam projection, in-image bounds, and the
declared lens cone; it does not impose a pinhole front-plane gate.

## Attribution and license

The integration is derived from HorizonRobotics/Sparse4D and the
`kdh044/sphere_stream_sparse4d` lineage. No separate license has yet been
selected for ROSA-authored code. Choose a project license before making the
repository public.
