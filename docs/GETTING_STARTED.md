# Getting started

Install ROSA ([INSTALL.md](INSTALL.md)) and prepare the data ([DATA.md](DATA.md)) first.

## Checkpoints

```bash
hf download danny042/Fisheye3DOD checkpoints/rosa_fisheye3dod_r18_20e.pth --repo-type dataset --local-dir .
hf download danny042/KITTI-360-fisheye checkpoints/rosa_kitti360_vov99_24e.pth --repo-type dataset --local-dir .
```

| Benchmark | Backbone | mAP | FDS / NDS | Config | Checkpoint |
|---|---|---|---|---|---|
| Fisheye3DOD | ResNet-18, 20e | 0.6143 | 0.6647 | [config](../configs/fisheye3dod.py) | `checkpoints/rosa_fisheye3dod_r18_20e.pth` |
| KITTI-360 | VoVNet-99, 24e | 0.3114 | 0.3377 | [config](../configs/kitti360.py) | `checkpoints/rosa_kitti360_vov99_24e.pth` |

## Evaluate

```bash
python tools/run.py evaluate configs/fisheye3dod.py checkpoints/rosa_fisheye3dod_r18_20e.pth
python tools/run.py evaluate configs/kitti360.py checkpoints/rosa_kitti360_vov99_24e.pth

# multiple GPUs
bash tools/dist_test.sh CONFIG CHECKPOINT NUM_GPUS
```

## Train

```bash
bash tools/dist_train.sh configs/fisheye3dod.py 2
bash tools/dist_train.sh configs/kitti360.py 2   # needs checkpoints/fcos3d_vovnet_imgbackbone-remapped.pth (DATA.md)
```

Recipe: AdamW (lr 2e-4, weight decay 1e-3, backbone lr x0.5), 500-iteration
linear warm-up and cosine schedule, gradient clipping at 25, 2 GPUs x 4 images,
seed 3407 (`ROSA_SEED` to change). Checkpoints and logs go to `work_dirs/`.
On two RTX PRO 6000 Blackwell GPUs training takes about 9 h (Fisheye3DOD) and
7 h (KITTI-360).

## Runtime

```bash
python tools/run.py benchmark configs/fisheye3dod.py \
    --checkpoint checkpoints/rosa_fisheye3dod_r18_20e.pth --output runtime.json
```

Batch size 1, 200 warm-up and 300 timed iterations; backbone, neck and decoder
layers are compiled with `torch.compile` (`--no-compile` for eager mode). On one
RTX PRO 6000 Blackwell: 43–47 FPS compiled (43.2 in the paper), 38–40 FPS eager.

## Repository layout

```
rosa/
├── decoder.py              six-stage decoder with the hybrid (ray + log-range) box state
├── evidence.py             box keypoints, camera/level and point attention
├── cues.py                 projection cues (Jacobian-based scale and distortion)
├── rfl.py                  ray feature lifting (spherical harmonics)
├── projection.py           OCam (Fisheye3DOD) and pinhole/MEI (KITTI-360) projection
├── losses.py               auxiliary ray losses
└── integrations/           detector, head, datasets and metrics (MMDetection3D)
projects/mmdet3d_plugin/    Sparse4D components (adapted, MIT) and CUDA ops
configs/                    fisheye3dod.py (ResNet-18), kitti360.py (VoVNet-99)
assets/                     anchors/, paper figures, demo frames
tools/                      run.py (train / evaluate / benchmark), install and dist scripts
```
