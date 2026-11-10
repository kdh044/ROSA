# Train and evaluate

Install ROSA ([INSTALL.md](INSTALL.md)) and download the data and checkpoints
([DATA.md](DATA.md)) first. Run everything from the repository root with the
environment activated.

| Benchmark | Config | Checkpoint | mAP | FDS / NDS |
|---|---|---|---|---|
| Fisheye3DOD | [configs/fisheye3dod.py](../configs/fisheye3dod.py) (ResNet-18, 20 epochs) | `checkpoints/rosa_fisheye3dod_r18_20e.pth` | 0.6143 | 0.6647 |
| KITTI-360 | [configs/kitti360.py](../configs/kitti360.py) (VoVNet-99, 24 epochs) | `checkpoints/rosa_kitti360_vov99_24e.pth` | 0.3114 | 0.3377 |

## Evaluate

```bash
python tools/run.py evaluate configs/fisheye3dod.py checkpoints/rosa_fisheye3dod_r18_20e.pth
python tools/run.py evaluate configs/kitti360.py checkpoints/rosa_kitti360_vov99_24e.pth
```

With several GPUs: `bash tools/dist_test.sh CONFIG CHECKPOINT NUM_GPUS`.

## Train

```bash
bash tools/dist_train.sh configs/fisheye3dod.py 2
bash tools/dist_train.sh configs/kitti360.py 2
```

The second argument is the number of GPUs. The reported models used 2 GPUs
with 4 images each (about 9 h for Fisheye3DOD and 7 h for KITTI-360 on RTX PRO
6000 Blackwell). Checkpoints and logs are written to `work_dirs/`.
`ROSA_SEED` changes the random seed (default 3407).

## Speed

```bash
python tools/run.py benchmark configs/fisheye3dod.py \
    --checkpoint checkpoints/rosa_fisheye3dod_r18_20e.pth --output runtime.json
```

Measures batch-size-1 inference with `torch.compile` (`--no-compile` for eager
mode): 43–47 FPS on one RTX PRO 6000 Blackwell (43.2 in the paper).

## Code layout

```
rosa/
├── decoder.py              six-stage decoder with the hybrid (ray + log-range) box state
├── aggregation.py          box keypoints, camera/level and point attention
├── cues.py                 projection cues (scale and distortion from the lens Jacobian)
├── rfl.py                  ray feature lifting (spherical harmonics)
├── projection.py           OCam (Fisheye3DOD) and pinhole/MEI (KITTI-360) projection
└── integrations/           detector, head, datasets and metrics (MMDetection3D)
projects/mmdet3d_plugin/    Sparse4D components and CUDA kernels
configs/                    fisheye3dod.py, kitti360.py
tools/                      run.py, dist_train.sh, dist_test.sh, install.sh
```
