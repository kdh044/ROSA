# Model zoo

Checkpoints are attached to the GitHub **Releases** page rather than committed
to the repository — Git refuses single files over 100 MB, and these are 95 MB
to 292 MB each.

| Model | Backbone | Schedule | mAP | FDS / NDS | Config | Weights |
|---|---|---|---|---|---|---|
| ROSA (Fisheye3DOD) | ResNet-18 | 20e | 0.6143 | 0.6647 | [config](../configs/fisheye3dod/rosa_r18_fisheye3dod_20e.py) | `rosa_fisheye3dod_r18_20e.pth` (95 MB) |
| ROSA (KITTI-360) | VoVNet-99 | 24e | 0.3114 | 0.3377 | [config](../configs/kitti360/rosa_vov99_kitti360_24e.py) | `rosa_kitti360_vov99_24e.pth` (268 MB) |

Supporting files on the same release:

| File | Size | Purpose |
|---|---|---|
| `fcos3d_vovnet_imgbackbone-remapped.pth` | 292 MB | FCOS3D VoVNet-99 initialization for the KITTI-360 arm |
| `kitti360_rosa_6class_infos.tar.gz` | — | `ImageSets-rosa-6class` info pickles |

## Reproducing the table

### Fisheye3DOD

```bash
export ROSA_DATA_ROOT=/path/to/Fisheye3DOD
export FISHEYE3DOD_OFFICIAL_ROOT=$PWD/external/fisheye3dod
export PYTHONPATH=$PWD:$FISHEYE3DOD_OFFICIAL_ROOT:$PYTHONPATH

torchrun --standalone --nproc_per_node=2 tools/run.py \
    configs/fisheye3dod/rosa_r18_fisheye3dod_20e.py \
    --launcher pytorch train
```

Validation runs every epoch. The reported number is the best-mAP checkpoint;
the ablation numbers in the paper are the final epoch, as stated there.

### KITTI-360

```bash
export ROSA_KITTI360_ROOT=/path/to/KITTI-360
export ROSA_VOVNET_CKPT=$PWD/weights/fcos3d_vovnet_imgbackbone-remapped.pth

torchrun --standalone --nproc_per_node=2 tools/train_kitti360.py \
    configs/kitti360/rosa_vov99_kitti360_24e.py --launcher pytorch
```

### Runtime

```bash
python tools/run.py configs/fisheye3dod/rosa_r18_fisheye3dod_20e.py \
    benchmark --checkpoint weights/rosa_fisheye3dod_r18_20e.pth \
    --output runtime.json --warmup 200 --iterations 300
```

Reported runtimes use batch size 1 with backbone and decoder layers compiled.
Warm-up matters: `torch.compile` needs a few hundred iterations before the
timing settles, which is why `--warmup 200` is not optional for a fair number.

## Training cost

Both arms were trained on two RTX PRO 6000 Blackwell GPUs at batch size 4 per
GPU. Fisheye3DOD takes roughly 9 hours for 20 epochs; KITTI-360 roughly 7
hours for 24 epochs.
