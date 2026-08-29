# Model zoo

Checkpoints are attached to the [Releases](../../releases) page rather than
committed to the repository — Git refuses single files over 100 MB. They are
plain HTTP downloads, so `wget` and `curl` work without authentication:

```bash
BASE=https://github.com/kdh044/ROSA/releases/download/v1.0.0
wget $BASE/rosa_fisheye3dod_r18_20e.pth
```

Optimizer state is stripped from every file; these are inference weights.

## ROSA

| Benchmark | Backbone | Schedule | mAP | FDS / NDS | Config | Weights |
|---|---|---|---|---|---|---|
| Fisheye3DOD | ResNet-18 | 20e | 0.6143 | 0.6647 | [config](../configs/fisheye3dod/rosa_r18_fisheye3dod_20e.py) | `rosa_fisheye3dod_r18_20e.pth` |
| KITTI-360 | VoVNet-99 | 24e | 0.3114 | 0.3377 | [config](../configs/kitti360/rosa_vov99_kitti360_24e.py) | `rosa_kitti360_vov99_24e.pth` |

## Baselines

Every baseline was retrained from scratch under the same recipe as ROSA, so
the gap between rows reflects the detector rather than the training budget.
Configs are under [`benchmarks/`](../benchmarks/).

### Fisheye3DOD (ResNet-18, 20 epochs)

| Method | mAP | FDS | Weights |
|---|---|---|---|
| PETR | 0.3303 | 0.4444 | `petr_fisheye3dod_r18_20e.pth` |
| SRCN3D | 0.3366 | 0.4364 | `srcn3d_fisheye3dod_r18_20e.pth` |
| DAPETR | 0.3604 | 0.4581 | `dapetr_fisheye3dod_r18_20e.pth` |
| Graph-DETR3D | 0.3876 | 0.4781 | `graphdetr3d_fisheye3dod_r18_20e.pth` |
| RepDETR3D | 0.4089 | 0.5012 | `repdetr3d_fisheye3dod_r18_20e.pth` |
| DETR3D | 0.4273 | 0.4976 | `detr3d_fisheye3dod_r18_20e.pth` |
| SparseBEV | 0.4696 | 0.5656 | `sparsebev_fisheye3dod_r18_20e.pth` |
| Sparse4D | 0.5644 | 0.6342 | `sparse4d_fisheye3dod_r18_20e.pth` |
| StreamPETR | — | — | `streampetr_fisheye3dod_r18_20e.pth` |

FisheyePETR and FisheyeBEVDet are released by the Fisheye3DOD authors; get
those two from the [original
repository](https://github.com/weiyangdaren/Fisheye3DOD) rather than here.

### KITTI-360 (VoVNet-99, 24 epochs)

| Method | mAP | NDS | Weights |
|---|---|---|---|
| DAPETR | 0.192 | 0.207 | `dapetr_kitti360_vov99_24e.pth` |
| PETR | 0.205 | 0.227 | `petr_kitti360_vov99_24e.pth` |
| SRCN3D | 0.227 | 0.254 | `srcn3d_kitti360_vov99_24e.pth` |
| Graph-DETR3D | 0.233 | 0.269 | `graphdetr3d_kitti360_vov99_24e.pth` |
| DETR3D | 0.234 | 0.270 | `detr3d_kitti360_vov99_24e.pth` |
| RepDETR3D | 0.261 | 0.289 | `repdetr3d_kitti360_vov99_24e.pth` |
| SparseBEV | 0.268 | 0.300 | `sparsebev_kitti360_vov99_24e.pth` |
| Sparse4D | 0.279 | 0.300 | `sparse4d_kitti360_vov99_24e.pth` |

## Supporting files

| File | Purpose |
|---|---|
| `fcos3d_vovnet_imgbackbone-remapped.pth` | FCOS3D VoVNet-99 initialization for the KITTI-360 arm |
| `kitti360_rosa_6class_infos.tar.gz` | `ImageSets-rosa-6class` info pickles (10,999 train / 2,860 val) |

## Reproducing the tables

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

Set `ROSA_SEED` to repeat any run under a different draw; it defaults to 3407.

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
