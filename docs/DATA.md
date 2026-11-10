# Data and checkpoints

Everything is on Hugging Face, in the original dataset layouts:

| Repository | Contents |
|---|---|
| [danny042/Fisheye3DOD](https://huggingface.co/datasets/danny042/Fisheye3DOD) | Fisheye3DOD (126 scenes, 163 GB), info files, ROSA checkpoint |
| [danny042/KITTI-360-fisheye](https://huggingface.co/datasets/danny042/KITTI-360-fisheye) | KITTI-360 cameras used by ROSA (94 GB), calibration, info files, ROSA checkpoint, VoVNet-99 initialization |

Run the commands below from the repository root. Nothing needs to be converted.

## Fisheye3DOD

```bash
hf download danny042/Fisheye3DOD --repo-type dataset --local-dir data/Fisheye3DOD \
    --include "scenes/*" --include "ImageSets-main126/*" --include "calib_results.txt"
cd data/Fisheye3DOD && for f in scenes/*.tar; do tar -xf "$f" && rm "$f"; done && cd -
```

Result:

```
data/Fisheye3DOD/
├── calib_results.txt
├── ImageSets-main126/            info files (8,820 train / 3,906 val frames)
└── train-Town01_Opt-.../ ...     126 scene folders
```

## KITTI-360

```bash
hf download danny042/KITTI-360-fisheye --repo-type dataset --local-dir data/KITTI-360 \
    --include "data_2d_raw/*" --include "calibration/*" --include "ImageSets-rosa-6class/*"
cd data/KITTI-360 && for f in data_2d_raw/*.tar; do tar -xf "$f" && rm "$f"; done && cd -
```

Result:

```
data/KITTI-360/
├── calibration/
├── data_2d_raw/                  two front pinhole + two side fisheye cameras, 9 drives
└── ImageSets-rosa-6class/        info files (10,999 train / 2,860 val frames, 6 classes)
```

KITTI-360 is released under CC BY-NC-SA 3.0.

## Checkpoints

```bash
# ROSA, Fisheye3DOD (ResNet-18)
hf download danny042/Fisheye3DOD checkpoints/rosa_fisheye3dod_r18_20e.pth --repo-type dataset --local-dir .
# ROSA, KITTI-360 (VoVNet-99)
hf download danny042/KITTI-360-fisheye checkpoints/rosa_kitti360_vov99_24e.pth --repo-type dataset --local-dir .
# VoVNet-99 initialization, only for KITTI-360 training (FCOS3D weights released with PETR)
hf download danny042/KITTI-360-fisheye checkpoints/fcos3d_vovnet_imgbackbone-remapped.pth --repo-type dataset --local-dir .
```

The files land in `checkpoints/`, where the configs expect them.

## Other locations

If the data is stored elsewhere, point ROSA to it:

```bash
export ROSA_DATA_ROOT=/path/to/Fisheye3DOD
export ROSA_KITTI360_ROOT=/path/to/KITTI-360
export ROSA_VOVNET_CKPT=/path/to/fcos3d_vovnet_imgbackbone-remapped.pth
```
