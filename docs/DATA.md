# Data preparation

Both benchmarks are mirrored on Hugging Face in their original layout,
together with the info files ROSA reads. Nothing has to be converted: download,
extract, and point ROSA at the folder.

```
data/
├── Fisheye3DOD/
└── KITTI-360/
```

Use `data/` inside the repository, or set `ROSA_DATA_ROOT` /
`ROSA_KITTI360_ROOT` to the folders.

## Fisheye3DOD

Source: [Fisheye3DOD](https://github.com/weiyangdaren/Fisheye3DOD), mirrored at
[`danny042/Fisheye3DOD`](https://huggingface.co/datasets/danny042/Fisheye3DOD)
(126 scenes, 163 GB).

```bash
hf download danny042/Fisheye3DOD --repo-type dataset --local-dir data/Fisheye3DOD \
    --include "scenes/*" --include "ImageSets-main126/*" --include "calib_results.txt"
cd data/Fisheye3DOD && for f in scenes/*.tar; do tar -xf "$f" && rm "$f"; done && cd -
```

```
data/Fisheye3DOD/
├── calib_results.txt
├── ImageSets-main126/
│   ├── fisheye3dod_infos_train.pkl      8,820 frames
│   └── fisheye3dod_infos_val.pkl        3,906 frames
├── train-Town01_Opt-ClearNight-.../
└── ...                                  126 scene folders
```

Evaluation uses the official Fisheye3DOD metric (objects within the class
ranges, up to 48 m). The anchors are in `assets/anchors/`.

## KITTI-360

Source: [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/) (CC BY-NC-SA
3.0), mirrored at
[`danny042/KITTI-360-fisheye`](https://huggingface.co/datasets/danny042/KITTI-360-fisheye).
ROSA uses the two rectified perspective cameras and the two MEI fisheyes
(94 GB); the LiDAR folder of the mirror is not needed.

```bash
hf download danny042/KITTI-360-fisheye --repo-type dataset --local-dir data/KITTI-360 \
    --include "data_2d_raw/*" --include "calibration/*" --include "ImageSets-rosa-6class/*"
cd data/KITTI-360 && for f in data_2d_raw/*.tar; do tar -xf "$f" && rm "$f"; done && cd -
```

```
data/KITTI-360/
├── calibration/
├── data_2d_raw/2013_05_28_drive_*_sync/image_0{0,1,2,3}/
└── ImageSets-rosa-6class/
    ├── kitti360_rosa_infos_train.pkl    10,999 frames
    └── kitti360_rosa_infos_val.pkl       2,860 frames
```

The info files hold six classes (car, truck, trailer, motorcycle, bicycle,
pedestrian) at 2 Hz from nine drives, in the nuScenes-style format of the
metric. The anchors (300 k-means centres of the training boxes) are in
`assets/anchors/`.

## Pretrained weights

All weights are on Hugging Face, next to the datasets:

```bash
# ROSA checkpoints
hf download danny042/Fisheye3DOD checkpoints/rosa_fisheye3dod_r18_20e.pth --repo-type dataset --local-dir .
hf download danny042/KITTI-360-fisheye checkpoints/rosa_kitti360_vov99_24e.pth --repo-type dataset --local-dir .
# VoVNet-99 initialization for KITTI-360 training (FCOS3D weights from PETR)
hf download danny042/KITTI-360-fisheye checkpoints/fcos3d_vovnet_imgbackbone-remapped.pth --repo-type dataset --local-dir .
```

The ResNet-18 ImageNet weights for Fisheye3DOD training are fetched
automatically by torchvision. `ROSA_VOVNET_CKPT` overrides the VoVNet path.
