# Data preparation

ROSA reads two benchmarks. Neither is redistributed here — download them from
their original sources and place them under `data/`, or point the environment
variables at wherever they already live.

```
data/
├── Fisheye3DOD/
└── KITTI-360/
```

## Fisheye3DOD

Download from the [official
repository](https://github.com/weiyangdaren/Fisheye3DOD). The expected layout
is the one the release ships:

```
data/Fisheye3DOD/
├── calib_results.txt
├── ImageSets-main126/
│   ├── fisheye3dod_infos_train.pkl
│   ├── fisheye3dod_infos_val.pkl
│   └── fisheye3dod_sensor_transform.pkl
├── train-Town01_Opt-ClearNight-.../
├── train-Town01_Opt-ClearNoon-.../
└── ...
```

We evaluate on 3,906 validation frames from 126 scenes, scoring objects
within 48 m. If your dataset is elsewhere:

```bash
export ROSA_DATA_ROOT=/path/to/Fisheye3DOD
```

The anchor file shipped in `assets/` is the 300-cluster k-means anchor set fit
on the training split with seed 3407. It is what the released config
uses; regenerating it is unnecessary.

## KITTI-360

Download from the [KITTI-360 website](https://www.cvlibs.net/datasets/kitti-360/).
ROSA uses the two perspective cameras and the two 205° MEI fisheyes, so you
need `data_2d_raw`, `data_3d_bboxes` and `calibration`. `data_3d_raw` is only
required if you want to render the LiDAR BEV figures.

```
data/KITTI-360/
├── calibration/
├── data_2d_raw/
├── data_3d_bboxes/
└── ImageSets-rosa-6class/
    ├── kitti360_rosa_infos_train.pkl
    └── kitti360_rosa_infos_val.pkl
```

```bash
export ROSA_KITTI360_ROOT=/path/to/KITTI-360
```

### Info files

The `ImageSets-rosa-6class` pickles convert KITTI-360 into the nuScenes-style
format the metric expects: six classes (car, truck, trailer, motorcycle,
bicycle, pedestrian), 2 Hz keyframes, 10,999 training and 2,860 validation
samples drawn from nine drives.

They are released alongside the checkpoints — see
[MODEL_ZOO.md](MODEL_ZOO.md) — but they can also be rebuilt from the raw
dataset:

```bash
export ROSA_KITTI360_ROOT=/path/to/KITTI-360

# parses the 3D bounding-box XML and the perspective/MEI calibration,
# and writes kitti360_rosa_infos_{train,val}.pkl
python tools/data_converter/build_kitti360_infos.py

# k-means anchor initialization over the training box centers
python tools/data_converter/build_kitti360_anchor.py \
    --train-pkl $ROSA_KITTI360_ROOT/ImageSets-rosa-6class/kitti360_rosa_infos_train.pkl \
    --num-anchor 300 --seed 3407 \
    --output assets/kitti360_rosa_train_kmeans300_seed3407.npy
```

`tools/data_converter/kitti360_mixed_camera.py` holds the standalone
perspective/MEI camera geometry the converter uses; it is the same projection
the detector runs at training time, so the info files and the model agree by
construction.

The Fisheye3DOD info pickles ship with that benchmark and need no conversion;
its anchor file is already in `assets/`.

### Pretrained backbone

The KITTI-360 arm starts from the FCOS3D-pretrained VoVNet-99 backbone
released with Fisheye3DOD, remapped so its tensors load under the
`img_backbone.` prefix. Download it from the release page and place it at:

```
weights/fcos3d_vovnet_imgbackbone-remapped.pth
```

or set `ROSA_VOVNET_CKPT`. Training without it is possible but will not
reproduce the reported numbers.

## Evaluation ranges

| Benchmark | Classes | Range |
|---|---|---|
| Fisheye3DOD | Car, Van, Truck, Bus, Pedestrian, Cyclist | 48 m |
| KITTI-360 | car, truck, trailer | 50 m |
| KITTI-360 | pedestrian, bicycle, motorcycle | 40 m |
