"""Clean R18 Sparse4D B0 baseline on the local Fisheye3DOD main126 split."""

import os
import pathlib

_base_ = ["./fisheye3dod_sparse4d_r18_b0_main126_30e.py"]

_package_root = pathlib.Path(
    os.environ.get(
        "FISHEYE3DOD_PACKAGE_ROOT",
        "/home/airlab/user/kdh/fisheye3dod_sparse4d_r50_official_20260726",
    )
).resolve()
_data_root = pathlib.Path(
    os.environ.get(
        "FISHEYE3DOD_DATA_ROOT",
        "/home/airlab/user/kdh/Fisheye3DODdataset",
    )
).resolve()
_ann_root = _data_root / "ImageSets-main126"

# Pin the released Main126 split in the config.  The original B0 launch used
# FISHEYE3DOD_ANN_ROOT=ImageSets-main126, but evaluation must remain
# reproducible even when that shell environment variable is absent.
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_root=str(_data_root),
        ann_file=str(_ann_root / "fisheye3dod_infos_train.pkl"),
    ),
)
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_root=str(_data_root),
        ann_file=str(_ann_root / "fisheye3dod_infos_val.pkl"),
    ),
)
test_dataloader = val_dataloader

# Two GPUs × four samples/GPU = global batch size 8.

# Keep the optimizer and warm-up unchanged. Only align cosine decay with 20 epochs.
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=20, val_interval=4)
param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1.0 / 3.0,
        by_epoch=False,
        begin=0,
        end=500,
    ),
    dict(
        type="CosineAnnealingLR",
        begin=0,
        end=20,
        T_max=20,
        by_epoch=True,
        eta_min_ratio=1e-3,
        convert_to_iter_based=True,
    ),
]

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=20,
        save_best="fisheye_bev/NDS",
        rule="greater",
    )
)

load_from = None
resume = False
work_dir = str(
    _package_root
    / "work_dirs/r18_b0_main126_20e_clean_2gpu_b4_seed3407_val4"
)

experiment_contract = dict(
    max_epochs=20,
    global_batch_size=8,
    clean_baseline=True,
)

del _package_root, _data_root, _ann_root
