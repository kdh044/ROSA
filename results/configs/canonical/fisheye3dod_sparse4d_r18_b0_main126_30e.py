"""R18 Sparse4D B0 production arm on the local Fisheye3DOD main126 split."""

import pathlib
import os

_base_ = ["./_fisheye3dod_sparse4d_r50_b0_base.py"]

_package_root = pathlib.Path(
    os.environ.get(
        "FISHEYE3DOD_PACKAGE_ROOT",
        "/home/airlab/user/kdh/"
        "fisheye3dod_sparse4d_r50_official_20260726",
    )
).resolve()

# Fisheye3DOD's released camera-only baselines use ImageNet-pretrained R18.
# Sparse4D still consumes four FPN levels, so expose all four R18 stages.
model = dict(
    img_backbone=dict(
        _delete_=True,
        type="mmdet.ResNet",
        depth=18,
        num_stages=4,
        frozen_stages=-1,
        norm_eval=False,
        style="pytorch",
        with_cp=False,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type="BN", requires_grad=True),
        init_cfg=dict(
            type="Pretrained",
            checkpoint="torchvision://resnet18",
        ),
    ),
    img_neck=dict(in_channels=[64, 128, 256, 512]),
)

load_from = None
resume = False
work_dir = str(_package_root / "work_dirs/r18_b0_main126_30e_seed3407")

experiment_contract = dict(
    _delete_=True,
    dataset="Fisheye3DOD",
    split="local-main126-8820-train-3906-val",
    backbone="ImageNet-pretrained ResNet-18",
    classes=(
        "Car",
        "Van",
        "Truck",
        "Bus",
        "Pedestrian",
        "Cyclist",
    ),
    camera_model="official OCamCalib native 800x800",
    gravity_center_boxes=True,
    anchor_z_is_bottom=False,
    temporal=False,
    decoder_count=6,
    query_and_output_budget=300,
    max_epochs=30,
    seed=3407,
)

del _package_root
