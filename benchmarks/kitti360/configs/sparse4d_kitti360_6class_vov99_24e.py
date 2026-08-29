"""Sparse4D side of the 24-epoch from-scratch pilot; see the ROSA config."""

_base_ = ["../sparse4d_kitti360_6class_vov99_20e.py"]

train_cfg = dict(max_epochs=24)

param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=500,
        start_factor=1.0 / 3.0,
        type="LinearLR"),
    dict(
        T_max=24,
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=24,
        eta_min_ratio=0.001,
        type="CosineAnnealingLR"),
]
