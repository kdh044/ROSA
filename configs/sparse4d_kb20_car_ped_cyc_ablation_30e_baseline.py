# 3-class 30epoch baseline for the fisheye view-support ablation.
#
# This is vanilla Sparse4D on the KB fisheye export: no view support and no
# ray-range auxiliary loss.

_base_ = ["./sparse4d_kb20_car_ped_cyc_100e_github_r50_b8.py"]

num_epochs = 30

param_scheduler = [
    dict(type="LinearLR", start_factor=1.0 / 3, begin=0, end=500, by_epoch=False),
    dict(type="CosineAnnealingLR", T_max=num_epochs, by_epoch=True, eta_min_ratio=1e-3),
]

train_cfg = dict(by_epoch=True, max_epochs=num_epochs, val_interval=1)

work_dir = "work_dirs/sparse4d_kb20_car_ped_cyc_ablation_30e_baseline"
