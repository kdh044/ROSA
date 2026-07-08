# 3-class 30epoch ablation: fisheye fusion support only.
#
# Support affects multi-view feature aggregation at train/test time. Range
# auxiliary supervision is disabled.

_base_ = ["./sparse4d_kb20_car_ped_cyc_ablation_30e_baseline.py"]

model = dict(
    head=dict(
        deformable_model=dict(
            use_fisheye_geometry_attention=True,
        ),
    ),
)

work_dir = "work_dirs/sparse4d_kb20_car_ped_cyc_ablation_30e_support"
