# 3-class 30epoch ablation: fisheye fusion support plus separated range aux.
#
# Fusion uses the learned fisheye support. Range aux deliberately does not use
# support_fusion; it uses hard-ish visibility and boundary quality only.

_base_ = ["./sparse4d_kb20_car_ped_cyc_ablation_30e_support.py"]

model = dict(
    head=dict(
        ray_range_aux=dict(
            enabled=True,
        ),
    ),
)

work_dir = "work_dirs/sparse4d_kb20_car_ped_cyc_ablation_30e_support_rangeaux"
