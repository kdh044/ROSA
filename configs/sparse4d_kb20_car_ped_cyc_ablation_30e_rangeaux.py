# 3-class 30epoch ablation: range auxiliary supervision only.
#
# There is no fusion support. The aux loss is weighted only by hard-ish
# visibility and boundary quality, not by learned confidence or fusion support.

_base_ = ["./sparse4d_kb20_car_ped_cyc_ablation_30e_baseline.py"]

model = dict(
    head=dict(
        ray_range_aux=dict(
            enabled=True,
        ),
    ),
)

work_dir = "work_dirs/sparse4d_kb20_car_ped_cyc_ablation_30e_rangeaux"
