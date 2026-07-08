# 3-class 30epoch ablation B-cleanrel: support without redundant radial cues.
#
# Same cue weights as B, but center/angle/aspect and x/y position are also
# masked from the learned view-confidence relation input. This keeps the
# ablation clean while leaving the original B result reproducible.

_base_ = ["./sparse4d_kb20_car_ped_cyc_ablation_30e_support_no_center_angle_aspect.py"]

model = dict(
    head=dict(
        deformable_model=dict(
            geometry_attention_cfg=dict(
                active_geometry_relation_names=(
                    "valid",
                    "boundary",
                    "relative_area",
                    "short_axis",
                    "anisotropy",
                ),
            ),
        ),
    ),
)

work_dir = "work_dirs/sparse4d_kb20_car_ped_cyc_ablation_30e_support_no_center_angle_aspect_cleanrel"
