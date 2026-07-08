# 3-class 30epoch ablation B: support without redundant radial cues.
#
# Keeps truncation/scale/short-axis/KB-anisotropy cues. Center, incidence angle,
# and projected aspect ratio are disabled and are not allocated learned quality
# weight parameters.

_base_ = ["./sparse4d_kb20_car_ped_cyc_ablation_30e_support.py"]

model = dict(
    head=dict(
        deformable_model=dict(
            geometry_attention_cfg=dict(
                center_weight=0.0,
                angle_weight=0.0,
                aspect_weight=0.0,
                learnable_quality_weight_names=(
                    "boundary_weight",
                    "relative_area_weight",
                    "short_axis_weight",
                    "anisotropy_weight",
                ),
            ),
        ),
    ),
)

work_dir = "work_dirs/sparse4d_kb20_car_ped_cyc_ablation_30e_support_no_center_angle_aspect"
