# 3-class 30epoch ablation C: minimal KB-anisotropy support.
#
# Tests whether the direct KB radial/tangential anisotropy cue alone is enough.
# The hard valid mask still applies in code, but boundary/center/angle/size/
# short-axis/aspect cues are disabled and are not allocated learned quality
# weight parameters.

_base_ = ["./sparse4d_kb20_car_ped_cyc_ablation_30e_support.py"]

model = dict(
    head=dict(
        deformable_model=dict(
            geometry_attention_cfg=dict(
                boundary_weight=0.0,
                center_weight=0.0,
                angle_weight=0.0,
                relative_area_weight=0.0,
                short_axis_weight=0.0,
                aspect_weight=0.0,
                anisotropy_weight=1.0,
                learnable_quality_weight_names=("anisotropy_weight",),
            ),
        ),
    ),
)

work_dir = "work_dirs/sparse4d_kb20_car_ped_cyc_ablation_30e_support_anisotropy_only"
