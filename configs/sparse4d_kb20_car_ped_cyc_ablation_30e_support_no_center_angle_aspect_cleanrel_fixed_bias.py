# 3-class 30epoch B diagnostic: clean core cues with fixed geometry bias scale.
#
# Tests whether the B collapse is caused by the learnable log-bias scale
# becoming unstable with the reduced geometry cue set.

_base_ = ["./sparse4d_kb20_car_ped_cyc_ablation_30e_support_no_center_angle_aspect_cleanrel.py"]

model = dict(
    head=dict(
        deformable_model=dict(
            geometry_attention_cfg=dict(
                learnable_bias_scale=False,
            ),
        ),
    ),
)

work_dir = "work_dirs/sparse4d_kb20_car_ped_cyc_ablation_30e_support_no_center_angle_aspect_cleanrel_fixed_bias"
