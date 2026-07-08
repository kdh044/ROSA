# 3-class 30epoch B diagnostic: clean core cues without learned view confidence.
#
# Tests whether the B collapse is caused by the learned view-confidence MLP
# interacting with the reduced geometry cue set.

_base_ = ["./sparse4d_kb20_car_ped_cyc_ablation_30e_support_no_center_angle_aspect_cleanrel.py"]

model = dict(
    head=dict(
        deformable_model=dict(
            geometry_attention_cfg=dict(
                use_learned_view_confidence=False,
            ),
        ),
    ),
)

work_dir = "work_dirs/sparse4d_kb20_car_ped_cyc_ablation_30e_support_no_center_angle_aspect_cleanrel_no_mlp"
