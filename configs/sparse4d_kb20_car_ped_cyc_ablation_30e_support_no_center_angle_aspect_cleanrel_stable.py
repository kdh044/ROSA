# 3-class 30epoch B-cleanrel stable rerun.
#
# Keeps the clean relation cues but softens the learned gates that caused the
# previous late-epoch NaN collapse.

_base_ = ["./sparse4d_kb20_car_ped_cyc_ablation_30e_support_no_center_angle_aspect_cleanrel.py"]

model = dict(
    head=dict(
        deformable_model=dict(
            geometry_attention_cfg=dict(
                bias_scale=0.5,
                max_learnable_quality_weight=0.5,
                learned_confidence_ramp_start=1000,
                learned_confidence_ramp_iters=3000,
                min_confidence=1.0e-3,
                min_support=1.0e-3,
            ),
        ),
    ),
)

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys=dict(
            img_backbone=dict(lr_mult=0.5),
            geometry_bias_scale=dict(lr_mult=0.25),
            quality_weight_logits=dict(lr_mult=0.25),
            view_confidence_mlp=dict(lr_mult=0.25),
        )
    )
)

work_dir = "work_dirs/sparse4d_kb20_car_ped_cyc_ablation_30e_support_no_center_angle_aspect_cleanrel_stable"
