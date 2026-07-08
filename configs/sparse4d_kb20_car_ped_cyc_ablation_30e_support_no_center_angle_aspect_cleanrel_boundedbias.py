# 3-class 30epoch B-cleanrel with bounded learned reliability penalty.
#
# Replaces log(support) bias with a finite negative penalty:
#   support_bias = -max_penalty * sigmoid(MLP(geometry_relation))
# Invalid projections receive the maximum finite penalty. This removes the
# dependence on a log-confidence floor as the main source of stability.

_base_ = ["./sparse4d_kb20_car_ped_cyc_ablation_30e_support_no_center_angle_aspect_cleanrel.py"]

model = dict(
    head=dict(
        deformable_model=dict(
            geometry_attention_cfg=dict(
                support_bias_mode="bounded_penalty",
                # This is the maximum finite logit penalty. With 4.0, the worst
                # valid observation is down-weighted by roughly exp(-4).
                bias_scale=4.0,
                learnable_bias_scale=False,
                # Let the penalty MLP learn the cue combination directly.
                use_learned_view_confidence=False,
                learnable_quality_weights=False,
                bounded_penalty_init=0.05,
                min_confidence=1.0e-6,
                min_support=1.0e-6,
            ),
        ),
    ),
)

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys=dict(
            img_backbone=dict(lr_mult=0.5),
            view_penalty_mlp=dict(lr_mult=0.25),
        )
    )
)

work_dir = "work_dirs/sparse4d_kb20_car_ped_cyc_ablation_30e_support_no_center_angle_aspect_cleanrel_boundedbias"
