# 3-class 30epoch all-relation stable rerun.
#
# Keeps all fisheye geometry relation cues learnable, but uses the softer gate
# schedule that stabilized cleanrel_stable. This checks whether the dropped
# center/angle/offset/aspect cues were harmful by themselves, or only too strong.

_base_ = ["./sparse4d_kb20_car_ped_cyc_ablation_30e_support.py"]

model = dict(
    head=dict(
        deformable_model=dict(
            geometry_attention_cfg=dict(
                # active_geometry_relation_names=None keeps all relation inputs:
                # valid, boundary, center, angle, x/y offset, area, short axis,
                # aspect, and anisotropy.
                active_geometry_relation_names=None,
                # learnable_quality_weight_names=None keeps all quality weights
                # learnable: boundary, center, angle, area, short-axis, aspect,
                # and anisotropy. valid remains a hard visibility mask; x/y
                # offsets are relation inputs to the learned confidence MLP.
                learnable_quality_weight_names=None,
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

work_dir = "work_dirs/sparse4d_kb20_car_ped_cyc_ablation_30e_support_allrel_stable"
