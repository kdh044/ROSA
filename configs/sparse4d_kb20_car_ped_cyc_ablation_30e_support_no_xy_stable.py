# 3-class 30epoch no-xy stable rerun.
#
# Keeps the interpretable fisheye quality cues learnable, including center,
# incidence angle, footprint scale, short axis, aspect, and KB anisotropy, but
# removes raw image x/y offsets from the learned confidence MLP to reduce
# dataset/camera-position bias.

_base_ = ["./sparse4d_kb20_car_ped_cyc_ablation_30e_support.py"]

model = dict(
    head=dict(
        deformable_model=dict(
            geometry_attention_cfg=dict(
                active_geometry_relation_names=(
                    "valid",
                    "boundary",
                    "center",
                    "angle",
                    "relative_area",
                    "short_axis",
                    "aspect",
                    "anisotropy",
                ),
                # Keep all quality powers learnable. x/y offsets are not quality
                # weights; they only feed the confidence MLP, and are masked out
                # above.
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

work_dir = "work_dirs/sparse4d_kb20_car_ped_cyc_ablation_30e_support_no_xy_stable"
