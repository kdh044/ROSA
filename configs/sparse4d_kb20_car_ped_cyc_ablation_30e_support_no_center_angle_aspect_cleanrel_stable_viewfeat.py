# 3-class 30epoch B-cleanrel stable + camera object feature confidence.
#
# Disables the fused CUDA aggregation path so per-camera object features can be
# summarized before final fusion and fed to the view-confidence head.

_base_ = ["./sparse4d_kb20_car_ped_cyc_ablation_30e_support_no_center_angle_aspect_cleanrel_stable.py"]

model = dict(
    use_deformable_func=False,
    head=dict(
        deformable_model=dict(
            use_deformable_func=False,
            geometry_attention_cfg=dict(
                use_view_feature_confidence=True,
            ),
        ),
    ),
)

work_dir = "work_dirs/sparse4d_kb20_car_ped_cyc_ablation_30e_support_no_center_angle_aspect_cleanrel_stable_viewfeat"
