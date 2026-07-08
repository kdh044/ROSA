# 3-class 30epoch object-centric local tangent patch only ablation.
#
# This isolates explicit query-local fisheye patch sampling from the learned
# geometry-aware view/keypoint fusion. It is the counterpart to cleanrel_stable
# for checking whether explicit local reprojection and learned geometry fusion
# are complementary or redundant.

_base_ = ["./sparse4d_kb20_car_ped_cyc_ablation_30e_baseline.py"]

model = dict(
    use_deformable_func=False,
    head=dict(
        deformable_model=dict(
            use_deformable_func=False,
            geometry_attention_cfg=dict(
                use_tangent_patch_sampling=True,
            ),
        ),
    ),
)

work_dir = "work_dirs/sparse4d_kb20_car_ped_cyc_ablation_30e_baseline_tangentpatch"
