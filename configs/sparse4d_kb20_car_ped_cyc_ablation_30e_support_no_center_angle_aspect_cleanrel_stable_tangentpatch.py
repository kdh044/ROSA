# 3-class 30epoch object-centric local tangent patch ablation.
#
# Each Sparse4D object keypoint is expanded into a small camera-specific
# angular patch on the local viewing tangent plane, then reprojected with the
# fisheye model and aggregated back to the original keypoint feature shape.
# This keeps the detector sparse while making feature extraction explicitly
# local-fisheye-geometry aware.

_base_ = [
    "./sparse4d_kb20_car_ped_cyc_ablation_30e_support_no_center_angle_aspect_cleanrel_stable.py"
]

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

work_dir = "work_dirs/sparse4d_kb20_car_ped_cyc_ablation_30e_support_no_center_angle_aspect_cleanrel_stable_tangentpatch"
