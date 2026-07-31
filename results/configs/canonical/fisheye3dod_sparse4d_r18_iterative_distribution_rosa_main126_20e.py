"""R18 two-step joint ROSA with learned log-range uncertainty."""

_base_ = ["./fisheye3dod_sparse4d_r18_closed_loop_rosa_main126_20e.py"]

model = dict(
    head=dict(
        closed_loop_rosa_cfg=dict(
            _delete_=True,
            embed_dims=256,
            hidden_dims=128,
            num_keypoints=13,
            num_cameras=4,
            num_levels=4,
            radial_proposal_mode="learned_log_gaussian",
            num_refinement_steps=2,
            initial_log_range_std=0.05,
            min_log_range_std=0.01,
            max_log_range_std=0.50,
            min_candidate_range_m=0.50,
            max_candidate_range_m=48.0,
            max_log_size_delta=0.20,
            max_yaw_delta_rad=0.25,
            max_tangent_delta_m=0.75,
            max_shape2_log_size_delta=0.05,
            max_shape2_yaw_delta_rad=0.08,
            keep_threshold_m=0.5,
            shape_loss_weight=1.0,
            distribution_nll_weight=0.25,
            candidate_loss_weight=0.5,
            radial_offset_loss_weight=0.5,
            center_move_loss_weight=1.0,
            center_keep_loss_weight=1.0,
            tangent_loss_weight=0.25,
            quality_loss_weight=0.5,
            intermediate_step_loss_weight=0.5,
            max_quality_delta_logit=1.0,
            enable_shape1=True,
            enable_tangent=True,
            enable_radial=True,
            enable_shape2_feedback=True,
            enable_quality_score=True,
            detach_candidate_geometry=True,
        )
    )
)

# Every native and joint-refinement parameter participates in the two-step
# graph; avoid DDP's extra unused-parameter traversal.
find_unused_parameters = False

work_dir = (
    "/home/airlab/user/kdh/fisheye3dod_sparse4d_r50_official_20260726/"
    "work_dirs/r18_iterative_distribution_rosa_main126_20e_2gpu_b4_"
    "seed3407_val4"
)

experiment_contract = dict(
    _delete_=True,
    dataset="Fisheye3DOD",
    split="local-main126-8820-train-3906-val",
    backbone="ImageNet-pretrained ResNet-18",
    camera_model="official OCamCalib native 800x800",
    native_sparse4d_keypoints="7 fixed + 6 learnable = 13",
    extra_surface_points=0,
    observation=(
        "native 13 points -> OCam -> SH-RFL -> point-wise camera-attention ROSA"
    ),
    radial_distribution=(
        "per-query learned Gaussian over log(GT range/current range); "
        "five standard-deviation points are reprojected"
    ),
    joint_refinement=(
        "shared observation/refinement trunk is applied twice; "
        "shape, tangential centre, range distribution and confidence use "
        "separate output heads"
    ),
    update_rule=(
        "full box update -> OCam reprojection -> full box update, with "
        "shared weights across the two iterations"
    ),
    quality_rule=(
        "bounded localization-quality residual gated by detached learned "
        "radial confidence"
    ),
    temporal=False,
    max_epochs=20,
    global_batch_size=8,
    seed=3407,
)
