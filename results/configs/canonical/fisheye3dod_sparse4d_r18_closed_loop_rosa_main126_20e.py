"""R18 final 13-keypoint OCam RFL/ROSA closed-loop experiment."""

_base_ = ["./fisheye3dod_sparse4d_r18_b0_main126_20e_clean.py"]

_data_root = "/home/airlab/user/kdh/Fisheye3DODdataset"
_ann_root = f"{_data_root}/ImageSets-main126"

# Pin the released 126-scene split in the experiment itself instead of
# depending on the launch shell's FISHEYE3DOD_ANN_ROOT environment variable.
train_dataloader = dict(
    dataset=dict(
        data_root=_data_root,
        ann_file=f"{_ann_root}/fisheye3dod_infos_train.pkl",
    )
)
val_dataloader = dict(
    dataset=dict(
        data_root=_data_root,
        ann_file=f"{_ann_root}/fisheye3dod_infos_val.pkl",
    )
)
test_dataloader = val_dataloader

_decoder0 = ["deformable", "ffn", "norm", "refine"]
_decoder_later = [
    "gnn",
    "norm",
    "deformable",
    "ffn",
    "norm",
    "refine",
]

custom_imports = dict(
    _delete_=True,
    imports=[
        "projects.mmdet3d_plugin",
        "fisheye3dod_official.dataset",
        "fisheye3dod_official.model",
        "fisheye3dod_official.metric",
        "final_surface_rosa_rfl",
    ],
    allow_failed_imports=False,
)

model = dict(
    head=dict(
        type="ClosedLoopROSAHead",
        num_decoder=6,
        operation_order=_decoder0 + _decoder_later * 5,
        cache_decoder_index=5,
        inference_output_decoder_index=5,
        inference_classification_decoder_index=5,
        inference_quality_decoder_index=5,
        inference_box_source_by_dimension=None,
        # Compatibility validation for the inherited late-head host only.
        # ClosedLoopROSAHead does not construct or sample these surface cells.
        canonical_task_splice_cfg=dict(
            contract_version=(
                "late-canonical-geometry/2026-07-25/"
                "v15-native-joint-score-residual"
            ),
            num_classes=6,
            patch_rows=1,
            patch_cols=1,
            min_patch_samples=1,
            projection_type="ocam_fisheye",
            anchor_z_is_bottom=False,
            generate_fisheye_mask=False,
            point_cloud_xy_half_extent=(48.0, 48.0),
            min_projected_fpn_cells=1.0,
        ),
        closed_loop_rosa_cfg=dict(
            embed_dims=256,
            hidden_dims=128,
            num_keypoints=13,
            num_cameras=4,
            num_levels=4,
            radial_step_m=0.5,
            candidate_target_sigma_m=0.5,
            max_log_size_delta=0.20,
            max_yaw_delta_rad=0.25,
            max_tangent_delta_m=0.75,
            max_shape2_log_size_delta=0.05,
            max_shape2_yaw_delta_rad=0.08,
            keep_threshold_m=0.5,
            shape_loss_weight=1.0,
            candidate_loss_weight=0.5,
            radial_offset_loss_weight=0.5,
            center_move_loss_weight=1.0,
            center_keep_loss_weight=1.0,
            tangent_loss_weight=0.25,
            quality_loss_weight=0.5,
            enable_shape1=True,
            enable_tangent=True,
            enable_radial=True,
            enable_shape2_feedback=True,
            enable_quality_score=True,
            detach_candidate_geometry=True,
        ),
    ),
)

# The appended output intentionally omits native output-6 classification,
# centerness and DN losses, so DDP must tolerate those absent native routes.
find_unused_parameters = True

work_dir = (
    "/home/airlab/user/kdh/fisheye3dod_sparse4d_r50_official_20260726/"
    "work_dirs/r18_closed_loop_rosa_main126_20e_2gpu_b4_seed3407_val4"
)

experiment_contract = dict(
    _delete_=True,
    dataset="Fisheye3DOD",
    split="local-main126-8820-train-3906-val",
    backbone="ImageNet-pretrained ResNet-18",
    camera_model="official OCamCalib native 800x800",
    native_sparse4d_keypoints="7 fixed + 6 learnable = 13",
    extra_surface_points=0,
    initial_flow=(
        "13 native keypoints -> OCam FPN observations -> SH-RFL -> "
        "query-conditioned camera-softmax ROSA"
    ),
    shape_center_flow=(
        "shape1 + tangent -> detached five radial hypotheses -> "
        "cross-camera candidate agreement -> uncertainty-gated radial update"
    ),
    feedback_flow=(
        "posterior-weighted candidate feature -> bounded shape2 + "
        "localization-quality rank residual"
    ),
    radial_candidate_offsets=(-2, -1, 0, 1, 2),
    radial_step_m=0.5,
    keep_threshold_m=0.5,
    temporal=False,
    max_epochs=20,
    global_batch_size=8,
    seed=3407,
)

del _data_root, _ann_root, _decoder0, _decoder_later
