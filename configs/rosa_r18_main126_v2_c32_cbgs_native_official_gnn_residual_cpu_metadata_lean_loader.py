"""C32 runtime arm with the unused camera-pose inverse removed."""

_base_ = "./rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_cpu_metadata.py"

_cameras = (
    "fisheye_camera_front",
    "fisheye_camera_left",
    "fisheye_camera_right",
    "fisheye_camera_rear",
)
_loader = dict(
    type="LoadOfficialFisheye3DODLeanForSparse4D",
    cam_names=_cameras,
    to_float32=True,
)
_train_pipeline = [
    dict(_loader),
    dict(
        type="LoadAnnotations3D",
        with_bbox=False,
        with_bbox_3d=True,
        with_label=False,
        with_label_3d=True,
    ),
    dict(type="Fisheye3DODEvaluatorRangeFilter"),
    dict(type="PackSparse4DInputs", ocam_lens_fov_deg=220.0),
]
_test_pipeline = [
    dict(_loader),
    dict(type="PackSparse4DInputs", ocam_lens_fov_deg=220.0),
]

train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            pipeline=_train_pipeline,
        ),
    ),
)
val_dataloader = dict(dataset=dict(pipeline=_test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=_test_pipeline))

work_dir = (
    "work_dirs/"
    "rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_"
    "cpu_metadata_lean_loader"
)

experiment_contract = dict(
    arm="C32-runtime-single-frame-CPU-metadata-lean-loader",
    lean_loader=True,
    changed_loader_operation=(
        "skip the cam2ego inverse that the exact train/test pipelines never pack"
    ),
    packed_model_inputs="bitwise unchanged",
    loader_runtime_claim="none; end-to-end dataloader throughput unmeasured",
    learned_parameters_added=0,
    buffers_added=0,
    user_numeric_hyperparameters_added=0,
)

del _train_pipeline
del _test_pipeline
del _loader
del _cameras
