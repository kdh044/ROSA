"""C32 runtime arm with compact uint8 host-to-device image transfer."""

_base_ = (
    "./rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_"
    "cpu_metadata_lean_loader.py"
)

custom_imports = dict(
    imports=[
        "projects.mmdet3d_plugin",
        "rosa.integrations.sparse4d",
        "rosa.integrations.optim",
        "rosa.integrations.fisheye3dod.dataset",
        "rosa.integrations.fisheye3dod.model",
        "rosa.integrations.fisheye3dod.metric",
        "rosa.integrations.fisheye3dod.runtime_transforms",
    ],
    allow_failed_imports=False,
)

_cameras = (
    "fisheye_camera_front",
    "fisheye_camera_left",
    "fisheye_camera_right",
    "fisheye_camera_rear",
)
_loader = dict(
    type="LoadOfficialFisheye3DODLeanForSparse4D",
    cam_names=_cameras,
    to_float32=False,
)
_packer = dict(type="PackFisheye3DODUint8Inputs", ocam_lens_fov_deg=220.0)
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
    dict(_packer),
]
_test_pipeline = [dict(_loader), dict(_packer)]

train_dataloader = dict(
    dataset=dict(dataset=dict(pipeline=_train_pipeline)),
)
val_dataloader = dict(dataset=dict(pipeline=_test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=_test_pipeline))

work_dir = (
    "work_dirs/"
    "rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_"
    "cpu_metadata_lean_loader_uint8_transfer"
)

experiment_contract = dict(
    _delete_=True,
    arm="C32-runtime-single-frame-uint8-image-transfer",
    changed_operation=(
        "retain released uint8 image tensors through packing and cast with the "
        "existing device preprocessor"
    ),
    preprocessed_model_inputs="bitwise unchanged",
    host_image_bytes_per_batch4_parent=122880000,
    host_image_bytes_per_batch4_candidate=30720000,
    theoretical_h2d_byte_reduction_per_batch4=92160000,
    gpu_cast_cost="unmeasured",
    peak_memory_effect="unmeasured",
    runtime_claim="none; paired CUDA profiler and latency gates pending",
    learned_parameters_added=0,
    buffers_added=0,
    user_numeric_hyperparameters_added=0,
)

del _train_pipeline
del _test_pipeline
del _packer
del _loader
del _cameras
