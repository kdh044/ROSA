"""C32 deployment arm with RGB fused into the uint8 packing copy."""

_base_ = (
    "./rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_"
    "cpu_metadata_lean_loader_uint8_transfer_pinned_inplace.py"
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
_packer = dict(type="PackFisheye3DODRGBUint8Inputs", ocam_lens_fov_deg=220.0)
_test_pipeline = [dict(_loader), dict(_packer)]

val_dataloader = dict(dataset=dict(pipeline=_test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=_test_pipeline))

model = dict(
    data_preprocessor=dict(
        type="Fisheye3DODSingleFrameInplaceDataPreprocessor",
        bgr_to_rgb=False,
    ),
)

work_dir = (
    "work_dirs/"
    "rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_"
    "cpu_metadata_lean_loader_rgb_uint8_transfer_pinned_inplace"
)

experiment_contract = dict(
    _delete_=True,
    arm="C32-runtime-RGB-uint8-fused-pack-copy",
    paired_parent="C32-runtime-owned-buffer-inplace-normalization",
    changed_operation=(
        "fuse exact BGR-to-RGB ordering into the already-required contiguous "
        "HWC-to-CHW uint8 pack copy and disable device channel conversion"
    ),
    training="fail closed; validation/test pipeline only",
    post_preprocessor_model_input="raw-bit equal required",
    learned_parameters_added=0,
    buffers_added=0,
    user_numeric_hyperparameters_added=0,
    runtime_claim="none; paired pipeline and CUDA gates pending",
)

del _test_pipeline
del _packer
del _loader
del _cameras
