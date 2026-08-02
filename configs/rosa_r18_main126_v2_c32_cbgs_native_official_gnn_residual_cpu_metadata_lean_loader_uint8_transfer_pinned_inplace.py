"""C32 runtime arm with owned-buffer in-place image normalization."""

_base_ = (
    "./rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_"
    "cpu_metadata_lean_loader_uint8_transfer_pinned.py"
)

model = dict(
    data_preprocessor=dict(
        type="Fisheye3DODSingleFrameInplaceDataPreprocessor",
    ),
)

work_dir = (
    "work_dirs/"
    "rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_"
    "cpu_metadata_lean_loader_uint8_transfer_pinned_inplace"
)

experiment_contract = dict(
    _delete_=True,
    arm="C32-runtime-owned-buffer-inplace-normalization",
    paired_parent="C32-runtime-pinned-asynchronous-uint8-image-transfer",
    changed_operation=(
        "reverse the fixed three-channel BGR axis with flip, then normalize "
        "the freshly allocated FP32 image with sub_ and div_"
    ),
    ingress="prediction-only contiguous uint8 BGR with fixed RGB conversion",
    training="fail closed; use the unchanged training configuration",
    model_math="same exact channel permutation and eager subtract/divide sequence",
    logical_fp32_temporary_bytes_removed_per_batch4=245760000,
    learned_parameters_added=0,
    buffers_added=0,
    user_numeric_hyperparameters_added=0,
    runtime_claim="none; raw-bit CUDA and paired latency/memory gates pending",
)
