"""Evaluation-only C32 affine OCam port pending official metric parity."""

_base_ = (
    "./rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_"
    "cpu_metadata_lean_loader_rgb_uint8_transfer_pinned_inplace.py"
)

model = dict(
    head=dict(
        deformable_model=dict(
            type=(
                "ROSAMetricGatedAffine3x3FullFOV"
                "DeformableFeatureAggregation"
            ),
        ),
    ),
)

work_dir = (
    "work_dirs/"
    "rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_"
    "cpu_metadata_lean_loader_rgb_uint8_transfer_pinned_inplace_"
    "affine3x3_metric_gate"
)

experiment_contract = dict(
    _delete_=True,
    arm="C32-runtime-affine3x3-OCam-official-metric-gate",
    paired_parent="C32-runtime-strict-homogeneous4x4-OCam",
    changed_operation=(
        "replace only the evaluation camera-frame 4x4 homogeneous transform "
        "with the mathematically equivalent 3x3 rotation plus translation"
    ),
    numerical_contract=(
        "small FP32 reduction-order differences allowed only after same-"
        "checkpoint official Main126 metric parity"
    ),
    training="fail closed",
    learned_parameters_added=0,
    buffers_added=0,
    checkpoint_keys_added=0,
    user_numeric_hyperparameters_added=0,
    runtime_claim="none; official metric and paired CUDA gates pending",
)
