"""Evaluation-only affine OCam arm with exact no-grad storage reuse."""

_base_ = (
    "./rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_"
    "cpu_metadata_lean_loader_rgb_uint8_transfer_pinned_inplace_"
    "affine3x3_metric_gate.py"
)

model = dict(
    head=dict(
        deformable_model=dict(
            type=(
                "ROSAMetricGatedInferenceReuseAffine3x3FullFOV"
                "DeformableFeatureAggregation"
            ),
        ),
    ),
)

work_dir = (
    "work_dirs/"
    "rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_"
    "cpu_metadata_lean_loader_rgb_uint8_transfer_pinned_inplace_"
    "affine3x3_inference_reuse_metric_gate"
)

experiment_contract = dict(
    _delete_=True,
    arm="C32-runtime-affine3x3-no-grad-storage-reuse-metric-gate",
    paired_parent="C32-runtime-affine3x3-OCam-official-metric-gate",
    changed_operation=(
        "reuse owned Horner, pixel, normalization, and final projection "
        "storage only during no-grad evaluation"
    ),
    numerical_contract=(
        "raw-bit equal to the affine parent for storage-reused operations; "
        "the inherited affine transform remains under the same official "
        "Main126 metric-parity gate"
    ),
    training="fail closed",
    grad_enabled_evaluation="fail closed",
    learned_parameters_added=0,
    buffers_added=0,
    checkpoint_keys_added=0,
    user_numeric_hyperparameters_added=0,
    runtime_claim="none; paired CUDA gates pending",
)
