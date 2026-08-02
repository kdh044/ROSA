"""Bit-exact homogeneous OCam runtime candidate with owned-storage reuse."""

_base_ = (
    "./rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_"
    "cpu_metadata_lean_loader_rgb_uint8_transfer_pinned_inplace.py"
)

model = dict(
    head=dict(
        deformable_model=dict(
            type=(
                "ROSAInferenceReuseExactHomogeneous4x4FullFOV"
                "DeformableFeatureAggregation"
            ),
        ),
    ),
)

work_dir = (
    "work_dirs/"
    "rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_"
    "cpu_metadata_lean_loader_rgb_uint8_transfer_pinned_inplace_"
    "homogeneous4x4_inference_reuse_exact"
)

experiment_contract = dict(
    _delete_=True,
    arm="C32-runtime-exact-homogeneous4x4-no-grad-storage-reuse",
    paired_parent=("C32-runtime-cpu-metadata-lean-loader-rgb-uint8-pinned-inplace"),
    changed_operation=(
        "reuse projection-owned intermediate storage without changing the "
        "historical homogeneous 4x4 reduction order"
    ),
    numerical_contract=(
        "raw-bit equal to the homogeneous parent for supported ordinary "
        "no-grad inference inputs"
    ),
    training="fail closed",
    grad_enabled_evaluation="fail closed",
    learned_parameters_added=0,
    buffers_added=0,
    checkpoint_keys_added=0,
    new_user_exposed_numeric_dimensions=0,
    official_metric_gate=(
        "still required for release evidence even though component outputs "
        "are raw-bit equal"
    ),
    runtime_claim="none; paired CUDA full-test-step gates pending",
)
