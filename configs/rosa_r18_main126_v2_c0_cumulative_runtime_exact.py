"""C0 control for the exact cumulative C32 inference-runtime candidate.

Both runtime arms inherit the same loader, transfer, preprocessing, OCam, and
detector-host optimizations.  This control changes only the released head
wrapper, so whole-test-step timing measures the five C32 residual additions
instead of mixing model and input-pipeline differences.
"""

_base_ = (
    "./rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_"
    "cpu_metadata_lean_loader_rgb_uint8_transfer_pinned_inplace_"
    "homogeneous4x4_inference_reuse_exact.py"
)

model = dict(head=dict(type="ROSANativeBaselineHead"))

work_dir = "work_dirs/rosa_r18_main126_v2_c0_cumulative_runtime_exact"

experiment_contract = dict(
    _delete_=True,
    arm="C0-cumulative-runtime-control",
    paired_candidate="C32-cumulative-runtime-exact",
    only_executable_delta="model.head.type",
    learned_parameters_added=0,
    buffers_added=0,
    user_numeric_hyperparameters_added=0,
    runtime_claim="none; paired whole-test-step CUDA gate pending",
)
