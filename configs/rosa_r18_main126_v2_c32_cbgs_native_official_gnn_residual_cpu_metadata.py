"""Runtime-only C32 arm that avoids temporal metadata CUDA round trips."""

_base_ = "./rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual.py"

model = dict(
    data_preprocessor=dict(type="Fisheye3DODSingleFrameDataPreprocessor"),
)

work_dir = (
    "work_dirs/"
    "rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_cpu_metadata"
)

experiment_contract = dict(
    _delete_=True,
    arm="C32-runtime-single-frame-CPU-metadata",
    paired_model="C32-official-Sparse4D-GNN-residual",
    temporal=False,
    changed_operation=(
        "prediction-only T_global/T_global_inv/timestamp remain on CPU instead "
        "of CUDA-to-CPU round trips in Sparse4D._build_metas"
    ),
    training_path="unchanged",
    checkpoint="reuse the identical C32 checkpoint",
    learned_parameters_added=0,
    buffers_added=0,
    user_numeric_hyperparameters_added=0,
    acceptance=(
        "bitwise prediction parity and preregistered paired CUDA latency/memory gate"
    ),
)
