"""C32 runtime arm with pinned asynchronous uint8 image transfer."""

_base_ = (
    "./rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_"
    "cpu_metadata_lean_loader_uint8_transfer.py"
)

# PyTorch can issue an asynchronous host-to-device copy only when the source
# tensor is in pinned host memory.  Keep this as an opt-in runtime arm because
# the extra pinning work is hardware- and workload-dependent.
train_dataloader = dict(pin_memory=True)
val_dataloader = dict(pin_memory=True)
test_dataloader = dict(pin_memory=True)

model = dict(data_preprocessor=dict(non_blocking=True))

work_dir = (
    "work_dirs/"
    "rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_"
    "cpu_metadata_lean_loader_uint8_transfer_pinned"
)

experiment_contract = dict(
    _delete_=True,
    arm="C32-runtime-pinned-asynchronous-uint8-image-transfer",
    paired_parent="C32-runtime-single-frame-uint8-image-transfer",
    changed_operation=(
        "pin collated host batches and request the existing MMEngine tensor "
        "copies with non_blocking=True"
    ),
    model_math="unchanged",
    checkpoint="reuse the identical C32 checkpoint",
    learned_parameters_added=0,
    buffers_added=0,
    user_numeric_hyperparameters_added=0,
    runtime_claim="none; preregistered paired CUDA measurement required",
)
