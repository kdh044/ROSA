"""C32: restore the released Sparse4D query-attention residual wrapper."""

_base_ = "./rosa_r18_main126_v2_c0_cbgs_native.py"

model = dict(
    head=dict(
        type="ROSAOfficialResidualNativeBaselineHead",
    ),
)

work_dir = "work_dirs/rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual"

experiment_contract = dict(
    _delete_=True,
    dataset="Fisheye3DOD Main126",
    backbone="ImageNet-pretrained ResNet-18",
    camera_model="four native 800x800 OCam views",
    temporal=False,
    queries=300,
    per_device_batch_size=4,
    distributed_world_size=2,
    global_batch_size=8,
    max_epochs=20,
    seed=3407,
    detector="native Sparse4D D0-D5",
    arm="C32-official-Sparse4D-GNN-residual",
    paired_baseline="C0-raw-CBGS-native",
    ablation="restore-mmcv-MHA-identity-plus-output-dropout",
    changed_operation=(
        "D1-D5 self-GNN: fc_after(attention) -> " "fc_after(query + dropout(attention))"
    ),
    dropout_source="reuse graph_model.dropout=0.1; no added setting",
    operation_order="unchanged; post-GNN LayerNorm remains unchanged",
    learned_parameters_added=0,
    buffers_added=0,
    user_numeric_hyperparameters_added=0,
    inference_shape="unchanged",
)
