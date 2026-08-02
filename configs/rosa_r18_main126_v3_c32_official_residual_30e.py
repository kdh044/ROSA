"""Thirty-epoch C32 control paired with fast CP16 ROSA.

This control changes only the training horizon/checkpoint retention of the
sealed C32 detector.  It exists so the 30-epoch ROSA arm is never compared
against the historical 20-epoch C32 result as if training duration were an
architectural contribution.
"""

_base_ = "./rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual.py"

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1.0 / 3.0,
        by_epoch=False,
        begin=0,
        end=500,
    ),
    dict(
        type="CosineAnnealingLR",
        begin=0,
        end=30,
        T_max=30,
        by_epoch=True,
        eta_min_ratio=1e-3,
        convert_to_iter_based=True,
    ),
]
train_cfg = dict(max_epochs=30)
default_hooks = dict(checkpoint=dict(max_keep_ckpts=5))
work_dir = "work_dirs/rosa_r18_main126_v3_c32_official_residual_30e"

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
    max_epochs=30,
    seed=3407,
    detector="native Sparse4D D0-D5",
    arm="C32-official-Sparse4D-GNN-residual-30e-control",
    paired_candidate="C32+ROSA-fast-CP16-v3",
    changed_from_historical_c32=(
        "training horizon 20->30 and checkpoint retention 20->5 only"
    ),
    learned_parameters_added=0,
    buffers_added=0,
    detector_numeric_hyperparameters_added=0,
)
