"""Fast CP16 ROSA paired with the corrected C32 Sparse4D detector."""

_base_ = "./rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual.py"

model = dict(
    head=dict(
        type="ROSAOfficialResidualRefinementHead",
        rosa_cfg=dict(
            _delete_=True,
            profile="fast",
        ),
    ),
)

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
default_hooks = dict(
    checkpoint=dict(
        max_keep_ckpts=5,
        save_best=["fisheye_bev/mAP", "fisheye_bev/NDS"],
        rule=["greater", "greater"],
    )
)
custom_hooks = [
    dict(type="SyncBuffersHook", priority="NORMAL"),
    dict(type="ROSASparseMilestoneCheckpointHook", milestones=(8,)),
]
work_dir = "work_dirs/rosa_r18_main126_v3_fast_cp16_official_residual"

sealed_resume_contract = dict(
    contract="rosa.c32-fast-cp16.30e-pair.v1",
    role="candidate",
)

experiment_contract = dict(
    _delete_=True,
    dataset="Fisheye3DOD Main126",
    backbone="ImageNet-pretrained ResNet-18",
    camera_model="four native 800x800 OCam views",
    temporal=False,
    queries=300,
    refined_queries=50,
    refinement_steps=1,
    per_device_batch_size=4,
    distributed_world_size=2,
    global_batch_size=8,
    max_epochs=30,
    seed=3407,
    detector="C32 corrected Sparse4D D0-D5 plus fast CP16 ROSA sidecar",
    arm="C32+ROSA-fast-CP16-v3",
    paired_baseline="C32-official-Sparse4D-GNN-residual-30e-control",
    changed_detector_operation=("retain C32 identity+dropout around D1-D5 self-GNN"),
    changed_sidecar_operation=(
        "one 50-query full-FOV ROSA refinement pass with signed CP16 RFL"
    ),
    learned_refiner_parameters=87_935,
    official_residual_parameters_added=0,
    official_residual_buffers_added=0,
    official_residual_numeric_hyperparameters_added=0,
    sidecar_profile_selection="fast (fixed CP16 topology)",
)
