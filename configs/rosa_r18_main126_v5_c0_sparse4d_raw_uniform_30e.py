"""Corrected Sparse4D C0 on the original raw-uniform 30-epoch schedule.

This is the architecture baseline.  It keeps the released Sparse4D query
attention residual restored, disables the ROSA sidecar, and unwraps only the
CBGS dataset wrapper.  The epoch-8 milestone gives an early result without
changing the 30-epoch cosine horizon.
"""

_base_ = "./rosa_r18_main126_v3_c32_official_residual_30e.py"

# The parent C32 control inherits CBGSDataset from C0-CBGS.  Restore the
# original dataset distribution while preserving its inner pipeline and the
# DefaultSampler.  This is frame-uniform sampling, not class balancing.
train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type={{_base_.train_dataloader.dataset.dataset.type}},
        data_root={{_base_.train_dataloader.dataset.dataset.data_root}},
        ann_file={{_base_.train_dataloader.dataset.dataset.ann_file}},
        pipeline={{_base_.train_dataloader.dataset.dataset.pipeline}},
        test_mode={{_base_.train_dataloader.dataset.dataset.test_mode}},
        filter_empty_gt={{_base_.train_dataloader.dataset.dataset.filter_empty_gt}},
        metainfo={{_base_.train_dataloader.dataset.dataset.metainfo}},
        with_velocity={{_base_.train_dataloader.dataset.dataset.with_velocity}},
    ),
)

work_dir = "work_dirs/rosa_r18_main126_v5_c0_sparse4d_raw_uniform_30e"

# This baseline is deliberately outside the old C32/ROSA paired-resume label.
sealed_resume_contract = None

experiment_contract = dict(
    _delete_=True,
    arm="C0-corrected-Sparse4D-raw-uniform-30e",
    dataset="Fisheye3DOD Main126",
    backbone="ImageNet-pretrained ResNet-18",
    camera_model="four native 800x800 OCam views",
    detector="corrected native Sparse4D D0-D5",
    temporal=False,
    queries=300,
    seed=3407,
    per_device_batch_size=4,
    distributed_world_size=2,
    global_batch_size=8,
    max_epochs=30,
    train_sampling="raw-uniform DefaultSampler (no CBGSDataset)",
    raw_train_frames=8_820,
    raw_steps_per_epoch=1_103,
    optimizer_steps_30e=33_090,
    early_metric_milestone_epoch=8,
    attention_contract=(
        "D1-D5 fc_after(query + dropout(attention)); released MMCV semantics"
    ),
    rosa_sidecar=False,
    learned_refiner_parameters=0,
    learned_parameters_added_vs_correct_sparse4d=0,
    user_numeric_hyperparameters_added=0,
    comparison_note=(
        "architecture baseline; CBGS results are not update-matched comparisons"
    ),
)
