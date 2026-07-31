"""C0: native B0 with the official raw-category CBGS train wrapper."""

_base_ = "./rosa_r18_main126_v2_b0_native.py"

_native_train_dataset = {{_base_.train_dataloader.dataset}}

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type="CBGSDataset",
        dataset=_native_train_dataset,
    ),
)

work_dir = "work_dirs/rosa_r18_main126_v2_c0_cbgs_native"

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
    rosa_sidecar=False,
    learned_refiner_parameters=0,
    projection_contract="full_fov",
    normalized_sampling_yaw=True,
    anchor_z_convention="gravity_center",
    centerness_target="detached_evaluator_xy",
    train_gt_range="official_class_radial_strict",
    train_sampling="mmdet3d.CBGSDataset(raw_pre_pipeline_categories)",
    raw_train_frames=8_820,
    cbgs_train_samples=25_457,
    b0_steps_per_epoch=1_103,
    cbgs_steps_per_epoch=3_183,
    b0_optimizer_steps_20e=22_060,
    cbgs_optimizer_steps_20e=63_660,
    optimizer_step_increase_20e=41_600,
    optimizer_step_multiplier=2.885766092475068,
)

del _native_train_dataset
