"""B0: corrected native Sparse4D under the exact ROSA v2 data contract."""

_base_ = "./rosa_r18_main126_v2.py"

model = dict(
    head=dict(
        type="ROSANativeBaselineHead",
        rosa_cfg=None,
    ),
)

work_dir = "work_dirs/rosa_r18_main126_v2_b0_native"

experiment_contract = dict(
    _delete_=True,
    dataset="Fisheye3DOD Main126",
    backbone="ImageNet-pretrained ResNet-18",
    camera_model="four native 800x800 OCam views",
    temporal=False,
    queries=300,
    global_batch_size=8,
    seed=3407,
    detector="native Sparse4D D0-D5",
    rosa_sidecar=False,
    learned_refiner_parameters=0,
    projection_contract="full_fov",
    normalized_sampling_yaw=True,
    anchor_z_convention="gravity_center",
    centerness_target="detached_evaluator_xy",
    train_gt_range="official_class_radial_strict",
)
