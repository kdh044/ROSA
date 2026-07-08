# Sparse4D KB20 3-class fine-tuning from official Sparse4Dv3 R50 weights.
# Run this after the current 2-class experiment finishes.

_base_ = ["./sparse4d_kb20_car_ped_100e_github_r50_b8.py"]

class_names = ["Car", "Pedestrian", "Cyclist"]
metainfo = dict(classes=class_names)
num_classes = len(class_names)

data_root = "/home/airlab/user/kdh/incoming/new_dataset/xtreme1_5seq_mftracker/"
ann_train = data_root + "pkl/kb20_seq29_49_train_idvel.pkl"
ann_val = data_root + "pkl/kb20_seq29_49_val_idvel.pkl"
cam_names = ["CAM_FRONT", "CAM_RIGHT", "CAM_LEFT", "CAM_BACK"]

point_cloud_range = [-20.0, -20.0, -5.0, 20.0, 20.0, 5.0]
max_eval_dist = 20.0
num_epochs = 100

model = dict(
    head=dict(
        instance_bank=dict(anchor="ckpt/kb20_car_ped_cyc_kmeans900.npy"),
        refine_layer=dict(num_cls=num_classes),
    )
)

train_pipeline = [
    dict(type="LoadFisheyeMultiView", cam_names=cam_names, to_float32=True),
    dict(type="ResizeFisheye", size=(960, 540)),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox=False,
        with_label=False,
    ),
    dict(type="ObjectRangeFilterWithInds", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(type="PackSparse4DInputs", ocam_lens_fov_deg=360.0, kb_lens_fov_deg=360.0),
]

val_pipeline = [
    dict(type="LoadFisheyeMultiView", cam_names=cam_names, to_float32=True),
    dict(type="ResizeFisheye", size=(960, 540)),
    dict(type="PackSparse4DInputs", ocam_lens_fov_deg=360.0, kb_lens_fov_deg=360.0),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    batch_sampler=dict(type="SceneGroupBatchSampler", allow_group_reuse=True),
    dataset=dict(
        data_root=data_root,
        ann_file=ann_train,
        pipeline=train_pipeline,
        modality=dict(use_lidar=False, use_camera=True),
        metainfo=metainfo,
        test_mode=False,
        box_type_3d="LiDAR",
        queue_length=1,
        with_velocity=True,
        filter_empty_gt=True,
        use_valid_flag=True,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    dataset=dict(
        data_root=data_root,
        ann_file=ann_val,
        pipeline=val_pipeline,
        modality=dict(use_lidar=False, use_camera=True),
        metainfo=metainfo,
        test_mode=True,
        box_type_3d="LiDAR",
        queue_length=1,
        with_velocity=True,
        filter_empty_gt=False,
        use_valid_flag=True,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type="FisheyeBev3DMetric",
    class_names=class_names,
    dist_thresholds=(0.5, 1.0, 2.0, 4.0),
    score_threshold=0.0,
    max_predictions=300,
    max_eval_dist=max_eval_dist,
    use_velocity_metric=True,
)
test_evaluator = val_evaluator

param_scheduler = [
    dict(type="LinearLR", start_factor=1.0 / 3, begin=0, end=500, by_epoch=False),
    dict(type="CosineAnnealingLR", T_max=num_epochs, by_epoch=True, eta_min_ratio=1e-3),
]

train_cfg = dict(by_epoch=True, max_epochs=num_epochs, val_interval=1)

load_from = (
    "/home/airlab/user/kdh/sphere_stream_sparse4d/"
    "ckpt/sparse4dv3_r50_kb20_compatible.pth"
)
resume = False
work_dir = "work_dirs/sparse4d_kb20_car_ped_cyc_100e_github_r50_b8"
