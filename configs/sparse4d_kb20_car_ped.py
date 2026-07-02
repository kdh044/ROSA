# Sparse4D KB20 4-camera fisheye baseline.
# Uses raw KB fisheye projection instead of MFTracker's equirectangular wrapper.

_base_ = ["./sparse4d_robosense_ocam_4cam.py"]

custom_imports = dict(
    _delete_=True,
    imports=["MFTracker", "projects.mmdet3d_plugin"],
    allow_failed_imports=False,
)

class_names = ["Car", "Pedestrian"]
metainfo = dict(classes=class_names)
num_classes = len(class_names)

data_root = "/home/airlab/user/kdh/incoming/new_dataset/xtreme1_5seq_mftracker/"
ann_train = data_root + "pkl/kb20_seq29_49_train_car_ped_vel.pkl"
ann_val = data_root + "pkl/kb20_seq29_49_val_car_ped_vel.pkl"
cam_names = ["CAM_FRONT", "CAM_RIGHT", "CAM_LEFT", "CAM_BACK"]

point_cloud_range = [-20.0, -20.0, -5.0, 20.0, 20.0, 5.0]
max_eval_dist = 20.0

model = dict(
    img_backbone=dict(with_cp=False),
    head=dict(
        instance_bank=dict(anchor="ckpt/kb20_car_ped_kmeans900.npy"),
        deformable_model=dict(projection_type="kb_fisheye"),
        refine_layer=dict(num_cls=num_classes),
    )
)

optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1.0e-4, weight_decay=0.001),
    clip_grad=dict(max_norm=25, norm_type=2),
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.5))),
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
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        ann_file=ann_train,
        pipeline=train_pipeline,
        metainfo=metainfo,
        use_valid_flag=True,
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=ann_val,
        pipeline=val_pipeline,
        metainfo=metainfo,
        use_valid_flag=True,
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type="FisheyeBev3DMetric",
    class_names=class_names,
    dist_thresholds=(0.5, 1.0, 2.0),
    score_threshold=0.0,
    max_predictions=300,
    max_eval_dist=max_eval_dist,
)
test_evaluator = val_evaluator

load_from = None
resume = False
