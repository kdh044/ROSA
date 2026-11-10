# ROSA on Fisheye3DOD (126 scenes): ResNet-18, 20 epochs, 2 GPUs x 4 images.
#
# Set ROSA_DATA_ROOT if the dataset is not at ./data/Fisheye3DOD.
import os

data_root = os.environ.get("ROSA_DATA_ROOT", "data/Fisheye3DOD")
ann_dir = "ImageSets-main126"  # relative to data_root
anchor_file = "assets/anchors/fisheye3dod.npy"
seed = int(os.environ.get("ROSA_SEED", 3407))

classes = ("Car", "Van", "Truck", "Bus", "Pedestrian", "Cyclist")
cameras = (
    "fisheye_camera_front",
    "fisheye_camera_left",
    "fisheye_camera_right",
    "fisheye_camera_rear",
)
num_epochs = 20

custom_imports = dict(
    imports=[
        "projects.mmdet3d_plugin",
        "rosa.integrations.box_loss",
        "rosa.integrations.optim",
        "rosa.integrations.fisheye3dod.dataset",
        "rosa.integrations.detector",
        "rosa.integrations.fisheye3dod.metric",
        "rosa.integrations.head",
    ],
    allow_failed_imports=False,
)
default_scope = "mmdet3d"

# ---------------------------------------------------------------- model
model = dict(
    type="ROSADetector",
    data_preprocessor=dict(
        type="Sparse4DDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    img_backbone=dict(
        type="mmdet.ResNet",
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=False,
        style="pytorch",
        with_cp=False,
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet18"),
    ),
    img_neck=dict(
        type="mmdet.FPN",
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=4,
        start_level=0,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
    ),
    head=dict(
        type="ROSAHead",
        num_decoder=6,
        hybrid_decoder_cfg=dict(
            embed_dims=256,
            num_classes=len(classes),
            num_cameras=4,
            num_learnable_points=6,
            fixed_keypoint_scale=0.45,
            # Fused DAF + camera RFL kernel (inference only, OCam).
            streaming_camera_rfl_cuda=True,
        ),
        instance_bank=dict(
            type="InstanceBank",
            num_anchor=300,
            embed_dims=256,
            anchor=anchor_file,
            feat_grad=False,
        ),
        sampler=dict(
            type="SparseBox3DTarget",
            num_dn_groups=5,
            num_temp_dn_groups=0,
            dn_noise_scale=[2.0] * 3 + [0.5] * 7,
            max_dn_gt=32,
            add_neg_dn=True,
            cls_weight=2.0,
            box_weight=0.25,
            reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,
        ),
        decoder=dict(type="SparseBox3DDecoder", num_output=300),
        loss_cls=dict(
            type="mmdet.FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0,
        ),
        loss_reg=dict(
            type="EvaluatorAlignedSparseBox3DLoss",
            loss_box=dict(type="mmdet.L1Loss", loss_weight=0.25),
            loss_centerness=dict(type="mmdet.CrossEntropyLoss", use_sigmoid=True),
            loss_yawness=dict(type="mmdet.GaussianFocalLoss"),
        ),
        reg_weights=[2.0] * 3 + [1.0] * 7,
        cls_threshold_to_reg=0.05,
        ray_aux_loss=dict(
            angular_weight=0.25,
            log_range_l1_weight=0.25,
            range_nll_weight=0.1,
            beta=0.1,
        ),
    ),
)

# ----------------------------------------------------------------- data
load_images = dict(
    type="LoadOfficialFisheye3DODForSparse4D",
    cam_names=cameras,
    to_float32=True,
)
train_pipeline = [
    load_images,
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox=False,
        with_label=False,
    ),
    # Ground truth beyond the evaluator's class ranges is dropped.
    dict(type="Fisheye3DODEvaluatorRangeFilter"),
    dict(type="PackSparse4DInputs", ocam_lens_fov_deg=220.0),
]
test_pipeline = [
    load_images,
    dict(type="PackSparse4DInputs", ocam_lens_fov_deg=220.0),
]
dataset = dict(
    type="Fisheye3DODSparse4DDataset",
    data_root=data_root,
    metainfo=dict(classes=classes),
    with_velocity=True,
)
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    collate_fn=dict(type="default_collate"),
    dataset=dict(
        **dataset,
        ann_file=os.path.join(ann_dir, "fisheye3dod_infos_train.pkl"),
        pipeline=train_pipeline,
        test_mode=False,
        filter_empty_gt=True,
    ),
)
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    collate_fn=dict(type="default_collate"),
    dataset=dict(
        **dataset,
        ann_file=os.path.join(ann_dir, "fisheye3dod_infos_val.pkl"),
        pipeline=test_pipeline,
        test_mode=True,
        filter_empty_gt=False,
    ),
)
test_dataloader = val_dataloader
val_evaluator = dict(type="Fisheye3DODOfficialMetric", prefix="fisheye_bev")
test_evaluator = val_evaluator

# ------------------------------------------------------------- schedule
optim_wrapper = dict(
    type="SeparateClipOptimWrapper",
    optimizer=dict(type="AdamW", lr=2e-4, weight_decay=1e-3),
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.5),
            # Range-uncertainty heads are clipped with their own norm.
            "head.fisheye_box_decoder.range_uncertainty_heads": dict(separate_clip=True),
        }
    ),
    clip_grad=dict(max_norm=25, norm_type=2, error_if_nonfinite=True),
)
param_scheduler = [
    dict(type="LinearLR", start_factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(
        type="CosineAnnealingLR",
        T_max=num_epochs,
        eta_min_ratio=1e-3,
        by_epoch=True,
        begin=0,
        end=num_epochs,
        convert_to_iter_based=True,
    ),
]
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=num_epochs, val_interval=4)
val_cfg = dict()
test_cfg = dict()

# -------------------------------------------------------------- runtime
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=5,
        save_best=["fisheye_bev/mAP", "fisheye_bev/NDS"],
        rule=["greater", "greater"],
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
)
custom_hooks = [dict(type="SyncBuffersHook", priority="NORMAL")]
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
randomness = dict(seed=seed, deterministic=True)
find_unused_parameters = False
load_from = None
resume = False
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
log_level = "INFO"
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(type="Det3DLocalVisualizer", vis_backends=vis_backends, name="visualizer")
work_dir = "work_dirs/fisheye3dod"
