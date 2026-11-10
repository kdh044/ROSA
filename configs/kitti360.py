# ROSA on KITTI-360 (six classes): VoVNet-99, 24 epochs, 2 GPUs x 4 images.
#
# KITTI-360 combines a rectified pinhole pair with two MEI fisheyes, so the
# decoder uses the mixed pinhole/MEI projection.  ROSA_KITTI360_ROOT points at
# the dataset; the FCOS3D VoVNet-99 initialization is downloaded to
# checkpoints/ (docs/DATA.md) or set with ROSA_VOVNET_CKPT.
import os

data_root = os.environ.get("ROSA_KITTI360_ROOT", "data/KITTI-360")
ann_dir = "ImageSets-rosa-6class"  # relative to data_root
anchor_file = "assets/anchors/kitti360.npy"
pretrained_backbone = os.environ.get(
    "ROSA_VOVNET_CKPT", "checkpoints/fcos3d_vovnet_imgbackbone-remapped.pth"
)
seed = int(os.environ.get("ROSA_SEED", 3407))

classes = ("car", "truck", "trailer", "motorcycle", "bicycle", "pedestrian")
cameras = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK_LEFT")
num_epochs = 24

custom_imports = dict(
    imports=[
        "projects.mmdet3d_plugin",
        "rosa.integrations.box_loss",
        "rosa.integrations.optim",
        "rosa.integrations.detector",
        "rosa.integrations.head",
        "rosa.integrations.kitti360.dataset",
        "rosa.integrations.kitti360.metric",
        "rosa.integrations.kitti360.backbone",
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
        type="ROSAVoVNetCP",
        spec_name="V-99-eSE",
        with_cp=False,
        input_ch=3,
        out_features=("stage2", "stage3", "stage4", "stage5"),
        frozen_stages=-1,
        norm_eval=False,
        init_cfg=None,
    ),
    img_neck=dict(
        type="mmdet.FPN",
        in_channels=[256, 512, 768, 1024],
        out_channels=256,
        num_outs=4,
        start_level=0,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
    ),
    head=dict(
        type="ROSAHead",
        num_decoder=6,
        decoder_cfg=dict(
            embed_dims=256,
            num_classes=len(classes),
            num_cameras=4,
            num_learnable_points=6,
            fixed_keypoint_scale=0.45,
            projection_type="pinhole_mei",
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
            type="ROSABoxLoss",
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
    type="LoadKITTI360Images",
    cameras=cameras,
    data_root=data_root,
    to_float32=True,
)
train_pipeline = [
    load_images,
    dict(type="KITTI360ImageAugment", training=True),
    dict(type="KITTI360GlobalRotScale", training=True),
    dict(type="KITTI360RangeFilter"),
    dict(type="PackSparse4DInputs"),
]
test_pipeline = [
    load_images,
    dict(type="KITTI360ImageAugment", training=False),
    dict(type="PackSparse4DInputs"),
]
dataset = dict(
    type="KITTI360Dataset",
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
        ann_file=os.path.join(ann_dir, "kitti360_rosa_infos_train.pkl"),
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
        ann_file=os.path.join(ann_dir, "kitti360_rosa_infos_val.pkl"),
        pipeline=test_pipeline,
        test_mode=True,
        filter_empty_gt=False,
    ),
)
test_dataloader = val_dataloader
val_evaluator = dict(
    type="KITTI360Metric",
    ann_file=os.path.join(data_root, ann_dir, "kitti360_rosa_infos_val.pkl"),
    prefix="kitti360_bev",
)
test_evaluator = val_evaluator

# ------------------------------------------------------------- schedule
optim_wrapper = dict(
    type="SeparateClipOptimWrapper",
    optimizer=dict(type="AdamW", lr=2e-4, weight_decay=1e-3),
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.5),
            # Range-uncertainty heads are clipped with their own norm.
            "head.rosa_decoder.range_uncertainty_heads": dict(separate_clip=True),
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
        save_best=["kitti360_bev/mAP", "kitti360_bev/NDS"],
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
find_unused_parameters = True
load_from = pretrained_backbone
resume = False
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
log_level = "INFO"
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(type="Det3DLocalVisualizer", vis_backends=vis_backends, name="visualizer")
work_dir = "work_dirs/kitti360"
