# ------------------------------------------------------------------------
# Sparse4D v3 (mmcv 2.x port) — RoboSense 4xOV fisheye.
#   3 classes (Car/Pedestrian/Cyclist), ±25 m, OcamCalib projection.
#   Single-frame input (Sparse4D handles temporal via InstanceBank).
#   depth_branch OFF for the initial smoke; turn on after wiring LiDAR
#   sweeps + MultiScaleDepthMapGenerator into the pipeline.
# ------------------------------------------------------------------------
default_scope = "mmdet3d"

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
log_level = "INFO"

custom_imports = dict(
    imports=["sphere_stream", "projects.mmdet3d_plugin"],
    allow_failed_imports=False,
)

# ── geometry / classes ────────────────────────────────────────────────
class_names = ["Car", "Pedestrian", "Cyclist"]
metainfo = dict(classes=class_names)
num_classes = len(class_names)

# Image / feature dimensions (matches sphere_stream): 1280x720 → 960x544.
img_size = (960, 544)

# Sparse4D model knobs
embed_dims = 256
num_groups = 8
num_decoder = 6
num_single_frame_decoder = 1
num_levels = 4
strides = [4, 8, 16, 32]
drop_out = 0.1
temporal = True
decouple_attn = True
with_quality_estimation = True
use_deformable_func = True  # CUDA op compiled via ops/setup.py

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
input_modality = dict(use_lidar=False, use_camera=True)

# ───────────────────────────────────────────────────────────────────────
#                                MODEL
# ───────────────────────────────────────────────────────────────────────
model = dict(
    type="Sparse4D",
    use_grid_mask=True,
    use_deformable_func=use_deformable_func,
    data_preprocessor=dict(
        type="Sparse4DDataPreprocessor",
        mean=img_norm_cfg["mean"], std=img_norm_cfg["std"],
        bgr_to_rgb=True, pad_size_divisor=32,
    ),
    img_backbone=dict(
        type="mmdet.ResNet",
        depth=50, num_stages=4, frozen_stages=-1,
        norm_eval=False, style="pytorch", with_cp=True,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type="BN", requires_grad=True),
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
    ),
    img_neck=dict(
        type="mmdet.FPN",
        num_outs=num_levels, start_level=0,
        out_channels=embed_dims,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048],
    ),
    # depth_branch=None,   # OFF for smoke test; see header
    head=dict(
        type="Sparse4DHead",
        cls_threshold_to_reg=0.05,
        decouple_attn=decouple_attn,
        instance_bank=dict(
            type="InstanceBank",
            num_anchor=900,
            embed_dims=embed_dims,
            anchor="ckpt/robosense_kmeans900.npy",
            anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
            num_temp_instances=600 if temporal else -1,
            confidence_decay=0.6,
            feat_grad=False,
        ),
        anchor_encoder=dict(
            type="SparseBox3DEncoder",
            vel_dims=2,                                    # RoboSense (vx, vy)
            embed_dims=[128, 32, 32, 64] if decouple_attn else 256,
            mode="cat" if decouple_attn else "add",
            output_fc=not decouple_attn,
            in_loops=1,
            out_loops=4 if decouple_attn else 2,
        ),
        num_single_frame_decoder=num_single_frame_decoder,
        operation_order=(
            ["gnn", "norm", "deformable", "ffn", "norm", "refine"]
            * num_single_frame_decoder
            + ["temp_gnn", "gnn", "norm", "deformable", "ffn", "norm", "refine"]
            * (num_decoder - num_single_frame_decoder)
        )[2:],
        temp_graph_model=dict(
            type="MultiheadAttention",
            embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
            num_heads=num_groups,
            batch_first=True,
            dropout=drop_out,
        ) if temporal else None,
        graph_model=dict(
            type="MultiheadAttention",
            embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
            num_heads=num_groups,
            batch_first=True,
            dropout=drop_out,
        ),
        norm_layer=dict(type="LN", normalized_shape=embed_dims),
        ffn=dict(
            type="AsymmetricFFN",
            in_channels=embed_dims * 2,
            pre_norm=dict(type="LN"),
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
            num_fcs=2, ffn_drop=drop_out,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        deformable_model=dict(
            type="DeformableFeatureAggregation",
            embed_dims=embed_dims,
            num_groups=num_groups,
            num_levels=num_levels,
            num_cams=4,
            attn_drop=0.15,
            use_deformable_func=use_deformable_func,
            use_camera_embed=True,
            residual_mode="cat",
            projection_type="ocam_fisheye",   # ← the key swap for fisheye
            kps_generator=dict(
                type="SparseBox3DKeyPointsGenerator",
                num_learnable_pts=6,
                fix_scale=[
                    [0, 0, 0],
                    [0.45, 0, 0], [-0.45, 0, 0],
                    [0, 0.45, 0], [0, -0.45, 0],
                    [0, 0, 0.45], [0, 0, -0.45],
                ],
            ),
        ),
        refine_layer=dict(
            type="SparseBox3DRefinementModule",
            embed_dims=embed_dims,
            num_cls=num_classes,
            refine_yaw=True,
            with_quality_estimation=with_quality_estimation,
        ),
        sampler=dict(
            type="SparseBox3DTarget",
            num_dn_groups=5,
            num_temp_dn_groups=3,
            dn_noise_scale=[2.0] * 3 + [0.5] * 7,
            max_dn_gt=32,
            add_neg_dn=True,
            cls_weight=2.0,
            box_weight=0.25,
            reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,
        ),
        loss_cls=dict(
            type="mmdet.FocalLoss",
            use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0,
        ),
        loss_reg=dict(
            type="SparseBox3DLoss",
            loss_box=dict(type="mmdet.L1Loss", loss_weight=0.25),
            loss_centerness=dict(
                type="mmdet.CrossEntropyLoss", use_sigmoid=True
            ),
            loss_yawness=dict(type="mmdet.GaussianFocalLoss"),
        ),
        decoder=dict(type="SparseBox3DDecoder"),
        reg_weights=[2.0] * 3 + [1.0] * 7,
    ),
)

# ───────────────────────────────────────────────────────────────────────
#                                DATA
# ───────────────────────────────────────────────────────────────────────
dataset_type = "FisheyeDataset"
data_root = "/home/airlab/user/kdh/RoboSense_images/"
ann_train = data_root + "pkl/sphere_stream_robosense_train_dommatched_v.pkl"
ann_val = data_root + "pkl/sphere_stream_robosense_val_dommatched_v.pkl"

cam_names = ["CAM_FRONT_OV", "CAM_LEFT_OV", "CAM_RIGHT_OV", "CAM_BACK_OV"]

train_pipeline = [
    dict(type="LoadFisheyeMultiView", cam_names=cam_names, to_float32=True),
    dict(type="ResizeFisheye", size=(960, 540)),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=True, with_label_3d=True,
        with_bbox=False, with_label=False,
    ),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(type="PackSparse4DInputs", ocam_lens_fov_deg=205.0),
]

val_pipeline = [
    dict(type="LoadFisheyeMultiView", cam_names=cam_names, to_float32=True),
    dict(type="ResizeFisheye", size=(960, 540)),
    dict(type="PackSparse4DInputs", ocam_lens_fov_deg=205.0),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    collate_fn=dict(type="default_collate"),
    sampler=dict(type="DefaultSampler", shuffle=False),
    batch_sampler=dict(type="SceneGroupBatchSampler"),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_train,
        pipeline=train_pipeline,
        modality=input_modality,
        metainfo=metainfo,
        test_mode=False,
        box_type_3d="LiDAR",
        queue_length=1,
        with_velocity=True,
        filter_empty_gt=True,
        use_valid_flag=False,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    collate_fn=dict(type="default_collate"),
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_val,
        pipeline=val_pipeline,
        modality=input_modality,
        metainfo=metainfo,
        test_mode=True,
        box_type_3d="LiDAR",
        queue_length=1,
        with_velocity=True,
        filter_empty_gt=False,
        use_valid_flag=False,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type="FisheyeBev3DMetric",
    class_names=class_names,
    dist_thresholds=(0.5, 1.0, 2.0),
    score_threshold=0.1,
    max_predictions=300,
)
test_evaluator = val_evaluator

# ───────────────────────────────────────────────────────────────────────
#                                OPTIM
# ───────────────────────────────────────────────────────────────────────
num_epochs = 24
val_interval = 1

optim_wrapper = dict(
    type="AmpOptimWrapper",
    dtype="bfloat16",
    optimizer=dict(type="AdamW", lr=6e-4, weight_decay=0.001),
    paramwise_cfg=dict(custom_keys={"img_backbone": dict(lr_mult=0.5)}),
    clip_grad=dict(max_norm=25, norm_type=2),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=1.0 / 3,
         begin=0, end=500, by_epoch=False),
    dict(type="CosineAnnealingLR", T_max=num_epochs,
         by_epoch=True, eta_min_ratio=1e-3),
]

train_cfg = dict(by_epoch=True, max_epochs=num_epochs, val_interval=val_interval)
val_cfg = dict()
test_cfg = dict()

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=10,
                    save_best="fisheye_bev/mAP", rule="greater"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
)

find_unused_parameters = True
load_from = None
resume = False

vis_backends = [dict(type="LocalVisBackend"),
                dict(type="TensorboardVisBackend")]
visualizer = dict(type="Det3DLocalVisualizer",
                  vis_backends=vis_backends, name="visualizer")
