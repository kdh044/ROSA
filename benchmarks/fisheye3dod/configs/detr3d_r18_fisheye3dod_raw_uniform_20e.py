import os
ann_train = ann_train
ann_val = ann_val
cameras = (
    'fisheye_camera_front',
    'fisheye_camera_left',
    'fisheye_camera_right',
    'fisheye_camera_rear',
)
classes = (
    'Car',
    'Van',
    'Truck',
    'Bus',
    'Pedestrian',
    'Cyclist',
)
custom_hooks = [
    dict(priority='NORMAL', type='SyncBuffersHook'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'detr3d',
        'projects.mmdet3d_plugin',
        'rosa.integrations.fisheye3dod.dataset',
        'rosa.integrations.fisheye3dod.metric',
    ])
data_root = data_root
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=20,
        rule='greater',
        save_best='fisheye_bev/NDS',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmdet3d'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
experiment_contract = dict(
    arm='DETR3D-native-OCam',
    backbone='ImageNet-pretrained ResNet-18',
    camera_model='native 800x800 calibrated OCam',
    dataset='Fisheye3DOD Main126 raw-uniform',
    detector_contract='official DETR3D except pinhole-to-OCam projection',
    global_batch_size=8,
    projection_contract='calibrated_full_fov',
    queries=300,
    seed=3407,
    temporal=False)
find_unused_parameters = False
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
metainfo = dict(
    classes=(
        'Car',
        'Van',
        'Truck',
        'Bus',
        'Pedestrian',
        'Cyclist',
    ))
model = dict(
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='Sparse4DDataPreprocessor'),
    img_backbone=dict(
        depth=18,
        frozen_stages=-1,
        init_cfg=dict(checkpoint='torchvision://resnet18', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=False,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='mmdet.ResNet',
        with_cp=False),
    img_neck=dict(
        add_extra_convs='on_output',
        in_channels=[
            64,
            128,
            256,
            512,
        ],
        num_outs=4,
        out_channels=256,
        relu_before_extra_convs=True,
        start_level=0,
        type='mmdet.FPN'),
    pts_bbox_head=dict(
        as_two_stage=False,
        bbox_coder=dict(
            max_num=300,
            num_classes=6,
            pc_range=[
                -48.0,
                -48.0,
                -5.0,
                48.0,
                48.0,
                5.0,
            ],
            post_center_range=[
                -48.0,
                -48.0,
                -5.0,
                48.0,
                48.0,
                5.0,
            ],
            type='NMSFreeCoder',
            voxel_size=[
                0.2,
                0.2,
                10.0,
            ]),
        in_channels=256,
        loss_bbox=dict(loss_weight=0.25, type='mmdet.L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=2.0,
            type='mmdet.FocalLoss',
            use_sigmoid=True),
        loss_iou=dict(loss_weight=0.0, type='mmdet.GIoULoss'),
        num_classes=6,
        num_query=300,
        positional_encoding=dict(
            normalize=True,
            num_feats=128,
            offset=-0.5,
            type='mmdet.SinePositionalEncoding'),
        sync_cls_avg_factor=True,
        transformer=dict(
            decoder=dict(
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    attn_cfgs=[
                        dict(
                            dropout=0.1,
                            embed_dims=256,
                            num_heads=8,
                            type='MultiheadAttention'),
                        dict(
                            embed_dims=256,
                            num_cams=4,
                            num_levels=4,
                            num_points=1,
                            pc_range=[
                                -48.0,
                                -48.0,
                                -5.0,
                                48.0,
                                48.0,
                                5.0,
                            ],
                            type='Detr3DCrossAtten'),
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=(
                        'self_attn',
                        'norm',
                        'cross_attn',
                        'norm',
                        'ffn',
                        'norm',
                    ),
                    type='BaseTransformerLayer'),
                type='Detr3DTransformerDecoder'),
            type='Detr3DTransformer'),
        type='DETR3DHead',
        with_box_refine=True),
    train_cfg=dict(
        pts=dict(
            assigner=dict(
                cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
                iou_cost=dict(type='mmdet.IoUCost', weight=0.0),
                pc_range=[
                    -48.0,
                    -48.0,
                    -5.0,
                    48.0,
                    48.0,
                    5.0,
                ],
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                type='HungarianAssigner3D'),
            grid_size=[
                480,
                480,
                1,
            ],
            out_size_factor=4,
            point_cloud_range=[
                -48.0,
                -48.0,
                -5.0,
                48.0,
                48.0,
                5.0,
            ],
            voxel_size=[
                0.2,
                0.2,
                10.0,
            ])),
    type='DETR3D',
    use_grid_mask=False)
optim_wrapper = dict(
    clip_grad=dict(error_if_nonfinite=True, max_norm=25, norm_type=2),
    optimizer=dict(lr=0.0002, type='AdamW', weight_decay=0.001),
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.5))),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=500,
        start_factor=0.3333333333333333,
        type='LinearLR'),
    dict(
        T_max=20,
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=20,
        eta_min_ratio=0.001,
        type='CosineAnnealingLR'),
]
point_cloud_range = [
    -48.0,
    -48.0,
    -5.0,
    48.0,
    48.0,
    5.0,
]
randomness = dict(deterministic=True, seed=3407)
resume = False
seed = 3407
test_cfg = dict()
test_dataloader = dict(
    batch_size=4,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file=
        ann_val,
        data_root=data_root,
        filter_empty_gt=False,
        metainfo=dict(
            classes=(
                'Car',
                'Van',
                'Truck',
                'Bus',
                'Pedestrian',
                'Cyclist',
            )),
        pipeline=[
            dict(
                cam_names=(
                    'fisheye_camera_front',
                    'fisheye_camera_left',
                    'fisheye_camera_right',
                    'fisheye_camera_rear',
                ),
                to_float32=True,
                type='LoadOfficialFisheye3DODForSparse4D'),
            dict(type='PackCalibratedSparse4DInputs'),
        ],
        test_mode=True,
        type='Fisheye3DODSparse4DDataset',
        with_velocity=True),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    prefix='fisheye_bev', type='Fisheye3DODReleasedMetricBridge')
test_pipeline = [
    dict(
        cam_names=(
            'fisheye_camera_front',
            'fisheye_camera_left',
            'fisheye_camera_right',
            'fisheye_camera_rear',
        ),
        to_float32=True,
        type='LoadOfficialFisheye3DODForSparse4D'),
    dict(type='PackCalibratedSparse4DInputs'),
]
train_cfg = dict(max_epochs=20, type='EpochBasedTrainLoop', val_interval=4)
train_dataloader = dict(
    batch_size=4,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file=
        ann_train,
        data_root=data_root,
        filter_empty_gt=True,
        metainfo=dict(
            classes=(
                'Car',
                'Van',
                'Truck',
                'Bus',
                'Pedestrian',
                'Cyclist',
            )),
        pipeline=[
            dict(
                cam_names=(
                    'fisheye_camera_front',
                    'fisheye_camera_left',
                    'fisheye_camera_right',
                    'fisheye_camera_rear',
                ),
                to_float32=True,
                type='LoadOfficialFisheye3DODForSparse4D'),
            dict(
                type='LoadAnnotations3D',
                with_bbox=False,
                with_bbox_3d=True,
                with_label=False,
                with_label_3d=True),
            dict(type='Fisheye3DODEvaluatorRangeFilter'),
            dict(type='PackCalibratedSparse4DInputs'),
        ],
        test_mode=False,
        type='Fisheye3DODSparse4DDataset',
        with_velocity=True),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        cam_names=(
            'fisheye_camera_front',
            'fisheye_camera_left',
            'fisheye_camera_right',
            'fisheye_camera_rear',
        ),
        to_float32=True,
        type='LoadOfficialFisheye3DODForSparse4D'),
    dict(
        type='LoadAnnotations3D',
        with_bbox=False,
        with_bbox_3d=True,
        with_label=False,
        with_label_3d=True),
    dict(type='Fisheye3DODEvaluatorRangeFilter'),
    dict(type='PackCalibratedSparse4DInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=4,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file=
        ann_val,
        data_root=data_root,
        filter_empty_gt=False,
        metainfo=dict(
            classes=(
                'Car',
                'Van',
                'Truck',
                'Bus',
                'Pedestrian',
                'Cyclist',
            )),
        pipeline=[
            dict(
                cam_names=(
                    'fisheye_camera_front',
                    'fisheye_camera_left',
                    'fisheye_camera_right',
                    'fisheye_camera_rear',
                ),
                to_float32=True,
                type='LoadOfficialFisheye3DODForSparse4D'),
            dict(type='PackCalibratedSparse4DInputs'),
        ],
        test_mode=True,
        type='Fisheye3DODSparse4DDataset',
        with_velocity=True),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    prefix='fisheye_bev', type='Fisheye3DODReleasedMetricBridge')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
voxel_size = [
    0.2,
    0.2,
    10.0,
]
work_dir = None  # path removed for release
