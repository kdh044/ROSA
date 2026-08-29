import os
ann_train = ann_train
ann_val = ann_val
cam_fov = 220.0
cam_type = 'cam_fisheye'
classes = (
    'Car',
    'Van',
    'Truck',
    'Bus',
    'Pedestrian',
    'Cyclist',
)
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'fisheye3dod.dataset',
        'fisheye3dod.models',
        'fisheye3dod.models.petr',
        'rosa.integrations.fisheye3dod.dataset',
        'rosa.integrations.fisheye3dod.metric',
    ])
data_root = None  # path removed for release
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
    arm='RepDETR3D-native-OCam-temporal-off',
    backbone='ImageNet-pretrained ResNet-18',
    camera_model='native 800x800 calibrated OCam',
    dataset='Fisheye3DOD Main126 raw-uniform',
    detector_contract=
    "official RepDETR3D except pinhole-to-native-OCam DeformableFeatureAggregationCuda projection; every forward is an independent sequence's first frame (memory bank reset, prev_exists=0, identity ego_pose); backbone/dataset/sampler/evaluator match Table 1",
    global_batch_size=8,
    projection_contract='calibrated_full_fov',
    queries=300,
    seed=3407,
    temporal=False)
find_unused_parameters = True
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 20
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
        bgr_to_rgb=False,
        img_keys=[
            'cam_fisheye',
        ],
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='FisheyeDet3DDataPreprocessor'),
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
                -60,
                -60,
                -10.0,
                60,
                60,
                10.0,
            ],
            type='StreamPETRNMSFreeCoder',
            voxel_size=[
                0.2,
                0.2,
                10.0,
            ]),
        code_size=8,
        in_channels=256,
        loss_bbox=dict(loss_weight=0.25, type='mmdet.L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=2.0,
            type='mmdet.FocalLoss',
            use_sigmoid=True),
        loss_iou=dict(loss_weight=0.0, type='mmdet.GIoULoss'),
        match_with_velo=False,
        memory_len=256,
        normedlinear=False,
        num_classes=6,
        num_propagated=256,
        num_query=300,
        topk_proposals=256,
        transformer=dict(
            decoder=dict(
                embed_dims=256,
                num_layers=6,
                transformerlayers=dict(
                    attn_cfgs=[
                        dict(
                            attn_drop=0.1,
                            batch_first=True,
                            dropout_layer=dict(drop_prob=0.1, type='Dropout'),
                            embed_dims=256,
                            num_heads=8,
                            type='PETRMultiheadAttention'),
                        dict(
                            bias=2.0,
                            embed_dims=256,
                            num_cams=4,
                            num_groups=8,
                            num_levels=4,
                            num_pts=13,
                            ocam_fov=220.0,
                            ocam_path=
                            None  # path removed for release,
                            type='RepDetr3DCrossAttnOCam'),
                    ],
                    batch_first=True,
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=(
                        'self_attn',
                        'norm',
                        'cross_attn',
                        'norm',
                        'ffn',
                        'norm',
                    ),
                    type='RepDetr3DTemporalDecoderLayer',
                    with_cp=False),
                type='RepDetr3DTemporalTransformerDecoder'),
            type='RepDetr3DTemporalTransformer'),
        type='RepDetr3DHead',
        with_dn=False,
        with_ego_pos=True),
    train_cfg=dict(
        input_key=dict(img_key='cam_fisheye', lidar_key='points'),
        pts=dict(
            assigner=dict(
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=[
                    -48.0,
                    -48.0,
                    -5.0,
                    48.0,
                    48.0,
                    5.0,
                ],
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                type='StreamPETRHungarianAssigner3D'),
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
    type='RepDetr3DOCam')
ocam_path = None  # path removed for release
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
    dataset=dict(
        ann_file=
        ann_val,
        data_root=None  # path removed for release,
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
                color_type='color',
                load_cam_names=[
                    'fisheye_camera_front',
                    'fisheye_camera_left',
                    'fisheye_camera_right',
                    'fisheye_camera_rear',
                ],
                load_cam_type='cam_fisheye',
                to_float32=True,
                type='LoadFisheye3DODMultiViewImageFromFiles'),
            dict(
                input_img_keys=[
                    'cam_fisheye',
                ],
                keys=[
                    'cam_fisheye',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                ],
                meta_keys=[
                    'cam2img',
                    'cam2lidar',
                    'lidar2img',
                    'lidar2cam',
                    'sample_idx',
                    'token',
                    'img_path',
                    'lidar_path',
                    'num_pts_feats',
                    'box_type_3d',
                ],
                type='Fisheye3DODPackDetInputs'),
        ],
        test_mode=True,
        type='Fisheye3DODDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    prefix='fisheye_bev', type='Fisheye3DODReleasedMetricBridge')
test_pipeline = [
    dict(
        color_type='color',
        load_cam_names=[
            'fisheye_camera_front',
            'fisheye_camera_left',
            'fisheye_camera_right',
            'fisheye_camera_rear',
        ],
        load_cam_type='cam_fisheye',
        to_float32=True,
        type='LoadFisheye3DODMultiViewImageFromFiles'),
    dict(
        input_img_keys=[
            'cam_fisheye',
        ],
        keys=[
            'cam_fisheye',
            'gt_bboxes_3d',
            'gt_labels_3d',
        ],
        meta_keys=[
            'cam2img',
            'cam2lidar',
            'lidar2img',
            'lidar2cam',
            'sample_idx',
            'token',
            'img_path',
            'lidar_path',
            'num_pts_feats',
            'box_type_3d',
        ],
        type='Fisheye3DODPackDetInputs'),
]
train_cfg = dict(max_epochs=20, type='EpochBasedTrainLoop', val_interval=4)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file=
        ann_train,
        data_root=None  # path removed for release,
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
                color_type='color',
                load_cam_names=[
                    'fisheye_camera_front',
                    'fisheye_camera_left',
                    'fisheye_camera_right',
                    'fisheye_camera_rear',
                ],
                load_cam_type='cam_fisheye',
                to_float32=True,
                type='LoadFisheye3DODMultiViewImageFromFiles'),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(type='Fisheye3DODEvaluatorRangeFilter'),
            dict(
                input_img_keys=[
                    'cam_fisheye',
                ],
                keys=[
                    'cam_fisheye',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                ],
                meta_keys=[
                    'cam2img',
                    'cam2lidar',
                    'lidar2cam',
                    'lidar2img',
                    'box_type_3d',
                    'token',
                ],
                type='Fisheye3DODPackDetInputs'),
        ],
        test_mode=False,
        type='Fisheye3DODDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        color_type='color',
        load_cam_names=[
            'fisheye_camera_front',
            'fisheye_camera_left',
            'fisheye_camera_right',
            'fisheye_camera_rear',
        ],
        load_cam_type='cam_fisheye',
        to_float32=True,
        type='LoadFisheye3DODMultiViewImageFromFiles'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='Fisheye3DODEvaluatorRangeFilter'),
    dict(
        input_img_keys=[
            'cam_fisheye',
        ],
        keys=[
            'cam_fisheye',
            'gt_bboxes_3d',
            'gt_labels_3d',
        ],
        meta_keys=[
            'cam2img',
            'cam2lidar',
            'lidar2cam',
            'lidar2img',
            'box_type_3d',
            'token',
        ],
        type='Fisheye3DODPackDetInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file=
        ann_val,
        data_root=None  # path removed for release,
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
                color_type='color',
                load_cam_names=[
                    'fisheye_camera_front',
                    'fisheye_camera_left',
                    'fisheye_camera_right',
                    'fisheye_camera_rear',
                ],
                load_cam_type='cam_fisheye',
                to_float32=True,
                type='LoadFisheye3DODMultiViewImageFromFiles'),
            dict(
                input_img_keys=[
                    'cam_fisheye',
                ],
                keys=[
                    'cam_fisheye',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                ],
                meta_keys=[
                    'cam2img',
                    'cam2lidar',
                    'lidar2img',
                    'lidar2cam',
                    'sample_idx',
                    'token',
                    'img_path',
                    'lidar_path',
                    'num_pts_feats',
                    'box_type_3d',
                ],
                type='Fisheye3DODPackDetInputs'),
        ],
        test_mode=True,
        type='Fisheye3DODDataset'),
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
