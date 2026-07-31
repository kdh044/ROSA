_decoder0 = [
    'deformable',
    'ffn',
    'norm',
    'refine',
]
_decoder_later = [
    'gnn',
    'norm',
    'deformable',
    'ffn',
    'norm',
    'refine',
]
anchor_file = '/home/airlab/user/kdh/fisheye3dod_sparse4d_r50_official_20260726/artifacts/fisheye3dod_main126_train_kmeans300_seed3407.npy'
ann_root = '/home/airlab/user/kdh/Fisheye3DODdataset/ImageSets-2hz'
ann_train = '/home/airlab/user/kdh/Fisheye3DODdataset/ImageSets-2hz/fisheye3dod_infos_train.pkl'
ann_val = '/home/airlab/user/kdh/Fisheye3DODdataset/ImageSets-2hz/fisheye3dod_infos_val.pkl'
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
        'projects.mmdet3d_plugin',
        'fisheye3dod_official.dataset',
        'fisheye3dod_official.model',
        'fisheye3dod_official.metric',
        'final_surface_rosa_rfl',
    ])
data_root = '/home/airlab/user/kdh/Fisheye3DODdataset'
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
embed_dims = 256
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
experiment_contract = dict(
    ablation=
    'ROSA camera Q-K and geometry scoring retained in the graph but effective valid-camera weights are uniform',
    backbone='ImageNet-pretrained ResNet-18',
    camera_model='official OCamCalib native 800x800',
    dataset='Fisheye3DOD',
    extra_surface_points=0,
    global_batch_size=8,
    joint_refinement=
    'shared observation/refinement trunk is applied twice; shape, tangential centre, range distribution and confidence use separate output heads',
    max_epochs=20,
    native_sparse4d_keypoints='7 fixed + 6 learnable = 13',
    observation=
    'native 13 points -> OCam -> SH-RFL -> point-wise camera-attention ROSA',
    quality_rule=
    'bounded localization-quality residual gated by detached learned radial confidence',
    radial_distribution=
    'per-query learned Gaussian over log(GT range/current range); five standard-deviation points are reprojected',
    ranking='native Sparse4D class-times-centerness; quality ablation off',
    refine_selection=
    'top 100 of 300 queries by untouched Sparse4D max-class-probability times centerness',
    seed=3407,
    split='local-main126-8820-train-3906-val',
    step1=
    'one current-box observation plus five learned log-Gaussian range hypotheses',
    step2=
    'one updated-box observation with shared shape, tangent and direct log-range heads; no second candidate search',
    temporal=False,
    unselected_queries='bitwise-identical native Sparse4D boxes and scores',
    update_rule=
    'full box update -> OCam reprojection -> full box update, with shared weights across the two iterations'
)
find_unused_parameters = False
fixed_keypoints = [
    [
        0.0,
        0.0,
        0.0,
    ],
    [
        0.45,
        0.0,
        0.0,
    ],
    [
        -0.45,
        0.0,
        0.0,
    ],
    [
        0.0,
        0.45,
        0.0,
    ],
    [
        0.0,
        -0.45,
        0.0,
    ],
    [
        0.0,
        0.0,
        0.45,
    ],
    [
        0.0,
        0.0,
        -0.45,
    ],
]
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
    head=dict(
        anchor_encoder=dict(
            embed_dims=[
                128,
                32,
                32,
                64,
            ],
            in_loops=1,
            mode='cat',
            out_loops=4,
            output_fc=False,
            type='SparseBox3DEncoder',
            vel_dims=2),
        cache_decoder_index=5,
        canonical_task_splice_cfg=dict(
            anchor_z_is_bottom=False,
            contract_version=
            'late-canonical-geometry/2026-07-25/v15-native-joint-score-residual',
            generate_fisheye_mask=False,
            min_patch_samples=1,
            min_projected_fpn_cells=1.0,
            num_classes=6,
            patch_cols=1,
            patch_rows=1,
            point_cloud_xy_half_extent=(
                48.0,
                48.0,
            ),
            projection_type='ocam_fisheye'),
        closed_loop_rosa_cfg=dict(
            candidate_loss_weight=0.5,
            center_keep_loss_weight=1.0,
            center_move_loss_weight=1.0,
            detach_candidate_geometry=True,
            distribution_nll_weight=0.25,
            embed_dims=256,
            enable_quality_score=True,
            enable_radial=True,
            enable_shape1=True,
            enable_shape2_feedback=True,
            enable_tangent=True,
            hidden_dims=128,
            initial_log_range_std=0.05,
            intermediate_step_loss_weight=0.5,
            keep_threshold_m=0.5,
            max_candidate_range_m=48.0,
            max_log_range_std=0.5,
            max_log_size_delta=0.2,
            max_quality_delta_logit=1.0,
            max_shape2_log_size_delta=0.05,
            max_shape2_yaw_delta_rad=0.08,
            max_tangent_delta_m=0.75,
            max_yaw_delta_rad=0.25,
            min_candidate_range_m=0.5,
            min_log_range_std=0.01,
            num_cameras=4,
            num_keypoints=13,
            num_levels=4,
            num_refinement_steps=2,
            quality_loss_weight=0.0,
            radial_offset_loss_weight=0.5,
            radial_proposal_mode='learned_log_gaussian',
            refine_topk=100,
            second_step_candidate_search=False,
            shape_loss_weight=1.0,
            tangent_loss_weight=0.25,
            uniform_camera_fusion=True),
        cls_threshold_to_reg=0.05,
        decoder=dict(num_output=300, type='SparseBox3DDecoder'),
        decouple_attn=True,
        deformable_model=dict(
            attn_drop=0.15,
            embed_dims=256,
            kps_generator=dict(
                anchor_z_is_bottom=False,
                fix_scale=[
                    [
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.45,
                        0.0,
                        0.0,
                    ],
                    [
                        -0.45,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.45,
                        0.0,
                    ],
                    [
                        0.0,
                        -0.45,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.45,
                    ],
                    [
                        0.0,
                        0.0,
                        -0.45,
                    ],
                ],
                num_learnable_pts=6,
                type='SparseBox3DKeyPointsGenerator'),
            num_cams=4,
            num_groups=8,
            num_levels=4,
            projection_type='ocam_fisheye',
            residual_mode='cat',
            type='DeformableFeatureAggregation',
            use_camera_embed=True,
            use_deformable_func=True),
        ffn=dict(
            act_cfg=dict(inplace=True, type='ReLU'),
            embed_dims=256,
            feedforward_channels=1024,
            ffn_drop=0.1,
            in_channels=512,
            num_fcs=2,
            pre_norm=dict(type='LN'),
            type='AsymmetricFFN'),
        graph_model=dict(
            batch_first=True,
            dropout=0.1,
            embed_dims=512,
            num_heads=8,
            type='MultiheadAttention'),
        inference_box_source_by_dimension=None,
        inference_classification_decoder_index=5,
        inference_output_decoder_index=5,
        inference_quality_decoder_index=5,
        instance_bank=dict(
            anchor=
            '/home/airlab/user/kdh/fisheye3dod_sparse4d_r50_official_20260726/artifacts/fisheye3dod_main126_train_kmeans300_seed3407.npy',
            anchor_handler=dict(
                anchor_z_is_bottom=False,
                type='SparseBox3DKeyPointsGenerator'),
            confidence_decay=0.6,
            embed_dims=256,
            feat_grad=False,
            num_anchor=300,
            num_temp_instances=0,
            type='InstanceBank'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=2.0,
            type='mmdet.FocalLoss',
            use_sigmoid=True),
        loss_reg=dict(
            loss_box=dict(loss_weight=0.25, type='mmdet.L1Loss'),
            loss_centerness=dict(
                type='mmdet.CrossEntropyLoss', use_sigmoid=True),
            loss_yawness=dict(type='mmdet.GaussianFocalLoss'),
            type='SparseBox3DLoss'),
        max_temporal_gnn_interactions=0,
        norm_layer=dict(normalized_shape=256, type='LN'),
        num_decoder=6,
        num_single_frame_decoder=1,
        operation_order=[
            'deformable',
            'ffn',
            'norm',
            'refine',
            'gnn',
            'norm',
            'deformable',
            'ffn',
            'norm',
            'refine',
            'gnn',
            'norm',
            'deformable',
            'ffn',
            'norm',
            'refine',
            'gnn',
            'norm',
            'deformable',
            'ffn',
            'norm',
            'refine',
            'gnn',
            'norm',
            'deformable',
            'ffn',
            'norm',
            'refine',
            'gnn',
            'norm',
            'deformable',
            'ffn',
            'norm',
            'refine',
        ],
        refine_layer=dict(
            embed_dims=256,
            num_cls=6,
            refine_yaw=True,
            type='SparseBox3DRefinementModule',
            with_quality_estimation=True),
        reg_weights=[
            2.0,
            2.0,
            2.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        sampler=dict(
            add_neg_dn=True,
            box_weight=0.25,
            cls_weight=2.0,
            dn_noise_scale=[
                2.0,
                2.0,
                2.0,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            max_dn_gt=32,
            num_dn_groups=5,
            num_temp_dn_groups=0,
            reg_weights=[
                2.0,
                2.0,
                2.0,
                0.5,
                0.5,
                0.5,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            type='SparseBox3DTarget'),
        temp_graph_model=dict(
            batch_first=True,
            dropout=0.1,
            embed_dims=512,
            num_heads=8,
            type='MultiheadAttention'),
        type='ClosedLoopROSAHead'),
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
    type='Fisheye3DODGravityCenterSparse4D',
    use_deformable_func=True,
    use_grid_mask=False)
num_decoder = 6
num_groups = 8
num_levels = 4
operation_order = [
    'deformable',
    'ffn',
    'norm',
    'refine',
    'gnn',
    'norm',
    'deformable',
    'ffn',
    'norm',
    'refine',
    'gnn',
    'norm',
    'deformable',
    'ffn',
    'norm',
    'refine',
    'gnn',
    'norm',
    'deformable',
    'ffn',
    'norm',
    'refine',
    'gnn',
    'norm',
    'deformable',
    'ffn',
    'norm',
    'refine',
    'gnn',
    'norm',
    'deformable',
    'ffn',
    'norm',
    'refine',
]
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
strides = [
    4,
    8,
    16,
    32,
]
test_cfg = dict()
test_dataloader = dict(
    batch_size=4,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file=
        '/home/airlab/user/kdh/Fisheye3DODdataset/ImageSets-main126/fisheye3dod_infos_val.pkl',
        data_root='/home/airlab/user/kdh/Fisheye3DODdataset',
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
            dict(ocam_lens_fov_deg=220.0, type='PackSparse4DInputs'),
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
    dict(ocam_lens_fov_deg=220.0, type='PackSparse4DInputs'),
]
train_cfg = dict(max_epochs=20, type='EpochBasedTrainLoop', val_interval=4)
train_dataloader = dict(
    batch_size=4,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file=
        '/home/airlab/user/kdh/Fisheye3DODdataset/ImageSets-main126/fisheye3dod_infos_train.pkl',
        data_root='/home/airlab/user/kdh/Fisheye3DODdataset',
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
            dict(type='OfficialFisheye3DODRangeFilter'),
            dict(ocam_lens_fov_deg=220.0, type='PackSparse4DInputs'),
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
    dict(type='OfficialFisheye3DODRangeFilter'),
    dict(ocam_lens_fov_deg=220.0, type='PackSparse4DInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=4,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file=
        '/home/airlab/user/kdh/Fisheye3DODdataset/ImageSets-main126/fisheye3dod_infos_val.pkl',
        data_root='/home/airlab/user/kdh/Fisheye3DODdataset',
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
            dict(ocam_lens_fov_deg=220.0, type='PackSparse4DInputs'),
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
work_dir = '/home/airlab/user/kdh/fisheye3dod_sparse4d_r50_official_20260726/work_dirs/r18_iterative_distribution_rosa_top100_uniform_camera_main126_20e_2gpu_b4_seed3407_val4'
