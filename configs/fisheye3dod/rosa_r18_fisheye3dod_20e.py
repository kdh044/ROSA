# ROSA on Fisheye3DOD (Main126 split), ResNet-18 backbone, 20 epochs.
#
# This is the configuration that produces the Fisheye3DOD numbers in the paper
# and enables the fused CUDA observation path used for the reported runtime.
# Set ROSA_DATA_ROOT if your dataset does not live at ./data/Fisheye3DOD.

import os

seed = int(os.environ.get("ROSA_SEED", 3407))
data_root = os.environ.get("ROSA_DATA_ROOT", "data/Fisheye3DOD")
ann_root = os.path.join(data_root, "ImageSets-main126")
anchor_file = "assets/fisheye3dod_main126_train_kmeans300_seed3407.npy"
ann_train = os.path.join(ann_root, "fisheye3dod_infos_train.pkl")
ann_val = os.path.join(ann_root, "fisheye3dod_infos_val.pkl")

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
    dict(milestones=(8, ), type='ROSASparseMilestoneCheckpointHook'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'projects.mmdet3d_plugin',
        'rosa.integrations.sparse4d',
        'rosa.integrations.optim',
        'rosa.integrations.fisheye3dod.dataset',
        'rosa.integrations.fisheye3dod.model',
        'rosa.integrations.fisheye3dod.metric',
        'rosa.integrations.fisheye_ray_sparse4d',
        'rosa.integrations.fisheye_box_sparse4d',
    ])
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=5,
        rule=[
            'greater',
            'greater',
        ],
        save_best=[
            'fisheye_bev/mAP',
            'fisheye_bev/NDS',
        ],
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
    arm='Fisheye-Box-Decoder-V49-AllStageSparse4DSizeHead-raw-uniform-20e',
    backbone='ImageNet-pretrained ResNet-18',
    baseline_arm=
    'Fisheye-Box-Decoder-V46-RFLBeforeROSA-SinglePass-raw-uniform-20e',
    cameras='four native 800x800 fisheye views',
    controlled_change=
    'replace the compact dedicated log-size expert at D0-D4 with the same four-hidden-block, two-LayerNorm capacity already used by V46 at D5, so all D0-D5 extent updates use one consistent C0-capacity task-isolated architecture',
    dataset='Fisheye3DOD Main126',
    decision_protocol=
    'train fresh for all 20 epochs and choose the next experiment only from the epoch-20 result; intermediate validation is diagnostic only',
    decoders=6,
    detector_baseline='C0-corrected-Sparse4D-raw-uniform-20e',
    diagnostic_parent='V46-e20-C0-same-query-size-route',
    distributed_world_size=2,
    free_box_queries=300,
    global_batch_size=8,
    hypothesis=
    "same-query routing shows that V46 recognition and pose satisfy the joint target when paired with C0 extent, while V48's alternative RFL-free evidence makes extent worse; the remaining structural difference is that C0 gives every recurrent stage native regression capacity whereas V46 gives it only to terminal D5",
    initializer=
    'no separate pre-D0 projection or FPN sampling; D0 first updates its query from the centre token already present in its 13-point read',
    invariants=
    'retain V46 camera-RFL-before-ROSA, post-fusion point-RFL, learned update gates, size inputs, losses, queries, 13 keypoints, D0-D5 schedule, temporal-off setting, seed, data, and single FPN/DAF read; apply the same rule to every stage and add no manual scale, clamp, class prior, stage selection, loss coefficient, or image read',
    keypoints='7 fixed semantic + 6 query-learned per D0-D5 stage',
    launch_gate=
    'run only if V32 wins the yaw diagnostics and V35 improves mASE without an unacceptable mAP or NDS regression',
    losses=
    'D0-D5 angular and log-range mean supervision; detached non-negative Gaussian excess-NLL uncertainty calibration with no physical sigma bounds',
    max_epochs=20,
    not_used=
    '2D detector, dense depth head, BEV conversion, probabilistic tracking decoder, cross-frame temporal cache, pre-D0 observation, post-D5 sidecar',
    observation=
    'per-point RFL; ROSA camera-level attention with analytic fisheye Jacobian; query-conditioned 13-point attention',
    per_device_batch_size=4,
    refinement=
    'current-state-conditioned residual with independently learned center, size, yaw, and velocity sigmoid update gates; no physical update bounds',
    runtime_hypothesis=
    'retain useful centre evidence without a seventh image-read path',
    seed=seed,
    state=
    'ray direction + log-range center; global log-size, sin/cos yaw, and Cartesian velocity',
    temporal=False,
    train_sampling='raw-uniform DefaultSampler (no CBGSDataset)',
    training_dn_groups=5,
    uncertainty_optimization=
    'detached uncertainty readouts retain the matched optimizer settings but use an independent gradient-clipping norm'
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
        cls_threshold_to_reg=0.05,
        decoder=dict(num_output=300, type='SparseBox3DDecoder'),
        inference_classification_decoder_index=5,
        inference_output_decoder_index=5,
        inference_quality_decoder_index=5,
        instance_bank=dict(
            anchor=
            anchor_file,
            anchor_handler=dict(
                anchor_z_is_bottom=False,
                type='ROSANormalizedKeyPointsGenerator'),
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
            type='ROSAEvaluatorAlignedSparseBox3DLoss'),
        num_decoder=6,
        ray_aux_loss=dict(
            angular_weight=0.25,
            beta=0.1,
            log_range_l1_weight=0.25,
            range_nll_weight=0.1,
            range_uncertainty_objective='detached_excess_nll'),
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
        type='FisheyeSparseBoxHead',
        v4_decoder_cfg=dict(
            attention_dims=64,
            camera_rfl_before_rosa=True,
            streaming_camera_rfl_cuda=True,
            checkpoint_point_chunks=False,
            d0_center_prefusion=True,
            dedicated_size_head=True,
            detach_range_uncertainty_input=True,
            dropout=0.1,
            embed_dims=256,
            ffn_dims=512,
            fixed_keypoint_scale=0.45,
            learned_face_yaw_gate=True,
            learned_update_gate=True,
            level_strides=(
                4.0,
                8.0,
                16.0,
                32.0,
            ),
            num_cameras=4,
            num_classes=6,
            num_heads=8,
            num_layers=6,
            num_learnable_points=6,
            num_queries=300,
            point_chunk_size=2,
            predict_range_uncertainty=True,
            separate_face_yaw_residual=True,
            sparse4d_size_head_layers=(
                0,
                1,
                2,
                3,
                4,
                5,
            ),
            sparse4d_task_heads=True,
            state_conditioned_refinement=True,
            state_conditioned_task_heads=True,
            structured_face_refinement=True,
            structured_face_refinement_targets=('yaw', ),
            use_fused_daf=True)),
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
    use_deformable_func=False,
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
    paramwise_cfg=dict(
        custom_keys=dict({
            'head.fisheye_box_decoder.range_uncertainty_heads':
            dict(rosa_sidecar=True),
            'img_backbone':
            dict(lr_mult=0.5)
        })),
    type='ROSAPairedClipOptimWrapper')
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
profile = 'balanced'
randomness = dict(deterministic=True, seed=seed)
resume = False
sealed_resume_contract = None
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
work_dir = 'work_dirs/rosa_r18_fisheye3dod_20e'
