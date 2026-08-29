# ROSA on KITTI-360, VoVNet-99 backbone, 24 epochs, six classes.
#
# KITTI-360 mixes a rectified pinhole stereo pair with two 205-degree MEI
# fisheyes, so this arm exercises the mixed-lens projection path.  Set
# ROSA_KITTI360_ROOT if the dataset does not live at ./data/KITTI-360, and
# point pretrained_backbone at the remapped FCOS3D VoVNet-99 weights
# described in docs/DATA.md.

import os

seed = int(os.environ.get("ROSA_SEED", 3407))
data_root = os.environ.get("ROSA_KITTI360_ROOT", "data/KITTI-360")
ann_root = os.path.join(data_root, "ImageSets-rosa-6class")
anchor_file = "assets/kitti360_rosa_train_kmeans300_seed3407.npy"
ann_train = os.path.join(ann_root, "kitti360_rosa_infos_train.pkl")
ann_val = os.path.join(ann_root, "kitti360_rosa_infos_val.pkl")
pretrained_backbone = os.environ.get(
    "ROSA_VOVNET_CKPT", "weights/fcos3d_vovnet_imgbackbone-remapped.pth"
)

_ann_root = ann_root
_kitti360_root = data_root
_pipeline_common_load = dict(
    cameras=(
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT',
        'CAM_BACK_LEFT',
    ),
    data_root=data_root,
    to_float32=True,
    type='LoadKITTI360MixedCameraImages')
anchor_file = anchor_file
ann_train = ann_train
ann_val = ann_val
cameras = (
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_RIGHT',
    'CAM_BACK_LEFT',
)
classes = (
    'car',
    'truck',
    'trailer',
    'motorcycle',
    'bicycle',
    'pedestrian',
)
custom_hooks = [
    dict(priority='NORMAL', type='SyncBuffersHook'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'projects.mmdet3d_plugin',
        'rosa.integrations.sparse4d_components',
        'rosa.integrations.optim',
        'rosa.integrations.fisheye3dod.model',
        'rosa.integrations.fisheye_ray_sparse4d',
        'rosa.integrations.fisheye_box_sparse4d',
        'rosa.integrations.kitti360.dataset',
        'rosa.integrations.kitti360.metric',
        'rosa.integrations.kitti360.backbone',
    ])
data_root = data_root
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=5,
        rule=[
            'greater',
            'greater',
        ],
        save_best=[
            'kitti360_bev/mAP',
            'kitti360_bev/NDS',
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
    approximations='see TABLE2_KITTI360_DAPETR_PARITY/ASSUMPTIONS.md',
    arm='KITTI360-ROSA-V49Logic-VOV99-6class-20e',
    counterpart='dapetr_kitti360_6class_vov99_20e.py',
    dataset=
    'KITTI-360 converted by the official FishBEVOD converter, 2 rectified pinhole + 2 native MEI fisheye cameras',
    distributed_world_size=2,
    global_batch_size=8,
    label_space=
    'car, truck, trailer, motorcycle, bicycle, pedestrian (riders merged into pedestrian, as in nuScenes); see rosa/integrations/kitti360/constants.py for why the FishBEVOD eleven-key range list is not used verbatim',
    max_epochs=20,
    new_for_kitti360=
    'projection_type=mixed_kitti360 selects the pinhole model for CAM_FRONT/CAM_FRONT_RIGHT and the MEI model for the two fisheye cameras; six driving-relevant classes; everything else (ResNet-18 from ImageNet, 300 queries, AdamW 2e-4, 20 epochs, seed 3407) is the Fisheye3DOD V49 setting, and every baseline in this table is trained with the same backbone, schedule and label space',
    per_device_batch_size=4,
    reference_metric=
    'DAPETR reports mAP 0.286 / NDS 0.306 on its own ten-class label space with a VoVNet-99 backbone, 24 epochs and an initialisation from a fisheye-pretrained detector -- cited for context only, not as a like-for-like row: every arm of this table is retrained here under the settings below',
    reference_paper='DAPETR (arxiv 2606.08680, ICRA 2026)',
    reused_v49_logic=
    'RFE (ray feature encoding), HFA (level/camera/point hierarchical attention), HybridBoxRefinement (gated spherical-center + log-size + yaw/face-correction residuals) -- all v4_decoder_cfg flags kept identical to the Fisheye3DOD V49 production config',
    seed=seed)
find_unused_parameters = True
launcher = 'pytorch'
load_from = pretrained_backbone
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
metainfo = dict(
    classes=(
        'car',
        'truck',
        'trailer',
        'motorcycle',
        'bicycle',
        'pedestrian',
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
                type='NormalizedSparseBoxKeyPointsGenerator'),
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
            type='EvaluatorAlignedSparseBox3DLoss'),
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
            projection_type='mixed_kitti360',
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
        frozen_stages=-1,
        init_cfg=None,
        input_ch=3,
        norm_eval=False,
        out_features=(
            'stage2',
            'stage3',
            'stage4',
            'stage5',
        ),
        spec_name='V-99-eSE',
        type='VoVNetCP',
        with_cp=False),
    img_neck=dict(
        add_extra_convs='on_output',
        in_channels=[
            256,
            512,
            768,
            1024,
        ],
        num_outs=4,
        out_channels=256,
        relu_before_extra_convs=True,
        start_level=0,
        type='mmdet.FPN'),
    type='Fisheye3DODGravityCenterSparse4D',
    use_deformable_func=False,
    use_grid_mask=False)
num_classes = 6
num_decoder = 6
num_groups = 8
num_levels = 4
num_queries = 300
optim_wrapper = dict(
    clip_grad=dict(error_if_nonfinite=True, max_norm=25, norm_type=2),
    optimizer=dict(lr=0.0002, type='AdamW', weight_decay=0.001),
    paramwise_cfg=dict(
        custom_keys=dict({
            'head.fisheye_box_decoder.range_uncertainty_heads':
            dict(separate_clip_group=True),
            'img_backbone':
            dict(lr_mult=0.5)
        })),
    type='PartitionedClipOptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=500,
        start_factor=0.3333333333333333,
        type='LinearLR'),
    dict(
        T_max=24,
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=24,
        eta_min_ratio=0.001,
        type='CosineAnnealingLR'),
]
point_cloud_range = [
    -51.2,
    -51.2,
    -5.0,
    51.2,
    51.2,
    3.0,
]
profile = 'balanced'
randomness = dict(deterministic=True, seed=seed)
resume = False
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
                'car',
                'truck',
                'trailer',
                'motorcycle',
                'bicycle',
                'pedestrian',
            )),
        pipeline=[
            dict(
                cameras=(
                    'CAM_FRONT',
                    'CAM_FRONT_RIGHT',
                    'CAM_BACK_RIGHT',
                    'CAM_BACK_LEFT',
                ),
                data_root=data_root,
                to_float32=True,
                type='LoadKITTI360MixedCameraImages'),
            dict(training=False, type='DAPETRImageAugment'),
            dict(type='PackSparse4DInputs'),
        ],
        test_mode=True,
        type='KITTI360MixedCameraDataset',
        with_velocity=True),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    ann_val,
    prefix='kitti360_bev',
    type='KITTI360DAPETRMetric')
test_pipeline = [
    dict(
        cameras=(
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_BACK_RIGHT',
            'CAM_BACK_LEFT',
        ),
        data_root=data_root,
        to_float32=True,
        type='LoadKITTI360MixedCameraImages'),
    dict(training=False, type='DAPETRImageAugment'),
    dict(type='PackSparse4DInputs'),
]
train_cfg = dict(max_epochs=24, type='EpochBasedTrainLoop', val_interval=4)
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
                'car',
                'truck',
                'trailer',
                'motorcycle',
                'bicycle',
                'pedestrian',
            )),
        pipeline=[
            dict(
                cameras=(
                    'CAM_FRONT',
                    'CAM_FRONT_RIGHT',
                    'CAM_BACK_RIGHT',
                    'CAM_BACK_LEFT',
                ),
                data_root=data_root,
                to_float32=True,
                type='LoadKITTI360MixedCameraImages'),
            dict(training=True, type='DAPETRImageAugment'),
            dict(training=True, type='DAPETRGlobalRotScale'),
            dict(type='KITTI360ObjectRangeFilter'),
            dict(type='PackSparse4DInputs'),
        ],
        test_mode=False,
        type='KITTI360MixedCameraDataset',
        with_velocity=True),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        cameras=(
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_BACK_RIGHT',
            'CAM_BACK_LEFT',
        ),
        data_root=data_root,
        to_float32=True,
        type='LoadKITTI360MixedCameraImages'),
    dict(training=True, type='DAPETRImageAugment'),
    dict(training=True, type='DAPETRGlobalRotScale'),
    dict(type='KITTI360ObjectRangeFilter'),
    dict(type='PackSparse4DInputs'),
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
                'car',
                'truck',
                'trailer',
                'motorcycle',
                'bicycle',
                'pedestrian',
            )),
        pipeline=[
            dict(
                cameras=(
                    'CAM_FRONT',
                    'CAM_FRONT_RIGHT',
                    'CAM_BACK_RIGHT',
                    'CAM_BACK_LEFT',
                ),
                data_root=data_root,
                to_float32=True,
                type='LoadKITTI360MixedCameraImages'),
            dict(training=False, type='DAPETRImageAugment'),
            dict(type='PackSparse4DInputs'),
        ],
        test_mode=True,
        type='KITTI360MixedCameraDataset',
        with_velocity=True),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    ann_val,
    prefix='kitti360_bev',
    type='KITTI360DAPETRMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'work_dirs/rosa_vov99_kitti360_24e'
