"""Official-split, native-OCam, six-class Sparse4D-R50 B0 contract."""

import pathlib
import os
import sys

_package_root = pathlib.Path(
    os.environ.get(
        "FISHEYE3DOD_PACKAGE_ROOT",
        "/home/airlab/user/kdh/"
        "fisheye3dod_sparse4d_r50_official_20260726",
    )
).resolve()
_workspace_root = _package_root.parent
_sparse4d_root = pathlib.Path(
    os.environ.get(
        "SPARSE4D_ROOT", _workspace_root / "sphere_stream_sparse4d"
    )
).resolve()
for _path in (_package_root, _sparse4d_root):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

custom_imports = dict(
    imports=[
        "projects.mmdet3d_plugin",
        "fisheye3dod_official.dataset",
        "fisheye3dod_official.model",
        "fisheye3dod_official.metric",
    ],
    allow_failed_imports=False,
)
default_scope = "mmdet3d"

seed = 3407
randomness = dict(seed=seed, deterministic=True)
classes = ("Car", "Van", "Truck", "Bus", "Pedestrian", "Cyclist")
metainfo = dict(classes=classes)
cameras = (
    "fisheye_camera_front",
    "fisheye_camera_left",
    "fisheye_camera_right",
    "fisheye_camera_rear",
)
point_cloud_range = [-48.0, -48.0, -5.0, 48.0, 48.0, 5.0]
strides = [4, 8, 16, 32]
num_levels = 4
embed_dims = 256
num_groups = 8
num_decoder = 6

data_root = str(
    pathlib.Path(
        os.environ.get(
            "FISHEYE3DOD_DATA_ROOT",
            _workspace_root / "Fisheye3DODdataset",
        )
    ).resolve()
)
ann_root = str(
    pathlib.Path(
        os.environ.get(
            "FISHEYE3DOD_ANN_ROOT",
            pathlib.Path(data_root) / "ImageSets-2hz",
        )
    ).resolve()
)
ann_train = str(pathlib.Path(ann_root) / "fisheye3dod_infos_train.pkl")
ann_val = str(pathlib.Path(ann_root) / "fisheye3dod_infos_val.pkl")
anchor_file = str(
    pathlib.Path(
        os.environ.get(
            "FISHEYE3DOD_ANCHOR",
            _package_root
            / "artifacts"
            / "fisheye3dod_main126_train_kmeans300_seed3407.npy",
        )
    ).resolve()
)

_decoder0 = ["deformable", "ffn", "norm", "refine"]
_decoder_later = [
    "gnn",
    "norm",
    "deformable",
    "ffn",
    "norm",
    "refine",
]
operation_order = _decoder0 + _decoder_later * 5
fixed_keypoints = [
    [0.0, 0.0, 0.0],
    [0.45, 0.0, 0.0],
    [-0.45, 0.0, 0.0],
    [0.0, 0.45, 0.0],
    [0.0, -0.45, 0.0],
    [0.0, 0.0, 0.45],
    [0.0, 0.0, -0.45],
]

model = dict(
    type="Fisheye3DODGravityCenterSparse4D",
    use_grid_mask=False,
    use_deformable_func=True,
    data_preprocessor=dict(
        type="Sparse4DDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    img_backbone=dict(
        type="mmdet.ResNet",
        depth=50,
        num_stages=4,
        frozen_stages=-1,
        norm_eval=False,
        style="pytorch",
        with_cp=False,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type="BN", requires_grad=True),
        # The audited original Sparse4Dv3 checkpoint supplies all R50
        # tensors.  Disabling an extra torchvision load keeps template
        # construction offline and deterministic.
        init_cfg=None,
    ),
    img_neck=dict(
        type="mmdet.FPN",
        num_outs=num_levels,
        start_level=0,
        out_channels=embed_dims,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048],
    ),
    head=dict(
        type="Sparse4DHead",
        num_decoder=num_decoder,
        num_single_frame_decoder=1,
        cls_threshold_to_reg=0.05,
        decouple_attn=True,
        max_temporal_gnn_interactions=0,
        cache_decoder_index=5,
        inference_output_decoder_index=5,
        inference_classification_decoder_index=5,
        inference_quality_decoder_index=5,
        operation_order=operation_order,
        instance_bank=dict(
            type="InstanceBank",
            num_anchor=300,
            embed_dims=embed_dims,
            anchor=anchor_file,
            anchor_handler=dict(
                type="SparseBox3DKeyPointsGenerator",
                anchor_z_is_bottom=False,
            ),
            num_temp_instances=0,
            confidence_decay=0.6,
            feat_grad=False,
        ),
        anchor_encoder=dict(
            type="SparseBox3DEncoder",
            vel_dims=2,
            embed_dims=[128, 32, 32, 64],
            mode="cat",
            output_fc=False,
            in_loops=1,
            out_loops=4,
        ),
        graph_model=dict(
            type="MultiheadAttention",
            embed_dims=embed_dims * 2,
            num_heads=num_groups,
            batch_first=True,
            dropout=0.1,
        ),
        temp_graph_model=dict(
            type="MultiheadAttention",
            embed_dims=embed_dims * 2,
            num_heads=num_groups,
            batch_first=True,
            dropout=0.1,
        ),
        norm_layer=dict(type="LN", normalized_shape=embed_dims),
        ffn=dict(
            type="AsymmetricFFN",
            in_channels=embed_dims * 2,
            pre_norm=dict(type="LN"),
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
            num_fcs=2,
            ffn_drop=0.1,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        deformable_model=dict(
            type="DeformableFeatureAggregation",
            embed_dims=embed_dims,
            num_groups=num_groups,
            num_levels=num_levels,
            num_cams=4,
            attn_drop=0.15,
            use_deformable_func=True,
            use_camera_embed=True,
            residual_mode="cat",
            projection_type="ocam_fisheye",
            kps_generator=dict(
                type="SparseBox3DKeyPointsGenerator",
                num_learnable_pts=6,
                anchor_z_is_bottom=False,
                fix_scale=fixed_keypoints,
            ),
        ),
        refine_layer=dict(
            type="SparseBox3DRefinementModule",
            embed_dims=embed_dims,
            num_cls=6,
            refine_yaw=True,
            with_quality_estimation=True,
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
        loss_cls=dict(
            type="mmdet.FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0,
        ),
        loss_reg=dict(
            type="SparseBox3DLoss",
            loss_box=dict(type="mmdet.L1Loss", loss_weight=0.25),
            loss_centerness=dict(
                type="mmdet.CrossEntropyLoss", use_sigmoid=True
            ),
            loss_yawness=dict(type="mmdet.GaussianFocalLoss"),
        ),
        decoder=dict(type="SparseBox3DDecoder", num_output=300),
        reg_weights=[2.0] * 3 + [1.0] * 7,
    ),
)

train_pipeline = [
    dict(
        type="LoadOfficialFisheye3DODForSparse4D",
        cam_names=cameras,
        to_float32=True,
    ),
    dict(
        type="LoadAnnotations3D",
        with_bbox=False,
        with_bbox_3d=True,
        with_label=False,
        with_label_3d=True,
    ),
    dict(type="OfficialFisheye3DODRangeFilter"),
    dict(type="PackSparse4DInputs", ocam_lens_fov_deg=220.0),
]
test_pipeline = [
    dict(
        type="LoadOfficialFisheye3DODForSparse4D",
        cam_names=cameras,
        to_float32=True,
    ),
    dict(type="PackSparse4DInputs", ocam_lens_fov_deg=220.0),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    collate_fn=dict(type="default_collate"),
    dataset=dict(
        type="Fisheye3DODSparse4DDataset",
        data_root=data_root,
        ann_file=ann_train,
        pipeline=train_pipeline,
        test_mode=False,
        filter_empty_gt=True,
        metainfo=metainfo,
        with_velocity=True,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    collate_fn=dict(type="default_collate"),
    dataset=dict(
        type="Fisheye3DODSparse4DDataset",
        data_root=data_root,
        ann_file=ann_val,
        pipeline=test_pipeline,
        test_mode=True,
        filter_empty_gt=False,
        metainfo=metainfo,
        with_velocity=True,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="Fisheye3DODReleasedMetricBridge",
    prefix="fisheye_bev",
)
test_evaluator = val_evaluator

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=2e-4, weight_decay=0.001),
    paramwise_cfg=dict(
        custom_keys=dict(img_backbone=dict(lr_mult=0.5))
    ),
    clip_grad=dict(max_norm=25, norm_type=2, error_if_nonfinite=True),
)

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=30, val_interval=1)
val_cfg = dict()
test_cfg = dict()
param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1.0 / 3.0,
        by_epoch=False,
        begin=0,
        end=500,
    ),
    dict(
        type="CosineAnnealingLR",
        begin=0,
        end=30,
        T_max=30,
        by_epoch=True,
        eta_min_ratio=1e-3,
        convert_to_iter_based=True,
    ),
]

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=3,
        save_best="fisheye_bev/NDS",
        rule="greater",
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
)
custom_hooks = [dict(type="SyncBuffersHook", priority="NORMAL")]
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="Det3DLocalVisualizer",
    vis_backends=vis_backends,
    name="visualizer",
)
log_level = "INFO"
launcher = "none"
find_unused_parameters = False
resume = False

experiment_contract = dict(
    dataset="Fisheye3DOD",
    split="released-2hz-10080-train-4464-val",
    classes=classes,
    camera_model="official OCamCalib native 800x800",
    gravity_center_boxes=True,
    anchor_z_is_bottom=False,
    temporal=False,
    decoder_count=6,
    query_and_output_budget=300,
    evaluator="released exact behavior; no added 50-box truncation",
    source="original sparse4dv3_r50.pth semantic temporal-free transfer",
    fair_comparison_epoch_budget=30,
    seed=seed,
)

del _path, _sparse4d_root, _workspace_root, _package_root
