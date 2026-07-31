"""Released top-100 ROSA R18 configuration for Fisheye3DOD Main126."""

import os
import pathlib
import sys

_project_root = pathlib.Path("{{ fileDirname }}").resolve().parent
_src_root = _project_root / "src"
_detector_host_root = pathlib.Path(
    os.environ.get(
        "ROSA_DETECTOR_HOST_ROOT",
        os.environ.get(
            "SPARSE4D_ROOT",
            _project_root / "external" / "detector_host",
        ),
    )
).resolve()
for _path in (_src_root, _detector_host_root):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

os.environ.setdefault(
    "FISHEYE3DOD_OFFICIAL_ROOT",
    str(_project_root / "external" / "fisheye3dod"),
)

data_root = str(
    pathlib.Path(
        os.environ.get(
            "FISHEYE3DOD_DATA_ROOT",
            _project_root.parent / "datasets" / "Fisheye3DODdataset",
        )
    ).resolve()
)
ann_root = pathlib.Path(
    os.environ.get(
        "FISHEYE3DOD_ANN_ROOT",
        pathlib.Path(data_root) / "ImageSets-main126",
    )
).resolve()
anchor_root = pathlib.Path(
    os.environ.get("FISHEYE3DOD_ANCHOR_ROOT", _project_root / "assets")
).resolve()
anchor_file = str(
    pathlib.Path(
        os.environ.get(
            "FISHEYE3DOD_ANCHOR",
            anchor_root / "fisheye3dod_main126_train_kmeans300_seed3407.npy",
        )
    ).resolve()
)
ann_train = str(ann_root / "fisheye3dod_infos_train.pkl")
ann_val = str(ann_root / "fisheye3dod_infos_val.pkl")

custom_imports = dict(
    imports=[
        "projects.mmdet3d_plugin",
        "rosa.integrations.sparse4d",
        "rosa.integrations.fisheye3dod.dataset",
        "rosa.integrations.fisheye3dod.model",
        "rosa.integrations.fisheye3dod.metric",
    ],
    allow_failed_imports=False,
)
default_scope = "mmdet3d"

seed = 3407
randomness = dict(seed=seed, deterministic=True)
find_unused_parameters = False

classes = ("Car", "Van", "Truck", "Bus", "Pedestrian", "Cyclist")
metainfo = dict(classes=classes)
cameras = (
    "fisheye_camera_front",
    "fisheye_camera_left",
    "fisheye_camera_right",
    "fisheye_camera_rear",
)
point_cloud_range = [-48.0, -48.0, -5.0, 48.0, 48.0, 5.0]
embed_dims = 256
num_levels = 4
num_groups = 8
num_decoder = 6
operation_order = ["deformable", "ffn", "norm", "refine"] + [
    "gnn",
    "norm",
    "deformable",
    "ffn",
    "norm",
    "refine",
] * 5
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
        depth=18,
        num_stages=4,
        frozen_stages=-1,
        norm_eval=False,
        style="pytorch",
        with_cp=False,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type="BN", requires_grad=True),
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet18"),
    ),
    img_neck=dict(
        type="mmdet.FPN",
        num_outs=num_levels,
        start_level=0,
        out_channels=embed_dims,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
        in_channels=[64, 128, 256, 512],
    ),
    head=dict(
        type="ROSARefinementHead",
        # One profile reproduces every published ROSA setting. Individual
        # fields are reserved for controlled ablations in frozen result configs.
        rosa_cfg=dict(profile="published"),
        num_decoder=num_decoder,
        num_single_frame_decoder=1,
        cls_threshold_to_reg=0.05,
        decouple_attn=True,
        max_temporal_gnn_interactions=0,
        cache_decoder_index=5,
        inference_output_decoder_index=5,
        inference_classification_decoder_index=5,
        inference_quality_decoder_index=5,
        inference_box_source_by_dimension=None,
        operation_order=operation_order,
        instance_bank=dict(
            type="InstanceBank",
            num_anchor=300,
            embed_dims=embed_dims,
            anchor=anchor_file,
            anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
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
                type="mmdet.CrossEntropyLoss",
                use_sigmoid=True,
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
    batch_size=4,
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
    batch_size=4,
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
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.5))),
    clip_grad=dict(
        max_norm=25,
        norm_type=2,
        error_if_nonfinite=True,
    ),
)
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
        end=20,
        T_max=20,
        by_epoch=True,
        eta_min_ratio=1e-3,
        convert_to_iter_based=True,
    ),
]
train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=20,
    val_interval=4,
)
val_cfg = dict()
test_cfg = dict()

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=20,
        save_best="fisheye_bev/NDS",
        rule="greater",
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
)
custom_hooks = [dict(type="SyncBuffersHook", priority="NORMAL")]
log_processor = dict(
    type="LogProcessor",
    window_size=50,
    by_epoch=True,
)
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(
        mp_start_method="fork",
        opencv_num_threads=0,
    ),
    dist_cfg=dict(backend="nccl"),
)
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="Det3DLocalVisualizer",
    vis_backends=vis_backends,
    name="visualizer",
)
log_level = "INFO"
load_from = None
resume = False
launcher = "pytorch"
work_dir = str(
    pathlib.Path(
        os.environ.get(
            "ROSA_WORK_DIR",
            _project_root / "work_dirs" / "rosa_r18_main126",
        )
    ).resolve()
)

experiment_contract = dict(
    dataset="Fisheye3DOD Main126",
    backbone="ImageNet-pretrained ResNet-18",
    camera_model="native 800x800 OCam",
    keypoints="7 fixed + 6 learned",
    queries=300,
    refined_queries=100,
    refinement_steps=2,
    temporal=False,
    global_batch_size=8,
    seed=seed,
    projection_contract="legacy_front_only",
)

# MMEngine serializes every remaining config variable when constructing a
# Runner; pathlib objects are setup helpers, not part of the experiment.
del _project_root, _src_root, _detector_host_root, _path, ann_root, anchor_root
