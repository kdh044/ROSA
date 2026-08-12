"""First 20-epoch 4-fisheye box-free ray-decoder experiment.

The data, backbone, query count, optimizer, batch size, seed, matcher, output
decoder, and evaluator are inherited from the corrected raw-uniform C0 arm.
The intervention replaces Sparse4D's box InstanceBank, 13-keypoint deformable
aggregation, and Cartesian refinement with six propagated fisheye ray stages.
"""

_base_ = "./rosa_r18_main126_v5_c0_sparse4d_raw_uniform_30e.py"

custom_imports = dict(
    imports=[
        "projects.mmdet3d_plugin",
        "rosa.integrations.sparse4d",
        "rosa.integrations.optim",
        "rosa.integrations.fisheye3dod.dataset",
        "rosa.integrations.fisheye3dod.model",
        "rosa.integrations.fisheye3dod.metric",
        "rosa.integrations.fisheye_ray_sparse4d",
    ],
    allow_failed_imports=False,
)

model = dict(
    # The V2 head consumes ordinary [B,C,D,H,W] FPN levels.  It does not use
    # the Sparse4D CUDA deformable-feature packing or its 13-point kernel.
    use_deformable_func=False,
    head=dict(
        _delete_=True,
        type="FisheyeSparseRayHead",
        # Reuse only the C0 experiment boundaries needed for a controlled
        # comparison.  No InstanceBank, anchor encoder, graph layer,
        # deformable aggregation, Cartesian refiner, or ROSA sidecar config is
        # carried into the V2 head.
        initial_anchor={{_base_.model.head.instance_bank.anchor}},
        sampler={{_base_.model.head.sampler}},
        loss_cls={{_base_.model.head.loss_cls}},
        loss_reg={{_base_.model.head.loss_reg}},
        decoder={{_base_.model.head.decoder}},
        reg_weights={{_base_.model.head.reg_weights}},
        num_decoder=6,
        cls_threshold_to_reg={{_base_.model.head.cls_threshold_to_reg}},
        inference_output_decoder_index=5,
        inference_classification_decoder_index=5,
        inference_quality_decoder_index=5,
        ray_decoder_cfg=dict(
            embed_dims=256,
            num_queries=300,
            num_layers=6,
            num_classes=6,
            num_heads=8,
            ffn_dims=512,
            num_cameras=4,
            level_strides=(4.0, 8.0, 16.0, 32.0),
            dropout=0.1,
            min_range_m=0.5,
            max_range_m=68.0,
        ),
        ray_aux_loss=dict(
            angular_weight=0.25,
            log_range_l1_weight=0.25,
            range_nll_weight=0.10,
            beta=0.10,
        ),
    ),
)

train_cfg = dict(max_epochs=20, val_interval=4)
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

work_dir = "work_dirs/fisheye_ray_decoder_v2_raw_uniform_20e"
sealed_resume_contract = None

experiment_contract = dict(
    _delete_=True,
    arm="Fisheye-Ray-Decoder-V2-raw-uniform-20e",
    baseline_arm="C0-corrected-Sparse4D-raw-uniform-20e",
    dataset="Fisheye3DOD Main126",
    backbone="ImageNet-pretrained ResNet-18",
    cameras="four native 800x800 fisheye views",
    lens_backend="generic native-fisheye interface; OCam for this dataset",
    temporal=False,
    queries=300,
    decoders=6,
    seed=3407,
    per_device_batch_size=4,
    distributed_world_size=2,
    global_batch_size=8,
    max_epochs=20,
    train_sampling="raw-uniform DefaultSampler (no CBGSDataset)",
    initial_query_state="normalized C0 anchor direction only; no range/shape/yaw",
    observation=(
        "one projected ray per query/camera/FPN level; analytic fisheye "
        "Jacobian; RFL value; one ROSA camera-level softmax"
    ),
    refinement="S2 tangent angle + Euclidean log-range + ray-relative box state",
    removed=(
        "Sparse4D 13 box keypoints, deformable aggregation, Cartesian native "
        "refinement, post-D5 ROSA sidecar, candidate re-observation"
    ),
    latency_target_vs_c0=1.10,
    latency_hard_limit_vs_c0=1.20,
)
