"""C0-based ray-initialized 13-point native-fisheye box decoder V3."""

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
        "rosa.integrations.fisheye_box_sparse4d",
    ],
    allow_failed_imports=False,
)

model = dict(
    # V3 owns native-fisheye sampling and consumes ordinary FPN tensors.
    use_deformable_func=False,
    head=dict(
        _delete_=True,
        type="FisheyeSparseBoxHead",
        # Retain the real C0 learned query/11D-anchor bank.  Its temporal slots
        # are zero in the matched C0 config and are explicitly rejected by V3.
        instance_bank={{_base_.model.head.instance_bank}},
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
        box_decoder_cfg=dict(
            embed_dims=256,
            num_queries=300,
            num_layers=6,
            num_classes=6,
            num_heads=8,
            attention_dims=64,
            ffn_dims=512,
            num_cameras=4,
            level_strides=(4.0, 8.0, 16.0, 32.0),
            # Main arm: C0-equivalent 7 fixed + 6 learned support points.
            # Set to zero for the controlled seven-point ablation.
            num_learnable_points=6,
            fixed_keypoint_scale=0.45,
            # The 98-GiB training GPUs can keep the vectorized 13-point
            # carrier resident.  Chunked checkpoint recomputation remains an
            # opt-in low-memory path, but is substantially slower.
            point_chunk_size=2,
            checkpoint_point_chunks=False,
            use_fused_daf=True,
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

# Remove the historical ROSA-sidecar optimizer key.  V3 contains no sidecar.
optim_wrapper = dict(
    paramwise_cfg=dict(
        _delete_=True,
        custom_keys={"img_backbone": dict(lr_mult=0.5)},
    )
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

work_dir = "work_dirs/fisheye_box_decoder_v3_raw_uniform_20e"
sealed_resume_contract = None

experiment_contract = dict(
    _delete_=True,
    arm="Fisheye-Box-Decoder-V3-13point-raw-uniform-20e",
    baseline_arm="C0-corrected-Sparse4D-raw-uniform-20e",
    dataset="Fisheye3DOD Main126",
    backbone="ImageNet-pretrained ResNet-18",
    cameras="four native 800x800 fisheye views",
    temporal=False,
    free_box_queries=300,
    training_dn_groups=5,
    decoders=6,
    keypoints="7 fixed semantic + 6 query-learned per D0-D5 stage",
    initializer="native-fisheye centre-ray gated residual to C0 11D anchor",
    observation=(
        "per-point RFL; ROSA camera-level attention with analytic fisheye "
        "Jacobian; query-conditioned 13-point attention"
    ),
    refinement="S2 tangent direction + log-range + size/yaw/velocity",
    seed=3407,
    per_device_batch_size=4,
    distributed_world_size=2,
    global_batch_size=8,
    max_epochs=20,
    train_sampling="raw-uniform DefaultSampler (no CBGSDataset)",
    not_used=(
        "2D detector, dense depth head, BEV conversion, probabilistic tracking "
        "decoder, cross-frame temporal cache, post-D5 sidecar"
    ),
)
