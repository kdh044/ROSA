"""Point-attention-off ablation of the fixed-budget Full model."""

_base_ = [
    "./fisheye3dod_sparse4d_r18_iterative_distribution_rosa_top100_main126_20e.py"
]

model = dict(
    head=dict(
        closed_loop_rosa_cfg=dict(
            # Camera attention remains learned; only the subsequent
            # query-conditioned competition across 13 points becomes a
            # validity-masked uniform mean.
            uniform_point_fusion=True,
        )
    )
)

default_hooks = dict(checkpoint=dict(max_keep_ckpts=5))

work_dir = (
    "/home/airlab/user/kdh/fisheye3dod_sparse4d_r50_official_20260726/"
    "work_dirs/r18_iterative_distribution_rosa_top100_point_attention_off_"
    "main126_20e_2gpu_b4_seed3407_val4"
)

experiment_contract = dict(
    ablation="Point-Attention-Off",
    controlled_change=(
        "replace query-conditioned 13-point attention by "
        "validity-masked uniform point fusion"
    ),
    unchanged=(
        "RFL, geometry-conditioned camera attention, adaptive radial "
        "hypotheses, reliability gate, two-pass refinement, native scores"
    ),
)
