"""Controlled ablation: uniform valid-camera fusion inside ROSA.

All native fisheye sampling, RFL, point aggregation, adaptive hypotheses,
and two-pass refinement remain enabled. Per-point camera logits are forced
equal before the validity-masked softmax, isolating learned camera
competition.
"""

_base_ = [
    "./fisheye3dod_sparse4d_r18_iterative_distribution_rosa_top100_main126_20e.py"
]

model = dict(
    head=dict(
        closed_loop_rosa_cfg=dict(
            uniform_camera_fusion=True,
        )
    )
)

work_dir = (
    "/home/airlab/user/kdh/fisheye3dod_sparse4d_r50_official_20260726/"
    "work_dirs/r18_iterative_distribution_rosa_top100_uniform_camera_"
    "main126_20e_2gpu_b4_seed3407_val4"
)

experiment_contract = dict(
    ablation=(
        "ROSA camera Q-K and geometry scoring retained in the graph but "
        "effective valid-camera weights are uniform"
    ),
)
