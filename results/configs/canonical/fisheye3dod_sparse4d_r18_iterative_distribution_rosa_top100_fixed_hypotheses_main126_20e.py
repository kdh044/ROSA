"""Fixed-metric radial-hypothesis ablation of the Full model."""

_base_ = [
    "./fisheye3dod_sparse4d_r18_iterative_distribution_rosa_top100_main126_20e.py"
]

model = dict(
    head=dict(
        closed_loop_rosa_cfg=dict(
            fixed_metric_radial_hypotheses=True,
            radial_step_m=0.5,
        )
    )
)

default_hooks = dict(checkpoint=dict(max_keep_ckpts=5))

work_dir = (
    "/home/airlab/user/kdh/fisheye3dod_sparse4d_r50_official_20260726/"
    "work_dirs/r18_iterative_distribution_rosa_top100_fixed_hypotheses_"
    "main126_20e_2gpu_b4_seed3407_val4"
)

experiment_contract = dict(
    ablation="Fixed-Radial-Hypotheses",
    controlled_change=(
        "replace adaptive pass-1 log-range hypotheses by "
        "[-1.0, -0.5, 0.0, 0.5, 1.0] metre offsets"
    ),
    unchanged=(
        "five candidate reprojections, candidate scorer/posterior/gate, "
        "second re-observation, shape/tangent refinement, native scores"
    ),
)
