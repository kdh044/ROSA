"""Controlled ablation: final top-100 model without the second pass.

Pass 1 retains native OCam observation, RFL, ROSA, learned log-range
hypotheses, posterior gating, shape, tangential, and radial updates. The
updated box B1 is emitted directly without re-observing it for a second joint
residual. This isolates the value of closed-loop re-observation.
"""

_base_ = [
    "./fisheye3dod_sparse4d_r18_iterative_distribution_rosa_top100_main126_20e.py"
]

model = dict(
    head=dict(
        closed_loop_rosa_cfg=dict(
            num_refinement_steps=1,
            second_step_candidate_search=False,
        )
    )
)

work_dir = (
    "/home/airlab/user/kdh/fisheye3dod_sparse4d_r50_official_20260726/"
    "work_dirs/r18_iterative_distribution_rosa_top100_no_second_pass_"
    "main126_20e_2gpu_b4_seed3407_val4"
)

experiment_contract = dict(
    ablation=(
        "Pass 1 adaptive hypothesis search retained; updated B1 is emitted "
        "without second-pass re-observation or bounded joint residual"
    ),
    num_reprojections=(
        "one current-box observation plus five pass-1 hypotheses"
    ),
)
