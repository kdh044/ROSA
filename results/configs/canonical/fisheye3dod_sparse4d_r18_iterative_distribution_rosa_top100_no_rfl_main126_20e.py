"""Controlled ablation: final top-100 model without RFL.

All projection, sampling, ROSA geometry, adaptive range hypotheses, radial
updates, and two-pass refinement remain enabled.  Only the SH ray-conditioned
feature residual is set exactly to zero.
"""

_base_ = [
    "./fisheye3dod_sparse4d_r18_iterative_distribution_rosa_top100_main126_20e.py"
]

model = dict(
    head=dict(
        closed_loop_rosa_cfg=dict(
            enable_rfl=False,
        )
    )
)

work_dir = (
    "/home/airlab/user/kdh/fisheye3dod_sparse4d_r50_official_20260726/"
    "work_dirs/r18_iterative_distribution_rosa_top100_no_rfl_"
    "main126_20e_2gpu_b4_seed3407_val4"
)

experiment_contract = dict(
    ablation=(
        "RFL residual disabled; visual features enter ROSA unchanged while "
        "all fisheye observation geometry and refinement paths are retained"
    ),
)
