"""Controlled 2x2 interaction ablation: RFL and ROSA geometry both off.

Native fisheye sampling, visual query-key camera attention, hard validity,
adaptive hypotheses, and two-pass refinement remain enabled. The SH
ray-conditioned feature residual and observation-quality geometry used for
camera scoring are both disabled.
"""

_base_ = [
    "./fisheye3dod_sparse4d_r18_iterative_distribution_rosa_top100_main126_20e.py"
]

model = dict(
    head=dict(
        closed_loop_rosa_cfg=dict(
            enable_rfl=False,
            enable_rosa_geometry=False,
        )
    )
)

work_dir = (
    "/home/airlab/user/kdh/fisheye3dod_sparse4d_r50_official_20260726/"
    "work_dirs/r18_iterative_distribution_rosa_top100_no_rfl_no_geometry_"
    "main126_20e_2gpu_b4_seed3407_val4"
)

experiment_contract = dict(
    ablation=(
        "RFL residual and ROSA observation-quality geometry disabled "
        "together for a full 2x2 interaction estimate"
    ),
)
