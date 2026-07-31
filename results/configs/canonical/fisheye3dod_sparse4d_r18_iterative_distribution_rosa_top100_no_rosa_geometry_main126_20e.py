"""Controlled ablation: final top-100 model without ROSA quality geometry.

Native fisheye projection, hard camera validity, RFL, visual query-key
attention, adaptive range hypotheses, and both refinement passes are retained.
Only the fisheye observation-quality descriptors are zeroed in camera scoring.
"""

_base_ = [
    "./fisheye3dod_sparse4d_r18_iterative_distribution_rosa_top100_main126_20e.py"
]

model = dict(
    head=dict(
        closed_loop_rosa_cfg=dict(
            enable_rosa_geometry=False,
        )
    )
)

work_dir = (
    "/home/airlab/user/kdh/fisheye3dod_sparse4d_r50_official_20260726/"
    "work_dirs/r18_iterative_distribution_rosa_top100_no_rosa_geometry_"
    "main126_20e_2gpu_b4_seed3407_val4"
)

experiment_contract = dict(
    ablation=(
        "ROSA fisheye observation-quality descriptors disabled; hard "
        "projection validity and visual query-key camera attention retained"
    ),
)
