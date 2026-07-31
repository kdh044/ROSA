"""Controlled ablation: train the final top-100 model without radial updates.

The learned range distribution, five pass-1 hypotheses, ROSA observations,
auxiliary range losses, two-pass re-observation, shape, and tangential updates
are retained.  Only the predicted radial displacement applied to the box is
set to zero.  This isolates the value of radial box movement while keeping the
rest of the training graph as close as possible to the full model.
"""

_base_ = [
    "./fisheye3dod_sparse4d_r18_iterative_distribution_rosa_top100_main126_20e.py"
]

model = dict(
    head=dict(
        closed_loop_rosa_cfg=dict(
            enable_radial=False,
        )
    )
)

# The radial confidence head has no downstream gradient when radial movement
# is disabled, so DDP must permit that intentionally unused parameter route.
find_unused_parameters = True

work_dir = (
    "/home/airlab/user/kdh/fisheye3dod_sparse4d_r50_official_20260726/"
    "work_dirs/r18_iterative_distribution_rosa_top100_no_radial_update_"
    "main126_20e_2gpu_b4_seed3407_val4"
)

experiment_contract = dict(
    ablation=(
        "radial box update disabled; learned distribution and candidate "
        "auxiliary supervision retained"
    ),
    radial_update="zero in both refinement passes",
)
