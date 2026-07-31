"""Reliability-gate-off ablation of the fixed-budget Full model."""

_base_ = [
    "./fisheye3dod_sparse4d_r18_iterative_distribution_rosa_top100_main126_20e.py"
]

model = dict(
    head=dict(
        closed_loop_rosa_cfg=dict(
            # Preserve hypotheses, posterior and all losses, but always apply
            # the predicted radial update.
            force_radial_gate_one=True,
        )
    )
)

# Validation history is retained in the logs and the best checkpoint is kept
# separately; five rolling epoch checkpoints avoid wasting the nearly full
# experiment volume.
default_hooks = dict(checkpoint=dict(max_keep_ckpts=5))

work_dir = (
    "/home/airlab/user/kdh/fisheye3dod_sparse4d_r50_official_20260726/"
    "work_dirs/r18_iterative_distribution_rosa_top100_gate_off_"
    "main126_20e_2gpu_b4_seed3407_val4"
)

experiment_contract = dict(
    ablation="Reliability-Gate-Off",
    controlled_change="force radial reliability gate to one",
    unchanged=(
        "RFL, ROSA camera/point attention, learned log-range hypotheses, "
        "candidate scorer, two-pass refinement, native Sparse4D scores"
    ),
)
