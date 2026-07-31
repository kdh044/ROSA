"""Top-100 coarse-to-fine iterative distribution ROSA on Main126."""

_base_ = [
    "./fisheye3dod_sparse4d_r18_iterative_distribution_rosa_main126_20e.py"
]

model = dict(
    head=dict(
        closed_loop_rosa_cfg=dict(
            refine_topk=100,
            second_step_candidate_search=False,
            # Keep native Sparse4D class-times-centerness ranking unchanged
            # in this first box-refinement experiment.  The quality head
            # remains in the graph with a zero loss for DDP consistency.
            quality_loss_weight=0.0,
        )
    )
)

work_dir = (
    "/home/airlab/user/kdh/fisheye3dod_sparse4d_r50_official_20260726/"
    "work_dirs/r18_iterative_distribution_rosa_top100_main126_20e_2gpu_b4_"
    "seed3407_val4"
)

experiment_contract = dict(
    refine_selection=(
        "top 100 of 300 queries by untouched Sparse4D "
        "max-class-probability times centerness"
    ),
    unselected_queries="bitwise-identical native Sparse4D boxes and scores",
    step1=(
        "one current-box observation plus five learned log-Gaussian "
        "range hypotheses"
    ),
    step2=(
        "one updated-box observation with shared shape, tangent and "
        "direct log-range heads; no second candidate search"
    ),
    ranking="native Sparse4D class-times-centerness; quality ablation off",
)
