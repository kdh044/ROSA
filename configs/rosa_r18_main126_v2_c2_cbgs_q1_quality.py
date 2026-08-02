"""C2: Q1 score-only quality residual under C1's raw-CBGS contract."""

_base_ = "./rosa_r18_main126_v2_c1_cbgs_balanced.py"

model = dict(
    head=dict(
        rosa_cfg=dict(
            enable_quality_score=True,
            quality_loss_weight=0.5,
        ),
    ),
)

work_dir = "work_dirs/rosa_r18_main126_v2_c2_cbgs_q1_quality"

experiment_contract = dict(
    arm="C2-raw-CBGS-Q1-quality-residual",
    paired_baseline="C1-raw-CBGS-balanced-ROSA",
    ablation="Q1-score-only-quality-residual",
    learned_refiner_parameters=89_493,
    quality_loss_weight=0.5,
    quality_delta_logit_bound=1.0,
    quality_target=(
        "label-correct fraction of official strict AP distance thresholds "
        "(0.5m, 1m, 2m, 4m)"
    ),
    quality_gradient_scope="detached compact quality head only",
    inference_contract=(
        "frozen native label/classification/centerness; residual changes rank score only"
    ),
)
