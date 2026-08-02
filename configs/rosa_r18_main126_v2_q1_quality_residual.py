"""Q1 ablation: score-only localization-quality residual for mAP ranking."""

_base_ = "./rosa_r18_main126_v2.py"

model = dict(
    head=dict(
        rosa_cfg=dict(
            profile="balanced",
            enable_quality_score=True,
            # Reuse the released ROSA coefficient as a fixed control, not a sweep.
            quality_loss_weight=0.5,
        ),
    ),
)

work_dir = "work_dirs/rosa_r18_main126_v2_q1_quality_residual"

experiment_contract = dict(
    refinement_profile="balanced",
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
