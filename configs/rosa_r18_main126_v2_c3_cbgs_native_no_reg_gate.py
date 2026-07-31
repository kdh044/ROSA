"""C3: raw-CBGS native detector without the matched regression score gate."""

_base_ = "./rosa_r18_main126_v2_c0_cbgs_native.py"

model = dict(
    head=dict(
        cls_threshold_to_reg=-1,
    ),
)

work_dir = "work_dirs/rosa_r18_main126_v2_c3_cbgs_native_no_reg_gate"

experiment_contract = dict(
    ablation="native-matched-regression-without-class-score-gate",
    cls_threshold_to_reg=-1,
    changed_training_field="model.head.cls_threshold_to_reg",
)
