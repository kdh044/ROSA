"""C1: balanced ROSA paired with C0's exact raw-CBGS train contract."""

_base_ = "./rosa_r18_main126_v2_c0_cbgs_native.py"

model = dict(
    head=dict(
        type="ROSARefinementHead",
        rosa_cfg=dict(
            _delete_=True,
            profile="balanced",
        ),
    ),
)

work_dir = "work_dirs/rosa_r18_main126_v2_c1_cbgs_balanced"

experiment_contract = dict(
    arm="C1-raw-CBGS-balanced-ROSA",
    paired_baseline="C0-raw-CBGS-native",
    detector="native Sparse4D D0-D5 plus balanced ROSA sidecar",
    rosa_sidecar=True,
    refinement_profile="balanced",
    learned_refiner_parameters=120_559,
)
