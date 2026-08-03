"""Fast CP16 ROSA with fixed-base orthogonal center refinement."""

_base_ = "./rosa_r18_main126_v3_fast_cp16_official_residual.py"

model = dict(
    head=dict(
        rosa_cfg=dict(
            _delete_=True,
            profile="fast",
            center_target_mode="fixed_base",
        ),
    ),
)

work_dir = "work_dirs/rosa_r18_main126_v4_fast_cp16_fixed_base_official_residual"

sealed_resume_contract = dict(
    contract="rosa.c32-fast-cp16-fixed-base.30e-pair.v1",
    role="candidate",
)

experiment_contract = dict(
    arm="C32+ROSA-fast-CP16-fixed-base-v4",
    paired_baseline="C32-official-Sparse4D-GNN-residual-30e-control",
    changed_sidecar_operation=(
        "one 50-query full-FOV ROSA pass with signed CP16 RFL and one "
        "fixed-base orthogonal radial/tangential center frame"
    ),
    center_frame="fixed base radial/tangent orthonormal residuals",
)
