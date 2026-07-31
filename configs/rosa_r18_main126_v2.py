"""Corrected ROSA v2 with one user-facing behavior profile."""

import os
import pathlib

_base_ = "./rosa_r18_main126.py"

profile = os.environ.get("ROSA_PROFILE", "balanced")
if profile not in {"balanced", "fast"}:
    raise ValueError("ROSA_PROFILE must be 'balanced' or 'fast'")

model = dict(
    head=dict(
        rosa_cfg=dict(
            _delete_=True,
            profile=profile,
        ),
        instance_bank=dict(
            anchor_handler=dict(
                type="ROSANormalizedBottomKeyPointsGenerator",
                anchor_z_is_bottom=True,
            ),
        ),
        deformable_model=dict(
            type="ROSAFullFOVDeformableFeatureAggregation",
            kps_generator=dict(
                type="ROSANormalizedBottomKeyPointsGenerator",
                anchor_z_is_bottom=True,
            ),
        ),
        loss_reg=dict(type="ROSAEvaluatorAlignedSparseBox3DLoss"),
    )
)

train_dataloader = dict(
    dataset=dict(
        pipeline=[
            dict(
                type="LoadOfficialFisheye3DODForSparse4D",
                cam_names=(
                    "fisheye_camera_front",
                    "fisheye_camera_left",
                    "fisheye_camera_right",
                    "fisheye_camera_rear",
                ),
                to_float32=True,
            ),
            dict(
                type="LoadAnnotations3D",
                with_bbox=False,
                with_bbox_3d=True,
                with_label=False,
                with_label_3d=True,
            ),
            dict(type="Fisheye3DODEvaluatorRangeFilter"),
            dict(type="PackSparse4DInputs", ocam_lens_fov_deg=220.0),
        ],
    ),
)

work_dir = str(
    pathlib.Path(
        os.environ.get(
            "ROSA_WORK_DIR",
            pathlib.Path("{{ fileDirname }}").resolve().parent
            / "work_dirs"
            / f"rosa_r18_main126_v2_{profile}",
        )
    ).resolve()
)

experiment_contract = dict(
    refinement_profile=profile,
    learned_refiner_parameters=120_559,
    candidate_range_m=68.0,
    candidate_count=3,
    yaw_gate="support",
    projection_contract="full_fov",
    native_projection_contract="full_fov",
    normalized_sampling_yaw=True,
    anchor_z_is_bottom=True,
    centerness_target="detached_evaluator_xy",
    train_gt_range="official_class_radial_strict",
)
