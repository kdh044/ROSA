from __future__ import annotations

import numpy as np
import torch
from mmdet3d.structures import LiDARInstance3DBoxes

from rosa.integrations.fisheye3dod.calibration import (
    convert_lidar2cam,
    convert_ocamcamera,
    project_rosa_compatible_numpy,
)
from rosa.integrations.fisheye3dod.constants import (
    EXPECTED_SCENES,
    EXPECTED_TRAIN_FRAMES,
    EXPECTED_TRAIN_FRAMES_PER_SCENE,
    EXPECTED_VAL_FRAMES,
    EXPECTED_VAL_FRAMES_PER_SCENE,
)
from rosa.integrations.fisheye3dod.dataset import (
    Fisheye3DODEvaluatorRangeFilter,
)


class _Calibration:
    _invpol = [100.0, 0.0]
    _xc = 400.0
    _yc = 400.0
    _affine = [1.0, 0.0, 0.0]
    _fov = 220.0


def test_calibration_conversion_preserves_released_coefficients():
    calibration = convert_ocamcamera(_Calibration())
    np.testing.assert_array_equal(
        calibration.polynomial,
        np.asarray(_Calibration._invpol, dtype=np.float32),
    )
    assert calibration.lens_fov_deg == 220.0
    assert convert_lidar2cam(np.eye(4)).shape == (4, 4)


def test_numpy_reference_separates_legacy_and_full_fov_contracts():
    calibration = convert_ocamcamera(_Calibration())
    angle = np.deg2rad(100.0)
    # ROSA OCam axes are [down, right, back].  This ray lies ten degrees
    # behind the optical plane but remains inside the declared 220° cone.
    ray = np.asarray(
        [[np.sin(angle), 0.0, -np.cos(angle)]],
        dtype=np.float64,
    )
    legacy = project_rosa_compatible_numpy(
        ray,
        np.eye(4),
        calibration,
        contract="legacy_front_only",
    )
    full_fov = project_rosa_compatible_numpy(
        ray,
        np.eye(4),
        calibration,
        contract="full_fov",
    )
    assert np.array_equal(legacy, np.full((1, 2), -1.0))
    assert np.all((full_fov >= 0) & (full_fov < 800))


def test_evaluator_range_filter_matches_strict_class_radial_contract():
    raw = torch.tensor(
        [
            [47.9, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0],
            [48.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0],
            [37.0, 0.0, 0.0, 1.0, 1.0, 1.7, 0.0],
            [35.9, 0.0, 0.0, 1.0, 1.0, 1.7, 0.0],
            [40.0, 40.0, 0.0, 2.0, 4.0, 1.5, 0.0],
            [1.0, 0.0, 100.0, 2.0, 4.0, 1.5, 0.0],
        ]
    )
    labels = np.asarray([0, 0, 4, 4, 0, 0], dtype=np.int64)
    annotation = {
        "gt_bboxes_3d": LiDARInstance3DBoxes(raw),
        "gt_labels_3d": labels,
        "instance_inds": np.arange(len(labels)) + 100,
    }
    result = Fisheye3DODEvaluatorRangeFilter()(
        {"ann_info": annotation}
    )
    filtered = result["ann_info"]
    assert filtered["gt_labels_3d"].tolist() == [0, 4, 0]
    assert filtered["instance_inds"].tolist() == [100, 103, 105]
    torch.testing.assert_close(
        filtered["gt_bboxes_3d"].tensor[:, :3],
        raw[[0, 3, 5], :3],
    )


def test_main126_raw_z_is_restored_as_gravity_center():
    raw = torch.tensor(
        [[4.0, -2.0, -1.3, 2.0, 4.0, 1.6, 0.0]],
    )
    boxes = LiDARInstance3DBoxes(
        raw,
        origin=(0.5, 0.5, 0.5),
    )

    # MMDetection3D stores LiDAR boxes at bottom center internally, while the
    # released Main126 pickle declares its input at the gravity center.
    torch.testing.assert_close(boxes.tensor[:, 2], raw[:, 2] - raw[:, 5] * 0.5)
    torch.testing.assert_close(boxes.gravity_center, raw[:, :3])


def test_main126_split_totals_match_the_released_scene_contract():
    assert EXPECTED_SCENES == 126
    assert EXPECTED_TRAIN_FRAMES == (
        EXPECTED_SCENES * EXPECTED_TRAIN_FRAMES_PER_SCENE
    )
    assert EXPECTED_VAL_FRAMES == (
        EXPECTED_SCENES * EXPECTED_VAL_FRAMES_PER_SCENE
    )
