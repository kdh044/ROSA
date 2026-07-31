from __future__ import annotations

import numpy as np

from rosa.integrations.fisheye3dod.calibration import (
    convert_lidar2cam,
    convert_ocamcamera,
    project_rosa_compatible_numpy,
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
