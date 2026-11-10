"""Convert Fisheye3DOD OCam calibration to ROSA coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class OcamLike(Protocol):
    _invpol: list[float]
    _xc: float
    _yc: float
    _affine: list[float]
    _fov: float


# Fisheye3DOD [right, down, forward] -> ROSA [down, right, back].
OFFICIAL_TO_ROSA_OCAM = np.asarray(
    (
        (0.0, 1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, -1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ),
    dtype=np.float32,
)


@dataclass(frozen=True)
class OCamCalibration:
    intrinsic: np.ndarray
    polynomial: np.ndarray
    lens_fov_deg: float


def convert_ocamcamera(ocam: OcamLike) -> OCamCalibration:
    """Convert an ``OcamCamera``; the polynomial is copied unchanged."""

    affine = np.asarray(ocam._affine, dtype=np.float32)
    if affine.shape != (3,):
        raise ValueError(f"expected three affine terms, got {affine.shape}")
    c, d, e = (float(value) for value in affine)
    # ROSA projector reads:
    #   u = A[1,0] * down + A[1,1] * right + A[0,2]
    #   v = A[0,0] * down + A[0,1] * right + A[1,2]
    intrinsic = np.asarray(
        (
            (c, d, float(ocam._yc)),
            (e, 1.0, float(ocam._xc)),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )
    polynomial = np.asarray(ocam._invpol, dtype=np.float32)
    lens_fov_deg = float(ocam._fov)
    if not np.isfinite(lens_fov_deg) or not 0 < lens_fov_deg <= 360:
        raise ValueError("OCam FOV must be finite and in (0,360]")
    return OCamCalibration(
        intrinsic=intrinsic,
        polynomial=polynomial,
        lens_fov_deg=lens_fov_deg,
    )


def convert_lidar2cam(lidar2cam: np.ndarray) -> np.ndarray:
    matrix = np.asarray(lidar2cam, dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"expected 4x4 lidar2cam, got {matrix.shape}")
    return OFFICIAL_TO_ROSA_OCAM @ matrix


__all__ = [
    "OFFICIAL_TO_ROSA_OCAM",
    "OCamCalibration",
    "convert_lidar2cam",
    "convert_ocamcamera",
]
