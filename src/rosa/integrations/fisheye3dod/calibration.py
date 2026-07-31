"""Convert released Fisheye3DOD OCam calibration to ROSA coordinates."""

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


# Released [right, down, forward] -> ROSA [down, right, back].
OFFICIAL_TO_ROSA_OCAM = np.asarray(
    (
        (0.0, 1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, -1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ),
    dtype=np.float32,
)
# Released name retained for source compatibility with experiment utilities.
OFFICIAL_TO_SPARSE_OCAM = OFFICIAL_TO_ROSA_OCAM


@dataclass(frozen=True)
class ROSAOcamCalibration:
    intrinsic: np.ndarray
    polynomial: np.ndarray
    lens_fov_deg: float


SparseOcamCalibration = ROSAOcamCalibration


def convert_ocamcamera(ocam: OcamLike) -> ROSAOcamCalibration:
    """Convert a released ``OcamCamera`` without changing its polynomial."""

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
    if polynomial.ndim != 1 or polynomial.size < 2:
        raise ValueError("released OCam inverse polynomial is missing")
    return ROSAOcamCalibration(
        intrinsic=intrinsic,
        polynomial=polynomial,
        lens_fov_deg=float(ocam._fov),
    )


def convert_lidar2cam(lidar2cam: np.ndarray) -> np.ndarray:
    matrix = np.asarray(lidar2cam, dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"expected 4x4 lidar2cam, got {matrix.shape}")
    return OFFICIAL_TO_ROSA_OCAM @ matrix


def project_rosa_compatible_numpy(
    points_lidar: np.ndarray,
    lidar2cam_rosa: np.ndarray,
    calibration: ROSAOcamCalibration,
    image_wh: tuple[int, int] = (800, 800),
    *,
    contract: str = "legacy_front_only",
) -> np.ndarray:
    """NumPy projection reference for either explicit OCam contract."""

    if contract not in {"legacy_front_only", "full_fov"}:
        raise ValueError(f"unknown projection contract: {contract}")
    points = np.asarray(points_lidar, dtype=np.float64).reshape(-1, 3)
    homogeneous = np.concatenate(
        (points, np.ones((len(points), 1), dtype=np.float64)), axis=1
    )
    camera = (np.asarray(lidar2cam_rosa, dtype=np.float64) @ homogeneous.T).T
    down, right, back = camera[:, 0], camera[:, 1], camera[:, 2]
    planar = np.sqrt(down * down + right * right)
    theta = np.arctan2(back, planar)
    rho = np.polynomial.polynomial.polyval(
        theta, calibration.polynomial.astype(np.float64)
    )
    scale = np.divide(rho, planar, out=np.zeros_like(rho), where=planar > 1e-12)
    image_down = down * scale
    image_right = right * scale
    intrinsic = calibration.intrinsic.astype(np.float64)
    u = intrinsic[1, 0] * image_down + intrinsic[1, 1] * image_right + intrinsic[0, 2]
    v = intrinsic[0, 0] * image_down + intrinsic[0, 1] * image_right + intrinsic[1, 2]
    width, height = image_wh
    incidence = np.arctan2(planar, -back)
    valid = (
        (incidence <= np.deg2rad(calibration.lens_fov_deg / 2))
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )
    if contract == "legacy_front_only":
        valid &= back < 0
    uv = np.stack((u, v), axis=-1)
    uv[~valid] = -1
    return uv


project_sparse_compatible_numpy = project_rosa_compatible_numpy


__all__ = [
    "OFFICIAL_TO_ROSA_OCAM",
    "OFFICIAL_TO_SPARSE_OCAM",
    "ROSAOcamCalibration",
    "SparseOcamCalibration",
    "convert_lidar2cam",
    "convert_ocamcamera",
    "project_rosa_compatible_numpy",
    "project_sparse_compatible_numpy",
]
