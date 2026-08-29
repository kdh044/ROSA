"""Per-camera ray geometry for KITTI-360's mixed pinhole + MEI rig.

The PETR family (PETR, DAPETR, StreamPETR) all build their position embedding
the same way: unproject every feature-map cell into a camera-frame viewing
ray, scale it by depth bins, and push the result through ``cam2lidar``.  Their
OCam ports each compute that ray grid with one shared ``OcamCamera``, which
KITTI-360 breaks in two ways at once -- the four cameras do not share a lens
model, and the two fisheyes are fed a *crop* of their native image.

Both corrections live here rather than in each head, because getting the crop
wrong is silent: the rays still look plausible, the network still trains, and
the only symptom is that features are lifted along the wrong direction.  One
implementation means one place to check, and the parity script exercises it
directly.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .constants import CAMERA_MODEL_IDS, CAMERAS, NATIVE_IMAGE_SIZES
from .dataset import _native_to_network_affine

PINHOLE, MEI = 0, 1


def unproject_pinhole(pixels, intrinsic):
    """Unit rays for a rectified pinhole camera; every pixel is valid."""
    x = (pixels[0] - intrinsic[0, 2]) / intrinsic[0, 0]
    y = (pixels[1] - intrinsic[1, 2]) / intrinsic[1, 1]
    rays = np.stack([x, y, np.ones_like(x)])
    return rays / np.linalg.norm(rays, axis=0), np.ones(x.shape, dtype=bool)


def unproject_mei(pixels, intrinsic, distortion, newton_steps: int = 12):
    """Invert the MEI model KITTI-360 uses for image_02 / image_03."""
    k1, k2, _p1, _p2, xi = distortion
    x = (pixels[0] - intrinsic[0, 2]) / intrinsic[0, 0]
    y = (pixels[1] - intrinsic[1, 2]) / intrinsic[1, 1]

    # Undo the radial polynomial with a 1-D Newton solve on the radius.
    distorted = np.sqrt(x * x + y * y)
    radius = distorted.copy()
    for _ in range(newton_steps):
        r2 = radius * radius
        value = radius * (1.0 + k1 * r2 + k2 * r2 * r2) - distorted
        derivative = 1.0 + 3.0 * k1 * r2 + 5.0 * k2 * r2 * r2
        derivative = np.where(np.abs(derivative) < 1e-12, 1e-12, derivative)
        radius = np.maximum(radius - value / derivative, 0.0)
    scale = np.where(distorted > 1e-12, radius / np.maximum(distorted, 1e-12), 0.0)
    mx, my = x * scale, y * scale

    # Invert the mirror: with m = u_xy / (u_z + xi) and ||u|| = 1,
    # s = (xi + sqrt(1 + r^2 (1 - xi^2))) / (r^2 + 1) and u = (m s, s - xi).
    # The discriminant turns negative exactly where the forward projection
    # stops being one-to-one, which is the usable field of view.
    r2 = mx * mx + my * my
    discriminant = 1.0 + r2 * (1.0 - xi * xi)
    valid = discriminant >= 0.0
    factor = (xi + np.sqrt(np.where(valid, discriminant, 0.0))) / (r2 + 1.0)
    rays = np.stack([mx * factor, my * factor, factor - xi])
    rays = rays / np.maximum(np.linalg.norm(rays, axis=0), 1e-12)
    forward = np.zeros_like(rays)
    forward[2] = 1.0
    return np.where(valid[None], rays, forward), valid


def make_mixed_native_geometry(camera_calibration, height, width):
    """Return per-camera ray grids ``[C,H,W,3]`` and validity ``[C,H,W]``.

    The feature grid indexes the *network input*, which for the two fisheyes
    is a crop of the 1400x1400 native image rescaled to ``IMAGE_SIZE``.
    Lifting a feature cell therefore has to undo that crop-and-resize before
    the camera model is inverted, otherwise the ray attached to a cell points
    somewhere else entirely than the pixels that produced its features.
    ``_native_to_network_affine`` already defines that mapping for the loader
    (native -> network); this inverts it.  The pinhole cameras are fed
    uncropped, so their affine is the identity and the two branches share one
    code path.
    """
    rays, valid = [], []
    for index, (channel, model_id, (native_w, native_h)) in enumerate(
        zip(CAMERAS, CAMERA_MODEL_IDS, NATIVE_IMAGE_SIZES)
    ):
        calibration = camera_calibration[channel]
        intrinsic = np.asarray(calibration["intrinsic"], dtype=np.float64)
        _affine, crop = _native_to_network_affine(index)
        if crop is None:
            x0, y0, x1, y1 = 0.0, 0.0, float(native_w), float(native_h)
        else:
            x0, y0, x1, y1 = (float(v) for v in crop)
        grid_y, grid_x = np.meshgrid(
            y0 + (np.arange(height, dtype=np.float64) + 0.5) * (y1 - y0) / height,
            x0 + (np.arange(width, dtype=np.float64) + 0.5) * (x1 - x0) / width,
            indexing="ij",
        )
        pixels = np.stack([grid_x.ravel(), grid_y.ravel()])
        if model_id == PINHOLE:
            camera_rays, camera_valid = unproject_pinhole(pixels, intrinsic)
        else:
            camera_rays, camera_valid = unproject_mei(
                pixels,
                intrinsic,
                np.asarray(calibration["distortion"], dtype=np.float64),
            )
        rays.append(camera_rays.T.reshape(height, width, 3))
        valid.append(camera_valid.reshape(height, width))
    rays = torch.from_numpy(np.stack(rays)).float()
    return F.normalize(rays, dim=-1), torch.from_numpy(np.stack(valid))


class MixedCameraGeometryMixin:
    """Supplies ``_make_native_geometry`` for the mixed rig.

    Mix in *before* the OCam head being ported so this override wins.  The
    subclass must set ``self._camera_calibration`` (channel -> dict with
    ``intrinsic`` and ``distortion``) before the parent constructor runs,
    because that constructor immediately calls ``_make_native_geometry`` to
    fill its cached buffers.
    """

    def _make_native_geometry(self, height, width):
        return make_mixed_native_geometry(
            self._camera_calibration, height, width
        )

    @staticmethod
    def _validate_calibration(camera_calibration):
        missing = [c for c in CAMERAS if c not in camera_calibration]
        if missing:
            raise ValueError(f"camera_calibration missing {missing}")
        return {
            channel: dict(camera_calibration[channel]) for channel in CAMERAS
        }
