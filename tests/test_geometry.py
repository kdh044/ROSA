import math

import torch

from rosa.geometry import ocam_project_points
from rosa.refiner import ROSAConfig


def _metas():
    intrinsic = torch.eye(3).reshape(1, 1, 3, 3)
    intrinsic[..., 0, 2] = 400.0
    intrinsic[..., 1, 2] = 400.0
    polynomial = torch.zeros(1, 1, 5)
    polynomial[..., 1] = -150.0
    return {
        "lidar2cam": torch.eye(4).reshape(1, 1, 4, 4),
        "ocam_intrinsic": intrinsic,
        "ocam_pol": polynomial,
        "image_wh": torch.full((1, 1, 2), 800.0),
        "ocam_lens_fov_deg": 220.0,
    }


def _ray_at_incidence(degrees):
    angle = math.radians(degrees)
    # OCam runtime convention: forward is -Z.
    return torch.tensor(
        (math.sin(angle), 0.0, -math.cos(angle)),
        dtype=torch.float32,
    ).reshape(1, 1, 1, 3)


def test_legacy_contract_reproduces_front_plane_clip():
    metas = _metas()
    ray_100 = _ray_at_incidence(100.0)
    legacy = ocam_project_points(
        ray_100,
        metas,
        metas["image_wh"],
        contract="legacy_front_only",
    )
    corrected = ocam_project_points(
        ray_100,
        metas,
        metas["image_wh"],
        contract="full_fov",
    )
    assert torch.equal(legacy, torch.full_like(legacy, 2.0))
    assert bool(((corrected >= 0) & (corrected < 1)).all())


def test_full_fov_still_rejects_ray_outside_declared_lens():
    metas = _metas()
    ray_115 = _ray_at_incidence(115.0)
    projected = ocam_project_points(
        ray_115,
        metas,
        metas["image_wh"],
        contract="full_fov",
    )
    assert torch.equal(projected, torch.full_like(projected, 2.0))


def test_published_checkpoint_contract_is_explicit_default():
    assert ROSAConfig().projection_contract == "legacy_front_only"
