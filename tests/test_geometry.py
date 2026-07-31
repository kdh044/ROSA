import math

import torch

from rosa.geometry import ocam_project_points, ocam_projection_scales
from rosa.observation import project_and_describe
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


def test_corrected_profiles_use_the_complete_declared_fisheye_fov():
    assert (
        ROSAConfig.from_mapping({"profile": "balanced"}).projection_contract
        == "full_fov"
    )
    assert (
        ROSAConfig.from_mapping({"profile": "fast"}).projection_contract == "full_fov"
    )


def test_full_fov_rejects_the_undefined_camera_origin():
    metas = _metas()
    projected = ocam_project_points(
        torch.zeros(1, 1, 1, 3),
        metas,
        metas["image_wh"],
        contract="full_fov",
    )
    assert torch.equal(projected, torch.full_like(projected, 2.0))


def test_full_fov_rejects_unstable_or_nonfinite_camera_positions():
    metas = _metas()
    points = torch.tensor(
        [
            [
                [5e-4, 0.0, 0.0],
                [float("nan"), 0.0, -1.0],
                [float("inf"), 0.0, -1.0],
                [1e5, 0.0, -1.0],
            ]
        ],
        dtype=torch.float32,
    ).unsqueeze(1)
    projected = ocam_project_points(
        points,
        metas,
        metas["image_wh"],
        contract="full_fov",
    )
    assert torch.equal(projected, torch.full_like(projected, 2.0))


def test_projection_supports_camera_conditioned_keypoints():
    metas = _metas()
    shared = _ray_at_incidence(45.0)
    conditioned = shared[:, None]
    expected = ocam_project_points(
        shared,
        metas,
        metas["image_wh"],
        contract="full_fov",
    )
    actual = ocam_project_points(
        conditioned,
        metas,
        metas["image_wh"],
        contract="full_fov",
    )
    assert torch.equal(actual, expected)


def test_full_fov_projection_scales_include_anisotropic_ocam_affine():
    metas = {
        name: value.double() if isinstance(value, torch.Tensor) else value
        for name, value in _metas().items()
    }
    intrinsic = metas["ocam_intrinsic"].clone()
    intrinsic[..., 0, 0] = 2.0
    intrinsic[..., 0, 1] = 0.4
    intrinsic[..., 1, 0] = -0.3
    intrinsic[..., 1, 1] = 0.7
    metas["ocam_intrinsic"] = intrinsic

    incidence = 0.7
    azimuth = 0.4
    ray = torch.tensor(
        (
            math.sin(incidence) * math.cos(azimuth),
            math.sin(incidence) * math.sin(azimuth),
            -math.cos(incidence),
        ),
        dtype=torch.float64,
    )
    radial_direction = torch.tensor(
        (
            math.cos(incidence) * math.cos(azimuth),
            math.cos(incidence) * math.sin(azimuth),
            math.sin(incidence),
        ),
        dtype=torch.float64,
    )
    tangential_direction = torch.tensor(
        (-math.sin(azimuth), math.cos(azimuth), 0.0),
        dtype=torch.float64,
    )
    shaped_ray = ray.reshape(1, 1, 1, 1, 3)

    legacy = ocam_projection_scales(
        shaped_ray,
        metas,
        contract="legacy_front_only",
    )
    identity_metas = dict(metas)
    identity_metas["ocam_intrinsic"] = torch.eye(
        3,
        dtype=torch.float64,
    ).reshape(1, 1, 3, 3)
    legacy_identity = ocam_projection_scales(
        shaped_ray,
        identity_metas,
        contract="legacy_front_only",
    )
    assert torch.equal(legacy[0], legacy_identity[0])
    assert torch.equal(legacy[1], legacy_identity[1])

    corrected = ocam_projection_scales(
        shaped_ray,
        metas,
        contract="full_fov",
    )
    epsilon = 1e-6
    for scale, direction in zip(
        corrected,
        (radial_direction, tangential_direction),
    ):
        positive = (ray * math.cos(epsilon) + direction * math.sin(epsilon)).reshape(
            1, 1, 1, 3
        )
        negative = (ray * math.cos(epsilon) - direction * math.sin(epsilon)).reshape(
            1, 1, 1, 3
        )
        positive_pixel = (
            ocam_project_points(
                positive,
                metas,
                metas["image_wh"],
                contract="full_fov",
            )[0, 0, 0, 0]
            * metas["image_wh"][0, 0]
        )
        negative_pixel = (
            ocam_project_points(
                negative,
                metas,
                metas["image_wh"],
                contract="full_fov",
            )[0, 0, 0, 0]
            * metas["image_wh"][0, 0]
        )
        finite_difference = torch.linalg.vector_norm(
            positive_pixel - negative_pixel
        ) / (2.0 * epsilon)
        torch.testing.assert_close(
            scale.squeeze(),
            finite_difference,
            rtol=1e-7,
            atol=1e-7,
        )


def test_full_fov_boundary_is_zero_on_declared_angular_cone():
    metas = {
        name: value.double() if isinstance(value, torch.Tensor) else value
        for name, value in _metas().items()
    }
    metas["ocam_lens_fov_deg"] = 180.0
    ray_on_cone = torch.tensor(
        (1.0, 0.0, 0.0),
        dtype=torch.float64,
    ).reshape(1, 1, 1, 3)
    anchor = torch.zeros(1, 1, 8, dtype=torch.float64)
    anchor[..., 7] = 1.0

    uv, geometry, _, valid = project_and_describe(
        ray_on_cone,
        anchor,
        metas,
        "full_fov",
    )

    assert bool(valid.item())
    torch.testing.assert_close(uv, torch.full_like(uv, 0.5))
    assert torch.equal(geometry[..., 5], torch.zeros_like(geometry[..., 5]))


def test_legacy_boundary_remains_the_rectangular_image_margin():
    metas = _metas()
    metas["ocam_lens_fov_deg"] = 90.0
    ray_near_cone = _ray_at_incidence(44.0)
    anchor = torch.zeros(1, 1, 8)
    anchor[..., 7] = 1.0

    uv, geometry, _, valid = project_and_describe(
        ray_near_cone,
        anchor,
        metas,
        "legacy_front_only",
    )
    expected = (
        torch.minimum(
            torch.minimum(uv[..., 0], 1.0 - uv[..., 0]),
            torch.minimum(uv[..., 1], 1.0 - uv[..., 1]),
        )
        .mul(2.0)
        .clamp(0.0, 1.0)
    )

    assert bool(valid.item())
    assert torch.equal(geometry[..., 5], expected)
    assert bool((geometry[..., 5] > 0.5).all())


def test_bottom_z_descriptor_uses_the_geometric_box_center():
    metas = _metas()
    anchor = torch.zeros(1, 1, 11)
    anchor[..., 0] = 1.0
    anchor[..., 2] = -1.0
    anchor[..., 7] = 1.0
    keypoint = torch.tensor([[[[1.0, 0.0, -0.5]]]])
    _, bottom_geometry, _, _ = project_and_describe(
        keypoint,
        anchor,
        metas,
        "full_fov",
        anchor_z_is_bottom=True,
    )
    _, centered_geometry, _, _ = project_and_describe(
        keypoint,
        anchor,
        metas,
        "legacy_front_only",
        anchor_z_is_bottom=False,
    )
    torch.testing.assert_close(
        bottom_geometry[..., -3:],
        torch.zeros_like(bottom_geometry[..., -3:]),
    )
    torch.testing.assert_close(
        centered_geometry[..., -1],
        torch.full_like(centered_geometry[..., -1], 0.5),
    )
