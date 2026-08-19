import math

import torch

from rosa.fisheye_projection import (
    FisheyeProjectionModel,
    LensProjection,
    NativeFisheyeProjector,
    OCamFisheyeProjection,
    _spherical_log_coordinates,
    sample_native_fisheye_fpn,
)
from rosa.geometry import ocam_project_points


def _metadata(batch=1, cameras=1, dtype=torch.float64):
    intrinsic = (
        torch.eye(dtype=dtype, n=3).reshape(1, 1, 3, 3).repeat(batch, cameras, 1, 1)
    )
    intrinsic[..., 0, 2] = 400.0
    intrinsic[..., 1, 2] = 400.0
    polynomial = torch.zeros(batch, cameras, 5, dtype=dtype)
    polynomial[..., 0] = -235.0
    polynomial[..., 1] = -150.0
    transform = (
        torch.eye(dtype=dtype, n=4).reshape(1, 1, 4, 4).repeat(batch, cameras, 1, 1)
    )
    return {
        "lidar2cam": transform,
        "ocam_intrinsic": intrinsic,
        "ocam_pol": polynomial,
        "image_wh": torch.full((batch, cameras, 2), 800.0, dtype=dtype),
        "ocam_lens_fov_deg": 220.0,
    }


def _ray(incidence_deg, azimuth_deg=0.0, dtype=torch.float64):
    incidence = math.radians(incidence_deg)
    azimuth = math.radians(azimuth_deg)
    return torch.tensor(
        (
            math.sin(incidence) * math.cos(azimuth),
            math.sin(incidence) * math.sin(azimuth),
            -math.cos(incidence),
        ),
        dtype=dtype,
    )


def test_ocam_backend_matches_released_full_fov_projection_on_valid_rays():
    metadata = _metadata()
    rays = torch.stack(
        tuple(_ray(angle, 25.0) for angle in (1.0, 30.0, 89.0, 105.0, 115.0)),
        dim=0,
    )[None]
    backend = OCamFisheyeProjection()

    actual = backend.project_rays(rays[:, :, None], metadata, metadata["image_wh"])
    expected = ocam_project_points(
        rays[:, :, None, :],
        metadata,
        metadata["image_wh"],
        contract="full_fov",
    )[:, 0, :, 0]
    expected_valid = expected[..., 0].ne(2.0)

    assert torch.equal(actual.valid[..., 0], expected_valid)
    torch.testing.assert_close(
        actual.uv[..., 0, :][expected_valid],
        expected[expected_valid],
        atol=1e-12,
        rtol=1e-12,
    )
    assert not actual.valid[0, -1, 0]


def test_ocam_analytic_jacobian_matches_angular_finite_difference():
    metadata = _metadata()
    incidence = math.radians(45.0)
    azimuth = math.radians(30.0)
    base = _ray(45.0, 30.0)[None, None, None]
    radial = torch.tensor(
        (
            math.cos(incidence) * math.cos(azimuth),
            math.cos(incidence) * math.sin(azimuth),
            math.sin(incidence),
        ),
        dtype=torch.float64,
    )
    tangential = torch.tensor(
        (-math.sin(azimuth), math.cos(azimuth), 0.0), dtype=torch.float64
    )
    epsilon = 1e-5
    backend = OCamFisheyeProjection()
    output = backend.project_rays(base, metadata, metadata["image_wh"])

    finite_columns = []
    for tangent in (radial, tangential):
        plus = base * math.cos(epsilon) + tangent * math.sin(epsilon)
        minus = base * math.cos(epsilon) - tangent * math.sin(epsilon)
        plus_uv = backend.project_rays(plus, metadata, metadata["image_wh"]).uv
        minus_uv = backend.project_rays(minus, metadata, metadata["image_wh"]).uv
        derivative = (plus_uv - minus_uv) / (2.0 * epsilon)
        derivative = derivative * metadata["image_wh"][:, None]
        finite_columns.append(derivative)
    finite = torch.stack(finite_columns, dim=-1)

    torch.testing.assert_close(
        output.jacobian,
        finite,
        atol=2e-5,
        rtol=2e-5,
    )


def test_spherical_log_relation_is_finite_at_axis_singularities():
    for target in ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (1.0, 1e-7, 0.0)):
        origin = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
        relation = _spherical_log_coordinates(origin, torch.tensor([target]))
        relation.sum().backward()
        torch.testing.assert_close(relation, torch.zeros_like(relation))
        assert torch.isfinite(origin.grad).all()

    origin = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    quarter_turn = _spherical_log_coordinates(origin, torch.tensor([[0.0, 1.0, 0.0]]))
    torch.testing.assert_close(quarter_turn, torch.tensor([[math.pi / 2.0, 0.0]]))
    quarter_turn.sum().backward()
    assert torch.isfinite(origin.grad).all()


class _TestLens(FisheyeProjectionModel):
    def project_rays(self, camera_rays, metadata, image_wh):
        del metadata, image_wh
        prefix = camera_rays.shape[:-1]
        return LensProjection(
            uv=camera_rays.new_full(prefix + (2,), 0.5),
            jacobian=torch.eye(
                2, dtype=camera_rays.dtype, device=camera_rays.device
            ).expand(prefix + (2, 2)),
            lens_margin=camera_rays.new_ones(prefix + (1,)),
            valid=torch.ones(prefix, dtype=torch.bool, device=camera_rays.device),
        )


class _RaySensitiveLens(FisheyeProjectionModel):
    def project_rays(self, camera_rays, metadata, image_wh):
        del metadata, image_wh
        prefix = camera_rays.shape[:-1]
        return LensProjection(
            uv=camera_rays[..., :2] * 0.25 + 0.5,
            jacobian=torch.eye(
                2, dtype=camera_rays.dtype, device=camera_rays.device
            ).expand(prefix + (2, 2)),
            lens_margin=camera_rays.new_ones(prefix + (1,)),
            valid=torch.ones(prefix, dtype=torch.bool, device=camera_rays.device),
        )


def test_rig_projector_accepts_non_ocam_backend_and_builds_overlap_geometry():
    metadata = _metadata(cameras=4, dtype=torch.float32)
    direction = torch.nn.functional.normalize(torch.randn(2, 3, 3), dim=-1)
    # Expand the one-batch calibration without tying the projector to OCam.
    metadata = {
        name: (
            value.expand(2, *value.shape[1:]).clone()
            if torch.is_tensor(value)
            else value
        )
        for name, value in metadata.items()
    }
    projector = NativeFisheyeProjector(_TestLens())

    output = projector(direction, metadata)

    assert output.uv.shape == (2, 3, 4, 2)
    assert output.geometry.rays.shape == (2, 3, 4, 3)
    assert output.geometry.tangent.shape == (2, 3, 4, 2)
    assert output.geometry.jacobian.shape == (2, 3, 4, 2, 2)
    assert output.geometry.valid.all()
    torch.testing.assert_close(
        output.geometry.overlap, torch.ones_like(output.geometry.overlap)
    )


def test_d0_is_translation_free_and_metric_projection_uses_translation():
    metadata = _metadata(dtype=torch.float64)
    translated = dict(metadata)
    translated["lidar2cam"] = metadata["lidar2cam"].clone()
    translated["lidar2cam"][..., :3, 3] = torch.tensor((0.0, 2.0, 0.0))
    direction = torch.nn.functional.normalize(
        torch.tensor([[[4.0, 1.0, -2.0]]], dtype=torch.float64), dim=-1
    )
    point = direction * 8.0
    projector = NativeFisheyeProjector(_RaySensitiveLens())

    d0_origin = projector(direction, metadata)
    d0_translated = projector(direction, translated)
    torch.testing.assert_close(d0_translated.uv, d0_origin.uv)
    torch.testing.assert_close(d0_translated.geometry.rays, d0_origin.geometry.rays)

    metric_origin = projector(direction, metadata, points=point)
    metric_translated = projector(direction, translated, points=point)
    assert not torch.allclose(metric_translated.uv, metric_origin.uv)
    expected_camera_ray = torch.nn.functional.normalize(
        point[:, :, None, :] + translated["lidar2cam"][:, None, :, :3, 3],
        dim=-1,
    )
    torch.testing.assert_close(
        metric_translated.geometry.rays,
        expected_camera_ray,
    )


def test_native_fisheye_fpn_sampling_has_one_coordinate_per_level_and_gradient():
    batch, queries, cameras, channels = 1, 3, 4, 8
    fpn = [
        torch.randn(batch, cameras, channels, size, size, requires_grad=True)
        for size in (16, 8, 4, 2)
    ]
    uv = torch.rand(batch, queries, cameras, 2) * 0.8 + 0.1

    sampled = sample_native_fisheye_fpn(fpn, uv)

    assert sampled.shape == (batch, queries, cameras, 4, channels)
    sampled.square().mean().backward()
    for feature in fpn:
        assert feature.grad is not None
        assert torch.count_nonzero(feature.grad) > 0
