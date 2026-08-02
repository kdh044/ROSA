import math

import torch

from rosa import geometry


def _homogeneous_camera_xyz(points, lidar2cam):
    homogeneous = torch.cat(
        (points, torch.ones_like(points[..., :1])),
        dim=-1,
    )
    if points.ndim == 4:
        homogeneous = homogeneous[:, None]
    transformed = torch.matmul(
        lidar2cam[:, :, None, None],
        homogeneous[..., None],
    )
    return transformed.squeeze(-1)[..., :3]


def _forward_and_gradients(transform, points, lidar2cam, probe):
    points = points.detach().clone().requires_grad_(True)
    lidar2cam = lidar2cam.detach().clone().requires_grad_(True)
    output = transform(points, lidar2cam)
    loss = (output * probe).sum() + output.square().mean()
    gradients = torch.autograd.grad(loss, (points, lidar2cam))
    return output.detach(), tuple(gradient.detach() for gradient in gradients)


def _projection_metas(batch, cameras, dtype=torch.float64):
    angles = torch.linspace(-0.9, 0.9, cameras, dtype=dtype)
    cosine = angles.cos()[None].expand(batch, -1)
    sine = angles.sin()[None].expand(batch, -1)
    lidar2cam = (
        torch.eye(dtype=dtype, n=4).reshape(1, 1, 4, 4).repeat(batch, cameras, 1, 1)
    )
    lidar2cam[..., 0, 0] = cosine
    lidar2cam[..., 0, 1] = -sine
    lidar2cam[..., 1, 0] = sine
    lidar2cam[..., 1, 1] = cosine
    translations = torch.linspace(-0.25, 0.25, cameras, dtype=dtype)
    lidar2cam[..., 0, 3] = translations
    lidar2cam[..., 1, 3] = translations.flip(0)

    intrinsic = (
        torch.eye(dtype=dtype, n=3).reshape(1, 1, 3, 3).repeat(batch, cameras, 1, 1)
    )
    intrinsic[..., 0, 2] = 400.0
    intrinsic[..., 1, 2] = 400.0
    polynomial = torch.zeros(batch, cameras, 5, dtype=dtype)
    polynomial[..., 1] = -150.0
    return {
        "lidar2cam": lidar2cam,
        "ocam_intrinsic": intrinsic,
        "ocam_pol": polynomial,
        "ocam_lens_fov_deg": 220.0,
    }


def test_affine_camera_transform_preserves_outputs_and_gradients():
    generator = torch.Generator().manual_seed(7101080)
    lidar2cam = torch.randn(2, 3, 4, 4, generator=generator, dtype=torch.float64)
    lidar2cam[..., 3, :] = torch.tensor((0.0, 0.0, 0.0, 1.0), dtype=torch.float64)

    for shape in ((2, 5, 7, 3), (2, 3, 5, 7, 3)):
        points = torch.randn(*shape, generator=generator, dtype=torch.float64)
        output_shape = (2, 3, 5, 7, 3)
        probe = torch.randn(*output_shape, generator=generator, dtype=torch.float64)
        expected, expected_gradients = _forward_and_gradients(
            _homogeneous_camera_xyz,
            points,
            lidar2cam,
            probe,
        )
        actual, actual_gradients = _forward_and_gradients(
            geometry._camera_xyz_affine,
            points,
            lidar2cam,
            probe,
        )

        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        for actual_gradient, expected_gradient in zip(
            actual_gradients, expected_gradients
        ):
            torch.testing.assert_close(
                actual_gradient,
                expected_gradient,
                rtol=1e-12,
                atol=1e-12,
            )
        assert torch.equal(
            actual_gradients[1][..., 3, :],
            torch.zeros_like(actual_gradients[1][..., 3, :]),
        )


def test_affine_camera_transform_preserves_projection_and_validity():
    generator = torch.Generator().manual_seed(7101080)
    batch, cameras, anchors, keypoints = 2, 4, 300, 13
    shared_points = torch.randn(
        batch,
        anchors,
        keypoints,
        3,
        generator=generator,
        dtype=torch.float64,
    )
    shared_points[..., 2] -= 0.5
    conditioned_points = shared_points[:, None].expand(-1, cameras, -1, -1, -1).clone()
    conditioned_points += (
        torch.randn(
            conditioned_points.shape,
            generator=generator,
            dtype=conditioned_points.dtype,
        )
        * 0.01
    )
    metas = _projection_metas(batch, cameras)
    image_wh = torch.full((batch, cameras, 2), 800.0, dtype=torch.float64)

    for points in (shared_points, conditioned_points):
        for contract in geometry.PROJECTION_CONTRACTS:
            actual = geometry.ocam_project_points(
                points,
                metas,
                image_wh,
                contract=contract,
                camera_transform="affine_3x3",
            )
            expected = geometry.ocam_project_points(
                points,
                metas,
                image_wh,
                contract=contract,
            )

            actual_valid = actual[..., 0].ne(2.0)
            expected_valid = expected[..., 0].ne(2.0)
            assert torch.equal(actual_valid, expected_valid)
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_affine_camera_transform_preserves_full_fov_boundary_masks():
    incidences = (0.0, 1e-4, 89.9999, 90.0, 90.0001, 109.9999, 110.0, 110.0001)
    points = torch.tensor(
        [
            (
                math.sin(math.radians(incidence)),
                0.0,
                -math.cos(math.radians(incidence)),
            )
            for incidence in incidences
        ],
        dtype=torch.float64,
    ).reshape(1, 1, len(incidences), 3)
    metas = _projection_metas(1, 1)
    image_wh = torch.full((1, 1, 2), 800.0, dtype=torch.float64)

    actual = geometry.ocam_project_points(
        points,
        metas,
        image_wh,
        contract="full_fov",
        camera_transform="affine_3x3",
    )
    expected = geometry.ocam_project_points(
        points,
        metas,
        image_wh,
        contract="full_fov",
    )

    assert torch.equal(actual[..., 0].ne(2.0), expected[..., 0].ne(2.0))
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_affine_camera_transform_does_not_build_homogeneous_points(monkeypatch):
    points = torch.randn(1, 2, 3, 3)
    lidar2cam = torch.eye(4).reshape(1, 1, 4, 4)

    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("homogeneous point allocation was reintroduced")

    monkeypatch.setattr(torch, "cat", unexpected_allocation)
    monkeypatch.setattr(torch, "ones_like", unexpected_allocation)
    actual = geometry._camera_xyz_affine(points, lidar2cam)
    assert actual.shape == (1, 1, 2, 3, 3)


def test_affine_camera_transform_is_explicit_and_default_remains_strict():
    points = torch.randn(1, 2, 3, 3)
    lidar2cam = torch.eye(4).reshape(1, 1, 4, 4)
    assert torch.equal(
        geometry._camera_xyz(points, lidar2cam),
        _homogeneous_camera_xyz(points, lidar2cam),
    )
