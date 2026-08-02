"""Exactness contracts for allocation-free scalar OCam fallbacks."""

import inspect

import pytest
import torch

from rosa import geometry


def _legacy_horner(coefficients: torch.Tensor, value: torch.Tensor):
    terms = coefficients.unbind(dim=-1)
    output = terms[-1].expand_as(value).clone()
    for coefficient in reversed(terms[:-1]):
        output = output * value + coefficient.expand_as(value)
    return output


def _legacy_sanitize(camera_xyz: torch.Tensor):
    finite_and_bounded = torch.isfinite(camera_xyz).all(dim=-1) & (
        camera_xyz.detach().abs().amax(dim=-1) < 1e4
    )
    bounded = torch.where(
        finite_and_bounded[..., None],
        camera_xyz,
        torch.zeros_like(camera_xyz),
    )
    radius_squared = (bounded * bounded).sum(dim=-1)
    geometry_valid = finite_and_bounded & radius_squared.gt(1e-6)
    fallback = torch.zeros_like(camera_xyz)
    fallback[..., 2] = -1.0
    safe = torch.where(geometry_valid[..., None], bounded, fallback)
    return safe, geometry_valid


def _legacy_project(points, metas, image_wh, *, contract):
    lidar2cam = metas["lidar2cam"]
    intrinsic = metas["ocam_intrinsic"]
    polynomial = metas["ocam_pol"]
    camera_xyz = geometry._camera_xyz(points, lidar2cam)
    safe_xyz, geometry_valid = _legacy_sanitize(camera_xyz)
    x, y, z = safe_xyz.unbind(dim=-1)
    planar_norm = torch.sqrt(x.square() + y.square() + 1e-12)
    theta_ocam = torch.atan2(z, planar_norm)
    rho = _legacy_horner(polynomial[:, :, None, None], theta_ocam)
    scale = torch.where(
        planar_norm > 1e-8,
        rho / planar_norm.clamp_min(1e-8),
        torch.zeros_like(rho),
    )
    image_x, image_y = x * scale, y * scale

    a11 = intrinsic[..., 1, 0][:, :, None, None]
    a12 = intrinsic[..., 1, 1][:, :, None, None]
    a13 = intrinsic[..., 0, 2][:, :, None, None]
    a21 = intrinsic[..., 0, 0][:, :, None, None]
    a22 = intrinsic[..., 0, 1][:, :, None, None]
    a23 = intrinsic[..., 1, 2][:, :, None, None]
    pixel_u = a11 * image_x + a12 * image_y + a13
    pixel_v = a21 * image_x + a22 * image_y + a23

    half_fov = geometry._half_fov_radians(
        metas.get("ocam_lens_fov_deg", 220.0),
        batch=points.shape[0],
        cameras=lidar2cam.shape[1],
        reference=points,
    )
    incidence = torch.atan2(planar_norm, -z)
    valid = geometry_valid & (incidence <= half_fov)
    if contract == "legacy_front_only":
        valid = valid & (z < 0)

    width = image_wh[..., 0][:, :, None, None]
    height = image_wh[..., 1][:, :, None, None]
    valid = (
        valid
        & torch.isfinite(pixel_u)
        & torch.isfinite(pixel_v)
        & pixel_u.ge(0)
        & pixel_u.lt(width)
        & pixel_v.ge(0)
        & pixel_v.lt(height)
    )
    normalized_u = pixel_u / width.clamp_min(1)
    normalized_v = pixel_v / height.clamp_min(1)
    sentinel = torch.full_like(normalized_u, 2.0)
    return torch.stack(
        (
            torch.where(valid, normalized_u, sentinel),
            torch.where(valid, normalized_v, sentinel),
        ),
        dim=-1,
    )


def _fixture(*, dtype, seed, camera_conditioned):
    generator = torch.Generator().manual_seed(seed)
    batch, cameras, queries, points = 2, 3, 11, 5
    if camera_conditioned:
        shape = (batch, cameras, queries, points, 3)
    else:
        shape = (batch, queries, points, 3)
    xyz = torch.randn(shape, dtype=dtype, generator=generator)
    xyz[..., 2].sub_(2.0)
    lidar2cam = torch.eye(4, dtype=dtype).reshape(1, 1, 4, 4).repeat(
        batch, cameras, 1, 1
    )
    lidar2cam[..., :3, 3] = (
        torch.randn(batch, cameras, 3, dtype=dtype, generator=generator) * 0.1
    )
    intrinsic = torch.zeros(batch, cameras, 3, 3, dtype=dtype)
    intrinsic[..., 0, 0] = 1.0
    intrinsic[..., 1, 0] = 1.0
    intrinsic[..., 0, 2] = 400.0
    intrinsic[..., 1, 2] = 400.0
    polynomial = torch.randn(
        batch, cameras, 5, dtype=dtype, generator=generator
    )
    polynomial[..., 0] = 120.0
    image_wh = torch.full((batch, cameras, 2), 800.0, dtype=dtype)
    metas = {
        "lidar2cam": lidar2cam,
        "ocam_intrinsic": intrinsic,
        "ocam_pol": polynomial,
        "ocam_lens_fov_deg": 220.0,
    }
    return xyz, metas, image_wh, generator


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("seed", (3407, 7101080, 918031))
@pytest.mark.parametrize("camera_conditioned", (False, True))
@pytest.mark.parametrize("contract", geometry.PROJECTION_CONTRACTS)
def test_scalar_fallback_projection_is_cpu_raw_bit_and_gradient_equal(
    dtype, seed, camera_conditioned, contract
):
    points, metas, image_wh, generator = _fixture(
        dtype=dtype,
        seed=seed,
        camera_conditioned=camera_conditioned,
    )
    parent_leaves = (
        points.requires_grad_(),
        metas["lidar2cam"].requires_grad_(),
        metas["ocam_intrinsic"].requires_grad_(),
        metas["ocam_pol"].requires_grad_(),
    )
    parent_metas = {
        **metas,
        "lidar2cam": parent_leaves[1],
        "ocam_intrinsic": parent_leaves[2],
        "ocam_pol": parent_leaves[3],
    }
    candidate_leaves = tuple(
        value.detach().clone().requires_grad_() for value in parent_leaves
    )
    candidate_metas = {
        **metas,
        "lidar2cam": candidate_leaves[1],
        "ocam_intrinsic": candidate_leaves[2],
        "ocam_pol": candidate_leaves[3],
    }

    parent = _legacy_project(
        parent_leaves[0], parent_metas, image_wh, contract=contract
    )
    candidate = geometry.ocam_project_points(
        candidate_leaves[0], candidate_metas, image_wh, contract=contract
    )
    assert torch.equal(candidate, parent)

    upstream = torch.randn(
        parent.shape,
        dtype=dtype,
        generator=generator,
    )
    parent_gradients = torch.autograd.grad(
        (parent * upstream).sum(), parent_leaves
    )
    candidate_gradients = torch.autograd.grad(
        (candidate * upstream).sum(), candidate_leaves
    )
    for candidate_gradient, parent_gradient in zip(
        candidate_gradients, parent_gradients
    ):
        assert torch.equal(candidate_gradient, parent_gradient)


def test_scalar_fallback_projection_preserves_invalid_ray_sentinels():
    points, metas, image_wh, _ = _fixture(
        dtype=torch.float32,
        seed=3407,
        camera_conditioned=False,
    )
    points[0, 0, 0] = torch.tensor((float("nan"), 0.0, -1.0))
    points[0, 0, 1] = 0.0
    points[0, 0, 2] = torch.tensor((float("inf"), 0.0, -1.0))
    metas["lidar2cam"][..., :3, 3] = 0.0
    parent = _legacy_project(points, metas, image_wh, contract="full_fov")
    candidate = geometry.ocam_project_points(
        points, metas, image_wh, contract="full_fov"
    )
    assert torch.equal(candidate, parent)
    invalid = candidate[0, :, 0, :3]
    assert torch.equal(invalid, torch.full_like(invalid, 2.0))


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_single_bound_reduction_matches_explicit_finiteness_oracle(dtype):
    rows = torch.tensor(
        [
            [0.0, 0.0, -1.0],
            [float("nan"), 0.0, -1.0],
            [0.0, float("nan"), -1.0],
            [0.0, 0.0, float("nan")],
            [float("inf"), 0.0, -1.0],
            [0.0, float("-inf"), -1.0],
            [0.0, 0.0, float("inf")],
            [9_999.0, 0.0, -1.0],
            [10_000.0, 0.0, -1.0],
            [-10_000.0, 0.0, -1.0],
            [5.0e-4, 0.0, 0.0],
            [1.0e-3, 0.0, 0.0],
        ],
        dtype=dtype,
    ).reshape(1, 1, -1, 1, 3)
    expected_safe, expected_valid = _legacy_sanitize(rows)

    actual_safe, actual_valid = geometry._sanitize_camera_xyz(rows)
    with torch.no_grad():
        reused_safe, reused_valid = (
            geometry._sanitize_camera_xyz_inference_storage_reuse(
                rows.clone()
            )
        )

    assert torch.equal(actual_safe, expected_safe)
    assert torch.equal(actual_valid, expected_valid)
    assert torch.equal(reused_safe, expected_safe)
    assert torch.equal(reused_valid, expected_valid)


@pytest.mark.parametrize("source", ("polynomial_nan", "intrinsic_inf"))
def test_range_predicates_reject_nonfinite_projected_pixels(source):
    points, metas, image_wh, _ = _fixture(
        dtype=torch.float32,
        seed=3407,
        camera_conditioned=False,
    )
    if source == "polynomial_nan":
        metas["ocam_pol"][..., 0] = float("nan")
    else:
        metas["ocam_intrinsic"][..., 1, 0] = float("inf")

    parent = _legacy_project(points, metas, image_wh, contract="full_fov")
    candidate = geometry.ocam_project_points(
        points,
        metas,
        image_wh,
        contract="full_fov",
    )
    with torch.no_grad():
        reused = geometry.ocam_project_points(
            points,
            metas,
            image_wh,
            contract="full_fov",
            inference_storage_reuse=True,
        )

    assert torch.equal(candidate, parent)
    assert torch.equal(reused, parent)
    assert torch.equal(candidate, torch.full_like(candidate, 2.0))


def test_scalar_fallback_source_avoids_full_fallback_inputs():
    sanitize_source = inspect.getsource(geometry._sanitize_camera_xyz)
    projection_source = inspect.getsource(geometry.ocam_project_points)
    assert "torch.zeros_like(camera_xyz)" not in sanitize_source
    assert "torch.isfinite(camera_xyz)" not in sanitize_source
    assert "torch.zeros_like(rho)" not in projection_source
    assert "torch.isfinite(pixel_" not in projection_source
    assert "torch.full_like(normalized_u" not in projection_source


@pytest.mark.parametrize("coefficient_count", (1, 2, 5, 8))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_horner_defers_only_the_redundant_full_clone(coefficient_count, dtype):
    generator = torch.Generator().manual_seed(3407 + coefficient_count)
    coefficients = torch.randn(
        2,
        3,
        1,
        1,
        coefficient_count,
        dtype=dtype,
        generator=generator,
        requires_grad=True,
    )
    value = torch.randn(
        2,
        3,
        11,
        5,
        dtype=dtype,
        generator=generator,
        requires_grad=True,
    )
    candidate_coefficients = coefficients.detach().clone().requires_grad_()
    candidate_value = value.detach().clone().requires_grad_()
    parent = _legacy_horner(coefficients, value)
    candidate = geometry._horner(candidate_coefficients, candidate_value)
    assert torch.equal(candidate, parent)
    assert candidate.stride() == parent.stride()
    upstream = torch.randn(
        parent.shape,
        dtype=dtype,
        generator=generator,
    )
    parent_gradients = torch.autograd.grad(
        (parent * upstream).sum(),
        (coefficients, value),
        allow_unused=True,
    )
    candidate_gradients = torch.autograd.grad(
        (candidate * upstream).sum(),
        (candidate_coefficients, candidate_value),
        allow_unused=True,
    )
    for candidate_gradient, parent_gradient in zip(
        candidate_gradients, parent_gradients
    ):
        if parent_gradient is None:
            assert candidate_gradient is None
        else:
            assert torch.equal(candidate_gradient, parent_gradient)
