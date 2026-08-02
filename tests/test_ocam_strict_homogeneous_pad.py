"""Raw-bit CPU contract for the strict OCam homogeneous-pad candidate."""

import pytest
import torch
from torch.nn import functional as F

from rosa import geometry


def _legacy_camera_xyz(points: torch.Tensor, lidar2cam: torch.Tensor) -> torch.Tensor:
    homogeneous = torch.cat(
        (points, torch.ones_like(points[..., :1])),
        dim=-1,
    )
    if points.ndim == 4:
        homogeneous = homogeneous[:, None]
    return torch.matmul(
        lidar2cam[:, :, None, None],
        homogeneous[..., None],
    ).squeeze(-1)[..., :3]


def _inputs(dtype: torch.dtype, camera_conditioned: bool, seed: int):
    generator = torch.Generator().manual_seed(seed)
    batch, cameras, queries, keypoints = 2, 4, 17, 13
    point_shape = (
        (batch, cameras, queries, keypoints, 3)
        if camera_conditioned
        else (batch, queries, keypoints, 3)
    )
    points = torch.randn(*point_shape, generator=generator, dtype=dtype)
    lidar2cam = torch.randn(
        batch,
        cameras,
        4,
        4,
        generator=generator,
        dtype=dtype,
    )
    lidar2cam[..., 3, :] = torch.tensor(
        (0.0, 0.0, 0.0, 1.0),
        dtype=dtype,
    )
    probe = torch.randn(
        batch,
        cameras,
        queries,
        keypoints,
        3,
        generator=generator,
        dtype=dtype,
    )
    return points, lidar2cam, probe


def _forward_and_gradients(transform, points, lidar2cam, probe):
    points = points.detach().clone().requires_grad_(True)
    lidar2cam = lidar2cam.detach().clone().requires_grad_(True)
    output = transform(points, lidar2cam)
    gradients = torch.autograd.grad((output * probe).sum(), (points, lidar2cam))
    return output.detach(), tuple(value.detach() for value in gradients)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("camera_conditioned", (False, True))
@pytest.mark.parametrize("seed", (3407, 7101080, 918031))
def test_strict_pad_preserves_cpu_forward_and_gradient_bits(
    dtype,
    camera_conditioned,
    seed,
):
    points, lidar2cam, probe = _inputs(dtype, camera_conditioned, seed)
    expected = _forward_and_gradients(
        _legacy_camera_xyz,
        points,
        lidar2cam,
        probe,
    )
    actual = _forward_and_gradients(
        geometry._camera_xyz,
        points,
        lidar2cam,
        probe,
    )

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1][0], expected[1][0])
    assert torch.equal(actual[1][1], expected[1][1])


@pytest.mark.parametrize("camera_conditioned", (False, True))
def test_strict_pad_preserves_homogeneous_input_and_matmul_shapes(
    monkeypatch,
    camera_conditioned,
):
    points, lidar2cam, _ = _inputs(torch.float32, camera_conditioned, 3407)
    legacy_homogeneous = torch.cat(
        (points, torch.ones_like(points[..., :1])),
        dim=-1,
    )
    padded_homogeneous = F.pad(
        points,
        (0, 1),
        mode="constant",
        value=1.0,
    )
    assert torch.equal(padded_homogeneous, legacy_homogeneous)
    assert padded_homogeneous.stride() == legacy_homogeneous.stride()

    calls = []
    original_matmul = torch.matmul

    def recorded_matmul(left, right):
        calls.append((tuple(left.shape), tuple(right.shape)))
        return original_matmul(left, right)

    monkeypatch.setattr(torch, "matmul", recorded_matmul)
    geometry._camera_xyz(points, lidar2cam)
    expected_right = (
        (2, 4, 17, 13, 4, 1)
        if camera_conditioned
        else (2, 1, 17, 13, 4, 1)
    )
    assert calls == [((2, 4, 1, 1, 4, 4), expected_right)]


def test_strict_pad_removes_python_ones_and_concat(monkeypatch):
    points, lidar2cam, _ = _inputs(torch.float32, False, 3407)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("separate homogeneous allocation was reintroduced")

    monkeypatch.setattr(torch, "ones_like", forbidden)
    monkeypatch.setattr(torch, "cat", forbidden)
    output = geometry._camera_xyz(points, lidar2cam)
    assert output.shape == (2, 4, 17, 13, 3)
