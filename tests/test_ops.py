import math

import torch

from rosa.ops import (
    exact_zero_detached_base_add,
    wrapped_angle_difference,
)


def _yaw_vector(angle: torch.Tensor) -> torch.Tensor:
    return torch.stack((angle.sin(), angle.cos()), dim=-1)


def test_wrapped_yaw_error_uses_the_short_path_across_pi():
    epsilon = 1e-4
    prediction_angle = torch.tensor(math.pi - epsilon, requires_grad=True)
    target_angle = torch.tensor(-math.pi + epsilon)

    error = wrapped_angle_difference(
        _yaw_vector(prediction_angle),
        _yaw_vector(target_angle),
    )

    torch.testing.assert_close(
        error,
        torch.tensor(-2.0 * epsilon),
        atol=1e-6,
        rtol=0.0,
    )
    error.square().backward()
    assert prediction_angle.grad is not None
    assert torch.isfinite(prediction_angle.grad)


def test_exact_zero_update_preserves_value_but_keeps_residual_gradient():
    base = torch.tensor([1.0, -2.0], requires_grad=True)
    delta = torch.zeros(2, requires_grad=True)
    output = exact_zero_detached_base_add(base, delta)

    assert torch.equal(output, base)
    output.sum().backward()
    assert base.grad is None
    torch.testing.assert_close(delta.grad, torch.ones_like(delta))
