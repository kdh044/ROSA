import math

import torch

from rosa.ray_losses import FisheyeRayAuxiliaryLoss
from rosa.ray_state import RayState


def _anchor(center):
    center = torch.as_tensor(center, dtype=torch.float32)
    shape = center.shape[:-1]
    return torch.cat(
        (
            center,
            torch.zeros(shape + (3,)),
            torch.tensor((0.0, 1.0)).expand(shape + (2,)),
            torch.zeros(shape + (3,)),
        ),
        dim=-1,
    )


def test_exact_ray_and_range_have_zero_angular_and_log_range_loss():
    target = _anchor([[[3.0, 4.0, 0.0], [0.0, 10.0, 0.0]]])
    state = RayState.from_sparse_anchor(target, range_std=1.0)
    loss = FisheyeRayAuxiliaryLoss()(state, target, torch.ones(1, 2, dtype=torch.bool))

    torch.testing.assert_close(loss["loss_ray_angular"], torch.tensor(0.0))
    torch.testing.assert_close(loss["loss_ray_log_range"], torch.tensor(0.0))
    torch.testing.assert_close(loss["loss_ray_range_nll"], torch.tensor(0.0))


def test_unmatched_queries_do_not_contribute_and_empty_batch_is_finite():
    target = _anchor([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]])
    state = RayState.from_sparse_anchor(target)
    wrong = RayState(
        direction=-state.direction,
        log_range=state.log_range + math.log(2.0),
        log_range_std=state.log_range_std,
        log_size=state.log_size,
        yaw=state.yaw,
        velocity=state.velocity,
    )
    module = FisheyeRayAuxiliaryLoss()

    masked = module(wrong, target, torch.tensor([[True, False]]))
    empty = module(wrong, target, torch.zeros(1, 2, dtype=torch.bool))

    assert masked["loss_ray_angular"] > 0
    assert masked["loss_ray_log_range"] > 0
    assert torch.isfinite(torch.stack(tuple(empty.values()))).all()
    torch.testing.assert_close(torch.stack(tuple(empty.values())), torch.zeros(3))


def test_auxiliary_loss_backpropagates_to_range_mean_and_uncertainty():
    target = _anchor([[[8.0, 1.0, 0.5]]])
    base = RayState.from_sparse_anchor(target)
    log_range = (base.log_range + 0.3).detach().requires_grad_(True)
    log_std = base.log_range_std.detach().requires_grad_(True)
    direction = (base.direction + 0.1).detach().requires_grad_(True)
    state = RayState(
        direction=direction,
        log_range=log_range,
        log_range_std=log_std,
        log_size=base.log_size,
        yaw=base.yaw,
        velocity=base.velocity,
    )

    losses = FisheyeRayAuxiliaryLoss()(
        state, target, torch.ones(1, 1, dtype=torch.bool)
    )
    sum(losses.values()).backward()

    assert direction.grad is not None and torch.count_nonzero(direction.grad) > 0
    assert log_range.grad is not None and torch.count_nonzero(log_range.grad) > 0
    assert log_std.grad is not None and torch.count_nonzero(log_std.grad) > 0
