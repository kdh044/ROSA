import math

import torch

from rosa.ray_state import (
    FisheyeRayStateRefinement,
    RayRefinementConfig,
    RayState,
    log_range_gaussian_nll,
    normalize_direction,
    spherical_exp_map,
    spherical_tangent_basis,
)


def test_tangent_basis_and_exp_map_are_spherical_and_finite_at_poles():
    direction = torch.tensor(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)))
    e1, e2 = spherical_tangent_basis(direction)

    torch.testing.assert_close((e1 * direction).sum(-1), torch.zeros(3))
    torch.testing.assert_close((e2 * direction).sum(-1), torch.zeros(3))
    torch.testing.assert_close(e1.square().sum(-1), torch.ones(3))
    torch.testing.assert_close(e2.square().sum(-1), torch.ones(3))

    moved = spherical_exp_map(
        direction, torch.tensor(((0.0, 0.0), (0.1, -0.2), (-0.3, 0.1)))
    )
    assert torch.isfinite(moved).all()
    torch.testing.assert_close(moved.square().sum(-1), torch.ones(3))
    torch.testing.assert_close(moved[0], direction[0])


def test_sparse_anchor_round_trip_preserves_box_and_local_velocity():
    center = torch.tensor(((3.0, 4.0, 1.0), (-5.0, 2.0, -0.5)))
    log_size = torch.tensor(((0.2, 1.1, 0.4), (-0.1, 0.7, 0.2)))
    angle = torch.tensor((2.8, -2.2))
    yaw = torch.stack((angle.sin(), angle.cos()), dim=-1)
    velocity = torch.tensor(((1.0, -2.0, 0.1), (-0.3, 2.4, -0.2)))
    anchor = torch.cat((center, log_size, yaw, velocity), dim=-1)

    state = RayState.from_sparse_anchor(anchor)
    decoded = state.to_sparse_anchor()

    torch.testing.assert_close(decoded, anchor, atol=2e-6, rtol=2e-6)


def test_d0_zero_initialization_creates_declared_box_free_priors():
    config = RayRefinementConfig(embed_dims=8, hidden_dims=16)
    module = FisheyeRayStateRefinement(config, initialize=True)
    feature = torch.randn(2, 5, 8)
    direction = normalize_direction(torch.randn(2, 5, 3))

    state = module(feature, direction)

    torch.testing.assert_close(state.direction, direction)
    torch.testing.assert_close(
        state.range, torch.full((2, 5, 1), config.initial_range_m)
    )
    torch.testing.assert_close(
        state.range_std,
        torch.full((2, 5, 1), config.initial_range_std_m),
    )
    expected_size = torch.tensor(config.initial_size_m).log().expand(2, 5, 3)
    torch.testing.assert_close(state.log_size, expected_size)
    torch.testing.assert_close(state.yaw, torch.tensor((0.0, 1.0)).expand(2, 5, 2))
    torch.testing.assert_close(state.velocity, torch.zeros(2, 5, 3))


def test_d1_zero_initialization_is_exact_state_identity():
    config = RayRefinementConfig(embed_dims=8, hidden_dims=16)
    initializer = FisheyeRayStateRefinement(config, initialize=True)
    refiner = FisheyeRayStateRefinement(config, initialize=False)
    feature = torch.randn(1, 4, 8)
    direction = normalize_direction(torch.randn(1, 4, 3))
    state = initializer(feature, direction)

    refined = refiner(feature, state.direction, state)

    for field in (
        "direction",
        "log_range",
        "log_range_std",
        "log_size",
        "yaw",
        "velocity",
    ):
        torch.testing.assert_close(getattr(refined, field), getattr(state, field))


def test_range_nll_uses_euclidean_range_not_optical_axis_depth():
    direction = normalize_direction(torch.tensor([[[1.0, 1.0, 1.0]]]))
    target_center = direction * 12.0
    state = RayState(
        direction=direction,
        log_range=torch.full((1, 1, 1), math.log(12.0)),
        log_range_std=torch.zeros(1, 1, 1),
        log_size=torch.zeros(1, 1, 3),
        yaw=torch.tensor([[[0.0, 1.0]]]),
        velocity=torch.zeros(1, 1, 3),
    )

    nll = log_range_gaussian_nll(state, target_center)

    torch.testing.assert_close(nll, torch.zeros_like(nll), atol=1e-6, rtol=0.0)
