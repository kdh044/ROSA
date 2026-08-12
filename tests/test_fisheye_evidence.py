import torch

from rosa.fisheye_evidence import (
    FisheyeEvidenceConfig,
    FisheyeObservationGeometry,
    UnifiedFisheyeEvidenceAttention,
    angular_fpn_geometry,
)


def _geometry(batch=2, queries=3, cameras=4, *, all_valid=True):
    rays = torch.randn(batch, queries, cameras, 3)
    rays = torch.nn.functional.normalize(rays, dim=-1)
    tangent = torch.randn(batch, queries, cameras, 2) * 0.1
    jacobian = torch.eye(2).reshape(1, 1, 1, 2, 2).repeat(batch, queries, cameras, 1, 1)
    jacobian = jacobian * 100.0
    lens_margin = torch.full((batch, queries, cameras, 1), 0.5)
    overlap = torch.full((batch, queries, cameras, 1), 0.5)
    valid = torch.ones(batch, queries, cameras, dtype=torch.bool)
    if not all_valid:
        valid[:, :, -1] = False
        valid[:, -1] = False
    return FisheyeObservationGeometry(
        rays=rays,
        tangent=tangent,
        jacobian=jacobian,
        lens_margin=lens_margin,
        overlap=overlap,
        valid=valid,
    )


def test_jacobian_level_matching_uses_angular_not_pixel_footprint():
    geometry = _geometry(batch=1, queries=1, cameras=1)
    strides = torch.tensor((8.0, 16.0, 32.0, 64.0))
    # 100 pixels/radian means stride 16 covers 0.16 radians.
    extent = torch.tensor([[[0.16, 0.16]]]).log()

    descriptor, compatibility = angular_fpn_geometry(
        geometry.tangent,
        geometry.jacobian,
        geometry.lens_margin,
        geometry.overlap,
        geometry.valid,
        strides,
        extent,
    )

    assert descriptor.shape == (1, 1, 1, 4, 9)
    assert compatibility.shape == (1, 1, 1, 4)
    assert compatibility.argmax(dim=-1).item() == 1


def test_isotropic_lens_jacobian_is_a_finite_calibration_gradient_boundary():
    geometry = _geometry(batch=1, queries=2, cameras=4)
    tangent = geometry.tangent.requires_grad_()
    # Equal singular values are common near locally isotropic lens regions.
    # Their decomposition has no unique backward direction, so calibrated
    # Jacobians intentionally condition the forward pass without receiving
    # optimizer gradients.
    jacobian = geometry.jacobian.requires_grad_()
    extent = torch.full((1, 2, 2), 0.16).log().requires_grad_()

    descriptor, compatibility = angular_fpn_geometry(
        tangent,
        jacobian,
        geometry.lens_margin,
        geometry.overlap,
        geometry.valid,
        torch.tensor((8.0, 16.0, 32.0, 64.0)),
        extent,
    )
    (descriptor.square().mean() + compatibility.square().mean()).backward()

    assert jacobian.grad is None
    assert tangent.grad is not None and torch.isfinite(tangent.grad).all()
    assert extent.grad is not None and torch.isfinite(extent.grad).all()


def test_attention_has_one_camera_level_distribution_per_head():
    config = FisheyeEvidenceConfig(
        embed_dims=16,
        num_heads=4,
        num_cameras=4,
        level_strides=(8.0, 16.0, 32.0, 64.0),
        geometry_hidden_dims=8,
    )
    module = UnifiedFisheyeEvidenceAttention(config)
    query = torch.randn(2, 3, 16)
    features = torch.randn(2, 3, 4, 4, 16)
    geometry = _geometry(all_valid=False)

    output = module(query, features, geometry)

    assert output.query.shape == query.shape
    assert output.attention.shape == (2, 3, 4, 4, 4)
    assert output.level_compatibility.shape == (2, 3, 4, 4)
    assert output.supported.shape == (2, 3)
    # Valid queries normalize over camera x level jointly, not separately.
    mass = output.attention.sum(dim=(2, 3))
    torch.testing.assert_close(mass[:, :-1], torch.ones_like(mass[:, :-1]))
    torch.testing.assert_close(mass[:, -1], torch.zeros_like(mass[:, -1]))
    torch.testing.assert_close(
        output.attention[:, :-1, -1],
        torch.zeros_like(output.attention[:, :-1, -1]),
    )
    assert output.supported[:, :-1].all()
    assert not output.supported[:, -1].any()


def test_all_invalid_evidence_is_exact_query_identity():
    config = FisheyeEvidenceConfig(
        embed_dims=8,
        num_heads=2,
        num_cameras=4,
        geometry_hidden_dims=8,
    )
    module = UnifiedFisheyeEvidenceAttention(config)
    query = torch.randn(1, 2, 8)
    features = torch.randn(1, 2, 4, 4, 8)
    geometry = _geometry(batch=1, queries=2, all_valid=True)
    geometry = FisheyeObservationGeometry(
        rays=geometry.rays,
        tangent=geometry.tangent,
        jacobian=geometry.jacobian,
        lens_margin=geometry.lens_margin,
        overlap=geometry.overlap,
        valid=torch.zeros_like(geometry.valid),
    )

    output = module(query, features, geometry)

    assert torch.equal(output.query, query)
    assert torch.count_nonzero(output.attention) == 0
    assert not output.supported.any()


def test_rfl_value_and_geometry_paths_receive_training_gradients():
    torch.manual_seed(3407)
    config = FisheyeEvidenceConfig(
        embed_dims=16,
        num_heads=4,
        num_cameras=4,
        geometry_hidden_dims=8,
    )
    module = UnifiedFisheyeEvidenceAttention(config)
    query = torch.randn(2, 3, 16, requires_grad=True)
    features = torch.randn(2, 3, 4, 4, 16, requires_grad=True)
    geometry = _geometry()

    output = module(query, features, geometry)
    loss = output.query.square().mean() + output.attention.square().mean()
    loss.backward()

    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert module.rfl.output_proj.weight.grad is not None
    assert torch.count_nonzero(module.rfl.output_proj.weight.grad) > 0
    assert module.geometry_key[-1].weight.grad is not None
    assert torch.count_nonzero(module.geometry_key[-1].weight.grad) > 0
    assert module.scale_match_raw.grad is not None
    assert torch.count_nonzero(module.scale_match_raw.grad) > 0
