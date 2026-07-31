import torch

from rosa.refiner import (
    ROSAConfig,
    ROSARefiner,
)


def _fixture(cfg=None):
    cfg = cfg or ROSAConfig(embed_dims=16, hidden_dims=8)
    stage = ROSARefiner(cfg, "ocam_fisheye")
    batch, queries, cameras, dims = 1, 2, 4, cfg.embed_dims
    anchor = torch.zeros(batch, queries, 11)
    anchor[..., 0] = torch.tensor((1.0, 2.0))
    anchor[..., 2] = -5.0
    anchor[..., 3:6] = torch.tensor((1.5, 3.0, 1.5)).log()
    anchor[..., 7] = 1.0
    query = torch.randn(batch, queries, dims)
    fpn = [torch.randn(batch, cameras, dims, size, size) for size in (16, 8, 4, 2)]
    lidar2cam = torch.eye(4).reshape(1, 1, 4, 4).repeat(batch, cameras, 1, 1)
    intrinsic = torch.eye(3).reshape(1, 1, 3, 3).repeat(batch, cameras, 1, 1)
    intrinsic[..., 0, 2] = 400.0
    intrinsic[..., 1, 2] = 400.0
    polynomial = torch.zeros(batch, cameras, 5)
    polynomial[..., 1] = -150.0
    metas = {
        "lidar2cam": lidar2cam,
        "ocam_intrinsic": intrinsic,
        "ocam_pol": polynomial,
        "image_wh": torch.full((batch, cameras, 2), 800.0),
        "ocam_lens_fov_deg": 205.0,
    }

    def keypoint_generator(box, _query):
        scale = box.new_tensor(
            (
                (0, 0, 0),
                (0.45, 0, 0),
                (-0.45, 0, 0),
                (0, 0.45, 0),
                (0, -0.45, 0),
                (0, 0, 0.45),
                (0, 0, -0.45),
                (0.2, 0.2, 0.2),
                (-0.2, 0.2, 0.2),
                (0.2, -0.2, 0.2),
                (-0.2, -0.2, 0.2),
                (0.2, 0.2, -0.2),
                (-0.2, -0.2, -0.2),
            )
        )
        return (
            box[..., :3, None].transpose(-1, -2)
            + scale[None, None] * box[..., 3:6].exp()[..., None, :]
        )

    return stage, anchor, query, fpn, metas, keypoint_generator


def test_closed_loop_is_exact_native_identity_at_initialization():
    stage, anchor, query, fpn, metas, generator = _fixture()
    output = stage(anchor, query, fpn, metas, generator)
    assert torch.equal(output.final_anchor, anchor)
    assert torch.count_nonzero(output.radial_gate) == 0
    assert torch.count_nonzero(output.quality_delta) == 0
    assert output.candidate_probabilities.shape == (1, 2, 5)
    assert output.candidate_features.shape == (1, 2, 5, 16)
    camera_mass = output.candidate_probabilities.sum(dim=-1)
    torch.testing.assert_close(camera_mass, torch.ones_like(camera_mass))


def test_closed_loop_sidecar_has_finite_backward():
    stage, anchor, query, fpn, metas, generator = _fixture()
    output = stage(anchor, query, fpn, metas, generator)
    objective = (
        output.candidate_logits.square().mean()
        + output.shape1_size_delta.square().mean()
        + output.quality_delta.square().mean()
    )
    objective.backward()
    gradients = [
        parameter.grad for parameter in stage.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_iterative_distribution_is_identity_and_distance_adaptive():
    cfg = ROSAConfig(
        embed_dims=16,
        hidden_dims=8,
        num_refinement_steps=2,
    )
    stage, anchor, query, fpn, metas, generator = _fixture(cfg)
    output = stage(anchor, query, fpn, metas, generator)

    assert torch.equal(output.final_anchor, anchor)
    assert output.iteration_anchors.shape == (1, 2, 2, 11)
    assert output.iteration_candidate_offsets.shape == (1, 2, 2, 5)
    assert output.iteration_candidate_probabilities.shape == (1, 2, 2, 5)
    torch.testing.assert_close(
        output.distribution_std,
        torch.full_like(output.distribution_std, cfg.initial_log_range_std),
        atol=1e-6,
        rtol=0.0,
    )
    # Both queries have the same learned relative uncertainty, so the query
    # at twice the range receives approximately twice the metric span.
    span = (
        output.iteration_candidate_offsets[:, :, 0, -1]
        - output.iteration_candidate_offsets[:, :, 0, 0]
    )
    torch.testing.assert_close(
        span[:, 1] / span[:, 0],
        torch.full_like(span[:, 0], 2.0),
        atol=2e-2,
        rtol=2e-2,
    )


def test_iterative_distribution_uses_shared_block_and_separate_heads():
    cfg = ROSAConfig(
        embed_dims=16,
        hidden_dims=8,
        num_refinement_steps=2,
    )
    stage, anchor, query, fpn, metas, generator = _fixture(cfg)
    assert not hasattr(stage, "shape2_head")
    shape_parameters = {id(value) for value in stage.shared_shape_head.parameters()}
    center_parameters = {id(value) for value in stage.shared_tangent_head.parameters()}
    assert shape_parameters.isdisjoint(center_parameters)

    output = stage(anchor, query, fpn, metas, generator)
    objective = (
        output.iteration_distribution_mean.square().mean()
        + output.iteration_distribution_std.square().mean()
        + output.candidate_logits.square().mean()
        + output.shape1_size_delta.square().mean()
    )
    objective.backward()
    gradients = [
        parameter.grad for parameter in stage.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_coarse_to_fine_second_step_uses_one_reprojection():
    cfg = ROSAConfig(
        embed_dims=16,
        hidden_dims=8,
        num_refinement_steps=2,
    )
    stage, anchor, query, fpn, metas, generator = _fixture(cfg)
    calls = 0

    def counted_generator(box, sidecar_query):
        nonlocal calls
        calls += 1
        return generator(box, sidecar_query)

    output = stage(anchor, query, fpn, metas, counted_generator)
    # Step 1: current box + five hypotheses. Step 2: updated box only.
    assert calls == 7
    assert output.iteration_candidate_offsets.shape == (1, 2, 2, 5)
    assert torch.equal(output.final_anchor, anchor)
    torch.testing.assert_close(
        output.posterior_entropy,
        torch.ones_like(output.posterior_entropy),
        atol=1e-6,
        rtol=0.0,
    )


def test_learned_distribution_can_disable_second_pass():
    cfg = ROSAConfig(
        embed_dims=16,
        hidden_dims=8,
        num_refinement_steps=1,
    )
    stage, anchor, query, fpn, metas, generator = _fixture(cfg)
    calls = 0

    def counted_generator(box, sidecar_query):
        nonlocal calls
        calls += 1
        return generator(box, sidecar_query)

    output = stage(anchor, query, fpn, metas, counted_generator)
    # One pass observes the current box and its five learned hypotheses.
    assert calls == 6
    assert output.iteration_anchors.shape == (1, 2, 1, 11)
    assert output.iteration_candidate_offsets.shape == (1, 2, 1, 5)
    assert torch.count_nonzero(output.shape2_size_delta) == 0
    assert torch.count_nonzero(output.shape2_yaw_delta) == 0


def test_uniform_camera_fusion_preserves_controlled_graph():
    cfg = ROSAConfig(
        embed_dims=16,
        hidden_dims=8,
        uniform_camera_fusion=True,
    )
    stage, anchor, query, fpn, metas, generator = _fixture(cfg)
    output = stage(anchor, query, fpn, metas, generator)
    assert torch.equal(output.final_anchor, anchor)
    objective = (
        output.candidate_logits.square().mean()
        + output.shape1_size_delta.square().mean()
    )
    objective.backward()
    assert stage.rosa.key_proj.weight.grad is not None
    assert torch.count_nonzero(stage.rosa.key_proj.weight.grad) == 0


def test_rfl_and_geometry_off_preserve_controlled_graph():
    cfg = ROSAConfig(
        embed_dims=16,
        hidden_dims=8,
        enable_rfl=False,
        enable_rosa_geometry=False,
    )
    stage, anchor, query, fpn, metas, generator = _fixture(cfg)
    output = stage(anchor, query, fpn, metas, generator)
    assert torch.equal(output.final_anchor, anchor)
    objective = (
        output.candidate_logits.square().mean()
        + output.shape1_size_delta.square().mean()
    )
    objective.backward()
    rfl_grad = stage.rosa.rfl.output_proj.weight.grad
    assert rfl_grad is not None
    assert torch.count_nonzero(rfl_grad) == 0
    key_grad = stage.rosa.key_proj.weight.grad
    assert key_grad is not None
    geometry_grad = key_grad[:, -stage.rosa.geometry_dims :]
    assert torch.count_nonzero(geometry_grad) == 0


def test_force_radial_gate_one_bypasses_learned_gate():
    cfg = ROSAConfig(
        embed_dims=16,
        hidden_dims=8,
        num_refinement_steps=2,
        force_radial_gate_one=True,
    )
    stage, anchor, query, fpn, metas, generator = _fixture(cfg)
    output = stage(anchor, query, fpn, metas, generator)
    assert torch.equal(output.radial_gate, torch.ones_like(output.radial_gate))


def test_uniform_point_fusion_preserves_controlled_graph():
    cfg = ROSAConfig(
        embed_dims=16,
        hidden_dims=8,
        uniform_point_fusion=True,
    )
    stage, anchor, query, fpn, metas, generator = _fixture(cfg)
    output = stage(anchor, query, fpn, metas, generator)
    assert torch.equal(output.final_anchor, anchor)
    objective = (
        output.candidate_logits.square().mean()
        + output.shape1_size_delta.square().mean()
    )
    objective.backward()
    assert stage.rosa.point_query.weight.grad is not None
    assert torch.count_nonzero(stage.rosa.point_query.weight.grad) == 0
    assert stage.rosa.point_key.weight.grad is not None
    assert torch.count_nonzero(stage.rosa.point_key.weight.grad) == 0


def test_fixed_metric_hypotheses_keep_two_pass_full_graph():
    cfg = ROSAConfig(
        embed_dims=16,
        hidden_dims=8,
        num_refinement_steps=2,
        fixed_metric_radial_hypotheses=True,
        radial_step_m=0.5,
    )
    stage, anchor, query, fpn, metas, generator = _fixture(cfg)
    output = stage(anchor, query, fpn, metas, generator)
    expected_near_clipped = anchor.new_tensor((-0.5, -0.5, 0.0, 0.5, 1.0))
    expected = anchor.new_tensor((-1.0, -0.5, 0.0, 0.5, 1.0))
    torch.testing.assert_close(
        output.iteration_candidate_offsets[0, 0, 0],
        expected_near_clipped,
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        output.iteration_candidate_offsets[0, 1, 0],
        expected,
        atol=1e-6,
        rtol=0.0,
    )
    assert output.iteration_anchors.shape == (1, 2, 2, 11)
    objective = (
        output.iteration_distribution_mean.square().mean()
        + output.candidate_logits.square().mean()
        + output.shape1_size_delta.square().mean()
    )
    objective.backward()
    gradients = [
        parameter.grad for parameter in stage.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
