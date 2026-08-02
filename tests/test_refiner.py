from collections import Counter
import sys
import types

import pytest
import torch

import rosa.refiner as refiner_module
import rosa.rfl as rfl_module
from rosa.refiner import (
    _CandidateEvidence,
    _candidate_confidence,
    ROSAConfig,
    ROSARefiner,
)


def _raw_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(
        left.dtype == right.dtype
        and left.shape == right.shape
        and torch.equal(
            left.detach().contiguous().view(torch.uint8),
            right.detach().contiguous().view(torch.uint8),
        )
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


def test_compact_lean_refiner_uses_only_private_prepared_sampling(monkeypatch):
    cfg = ROSAConfig.from_mapping(
        {
            "profile": "fast",
            "embed_dims": 16,
            "hidden_dims": 8,
        }
    )
    stage, anchor, query, fpn, metas, generator = _fixture(cfg)
    private_calls = []
    public_calls = []
    released_private = refiner_module._sample_multiscale_features_prepared_inference
    released_public = refiner_module.sample_multiscale_features

    def private(*args, **kwargs):
        private_calls.append(True)
        return released_private(*args, **kwargs)

    def public(*args, **kwargs):
        public_calls.append(True)
        return released_public(*args, **kwargs)

    monkeypatch.setattr(
        refiner_module,
        "_sample_multiscale_features_prepared_inference",
        private,
    )
    monkeypatch.setattr(refiner_module, "sample_multiscale_features", public)

    stage.eval()
    with torch.no_grad():
        stage(
            anchor,
            query,
            fpn,
            metas,
            generator,
            materialize_diagnostics=False,
            return_final_anchor_only=True,
        )
    assert len(private_calls) == 2
    assert public_calls == []

    private_calls.clear()
    stage.train()
    stage(anchor, query, fpn, metas, generator)
    assert private_calls == []
    assert len(public_calls) == 2


def test_compact_lean_refiner_reuses_exactly_two_rfl_delta_buffers():
    cfg = ROSAConfig.from_mapping(
        {
            "profile": "fast",
            "embed_dims": 16,
            "hidden_dims": 8,
        }
    )
    stage, anchor, query, fpn, metas, generator = _fixture(cfg)
    stage.eval()
    observed_calls = Counter()
    private_code = type(stage.rosa.rfl)._forward_lifted_owned_delta_inference.__code__
    finish_code = rfl_module._mask_cast_and_add_owned_delta.__code__

    def count_private_rfl(frame, event, _arg):
        if event == "call" and frame.f_code is private_code:
            observed_calls["private_rfl"] += 1
        elif event == "call" and frame.f_code is finish_code:
            observed_calls["owned_finish"] += 1

    with torch.no_grad():
        stage.rosa.rfl.output_proj.weight.normal_()
        previous_profile = sys.getprofile()
        sys.setprofile(count_private_rfl)
        try:
            stage(
                anchor,
                query,
                fpn,
                metas,
                generator,
                materialize_diagnostics=False,
                return_final_anchor_only=True,
            )
        finally:
            sys.setprofile(previous_profile)
    # Fast refinement observes the base anchors once and all three candidates
    # once, and each observation must consume exactly one owned delta.
    assert observed_calls == Counter({"private_rfl": 2, "owned_finish": 2})


def test_compact_lean_refiner_reuses_pointwise_storage_for_base_and_candidates():
    cfg = ROSAConfig.from_mapping(
        {
            "profile": "fast",
            "embed_dims": 16,
            "hidden_dims": 8,
        }
    )
    stage, anchor, query, fpn, metas, generator = _fixture(cfg)
    stage.eval()
    calls = []
    implementation_code = type(stage.rosa)._forward_impl.__code__

    def capture_private_flags(frame, event, _arg):
        if event == "call" and frame.f_code is implementation_code:
            calls.append(
                (
                    frame.f_locals["observation_usage"],
                    frame.f_locals["reuse_owned_rfl_delta"],
                    frame.f_locals["reuse_owned_pointwise_storage"],
                )
            )

    with torch.no_grad():
        stage.rosa.rfl.output_proj.weight.normal_()
        previous_profile = sys.getprofile()
        sys.setprofile(capture_private_flags)
        try:
            result = stage(
                anchor,
                query,
                fpn,
                metas,
                generator,
                materialize_diagnostics=False,
                return_final_anchor_only=True,
            )
        finally:
            sys.setprofile(previous_profile)
    assert result.shape == anchor.shape
    assert calls == [
        ("base_support", True, True),
        ("candidate", True, True),
    ]


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


@pytest.mark.parametrize(
    ("module_name", "relation_index"),
    (("rosa", 2), ("rfl", 1)),
)
def test_local_relation_hooks_keep_the_legacy_tensor_abi(
    module_name,
    relation_index,
):
    stage, anchor, query, fpn, metas, generator = _fixture()
    module = stage.rosa if module_name == "rosa" else stage.rosa.rfl
    observed = []

    def inspect_relation(_module, args):
        relation = args[relation_index]
        assert torch.is_tensor(relation)
        assert relation.shape[-1] == 19
        observed.append(relation[..., 13])

    handle = module.register_forward_pre_hook(inspect_relation)
    try:
        output = stage(anchor, query, fpn, metas, generator)
    finally:
        handle.remove()

    assert observed
    assert torch.equal(output.final_anchor, anchor)


def test_global_relation_hook_keeps_the_legacy_tensor_abi():
    stage, anchor, query, fpn, metas, generator = _fixture()
    observed = []

    def inspect_relation(module, args):
        if module is stage.rosa.rfl:
            relation = args[1]
            assert torch.is_tensor(relation)
            assert relation.shape[-1] == 19
            observed.append(relation[..., 8:11])

    handle = torch.nn.modules.module.register_module_forward_pre_hook(inspect_relation)
    try:
        output = stage(anchor, query, fpn, metas, generator)
    finally:
        handle.remove()

    assert observed
    assert torch.equal(output.final_anchor, anchor)


def test_instance_rfl_forward_override_keeps_the_legacy_tensor_abi():
    stage, anchor, query, fpn, metas, generator = _fixture()
    released_forward = stage.rosa.rfl.forward
    observed = []

    def custom_forward(_self, features, relation):
        assert torch.is_tensor(relation)
        assert relation.shape[-1] == 19
        observed.append(relation[..., 13])
        return released_forward(features, relation)

    stage.rosa.rfl.forward = types.MethodType(
        custom_forward,
        stage.rosa.rfl,
    )
    output = stage(anchor, query, fpn, metas, generator)

    assert observed
    assert torch.equal(output.final_anchor, anchor)


def test_class_pointwise_forward_override_keeps_the_legacy_tensor_abi(
    monkeypatch,
):
    stage, anchor, query, fpn, metas, generator = _fixture()
    pointwise_type = type(stage.rosa)
    released_forward = pointwise_type.forward
    observed = []

    def custom_forward(self, features, geometry, relation, valid, sidecar_query):
        assert torch.is_tensor(relation)
        assert relation.shape[-1] == 19
        observed.append(relation[..., 8:11])
        return released_forward(
            self,
            features,
            geometry,
            relation,
            valid,
            sidecar_query,
        )

    monkeypatch.setattr(pointwise_type, "forward", custom_forward)
    output = stage(anchor, query, fpn, metas, generator)

    assert observed
    assert torch.equal(output.final_anchor, anchor)


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


def test_balanced_profile_vectorizes_three_full_span_candidates():
    cfg = ROSAConfig.from_mapping(
        {
            "profile": "balanced",
            "embed_dims": 16,
            "hidden_dims": 8,
        }
    )
    stage, anchor, query, fpn, metas, generator = _fixture(cfg)
    query_counts = []

    def counted_generator(box, sidecar_query):
        query_counts.append(int(box.shape[1]))
        return generator(box, sidecar_query)

    output = stage(anchor, query, fpn, metas, counted_generator)

    # Current boxes, all three candidates in one batch, then the second pass.
    assert query_counts == [2, 6, 2]
    assert output.iteration_candidate_offsets.shape == (1, 2, 2, 3)
    assert output.candidate_features.shape == (1, 2, 3, 16)
    assert torch.equal(output.final_anchor, anchor)

    legacy, *_ = _fixture(ROSAConfig(embed_dims=16, hidden_dims=8))
    legacy_output = legacy(anchor, query, fpn, metas, generator)
    torch.testing.assert_close(
        output.iteration_candidate_offsets[:, :, 0, (0, 2)],
        legacy_output.iteration_candidate_offsets[:, :, 0, (0, 4)],
    )


def test_fast_profile_uses_one_vectorized_refinement_pass():
    cfg = ROSAConfig.from_mapping(
        {
            "profile": "fast",
            "embed_dims": 16,
            "hidden_dims": 8,
        }
    )
    stage, anchor, query, fpn, metas, generator = _fixture(cfg)
    query_counts = []

    def counted_generator(box, sidecar_query):
        query_counts.append(int(box.shape[1]))
        return generator(box, sidecar_query)

    output = stage(anchor, query, fpn, metas, counted_generator)
    assert query_counts == [2, 6]
    assert output.iteration_candidate_offsets.shape == (1, 2, 1, 3)
    assert output.iteration_anchors.shape == (1, 2, 1, 11)


@pytest.mark.parametrize("profile", ("fast", "published"))
def test_lean_inference_skips_only_diagnostic_materialization(profile):
    cfg = ROSAConfig.from_mapping(
        {
            "profile": profile,
            "embed_dims": 16,
            "hidden_dims": 8,
        }
    )
    stage, anchor, query, fpn, metas, generator = _fixture(cfg)
    stage.eval()
    parameter_generator = torch.Generator().manual_seed(7101080)
    with torch.no_grad():
        for parameter in stage.parameters():
            parameter.uniform_(-0.05, 0.05, generator=parameter_generator)
        full = stage(
            anchor,
            query,
            fpn,
            metas,
            generator,
            materialize_diagnostics=True,
        )
        lean = stage(
            anchor,
            query,
            fpn,
            metas,
            generator,
            materialize_diagnostics=False,
        )

    for name, expected in vars(full).items():
        actual = getattr(lean, name)
        if name.startswith("iteration_"):
            assert torch.is_tensor(expected)
            assert actual is None
        elif torch.is_tensor(expected):
            assert torch.equal(actual, expected), name
        else:
            assert actual == expected


def test_fast_final_anchor_only_is_raw_exact_and_skips_return_postprocessing(
    monkeypatch,
):
    cfg = ROSAConfig.from_mapping(
        {
            "profile": "fast",
            "embed_dims": 16,
            "hidden_dims": 8,
        }
    )
    stage, anchor, query, fpn, metas, generator = _fixture(cfg)
    stage.eval()
    parameter_generator = torch.Generator().manual_seed(314159)
    with torch.no_grad():
        for parameter in stage.parameters():
            parameter.uniform_(-0.05, 0.05, generator=parameter_generator)

    calls = []
    released = refiner_module._radial_basis

    def counted_radial_basis(center_xy):
        calls.append(tuple(center_xy.shape))
        return released(center_xy)

    monkeypatch.setattr(refiner_module, "_radial_basis", counted_radial_basis)
    with torch.no_grad():
        full = stage(
            anchor,
            query,
            fpn,
            metas,
            generator,
            materialize_diagnostics=False,
        )
        full_call_count = len(calls)
        rng_before = torch.get_rng_state().clone()
        final_anchor = stage(
            anchor,
            query,
            fpn,
            metas,
            generator,
            materialize_diagnostics=False,
            return_final_anchor_only=True,
        )
        minimal_call_count = len(calls) - full_call_count

    assert torch.is_tensor(final_anchor)
    assert _raw_equal(final_anchor, full.final_anchor)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert minimal_call_count + 1 == full_call_count


def test_final_anchor_only_rejects_training_or_full_diagnostics():
    cfg = ROSAConfig.from_mapping(
        {
            "profile": "fast",
            "embed_dims": 16,
            "hidden_dims": 8,
        }
    )
    stage, anchor, query, fpn, metas, generator = _fixture(cfg)
    stage.eval()
    with torch.no_grad(), pytest.raises(ValueError, match="cannot materialize"):
        stage(
            anchor,
            query,
            fpn,
            metas,
            generator,
            return_final_anchor_only=True,
        )

    stage.train()
    with torch.no_grad(), pytest.raises(RuntimeError, match="no-grad eval"):
        stage(
            anchor,
            query,
            fpn,
            metas,
            generator,
            materialize_diagnostics=False,
            return_final_anchor_only=True,
        )


def test_final_anchor_only_rejects_legacy_quality_head_even_when_score_is_disabled():
    cfg = ROSAConfig.from_mapping(
        {
            "profile": "published",
            "embed_dims": 16,
            "hidden_dims": 8,
            "enable_quality_score": False,
        }
    )
    stage, anchor, query, fpn, metas, generator = _fixture(cfg)
    assert stage.quality_head is not None
    stage.eval()

    with torch.no_grad(), pytest.raises(RuntimeError, match="no quality head"):
        stage(
            anchor,
            query,
            fpn,
            metas,
            generator,
            materialize_diagnostics=False,
            return_final_anchor_only=True,
        )


def test_fast_final_anchor_only_is_one_full_compiled_graph():
    cfg = ROSAConfig.from_mapping(
        {
            "profile": "fast",
            "embed_dims": 16,
            "hidden_dims": 8,
        }
    )
    stage, anchor, query, fpn, metas, generator = _fixture(cfg)
    stage.eval()
    parameter_generator = torch.Generator().manual_seed(271828)
    with torch.no_grad():
        for parameter in stage.parameters():
            parameter.uniform_(-0.05, 0.05, generator=parameter_generator)

    def full_anchor(*args):
        return stage(*args, materialize_diagnostics=False).final_anchor

    def minimal_anchor(*args):
        return stage(
            *args,
            materialize_diagnostics=False,
            return_final_anchor_only=True,
        )

    compiled_full = torch.compile(full_anchor, backend="eager", fullgraph=True)
    compiled_minimal = torch.compile(minimal_anchor, backend="eager", fullgraph=True)
    inputs = (anchor, query, fpn, metas, generator)

    with torch.no_grad():
        eager_full = full_anchor(*inputs)
        eager_minimal = minimal_anchor(*inputs)
        actual_full = compiled_full(*inputs)
        actual_minimal = compiled_minimal(*inputs)

    assert _raw_equal(eager_full, eager_minimal)
    assert torch.count_nonzero(eager_full - anchor) > 0
    assert _raw_equal(actual_full, eager_full)
    assert _raw_equal(actual_minimal, eager_minimal)
    assert _raw_equal(actual_full, actual_minimal)


def test_lean_inference_hook_keeps_full_observation_abi():
    cfg = ROSAConfig.from_mapping(
        {
            "profile": "fast",
            "embed_dims": 16,
            "hidden_dims": 8,
        }
    )
    stage, anchor, query, fpn, metas, generator = _fixture(cfg)
    stage.eval()
    observed_relations = []

    def inspect_relation(_module, args):
        observed_relations.append(args[1])

    handle = stage.rosa.rfl.register_forward_pre_hook(inspect_relation)
    try:
        with torch.no_grad():
            stage(
                anchor,
                query,
                fpn,
                metas,
                generator,
                materialize_diagnostics=False,
            )
    finally:
        handle.remove()

    assert observed_relations
    assert all(torch.is_tensor(relation) for relation in observed_relations)
    assert all(relation.shape[-1] == 19 for relation in observed_relations)


def test_compact_refiner_parameter_contract():
    balanced = ROSARefiner(ROSAConfig.from_mapping({"profile": "balanced"}))
    fast = ROSARefiner(ROSAConfig.from_mapping({"profile": "fast"}))
    assert sum(parameter.numel() for parameter in balanced.parameters()) == 87_935
    assert sum(parameter.numel() for parameter in fast.parameters()) == 87_935
    assert len(balanced.state_dict()) == 40

    rfl = fast.rosa.rfl
    assert sum(parameter.numel() for parameter in rfl.parameters()) == 8_336
    learned_macs_per_observation = (
        rfl.appearance_proj.weight.numel()
        + rfl.ray_proj.weight.numel()
        + rfl.output_proj.weight.numel()
    )
    assert learned_macs_per_observation == 8_336
    fast_observations = 4 * (50 + 150) * 4 * 13
    assert fast_observations == 41_600
    assert learned_macs_per_observation * fast_observations == 346_777_600
    assert rfl.hidden_dims * fast_observations == 665_600
    assert (
        learned_macs_per_observation + rfl.hidden_dims
    ) * fast_observations == 347_443_200


def test_compact_candidate_confidence_uses_only_supported_hypotheses():
    probabilities = torch.tensor(
        (
            (0.5, 0.5, 0.0),
            (1.0, 0.0, 0.0),
            (0.6, 0.0, 0.4),
            (1 / 3, 1 / 3, 1 / 3),
        )
    )
    supported = torch.tensor(
        (
            (True, True, False),
            (True, False, False),
            (True, False, True),
            (False, False, False),
        )
    )
    entropy, margin = _candidate_confidence(
        probabilities,
        supported,
        legacy=False,
    )
    torch.testing.assert_close(
        entropy[:2],
        torch.tensor((1.0, 0.0)),
    )
    torch.testing.assert_close(margin, torch.tensor((0.0, 1.0, 0.2, 0.0)))
    assert 0.9 < float(entropy[2]) < 1.0
    assert entropy[3] == 0


def test_balanced_profile_keeps_unsupported_boxes_exactly_unchanged():
    cfg = ROSAConfig.from_mapping(
        {
            "profile": "balanced",
            "embed_dims": 16,
            "hidden_dims": 8,
        }
    )
    stage, anchor, query, fpn, metas, generator = _fixture(cfg)
    anchor[:, 1, 2] = 5.0
    with torch.no_grad():
        stage.shared_shape_head[-1].bias.fill_(1.0)
        stage.shared_tangent_head[-1].bias.fill_(1.0)
        stage.range_distribution_head[-1].bias[0] = 0.2
        stage.center_gate_head[-1].bias.fill_(2.0)

    output = stage(anchor, query, fpn, metas, generator)
    assert output.initial_supported[0, 0]
    assert not output.initial_supported[0, 1]
    assert torch.equal(output.final_anchor[:, 1], anchor[:, 1])
    assert not torch.equal(output.final_anchor[:, 0], anchor[:, 0])


def test_yaw_off_keeps_yaw_while_size_refinement_remains_active():
    cfg = ROSAConfig.from_mapping(
        {
            "profile": "balanced",
            "embed_dims": 16,
            "hidden_dims": 8,
            "yaw_update_steps": "off",
        }
    )
    stage, anchor, query, fpn, metas, generator = _fixture(cfg)
    with torch.no_grad():
        stage.shared_shape_head[-1].bias[:3].fill_(0.5)
        stage.shared_shape_head[-1].bias[3] = 1.0

    output = stage(anchor, query, fpn, metas, generator)
    assert torch.equal(output.final_anchor[..., 6:8], anchor[..., 6:8])
    assert not torch.equal(output.final_anchor[..., 3:6], anchor[..., 3:6])


@pytest.mark.parametrize("fixed_metric", (False, True))
def test_partial_candidate_support_has_an_exact_zero_initial_prior(fixed_metric):
    config = ROSAConfig.from_mapping({"profile": "balanced"})
    stage = ROSARefiner(config)
    probabilities = torch.tensor([[[0.5, 0.5, 0.0]]])
    supported = torch.tensor([[[True, True, False]]])
    offsets = torch.tensor([[[-1.0, 0.0, 1.0]]])
    features = torch.zeros(1, 1, 3, config.embed_dims)
    evidence = _CandidateEvidence(
        features=features,
        supported=supported,
        logits=torch.zeros_like(probabilities),
        probabilities=probabilities,
        offsets=offsets,
        radial_delta=torch.zeros(1, 1),
        entropy=torch.zeros(1, 1),
        margin=torch.zeros(1, 1),
        weighted_feature=torch.zeros(1, 1, config.embed_dims),
    )

    proposal = stage._posterior_radial_proposal(
        evidence,
        current_range=torch.full((1, 1), 10.0),
        distribution_mean=torch.zeros(1, 1),
        distribution_std=torch.full((1, 1), 0.05),
        sigma_points=torch.tensor((-2.0, 0.0, 2.0)),
        fixed_metric=fixed_metric,
        step=0,
    )
    assert torch.equal(proposal, torch.zeros_like(proposal))


def test_corrected_geometry_normalizes_yaw_without_changing_native_box():
    cfg = ROSAConfig.from_mapping(
        {
            "profile": "fast",
            "embed_dims": 16,
            "hidden_dims": 8,
        }
    )
    stage, anchor, query, fpn, metas, generator = _fixture(cfg)
    anchor[..., 6] = 0.0
    anchor[..., 7] = 2.0
    observed_norms = []

    def captured_generator(box, sidecar_query):
        observed_norms.append(torch.linalg.vector_norm(box[..., 6:8], dim=-1))
        return generator(box, sidecar_query)

    output = stage(anchor, query, fpn, metas, captured_generator)
    assert observed_norms
    for norm in observed_norms:
        torch.testing.assert_close(norm, torch.ones_like(norm))
    assert torch.equal(output.final_anchor[..., 6:8], anchor[..., 6:8])
