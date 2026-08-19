from dataclasses import fields

import pytest
import torch

from rosa.fisheye_box_decoder import FisheyeBoxKeypointGenerator
from rosa.fisheye_projection import (
    FisheyeProjectionModel,
    LensProjection,
    NativeFisheyeProjector,
)
from rosa.hybrid_box_decoder import (
    FisheyeBoxDecoderV4Config,
    FisheyeCenterQueryInitializer,
    FisheyeSparseBoxDecoderV4,
    HybridBoxRefinement,
    HybridBoxState,
)
from rosa.ray_losses import (
    FisheyeHybridRayAuxiliaryLoss,
    RayAuxiliaryLossConfig,
)
from rosa.sampling import DeterministicBilinearSampler


class _LinearLens(FisheyeProjectionModel):
    def project_rays(self, camera_rays, metadata, image_wh):
        prefix = camera_rays.shape[:-1]
        uv = camera_rays[..., :2] * 0.15 + 0.5
        jacobian = camera_rays.new_tensor(((80.0, 0.0), (0.0, 60.0)))
        return LensProjection(
            uv=uv,
            jacobian=jacobian.expand(prefix + (2, 2)),
            lens_margin=camera_rays.new_full(prefix + (1,), 0.75),
            valid=torch.ones(prefix, dtype=torch.bool, device=camera_rays.device),
        )


def _metadata(batch=1, cameras=4):
    transform = torch.eye(4).reshape(1, 1, 4, 4).repeat(batch, cameras, 1, 1)
    return {
        "lidar2cam": transform,
        "image_wh": torch.full((batch, cameras, 2), 64.0),
    }


def _anchor(batch=1, queries=3):
    anchor = torch.tensor((10.0, 2.0, 0.5, 0.3, 1.2, 0.1, 0.6, 0.8, 1.0, -2.0, 0.2))
    return anchor.reshape(1, 1, 11).repeat(batch, queries, 1)


def _config(**overrides):
    values = dict(
        embed_dims=16,
        num_queries=3,
        num_layers=6,
        num_classes=6,
        num_heads=4,
        attention_dims=16,
        ffn_dims=32,
        num_cameras=4,
        level_strides=(4.0, 8.0),
        num_learnable_points=6,
        checkpoint_point_chunks=False,
        use_fused_daf=False,
        dropout=0.0,
    )
    values.update(overrides)
    return FisheyeBoxDecoderV4Config(**values)


def test_hybrid_state_round_trip_preserves_sparse4d_anchor():
    anchor = _anchor(2, 3)
    decoded = HybridBoxState.from_sparse_anchor(anchor).to_sparse_anchor()
    torch.testing.assert_close(decoded, anchor)


def test_center_bearing_change_cannot_rotate_global_yaw_or_velocity():
    state = HybridBoxState.from_sparse_anchor(_anchor())
    changed = HybridBoxState(
        direction=torch.nn.functional.normalize(
            torch.tensor((1.0, -3.0, 0.4)).reshape(1, 1, 3).expand_as(state.direction),
            dim=-1,
        ),
        log_range=state.log_range,
        log_size=state.log_size,
        yaw=state.yaw,
        velocity=state.velocity,
    )
    original_anchor = state.to_sparse_anchor()
    changed_anchor = changed.to_sparse_anchor()
    torch.testing.assert_close(changed_anchor[..., 6:11], original_anchor[..., 6:11])


def test_v4_refinement_is_exact_identity_at_initialization():
    state = HybridBoxState.from_sparse_anchor(_anchor())
    refinement = HybridBoxRefinement(16)
    refined = refinement(torch.randn(1, 3, 16), state)
    torch.testing.assert_close(refined.to_sparse_anchor(), state.to_sparse_anchor())


def test_structured_face_refinement_only_changes_size_and_yaw():
    state = HybridBoxState.from_sparse_anchor(_anchor())
    refinement = HybridBoxRefinement(
        16,
        structured_face_refinement=True,
    )
    feature = torch.randn(1, 3, 16)
    signed_faces = torch.ones(1, 3, 3, 16, requires_grad=True)
    identity = refinement(feature, state, signed_face_features=signed_faces)
    torch.testing.assert_close(identity.to_sparse_anchor(), state.to_sparse_anchor())

    assert refinement.face_refinement is not None
    with torch.no_grad():
        refinement.face_refinement.weight.fill_(0.01)
    refined = refinement(feature, state, signed_face_features=signed_faces)
    torch.testing.assert_close(refined.direction, state.direction)
    torch.testing.assert_close(refined.log_range, state.log_range)
    torch.testing.assert_close(refined.velocity, state.velocity)
    assert not torch.equal(refined.log_size, state.log_size)
    assert not torch.equal(refined.yaw, state.yaw)

    refined.to_sparse_anchor().sum().backward()
    assert refinement.face_refinement.weight.grad is not None
    assert signed_faces.grad is not None


@pytest.mark.parametrize(
    ("targets", "size_changes", "yaw_changes"),
    (
        (("size",), True, False),
        (("yaw",), False, True),
        (("size", "yaw"), True, True),
    ),
)
def test_structured_face_refinement_can_isolate_geometry_target(
    targets, size_changes, yaw_changes
):
    state = HybridBoxState.from_sparse_anchor(_anchor())
    refinement = HybridBoxRefinement(
        16,
        structured_face_refinement=True,
        structured_face_refinement_targets=targets,
    )
    assert refinement.face_refinement is not None
    with torch.no_grad():
        refinement.face_refinement.weight.fill_(0.01)
    refined = refinement(
        torch.randn(1, 3, 16),
        state,
        signed_face_features=torch.ones(1, 3, 3, 16),
    )
    assert (not torch.equal(refined.log_size, state.log_size)) is size_changes
    assert (not torch.equal(refined.yaw, state.yaw)) is yaw_changes
    torch.testing.assert_close(refined.direction, state.direction)
    torch.testing.assert_close(refined.log_range, state.log_range)
    torch.testing.assert_close(refined.velocity, state.velocity)


def test_separate_face_yaw_residual_uses_its_own_learned_update():
    state = HybridBoxState.from_sparse_anchor(_anchor())
    shared_gate = HybridBoxRefinement(
        16,
        learned_update_gate=True,
        structured_face_refinement=True,
    )
    separate = HybridBoxRefinement(
        16,
        learned_update_gate=True,
        structured_face_refinement=True,
        separate_face_yaw_residual=True,
    )
    separate.load_state_dict(shared_gate.state_dict())
    assert shared_gate.face_refinement is not None
    assert separate.face_refinement is not None
    assert shared_gate.update_gate is not None
    assert separate.update_gate is not None
    with torch.no_grad():
        shared_gate.face_refinement.weight.zero_()
        separate.face_refinement.weight.zero_()
        shared_gate.face_refinement.weight[3].fill_(0.01)
        separate.face_refinement.weight[3].fill_(0.01)
        shared_gate.update_gate.weight.zero_()
        separate.update_gate.weight.zero_()
        shared_gate.update_gate.bias.fill_(-6.0)
        separate.update_gate.bias.fill_(-6.0)

    feature = torch.randn(1, 3, 16)
    signed_faces = torch.ones(1, 3, 3, 16)
    shared_state = shared_gate(
        feature, state, signed_face_features=signed_faces
    )
    separate_state = separate(
        feature, state, signed_face_features=signed_faces
    )
    shared_change = torch.linalg.vector_norm(shared_state.yaw - state.yaw, dim=-1)
    separate_change = torch.linalg.vector_norm(separate_state.yaw - state.yaw, dim=-1)
    assert torch.all(separate_change > shared_change * 100.0)
    torch.testing.assert_close(separate_state.log_size, shared_state.log_size)
    torch.testing.assert_close(separate_state.direction, shared_state.direction)
    torch.testing.assert_close(separate_state.log_range, shared_state.log_range)
    torch.testing.assert_close(separate_state.velocity, shared_state.velocity)


def test_learned_face_yaw_gate_controls_only_the_separate_face_update():
    state = HybridBoxState.from_sparse_anchor(_anchor())
    refinement = HybridBoxRefinement(
        16,
        learned_update_gate=True,
        structured_face_refinement=True,
        structured_face_refinement_targets=("yaw",),
        separate_face_yaw_residual=True,
        learned_face_yaw_gate=True,
    )
    assert refinement.face_refinement is not None
    assert refinement.face_yaw_gate is not None
    with torch.no_grad():
        refinement.face_refinement.weight.zero_()
        refinement.face_refinement.weight[3, 0] = 0.25
        refinement.face_yaw_gate.weight.zero_()
        refinement.face_yaw_gate.bias.fill_(-6.0)

    feature = torch.randn(1, 3, 16)
    signed_faces = torch.ones(1, 3, 3, 16)
    low_trust = refinement(
        feature, state, signed_face_features=signed_faces
    )
    with torch.no_grad():
        refinement.face_yaw_gate.bias.fill_(6.0)
    high_trust = refinement(
        feature, state, signed_face_features=signed_faces
    )

    low_change = torch.linalg.vector_norm(low_trust.yaw - state.yaw, dim=-1)
    high_change = torch.linalg.vector_norm(high_trust.yaw - state.yaw, dim=-1)
    assert torch.all(high_change > low_change * 100.0)
    torch.testing.assert_close(high_trust.log_size, state.log_size)
    torch.testing.assert_close(high_trust.direction, state.direction)
    torch.testing.assert_close(high_trust.log_range, state.log_range)
    torch.testing.assert_close(high_trust.velocity, state.velocity)
    high_change.sum().backward()
    assert refinement.face_yaw_gate.bias.grad is not None
    assert torch.count_nonzero(refinement.face_yaw_gate.bias.grad) > 0


def test_symmetric_face_size_evidence_drives_only_size():
    state = HybridBoxState.from_sparse_anchor(_anchor())
    refinement = HybridBoxRefinement(
        16,
        learned_update_gate=True,
        structured_face_refinement=True,
        structured_face_refinement_targets=("size",),
        separate_face_size_residual=True,
        symmetric_face_size_evidence=True,
    )
    assert refinement.face_refinement is not None
    with torch.no_grad():
        refinement.face_refinement.weight.zero_()
        refinement.face_refinement.weight[0, 0] = 0.25

    feature = torch.zeros(1, 3, 16)
    signed_faces = torch.zeros(1, 3, 3, 16)
    symmetric_faces = torch.ones(1, 3, 3, 16, requires_grad=True)
    refined = refinement(
        feature,
        state,
        signed_face_features=signed_faces,
        symmetric_face_features=symmetric_faces,
    )
    assert torch.all(refined.log_size[..., 0] > state.log_size[..., 0])
    torch.testing.assert_close(refined.log_size[..., 1:], state.log_size[..., 1:])
    torch.testing.assert_close(refined.yaw, state.yaw)
    torch.testing.assert_close(refined.direction, state.direction)
    torch.testing.assert_close(refined.log_range, state.log_range)
    torch.testing.assert_close(refined.velocity, state.velocity)
    refined.log_size.sum().backward()
    assert symmetric_faces.grad is not None
    assert torch.count_nonzero(symmetric_faces.grad) > 0


def test_separate_face_size_residual_uses_its_own_learned_update():
    state = HybridBoxState.from_sparse_anchor(_anchor())
    shared_gate = HybridBoxRefinement(
        16,
        learned_update_gate=True,
        structured_face_refinement=True,
    )
    separate = HybridBoxRefinement(
        16,
        learned_update_gate=True,
        structured_face_refinement=True,
        separate_face_size_residual=True,
    )
    separate.load_state_dict(shared_gate.state_dict())
    assert shared_gate.face_refinement is not None
    assert separate.face_refinement is not None
    assert shared_gate.update_gate is not None
    assert separate.update_gate is not None
    with torch.no_grad():
        shared_gate.face_refinement.weight.zero_()
        separate.face_refinement.weight.zero_()
        shared_gate.face_refinement.weight[:3].fill_(0.01)
        separate.face_refinement.weight[:3].fill_(0.01)
        shared_gate.update_gate.weight.zero_()
        separate.update_gate.weight.zero_()
        shared_gate.update_gate.bias.fill_(-6.0)
        separate.update_gate.bias.fill_(-6.0)

    feature = torch.randn(1, 3, 16)
    signed_faces = torch.ones(1, 3, 3, 16)
    shared_state = shared_gate(
        feature, state, signed_face_features=signed_faces
    )
    separate_state = separate(
        feature, state, signed_face_features=signed_faces
    )
    shared_change = (shared_state.log_size - state.log_size).abs()
    separate_change = (separate_state.log_size - state.log_size).abs()
    assert torch.all(separate_change > shared_change * 100.0)
    torch.testing.assert_close(separate_state.yaw, shared_state.yaw)
    torch.testing.assert_close(separate_state.direction, shared_state.direction)
    torch.testing.assert_close(separate_state.log_range, shared_state.log_range)
    torch.testing.assert_close(separate_state.velocity, shared_state.velocity)


def test_structured_face_refinement_requires_complete_face_tensor():
    state = HybridBoxState.from_sparse_anchor(_anchor())
    refinement = HybridBoxRefinement(16, structured_face_refinement=True)
    with pytest.raises(ValueError, match=r"requires \[B,Q,3,embed_dims\]"):
        refinement(torch.randn(1, 3, 16), state)


def test_learned_trust_refinement_is_identity_and_uses_state_feature():
    state = HybridBoxState.from_sparse_anchor(_anchor())
    refinement = HybridBoxRefinement(
        16,
        state_conditioned=True,
        learned_update_gate=True,
    )
    feature = torch.randn(1, 3, 16, requires_grad=True)
    state_feature = torch.randn(1, 3, 16, requires_grad=True)
    refined = refinement(feature, state, state_feature)
    torch.testing.assert_close(refined.to_sparse_anchor(), state.to_sparse_anchor())
    refined.to_sparse_anchor().sum().backward()
    assert refinement.update_gate is not None
    assert refinement.update_gate.weight.grad is not None
    assert feature.grad is not None and state_feature.grad is not None


def test_c0_additive_size_update_removes_only_size_gate_attenuation():
    state = HybridBoxState.from_sparse_anchor(_anchor())
    gated = HybridBoxRefinement(16, learned_update_gate=True)
    additive = HybridBoxRefinement(
        16,
        learned_update_gate=True,
        c0_additive_size_update=True,
    )
    additive.load_state_dict(gated.state_dict())
    with torch.no_grad():
        gated.trunk[-1].bias.copy_(
            torch.tensor((0.2, -0.1, 0.3, 0.4, -0.2, 0.6, 0.5, 0.7, -0.8, 0.9))
        )
        additive.trunk[-1].bias.copy_(gated.trunk[-1].bias)

    feature = torch.randn(1, 3, 16)
    gated_state = gated(feature, state)
    additive_state = additive(feature, state)

    torch.testing.assert_close(additive_state.direction, gated_state.direction)
    torch.testing.assert_close(additive_state.log_range, gated_state.log_range)
    torch.testing.assert_close(additive_state.yaw, gated_state.yaw)
    torch.testing.assert_close(additive_state.velocity, gated_state.velocity)
    expected_size = state.log_size + gated.trunk[-1].bias[3:6]
    torch.testing.assert_close(
        additive_state.log_size,
        expected_size,
    )
    torch.testing.assert_close(
        gated_state.log_size,
        state.log_size + 0.5 * gated.trunk[-1].bias[3:6],
    )


def test_c0_additive_size_update_requires_other_learned_gates():
    with pytest.raises(ValueError, match="requires the remaining learned update gates"):
        HybridBoxRefinement(16, c0_additive_size_update=True)
    with pytest.raises(ValueError, match="requires the remaining learned update gates"):
        _config(c0_additive_size_update=True)


def test_c0_additive_size_update_can_be_terminal_only():
    torch.manual_seed(3407)
    baseline = FisheyeSparseBoxDecoderV4(
        _config(learned_update_gate=True),
        projector=NativeFisheyeProjector(_LinearLens()),
        sampler=DeterministicBilinearSampler(),
    )
    torch.manual_seed(3407)
    terminal = FisheyeSparseBoxDecoderV4(
        _config(
            learned_update_gate=True,
            c0_additive_size_update_layers=(5,),
        ),
        projector=NativeFisheyeProjector(_LinearLens()),
        sampler=DeterministicBilinearSampler(),
    )

    assert all(
        not layer.refinement.c0_additive_size_update
        for layer in terminal.layers[:5]
    )
    assert terminal.layers[5].refinement.c0_additive_size_update
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(
            terminal.state_dict()[name], value, rtol=0.0, atol=0.0
        )


@pytest.mark.parametrize(
    ("layers", "message"),
    [
        ((), "cannot be empty"),
        ((5, 5), "must be unique"),
        ((6,), "outside D0-D5"),
    ],
)
def test_c0_additive_size_layer_selection_is_validated(layers, message):
    with pytest.raises(ValueError, match=message):
        _config(
            learned_update_gate=True,
            c0_additive_size_update_layers=layers,
        )


def test_vector_yaw_residual_matches_sparse4d_geometry_without_angle_bound():
    state = HybridBoxState.from_sparse_anchor(_anchor())
    refinement = HybridBoxRefinement(16, yaw_update_mode="vector_residual")
    assert refinement.output_dims == 11
    feature = torch.randn(1, 3, 16)
    identity = refinement(feature, state)
    torch.testing.assert_close(identity.to_sparse_anchor(), state.to_sparse_anchor())

    with torch.no_grad():
        refinement.trunk[-1].bias[6:8] = torch.tensor((0.2, -0.1))
        refinement.trunk[-1].bias[8:11] = torch.tensor((1.0, -2.0, 0.5))
    refined = refinement(feature, state)
    expected_yaw = torch.nn.functional.normalize(
        state.yaw + state.yaw.new_tensor((0.2, -0.1)), dim=-1
    )
    torch.testing.assert_close(refined.yaw, expected_yaw)
    torch.testing.assert_close(
        refined.velocity,
        state.velocity + state.velocity.new_tensor((1.0, -2.0, 0.5)),
    )


def test_v4_config_has_no_manual_update_or_pdf_parameters():
    names = {field.name for field in fields(FisheyeBoxDecoderV4Config)}
    banned = {
        "max_angular_delta_rad",
        "max_log_range_delta",
        "max_log_size_delta",
        "max_yaw_delta_rad",
        "max_velocity_delta",
        "min_range_m",
        "max_range_m",
        "min_range_std_m",
        "max_range_std_m",
    }
    assert names.isdisjoint(banned)


def test_v4_uncertainty_is_optional_auxiliary_readout_not_box_state():
    decoder = FisheyeSparseBoxDecoderV4(
        _config(predict_range_uncertainty=True),
        projector=NativeFisheyeProjector(_LinearLens()),
        sampler=DeterministicBilinearSampler(),
    )
    fpn = [torch.randn(1, 4, 16, size, size) for size in (8, 4)]
    output = decoder(fpn, _metadata(), torch.randn(1, 3, 16), _anchor())

    assert output.raw_log_range_std is not None
    assert len(output.raw_log_range_std) == 6
    assert all(value.shape == (1, 3, 1) for value in output.raw_log_range_std)
    assert all(torch.count_nonzero(value) == 0 for value in output.raw_log_range_std)
    assert not hasattr(output.states[0], "log_range_std")
    assert HybridBoxRefinement.output_dims == 10


def test_state_conditioned_uncertainty_is_detached_from_detector_features():
    decoder = FisheyeSparseBoxDecoderV4(
        _config(
            predict_range_uncertainty=True,
            detach_range_uncertainty_input=True,
            state_conditioned_uncertainty=True,
        ),
        projector=NativeFisheyeProjector(_LinearLens()),
        sampler=DeterministicBilinearSampler(),
    )
    fpn = [torch.randn(1, 4, 16, size, size) for size in (8, 4)]
    instance_feature = torch.randn(1, 3, 16, requires_grad=True)
    output = decoder(fpn, _metadata(), instance_feature, _anchor())

    assert output.raw_log_range_std is not None
    sum(value.sum() for value in output.raw_log_range_std).backward()
    assert instance_feature.grad is None
    assert all(
        head.weight.grad is not None for head in decoder.range_uncertainty_heads
    )
    assert all(
        parameter.grad is None
        for name, parameter in decoder.named_parameters()
        if not name.startswith("range_uncertainty_heads.")
    )


def test_state_conditioned_uncertainty_requires_detached_readout():
    with pytest.raises(ValueError, match="requires a detached uncertainty"):
        _config(
            predict_range_uncertainty=True,
            state_conditioned_uncertainty=True,
        )


def test_shared_and_targeted_face_paths_are_mutually_exclusive():
    with pytest.raises(ValueError, match="either shared contrast or targeted"):
        _config(
            structured_face_contrast=True,
            structured_face_refinement=True,
        )


def test_hybrid_ray_supervision_has_three_finite_losses_and_gradients():
    direction = torch.tensor([[[1.0, 0.1, 0.0], [0.0, 1.0, 0.0]]], requires_grad=True)
    log_range = torch.tensor([[[2.0], [3.0]]], requires_grad=True)
    raw_std = torch.zeros(1, 2, 1, requires_grad=True)
    target = torch.tensor([[[8.0, 1.0, 0.0, 0.0], [0.0, 20.0, 0.0, 0.0]]])
    matched = torch.tensor([[True, False]])
    loss_module = FisheyeHybridRayAuxiliaryLoss(RayAuxiliaryLossConfig())

    losses = loss_module(direction, log_range, raw_std, target, matched)
    assert set(losses) == {
        "loss_ray_angular",
        "loss_ray_log_range",
        "loss_ray_range_nll",
    }
    assert all(torch.isfinite(value) for value in losses.values())
    sum(losses.values()).backward()
    assert direction.grad is not None and torch.isfinite(direction.grad).all()
    assert log_range.grad is not None and torch.isfinite(log_range.grad).all()
    assert raw_std.grad is not None and torch.isfinite(raw_std.grad).all()


def test_v4_keeps_thirteen_points_without_a_pre_d0_initializer():
    decoder = FisheyeSparseBoxDecoderV4(
        _config(),
        projector=NativeFisheyeProjector(_LinearLens()),
        sampler=DeterministicBilinearSampler(),
    )
    assert not hasattr(decoder, "initializer")
    assert len(decoder.layers) == 6
    assert all(
        isinstance(layer.keypoints, FisheyeBoxKeypointGenerator)
        for layer in decoder.layers
    )
    assert all(layer.config.num_points == 13 for layer in decoder.layers)


def test_pre_d0_feature_only_observes_image_without_changing_box_state():
    config = _config(pre_d0_query_evidence=True)
    decoder = FisheyeSparseBoxDecoderV4(
        config,
        projector=NativeFisheyeProjector(_LinearLens()),
        sampler=DeterministicBilinearSampler(),
    )
    assert isinstance(decoder.initializer, FisheyeCenterQueryInitializer)
    fpn = [
        torch.randn(1, 4, 16, size, size, requires_grad=True)
        for size in (8, 4)
    ]
    query = torch.randn(1, 3, 16, requires_grad=True)
    state = HybridBoxState.from_sparse_anchor(_anchor())
    before = state.to_sparse_anchor().clone()

    initialized_query = decoder.initializer(query, state, fpn, _metadata())

    torch.testing.assert_close(state.to_sparse_anchor(), before)
    assert initialized_query.shape == query.shape
    initialized_query.square().mean().backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert all(feature.grad is not None for feature in fpn)


def test_state_conditioned_task_heads_receive_query_plus_current_box_embedding():
    decoder = FisheyeSparseBoxDecoderV4(
        _config(state_conditioned_task_heads=True),
        projector=NativeFisheyeProjector(_LinearLens()),
        sampler=DeterministicBilinearSampler(),
    )
    observed = {}

    def save_output(name):
        def hook(_module, _inputs, output):
            observed[name] = output.detach()

        return hook

    def save_input(name):
        def hook(_module, inputs):
            observed[name] = inputs[0].detach()

        return hook

    handles = [
        decoder.layers[0].position_encoder.register_forward_hook(
            save_output("position")
        ),
        decoder.layers[0].ffn_norm.register_forward_hook(save_output("query")),
        decoder.classifiers[0].register_forward_pre_hook(save_input("task")),
        decoder.quality_heads[0].register_forward_pre_hook(save_input("quality")),
    ]
    try:
        fpn = [torch.randn(1, 4, 16, size, size) for size in (8, 4)]
        decoder(fpn, _metadata(), torch.randn(1, 3, 16), _anchor())
    finally:
        for handle in handles:
            handle.remove()

    expected = observed["query"] + observed["position"]
    torch.testing.assert_close(observed["task"], expected)
    torch.testing.assert_close(observed["quality"], expected)


def test_d0_center_prefusion_reuses_only_the_first_thirteen_point_read():
    decoder = FisheyeSparseBoxDecoderV4(
        _config(d0_center_prefusion=True),
        projector=NativeFisheyeProjector(_LinearLens()),
        sampler=DeterministicBilinearSampler(),
    )
    assert not hasattr(decoder, "initializer")
    assert decoder.layers[0].evidence.center_prefusion
    assert hasattr(decoder.layers[0].evidence, "center_residual_proj")
    assert all(
        not layer.evidence.center_prefusion for layer in decoder.layers[1:]
    )

    fpn = [
        torch.randn(1, 4, 16, size, size, requires_grad=True)
        for size in (8, 4)
    ]
    output = decoder(fpn, _metadata(), torch.randn(1, 3, 16), _anchor())
    output.classification[-1].square().mean().backward()
    assert decoder.layers[0].evidence.center_residual_proj.weight.grad is not None


def test_sparse4d_task_heads_restore_two_hidden_blocks_and_output_shapes():
    decoder = FisheyeSparseBoxDecoderV4(
        _config(
            state_conditioned_task_heads=True,
            sparse4d_task_heads=True,
        ),
        projector=NativeFisheyeProjector(_LinearLens()),
        sampler=DeterministicBilinearSampler(),
    )
    classifier = decoder.classifiers[0]
    quality = decoder.quality_heads[0]
    assert isinstance(classifier, torch.nn.Sequential)
    assert isinstance(quality, torch.nn.Sequential)
    assert [type(module) for module in classifier] == [
        torch.nn.Linear,
        torch.nn.ReLU,
        torch.nn.LayerNorm,
        torch.nn.Linear,
        torch.nn.ReLU,
        torch.nn.LayerNorm,
        torch.nn.Linear,
    ]
    expected_bias = torch.full_like(classifier[-1].bias, -4.59511985013459)
    torch.testing.assert_close(classifier[-1].bias, expected_bias)

    fpn = [torch.randn(1, 4, 16, size, size) for size in (8, 4)]
    output = decoder(fpn, _metadata(), torch.randn(1, 3, 16), _anchor())
    assert output.classification[-1].shape == (1, 3, 6)
    assert output.quality[-1].shape == (1, 3, 2)


def test_sparse4d_task_routing_separates_classification_and_quality_inputs():
    decoder = FisheyeSparseBoxDecoderV4(
        _config(
            state_conditioned_task_heads=True,
            sparse4d_task_heads=True,
            sparse4d_task_routing=True,
        ),
        projector=NativeFisheyeProjector(_LinearLens()),
        sampler=DeterministicBilinearSampler(),
    )
    layer_output = []
    classifier_input = []
    quality_input = []
    handles = (
        decoder.layers[0].register_forward_hook(
            lambda _module, _inputs, output: layer_output.append(output)
        ),
        decoder.classifiers[0].register_forward_pre_hook(
            lambda _module, inputs: classifier_input.append(inputs[0])
        ),
        decoder.quality_heads[0].register_forward_pre_hook(
            lambda _module, inputs: quality_input.append(inputs[0])
        ),
    )
    try:
        fpn = [torch.randn(1, 4, 16, size, size) for size in (8, 4)]
        decoder(fpn, _metadata(), torch.randn(1, 3, 16), _anchor())
    finally:
        for handle in handles:
            handle.remove()

    updated_query = layer_output[0][0]
    state_conditioned_feature = layer_output[0][3]
    torch.testing.assert_close(classifier_input[0], updated_query)
    torch.testing.assert_close(quality_input[0], state_conditioned_feature)
    assert not torch.equal(classifier_input[0], quality_input[0])


def test_sparse4d_refinement_trunk_restores_four_hidden_blocks():
    decoder = FisheyeSparseBoxDecoderV4(
        _config(sparse4d_refinement_trunk=True),
        projector=NativeFisheyeProjector(_LinearLens()),
        sampler=DeterministicBilinearSampler(),
    )
    trunk = decoder.layers[0].refinement.trunk
    assert [type(module) for module in trunk] == [
        torch.nn.Linear,
        torch.nn.ReLU,
        torch.nn.Linear,
        torch.nn.ReLU,
        torch.nn.LayerNorm,
        torch.nn.Linear,
        torch.nn.ReLU,
        torch.nn.Linear,
        torch.nn.ReLU,
        torch.nn.LayerNorm,
        torch.nn.Linear,
    ]
    assert torch.count_nonzero(trunk[-1].weight) == 0
    assert torch.count_nonzero(trunk[-1].bias) == 0

    fpn = [torch.randn(1, 4, 16, size, size) for size in (8, 4)]
    output = decoder(fpn, _metadata(), torch.randn(1, 3, 16), _anchor())
    assert output.prediction[-1].shape == (1, 3, 10)


def test_dedicated_size_head_isolates_extent_with_small_identity_branch():
    torch.manual_seed(3407)
    baseline = HybridBoxRefinement(
        16,
        learned_update_gate=True,
    )
    torch.manual_seed(3407)
    dedicated = HybridBoxRefinement(
        16,
        learned_update_gate=True,
        dedicated_size_head=True,
    )
    assert dedicated.size_trunk is not None
    assert [type(module) for module in dedicated.size_trunk] == [
        torch.nn.Linear,
        torch.nn.ReLU,
        torch.nn.LayerNorm,
        torch.nn.Linear,
    ]
    dedicated_state = dedicated.state_dict()
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(dedicated_state[name], value, rtol=0.0, atol=0.0)

    state = HybridBoxState.from_sparse_anchor(_anchor())
    feature = torch.randn(1, 3, 16)
    baseline_identity = baseline(feature, state)
    dedicated_identity = dedicated(feature, state)
    torch.testing.assert_close(
        dedicated_identity.to_sparse_anchor(),
        baseline_identity.to_sparse_anchor(),
        rtol=0.0,
        atol=0.0,
    )

    with torch.no_grad():
        dedicated.size_trunk[-1].bias.copy_(torch.tensor((0.1, -0.2, 0.3)))
    refined = dedicated(feature, state)
    torch.testing.assert_close(
        refined.log_size,
        state.log_size + 0.5 * state.log_size.new_tensor((0.1, -0.2, 0.3)),
    )
    torch.testing.assert_close(refined.direction, state.direction)
    torch.testing.assert_close(refined.log_range, state.log_range)
    torch.testing.assert_close(refined.yaw, state.yaw)
    torch.testing.assert_close(refined.velocity, state.velocity)
    refined.log_size.sum().backward()
    assert dedicated.size_trunk[-1].weight.grad is not None
    assert torch.count_nonzero(dedicated.size_trunk[-1].weight.grad) > 0


def test_dedicated_size_gate_uses_extent_representation_and_preserves_reset():
    torch.manual_seed(3407)
    baseline = HybridBoxRefinement(
        16,
        learned_update_gate=True,
        dedicated_size_head=True,
    )
    torch.manual_seed(3407)
    dedicated_gate = HybridBoxRefinement(
        16,
        learned_update_gate=True,
        dedicated_size_head=True,
        dedicated_size_gate=True,
    )
    assert dedicated_gate.size_gate is not None
    dedicated_state = dedicated_gate.state_dict()
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(dedicated_state[name], value, rtol=0.0, atol=0.0)
    assert torch.count_nonzero(dedicated_gate.size_gate.weight) == 0
    assert torch.count_nonzero(dedicated_gate.size_gate.bias) == 0

    state = HybridBoxState.from_sparse_anchor(_anchor())
    feature = torch.randn(1, 3, 16)
    torch.testing.assert_close(
        dedicated_gate(feature, state).to_sparse_anchor(),
        baseline(feature, state).to_sparse_anchor(),
        rtol=0.0,
        atol=0.0,
    )

    with torch.no_grad():
        dedicated_gate.size_trunk[-1].bias.copy_(
            torch.tensor((0.1, -0.2, 0.3))
        )
    refined = dedicated_gate(feature, state)
    torch.testing.assert_close(
        refined.log_size,
        state.log_size + 0.5 * state.log_size.new_tensor((0.1, -0.2, 0.3)),
    )
    refined.log_size.sum().backward()
    assert dedicated_gate.size_gate.weight.grad is not None
    assert torch.count_nonzero(dedicated_gate.size_gate.weight.grad) > 0


def test_dedicated_size_gate_can_be_terminal_without_changing_earlier_layers():
    torch.manual_seed(3407)
    baseline = FisheyeSparseBoxDecoderV4(
        _config(
            learned_update_gate=True,
            dedicated_size_head=True,
        ),
        projector=NativeFisheyeProjector(_LinearLens()),
        sampler=DeterministicBilinearSampler(),
    )
    torch.manual_seed(3407)
    terminal = FisheyeSparseBoxDecoderV4(
        _config(
            learned_update_gate=True,
            dedicated_size_head=True,
            dedicated_size_gate=True,
            dedicated_size_gate_layers=(5,),
        ),
        projector=NativeFisheyeProjector(_LinearLens()),
        sampler=DeterministicBilinearSampler(),
    )
    assert all(layer.refinement.size_gate is None for layer in terminal.layers[:5])
    assert terminal.layers[5].refinement.size_gate is not None
    terminal_state = terminal.state_dict()
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(terminal_state[name], value, rtol=0.0, atol=0.0)


def test_sparse4d_capacity_size_head_can_be_terminal_and_identity_preserving():
    torch.manual_seed(3407)
    baseline = FisheyeSparseBoxDecoderV4(
        _config(
            learned_update_gate=True,
            dedicated_size_head=True,
        ),
        projector=NativeFisheyeProjector(_LinearLens()),
        sampler=DeterministicBilinearSampler(),
    )
    torch.manual_seed(3407)
    terminal = FisheyeSparseBoxDecoderV4(
        _config(
            learned_update_gate=True,
            dedicated_size_head=True,
            sparse4d_size_head_layers=(5,),
        ),
        projector=NativeFisheyeProjector(_LinearLens()),
        sampler=DeterministicBilinearSampler(),
    )
    compact_types = [
        torch.nn.Linear,
        torch.nn.ReLU,
        torch.nn.LayerNorm,
        torch.nn.Linear,
    ]
    for layer in terminal.layers[:5]:
        assert [type(module) for module in layer.refinement.size_trunk] == (
            compact_types
        )
    terminal_size = terminal.layers[5].refinement.size_trunk
    assert [type(module) for module in terminal_size] == [
        torch.nn.Linear,
        torch.nn.ReLU,
        torch.nn.Linear,
        torch.nn.ReLU,
        torch.nn.LayerNorm,
        torch.nn.Linear,
        torch.nn.ReLU,
        torch.nn.Linear,
        torch.nn.ReLU,
        torch.nn.LayerNorm,
        torch.nn.Linear,
    ]
    assert torch.count_nonzero(terminal_size[-1].weight) == 0
    assert torch.count_nonzero(terminal_size[-1].bias) == 0

    terminal_state = terminal.state_dict()
    for name, value in baseline.state_dict().items():
        if name.startswith("layers.5.refinement.size_trunk."):
            continue
        torch.testing.assert_close(terminal_state[name], value, rtol=0.0, atol=0.0)

    fpn = [torch.randn(1, 4, 16, size, size) for size in (8, 4)]
    query = torch.randn(1, 3, 16)
    baseline_output = baseline(fpn, _metadata(), query, _anchor())
    terminal_output = terminal(fpn, _metadata(), query, _anchor())
    for expected, observed in zip(
        baseline_output.prediction,
        terminal_output.prediction,
    ):
        torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)

    terminal_output.prediction[-1][..., 3:6].sum().backward()
    assert terminal_size[-1].weight.grad is not None
    assert torch.count_nonzero(terminal_size[-1].weight.grad) > 0


def test_structured_face_contrast_is_identity_at_reset_and_learns_signed_faces():
    baseline = FisheyeSparseBoxDecoderV4(
        _config(structured_face_contrast=False),
        projector=NativeFisheyeProjector(_LinearLens()),
        sampler=DeterministicBilinearSampler(),
    )
    structured = FisheyeSparseBoxDecoderV4(
        _config(structured_face_contrast=True),
        projector=NativeFisheyeProjector(_LinearLens()),
        sampler=DeterministicBilinearSampler(),
    )
    missing, unexpected = structured.load_state_dict(
        baseline.state_dict(), strict=False
    )
    assert not unexpected
    assert missing == [
        f"layers.{index}.evidence.face_contrast_weight"
        for index in range(6)
    ]

    baseline_evidence = baseline.layers[0].evidence
    structured_evidence = structured.layers[0].evidence
    query = torch.randn(1, 3, 16)
    point_features = torch.randn(1, 3, 13, 16)
    point_valid = torch.ones(1, 3, 13, dtype=torch.bool)
    expected, expected_attention, expected_faces = baseline_evidence._fuse_points(
        query, point_features, point_valid
    )
    observed, observed_attention, observed_faces = structured_evidence._fuse_points(
        query, point_features, point_valid
    )
    torch.testing.assert_close(observed, expected)
    torch.testing.assert_close(observed_attention, expected_attention)
    assert expected_faces is None
    assert observed_faces is not None

    observed.square().mean().backward()
    weight = structured_evidence.face_contrast_weight
    assert weight is not None
    assert weight.grad is not None
    assert torch.count_nonzero(weight.grad) > 0


def test_structured_face_contrast_ignores_unsupported_endpoint_pairs():
    decoder = FisheyeSparseBoxDecoderV4(
        _config(structured_face_contrast=True),
        projector=NativeFisheyeProjector(_LinearLens()),
        sampler=DeterministicBilinearSampler(),
    )
    evidence = decoder.layers[0].evidence
    assert evidence.face_contrast_weight is not None
    with torch.no_grad():
        evidence.face_contrast_weight.fill_(0.25)

    query = torch.randn(1, 3, 16)
    point_features = torch.randn(1, 3, 13, 16)
    point_valid = torch.ones(1, 3, 13, dtype=torch.bool)
    point_valid[:, :, 1:7] = False
    structured_output, attention, signed_faces = evidence._fuse_points(
        query, point_features, point_valid
    )
    assert signed_faces is not None
    assert torch.count_nonzero(signed_faces) == 0

    with torch.no_grad():
        evidence.face_contrast_weight.zero_()
    baseline_output, baseline_attention, _ = evidence._fuse_points(
        query, point_features, point_valid
    )
    torch.testing.assert_close(structured_output, baseline_output)
    torch.testing.assert_close(attention, baseline_attention)


def test_symmetric_face_features_are_endpoint_order_invariant():
    decoder = FisheyeSparseBoxDecoderV4(
        _config(
            structured_face_refinement=True,
            symmetric_face_size_evidence=True,
        ),
        projector=NativeFisheyeProjector(_LinearLens()),
        sampler=DeterministicBilinearSampler(),
    )
    evidence = decoder.layers[0].evidence
    assert evidence.return_symmetric_faces
    point_features = torch.randn(1, 3, 13, 16)
    point_valid = torch.ones(1, 3, 13, dtype=torch.bool)
    observed = evidence._symmetric_face_features(point_features, point_valid)
    assert observed is not None

    normalized = evidence.point_norm(point_features)
    expected = 0.5 * (
        normalized[:, :, (1, 3, 5)]
        + normalized[:, :, (2, 4, 6)]
    ) - normalized[:, :, 0:1]
    torch.testing.assert_close(observed, expected)

    swapped = point_features.clone()
    swapped[:, :, (1, 3, 5)] = point_features[:, :, (2, 4, 6)]
    swapped[:, :, (2, 4, 6)] = point_features[:, :, (1, 3, 5)]
    swapped_observed = evidence._symmetric_face_features(swapped, point_valid)
    torch.testing.assert_close(swapped_observed, observed)

    unsupported = point_valid.clone()
    unsupported[:, :, 0] = False
    unsupported_observed = evidence._symmetric_face_features(
        point_features, unsupported
    )
    assert unsupported_observed is not None
    assert torch.count_nonzero(unsupported_observed) == 0


def test_targeted_face_path_is_decoder_identity_at_reset_and_receives_gradients():
    baseline = FisheyeSparseBoxDecoderV4(
        _config(),
        projector=NativeFisheyeProjector(_LinearLens()),
        sampler=DeterministicBilinearSampler(),
    )
    targeted = FisheyeSparseBoxDecoderV4(
        _config(structured_face_refinement=True),
        projector=NativeFisheyeProjector(_LinearLens()),
        sampler=DeterministicBilinearSampler(),
    )
    missing, unexpected = targeted.load_state_dict(
        baseline.state_dict(), strict=False
    )
    assert not unexpected
    assert missing == [
        name
        for index in range(6)
        for name in (f"layers.{index}.refinement.face_refinement.weight",)
    ]

    fpn = [torch.randn(1, 4, 16, size, size) for size in (8, 4)]
    query = torch.randn(1, 3, 16)
    expected = baseline(fpn, _metadata(), query, _anchor())
    observed = targeted(fpn, _metadata(), query, _anchor())
    for expected_box, observed_box in zip(
        expected.prediction, observed.prediction
    ):
        torch.testing.assert_close(observed_box, expected_box)
    assert all(layer.evidence.return_signed_faces for layer in targeted.layers)

    sum(value.square().mean() for value in observed.prediction).backward()
    assert all(
        layer.refinement.face_refinement.weight.grad is not None
        for layer in targeted.layers
    )


def test_targeted_face_path_preserves_same_seed_baseline_initialization():
    torch.manual_seed(3407)
    baseline = FisheyeSparseBoxDecoderV4(
        _config(),
        projector=NativeFisheyeProjector(_LinearLens()),
        sampler=DeterministicBilinearSampler(),
    )
    torch.manual_seed(3407)
    targeted = FisheyeSparseBoxDecoderV4(
        _config(structured_face_refinement=True),
        projector=NativeFisheyeProjector(_LinearLens()),
        sampler=DeterministicBilinearSampler(),
    )

    baseline_state = baseline.state_dict()
    targeted_state = targeted.state_dict()
    for name, value in baseline_state.items():
        torch.testing.assert_close(targeted_state[name], value, rtol=0.0, atol=0.0)
    face_parameters = {
        name: value
        for name, value in targeted_state.items()
        if ".face_refinement." in name
    }
    assert len(face_parameters) == 6
    assert all(torch.count_nonzero(value) == 0 for value in face_parameters.values())


def test_face_yaw_gate_preserves_same_seed_parent_initialization():
    parent_config = _config(
        structured_face_refinement=True,
        structured_face_refinement_targets=("yaw",),
        separate_face_yaw_residual=True,
    )
    gated_config = _config(
        structured_face_refinement=True,
        structured_face_refinement_targets=("yaw",),
        separate_face_yaw_residual=True,
        learned_face_yaw_gate=True,
    )
    torch.manual_seed(3407)
    parent = FisheyeSparseBoxDecoderV4(
        parent_config,
        projector=NativeFisheyeProjector(_LinearLens()),
        sampler=DeterministicBilinearSampler(),
    )
    torch.manual_seed(3407)
    gated = FisheyeSparseBoxDecoderV4(
        gated_config,
        projector=NativeFisheyeProjector(_LinearLens()),
        sampler=DeterministicBilinearSampler(),
    )
    parent_state = parent.state_dict()
    gated_state = gated.state_dict()
    for name, value in parent_state.items():
        torch.testing.assert_close(gated_state[name], value, rtol=0.0, atol=0.0)
    gate_parameters = {
        name: value for name, value in gated_state.items() if ".face_yaw_gate." in name
    }
    assert len(gate_parameters) == 12
    assert all(torch.count_nonzero(value) == 0 for value in gate_parameters.values())


def test_targeted_face_path_can_be_restricted_to_final_stage():
    torch.manual_seed(3407)
    baseline = FisheyeSparseBoxDecoderV4(
        _config(),
        projector=NativeFisheyeProjector(_LinearLens()),
        sampler=DeterministicBilinearSampler(),
    )
    torch.manual_seed(3407)
    final_stage = FisheyeSparseBoxDecoderV4(
        _config(
            structured_face_refinement=True,
            structured_face_refinement_layers=(5,),
        ),
        projector=NativeFisheyeProjector(_LinearLens()),
        sampler=DeterministicBilinearSampler(),
    )
    baseline_state = baseline.state_dict()
    final_stage_state = final_stage.state_dict()
    for name, value in baseline_state.items():
        torch.testing.assert_close(
            final_stage_state[name], value, rtol=0.0, atol=0.0
        )
    missing, unexpected = final_stage.load_state_dict(
        baseline_state, strict=False
    )
    assert not unexpected
    assert missing == ["layers.5.refinement.face_refinement.weight"]
    assert [layer.evidence.return_signed_faces for layer in final_stage.layers] == [
        False,
        False,
        False,
        False,
        False,
        True,
    ]
    assert [
        layer.refinement.face_refinement is not None
        for layer in final_stage.layers
    ] == [False, False, False, False, False, True]

    fpn = [torch.randn(1, 4, 16, size, size) for size in (8, 4)]
    query = torch.randn(1, 3, 16)
    expected = baseline(fpn, _metadata(), query, _anchor())
    observed = final_stage(fpn, _metadata(), query, _anchor())
    for expected_box, observed_box in zip(
        expected.prediction, observed.prediction
    ):
        torch.testing.assert_close(observed_box, expected_box)

    sum(value.square().mean() for value in observed.prediction).backward()
    assert final_stage.layers[5].refinement.face_refinement.weight.grad is not None


@pytest.mark.parametrize(
    "flag",
    ("separate_face_size_residual", "separate_face_yaw_residual"),
)
def test_separate_face_residual_is_restricted_with_final_stage(flag):
    decoder = FisheyeSparseBoxDecoderV4(
        _config(
            structured_face_refinement=True,
            structured_face_refinement_layers=(5,),
            **{flag: True},
        ),
        projector=NativeFisheyeProjector(_LinearLens()),
        sampler=DeterministicBilinearSampler(),
    )
    assert [
        getattr(layer.refinement, flag)
        for layer in decoder.layers
    ] == [False, False, False, False, False, True]


@pytest.mark.parametrize(
    "overrides",
    (
        dict(structured_face_refinement_layers=(5,)),
        dict(
            structured_face_refinement=True,
            structured_face_refinement_layers=(),
        ),
        dict(
            structured_face_refinement=True,
            structured_face_refinement_layers=(5, 5),
        ),
        dict(
            structured_face_refinement=True,
            structured_face_refinement_layers=(6,),
        ),
        dict(structured_face_refinement_targets=()),
        dict(structured_face_refinement_targets=("size", "size")),
        dict(structured_face_refinement_targets=("center",)),
        dict(separate_face_size_residual=True),
        dict(
            structured_face_refinement=True,
            structured_face_refinement_targets=("yaw",),
            separate_face_size_residual=True,
        ),
        dict(separate_face_yaw_residual=True),
        dict(
            structured_face_refinement=True,
            structured_face_refinement_targets=("size",),
            separate_face_yaw_residual=True,
        ),
        dict(learned_face_yaw_gate=True),
        dict(
            structured_face_refinement=True,
            structured_face_refinement_targets=("yaw",),
            learned_face_yaw_gate=True,
        ),
        dict(symmetric_face_size_evidence=True),
        dict(
            structured_face_refinement=True,
            structured_face_refinement_targets=("yaw",),
            symmetric_face_size_evidence=True,
        ),
    ),
)
def test_targeted_face_stage_selection_rejects_invalid_configs(overrides):
    with pytest.raises(ValueError):
        _config(**overrides)


def test_complete_v4_decoder_propagates_six_stages_and_gradients():
    decoder = FisheyeSparseBoxDecoderV4(
        _config(),
        projector=NativeFisheyeProjector(_LinearLens()),
        sampler=DeterministicBilinearSampler(),
    )
    fpn = [torch.randn(1, 4, 16, size, size, requires_grad=True) for size in (8, 4)]
    query = torch.randn(1, 3, 16, requires_grad=True)
    output = decoder(fpn, _metadata(), query, _anchor())

    assert len(output.prediction) == len(output.states) == 6
    assert all(value.shape == (1, 3, 10) for value in output.prediction)
    assert all(value.shape == (1, 3, 13, 4) for value in output.point_attention)
    loss = sum(value.square().mean() for value in output.prediction)
    loss = loss + sum(value.square().mean() for value in output.classification)
    loss.backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert all(feature.grad is not None for feature in fpn)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in decoder.parameters()
    )
