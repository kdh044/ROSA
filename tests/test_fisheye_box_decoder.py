import torch

from rosa.fisheye_box_decoder import (
    FisheyeBoxDecoderConfig,
    FisheyeBoxEvidenceAttention,
    FisheyeBoxKeypointGenerator,
    FisheyeSparseBoxDecoder,
    project_box_keypoints,
    sample_box_keypoints_fpn,
)
from rosa.fisheye_evidence import FisheyeObservationGeometry
from rosa.fisheye_projection import (
    FisheyeProjectionModel,
    LensProjection,
    NativeFisheyeProjector,
)
from rosa.ray_state import RayState
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
            valid=torch.ones(
                prefix, dtype=torch.bool, device=camera_rays.device
            ),
        )


def _metadata(batch=1, cameras=4):
    transform = torch.eye(4).reshape(1, 1, 4, 4).repeat(batch, cameras, 1, 1)
    for camera in range(cameras):
        transform[:, camera, 1, 3] = float(camera) * 0.1
    return {
        "lidar2cam": transform,
        "image_wh": torch.full((batch, cameras, 2), 64.0),
    }


def _anchor(batch=1, queries=3):
    anchor = torch.tensor(
        (10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    )
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
        dropout=0.0,
    )
    values.update(overrides)
    return FisheyeBoxDecoderConfig(**values)


def _point_geometry(batch=1, queries=3, points=13, cameras=4):
    rays = torch.randn(batch, queries, points, cameras, 3)
    rays = torch.nn.functional.normalize(rays, dim=-1)
    tangent = torch.randn(batch, queries, points, cameras, 2) * 0.1
    jacobian = torch.eye(2).reshape(1, 1, 1, 1, 2, 2)
    jacobian = jacobian.repeat(batch, queries, points, cameras, 1, 1) * 80.0
    return FisheyeObservationGeometry(
        rays=rays,
        tangent=tangent,
        jacobian=jacobian,
        lens_margin=torch.full((batch, queries, points, cameras, 1), 0.75),
        overlap=torch.full((batch, queries, points, cameras, 1), 0.5),
        valid=torch.ones(batch, queries, points, cameras, dtype=torch.bool),
    )


def test_keypoint_generator_has_seven_semantic_and_six_learned_points():
    query = torch.randn(2, 3, 16, requires_grad=True)
    state = RayState.from_sparse_anchor(_anchor(2, 3))
    generator = FisheyeBoxKeypointGenerator(16, 6, fixed_scale=0.45)
    points = generator(query, state)

    assert points.shape == (2, 3, 13, 3)
    center = state.center
    torch.testing.assert_close(points[..., 0, :], center)
    expected_offsets = torch.tensor(
        (
            (0.45, 0.0, 0.0),
            (-0.45, 0.0, 0.0),
            (0.0, 0.45, 0.0),
            (0.0, -0.45, 0.0),
            (0.0, 0.0, 0.45),
            (0.0, 0.0, -0.45),
        )
    )
    torch.testing.assert_close(points[0, 0, 1:7] - center[0, 0], expected_offsets)
    learned_local_bound = (points[..., 7:, :] - center[..., None, :]).abs()
    assert bool((learned_local_bound <= 0.5 + 1e-6).all())
    points.square().mean().backward()
    assert generator.learnable_fc.weight.grad is not None


def test_keypoint_generator_supports_fixed_seven_point_ablation():
    generator = FisheyeBoxKeypointGenerator(16, 0)
    points = generator(torch.randn(1, 3, 16), RayState.from_sparse_anchor(_anchor()))
    assert points.shape == (1, 3, 7, 3)
    assert generator.learnable_fc is None


def test_box_projection_and_multiscale_sampling_keep_point_dimension():
    projector = NativeFisheyeProjector(_LinearLens())
    state = RayState.from_sparse_anchor(_anchor())
    points = FisheyeBoxKeypointGenerator(16, 6)(torch.randn(1, 3, 16), state)
    projection = project_box_keypoints(projector, points, _metadata())
    assert projection.uv.shape == (1, 3, 13, 4, 2)
    assert projection.geometry.rays.shape == (1, 3, 13, 4, 3)

    fpn = [
        torch.randn(1, 4, 16, size, size, requires_grad=True)
        for size in (8, 4)
    ]
    sampled = sample_box_keypoints_fpn(
        fpn,
        projection.uv,
        sampler=DeterministicBilinearSampler(),
    )
    assert sampled.shape == (1, 3, 13, 4, 2, 16)
    sampled.square().mean().backward()
    assert all(feature.grad is not None for feature in fpn)


def test_factorized_attention_normalizes_camera_level_then_points():
    config = _config()
    module = FisheyeBoxEvidenceAttention(config)
    query = torch.randn(1, 3, 16, requires_grad=True)
    sampled = torch.randn(1, 3, 13, 4, 2, 16, requires_grad=True)
    geometry = _point_geometry()
    output = module(
        query,
        sampled,
        geometry,
        torch.full((1, 3, 2), -2.0),
    )
    assert output.camera_level_attention.shape == (1, 3, 13, 4, 2, 4)
    assert output.point_attention.shape == (1, 3, 13, 4)
    torch.testing.assert_close(
        output.camera_level_attention.sum(dim=(3, 4)),
        torch.ones(1, 3, 13, 4),
    )
    torch.testing.assert_close(
        output.point_attention.sum(dim=2),
        torch.ones(1, 3, 4),
    )
    output.query.square().mean().backward()
    assert query.grad is not None and sampled.grad is not None


def test_initializer_is_exact_anchor_identity_at_reset():
    config = _config()
    decoder = FisheyeSparseBoxDecoder(
        config,
        projector=NativeFisheyeProjector(_LinearLens()),
        sampler=DeterministicBilinearSampler(),
    )
    state = RayState.from_sparse_anchor(_anchor())
    fpn = [torch.randn(1, 4, 16, size, size) for size in (8, 4)]
    output = decoder.initializer(
        torch.randn(1, 3, 16),
        state,
        fpn,
        _metadata(),
    )
    torch.testing.assert_close(output.state.to_sparse_anchor(), state.to_sparse_anchor())
    assert output.gate.shape == (1, 3, 1)
    assert bool(((output.gate >= 0.0) & (output.gate <= 1.0)).all())


def test_complete_six_stage_decoder_propagates_thirteen_point_boxes_and_gradients():
    config = _config()
    decoder = FisheyeSparseBoxDecoder(
        config,
        projector=NativeFisheyeProjector(_LinearLens()),
        sampler=DeterministicBilinearSampler(),
    )
    fpn = [
        torch.randn(1, 4, 16, size, size, requires_grad=True)
        for size in (8, 4)
    ]
    query = torch.randn(1, 3, 16, requires_grad=True)
    output = decoder(fpn, _metadata(), query, _anchor())

    assert len(output.prediction) == len(output.states) == 6
    assert all(value.shape == (1, 3, 10) for value in output.prediction)
    # The checkpointed training path intentionally does not retain the large
    # camera-by-level diagnostic carrier.  Point attention is still available
    # and all evidence gradients must propagate through recomputation.
    assert all(
        value.shape == (1, 3, 13, 0, 0, 4)
        for value in output.camera_level_attention
    )
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


def test_complete_decoder_eval_exposes_camera_level_diagnostics():
    config = _config()
    decoder = FisheyeSparseBoxDecoder(
        config,
        projector=NativeFisheyeProjector(_LinearLens()),
        sampler=DeterministicBilinearSampler(),
    ).eval()
    fpn = [torch.randn(1, 4, 16, size, size) for size in (8, 4)]

    with torch.no_grad():
        output = decoder(
            fpn,
            _metadata(),
            torch.randn(1, 3, 16),
            _anchor(),
        )

    assert all(
        value.shape == (1, 3, 13, 4, 2, 4)
        for value in output.camera_level_attention
    )
