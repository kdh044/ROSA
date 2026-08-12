import torch

from rosa.fisheye_decoder import (
    FisheyeRayDecoder,
    FisheyeRayDecoderConfig,
    SphericalRayQueryBank,
    state_log_angular_extent,
)
from rosa.fisheye_projection import (
    FisheyeProjectionModel,
    LensProjection,
    NativeFisheyeProjector,
)
from rosa.ray_state import RayState
from rosa.sampling import DeterministicBilinearSampler


class _AlwaysVisibleLens(FisheyeProjectionModel):
    def project_rays(self, camera_rays, metadata, image_wh):
        del metadata, image_wh
        # A smooth synthetic map keeps decoder tests independent of OCam.
        uv = camera_rays[..., :2] * 0.2 + 0.5
        prefix = camera_rays.shape[:-1]
        jacobian = camera_rays.new_tensor(((100.0, 0.0), (0.0, 80.0)))
        return LensProjection(
            uv=uv,
            jacobian=jacobian.expand(prefix + (2, 2)),
            lens_margin=camera_rays.new_full(prefix + (1,), 0.5),
            valid=torch.ones(prefix, dtype=torch.bool, device=camera_rays.device),
        )


class _CountingSampler(DeterministicBilinearSampler):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, feature, uv):
        self.calls += 1
        return super().forward(feature, uv)


def _metadata(batch, cameras):
    transform = torch.eye(4).reshape(1, 1, 4, 4).repeat(batch, cameras, 1, 1)
    # Give cameras distinct optical axes around the rig.
    for camera, quarter_turns in enumerate((0, 1, 2, 3)):
        angle = torch.tensor(quarter_turns * torch.pi / 2)
        cosine, sine = angle.cos(), angle.sin()
        transform[:, camera, 0, 0] = cosine
        transform[:, camera, 0, 1] = -sine
        transform[:, camera, 1, 0] = sine
        transform[:, camera, 1, 1] = cosine
    return {
        "lidar2cam": transform,
        "image_wh": torch.full((batch, cameras, 2), 64.0),
    }


def test_query_bank_discards_sparse_box_range_shape_and_yaw():
    anchors = torch.randn(7, 11)
    anchors[..., 6:8] = torch.tensor((0.0, 1.0))
    bank = SphericalRayQueryBank.from_sparse_anchors(anchors, embed_dims=16)

    query, direction = bank(2)

    assert query.shape == (2, 7, 16)
    assert direction.shape == (2, 7, 3)
    torch.testing.assert_close(direction.square().sum(-1), torch.ones(2, 7))
    # No parameter has the old 11-value box carrier.
    assert all(parameter.shape[-1] != 11 for parameter in bank.parameters())


def test_six_layer_decoder_propagates_queries_and_samples_one_ray_per_level():
    torch.manual_seed(3407)
    batch, cameras, queries, channels = 2, 4, 7, 16
    config = FisheyeRayDecoderConfig(
        embed_dims=channels,
        num_queries=queries,
        num_layers=6,
        num_classes=6,
        num_heads=4,
        ffn_dims=32,
        num_cameras=cameras,
        level_strides=(8.0, 16.0, 32.0, 64.0),
    )
    sampler = _CountingSampler()
    decoder = FisheyeRayDecoder(
        torch.randn(queries, 3),
        config,
        projector=NativeFisheyeProjector(_AlwaysVisibleLens()),
        sampler=sampler,
    )
    fpn = [
        torch.randn(batch, cameras, channels, size, size, requires_grad=True)
        for size in (16, 8, 4, 2)
    ]

    output = decoder(fpn, _metadata(batch, cameras))

    assert len(output.classification) == len(output.prediction) == 6
    assert len(output.quality) == 6
    assert len(output.states) == len(output.attention) == 6
    assert sampler.calls == 6 * 4
    assert decoder.layers[0].evidence.extent_head is not None
    assert all(layer.evidence.extent_head is None for layer in decoder.layers[1:])
    for classification, prediction, quality, attention in zip(
        output.classification, output.prediction, output.quality, output.attention
    ):
        assert classification.shape == (batch, queries, 6)
        assert prediction.shape == (batch, queries, 10)
        assert quality.shape == (batch, queries, 2)
        assert attention.shape == (batch, queries, cameras, 4, 4)
        torch.testing.assert_close(
            attention.sum(dim=(2, 3)),
            torch.ones(batch, queries, 4),
        )
    # Zero-initialized refiners propagate the same state at construction;
    # decoder query features still change through all six attention blocks.
    for state in output.states[1:]:
        torch.testing.assert_close(state.center, output.states[0].center)
    assert not torch.equal(
        output.query, decoder.query_bank.feature[None].expand_as(output.query)
    )

    loss = sum(value.square().mean() for value in output.classification)
    loss = loss + output.prediction[-1].square().mean()
    loss.backward()
    assert decoder.query_bank.feature.grad is not None
    assert torch.count_nonzero(decoder.query_bank.feature.grad) > 0
    assert decoder.query_bank.direction.grad is not None
    assert torch.count_nonzero(decoder.query_bank.direction.grad) > 0
    for layer in decoder.layers:
        assert layer.evidence.output_proj.weight.grad is not None
        assert torch.count_nonzero(layer.evidence.output_proj.weight.grad) > 0


def test_metric_size_is_converted_to_range_conditioned_angular_extent():
    anchors = torch.tensor([[[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]])
    state = RayState.from_sparse_anchor(anchors)

    extent = state_log_angular_extent(state).exp()

    expected = 2.0 * torch.atan(torch.tensor([[[0.05, 0.05]]]))
    torch.testing.assert_close(extent, expected)


def test_direction_only_training_queries_are_propagated_without_box_state():
    torch.manual_seed(17)
    batch, cameras, free_queries, dn_queries, channels = 2, 4, 5, 6, 16
    config = FisheyeRayDecoderConfig(
        embed_dims=channels,
        num_queries=free_queries,
        num_layers=2,
        num_heads=4,
        ffn_dims=32,
        num_cameras=cameras,
        level_strides=(8.0, 16.0),
    )
    decoder = FisheyeRayDecoder(
        torch.randn(free_queries, 3),
        config,
        projector=NativeFisheyeProjector(_AlwaysVisibleLens()),
    )
    fpn = [torch.randn(batch, cameras, channels, size, size) for size in (8, 4)]
    noisy_gt_directions = torch.randn(batch, dn_queries, 3)
    total = free_queries + dn_queries
    mask = torch.ones(total, total, dtype=torch.bool)
    mask[:free_queries, :free_queries] = False
    mask[free_queries:, free_queries:] = False

    output = decoder(
        fpn,
        _metadata(batch, cameras),
        extra_directions=noisy_gt_directions,
        self_attention_mask=mask,
    )

    assert output.classification[0].shape == (batch, total, 6)
    assert output.prediction[-1].shape == (batch, total, 10)
    assert output.states[0].direction.shape == (batch, total, 3)
    torch.testing.assert_close(
        output.states[0].direction.square().sum(dim=-1),
        torch.ones(batch, total),
    )


def test_masked_direction_only_dn_cannot_change_free_queries():
    torch.manual_seed(23)
    batch, cameras, free_queries, dn_queries, channels = 1, 4, 5, 4, 16
    config = FisheyeRayDecoderConfig(
        embed_dims=channels,
        num_queries=free_queries,
        num_layers=2,
        num_heads=4,
        ffn_dims=32,
        num_cameras=cameras,
        level_strides=(8.0, 16.0),
    )
    decoder = FisheyeRayDecoder(
        torch.randn(free_queries, 3),
        config,
        projector=NativeFisheyeProjector(_AlwaysVisibleLens()),
    ).eval()
    fpn = [torch.randn(batch, cameras, channels, size, size) for size in (8, 4)]
    metadata = _metadata(batch, cameras)
    reference = decoder(fpn, metadata)
    total = free_queries + dn_queries
    mask = torch.ones(total, total, dtype=torch.bool)
    mask[:free_queries, :free_queries] = False
    mask[free_queries:, free_queries:] = False

    with_dn = decoder(
        fpn,
        metadata,
        extra_directions=torch.randn(batch, dn_queries, 3),
        self_attention_mask=mask,
    )

    for expected, actual in zip(reference.prediction, with_dn.prediction):
        torch.testing.assert_close(
            actual[:, :free_queries], expected, atol=2e-6, rtol=2e-6
        )
    for expected, actual in zip(reference.classification, with_dn.classification):
        torch.testing.assert_close(
            actual[:, :free_queries], expected, atol=2e-6, rtol=2e-6
        )
