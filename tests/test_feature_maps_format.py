"""Exact contracts for the single-copy fused-DAF feature layout."""

from __future__ import annotations

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from projects.mmdet3d_plugin.ops import feature_maps_format


def _historical_feature_maps_format(feature_maps):
    """Frozen pre-optimization implementation used as the value oracle."""

    if isinstance(feature_maps[0], (list, tuple)):
        formatted = [
            _historical_feature_maps_format(group) for group in feature_maps
        ]
        return [
            torch.cat([value[0] for value in formatted], dim=1),
            torch.cat([value[1] for value in formatted], dim=0),
            torch.cat([value[2] for value in formatted], dim=0),
        ]

    batch, cameras = feature_maps[0].shape[:2]
    spatial_shapes = []
    column_features = []
    for feature in feature_maps:
        spatial_shapes.append(feature.shape[-2:])
        column_features.append(
            torch.reshape(
                feature,
                (batch, cameras, feature.shape[2], -1),
            )
        )
    column_features = (
        torch.cat(column_features, dim=-1)
        .permute(0, 1, 3, 2)
        .flatten(1, 2)
    )
    spatial_shapes = torch.tensor(
        [spatial_shapes] * cameras,
        dtype=torch.int64,
        device=column_features.device,
    )
    starts = spatial_shapes[..., 0] * spatial_shapes[..., 1]
    starts = starts.flatten().cumsum(dim=0)
    starts = torch.cat([torch.tensor([0]).to(starts), starts[:-1]])
    return [
        column_features,
        spatial_shapes,
        starts.reshape(cameras, -1),
    ]


def _features(dtype=torch.float32, *, noncontiguous=False):
    generator = torch.Generator().manual_seed(3407)
    output = []
    for height, width in ((7, 5), (4, 3), (2, 2), (1, 1)):
        if noncontiguous:
            feature = torch.randn(
                2,
                4,
                8,
                height * 2,
                width * 2,
                dtype=dtype,
                generator=generator,
            )[..., ::2, ::2]
            assert not feature.is_contiguous()
        else:
            feature = torch.randn(
                2,
                4,
                8,
                height,
                width,
                dtype=dtype,
                generator=generator,
            )
        output.append(feature)
    return output


@pytest.mark.parametrize(
    "dtype",
    (torch.float16, torch.float32, torch.float64, torch.bfloat16),
)
@pytest.mark.parametrize("noncontiguous", (False, True))
def test_single_copy_layout_is_raw_exact(dtype, noncontiguous):
    features = _features(dtype, noncontiguous=noncontiguous)
    expected = _historical_feature_maps_format(features)
    actual = feature_maps_format(features)

    assert torch.equal(actual[0], expected[0])
    assert actual[0].shape == expected[0].shape
    assert actual[0].dtype == expected[0].dtype
    assert actual[0].stride() == expected[0].stride()
    for candidate, parent in zip(actual[1:], expected[1:]):
        assert candidate.dtype == torch.int32
        assert torch.equal(candidate, parent.to(torch.int32))
        assert candidate.shape == parent.shape
        assert candidate.stride() == parent.stride()
    assert actual[0].is_contiguous()


def test_single_copy_layout_preserves_exact_backward_mapping():
    parent_inputs = [value.requires_grad_() for value in _features()]
    candidate_inputs = [
        value.detach().clone().requires_grad_() for value in parent_inputs
    ]
    parent = _historical_feature_maps_format(parent_inputs)[0]
    candidate = feature_maps_format(candidate_inputs)[0]
    upstream = torch.randn(
        parent.shape,
        dtype=parent.dtype,
        generator=torch.Generator().manual_seed(7101080),
    )

    parent_gradients = torch.autograd.grad((parent * upstream).sum(), parent_inputs)
    candidate_gradients = torch.autograd.grad(
        (candidate * upstream).sum(),
        candidate_inputs,
    )
    for actual, expected in zip(candidate_gradients, parent_gradients):
        assert torch.equal(actual, expected)


def test_single_copy_layout_preserves_nested_feature_groups():
    first = _features()
    second = [value.mul(0.5) for value in _features()]
    expected = _historical_feature_maps_format([first, second])
    actual = feature_maps_format([first, second])
    assert torch.equal(actual[0], expected[0])
    for candidate, parent in zip(actual[1:], expected[1:]):
        assert candidate.dtype == torch.int32
        assert torch.equal(candidate, parent.to(torch.int32))
        assert candidate.stride() == parent.stride()


def test_single_copy_layout_round_trips_through_inverse_format():
    features = _features()
    expected_groups = feature_maps_format(
        _historical_feature_maps_format(features),
        inverse=True,
    )
    reconstructed_groups = feature_maps_format(
        feature_maps_format(features),
        inverse=True,
    )
    assert len(reconstructed_groups) == len(expected_groups) == 1
    reconstructed = reconstructed_groups[0]
    expected_reconstructed = expected_groups[0]
    assert len(reconstructed) == len(expected_reconstructed) == len(features)
    for actual, parent, source in zip(
        reconstructed,
        expected_reconstructed,
        features,
    ):
        assert torch.equal(actual, parent)
        assert torch.equal(actual, source)
        assert actual.stride() == parent.stride()


def test_single_copy_layout_removes_one_full_feature_storage():
    class Capture(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.tensors = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            output = func(*args, **(kwargs or {}))

            def retain(value):
                if isinstance(value, torch.Tensor):
                    self.tensors.append(value)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        retain(item)

            retain(output)
            return output

    features = _features()
    final_elements = 2 * 4 * 8 * sum(
        height * width for height, width in ((7, 5), (4, 3), (2, 2), (1, 1))
    )
    final_bytes = final_elements * torch.tensor([], dtype=torch.float32).element_size()
    input_storages = {value.untyped_storage()._cdata for value in features}

    counts = []
    for formatter in (_historical_feature_maps_format, feature_maps_format):
        capture = Capture()
        with capture:
            output = formatter(features)
        assert output[0].untyped_storage().nbytes() == final_bytes
        storages = {
            value.untyped_storage()._cdata: value.untyped_storage().nbytes()
            for value in capture.tensors
            if value.untyped_storage()._cdata not in input_storages
        }
        counts.append(sum(size == final_bytes for size in storages.values()))

    assert counts == [2, 1]
    production_bytes = 4 * 4 * 256 * 53_125 * 4
    assert production_bytes == 870_400_000


def test_fused_daf_int32_conversion_reuses_preformatted_metadata_storage():
    formatted = feature_maps_format(_features())
    _, spatial_shapes, starts = formatted
    assert spatial_shapes.dtype == starts.dtype == torch.int32
    assert spatial_shapes.is_contiguous() and starts.is_contiguous()
    # These are the exact conversions at every fused-DAF boundary.  When the
    # formatter already emits contiguous int32, both operations are aliases
    # instead of allocating and launching dtype-conversion work.
    assert spatial_shapes.contiguous().int() is spatial_shapes
    assert starts.contiguous().int() is starts
