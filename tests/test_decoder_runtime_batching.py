"""Exact contracts for dense legacy decoder batching."""

from __future__ import annotations

import inspect
from types import MethodType

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from projects.mmdet3d_plugin.models.detection3d.decoder import (
    SparseBox3DDecoder,
    _decode_dense_legacy_batch,
)


CPU_FIELDS = (
    "boxes_3d",
    "scores_3d",
    "labels_3d",
    "query_indices",
    "cls_scores",
)


def _fixture(dtype=torch.float32):
    generator = torch.Generator().manual_seed(3407)
    classification = torch.randn(4, 300, 6, dtype=dtype, generator=generator)
    # Exercise top-k and post-centerness ties without changing their code path.
    classification[:, :12] = 0.0
    boxes = torch.randn(4, 300, 11, dtype=dtype, generator=generator)
    boxes[0, 0, 6] = -0.0
    boxes[0, 0, 7] = 0.0
    boxes[0, 1, 6] = float("nan")
    boxes[0, 2, 5] = float("inf")
    quality = torch.randn(4, 300, 2, dtype=dtype, generator=generator)
    quality[:, :12, 0] = 0.0
    instance_ids = torch.arange(1200, dtype=torch.long).reshape(4, 300)
    return [classification], [boxes], instance_ids, [quality]


def _decoder(**overrides):
    settings = {
        "num_output": 300,
        "score_threshold": None,
        "sorted": True,
        "selection_score_mode": "legacy",
        "duplicate_bev_radius": 0.0,
    }
    settings.update(overrides)
    return SparseBox3DDecoder(**settings)


def _raw_equal(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    assert actual.shape == expected.shape
    assert actual.stride() == expected.stride()
    if actual.is_floating_point():
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    else:
        assert torch.equal(actual, expected)


def _assert_outputs_equal(actual, expected):
    assert len(actual) == len(expected)
    for actual_sample, expected_sample in zip(actual, expected):
        assert actual_sample.keys() == expected_sample.keys()
        for name in actual_sample:
            _raw_equal(actual_sample[name], expected_sample[name])


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_dense_batched_decode_is_integer_view_exact(dtype):
    inputs = _fixture(dtype)
    input_bits = [
        value[0].view(torch.uint8).clone()
        if isinstance(value, list)
        else value.view(torch.uint8).clone()
        for value in inputs
    ]
    initial_rng = torch.get_rng_state().clone()
    expected = _decoder().decode(*inputs)
    expected_rng = torch.get_rng_state().clone()
    torch.set_rng_state(initial_rng)
    with torch.inference_mode():
        actual = _decoder().decode(*inputs)
    actual_rng = torch.get_rng_state().clone()

    _assert_outputs_equal(actual, expected)
    assert torch.equal(actual_rng, expected_rng)
    for value, before in zip(inputs, input_bits):
        tensor = value[0] if isinstance(value, list) else value
        assert torch.equal(tensor.view(torch.uint8), before)


def test_dense_batched_decode_reduces_pointwise_launch_groups():
    class Operations(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.names = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.names.append(str(func))
            return func(*args, **(kwargs or {}))

    inputs = _fixture()
    parent_log = Operations()
    candidate_log = Operations()
    with parent_log:
        _decoder().decode(*inputs)
    with torch.inference_mode(), candidate_log:
        output = _decoder().decode(*inputs)

    assert sum("atan2" in name for name in parent_log.names) == 4
    assert sum("aten.exp.default" in name for name in parent_log.names) == 4
    assert sum("atan2" in name for name in candidate_log.names) == 1
    assert sum("aten.exp.default" in name for name in candidate_log.names) == 1
    assert inspect.getsource(_decode_dense_legacy_batch).count(".cpu()") == 5

    for name in CPU_FIELDS:
        storages = {
            sample[name].untyped_storage()._cdata for sample in output
        }
        assert len(storages) == len(output)
        assert all(sample[name].is_contiguous() for sample in output)


@pytest.mark.parametrize(
    "case",
    (
        "threshold",
        "short_topk",
        "unsorted",
        "no_quality",
        "no_instance_ids",
        "subclass",
        "shadow_decode_box",
        "box_dimension",
    ),
)
def test_dense_batched_decode_fails_closed(case):
    class ExtendedDecoder(SparseBox3DDecoder):
        pass

    overrides = {}
    decoder_type = SparseBox3DDecoder
    if case == "threshold":
        overrides["score_threshold"] = 0.5
    elif case == "short_topk":
        overrides["num_output"] = 299
    elif case == "unsorted":
        overrides["sorted"] = False
    elif case == "subclass":
        decoder_type = ExtendedDecoder
    settings = {
        "num_output": 300,
        "score_threshold": None,
        "sorted": True,
        "selection_score_mode": "legacy",
        "duplicate_bev_radius": 0.0,
    }
    settings.update(overrides)
    reference = decoder_type(**settings)
    candidate = decoder_type(**settings)
    inputs = list(_fixture())
    if case == "no_quality":
        inputs[3] = None
    elif case == "no_instance_ids":
        inputs[2] = None
    elif case == "box_dimension":
        inputs[1] = [inputs[1][0][..., :10].clone()]
    elif case == "shadow_decode_box":
        reference_decode = reference.decode_box
        candidate_decode = candidate.decode_box

        def reference_shadow(_self, box):
            return reference_decode(box)

        def candidate_shadow(_self, box):
            return candidate_decode(box)

        reference.decode_box = MethodType(reference_shadow, reference)
        candidate.decode_box = MethodType(candidate_shadow, candidate)

    expected = reference.decode(*inputs)
    operation_log = []

    class Operations(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            operation_log.append(str(func))
            return func(*args, **(kwargs or {}))

    with torch.inference_mode(), Operations():
        actual = candidate.decode(*inputs)

    _assert_outputs_equal(actual, expected)
    assert sum("atan2" in name for name in operation_log) == 4
