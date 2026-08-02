"""Exact contracts for CUDA-only dense legacy decoder batching."""

from __future__ import annotations

import inspect
from types import MethodType

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from projects.mmdet3d_plugin.models.detection3d.decoder import (
    SparseBox3DDecoder,
    _can_batch_dense_legacy_decode,
    _decode_dense_legacy_batch,
    _matches_batch_dense_legacy_contract,
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


def _dense_tail_inputs(inputs, output_idx=-1):
    """Freeze the released top-k/centerness route immediately before decode."""

    classifications, predictions, instance_ids, qualities = inputs
    cls_scores = classifications[output_idx].sigmoid()
    cls_scores, cls_ids = cls_scores.max(dim=-1)
    cls_scores = cls_scores.unsqueeze(dim=-1)
    box_preds = predictions[output_idx]
    _, _, num_cls = cls_scores.shape
    cls_scores, indices = cls_scores.flatten(start_dim=1).topk(
        300,
        dim=1,
        sorted=True,
    )
    quality = qualities[output_idx]
    centerness = torch.gather(quality[..., 0], 1, indices // num_cls)
    scores_origin = cls_scores.clone()
    cls_scores *= centerness.sigmoid()
    cls_scores, sort_indices = torch.sort(
        cls_scores,
        dim=1,
        descending=True,
    )
    indices = torch.gather(indices, 1, sort_indices)
    return (
        cls_scores,
        cls_ids,
        box_preds,
        indices,
        scores_origin,
        instance_ids,
        quality,
    )


def _frozen_decode_box(box):
    yaw = torch.atan2(box[:, 6], box[:, 7])
    return torch.cat(
        [
            box[:, [0, 1, 2]],
            box[:, [3, 4, 5]].exp(),
            yaw[:, None],
            box[:, 8:],
        ],
        dim=-1,
    )


def _frozen_per_sample_dense_tail(
    cls_scores,
    cls_ids,
    box_preds,
    indices,
    scores_origin,
    instance_ids,
    quality,
):
    """The exact released loop, kept independent of production methods."""

    del quality
    output = []
    for batch_index in range(cls_scores.shape[0]):
        query_indices = indices[batch_index]
        labels = cls_ids[batch_index][query_indices]
        boxes = _frozen_decode_box(box_preds[batch_index, query_indices])
        output.append(
            {
                "boxes_3d": boxes.cpu(),
                "scores_3d": cls_scores[batch_index].cpu(),
                "labels_3d": labels.cpu(),
                "query_indices": query_indices.cpu(),
                "cls_scores": scores_origin[batch_index].cpu(),
                "instance_ids": instance_ids[batch_index, query_indices],
            }
        )
    return output


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


@pytest.mark.parametrize(
    "dtype",
    (torch.float16, torch.bfloat16, torch.float32, torch.float64),
)
def test_dense_batched_tail_is_integer_view_exact(dtype):
    inputs = _fixture(dtype)
    input_bits = [
        (
            value[0].view(torch.uint8).clone()
            if isinstance(value, list)
            else value.view(torch.uint8).clone()
        )
        for value in inputs
    ]
    tail_inputs = _dense_tail_inputs(inputs)
    initial_rng = torch.get_rng_state().clone()
    with torch.inference_mode():
        expected = _frozen_per_sample_dense_tail(*tail_inputs)
    expected_rng = torch.get_rng_state().clone()
    torch.set_rng_state(initial_rng)
    with torch.inference_mode():
        actual = _decode_dense_legacy_batch(*tail_inputs[:-1])
    actual_rng = torch.get_rng_state().clone()

    _assert_outputs_equal(actual, expected)
    assert torch.equal(actual_rng, expected_rng)
    for value, before in zip(inputs, input_bits):
        tensor = value[0] if isinstance(value, list) else value
        assert torch.equal(tensor.view(torch.uint8), before)


def test_dense_batched_tail_reduces_pointwise_launch_groups():
    class Operations(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.names = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.names.append(str(func))
            return func(*args, **(kwargs or {}))

    tail_inputs = _dense_tail_inputs(_fixture())
    parent_log = Operations()
    candidate_log = Operations()
    with torch.inference_mode(), parent_log:
        _frozen_per_sample_dense_tail(*tail_inputs)
    with torch.inference_mode(), candidate_log:
        output = _decode_dense_legacy_batch(*tail_inputs[:-1])

    assert sum("atan2" in name for name in parent_log.names) == 4
    assert sum("aten.exp.default" in name for name in parent_log.names) == 4
    assert sum("atan2" in name for name in candidate_log.names) == 1
    assert sum("aten.exp.default" in name for name in candidate_log.names) == 1
    assert inspect.getsource(_decode_dense_legacy_batch).count(".cpu()") == 5

    for name in CPU_FIELDS:
        storages = {sample[name].untyped_storage()._cdata for sample in output}
        assert len(storages) == len(output)
        assert all(sample[name].is_contiguous() for sample in output)


def test_main126_contract_matches_except_for_the_cpu_device():
    inputs = _fixture()
    tail_inputs = _dense_tail_inputs(inputs)
    with torch.inference_mode():
        assert _matches_batch_dense_legacy_contract(
            _decoder(),
            *tail_inputs,
        )
        assert not _can_batch_dense_legacy_decode(
            _decoder(),
            *tail_inputs,
        )

    operation_log = []

    class Operations(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            operation_log.append(str(func))
            return func(*args, **(kwargs or {}))

    with torch.inference_mode():
        expected = _frozen_per_sample_dense_tail(*tail_inputs)
    with torch.inference_mode(), Operations():
        actual = _decoder().decode(*inputs)
    _assert_outputs_equal(actual, expected)
    assert sum("atan2" in name for name in operation_log) == 4
    assert sum("aten.exp.default" in name for name in operation_log) == 4

    # The released CPU loop exposes shared batch storages for these views.
    for name in ("scores_3d", "query_indices", "cls_scores"):
        expected_storages = {
            sample[name].untyped_storage()._cdata for sample in expected
        }
        actual_storages = {sample[name].untyped_storage()._cdata for sample in actual}
        assert len(expected_storages) == len(actual_storages) == 1


def test_dense_batch_contract_rejects_class_decode_box_monkeypatch(monkeypatch):
    tail_inputs = _dense_tail_inputs(_fixture())
    released_decode_box = SparseBox3DDecoder.decode_box

    def patched_decode_box(self, box):
        return released_decode_box(self, box) + 12345.0

    with torch.inference_mode():
        assert _matches_batch_dense_legacy_contract(_decoder(), *tail_inputs)
    monkeypatch.setattr(
        SparseBox3DDecoder,
        "decode_box",
        patched_decode_box,
    )
    with torch.inference_mode():
        assert not _matches_batch_dense_legacy_contract(
            _decoder(),
            *tail_inputs,
        )

    with torch.inference_mode():
        output = _decoder().decode(*_fixture())
    assert bool((output[0]["boxes_3d"][:, 0] > 12000.0).all())


@pytest.mark.parametrize(
    "case",
    (
        "integer_box",
        "mixed_float_dtype",
        "quantized_box",
        "mkldnn_box",
        "sparse_box",
        "meta_box",
        "named_box",
        "integer_quality",
        "non_long_instance_ids",
    ),
)
def test_dense_batch_contract_rejects_nonconventional_tensors(case):
    tail_inputs = list(_dense_tail_inputs(_fixture()))
    box_preds = tail_inputs[2]
    if case == "integer_box":
        tail_inputs[2] = box_preds.to(torch.int64)
    elif case == "mixed_float_dtype":
        tail_inputs[2] = box_preds.to(torch.float64)
    elif case == "quantized_box":
        tail_inputs[2] = torch.quantize_per_tensor(
            box_preds,
            scale=0.1,
            zero_point=0,
            dtype=torch.qint8,
        )
    elif case == "mkldnn_box":
        tail_inputs[2] = box_preds.to_mkldnn()
    elif case == "sparse_box":
        tail_inputs[2] = torch.sparse_coo_tensor(
            torch.empty((3, 0), dtype=torch.long),
            torch.empty((0,), dtype=box_preds.dtype),
            size=box_preds.shape,
        )
    elif case == "meta_box":
        tail_inputs[2] = torch.empty(box_preds.shape, device="meta")
    elif case == "named_box":
        tail_inputs[2] = box_preds.refine_names("batch", "query", "state")
    elif case == "integer_quality":
        tail_inputs[6] = tail_inputs[6].to(torch.int64)
    elif case == "non_long_instance_ids":
        tail_inputs[5] = tail_inputs[5].to(torch.int32)

    with torch.inference_mode():
        assert not _matches_batch_dense_legacy_contract(
            _decoder(),
            *tail_inputs,
        )


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

    with torch.inference_mode():
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
