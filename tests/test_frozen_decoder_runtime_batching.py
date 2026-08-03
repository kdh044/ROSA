"""Exact contracts for the CUDA-only frozen decoder box batching."""

from __future__ import annotations

from types import MethodType

import pytest
import torch
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

import rosa.integrations.sparse4d_scores as score_module
import rosa.rfl as rfl_module
from rosa.integrations.sparse4d_scores import (
    _can_batch_frozen_decode_box,
    _decode_frozen_box_batch,
    _matches_frozen_decode_box_batch_contract,
    _pack_frozen_decode_fields,
    decode_frozen_jointscore_rank,
)

decoder_module = pytest.importorskip(
    "projects.mmdet3d_plugin.models.detection3d.decoder",
    reason="optional Sparse4D detector host is not available",
)
SparseBox3DDecoder = decoder_module.SparseBox3DDecoder

FIELDS = (
    "boxes_3d",
    "scores_3d",
    "labels_3d",
    "query_indices",
    "cls_scores",
    "instance_ids",
)


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


def _fixture(dtype=torch.float32, *, noncontiguous=False):
    generator = torch.Generator().manual_seed(3407)
    if noncontiguous:
        storage = torch.randn(4, 300, 22, dtype=dtype, generator=generator)
        boxes = storage[..., ::2]
        assert not boxes.is_contiguous()
    else:
        boxes = torch.randn(4, 300, 11, dtype=dtype, generator=generator)
    boxes[0, 0, 6] = -0.0
    boxes[0, 0, 7] = 0.0
    boxes[0, 1, 6] = float("nan")
    boxes[0, 2, 3] = float("inf")
    boxes[0, 3, 4] = float("-inf")
    indices = torch.stack([torch.randperm(300, generator=generator) for _ in range(4)])
    scores = torch.randn(4, 300, dtype=dtype, generator=generator)
    scores[:, :12] = 0.0
    scores[0, 0] = -0.0
    scores[0, 1] = float("nan")
    scores[0, 2] = float("inf")
    class_origin = torch.randn(4, 300, dtype=dtype, generator=generator)
    class_origin[:, :12] = 0.5
    labels = torch.randint(0, 6, (4, 300), dtype=torch.long, generator=generator)
    instance_ids = torch.arange(1200, dtype=torch.long).reshape(4, 300)
    return boxes, indices, scores, labels, class_origin, instance_ids


def _reference_decoded(decoder, boxes, indices):
    return torch.stack(
        [
            decoder.decode_box(boxes[batch_index, indices[batch_index]])
            for batch_index in range(boxes.shape[0])
        ],
        dim=0,
    )


def _raw_equal(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    assert actual.shape == expected.shape
    assert actual.stride() == expected.stride()
    assert torch.equal(
        actual.contiguous().view(torch.uint8),
        expected.contiguous().view(torch.uint8),
    )


def _assert_outputs_equal(actual, expected):
    assert len(actual) == len(expected)
    for actual_sample, expected_sample in zip(actual, expected):
        assert actual_sample.keys() == expected_sample.keys()
        for name in FIELDS:
            _raw_equal(actual_sample[name], expected_sample[name])


@pytest.mark.parametrize(
    "dtype",
    (torch.float16, torch.bfloat16, torch.float32, torch.float64),
)
@pytest.mark.parametrize("noncontiguous", (False, True))
def test_frozen_box_batch_and_packed_fields_are_raw_exact(dtype, noncontiguous):
    boxes, indices, scores, labels, class_origin, instance_ids = _fixture(
        dtype,
        noncontiguous=noncontiguous,
    )
    inputs = (boxes, indices, scores, labels, class_origin, instance_ids)
    input_bits = [value.contiguous().view(torch.uint8).clone() for value in inputs]
    initial_rng = torch.get_rng_state().clone()
    decoder = _decoder()
    with torch.inference_mode():
        expected_decoded = _reference_decoded(decoder, boxes, indices)
        expected = _pack_frozen_decode_fields(
            expected_decoded,
            scores,
            labels,
            indices,
            class_origin,
            torch.gather(instance_ids, 1, indices),
        )
    expected_rng = torch.get_rng_state().clone()
    torch.set_rng_state(initial_rng)
    with torch.inference_mode():
        actual_decoded = _decode_frozen_box_batch(decoder, boxes, indices)
        actual = _pack_frozen_decode_fields(
            actual_decoded,
            scores,
            labels,
            indices,
            class_origin,
            torch.gather(instance_ids, 1, indices),
        )
    actual_rng = torch.get_rng_state().clone()

    _raw_equal(actual_decoded, expected_decoded)
    _assert_outputs_equal(actual, expected)
    assert torch.equal(actual_rng, expected_rng)
    for value, before in zip(inputs, input_bits):
        assert torch.equal(value.contiguous().view(torch.uint8), before)
    for name in FIELDS:
        storages = {sample[name].untyped_storage()._cdata for sample in actual}
        assert len(storages) == len(actual)
        assert all(sample[name]._base is None for sample in actual)
        assert all(sample[name].is_contiguous() for sample in actual)


def test_frozen_box_batch_calls_released_decoder_once_with_flat_abi():
    boxes, indices, *_ = _fixture()
    released = _decoder()
    calls = []

    class ObservedDecoder:
        def decode_box(self, selected):
            calls.append((selected.shape, selected.stride(), selected.dtype))
            return released.decode_box(selected)

    with torch.inference_mode():
        actual = _decode_frozen_box_batch(ObservedDecoder(), boxes, indices)
        expected = _reference_decoded(released, boxes, indices)

    assert calls == [(torch.Size((1200, 11)), (11, 1), torch.float32)]
    _raw_equal(actual, expected)


def test_frozen_box_batch_reduces_decode_pointwise_groups():
    class Operations(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.names = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            del types
            self.names.append(str(func))
            return func(*args, **(kwargs or {}))

    boxes, indices, *_ = _fixture()
    reference_log = Operations()
    candidate_log = Operations()
    with torch.inference_mode(), reference_log:
        _reference_decoded(_decoder(), boxes, indices)
    with torch.inference_mode(), candidate_log:
        _decode_frozen_box_batch(_decoder(), boxes, indices)

    assert sum("atan2" in name for name in reference_log.names) == 4
    assert sum("aten.exp.default" in name for name in reference_log.names) == 4
    assert sum("atan2" in name for name in candidate_log.names) == 1
    assert sum("aten.exp.default" in name for name in candidate_log.names) == 1


def _rank_fixture(dtype):
    generator = torch.Generator().manual_seed(7123)
    classification = torch.randn(4, 300, 6, dtype=dtype, generator=generator)
    boxes = torch.randn(4, 300, 11, dtype=dtype, generator=generator)
    boxes[0, 0, 6] = -0.0
    boxes[0, 0, 7] = 0.0
    boxes[0, 1, 6] = float("nan")
    frozen_label = torch.randint(0, 6, (4, 300), generator=generator)
    native_probability = torch.rand(4, 300, dtype=dtype, generator=generator)
    native_probability[:, :16] = 0.5
    native_probability[0, 0] = -0.0
    native_probability[0, 1] = float("nan")
    joint_score = torch.rand(4, 300, dtype=dtype, generator=generator)
    joint_score[:, :16] = 0.25
    joint_score[0, 0] = -0.0
    joint_score[0, 1] = float("nan")
    joint_score[0, 2] = float("inf")
    joint_score[0, 3] = float("-inf")
    instance_ids = torch.arange(1200, dtype=torch.long).reshape(4, 300)
    return (
        classification,
        boxes,
        frozen_label,
        native_probability,
        joint_score,
        instance_ids,
    )


def _reference_rank_decode(inputs):
    classification, boxes, frozen_label, native_probability, joint_score, ids = inputs
    del classification
    class_origin, indices = native_probability.topk(300, dim=1, sorted=True)
    labels = torch.gather(frozen_label, 1, indices)
    scores = torch.gather(joint_score, 1, indices)
    scores, order = torch.sort(scores, dim=1, descending=True)
    indices = torch.gather(indices, 1, order)
    labels = torch.gather(labels, 1, order)
    class_origin = torch.gather(class_origin, 1, order)
    decoded = _reference_decoded(_decoder(), boxes, indices)
    selected_ids = torch.gather(ids, 1, indices)
    return (
        _pack_frozen_decode_fields(
            decoded,
            scores,
            labels,
            indices,
            class_origin,
            selected_ids,
        ),
        indices,
        decoded,
    )


@pytest.mark.parametrize(
    "dtype",
    (torch.float16, torch.bfloat16, torch.float32, torch.float64),
)
def test_two_stage_rank_ties_and_low_precision_remain_raw_exact(dtype):
    inputs = _rank_fixture(dtype)
    with torch.inference_mode():
        expected, final_indices, expected_decoded = _reference_rank_decode(inputs)
        actual = decode_frozen_jointscore_rank(
            _decoder(),
            [inputs[0]],
            [inputs[1]],
            inputs[2],
            inputs[3],
            inputs[4],
            instance_id=inputs[5],
            output_idx=0,
            validate_values=False,
            pack_host_transfers=True,
        )
        batched_decoded = _decode_frozen_box_batch(
            _decoder(),
            inputs[1],
            final_indices,
        )

    _assert_outputs_equal(actual, expected)
    _raw_equal(batched_decoded, expected_decoded)


def _contract_inputs():
    boxes, indices, *_ = _fixture()
    return _decoder(), boxes, indices


def test_frozen_box_batch_contract_matches_cpu_but_cuda_gate_does_not():
    inputs = _contract_inputs()
    with torch.no_grad():
        assert _matches_frozen_decode_box_batch_contract(*inputs)
        assert not _can_batch_frozen_decode_box(*inputs)
    with torch.inference_mode():
        assert _matches_frozen_decode_box_batch_contract(*inputs)
        assert not _can_batch_frozen_decode_box(*inputs)


def test_frozen_box_batch_contract_rejects_observable_execution_modes(monkeypatch):
    inputs = _contract_inputs()
    assert not _matches_frozen_decode_box_batch_contract(*inputs)
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        assert not _matches_frozen_decode_box_batch_contract(*inputs)

    monkeypatch.setattr(torch._dynamo, "is_compiling", lambda: True)
    with torch.inference_mode():
        assert not _matches_frozen_decode_box_batch_contract(*inputs)


def test_frozen_box_batch_contract_rejects_override_modes():
    class FunctionMode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            del types
            return func(*args, **(kwargs or {}))

    class DispatchMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            del types
            return func(*args, **(kwargs or {}))

    inputs = _contract_inputs()
    with torch.inference_mode(), FunctionMode():
        assert not _matches_frozen_decode_box_batch_contract(*inputs)
    with torch.inference_mode(), DispatchMode():
        assert not _matches_frozen_decode_box_batch_contract(*inputs)


@pytest.mark.parametrize(
    ("owner", "name"),
    (
        (score_module, "_decode_frozen_box_batch"),
        (score_module, "_matches_released_sparse4d_decoder"),
        (score_module, "_matches_frozen_decode_box_batch_contract"),
        (rfl_module, "_plain_eager_no_grad_execution"),
        (torch, "gather"),
        (torch, "atan2"),
        (torch, "cat"),
        (torch, "stack"),
        (torch.Tensor, "__getitem__"),
        (torch.Tensor, "expand_as"),
        (torch.Tensor, "reshape"),
        (torch.Tensor, "exp"),
    ),
)
def test_frozen_box_batch_contract_rejects_primitive_override(
    monkeypatch,
    owner,
    name,
):
    inputs = _contract_inputs()
    monkeypatch.setattr(owner, name, lambda *_args, **_kwargs: None)
    with torch.inference_mode():
        assert not _can_batch_frozen_decode_box(*inputs)


@pytest.mark.parametrize(
    "case",
    (
        "integer_box",
        "box_dimension",
        "index_dtype",
        "short_query_axis",
        "empty_batch",
        "quantized_box",
        "mkldnn_box",
        "sparse_box",
        "meta_box",
        "named_box",
        "tensor_subclass",
        "requires_grad",
    ),
)
def test_frozen_box_batch_contract_rejects_nonconventional_tensors(case):
    decoder, boxes, indices = _contract_inputs()
    if case == "integer_box":
        boxes = boxes.to(torch.int64)
    elif case == "box_dimension":
        boxes = boxes[..., :10]
    elif case == "index_dtype":
        indices = indices.to(torch.int32)
    elif case == "short_query_axis":
        boxes, indices = boxes[:, :299], indices[:, :299]
    elif case == "empty_batch":
        boxes, indices = boxes[:0], indices[:0]
    elif case == "quantized_box":
        boxes = torch.quantize_per_tensor(
            boxes,
            scale=0.1,
            zero_point=0,
            dtype=torch.qint8,
        )
    elif case == "mkldnn_box":
        boxes = boxes.to_mkldnn()
    elif case == "sparse_box":
        boxes = torch.sparse_coo_tensor(
            torch.empty((3, 0), dtype=torch.long),
            torch.empty((0,), dtype=boxes.dtype),
            size=boxes.shape,
        )
    elif case == "meta_box":
        boxes = torch.empty(boxes.shape, device="meta", dtype=boxes.dtype)
    elif case == "named_box":
        boxes = boxes.refine_names("batch", "query", "state")
    elif case == "tensor_subclass":

        class TensorSubclass(torch.Tensor):
            pass

        boxes = torch.Tensor._make_subclass(TensorSubclass, boxes)
    elif case == "requires_grad":
        boxes = boxes.requires_grad_(True)

    with torch.inference_mode():
        assert not _matches_frozen_decode_box_batch_contract(
            decoder,
            boxes,
            indices,
        )


@pytest.mark.parametrize(
    "case",
    (
        "subclass",
        "shadow_decode_box",
        "class_decode_box",
        "threshold",
        "truthy_sorted",
        "boolean_num_output",
        "boolean_duplicate_radius",
    ),
)
def test_frozen_box_batch_contract_rejects_decoder_abi_changes(monkeypatch, case):
    decoder, boxes, indices = _contract_inputs()
    if case == "subclass":

        class ExtendedDecoder(SparseBox3DDecoder):
            pass

        decoder = ExtendedDecoder(
            num_output=300,
            score_threshold=None,
            sorted=True,
            selection_score_mode="legacy",
            duplicate_bev_radius=0.0,
        )
    elif case == "shadow_decode_box":
        released = decoder.decode_box

        def shadow(_self, selected):
            return released(selected)

        decoder.decode_box = MethodType(shadow, decoder)
    elif case == "class_decode_box":
        released = SparseBox3DDecoder.decode_box

        def patched(self, selected):
            return released(self, selected)

        monkeypatch.setattr(SparseBox3DDecoder, "decode_box", patched)
    elif case == "threshold":
        decoder.score_threshold = 0.0
    elif case == "truthy_sorted":
        decoder.sorted = 1
    elif case == "boolean_num_output":
        decoder.num_output = True
    elif case == "boolean_duplicate_radius":
        decoder.duplicate_bev_radius = False

    with torch.inference_mode():
        assert not _matches_frozen_decode_box_batch_contract(
            decoder,
            boxes,
            indices,
        )


def test_cpu_decode_keeps_existing_fallback(monkeypatch):
    inputs = _rank_fixture(torch.float32)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("CPU decode entered the CUDA-only box batch")

    monkeypatch.setattr(score_module, "_decode_frozen_box_batch", forbidden)
    with torch.inference_mode():
        expected, _, _ = _reference_rank_decode(inputs)
        actual = decode_frozen_jointscore_rank(
            _decoder(),
            [inputs[0]],
            [inputs[1]],
            inputs[2],
            inputs[3],
            inputs[4],
            instance_id=inputs[5],
            output_idx=0,
            validate_values=False,
            pack_host_transfers=True,
        )
    _assert_outputs_equal(actual, expected)
