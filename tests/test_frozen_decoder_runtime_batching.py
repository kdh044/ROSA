"""Exactness checks for the frozen decoder's batched box transform."""

from __future__ import annotations

from types import MethodType

import pytest
import torch

import rosa.integrations.sparse4d_scores as score_module
from rosa.integrations.sparse4d_scores import (
    _decode_frozen_box_batch,
    _matches_released_sparse4d_decoder,
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


def _decoder():
    return SparseBox3DDecoder(
        num_output=300,
        score_threshold=None,
        sorted=True,
        selection_score_mode="legacy",
        duplicate_bev_radius=0.0,
    )


def _fixture(dtype, *, noncontiguous=False):
    generator = torch.Generator().manual_seed(3407)
    if noncontiguous:
        boxes = torch.randn(4, 300, 22, dtype=dtype, generator=generator)[..., ::2]
        assert not boxes.is_contiguous()
    else:
        boxes = torch.randn(4, 300, 11, dtype=dtype, generator=generator)
    boxes[0, 0, 6:8] = torch.tensor([-0.0, 0.0], dtype=dtype)
    boxes[0, 1, 6] = float("nan")
    boxes[0, 2, 3] = float("inf")
    boxes[0, 3, 4] = float("-inf")
    indices = torch.stack([torch.randperm(300, generator=generator) for _ in range(4)])
    scores = torch.randn(4, 300, dtype=dtype, generator=generator)
    scores[:, :12] = 0.0
    scores[0, :4] = torch.tensor(
        [-0.0, float("nan"), float("inf"), float("-inf")],
        dtype=dtype,
    )
    class_origin = torch.randn(4, 300, dtype=dtype, generator=generator)
    class_origin[:, :12] = 0.5
    labels = torch.randint(0, 6, (4, 300), generator=generator)
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
    assert (actual.dtype, actual.device, actual.shape, actual.stride()) == (
        expected.dtype,
        expected.device,
        expected.shape,
        expected.stride(),
    )
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
def test_batched_box_transform_and_packed_abi_are_raw_exact(dtype, noncontiguous):
    inputs = _fixture(dtype, noncontiguous=noncontiguous)
    boxes, indices, scores, labels, class_origin, instance_ids = inputs
    input_bits = [value.contiguous().view(torch.uint8).clone() for value in inputs]
    decoder = _decoder()
    decoder_state = dict(decoder.__dict__)
    initial_rng = torch.get_rng_state().clone()

    with torch.no_grad():
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
    with torch.no_grad():
        actual_decoded = _decode_frozen_box_batch(decoder, boxes, indices)
        actual = _pack_frozen_decode_fields(
            actual_decoded,
            scores,
            labels,
            indices,
            class_origin,
            torch.gather(instance_ids, 1, indices),
        )

    _raw_equal(actual_decoded, expected_decoded)
    _assert_outputs_equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert decoder.__dict__ == decoder_state
    for value, before in zip(inputs, input_bits):
        assert torch.equal(value.contiguous().view(torch.uint8), before)
    for name in FIELDS:
        storages = {sample[name].untyped_storage()._cdata for sample in actual}
        assert len(storages) == len(actual)
        assert all(sample[name]._base is None for sample in actual)
        assert all(sample[name].is_contiguous() for sample in actual)


def test_batched_box_transform_calls_decoder_once_with_flat_abi():
    boxes, indices, *_ = _fixture(torch.float32)
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


def _rank_fixture(dtype):
    generator = torch.Generator().manual_seed(7123)
    classification = torch.randn(4, 300, 6, dtype=dtype, generator=generator)
    boxes = torch.randn(4, 300, 11, dtype=dtype, generator=generator)
    boxes[0, 0, 6:8] = torch.tensor([-0.0, 0.0], dtype=dtype)
    boxes[0, 1, 6] = float("nan")
    frozen_label = torch.randint(0, 6, (4, 300), generator=generator)
    native_probability = torch.rand(4, 300, dtype=dtype, generator=generator)
    native_probability[:, :16] = 0.5
    native_probability[0, :3] = torch.tensor(
        [-0.0, float("nan"), float("inf")], dtype=dtype
    )
    joint_score = torch.rand(4, 300, dtype=dtype, generator=generator)
    joint_score[:, :16] = 0.25
    joint_score[0, :4] = torch.tensor(
        [-0.0, float("nan"), float("inf"), float("-inf")], dtype=dtype
    )
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
    _, boxes, frozen_label, native_probability, joint_score, instance_ids = inputs
    class_origin, indices = native_probability.topk(300, dim=1, sorted=True)
    labels = torch.gather(frozen_label, 1, indices)
    scores = torch.gather(joint_score, 1, indices)
    scores, order = torch.sort(scores, dim=1, descending=True)
    indices = torch.gather(indices, 1, order)
    labels = torch.gather(labels, 1, order)
    class_origin = torch.gather(class_origin, 1, order)
    return (
        [
            {
                "boxes_3d": _decoder().decode_box(boxes[index, indices[index]]).cpu(),
                "scores_3d": scores[index].cpu(),
                "labels_3d": labels[index].cpu(),
                "query_indices": indices[index].cpu(),
                "cls_scores": class_origin[index].cpu(),
                "instance_ids": instance_ids[index, indices[index]].cpu(),
            }
            for index in range(boxes.shape[0])
        ],
        indices,
    )


@pytest.mark.parametrize(
    "dtype",
    (torch.float16, torch.bfloat16, torch.float32, torch.float64),
)
def test_two_stage_rank_ties_and_special_values_remain_raw_exact(dtype):
    inputs = _rank_fixture(dtype)
    with torch.inference_mode():
        expected, indices = _reference_rank_decode(inputs)
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
        batched = _decode_frozen_box_batch(_decoder(), inputs[1], indices)
        released = _reference_decoded(_decoder(), inputs[1], indices)

    _assert_outputs_equal(actual, expected)
    _raw_equal(batched, released)


def test_released_decoder_abi_match_is_exact(monkeypatch):
    decoder = _decoder()
    assert _matches_released_sparse4d_decoder(decoder)

    class ExtendedDecoder(SparseBox3DDecoder):
        pass

    assert not _matches_released_sparse4d_decoder(ExtendedDecoder())
    decoder.decode_box = MethodType(lambda self, boxes: boxes, decoder)
    assert not _matches_released_sparse4d_decoder(decoder)

    with monkeypatch.context() as patch:
        released = SparseBox3DDecoder.decode_box
        patch.setattr(
            SparseBox3DDecoder,
            "decode_box",
            lambda self, boxes: released(self, boxes),
        )
        assert not _matches_released_sparse4d_decoder(_decoder())


def test_cpu_decode_keeps_existing_fallback(monkeypatch):
    inputs = _rank_fixture(torch.float32)
    with torch.inference_mode():
        expected, _ = _reference_rank_decode(inputs)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("CPU decode entered the CUDA-only box batch")

    monkeypatch.setattr(score_module, "_decode_frozen_box_batch", forbidden)
    with torch.inference_mode():
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
