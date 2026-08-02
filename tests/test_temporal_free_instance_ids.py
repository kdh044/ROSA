"""Exact contracts for fresh IDs in the non-temporal Main126 path."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from projects.mmdet3d_plugin.models.instance_bank import (
    InstanceBank,
    _fresh_non_temporal_instance_id_shape,
)

RESET_STATE = (
    "cached_feature",
    "cached_anchor",
    "metas",
    "mask",
    "confidence",
    "temp_confidence",
    "instance_id",
    "_last_cache_update_multiplier",
    "_last_cache_unsuppressed_count",
    "_last_fixed_selection_indices",
    "_last_cache_selection_indices",
)


def _bank(*, num_temporal=0, bank_type=InstanceBank):
    return bank_type(
        num_anchor=300,
        embed_dims=8,
        anchor=np.zeros((300, 11), dtype=np.float32),
        num_temp_instances=num_temporal,
        anchor_grad=False,
        feat_grad=False,
    ).eval()


def _fixture(*, queries=300):
    generator = torch.Generator().manual_seed(3407)
    confidence = torch.randn(4, queries, 6, generator=generator)
    confidence[0, 0] = float("nan")
    confidence[0, 1, 0] = float("inf")
    confidence[0, 2, 0] = float("-inf")
    confidence[0, 3, 0] = -0.0
    anchor = torch.randn(4, queries, 11, generator=generator)
    return confidence, anchor


def _historical_get_instance_id(bank, confidence, anchor=None, threshold=None):
    del anchor
    scores = confidence.max(dim=-1).values.sigmoid()
    instance_id = scores.new_full(scores.shape, -1).long()
    if (
        bank.instance_id is not None
        and bank.instance_id.shape[0] == instance_id.shape[0]
    ):
        instance_id[:, : bank.instance_id.shape[1]] = bank.instance_id
    mask = instance_id < 0
    if threshold is not None:
        mask = mask & (scores >= threshold)
    num_new_instance = mask.sum()
    new_ids = torch.arange(num_new_instance).to(instance_id) + bank.prev_id
    instance_id[torch.where(mask)] = new_ids
    bank.prev_id += num_new_instance
    if bank.num_temp_instances > 0:
        bank.update_instance_id(instance_id, scores)
    return instance_id


def _assert_value_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        assert isinstance(expected, torch.Tensor)
        assert actual.dtype == expected.dtype
        assert actual.device == expected.device
        assert actual.shape == expected.shape
        assert actual.stride() == expected.stride()
        assert torch.equal(actual, expected)
    else:
        assert type(actual) is type(expected)
        assert actual == expected


def _assert_bank_runtime_equal(actual, expected):
    _assert_value_equal(actual.prev_id, expected.prev_id)
    for name in RESET_STATE:
        _assert_value_equal(
            actual.__dict__.get(name),
            expected.__dict__.get(name),
        )


@pytest.mark.parametrize(
    "dtype",
    (torch.float16, torch.bfloat16, torch.float32, torch.float64),
)
def test_fresh_non_temporal_ids_are_integer_view_exact(dtype):
    confidence, anchor = _fixture()
    confidence = confidence.to(dtype)
    anchor = anchor.to(dtype)
    confidence_before = confidence.view(torch.uint8).clone()
    anchor_before = anchor.view(torch.uint8).clone()
    reference = _bank()
    candidate = _bank()
    initial_rng = torch.get_rng_state().clone()

    with torch.inference_mode():
        expected = _historical_get_instance_id(reference, confidence, anchor)
    reference_rng = torch.get_rng_state().clone()
    torch.set_rng_state(initial_rng)
    with torch.inference_mode():
        actual = candidate.get_instance_id(confidence, anchor)
    candidate_rng = torch.get_rng_state().clone()

    _assert_value_equal(actual, expected)
    _assert_bank_runtime_equal(candidate, reference)
    assert torch.equal(reference_rng, candidate_rng)
    assert torch.equal(confidence.view(torch.uint8), confidence_before)
    assert torch.equal(anchor.view(torch.uint8), anchor_before)
    assert torch.equal(
        actual,
        torch.arange(1200, dtype=torch.long).reshape(4, 300),
    )


def test_fresh_non_temporal_ids_skip_confidence_and_scatter_graph():
    class Operations(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.names = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.names.append(str(func))
            return func(*args, **(kwargs or {}))

    confidence, anchor = _fixture()
    parent_log = Operations()
    candidate_log = Operations()
    with torch.inference_mode(), parent_log:
        _historical_get_instance_id(_bank(), confidence, anchor)
    with torch.inference_mode(), candidate_log:
        _bank().get_instance_id(confidence, anchor)

    assert any("max.dim" in name for name in parent_log.names)
    assert any("sigmoid" in name for name in parent_log.names)
    assert any("index_put" in name for name in parent_log.names)
    assert not any("max.dim" in name for name in candidate_log.names)
    assert not any("sigmoid" in name for name in candidate_log.names)
    assert not any("index_put" in name for name in candidate_log.names)
    assert sum("arange" in name for name in candidate_log.names) == 1


@pytest.mark.parametrize(
    "case",
    (
        "integer",
        "boolean",
        "complex",
        "quantized",
        "mkldnn",
        "sparse",
        "meta",
        "named",
    ),
)
def test_fresh_id_specialization_rejects_nonconventional_tensors(case):
    confidence, anchor = _fixture()
    if case == "integer":
        confidence = confidence.to(torch.int64)
    elif case == "boolean":
        confidence = confidence > 0
    elif case == "complex":
        confidence = confidence.to(torch.complex64)
    elif case == "quantized":
        confidence = torch.quantize_per_tensor(
            confidence,
            scale=0.1,
            zero_point=0,
            dtype=torch.qint8,
        )
    elif case == "mkldnn":
        confidence = confidence.to_mkldnn()
    elif case == "sparse":
        confidence = torch.sparse_coo_tensor(
            torch.empty((3, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
            size=confidence.shape,
        )
    elif case == "meta":
        confidence = torch.empty(confidence.shape, device="meta")
    elif case == "named":
        confidence = confidence.refine_names("batch", "query", "class")

    with torch.inference_mode():
        assert (
            _fresh_non_temporal_instance_id_shape(
                _bank(),
                confidence,
                anchor,
                None,
            )
            is None
        )


@pytest.mark.parametrize("dtype", (torch.int64, torch.bool))
def test_supported_nonfloating_ids_use_the_historical_route(dtype):
    confidence, anchor = _fixture()
    confidence = confidence.to(dtype)
    reference = _bank()
    candidate = _bank()
    operation_log = []

    class Operations(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            operation_log.append(str(func))
            return func(*args, **(kwargs or {}))

    with torch.inference_mode():
        expected = _historical_get_instance_id(reference, confidence, anchor)
    with torch.inference_mode(), Operations():
        actual = candidate.get_instance_id(confidence, anchor)

    _assert_value_equal(actual, expected)
    _assert_bank_runtime_equal(candidate, reference)
    assert any("max.dim" in name for name in operation_log)


@pytest.mark.parametrize(
    "case",
    (
        "training",
        "grad_enabled",
        "threshold",
        "cached_state",
        "existing_ids",
        "nonzero_prev_id",
        "query_count",
        "missing_anchor",
        "temporal",
        "subclass",
    ),
)
def test_fresh_id_specialization_fails_closed(case):
    class ExtendedInstanceBank(InstanceBank):
        pass

    bank_type = ExtendedInstanceBank if case == "subclass" else InstanceBank
    num_temporal = 1 if case == "temporal" else 0
    reference = _bank(num_temporal=num_temporal, bank_type=bank_type)
    candidate = _bank(num_temporal=num_temporal, bank_type=bank_type)
    queries = 299 if case == "query_count" else 300
    confidence, anchor = _fixture(queries=queries)
    threshold = 0.5 if case == "threshold" else None
    if case == "training":
        reference.train()
        candidate.train()
    if case == "cached_state":
        reference.cached_anchor = torch.zeros(1)
        candidate.cached_anchor = torch.zeros(1)
    if case == "existing_ids":
        ids = torch.zeros(4, queries, dtype=torch.long)
        reference.instance_id = ids.clone()
        candidate.instance_id = ids.clone()
    if case == "nonzero_prev_id":
        reference.prev_id = 17
        candidate.prev_id = 17
    if case == "missing_anchor":
        anchor = None

    context = nullcontext() if case == "grad_enabled" else torch.inference_mode()
    with context:
        expected = _historical_get_instance_id(
            reference,
            confidence,
            anchor,
            threshold,
        )
    operation_log = []

    class Operations(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            operation_log.append(str(func))
            return func(*args, **(kwargs or {}))

    context = nullcontext() if case == "grad_enabled" else torch.inference_mode()
    with context, Operations():
        actual = candidate.get_instance_id(confidence, anchor, threshold)

    _assert_value_equal(actual, expected)
    _assert_bank_runtime_equal(candidate, reference)
    assert any("max.dim" in name for name in operation_log)
