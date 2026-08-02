"""Structural contracts for checkpoints that may resume ROSA training.

The contract deliberately hashes tensor *schemas*, not tensor values.  The
private file-descriptor snapshot in :mod:`tools.run` already keeps validation
and MMEngine loading on the same bytes; these helpers make sure those bytes
contain every model tensor and the exact optimizer/scheduler topology expected
by the current runner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from typing import Any

import torch

_SCHEMA_VERSION = 1
_SCHEDULER_PROGRESS_FIELDS = frozenset({"last_step", "_global_step", "_last_value"})
_OPTIMIZER_DYNAMIC_GROUP_FIELDS = frozenset({"lr"})


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: Any) -> Any:
    """Return a JSON value that preserves container and scalar types."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("checkpoint contracts require finite floats")
        return {"float": value}
    if isinstance(value, torch.Tensor):
        return {
            "tensor": {
                "shape": [int(dimension) for dimension in value.shape],
                "dtype": str(value.dtype),
                "layout": str(value.layout),
            }
        }
    if isinstance(value, Mapping):
        records = []
        for key, item in value.items():
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                raise TypeError(
                    "checkpoint contract mapping keys must be strings or integers"
                )
            records.append(
                {
                    "key_type": type(key).__name__,
                    "key": key,
                    "value": _canonical(item),
                }
            )
        records.sort(key=lambda record: (record["key_type"], str(record["key"])))
        return {"mapping": records}
    if isinstance(value, tuple):
        return {"tuple": [_canonical(item) for item in value]}
    if isinstance(value, list):
        return {"list": [_canonical(item) for item in value]}
    raise TypeError(f"unsupported checkpoint contract value {type(value).__name__}")


def model_state_contract(state_dict: Mapping) -> dict[str, Any]:
    """Describe every model state key, shape, dtype, layout, and module version."""

    if not isinstance(state_dict, Mapping):
        raise TypeError("model state_dict must be a mapping")
    records = []
    elements = 0
    sidecar_keys = 0
    for key in sorted(state_dict):
        if not isinstance(key, str):
            raise TypeError("model state keys must be strings")
        tensor = state_dict[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"model state {key!r} is not a tensor")
        elements += int(tensor.numel())
        normalized = key.removeprefix("module.")
        if normalized.startswith("head.rosa_refiner."):
            sidecar_keys += 1
        records.append(
            {
                "key": key,
                "shape": [int(dimension) for dimension in tensor.shape],
                "dtype": str(tensor.dtype),
                "layout": str(tensor.layout),
            }
        )

    metadata = getattr(state_dict, "_metadata", None)
    metadata_record = None if metadata is None else _canonical(metadata)
    return {
        "version": _SCHEMA_VERSION,
        "key_count": len(records),
        "sidecar_key_count": sidecar_keys,
        "element_count": elements,
        "state_schema_sha256": _json_sha256(records),
        "module_metadata_sha256": (
            None if metadata_record is None else _json_sha256(metadata_record)
        ),
    }


def optimizer_state_contract(state_dict: Mapping) -> dict[str, Any]:
    """Describe optimizer groups and per-parameter state without hashing weights."""

    if not isinstance(state_dict, Mapping):
        raise TypeError("optimizer state must be a mapping")
    state = state_dict.get("state")
    groups = state_dict.get("param_groups")
    if not isinstance(state, Mapping):
        raise TypeError("optimizer state must contain a state mapping")
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)):
        raise TypeError("optimizer state must contain a param_groups sequence")

    parameter_ids: list[int] = []
    definitions = []
    group_counts = []
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            raise TypeError(f"optimizer param group {index} is not a mapping")
        params = group.get("params")
        if not isinstance(params, Sequence) or isinstance(params, (str, bytes)):
            raise TypeError(f"optimizer param group {index} has no params sequence")
        ids = []
        for parameter_id in params:
            if not isinstance(parameter_id, int) or isinstance(parameter_id, bool):
                raise TypeError("optimizer parameter ids must be exact integers")
            ids.append(parameter_id)
        parameter_ids.extend(ids)
        group_counts.append(len(ids))
        static = {
            key: value
            for key, value in group.items()
            if key not in _OPTIMIZER_DYNAMIC_GROUP_FIELDS and key != "params"
        }
        definitions.append(
            {
                "parameter_count": len(ids),
                "settings": _canonical(static),
            }
        )

    if len(parameter_ids) != len(set(parameter_ids)):
        raise ValueError("optimizer parameter ids are duplicated across groups")
    parameter_set = set(parameter_ids)
    state_ids = set(state)
    if any(
        not isinstance(value, int) or isinstance(value, bool) for value in state_ids
    ):
        raise TypeError("optimizer state parameter ids must be exact integers")
    if not state_ids.issubset(parameter_set):
        raise ValueError("optimizer state refers to parameters outside param_groups")

    state_records = [
        {
            "parameter_id": parameter_id,
            "state": _canonical(state[parameter_id]),
        }
        for parameter_id in sorted(state_ids)
    ]
    extra = {
        key: value
        for key, value in state_dict.items()
        if key not in {"state", "param_groups"}
    }
    base_settings = extra.get("base_param_settings")
    if isinstance(base_settings, Mapping):
        extra["base_param_settings"] = {
            key: value for key, value in base_settings.items() if key != "lr"
        }
    return {
        "version": _SCHEMA_VERSION,
        "group_count": len(groups),
        "group_parameter_counts": group_counts,
        "parameter_count": len(parameter_ids),
        "definition_sha256": _json_sha256(definitions),
        "state_parameter_count": len(state_ids),
        "state_schema_sha256": _json_sha256(state_records),
        "extra_schema_sha256": _json_sha256(_canonical(extra)),
    }


def _scheduler_states(payload: Any) -> list[tuple[str, Mapping]]:
    states: list[tuple[str, Mapping]] = []
    if isinstance(payload, Mapping):
        for name in sorted(payload):
            schedulers = payload[name]
            if not isinstance(name, str):
                raise TypeError("scheduler group names must be strings")
            if not isinstance(schedulers, Sequence) or isinstance(
                schedulers, (str, bytes)
            ):
                raise TypeError("scheduler groups must contain sequences")
            for index, scheduler in enumerate(schedulers):
                if not isinstance(scheduler, Mapping):
                    raise TypeError("scheduler state must be a mapping")
                states.append((f"{name}:{index}", scheduler))
        return states
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise TypeError("param_schedulers must be a sequence or mapping")
    for index, scheduler in enumerate(payload):
        if not isinstance(scheduler, Mapping):
            raise TypeError("scheduler state must be a mapping")
        states.append((str(index), scheduler))
    return states


def scheduler_state_contract(payload: Any) -> dict[str, Any]:
    """Describe scheduler definitions separately from their progress counters."""

    states = _scheduler_states(payload)
    full_records = []
    definitions = []
    horizons = []
    for name, state in states:
        full_records.append({"name": name, "state": _canonical(state)})
        definition = {
            key: value
            for key, value in state.items()
            if key not in _SCHEDULER_PROGRESS_FIELDS
        }
        definitions.append({"name": name, "state": _canonical(definition)})
        horizons.append(
            {
                "name": name,
                "begin": state.get("begin"),
                "end": state.get("end"),
                "T_max": state.get("T_max"),
                "total_iters": state.get("total_iters"),
                "by_epoch": state.get("by_epoch"),
                "param_name": state.get("param_name"),
            }
        )
    return {
        "version": _SCHEMA_VERSION,
        "scheduler_count": len(states),
        "definition_sha256": _json_sha256(definitions),
        "payload_sha256": _json_sha256(full_records),
        "horizons": horizons,
    }


def checkpoint_payload_contract(checkpoint: Mapping) -> dict[str, Any]:
    """Build the complete structural descriptor stored beside a checkpoint."""

    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint root must be a mapping")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise TypeError("checkpoint must contain a model state_dict mapping")
    optimizer = checkpoint.get("optimizer")
    schedulers = checkpoint.get("param_schedulers")
    return {
        "version": _SCHEMA_VERSION,
        "model_state": model_state_contract(state_dict),
        "optimizer_state": (
            None if optimizer is None else optimizer_state_contract(optimizer)
        ),
        "scheduler_state": (
            None if schedulers is None else scheduler_state_contract(schedulers)
        ),
    }


def runner_scheduler_payload(runner) -> Any:
    schedulers = runner.param_schedulers
    if isinstance(schedulers, Mapping):
        return {
            name: [scheduler.state_dict() for scheduler in values]
            for name, values in schedulers.items()
        }
    return [scheduler.state_dict() for scheduler in schedulers]


def validate_runner_model_contract(runner, expected: Mapping) -> None:
    """Require every checkpoint model tensor to exist in the current runner."""

    from mmengine.model import is_model_wrapper

    if not isinstance(expected, Mapping):
        raise TypeError("sealed checkpoint payload contract must be a mapping")
    model = runner.model.module if is_model_wrapper(runner.model) else runner.model
    observed_model = model_state_contract(model.state_dict())
    if observed_model != expected.get("model_state"):
        raise ValueError("resume checkpoint model state schema differs from runner")


def validate_runner_resume_contract(runner, expected: Mapping) -> None:
    """Fail before MMEngine's permissive loader can omit any resumed state."""

    validate_runner_model_contract(runner, expected)

    expected_optimizer = expected.get("optimizer_state")
    expected_scheduler = expected.get("scheduler_state")
    if not isinstance(expected_optimizer, Mapping):
        raise ValueError("resume checkpoint is missing optimizer state")
    if not isinstance(expected_scheduler, Mapping):
        raise ValueError("resume checkpoint is missing param_scheduler state")

    observed_optimizer = optimizer_state_contract(runner.optim_wrapper.state_dict())
    optimizer_definition_fields = (
        "version",
        "group_count",
        "group_parameter_counts",
        "parameter_count",
        "definition_sha256",
        "extra_schema_sha256",
    )
    if any(
        observed_optimizer.get(field) != expected_optimizer.get(field)
        for field in optimizer_definition_fields
    ):
        raise ValueError("resume checkpoint optimizer topology differs from runner")

    observed_scheduler = scheduler_state_contract(runner_scheduler_payload(runner))
    scheduler_definition_fields = (
        "version",
        "scheduler_count",
        "definition_sha256",
        "horizons",
    )
    if any(
        observed_scheduler.get(field) != expected_scheduler.get(field)
        for field in scheduler_definition_fields
    ):
        raise ValueError("resume checkpoint scheduler definition differs from runner")


__all__ = [
    "checkpoint_payload_contract",
    "model_state_contract",
    "optimizer_state_contract",
    "runner_scheduler_payload",
    "scheduler_state_contract",
    "validate_runner_model_contract",
    "validate_runner_resume_contract",
]
