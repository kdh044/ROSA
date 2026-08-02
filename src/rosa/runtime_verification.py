"""Pure contracts for the paired C32 runtime verification harness.

This module deliberately contains no CUDA discovery or model construction.  It
is safe to import in CPU-only tests and by the CLI's disarmed planning path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import operator
from pathlib import Path
import struct
from typing import Any, Iterable

import numpy as np
import torch


PARENT_ARM = "parent"
CANDIDATE_ARM = "candidate"
PCCP = "PCCP"
CPPC = "CPPC"
HOST_METADATA_KEYS = ("T_global", "T_global_inv", "timestamp")
EXPECTED_MODEL_TYPE = "Fisheye3DODGravityCenterSparse4D"
EXPECTED_HEAD_TYPE = "ROSAOfficialResidualNativeBaselineHead"
EXPECTED_PARENT_PREPROCESSOR = "Sparse4DDataPreprocessor"
EXPECTED_CANDIDATE_PREPROCESSOR = "Fisheye3DODSingleFrameDataPreprocessor"
NON_EXECUTABLE_CONFIG_KEYS = ("work_dir", "experiment_contract")


class VerificationContractError(ValueError):
    """Raised when a frozen paired-runtime contract is not satisfied."""


@dataclass(frozen=True)
class ScheduleEntry:
    """One model invocation in a preregistered alternating round."""

    phase: str
    round_index: int
    pattern: str
    batch_index: int
    slot_index: int
    arm: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _exact_integer(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise VerificationContractError(f"{name} must be an exact integer")
    try:
        observed = operator.index(value)
    except TypeError as error:
        raise VerificationContractError(
            f"{name} must be an exact integer"
        ) from error
    if observed < minimum:
        raise VerificationContractError(f"{name} must be >= {minimum}")
    return observed


def validate_runtime_parameters(
    *,
    capture_batches: int,
    warmup_rounds: int,
    measured_rounds: int,
    expected_batch_size: int,
    profile_batch_index: int,
) -> dict[str, int]:
    """Validate the immutable timing/capture shape before any execution."""

    capture_batches = _exact_integer(
        "capture_batches", capture_batches, minimum=1
    )
    warmup_rounds = _exact_integer("warmup_rounds", warmup_rounds, minimum=0)
    measured_rounds = _exact_integer(
        "measured_rounds", measured_rounds, minimum=2
    )
    expected_batch_size = _exact_integer(
        "expected_batch_size", expected_batch_size, minimum=1
    )
    profile_batch_index = _exact_integer(
        "profile_batch_index", profile_batch_index, minimum=0
    )
    if warmup_rounds % 2:
        raise VerificationContractError(
            "warmup_rounds must be even to balance PCCP and CPPC"
        )
    if measured_rounds % 2:
        raise VerificationContractError(
            "measured_rounds must be even to balance PCCP and CPPC"
        )
    if profile_batch_index >= capture_batches:
        raise VerificationContractError(
            "profile_batch_index must select a captured batch"
        )
    return {
        "capture_batches": capture_batches,
        "warmup_rounds": warmup_rounds,
        "measured_rounds": measured_rounds,
        "expected_batch_size": expected_batch_size,
        "profile_batch_index": profile_batch_index,
    }


def build_alternating_schedule(
    *,
    rounds: int,
    batch_count: int,
    phase: str,
) -> tuple[ScheduleEntry, ...]:
    """Return balanced same-batch PCCP/CPPC invocation order.

    Every captured batch is run in all four slots of a round before advancing
    to the next batch.  Even rounds use PCCP and odd rounds use CPPC.
    """

    rounds = _exact_integer("rounds", rounds, minimum=0)
    batch_count = _exact_integer("batch_count", batch_count, minimum=1)
    if rounds % 2:
        raise VerificationContractError(
            "rounds must be even to balance PCCP and CPPC"
        )
    if not isinstance(phase, str) or not phase:
        raise VerificationContractError("phase must be a nonempty string")
    entries: list[ScheduleEntry] = []
    arm_for_token = {"P": PARENT_ARM, "C": CANDIDATE_ARM}
    for round_index in range(rounds):
        pattern = PCCP if round_index % 2 == 0 else CPPC
        for batch_index in range(batch_count):
            for slot_index, token in enumerate(pattern):
                entries.append(
                    ScheduleEntry(
                        phase=phase,
                        round_index=round_index,
                        pattern=pattern,
                        batch_index=batch_index,
                        slot_index=slot_index,
                        arm=arm_for_token[token],
                    )
                )
    return tuple(entries)


def percentile(values: Sequence[float], quantile: float) -> float:
    """Linear-interpolated percentile with an explicit, testable definition."""

    if not values:
        raise VerificationContractError("percentile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise VerificationContractError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise VerificationContractError("percentile values must be finite")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def summarize_timings(
    records: Sequence[Mapping[str, Any]],
    *,
    arm: str,
) -> dict[str, float | int]:
    """Summarize raw nanosecond records without discarding the raw samples."""

    selected = [record for record in records if record.get("arm") == arm]
    if not selected:
        raise VerificationContractError(f"no timing records for {arm}")
    elapsed_ns: list[int] = []
    sample_count = 0
    for record in selected:
        elapsed = _exact_integer("elapsed_ns", record.get("elapsed_ns"), minimum=1)
        samples = _exact_integer(
            "sample_count", record.get("sample_count"), minimum=1
        )
        elapsed_ns.append(elapsed)
        sample_count += samples
    milliseconds = [value / 1_000_000.0 for value in elapsed_ns]
    total_seconds = sum(elapsed_ns) / 1_000_000_000.0
    return {
        "iterations": len(elapsed_ns),
        "samples": sample_count,
        "mean_iteration_ms": sum(milliseconds) / len(milliseconds),
        "median_iteration_ms": percentile(milliseconds, 0.5),
        "p95_iteration_ms": percentile(milliseconds, 0.95),
        "mean_latency_per_sample_ms": 1000.0 * total_seconds / sample_count,
        "samples_per_second": sample_count / total_seconds,
    }


def _mapping(config: Any) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    if not isinstance(config, Mapping):
        raise VerificationContractError("config must be a mapping")
    return copy.deepcopy(dict(config))


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise VerificationContractError(
                "config is missing " + ".".join(keys)
            )
        value = value[key]
    return value


def _require_zero_temporal_slots(config: Mapping[str, Any], label: str) -> None:
    raw = _nested(config, "model", "head", "instance_bank", "num_temp_instances")
    if isinstance(raw, bool):
        raise VerificationContractError(
            f"{label} num_temp_instances must be exact integer zero"
        )
    try:
        count = operator.index(raw)
    except TypeError as error:
        raise VerificationContractError(
            f"{label} num_temp_instances must be exact integer zero"
        ) from error
    if count != 0:
        raise VerificationContractError(
            f"{label} num_temp_instances must be exact integer zero"
        )


def validate_config_pair(
    parent_config: Any,
    candidate_config: Any,
    *,
    expected_batch_size: int,
) -> dict[str, Any]:
    """Prove that the only executable config delta is the preprocessor type."""

    expected_batch_size = _exact_integer(
        "expected_batch_size", expected_batch_size, minimum=1
    )
    parent = _mapping(parent_config)
    candidate = _mapping(candidate_config)
    for label, config in ((PARENT_ARM, parent), (CANDIDATE_ARM, candidate)):
        if _nested(config, "model", "type") != EXPECTED_MODEL_TYPE:
            raise VerificationContractError(
                f"{label} model must be {EXPECTED_MODEL_TYPE}"
            )
        if _nested(config, "model", "head", "type") != EXPECTED_HEAD_TYPE:
            raise VerificationContractError(
                f"{label} head must be {EXPECTED_HEAD_TYPE}"
            )
        _require_zero_temporal_slots(config, label)
        order = _nested(config, "model", "head", "operation_order")
        if not isinstance(order, (list, tuple)) or "temp_gnn" in order:
            raise VerificationContractError(
                f"{label} operation_order must be a temporal-free sequence"
            )
        batch_size = _nested(config, "test_dataloader", "batch_size")
        if batch_size != expected_batch_size:
            raise VerificationContractError(
                f"{label} test batch size {batch_size} != {expected_batch_size}"
            )
        if _nested(config, "test_dataloader", "sampler", "shuffle") is not False:
            raise VerificationContractError(
                f"{label} test sampler must set shuffle=False"
            )
        for forbidden in ("tta_model", "tta_pipeline"):
            if forbidden in config:
                raise VerificationContractError(
                    f"{label} config must not enable {forbidden}"
                )

    parent_preprocessor = _nested(parent, "model", "data_preprocessor", "type")
    candidate_preprocessor = _nested(
        candidate, "model", "data_preprocessor", "type"
    )
    if parent_preprocessor != EXPECTED_PARENT_PREPROCESSOR:
        raise VerificationContractError(
            f"parent preprocessor must be {EXPECTED_PARENT_PREPROCESSOR}"
        )
    if candidate_preprocessor != EXPECTED_CANDIDATE_PREPROCESSOR:
        raise VerificationContractError(
            f"candidate preprocessor must be {EXPECTED_CANDIDATE_PREPROCESSOR}"
        )

    normalized_parent = copy.deepcopy(parent)
    normalized_candidate = copy.deepcopy(candidate)
    normalized_candidate["model"]["data_preprocessor"]["type"] = (
        normalized_parent["model"]["data_preprocessor"]["type"]
    )
    for config in (normalized_parent, normalized_candidate):
        for key in NON_EXECUTABLE_CONFIG_KEYS:
            config.pop(key, None)
    if normalized_parent != normalized_candidate:
        raise VerificationContractError(
            "parent/candidate executable configs differ beyond the preprocessor type"
        )
    return {
        "model_type": EXPECTED_MODEL_TYPE,
        "head_type": EXPECTED_HEAD_TYPE,
        "parent_preprocessor": parent_preprocessor,
        "candidate_preprocessor": candidate_preprocessor,
        "test_batch_size": expected_batch_size,
        "normalized_config_sha256": canonical_sha256(normalized_parent),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise VerificationContractError(f"required file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _RawFingerprinter:
    def __init__(self) -> None:
        self.digest = hashlib.sha256()
        self.nodes = 0
        self.tensor_bytes = 0

    def _tag(self, value: str) -> None:
        encoded = value.encode("utf-8")
        self.digest.update(len(encoded).to_bytes(8, "little"))
        self.digest.update(encoded)

    def visit(self, value: Any, *, depth: int = 0) -> None:
        if depth > 64:
            raise VerificationContractError("raw fingerprint nesting is too deep")
        self.nodes += 1
        if value is None:
            self._tag("none")
            return
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided or value.is_quantized:
                raise VerificationContractError(
                    "raw fingerprint supports only dense non-quantized tensors"
                )
            self._tag("torch.Tensor")
            self._tag(str(value.dtype))
            self._tag(str(tuple(value.shape)))
            self._tag(str(tuple(value.stride())))
            cpu = value.detach().to(device="cpu").contiguous()
            raw = cpu.view(torch.uint8).reshape(-1).numpy().tobytes()
            self.tensor_bytes += len(raw)
            self.digest.update(len(raw).to_bytes(8, "little"))
            self.digest.update(raw)
            return
        if isinstance(value, np.ndarray):
            self._tag("numpy.ndarray")
            self._tag(value.dtype.str)
            self._tag(str(tuple(value.shape)))
            self._tag(str(tuple(value.strides)))
            raw = np.ascontiguousarray(value).tobytes()
            self.tensor_bytes += len(raw)
            self.digest.update(len(raw).to_bytes(8, "little"))
            self.digest.update(raw)
            return
        if isinstance(value, np.generic):
            self._tag(f"numpy.scalar:{value.dtype.str}")
            self.digest.update(value.tobytes())
            return
        if isinstance(value, Mapping):
            self._tag(f"mapping:{type(value).__module__}.{type(value).__qualname__}")
            for key, item in value.items():
                self.visit(key, depth=depth + 1)
                self.visit(item, depth=depth + 1)
            return
        if isinstance(value, (list, tuple)):
            self._tag(f"sequence:{type(value).__name__}:{len(value)}")
            for item in value:
                self.visit(item, depth=depth + 1)
            return
        if isinstance(value, (str, bytes, bool, int)):
            self._tag(f"scalar:{type(value).__name__}")
            raw = value if isinstance(value, bytes) else str(value).encode("utf-8")
            self.digest.update(len(raw).to_bytes(8, "little"))
            self.digest.update(raw)
            return
        if isinstance(value, float):
            self._tag("scalar:float64")
            self.digest.update(struct.pack("!d", value))
            return
        if hasattr(value, "tensor") and isinstance(value.tensor, torch.Tensor):
            self._tag(f"tensor-wrapper:{type(value).__module__}.{type(value).__qualname__}")
            self.visit(value.tensor, depth=depth + 1)
            return
        if hasattr(value, "to_dict") and callable(value.to_dict):
            self._tag(f"data-element:{type(value).__module__}.{type(value).__qualname__}")
            self.visit(value.to_dict(), depth=depth + 1)
            return
        raise VerificationContractError(
            f"unsupported raw fingerprint value: {type(value).__module__}."
            f"{type(value).__qualname__}"
        )


def raw_fingerprint(value: Any) -> dict[str, Any]:
    """Return a compact raw-bit digest of nested inference values."""

    visitor = _RawFingerprinter()
    visitor.visit(value)
    return {
        "sha256": visitor.digest.hexdigest(),
        "nodes": visitor.nodes,
        "tensor_bytes": visitor.tensor_bytes,
    }


def require_same_fingerprint(
    label: str,
    parent_value: Any,
    candidate_value: Any,
) -> dict[str, Any]:
    parent = raw_fingerprint(parent_value)
    candidate = raw_fingerprint(candidate_value)
    if parent != candidate:
        raise VerificationContractError(
            f"raw-bit mismatch for {label}: parent={parent}, candidate={candidate}"
        )
    return parent


def _sample_identity(sample: Any, index: int) -> dict[str, Any]:
    metainfo = getattr(sample, "metainfo", None)
    if metainfo is None and isinstance(sample, Mapping):
        metainfo = sample.get("metainfo")
    if not isinstance(metainfo, Mapping):
        raise VerificationContractError(
            f"data sample {index} has no metainfo mapping"
        )
    keys = ("sample_idx", "token", "scene_token", "frame_idx", "timestamp")
    identity = {key: metainfo[key] for key in keys if key in metainfo}
    if not identity:
        raise VerificationContractError(
            f"data sample {index} has no stable identity fields"
        )
    return identity


def build_batch_manifest(
    batch: Mapping[str, Any],
    *,
    batch_index: int,
    expected_batch_size: int,
) -> dict[str, Any]:
    """Hash one captured CPU batch and validate the exact metadata carrier."""

    if not isinstance(batch, Mapping):
        raise VerificationContractError("captured batch must be a mapping")
    inputs = batch.get("inputs")
    samples = batch.get("data_samples")
    if not isinstance(inputs, Mapping):
        raise VerificationContractError("captured batch inputs must be a mapping")
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise VerificationContractError(
            "captured batch data_samples must be a sequence"
        )
    if len(samples) != expected_batch_size:
        raise VerificationContractError(
            f"captured batch {batch_index} has {len(samples)} samples, expected "
            f"{expected_batch_size}"
        )
    for name in HOST_METADATA_KEYS:
        value = inputs.get(name)
        if not isinstance(value, torch.Tensor):
            raise VerificationContractError(f"captured {name} must be a tensor")
        if value.device.type != "cpu":
            raise VerificationContractError(f"captured {name} must be on CPU")
        if value.ndim == 0 or value.shape[0] != expected_batch_size:
            raise VerificationContractError(
                f"captured {name} must have batch dimension {expected_batch_size}"
            )
        if not value.is_contiguous():
            raise VerificationContractError(f"captured {name} must be contiguous")
    identities = [
        _sample_identity(sample, index) for index, sample in enumerate(samples)
    ]
    input_fingerprints = {
        name: raw_fingerprint(value) for name, value in inputs.items()
    }
    manifest = {
        "batch_index": batch_index,
        "batch_size": len(samples),
        "sample_identities": identities,
        "input_fingerprints": input_fingerprints,
    }
    manifest["sha256"] = canonical_sha256(manifest)
    return manifest


def validate_expected_manifest(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    if canonical_sha256(observed) != canonical_sha256(expected):
        raise VerificationContractError("captured sample manifest does not match expected")


def classify_transfer_event(name: str) -> str | None:
    normalized = " ".join(str(name).lower().replace("_", " ").split())
    if "memcpy" not in normalized and "mem copy" not in normalized:
        return None
    compact = normalized.replace(" ", "")
    if "dtoh" in compact or "devicetohost" in compact or "device->host" in compact:
        return "device_to_host"
    if "htod" in compact or "hosttodevice" in compact or "host->device" in compact:
        return "host_to_device"
    if "dtod" in compact or "devicetodevice" in compact:
        return "device_to_device"
    return "unclassified_memcpy"


def summarize_transfer_event_names(names: Iterable[str]) -> dict[str, Any]:
    counts = {
        "device_to_host": 0,
        "host_to_device": 0,
        "device_to_device": 0,
        "unclassified_memcpy": 0,
    }
    matching: dict[str, int] = {}
    for name in names:
        kind = classify_transfer_event(name)
        lowered = str(name).lower()
        if kind is not None:
            counts[kind] += 1
            matching[str(name)] = matching.get(str(name), 0) + 1
        elif "item" in lowered or "local_scalar_dense" in lowered:
            matching[str(name)] = matching.get(str(name), 0) + 1
    return {"counts": counts, "matching_event_names": matching}


def validate_transfer_reduction(
    parent_summary: Mapping[str, Any],
    candidate_summary: Mapping[str, Any],
    *,
    minimum_device_to_host: int,
    minimum_host_to_device: int,
) -> dict[str, int]:
    """Fail unless the profiler observes the preregistered transfer reduction."""

    minimum_device_to_host = _exact_integer(
        "minimum_device_to_host", minimum_device_to_host, minimum=0
    )
    minimum_host_to_device = _exact_integer(
        "minimum_host_to_device", minimum_host_to_device, minimum=0
    )
    parent_counts = _nested(parent_summary, "counts")
    candidate_counts = _nested(candidate_summary, "counts")
    reductions = {
        "device_to_host": int(parent_counts["device_to_host"])
        - int(candidate_counts["device_to_host"]),
        "host_to_device": int(parent_counts["host_to_device"])
        - int(candidate_counts["host_to_device"]),
    }
    if reductions["device_to_host"] < minimum_device_to_host:
        raise VerificationContractError(
            "profiler D2H reduction is below the preregistered minimum: "
            f"{reductions['device_to_host']} < {minimum_device_to_host}"
        )
    if reductions["host_to_device"] < minimum_host_to_device:
        raise VerificationContractError(
            "profiler H2D reduction is below the preregistered minimum: "
            f"{reductions['host_to_device']} < {minimum_host_to_device}"
        )
    return reductions


__all__ = [
    "CANDIDATE_ARM",
    "CPPC",
    "EXPECTED_CANDIDATE_PREPROCESSOR",
    "EXPECTED_HEAD_TYPE",
    "EXPECTED_MODEL_TYPE",
    "EXPECTED_PARENT_PREPROCESSOR",
    "HOST_METADATA_KEYS",
    "PARENT_ARM",
    "PCCP",
    "ScheduleEntry",
    "VerificationContractError",
    "build_alternating_schedule",
    "build_batch_manifest",
    "canonical_json",
    "canonical_sha256",
    "classify_transfer_event",
    "file_sha256",
    "percentile",
    "raw_fingerprint",
    "require_same_fingerprint",
    "summarize_timings",
    "summarize_transfer_event_names",
    "validate_config_pair",
    "validate_expected_manifest",
    "validate_runtime_parameters",
    "validate_transfer_reduction",
]
