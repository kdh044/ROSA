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
CPU_METADATA_PROTOCOL = "cpu-metadata"
REPRESENTATION_PROTOCOL = "lean-loader-uint8-pinned"
C0_C32_NEAR_NATIVE_PROTOCOL = "c0-c32-near-native"
MODEL_ONLY_SCOPE = "model-only-common-preprocessor-boundary"
INPUT_PIPELINE_SCOPE = "end-to-end-loader-collate-transfer"
C0_C32_FULL_DETECTOR_SCOPE = "full-detector-test-step"
C0_C32_COMMON_MODEL_SCOPE = "common-boundary-raw-model-predict"
PCCP = "PCCP"
CPPC = "CPPC"
PHYSICAL_ROUND_ORDER = (PCCP, CPPC, PCCP, CPPC, PCCP, CPPC)
PHYSICAL_ROUND_COUNT = len(PHYSICAL_ROUND_ORDER)
INVOCATIONS_PER_ARM_PER_BLOCK = 2
ITERATIONS_PER_ARM_GRANULARITY = PHYSICAL_ROUND_COUNT * INVOCATIONS_PER_ARM_PER_BLOCK
MINIMUM_WARMUP_ITERATIONS_PER_ARM = 50
MINIMUM_MEASURED_ITERATIONS_PER_ARM = 300
DEFAULT_WARMUP_ITERATIONS_PER_ARM = 60
DEFAULT_MEASURED_ITERATIONS_PER_ARM = 300
BOOTSTRAP_SEED = 3407
BOOTSTRAP_SAMPLES = 20_000
STRICT_MEDIAN_REDUCTION_MINIMUM = 0.02
STRICT_MEDIAN_RATIO_CI95_UPPER_EXCLUSIVE = 1.0
STRICT_P95_REGRESSION_MAXIMUM = 0.01
STRICT_P95_RATIO_MAXIMUM = 1.0 + STRICT_P95_REGRESSION_MAXIMUM
NEAR_NATIVE_MEDIAN_RATIO_CI95_UPPER_MAXIMUM = 1.10
NEAR_NATIVE_P95_RATIO_MAXIMUM = 1.10
HOST_METADATA_KEYS = ("T_global", "T_global_inv", "timestamp")
EXPECTED_MODEL_TYPE = "Fisheye3DODGravityCenterSparse4D"
EXPECTED_HEAD_TYPE = "ROSAOfficialResidualNativeBaselineHead"
EXPECTED_C0_HEAD_TYPE = "ROSANativeBaselineHead"
EXPECTED_PARENT_PREPROCESSOR = "Sparse4DDataPreprocessor"
EXPECTED_CANDIDATE_PREPROCESSOR = "Fisheye3DODSingleFrameDataPreprocessor"
EXPECTED_REPRESENTATION_PREPROCESSOR = "Fisheye3DODSingleFrameDataPreprocessor"
EXPECTED_LEAN_LOADER = "LoadOfficialFisheye3DODLeanForSparse4D"
EXPECTED_PARENT_PACKER = "PackSparse4DInputs"
EXPECTED_CANDIDATE_PACKER = "PackFisheye3DODUint8Inputs"
EXPECTED_RUNTIME_TRANSFORM_IMPORT = "rosa.integrations.fisheye3dod.runtime_transforms"
EXPECTED_CUMULATIVE_PREPROCESSOR = "Fisheye3DODSingleFrameInplaceDataPreprocessor"
EXPECTED_CUMULATIVE_LOADER = "LoadOfficialFisheye3DODLeanForSparse4D"
EXPECTED_CUMULATIVE_PACKER = "PackFisheye3DODRGBUint8Inputs"
EXPECTED_CUMULATIVE_DEFORMABLE_MODEL = (
    "ROSAInferenceReuseExactHomogeneous4x4FullFOVDeformableFeatureAggregation"
)
REPRESENTATION_STREAM_POLICY = "torch-current-default-stream-only"
MODEL_ONLY_MEDIAN_RATIO_CI95_UPPER_MAXIMUM = 1.01
MODEL_ONLY_P95_RATIO_MAXIMUM = 1.01
NON_EXECUTABLE_CONFIG_KEYS = ("work_dir", "experiment_contract")
C0_C32_PEAK_MEMORY_RATIO_MAXIMUM = 1.02
_MAIN126_ANCHOR_RELATIVE_PATH = (
    "assets/fisheye3dod_main126_train_kmeans300_seed3407.npy"
)
_MAIN126_ANCHOR_SHA256 = (
    "2a08ac6a28b6eb519e59e9d9c16bae009184785f2b8745cd857576a28496b812"
)

# Representation-runtime v2 is deliberately a single immutable experiment.
# These are equality constraints, not user-tunable defaults.  Keeping them in
# the CUDA-free contract module lets the dry-run reject protocol drift before
# a model, dataloader, or accelerator API is touched.
REPRESENTATION_CAPTURE_BATCHES = 2
REPRESENTATION_WARMUP_ITERATIONS_PER_ARM = 60
REPRESENTATION_MEASURED_ITERATIONS_PER_ARM = 300
REPRESENTATION_EXPECTED_BATCH_SIZE = 4
REPRESENTATION_DATALOADER_WORKERS = 0
REPRESENTATION_SEED = 3407
REPRESENTATION_SEQUENCE_PAIRS_PER_ARM = (
    REPRESENTATION_WARMUP_ITERATIONS_PER_ARM
    + REPRESENTATION_MEASURED_ITERATIONS_PER_ARM
)
REPRESENTATION_C32_E20_CHECKPOINT_SHA256 = (
    "cb5b0956ab8357d64bbcde919d8aa10aa772d03c73252c137709529e6299e85d"
)
REPRESENTATION_PARENT_NORMALIZED_CONFIG_SHA256 = (
    "8204c92a8287c67a425bad6c41c3ad16fb0e0d13c4de7ad9cdbd68be47a43147"
)
REPRESENTATION_CANDIDATE_NORMALIZED_CONFIG_SHA256 = (
    "b0590a50c17894b0c6f3f839c377a637c21d49a75906dd7a22a271e4e61e8a51"
)
REPRESENTATION_PREREGISTRATION_SCHEMA = (
    "rosa.c32_representation_runtime_preregistration.v2"
)
REPRESENTATION_SEQUENCE_SCHEMA = "rosa.live_representation_pair_sequence.v2"
REPRESENTATION_BOUNDARY_RECEIPT_SCHEMA = (
    "rosa.timed_common_preprocessor_boundary_receipt.v3"
)
POINT_PAYLOAD_INPUT_KEYS = frozenset(
    {
        "points",
        "lidar_points",
        "semantic_lidar_points",
        "semantic_points",
        "raw_points",
        "point_cloud",
        "pointcloud",
        "pts",
    }
)


class VerificationContractError(ValueError):
    """Raised when a frozen paired-runtime contract is not satisfied."""


@dataclass(frozen=True)
class ScheduleEntry:
    """One model invocation in a preregistered alternating round."""

    phase: str
    round_index: int
    pattern: str
    slot_index: int
    segment_iteration_index: int
    batch_index: int
    pair_index: int
    arm: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _exact_integer(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise VerificationContractError(f"{name} must be an exact integer")
    try:
        observed = operator.index(value)
    except TypeError as error:
        raise VerificationContractError(f"{name} must be an exact integer") from error
    if observed < minimum:
        raise VerificationContractError(f"{name} must be >= {minimum}")
    return observed


def validate_runtime_parameters(
    *,
    capture_batches: int,
    warmup_iterations_per_arm: int,
    measured_iterations_per_arm: int,
    expected_batch_size: int,
    profile_batch_index: int,
) -> dict[str, int]:
    """Validate the immutable timing/capture shape before any execution."""

    capture_batches = _exact_integer("capture_batches", capture_batches, minimum=1)
    warmup_iterations_per_arm = _exact_integer(
        "warmup_iterations_per_arm",
        warmup_iterations_per_arm,
        minimum=MINIMUM_WARMUP_ITERATIONS_PER_ARM,
    )
    measured_iterations_per_arm = _exact_integer(
        "measured_iterations_per_arm",
        measured_iterations_per_arm,
        minimum=MINIMUM_MEASURED_ITERATIONS_PER_ARM,
    )
    expected_batch_size = _exact_integer(
        "expected_batch_size", expected_batch_size, minimum=1
    )
    profile_batch_index = _exact_integer(
        "profile_batch_index", profile_batch_index, minimum=0
    )
    if warmup_iterations_per_arm % ITERATIONS_PER_ARM_GRANULARITY:
        raise VerificationContractError(
            "warmup_iterations_per_arm must be a multiple of "
            f"{ITERATIONS_PER_ARM_GRANULARITY} so all six physical rounds "
            "have equal PCCP/CPPC arm segments"
        )
    if measured_iterations_per_arm % ITERATIONS_PER_ARM_GRANULARITY:
        raise VerificationContractError(
            "measured_iterations_per_arm must be a multiple of "
            f"{ITERATIONS_PER_ARM_GRANULARITY} so all six physical rounds "
            "have equal PCCP/CPPC arm segments"
        )
    if profile_batch_index >= capture_batches:
        raise VerificationContractError(
            "profile_batch_index must select a captured batch"
        )
    return {
        "capture_batches": capture_batches,
        "warmup_iterations_per_arm": warmup_iterations_per_arm,
        "measured_iterations_per_arm": measured_iterations_per_arm,
        "physical_rounds": PHYSICAL_ROUND_COUNT,
        "warmup_iterations_per_segment": (
            warmup_iterations_per_arm // ITERATIONS_PER_ARM_GRANULARITY
        ),
        "measured_iterations_per_segment": (
            measured_iterations_per_arm // ITERATIONS_PER_ARM_GRANULARITY
        ),
        "expected_batch_size": expected_batch_size,
        "profile_batch_index": profile_batch_index,
    }


def validate_representation_protocol_freeze(
    *,
    parameters: Mapping[str, Any],
    dataloader_workers: int,
    seed: int,
    checkpoint_sha256: str,
    config_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject every deviation from the one preregistered representation run.

    This is intentionally stricter than :func:`validate_runtime_parameters`:
    values that are useful CLI options for the older CPU-metadata protocol are
    not tuning knobs in the representation experiment.
    """

    expected_parameters = {
        "capture_batches": REPRESENTATION_CAPTURE_BATCHES,
        "warmup_iterations_per_arm": (REPRESENTATION_WARMUP_ITERATIONS_PER_ARM),
        "measured_iterations_per_arm": (REPRESENTATION_MEASURED_ITERATIONS_PER_ARM),
        "expected_batch_size": REPRESENTATION_EXPECTED_BATCH_SIZE,
    }
    if not isinstance(parameters, Mapping):
        raise VerificationContractError("representation parameters must be a mapping")
    for name, expected in expected_parameters.items():
        if parameters.get(name) != expected:
            raise VerificationContractError(
                f"representation protocol freezes {name}={expected}"
            )
    if dataloader_workers != REPRESENTATION_DATALOADER_WORKERS:
        raise VerificationContractError(
            "representation protocol freezes dataloader_workers=0"
        )
    if seed != REPRESENTATION_SEED:
        raise VerificationContractError(
            f"representation protocol freezes seed={REPRESENTATION_SEED}"
        )
    if checkpoint_sha256 != REPRESENTATION_C32_E20_CHECKPOINT_SHA256:
        raise VerificationContractError(
            "representation protocol requires the exact official C32 e20 "
            "checkpoint SHA256"
        )
    if not isinstance(config_contract, Mapping):
        raise VerificationContractError(
            "representation config contract must be a mapping"
        )
    parent_digest = config_contract.get("parent_normalized_config_sha256")
    candidate_digest = config_contract.get("candidate_normalized_config_sha256")
    if parent_digest != REPRESENTATION_PARENT_NORMALIZED_CONFIG_SHA256:
        raise VerificationContractError(
            "representation parent normalized config digest changed"
        )
    if candidate_digest != REPRESENTATION_CANDIDATE_NORMALIZED_CONFIG_SHA256:
        raise VerificationContractError(
            "representation candidate normalized config digest changed"
        )
    return {
        "capture_batches": REPRESENTATION_CAPTURE_BATCHES,
        "warmup_iterations_per_arm": (REPRESENTATION_WARMUP_ITERATIONS_PER_ARM),
        "measured_iterations_per_arm": (REPRESENTATION_MEASURED_ITERATIONS_PER_ARM),
        "sequence_pairs_per_arm": REPRESENTATION_SEQUENCE_PAIRS_PER_ARM,
        "expected_batch_size": REPRESENTATION_EXPECTED_BATCH_SIZE,
        "dataloader_workers": REPRESENTATION_DATALOADER_WORKERS,
        "seed": REPRESENTATION_SEED,
        "checkpoint_sha256": REPRESENTATION_C32_E20_CHECKPOINT_SHA256,
        "parent_normalized_config_sha256": (
            REPRESENTATION_PARENT_NORMALIZED_CONFIG_SHA256
        ),
        "candidate_normalized_config_sha256": (
            REPRESENTATION_CANDIDATE_NORMALIZED_CONFIG_SHA256
        ),
    }


def build_alternating_schedule(
    *,
    iterations_per_arm: int,
    batch_count: int,
    phase: str,
) -> tuple[ScheduleEntry, ...]:
    """Return six physical rounds of contiguous P/C arm segments.

    A physical round contains four contiguous arm segments in its PCCP or CPPC
    order.  Matched iteration indices in adjacent segments replay the same
    captured batch and form a pair.  Equal segment lengths require per-arm
    iterations to be a multiple of twelve.
    """

    iterations_per_arm = _exact_integer(
        "iterations_per_arm", iterations_per_arm, minimum=1
    )
    batch_count = _exact_integer("batch_count", batch_count, minimum=1)
    if iterations_per_arm % ITERATIONS_PER_ARM_GRANULARITY:
        raise VerificationContractError(
            "iterations_per_arm must be a multiple of "
            f"{ITERATIONS_PER_ARM_GRANULARITY}"
        )
    if not isinstance(phase, str) or not phase:
        raise VerificationContractError("phase must be a nonempty string")
    entries: list[ScheduleEntry] = []
    arm_for_token = {"P": PARENT_ARM, "C": CANDIDATE_ARM}
    iterations_per_segment = iterations_per_arm // ITERATIONS_PER_ARM_GRANULARITY
    for round_index, pattern in enumerate(PHYSICAL_ROUND_ORDER):
        pairs_per_round = 2 * iterations_per_segment
        for slot_index, token in enumerate(pattern):
            pair_group = 0 if slot_index < 2 else 1
            for segment_iteration_index in range(iterations_per_segment):
                pair_index = (
                    round_index * pairs_per_round
                    + pair_group * iterations_per_segment
                    + segment_iteration_index
                )
                entries.append(
                    ScheduleEntry(
                        phase=phase,
                        round_index=round_index,
                        pattern=pattern,
                        slot_index=slot_index,
                        segment_iteration_index=segment_iteration_index,
                        batch_index=pair_index % batch_count,
                        pair_index=pair_index,
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
        samples = _exact_integer("sample_count", record.get("sample_count"), minimum=1)
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


def _timing_pairs(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_pairs: int,
    required_pair_fields: Sequence[str] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected_pairs = _exact_integer("expected_pairs", expected_pairs, minimum=1)
    if expected_pairs % ITERATIONS_PER_ARM_GRANULARITY:
        raise VerificationContractError(
            "expected_pairs must preserve equal six-round arm segments"
        )
    grouped: dict[int, dict[str, Mapping[str, Any]]] = {}
    for record in records:
        arm = record.get("arm")
        if arm not in (PARENT_ARM, CANDIDATE_ARM):
            raise VerificationContractError("timing record has an invalid arm")
        pair_index = _exact_integer("pair_index", record.get("pair_index"), minimum=0)
        pair = grouped.setdefault(pair_index, {})
        if arm in pair:
            raise VerificationContractError(
                f"timing pair {pair_index} repeats arm {arm}"
            )
        pair[arm] = record
    if sorted(grouped) != list(range(expected_pairs)):
        raise VerificationContractError(
            f"timing pairs must be contiguous 0..{expected_pairs - 1}"
        )

    parent_ns: list[int] = []
    candidate_ns: list[int] = []
    sample_counts: list[int] = []
    round_pair_counts = [0] * PHYSICAL_ROUND_COUNT
    if isinstance(required_pair_fields, (str, bytes)):
        raise VerificationContractError(
            "required_pair_fields must be a sequence of field names"
        )
    normalized_required_fields = tuple(required_pair_fields)
    if any(
        not isinstance(name, str) or not name for name in normalized_required_fields
    ):
        raise VerificationContractError(
            "required_pair_fields must contain nonempty strings"
        )
    comparable_fields = (
        "phase",
        "round_index",
        "pattern",
        "segment_iteration_index",
        "batch_index",
        "pair_index",
    ) + normalized_required_fields
    for pair_index in range(expected_pairs):
        pair = grouped[pair_index]
        if set(pair) != {PARENT_ARM, CANDIDATE_ARM}:
            raise VerificationContractError(
                f"timing pair {pair_index} must contain exactly one P and one C"
            )
        parent = pair[PARENT_ARM]
        candidate = pair[CANDIDATE_ARM]
        for name in normalized_required_fields:
            if name not in parent or name not in candidate:
                raise VerificationContractError(
                    f"timing pair {pair_index} is missing required field {name}"
                )
        if any(parent.get(name) != candidate.get(name) for name in comparable_fields):
            raise VerificationContractError(
                f"timing pair {pair_index} does not share schedule metadata"
            )
        round_index = _exact_integer(
            "round_index", parent.get("round_index"), minimum=0
        )
        if round_index >= PHYSICAL_ROUND_COUNT:
            raise VerificationContractError("timing record has an invalid round")
        if parent.get("pattern") != PHYSICAL_ROUND_ORDER[round_index]:
            raise VerificationContractError(
                "timing record violates the physical PCCP/CPPC order"
            )
        if parent.get("phase") != "measured":
            raise VerificationContractError(
                "paired bootstrap accepts measured timing records only"
            )
        pairs_per_round = expected_pairs // PHYSICAL_ROUND_COUNT
        iterations_per_segment = pairs_per_round // 2
        pair_position_in_round = pair_index % pairs_per_round
        first_slot = 0 if pair_position_in_round < iterations_per_segment else 2
        expected_slots = {first_slot, first_slot + 1}
        observed_slots = {
            _exact_integer("slot_index", parent.get("slot_index"), minimum=0),
            _exact_integer("slot_index", candidate.get("slot_index"), minimum=0),
        }
        if observed_slots != expected_slots:
            raise VerificationContractError(
                f"timing pair {pair_index} does not match adjacent arm segments"
            )
        parent_slot = operator.index(parent["slot_index"])
        candidate_slot = operator.index(candidate["slot_index"])
        pattern = PHYSICAL_ROUND_ORDER[round_index]
        if pattern[parent_slot] != "P" or pattern[candidate_slot] != "C":
            raise VerificationContractError(
                f"timing pair {pair_index} assigns arms to the wrong segment"
            )
        parent_elapsed = _exact_integer(
            "elapsed_ns", parent.get("elapsed_ns"), minimum=1
        )
        candidate_elapsed = _exact_integer(
            "elapsed_ns", candidate.get("elapsed_ns"), minimum=1
        )
        parent_samples = _exact_integer(
            "sample_count", parent.get("sample_count"), minimum=1
        )
        candidate_samples = _exact_integer(
            "sample_count", candidate.get("sample_count"), minimum=1
        )
        if parent_samples != candidate_samples:
            raise VerificationContractError(
                f"timing pair {pair_index} has different sample counts"
            )
        parent_ns.append(parent_elapsed)
        candidate_ns.append(candidate_elapsed)
        sample_counts.append(parent_samples)
        round_pair_counts[round_index] += 1

    expected_per_round, remainder = divmod(expected_pairs, PHYSICAL_ROUND_COUNT)
    if remainder or round_pair_counts != [expected_per_round] * PHYSICAL_ROUND_COUNT:
        raise VerificationContractError(
            "timing pairs must be equally distributed across all six rounds"
        )
    return (
        np.asarray(parent_ns, dtype=np.float64),
        np.asarray(candidate_ns, dtype=np.float64),
        np.asarray(sample_counts, dtype=np.float64),
    )


def _confidence_interval(values: Sequence[float]) -> dict[str, float]:
    return {
        "lower": percentile(values, 0.025),
        "upper": percentile(values, 0.975),
    }


def paired_bootstrap_confidence_intervals(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_pairs: int,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    required_pair_fields: Sequence[str] = (),
) -> dict[str, Any]:
    """Compute deterministic percentile CIs by resampling paired P/C calls."""

    bootstrap_samples = _exact_integer(
        "bootstrap_samples", bootstrap_samples, minimum=1
    )
    seed = _exact_integer("bootstrap seed", seed, minimum=0)
    parent_ns, candidate_ns, sample_counts = _timing_pairs(
        records,
        expected_pairs=expected_pairs,
        required_pair_fields=required_pair_fields,
    )
    point_estimates = {
        "median_candidate_over_parent": (
            percentile(candidate_ns.tolist(), 0.5) / percentile(parent_ns.tolist(), 0.5)
        ),
        "p95_candidate_over_parent": (
            percentile(candidate_ns.tolist(), 0.95)
            / percentile(parent_ns.tolist(), 0.95)
        ),
        "throughput_candidate_over_parent": (
            float(parent_ns.sum()) / float(candidate_ns.sum())
        ),
    }
    pairs_per_round = len(parent_ns) // PHYSICAL_ROUND_COUNT
    per_round = []
    for round_index, pattern in enumerate(PHYSICAL_ROUND_ORDER):
        start = round_index * pairs_per_round
        stop = start + pairs_per_round
        round_parent = parent_ns[start:stop]
        round_candidate = candidate_ns[start:stop]
        per_round.append(
            {
                "round_index": round_index,
                "pattern": pattern,
                "pairs": pairs_per_round,
                "median_candidate_over_parent": (
                    percentile(round_candidate.tolist(), 0.5)
                    / percentile(round_parent.tolist(), 0.5)
                ),
                "p95_candidate_over_parent": (
                    percentile(round_candidate.tolist(), 0.95)
                    / percentile(round_parent.tolist(), 0.95)
                ),
                "throughput_candidate_over_parent": (
                    float(round_parent.sum()) / float(round_candidate.sum())
                ),
            }
        )

    generated = np.random.Generator(np.random.PCG64(seed))
    bootstrap_values: dict[str, list[float]] = {name: [] for name in point_estimates}
    remaining = bootstrap_samples
    chunk_size = 512
    while remaining:
        count = min(chunk_size, remaining)
        within_round_indices = generated.integers(
            0,
            pairs_per_round,
            size=(count, PHYSICAL_ROUND_COUNT, pairs_per_round),
            endpoint=False,
        )
        round_offsets = (
            np.arange(PHYSICAL_ROUND_COUNT, dtype=np.int64).reshape(
                1, PHYSICAL_ROUND_COUNT, 1
            )
            * pairs_per_round
        )
        indices = (within_round_indices + round_offsets).reshape(count, len(parent_ns))
        parent_samples_ns = parent_ns[indices]
        candidate_samples_ns = candidate_ns[indices]
        sampled_counts = sample_counts[indices]
        parent_medians = np.median(parent_samples_ns, axis=1)
        candidate_medians = np.median(candidate_samples_ns, axis=1)
        parent_p95 = np.quantile(parent_samples_ns, 0.95, axis=1, method="linear")
        candidate_p95 = np.quantile(candidate_samples_ns, 0.95, axis=1, method="linear")
        sample_totals = sampled_counts.sum(axis=1)
        parent_throughput = sample_totals / parent_samples_ns.sum(axis=1)
        candidate_throughput = sample_totals / candidate_samples_ns.sum(axis=1)
        values = {
            "median_candidate_over_parent": candidate_medians / parent_medians,
            "p95_candidate_over_parent": candidate_p95 / parent_p95,
            "throughput_candidate_over_parent": (
                candidate_throughput / parent_throughput
            ),
        }
        for name, array in values.items():
            bootstrap_values[name].extend(float(value) for value in array)
        remaining -= count

    metrics = {}
    for name, point in point_estimates.items():
        metrics[name] = {
            "point_estimate": point,
            "ci95": _confidence_interval(bootstrap_values[name]),
        }
    return {
        "method": (
            "six-round-stratified paired percentile bootstrap of matched "
            "iterations in adjacent P/C arm segments"
        ),
        "rng": "numpy.random.PCG64",
        "seed": seed,
        "bootstrap_samples": bootstrap_samples,
        "pairs": len(parent_ns),
        "pairs_per_round": pairs_per_round,
        "physical_round_order": list(PHYSICAL_ROUND_ORDER),
        "required_pair_fields": list(required_pair_fields),
        "per_round_point_estimates": per_round,
        "metrics": metrics,
    }


def evaluate_strict_lightweight_gate(
    paired_bootstrap: Mapping[str, Any],
    memory: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the preregistered strict lightweight adoption decision."""

    metrics = _nested(paired_bootstrap, "metrics")
    median_ratio = float(
        _nested(metrics, "median_candidate_over_parent", "point_estimate")
    )
    median_upper = float(
        _nested(metrics, "median_candidate_over_parent", "ci95", "upper")
    )
    p95_ratio = float(_nested(metrics, "p95_candidate_over_parent", "point_estimate"))
    parent_memory = _nested(memory, PARENT_ARM)
    candidate_memory = _nested(memory, CANDIDATE_ARM)
    parent_peak_allocated = _exact_integer(
        "parent peak_allocated_bytes",
        _nested(parent_memory, "peak_allocated_bytes"),
        minimum=0,
    )
    candidate_peak_allocated = _exact_integer(
        "candidate peak_allocated_bytes",
        _nested(candidate_memory, "peak_allocated_bytes"),
        minimum=0,
    )
    parent_peak_reserved = _exact_integer(
        "parent peak_reserved_bytes",
        _nested(parent_memory, "peak_reserved_bytes"),
        minimum=0,
    )
    candidate_peak_reserved = _exact_integer(
        "candidate peak_reserved_bytes",
        _nested(candidate_memory, "peak_reserved_bytes"),
        minimum=0,
    )
    if not all(
        math.isfinite(value) for value in (median_ratio, median_upper, p95_ratio)
    ):
        raise VerificationContractError("decision metrics must be finite")
    checks = {
        "median_reduction_at_least_2_percent": {
            "observed_candidate_over_parent": median_ratio,
            "threshold_max_inclusive": 1.0 - STRICT_MEDIAN_REDUCTION_MINIMUM,
            "passed": median_ratio <= 1.0 - STRICT_MEDIAN_REDUCTION_MINIMUM,
        },
        "median_ratio_ci95_upper_below_1": {
            "observed": median_upper,
            "threshold_max_exclusive": STRICT_MEDIAN_RATIO_CI95_UPPER_EXCLUSIVE,
            "passed": median_upper < STRICT_MEDIAN_RATIO_CI95_UPPER_EXCLUSIVE,
        },
        "p95_regression_at_most_1_percent": {
            "observed_candidate_over_parent": p95_ratio,
            "threshold_max_inclusive": STRICT_P95_RATIO_MAXIMUM,
            "passed": p95_ratio <= STRICT_P95_RATIO_MAXIMUM,
        },
        "peak_allocated_memory_not_worse": {
            "parent_bytes": parent_peak_allocated,
            "candidate_bytes": candidate_peak_allocated,
            "passed": candidate_peak_allocated <= parent_peak_allocated,
        },
        "peak_reserved_memory_not_worse": {
            "parent_bytes": parent_peak_reserved,
            "candidate_bytes": candidate_peak_reserved,
            "passed": candidate_peak_reserved <= parent_peak_reserved,
        },
    }
    passed = all(check["passed"] for check in checks.values())
    return {
        "gate": "strict_lightweight_adoption",
        "passed": passed,
        "decision": (
            "ADOPT_CANDIDATE" if passed else "RETAIN_PARENT_AND_REPORT_INCONCLUSIVE"
        ),
        "checks": checks,
    }


def evaluate_c32_near_native_gate(
    paired_bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the preregistered C32/C0 near-native statement gate."""

    metrics = _nested(paired_bootstrap, "metrics")
    median_upper = float(
        _nested(metrics, "median_candidate_over_parent", "ci95", "upper")
    )
    p95_ratio = float(_nested(metrics, "p95_candidate_over_parent", "point_estimate"))
    if not all(math.isfinite(value) for value in (median_upper, p95_ratio)):
        raise VerificationContractError("near-native metrics must be finite")
    checks = {
        "median_ratio_ci95_upper_at_most_1_10": {
            "observed": median_upper,
            "threshold_max_inclusive": (NEAR_NATIVE_MEDIAN_RATIO_CI95_UPPER_MAXIMUM),
            "passed": (median_upper <= NEAR_NATIVE_MEDIAN_RATIO_CI95_UPPER_MAXIMUM),
        },
        "p95_ratio_at_most_1_10": {
            "observed_candidate_over_parent": p95_ratio,
            "threshold_max_inclusive": NEAR_NATIVE_P95_RATIO_MAXIMUM,
            "passed": p95_ratio <= NEAR_NATIVE_P95_RATIO_MAXIMUM,
        },
    }
    passed = all(check["passed"] for check in checks.values())
    return {
        "gate": "c32_vs_c0_near_native_statement",
        "passed": passed,
        "decision": (
            "NEAR_NATIVE_SUPPORTED" if passed else "NEAR_NATIVE_NOT_SUPPORTED"
        ),
        "checks": checks,
    }


def evaluate_c0_c32_near_native_gate(
    paired_bootstrap: Mapping[str, Any],
    memory: Mapping[str, Any],
) -> dict[str, Any]:
    """Gate a real C32/C0 full-detector comparison, including memory.

    This differs from :func:`evaluate_c32_near_native_gate`, which predates a
    true C0 arm and therefore cannot authorize a near-native claim on its own.
    Peak allocated and reserved memory are both allowed at most two percent
    regression, as frozen by this protocol before any CUDA measurement.
    """

    latency_gate = evaluate_c32_near_native_gate(paired_bootstrap)
    parent_memory = _nested(memory, PARENT_ARM)
    candidate_memory = _nested(memory, CANDIDATE_ARM)
    memory_checks: dict[str, Any] = {}
    for field in ("peak_allocated_bytes", "peak_reserved_bytes"):
        parent_value = _exact_integer(
            f"parent {field}", _nested(parent_memory, field), minimum=1
        )
        candidate_value = _exact_integer(
            f"candidate {field}",
            _nested(candidate_memory, field),
            minimum=1,
        )
        ratio = candidate_value / parent_value
        memory_checks[f"{field}_regression_at_most_2_percent"] = {
            "parent_bytes": parent_value,
            "candidate_bytes": candidate_value,
            "candidate_over_parent": ratio,
            "threshold_max_inclusive": C0_C32_PEAK_MEMORY_RATIO_MAXIMUM,
            "passed": ratio <= C0_C32_PEAK_MEMORY_RATIO_MAXIMUM,
        }
    checks = {
        **latency_gate["checks"],
        **memory_checks,
    }
    passed = all(check["passed"] for check in checks.values())
    return {
        "gate": "c32_vs_c0_full_detector_near_native",
        "benchmark_scope": C0_C32_FULL_DETECTOR_SCOPE,
        "passed": passed,
        "decision": (
            "NEAR_NATIVE_SUPPORTED"
            if passed
            else "NEAR_NATIVE_NOT_SUPPORTED"
        ),
        "speed_claim_eligible": passed,
        "checks": checks,
    }


def evaluate_model_only_non_regression_gate(
    paired_bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    """Gate only common-boundary model math; it is not a speed-adoption gate."""

    metrics = _nested(paired_bootstrap, "metrics")
    median_upper = float(
        _nested(metrics, "median_candidate_over_parent", "ci95", "upper")
    )
    p95_ratio = float(_nested(metrics, "p95_candidate_over_parent", "point_estimate"))
    if not all(math.isfinite(value) for value in (median_upper, p95_ratio)):
        raise VerificationContractError("model-only decision metrics must be finite")
    checks = {
        "median_ratio_ci95_upper_at_most_1_01": {
            "observed": median_upper,
            "threshold_max_inclusive": (MODEL_ONLY_MEDIAN_RATIO_CI95_UPPER_MAXIMUM),
            "passed": (median_upper <= MODEL_ONLY_MEDIAN_RATIO_CI95_UPPER_MAXIMUM),
        },
        "p95_ratio_at_most_1_01": {
            "observed_candidate_over_parent": p95_ratio,
            "threshold_max_inclusive": MODEL_ONLY_P95_RATIO_MAXIMUM,
            "passed": p95_ratio <= MODEL_ONLY_P95_RATIO_MAXIMUM,
        },
    }
    passed = all(check["passed"] for check in checks.values())
    return {
        "gate": "model_only_common_boundary_non_regression",
        "benchmark_scope": MODEL_ONLY_SCOPE,
        "passed": passed,
        "decision": "MODEL_ONLY_NON_REGRESSION" if passed else "INCONCLUSIVE",
        "speed_claim_eligible": False,
        "checks": checks,
    }


def evaluate_input_pipeline_gate(
    paired_bootstrap: Mapping[str, Any],
    representation_pairs: Sequence[Mapping[str, Any]],
    *,
    common_boundary_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Gate loader+collate+transfer separately from model execution."""

    if not isinstance(representation_pairs, Sequence) or not representation_pairs:
        raise VerificationContractError(
            "input-pipeline gate requires representation pair manifests"
        )
    boundary_contract = validate_timed_common_boundary_receipt(
        common_boundary_receipt,
        representation_pairs,
    )
    metrics = _nested(paired_bootstrap, "metrics")
    median_ratio = float(
        _nested(metrics, "median_candidate_over_parent", "point_estimate")
    )
    median_upper = float(
        _nested(metrics, "median_candidate_over_parent", "ci95", "upper")
    )
    p95_ratio = float(_nested(metrics, "p95_candidate_over_parent", "point_estimate"))
    if not all(
        math.isfinite(value) for value in (median_ratio, median_upper, p95_ratio)
    ):
        raise VerificationContractError("input-pipeline metrics must be finite")
    byte_records = []
    for pair in representation_pairs:
        if pair.get("schema") != "rosa.representation_pair.v1":
            raise VerificationContractError(
                "input-pipeline gate received an invalid pair manifest"
            )
        parent_bytes = _exact_integer(
            "parent image bytes",
            _nested(pair, "arms", PARENT_ARM, "ingress", "image_bytes"),
            minimum=1,
        )
        candidate_bytes = _exact_integer(
            "candidate image bytes",
            _nested(pair, "arms", CANDIDATE_ARM, "ingress", "image_bytes"),
            minimum=1,
        )
        byte_records.append(
            {
                "batch_index": pair.get("batch_index"),
                "parent": parent_bytes,
                "candidate": candidate_bytes,
                "passed": candidate_bytes < parent_bytes,
            }
        )
    checks = {
        "common_preprocessor_boundary_raw_bit_parity": {
            "receipt_sha256": boundary_contract["sha256"],
            "timed_pairs": boundary_contract["pairs"],
            "passed": True,
        },
        "candidate_ingress_bytes_strictly_lower": {
            "batches": byte_records,
            "passed": all(record["passed"] for record in byte_records),
        },
        "median_reduction_at_least_2_percent": {
            "observed_candidate_over_parent": median_ratio,
            "threshold_max_inclusive": 1.0 - STRICT_MEDIAN_REDUCTION_MINIMUM,
            "passed": median_ratio <= 1.0 - STRICT_MEDIAN_REDUCTION_MINIMUM,
        },
        "median_ratio_ci95_upper_below_1": {
            "observed": median_upper,
            "threshold_max_exclusive": STRICT_MEDIAN_RATIO_CI95_UPPER_EXCLUSIVE,
            "passed": median_upper < STRICT_MEDIAN_RATIO_CI95_UPPER_EXCLUSIVE,
        },
        "p95_regression_at_most_1_percent": {
            "observed_candidate_over_parent": p95_ratio,
            "threshold_max_inclusive": STRICT_P95_RATIO_MAXIMUM,
            "passed": p95_ratio <= STRICT_P95_RATIO_MAXIMUM,
        },
    }
    passed = all(check["passed"] for check in checks.values())
    return {
        "gate": "end_to_end_input_pipeline_adoption",
        "benchmark_scope": INPUT_PIPELINE_SCOPE,
        "passed": passed,
        "decision": (
            "INPUT_PIPELINE_CANDIDATE_ELIGIBLE"
            if passed
            else "RETAIN_PARENT_AND_REPORT_INCONCLUSIVE"
        ),
        "speed_claim_eligible": passed,
        "scope_excludes_model_execution": True,
        "checks": checks,
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
            raise VerificationContractError("config is missing " + ".".join(keys))
        value = value[key]
    return value


def validate_preregistration_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the harness to the exact v1 future-CUDA measurement contract."""

    if not isinstance(plan, Mapping):
        raise VerificationContractError("preregistration plan must be a mapping")
    expected_schema = "rosa.exacte94_lightweight_fresh30_runtime_preregistration.v1"
    if plan.get("schema") != expected_schema:
        raise VerificationContractError("unexpected preregistration schema")
    if plan.get("status") != "PREPARATION_ONLY_DISARMED":
        raise VerificationContractError("preregistration status is not frozen")
    if plan.get("execution_authorized") is not False:
        raise VerificationContractError("preregistration must remain non-authorizing")
    gate = _nested(plan, "future_cuda_gate")
    if _nested(gate, "correctness_before_timing") is not True:
        raise VerificationContractError(
            "preregistration must require correctness before timing"
        )
    model_benchmark = _nested(gate, "model_test_step_benchmark")
    if _nested(model_benchmark, "warmup_min") != MINIMUM_WARMUP_ITERATIONS_PER_ARM:
        raise VerificationContractError("preregistration warmup minimum changed")
    if (
        _nested(model_benchmark, "measured_iterations_per_arm_min")
        != MINIMUM_MEASURED_ITERATIONS_PER_ARM
    ):
        raise VerificationContractError(
            "preregistration measured-iteration minimum changed"
        )
    observed_order = tuple(_nested(gate, "alternating_round_order"))
    if observed_order != PHYSICAL_ROUND_ORDER:
        raise VerificationContractError(
            "preregistration physical PCCP/CPPC order changed"
        )
    required_reporting = _nested(gate, "required_reporting")
    expected_reporting = {
        "raw per-iteration samples",
        "paired median",
        "p95",
        "throughput",
        "95% paired-bootstrap confidence interval",
        "allocated/reserved/peak memory",
        "full environment and hashes",
    }
    if (
        not isinstance(required_reporting, (list, tuple))
        or set(required_reporting) != expected_reporting
    ):
        raise VerificationContractError(
            "preregistration reporting requirements changed"
        )
    expected_environment_pins = {
        "GPU model",
        "driver",
        "CUDA",
        "PyTorch",
        "precision/autocast",
        "TF32/matmul settings",
        "determinism",
        "source/config/checkpoint hashes",
        "captured sample IDs",
        "batch size",
        "clock/power observations",
        "no concurrent GPU workload",
    }
    environment_pins = _nested(gate, "environment_pins")
    if (
        not isinstance(environment_pins, (list, tuple))
        or set(environment_pins) != expected_environment_pins
    ):
        raise VerificationContractError("preregistration environment pins changed")
    strict_gate = _nested(gate, "strict_lightweight_adoption_gate")
    expected_gate = {
        "same_checkpoint_and_exact_outputs": True,
        "median_reduction_min_fraction": STRICT_MEDIAN_REDUCTION_MINIMUM,
        "upper_95ci_candidate_over_parent_ratio_max_exclusive": (
            STRICT_MEDIAN_RATIO_CI95_UPPER_EXCLUSIVE
        ),
        "p95_regression_max_fraction": STRICT_P95_REGRESSION_MAXIMUM,
        "peak_memory_regression_allowed": False,
    }
    if dict(strict_gate) != expected_gate:
        raise VerificationContractError(
            "preregistration strict-lightweight gate changed"
        )
    near_native_gate = _nested(gate, "c32_vs_c0_near_native_gate")
    expected_near_native_gate = {
        "same_checkpoint_and_inputs": True,
        "upper_95ci_median_ratio_max": (NEAR_NATIVE_MEDIAN_RATIO_CI95_UPPER_MAXIMUM),
        "p95_ratio_max": NEAR_NATIVE_P95_RATIO_MAXIMUM,
    }
    if dict(near_native_gate) != expected_near_native_gate:
        raise VerificationContractError(
            "preregistration C32/C0 near-native gate changed"
        )
    if gate.get("historical_062d236_random_weight_gpu_numbers_eligible") is not False:
        raise VerificationContractError(
            "historical random-weight GPU timings must remain ineligible"
        )
    return {
        "schema": expected_schema,
        "execution_authorized": False,
        "correctness_before_timing": True,
        "minimum_warmup_iterations_per_arm": MINIMUM_WARMUP_ITERATIONS_PER_ARM,
        "minimum_measured_iterations_per_arm": (MINIMUM_MEASURED_ITERATIONS_PER_ARM),
        "physical_round_order": list(PHYSICAL_ROUND_ORDER),
        "environment_pins": sorted(expected_environment_pins),
        "strict_lightweight_gate": expected_gate,
        "c32_vs_c0_near_native_gate": expected_near_native_gate,
        "bootstrap": {
            "method": (
                "six-round-stratified paired percentile bootstrap of matched "
                "iterations in adjacent P/C arm segments"
            ),
            "seed": BOOTSTRAP_SEED,
            "samples": BOOTSTRAP_SAMPLES,
        },
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_representation_sequence_document(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the immutable 360-pair arm-specific preregistration payload."""

    if not isinstance(document, Mapping):
        raise VerificationContractError(
            "representation sequence manifest must be a mapping"
        )
    if document.get("schema") != REPRESENTATION_SEQUENCE_SCHEMA:
        raise VerificationContractError(
            "unexpected representation sequence manifest schema"
        )
    if document.get("protocol") != REPRESENTATION_PROTOCOL:
        raise VerificationContractError(
            "representation sequence manifest protocol changed"
        )
    if document.get("pairs_per_arm") != REPRESENTATION_SEQUENCE_PAIRS_PER_ARM:
        raise VerificationContractError(
            "representation sequence manifest must contain exactly 360 pairs"
        )
    pairs = document.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != (
        REPRESENTATION_SEQUENCE_PAIRS_PER_ARM
    ):
        raise VerificationContractError(
            "representation sequence pair payload is incomplete"
        )
    observed_identities: set[str] = set()
    sequence_input_keys = None
    for pair_index, pair in enumerate(pairs):
        if not isinstance(pair, Mapping):
            raise VerificationContractError(
                f"representation sequence pair {pair_index} is not a mapping"
            )
        if pair.get("schema") != "rosa.representation_pair.v1":
            raise VerificationContractError(
                f"representation sequence pair {pair_index} schema changed"
            )
        if pair.get("batch_index") != pair_index:
            raise VerificationContractError(
                "representation sequence pair indices are not contiguous"
            )
        if not _is_sha256(pair.get("semantic_sha256")):
            raise VerificationContractError(
                f"representation sequence pair {pair_index} lacks semantic SHA"
            )
        arms = pair.get("arms")
        if not isinstance(arms, Mapping) or set(arms) != {
            PARENT_ARM,
            CANDIDATE_ARM,
        }:
            raise VerificationContractError(
                f"representation sequence pair {pair_index} arms changed"
            )
        pair_input_keys = {}
        for arm in (PARENT_ARM, CANDIDATE_ARM):
            arm_manifest = arms[arm]
            if not isinstance(arm_manifest, Mapping):
                raise VerificationContractError(
                    f"representation sequence pair {pair_index} {arm} "
                    "manifest is invalid"
                )
            if arm_manifest.get("arm") != arm:
                raise VerificationContractError(
                    f"representation sequence pair {pair_index} arm label changed"
                )
            if arm_manifest.get("batch_index") != pair_index:
                raise VerificationContractError(
                    "representation arm manifest batch index changed"
                )
            if arm_manifest.get("semantic_sha256") != pair.get("semantic_sha256"):
                raise VerificationContractError(
                    "representation arm semantic digest changed"
                )
            fingerprint = arm_manifest.get("representation_fingerprint")
            if not isinstance(fingerprint, Mapping) or not _is_sha256(
                fingerprint.get("sha256")
            ):
                raise VerificationContractError(
                    "representation arm fingerprint is incomplete"
                )
            if not _is_sha256(arm_manifest.get("sha256")):
                raise VerificationContractError(
                    "representation arm manifest SHA is incomplete"
                )
            arm_payload = {
                key: value
                for key, value in arm_manifest.items()
                if key != "sha256"
            }
            if arm_manifest["sha256"] != canonical_sha256(arm_payload):
                raise VerificationContractError(
                    "representation arm manifest SHA mismatch"
                )
            if arm_manifest.get("sample_identities") != pair.get(
                "sample_identities"
            ):
                raise VerificationContractError(
                    "representation arm sample identities changed"
                )
            input_keys = arm_manifest.get("input_keys")
            if (
                not isinstance(input_keys, list)
                or input_keys != sorted(input_keys)
                or "img" not in input_keys
                or any(
                    key in POINT_PAYLOAD_INPUT_KEYS
                    for key in input_keys
                )
            ):
                raise VerificationContractError(
                    "representation arm input-key/LiDAR scope changed"
                )
            pair_input_keys[arm] = input_keys
        if pair_input_keys[PARENT_ARM] != pair_input_keys[CANDIDATE_ARM]:
            raise VerificationContractError(
                "representation pair arms expose different input keys"
            )
        if sequence_input_keys is None:
            sequence_input_keys = pair_input_keys[PARENT_ARM]
        elif pair_input_keys[PARENT_ARM] != sequence_input_keys:
            raise VerificationContractError(
                "representation sequence input-key set changed between pairs"
            )
        if not _is_sha256(pair.get("sha256")):
            raise VerificationContractError(
                f"representation sequence pair {pair_index} SHA is missing"
            )
        if (
            canonical_sha256(
                {key: value for key, value in pair.items() if key != "sha256"}
            )
            != pair["sha256"]
        ):
            raise VerificationContractError(
                f"representation sequence pair {pair_index} SHA mismatch"
            )
        identities = pair.get("sample_identities")
        if not isinstance(identities, list) or len(identities) != (
            REPRESENTATION_EXPECTED_BATCH_SIZE
        ):
            raise VerificationContractError(
                "representation sequence sample identities are incomplete"
            )
        for identity in identities:
            digest = canonical_sha256(identity)
            if digest in observed_identities:
                raise VerificationContractError(
                    "representation sequence repeats a sample identity"
                )
            observed_identities.add(digest)
    expected_document_sha = canonical_sha256(pairs)
    if document.get("sha256") != expected_document_sha:
        raise VerificationContractError("representation sequence document SHA mismatch")
    return {
        "schema": REPRESENTATION_SEQUENCE_SCHEMA,
        "pairs_per_arm": REPRESENTATION_SEQUENCE_PAIRS_PER_ARM,
        "samples": len(observed_identities),
        "input_keys": sequence_input_keys,
        "sha256": expected_document_sha,
    }


def validate_representation_preregistration_plan(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Accept a disarmed placeholder, but mark only a fully sealed plan ready."""

    if not isinstance(plan, Mapping):
        raise VerificationContractError(
            "representation preregistration plan must be a mapping"
        )
    if plan.get("schema") != REPRESENTATION_PREREGISTRATION_SCHEMA:
        raise VerificationContractError(
            "unexpected representation preregistration schema"
        )
    if plan.get("protocol") != REPRESENTATION_PROTOCOL:
        raise VerificationContractError(
            "representation preregistration protocol changed"
        )
    if plan.get("execution_authorized") is not False:
        raise VerificationContractError(
            "representation preregistration is never execution authorization"
        )
    status = plan.get("status")
    if status not in ("PLACEHOLDER_DISARMED", "SEALED"):
        raise VerificationContractError(
            "representation preregistration status is invalid"
        )
    frozen = plan.get("frozen_contract")
    expected_frozen = {
        "capture_batches": REPRESENTATION_CAPTURE_BATCHES,
        "warmup_iterations_per_arm": (REPRESENTATION_WARMUP_ITERATIONS_PER_ARM),
        "measured_iterations_per_arm": (REPRESENTATION_MEASURED_ITERATIONS_PER_ARM),
        "sequence_pairs_per_arm": REPRESENTATION_SEQUENCE_PAIRS_PER_ARM,
        "expected_batch_size": REPRESENTATION_EXPECTED_BATCH_SIZE,
        "dataloader_workers": REPRESENTATION_DATALOADER_WORKERS,
        "seed": REPRESENTATION_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "physical_round_order": list(PHYSICAL_ROUND_ORDER),
        "checkpoint_sha256": REPRESENTATION_C32_E20_CHECKPOINT_SHA256,
        "parent_normalized_config_sha256": (
            REPRESENTATION_PARENT_NORMALIZED_CONFIG_SHA256
        ),
        "candidate_normalized_config_sha256": (
            REPRESENTATION_CANDIDATE_NORMALIZED_CONFIG_SHA256
        ),
    }
    if frozen != expected_frozen:
        raise VerificationContractError(
            "representation preregistration frozen contract changed"
        )
    required_sections = (
        "sequence_manifest",
        "rosa_source_closure",
        "detector_host_source_closure",
        "official_source_closure",
        "dataset_payload_closure",
    )
    sections = {}
    for name in required_sections:
        section = plan.get(name)
        if not isinstance(section, Mapping):
            raise VerificationContractError(
                f"representation preregistration lacks {name}"
            )
        section_status = section.get("status")
        if section_status not in ("MISSING", "SEALED"):
            raise VerificationContractError(
                f"representation preregistration {name} status is invalid"
            )
        if section_status == "SEALED":
            if not isinstance(section.get("file"), str) or not _is_sha256(
                section.get("sha256")
            ):
                raise VerificationContractError(
                    f"representation preregistration {name} seal is incomplete"
                )
        sections[name] = section_status
    ready = status == "SEALED" and all(value == "SEALED" for value in sections.values())
    if status == "SEALED" and not ready:
        raise VerificationContractError(
            "SEALED representation preregistration has missing closures"
        )
    return {
        "schema": REPRESENTATION_PREREGISTRATION_SCHEMA,
        "status": status,
        "ready_for_armed_execution": ready,
        "sections": sections,
        "frozen_contract": expected_frozen,
    }


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
    candidate_preprocessor = _nested(candidate, "model", "data_preprocessor", "type")
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
    normalized_candidate["model"]["data_preprocessor"]["type"] = normalized_parent[
        "model"
    ]["data_preprocessor"]["type"]
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


def _normalize_main126_anchor_for_digest(config: dict[str, Any]) -> None:
    """Replace a worktree-local anchor path with its content identity."""

    anchor_file = config.get("anchor_file")
    instance_anchor = _nested(config, "model", "head", "instance_bank", "anchor")
    if not isinstance(anchor_file, str) or not isinstance(instance_anchor, str):
        raise VerificationContractError("Main126 anchor paths must be strings")
    anchor_path = Path(anchor_file).expanduser().resolve()
    if anchor_path != Path(instance_anchor).expanduser().resolve():
        raise VerificationContractError(
            "top-level and instance-bank Main126 anchor paths differ"
        )
    if (
        anchor_path.name != Path(_MAIN126_ANCHOR_RELATIVE_PATH).name
        or anchor_path.parent.name != "assets"
        or not anchor_path.is_file()
    ):
        raise VerificationContractError(
            "Main126 anchor must resolve to the released project asset"
        )
    digest = hashlib.sha256(anchor_path.read_bytes()).hexdigest()
    if digest != _MAIN126_ANCHOR_SHA256:
        raise VerificationContractError("Main126 anchor content hash changed")
    identity = {
        "project_relative_path": _MAIN126_ANCHOR_RELATIVE_PATH,
        "sha256": digest,
    }
    config["anchor_file"] = copy.deepcopy(identity)
    config["model"]["head"]["instance_bank"]["anchor"] = copy.deepcopy(identity)


def validate_c0_c32_near_native_config_pair(
    parent_config: Any,
    candidate_config: Any,
    *,
    expected_batch_size: int,
) -> dict[str, Any]:
    """Prove that C0 and C32 differ only by the audited head class.

    The parent is exact C0 ``ROSANativeBaselineHead`` and the candidate is
    exact C32 ``ROSAOfficialResidualNativeBaselineHead``.  Both retain the
    same R18 detector, cumulative uint8 runtime path, exact OCam projection,
    operation order, and temporal-free instance bank.  Non-executable
    experiment bookkeeping is intentionally excluded from the equality proof.
    """

    expected_batch_size = _exact_integer(
        "expected_batch_size", expected_batch_size, minimum=1
    )
    parent = _mapping(parent_config)
    candidate = _mapping(candidate_config)
    expected_heads = {
        PARENT_ARM: EXPECTED_C0_HEAD_TYPE,
        CANDIDATE_ARM: EXPECTED_HEAD_TYPE,
    }
    for label, config in ((PARENT_ARM, parent), (CANDIDATE_ARM, candidate)):
        if _nested(config, "model", "type") != EXPECTED_MODEL_TYPE:
            raise VerificationContractError(
                f"{label} model must be {EXPECTED_MODEL_TYPE}"
            )
        observed_head = _nested(config, "model", "head", "type")
        if observed_head != expected_heads[label]:
            raise VerificationContractError(
                f"{label} head must be {expected_heads[label]}"
            )
        backbone = _nested(config, "model", "img_backbone")
        if (
            _nested(backbone, "type") != "mmdet.ResNet"
            or _nested(backbone, "depth") != 18
        ):
            raise VerificationContractError(
                f"{label} must use the frozen mmdet.ResNet depth-18 backbone"
            )
        init_cfg = _nested(backbone, "init_cfg")
        if (
            _nested(init_cfg, "type") != "Pretrained"
            or _nested(init_cfg, "checkpoint") != "torchvision://resnet18"
        ):
            raise VerificationContractError(
                f"{label} must retain the ImageNet ResNet-18 initialization contract"
            )
        _require_zero_temporal_slots(config, label)
        order = _nested(config, "model", "head", "operation_order")
        if not isinstance(order, (list, tuple)):
            raise VerificationContractError(
                f"{label} operation_order must be an ordered sequence"
            )
        if "temp_gnn" in order or list(order).count("gnn") != 5:
            raise VerificationContractError(
                f"{label} must contain exactly five temporal-free GNN stages"
            )
        if any(
            index + 1 >= len(order) or order[index + 1] != "norm"
            for index, operation in enumerate(order)
            if operation == "gnn"
        ):
            raise VerificationContractError(
                f"{label} must retain post-GNN LayerNorm at every stage"
            )
        dataloader = _nested(config, "test_dataloader")
        if _nested(dataloader, "batch_size") != expected_batch_size:
            raise VerificationContractError(
                f"{label} test batch size must equal {expected_batch_size}"
            )
        if _nested(dataloader, "sampler", "shuffle") is not False:
            raise VerificationContractError(
                f"{label} test sampler must set shuffle=False"
            )
        dataset = _nested(dataloader, "dataset")
        if (
            _nested(dataset, "type") != "Fisheye3DODSparse4DDataset"
            or _nested(dataset, "test_mode") is not True
        ):
            raise VerificationContractError(
                f"{label} must use the Main126 Fisheye3DOD test dataset"
            )
        if not str(_nested(dataset, "ann_file")).endswith(
            "/ImageSets-main126/fisheye3dod_infos_val.pkl"
        ):
            raise VerificationContractError(
                f"{label} must use the Main126 validation annotations"
            )
        pipeline = _nested(dataset, "pipeline")
        if not isinstance(pipeline, list) or len(pipeline) != 2:
            raise VerificationContractError(
                f"{label} test pipeline must contain the frozen loader and packer"
            )
        if (
            not isinstance(pipeline[0], Mapping)
            or pipeline[0].get("type")
            != EXPECTED_CUMULATIVE_LOADER
            or pipeline[0].get("to_float32") is not False
            or not isinstance(pipeline[1], Mapping)
            or pipeline[1].get("type") != EXPECTED_CUMULATIVE_PACKER
        ):
            raise VerificationContractError(
                f"{label} must retain the frozen cumulative uint8 input pipeline"
            )
        if _nested(config, "model", "data_preprocessor", "type") != (
            EXPECTED_CUMULATIVE_PREPROCESSOR
        ):
            raise VerificationContractError(
                f"{label} must use {EXPECTED_CUMULATIVE_PREPROCESSOR}"
            )
        preprocessor = _nested(config, "model", "data_preprocessor")
        if (
            preprocessor.get("non_blocking") is not True
            or preprocessor.get("bgr_to_rgb") is not False
            or any(
                key in preprocessor
                for key in ("stream", "cuda_stream", "transfer_stream")
            )
        ):
            raise VerificationContractError(
                f"{label} must retain nonblocking default-stream RGB preprocessing"
            )
        if dataloader.get("pin_memory") is not True:
            raise VerificationContractError(
                f"{label} cumulative test dataloader must pin host memory"
            )
        if _nested(config, "model", "head", "deformable_model", "type") != (
            EXPECTED_CUMULATIVE_DEFORMABLE_MODEL
        ):
            raise VerificationContractError(
                f"{label} must use the exact homogeneous OCam reuse path"
            )
        for forbidden in ("tta_model", "tta_pipeline"):
            if forbidden in config:
                raise VerificationContractError(
                    f"{label} config must not enable {forbidden}"
                )

    normalized_parent = copy.deepcopy(parent)
    normalized_candidate = copy.deepcopy(candidate)
    normalized_candidate["model"]["head"]["type"] = EXPECTED_C0_HEAD_TYPE
    for config in (normalized_parent, normalized_candidate):
        for key in NON_EXECUTABLE_CONFIG_KEYS:
            config.pop(key, None)
        _normalize_main126_anchor_for_digest(config)
    if normalized_parent != normalized_candidate:
        raise VerificationContractError(
            "C0/C32 executable configs differ beyond the audited head type"
        )
    normalized_preprocessor = _nested(
        normalized_parent, "model", "data_preprocessor"
    )
    return {
        "protocol": C0_C32_NEAR_NATIVE_PROTOCOL,
        "model_type": EXPECTED_MODEL_TYPE,
        "parent_head_type": EXPECTED_C0_HEAD_TYPE,
        "candidate_head_type": EXPECTED_HEAD_TYPE,
        "backbone": "mmdet.ResNet-18",
        "input_dtype": "torch.uint8",
        "loader": EXPECTED_CUMULATIVE_LOADER,
        "packer": EXPECTED_CUMULATIVE_PACKER,
        "preprocessor": EXPECTED_CUMULATIVE_PREPROCESSOR,
        "deformable_model": EXPECTED_CUMULATIVE_DEFORMABLE_MODEL,
        "preprocessor_sha256": canonical_sha256(normalized_preprocessor),
        "test_batch_size": expected_batch_size,
        "gnn_stages": 5,
        "normalized_config_sha256": canonical_sha256(normalized_parent),
        "only_executable_delta": "model.head.type",
    }


_REPRESENTATION_PIPELINE_PATHS = (
    ("train_dataloader", "dataset", "dataset", "pipeline"),
    ("val_dataloader", "dataset", "pipeline"),
    ("test_dataloader", "dataset", "pipeline"),
)


def _pipeline_at(config: Mapping[str, Any], path: Sequence[str]) -> list[Any]:
    pipeline = _nested(config, *path)
    if not isinstance(pipeline, list) or len(pipeline) < 2:
        raise VerificationContractError(
            "config pipeline must be a list containing loader and packer"
        )
    return pipeline


def _exact_bool_setting(
    mapping: Mapping[str, Any],
    key: str,
    *,
    default: bool,
    label: str,
) -> bool:
    value = mapping.get(key, default)
    if type(value) is not bool:
        raise VerificationContractError(f"{label} must be an exact boolean")
    return value


def validate_representation_config_pair(
    parent_config: Any,
    candidate_config: Any,
    *,
    expected_batch_size: int,
) -> dict[str, Any]:
    """Seal the lean FP32 versus pinned uint8 representation-only delta."""

    expected_batch_size = _exact_integer(
        "expected_batch_size", expected_batch_size, minimum=1
    )
    parent = _mapping(parent_config)
    candidate = _mapping(candidate_config)
    digest_parent = copy.deepcopy(parent)
    digest_candidate = copy.deepcopy(candidate)
    for config in (digest_parent, digest_candidate):
        for key in NON_EXECUTABLE_CONFIG_KEYS:
            config.pop(key, None)
        _normalize_main126_anchor_for_digest(config)
    parent_normalized_config_sha256 = canonical_sha256(digest_parent)
    candidate_normalized_config_sha256 = canonical_sha256(digest_candidate)
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
        if _nested(config, "test_dataloader", "batch_size") != expected_batch_size:
            raise VerificationContractError(
                f"{label} test batch size must equal {expected_batch_size}"
            )
        if _nested(config, "test_dataloader", "sampler", "shuffle") is not False:
            raise VerificationContractError(
                f"{label} test sampler must set shuffle=False"
            )
        preprocessor = _nested(config, "model", "data_preprocessor")
        if _nested(preprocessor, "type") != EXPECTED_REPRESENTATION_PREPROCESSOR:
            raise VerificationContractError(
                f"{label} preprocessor must be "
                f"{EXPECTED_REPRESENTATION_PREPROCESSOR}"
            )
        for forbidden in ("tta_model", "tta_pipeline"):
            if forbidden in config:
                raise VerificationContractError(
                    f"{label} config must not enable {forbidden}"
                )
        for stream_key in ("stream", "cuda_stream", "transfer_stream"):
            if stream_key in preprocessor:
                raise VerificationContractError(
                    f"{label} must not configure a custom CUDA stream"
                )

    normalized_parent = copy.deepcopy(parent)
    normalized_candidate = copy.deepcopy(candidate)
    for path in _REPRESENTATION_PIPELINE_PATHS:
        parent_pipeline = _pipeline_at(normalized_parent, path)
        candidate_pipeline = _pipeline_at(normalized_candidate, path)
        parent_loader = parent_pipeline[0]
        candidate_loader = candidate_pipeline[0]
        parent_packer = parent_pipeline[-1]
        candidate_packer = candidate_pipeline[-1]
        if not all(
            isinstance(value, Mapping)
            for value in (
                parent_loader,
                candidate_loader,
                parent_packer,
                candidate_packer,
            )
        ):
            raise VerificationContractError(
                "representation pipelines require mapping loader/packer entries"
            )
        if (
            parent_loader.get("type") != EXPECTED_LEAN_LOADER
            or candidate_loader.get("type") != EXPECTED_LEAN_LOADER
        ):
            raise VerificationContractError(
                f"both representation arms must use {EXPECTED_LEAN_LOADER}"
            )
        if parent_loader.get("to_float32") is not True:
            raise VerificationContractError("parent lean loader must emit FP32 images")
        if candidate_loader.get("to_float32") is not False:
            raise VerificationContractError(
                "candidate lean loader must retain uint8 images"
            )
        if parent_packer.get("type") != EXPECTED_PARENT_PACKER:
            raise VerificationContractError(
                f"parent packer must be {EXPECTED_PARENT_PACKER}"
            )
        if candidate_packer.get("type") != EXPECTED_CANDIDATE_PACKER:
            raise VerificationContractError(
                f"candidate packer must be {EXPECTED_CANDIDATE_PACKER}"
            )
        candidate_loader["to_float32"] = True
        candidate_packer["type"] = EXPECTED_PARENT_PACKER

    pin_memory = {}
    for loader_name in (
        "train_dataloader",
        "val_dataloader",
        "test_dataloader",
    ):
        parent_loader_config = _nested(parent, loader_name)
        candidate_loader_config = _nested(candidate, loader_name)
        parent_pin = _exact_bool_setting(
            parent_loader_config,
            "pin_memory",
            default=False,
            label=f"parent {loader_name}.pin_memory",
        )
        candidate_pin = _exact_bool_setting(
            candidate_loader_config,
            "pin_memory",
            default=False,
            label=f"candidate {loader_name}.pin_memory",
        )
        if parent_pin or not candidate_pin:
            raise VerificationContractError(
                "representation pair requires parent pin_memory=False and "
                "candidate pin_memory=True"
            )
        pin_memory[loader_name] = {
            PARENT_ARM: parent_pin,
            CANDIDATE_ARM: candidate_pin,
        }
        normalized_candidate[loader_name].pop("pin_memory", None)
        if "pin_memory" in normalized_parent[loader_name]:
            normalized_candidate[loader_name]["pin_memory"] = normalized_parent[
                loader_name
            ]["pin_memory"]

    parent_preprocessor = _nested(parent, "model", "data_preprocessor")
    candidate_preprocessor = _nested(candidate, "model", "data_preprocessor")
    parent_non_blocking = _exact_bool_setting(
        parent_preprocessor,
        "non_blocking",
        default=False,
        label="parent data_preprocessor.non_blocking",
    )
    candidate_non_blocking = _exact_bool_setting(
        candidate_preprocessor,
        "non_blocking",
        default=False,
        label="candidate data_preprocessor.non_blocking",
    )
    if parent_non_blocking or not candidate_non_blocking:
        raise VerificationContractError(
            "representation pair requires parent non_blocking=False and "
            "candidate non_blocking=True"
        )
    normalized_candidate["model"]["data_preprocessor"].pop("non_blocking", None)
    if "non_blocking" in normalized_parent["model"]["data_preprocessor"]:
        normalized_candidate["model"]["data_preprocessor"]["non_blocking"] = (
            normalized_parent["model"]["data_preprocessor"]["non_blocking"]
        )

    parent_imports = _nested(parent, "custom_imports", "imports")
    candidate_imports = _nested(candidate, "custom_imports", "imports")
    if not isinstance(parent_imports, list) or not isinstance(candidate_imports, list):
        raise VerificationContractError("custom imports must be ordered lists")
    if candidate_imports != parent_imports + [EXPECTED_RUNTIME_TRANSFORM_IMPORT]:
        raise VerificationContractError(
            "candidate custom imports must add only the uint8 runtime transform"
        )
    normalized_candidate["custom_imports"]["imports"] = copy.deepcopy(
        normalized_parent["custom_imports"]["imports"]
    )
    for config in (normalized_parent, normalized_candidate):
        for key in NON_EXECUTABLE_CONFIG_KEYS:
            config.pop(key, None)
        _normalize_main126_anchor_for_digest(config)
    if normalized_parent != normalized_candidate:
        raise VerificationContractError(
            "representation configs differ beyond the audited lean FP32/uint8 "
            "pinned-transfer surface"
        )
    return {
        "protocol": REPRESENTATION_PROTOCOL,
        "model_type": EXPECTED_MODEL_TYPE,
        "head_type": EXPECTED_HEAD_TYPE,
        "preprocessor": EXPECTED_REPRESENTATION_PREPROCESSOR,
        "parent_ingress_dtype": "torch.float32",
        "candidate_ingress_dtype": "torch.uint8",
        "loader": EXPECTED_LEAN_LOADER,
        "parent_packer": EXPECTED_PARENT_PACKER,
        "candidate_packer": EXPECTED_CANDIDATE_PACKER,
        "pin_memory": pin_memory,
        "non_blocking": {
            PARENT_ARM: parent_non_blocking,
            CANDIDATE_ARM: candidate_non_blocking,
        },
        "stream_policy": REPRESENTATION_STREAM_POLICY,
        "test_batch_size": expected_batch_size,
        "normalized_config_sha256": canonical_sha256(normalized_parent),
        "parent_normalized_config_sha256": parent_normalized_config_sha256,
        "candidate_normalized_config_sha256": (candidate_normalized_config_sha256),
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
        if isinstance(value, type):
            # MMDetection keeps the 3D box constructor itself in sample
            # metadata (for example, ``LiDARInstance3DBoxes``).  A class
            # reference has no portable raw bytes; its import identity is the
            # stable semantic value that must survive cloning and replay.
            self._tag("python.type")
            self._tag(value.__module__)
            self._tag(value.__qualname__)
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
            self._tag(
                f"tensor-wrapper:{type(value).__module__}.{type(value).__qualname__}"
            )
            self.visit(value.tensor, depth=depth + 1)
            return
        if hasattr(value, "to_dict") and callable(value.to_dict):
            self._tag(
                f"data-element:{type(value).__module__}.{type(value).__qualname__}"
            )
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


def finite_output_schema(value: Any) -> dict[str, Any]:
    """Validate an inference output and describe structure without its values.

    C0 and C32 are expected to produce different predictions.  Raw-output
    equality would therefore be the wrong correctness condition.  This helper
    instead rejects non-finite numeric values and returns a deterministic
    schema containing container types, keys, tensor shapes, dtypes, and
    devices.  Two arms can then be required to share a well-formed output
    contract individually while retaining their scientifically meaningful
    value and detection-count deltas.
    """

    def visit(item: Any, *, path: str, depth: int) -> dict[str, Any]:
        if depth > 64:
            raise VerificationContractError(
                f"output schema nesting is too deep at {path}"
            )
        if item is None:
            return {"type": "none"}
        if isinstance(item, torch.Tensor):
            if item.layout != torch.strided or item.is_quantized:
                raise VerificationContractError(
                    f"output tensor at {path} must be dense and non-quantized"
                )
            if (item.is_floating_point() or item.is_complex()) and not bool(
                torch.isfinite(item).all()
            ):
                raise VerificationContractError(
                    f"output tensor at {path} contains NaN or Inf"
                )
            return {
                "type": "torch.Tensor",
                "dtype": str(item.dtype),
                "shape": list(item.shape),
                "device_type": item.device.type,
                "layout": str(item.layout),
            }
        if isinstance(item, np.ndarray):
            if np.issubdtype(item.dtype, np.inexact) and not bool(
                np.isfinite(item).all()
            ):
                raise VerificationContractError(
                    f"output array at {path} contains NaN or Inf"
                )
            return {
                "type": "numpy.ndarray",
                "dtype": item.dtype.str,
                "shape": list(item.shape),
            }
        if isinstance(item, np.generic):
            if np.issubdtype(item.dtype, np.inexact) and not bool(
                np.isfinite(item)
            ):
                raise VerificationContractError(
                    f"output scalar at {path} contains NaN or Inf"
                )
            return {"type": f"numpy.scalar:{item.dtype.str}"}
        if isinstance(item, Mapping):
            return {
                "type": f"mapping:{type(item).__module__}.{type(item).__qualname__}",
                "items": [
                    {
                        "key": str(key),
                        "schema": visit(
                            child,
                            path=f"{path}.{key}",
                            depth=depth + 1,
                        ),
                    }
                    for key, child in item.items()
                ],
            }
        if isinstance(item, (list, tuple)):
            return {
                "type": f"sequence:{type(item).__name__}",
                "length": len(item),
                "items": [
                    visit(
                        child,
                        path=f"{path}[{index}]",
                        depth=depth + 1,
                    )
                    for index, child in enumerate(item)
                ],
            }
        if isinstance(item, float):
            if not math.isfinite(item):
                raise VerificationContractError(
                    f"output scalar at {path} contains NaN or Inf"
                )
            return {"type": "float"}
        if isinstance(item, (str, bytes, bool, int)):
            return {"type": type(item).__name__}
        if hasattr(item, "tensor") and isinstance(item.tensor, torch.Tensor):
            return {
                "type": (
                    f"tensor-wrapper:{type(item).__module__}."
                    f"{type(item).__qualname__}"
                ),
                "tensor": visit(
                    item.tensor,
                    path=f"{path}.tensor",
                    depth=depth + 1,
                ),
            }
        if hasattr(item, "to_dict") and callable(item.to_dict):
            return {
                "type": (
                    f"data-element:{type(item).__module__}."
                    f"{type(item).__qualname__}"
                ),
                "value": visit(
                    item.to_dict(),
                    path=f"{path}.to_dict",
                    depth=depth + 1,
                ),
            }
        raise VerificationContractError(
            f"unsupported inference output at {path}: "
            f"{type(item).__module__}.{type(item).__qualname__}"
        )

    schema = visit(value, path="output", depth=0)
    return {
        "schema": schema,
        "sha256": canonical_sha256(schema),
    }


def _sample_identity(sample: Any, index: int) -> dict[str, Any]:
    metainfo = getattr(sample, "metainfo", None)
    if metainfo is None and isinstance(sample, Mapping):
        metainfo = sample.get("metainfo")
    if not isinstance(metainfo, Mapping):
        raise VerificationContractError(f"data sample {index} has no metainfo mapping")
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


def semantic_sample_digest(
    batch: Mapping[str, Any],
    *,
    expected_batch_size: int,
) -> dict[str, Any]:
    """Return the cheap semantic identity key used by live paired timing."""

    expected_batch_size = _exact_integer(
        "expected_batch_size", expected_batch_size, minimum=1
    )
    if not isinstance(batch, Mapping):
        raise VerificationContractError("semantic batch must be a mapping")
    samples = batch.get("data_samples")
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise VerificationContractError(
            "semantic batch data_samples must be a sequence"
        )
    if len(samples) != expected_batch_size:
        raise VerificationContractError(
            f"semantic batch has {len(samples)} samples, expected "
            f"{expected_batch_size}"
        )
    identities = [
        _sample_identity(sample, index) for index, sample in enumerate(samples)
    ]
    return {
        "sample_identities": identities,
        "sha256": canonical_sha256(identities),
    }


def build_representation_batch_manifest(
    batch: Mapping[str, Any],
    *,
    arm: str,
    batch_index: int,
    expected_batch_size: int,
    observed_pinned: bool | None = None,
) -> dict[str, Any]:
    """Validate and fingerprint one arm-specific CPU ingress representation."""

    if arm not in (PARENT_ARM, CANDIDATE_ARM):
        raise VerificationContractError("representation arm is invalid")
    batch_index = _exact_integer("batch_index", batch_index, minimum=0)
    expected_batch_size = _exact_integer(
        "expected_batch_size", expected_batch_size, minimum=1
    )
    if not isinstance(batch, Mapping):
        raise VerificationContractError("representation batch must be a mapping")
    inputs = batch.get("inputs")
    samples = batch.get("data_samples")
    if not isinstance(inputs, Mapping):
        raise VerificationContractError("representation batch inputs must be a mapping")
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise VerificationContractError(
            "representation batch data_samples must be a sequence"
        )
    if len(samples) != expected_batch_size:
        raise VerificationContractError(
            f"representation batch {batch_index} has {len(samples)} samples, "
            f"expected {expected_batch_size}"
        )
    image = inputs.get("img")
    if not isinstance(image, torch.Tensor):
        raise VerificationContractError("representation img must be a tensor")
    if image.layout != torch.strided or image.device.type != "cpu":
        raise VerificationContractError("representation img must be a dense CPU tensor")
    if not image.is_contiguous():
        raise VerificationContractError("representation img must be contiguous")
    expected_shape = (expected_batch_size, 4, 3, 800, 800)
    if tuple(image.shape) != expected_shape:
        raise VerificationContractError(
            f"representation img shape {tuple(image.shape)} != {expected_shape}"
        )
    expected_dtype = torch.float32 if arm == PARENT_ARM else torch.uint8
    if image.dtype != expected_dtype:
        raise VerificationContractError(
            f"{arm} representation img dtype {image.dtype} != {expected_dtype}"
        )
    if arm == PARENT_ARM:
        if not bool(torch.isfinite(image).all()):
            raise VerificationContractError("parent FP32 image must be finite")
        if bool(((image < 0) | (image > 255)).any()) or not torch.equal(
            image, image.trunc()
        ):
            raise VerificationContractError(
                "parent FP32 image must be an exact expanded uint8 image"
            )
    if observed_pinned is None:
        observed_pinned = bool(image.is_pinned())
    if type(observed_pinned) is not bool:
        raise VerificationContractError("observed_pinned must be an exact boolean")
    expected_pinned = arm == CANDIDATE_ARM
    if observed_pinned is not expected_pinned:
        raise VerificationContractError(
            f"{arm} representation pinned={observed_pinned}, expected "
            f"{expected_pinned}"
        )
    for name in HOST_METADATA_KEYS:
        value = inputs.get(name)
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            raise VerificationContractError(
                f"representation {name} must be a CPU tensor"
            )
        if value.ndim == 0 or value.shape[0] != expected_batch_size:
            raise VerificationContractError(
                f"representation {name} must have the frozen batch dimension"
            )
        if not value.is_contiguous():
            raise VerificationContractError(f"representation {name} must be contiguous")
    identities = [
        _sample_identity(sample, index) for index, sample in enumerate(samples)
    ]
    canonical_image = image if image.dtype == torch.float32 else image.float()
    common_input_fingerprints = {
        name: raw_fingerprint(inputs[name]) for name in sorted(inputs) if name != "img"
    }
    semantic_payload = {
        "batch_index": batch_index,
        "sample_identities": identities,
        "data_samples": raw_fingerprint(samples),
        "canonical_fp32_image": raw_fingerprint(canonical_image),
        "common_inputs": common_input_fingerprints,
    }
    representation_fingerprint = raw_fingerprint(batch)
    manifest = {
        "arm": arm,
        "batch_index": batch_index,
        "batch_size": expected_batch_size,
        "input_keys": sorted(inputs),
        "sample_identities": identities,
        "semantic_sha256": canonical_sha256(semantic_payload),
        "representation_fingerprint": representation_fingerprint,
        "ingress": {
            "image_dtype": str(image.dtype),
            "image_shape": list(image.shape),
            "image_stride": list(image.stride()),
            "image_bytes": image.numel() * image.element_size(),
            "device": str(image.device),
            "contiguous": image.is_contiguous(),
            "pinned": observed_pinned,
            "non_blocking": arm == CANDIDATE_ARM,
            "stream_policy": REPRESENTATION_STREAM_POLICY,
        },
    }
    manifest["sha256"] = canonical_sha256(manifest)
    return manifest


def validate_representation_ingress(
    batch: Mapping[str, Any],
    *,
    arm: str,
    expected_batch_size: int,
    observed_pinned: bool | None = None,
) -> dict[str, Any]:
    """Cheap structural ingress check suitable between timed components."""

    if arm not in (PARENT_ARM, CANDIDATE_ARM):
        raise VerificationContractError("representation arm is invalid")
    expected_batch_size = _exact_integer(
        "expected_batch_size", expected_batch_size, minimum=1
    )
    if not isinstance(batch, Mapping) or not isinstance(batch.get("inputs"), Mapping):
        raise VerificationContractError(
            "representation ingress must contain an inputs mapping"
        )
    image = batch["inputs"].get("img")
    if not isinstance(image, torch.Tensor):
        raise VerificationContractError("representation ingress img must be a tensor")
    expected_dtype = torch.float32 if arm == PARENT_ARM else torch.uint8
    expected_shape = (expected_batch_size, 4, 3, 800, 800)
    if (
        image.device.type != "cpu"
        or image.layout != torch.strided
        or image.dtype != expected_dtype
        or tuple(image.shape) != expected_shape
        or not image.is_contiguous()
    ):
        raise VerificationContractError(
            f"{arm} ingress violates the frozen CPU/dtype/shape/layout contract"
        )
    if observed_pinned is None:
        observed_pinned = bool(image.is_pinned())
    if type(observed_pinned) is not bool:
        raise VerificationContractError("observed_pinned must be an exact boolean")
    expected_pinned = arm == CANDIDATE_ARM
    if observed_pinned is not expected_pinned:
        raise VerificationContractError(
            f"{arm} ingress pinned={observed_pinned}, expected {expected_pinned}"
        )
    samples = batch.get("data_samples")
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise VerificationContractError(
            "representation ingress data_samples must be a sequence"
        )
    if len(samples) != expected_batch_size:
        raise VerificationContractError(
            "representation ingress data sample count changed"
        )
    return {
        "arm": arm,
        "input_keys": sorted(batch["inputs"]),
        "image_dtype": str(image.dtype),
        "image_shape": list(image.shape),
        "image_stride": list(image.stride()),
        "image_bytes": image.numel() * image.element_size(),
        "pinned": observed_pinned,
        "non_blocking": arm == CANDIDATE_ARM,
        "stream_policy": REPRESENTATION_STREAM_POLICY,
    }


def build_representation_pair_manifest(
    parent_batch: Mapping[str, Any],
    candidate_batch: Mapping[str, Any],
    *,
    batch_index: int,
    expected_batch_size: int,
    observed_pinned: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Bind two distinct representations to one exact semantic batch."""

    observed_pinned = observed_pinned or {}
    if not isinstance(observed_pinned, Mapping):
        raise VerificationContractError("observed_pinned must be a mapping")
    parent = build_representation_batch_manifest(
        parent_batch,
        arm=PARENT_ARM,
        batch_index=batch_index,
        expected_batch_size=expected_batch_size,
        observed_pinned=observed_pinned.get(PARENT_ARM),
    )
    candidate = build_representation_batch_manifest(
        candidate_batch,
        arm=CANDIDATE_ARM,
        batch_index=batch_index,
        expected_batch_size=expected_batch_size,
        observed_pinned=observed_pinned.get(CANDIDATE_ARM),
    )
    if parent["sample_identities"] != candidate["sample_identities"]:
        raise VerificationContractError(
            "representation pair has swapped or different sample identities"
        )
    if parent["semantic_sha256"] != candidate["semantic_sha256"]:
        raise VerificationContractError(
            "representation pair does not share one semantic batch digest"
        )
    if parent["representation_fingerprint"] == candidate["representation_fingerprint"]:
        raise VerificationContractError(
            "representation pair must preserve distinct FP32 and uint8 ingress"
        )
    pair = {
        "schema": "rosa.representation_pair.v1",
        "batch_index": batch_index,
        "semantic_sha256": parent["semantic_sha256"],
        "sample_identities": parent["sample_identities"],
        "arms": {
            PARENT_ARM: parent,
            CANDIDATE_ARM: candidate,
        },
    }
    pair["sha256"] = canonical_sha256(pair)
    return pair


def validate_representation_snapshot(
    batch: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    """Reject mutation of a captured reusable arm representation."""

    expected = _nested(manifest, "representation_fingerprint")
    observed = raw_fingerprint(batch)
    if observed != expected:
        raise VerificationContractError(
            f"captured representation mutated for {label}: "
            f"expected={expected}, observed={observed}"
        )
    return observed


def validate_live_representation_observation(
    batch: Mapping[str, Any],
    expected_pair: Mapping[str, Any],
    *,
    arm: str,
    sequence_position: int,
    expected_batch_size: int,
    observed_pinned: bool | None = None,
) -> dict[str, Any]:
    """Rebuild and compare the full live arm snapshot at its timing position.

    The returned semantic SHA is computed from ``batch``.  It is never copied
    from the preregistered pair, which is important because sample identity
    alone does not bind pixels or non-image metadata.
    """

    if arm not in (PARENT_ARM, CANDIDATE_ARM):
        raise VerificationContractError("live representation arm is invalid")
    sequence_position = _exact_integer(
        "sequence_position", sequence_position, minimum=0
    )
    if not isinstance(expected_pair, Mapping):
        raise VerificationContractError(
            "live representation expected pair must be a mapping"
        )
    if expected_pair.get("batch_index") != sequence_position:
        raise VerificationContractError("live representation sequence position changed")
    arms = expected_pair.get("arms")
    if not isinstance(arms, Mapping) or arm not in arms:
        raise VerificationContractError(
            "live representation expected arm manifest is missing"
        )
    expected_arm = arms[arm]
    observed_arm = build_representation_batch_manifest(
        batch,
        arm=arm,
        batch_index=sequence_position,
        expected_batch_size=expected_batch_size,
        observed_pinned=observed_pinned,
    )
    if observed_arm != expected_arm:
        raise VerificationContractError(
            f"live {arm} representation snapshot differs from preregistered "
            f"pair {sequence_position}"
        )
    observed_semantic_sha256 = observed_arm["semantic_sha256"]
    if observed_semantic_sha256 != expected_pair.get(
        "semantic_sha256"
    ) or observed_semantic_sha256 != expected_arm.get("semantic_sha256"):
        raise VerificationContractError(
            f"live {arm} semantic digest differs at pair {sequence_position}"
        )
    return {
        "arm": arm,
        "sequence_position": sequence_position,
        "observed_semantic_sha256": observed_semantic_sha256,
        "observed_representation_fingerprint": observed_arm[
            "representation_fingerprint"
        ],
        "observed_arm_manifest_sha256": observed_arm["sha256"],
        "exact_arm_manifest_match": True,
    }


def validate_common_preprocessor_boundary(
    parent_processed: Mapping[str, Any],
    candidate_processed: Mapping[str, Any],
    *,
    expected_device_type: str | None = None,
) -> dict[str, Any]:
    """Prove both ingress formats become the same FP32 model-boundary batch."""

    if not isinstance(parent_processed, Mapping) or not isinstance(
        candidate_processed, Mapping
    ):
        raise VerificationContractError(
            "common preprocessor boundary values must be mappings"
        )
    for label, processed in (
        (PARENT_ARM, parent_processed),
        (CANDIDATE_ARM, candidate_processed),
    ):
        inputs = processed.get("inputs")
        if not isinstance(inputs, Mapping):
            raise VerificationContractError(
                f"{label} common boundary inputs must be a mapping"
            )
        image = inputs.get("img")
        if not isinstance(image, torch.Tensor) or image.dtype != torch.float32:
            raise VerificationContractError(
                f"{label} common boundary img must be torch.float32"
            )
        if (
            expected_device_type is not None
            and image.device.type != expected_device_type
        ):
            raise VerificationContractError(
                f"{label} common boundary img device {image.device.type} != "
                f"{expected_device_type}"
            )
    fingerprint = require_same_fingerprint(
        "common preprocessor boundary",
        parent_processed,
        candidate_processed,
    )
    return {
        "parity": "raw-bit exact",
        "fingerprint": fingerprint,
        "image_dtype": "torch.float32",
        "device_type": expected_device_type,
    }


def build_timed_common_boundary_receipt(
    observations: Mapping[int, Mapping[str, Mapping[str, Any]]],
    expected_pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Close all timed semantic pairs at the shared post-preprocessor boundary."""

    if not isinstance(observations, Mapping):
        raise VerificationContractError(
            "common-boundary observations must be a mapping"
        )
    if not isinstance(expected_pairs, Sequence) or isinstance(
        expected_pairs, (str, bytes)
    ):
        raise VerificationContractError(
            "common-boundary expected pairs must be a sequence"
        )
    if len(expected_pairs) != REPRESENTATION_SEQUENCE_PAIRS_PER_ARM:
        raise VerificationContractError(
            "common-boundary receipt requires all 360 timed semantic pairs"
        )
    if set(observations) != set(range(len(expected_pairs))):
        raise VerificationContractError(
            "common-boundary observations are incomplete or non-contiguous"
        )
    records = []
    for pair_index, expected_pair in enumerate(expected_pairs):
        _validate_sealed_representation_pair(
            expected_pair,
            pair_index=pair_index,
        )
        observed = observations[pair_index]
        if not isinstance(observed, Mapping) or set(observed) != {
            PARENT_ARM,
            CANDIDATE_ARM,
        }:
            raise VerificationContractError(
                f"common-boundary pair {pair_index} lacks one arm"
            )
        pair_arms = expected_pair.get("arms")
        if not isinstance(pair_arms, Mapping) or set(pair_arms) != {
            PARENT_ARM,
            CANDIDATE_ARM,
        }:
            raise VerificationContractError(
                f"sealed representation pair {pair_index} lacks two arms"
            )
        arm_records = {}
        for arm in (PARENT_ARM, CANDIDATE_ARM):
            arm_observation = observed[arm]
            if not isinstance(arm_observation, Mapping):
                raise VerificationContractError(
                    f"common-boundary pair {pair_index} {arm} is invalid"
                )
            if arm_observation.get("arm") != arm:
                raise VerificationContractError("common-boundary arm label changed")
            if arm_observation.get("sequence_position") != pair_index:
                raise VerificationContractError(
                    "common-boundary sequence position changed"
                )
            if arm_observation.get("observed_semantic_sha256") != (
                expected_pair.get("semantic_sha256")
            ):
                raise VerificationContractError(
                    "common-boundary observed semantic digest changed"
                )
            if arm_observation.get("exact_arm_manifest_match") is not True:
                raise VerificationContractError(
                    "common-boundary ingress arm snapshot was not exact"
                )
            expected_arm = pair_arms[arm]
            if arm_observation.get("observed_arm_manifest_sha256") != (
                expected_arm.get("sha256")
            ):
                raise VerificationContractError(
                    "live arm manifest SHA is not bound to the sealed pair"
                )
            if arm_observation.get("observed_representation_fingerprint") != (
                expected_arm.get("representation_fingerprint")
            ):
                raise VerificationContractError(
                    "live arm representation fingerprint is not sealed"
                )
            boundary = _validate_raw_fingerprint_schema(
                arm_observation.get("boundary_fingerprint"),
                label=f"timed pair {pair_index} {arm} boundary",
            )
            arm_records[arm] = {
                "raw_boundary_fingerprint": boundary,
                "live_arm_manifest_sha256": arm_observation[
                    "observed_arm_manifest_sha256"
                ],
                "sealed_arm_manifest_sha256": expected_arm["sha256"],
                "live_representation_fingerprint": arm_observation[
                    "observed_representation_fingerprint"
                ],
                "sealed_representation_fingerprint": expected_arm[
                    "representation_fingerprint"
                ],
            }
        if (
            arm_records[PARENT_ARM]["raw_boundary_fingerprint"]
            != arm_records[CANDIDATE_ARM]["raw_boundary_fingerprint"]
        ):
            raise VerificationContractError(
                f"common preprocessor boundary mismatch at timed pair {pair_index}"
            )
        records.append(
            {
                "pair_index": pair_index,
                "sealed_pair_sha256": expected_pair.get("sha256"),
                "sealed_pair_identity_sha256": canonical_sha256(
                    expected_pair.get("sample_identities")
                ),
                "semantic_sha256": observed[PARENT_ARM][
                    "observed_semantic_sha256"
                ],
                "arms": arm_records,
                "parity": "raw-bit exact",
            }
        )
    receipt = {
        "schema": REPRESENTATION_BOUNDARY_RECEIPT_SCHEMA,
        "protocol": REPRESENTATION_PROTOCOL,
        "pairs": len(records),
        "all_timed_pairs_covered": True,
        "parity": "raw-bit exact",
        "records": records,
    }
    receipt["sha256"] = canonical_sha256(receipt)
    return receipt


def _validate_raw_fingerprint_schema(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "sha256",
        "nodes",
        "tensor_bytes",
    }:
        raise VerificationContractError(f"{label} has an invalid raw schema")
    if not _is_sha256(value.get("sha256")):
        raise VerificationContractError(f"{label} has an invalid SHA256")
    nodes = _exact_integer(f"{label}.nodes", value.get("nodes"), minimum=1)
    tensor_bytes = _exact_integer(
        f"{label}.tensor_bytes",
        value.get("tensor_bytes"),
        minimum=0,
    )
    return {
        "sha256": value["sha256"],
        "nodes": nodes,
        "tensor_bytes": tensor_bytes,
    }


def _validate_sealed_representation_pair(
    pair: Any,
    *,
    pair_index: int,
) -> None:
    if not isinstance(pair, Mapping):
        raise VerificationContractError(
            f"sealed representation pair {pair_index} is invalid"
        )
    pair_sha256 = pair.get("sha256")
    if not _is_sha256(pair_sha256) or pair_sha256 != canonical_sha256(
        {key: value for key, value in pair.items() if key != "sha256"}
    ):
        raise VerificationContractError(
            f"sealed representation pair {pair_index} SHA mismatch"
        )
    pair_arms = pair.get("arms")
    if not isinstance(pair_arms, Mapping) or set(pair_arms) != {
        PARENT_ARM,
        CANDIDATE_ARM,
    }:
        raise VerificationContractError(
            f"sealed representation pair {pair_index} lacks two arms"
        )
    for arm in (PARENT_ARM, CANDIDATE_ARM):
        arm_manifest = pair_arms[arm]
        if not isinstance(arm_manifest, Mapping):
            raise VerificationContractError(
                f"sealed representation pair {pair_index} {arm} is invalid"
            )
        arm_sha256 = arm_manifest.get("sha256")
        if not _is_sha256(arm_sha256) or arm_sha256 != canonical_sha256(
            {
                key: value
                for key, value in arm_manifest.items()
                if key != "sha256"
            }
        ):
            raise VerificationContractError(
                f"sealed representation pair {pair_index} {arm} SHA mismatch"
            )
        _validate_raw_fingerprint_schema(
            arm_manifest.get("representation_fingerprint"),
            label=f"sealed pair {pair_index} {arm} representation",
        )


def validate_timed_common_boundary_receipt(
    receipt: Mapping[str, Any],
    expected_pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute the receipt closure and bind it to the timed pair manifest."""

    if not isinstance(receipt, Mapping):
        raise VerificationContractError(
            "timed common-boundary receipt must be a mapping"
        )
    if set(receipt) != {
        "schema",
        "protocol",
        "pairs",
        "all_timed_pairs_covered",
        "parity",
        "records",
        "sha256",
    }:
        raise VerificationContractError(
            "timed common-boundary receipt schema fields changed"
        )
    if receipt.get("schema") != REPRESENTATION_BOUNDARY_RECEIPT_SCHEMA:
        raise VerificationContractError("timed common-boundary receipt schema changed")
    if receipt.get("protocol") != REPRESENTATION_PROTOCOL:
        raise VerificationContractError(
            "timed common-boundary receipt protocol changed"
        )
    if receipt.get("pairs") != REPRESENTATION_SEQUENCE_PAIRS_PER_ARM:
        raise VerificationContractError(
            "timed common-boundary receipt does not cover 360 pairs"
        )
    if receipt.get("all_timed_pairs_covered") is not True:
        raise VerificationContractError("timed common-boundary receipt is incomplete")
    if receipt.get("parity") != "raw-bit exact":
        raise VerificationContractError(
            "timed common-boundary parity is not raw-bit exact"
        )
    records = receipt.get("records")
    if not isinstance(records, list) or len(records) != (
        REPRESENTATION_SEQUENCE_PAIRS_PER_ARM
    ):
        raise VerificationContractError("timed common-boundary records are incomplete")
    if len(expected_pairs) != len(records):
        raise VerificationContractError(
            "timed common-boundary receipt/pair count differs"
        )
    for pair_index, (record, pair) in enumerate(zip(records, expected_pairs)):
        _validate_sealed_representation_pair(pair, pair_index=pair_index)
        if not isinstance(record, Mapping):
            raise VerificationContractError(
                "timed common-boundary record must be a mapping"
            )
        if set(record) != {
            "pair_index",
            "sealed_pair_sha256",
            "sealed_pair_identity_sha256",
            "semantic_sha256",
            "arms",
            "parity",
        }:
            raise VerificationContractError(
                "timed common-boundary record schema changed"
            )
        if record.get("pair_index") != pair_index:
            raise VerificationContractError("timed common-boundary pair order changed")
        if record.get("semantic_sha256") != pair.get("semantic_sha256"):
            raise VerificationContractError(
                "timed common-boundary semantic sequence changed"
            )
        if record.get("sealed_pair_sha256") != pair.get("sha256"):
            raise VerificationContractError(
                "timed common-boundary sealed pair SHA changed"
            )
        if record.get("sealed_pair_identity_sha256") != canonical_sha256(
            pair.get("sample_identities")
        ):
            raise VerificationContractError(
                "timed common-boundary sealed pair identity changed"
            )
        if record.get("parity") != "raw-bit exact":
            raise VerificationContractError("timed common-boundary pair parity changed")
        record_arms = record.get("arms")
        pair_arms = pair.get("arms")
        if not isinstance(record_arms, Mapping) or set(record_arms) != {
            PARENT_ARM,
            CANDIDATE_ARM,
        }:
            raise VerificationContractError(
                "timed common-boundary record does not store two arms"
            )
        if not isinstance(pair_arms, Mapping) or set(pair_arms) != {
            PARENT_ARM,
            CANDIDATE_ARM,
        }:
            raise VerificationContractError(
                "timed common-boundary sealed pair lacks two arms"
            )
        boundaries = {}
        for arm in (PARENT_ARM, CANDIDATE_ARM):
            arm_record = record_arms[arm]
            sealed_arm = pair_arms[arm]
            if not isinstance(arm_record, Mapping):
                raise VerificationContractError(
                    "timed common-boundary arm record is invalid"
                )
            if set(arm_record) != {
                "raw_boundary_fingerprint",
                "live_arm_manifest_sha256",
                "sealed_arm_manifest_sha256",
                "live_representation_fingerprint",
                "sealed_representation_fingerprint",
            }:
                raise VerificationContractError(
                    "timed common-boundary arm record schema changed"
                )
            boundaries[arm] = _validate_raw_fingerprint_schema(
                arm_record.get("raw_boundary_fingerprint"),
                label=f"timed pair {pair_index} {arm} boundary",
            )
            if arm_record.get("live_arm_manifest_sha256") != sealed_arm.get(
                "sha256"
            ) or arm_record.get("sealed_arm_manifest_sha256") != sealed_arm.get(
                "sha256"
            ):
                raise VerificationContractError(
                    "timed common-boundary arm manifest SHA changed"
                )
            sealed_representation = sealed_arm.get("representation_fingerprint")
            _validate_raw_fingerprint_schema(
                sealed_representation,
                label=f"sealed pair {pair_index} {arm} representation",
            )
            if arm_record.get("live_representation_fingerprint") != (
                sealed_representation
            ) or arm_record.get("sealed_representation_fingerprint") != (
                sealed_representation
            ):
                raise VerificationContractError(
                    "timed common-boundary arm representation is not sealed"
                )
        if boundaries[PARENT_ARM] != boundaries[CANDIDATE_ARM]:
            raise VerificationContractError(
                "timed common-boundary two-arm raw fingerprints differ"
            )
    payload = {key: value for key, value in receipt.items() if key != "sha256"}
    if receipt.get("sha256") != canonical_sha256(payload):
        raise VerificationContractError("timed common-boundary receipt SHA mismatch")
    return {
        "schema": REPRESENTATION_BOUNDARY_RECEIPT_SCHEMA,
        "pairs": len(records),
        "all_timed_pairs_covered": True,
        "parity": "raw-bit exact",
        "sha256": receipt["sha256"],
    }


def validate_pipeline_memory_baseline_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Require an explicit release boundary before pipeline-only timing."""

    if not isinstance(receipt, Mapping):
        raise VerificationContractError(
            "pipeline memory baseline receipt must be a mapping"
        )
    if receipt.get("schema") != "rosa.pipeline_memory_baseline_receipt.v3":
        raise VerificationContractError(
            "pipeline memory baseline receipt schema changed"
        )
    if "release_actions" in receipt:
        raise VerificationContractError(
            "pipeline memory receipt must not trust self-reported release booleans"
        )
    snapshots = {}
    for name in (
        "clean_pipeline_baseline",
        "before_release",
        "after_release",
        "pipeline_baseline",
    ):
        snapshot = receipt.get(name)
        if not isinstance(snapshot, Mapping):
            raise VerificationContractError(f"pipeline memory receipt lacks {name}")
        normalized = {}
        for key in ("allocated_bytes", "reserved_bytes"):
            normalized[key] = _exact_integer(
                f"pipeline memory {name}.{key}",
                snapshot.get(key),
                minimum=0,
            )
        snapshots[name] = normalized
    if snapshots["pipeline_baseline"] != snapshots["after_release"]:
        raise VerificationContractError(
            "pipeline CUDA baseline changed after the release receipt"
        )
    for key in ("allocated_bytes", "reserved_bytes"):
        if snapshots["after_release"][key] > snapshots[
            "clean_pipeline_baseline"
        ][key]:
            raise VerificationContractError(
                f"pipeline release did not return {key} to the clean baseline"
            )

    liveness = receipt.get("liveness")
    if not isinstance(liveness, Mapping) or set(liveness) != {"before", "after"}:
        raise VerificationContractError(
            "pipeline memory receipt lacks measured object liveness"
        )
    expected_categories = {
        "full_models",
        "raw_cpu_batches",
        "processed_cuda_batches",
        "pipeline_preprocessors",
    }
    normalized_liveness = {}
    for phase in ("before", "after"):
        phase_value = liveness[phase]
        if not isinstance(phase_value, Mapping) or set(phase_value) != (
            expected_categories
        ):
            raise VerificationContractError(
                f"pipeline liveness {phase} categories changed"
            )
        normalized_liveness[phase] = {}
        for category in sorted(expected_categories):
            value = phase_value[category]
            if not isinstance(value, Mapping) or set(value) != {
                "weakref_targets",
                "weakref_alive",
                "tracked_container_targets",
                "tracked_container_alive",
            }:
                raise VerificationContractError(
                    f"pipeline liveness {phase}.{category} schema changed"
                )
            normalized = {
                key: _exact_integer(
                    f"pipeline liveness {phase}.{category}.{key}",
                    value.get(key),
                    minimum=0,
                )
                for key in (
                    "weakref_targets",
                    "weakref_alive",
                    "tracked_container_targets",
                    "tracked_container_alive",
                )
            }
            if normalized["weakref_alive"] > normalized["weakref_targets"] or (
                normalized["tracked_container_alive"]
                > normalized["tracked_container_targets"]
            ):
                raise VerificationContractError(
                    "pipeline liveness alive count exceeds targets"
                )
            normalized_liveness[phase][category] = normalized

    for category in ("full_models", "raw_cpu_batches", "processed_cuda_batches"):
        before = normalized_liveness["before"][category]
        after = normalized_liveness["after"][category]
        if (
            after["weakref_targets"] != before["weakref_targets"]
            or after["tracked_container_targets"]
            != before["tracked_container_targets"]
        ):
            raise VerificationContractError(
                f"pipeline liveness target set changed for {category}"
            )
        if before["weakref_targets"] + before["tracked_container_targets"] < 1:
            raise VerificationContractError(
                f"pipeline liveness did not measure {category}"
            )
        if (
            before["weakref_alive"] != before["weakref_targets"]
            or before["tracked_container_alive"]
            != before["tracked_container_targets"]
        ):
            raise VerificationContractError(
                f"pipeline liveness before-release {category} was incomplete"
            )
        if after["weakref_alive"] or after["tracked_container_alive"]:
            raise VerificationContractError(
                f"pipeline release retained live {category} objects"
            )
    preprocessor_before = normalized_liveness["before"]["pipeline_preprocessors"]
    preprocessor_after = normalized_liveness["after"]["pipeline_preprocessors"]
    if (
        preprocessor_after["weakref_targets"]
        != preprocessor_before["weakref_targets"]
        or preprocessor_after["tracked_container_targets"]
        != preprocessor_before["tracked_container_targets"]
    ):
        raise VerificationContractError(
            "pipeline preprocessor liveness target set changed"
        )
    if preprocessor_before["weakref_targets"] != 2:
        raise VerificationContractError(
            "pipeline liveness must measure two independent preprocessors"
        )
    if (
        preprocessor_before["weakref_alive"] != 2
        or preprocessor_after["weakref_alive"] != 2
    ):
        raise VerificationContractError(
            "pipeline preprocessors were not actually preserved"
        )
    payload = {key: value for key, value in receipt.items() if key != "sha256"}
    if receipt.get("sha256") != canonical_sha256(payload):
        raise VerificationContractError("pipeline memory baseline receipt SHA mismatch")
    return {
        "schema": receipt["schema"],
        "clean_pipeline_baseline": snapshots["clean_pipeline_baseline"],
        "before_release": snapshots["before_release"],
        "after_release": snapshots["after_release"],
        "pipeline_baseline": snapshots["pipeline_baseline"],
        "liveness": normalized_liveness,
        "sha256": receipt["sha256"],
    }


def validate_expected_manifest(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    if canonical_sha256(observed) != canonical_sha256(expected):
        raise VerificationContractError(
            "captured sample manifest does not match expected"
        )


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
    "BOOTSTRAP_SAMPLES",
    "BOOTSTRAP_SEED",
    "CANDIDATE_ARM",
    "CPU_METADATA_PROTOCOL",
    "CPPC",
    "DEFAULT_MEASURED_ITERATIONS_PER_ARM",
    "DEFAULT_WARMUP_ITERATIONS_PER_ARM",
    "EXPECTED_CANDIDATE_PREPROCESSOR",
    "EXPECTED_HEAD_TYPE",
    "EXPECTED_MODEL_TYPE",
    "EXPECTED_PARENT_PREPROCESSOR",
    "INPUT_PIPELINE_SCOPE",
    "HOST_METADATA_KEYS",
    "ITERATIONS_PER_ARM_GRANULARITY",
    "MINIMUM_MEASURED_ITERATIONS_PER_ARM",
    "MINIMUM_WARMUP_ITERATIONS_PER_ARM",
    "MODEL_ONLY_SCOPE",
    "NEAR_NATIVE_MEDIAN_RATIO_CI95_UPPER_MAXIMUM",
    "NEAR_NATIVE_P95_RATIO_MAXIMUM",
    "PARENT_ARM",
    "PCCP",
    "PHYSICAL_ROUND_ORDER",
    "POINT_PAYLOAD_INPUT_KEYS",
    "REPRESENTATION_PROTOCOL",
    "REPRESENTATION_STREAM_POLICY",
    "REPRESENTATION_BOUNDARY_RECEIPT_SCHEMA",
    "REPRESENTATION_C32_E20_CHECKPOINT_SHA256",
    "REPRESENTATION_CANDIDATE_NORMALIZED_CONFIG_SHA256",
    "REPRESENTATION_CAPTURE_BATCHES",
    "REPRESENTATION_DATALOADER_WORKERS",
    "REPRESENTATION_EXPECTED_BATCH_SIZE",
    "REPRESENTATION_MEASURED_ITERATIONS_PER_ARM",
    "REPRESENTATION_PARENT_NORMALIZED_CONFIG_SHA256",
    "REPRESENTATION_PREREGISTRATION_SCHEMA",
    "REPRESENTATION_SEED",
    "REPRESENTATION_SEQUENCE_PAIRS_PER_ARM",
    "REPRESENTATION_SEQUENCE_SCHEMA",
    "REPRESENTATION_WARMUP_ITERATIONS_PER_ARM",
    "ScheduleEntry",
    "VerificationContractError",
    "build_alternating_schedule",
    "build_batch_manifest",
    "build_representation_batch_manifest",
    "build_representation_pair_manifest",
    "build_timed_common_boundary_receipt",
    "canonical_json",
    "canonical_sha256",
    "classify_transfer_event",
    "evaluate_c32_near_native_gate",
    "evaluate_input_pipeline_gate",
    "evaluate_model_only_non_regression_gate",
    "evaluate_strict_lightweight_gate",
    "file_sha256",
    "paired_bootstrap_confidence_intervals",
    "percentile",
    "raw_fingerprint",
    "require_same_fingerprint",
    "semantic_sample_digest",
    "summarize_timings",
    "summarize_transfer_event_names",
    "validate_config_pair",
    "validate_common_preprocessor_boundary",
    "validate_expected_manifest",
    "validate_preregistration_plan",
    "validate_pipeline_memory_baseline_receipt",
    "validate_representation_config_pair",
    "validate_representation_ingress",
    "validate_live_representation_observation",
    "validate_representation_preregistration_plan",
    "validate_representation_protocol_freeze",
    "validate_representation_sequence_document",
    "validate_representation_snapshot",
    "validate_runtime_parameters",
    "validate_timed_common_boundary_receipt",
    "validate_transfer_reduction",
]
