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
PHYSICAL_ROUND_ORDER = (PCCP, CPPC, PCCP, CPPC, PCCP, CPPC)
PHYSICAL_ROUND_COUNT = len(PHYSICAL_ROUND_ORDER)
INVOCATIONS_PER_ARM_PER_BLOCK = 2
ITERATIONS_PER_ARM_GRANULARITY = (
    PHYSICAL_ROUND_COUNT * INVOCATIONS_PER_ARM_PER_BLOCK
)
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
        raise VerificationContractError(
            f"{name} must be an exact integer"
        ) from error
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

    capture_batches = _exact_integer(
        "capture_batches", capture_batches, minimum=1
    )
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
    iterations_per_segment = (
        iterations_per_arm // ITERATIONS_PER_ARM_GRANULARITY
    )
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


def _timing_pairs(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_pairs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected_pairs = _exact_integer(
        "expected_pairs", expected_pairs, minimum=1
    )
    if expected_pairs % ITERATIONS_PER_ARM_GRANULARITY:
        raise VerificationContractError(
            "expected_pairs must preserve equal six-round arm segments"
        )
    grouped: dict[int, dict[str, Mapping[str, Any]]] = {}
    for record in records:
        arm = record.get("arm")
        if arm not in (PARENT_ARM, CANDIDATE_ARM):
            raise VerificationContractError("timing record has an invalid arm")
        pair_index = _exact_integer(
            "pair_index", record.get("pair_index"), minimum=0
        )
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
    comparable_fields = (
        "phase",
        "round_index",
        "pattern",
        "segment_iteration_index",
        "batch_index",
        "pair_index",
    )
    for pair_index in range(expected_pairs):
        pair = grouped[pair_index]
        if set(pair) != {PARENT_ARM, CANDIDATE_ARM}:
            raise VerificationContractError(
                f"timing pair {pair_index} must contain exactly one P and one C"
            )
        parent = pair[PARENT_ARM]
        candidate = pair[CANDIDATE_ARM]
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
            _exact_integer(
                "slot_index", candidate.get("slot_index"), minimum=0
            ),
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

    expected_per_round, remainder = divmod(
        expected_pairs, PHYSICAL_ROUND_COUNT
    )
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
) -> dict[str, Any]:
    """Compute deterministic percentile CIs by resampling paired P/C calls."""

    bootstrap_samples = _exact_integer(
        "bootstrap_samples", bootstrap_samples, minimum=1
    )
    seed = _exact_integer("bootstrap seed", seed, minimum=0)
    parent_ns, candidate_ns, sample_counts = _timing_pairs(
        records,
        expected_pairs=expected_pairs,
    )
    point_estimates = {
        "median_candidate_over_parent": (
            percentile(candidate_ns.tolist(), 0.5)
            / percentile(parent_ns.tolist(), 0.5)
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
    bootstrap_values: dict[str, list[float]] = {
        name: [] for name in point_estimates
    }
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
            np.arange(PHYSICAL_ROUND_COUNT, dtype=np.int64)
            .reshape(1, PHYSICAL_ROUND_COUNT, 1)
            * pairs_per_round
        )
        indices = (within_round_indices + round_offsets).reshape(
            count, len(parent_ns)
        )
        parent_samples_ns = parent_ns[indices]
        candidate_samples_ns = candidate_ns[indices]
        sampled_counts = sample_counts[indices]
        parent_medians = np.median(parent_samples_ns, axis=1)
        candidate_medians = np.median(candidate_samples_ns, axis=1)
        parent_p95 = np.quantile(
            parent_samples_ns, 0.95, axis=1, method="linear"
        )
        candidate_p95 = np.quantile(
            candidate_samples_ns, 0.95, axis=1, method="linear"
        )
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
    p95_ratio = float(
        _nested(metrics, "p95_candidate_over_parent", "point_estimate")
    )
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
    if not all(math.isfinite(value) for value in (median_ratio, median_upper, p95_ratio)):
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
            "ADOPT_CANDIDATE"
            if passed
            else "RETAIN_PARENT_AND_REPORT_INCONCLUSIVE"
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
    p95_ratio = float(
        _nested(metrics, "p95_candidate_over_parent", "point_estimate")
    )
    if not all(math.isfinite(value) for value in (median_upper, p95_ratio)):
        raise VerificationContractError("near-native metrics must be finite")
    checks = {
        "median_ratio_ci95_upper_at_most_1_10": {
            "observed": median_upper,
            "threshold_max_inclusive": (
                NEAR_NATIVE_MEDIAN_RATIO_CI95_UPPER_MAXIMUM
            ),
            "passed": (
                median_upper <= NEAR_NATIVE_MEDIAN_RATIO_CI95_UPPER_MAXIMUM
            ),
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
            "NEAR_NATIVE_SUPPORTED"
            if passed
            else "NEAR_NATIVE_NOT_SUPPORTED"
        ),
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
            raise VerificationContractError(
                "config is missing " + ".".join(keys)
            )
        value = value[key]
    return value


def validate_preregistration_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the harness to the exact v1 future-CUDA measurement contract."""

    if not isinstance(plan, Mapping):
        raise VerificationContractError("preregistration plan must be a mapping")
    expected_schema = (
        "rosa.exacte94_lightweight_fresh30_runtime_preregistration.v1"
    )
    if plan.get("schema") != expected_schema:
        raise VerificationContractError("unexpected preregistration schema")
    if plan.get("status") != "PREPARATION_ONLY_DISARMED":
        raise VerificationContractError("preregistration status is not frozen")
    if plan.get("execution_authorized") is not False:
        raise VerificationContractError(
            "preregistration must remain non-authorizing"
        )
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
    if not isinstance(required_reporting, (list, tuple)) or set(
        required_reporting
    ) != expected_reporting:
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
    if not isinstance(environment_pins, (list, tuple)) or set(
        environment_pins
    ) != expected_environment_pins:
        raise VerificationContractError(
            "preregistration environment pins changed"
        )
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
        "upper_95ci_median_ratio_max": (
            NEAR_NATIVE_MEDIAN_RATIO_CI95_UPPER_MAXIMUM
        ),
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
        "minimum_measured_iterations_per_arm": (
            MINIMUM_MEASURED_ITERATIONS_PER_ARM
        ),
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
    "BOOTSTRAP_SAMPLES",
    "BOOTSTRAP_SEED",
    "CANDIDATE_ARM",
    "CPPC",
    "DEFAULT_MEASURED_ITERATIONS_PER_ARM",
    "DEFAULT_WARMUP_ITERATIONS_PER_ARM",
    "EXPECTED_CANDIDATE_PREPROCESSOR",
    "EXPECTED_HEAD_TYPE",
    "EXPECTED_MODEL_TYPE",
    "EXPECTED_PARENT_PREPROCESSOR",
    "HOST_METADATA_KEYS",
    "ITERATIONS_PER_ARM_GRANULARITY",
    "MINIMUM_MEASURED_ITERATIONS_PER_ARM",
    "MINIMUM_WARMUP_ITERATIONS_PER_ARM",
    "NEAR_NATIVE_MEDIAN_RATIO_CI95_UPPER_MAXIMUM",
    "NEAR_NATIVE_P95_RATIO_MAXIMUM",
    "PARENT_ARM",
    "PCCP",
    "PHYSICAL_ROUND_ORDER",
    "ScheduleEntry",
    "VerificationContractError",
    "build_alternating_schedule",
    "build_batch_manifest",
    "canonical_json",
    "canonical_sha256",
    "classify_transfer_event",
    "evaluate_c32_near_native_gate",
    "evaluate_strict_lightweight_gate",
    "file_sha256",
    "paired_bootstrap_confidence_intervals",
    "percentile",
    "raw_fingerprint",
    "require_same_fingerprint",
    "summarize_timings",
    "summarize_transfer_event_names",
    "validate_config_pair",
    "validate_expected_manifest",
    "validate_preregistration_plan",
    "validate_runtime_parameters",
    "validate_transfer_reduction",
]
