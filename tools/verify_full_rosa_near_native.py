#!/usr/bin/env python3
"""Disarmed native/ROSA full-model CUDA benchmark protocol.

The default command is a standard-library-only dry run.  It hashes and prints
the immutable experiment plan but does not import Torch, NumPy, MMEngine,
MMDetection, ROSA, or a CUDA extension; inspect a GPU; create an output
directory; or acquire a runtime lock.

The separately armed path is intentionally difficult to invoke accidentally.
It requires two explicit attestations, revalidates every source/checkpoint/DAF
input, rejects a non-default current stream and concurrent GPU processes, then
measures the exact same captured batches in one isolated native/compiled-fast,
native/compiled-fast-bounded, native/eager-fast, or native/balanced paired
alternating protocol.  The bounded arm is a separately labelled benchmark-only
route: output rows are matched by query ID, their labels and instance IDs remain
exact, and probabilities plus decoded boxes use fixed unit-based absolute
tolerances.  An eager-repeat baseline is recorded, and official full-metric
parity remains mandatory before promotion.  The default native detector is C0;
``--detector-pair c32`` selects
the exact official-residual C32 head on both sides of the ROSA-only derivation.
Candidate variants are deliberately separate process invocations with separate
output directories.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
from dataclasses import asdict
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from types import ModuleType, SimpleNamespace
from typing import Any, NamedTuple

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
VERIFY_C0_C32_TOOL = ROOT / "tools/verify_c0_c32_near_native.py"

PROTOCOL = "full-rosa-c0-fast-variants-balanced-near-native-v4"
C32_PROTOCOL = "full-rosa-c32-fast-variants-balanced-near-native-v3"
PLAN_SCHEMA = "rosa.full_rosa_near_native_plan.v4"
EVIDENCE_SCHEMA = "rosa.full_rosa_near_native_evidence.v4"
COMPLETION_SCHEMA = "rosa.full_rosa_near_native_completion.v4"
FAILURE_SCHEMA = "rosa.full_rosa_near_native_failure.v4"
IDLE_RESERVED_TOKEN = "I_HAVE_VERIFIED_THE_GPU_IS_IDLE_AND_RESERVED"
CURRENT_STREAM_TOKEN = "I_HAVE_VERIFIED_THE_CURRENT_STREAM_IS_DEFAULT"

DETECTOR_PAIR_C0 = "c0"
DETECTOR_PAIR_C32 = "c32"
DETECTOR_PAIRS = (DETECTOR_PAIR_C0, DETECTOR_PAIR_C32)
NATIVE_ARM = "c0_native"
C32_NATIVE_ARM = "c32_native"
NATIVE_ARMS = frozenset((NATIVE_ARM, C32_NATIVE_ARM))
FAST_ARM = "rosa_fast"
FAST_COMPILED_BOUNDED_ARM = "rosa_fast_compiled_bounded"
FAST_EAGER_ARM = "rosa_fast_eager"
BALANCED_ARM = "rosa_balanced"
FAST_MODEL_ARMS = frozenset((FAST_ARM, FAST_COMPILED_BOUNDED_ARM, FAST_EAGER_ARM))
COMPILED_ARMS = frozenset((FAST_ARM, FAST_COMPILED_BOUNDED_ARM))
EAGER_ARMS = frozenset((FAST_EAGER_ARM, BALANCED_ARM))
CANDIDATE_ARMS = frozenset((*FAST_MODEL_ARMS, BALANCED_ARM))
RAW_EXACT_COMPILED_ARMS = frozenset((FAST_ARM,))
BOUNDED_PARITY_COMPILED_ARMS = frozenset((FAST_COMPILED_BOUNDED_ARM,))
RAW_EXACT_FAST_COMPILE_MODE = "reduce-overhead"
BOUNDED_FAST_COMPILE_MODE = "max-autotune-no-cudagraphs"
COMPILED_FAST_FULLGRAPH = False
COMPILED_BOUNDED_FAST_FULLGRAPH = True
COMPILED_BOUNDED_QUERY_FIELD = "query_indices"
COMPILED_BOUNDED_EXACT_ASSOCIATION_FIELDS = (
    "labels_3d",
    "instance_ids",
)
COMPILED_BOUNDED_SCORE_FIELDS = (
    "scores_3d",
    "cls_scores",
)
COMPILED_BOUNDED_BOX_FIELD = "bboxes_3d"
COMPILED_BOUNDED_BOX_DIM = 10
COMPILED_BOUNDED_QUERY_COUNT = 300
COMPILED_BOUNDED_OUTPUT_LIMIT = COMPILED_BOUNDED_QUERY_COUNT
COMPILED_BOUNDED_SCORE_MAX_ABS = {
    "scores_3d": 3.5e-6,
    "cls_scores": 4.0e-6,
}
COMPILED_BOUNDED_BOX_MAX_ABS = 3.0e-3
COMPILED_BOUNDED_NUMERICAL_TIE_MAX_GAP = 7.0e-6
# The fixed ceilings round up the deterministic eager-repeat maxima while
# staying within a 1.65x envelope; no value is calibrated from the candidate.
COMPILED_BOUNDED_CALIBRATION_SCREEN = {
    "route": "eager_vs_eager",
    "passes": 4,
    "pairs": 6,
    "observed_max_abs": {
        "scores_3d": 2.3134053e-6,
        "cls_scores": 2.6524067e-6,
        "bboxes_3d": 1.82723999e-3,
    },
    "maximum_rounded_ceiling_multiplier": 1.65,
}
COMPILED_BOUNDED_BOX_MAX_REL = 1.0e-5
COMPILED_BOUNDED_BOX_RELATIVE_FLOOR = 1.0e-12
COMPILED_BOUNDED_BOX_FIELDS = (COMPILED_BOUNDED_BOX_FIELD,)
COMPILED_BOUNDED_RAW_EXACT_SCORE_FIELDS = COMPILED_BOUNDED_SCORE_FIELDS
COMPILED_BOUNDED_RAW_EXACT_INTEGER_FIELDS = (
    *COMPILED_BOUNDED_EXACT_ASSOCIATION_FIELDS,
    COMPILED_BOUNDED_QUERY_FIELD,
)
COMPILED_BOUNDED_RAW_EXACT_FIELDS = (
    *COMPILED_BOUNDED_RAW_EXACT_SCORE_FIELDS,
    *COMPILED_BOUNDED_RAW_EXACT_INTEGER_FIELDS,
)
COMPILED_PARITY_DIAGNOSTIC_SCHEMA = "rosa.compiled_parity_diagnostic.v2"
COMPILED_PARITY_DIAGNOSTIC_FILENAME = "COMPILED_PARITY_DIAGNOSTIC.json"
COMPILED_PARITY_DIAGNOSTIC_TOPK = 50
INDUCTOR_COMPILE_TASK_MODULE_PREFIX = "torch._inductor.runtime.compile_tasks."
WARMUP_ITERATIONS_PER_ARM = 60
MEASURED_ITERATIONS_PER_ARM = 300
CAPTURE_BATCHES = 2
EXPECTED_BATCH_SIZE = 4
SEED = 3407
BOOTSTRAP_SEED = 3407
BOOTSTRAP_SAMPLES = 20_000
EXPECTED_CHECKPOINT_REFINEMENT_PROFILE = "balanced"
EXPECTED_CHECKPOINT_ROSA_CONTRACT_VERSION = 3
EXPECTED_CHECKPOINT_RFL_TOPOLOGY_CONTRACT = "rosa.rfl.signed_gated_cp16.v3"
FAST_CHECKPOINT_ALLOWED_DELTAS = {
    "refine_topk": {"checkpoint": 100, "runtime": 50},
    "num_refinement_steps": {"checkpoint": 2, "runtime": 1},
}

PHYSICAL_ROUND_ORDER = ("PCCP", "CPPC", "PCCP", "CPPC", "PCCP", "CPPC")

DEFAULT_NATIVE_CONFIG = (
    ROOT / "configs/rosa_r18_main126_v2_c0_cumulative_runtime_exact.py"
)
DEFAULT_C32_NATIVE_CONFIG = ROOT / (
    "configs/rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_"
    "cpu_metadata_lean_loader_rgb_uint8_transfer_pinned_inplace_"
    "homogeneous4x4_inference_reuse_exact.py"
)
DEFAULT_DETECTOR_HOST_ROOT = Path(
    "/home/airlab/user/kdh/ROSA_host_wt_exacte94_native_eval_p1p2_strict"
)
DEFAULT_OFFICIAL_ROOT = Path("/home/airlab/user/kdh/dependencies/fisheye3dod")
EXPECTED_DETECTOR_HOST_REVISION = "ed800ce89336c9af28415974aeceebf0806ac588"
EXPECTED_OFFICIAL_REVISION = "a28726ed002a02cb1328960848c0f4f2473748de"
EXPECTED_OFFICIAL_SOURCE_SHA256 = (
    "007f8d0f1f2645fa425085f2368215978eef0ad60467fbc9f67c331d2b1913db"
)
DEFAULT_MAIN126_ANNOTATION = Path(
    "/home/airlab/user/kdh/datasets/Fisheye3DODdataset/"
    "ImageSets-main126/fisheye3dod_infos_val.pkl"
)
DEFAULT_COMPILED_BOUNDED_CALIBRATION_ARTIFACT = Path(
    "/home/airlab/user/kdh/ROSA/results/raw_metrics/benchmarks/diagnostics/"
    "20260803_native_atomic_repeatability_envelope/measurements.json"
)
EXPECTED_COMPILED_BOUNDED_CALIBRATION_SHA256 = (
    "3fec68c6d72ede2d12de9a13906b54bca56e6a06052f9ea993a3eb318397f6ea"
)
EXPECTED_MAIN126_ANNOTATION_SHA256 = (
    "efafe89c6dc34d116f0ce01fa818505a6387248d9a3ca6dcac035418a664da6f"
)
DEFAULT_DAF_EXTENSION_SHA256 = (
    "1831505bfb68183344ec8daa60c8d4d38bc806b60516bb3389e6190f463ffa30"
)
DAF_EXTENSION_BASENAME = "deformable_aggregation_ext.cpython-310-x86_64-linux-gnu.so"
BLESSED_DAF_BUILD_INPUT_SHA256 = {
    "projects/mmdet3d_plugin/ops/setup.py": (
        "f98cc5a3b35f6d454a33968a91c5a87c49023f50c9824fe7d05d86fdd4c1d598"
    ),
    "projects/mmdet3d_plugin/ops/src/deformable_aggregation.cpp": (
        "8685b1035b15c43c75914e9240681b33b2cd5fb852e56e06eb2e8812cbafd48f"
    ),
    "projects/mmdet3d_plugin/ops/src/deformable_aggregation_cuda.cu": (
        "7abb58450621fbe42395953397c480f1689cf6da315075b52d8a770e44168702"
    ),
}
BLESSED_DAF_WRAPPER_SHA256 = (
    "9695730cdefc1fbaf40cd990b08716d1dfd78b0f90eaef24a1aa76b78a4215b0"
)
DAF_SUPPORTED_CUDA_CAPABILITIES = (
    (8, 0),
    (8, 6),
    (8, 9),
    (9, 0),
    (12, 0),
)
ROSA_SIDECAR_STATE_PREFIX = "head.rosa_refiner."
HOST_SOURCE_FILES = (
    "projects/mmdet3d_plugin/__init__.py",
    "projects/mmdet3d_plugin/ops/__init__.py",
    "projects/mmdet3d_plugin/ops/deformable_aggregation.py",
    "projects/mmdet3d_plugin/models/sparse4d.py",
    "projects/mmdet3d_plugin/models/sparse4d_head.py",
    "projects/mmdet3d_plugin/models/instance_bank.py",
    "projects/mmdet3d_plugin/models/detection3d/decoder.py",
    "projects/mmdet3d_plugin/models/blocks.py",
)
HOST_DAF_SOURCE_FILES = (
    "projects/mmdet3d_plugin/ops/__init__.py",
    "projects/mmdet3d_plugin/ops/deformable_aggregation.py",
    "projects/mmdet3d_plugin/ops/setup.py",
    "projects/mmdet3d_plugin/ops/src/deformable_aggregation.cpp",
    "projects/mmdet3d_plugin/ops/src/deformable_aggregation_cuda.cu",
)

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
DEVICE_PATTERN = re.compile(r"^cuda:([0-9]+)$")
GPU_UUID_PATTERN = re.compile(r"^GPU-[0-9A-Fa-f-]+$")
ARMED_IMPORT_ROOTS = frozenset(
    {"torch", "numpy", "mmengine", "mmcv", "mmdet", "mmdet3d", "rosa"}
)


class ProtocolError(RuntimeError):
    """The preregistered runtime contract was violated."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument(
        "--detector-pair",
        choices=DETECTOR_PAIRS,
        default=DETECTOR_PAIR_C0,
        help=(
            "native detector shared by the pair; c0 preserves the historical "
            "default and c32 selects the exact official-residual wrapper"
        ),
    )
    parser.add_argument(
        "--native-config",
        type=Path,
        default=None,
        help="optional exact frozen native config for the selected detector pair",
    )
    parser.add_argument(
        "--detector-host-root", type=Path, default=DEFAULT_DETECTOR_HOST_ROOT
    )
    parser.add_argument("--expected-detector-host-revision", required=True)
    parser.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL_ROOT)
    parser.add_argument("--expected-official-revision", required=True)
    parser.add_argument(
        "--expected-daf-extension-sha256",
        default=DEFAULT_DAF_EXTENSION_SHA256,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--candidate-profile",
        choices=("fast", "fast_compiled_bounded", "fast_eager", "balanced"),
        default="fast",
        help=(
            "Run exactly one isolated C0-vs-ROSA candidate protocol; "
            "fast keeps raw-exact compiled/eager parity, "
            "fast_compiled_bounded is a benchmark-only fixed-tolerance arm, "
            "and fast_eager uses the exact fast model without torch.compile."
        ),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--execution-state", choices=("DISARMED", "ARMED"), default="DISARMED"
    )
    parser.add_argument("--idle-reserved-token")
    parser.add_argument("--current-stream-token")
    return parser


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    def encode(item: Any) -> str:
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"unsupported canonical JSON type: {type(item).__name__}")

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=encode,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_file(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise ProtocolError(f"{label} does not exist: {path}") from error
    if not resolved.is_file():
        raise ProtocolError(f"{label} must be a file: {resolved}")
    return resolved


def _resolve_directory(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise ProtocolError(f"{label} does not exist: {path}") from error
    if not resolved.is_dir():
        raise ProtocolError(f"{label} must be a directory: {resolved}")
    return resolved


def _validate_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ProtocolError(f"{label} must be 64 lowercase hexadecimal digits")
    return value


def _validate_revision(value: str, label: str) -> str:
    if not isinstance(value, str) or REVISION_PATTERN.fullmatch(value) is None:
        raise ProtocolError(f"{label} must be a full lowercase 40-character SHA")
    return value


def _run_git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _git_snapshot(root: Path) -> dict[str, Any]:
    return {
        "root": str(root),
        "revision": _run_git(root, "rev-parse", "HEAD"),
        "tree": _run_git(root, "rev-parse", "HEAD^{tree}"),
        "status": _run_git(root, "status", "--porcelain", "--untracked-files=all"),
    }


def _source_contract(root: Path, *, suffixes: tuple[str, ...]) -> dict[str, Any]:
    files: dict[str, str] = {}
    for path in sorted(
        candidate for candidate in root.rglob("*") if candidate.is_file()
    ):
        if path.suffix not in suffixes or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        files[relative] = file_sha256(path)
    if not files:
        raise ProtocolError(f"source contract is empty: {root}")
    return {"root": str(root), "files": files, "sha256": canonical_sha256(files)}


def _listed_source_contract(root: Path, relatives: Sequence[str]) -> dict[str, Any]:
    files = {
        relative: file_sha256(_resolve_file(root / relative, f"source {relative}"))
        for relative in relatives
    }
    return {"root": str(root), "files": files, "sha256": canonical_sha256(files)}


def _file_contract(path: Path, label: str) -> dict[str, Any]:
    resolved = _resolve_file(path, label)
    return {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def detector_pair_contract(detector_pair: str) -> dict[str, Any]:
    """Return the exact immutable-by-construction detector-pair contract."""

    if detector_pair == DETECTOR_PAIR_C0:
        return {
            "detector_pair": DETECTOR_PAIR_C0,
            "native_arm": NATIVE_ARM,
            "native_head_type": "ROSANativeBaselineHead",
            "candidate_head_type": "ROSARefinementHead",
            "native_profile": "native C0",
            "protocol": PROTOCOL,
        }
    if detector_pair == DETECTOR_PAIR_C32:
        return {
            "detector_pair": DETECTOR_PAIR_C32,
            "native_arm": C32_NATIVE_ARM,
            "native_head_type": "ROSAOfficialResidualNativeBaselineHead",
            "candidate_head_type": "ROSAOfficialResidualRefinementHead",
            "native_profile": "native C32 official residual",
            "protocol": C32_PROTOCOL,
        }
    raise ProtocolError("detector pair must be c0 or c32")


def selected_native_arm(detector_pair: str = DETECTOR_PAIR_C0) -> str:
    return str(detector_pair_contract(detector_pair)["native_arm"])


def default_native_config(detector_pair: str = DETECTOR_PAIR_C0) -> Path:
    detector_pair_contract(detector_pair)
    if detector_pair == DETECTOR_PAIR_C0:
        return DEFAULT_NATIVE_CONFIG
    return DEFAULT_C32_NATIVE_CONFIG


def _detector_pair_from_native_arm(native_arm: str) -> str:
    if native_arm == NATIVE_ARM:
        return DETECTOR_PAIR_C0
    if native_arm == C32_NATIVE_ARM:
        return DETECTOR_PAIR_C32
    raise ProtocolError("native arm must be c0_native or c32_native")


def selected_candidate(profile: str) -> str:
    if profile == "fast":
        return FAST_ARM
    if profile == "fast_compiled_bounded":
        return FAST_COMPILED_BOUNDED_ARM
    if profile == "fast_eager":
        return FAST_EAGER_ARM
    if profile == "balanced":
        return BALANCED_ARM
    raise ProtocolError(
        "candidate profile must be fast, fast_compiled_bounded, "
        "fast_eager, or balanced"
    )


def candidate_model_profile(candidate_arm: str) -> str:
    if candidate_arm in FAST_MODEL_ARMS:
        return "fast"
    if candidate_arm == BALANCED_ARM:
        return "balanced"
    raise ProtocolError("unknown ROSA candidate arm")


def candidate_execution_mode(candidate_arm: str) -> str:
    if candidate_arm in COMPILED_ARMS:
        return "compiled"
    if candidate_arm in EAGER_ARMS:
        return "eager"
    raise ProtocolError("unknown ROSA candidate arm")


def compiler_guard_contract(candidate_arm: str) -> dict[str, Any]:
    if candidate_arm in COMPILED_ARMS:
        return {
            "applicable": True,
            "requirement": (
                "torch._dynamo counters must remain raw-exact across the entire "
                "recorded warmup and measured timing window"
            ),
        }
    if candidate_arm == FAST_EAGER_ARM:
        return {
            "applicable": False,
            "reason": "fast_eager candidate is the uncompiled fast reference path",
        }
    if candidate_arm == BALANCED_ARM:
        return {
            "applicable": False,
            "reason": "balanced candidate is eager-only",
        }
    raise ProtocolError("unknown ROSA candidate arm")


def compiled_parity_contract(candidate_arm: str) -> dict[str, Any]:
    """Return the immutable eager/compiled correctness contract for one arm."""

    if candidate_arm in RAW_EXACT_COMPILED_ARMS:
        return {
            "applicable": True,
            "mode": "raw_exact_all_output_fields",
            "all_output_values_raw_exact": True,
        }
    if candidate_arm in BOUNDED_PARITY_COMPILED_ARMS:
        return {
            "applicable": True,
            "mode": "query_aligned_fixed_absolute_tolerance_v3",
            "all_output_values_raw_exact": False,
            "alignment_key": COMPILED_BOUNDED_QUERY_FIELD,
            "exact_fields_after_alignment": list(
                COMPILED_BOUNDED_EXACT_ASSOCIATION_FIELDS
            ),
            "bounded_float_fields": [
                *COMPILED_BOUNDED_SCORE_FIELDS,
                COMPILED_BOUNDED_BOX_FIELD,
            ],
            "absolute_thresholds": {
                **COMPILED_BOUNDED_SCORE_MAX_ABS,
                COMPILED_BOUNDED_BOX_FIELD: COMPILED_BOUNDED_BOX_MAX_ABS,
            },
            "failure_diagnostic": {
                "schema": COMPILED_PARITY_DIAGNOSTIC_SCHEMA,
                "filename": COMPILED_PARITY_DIAGNOSTIC_FILENAME,
                "atomic_before_strict_enforcement": True,
                "all_fields_collected_before_enforcement": True,
                "query_rank_prefix": COMPILED_PARITY_DIAGNOSTIC_TOPK,
                "raw_bit_distribution_is_diagnostic_only": True,
            },
            "threshold_basis": {
                "decoded_box": "3 mm, 0.003 rad, or 3 mm/s by component unit",
                "dynamic_candidate_calibration_forbidden": True,
                "fixed_calibration_screen": copy.deepcopy(
                    COMPILED_BOUNDED_CALIBRATION_SCREEN
                ),
            },
            "query_contract": {
                "maximum_count": COMPILED_BOUNDED_OUTPUT_LIMIT,
                "count_matches_decoder_output": True,
                "unique": True,
                "inclusive_range": [0, COMPILED_BOUNDED_QUERY_COUNT - 1],
                "set_exact_across_all_three_comparisons": True,
            },
            "rank_inversion_policy": {
                "maximum_reference_score_gap": (COMPILED_BOUNDED_NUMERICAL_TIE_MAX_GAP),
                "larger_gap_inversion_forbidden": True,
            },
            "required_comparisons": [
                "eager_reference_a_vs_eager_reference_b",
                "eager_reference_a_vs_compiled",
                "eager_reference_b_vs_compiled",
            ],
            "rank_displacement_recorded": True,
            "benchmark_only": True,
            "official_full_metric_parity_required_before_promotion": True,
        }
    if candidate_arm == FAST_EAGER_ARM:
        return {
            "applicable": False,
            "reason": "candidate is the uncompiled fast reference path",
        }
    if candidate_arm == BALANCED_ARM:
        return {
            "applicable": False,
            "reason": "balanced candidate is eager-only",
        }
    raise ProtocolError("unknown ROSA candidate arm")


def compiled_parity_calibration_contract(candidate_arm: str) -> dict[str, Any]:
    """Verify the eager-only artifact that fixed the bounded ceilings."""

    if candidate_arm not in BOUNDED_PARITY_COMPILED_ARMS:
        return {
            "applicable": False,
            "reason": "candidate does not use calibrated bounded parity",
        }
    artifact = _file_contract(
        DEFAULT_COMPILED_BOUNDED_CALIBRATION_ARTIFACT,
        "compiled parity calibration artifact",
    )
    if artifact["sha256"] != EXPECTED_COMPILED_BOUNDED_CALIBRATION_SHA256:
        raise ProtocolError("compiled parity calibration artifact hash drifted")
    return {
        "applicable": True,
        "verified_before_accelerator_import": True,
        "artifact": artifact,
        "source_revision": "bc282911f44884c27f5e7d1c552d38ed43c477ce",
        "checkpoint_sha256": (
            "8a56636d18a77fe9d01190d2a25d897b2225fca382d996cb3a472f5f45e9061e"
        ),
    }


def candidate_compile_contract(candidate_arm: str) -> dict[str, Any]:
    """Describe the exact compile route without exposing a user setting."""

    if candidate_arm in RAW_EXACT_COMPILED_ARMS:
        return {
            "applicable": True,
            "route": "frozen tools.run._compile_fast_inference",
            "kwargs": {
                "mode": RAW_EXACT_FAST_COMPILE_MODE,
                "fullgraph": COMPILED_FAST_FULLGRAPH,
            },
            "benchmark_local_override": False,
        }
    if candidate_arm in BOUNDED_PARITY_COMPILED_ARMS:
        return {
            "applicable": True,
            "route": "benchmark-local head.rosa_refiner.compile_fast_anchor",
            "kwargs": {
                "mode": BOUNDED_FAST_COMPILE_MODE,
                "fullgraph": COMPILED_BOUNDED_FAST_FULLGRAPH,
            },
            "compiled_boundary": "one-step anchor-only ROSA core",
            "scores_ranking_and_decode_remain_eager": True,
            "production_fast_route_unchanged": True,
            "benchmark_local_override": True,
        }
    if candidate_arm in EAGER_ARMS:
        return {
            "applicable": False,
            "reason": "candidate is eager-only",
        }
    raise ProtocolError("unknown ROSA candidate arm")


def _compile_candidate_refiner(run_tool, model, config, candidate_arm: str):
    """Compile one arm through its frozen production or benchmark-local route."""

    contract = candidate_compile_contract(candidate_arm)
    if not contract["applicable"]:
        raise ProtocolError(f"{candidate_arm} is not a compiled candidate")
    head = getattr(model, "head", None)
    refiner = getattr(head, "rosa_refiner", None)
    if refiner is None:
        raise ProtocolError(f"{candidate_arm} is missing its ROSA refiner")
    state_attribute = (
        "_compiled_fast_anchor_impl"
        if candidate_arm in BOUNDED_PARITY_COMPILED_ARMS
        else "_compiled_call_impl"
    )
    if getattr(refiner, state_attribute, None) is not None:
        raise ProtocolError(
            f"{candidate_arm} refiner was compiled before its audited compile call"
        )

    if candidate_arm in RAW_EXACT_COMPILED_ARMS:
        compiled = bool(
            run_tool._compile_fast_inference(
                SimpleNamespace(model=model),
                config,
            )
        )
        observed_route = "frozen tools.run._compile_fast_inference"
    elif candidate_arm in BOUNDED_PARITY_COMPILED_ARMS:
        compile_method = getattr(refiner, "compile_fast_anchor", None)
        if not callable(compile_method):
            raise ProtocolError(
                "bounded candidate refiner has no fast-anchor compile method"
            )
        kwargs = dict(contract["kwargs"])
        compile_method(**kwargs)
        compiled = getattr(refiner, state_attribute, None) is not None
        observed_route = "benchmark-local head.rosa_refiner.compile_fast_anchor"
    else:
        raise ProtocolError("compiled candidate has no compile route")

    if not compiled:
        raise ProtocolError(
            f"{candidate_arm} did not compile its ROSA refiner through the audited route"
        )
    return {
        "arm": candidate_arm,
        "compiled": True,
        "passed": True,
        "observed_route": observed_route,
        "observed_kwargs": dict(contract["kwargs"]),
    }


def selected_arms(
    profile: str, detector_pair: str = DETECTOR_PAIR_C0
) -> tuple[str, str]:
    return selected_native_arm(detector_pair), selected_candidate(profile)


def build_alternating_schedule(
    *,
    iterations_per_arm: int,
    phase: str,
    candidate_arm: str,
    batch_count: int,
    detector_pair: str = DETECTOR_PAIR_C0,
) -> tuple[dict[str, Any], ...]:
    if type(iterations_per_arm) is not int or iterations_per_arm <= 0:
        raise ProtocolError("iterations_per_arm must be a positive exact integer")
    if iterations_per_arm % 6:
        raise ProtocolError("iterations_per_arm must be divisible by six rounds")
    if type(batch_count) is not int or batch_count <= 0:
        raise ProtocolError("batch_count must be a positive exact integer")
    if not isinstance(phase, str) or not phase:
        raise ProtocolError("phase must be a nonempty string")
    if candidate_arm not in CANDIDATE_ARMS:
        raise ProtocolError(
            "candidate_arm must be raw-exact compiled fast, bounded-parity "
            "compiled fast, eager fast, or balanced"
        )
    native_arm = selected_native_arm(detector_pair)
    if iterations_per_arm % 12:
        raise ProtocolError("iterations_per_arm must be divisible by twelve")
    iterations_per_segment = iterations_per_arm // 12
    entries: list[dict[str, Any]] = []
    for round_index, pattern in enumerate(PHYSICAL_ROUND_ORDER):
        pairs_per_round = 2 * iterations_per_segment
        for slot_index, token in enumerate(pattern):
            pair_group = 0 if slot_index < 2 else 1
            arm = native_arm if token == "P" else candidate_arm
            for segment_iteration in range(iterations_per_segment):
                pair_index = (
                    round_index * pairs_per_round
                    + pair_group * iterations_per_segment
                    + segment_iteration
                )
                entries.append(
                    {
                        "phase": phase,
                        "round_index": round_index,
                        "pattern": pattern,
                        "segment_iteration_index": segment_iteration,
                        "slot_index": slot_index,
                        "pair_index": pair_index,
                        "batch_index": pair_index % batch_count,
                        "arm": arm,
                    }
                )
    expected = iterations_per_arm * 2
    if len(entries) != expected:
        raise ProtocolError("alternating schedule construction drifted")
    return tuple(entries)


def validate_execution_gate(args: argparse.Namespace) -> None:
    if not args.execute:
        return
    if args.execution_state != "ARMED":
        raise ProtocolError("--execute requires --execution-state ARMED")
    if args.idle_reserved_token != IDLE_RESERVED_TOKEN:
        raise ProtocolError("--execute requires the exact idle/reserved token")
    if args.current_stream_token != CURRENT_STREAM_TOKEN:
        raise ProtocolError("--execute requires the exact current-stream token")


def _reject_preimported_armed_modules() -> None:
    imported = sorted(
        name for name in sys.modules if name.split(".", 1)[0] in ARMED_IMPORT_ROOTS
    )
    if imported:
        raise ProtocolError(
            "armed execution requires a fresh standard-library-only process; "
            f"preimported module: {imported[0]}"
        )


def _gpu_selection_contract(args: argparse.Namespace) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not args.execute:
        return {
            "execution_requires_one_physical_uuid": True,
            "observed_cuda_visible_devices": visible,
            "physical_uuid": None,
        }
    if args.device != "cuda:0":
        raise ProtocolError("single-visible-GPU execution requires --device cuda:0")
    if visible is None:
        raise ProtocolError(
            "--execute requires CUDA_VISIBLE_DEVICES to contain one physical GPU UUID"
        )
    tokens = visible.split(",")
    if len(tokens) != 1 or GPU_UUID_PATTERN.fullmatch(tokens[0]) is None:
        raise ProtocolError(
            "CUDA_VISIBLE_DEVICES must be one exact GPU-... UUID during execution"
        )
    physical_uuid = _normalize_gpu_uuid(tokens[0])
    if tokens[0] != physical_uuid:
        raise ProtocolError(
            "CUDA_VISIBLE_DEVICES must use the canonical GPU- prefix and "
            "lower-case UUID suffix"
        )
    return {
        "execution_requires_one_physical_uuid": True,
        "observed_cuda_visible_devices": visible,
        "physical_uuid": physical_uuid,
        "logical_device": "cuda:0",
        "all_visible_gpu_rng_calls_confined_to_locked_device": True,
    }


def _arms_plan(
    arms: Sequence[str], detector_pair: str = DETECTOR_PAIR_C0
) -> dict[str, Any]:
    pair = detector_pair_contract(detector_pair)
    native_arm = pair["native_arm"]
    if len(arms) != 2 or arms[0] != native_arm or arms[1] not in CANDIDATE_ARMS:
        raise ProtocolError("arms do not match the selected detector pair")
    records: dict[str, Any] = {
        native_arm: {
            "profile": pair["native_profile"],
            "detector_pair": detector_pair,
            "head_type": pair["native_head_type"],
            "config_derivation": "unchanged frozen native config",
            "compile": False,
        }
    }
    if FAST_ARM in arms:
        records[FAST_ARM] = {
            "profile": "fast",
            "model_profile": "fast",
            "execution_mode": "compiled",
            "detector_pair": detector_pair,
            "head_type": pair["candidate_head_type"],
            "config_derivation": {
                "base": native_arm,
                "only_changes": {
                    "model.head.type": pair["candidate_head_type"],
                    "model.head.rosa_cfg": {"profile": "fast"},
                },
            },
            "compile": {
                "enabled": True,
                "target": "head.rosa_refiner",
                "mode": RAW_EXACT_FAST_COMPILE_MODE,
                "fullgraph": COMPILED_FAST_FULLGRAPH,
                "compilation_excluded_from_timing": True,
                "invocation_contract": candidate_compile_contract(FAST_ARM),
            },
            "compiled_eager_parity": compiled_parity_contract(FAST_ARM),
        }
    if FAST_COMPILED_BOUNDED_ARM in arms:
        records[FAST_COMPILED_BOUNDED_ARM] = {
            "profile": "fast_compiled_bounded",
            "model_profile": "fast",
            "execution_mode": "compiled",
            "detector_pair": detector_pair,
            "head_type": pair["candidate_head_type"],
            "config_derivation": {
                "base": native_arm,
                "only_changes": {
                    "model.head.type": pair["candidate_head_type"],
                    "model.head.rosa_cfg": {"profile": "fast"},
                },
            },
            "compile": {
                "enabled": True,
                "target": "head.rosa_refiner",
                "mode": BOUNDED_FAST_COMPILE_MODE,
                "fullgraph": COMPILED_FAST_FULLGRAPH,
                "compilation_excluded_from_timing": True,
                "invocation_contract": candidate_compile_contract(
                    FAST_COMPILED_BOUNDED_ARM
                ),
            },
            "compiled_eager_parity": compiled_parity_contract(
                FAST_COMPILED_BOUNDED_ARM
            ),
        }
    if FAST_EAGER_ARM in arms:
        records[FAST_EAGER_ARM] = {
            "profile": "fast_eager",
            "model_profile": "fast",
            "execution_mode": "eager",
            "detector_pair": detector_pair,
            "head_type": pair["candidate_head_type"],
            "config_derivation": {
                "base": native_arm,
                "only_changes": {
                    "model.head.type": pair["candidate_head_type"],
                    "model.head.rosa_cfg": {"profile": "fast"},
                },
            },
            "compile": False,
            "compile_forbidden": True,
            "compile_state_attestation": (
                "head.rosa_refiner._compiled_call_impl must remain None"
            ),
        }
    if BALANCED_ARM in arms:
        records[BALANCED_ARM] = {
            "profile": "balanced",
            "model_profile": "balanced",
            "execution_mode": "eager",
            "detector_pair": detector_pair,
            "head_type": pair["candidate_head_type"],
            "config_derivation": {
                "base": native_arm,
                "only_changes": {
                    "model.head.type": pair["candidate_head_type"],
                    "model.head.rosa_cfg": {"profile": "balanced"},
                },
            },
            "compile": False,
        }
    return records


def prepare_plan(
    args: argparse.Namespace, *, allow_existing_output: bool = False
) -> dict[str, Any]:
    if not isinstance(allow_existing_output, bool):
        raise TypeError("allow_existing_output must be bool")
    pair = detector_pair_contract(args.detector_pair)
    detector_pair = pair["detector_pair"]
    native_arm = pair["native_arm"]
    expected_checkpoint = _validate_sha256(
        args.expected_checkpoint_sha256, "expected checkpoint SHA256"
    )
    expected_daf = _validate_sha256(
        args.expected_daf_extension_sha256, "expected DAF extension SHA256"
    )
    expected_source = _validate_revision(
        args.expected_source_revision, "expected source revision"
    )
    expected_host = _validate_revision(
        args.expected_detector_host_revision, "expected detector-host revision"
    )
    expected_official = _validate_revision(
        args.expected_official_revision, "expected official revision"
    )
    if expected_daf != DEFAULT_DAF_EXTENSION_SHA256:
        raise ProtocolError("DAF SHA256 must match the blessed binary")
    if expected_host != EXPECTED_DETECTOR_HOST_REVISION:
        raise ProtocolError("detector-host revision is not the frozen revision")
    if expected_official != EXPECTED_OFFICIAL_REVISION:
        raise ProtocolError("official revision is not the frozen revision")
    if DEVICE_PATTERN.fullmatch(args.device) is None:
        raise ProtocolError("device must have exact form cuda:N")
    gpu_selection = _gpu_selection_contract(args)

    checkpoint = _file_contract(args.checkpoint, "checkpoint")
    if checkpoint["sha256"] != expected_checkpoint:
        raise ProtocolError("checkpoint hash differs from the expected SHA256")
    expected_native_config = default_native_config(detector_pair).resolve(strict=True)
    requested_native_config = (
        expected_native_config if args.native_config is None else args.native_config
    )
    native_config_path = _resolve_file(requested_native_config, "native config")
    if native_config_path != expected_native_config:
        raise ProtocolError(
            f"native config must be the frozen cumulative {detector_pair.upper()} config"
        )
    native_config = _file_contract(native_config_path, "native config")
    annotation = _file_contract(DEFAULT_MAIN126_ANNOTATION, "Main126 annotation")
    if annotation["sha256"] != EXPECTED_MAIN126_ANNOTATION_SHA256:
        raise ProtocolError("Main126 annotation differs from the frozen dataset")
    detector_root = _resolve_directory(args.detector_host_root, "detector host")
    official_root = _resolve_directory(args.official_root, "official repository")
    repository = _git_snapshot(ROOT)
    detector_git = _git_snapshot(detector_root)
    official_git = _git_snapshot(official_root)
    if repository["revision"] != expected_source:
        raise ProtocolError("source HEAD differs from --expected-source-revision")
    if detector_git["revision"] != expected_host:
        raise ProtocolError("detector-host HEAD differs from its expected revision")
    if official_git["revision"] != expected_official:
        raise ProtocolError("official HEAD differs from its expected revision")

    detector_daf_sources = _listed_source_contract(detector_root, HOST_DAF_SOURCE_FILES)
    for relative, digest in BLESSED_DAF_BUILD_INPUT_SHA256.items():
        if detector_daf_sources["files"].get(relative) != digest:
            raise ProtocolError(f"DAF build input drifted: {relative}")
    wrapper_relative = "projects/mmdet3d_plugin/ops/deformable_aggregation.py"
    if (
        detector_daf_sources["files"].get(wrapper_relative)
        != BLESSED_DAF_WRAPPER_SHA256
    ):
        raise ProtocolError("DAF Python wrapper drifted")
    official_sources = _source_contract(official_root, suffixes=(".py",))
    if official_sources["sha256"] != EXPECTED_OFFICIAL_SOURCE_SHA256:
        raise ProtocolError("patched official Python source contract drifted")

    expected_daf_path = (
        detector_root / "projects/mmdet3d_plugin/ops" / DAF_EXTENSION_BASENAME
    )
    try:
        expected_daf_path.lstat()
    except FileNotFoundError as error:
        raise ProtocolError(
            f"DAF extension binary does not exist: {expected_daf_path}"
        ) from error
    if expected_daf_path.is_symlink():
        raise ProtocolError("DAF extension must be a regular non-symlink file")
    daf_path = _resolve_file(expected_daf_path, "DAF extension binary")
    daf_binary = _file_contract(daf_path, "DAF extension binary")
    if daf_binary["sha256"] != expected_daf:
        raise ProtocolError("DAF extension hash differs from the expected SHA256")

    output_dir = args.output_dir.expanduser().resolve()
    if _is_within(output_dir, ROOT):
        raise ProtocolError("benchmark output must be outside the source worktree")
    if not output_dir.parent.is_dir():
        raise ProtocolError("benchmark output parent must already exist")
    if output_dir.exists() and not allow_existing_output:
        raise ProtocolError("benchmark output path already exists")

    arms = selected_arms(args.candidate_profile, detector_pair)
    candidate_arm = selected_candidate(args.candidate_profile)
    calibration_contract = compiled_parity_calibration_contract(candidate_arm)
    schedules = {}
    for phase, iterations in (
        ("warmup", WARMUP_ITERATIONS_PER_ARM),
        ("measured", MEASURED_ITERATIONS_PER_ARM),
    ):
        schedule = build_alternating_schedule(
            iterations_per_arm=iterations,
            phase=phase,
            candidate_arm=candidate_arm,
            batch_count=CAPTURE_BATCHES,
            detector_pair=detector_pair,
        )
        schedules[phase] = {
            "iterations_per_arm": iterations,
            "total_invocations": len(schedule),
            "sha256": canonical_sha256(schedule),
        }
    comparisons = {
        candidate_arm: {
            "detector_pair": detector_pair,
            "parent_arm": native_arm,
            "candidate_arm": candidate_arm,
            "physical_round_order": list(PHYSICAL_ROUND_ORDER),
            "schedules": schedules,
            "one_candidate_per_process": True,
            "compiled_eager_parity": compiled_parity_contract(candidate_arm),
        }
    }

    return {
        "schema": PLAN_SCHEMA,
        "status": "DRY_RUN",
        "protocol": pair["protocol"],
        "detector_pair": detector_pair,
        "detector_pair_contract": {
            **pair,
            "candidate_derivation_only_changes": [
                "model.head.type",
                "model.head.rosa_cfg",
            ],
            "shared_backbone": "ResNet-18",
            "shared_dataset": "Fisheye3DOD Main126 validation",
            "shared_schedule_and_effective_config_outside_head": True,
            "shared_checkpoint_path_and_sha256": True,
            "shared_native_state_raw_exact": True,
        },
        "execution_requested": bool(args.execute),
        "execution_state": args.execution_state,
        "accelerator_modules_imported": False,
        "cuda_queried": False,
        "output_created": False,
        "repository": repository,
        "rosa_source": _source_contract(SRC_ROOT, suffixes=(".py",)),
        "tool": _file_contract(Path(__file__), "benchmark tool"),
        "native_config": native_config,
        "dataset": {
            "name": "Fisheye3DOD Main126 validation",
            "type": "Fisheye3DODSparse4DDataset",
            "annotation": annotation,
            "expected_batch_size": EXPECTED_BATCH_SIZE,
            "captured_batch_count": CAPTURE_BATCHES,
        },
        "effective_configs": "materialized and hashed only after armed imports",
        "checkpoint": checkpoint,
        "checkpoint_schema": "read only after armed imports",
        "detector_host": {
            "git": detector_git,
            "source_contract": _listed_source_contract(
                detector_root, HOST_SOURCE_FILES
            ),
            "daf_source_contract": detector_daf_sources,
            "daf_binary": daf_binary,
            "loaded_modules": "verified only after armed imports",
        },
        "official_fisheye3dod": {
            "git": official_git,
            "source_contract": official_sources,
            "locally_patched_worktree_allowed": True,
            "status_and_python_source_hashes_frozen_across_timing": True,
            "loaded_modules": "verified only after armed imports",
        },
        "device": args.device,
        "gpu_selection": gpu_selection,
        "output_dir": str(output_dir),
        "candidate_profile": args.candidate_profile,
        "candidate_model_profile": candidate_model_profile(candidate_arm),
        "candidate_execution_mode": candidate_execution_mode(candidate_arm),
        "arms": _arms_plan(arms, detector_pair),
        "comparisons": comparisons,
        "compiled_parity_calibration": calibration_contract,
        "parameters": {
            "capture_batches": CAPTURE_BATCHES,
            "expected_batch_size": EXPECTED_BATCH_SIZE,
            "warmup_iterations_per_arm": WARMUP_ITERATIONS_PER_ARM,
            "measured_iterations_per_arm": MEASURED_ITERATIONS_PER_ARM,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "seed": SEED,
        },
        "same_input_contract": {
            "capture_once_from": native_arm,
            "captured_batches": CAPTURE_BATCHES,
            "replay": "independent pinned CPU clone outside timed interval",
            "pairing": (
                "native and candidate with one pair_index receive the same "
                "captured batch in each independent protocol"
            ),
            "one_candidate_profile_per_process": True,
            "same_checkpoint_path_and_sha256_for_every_arm": True,
        },
        "measurement": {
            "scope": "full model.test_step",
            "includes": [
                "data preprocessing",
                "host-to-device transfer",
                "backbone",
                "neck",
                "native Sparse4D head",
                "ROSA sidecar when selected",
                "decode and device-to-host result materialization",
            ],
            "excluded": [
                "source batch cloning",
                "model/config/checkpoint construction",
                "torch.compile compilation",
                "output validation and serialization",
            ],
            "timer": "perf_counter_ns bracketed by device synchronizations",
            "statistics": [
                "mean",
                "median",
                "linear p95",
                "six-round stratified paired-bootstrap 95% CI",
            ],
            "memory": (
                "three probes per arm with exactly one benchmark model resident; "
                "maximum allocated/reserved and their increments; documented "
                "cuBLAS workspaces cleared only between isolated arm builds"
            ),
            "compiler_guard": compiler_guard_contract(candidate_arm),
            "candidate_compile": candidate_compile_contract(candidate_arm),
        },
        "near_native_gate": {
            "source": "existing C0/C32 full-detector gate",
            "detector_pair": detector_pair,
            "ratio": f"ROSA candidate / {detector_pair.upper()} native",
            "scope": "full model.test_step only",
            "median_ratio_ci95_upper_max": 1.10,
            "p95_ratio_max": 1.10,
            "peak_allocated_ratio_max": 1.02,
            "peak_reserved_ratio_max": 1.02,
            "failed_gate_policy": "make no near-native claim and report measurements",
        },
        "output_contract": {
            "sample_count": EXPECTED_BATCH_SIZE,
            "fields": [
                "bboxes_3d",
                "scores_3d",
                "cls_scores",
                "labels_3d",
                "query_indices",
                "instance_ids",
            ],
            "box_and_score_values_finite": True,
            "labels_integral_and_in_six_class_range": True,
            "compiled_eager_parity": compiled_parity_contract(candidate_arm),
            "preflight_only_not_in_timed_interval": True,
            "bounded_payload": "owned detached CPU copies, released before timing",
        },
        "promotion_contract": {
            "official_full_metric_parity_required_before_promotion": (
                candidate_arm in BOUNDED_PARITY_COMPILED_ARMS
            ),
            "bounded_arm_status": (
                "PENDING_OFFICIAL_FULL_METRIC_PARITY"
                if candidate_arm in BOUNDED_PARITY_COMPILED_ARMS
                else "NOT_APPLICABLE"
            ),
        },
        "guards": {
            "idle_token_required": IDLE_RESERVED_TOKEN,
            "current_stream_token_required": CURRENT_STREAM_TOKEN,
            "current_stream_must_equal_default": True,
            "physical_gpu_selected_by_torch_uuid": True,
            "single_visible_physical_gpu_uuid_required": True,
            "cooperative_physical_lock_acquired_before_accelerator_import": True,
            "foreign_compute_processes_rejected_at_recorded_boundaries": True,
            "continuous_exclusion_relies_on_idle_reserved_attestation": True,
            "source_checkpoint_config_and_daf_revalidated_before_import": True,
            "exclusive_lock_keyed_by_physical_gpu_uuid": True,
            "daf_supported_cuda_capabilities": [
                list(value) for value in DAF_SUPPORTED_CUDA_CAPABILITIES
            ],
        },
        "claim_status": "NO_NEAR_NATIVE_CLAIM_UNTIL_COMPLETE_IDLE_GPU_RUN",
    }


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_text(path: Path, payload: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    existed_before = path.exists()
    published = False
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        published = True
        _fsync_directory(path.parent)
    except BaseException:
        if published and not existed_before:
            path.unlink(missing_ok=True)
            try:
                _fsync_directory(path.parent)
            except OSError:
                pass
        raise
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _seal_directory(output_dir: Path) -> None:
    files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [
        f"{file_sha256(path)}  {path.relative_to(output_dir).as_posix()}"
        for path in files
    ]
    seal = output_dir / "SHA256SUMS"
    _atomic_text(seal, "\n".join(lines) + "\n")


def _load_armed_verifier():
    """Import the established Torch verifier only beyond the arming boundary."""

    name = "rosa_full_speed_armed_c0_c32_helpers"
    specification = importlib.util.spec_from_file_location(name, VERIFY_C0_C32_TOOL)
    if specification is None or specification.loader is None:
        raise ProtocolError("could not load armed verifier helpers")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _normalize_gpu_uuid(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="strict")
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError("Torch device properties did not expose a GPU UUID")
    normalized = value.strip()
    if normalized.upper().startswith("GPU-"):
        normalized = normalized[4:]
    # ``nvidia-smi --id`` treats the hexadecimal UUID suffix as
    # case-sensitive on the deployed driver.  Its own query output uses a
    # lower-case suffix, so keep one comparison/selector representation here.
    return "GPU-" + normalized.lower()


def _logical_device_identity(torch, device) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    raw_uuid = getattr(properties, "uuid", None)
    if not isinstance(raw_uuid, (str, bytes)):
        torch_cuuid = getattr(getattr(torch, "_C", None), "_CUuuid", None)
        if torch_cuuid is None or type(raw_uuid) is not torch_cuuid:
            raise ProtocolError("Torch device properties did not expose a GPU UUID")
        # PyTorch 2.11 exposes UUID as a private immutable ``_CUuuid`` value.
        # Its string representation is the CUDA UUID suffix printed in the
        # device-properties repr.  Convert only that exact Torch-owned type;
        # accepting arbitrary ``__str__`` objects would weaken attestation.
        raw_uuid = str(raw_uuid)
    return {
        "logical_device": str(device),
        "logical_index": device.index,
        "gpu_uuid": _normalize_gpu_uuid(raw_uuid),
        "torch_name": properties.name,
    }


def _nvidia_smi_snapshot(gpu_uuid: str) -> dict[str, Any]:
    gpu_uuid = _normalize_gpu_uuid(gpu_uuid)
    gpu_fields = (
        "timestamp",
        "index",
        "uuid",
        "name",
        "driver_version",
        "pstate",
        "power.draw",
        "power.limit",
        "clocks.current.graphics",
        "clocks.current.memory",
    )
    gpu = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_uuid}",
            "--query-gpu=" + ",".join(gpu_fields),
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    rows = [row.strip() for row in gpu.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise ProtocolError("nvidia-smi did not return exactly one selected GPU")
    values = [value.strip() for value in rows[0].split(",")]
    uuid_index = gpu_fields.index("uuid")
    if (
        len(values) != len(gpu_fields)
        or _normalize_gpu_uuid(values[uuid_index]) != gpu_uuid
    ):
        raise ProtocolError("nvidia-smi UUID differs from the Torch logical device")
    app_fields = ("pid", "process_name", "used_gpu_memory", "gpu_uuid")
    apps = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_uuid}",
            "--query-compute-apps=" + ",".join(app_fields),
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    app_rows = [row.strip() for row in apps.stdout.splitlines() if row.strip()]
    foreign = []
    for row in app_rows:
        try:
            pid = int(row.split(",", 1)[0].strip())
        except ValueError as error:
            raise ProtocolError(
                f"could not parse compute process row: {row}"
            ) from error
        if pid != os.getpid():
            foreign.append(row)
    if foreign:
        raise ProtocolError("selected physical GPU has a foreign compute process")
    return {
        "gpu_fields": list(gpu_fields),
        "gpu_row": rows[0],
        "selected_gpu_uuid": gpu_uuid,
        "compute_app_fields": list(app_fields),
        "compute_app_rows": app_rows,
        "foreign_compute_processes": 0,
        "current_pid": os.getpid(),
        "wall_time_ns": time.time_ns(),
    }


def _acquire_physical_gpu_lock(gpu_uuid: str) -> tuple[int, Path]:
    gpu_uuid = _normalize_gpu_uuid(gpu_uuid)
    safe_uuid = re.sub(r"[^A-Za-z0-9_.-]", "_", gpu_uuid)
    # A fixed system-wide location prevents two runs with different output
    # parents from benchmarking the same physical GPU concurrently.
    lock = Path("/tmp") / f".full_rosa_near_native_{safe_uuid}.lock"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock, flags, 0o600)
    except FileExistsError as error:
        raise ProtocolError(f"exclusive physical-GPU lock exists: {lock}") from error
    payload = memoryview(f"pid={os.getpid()}\ngpu_uuid={gpu_uuid}\n".encode("ascii"))
    try:
        while payload:
            written = os.write(descriptor, payload)
            if written <= 0:
                raise OSError("physical-GPU lock write made no progress")
            payload = payload[written:]
        os.fsync(descriptor)
    except BaseException as error:
        try:
            _release_physical_gpu_lock(descriptor, lock)
        except OSError as cleanup_error:
            if hasattr(error, "add_note"):
                error.add_note(f"lock cleanup also failed: {cleanup_error}")
        raise
    return descriptor, lock


def _release_physical_gpu_lock(descriptor: int, lock: Path) -> None:
    try:
        owned = os.fstat(descriptor)
        try:
            observed = lock.stat(follow_symlinks=False)
        except FileNotFoundError:
            observed = None
    finally:
        os.close(descriptor)
    if observed is not None and (observed.st_dev, observed.st_ino) == (
        owned.st_dev,
        owned.st_ino,
    ):
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def _expected_candidate_config(
    native_config: Mapping[str, Any],
    profile: str,
    detector_pair: str = DETECTOR_PAIR_C0,
) -> dict[str, Any]:
    if profile not in ("fast", "balanced"):
        raise ProtocolError("candidate profile must be fast or balanced")
    pair = detector_pair_contract(detector_pair)
    expected = copy.deepcopy(dict(native_config))
    model = expected.get("model")
    if not isinstance(model, Mapping):
        raise ProtocolError("native config has no model mapping")
    head = model.get("head")
    if not isinstance(head, Mapping):
        raise ProtocolError("native config has no model.head mapping")
    if head.get("type") != pair["native_head_type"]:
        raise ProtocolError(
            f"native config is not the exact {detector_pair.upper()} head"
        )
    if head.get("rosa_cfg") is not None:
        raise ProtocolError("native config unexpectedly contains ROSA sidecar settings")
    head["type"] = pair["candidate_head_type"]
    head["rosa_cfg"] = {"profile": profile}
    return expected


def _derive_configs(
    verifier,
    native_config_path: Path,
    arms: Sequence[str],
    detector_pair: str = DETECTOR_PAIR_C0,
):
    pair = detector_pair_contract(detector_pair)
    native_arm = pair["native_arm"]
    if len(arms) != 2 or arms[0] != native_arm or arms[1] not in CANDIDATE_ARMS:
        raise ProtocolError("config arms do not match the selected detector pair")
    native = verifier.Config.fromfile(native_config_path, import_custom_modules=False)
    native_dict = native.to_dict()
    configs = {native_arm: native}
    for arm in arms:
        if arm == native_arm:
            continue
        profile = candidate_model_profile(arm)
        candidate = copy.deepcopy(native)
        candidate.model.head.type = pair["candidate_head_type"]
        candidate.model.head.rosa_cfg = {"profile": profile}
        expected = _expected_candidate_config(native_dict, profile, detector_pair)
        if candidate.to_dict() != expected:
            raise ProtocolError(
                f"{arm} config differs from {detector_pair.upper()} outside "
                "head.type and head.rosa_cfg"
            )
        configs[arm] = candidate
    return configs


def _validate_effective_dataset_configs(
    configs: Mapping[str, Any], planned_annotation: Path
) -> None:
    planned = planned_annotation.resolve(strict=True)
    for arm, config in configs.items():
        loader = config.test_dataloader
        dataset = loader.dataset
        if int(loader.batch_size) != EXPECTED_BATCH_SIZE:
            raise ProtocolError(f"{arm} test batch size is not {EXPECTED_BATCH_SIZE}")
        if dataset.type != "Fisheye3DODSparse4DDataset":
            raise ProtocolError(f"{arm} uses the wrong validation dataset type")
        observed = Path(dataset.ann_file).expanduser().resolve(strict=True)
        if observed != planned:
            raise ProtocolError(f"{arm} resolves a different Main126 annotation")


def _validate_native_checkpoint_exclusions(excluded: Sequence[str]) -> None:
    invalid = [
        name
        for name in excluded
        if not isinstance(name, str) or not name.startswith(ROSA_SIDECAR_STATE_PREFIX)
    ]
    if invalid:
        raise ProtocolError(
            "native checkpoint filtering would discard non-ROSA keys: "
            + ", ".join(str(name) for name in invalid[:5])
        )


def _register_runtime_modules(configs: Mapping[str, Any]) -> None:
    from mmengine.utils import import_modules_from_strings
    from mmdet3d.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    for config in configs.values():
        import_modules_from_strings(**config.custom_imports)


def _validate_loaded_sources(
    *,
    loaded: Mapping[str, Any],
    planned_files: Mapping[str, str],
    label: str,
) -> None:
    observed = loaded.get("files")
    if not isinstance(observed, Mapping) or not observed:
        raise ProtocolError(f"loaded {label} module contract is empty")
    for record in observed.values():
        relative = record.get("relative_path")
        digest = record.get("sha256")
        if planned_files.get(relative) != digest:
            raise ProtocolError(
                f"loaded {label} module differs from planned source: {relative}"
            )


def _validate_loaded_provenance(
    loaded: Mapping[str, Any], plan: Mapping[str, Any]
) -> None:
    _validate_loaded_sources(
        loaded=loaded["rosa_modules"],
        planned_files=plan["rosa_source"]["files"],
        label="ROSA",
    )
    _validate_loaded_sources(
        loaded=loaded["host_modules"],
        planned_files=plan["detector_host"]["source_contract"]["files"],
        label="detector-host",
    )
    _validate_loaded_sources(
        loaded=loaded["official_modules"],
        planned_files=plan["official_fisheye3dod"]["source_contract"]["files"],
        label="official Fisheye3DOD",
    )
    daf = loaded["daf_extension"]
    if daf.get("sha256") != plan["detector_host"]["daf_binary"]["sha256"]:
        raise ProtocolError("loaded DAF binary differs from the planned binary")
    if (
        daf.get("source_contract", {}).get("files")
        != plan["detector_host"]["daf_source_contract"]["files"]
    ):
        raise ProtocolError("loaded DAF source closure differs from the plan")


def _capture_loaded_provenance(
    verifier,
    *,
    detector_root: Path,
    official_root: Path,
    expected_daf_sha256: str,
) -> dict[str, Any]:
    return {
        "rosa_modules": verifier._loaded_rosa_module_contract(SRC_ROOT),
        "host_modules": verifier._loaded_host_module_contract(detector_root),
        "daf_extension": verifier._loaded_daf_extension_contract(
            detector_root, expected_sha256=expected_daf_sha256
        ),
        "official_modules": verifier._loaded_official_module_contract(official_root),
    }


def _attest_armed_schedules(verifier, plan: Mapping[str, Any]) -> None:
    pair = detector_pair_contract(plan.get("detector_pair"))
    if plan.get("detector_pair_contract", {}).get("native_arm") != pair["native_arm"]:
        raise ProtocolError("planned detector-pair native arm drifted")
    native_arm = pair["native_arm"]
    for candidate_arm, comparison in plan["comparisons"].items():
        for phase, iterations in (
            ("warmup", WARMUP_ITERATIONS_PER_ARM),
            ("measured", MEASURED_ITERATIONS_PER_ARM),
        ):
            serialized = []
            for entry in verifier.build_alternating_schedule(
                iterations_per_arm=iterations,
                batch_count=CAPTURE_BATCHES,
                phase=phase,
            ):
                record = entry.to_dict()
                record["arm"] = (
                    native_arm
                    if record["arm"] == verifier.PARENT_ARM
                    else candidate_arm
                )
                serialized.append(record)
            expected = comparison["schedules"][phase]
            if (
                len(serialized) != expected["total_invocations"]
                or canonical_sha256(serialized) != expected["sha256"]
            ):
                raise ProtocolError(
                    f"armed {candidate_arm} {phase} schedule differs from dry-run"
                )


def _checkpoint_copy(state_dict):
    copied = copy.copy(state_dict)
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        copied._metadata = copy.copy(metadata)
    return copied


def _configured_head_contract(config) -> tuple[Any, Any]:
    model = (
        config.get("model")
        if isinstance(config, Mapping)
        else getattr(config, "model", None)
    )
    if model is None:
        raise ProtocolError("candidate config has no model")
    head = (
        model.get("head")
        if isinstance(model, Mapping)
        else getattr(model, "head", None)
    )
    if head is None:
        raise ProtocolError("candidate config has no model.head")
    if isinstance(head, Mapping):
        return head.get("type"), head.get("rosa_cfg")
    return getattr(head, "type", None), getattr(head, "rosa_cfg", None)


def _validate_checkpoint_semantics(
    run_tool,
    checkpoint,
    config,
    arm: str,
    *,
    detector_pair: str | None = None,
):
    pair_name = DETECTOR_PAIR_C0 if detector_pair is None else detector_pair
    pair = detector_pair_contract(pair_name)
    if detector_pair is not None:
        observed_head_type, observed_rosa_cfg = _configured_head_contract(config)
        expected_rosa_cfg = {"profile": candidate_model_profile(arm)}
        if observed_head_type != pair["candidate_head_type"]:
            raise ProtocolError(
                f"{arm} config does not use the exact {pair_name.upper()} "
                "ROSA candidate head"
            )
        if (
            not isinstance(observed_rosa_cfg, Mapping)
            or dict(observed_rosa_cfg) != expected_rosa_cfg
        ):
            raise ProtocolError(
                f"{arm} config does not contain the exact ROSA profile selector"
            )
    if not isinstance(checkpoint, Mapping):
        raise ProtocolError("checkpoint is not a mapping")
    metadata = checkpoint.get("meta")
    if not isinstance(metadata, Mapping):
        raise ProtocolError("checkpoint metadata is missing")
    source_mapping = metadata.get("rosa_config")
    if not isinstance(source_mapping, Mapping):
        raise ProtocolError("checkpoint meta.rosa_config is required")
    if metadata.get("refinement_profile") != EXPECTED_CHECKPOINT_REFINEMENT_PROFILE:
        raise ProtocolError("checkpoint refinement profile is not balanced")
    if (
        metadata.get("rosa_contract_version")
        != EXPECTED_CHECKPOINT_ROSA_CONTRACT_VERSION
    ):
        raise ProtocolError("checkpoint ROSA contract version drifted")
    if (
        metadata.get("rfl_topology_contract")
        != EXPECTED_CHECKPOINT_RFL_TOPOLOGY_CONTRACT
    ):
        raise ProtocolError("checkpoint RFL topology contract drifted")

    effective_loader = getattr(run_tool, "_effective_rosa_config", None)
    state_loader = getattr(run_tool, "_checkpoint_state_dict", None)
    state_has_rosa = getattr(run_tool, "_state_dict_has_rosa", None)
    if not all(
        callable(value) for value in (effective_loader, state_loader, state_has_rosa)
    ):
        raise ProtocolError("run tool lacks strict checkpoint semantic helpers")
    try:
        target_effective = effective_loader(config)
        source_effective = type(target_effective).from_mapping(source_mapping)
        checkpoint_has_rosa = state_has_rosa(state_loader(checkpoint))
    except (TypeError, ValueError) as error:
        raise ProtocolError(
            f"{arm} checkpoint ROSA metadata is invalid: {error}"
        ) from error
    if checkpoint_has_rosa is not True:
        raise ProtocolError("checkpoint must contain the ROSA sidecar state")
    if (
        getattr(source_effective, "rfl_topology_contract", None)
        != EXPECTED_CHECKPOINT_RFL_TOPOLOGY_CONTRACT
        or getattr(target_effective, "rfl_topology_contract", None)
        != EXPECTED_CHECKPOINT_RFL_TOPOLOGY_CONTRACT
    ):
        raise ProtocolError("checkpoint/runtime RFL topology is not factorized v3")

    source_config = asdict(source_effective)
    target_config = asdict(target_effective)
    observed_projection = metadata.get("projection_contract")
    expected_projection = source_config["projection_contract"]
    if (
        observed_projection != expected_projection
        or target_config["projection_contract"] != expected_projection
    ):
        raise ProtocolError("checkpoint projection contract differs from runtime")

    observed_deltas = {
        name: {"checkpoint": source_config[name], "runtime": target_config[name]}
        for name in source_config
        if source_config[name] != target_config[name]
    }
    expected_deltas = FAST_CHECKPOINT_ALLOWED_DELTAS if arm in FAST_MODEL_ARMS else {}
    if arm not in CANDIDATE_ARMS or observed_deltas != expected_deltas:
        raise ProtocolError(
            f"{arm} checkpoint/runtime ROSA config delta is not allowed: "
            f"{observed_deltas}"
        )
    return {
        "detector_pair": pair_name,
        "candidate_head_type": pair["candidate_head_type"],
        "validated_against_arm": arm,
        "source_refinement_profile": metadata["refinement_profile"],
        "rosa_contract_version": metadata["rosa_contract_version"],
        "rfl_topology_contract": metadata["rfl_topology_contract"],
        "projection_contract": observed_projection,
        "checkpoint_has_rosa_state": True,
        "source_effective_config_sha256": canonical_sha256(source_config),
        "runtime_effective_config_sha256": canonical_sha256(target_config),
        "compared_effective_fields": sorted(source_config),
        "allowed_runtime_delta": expected_deltas,
        "observed_runtime_delta": observed_deltas,
        "runtime_model_profile": candidate_model_profile(arm),
        "runtime_execution_mode": candidate_execution_mode(arm),
        "all_other_effective_fields_exact": True,
        "validator": "full effective ROSA config comparison",
    }


def _build_models(verifier, configs, state_dict, arms, device):
    from mmengine.utils import import_modules_from_strings
    from mmdet3d.registry import MODELS
    from mmdet3d.utils import register_all_modules
    from rosa.integrations.sparse4d import (
        ROSANativeBaselineHead,
        ROSAOfficialResidualNativeBaselineHead,
        ROSAOfficialResidualRefinementHead,
        ROSARefinementHead,
    )

    native_arms = [arm for arm in configs if arm in NATIVE_ARMS]
    if len(native_arms) != 1:
        raise ProtocolError("configs must contain exactly one registered native arm")
    native_arm = native_arms[0]
    detector_pair = _detector_pair_from_native_arm(native_arm)
    pair = detector_pair_contract(detector_pair)
    expected_native_head = (
        ROSANativeBaselineHead
        if detector_pair == DETECTOR_PAIR_C0
        else ROSAOfficialResidualNativeBaselineHead
    )
    expected_candidate_head = (
        ROSARefinementHead
        if detector_pair == DETECTOR_PAIR_C0
        else ROSAOfficialResidualRefinementHead
    )

    register_all_modules(init_default_scope=True)
    for config in configs.values():
        import_modules_from_strings(**config.custom_imports)
    models, contracts, load_contracts = {}, {}, {}
    for arm in arms:
        if arm != native_arm and arm not in CANDIDATE_ARMS:
            raise ProtocolError(f"unregistered model arm: {arm}")
        config = configs[arm]
        verifier._seed_everything(SEED)
        model = MODELS.build(copy.deepcopy(config.model))
        if arm == native_arm:
            if type(model.head) is not expected_native_head:
                raise ProtocolError(
                    f"native arm did not build the exact {detector_pair.upper()} head"
                )
            model_keys = set(model.state_dict())
            missing = sorted(model_keys - set(state_dict))
            filtered = _checkpoint_copy(state_dict)
            for name in tuple(filtered):
                if name not in model_keys:
                    del filtered[name]
            if missing:
                raise ProtocolError(f"checkpoint misses native keys: {missing[:5]}")
            result = model.load_state_dict(filtered, strict=True)
            excluded = sorted(set(state_dict) - model_keys)
            _validate_native_checkpoint_exclusions(excluded)
            load_contracts[arm] = {
                "detector_pair": detector_pair,
                "head_type": pair["native_head_type"],
                "strict_native_subset": True,
                "excluded_sidecar_keys": excluded,
                "excluded_count": len(excluded),
                "missing_keys": list(result.missing_keys),
                "unexpected_keys": list(result.unexpected_keys),
            }
        else:
            if type(model.head) is not expected_candidate_head:
                raise ProtocolError(
                    f"{arm} did not build exact {pair['candidate_head_type']}"
                )
            result = model.load_state_dict(_checkpoint_copy(state_dict), strict=True)
            profile = candidate_model_profile(arm)
            expected_topk = 50 if arm in FAST_MODEL_ARMS else 100
            expected_steps = 1 if arm in FAST_MODEL_ARMS else 2
            if (
                model.head.rosa_cfg.refine_topk != expected_topk
                or model.head.rosa_cfg.num_refinement_steps != expected_steps
            ):
                raise ProtocolError(f"{profile} compute-budget contract drifted")
            load_contracts[arm] = {
                "detector_pair": detector_pair,
                "head_type": pair["candidate_head_type"],
                "strict_full_checkpoint": True,
                "missing_keys": list(result.missing_keys),
                "unexpected_keys": list(result.unexpected_keys),
            }
        backbone = getattr(model, "img_backbone", None)
        if int(getattr(backbone, "depth", -1)) != 18:
            raise ProtocolError(f"{arm} is not the exact ResNet-18 detector")
        contracts[arm] = verifier._model_contract(model)
        model.to(device=device).eval()
        models[arm] = model
    if native_arm in models:
        native_state = models[native_arm].state_dict()
        for arm in arms:
            if arm == native_arm:
                continue
            candidate_state = models[arm].state_dict()
            candidate_native_state = _project_shared_native_state(
                native_state, candidate_state
            )
            verifier.require_same_fingerprint(
                f"{arm} shared native checkpoint state",
                native_state,
                candidate_native_state,
            )
    return models, contracts, load_contracts


def _project_shared_native_state(native_state, candidate_state):
    """Project candidate tensors without changing the state container ABI."""

    missing = [name for name in native_state if name not in candidate_state]
    if missing:
        raise ProtocolError(f"candidate misses native state key: {missing[0]}")
    projected = copy.copy(native_state)
    if type(projected) is not type(native_state):
        raise ProtocolError("native state copy changed the mapping container type")
    for name in native_state:
        projected[name] = candidate_state[name]
    if tuple(projected) != tuple(native_state):
        raise ProtocolError("candidate native-state projection changed key order")
    return projected


def _require_current_default_stream(torch, device) -> dict[str, Any]:
    current = torch.cuda.current_stream(device)
    default = torch.cuda.default_stream(device)
    current_id = int(current.cuda_stream)
    default_id = int(default.cuda_stream)
    if current_id != default_id:
        raise ProtocolError("current CUDA stream is not the default stream")
    return {
        "policy": "current CUDA stream must equal default stream",
        "current_stream": current_id,
        "default_stream": default_id,
    }


def _require_owned_cpu_tensor(torch, value, label: str):
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.is_quantized
        or value.device.type != "cpu"
        or not value.is_contiguous()
        or value.requires_grad
        or value.grad_fn is not None
        or getattr(value, "_base", None) is not None
    ):
        raise ProtocolError(f"{label} is not an owned detached CPU tensor")
    return value


def _capture_detached_cpu_parity_payload(verifier, torch, output):
    """Copy audited fields without retaining CUDA/CUDAGraph output storage."""
    if not isinstance(output, Sequence) or isinstance(output, (str, bytes)):
        raise ProtocolError("parity payload requires a prediction sequence")
    payload = []
    field_names = (
        COMPILED_BOUNDED_BOX_FIELD,
        *COMPILED_BOUNDED_SCORE_FIELDS,
        *COMPILED_BOUNDED_EXACT_ASSOCIATION_FIELDS,
        COMPILED_BOUNDED_QUERY_FIELD,
    )
    for sample_index, sample in enumerate(output):
        fields = {}
        for name in field_names:
            try:
                source = verifier._prediction_tensor(sample, name)
            except Exception as error:
                raise ProtocolError(
                    f"parity payload sample {sample_index} lacks {name}"
                ) from error
            if not isinstance(source, torch.Tensor) or source.layout != torch.strided:
                raise ProtocolError(f"parity payload field {name} is not dense")
            owned = _require_owned_cpu_tensor(
                torch,
                source.detach().to(device="cpu", copy=True).contiguous(),
                f"parity payload field {name}",
            )
            if (
                source.device.type == "cpu"
                and source.numel()
                and source.untyped_storage().data_ptr()
                == owned.untyped_storage().data_ptr()
            ):
                raise ProtocolError(
                    f"parity payload field {name} aliases decoder output storage"
                )
            fields[name] = owned
        payload.append(fields)
    return tuple(payload)


def _preflight_outputs(
    verifier, torch, models, source_batch, device, *, detached_payloads=None
):
    records = {}
    source_before = verifier.raw_fingerprint(source_batch)
    with torch.inference_mode():
        for arm, model in models.items():
            verifier._reset_runtime_state(model)
            stream = _require_current_default_stream(torch, device)
            output = model.test_step(
                verifier._clone_pinned_cpu_ingress_batch(source_batch)
            )
            torch.cuda.synchronize(device)
            record = verifier._validate_detection_output(output)
            record["stream"] = stream
            records[arm] = record
            if detached_payloads is not None:
                detached_payloads[arm] = _capture_detached_cpu_parity_payload(
                    verifier, torch, output
                )
            del output
    if verifier.raw_fingerprint(source_batch) != source_before:
        raise ProtocolError("output preflight mutated the captured source batch")
    return records


def _candidate_preflight_by_batch(
    verifier,
    torch,
    model,
    sources,
    device,
    candidate_arm: str,
    *,
    capture_detached_payload: bool,
):
    if candidate_arm not in CANDIDATE_ARMS:
        raise ProtocolError("candidate preflight received an unknown arm")
    if type(capture_detached_payload) is not bool:
        raise TypeError("capture_detached_payload must be bool")
    records = {}
    payloads = {}
    for batch_index, source in enumerate(sources):
        key = str(batch_index)
        models = {candidate_arm: model}
        batch_payloads = {} if capture_detached_payload else None
        batch_records = _preflight_outputs(
            verifier,
            torch,
            models,
            source,
            device,
            detached_payloads=batch_payloads,
        )
        if batch_payloads is not None:
            payloads[key] = batch_payloads[candidate_arm]
        records[key] = batch_records[candidate_arm]
    return records, payloads


def _cuda_allocator_snapshot(torch, device) -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def _stabilize_cuda_allocator(torch, device) -> dict[str, int]:
    torch.cuda.synchronize(device)
    gc.collect()
    clear_cublas = getattr(
        getattr(torch, "_C", None), "_cuda_clearCublasWorkspaces", None
    )
    if not callable(clear_cublas):
        raise ProtocolError("Torch does not expose the documented cuBLAS cleanup API")
    clear_cublas()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return _cuda_allocator_snapshot(torch, device)


def _require_allocator_floor(
    expected: Mapping[str, int], observed: Mapping[str, int], *, label: str
) -> None:
    if dict(observed) != dict(expected):
        raise ProtocolError(
            f"{label} did not return to the isolated allocator floor: "
            f"expected={dict(expected)}, observed={dict(observed)}"
        )


def _reset_compiler_cache(torch) -> None:
    dynamo = getattr(torch, "_dynamo", None)
    reset = getattr(dynamo, "reset", None)
    if not callable(reset):
        raise ProtocolError("Torch does not expose torch._dynamo.reset")
    reset()


def _release_candidate_compiled_callable(model, candidate_arm: str) -> dict[str, Any]:
    """Detach a disposable compiled callable before global compiler cleanup."""

    if candidate_arm in BOUNDED_PARITY_COMPILED_ARMS:
        state_attribute = "_compiled_fast_anchor_impl"
    elif candidate_arm in RAW_EXACT_COMPILED_ARMS:
        state_attribute = "_compiled_call_impl"
    else:
        raise ProtocolError(
            f"{candidate_arm} has no audited compiled callable to release"
        )
    head = getattr(model, "head", None)
    refiner = getattr(head, "rosa_refiner", None)
    if refiner is None or not hasattr(refiner, state_attribute):
        raise ProtocolError(
            f"{candidate_arm} is missing compiled state {state_attribute}"
        )
    compiled_call = getattr(refiner, state_attribute)
    observed_compiled = compiled_call is not None
    setattr(refiner, state_attribute, None)
    if getattr(refiner, state_attribute) is not None:
        raise ProtocolError(f"{candidate_arm} compiled callable could not be detached")
    del compiled_call
    return {
        "arm": candidate_arm,
        "attribute": f"head.rosa_refiner.{state_attribute}",
        "observed_compiled": observed_compiled,
        "detached": observed_compiled,
        "passed": True,
    }


class _InductorCompileTaskSnapshot(NamedTuple):
    cache_ids: frozenset[int]
    sys_entries: frozenset[tuple[str, int]]


def _inductor_module_name(module, *, expected_name: str | None = None) -> str:
    name = getattr(module, "__name__", None)
    if (
        not isinstance(module, ModuleType)
        or not isinstance(name, str)
        or not name.startswith(INDUCTOR_COMPILE_TASK_MODULE_PREFIX)
        or (expected_name is not None and name != expected_name)
    ):
        raise ProtocolError("Torch Inductor compile-task module registry is malformed")
    return name


def _inductor_code_cache_registries():
    codecache_module = sys.modules.get("torch._inductor.codecache")
    if codecache_module is None:
        return None
    if not isinstance(codecache_module, ModuleType):
        raise ProtocolError("Torch Inductor code-cache module registry is malformed")
    py_code_cache = getattr(codecache_module, "PyCodeCache", None)
    tracked_modules = getattr(py_code_cache, "modules", None)
    tracked_no_attr = getattr(py_code_cache, "modules_no_attr", None)
    if not isinstance(tracked_modules, list) or not isinstance(tracked_no_attr, dict):
        raise ProtocolError("Torch does not expose the expected Inductor code cache")
    return tracked_modules, tracked_no_attr


def _inductor_compile_task_module_snapshot() -> _InductorCompileTaskSnapshot:
    registries = _inductor_code_cache_registries()
    if registries is None:
        cache_ids = set()
    else:
        tracked_modules, tracked_no_attr = registries
        cache_ids = {id(module) for module in tracked_modules}
        for module in (*tracked_modules, *tracked_no_attr.values()):
            _inductor_module_name(module)
        if any(id(module) not in cache_ids for module in tracked_no_attr.values()):
            raise ProtocolError("Torch Inductor code-cache registries disagree")

    sys_entries = set()
    for name, module in tuple(sys.modules.items()):
        if name.startswith(INDUCTOR_COMPILE_TASK_MODULE_PREFIX):
            _inductor_module_name(module, expected_name=name)
            sys_entries.add((name, id(module)))
    return _InductorCompileTaskSnapshot(
        cache_ids=frozenset(cache_ids),
        sys_entries=frozenset(sys_entries),
    )


def _release_new_inductor_compile_modules(
    baseline: _InductorCompileTaskSnapshot,
    observed: _InductorCompileTaskSnapshot,
) -> dict[str, Any]:
    if not isinstance(baseline, _InductorCompileTaskSnapshot) or not isinstance(
        observed, _InductorCompileTaskSnapshot
    ):
        raise TypeError("Inductor cleanup requires validated snapshots")
    if not baseline.cache_ids.issubset(observed.cache_ids):
        raise ProtocolError("Torch Inductor removed a pre-existing compile-task module")
    if _inductor_compile_task_module_snapshot() != observed:
        raise ProtocolError(
            "Torch Inductor compile-task modules changed during isolated cleanup"
        )
    created_ids = observed.cache_ids - baseline.cache_ids
    created_sys_entries = observed.sys_entries - baseline.sys_entries
    if not created_ids and not created_sys_entries:
        if observed != baseline:
            raise ProtocolError(
                "Torch Inductor removed a pre-existing sys.modules entry"
            )
        return {
            "prefix": INDUCTOR_COMPILE_TASK_MODULE_PREFIX,
            "baseline_count": len(baseline.cache_ids),
            "created_count": 0,
            "removed_module_names": [],
            "tracked_list_references_removed": 0,
            "tracked_dict_entries_removed": 0,
            "sys_module_entries_removed": 0,
            "baseline_sys_module_entries_restored": 0,
            "passed": True,
        }

    registries = _inductor_code_cache_registries()
    if created_ids and registries is None:
        raise ProtocolError("Torch does not expose the expected Inductor code cache")
    tracked_modules, tracked_no_attr = registries or ([], {})

    baseline_by_id = {}
    created_by_id = {}
    for module in tracked_modules:
        name = _inductor_module_name(module)
        if id(module) in baseline.cache_ids:
            baseline_by_id[id(module)] = module
        elif id(module) in created_ids:
            created_by_id[id(module)] = module
        else:
            raise ProtocolError("Torch Inductor code cache contains an unknown module")
    if set(created_by_id) != set(created_ids):
        raise ProtocolError("Torch Inductor created-module registry is incomplete")

    removed_list_references = sum(
        id(module) in created_ids for module in tracked_modules
    )
    kept_modules = [
        module for module in tracked_modules if id(module) not in created_ids
    ]
    tracked_modules[:] = kept_modules

    created_paths = [
        path for path, module in tracked_no_attr.items() if id(module) in created_ids
    ]
    for path in created_paths:
        del tracked_no_attr[path]

    created_names = {_inductor_module_name(module) for module in created_by_id.values()}
    removed_sys_entries = 0
    for name, module_id in sorted(created_sys_entries):
        module = sys.modules.get(name)
        if module is None or id(module) != module_id:
            raise ProtocolError(
                "Torch Inductor sys.modules entry changed before release"
            )
        removed = sys.modules.pop(name)
        created_names.add(name)
        removed_sys_entries += 1
        del removed

    restored_sys_entries = 0
    for name, module_id in sorted(baseline.sys_entries):
        current = sys.modules.get(name)
        if current is not None and id(current) == module_id:
            continue
        if current is not None:
            raise ProtocolError(
                "Torch Inductor baseline sys.modules entry was replaced"
            )
        previous = baseline_by_id.get(module_id)
        if previous is None or _inductor_module_name(previous) != name:
            raise ProtocolError("Torch Inductor baseline module cannot be restored")
        sys.modules[name] = previous
        restored_sys_entries += 1

    if _inductor_compile_task_module_snapshot() != baseline:
        raise ProtocolError("Torch Inductor compile-task module release was incomplete")
    return {
        "prefix": INDUCTOR_COMPILE_TASK_MODULE_PREFIX,
        "baseline_count": len(baseline.cache_ids),
        "created_count": len(created_ids),
        "removed_module_names": sorted(created_names),
        "tracked_list_references_removed": removed_list_references,
        "tracked_dict_entries_removed": len(created_paths),
        "sys_module_entries_removed": removed_sys_entries,
        "baseline_sys_module_entries_restored": restored_sys_entries,
        "passed": True,
    }


def _attest_candidate_compile_state(
    model,
    candidate_arm: str,
    *,
    expected_compiled: bool,
    phase: str,
) -> dict[str, Any]:
    if type(expected_compiled) is not bool:
        raise TypeError("expected_compiled must be bool")
    if not isinstance(phase, str) or not phase:
        raise TypeError("phase must be a nonempty string")
    model_profile = candidate_model_profile(candidate_arm)
    execution_mode = candidate_execution_mode(candidate_arm)
    if candidate_arm in EAGER_ARMS and expected_compiled:
        raise ProtocolError(f"{candidate_arm} forbids compiled execution")
    head = getattr(model, "head", None)
    refiner = getattr(head, "rosa_refiner", None)
    if refiner is None:
        raise ProtocolError(f"{candidate_arm} is missing its ROSA refiner")
    state_attribute = (
        "_compiled_fast_anchor_impl"
        if candidate_arm in BOUNDED_PARITY_COMPILED_ARMS
        else "_compiled_call_impl"
    )
    missing = object()
    compiled_call = getattr(refiner, state_attribute, missing)
    if compiled_call is missing:
        raise ProtocolError(
            f"{candidate_arm} refiner does not expose its compile-state attribute"
        )
    observed_compiled = compiled_call is not None
    if observed_compiled is not expected_compiled:
        expected = "compiled" if expected_compiled else "uncompiled"
        observed = "compiled" if observed_compiled else "uncompiled"
        raise ProtocolError(
            f"{candidate_arm} must be {expected} during {phase}, observed {observed}"
        )
    return {
        "arm": candidate_arm,
        "model_profile": model_profile,
        "execution_mode": execution_mode,
        "phase": phase,
        "attribute": f"head.rosa_refiner.{state_attribute}",
        "expected_compiled": expected_compiled,
        "observed_compiled": observed_compiled,
        "passed": True,
    }


def _measure_single_model_memory(
    verifier, torch, model, source_batch, device, arm: str
) -> dict[str, Any]:
    stream_before = _require_current_default_stream(torch, device)
    records = [verifier._memory_probe(model, source_batch, device) for _ in range(3)]
    required_fields = (
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "peak_allocated_increment_bytes",
        "peak_reserved_increment_bytes",
    )
    for record in records:
        if not isinstance(record, Mapping) or any(
            type(record.get(name)) is not int or record[name] < 0
            for name in required_fields
        ):
            raise ProtocolError(f"{arm} memory probe returned an invalid record")
    stream_after = _require_current_default_stream(torch, device)
    return {
        "peak_allocated_bytes": max(
            record["peak_allocated_bytes"] for record in records
        ),
        "peak_reserved_bytes": max(record["peak_reserved_bytes"] for record in records),
        "peak_allocated_increment_bytes": max(
            record["peak_allocated_increment_bytes"] for record in records
        ),
        "peak_reserved_increment_bytes": max(
            record["peak_reserved_increment_bytes"] for record in records
        ),
        "probe_records": records,
        "aggregation": "maximum of three isolated single-model probes",
        "resident_benchmark_models": [arm],
        "peer_model_resident": False,
        "stream_before": stream_before,
        "stream_after": stream_after,
    }


def _measure_isolated_memory(
    verifier,
    torch,
    run_tool,
    configs,
    state_dict,
    sources,
    device,
    candidate_arm: str,
    *,
    native_arm: str = NATIVE_ARM,
):
    _detector_pair_from_native_arm(native_arm)
    if native_arm not in configs:
        raise ProtocolError("selected native arm is missing from configs")
    arms = (native_arm, candidate_arm)
    allocator_floor = _stabilize_cuda_allocator(torch, device)
    memory: dict[str, Any] = {
        "contract": {
            "one_benchmark_model_resident_per_probe": True,
            "peer_model_resident": False,
            "allocator_floor": allocator_floor,
            "probe_count_per_arm": 3,
            "between_arm_cleanup": [
                {
                    "phase": "model_release",
                    "steps": ["models.clear", "gc.collect"],
                },
                {
                    "phase": "compiled_arm_only",
                    "steps": [
                        "detach disposable compiled callable",
                        "torch._dynamo.reset",
                        "release newly created torch._inductor compile-task modules",
                        "gc.collect",
                    ],
                },
                {
                    "phase": "allocator_stabilization",
                    "steps": [
                        "torch.cuda.synchronize",
                        "gc.collect",
                        "torch._C._cuda_clearCublasWorkspaces",
                        "torch.cuda.empty_cache",
                        "torch.cuda.synchronize",
                    ],
                },
            ],
        }
    }
    contracts: dict[str, Any] = {}
    load_contracts: dict[str, Any] = {}

    for arm in arms:
        before_build = _stabilize_cuda_allocator(torch, device)
        _require_allocator_floor(
            allocator_floor, before_build, label=f"before {arm} memory build"
        )
        models: dict[str, Any] = {}
        compiler_cache_created = False
        compiler_module_baseline = None
        compiler_cleanup_attestation = None
        compiled_callable_release_attestation = None
        compile_state_attestations: list[dict[str, Any]] = []
        try:
            models, built_contracts, built_load_contracts = _build_models(
                verifier, configs, state_dict, (arm,), device
            )
            contracts[arm] = built_contracts[arm]
            load_contracts[arm] = built_load_contracts[arm]
            compiled = False
            compile_invocation_attestation = None
            if arm in CANDIDATE_ARMS:
                compile_state_attestations.append(
                    _attest_candidate_compile_state(
                        models[arm],
                        arm,
                        expected_compiled=False,
                        phase="after isolated memory model build",
                    )
                )
            if arm in COMPILED_ARMS:
                compiler_module_baseline = _inductor_compile_task_module_snapshot()
                compiler_cache_created = True
                compile_invocation_attestation = _compile_candidate_refiner(
                    run_tool,
                    models[arm],
                    configs[arm],
                    arm,
                )
                compiled = bool(compile_invocation_attestation["compiled"])
            if arm in CANDIDATE_ARMS:
                compile_state_attestations.append(
                    _attest_candidate_compile_state(
                        models[arm],
                        arm,
                        expected_compiled=arm in COMPILED_ARMS,
                        phase="before isolated memory output preflight",
                    )
                )
            preflight = {
                str(batch_index): _preflight_outputs(
                    verifier, torch, {arm: models[arm]}, source, device
                )[arm]
                for batch_index, source in enumerate(sources)
            }
            if arm in CANDIDATE_ARMS:
                compile_state_attestations.append(
                    _attest_candidate_compile_state(
                        models[arm],
                        arm,
                        expected_compiled=arm in COMPILED_ARMS,
                        phase="after isolated memory output preflight",
                    )
                )
            record = _measure_single_model_memory(
                verifier, torch, models[arm], sources[0], device, arm
            )
            if arm in CANDIDATE_ARMS:
                compile_state_attestations.append(
                    _attest_candidate_compile_state(
                        models[arm],
                        arm,
                        expected_compiled=arm in COMPILED_ARMS,
                        phase="after isolated memory probes",
                    )
                )
            record.update(
                {
                    "allocator_before_build": before_build,
                    "compiled": compiled,
                    "compile_invocation_attestation": (compile_invocation_attestation),
                    "execution_mode": (
                        candidate_execution_mode(arm)
                        if arm in CANDIDATE_ARMS
                        else "native"
                    ),
                    "compile_state_attestations": compile_state_attestations,
                    "preflight_by_batch": preflight,
                }
            )
            memory[arm] = record
        finally:
            compiler_module_observed = (
                _inductor_compile_task_module_snapshot()
                if compiler_cache_created
                else None
            )
            if compiler_cache_created:
                compiled_callable_release_attestation = (
                    _release_candidate_compiled_callable(models[arm], arm)
                )
            models.clear()
            # torch.compile keeps the original bound method in its returned
            # callable.  Detach that private edge before releasing the model;
            # then collect before resetting global Dynamo/CUDA-graph caches.
            gc.collect()
            if compiler_cache_created:
                if compiler_module_baseline is None or compiler_module_observed is None:
                    raise ProtocolError("compiled module cleanup snapshot is missing")
                _reset_compiler_cache(torch)
                compiler_cleanup_attestation = _release_new_inductor_compile_modules(
                    compiler_module_baseline,
                    compiler_module_observed,
                )
                gc.collect()
            after_cleanup = _stabilize_cuda_allocator(torch, device)
        _require_allocator_floor(
            allocator_floor, after_cleanup, label=f"after {arm} memory cleanup"
        )
        memory[arm]["compiler_cache_cleanup_attestation"] = compiler_cleanup_attestation
        memory[arm][
            "compiled_callable_release_attestation"
        ] = compiled_callable_release_attestation
        memory[arm]["allocator_after_cleanup"] = after_cleanup

    return memory, contracts, load_contracts


def _compiler_counter_snapshot(torch) -> dict[str, Any]:
    dynamo = getattr(torch, "_dynamo", None)
    utilities = getattr(dynamo, "utils", None)
    counters = getattr(utilities, "counters", None)
    if not isinstance(counters, Mapping):
        raise ProtocolError("Torch does not expose torch._dynamo.utils.counters")
    values: dict[str, dict[str, int]] = {}
    for category, records in sorted(counters.items(), key=lambda item: str(item[0])):
        if not isinstance(category, str) or not isinstance(records, Mapping):
            raise ProtocolError("Torch compiler counters have an unsupported shape")
        normalized: dict[str, int] = {}
        for name, value in sorted(records.items(), key=lambda item: str(item[0])):
            if not isinstance(name, str) or type(value) is not int or value < 0:
                raise ProtocolError(
                    "Torch compiler counter is not a nonnegative integer"
                )
            normalized[name] = value
        if normalized:
            values[category] = normalized
    return {
        "source": "torch._dynamo.utils.counters",
        "values": values,
        "sha256": canonical_sha256(values),
    }


def _require_compiler_counters_unchanged(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    if before.get("values") != after.get("values"):
        raise ProtocolError(
            "Torch compiler/recompile counters changed during recorded timing"
        )
    return {
        "window": "entire recorded warmup and measured timing protocol",
        "compilation_or_recompile_observed": False,
        "before": dict(before),
        "after": dict(after),
    }


def _require_all_batch_compiled_parity(
    eager_by_batch: Mapping[str, Any],
    compiled_by_batch: Mapping[str, Any],
    *,
    batch_count: int,
) -> dict[str, Any]:
    expected_keys = {str(index) for index in range(batch_count)}
    if set(eager_by_batch) != expected_keys or set(compiled_by_batch) != expected_keys:
        raise ProtocolError("eager/compiled parity did not cover every captured batch")
    parity = {}
    for batch_index in range(batch_count):
        key = str(batch_index)
        eager_record = eager_by_batch[key]
        compiled_record = compiled_by_batch[key]
        if not isinstance(eager_record, Mapping) or not isinstance(
            compiled_record, Mapping
        ):
            raise ProtocolError("eager/compiled parity record is not a mapping")
        if "raw_fingerprint" not in eager_record or "raw_fingerprint" not in (
            compiled_record
        ):
            raise ProtocolError("eager/compiled parity record lacks raw fingerprint")
        if eager_record["raw_fingerprint"] != compiled_record["raw_fingerprint"]:
            raise ProtocolError(
                "compiled fast arm is not raw-exact with eager inference "
                f"for captured batch {batch_index}"
            )
        parity[key] = {
            "raw_exact": True,
            "eager": eager_record,
            "compiled": compiled_record,
        }
    return parity


def _error_distribution_record(torch, values) -> dict[str, Any]:
    if values.ndim != 1 or not bool(torch.isfinite(values).all()):
        raise ProtocolError("parity errors must be a finite vector")
    count = int(values.numel())
    if not count:
        return {"count": 0, "max": 0.0, "mean": 0.0, "p99_nearest_rank": 0.0}
    ordered = torch.sort(values).values
    p99_index = max(0, (99 * count + 99) // 100 - 1)
    return {
        "count": count,
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "p99_nearest_rank": float(ordered[p99_index].item()),
    }


def _paired_sample_field(
    torch,
    reference_sample: Mapping[str, Any],
    candidate_sample: Mapping[str, Any],
    field: str,
    *,
    sample_index: int,
    comparison_label: str,
):
    reference = _require_owned_cpu_tensor(
        torch,
        reference_sample.get(field),
        f"{comparison_label} reference {field}[{sample_index}]",
    )
    candidate = _require_owned_cpu_tensor(
        torch,
        candidate_sample.get(field),
        f"{comparison_label} candidate {field}[{sample_index}]",
    )
    if (
        reference.dtype != candidate.dtype
        or reference.shape != candidate.shape
        or reference.stride() != candidate.stride()
    ):
        raise ProtocolError(
            f"{comparison_label} changed {field} tensor schema in sample "
            f"{sample_index}"
        )
    return reference, candidate


def _query_aligned_sample(
    torch,
    reference_sample: Mapping[str, Any],
    candidate_sample: Mapping[str, Any],
    *,
    sample_index: int,
    comparison_label: str,
    expected_count: int,
):
    if not isinstance(reference_sample, Mapping) or not isinstance(
        candidate_sample, Mapping
    ):
        raise ProtocolError(f"{comparison_label} sample is not a mapping")
    reference_query, candidate_query = _paired_sample_field(
        torch,
        reference_sample,
        candidate_sample,
        COMPILED_BOUNDED_QUERY_FIELD,
        sample_index=sample_index,
        comparison_label=comparison_label,
    )
    count = int(reference_query.numel())
    if any(
        value.ndim != 1
        or value.dtype == torch.bool
        or value.is_floating_point()
        or value.is_complex()
        for value in (reference_query, candidate_query)
    ):
        raise ProtocolError(f"{comparison_label} query_indices must be integral")
    if (
        type(expected_count) is not int
        or expected_count < 0
        or expected_count > COMPILED_BOUNDED_OUTPUT_LIMIT
    ):
        raise ProtocolError(
            f"{comparison_label} decoder count must be in [0, "
            f"{COMPILED_BOUNDED_OUTPUT_LIMIT}]"
        )
    if count != expected_count:
        raise ProtocolError(
            f"{comparison_label} query count does not match the decoder record "
            f"in sample {sample_index}"
        )
    if any(
        bool((value < 0).any()) or bool((value >= COMPILED_BOUNDED_QUERY_COUNT).any())
        for value in (reference_query, candidate_query)
    ):
        raise ProtocolError(
            f"{comparison_label} query_indices must be in [0, "
            f"{COMPILED_BOUNDED_QUERY_COUNT - 1}]"
        )
    if any(
        int(torch.unique(value).numel()) != count
        for value in (reference_query, candidate_query)
    ):
        raise ProtocolError(
            f"{comparison_label} query_indices must be unique in sample {sample_index}"
        )
    reference_order = torch.argsort(reference_query)
    candidate_order = torch.argsort(candidate_query)
    query_set = reference_query[reference_order]
    if not torch.equal(query_set, candidate_query[candidate_order]):
        raise ProtocolError(
            f"{comparison_label} changed the query set in sample {sample_index}"
        )

    aligned = {}
    raw = {}
    for field in (
        *COMPILED_BOUNDED_EXACT_ASSOCIATION_FIELDS,
        *COMPILED_BOUNDED_SCORE_FIELDS,
        COMPILED_BOUNDED_BOX_FIELD,
    ):
        reference, candidate = _paired_sample_field(
            torch,
            reference_sample,
            candidate_sample,
            field,
            sample_index=sample_index,
            comparison_label=comparison_label,
        )
        if reference.ndim == 0 or int(reference.shape[0]) != count:
            raise ProtocolError(
                f"{comparison_label} {field} has no value for every query"
            )
        raw[field] = (reference, candidate)
        aligned[field] = (
            reference[reference_order],
            candidate[candidate_order],
        )

    exact = {}
    for field in COMPILED_BOUNDED_EXACT_ASSOCIATION_FIELDS:
        reference, candidate = aligned[field]
        if any(
            value.ndim != 1
            or value.dtype == torch.bool
            or value.is_floating_point()
            or value.is_complex()
            for value in (reference, candidate)
        ):
            raise ProtocolError(f"{comparison_label} {field} must be integral")
        if not torch.equal(reference, candidate):
            raise ProtocolError(
                f"{comparison_label} changed query-aligned exact field {field} "
                f"in sample {sample_index}"
            )
        exact[field] = reference

    score_errors = {}
    for field in COMPILED_BOUNDED_SCORE_FIELDS:
        reference, candidate = aligned[field]
        if reference.ndim != 1 or not reference.is_floating_point():
            raise ProtocolError(f"{comparison_label} {field} must be a float vector")
        if not bool(torch.isfinite(reference).all()) or not bool(
            torch.isfinite(candidate).all()
        ):
            raise ProtocolError(f"{comparison_label} {field} contains NaN/Inf")
        if field == "scores_3d" and any(
            value.numel() > 1 and not bool((value[:-1] >= value[1:]).all())
            for value in raw[field]
        ):
            raise ProtocolError(
                f"{comparison_label} scores_3d must be non-increasing in raw "
                f"decoder order for sample {sample_index}"
            )
        score_errors[field] = (candidate.double() - reference.double()).abs()

    reference_rank = reference_order[:, None] - reference_order[None, :]
    candidate_rank = candidate_order[:, None] - candidate_order[None, :]
    inversions = torch.triu(reference_rank * candidate_rank < 0, diagonal=1)
    reference_scores = aligned["scores_3d"][0].double()
    inversion_gaps = (reference_scores[:, None] - reference_scores[None, :]).abs()[
        inversions
    ]
    if bool((inversion_gaps > COMPILED_BOUNDED_NUMERICAL_TIE_MAX_GAP).any()):
        raise ProtocolError(
            f"{comparison_label} rank inversion crossed a non-tied score gap "
            f"in sample {sample_index}"
        )

    reference_box, candidate_box = aligned[COMPILED_BOUNDED_BOX_FIELD]
    expected_box_shape = (count, COMPILED_BOUNDED_BOX_DIM)
    if (
        tuple(reference_box.shape) != expected_box_shape
        or not reference_box.is_floating_point()
    ):
        raise ProtocolError(
            f"{comparison_label} decoded boxes must have shape {expected_box_shape}"
        )
    if not bool(torch.isfinite(reference_box).all()) or not bool(
        torch.isfinite(candidate_box).all()
    ):
        raise ProtocolError(f"{comparison_label} decoded boxes contain NaN/Inf")
    return {
        "query_set": query_set,
        "rank_displacement": (candidate_order - reference_order).abs().double(),
        "rank_inversion_gaps": inversion_gaps,
        "exact": exact,
        "score_errors": score_errors,
        "box_errors": (candidate_box.double() - reference_box.double()).abs(),
    }


def _paired_payload_values(torch, eager_payload, compiled_payload, field: str):
    eager_values, compiled_values = [], []
    for sample_index, (eager_sample, compiled_sample) in enumerate(
        zip(eager_payload, compiled_payload)
    ):
        if not isinstance(eager_sample, Mapping) or not isinstance(
            compiled_sample, Mapping
        ):
            raise ProtocolError("bounded parity sample is not a mapping")
        eager = _require_owned_cpu_tensor(
            torch, eager_sample.get(field), f"eager {field}[{sample_index}]"
        )
        compiled = _require_owned_cpu_tensor(
            torch, compiled_sample.get(field), f"compiled {field}[{sample_index}]"
        )
        if (
            eager.dtype != compiled.dtype
            or eager.shape != compiled.shape
            or eager.stride() != compiled.stride()
        ):
            raise ProtocolError(f"compiled bounded arm changed {field} schema")
        eager_values.append(eager)
        compiled_values.append(compiled)
    return eager_values, compiled_values


def _raw_element_equal_mask(torch, eager, compiled):
    element_bytes = eager.element_size()
    eager_bytes = eager.contiguous().view(torch.uint8).reshape(-1, element_bytes)
    compiled_bytes = compiled.contiguous().view(torch.uint8).reshape(-1, element_bytes)
    return eager_bytes.eq(compiled_bytes).all(dim=1)


def _integer_distribution_record(values: Sequence[int]) -> dict[str, Any]:
    count = len(values)
    if not count:
        return {"count": 0, "max": 0, "mean": 0.0, "p99_nearest_rank": 0}
    ordered = sorted(int(value) for value in values)
    p99_index = max(0, (99 * count + 99) // 100 - 1)
    return {
        "count": count,
        "max": ordered[-1],
        "mean": float(sum(ordered) / count),
        "p99_nearest_rank": ordered[p99_index],
    }


def _floating_ulp_distances(torch, eager, compiled) -> list[int] | None:
    formats = {
        torch.float16: (torch.int16, 16),
        torch.bfloat16: (torch.int16, 16),
        torch.float32: (torch.int32, 32),
        torch.float64: (torch.int64, 64),
    }
    spec = formats.get(eager.dtype)
    if spec is None:
        return None
    integer_dtype, width = spec
    eager_words = eager.contiguous().view(integer_dtype).reshape(-1).tolist()
    compiled_words = compiled.contiguous().view(integer_dtype).reshape(-1).tolist()
    mask = (1 << width) - 1
    sign = 1 << (width - 1)

    def ordered_key(word: int) -> int:
        bits = int(word) & mask
        return ((~bits) & mask) if bits & sign else bits | sign

    return [
        abs(ordered_key(left) - ordered_key(right))
        for left, right in zip(eager_words, compiled_words)
    ]


def _floating_field_diagnostic(torch, eager_values, compiled_values) -> dict[str, Any]:
    absolute_parts = []
    relative_parts = []
    ulp_values = []
    ulp_applicable = True
    raw_mismatch_count = 0
    numeric_mismatch_count = 0
    first_mismatch = None
    samples = []
    for sample_index, (eager, compiled) in enumerate(
        zip(eager_values, compiled_values)
    ):
        raw_mismatch = ~_raw_element_equal_mask(torch, eager, compiled)
        sample_raw_mismatches = int(raw_mismatch.sum().item())
        eager_double = eager.double().reshape(-1)
        compiled_double = compiled.double().reshape(-1)
        absolute = (compiled_double - eager_double).abs()
        scale = torch.maximum(eager_double.abs(), compiled_double.abs()).clamp_min(
            COMPILED_BOUNDED_BOX_RELATIVE_FLOOR
        )
        numeric_mismatches = int(absolute.ne(0).sum().item())
        sample_ulps = _floating_ulp_distances(torch, eager, compiled)
        if sample_ulps is None:
            ulp_applicable = False
        else:
            ulp_values.extend(sample_ulps)
        if first_mismatch is None and sample_raw_mismatches:
            flat_index = int(raw_mismatch.nonzero(as_tuple=False)[0].item())
            first_mismatch = {
                "sample_index": sample_index,
                "flat_index": flat_index,
                "eager": float(eager_double[flat_index].item()),
                "compiled": float(compiled_double[flat_index].item()),
                "absolute_error": float(absolute[flat_index].item()),
                "ulp_error": (
                    None if sample_ulps is None else int(sample_ulps[flat_index])
                ),
            }
        raw_mismatch_count += sample_raw_mismatches
        numeric_mismatch_count += numeric_mismatches
        absolute_parts.append(absolute)
        relative_parts.append(absolute / scale)
        samples.append(
            {
                "sample_index": sample_index,
                "element_count": int(eager.numel()),
                "raw_mismatch_count": sample_raw_mismatches,
                "numeric_mismatch_count": numeric_mismatches,
            }
        )
    absolute = torch.cat(absolute_parts) if absolute_parts else torch.empty(0)
    relative = torch.cat(relative_parts) if relative_parts else torch.empty(0)
    return {
        "raw_exact": raw_mismatch_count == 0,
        "raw_mismatch_count": raw_mismatch_count,
        "numeric_mismatch_count": numeric_mismatch_count,
        "first_mismatch": first_mismatch,
        "absolute_error": _error_distribution_record(torch, absolute),
        "relative_error": _error_distribution_record(torch, relative),
        "ulp_error": (
            _integer_distribution_record(ulp_values)
            if ulp_applicable
            else {"applicable": False, "reason": "unsupported floating dtype"}
        ),
        "samples": samples,
    }


def _integer_field_diagnostic(torch, eager_values, compiled_values) -> dict[str, Any]:
    mismatch_count = 0
    first_mismatch = None
    samples = []
    for sample_index, (eager, compiled) in enumerate(
        zip(eager_values, compiled_values)
    ):
        mismatch = ~_raw_element_equal_mask(torch, eager, compiled)
        sample_mismatches = int(mismatch.sum().item())
        if first_mismatch is None and sample_mismatches:
            flat_index = int(mismatch.nonzero(as_tuple=False)[0].item())
            first_mismatch = {
                "sample_index": sample_index,
                "flat_index": flat_index,
                "eager": int(eager.reshape(-1)[flat_index].item()),
                "compiled": int(compiled.reshape(-1)[flat_index].item()),
            }
        mismatch_count += sample_mismatches
        samples.append(
            {
                "sample_index": sample_index,
                "element_count": int(eager.numel()),
                "mismatch_count": sample_mismatches,
            }
        )
    return {
        "raw_exact": mismatch_count == 0,
        "mismatch_count": mismatch_count,
        "first_mismatch": first_mismatch,
        "samples": samples,
    }


def _query_rank_diagnostic(eager_values, compiled_values) -> dict[str, Any]:
    samples = []
    first_mismatch = None
    for sample_index, (eager, compiled) in enumerate(
        zip(eager_values, compiled_values)
    ):
        eager_flat = eager.reshape(-1)
        compiled_flat = compiled.reshape(-1)
        mismatch = eager_flat.ne(compiled_flat)
        mismatch_count = int(mismatch.sum().item())
        sample_first = None
        if mismatch_count:
            position = int(mismatch.nonzero(as_tuple=False)[0].item())
            sample_first = {
                "position": position,
                "eager_query": int(eager_flat[position].item()),
                "compiled_query": int(compiled_flat[position].item()),
            }
            if first_mismatch is None:
                first_mismatch = {"sample_index": sample_index, **sample_first}
        prefix_length = min(
            COMPILED_PARITY_DIAGNOSTIC_TOPK,
            int(eager_flat.numel()),
            int(compiled_flat.numel()),
        )
        eager_prefix = [int(value) for value in eager_flat[:prefix_length].tolist()]
        compiled_prefix = [
            int(value) for value in compiled_flat[:prefix_length].tolist()
        ]
        eager_set = set(eager_prefix)
        compiled_set = set(compiled_prefix)
        samples.append(
            {
                "sample_index": sample_index,
                "rank_length": int(eager_flat.numel()),
                "full_order_exact": mismatch_count == 0,
                "position_mismatch_count": mismatch_count,
                "first_mismatch": sample_first,
                "topk50": {
                    "requested_prefix_length": COMPILED_PARITY_DIAGNOSTIC_TOPK,
                    "observed_prefix_length": prefix_length,
                    "order_exact": eager_prefix == compiled_prefix,
                    "set_exact": eager_set == compiled_set,
                    "eager_only_queries": sorted(eager_set - compiled_set),
                    "compiled_only_queries": sorted(compiled_set - eager_set),
                },
            }
        )
    return {
        "full_order_exact": all(item["full_order_exact"] for item in samples),
        "position_mismatch_count": sum(
            item["position_mismatch_count"] for item in samples
        ),
        "first_mismatch": first_mismatch,
        "topk50_order_exact": all(item["topk50"]["order_exact"] for item in samples),
        "topk50_set_exact": all(item["topk50"]["set_exact"] for item in samples),
        "samples": samples,
    }


def _parity_violation(code, message, *, batch_index=None, field=None):
    record = {"code": code, "message": message}
    if batch_index is not None:
        record["batch_index"] = batch_index
    if field is not None:
        record["field"] = field
    return record


def _build_all_batch_compiled_bounded_parity_diagnostic(
    verifier,
    torch,
    eager_by_batch,
    compiled_by_batch,
    eager_payload_by_batch,
    compiled_payload_by_batch,
    *,
    batch_count,
):
    expected_keys = {str(index) for index in range(batch_count)}
    named = {
        "eager_records": eager_by_batch,
        "compiled_records": compiled_by_batch,
        "eager_payloads": eager_payload_by_batch,
        "compiled_payloads": compiled_payload_by_batch,
    }
    observed = {name: set(mapping) for name, mapping in named.items()}
    coverage_exact = all(keys == expected_keys for keys in observed.values())
    violations = []
    if not coverage_exact:
        violations.append(
            _parity_violation(
                "batch_coverage",
                "bounded eager/compiled parity did not cover every captured batch",
            )
        )
    batches = {}
    for batch_index in range(batch_count):
        key = str(batch_index)
        missing = [name for name, mapping in named.items() if key not in mapping]
        if missing:
            batches[key] = {"available": False, "missing_mappings": missing}
            continue
        eager_record = eager_by_batch[key]
        compiled_record = compiled_by_batch[key]
        if not isinstance(eager_record, Mapping) or not isinstance(
            compiled_record, Mapping
        ):
            message = "bounded eager/compiled record is not a mapping"
            violations.append(
                _parity_violation("record_schema", message, batch_index=batch_index)
            )
            batches[key] = {"available": False, "error": message}
            continue
        batch = {"available": True, "schema_and_stream": {}, "fields": {}}
        for field in (
            "sample_count",
            "detections_per_sample",
            "finite_schema",
            "stream",
        ):
            present = field in eager_record and field in compiled_record
            exact = present and eager_record[field] == compiled_record[field]
            batch["schema_and_stream"][field] = {
                "present_in_both": present,
                "exact": exact,
            }
            if not present:
                message = f"bounded eager/compiled record lacks schema field {field}"
                violations.append(
                    _parity_violation(
                        "missing_schema_field",
                        message,
                        batch_index=batch_index,
                        field=field,
                    )
                )
            elif not exact:
                message = f"compiled bounded arm changed output schema/stream field {field} for captured batch {batch_index}"
                violations.append(
                    _parity_violation(
                        "schema_or_stream_mismatch",
                        message,
                        batch_index=batch_index,
                        field=field,
                    )
                )
        eager_payload = eager_payload_by_batch[key]
        compiled_payload = compiled_payload_by_batch[key]
        payload_shape_exact = bool(
            isinstance(eager_payload, Sequence)
            and not isinstance(eager_payload, (str, bytes))
            and isinstance(compiled_payload, Sequence)
            and not isinstance(compiled_payload, (str, bytes))
            and len(eager_payload) == len(compiled_payload)
            and len(eager_payload) == eager_record.get("sample_count")
        )
        batch["payload_shape_exact"] = payload_shape_exact
        if not payload_shape_exact:
            message = (
                f"bounded parity payload shape drifted for captured batch {batch_index}"
            )
            violations.append(
                _parity_violation("payload_shape", message, batch_index=batch_index)
            )
            for field in (
                *COMPILED_BOUNDED_RAW_EXACT_FIELDS,
                *COMPILED_BOUNDED_BOX_FIELDS,
            ):
                batch["fields"][field] = {"available": False, "error": message}
            batches[key] = batch
            continue
        for field in COMPILED_BOUNDED_RAW_EXACT_FIELDS:
            try:
                eager_values, compiled_values = _paired_payload_values(
                    torch, eager_payload, compiled_payload, field
                )
            except ProtocolError as error:
                message = str(error)
                batch["fields"][field] = {"available": False, "error": message}
                violations.append(
                    _parity_violation(
                        "field_schema", message, batch_index=batch_index, field=field
                    )
                )
                continue
            is_score = field in COMPILED_BOUNDED_RAW_EXACT_SCORE_FIELDS
            record = {
                "available": True,
                "kind": "raw_exact_float" if is_score else "raw_exact_integer",
            }
            if is_score and all(
                value.is_floating_point() and bool(torch.isfinite(value).all())
                for value in (*eager_values, *compiled_values)
            ):
                record.update(
                    _floating_field_diagnostic(torch, eager_values, compiled_values)
                )
            elif not is_score and all(
                not value.is_floating_point() and not value.is_complex()
                for value in eager_values
            ):
                record.update(
                    _integer_field_diagnostic(torch, eager_values, compiled_values)
                )
            else:
                message = (
                    f"score field {field} must be floating point"
                    if is_score
                    else f"integer field {field} changed type"
                )
                record.update({"strict_passed": False, "error": message})
                violations.append(
                    _parity_violation(
                        "field_type", message, batch_index=batch_index, field=field
                    )
                )
                batch["fields"][field] = record
                continue
            eager_fingerprint = verifier.raw_fingerprint(eager_values)
            compiled_fingerprint = verifier.raw_fingerprint(compiled_values)
            raw_exact = eager_fingerprint == compiled_fingerprint
            record.update(
                {
                    "raw_exact": raw_exact,
                    "strict_passed": raw_exact,
                    "eager_fingerprint": eager_fingerprint,
                    "compiled_fingerprint": compiled_fingerprint,
                }
            )
            if not raw_exact:
                message = f"compiled bounded arm changed raw-exact field {field} for captured batch {batch_index}"
                violations.append(
                    _parity_violation(
                        "raw_exact_field", message, batch_index=batch_index, field=field
                    )
                )
            if field == "query_indices":
                record["ranking"] = _query_rank_diagnostic(
                    eager_values, compiled_values
                )
            batch["fields"][field] = record
        field = COMPILED_BOUNDED_BOX_FIELDS[0]
        try:
            eager_values, compiled_values = _paired_payload_values(
                torch, eager_payload, compiled_payload, field
            )
            valid = all(
                value.is_floating_point() and bool(torch.isfinite(value).all())
                for value in (*eager_values, *compiled_values)
            )
        except ProtocolError as error:
            valid = False
            message = str(error)
        if not valid:
            message = locals().get("message", "bounded box payload contains NaN or Inf")
            box = {"available": False, "strict_passed": False, "error": message}
            violations.append(
                _parity_violation(
                    "box_schema", message, batch_index=batch_index, field=field
                )
            )
        else:
            box = {"available": True, "kind": "bounded_float"}
            box.update(_floating_field_diagnostic(torch, eager_values, compiled_values))
            tolerance_passed = bool(
                box["absolute_error"]["max"] <= COMPILED_BOUNDED_BOX_MAX_ABS
                and box["relative_error"]["max"] <= COMPILED_BOUNDED_BOX_MAX_REL
            )
            box["strict_passed"] = tolerance_passed
            box["tolerance_passed"] = tolerance_passed
            if not tolerance_passed:
                message = (
                    f"compiled bounded arm exceeded box tolerance for {field} in captured batch {batch_index}: "
                    f"max_abs={box['absolute_error']['max']}, max_rel={box['relative_error']['max']}"
                )
                violations.append(
                    _parity_violation(
                        "box_tolerance", message, batch_index=batch_index, field=field
                    )
                )
        batch["fields"][field] = box
        batches[key] = batch
    return {
        "schema": COMPILED_PARITY_DIAGNOSTIC_SCHEMA,
        "mode": "raw_exact_ranking_and_scores_bounded_box_floats_v1",
        "batch_count": batch_count,
        "coverage": {
            "exact": coverage_exact,
            "expected_keys": sorted(expected_keys),
            "observed_keys": {name: sorted(keys) for name, keys in observed.items()},
        },
        "diagnostic_contract": {
            "all_fields_collected_before_enforcement": True,
            "raw_bit_distribution_is_diagnostic_only": True,
            "float_fields": [
                *COMPILED_BOUNDED_RAW_EXACT_SCORE_FIELDS,
                *COMPILED_BOUNDED_BOX_FIELDS,
            ],
            "integer_fields": list(COMPILED_BOUNDED_RAW_EXACT_INTEGER_FIELDS),
            "query_rank_prefix": COMPILED_PARITY_DIAGNOSTIC_TOPK,
            "p99_definition": "nearest-rank ceil(0.99 * count)",
        },
        "passed": not violations,
        "violation_count": len(violations),
        "violations": violations,
        "batches": batches,
    }


def _enforce_all_batch_compiled_bounded_parity(
    verifier,
    torch,
    eager_by_batch: Mapping[str, Any],
    compiled_by_batch: Mapping[str, Any],
    eager_payload_by_batch: Mapping[str, Any],
    compiled_payload_by_batch: Mapping[str, Any],
    *,
    batch_count: int,
    comparison_label: str = "eager_reference_vs_compiled",
) -> dict[str, Any]:
    """Match rows by query ID, then apply the immutable absolute-error gate."""

    if not isinstance(comparison_label, str) or not comparison_label:
        raise TypeError("comparison_label must be a non-empty string")

    expected_keys = {str(index) for index in range(batch_count)}
    mappings = (
        eager_by_batch,
        compiled_by_batch,
        eager_payload_by_batch,
        compiled_payload_by_batch,
    )
    if any(set(mapping) != expected_keys for mapping in mappings):
        raise ProtocolError(
            f"{comparison_label} parity did not cover every captured batch"
        )

    parity = {}
    for batch_index in range(batch_count):
        key = str(batch_index)
        eager_record = eager_by_batch[key]
        compiled_record = compiled_by_batch[key]
        if not isinstance(eager_record, Mapping) or not isinstance(
            compiled_record, Mapping
        ):
            raise ProtocolError(f"{comparison_label} record is not a mapping")
        for field in (
            "sample_count",
            "detections_per_sample",
            "finite_schema",
            "stream",
        ):
            if field not in eager_record or field not in compiled_record:
                raise ProtocolError(
                    f"{comparison_label} record lacks schema field {field}"
                )
            if eager_record[field] != compiled_record[field]:
                raise ProtocolError(
                    f"{comparison_label} changed output schema/stream field {field} "
                    f"for captured batch {batch_index}"
                )

        eager_payload = eager_payload_by_batch[key]
        compiled_payload = compiled_payload_by_batch[key]
        if (
            not isinstance(eager_payload, Sequence)
            or isinstance(eager_payload, (str, bytes))
            or not isinstance(compiled_payload, Sequence)
            or isinstance(compiled_payload, (str, bytes))
            or len(eager_payload) != len(compiled_payload)
            or len(eager_payload) != eager_record["sample_count"]
        ):
            raise ProtocolError(
                f"{comparison_label} payload shape drifted for captured batch "
                f"{batch_index}"
            )
        detections_per_sample = eager_record["detections_per_sample"]
        if (
            not isinstance(detections_per_sample, Sequence)
            or isinstance(detections_per_sample, (str, bytes))
            or len(detections_per_sample) != len(eager_payload)
        ):
            raise ProtocolError(
                f"{comparison_label} decoder counts do not cover captured batch "
                f"{batch_index}"
            )

        query_sets = []
        rank_displacements = []
        rank_inversion_gaps = []
        exact_reference = {
            field: [] for field in COMPILED_BOUNDED_EXACT_ASSOCIATION_FIELDS
        }
        score_errors = {field: [] for field in COMPILED_BOUNDED_SCORE_FIELDS}
        box_errors = []
        for sample_index, (reference_sample, candidate_sample) in enumerate(
            zip(eager_payload, compiled_payload)
        ):
            aligned = _query_aligned_sample(
                torch,
                reference_sample,
                candidate_sample,
                sample_index=sample_index,
                comparison_label=comparison_label,
                expected_count=detections_per_sample[sample_index],
            )
            query_sets.append(aligned["query_set"])
            rank_displacements.append(aligned["rank_displacement"])
            rank_inversion_gaps.append(aligned["rank_inversion_gaps"])
            for field in COMPILED_BOUNDED_EXACT_ASSOCIATION_FIELDS:
                exact_reference[field].append(aligned["exact"][field])
            for field in COMPILED_BOUNDED_SCORE_FIELDS:
                score_errors[field].append(aligned["score_errors"][field])
            absolute = aligned["box_errors"]
            box_errors.append(absolute.reshape(-1))

        exact_field_records = {
            field: {
                "query_aligned_raw_exact": True,
                "fingerprint": verifier.raw_fingerprint(values),
            }
            for field, values in exact_reference.items()
        }
        bounded_field_records = {}
        for field, errors in score_errors.items():
            record = _error_distribution_record(torch, torch.cat(errors))
            if record["max"] > COMPILED_BOUNDED_SCORE_MAX_ABS[field]:
                raise ProtocolError(
                    f"{comparison_label} exceeded {field} absolute tolerance in "
                    f"captured batch {batch_index}: max_abs={record['max']}, "
                    f"limit={COMPILED_BOUNDED_SCORE_MAX_ABS[field]}"
                )
            bounded_field_records[field] = {"absolute_error": record}

        aggregate_box_record = _error_distribution_record(torch, torch.cat(box_errors))
        if aggregate_box_record["max"] > COMPILED_BOUNDED_BOX_MAX_ABS:
            raise ProtocolError(
                f"{comparison_label} exceeded decoded-box absolute tolerance in "
                f"captured batch {batch_index}: max_abs={aggregate_box_record['max']}, "
                f"limit={COMPILED_BOUNDED_BOX_MAX_ABS}"
            )
        bounded_field_records[COMPILED_BOUNDED_BOX_FIELD] = {
            "absolute_error": aggregate_box_record,
        }
        rank_record = _error_distribution_record(torch, torch.cat(rank_displacements))
        rank_record["changed_query_count"] = int(
            torch.count_nonzero(torch.cat(rank_displacements)).item()
        )
        inversion_gaps = torch.cat(rank_inversion_gaps)
        rank_record["inversion_count"] = int(inversion_gaps.numel())
        rank_record["maximum_inversion_score_gap"] = (
            float(inversion_gaps.max().item()) if inversion_gaps.numel() else 0.0
        )
        parity[key] = {
            "mode": "query_aligned_fixed_absolute_tolerance_v3",
            "comparison": comparison_label,
            "passed": True,
            "schema_and_stream_exact": True,
            "query_alignment": {
                "key": COMPILED_BOUNDED_QUERY_FIELD,
                "query_set_exact": True,
                "query_ids_unique_per_sample": True,
                "fingerprint": verifier.raw_fingerprint(query_sets),
            },
            "rank_displacement": rank_record,
            "exact_fields_after_alignment": exact_field_records,
            "bounded_float_fields": bounded_field_records,
            "absolute_thresholds": {
                **COMPILED_BOUNDED_SCORE_MAX_ABS,
                COMPILED_BOUNDED_BOX_FIELD: COMPILED_BOUNDED_BOX_MAX_ABS,
                "p99_definition": "nearest-rank ceil(0.99 * count)",
            },
            "owned_detached_cpu_payloads": True,
            "promotion_status": "PENDING_OFFICIAL_FULL_METRIC_PARITY",
        }
    return parity


def _require_all_batch_compiled_bounded_parity(
    verifier,
    torch,
    eager_by_batch,
    compiled_by_batch,
    eager_payload_by_batch,
    compiled_payload_by_batch,
    *,
    batch_count,
    comparison_label="eager_reference_vs_compiled",
):
    return _enforce_all_batch_compiled_bounded_parity(
        verifier,
        torch,
        eager_by_batch,
        compiled_by_batch,
        eager_payload_by_batch,
        compiled_payload_by_batch,
        batch_count=batch_count,
        comparison_label=comparison_label,
    )


def _record_and_require_all_batch_compiled_bounded_parity(
    diagnostic_path,
    verifier,
    torch,
    eager_by_batch,
    compiled_by_batch,
    eager_payload_by_batch,
    compiled_payload_by_batch,
    *,
    batch_count,
    comparison_label="eager_reference_vs_compiled",
):
    diagnostic = _build_all_batch_compiled_bounded_parity_diagnostic(
        verifier,
        torch,
        eager_by_batch,
        compiled_by_batch,
        eager_payload_by_batch,
        compiled_payload_by_batch,
        batch_count=batch_count,
    )
    diagnostic["comparison_label"] = comparison_label
    diagnostic["raw_bit_diagnostic_passed"] = diagnostic.pop("passed")
    diagnostic["raw_bit_distribution_is_diagnostic_only"] = True
    diagnostic["aligned_gate"] = {"status": "PENDING", "passed": False}
    diagnostic["passed"] = False
    _atomic_json(diagnostic_path, diagnostic)
    try:
        result = _enforce_all_batch_compiled_bounded_parity(
            verifier,
            torch,
            eager_by_batch,
            compiled_by_batch,
            eager_payload_by_batch,
            compiled_payload_by_batch,
            batch_count=batch_count,
            comparison_label=comparison_label,
        )
    except ProtocolError as error:
        diagnostic["aligned_gate"] = {
            "status": "FAILED",
            "passed": False,
            "error": str(error),
        }
        _atomic_json(diagnostic_path, diagnostic)
        raise
    diagnostic["aligned_gate"] = {
        "status": "PASSED",
        "passed": True,
        "result": result,
    }
    diagnostic["passed"] = True
    _atomic_json(diagnostic_path, diagnostic)
    return result


def _run_pair_protocol(
    verifier,
    torch,
    models,
    sources,
    manifests,
    device,
    candidate_arm,
    native_arm=NATIVE_ARM,
):
    detector_pair = _detector_pair_from_native_arm(native_arm)
    _require_current_default_stream(torch, device)
    compiler_before = (
        _compiler_counter_snapshot(torch) if candidate_arm in COMPILED_ARMS else None
    )
    result = verifier._run_scope(
        scope=verifier.C0_C32_FULL_DETECTOR_SCOPE,
        models={
            verifier.PARENT_ARM: models[native_arm],
            verifier.CANDIDATE_ARM: models[candidate_arm],
        },
        sources=sources,
        manifests=manifests,
        device=device,
    )
    compiler_contract = compiler_guard_contract(candidate_arm)
    if compiler_before is None:
        result["compiler_guard"] = compiler_contract
    else:
        compiler_after = _compiler_counter_snapshot(torch)
        result["compiler_guard"] = _require_compiler_counters_unchanged(
            compiler_before, compiler_after
        )
        result["compiler_guard"].update(compiler_contract)
    result["stream_after"] = _require_current_default_stream(torch, device)
    result["arm_aliases"] = {
        verifier.PARENT_ARM: native_arm,
        verifier.CANDIDATE_ARM: candidate_arm,
    }
    result["detector_pair"] = detector_pair
    result["candidate_model_profile"] = candidate_model_profile(candidate_arm)
    result["candidate_execution_mode"] = candidate_execution_mode(candidate_arm)
    return result


def _evaluate_near_native_gate(
    verifier,
    timings: Mapping[str, Any],
    memory: Mapping[str, Any],
    candidate_arm: str,
    native_arm: str = NATIVE_ARM,
) -> Mapping[str, Any]:
    detector_pair = _detector_pair_from_native_arm(native_arm)
    paired = timings.get("paired_bootstrap")
    if not isinstance(paired, Mapping):
        raise ProtocolError("paired timing result is missing its bootstrap evidence")
    logical_memory = {
        verifier.PARENT_ARM: memory[native_arm],
        verifier.CANDIDATE_ARM: memory[candidate_arm],
    }
    gate = verifier.evaluate_c0_c32_near_native_gate(paired, logical_memory)
    checks = gate.get("checks") if isinstance(gate, Mapping) else None
    expected_thresholds = {
        "median_ratio_ci95_upper_at_most_1_10": 1.10,
        "p95_ratio_at_most_1_10": 1.10,
        "peak_allocated_bytes_regression_at_most_2_percent": 1.02,
        "peak_reserved_bytes_regression_at_most_2_percent": 1.02,
    }
    if not isinstance(checks, Mapping) or any(
        not isinstance(checks.get(name), Mapping)
        or float(checks[name].get("threshold_max_inclusive", float("nan"))) != threshold
        for name, threshold in expected_thresholds.items()
    ):
        raise ProtocolError("runtime near-native gate differs from the frozen plan")
    passed = all(bool(check.get("passed")) for check in checks.values())
    if (
        gate.get("passed") is not passed
        or gate.get("speed_claim_eligible") is not passed
        or gate.get("decision")
        != ("NEAR_NATIVE_SUPPORTED" if passed else "NEAR_NATIVE_NOT_SUPPORTED")
    ):
        raise ProtocolError("runtime near-native decision is internally inconsistent")
    return {
        "gate": "rosa_vs_native_full_detector_near_native",
        "detector_pair": detector_pair,
        "parent_arm": native_arm,
        "candidate_arm": candidate_arm,
        "benchmark_scope": "full model.test_step",
        "passed": passed,
        "decision": (
            "NEAR_NATIVE_SUPPORTED" if passed else "NEAR_NATIVE_NOT_SUPPORTED"
        ),
        "speed_claim_eligible": passed,
        "checks": checks,
        "threshold_evaluator": "pinned C0/C32 full-detector implementation",
    }


def execute_protocol(args: argparse.Namespace, prepared_plan: Mapping[str, Any]):
    """Import and execute accelerator code only after repeated hard guards."""

    pair = detector_pair_contract(args.detector_pair)
    detector_pair = pair["detector_pair"]
    native_arm = pair["native_arm"]
    if (
        prepared_plan.get("detector_pair") != detector_pair
        or prepared_plan.get("detector_pair_contract", {}).get("native_arm")
        != native_arm
    ):
        raise ProtocolError("prepared detector-pair contract differs from CLI")
    validate_execution_gate(args)
    repeated = prepare_plan(args)
    if repeated != prepared_plan:
        raise ProtocolError("protocol inputs changed after dry-run preparation")
    if prepared_plan["repository"]["status"]:
        raise ProtocolError("armed benchmark requires a clean ROSA worktree")
    if prepared_plan["detector_host"]["git"]["status"]:
        raise ProtocolError("armed benchmark requires a clean detector host")
    detector_root = Path(prepared_plan["detector_host"]["git"]["root"])
    official_root = Path(prepared_plan["official_fisheye3dod"]["git"]["root"])
    os.environ["ROSA_DETECTOR_HOST_ROOT"] = str(detector_root)
    os.environ["SPARSE4D_ROOT"] = str(detector_root)
    os.environ["FISHEYE3DOD_OFFICIAL_ROOT"] = str(official_root)
    existing_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing_workspace not in (None, ":4096:8"):
        raise ProtocolError("CUBLAS_WORKSPACE_CONFIG conflicts with the protocol")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    selected_gpu_uuid = prepared_plan["gpu_selection"].get("physical_uuid")
    if not isinstance(selected_gpu_uuid, str):
        raise ProtocolError("armed plan does not contain one physical GPU UUID")
    _reject_preimported_armed_modules()
    lock_fd, lock_path = _acquire_physical_gpu_lock(selected_gpu_uuid)
    output_dir = Path(prepared_plan["output_dir"])
    output_owned = False
    try:
        preimport_gpu = _nvidia_smi_snapshot(selected_gpu_uuid)
        verifier = _load_armed_verifier()
        _attest_armed_schedules(verifier, prepared_plan)
        helper_constants = (
            verifier.DEFAULT_WARMUP_ITERATIONS_PER_ARM,
            verifier.DEFAULT_MEASURED_ITERATIONS_PER_ARM,
            verifier.CAPTURE_BATCHES,
            verifier.EXPECTED_BATCH_SIZE,
            verifier.BOOTSTRAP_SEED,
            verifier.BOOTSTRAP_SAMPLES,
        )
        planned_constants = (
            WARMUP_ITERATIONS_PER_ARM,
            MEASURED_ITERATIONS_PER_ARM,
            CAPTURE_BATCHES,
            EXPECTED_BATCH_SIZE,
            BOOTSTRAP_SEED,
            BOOTSTRAP_SAMPLES,
        )
        if helper_constants != planned_constants:
            raise ProtocolError("armed helper constants differ from the frozen plan")
        if (
            tuple(verifier.DAF_SUPPORTED_CUDA_CAPABILITIES)
            != DAF_SUPPORTED_CUDA_CAPABILITIES
        ):
            raise ProtocolError(
                "DAF capability contract differs from the pinned verifier"
            )
        torch = verifier.torch
        if not torch.cuda.is_available():
            raise ProtocolError("armed execution requires CUDA")
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        capability = tuple(torch.cuda.get_device_capability(device))
        if capability not in DAF_SUPPORTED_CUDA_CAPABILITIES:
            raise ProtocolError(
                f"DAF extension does not contain code for CUDA capability {capability}"
            )
        identity = _logical_device_identity(torch, device)
        if identity["gpu_uuid"] != selected_gpu_uuid:
            raise ProtocolError(
                "Torch logical device differs from the pre-locked physical GPU"
            )
        identity["capability"] = list(capability)
        stream = _require_current_default_stream(torch, device)
    except Exception:
        _release_physical_gpu_lock(lock_fd, lock_path)
        raise

    try:
        initial_gpu = _nvidia_smi_snapshot(identity["gpu_uuid"])
        output_dir.mkdir(parents=False, exist_ok=False)
        output_owned = True
        _fsync_directory(output_dir.parent)
        armed_plan = copy.deepcopy(prepared_plan)
        armed_plan.update(
            {
                "status": "ARMED",
                "accelerator_modules_imported": True,
                "cuda_queried": True,
                "output_created": True,
                "physical_gpu": identity,
                "initial_stream": stream,
                "exclusive_lock": str(lock_path),
            }
        )
        _atomic_json(output_dir / "PLAN.json", armed_plan)

        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        verifier._seed_everything(SEED)
        verifier._pin_import_roots(detector_root)
        verifier._pin_official_import_roots(official_root)
        arms = selected_arms(args.candidate_profile, detector_pair)
        candidate_arm = selected_candidate(args.candidate_profile)
        configs = _derive_configs(
            verifier,
            Path(prepared_plan["native_config"]["path"]),
            arms,
            detector_pair,
        )
        _validate_effective_dataset_configs(
            configs,
            Path(prepared_plan["dataset"]["annotation"]["path"]),
        )
        effective_configs = {
            arm: {
                "sha256": canonical_sha256(config.to_dict()),
                "model_sha256": canonical_sha256(config.to_dict()["model"]),
            }
            for arm, config in configs.items()
        }
        run_tool = verifier._load_run_tool()
        checkpoint_path = Path(args.checkpoint).resolve()
        checkpoint = run_tool._load_checkpoint_without_executing_code(checkpoint_path)
        if not isinstance(checkpoint, Mapping):
            raise ProtocolError("checkpoint is not a mapping")
        checkpoint_semantic_contract = _validate_checkpoint_semantics(
            run_tool,
            checkpoint,
            configs[candidate_arm],
            candidate_arm,
            detector_pair=detector_pair,
        )
        state_dict = run_tool._checkpoint_state_dict(checkpoint)
        if not isinstance(state_dict, Mapping):
            raise ProtocolError("checkpoint has no recognizable state_dict")
        checkpoint_schema = verifier._checkpoint_schema(state_dict)

        _register_runtime_modules(configs)
        prepared = SimpleNamespace(
            parent_config=configs[native_arm], expected_manifest=None
        )
        sources, manifests = verifier._capture_batches(prepared)
        _atomic_json(output_dir / "SAMPLE_MANIFEST.json", manifests)

        comparison_results = {}
        model_contracts = {}
        load_contracts = {}
        compiled = {candidate_arm: False}
        gpu_snapshots = [preimport_gpu, initial_gpu]
        loaded_preflight = _capture_loaded_provenance(
            verifier,
            detector_root=detector_root,
            official_root=official_root,
            expected_daf_sha256=args.expected_daf_extension_sha256,
        )
        _validate_loaded_provenance(loaded_preflight, prepared_plan)

        pair_dir = output_dir / candidate_arm
        pair_dir.mkdir(exist_ok=False)
        _fsync_directory(output_dir)
        gpu_snapshots.append(_nvidia_smi_snapshot(identity["gpu_uuid"]))
        memory, memory_model_contracts, memory_load_contracts = (
            _measure_isolated_memory(
                verifier,
                torch,
                run_tool,
                configs,
                state_dict,
                sources,
                device,
                candidate_arm,
                native_arm=native_arm,
            )
        )
        _atomic_json(pair_dir / "MEMORY.json", memory)
        gpu_snapshots.append(_nvidia_smi_snapshot(identity["gpu_uuid"]))

        pair_arms = (native_arm, candidate_arm)
        models, pair_model_contracts, pair_load_contracts = _build_models(
            verifier, configs, state_dict, pair_arms, device
        )
        model_contracts[candidate_arm] = pair_model_contracts
        load_contracts[candidate_arm] = pair_load_contracts
        timing_compile_state_attestations = [
            _attest_candidate_compile_state(
                models[candidate_arm],
                candidate_arm,
                expected_compiled=False,
                phase="after paired timing model build",
            )
        ]

        eager_candidate_preflight = None
        eager_candidate_payloads = None
        eager_repeat_preflight = None
        eager_repeat_payloads = None
        eager_repeat_parity = None
        eager_candidate_rng = None
        eager_candidate_state = None
        timing_compile_invocation_attestation = None
        if candidate_arm in COMPILED_ARMS:
            eager_candidate_rng = verifier._capture_rng()
            eager_candidate_state = verifier.raw_fingerprint(
                models[candidate_arm].state_dict()
            )
            (
                eager_candidate_preflight,
                eager_candidate_payloads,
            ) = _candidate_preflight_by_batch(
                verifier,
                torch,
                models[candidate_arm],
                sources,
                device,
                candidate_arm,
                capture_detached_payload=(
                    candidate_arm in BOUNDED_PARITY_COMPILED_ARMS
                ),
            )
            verifier.require_same_fingerprint(
                "eager fast preflight RNG",
                eager_candidate_rng,
                verifier._capture_rng(),
            )
            if (
                verifier.raw_fingerprint(models[candidate_arm].state_dict())
                != eager_candidate_state
            ):
                raise ProtocolError("eager fast preflight mutated model state")
            if candidate_arm in BOUNDED_PARITY_COMPILED_ARMS:
                verifier._restore_rng(eager_candidate_rng)
                (
                    eager_repeat_preflight,
                    eager_repeat_payloads,
                ) = _candidate_preflight_by_batch(
                    verifier,
                    torch,
                    models[candidate_arm],
                    sources,
                    device,
                    candidate_arm,
                    capture_detached_payload=True,
                )
                verifier.require_same_fingerprint(
                    "second eager reference RNG",
                    eager_candidate_rng,
                    verifier._capture_rng(),
                )
                if (
                    verifier.raw_fingerprint(models[candidate_arm].state_dict())
                    != eager_candidate_state
                ):
                    raise ProtocolError(
                        "second eager reference preflight mutated model state"
                    )
                eager_repeat_parity = _require_all_batch_compiled_bounded_parity(
                    verifier,
                    torch,
                    eager_candidate_preflight,
                    eager_repeat_preflight,
                    eager_candidate_payloads,
                    eager_repeat_payloads,
                    batch_count=len(sources),
                    comparison_label=("eager_reference_a_vs_eager_reference_b"),
                )
            timing_compile_state_attestations.append(
                _attest_candidate_compile_state(
                    models[candidate_arm],
                    candidate_arm,
                    expected_compiled=False,
                    phase="after eager reference preflight(s)",
                )
            )
            verifier._restore_rng(eager_candidate_rng)
            timing_compile_invocation_attestation = _compile_candidate_refiner(
                run_tool,
                models[candidate_arm],
                configs[candidate_arm],
                candidate_arm,
            )
            compiled[candidate_arm] = bool(
                timing_compile_invocation_attestation["compiled"]
            )
            timing_compile_state_attestations.append(
                _attest_candidate_compile_state(
                    models[candidate_arm],
                    candidate_arm,
                    expected_compiled=True,
                    phase="after fast compilation",
                )
            )

        _require_current_default_stream(torch, device)
        output_preflight = _preflight_outputs(
            verifier, torch, models, sources[0], device
        )
        compiled_parity = None
        compiled_parity_diagnostic = {
            "applicable": False,
            "reason": "candidate does not use bounded compiled parity",
        }
        if candidate_arm == FAST_EAGER_ARM:
            compiled_parity = {
                "applicable": False,
                "reason": "candidate is the uncompiled fast reference path",
                "raw_eager_reference_path": True,
            }
        if eager_candidate_preflight is not None:
            verifier.require_same_fingerprint(
                "compiled fast preflight RNG",
                eager_candidate_rng,
                verifier._capture_rng(),
            )
            if (
                verifier.raw_fingerprint(models[candidate_arm].state_dict())
                != eager_candidate_state
            ):
                raise ProtocolError("compiled fast preflight mutated model state")
        timing_compile_state_attestations.append(
            _attest_candidate_compile_state(
                models[candidate_arm],
                candidate_arm,
                expected_compiled=candidate_arm in COMPILED_ARMS,
                phase="after paired output preflight",
            )
        )

        compiler_warmup_by_batch = None
        compiler_warmup_payloads = None
        parity_payload_release = {
            "applicable": False,
            "reason": "candidate does not use bounded compiled parity payloads",
        }
        if candidate_arm == FAST_EAGER_ARM:
            compiler_warmup_by_batch = {
                "applicable": False,
                "reason": "torch.compile is forbidden for fast_eager",
            }
        if candidate_arm in COMPILED_ARMS:
            warmup_rng = verifier._capture_rng()
            warmup_state = verifier.raw_fingerprint(models[candidate_arm].state_dict())
            (
                compiler_warmup_by_batch,
                compiler_warmup_payloads,
            ) = _candidate_preflight_by_batch(
                verifier,
                torch,
                models[candidate_arm],
                sources,
                device,
                candidate_arm,
                capture_detached_payload=(
                    candidate_arm in BOUNDED_PARITY_COMPILED_ARMS
                ),
            )
            verifier.require_same_fingerprint(
                "all-batch compiler warmup RNG",
                warmup_rng,
                verifier._capture_rng(),
            )
            if (
                verifier.raw_fingerprint(models[candidate_arm].state_dict())
                != warmup_state
            ):
                raise ProtocolError("all-batch compiler warmup mutated model state")
            if candidate_arm in RAW_EXACT_COMPILED_ARMS:
                compiled_parity = _require_all_batch_compiled_parity(
                    eager_candidate_preflight,
                    compiler_warmup_by_batch,
                    batch_count=len(sources),
                )
            elif candidate_arm in BOUNDED_PARITY_COMPILED_ARMS:
                diagnostic_path = pair_dir / COMPILED_PARITY_DIAGNOSTIC_FILENAME
                compiled_vs_eager_a = (
                    _record_and_require_all_batch_compiled_bounded_parity(
                        diagnostic_path,
                        verifier,
                        torch,
                        eager_candidate_preflight,
                        compiler_warmup_by_batch,
                        eager_candidate_payloads,
                        compiler_warmup_payloads,
                        batch_count=len(sources),
                        comparison_label="eager_reference_a_vs_compiled",
                    )
                )
                compiled_parity_diagnostic = {
                    "applicable": True,
                    "path": str(diagnostic_path),
                    "sha256": file_sha256(diagnostic_path),
                    "written_atomically_before_strict_enforcement": True,
                    "raw_bit_distribution_is_diagnostic_only": True,
                }
                compiled_vs_eager_b = _require_all_batch_compiled_bounded_parity(
                    verifier,
                    torch,
                    eager_repeat_preflight,
                    compiler_warmup_by_batch,
                    eager_repeat_payloads,
                    compiler_warmup_payloads,
                    batch_count=len(sources),
                    comparison_label="eager_reference_b_vs_compiled",
                )
                compiled_parity = {
                    "mode": "query_aligned_fixed_absolute_tolerance_v3",
                    "passed": True,
                    "comparisons": {
                        "eager_reference_a_vs_eager_reference_b": (eager_repeat_parity),
                        "eager_reference_a_vs_compiled": compiled_vs_eager_a,
                        "eager_reference_b_vs_compiled": compiled_vs_eager_b,
                    },
                    "promotion_status": "PENDING_OFFICIAL_FULL_METRIC_PARITY",
                }
                for payload_mapping in (
                    eager_candidate_payloads,
                    eager_repeat_payloads,
                    compiler_warmup_payloads,
                ):
                    payload_mapping.clear()
                eager_candidate_payloads = None
                eager_repeat_payloads = None
                compiler_warmup_payloads = None
                gc.collect()
                parity_payload_release = {
                    "applicable": True,
                    "capture": ("detach().to(device='cpu', copy=True).contiguous()"),
                    "retained_output_containers": False,
                    "retained_cuda_or_cudagraph_storage": False,
                    "payload_mappings_cleared_before_recorded_timing": True,
                    "released_payload_sets": [
                        "eager_reference_a",
                        "eager_reference_b",
                        "compiled",
                    ],
                }
            else:
                raise ProtocolError("compiled arm has no parity contract")
            verifier.require_same_fingerprint(
                "post-parity compiler warmup RNG",
                warmup_rng,
                verifier._capture_rng(),
            )
            if (
                verifier.raw_fingerprint(models[candidate_arm].state_dict())
                != warmup_state
            ):
                raise ProtocolError("compiled parity processing mutated model state")
        timing_compile_state_attestations.append(
            _attest_candidate_compile_state(
                models[candidate_arm],
                candidate_arm,
                expected_compiled=candidate_arm in COMPILED_ARMS,
                phase="immediately before recorded timing",
            )
        )

        _atomic_json(pair_dir / "OUTPUT_PREFLIGHT.json", output_preflight)
        gpu_snapshots.append(_nvidia_smi_snapshot(identity["gpu_uuid"]))
        timings = _run_pair_protocol(
            verifier,
            torch,
            models,
            sources,
            manifests,
            device,
            candidate_arm,
            native_arm,
        )
        timing_compile_state_attestations.append(
            _attest_candidate_compile_state(
                models[candidate_arm],
                candidate_arm,
                expected_compiled=candidate_arm in COMPILED_ARMS,
                phase="immediately after recorded timing",
            )
        )
        warmup_records = timings.pop("warmup_records")
        measured_records = timings.pop("measured_records")
        near_native_gate = _evaluate_near_native_gate(
            verifier,
            timings,
            memory,
            candidate_arm,
            native_arm,
        )
        _write_jsonl(pair_dir / "WARMUP.jsonl", warmup_records)
        _write_jsonl(pair_dir / "TIMINGS.jsonl", measured_records)
        _atomic_json(pair_dir / "STATISTICS.json", timings)
        comparison_results[candidate_arm] = {
            "detector_pair": detector_pair,
            "parent_arm": native_arm,
            "candidate_arm": candidate_arm,
            "output_preflight": output_preflight,
            "compiled_eager_parity": compiled_parity,
            "compiled_parity_contract": compiled_parity_contract(candidate_arm),
            "compiled_parity_diagnostic": compiled_parity_diagnostic,
            "compiled_parity_calibration": prepared_plan["compiled_parity_calibration"],
            "parity_payload_release": parity_payload_release,
            "compiler_warmup_by_batch": compiler_warmup_by_batch,
            "compile_state_attestations": timing_compile_state_attestations,
            "compile_invocation_attestation": (timing_compile_invocation_attestation),
            "memory": memory,
            "statistics": timings,
            "near_native_gate": near_native_gate,
            "warmup_sha256": file_sha256(pair_dir / "WARMUP.jsonl"),
            "timings_sha256": file_sha256(pair_dir / "TIMINGS.jsonl"),
        }
        gpu_snapshots.append(_nvidia_smi_snapshot(identity["gpu_uuid"]))
        del models
        torch.cuda.synchronize(device)
        gc.collect()
        torch.cuda.empty_cache()
        _require_current_default_stream(torch, device)

        final_plan = prepare_plan(args, allow_existing_output=True)
        if final_plan != prepared_plan:
            raise ProtocolError("source/checkpoint/config/DAF changed during timing")
        loaded_provenance = _capture_loaded_provenance(
            verifier,
            detector_root=detector_root,
            official_root=official_root,
            expected_daf_sha256=args.expected_daf_extension_sha256,
        )
        _validate_loaded_provenance(loaded_provenance, final_plan)
        environment = verifier._environment(device, gpu_snapshots)
        evidence = {
            "schema": EVIDENCE_SCHEMA,
            "status": "SEALED_PENDING_COMPLETION",
            "protocol": final_plan["protocol"],
            "detector_pair": detector_pair,
            "detector_pair_contract": final_plan["detector_pair_contract"],
            "plan_sha256": file_sha256(output_dir / "PLAN.json"),
            "repository": final_plan["repository"],
            "rosa_source": final_plan["rosa_source"],
            "native_config": final_plan["native_config"],
            "dataset": final_plan["dataset"],
            "effective_configs": effective_configs,
            "checkpoint": final_plan["checkpoint"],
            "checkpoint_schema": checkpoint_schema,
            "checkpoint_semantic_contract": checkpoint_semantic_contract,
            "detector_host": final_plan["detector_host"],
            "official_fisheye3dod": final_plan["official_fisheye3dod"],
            "loaded_provenance": loaded_provenance,
            "loaded_provenance_preflight": loaded_preflight,
            "sample_manifest_sha256": file_sha256(output_dir / "SAMPLE_MANIFEST.json"),
            "model_contracts": model_contracts,
            "checkpoint_load_contracts": load_contracts,
            "memory_model_contracts": memory_model_contracts,
            "memory_checkpoint_load_contracts": memory_load_contracts,
            "compiled": compiled,
            "candidate_model_profile": candidate_model_profile(candidate_arm),
            "candidate_execution_mode": candidate_execution_mode(candidate_arm),
            "output_contract": {
                "finite_and_well_formed": True,
                "values_allowed_to_differ": True,
                "compiled_eager_parity": compiled_parity_contract(candidate_arm),
                "fast_eager_is_the_uncompiled_fast_reference_path": (
                    candidate_arm == FAST_EAGER_ARM
                ),
            },
            "promotion_contract": final_plan["promotion_contract"],
            "comparisons": comparison_results,
            "environment": environment,
            "physical_gpu": identity,
            "nvidia_smi": gpu_snapshots,
        }
        _atomic_json(output_dir / "EVIDENCE.json", evidence)
        _seal_directory(output_dir)
        completion = {
            "schema": COMPLETION_SCHEMA,
            "status": "COMPLETE",
            "evidence_sha256": file_sha256(output_dir / "EVIDENCE.json"),
            "seal_sha256": file_sha256(output_dir / "SHA256SUMS"),
            "completed_wall_time_ns": time.time_ns(),
        }
        _atomic_json(output_dir / "COMPLETE.json", completion)
        returned = copy.deepcopy(evidence)
        returned["status"] = "COMPLETE"
        returned["completion"] = completion
        return returned
    except Exception as error:
        if output_owned and not (output_dir / "COMPLETE.json").exists():
            _atomic_json(
                output_dir / "FAILURE.json",
                {
                    "schema": FAILURE_SCHEMA,
                    "status": "FAILED",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            _seal_directory(output_dir)
        raise
    finally:
        _release_physical_gpu_lock(lock_fd, lock_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_execution_gate(args)
        plan = prepare_plan(args)
        if not args.execute:
            print(json.dumps(plan, indent=2, ensure_ascii=False, sort_keys=True))
            return 0
        evidence = execute_protocol(args, plan)
        print(json.dumps(evidence, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    except ProtocolError as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
