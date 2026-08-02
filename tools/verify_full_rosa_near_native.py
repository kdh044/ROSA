#!/usr/bin/env python3
"""Disarmed C0/ROSA full-model CUDA benchmark protocol.

The default command is a standard-library-only dry run.  It hashes and prints
the immutable experiment plan but does not import Torch, NumPy, MMEngine,
MMDetection, ROSA, or a CUDA extension; inspect a GPU; create an output
directory; or acquire a runtime lock.

The separately armed path is intentionally difficult to invoke accidentally.
It requires two explicit attestations, revalidates every source/checkpoint/DAF
input, rejects a non-default current stream and concurrent GPU processes, then
measures the exact same captured batches in one isolated C0/fast or C0/balanced
paired alternating protocol.  Fast and balanced are deliberately separate
process invocations with separate output directories.
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
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
VERIFY_C0_C32_TOOL = ROOT / "tools/verify_c0_c32_near_native.py"

PROTOCOL = "full-rosa-c0-fast-balanced-near-native-v1"
PLAN_SCHEMA = "rosa.full_rosa_near_native_plan.v1"
EVIDENCE_SCHEMA = "rosa.full_rosa_near_native_evidence.v1"
IDLE_RESERVED_TOKEN = "I_HAVE_VERIFIED_THE_GPU_IS_IDLE_AND_RESERVED"
CURRENT_STREAM_TOKEN = "I_HAVE_VERIFIED_THE_CURRENT_STREAM_IS_DEFAULT"

NATIVE_ARM = "c0_native"
FAST_ARM = "rosa_fast"
BALANCED_ARM = "rosa_balanced"
WARMUP_ITERATIONS_PER_ARM = 60
MEASURED_ITERATIONS_PER_ARM = 300
CAPTURE_BATCHES = 2
EXPECTED_BATCH_SIZE = 4
SEED = 3407
BOOTSTRAP_SEED = 3407
BOOTSTRAP_SAMPLES = 20_000
EXPECTED_CHECKPOINT_REFINEMENT_PROFILE = "balanced"
EXPECTED_CHECKPOINT_ROSA_CONTRACT_VERSION = 2
FAST_CHECKPOINT_ALLOWED_DELTAS = {
    "refine_topk": {"checkpoint": 100, "runtime": 50},
    "num_refinement_steps": {"checkpoint": 2, "runtime": 1},
}

PHYSICAL_ROUND_ORDER = ("PCCP", "CPPC", "PCCP", "CPPC", "PCCP", "CPPC")

DEFAULT_NATIVE_CONFIG = (
    ROOT / "configs/rosa_r18_main126_v2_c0_cumulative_runtime_exact.py"
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
    parser.add_argument("--native-config", type=Path, default=DEFAULT_NATIVE_CONFIG)
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
        choices=("fast", "balanced"),
        default="fast",
        help="Run exactly one isolated C0-vs-ROSA profile protocol.",
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


def selected_candidate(profile: str) -> str:
    if profile == "fast":
        return FAST_ARM
    if profile == "balanced":
        return BALANCED_ARM
    raise ProtocolError("candidate profile must be fast or balanced")


def selected_arms(profile: str) -> tuple[str, str]:
    return NATIVE_ARM, selected_candidate(profile)


def build_alternating_schedule(
    *,
    iterations_per_arm: int,
    phase: str,
    candidate_arm: str,
    batch_count: int,
) -> tuple[dict[str, Any], ...]:
    if type(iterations_per_arm) is not int or iterations_per_arm <= 0:
        raise ProtocolError("iterations_per_arm must be a positive exact integer")
    if iterations_per_arm % 6:
        raise ProtocolError("iterations_per_arm must be divisible by six rounds")
    if type(batch_count) is not int or batch_count <= 0:
        raise ProtocolError("batch_count must be a positive exact integer")
    if not isinstance(phase, str) or not phase:
        raise ProtocolError("phase must be a nonempty string")
    if candidate_arm not in (FAST_ARM, BALANCED_ARM):
        raise ProtocolError("candidate_arm must be fast or balanced")
    if iterations_per_arm % 12:
        raise ProtocolError("iterations_per_arm must be divisible by twelve")
    iterations_per_segment = iterations_per_arm // 12
    entries: list[dict[str, Any]] = []
    for round_index, pattern in enumerate(PHYSICAL_ROUND_ORDER):
        pairs_per_round = 2 * iterations_per_segment
        for slot_index, token in enumerate(pattern):
            pair_group = 0 if slot_index < 2 else 1
            arm = NATIVE_ARM if token == "P" else candidate_arm
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


def _arms_plan(arms: Sequence[str]) -> dict[str, Any]:
    records: dict[str, Any] = {
        NATIVE_ARM: {
            "profile": "native C0",
            "head_type": "ROSANativeBaselineHead",
            "config_derivation": "unchanged frozen native config",
            "compile": False,
        }
    }
    if FAST_ARM in arms:
        records[FAST_ARM] = {
            "profile": "fast",
            "head_type": "ROSARefinementHead",
            "config_derivation": {
                "base": NATIVE_ARM,
                "only_changes": {
                    "model.head.type": "ROSARefinementHead",
                    "model.head.rosa_cfg": {"profile": "fast"},
                },
            },
            "compile": {
                "enabled": True,
                "target": "head.rosa_refiner",
                "mode": "reduce-overhead",
                "fullgraph": False,
                "compilation_excluded_from_timing": True,
            },
        }
    if BALANCED_ARM in arms:
        records[BALANCED_ARM] = {
            "profile": "balanced",
            "head_type": "ROSARefinementHead",
            "config_derivation": {
                "base": NATIVE_ARM,
                "only_changes": {
                    "model.head.type": "ROSARefinementHead",
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
    native_config_path = _resolve_file(args.native_config, "native config")
    if native_config_path != DEFAULT_NATIVE_CONFIG.resolve(strict=True):
        raise ProtocolError("native config must be the frozen cumulative C0 config")
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

    arms = selected_arms(args.candidate_profile)
    candidate_arm = selected_candidate(args.candidate_profile)
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
        )
        schedules[phase] = {
            "iterations_per_arm": iterations,
            "total_invocations": len(schedule),
            "sha256": canonical_sha256(schedule),
        }
    comparisons = {
        candidate_arm: {
            "parent_arm": NATIVE_ARM,
            "candidate_arm": candidate_arm,
            "physical_round_order": list(PHYSICAL_ROUND_ORDER),
            "schedules": schedules,
            "one_candidate_per_process": True,
        }
    }

    return {
        "schema": PLAN_SCHEMA,
        "status": "DRY_RUN",
        "protocol": PROTOCOL,
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
        "arms": _arms_plan(arms),
        "comparisons": comparisons,
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
            "capture_once_from": NATIVE_ARM,
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
                "maximum allocated/reserved and their increments"
            ),
            "compiler_guard": (
                "torch._dynamo counters must remain raw-exact across the entire "
                "recorded warmup and measured timing window"
            ),
        },
        "near_native_gate": {
            "source": "existing C0/C32 full-detector gate",
            "scope": "full model.test_step only",
            "median_ratio_ci95_upper_max": 1.10,
            "p95_ratio_max": 1.10,
            "peak_allocated_ratio_max": 1.02,
            "peak_reserved_ratio_max": 1.02,
            "failed_gate_policy": "make no near-native claim and report measurements",
        },
        "output_contract": {
            "sample_count": EXPECTED_BATCH_SIZE,
            "fields": ["bboxes_3d", "scores_3d", "labels_3d"],
            "box_and_score_values_finite": True,
            "labels_integral_and_in_six_class_range": True,
            "values_may_differ_between_profiles": True,
            "compiled_fast_must_be_raw_exact_with_eager_fast": True,
            "preflight_only_not_in_timed_interval": True,
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
    return {
        "logical_device": str(device),
        "logical_index": device.index,
        "gpu_uuid": _normalize_gpu_uuid(getattr(properties, "uuid", None)),
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
    native_config: Mapping[str, Any], profile: str
) -> dict[str, Any]:
    if profile not in ("fast", "balanced"):
        raise ProtocolError("candidate profile must be fast or balanced")
    expected = copy.deepcopy(dict(native_config))
    model = expected.get("model")
    if not isinstance(model, Mapping):
        raise ProtocolError("native config has no model mapping")
    head = model.get("head")
    if not isinstance(head, Mapping):
        raise ProtocolError("native config has no model.head mapping")
    if head.get("type") != "ROSANativeBaselineHead":
        raise ProtocolError("native config is not the exact C0 head")
    head["type"] = "ROSARefinementHead"
    head["rosa_cfg"] = {"profile": profile}
    return expected


def _derive_configs(verifier, native_config_path: Path, arms: Sequence[str]):
    native = verifier.Config.fromfile(native_config_path, import_custom_modules=False)
    native_dict = native.to_dict()
    configs = {NATIVE_ARM: native}
    for arm, profile in ((FAST_ARM, "fast"), (BALANCED_ARM, "balanced")):
        if arm not in arms:
            continue
        candidate = copy.deepcopy(native)
        candidate.model.head.type = "ROSARefinementHead"
        candidate.model.head.rosa_cfg = {"profile": profile}
        expected = _expected_candidate_config(native_dict, profile)
        if candidate.to_dict() != expected:
            raise ProtocolError(
                f"{arm} config differs from C0 outside head.type and head.rosa_cfg"
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
                    NATIVE_ARM
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


def _validate_checkpoint_semantics(run_tool, checkpoint, config, arm: str):
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
    expected_deltas = FAST_CHECKPOINT_ALLOWED_DELTAS if arm == FAST_ARM else {}
    if arm not in (FAST_ARM, BALANCED_ARM) or observed_deltas != expected_deltas:
        raise ProtocolError(
            f"{arm} checkpoint/runtime ROSA config delta is not allowed: "
            f"{observed_deltas}"
        )
    return {
        "validated_against_arm": arm,
        "source_refinement_profile": metadata["refinement_profile"],
        "rosa_contract_version": metadata["rosa_contract_version"],
        "projection_contract": observed_projection,
        "checkpoint_has_rosa_state": True,
        "source_effective_config_sha256": canonical_sha256(source_config),
        "runtime_effective_config_sha256": canonical_sha256(target_config),
        "compared_effective_fields": sorted(source_config),
        "allowed_runtime_delta": expected_deltas,
        "observed_runtime_delta": observed_deltas,
        "all_other_effective_fields_exact": True,
        "validator": "full effective ROSA config comparison",
    }


def _build_models(verifier, configs, state_dict, arms, device):
    from mmengine.utils import import_modules_from_strings
    from mmdet3d.registry import MODELS
    from mmdet3d.utils import register_all_modules
    from rosa.integrations.sparse4d import ROSANativeBaselineHead, ROSARefinementHead

    register_all_modules(init_default_scope=True)
    for config in configs.values():
        import_modules_from_strings(**config.custom_imports)
    models, contracts, load_contracts = {}, {}, {}
    for arm in arms:
        config = configs[arm]
        verifier._seed_everything(SEED)
        model = MODELS.build(copy.deepcopy(config.model))
        if arm == NATIVE_ARM:
            if type(model.head) is not ROSANativeBaselineHead:
                raise ProtocolError("native arm did not build the exact C0 head")
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
                "strict_native_subset": True,
                "excluded_sidecar_keys": excluded,
                "excluded_count": len(excluded),
                "missing_keys": list(result.missing_keys),
                "unexpected_keys": list(result.unexpected_keys),
            }
        else:
            if type(model.head) is not ROSARefinementHead:
                raise ProtocolError(f"{arm} did not build ROSARefinementHead")
            result = model.load_state_dict(_checkpoint_copy(state_dict), strict=True)
            profile = "fast" if arm == FAST_ARM else "balanced"
            expected_topk = 50 if arm == FAST_ARM else 100
            expected_steps = 1 if arm == FAST_ARM else 2
            if (
                model.head.rosa_cfg.refine_topk != expected_topk
                or model.head.rosa_cfg.num_refinement_steps != expected_steps
            ):
                raise ProtocolError(f"{profile} compute-budget contract drifted")
            load_contracts[arm] = {
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
    if NATIVE_ARM in models:
        native_state = models[NATIVE_ARM].state_dict()
        for arm in arms:
            if arm == NATIVE_ARM:
                continue
            candidate_state = models[arm].state_dict()
            candidate_native_state = {
                name: candidate_state[name] for name in native_state
            }
            verifier.require_same_fingerprint(
                f"{arm} shared native checkpoint state",
                native_state,
                candidate_native_state,
            )
    return models, contracts, load_contracts


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


def _preflight_outputs(verifier, torch, models, source_batch, device):
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
            del output
    if verifier.raw_fingerprint(source_batch) != source_before:
        raise ProtocolError("output preflight mutated the captured source batch")
    return records


def _cuda_allocator_snapshot(torch, device) -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def _stabilize_cuda_allocator(torch, device) -> dict[str, int]:
    torch.cuda.synchronize(device)
    gc.collect()
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
):
    arms = (NATIVE_ARM, candidate_arm)
    allocator_floor = _stabilize_cuda_allocator(torch, device)
    memory: dict[str, Any] = {
        "contract": {
            "one_benchmark_model_resident_per_probe": True,
            "peer_model_resident": False,
            "allocator_floor": allocator_floor,
            "probe_count_per_arm": 3,
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
        try:
            models, built_contracts, built_load_contracts = _build_models(
                verifier, configs, state_dict, (arm,), device
            )
            contracts[arm] = built_contracts[arm]
            load_contracts[arm] = built_load_contracts[arm]
            compiled = False
            if arm == FAST_ARM:
                compiled = bool(
                    run_tool._compile_fast_inference(
                        SimpleNamespace(model=models[arm]), configs[arm]
                    )
                )
                compiler_cache_created = compiled
                if not compiled:
                    raise ProtocolError(
                        "isolated fast memory model did not compile its ROSA refiner"
                    )
            preflight = {
                str(batch_index): _preflight_outputs(
                    verifier, torch, {arm: models[arm]}, source, device
                )[arm]
                for batch_index, source in enumerate(sources)
            }
            record = _measure_single_model_memory(
                verifier, torch, models[arm], sources[0], device, arm
            )
            record.update(
                {
                    "allocator_before_build": before_build,
                    "compiled": compiled,
                    "preflight_by_batch": preflight,
                }
            )
            memory[arm] = record
        finally:
            models.clear()
            if compiler_cache_created:
                _reset_compiler_cache(torch)
            after_cleanup = _stabilize_cuda_allocator(torch, device)
        _require_allocator_floor(
            allocator_floor, after_cleanup, label=f"after {arm} memory cleanup"
        )
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


def _run_pair_protocol(
    verifier,
    torch,
    models,
    sources,
    manifests,
    device,
    candidate_arm,
):
    _require_current_default_stream(torch, device)
    compiler_before = (
        _compiler_counter_snapshot(torch) if candidate_arm == FAST_ARM else None
    )
    result = verifier._run_scope(
        scope=verifier.C0_C32_FULL_DETECTOR_SCOPE,
        models={
            verifier.PARENT_ARM: models[NATIVE_ARM],
            verifier.CANDIDATE_ARM: models[candidate_arm],
        },
        sources=sources,
        manifests=manifests,
        device=device,
    )
    if compiler_before is None:
        result["compiler_guard"] = {
            "applicable": False,
            "reason": "balanced candidate is eager-only",
        }
    else:
        compiler_after = _compiler_counter_snapshot(torch)
        result["compiler_guard"] = _require_compiler_counters_unchanged(
            compiler_before, compiler_after
        )
        result["compiler_guard"]["applicable"] = True
    result["stream_after"] = _require_current_default_stream(torch, device)
    result["arm_aliases"] = {
        verifier.PARENT_ARM: NATIVE_ARM,
        verifier.CANDIDATE_ARM: candidate_arm,
    }
    return result


def _evaluate_near_native_gate(
    verifier,
    timings: Mapping[str, Any],
    memory: Mapping[str, Any],
    candidate_arm: str,
) -> Mapping[str, Any]:
    paired = timings.get("paired_bootstrap")
    if not isinstance(paired, Mapping):
        raise ProtocolError("paired timing result is missing its bootstrap evidence")
    logical_memory = {
        verifier.PARENT_ARM: memory[NATIVE_ARM],
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
        "parent_arm": NATIVE_ARM,
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
        arms = selected_arms(args.candidate_profile)
        candidate_arm = selected_candidate(args.candidate_profile)
        configs = _derive_configs(verifier, Path(args.native_config), arms)
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
        )
        state_dict = run_tool._checkpoint_state_dict(checkpoint)
        if not isinstance(state_dict, Mapping):
            raise ProtocolError("checkpoint has no recognizable state_dict")
        checkpoint_schema = verifier._checkpoint_schema(state_dict)

        _register_runtime_modules(configs)
        prepared = SimpleNamespace(
            parent_config=configs[NATIVE_ARM], expected_manifest=None
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
            )
        )
        _atomic_json(pair_dir / "MEMORY.json", memory)
        gpu_snapshots.append(_nvidia_smi_snapshot(identity["gpu_uuid"]))

        pair_arms = (NATIVE_ARM, candidate_arm)
        models, pair_model_contracts, pair_load_contracts = _build_models(
            verifier, configs, state_dict, pair_arms, device
        )
        model_contracts[candidate_arm] = pair_model_contracts
        load_contracts[candidate_arm] = pair_load_contracts

        eager_candidate_preflight = None
        eager_candidate_rng = None
        eager_candidate_state = None
        if candidate_arm == FAST_ARM:
            eager_candidate_rng = verifier._capture_rng()
            eager_candidate_state = verifier.raw_fingerprint(
                models[candidate_arm].state_dict()
            )
            eager_candidate_preflight = {
                str(batch_index): _preflight_outputs(
                    verifier,
                    torch,
                    {candidate_arm: models[candidate_arm]},
                    source,
                    device,
                )[candidate_arm]
                for batch_index, source in enumerate(sources)
            }
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
            verifier._restore_rng(eager_candidate_rng)
            compiled[candidate_arm] = bool(
                run_tool._compile_fast_inference(
                    SimpleNamespace(model=models[candidate_arm]),
                    configs[candidate_arm],
                )
            )
            if not compiled[candidate_arm]:
                raise ProtocolError("fast arm did not compile its ROSA refiner")

        _require_current_default_stream(torch, device)
        output_preflight = _preflight_outputs(
            verifier, torch, models, sources[0], device
        )
        compiled_parity = None
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

        compiler_warmup_by_batch = None
        if candidate_arm == FAST_ARM:
            warmup_rng = verifier._capture_rng()
            warmup_state = verifier.raw_fingerprint(models[candidate_arm].state_dict())
            compiler_warmup_by_batch = {
                str(batch_index): _preflight_outputs(
                    verifier,
                    torch,
                    {candidate_arm: models[candidate_arm]},
                    source,
                    device,
                )[candidate_arm]
                for batch_index, source in enumerate(sources)
            }
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
            compiled_parity = _require_all_batch_compiled_parity(
                eager_candidate_preflight,
                compiler_warmup_by_batch,
                batch_count=len(sources),
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
        )
        warmup_records = timings.pop("warmup_records")
        measured_records = timings.pop("measured_records")
        near_native_gate = _evaluate_near_native_gate(
            verifier,
            timings,
            memory,
            candidate_arm,
        )
        _write_jsonl(pair_dir / "WARMUP.jsonl", warmup_records)
        _write_jsonl(pair_dir / "TIMINGS.jsonl", measured_records)
        _atomic_json(pair_dir / "STATISTICS.json", timings)
        comparison_results[candidate_arm] = {
            "parent_arm": NATIVE_ARM,
            "candidate_arm": candidate_arm,
            "output_preflight": output_preflight,
            "compiled_eager_parity": compiled_parity,
            "compiler_warmup_by_batch": compiler_warmup_by_batch,
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
            "protocol": PROTOCOL,
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
            "output_contract": {
                "finite_and_well_formed": True,
                "values_allowed_to_differ": True,
            },
            "comparisons": comparison_results,
            "environment": environment,
            "physical_gpu": identity,
            "nvidia_smi": gpu_snapshots,
        }
        _atomic_json(output_dir / "EVIDENCE.json", evidence)
        _seal_directory(output_dir)
        completion = {
            "schema": "rosa.full_rosa_near_native_completion.v1",
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
                    "schema": "rosa.full_rosa_near_native_failure.v1",
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
