#!/usr/bin/env python3
"""Disarmed exact C0-vs-C32 full-detector CUDA runtime protocol.

The default path performs CPU-only validation and prints a sealed plan.  It
does not inspect CUDA, build a model or dataloader, create a directory, or
acquire a run lock.  A real run additionally requires ``--execute``, explicit
``ARMED`` state, the exact idle/reserved token, an expected source revision,
and an expected checkpoint SHA256.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import gc
import hashlib
import importlib
from importlib.machinery import ExtensionFileLoader
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import time
import traceback
from types import ModuleType
from typing import Any

import numpy as np
import torch
from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
while str(SRC_ROOT) in sys.path:
    sys.path.remove(str(SRC_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from rosa.runtime_verification import (  # noqa: E402
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    C0_C32_COMMON_MODEL_SCOPE,
    C0_C32_FULL_DETECTOR_SCOPE,
    C0_C32_NEAR_NATIVE_PROTOCOL,
    CANDIDATE_ARM,
    DEFAULT_MEASURED_ITERATIONS_PER_ARM,
    DEFAULT_WARMUP_ITERATIONS_PER_ARM,
    PARENT_ARM,
    PHYSICAL_ROUND_ORDER,
    VerificationContractError,
    build_alternating_schedule,
    build_batch_manifest,
    canonical_sha256,
    evaluate_c0_c32_near_native_gate,
    evaluate_c32_near_native_gate,
    file_sha256,
    finite_output_schema,
    paired_bootstrap_confidence_intervals,
    raw_fingerprint,
    require_same_fingerprint,
    semantic_sample_digest,
    summarize_timings,
    validate_c0_c32_near_native_config_pair,
    validate_expected_manifest,
    validate_runtime_parameters,
)

IDLE_RESERVED_TOKEN = "I_HAVE_VERIFIED_THE_GPU_IS_IDLE_AND_RESERVED"
EXPECTED_BATCH_SIZE = 4
CAPTURE_BATCHES = 2
EXPECTED_CLASSES = ("Car", "Van", "Truck", "Bus", "Pedestrian", "Cyclist")
EXPECTED_ANNOTATION_SHA256 = (
    "efafe89c6dc34d116f0ce01fa818505a6387248d9a3ca6dcac035418a664da6f"
)
DEFAULT_DETECTOR_HOST_ROOT = Path(
    "/home/airlab/user/kdh/ROSA_host_wt_exacte94_native_eval_p1p2_strict"
)
DEFAULT_OFFICIAL_ROOT = Path("/home/airlab/user/kdh/dependencies/fisheye3dod")
EXPECTED_OFFICIAL_REVISION = "a28726ed002a02cb1328960848c0f4f2473748de"
EXPECTED_OFFICIAL_SOURCE_SHA256 = (
    "007f8d0f1f2645fa425085f2368215978eef0ad60467fbc9f67c331d2b1913db"
)
DEFAULT_DAF_EXTENSION_SHA256 = (
    "1831505bfb68183344ec8daa60c8d4d38bc806b60516bb3389e6190f463ffa30"
)
DAF_EXTENSION_BASENAME = (
    "deformable_aggregation_ext.cpython-310-x86_64-linux-gnu.so"
)
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
HOST_DAF_SOURCE_FILES = (
    "projects/mmdet3d_plugin/ops/__init__.py",
    "projects/mmdet3d_plugin/ops/deformable_aggregation.py",
    "projects/mmdet3d_plugin/ops/setup.py",
    "projects/mmdet3d_plugin/ops/src/deformable_aggregation.cpp",
    "projects/mmdet3d_plugin/ops/src/deformable_aggregation_cuda.cu",
)
REQUIRED_DAF_EXTENSION_SYMBOLS = (
    "deformable_aggregation_forward",
    "deformable_aggregation_backward",
    "deformable_aggregation_per_camera_forward",
    "deformable_aggregation_per_camera_backward",
    "deformable_aggregation_per_camera_deterministic_backward",
)
DAF_SUPPORTED_CUDA_CAPABILITIES = (
    (8, 0),
    (8, 6),
    (8, 9),
    (9, 0),
    (12, 0),
)
HOST_MODULE_FILES = {
    "projects.mmdet3d_plugin": "projects/mmdet3d_plugin/__init__.py",
    "projects.mmdet3d_plugin.ops": "projects/mmdet3d_plugin/ops/__init__.py",
    "projects.mmdet3d_plugin.ops.deformable_aggregation": (
        "projects/mmdet3d_plugin/ops/deformable_aggregation.py"
    ),
    "projects.mmdet3d_plugin.models.sparse4d": (
        "projects/mmdet3d_plugin/models/sparse4d.py"
    ),
    "projects.mmdet3d_plugin.models.sparse4d_head": (
        "projects/mmdet3d_plugin/models/sparse4d_head.py"
    ),
    "projects.mmdet3d_plugin.models.instance_bank": (
        "projects/mmdet3d_plugin/models/instance_bank.py"
    ),
    "projects.mmdet3d_plugin.models.detection3d.decoder": (
        "projects/mmdet3d_plugin/models/detection3d/decoder.py"
    ),
    "projects.mmdet3d_plugin.models.blocks": (
        "projects/mmdet3d_plugin/models/blocks.py"
    ),
}
PARENT_CONFIG = ROOT / (
    "configs/rosa_r18_main126_v2_c0_cumulative_runtime_exact.py"
)
CANDIDATE_CONFIG = ROOT / (
    "configs/rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_"
    "cpu_metadata_lean_loader_rgb_uint8_transfer_pinned_inplace_"
    "homogeneous4x4_inference_reuse_exact.py"
)
FROZEN_CONFIG_FILES = (
    str(PARENT_CONFIG.relative_to(ROOT)),
    str(CANDIDATE_CONFIG.relative_to(ROOT)),
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class PreparedProtocol:
    args: argparse.Namespace
    parent_config: Config
    candidate_config: Config
    checkpoint: Mapping[str, Any]
    state_dict: Mapping[str, torch.Tensor]
    expected_manifest: Mapping[str, Any] | None
    detector_host_root: Path
    official_root: Path
    plan: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument(
        "--detector-host-root",
        type=Path,
        default=DEFAULT_DETECTOR_HOST_ROOT,
    )
    parser.add_argument("--expected-detector-host-revision", required=True)
    parser.add_argument(
        "--official-root",
        type=Path,
        default=DEFAULT_OFFICIAL_ROOT,
    )
    parser.add_argument(
        "--expected-daf-extension-sha256",
        default=DEFAULT_DAF_EXTENSION_SHA256,
    )
    parser.add_argument("--expected-sample-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Permit the separately armed CUDA path.",
    )
    parser.add_argument(
        "--execution-state",
        choices=("DISARMED", "ARMED"),
        default="DISARMED",
    )
    parser.add_argument(
        "--idle-reserved-token",
        help=f"Required with --execute; exact value is {IDLE_RESERVED_TOKEN!r}.",
    )
    return parser


def _run_git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _run_git_at(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _git_blob_sha256(root: Path, revision: str, relative: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "show", f"{revision}:{relative}"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def _git_snapshot_at(root: Path, *, require_clean: bool) -> dict[str, Any]:
    revision = _run_git_at(root, "rev-parse", "HEAD")
    tree = _run_git_at(root, "rev-parse", "HEAD^{tree}")
    status = _run_git_at(root, "status", "--porcelain", "--untracked-files=all")
    if require_clean and status:
        raise VerificationContractError(
            f"runtime dependency must be clean: {root}"
        )
    return {
        "root": str(root),
        "revision": revision,
        "tree": tree,
        "clean": not bool(status),
        "status_sha256": canonical_sha256(status.splitlines()),
    }


def _git_snapshot(*, require_clean: bool) -> dict[str, Any]:
    return _git_snapshot_at(ROOT, require_clean=require_clean)


def _resolve_directory(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise VerificationContractError(f"{label} does not exist: {path}") from error
    if not resolved.is_dir():
        raise VerificationContractError(f"{label} must be a directory: {resolved}")
    return resolved


def _pin_import_roots(detector_host_root: Path) -> None:
    roots = (str(SRC_ROOT), str(detector_host_root))
    for root in roots:
        while root in sys.path:
            sys.path.remove(root)
    sys.path.insert(0, roots[1])
    sys.path.insert(0, roots[0])


def _pin_official_import_roots(official_root: Path) -> None:
    roots = (str(official_root), str(official_root.parent))
    for root in roots:
        while root in sys.path:
            sys.path.remove(root)
    sys.path[2:2] = list(roots)


def _is_module_within(module: Any, root: Path) -> bool:
    source = getattr(module, "__file__", None)
    if source is not None:
        return _is_within(Path(source).resolve(), root)
    search = getattr(module, "__path__", None)
    return search is not None and all(
        _is_within(Path(entry).resolve(), root) for entry in search
    )


def _reject_foreign_preimported_host(root: Path) -> None:
    foreign = [
        name
        for name, module in sys.modules.items()
        if (name == "projects" or name.startswith("projects."))
        and module is not None
        and not _is_module_within(module, root)
    ]
    if foreign:
        raise VerificationContractError(
            "foreign detector-host modules were imported before protocol setup: "
            + ", ".join(sorted(foreign)[:8])
        )


def _reject_foreign_preimported_rosa(root: Path) -> None:
    foreign = [
        name
        for name, module in sys.modules.items()
        if (name == "rosa" or name.startswith("rosa."))
        and module is not None
        and not _is_module_within(module, root)
    ]
    if foreign:
        raise VerificationContractError(
            "foreign ROSA modules were imported before protocol setup: "
            + ", ".join(sorted(foreign)[:8])
        )


def _reject_foreign_preimported_official(root: Path) -> None:
    official_prefix = root.name
    relevant = []
    for name, module in sys.modules.items():
        if module is None:
            continue
        if not (
            name == official_prefix
            or name.startswith(official_prefix + ".")
            or name == "utils"
            or name == "utils.fisheye3dod_calib"
        ):
            continue
        if not _is_module_within(module, root):
            relevant.append(name)
    if relevant:
        raise VerificationContractError(
            "foreign official Fisheye3DOD modules were pre-imported: "
            + ", ".join(sorted(relevant)[:8])
        )


def _python_source_contract(root: Path, *, label: str) -> dict[str, Any]:
    files = {}
    for path in sorted(root.rglob("*.py")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        files[relative] = file_sha256(path)
    if not files:
        raise VerificationContractError(f"{label} contains no Python sources")
    return {
        "root": str(root),
        "files": files,
        "sha256": canonical_sha256(files),
    }


def _loaded_python_module_contract(
    root: Path,
    *,
    module_prefixes: tuple[str, ...],
    exact_names: tuple[str, ...] = (),
    label: str,
) -> dict[str, Any]:
    files = {}
    identity = {}
    for name, module in sorted(sys.modules.items()):
        if module is None or not (
            name in exact_names
            or any(name == prefix or name.startswith(prefix + ".") for prefix in module_prefixes)
        ):
            continue
        source = getattr(module, "__file__", None)
        if source is None:
            search = getattr(module, "__path__", None)
            if search is None:
                raise VerificationContractError(
                    f"{label} module has no source or namespace path: {name}"
                )
            paths = [Path(entry).resolve() for entry in search]
            if not paths or any(not _is_within(path, root) for path in paths):
                raise VerificationContractError(
                    f"{label} namespace resolved outside pinned root: {name}"
                )
            relative_paths = [str(path.relative_to(root)) for path in paths]
            record = {
                "kind": "namespace",
                "paths": [str(path) for path in paths],
                "relative_paths": relative_paths,
            }
            files[name] = record
            identity[name] = {
                "kind": "namespace",
                "relative_paths": relative_paths,
            }
            continue
        path = Path(source).resolve()
        if not _is_within(path, root):
            raise VerificationContractError(
                f"{label} module resolved outside pinned root: {name}"
            )
        relative = str(path.relative_to(root))
        record = {
            "kind": "source",
            "path": str(path),
            "relative_path": relative,
            "sha256": file_sha256(path),
        }
        files[name] = record
        identity[name] = {
            "kind": "source",
            "relative_path": relative,
            "sha256": record["sha256"],
        }
    if not files:
        raise VerificationContractError(f"no loaded {label} modules were observed")
    return {
        "root": str(root),
        "files": files,
        "sha256": canonical_sha256(identity),
    }


def _loaded_rosa_module_contract(root: Path) -> dict[str, Any]:
    _reject_foreign_preimported_rosa(root)
    return _loaded_python_module_contract(
        root,
        module_prefixes=("rosa",),
        label="ROSA",
    )


def _loaded_official_module_contract(root: Path) -> dict[str, Any]:
    _reject_foreign_preimported_official(root)
    return _loaded_python_module_contract(
        root,
        module_prefixes=(root.name,),
        exact_names=("utils", "utils.fisheye3dod_calib"),
        label="official Fisheye3DOD",
    )


def _host_source_contract(root: Path) -> dict[str, Any]:
    files = {}
    for module_name, relative in HOST_MODULE_FILES.items():
        path = _resolve_file(root / relative, f"detector-host source {relative}")
        files[module_name] = {
            "relative_path": relative,
            "sha256": file_sha256(path),
        }
    return {
        "root": str(root),
        "files": files,
        "sha256": canonical_sha256(files),
    }


def _host_daf_source_contract(root: Path) -> dict[str, Any]:
    files = {}
    for relative in HOST_DAF_SOURCE_FILES:
        path = _resolve_file(root / relative, f"DAF source {relative}")
        files[relative] = file_sha256(path)
    return {
        "root": str(root),
        "files": files,
        "sha256": canonical_sha256(files),
    }


def _daf_binary_file_contract(
    root: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Verify the pinned DAF binary without importing it or touching CUDA."""

    expected_sha256 = _validate_sha256(
        expected_sha256,
        "expected DAF extension SHA256",
    )
    ops_root = _resolve_directory(
        root / "projects/mmdet3d_plugin/ops",
        "pinned detector-host ops root",
    )
    expected_path = ops_root / DAF_EXTENSION_BASENAME
    try:
        expected_path.lstat()
    except FileNotFoundError as error:
        raise VerificationContractError(
            f"pinned DAF extension binary does not exist: {expected_path}"
        ) from error
    if expected_path.is_symlink():
        raise VerificationContractError(
            "pinned DAF extension binary must be a regular non-symlink file"
        )
    path = _resolve_file(expected_path, "pinned DAF extension binary")
    if path.parent != ops_root or path.name != DAF_EXTENSION_BASENAME:
        raise VerificationContractError(
            "pinned DAF extension binary escaped its exact host path"
        )
    digest = file_sha256(path)
    if digest != expected_sha256:
        raise VerificationContractError(
            f"pinned DAF extension binary hash is {digest}, "
            f"expected {expected_sha256}"
        )
    return {
        "path": str(path),
        "basename": path.name,
        "sha256": digest,
        "size_bytes": path.stat().st_size,
        "regular_non_symlink": True,
    }


def _loaded_daf_extension_contract(
    root: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    ops = importlib.import_module("projects.mmdet3d_plugin.ops")
    if getattr(ops, "_DAF_BUILT", None) is not True:
        raise VerificationContractError(
            "pinned detector host has no loadable deformable aggregation extension"
        )
    extension = getattr(ops, "deformable_aggregation_ext", None)
    expected_module_name = (
        "projects.mmdet3d_plugin.ops.deformable_aggregation_ext"
    )
    if (
        type(extension) is not ModuleType
        or getattr(extension, "__name__", None) != expected_module_name
        or sys.modules.get(expected_module_name) is not extension
    ):
        raise VerificationContractError(
            "DAF extension module identity is not the pinned pybind module"
        )
    source = getattr(extension, "__file__", None)
    if source is None:
        raise VerificationContractError("loaded DAF extension has no binary path")
    path = Path(source).resolve()
    expected_parent = (root / "projects/mmdet3d_plugin/ops").resolve()
    if path.parent != expected_parent:
        raise VerificationContractError(
            "DAF extension resolved outside the pinned detector host"
        )
    if path.name != DAF_EXTENSION_BASENAME:
        raise VerificationContractError(
            f"unexpected DAF extension basename: {path.name}"
        )
    specification = getattr(extension, "__spec__", None)
    loader = getattr(specification, "loader", None)
    origin = getattr(specification, "origin", None)
    if (
        not isinstance(loader, ExtensionFileLoader)
        or origin is None
        or Path(origin).resolve() != path
    ):
        raise VerificationContractError(
            "DAF extension was not loaded by ExtensionFileLoader from its binary"
        )
    digest = file_sha256(path)
    if digest != expected_sha256:
        raise VerificationContractError(
            f"DAF extension hash is {digest}, expected {expected_sha256}"
        )
    invalid = [
        name
        for name in REQUIRED_DAF_EXTENSION_SYMBOLS
        if not callable(getattr(extension, name, None))
        or not inspect.isbuiltin(getattr(extension, name, None))
    ]
    if invalid:
        raise VerificationContractError(
            "DAF extension lacks required builtin bindings: "
            + ", ".join(invalid)
        )
    return {
        "path": str(path),
        "sha256": digest,
        "required_symbols": list(REQUIRED_DAF_EXTENSION_SYMBOLS),
        "source_contract": _host_daf_source_contract(root),
    }


def _loaded_host_module_contract(root: Path) -> dict[str, Any]:
    files = {}
    identity = {}
    for module_name, relative in HOST_MODULE_FILES.items():
        module = importlib.import_module(module_name)
        source = getattr(module, "__file__", None)
        if source is None:
            raise VerificationContractError(
                f"detector-host module has no source file: {module_name}"
            )
        path = Path(source).resolve()
        expected = (root / relative).resolve()
        if path != expected:
            raise VerificationContractError(
                f"detector-host module resolved outside pinned source: {module_name}"
            )
        files[module_name] = {
            "path": str(path),
            "relative_path": relative,
            "sha256": file_sha256(path),
        }
        identity[module_name] = {
            "relative_path": relative,
            "sha256": files[module_name]["sha256"],
        }
    root_text = str(root)
    if root_text not in sys.path:
        raise VerificationContractError(
            "pinned detector-host root is absent from sys.path"
        )
    contract = {
        "root": root_text,
        "files": files,
        "sys_path": list(sys.path),
        "sys_path_index": sys.path.index(root_text),
    }
    contract["sha256"] = canonical_sha256(identity)
    return contract


def _resolve_file(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise VerificationContractError(f"{label} does not exist: {path}") from error
    if not resolved.is_file():
        raise VerificationContractError(f"{label} must be a file: {resolved}")
    return resolved


def _load_json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationContractError(f"invalid {label}: {path}") from error
    if not isinstance(value, Mapping):
        raise VerificationContractError(f"{label} must contain a JSON object")
    return value


def _load_run_tool():
    specification = importlib.util.spec_from_file_location(
        "rosa_run_tool_for_c0_c32_runtime", ROOT / "tools/run.py"
    )
    if specification is None or specification.loader is None:
        raise VerificationContractError("could not load tools/run.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _checkpoint_schema(state_dict: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    records = []
    for name, value in state_dict.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise VerificationContractError(
                "checkpoint state_dict must map strings to tensors"
            )
        records.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "numel": value.numel(),
            }
        )
    if not records:
        raise VerificationContractError("checkpoint state_dict is empty")
    return {
        "keys": len(records),
        "parameters_and_buffers_numel": sum(item["numel"] for item in records),
        "ordered_schema_sha256": canonical_sha256(records),
    }


def _frozen_config_contract() -> dict[str, Any]:
    observed = {}
    revision = _run_git("rev-parse", "HEAD")
    for relative in FROZEN_CONFIG_FILES:
        path = _resolve_file(ROOT / relative, f"frozen config {relative}")
        digest = file_sha256(path)
        expected = _git_blob_sha256(ROOT, revision, relative)
        if digest != expected:
            raise VerificationContractError(
                f"frozen config drift: {relative} has {digest}, expected {expected}"
            )
        observed[relative] = digest
    return observed


def _validate_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise VerificationContractError(f"{label} must be 64 lowercase hex digits")
    return value


def _validate_revision(value: str) -> str:
    if not isinstance(value, str) or REVISION_PATTERN.fullmatch(value) is None:
        raise VerificationContractError(
            "expected source revision must be a full 40-character lowercase SHA"
        )
    return value


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def prepare_protocol(args: argparse.Namespace) -> PreparedProtocol:
    """Perform CPU-only validation without constructing runtime objects."""

    _reject_foreign_preimported_rosa(SRC_ROOT)
    rosa_sources = _python_source_contract(SRC_ROOT, label="ROSA source root")
    expected_checkpoint_sha256 = _validate_sha256(
        args.expected_checkpoint_sha256, "expected checkpoint SHA256"
    )
    expected_daf_extension_sha256 = _validate_sha256(
        args.expected_daf_extension_sha256,
        "expected DAF extension SHA256",
    )
    if expected_daf_extension_sha256 != DEFAULT_DAF_EXTENSION_SHA256:
        raise VerificationContractError(
            "the exact C0/C32 protocol forbids overriding the blessed DAF binary"
        )
    expected_revision = _validate_revision(args.expected_source_revision)
    expected_host_revision = _validate_revision(
        args.expected_detector_host_revision
    )
    repository = _git_snapshot(require_clean=False)
    if repository["revision"] != expected_revision:
        raise VerificationContractError(
            "current source revision does not match --expected-source-revision"
        )
    detector_host_root = _resolve_directory(
        args.detector_host_root,
        "detector host root",
    )
    detector_host = _git_snapshot_at(detector_host_root, require_clean=False)
    if detector_host["revision"] != expected_host_revision:
        raise VerificationContractError(
            "detector-host revision does not match the expected revision"
        )
    for variable in ("ROSA_DETECTOR_HOST_ROOT", "SPARSE4D_ROOT"):
        existing = os.environ.get(variable)
        if existing is not None and Path(existing).expanduser().resolve() != (
            detector_host_root
        ):
            raise VerificationContractError(
                f"{variable} conflicts with --detector-host-root"
            )
        os.environ[variable] = str(detector_host_root)
    _pin_import_roots(detector_host_root)
    _reject_foreign_preimported_host(detector_host_root)
    detector_host_sources = _host_source_contract(detector_host_root)
    detector_host_daf_sources = _host_daf_source_contract(detector_host_root)
    for relative, expected in BLESSED_DAF_BUILD_INPUT_SHA256.items():
        observed = detector_host_daf_sources["files"].get(relative)
        if observed != expected:
            raise VerificationContractError(
                f"DAF build input drift: {relative} has {observed}, expected {expected}"
            )
    wrapper_relative = (
        "projects/mmdet3d_plugin/ops/deformable_aggregation.py"
    )
    if detector_host_daf_sources["files"].get(wrapper_relative) != (
        BLESSED_DAF_WRAPPER_SHA256
    ):
        raise VerificationContractError("DAF Python wrapper drifted")
    daf_binary = _daf_binary_file_contract(
        detector_host_root,
        expected_sha256=expected_daf_extension_sha256,
    )
    official_root = _resolve_directory(
        args.official_root,
        "official Fisheye3DOD root",
    )
    _resolve_file(
        official_root / "dataset/fisheye3dod_metric.py",
        "official Fisheye3DOD metric",
    )
    _resolve_file(
        official_root / "utils/fisheye3dod_calib.py",
        "official Fisheye3DOD calibration module",
    )
    _pin_official_import_roots(official_root)
    existing_official = os.environ.get("FISHEYE3DOD_OFFICIAL_ROOT")
    if existing_official is not None and Path(existing_official).expanduser().resolve() != (
        official_root
    ):
        raise VerificationContractError(
            "FISHEYE3DOD_OFFICIAL_ROOT conflicts with --official-root"
        )
    os.environ["FISHEYE3DOD_OFFICIAL_ROOT"] = str(official_root)
    _reject_foreign_preimported_official(official_root)
    official_sources = _python_source_contract(
        official_root,
        label="official Fisheye3DOD root",
    )
    official_git = _git_snapshot_at(official_root, require_clean=False)
    if official_git["revision"] != EXPECTED_OFFICIAL_REVISION:
        raise VerificationContractError(
            "official Fisheye3DOD base revision differs from the pinned revision"
        )
    if official_sources["sha256"] != EXPECTED_OFFICIAL_SOURCE_SHA256:
        raise VerificationContractError(
            "patched official Fisheye3DOD Python source contract drifted"
        )
    if (
        not isinstance(args.device, str)
        or re.fullmatch(r"cuda:[0-9]+", args.device) is None
    ):
        raise VerificationContractError("device must have exact form cuda:N")
    output_dir = args.output_dir.expanduser().resolve()
    if _is_within(output_dir, ROOT):
        raise VerificationContractError(
            "runtime output must be outside the source worktree"
        )
    if not output_dir.parent.is_dir():
        raise VerificationContractError("runtime output parent must already exist")

    frozen_configs = _frozen_config_contract()
    parent_config = Config.fromfile(PARENT_CONFIG, import_custom_modules=False)
    candidate_config = Config.fromfile(CANDIDATE_CONFIG, import_custom_modules=False)
    config_contract = validate_c0_c32_near_native_config_pair(
        parent_config,
        candidate_config,
        expected_batch_size=EXPECTED_BATCH_SIZE,
    )
    parent_annotation = _resolve_file(
        Path(parent_config.test_dataloader.dataset.ann_file),
        "parent Main126 annotation",
    )
    candidate_annotation = _resolve_file(
        Path(candidate_config.test_dataloader.dataset.ann_file),
        "candidate Main126 annotation",
    )
    if parent_annotation != candidate_annotation:
        raise VerificationContractError(
            "C0 and C32 resolve different Main126 annotation files"
        )
    annotation_sha256 = file_sha256(parent_annotation)
    if annotation_sha256 != EXPECTED_ANNOTATION_SHA256:
        raise VerificationContractError(
            "Main126 validation annotation hash does not match the frozen dataset"
        )

    checkpoint_path = _resolve_file(args.checkpoint, "checkpoint")
    observed_checkpoint_sha256 = file_sha256(checkpoint_path)
    if observed_checkpoint_sha256 != expected_checkpoint_sha256:
        raise VerificationContractError(
            "checkpoint hash does not match --expected-checkpoint-sha256"
        )
    run_tool = _load_run_tool()
    checkpoint = run_tool._load_checkpoint_without_executing_code(checkpoint_path)
    if not isinstance(checkpoint, Mapping):
        raise VerificationContractError("checkpoint must contain a mapping")
    state_dict = run_tool._checkpoint_state_dict(checkpoint)
    if not isinstance(state_dict, Mapping):
        raise VerificationContractError("checkpoint has no recognizable state_dict")
    if run_tool._state_dict_has_rosa(state_dict):
        raise VerificationContractError(
            "C0/C32 protocol requires a native checkpoint without ROSA sidecar state"
        )
    checkpoint_schema = _checkpoint_schema(state_dict)

    expected_manifest = None
    expected_manifest_record = None
    if args.expected_sample_manifest is not None:
        manifest_path = _resolve_file(
            args.expected_sample_manifest, "expected sample manifest"
        )
        expected_manifest = _load_json_mapping(
            manifest_path, "expected sample manifest"
        )
        expected_manifest_record = {
            "path": str(manifest_path),
            "sha256": file_sha256(manifest_path),
        }

    parameters = validate_runtime_parameters(
        capture_batches=CAPTURE_BATCHES,
        warmup_iterations_per_arm=DEFAULT_WARMUP_ITERATIONS_PER_ARM,
        measured_iterations_per_arm=DEFAULT_MEASURED_ITERATIONS_PER_ARM,
        expected_batch_size=EXPECTED_BATCH_SIZE,
        profile_batch_index=0,
    )
    if (
        parameters["warmup_iterations_per_arm"] != 60
        or parameters["measured_iterations_per_arm"] != 300
    ):
        raise VerificationContractError("frozen 60/300 schedule constants drifted")
    schedules = {}
    for phase, count in (
        ("warmup", DEFAULT_WARMUP_ITERATIONS_PER_ARM),
        ("measured", DEFAULT_MEASURED_ITERATIONS_PER_ARM),
    ):
        serialized = [
            entry.to_dict()
            for entry in build_alternating_schedule(
                iterations_per_arm=count,
                batch_count=CAPTURE_BATCHES,
                phase=phase,
            )
        ]
        schedules[phase] = {
            "iterations_per_arm": count,
            "total_invocations": len(serialized),
            "sha256": canonical_sha256(serialized),
        }

    plan = {
        "schema": "rosa.c0_c32_near_native_runtime_plan.v1",
        "status": "DRY_RUN",
        "protocol": C0_C32_NEAR_NATIVE_PROTOCOL,
        "execution_requested": bool(args.execute),
        "execution_state": args.execution_state,
        "cuda_queried": False,
        "output_created": False,
        "repository": repository,
        "rosa_source": {
            "source_contract": rosa_sources,
            "loaded_modules": "verified after armed custom imports",
        },
        "detector_host": {
            "git": detector_host,
            "source_contract": detector_host_sources,
            "daf_source_contract": detector_host_daf_sources,
            "expected_daf_extension_sha256": expected_daf_extension_sha256,
            "daf_binary_contract": daf_binary,
            "loaded_daf_extension": "verified only after armed custom imports",
            "loaded_modules": "verified only after armed custom imports",
        },
        "official_fisheye3dod": {
            "root": str(official_root),
            "identity": "pinned locally patched official source",
            "git": official_git,
            "source_contract": official_sources,
            "loaded_modules": "verified after armed dataset capture",
        },
        "frozen_config_files": frozen_configs,
        "parent_config": {
            "path": str(PARENT_CONFIG),
            "sha256": file_sha256(PARENT_CONFIG),
            "effective_sha256": canonical_sha256(parent_config.to_dict()),
        },
        "candidate_config": {
            "path": str(CANDIDATE_CONFIG),
            "sha256": file_sha256(CANDIDATE_CONFIG),
            "effective_sha256": canonical_sha256(candidate_config.to_dict()),
        },
        "config_contract": config_contract,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": observed_checkpoint_sha256,
            "schema": checkpoint_schema,
        },
        "dataset": {
            "name": "Fisheye3DOD Main126 validation",
            "annotation_path": str(parent_annotation),
            "annotation_sha256": annotation_sha256,
            "ordered_sampler": "DefaultSampler(shuffle=False)",
        },
        "expected_sample_manifest": expected_manifest_record,
        "output_dir": str(output_dir),
        "device": args.device,
        "parameters": parameters,
        "physical_round_order": list(PHYSICAL_ROUND_ORDER),
        "schedules": schedules,
        "scopes": {
            "primary": {
                "name": C0_C32_FULL_DETECTOR_SCOPE,
                "call": "model.test_step(cpu_batch_clone)",
                "includes": [
                    "data preprocessing",
                    "host-to-device transfer",
                    "backbone",
                    "neck",
                    "head",
                    "prediction postprocess",
                ],
                "excludes": [
                    "batch deepcopy/preparation",
                    "evaluator",
                    "serialization",
                ],
            },
            "diagnostic": {
                "name": C0_C32_COMMON_MODEL_SCOPE,
                "call": "model._run_forward(common_processed_clone, mode='predict')",
                "includes": ["backbone", "neck", "head", "prediction postprocess"],
                "excludes": [
                    "batch deepcopy/preparation",
                    "data preprocessing",
                    "host-to-device transfer",
                    "evaluator",
                    "serialization",
                ],
                "speed_claim_eligible": False,
            },
        },
        "measurement": {
            "clock": "time.perf_counter_ns",
            "synchronization": "torch.cuda.synchronize immediately before and after call",
            "stream": "current stream must equal CUDA default stream",
            "precision": "uint8 CPU ingress; FP32 model math; autocast disabled",
            "pinned_ingress": (
                "captured and independently cloned full-detector images must "
                "remain pinned"
            ),
            "daf_supported_cuda_capabilities": [
                list(value) for value in DAF_SUPPORTED_CUDA_CAPABILITIES
            ],
            "paired_bootstrap": {
                "rng": "numpy.random.PCG64",
                "seed": BOOTSTRAP_SEED,
                "samples": BOOTSTRAP_SAMPLES,
                "stratified_by": "six physical rounds",
            },
        },
        "near_native_gate": {
            "median_ratio_ci95_upper_max": 1.10,
            "p95_ratio_max": 1.10,
            "peak_allocated_ratio_max": 1.02,
            "peak_reserved_ratio_max": 1.02,
            "primary_scope_only": C0_C32_FULL_DETECTOR_SCOPE,
        },
        "claim_status": "NO_SPEED_CLAIM_UNTIL_COMPLETE_IDLE_GPU_RUN",
    }
    plan["sha256"] = canonical_sha256(plan)
    return PreparedProtocol(
        args=args,
        parent_config=parent_config,
        candidate_config=candidate_config,
        checkpoint=checkpoint,
        state_dict=state_dict,
        expected_manifest=expected_manifest,
        detector_host_root=detector_host_root,
        official_root=official_root,
        plan=plan,
    )


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True) + "\n")
    temporary.replace(path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()],
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def _clone_batch(batch: Any) -> Any:
    return copy.deepcopy(batch)


def _clone_pinned_cpu_ingress_batch(batch: Mapping[str, Any]) -> Any:
    """Deep-copy a batch while retaining an independent pinned image.

    ``Tensor.__deepcopy__`` allocates ordinary CPU storage, so blindly using
    :func:`copy.deepcopy` silently removes the pinned-memory property that the
    full-detector timing scope is meant to measure.  Skip that one tensor in
    the generic deepcopy, then copy it into newly allocated pinned storage.
    This preparation remains outside the timed region.
    """

    _validate_cpu_ingress(batch, require_pinned=True, label="source")
    source_image = batch["inputs"]["img"]
    cloned = copy.deepcopy(batch, memo={id(source_image): source_image})
    cloned_image = torch.empty(
        source_image.shape,
        dtype=source_image.dtype,
        device="cpu",
        pin_memory=True,
    )
    cloned_image.copy_(source_image)
    cloned["inputs"]["img"] = cloned_image
    _validate_cpu_ingress(cloned, require_pinned=True, label="cloned")
    return cloned


def _reset_runtime_state(model: torch.nn.Module) -> None:
    bank = getattr(getattr(model, "head", None), "instance_bank", None)
    reset = getattr(bank, "reset", None)
    if callable(reset):
        reset()


def _runtime_state_snapshot(model: torch.nn.Module) -> dict[str, Any]:
    selected = {
        "cached_feature",
        "cached_anchor",
        "metas",
        "mask",
        "confidence",
        "temp_confidence",
        "instance_id",
    }
    captured = {}
    for module_name, module in model.named_modules():
        for name, value in vars(module).items():
            if name not in selected and not name.startswith("_last_"):
                continue
            key = f"{module_name}.{name}" if module_name else name
            captured[key] = raw_fingerprint(value)
    return captured


def _ephemeral_id_counter_contract(
    model: torch.nn.Module,
    *,
    expected_invocations: int,
) -> dict[str, Any]:
    """Attest the expected non-temporal ID counter separately from caches."""

    bank = getattr(getattr(model, "head", None), "instance_bank", None)
    if bank is None or getattr(bank, "num_temp_instances", None) != 0:
        raise VerificationContractError(
            "ephemeral ID attestation requires a zero-temporal instance bank"
        )
    num_anchor = getattr(bank, "num_anchor", None)
    if type(num_anchor) is not int or num_anchor <= 0:
        raise VerificationContractError("instance bank has an invalid anchor count")
    expected = expected_invocations * EXPECTED_BATCH_SIZE * num_anchor
    value = getattr(bank, "prev_id", None)
    if expected == 0:
        if type(value) is not int or value != 0:
            raise VerificationContractError(
                "reset non-temporal ID counter must be exact integer zero"
            )
        observed = value
        value_type = "int"
        value_device = "cpu"
    else:
        if (
            type(value) is not torch.Tensor
            or value.numel() != 1
            or value.dtype != torch.long
        ):
            raise VerificationContractError(
                "advanced non-temporal ID counter must be one int64 tensor"
            )
        observed = int(value.detach().cpu().item())
        value_type = "torch.Tensor"
        value_device = str(value.device)
    if observed != expected:
        raise VerificationContractError(
            f"non-temporal ID counter is {observed}, expected {expected}"
        )
    return {
        "expected_invocations": expected_invocations,
        "batch_size": EXPECTED_BATCH_SIZE,
        "num_anchor": num_anchor,
        "expected_value": expected,
        "observed_value": observed,
        "value_type": value_type,
        "value_device": value_device,
        "persistent_temporal_state": False,
    }


def _module_schema(model: torch.nn.Module) -> list[dict[str, str]]:
    return [
        {
            "path": path,
            "type": f"{type(module).__module__}.{type(module).__qualname__}",
        }
        for path, module in model.named_modules()
    ]


def _model_contract(model: torch.nn.Module) -> dict[str, Any]:
    parameters = [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "requires_grad": value.requires_grad,
            "numel": value.numel(),
        }
        for name, value in model.named_parameters()
    ]
    buffers = [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": value.numel(),
        }
        for name, value in model.named_buffers()
    ]
    return {
        "model_type": f"{type(model).__module__}.{type(model).__qualname__}",
        "head_type": (f"{type(model.head).__module__}.{type(model.head).__qualname__}"),
        "parameters": {
            "count": sum(item["numel"] for item in parameters),
            "trainable": sum(
                item["numel"] for item in parameters if item["requires_grad"]
            ),
            "schema_sha256": canonical_sha256(parameters),
        },
        "buffers": {
            "count": sum(item["numel"] for item in buffers),
            "schema_sha256": canonical_sha256(buffers),
        },
        "state_keys": list(model.state_dict()),
        "module_schema": _module_schema(model),
    }


def _normalize_module_schema(
    schema: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    normalized = copy.deepcopy(list(schema))
    head_types = {
        "rosa.integrations.sparse4d.ROSANativeBaselineHead",
        "rosa.integrations.sparse4d.ROSAOfficialResidualNativeBaselineHead",
    }
    for record in normalized:
        if record["path"] == "head" and record["type"] in head_types:
            record["type"] = "rosa.integrations.sparse4d.ROSANativeBaselineHead"
    return normalized


def _build_models(prepared: PreparedProtocol, device: torch.device):
    _reject_foreign_preimported_host(prepared.detector_host_root)
    _reject_foreign_preimported_rosa(SRC_ROOT)
    _reject_foreign_preimported_official(prepared.official_root)
    from mmengine.utils import import_modules_from_strings
    from mmdet3d.registry import MODELS
    from mmdet3d.utils import register_all_modules
    from rosa.integrations.sparse4d import (
        ROSANativeBaselineHead,
        ROSAOfficialResidualNativeBaselineHead,
    )

    register_all_modules(init_default_scope=True)
    import_modules_from_strings(**prepared.parent_config.custom_imports)
    import_modules_from_strings(**prepared.candidate_config.custom_imports)
    dataset_bridge = importlib.import_module(
        "rosa.integrations.fisheye3dod.dataset"
    )
    metric_bridge = importlib.import_module(
        "rosa.integrations.fisheye3dod.metric"
    )
    constants = importlib.import_module(
        "rosa.integrations.fisheye3dod.constants"
    )
    configured_official = Path(
        os.environ.get("FISHEYE3DOD_OFFICIAL_ROOT", "")
    ).resolve()
    if (
        configured_official != prepared.official_root
        or Path(dataset_bridge._official_root).resolve() != prepared.official_root
        or Path(metric_bridge.OFFICIAL_ROOT).resolve() != prepared.official_root
        or Path(constants.DEFAULT_OFFICIAL_REPO).resolve()
        != prepared.official_root
    ):
        raise VerificationContractError(
            "loaded Fisheye3DOD bridges do not use --official-root"
        )
    loaded_host_modules = _loaded_host_module_contract(
        prepared.detector_host_root
    )
    if loaded_host_modules["sha256"] != prepared.plan["detector_host"][
        "source_contract"
    ]["sha256"]:
        raise VerificationContractError(
            "loaded detector-host module hashes differ from the pinned source"
        )
    loaded_rosa_modules = _loaded_rosa_module_contract(SRC_ROOT)
    loaded_daf_extension = _loaded_daf_extension_contract(
        prepared.detector_host_root,
        expected_sha256=prepared.plan["detector_host"][
            "expected_daf_extension_sha256"
        ],
    )
    expected_classes = {
        PARENT_ARM: ROSANativeBaselineHead,
        CANDIDATE_ARM: ROSAOfficialResidualNativeBaselineHead,
    }
    models = {}
    contracts = {}
    for arm, config in (
        (PARENT_ARM, prepared.parent_config),
        (CANDIDATE_ARM, prepared.candidate_config),
    ):
        _seed_everything(3407)
        model = MODELS.build(copy.deepcopy(config.model))
        model.load_state_dict(prepared.state_dict, strict=True)
        if type(model.head) is not expected_classes[arm]:
            raise VerificationContractError(
                f"{arm} built unexpected exact head class {type(model.head)}"
            )
        contracts[arm] = _model_contract(model)
        model.to(device=device)
        model.eval()
        models[arm] = model

    parent_contract = contracts[PARENT_ARM]
    candidate_contract = contracts[CANDIDATE_ARM]
    for key in ("model_type", "parameters", "buffers", "state_keys"):
        if parent_contract[key] != candidate_contract[key]:
            raise VerificationContractError(
                f"C0/C32 runtime graph differs unexpectedly at {key}"
            )
    if _normalize_module_schema(parent_contract["module_schema"]) != (
        _normalize_module_schema(candidate_contract["module_schema"])
    ):
        raise VerificationContractError(
            "C0/C32 module graphs differ beyond the exact head class"
        )
    state_fingerprint = require_same_fingerprint(
        "same loaded checkpoint values",
        models[PARENT_ARM].state_dict(),
        models[CANDIDATE_ARM].state_dict(),
    )
    return (
        models,
        contracts,
        state_fingerprint,
        loaded_host_modules,
        loaded_rosa_modules,
        loaded_daf_extension,
    )


def _build_dataloader(prepared: PreparedProtocol):
    from mmengine.runner import Runner

    config = copy.deepcopy(prepared.parent_config.test_dataloader)
    config["num_workers"] = 0
    config["persistent_workers"] = False
    return Runner.build_dataloader(config, seed=3407, diff_rank_seed=False)


def _validate_cpu_ingress(
    batch: Mapping[str, Any],
    *,
    require_pinned: bool = False,
    label: str = "captured",
) -> None:
    if not isinstance(batch, Mapping) or not isinstance(batch.get("inputs"), Mapping):
        raise VerificationContractError("captured batch must contain inputs")
    image = batch["inputs"].get("img")
    expected_shape = (EXPECTED_BATCH_SIZE, 4, 3, 800, 800)
    if (
        not isinstance(image, torch.Tensor)
        or image.device.type != "cpu"
        or image.dtype != torch.uint8
        or tuple(image.shape) != expected_shape
        or not image.is_contiguous()
    ):
        raise VerificationContractError(
            f"{label} image must be contiguous CPU uint8 [4,4,3,800,800]"
        )
    if require_pinned and not image.is_pinned():
        raise VerificationContractError(
            f"{label} image must use pinned CPU storage"
        )


def _capture_batches(prepared: PreparedProtocol):
    iterator = iter(_build_dataloader(prepared))
    batches = []
    manifests = []
    seen = set()
    for batch_index in range(CAPTURE_BATCHES):
        try:
            batch = next(iterator)
        except StopIteration as error:
            raise VerificationContractError(
                "Main126 dataloader ended before two batches were captured"
            ) from error
        _validate_cpu_ingress(batch, require_pinned=True, label="captured")
        manifest = build_batch_manifest(
            batch,
            batch_index=batch_index,
            expected_batch_size=EXPECTED_BATCH_SIZE,
        )
        semantic = semantic_sample_digest(
            batch, expected_batch_size=EXPECTED_BATCH_SIZE
        )
        for identity in semantic["sample_identities"]:
            digest = canonical_sha256(identity)
            if digest in seen:
                raise VerificationContractError(
                    "captured Main126 sample identity repeats"
                )
            seen.add(digest)
        manifest["semantic_sha256"] = semantic["sha256"]
        batches.append(batch)
        manifests.append(manifest)
    document = {
        "schema": "rosa.c0_c32_main126_batches.v1",
        "ordered": True,
        "batches": manifests,
    }
    document["sha256"] = canonical_sha256(document["batches"])
    if prepared.expected_manifest is not None:
        validate_expected_manifest(document, prepared.expected_manifest)
    return batches, document


def _require_default_stream(device: torch.device) -> dict[str, Any]:
    current = torch.cuda.current_stream(device)
    default = torch.cuda.default_stream(device)
    current_id = int(current.cuda_stream)
    default_id = int(default.cuda_stream)
    if current_id != default_id:
        raise VerificationContractError(
            "C0/C32 timing requires the CUDA default stream"
        )
    return {
        "policy": "current CUDA stream must equal default stream",
        "current_stream": current_id,
        "default_stream": default_id,
    }


@torch.inference_mode()
def _build_common_boundary(
    models: Mapping[str, torch.nn.Module],
    cpu_batches: Sequence[Any],
    manifests: Mapping[str, Any],
):
    common = []
    records = []
    for batch_index, source in enumerate(cpu_batches):
        source_before = raw_fingerprint(source)
        outputs = {}
        for arm in (PARENT_ARM, CANDIDATE_ARM):
            outputs[arm] = models[arm].data_preprocessor(
                _clone_pinned_cpu_ingress_batch(source), training=False
            )
        boundary = require_same_fingerprint(
            f"common preprocessor boundary batch {batch_index}",
            outputs[PARENT_ARM],
            outputs[CANDIDATE_ARM],
        )
        source_after = raw_fingerprint(source)
        if source_before != source_after:
            raise VerificationContractError(
                "data preprocessor mutated the sealed CPU source batch"
            )
        common.append(outputs[PARENT_ARM])
        records.append(
            {
                "batch_index": batch_index,
                "semantic_sha256": manifests["batches"][batch_index]["semantic_sha256"],
                "boundary": boundary,
            }
        )
        del outputs[CANDIDATE_ARM]
    return common, records


def _prediction_tensor(sample: Any, name: str) -> torch.Tensor:
    predictions = getattr(sample, "pred_instances_3d", None)
    if predictions is None:
        raise VerificationContractError("prediction sample lacks pred_instances_3d")
    value = getattr(predictions, name, None)
    if name == "bboxes_3d" and hasattr(value, "tensor"):
        value = value.tensor
    if not isinstance(value, torch.Tensor):
        raise VerificationContractError(f"prediction lacks tensor field {name}")
    return value


def _validate_detection_output(output: Any) -> dict[str, Any]:
    if not isinstance(output, Sequence) or isinstance(output, (str, bytes)):
        raise VerificationContractError("predict output must be a sample sequence")
    if len(output) != EXPECTED_BATCH_SIZE:
        raise VerificationContractError(
            f"predict output has {len(output)} samples, expected {EXPECTED_BATCH_SIZE}"
        )
    counts = []
    for sample in output:
        boxes = _prediction_tensor(sample, "bboxes_3d")
        scores = _prediction_tensor(sample, "scores_3d")
        labels = _prediction_tensor(sample, "labels_3d")
        count = int(boxes.shape[0])
        if scores.ndim != 1 or labels.ndim != 1:
            raise VerificationContractError("scores and labels must be vectors")
        if scores.shape[0] != count or labels.shape[0] != count:
            raise VerificationContractError(
                "prediction box, score, and label counts must agree"
            )
        if not bool(torch.isfinite(boxes).all()) or not bool(
            torch.isfinite(scores).all()
        ):
            raise VerificationContractError("prediction contains NaN or Inf")
        if labels.is_floating_point() or labels.is_complex():
            raise VerificationContractError("prediction labels must be integral")
        if count and (
            int(labels.min()) < 0 or int(labels.max()) >= len(EXPECTED_CLASSES)
        ):
            raise VerificationContractError("prediction label is outside six classes")
        counts.append(count)
    return {
        "sample_count": len(output),
        "detections_per_sample": counts,
        "finite_schema": finite_output_schema(output),
        "raw_fingerprint": raw_fingerprint(output),
    }


@torch.inference_mode()
def _attest_native_runtime_shortcuts(
    models: Mapping[str, torch.nn.Module],
    source_batch: Mapping[str, Any],
) -> dict[str, Any]:
    """Count required shortcut hits in an untimed, fully restored preflight."""

    host_detector = importlib.import_module(
        "projects.mmdet3d_plugin.models.sparse4d"
    )
    instance_module = importlib.import_module(
        "projects.mmdet3d_plugin.models.instance_bank"
    )
    decoder_module = importlib.import_module(
        "projects.mmdet3d_plugin.models.detection3d.decoder"
    )
    rosa_sparse4d = importlib.import_module("rosa.integrations.sparse4d")
    originals = {
        "feature_maps_format": host_detector.feature_maps_format,
        "fresh_id_predicate": instance_module._fresh_non_temporal_instance_id_shape,
        "dense_decoder_predicate": decoder_module._can_batch_dense_legacy_decode,
        "ocam_project_points": rosa_sparse4d.ocam_project_points,
        "inplace_residual_predicate": (
            rosa_sparse4d._can_use_inplace_official_residual
        ),
    }
    records = {}
    for arm in (PARENT_ARM, CANDIDATE_ARM):
        counts = {
            "feature_maps_format_calls": 0,
            "int32_metadata_hits": 0,
            "fresh_id_predicate_calls": 0,
            "fresh_id_shortcut_hits": 0,
            "dense_decoder_predicate_calls": 0,
            "dense_decoder_shortcut_hits": 0,
            "ocam_projection_calls": 0,
            "exact_homogeneous_storage_reuse_hits": 0,
            "inplace_residual_predicate_calls": 0,
            "inplace_residual_hits": 0,
        }

        def feature_maps_format(*args, **kwargs):
            result = originals["feature_maps_format"](*args, **kwargs)
            counts["feature_maps_format_calls"] += 1
            if (
                isinstance(result, (list, tuple))
                and len(result) == 3
                and type(result[1]) is torch.Tensor
                and type(result[2]) is torch.Tensor
                and result[1].dtype == torch.int32
                and result[2].dtype == torch.int32
            ):
                counts["int32_metadata_hits"] += 1
            return result

        def fresh_id_predicate(*args, **kwargs):
            result = originals["fresh_id_predicate"](*args, **kwargs)
            counts["fresh_id_predicate_calls"] += 1
            if result is not None:
                counts["fresh_id_shortcut_hits"] += 1
            return result

        def dense_decoder_predicate(*args, **kwargs):
            result = originals["dense_decoder_predicate"](*args, **kwargs)
            counts["dense_decoder_predicate_calls"] += 1
            if result is True:
                counts["dense_decoder_shortcut_hits"] += 1
            return result

        def ocam_project_points(*args, **kwargs):
            counts["ocam_projection_calls"] += 1
            if (
                kwargs.get("contract") == "full_fov"
                and kwargs.get("camera_transform") == "homogeneous_4x4"
                and kwargs.get("inference_storage_reuse") is True
            ):
                counts["exact_homogeneous_storage_reuse_hits"] += 1
            return originals["ocam_project_points"](*args, **kwargs)

        def inplace_residual_predicate(*args, **kwargs):
            result = originals["inplace_residual_predicate"](*args, **kwargs)
            counts["inplace_residual_predicate_calls"] += 1
            if result is True:
                counts["inplace_residual_hits"] += 1
            return result

        _reset_runtime_state(models[arm])
        persistent_before = _runtime_state_snapshot(models[arm])
        source_before = raw_fingerprint(source_batch)
        output_contracts = []
        try:
            host_detector.feature_maps_format = feature_maps_format
            instance_module._fresh_non_temporal_instance_id_shape = (
                fresh_id_predicate
            )
            decoder_module._can_batch_dense_legacy_decode = (
                dense_decoder_predicate
            )
            rosa_sparse4d.ocam_project_points = ocam_project_points
            rosa_sparse4d._can_use_inplace_official_residual = (
                inplace_residual_predicate
            )
            for invocation in range(2):
                output = models[arm].test_step(
                    _clone_pinned_cpu_ingress_batch(source_batch)
                )
                output_contract = _validate_detection_output(output)
                output_contract["invocation"] = invocation
                output_contracts.append(output_contract)
                del output
        finally:
            host_detector.feature_maps_format = originals["feature_maps_format"]
            instance_module._fresh_non_temporal_instance_id_shape = originals[
                "fresh_id_predicate"
            ]
            decoder_module._can_batch_dense_legacy_decode = originals[
                "dense_decoder_predicate"
            ]
            rosa_sparse4d.ocam_project_points = originals["ocam_project_points"]
            rosa_sparse4d._can_use_inplace_official_residual = originals[
                "inplace_residual_predicate"
            ]

        if (
            host_detector.feature_maps_format
            is not originals["feature_maps_format"]
            or instance_module._fresh_non_temporal_instance_id_shape
            is not originals["fresh_id_predicate"]
            or decoder_module._can_batch_dense_legacy_decode
            is not originals["dense_decoder_predicate"]
            or rosa_sparse4d.ocam_project_points
            is not originals["ocam_project_points"]
            or rosa_sparse4d._can_use_inplace_official_residual
            is not originals["inplace_residual_predicate"]
        ):
            raise VerificationContractError(
                "runtime shortcut instrumentation was not exactly restored"
            )
        expected_twice = (
            "feature_maps_format_calls",
            "int32_metadata_hits",
            "fresh_id_predicate_calls",
            "fresh_id_shortcut_hits",
            "dense_decoder_predicate_calls",
            "dense_decoder_shortcut_hits",
        )
        if any(counts[name] != 2 for name in expected_twice):
            raise VerificationContractError(
                f"{arm} did not hit every required native runtime shortcut: {counts}"
            )
        if (
            counts["ocam_projection_calls"] != 12
            or counts["exact_homogeneous_storage_reuse_hits"] != 12
        ):
            raise VerificationContractError(
                f"{arm} did not execute twelve exact OCam reuse projections: {counts}"
            )
        expected_inplace_hits = 0 if arm == PARENT_ARM else 10
        if (
            counts["inplace_residual_predicate_calls"] != expected_inplace_hits
            or counts["inplace_residual_hits"] != expected_inplace_hits
        ):
            raise VerificationContractError(
                f"{arm} in-place residual receipt is invalid: {counts}"
            )
        if persistent_before != _runtime_state_snapshot(models[arm]):
            raise VerificationContractError(
                f"{arm} shortcut preflight mutated persistent temporal state"
            )
        id_counter = _ephemeral_id_counter_contract(
            models[arm],
            expected_invocations=2,
        )
        if source_before != raw_fingerprint(source_batch):
            raise VerificationContractError(
                f"{arm} shortcut preflight mutated the sealed source batch"
            )
        records[arm] = {
            "counts": counts,
            "ephemeral_id_counter": id_counter,
            "outputs": output_contracts,
        }
    return {
        "timed_code_instrumented": False,
        "instrumentation_restored_before_timing": True,
        "arms": records,
    }


@torch.inference_mode()
def _preflight_outputs_and_state(
    models: Mapping[str, torch.nn.Module],
    cpu_batches: Sequence[Any],
    common_batches: Sequence[Any],
) -> dict[str, Any]:
    frozen_state = require_same_fingerprint(
        "preflight model state",
        models[PARENT_ARM].state_dict(),
        models[CANDIDATE_ARM].state_dict(),
    )
    initial_rng = _capture_rng()
    records = {PARENT_ARM: [], CANDIDATE_ARM: []}
    id_counter_before = {}
    id_counter_after = {}
    for arm in (PARENT_ARM, CANDIDATE_ARM):
        model = models[arm]
        _reset_runtime_state(model)
        runtime_before = _runtime_state_snapshot(model)
        id_counter_before[arm] = _ephemeral_id_counter_contract(
            model, expected_invocations=0
        )
        _restore_rng(initial_rng)
        for batch_index, batch in enumerate(cpu_batches):
            output = model.test_step(_clone_pinned_cpu_ingress_batch(batch))
            record = _validate_detection_output(output)
            record.update(
                {"batch_index": batch_index, "scope": C0_C32_FULL_DETECTOR_SCOPE}
            )
            records[arm].append(record)
            del output
        for batch_index, batch in enumerate(common_batches):
            output = model._run_forward(_clone_batch(batch), mode="predict")
            record = _validate_detection_output(output)
            record.update(
                {"batch_index": batch_index, "scope": C0_C32_COMMON_MODEL_SCOPE}
            )
            records[arm].append(record)
            del output
        rng_after = _capture_rng()
        require_same_fingerprint(
            f"{arm} preflight RNG stability", initial_rng, rng_after
        )
        if runtime_before != _runtime_state_snapshot(model):
            raise VerificationContractError(
                f"{arm} inference unexpectedly mutated temporal runtime state"
            )
        id_counter_after[arm] = _ephemeral_id_counter_contract(
            model,
            expected_invocations=len(cpu_batches) + len(common_batches),
        )
        if raw_fingerprint(model.state_dict()) != frozen_state:
            raise VerificationContractError(
                f"{arm} inference unexpectedly mutated model state"
            )
    _restore_rng(initial_rng)
    return {
        "outputs_are_allowed_to_differ": True,
        "model_state_before": frozen_state,
        "initial_rng": raw_fingerprint(initial_rng),
        "ephemeral_id_counter": {
            "before": id_counter_before,
            "after": id_counter_after,
        },
        "arms": records,
    }


def _nvidia_smi_snapshot(device_index: int) -> dict[str, Any]:
    fields = (
        "index",
        "uuid",
        "name",
        "driver_version",
        "pstate",
        "temperature.gpu",
        "power.draw",
        "power.limit",
        "clocks.current.graphics",
        "clocks.current.sm",
        "clocks.current.memory",
        "memory.used",
        "memory.total",
        "utilization.gpu",
        "compute_mode",
    )
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={device_index}",
            "--query-gpu=" + ",".join(fields),
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise VerificationContractError("nvidia-smi did not return one selected GPU")
    app_fields = ("pid", "process_name", "used_gpu_memory", "gpu_uuid")
    apps = subprocess.run(
        [
            "nvidia-smi",
            f"--id={device_index}",
            "--query-compute-apps=" + ",".join(app_fields),
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    app_rows = [line.strip() for line in apps.stdout.splitlines() if line.strip()]
    foreign = []
    for row in app_rows:
        try:
            pid = int(row.split(",", 1)[0].strip())
        except ValueError as error:
            raise VerificationContractError(
                f"could not parse nvidia-smi compute row: {row}"
            ) from error
        if pid != os.getpid():
            foreign.append(row)
    if foreign:
        raise VerificationContractError(
            "selected GPU has a foreign compute process; run remains disallowed"
        )
    return {
        "gpu_fields": list(fields),
        "gpu_row": rows[0],
        "compute_app_fields": list(app_fields),
        "compute_app_rows": app_rows,
        "current_pid": os.getpid(),
        "foreign_compute_processes": 0,
        "wall_time_ns": time.time_ns(),
    }


@torch.inference_mode()
def _memory_probe(
    model: torch.nn.Module,
    source_batch: Any,
    device: torch.device,
) -> dict[str, int]:
    _reset_runtime_state(model)
    gc.collect()
    torch.cuda.empty_cache()
    batch = _clone_pinned_cpu_ingress_batch(source_batch)
    torch.cuda.synchronize(device)
    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    output = model.test_step(batch)
    torch.cuda.synchronize(device)
    sample_count = len(output)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    allocated_with_output = torch.cuda.memory_allocated(device)
    reserved_with_output = torch.cuda.memory_reserved(device)
    if sample_count != EXPECTED_BATCH_SIZE:
        raise VerificationContractError("memory probe returned wrong sample count")
    del output, batch
    torch.cuda.synchronize(device)
    return {
        "allocated_before_bytes": allocated_before,
        "reserved_before_bytes": reserved_before,
        "allocated_with_output_bytes": allocated_with_output,
        "reserved_with_output_bytes": reserved_with_output,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_increment_bytes": max(0, peak_allocated - allocated_before),
        "peak_reserved_increment_bytes": max(0, peak_reserved - reserved_before),
        "sample_count": sample_count,
    }


def _measure_memory(
    models: Mapping[str, torch.nn.Module],
    source_batch: Any,
    device: torch.device,
) -> dict[str, Any]:
    records = {PARENT_ARM: [], CANDIDATE_ARM: []}
    for pattern in ("PC", "CP", "PC"):
        for token in pattern:
            arm = PARENT_ARM if token == "P" else CANDIDATE_ARM
            records[arm].append(_memory_probe(models[arm], source_batch, device))
    summary = {}
    for arm in (PARENT_ARM, CANDIDATE_ARM):
        summary[arm] = {
            "peak_allocated_bytes": max(
                record["peak_allocated_bytes"] for record in records[arm]
            ),
            "peak_reserved_bytes": max(
                record["peak_reserved_bytes"] for record in records[arm]
            ),
            "probe_records": records[arm],
            "aggregation": "maximum of three interleaved probes",
        }
    return summary


@torch.inference_mode()
def _timed_segment(
    entries,
    *,
    models: Mapping[str, torch.nn.Module],
    sources: Sequence[Any],
    manifests: Mapping[str, Any],
    scope: str,
    device: torch.device,
    sequence_offset: int,
) -> list[dict[str, Any]]:
    records = []
    stream = _require_default_stream(device)
    for local_index, entry in enumerate(entries):
        source = sources[entry.batch_index]
        if scope == C0_C32_FULL_DETECTOR_SCOPE:
            batch = _clone_pinned_cpu_ingress_batch(source)
        else:
            batch = _clone_batch(source)
        torch.cuda.synchronize(device)
        start_ns = time.perf_counter_ns()
        if scope == C0_C32_FULL_DETECTOR_SCOPE:
            output = models[entry.arm].test_step(batch)
        elif scope == C0_C32_COMMON_MODEL_SCOPE:
            output = models[entry.arm]._run_forward(batch, mode="predict")
        else:
            raise VerificationContractError(f"unsupported timing scope: {scope}")
        torch.cuda.synchronize(device)
        elapsed_ns = time.perf_counter_ns() - start_ns
        if len(output) != EXPECTED_BATCH_SIZE:
            raise VerificationContractError("timed predict returned wrong sample count")
        record = entry.to_dict()
        record.update(
            {
                "sequence_index": sequence_offset + local_index,
                "benchmark_scope": scope,
                "semantic_batch_sha256": manifests["batches"][entry.batch_index][
                    "semantic_sha256"
                ],
                "source_batch_sha256": manifests["batches"][entry.batch_index][
                    "sha256"
                ],
                "elapsed_ns": elapsed_ns,
                "sample_count": len(output),
                "stream": stream,
            }
        )
        records.append(record)
        del output, batch
    return records


def _run_scope(
    *,
    scope: str,
    models: Mapping[str, torch.nn.Module],
    sources: Sequence[Any],
    manifests: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    warmup_schedule = build_alternating_schedule(
        iterations_per_arm=DEFAULT_WARMUP_ITERATIONS_PER_ARM,
        batch_count=CAPTURE_BATCHES,
        phase="warmup",
    )
    measured_schedule = build_alternating_schedule(
        iterations_per_arm=DEFAULT_MEASURED_ITERATIONS_PER_ARM,
        batch_count=CAPTURE_BATCHES,
        phase="measured",
    )
    source_before = [raw_fingerprint(batch) for batch in sources]
    state_before = {
        arm: raw_fingerprint(model.state_dict()) for arm, model in models.items()
    }
    rng_before = _capture_rng()
    warmup_records = []
    measured_records = []
    round_state = []
    for round_index, pattern in enumerate(PHYSICAL_ROUND_ORDER):
        runtime_before = {}
        arm_invocations = {PARENT_ARM: 0, CANDIDATE_ARM: 0}
        for arm, model in models.items():
            _reset_runtime_state(model)
            runtime_before[arm] = _runtime_state_snapshot(model)
            _ephemeral_id_counter_contract(model, expected_invocations=0)
        for phase, schedule, destination in (
            ("warmup", warmup_schedule, warmup_records),
            ("measured", measured_schedule, measured_records),
        ):
            del phase
            for slot_index, token in enumerate(pattern):
                expected_arm = PARENT_ARM if token == "P" else CANDIDATE_ARM
                segment = tuple(
                    entry
                    for entry in schedule
                    if entry.round_index == round_index
                    and entry.slot_index == slot_index
                )
                if not segment or any(
                    entry.arm != expected_arm or entry.pattern != pattern
                    for entry in segment
                ):
                    raise VerificationContractError(
                        "timing schedule violates physical PCCP/CPPC order"
                    )
                destination.extend(
                    _timed_segment(
                        segment,
                        models=models,
                        sources=sources,
                        manifests=manifests,
                        scope=scope,
                        device=device,
                        sequence_offset=len(destination),
                    )
                )
                arm_invocations[expected_arm] += len(segment)
        runtime_after = {
            arm: _runtime_state_snapshot(model) for arm, model in models.items()
        }
        if runtime_before != runtime_after:
            raise VerificationContractError(
                f"{scope} unexpectedly mutated temporal runtime state"
            )
        id_counters = {
            arm: _ephemeral_id_counter_contract(
                model,
                expected_invocations=arm_invocations[arm],
            )
            for arm, model in models.items()
        }
        round_state.append(
            {
                "round_index": round_index,
                "pattern": pattern,
                "runtime_state_sha256": canonical_sha256(runtime_after),
                "ephemeral_id_counter": id_counters,
            }
        )
    rng_after = _capture_rng()
    require_same_fingerprint(f"{scope} RNG stability", rng_before, rng_after)
    source_after = [raw_fingerprint(batch) for batch in sources]
    if source_before != source_after:
        raise VerificationContractError(f"{scope} mutated a sealed source batch")
    state_after = {
        arm: raw_fingerprint(model.state_dict()) for arm, model in models.items()
    }
    if state_before != state_after:
        raise VerificationContractError(f"{scope} mutated model state_dict")
    if len(warmup_records) != 2 * DEFAULT_WARMUP_ITERATIONS_PER_ARM:
        raise VerificationContractError("warmup invocation count drifted")
    if len(measured_records) != 2 * DEFAULT_MEASURED_ITERATIONS_PER_ARM:
        raise VerificationContractError("measured invocation count drifted")
    paired = paired_bootstrap_confidence_intervals(
        measured_records,
        expected_pairs=DEFAULT_MEASURED_ITERATIONS_PER_ARM,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
        required_pair_fields=(
            "benchmark_scope",
            "semantic_batch_sha256",
            "source_batch_sha256",
        ),
    )
    statistics = {
        arm: summarize_timings(measured_records, arm=arm)
        for arm in (PARENT_ARM, CANDIDATE_ARM)
    }
    return {
        "scope": scope,
        "warmup_records": warmup_records,
        "measured_records": measured_records,
        "paired_bootstrap": paired,
        "statistics": statistics,
        "state_and_rng": {
            "source_fingerprints": source_before,
            "model_state": state_before,
            "rng": raw_fingerprint(rng_before),
            "rounds": round_state,
        },
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment(device: torch.device, snapshots: Sequence[Mapping[str, Any]]):
    properties = torch.cuda.get_device_properties(device)
    clock = time.get_clock_info("perf_counter")
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            name: _package_version(name)
            for name in ("torch", "numpy", "mmengine", "mmcv", "mmdet", "mmdet3d")
        },
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": {
            "index": device.index,
            "name": properties.name,
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": properties.total_memory,
        },
        "precision": {
            "autocast_enabled": torch.is_autocast_enabled(),
            "default_dtype": str(torch.get_default_dtype()),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
        "determinism": {
            "algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "perf_counter": {
            "implementation": clock.implementation,
            "monotonic": clock.monotonic,
            "adjustable": clock.adjustable,
            "resolution_seconds": clock.resolution,
        },
        "default_stream": _require_default_stream(device),
        "nvidia_smi_snapshots": list(snapshots),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "ROSA_DETECTOR_HOST_ROOT",
                "SPARSE4D_ROOT",
                "FISHEYE3DOD_OFFICIAL_ROOT",
                "FISHEYE3DOD_DATA_ROOT",
                "FISHEYE3DOD_ANN_ROOT",
            )
        },
    }


def _write_report(path: Path, summary: Mapping[str, Any]) -> None:
    gate = summary["near_native_gate"]
    full = summary["scopes"][C0_C32_FULL_DETECTOR_SCOPE]
    model = summary["scopes"][C0_C32_COMMON_MODEL_SCOPE]
    lines = [
        "# C0 vs C32 near-native CUDA runtime report",
        "",
        f"Status: **{summary['status']}**",
        "",
        "The primary result is the measured full `model.test_step` scope. "
        "Batch cloning/preparation, evaluation, and serialization are excluded.",
        "",
        f"Near-native decision: **{gate['decision']}**",
        "",
        f"Full-scope C32/C0 median ratio: "
        f"{full['paired_bootstrap']['metrics']['median_candidate_over_parent']['point_estimate']:.6f}",
        "",
        f"Full-scope C32/C0 p95 ratio: "
        f"{full['paired_bootstrap']['metrics']['p95_candidate_over_parent']['point_estimate']:.6f}",
        "",
        f"Diagnostic common-boundary median ratio: "
        f"{model['paired_bootstrap']['metrics']['median_candidate_over_parent']['point_estimate']:.6f}",
        "",
        "No component sums were synthesized. The common-boundary scope is "
        "diagnostic only and cannot independently authorize a speed claim.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _seal_directory(output_dir: Path) -> None:
    files = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [f"{file_sha256(path)}  {path.name}" for path in files]
    (output_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _acquire_lock(output_parent: Path, device_index: int) -> tuple[int, Path]:
    lock = output_parent / f".c0_c32_near_native_cuda{device_index}.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise VerificationContractError(
            f"exclusive runtime lock already exists: {lock}"
        ) from error
    os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
    os.fsync(descriptor)
    return descriptor, lock


def execute_protocol(prepared: PreparedProtocol) -> dict[str, Any]:
    """Run the CUDA protocol only after every explicit authorization check."""

    args = prepared.args
    if (
        args.execution_state != "ARMED"
        or args.idle_reserved_token != IDLE_RESERVED_TOKEN
    ):
        raise VerificationContractError(
            "execution requires exact ARMED state and idle/reserved token"
        )
    repository = _git_snapshot(require_clean=True)
    if repository["revision"] != args.expected_source_revision:
        raise VerificationContractError("source revision changed after dry-run")
    rosa_sources = _python_source_contract(SRC_ROOT, label="ROSA source root")
    if rosa_sources != prepared.plan["rosa_source"]["source_contract"]:
        raise VerificationContractError("ROSA source changed after dry-run")
    detector_host = _git_snapshot_at(
        prepared.detector_host_root,
        require_clean=True,
    )
    detector_host_sources = _host_source_contract(prepared.detector_host_root)
    detector_host_daf_sources = _host_daf_source_contract(
        prepared.detector_host_root
    )
    daf_binary = _daf_binary_file_contract(
        prepared.detector_host_root,
        expected_sha256=DEFAULT_DAF_EXTENSION_SHA256,
    )
    if (
        detector_host != prepared.plan["detector_host"]["git"]
        or detector_host_sources
        != prepared.plan["detector_host"]["source_contract"]
        or detector_host_daf_sources
        != prepared.plan["detector_host"]["daf_source_contract"]
        or daf_binary
        != prepared.plan["detector_host"]["daf_binary_contract"]
    ):
        raise VerificationContractError(
            "detector-host source or DAF binary changed after dry-run"
        )
    official_sources = _python_source_contract(
        prepared.official_root,
        label="official Fisheye3DOD root",
    )
    official_git = _git_snapshot_at(
        prepared.official_root,
        require_clean=False,
    )
    if (
        official_sources
        != prepared.plan["official_fisheye3dod"]["source_contract"]
        or official_git != prepared.plan["official_fisheye3dod"]["git"]
    ):
        raise VerificationContractError(
            "official Fisheye3DOD source changed after dry-run"
        )
    output_dir = Path(prepared.plan["output_dir"])
    if output_dir.exists():
        raise VerificationContractError(
            f"refusing to overwrite runtime output: {output_dir}"
        )
    device_index = int(args.device.split(":", 1)[1])
    lock_fd, lock_path = _acquire_lock(output_dir.parent, device_index)
    try:
        output_dir.mkdir(parents=False, exist_ok=False)
        armed_plan = copy.deepcopy(prepared.plan)
        armed_plan.update(
            {
                "status": "ARMED",
                "repository": repository,
                "exclusive_lock": str(lock_path),
                "output_created": True,
            }
        )
        _atomic_json(output_dir / "PLAN.json", armed_plan)
        existing_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if existing_workspace not in (None, ":4096:8"):
            raise VerificationContractError(
                "CUBLAS_WORKSPACE_CONFIG conflicts with deterministic protocol"
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        armed_plan["cuda_queried"] = True
        _atomic_json(output_dir / "PLAN.json", armed_plan)
        snapshots = [_nvidia_smi_snapshot(device_index)]
        if not torch.cuda.is_available():
            raise VerificationContractError("armed run requires CUDA")
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        capability = tuple(torch.cuda.get_device_capability(device))
        if capability not in DAF_SUPPORTED_CUDA_CAPABILITIES:
            raise VerificationContractError(
                f"DAF extension does not contain code for CUDA capability {capability}"
            )
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        _seed_everything(3407)
        _require_default_stream(device)

        (
            models,
            model_contracts,
            checkpoint_state,
            loaded_host_modules,
            loaded_rosa_modules,
            loaded_daf_extension,
        ) = _build_models(prepared, device)
        cpu_batches, manifests = _capture_batches(prepared)
        loaded_official_modules = _loaded_official_module_contract(
            prepared.official_root
        )
        _atomic_json(output_dir / "SAMPLE_MANIFEST.json", manifests)
        shortcut_preflight = _attest_native_runtime_shortcuts(
            models,
            cpu_batches[0],
        )
        _atomic_json(
            output_dir / "NATIVE_SHORTCUT_PREFLIGHT.json",
            shortcut_preflight,
        )
        common_batches, boundary = _build_common_boundary(
            models, cpu_batches, manifests
        )
        _atomic_json(output_dir / "COMMON_BOUNDARY.json", boundary)
        preflight = _preflight_outputs_and_state(models, cpu_batches, common_batches)
        _atomic_json(output_dir / "OUTPUT_STATE_PREFLIGHT.json", preflight)
        loaded_rosa_modules = _loaded_rosa_module_contract(SRC_ROOT)
        loaded_official_modules = _loaded_official_module_contract(
            prepared.official_root
        )
        _atomic_json(
            output_dir / "SOURCE_MODULE_PREFLIGHT.json",
            {
                "rosa": loaded_rosa_modules,
                "official_fisheye3dod": loaded_official_modules,
            },
        )

        snapshots.append(_nvidia_smi_snapshot(device_index))
        memory = _measure_memory(models, cpu_batches[0], device)
        _atomic_json(output_dir / "FULL_SCOPE_MEMORY.json", memory)

        snapshots.append(_nvidia_smi_snapshot(device_index))
        full = _run_scope(
            scope=C0_C32_FULL_DETECTOR_SCOPE,
            models=models,
            sources=cpu_batches,
            manifests=manifests,
            device=device,
        )
        _write_jsonl(output_dir / "FULL_WARMUP.jsonl", full.pop("warmup_records"))
        _write_jsonl(output_dir / "FULL_TIMINGS.jsonl", full.pop("measured_records"))
        _atomic_json(output_dir / "FULL_STATISTICS.json", full)

        snapshots.append(_nvidia_smi_snapshot(device_index))
        diagnostic = _run_scope(
            scope=C0_C32_COMMON_MODEL_SCOPE,
            models=models,
            sources=common_batches,
            manifests=manifests,
            device=device,
        )
        _write_jsonl(
            output_dir / "MODEL_ONLY_WARMUP.jsonl",
            diagnostic.pop("warmup_records"),
        )
        _write_jsonl(
            output_dir / "MODEL_ONLY_TIMINGS.jsonl",
            diagnostic.pop("measured_records"),
        )
        _atomic_json(output_dir / "MODEL_ONLY_STATISTICS.json", diagnostic)
        snapshots.append(_nvidia_smi_snapshot(device_index))

        final_repository = _git_snapshot(require_clean=True)
        final_frozen_configs = _frozen_config_contract()
        if (
            final_repository != repository
            or final_frozen_configs != prepared.plan["frozen_config_files"]
        ):
            raise VerificationContractError(
                "ROSA repository or frozen configs changed during timing"
            )
        final_detector_host = _git_snapshot_at(
            prepared.detector_host_root,
            require_clean=True,
        )
        final_detector_sources = _host_source_contract(
            prepared.detector_host_root
        )
        final_detector_daf_sources = _host_daf_source_contract(
            prepared.detector_host_root
        )
        final_rosa_sources = _python_source_contract(
            SRC_ROOT,
            label="ROSA source root",
        )
        final_official_sources = _python_source_contract(
            prepared.official_root,
            label="official Fisheye3DOD root",
        )
        final_official_git = _git_snapshot_at(
            prepared.official_root,
            require_clean=False,
        )
        if (
            final_detector_host != detector_host
            or final_detector_sources != detector_host_sources
            or final_detector_daf_sources != detector_host_daf_sources
            or _loaded_host_module_contract(prepared.detector_host_root)
            != loaded_host_modules
            or _loaded_daf_extension_contract(
                prepared.detector_host_root,
                expected_sha256=prepared.plan["detector_host"][
                    "expected_daf_extension_sha256"
                ],
            )
            != loaded_daf_extension
        ):
            raise VerificationContractError(
                "detector-host source or imported module closure changed during timing"
            )
        if (
            final_rosa_sources != rosa_sources
            or _loaded_rosa_module_contract(SRC_ROOT) != loaded_rosa_modules
        ):
            raise VerificationContractError(
                "ROSA source or imported module closure changed during timing"
            )
        if (
            final_official_sources != official_sources
            or final_official_git != official_git
            or _loaded_official_module_contract(prepared.official_root)
            != loaded_official_modules
        ):
            raise VerificationContractError(
                "official Fisheye3DOD source or module closure changed during timing"
            )

        near_native = evaluate_c0_c32_near_native_gate(full["paired_bootstrap"], memory)
        diagnostic_gate = evaluate_c32_near_native_gate(diagnostic["paired_bootstrap"])
        diagnostic_gate.update(
            {
                "benchmark_scope": C0_C32_COMMON_MODEL_SCOPE,
                "speed_claim_eligible": False,
                "decision_scope": "DIAGNOSTIC_ONLY",
            }
        )
        environment = _environment(device, snapshots)
        _atomic_json(output_dir / "ENVIRONMENT.json", environment)
        summary = {
            "schema": "rosa.c0_c32_near_native_runtime_evidence.v1",
            "status": "COMPLETE",
            "protocol": C0_C32_NEAR_NATIVE_PROTOCOL,
            "repository": final_repository,
            "rosa_source": {
                "source_contract": final_rosa_sources,
                "loaded_modules": loaded_rosa_modules,
            },
            "detector_host": {
                "git": final_detector_host,
                "source_contract": final_detector_sources,
                "daf_source_contract": final_detector_daf_sources,
                "loaded_daf_extension": loaded_daf_extension,
                "loaded_modules": loaded_host_modules,
            },
            "official_fisheye3dod": {
                "identity": "pinned locally patched official source",
                "git": final_official_git,
                "source_contract": final_official_sources,
                "loaded_modules": loaded_official_modules,
            },
            "plan_sha256": file_sha256(output_dir / "PLAN.json"),
            "checkpoint": prepared.plan["checkpoint"],
            "dataset": prepared.plan["dataset"],
            "sample_manifest_sha256": file_sha256(output_dir / "SAMPLE_MANIFEST.json"),
            "model_contracts": model_contracts,
            "same_checkpoint_state": checkpoint_state,
            "output_contract": {
                "finite_and_well_formed": True,
                "values_allowed_to_differ": True,
                "persistent_temporal_and_model_state_mutation": False,
                "rng_mutation": False,
                "expected_ephemeral_id_counter_attested": True,
            },
            "native_shortcut_preflight": shortcut_preflight,
            "measurement_contract": prepared.plan["measurement"],
            "scopes": {
                C0_C32_FULL_DETECTOR_SCOPE: full,
                C0_C32_COMMON_MODEL_SCOPE: diagnostic,
            },
            "memory": memory,
            "near_native_gate": near_native,
            "diagnostic_gate": diagnostic_gate,
            "environment": environment,
        }
        _atomic_json(output_dir / "EVIDENCE.json", summary)
        _write_report(output_dir / "REPORT.md", summary)
        _seal_directory(output_dir)
        return summary
    except Exception as error:
        if output_dir.is_dir():
            failure = {
                "schema": "rosa.c0_c32_near_native_runtime_failure.v1",
                "status": "FAILED",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            _atomic_json(output_dir / "FAILURE.json", failure)
            _seal_directory(output_dir)
        raise
    finally:
        os.close(lock_fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.execute:
        if args.execution_state != "ARMED":
            parser.error("--execute requires --execution-state ARMED")
        if args.idle_reserved_token != IDLE_RESERVED_TOKEN:
            parser.error(
                "--execute requires --idle-reserved-token " + IDLE_RESERVED_TOKEN
            )
    prepared = prepare_protocol(args)
    if not args.execute:
        print(json.dumps(prepared.plan, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    result = execute_protocol(prepared)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
