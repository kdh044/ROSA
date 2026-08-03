#!/usr/bin/env python3
"""Train, evaluate, or benchmark ROSA through MMEngine."""

from __future__ import annotations

import argparse
import builtins
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import stat
import statistics
import sys
import tempfile
import time
import warnings

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.logging.history_buffer import HistoryBuffer

Runner = None


_FAST_PROFILES = frozenset({"fast"})
_KNOWN_RELEASE_ARCHITECTURE = "ROSA R18 Main126"
_ROSA_CONTRACT_VERSION = 3
_SEALED_TRAINING_CONTRACT_VERSION = 1
_SEALED_PAIR_CONTRACT = "rosa.c32-fast-cp16.30e-pair.v1"
_SEALED_PAIR_HEADS = {
    "control": "ROSAOfficialResidualNativeBaselineHead",
    "candidate": "ROSAOfficialResidualRefinementHead",
}
_SEALED_CONFIG_EXCLUDED_FIELDS = frozenset({"load_from", "resume", "work_dir"})
_ROSA_TOPOLOGY_FIELDS = (
    "architecture_profile",
    "rfl_topology_contract",
    "embed_dims",
    "hidden_dims",
    "num_keypoints",
    "num_cameras",
    "num_levels",
    "enable_quality_score",
)
_ROSA_GEOMETRY_FIELDS = ("anchor_z_is_bottom", "center_target_mode")
_ROSA_REFINER_STATE_PREFIXES = {
    "modern": "head.rosa_refiner.",
    "legacy": "head.canonical_sixth_block.",
}
_OBJECT_DEPTH_TRAIN_ONLY_KEYS = tuple(
    f"depth_branch.depth_layers.{level}.{field}"
    for level in range(3)
    for field in ("weight", "bias")
)
_OBJECT_DEPTH_TRAIN_ONLY_SHAPES = {
    f"depth_branch.depth_layers.{level}.weight": (1, 256, 1, 1) for level in range(3)
} | {f"depth_branch.depth_layers.{level}.bias": (1,) for level in range(3)}
_OBJECT_DEPTH_TRAIN_ONLY_ELEMENTS = 771
_OBJECT_DEPTH_SOURCE_STATE_KEYS = 601
_OBJECT_DEPTH_DEPLOY_STATE_KEYS = 595
_OBJECT_DEPTH_DEPLOY_STATE_ELEMENTS = 29_676_729
_OBJECT_DEPTH_SOURCE_STRUCTURE_SHA256 = (
    "e59e960c6dc00509920af1e805cea44617cb72d3efa05b6d17c4bb4dd4db62e7"
)
_OBJECT_DEPTH_DEPLOY_STRUCTURE_SHA256 = (
    "44ee67b0ef631edaeafa3882195201bec05d8f4a94c58b83410f4f650b325938"
)
_OBJECT_DEPTH_TRAIN_ONLY_METADATA_KEYS = (
    "depth_branch",
    "depth_branch.depth_layers",
    "depth_branch.depth_layers.0",
    "depth_branch.depth_layers.1",
    "depth_branch.depth_layers.2",
)
_OBJECT_DEPTH_SOURCE_METADATA_KEYS = 507
_OBJECT_DEPTH_DEPLOY_METADATA_KEYS = 502
_OBJECT_DEPTH_SOURCE_METADATA_SHA256 = (
    "b8ebaaa6f0609314758355ae7fb68f98917929f741508f835946725c2062d989"
)
_OBJECT_DEPTH_DEPLOY_METADATA_SHA256 = (
    "cab2f1b69ace770a6d091e65d1d10915de81efc0349437e0feb3862a371d073d"
)
_C73_APPROVED_SOURCE_SHA256 = frozenset(
    {
        "4ad519785c4aaf92af83eeeac39a26d0ccdb5167492b0ba3f4c04c4dd99ab6cb",
        "6835bdc7be22798ecd6278334b826a862dddf2072eeb71d97b7f41e58a9fa1e8",
    }
)
_OBJECT_DEPTH_DEPLOY_CONTRACT = "rosa.native.object_lidar_depth.train_only_strip.v1"
_C73_DEPLOY_BEHAVIOR = "rosa.c73.official_sparse4d_gnn_residual.v1"
_NATIVE_DEPLOY_TARGET_HEADS = {
    _C73_DEPLOY_BEHAVIOR: "ROSAOfficialResidualNativeBaselineHead",
}
_C73_DEPLOY_DEFORMABLE_TYPES = frozenset(
    {
        "ROSAFullFOVDeformableFeatureAggregation",
        ("ROSAInferenceReuseExactHomogeneous4x4FullFOV" "DeformableFeatureAggregation"),
    }
)
_C73_DEPLOY_KEYPOINT_GENERATOR = "ROSANormalizedKeyPointsGenerator"
_C73_DEPLOY_META_KEYS = frozenset(
    {
        "deploy_contract",
        "native_deploy_behavior",
        "native_deploy_target_head_type",
        "native_deploy_allowed_deformable_types",
        "native_deploy_keypoint_generator_type",
        "native_deploy_projection_type",
        "native_deploy_anchor_z_is_bottom",
        "source_checkpoint_sha256",
        "source_state_keys",
        "deploy_state_keys",
        "deploy_state_structure_sha256",
        "deploy_state_metadata_sha256",
        "stripped_state_keys",
        "stripped_elements",
        "optimizer_state_included",
        "scheduler_state_included",
    }
)


def _CheckpointContractHook(metadata: Mapping):
    """Build the hook lazily so checkpoint inspection stays device-agnostic."""

    from rosa.mmengine_hooks import CheckpointContractHook

    return CheckpointContractHook(metadata)


def _ResumeContractValidationHook(payload_contract: Mapping):
    """Build the pre-load validator lazily with MMEngine's Hook base class."""

    from rosa.mmengine_hooks import ResumeContractValidationHook

    return ResumeContractValidationHook(payload_contract)


def _runner_class():
    """Resolve Runner lazily while preserving the module test seam."""

    if Runner is not None:
        return Runner
    from mmengine.runner import Runner as MMEngineRunner

    return MMEngineRunner


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("config", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--launcher",
        choices=("none", "pytorch", "slurm", "mpi"),
        default="none",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config entries as key=value pairs.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train")
    _add_common_arguments(train)
    train.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        help="Resume from a trusted local checkpoint path or auto-discover it.",
    )
    train.add_argument("--amp", action="store_true")

    evaluate = commands.add_parser("evaluate")
    _add_common_arguments(evaluate)
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("--dump", type=Path)
    evaluate.add_argument(
        "--allow-checkpoint-contract-mismatch",
        action="store_true",
    )

    benchmark = commands.add_parser("benchmark")
    _add_common_arguments(benchmark)
    benchmark.add_argument("--checkpoint", type=Path)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--warmup", type=int, default=20)
    benchmark.add_argument("--iterations", type=int, default=100)
    benchmark.add_argument(
        "--baseline",
        action="store_true",
        help="Benchmark the same config with the ROSA sidecar removed.",
    )
    benchmark.add_argument(
        "--allow-checkpoint-contract-mismatch",
        action="store_true",
    )

    export_deploy = commands.add_parser(
        "export-deploy",
        help=(
            "Export an approved C73 checkpoint as weights-only native C32 "
            "behavior by removing the exact train-only object-LiDAR depth branch."
        ),
    )
    export_deploy.add_argument("checkpoint", type=Path)
    export_deploy.add_argument("output", type=Path)

    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Config:
    config = Config.fromfile(args.config)
    if args.cfg_options:
        config.merge_from_dict(args.cfg_options)
    if args.work_dir:
        config.work_dir = str(args.work_dir.resolve())
    config.launcher = args.launcher
    return config


def _rosa_profile(config: Config) -> str | None:
    rosa_config = config.model.get("head", {}).get("rosa_cfg")
    return rosa_config.get("profile", "explicit") if rosa_config is not None else None


def _effective_rosa_config(config: Config):
    rosa_mapping = config.model.get("head", {}).get("rosa_cfg")
    if rosa_mapping is None:
        return None

    from rosa import ROSAConfig

    return ROSAConfig.from_mapping(rosa_mapping)


def _json_sha256(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _config_sha256(config: Config) -> str:
    """Hash the resolved experiment while excluding load-location controls."""

    payload = {
        key: value
        for key, value in config.to_dict().items()
        if key not in _SEALED_CONFIG_EXCLUDED_FIELDS
    }
    return _json_sha256(payload)


def _runtime_identity() -> dict:
    packages = {}
    for distribution in ("mmcv", "mmengine", "mmdet", "mmdet3d", "numpy"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "python_implementation": sys.implementation.name,
        "python_version": ".".join(str(value) for value in sys.version_info[:3]),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        "packages": packages,
    }


def _revision_identity(path: Path) -> dict:
    from rosa.provenance import read_source_revision

    revision = read_source_revision(path)
    return {
        "head": revision.get("head"),
        "tracked_dirty": revision.get("tracked_dirty"),
        "tracked_diff_sha256": revision.get("tracked_diff_sha256"),
        "stable_capture": revision.get("stable_capture"),
        "error": revision.get("error"),
    }


def _current_source_identity() -> dict:
    source_root = Path(__file__).resolve().parents[1]
    detector_root = Path(
        os.environ.get(
            "ROSA_DETECTOR_HOST_ROOT",
            os.environ.get(
                "SPARSE4D_ROOT",
                source_root / "external" / "detector_host",
            ),
        )
    ).resolve()
    fisheye_root = Path(
        os.environ.get(
            "FISHEYE3DOD_OFFICIAL_ROOT",
            source_root / "external" / "fisheye3dod",
        )
    ).resolve()
    return {
        "rosa": _revision_identity(source_root),
        "detector_host": _revision_identity(detector_root),
        "fisheye3dod": _revision_identity(fisheye_root),
    }


def _sealed_resume_spec(config: Config) -> Mapping | None:
    spec = config.get("sealed_resume_contract")
    if spec is None:
        return None
    if not isinstance(spec, Mapping):
        raise TypeError("sealed_resume_contract must be a mapping")
    if spec.get("contract") != _SEALED_PAIR_CONTRACT:
        raise ValueError("unknown sealed_resume_contract")
    role = spec.get("role")
    if role not in _SEALED_PAIR_HEADS:
        raise ValueError("sealed_resume_contract role must be control or candidate")
    head_type = config.model.get("head", {}).get("type")
    expected_head = _SEALED_PAIR_HEADS[role]
    if head_type != expected_head:
        raise ValueError(f"sealed {role} arm requires exact head {expected_head!r}")
    return spec


def _sealed_training_contract(config: Config) -> dict | None:
    spec = _sealed_resume_spec(config)
    if spec is None:
        return None

    max_epochs = config.train_cfg.get("max_epochs")
    if isinstance(max_epochs, bool) or max_epochs != 30:
        raise ValueError("sealed C32/ROSA pair requires exactly 30 epochs")
    schedulers = config.get("param_scheduler")
    if not isinstance(schedulers, (list, tuple)) or len(schedulers) != 2:
        raise ValueError("sealed C32/ROSA pair requires exactly two schedulers")
    cosine = schedulers[1]
    if (
        not isinstance(cosine, Mapping)
        or cosine.get("type") != "CosineAnnealingLR"
        or cosine.get("begin") != 0
        or cosine.get("end") != 30
        or cosine.get("T_max") != 30
        or cosine.get("by_epoch") is not True
        or cosine.get("convert_to_iter_based") is not True
    ):
        raise ValueError("sealed C32/ROSA pair requires the exact 30e cosine horizon")

    checkpoint_policy = config.default_hooks.get("checkpoint", {})
    if (
        checkpoint_policy.get("type") != "CheckpointHook"
        or checkpoint_policy.get("interval") != 1
        or checkpoint_policy.get("max_keep_ckpts") != 5
        or checkpoint_policy.get("save_best") != ["fisheye_bev/mAP", "fisheye_bev/NDS"]
        or checkpoint_policy.get("rule") != ["greater", "greater"]
    ):
        raise ValueError(
            "sealed C32/ROSA pair requires per-epoch checkpoints, last-five, "
            "and mAP/NDS best"
        )
    custom_hooks = config.get("custom_hooks")
    if not isinstance(custom_hooks, (list, tuple)):
        raise ValueError("sealed C32/ROSA pair requires custom_hooks")
    milestone_hooks = [
        hook
        for hook in custom_hooks
        if hook.get("type") == "ROSASparseMilestoneCheckpointHook"
    ]
    if len(milestone_hooks) != 1 or tuple(milestone_hooks[0].get("milestones", ())) != (
        8,
    ):
        raise ValueError("sealed C32/ROSA pair requires the protected e8 milestone")

    source_identity = _current_source_identity()
    for name, revision in source_identity.items():
        if (
            revision.get("stable_capture") is not True
            or not isinstance(revision.get("head"), str)
            or not isinstance(revision.get("tracked_dirty"), bool)
            or not isinstance(revision.get("tracked_diff_sha256"), str)
        ):
            raise ValueError(f"sealed source identity is unavailable for {name}")

    filename = getattr(config, "filename", None)
    return {
        "version": _SEALED_TRAINING_CONTRACT_VERSION,
        "contract": _SEALED_PAIR_CONTRACT,
        "role": spec["role"],
        "head_type": config.model.head.type,
        "config_name": None if filename is None else Path(filename).name,
        "config_sha256": _config_sha256(config),
        "max_epochs": max_epochs,
        "optimizer_config_sha256": _json_sha256(config.optim_wrapper.to_dict()),
        "scheduler_config_sha256": _json_sha256(
            [scheduler.to_dict() for scheduler in schedulers]
        ),
        "checkpoint_policy_sha256": _json_sha256(checkpoint_policy.to_dict()),
        "runtime_identity": _runtime_identity(),
        "source_identity": source_identity,
    }


def _checkpoint_contract_metadata(config: Config) -> dict | None:
    effective = _effective_rosa_config(config)
    metadata = {}
    if effective is not None:
        metadata.update(
            {
                "rosa_contract_version": _ROSA_CONTRACT_VERSION,
                "rfl_topology_contract": effective.rfl_topology_contract,
                "projection_contract": effective.projection_contract,
                "refinement_profile": _rosa_profile(config),
                "rosa_config": asdict(effective),
            }
        )
    sealed = _sealed_training_contract(config)
    if sealed is not None:
        metadata["sealed_training_contract"] = sealed
    return metadata or None


def _checkpoint_state_dict(checkpoint: Mapping) -> Mapping | None:
    state_dict = checkpoint.get("state_dict")
    if isinstance(state_dict, Mapping):
        return state_dict
    if checkpoint and all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in checkpoint.items()
    ):
        return checkpoint
    return None


def _state_dict_has_rosa(state_dict: Mapping | None) -> bool | None:
    if state_dict is None:
        return None
    return any(
        any(part == "rosa" or part.startswith("rosa_") for part in key.split("."))
        for key in state_dict
        if isinstance(key, str)
    )


def _expected_rosa_refiner_state_shapes(effective) -> dict[str, tuple[int, ...]]:
    """Build the config-owned refiner schema without advancing caller RNG."""

    from rosa import ROSARefiner

    with torch.random.fork_rng(devices=[]):
        state_dict = ROSARefiner(
            effective,
            projection_type="ocam_fisheye",
        ).state_dict()
    return {
        key: tuple(int(dimension) for dimension in value.shape)
        for key, value in state_dict.items()
    }


def _split_rosa_refiner_state_key(key: str) -> tuple[str, str] | None:
    """Return the loader-compatible namespace and relative refiner key."""

    # MMEngine's checkpoint loader removes one leading ``module.`` before
    # loading a DDP checkpoint. Mirror exactly that accepted spelling here.
    normalized = key.removeprefix("module.")
    for namespace, prefix in _ROSA_REFINER_STATE_PREFIXES.items():
        if normalized.startswith(prefix):
            return namespace, normalized[len(prefix) :]
    return None


def _summarize_state_names(names: list[str], *, limit: int = 4) -> str:
    shown = ", ".join(repr(name) for name in names[:limit])
    omitted = len(names) - limit
    return f"{shown}, ... (+{omitted})" if omitted > 0 else shown


def _factorized_refiner_state_mismatches(
    state_dict: Mapping | None,
    effective,
) -> list[str]:
    """Validate every compact ROSA refiner key and exact tensor shape."""

    if not isinstance(state_dict, Mapping):
        return ["factorized compact checkpoint has no state_dict mapping"]

    expected = _expected_rosa_refiner_state_shapes(effective)
    observed: dict[str, object] = {}
    namespaces: set[str] = set()
    duplicate_keys: set[str] = set()
    unrecognized_rosa_keys: list[str] = []
    for key, value in state_dict.items():
        if not isinstance(key, str):
            continue
        split = _split_rosa_refiner_state_key(key)
        if split is None:
            if any(
                part == "rosa" or part.startswith("rosa_") for part in key.split(".")
            ):
                unrecognized_rosa_keys.append(key)
            continue
        namespace, relative_key = split
        namespaces.add(namespace)
        if relative_key in observed:
            duplicate_keys.add(relative_key)
        else:
            observed[relative_key] = value

    mismatches: list[str] = []
    if not observed:
        mismatches.append("factorized compact checkpoint lacks ROSA refiner state")
    if len(namespaces) > 1:
        mismatches.append(
            "factorized compact ROSA state mixes modern and legacy refiner prefixes"
        )
    if duplicate_keys:
        names = sorted(duplicate_keys)
        mismatches.append(
            "factorized compact ROSA state duplicates keys "
            f"[{_summarize_state_names(names)}]"
        )
    if unrecognized_rosa_keys:
        names = sorted(unrecognized_rosa_keys)
        mismatches.append(
            "factorized compact checkpoint has unrecognized ROSA state keys "
            f"[{_summarize_state_names(names)}]"
        )

    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    non_tensor = sorted(
        key for key, value in observed.items() if not isinstance(value, torch.Tensor)
    )
    wrong_shape = sorted(
        (
            key,
            tuple(int(dimension) for dimension in value.shape),
            expected[key],
        )
        for key, value in observed.items()
        if key in expected
        and isinstance(value, torch.Tensor)
        and tuple(value.shape) != expected[key]
    )
    if missing:
        mismatches.append(
            "factorized compact ROSA state is missing keys "
            f"[{_summarize_state_names(missing)}]"
        )
    if unexpected:
        mismatches.append(
            "factorized compact ROSA state has unexpected keys "
            f"[{_summarize_state_names(unexpected)}]"
        )
    if non_tensor:
        mismatches.append(
            "factorized compact ROSA state has non-tensor values at "
            f"[{_summarize_state_names(non_tensor)}]"
        )
    if wrong_shape:
        rendered = [
            f"{key!r}: {observed_shape} != {expected_shape}"
            for key, observed_shape, expected_shape in wrong_shape
        ]
        mismatches.append(
            "factorized compact ROSA state has wrong shapes "
            f"[{_summarize_state_names(rendered)}]"
        )
    return mismatches


def _load_checkpoint_without_executing_code(checkpoint_path: Path):
    """Load MMEngine checkpoints with a narrow weights-only allowlist."""

    try:
        from numpy._core.multiarray import _reconstruct, scalar
    except ImportError:  # pragma: no cover - NumPy 1.x compatibility
        from numpy.core.multiarray import _reconstruct, scalar

    dtype_specs = (
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
        "datetime64",
        "timedelta64",
        "object",
        "str",
        "bytes",
    )
    dtype_classes = {type(np.dtype(specification)) for specification in dtype_specs}
    safe_globals = [
        HistoryBuffer,
        np.ndarray,
        np.dtype,
        _reconstruct,
        scalar,
        builtins.getattr,
        *dtype_classes,
    ]
    with torch.serialization.safe_globals(safe_globals):
        return torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )


class _PrivateCheckpointSnapshot:
    """One process-local, unlinked checkpoint snapshot held by a descriptor."""

    __slots__ = ("path", "source_path", "private_directory", "sha256", "byte_count")

    def __init__(
        self,
        *,
        path: Path,
        source_path: Path,
        private_directory: Path,
        sha256: str,
        byte_count: int,
    ) -> None:
        self.path = path
        self.source_path = source_path
        self.private_directory = private_directory
        self.sha256 = sha256
        self.byte_count = byte_count


def _regular_file_signature(file_stat: os.stat_result) -> tuple[int, ...]:
    """Return fields that expose replacement or ordinary in-place mutation."""

    return (
        int(file_stat.st_dev),
        int(file_stat.st_ino),
        int(file_stat.st_mode),
        int(file_stat.st_size),
        int(file_stat.st_mtime_ns),
        int(file_stat.st_ctime_ns),
    )


@contextmanager
def _private_checkpoint_snapshot(checkpoint_path: Path):
    """Copy a local regular file once and expose only its held descriptor.

    Validation and MMEngine loading must both use ``snapshot.path``.  The
    copied file is unlinked before yielding, so replacing the source pathname,
    mutating the source inode, or guessing the private directory cannot change
    the bytes reached through ``/proc/self/fd``.  The descriptor and its 0700
    directory remain alive for the caller's load window and are released on
    every normal or exceptional exit.

    Under torchrun each DDP process executes this function independently.  A
    descriptor path is process-local by definition, so it must never be passed
    between ranks; every rank validates and loads its own exact-byte snapshot.
    """

    source_path = Path(checkpoint_path).expanduser().resolve(strict=True)
    read_flags = os.O_RDONLY
    read_flags |= getattr(os, "O_CLOEXEC", 0)
    read_flags |= getattr(os, "O_NOFOLLOW", 0)
    read_flags |= getattr(os, "O_NONBLOCK", 0)
    create_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    create_flags |= getattr(os, "O_CLOEXEC", 0)
    create_flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_flags = os.O_RDONLY
    directory_flags |= getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)

    source_fd: int | None = None
    directory_fd: int | None = None
    writable_snapshot_fd: int | None = None
    sealed_snapshot_fd: int | None = None
    with tempfile.TemporaryDirectory(prefix="rosa-checkpoint-") as directory_name:
        private_directory = Path(directory_name)
        directory_stat = private_directory.stat()
        if not stat.S_ISDIR(directory_stat.st_mode) or (
            stat.S_IMODE(directory_stat.st_mode) & 0o077
        ):
            raise RuntimeError("checkpoint snapshot directory must be private")
        try:
            source_fd = os.open(source_path, read_flags)
            source_before = os.fstat(source_fd)
            if not stat.S_ISREG(source_before.st_mode):
                raise ValueError("checkpoint source must be a regular file")

            directory_fd = os.open(private_directory, directory_flags)
            opened_directory_stat = os.fstat(directory_fd)
            if (
                not stat.S_ISDIR(opened_directory_stat.st_mode)
                or stat.S_IMODE(opened_directory_stat.st_mode) & 0o077
                or int(opened_directory_stat.st_dev) != int(directory_stat.st_dev)
                or int(opened_directory_stat.st_ino) != int(directory_stat.st_ino)
            ):
                raise RuntimeError("checkpoint snapshot directory changed")
            writable_snapshot_fd = os.open(
                "checkpoint.pth",
                create_flags,
                0o600,
                dir_fd=directory_fd,
            )
            digest = hashlib.sha256()
            byte_count = 0
            while True:
                block = os.read(source_fd, 1024 * 1024)
                if not block:
                    break
                digest.update(block)
                byte_count += len(block)
                remaining = memoryview(block)
                while remaining:
                    written = os.write(writable_snapshot_fd, remaining)
                    if written <= 0:
                        raise OSError("checkpoint snapshot write made no progress")
                    remaining = remaining[written:]
            os.fsync(writable_snapshot_fd)

            source_after = os.fstat(source_fd)
            if _regular_file_signature(source_before) != _regular_file_signature(
                source_after
            ):
                raise RuntimeError("checkpoint source changed while being snapshotted")
            if byte_count != int(source_before.st_size):
                raise RuntimeError("checkpoint snapshot byte count changed during copy")

            snapshot_stat = os.fstat(writable_snapshot_fd)
            if not stat.S_ISREG(snapshot_stat.st_mode):
                raise RuntimeError("checkpoint snapshot is not a regular file")
            if int(snapshot_stat.st_size) != byte_count:
                raise RuntimeError("checkpoint snapshot size is incomplete")

            sealed_snapshot_fd = os.open(
                "checkpoint.pth",
                read_flags,
                dir_fd=directory_fd,
            )
            sealed_stat = os.fstat(sealed_snapshot_fd)
            if (
                not stat.S_ISREG(sealed_stat.st_mode)
                or int(sealed_stat.st_dev) != int(snapshot_stat.st_dev)
                or int(sealed_stat.st_ino) != int(snapshot_stat.st_ino)
                or int(sealed_stat.st_size) != byte_count
            ):
                raise RuntimeError("checkpoint snapshot descriptor changed")

            os.unlink("checkpoint.pth", dir_fd=directory_fd)
            os.close(writable_snapshot_fd)
            writable_snapshot_fd = None
            os.close(directory_fd)
            directory_fd = None
            os.close(source_fd)
            source_fd = None
            descriptor_path = Path(f"/proc/self/fd/{sealed_snapshot_fd}")
            if not descriptor_path.is_file():
                raise RuntimeError(
                    "checkpoint sealing requires Linux procfs descriptors"
                )
            yield _PrivateCheckpointSnapshot(
                path=descriptor_path,
                source_path=source_path,
                private_directory=private_directory,
                sha256=digest.hexdigest(),
                byte_count=byte_count,
            )
        finally:
            for descriptor in (
                sealed_snapshot_fd,
                writable_snapshot_fd,
                directory_fd,
                source_fd,
            ):
                if descriptor is not None:
                    os.close(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_approved_c73_checkpoint(source: Path):
    """Hash one private byte snapshot, then deserialize that exact snapshot."""

    digest = hashlib.sha256()
    byte_count = 0
    with source.open("rb") as input_stream, tempfile.TemporaryFile() as snapshot:
        for block in iter(lambda: input_stream.read(1024 * 1024), b""):
            digest.update(block)
            snapshot.write(block)
            byte_count += len(block)
        source_sha256 = digest.hexdigest()
        if source_sha256 not in _C73_APPROVED_SOURCE_SHA256:
            raise ValueError(
                "source is not an approved C73 official-residual checkpoint: "
                f"{source_sha256}"
            )
        snapshot.flush()
        os.fsync(snapshot.fileno())
        snapshot.seek(0)
        checkpoint = _load_checkpoint_without_executing_code(snapshot)
    return checkpoint, source_sha256, byte_count


def _state_structure_sha256(state_dict: Mapping) -> str:
    """Hash every state key, shape, dtype, and layout in sorted-key order."""

    records = []
    for key in sorted(state_dict):
        if not isinstance(key, str):
            raise TypeError("checkpoint state keys must be strings")
        tensor = state_dict[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"checkpoint state {key!r} is not a tensor")
        records.append(
            {
                "key": key,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "layout": str(tensor.layout),
            }
        )
    payload = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _state_metadata_sha256(metadata: Mapping) -> str:
    """Hash state-dict module-version metadata without coercing values."""

    for key, value in metadata.items():
        if not isinstance(key, str):
            raise TypeError("checkpoint state metadata keys must be strings")
        if not isinstance(value, Mapping):
            raise TypeError(f"checkpoint state metadata {key!r} is not a mapping")
        if set(value) != {"version"} or not isinstance(value["version"], int):
            raise ValueError(
                f"checkpoint state metadata {key!r} must contain one integer version"
            )
    payload = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _object_depth_deploy_state_dict(state_dict: Mapping) -> OrderedDict:
    """Remove only C68/C73's six audited train-only depth tensors."""

    if len(state_dict) != _OBJECT_DEPTH_SOURCE_STATE_KEYS:
        raise ValueError(
            "object-depth source checkpoint must contain exactly "
            f"{_OBJECT_DEPTH_SOURCE_STATE_KEYS} state keys, got "
            f"{len(state_dict)}"
        )
    if _state_dict_has_rosa(state_dict):
        raise ValueError("object-depth native deployment cannot contain ROSA state")
    source_structure = _state_structure_sha256(state_dict)
    if source_structure != _OBJECT_DEPTH_SOURCE_STRUCTURE_SHA256:
        raise ValueError(
            "object-depth source state structure changed: expected "
            f"{_OBJECT_DEPTH_SOURCE_STRUCTURE_SHA256}, got {source_structure}"
        )
    source_metadata = getattr(state_dict, "_metadata", None)
    if not isinstance(source_metadata, Mapping):
        raise ValueError("object-depth source state is missing module metadata")
    if len(source_metadata) != _OBJECT_DEPTH_SOURCE_METADATA_KEYS:
        raise ValueError(
            "object-depth source metadata must contain exactly "
            f"{_OBJECT_DEPTH_SOURCE_METADATA_KEYS} entries, got "
            f"{len(source_metadata)}"
        )
    source_metadata_sha256 = _state_metadata_sha256(source_metadata)
    if source_metadata_sha256 != _OBJECT_DEPTH_SOURCE_METADATA_SHA256:
        raise ValueError(
            "object-depth source metadata changed: expected "
            f"{_OBJECT_DEPTH_SOURCE_METADATA_SHA256}, got "
            f"{source_metadata_sha256}"
        )

    depth_keys = {key for key in state_dict if key.startswith("depth_branch.")}
    expected_keys = set(_OBJECT_DEPTH_TRAIN_ONLY_KEYS)
    if depth_keys != expected_keys:
        missing = sorted(expected_keys.difference(depth_keys))
        unexpected = sorted(depth_keys.difference(expected_keys))
        raise ValueError(
            "train-only object-depth key set changed: "
            f"missing={missing}, unexpected={unexpected}"
        )

    removed_elements = 0
    for key in _OBJECT_DEPTH_TRAIN_ONLY_KEYS:
        tensor = state_dict[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"checkpoint state {key!r} is not a tensor")
        expected_shape = _OBJECT_DEPTH_TRAIN_ONLY_SHAPES[key]
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"train-only object-depth shape changed for {key}: "
                f"expected {expected_shape}, got {tuple(tensor.shape)}"
            )
        removed_elements += tensor.numel()
    if removed_elements != _OBJECT_DEPTH_TRAIN_ONLY_ELEMENTS:
        raise ValueError(
            "train-only object-depth element count changed: "
            f"expected {_OBJECT_DEPTH_TRAIN_ONLY_ELEMENTS}, got "
            f"{removed_elements}"
        )

    deploy_state = OrderedDict(
        (key, value) for key, value in state_dict.items() if key not in expected_keys
    )
    if len(deploy_state) != _OBJECT_DEPTH_DEPLOY_STATE_KEYS:
        raise ValueError(
            "native deploy state must contain exactly "
            f"{_OBJECT_DEPTH_DEPLOY_STATE_KEYS} keys"
        )
    deploy_elements = sum(tensor.numel() for tensor in deploy_state.values())
    if deploy_elements != _OBJECT_DEPTH_DEPLOY_STATE_ELEMENTS:
        raise ValueError(
            "native deploy state element count changed: expected "
            f"{_OBJECT_DEPTH_DEPLOY_STATE_ELEMENTS}, got {deploy_elements}"
        )
    deploy_structure = _state_structure_sha256(deploy_state)
    if deploy_structure != _OBJECT_DEPTH_DEPLOY_STRUCTURE_SHA256:
        raise ValueError(
            "native deploy state structure changed: expected "
            f"{_OBJECT_DEPTH_DEPLOY_STRUCTURE_SHA256}, got {deploy_structure}"
        )

    depth_metadata_keys = set(_OBJECT_DEPTH_TRAIN_ONLY_METADATA_KEYS)
    observed_depth_metadata = {
        key
        for key in source_metadata
        if key == "depth_branch" or key.startswith("depth_branch.")
    }
    if observed_depth_metadata != depth_metadata_keys:
        raise ValueError(
            "train-only object-depth metadata key set changed: "
            f"{sorted(observed_depth_metadata)}"
        )
    deploy_metadata = OrderedDict(
        (key, value)
        for key, value in source_metadata.items()
        if key not in depth_metadata_keys
    )
    if len(deploy_metadata) != _OBJECT_DEPTH_DEPLOY_METADATA_KEYS:
        raise ValueError(
            "native deploy metadata must contain exactly "
            f"{_OBJECT_DEPTH_DEPLOY_METADATA_KEYS} entries"
        )
    deploy_metadata_sha256 = _state_metadata_sha256(deploy_metadata)
    if deploy_metadata_sha256 != _OBJECT_DEPTH_DEPLOY_METADATA_SHA256:
        raise ValueError(
            "native deploy metadata changed: expected "
            f"{_OBJECT_DEPTH_DEPLOY_METADATA_SHA256}, got "
            f"{deploy_metadata_sha256}"
        )
    deploy_state._metadata = deploy_metadata
    return deploy_state


def export_deploy(args: argparse.Namespace) -> dict:
    """Write a non-overwriting, behavior-bound C73 deployment checkpoint."""

    source = args.checkpoint.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == output:
        raise ValueError("deployment output must differ from its source checkpoint")
    if output.exists():
        raise FileExistsError(output)

    checkpoint, source_sha256, source_bytes = _load_approved_c73_checkpoint(source)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("source checkpoint must be a mapping")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("source must be an MMEngine checkpoint with state_dict")
    deploy_state = _object_depth_deploy_state_dict(state_dict)

    payload = {
        "meta": {
            "deploy_contract": _OBJECT_DEPTH_DEPLOY_CONTRACT,
            "native_deploy_behavior": _C73_DEPLOY_BEHAVIOR,
            "native_deploy_target_head_type": (
                _NATIVE_DEPLOY_TARGET_HEADS[_C73_DEPLOY_BEHAVIOR]
            ),
            "native_deploy_allowed_deformable_types": sorted(
                _C73_DEPLOY_DEFORMABLE_TYPES
            ),
            "native_deploy_keypoint_generator_type": (_C73_DEPLOY_KEYPOINT_GENERATOR),
            "native_deploy_projection_type": "ocam_fisheye",
            "native_deploy_anchor_z_is_bottom": False,
            "source_checkpoint_sha256": source_sha256,
            "source_state_keys": len(state_dict),
            "deploy_state_keys": len(deploy_state),
            "deploy_state_structure_sha256": (_OBJECT_DEPTH_DEPLOY_STRUCTURE_SHA256),
            "deploy_state_metadata_sha256": (_OBJECT_DEPTH_DEPLOY_METADATA_SHA256),
            "stripped_state_keys": list(_OBJECT_DEPTH_TRAIN_ONLY_KEYS),
            "stripped_elements": _OBJECT_DEPTH_TRAIN_ONLY_ELEMENTS,
            "optimizer_state_included": False,
            "scheduler_state_included": False,
        },
        "state_dict": deploy_state,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        # Hard-linking is atomic and fails if another process created output;
        # unlike os.replace(), it can never overwrite a checkpoint.
        os.link(temporary_path, output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    result = {
        "source": str(source),
        "output": str(output),
        "source_sha256": source_sha256,
        "output_sha256": _sha256_file(output),
        "source_bytes": source_bytes,
        "output_bytes": output.stat().st_size,
        "state_keys": len(deploy_state),
        "stripped_state_keys": list(_OBJECT_DEPTH_TRAIN_ONLY_KEYS),
        "stripped_elements": _OBJECT_DEPTH_TRAIN_ONLY_ELEMENTS,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _topology_signature(config) -> tuple:
    return tuple(getattr(config, name) for name in _ROSA_TOPOLOGY_FIELDS)


def _geometry_signature(config) -> tuple:
    return tuple(getattr(config, name) for name in _ROSA_GEOMETRY_FIELDS)


def _c73_deploy_payload_mismatches(checkpoint: Mapping, config: Config) -> list[str]:
    """Validate the complete exported payload and its executable target."""

    mismatches = []
    if set(checkpoint) != {"meta", "state_dict"}:
        mismatches.append("deploy checkpoint top-level keys changed")
    metadata = checkpoint.get("meta")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(metadata, Mapping):
        return [*mismatches, "deploy checkpoint metadata is missing"]
    if not isinstance(state_dict, Mapping):
        return [*mismatches, "deploy checkpoint state_dict is missing"]

    if set(metadata) != _C73_DEPLOY_META_KEYS:
        mismatches.append("deploy checkpoint metadata key set changed")
    expected_metadata = {
        "deploy_contract": _OBJECT_DEPTH_DEPLOY_CONTRACT,
        "native_deploy_behavior": _C73_DEPLOY_BEHAVIOR,
        "native_deploy_target_head_type": (
            _NATIVE_DEPLOY_TARGET_HEADS[_C73_DEPLOY_BEHAVIOR]
        ),
        "native_deploy_allowed_deformable_types": sorted(_C73_DEPLOY_DEFORMABLE_TYPES),
        "native_deploy_keypoint_generator_type": (_C73_DEPLOY_KEYPOINT_GENERATOR),
        "native_deploy_projection_type": "ocam_fisheye",
        "native_deploy_anchor_z_is_bottom": False,
        "source_state_keys": _OBJECT_DEPTH_SOURCE_STATE_KEYS,
        "deploy_state_keys": _OBJECT_DEPTH_DEPLOY_STATE_KEYS,
        "deploy_state_structure_sha256": (_OBJECT_DEPTH_DEPLOY_STRUCTURE_SHA256),
        "deploy_state_metadata_sha256": (_OBJECT_DEPTH_DEPLOY_METADATA_SHA256),
        "stripped_state_keys": list(_OBJECT_DEPTH_TRAIN_ONLY_KEYS),
        "stripped_elements": _OBJECT_DEPTH_TRAIN_ONLY_ELEMENTS,
        "optimizer_state_included": False,
        "scheduler_state_included": False,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            mismatches.append(f"deploy checkpoint metadata {key!r} changed")
    if metadata.get("source_checkpoint_sha256") not in _C73_APPROVED_SOURCE_SHA256:
        mismatches.append("deploy checkpoint source SHA256 is not approved")

    if len(state_dict) != _OBJECT_DEPTH_DEPLOY_STATE_KEYS:
        mismatches.append("deploy checkpoint state key count changed")
    try:
        structure_sha256 = _state_structure_sha256(state_dict)
    except (TypeError, ValueError) as error:
        mismatches.append(f"deploy checkpoint state structure is invalid ({error})")
    else:
        if structure_sha256 != _OBJECT_DEPTH_DEPLOY_STRUCTURE_SHA256:
            mismatches.append("deploy checkpoint state structure changed")
        elements = sum(tensor.numel() for tensor in state_dict.values())
        if elements != _OBJECT_DEPTH_DEPLOY_STATE_ELEMENTS:
            mismatches.append("deploy checkpoint state element count changed")
    state_metadata = getattr(state_dict, "_metadata", None)
    if not isinstance(state_metadata, Mapping):
        mismatches.append("deploy checkpoint module metadata is missing")
    else:
        if len(state_metadata) != _OBJECT_DEPTH_DEPLOY_METADATA_KEYS:
            mismatches.append("deploy checkpoint module metadata count changed")
        try:
            metadata_sha256 = _state_metadata_sha256(state_metadata)
        except (TypeError, ValueError) as error:
            mismatches.append(f"deploy checkpoint module metadata is invalid ({error})")
        else:
            if metadata_sha256 != _OBJECT_DEPTH_DEPLOY_METADATA_SHA256:
                mismatches.append("deploy checkpoint module metadata changed")
    if any(key.startswith("depth_branch.") for key in state_dict):
        mismatches.append("deploy checkpoint retained train-only depth state")
    if _state_dict_has_rosa(state_dict):
        mismatches.append("deploy checkpoint contains ROSA sidecar state")

    head_config = config.model.get("head", {})
    expected_head_type = _NATIVE_DEPLOY_TARGET_HEADS[_C73_DEPLOY_BEHAVIOR]
    if head_config.get("type") != expected_head_type:
        mismatches.append(f"deploy behavior requires head {expected_head_type!r}")
    deformable = head_config.get("deformable_model", {})
    if deformable.get("type") not in _C73_DEPLOY_DEFORMABLE_TYPES:
        mismatches.append("deploy behavior requires exact homogeneous OCam projection")
    if deformable.get("projection_type") != "ocam_fisheye":
        mismatches.append("deploy behavior requires OCam projection_type")
    keypoint_generator = deformable.get("kps_generator", {})
    if keypoint_generator.get("type") != _C73_DEPLOY_KEYPOINT_GENERATOR:
        mismatches.append("deploy behavior requires normalized keypoint geometry")
    if bool(keypoint_generator.get("anchor_z_is_bottom", False)):
        mismatches.append("deploy behavior requires gravity-centred anchor z")
    return mismatches


def _attach_runner_model_contract(metadata: dict, runner) -> None:
    sealed = metadata.get("sealed_training_contract")
    if not isinstance(sealed, dict):
        return
    from mmengine.model import is_model_wrapper
    from rosa.checkpoint_contract import model_state_contract

    model = runner.model.module if is_model_wrapper(runner.model) else runner.model
    sealed["model_state_contract"] = model_state_contract(model.state_dict())


def _sealed_checkpoint_mismatches(
    checkpoint: Mapping,
    metadata: Mapping,
    config: Config,
    *,
    require_resume_state: bool,
) -> tuple[dict | None, list[str]]:
    """Validate exact training, model, optimizer, and scheduler identities."""

    expected = _sealed_training_contract(config)
    if expected is None:
        return None, []
    observed = metadata.get("sealed_training_contract")
    if not isinstance(observed, Mapping):
        return None, ["checkpoint lacks sealed_training_contract metadata"]

    mismatches = []
    for field, expected_value in expected.items():
        if observed.get(field) != expected_value:
            label = field.replace("_", " ")
            mismatches.append(f"sealed {label} differs from current config/runtime")

    expected_model = observed.get("model_state_contract")
    if not isinstance(expected_model, Mapping):
        mismatches.append("sealed checkpoint lacks the full model state contract")

    stored_payload = metadata.get("sealed_payload_contract")
    if not isinstance(stored_payload, Mapping):
        mismatches.append("checkpoint lacks sealed_payload_contract metadata")
        return None, mismatches

    from rosa.checkpoint_contract import checkpoint_payload_contract

    try:
        payload = checkpoint_payload_contract(checkpoint)
    except (TypeError, ValueError) as error:
        mismatches.append(f"sealed checkpoint payload is invalid ({error})")
        return None, mismatches
    if payload != stored_payload:
        mismatches.append("sealed checkpoint payload schema differs from metadata")
    if isinstance(expected_model, Mapping) and payload["model_state"] != expected_model:
        mismatches.append(
            "sealed checkpoint omits or changes native/sidecar model state"
        )

    if require_resume_state:
        if payload["optimizer_state"] is None:
            mismatches.append("resume checkpoint is missing optimizer state")
        if payload["scheduler_state"] is None:
            mismatches.append("resume checkpoint is missing param_scheduler state")
        epoch = metadata.get("epoch")
        iteration = metadata.get("iter")
        if (
            not isinstance(epoch, int)
            or isinstance(epoch, bool)
            or not 0 <= epoch <= expected["max_epochs"]
        ):
            mismatches.append("resume checkpoint epoch is outside the sealed horizon")
        if (
            not isinstance(iteration, int)
            or isinstance(iteration, bool)
            or iteration < 0
        ):
            mismatches.append("resume checkpoint iter must be a non-negative integer")
        optimizer_state = payload["optimizer_state"]
        if (
            isinstance(epoch, int)
            and not isinstance(epoch, bool)
            and epoch > 0
            and isinstance(optimizer_state, Mapping)
            and optimizer_state.get("state_parameter_count")
            != optimizer_state.get("parameter_count")
        ):
            mismatches.append("resume checkpoint optimizer state is incomplete")
    return payload, mismatches


def _validate_checkpoint_contract(
    checkpoint_path: Path,
    config: Config,
    *,
    allow_mismatch: bool,
    require_resume_state: bool = False,
) -> dict | None:
    """Reject silent evaluation under a different projection/topology."""

    checkpoint = _load_checkpoint_without_executing_code(checkpoint_path)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    metadata = checkpoint.get("meta")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    rosa_mapping = config.model.get("head", {}).get("rosa_cfg")
    effective = _effective_rosa_config(config)
    target_has_rosa = rosa_mapping is not None
    checkpoint_state_dict = _checkpoint_state_dict(checkpoint)
    checkpoint_has_rosa = _state_dict_has_rosa(checkpoint_state_dict)
    compact_target = bool(
        target_has_rosa and effective.architecture_profile == "compact"
    )
    mismatches: list[str] = []
    integrity_mismatches: list[str] = []
    sealed_payload, sealed_mismatches = _sealed_checkpoint_mismatches(
        checkpoint,
        metadata,
        config,
        require_resume_state=require_resume_state,
    )
    integrity_mismatches.extend(sealed_mismatches)

    deploy_contract = metadata.get("deploy_contract")
    deploy_behavior = metadata.get("native_deploy_behavior")
    if deploy_contract is not None:
        if deploy_contract != _OBJECT_DEPTH_DEPLOY_CONTRACT:
            mismatches.append(f"unknown native deploy contract {deploy_contract!r}")
        else:
            mismatches.extend(_c73_deploy_payload_mismatches(checkpoint, config))
    elif deploy_behavior is not None:
        mismatches.append("native deploy behavior is missing its deploy contract")

    if checkpoint_has_rosa is not None and checkpoint_has_rosa != target_has_rosa:
        mismatch = (
            "topology "
            f"checkpoint ROSA={checkpoint_has_rosa} != "
            f"config ROSA={target_has_rosa}"
        )
        (integrity_mismatches if compact_target else mismatches).append(mismatch)

    checkpoint_contract = metadata.get("projection_contract")
    if target_has_rosa and checkpoint_has_rosa is not False:
        if checkpoint_contract is None:
            mismatches.append("ROSA checkpoint lacks projection_contract metadata")
        elif checkpoint_contract != effective.projection_contract:
            mismatches.append(
                "projection "
                f"{checkpoint_contract!r} != "
                f"{effective.projection_contract!r}"
            )

        checkpoint_version = metadata.get("rosa_contract_version")
        checkpoint_rfl_topology = metadata.get("rfl_topology_contract")
        expected_rfl_topology = effective.rfl_topology_contract
        if compact_target:
            if checkpoint_version != _ROSA_CONTRACT_VERSION:
                integrity_mismatches.append(
                    "factorized compact RFL requires checkpoint contract "
                    f"v{_ROSA_CONTRACT_VERSION}, got {checkpoint_version!r}"
                )
            if checkpoint_rfl_topology != expected_rfl_topology:
                integrity_mismatches.append(
                    "RFL topology "
                    f"{checkpoint_rfl_topology!r} != {expected_rfl_topology!r}"
                )
        elif (
            checkpoint_rfl_topology is not None
            and checkpoint_rfl_topology != expected_rfl_topology
        ):
            mismatches.append(
                "RFL topology "
                f"{checkpoint_rfl_topology!r} != {expected_rfl_topology!r}"
            )

    if compact_target:
        integrity_mismatches.extend(
            _factorized_refiner_state_mismatches(
                checkpoint_state_dict,
                effective,
            )
        )

    checkpoint_rosa_config = metadata.get("rosa_config")
    if target_has_rosa and isinstance(checkpoint_rosa_config, Mapping):
        from rosa import ROSAConfig

        try:
            source_effective = ROSAConfig.from_mapping(checkpoint_rosa_config)
        except (TypeError, ValueError) as error:
            mismatch = f"invalid checkpoint ROSA config ({error})"
            (integrity_mismatches if compact_target else mismatches).append(mismatch)
        else:
            if _topology_signature(source_effective) != _topology_signature(effective):
                mismatch = "ROSA learned topology differs from the checkpoint"
                (integrity_mismatches if compact_target else mismatches).append(
                    mismatch
                )
            if _geometry_signature(source_effective) != _geometry_signature(effective):
                mismatches.append(
                    "ROSA box-coordinate geometry differs from the checkpoint"
                )
    elif (
        target_has_rosa
        and metadata.get("architecture") == _KNOWN_RELEASE_ARCHITECTURE
        and effective.architecture_profile != "legacy"
    ):
        integrity_mismatches.append(
            "published checkpoint cannot initialize compact ROSA topology"
        )

    if integrity_mismatches:
        message = "checkpoint/config topology integrity mismatch: " + "; ".join(
            [*integrity_mismatches, *mismatches]
        )
        raise ValueError(message)
    if not mismatches:
        return sealed_payload
    message = "checkpoint/config contract mismatch: " + "; ".join(mismatches)
    if not allow_mismatch:
        raise ValueError(message)
    warnings.warn(message, stacklevel=2)
    return sealed_payload


def _compile_fast_inference(runner, config: Config) -> bool:
    """Compile only the latency-oriented sidecar after weights are loaded."""

    if _rosa_profile(config) is not None:
        _effective_rosa_config(config)
    if _rosa_profile(config) not in _FAST_PROFILES:
        return False
    from mmengine.model import is_model_wrapper

    model = runner.model.module if is_model_wrapper(runner.model) else runner.model
    refiner = getattr(getattr(model, "head", None), "rosa_refiner", None)
    if refiner is None:
        raise RuntimeError("fast profile is missing its ROSA refiner")
    if getattr(refiner, "_compiled_call_impl", None) is None:
        refiner.compile(mode="reduce-overhead", fullgraph=False)
    return True


def _resolve_train_resume_checkpoint(
    resume: str,
    config: Config,
) -> Path | None:
    """Resolve the exact local checkpoint MMEngine will resume from."""

    if resume != "auto":
        return Path(resume).expanduser().resolve()

    from mmengine.runner.checkpoint import find_latest_checkpoint

    discovered = find_latest_checkpoint(
        str(Path(config.work_dir).expanduser().resolve())
    )
    return None if discovered is None else Path(discovered).expanduser().resolve()


def _run_training(
    args: argparse.Namespace,
    config: Config,
    *,
    resume_payload_contract: Mapping | None = None,
) -> None:
    """Build and run MMEngine after any resume checkpoint has been sealed."""

    if args.amp:
        if config.optim_wrapper.type != "OptimWrapper":
            raise ValueError("--amp requires a plain OptimWrapper")
        config.optim_wrapper.type = "AmpOptimWrapper"
        config.optim_wrapper.setdefault("loss_scale", "dynamic")
    runner = _runner_class().from_cfg(config)
    provenance = None
    if getattr(runner, "rank", 0) == 0:
        from rosa.provenance import capture_external_provenance

        provenance = capture_external_provenance(
            Path(__file__).resolve().parents[1],
            runner.work_dir,
        )
    metadata = _checkpoint_contract_metadata(config) or {}
    _attach_runner_model_contract(metadata, runner)
    if provenance is not None:
        metadata["external_provenance"] = provenance
        metadata["source_revision"] = provenance["source_revision"]
    if resume_payload_contract is not None:
        runner.register_hook(
            _ResumeContractValidationHook(resume_payload_contract),
            priority="HIGHEST",
        )
    if metadata:
        runner.register_hook(
            _CheckpointContractHook(metadata),
            priority="VERY_LOW",
        )
    runner.train()


def train(args: argparse.Namespace, config: Config) -> None:
    if not args.resume:
        _run_training(args, config)
        return

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    resume_checkpoint = _resolve_train_resume_checkpoint(args.resume, config)
    if resume_checkpoint is None:
        # A missing auto checkpoint means a sealed fresh run. Disable Runner's
        # later discovery pass so a newly-created last_checkpoint is ignored.
        config.resume = False
        config.load_from = None
        _run_training(args, config)
        return

    # Keep this process-local descriptor alive until Runner has completed its
    # internal load_or_resume call. Replacing or overwriting the original path
    # after validation therefore cannot change the bytes MMEngine consumes.
    with _private_checkpoint_snapshot(resume_checkpoint) as snapshot:
        resume_payload_contract = _validate_checkpoint_contract(
            snapshot.path,
            config,
            allow_mismatch=False,
            require_resume_state=True,
        )
        config.resume = True
        config.load_from = str(snapshot.path)
        _run_training(
            args,
            config,
            resume_payload_contract=resume_payload_contract,
        )


def evaluate(args: argparse.Namespace, config: Config) -> None:
    _validate_checkpoint_contract(
        args.checkpoint.resolve(),
        config,
        allow_mismatch=args.allow_checkpoint_contract_mismatch,
    )
    # The contract check above has already parsed the checkpoint with a narrow
    # weights-only allowlist. MMEngine still calls torch.load without an
    # explicit weights_only argument, whose PyTorch 2.6+ default cannot restore
    # MMEngine's HistoryBuffer state. Permit that second, trusted local load.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    config.load_from = str(args.checkpoint.resolve())
    config.resume = False
    if args.dump:
        output = args.dump.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        dump_metric = dict(type="DumpResults", out_file_path=str(output))
        evaluator = config.test_evaluator
        config.test_evaluator = (
            [*evaluator, dump_metric]
            if isinstance(evaluator, (list, tuple))
            else [evaluator, dump_metric]
        )
    runner = _runner_class().from_cfg(config)
    runner.load_or_resume()
    _compile_fast_inference(runner, config)
    runner.test()


def benchmark(args: argparse.Namespace, config: Config) -> None:
    """Measure end-to-end single-GPU ``test_step`` latency."""

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires CUDA")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be nonnegative and iterations positive")
    # A baseline intentionally removes the sidecar, so its latency is
    # independent of the ROSA projection/profile contract.
    if args.checkpoint is not None and not args.baseline:
        _validate_checkpoint_contract(
            args.checkpoint.resolve(),
            config,
            allow_mismatch=args.allow_checkpoint_contract_mismatch,
        )
    if args.baseline:
        head_config = config.model.get("head")
        if head_config is None or "rosa_cfg" not in head_config:
            raise ValueError("--baseline requires a ROSA config")
        head_config.type = "Sparse4DHead"
        head_config.pop("rosa_cfg")
        head_config.pop("inference_box_source_by_dimension", None)
    if args.checkpoint is not None:
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        config.load_from = str(args.checkpoint.resolve())
    else:
        config.load_from = None
    config.resume = False
    if args.work_dir is None:
        config.work_dir = str(args.output.resolve().parent / "_benchmark")

    runner = _runner_class().from_cfg(config)
    runner.load_or_resume()
    compiled = _compile_fast_inference(runner, config)
    model = runner.model.eval()
    iterator = iter(runner.test_dataloader)
    timings: list[float] = []
    sample_count = 0
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for index in range(args.warmup + args.iterations):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(runner.test_dataloader)
                batch = next(iterator)
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = model.test_step(batch)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if index >= args.warmup:
                timings.append(elapsed)
                sample_count += len(output)

    total_time = sum(timings)
    profile = "baseline" if args.baseline else (_rosa_profile(config) or "none")
    result = {
        "config": str(args.config.resolve()),
        "profile": profile,
        "compiled_refiner": compiled,
        "checkpoint": (
            str(args.checkpoint.resolve()) if args.checkpoint is not None else None
        ),
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "samples": sample_count,
        "mean_batch_size": sample_count / len(timings),
        "mean_iteration_ms": 1000.0 * statistics.mean(timings),
        "median_iteration_ms": 1000.0 * statistics.median(timings),
        "p95_iteration_ms": 1000.0 * float(torch.tensor(timings).quantile(0.95)),
        "mean_latency_per_sample_ms": 1000.0 * total_time / sample_count,
        "samples_per_second": sample_count / total_time,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    if args.command == "export-deploy":
        export_deploy(args)
        return
    config = load_config(args)
    if args.command == "train":
        train(args, config)
    elif args.command == "evaluate":
        evaluate(args, config)
    else:
        benchmark(args, config)


if __name__ == "__main__":
    main()
