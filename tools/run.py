#!/usr/bin/env python3
"""Train, evaluate, or benchmark ROSA through MMEngine."""

from __future__ import annotations

import argparse
import builtins
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import tempfile
import time
import warnings

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.logging.history_buffer import HistoryBuffer


_FAST_PROFILES = frozenset({"fast", "compact_v2_fast"})
_KNOWN_RELEASE_ARCHITECTURE = "ROSA R18 Main126"
_ROSA_TOPOLOGY_FIELDS = (
    "architecture_profile",
    "embed_dims",
    "hidden_dims",
    "num_keypoints",
    "num_cameras",
    "num_levels",
    "enable_quality_score",
)
_ROSA_GEOMETRY_FIELDS = (
    "anchor_z_is_bottom",
)
_OBJECT_DEPTH_TRAIN_ONLY_KEYS = tuple(
    f"depth_branch.depth_layers.{level}.{field}"
    for level in range(3)
    for field in ("weight", "bias")
)
_OBJECT_DEPTH_TRAIN_ONLY_SHAPES = {
    f"depth_branch.depth_layers.{level}.weight": (1, 256, 1, 1)
    for level in range(3)
} | {
    f"depth_branch.depth_layers.{level}.bias": (1,)
    for level in range(3)
}
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
_OBJECT_DEPTH_DEPLOY_CONTRACT = (
    "rosa.native.object_lidar_depth.train_only_strip.v1"
)
_C73_DEPLOY_BEHAVIOR = "rosa.c73.official_sparse4d_gnn_residual.v1"
_NATIVE_DEPLOY_TARGET_HEADS = {
    _C73_DEPLOY_BEHAVIOR: "ROSAOfficialResidualNativeBaselineHead",
}
_C73_DEPLOY_DEFORMABLE_TYPES = frozenset(
    {
        "ROSAFullFOVDeformableFeatureAggregation",
        (
            "ROSAInferenceReuseExactHomogeneous4x4FullFOV"
            "DeformableFeatureAggregation"
        ),
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

    from mmengine.hooks import Hook

    class CheckpointContractHook(Hook):
        def __init__(self, hook_metadata: Mapping) -> None:
            self.metadata = dict(hook_metadata)

        def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
            del runner
            checkpoint.setdefault("meta", {}).update(self.metadata)

    return CheckpointContractHook(metadata)


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
    return (
        rosa_config.get("profile", "explicit")
        if rosa_config is not None
        else None
    )


def _effective_rosa_config(config: Config):
    rosa_mapping = config.model.get("head", {}).get("rosa_cfg")
    if rosa_mapping is None:
        return None

    from rosa import ROSAConfig

    return ROSAConfig.from_mapping(rosa_mapping)


def _checkpoint_contract_metadata(config: Config) -> dict | None:
    effective = _effective_rosa_config(config)
    if effective is None:
        return None
    return {
        "rosa_contract_version": 2,
        "projection_contract": effective.projection_contract,
        "refinement_profile": _rosa_profile(config),
        "rosa_config": asdict(effective),
    }


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

    depth_keys = {
        key for key in state_dict if key.startswith("depth_branch.")
    }
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
        (key, value)
        for key, value in state_dict.items()
        if key not in expected_keys
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
            "native_deploy_keypoint_generator_type": (
                _C73_DEPLOY_KEYPOINT_GENERATOR
            ),
            "native_deploy_projection_type": "ocam_fisheye",
            "native_deploy_anchor_z_is_bottom": False,
            "source_checkpoint_sha256": source_sha256,
            "source_state_keys": len(state_dict),
            "deploy_state_keys": len(deploy_state),
            "deploy_state_structure_sha256": (
                _OBJECT_DEPTH_DEPLOY_STRUCTURE_SHA256
            ),
            "deploy_state_metadata_sha256": (
                _OBJECT_DEPTH_DEPLOY_METADATA_SHA256
            ),
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
        "native_deploy_allowed_deformable_types": sorted(
            _C73_DEPLOY_DEFORMABLE_TYPES
        ),
        "native_deploy_keypoint_generator_type": (
            _C73_DEPLOY_KEYPOINT_GENERATOR
        ),
        "native_deploy_projection_type": "ocam_fisheye",
        "native_deploy_anchor_z_is_bottom": False,
        "source_state_keys": _OBJECT_DEPTH_SOURCE_STATE_KEYS,
        "deploy_state_keys": _OBJECT_DEPTH_DEPLOY_STATE_KEYS,
        "deploy_state_structure_sha256": (
            _OBJECT_DEPTH_DEPLOY_STRUCTURE_SHA256
        ),
        "deploy_state_metadata_sha256": (
            _OBJECT_DEPTH_DEPLOY_METADATA_SHA256
        ),
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
        mismatches.append(
            f"deploy behavior requires head {expected_head_type!r}"
        )
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


def _validate_checkpoint_contract(
    checkpoint_path: Path,
    config: Config,
    *,
    allow_mismatch: bool,
) -> None:
    """Reject silent evaluation under a different projection/topology."""

    checkpoint = _load_checkpoint_without_executing_code(checkpoint_path)
    if not isinstance(checkpoint, Mapping):
        return
    metadata = checkpoint.get("meta")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    rosa_mapping = config.model.get("head", {}).get("rosa_cfg")
    effective = _effective_rosa_config(config)
    target_has_rosa = rosa_mapping is not None
    checkpoint_has_rosa = _state_dict_has_rosa(
        _checkpoint_state_dict(checkpoint)
    )
    mismatches = []

    deploy_contract = metadata.get("deploy_contract")
    deploy_behavior = metadata.get("native_deploy_behavior")
    if deploy_contract is not None:
        if deploy_contract != _OBJECT_DEPTH_DEPLOY_CONTRACT:
            mismatches.append(f"unknown native deploy contract {deploy_contract!r}")
        else:
            mismatches.extend(_c73_deploy_payload_mismatches(checkpoint, config))
    elif deploy_behavior is not None:
        mismatches.append("native deploy behavior is missing its deploy contract")

    if (
        checkpoint_has_rosa is not None
        and checkpoint_has_rosa != target_has_rosa
    ):
        mismatches.append(
            "topology "
            f"checkpoint ROSA={checkpoint_has_rosa} != "
            f"config ROSA={target_has_rosa}"
        )

    checkpoint_contract = metadata.get("projection_contract")
    if target_has_rosa and checkpoint_has_rosa is not False:
        if checkpoint_contract is None:
            mismatches.append(
                "ROSA checkpoint lacks projection_contract metadata"
            )
        elif checkpoint_contract != effective.projection_contract:
            mismatches.append(
                "projection "
                f"{checkpoint_contract!r} != "
                f"{effective.projection_contract!r}"
            )

    checkpoint_rosa_config = metadata.get("rosa_config")
    if target_has_rosa and isinstance(checkpoint_rosa_config, Mapping):
        from rosa import ROSAConfig

        try:
            source_effective = ROSAConfig.from_mapping(checkpoint_rosa_config)
        except (TypeError, ValueError) as error:
            mismatches.append(f"invalid checkpoint ROSA config ({error})")
        else:
            if _topology_signature(source_effective) != _topology_signature(
                effective
            ):
                mismatches.append(
                    "ROSA learned topology differs from the checkpoint"
                )
            if _geometry_signature(source_effective) != _geometry_signature(
                effective
            ):
                mismatches.append(
                    "ROSA box-coordinate geometry differs from the checkpoint"
                )
    elif (
        target_has_rosa
        and metadata.get("architecture") == _KNOWN_RELEASE_ARCHITECTURE
        and effective.architecture_profile != "legacy"
    ):
        mismatches.append(
            "published checkpoint cannot initialize compact ROSA topology"
        )

    if not mismatches:
        return
    message = "checkpoint/config contract mismatch: " + "; ".join(mismatches)
    if not allow_mismatch:
        raise ValueError(message)
    warnings.warn(message, stacklevel=2)


def _compile_fast_inference(runner, config: Config) -> bool:
    """Compile only the latency-oriented sidecar after weights are loaded."""

    from mmengine.model import is_model_wrapper

    if _rosa_profile(config) not in _FAST_PROFILES:
        return False
    model = runner.model.module if is_model_wrapper(runner.model) else runner.model
    refiner = getattr(getattr(model, "head", None), "rosa_refiner", None)
    if refiner is None:
        raise RuntimeError("fast profile is missing its ROSA refiner")
    if getattr(refiner, "_compiled_call_impl", None) is None:
        refiner.compile(mode="reduce-overhead", fullgraph=False)
    return True


def train(args: argparse.Namespace, config: Config) -> None:
    from mmengine.runner import Runner

    if args.resume:
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        config.resume = True
        config.load_from = None if args.resume == "auto" else args.resume
    if args.amp:
        if config.optim_wrapper.type != "OptimWrapper":
            raise ValueError("--amp requires a plain OptimWrapper")
        config.optim_wrapper.type = "AmpOptimWrapper"
        config.optim_wrapper.setdefault("loss_scale", "dynamic")
    runner = Runner.from_cfg(config)
    provenance = None
    if getattr(runner, "rank", 0) == 0:
        from rosa.provenance import capture_external_provenance

        provenance = capture_external_provenance(
            Path(__file__).resolve().parents[1],
            runner.work_dir,
        )
    metadata = _checkpoint_contract_metadata(config) or {}
    if provenance is not None:
        metadata["external_provenance"] = provenance
        metadata["source_revision"] = provenance["source_revision"]
    if metadata:
        runner.register_hook(
            _CheckpointContractHook(metadata),
            priority="VERY_LOW",
        )
    runner.train()


def evaluate(args: argparse.Namespace, config: Config) -> None:
    from mmengine.runner import Runner

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
    runner = Runner.from_cfg(config)
    runner.load_or_resume()
    _compile_fast_inference(runner, config)
    runner.test()


def benchmark(args: argparse.Namespace, config: Config) -> None:
    """Measure end-to-end single-GPU ``test_step`` latency."""

    from mmengine.runner import Runner

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

    runner = Runner.from_cfg(config)
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
    profile = (
        "baseline"
        if args.baseline
        else (_rosa_profile(config) or "none")
    )
    result = {
        "config": str(args.config.resolve()),
        "profile": profile,
        "compiled_refiner": compiled,
        "checkpoint": (
            str(args.checkpoint.resolve())
            if args.checkpoint is not None
            else None
        ),
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "samples": sample_count,
        "mean_batch_size": sample_count / len(timings),
        "mean_iteration_ms": 1000.0 * statistics.mean(timings),
        "median_iteration_ms": 1000.0 * statistics.median(timings),
        "p95_iteration_ms": 1000.0
        * float(torch.tensor(timings).quantile(0.95)),
        "mean_latency_per_sample_ms": 1000.0 * total_time / sample_count,
        "samples_per_second": sample_count / total_time,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated()
        / (1024.0 * 1024.0),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
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
