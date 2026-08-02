#!/usr/bin/env python3
"""Create a non-resumable CP16 benchmark fixture from the zero-RFL smoke run.

This is deliberately not a general checkpoint converter.  The dense compact
RFL and signed-gated CP16 RFL are different learned functions, so trained
non-zero dense weights cannot be converted faithfully.  The one approved
case is the epoch-1 smoke checkpoint whose dense RFL output projection is
exactly zero.  Both topologies then produce an exact zero residual regardless
of the newly initialized ray factor, which makes the result useful for runtime
plumbing and latency comparisons only.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Mapping
import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
DEFAULT_NATIVE_CONFIG = ROOT / (
    "configs/rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_"
    "cpu_metadata_lean_loader_rgb_uint8_transfer_pinned_inplace_"
    "homogeneous4x4_inference_reuse_exact.py"
)

SOURCE_CONTRACT_VERSION = 2
TARGET_CONTRACT_VERSION = 3
SOURCE_PROFILE = "balanced"
TARGET_TOPOLOGY = "rosa.rfl.signed_gated_cp16.v3"
TARGET_HEAD = "ROSAOfficialResidualRefinementHead"
MODEL_BUILD_SEED = 3407
RAY_INITIALIZATION_SEED = 3407

RFL_PREFIX = "head.rosa_refiner.rosa.rfl."
APPEARANCE_KEY = f"{RFL_PREFIX}appearance_proj.weight"
RAY_KEY = f"{RFL_PREFIX}ray_proj.weight"
OUTPUT_KEY = f"{RFL_PREFIX}output_proj.weight"
SOURCE_APPEARANCE_SHAPE = (16, 256)
SOURCE_OUTPUT_SHAPE = (256, 144)
TARGET_RAY_SHAPE = (16, 9)
TARGET_OUTPUT_SHAPE = (256, 16)


class ConversionError(ValueError):
    """The input is not the single zero-output topology conversion we allow."""


def _load_run_tool():
    path = Path(__file__).resolve().with_name("run.py")
    spec = importlib.util.spec_from_file_location("rosa_run_for_cp16_fixture", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the repository checkpoint helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _state_schema(state_dict: Mapping[str, torch.Tensor]) -> list[dict[str, Any]]:
    records = []
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ConversionError("state_dict must map string keys to tensors")
        records.append(
            {
                "key": name,
                "shape": [int(value) for value in tensor.shape],
                "dtype": str(tensor.dtype),
                "layout": str(tensor.layout),
            }
        )
    return records


def _state_value_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if tensor.layout != torch.strided:
            raise ConversionError(f"state tensor {name!r} is not strided")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _copy_metadata(source, target) -> None:
    metadata = getattr(source, "_metadata", None)
    if metadata is not None:
        target._metadata = copy.deepcopy(metadata)


def _require_tensor(
    state_dict: Mapping[str, torch.Tensor],
    name: str,
    shape: tuple[int, ...],
) -> torch.Tensor:
    value = state_dict.get(name)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        observed = None if value is None else tuple(getattr(value, "shape", ()))
        raise ConversionError(f"state {name!r} has shape {observed}, expected {shape}")
    if (
        value.dtype is not torch.float32
        or value.layout != torch.strided
        or value.is_quantized
    ):
        raise ConversionError(f"state {name!r} must be a plain FP32 tensor")
    return value


def convert_state_dict(
    source_state: Mapping[str, torch.Tensor],
    target_template: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Copy every unchanged tensor and replace only the zero RFL topology."""

    if not isinstance(source_state, Mapping) or not isinstance(
        target_template, Mapping
    ):
        raise TypeError("source and target state_dict values must be mappings")

    appearance = _require_tensor(
        source_state,
        APPEARANCE_KEY,
        SOURCE_APPEARANCE_SHAPE,
    )
    dense_output = _require_tensor(source_state, OUTPUT_KEY, SOURCE_OUTPUT_SHAPE)
    if RAY_KEY in source_state:
        raise ConversionError("source already contains a ray projection")
    if not torch.isfinite(appearance).all():
        raise ConversionError("source appearance projection is not finite")
    if torch.count_nonzero(dense_output).item() != 0:
        raise ConversionError(
            "dense RFL output projection is non-zero and cannot be converted"
        )

    _require_tensor(target_template, APPEARANCE_KEY, SOURCE_APPEARANCE_SHAPE)
    _require_tensor(target_template, RAY_KEY, TARGET_RAY_SHAPE)
    _require_tensor(target_template, OUTPUT_KEY, TARGET_OUTPUT_SHAPE)

    source_keys = set(source_state)
    target_keys = set(target_template)
    if source_keys - target_keys:
        unexpected = sorted(source_keys - target_keys)
        raise ConversionError(
            f"source has keys outside the target schema: {unexpected}"
        )
    if target_keys - source_keys != {RAY_KEY}:
        missing = sorted(target_keys - source_keys)
        raise ConversionError(
            "target differs by more than the CP16 ray projection: " f"{missing}"
        )

    converted: OrderedDict[str, torch.Tensor] = OrderedDict()
    ray_generator = torch.Generator(device="cpu")
    ray_generator.manual_seed(RAY_INITIALIZATION_SEED)
    for name, target_value in target_template.items():
        if not isinstance(target_value, torch.Tensor):
            raise ConversionError(f"target state {name!r} is not a tensor")
        target_value = target_value.detach().cpu()
        if name == RAY_KEY:
            value = torch.empty_like(target_value)
            torch.nn.init.xavier_uniform_(value, generator=ray_generator)
        elif name == OUTPUT_KEY:
            value = torch.zeros_like(target_value)
        else:
            source_value = source_state.get(name)
            if not isinstance(source_value, torch.Tensor):
                raise ConversionError(f"source is missing target state {name!r}")
            source_value = source_value.detach().cpu()
            if (
                source_value.shape != target_value.shape
                or source_value.dtype != target_value.dtype
                or source_value.layout != target_value.layout
            ):
                raise ConversionError(
                    f"unchanged state {name!r} differs: "
                    f"source={tuple(source_value.shape)}/{source_value.dtype}/"
                    f"{source_value.layout}, target={tuple(target_value.shape)}/"
                    f"{target_value.dtype}/{target_value.layout}"
                )
            value = source_value.clone(memory_format=torch.preserve_format)
        converted[name] = value

    _copy_metadata(target_template, converted)
    if torch.count_nonzero(converted[OUTPUT_KEY]).item() != 0:
        raise AssertionError("converted CP16 output projection must remain zero")
    if not torch.isfinite(converted[RAY_KEY]).all():
        raise AssertionError("deterministic CP16 ray projection is not finite")
    return converted


def _validate_source_checkpoint(checkpoint: Mapping) -> tuple[Mapping, Mapping]:
    if not isinstance(checkpoint, Mapping):
        raise ConversionError("checkpoint root must be a mapping")
    state = checkpoint.get("state_dict")
    metadata = checkpoint.get("meta")
    if not isinstance(state, Mapping) or not isinstance(metadata, Mapping):
        raise ConversionError("checkpoint requires state_dict and meta mappings")
    training_fields = {"optimizer", "param_schedulers", "message_hub"}
    if not training_fields.issubset(checkpoint):
        raise ConversionError(
            "source smoke checkpoint is missing its expected training payload"
        )
    if metadata.get("rosa_contract_version") != SOURCE_CONTRACT_VERSION:
        raise ConversionError("source must use the dense compact v2 contract")
    if metadata.get("refinement_profile") != SOURCE_PROFILE:
        raise ConversionError("source must be the balanced smoke profile")
    if metadata.get("rfl_topology_contract") is not None:
        raise ConversionError("source unexpectedly declares a modern RFL topology")
    if metadata.get("projection_contract") != "full_fov":
        raise ConversionError("source must use the full-FOV projection contract")

    from rosa.config import ROSAConfig

    source_config = metadata.get("rosa_config")
    if not isinstance(source_config, Mapping):
        raise ConversionError("source is missing meta.rosa_config")
    observed = asdict(ROSAConfig.from_mapping(source_config))
    expected = asdict(ROSAConfig.from_mapping({"profile": SOURCE_PROFILE}))
    if observed != expected:
        raise ConversionError("source ROSA behavior is not the balanced profile")
    return state, metadata


def _build_target(config_path: Path):
    from mmengine.config import Config
    from mmengine.utils import import_modules_from_strings
    from mmdet3d.registry import MODELS
    from mmdet3d.utils import register_all_modules
    from rosa.config import ROSAConfig

    config_path = config_path.expanduser().resolve(strict=True)
    config = Config.fromfile(str(config_path))
    register_all_modules(init_default_scope=True)
    import_modules_from_strings(**config.custom_imports)

    model_config = copy.deepcopy(config.model)
    model_config.head.type = TARGET_HEAD
    model_config.head.rosa_cfg = {"profile": SOURCE_PROFILE}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(MODEL_BUILD_SEED)
        model = MODELS.build(model_config)
    model.cpu().eval()
    if type(model.head).__name__ != TARGET_HEAD:
        raise ConversionError("target did not build the exact C32 ROSA head")
    effective = ROSAConfig.from_mapping(model_config.head.rosa_cfg)
    if effective.rfl_topology_contract != TARGET_TOPOLOGY:
        raise ConversionError("target config did not build signed-gated CP16")
    return model, model_config, effective, config_path


def _target_metadata(
    *,
    source_metadata: Mapping,
    source_sha256: str,
    source_byte_count: int,
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    target_config,
    effective,
    config_path: Path,
) -> dict[str, Any]:
    from rosa.provenance import read_source_revision

    source_epoch = source_metadata.get("epoch")
    source_iteration = source_metadata.get("iter")
    return {
        "rosa_contract_version": TARGET_CONTRACT_VERSION,
        "rfl_topology_contract": TARGET_TOPOLOGY,
        "projection_contract": effective.projection_contract,
        "refinement_profile": SOURCE_PROFILE,
        "rosa_config": asdict(effective),
        "benchmark_only": True,
        "resumable": False,
        "weights_only": True,
        "fixture_contract": "rosa.cp16.zero-rfl-benchmark-fixture.v1",
        "conversion_provenance": {
            "source_sha256": source_sha256,
            "source_byte_count": int(source_byte_count),
            "source_epoch": source_epoch,
            "source_iter": source_iteration,
            "source_contract_version": SOURCE_CONTRACT_VERSION,
            "source_refinement_profile": SOURCE_PROFILE,
            "source_rfl_topology": "rosa.rfl.dense_outer_product.v2",
            "target_rfl_topology": TARGET_TOPOLOGY,
            "target_head": TARGET_HEAD,
            "target_native_config": str(config_path),
            "target_model_config_sha256": _canonical_sha256(target_config.to_dict()),
            "repository_revision": read_source_revision(ROOT),
            "converter_sha256": _sha256_file(Path(__file__).resolve()),
            "model_build_seed": MODEL_BUILD_SEED,
            "ray_initialization_seed": RAY_INITIALIZATION_SEED,
            "ray_initialization": "torch.nn.init.xavier_uniform_(gain=1.0)",
            "conversion_rule": (
                "copy every identical key/shape/dtype/layout; require the dense "
                "256x144 output projection to be exactly zero; replace it with "
                "an exact-zero 256x16 output projection; add only a deterministic "
                "16x9 ray projection"
            ),
            "stripped_root_fields": [
                "message_hub",
                "optimizer",
                "param_schedulers",
            ],
            "source_state_key_count": len(source_state),
            "target_state_key_count": len(target_state),
            "source_state_schema_sha256": _canonical_sha256(
                _state_schema(source_state)
            ),
            "target_state_schema_sha256": _canonical_sha256(
                _state_schema(target_state)
            ),
            "target_state_value_sha256": _state_value_sha256(target_state),
        },
    }


def _atomic_save(payload: Mapping, output: Path, *, overwrite: bool) -> Path:
    output = output.expanduser()
    if not output.name or output.name in (".", ".."):
        raise ValueError("output must name a checkpoint file")
    parent = output.parent.resolve(strict=True)
    parent_stat = parent.stat()
    if not stat.S_ISDIR(parent_stat.st_mode):
        raise ValueError("output parent must be a directory")
    destination = parent / output.name
    destination_exists = os.path.lexists(destination)
    if destination_exists and not overwrite:
        raise FileExistsError(f"output already exists: {destination}")
    if destination_exists:
        destination_stat = destination.lstat()
        if not stat.S_ISREG(destination_stat.st_mode):
            raise ValueError("existing output must be a regular file")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
            temporary.unlink()
        directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()
    return destination


def convert_checkpoint(
    source: Path,
    output: Path,
    *,
    expected_source_sha256: str,
    native_config: Path = DEFAULT_NATIVE_CONFIG,
    overwrite: bool = False,
) -> dict[str, Any]:
    expected_source_sha256 = expected_source_sha256.strip()
    if len(expected_source_sha256) != 64 or any(
        value not in "0123456789abcdef" for value in expected_source_sha256
    ):
        raise ValueError("expected source SHA256 must be 64 lowercase hex characters")

    source_path = source.expanduser().resolve(strict=True)
    output_parent = output.expanduser().parent.resolve(strict=True)
    output_path = output_parent / output.name
    if output_path == source_path:
        raise ValueError("output must not replace the source checkpoint")

    run_tool = _load_run_tool()
    with run_tool._private_checkpoint_snapshot(source_path) as snapshot:
        if snapshot.sha256 != expected_source_sha256:
            raise ConversionError(
                "source SHA256 differs from the explicit expectation: "
                f"{snapshot.sha256}"
            )
        checkpoint = run_tool._load_checkpoint_without_executing_code(snapshot.path)
        source_state, source_metadata = _validate_source_checkpoint(checkpoint)

        model, model_config, effective, config_path = _build_target(native_config)
        target_template = model.state_dict()
        converted_state = convert_state_dict(source_state, target_template)
        load_result = model.load_state_dict(converted_state, strict=True)
        if load_result.missing_keys or load_result.unexpected_keys:
            raise AssertionError("strict target state validation unexpectedly failed")

        validated_state = OrderedDict(
            (name, value.detach().cpu().clone(memory_format=torch.preserve_format))
            for name, value in model.state_dict().items()
        )
        _copy_metadata(model.state_dict(), validated_state)
        for name, converted_value in converted_state.items():
            if not torch.equal(validated_state[name], converted_value):
                raise AssertionError(f"strict load changed converted state {name!r}")

        metadata = _target_metadata(
            source_metadata=source_metadata,
            source_sha256=snapshot.sha256,
            source_byte_count=snapshot.byte_count,
            source_state=source_state,
            target_state=validated_state,
            target_config=model_config,
            effective=effective,
            config_path=config_path,
        )
        payload = {"meta": metadata, "state_dict": validated_state}
        destination = _atomic_save(payload, output, overwrite=overwrite)

    output_sha256 = _sha256_file(destination)
    return {
        "output": str(destination),
        "output_sha256": output_sha256,
        "output_byte_count": destination.stat().st_size,
        "source_sha256": expected_source_sha256,
        "state_key_count": len(validated_state),
        "state_value_sha256": metadata["conversion_provenance"][
            "target_state_value_sha256"
        ],
        "benchmark_only": True,
        "resumable": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--native-config", type=Path, default=DEFAULT_NATIVE_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = convert_checkpoint(
        args.source,
        args.output,
        expected_source_sha256=args.expected_source_sha256,
        native_config=args.native_config,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
