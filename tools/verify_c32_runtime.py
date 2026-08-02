#!/usr/bin/env python3
"""Disarmed-by-default paired runtime verification for C32 CPU metadata.

The default invocation performs CPU-only static validation and prints the
frozen plan.  It does not construct a model, build a dataloader, inspect CUDA,
or create an output directory.  Execution requires both ``--execute`` and the
exact idle-GPU confirmation token shown by ``--help``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import gc
import importlib.metadata
import importlib.util
import json
import operator
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch
from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rosa.runtime_verification import (  # noqa: E402
    CANDIDATE_ARM,
    HOST_METADATA_KEYS,
    PARENT_ARM,
    VerificationContractError,
    build_alternating_schedule,
    build_batch_manifest,
    canonical_sha256,
    file_sha256,
    raw_fingerprint,
    require_same_fingerprint,
    summarize_timings,
    summarize_transfer_event_names,
    validate_config_pair,
    validate_expected_manifest,
    validate_runtime_parameters,
    validate_transfer_reduction,
)


IDLE_GPU_CONFIRMATION = "I_HAVE_VERIFIED_THE_GPU_IS_IDLE_AND_RESERVED"
DEFAULT_PARENT_CONFIG = (
    ROOT / "configs/rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual.py"
)
DEFAULT_CANDIDATE_CONFIG = ROOT / (
    "configs/"
    "rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_cpu_metadata.py"
)


@dataclass
class PreparedVerification:
    args: argparse.Namespace
    parent_config: Config
    candidate_config: Config
    checkpoint: Mapping[str, Any]
    state_dict: Mapping[str, torch.Tensor]
    expected_manifest: Mapping[str, Any] | None
    plan: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-config", type=Path, default=DEFAULT_PARENT_CONFIG)
    parser.add_argument(
        "--candidate-config", type=Path, default=DEFAULT_CANDIDATE_CONFIG
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-sample-manifest", type=Path)
    parser.add_argument("--capture-batches", type=int, default=2)
    parser.add_argument("--expected-batch-size", type=int, default=4)
    parser.add_argument("--profile-batch-index", type=int, default=0)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--measured-rounds", type=int, default=10)
    parser.add_argument("--dataloader-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minimum-dtoh-reduction", type=int, default=12)
    parser.add_argument("--minimum-htod-reduction", type=int, default=3)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Arm model construction, CUDA parity, profiling, and timing.",
    )
    parser.add_argument(
        "--idle-gpu-confirmation",
        help=(
            "Required with --execute; must equal "
            f"{IDLE_GPU_CONFIRMATION!r}."
        ),
    )
    return parser


def _load_run_tool():
    specification = importlib.util.spec_from_file_location(
        "rosa_run_tool_for_runtime_verification", ROOT / "tools/run.py"
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load tools/run.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


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


def _checkpoint_schema(state_dict: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    schema = []
    for name, value in state_dict.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise VerificationContractError(
                "checkpoint state_dict must map strings to tensors"
            )
        schema.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "numel": value.numel(),
            }
        )
    if not schema:
        raise VerificationContractError("checkpoint state_dict is empty")
    return {
        "keys": len(schema),
        "parameters_and_buffers_numel": sum(item["numel"] for item in schema),
        "sha256": canonical_sha256(schema),
    }


def prepare_verification(args: argparse.Namespace) -> PreparedVerification:
    """Perform the complete CPU-only planning and frozen-input validation."""

    parameters = validate_runtime_parameters(
        capture_batches=args.capture_batches,
        warmup_rounds=args.warmup_rounds,
        measured_rounds=args.measured_rounds,
        expected_batch_size=args.expected_batch_size,
        profile_batch_index=args.profile_batch_index,
    )
    try:
        dataloader_workers = operator.index(args.dataloader_workers)
    except TypeError as error:
        raise VerificationContractError(
            "dataloader_workers must be an exact integer"
        ) from error
    if isinstance(args.dataloader_workers, bool) or dataloader_workers < 0:
        raise VerificationContractError("dataloader_workers must be nonnegative")
    try:
        seed = operator.index(args.seed)
    except TypeError as error:
        raise VerificationContractError("seed must be an exact integer") from error
    if isinstance(args.seed, bool) or seed < 0:
        raise VerificationContractError("seed must be nonnegative")
    try:
        minimum_dtoh = operator.index(args.minimum_dtoh_reduction)
        minimum_htod = operator.index(args.minimum_htod_reduction)
    except TypeError as error:
        raise VerificationContractError(
            "transfer minima must be exact integers"
        ) from error
    if (
        isinstance(args.minimum_dtoh_reduction, bool)
        or isinstance(args.minimum_htod_reduction, bool)
        or minimum_dtoh < 0
        or minimum_htod < 0
    ):
        raise VerificationContractError("transfer minima must be nonnegative")

    parent_path = _resolve_file(args.parent_config, "parent config")
    candidate_path = _resolve_file(args.candidate_config, "candidate config")
    checkpoint_path = _resolve_file(args.checkpoint, "checkpoint")
    parent_config = Config.fromfile(parent_path, import_custom_modules=False)
    candidate_config = Config.fromfile(candidate_path, import_custom_modules=False)
    config_contract = validate_config_pair(
        parent_config,
        candidate_config,
        expected_batch_size=parameters["expected_batch_size"],
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
            "C32 native runtime verification requires a checkpoint without ROSA state"
        )
    checkpoint_schema = _checkpoint_schema(state_dict)

    expected_manifest = None
    expected_manifest_path = None
    if args.expected_sample_manifest is not None:
        expected_manifest_path = _resolve_file(
            args.expected_sample_manifest, "expected sample manifest"
        )
        expected_manifest = _load_json_mapping(
            expected_manifest_path, "expected sample manifest"
        )

    warmup_schedule = build_alternating_schedule(
        rounds=parameters["warmup_rounds"],
        batch_count=parameters["capture_batches"],
        phase="warmup",
    )
    measured_schedule = build_alternating_schedule(
        rounds=parameters["measured_rounds"],
        batch_count=parameters["capture_batches"],
        phase="measured",
    )
    schedule_payload = {
        "warmup": [entry.to_dict() for entry in warmup_schedule],
        "measured": [entry.to_dict() for entry in measured_schedule],
    }
    plan = {
        "schema_version": 1,
        "status": "DRY_RUN",
        "execution_requested": bool(args.execute),
        "cuda_queried": False,
        "parent_config": {
            "path": str(parent_path),
            "sha256": file_sha256(parent_path),
        },
        "candidate_config": {
            "path": str(candidate_path),
            "sha256": file_sha256(candidate_path),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
            "schema": checkpoint_schema,
        },
        "expected_sample_manifest": (
            None
            if expected_manifest_path is None
            else {
                "path": str(expected_manifest_path),
                "sha256": file_sha256(expected_manifest_path),
            }
        ),
        "output_dir": str(args.output_dir.expanduser().resolve()),
        "device": str(args.device),
        "seed": seed,
        "dataloader_workers": dataloader_workers,
        "parameters": parameters,
        "minimum_transfer_reduction": {
            "device_to_host": minimum_dtoh,
            "host_to_device": minimum_htod,
        },
        "config_contract": config_contract,
        "schedule": schedule_payload,
        "schedule_sha256": canonical_sha256(schedule_payload),
    }
    return PreparedVerification(
        args=args,
        parent_config=parent_config,
        candidate_config=candidate_config,
        checkpoint=checkpoint,
        state_dict=state_dict,
        expected_manifest=expected_manifest,
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


def _git_snapshot(repository: Path, *, require_clean: bool) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout.strip()

    revision = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    status = git("status", "--porcelain", "--untracked-files=all")
    if require_clean and status:
        raise VerificationContractError(
            "runtime verification repository must be clean before execution"
        )
    return {
        "root": str(repository.resolve()),
        "revision": revision,
        "tree": tree,
        "clean": not bool(status),
        "status_sha256": canonical_sha256(status.splitlines()),
    }


def _seed_everything(seed: int, *, cuda: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def _capture_rng(*, cuda: bool) -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
    }
    if cuda:
        state["torch_cuda"] = [value.clone() for value in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng(state: Mapping[str, Any], *, cuda: bool) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if cuda:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _rng_digest(state: Mapping[str, Any]) -> dict[str, Any]:
    return raw_fingerprint(state)


def _clone_batch(batch: Any) -> Any:
    return copy.deepcopy(batch)


def _reset_runtime_state(model: torch.nn.Module) -> None:
    head = getattr(model, "head", None)
    bank = getattr(head, "instance_bank", None)
    reset = getattr(bank, "reset", None)
    if callable(reset):
        reset()


def _model_contract(model: torch.nn.Module) -> dict[str, Any]:
    return {
        "type": f"{type(model).__module__}.{type(model).__qualname__}",
        "parameters": sum(value.numel() for value in model.parameters()),
        "trainable_parameters": sum(
            value.numel() for value in model.parameters() if value.requires_grad
        ),
        "buffers": sum(value.numel() for value in model.buffers()),
        "state_keys": list(model.state_dict()),
    }


def _build_models(prepared: PreparedVerification, device: torch.device):
    from mmengine.utils import import_modules_from_strings
    from mmdet3d.registry import MODELS
    from mmdet3d.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    import_modules_from_strings(**prepared.parent_config.custom_imports)
    import_modules_from_strings(**prepared.candidate_config.custom_imports)
    models = {}
    contracts = {}
    for label, config in (
        (PARENT_ARM, prepared.parent_config),
        (CANDIDATE_ARM, prepared.candidate_config),
    ):
        _seed_everything(prepared.plan["seed"], cuda=True)
        model = MODELS.build(copy.deepcopy(config.model))
        model.load_state_dict(prepared.state_dict, strict=True)
        contract = _model_contract(model)
        model.to(device=device)
        model.eval()
        models[label] = model
        contracts[label] = contract
    if contracts[PARENT_ARM] != contracts[CANDIDATE_ARM]:
        raise VerificationContractError(
            "parent/candidate model state schema or parameter counts differ"
        )
    return models, contracts


def _build_dataloader(prepared: PreparedVerification):
    from mmengine.runner import Runner

    dataloader_config = copy.deepcopy(prepared.parent_config.test_dataloader)
    workers = prepared.plan["dataloader_workers"]
    dataloader_config["num_workers"] = workers
    dataloader_config["persistent_workers"] = bool(workers)
    return Runner.build_dataloader(
        dataloader_config,
        seed=prepared.plan["seed"],
        diff_rank_seed=False,
    )


def _capture_batches(prepared: PreparedVerification):
    dataloader = _build_dataloader(prepared)
    iterator = iter(dataloader)
    batches = []
    manifests = []
    observed_identities = set()
    for batch_index in range(prepared.plan["parameters"]["capture_batches"]):
        try:
            batch = next(iterator)
        except StopIteration as error:
            raise VerificationContractError(
                "test dataloader ended before all preregistered batches were captured"
            ) from error
        manifest = build_batch_manifest(
            batch,
            batch_index=batch_index,
            expected_batch_size=prepared.plan["parameters"]["expected_batch_size"],
        )
        for identity in manifest["sample_identities"]:
            digest = canonical_sha256(identity)
            if digest in observed_identities:
                raise VerificationContractError("captured sample identities repeat")
            observed_identities.add(digest)
        batches.append(batch)
        manifests.append(manifest)
    manifest_document = {
        "schema_version": 1,
        "batches": manifests,
    }
    manifest_document["sha256"] = canonical_sha256(manifest_document["batches"])
    if prepared.expected_manifest is not None:
        validate_expected_manifest(manifest_document, prepared.expected_manifest)
    return batches, manifest_document


@torch.inference_mode()
def _metadata_snapshot(model: torch.nn.Module, batch: Any) -> dict[str, Any]:
    processed = model.data_preprocessor(_clone_batch(batch), training=False)
    if not isinstance(processed, Mapping):
        raise VerificationContractError("data preprocessor did not return a mapping")
    inputs = processed.get("inputs")
    samples = processed.get("data_samples")
    if not isinstance(inputs, Mapping):
        raise VerificationContractError("processed inputs are not a mapping")
    metas = model._build_metas(inputs, samples)
    if "img_metas" not in metas:
        raise VerificationContractError("Sparse4D internal metas lack img_metas")
    host_values = {name: inputs[name] for name in HOST_METADATA_KEYS}
    snapshot = {
        "img_metas": raw_fingerprint(metas["img_metas"]),
        "host_values": raw_fingerprint(host_values),
        "host_devices": {name: str(inputs[name].device) for name in HOST_METADATA_KEYS},
        "host_dtypes": {name: str(inputs[name].dtype) for name in HOST_METADATA_KEYS},
        "host_shapes": {name: list(inputs[name].shape) for name in HOST_METADATA_KEYS},
    }
    del metas, processed
    return snapshot


def _verify_metadata(models: Mapping[str, torch.nn.Module], batches: Sequence[Any]):
    records = []
    for batch_index, batch in enumerate(batches):
        parent = _metadata_snapshot(models[PARENT_ARM], batch)
        candidate = _metadata_snapshot(models[CANDIDATE_ARM], batch)
        for key in ("img_metas", "host_values", "host_dtypes", "host_shapes"):
            if parent[key] != candidate[key]:
                raise VerificationContractError(
                    f"internal metadata mismatch for batch {batch_index}: {key}"
                )
        if not all(
            device.startswith("cuda") for device in parent["host_devices"].values()
        ):
            raise VerificationContractError(
                "parent temporal metadata did not follow the CUDA baseline path"
            )
        if set(candidate["host_devices"].values()) != {"cpu"}:
            raise VerificationContractError(
                "candidate temporal metadata did not remain on CPU"
            )
        records.append(
            {
                "batch_index": batch_index,
                "parent": parent,
                "candidate": candidate,
            }
        )
    return records


def _runtime_state_snapshot(model: torch.nn.Module) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    bank_names = {
        "cached_feature",
        "cached_anchor",
        "metas",
        "mask",
        "confidence",
        "temp_confidence",
        "instance_id",
        "prev_id",
    }
    for module_name, module in model.named_modules():
        for name, value in vars(module).items():
            if not (name.startswith("_last_") or name in bank_names):
                continue
            path = f"{module_name}.{name}" if module_name else name
            try:
                captured[path] = raw_fingerprint(value)
            except VerificationContractError:
                captured[path] = {
                    "unsupported_type": (
                        f"{type(value).__module__}.{type(value).__qualname__}"
                    )
                }
    return captured


@torch.inference_mode()
def _run_output_sequence(model: torch.nn.Module, batches: Sequence[Any]):
    output_records = []
    for batch_index, batch in enumerate(batches):
        output = model.test_step(_clone_batch(batch))
        output_records.append(
            {
                "batch_index": batch_index,
                "output": raw_fingerprint(output),
                "runtime_state": _runtime_state_snapshot(model),
                "sample_count": len(output),
            }
        )
        del output
    return output_records


def _verify_outputs_and_state(
    models: Mapping[str, torch.nn.Module], batches: Sequence[Any]
) -> dict[str, Any]:
    for model in models.values():
        _reset_runtime_state(model)
    initial_rng = _capture_rng(cuda=True)
    _restore_rng(initial_rng, cuda=True)
    parent = _run_output_sequence(models[PARENT_ARM], batches)
    parent_rng = _capture_rng(cuda=True)
    _restore_rng(initial_rng, cuda=True)
    candidate = _run_output_sequence(models[CANDIDATE_ARM], batches)
    candidate_rng = _capture_rng(cuda=True)
    if parent != candidate:
        raise VerificationContractError(
            "parent/candidate raw output or runtime-state fingerprints differ"
        )
    require_same_fingerprint("post-inference RNG", parent_rng, candidate_rng)
    _restore_rng(initial_rng, cuda=True)
    return {
        "batches": parent,
        "initial_rng": _rng_digest(initial_rng),
        "final_rng": _rng_digest(parent_rng),
    }


@torch.inference_mode()
def _profile_arm(
    label: str,
    model: torch.nn.Module,
    batch: Any,
    trace_path: Path,
) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    _reset_runtime_state(model)
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as profiler:
        output = model.test_step(_clone_batch(batch))
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(trace_path))
    event_names = [event.name for event in profiler.events()]
    summary = summarize_transfer_event_names(event_names)
    summary.update(
        {
            "arm": label,
            "events": len(event_names),
            "sample_count": len(output),
            "trace": str(trace_path),
            "trace_sha256": file_sha256(trace_path),
        }
    )
    del output
    return summary


@torch.inference_mode()
def _memory_probe(model: torch.nn.Module, batch: Any) -> dict[str, int]:
    _reset_runtime_state(model)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = model.test_step(_clone_batch(batch))
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    after = torch.cuda.memory_allocated()
    sample_count = len(output)
    del output
    torch.cuda.synchronize()
    return {
        "allocated_before_bytes": before,
        "allocated_after_output_bytes": after,
        "peak_allocated_bytes": peak,
        "peak_increment_over_before_bytes": max(0, peak - before),
        "sample_count": sample_count,
    }


@torch.inference_mode()
def _run_schedule(
    schedule,
    models: Mapping[str, torch.nn.Module],
    batches: Sequence[Any],
    *,
    expected_batch_size: int,
) -> list[dict[str, Any]]:
    records = []
    for sequence_index, entry in enumerate(schedule):
        batch = _clone_batch(batches[entry.batch_index])
        model = models[entry.arm]
        torch.cuda.synchronize()
        start_ns = time.perf_counter_ns()
        output = model.test_step(batch)
        torch.cuda.synchronize()
        elapsed_ns = time.perf_counter_ns() - start_ns
        sample_count = len(output)
        if sample_count != expected_batch_size:
            raise VerificationContractError(
                f"timed output has {sample_count} samples, expected "
                f"{expected_batch_size}"
            )
        record = entry.to_dict()
        record.update(
            {
                "sequence_index": sequence_index,
                "elapsed_ns": elapsed_ns,
                "sample_count": sample_count,
            }
        )
        records.append(record)
        del output, batch
    return records


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment(device: torch.device) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "packages": {
            name: _package_version(name)
            for name in ("mmengine", "mmcv", "mmdet", "mmdet3d", "numpy")
        },
        "device": {
            "requested": str(device),
            "name": properties.name,
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": properties.total_memory,
        },
        "determinism": {
            "algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "ROSA_DETECTOR_HOST_ROOT",
                "SPARSE4D_ROOT",
                "FISHEYE3DOD_OFFICIAL_ROOT",
            )
        },
    }


def execute_verification(prepared: PreparedVerification) -> dict[str, Any]:
    """Execute the armed GPU protocol.  Never called by the default CLI path."""

    args = prepared.args
    if args.idle_gpu_confirmation != IDLE_GPU_CONFIRMATION:
        raise VerificationContractError(
            "--execute requires the exact idle-GPU confirmation token"
        )
    if not isinstance(args.device, str) or not args.device.startswith("cuda:"):
        raise VerificationContractError("armed verification requires device cuda:N")
    try:
        device_index = int(args.device.split(":", 1)[1])
    except (IndexError, ValueError) as error:
        raise VerificationContractError("armed verification requires device cuda:N") from error
    if device_index < 0:
        raise VerificationContractError("CUDA device index must be nonnegative")
    existing_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing_workspace not in (None, ":4096:8"):
        raise VerificationContractError(
            "CUBLAS_WORKSPACE_CONFIG conflicts with the frozen :4096:8 contract"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    repository = _git_snapshot(ROOT, require_clean=True)
    output_dir = Path(prepared.plan["output_dir"])
    if output_dir.exists():
        raise VerificationContractError(
            f"output directory already exists; refusing overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    armed_plan = copy.deepcopy(prepared.plan)
    armed_plan.update(
        {
            "status": "ARMED",
            "cuda_queried": False,
            "repository": repository,
        }
    )
    _atomic_json(output_dir / "plan.json", armed_plan)

    try:
        cuda_available = torch.cuda.is_available()
        armed_plan["cuda_queried"] = True
        armed_plan["cuda_available"] = cuda_available
        _atomic_json(output_dir / "plan.json", armed_plan)
        if not cuda_available:
            raise VerificationContractError("armed verification requires CUDA")
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        _seed_everything(prepared.plan["seed"], cuda=True)

        models, model_contracts = _build_models(prepared, device)
        batches, sample_manifest = _capture_batches(prepared)
        _atomic_json(output_dir / "captured_batches.json", sample_manifest)

        metadata_parity = _verify_metadata(models, batches)
        _atomic_json(output_dir / "metadata_parity.json", metadata_parity)
        output_parity = _verify_outputs_and_state(models, batches)
        _atomic_json(output_dir / "output_state_rng_parity.json", output_parity)

        profile_batch = batches[
            prepared.plan["parameters"]["profile_batch_index"]
        ]
        profile_parent = _profile_arm(
            PARENT_ARM,
            models[PARENT_ARM],
            profile_batch,
            output_dir / "profile_parent.json",
        )
        profile_candidate = _profile_arm(
            CANDIDATE_ARM,
            models[CANDIDATE_ARM],
            profile_batch,
            output_dir / "profile_candidate.json",
        )
        transfer_reduction = validate_transfer_reduction(
            profile_parent,
            profile_candidate,
            minimum_device_to_host=prepared.plan["minimum_transfer_reduction"][
                "device_to_host"
            ],
            minimum_host_to_device=prepared.plan["minimum_transfer_reduction"][
                "host_to_device"
            ],
        )
        profile_summary = {
            PARENT_ARM: profile_parent,
            CANDIDATE_ARM: profile_candidate,
            "observed_reduction": transfer_reduction,
        }
        _atomic_json(output_dir / "transfer_profile_summary.json", profile_summary)

        memory = {
            label: _memory_probe(model, profile_batch)
            for label, model in models.items()
        }
        _atomic_json(output_dir / "memory.json", memory)

        for model in models.values():
            _reset_runtime_state(model)
        warmup_schedule = build_alternating_schedule(
            rounds=prepared.plan["parameters"]["warmup_rounds"],
            batch_count=len(batches),
            phase="warmup",
        )
        warmup_records = _run_schedule(
            warmup_schedule,
            models,
            batches,
            expected_batch_size=prepared.plan["parameters"]["expected_batch_size"],
        )
        _write_jsonl(output_dir / "warmup_timings.jsonl", warmup_records)

        measured_schedule = build_alternating_schedule(
            rounds=prepared.plan["parameters"]["measured_rounds"],
            batch_count=len(batches),
            phase="measured",
        )
        timing_rng_before = _capture_rng(cuda=True)
        measured_records = _run_schedule(
            measured_schedule,
            models,
            batches,
            expected_batch_size=prepared.plan["parameters"]["expected_batch_size"],
        )
        timing_rng_after = _capture_rng(cuda=True)
        require_same_fingerprint(
            "timing RNG stability", timing_rng_before, timing_rng_after
        )
        _write_jsonl(output_dir / "raw_timings.jsonl", measured_records)

        statistics = {
            PARENT_ARM: summarize_timings(measured_records, arm=PARENT_ARM),
            CANDIDATE_ARM: summarize_timings(measured_records, arm=CANDIDATE_ARM),
        }
        statistics["candidate_vs_parent"] = {
            "median_latency_ratio": (
                statistics[CANDIDATE_ARM]["median_iteration_ms"]
                / statistics[PARENT_ARM]["median_iteration_ms"]
            ),
            "p95_latency_ratio": (
                statistics[CANDIDATE_ARM]["p95_iteration_ms"]
                / statistics[PARENT_ARM]["p95_iteration_ms"]
            ),
            "fps_ratio": (
                statistics[CANDIDATE_ARM]["samples_per_second"]
                / statistics[PARENT_ARM]["samples_per_second"]
            ),
        }
        summary = {
            "schema_version": 1,
            "status": "COMPLETE",
            "plan_sha256": file_sha256(output_dir / "plan.json"),
            "repository": repository,
            "environment": _environment(device),
            "model_contracts": model_contracts,
            "captured_batches": sample_manifest,
            "parity": {
                "metadata": "raw-bit match",
                "outputs_runtime_state_rng": "raw-bit match",
            },
            "transfer_profile": profile_summary,
            "memory": memory,
            "timing_statistics": statistics,
            "raw_artifacts": {
                name: {
                    "path": str(output_dir / name),
                    "sha256": file_sha256(output_dir / name),
                }
                for name in (
                    "captured_batches.json",
                    "metadata_parity.json",
                    "output_state_rng_parity.json",
                    "profile_parent.json",
                    "profile_candidate.json",
                    "transfer_profile_summary.json",
                    "memory.json",
                    "warmup_timings.jsonl",
                    "raw_timings.jsonl",
                )
            },
        }
        _atomic_json(output_dir / "summary.json", summary)
        return summary
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "FAILED",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        _atomic_json(output_dir / "failure.json", failure)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.execute and args.idle_gpu_confirmation != IDLE_GPU_CONFIRMATION:
        parser.error(
            "--execute requires --idle-gpu-confirmation "
            + IDLE_GPU_CONFIRMATION
        )
    prepared = prepare_verification(args)
    if not args.execute:
        print(json.dumps(prepared.plan, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    result = execute_verification(prepared)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
