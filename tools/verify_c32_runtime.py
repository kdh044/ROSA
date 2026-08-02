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
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    CANDIDATE_ARM,
    CPU_METADATA_PROTOCOL,
    DEFAULT_MEASURED_ITERATIONS_PER_ARM,
    DEFAULT_WARMUP_ITERATIONS_PER_ARM,
    HOST_METADATA_KEYS,
    INPUT_PIPELINE_SCOPE,
    ITERATIONS_PER_ARM_GRANULARITY,
    PARENT_ARM,
    PHYSICAL_ROUND_ORDER,
    MODEL_ONLY_SCOPE,
    REPRESENTATION_PROTOCOL,
    REPRESENTATION_STREAM_POLICY,
    VerificationContractError,
    build_alternating_schedule,
    build_batch_manifest,
    build_representation_pair_manifest,
    canonical_sha256,
    evaluate_c32_near_native_gate,
    evaluate_input_pipeline_gate,
    evaluate_model_only_non_regression_gate,
    evaluate_strict_lightweight_gate,
    file_sha256,
    paired_bootstrap_confidence_intervals,
    raw_fingerprint,
    require_same_fingerprint,
    semantic_sample_digest,
    summarize_timings,
    summarize_transfer_event_names,
    validate_config_pair,
    validate_common_preprocessor_boundary,
    validate_expected_manifest,
    validate_runtime_parameters,
    validate_preregistration_plan,
    validate_representation_config_pair,
    validate_representation_ingress,
    validate_representation_snapshot,
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
DEFAULT_REPRESENTATION_PARENT_CONFIG = ROOT / (
    "configs/"
    "rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_"
    "cpu_metadata_lean_loader.py"
)
DEFAULT_REPRESENTATION_CANDIDATE_CONFIG = ROOT / (
    "configs/"
    "rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_"
    "cpu_metadata_lean_loader_uint8_transfer_pinned.py"
)
DEFAULT_PREREGISTRATION_DIR = Path(
    "/home/airlab/user/kdh/ROSA/results/raw_metrics/diagnostics/"
    "v1_exacte94_lightweight_fresh30_runtime_preregistration_20260802"
)
EXPECTED_PREREGISTRATION_HASHES = {
    "PLAN.json": "f82b46f0dfdec8d1482d9b0dc219aada2f4288b7eab322dd87e64b5478cb616d",
    "REPORT.md": "93550205c0b330509d3340126f072c39850062324fd22da0ffbb354a4fca7ed5",
    "SHA256SUMS": "9423422f81f0956ec5fdf65f8d6acc318d2ccd0082b84287dea82274b801cb1a",
}


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
    parser.add_argument(
        "--protocol",
        choices=(CPU_METADATA_PROTOCOL, REPRESENTATION_PROTOCOL),
        default=CPU_METADATA_PROTOCOL,
    )
    parser.add_argument("--parent-config", type=Path)
    parser.add_argument(
        "--candidate-config", type=Path
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--preregistration-dir",
        type=Path,
        default=DEFAULT_PREREGISTRATION_DIR,
    )
    parser.add_argument("--expected-sample-manifest", type=Path)
    parser.add_argument("--capture-batches", type=int, default=2)
    parser.add_argument("--expected-batch-size", type=int, default=4)
    parser.add_argument("--profile-batch-index", type=int, default=0)
    parser.add_argument(
        "--warmup-iterations-per-arm",
        type=int,
        default=DEFAULT_WARMUP_ITERATIONS_PER_ARM,
    )
    parser.add_argument(
        "--measured-iterations-per-arm",
        type=int,
        default=DEFAULT_MEASURED_ITERATIONS_PER_ARM,
    )
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
        "--execution-state",
        choices=("DISARMED", "ARMED"),
        default="DISARMED",
        help="Execution is impossible unless this is explicitly ARMED.",
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


def _load_preregistration(directory: Path) -> tuple[Mapping[str, Any], dict[str, Any]]:
    try:
        resolved = directory.expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise VerificationContractError(
            f"preregistration directory does not exist: {directory}"
        ) from error
    if not resolved.is_dir():
        raise VerificationContractError(
            f"preregistration path must be a directory: {resolved}"
        )
    files = {}
    for name, expected_sha256 in EXPECTED_PREREGISTRATION_HASHES.items():
        path = _resolve_file(resolved / name, f"preregistration {name}")
        observed_sha256 = file_sha256(path)
        if observed_sha256 != expected_sha256:
            raise VerificationContractError(
                f"preregistration {name} hash mismatch: "
                f"{observed_sha256} != {expected_sha256}"
            )
        files[name] = {
            "path": str(path),
            "sha256": observed_sha256,
        }
    plan = _load_json_mapping(resolved / "PLAN.json", "preregistration PLAN.json")
    contract = validate_preregistration_plan(plan)
    return plan, {
        "directory": str(resolved),
        "files": files,
        "contract": contract,
    }


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

    if args.protocol == CPU_METADATA_PROTOCOL:
        parent_config_argument = args.parent_config or DEFAULT_PARENT_CONFIG
        candidate_config_argument = args.candidate_config or DEFAULT_CANDIDATE_CONFIG
    elif args.protocol == REPRESENTATION_PROTOCOL:
        parent_config_argument = (
            args.parent_config or DEFAULT_REPRESENTATION_PARENT_CONFIG
        )
        candidate_config_argument = (
            args.candidate_config or DEFAULT_REPRESENTATION_CANDIDATE_CONFIG
        )
    else:  # argparse prevents this; retain fail-closed direct-call behavior.
        raise VerificationContractError(f"unsupported protocol: {args.protocol}")

    parameters = validate_runtime_parameters(
        capture_batches=args.capture_batches,
        warmup_iterations_per_arm=args.warmup_iterations_per_arm,
        measured_iterations_per_arm=args.measured_iterations_per_arm,
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

    preregistration_plan, preregistration = _load_preregistration(
        args.preregistration_dir
    )
    preregistration["parsed_plan_sha256"] = canonical_sha256(
        preregistration_plan
    )

    parent_path = _resolve_file(parent_config_argument, "parent config")
    candidate_path = _resolve_file(candidate_config_argument, "candidate config")
    checkpoint_path = _resolve_file(args.checkpoint, "checkpoint")
    parent_config = Config.fromfile(parent_path, import_custom_modules=False)
    candidate_config = Config.fromfile(candidate_path, import_custom_modules=False)
    if args.protocol == CPU_METADATA_PROTOCOL:
        config_contract = validate_config_pair(
            parent_config,
            candidate_config,
            expected_batch_size=parameters["expected_batch_size"],
        )
    else:
        config_contract = validate_representation_config_pair(
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
        iterations_per_arm=parameters["warmup_iterations_per_arm"],
        batch_count=parameters["capture_batches"],
        phase="warmup",
    )
    measured_schedule = build_alternating_schedule(
        iterations_per_arm=parameters["measured_iterations_per_arm"],
        batch_count=parameters["capture_batches"],
        phase="measured",
    )
    def schedule_contract(schedule, iterations_per_arm):
        serialized = [entry.to_dict() for entry in schedule]
        return {
            "iterations_per_arm": iterations_per_arm,
            "total_invocations": len(schedule),
            "iterations_per_arm_segment": (
                iterations_per_arm // ITERATIONS_PER_ARM_GRANULARITY
            ),
            "sha256": canonical_sha256(serialized),
        }

    schedule_payload = {
        "physical_round_order": list(PHYSICAL_ROUND_ORDER),
        "batch_mapping": (
            "matched iteration indices in adjacent P/C arm segments use "
            "pair_index modulo capture_batches"
        ),
        "pair_mapping": (
            "same iteration index across adjacent arm segments 0/1 and 2/3"
        ),
        "warmup": schedule_contract(
            warmup_schedule,
            parameters["warmup_iterations_per_arm"],
        ),
        "measured": schedule_contract(
            measured_schedule,
            parameters["measured_iterations_per_arm"],
        ),
    }
    plan = {
        "schema_version": 2,
        "status": "DRY_RUN",
        "protocol": args.protocol,
        "execution_state": args.execution_state,
        "execution_requested": bool(args.execute),
        "cuda_queried": False,
        "preregistration": preregistration,
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
    if args.protocol == REPRESENTATION_PROTOCOL:
        plan["benchmark_scopes"] = {
            "model_only": {
                "name": MODEL_ONLY_SCOPE,
                "includes": ["model._run_forward(mode='predict')"],
                "excludes": ["loader", "collate", "pinning", "preprocessor", "H2D"],
                "warmup_iterations_per_arm": parameters[
                    "warmup_iterations_per_arm"
                ],
                "measured_iterations_per_arm": parameters[
                    "measured_iterations_per_arm"
                ],
                "semantic_pairing": (
                    "replay captured arm-specific representations by shared "
                    "semantic digest"
                ),
            },
            "input_pipeline_end_to_end": {
                "name": INPUT_PIPELINE_SCOPE,
                "includes": [
                    "dataset loader/transforms",
                    "collate/pinning",
                    "data preprocessor/H2D",
                ],
                "excludes": ["model execution", "evaluator", "serialization"],
                "warmup_iterations_per_arm": parameters[
                    "warmup_iterations_per_arm"
                ],
                "measured_iterations_per_arm": parameters[
                    "measured_iterations_per_arm"
                ],
                "semantic_pairing": (
                    "advance independent parent/candidate dataloaders in the "
                    "same pair order and reject digest mismatch or reuse"
                ),
                "prevalidation": (
                    "fingerprint the complete warmup-plus-measured semantic "
                    "sequence, then rebuild both dataloaders for timing"
                ),
                "filesystem_cache_policy": (
                    "no cache flush; report as a prevalidated warm-cache run"
                ),
            },
            "combined_full_inference": {
                "measured": False,
                "decision": "NO_FULL_INFERENCE_SPEED_CLAIM",
            },
        }
        plan["transfer_contract"] = {
            "parent": {"dtype": "torch.float32", "pinned": False, "non_blocking": False},
            "candidate": {"dtype": "torch.uint8", "pinned": True, "non_blocking": True},
            "stream_policy": REPRESENTATION_STREAM_POLICY,
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


def _build_dataloader(
    prepared: PreparedVerification,
    *,
    arm: str = PARENT_ARM,
):
    from mmengine.runner import Runner

    if arm == PARENT_ARM:
        source_config = prepared.parent_config
    elif arm == CANDIDATE_ARM:
        source_config = prepared.candidate_config
    else:
        raise VerificationContractError(f"invalid dataloader arm: {arm}")
    dataloader_config = copy.deepcopy(source_config.test_dataloader)
    workers = prepared.plan["dataloader_workers"]
    dataloader_config["num_workers"] = workers
    dataloader_config["persistent_workers"] = bool(workers)
    return Runner.build_dataloader(
        dataloader_config,
        seed=prepared.plan["seed"],
        diff_rank_seed=False,
    )


def _build_representation_dataloaders(
    prepared: PreparedVerification,
) -> dict[str, Any]:
    return {
        arm: _build_dataloader(prepared, arm=arm)
        for arm in (PARENT_ARM, CANDIDATE_ARM)
    }


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


def _capture_representation_batches(prepared: PreparedVerification):
    dataloaders = _build_representation_dataloaders(prepared)
    iterators = {arm: iter(loader) for arm, loader in dataloaders.items()}
    batches = {PARENT_ARM: [], CANDIDATE_ARM: []}
    pair_manifests = []
    observed_identities = set()
    for batch_index in range(prepared.plan["parameters"]["capture_batches"]):
        current = {}
        for arm in (PARENT_ARM, CANDIDATE_ARM):
            try:
                current[arm] = next(iterators[arm])
            except StopIteration as error:
                raise VerificationContractError(
                    f"{arm} dataloader ended before representation capture "
                    "completed"
                ) from error
        pair = build_representation_pair_manifest(
            current[PARENT_ARM],
            current[CANDIDATE_ARM],
            batch_index=batch_index,
            expected_batch_size=prepared.plan["parameters"][
                "expected_batch_size"
            ],
        )
        for identity in pair["sample_identities"]:
            digest = canonical_sha256(identity)
            if digest in observed_identities:
                raise VerificationContractError(
                    "captured representation sample identities repeat"
                )
            observed_identities.add(digest)
        for arm in (PARENT_ARM, CANDIDATE_ARM):
            batches[arm].append(current[arm])
        pair_manifests.append(pair)
    document = {
        "schema": "rosa.representation_pair_manifest.v1",
        "protocol": REPRESENTATION_PROTOCOL,
        "pairs": pair_manifests,
    }
    document["sha256"] = canonical_sha256(document["pairs"])
    if prepared.expected_manifest is not None:
        validate_expected_manifest(document, prepared.expected_manifest)
    return batches, document


def _prevalidate_live_representation_sequence(
    prepared: PreparedVerification,
    *,
    pairs_per_arm: int,
) -> dict[str, Any]:
    """Fingerprint every semantic pair before the live pipeline timing pass."""

    if isinstance(pairs_per_arm, bool):
        raise VerificationContractError("pairs_per_arm must be an exact integer")
    try:
        pairs_per_arm = operator.index(pairs_per_arm)
    except TypeError as error:
        raise VerificationContractError(
            "pairs_per_arm must be an exact integer"
        ) from error
    if pairs_per_arm < 1:
        raise VerificationContractError("pairs_per_arm must be positive")
    dataloaders = _build_representation_dataloaders(prepared)
    iterators = {arm: iter(loader) for arm, loader in dataloaders.items()}
    pairs = []
    observed_identities = set()
    for sequence_index in range(pairs_per_arm):
        current = {}
        for arm in (PARENT_ARM, CANDIDATE_ARM):
            try:
                current[arm] = next(iterators[arm])
            except StopIteration as error:
                raise VerificationContractError(
                    f"{arm} dataloader ended during full representation "
                    "prevalidation"
                ) from error
        pair = build_representation_pair_manifest(
            current[PARENT_ARM],
            current[CANDIDATE_ARM],
            batch_index=sequence_index,
            expected_batch_size=prepared.plan["parameters"][
                "expected_batch_size"
            ],
        )
        for identity in pair["sample_identities"]:
            digest = canonical_sha256(identity)
            if digest in observed_identities:
                raise VerificationContractError(
                    "live representation prevalidation repeated a sample"
                )
            observed_identities.add(digest)
        pairs.append(pair)
        del current
    document = {
        "schema": "rosa.live_representation_pair_sequence.v1",
        "protocol": REPRESENTATION_PROTOCOL,
        "pairs_per_arm": pairs_per_arm,
        "pairs": pairs,
    }
    document["sha256"] = canonical_sha256(pairs)
    return document


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


@torch.inference_mode()
def _verify_common_preprocessor_boundaries(
    models: Mapping[str, torch.nn.Module],
    batches: Mapping[str, Sequence[Any]],
    representation_manifest: Mapping[str, Any],
):
    pairs = representation_manifest.get("pairs")
    if not isinstance(pairs, Sequence):
        raise VerificationContractError(
            "representation manifest has no pair sequence"
        )
    processed = {PARENT_ARM: [], CANDIDATE_ARM: []}
    records = []
    for batch_index, pair in enumerate(pairs):
        arm_outputs = {}
        for arm in (PARENT_ARM, CANDIDATE_ARM):
            source = batches[arm][batch_index]
            arm_manifest = pair["arms"][arm]
            validate_representation_snapshot(
                source,
                arm_manifest,
                label=f"{arm} batch {batch_index} before preprocessing",
            )
            arm_outputs[arm] = models[arm].data_preprocessor(
                _clone_batch(source),
                training=False,
            )
            validate_representation_snapshot(
                source,
                arm_manifest,
                label=f"{arm} batch {batch_index} after preprocessing",
            )
        boundary = validate_common_preprocessor_boundary(
            arm_outputs[PARENT_ARM],
            arm_outputs[CANDIDATE_ARM],
            expected_device_type="cuda",
        )
        for arm in (PARENT_ARM, CANDIDATE_ARM):
            processed[arm].append(arm_outputs[arm])
        records.append(
            {
                "batch_index": batch_index,
                "semantic_sha256": pair["semantic_sha256"],
                "boundary": boundary,
            }
        )
    return processed, records


def _validate_captured_representation_batches(
    batches: Mapping[str, Sequence[Any]],
    representation_manifest: Mapping[str, Any],
    *,
    phase: str,
) -> None:
    for batch_index, pair in enumerate(representation_manifest["pairs"]):
        for arm in (PARENT_ARM, CANDIDATE_ARM):
            validate_representation_snapshot(
                batches[arm][batch_index],
                pair["arms"][arm],
                label=f"{arm} batch {batch_index} {phase}",
            )


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
            except VerificationContractError as error:
                raise VerificationContractError(
                    f"cannot fingerprint selected runtime state {path}: {error}"
                ) from error
    return captured


@torch.inference_mode()
def _run_output_sequence(model: torch.nn.Module, batches: Sequence[Any]):
    output_records = []
    for batch_index, batch in enumerate(batches):
        output = model.test_step(_clone_batch(batch))
        expected_samples = len(batch["data_samples"])
        if len(output) != expected_samples:
            raise VerificationContractError(
                f"parity output has {len(output)} samples, expected "
                f"{expected_samples}"
            )
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
    models: Mapping[str, torch.nn.Module],
    batches: Sequence[Any] | Mapping[str, Sequence[Any]],
) -> dict[str, Any]:
    for model in models.values():
        _reset_runtime_state(model)
    state_dict_before = require_same_fingerprint(
        "pre-inference model state_dict",
        models[PARENT_ARM].state_dict(),
        models[CANDIDATE_ARM].state_dict(),
    )
    initial_rng = _capture_rng(cuda=True)
    _restore_rng(initial_rng, cuda=True)
    parent_batches = (
        batches[PARENT_ARM] if isinstance(batches, Mapping) else batches
    )
    candidate_batches = (
        batches[CANDIDATE_ARM] if isinstance(batches, Mapping) else batches
    )
    parent = _run_output_sequence(models[PARENT_ARM], parent_batches)
    parent_rng = _capture_rng(cuda=True)
    _restore_rng(initial_rng, cuda=True)
    candidate = _run_output_sequence(models[CANDIDATE_ARM], candidate_batches)
    candidate_rng = _capture_rng(cuda=True)
    if parent != candidate:
        raise VerificationContractError(
            "parent/candidate raw output or runtime-state fingerprints differ"
        )
    require_same_fingerprint("post-inference RNG", parent_rng, candidate_rng)
    state_dict_after = require_same_fingerprint(
        "post-inference model state_dict",
        models[PARENT_ARM].state_dict(),
        models[CANDIDATE_ARM].state_dict(),
    )
    _restore_rng(initial_rng, cuda=True)
    return {
        "batches": parent,
        "state_dict_before": state_dict_before,
        "state_dict_after": state_dict_after,
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
    allocated_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    output = model.test_step(_clone_batch(batch))
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    allocated_after = torch.cuda.memory_allocated()
    reserved_after = torch.cuda.memory_reserved()
    sample_count = len(output)
    del output
    torch.cuda.synchronize()
    return {
        "allocated_before_bytes": allocated_before,
        "reserved_before_bytes": reserved_before,
        "allocated_after_output_bytes": allocated_after,
        "reserved_after_output_bytes": reserved_after,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_increment_over_before_bytes": max(
            0, peak_allocated - allocated_before
        ),
        "peak_reserved_increment_over_before_bytes": max(
            0, peak_reserved - reserved_before
        ),
        "sample_count": sample_count,
    }


@torch.inference_mode()
def _run_schedule(
    schedule,
    models: Mapping[str, torch.nn.Module],
    batches: Sequence[Any],
    *,
    expected_batch_size: int,
    sequence_offset: int = 0,
) -> list[dict[str, Any]]:
    records = []
    for local_index, entry in enumerate(schedule):
        sequence_index = sequence_offset + local_index
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


def _require_default_cuda_stream(device: torch.device) -> dict[str, Any]:
    current = torch.cuda.current_stream(device)
    default = torch.cuda.default_stream(device)
    current_id = int(current.cuda_stream)
    default_id = int(default.cuda_stream)
    if current_id != default_id:
        raise VerificationContractError(
            "representation protocol requires the current CUDA default stream"
        )
    return {
        "policy": REPRESENTATION_STREAM_POLICY,
        "current_stream": current_id,
        "default_stream": default_id,
    }


@torch.inference_mode()
def _run_model_only_schedule(
    schedule,
    models: Mapping[str, torch.nn.Module],
    processed_batches: Mapping[str, Sequence[Any]],
    representation_pairs: Sequence[Mapping[str, Any]],
    *,
    expected_batch_size: int,
    device: torch.device,
    sequence_offset: int = 0,
) -> list[dict[str, Any]]:
    """Time model math from the already-proven common FP32 boundary."""

    records = []
    stream = _require_default_cuda_stream(device)
    for local_index, entry in enumerate(schedule):
        sequence_index = sequence_offset + local_index
        source = processed_batches[entry.arm][entry.batch_index]
        batch = _clone_batch(source)
        model = models[entry.arm]
        torch.cuda.synchronize(device)
        start_ns = time.perf_counter_ns()
        output = model._run_forward(batch, mode="predict")
        torch.cuda.synchronize(device)
        elapsed_ns = time.perf_counter_ns() - start_ns
        sample_count = len(output)
        if sample_count != expected_batch_size:
            raise VerificationContractError(
                f"model-only output has {sample_count} samples, expected "
                f"{expected_batch_size}"
            )
        record = entry.to_dict()
        record.update(
            {
                "sequence_index": sequence_index,
                "benchmark_scope": MODEL_ONLY_SCOPE,
                "semantic_batch_sha256": representation_pairs[
                    entry.batch_index
                ]["semantic_sha256"],
                "elapsed_ns": elapsed_ns,
                "sample_count": sample_count,
                "stream": stream,
            }
        )
        records.append(record)
        del output, batch
    return records


@torch.inference_mode()
def _run_input_pipeline_schedule(
    schedule,
    models: Mapping[str, torch.nn.Module],
    iterators: Mapping[str, Any],
    *,
    expected_batch_size: int,
    device: torch.device,
    expected_pair_sequence: Sequence[Mapping[str, Any]],
    stream_positions: dict[str, int],
    sequence_offset: int = 0,
) -> list[dict[str, Any]]:
    """Time loader+collate and preprocessor/H2D, excluding model execution."""

    records = []
    stream = _require_default_cuda_stream(device)
    for local_index, entry in enumerate(schedule):
        sequence_index = sequence_offset + local_index
        torch.cuda.synchronize(device)
        loader_start_ns = time.perf_counter_ns()
        try:
            batch = next(iterators[entry.arm])
        except StopIteration as error:
            raise VerificationContractError(
                f"{entry.arm} input pipeline ended before the sealed schedule"
            ) from error
        loader_collate_ns = time.perf_counter_ns() - loader_start_ns

        identity = semantic_sample_digest(
            batch,
            expected_batch_size=expected_batch_size,
        )
        ingress = validate_representation_ingress(
            batch,
            arm=entry.arm,
            expected_batch_size=expected_batch_size,
        )
        stream_position = stream_positions[entry.arm]
        if stream_position >= len(expected_pair_sequence):
            raise VerificationContractError(
                "input pipeline exceeded the prevalidated pair sequence"
            )
        expected_pair = expected_pair_sequence[stream_position]
        if identity["sample_identities"] != expected_pair["sample_identities"]:
            raise VerificationContractError(
                f"{entry.arm} live input pipeline diverged from prevalidated "
                f"semantic pair {stream_position}"
            )
        stream_positions[entry.arm] = stream_position + 1

        torch.cuda.synchronize(device)
        preprocessor_start_ns = time.perf_counter_ns()
        processed = models[entry.arm].data_preprocessor(batch, training=False)
        torch.cuda.synchronize(device)
        preprocessor_transfer_ns = time.perf_counter_ns() - preprocessor_start_ns
        processed_inputs = processed.get("inputs") if isinstance(processed, Mapping) else None
        processed_samples = (
            processed.get("data_samples") if isinstance(processed, Mapping) else None
        )
        processed_image = (
            processed_inputs.get("img")
            if isinstance(processed_inputs, Mapping)
            else None
        )
        if (
            not isinstance(processed_image, torch.Tensor)
            or processed_image.dtype != torch.float32
            or processed_image.device.type != "cuda"
        ):
            raise VerificationContractError(
                "input pipeline did not reach the common CUDA FP32 boundary"
            )
        if not isinstance(processed_samples, Sequence) or len(processed_samples) != expected_batch_size:
            raise VerificationContractError(
                "input pipeline changed the batch sample count"
            )
        record = entry.to_dict()
        record.update(
            {
                "sequence_index": sequence_index,
                "benchmark_scope": INPUT_PIPELINE_SCOPE,
                "semantic_batch_sha256": expected_pair["semantic_sha256"],
                "sample_identities": identity["sample_identities"],
                "prevalidated_stream_position": stream_position,
                "loader_collate_ns": loader_collate_ns,
                "preprocessor_transfer_ns": preprocessor_transfer_ns,
                "validation_gap_excluded": True,
                "elapsed_ns": loader_collate_ns + preprocessor_transfer_ns,
                "sample_count": expected_batch_size,
                "ingress": ingress,
                "stream": stream,
            }
        )
        records.append(record)
        del processed, batch
    return records


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _nvidia_smi_snapshot(device_index: int) -> dict[str, Any]:
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
    gpu_result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={device_index}",
            "--query-gpu=" + ",".join(gpu_fields),
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    gpu_rows = [
        row.strip() for row in gpu_result.stdout.splitlines() if row.strip()
    ]
    if len(gpu_rows) != 1:
        raise VerificationContractError(
            "nvidia-smi must return exactly one selected GPU row"
        )
    app_fields = ("pid", "process_name", "used_gpu_memory", "gpu_uuid")
    app_result = subprocess.run(
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
    app_rows = [
        row.strip() for row in app_result.stdout.splitlines() if row.strip()
    ]
    foreign_rows = []
    for row in app_rows:
        try:
            pid = int(row.split(",", 1)[0].strip())
        except ValueError as error:
            raise VerificationContractError(
                f"could not parse nvidia-smi compute-app row: {row}"
            ) from error
        if pid != os.getpid():
            foreign_rows.append(row)
    if foreign_rows:
        raise VerificationContractError(
            "concurrent GPU compute process detected by nvidia-smi"
        )
    return {
        "gpu_fields": list(gpu_fields),
        "gpu_row": gpu_rows[0],
        "compute_app_fields": list(app_fields),
        "compute_app_rows": app_rows,
        "current_pid": os.getpid(),
        "foreign_compute_processes": 0,
    }


def _environment(
    device: torch.device,
    *,
    nvidia_smi_start: Mapping[str, Any],
) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    get_autocast_dtype = getattr(torch, "get_autocast_dtype", None)
    autocast_dtype = (
        str(get_autocast_dtype("cuda"))
        if callable(get_autocast_dtype)
        else None
    )
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
        "precision": {
            "default_dtype": str(torch.get_default_dtype()),
            "autocast_enabled": torch.is_autocast_enabled(),
            "autocast_dtype": autocast_dtype,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
        "nvidia_smi": {
            "start": dict(nvidia_smi_start),
            "end": _nvidia_smi_snapshot(device.index),
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


def _preregistered_decisions(
    paired_bootstrap: Mapping[str, Any],
    memory: Mapping[str, Any],
) -> dict[str, Any]:
    strict_decision = evaluate_strict_lightweight_gate(
        paired_bootstrap,
        memory,
    )
    near_native_calculation = evaluate_c32_near_native_gate(
        paired_bootstrap
    )
    return {
        "strict_lightweight": strict_decision,
        "c32_vs_c0_near_native": {
            "numerical_gate_on_current_pair": near_native_calculation,
            "claim_eligible": False,
            "decision": "NOT_APPLICABLE_TO_CURRENT_ARM_PAIR",
            "reason": (
                "this sealed CLI compares frozen C32 against the C32 "
                "CPU-metadata candidate, not C0 against C32"
            ),
        },
    }


def _segmented_records(
    warmup_schedule,
    measured_schedule,
    *,
    models: Mapping[str, torch.nn.Module],
    run_segment,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute the literal six-round PCCP/CPPC physical segment order."""

    warmup_records = []
    measured_records = []
    for round_index, expected_pattern in enumerate(PHYSICAL_ROUND_ORDER):
        for model in models.values():
            _reset_runtime_state(model)
        for slot_index, token in enumerate(expected_pattern):
            expected_arm = PARENT_ARM if token == "P" else CANDIDATE_ARM
            segment_warmup = tuple(
                entry
                for entry in warmup_schedule
                if entry.round_index == round_index
                and entry.slot_index == slot_index
            )
            segment_measured = tuple(
                entry
                for entry in measured_schedule
                if entry.round_index == round_index
                and entry.slot_index == slot_index
            )
            if not segment_warmup or not segment_measured:
                raise VerificationContractError(
                    f"physical round {round_index} slot {slot_index} has no "
                    "timing entries"
                )
            if any(
                entry.pattern != expected_pattern or entry.arm != expected_arm
                for entry in segment_warmup + segment_measured
            ):
                raise VerificationContractError(
                    "runtime schedule violates the preregistered physical "
                    "arm-segment order"
                )
            warmup_records.extend(
                run_segment(
                    segment_warmup,
                    sequence_offset=len(warmup_records),
                )
            )
            measured_records.extend(
                run_segment(
                    segment_measured,
                    sequence_offset=len(measured_records),
                )
            )
    return warmup_records, measured_records


def _validate_semantic_record_pairs(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_pairs: int,
    expected_scope: str,
    expected_phase: str,
    require_unique_semantic_batches: bool,
) -> dict[str, Any]:
    grouped: dict[int, dict[str, Mapping[str, Any]]] = {}
    for record in records:
        if record.get("benchmark_scope") != expected_scope:
            raise VerificationContractError(
                "timing record benchmark scope changed"
            )
        if record.get("phase") != expected_phase:
            raise VerificationContractError("timing record phase changed")
        raw_pair_index = record.get("pair_index")
        try:
            pair_index = operator.index(raw_pair_index)
        except TypeError as error:
            raise VerificationContractError(
                "timing pair index must be an exact integer"
            ) from error
        if isinstance(raw_pair_index, bool) or pair_index < 0:
            raise VerificationContractError("timing pair index must be nonnegative")
        arm = record.get("arm")
        if arm not in (PARENT_ARM, CANDIDATE_ARM):
            raise VerificationContractError("timing record arm changed")
        pair = grouped.setdefault(pair_index, {})
        if arm in pair:
            raise VerificationContractError(
                f"timing pair {pair_index} repeats {arm}"
            )
        pair[arm] = record
    if sorted(grouped) != list(range(expected_pairs)):
        raise VerificationContractError(
            "timing semantic pairs are missing or non-contiguous"
        )
    digests = []
    for pair_index in range(expected_pairs):
        pair = grouped[pair_index]
        if set(pair) != {PARENT_ARM, CANDIDATE_ARM}:
            raise VerificationContractError(
                f"timing pair {pair_index} lacks one representation arm"
            )
        parent_digest = pair[PARENT_ARM].get("semantic_batch_sha256")
        candidate_digest = pair[CANDIDATE_ARM].get("semantic_batch_sha256")
        if (
            not isinstance(parent_digest, str)
            or len(parent_digest) != 64
            or parent_digest != candidate_digest
        ):
            raise VerificationContractError(
                f"timing pair {pair_index} semantic digest mismatch"
            )
        digests.append(parent_digest)
    if require_unique_semantic_batches and len(set(digests)) != len(digests):
        raise VerificationContractError(
            "live timing reused a semantic batch across pair indices"
        )
    return {
        "pairs": expected_pairs,
        "unique_semantic_batches": len(set(digests)),
        "unique_semantic_batches_required": require_unique_semantic_batches,
        "scope": expected_scope,
        "phase": expected_phase,
    }


def _execute_representation_protocol(
    prepared: PreparedVerification,
    *,
    device: torch.device,
    models: Mapping[str, torch.nn.Module],
    model_contracts: Mapping[str, Any],
    output_dir: Path,
    repository: Mapping[str, Any],
    nvidia_smi_start: Mapping[str, Any],
) -> dict[str, Any]:
    batches, representation_manifest = _capture_representation_batches(prepared)
    _atomic_json(
        output_dir / "representation_pairs.json",
        representation_manifest,
    )
    processed_batches, boundary_records = _verify_common_preprocessor_boundaries(
        models,
        batches,
        representation_manifest,
    )
    _atomic_json(
        output_dir / "common_preprocessor_boundary_parity.json",
        boundary_records,
    )
    output_parity = _verify_outputs_and_state(models, batches)
    _atomic_json(output_dir / "output_state_rng_parity.json", output_parity)

    profile_index = prepared.plan["parameters"]["profile_batch_index"]
    combined_profiles = {
        arm: _profile_arm(
            arm,
            models[arm],
            batches[arm][profile_index],
            output_dir / f"combined_test_step_profile_{arm}.json",
        )
        for arm in (PARENT_ARM, CANDIDATE_ARM)
    }
    combined_profiles["benchmark_scope"] = (
        "informational-combined-test-step-not-used-for-scope-gates"
    )
    _atomic_json(output_dir / "combined_test_step_profiles.json", combined_profiles)
    combined_memory = {
        arm: _memory_probe(models[arm], batches[arm][profile_index])
        for arm in (PARENT_ARM, CANDIDATE_ARM)
    }
    _atomic_json(output_dir / "combined_test_step_memory.json", combined_memory)
    _validate_captured_representation_batches(
        batches,
        representation_manifest,
        phase="after parity/profile/memory",
    )

    parameters = prepared.plan["parameters"]
    warmup_schedule = build_alternating_schedule(
        iterations_per_arm=parameters["warmup_iterations_per_arm"],
        batch_count=len(representation_manifest["pairs"]),
        phase="warmup",
    )
    measured_schedule = build_alternating_schedule(
        iterations_per_arm=parameters["measured_iterations_per_arm"],
        batch_count=len(representation_manifest["pairs"]),
        phase="measured",
    )

    processed_before = {
        arm: [raw_fingerprint(batch) for batch in processed_batches[arm]]
        for arm in (PARENT_ARM, CANDIDATE_ARM)
    }
    timing_rng_before = _capture_rng(cuda=True)
    model_warmup, model_measured = _segmented_records(
        warmup_schedule,
        measured_schedule,
        models=models,
        run_segment=lambda schedule, sequence_offset: _run_model_only_schedule(
            schedule,
            models,
            processed_batches,
            representation_manifest["pairs"],
            expected_batch_size=parameters["expected_batch_size"],
            device=device,
            sequence_offset=sequence_offset,
        ),
    )
    timing_rng_after = _capture_rng(cuda=True)
    require_same_fingerprint(
        "model-only timing RNG stability",
        timing_rng_before,
        timing_rng_after,
    )
    processed_after = {
        arm: [raw_fingerprint(batch) for batch in processed_batches[arm]]
        for arm in (PARENT_ARM, CANDIDATE_ARM)
    }
    if processed_before != processed_after:
        raise VerificationContractError(
            "cached common-boundary representation mutated during model-only timing"
        )
    _write_jsonl(output_dir / "model_only_warmup_timings.jsonl", model_warmup)
    _write_jsonl(output_dir / "model_only_raw_timings.jsonl", model_measured)
    model_semantics = _validate_semantic_record_pairs(
        model_measured,
        expected_pairs=parameters["measured_iterations_per_arm"],
        expected_scope=MODEL_ONLY_SCOPE,
        expected_phase="measured",
        require_unique_semantic_batches=False,
    )
    model_bootstrap = paired_bootstrap_confidence_intervals(
        model_measured,
        expected_pairs=parameters["measured_iterations_per_arm"],
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
        required_pair_fields=("benchmark_scope", "semantic_batch_sha256"),
    )
    _atomic_json(output_dir / "model_only_paired_bootstrap.json", model_bootstrap)

    live_pair_sequence = _prevalidate_live_representation_sequence(
        prepared,
        pairs_per_arm=(
            parameters["warmup_iterations_per_arm"]
            + parameters["measured_iterations_per_arm"]
        ),
    )
    _atomic_json(
        output_dir / "live_representation_pair_sequence.json",
        live_pair_sequence,
    )
    live_dataloaders = _build_representation_dataloaders(prepared)
    live_iterators = {
        arm: iter(loader) for arm, loader in live_dataloaders.items()
    }
    live_stream_positions = {PARENT_ARM: 0, CANDIDATE_ARM: 0}
    pipeline_warmup, pipeline_measured = _segmented_records(
        warmup_schedule,
        measured_schedule,
        models=models,
        run_segment=lambda schedule, sequence_offset: _run_input_pipeline_schedule(
            schedule,
            models,
            live_iterators,
            expected_batch_size=parameters["expected_batch_size"],
            device=device,
            expected_pair_sequence=live_pair_sequence["pairs"],
            stream_positions=live_stream_positions,
            sequence_offset=sequence_offset,
        ),
    )
    expected_live_position = (
        parameters["warmup_iterations_per_arm"]
        + parameters["measured_iterations_per_arm"]
    )
    if live_stream_positions != {
        PARENT_ARM: expected_live_position,
        CANDIDATE_ARM: expected_live_position,
    }:
        raise VerificationContractError(
            "input pipeline did not consume the full prevalidated pair sequence"
        )
    _write_jsonl(
        output_dir / "input_pipeline_warmup_timings.jsonl",
        pipeline_warmup,
    )
    _write_jsonl(
        output_dir / "input_pipeline_raw_timings.jsonl",
        pipeline_measured,
    )
    pipeline_semantics = _validate_semantic_record_pairs(
        pipeline_measured,
        expected_pairs=parameters["measured_iterations_per_arm"],
        expected_scope=INPUT_PIPELINE_SCOPE,
        expected_phase="measured",
        require_unique_semantic_batches=True,
    )
    pipeline_bootstrap = paired_bootstrap_confidence_intervals(
        pipeline_measured,
        expected_pairs=parameters["measured_iterations_per_arm"],
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
        required_pair_fields=("benchmark_scope", "semantic_batch_sha256"),
    )
    _atomic_json(
        output_dir / "input_pipeline_paired_bootstrap.json",
        pipeline_bootstrap,
    )

    decisions = {
        "model_only": evaluate_model_only_non_regression_gate(model_bootstrap),
        "input_pipeline_end_to_end": evaluate_input_pipeline_gate(
            pipeline_bootstrap,
            representation_manifest["pairs"],
            common_boundary_parity=True,
        ),
        "combined_full_inference": {
            "measured": False,
            "decision": "NO_FULL_INFERENCE_SPEED_CLAIM",
            "reason": (
                "model-only and loader+collate+transfer are intentionally "
                "separate benchmark scopes"
            ),
        },
        "c32_vs_c0_near_native": {
            "claim_eligible": False,
            "decision": "NOT_APPLICABLE_TO_CURRENT_ARM_PAIR",
        },
    }
    _validate_captured_representation_batches(
        batches,
        representation_manifest,
        phase="after all timing scopes",
    )
    _atomic_json(output_dir / "representation_scope_decisions.json", decisions)
    environment = _environment(device, nvidia_smi_start=nvidia_smi_start)
    _atomic_json(output_dir / "environment.json", environment)
    artifacts = (
        "representation_pairs.json",
        "common_preprocessor_boundary_parity.json",
        "output_state_rng_parity.json",
        "combined_test_step_profile_parent.json",
        "combined_test_step_profile_candidate.json",
        "combined_test_step_profiles.json",
        "combined_test_step_memory.json",
        "live_representation_pair_sequence.json",
        "model_only_warmup_timings.jsonl",
        "model_only_raw_timings.jsonl",
        "model_only_paired_bootstrap.json",
        "input_pipeline_warmup_timings.jsonl",
        "input_pipeline_raw_timings.jsonl",
        "input_pipeline_paired_bootstrap.json",
        "representation_scope_decisions.json",
        "environment_start.json",
        "environment.json",
    )
    summary = {
        "schema": "rosa.c32_representation_runtime.v1",
        "status": "COMPLETE",
        "protocol": REPRESENTATION_PROTOCOL,
        "repository": repository,
        "environment": environment,
        "model_contracts": model_contracts,
        "representation_manifest": representation_manifest,
        "parity": {
            "common_preprocessor_boundary": "raw-bit exact",
            "outputs_runtime_state_rng": "raw-bit exact",
        },
        "measurement_contract": prepared.plan["benchmark_scopes"],
        "physical_round_order": list(PHYSICAL_ROUND_ORDER),
        "semantic_pair_validation": {
            "model_only": model_semantics,
            "input_pipeline_end_to_end": pipeline_semantics,
        },
        "results": {
            "model_only": model_bootstrap,
            "input_pipeline_end_to_end": pipeline_bootstrap,
        },
        "decisions": decisions,
        "full_inference_speed_claim_authorized": False,
        "raw_artifacts": {
            name: {
                "path": str(output_dir / name),
                "sha256": file_sha256(output_dir / name),
            }
            for name in artifacts
        },
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def execute_verification(prepared: PreparedVerification) -> dict[str, Any]:
    """Execute the armed GPU protocol.  Never called by the default CLI path."""

    args = prepared.args
    if args.execution_state != "ARMED":
        raise VerificationContractError(
            "runtime execution requires --execution-state ARMED"
        )
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
        nvidia_smi_start = _nvidia_smi_snapshot(device_index)
        _atomic_json(output_dir / "environment_start.json", nvidia_smi_start)

        models, model_contracts = _build_models(prepared, device)
        if prepared.plan["protocol"] == REPRESENTATION_PROTOCOL:
            return _execute_representation_protocol(
                prepared,
                device=device,
                models=models,
                model_contracts=model_contracts,
                output_dir=output_dir,
                repository=repository,
                nvidia_smi_start=nvidia_smi_start,
            )
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

        warmup_schedule = build_alternating_schedule(
            iterations_per_arm=prepared.plan["parameters"][
                "warmup_iterations_per_arm"
            ],
            batch_count=len(batches),
            phase="warmup",
        )
        measured_schedule = build_alternating_schedule(
            iterations_per_arm=prepared.plan["parameters"][
                "measured_iterations_per_arm"
            ],
            batch_count=len(batches),
            phase="measured",
        )
        warmup_records = []
        measured_records = []
        timing_rng_before = _capture_rng(cuda=True)
        for round_index, expected_pattern in enumerate(PHYSICAL_ROUND_ORDER):
            for model in models.values():
                _reset_runtime_state(model)
            for slot_index, token in enumerate(expected_pattern):
                expected_arm = (
                    PARENT_ARM if token == "P" else CANDIDATE_ARM
                )
                segment_warmup = tuple(
                    entry
                    for entry in warmup_schedule
                    if entry.round_index == round_index
                    and entry.slot_index == slot_index
                )
                segment_measured = tuple(
                    entry
                    for entry in measured_schedule
                    if entry.round_index == round_index
                    and entry.slot_index == slot_index
                )
                if not segment_warmup or not segment_measured:
                    raise VerificationContractError(
                        f"physical round {round_index} slot {slot_index} "
                        "has no timing entries"
                    )
                if any(
                    entry.pattern != expected_pattern
                    or entry.arm != expected_arm
                    for entry in segment_warmup + segment_measured
                ):
                    raise VerificationContractError(
                        "runtime schedule violates the preregistered "
                        "physical arm-segment order"
                    )
                warmup_records.extend(
                    _run_schedule(
                        segment_warmup,
                        models,
                        batches,
                        expected_batch_size=prepared.plan["parameters"][
                            "expected_batch_size"
                        ],
                        sequence_offset=len(warmup_records),
                    )
                )
                measured_records.extend(
                    _run_schedule(
                        segment_measured,
                        models,
                        batches,
                        expected_batch_size=prepared.plan["parameters"][
                            "expected_batch_size"
                        ],
                        sequence_offset=len(measured_records),
                    )
                )
        timing_rng_after = _capture_rng(cuda=True)
        require_same_fingerprint(
            "timing RNG stability", timing_rng_before, timing_rng_after
        )
        _write_jsonl(output_dir / "warmup_timings.jsonl", warmup_records)
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
        paired_bootstrap = paired_bootstrap_confidence_intervals(
            measured_records,
            expected_pairs=prepared.plan["parameters"][
                "measured_iterations_per_arm"
            ],
            bootstrap_samples=BOOTSTRAP_SAMPLES,
            seed=BOOTSTRAP_SEED,
        )
        decisions = _preregistered_decisions(paired_bootstrap, memory)
        _atomic_json(output_dir / "paired_bootstrap.json", paired_bootstrap)
        _atomic_json(output_dir / "preregistered_decision.json", decisions)
        environment = _environment(
            device,
            nvidia_smi_start=nvidia_smi_start,
        )
        _atomic_json(output_dir / "environment.json", environment)
        summary = {
            "schema_version": 2,
            "status": "COMPLETE",
            "plan_sha256": file_sha256(output_dir / "plan.json"),
            "repository": repository,
            "environment": environment,
            "model_contracts": model_contracts,
            "captured_batches": sample_manifest,
            "parity": {
                "metadata": "raw-bit match",
                "outputs_runtime_state_rng": "raw-bit match",
            },
            "transfer_profile": profile_summary,
            "memory": memory,
            "measurement_contract": {
                "physical_round_order": list(PHYSICAL_ROUND_ORDER),
                "warmup_invocations_per_arm_excluded": prepared.plan[
                    "parameters"
                ]["warmup_iterations_per_arm"],
                "measured_invocations_per_arm": prepared.plan["parameters"][
                    "measured_iterations_per_arm"
                ],
                "warmup_iterations_per_arm_segment": prepared.plan[
                    "parameters"
                ]["warmup_iterations_per_segment"],
                "measured_iterations_per_arm_segment": prepared.plan[
                    "parameters"
                ]["measured_iterations_per_segment"],
                "dataloader_io_in_primary_timing": False,
                "evaluator_serialization_in_primary_timing": False,
            },
            "timing_statistics": statistics,
            "paired_bootstrap": paired_bootstrap,
            "preregistered_decisions": decisions,
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
                    "paired_bootstrap.json",
                    "preregistered_decision.json",
                    "environment_start.json",
                    "environment.json",
                )
            },
        }
        _atomic_json(output_dir / "summary.json", summary)
        return summary
    except Exception as error:
        failure = {
            "schema_version": 2,
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
    if args.execute:
        if args.execution_state != "ARMED":
            parser.error("--execute requires --execution-state ARMED")
        if args.idle_gpu_confirmation != IDLE_GPU_CONFIRMATION:
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
