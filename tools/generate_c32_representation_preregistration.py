#!/usr/bin/env python3
"""Generate the disarmed 360-pair C32 representation preregistration.

The default action only prints the frozen plan.  ``--generate`` reads the
validation dataset and hashes source/data payloads; it never builds a model,
runs inference, performs timing, or authorizes the later CUDA benchmark.
Run it only after active training has released enough host I/O bandwidth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from mmengine import Config

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import verify_c32_runtime as runtime_tool  # noqa: E402
from rosa.runtime_verification import (  # noqa: E402
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    PHYSICAL_ROUND_ORDER,
    REPRESENTATION_CAPTURE_BATCHES,
    REPRESENTATION_C32_E20_CHECKPOINT_SHA256,
    REPRESENTATION_DATALOADER_WORKERS,
    REPRESENTATION_EXPECTED_BATCH_SIZE,
    REPRESENTATION_MEASURED_ITERATIONS_PER_ARM,
    REPRESENTATION_PREREGISTRATION_SCHEMA,
    REPRESENTATION_PROTOCOL,
    REPRESENTATION_SEED,
    REPRESENTATION_SEQUENCE_PAIRS_PER_ARM,
    REPRESENTATION_WARMUP_ITERATIONS_PER_ARM,
    VerificationContractError,
    canonical_sha256,
    file_sha256,
    validate_representation_config_pair,
    validate_representation_protocol_freeze,
    validate_representation_sequence_document,
)

CONFIRMATION = "I_ACCEPT_READ_ONLY_360_PAIR_PREREGISTRATION_NO_TIMING"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent-config",
        type=Path,
        default=runtime_tool.DEFAULT_REPRESENTATION_PARENT_CONFIG,
    )
    parser.add_argument(
        "--candidate-config",
        type=Path,
        default=runtime_tool.DEFAULT_REPRESENTATION_CANDIDATE_CONFIG,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--detector-host-root",
        type=Path,
        default=runtime_tool.DEFAULT_DETECTOR_HOST_ROOT,
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=runtime_tool.DEFAULT_OFFICIAL_ROOT,
    )
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--confirmation")
    return parser


def _source_document(root: Path, label: str) -> dict[str, Any]:
    root = runtime_tool._resolve_dir(root, f"{label} root")
    files = []
    for path in runtime_tool._source_inventory(root):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    if not files:
        raise VerificationContractError(f"{label} source inventory is empty")
    document = {
        "schema": "rosa.complete_source_content_manifest.v2",
        "label": label,
        "root": str(root),
        "excluded_parts": list(runtime_tool.SOURCE_CLOSURE_EXCLUDED_PARTS),
        "files": files,
    }
    document["sha256"] = canonical_sha256(document)
    runtime_tool._validate_source_closure(
        document,
        expected_root=root,
        label=label,
    )
    return document


def _dataset_document(
    parent_config: Config,
    candidate_config: Config,
    sequence: Mapping[str, Any],
) -> dict[str, Any]:
    document = runtime_tool._derive_dataset_payload_contract(
        parent_config=parent_config,
        candidate_config=candidate_config,
        representation_sequence=sequence,
    )
    runtime_tool._validate_dataset_payload_closure(
        document,
        parent_config=parent_config,
        candidate_config=candidate_config,
        representation_sequence=sequence,
    )
    return document


def _write_json(path: Path, value: Any) -> None:
    runtime_tool._atomic_json(path, value)


def _plan_only(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": REPRESENTATION_PREREGISTRATION_SCHEMA,
        "status": "DISARMED_GENERATOR_PLAN",
        "execution_authorized": False,
        "benchmark_execution": False,
        "timing_execution": False,
        "pairs_to_read": REPRESENTATION_SEQUENCE_PAIRS_PER_ARM,
        "samples_to_read_per_arm": (
            REPRESENTATION_SEQUENCE_PAIRS_PER_ARM * REPRESENTATION_EXPECTED_BATCH_SIZE
        ),
        "output_dir": str(args.output_dir.expanduser().resolve()),
        "confirmation_required": CONFIRMATION,
    }


def generate(args: argparse.Namespace) -> dict[str, Any]:
    if args.confirmation != CONFIRMATION:
        raise VerificationContractError(
            f"--generate requires --confirmation {CONFIRMATION}"
        )
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise VerificationContractError(
            f"output directory already exists: {output_dir}"
        )
    parent_path = runtime_tool._resolve_file(args.parent_config, "parent config")
    candidate_path = runtime_tool._resolve_file(
        args.candidate_config,
        "candidate config",
    )
    checkpoint = runtime_tool._resolve_file(args.checkpoint, "checkpoint")
    parent = Config.fromfile(parent_path, import_custom_modules=False)
    candidate = Config.fromfile(candidate_path, import_custom_modules=False)
    contract = validate_representation_config_pair(
        parent,
        candidate,
        expected_batch_size=REPRESENTATION_EXPECTED_BATCH_SIZE,
    )
    parameters = {
        "capture_batches": REPRESENTATION_CAPTURE_BATCHES,
        "warmup_iterations_per_arm": REPRESENTATION_WARMUP_ITERATIONS_PER_ARM,
        "measured_iterations_per_arm": REPRESENTATION_MEASURED_ITERATIONS_PER_ARM,
        "expected_batch_size": REPRESENTATION_EXPECTED_BATCH_SIZE,
        "profile_batch_index": 0,
    }
    frozen = validate_representation_protocol_freeze(
        parameters=parameters,
        dataloader_workers=REPRESENTATION_DATALOADER_WORKERS,
        seed=REPRESENTATION_SEED,
        checkpoint_sha256=file_sha256(checkpoint),
        config_contract=contract,
    )
    prepared = SimpleNamespace(
        parent_config=parent,
        candidate_config=candidate,
        plan={
            "seed": REPRESENTATION_SEED,
            "dataloader_workers": REPRESENTATION_DATALOADER_WORKERS,
            "parameters": parameters,
        },
    )
    sequence = runtime_tool._prevalidate_live_representation_sequence(
        prepared,
        pairs_per_arm=REPRESENTATION_SEQUENCE_PAIRS_PER_ARM,
    )
    validate_representation_sequence_document(sequence)
    documents = {
        "REPRESENTATION_SEQUENCE_MANIFEST.json": sequence,
        "ROSA_SOURCE_CLOSURE.json": _source_document(ROOT, "ROSA"),
        "DETECTOR_HOST_SOURCE_CLOSURE.json": _source_document(
            args.detector_host_root,
            "detector host",
        ),
        "OFFICIAL_SOURCE_CLOSURE.json": _source_document(
            args.official_root,
            "official",
        ),
        "DATASET_PAYLOAD_CLOSURE.json": _dataset_document(
            parent,
            candidate,
            sequence,
        ),
    }
    section_files = {
        "sequence_manifest": "REPRESENTATION_SEQUENCE_MANIFEST.json",
        "rosa_source_closure": "ROSA_SOURCE_CLOSURE.json",
        "detector_host_source_closure": "DETECTOR_HOST_SOURCE_CLOSURE.json",
        "official_source_closure": "OFFICIAL_SOURCE_CLOSURE.json",
        "dataset_payload_closure": "DATASET_PAYLOAD_CLOSURE.json",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    for filename, document in documents.items():
        _write_json(output_dir / filename, document)
    plan = {
        "schema": REPRESENTATION_PREREGISTRATION_SCHEMA,
        "status": "SEALED",
        "protocol": REPRESENTATION_PROTOCOL,
        "execution_authorized": False,
        "frozen_contract": {
            **frozen,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "physical_round_order": list(PHYSICAL_ROUND_ORDER),
        },
        "generator_contract": {
            "benchmark_execution": False,
            "timing_execution": False,
            "model_built": False,
            "checkpoint_loaded": False,
            "checkpoint_content_only_hashed": True,
        },
    }
    for section, filename in section_files.items():
        plan[section] = {
            "status": "SEALED",
            "file": filename,
            "sha256": file_sha256(output_dir / filename),
        }
    _write_json(output_dir / "PLAN.json", plan)
    entries = []
    for path in sorted(output_dir.iterdir(), key=lambda value: value.name):
        entries.append(f"{file_sha256(path)}  {path.name}")
    (output_dir / "SHA256SUMS").write_text(
        "\n".join(entries) + "\n",
        encoding="utf-8",
    )
    # Close the time-of-check/time-of-use gap for source and dataset hashing.
    runtime_tool._load_representation_preregistration(
        output_dir,
        explicit_sha256=file_sha256(output_dir / "SHA256SUMS"),
    )
    runtime_tool._validate_source_closure(
        documents["ROSA_SOURCE_CLOSURE.json"],
        expected_root=ROOT,
        label="ROSA",
    )
    runtime_tool._validate_source_closure(
        documents["DETECTOR_HOST_SOURCE_CLOSURE.json"],
        expected_root=Path(args.detector_host_root),
        label="detector host",
    )
    runtime_tool._validate_source_closure(
        documents["OFFICIAL_SOURCE_CLOSURE.json"],
        expected_root=Path(args.official_root),
        label="official",
    )
    runtime_tool._validate_dataset_payload_closure(
        documents["DATASET_PAYLOAD_CLOSURE.json"],
        parent_config=parent,
        candidate_config=candidate,
        representation_sequence=sequence,
    )
    return {
        "status": "SEALED_DISARMED_PREREGISTRATION",
        "execution_authorized": False,
        "output_dir": str(output_dir),
        "sha256sums_sha256": file_sha256(output_dir / "SHA256SUMS"),
        "pairs": REPRESENTATION_SEQUENCE_PAIRS_PER_ARM,
        "checkpoint_sha256": REPRESENTATION_C32_E20_CHECKPOINT_SHA256,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.generate:
        print(json.dumps(_plan_only(args), indent=2, sort_keys=True))
        return 0
    print(json.dumps(generate(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
