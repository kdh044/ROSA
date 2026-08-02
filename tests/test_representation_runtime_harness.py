from __future__ import annotations

import copy
import gc
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

from mmengine import Config
import numpy as np
import pytest
import torch

from rosa.runtime_verification import (
    CANDIDATE_ARM,
    INPUT_PIPELINE_SCOPE,
    MODEL_ONLY_SCOPE,
    PARENT_ARM,
    PHYSICAL_ROUND_ORDER,
    REPRESENTATION_C32_E20_CHECKPOINT_SHA256,
    REPRESENTATION_CANDIDATE_NORMALIZED_CONFIG_SHA256,
    REPRESENTATION_PARENT_NORMALIZED_CONFIG_SHA256,
    REPRESENTATION_PROTOCOL,
    REPRESENTATION_SEQUENCE_PAIRS_PER_ARM,
    VerificationContractError,
    build_alternating_schedule,
    build_representation_pair_manifest,
    build_timed_common_boundary_receipt,
    evaluate_input_pipeline_gate,
    evaluate_model_only_non_regression_gate,
    paired_bootstrap_confidence_intervals,
    canonical_sha256,
    validate_common_preprocessor_boundary,
    validate_live_representation_observation,
    validate_pipeline_memory_baseline_receipt,
    validate_representation_config_pair,
    validate_representation_ingress,
    validate_representation_protocol_freeze,
    validate_representation_sequence_document,
    validate_representation_snapshot,
    validate_timed_common_boundary_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
STEM = (
    "rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_"
    "cpu_metadata_lean_loader"
)
PARENT_CONFIG = CONFIG_DIR / f"{STEM}.py"
CANDIDATE_CONFIG = CONFIG_DIR / f"{STEM}_uint8_transfer_pinned.py"
_SPEC = importlib.util.spec_from_file_location(
    "rosa_verify_representation_runtime_tool",
    ROOT / "tools" / "verify_c32_runtime.py",
)
assert _SPEC is not None and _SPEC.loader is not None
verify_tool = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = verify_tool
_SPEC.loader.exec_module(verify_tool)
_GENERATOR_SPEC = importlib.util.spec_from_file_location(
    "rosa_generate_representation_preregistration_tool",
    ROOT / "tools" / "generate_c32_representation_preregistration.py",
)
assert _GENERATOR_SPEC is not None and _GENERATOR_SPEC.loader is not None
generator_tool = importlib.util.module_from_spec(_GENERATOR_SPEC)
sys.modules[_GENERATOR_SPEC.name] = generator_tool
_GENERATOR_SPEC.loader.exec_module(generator_tool)


def _batch(*, dtype: torch.dtype, sample_idx: int = 10) -> dict:
    compact = (
        torch.arange(4 * 3 * 800 * 800, dtype=torch.int64)
        .remainder(256)
        .to(torch.uint8)
        .reshape(1, 4, 3, 800, 800)
    )
    image = compact if dtype == torch.uint8 else compact.float()
    return {
        "inputs": {
            "img": image,
            "T_global": torch.eye(4).reshape(1, 4, 4),
            "T_global_inv": torch.eye(4).reshape(1, 4, 4),
            "timestamp": torch.tensor([123.0], dtype=torch.float64),
            "lidar2cam": torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 4, 1, 1),
        },
        "data_samples": [
            {
                "metainfo": {
                    "sample_idx": sample_idx,
                    "scene_token": "scene-a",
                    "timestamp": 123,
                }
            }
        ],
    }


def _pair_manifest(sample_idx: int = 10):
    parent = _batch(dtype=torch.float32, sample_idx=sample_idx)
    candidate = _batch(dtype=torch.uint8, sample_idx=sample_idx)
    manifest = build_representation_pair_manifest(
        parent,
        candidate,
        batch_index=0,
        expected_batch_size=1,
        observed_pinned={PARENT_ARM: False, CANDIDATE_ARM: True},
    )
    return parent, candidate, manifest


def _scope_records(
    *,
    parent_ns: int,
    candidate_ns: int,
    scope: str,
) -> list[dict]:
    schedule = build_alternating_schedule(
        iterations_per_arm=300,
        batch_count=2,
        phase="measured",
    )
    records = []
    for sequence_index, entry in enumerate(schedule):
        record = entry.to_dict()
        record.update(
            {
                "sequence_index": sequence_index,
                "benchmark_scope": scope,
                "semantic_batch_sha256": f"{entry.pair_index:064x}",
                "elapsed_ns": (parent_ns if entry.arm == PARENT_ARM else candidate_ns),
                "sample_count": 4,
            }
        )
        records.append(record)
    return records


def _timed_boundary_fixture():
    pairs = []
    observations = {}
    for pair_index in range(REPRESENTATION_SEQUENCE_PAIRS_PER_ARM):
        semantic = f"{pair_index + 1:064x}"
        identities = [
            {"token": f"sample-{pair_index:03d}-{sample_index}"}
            for sample_index in range(4)
        ]
        arms = {}
        for arm, offset, image_bytes in (
            (PARENT_ARM, 2000, 400),
            (CANDIDATE_ARM, 3000, 100),
        ):
            arm_manifest = {
                "arm": arm,
                "batch_index": pair_index,
                "input_keys": ["img"],
                "sample_identities": identities,
                "semantic_sha256": semantic,
                "representation_fingerprint": {
                    "sha256": f"{pair_index + offset:064x}",
                    "nodes": 1,
                    "tensor_bytes": 1,
                },
                "ingress": {"image_bytes": image_bytes},
            }
            arm_manifest["sha256"] = canonical_sha256(arm_manifest)
            arms[arm] = arm_manifest
        pair = {
            "schema": "rosa.representation_pair.v1",
            "batch_index": pair_index,
            "sample_identities": identities,
            "semantic_sha256": semantic,
            "arms": arms,
        }
        pair["sha256"] = canonical_sha256(pair)
        pairs.append(pair)
        boundary = {
            "sha256": f"{pair_index + 1000:064x}",
            "nodes": 10,
            "tensor_bytes": 100,
        }
        observations[pair_index] = {
            arm: {
                "arm": arm,
                "sequence_position": pair_index,
                "observed_semantic_sha256": semantic,
                "observed_representation_fingerprint": {
                    "sha256": f"{pair_index + offset:064x}",
                    "nodes": 1,
                    "tensor_bytes": 1,
                },
                "observed_arm_manifest_sha256": arms[arm]["sha256"],
                "exact_arm_manifest_match": True,
                "boundary_fingerprint": boundary,
            }
            for arm, offset in ((PARENT_ARM, 2000), (CANDIDATE_ARM, 3000))
        }
    receipt = build_timed_common_boundary_receipt(observations, pairs)
    return pairs, observations, receipt


def test_exact_optimization_configs_satisfy_sealed_representation_contract():
    parent = Config.fromfile(PARENT_CONFIG, import_custom_modules=False)
    candidate = Config.fromfile(CANDIDATE_CONFIG, import_custom_modules=False)

    contract = validate_representation_config_pair(
        parent,
        candidate,
        expected_batch_size=4,
    )

    assert contract["protocol"] == REPRESENTATION_PROTOCOL
    assert contract["parent_ingress_dtype"] == "torch.float32"
    assert contract["candidate_ingress_dtype"] == "torch.uint8"
    assert contract["pin_memory"]["test_dataloader"] == {
        PARENT_ARM: False,
        CANDIDATE_ARM: True,
    }
    assert contract["non_blocking"] == {
        PARENT_ARM: False,
        CANDIDATE_ARM: True,
    }
    assert contract["stream_policy"] == "torch-current-default-stream-only"
    assert contract["parent_normalized_config_sha256"] == (
        REPRESENTATION_PARENT_NORMALIZED_CONFIG_SHA256
    )
    assert contract["candidate_normalized_config_sha256"] == (
        REPRESENTATION_CANDIDATE_NORMALIZED_CONFIG_SHA256
    )


def _frozen_protocol_arguments():
    return {
        "parameters": {
            "capture_batches": 2,
            "warmup_iterations_per_arm": 60,
            "measured_iterations_per_arm": 300,
            "expected_batch_size": 4,
            "profile_batch_index": 0,
        },
        "dataloader_workers": 0,
        "seed": 3407,
        "checkpoint_sha256": REPRESENTATION_C32_E20_CHECKPOINT_SHA256,
        "config_contract": {
            "parent_normalized_config_sha256": (
                REPRESENTATION_PARENT_NORMALIZED_CONFIG_SHA256
            ),
            "candidate_normalized_config_sha256": (
                REPRESENTATION_CANDIDATE_NORMALIZED_CONFIG_SHA256
            ),
        },
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("capture_batches", 3, "capture_batches=2"),
        ("warmup_iterations_per_arm", 72, "warmup_iterations_per_arm=60"),
        ("measured_iterations_per_arm", 312, "measured_iterations_per_arm=300"),
        ("expected_batch_size", 8, "expected_batch_size=4"),
    ),
)
def test_representation_timing_shape_is_not_tunable(field, value, message):
    arguments = _frozen_protocol_arguments()
    arguments["parameters"][field] = value
    with pytest.raises(VerificationContractError, match=message):
        validate_representation_protocol_freeze(**arguments)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("dataloader_workers", 1, "workers=0"),
        ("seed", 1, "seed=3407"),
        ("checkpoint_sha256", "0" * 64, "C32 e20"),
    ),
)
def test_representation_environment_identity_is_not_tunable(
    field,
    value,
    message,
):
    arguments = _frozen_protocol_arguments()
    arguments[field] = value
    with pytest.raises(VerificationContractError, match=message):
        validate_representation_protocol_freeze(**arguments)


def test_representation_config_digests_are_hard_pinned():
    arguments = _frozen_protocol_arguments()
    arguments["config_contract"]["candidate_normalized_config_sha256"] = "0" * 64
    with pytest.raises(VerificationContractError, match="candidate normalized"):
        validate_representation_protocol_freeze(**arguments)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda config: config["test_dataloader"].update(pin_memory=False),
            "pin_memory",
        ),
        (
            lambda config: config["model"]["data_preprocessor"].update(
                non_blocking=False
            ),
            "non_blocking",
        ),
        (
            lambda config: config["test_dataloader"]["dataset"]["pipeline"][-1].update(
                type="PackSparse4DInputs"
            ),
            "candidate packer",
        ),
        (
            lambda config: config["model"]["head"].update(hidden_drift=1),
            "differ beyond",
        ),
        (
            lambda config: config["model"]["data_preprocessor"].update(
                transfer_stream="side"
            ),
            "custom CUDA stream",
        ),
    ),
)
def test_representation_config_drift_fails_closed(mutation, message):
    parent = Config.fromfile(PARENT_CONFIG, import_custom_modules=False).to_dict()
    candidate = Config.fromfile(CANDIDATE_CONFIG, import_custom_modules=False).to_dict()
    mutation(candidate)

    with pytest.raises(VerificationContractError, match=message):
        validate_representation_config_pair(
            parent,
            candidate,
            expected_batch_size=4,
        )


def test_semantic_digest_binds_distinct_fp32_and_uint8_representations():
    parent, candidate, pair = _pair_manifest()

    assert pair["arms"][PARENT_ARM]["semantic_sha256"] == pair["semantic_sha256"]
    assert pair["arms"][CANDIDATE_ARM]["semantic_sha256"] == pair["semantic_sha256"]
    assert pair["arms"][PARENT_ARM]["representation_fingerprint"] != (
        pair["arms"][CANDIDATE_ARM]["representation_fingerprint"]
    )
    assert pair["arms"][PARENT_ARM]["ingress"]["image_bytes"] == (
        4 * pair["arms"][CANDIDATE_ARM]["ingress"]["image_bytes"]
    )
    validate_representation_snapshot(
        parent,
        pair["arms"][PARENT_ARM],
        label="parent",
    )
    validate_representation_snapshot(
        candidate,
        pair["arms"][CANDIDATE_ARM],
        label="candidate",
    )


def test_pair_swap_and_mutation_fail_closed():
    parent, candidate, pair = _pair_manifest()
    swapped_candidate = _batch(dtype=torch.uint8, sample_idx=11)

    with pytest.raises(VerificationContractError, match="swapped or different"):
        build_representation_pair_manifest(
            parent,
            swapped_candidate,
            batch_index=0,
            expected_batch_size=1,
            observed_pinned={PARENT_ARM: False, CANDIDATE_ARM: True},
        )

    candidate["inputs"]["img"][0, 0, 0, 0, 0] ^= 1
    with pytest.raises(VerificationContractError, match="mutated"):
        validate_representation_snapshot(
            candidate,
            pair["arms"][CANDIDATE_ARM],
            label="candidate",
        )


def test_live_arm_snapshot_recomputes_pixels_and_semantic_metadata():
    parent, candidate, pair = _pair_manifest()

    observed = validate_live_representation_observation(
        candidate,
        pair,
        arm=CANDIDATE_ARM,
        sequence_position=0,
        expected_batch_size=1,
        observed_pinned=True,
    )

    assert observed["observed_semantic_sha256"] == pair["semantic_sha256"]
    assert (
        observed["observed_arm_manifest_sha256"]
        == pair["arms"][CANDIDATE_ARM]["sha256"]
    )
    pixel_attack = copy.deepcopy(candidate)
    pixel_attack["inputs"]["img"][0, 0, 0, 0, 0] ^= 1
    with pytest.raises(VerificationContractError, match="snapshot differs"):
        validate_live_representation_observation(
            pixel_attack,
            pair,
            arm=CANDIDATE_ARM,
            sequence_position=0,
            expected_batch_size=1,
            observed_pinned=True,
        )
    metadata_attack = copy.deepcopy(parent)
    metadata_attack["inputs"]["timestamp"][0] += 1
    with pytest.raises(VerificationContractError, match="snapshot differs"):
        validate_live_representation_observation(
            metadata_attack,
            pair,
            arm=PARENT_ARM,
            sequence_position=0,
            expected_batch_size=1,
            observed_pinned=False,
        )


def test_live_arm_snapshot_rejects_expected_digest_copy_attack():
    _parent, candidate, pair = _pair_manifest()
    attacked = copy.deepcopy(candidate)
    attacked["data_samples"][0]["metainfo"]["timestamp"] = 999

    with pytest.raises(VerificationContractError, match="snapshot differs"):
        validate_live_representation_observation(
            attacked,
            pair,
            arm=CANDIDATE_ARM,
            sequence_position=0,
            expected_batch_size=1,
            observed_pinned=True,
        )


def test_ingress_dtype_layout_and_pinning_are_fail_closed():
    parent = _batch(dtype=torch.float32)
    candidate = _batch(dtype=torch.uint8)

    validate_representation_ingress(
        parent,
        arm=PARENT_ARM,
        expected_batch_size=1,
        observed_pinned=False,
    )
    validate_representation_ingress(
        candidate,
        arm=CANDIDATE_ARM,
        expected_batch_size=1,
        observed_pinned=True,
    )
    with pytest.raises(VerificationContractError, match="dtype/shape/layout"):
        validate_representation_ingress(
            candidate,
            arm=PARENT_ARM,
            expected_batch_size=1,
            observed_pinned=False,
        )
    with pytest.raises(VerificationContractError, match="pinned"):
        validate_representation_ingress(
            candidate,
            arm=CANDIDATE_ARM,
            expected_batch_size=1,
            observed_pinned=False,
        )
    transposed = copy.deepcopy(candidate)
    transposed["inputs"]["img"] = transposed["inputs"]["img"].transpose(-1, -2)
    with pytest.raises(VerificationContractError, match="dtype/shape/layout"):
        validate_representation_ingress(
            transposed,
            arm=CANDIDATE_ARM,
            expected_batch_size=1,
            observed_pinned=True,
        )


def test_common_preprocessor_boundary_is_separate_and_raw_bit_exact():
    common = {
        "inputs": {
            "img": torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2),
            "timestamp": torch.tensor([1.0]),
        },
        "data_samples": [{"metainfo": {"sample_idx": 10}}],
    }

    boundary = validate_common_preprocessor_boundary(
        common,
        copy.deepcopy(common),
        expected_device_type="cpu",
    )

    assert boundary["parity"] == "raw-bit exact"
    changed = copy.deepcopy(common)
    changed["inputs"]["img"][0, 0, 0, 0] += 1
    with pytest.raises(VerificationContractError, match="raw-bit mismatch"):
        validate_common_preprocessor_boundary(
            common,
            changed,
            expected_device_type="cpu",
        )


def test_timed_common_boundary_receipt_covers_every_360_pair():
    pairs, _observations, receipt = _timed_boundary_fixture()

    contract = validate_timed_common_boundary_receipt(receipt, pairs)

    assert contract["pairs"] == 360
    assert contract["all_timed_pairs_covered"] is True
    assert contract["parity"] == "raw-bit exact"


def test_sequence_rejects_point_payload_key_and_arm_input_key_drift():
    pairs, _observations, _receipt = _timed_boundary_fixture()
    document = {
        "schema": "rosa.live_representation_pair_sequence.v2",
        "protocol": REPRESENTATION_PROTOCOL,
        "pairs_per_arm": 360,
        "pairs": pairs,
        "sha256": canonical_sha256(pairs),
    }
    contract = validate_representation_sequence_document(document)
    assert contract["input_keys"] == ["img"]

    point_payload = copy.deepcopy(document)
    arm = point_payload["pairs"][0]["arms"][CANDIDATE_ARM]
    arm["input_keys"] = ["img", "semantic_lidar_points"]
    arm["sha256"] = canonical_sha256(
        {key: value for key, value in arm.items() if key != "sha256"}
    )
    pair = point_payload["pairs"][0]
    pair["sha256"] = canonical_sha256(
        {key: value for key, value in pair.items() if key != "sha256"}
    )
    point_payload["sha256"] = canonical_sha256(point_payload["pairs"])
    with pytest.raises(VerificationContractError, match="LiDAR scope"):
        validate_representation_sequence_document(point_payload)


def test_timed_common_boundary_receipt_rejects_missing_or_changed_pair():
    pairs, observations, receipt = _timed_boundary_fixture()
    observations.pop(17)
    with pytest.raises(VerificationContractError, match="incomplete"):
        build_timed_common_boundary_receipt(observations, pairs)

    changed = copy.deepcopy(receipt)
    changed["records"][17]["arms"][PARENT_ARM][
        "raw_boundary_fingerprint"
    ]["sha256"] = "f" * 64
    with pytest.raises(VerificationContractError, match="fingerprints differ"):
        validate_timed_common_boundary_receipt(changed, pairs)


def test_timed_common_boundary_receipt_rejects_garbage_raw_fingerprint():
    pairs, _observations, receipt = _timed_boundary_fixture()
    forged = copy.deepcopy(receipt)
    forged["records"][0]["arms"][PARENT_ARM][
        "raw_boundary_fingerprint"
    ] = "one-common-garbage-value"
    forged["sha256"] = canonical_sha256(
        {key: value for key, value in forged.items() if key != "sha256"}
    )

    with pytest.raises(VerificationContractError, match="invalid raw schema"):
        validate_timed_common_boundary_receipt(forged, pairs)


def test_timed_common_boundary_receipt_rejects_single_common_forgery():
    pairs, _observations, receipt = _timed_boundary_fixture()
    forged = copy.deepcopy(receipt)
    record = forged["records"][0]
    common = record["arms"][PARENT_ARM]["raw_boundary_fingerprint"]
    record.pop("arms")
    record["boundary_fingerprint"] = common
    forged["sha256"] = canonical_sha256(
        {key: value for key, value in forged.items() if key != "sha256"}
    )

    with pytest.raises(VerificationContractError, match="record schema"):
        validate_timed_common_boundary_receipt(forged, pairs)


def test_timed_common_boundary_receipt_rejects_forged_live_arm_sha():
    pairs, _observations, receipt = _timed_boundary_fixture()
    forged = copy.deepcopy(receipt)
    forged["records"][9]["arms"][CANDIDATE_ARM][
        "live_arm_manifest_sha256"
    ] = "f" * 64
    forged["sha256"] = canonical_sha256(
        {key: value for key, value in forged.items() if key != "sha256"}
    )

    with pytest.raises(VerificationContractError, match="arm manifest SHA"):
        validate_timed_common_boundary_receipt(forged, pairs)


def _pipeline_memory_receipt():
    before = {
        "full_models": {
            "weakref_targets": 20,
            "weakref_alive": 20,
            "tracked_container_targets": 0,
            "tracked_container_alive": 0,
        },
        "raw_cpu_batches": {
            "weakref_targets": 8,
            "weakref_alive": 8,
            "tracked_container_targets": 12,
            "tracked_container_alive": 12,
        },
        "processed_cuda_batches": {
            "weakref_targets": 8,
            "weakref_alive": 8,
            "tracked_container_targets": 12,
            "tracked_container_alive": 12,
        },
        "pipeline_preprocessors": {
            "weakref_targets": 2,
            "weakref_alive": 2,
            "tracked_container_targets": 0,
            "tracked_container_alive": 0,
        },
    }
    after = copy.deepcopy(before)
    for category in (
        "full_models",
        "raw_cpu_batches",
        "processed_cuda_batches",
    ):
        after[category]["weakref_alive"] = 0
        after[category]["tracked_container_alive"] = 0
    receipt = {
        "schema": "rosa.pipeline_memory_baseline_receipt.v3",
        "clean_pipeline_baseline": {
            "allocated_bytes": 100,
            "reserved_bytes": 500,
        },
        "before_release": {"allocated_bytes": 1000, "reserved_bytes": 2000},
        "after_release": {"allocated_bytes": 100, "reserved_bytes": 500},
        "pipeline_baseline": {"allocated_bytes": 100, "reserved_bytes": 500},
        "liveness": {"before": before, "after": after},
    }
    receipt["sha256"] = canonical_sha256(receipt)
    return receipt


def test_pipeline_memory_release_receipt_is_required_and_closed():
    receipt = _pipeline_memory_receipt()
    contract = validate_pipeline_memory_baseline_receipt(receipt)
    assert contract["pipeline_baseline"] == receipt["after_release"]

    retained = copy.deepcopy(receipt)
    retained["liveness"]["after"]["full_models"]["weakref_alive"] = 1
    retained["sha256"] = canonical_sha256(
        {key: value for key, value in retained.items() if key != "sha256"}
    )
    with pytest.raises(VerificationContractError, match="retained live"):
        validate_pipeline_memory_baseline_receipt(retained)

    changed_baseline = copy.deepcopy(receipt)
    changed_baseline["pipeline_baseline"]["allocated_bytes"] += 1
    changed_baseline["sha256"] = canonical_sha256(
        {key: value for key, value in changed_baseline.items() if key != "sha256"}
    )
    with pytest.raises(VerificationContractError, match="baseline changed"):
        validate_pipeline_memory_baseline_receipt(changed_baseline)


def test_pipeline_memory_receipt_rejects_reserved_growth_and_self_report():
    reserved_growth = _pipeline_memory_receipt()
    reserved_growth["after_release"]["reserved_bytes"] += 1
    reserved_growth["pipeline_baseline"]["reserved_bytes"] += 1
    reserved_growth["sha256"] = canonical_sha256(
        {
            key: value
            for key, value in reserved_growth.items()
            if key != "sha256"
        }
    )
    with pytest.raises(VerificationContractError, match="reserved_bytes"):
        validate_pipeline_memory_baseline_receipt(reserved_growth)

    self_report = _pipeline_memory_receipt()
    self_report["release_actions"] = {"full_models_released": True}
    self_report["sha256"] = canonical_sha256(
        {key: value for key, value in self_report.items() if key != "sha256"}
    )
    with pytest.raises(VerificationContractError, match="self-reported"):
        validate_pipeline_memory_baseline_receipt(self_report)


def _actual_cpu_release_probe():
    models = {
        PARENT_ARM: torch.nn.Linear(3, 2),
        CANDIDATE_ARM: torch.nn.Linear(3, 2),
    }
    raw_batches = {
        PARENT_ARM: [{"inputs": {"img": torch.ones(1, 3)}}],
        CANDIDATE_ARM: [{"inputs": {"img": torch.ones(1, 3)}}],
    }
    processed_batches = {
        PARENT_ARM: [{"inputs": {"img": torch.ones(1, 3)}}],
        CANDIDATE_ARM: [{"inputs": {"img": torch.ones(1, 3)}}],
    }
    preprocessors = {
        PARENT_ARM: torch.nn.Identity(),
        CANDIDATE_ARM: torch.nn.Identity(),
    }
    probe = verify_tool._capture_release_liveness_probe(
        models=models,
        raw_cpu_batches=raw_batches,
        processed_batches=processed_batches,
        pipeline_preprocessors=preprocessors,
        processed_tensor_device="cpu",
    )
    before = verify_tool._observe_release_liveness(probe)
    return probe, preprocessors, before


def test_cpu_release_probe_measures_real_object_liveness_without_cuda():
    probe, preprocessors, before = _actual_cpu_release_probe()
    gc.collect()
    after = verify_tool._observe_release_liveness(probe)

    assert before["full_models"]["weakref_alive"] >= 2
    assert before["raw_cpu_batches"]["weakref_alive"] == 2
    assert before["processed_cuda_batches"]["weakref_alive"] == 2
    for category in (
        "full_models",
        "raw_cpu_batches",
        "processed_cuda_batches",
    ):
        assert after[category]["weakref_alive"] == 0
        assert after[category]["tracked_container_alive"] == 0
    assert after["pipeline_preprocessors"]["weakref_alive"] == 2
    assert set(preprocessors) == {PARENT_ARM, CANDIDATE_ARM}

    receipt = {
        "schema": "rosa.pipeline_memory_baseline_receipt.v3",
        "clean_pipeline_baseline": {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
        },
        "before_release": {"allocated_bytes": 0, "reserved_bytes": 0},
        "after_release": {"allocated_bytes": 0, "reserved_bytes": 0},
        "pipeline_baseline": {"allocated_bytes": 0, "reserved_bytes": 0},
        "liveness": {"before": before, "after": after},
    }
    receipt["sha256"] = canonical_sha256(receipt)
    validate_pipeline_memory_baseline_receipt(receipt)


def test_semantic_pair_swap_is_rejected_before_bootstrap():
    records = _scope_records(
        parent_ns=100,
        candidate_ns=90,
        scope=INPUT_PIPELINE_SCOPE,
    )
    candidate = next(
        record
        for record in records
        if record["arm"] == CANDIDATE_ARM and record["pair_index"] == 0
    )
    candidate["semantic_batch_sha256"] = f"{999:064x}"

    with pytest.raises(VerificationContractError, match="schedule metadata"):
        paired_bootstrap_confidence_intervals(
            records,
            expected_pairs=300,
            bootstrap_samples=1,
            required_pair_fields=("benchmark_scope", "semantic_batch_sha256"),
        )


def test_model_only_and_input_pipeline_gates_are_disjoint():
    pairs, _observations, boundary_receipt = _timed_boundary_fixture()
    model_bootstrap = paired_bootstrap_confidence_intervals(
        _scope_records(parent_ns=100, candidate_ns=100, scope=MODEL_ONLY_SCOPE),
        expected_pairs=300,
        bootstrap_samples=20,
        required_pair_fields=("benchmark_scope", "semantic_batch_sha256"),
    )
    pipeline_bootstrap = paired_bootstrap_confidence_intervals(
        _scope_records(
            parent_ns=100,
            candidate_ns=90,
            scope=INPUT_PIPELINE_SCOPE,
        ),
        expected_pairs=300,
        bootstrap_samples=20,
        required_pair_fields=("benchmark_scope", "semantic_batch_sha256"),
    )

    model_gate = evaluate_model_only_non_regression_gate(model_bootstrap)
    pipeline_gate = evaluate_input_pipeline_gate(
        pipeline_bootstrap,
        pairs,
        common_boundary_receipt=boundary_receipt,
    )

    assert model_gate["benchmark_scope"] == MODEL_ONLY_SCOPE
    assert model_gate["speed_claim_eligible"] is False
    assert pipeline_gate["benchmark_scope"] == INPUT_PIPELINE_SCOPE
    assert pipeline_gate["scope_excludes_model_execution"] is True
    assert pipeline_gate["passed"] is True
    changed_receipt = copy.deepcopy(boundary_receipt)
    changed_receipt["records"][0]["semantic_sha256"] = "f" * 64
    with pytest.raises(VerificationContractError, match="semantic sequence"):
        evaluate_input_pipeline_gate(
            pipeline_bootstrap,
            pairs,
            common_boundary_receipt=changed_receipt,
        )


def test_representation_plan_is_cpu_only_and_preserves_six_rounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint = tmp_path / "native.pth"
    output = tmp_path / "must-not-exist"
    torch.save({"state_dict": {"head.decoder.weight": torch.ones(1)}}, checkpoint)

    def forbidden_cuda_query(*_args, **_kwargs):
        raise AssertionError("CPU representation planning queried CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden_cuda_query)
    original_file_sha256 = verify_tool.file_sha256

    def frozen_checkpoint_digest(path):
        if Path(path).resolve() == checkpoint.resolve():
            return REPRESENTATION_C32_E20_CHECKPOINT_SHA256
        return original_file_sha256(path)

    monkeypatch.setattr(verify_tool, "file_sha256", frozen_checkpoint_digest)
    args = verify_tool.build_parser().parse_args(
        [
            "--protocol",
            REPRESENTATION_PROTOCOL,
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output),
        ]
    )

    prepared = verify_tool.prepare_verification(args)

    assert prepared.plan["status"] == "DRY_RUN"
    assert prepared.plan["execution_state"] == "DISARMED"
    assert prepared.plan["protocol"] == REPRESENTATION_PROTOCOL
    assert prepared.plan["schedule"]["physical_round_order"] == list(
        PHYSICAL_ROUND_ORDER
    )
    assert prepared.plan["parameters"]["measured_iterations_per_arm"] == 300
    assert prepared.plan["armed_execution_ready"] is False
    assert (
        prepared.plan["representation_protocol_freeze"]["checkpoint_sha256"]
        == REPRESENTATION_C32_E20_CHECKPOINT_SHA256
    )
    assert prepared.plan["benchmark_scopes"]["model_only"]["name"] == (MODEL_ONLY_SCOPE)
    assert (
        prepared.plan["benchmark_scopes"]["input_pipeline_end_to_end"]["name"]
        == INPUT_PIPELINE_SCOPE
    )
    assert (
        prepared.plan["benchmark_scopes"]["combined_full_inference"]["measured"]
        is False
    )
    assert not output.exists()


def test_representation_preregistration_loader_is_immutable_and_fail_closed(
    tmp_path: Path,
):
    source = verify_tool.DEFAULT_REPRESENTATION_PREREGISTRATION_DIR
    copied = tmp_path / "prereg"
    shutil.copytree(source, copied)
    receipt_sha = verify_tool.file_sha256(copied / "SHA256SUMS")

    _plan, sequence, contract = verify_tool._load_representation_preregistration(
        copied,
        explicit_sha256=receipt_sha,
    )

    assert sequence is None
    assert contract["ready_for_armed_execution"] is False
    with pytest.raises(VerificationContractError, match="requires"):
        verify_tool._load_representation_preregistration(
            copied,
            explicit_sha256=None,
        )
    plan = json.loads((copied / "PLAN.json").read_text(encoding="utf-8"))
    plan["frozen_contract"]["seed"] = 1
    (copied / "PLAN.json").write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(VerificationContractError, match="artifact hash mismatch"):
        verify_tool._load_representation_preregistration(
            copied,
            explicit_sha256=receipt_sha,
        )


def test_preregistration_rejects_artifact_outside_sha_closure(tmp_path: Path):
    copied = tmp_path / "prereg"
    shutil.copytree(
        verify_tool.DEFAULT_REPRESENTATION_PREREGISTRATION_DIR,
        copied,
    )
    (copied / "UNSEALED.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(VerificationContractError, match="unsealed or missing"):
        verify_tool._load_representation_preregistration(
            copied,
            explicit_sha256=verify_tool.file_sha256(copied / "SHA256SUMS"),
        )


def test_logical_cuda_device_is_mapped_to_nvidia_smi_by_torch_uuid(monkeypatch):
    properties = SimpleNamespace(uuid="GPU-test-uuid", name="Test GPU")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: properties,
    )
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        if any("--query-gpu=" in item for item in command):
            return SimpleNamespace(
                stdout=(
                    "2026/08/02, 7, GPU-test-uuid, Test GPU, 999, P0, "
                    "100, 200, 1000, 2000\n"
                ),
                stderr="",
            )
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(verify_tool.subprocess, "run", fake_run)
    device = torch.device("cuda:0")

    identity = verify_tool._logical_cuda_device_uuid(device)
    snapshot = verify_tool._nvidia_smi_snapshot(identity["gpu_uuid"])

    assert identity["logical_index"] == 0
    assert identity["gpu_uuid"] == "GPU-test-uuid"
    assert snapshot["selected_gpu_uuid"] == "GPU-test-uuid"
    assert all("--id=GPU-test-uuid" in command for command in commands)
    assert all("--id=0" not in command for command in commands)


def test_nvidia_smi_uuid_mismatch_fails_closed(monkeypatch):
    def fake_run(command, **_kwargs):
        if any("--query-gpu=" in item for item in command):
            return SimpleNamespace(
                stdout=(
                    "2026/08/02, 0, GPU-wrong, Test GPU, 999, P0, "
                    "100, 200, 1000, 2000\n"
                ),
                stderr="",
            )
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(verify_tool.subprocess, "run", fake_run)
    with pytest.raises(VerificationContractError, match="does not match"):
        verify_tool._nvidia_smi_snapshot("GPU-expected")


def test_final_receipt_closes_immutable_plan_summary_and_every_artifact(
    tmp_path: Path,
):
    output = tmp_path / "run"
    output.mkdir()
    (output / "plan.json").write_text('{"immutable":true}\n', encoding="utf-8")
    (output / "summary.json").write_text('{"status":"COMPLETE"}\n', encoding="utf-8")
    (output / "raw.jsonl").write_text('{"elapsed_ns":1}\n', encoding="utf-8")

    sealed = verify_tool._seal_final_artifact_receipt(output)

    names = {item["name"] for item in sealed["receipt"]["artifacts"]}
    assert names == {"plan.json", "summary.json", "raw.jsonl"}
    assert sealed["receipt"]["self_exclusion"] == "final_receipt.json"
    assert verify_tool.file_sha256(output / "final_receipt.json") == sealed["sha256"]
    with pytest.raises(VerificationContractError, match="already exists"):
        verify_tool._seal_final_artifact_receipt(output)


def test_final_receipt_revalidates_live_closure_before_and_after_write(
    tmp_path: Path,
):
    output = tmp_path / "run"
    output.mkdir()
    (output / "plan.json").write_text("{}\n", encoding="utf-8")
    (output / "summary.json").write_text("{}\n", encoding="utf-8")
    expected = {
        "source_closures": {"rosa": "a" * 64},
        "dataset_payload_closure": {"sha256": "b" * 64},
        "git_snapshots": {"rosa": {"tree": "c" * 40}},
    }
    calls = []

    def verify_live_closure():
        calls.append(True)
        return copy.deepcopy(expected)

    sealed = verify_tool._seal_final_artifact_receipt(
        output,
        expected_live_closure=expected,
        closure_verifier=verify_live_closure,
    )

    assert len(calls) == 2
    names = {item["name"] for item in sealed["receipt"]["artifacts"]}
    assert "content_closure_finalize.json" in names
    assert sealed["receipt"]["live_content_closure_sha256"] == canonical_sha256(
        expected
    )
    verify_tool._verify_final_receipt_live_closure(
        sealed["receipt"],
        expected,
    )
    changed = copy.deepcopy(expected)
    changed["dataset_payload_closure"]["sha256"] = "d" * 64
    with pytest.raises(VerificationContractError, match="changed after sealing"):
        verify_tool._verify_final_receipt_live_closure(
            sealed["receipt"],
            changed,
        )


def test_final_receipt_fails_if_external_closure_changes_during_finalize(
    tmp_path: Path,
):
    output = tmp_path / "run"
    output.mkdir()
    (output / "plan.json").write_text("{}\n", encoding="utf-8")
    (output / "summary.json").write_text("{}\n", encoding="utf-8")
    expected = {"checkpoint": "a" * 64}
    calls = 0

    def verify_live_closure():
        nonlocal calls
        calls += 1
        if calls == 1:
            return copy.deepcopy(expected)
        return {"checkpoint": "b" * 64}

    with pytest.raises(VerificationContractError, match="changed while"):
        verify_tool._seal_final_artifact_receipt(
            output,
            expected_live_closure=expected,
            closure_verifier=verify_live_closure,
        )


def test_exact_source_content_closure_detects_dirty_payload_mutation(tmp_path: Path):
    root = tmp_path / "external"
    root.mkdir()
    source = root / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    record = {
        "path": "module.py",
        "size": source.stat().st_size,
        "sha256": verify_tool.file_sha256(source),
    }
    document = {
        "schema": "rosa.complete_source_content_manifest.v2",
        "label": "external",
        "root": str(root.resolve()),
        "excluded_parts": list(verify_tool.SOURCE_CLOSURE_EXCLUDED_PARTS),
        "files": [record],
    }
    document["sha256"] = canonical_sha256(document)

    contract = verify_tool._validate_source_closure(
        document,
        expected_root=root,
        label="external",
    )
    assert contract["files"] == 1
    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(VerificationContractError, match="payload changed"):
        verify_tool._validate_source_closure(
            document,
            expected_root=root,
            label="external",
        )


def _mini_dataset_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    camera_names = (
        "fisheye_camera_front",
        "fisheye_camera_left",
        "fisheye_camera_right",
        "fisheye_camera_rear",
    )
    cameras = {}
    for camera_index, camera_name in enumerate(camera_names):
        relative = Path("images") / f"{camera_name}.png"
        image = data_root / relative
        image.parent.mkdir(exist_ok=True)
        image.write_bytes(bytes([camera_index + 1]) * (camera_index + 2))
        ocam = SimpleNamespace(
            _pol=np.array([1.0, camera_index], dtype=np.float64),
            _invpol=np.array([2.0, camera_index], dtype=np.float64),
            _xc=400.0,
            _yc=400.0,
            _affine=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            _img_size=np.array([800, 800], dtype=np.int64),
            _fov=220.0,
        )
        cameras[camera_name] = {
            "cam_path": relative.as_posix(),
            "image_size": [800, 800],
            "cam2img": ocam,
            "cam2lidar": np.eye(4, dtype=np.float64),
            "lidar2cam": np.eye(4, dtype=np.float64),
            "lidar2img": np.eye(4, dtype=np.float64),
        }
    annotation = data_root / "infos_val.pkl"
    annotation.write_bytes(b"sealed-mini-annotation")
    annotation_payload = {
        "data_list": [
            {
                "token": "sample-0",
                "pc_info": {
                    "pts_path": "points/sample-0.bin",
                    "semantic_pts_path": "points/sample-0.label",
                },
                "cam_info": {"cam_fisheye": cameras},
            }
        ]
    }
    pipeline_parent = [
        {
            "type": verify_tool.EXPECTED_LEAN_LOADER,
            "cam_names": camera_names,
            "to_float32": True,
        },
        {"type": "PackSparse4DInputs"},
    ]
    pipeline_candidate = copy.deepcopy(pipeline_parent)
    pipeline_candidate[0]["to_float32"] = False
    parent_path = tmp_path / "parent.py"
    candidate_path = tmp_path / "candidate.py"
    parent_path.write_text("# parent\n", encoding="utf-8")
    candidate_path.write_text("# candidate\n", encoding="utf-8")
    parent = Config(
        {
            "test_dataloader": {
                "dataset": {
                    "data_root": str(data_root),
                    "ann_file": str(annotation),
                    "pipeline": pipeline_parent,
                }
            }
        },
        filename=parent_path,
    )
    candidate = Config(
        {
            "test_dataloader": {
                "dataset": {
                    "data_root": str(data_root),
                    "ann_file": str(annotation),
                    "pipeline": pipeline_candidate,
                }
            }
        },
        filename=candidate_path,
    )
    sequence = {
        "schema": "rosa.test-mini-sequence.v1",
        "sha256": "e" * 64,
        "pairs": [
            {
                "sample_identities": [{"token": "sample-0"}],
                "arms": {
                    PARENT_ARM: {"input_keys": ["img"]},
                    CANDIDATE_ARM: {"input_keys": ["img"]},
                },
            }
        ],
    }
    monkeypatch.setattr(verify_tool, "REPRESENTATION_SEQUENCE_PAIRS_PER_ARM", 1)
    monkeypatch.setattr(verify_tool, "REPRESENTATION_EXPECTED_BATCH_SIZE", 1)
    monkeypatch.setattr(
        verify_tool,
        "validate_representation_sequence_document",
        lambda _sequence: {"sha256": "e" * 64},
    )
    monkeypatch.setattr(
        verify_tool,
        "validate_representation_config_pair",
        lambda *_args, **_kwargs: {
            "parent_normalized_config_sha256": "a" * 64,
            "candidate_normalized_config_sha256": "b" * 64,
        },
    )
    monkeypatch.setattr(verify_tool, "load", lambda _path: annotation_payload)
    document = verify_tool._derive_dataset_payload_contract(
        parent_config=parent,
        candidate_config=candidate,
        representation_sequence=sequence,
    )
    return document, parent, candidate, sequence, annotation_payload


@pytest.mark.parametrize(
    "attack",
    ("missing_camera", "extra_unrelated", "substituted_camera", "unrelated_only"),
)
def test_dataset_closure_rejects_missing_extra_substitute_and_unrelated_files(
    attack: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    document, parent, candidate, sequence, _payload = _mini_dataset_closure(
        tmp_path,
        monkeypatch,
    )
    assert document["camera_reference_count"] == 4
    assert document["expected_file_count"] == 5
    assert document["lidar_scope_proof"]["annotation_lidar_reference_count"] == 2
    forged = copy.deepcopy(document)
    unrelated = {
        "root": document["annotation_binding"]["root"],
        "path": "unrelated.bin",
        "size": 0,
        "sha256": "f" * 64,
        "role": "unrelated",
    }
    if attack == "missing_camera":
        forged["camera_references"].pop()
        forged["camera_reference_count"] -= 1
        forged["files"].pop()
        forged["expected_file_count"] -= 1
    elif attack == "extra_unrelated":
        forged["files"].append(unrelated)
        forged["expected_file_count"] += 1
    elif attack == "substituted_camera":
        forged["camera_references"][0]["path"] = forged["camera_references"][1][
            "path"
        ]
        forged["camera_references"][0]["image_sha256"] = forged[
            "camera_references"
        ][1]["image_sha256"]
    else:
        forged["files"] = [unrelated]
        forged["expected_file_count"] = 1
    forged.pop("sha256")
    forged["sha256"] = canonical_sha256(forged)

    with pytest.raises(VerificationContractError, match="differs from the re-derived"):
        verify_tool._validate_dataset_payload_closure(
            forged,
            parent_config=parent,
            candidate_config=candidate,
            representation_sequence=sequence,
        )


def test_dataset_closure_rederives_identity_order_and_ocam_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    document, parent, candidate, sequence, payload = _mini_dataset_closure(
        tmp_path,
        monkeypatch,
    )
    changed_sequence = copy.deepcopy(sequence)
    changed_sequence["pairs"][0]["sample_identities"][0]["token"] = "substitute"
    with pytest.raises(VerificationContractError, match="identity/order changed"):
        verify_tool._validate_dataset_payload_closure(
            document,
            parent_config=parent,
            candidate_config=candidate,
            representation_sequence=changed_sequence,
        )

    first_camera = payload["data_list"][0]["cam_info"]["cam_fisheye"][
        "fisheye_camera_front"
    ]
    first_camera["cam2img"]._pol[0] += 0.5
    with pytest.raises(VerificationContractError, match="differs from the re-derived"):
        verify_tool._validate_dataset_payload_closure(
            document,
            parent_config=parent,
            candidate_config=candidate,
            representation_sequence=sequence,
        )


def test_live_sequence_prevalidation_fingerprints_every_semantic_pair(
    monkeypatch: pytest.MonkeyPatch,
):
    parent_batches = [
        _batch(dtype=torch.float32, sample_idx=index) for index in (10, 11)
    ]
    candidate_batches = [
        _batch(dtype=torch.uint8, sample_idx=index) for index in (10, 11)
    ]
    monkeypatch.setattr(
        verify_tool,
        "_build_representation_dataloaders",
        lambda _prepared: {
            PARENT_ARM: parent_batches,
            CANDIDATE_ARM: candidate_batches,
        },
    )
    monkeypatch.setattr(
        torch.Tensor,
        "is_pinned",
        lambda tensor: tensor.dtype == torch.uint8,
    )
    prepared = SimpleNamespace(plan={"parameters": {"expected_batch_size": 1}})

    document = verify_tool._prevalidate_live_representation_sequence(
        prepared,
        pairs_per_arm=2,
    )

    assert document["pairs_per_arm"] == 2
    assert [
        pair["sample_identities"][0]["sample_idx"] for pair in document["pairs"]
    ] == [
        10,
        11,
    ]
    assert len({pair["semantic_sha256"] for pair in document["pairs"]}) == 2


def test_manifest_generator_default_is_disarmed_cpu_static(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    output = tmp_path / "must-not-exist"

    def forbidden_cuda_query(*_args, **_kwargs):
        raise AssertionError("disarmed manifest generator queried CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden_cuda_query)
    result = generator_tool.main(
        [
            "--checkpoint",
            str(tmp_path / "not-read-in-plan-mode.pth"),
            "--output-dir",
            str(output),
        ]
    )
    printed = json.loads(capsys.readouterr().out)

    assert result == 0
    assert printed["status"] == "DISARMED_GENERATOR_PLAN"
    assert printed["benchmark_execution"] is False
    assert printed["timing_execution"] is False
    assert not output.exists()


def test_execute_requires_explicit_armed_state_before_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        verify_tool,
        "prepare_verification",
        lambda _args: pytest.fail("DISARMED request reached preparation"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: pytest.fail("DISARMED request queried CUDA"),
    )

    with pytest.raises(SystemExit) as error:
        verify_tool.main(
            [
                "--protocol",
                REPRESENTATION_PROTOCOL,
                "--checkpoint",
                str(tmp_path / "checkpoint.pth"),
                "--output-dir",
                str(tmp_path / "out"),
                "--execute",
                "--idle-gpu-confirmation",
                verify_tool.IDLE_GPU_CONFIRMATION,
            ]
        )

    assert error.value.code == 2


def test_direct_execution_rejects_disarmed_object_before_cuda_query(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: pytest.fail("DISARMED direct call queried CUDA"),
    )
    prepared = SimpleNamespace(
        args=SimpleNamespace(
            execution_state="DISARMED",
            idle_gpu_confirmation=verify_tool.IDLE_GPU_CONFIRMATION,
        )
    )

    with pytest.raises(VerificationContractError, match="execution-state ARMED"):
        verify_tool.execute_verification(prepared)


def test_placeholder_preregistration_blocks_armed_execution_before_cuda_query(
    monkeypatch,
):
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: pytest.fail("placeholder ARMED request queried CUDA"),
    )
    prepared = SimpleNamespace(
        args=SimpleNamespace(
            execution_state="ARMED",
            idle_gpu_confirmation=verify_tool.IDLE_GPU_CONFIRMATION,
            device="cuda:0",
        ),
        plan={
            "protocol": REPRESENTATION_PROTOCOL,
            "armed_execution_ready": False,
        },
    )

    with pytest.raises(VerificationContractError, match="DISARMED placeholder"):
        verify_tool.execute_verification(prepared)
