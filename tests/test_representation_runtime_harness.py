from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

from mmengine import Config
import pytest
import torch

from rosa.runtime_verification import (
    CANDIDATE_ARM,
    INPUT_PIPELINE_SCOPE,
    MODEL_ONLY_SCOPE,
    PARENT_ARM,
    PHYSICAL_ROUND_ORDER,
    REPRESENTATION_PROTOCOL,
    VerificationContractError,
    build_alternating_schedule,
    build_representation_pair_manifest,
    evaluate_input_pipeline_gate,
    evaluate_model_only_non_regression_gate,
    paired_bootstrap_confidence_intervals,
    validate_common_preprocessor_boundary,
    validate_representation_config_pair,
    validate_representation_ingress,
    validate_representation_snapshot,
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
                "elapsed_ns": (
                    parent_ns if entry.arm == PARENT_ARM else candidate_ns
                ),
                "sample_count": 4,
            }
        )
        records.append(record)
    return records


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
    _, _, pair = _pair_manifest()
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
        [pair],
        common_boundary_parity=True,
    )

    assert model_gate["benchmark_scope"] == MODEL_ONLY_SCOPE
    assert model_gate["speed_claim_eligible"] is False
    assert pipeline_gate["benchmark_scope"] == INPUT_PIPELINE_SCOPE
    assert pipeline_gate["scope_excludes_model_execution"] is True
    assert pipeline_gate["passed"] is True
    with pytest.raises(VerificationContractError, match="boundary parity"):
        evaluate_input_pipeline_gate(
            pipeline_bootstrap,
            [pair],
            common_boundary_parity=False,
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
    assert prepared.plan["benchmark_scopes"]["model_only"]["name"] == (
        MODEL_ONLY_SCOPE
    )
    assert prepared.plan["benchmark_scopes"]["input_pipeline_end_to_end"][
        "name"
    ] == INPUT_PIPELINE_SCOPE
    assert prepared.plan["benchmark_scopes"]["combined_full_inference"][
        "measured"
    ] is False
    assert not output.exists()


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
    prepared = SimpleNamespace(
        plan={"parameters": {"expected_batch_size": 1}}
    )

    document = verify_tool._prevalidate_live_representation_sequence(
        prepared,
        pairs_per_arm=2,
    )

    assert document["pairs_per_arm"] == 2
    assert [pair["sample_identities"][0]["sample_idx"] for pair in document["pairs"]] == [
        10,
        11,
    ]
    assert len({pair["semantic_sha256"] for pair in document["pairs"]}) == 2


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
