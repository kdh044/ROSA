from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from mmengine.config import Config
import numpy as np
import pytest
import torch

from rosa.runtime_verification import (
    BOOTSTRAP_SEED,
    CANDIDATE_ARM,
    DEFAULT_MEASURED_ITERATIONS_PER_ARM,
    DEFAULT_WARMUP_ITERATIONS_PER_ARM,
    PARENT_ARM,
    PHYSICAL_ROUND_ORDER,
    VerificationContractError,
    build_alternating_schedule,
    build_batch_manifest,
    classify_transfer_event,
    evaluate_c32_near_native_gate,
    evaluate_strict_lightweight_gate,
    paired_bootstrap_confidence_intervals,
    percentile,
    raw_fingerprint,
    require_same_fingerprint,
    summarize_timings,
    summarize_transfer_event_names,
    validate_config_pair,
    validate_expected_manifest,
    validate_preregistration_plan,
    validate_runtime_parameters,
    validate_transfer_reduction,
)


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "rosa_verify_c32_runtime_tool",
    ROOT / "tools" / "verify_c32_runtime.py",
)
assert _SPEC is not None and _SPEC.loader is not None
verify_tool = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = verify_tool
_SPEC.loader.exec_module(verify_tool)


def _config(preprocessor: str) -> dict:
    return {
        "model": {
            "type": "Fisheye3DODGravityCenterSparse4D",
            "data_preprocessor": {"type": preprocessor},
            "head": {
                "type": "ROSAOfficialResidualNativeBaselineHead",
                "instance_bank": {"num_temp_instances": 0},
                "operation_order": ["gnn", "norm", "deformable", "refine"],
            },
        },
        "test_dataloader": {
            "batch_size": 4,
            "sampler": {"type": "DefaultSampler", "shuffle": False},
        },
        "work_dir": "ignored",
        "experiment_contract": {"arm": preprocessor},
    }


def _paired_configs() -> tuple[dict, dict]:
    parent = _config("Sparse4DDataPreprocessor")
    candidate = _config("Fisheye3DODSingleFrameDataPreprocessor")
    return parent, candidate


def test_schedule_is_balanced_and_preregistered_per_batch():
    entries = build_alternating_schedule(
        iterations_per_arm=DEFAULT_MEASURED_ITERATIONS_PER_ARM,
        batch_count=2,
        phase="measured",
    )

    assert len(entries) == 600
    assert tuple(
        entries[round_index * 100].pattern for round_index in range(6)
    ) == PHYSICAL_ROUND_ORDER
    assert [entry.arm for entry in entries[:100]] == (
        [PARENT_ARM] * 25
        + [CANDIDATE_ARM] * 50
        + [PARENT_ARM] * 25
    )
    assert [entry.arm for entry in entries[100:200]] == (
        [CANDIDATE_ARM] * 25
        + [PARENT_ARM] * 50
        + [CANDIDATE_ARM] * 25
    )
    assert sum(entry.arm == PARENT_ARM for entry in entries) == 300
    assert sum(entry.arm == CANDIDATE_ARM for entry in entries) == 300
    assert [entry.batch_index for entry in entries[:12]] == [
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
    ]
    assert [entry.pair_index for entry in entries[:8]] == list(range(8))
    assert [entry.pair_index for entry in entries[25:33]] == list(range(8))
    for round_index, pattern in enumerate(PHYSICAL_ROUND_ORDER):
        round_entries = [
            entry for entry in entries if entry.round_index == round_index
        ]
        assert len(round_entries) == 100
        assert {entry.pattern for entry in round_entries} == {pattern}
        assert sum(entry.arm == PARENT_ARM for entry in round_entries) == 50
        assert sum(entry.arm == CANDIDATE_ARM for entry in round_entries) == 50
        for slot_index in range(4):
            segment = [
                entry
                for entry in round_entries
                if entry.slot_index == slot_index
            ]
            assert len(segment) == 25
            assert len({entry.arm for entry in segment}) == 1
    for batch_index in (0, 1):
        assert sum(
            entry.arm == PARENT_ARM and entry.batch_index == batch_index
            for entry in entries
        ) == 150
        assert sum(
            entry.arm == CANDIDATE_ARM and entry.batch_index == batch_index
            for entry in entries
        ) == 150

    warmup = build_alternating_schedule(
        iterations_per_arm=DEFAULT_WARMUP_ITERATIONS_PER_ARM,
        batch_count=2,
        phase="warmup",
    )
    assert len(warmup) == 120
    for round_index in range(6):
        round_entries = [
            entry for entry in warmup if entry.round_index == round_index
        ]
        assert sum(entry.arm == PARENT_ARM for entry in round_entries) == 10
        assert sum(entry.arm == CANDIDATE_ARM for entry in round_entries) == 10
        assert all(
            sum(entry.slot_index == slot_index for entry in round_entries) == 5
            for slot_index in range(4)
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            {"warmup_iterations_per_arm": 49},
            "warmup_iterations_per_arm must be >= 50",
        ),
        (
            {"warmup_iterations_per_arm": 50},
            "must be a multiple of 12",
        ),
        (
            {"measured_iterations_per_arm": 299},
            "measured_iterations_per_arm must be >= 300",
        ),
        (
            {"measured_iterations_per_arm": 301},
            "must be a multiple of 12",
        ),
        ({"capture_batches": 0}, "capture_batches must be >= 1"),
        ({"profile_batch_index": 2}, "select a captured batch"),
    ],
)
def test_runtime_parameters_fail_closed(overrides, match):
    values = {
        "capture_batches": 2,
        "warmup_iterations_per_arm": DEFAULT_WARMUP_ITERATIONS_PER_ARM,
        "measured_iterations_per_arm": DEFAULT_MEASURED_ITERATIONS_PER_ARM,
        "expected_batch_size": 4,
        "profile_batch_index": 0,
    }
    values.update(overrides)

    with pytest.raises(VerificationContractError, match=match):
        validate_runtime_parameters(**values)


def test_timing_summary_uses_raw_iterations_and_sample_counts():
    records = [
        {"arm": PARENT_ARM, "elapsed_ns": 1_000_000, "sample_count": 4},
        {"arm": CANDIDATE_ARM, "elapsed_ns": 500_000, "sample_count": 4},
        {"arm": PARENT_ARM, "elapsed_ns": 3_000_000, "sample_count": 4},
    ]

    summary = summarize_timings(records, arm=PARENT_ARM)

    assert percentile([1.0, 3.0], 0.5) == 2.0
    assert summary["iterations"] == 2
    assert summary["samples"] == 8
    assert summary["median_iteration_ms"] == 2.0
    assert summary["p95_iteration_ms"] == pytest.approx(2.9)
    assert summary["samples_per_second"] == pytest.approx(2000.0)


def _constant_timing_records(parent_ns: int, candidate_ns: int):
    schedule = build_alternating_schedule(
        iterations_per_arm=DEFAULT_MEASURED_ITERATIONS_PER_ARM,
        batch_count=2,
        phase="measured",
    )
    records = []
    for sequence_index, entry in enumerate(schedule):
        record = entry.to_dict()
        record.update(
            {
                "sequence_index": sequence_index,
                "elapsed_ns": (
                    parent_ns if entry.arm == PARENT_ARM else candidate_ns
                ),
                "sample_count": 4,
            }
        )
        records.append(record)
    return records


def test_deterministic_paired_bootstrap_uses_all_300_pairs():
    records = _constant_timing_records(100, 90)

    first = paired_bootstrap_confidence_intervals(
        records,
        expected_pairs=300,
        bootstrap_samples=200,
        seed=BOOTSTRAP_SEED,
    )
    second = paired_bootstrap_confidence_intervals(
        records,
        expected_pairs=300,
        bootstrap_samples=200,
        seed=BOOTSTRAP_SEED,
    )

    assert first == second
    assert first["pairs"] == 300
    assert first["physical_round_order"] == list(PHYSICAL_ROUND_ORDER)
    median = first["metrics"]["median_candidate_over_parent"]
    assert median["point_estimate"] == pytest.approx(0.9)
    assert median["ci95"] == pytest.approx({"lower": 0.9, "upper": 0.9})
    assert first["metrics"]["p95_candidate_over_parent"][
        "point_estimate"
    ] == pytest.approx(0.9)
    assert first["metrics"]["throughput_candidate_over_parent"][
        "point_estimate"
    ] == pytest.approx(100 / 90)


def test_paired_bootstrap_rejects_missing_or_misaligned_pairs():
    records = _constant_timing_records(100, 90)
    with pytest.raises(VerificationContractError, match="exactly one P and one C"):
        paired_bootstrap_confidence_intervals(
            records[:-1],
            expected_pairs=300,
            bootstrap_samples=1,
        )

    changed = copy.deepcopy(records)
    candidate = next(
        record
        for record in changed
        if record["arm"] == CANDIDATE_ARM and record["pair_index"] == 0
    )
    candidate["batch_index"] = 1
    with pytest.raises(VerificationContractError, match="schedule metadata"):
        paired_bootstrap_confidence_intervals(
            changed,
            expected_pairs=300,
            bootstrap_samples=1,
        )


def test_strict_lightweight_decision_gate_is_fail_closed():
    bootstrap = paired_bootstrap_confidence_intervals(
        _constant_timing_records(100, 90),
        expected_pairs=300,
        bootstrap_samples=20,
    )
    memory = {
        PARENT_ARM: {
            "peak_allocated_bytes": 100,
            "peak_reserved_bytes": 200,
        },
        CANDIDATE_ARM: {
            "peak_allocated_bytes": 90,
            "peak_reserved_bytes": 190,
        },
    }

    passing = evaluate_strict_lightweight_gate(bootstrap, memory)

    assert passing["passed"] is True
    assert passing["decision"] == "ADOPT_CANDIDATE"
    reported = verify_tool._preregistered_decisions(bootstrap, memory)
    assert reported["strict_lightweight"]["passed"] is True
    assert reported["c32_vs_c0_near_native"]["claim_eligible"] is False
    assert reported["c32_vs_c0_near_native"]["decision"] == (
        "NOT_APPLICABLE_TO_CURRENT_ARM_PAIR"
    )

    inconclusive_bootstrap = copy.deepcopy(bootstrap)
    inconclusive_bootstrap["metrics"]["median_candidate_over_parent"]["ci95"][
        "upper"
    ] = 1.0
    memory[CANDIDATE_ARM]["peak_reserved_bytes"] = 201
    failing = evaluate_strict_lightweight_gate(
        inconclusive_bootstrap,
        memory,
    )
    assert failing["passed"] is False
    assert failing["decision"] == "RETAIN_PARENT_AND_REPORT_INCONCLUSIVE"
    assert failing["checks"]["median_ratio_ci95_upper_below_1"]["passed"] is False
    assert failing["checks"]["peak_reserved_memory_not_worse"]["passed"] is False


def test_c32_near_native_gate_uses_ci_upper_and_p95_limits():
    bootstrap = paired_bootstrap_confidence_intervals(
        _constant_timing_records(100, 105),
        expected_pairs=300,
        bootstrap_samples=20,
    )

    passing = evaluate_c32_near_native_gate(bootstrap)

    assert passing["passed"] is True
    assert passing["decision"] == "NEAR_NATIVE_SUPPORTED"

    boundary = copy.deepcopy(bootstrap)
    boundary["metrics"]["median_candidate_over_parent"]["ci95"]["upper"] = 1.10
    boundary["metrics"]["p95_candidate_over_parent"]["point_estimate"] = 1.10
    assert evaluate_c32_near_native_gate(boundary)["passed"] is True

    failing_bootstrap = copy.deepcopy(bootstrap)
    failing_bootstrap["metrics"]["median_candidate_over_parent"]["ci95"][
        "upper"
    ] = 1.1000001
    failing_bootstrap["metrics"]["p95_candidate_over_parent"][
        "point_estimate"
    ] = 1.1000001
    failing = evaluate_c32_near_native_gate(failing_bootstrap)
    assert failing["passed"] is False
    assert failing["decision"] == "NEAR_NATIVE_NOT_SUPPORTED"
    assert failing["checks"][
        "median_ratio_ci95_upper_at_most_1_10"
    ]["passed"] is False
    assert failing["checks"]["p95_ratio_at_most_1_10"]["passed"] is False


def test_exact_preregistration_artifact_is_bound_and_tampering_fails():
    preregistration_plan = json.loads(
        (verify_tool.DEFAULT_PREREGISTRATION_DIR / "PLAN.json").read_text(
            encoding="utf-8"
        )
    )

    contract = validate_preregistration_plan(preregistration_plan)
    loaded, evidence = verify_tool._load_preregistration(
        verify_tool.DEFAULT_PREREGISTRATION_DIR
    )

    assert loaded == preregistration_plan
    assert contract["minimum_warmup_iterations_per_arm"] == 50
    assert contract["minimum_measured_iterations_per_arm"] == 300
    assert contract["physical_round_order"] == list(PHYSICAL_ROUND_ORDER)
    assert contract["c32_vs_c0_near_native_gate"] == {
        "same_checkpoint_and_inputs": True,
        "upper_95ci_median_ratio_max": 1.1,
        "p95_ratio_max": 1.1,
    }
    assert evidence["files"]["PLAN.json"]["sha256"] == (
        verify_tool.EXPECTED_PREREGISTRATION_HASHES["PLAN.json"]
    )

    changed = copy.deepcopy(preregistration_plan)
    changed["future_cuda_gate"]["model_test_step_benchmark"][
        "measured_iterations_per_arm_min"
    ] = 299
    with pytest.raises(VerificationContractError, match="minimum changed"):
        validate_preregistration_plan(changed)


def test_config_pair_allows_only_the_preprocessor_delta():
    parent, candidate = _paired_configs()

    contract = validate_config_pair(parent, candidate, expected_batch_size=4)

    assert contract["parent_preprocessor"] == "Sparse4DDataPreprocessor"
    assert contract["candidate_preprocessor"] == (
        "Fisheye3DODSingleFrameDataPreprocessor"
    )

    candidate["model"]["head"]["operation_order"].append("temp_gnn")
    with pytest.raises(VerificationContractError, match="temporal-free"):
        validate_config_pair(parent, candidate, expected_batch_size=4)


def test_frozen_c32_configs_satisfy_the_runtime_contract():
    parent = Config.fromfile(
        ROOT
        / "configs/rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual.py",
        import_custom_modules=False,
    )
    candidate = Config.fromfile(
        ROOT
        / (
            "configs/rosa_r18_main126_v2_c32_cbgs_native_official_gnn_"
            "residual_cpu_metadata.py"
        ),
        import_custom_modules=False,
    )

    contract = validate_config_pair(parent, candidate, expected_batch_size=4)

    assert contract["model_type"] == "Fisheye3DODGravityCenterSparse4D"


def test_config_pair_rejects_other_deltas_and_shuffled_samples():
    parent, candidate = _paired_configs()
    candidate["model"]["head"]["instance_bank"]["num_temp_instances"] = 1
    with pytest.raises(VerificationContractError, match="exact integer zero"):
        validate_config_pair(parent, candidate, expected_batch_size=4)

    parent, candidate = _paired_configs()
    candidate["test_dataloader"]["sampler"]["shuffle"] = True
    with pytest.raises(VerificationContractError, match="shuffle=False"):
        validate_config_pair(parent, candidate, expected_batch_size=4)

    parent, candidate = _paired_configs()
    candidate["model"]["head"]["new_behavior"] = True
    with pytest.raises(VerificationContractError, match="differ beyond"):
        validate_config_pair(parent, candidate, expected_batch_size=4)


def test_raw_fingerprints_compare_tensor_and_array_bits():
    value = {
        "tensor": torch.tensor([1.0, -0.0], dtype=torch.float32),
        "array": np.array([1, 2], dtype=np.int16),
    }
    clone = copy.deepcopy(value)

    assert raw_fingerprint(value) == raw_fingerprint(clone)
    assert require_same_fingerprint("test", value, clone) == raw_fingerprint(value)
    clone["tensor"][1] = 0.0
    with pytest.raises(VerificationContractError, match="raw-bit mismatch"):
        require_same_fingerprint("test", value, clone)


def test_batch_manifest_locks_real_sample_identity_and_cpu_metadata():
    batch = {
        "inputs": {
            "img": torch.arange(8, dtype=torch.float32).reshape(2, 4),
            "T_global": torch.eye(4).repeat(2, 1, 1),
            "T_global_inv": torch.eye(4).repeat(2, 1, 1),
            "timestamp": torch.tensor([1.0, 2.0], dtype=torch.float64),
        },
        "data_samples": [
            SimpleNamespace(metainfo={"sample_idx": 10}),
            SimpleNamespace(metainfo={"sample_idx": 11}),
        ],
    }

    manifest = build_batch_manifest(batch, batch_index=0, expected_batch_size=2)

    assert manifest["batch_size"] == 2
    assert manifest["sample_identities"] == [
        {"sample_idx": 10},
        {"sample_idx": 11},
    ]
    validate_expected_manifest(manifest, copy.deepcopy(manifest))
    changed = copy.deepcopy(manifest)
    changed["sample_identities"][0]["sample_idx"] = 99
    with pytest.raises(VerificationContractError, match="does not match"):
        validate_expected_manifest(manifest, changed)

    batch["inputs"]["timestamp"] = batch["inputs"]["timestamp"][::2]
    with pytest.raises(VerificationContractError, match="batch dimension"):
        build_batch_manifest(batch, batch_index=0, expected_batch_size=2)


def test_transfer_profile_classification_and_preregistered_reduction():
    assert classify_transfer_event("[CUDA memcpy DtoH]") == "device_to_host"
    assert classify_transfer_event("cudaMemcpy HtoD") == "host_to_device"
    assert classify_transfer_event("aten::matmul") is None
    parent = summarize_transfer_event_names(
        ["[CUDA memcpy DtoH]"] * 13 + ["cudaMemcpy HtoD"] * 4
    )
    candidate = summarize_transfer_event_names(
        ["[CUDA memcpy DtoH]", "cudaMemcpy HtoD"]
    )

    assert validate_transfer_reduction(
        parent,
        candidate,
        minimum_device_to_host=12,
        minimum_host_to_device=3,
    ) == {"device_to_host": 12, "host_to_device": 3}
    with pytest.raises(VerificationContractError, match="D2H reduction"):
        validate_transfer_reduction(
            parent,
            candidate,
            minimum_device_to_host=13,
            minimum_host_to_device=3,
        )


def test_nvidia_smi_snapshot_records_clock_power_and_rejects_foreign_compute(
    monkeypatch: pytest.MonkeyPatch,
):
    responses = iter(
        [
            "2026/08/02, 0, GPU-abc, Test GPU, 999.1, P0, 100, 200, 1500, 900\n",
            f"{verify_tool.os.getpid()}, python, 123, GPU-abc\n",
        ]
    )

    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout=next(responses), stderr="")

    monkeypatch.setattr(verify_tool.subprocess, "run", fake_run)
    snapshot = verify_tool._nvidia_smi_snapshot(0)

    assert snapshot["foreign_compute_processes"] == 0
    assert "power.draw" in snapshot["gpu_fields"]
    assert "clocks.current.graphics" in snapshot["gpu_fields"]

    responses = iter(
        [
            "2026/08/02, 0, GPU-abc, Test GPU, 999.1, P0, 100, 200, 1500, 900\n",
            "999999, other-process, 123, GPU-abc\n",
        ]
    )
    with pytest.raises(VerificationContractError, match="concurrent GPU"):
        verify_tool._nvidia_smi_snapshot(0)


def test_default_cli_path_is_disarmed_and_does_not_create_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    output_dir = tmp_path / "must-not-exist"

    def forbidden_cuda_query(*args, **kwargs):
        raise AssertionError("disarmed CLI queried CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden_cuda_query)
    monkeypatch.setattr(
        verify_tool,
        "prepare_verification",
        lambda args: SimpleNamespace(plan={"status": "DRY_RUN"}),
    )
    monkeypatch.setattr(
        verify_tool,
        "execute_verification",
        lambda prepared: pytest.fail("default CLI executed the runtime protocol"),
    )

    result = verify_tool.main(
        [
            "--checkpoint",
            str(tmp_path / "checkpoint.pth"),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result == 0
    assert '"status": "DRY_RUN"' in capsys.readouterr().out
    assert not output_dir.exists()


def test_cpu_preparation_loads_one_native_checkpoint_without_cuda_or_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    parent_path = tmp_path / "parent.py"
    candidate_path = tmp_path / "candidate.py"
    checkpoint_path = tmp_path / "native.pth"
    output_dir = tmp_path / "runtime-output"
    parent_path.write_text("# parent fixture\n", encoding="utf-8")
    candidate_path.write_text("# candidate fixture\n", encoding="utf-8")
    torch.save({"state_dict": {"head.decoder.weight": torch.ones(1)}}, checkpoint_path)
    parent, candidate = _paired_configs()

    def fake_fromfile(path, import_custom_modules=False):
        assert import_custom_modules is False
        mapping = parent if Path(path) == parent_path else candidate
        return Config(copy.deepcopy(mapping))

    def forbidden_cuda_query(*args, **kwargs):
        raise AssertionError("CPU preparation queried CUDA")

    monkeypatch.setattr(verify_tool.Config, "fromfile", staticmethod(fake_fromfile))
    monkeypatch.setattr(torch.cuda, "is_available", forbidden_cuda_query)
    args = verify_tool.build_parser().parse_args(
        [
            "--parent-config",
            str(parent_path),
            "--candidate-config",
            str(candidate_path),
            "--checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    prepared = verify_tool.prepare_verification(args)

    assert prepared.plan["status"] == "DRY_RUN"
    assert prepared.plan["cuda_queried"] is False
    assert prepared.plan["checkpoint"]["schema"]["keys"] == 1
    assert prepared.plan["parameters"]["warmup_iterations_per_arm"] == 60
    assert prepared.plan["parameters"]["measured_iterations_per_arm"] == 300
    assert prepared.plan["schedule"]["physical_round_order"] == list(
        PHYSICAL_ROUND_ORDER
    )
    assert prepared.plan["schedule"]["warmup"]["total_invocations"] == 120
    assert prepared.plan["schedule"]["measured"]["total_invocations"] == 600
    assert prepared.plan["preregistration"]["contract"][
        "execution_authorized"
    ] is False
    assert not output_dir.exists()

    torch.save(
        {"state_dict": {"head.canonical_sixth_block.rosa.weight": torch.ones(1)}},
        checkpoint_path,
    )
    with pytest.raises(VerificationContractError, match="without ROSA state"):
        verify_tool.prepare_verification(args)
    assert not output_dir.exists()


def test_execute_requires_exact_idle_gpu_confirmation(monkeypatch, tmp_path):
    monkeypatch.setattr(
        verify_tool,
        "prepare_verification",
        lambda args: pytest.fail("invalid confirmation reached preparation"),
    )
    monkeypatch.setattr(
        verify_tool,
        "execute_verification",
        lambda prepared: pytest.fail("invalid confirmation reached execution"),
    )

    with pytest.raises(SystemExit) as error:
        verify_tool.main(
            [
                "--checkpoint",
                str(tmp_path / "checkpoint.pth"),
                "--output-dir",
                str(tmp_path / "out"),
                "--execute",
                "--idle-gpu-confirmation",
                "not-confirmed",
            ]
        )

    assert error.value.code == 2
