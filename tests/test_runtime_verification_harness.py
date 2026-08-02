from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

from mmengine.config import Config
import numpy as np
import pytest
import torch

from rosa.runtime_verification import (
    CANDIDATE_ARM,
    CPPC,
    PARENT_ARM,
    PCCP,
    VerificationContractError,
    build_alternating_schedule,
    build_batch_manifest,
    classify_transfer_event,
    percentile,
    raw_fingerprint,
    require_same_fingerprint,
    summarize_timings,
    summarize_transfer_event_names,
    validate_config_pair,
    validate_expected_manifest,
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
        rounds=2,
        batch_count=2,
        phase="measured",
    )

    assert len(entries) == 16
    assert [entry.pattern for entry in entries[:8]] == [PCCP] * 8
    assert [entry.pattern for entry in entries[8:]] == [CPPC] * 8
    assert [entry.arm for entry in entries[:4]] == [
        PARENT_ARM,
        CANDIDATE_ARM,
        CANDIDATE_ARM,
        PARENT_ARM,
    ]
    assert [entry.arm for entry in entries[8:12]] == [
        CANDIDATE_ARM,
        PARENT_ARM,
        PARENT_ARM,
        CANDIDATE_ARM,
    ]
    assert sum(entry.arm == PARENT_ARM for entry in entries) == 8
    assert sum(entry.arm == CANDIDATE_ARM for entry in entries) == 8


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"warmup_rounds": 1}, "warmup_rounds must be even"),
        ({"measured_rounds": 3}, "measured_rounds must be even"),
        ({"capture_batches": 0}, "capture_batches must be >= 1"),
        ({"profile_batch_index": 2}, "select a captured batch"),
    ],
)
def test_runtime_parameters_fail_closed(overrides, match):
    values = {
        "capture_batches": 2,
        "warmup_rounds": 2,
        "measured_rounds": 10,
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
