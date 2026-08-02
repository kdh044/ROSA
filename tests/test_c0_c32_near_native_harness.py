"""CPU-only contracts for the exact C0-vs-C32 runtime protocol."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

from mmengine.config import Config
import pytest
import torch

from rosa.runtime_verification import (
    C0_C32_FULL_DETECTOR_SCOPE,
    CANDIDATE_ARM,
    DEFAULT_MEASURED_ITERATIONS_PER_ARM,
    PARENT_ARM,
    VerificationContractError,
    build_alternating_schedule,
    evaluate_c0_c32_near_native_gate,
    file_sha256,
    finite_output_schema,
    paired_bootstrap_confidence_intervals,
    validate_c0_c32_near_native_config_pair,
)

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "rosa_verify_c0_c32_near_native_tool",
    ROOT / "tools" / "verify_c0_c32_near_native.py",
)
assert _SPEC is not None and _SPEC.loader is not None
verify_tool = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = verify_tool
_SPEC.loader.exec_module(verify_tool)


def _detector_host_arguments() -> list[str]:
    root = verify_tool.DEFAULT_DETECTOR_HOST_ROOT.resolve()
    revision = verify_tool._run_git_at(root, "rev-parse", "HEAD")
    return [
        "--detector-host-root",
        str(root),
        "--expected-detector-host-revision",
        revision,
    ]


def _clear_preimported_projects(monkeypatch) -> None:
    for name in tuple(sys.modules):
        if name == "projects" or name.startswith("projects."):
            monkeypatch.delitem(sys.modules, name)


def _configs() -> tuple[Config, Config]:
    return (
        Config.fromfile(
            ROOT / "configs/rosa_r18_main126_v2_c0_cumulative_runtime_exact.py",
            import_custom_modules=False,
        ),
        Config.fromfile(
            ROOT
            / (
                "configs/rosa_r18_main126_v2_c32_cbgs_native_official_gnn_"
                "residual_cpu_metadata_lean_loader_rgb_uint8_transfer_pinned_"
                "inplace_homogeneous4x4_inference_reuse_exact.py"
            ),
            import_custom_modules=False,
        ),
    )


def _bootstrap(median_upper: float, p95_ratio: float) -> dict:
    return {
        "metrics": {
            "median_candidate_over_parent": {
                "point_estimate": median_upper - 0.01,
                "ci95": {"lower": median_upper - 0.02, "upper": median_upper},
            },
            "p95_candidate_over_parent": {
                "point_estimate": p95_ratio,
                "ci95": {"lower": p95_ratio - 0.01, "upper": p95_ratio + 0.01},
            },
        }
    }


def _memory(parent: int, candidate: int) -> dict:
    return {
        PARENT_ARM: {
            "peak_allocated_bytes": parent,
            "peak_reserved_bytes": parent,
        },
        CANDIDATE_ARM: {
            "peak_allocated_bytes": candidate,
            "peak_reserved_bytes": candidate,
        },
    }


def test_real_configs_differ_only_by_exact_head_type() -> None:
    parent, candidate = _configs()

    contract = validate_c0_c32_near_native_config_pair(
        parent, candidate, expected_batch_size=4
    )

    assert contract["parent_head_type"] == "ROSANativeBaselineHead"
    assert contract["candidate_head_type"] == ("ROSAOfficialResidualNativeBaselineHead")
    assert contract["backbone"] == "mmdet.ResNet-18"
    assert contract["gnn_stages"] == 5
    assert contract["input_dtype"] == "torch.uint8"
    assert contract["packer"] == "PackFisheye3DODRGBUint8Inputs"
    assert contract["deformable_model"].startswith(
        "ROSAInferenceReuseExactHomogeneous4x4"
    )
    assert contract["only_executable_delta"] == "model.head.type"


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda config: config.model.head.update(type="ROSANativeBaselineHead"),
            "candidate head must be",
        ),
        (
            lambda config: config.model.img_backbone.update(depth=50),
            "ResNet depth-18",
        ),
        (
            lambda config: config.test_dataloader.sampler.update(shuffle=True),
            "shuffle=False",
        ),
        (
            lambda config: config.model.data_preprocessor.update(mean=[0, 0, 0]),
            "executable configs differ",
        ),
    ],
)
def test_config_contract_fails_on_head_graph_data_or_preprocessor_drift(
    mutation, match
) -> None:
    parent, candidate = _configs()
    candidate = Config(copy.deepcopy(candidate.to_dict()))
    mutation(candidate)

    with pytest.raises(VerificationContractError, match=match):
        validate_c0_c32_near_native_config_pair(
            parent, candidate, expected_batch_size=4
        )


def test_near_native_gate_requires_latency_and_both_memory_limits() -> None:
    passed = evaluate_c0_c32_near_native_gate(
        _bootstrap(1.10, 1.10), _memory(1_000, 1_020)
    )
    assert passed["passed"] is True
    assert passed["speed_claim_eligible"] is True
    assert passed["benchmark_scope"] == C0_C32_FULL_DETECTOR_SCOPE

    latency_failure = evaluate_c0_c32_near_native_gate(
        _bootstrap(1.100001, 1.10), _memory(1_000, 1_000)
    )
    memory_failure = evaluate_c0_c32_near_native_gate(
        _bootstrap(1.09, 1.09), _memory(1_000, 1_021)
    )
    assert latency_failure["passed"] is False
    assert memory_failure["passed"] is False


def test_finite_output_schema_allows_value_delta_but_rejects_nonfinite() -> None:
    first = finite_output_schema([{"scores": torch.tensor([0.1, 0.9])}])
    second = finite_output_schema([{"scores": torch.tensor([0.2, 0.8])}])

    assert first == second
    with pytest.raises(VerificationContractError, match="NaN or Inf"):
        finite_output_schema([{"scores": torch.tensor([float("nan")])}])


class _FakeBoxes:
    def __init__(self, value: torch.Tensor) -> None:
        self.tensor = value


class _FakePredictions:
    def __init__(self, boxes, scores, labels) -> None:
        self.bboxes_3d = _FakeBoxes(boxes)
        self.scores_3d = scores
        self.labels_3d = labels


class _FakeSample:
    def __init__(self, predictions: _FakePredictions) -> None:
        self.pred_instances_3d = predictions

    def to_dict(self):
        return {
            "pred_instances_3d": {
                "bboxes_3d": self.pred_instances_3d.bboxes_3d.tensor,
                "scores_3d": self.pred_instances_3d.scores_3d,
                "labels_3d": self.pred_instances_3d.labels_3d,
            }
        }


def _fake_output(*, score: float = 0.5, label: int = 0):
    return [
        _FakeSample(
            _FakePredictions(
                torch.zeros(1, 10),
                torch.tensor([score]),
                torch.tensor([label], dtype=torch.long),
            )
        )
        for _ in range(4)
    ]


def test_detection_output_contract_rejects_nan_bad_label_and_count() -> None:
    valid = verify_tool._validate_detection_output(_fake_output())
    assert valid["sample_count"] == 4
    assert valid["detections_per_sample"] == [1, 1, 1, 1]

    with pytest.raises(VerificationContractError, match="NaN or Inf"):
        verify_tool._validate_detection_output(_fake_output(score=float("nan")))
    with pytest.raises(VerificationContractError, match="outside six classes"):
        verify_tool._validate_detection_output(_fake_output(label=6))
    with pytest.raises(VerificationContractError, match="expected 4"):
        verify_tool._validate_detection_output(_fake_output()[:3])


def test_dry_run_never_queries_cuda_or_creates_output(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "tiny.pth"
    torch.save({"state_dict": {"weight": torch.ones(1)}}, checkpoint)
    output = tmp_path / "never-created"
    revision = verify_tool._run_git("rev-parse", "HEAD")
    _clear_preimported_projects(monkeypatch)

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("dry-run queried CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "current_stream", forbidden)
    monkeypatch.setattr(torch.cuda, "device_count", forbidden)
    args = verify_tool.build_parser().parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--expected-checkpoint-sha256",
            file_sha256(checkpoint),
            "--expected-source-revision",
            revision,
            "--output-dir",
            str(output),
        ]
        + _detector_host_arguments()
    )

    prepared = verify_tool.prepare_protocol(args)

    assert prepared.plan["cuda_queried"] is False
    assert prepared.plan["output_created"] is False
    assert prepared.plan["parameters"]["warmup_iterations_per_arm"] == 60
    assert prepared.plan["parameters"]["measured_iterations_per_arm"] == 300
    assert prepared.plan["detector_host"]["git"]["revision"] == (
        args.expected_detector_host_revision
    )
    assert prepared.plan["detector_host"]["source_contract"]["files"]
    assert output.exists() is False


def test_dry_run_rejects_foreign_preimported_detector_host(
    tmp_path, monkeypatch
) -> None:
    checkpoint = tmp_path / "tiny.pth"
    torch.save({"state_dict": {"weight": torch.ones(1)}}, checkpoint)
    revision = verify_tool._run_git("rev-parse", "HEAD")
    _clear_preimported_projects(monkeypatch)
    foreign = ModuleType("projects.mmdet3d_plugin")
    foreign.__file__ = str(tmp_path / "foreign" / "__init__.py")
    monkeypatch.setitem(sys.modules, foreign.__name__, foreign)
    args = verify_tool.build_parser().parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--expected-checkpoint-sha256",
            file_sha256(checkpoint),
            "--expected-source-revision",
            revision,
            "--output-dir",
            str(tmp_path / "never-created"),
        ]
        + _detector_host_arguments()
    )

    with pytest.raises(VerificationContractError, match="foreign detector-host"):
        verify_tool.prepare_protocol(args)


def test_dry_run_rejects_wrong_checkpoint_before_any_cuda_query(
    tmp_path, monkeypatch
) -> None:
    checkpoint = tmp_path / "tiny.pth"
    torch.save({"state_dict": {"weight": torch.ones(1)}}, checkpoint)
    revision = verify_tool._run_git("rev-parse", "HEAD")
    _clear_preimported_projects(monkeypatch)
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: (_ for _ in ()).throw(AssertionError("CUDA queried")),
    )
    args = verify_tool.build_parser().parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--expected-checkpoint-sha256",
            "0" * 64,
            "--expected-source-revision",
            revision,
            "--output-dir",
            str(tmp_path / "never-created"),
        ]
        + _detector_host_arguments()
    )

    with pytest.raises(VerificationContractError, match="checkpoint hash"):
        verify_tool.prepare_protocol(args)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--execute", "--execution-state", "DISARMED"],
        ["--execute", "--execution-state", "ARMED"],
    ],
)
def test_execute_requires_both_exact_arming_controls(monkeypatch, arguments) -> None:
    monkeypatch.setattr(
        verify_tool,
        "prepare_protocol",
        lambda args: (_ for _ in ()).throw(AssertionError("prepare must not run")),
    )
    base = [
        "--checkpoint",
        "/does/not/matter",
        "--expected-checkpoint-sha256",
        "0" * 64,
        "--expected-source-revision",
        "0" * 40,
        "--output-dir",
        "/does/not/matter",
    ] + _detector_host_arguments()

    with pytest.raises(SystemExit) as caught:
        verify_tool.main(base + arguments)
    assert caught.value.code == 2


def test_module_schema_normalization_allows_only_exact_head_path() -> None:
    parent = [
        {"path": "", "type": "detector.Type"},
        {
            "path": "head",
            "type": "rosa.integrations.sparse4d.ROSANativeBaselineHead",
        },
        {"path": "head.layers.0", "type": "torch.nn.Linear"},
    ]
    candidate = copy.deepcopy(parent)
    candidate[1][
        "type"
    ] = "rosa.integrations.sparse4d.ROSAOfficialResidualNativeBaselineHead"
    assert verify_tool._normalize_module_schema(parent) == (
        verify_tool._normalize_module_schema(candidate)
    )
    candidate[2]["type"] = "torch.nn.Identity"
    assert verify_tool._normalize_module_schema(parent) != (
        verify_tool._normalize_module_schema(candidate)
    )


def test_paired_timing_rejects_semantic_batch_swap() -> None:
    schedule = build_alternating_schedule(
        iterations_per_arm=DEFAULT_MEASURED_ITERATIONS_PER_ARM,
        batch_count=2,
        phase="measured",
    )
    records = []
    for entry in schedule:
        record = entry.to_dict()
        record.update(
            {
                "elapsed_ns": 1_000_000,
                "sample_count": 4,
                "benchmark_scope": C0_C32_FULL_DETECTOR_SCOPE,
                "semantic_batch_sha256": f"semantic-{entry.batch_index}",
                "source_batch_sha256": f"source-{entry.batch_index}",
            }
        )
        records.append(record)
    candidate = next(
        record
        for record in records
        if record["pair_index"] == 0 and record["arm"] == CANDIDATE_ARM
    )
    candidate["semantic_batch_sha256"] = "swapped"

    with pytest.raises(VerificationContractError, match="does not share"):
        paired_bootstrap_confidence_intervals(
            records,
            expected_pairs=DEFAULT_MEASURED_ITERATIONS_PER_ARM,
            required_pair_fields=(
                "benchmark_scope",
                "semantic_batch_sha256",
                "source_batch_sha256",
            ),
        )
