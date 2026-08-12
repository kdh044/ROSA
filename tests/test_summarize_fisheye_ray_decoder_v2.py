import importlib.util
import json
from pathlib import Path

import pytest

_TOOL = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "summarize_fisheye_ray_decoder_v2.py"
)
_SPEC = importlib.util.spec_from_file_location("summarize_fisheye_ray_v2", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
build_summary = _MODULE.build_summary
write_json_atomic = _MODULE.write_json_atomic


def _validation_line(epoch, map_value, nds_value):
    return (
        f"Epoch(val) [{epoch}][489/489] "
        f"fisheye_bev/mAP: {map_value:.4f} "
        "fisheye_bev/mATE: 0.4000 fisheye_bev/mASE: 0.1000 "
        "fisheye_bev/mAOE: 0.3000 "
        f"fisheye_bev/NDS: {nds_value:.4f} "
        "fisheye_bev/Car_AP: 0.6000\n"
    )


def test_completed_run_summary_is_hashed_and_compared(tmp_path):
    work_dir = tmp_path / "experiment"
    log_dir = work_dir / "timestamp"
    log_dir.mkdir(parents=True)
    (log_dir / "run.log").write_text(
        _validation_line(4, 0.40, 0.50) + _validation_line(20, 0.60, 0.70),
        encoding="utf-8",
    )
    (work_dir / "epoch_20.pth").write_bytes(b"epoch")
    (work_dir / "best_fisheye_bev_mAP_epoch_20.pth").write_bytes(b"map")
    (work_dir / "best_fisheye_bev_NDS_epoch_20.pth").write_bytes(b"best")
    snapshot = work_dir / "execution_snapshot_v1" / "rosa_ray_decoder_v2_source.tar.gz"
    snapshot.parent.mkdir()
    snapshot.write_bytes(b"source")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "arm": "C0",
                "metrics": {
                    "mAP": 0.55,
                    "NDS": 0.65,
                    "mATE": 0.45,
                    "mASE": 0.09,
                    "mAOE": 0.35,
                    "class_AP": {"Car": 0.58},
                },
            }
        ),
        encoding="utf-8",
    )

    summary = build_summary(work_dir, baseline)
    output = tmp_path / "summary.json"
    write_json_atomic(output, summary)

    assert summary["status"] == "complete"
    assert tuple(summary["validation_history"]) == ("4", "20")
    assert summary["delta_vs_c0"]["mAP"] == pytest.approx(0.05)
    assert summary["delta_vs_c0"]["NDS"] == pytest.approx(0.05)
    assert summary["class_AP_delta_vs_c0"]["Car"] == pytest.approx(0.02)
    assert len(summary["checkpoints"]) == 3
    assert len(summary["execution_snapshot"]["sha256"]) == 64
    assert json.loads(output.read_text(encoding="utf-8")) == summary
