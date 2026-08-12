"""Summarize a completed V2 run and compare it with the sealed C0 result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

EPOCH_PATTERN = re.compile(r"Epoch\(val\)\s+\[(\d+)\]\[\s*\d+/\d+\]")
METRIC_PATTERN = re.compile(r"fisheye_bev/([^:\s]+):\s*(-?\d+(?:\.\d+)?)")
GLOBAL_METRICS = ("mAP", "NDS", "mATE", "mASE", "mAOE")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_record(path: Path) -> dict:
    return {
        "file": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def parse_validation_history(log_paths: list[Path]) -> dict[int, dict[str, float]]:
    history: dict[int, dict[str, float]] = {}
    for log_path in sorted(
        log_paths, key=lambda path: (path.stat().st_mtime, str(path))
    ):
        with log_path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                epoch_match = EPOCH_PATTERN.search(line)
                if epoch_match is None:
                    continue
                metrics = {
                    name: float(value) for name, value in METRIC_PATTERN.findall(line)
                }
                if all(name in metrics for name in GLOBAL_METRICS):
                    history[int(epoch_match.group(1))] = metrics
    return history


def build_summary(
    work_dir: Path,
    baseline_reference: Path,
    *,
    final_epoch: int = 20,
) -> dict:
    work_dir = work_dir.resolve()
    log_paths = list(work_dir.rglob("*.log"))
    if not log_paths:
        raise FileNotFoundError(f"no MMEngine logs found below {work_dir}")
    history = parse_validation_history(log_paths)
    if final_epoch not in history:
        raise RuntimeError(f"official epoch-{final_epoch} validation is missing")

    with baseline_reference.open("r", encoding="utf-8") as stream:
        baseline = json.load(stream)
    final_metrics = history[final_epoch]
    baseline_metrics = baseline["metrics"]
    deltas = {
        name: final_metrics[name] - float(baseline_metrics[name])
        for name in GLOBAL_METRICS
    }
    baseline_class_ap = baseline_metrics.get("class_AP", {})
    class_ap = {
        name[: -len("_AP")]: value
        for name, value in final_metrics.items()
        if name.endswith("_AP") and "@" not in name
    }
    class_ap_delta = {
        name: value - float(baseline_class_ap[name])
        for name, value in class_ap.items()
        if name in baseline_class_ap
    }

    best_map = sorted(work_dir.glob("best_fisheye_bev_mAP_*.pth"))
    best_nds = sorted(work_dir.glob("best_fisheye_bev_NDS_*.pth"))
    if not best_map or not best_nds:
        raise FileNotFoundError("best-mAP and best-NDS checkpoints are required")
    checkpoint_paths = [work_dir / f"epoch_{final_epoch}.pth", *best_map, *best_nds]
    missing_checkpoints = [str(path) for path in checkpoint_paths if not path.is_file()]
    if missing_checkpoints:
        raise FileNotFoundError(
            f"missing completion checkpoints: {missing_checkpoints}"
        )
    snapshot = work_dir / "execution_snapshot_v1" / "rosa_ray_decoder_v2_source.tar.gz"
    if not snapshot.is_file():
        raise FileNotFoundError(f"missing execution source snapshot: {snapshot}")

    return {
        "status": "complete",
        "arm": "Fisheye-Ray-Decoder-V2-raw-uniform-20e",
        "work_dir": str(work_dir),
        "final_epoch": final_epoch,
        "validation_history": {str(epoch): history[epoch] for epoch in sorted(history)},
        "final_metrics": final_metrics,
        "class_AP": class_ap,
        "baseline": {
            "arm": baseline["arm"],
            "reference": str(baseline_reference.resolve()),
            "metrics": baseline_metrics,
        },
        "delta_vs_c0": deltas,
        "class_AP_delta_vs_c0": class_ap_delta,
        "checkpoints": [artifact_record(path) for path in checkpoint_paths],
        "execution_snapshot": artifact_record(snapshot),
        "logs": [str(path.resolve()) for path in sorted(log_paths)],
    }


def write_json_atomic(output: Path, payload: dict) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--baseline-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-epoch", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(
        args.work_dir,
        args.baseline_reference,
        final_epoch=args.final_epoch,
    )
    write_json_atomic(args.output, summary)
    print(json.dumps(summary["delta_vs_c0"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
