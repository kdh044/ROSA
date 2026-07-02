#!/usr/bin/env python3
"""GT 기준 sparse multi-view ray matching 기초 검증.

학습은 하지 않고, 학습된 checkpoint에서 multi-view image feature를 추출해
object별 range reliability curve를 저장한다.
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from copy import deepcopy
from pathlib import Path

import torch
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_state_dict
from mmdet3d.registry import MODELS

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmdet3d_plugin.models.ray_matching import (  # noqa: E402
    compute_gt_ray_keypoint_matching,
    compute_gt_ray_matching,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="GT 기준 sparse multi-view ray range matching curve 검증"
    )
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument(
        "--out-dir",
        default="work_dirs/_ray_range_gt_check",
        help="Directory for csv/pkl/plot outputs",
    )
    parser.add_argument("--max-frames", type=int, default=5)
    parser.add_argument("--max-objects", type=int, default=20)
    parser.add_argument("--range-radius", type=float, default=2.0)
    parser.add_argument("--num-candidates", type=int, default=21)
    parser.add_argument("--feature-level", type=int, default=0)
    parser.add_argument(
        "--sampling-mode",
        choices=["center", "sparse-keypoints"],
        default="center",
        help="center: 3D center one-point check, sparse-keypoints: fixed Sparse4D box keypoints",
    )
    parser.add_argument("--min-keypoints-per-view", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    return parser.parse_args()


def _disable_deformable_cuda_format(cfg):
    cfg.model.use_deformable_func = False
    if "head" in cfg.model and "deformable_model" in cfg.model.head:
        cfg.model.head.deformable_model.use_deformable_func = False
    return cfg


def _get_fixed_keypoint_scales(cfg, device, dtype):
    deformable_cfg = cfg.model.head.deformable_model
    kps_cfg = deformable_cfg.get("kps_generator", {})
    fix_scale = kps_cfg.get("fix_scale", [[0.0, 0.0, 0.0]])
    return torch.as_tensor(fix_scale, device=device, dtype=dtype)


def _decoded_box_centers(boxes: torch.Tensor) -> torch.Tensor:
    centers = boxes[:, :3].clone()
    centers[:, 2] = centers[:, 2] + boxes[:, 5] * 0.5
    return centers


def _build_gt_val_loader(cfg, num_workers: int):
    loader_cfg = deepcopy(cfg.val_dataloader)
    loader_cfg["batch_size"] = 1
    loader_cfg["num_workers"] = int(num_workers)
    loader_cfg["persistent_workers"] = False
    dataset = loader_cfg["dataset"]
    dataset["pipeline"] = deepcopy(cfg.train_pipeline)
    dataset["test_mode"] = False
    dataset["filter_empty_gt"] = False
    if hasattr(cfg, "ann_val"):
        dataset["ann_file"] = cfg.ann_val
    return Runner.build_dataloader(loader_cfg)


def _tensor_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: _tensor_to_cpu(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_tensor_to_cpu(v) for v in x)
    return x


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_object_curve(path: Path, rows: list[dict], title: str):
    import matplotlib.pyplot as plt

    x = [float(r["range_offset"]) for r in rows]
    y_rel = [float(r["reliability"]) for r in rows]
    y_match = [float(r["matching_confidence"]) for r in rows]
    y_q = [float(r["projection_quality"]) for r in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(x, y_rel, marker="o", label="reliability")
    plt.plot(x, y_match, marker=".", label="matching")
    plt.plot(x, y_q, marker=".", label="projection_quality")
    plt.axvline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("range offset from GT (m)")
    plt.ylabel("score")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    args = parse_args()
    if args.num_candidates < 3 or args.num_candidates % 2 == 0:
        raise ValueError("--num-candidates must be an odd integer >= 3")

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    _disable_deformable_cuda_format(cfg)
    init_default_scope(cfg.get("default_scope", "mmdet3d"))

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[GT 기준 검증] CUDA를 사용할 수 없어 CPU로 실행합니다")
        args.device = "cpu"
    device = torch.device(args.device)

    model = MODELS.build(cfg.model)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    load_state_dict(model, state_dict, strict=False)
    model.to(device)
    model.eval()

    loader = _build_gt_val_loader(cfg, args.num_workers)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    offsets = torch.linspace(
        -args.range_radius,
        args.range_radius,
        args.num_candidates,
        device=device,
    )

    all_rows = []
    summary_rows = []
    dumped = []
    frames_done = 0

    for frame_idx, batch in enumerate(loader):
        if frames_done >= args.max_frames:
            break
        with torch.no_grad():
            data = model.data_preprocessor(batch, training=False)
            inputs = data["inputs"]
            data_samples = data["data_samples"]
            metas = model._build_metas(inputs, data_samples)
            feature_maps = model.extract_feat(metas["img"], metas=metas)

            gt = data_samples[0].gt_instances_3d
            if "bboxes_3d" not in gt or gt.bboxes_3d.tensor.numel() == 0:
                continue
            boxes = gt.bboxes_3d.tensor.to(device)
            labels = gt.labels_3d.to(device) if "labels_3d" in gt else None
            centers = _decoded_box_centers(boxes)
            if centers.shape[0] > args.max_objects:
                centers = centers[: args.max_objects]
                boxes = boxes[: args.max_objects]
                if labels is not None:
                    labels = labels[: args.max_objects]
            centers = centers[None]

            if args.sampling_mode == "sparse-keypoints":
                fixed_scales = _get_fixed_keypoint_scales(
                    cfg, device=boxes.device, dtype=boxes.dtype
                )
                result = compute_gt_ray_keypoint_matching(
                    feature_maps=list(feature_maps),
                    boxes_lidar=boxes[None],
                    metas=metas,
                    offsets=offsets,
                    fixed_scales=fixed_scales,
                    feature_level=args.feature_level,
                    projection_type=cfg.model.head.deformable_model.projection_type,
                    min_keypoints_per_view=args.min_keypoints_per_view,
                )
            else:
                result = compute_gt_ray_matching(
                    feature_maps=list(feature_maps),
                    centers_lidar=centers,
                    metas=metas,
                    offsets=offsets,
                    feature_level=args.feature_level,
                    projection_type=cfg.model.head.deformable_model.projection_type,
                )

        token = data_samples[0].metainfo.get("token", str(frame_idx))
        cam_names = data_samples[0].metainfo.get("cam_names", [])
        reliability = result.reliability[0].detach().cpu()
        matching = result.matching_confidence[0].detach().cpu()
        proj_q = result.projection_quality[0].detach().cpu()
        cand_ranges = result.candidate_ranges[0].detach().cpu()
        valid = result.valid_mask[0].detach().cpu()
        ref_cam = result.reference_camera[0].detach().cpu()
        visible_count = result.visible_count[0].detach().cpu()
        labels_cpu = labels.detach().cpu() if labels is not None else None

        for obj_idx in range(reliability.shape[0]):
            if int(visible_count[obj_idx].item()) < 2:
                continue
            obj_rows = []
            rel_obj = reliability[obj_idx]
            finite = torch.isfinite(rel_obj)
            if not finite.any():
                continue
            best_idx = int(torch.nan_to_num(rel_obj, nan=-1e9).argmax().item())
            ref = int(ref_cam[obj_idx].item())
            ref_name = cam_names[ref] if ref < len(cam_names) else str(ref)
            label = int(labels_cpu[obj_idx].item()) if labels_cpu is not None else -1
            gt_range = float(cand_ranges[obj_idx, args.num_candidates // 2].item())
            for k in range(args.num_candidates):
                row = dict(
                    frame_index=frame_idx,
                    token=token,
                    object_index=obj_idx,
                    label=label,
                    sampling_mode=args.sampling_mode,
                    reference_camera=ref,
                    reference_camera_name=ref_name,
                    center_visible_views=int(visible_count[obj_idx].item()),
                    candidate_index=k,
                    range_m=float(cand_ranges[obj_idx, k].item()),
                    range_offset=float(offsets.detach().cpu()[k].item()),
                    candidate_valid_views=int(valid[:, obj_idx, k].sum().item()),
                    matching_confidence=float(matching[obj_idx, k].item()),
                    projection_quality=float(proj_q[obj_idx, k].item()),
                    reliability=float(reliability[obj_idx, k].item()),
                )
                all_rows.append(row)
                obj_rows.append(row)
            summary_rows.append(
                dict(
                    frame_index=frame_idx,
                    token=token,
                    object_index=obj_idx,
                    label=label,
                    sampling_mode=args.sampling_mode,
                    reference_camera=ref,
                    reference_camera_name=ref_name,
                    center_visible_views=int(visible_count[obj_idx].item()),
                    gt_range_m=gt_range,
                    best_index=best_idx,
                    best_range_m=float(cand_ranges[obj_idx, best_idx].item()),
                    best_offset_m=float(offsets.detach().cpu()[best_idx].item()),
                    best_reliability=float(reliability[obj_idx, best_idx].item()),
                    gt_reliability=float(
                        reliability[obj_idx, args.num_candidates // 2].item()
                    ),
                )
            )
            if args.plot:
                plot_name = f"frame{frame_idx:04d}_obj{obj_idx:03d}.png"
                _plot_object_curve(
                    out_dir / "plots" / plot_name,
                    obj_rows,
                    f"{token} obj={obj_idx} ref={ref_name}",
                )

        dumped.append(
            dict(
                frame_index=frame_idx,
                token=token,
                sampling_mode=args.sampling_mode,
                boxes=_tensor_to_cpu(boxes),
                labels=_tensor_to_cpu(labels) if labels is not None else None,
                result=_tensor_to_cpu(result.__dict__),
            )
        )
        frames_done += 1
        print(
            f"[GT 기준 검증] mode={args.sampling_mode} "
            f"frame={frame_idx} token={token} "
            f"rows={len(all_rows)} summaries={len(summary_rows)}"
        )

    _write_csv(out_dir / "curves.csv", all_rows)
    _write_csv(out_dir / "summary.csv", summary_rows)
    with (out_dir / "curves.pkl").open("wb") as f:
        pickle.dump(dumped, f)

    print(f"[GT 기준 검증] wrote {len(all_rows)} curve rows to {out_dir}")
    if not all_rows:
        print("[GT 기준 검증] 유효한 multi-view object를 찾지 못했습니다")
    elif summary_rows:
        offsets_cpu = torch.tensor([r["best_offset_m"] for r in summary_rows]).abs()
        near_04 = float((offsets_cpu <= 0.4).float().mean())
        near_06 = float((offsets_cpu <= 0.6).float().mean())
        edge = float((offsets_cpu >= args.range_radius - 1e-6).float().mean())
        print(
            "[GT 기준 검증] "
            f"objects={len(summary_rows)} "
            f"정답근처(|offset|<=0.4m)={near_04:.3f} "
            f"정답근처(|offset|<=0.6m)={near_06:.3f} "
            f"경계후보비율={edge:.3f} "
            f"median_abs_offset={float(offsets_cpu.median()):.3f}m"
        )


if __name__ == "__main__":
    main()
