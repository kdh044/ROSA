"""Run MMEngine Sparse4D test and dump predictions for KB visualization."""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--dump-path", required=True)
    parser.add_argument("--vis-pred-path", required=True)
    parser.add_argument("--work-dir")
    parser.add_argument("--score-thr", type=float, default=0.0)
    parser.add_argument("--max-eval-dist", type=float, default=20.0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    return parser.parse_args()


def _set_workers(dataloader_cfg, num_workers: int):
    dataloader_cfg["num_workers"] = int(num_workers)
    dataloader_cfg["persistent_workers"] = False
    dataloader_cfg["batch_size"] = 1
    return dataloader_cfg


def _np(x, shape=None, dtype=np.float32):
    if x is None:
        return np.zeros(shape, dtype=dtype)
    arr = np.asarray(x, dtype=dtype)
    if shape is not None and arr.size == 0:
        return np.zeros(shape, dtype=dtype)
    return arr


def convert_dump_to_vis(dump_path: str, vis_pred_path: str):
    with open(dump_path, "rb") as f:
        rows = pickle.load(f)

    vis_rows = []
    for row in rows:
        centers = _np(row.get("pred_centers"), (0, 2), np.float32)
        n = centers.shape[0]
        z = _np(row.get("pred_z"), (n,), np.float32).reshape(n, 1)
        sizes = _np(row.get("pred_sizes"), (n, 3), np.float32)
        yaws = _np(row.get("pred_yaws"), (n,), np.float32).reshape(n, 1)
        velos = _np(row.get("pred_velos"), (n, 2), np.float32)
        if n:
            boxes = np.concatenate([centers, z, sizes, yaws, velos], axis=1)
        else:
            boxes = np.zeros((0, 9), dtype=np.float32)
        vis_rows.append(
            dict(
                token=row.get("token"),
                pred_boxes=boxes.astype(np.float32),
                pred_scores=_np(row.get("pred_scores"), (n,), np.float32),
                pred_labels=_np(row.get("pred_labels"), (n,), np.int64),
            )
        )

    Path(vis_pred_path).parent.mkdir(parents=True, exist_ok=True)
    with open(vis_pred_path, "wb") as f:
        pickle.dump(vis_rows, f)
    return len(vis_rows)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.load_from = args.checkpoint
    cfg.resume = False
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = os.path.join(
            "work_dirs", "_vis_dump_" + Path(args.config).stem)

    cfg.test_dataloader = _set_workers(cfg.test_dataloader, args.num_workers)
    evaluator = cfg.test_evaluator.copy()
    evaluator.update(
        type="FisheyeBev3DDumpMetric",
        dump_path=args.dump_path,
        score_threshold=args.score_thr,
        max_eval_dist=args.max_eval_dist,
    )
    cfg.test_evaluator = evaluator

    Path(args.dump_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.vis_pred_path).parent.mkdir(parents=True, exist_ok=True)

    runner = Runner.from_cfg(cfg)
    runner.test()
    n = convert_dump_to_vis(args.dump_path, args.vis_pred_path)
    print(f"[test_mmengine_dump] wrote {n} visual prediction rows to "
          f"{args.vis_pred_path}")


if __name__ == "__main__":
    main()
