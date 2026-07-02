"""Build kmeans anchors from a RoboSense v2 pkl.

Usage:
    python tools/anchor_generator_robosense.py \
        --ann_file /home/airlab/user/kdh/RoboSense_images/pkl/sphere_stream_robosense_train_dommatched_v.pkl \
        --num_anchor 900 --detection_range 25 \
        --output_file_name ckpt/robosense_kmeans900.npy

Output: (num_anchor, 11) anchor array with the same layout as the original
Sparse4D box3d: [X, Y, Z, log(W), log(L), log(H), SIN_YAW, COS_YAW, VX, VY, VZ].
"""
from __future__ import annotations
import argparse
import os
import pickle

import numpy as np
from sklearn.cluster import KMeans

from projects.mmdet3d_plugin.core.box3d import (
    X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ,
)


def load_robosense_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def gather_gt_boxes(pkl: dict, detection_range: float, use_valid_flag: bool):
    boxes = []
    for sample in pkl["data_list"]:
        for inst in sample.get("instances", []):
            if use_valid_flag and not inst.get("bbox_3d_isvalid", True):
                continue
            b = inst["bbox_3d"]                          # [x, y, z, w, l, h, yaw]
            if len(b) < 7:
                continue
            x, y, z, w, l, h, yaw = b[:7]
            d = (x * x + y * y) ** 0.5                   # BEV range
            if d > detection_range:
                continue
            vx, vy = inst.get("velocity", (0.0, 0.0))[:2]
            boxes.append([x, y, z, w, l, h, yaw, float(vx), float(vy)])
    return np.asarray(boxes, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_file", required=True)
    ap.add_argument("--num_anchor", type=int, default=900)
    ap.add_argument("--detection_range", type=float, default=25.0,
                    help="BEV range filter (radius, meters). Boxes farther "
                         "than this are excluded before clustering.")
    ap.add_argument("--output_file_name", required=True)
    ap.add_argument("--use_valid_flag", action="store_true", default=True)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print(f"[anchor] loading {args.ann_file}")
    pkl = load_robosense_pkl(args.ann_file)
    print(f"[anchor]   classes: {pkl['metainfo'].get('classes')}")
    print(f"[anchor]   {len(pkl['data_list'])} samples")

    gt = gather_gt_boxes(pkl, args.detection_range, args.use_valid_flag)
    print(f"[anchor] gathered {len(gt)} GT boxes within "
          f"{args.detection_range:.1f}m")
    if len(gt) < args.num_anchor:
        raise RuntimeError(
            f"not enough GT boxes ({len(gt)}) for {args.num_anchor} clusters"
        )

    print("[anchor] running kmeans (XYZ only) — please wait")
    km = KMeans(n_clusters=args.num_anchor, verbose=int(args.verbose), n_init=4)
    km.fit(gt[:, :3])
    centers = km.cluster_centers_                                # (K, 3)

    mean_wlh = gt[:, 3:6].mean(axis=0)                            # (3,)
    log_wlh = np.log(np.maximum(mean_wlh, 1e-6))

    anchor = np.zeros((args.num_anchor, 11), dtype=np.float32)
    anchor[:, [X, Y, Z]] = centers
    anchor[:, [W, L, H]] = log_wlh
    anchor[:, SIN_YAW] = 0.0
    anchor[:, COS_YAW] = 1.0
    # VX/VY/VZ start at 0; the refine head learns the velocity head.
    anchor[:, [VX, VY, VZ]] = 0.0

    os.makedirs(os.path.dirname(args.output_file_name) or ".", exist_ok=True)
    np.save(args.output_file_name, anchor)
    print(f"[anchor] wrote {anchor.shape} → {args.output_file_name}")
    print(f"[anchor]   anchor XYZ range:  x∈[{centers[:,0].min():.2f}, "
          f"{centers[:,0].max():.2f}]  y∈[{centers[:,1].min():.2f}, "
          f"{centers[:,1].max():.2f}]  z∈[{centers[:,2].min():.2f}, "
          f"{centers[:,2].max():.2f}]")
    print(f"[anchor]   mean W,L,H (m):    {mean_wlh}")


if __name__ == "__main__":
    main()
