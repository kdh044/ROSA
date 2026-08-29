"""K-means anchor initialization for the KITTI-360 DAPETR-parity arm.

Mirrors the Fisheye3DOD anchor file's structure (verified empirically):
only the [x, y, z] center varies per anchor via k-means over GT box
centers; size/yaw/velocity are fixed canonical defaults the same for
every anchor row, refined during training.
"""

from __future__ import annotations

import argparse

import mmengine
import numpy as np
from sklearn.cluster import KMeans


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pkl", required=True)
    parser.add_argument("--num-anchor", type=int, default=900)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = mmengine.load(args.train_pkl)
    centers = []
    sizes = []
    for info in data["data_list"]:
        for instance in info.get("instances", ()):
            box = instance["bbox_3d"]
            if int(instance["bbox_label_3d"]) < 0:
                continue
            centers.append(box[:3])
            sizes.append(box[3:6])
    centers = np.asarray(centers, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float64)
    print(f"fit k-means on {len(centers)} GT centers -> {args.num_anchor} clusters")

    kmeans = KMeans(
        n_clusters=args.num_anchor, random_state=args.seed, n_init=10
    ).fit(centers)
    anchor_centers = kmeans.cluster_centers_.astype(np.float32)

    mean_size = sizes.mean(axis=0).astype(np.float32)
    print("mean GT size (l,w,h):", mean_size)

    num = anchor_centers.shape[0]
    anchor = np.zeros((num, 11), dtype=np.float32)
    anchor[:, :3] = anchor_centers
    anchor[:, 3:6] = mean_size
    anchor[:, 6] = 0.0  # sin(yaw)
    anchor[:, 7] = 1.0  # cos(yaw)
    anchor[:, 8:11] = 0.0  # velocity

    np.save(args.output, anchor)
    print(f"wrote {args.output} shape={anchor.shape}")


if __name__ == "__main__":
    main()
