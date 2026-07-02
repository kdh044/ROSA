# ------------------------------------------------------------------------
# Ported from Sparse4D's MultiScaleDepthMapGenerator
# (projects/mmdet3d_plugin/datasets/pipelines/transform.py)
# to mmcv 2.x / mmdet3d 1.4 (TRANSFORMS registry, BaseTransform).
#
# Projects LiDAR points to each camera's image plane via `lidar2img` and
# emits a list of multi-scale sparse depth maps (one per FPN level).
# This is for dense depth-branch supervision only. It is not the
# object-query center ray/range target used by the current research.
# ------------------------------------------------------------------------
from typing import List

import numpy as np

from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS

__all__ = ["MultiScaleDepthMapGenerator"]


@TRANSFORMS.register_module()
class MultiScaleDepthMapGenerator(BaseTransform):
    def __init__(self, downsample=1, max_depth=60):
        if not isinstance(downsample, (list, tuple)):
            downsample = [downsample]
        self.downsample = list(downsample)
        self.max_depth = float(max_depth)

    def transform(self, results: dict) -> dict:
        points = results["points"]
        if hasattr(points, "tensor"):
            points = points.tensor.cpu().numpy()
        elif hasattr(points, "numpy"):
            points = points.numpy()
        points = np.asarray(points)[..., :3, None]   # (N_pts, 3, 1)

        lidar2img = results["lidar2img"]             # (N_cam, 4, 4)
        img_shape = results["img_shape"]             # list[(H, W, C)]

        gt_depth: List[List[np.ndarray]] = []
        for i in range(len(lidar2img)):
            l2i = np.asarray(lidar2img[i], dtype=np.float64)
            H, W = img_shape[i][:2]
            pts_cam = (
                np.squeeze(l2i[:3, :3] @ points, axis=-1) + l2i[:3, 3]
            )  # (N_pts, 3)
            depths = pts_cam[:, 2]
            mask = depths > 0.1
            pts_cam = pts_cam[mask]
            depths = depths[mask]
            U = np.round(pts_cam[:, 0] / depths).astype(np.int32)
            V = np.round(pts_cam[:, 1] / depths).astype(np.int32)
            in_img = (V >= 0) & (V < H) & (U >= 0) & (U < W)
            U, V, depths = U[in_img], V[in_img], depths[in_img]
            # Painter algorithm: sort by depth descending; far points
            # written first, closer points overwrite.
            sort_idx = np.argsort(depths)[::-1]
            U, V, depths = U[sort_idx], V[sort_idx], depths[sort_idx]
            depths = np.clip(depths, 0.1, self.max_depth)
            for j, downsample in enumerate(self.downsample):
                if len(gt_depth) < j + 1:
                    gt_depth.append([])
                h, w = (int(H / downsample), int(W / downsample))
                u = np.floor(U / downsample).astype(np.int32)
                v = np.floor(V / downsample).astype(np.int32)
                depth_map = np.full((h, w), -1.0, dtype=np.float32)
                depth_map[v, u] = depths.astype(np.float32)
                gt_depth[j].append(depth_map)

        results["gt_depth"] = [np.stack(x, axis=0) for x in gt_depth]
        return results
