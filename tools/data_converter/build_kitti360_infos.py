#!/usr/bin/env python3
"""Build KITTI-360 -> DAPETR-parity mmengine info pkls.

Uses kitti360_mixed_camera.py's verified calibration/pose loaders plus a
direct XML parse (for raw box-local vertices, needed for dims+yaw) instead
of either of the two prior, untrusted conversion pipelines. See
TABLE2_KITTI360_DAPETR_PARITY docs/{split_definition,class_mapping,
crop_resize_choice}.md for the documented assumptions this encodes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mmengine
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kitti360_mixed_camera import (  # noqa: E402
    camera_to_world,
    load_cam_to_pose,
    load_mei,
    load_perspective,
    load_poses,
    parse_opencv_matrix,
)

DATA_ROOT = Path(os.environ.get("ROSA_KITTI360_ROOT", "data/KITTI-360"))
CALIB_ROOT = DATA_ROOT / "calibration"
OUT_ROOT = DATA_ROOT / "ImageSets-dapetr-v49-port"

VAL_SEQUENCES = ["0009", "0010"]
TRAIN_SEQUENCES = ["0000", "0002", "0003", "0004", "0005", "0006", "0007"]
ALL_SEQUENCES = TRAIN_SEQUENCES + VAL_SEQUENCES
SUBSAMPLE_STRIDE = 5  # native ~9.6-10Hz -> ~1.92-2Hz

CAMERA_KEYS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK_LEFT"]
IMAGE_DIRS = ["image_00", "image_01", "image_02", "image_03"]
IMAGE_FOLDERS = ["data_rect", "data_rect", "data_rgb", "data_rgb"]
CAMERA_MODEL_IDS = [0, 0, 1, 1]  # 0=pinhole, 1=MEI

CLASS_MAP = {
    "car": "car", "truck": "truck", "bus": "bus", "trailer": "trailer",
    "bicycle": "bicycle", "motorcycle": "motorcycle", "pedestrian": "pedestrian",
    "bigPole": "pole", "smallPole": "pole",
    "unknownObject": "object", "trafficSign": "traffic_sign",
}
CLASSES = (
    "car", "truck", "trailer", "bus", "bicycle", "motorcycle",
    "pedestrian", "pole", "object", "traffic_sign",
)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Must match rosa.integrations.kitti360.metric.CLASS_RANGE exactly -- this is
# the single source of truth both eval-time filtering and GT construction
# key off of. Without this filter, static objects (timestamp=-1, valid
# across a start_frame..end_frame window that can span hundreds of native
# frames = hundreds of meters of ego travel) get included as "active" at
# every frame in that window regardless of actual ego-distance at that
# specific frame -- e.g. a pole first seen 20s/180m away was still being
# emitted as a training target at the current frame, with no real object
# anywhere near its projected image location ("floating" boxes). Confirmed
# empirically: token 0010:0000003110 had 250/325 (77%) instances beyond
# 50m, median 99m, max 199m, before this filter was added.
CLASS_RANGE = {
    "car": 50.0, "truck": 50.0, "trailer": 50.0, "bus": 50.0,
    "bicycle": 40.0, "motorcycle": 40.0, "pedestrian": 40.0,
    "pole": 30.0, "object": 50.0, "traffic_sign": 40.0,
}


def _rect_corners_bev(x: float, y: float, l: float, w: float, yaw: float) -> np.ndarray:
    # CCW winding (required by _clip_polygon's inward-normal convention):
    # (dx,dy) -> (-dx,dy) -> (-dx,-dy) -> (dx,-dy).
    dx, dy = l / 2.0, w / 2.0
    local = np.array([[dx, dy], [-dx, dy], [-dx, -dy], [dx, -dy]])
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([x, y])


def _polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _clip_polygon(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman: clip `subject` polygon against convex `clip` polygon."""
    output = subject
    n = len(clip)
    for i in range(n):
        if len(output) == 0:
            break
        a, b = clip[i], clip[(i + 1) % n]
        edge = b - a
        normal = np.array([-edge[1], edge[0]])  # inward normal for CCW clip poly
        inputs = output
        output = []
        for j in range(len(inputs)):
            cur, prev = inputs[j], inputs[j - 1]
            cur_in = np.dot(normal, cur - a) >= 0
            prev_in = np.dot(normal, prev - a) >= 0
            if cur_in:
                if not prev_in:
                    d = edge_intersect(prev, cur, a, b)
                    if d is not None:
                        output.append(d)
                output.append(cur)
            elif prev_in:
                d = edge_intersect(prev, cur, a, b)
                if d is not None:
                    output.append(d)
        output = np.array(output) if output else np.empty((0, 2))
    return output


def edge_intersect(p1, p2, a, b):
    d1, d2 = p2 - p1, b - a
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-12:
        return None
    t = ((a[0] - p1[0]) * d2[1] - (a[1] - p1[1]) * d2[0]) / denom
    return p1 + t * d1


def bev_iou(box_a: tuple, box_b: tuple) -> float:
    """box_* = (x, y, l, w, yaw). Oriented-rectangle IoU via polygon clipping."""
    xa, ya, la, wa, yawa = box_a
    xb, yb, lb, wb, yawb = box_b
    if math.hypot(xa - xb, ya - yb) > (la + lb + wa + wb):
        return 0.0  # cheap reject
    poly_a = _rect_corners_bev(xa, ya, la, wa, yawa)
    poly_b = _rect_corners_bev(xb, yb, lb, wb, yawb)
    # ensure CCW winding for the clip polygon (yaw rotation of a rectangle
    # built from the corner order above is already CCW for yaw in [-pi,pi))
    if _polygon_area(poly_b) < 0:
        poly_b = poly_b[::-1]
    inter = _clip_polygon(poly_a, poly_b)
    inter_area = _polygon_area(inter)
    area_a = la * wa
    area_b = lb * wb
    union = area_a + area_b - inter_area
    if union <= 1e-9:
        return 0.0
    return inter_area / union


def dedup_instances(instances: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """Per-class oriented-BEV-IoU dedup, deterministic: keep lowest instance_id.

    KITTI-360's raw XML annotations contain genuine near/exact-duplicate
    objects (confirmed: distinct instanceIds with bit-identical `transform`
    translation, e.g. sequence 0010 car instanceIds 493-498 all share the
    exact same world position to full float precision) -- an upstream
    annotation-tool artifact, not a bug in this converter's own logic. Drop
    same-class instances within a frame that overlap an already-kept one
    above `iou_threshold`, keeping the numerically-lowest instance_token
    (parsed from the trailing ``:<id>`` so ordering is deterministic
    regardless of input order).
    """
    def _iid(inst):
        return int(inst["instance_token"].rsplit(":", 1)[1])

    by_class: dict[int, list[dict]] = {}
    for inst in instances:
        by_class.setdefault(inst["bbox_label_3d"], []).append(inst)

    kept: list[dict] = []
    for cls, group in by_class.items():
        group = sorted(group, key=_iid)
        survivors: list[dict] = []
        for inst in group:
            x, y, _, l, w, _, yaw = inst["bbox_3d"]
            box = (x, y, l, w, yaw)
            is_dup = False
            for s in survivors:
                sx, sy, _, sl, sw, _, syaw = s["bbox_3d"]
                if bev_iou(box, (sx, sy, sl, sw, syaw)) > iou_threshold:
                    is_dup = True
                    break
            if not is_dup:
                survivors.append(inst)
        kept.extend(survivors)
    return kept


@dataclass
class RawBox:
    label: str
    instance_id: int
    timestamp: int
    start_frame: int
    end_frame: int
    transform: np.ndarray  # 4x4, local->world
    local_vertices: np.ndarray  # 8x3, box-local (pre-transform)


def load_raw_boxes(xml_path: Path) -> list[RawBox]:
    out = []
    for child in ET.parse(xml_path).getroot():
        t_node, v_node = child.find("transform"), child.find("vertices")
        if t_node is None or v_node is None:
            continue
        label = child.findtext("label")
        if label not in CLASS_MAP:
            continue
        out.append(RawBox(
            label=label,
            instance_id=int(child.findtext("instanceId")),
            timestamp=int(child.findtext("timestamp")),
            start_frame=int(child.findtext("start_frame")),
            end_frame=int(child.findtext("end_frame")),
            transform=parse_opencv_matrix(t_node),
            local_vertices=parse_opencv_matrix(v_node),
        ))
    return out


def box_to_ego(box: RawBox, world_to_ego: np.ndarray) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Return (xyz, yaw, lwh) in ego frame, or None if degenerate.

    KITTI-360's per-object ``transform`` is NOT a rigid SE(3) transform --
    its 3x3 part encodes rotation * anisotropic scale together (verified
    empirically: raw local ``vertices`` always span exactly [1,1,1] extent,
    a canonical unit box, and the actual physical dims come out through the
    transform's column norms, e.g. a real car measures ~4.5x1.9x1.5m this
    way). Naively using ``transform[:3,:3]`` as a rotation for yaw would be
    wrong (it's not orthonormal) -- must factor out the per-column scale
    first to get a pure rotation before extracting yaw.
    """
    R_world = box.transform[:3, :3]
    T_world = box.transform[:3, 3]
    local_extent = box.local_vertices.max(axis=0) - box.local_vertices.min(axis=0)
    center_local = box.local_vertices.mean(axis=0)
    center_world = R_world @ center_local + T_world
    center_ego = world_to_ego[:3, :3] @ center_world + world_to_ego[:3, 3]

    col_norms = np.linalg.norm(R_world, axis=0)
    if not np.all(np.isfinite(col_norms)) or np.any(col_norms <= 1e-6):
        return None
    lwh = local_extent * col_norms
    if not np.all(np.isfinite(lwh)) or np.any(lwh <= 1e-3):
        return None

    R_pure_world = R_world / col_norms[None, :]  # column-wise unscale -> pure rotation
    R_pure_ego = world_to_ego[:3, :3] @ R_pure_world
    yaw = math.atan2(R_pure_ego[1, 0], R_pure_ego[0, 0])
    return center_ego.astype(np.float64), float(yaw), lwh.astype(np.float64)


def build_calibration() -> dict:
    """Per-camera FIXED (rig) calibration: lidar2cam, camera_intrinsic, distortion."""
    cam2pose = load_cam_to_pose(str(CALIB_ROOT / "calib_cam_to_pose.txt"))
    out = {}
    for idx, (key, cam_dir) in enumerate(zip(CAMERA_KEYS, IMAGE_DIRS)):
        c2p = cam2pose[cam_dir]
        if idx < 2:
            # Pinhole images come from data_rect/ (already rectified), so
            # the extrinsic must include R_rect -- P_rect (used below as
            # camera_intrinsic) operates in the RECTIFIED camera frame, not
            # the raw frame calib_cam_to_pose.txt's cam2pose describes.
            # ego_to_cam = R_rect @ inv(cam2pose), the inverse of the
            # already-verified (Phase 1) camera_to_world(pose, cam2pose,
            # rectification) = pose_to_world @ cam2pose @ inv(rectification).
            persp = load_perspective(str(CALIB_ROOT / "perspective.txt"), idx)
            lidar2cam = (persp.rectification @ np.linalg.inv(c2p)).astype(np.float32)
        else:
            lidar2cam = np.linalg.inv(c2p).astype(np.float32)
        if idx < 2:
            intrinsic = persp.projection[:3, :3].astype(np.float32)
            distortion = np.zeros(5, dtype=np.float32)
        else:
            mei = load_mei(str(CALIB_ROOT / f"image_{idx:02d}.yaml"))
            intrinsic = np.array(
                [[mei.gamma1, 0.0, mei.u0], [0.0, mei.gamma2, mei.v0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            distortion = np.array([mei.k1, mei.k2, mei.p1, mei.p2, mei.xi], dtype=np.float32)
        out[key] = dict(
            lidar2cam=lidar2cam.tolist(),
            camera_intrinsic=intrinsic.tolist(),
            distortion_coefficients=distortion.tolist(),
            camera_model_id=CAMERA_MODEL_IDS[idx],
        )
    return out


def enumerate_frames(seq: str) -> list[int]:
    ts_path = DATA_ROOT / "data_2d_raw" / f"2013_05_28_drive_{seq}_sync" / "image_00" / "timestamps.txt"
    n = len(ts_path.read_text().splitlines())
    poses = load_poses(str(DATA_ROOT / "data_poses" / f"2013_05_28_drive_{seq}_sync" / "poses.txt"))
    frames = []
    for frame in range(0, n, SUBSAMPLE_STRIDE):
        if frame not in poses:
            continue
        ok = True
        for cam_dir, folder in zip(IMAGE_DIRS, IMAGE_FOLDERS):
            img_path = DATA_ROOT / "data_2d_raw" / f"2013_05_28_drive_{seq}_sync" / cam_dir / folder / f"{frame:010d}.png"
            if not img_path.exists():
                ok = False
                break
        if ok:
            frames.append(frame)
    return frames


def convert_sequence(seq: str, calibration: dict, min_pts_placeholder: int = 1) -> list[dict]:
    seq_name = f"2013_05_28_drive_{seq}_sync"
    poses = load_poses(str(DATA_ROOT / "data_poses" / seq_name / "poses.txt"))
    boxes = load_raw_boxes(DATA_ROOT / "data_3d_bboxes" / "train" / f"{seq_name}.xml")
    frames = enumerate_frames(seq)

    data_list = []
    for frame in frames:
        pose_to_world = poses[frame]
        world_to_ego = np.linalg.inv(pose_to_world)

        active = [
            b for b in boxes
            if b.timestamp == frame or (b.timestamp == -1 and b.start_frame <= frame <= b.end_frame)
        ]
        instances = []
        for b in active:
            result = box_to_ego(b, world_to_ego)
            if result is None:
                continue
            xyz, yaw, lwh = result
            target_class = CLASS_MAP[b.label]
            ego_dist = float(math.hypot(xyz[0], xyz[1]))
            if ego_dist >= CLASS_RANGE[target_class]:
                continue
            instances.append(dict(
                bbox_3d=[float(xyz[0]), float(xyz[1]), float(xyz[2]),
                         float(lwh[0]), float(lwh[1]), float(lwh[2]), yaw],
                bbox_label_3d=CLASS_TO_IDX[target_class],
                velocity=[0.0, 0.0],
                num_lidar_pts=min_pts_placeholder,
                instance_token=f"{seq}:{b.label}:{b.instance_id}",
            ))
        instances = dedup_instances(instances)

        images = {}
        for key, cam_dir, folder in zip(CAMERA_KEYS, IMAGE_DIRS, IMAGE_FOLDERS):
            rel = f"data_2d_raw/{seq_name}/{cam_dir}/{folder}/{frame:010d}.png"
            images[key] = dict(
                img_path=rel,
                lidar2cam=calibration[key]["lidar2cam"],
                camera_intrinsic=calibration[key]["camera_intrinsic"],
                distortion_coefficients=calibration[key]["distortion_coefficients"],
            )

        data_list.append(dict(
            token=f"{seq}:{frame:010d}",
            sequence=seq,
            frame_id=frame,
            images=images,
            instances=instances,
        ))
    return data_list


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", nargs="+", default=None,
                         help="override sequence list (default: all)")
    parser.add_argument("--out-suffix", default="", help="e.g. '_smoke' for gate runs")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    calibration = build_calibration()

    seqs = args.sequences or ALL_SEQUENCES
    train_seqs = [s for s in seqs if s in TRAIN_SEQUENCES]
    val_seqs = [s for s in seqs if s in VAL_SEQUENCES]

    for split_name, split_seqs in [("train", train_seqs), ("val", val_seqs)]:
        data_list = []
        for seq in split_seqs:
            frames_data = convert_sequence(seq, calibration)
            data_list.extend(frames_data)
            n_inst = sum(len(f["instances"]) for f in frames_data)
            print(f"  seq {seq} ({split_name}): {len(frames_data)} frames, {n_inst} instances")
        out = dict(
            metainfo=dict(classes=CLASSES, version="dapetr-v49-port-v1"),
            data_list=data_list,
        )
        out_path = OUT_ROOT / f"kitti360_dapetr_infos_{split_name}{args.out_suffix}.pkl"
        mmengine.dump(out, str(out_path))
        n_inst_total = sum(len(f["instances"]) for f in data_list)
        print(f"{split_name}: {len(data_list)} frames, {n_inst_total} instances -> {out_path}")


if __name__ == "__main__":
    main()
