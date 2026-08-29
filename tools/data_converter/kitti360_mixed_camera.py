"""Standalone KITTI-360 mixed perspective/MEI camera geometry.

This module is intentionally isolated from the running Fisheye3DOD jobs.
After projection parity is verified it can be wired into the Sparse4D data
adapter without changing the OCam experiment protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import yaml


@dataclass(frozen=True)
class PerspectiveCalibration:
    projection: np.ndarray
    rectification: np.ndarray
    image_wh: tuple[int, int]


@dataclass(frozen=True)
class MeiCalibration:
    xi: float
    k1: float
    k2: float
    p1: float
    p2: float
    gamma1: float
    gamma2: float
    u0: float
    v0: float
    image_wh: tuple[int, int]


@dataclass(frozen=True)
class BoxRecord:
    label: str
    instance_id: int
    timestamp: int
    start_frame: int
    end_frame: int
    transform: np.ndarray
    vertices_world: np.ndarray


def read_opencv_yaml(path: str | Path) -> dict:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    text = "\n".join(lines[1:])
    text = re.sub(r":([^ ])", r": \1", text)
    return yaml.safe_load(text)


def load_cam_to_pose(path: str | Path) -> dict[str, np.ndarray]:
    output = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        name, values = line.split(":", 1)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3] = np.asarray(
            [float(value) for value in values.split()], dtype=np.float64
        ).reshape(3, 4)
        output[name] = matrix
    return output


def load_perspective(path: str | Path, camera_id: int) -> PerspectiveCalibration:
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        name, payload = line.split(":", 1)
        try:
            values[name] = np.asarray(
                [float(value) for value in payload.split()], dtype=np.float64
            )
        except ValueError:
            continue
    projection = values[f"P_rect_{camera_id:02d}"].reshape(3, 4)
    rectification = np.eye(4, dtype=np.float64)
    rectification[:3, :3] = values[f"R_rect_{camera_id:02d}"].reshape(3, 3)
    width, height = values[f"S_rect_{camera_id:02d}"].astype(int)
    return PerspectiveCalibration(
        projection=projection,
        rectification=rectification,
        image_wh=(int(width), int(height)),
    )


def load_mei(path: str | Path) -> MeiCalibration:
    value = read_opencv_yaml(path)
    distortion = value["distortion_parameters"]
    projection = value["projection_parameters"]
    return MeiCalibration(
        xi=float(value["mirror_parameters"]["xi"]),
        k1=float(distortion["k1"]),
        k2=float(distortion["k2"]),
        p1=float(distortion["p1"]),
        p2=float(distortion["p2"]),
        gamma1=float(projection["gamma1"]),
        gamma2=float(projection["gamma2"]),
        u0=float(projection["u0"]),
        v0=float(projection["v0"]),
        image_wh=(int(value["image_width"]), int(value["image_height"])),
    )


def load_poses(path: str | Path) -> dict[int, np.ndarray]:
    values = np.loadtxt(path)
    output = {}
    for row in values:
        pose = np.eye(4, dtype=np.float64)
        pose[:3] = row[1:].reshape(3, 4)
        output[int(row[0])] = pose
    return output


def camera_to_world(
    pose_to_world: np.ndarray,
    camera_to_pose: np.ndarray,
    rectification: np.ndarray | None = None,
) -> np.ndarray:
    output = pose_to_world @ camera_to_pose
    if rectification is not None:
        output = output @ np.linalg.inv(rectification)
    return output


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    homogeneous = np.concatenate(
        (points, np.ones((len(points), 1), dtype=np.float64)), axis=1
    )
    return (np.asarray(transform, dtype=np.float64) @ homogeneous.T).T[:, :3]


def world_to_camera(
    points_world: np.ndarray, camera_to_world_matrix: np.ndarray
) -> np.ndarray:
    return transform_points(points_world, np.linalg.inv(camera_to_world_matrix))


def project_perspective(
    points_camera: np.ndarray, calibration: PerspectiveCalibration
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    # KITTI-360's released cam_to_pose already contains the stereo baseline.
    # The official viewer therefore uses only P_rect[:3, :3] after applying
    # that camera pose.  Adding P_rect[:, 3] here would apply the image_01
    # baseline twice and shift nearby boxes horizontally.
    projected = (calibration.projection[:, :3] @ points.T).T
    depth = projected[:, 2]
    safe = np.where(np.abs(depth) > 1e-12, depth, 1e-12)
    uv = projected[:, :2] / safe[:, None]
    width, height = calibration.image_wh
    valid = (
        (depth > 0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    return uv, valid


def project_mei(
    points_camera: np.ndarray,
    calibration: MeiCalibration,
    *,
    include_tangential_distortion: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Project with KITTI-360's unified MEI model.

    The official viewer ignores p1/p2.  The default therefore matches that
    executed reference; the optional branch is available for a later
    calibration sensitivity check.
    """
    points = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    norm = np.linalg.norm(points, axis=1).clip(min=1e-12)
    unit = points / norm[:, None]
    denominator = unit[:, 2] + calibration.xi
    safe = np.where(np.abs(denominator) > 1e-12, denominator, 1e-12)
    x = unit[:, 0] / safe
    y = unit[:, 1] / safe
    radius2 = x * x + y * y
    radial = 1.0 + calibration.k1 * radius2 + calibration.k2 * radius2**2
    if include_tangential_distortion:
        distorted_x = (
            x * radial
            + 2.0 * calibration.p1 * x * y
            + calibration.p2 * (radius2 + 2.0 * x * x)
        )
        distorted_y = (
            y * radial
            + calibration.p1 * (radius2 + 2.0 * y * y)
            + 2.0 * calibration.p2 * x * y
        )
    else:
        distorted_x = x * radial
        distorted_y = y * radial
    uv = np.stack(
        (
            calibration.gamma1 * distorted_x + calibration.u0,
            calibration.gamma2 * distorted_y + calibration.v0,
        ),
        axis=-1,
    )
    width, height = calibration.image_wh
    valid = (
        (points[:, 2] > 0)
        & (denominator > 0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    return uv, valid


def parse_opencv_matrix(node: ET.Element) -> np.ndarray:
    rows = int(node.findtext("rows"))
    columns = int(node.findtext("cols"))
    values = [float(value) for value in node.findtext("data", "").split()]
    return np.asarray(values, dtype=np.float64).reshape(rows, columns)


def load_box_records(path: str | Path) -> list[BoxRecord]:
    output = []
    for child in ET.parse(path).getroot():
        transform_node = child.find("transform")
        vertices_node = child.find("vertices")
        if transform_node is None or vertices_node is None:
            continue
        transform = parse_opencv_matrix(transform_node)
        vertices = parse_opencv_matrix(vertices_node)
        output.append(
            BoxRecord(
                label=str(child.findtext("label")),
                instance_id=int(child.findtext("instanceId")),
                timestamp=int(child.findtext("timestamp")),
                start_frame=int(child.findtext("start_frame")),
                end_frame=int(child.findtext("end_frame")),
                transform=transform,
                vertices_world=transform_points(vertices, transform),
            )
        )
    return output


def active_box_records(
    records: list[BoxRecord], frame: int, classes: set[str]
) -> list[BoxRecord]:
    return [
        record
        for record in records
        if record.label in classes
        and (
            record.timestamp == frame
            or (
                record.timestamp == -1
                and record.start_frame <= frame <= record.end_frame
            )
        )
    ]
