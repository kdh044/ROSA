"""KITTI-360 calibration readers (perspective and MEI cameras)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

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
