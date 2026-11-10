"""KITTI-360 benchmark constants (DAPETR protocol)."""

from __future__ import annotations


# Camera order used by DAPETR's KITTI-360 converter.
CAMERAS = (
    "CAM_FRONT",        # KITTI-360 image_00, rectified pinhole.
    "CAM_FRONT_RIGHT",  # KITTI-360 image_01, rectified pinhole.
    "CAM_BACK_RIGHT",   # KITTI-360 image_02, native MEI fisheye.
    "CAM_BACK_LEFT",    # KITTI-360 image_03, native MEI fisheye.
)

# Evaluated classes.  Of the eleven nuScenes classes used by DAPETR,
# trafficsign, barrier, traffic_cone and construction_vehicle have no usable
# ground truth on converted KITTI-360, and bus has only 74 instances in the
# whole dataset, so they are excluded.
CLASSES = (
    "car",
    "truck",
    "trailer",
    "motorcycle",
    "bicycle",
    "pedestrian",
)

# Riders are annotated as separate person-sized boxes and are merged into
# ``pedestrian``.
KITTI360_LABEL_TO_CLASS = {
    "car": "car",
    "truck": "truck",
    "trailer": "trailer",
    "caravan": "trailer",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "pedestrian": "pedestrian",
    "rider": "pedestrian",
}

# nuScenes category names emitted by DAPETR's KITTI-360 converter, mapped to
# the same label space.
NUSC_CATEGORY_TO_CLASS = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.trailer": "trailer",
    "vehicle.caravan": "trailer",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.personal_mobility": "pedestrian",
}

IMAGE_SIZE = (1408, 376)  # width, height
NATIVE_IMAGE_SIZES = (
    (1408, 376),
    (1408, 376),
    (1400, 1400),
    (1400, 1400),
)

# [x0, y0, x1, y1] from the DAPETR mixed-camera config.
FISHEYE_CROPS = (
    (14, 600, 1364, 960),
    (50, 600, 1400, 960),
)

CAMERA_MODEL_IDS = (0, 0, 1, 1)  # 0=pinhole, 1=MEI

# DAPETR/PETR benchmark range.
POINT_CLOUD_RANGE = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
POST_CENTER_RANGE = (-61.2, -61.2, -10.0, 61.2, 61.2, 10.0)

