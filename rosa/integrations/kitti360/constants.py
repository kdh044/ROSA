"""ICRA 2026 DAPETR KITTI-360 experimental constants."""

from __future__ import annotations


# Preserve the channel order used by the released FishBEVOD converter.
CAMERAS = (
    "CAM_FRONT",        # KITTI-360 image_00, rectified pinhole.
    "CAM_FRONT_RIGHT",  # KITTI-360 image_01, rectified pinhole.
    "CAM_BACK_RIGHT",   # KITTI-360 image_02, native MEI fisheye.
    "CAM_BACK_LEFT",    # KITTI-360 image_03, native MEI fisheye.
)

# Benchmark label space for this experiment.
#
# The public FishBEVOD/DAPETR setup evaluates the eleven keys of the nuScenes
# ``detection_cvpr_2019`` class_range, but on converted KITTI-360 four of them
# can never score above zero:
#
#   trafficsign          the converter writes ``object.trafficsign`` while the
#                        evaluator's mapping only recognizes
#                        ``static_object.trafficsign`` -> no ground truth;
#   barrier              same prefix mismatch for ``object.smallpole`` /
#                        ``object.lamp`` vs ``movable_object.pushable_pullable``;
#   traffic_cone         KITTI-360 has no such label at all;
#   construction_vehicle its ground truth comes from buildings and garages,
#                        which the ten-class training list never predicts.
#
# Averaging four structurally-zero APs into mAP only rescales every method by
# the same constant, so this benchmark keeps the classes that actually carry
# driving-relevant ground truth.  ``bus`` is additionally dropped: the whole
# dataset holds 74 instances and FishBEVOD's own table reports AP 0.000/0.002
# for it, which is noise rather than signal.
CLASSES = (
    "car",
    "truck",
    "trailer",
    "motorcycle",
    "bicycle",
    "pedestrian",
)

# KITTI-360 annotates a cyclist/motorcyclist as a separate person-sized box
# (median 1.23 x 0.94 x 1.81 m, the same height as ``pedestrian``) next to the
# vehicle box, exactly as nuScenes does, so riders join the pedestrian class.
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

# nuScenes category names emitted by the FishBEVOD converter, mapped to the
# same label space.  Both spellings are accepted so the info builder can read
# either the converter's tables or the raw KITTI-360 XML.
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

# [x0, y0, x1, y1] from the released DAPETR mixed-camera config.
FISHEYE_CROPS = (
    (14, 600, 1364, 960),
    (50, 600, 1400, 960),
)

CAMERA_MODEL_IDS = (0, 0, 1, 1)  # 0=pinhole, 1=MEI

# DAPETR/PETR benchmark range.
POINT_CLOUD_RANGE = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
POST_CENTER_RANGE = (-61.2, -61.2, -10.0, 61.2, 61.2, 10.0)

