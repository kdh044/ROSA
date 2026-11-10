"""Sparse4D components used by ROSA.

Adapted from Sparse4D (https://github.com/HorizonRobotics/Sparse4D, MIT)
for MMDetection3D 1.x and fisheye cameras. Importing the package registers the
modules so that configs can refer to them by ``type=...``.
"""
from projects.mmdet3d_plugin.models import *  # noqa: F401,F403
from projects.mmdet3d_plugin.datasets import *  # noqa: F401,F403
from projects.mmdet3d_plugin.data_preprocessor import *  # noqa: F401,F403

