"""SphereStream-Sparse4D — fisheye 3D detection on Sparse4D v3 backbone.

Importing this package registers all custom modules into mmdet3d/mmengine
registries so configs can reference them by ``type=...``.
"""
from projects.mmdet3d_plugin.models import *  # noqa: F401,F403
from projects.mmdet3d_plugin.datasets import *  # noqa: F401,F403
from projects.mmdet3d_plugin.data_preprocessor import *  # noqa: F401,F403

__version__ = "0.1.0"
