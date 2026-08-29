"""Register FishBEVOD's released VoVNet-99 implementation in MMEngine."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


def register_fishbevod_vovnet(source: str | os.PathLike[str] | None = None) -> None:
    """Load the released VoVNetCP source without copying third-party code."""

    if source is None:
        source = os.environ.get("FISHBEVOD_VOVNET_SOURCE")
    if source is None:
        # FishBEVOD ships an MMDetection 2.x source (``mmcv.runner``).  Use
        # the repository's behavior-equivalent MMEngine port so the exact
        # V-99-eSE topology can run in the Sparse4D/MMDet3D 1.4 host.
        # Released layout: the Fisheye3DOD checkout sits next to this
        # repository, or FISHEYE3DOD_OFFICIAL_ROOT points at it.
        root = os.environ.get("FISHEYE3DOD_OFFICIAL_ROOT")
        base = (
            Path(root)
            if root
            else Path(__file__).resolve().parents[3] / "external" / "fisheye3dod"
        )
        source = base / "models" / "petr" / "vovnetcp.py"
    path = Path(source).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"FishBEVOD VoVNetCP source not found: {path}; set "
            "FISHBEVOD_VOVNET_SOURCE"
        )
    # Load under the *canonical* dotted name rather than a synthetic one.
    #
    # ``VoVNetCP`` decorates itself with ``@MODELS.register_module()`` at class
    # definition, and that registry is global, so executing the file twice is
    # a hard error ("already registered") no matter which module names the two
    # copies live under.  Anything that imports ``fisheye3dod.models.petr``
    # the normal way -- every PETR-family head ported here does -- runs
    # ``from .vovnetcp import VoVNetCP`` inside that package's ``__init__``.
    # Registering the file under the name that import resolves to means the
    # second import is a cache hit instead of a second execution, so load
    # order between this module and ``fisheye3dod`` stops mattering.  Checking
    # the registry first is kept for the reverse order (``fisheye3dod``
    # already imported), where nothing further is needed.
    module_name = "fisheye3dod.models.petr.vovnetcp"
    if module_name in sys.modules:
        return

    from mmdet3d.registry import MODELS

    if "VoVNetCP" in MODELS.module_dict:
        return
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import VoVNetCP from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    # ``from .vovnetcp import VoVNetCP`` resolves the submodule through
    # ``sys.modules`` but then reads it as an attribute of the parent package,
    # which importlib only sets when it performed the import itself.
    parent = sys.modules.get("fisheye3dod.models.petr")
    if parent is not None:
        setattr(parent, "vovnetcp", module)


register_fishbevod_vovnet()


__all__ = ["register_fishbevod_vovnet"]
