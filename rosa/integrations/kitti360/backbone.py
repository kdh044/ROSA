"""Register the VoVNet-99 (``VoVNetCP``) backbone of the Fisheye3DOD checkout."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


def register_vovnet(source: str | os.PathLike[str] | None = None) -> None:
    """Load ``models/petr/vovnetcp.py`` from the Fisheye3DOD checkout."""

    if source is None:
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
            f"VoVNetCP source not found: {path}; set FISHEYE3DOD_OFFICIAL_ROOT"
        )
    # Load under the package's own module name so that a later
    # ``from .vovnetcp import VoVNetCP`` does not register the class twice.
    module_name = "fisheye3dod.models.petr.vovnetcp"
    if module_name in sys.modules:
        return

    from mmdet3d.registry import MODELS

    if "VoVNetCP" in MODELS.module_dict:
        return
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    parent = sys.modules.get("fisheye3dod.models.petr")
    if parent is not None:
        setattr(parent, "vovnetcp", module)


register_vovnet()

from mmdet3d.registry import MODELS  # noqa: E402

_VoVNetCP = MODELS.get("VoVNetCP")


@MODELS.register_module()
class ROSAVoVNetCP(_VoVNetCP):
    """``VoVNetCP`` with optional gradient checkpointing.

    The official implementation always checkpoints its OSA modules during
    training, which breaks DDP with ``find_unused_parameters=True``.  ROSA was
    trained with ``with_cp=False``.
    """

    def __init__(self, *args, with_cp: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for module in self.modules():
            if hasattr(module, "use_checkpoint"):
                module.use_checkpoint = with_cp


__all__ = ["ROSAVoVNetCP", "register_vovnet"]
