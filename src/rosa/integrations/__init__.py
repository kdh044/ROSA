"""Optional detector and dataset bindings for ROSA.

Importing :mod:`rosa` never imports this package.  Framework registrations are
performed only when an integration module is named explicitly in an MMEngine
``custom_imports`` list.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


def __getattr__(name: str) -> Any:
    if name in {
        "ROSARefinementHead",
        "ROSAOfficialResidualRefinementHead",
        "ClosedLoopROSAHead",
    }:
        module = import_module(".sparse4d", __name__)
        return getattr(module, name)
    raise AttributeError(name)


__all__ = [
    "ClosedLoopROSAHead",
    "ROSAOfficialResidualRefinementHead",
    "ROSARefinementHead",
]
