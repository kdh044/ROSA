"""Small runtime bridge between ROSA and a Sparse4D decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

_REQUIRED_NATIVE_REFINE_KWARGS = {
    "time_interval",
    "return_cls",
}
_OPTIONAL_NATIVE_REFINE_KWARGS = {
    "classification_feature",
    "regression_feature",
    "score_alignment_feature",
}


@dataclass(frozen=True)
class NativeRefineCall:
    """Arguments observed at the last native Sparse4D refinement module."""

    args: tuple[Any, ...]
    kwargs: Mapping[str, Any]

    def __post_init__(self) -> None:
        if len(self.args) != 3:
            raise ValueError("native refine call must be (query, anchor, anchor_embed)")
        if not all(torch.is_tensor(value) for value in self.args):
            raise TypeError("native refine positional arguments must be tensors")

    @property
    def query(self) -> Tensor:
        return self.args[0]

    @property
    def anchor(self) -> Tensor:
        return self.args[1]

    @property
    def anchor_embed(self) -> Tensor:
        return self.args[2]


class SingleCallCapture:
    """Capture exactly one kwargs-aware native refine invocation."""

    def __init__(self, *, force_return_cls: bool = False) -> None:
        self.call: NativeRefineCall | None = None
        self.force_return_cls = bool(force_return_cls)

    def __call__(self, _module: nn.Module, args, kwargs):
        if self.call is not None:
            raise RuntimeError("native final refine was invoked more than once")
        call_kwargs = dict(kwargs)
        changed = self.force_return_cls and call_kwargs.get("return_cls") is not True
        if changed:
            call_kwargs["return_cls"] = True
        self.call = NativeRefineCall(tuple(args), call_kwargs)
        return (args, call_kwargs) if changed else None

    def take(self) -> NativeRefineCall:
        if self.call is None:
            raise RuntimeError("native final refine call was not captured")
        call, self.call = self.call, None
        return call


def validate_native_call(call: NativeRefineCall) -> None:
    """Fail closed if the audited Sparse4D refine signature changes."""

    observed = set(call.kwargs)
    extended = _REQUIRED_NATIVE_REFINE_KWARGS | _OPTIONAL_NATIVE_REFINE_KWARGS
    if observed not in (_REQUIRED_NATIVE_REFINE_KWARGS, extended):
        raise RuntimeError(
            "native refine keyword signature changed: expected either "
            f"{sorted(_REQUIRED_NATIVE_REFINE_KWARGS)} or {sorted(extended)}, "
            f"got {sorted(observed)}"
        )
    populated_sidecars = {
        name: call.kwargs[name]
        for name in _OPTIONAL_NATIVE_REFINE_KWARGS
        if name in observed and call.kwargs[name] is not None
    }
    if populated_sidecars:
        raise RuntimeError(
            "native refine sidecars must be None at the audited call boundary: "
            f"{sorted(populated_sidecars)}"
        )
    if call.kwargs["return_cls"] is not True:
        raise RuntimeError("native capture must materialize class/quality")


def restore_fpn(feature_maps) -> list[Tensor]:
    """Convert packed Sparse4D features to ``[B,C,D,H,W]`` levels.

    The import of the Sparse4D CUDA/plugin package is intentionally delayed
    until its packed representation is actually supplied.
    """

    if (
        isinstance(feature_maps, (list, tuple))
        and feature_maps
        and all(torch.is_tensor(value) and value.ndim == 5 for value in feature_maps)
    ):
        return list(feature_maps)
    if (
        isinstance(feature_maps, (list, tuple))
        and len(feature_maps) == 3
        and torch.is_tensor(feature_maps[0])
        and feature_maps[0].ndim == 3
    ):
        try:
            from projects.mmdet3d_plugin.ops import feature_maps_format
        except ImportError as exc:  # pragma: no cover - framework environment
            raise ImportError(
                "packed FPN restoration requires the Sparse4D plugin on " "PYTHONPATH"
            ) from exc
        camera_groups = feature_maps_format(feature_maps, inverse=True)
        if not camera_groups or not camera_groups[0]:
            raise RuntimeError("packed FPN restored no camera/level tensors")
        num_levels = len(camera_groups[0])
        if any(len(group) != num_levels for group in camera_groups):
            raise RuntimeError("camera groups disagree on FPN level count")
        restored = []
        for level in range(num_levels):
            shapes = {tuple(group[level].shape[-2:]) for group in camera_groups}
            if len(shapes) != 1:
                raise RuntimeError(
                    f"camera FPN shapes disagree at level {level}: " f"{sorted(shapes)}"
                )
            # Equal camera shapes are packed as one group by Sparse4D.  Its
            # inverse formatter already returns the required [B,C,D,H,W]
            # tensor as a strided view of the packed feature storage.  Calling
            # ``cat`` on that single tensor would materialize every dense FPN
            # level again solely to make it contiguous.  Both ROSA samplers
            # accept the inverse formatter's strides, so retain the view and
            # reserve concatenation for the heterogeneous multi-group case.
            restored.append(
                camera_groups[0][level]
                if len(camera_groups) == 1
                else torch.cat(
                    [group[level] for group in camera_groups],
                    dim=1,
                )
            )
        return restored
    raise TypeError("unsupported Sparse4D FPN representation")


def detached_fpn(feature_maps) -> list[Tensor]:
    """Restore the FPN and close the gradient boundary to the detector."""

    restored = restore_fpn(feature_maps)
    values = [value.detach() for value in restored]
    for value, source in zip(values, restored):
        if value.requires_grad or value.grad_fn is not None:
            raise RuntimeError("ROSA FPN detach boundary is not closed")
        if (
            value.shape != source.shape
            or value.dtype != source.dtype
            or value.device != source.device
            or value.stride() != source.stride()
        ):
            raise RuntimeError("ROSA FPN detach changed the tensor contract")
    return values


def find_operation_indices(
    operation_order: Sequence[str], operation: str
) -> tuple[int, ...]:
    return tuple(
        index for index, observed in enumerate(operation_order) if observed == operation
    )


def remap_legacy_refiner_state_dict(
    state_dict: dict,
    *,
    prefix: str = "",
) -> int:
    """Move old ``canonical_sixth_block`` keys to ``rosa_refiner``.

    Returns the number of consumed legacy keys. Existing modern keys win when
    a mixed checkpoint contains both names.
    """

    legacy_prefix = prefix + "canonical_sixth_block."
    modern_prefix = prefix + "rosa_refiner."
    consumed = 0
    for key in list(state_dict):
        if not key.startswith(legacy_prefix):
            continue
        modern_key = modern_prefix + key[len(legacy_prefix) :]
        if modern_key not in state_dict:
            state_dict[modern_key] = state_dict[key]
        state_dict.pop(key)
        consumed += 1
    return consumed


__all__ = [
    "NativeRefineCall",
    "SingleCallCapture",
    "detached_fpn",
    "find_operation_indices",
    "remap_legacy_refiner_state_dict",
    "restore_fpn",
    "validate_native_call",
]
