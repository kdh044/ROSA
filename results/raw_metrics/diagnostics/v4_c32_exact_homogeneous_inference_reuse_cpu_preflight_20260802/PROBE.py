#!/usr/bin/env python3
"""Reproduce exact homogeneous OCam operator-output storage accounting."""

from __future__ import annotations

import json

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from rosa.geometry import ocam_project_points


SEED = 3407
SHAPE = {"batch": 4, "cameras": 4, "anchors": 300, "keypoints": 13}
EXPECTED = {
    "parent": {"owned_output_storage_bytes": 24_336_280, "storages": 95},
    "candidate": {"owned_output_storage_bytes": 14_851_464, "storages": 51},
}


def _walk(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)


def _storage_id(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage()._cdata


class _Capture(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.tensors = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        # Strong references prevent allocator recycling before accounting.
        self.tensors.extend(_walk(output))
        return output


def _fixture():
    generator = torch.Generator().manual_seed(SEED)
    batch = SHAPE["batch"]
    cameras = SHAPE["cameras"]
    anchors = SHAPE["anchors"]
    keypoints = SHAPE["keypoints"]
    dtype = torch.float32
    points = torch.randn(
        (batch, anchors, keypoints, 3),
        generator=generator,
        dtype=dtype,
    )
    points[..., 2].sub_(2.0)
    lidar2cam = torch.eye(4, dtype=dtype).expand(
        batch, cameras, 4, 4
    ).clone()
    lidar2cam[..., :3, 3] = torch.randn(
        (batch, cameras, 3),
        generator=generator,
        dtype=dtype,
    ).mul_(0.05)
    intrinsic = torch.zeros((batch, cameras, 3, 3), dtype=dtype)
    intrinsic[..., 0, 0] = 1.0
    intrinsic[..., 0, 1] = -0.000077
    intrinsic[..., 0, 2] = 398.967516
    intrinsic[..., 1, 0] = 0.999625
    intrinsic[..., 1, 1] = -0.000148
    intrinsic[..., 1, 2] = 399.029971
    intrinsic[..., 2, 2] = 1.0
    coefficients = torch.tensor(
        (
            331.189089,
            214.532114,
            16.021439,
            26.018342,
            17.311014,
            4.130653,
            2.940404,
            4.460478,
            2.256958,
            0.36679,
        ),
        dtype=dtype,
    )
    polynomial = coefficients.expand(batch, cameras, -1).clone()
    image_wh = torch.full((batch, cameras, 2), 800.0, dtype=dtype)
    metadata = {
        "lidar2cam": lidar2cam,
        "ocam_intrinsic": intrinsic,
        "ocam_pol": polynomial,
        "ocam_lens_fov_deg": 220.0,
    }
    return points, metadata, image_wh


def _capture(reuse: bool):
    points, metadata, image_wh = _fixture()
    input_storage_ids = {
        _storage_id(tensor)
        for tensor in _walk((points, metadata, image_wh))
    }
    capture = _Capture()
    with torch.no_grad(), capture:
        output = ocam_project_points(
            points,
            metadata,
            image_wh,
            contract="full_fov",
            camera_transform="homogeneous_4x4",
            inference_storage_reuse=reuse,
        )
    storages = {}
    for tensor in capture.tensors:
        storage = tensor.untyped_storage()
        storages.setdefault(storage._cdata, storage.nbytes())
    owned = {
        key: value
        for key, value in storages.items()
        if key not in input_storage_ids
    }
    record = {
        "owned_output_storage_bytes": sum(owned.values()),
        "storages": len(owned),
        "captured_output_storage_bytes_including_inputs": sum(storages.values()),
        "captured_storages_including_inputs": len(storages),
        "excluded_input_alias_bytes": sum(storages.values()) - sum(owned.values()),
        "excluded_input_alias_storages": len(storages) - len(owned),
        "output_stride": list(output.stride()),
    }
    return output.clone(), record


def main() -> None:
    torch.set_num_threads(1)
    parent_output, parent = _capture(False)
    candidate_output, candidate = _capture(True)
    raw_equal = torch.equal(parent_output, candidate_output)
    stride_equal = parent["output_stride"] == candidate["output_stride"]
    if parent["owned_output_storage_bytes"] != EXPECTED["parent"][
        "owned_output_storage_bytes"
    ] or parent["storages"] != EXPECTED["parent"]["storages"]:
        raise RuntimeError(f"parent storage receipt drifted: {parent}")
    if candidate["owned_output_storage_bytes"] != EXPECTED["candidate"][
        "owned_output_storage_bytes"
    ] or candidate["storages"] != EXPECTED["candidate"]["storages"]:
        raise RuntimeError(f"candidate storage receipt drifted: {candidate}")
    if not raw_equal or not stride_equal:
        raise RuntimeError("candidate projection changed output values or strides")
    removed = (
        parent["owned_output_storage_bytes"]
        - candidate["owned_output_storage_bytes"]
    )
    payload = {
        "schema": "rosa.c32.exact_homogeneous.storage_probe.v1",
        "scope": (
            "cumulative distinct operator-output storage while every emitted "
            "tensor is retained; excludes pre-existing input storage aliases"
        ),
        "not_measured": ["peak allocator memory", "RSS", "CUDA", "latency"],
        "device": "cpu",
        "dtype": "torch.float32",
        "seed": SEED,
        "shape": SHAPE,
        "camera_conditioned_points": False,
        "contract": "full_fov",
        "camera_transform": "homogeneous_4x4",
        "parent": parent,
        "candidate": candidate,
        "bytes_removed": removed,
        "fraction_removed": removed / parent["owned_output_storage_bytes"],
        "storages_removed": parent["storages"] - candidate["storages"],
        "raw_bit_equal": raw_equal,
        "stride_equal": stride_equal,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
