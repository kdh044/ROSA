#!/usr/bin/env python3
"""Dump and evaluate fixed native Sparse4D decoder-stage routes.

The dump command executes D0--D5 once per validation batch and stores only
the raw decoder inputs needed for post-hoc routing.  The evaluate command
then measures two fixed, dataset-global routing matrices without rerunning
the detector:

* ``score_bundle``: ``B=b, C=s, Q=s``
* ``box_aligned_quality``: ``B=b, C=c, Q=b``

This tool deliberately ignores the decoder's emitted ``cls_scores`` field.
That diagnostic is not aligned after centerness re-sorting in the legacy
decoder; routing always starts from the saved raw classification and quality
logits instead.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable

import torch
from torch import Tensor

DUMP_SCHEMA = "rosa.stage_routes.raw.v1"
METRIC_SCHEMA = "rosa.stage_routes.metrics.v1"
NATIVE_STAGES = 6
MATRIX_SCORE_BUNDLE = "score_bundle"
MATRIX_BOX_ALIGNED_QUALITY = "box_aligned_quality"
MATRIX_NAMES = (
    MATRIX_SCORE_BUNDLE,
    MATRIX_BOX_ALIGNED_QUALITY,
)
_RAW_OUTPUT_NAMES = ("prediction", "classification", "quality")
_DN_OUTPUT_KEYS = frozenset(
    {
        "dn_prediction",
        "dn_classification",
        "dn_reg_target",
        "dn_cls_target",
        "dn_valid_mask",
        "temp_dn_reg_target",
        "temp_dn_cls_target",
        "temp_dn_valid_mask",
        "dn_id_target",
    }
)


@dataclass(frozen=True)
class RouteSpec:
    """One dataset-global source assignment for decoder inputs."""

    matrix: str
    box_stage: int
    classification_stage: int
    quality_stage: int

    @property
    def key(self) -> str:
        return (
            f"B{self.box_stage}_C{self.classification_stage}" f"_Q{self.quality_stage}"
        )


@dataclass(frozen=True)
class CapturedRefineInput:
    """The row-aligned query and anchor supplied to one native refine op."""

    query: Tensor
    anchor: Tensor


class NativeRefineCapture:
    """Capture every native refine input exactly once for one forward."""

    def __init__(self, num_stages: int = NATIVE_STAGES) -> None:
        self.num_stages = int(num_stages)
        self.records: list[CapturedRefineInput | None] = []
        self.reset()

    def reset(self) -> None:
        self.records = [None] * self.num_stages

    def hook(self, stage: int):
        stage = int(stage)

        def capture(_module, args, kwargs):
            del kwargs
            if self.records[stage] is not None:
                raise RuntimeError(f"native D{stage} refine executed twice")
            if len(args) < 2 or not all(torch.is_tensor(x) for x in args[:2]):
                raise RuntimeError(
                    f"native D{stage} refine signature lost query/anchor tensors"
                )
            self.records[stage] = CapturedRefineInput(args[0], args[1])
            return None

        return capture


def route_specs(matrix: str) -> tuple[RouteSpec, ...]:
    """Return deterministic global routes for one declared matrix."""

    if matrix == MATRIX_SCORE_BUNDLE:
        return tuple(
            RouteSpec(matrix, box_stage, score_stage, score_stage)
            for box_stage in range(NATIVE_STAGES)
            for score_stage in range(NATIVE_STAGES)
        )
    if matrix == MATRIX_BOX_ALIGNED_QUALITY:
        return tuple(
            RouteSpec(matrix, box_stage, classification_stage, box_stage)
            for box_stage in range(NATIVE_STAGES)
            for classification_stage in range(NATIVE_STAGES)
        )
    raise ValueError(f"unknown route matrix {matrix!r}; expected {MATRIX_NAMES}")


def selected_matrices(selection: str) -> tuple[str, ...]:
    if selection == "all":
        return MATRIX_NAMES
    if selection not in MATRIX_NAMES:
        raise ValueError(f"unknown matrix selection {selection!r}")
    return (selection,)


def native_refine_indices(operation_order: Sequence[str]) -> tuple[int, ...]:
    indices = tuple(
        index
        for index, operation in enumerate(operation_order)
        if operation == "refine"
    )
    if len(indices) != NATIVE_STAGES:
        raise RuntimeError(
            f"stage routing requires exactly {NATIVE_STAGES} refine ops, "
            f"found {len(indices)}"
        )
    return indices


def validate_head_contract(head: Any) -> tuple[int, ...]:
    """Fail closed unless inference is temporal-free, DN-free, and diagnostic."""

    if bool(getattr(head, "training", True)):
        raise RuntimeError("stage routing requires head.eval()")
    if int(getattr(head, "num_decoder", -1)) != NATIVE_STAGES:
        raise RuntimeError("stage routing requires native D0-D5")
    if getattr(head, "max_temporal_gnn_interactions", None) != 0:
        raise RuntimeError("stage routing requires temporal GNN interactions off")

    instance_bank = getattr(head, "instance_bank", None)
    if (
        instance_bank is None
        or int(getattr(instance_bank, "num_temp_instances", -1)) != 0
    ):
        raise RuntimeError("stage routing requires num_temp_instances=0")

    sampler = getattr(head, "sampler", None)
    if sampler is None:
        raise RuntimeError("stage routing requires the native sampler contract")
    if int(getattr(sampler, "num_temp_dn_groups", -1)) != 0:
        raise RuntimeError("stage routing requires num_temp_dn_groups=0")
    if getattr(sampler, "dn_metas", None) is not None:
        raise RuntimeError("stage routing requires empty denoising state")
    if getattr(head, "_diagnostic_return_all_cls", False) is not True:
        raise RuntimeError("stage routing requires _diagnostic_return_all_cls=True")

    return native_refine_indices(getattr(head, "operation_order", ()))


def _finite(name: str, tensor: Tensor) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise RuntimeError(f"{name} contains a non-finite value")


def validate_native_batch(
    model_outs: Mapping[str, Any],
    captures: Sequence[CapturedRefineInput | None],
) -> dict[str, list[Tensor]]:
    """Validate D0--D5 tensors and prove row continuity at runtime."""

    leaked_dn = sorted(_DN_OUTPUT_KEYS.intersection(model_outs))
    if leaked_dn:
        raise RuntimeError(f"eval forward unexpectedly emitted DN state: {leaked_dn}")
    if len(captures) != NATIVE_STAGES or any(item is None for item in captures):
        raise RuntimeError("every native refine op must execute exactly once")

    native: dict[str, list[Tensor]] = {}
    for name in _RAW_OUTPUT_NAMES:
        values = model_outs.get(name)
        if not isinstance(values, (list, tuple)):
            raise RuntimeError(f"model_outs[{name!r}] must be a stage list")
        if len(values) not in (NATIVE_STAGES, NATIVE_STAGES + 1):
            raise RuntimeError(
                f"{name} must contain D0-D5 with at most one ROSA output, "
                f"got {len(values)} entries"
            )
        selected = list(values[:NATIVE_STAGES])
        if any(not torch.is_tensor(value) for value in selected):
            raise RuntimeError(
                f"all D0-D5 {name} tensors must materialize; "
                "keep eval mode and enable diagnostic all-cls"
            )
        native[name] = selected

    prediction = native["prediction"]
    batch, queries = prediction[0].shape[:2]
    if batch <= 0 or queries <= 0:
        raise RuntimeError("native prediction batch/query axes must be nonempty")
    for name, values in native.items():
        reference_shape = values[0].shape
        for stage, value in enumerate(values):
            if value.shape[:2] != (batch, queries):
                raise RuntimeError(
                    f"D{stage} {name} changed the batch/query axes: "
                    f"{tuple(value.shape[:2])} != {(batch, queries)}"
                )
            if value.shape != reference_shape:
                raise RuntimeError(
                    f"D{stage} {name} shape {tuple(value.shape)} differs from "
                    f"D0 {tuple(reference_shape)}"
                )
            _finite(f"D{stage} {name}", value)

    typed_captures = [item for item in captures if item is not None]
    for stage, capture in enumerate(typed_captures):
        if capture.query.shape[:2] != (batch, queries):
            raise RuntimeError(f"D{stage} refine query axes are not row-aligned")
        if capture.anchor.shape != prediction[stage].shape:
            raise RuntimeError(f"D{stage} refine anchor shape changed")
        _finite(f"D{stage} refine query", capture.query)
        _finite(f"D{stage} refine anchor", capture.anchor)

    # Between refinements, Sparse4D may update row contents but must not sort
    # or select rows.  In this temporal-free release, the exact D_s prediction
    # object becomes the D_{s+1} anchor input, so bitwise equality is a strong
    # runtime proof of the row-identity assumption used by score routing.
    for stage in range(NATIVE_STAGES - 1):
        next_anchor = typed_captures[stage + 1].anchor
        if not torch.equal(prediction[stage], next_anchor):
            max_error = float(
                (prediction[stage] - next_anchor).abs().max().detach().cpu()
            )
            raise RuntimeError(
                f"row identity failed between D{stage} and D{stage + 1}; "
                f"anchor max_abs_error={max_error:.8g}"
            )

    instance_id = model_outs.get("instance_id")
    if not torch.is_tensor(instance_id) or instance_id.shape != (batch, queries):
        raise RuntimeError(
            "production decode requires one aligned instance_id per query"
        )
    return native


def _cpu_clone(tensor: Tensor) -> Tensor:
    return tensor.detach().to(device="cpu").contiguous().clone()


def _sample_metainfo(data_sample: Any) -> Mapping[str, Any]:
    metainfo = getattr(data_sample, "metainfo", None)
    return metainfo if isinstance(metainfo, Mapping) else {}


def _sample_ground_truth(data_sample: Any) -> dict[str, Any]:
    annotation = getattr(data_sample, "eval_ann_info", None)
    if not isinstance(annotation, Mapping):
        raise RuntimeError("test sample is missing eval_ann_info")
    boxes = annotation.get("gt_bboxes_3d")
    labels = annotation.get("gt_labels_3d")
    if boxes is None or not hasattr(boxes, "tensor") or labels is None:
        raise RuntimeError("eval_ann_info is missing LiDAR boxes or labels")
    box_tensor = _cpu_clone(boxes.tensor)
    label_tensor = torch.as_tensor(labels, dtype=torch.long).cpu().contiguous()
    if box_tensor.ndim != 2 or label_tensor.ndim != 1:
        raise RuntimeError("ground-truth boxes/labels have invalid ranks")
    if box_tensor.shape[0] != label_tensor.shape[0]:
        raise RuntimeError("ground-truth boxes and labels are misaligned")
    return {
        "boxes": box_tensor,
        "labels": label_tensor,
        # BaseInstance3DBoxes.tensor is already stored in its canonical
        # bottom-centred representation, irrespective of the constructor's
        # source origin.  Reconstructing it with gravity-centre origin would
        # subtract half the height for a second time.
        "box_origin": [0.5, 0.5, 0.0],
    }


def pack_raw_batch(
    model_outs: Mapping[str, Any],
    data_samples: Sequence[Any],
    captures: Sequence[CapturedRefineInput | None],
) -> list[dict[str, Any]]:
    """Convert one validated live batch to simple CPU tensor records."""

    native = validate_native_batch(model_outs, captures)
    batch = native["prediction"][0].shape[0]
    if len(data_samples) != batch:
        raise RuntimeError("data-sample count disagrees with decoder batch")
    # Transfer one stacked tensor per output family rather than issuing a
    # separate device-to-host copy for every sample and decoder stage.
    stacked_native = {
        name: _cpu_clone(torch.stack(values, dim=1)) for name, values in native.items()
    }
    instance_id = _cpu_clone(model_outs["instance_id"])
    records = []
    for batch_index, data_sample in enumerate(data_samples):
        metainfo = _sample_metainfo(data_sample)
        token = metainfo.get("token", getattr(data_sample, "token", None))
        sample_idx = metainfo.get(
            "sample_idx", getattr(data_sample, "sample_idx", token)
        )
        if token is None:
            raise RuntimeError("test sample is missing its token")
        records.append(
            {
                "token": str(token),
                "sample_idx": str(sample_idx),
                "prediction": stacked_native["prediction"][batch_index].clone(),
                "classification": stacked_native["classification"][batch_index].clone(),
                "quality": stacked_native["quality"][batch_index].clone(),
                "instance_id": instance_id[batch_index].clone(),
                "ground_truth": _sample_ground_truth(data_sample),
            }
        )
    return records


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)


def decoder_contract(decoder: Any) -> dict[str, Any]:
    fields = (
        "num_output",
        "score_threshold",
        "sorted",
        "selection_score_mode",
        "duplicate_bev_radius",
    )
    return {name: _jsonable(getattr(decoder, name, None)) for name in fields}


def make_dump_payload(
    *,
    samples: Sequence[Mapping[str, Any]],
    dataset_meta: Mapping[str, Any],
    operation_order: Sequence[str],
    decoder: Any,
    config_path: Path,
    config_fingerprint: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    payload = {
        "schema": DUMP_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "config": str(config_path.resolve()),
            "effective_config_sha256": config_fingerprint,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
        },
        "contract": {
            "native_stages": NATIVE_STAGES,
            "temporal_enabled": False,
            "denoising_enabled": False,
            "diagnostic_all_classification": True,
            "row_identity_verified": True,
            "raw_decoder_inputs": True,
            "emitted_cls_scores_stored": False,
            "decoded_box_origin": [0.5, 0.5, 0.5],
            "operation_order": list(operation_order),
            "decoder": decoder_contract(decoder),
        },
        "dataset_meta": _jsonable(dataset_meta),
        "sample_count": len(samples),
        "samples": list(samples),
    }
    validate_dump_payload(payload)
    return payload


def validate_dump_payload(payload: Mapping[str, Any]) -> None:
    """Validate the portable raw-input schema before save or evaluation."""

    if payload.get("schema") != DUMP_SCHEMA:
        raise ValueError(
            f"unsupported stage-route dump schema {payload.get('schema')!r}"
        )
    contract = payload.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("stage-route dump is missing its contract")
    exact_flags = {
        "native_stages": NATIVE_STAGES,
        "temporal_enabled": False,
        "denoising_enabled": False,
        "diagnostic_all_classification": True,
        "row_identity_verified": True,
        "raw_decoder_inputs": True,
        "emitted_cls_scores_stored": False,
    }
    for name, expected in exact_flags.items():
        if contract.get(name) != expected:
            raise ValueError(
                f"invalid stage-route contract {name}: "
                f"{contract.get(name)!r} != {expected!r}"
            )
    operation_order = contract.get("operation_order")
    if not isinstance(operation_order, list):
        raise ValueError("stage-route contract is missing operation_order")
    native_refine_indices(operation_order)
    if contract.get("decoded_box_origin") != [0.5, 0.5, 0.5]:
        raise ValueError("stage-route decoded-box origin contract changed")
    if not isinstance(contract.get("decoder"), Mapping):
        raise ValueError("stage-route dump is missing its decoder contract")
    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("stage-route dump is missing source provenance")
    for name in ("effective_config_sha256", "checkpoint_sha256"):
        digest = source.get(name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"stage-route source has invalid {name}")
    if not isinstance(payload.get("dataset_meta"), Mapping):
        raise ValueError("stage-route dump is missing dataset_meta")

    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("stage-route dump contains no samples")
    if payload.get("sample_count") != len(samples):
        raise ValueError("stage-route sample_count is inconsistent")
    tokens: set[str] = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise ValueError(f"sample {index} is not a mapping")
        token = sample.get("token")
        if not isinstance(token, str) or not token:
            raise ValueError(f"sample {index} has no token")
        if token in tokens:
            raise ValueError(f"duplicate sample token {token!r}")
        tokens.add(token)
        tensors = [sample.get(name) for name in _RAW_OUTPUT_NAMES]
        if any(not torch.is_tensor(value) for value in tensors):
            raise ValueError(f"sample {token} is missing raw decoder tensors")
        prediction, classification, quality = tensors
        if any(value.ndim != 3 for value in tensors):
            raise ValueError(f"sample {token} raw decoder tensors must be rank 3")
        if any(value.shape[0] != NATIVE_STAGES for value in tensors):
            raise ValueError(f"sample {token} does not contain D0-D5")
        query_count = prediction.shape[1]
        if query_count <= 0 or any(value.shape[1] != query_count for value in tensors):
            raise ValueError(f"sample {token} query axes disagree")
        if any(value.device.type != "cpu" for value in tensors):
            raise ValueError(f"sample {token} raw tensors must be on CPU")
        if any(not bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError(f"sample {token} raw tensors are non-finite")
        if (
            prediction.shape[-1] < 9
            or classification.shape[-1] <= 0
            or quality.shape[-1] <= 0
        ):
            raise ValueError(f"sample {token} raw decoder feature axes are invalid")
        instance_id = sample.get("instance_id")
        if not torch.is_tensor(instance_id) or instance_id.shape != (query_count,):
            raise ValueError(f"sample {token} instance_id is misaligned")
        if instance_id.device.type != "cpu" or instance_id.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError(f"sample {token} instance_id must be a CPU integer")
        ground_truth = sample.get("ground_truth")
        if not isinstance(ground_truth, Mapping):
            raise ValueError(f"sample {token} is missing ground truth")
        gt_boxes = ground_truth.get("boxes")
        gt_labels = ground_truth.get("labels")
        if (
            not torch.is_tensor(gt_boxes)
            or not torch.is_tensor(gt_labels)
            or gt_boxes.ndim != 2
            or gt_labels.ndim != 1
            or gt_boxes.shape[0] != gt_labels.shape[0]
        ):
            raise ValueError(f"sample {token} ground truth is malformed")
        if (
            gt_boxes.device.type != "cpu"
            or gt_labels.device.type != "cpu"
            or not bool(torch.isfinite(gt_boxes).all())
        ):
            raise ValueError(f"sample {token} ground truth must be finite CPU data")
        if ground_truth.get("box_origin") != [0.5, 0.5, 0.0]:
            raise ValueError(
                f"sample {token} ground truth must store canonical box tensors"
            )


def _stage_list(tensor: Tensor) -> list[Tensor]:
    return [tensor[stage].unsqueeze(0) for stage in range(NATIVE_STAGES)]


def decode_raw_route(
    decoder: Any,
    sample: Mapping[str, Any],
    route: RouteSpec,
) -> dict[str, Tensor]:
    """Decode one fixed route from raw logits, never emitted diagnostics."""

    if not all(
        0 <= stage < NATIVE_STAGES
        for stage in (
            route.box_stage,
            route.classification_stage,
            route.quality_stage,
        )
    ):
        raise ValueError(f"route {route.key} selects a non-native stage")
    predictions = _stage_list(sample["prediction"])
    classifications = _stage_list(sample["classification"])
    qualities = _stage_list(sample["quality"])
    classifications[route.box_stage] = classifications[route.classification_stage]
    qualities[route.box_stage] = qualities[route.quality_stage]
    instance_id = sample["instance_id"].unsqueeze(0)
    decoded_batch = decoder.decode(
        classifications,
        predictions,
        instance_id,
        qualities,
        output_idx=route.box_stage,
    )
    if not isinstance(decoded_batch, list) or len(decoded_batch) != 1:
        raise RuntimeError("decoder must emit exactly one routed sample")
    decoded = decoded_batch[0]
    required = ("boxes_3d", "scores_3d", "labels_3d")
    if not isinstance(decoded, Mapping) or any(
        name not in decoded for name in required
    ):
        raise RuntimeError("decoder route output is missing boxes/scores/labels")
    # Intentionally construct a fresh mapping.  In particular, never forward
    # the legacy decoder's misaligned emitted `cls_scores` diagnostic.
    return {name: decoded[name] for name in required}


def routed_metric_records(
    payload: Mapping[str, Any],
    decoder: Any,
    route: RouteSpec,
    box_factory: Callable[[Tensor, Sequence[float]], Any],
) -> list[dict[str, Any]]:
    """Build evaluator records using one route for every dumped sample."""

    records = []
    prediction_origin = payload["contract"]["decoded_box_origin"]
    for sample in payload["samples"]:
        decoded = decode_raw_route(decoder, sample, route)
        ground_truth = sample["ground_truth"]
        records.append(
            {
                "token": sample["token"],
                "sample_idx": sample["sample_idx"],
                "gt_instances_3d": {
                    "gt_bboxes_3d": box_factory(
                        ground_truth["boxes"],
                        ground_truth["box_origin"],
                    ),
                    "gt_labels_3d": ground_truth["labels"].numpy(),
                },
                "pred_instances_3d": {
                    "bboxes_3d": box_factory(decoded["boxes_3d"], prediction_origin),
                    "scores_3d": decoded["scores_3d"],
                    "labels_3d": decoded["labels_3d"],
                },
            }
        )
    return records


def _atomic_torch_save(value: Any, path: Path, *, force: bool) -> None:
    path = path.resolve()
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}; pass --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_json_save(value: Any, path: Path, *, force: bool) -> None:
    path = path.resolve()
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}; pass --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_jsonable(value), indent=2, ensure_ascii=False) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(payload)
            temporary = Path(stream.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_dump(path: Path) -> dict[str, Any]:
    payload = torch.load(path.resolve(), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("stage-route dump root must be a dictionary")
    validate_dump_payload(payload)
    return payload


def _config_from_args(args: argparse.Namespace):
    from mmengine.config import Config

    config = Config.fromfile(args.config)
    if args.cfg_options:
        config.merge_from_dict(args.cfg_options)
    return config


def _config_fingerprint(config: Any) -> str:
    return hashlib.sha256(config.pretty_text.encode("utf-8")).hexdigest()


_RUN_TOOL_MODULE_NAME = "_rosa_stage_route_run_tool"


def _checkpoint_run_tool():
    """Load the sibling run tool without relying on ambient ``sys.path``."""

    existing = sys.modules.get(_RUN_TOOL_MODULE_NAME)
    if existing is not None:
        return existing
    module_path = Path(__file__).resolve().with_name("run.py")
    spec = importlib.util.spec_from_file_location(_RUN_TOOL_MODULE_NAME, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load checkpoint contract tool from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_RUN_TOOL_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_RUN_TOOL_MODULE_NAME, None)
        raise
    return module


def _validate_checkpoint_contract(checkpoint: Path, config: Any) -> None:
    """Apply the train/evaluate topology contract before MMEngine loading."""

    _checkpoint_run_tool()._validate_checkpoint_contract(
        checkpoint,
        config,
        allow_mismatch=False,
    )


def _dataset_metainfo(dataset: Any) -> Mapping[str, Any]:
    current = dataset
    for _ in range(8):
        metainfo = getattr(current, "metainfo", None)
        if isinstance(metainfo, Mapping):
            return metainfo
        nested = getattr(current, "dataset", None)
        if nested is None or nested is current:
            break
        current = nested
    raise RuntimeError("test dataset exposes no metainfo")


def _unwrapped_model(model: Any) -> Any:
    from mmengine.model import is_model_wrapper

    while is_model_wrapper(model):
        model = model.module
    return model


def dump_command(args: argparse.Namespace) -> None:
    from mmengine.runner import Runner

    config = _config_from_args(args)
    fingerprint = _config_fingerprint(config)
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config.resume = False
    config.launcher = "none"
    if args.work_dir is not None:
        config.work_dir = str(args.work_dir.resolve())
    elif not config.get("work_dir"):
        config.work_dir = str(args.output.resolve().parent / "_stage_routes")

    # MMEngine checkpoints may contain its own trusted HistoryBuffer metadata.
    # This affects checkpoint loading only; the produced dump is weights-only
    # loadable because its schema contains tensors and primitive containers.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    with _checkpoint_run_tool()._private_checkpoint_snapshot(checkpoint) as snapshot:
        _validate_checkpoint_contract(snapshot.path, config)
        config.load_from = str(snapshot.path)
        runner = Runner.from_cfg(config)
        runner.load_or_resume()
        checkpoint_sha256 = snapshot.sha256
    model = _unwrapped_model(runner.model)
    model.eval()
    head = getattr(model, "head", None)
    if head is None:
        raise RuntimeError("configured model exposes no Sparse4D head")

    previous_marker = getattr(head, "_diagnostic_return_all_cls", None)
    had_marker = hasattr(head, "_diagnostic_return_all_cls")
    head._diagnostic_return_all_cls = True
    refine_indices = validate_head_contract(head)
    capture = NativeRefineCapture()
    handles = [
        head.layers[layer_index].register_forward_pre_hook(
            capture.hook(stage), with_kwargs=True
        )
        for stage, layer_index in enumerate(refine_indices)
    ]

    samples: list[dict[str, Any]] = []
    try:
        with torch.inference_mode():
            for batch_index, batch in enumerate(runner.test_dataloader):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
                processed = model.data_preprocessor(batch, training=False)
                if not isinstance(processed, Mapping):
                    raise RuntimeError("data preprocessor must return a mapping")
                data_samples = processed.get("data_samples")
                if not isinstance(data_samples, Sequence):
                    raise RuntimeError("test batch exposes no data_samples")
                capture.reset()
                model_outs = model(
                    processed["inputs"],
                    data_samples=data_samples,
                    mode="tensor",
                )
                packed = pack_raw_batch(model_outs, data_samples, capture.records)
                if args.max_samples is not None:
                    remaining = args.max_samples - len(samples)
                    packed = packed[: max(remaining, 0)]
                samples.extend(packed)
                print(
                    f"dumped {len(samples)} samples after batch {batch_index + 1}",
                    flush=True,
                )
                if args.max_samples is not None and len(samples) >= args.max_samples:
                    break
        validate_head_contract(head)
    finally:
        for handle in handles:
            handle.remove()
        if had_marker:
            head._diagnostic_return_all_cls = previous_marker
        else:
            delattr(head, "_diagnostic_return_all_cls")

    if not samples:
        raise RuntimeError("stage-route dump collected no validation samples")
    payload = make_dump_payload(
        samples=samples,
        dataset_meta=_dataset_metainfo(runner.test_dataloader.dataset),
        operation_order=head.operation_order,
        decoder=head.decoder,
        config_path=args.config,
        config_fingerprint=fingerprint,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
    )
    _atomic_torch_save(payload, args.output, force=args.force)
    print(
        json.dumps(
            {
                "schema": DUMP_SCHEMA,
                "output": str(args.output.resolve()),
                "sample_count": len(samples),
                "checkpoint_sha256": payload["source"]["checkpoint_sha256"],
            },
            indent=2,
        )
    )


def _build_decoder_and_metric(config: Any):
    from mmengine.utils import import_modules_from_strings
    from mmdet3d.registry import METRICS, TASK_UTILS
    from mmdet3d.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    custom_imports = config.get("custom_imports")
    if custom_imports:
        import_modules_from_strings(**custom_imports)
    decoder = TASK_UTILS.build(config.model.head.decoder)
    evaluator_config = config.test_evaluator
    if isinstance(evaluator_config, (list, tuple)):
        if len(evaluator_config) != 1:
            raise RuntimeError("stage routing requires exactly one test metric")
        evaluator_config = evaluator_config[0]
    metric = METRICS.build(evaluator_config)
    return decoder, metric


def evaluate_command(args: argparse.Namespace) -> None:
    payload = load_dump(args.dump)
    config = _config_from_args(args)
    fingerprint = _config_fingerprint(config)
    expected_fingerprint = payload["source"]["effective_config_sha256"]
    if fingerprint != expected_fingerprint and not args.allow_config_mismatch:
        raise ValueError(
            "effective config does not match the raw dump; pass "
            "--allow-config-mismatch only for an audited relocation"
        )
    decoder, metric = _build_decoder_and_metric(config)
    if decoder_contract(decoder) != payload["contract"]["decoder"]:
        raise ValueError("decoder contract differs from the raw dump")
    metric.dataset_meta = payload["dataset_meta"]

    from mmdet3d.structures import LiDARInstance3DBoxes

    def box_factory(tensor: Tensor, origin: Sequence[float]):
        return LiDARInstance3DBoxes(
            tensor,
            box_dim=tensor.shape[-1],
            origin=tuple(float(value) for value in origin),
        )

    matrices: dict[str, Any] = {}
    for matrix in selected_matrices(args.matrix):
        cells = []
        for route in route_specs(matrix):
            records = routed_metric_records(
                payload,
                decoder,
                route,
                box_factory,
            )
            metrics = metric.compute_metrics(records)
            cells.append({**asdict(route), "key": route.key, "metrics": metrics})
            print(f"evaluated {matrix} {route.key}", flush=True)
        matrices[matrix] = {
            "definition": (
                "B=b,C=s,Q=s" if matrix == MATRIX_SCORE_BUNDLE else "B=b,C=c,Q=b"
            ),
            "global_route": True,
            "cells": cells,
        }

    result = {
        "schema": METRIC_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_dump": str(args.dump.resolve()),
        "source_checkpoint_sha256": payload["source"]["checkpoint_sha256"],
        "sample_count": payload["sample_count"],
        "raw_decoder_inputs": True,
        "emitted_cls_scores_used": False,
        "matrices": matrices,
    }
    _atomic_json_save(result, args.output, force=args.force)
    print(
        json.dumps(
            {
                "schema": METRIC_SCHEMA,
                "output": str(args.output.resolve()),
                "matrices": list(matrices),
            },
            indent=2,
        )
    )


def _add_config_options(parser: argparse.ArgumentParser) -> None:
    from mmengine.config import DictAction

    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override effective MMEngine config entries as key=value pairs.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    dump = commands.add_parser(
        "dump", help="Run D0-D5 once and save raw decoder inputs."
    )
    _add_config_options(dump)
    dump.add_argument("checkpoint", type=Path)
    dump.add_argument("--output", type=Path, required=True)
    dump.add_argument("--work-dir", type=Path)
    dump.add_argument("--max-batches", type=int)
    dump.add_argument("--max-samples", type=int)
    dump.add_argument("--force", action="store_true")

    evaluate = commands.add_parser(
        "evaluate", help="Evaluate fixed global routes from one raw dump."
    )
    _add_config_options(evaluate)
    evaluate.add_argument("dump", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument(
        "--matrix",
        choices=("all", *MATRIX_NAMES),
        default="all",
    )
    evaluate.add_argument("--allow-config-mismatch", action="store_true")
    evaluate.add_argument("--force", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.command == "dump":
        if args.max_batches is not None and args.max_batches <= 0:
            raise ValueError("--max-batches must be positive")
        if args.max_samples is not None and args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "dump":
        dump_command(args)
    else:
        evaluate_command(args)


if __name__ == "__main__":
    main()
