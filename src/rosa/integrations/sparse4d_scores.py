"""Score, loss, and decode helpers retained by the Sparse4D binding."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from .. import rfl as _rfl


def _pack_frozen_decode_fields(
    decoded_boxes: Tensor,
    scores: Tensor,
    labels: Tensor,
    indices: Tensor,
    class_origin: Tensor,
    instance_ids: Tensor,
) -> list[dict]:
    """Consolidate an already-decoded batch into two host transfers."""

    box_width = decoded_boxes.shape[-1]
    floating_cpu = torch.cat(
        (decoded_boxes, scores[..., None], class_origin[..., None]),
        dim=-1,
    ).cpu()
    integer_cpu = torch.stack((labels, indices, instance_ids), dim=-1).cpu()
    return [
        {
            "boxes_3d": floating_cpu[index, :, :box_width].clone(),
            "scores_3d": floating_cpu[index, :, box_width].clone(),
            "labels_3d": integer_cpu[index, :, 0].clone(),
            "query_indices": integer_cpu[index, :, 1].clone(),
            "cls_scores": floating_cpu[index, :, box_width + 1].clone(),
            "instance_ids": integer_cpu[index, :, 2].clone(),
        }
        for index in range(decoded_boxes.shape[0])
    ]


def _matches_released_sparse4d_decoder(decoder) -> bool:
    """Seal the host decoder implementation before batching its transfers."""

    try:
        from projects.mmdet3d_plugin.models.detection3d.decoder import (
            SparseBox3DDecoder,
            _RELEASED_DECODE_BOX_IMPL,
        )
    except ImportError:
        return False
    return bool(
        type(decoder) is SparseBox3DDecoder
        and "decode_box" not in decoder.__dict__
        and type(decoder).decode_box is _RELEASED_DECODE_BOX_IMPL
    )


def _decode_frozen_box_batch(
    decoder,
    box_predictions: Tensor,
    indices: Tensor,
) -> Tensor:
    """Gather and decode the released dense box batch with one method call."""

    batch_size, num_query = indices.shape
    selected_boxes = torch.gather(
        box_predictions,
        1,
        indices[..., None].expand_as(box_predictions),
    )
    decoded_boxes = decoder.decode_box(selected_boxes.reshape(-1, 11))
    return decoded_boxes.reshape(batch_size, num_query, -1)


def _probability_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def stable_native_joint_logit(
    selected_class_logit: Tensor,
    centerness_logit: Tensor,
    *,
    validate_values: bool = True,
) -> Tensor:
    """Compute ``logit(sigmoid(class) * sigmoid(centerness))`` stably."""

    if (
        selected_class_logit.shape != centerness_logit.shape
        or not selected_class_logit.is_floating_point()
    ):
        raise ValueError("native class/centerness logits must be floating/aligned")
    if not isinstance(validate_values, bool):
        raise TypeError("validate_values must be a bool")
    if validate_values and (
        not bool(torch.isfinite(selected_class_logit).all())
        or not bool(torch.isfinite(centerness_logit).all())
    ):
        raise ValueError("native class/centerness logits must be finite")
    dtype = _probability_dtype(selected_class_logit.dtype)
    selected = selected_class_logit.detach().to(dtype=dtype)
    centerness = centerness_logit.detach().to(dtype=dtype)
    result = -torch.logsumexp(
        torch.stack((-selected, -centerness, -selected - centerness), dim=-1),
        dim=-1,
    )
    if validate_values and not bool(torch.isfinite(result).all()):
        raise FloatingPointError("native joint logit is non-finite")
    return result.detach()


def stable_joint_probability(
    joint_logit: Tensor,
    native_joint_score: Tensor,
    effective_delta: Tensor,
    *,
    validate_values: bool = True,
) -> Tensor:
    """Emit sigmoid score with a bit-exact zero-residual identity bridge."""

    if (
        joint_logit.ndim != 2
        or native_joint_score.shape != joint_logit.shape
        or effective_delta.shape != joint_logit.shape
    ):
        raise ValueError("joint score tensors are misaligned")
    if not isinstance(validate_values, bool):
        raise TypeError("validate_values must be a bool")
    dtype = _probability_dtype(joint_logit.dtype)
    raw = joint_logit.to(dtype=dtype).sigmoid()
    native = native_joint_score.detach().to(dtype=dtype)
    delta = effective_delta.to(dtype=dtype)
    exact_identity = native + (raw - raw.detach())
    probability = torch.where(delta.eq(0), exact_identity, raw)
    if validate_values and not bool(torch.isfinite(probability).all()):
        raise FloatingPointError("joint probability is non-finite")
    return probability


def compose_quality_rank(
    baseline_classification: Tensor,
    baseline_quality: Tensor,
    delta: Tensor,
    supported: Tensor,
    *,
    centerness_index: int,
    validate_values: bool = True,
) -> dict[str, Tensor]:
    """Compose ROSA quality residual with frozen native class identity."""

    if (
        baseline_classification.ndim != 3
        or baseline_quality.shape[:2] != baseline_classification.shape[:2]
        or delta.shape != baseline_classification.shape[:2] + (1,)
        or supported.shape != baseline_classification.shape[:2]
        or supported.dtype != torch.bool
    ):
        raise ValueError("quality-rank tensors are misaligned")
    native_probability = baseline_classification.detach().sigmoid()
    frozen_label = native_probability.argmax(dim=-1)
    selected_logit = torch.gather(
        baseline_classification.detach(), -1, frozen_label[..., None]
    ).squeeze(-1)
    selected_probability = torch.gather(
        native_probability, -1, frozen_label[..., None]
    ).squeeze(-1)
    centerness_logit = baseline_quality[..., int(centerness_index)].detach()
    centerness_probability = centerness_logit.sigmoid()
    native_joint_score = (selected_probability * centerness_probability).detach()
    native_joint_logit = stable_native_joint_logit(
        selected_logit,
        centerness_logit,
        validate_values=validate_values,
    )
    eligible = supported.detach() & torch.isfinite(delta.squeeze(-1))
    effective_delta = torch.where(
        eligible, delta.squeeze(-1), torch.zeros_like(delta.squeeze(-1))
    )
    joint_logit = native_joint_logit + effective_delta
    emitted = stable_joint_probability(
        joint_logit,
        native_joint_score,
        effective_delta,
        validate_values=validate_values,
    )
    return {
        "frozen_label": frozen_label,
        "native_selected_probability": selected_probability.detach(),
        "native_joint_score": native_joint_score,
        "native_joint_logit": native_joint_logit,
        "joint_logit": joint_logit,
        "effective_delta": effective_delta,
        "emitted_joint_score": emitted,
        "eligible": eligible,
    }


def compose_native_identity_rank(
    baseline_classification: Tensor,
    baseline_quality: Tensor,
    *,
    centerness_index: int,
) -> dict[str, Tensor]:
    """Build the three decode fields when score refinement is disabled."""

    if (
        baseline_classification.ndim != 3
        or baseline_quality.shape[:2] != baseline_classification.shape[:2]
    ):
        raise ValueError("native identity-rank tensors are misaligned")
    native_probability = baseline_classification.detach().sigmoid()
    frozen_label = native_probability.argmax(dim=-1)
    selected_probability = torch.gather(
        native_probability,
        -1,
        frozen_label[..., None],
    ).squeeze(-1)
    centerness_probability = baseline_quality[
        ..., int(centerness_index)
    ].detach().sigmoid()
    # The released zero-residual bridge promotes half/bfloat16 probabilities
    # to FP32 after forming the native low-precision product.  Preserve both
    # that arithmetic order and its public dtype without rebuilding the
    # log-sum-exp identity machinery.
    emitted = (selected_probability * centerness_probability).to(
        dtype=_probability_dtype(baseline_classification.dtype)
    ).detach()
    return {
        "frozen_label": frozen_label,
        "native_selected_probability": selected_probability.detach(),
        "emitted_joint_score": emitted,
    }


def _compose_native_identity_rank_and_selection(
    baseline_classification: Tensor,
    baseline_quality: Tensor,
    *,
    centerness_index: int,
) -> tuple[dict[str, Tensor], Tensor]:
    """Build exact native rank fields and the pre-refinement selection score.

    The released selection uses ``amax`` while identity ranking uses
    ``argmax`` followed by ``gather``. Those reductions differ in NaN bit
    patterns, so retain both values while sharing their sigmoid inputs.
    """

    if (
        baseline_classification.ndim != 3
        or baseline_quality.shape[:2] != baseline_classification.shape[:2]
    ):
        raise ValueError("native identity-rank tensors are misaligned")
    native_probability = baseline_classification.detach().sigmoid()
    selection_probability = native_probability.amax(dim=-1)
    selected_probability, frozen_label = native_probability.max(dim=-1)
    centerness_probability = (
        baseline_quality[..., int(centerness_index)].detach().sigmoid()
    )
    selection_score = selection_probability * centerness_probability
    low_precision_product = selected_probability * centerness_probability
    emitted = low_precision_product.to(
        dtype=_probability_dtype(baseline_classification.dtype)
    ).detach()
    return (
        {
            "frozen_label": frozen_label,
            "native_selected_probability": selected_probability.detach(),
            "emitted_joint_score": emitted,
        },
        selection_score,
    )


def _can_reuse_native_identity_rank_selection(
    baseline_classification: Tensor,
    baseline_quality: Tensor,
    *,
    centerness_index: int,
) -> bool:
    """Seal shared native-rank computation to plain eager inference."""

    tensors = (baseline_classification, baseline_quality)
    return bool(
        compose_native_identity_rank is _RELEASED_COMPOSE_NATIVE_IDENTITY_RANK
        and _compose_native_identity_rank_and_selection
        is _RELEASED_COMPOSE_NATIVE_IDENTITY_RANK_AND_SELECTION
        and _probability_dtype is _RELEASED_PROBABILITY_DTYPE
        and _rfl._plain_eager_no_grad_execution is _RELEASED_PLAIN_EAGER_NO_GRAD
        and _rfl._plain_eager_no_grad_execution()
        and torch.gather is _RELEASED_TORCH_GATHER
        and torch.Tensor.detach is _RELEASED_TENSOR_DETACH
        and torch.Tensor.sigmoid is _RELEASED_TENSOR_SIGMOID
        and torch.Tensor.amax is _RELEASED_TENSOR_AMAX
        and torch.Tensor.argmax is _RELEASED_TENSOR_ARGMAX
        and torch.Tensor.max is _RELEASED_TENSOR_MAX
        and torch.Tensor.squeeze is _RELEASED_TENSOR_SQUEEZE
        and torch.Tensor.to is _RELEASED_TENSOR_TO
        and torch.Tensor.__mul__ is _RELEASED_TENSOR_MULTIPLY
        and torch.Tensor.__getitem__ is _RELEASED_TENSOR_GETITEM
        and type(centerness_index) is int
        and type(baseline_classification) is torch.Tensor
        and type(baseline_quality) is torch.Tensor
        and baseline_classification.ndim == 3
        and baseline_quality.ndim == 3
        and baseline_quality.shape[:2] == baseline_classification.shape[:2]
        and 0 <= centerness_index < baseline_quality.shape[-1]
        and all(
            value.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
            and value.device == baseline_classification.device
            and value.device.type in ("cpu", "cuda")
            and value.layout == torch.strided
            and not value.is_quantized
            and not value.requires_grad
            and value.is_contiguous()
            and all(name is None for name in value.names)
            for value in tensors
        )
    )


def ddp_mean(value: Tensor) -> Tensor:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return value
    reduced = value.clone()
    torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
    return reduced / torch.distributed.get_world_size()


def ddp_mean_mask_count(mask: Tensor, dtype: torch.dtype) -> Tensor:
    if mask.dtype != torch.bool:
        raise TypeError("DDP count mask must be boolean")
    return ddp_mean(mask.sum().to(dtype=dtype))


@dataclass
class JointScoreLoss:
    loss: Tensor
    probability: Tensor
    positive_count: Tensor
    negative_count: Tensor


def jointscore_quality_focal_loss(
    joint_logit: Tensor,
    effective_delta: Tensor,
    native_joint_score: Tensor,
    targets: Tensor,
    eligible: Tensor,
    *,
    gamma: float,
) -> JointScoreLoss:
    """Quality-focal loss on the exact score emitted during inference."""

    if (
        joint_logit.ndim != 2
        or any(
            value.shape != joint_logit.shape
            for value in (
                effective_delta,
                native_joint_score,
                targets,
                eligible,
            )
        )
        or eligible.dtype != torch.bool
        or not math.isfinite(float(gamma))
        or float(gamma) < 0
    ):
        raise ValueError("joint-score loss tensors or gamma are invalid")
    dtype = _probability_dtype(joint_logit.dtype)
    logits = joint_logit.to(dtype=dtype)
    probability = stable_joint_probability(logits, native_joint_score, effective_delta)
    targets = targets.detach().to(dtype=probability.dtype)
    eligible = eligible.detach()
    if bool((eligible & ~torch.isfinite(targets)).any()):
        raise FloatingPointError("eligible quality target is non-finite")
    if bool((eligible & ((targets < 0) | (targets > 1))).any()):
        raise ValueError("quality target must be in [0,1]")
    positive = eligible & targets.gt(0)
    negative = eligible & targets.eq(0)
    positive_count = ddp_mean_mask_count(positive, probability.dtype)
    negative_count = ddp_mean_mask_count(negative, probability.dtype)
    per_query = (targets - probability).abs().pow(
        float(gamma)
    ) * F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    positive_mean = (
        per_query * positive.to(per_query.dtype)
    ).sum() / positive_count.clamp_min(1)
    negative_mean = (
        per_query * negative.to(per_query.dtype)
    ).sum() / negative_count.clamp_min(1)
    positive_active = bool(positive_count.detach().gt(0).item())
    negative_active = bool(negative_count.detach().gt(0).item())
    if positive_active and negative_active:
        loss = (positive_mean + negative_mean) / 2
    elif positive_active:
        loss = positive_mean
    elif negative_active:
        loss = negative_mean
    else:
        loss = effective_delta.sum().to(dtype=dtype) * 0
    return JointScoreLoss(
        loss=loss,
        probability=probability,
        positive_count=positive_count,
        negative_count=negative_count,
    )


def task_box_loss(
    loss_reg,
    final_reg: Tensor,
    reg_target: Tensor,
    weighted_reg: Tensor,
    selected: Tensor,
    task_slice: slice,
) -> Tensor:
    """Apply Sparse4D's native box loss to one state slice."""

    average = ddp_mean_mask_count(selected, final_reg.dtype).clamp_min(1)
    mask = selected.reshape(-1)
    flat_reg = final_reg.flatten(end_dim=1)[mask]
    if flat_reg.shape[0] == 0:
        return final_reg[..., task_slice].sum() * 0
    flat_target = reg_target.flatten(end_dim=1)[mask]
    flat_target = torch.where(flat_target.isnan(), flat_reg.new_zeros(()), flat_target)
    dimension_mask = torch.zeros_like(weighted_reg)
    dimension_mask[..., task_slice] = 1
    flat_weight = (weighted_reg * dimension_mask).flatten(end_dim=1)[mask]
    return loss_reg.loss_box(
        flat_reg,
        flat_target,
        weight=flat_weight,
        avg_factor=average,
    )


def decode_frozen_jointscore_rank(
    decoder,
    classification: Sequence[Tensor],
    predictions: Sequence[Tensor],
    frozen_label: Tensor,
    native_selected_probability: Tensor,
    emitted_joint_score: Tensor,
    *,
    instance_id: Tensor | None,
    output_idx: int,
    validate_values: bool = True,
    pack_host_transfers: bool = False,
) -> list[dict]:
    """Decode once from the frozen native label and ROSA joint score."""

    if getattr(decoder, "selection_score_mode", "legacy") != "legacy":
        raise ValueError("frozen-label decode requires legacy selection")
    if not bool(getattr(decoder, "sorted", False)):
        raise RuntimeError("exact native identity requires sorted top-k")
    output_idx = int(output_idx)
    if not 0 <= output_idx < len(predictions):
        raise IndexError("output_idx is outside the prediction list")
    final_classification = classification[output_idx]
    box_predictions = predictions[output_idx]
    expected = final_classification.shape[:2]
    if (
        final_classification.ndim != 3
        or frozen_label.shape != expected
        or frozen_label.dtype != torch.long
        or native_selected_probability.shape != expected
        or emitted_joint_score.shape != expected
        or box_predictions.shape[:2] != expected
    ):
        raise ValueError("frozen-label decoder tensors are misaligned")
    if int(decoder.num_output) != int(expected[1]):
        raise RuntimeError("decoder must emit exactly one box per query")
    threshold = getattr(decoder, "score_threshold", None)
    if threshold is not None and float(threshold) > 0:
        raise RuntimeError("positive score-threshold filtering is forbidden")
    if instance_id is None or instance_id.shape != expected:
        raise RuntimeError("decode requires aligned non-null instance IDs")
    if not isinstance(validate_values, bool):
        raise TypeError("validate_values must be a bool")
    if not isinstance(pack_host_transfers, bool):
        raise TypeError("pack_host_transfers must be a bool")
    if validate_values and not all(
        bool(torch.isfinite(value).all())
        for value in (
            final_classification,
            native_selected_probability,
            emitted_joint_score,
        )
    ):
        raise FloatingPointError("decoder received non-finite scores")

    batch_size, num_query = emitted_joint_score.shape
    class_origin, indices = native_selected_probability.topk(
        num_query, dim=1, sorted=bool(decoder.sorted)
    )
    labels = torch.gather(frozen_label, 1, indices)
    scores = torch.gather(emitted_joint_score, 1, indices)
    scores, order = torch.sort(scores, dim=1, descending=True)
    indices = torch.gather(indices, 1, order)
    labels = torch.gather(labels, 1, order)
    class_origin = torch.gather(class_origin, 1, order)

    decoded_per_batch = None
    floating_inputs = (box_predictions, scores, class_origin)
    integer_inputs = (labels, indices, instance_id)
    can_pack = bool(
        pack_host_transfers
        and not validate_values
        and not torch.is_grad_enabled()
        and batch_size > 0
        and all(
            type(value) is torch.Tensor
            for value in floating_inputs + integer_inputs
        )
        and all(
            value.layout == torch.strided
            for value in floating_inputs + integer_inputs
        )
        and all(
            not value.is_quantized
            for value in floating_inputs + integer_inputs
        )
        and all(
            value.device.type == "cuda"
            for value in floating_inputs + integer_inputs
        )
        and all(
            value.device == box_predictions.device
            for value in floating_inputs + integer_inputs
        )
        and all(value.dtype == box_predictions.dtype for value in floating_inputs)
        and all(value.dtype == torch.long for value in integer_inputs)
        and all(
            all(name is None for name in value.names)
            for value in floating_inputs + integer_inputs
        )
    )
    if (
        can_pack
        and _rfl._plain_eager_no_grad_execution()
        and _matches_released_sparse4d_decoder(decoder)
        and box_predictions.shape == (batch_size, num_query, 11)
        and box_predictions.dtype
        in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    ):
        decoded_boxes = _decode_frozen_box_batch(
            decoder,
            box_predictions,
            indices,
        )
        if (
            type(decoded_boxes) is torch.Tensor
            and decoded_boxes.layout == torch.strided
            and not decoded_boxes.is_quantized
            and decoded_boxes.device == box_predictions.device
            and decoded_boxes.dtype == box_predictions.dtype
            and all(name is None for name in decoded_boxes.names)
        ):
            selected_instance_ids = torch.gather(instance_id, 1, indices)
            return _pack_frozen_decode_fields(
                decoded_boxes,
                scores,
                labels,
                indices,
                class_origin,
                selected_instance_ids,
            )
    if can_pack and _matches_released_sparse4d_decoder(decoder):
        decoded_per_batch = [
            decoder.decode_box(box_predictions[index, indices[index]])
            for index in range(batch_size)
        ]
        decoded_boxes = torch.stack(decoded_per_batch, dim=0)
        if (
            type(decoded_boxes) is torch.Tensor
            and decoded_boxes.layout == torch.strided
            and not decoded_boxes.is_quantized
            and decoded_boxes.device == box_predictions.device
            and decoded_boxes.dtype == box_predictions.dtype
            and all(name is None for name in decoded_boxes.names)
        ):
            selected_instance_ids = torch.gather(instance_id, 1, indices)
            return _pack_frozen_decode_fields(
                decoded_boxes,
                scores,
                labels,
                indices,
                class_origin,
                selected_instance_ids,
            )

    output = []
    for batch_index in range(batch_size):
        query_indices = indices[batch_index]
        decoded = (
            decoded_per_batch[batch_index]
            if decoded_per_batch is not None
            else decoder.decode_box(box_predictions[batch_index, query_indices])
        )
        item = {
            "boxes_3d": decoded.cpu(),
            "scores_3d": scores[batch_index].cpu(),
            "labels_3d": labels[batch_index].cpu(),
            "query_indices": query_indices.cpu(),
            "cls_scores": class_origin[batch_index].cpu(),
            "instance_ids": instance_id[batch_index, query_indices].cpu(),
        }
        if any(
            item[name].numel() != num_query
            for name in ("labels_3d", "query_indices", "instance_ids")
        ):
            raise RuntimeError("frozen-label decoder dropped a query")
        output.append(item)
    return output


_RELEASED_COMPOSE_NATIVE_IDENTITY_RANK = compose_native_identity_rank
_RELEASED_COMPOSE_NATIVE_IDENTITY_RANK_AND_SELECTION = (
    _compose_native_identity_rank_and_selection
)
_RELEASED_PROBABILITY_DTYPE = _probability_dtype
_RELEASED_PLAIN_EAGER_NO_GRAD = _rfl._plain_eager_no_grad_execution
_RELEASED_TORCH_GATHER = torch.gather
_RELEASED_TENSOR_DETACH = torch.Tensor.detach
_RELEASED_TENSOR_SIGMOID = torch.Tensor.sigmoid
_RELEASED_TENSOR_AMAX = torch.Tensor.amax
_RELEASED_TENSOR_ARGMAX = torch.Tensor.argmax
_RELEASED_TENSOR_MAX = torch.Tensor.max
_RELEASED_TENSOR_SQUEEZE = torch.Tensor.squeeze
_RELEASED_TENSOR_TO = torch.Tensor.to
_RELEASED_TENSOR_MULTIPLY = torch.Tensor.__mul__
_RELEASED_TENSOR_GETITEM = torch.Tensor.__getitem__


__all__ = [
    "JointScoreLoss",
    "compose_native_identity_rank",
    "compose_quality_rank",
    "ddp_mean_mask_count",
    "decode_frozen_jointscore_rank",
    "jointscore_quality_focal_loss",
    "stable_joint_probability",
    "stable_native_joint_logit",
    "task_box_loss",
]
