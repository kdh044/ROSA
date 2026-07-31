"""Score, loss, and decode helpers retained by the Sparse4D binding."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor
import torch.nn.functional as F


def _probability_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def stable_native_joint_logit(
    selected_class_logit: Tensor,
    centerness_logit: Tensor,
) -> Tensor:
    """Compute ``logit(sigmoid(class) * sigmoid(centerness))`` stably."""

    if (
        selected_class_logit.shape != centerness_logit.shape
        or not selected_class_logit.is_floating_point()
        or not bool(torch.isfinite(selected_class_logit).all())
        or not bool(torch.isfinite(centerness_logit).all())
    ):
        raise ValueError("native class/centerness logits must be finite/aligned")
    dtype = _probability_dtype(selected_class_logit.dtype)
    selected = selected_class_logit.detach().to(dtype=dtype)
    centerness = centerness_logit.detach().to(dtype=dtype)
    result = -torch.logsumexp(
        torch.stack((-selected, -centerness, -selected - centerness), dim=-1),
        dim=-1,
    )
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("native joint logit is non-finite")
    return result.detach()


def stable_joint_probability(
    joint_logit: Tensor,
    native_joint_score: Tensor,
    effective_delta: Tensor,
) -> Tensor:
    """Emit sigmoid score with a bit-exact zero-residual identity bridge."""

    if (
        joint_logit.ndim != 2
        or native_joint_score.shape != joint_logit.shape
        or effective_delta.shape != joint_logit.shape
    ):
        raise ValueError("joint score tensors are misaligned")
    dtype = _probability_dtype(joint_logit.dtype)
    raw = joint_logit.to(dtype=dtype).sigmoid()
    native = native_joint_score.detach().to(dtype=dtype)
    delta = effective_delta.to(dtype=dtype)
    exact_identity = native + (raw - raw.detach())
    probability = torch.where(delta.eq(0), exact_identity, raw)
    if not bool(torch.isfinite(probability).all()):
        raise FloatingPointError("joint probability is non-finite")
    return probability


def compose_quality_rank(
    baseline_classification: Tensor,
    baseline_quality: Tensor,
    delta: Tensor,
    supported: Tensor,
    *,
    centerness_index: int,
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
    native_joint_logit = stable_native_joint_logit(selected_logit, centerness_logit)
    eligible = supported.detach() & torch.isfinite(delta.squeeze(-1))
    effective_delta = torch.where(
        eligible, delta.squeeze(-1), torch.zeros_like(delta.squeeze(-1))
    )
    joint_logit = native_joint_logit + effective_delta
    emitted = stable_joint_probability(joint_logit, native_joint_score, effective_delta)
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
    if not all(
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
    output = []
    for batch_index in range(batch_size):
        query_indices = indices[batch_index]
        item = {
            "boxes_3d": decoder.decode_box(
                box_predictions[batch_index, query_indices]
            ).cpu(),
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


__all__ = [
    "JointScoreLoss",
    "compose_quality_rank",
    "ddp_mean_mask_count",
    "decode_frozen_jointscore_rank",
    "jointscore_quality_focal_loss",
    "stable_joint_probability",
    "stable_native_joint_logit",
    "task_box_loss",
]
