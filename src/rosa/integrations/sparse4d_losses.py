"""Sparse4D matcher and box-loss reuse for the released ROSA path."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from .sparse4d_scores import (
    ddp_mean_mask_count,
    jointscore_quality_focal_loss,
    task_box_loss,
)

_SIZE = slice(3, 6)
_YAW = slice(6, 8)
_BEV_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)


def _radial_basis(center_xy: Tensor) -> tuple[Tensor, Tensor]:
    radial = F.normalize(center_xy, dim=-1, eps=1e-6)
    tangent = torch.stack((-radial[..., 1], radial[..., 0]), dim=-1)
    return radial, tangent


def _selected_mean(value: Tensor, mask: Tensor) -> Tensor:
    count = ddp_mean_mask_count(mask, value.dtype).clamp_min(1)
    return (value * mask.to(dtype=value.dtype)).sum() / count


def _task_loss(
    head,
    final_reg: Tensor,
    reg_target: Tensor,
    weighted_reg: Tensor,
    selected: Tensor,
    task_slice: slice,
) -> Tensor:
    return task_box_loss(
        head.loss_reg,
        final_reg,
        reg_target,
        weighted_reg,
        selected,
        task_slice,
    )


def rosa_losses(head, record: dict, data: dict) -> dict:
    cfg = head.rosa_cfg
    baseline_cls = record["baseline_classification"]
    full_baseline_reg = record["baseline_anchor"][..., : len(head.reg_weights)]
    full_final_reg = record["final_anchor"][..., : len(head.reg_weights)]
    with torch.no_grad():
        cls_target, reg_target, reg_weights = head.sampler.sample(
            baseline_cls,
            full_baseline_reg,
            data[head.gt_cls_key],
            data[head.gt_reg_key],
        )
        reg_target = reg_target[..., : len(head.reg_weights)].detach()
        reg_weights = reg_weights[..., : len(head.reg_weights)].detach()
        matched_full = (
            cls_target.ge(0)
            & cls_target.lt(int(baseline_cls.shape[-1]))
            & torch.isfinite(reg_target[..., :8]).all(dim=-1)
        ).detach()
        safe_target_full = torch.where(
            torch.isfinite(reg_target),
            reg_target,
            full_baseline_reg.detach(),
        )
        selected_indices = record.get("selected_indices")
        if selected_indices is None:
            baseline_reg = full_baseline_reg
            final_reg = full_final_reg
            safe_target = safe_target_full
            matched = matched_full
        else:
            reg_index = selected_indices[..., None].expand(
                -1, -1, full_baseline_reg.shape[-1]
            )
            baseline_reg = full_baseline_reg.gather(1, reg_index)
            final_reg = full_final_reg.gather(1, reg_index)
            safe_target = safe_target_full.gather(1, reg_index)
            reg_target = reg_target.gather(1, reg_index)
            reg_weights = reg_weights.gather(1, reg_index)
            matched = matched_full.gather(1, selected_indices)
        matched = (matched & record["supported"]).detach()
        initial_error = safe_target[..., :2] - baseline_reg[..., :2]
        initial_distance = torch.linalg.vector_norm(initial_error, dim=-1)
        keep = (matched & initial_distance.lt(cfg.keep_threshold_m)).detach()
        move = (matched & ~keep).detach()

    weighted = (reg_weights * baseline_reg.new_tensor(head.reg_weights)).detach()
    step_bases = record["iteration_base_anchors"][..., : len(head.reg_weights)]
    step_outputs = record["iteration_anchors"][..., : len(head.reg_weights)]
    num_steps = int(step_outputs.shape[2])
    if num_steps != cfg.num_refinement_steps:
        raise RuntimeError("iterative refinement record has wrong length")
    step_weights = final_reg.new_tensor(
        [
            cfg.intermediate_step_loss_weight if index < num_steps - 1 else 1.0
            for index in range(num_steps)
        ]
    )
    step_weights = step_weights / step_weights.sum()

    zero = final_reg.sum() * 0
    size_loss = zero
    yaw_loss = zero
    distribution_loss = zero
    candidate_loss = zero
    radial_offset_loss = zero
    tangent_loss = zero
    for step in range(num_steps):
        weight = step_weights[step]
        step_base = step_bases[:, :, step]
        step_output = step_outputs[:, :, step]
        size_loss = size_loss + weight * _task_loss(
            head, step_output, reg_target, weighted, matched, _SIZE
        )
        yaw_loss = yaw_loss + weight * _task_loss(
            head, step_output, reg_target, weighted, matched, _YAW
        )
        with torch.no_grad():
            radial, tangent = _radial_basis(step_base[..., :2])
            error_xy = safe_target[..., :2] - step_base[..., :2]
            radial_target = (error_xy * radial).sum(dim=-1)
            tangent_target = (error_xy * tangent).sum(dim=-1)
            target_range = torch.linalg.vector_norm(
                safe_target[..., :2], dim=-1
            ).clamp_min(cfg.min_candidate_range_m)
            base_range = torch.linalg.vector_norm(step_base[..., :2], dim=-1).clamp_min(
                cfg.min_candidate_range_m
            )
            log_range_target = torch.log(target_range / base_range)

        mean = record["iteration_distribution_mean"][:, :, step]
        std = record["iteration_distribution_std"][:, :, step]
        nll = F.gaussian_nll_loss(
            mean,
            log_range_target,
            std.square().clamp_min(1e-8),
            full=True,
            reduction="none",
        )
        distribution_loss = distribution_loss + weight * _selected_mean(nll, matched)
        offsets = record["iteration_candidate_offsets"][:, :, step]
        supported = record["iteration_candidate_supported"][:, :, step]
        probabilities = record["iteration_candidate_probabilities"][:, :, step]
        with torch.no_grad():
            spacing = (
                (offsets[..., 1:] - offsets[..., :-1])
                .abs()
                .mean(dim=-1)
                .clamp_min(0.05)
            )
            soft_target = torch.softmax(
                -(offsets.detach() - radial_target[..., None]).square()
                / (2 * spacing[..., None].square()),
                dim=-1,
            )
            target_valid = supported & matched[..., None]
            target_distribution = soft_target * target_valid.to(dtype=soft_target.dtype)
            target_distribution = target_distribution / (
                target_distribution.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            )
        per_query = -(target_distribution * probabilities.clamp_min(1e-8).log()).sum(
            dim=-1
        )
        candidate_mask = matched & supported.any(dim=-1)
        if step > 0:
            candidate_mask = torch.zeros_like(candidate_mask)
        candidate_loss = candidate_loss + weight * _selected_mean(
            per_query, candidate_mask
        )
        radial_offset_loss = radial_offset_loss + weight * _selected_mean(
            F.smooth_l1_loss(
                record["iteration_radial_safe"][:, :, step],
                radial_target,
                reduction="none",
                beta=0.1,
            ),
            matched,
        )
        tangent_loss = tangent_loss + weight * _selected_mean(
            F.smooth_l1_loss(
                record["iteration_tangent_delta"][:, :, step],
                tangent_target,
                reduction="none",
                beta=0.1,
            ),
            matched,
        )

    final_error = F.smooth_l1_loss(
        final_reg[..., :2],
        safe_target[..., :2],
        reduction="none",
        beta=0.1,
    ).sum(dim=-1)
    center_move_loss = _selected_mean(final_error, move)
    center_keep_loss = _selected_mean(
        torch.linalg.vector_norm(final_reg[..., :2] - baseline_reg[..., :2], dim=-1),
        keep,
    )
    with torch.no_grad():
        final_distance = torch.linalg.vector_norm(
            full_final_reg[..., :2] - safe_target_full[..., :2],
            dim=-1,
        )
        quality_target = (
            torch.stack(
                [final_distance.le(threshold) for threshold in _BEV_THRESHOLDS],
                dim=-1,
            )
            .float()
            .mean(dim=-1)
        )
        quality_target = torch.where(
            matched_full, quality_target, torch.zeros_like(quality_target)
        )
    quality = jointscore_quality_focal_loss(
        record["joint_logit"],
        record["effective_jointscore_delta"],
        record["native_joint_score"],
        quality_target,
        record["jointscore_eligible"].detach(),
        gamma=float(head.sampler.gamma),
    )
    return {
        "loss_cl_shape_size_6": size_loss * cfg.shape_loss_weight,
        "loss_cl_shape_yaw_6": yaw_loss * cfg.shape_loss_weight,
        "loss_cl_range_distribution_6": (
            distribution_loss * cfg.distribution_nll_weight
        ),
        "loss_cl_candidate_soft_6": (candidate_loss * cfg.candidate_loss_weight),
        "loss_cl_radial_offset_6": (radial_offset_loss * cfg.radial_offset_loss_weight),
        "loss_cl_center_move_6": (center_move_loss * cfg.center_move_loss_weight),
        "loss_cl_center_keep_6": (center_keep_loss * cfg.center_keep_loss_weight),
        "loss_cl_tangent_6": tangent_loss * cfg.tangent_loss_weight,
        "loss_cl_quality_6": quality.loss * cfg.quality_loss_weight,
        "cl_matched_count": ddp_mean_mask_count(matched, final_reg.dtype).detach(),
        "cl_keep_count": ddp_mean_mask_count(keep, final_reg.dtype).detach(),
        "cl_move_count": ddp_mean_mask_count(move, final_reg.dtype).detach(),
        "cl_radial_gate_mean": record["radial_gate"].detach().mean(),
        "cl_radial_abs_mean": record["radial_safe"].detach().abs().mean(),
        "cl_tangent_abs_mean": (record["tangent_delta"].detach().abs().mean()),
        "cl_posterior_entropy_mean": (record["posterior_entropy"].detach().mean()),
        "cl_range_distribution_mean_abs": (
            record["distribution_mean"].detach().abs().mean()
        ),
        "cl_range_distribution_std_mean": (record["distribution_std"].detach().mean()),
        "cl_quality_delta_abs_mean": (record["quality_delta"].detach().abs().mean()),
    }


__all__ = ["rosa_losses"]
