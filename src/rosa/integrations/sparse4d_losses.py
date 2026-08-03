"""Sparse4D matcher and box-loss reuse for the released ROSA path."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from ..attention import masked_softmax
from ..ops import wrapped_angle_difference
from .fisheye3dod.constants import OFFICIAL_AP_DISTANCES
from .sparse4d_scores import (
    ddp_mean_mask_count,
    jointscore_quality_focal_loss,
    task_box_loss,
)

_SIZE = slice(3, 6)
_YAW = slice(6, 8)
_BEV_THRESHOLDS = OFFICIAL_AP_DISTANCES
_CENTERNESS_TARGET_MODES = frozenset(("exp_bev", "strict_ap_fraction"))


def validate_centerness_target_mode(mode: str) -> str:
    """Validate the opt-in native centerness supervision contract."""

    mode = str(mode)
    if mode not in _CENTERNESS_TARGET_MODES:
        raise ValueError(
            "centerness_target_mode must be one of "
            f"{sorted(_CENTERNESS_TARGET_MODES)}, got {mode!r}"
        )
    return mode


def evaluator_aligned_centerness_target(
    prediction: Tensor,
    target: Tensor,
    *,
    mode: str = "exp_bev",
) -> Tensor:
    """Return a detached score target based on the evaluator's BEV distance."""

    if (
        prediction.shape != target.shape
        or prediction.shape[-1] < 2
        or not prediction.is_floating_point()
    ):
        raise ValueError("centerness prediction/target boxes must be aligned")
    distance = torch.linalg.vector_norm(
        target[..., :2] - prediction[..., :2],
        dim=-1,
    )
    mode = validate_centerness_target_mode(mode)
    if mode == "exp_bev":
        return torch.exp(-distance).detach()
    return (
        torch.stack(
            [distance.lt(threshold) for threshold in _BEV_THRESHOLDS],
            dim=-1,
        )
        .to(dtype=prediction.dtype)
        .mean(dim=-1)
        .detach()
    )


def evaluator_aligned_quality_target(
    prediction: Tensor,
    target: Tensor,
    matched: Tensor,
    predicted_class: Tensor,
    target_class: Tensor,
) -> Tensor:
    """Score label-correct boxes by the official strict AP distance tests."""

    expected = prediction.shape[:-1]
    if (
        prediction.shape != target.shape
        or prediction.shape[-1] < 2
        or not prediction.is_floating_point()
        or matched.shape != expected
        or matched.dtype != torch.bool
        or predicted_class.shape != expected
        or target_class.shape != expected
    ):
        raise ValueError("quality-target tensors must be aligned")
    distance = torch.linalg.vector_norm(
        target[..., :2] - prediction[..., :2],
        dim=-1,
    )
    localization = (
        torch.stack(
            [distance.lt(threshold) for threshold in _BEV_THRESHOLDS],
            dim=-1,
        )
        .to(dtype=prediction.dtype)
        .mean(dim=-1)
    )
    valid = matched & predicted_class.eq(target_class)
    return torch.where(valid, localization, torch.zeros_like(localization)).detach()


def _radial_basis(center_xy: Tensor) -> tuple[Tensor, Tensor]:
    radial = F.normalize(center_xy, dim=-1, eps=1e-6)
    tangent = torch.stack((-radial[..., 1], radial[..., 0]), dim=-1)
    return radial, tangent


def _selected_mean(value: Tensor, mask: Tensor) -> Tensor:
    count = ddp_mean_mask_count(mask, value.dtype).clamp_min(1)
    return (value * mask.to(dtype=value.dtype)).sum() / count


def _yaw_selected_mean(
    value: Tensor,
    mask: Tensor,
    class_weight: Tensor | None,
) -> Tensor:
    """Reduce a yaw-only loss without changing its matched-count denominator."""

    if class_weight is None:
        return _selected_mean(value, mask)
    if class_weight.shape != mask.shape:
        raise ValueError("yaw class weights must align with the query mask")
    count = ddp_mean_mask_count(mask, value.dtype).clamp_min(1)
    return (
        value * class_weight.to(dtype=value.dtype) * mask.to(dtype=value.dtype)
    ).sum() / count


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


def _circular_yaw_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    beta: float,
    class_weight: Tensor | None = None,
) -> Tensor:
    error = wrapped_angle_difference(
        prediction[..., _YAW],
        target[..., _YAW],
    )
    per_query = F.smooth_l1_loss(
        error,
        torch.zeros_like(error),
        reduction="none",
        beta=beta,
    )
    return _yaw_selected_mean(per_query, mask, class_weight)


def _sequential_center_targets(
    step_base: Tensor,
    target_xy: Tensor,
    tangent_delta: Tensor,
    *,
    min_range: float,
    max_range: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Targets in the same tangent-then-radial frame used by forward."""

    _, tangent = _radial_basis(step_base[..., :2])
    error_xy = target_xy - step_base[..., :2]
    tangent_target = (error_xy * tangent).sum(dim=-1)
    tangent_anchor = (
        step_base[..., :2]
        + tangent * tangent_delta.detach()[..., None]
    )
    radial, _ = _radial_basis(tangent_anchor)
    radial_target = ((target_xy - tangent_anchor) * radial).sum(dim=-1)
    base_range = torch.linalg.vector_norm(
        tangent_anchor,
        dim=-1,
    ).clamp_min(min_range)
    target_range = (base_range + radial_target).clamp(
        min=min_range,
        max=max_range,
    )
    log_range_target = torch.log(target_range / base_range)
    return radial_target, tangent_target, log_range_target


def _fixed_base_center_targets(
    step_base: Tensor,
    target_xy: Tensor,
    *,
    min_range: float,
    max_range: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Project XY supervision onto one fixed orthonormal BEV frame."""

    radial, tangent = _radial_basis(step_base[..., :2])
    error_xy = target_xy - step_base[..., :2]
    radial_target = (error_xy * radial).sum(dim=-1)
    tangent_target = (error_xy * tangent).sum(dim=-1)
    base_range = torch.linalg.vector_norm(step_base[..., :2], dim=-1).clamp_min(
        min_range
    )
    target_radial_coordinate = (base_range + radial_target).clamp(
        min=min_range,
        max=max_range,
    )
    log_range_target = torch.log(target_radial_coordinate / base_range)
    return radial_target, tangent_target, log_range_target


def _candidate_target_distribution(
    offsets: Tensor,
    radial_target: Tensor,
    sigma: Tensor,
    supported: Tensor,
    selected: Tensor,
    *,
    use_masked_softmax: bool,
) -> Tensor:
    """Build a normalized target over candidates visible to the query."""

    logits = (
        -(offsets.detach() - radial_target[..., None]).square()
        / (2 * sigma[..., None].square())
    )
    valid = supported & selected[..., None]
    if use_masked_softmax:
        return masked_softmax(logits, valid, dim=-1)

    # Published training normalized after masking; retain it only for exact
    # release reproduction.
    distribution = torch.softmax(logits, dim=-1) * valid.to(logits.dtype)
    return distribution / distribution.sum(dim=-1, keepdim=True).clamp_min(1e-8)


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
            matched_base = matched_full
            selected_cls_target = cls_target
        else:
            reg_index = selected_indices[..., None].expand(
                -1, -1, full_baseline_reg.shape[-1]
            )
            baseline_reg = full_baseline_reg.gather(1, reg_index)
            final_reg = full_final_reg.gather(1, reg_index)
            safe_target = safe_target_full.gather(1, reg_index)
            reg_target = reg_target.gather(1, reg_index)
            reg_weights = reg_weights.gather(1, reg_index)
            matched_base = matched_full.gather(1, selected_indices)
            selected_cls_target = cls_target.gather(1, selected_indices)
        matched_base = matched_base.detach()
        matched = (matched_base & record["supported"]).detach()
        initial_error = safe_target[..., :2] - baseline_reg[..., :2]
        initial_distance = torch.linalg.vector_norm(initial_error, dim=-1)
        keep = (matched & initial_distance.lt(cfg.keep_threshold_m)).detach()
        move = (matched & ~keep).detach()
        yaw_class_weight = None
        if cfg.yaw_class_loss_weights is not None:
            class_weights = baseline_reg.new_tensor(cfg.yaw_class_loss_weights)
            yaw_class_weight = class_weights[
                selected_cls_target.clamp(min=0, max=class_weights.numel() - 1)
            ].detach()

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
    yaw_keep_loss = zero
    yaw_improvement_loss = zero
    distribution_loss = zero
    candidate_loss = zero
    radial_offset_loss = zero
    center_gate_loss = zero
    tangent_loss = zero
    center_gate_supervised_count = zero.detach()
    center_gate_positive_count = zero.detach()
    for step in range(num_steps):
        weight = step_weights[step]
        step_base = step_bases[:, :, step]
        step_output = step_outputs[:, :, step]
        step_supported = record.get("iteration_observation_supported")
        step_supported = (
            step_supported[:, :, step]
            if step_supported is not None
            else record["supported"]
        )
        guarded_mask = (matched_base & step_supported).detach()
        size_mask = guarded_mask if cfg.mask_unsupported_updates else matched
        yaw_mask = (
            guarded_mask
            if (
                cfg.mask_unsupported_updates
                or cfg.yaw_update_mode == "guarded_shared"
            )
            else matched
        )
        center_mask = (
            guarded_mask
            if (
                cfg.mask_unsupported_updates
                or cfg.center_update_mode == "posterior_gate"
            )
            else matched
        )
        size_loss = size_loss + weight * _task_loss(
            head, step_output, reg_target, weighted, size_mask, _SIZE
        )
        yaw_active = (
            cfg.enable_shape_refinement
            and cfg.yaw_update_steps != "off"
            and (step == 0 or cfg.yaw_update_steps == "both")
        )
        if yaw_active:
            if cfg.yaw_update_mode == "legacy_shared":
                yaw_step_loss = _task_loss(
                    head,
                    step_output,
                    reg_target,
                    weighted,
                    yaw_mask,
                    _YAW,
                )
            else:
                yaw_step_loss = _circular_yaw_loss(
                    step_output,
                    safe_target,
                    yaw_mask,
                    cfg.yaw_loss_beta_rad,
                    yaw_class_weight,
                )
                base_yaw_error = wrapped_angle_difference(
                    step_base[..., _YAW],
                    safe_target[..., _YAW],
                ).abs()
                output_yaw_error = wrapped_angle_difference(
                    step_output[..., _YAW],
                    safe_target[..., _YAW],
                ).abs()
                applied_yaw = wrapped_angle_difference(
                    step_output[..., _YAW],
                    step_base[..., _YAW],
                )
                yaw_keep = (
                    yaw_mask
                    & base_yaw_error.lt(cfg.yaw_keep_threshold_rad)
                ).detach()
                yaw_keep_loss = yaw_keep_loss + weight * _yaw_selected_mean(
                    F.smooth_l1_loss(
                        applied_yaw,
                        torch.zeros_like(applied_yaw),
                        reduction="none",
                        beta=cfg.yaw_loss_beta_rad,
                    ),
                    yaw_keep,
                    yaw_class_weight,
                )
                yaw_improvement_loss = (
                    yaw_improvement_loss
                    + weight
                    * _yaw_selected_mean(
                        F.relu(output_yaw_error - base_yaw_error),
                        yaw_mask,
                        yaw_class_weight,
                    )
                )
            yaw_loss = yaw_loss + weight * yaw_step_loss

        with torch.no_grad():
            if cfg.center_target_mode == "fixed_base":
                (
                    radial_target,
                    tangent_target,
                    log_range_target,
                ) = _fixed_base_center_targets(
                    step_base,
                    safe_target[..., :2],
                    min_range=cfg.min_candidate_range_m,
                    max_range=cfg.max_candidate_range_m,
                )
            elif cfg.center_target_mode == "sequential":
                (
                    radial_target,
                    tangent_target,
                    log_range_target,
                ) = _sequential_center_targets(
                    step_base,
                    safe_target[..., :2],
                    record["iteration_tangent_delta"][:, :, step],
                    min_range=cfg.min_candidate_range_m,
                    max_range=cfg.max_candidate_range_m,
                )
            else:
                radial, tangent = _radial_basis(step_base[..., :2])
                error_xy = safe_target[..., :2] - step_base[..., :2]
                radial_target = (error_xy * radial).sum(dim=-1)
                tangent_target = (error_xy * tangent).sum(dim=-1)
                target_range = torch.linalg.vector_norm(
                    safe_target[..., :2], dim=-1
                ).clamp_min(cfg.min_candidate_range_m)
                base_range = torch.linalg.vector_norm(
                    step_base[..., :2], dim=-1
                ).clamp_min(cfg.min_candidate_range_m)
                log_range_target = torch.log(target_range / base_range)

        center_aux_mask = center_mask
        if cfg.center_supervision_mode == "move_or_keep":
            center_aux_mask = center_aux_mask & move
        radial_aux_active = (
            cfg.enable_radial or cfg.train_disabled_branch_aux
        )
        tangent_aux_active = (
            cfg.enable_tangent or cfg.train_disabled_branch_aux
        )

        mean = record["iteration_distribution_mean"][:, :, step]
        std = record["iteration_distribution_std"][:, :, step]
        nll = F.gaussian_nll_loss(
            mean,
            log_range_target,
            std.square().clamp_min(1e-8),
            full=True,
            reduction="none",
        )
        if radial_aux_active:
            distribution_loss = (
                distribution_loss
                + weight * _selected_mean(nll, center_aux_mask)
            )
        offsets = record["iteration_candidate_offsets"][:, :, step]
        supported = record["iteration_candidate_supported"][:, :, step]
        probabilities = record["iteration_candidate_probabilities"][:, :, step]
        with torch.no_grad():
            sigma = (
                (
                    (offsets[..., 1:] - offsets[..., :-1])
                    .abs()
                    .mean(dim=-1)
                    .clamp_min(0.05)
                )
                if cfg.candidate_target_mode == "legacy_spacing"
                else torch.full_like(
                    radial_target,
                    cfg.candidate_target_sigma_m,
                )
            )
            target_distribution = _candidate_target_distribution(
                offsets,
                radial_target,
                sigma,
                supported,
                center_aux_mask,
                use_masked_softmax=(
                    cfg.candidate_target_mode == "fixed_sigma"
                ),
            )
        per_query = -(target_distribution * probabilities.clamp_min(1e-8).log()).sum(
            dim=-1
        )
        candidate_mask = center_aux_mask & supported.any(dim=-1)
        if step > 0:
            candidate_mask = torch.zeros_like(candidate_mask)
        if radial_aux_active:
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
                center_aux_mask,
            )
            if cfg.center_update_mode == "posterior_gate":
                proposal = record["iteration_radial_proposal"][:, :, step]
                detached_proposal = proposal.detach()
                if cfg.center_target_mode == "fixed_base":
                    radial, tangent = _radial_basis(step_base[..., :2])
                    tangent_anchor_xy = (
                        step_base[..., :2]
                        + tangent
                        * record["iteration_tangent_delta"][:, :, step].detach()[
                            ..., None
                        ]
                    )
                    before_error = torch.linalg.vector_norm(
                        safe_target[..., :2] - tangent_anchor_xy,
                        dim=-1,
                    )
                    proposed_xy = (
                        tangent_anchor_xy + radial * detached_proposal[..., None]
                    )
                    after_error = torch.linalg.vector_norm(
                        safe_target[..., :2] - proposed_xy,
                        dim=-1,
                    )
                    gate_target = (
                        after_error + cfg.center_gate_margin_m < before_error
                    ).to(proposal.dtype)
                    # A zero-initialized posterior produces exactly zero
                    # proposals.  Supervising those as rejects closes the gate
                    # before its proposal branch has learned anything.
                    proposal_is_informative = detached_proposal.abs().ge(
                        cfg.center_gate_margin_m
                    )
                else:
                    gate_target = (
                        (radial_target - detached_proposal).abs()
                        + cfg.center_gate_margin_m
                        < radial_target.abs()
                    ).to(proposal.dtype)
                    proposal_is_informative = torch.ones_like(
                        gate_target, dtype=torch.bool
                    )
                gate_bce = F.binary_cross_entropy_with_logits(
                    record["iteration_radial_gate_logit"][:, :, step],
                    gate_target,
                    reduction="none",
                )
                gate_supervision_mask = (
                    center_aux_mask & supported.any(dim=-1) & proposal_is_informative
                )
                center_gate_loss = center_gate_loss + weight * _selected_mean(
                    gate_bce,
                    gate_supervision_mask,
                )
                with torch.no_grad():
                    center_gate_supervised_count = (
                        center_gate_supervised_count
                        + ddp_mean_mask_count(
                            gate_supervision_mask,
                            proposal.dtype,
                        )
                    )
                    center_gate_positive_count = (
                        center_gate_positive_count
                        + ddp_mean_mask_count(
                            gate_supervision_mask & gate_target.bool(),
                            proposal.dtype,
                        )
                    )
        if tangent_aux_active:
            tangent_loss = tangent_loss + weight * _selected_mean(
                F.smooth_l1_loss(
                    record["iteration_tangent_delta"][:, :, step],
                    tangent_target,
                    reduction="none",
                    beta=0.1,
                ),
                center_aux_mask,
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
        if cfg.architecture_profile == "legacy":
            quality_target = (
                torch.stack(
                    [
                        final_distance.le(threshold)
                        for threshold in _BEV_THRESHOLDS
                    ],
                    dim=-1,
                )
                .float()
                .mean(dim=-1)
            )
            quality_target = torch.where(
                matched_full,
                quality_target,
                torch.zeros_like(quality_target),
            )
        else:
            frozen_label = record.get("frozen_native_label")
            if frozen_label is None:
                frozen_label = baseline_cls.detach().argmax(dim=-1)
            quality_target = evaluator_aligned_quality_target(
                full_final_reg,
                safe_target_full,
                matched_full,
                frozen_label,
                cls_target,
            )
    quality = jointscore_quality_focal_loss(
        record["joint_logit"],
        record["effective_jointscore_delta"],
        record["native_joint_score"],
        quality_target,
        record["jointscore_eligible"].detach(),
        gamma=float(head.sampler.gamma),
    )
    with torch.no_grad():
        baseline_yaw_error = wrapped_angle_difference(
            baseline_reg[..., _YAW],
            safe_target[..., _YAW],
        ).abs()
        final_yaw_error = wrapped_angle_difference(
            final_reg[..., _YAW],
            safe_target[..., _YAW],
        ).abs()
        yaw_improved = final_yaw_error.lt(baseline_yaw_error)
        final_center_error = torch.linalg.vector_norm(
            final_reg[..., :2] - safe_target[..., :2],
            dim=-1,
        )
        center_improved = final_center_error.lt(initial_distance)
        center_harmed = final_center_error.gt(initial_distance)
        iteration_yaw_raw = record.get("iteration_yaw_raw_delta")
        iteration_yaw_gate = record.get("iteration_yaw_gate")
        iteration_yaw_delta = record.get("iteration_yaw_delta")
        iteration_support = record.get("iteration_observation_supported")
        if iteration_yaw_raw is None:
            iteration_yaw_raw = final_reg.new_zeros(
                *final_reg.shape[:2],
                num_steps,
                1,
            )
            iteration_yaw_gate = torch.zeros_like(iteration_yaw_raw)
            iteration_yaw_delta = torch.zeros_like(iteration_yaw_raw)
            iteration_support = matched[..., None].expand(-1, -1, num_steps)
        yaw_limits = final_reg.new_tensor(
            [
                cfg.max_yaw_delta_rad
                if index == 0
                else cfg.max_second_yaw_delta_rad
                for index in range(num_steps)
            ]
        )
        yaw_saturated = (
            iteration_yaw_raw.squeeze(-1).abs()
            >= 0.95 * yaw_limits[None, None, :]
        )
        yaw_diagnostic_mask = (
            matched_base[..., None] & iteration_support
        )
    yaw_loss_weight = (
        cfg.shape_loss_weight
        if cfg.yaw_update_mode == "legacy_shared"
        else cfg.yaw_loss_weight
    )
    return {
        "loss_cl_shape_size_6": size_loss * cfg.shape_loss_weight,
        "loss_cl_shape_yaw_6": yaw_loss * yaw_loss_weight,
        "loss_cl_yaw_keep_6": yaw_keep_loss * cfg.yaw_keep_loss_weight,
        "loss_cl_yaw_no_harm_6": (
            yaw_improvement_loss * cfg.yaw_improvement_loss_weight
        ),
        "loss_cl_range_distribution_6": (
            distribution_loss * cfg.distribution_nll_weight
        ),
        "loss_cl_candidate_soft_6": (candidate_loss * cfg.candidate_loss_weight),
        "loss_cl_radial_offset_6": (radial_offset_loss * cfg.radial_offset_loss_weight),
        "loss_cl_center_gate_6": center_gate_loss * cfg.center_gate_loss_weight,
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
        "cl_center_baseline_error_mean": _selected_mean(
            initial_distance, matched
        ).detach(),
        "cl_center_final_error_mean": _selected_mean(
            final_center_error, matched
        ).detach(),
        "cl_center_improvement_rate": _selected_mean(
            center_improved.to(final_reg.dtype), matched
        ).detach(),
        "cl_center_harm_rate": _selected_mean(
            center_harmed.to(final_reg.dtype), matched
        ).detach(),
        "cl_center_gate_supervised_count": center_gate_supervised_count,
        "cl_center_gate_positive_rate": (
            center_gate_positive_count / center_gate_supervised_count.clamp_min(1.0)
        ),
        "cl_yaw_baseline_error_mean": _selected_mean(
            baseline_yaw_error, matched
        ).detach(),
        "cl_yaw_final_error_mean": _selected_mean(
            final_yaw_error, matched
        ).detach(),
        "cl_yaw_improvement_rate": _selected_mean(
            yaw_improved.to(final_reg.dtype), matched
        ).detach(),
        "cl_yaw_raw_abs_mean": iteration_yaw_raw.detach().abs().mean(),
        "cl_yaw_delta_abs_mean": iteration_yaw_delta.detach().abs().mean(),
        "cl_yaw_gate_mean": iteration_yaw_gate.detach().mean(),
        "cl_yaw_saturation_rate": _selected_mean(
            yaw_saturated.to(final_reg.dtype),
            yaw_diagnostic_mask,
        ).detach(),
        "cl_posterior_entropy_mean": (record["posterior_entropy"].detach().mean()),
        "cl_range_distribution_mean_abs": (
            record["distribution_mean"].detach().abs().mean()
        ),
        "cl_range_distribution_std_mean": (record["distribution_std"].detach().mean()),
        "cl_quality_delta_abs_mean": (record["quality_delta"].detach().abs().mean()),
    }


__all__ = [
    "evaluator_aligned_centerness_target",
    "evaluator_aligned_quality_target",
    "rosa_losses",
    "validate_centerness_target_mode",
]
