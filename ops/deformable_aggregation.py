import torch
from torch.autograd.function import Function, once_differentiable

from . import deformable_aggregation_ext


class DeformableAggregationFunction(Function):
    @staticmethod
    def forward(
        ctx,
        mc_ms_feat,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    ):
        # output: [bs, num_pts, num_embeds]
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()
        output = deformable_aggregation_ext.deformable_aggregation_forward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        ctx.save_for_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        ) = ctx.saved_tensors
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()

        grad_mc_ms_feat = torch.zeros_like(mc_ms_feat)
        grad_sampling_location = torch.zeros_like(sampling_location)
        grad_weights = torch.zeros_like(weights)
        deformable_aggregation_ext.deformable_aggregation_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
            grad_output.contiguous(),
            grad_mc_ms_feat,
            grad_sampling_location,
            grad_weights,
        )
        return (
            grad_mc_ms_feat,
            None,
            None,
            grad_sampling_location,
            grad_weights,
        )


class DeformableAggregationPerCameraFunction(Function):
    """Return fixed-order fused and per-camera DAF values in one call.

    The Phase-1 CUDA forward has one writer per output and does not use
    ``atomicAdd``.  Backward deliberately preserves the legacy mathematical
    objective, but feature-map scatter, weight reduction, and sampling-location
    reduction still use atomics.  Consequently none of the three floating-point
    input gradients is advertised as bit-deterministic in Phase 1.
    """

    @staticmethod
    def forward(
        ctx,
        mc_ms_feat,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    ):
        # Make unused fused/view outputs reach backward as ``None`` so both
        # public-output contracts are exercised explicitly instead of relying
        # on autograd's materialized zero gradients.
        ctx.set_materialize_grads(False)
        if not hasattr(
            deformable_aggregation_ext,
            "deformable_aggregation_per_camera_forward",
        ):
            raise RuntimeError(
                "The loaded deformable_aggregation_ext predates the "
                "single-pass per-camera API. Rebuild the extension before "
                "enabling ray_object_single_pass_daf."
            )
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()
        output, view_output = (
            deformable_aggregation_ext.deformable_aggregation_per_camera_forward(
                mc_ms_feat,
                spatial_shape,
                scale_start_index,
                sampling_location,
                weights,
            )
        )
        ctx.save_for_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        return output, view_output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, grad_view_output):
        (
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        ) = ctx.saved_tensors
        if grad_output is None:
            grad_output = mc_ms_feat.new_zeros(
                mc_ms_feat.shape[0],
                sampling_location.shape[1],
                mc_ms_feat.shape[-1],
            )
        if grad_view_output is None:
            grad_view_output = mc_ms_feat.new_zeros(
                mc_ms_feat.shape[0],
                sampling_location.shape[1],
                spatial_shape.shape[0],
                mc_ms_feat.shape[-1],
            )
        grad_mc_ms_feat = torch.zeros_like(mc_ms_feat)
        grad_sampling_location = torch.zeros_like(sampling_location)
        grad_weights = torch.zeros_like(weights)
        deformable_aggregation_ext.deformable_aggregation_per_camera_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
            grad_output.contiguous().float(),
            grad_view_output.contiguous().float(),
            grad_mc_ms_feat,
            grad_sampling_location,
            grad_weights,
        )
        return (
            grad_mc_ms_feat,
            None,
            None,
            grad_sampling_location,
            grad_weights,
        )


class DeformableAggregationPerCameraDeterministicFunction(Function):
    """Fixed-order per-camera DAF with an atomic-free deterministic transpose.

    This experimental candidate shares the production fixed-order forward but
    routes backward through single-writer location/weight reductions and a
    sorted-edge destination gather for the feature-map gradient.
    """

    @staticmethod
    def forward(
        ctx,
        mc_ms_feat,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    ):
        ctx.set_materialize_grads(False)
        required = (
            "deformable_aggregation_per_camera_forward",
            "deformable_aggregation_per_camera_deterministic_backward",
        )
        missing = [
            name for name in required if not hasattr(deformable_aggregation_ext, name)
        ]
        if missing:
            raise RuntimeError(
                "The loaded deformable_aggregation_ext lacks deterministic "
                f"per-camera symbols: {missing}. Rebuild the isolated extension."
            )
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()
        output, view_output = (
            deformable_aggregation_ext.deformable_aggregation_per_camera_forward(
                mc_ms_feat,
                spatial_shape,
                scale_start_index,
                sampling_location,
                weights,
            )
        )
        ctx.save_for_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        return output, view_output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, grad_view_output):
        (
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        ) = ctx.saved_tensors
        if grad_output is None:
            grad_output = mc_ms_feat.new_zeros(
                mc_ms_feat.shape[0],
                sampling_location.shape[1],
                mc_ms_feat.shape[-1],
            )
        if grad_view_output is None:
            grad_view_output = mc_ms_feat.new_zeros(
                mc_ms_feat.shape[0],
                sampling_location.shape[1],
                spatial_shape.shape[0],
                mc_ms_feat.shape[-1],
            )
        grad_mc_ms_feat = torch.zeros_like(mc_ms_feat)
        grad_sampling_location = torch.zeros_like(sampling_location)
        grad_weights = torch.zeros_like(weights)
        (
            deformable_aggregation_ext
            .deformable_aggregation_per_camera_deterministic_backward(
                mc_ms_feat,
                spatial_shape,
                scale_start_index,
                sampling_location,
                weights,
                grad_output.contiguous().float(),
                grad_view_output.contiguous().float(),
                grad_mc_ms_feat,
                grad_sampling_location,
                grad_weights,
            )
        )
        return (
            grad_mc_ms_feat,
            None,
            None,
            grad_sampling_location,
            grad_weights,
        )
