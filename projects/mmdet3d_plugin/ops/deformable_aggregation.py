import torch
from torch.autograd.function import Function, once_differentiable

from . import deformable_aggregation_ext




class DeformableAggregationPerCameraFunction(Function):
    """Return fused and per-camera DAF values in one call.

    The forward is atomic-free; the backward uses atomics and is not
    bit-deterministic.
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
        # Unused outputs reach backward as ``None``.
        ctx.set_materialize_grads(False)
        if not hasattr(
            deformable_aggregation_ext,
            "deformable_aggregation_per_camera_forward",
        ):
            raise RuntimeError(
                "The loaded deformable_aggregation_ext lacks the "
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


