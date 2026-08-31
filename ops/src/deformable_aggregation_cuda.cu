
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <THC/THCAtomics.cuh>

#include <cstdint>
#include <climits>
#include <iostream>
#include <stdlib.h>


__device__ float bilinear_sampling(
    const float *&bottom_data, const int &height, const int &width,
    const int &num_embeds, const float &h_im, const float &w_im,
    const int &base_ptr
) {
  const int h_low = floorf(h_im);
  const int w_low = floorf(w_im);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const float lh = h_im - h_low;
  const float lw = w_im - w_low;
  const float hh = 1 - lh, hw = 1 - lw;

  const int w_stride = num_embeds;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
  }
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
  }
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
  }

  const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}


__device__ void bilinear_sampling_grad(
    const float *&bottom_data, const float &weight,
    const int &height, const int &width,
    const int &num_embeds, const float &h_im, const float &w_im,
    const int &base_ptr,
    const float &grad_output,
    float *&grad_mc_ms_feat, float *grad_sampling_location, float *grad_weights) {
  const int h_low = floorf(h_im);
  const int w_low = floorf(w_im);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const float lh = h_im - h_low;
  const float lw = w_im - w_low;
  const float hh = 1 - lh, hw = 1 - lw;

  const int w_stride = num_embeds;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

  const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const float top_grad_mc_ms_feat = grad_output * weight;
  float grad_h_weight = 0, grad_w_weight = 0;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_h_weight -= hw * v1;
    grad_w_weight -= hh * v1;
    atomicAdd(grad_mc_ms_feat + ptr1, w1 * top_grad_mc_ms_feat);
  }
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;
    atomicAdd(grad_mc_ms_feat + ptr2, w2 * top_grad_mc_ms_feat);
  }
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    atomicAdd(grad_mc_ms_feat + ptr3, w3 * top_grad_mc_ms_feat);
  }
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    atomicAdd(grad_mc_ms_feat + ptr4, w4 * top_grad_mc_ms_feat);
  }

  const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  atomicAdd(grad_weights, grad_output * val);
  atomicAdd(grad_sampling_location, width * grad_w_weight * top_grad_mc_ms_feat);
  atomicAdd(grad_sampling_location + 1, height * grad_h_weight * top_grad_mc_ms_feat);
}


__global__ void deformable_aggregation_kernel(
    const int num_kernels,
    float* output,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;

    const float weight = *(weights + idx / (num_embeds / num_groups));
    const int channel_index = idx % num_embeds;
    idx /= num_embeds;
    const int scale_index = idx % num_scale;
    idx /= num_scale;

    const int cam_index = idx % num_cams;
    idx /= num_cams;
    const int pts_index = idx % num_pts;
    idx /= num_pts;

    int anchor_index = idx % num_anchors;
    idx /= num_anchors;
    const int batch_index = idx % batch_size;
    idx /= batch_size;

    anchor_index = batch_index * num_anchors + anchor_index;
    const int loc_offset = ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;

    const float loc_w = sample_location[loc_offset];
    if (loc_w <= 0 || loc_w >= 1) return;
    const float loc_h = sample_location[loc_offset + 1];
    if (loc_h <= 0 || loc_h >= 1) return;
    
    int cam_scale_index = cam_index * num_scale + scale_index;
    const int value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;

    cam_scale_index = cam_scale_index << 1;
    const int h = spatial_shape[cam_scale_index];
    const int w = spatial_shape[cam_scale_index + 1];

    const float h_im = loc_h * h - 0.5;
    const float w_im = loc_w * w - 0.5;

    atomicAdd(
        output + anchor_index * num_embeds + channel_index,
        bilinear_sampling(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset) * weight
    );
}


// Phase-1 deterministic forward for ROSA.  Each thread is the unique writer
// for one (batch, anchor, camera, channel) output and accumulates points and
// levels in a fixed order.  Unlike deformable_aggregation_kernel, this kernel
// needs no atomicAdd in the forward pass.
__global__ void deformable_aggregation_per_camera_kernel(
    const int num_kernels,
    float* view_output,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;

    const int output_index = idx;
    const int channel_index = idx % num_embeds;
    idx /= num_embeds;
    const int cam_index = idx % num_cams;
    idx /= num_cams;
    const int local_anchor_index = idx % num_anchors;
    idx /= num_anchors;
    const int batch_index = idx;
    const int anchor_index = batch_index * num_anchors + local_anchor_index;
    const int group_index = channel_index / (num_embeds / num_groups);

    float accumulator = 0.0f;
    for (int pts_index = 0; pts_index < num_pts; ++pts_index) {
        const int loc_offset =
            ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;
        const float loc_w = sample_location[loc_offset];
        if (loc_w <= 0 || loc_w >= 1) continue;
        const float loc_h = sample_location[loc_offset + 1];
        if (loc_h <= 0 || loc_h >= 1) continue;

        for (int scale_index = 0; scale_index < num_scale; ++scale_index) {
            int cam_scale_index = cam_index * num_scale + scale_index;
            const int value_offset =
                (batch_index * num_feat + scale_start_index[cam_scale_index])
                * num_embeds + channel_index;
            cam_scale_index = cam_scale_index << 1;
            const int h = spatial_shape[cam_scale_index];
            const int w = spatial_shape[cam_scale_index + 1];
            const float h_im = loc_h * h - 0.5;
            const float w_im = loc_w * w - 0.5;
            const int weight_offset =
                (((((anchor_index * num_pts + pts_index) * num_cams + cam_index)
                    * num_scale + scale_index) * num_groups) + group_index);
            accumulator += bilinear_sampling(
                mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset
            ) * weights[weight_offset];
        }
    }
    view_output[output_index] = accumulator;
}


// Keep the camera reduction order explicit so the base output is also
// reproducible for a fixed input/binary.  The view layout is [B,N,C,D].
__global__ void deformable_aggregation_camera_sum_kernel(
    const int num_kernels,
    float* output,
    const float* view_output,
    int num_cams,
    int num_embeds
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;
    const int channel_index = idx % num_embeds;
    const int anchor_index = idx / num_embeds;
    float accumulator = 0.0f;
    for (int cam_index = 0; cam_index < num_cams; ++cam_index) {
        accumulator += view_output[
            (anchor_index * num_cams + cam_index) * num_embeds + channel_index
        ];
    }
    output[idx] = accumulator;
}


// V49 inference-only persistent fusion.  One block owns one flattened
// (batch, query, box-point) anchor.  The block keeps the four camera features
// and their compact CP16 RFL residuals in shared memory, so the large
// [B,N,C,D] carrier never reaches global memory.  The implementation is
// intentionally specialized to the released V49 topology (C=4, D=256,
// H=8, K=8, R=16, P_DAF=1); the C++ boundary rejects every other shape and
// falls back to the ordinary autograd path at the Python call site.
__global__ void deformable_aggregation_rfl_rosa_kernel(
    float* point_output,
    float* camera_attention,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    const float* camera_logits,
    const float* camera_rays,
    const bool* camera_valid,
    const float* query_heads,
    const float* appearance_weight,
    const float* ray_weight,
    const float* output_weight,
    const float* key_rfl_weight,
    const float* reliability_rfl_weight,
    int batch_size,
    int num_feat,
    int num_scale,
    int num_anchors,
    int points_per_query
) {
    constexpr int kCams = 4;
    constexpr int kDims = 256;
    constexpr int kHeads = 8;
    constexpr int kKeyDims = 8;
    constexpr int kRank = 16;
    constexpr int kBasis = 9;
    constexpr int kGroupDims = kDims / kHeads;
    constexpr float kLayerNormEps = 1.0e-5f;
    constexpr float kRayNormEps = 1.0e-6f;
    constexpr float kInvSqrtKey = 0.3535533905932738f;

    const int local_anchor = blockIdx.x;
    const int batch_index = blockIdx.y;
    if (local_anchor >= num_anchors || batch_index >= batch_size) return;
    const int channel = threadIdx.x;
    if (channel >= kDims) return;
    const int anchor = batch_index * num_anchors + local_anchor;

    __shared__ float base[kCams * kDims];
    __shared__ float residual[kCams * kDims];
    __shared__ float appearance[kCams * kRank];
    __shared__ float ray_factor[kCams * kRank];
    __shared__ float mean[kCams];
    __shared__ float inv_std[kCams];
    __shared__ float scores[kCams * kHeads];

    // Conditional level DAF.  V49 presents every box point as an independent
    // one-keypoint anchor, so there is exactly one sampling location per view.
    const int group = channel / kGroupDims;
    for (int camera = 0; camera < kCams; ++camera) {
        float value = 0.0f;
        const int loc_offset = ((anchor * kCams + camera) << 1);
        const float loc_w = sample_location[loc_offset];
        const float loc_h = sample_location[loc_offset + 1];
        if (loc_w > 0.0f && loc_w < 1.0f && loc_h > 0.0f && loc_h < 1.0f) {
            for (int level = 0; level < num_scale; ++level) {
                int camera_level = camera * num_scale + level;
                const int value_offset =
                    (batch_index * num_feat + scale_start_index[camera_level])
                    * kDims + channel;
                const int shape_offset = camera_level << 1;
                const int h = spatial_shape[shape_offset];
                const int w = spatial_shape[shape_offset + 1];
                const float h_im = loc_h * h - 0.5f;
                const float w_im = loc_w * w - 0.5f;
                const int weight_offset =
                    (((anchor * kCams + camera) * num_scale + level)
                     * kHeads + group);
                value += bilinear_sampling(
                    mc_ms_feat, h, w, kDims, h_im, w_im, value_offset
                ) * weights[weight_offset];
            }
        }
        base[camera * kDims + channel] = isfinite(value) ? value : 0.0f;
    }
    __syncthreads();

    // Parameter-free LayerNorm used by compact RFL.
    if (channel < kCams) {
        float sum = 0.0f;
        float square_sum = 0.0f;
        const int offset = channel * kDims;
        for (int d = 0; d < kDims; ++d) {
            const float x = base[offset + d];
            sum += x;
            square_sum += x * x;
        }
        const float mu = sum / static_cast<float>(kDims);
        const float variance = fmaxf(
            square_sum / static_cast<float>(kDims) - mu * mu, 0.0f
        );
        mean[channel] = mu;
        inv_std[channel] = rsqrtf(variance + kLayerNormEps);
    }
    __syncthreads();

    // CP16 appearance and degree-2 SH ray factors.
    if (channel < kCams * kRank) {
        const int camera = channel / kRank;
        const int rank = channel % kRank;
        float app = 0.0f;
        const int feature_offset = camera * kDims;
        const int app_weight_offset = rank * kDims;
        for (int d = 0; d < kDims; ++d) {
            const float normalized =
                (base[feature_offset + d] - mean[camera]) * inv_std[camera];
            app += appearance_weight[app_weight_offset + d] * normalized;
        }
        appearance[channel] = app;

        const int ray_offset = (anchor * kCams + camera) * 3;
        float x = camera_rays[ray_offset];
        float y = camera_rays[ray_offset + 1];
        float z = camera_rays[ray_offset + 2];
        const float norm = fmaxf(sqrtf(x * x + y * y + z * z), kRayNormEps);
        x /= norm;
        y /= norm;
        z /= norm;
        constexpr float c0 = 0.28209479177387814f;
        constexpr float c1 = 0.4886025119029199f;
        constexpr float c2_cross = 1.0925484305920792f;
        constexpr float c2_z = 0.31539156525252005f;
        constexpr float c2_xy = 0.5462742152960396f;
        const float basis[kBasis] = {
            c0, c1 * x, c1 * y, c1 * z,
            c2_cross * x * y, c2_cross * y * z, c2_cross * z * x,
            c2_z * (3.0f * z * z - 1.0f), c2_xy * (x * x - y * y)
        };
        float ray_value = 0.0f;
        const int ray_weight_offset = rank * kBasis;
        for (int basis_index = 0; basis_index < kBasis; ++basis_index) {
            ray_value += ray_weight[ray_weight_offset + basis_index]
                * basis[basis_index];
        }
        ray_factor[channel] = camera_valid[anchor * kCams + camera]
            ? ray_value : 0.0f;
    }
    __syncthreads();

    // RFL output projection.  Each channel thread owns all four cameras.
    for (int camera = 0; camera < kCams; ++camera) {
        float value = 0.0f;
        const int factor_offset = camera * kRank;
        const int output_weight_offset = channel * kRank;
        for (int rank = 0; rank < kRank; ++rank) {
            value += output_weight[output_weight_offset + rank]
                * appearance[factor_offset + rank]
                * ray_factor[factor_offset + rank];
        }
        residual[camera * kDims + channel] =
            camera_valid[anchor * kCams + camera] && isfinite(value)
            ? value : 0.0f;
    }
    __syncthreads();

    if (channel < kCams * kHeads) {
        const int camera = channel / kHeads;
        const int head = channel % kHeads;
        float query_key = 0.0f;
        const int query_index = local_anchor / points_per_query;
        const int query_offset =
            ((batch_index * (num_anchors / points_per_query) + query_index)
             * kHeads + head) * kKeyDims;
        for (int k = 0; k < kKeyDims; ++k) {
            float projected_key = 0.0f;
            const int compact_key = head * kKeyDims + k;
            const int factor_offset = camera * kRank;
            for (int rank = 0; rank < kRank; ++rank) {
                const float lifted = appearance[factor_offset + rank]
                    * ray_factor[factor_offset + rank];
                projected_key += key_rfl_weight[
                    compact_key * kRank + rank
                ] * lifted;
            }
            query_key += query_heads[query_offset + k] * projected_key;
        }
        float reliability = 0.0f;
        const int factor_offset = camera * kRank;
        for (int rank = 0; rank < kRank; ++rank) {
            reliability += reliability_rfl_weight[head * kRank + rank]
                * appearance[factor_offset + rank]
                * ray_factor[factor_offset + rank];
        }
        const int score_offset = (anchor * kCams + camera) * kHeads + head;
        scores[channel] = camera_valid[anchor * kCams + camera]
            ? camera_logits[score_offset] + query_key * kInvSqrtKey + reliability
            : -1.0e30f;
    }
    __syncthreads();

    // Four-camera softmax.  One thread owns one attention head.
    if (channel < kHeads) {
        float maximum = -1.0e30f;
        for (int camera = 0; camera < kCams; ++camera) {
            maximum = fmaxf(maximum, scores[camera * kHeads + channel]);
        }
        float denominator = 0.0f;
        float probabilities[kCams];
        for (int camera = 0; camera < kCams; ++camera) {
            const bool supported = camera_valid[anchor * kCams + camera];
            const float probability = supported ? expf(
                scores[camera * kHeads + channel] - maximum
            ) : 0.0f;
            probabilities[camera] = probability;
            denominator += probability;
        }
        const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
        for (int camera = 0; camera < kCams; ++camera) {
            const float probability = probabilities[camera] * inverse;
            scores[camera * kHeads + channel] = probability;
            camera_attention[
                (anchor * kCams + camera) * kHeads + channel
            ] = probability;
        }
    }
    __syncthreads();

    const int head = channel / kGroupDims;
    float fused = 0.0f;
    for (int camera = 0; camera < kCams; ++camera) {
        fused += scores[camera * kHeads + head]
            * (base[camera * kDims + channel]
               + residual[camera * kDims + channel]);
    }
    point_output[anchor * kDims + channel] = fused;
}


__global__ void deformable_aggregation_grad_kernel(
    const int num_kernels,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    const float* grad_output,
    float* grad_mc_ms_feat,
    float* grad_sampling_location,
    float* grad_weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;

    const int weights_ptr = idx / (num_embeds / num_groups);
    const int channel_index = idx % num_embeds;
    idx /= num_embeds;
    const int scale_index = idx % num_scale;
    idx /= num_scale;

    const int cam_index = idx % num_cams;
    idx /= num_cams;
    const int pts_index = idx % num_pts;
    idx /= num_pts;

    int anchor_index = idx % num_anchors;
    idx /= num_anchors;
    const int batch_index = idx % batch_size;
    idx /= batch_size;

    anchor_index = batch_index * num_anchors + anchor_index;
    const int loc_offset = ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;

    const float loc_w = sample_location[loc_offset];
    if (loc_w <= 0 || loc_w >= 1) return;
    const float loc_h = sample_location[loc_offset + 1];
    if (loc_h <= 0 || loc_h >= 1) return;
    
    const float grad = grad_output[anchor_index*num_embeds + channel_index];

    int cam_scale_index = cam_index * num_scale + scale_index;
    const int value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;

    cam_scale_index = cam_scale_index << 1;
    const int h = spatial_shape[cam_scale_index];
    const int w = spatial_shape[cam_scale_index + 1];

    const float h_im = loc_h * h - 0.5;
    const float w_im = loc_w * w - 0.5;

    /* atomicAdd( */
    /*     output + anchor_index * num_embeds + channel_index, */
    /*     bilinear_sampling(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset) * weight */
    /* ); */
    const float weight = weights[weights_ptr];
    float *grad_weights_ptr = grad_weights + weights_ptr;
    float *grad_location_ptr = grad_sampling_location + loc_offset;
    bilinear_sampling_grad(
        mc_ms_feat, weight, h, w, num_embeds, h_im, w_im,
        value_offset,
        grad,
        grad_mc_ms_feat, grad_location_ptr, grad_weights_ptr
    );
}


// Phase-1 per-camera backward.  This preserves the existing mathematical
// objective by adding the fused-output gradient to the corresponding camera
// gradient before applying the linear DAF transpose.  grad_mc_ms_feat still
// uses atomic scatter, while grad_weights and grad_sampling_location also use
// atomic reductions inside bilinear_sampling_grad (as does the legacy kernel).
// The Phase-1 operator is deterministic in forward but is therefore not
// claimed to be bit-deterministic in any of its three floating-point input
// gradients.
__global__ void deformable_aggregation_per_camera_grad_kernel(
    const int num_kernels,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    const float* grad_output,
    const float* grad_view_output,
    float* grad_mc_ms_feat,
    float* grad_sampling_location,
    float* grad_weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;

    const int weights_ptr = idx / (num_embeds / num_groups);
    const int channel_index = idx % num_embeds;
    idx /= num_embeds;
    const int scale_index = idx % num_scale;
    idx /= num_scale;

    const int cam_index = idx % num_cams;
    idx /= num_cams;
    const int pts_index = idx % num_pts;
    idx /= num_pts;

    int anchor_index = idx % num_anchors;
    idx /= num_anchors;
    const int batch_index = idx % batch_size;
    idx /= batch_size;

    anchor_index = batch_index * num_anchors + anchor_index;
    const int loc_offset =
        ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;

    const float loc_w = sample_location[loc_offset];
    if (loc_w <= 0 || loc_w >= 1) return;
    const float loc_h = sample_location[loc_offset + 1];
    if (loc_h <= 0 || loc_h >= 1) return;

    const int fused_grad_offset = anchor_index * num_embeds + channel_index;
    const int view_grad_offset =
        (anchor_index * num_cams + cam_index) * num_embeds + channel_index;
    const float grad =
        grad_output[fused_grad_offset] + grad_view_output[view_grad_offset];

    int cam_scale_index = cam_index * num_scale + scale_index;
    const int value_offset =
        (batch_index * num_feat + scale_start_index[cam_scale_index])
        * num_embeds + channel_index;
    cam_scale_index = cam_scale_index << 1;
    const int h = spatial_shape[cam_scale_index];
    const int w = spatial_shape[cam_scale_index + 1];
    const float h_im = loc_h * h - 0.5;
    const float w_im = loc_w * w - 0.5;

    const float weight = weights[weights_ptr];
    float *grad_weights_ptr = grad_weights + weights_ptr;
    float *grad_location_ptr = grad_sampling_location + loc_offset;
    bilinear_sampling_grad(
        mc_ms_feat, weight, h, w, num_embeds, h_im, w_im,
        value_offset, grad,
        grad_mc_ms_feat, grad_location_ptr, grad_weights_ptr
    );
}


// --------------------------------------------------------------------------
// Experimental deterministic backward for the fixed-order per-camera path.
//
// grad_weights and grad_sampling_location use one writer per destination and
// reduce channels/levels in explicit loop order.  grad_mc_ms_feat is a true
// destination gather.  We first materialize the four bilinear edges for every
// (B, anchor, point, camera, level), sort the edges by
//
//   (destination feature pixel, anchor-major source rank),
//
// and then let one thread own each destination feature scalar.  The sorted
// source rank fixes the floating-point addition order without atomicAdd.
// Edge sorting is intentionally performed by the C++ wrapper between the two
// launch functions below.

__device__ __forceinline__ void daf_bilinear_derivatives(
    const float* bottom_data,
    int height,
    int width,
    int num_embeds,
    float h_im,
    float w_im,
    int base_ptr,
    float* grad_h_weight,
    float* grad_w_weight
) {
    const int h_low = floorf(h_im);
    const int w_low = floorf(w_im);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;
    const float lh = h_im - h_low;
    const float lw = w_im - w_low;
    const float hh = 1.0f - lh;
    const float hw = 1.0f - lw;
    const int w_stride = num_embeds;
    const int h_stride = width * w_stride;
    const int h_low_ptr_offset = h_low * h_stride;
    const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
    const int w_low_ptr_offset = w_low * w_stride;
    const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

    float dh = 0.0f;
    float dw = 0.0f;
    if (h_low >= 0 && w_low >= 0) {
        const float v = bottom_data[
            h_low_ptr_offset + w_low_ptr_offset + base_ptr
        ];
        dh -= hw * v;
        dw -= hh * v;
    }
    if (h_low >= 0 && w_high <= width - 1) {
        const float v = bottom_data[
            h_low_ptr_offset + w_high_ptr_offset + base_ptr
        ];
        dh -= lw * v;
        dw += hh * v;
    }
    if (h_high <= height - 1 && w_low >= 0) {
        const float v = bottom_data[
            h_high_ptr_offset + w_low_ptr_offset + base_ptr
        ];
        dh += hw * v;
        dw -= lh * v;
    }
    if (h_high <= height - 1 && w_high <= width - 1) {
        const float v = bottom_data[
            h_high_ptr_offset + w_high_ptr_offset + base_ptr
        ];
        dh += lw * v;
        dw += lh * v;
    }
    *grad_h_weight = dh;
    *grad_w_weight = dw;
}


__global__ void daf_deterministic_edge_kernel(
    int num_edges,
    int64_t* edge_keys,
    float* edge_coefficients,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    int num_cams,
    int num_feat,
    int num_scale,
    int num_anchors,
    int num_pts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges) return;
    const int output_index = idx;

    const int corner_index = idx & 3;
    idx >>= 2;
    const int scale_index = idx % num_scale;
    idx /= num_scale;
    const int cam_index = idx % num_cams;
    idx /= num_cams;
    const int pts_index = idx % num_pts;
    idx /= num_pts;
    const int local_anchor_index = idx % num_anchors;
    idx /= num_anchors;
    const int batch_index = idx;

    edge_keys[output_index] = INT64_MAX;
    edge_coefficients[output_index] = 0.0f;

    const int anchor_index = batch_index * num_anchors + local_anchor_index;
    const int loc_offset =
        ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;
    const float loc_w = sample_location[loc_offset];
    const float loc_h = sample_location[loc_offset + 1];
    if (loc_w <= 0.0f || loc_w >= 1.0f ||
        loc_h <= 0.0f || loc_h >= 1.0f) {
        return;
    }

    const int cam_scale_index = cam_index * num_scale + scale_index;
    const int height = spatial_shape[cam_scale_index << 1];
    const int width = spatial_shape[(cam_scale_index << 1) + 1];
    const float h_im = loc_h * height - 0.5f;
    const float w_im = loc_w * width - 0.5f;
    const int h_low = floorf(h_im);
    const int w_low = floorf(w_im);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;
    const float lh = h_im - h_low;
    const float lw = w_im - w_low;
    const float hh = 1.0f - lh;
    const float hw = 1.0f - lw;

    int row = h_low;
    int col = w_low;
    float coefficient = hh * hw;
    if (corner_index == 1) {
        col = w_high;
        coefficient = hh * lw;
    } else if (corner_index == 2) {
        row = h_high;
        coefficient = lh * hw;
    } else if (corner_index == 3) {
        row = h_high;
        col = w_high;
        coefficient = lh * lw;
    }
    if (row < 0 || row >= height || col < 0 || col >= width) return;

    const int local_feature_index =
        scale_start_index[cam_scale_index] + row * width + col;
    const int64_t global_feature_index =
        static_cast<int64_t>(batch_index) * num_feat + local_feature_index;
    const int64_t sources_per_feature =
        static_cast<int64_t>(num_anchors) * num_pts * num_cams * num_scale;
    const int64_t source_rank =
        (((static_cast<int64_t>(local_anchor_index) * num_pts + pts_index)
          * num_cams + cam_index) * num_scale + scale_index);
    edge_keys[output_index] =
        global_feature_index * sources_per_feature + source_rank;
    edge_coefficients[output_index] = coefficient;
}


__global__ void daf_deterministic_weight_grad_kernel(
    int num_kernels,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* grad_output,
    const float* grad_view_output,
    float* grad_weights,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;
    const int output_index = idx;
    const int group_index = idx % num_groups;
    idx /= num_groups;
    const int scale_index = idx % num_scale;
    idx /= num_scale;
    const int cam_index = idx % num_cams;
    idx /= num_cams;
    const int pts_index = idx % num_pts;
    idx /= num_pts;
    const int local_anchor_index = idx % num_anchors;
    idx /= num_anchors;
    const int batch_index = idx;
    const int anchor_index = batch_index * num_anchors + local_anchor_index;
    const int loc_offset =
        ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;
    const float loc_w = sample_location[loc_offset];
    const float loc_h = sample_location[loc_offset + 1];
    if (loc_w <= 0.0f || loc_w >= 1.0f ||
        loc_h <= 0.0f || loc_h >= 1.0f) {
        grad_weights[output_index] = 0.0f;
        return;
    }

    const int cam_scale_index = cam_index * num_scale + scale_index;
    const int height = spatial_shape[cam_scale_index << 1];
    const int width = spatial_shape[(cam_scale_index << 1) + 1];
    const float h_im = loc_h * height - 0.5f;
    const float w_im = loc_w * width - 0.5f;
    const int channels_per_group = num_embeds / num_groups;
    const int channel_begin = group_index * channels_per_group;
    const int channel_end = channel_begin + channels_per_group;
    float accumulator = 0.0f;
    for (int channel_index = channel_begin;
         channel_index < channel_end;
         ++channel_index) {
        const int value_offset =
            (batch_index * num_feat + scale_start_index[cam_scale_index])
            * num_embeds + channel_index;
        const float value = bilinear_sampling(
            mc_ms_feat, height, width, num_embeds,
            h_im, w_im, value_offset
        );
        const int fused_grad_offset = anchor_index * num_embeds + channel_index;
        const int view_grad_offset =
            (anchor_index * num_cams + cam_index) * num_embeds + channel_index;
        const float grad =
            grad_output[fused_grad_offset] + grad_view_output[view_grad_offset];
        accumulator += grad * value;
    }
    grad_weights[output_index] = accumulator;
}


__global__ void daf_deterministic_location_grad_kernel(
    int num_kernels,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    const float* grad_output,
    const float* grad_view_output,
    float* grad_sampling_location,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;
    const int cam_index = idx % num_cams;
    idx /= num_cams;
    const int pts_index = idx % num_pts;
    idx /= num_pts;
    const int local_anchor_index = idx % num_anchors;
    idx /= num_anchors;
    const int batch_index = idx;
    const int anchor_index = batch_index * num_anchors + local_anchor_index;
    const int loc_offset =
        ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;
    const float loc_w = sample_location[loc_offset];
    const float loc_h = sample_location[loc_offset + 1];
    if (loc_w <= 0.0f || loc_w >= 1.0f ||
        loc_h <= 0.0f || loc_h >= 1.0f) {
        grad_sampling_location[loc_offset] = 0.0f;
        grad_sampling_location[loc_offset + 1] = 0.0f;
        return;
    }

    const int channels_per_group = num_embeds / num_groups;
    float accumulator_w = 0.0f;
    float accumulator_h = 0.0f;
    for (int scale_index = 0; scale_index < num_scale; ++scale_index) {
        const int cam_scale_index = cam_index * num_scale + scale_index;
        const int height = spatial_shape[cam_scale_index << 1];
        const int width = spatial_shape[(cam_scale_index << 1) + 1];
        const float h_im = loc_h * height - 0.5f;
        const float w_im = loc_w * width - 0.5f;
        for (int channel_index = 0;
             channel_index < num_embeds;
             ++channel_index) {
            const int group_index = channel_index / channels_per_group;
            const int weight_offset =
                (((((anchor_index * num_pts + pts_index) * num_cams + cam_index)
                    * num_scale + scale_index) * num_groups) + group_index);
            const int value_offset =
                (batch_index * num_feat + scale_start_index[cam_scale_index])
                * num_embeds + channel_index;
            float grad_h_weight = 0.0f;
            float grad_w_weight = 0.0f;
            daf_bilinear_derivatives(
                mc_ms_feat, height, width, num_embeds,
                h_im, w_im, value_offset,
                &grad_h_weight, &grad_w_weight
            );
            const int fused_grad_offset =
                anchor_index * num_embeds + channel_index;
            const int view_grad_offset =
                (anchor_index * num_cams + cam_index) * num_embeds
                + channel_index;
            const float grad =
                grad_output[fused_grad_offset]
                + grad_view_output[view_grad_offset];
            const float top_grad = grad * weights[weight_offset];
            accumulator_w += width * grad_w_weight * top_grad;
            accumulator_h += height * grad_h_weight * top_grad;
        }
    }
    grad_sampling_location[loc_offset] = accumulator_w;
    grad_sampling_location[loc_offset + 1] = accumulator_h;
}


__global__ void daf_deterministic_edge_segment_kernel(
    int64_t num_edges,
    const int64_t* sorted_edge_keys,
    int64_t* segment_starts,
    int64_t* segment_ends,
    int64_t sources_per_feature
) {
    const int64_t edge_index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (edge_index >= num_edges) return;
    const int64_t key = sorted_edge_keys[edge_index];
    if (key == INT64_MAX) return;
    const int64_t destination = key / sources_per_feature;
    if (edge_index == 0 ||
        sorted_edge_keys[edge_index - 1] / sources_per_feature != destination) {
        segment_starts[destination] = edge_index;
    }
    if (edge_index + 1 == num_edges ||
        sorted_edge_keys[edge_index + 1] == INT64_MAX ||
        sorted_edge_keys[edge_index + 1] / sources_per_feature != destination) {
        segment_ends[destination] = edge_index + 1;
    }
}


__global__ void daf_deterministic_feature_grad_kernel(
    int num_kernels,
    const int64_t* sorted_edge_keys,
    const float* sorted_edge_coefficients,
    const int64_t* segment_starts,
    const int64_t* segment_ends,
    const float* weights,
    const float* grad_output,
    const float* grad_view_output,
    float* grad_mc_ms_feat,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;
    const int output_index = idx;
    const int channel_index = idx % num_embeds;
    const int64_t global_feature_index = idx / num_embeds;
    const int batch_index = global_feature_index / num_feat;
    const int64_t sources_per_feature =
        static_cast<int64_t>(num_anchors) * num_pts * num_cams * num_scale;
    const int64_t key_begin = global_feature_index * sources_per_feature;
    const int64_t edge_begin = segment_starts[global_feature_index];
    if (edge_begin < 0) {
        grad_mc_ms_feat[output_index] = 0.0f;
        return;
    }
    const int64_t edge_end = segment_ends[global_feature_index];
    const int group_index =
        channel_index / (num_embeds / num_groups);
    float accumulator = 0.0f;
    for (int64_t edge_index = edge_begin;
         edge_index < edge_end;
         ++edge_index) {
        const int64_t source_rank = sorted_edge_keys[edge_index] - key_begin;
        int64_t source = source_rank;
        const int scale_index = source % num_scale;
        source /= num_scale;
        const int cam_index = source % num_cams;
        source /= num_cams;
        const int pts_index = source % num_pts;
        source /= num_pts;
        const int local_anchor_index = source;
        const int anchor_index = batch_index * num_anchors + local_anchor_index;
        const int weight_offset =
            (((((anchor_index * num_pts + pts_index) * num_cams + cam_index)
                * num_scale + scale_index) * num_groups) + group_index);
        const int fused_grad_offset =
            anchor_index * num_embeds + channel_index;
        const int view_grad_offset =
            (anchor_index * num_cams + cam_index) * num_embeds + channel_index;
        const float grad =
            grad_output[fused_grad_offset] + grad_view_output[view_grad_offset];
        const float top_grad = grad * weights[weight_offset];
        accumulator += sorted_edge_coefficients[edge_index] * top_grad;
    }
    grad_mc_ms_feat[output_index] = accumulator;
}


void deformable_aggregation_per_camera_deterministic_mark_segments(
    const int64_t* sorted_edge_keys,
    int64_t* segment_starts,
    int64_t* segment_ends,
    int64_t num_edges,
    int64_t sources_per_feature
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    daf_deterministic_edge_segment_kernel
        <<<(int)ceil(((double)num_edges / 128)), 128, 0, stream>>>(
        num_edges, sorted_edge_keys, segment_starts, segment_ends,
        sources_per_feature
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void deformable_aggregation_per_camera_deterministic_build_edges(
    int64_t* edge_keys,
    float* edge_coefficients,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_scale,
    int num_anchors,
    int num_pts
) {
    const int num_edges =
        batch_size * num_anchors * num_pts * num_cams * num_scale * 4;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    daf_deterministic_edge_kernel
        <<<(int)ceil(((double)num_edges / 128)), 128, 0, stream>>>(
        num_edges, edge_keys, edge_coefficients,
        spatial_shape, scale_start_index, sample_location,
        num_cams, num_feat, num_scale, num_anchors, num_pts
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void deformable_aggregation_per_camera_deterministic_grad(
    const int64_t* sorted_edge_keys,
    const float* sorted_edge_coefficients,
    const int64_t* segment_starts,
    const int64_t* segment_ends,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    const float* grad_output,
    const float* grad_view_output,
    float* grad_mc_ms_feat,
    float* grad_sampling_location,
    float* grad_weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int num_weight_kernels =
        batch_size * num_anchors * num_pts * num_cams * num_scale * num_groups;
    daf_deterministic_weight_grad_kernel
        <<<(int)ceil(((double)num_weight_kernels / 128)), 128, 0, stream>>>(
        num_weight_kernels,
        mc_ms_feat, spatial_shape, scale_start_index, sample_location,
        grad_output, grad_view_output, grad_weights,
        num_cams, num_feat, num_embeds, num_scale,
        num_anchors, num_pts, num_groups
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int num_location_kernels =
        batch_size * num_anchors * num_pts * num_cams;
    daf_deterministic_location_grad_kernel
        <<<(int)ceil(((double)num_location_kernels / 128)), 128, 0, stream>>>(
        num_location_kernels,
        mc_ms_feat, spatial_shape, scale_start_index, sample_location,
        weights, grad_output, grad_view_output, grad_sampling_location,
        num_cams, num_feat, num_embeds, num_scale,
        num_anchors, num_pts, num_groups
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int num_feature_kernels = batch_size * num_feat * num_embeds;
    daf_deterministic_feature_grad_kernel
        <<<(int)ceil(((double)num_feature_kernels / 128)), 128, 0, stream>>>(
        num_feature_kernels,
        sorted_edge_keys, sorted_edge_coefficients,
        segment_starts, segment_ends,
        weights, grad_output, grad_view_output, grad_mc_ms_feat,
        num_cams, num_feat, num_embeds, num_scale,
        num_anchors, num_pts, num_groups
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


__device__ __forceinline__ float v49_safe_value(float value) {
    return isfinite(value) ? value : 0.0f;
}


__device__ __forceinline__ float v49_sigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}


__device__ __forceinline__ void v49_normalize3(
    float &x, float &y, float &z
) {
    x = v49_safe_value(x);
    y = v49_safe_value(y);
    z = v49_safe_value(z);
    const float norm = sqrtf(x * x + y * y + z * z);
    if (!(norm > 1.0e-6f)) {
        x = 1.0f;
        y = 0.0f;
        z = 0.0f;
        return;
    }
    const float inv = 1.0f / fmaxf(norm, 1.0e-6f);
    x *= inv;
    y *= inv;
    z *= inv;
}


__global__ void v49_state_geometry_inputs_kernel(
    int elements, float* position_inputs, float* log_extent,
    const float* direction, const float* log_range, const float* log_size,
    const float* yaw, const float* velocity
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    float x = direction[index * 3], y = direction[index * 3 + 1];
    float z = direction[index * 3 + 2];
    v49_normalize3(x, y, z);
    float ys = v49_safe_value(yaw[index * 2]);
    float yc = v49_safe_value(yaw[index * 2 + 1]);
    const float yn = sqrtf(ys * ys + yc * yc);
    if (yn > 1.0e-6f) {
        const float inv = 1.0f / fmaxf(yn, 1.0e-6f);
        ys *= inv; yc *= inv;
    } else {
        ys = 0.0f; yc = 1.0f;
    }
    const float c0 = 0.28209479177387814f;
    const float c1 = 0.4886025119029199f;
    const float c2_cross = 1.0925484305920792f;
    const float c2_z = 0.31539156525252005f;
    const float c2_xy = 0.5462742152960396f;
    float* out = position_inputs + index * 18;
    out[0] = c0; out[1] = c1 * x; out[2] = c1 * y; out[3] = c1 * z;
    out[4] = c2_cross * x * y; out[5] = c2_cross * y * z;
    out[6] = c2_cross * z * x; out[7] = c2_z * (3.0f * z * z - 1.0f);
    out[8] = c2_xy * (x * x - y * y);
    out[9] = log_range[index];
    out[10] = log_size[index * 3]; out[11] = log_size[index * 3 + 1];
    out[12] = log_size[index * 3 + 2]; out[13] = ys; out[14] = yc;
    out[15] = velocity[index * 3]; out[16] = velocity[index * 3 + 1];
    out[17] = velocity[index * 3 + 2];

    const float distance = fmaxf(expf(log_range[index]), 1.0e-6f);
    const float sx = expf(log_size[index * 3]);
    const float sy = expf(log_size[index * 3 + 1]);
    const float sz = expf(log_size[index * 3 + 2]);
    const float horizontal = fmaxf(
        2.0f * atanf(0.5f * fmaxf(sx, sy) / distance), 1.0e-6f);
    const float vertical = fmaxf(
        2.0f * atanf(0.5f * sz / distance), 1.0e-6f);
    log_extent[index * 2] = logf(horizontal);
    log_extent[index * 2 + 1] = logf(vertical);
}


void v49_state_geometry_inputs(
    float* position_inputs, float* log_extent, const float* direction,
    const float* log_range, const float* log_size, const float* yaw,
    const float* velocity, int elements
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    v49_state_geometry_inputs_kernel<<<
        (int)ceil((double)elements / 128), 128, 0, stream>>>(
        elements, position_inputs, log_extent, direction, log_range,
        log_size, yaw, velocity);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


__global__ void v49_build_keypoints_kernel(
    int total, float* points, float* point_rays, const float* direction,
    const float* log_range, const float* log_size, const float* yaw,
    const float* learned, int queries, int points_per_query, float fixed_scale
) {
    const int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= total) return;
    const int query_flat = flat / points_per_query;
    const int point = flat - query_flat * points_per_query;
    float dx = direction[query_flat * 3], dy = direction[query_flat * 3 + 1];
    float dz = direction[query_flat * 3 + 2];
    v49_normalize3(dx, dy, dz);
    const float distance = expf(log_range[query_flat]);
    const float cx = dx * distance, cy = dy * distance, cz = dz * distance;
    float sx = 0.0f, sy = 0.0f, sz = 0.0f;
    if (point == 1) sx = fixed_scale;
    else if (point == 2) sx = -fixed_scale;
    else if (point == 3) sy = fixed_scale;
    else if (point == 4) sy = -fixed_scale;
    else if (point == 5) sz = fixed_scale;
    else if (point == 6) sz = -fixed_scale;
    else if (point >= 7) {
        const int offset = query_flat * (points_per_query - 7) * 3
            + (point - 7) * 3;
        sx = v49_sigmoid(learned[offset]) - 0.5f;
        sy = v49_sigmoid(learned[offset + 1]) - 0.5f;
        sz = v49_sigmoid(learned[offset + 2]) - 0.5f;
    }
    sx *= expf(log_size[query_flat * 3]);
    sy *= expf(log_size[query_flat * 3 + 1]);
    sz *= expf(log_size[query_flat * 3 + 2]);
    float ys = v49_safe_value(yaw[query_flat * 2]);
    float yc = v49_safe_value(yaw[query_flat * 2 + 1]);
    const float yn = sqrtf(ys * ys + yc * yc);
    if (yn > 1.0e-6f) {
        const float inv = 1.0f / fmaxf(yn, 1.0e-6f);
        ys *= inv; yc *= inv;
    } else {
        ys = 0.0f; yc = 1.0f;
    }
    const float px = cx + yc * sx - ys * sy;
    const float py = cy + ys * sx + yc * sy;
    const float pz = cz + sz;
    points[flat * 3] = px; points[flat * 3 + 1] = py;
    points[flat * 3 + 2] = pz;
    float rx = px, ry = py, rz = pz;
    v49_normalize3(rx, ry, rz);
    point_rays[flat * 3] = rx; point_rays[flat * 3 + 1] = ry;
    point_rays[flat * 3 + 2] = rz;
}


void v49_build_keypoints(
    float* points, float* point_rays, const float* direction,
    const float* log_range, const float* log_size, const float* yaw,
    const float* learned, int batch, int queries, int points_per_query,
    float fixed_scale
) {
    const int total = batch * queries * points_per_query;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    v49_build_keypoints_kernel<<<
        (int)ceil((double)total / 128), 128, 0, stream>>>(
        total, points, point_rays, direction, log_range, log_size, yaw,
        learned, queries, points_per_query, fixed_scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


__global__ void v49_point_attention_kernel(
    int query_heads, float* attention, float* evidence, const float* query,
    const float* key, const float* reliability, const float* values,
    const bool* valid
) {
    const int flat = blockIdx.x;
    const int tid = threadIdx.x;
    if (flat >= query_heads || tid >= 32) return;
    const int query_index = flat / 8;
    const int head = flat - query_index * 8;
    __shared__ float logits[13];
    __shared__ float weights[13];
    __shared__ int any_valid;
    if (tid == 0) any_valid = 0;
    if (tid < 13) {
        const bool supported = valid[query_index * 13 + tid];
        float score = -3.402823466e+38F;
        if (supported) {
            score = reliability[(query_index * 13 + tid) * 8 + head];
            const int qo = (query_index * 8 + head) * 8;
            const int ko = ((query_index * 13 + tid) * 8 + head) * 8;
            float dot = 0.0f;
            #pragma unroll
            for (int channel = 0; channel < 8; ++channel)
                dot += query[qo + channel] * key[ko + channel];
            score += dot * 0.3535533905932738f;
            atomicExch(&any_valid, 1);
        }
        logits[tid] = score;
    }
    __syncthreads();
    if (tid == 0) {
        if (!any_valid) {
            #pragma unroll
            for (int point = 0; point < 13; ++point) weights[point] = 0.0f;
        } else {
            float maximum = -3.402823466e+38F;
            #pragma unroll
            for (int point = 0; point < 13; ++point)
                maximum = fmaxf(maximum, logits[point]);
            float denominator = 0.0f;
            #pragma unroll
            for (int point = 0; point < 13; ++point) {
                const bool supported = valid[query_index * 13 + point];
                weights[point] = supported ? expf(logits[point] - maximum) : 0.0f;
                denominator += weights[point];
            }
            const float inv = denominator > 0.0f
                ? 1.0f / fmaxf(denominator, 1.0e-8f) : 0.0f;
            #pragma unroll
            for (int point = 0; point < 13; ++point) weights[point] *= inv;
        }
    }
    __syncthreads();
    if (tid < 13)
        attention[(query_index * 13 + tid) * 8 + head] = weights[tid];
    if (tid < 32) {
        float sum = 0.0f;
        #pragma unroll
        for (int point = 0; point < 13; ++point) {
            const int vo = (((query_index * 13 + point) * 8 + head) * 32 + tid);
            sum += weights[point] * values[vo];
        }
        evidence[(query_index * 8 + head) * 32 + tid] = sum;
    }
}


void v49_point_attention(
    float* attention, float* evidence, const float* query, const float* key,
    const float* reliability, const float* values, const bool* valid,
    int query_count
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    v49_point_attention_kernel<<<query_count * 8, 32, 0, stream>>>(
        query_count * 8, attention, evidence, query, key, reliability,
        values, valid);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


__global__ void v49_project_geometry_kernel(
    int total_points, float* uv, float* rays, bool* valid,
    float* descriptor, float* compatibility, const float* points,
    const float* lidar2cam, const float* image_wh, const float* intrinsic,
    const float* polynomial, const float* half_fov, const float* log_extent,
    const float* level_strides, int points_count, int cameras,
    int coefficients, int levels, int points_per_query
) {
    const int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= total_points) return;
    const int b = flat / points_count;
    const int n = flat - b * points_count;
    const int query = n / points_per_query;
    float ox = points[flat * 3], oy = points[flat * 3 + 1];
    float oz = points[flat * 3 + 2];
    v49_normalize3(ox, oy, oz);
    float refy = fabsf(oz) < 0.95f ? 0.0f : 1.0f;
    float refz = fabsf(oz) < 0.95f ? 1.0f : 0.0f;
    float e1x = refy * oz - refz * oy;
    float e1y = refz * ox;
    float e1z = -refy * ox;
    v49_normalize3(e1x, e1y, e1z);
    float e2x = oy * e1z - oz * e1y;
    float e2y = oz * e1x - ox * e1z;
    float e2z = ox * e1y - oy * e1x;
    v49_normalize3(e2x, e2y, e2z);

    float tangents[8], log_singular[8], margins[4];
    bool local_valid[4];
    int valid_count = 0;
    for (int cidx = 0; cidx < cameras; ++cidx) {
        const int t = (b * cameras + cidx) * 16;
        const float px = points[flat * 3];
        const float py = points[flat * 3 + 1];
        const float pz = points[flat * 3 + 2];
        float x = lidar2cam[t] * px + lidar2cam[t + 1] * py
            + lidar2cam[t + 2] * pz + lidar2cam[t + 3];
        float y = lidar2cam[t + 4] * px + lidar2cam[t + 5] * py
            + lidar2cam[t + 6] * pz + lidar2cam[t + 7];
        float z = lidar2cam[t + 8] * px + lidar2cam[t + 9] * py
            + lidar2cam[t + 10] * pz + lidar2cam[t + 11];
        const float camera_norm = sqrtf(x*x + y*y + z*z);
        const bool geometry_ok = isfinite(x) && isfinite(y) && isfinite(z)
            && camera_norm > 1.0e-6f;
        v49_normalize3(x, y, z);
        const int ray_offset = (flat * cameras + cidx) * 3;
        rays[ray_offset] = x; rays[ray_offset + 1] = y;
        rays[ray_offset + 2] = z;
        const float planar = sqrtf(x*x + y*y + 1.0e-12f);
        const float theta = atan2f(z, planar);
        const int po = (b * cameras + cidx) * coefficients;
        float rho = polynomial[po + coefficients - 1];
        for (int k = coefficients - 2; k >= 0; --k)
            rho = rho * theta + polynomial[po + k];
        float rho_d = coefficients > 1
            ? (coefficients - 1) * polynomial[po + coefficients - 1] : 0.0f;
        for (int k = coefficients - 2; k >= 1; --k)
            rho_d = rho_d * theta + k * polynomial[po + k];
        const float scale = planar > 1.0e-8f ? rho / fmaxf(planar,1.0e-8f) : 0.0f;
        const float ix = x * scale, iy = y * scale;
        const int io = (b * cameras + cidx) * 9;
        const float a11=intrinsic[io+3], a12=intrinsic[io+4], a13=intrinsic[io+2];
        const float a21=intrinsic[io], a22=intrinsic[io+1], a23=intrinsic[io+5];
        const int wo = (b * cameras + cidx) * 2;
        const float u=(a11*ix+a12*iy+a13)/image_wh[wo];
        const float v=(a21*ix+a22*iy+a23)/image_wh[wo+1];
        const float incidence=atan2f(planar,-z);
        const bool ok=geometry_ok && incidence<=half_fov[b*cameras+cidx]
            && u>=0.0f && u<=1.0f && v>=0.0f && v<=1.0f;
        local_valid[cidx]=ok; valid_count += ok ? 1 : 0;
        const int uo=(flat*cameras+cidx)*2;
        uv[uo]=ok?u:0.0f; uv[uo+1]=ok?v:0.0f;
        valid[flat*cameras+cidx]=ok;

        const float radial_x=planar>1.0e-8f?x/planar:1.0f;
        const float radial_y=planar>1.0e-8f?y/planar:0.0f;
        const float tangential_scale=fabsf(sinf(incidence))>1.0e-4f
            ? rho/fmaxf(fabsf(sinf(incidence)),1.0e-4f) : rho_d;
        const float ru=(a11*radial_x+a12*radial_y)*rho_d;
        const float rv=(a21*radial_x+a22*radial_y)*rho_d;
        const float tu=(a11*(-radial_y)+a12*radial_x)*tangential_scale;
        const float tv=(a21*(-radial_y)+a22*radial_x)*tangential_scale;
        const float tr=ru*ru+tu*tu+rv*rv+tv*tv;
        const float det2=(ru*tv-tu*rv)*(ru*tv-tu*rv);
        const float root=sqrtf(fmaxf(tr*tr-4.0f*det2,0.0f));
        const float s0=fmaxf(sqrtf(fmaxf((tr+root)*0.5f,1.0e-8f)),1.0e-4f);
        const float s1=fmaxf(sqrtf(fmaxf((tr-root)*0.5f,1.0e-8f)),1.0e-4f);
        log_singular[cidx*2]=ok?logf(s0):0.0f;
        log_singular[cidx*2+1]=ok?logf(s1):0.0f;
        const float border=fminf(fminf(u,1.0f-u),fminf(v,1.0f-v))*2.0f;
        const float angular=(half_fov[b*cameras+cidx]-incidence)
            / fmaxf(half_fov[b*cameras+cidx],1.0e-6f);
        margins[cidx]=ok?fminf(fminf(fmaxf(border,0.0f),1.0f),
                               fminf(fmaxf(angular,0.0f),1.0f)):0.0f;

        float tx=-lidar2cam[t+8], ty=-lidar2cam[t+9], tz=-lidar2cam[t+10];
        v49_normalize3(tx,ty,tz);
        const float tc0=tx*e1x+ty*e1y+tz*e1z;
        const float tc1=tx*e2x+ty*e2y+tz*e2z;
        const float sine=sqrtf(tc0*tc0+tc1*tc1);
        const float cosine=fminf(fmaxf(ox*tx+oy*ty+oz*tz,-1.0f),1.0f);
        const float factor=sine>1.0e-7f?atan2f(sine,cosine)/fmaxf(sine,1.0e-7f):0.0f;
        tangents[cidx*2]=tc0*factor; tangents[cidx*2+1]=tc1*factor;
    }
    const float overlap=fmaxf((float)valid_count-1.0f,0.0f)/3.0f;
    const float ex0=log_extent[(b*(points_count/points_per_query)+query)*2];
    const float ex1=log_extent[(b*(points_count/points_per_query)+query)*2+1];
    for(int cidx=0;cidx<cameras;++cidx) for(int l=0;l<levels;++l) {
        const int co=((flat*cameras+cidx)*levels+l);
        if(!local_valid[cidx]) {
            compatibility[co]=0.0f;
            for(int j=0;j<9;++j) descriptor[co*9+j]=0.0f;
            continue;
        }
        const float ls=logf(level_strides[l]);
        const float comp=-0.5f*((ls-log_singular[cidx*2]-ex0)*(ls-log_singular[cidx*2]-ex0)
            +(ls-log_singular[cidx*2+1]-ex1)*(ls-log_singular[cidx*2+1]-ex1));
        compatibility[co]=comp;
        float* out=descriptor+co*9;
        out[0]=tangents[cidx*2]; out[1]=tangents[cidx*2+1];
        out[2]=log_singular[cidx*2]; out[3]=log_singular[cidx*2+1];
        out[4]=fabsf(out[2]-out[3]); out[5]=margins[cidx];
        out[6]=overlap; out[7]=ls; out[8]=comp;
    }
}


void v49_project_geometry(
    float* uv, float* rays, bool* valid, float* descriptor,
    float* compatibility, const float* points, const float* lidar2cam,
    const float* image_wh, const float* intrinsic, const float* polynomial,
    const float* half_fov, const float* log_extent, const float* level_strides,
    int batch, int points_count, int cameras, int coefficients, int levels,
    int points_per_query
) {
    const int total=batch*points_count;
    cudaStream_t stream=at::cuda::getCurrentCUDAStream();
    v49_project_geometry_kernel<<<(int)ceil((double)total/128),128,0,stream>>>(
        total,uv,rays,valid,descriptor,compatibility,points,lidar2cam,image_wh,
        intrinsic,polynomial,half_fov,log_extent,level_strides,points_count,
        cameras,coefficients,levels,points_per_query);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


__global__ void v49_hybrid_state_update_kernel(
    int elements,
    float* out_direction,
    float* out_log_range,
    float* out_log_size,
    float* out_yaw,
    float* out_velocity,
    const float* direction,
    const float* log_range,
    const float* log_size,
    const float* yaw,
    const float* velocity,
    const float* raw,
    const float* update_gate_logits,
    const float* face_yaw,
    const float* face_yaw_gate_logits
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;

    const int d3 = index * 3;
    const int d2 = index * 2;
    const int d10 = index * 10;
    const int d4 = index * 4;
    const float center_gate = v49_sigmoid(update_gate_logits[d4]);
    const float size_gate = v49_sigmoid(update_gate_logits[d4 + 1]);
    const float yaw_gate = v49_sigmoid(update_gate_logits[d4 + 2]);
    const float velocity_gate = v49_sigmoid(update_gate_logits[d4 + 3]);

    float rx = direction[d3];
    float ry = direction[d3 + 1];
    float rz = direction[d3 + 2];
    v49_normalize3(rx, ry, rz);

    // Stable right-handed tangent frame, identical to spherical_tangent_basis.
    float refx = 0.0f;
    float refy = fabsf(rz) < 0.95f ? 0.0f : 1.0f;
    float refz = fabsf(rz) < 0.95f ? 1.0f : 0.0f;
    float e1x = refy * rz - refz * ry;
    float e1y = refz * rx - refx * rz;
    float e1z = refx * ry - refy * rx;
    v49_normalize3(e1x, e1y, e1z);
    float e2x = ry * e1z - rz * e1y;
    float e2y = rz * e1x - rx * e1z;
    float e2z = rx * e1y - ry * e1x;
    v49_normalize3(e2x, e2y, e2z);

    const float delta0 = raw[d10] * center_gate;
    const float delta1 = raw[d10 + 1] * center_gate;
    const float tx = delta0 * e1x + delta1 * e2x;
    const float ty = delta0 * e1y + delta1 * e2y;
    const float tz = delta0 * e1z + delta1 * e2z;
    const float angle = sqrtf(tx * tx + ty * ty + tz * tz);
    const float sinc = angle > 1.0e-8f ? sinf(angle) / angle : 1.0f;
    const float cosine = cosf(angle);
    float moved_x = rx * cosine + tx * sinc;
    float moved_y = ry * cosine + ty * sinc;
    float moved_z = rz * cosine + tz * sinc;
    v49_normalize3(moved_x, moved_y, moved_z);
    out_direction[d3] = moved_x;
    out_direction[d3 + 1] = moved_y;
    out_direction[d3 + 2] = moved_z;
    out_log_range[index] = log_range[index] + raw[d10 + 2] * center_gate;

    out_log_size[d3] = log_size[d3] + raw[d10 + 3] * size_gate;
    out_log_size[d3 + 1] = log_size[d3 + 1] + raw[d10 + 4] * size_gate;
    out_log_size[d3 + 2] = log_size[d3 + 2] + raw[d10 + 5] * size_gate;

    float sin_yaw = yaw[d2];
    float cos_yaw = yaw[d2 + 1];
    const float yaw_norm = sqrtf(sin_yaw * sin_yaw + cos_yaw * cos_yaw);
    if (!(yaw_norm > 1.0e-6f) || !isfinite(yaw_norm)) {
        sin_yaw = 0.0f;
        cos_yaw = 1.0f;
    } else {
        const float inv_yaw = 1.0f / fmaxf(yaw_norm, 1.0e-6f);
        sin_yaw *= inv_yaw;
        cos_yaw *= inv_yaw;
    }
    const float yaw_delta = raw[d10 + 6] * yaw_gate
        + face_yaw[index] * v49_sigmoid(face_yaw_gate_logits[index]);
    const float sin_delta = sinf(yaw_delta);
    const float cos_delta = cosf(yaw_delta);
    float rotated_sin = sin_yaw * cos_delta + cos_yaw * sin_delta;
    float rotated_cos = cos_yaw * cos_delta - sin_yaw * sin_delta;
    const float rotated_norm = sqrtf(
        rotated_sin * rotated_sin + rotated_cos * rotated_cos
    );
    const float inv_rotated = 1.0f / fmaxf(rotated_norm, 1.0e-6f);
    out_yaw[d2] = rotated_sin * inv_rotated;
    out_yaw[d2 + 1] = rotated_cos * inv_rotated;

    out_velocity[d3] = velocity[d3] + raw[d10 + 7] * velocity_gate;
    out_velocity[d3 + 1] = velocity[d3 + 1] + raw[d10 + 8] * velocity_gate;
    out_velocity[d3 + 2] = velocity[d3 + 2] + raw[d10 + 9] * velocity_gate;
}


void v49_hybrid_state_update(
    float* out_direction,
    float* out_log_range,
    float* out_log_size,
    float* out_yaw,
    float* out_velocity,
    const float* direction,
    const float* log_range,
    const float* log_size,
    const float* yaw,
    const float* velocity,
    const float* raw,
    const float* update_gate_logits,
    const float* face_yaw,
    const float* face_yaw_gate_logits,
    int elements
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    v49_hybrid_state_update_kernel
        <<<(int)ceil(((double)elements / 128)), 128, 0, stream>>>(
        elements, out_direction, out_log_range, out_log_size, out_yaw,
        out_velocity, direction, log_range, log_size, yaw, velocity, raw,
        update_gate_logits, face_yaw, face_yaw_gate_logits
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void deformable_aggregation(
    float* output,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups
) {
    const int num_kernels = batch_size * num_pts * num_embeds * num_anchors * num_cams * num_scale;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    deformable_aggregation_kernel
        <<<(int)ceil(((double)num_kernels/128)), 128, 0, stream>>>(
        num_kernels, output,
        mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
        batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void deformable_aggregation_per_camera(
    float* output,
    float* view_output,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int num_view_kernels =
        batch_size * num_anchors * num_cams * num_embeds;
    deformable_aggregation_per_camera_kernel
        <<<(int)ceil(((double)num_view_kernels / 128)), 128, 0, stream>>>(
        num_view_kernels, view_output,
        mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
        batch_size, num_cams, num_feat, num_embeds, num_scale,
        num_anchors, num_pts, num_groups
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int num_output_kernels = batch_size * num_anchors * num_embeds;
    deformable_aggregation_camera_sum_kernel
        <<<(int)ceil(((double)num_output_kernels / 128)), 128, 0, stream>>>(
        num_output_kernels, output, view_output, num_cams, num_embeds
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void deformable_aggregation_rfl_rosa(
    float* point_output,
    float* camera_attention,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    const float* camera_logits,
    const float* camera_rays,
    const bool* camera_valid,
    const float* query_heads,
    const float* appearance_weight,
    const float* ray_weight,
    const float* output_weight,
    const float* key_rfl_weight,
    const float* reliability_rfl_weight,
    int batch_size,
    int num_feat,
    int num_scale,
    int num_anchors,
    int points_per_query
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid(num_anchors, batch_size);
    deformable_aggregation_rfl_rosa_kernel<<<grid, 256, 0, stream>>>(
        point_output, camera_attention,
        mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
        camera_logits, camera_rays, camera_valid, query_heads,
        appearance_weight, ray_weight, output_weight, key_rfl_weight,
        reliability_rfl_weight,
        batch_size, num_feat, num_scale, num_anchors, points_per_query
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void deformable_aggregation_grad(
  const float* mc_ms_feat,
  const int* spatial_shape,
  const int* scale_start_index,
  const float* sample_location,
  const float* weights,
  const float* grad_output,
  float* grad_mc_ms_feat,
  float* grad_sampling_location,
  float* grad_weights,
  int batch_size,
  int num_cams,
  int num_feat,
  int num_embeds,
  int num_scale,
  int num_anchors,
  int num_pts,
  int num_groups
) {
    const int num_kernels = batch_size * num_pts * num_embeds * num_anchors * num_cams * num_scale;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    deformable_aggregation_grad_kernel
        <<<(int)ceil(((double)num_kernels/128)), 128, 0, stream>>>(
        num_kernels,
        mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
        grad_output, grad_mc_ms_feat, grad_sampling_location, grad_weights,
        batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void deformable_aggregation_per_camera_grad(
  const float* mc_ms_feat,
  const int* spatial_shape,
  const int* scale_start_index,
  const float* sample_location,
  const float* weights,
  const float* grad_output,
  const float* grad_view_output,
  float* grad_mc_ms_feat,
  float* grad_sampling_location,
  float* grad_weights,
  int batch_size,
  int num_cams,
  int num_feat,
  int num_embeds,
  int num_scale,
  int num_anchors,
  int num_pts,
  int num_groups
) {
    const int num_kernels =
        batch_size * num_pts * num_embeds * num_anchors * num_cams * num_scale;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    deformable_aggregation_per_camera_grad_kernel
        <<<(int)ceil(((double)num_kernels / 128)), 128, 0, stream>>>(
        num_kernels,
        mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
        grad_output, grad_view_output,
        grad_mc_ms_feat, grad_sampling_location, grad_weights,
        batch_size, num_cams, num_feat, num_embeds, num_scale,
        num_anchors, num_pts, num_groups
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
