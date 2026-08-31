#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/ops/sort.h>

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <vector>

namespace {

constexpr int64_t kCudaIndexLimit = std::numeric_limits<int>::max();

void check_index_product(
  std::initializer_list<int64_t> factors,
  const char* description
) {
  int64_t product = 1;
  for (const int64_t factor : factors) {
    TORCH_CHECK(factor > 0, description, " contains a non-positive dimension");
    TORCH_CHECK(
      product <= kCudaIndexLimit / factor,
      description, " exceeds the CUDA kernel int32 indexing limit"
    );
    product *= factor;
  }
}

void check_cuda_float_tensor(
  const at::Tensor &tensor,
  const at::Device &device,
  const char* name
) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.device() == device, name, " must be on ", device);
  TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(
    tensor.numel() <= kCudaIndexLimit,
    name, " exceeds the CUDA kernel int32 indexing limit"
  );
}

void check_cuda_int_tensor(
  const at::Tensor &tensor,
  const at::Device &device,
  const char* name
) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.device() == device, name, " must be on ", device);
  TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must be int32");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(
    tensor.numel() <= kCudaIndexLimit,
    name, " exceeds the CUDA kernel int32 indexing limit"
  );
}

void check_cuda_bool_tensor(
  const at::Tensor &tensor,
  const at::Device &device,
  const char* name
) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.device() == device, name, " must be on ", device);
  TORCH_CHECK(tensor.scalar_type() == at::kBool, name, " must be bool");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(
    tensor.numel() <= kCudaIndexLimit,
    name, " exceeds the CUDA kernel int32 indexing limit"
  );
}

void check_common_inputs(
  const at::Tensor &mc_ms_feat,
  const at::Tensor &spatial_shape,
  const at::Tensor &scale_start_index,
  const at::Tensor &sampling_location,
  const at::Tensor &weights
) {
  TORCH_CHECK(mc_ms_feat.dim() == 3, "mc_ms_feat must be [B,F,D]");
  TORCH_CHECK(spatial_shape.dim() == 3, "spatial_shape must be [C,L,2]");
  TORCH_CHECK(
    scale_start_index.dim() == 2,
    "scale_start_index must be [C,L]"
  );
  TORCH_CHECK(
    sampling_location.dim() == 5,
    "sampling_location must be [B,N,P,C,2]"
  );
  TORCH_CHECK(weights.dim() == 6, "weights must be [B,N,P,C,L,G]");

  const at::Device device = mc_ms_feat.device();
  check_cuda_float_tensor(mc_ms_feat, device, "mc_ms_feat");
  check_cuda_int_tensor(spatial_shape, device, "spatial_shape");
  check_cuda_int_tensor(scale_start_index, device, "scale_start_index");
  check_cuda_float_tensor(sampling_location, device, "sampling_location");
  check_cuda_float_tensor(weights, device, "weights");

  const int64_t batch = mc_ms_feat.size(0);
  const int64_t num_feat = mc_ms_feat.size(1);
  const int64_t embeds = mc_ms_feat.size(2);
  const int64_t cameras = spatial_shape.size(0);
  const int64_t levels = spatial_shape.size(1);
  const int64_t anchors = sampling_location.size(1);
  const int64_t points = sampling_location.size(2);
  const int64_t groups = weights.size(5);

  TORCH_CHECK(spatial_shape.size(2) == 2, "spatial_shape last dimension must be 2");
  TORCH_CHECK(
    scale_start_index.size(0) == cameras
      && scale_start_index.size(1) == levels,
    "scale_start_index shape must match spatial_shape [C,L]"
  );
  TORCH_CHECK(sampling_location.size(0) == batch, "sampling batch mismatch");
  TORCH_CHECK(sampling_location.size(3) == cameras, "sampling camera mismatch");
  TORCH_CHECK(sampling_location.size(4) == 2, "sampling last dimension must be 2");
  TORCH_CHECK(weights.size(0) == batch, "weights batch mismatch");
  TORCH_CHECK(weights.size(1) == anchors, "weights anchor mismatch");
  TORCH_CHECK(weights.size(2) == points, "weights point mismatch");
  TORCH_CHECK(weights.size(3) == cameras, "weights camera mismatch");
  TORCH_CHECK(weights.size(4) == levels, "weights level mismatch");
  TORCH_CHECK(groups > 0, "weights group dimension must be positive");
  TORCH_CHECK(embeds % groups == 0, "embedding dimension must be divisible by groups");

  check_index_product({batch, num_feat, embeds}, "feature indexing product");
  check_index_product(
    {batch, anchors, points, cameras, levels, embeds},
    "DAF launch product"
  );
  check_index_product({batch, anchors, cameras, embeds}, "view output product");
  check_index_product({batch, anchors, embeds}, "fused output product");
}

void check_like(
  const at::Tensor &tensor,
  const at::Tensor &reference,
  const char* name
) {
  check_cuda_float_tensor(tensor, reference.device(), name);
  TORCH_CHECK(tensor.sizes() == reference.sizes(), name, " shape mismatch");
}

void check_fused_grad(
  const at::Tensor &grad_output,
  const at::Tensor &mc_ms_feat,
  const at::Tensor &sampling_location
) {
  check_cuda_float_tensor(grad_output, mc_ms_feat.device(), "grad_output");
  TORCH_CHECK(grad_output.dim() == 3, "grad_output must be [B,N,D]");
  TORCH_CHECK(
    grad_output.size(0) == mc_ms_feat.size(0)
      && grad_output.size(1) == sampling_location.size(1)
      && grad_output.size(2) == mc_ms_feat.size(2),
    "grad_output shape mismatch"
  );
}

void check_view_grad(
  const at::Tensor &grad_view_output,
  const at::Tensor &mc_ms_feat,
  const at::Tensor &spatial_shape,
  const at::Tensor &sampling_location
) {
  check_cuda_float_tensor(
    grad_view_output, mc_ms_feat.device(), "grad_view_output"
  );
  TORCH_CHECK(grad_view_output.dim() == 4, "grad_view_output must be [B,N,C,D]");
  TORCH_CHECK(
    grad_view_output.size(0) == mc_ms_feat.size(0)
      && grad_view_output.size(1) == sampling_location.size(1)
      && grad_view_output.size(2) == spatial_shape.size(0)
      && grad_view_output.size(3) == mc_ms_feat.size(2),
    "grad_view_output shape mismatch"
  );
}

}  // namespace

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
);
  

/* feat: bs, num_feat, c */
/* _spatial_shape: cam, scale, 2 */
/* _scale_start_index: cam, scale */
/* _sampling_location: bs, anchor, pts, cam, 2 */
/* _weights: bs, anchor, pts, cam, scale, group */
/* output: bs, anchor, c */
/* kernel: bs, anchor, pts, c */


at::Tensor deformable_aggregation_forward(
  const at::Tensor &_mc_ms_feat,
  const at::Tensor &_spatial_shape,
  const at::Tensor &_scale_start_index,
  const at::Tensor &_sampling_location,
  const at::Tensor &_weights
) {
  check_common_inputs(
    _mc_ms_feat, _spatial_shape, _scale_start_index,
    _sampling_location, _weights
  );
  at::DeviceGuard guard(_mc_ms_feat.device());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_mc_ms_feat));
  int batch_size = _mc_ms_feat.size(0);
  int num_feat = _mc_ms_feat.size(1);
  int num_embeds = _mc_ms_feat.size(2);
  int num_cams = _spatial_shape.size(0);
  int num_scale = _spatial_shape.size(1);
  int num_anchors = _sampling_location.size(1);
  int num_pts = _sampling_location.size(2);
  int num_groups = _weights.size(5);

  const float* mc_ms_feat = _mc_ms_feat.data_ptr<float>();
  const int* spatial_shape = _spatial_shape.data_ptr<int>();
  const int* scale_start_index = _scale_start_index.data_ptr<int>();
  const float* sampling_location = _sampling_location.data_ptr<float>();
  const float* weights = _weights.data_ptr<float>();

  auto output = at::zeros({batch_size, num_anchors, num_embeds}, _mc_ms_feat.options());
  deformable_aggregation(
    output.data_ptr<float>(),
    mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights,
    batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups
  );
  return output;
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
);


/*
 * Return both the fixed-order all-camera sum and the per-camera values.
 * Keeping this API separate preserves the legacy operator for controlled A/B
 * testing and old checkpoints.
 */
std::vector<at::Tensor> deformable_aggregation_per_camera_forward(
  const at::Tensor &_mc_ms_feat,
  const at::Tensor &_spatial_shape,
  const at::Tensor &_scale_start_index,
  const at::Tensor &_sampling_location,
  const at::Tensor &_weights
) {
  check_common_inputs(
    _mc_ms_feat, _spatial_shape, _scale_start_index,
    _sampling_location, _weights
  );
  at::DeviceGuard guard(_mc_ms_feat.device());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_mc_ms_feat));
  int batch_size = _mc_ms_feat.size(0);
  int num_feat = _mc_ms_feat.size(1);
  int num_embeds = _mc_ms_feat.size(2);
  int num_cams = _spatial_shape.size(0);
  int num_scale = _spatial_shape.size(1);
  int num_anchors = _sampling_location.size(1);
  int num_pts = _sampling_location.size(2);
  int num_groups = _weights.size(5);

  const float* mc_ms_feat = _mc_ms_feat.data_ptr<float>();
  const int* spatial_shape = _spatial_shape.data_ptr<int>();
  const int* scale_start_index = _scale_start_index.data_ptr<int>();
  const float* sampling_location = _sampling_location.data_ptr<float>();
  const float* weights = _weights.data_ptr<float>();

  auto output = at::zeros(
    {batch_size, num_anchors, num_embeds}, _mc_ms_feat.options()
  );
  auto view_output = at::zeros(
    {batch_size, num_anchors, num_cams, num_embeds}, _mc_ms_feat.options()
  );
  deformable_aggregation_per_camera(
    output.data_ptr<float>(), view_output.data_ptr<float>(),
    mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights,
    batch_size, num_cams, num_feat, num_embeds, num_scale,
    num_anchors, num_pts, num_groups
  );
  return {output, view_output};
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
);

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
);

void v49_project_geometry(
  float* uv, float* rays, bool* valid, float* descriptor,
  float* compatibility, const float* points, const float* lidar2cam,
  const float* image_wh, const float* intrinsic, const float* polynomial,
  const float* half_fov, const float* log_extent, const float* level_strides,
  int batch, int points_count, int cameras, int coefficients, int levels,
  int points_per_query
);

void v49_state_geometry_inputs(
  float* position_inputs, float* log_extent, const float* direction,
  const float* log_range, const float* log_size, const float* yaw,
  const float* velocity, int elements
);

void v49_build_keypoints(
  float* points, float* point_rays, const float* direction,
  const float* log_range, const float* log_size, const float* yaw,
  const float* learned, int batch, int queries, int points_per_query,
  float fixed_scale
);

void v49_point_attention(
  float* attention, float* evidence, const float* query, const float* key,
  const float* reliability, const float* values, const bool* valid,
  int query_count
);

std::vector<at::Tensor> v49_point_attention_forward(
  const at::Tensor &_query, const at::Tensor &_key,
  const at::Tensor &_reliability, const at::Tensor &_values,
  const at::Tensor &_valid
) {
  const at::Device device = _query.device();
  for (const auto &item : std::vector<std::pair<const at::Tensor*, const char*>>{
      {&_query, "query"}, {&_key, "key"},
      {&_reliability, "reliability"}, {&_values, "values"}}) {
    check_cuda_float_tensor(*item.first, device, item.second);
  }
  check_cuda_bool_tensor(_valid, device, "valid");
  TORCH_CHECK(_query.dim() == 4 && _query.size(2) == 8 && _query.size(3) == 8,
              "point query must be [B,Q,8,8]");
  const int64_t batch = _query.size(0), queries = _query.size(1);
  TORCH_CHECK(_key.sizes() == at::IntArrayRef({batch, queries, 13, 8, 8}),
              "point key must be [B,Q,13,8,8]");
  TORCH_CHECK(_reliability.sizes()
              == at::IntArrayRef({batch, queries, 13, 8}),
              "point reliability must be [B,Q,13,8]");
  TORCH_CHECK(_values.sizes()
              == at::IntArrayRef({batch, queries, 13, 8, 32}),
              "point values must be [B,Q,13,8,32]");
  TORCH_CHECK(_valid.sizes() == at::IntArrayRef({batch, queries, 13}),
              "point valid must be [B,Q,13]");
  check_index_product({batch, queries, 13, 8}, "V49 point attention");
  at::DeviceGuard guard(device);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_query));
  auto attention = at::empty({batch, queries, 13, 8}, _query.options());
  auto evidence = at::empty({batch, queries, 8, 32}, _query.options());
  v49_point_attention(
    attention.data_ptr<float>(), evidence.data_ptr<float>(),
    _query.data_ptr<float>(), _key.data_ptr<float>(),
    _reliability.data_ptr<float>(), _values.data_ptr<float>(),
    _valid.data_ptr<bool>(), static_cast<int>(batch * queries));
  return {attention, evidence};
}

std::vector<at::Tensor> v49_state_geometry_inputs_forward(
  const at::Tensor &_direction, const at::Tensor &_log_range,
  const at::Tensor &_log_size, const at::Tensor &_yaw,
  const at::Tensor &_velocity
) {
  const at::Device device = _direction.device();
  for (const auto &item : std::vector<std::pair<const at::Tensor*, const char*>>{
      {&_direction, "direction"}, {&_log_range, "log_range"},
      {&_log_size, "log_size"}, {&_yaw, "yaw"},
      {&_velocity, "velocity"}}) {
    check_cuda_float_tensor(*item.first, device, item.second);
  }
  TORCH_CHECK(_direction.dim() == 3 && _direction.size(2) == 3,
              "direction must be [B,Q,3]");
  const int64_t batch = _direction.size(0), queries = _direction.size(1);
  TORCH_CHECK(_log_range.sizes() == at::IntArrayRef({batch, queries, 1}),
              "log_range must be [B,Q,1]");
  TORCH_CHECK(_log_size.sizes() == at::IntArrayRef({batch, queries, 3}),
              "log_size must be [B,Q,3]");
  TORCH_CHECK(_yaw.sizes() == at::IntArrayRef({batch, queries, 2}),
              "yaw must be [B,Q,2]");
  TORCH_CHECK(_velocity.sizes() == at::IntArrayRef({batch, queries, 3}),
              "velocity must be [B,Q,3]");
  check_index_product({batch, queries, 18}, "V49 state geometry inputs");
  at::DeviceGuard guard(device);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_direction));
  auto position_inputs = at::empty({batch, queries, 18}, _direction.options());
  auto log_extent = at::empty({batch, queries, 2}, _direction.options());
  v49_state_geometry_inputs(
    position_inputs.data_ptr<float>(), log_extent.data_ptr<float>(),
    _direction.data_ptr<float>(), _log_range.data_ptr<float>(),
    _log_size.data_ptr<float>(), _yaw.data_ptr<float>(),
    _velocity.data_ptr<float>(), static_cast<int>(batch * queries));
  return {position_inputs, log_extent};
}

std::vector<at::Tensor> v49_build_keypoints_project_geometry_forward(
  const at::Tensor &_direction, const at::Tensor &_log_range,
  const at::Tensor &_log_size, const at::Tensor &_yaw,
  const at::Tensor &_learned, const at::Tensor &_lidar2cam,
  const at::Tensor &_image_wh, const at::Tensor &_intrinsic,
  const at::Tensor &_polynomial, const at::Tensor &_half_fov,
  const at::Tensor &_log_extent, const at::Tensor &_level_strides,
  int64_t points_per_query, double fixed_scale
) {
  const at::Device device = _direction.device();
  for (const auto &item : std::vector<std::pair<const at::Tensor*, const char*>>{
      {&_direction, "direction"}, {&_log_range, "log_range"},
      {&_log_size, "log_size"}, {&_yaw, "yaw"}, {&_learned, "learned"},
      {&_lidar2cam, "lidar2cam"}, {&_image_wh, "image_wh"},
      {&_intrinsic, "intrinsic"}, {&_polynomial, "polynomial"},
      {&_half_fov, "half_fov"}, {&_log_extent, "log_extent"},
      {&_level_strides, "level_strides"}}) {
    check_cuda_float_tensor(*item.first, device, item.second);
  }
  TORCH_CHECK(_direction.dim() == 3 && _direction.size(2) == 3,
              "direction must be [B,Q,3]");
  const int64_t batch = _direction.size(0), queries = _direction.size(1);
  TORCH_CHECK(points_per_query == 13, "V49 requires 13 points per query");
  TORCH_CHECK(_log_range.sizes() == at::IntArrayRef({batch, queries, 1}),
              "log_range must be [B,Q,1]");
  TORCH_CHECK(_log_size.sizes() == at::IntArrayRef({batch, queries, 3}),
              "log_size must be [B,Q,3]");
  TORCH_CHECK(_yaw.sizes() == at::IntArrayRef({batch, queries, 2}),
              "yaw must be [B,Q,2]");
  TORCH_CHECK(_learned.sizes() == at::IntArrayRef({batch, queries, 18}),
              "learned keypoints must be [B,Q,18]");
  TORCH_CHECK(_log_extent.sizes() == at::IntArrayRef({batch, queries, 2}),
              "log_extent must be [B,Q,2]");
  TORCH_CHECK(fixed_scale > 0.0 && fixed_scale <= 0.5,
              "fixed keypoint scale is invalid");
  TORCH_CHECK(_lidar2cam.dim() == 4 && _lidar2cam.size(0) == batch
              && _lidar2cam.size(2) == 4 && _lidar2cam.size(3) == 4,
              "lidar2cam must be [B,C,4,4]");
  const int64_t cameras = _lidar2cam.size(1);
  TORCH_CHECK(cameras == 4, "V49 fused projection requires four cameras");
  TORCH_CHECK(_image_wh.sizes() == at::IntArrayRef({batch, cameras, 2}),
              "image_wh must be [B,C,2]");
  TORCH_CHECK(_intrinsic.sizes() == at::IntArrayRef({batch, cameras, 3, 3}),
              "intrinsic must be [B,C,3,3]");
  TORCH_CHECK(_polynomial.dim() == 3 && _polynomial.size(0) == batch
              && _polynomial.size(1) == cameras,
              "polynomial must be [B,C,K]");
  TORCH_CHECK(_half_fov.sizes() == at::IntArrayRef({batch, cameras}),
              "half_fov must be [B,C]");
  TORCH_CHECK(_level_strides.dim() == 1 && _level_strides.numel() > 0,
              "level_strides must be [L]");
  const int64_t count = queries * points_per_query;
  const int64_t levels = _level_strides.numel();
  check_index_product({batch, count, cameras, levels, 9},
                      "V49 keypoint projection output");
  at::DeviceGuard guard(device);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_direction));
  auto points = at::empty({batch, count, 3}, _direction.options());
  auto point_rays = at::empty({batch, count, 3}, _direction.options());
  auto uv = at::empty({batch, count, cameras, 2}, _direction.options());
  auto rays = at::empty({batch, count, cameras, 3}, _direction.options());
  auto valid = at::empty({batch, count, cameras},
                         _direction.options().dtype(at::kBool));
  auto descriptor = at::empty({batch, count, cameras, levels, 9},
                              _direction.options());
  auto compatibility = at::empty({batch, count, cameras, levels},
                                 _direction.options());
  v49_build_keypoints(
    points.data_ptr<float>(), point_rays.data_ptr<float>(),
    _direction.data_ptr<float>(), _log_range.data_ptr<float>(),
    _log_size.data_ptr<float>(), _yaw.data_ptr<float>(),
    _learned.data_ptr<float>(), static_cast<int>(batch),
    static_cast<int>(queries), static_cast<int>(points_per_query),
    static_cast<float>(fixed_scale));
  v49_project_geometry(
    uv.data_ptr<float>(), rays.data_ptr<float>(), valid.data_ptr<bool>(),
    descriptor.data_ptr<float>(), compatibility.data_ptr<float>(),
    points.data_ptr<float>(), _lidar2cam.data_ptr<float>(),
    _image_wh.data_ptr<float>(), _intrinsic.data_ptr<float>(),
    _polynomial.data_ptr<float>(), _half_fov.data_ptr<float>(),
    _log_extent.data_ptr<float>(), _level_strides.data_ptr<float>(),
    static_cast<int>(batch), static_cast<int>(count),
    static_cast<int>(cameras), static_cast<int>(_polynomial.size(2)),
    static_cast<int>(levels), static_cast<int>(points_per_query));
  return {points, point_rays, uv, rays, valid, descriptor, compatibility};
}

std::vector<at::Tensor> v49_project_geometry_forward(
  const at::Tensor &_points, const at::Tensor &_lidar2cam,
  const at::Tensor &_image_wh, const at::Tensor &_intrinsic,
  const at::Tensor &_polynomial, const at::Tensor &_half_fov,
  const at::Tensor &_log_extent, const at::Tensor &_level_strides,
  int64_t points_per_query
) {
  const at::Device device = _points.device();
  for (const auto &item : std::vector<std::pair<const at::Tensor*, const char*>>{
      {&_points, "points"}, {&_lidar2cam, "lidar2cam"},
      {&_image_wh, "image_wh"}, {&_intrinsic, "intrinsic"},
      {&_polynomial, "polynomial"}, {&_half_fov, "half_fov"},
      {&_log_extent, "log_extent"}, {&_level_strides, "level_strides"}}) {
    check_cuda_float_tensor(*item.first, device, item.second);
  }
  TORCH_CHECK(_points.dim() == 3 && _points.size(2) == 3,
              "points must be [B,N,3]");
  const int64_t batch = _points.size(0), count = _points.size(1);
  TORCH_CHECK(_lidar2cam.dim() == 4 && _lidar2cam.size(0) == batch
              && _lidar2cam.size(2) == 4 && _lidar2cam.size(3) == 4,
              "lidar2cam must be [B,C,4,4]");
  const int64_t cameras = _lidar2cam.size(1);
  TORCH_CHECK(cameras == 4, "V49 fused projection requires four cameras");
  TORCH_CHECK(_image_wh.sizes() == at::IntArrayRef({batch, cameras, 2}),
              "image_wh must be [B,C,2]");
  TORCH_CHECK(_intrinsic.sizes() == at::IntArrayRef({batch, cameras, 3, 3}),
              "intrinsic must be [B,C,3,3]");
  TORCH_CHECK(_polynomial.dim() == 3 && _polynomial.size(0) == batch
              && _polynomial.size(1) == cameras,
              "polynomial must be [B,C,K]");
  TORCH_CHECK(_half_fov.sizes() == at::IntArrayRef({batch, cameras}),
              "half_fov must be [B,C]");
  TORCH_CHECK(points_per_query > 0 && count % points_per_query == 0,
              "invalid points_per_query");
  const int64_t queries = count / points_per_query;
  TORCH_CHECK(_log_extent.sizes() == at::IntArrayRef({batch, queries, 2}),
              "log_extent must be [B,Q,2]");
  TORCH_CHECK(_level_strides.dim() == 1 && _level_strides.numel() > 0,
              "level_strides must be [L]");
  const int64_t levels = _level_strides.numel();
  check_index_product({batch, count, cameras, levels, 9},
                      "V49 projection geometry output");
  at::DeviceGuard guard(device);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_points));
  auto uv = at::empty({batch, count, cameras, 2}, _points.options());
  auto rays = at::empty({batch, count, cameras, 3}, _points.options());
  auto valid = at::empty({batch, count, cameras},
                         _points.options().dtype(at::kBool));
  auto descriptor = at::empty({batch, count, cameras, levels, 9},
                              _points.options());
  auto compatibility = at::empty({batch, count, cameras, levels},
                                 _points.options());
  v49_project_geometry(
    uv.data_ptr<float>(), rays.data_ptr<float>(), valid.data_ptr<bool>(),
    descriptor.data_ptr<float>(), compatibility.data_ptr<float>(),
    _points.data_ptr<float>(), _lidar2cam.data_ptr<float>(),
    _image_wh.data_ptr<float>(), _intrinsic.data_ptr<float>(),
    _polynomial.data_ptr<float>(), _half_fov.data_ptr<float>(),
    _log_extent.data_ptr<float>(), _level_strides.data_ptr<float>(),
    static_cast<int>(batch), static_cast<int>(count),
    static_cast<int>(cameras), static_cast<int>(_polynomial.size(2)),
    static_cast<int>(levels), static_cast<int>(points_per_query)
  );
  return {uv, rays, valid, descriptor, compatibility};
}


/*
 * Forward-only fusion of V49's post-head hybrid-state update.  The learned
 * linear heads remain in cuBLAS; this removes the many tiny slice, sigmoid,
 * spherical-map, yaw-composition, and state-assembly launches after them.
 */
std::vector<at::Tensor> v49_hybrid_state_update_forward(
  const at::Tensor &_direction,
  const at::Tensor &_log_range,
  const at::Tensor &_log_size,
  const at::Tensor &_yaw,
  const at::Tensor &_velocity,
  const at::Tensor &_raw,
  const at::Tensor &_update_gate_logits,
  const at::Tensor &_face_yaw,
  const at::Tensor &_face_yaw_gate_logits
) {
  const at::Device device = _direction.device();
  check_cuda_float_tensor(_direction, device, "direction");
  check_cuda_float_tensor(_log_range, device, "log_range");
  check_cuda_float_tensor(_log_size, device, "log_size");
  check_cuda_float_tensor(_yaw, device, "yaw");
  check_cuda_float_tensor(_velocity, device, "velocity");
  check_cuda_float_tensor(_raw, device, "raw");
  check_cuda_float_tensor(
    _update_gate_logits, device, "update_gate_logits"
  );
  check_cuda_float_tensor(_face_yaw, device, "face_yaw");
  check_cuda_float_tensor(
    _face_yaw_gate_logits, device, "face_yaw_gate_logits"
  );
  TORCH_CHECK(_direction.dim() == 3 && _direction.size(2) == 3,
              "direction must be [B,Q,3]");
  const int64_t batch = _direction.size(0);
  const int64_t queries = _direction.size(1);
  TORCH_CHECK(_log_range.sizes() == at::IntArrayRef({batch, queries, 1}),
              "log_range must be [B,Q,1]");
  TORCH_CHECK(_log_size.sizes() == at::IntArrayRef({batch, queries, 3}),
              "log_size must be [B,Q,3]");
  TORCH_CHECK(_yaw.sizes() == at::IntArrayRef({batch, queries, 2}),
              "yaw must be [B,Q,2]");
  TORCH_CHECK(_velocity.sizes() == at::IntArrayRef({batch, queries, 3}),
              "velocity must be [B,Q,3]");
  TORCH_CHECK(_raw.sizes() == at::IntArrayRef({batch, queries, 10}),
              "raw must be [B,Q,10]");
  TORCH_CHECK(
    _update_gate_logits.sizes() == at::IntArrayRef({batch, queries, 4}),
    "update_gate_logits must be [B,Q,4]"
  );
  TORCH_CHECK(_face_yaw.sizes() == at::IntArrayRef({batch, queries, 1}),
              "face_yaw must be [B,Q,1]");
  TORCH_CHECK(
    _face_yaw_gate_logits.sizes() == at::IntArrayRef({batch, queries, 1}),
    "face_yaw_gate_logits must be [B,Q,1]"
  );
  check_index_product({batch, queries}, "V49 state-update indexing product");

  at::DeviceGuard guard(device);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_direction));
  auto out_direction = at::empty_like(_direction);
  auto out_log_range = at::empty_like(_log_range);
  auto out_log_size = at::empty_like(_log_size);
  auto out_yaw = at::empty_like(_yaw);
  auto out_velocity = at::empty_like(_velocity);
  v49_hybrid_state_update(
    out_direction.data_ptr<float>(), out_log_range.data_ptr<float>(),
    out_log_size.data_ptr<float>(), out_yaw.data_ptr<float>(),
    out_velocity.data_ptr<float>(), _direction.data_ptr<float>(),
    _log_range.data_ptr<float>(), _log_size.data_ptr<float>(),
    _yaw.data_ptr<float>(), _velocity.data_ptr<float>(),
    _raw.data_ptr<float>(), _update_gate_logits.data_ptr<float>(),
    _face_yaw.data_ptr<float>(), _face_yaw_gate_logits.data_ptr<float>(),
    static_cast<int>(batch * queries)
  );
  return {
    out_direction, out_log_range, out_log_size, out_yaw, out_velocity
  };
}


/*
 * Released V49 inference topology fused into one persistent CUDA block per
 * flattened box point.  Training deliberately keeps the ordinary autograd
 * path; this forward-only API is an opt-in deployment optimization.
 */
std::vector<at::Tensor> deformable_aggregation_rfl_rosa_forward(
  const at::Tensor &_mc_ms_feat,
  const at::Tensor &_spatial_shape,
  const at::Tensor &_scale_start_index,
  const at::Tensor &_sampling_location,
  const at::Tensor &_weights,
  const at::Tensor &_camera_logits,
  const at::Tensor &_camera_rays,
  const at::Tensor &_camera_valid,
  const at::Tensor &_query_heads,
  const at::Tensor &_appearance_weight,
  const at::Tensor &_ray_weight,
  const at::Tensor &_output_weight,
  const at::Tensor &_key_rfl_weight,
  const at::Tensor &_reliability_rfl_weight,
  int64_t points_per_query
) {
  check_common_inputs(
    _mc_ms_feat, _spatial_shape, _scale_start_index,
    _sampling_location, _weights
  );
  const at::Device device = _mc_ms_feat.device();
  check_cuda_float_tensor(_camera_logits, device, "camera_logits");
  check_cuda_float_tensor(_camera_rays, device, "camera_rays");
  check_cuda_bool_tensor(_camera_valid, device, "camera_valid");
  check_cuda_float_tensor(_query_heads, device, "query_heads");
  check_cuda_float_tensor(_appearance_weight, device, "appearance_weight");
  check_cuda_float_tensor(_ray_weight, device, "ray_weight");
  check_cuda_float_tensor(_output_weight, device, "output_weight");
  check_cuda_float_tensor(_key_rfl_weight, device, "key_rfl_weight");
  check_cuda_float_tensor(
    _reliability_rfl_weight, device, "reliability_rfl_weight"
  );

  constexpr int64_t kCams = 4;
  constexpr int64_t kDims = 256;
  constexpr int64_t kHeads = 8;
  constexpr int64_t kKeyDims = 8;
  constexpr int64_t kRank = 16;
  constexpr int64_t kBasis = 9;
  const int64_t batch = _mc_ms_feat.size(0);
  const int64_t anchors = _sampling_location.size(1);
  const int64_t levels = _spatial_shape.size(1);
  TORCH_CHECK(_mc_ms_feat.size(2) == kDims, "streaming V49 requires D=256");
  TORCH_CHECK(_spatial_shape.size(0) == kCams, "streaming V49 requires C=4");
  TORCH_CHECK(_sampling_location.size(2) == 1, "streaming V49 requires P_DAF=1");
  TORCH_CHECK(_weights.size(5) == kHeads, "streaming V49 requires H=8");
  TORCH_CHECK(points_per_query > 0, "points_per_query must be positive");
  TORCH_CHECK(
    anchors % points_per_query == 0,
    "flattened anchors must divide by points_per_query"
  );
  const int64_t queries = anchors / points_per_query;
  TORCH_CHECK(
    _camera_logits.sizes() == at::IntArrayRef({batch, anchors, kCams, kHeads}),
    "camera_logits must be [B,N,4,8]"
  );
  TORCH_CHECK(
    _camera_rays.sizes() == at::IntArrayRef({batch, anchors, kCams, 3}),
    "camera_rays must be [B,N,4,3]"
  );
  TORCH_CHECK(
    _camera_valid.sizes() == at::IntArrayRef({batch, anchors, kCams}),
    "camera_valid must be [B,N,4]"
  );
  TORCH_CHECK(
    _query_heads.sizes()
      == at::IntArrayRef({batch, queries, kHeads, kKeyDims}),
    "query_heads must be [B,Q,8,8]"
  );
  TORCH_CHECK(
    _appearance_weight.sizes() == at::IntArrayRef({kRank, kDims}),
    "appearance_weight must be [16,256]"
  );
  TORCH_CHECK(
    _ray_weight.sizes() == at::IntArrayRef({kRank, kBasis}),
    "ray_weight must be [16,9]"
  );
  TORCH_CHECK(
    _output_weight.sizes() == at::IntArrayRef({kDims, kRank}),
    "output_weight must be [256,16]"
  );
  TORCH_CHECK(
    _key_rfl_weight.sizes()
      == at::IntArrayRef({kHeads * kKeyDims, kRank}),
    "key_rfl_weight must be [64,16]"
  );
  TORCH_CHECK(
    _reliability_rfl_weight.sizes() == at::IntArrayRef({kHeads, kRank}),
    "reliability_rfl_weight must be [8,16]"
  );

  at::DeviceGuard guard(device);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_mc_ms_feat));
  auto point_output = at::empty({batch, anchors, kDims}, _mc_ms_feat.options());
  auto camera_attention = at::empty(
    {batch, anchors, kCams, kHeads}, _mc_ms_feat.options()
  );
  deformable_aggregation_rfl_rosa(
    point_output.data_ptr<float>(), camera_attention.data_ptr<float>(),
    _mc_ms_feat.data_ptr<float>(), _spatial_shape.data_ptr<int>(),
    _scale_start_index.data_ptr<int>(), _sampling_location.data_ptr<float>(),
    _weights.data_ptr<float>(), _camera_logits.data_ptr<float>(),
    _camera_rays.data_ptr<float>(), _camera_valid.data_ptr<bool>(),
    _query_heads.data_ptr<float>(), _appearance_weight.data_ptr<float>(),
    _ray_weight.data_ptr<float>(), _output_weight.data_ptr<float>(),
    _key_rfl_weight.data_ptr<float>(),
    _reliability_rfl_weight.data_ptr<float>(),
    static_cast<int>(batch), static_cast<int>(_mc_ms_feat.size(1)),
    static_cast<int>(levels), static_cast<int>(anchors),
    static_cast<int>(points_per_query)
  );
  return {point_output, camera_attention};
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
);


void deformable_aggregation_backward(
  const at::Tensor &_mc_ms_feat,
  const at::Tensor &_spatial_shape,
  const at::Tensor &_scale_start_index,
  const at::Tensor &_sampling_location,
  const at::Tensor &_weights,
  const at::Tensor &_grad_output,
  at::Tensor &_grad_mc_ms_feat,
  at::Tensor &_grad_sampling_location,
  at::Tensor &_grad_weights
) {
  check_common_inputs(
    _mc_ms_feat, _spatial_shape, _scale_start_index,
    _sampling_location, _weights
  );
  check_fused_grad(_grad_output, _mc_ms_feat, _sampling_location);
  check_like(_grad_mc_ms_feat, _mc_ms_feat, "grad_mc_ms_feat");
  check_like(
    _grad_sampling_location,
    _sampling_location,
    "grad_sampling_location"
  );
  check_like(_grad_weights, _weights, "grad_weights");
  at::DeviceGuard guard(_mc_ms_feat.device());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_mc_ms_feat));
  int batch_size = _mc_ms_feat.size(0);
  int num_feat = _mc_ms_feat.size(1);
  int num_embeds = _mc_ms_feat.size(2);
  int num_cams = _spatial_shape.size(0);
  int num_scale = _spatial_shape.size(1);
  int num_anchors = _sampling_location.size(1);
  int num_pts = _sampling_location.size(2);
  int num_groups = _weights.size(5);

  const float* mc_ms_feat = _mc_ms_feat.data_ptr<float>();
  const int* spatial_shape = _spatial_shape.data_ptr<int>();
  const int* scale_start_index = _scale_start_index.data_ptr<int>();
  const float* sampling_location = _sampling_location.data_ptr<float>();
  const float* weights = _weights.data_ptr<float>();
  const float* grad_output = _grad_output.data_ptr<float>();

  float* grad_mc_ms_feat = _grad_mc_ms_feat.data_ptr<float>();
  float* grad_sampling_location = _grad_sampling_location.data_ptr<float>();
  float* grad_weights = _grad_weights.data_ptr<float>();

  deformable_aggregation_grad(
    mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights,
    grad_output, grad_mc_ms_feat, grad_sampling_location, grad_weights,
    batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups
  );
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
);


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
);


void deformable_aggregation_per_camera_deterministic_mark_segments(
  const int64_t* sorted_edge_keys,
  int64_t* segment_starts,
  int64_t* segment_ends,
  int64_t num_edges,
  int64_t sources_per_feature
);


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
);


void deformable_aggregation_per_camera_backward(
  const at::Tensor &_mc_ms_feat,
  const at::Tensor &_spatial_shape,
  const at::Tensor &_scale_start_index,
  const at::Tensor &_sampling_location,
  const at::Tensor &_weights,
  const at::Tensor &_grad_output,
  const at::Tensor &_grad_view_output,
  at::Tensor &_grad_mc_ms_feat,
  at::Tensor &_grad_sampling_location,
  at::Tensor &_grad_weights
) {
  check_common_inputs(
    _mc_ms_feat, _spatial_shape, _scale_start_index,
    _sampling_location, _weights
  );
  check_fused_grad(_grad_output, _mc_ms_feat, _sampling_location);
  check_view_grad(
    _grad_view_output, _mc_ms_feat, _spatial_shape, _sampling_location
  );
  check_like(_grad_mc_ms_feat, _mc_ms_feat, "grad_mc_ms_feat");
  check_like(
    _grad_sampling_location,
    _sampling_location,
    "grad_sampling_location"
  );
  check_like(_grad_weights, _weights, "grad_weights");
  at::DeviceGuard guard(_mc_ms_feat.device());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_mc_ms_feat));
  int batch_size = _mc_ms_feat.size(0);
  int num_feat = _mc_ms_feat.size(1);
  int num_embeds = _mc_ms_feat.size(2);
  int num_cams = _spatial_shape.size(0);
  int num_scale = _spatial_shape.size(1);
  int num_anchors = _sampling_location.size(1);
  int num_pts = _sampling_location.size(2);
  int num_groups = _weights.size(5);

  deformable_aggregation_per_camera_grad(
    _mc_ms_feat.data_ptr<float>(),
    _spatial_shape.data_ptr<int>(),
    _scale_start_index.data_ptr<int>(),
    _sampling_location.data_ptr<float>(),
    _weights.data_ptr<float>(),
    _grad_output.data_ptr<float>(),
    _grad_view_output.data_ptr<float>(),
    _grad_mc_ms_feat.data_ptr<float>(),
    _grad_sampling_location.data_ptr<float>(),
    _grad_weights.data_ptr<float>(),
    batch_size, num_cams, num_feat, num_embeds, num_scale,
    num_anchors, num_pts, num_groups
  );
}


/*
 * Experimental deterministic transpose for the fixed-order per-camera DAF.
 *
 * The CUDA implementation uses single-writer reductions for location and
 * weight gradients.  Feature gradients are destination gathers over bilinear
 * edges sorted by a unique composite key, so no floating-point atomic is used.
 * The valid edge keys are unique; equal INT64_MAX sentinels represent only
 * discarded out-of-image corners and are never read by the gather kernel.
 */
void deformable_aggregation_per_camera_deterministic_backward(
  const at::Tensor &_mc_ms_feat,
  const at::Tensor &_spatial_shape,
  const at::Tensor &_scale_start_index,
  const at::Tensor &_sampling_location,
  const at::Tensor &_weights,
  const at::Tensor &_grad_output,
  const at::Tensor &_grad_view_output,
  at::Tensor &_grad_mc_ms_feat,
  at::Tensor &_grad_sampling_location,
  at::Tensor &_grad_weights
) {
  check_common_inputs(
    _mc_ms_feat, _spatial_shape, _scale_start_index,
    _sampling_location, _weights
  );
  check_fused_grad(_grad_output, _mc_ms_feat, _sampling_location);
  check_view_grad(
    _grad_view_output, _mc_ms_feat, _spatial_shape, _sampling_location
  );
  check_like(_grad_mc_ms_feat, _mc_ms_feat, "grad_mc_ms_feat");
  check_like(
    _grad_sampling_location,
    _sampling_location,
    "grad_sampling_location"
  );
  check_like(_grad_weights, _weights, "grad_weights");
  at::DeviceGuard guard(_mc_ms_feat.device());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_mc_ms_feat));

  const int batch_size = _mc_ms_feat.size(0);
  const int num_feat = _mc_ms_feat.size(1);
  const int num_embeds = _mc_ms_feat.size(2);
  const int num_cams = _spatial_shape.size(0);
  const int num_scale = _spatial_shape.size(1);
  const int num_anchors = _sampling_location.size(1);
  const int num_pts = _sampling_location.size(2);
  const int num_groups = _weights.size(5);
  check_index_product(
    {batch_size, num_anchors, num_pts, num_cams, num_scale, 4},
    "deterministic backward edge product"
  );
  const int64_t num_edges =
    static_cast<int64_t>(batch_size) * num_anchors * num_pts
    * num_cams * num_scale * 4;
  const int64_t sources_per_feature =
    static_cast<int64_t>(num_anchors) * num_pts * num_cams * num_scale;
  TORCH_CHECK(
    static_cast<int64_t>(batch_size) * num_feat
      <= std::numeric_limits<int64_t>::max() / sources_per_feature,
    "deterministic backward composite edge key exceeds int64"
  );

  auto key_options = _mc_ms_feat.options().dtype(at::kLong);
  auto edge_keys = at::empty({num_edges}, key_options);
  auto edge_coefficients = at::empty({num_edges}, _mc_ms_feat.options());
  deformable_aggregation_per_camera_deterministic_build_edges(
    edge_keys.data_ptr<int64_t>(),
    edge_coefficients.data_ptr<float>(),
    _spatial_shape.data_ptr<int>(),
    _scale_start_index.data_ptr<int>(),
    _sampling_location.data_ptr<float>(),
    batch_size, num_cams, num_feat, num_scale, num_anchors, num_pts
  );

  // Valid keys are globally unique, making their order independent of how
  // the sort implementation places the duplicate invalid sentinels.
  auto sorted_result = at::sort(edge_keys, 0, false);
  auto sorted_edge_keys = std::get<0>(sorted_result);
  auto sorted_indices = std::get<1>(sorted_result);
  auto sorted_edge_coefficients =
    edge_coefficients.index_select(0, sorted_indices);
  auto segment_starts = at::full(
    {static_cast<int64_t>(batch_size) * num_feat}, -1, key_options
  );
  auto segment_ends = at::full(
    {static_cast<int64_t>(batch_size) * num_feat}, -1, key_options
  );
  deformable_aggregation_per_camera_deterministic_mark_segments(
    sorted_edge_keys.data_ptr<int64_t>(),
    segment_starts.data_ptr<int64_t>(),
    segment_ends.data_ptr<int64_t>(),
    num_edges,
    sources_per_feature
  );

  deformable_aggregation_per_camera_deterministic_grad(
    sorted_edge_keys.data_ptr<int64_t>(),
    sorted_edge_coefficients.data_ptr<float>(),
    segment_starts.data_ptr<int64_t>(),
    segment_ends.data_ptr<int64_t>(),
    _mc_ms_feat.data_ptr<float>(),
    _spatial_shape.data_ptr<int>(),
    _scale_start_index.data_ptr<int>(),
    _sampling_location.data_ptr<float>(),
    _weights.data_ptr<float>(),
    _grad_output.data_ptr<float>(),
    _grad_view_output.data_ptr<float>(),
    _grad_mc_ms_feat.data_ptr<float>(),
    _grad_sampling_location.data_ptr<float>(),
    _grad_weights.data_ptr<float>(),
    batch_size, num_cams, num_feat, num_embeds, num_scale,
    num_anchors, num_pts, num_groups
  );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "deformable_aggregation_forward",
    &deformable_aggregation_forward,
    "deformable_aggregation_forward"
  );
  m.def(
    "deformable_aggregation_backward",
    &deformable_aggregation_backward,
    "deformable_aggregation_backward"
  );
  m.def(
    "deformable_aggregation_per_camera_forward",
    &deformable_aggregation_per_camera_forward,
    "deformable_aggregation_per_camera_forward"
  );
  m.def(
    "deformable_aggregation_rfl_rosa_forward",
    &deformable_aggregation_rfl_rosa_forward,
    "V49 streaming DAF + camera RFL + ROSA forward"
  );
  m.def(
    "v49_hybrid_state_update_forward",
    &v49_hybrid_state_update_forward,
    "V49 fused hybrid-state update forward"
  );
  m.def(
    "v49_project_geometry_forward", &v49_project_geometry_forward,
    "V49 fused calibrated fisheye projection and geometry forward"
  );
  m.def(
    "v49_state_geometry_inputs_forward", &v49_state_geometry_inputs_forward,
    "V49 fused position input and angular extent forward"
  );
  m.def(
    "v49_build_keypoints_project_geometry_forward",
    &v49_build_keypoints_project_geometry_forward,
    "V49 fused keypoint construction and calibrated projection forward"
  );
  m.def(
    "v49_point_attention_forward", &v49_point_attention_forward,
    "V49 fused 13-point attention and value reduction forward"
  );
  m.def(
    "deformable_aggregation_per_camera_backward",
    &deformable_aggregation_per_camera_backward,
    "deformable_aggregation_per_camera_backward"
  );
  m.def(
    "deformable_aggregation_per_camera_deterministic_backward",
    &deformable_aggregation_per_camera_deterministic_backward,
    "deformable_aggregation_per_camera_deterministic_backward"
  );
}
