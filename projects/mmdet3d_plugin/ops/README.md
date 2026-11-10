# Fused CUDA operators

The released configs run the fused observation path, which needs these kernels.
Without them ROSA raises a `RuntimeError` instead of falling back.

## Build

Built by `pip install -e . --no-build-isolation` at the repository root
(done by `tools/install.sh`). Needs `nvcc` and a PyTorch built against the
same CUDA version.

## Running without them

Set `streaming_camera_rfl_cuda=False` and `use_fused_daf=False` in the config.
The PyTorch path is slower but scores the same — full validation with every
fusion on gives 0.6143 mAP / 0.6647 FDS, the Table 1 numbers. The two paths
agree to within 1.2e-3 on the decoder outputs.

## Kernels

The extension builds on Sparse4D's deformable-aggregation kernel:

- `deformable_aggregation_per_camera_function`: sampling that keeps the camera
  axis, so that camera attention can weight it (training and inference).
- `deformable_aggregation_rfl_rosa_inference`: sampling with ray feature
  lifting and camera attention folded in (inference).
- `rosa_*`: keypoint construction and projection with the projection cues,
  hybrid box-state update, and point attention (inference).
