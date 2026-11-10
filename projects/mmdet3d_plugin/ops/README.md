# Fused CUDA operators

The released configs run the fused observation path, which needs these kernels.
They are required; importing them without the build raises an `ImportError`.

## Build

Built by `pip install -e . --no-build-isolation` at the repository root
(done by `tools/install.sh`). Needs `nvcc` and a PyTorch built against the
same CUDA version.

## Kernels

The extension builds on Sparse4D's deformable-aggregation kernel:

- `deformable_aggregation_per_camera_function`: sampling that keeps the camera
  axis, so that camera attention can weight it (training and inference).
- `deformable_aggregation_rfl_rosa_inference`: sampling with ray feature
  lifting and camera attention folded in (inference).
- `rosa_*`: keypoint construction and projection with the projection cues,
  hybrid box-state update, and point attention (inference).
