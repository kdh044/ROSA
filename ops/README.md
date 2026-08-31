# Fused CUDA operators

The released configs run the fused observation path, which needs these kernels.
Without them ROSA raises `RuntimeError: fused V3 sampling requires the Sparse4D
CUDA host` instead of falling back.

## Build

Copy over the Sparse4D host's `ops/` directory and rebuild the extension:

```bash
cp -r ops/* $FISHEYE3DOD_OFFICIAL_ROOT/projects/mmdet3d_plugin/ops/
cd $FISHEYE3DOD_OFFICIAL_ROOT/projects/mmdet3d_plugin/ops
python setup.py develop
```

Needs `nvcc` and a torch built against the same CUDA version.

## Running without them

Set `streaming_camera_rfl_cuda=False` and `use_fused_daf=False` in the config.
The PyTorch path is slower but scores the same — full validation with every
fusion on gives 0.6143 mAP / 0.6647 FDS, the Table 1 numbers. The two paths
agree to 1.2e-3 across the decoder outputs.

## What was added

Sparse4D's released kernel provides `deformable_aggregation_function`. On top
of that: `deformable_aggregation_per_camera_function` (sampling that keeps the
camera axis, so camera attention can weight it), `*_rfl_rosa_inference` (ray
feature lifting folded into sampling), and the `v49_*` kernels for keypoint
projection with the Eq. 1 cues, the hybrid box state update, and point
attention. `v49` is just the revision tag of the kernel set the checkpoints
were measured with.
