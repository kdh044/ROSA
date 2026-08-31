# Installation

ROSA runs on the MMEngine registry with the MMDetection3D 1.4 data pipeline.
The versions below are the ones the reported results were produced with; other
combinations may work but have not been verified.

| Component | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.11.0 + cu128 |
| mmcv | 2.2.0 |
| mmengine | 0.10.7 |
| mmdet | 3.2.0 |
| mmdet3d | 1.4.0 |
| numpy | 2.2 |

> **mmcv 1.x will not work.** The Fisheye3DOD release targets MMDetection 2.x
> and imports `mmcv.runner`, which no longer exists in mmcv 2.x. ROSA ports the
> pieces it needs onto the MMEngine registry, so install `mmcv` (2.x), not
> `mmcv-full`.

## 1. Environment

```bash
conda create -n rosa python=3.10 -y
conda activate rosa
```

Install PyTorch for your CUDA toolkit first — do not let pip resolve it as a
transitive dependency:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## 2. OpenMMLab and the rest

```bash
pip install -r requirements.txt
```

If `mmcv` has no wheel for your torch/CUDA pair, build it from source:

```bash
pip install -U openmim
mim install "mmcv==2.2.0"
```

## 3. External checkout

ROSA reuses the released Fisheye3DOD evaluator and its VoVNet-99
implementation rather than copying third-party code into this repository.
Clone it and point ROSA at it:

```bash
git clone https://github.com/weiyangdaren/Fisheye3DOD external/fisheye3dod
export FISHEYE3DOD_OFFICIAL_ROOT=$PWD/external/fisheye3dod
export PYTHONPATH=$PWD:$FISHEYE3DOD_OFFICIAL_ROOT:$PYTHONPATH
```

`external/` is the default location, so if you clone there the environment
variable is optional.

## 4. Fused CUDA kernels (required)

The released configs enable the fused observation path
(`streaming_camera_rfl_cuda`, `use_fused_daf`), which the reported runtimes
were measured with. Those kernels extend the Sparse4D deformable-aggregation
extension and are **not** built automatically — without them the model raises
`RuntimeError: fused V3 sampling requires the Sparse4D CUDA host` on the first
forward pass. Copy them into the host checkout and build:

```bash
cp -r ops/* $FISHEYE3DOD_OFFICIAL_ROOT/projects/mmdet3d_plugin/ops/
cd $FISHEYE3DOD_OFFICIAL_ROOT/projects/mmdet3d_plugin/ops
python setup.py develop
```

This needs `nvcc` and a PyTorch built against the same CUDA version; the build
takes a few minutes. Check it with:

```bash
python -c "
from projects.mmdet3d_plugin.ops import v49_point_attention_inference
print('fused kernels OK')
"
```

To skip the build, set `streaming_camera_rfl_cuda=False` and
`use_fused_daf=False` in the config. The PyTorch path is slower but scores the
same 0.6143 mAP / 0.6647 FDS. See `ops/README.md`.

## 5. Verify

```bash
python -c "
from rosa.integrations import sparse4d, optim
from rosa.integrations.fisheye3dod import dataset, model, metric
print('ROSA import OK')
"
```

For the KITTI-360 arm:

```bash
python -c "
from rosa.integrations.kitti360 import dataset, metric, backbone
print('KITTI-360 import OK')
"
```
