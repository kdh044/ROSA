# ROSA

**[WACV 2027] Ray-aware Observation Selection and Aggregation for Surround-View Fisheye 3D Detection**

DongHun Kim, Sang Jun Lee, HyungGi Jo<sup>†</sup> &nbsp;&nbsp;<sub><sup>†</sup> Corresponding author</sub>

ROSA is a camera-only 3D object detector for **surround-view fisheye cameras**.
It works directly on the raw fisheye images, without undistortion, and decides
which camera and which feature scale to trust for each observation of an
object, based on where and how that object appears in each fisheye view.

<p align="center">
  <img src="assets/demo/fisheye3dod.jpg" width="100%">
  <img src="assets/demo/kitti360.jpg" width="100%">
</p>

## News

- **[2026/11/10]** Datasets and checkpoints are now available on Hugging Face.
- **[2026/10/15]** ROSA has been accepted to WACV 2027!
- **[2026/10/15]** The official implementation is now publicly available.

## Overview

<p align="center">
  <img src="assets/figures/motivation.png" width="100%">
</p>

<p align="center">
  <img src="assets/figures/overview.png" width="100%">
</p>

<p align="center">
  <img src="assets/figures/box_refinement.png" width="45%">
  &nbsp;
  <img src="assets/figures/decoder_comparison.png" width="45%">
</p>

## Qualitative results

<p align="center">
  <img src="assets/figures/qualitative_fisheye3dod.jpg" width="100%">
</p>

<p align="center">
  <img src="assets/figures/qualitative_kitti360.jpg" width="100%">
</p>

<p align="center">
  <img src="assets/figures/qualitative_bev.jpg" width="100%">
</p>

## Analysis

<p align="center">
  <img src="assets/figures/aggregation_weights.jpg" width="100%">
</p>

<p align="center">
  <img src="assets/figures/fpn_resolution.jpg" width="80%">
</p>

## Getting started

| | |
|---|---|
| [Installation](docs/INSTALL.md) | environment, CUDA ops |
| [Data](docs/DATA.md) | Fisheye3DOD and KITTI-360 download, pretrained weights |
| [Getting started](docs/GETTING_STARTED.md) | checkpoints, evaluation, training, runtime |
| [Results](docs/RESULTS.md) | comparison tables and ablation |

## Citation

Accepted to WACV 2027. The proceedings citation will be added upon publication.

```bibtex
@unpublished{ROSA,
  title  = {ROSA: Ray-aware Observation Selection and Aggregation for
            Surround-View Fisheye 3D Detection},
  author = {Kim, DongHun and Lee, Sang Jun and Jo, HyungGi},
  year   = {2026},
  note   = {WACV 2027}
}
```

## License and acknowledgement

ROSA is released under the [MIT License](LICENSE). It is built on
[Sparse4D](https://github.com/HorizonRobotics/Sparse4D) (MIT, see [its license](projects/mmdet3d_plugin/LICENSE)) and evaluated on
[Fisheye3DOD](https://github.com/weiyangdaren/Fisheye3DOD) and
[KITTI-360](https://www.cvlibs.net/datasets/kitti-360/). We thank the authors
for their great work and for making it publicly available.
