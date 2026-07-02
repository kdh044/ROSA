<div align="center">   
  
# Sparse4D: Sparse-based End-to-end Multi-view Temporal Perception
</div>

> [Github](https://github.com/HorizonRobotics/Sparse4D) \
> [Sparse4D v1: Multi-view 3D Object Detection with Sparse Spatial-Temporal Fusion](https://arxiv.org/abs/2211.10581) \
> [Sparse4D v2: Recurrent Temporal Fusion with Sparse Model](https://arxiv.org/abs/2305.14018) \
> [Sparse4D v3: Advancing End-to-End 3D Detection and Tracking](https://arxiv.org/abs/2311.11722) \
> [SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation](https://arxiv.org/abs/2405.19620) \
> [Chinese Interpretation of the Papers](https://zhuanlan.zhihu.com/p/637096473)

**【2024/06】Our follow-up project, SparseDrive (an end-to-end planning model based on the sparse framework), has been released!!! [arxiv](https://arxiv.org/abs/2405.19620) & [github](https://github.com/swc-17/SparseDrive).**

## Local Research Note: Fisheye-Aware Graph Query Aggregation

This fork is being used for 4-fisheye 3D MOT experiments on KB20/SphereStream-style data. The target research direction is:

> Fisheye-Aware Graph Query Aggregation for 4-Fisheye 3D MOT

The core position is that a fisheye query should not only ask "where is the 3D query projected?", but also "which projected node-view observation is reliable enough to update the temporal query?" The method should be presented as three connected modules instead of a single geometry-MLP add-on.

### Proposed Modules

| Module | Purpose | Main references | Difference in this work |
| --- | --- | --- | --- |
| Fisheye Graph Node Sampling | Build fixed 3D graph nodes around each object query/box instead of sampling only the object center. Nodes can include center, front/back, left/right, top/bottom, and box-edge samples. | Graph-DETR3D, Sparse4D, Graph-DETR4D if the citation/source is verified | Use fixed object/box nodes for a short-cycle implementation, then evaluate their fisheye projection quality. |
| Node-View Reliability Estimator | Estimate reliability for each object query x graph node x camera view. Candidate inputs are projection validity, projected footprint, radial distance, truncation ratio, visible camera count, and area ratio to the best view. | Fisheye3DOD, FishBEV, DaF-BEVSeg/FisheyeBEVSeg, MIC-BEV | Move from camera-level or BEV-cell-level importance to object graph node-view reliability. |
| Reliability-Guided Temporal Query Update | Use node-view reliability to modulate graph aggregation and temporal query updates so poor fisheye observations do not destabilize query persistence and tracking. | Sparse4D v3, StreamPETR, UA-Track, Graph-DETR4D if verified | Focus on temporal query update, not post-hoc tracking association cost. |

### Minimal Reading Plan

| Priority | Paper | Why it matters |
| ---: | --- | --- |
| 1 | Graph-DETR3D | Direct basis for graph node feature aggregation around DETR3D-style queries, especially for border/overlap regions. |
| 2 | Graph-DETR4D | Useful if a valid source is found for temporal graph extension; verify before citing in a paper. |
| 3 | MIC-BEV | Defense paper for camera/view importance; contrast against infrastructure-camera BEV-cell importance. |
| 4 | UA-Track | Defense paper for uncertainty-aware 3D MOT; contrast against object-state uncertainty and query denoising. |
| 5 | ORA3D | Defense paper for overlap-region/depth-prior handling; contrast against fisheye node-view reliability. |

### Current Ablation Notes

Local validation results suggest that velocity labels and class composition are both confounders:

| Run | Classes | Velocity labels | Best mAP | Final mAP | Final NDS | Final mAVE | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `sparse4d_kb20_car_ped_100e_github_r50_b8` | Car, Pedestrian | zero-valued | 0.4371 @ epoch 25 | 0.3952 | 0.4515 | 0.0028 | Strongest local baseline so far. |
| `sparse4d_kb20_car_ped_cyc_100e_github_r50_b8` | Car, Pedestrian, Cyclist | real | 0.2807 @ epoch 25 | 0.2253 | 0.2216 | 0.9601 | Real velocity heavily hurts NDS/mAVE; Cyclist remains weak. |
| `sparse4d_kb20_car_ped_cyc_zerovel_30e_github_r50_b8` | Car, Pedestrian, Cyclist | zero-valued | 0.2719 @ epoch 20 | 0.2447 | 0.3446 | 0.0033 | Finished successfully; zero velocity fixes mAVE/NDS but does not recover 2-class mAP. |

Interpretation: real velocity is a major metric/label issue, especially for NDS and mAVE. However, the 3-class zero-velocity run still underperforms the 2-class baseline, so Cyclist difficulty, data balance, and fisheye view-quality issues remain likely causes. The proposed node-view reliability modules are orthogonal to the velocity ablation and should be evaluated after the label/class confounds are controlled.


### Current Experiment: Sparse Multi-view Ray Matching

The next experiment is an oracle sanity check for object-query-level sparse multi-view ray matching in the 4-fisheye setup. The research question is:

> When candidate depths around a GT object center are reprojected into the visible fisheye views, does the multi-view feature consistency score peak near the true depth?

The key hypothesis is that overlap in a 4-fisheye rig is not just redundant coverage. For each object query, multiple overlapping fisheye views may support different candidate 3D locations with different reliability. The method should therefore give higher weight to the visible observations that best support the query's 3D position, using projection quality, footprint, truncation, and feature consistency instead of treating all views equally.

This experiment differs from ORA3D-style dense overlap supervision. ORA3D uses dense pixel-level disparity/depth cues over pinhole overlap regions. This fork should first test sparse object-query-level ray matching:

```text
dense full-image overlap matching: no
object-query visible fisheye views: yes
```

Initial procedure:

1. Select GT objects visible in at least two cameras.
2. Choose an oracle reference camera where the object is best observed, using projected area, image-center distance, truncation, and projection validity.
3. Project the GT object center into the reference camera and inverse-project the fisheye pixel into a 3D ray.
4. Sample candidate depths around GT depth, for example `d_gt +/- 2m` with `K=21`.
5. Reproject each candidate 3D point into all four fisheye cameras.
6. Keep only valid visible views. A valid view must pass image-bound checks, fisheye FOV checks, and projection-quality filtering. Views outside the image, outside the lens cone, heavily truncated, or on invalid regions must not pollute the matching score.
7. Sample backbone/neck/object-region features from the valid views.
8. Compute the candidate score with reference-to-visible cosine similarity first:

```text
score(d_k) = mean_c cosine(f_ref(d_k), f_c(d_k))
```

Later variants can compare all visible-view pairs:

```text
score(d_k) = mean_{i,j} cosine(f_i(d_k), f_j(d_k))
```

Success criterion: the score curve should have a clear local peak near `d_gt`. If all candidate depths have similar scores, likely causes include low feature resolution, small objects, view appearance gap, bad reference-view choice, or sampling background instead of the object.

Implementation should start as a standalone diagnostic script under `tools/`, not inside the training loop. A suitable target is:

```text
tools/ray_matching_sanity_check.py
```

The script should reuse the existing config/model/dataloader path from `projects/mmdet3d_plugin`, extract features from a trained checkpoint, dump per-object cost curves, and write results under:

```text
work_dirs/_ray_matching_sanity/
```

The first version should not train a new model. It should only answer whether GT-centered sparse multi-view ray matching produces a meaningful cost curve. If this sanity check works, the next stage is to replace oracle GT depth candidates with baseline predicted depth or temporal-prior depth candidates and then connect the score to query depth/update reliability.


## Overall Architecture
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="resources/sparse4d_architecture.jpg" width="1000">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Overall Framework of Sparse4D, which conforms to an encoder-decoder structure. The inputs mainly consists of three components: multi-view images, newly initialized instances, propagated instances from previous frame. The output is the refined instances (3D anchor boxes and corresponding features), serve as the perception results for the current frame. Additionally, a subset of these refined instances is selected and propagated to the next frame.</div>
</center>


<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="resources/efficient_deformable_aggregation.jpg" width="1000">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Illustration of our Efficient Deformable Aggregation Module. (a) The basic pipeline: we first generate multiple 3D key points inside 3D anchor, then sampling multi-scale/view image feature for each keypoint, and fuse these feature with predicted weight. (b) The parallel implementation: to further improve speed and reduce memory cost, we achieve a parallel implementation, where feature sampling and multi-view/scale weighted sum are combined as a CUDA operation. Our CUDA implementation supports handling different feature resolutions from different views. </div>
</center>


## nuScenes Benchmark
### Results on Validation Split
These experiments were conducted using 8 RTX 3090 GPUs with 24 GB memory.
|model | backbone |pretrain| img size | Epoch | Traning | FPS | NDS | mAP |  AMOTA |AMOTP |IDS| config | ckpt | log |
|  :----:  | :---: | :---: | :---: | :---: | :---:| :---:|:---:|:---: | :---: | :----: | :----: | :---: | :----: | :----: |
|Sparse4D-T4 |Res101|[FCOS3D](https://github.com/linxuewu/Sparse4D/releases/download/v0.0/fcos3d.pth)|640x1600|24|2Day5H|2.9|0.5438|0.4409|-|-|-|[cfg](https://github.com/linxuewu/Sparse4D/blob/v2.0/projects/configs/sparse4d_r101_H4.py)|[ckpt](https://github.com/linxuewu/Sparse4D/releases/download/v0.0/sparse4dv1_r101_H4_release.pth)|[log](https://github.com/linxuewu/Sparse4D/releases/download/v0.0/sparse4d.log)|
|Sparse4Dv2|Res50|[ImageNet]()|256x704| 100 |15H | 20.3 |0.5384|0.4392|-|-|-|[cfg](https://github.com/linxuewu/Sparse4D/blob/v2.0/projects/configs/sparse4dv2_r50_HInf_256x704.py)|[ckpt](https://github.com/linxuewu/Sparse4D/releases/download/v0.0/sparse4dv2_r50_HInf_256x704.pth)|[log](https://github.com/linxuewu/Sparse4D/releases/download/v0.0/sparse4dv2_r50_HInf_256x704.log.json)|
|Sparse4Dv2|Res101|[nuImage](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r101_fpn_1x_nuim/cascade_mask_rcnn_r101_fpn_1x_nuim_20201024_134804-45215b1e.pth)|512x1408| 100 |2Day | 8.4 |0.5939|0.5051|-|-|-|-|-|-|
|Sparse4Dv3|Res50|[ImageNet]()|256x704| 100 |22H | 19.8 |0.5637|0.4646|0.477|1.167|456|[cfg](https://github.com/HorizonRobotics/Sparse4D/blob/main/projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py)|[ckpt](https://github.com/HorizonRobotics/Sparse4D/releases/download/v3.0/sparse4dv3_r50.pth)|[log](https://github.com/HorizonRobotics/Sparse4D/releases/download/v3.0/sparse4dv3_r50.log)
|Sparse4Dv3|Res101|[nuImage](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r101_fpn_1x_nuim/cascade_mask_rcnn_r101_fpn_1x_nuim_20201024_134804-45215b1e.pth)|512x1408| 100 |2Day | 8.2 |0.623|0.537|0.567|1.027|557|-|-|-|

### Results on Test Split
|model| backbone | img size | NDS | mAP |mATE| mASE | mAOE |mAVE| mAAE | AMOTA |AMOTP |IDS|
| :---: | :---: | :---: | :---: | :---:|:---:|:---: | :---: | :----: | :----: | :----: | :----: | :----: |
|Sparse4D-T4|[VoV-99](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_L_coco_det_sys_o365.pth)|640x1600|0.595|0.511|0.533|0.263|0.369|0.317|0.124|-|-|-|
|Sparse4Dv2|[VoV-99](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_L_coco_det_sys_o365.pth)|640x1600|0.638|0.556|0.462|0.238|0.328|0.264|0.115|-|-|-|
|Sparse4Dv3|[VoV-99](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_L_coco_det_sys_o365.pth)|640x1600|0.656|0.570|0.412|0.236|0.312|0.210|0.117|0.574|0.970|669|
|Sparse4Dv3-offline|[EVA02-large](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_L_coco_det_sys_o365.pth)|640x1600|0.719|0.668|0.346|0.234|0.279|0.142|0.145|0.677|0.761|514|

PS: In the nuscenes leaderboard, Sparse4Dv3 selected external data=True because the eva02-large pretraining utilized imagenet, object365, and coco, as well as supervised by CLIP. Therefore, we consider using the model pre-trained with eva02 as incorporating external data. **However, we did not use external 3D detection data for training.** This clarification is provided to facilitate fair comparisons.


## Quick Start
[Quick Start](docs/quick_start.md)

## Citation
```
@misc{2311.11722,
    Author = {Xuewu Lin and Zixiang Pei and Tianwei Lin and Lichao Huang and Zhizhong Su},
    Title = {Sparse4D v3: Advancing End-to-End 3D Detection and Tracking},
    Year = {2023},
    Eprint = {arXiv:2311.11722},
}
@misc{2305.14018,
    Author = {Xuewu Lin and Tianwei Lin and Zixiang Pei and Lichao Huang and Zhizhong Su},
    Title = {Sparse4D v2: Recurrent Temporal Fusion with Sparse Model},
    Year = {2023},
    Eprint = {arXiv:2305.14018},
}
@misc{2211.10581,
    Author = {Xuewu Lin and Tianwei Lin and Zixiang Pei and Lichao Huang and Zhizhong Su},
    Title = {Sparse4D: Multi-view 3D Object Detection with Sparse Spatial-Temporal Fusion},
    Year = {2022},
    Eprint = {arXiv:2211.10581},
}
```

## Acknowledgement
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [DETR3D](https://github.com/WangYueFt/detr3d) 
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)
- [SOLOFusion](https://github.com/Divadi/SOLOFusion/tree/main/configs/solofusion)
- [StreamPETR](https://github.com/exiawsh/StreamPETR)
