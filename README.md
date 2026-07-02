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

## 로컬 연구 메모: Fisheye-Aware Graph Query Aggregation

이 fork는 KB20/SphereStream 계열 4-fisheye 3D MOT 실험을 위한 작업 공간이다. 현재 연구 방향은 다음과 같이 둔다.

> Fisheye-Aware Graph Query Aggregation for 4-Fisheye 3D MOT

핵심 문제의식은 fisheye query가 단순히 "3D query가 이미지 어디에 투영되는가"만 보는 것이 아니라, "어떤 view/node 관측이 이 query의 3D 위치를 가장 잘 지지하는가"를 함께 판단해야 한다는 점이다. 따라서 방법은 단순한 geometry MLP 추가가 아니라, object query 중심의 visible multi-view 관측을 신뢰도에 따라 다르게 반영하는 구조로 정리한다.

### 제안 모듈

| 모듈 | 목적 | 참고 문헌 | 본 연구의 차이 |
| --- | --- | --- | --- |
| Fisheye Graph Node Sampling | object query/box 주변에 고정된 3D graph node를 만든다. center만 쓰지 않고 front/back, left/right, top/bottom, box edge sample 등을 후보로 둔다. | Graph-DETR3D, Sparse4D, Graph-DETR4D 확인 필요 | 짧은 주기 구현에서는 고정 object/box node를 먼저 쓰고, fisheye projection 품질을 평가한다. |
| Node-View Reliability Estimator | 각 object query x graph node x camera view별 신뢰도를 추정한다. 입력 후보는 projection validity, projected footprint, radial distance, truncation ratio, visible view 수, best-view 대비 area ratio 등이다. | Fisheye3DOD, FishBEV, DaF-BEVSeg/FisheyeBEVSeg, MIC-BEV | camera-level 또는 BEV-cell-level importance가 아니라 object graph node-view reliability로 옮긴다. |
| Reliability-Guided Temporal Query Update | view/node 신뢰도를 이용해 graph aggregation과 temporal query update를 조절한다. 좋지 않은 fisheye 관측이 temporal query persistence와 tracking을 흔들지 않게 하는 것이 목적이다. | Sparse4D v3, StreamPETR, UA-Track, Graph-DETR4D 확인 필요 | post-hoc tracking association cost가 아니라 temporal query update 자체에 집중한다. |

### 읽을 논문 우선순위

| 우선순위 | 논문 | 필요한 이유 |
| ---: | --- | --- |
| 1 | Graph-DETR3D | DETR3D-style query 주변 graph node feature aggregation의 직접적인 기반이다. 특히 border/overlap 영역 논리에 필요하다. |
| 2 | Graph-DETR4D | temporal graph extension을 방어할 때 유용할 수 있다. 실제 인용 가능한 source 확인 필요. |
| 3 | MIC-BEV | camera/view importance 관련 방어 논문이다. infrastructure camera의 BEV-cell importance와 대비할 수 있다. |
| 4 | UA-Track | uncertainty-aware 3D MOT 방어 논문이다. object-state uncertainty/query denoising과 대비한다. |
| 5 | ORA3D | overlap-region/depth-prior 처리 관련 방어 논문이다. 본 연구는 dense overlap disparity가 아니라 fisheye object-query sparse matching이라는 점을 대비한다. |

### 현재 ablation 메모

현재 로컬 validation 결과에서는 velocity label과 class 구성이 모두 confounder로 보인다.

| Run | Classes | Velocity labels | Best mAP | Final mAP | Final NDS | Final mAVE | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `sparse4d_kb20_car_ped_100e_github_r50_b8` | Car, Pedestrian | zero-valued | 0.4371 @ epoch 25 | 0.3952 | 0.4515 | 0.0028 | 현재 가장 강한 2-class baseline. |
| `sparse4d_kb20_car_ped_cyc_100e_github_r50_b8` | Car, Pedestrian, Cyclist | real | 0.2807 @ epoch 25 | 0.2253 | 0.2216 | 0.9601 | real velocity가 NDS/mAVE를 크게 악화한다. Cyclist도 약하다. |
| `sparse4d_kb20_car_ped_cyc_zerovel_30e_github_r50_b8` | Car, Pedestrian, Cyclist | zero-valued | 0.2719 @ epoch 20 | 0.2447 | 0.3446 | 0.0033 | zero velocity로 mAVE/NDS는 회복되지만 2-class mAP까지는 회복되지 않는다. |

해석: real velocity는 NDS/mAVE에 큰 영향을 주는 label/metric 이슈다. 하지만 3-class zero-velocity run도 2-class baseline보다 낮으므로 Cyclist 난이도, data balance, fisheye view-quality 문제가 여전히 남아 있다. node-view reliability 계열 모듈은 velocity ablation과 독립적인 축이며, label/class confounder를 통제한 뒤 평가하는 것이 맞다.

### 현재 실험: Sparse Multi-view Ray Matching

다음 실험은 4-fisheye 환경에서 object-query-level sparse multi-view ray matching이 성립하는지 보는 oracle sanity check다. 핵심 질문은 다음이다.

> GT object center 주변의 depth 후보를 visible fisheye view들에 재투영했을 때, multi-view feature consistency score가 실제 depth 근처에서 peak를 만드는가?

핵심 가설은 4-fisheye overlap이 단순 중복 관측이 아니라는 점이다. 각 object query에 대해 여러 겹침 fisheye view가 서로 다른 후보 3D 위치를 서로 다른 신뢰도로 지지할 수 있다. 따라서 모든 view를 동일하게 평균내지 않고, projection quality, footprint, truncation, feature consistency를 이용해 query의 3D 위치를 가장 잘 지지하는 visible 관측을 더 강하게 반영해야 한다.

이 실험은 ORA3D식 dense overlap supervision과 다르다. ORA3D는 pinhole overlap 영역 전체에서 dense pixel-level disparity/depth cue를 사용한다. 우리는 먼저 4-fisheye 환경에서 object-query-level sparse ray matching만 검증한다.

```text
dense full-image overlap matching: no
object-query visible fisheye views: yes
```

초기 절차:

1. 두 개 이상 camera에 보이는 GT object를 고른다.
2. projected area, image-center distance, truncation, projection validity를 기준으로 object를 가장 잘 보는 oracle reference camera를 선택한다.
3. GT object center를 reference camera에 projection하고, 해당 fisheye pixel을 inverse projection해서 3D ray를 만든다.
4. GT depth 주변 후보를 만든다. 예: `d_gt +/- 2m`, `K=21`.
5. 각 후보 3D point를 네 개 fisheye camera에 모두 재투영한다.
6. image bound, fisheye FOV, projection-quality filter를 통과한 visible view만 남긴다. image 밖, lens cone 밖, 심한 truncation, invalid region의 view는 matching score에 넣지 않는다.
7. valid view에서 backbone/neck/object-region feature를 sampling한다.
8. 먼저 reference-to-visible cosine similarity로 후보 depth score를 계산한다.

```text
score(d_k) = mean_c cosine(f_ref(d_k), f_c(d_k))
```

이후 variant에서는 visible view all-pair similarity도 비교한다.

```text
score(d_k) = mean_{i,j} cosine(f_i(d_k), f_j(d_k))
```

성공 기준은 `d_gt` 근처에서 score curve가 뚜렷한 local peak를 보이는 것이다. 모든 depth 후보의 score가 비슷하면 feature 해상도가 낮거나, object가 너무 작거나, view 간 appearance 차이가 크거나, reference view 선택이 나쁘거나, sampling point가 object가 아니라 background를 찍는 문제가 있을 수 있다.

구현은 training loop 안에 바로 넣지 않고, `tools/` 아래 독립 진단 스크립트로 시작한다.

```text
tools/ray_matching_sanity_check.py
```

이 스크립트는 `projects/mmdet3d_plugin`의 기존 config/model/dataloader 경로를 재사용하고, 학습된 checkpoint에서 feature를 추출해 object별 cost curve를 dump한다. 결과는 아래에 저장한다.

```text
work_dirs/_ray_matching_sanity/
```

첫 버전에서는 새 학습을 하지 않는다. 목표는 GT-centered sparse multi-view ray matching이 의미 있는 cost curve를 만드는지 확인하는 것이다. 이 sanity check가 통과되면 다음 단계에서 oracle GT depth 후보를 baseline predicted depth 또는 temporal-prior depth 후보로 바꾸고, matching score를 query depth/update reliability에 연결한다.

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
