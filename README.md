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

## 로컬 연구 메모: Object-Query 중심 Sparse Multi-view Ray Matching

이 fork는 KB20/SphereStream 계열 4-fisheye 3D detection 실험을 위한 작업 공간이다. 현재 연구 방향은 다음과 같이 둔다.

> Object-Query 중심 Sparse Multi-view Ray Matching for 4-Fisheye Temporal 3D Detection

핵심 문제의식은 fisheye query가 단순히 "3D query가 이미지 어디에 투영되는가"만 보는 것이 아니라, "visible overlap view들 중 어떤 관측이 이 query의 3D 위치를 가장 잘 지지하는가"를 함께 판단해야 한다는 점이다. 현재 방향은 object 전체를 촘촘한 graph node로 투영하는 방식이 아니다. 먼저 object center 한 점을 기준으로 sparse ray-depth 후보를 만들고, visible fisheye view들의 feature 일치도와 투영 품질을 이용해 query의 depth와 update 신뢰도를 안정화한다.

### 제안 모듈

| 모듈 | 목적 | 참고 문헌 | 본 연구의 차이 |
| --- | --- | --- | --- |
| Object-Center Ray Candidate Sampling | object query의 중심점 하나를 기준으로 reference ray를 만들고, 그 ray 위에 sparse depth 후보를 생성한다. front/back, left/right, box edge 등 여러 graph node는 첫 단계에서 쓰지 않는다. | Sparse4D, ORA3D | dense overlap disparity나 box 전체 graph sampling이 아니라 object center query의 depth 후보만 sparse하게 검증한다. |
| Visible-View Support Scoring | 각 depth 후보가 visible fisheye view들에서 얼마나 잘 지지되는지 계산한다. 신뢰도는 feature matching 신뢰도와 투영 품질 신뢰도를 곱해서 만든다. | ORA3D, MIC-BEV, FishBEV, Fisheye3DOD | 모든 view를 동일하게 평균내지 않고, query의 3D 위치를 가장 잘 지지하는 visible 관측을 더 강하게 반영한다. |
| Reliability-Guided Temporal 3D Position Update | matching 기반 신뢰도를 이용해 매 시점 3D 위치/depth update를 안정화한다. 별도의 tracking 장치를 새로 만드는 것이 아니라 temporal 3D 위치 추정 안정성에 집중한다. | Sparse4D v3, StreamPETR, UA-Track | 추적 성능은 부가 결과로 볼 수 있지만, 주된 목표는 post-hoc association이 아니라 query의 시간적 3D 위치 추정 안정화다. |

신뢰도 결합은 다음 형태를 기본으로 둔다.

```text
reliability(d_k) = matching_confidence(d_k) * projection_quality(d_k)
```

- `matching_confidence(d_k)`: visible view들에서 샘플링한 feature consistency. 첫 버전은 reference-to-visible cosine similarity 평균으로 계산한다.
- `projection_quality(d_k)`: image bound, fisheye FOV, projected footprint, image-center distance, truncation, invalid region 여부로 계산한다.
- 유효한 visible view가 두 개 미만이면 multi-view matching 대상에서 제외한다.

Temporal depth 시작값은 object 상태에 따라 구분한다.

```text
기존 object query: 이전 frame의 depth / temporal prior 주변에서 후보 생성
신규 object query: detector baseline depth 주변에서 후보 생성
oracle 기초 검증: GT depth 주변에서 후보 생성
```

첫 실험은 oracle 기초 검증이므로 GT depth 주변 후보를 사용한다. 이 단계가 통과되면 실제 적용 단계에서 baseline predicted depth 또는 temporal-prior depth 주변 후보로 바꾼다.

### 읽을 논문 우선순위

| 우선순위 | 논문 | 필요한 이유 |
| ---: | --- | --- |
| 1 | ORA3D | dense overlap disparity/depth supervision과 본 연구의 object-query sparse ray matching 차이를 명확히 대비하기 위해 필요하다. |
| 2 | Sparse4D v3 | object query, temporal instance bank, query update 흐름의 직접적인 기반이다. |
| 3 | MIC-BEV | 여러 camera/view 관측을 동일하게 보지 않고 중요도를 다르게 두는 논리를 방어하는 데 필요하다. |
| 4 | FishBEV / Fisheye3DOD / DaF-BEVSeg | fisheye projection, view 품질, 주변부 왜곡 문제를 설명하는 배경으로 사용한다. |
| 5 | UA-Track | uncertainty-aware temporal 3D 추정과 대비한다. 단, 본 연구는 별도 tracking association보다 query depth/update reliability에 집중한다. |
| 6 | Graph-DETR3D | graph node aggregation은 이번 1차 방법의 핵심이 아니므로 참고 수준으로 둔다. 여러 node 확장은 후속 ablation 후보로만 남긴다. |

### 현재 ablation 메모

현재 로컬 validation 결과에서는 velocity label과 class 구성이 모두 confounder로 보인다.

| Run | Classes | Velocity labels | Best mAP | Final mAP | Final NDS | Final mAVE | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `sparse4d_kb20_car_ped_100e_github_r50_b8` | Car, Pedestrian | zero-valued | 0.4371 @ epoch 25 | 0.3952 | 0.4515 | 0.0028 | 현재 가장 강한 2-class baseline. |
| `sparse4d_kb20_car_ped_cyc_100e_github_r50_b8` | Car, Pedestrian, Cyclist | real | 0.2807 @ epoch 25 | 0.2253 | 0.2216 | 0.9601 | real velocity가 NDS/mAVE를 크게 악화한다. Cyclist도 약하다. |
| `sparse4d_kb20_car_ped_cyc_zerovel_30e_github_r50_b8` | Car, Pedestrian, Cyclist | zero-valued | 0.2719 @ epoch 20 | 0.2447 | 0.3446 | 0.0033 | zero velocity로 mAVE/NDS는 회복되지만 2-class mAP까지는 회복되지 않는다. |

해석: real velocity는 NDS/mAVE에 큰 영향을 주는 label/metric 이슈다. 하지만 3-class zero-velocity run도 2-class baseline보다 낮으므로 Cyclist 난이도, data balance, fisheye view-quality 문제가 여전히 남아 있다. sparse multi-view ray matching 계열 실험은 velocity ablation과 독립적인 축이며, label/class confounder를 통제한 뒤 평가하는 것이 맞다.

### 현재 실험: Sparse Multi-view Ray Matching 기초 검증

다음 실험은 4-fisheye 환경에서 object-query-level sparse multi-view ray matching이 성립하는지 보는 oracle 기반 기초 검증 실험이다. 핵심 질문은 다음이다.

> GT object center 주변의 depth 후보를 visible fisheye view들에 재투영했을 때, multi-view feature consistency score가 실제 depth 근처에서 peak를 만드는가?

핵심 가설은 4-fisheye overlap이 단순 중복 관측이 아니라는 점이다. 각 object query에 대해 여러 겹침 fisheye view가 서로 다른 후보 3D 위치를 서로 다른 신뢰도로 지지할 수 있다. 따라서 모든 view를 동일하게 평균내지 않고, projection quality와 feature consistency를 이용해 query의 3D 위치를 가장 잘 지지하는 visible 관측을 더 강하게 반영해야 한다.

이 실험은 ORA3D식 dense overlap supervision과 다르다. ORA3D는 pinhole overlap 영역 전체에서 dense pixel-level disparity/depth cue를 사용한다. 우리는 먼저 4-fisheye 환경에서 object-query center 기반 sparse ray matching만 검증한다.

```text
dense full-image overlap matching: no
object-query visible fisheye views: yes
object center ray only: yes
multi-node box graph: no
```

초기 절차:

1. 두 개 이상 camera에 보이는 GT object를 고른다.
2. projected area, image-center distance, truncation, projection validity를 기준으로 object를 가장 잘 보는 oracle reference camera를 선택한다.
3. GT object center를 reference camera에 projection하고, 해당 fisheye pixel을 inverse projection해서 3D ray를 만든다.
4. GT depth 주변 후보를 만든다. 예: `d_gt +/- 2m`, `K=21`.
5. 각 후보 3D point를 네 개 fisheye camera에 모두 재투영한다.
6. image bound, fisheye FOV, projection-quality filter를 통과한 visible view만 남긴다. image 밖, lens cone 밖, 심한 truncation, invalid region의 view는 matching score에 넣지 않는다.
7. valid view에서 backbone/neck/object-region feature를 sampling한다.
8. 먼저 reference-to-visible cosine similarity로 후보 depth의 matching confidence를 계산한다.
9. projection quality와 matching confidence를 곱해 최종 reliability curve를 만든다.

```text
matching_confidence(d_k) = mean_c cosine(f_ref(d_k), f_c(d_k))
reliability(d_k) = matching_confidence(d_k) * projection_quality(d_k)
```

이후 variant에서는 visible view all-pair similarity도 비교한다.

```text
matching_confidence(d_k) = mean_{i,j} cosine(f_i(d_k), f_j(d_k))
```

성공 기준은 `d_gt` 근처에서 reliability curve가 뚜렷한 local peak를 보이는 것이다. 모든 depth 후보의 score가 비슷하면 feature 해상도가 낮거나, object가 너무 작거나, view 간 appearance 차이가 크거나, reference view 선택이 나쁘거나, sampling point가 object가 아니라 background를 찍는 문제가 있을 수 있다.

구현은 두 층으로 나눈다. 실제 모델 로직은 `projects/mmdet3d_plugin/` 안에 넣고, 그 모델을 실행하거나 중간 결과를 뽑는 바깥 실험 스크립트만 `tools/research/`에 모은다. 즉 `tools/research/`는 method를 구현하는 위치가 아니라, 실험 실행/검증 entrypoint를 한곳에 모아두는 위치다.

```text
tools/research/ray_matching_oracle_check.py
```

이 스크립트는 `projects/mmdet3d_plugin`의 config/model/dataloader를 호출해 object별 depth 후보 curve를 dump한다. 초기 oracle 검증은 외부 스크립트에서 시작할 수 있지만, 실제 방법 구현 단계에서는 ray 후보 생성, visible-view scoring, reliability-guided update 로직을 `projects/mmdet3d_plugin` 안의 모델 코드로 옮긴다. 결과는 아래에 저장한다.

```text
work_dirs/_ray_matching_oracle_check/
```

첫 버전에서는 새 학습을 하지 않는다. 목표는 GT-centered sparse multi-view ray matching이 의미 있는 reliability curve를 만드는지 확인하는 것이다. 이 기초 검증이 통과되면 다음 단계에서 oracle GT depth 후보를 baseline predicted depth 또는 temporal-prior depth 후보로 바꾸고, matching score를 query depth/update reliability에 연결한다.

코드 배치 원칙은 다음과 같다.

```text
projects/mmdet3d_plugin/models/projection.py        # fisheye projection / inverse ray / projection validity
projects/mmdet3d_plugin/models/ray_matching.py      # depth 후보, visible-view matching, reliability 계산
projects/mmdet3d_plugin/models/blocks.py            # 기존 feature sampling/fusion에 reliability weight 연결
projects/mmdet3d_plugin/models/sparse4d_head.py     # query depth/update에 reliability 반영
projects/mmdet3d_plugin/datasets/transforms/        # camera/GT/meta packing 보강
configs/                                            # ablation별 설정
tools/research/                                    # 위 모델을 호출하는 실험 실행 스크립트
```

따라서 앞으로 모델을 많이 수정해야 하는 것은 맞다. 다만 그 코드는 `tools/research/`에 넣지 않고, `projects/mmdet3d_plugin/`에 실제 모듈로 넣는다. `tools/research/`는 실행 스크립트가 흩어지지 않게 모으는 역할만 한다.

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
