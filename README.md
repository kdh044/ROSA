# SphereStream Sparse4D 연구 작업공간

이 repository는 HorizonRobotics Sparse4D를 기반으로 KB20/SphereStream 계열 4-fisheye temporal 3D detection 실험을 진행하기 위한 private fork다.

Upstream 참고:

- Sparse4D 원본: https://github.com/HorizonRobotics/Sparse4D
- Sparse4D v3 paper: https://arxiv.org/abs/2311.11722

## 최종 연구 정의

> 4-fisheye temporal 3D detection에서 fisheye ray geometry와 object-level overlap supervision을 이용해 query feature를 range-aware하게 만들고, projection quality로 visible view/keypoint support를 조절하는 연구.

핵심 문제의식은 fisheye query가 단순히 "3D query가 이미지 어디에 투영되는가"만 보는 것이 아니라, "visible overlap view들 중 어떤 관측이 이 query를 가장 안정적으로 지지하는가"를 함께 판단해야 한다는 점이다.

현재까지의 GT 기준 range 후보 검증에서는 feature matching score가 정답 range를 충분히 안정적으로 고르지 못했다. 따라서 "sparse ray matching으로 depth/range를 직접 찾거나 보정한다"는 강한 주장은 메인 claim으로 두지 않는다. Range 후보 실험은 보조 ablation으로 남기고, 주된 방향은 fisheye visible view support weighting이다.

단순히 `query feature -> range bin auxiliary loss` 하나를 붙이는 것은 novelty가 약하다. 본 연구의 핵심은 range head 자체가 아니라, 4-fisheye overlap에서 어떤 object/query에 range auxiliary를 줄지, 어떤 view/keypoint를 더 믿을지, fisheye ray geometry를 어떻게 feature에 넣을지를 함께 설계하는 것이다.

우리가 쓰는 신호는 dense depth map이 아니다. `MultiScaleDepthMapGenerator`처럼 LiDAR surface point를 이미지 전체에 투영해 픽셀별 depth map을 만드는 방식과 구분한다. 본 연구에서의 target은 object query / sparse keypoint가 각 fisheye view에서 얼마나 유효하고 안정적인 관측을 갖는지 평가하는 view support signal이며, 학습에서는 object/query 단위 range-aware auxiliary를 제한적으로 사용한다.

```text
dense pixel depth map target: no  # 이미지 전체 픽셀 깊이 지도 아님
2D bbox center target: no         # 2D 박스 중심 아님
direct depth/range selection claim: no
range aux only as novelty: no
fisheye ray geometry PE: yes
object-level overlap supervision: yes
query/keypoint visible-view support: yes
projection-quality reliability signal: yes
temporal query update weighting: yes
```

현재 방향은 object 전체를 dense하게 matching하거나 graph node를 촘촘히 뿌리는 방식이 아니다. Sparse4D가 이미 사용하는 sparse box keypoint를 4개 fisheye view에 투영하고, 각 view/keypoint가 image bound, fisheye FOV, 중심부 거리, projected footprint, truncation, feature support 측면에서 얼마나 믿을 만한지 계산한다. 이 support로 feature aggregation과 temporal query update를 안정화한다.

## 제안 모듈

| 모듈 | 목적 | 본 연구의 차이 |
| --- | --- | --- |
| Fisheye Ray Geometry Encoding | 각 pixel/keypoint가 어떤 fisheye ray, incidence angle, radial distortion 영역에 있는지 feature에 주입한다. | 일반 image coordinate PE가 아니라 fisheye ray geometry를 명시한다. |
| Object-level Overlap Range Auxiliary | 두 개 이상 fisheye view에 보이는 object/query만 골라 range-aware auxiliary loss를 준다. | ORA3D식 full-overlap dense disparity가 아니라 object/query 단위 overlap supervision이다. |
| Projection-quality Weighted Supervision | projected area, image-center distance, truncation, FOV validity 등으로 range auxiliary와 view aggregation weight를 조절한다. | 모든 view/loss를 동일하게 쓰지 않고, 좋은 관측을 더 강하게 반영한다. |
| Reliability-Guided Temporal Query Update | support가 높은 view feature를 더 강하게 쓰고, support가 낮은 query는 temporal prior를 더 보존한다. | depth를 직접 찾는 것이 아니라 temporal query update의 신뢰도를 조절한다. |

## 핵심 contribution

최종 contribution은 다음 네 가지로 둔다.

```text
1. fisheye ray geometry-aware PE
2. object-level overlap selection
3. projection-quality weighted range auxiliary
4. visible view support를 temporal query update에 반영
```

신뢰도와 auxiliary loss는 다음 형태를 기본으로 둔다.

```text
support_c = feature_support_c * projection_quality_c
query_feature = sum_c support_c * view_feature_c
L_range = projection_quality_c * CE(P(range_bin), range_gt_bin)
```

- `feature_support_c`: view c에서 query/keypoint feature가 얼마나 안정적으로 관측되는지.
- `projection_quality_c`: image bound, fisheye FOV, projected footprint, image-center distance, truncation, radial distortion strength, invalid region 여부.
- `range_gt_bin`: GT 3D box 또는 matched query의 object-level range bin. Dense pixel depth가 아니다.
- 유효한 visible view가 두 개 미만이면 overlap range auxiliary 대상에서 제외한다.

Projection quality와 range target은 GT 기준 실험과 실제 inference/training에서 사용하는 box가 다르다.

```text
GT 기준 실험: GT box를 이용해 projected area / truncation / visibility 계산
실제 적용: predicted query box 또는 anchor box를 이용해 projection quality 계산
```

Range 후보 실험은 보조 ablation으로만 유지한다.

```text
main: 현재 query 위치의 visible view/keypoint support 계산
main training: object-level overlap range auxiliary + projection-quality weighting
ablation: GT range 주변 sparse keypoint range search
hold: feature score로 direct depth/range 선택
```

## 현재 실험

현재까지 수행한 것은 학습이 아니라 GT 기준 기초 검증이다.

> GT object 주변 range 후보를 visible fisheye view들에 재투영했을 때, feature score가 실제 range 근처에서 충분히 강한 peak를 만드는가?

결과는 direct depth/range refinement claim을 지지하기에 부족하다.

```text
center one-point level0:
정답 +/-0.4m 근처 최고점: 13.1%
경계 후보 최고점: 59.0%

sparse keypoint level0:
정답 +/-0.4m 근처 최고점: 28.7%
경계 후보 최고점: 23.0%

sparse keypoint level0 + z_center 수정:
정답 +/-0.4m 근처 최고점: 35.2%
경계 후보 최고점: 18.0%
```

따라서 `sparse keypoint`는 center 한 점보다 낫지만, "정답 range를 직접 고른다"는 메인 주장으로는 약하다. 이후 실험은 range search를 보조 ablation으로 두고, visible view support weighting을 주 방향으로 둔다.

ORA3D와의 차이:

```text
dense full-image overlap matching: no
object-query visible fisheye views: yes
sparse keypoint visible-view support: yes
object-level range auxiliary: yes
ray geometry PE: yes
direct depth/range selection: no
```

## ORA3D / pseudo disparity 관련 판단

Raw fisheye image에서 ORA3D식 dense pseudo disparity를 그대로 만드는 것은 메인 방향으로 두지 않는다. ORA3D의 핵심은 overlap region feature에 depth-aware auxiliary supervision을 주는 철학이지, pseudo disparity를 완전한 GT로 믿는 것이 아니다.

Fisheye에서는 pinhole/rectified stereo와 달리 대응점이 직선 scanline이 아니라 curved epipolar path를 따라 움직인다. 따라서 raw fisheye overlap 전체에 traditional stereo를 바로 적용해 dense pseudo depth를 만들면 주변부 왜곡, occlusion, scale 변화, view 간 appearance 차이 때문에 noise가 커질 수 있다.

```text
raw fisheye dense pseudo disparity as GT: no
confidence-filtered weak auxiliary: optional
object/query-level range auxiliary: safer
ray / spherical / camera geometry PE: yes
```

우리 데이터에는 3D box GT가 있으므로 dense pixel pseudo depth를 억지로 만들기보다 object/query 단위의 sparse keypoint projection quality와 visible view support를 우선 사용한다. Pseudo disparity를 쓰더라도 confidence가 높은 영역만 보조 신호로 제한하고, 주변부/occlusion/boundary/robot-body region은 mask out해야 한다.

남은 검증:

1. `kb_lens_fov_deg=360`처럼 너무 느슨한 FOV mask를 실제 렌즈 FOV 근처로 제한한다.
2. GT bbox의 `z_bottom`을 geometric center로 변환한 기준을 유지한다.
3. GT box/keypoint overlay로 projection/extrinsic이 실제 이미지에 맞는지 계속 확인한다.
4. Range search가 충분히 좋아지지 않으면 direct depth refinement는 폐기하고 view support weighting만 유지한다.
5. Baseline ablation을 `range aux only -> ray PE -> object-overlap mask -> projection-quality weighting` 순서로 구성한다.


## Feature 샘플링 shape

View support / range-search ablation에서 맞춰야 할 tensor shape는 다음과 같이 둔다.

```text
feature_maps[l]           = (B, N_cam, C_l, H_l, W_l)
query/keypoint 3D points  = (B, N_obj, N_key, 3)
projection 결과           = (B, N_cam, N_obj, N_key, 2)
valid view/keypoint mask  = (B, N_cam, N_obj, N_key)
sampled feature           = (B, N_cam, N_obj, N_key, C_l)
projection_quality        = (B, N_cam, N_obj, N_key)
view_support              = (B, N_cam, N_obj)
query_support             = (B, N_obj)
range_logits              = (B, N_obj, N_bins)       # 학습 auxiliary
range_aux_weight          = (B, N_obj)
```

Range-search ablation에서는 여기에 후보 축 `K`가 추가된다.

```text
candidate keypoints       = (B, N_obj, K, N_key, 3)
candidate projection      = (B, N_cam, N_obj, K, N_key, 2)
candidate reliability     = (B, N_obj, K)
```

## 코드 배치

실제 모델 로직은 `projects/mmdet3d_plugin/` 안에 넣고, 그 모델을 실행하거나 중간 결과를 뽑는 바깥 실험 스크립트만 `tools/research/`에 모은다.

```text
projects/mmdet3d_plugin/models/projection.py        # fisheye projection / projection validity
projects/mmdet3d_plugin/models/ray_matching.py      # range-search ablation / visible-view support helper
projects/mmdet3d_plugin/models/positional_encoding.py # ray PE / camera geometry encoding 후보
projects/mmdet3d_plugin/models/blocks.py            # 기존 feature sampling/fusion에 reliability weight 연결
projects/mmdet3d_plugin/models/sparse4d_head.py     # temporal query update에 reliability 반영
projects/mmdet3d_plugin/datasets/transforms/        # camera/GT/meta packing 보강
configs/                                            # ablation별 설정
tools/research/                                    # 위 모델을 호출하는 실험 실행 스크립트
```

따라서 앞으로 모델을 많이 수정해야 하는 것은 맞다. 다만 그 코드는 `tools/research/`에 넣지 않고, `projects/mmdet3d_plugin/`에 실제 모듈로 넣는다. `tools/research/`는 실행 스크립트가 흩어지지 않게 모으는 역할만 한다.

현재 예정 실행 스크립트:

```text
tools/research/ray_range_gt_check.py
```

결과 저장 위치:

```text
work_dirs/_ray_range_gt_check/
```

## 지금 볼 코드

우선 계속 봐야 할 곳은 `projects/mmdet3d_plugin/models/`가 맞다. 다만 config와 metadata packing은 같이 확인해야 한다.

| 우선순위 | 파일 | 볼 내용 |
| ---: | --- | --- |
| 1 | `projects/mmdet3d_plugin/models/projection.py` | KB fisheye projection, FOV/image-bound validity, projection quality |
| 2 | `projects/mmdet3d_plugin/models/ray_matching.py` | range-search ablation, sparse keypoint support scoring helper |
| 3 | `projects/mmdet3d_plugin/models/blocks.py` | feature sampling/fusion에 projection-quality weight를 연결할 지점 |
| 4 | `projects/mmdet3d_plugin/models/sparse4d_head.py` | range auxiliary head와 temporal query update reliability를 반영할 지점 |
| 5 | `projects/mmdet3d_plugin/datasets/transforms/pack_sparse4d.py` | camera matrix, fisheye params, GT/meta가 모델로 들어오는 흐름 |
| 6 | `configs/` | `range aux only`, `ray PE`, `overlap mask`, `quality weighting` ablation 설정 |

## 현재 ablation 메모

현재 로컬 validation 결과에서는 velocity label과 class 구성이 모두 confounder로 보인다.

| Run | Classes | Velocity labels | Best mAP | Final mAP | Final NDS | Final mAVE | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `sparse4d_kb20_car_ped_100e_github_r50_b8` | Car, Pedestrian | zero-valued | 0.4371 @ epoch 25 | 0.3952 | 0.4515 | 0.0028 | 현재 가장 강한 2-class baseline. |
| `sparse4d_kb20_car_ped_cyc_100e_github_r50_b8` | Car, Pedestrian, Cyclist | real | 0.2807 @ epoch 25 | 0.2253 | 0.2216 | 0.9601 | real velocity가 NDS/mAVE를 크게 악화한다. Cyclist도 약하다. |
| `sparse4d_kb20_car_ped_cyc_zerovel_30e_github_r50_b8` | Car, Pedestrian, Cyclist | zero-valued | 0.2719 @ epoch 20 | 0.2447 | 0.3446 | 0.0033 | zero velocity로 mAVE/NDS는 회복되지만 2-class mAP까지는 회복되지 않는다. |

해석: real velocity는 NDS/mAVE에 큰 영향을 주는 label/metric 이슈다. 하지만 3-class zero-velocity run도 2-class baseline보다 낮으므로 Cyclist 난이도, data balance, fisheye view-quality 문제가 여전히 남아 있다. sparse multi-view ray matching 계열 실험은 velocity ablation과 독립적인 축이며, label/class confounder를 통제한 뒤 평가하는 것이 맞다.

## 관련 논문 우선순위

| 우선순위 | 논문 | 필요한 이유 |
| ---: | --- | --- |
| 1 | ORA3D | dense overlap disparity/depth supervision과 본 연구의 object-level overlap range auxiliary 차이를 대비한다. |
| 2 | Sparse4D v3 | object query, temporal instance bank, query update 흐름의 직접적인 기반이다. |
| 3 | MIC-BEV | 여러 camera/view 관측을 동일하게 보지 않고 projection-quality support로 중요도를 다르게 두는 논리를 방어한다. |
| 4 | FishBEV / Fisheye3DOD / DaF-BEVSeg | fisheye projection, view 품질, 주변부 왜곡 문제를 설명하는 배경으로 사용한다. |
| 5 | UA-Track | uncertainty-aware temporal 3D 추정과 대비한다. |
| 6 | Graph-DETR3D | 이번 1차 방법의 핵심은 아니며, 여러 node 확장 ablation 후보로만 남긴다. |
