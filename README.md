# SphereStream Sparse4D 연구 작업공간

이 repository는 HorizonRobotics Sparse4D를 기반으로 KB20/SphereStream 계열 4-fisheye temporal 3D detection 실험을 진행하기 위한 private fork다.

Upstream 참고:

- Sparse4D 원본: https://github.com/HorizonRobotics/Sparse4D
- Sparse4D v3 paper: https://arxiv.org/abs/2311.11722

## 최종 연구 정의

> 4-fisheye 환경에서 object query 중심 ray-depth 후보를 visible overlap view들에 재투영하고, feature 일치도와 어안 투영 품질로 각 후보의 지지도를 계산해 temporal query의 3D 위치를 보정하는 연구.

핵심 문제의식은 fisheye query가 단순히 "3D query가 이미지 어디에 투영되는가"만 보는 것이 아니라, "visible overlap view들 중 어떤 관측이 이 query의 3D 위치를 가장 잘 지지하는가"를 함께 판단해야 한다는 점이다.

현재 방향은 object 전체를 촘촘한 graph node로 투영하는 방식이 아니다. 먼저 object center 한 점을 기준으로 reference camera ray를 만들고, 그 ray 방향의 거리/range 후보를 visible fisheye view들에 재투영한다. 각 후보가 여러 view에서 얼마나 일관되게 지지되는지를 feature 일치도와 어안 투영 품질로 계산해 query의 거리/range와 temporal update 신뢰도를 안정화한다.

## 제안 모듈

| 모듈 | 목적 | 본 연구의 차이 |
| --- | --- | --- |
| Object-Center Fisheye Ray Candidate Sampling | object query의 중심점 하나를 기준으로 reference camera ray를 만들고, 그 ray 위에 sparse range 후보를 생성한다. | dense overlap disparity나 box 전체 graph sampling이 아니라 object center query의 ray 방향 거리 후보만 sparse하게 검증한다. |
| Visible-View Ray Matching Support Scoring | 각 range 후보가 visible fisheye view들에서 얼마나 잘 지지되는지 계산한다. | 모든 view를 동일하게 평균내지 않고, query의 3D 위치를 가장 잘 지지하는 visible 관측을 더 강하게 반영한다. |
| Reliability-Guided Temporal Query Refinement | matching 기반 지지도를 이용해 매 시점 3D 위치/range update를 안정화한다. | 별도 tracking association이 아니라 temporal query의 3D 위치 추정 안정성에 집중한다. |

신뢰도 결합은 다음 형태를 기본으로 둔다.

```text
reliability(r_k) = matching_confidence(r_k) * projection_quality(r_k)
```

- `matching_confidence(r_k)`: visible view들에서 샘플링한 feature consistency.
- `projection_quality(r_k)`: image bound, fisheye FOV, projected footprint, image-center distance, truncation, invalid region 여부.
- 유효한 visible view가 두 개 미만이면 multi-view matching 대상에서 제외한다.

Reference view feature는 range 후보마다 바뀌지 않는다. Reference camera에서 query center pixel을 잡고 inverse projection해서 ray를 만들면, 그 ray 위의 모든 후보점은 reference camera에 다시 투영될 때 같은 pixel로 돌아온다.

```text
f_ref = reference camera의 query center feature
f_c(r_k) = range 후보 r_k를 camera c에 재투영한 위치의 feature
matching_confidence(r_k) = mean_c cosine(f_ref, f_c(r_k))
```

이후 variant에서는 visible view all-pair similarity도 비교한다.

```text
matching_confidence(r_k) = mean_{i,j} cosine(f_i(r_k), f_j(r_k))
```

Projection quality는 oracle 실험과 실제 inference에서 사용하는 box가 다르다.

```text
oracle 실험: GT box를 이용해 projected area / truncation / visibility 계산
실제 적용: predicted query box 또는 anchor box를 이용해 projection quality 계산
```

Temporal range 시작값은 object 상태에 따라 구분한다.

```text
기존 object query: 이전 frame의 range / temporal prior 주변에서 후보 생성
신규 object query: detector baseline range 주변에서 후보 생성
oracle 기초 검증: GT range 주변에서 후보 생성
```

## 현재 실험

오늘은 학습하지 않고 oracle 기반 기초 검증만 수행한다.

> GT object center 주변의 range 후보를 visible fisheye view들에 재투영했을 때, multi-view feature consistency와 projection quality를 곱한 reliability curve가 실제 range 근처에서 peak를 만드는가?

ORA3D와의 차이:

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
4. GT range 주변 후보를 만든다. 예: `r_gt +/- 2m`, `K=21`.
5. 각 후보 3D point를 네 개 fisheye camera에 모두 재투영한다.
6. image bound, fisheye FOV, projection-quality filter를 통과한 visible view만 남긴다.
7. valid view에서 feature를 sampling한다.
8. reference feature `f_ref`는 고정하고, 각 range 후보의 non-reference visible view feature `f_c(r_k)`와 cosine similarity를 계산한다.
9. projection quality와 matching confidence를 곱해 최종 reliability curve를 만든다.

```text
matching_confidence(r_k) = mean_c cosine(f_ref, f_c(r_k))
reliability(r_k) = matching_confidence(r_k) * projection_quality(r_k)
```

성공 기준은 `r_gt` 근처에서 reliability curve가 뚜렷한 local peak를 보이는 것이다.

결과가 좋으면 다음 순서로 확장한다.

```text
1. GT range 주변 oracle check
2. baseline predicted range 주변 check
3. temporal prior range 주변 check
4. reliability를 query update에 연결
5. auxiliary loss로 학습
```

Feature는 다음 우선순위로 비교한다.

```text
1. neck/FPN feature
2. object-region pooled feature
3. 2D aux feature
4. raw backbone feature
```

## 코드 배치

실제 모델 로직은 `projects/mmdet3d_plugin/` 안에 넣고, 그 모델을 실행하거나 중간 결과를 뽑는 바깥 실험 스크립트만 `tools/research/`에 모은다.

```text
projects/mmdet3d_plugin/models/projection.py        # fisheye projection / inverse ray / projection validity
projects/mmdet3d_plugin/models/ray_matching.py      # range 후보, visible-view matching, reliability 계산
projects/mmdet3d_plugin/models/blocks.py            # 기존 feature sampling/fusion에 reliability weight 연결
projects/mmdet3d_plugin/models/sparse4d_head.py     # query range/update에 reliability 반영
projects/mmdet3d_plugin/datasets/transforms/        # camera/GT/meta packing 보강
configs/                                            # ablation별 설정
tools/research/                                    # 위 모델을 호출하는 실험 실행 스크립트
```

따라서 앞으로 모델을 많이 수정해야 하는 것은 맞다. 다만 그 코드는 `tools/research/`에 넣지 않고, `projects/mmdet3d_plugin/`에 실제 모듈로 넣는다. `tools/research/`는 실행 스크립트가 흩어지지 않게 모으는 역할만 한다.

현재 예정 실행 스크립트:

```text
tools/research/ray_matching_oracle_check.py
```

결과 저장 위치:

```text
work_dirs/_ray_matching_oracle_check/
```

## 지금 볼 코드

우선 계속 봐야 할 곳은 `projects/mmdet3d_plugin/models/`가 맞다. 다만 config와 metadata packing은 같이 확인해야 한다.

| 우선순위 | 파일 | 볼 내용 |
| ---: | --- | --- |
| 1 | `projects/mmdet3d_plugin/models/projection.py` | KB fisheye projection, 앞으로 추가할 inverse ray / projection quality |
| 2 | `projects/mmdet3d_plugin/models/ray_matching.py` | 다음에 만들 ray 후보, support scoring, reliability 계산 모듈 |
| 3 | `projects/mmdet3d_plugin/models/blocks.py` | feature sampling/fusion에 reliability weight를 연결할 지점 |
| 4 | `projects/mmdet3d_plugin/models/sparse4d_head.py` | temporal query update에 reliability를 반영할 지점 |
| 5 | `projects/mmdet3d_plugin/datasets/transforms/pack_sparse4d.py` | camera matrix, fisheye params, GT/meta가 모델로 들어오는 흐름 |
| 6 | `configs/` | 실험별 ablation 스위치와 dataset/model 설정 |

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
| 1 | ORA3D | dense overlap disparity/depth supervision과 본 연구의 object-query sparse ray matching 차이를 대비한다. |
| 2 | Sparse4D v3 | object query, temporal instance bank, query update 흐름의 직접적인 기반이다. |
| 3 | MIC-BEV | 여러 camera/view 관측을 동일하게 보지 않고 중요도를 다르게 두는 논리를 방어한다. |
| 4 | FishBEV / Fisheye3DOD / DaF-BEVSeg | fisheye projection, view 품질, 주변부 왜곡 문제를 설명하는 배경으로 사용한다. |
| 5 | UA-Track | uncertainty-aware temporal 3D 추정과 대비한다. |
| 6 | Graph-DETR3D | 이번 1차 방법의 핵심은 아니며, 여러 node 확장 ablation 후보로만 남긴다. |
