# SphereStream Sparse4D 연구 작업공간

이 repository는 HorizonRobotics Sparse4D를 기반으로 KB20/SphereStream 계열 4-fisheye temporal 3D detection 실험을 진행하기 위한 private fork다.

Upstream 참고:

- Sparse4D 원본: https://github.com/HorizonRobotics/Sparse4D
- Sparse4D v3 paper: https://arxiv.org/abs/2311.11722

## 최종 연구 정의

> 4-fisheye temporal 3D detection에서 Sparse4D의 object query / sparse keypoint가 각 fisheye view에서 얼마나 안정적으로 관측되는지 계산하고, 그 fisheye geometry confidence를 multi-view attention weight에 반영하는 연구.

핵심 문제의식은 fisheye query가 단순히 "3D query가 이미지 어디에 투영되는가"만 보는 것이 아니라, "visible overlap view들 중 어떤 관측이 이 query를 가장 안정적으로 지지하는가"를 함께 판단해야 한다는 점이다.

현재까지의 GT 기준 range 후보 검증에서는 feature matching score가 정답 range를 충분히 안정적으로 고르지 못했다. 따라서 "sparse ray matching으로 depth/range를 직접 찾거나 보정한다"는 강한 주장은 현재 메인 claim으로 두지 않는다. 다만 이 결론은 전체 집계 기준이며, 근거리/큰 객체/visible view가 많은 경우에는 성립할 가능성이 남아 있다. Range 후보 실험은 조건별 보조 분석으로 유지하고, 주된 방향은 fisheye geometry 기반 visible view/keypoint confidence weighting이다.

단순히 `query feature -> range auxiliary loss` 하나를 붙이는 것은 novelty가 약하다. 본 연구의 핵심은 range head 자체가 아니라, 4-fisheye overlap에서 어떤 view/keypoint를 더 믿을지, fisheye ray geometry를 어떻게 attention에 반영할지를 설계하는 것이다. Range auxiliary는 `viewconf` config에 보조 loss로 구현했지만, 단독 contribution이 아니라 geometry-aware view confidence를 range-aware하게 학습시키는 보조 신호로 둔다.

우리가 쓰는 신호는 dense depth map이 아니다. `MultiScaleDepthMapGenerator`처럼 LiDAR surface point를 이미지 전체에 투영해 픽셀별 depth map을 만드는 방식과 구분한다. 본 연구에서의 핵심 신호는 object query / sparse keypoint가 각 fisheye view에서 얼마나 유효하고 안정적인 관측을 갖는지 평가하는 view support signal이다.

```text
dense pixel depth map target: no  # 이미지 전체 픽셀 깊이 지도 아님
2D bbox center target: no         # 2D 박스 중심 아님
direct depth/range selection claim: no
range aux only as novelty: no
fisheye geometry confidence attention: yes
learnable geometry bias gate: yes
query/keypoint visible-view support: yes
projection-quality reliability signal: yes
object-level ray-range auxiliary: yes  # 보조 loss, 단독 novelty 아님
temporal query update weighting: yes
```

현재 방향은 object 전체를 dense하게 matching하거나 graph node를 촘촘히 뿌리는 방식이 아니다. Sparse4D가 이미 사용하는 sparse box keypoint를 4개 fisheye view에 투영하고, 각 view/keypoint가 image bound, fisheye FOV, 중심부 거리, ray incidence angle, projected footprint, truncation, feature support 측면에서 얼마나 믿을 만한지 계산한다. 이 support를 attention logit에 bias로 넣어 feature aggregation과 temporal query update를 안정화한다.

## 제안 모듈

| 모듈 | 목적 | 본 연구의 차이 |
| --- | --- | --- |
| Object-Centric Local Tangent Patch Sampling | Sparse4D keypoint가 각 fisheye view에 투영된 뒤, 그 한 점만 feature로 찍지 않고 keypoint 주변의 작은 ray-local patch에서 여러 feature를 모아 하나의 observation으로 만든다. | 전체 이미지를 undistort하거나 spherical feature map으로 바꾸지 않고, sparse query가 실제로 참조하는 위치 주변만 국소적으로 보정한다. |
| Fisheye Geometry Relation Encoding | 각 query/keypoint와 camera 사이의 image bound, center distance, ray incidence angle, fisheye FOV 유효성을 계산한다. | 일반 image coordinate만 쓰지 않고 fisheye ray geometry를 명시한다. |
| Geometry-biased View/Keypoint Attention | Sparse4D attention logit에 geometry confidence bias를 더한다. | feature만으로 view weight를 학습하는 기존 Sparse4D에 fisheye 관측 품질 prior를 넣는다. |
| Learnable Geometry Bias Gate | geometry bias의 세기를 layer별 학습 파라미터로 둔다. | 기하 prior를 강제로 곱하지 않고, 학습이 필요한 만큼만 사용하게 한다. |
| Learned Geometry View Confidence | image/FOV/center/angle relation을 MLP로 attention bias에 추가한다. | 기하적으로 가능한 view 중 실제로 얼마나 믿을지는 학습으로 조절한다. |
| Object-level Ray-Range Auxiliary | matched query가 visible overlap view에서 GT object center의 camera range를 예측하게 한다. | dense pixel depth map 대신 object/query 단위 보조 신호만 사용한다. |
| Reliability-Guided Temporal Query Update | support가 높은 view feature를 더 강하게 쓰고, support가 낮은 query는 temporal prior를 더 보존한다. | depth를 직접 찾는 것이 아니라 temporal query update의 신뢰도를 조절한다. |

## 핵심 contribution

현재 구현된 contribution과 다음 단계 contribution을 분리한다.

현재 구현되어 결과가 나온 축은 다음이다.

```text
1. fisheye geometry relation encoding
2. query/keypoint-level visible view confidence
3. learnable geometry bias gate for Sparse4D attention
4. learned geometry-relation view confidence
```

다음 단계에서 논문 main claim으로 올릴 후보는 다음이다.

```text
5. yaw-specific fisheye observation attention
6. object-level ray-range auxiliary as weak supervision  # 보조/ablation
7. local tangent patch sampling                         # 보조/ablation
```

중요한 표현상 주의점:

```text
나쁜 표현:
  We add another attention module.

좋은 표현:
  We decouple orientation evidence aggregation from generic instance feature
  aggregation by learning fisheye-geometry-conditioned yaw reliability for
  each camera-keypoint observation.
```

즉 novelty는 "attention을 하나 더 붙였다"가 아니라, **어안에서 yaw를 설명하는
관측과 class/center를 설명하는 관측이 다르므로, yaw feature aggregation을
공통 instance aggregation에서 분리한다**는 점에 둔다.

현재 attention bias는 다음 형태를 기본으로 둔다. 어안 주변부에서는 객체가
단순히 작아지는 것이 아니라 한 축은 늘어나고 다른 축은 눌리는 비등방성
왜곡이 생기므로, 화면상 면적만으로 view 품질을 판단하지 않는다.
현재 구현은 projected footprint 기반 proxy뿐 아니라 KB 모델의 radial /
tangential local scale ratio를 직접 계산한 `q_kb_anisotropy`도 사용한다.

Local tangent patch는 image crop이 아니다. 기존 Sparse4D는 여러 box keypoint를
여러 camera/scale에 투영하지만, 각 projected keypoint observation은 여전히
distorted fisheye feature map 위의 한 sampling point다. 본 연구에서는 각
keypoint를 중심으로 camera ray 기준 작은 국소 관측 영역을 만들고, 그 영역의
점들을 다시 KB/OCam fisheye projection으로 이미지에 찍어 여러 feature를
sample한 뒤 하나의 keypoint feature로 합친다.

```text
Sparse4D:
3D keypoint -> fisheye projection -> one sampled feature

Ours:
3D keypoint
    -> camera ray 중심 local tangent patch 생성
    -> patch points를 fisheye model로 재투영
    -> multiple sampled features
    -> weighted sum으로 keypoint observation 생성
```

이 모듈의 목적은 3D box 표현을 바꾸는 것이 아니다. 전체 fisheye image를
펴지 않고도, query가 실제로 보는 국소 feature 관측을 더 안정적으로 만드는
것이다. 특히 이미지 주변부, 카메라 경계부, 근거리 객체처럼 한 점 sampling이
불안정한 경우를 겨냥한다.

```text
geo_conf_{c,k}
    = q_valid
      * q_bound
      * q_center
      * q_angle
      * q_relative_area
      * q_short_axis
      * q_shape_aspect
      * q_kb_anisotropy
attention_logit_{c,l,k,g}
    = sparse4d_logit_{c,l,k,g}
      + s_l * log(clamp(geo_conf_{c,k}))
      + MLP(r_{c,k})_g
attention_weight = softmax(attention_logit)
query_feature = sum attention_weight * sampled_view_feature
```

- `q_bound`: projection point가 이미지 경계에서 충분히 떨어져 있는 정도.
- `q_center`: fisheye 중심부에 가까운 정도.
- `q_angle`: camera ray incidence angle이 너무 크지 않은 정도.
- `q_relative_area`: 같은 object의 visible view 중 상대 footprint 면적.
- `q_short_axis`: projected footprint의 짧은 축이 충분히 큰 정도.
- `q_shape_aspect`: projected footprint의 long/short 축 비율이 과도하게 크지 않은 정도. 현재 default weight는 0으로 두어 noisy pose/truncation cue가 초반을 흔들지 않게 한다.
- `q_kb_anisotropy`: KB projection에서 직접 계산한 radial/tangential local scale imbalance. `theta_d(theta)`의 radial scale `d theta_d / d theta`와 tangential scale `theta_d / sin(theta)`의 비율이 1에 가까울수록 좋은 view로 본다.
- `q_valid`: image bound / fisheye FOV를 통과하는지 여부.
- `s_l`: deformable aggregation layer별 learnable geometry bias scale. 현재 default는 1.0에서 시작하고 학습으로 세기를 조절한다.
- `r_{c,k}`: valid 여부, boundary confidence, center confidence, angle confidence, normalized image 좌표, relative area, short-axis confidence, shape-aspect confidence, KB anisotropy confidence를 묶은 relation feature.

KB anisotropy는 다음 직관으로 계산한다.

```text
theta_d(theta)
    = theta * (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)

radial_scale
    = d theta_d / d theta

tangential_scale
    = theta_d / sin(theta)

anisotropy
    = max(radial_scale, tangential_scale)
      / min(radial_scale, tangential_scale)

q_kb_anisotropy
    = clamp((max_ratio - anisotropy) / (max_ratio - 1), 0, 1)
```

즉 "사람이 이 구간에서 이렇게 휜다"를 semantic label로 직접 학습시키는 것이
아니라, KB calibration 수식에서 해당 ray 위치의 방향별 왜곡 불균형을 계산해
view support prior로 넣는다.

직접 보정 ablation에서는 CUDA DAF를 끄고 camera별 sampled feature를 노출시킨 뒤,
같은 geometry relation으로 lightweight adapter를 적용한다.

```text
view_feature'_{c,k}
    = Adapter(view_feature_{c,k}, r_{c,k})
fusion = Σ attention_weight_{c,k} * view_feature'_{c,k}
```

이는 "사람이 어느 위치에서 어떻게 휜다"를 semantic하게 직접 지도하는 것이
아니라, KB projection에서 계산되는 local fisheye distortion descriptor로
view feature를 보정하는 실험이다.

Range auxiliary는 다음 원칙으로 구현한다.

```text
dense pixel depth map target: no
GT center as model input: no
matched GT center as training target: yes
object/query-level ray-range target: yes
overlap object only: yes
```

구현상으로는 matched query feature와 predicted query-camera relation을 넣어 camera별 normalized log range를 예측한다. Loss는 GT 3D box center를 각 fisheye camera 좌표계로 변환한 range를 target으로 쓰며, projection quality가 낮거나 visible view가 2개 미만인 query는 제외한다.

```text
pred_range_{q,c} = RayRangeHead(query_feature_q + RelationEncoder(pred_relation_{q,c}))
target_range_{q,c} = || T_lidar_to_cam_c * GT_center_q ||
L_ray_range = q_projection * SmoothL1(norm_log(pred_range), norm_log(target_range))
```

Range 후보 실험은 보조 ablation으로만 유지한다.

```text
main: 현재 query 위치의 visible view/keypoint support 계산
main training: object-level overlap range auxiliary + projection-quality weighting
ablation: GT range 주변 sparse keypoint range search
hold: feature score로 direct depth/range 선택
```

## 현재 실험

2026-07-08 기준 최신 상태.

### 완료된 구현

```text
1. KB fisheye projection 기반 Sparse4D 3-class baseline
2. fisheye geometry confidence 계산
   - valid
   - boundary
   - relative_area
   - short_axis
   - KB anisotropy
3. geometry confidence를 Sparse4D aggregation attention logit에 주입
4. geometry bias scale / quality exponent / view confidence MLP 학습
5. late-epoch NaN 방지를 위한 stable schedule
   - bias_scale=0.5
   - max_learnable_quality_weight=0.5
   - learned_confidence ramp: start 1000 iter, 3000 iter
   - geometry 관련 lr_mult=0.25
6. local tangent patch sampling 구현
   - 현재 main claim이 아니라 ablation / 보조 모듈
7. view feature confidence / view relation attention scaffold 구현
   - `*_stable_viewfeat.py`
   - `*_stable_relattn.py`
8. hospital entrance sequence를 train split에 추가한 config 작성
   - val은 기존 KB val 유지
```

### 현재까지의 핵심 결과

기존 KB val 기준 비교:

```text
Sparse4D baseline best:
  epoch 10
  mAP 0.2921 / NDS 0.2477 / mAOE 0.9843
  Car 0.4465 / Ped 0.2873 / Cyclist 0.1425

cleanrel_stable best:
  epoch 15
  mAP 0.3055 / NDS 0.2810 / mAOE 0.8246
  Car 0.4597 / Ped 0.2961 / Cyclist 0.1607

delta:
  mAP  +0.0134
  NDS  +0.0333
  mAOE -0.1597 rad
  Cyclist AP +0.0182
```

해석:

```text
cleanrel_stable은 단일 best checkpoint 기준 baseline보다 낫다.
특히 mAOE 감소가 연구 방향과 잘 맞는다.
다만 seed/epoch variance가 있으므로 논문용으로는 같은 seed rerun이 필요하다.
```

### 현재 실행 중인 실험

병원 로비 입구 시퀀스를 train에 추가하고, val은 기존 KB val로 유지한 rerun.

```text
config:
  configs/sparse4d_kb20_car_ped_cyc_ablation_30e_cleanrel_stable_hospitaltrain.py

train:
  /home/airlab/user/kdh/new_dataset/kb20_plus_hospital/pkl/
  kb20_seq29_49_hospital_train_idvel.pkl

val:
  /home/airlab/user/kdh/new_dataset/kb20_plus_hospital/pkl/
  kb20_seq29_49_hospital_val_idvel.pkl

work_dir:
  work_dirs/sparse4d_kb20_car_ped_cyc_ablation_30e_cleanrel_stable_hospitaltrain
```

이 run의 목적:

```text
1. hospital sequence 추가가 Car/Ped 안정성에 도움 되는지 확인
2. box 수 증가가 cleanrel_stable의 overfit/variance를 줄이는지 확인
3. id 기반 velocity pkl + mAVE metric 정리 이후의 기준 성능 확보
```

주의:

```text
velocity/mAVE 관련 loader/metric 수정은 MFTracker 쪽 코드에 있으며,
현재 이 Sparse4D repository commit에는 포함되어 있지 않다.
```

### 다음에 할 일

1. `cleanrel_stable_hospitaltrain` 30epoch 결과 확인.
2. 같은 train/val split에서 vanilla Sparse4D baseline도 rerun.
3. 같은 seed 기준으로 baseline vs cleanrel_stable 재비교.
4. mAP/NDS만 보지 말고 다음 breakdown을 확인.

```text
mAOE by class
mAOE by distance
mAOE by radial image position
mAOE in camera overlap region
mAOE by visible camera count
Car / Pedestrian / Cyclist AP
```

5. 그 다음 main novelty를 `yaw-specific fisheye observation attention`으로 확장.

최소 구조:

```text
F_common = Σ softmax(logit_common + bias_common) * f_obs
F_yaw    = Σ softmax(logit_yaw    + bias_yaw)    * f_obs

F_common -> class / center / size / velocity
F_yaw    -> yaw sin/cos / yawness
```

안정화 구조:

```text
F_yaw_final = F_common + λ * (F_yaw - F_common)
```

`λ`는 0 근처에서 시작하는 learnable scalar/gate로 두어, yaw branch가 초반에
망해도 기존 cleanrel_stable feature에서 크게 벗어나지 않게 한다.

6. patch는 main이 아니라 다음 ablation으로만 사용.

```text
baseline
cleanrel_stable
cleanrel_stable + yaw-specific attention
cleanrel_stable + tangent patch
cleanrel_stable + yaw-specific attention + tangent patch
```

### GA-HF / DAPETR와의 구분

GA-HF와 DAPETR 모두 "fisheye geometry를 고려해야 한다"는 문제의식은 겹친다.
하지만 본 연구의 단위는 다르다.

```text
GA-HF:
  BEV-level
  LiDAR + fisheye fusion
  polar/cartesian BEV alignment
  warped camera BEV feature 보정

DAPETR:
  PETR-style projection-free dense image-token adaptation
  distortion-aware PE / FiLM co-modulation

Ours:
  Sparse4D query-level
  camera-keypoint observation aggregation
  fisheye geometry-conditioned view/keypoint reliability
  yaw-specific observation attention for mAOE
```

따라서 논문 표현은 넓은 `geometry-aware fisheye fusion`이 아니라 다음처럼 둔다.

```text
yaw-aware query-level observation aggregation
for sparse fisheye 3D detection
```

현재까지 수행한 기초 검증은 다음과 같다.

> GT object 주변 range 후보를 visible fisheye view들에 재투영했을 때, feature score가 실제 range 근처에서 충분히 강한 peak를 만드는가?

전체 집계 결과는 direct depth/range refinement claim을 메인으로 지지하기에는 부족하다.

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

따라서 `sparse keypoint`는 center 한 점보다 낫지만, 전체 평균 기준으로는 "정답 range를 직접 고른다"는 메인 주장으로 약하다. 이후 실험은 range search를 보조 ablation으로 두고, visible view support weighting을 주 방향으로 둔다. 단, range search는 완전히 폐기하지 않고 거리, projected size, visible view 수, feature level에 따른 breakdown으로 계속 확인한다.

조건별 재확인 결과, `level0 + z_center`에서는 visible view 수에 따라 차이가 컸다.

```text
level0 + z_center, 122 objects:
전체:         |offset|<=0.4m 35.2%, |offset|<=0.6m 40.2%
visible 2개:  |offset|<=0.4m 42.2%, |offset|<=0.6m 48.9%
visible 3개:  |offset|<=0.4m 54.8%, |offset|<=0.6m 61.3%
visible 4개:  |offset|<=0.4m 15.2%, |offset|<=0.6m 17.4%
```

이 결과는 direct range matching을 완전히 버리라는 뜻이 아니라, 좋은 visible-view selection 없이는 score가 오염된다는 뜻에 가깝다. 특히 4개 view를 모두 쓰는 경우가 오히려 나쁜 것은 bad view를 걸러내는 view confidence가 필요하다는 근거로 해석한다. 여기서 `visible 4개`는 실제 객체가 네 카메라에서 모두 잘 보인다는 뜻이 아니라, 느슨한 projection-valid 조건에서 네 view가 후보로 들어왔다는 뜻이다.

이후 진단 스크립트는 기본적으로 reference view와 projection quality가 가장 높은 support view 1개만 비교하도록 수정했다. 이 설정은 보통 두 어안 view가 겹쳐 같은 객체를 보는 상황에 더 맞다.

```text
level0 + z_center + top-1 support view, 223 objects:
전체:          |offset|<=0.4m 39.9%, |offset|<=0.6m 48.9%
0-5m:          |offset|<=0.4m 54.5%, |offset|<=0.6m 66.7%
visible 2개:   |offset|<=0.4m 46.4%, |offset|<=0.6m 57.1%
visible 3개:   |offset|<=0.4m 59.2%, |offset|<=0.6m 67.6%
Pedestrian:    |offset|<=0.4m 62.5%, |offset|<=0.6m 75.0%
```

따라서 현재 해석은 "range cue가 없다"가 아니라, "기하학적으로 가능한 모든 view를 평균하면 bad/occluded view가 score를 오염한다"이다. 기하학적으로 보이지 않는 view는 hard mask로 버리고, 기하학적으로는 보이지만 가림/측면/잘림/feature 약화가 있는 view는 learned view confidence로 낮추는 방향이 맞다.

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
4. Range search를 거리 bucket, projected size, visible view 수, feature level별로 다시 나눠 본다.
5. 조건별 breakdown에서도 충분히 좋아지지 않으면 direct depth refinement는 메인 claim에서 제외하고 view support weighting만 유지한다.
6. Baseline ablation을 `range aux only -> ray PE -> object-overlap mask -> projection-quality weighting` 순서로 구성한다.


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
ray_range_pred           = (B, N_obj, N_cam)        # 학습 auxiliary
ray_range_aux_weight     = (B, N_obj, N_cam)
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
projects/mmdet3d_plugin/models/blocks.py            # geometry-biased + learned view/keypoint attention
projects/mmdet3d_plugin/models/sparse4d_head.py     # object-level ray-range auxiliary / temporal query update 후보
projects/mmdet3d_plugin/datasets/transforms/        # camera/GT/meta packing 보강
configs/sparse4d_kb20_car_ped_viewconf.py           # fisheye geometry attention ablation
configs/                                            # 기타 ablation별 설정
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
| 3 | `projects/mmdet3d_plugin/models/blocks.py` | DeformableFeatureAggregation attention에 fisheye geometry bias와 learned view confidence MLP를 넣은 지점 |
| 4 | `projects/mmdet3d_plugin/models/sparse4d_head.py` | object-level ray-range auxiliary head를 구현한 지점 |
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
