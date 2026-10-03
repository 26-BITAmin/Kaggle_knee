# 디스커션 정리

대회 디스커션 148개를 전부 읽고 주제별로 정리한 것

2026-09-26 133개로 처음 정리하고 2026-10-03 새 토픽 15개와 기존 토픽의 새 댓글을 반영했다

원문은 `refs/discussions/` 에 수집돼 있다 (gitignore 대상)
`python scripts/collect_discussions.py` 로 재수집한다

읽으면서 확인된 사실은 `docs/competition.md` 와 `docs/baseline.md` 에도 반영돼 있다
이 문서는 **스레드 단위 출처**를 남기기 위한 것이다

---

## 1. 레이블 - 이 대회의 병목

### 58개 gold 레이블이 전부다

`train.csv` 4,407개 중 58개만 레이블이 있고, 그 58개는 12개 컬럼이 **전부** 채워져 있다
부분 레이블은 없다 (all-or-nothing)

나머지 4,349개는 `Report` 텍스트만 있다
`test.csv`에는 `Report`가 없다 - 호스트가 명시적으로 확인했다

> 733932, 738851, 734118, 733592, 733836, 734106, 734055

### 레이블 추출 방식별 성능 (gold 58건 대비 macro AUC)

| 방식 | AUC |
|---|---|
| 정규식 / 사전 추출 | 0.8136 |
| LLM 판독문 독해 | 0.8780 |
| + Synovitis 미결정 셀을 Effusion으로 보충 | 0.8873 |
| + 12개 전부에 학습 기반 보충 | 0.8805 (악화) |

보충은 근거 있는 한 컬럼만 하고 일반화하지 않는다

> 733932

### 공개 레이블 테이블 비교

누군가 공개 테이블들을 gold 58건에 대고 직접 측정했다

| 테이블 | macro AUC |
|---|---|
| stevenleehans v4 blend | 0.893 |
| stevenleehans v2 | 0.887 |
| pilkwang report_labels_v2 | 0.870 |

**독립적인 두 테이블을 평균하면 여러 레이블에서 각각보다 낫다**

> 737454

더 높은 수치를 보고한 사례도 있다
LLM 추출 레이블이 gold에서 0.9083, 단일 fold LB 0.935

> 743148, 738172

### "언급 없음"을 어떻게 볼 것인가

전체 레이블 셀의 25.4%가 미언급이고 소견별 편차가 극단적이다

미언급을 음성으로 뭉개면 안 된다
0.5(미결정)로 두고 손실에서 마스킹하거나 신뢰도를 샘플 가중치로 쓴다

> 733932, 737650

### Synovitis가 유독 안 되는 이유

모든 공개 노트북에서 Synovitis가 12개 중 최악이다 (0.62~0.72)
원인은 모델이 아니라 지도 신호다

- 9개 언어를 통틀어 **판독문의 약 16%만** synovitis 용어를 언급한다
  (`synovit/sinovit/υμεν/синови/sinovyal`)
  나머지 84%는 어떤 레이블 테이블을 쓰든 **추론된 값**이다
- 레이블 테이블 자체가 gold와 Synovitis에서 ~0.79 밖에 안 맞는다 (ACL은 ~0.99)
- **RadImageNet 사전학습 인코더가 0.78을 냈다** - ImageNet 계열은 0.62~0.72
  이미지 신호는 존재하는데 자연영상 피처가 못 본다는 뜻이다

실무 결론: Synovitis에 아키텍처 튜닝 시간을 쓰지 말 것
의료영상 사전학습 멤버를 앙상블에 하나 넣는 것이 낫다

> 737566

### 판독문과 gold 레이블이 안 맞는 사례

판독문 기반 리뷰와 제공된 레이블이 어긋나는 경우가 보고됐다
gold 자체의 오류 가능성도 제기됐다

Synovitis/Effusion 판정 기준에 대한 질문도 있었다
(비조영 검사에서 synovitis를 effusion 대리지표로 채점했는지)

> 733826, 737155, 742106, 733864

### 레이블을 더 다듬는 것은 효과가 갈린다

- **효과 없음 쪽** - LB 약 0.94 단일 모델 보유자 (Tucker Arrants)
  Qwen 단일 프롬프트 추출만으로 대부분 0.95 초반
  추출 실패 수천 건을 고쳐도 LB 변화 없음
  "판독문 -> 레이블과 이미지 -> 레이블의 간극은 추출을 다듬어서 메울 수 없다"
- **효과 있음 쪽** - DINOv2 가 레이블만 바꿔서 LB 0.931 -> 0.942 (SpeedSci)
  레이블 버전에 따라 DINOv2 와 ConvNeXt 의 우열이 뒤집혔다는 보고도 있다
- gold 58건으로는 레이블 간 차이를 가를 수 없다는 데는 모두 동의한다
  LB 가 gold 와 같은 판독의가 만든 더 큰 표본이라 비교 기준으로는 LB 가 낫다

> 735304, 745214

### 레이블 다루는 법 - 0.94 이상 팀들의 공통점

- **soft label 을 유지한다** - 0/1 로 반올림하면 나빠진다
- **0.5 근처로 몰린 타깃은 BCE 로 거의 학습되지 않는다**
  타깃을 날카롭게 만들자 gold 0.653 -> 0.725 (0.74 OOF 수준 팀의 측정)
- **약한 소견은 낮게 매긴다** - "경미한 삼출"이 gold 에서 음성인 경우가 있다
  모든 언급을 0.85 로 두던 것을 고치자 Effusion 0.628 -> 0.726

> 738339, 735304

### OOF 예측을 레이블에 섞기 - 가장 많이 반복된 비법

| 보고자 | 방법 | LB |
|---|---|---|
| Archit Konde | 2.5D CoAtNet 224, 추출 레이블에 OOF 예측을 섞음 | **0.950** 단일 fold |
| Raymond Yuen | 여러 약한 레이블로 teacher 학습 -> pseudo label 로 최종 학습 | 0.938 -> **0.949** |
| tennogh | OOF pseudo label, 288px | 0.943 |

핵심 조건

- **반드시 out-of-fold 예측이어야 한다** - fold 안 예측은 레이블을 따라 말할 뿐이다
- 레이블을 예측으로 **대체하면 나빠지고 섞어야 한다**
  50/50 에서 이미 올랐고 모델 쪽으로 더 치우치자 조금 더 올랐다
- 효과는 판독문이 침묵하는 셀(특히 Synovitis)에서 나온다
  이미지 모델이 이미 레이블보다 gold 에 가까운 곳이다
- 노이즈 셀을 마스킹하는 것은 효과가 없었다 (여러 명)
- 섞은 레이블로 검증하면 CV 가 당연히 오르므로 검증은 원래 레이블로 한다
- **반례도 있다** - 50/50 블렌드가 로컬 CV 와 gold 는 올렸지만 LB 는 그대로였다 (Nicolai Karcher)

Chris Deotte 의 지적 - pseudo label 을 쓴 "단일 모델"은 앙상블 지식을 증류한 것이라
순수 단일 모델과 같은 선상에서 비교하면 안 된다

> 735304

---

## 2. 모델

처음 정리할 때는 "여기는 병목이 아니다"로 봤다
그 근거(아래 ablation)는 OOF 0.78 수준 팀의 측정이다
0.93 이상 팀들의 보고를 보면 입력 기하와 미세조정 범위는 차이를 만든다

### 상위 단일 모델 구성

| 보고자 | 구성 | LB |
|---|---|---|
| CoolinLai | ResNet50 224px 5-fold | **0.954** |
| NguyenThanhNhan | Qwen 3.5 2B VLM, 384px, 비전 인코더 + LoRA, 단일 fold 3.5시간 | 0.950 |
| Scott Willis | 작은 ResNet 단일 fold, 자체 Gemma 4 레이블(gold 약 0.89), 채점 5분 | 0.949 |
| SpeedSci | DINOv2 / EffNet-B4 / ResNet50 각각 5-fold | 0.942 / 0.940 / 0.940 |
| (팀원 모집 글) | CoAtNet 384 / ConvNeXt-T / RadImageNet ResNet50 | 0.935 / 0.930 / 0.924 |
| Cody_Null | 단일 모델, val 0.90, gold 0.92~0.93 | 0.931 |
| kmn | CoAtNet 단일 모델 | 0.930 |
| Tom Aindow | DINOv2, 392px, 150mm, 스터디당 32장 무작위 | 0.915 |

- 상위권은 **CNN(ResNet, EfficientNet, CoAtNet)이 다수**다
- "224 / 288 이 최적점, 해상도는 큰 동인이 아니다" (Tucker Arrants, ResNet/EffNet 기준)

> 735304, 745091, 745214

### 입력 기하 - 시야와 슬라이스

- **시야(크롭)** - 100mm 는 구조를 잘라먹는다
  100 -> 140mm 로 "크게 좋아졌고" 160mm 는 140mm 와 비슷 (Cody_Null)
  모델마다 최적 시야가 다르다는 정정도 있다 (Parag)
- **슬라이스 범위** - 0.92 이상 DINOv2 들은 넓게 본다
  스터디 전체에서 32장 무작위 (Tom Aindow, 0.915)
  140mm 범위에서 시리즈당 16장, 336px, 백본 전체 미세조정 (744230 작성자)
- 아래 "중앙 9장 인접이 등간격보다 낫다"는 측정과 방향이 다르다
  그쪽은 등간격 9장을 24장 범위에 뿌린 것과의 비교다

> 735304, 744230

### 백본 미세조정 범위

"푸는 층 수와 점수 사이에 강한 상관이 있다" (Tim Krige, 작은 DINOv2 기준)
대회에 따라 head 만, 마지막 몇 층, 전체 해제가 각각 최적이라고 한다

> 735304

### 앙상블은 서로 다른 모델끼리

- "서로 다르게 틀리는 모델들의 앙상블이 서로 동의하는 모델들보다 낫다" (Dread Development, Chris Deotte 동의)
- 공개 앙상블에 상관 0.83 짜리 외부 모델 하나를 섞어 public +0.004 (charlessavas 팀)
- 팀 병합 제안들도 "공개 노트북과 다르게 만든 모델"을 찾는다
  공개 Raptor 와 gold 순위상관 0.925 인 ConvNeXt-T 를 강점으로 내세운 사례가 있다

> 742926, 735304, 745091

### 인코더 크기는 무의미하다

DINOv2-Small → Base (22M → 87M, step당 연산 3.9배)

**CV +0.0011, 측정된 노이즈 바닥 0.0020** → 통계적으로 null
12개 중 5개만 Base 쪽으로 움직였고 macro 이득 전부가 가장 약한 레이블 하나에서 나왔다

별도로 224 해상도에서 Base vs Small을 비교한 사례도 같은 결론이다

> 735154, 738096

교훈 하나 더: **체크포인트는 자기 전처리 계약을 갖고 다닌다**
인코더 가중치만 바꾸고 정규화 상수를 안 바꾸면 조용한 핸디캡이 된다

### 무엇이 실제로 움직였나 (통제된 ablation)

EfficientNet-B0, 4슬롯, scanner-aware 5-fold 기준

| 설정 | OOF |
|---|---|
| 192px, 크롭 없음, 24장 등간격 | 0.7695 |
| 288px, 140mm 크롭, 12장 인접 | 0.7777 |
| 288px, 140mm 크롭, 24장 중 중앙 9장 | 0.7815 |

**슬라이스 기하가 해상도보다 중요했다** - 중앙 9장 인접이 등간격 9장보다 +0.018

움직이지 않은 것 (fold 0, 기준 0.7914)

| | OOF | 차이 |
|---|---|---|
| EMA decay 0.999 | 0.7940 | +0.003 |
| Mixup alpha 0.2 | 0.7937 | +0.002 |
| 60 epoch cosine | 0.7922 | +0.001 |
| Asymmetric loss | 0.6526 | **-0.139** |

세 가지 정규화가 전부 노이즈 안에서 논다 → 학습 레짐은 병목이 아니다

> 737597

### 좌우 정규화는 필요 없을 수 있다

위 스레드에서 laterality를 통제 실험으로 기각했다

| | 평균 절대 AUC 격차 |
|---|---|
| 좌우 의존 타깃 (medial/lateral) | 0.014 |
| 나머지 타깃 | 0.031 |

오히려 좌우 의존 타깃이 **덜** 영향받았다
`Laterality` 태그는 4,407개 중 2,185개에만 있다

> 루트 노트북(`pilkwang`)은 좌우 정규화를 하고 있어서 서로 어긋난다
> 우리가 직접 재보고 판단할 항목이다

> 737597

### 슬라이스 집계 구조

역대 RSNA 우승 솔루션이 pooling 대신 recurrent head를 쓴다는 지적
(2019 출혈 1위 "Sequential model wins", 2022 경추)

> 738495

---

## 3. 데이터의 함정

### 파일명 정렬은 틀린다

파일명은 SOP Instance UID라 고유할 뿐 순서가 아니다
물리 위치와의 순위상관이 ρ ≈ 0.01

`ImageOrientationPatient` × `ImagePositionPatient` 로 복원해야 한다

> 741310, 루트 노트북 §3b

### 한 스터디에 무릎이 둘인 경우

약 7개 학습 스터디에서 양쪽 무릎이 한 DICOM에 들어 있다
순진한 크롭이 무릎을 아예 못 잡거나 뼈가 없는 영역을 골절로 판정한 사례가 보고됐다

테스트셋에도 있다면 어느 쪽 무릎이 레이블 대상인지 알 방법이 없다는 지적이 있다

> 735639

### 손상된 DICOM 2건

픽셀 버퍼 크기가 헤더 선언의 **정확히 절반**이라 `dcm.pixel_array`가 터진다

```
1.2.826.0.1.3680043.8.498.34685905030370793639196564723935583035
  ValueError: 409600 vs 819200 bytes
1.2.826.0.1.3680043.8.498.37833587429731221455928642963031995680
  ValueError: 173056 vs 346112 bytes
```

호스트가 확인했다 캐시 빌드 시 try/except 필수

> 737163, 738708

### `Fluid_Sensitive` 와 `Fat_Suppression` 이 동일하다

24,371개 전부 같다 (0/0 10,361건, 1/1 14,010건, 불일치 0건)
이름은 물리적으로 독립인 두 속성인데 주어진 컬럼은 한 축만 담는다

DICOM 헤더(TR/TE, `ScanningSequence`, `SeriesDescription`)에서 복원해야 한다

> 737312, 우리 EDA(`docs/eda.md` 5절)에서 직접 검증함

### `PatientSex` 가 문서에는 있는데 파일에는 없다

Data 페이지는 `train.csv`에 `StudyInstanceUID, PatientSex, Report` + 12 레이블이 있다고 적었지만
실제 파일은 14컬럼이고 `PatientSex`가 없다

다만 **DICOM 헤더(0010,0040)에는 있고 테스트셋에서도 읽힌다**

학습셋 분포 M=2,076 / F=1,894

| 타깃 | M 유병률 | F 유병률 |
|---|---|---|
| ACL | ~54% | ~32% |
| Medial OA | ~12% | ~45% |

남성은 외상성(ACL/MCL/Fracture), 여성은 퇴행성(OA)이 높다

> 733423, 734004

### 스캐너 그룹이 CV를 오염시킨다

DICOM에 `Manufacturer`(Siemens/Philips/GE)와 `MagneticFieldStrength`(1.5T/3T)가 있고
4,407개에 약 13개 스캐너 그룹이 있다

| 그룹 | 스터디 |
|---|---|
| Siemens_1.5T | 1,148 |
| Siemens_3T | 781 |
| GE_1.5T | 698 |
| Philips_3T | 663 |
| Philips_1.5T | 619 |

**픽셀 없이 메타데이터만으로 분류기를 학습**해서 분할 방식을 비교했다

| 분할 | macro AUC |
|---|---|
| random 5-fold | 0.652 |
| scanner-grouped | 0.598 |

메타데이터만으로 0.05 차이가 난다
OA 타깃에서 가장 크게 떨어졌다 (0.07~0.09) - 자장 세기에 따른 연골 대조 차이와 일치한다

> 734004

### 기타

- `FrameOfReferenceUID`가 같은 스터디 안에서도 시리즈마다 다르다
- 전체 데이터셋 크기가 569.76 GB로 표시되는데 실제 압축 해제는 약 530 GB (819,640 파일)

> 736457, 739693

---

## 4. 제출 환경에서 조용히 깨지는 것들

### 기본 GPU(P100)가 사전설치 PyTorch와 호환되지 않는다

PyTorch 2.10.0+cu128은 sm_70 이상이 필요한데 P100은 sm_60이다
**`cuda.is_available()`는 여전히 True를 반환해서 조용히 실패한다**

`kernel-metadata.json`에 명시할 것

```json
"machine_shape": "NvidiaTeslaT4"
```

### DICOM 코덱 플러그인이 없고 인터넷도 없다

이 데이터셋에 JPEG Lossless / JPEG2000 시리즈가 있는데
`pylibjpeg` / `gdcm`이 커널에 설치돼 있지 않다
채점 실행은 인터넷이 꺼져 있어 그때 설치할 수 없다

로컬에서 휠을 받아 Dataset으로 올리고 오프라인 설치해야 한다

```
pip download --only-binary=:all: --platform manylinux2014_x86_64 \
  --python-version 312 --no-deps -d ./wheels \
  pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg python-gdcm
```

### `/kaggle/input/` 경로가 예상과 다르다

대회 데이터가 `/kaggle/input/competitions/<slug>/` 아래에 있고
붙인 Dataset도 `/kaggle/input/datasets/<owner>/<dataset>/` 로 갈 수 있다

**하드코딩하지 말고 런타임에 탐색해야 한다**
단 깊이 제한 없는 `os.walk`는 `train_series/` 전체를 훑으므로 깊이를 제한할 것

> 735854

### 앙상블 소스 누락

커널의 `model_sources` / `kernel_sources`가 빠지면
앙상블 멤버가 **조용히 드롭된다**
실제로 20개 DINOv2 멤버가 전부 빠져 0.940 → 0.937이 된 사례가 있다

> 742050

### T4/P100 에서 bf16 이 50배 느려진다

T4 와 P100 에는 bf16 연산기가 없다
`torch.autocast(dtype=torch.bfloat16)` 는 에러 없이 에뮬레이션으로 돌아 forward 가 약 50배 느려진다

실제 사례

- 체크포인트 10개 x TTA 5뷰 x 숨은 테스트 1,322건 = 약 17시간
- 8시간 시간 가드가 걸려 절반 넘는 스터디가 0.5 로 채워졌다
- 0.86 짜리 모델이 public 0.710 을 받았고 fp16 으로 바꾸자 0.851
- 공개 테스트 3건에서는 아무 이상이 없었다

점검 목록

- `torch.cuda.is_bf16_supported(including_emulation=False)` 로 확인한다
  인자 없이 부르면 버전에 따라 True 가 나온다
- 스터디당 초를 로그로 남기고 약 1,300건으로 환산해 본다
- 시간 가드가 있으면 대체값으로 채운 스터디 수를 로그로 남긴다
  부분 폴백은 나쁜 모델과 구분되지 않는다
- 학습 데이터 일부에 똑같은 추론 경로를 돌려 gold 와 대조하면 파이프라인 문제인지 가를 수 있다

> 744230

---

## 5. 대회 규칙

### 외부 데이터

호스트 판정

- **KneeCoT: 불허** - 기관과의 공식 데이터 사용 동의가 필요해서
  동의를 받을 수 있는 참가자와 없는 참가자 사이에 불공정이 생긴다는 해석
- 등록이 필요한 데이터셋(Stanford MRNet, OAI 등)도 같은 논리로 질의가 이어졌다
- 재배포 제한이 있는 가중치와 우승자 공개 의무(2.8.a)의 충돌도 논의 중이다

> 734109, 737950, 743416, 735497, 743420, 733652

2026-10-03 기준 추가 질의도 전부 **호스트 미답변**이다

- MRNet 약관은 파생물 생성을 금지한다는 지적이 나왔다
- "상위 10팀만 정밀 감사되므로 그 아래는 외부 데이터를 써도 걸리지 않는다"는 우려가 있다
  외부 데이터를 쓰는 팀과 안 쓰는 팀 사이 불공정 문제로 번지는 중이다

> 743416, 743420, 733652

### 다른 참가자가 공개한 가중치

대회 데이터로 학습된 남의 공개 가중치를 앙상블하거나 증류한 제출이 상금 자격이 있는지 질의 중이다 (**미답변**)

- 2.6.b 는 외부 모델을 허용하고 3.6.b 는 공개 코드를 오픈소스 라이선스로 간주한다
- 그러나 2.8 은 우승자에게 학습 코드를 요구하는데 남의 가중치는 학습 코드가 없다
- 3.14.a 는 "자신의 독창적인 작업"을 요구한다

공개 앙상블(d4-lite 등)을 그대로 쓰는 우리 제출도 여기에 해당한다

> 744056

### RadImageNet 라이선스 충돌

- Kaggle 데이터셋(`marwanmath/resnet-50-radimagenet-marwan`)은 CC-BY-NC-SA-4.0
- Hugging Face(`Lab-Rasool/RadImageNet`)의 같은 파일은 MIT
- 규칙 6.c 는 상업적 제한 없는 OSI 라이선스를 요구한다

> 745283, 735121

### 일정

- 팀 병합 마감 **10월 15일**
- 최종 제출 선택 **10월 22일** (2개 선택)

> 744230, 745091, 743356

### LLM API 사용

판독문을 상용 호스팅 LLM에 보내는 것은 **허용**
단 도구가 모든 참가자에게 합리적으로 접근 가능하고 비용이 최소여야 한다는 단서가 붙었다

> 733965, 733873

### 런타임

- 제출 런타임 제한 9시간
- Efficiency Prize는 `RuntimeSeconds` 기준
  공개 LB 상위 50 중 Efficiency LB 상위 50에 같이 든 팀이 약 32%뿐이다
- GPU 노트북이 Efficiency Prize 대상인지에 대한 질의가 있었다

> 736678, 740474, 738129, 733753, 733475

---

## 6. 리더보드 지형

- 공개 포크 약 200팀이 **정확히 0.941**에 몰려 있다
  전부 `mattiaangeli/bend-the-knee-to-the-dinosaurs` 계열이고
  그 뿌리는 `pilkwang/rsna-knee-baseline-v1` (0.891) 하나다
- 단일 모델 공개 언급은 0.887~0.903 (DINOv2)
- 0.92~0.94는 대부분 20개 이상 체크포인트 앙상블
- **OOF가 LB를 과소평가한다는 보고가 여럿 있다**
  0.7815 OOF → 0.870 public, 0.9083 gold → 0.935 LB
  레이블이 판독문 유래라 OOF는 "판독문을 맞추는 능력"을 재기 때문으로 보인다
- 5-fold 앙상블(0.938)과 단일 fold 마지막 epoch(0.939)이 같더라는 보고도 있다

> 735304, 736635, 742050, 740610, 740375

2026-10-03 갱신

- 1위 0.961, 참가 4,640팀
- d4-lite 계열 0.943 에 622팀이 동점이다 (우리 포함)
- Chris Deotte 예측 - 금메달 경계 약 0.96, 1위 약 0.975
- **셰이크업**
  - "공개 노트북 블렌드(0.945)는 과적합됐고, 공개 노트북 없이 0.935 이상이면 은메달권으로 오를 수 있다"는 의견 (근거 없는 주장)
  - 3,000~6,750팀 규모 대회 11개 분석 - 동메달 경계 바로 안쪽과 바로 바깥쪽 팀이 최종 메달을 받은 비율이 둘 다 24%
- 판독문 유래 레이블 val 에서 LB 로 +0.02~0.03 오르는 현상은 여러 팀이 똑같이 겪는다
  LB 정답은 노이즈가 없기 때문으로 해석된다

> 742327, 735767, 735304

---

## 7. 우리가 확인해야 할 미해결 항목

- [ ] 좌우 정규화가 실제로 필요한가 (737597은 불필요, 루트 노트북은 수행)
- [ ] scanner-grouped CV를 쓸 것인가 (메타데이터만으로 0.05 격차)
- [ ] `PatientSex` 를 피처로 쓸 것인가 (테스트 헤더에 존재, 유병률 차이 큼)
- [ ] 양쪽 무릎 스터디 7건을 어떻게 처리할 것인가
- [ ] RadImageNet 계열을 Synovitis 전용 멤버로 넣을 것인가
- [ ] public/private 분할이 site 층화인지 site 홀드아웃인지 (호스트 미답변)
- [ ] 숨은 테스트 스터디에 세 평면이 항상 다 있는지 (744519, 호스트 미답변)
- [ ] 시리즈 중앙 연속 9장만 보는 우리 슬라이스 선택이 내외측 구획을 놓치는지
- [ ] OOF pseudo label 을 섞을 것인가 (5-fold OOF 가 먼저 필요)
- [ ] 공개 가중치를 쓴 제출의 상금 자격 (744056, 호스트 미답변)

<!-- INDEX -->

## 전체 목록 (148개)

득표순. `python scripts/make_discussion_index.py` 로 갱신한다.

| 표 | 댓글 | 제목 |
|---|---|---|
| 83 | 12 | [Knee Abnormality Detection AI Challenge Overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343) |
| 61 | 98 | [Best single-model score](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304) |
| 54 | 9 | ["Not addressed" is a label too — what we learned reading 4,407 knee reports with an LLM](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932) |
| 40 | 20 | [I got a perfect usecase for this competition.](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733497) |
| 27 | 2 | [Possible inconsistencies between MRI reports and provided labels](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826) |
| 21 | 7 | [Rules clarification: external knee-MRI datasets, and using an LLM API to derive labels from the reports](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733652) |
| 20 | 13 | [Use of Commercially Hosted LLMs](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965) |
| 20 | 6 | [Sharing my knee MRI](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/736170) |
| 19 | 20 | [RSNA Raptor Weights](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737696) |
| 18 | 2 | [0.932 LB within one day. Tested for DICOM metadata shortcut](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733517) |
| 17 | 8 | [What Could the Final Ceiling Be for RSNA 2026?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735767) |
| 17 | 1 | [Welcome to Knee Abnormality Detection Challenge!](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733375) |
| 15 | 14 | [Does Kaggle Do Anything About Byte-for-Byte Public Notebook Theft?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735348) |
| 15 | 14 | [Plateau at 0.903: sharing my pipeline and ablations, and looking for advice](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740610) |
| 14 | 3 | [Scaling the encoder bought us nothing (+0.0011)](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735154) |
| 13 | 13 | [Too many noise public notebookS.](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/736268) |
| 13 | 11 | [What is the real bottleneck now: labels, image pipeline, or model capacity?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735826) |
| 13 | 4 | [Maybe worth checking the Efficiency LB?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/736678) |
| 12 | 2 | [Please clarify which external knee MRI datasets are allowed](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416) |
| 11 | 2 | [Are you actually looking at the data?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/736301) |
| 11 | 0 | [Annotation criteria question: how were Synovitis and Effusion graded?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737155) |
| 10 | 5 | [My account was banned due to a notebook. Just reinstated](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737820) |
| 10 | 4 | [corrupt files](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737163) |
| 10 | 2 | [Is the gated KneeCoT dataset permitted as external data?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734109) |
| 9 | 16 | [Using Dino3](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733313) |
| 8 | 5 | [Two knees are better than one?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735639) |
| 8 | 3 | [Data/Reporting Inconsistencies](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733491) |
| 7 | 3 | [58 labelled studies out of 4,407 — the rest of the supervision is in the reports](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734106) |
| 7 | 3 | [Public/private test split — stratified by site, or entire sites held out?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734681) |
| 7 | 3 | [LB OOF relationship](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/736635) |
| 7 | 0 | [Healthcare professionals: what knee MRI resources do you use in your country or language?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/736720) |
| 6 | 13 | [Autonomous Researchers on the Kaggle Leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743916) |
| 6 | 1 | [How to get started + Competition's Official Discord](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/730709) |
| 6 | 0 | [DICOM metadata findings: scanner-grouped CV and PatientSex priors](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734004) |
| 6 | 0 | [Are CC-BY-NC pretrained weights compatible with the winners open-licence obligation?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735121) |
| 6 | 0 | [Why everyone's Synovitis AUC is stuck around 0.6-0.7 (it's not your model)](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737566) |
| 5 | 2 | [Dinov2 Base vs Small at 224 resolution](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/738096) |
| 5 | 0 | [Request for official clarification: Rule 2.6(a) and registration-gated public datasets (MRNet, OAI, etc.)](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735497) |
| 4 | 23 | [Does anyone try to generate report text from images?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743148) |
| 4 | 9 | [External data behind an institutional data use agreement - does the KneeCoT ruling generalize?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737950) |
| 4 | 7 | [A lot of submissions are just forks of someone's solution](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737635) |
| 4 | 5 | [Data download](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734671) |
| 4 | 5 | [Submission running for more than 2 hours](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/738129) |
| 4 | 5 | [Kernel Stuck in queued for hours on GPU T4 -anyone else ?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743746) |
| 4 | 1 | [External data: are ungated, openly-licensed OAI-derivative datasets allowed?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/738156) |
| 4 | 0 | [What we've ruled out at 0.79 OOF — and what's left](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737597) |
| 3 | 4 | [Clarification on Training Data](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733836) |
| 3 | 4 | [Broken download link](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741916) |
| 3 | 4 | [How bad will the private LB shake-up be?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742327) |
| 3 | 3 | [train.csv has 4,407 studies and 58 labels — the other 4,349 have reports](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734055) |
| 3 | 2 | [Use model in competiton](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733478) |
| 3 | 2 | [Title: Expected GPU inference runtime after CPU preprocessing?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735647) |
| 3 | 2 | [Meme thread](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742053) |
| 3 | 1 | [How the ground truth labels are labelled ?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733864) |
| 3 | 1 | [Rules clarification: may Competition Data be sent to third-party LLM APIs?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733873) |
| 3 | 1 | [YOLO used to be useful in such competition.](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734357) |
| 3 | 0 | [How I Transferred the RSNA Dataset from Kaggle to Google Drive](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/736098) |
| 3 | 0 | [Has anyone tried YOLO / ROI cropping for the knee joint?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/736999) |
| 3 | 0 | [External data rule — datasets that are free to access but carry research-use agreements](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/738111) |
| 2 | 10 | [Looking for teammates—open to merging or collaborating.](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744511) |
| 2 | 6 | [Why ~200 public forks sit at exactly 0.941](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742050) |
| 2 | 5 | [Is the Osteoarthritis Initiative (OAI) permitted as external data?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819) |
| 2 | 3 | [train.csv: PatientSex is documented but not present in the file](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733423) |
| 2 | 3 | [Fluid_Sensitive and Fat_Suppression appear identical across all series](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737312) |
| 2 | 2 | [PSA: bf16 on Kaggle's T4/P100 silently cost me 0.14 AUC (0.851 → 0.710). How to spot it.](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744230) |
| 2 | 1 | [what is u label auc on gold?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/738172) |
| 2 | 1 | [Optimizations for blazingly fast submissions (<10 minutes!) + Looking for teammates :)](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743374) |
| 2 | 1 | [Imaging modalities / planes in private test set](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744519) |
| 2 | 0 | [Strong-pipeline replication: SOFT wins 3/3 seeds (+0.014 AUC), but the 58-study CI crosses zero](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734105) |
| 2 | 0 | [Title: Are RadImageNet ResNet-50 weights allowed for prize-eligible submissions?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745283) |
| 1 | 13 | [9h runtime question](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740474) |
| 1 | 12 | [Why do I feel that competitions like these are increasingly becoming accessible only to people with premium GPUs or Colab Pro subscriptions?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737449) |
| 1 | 11 | [Working with our own GPU](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/739000) |
| 1 | 6 | [Looking for guidance on MRI report terminology for the 12 abnormalities](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735596) |
| 1 | 5 | [looking for help](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744495) |
| 1 | 4 | [Scoring error - submission formatting](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/739200) |
| 1 | 4 | [One fold(No valid setup) model is better than Five fold ensemble models?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740375) |
| 1 | 3 | [Stuck around 0.74 OOF - what separates the 0.87 tier from 0.93+?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/738339) |
| 1 | 3 | [How do you optimize labels?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745214) |
| 1 | 2 | [Classifying labels into 12 abnormalities using Report in train.csv file](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734623) |
| 1 | 2 | [How to find issue with failed submission](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743779) |
| 1 | 1 | [[PSA] 3 code-submission gotchas: GPU incompatibility, missing DICOM codecs, input path quirks](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735854) |
| 1 | 1 | [EDA: Only 58/4407 training studies have ground-truth labels what that means for validation](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735855) |
| 1 | 1 | [How to use multilingual reports](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/736711) |
| 1 | 1 | [Unmentioned finding = negative, or unknown? (weak labels from report text)](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737650) |
| 1 | 1 | [Corrupted DICOM Pixel Data Found in 2 Training Studies (ValueError Traceback)](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/738708) |
| 1 | 1 | [Things that helped (and didn't) when approaching RSNA Knee from scratch](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740162) |
| 1 | 1 | [From AlexNet to DINOv3: A Practical Backbone History for the RSNA Knee MRI 2026 Challenge](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740164) |
| 1 | 0 | [Weak labels for all 12 findings + how recoverable each one actually is](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734117) |
| 1 | 0 | [Question about Winner License and Apache 2.0 Pretrained Models](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/739905) |
| 1 | 0 | [Six data traps in RSNA Knee: slice order, missing laterality, silent reports (EDA)](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741310) |
| 1 | 0 | [Download data problem](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742077) |
| 1 | 0 | [Rule 2.5.a.4 vs 2.8.a.1 — do weights trained on no-redistribution external data have to be published? (yes/no)](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743420) |
| 0 | 10 | [Hardware Specifications](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743303) |
| 0 | 6 | [Am new here: Is the Kaggle GPU enough for this competition?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743698) |
| 0 | 5 | [comple and run successfully but output submsson.csv not found error](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734676) |
| 0 | 4 | [No Reports for Test Data?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/738851) |
| 0 | 3 | [Is the dataset for this competition really 569.76 GB?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/739693) |
| 0 | 3 | [What is difference between study and series?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743534) |
| 0 | 3 | [Prize eligibility when using publicly shared weights trained on the competition data](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744056) |
| 0 | 2 | [Efficiency Prize: are GPU notebooks eligible, and what does RuntimeSeconds cover?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733475) |
| 0 | 2 | [About the "Report" in real test set](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733592) |
| 0 | 2 | [Clarification requested: exact ranking criteria for the Efficiency Track](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733753) |
| 0 | 2 | [Eligibility under 18](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733797) |
| 0 | 2 | [The leaderboard shows 3 decimals and I spent 20 submissions inside one of them](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740439) |
| 0 | 2 | [Clarification on Manual Annotation of Unlabeled Training MRIs](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740718) |
| 0 | 2 | [anybody here, maybe not using image models to do this???](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742190) |
| 0 | 1 | [[Baseline Notebook] 2.5D ResNet-34 PyTorch Pipeline (Fold 0) \| Preprocessed 3D Volumes](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733934) |
| 0 | 1 | [Eligibility question: minor participant + parental consent (Section 3.1.a.ii)](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733971) |
| 0 | 1 | [reports will be unavailable for the hidden test set?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734118) |
| 0 | 1 | [Doubt regarding Compute usgae](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735017) |
| 0 | 1 | [question about competition rule](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737018) |
| 0 | 1 | [Minor participation — is a Sponsor-approved exception possible?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737350) |
| 0 | 1 | [Has anyone looked at Pillar-0?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/738036) |
| 0 | 1 | [Which direction to focus on](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/739228) |
| 0 | 1 | [Metric edge cases: single-class label columns, and corrupted DICOMs in the test set](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740094) |
| 0 | 1 | [is the full DICOM dataset available?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742061) |
| 0 | 1 | [Can we select 2 final submissions?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743356) |
| 0 | 1 | [Looking for teammates: 0.942 solo, own 5-fold models + a GPU to share](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745091) |
| 0 | 1 | [The submission is stuck somehow. I need help.](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745441) |
| 0 | 0 | [Labeling the 4,349 report-only studies without an LLM, and what it scores on the 58 gold](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734095) |
| 0 | 0 | [Clarification on MIRA Section 6](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734131) |
| 0 | 0 | [Prize payment](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735295) |
| 0 | 0 | [Openly available datasets](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735296) |
| 0 | 0 | [Clarification on FrameOfReferenceUID in the released DICOM files](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/736457) |
| 0 | 0 | [Benchmarked the public report-label tables against the 58 annotated studies](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737454) |
| 0 | 0 | [Has anyone tried a GRU/LSTM head over slices here?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/738495) |
| 0 | 0 | [Final model release with mixed upstream weight licences](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/739758) |
| 0 | 0 | [Winner's obligations when the submitted weights are a public checkpoint someone else trained](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741212) |
| 0 | 0 | [Can't wait to see how the competition wraps up and check out all the new solutions!](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744055) |
| 0 | 0 | [Local validation vs public LB, and validating public resources](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744961) |
| -1 | 3 | [Only 58 Gold Labels.](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737418) |
| -1 | 1 | [Knee Joint Anatomy](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735323) |
| -1 | 1 | [can we clone other compititor model pipline and improve it??](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/736348) |
| -1 | 1 | [Errors on gold labels](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742106) |
| -1 | 1 | [Data downloading problem](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742474) |
| -1 | 0 | [RSNA Knee — submission 56338136 fails during hidden rerun; V20 CPU run succeeds](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742122) |
| -2 | 6 | [What kind of Hardware are using?(GPU,CPU,RAM,VRAM)](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737333) |
| -2 | 3 | [Your notebook hit an unhandled error while rerunning your code.](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734062) |
| -3 | 1 | [Title: Building an open analytics companion for this competition — what would you want in it?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742770) |
| -4 | 8 | [Suggestions for improving a 0.940 RSNA Knee ensemble](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742926) |
| -4 | 0 | [asking to join inn team](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/736497) |
| -5 | 5 | [Looking for a Team — RSNA Knee Abnormalities Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/736498) |
| -6 | 1 | [DOES Ensemble allowed in RSNA?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740624) |
| -8 | 2 | [🤝 [Team Merger] Looking for a 0.940+ Partner to Ensemble and Push for Gold](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741459) |
| -10 | 8 | [Multimodal Fusion Bottlenecks: Aligning 3D MRI Volume Embeddings with Multilingual Report Transformers](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742267) |
| -10 | 4 | [Team Finding](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735599) |
| -16 | 4 | [Stuck at the 0.940 Baseline , how to increase my score and what should be my next step ??](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741469) |
