# RSNA Knee Abnormality Detection - 대회 요약

- 페이지: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
- slug: `rsna-knee-abnormality-detection`
- 마감: **2026-10-22 23:59 UTC**
- 상금: $77,000 (Research) / 참가 3,832팀
- 참가 상태: 규칙 동의 완료 (`userHasEntered=True`, 계정 `hyejinkwak`)

## 과제

무릎 MRI(DICOM)에서 12개 소견을 각각 이진 분류하는 멀티라벨 문제

레이블은 **검사(study) 단위**, 한쪽 무릎 기준
애매한 경우는 specificity를 위해 음성으로 처리됨

## 평가

**Macro-averaged AUC ROC** (12개 컬럼 AUC의 평균)
별도로 `RuntimeSeconds` 기반 **Efficiency Prize**

## 12개 레이블

판독 기준 요약 (대회 Overview 디스커션 733343)

| # | 소견 | 양성 기준 |
|---|---|---|
| 1 | ACL tear | 고도 부분 또는 완전 파열 (섬유 50% 초과 단절) |
| 2 | MCL tear | 고도 부분 또는 완전 급성 파열 + 주변 부종 |
| 3 | Medial meniscus tear | 2장 이상에서 반월상 표면에 닿는 이상 신호, 또는 형태 이상 |
| 4 | Lateral meniscus tear | 외측 반월상에 동일 기준 적용 |
| 5 | Medial compartment OA | 내측 구획 연골 두께 50% 초과 소실이 약 1cm 이상 |
| 6 | Lateral compartment OA | 외측 구획에 동일 기준 |
| 7 | Patellofemoral compartment OA | 슬개대퇴 구획에 동일 기준 |
| 8 | Joint effusion | 중등도 이상 관절액 |
| 9 | Synovitis | 활막 염증 및 비후 |
| 10 | Baker (popliteal) cyst | 슬와부 중등도 이상 액체 저류 |
| 11 | Contusion | 골절선 없는 골수 부종 (타박) |
| 12 | Acute fracture | 급성 피질 골절선 |

> 실제 제출 컬럼명과 순서는 `sample_submission.csv` 기준
> `python scripts/sync_labels.py` 로 `src/config.py`에 확정한다 (아직 미실행)

## 데이터 구조

API로 확인한 실제 파일 (`kaggle competitions files`)

```
sample_submission.csv     470 B
test.csv                  212 B
test_series.csv         2,213 B
test_series/<study_uid>/<series_uid>/<instance_uid>.dcm    각 1.8 MB
```

- 학습 스터디 4,407개
- DICOM 1장당 약 1.8 MB - 전체 수백 GB이므로 로컬 다운로드 금지
- 한 검사에 여러 series(시퀀스, 평면)가 있고 슬라이스를 가로질러 판단해야 함
- fluid sensitive 시퀀스(지방억제 T2/PD)가 대부분 소견 탐지의 핵심

## 이 대회의 핵심 구조: 레이블이 병목

**`train.csv`의 gold label은 4,407개 중 58개뿐이고 나머지는 NaN이다 (1.3%)**

대신 `train.csv`에 `Report` 컬럼(원본 판독문, 다국어)이 있다
`test.csv`에는 `Report`가 **없다** - 즉 텍스트는 **타깃 생성용이지 모델 입력이 아니다**

그래서 사실상 모든 참가자가 판독문에서 레이블을 만들어 학습한다
레이블 품질이 downstream 성능의 천장이다 (디스커션 733932)

- regex/사전 기반 추출: gold 58건 대비 macro AUC **0.8136**
- LLM으로 판독문 독해: **0.8780**
- 전체 레이블 셀의 **25.4%가 "판독문이 언급하지 않음"**

"언급 없음"의 의미가 소견마다 다르다는 점이 중요하다

| 소견 | gold AUC | 미언급 비율 |
|---|---|---|
| Synovitis | 0.678 | 83.7% |
| Baker's | 0.946 | 48.2% |
| Fracture | 0.793 | 42.9% |
| ACL | 0.993 | 8.3% |
| Medial meniscus | 0.954 | 5.5% |

Synovitis는 흔하지만 판독문에 거의 안 적힌다 (gold 58건 중 27건이 양성)
Effusion 필드가 Synovitis를 더 잘 예측한다 (0.7115 vs 0.6780, 동반 발생)
미결정 셀만 effusion으로 채우면 0.678 → 0.790, 전체 키 0.8780 → 0.8873
단, 이걸 12개 전부에 일반화하면 오히려 나빠진다 (0.8805)

## 공개 레이블 데이터셋 (판독문 유래)

| 제작자 | 데이터셋 |
|---|---|
| Pilkwang Kim | `pilkwang/rsna-knee-llm-labels` (최초, 2026-08-06) |
| barun2104 | `barun2104/rsna-knee-stratified-folds-and-llm-soft-labels` |
| lixin73 | `lixin73/rsna-knee-llm-report-labels-sol56` |
| stevenleehans | `stevenleehans/rsna-knee-llm-report-labels` |

## 리더보드 현황

- 공개 베이스라인 포크 약 200팀이 **정확히 0.941**에 몰려 있음
- 단일 모델 최고 공개 언급: 0.887~0.903 (DINOv2)
- 0.92~0.94대는 대부분 20개 이상 체크포인트 앙상블
- 주력 백본: **DINOv2/DINOv3**, CoAtNet
- `mattiaangeli/bend-the-knee-to-the-dinosaurs`의 `.ipynb` 메타데이터에
  버전별 `rsna_optimization` 항목이 있어 사실상 ablation 로그로 쓸 수 있음
  (`kaggle kernels pull mattiaangeli/bend-the-knee-to-the-dinosaurs -m`)

## 추론 속도 (Efficiency Prize)

CoAtNet 기준 40분 → 10분 이하로 줄인 사례 (디스커션 743374)

- GPU당 모델 1벌 + study 샤딩 (DataParallel 대신)
- DICOM을 스레드로 동시 1회 읽기: 단일 59 MB/s → 16스레드 372 MB/s
- pydicom 대신 헤더 직접 파싱: 0.307 → 0.058 ms/file
- PixelData 오프셋에서 `np.frombuffer` 직접 읽기: 1.22 → 0.33 ms/slice
- 1/99 백분위수를 strided 서브샘플로 계산 (np.percentile 정렬이 디코드의 60%)
- autocast 없이 fp16 가중치 + half 입력: T4에서 2.08배
- uint8로 GPU 전송 후 device에서 정규화, pinned memory, persistent workers

## 규칙 - 외부 데이터와 도구

### 판독문을 외부 LLM 에 보내도 되는가 - **허용**

호스트 공식 입장 (디스커션 733965, 규칙 2.6.b)

> submitting Competition Data, including report text, to an external LLM or API
> for inference or other computational processing (for example, extracting labels
> from reports) will not, by itself, be considered prohibited PRIVATE SHARING

단서는 두 가지다

- 서비스가 모든 참가자에게 합리적으로 접근 가능할 것
- 비용이 최소일 것

호스트가 특정 서비스의 접근성과 비용을 판단할 권한을 유보하고 있다

### 다른 참가자의 공개 데이터셋을 써도 되는가 - **허용**

금지되는 것은 **비공개 공유**다

> The PRIVATE SHARING restriction is intended to prohibit sharing Competition Data,
> code, or competition-specific work product with other participants, teams, or
> third parties for collaboration or competitive use outside the registered Team

Kaggle 에 전체 공개된 데이터셋은 반대 경우다
규칙이 요구하는 "equally accessible to all participants at no cost" 를 만족한다

우리가 쓰는 것

| 데이터셋 | 라이선스 | 용도 |
|---|---|---|
| `stevenleehans/rsna-knee-llm-report-labels` | CC0-1.0 | 학습 타깃 (v4_blend) |

### 판정 기준은 "누구나 같은 조건으로 접근 가능한가"

호스트가 실제로 **불허**한 사례 (734109)

> KneeCoT: 기관과의 공식 데이터 사용 동의가 필요해서, 동의를 받을 수 있는
> 참가자와 없는 참가자 사이에 불공정이 생긴다

| 대상 | 접근 조건 | 판정 |
|---|---|---|
| 공개 Kaggle 데이터셋 (CC0) | 계정만 있으면 누구나 | 문제없음 |
| KneeCoT | 기관 동의서 필요 | **불허** (호스트 확인) |
| Stanford MRNet, OAI | 등록 필요 | 질의 중 (735497, 741819) |

### 우승자 공개 의무와 라이선스 충돌

규칙 2.8.a 는 우승자에게 학습/추론 코드와 가중치 공개를 요구한다
남의 자산에 기반했으면 그 라이선스가 걸린다

- 우리가 쓰는 레이블은 **CC0-1.0** 이라 제약이 없다
- **RadImageNet 은 CC-BY-NC-SA-4.0 이다** - 공개 의무와 충돌 가능성이 제기됐다 (735121)
  Synovitis 개선용으로 검토할 때 라이선스를 먼저 확인할 것 (737566)
- DINOv3 도 별도 라이선스가 있고, 등록 여부 질의가 있었다 (733313)
- 재배포 금지 가중치와 2.8.a 의 충돌은 여러 스레드에서 다룬다 (739758, 741212, 743420)
- Apache-2.0 사전학습 모델과 우승자 라이선스 관계 질의도 있다 (739905)

> 우리는 DINOv2 (Apache-2.0 계열 timm 배포) 를 쓴다

## 주의사항

- 커널 `model_sources` / `kernel_sources` 누락 시 앙상블 멤버가 조용히 빠진다
  (실제로 0.940 → 0.937로 떨어진 사례)
- 외부 무릎 MRI 데이터셋 사용 가부는 별도 디스커션에서 다룸 (743416, 733652)
- 상용 LLM API 사용 규칙도 별도 스레드 있음 (733965)
