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

## 주의사항

- 커널 `model_sources` / `kernel_sources` 누락 시 앙상블 멤버가 조용히 빠진다
  (실제로 0.940 → 0.937로 떨어진 사례)
- 외부 무릎 MRI 데이터셋 사용 가부는 별도 디스커션에서 다룸 (743416, 733652)
- 상용 LLM API 사용 규칙도 별도 스레드 있음 (733965)
