# RSNA Knee - 진행 기록

**이 문서 하나에 진행 상황과 결과를 전부 기록한다.**
실험을 돌릴 때마다 "실험 기록" 표에 한 줄 추가하고, 결과가 나오면 성적표를 갱신한다
새 세션을 시작할 때는 이 문서부터 읽으면 된다

2026-09-16 셋업 시작 / 마감 2026-10-22 / 마지막 갱신 2026-09-29

## 결과 요약

| 시점 | 구성 | val | gold | LB |
|---|---|---|---|---|
| 1차 | 224px 단독 30 epoch | 0.8420 | 0.8928 | **0.873** |
| 2차 | 224 + 336 rank 평균 | 0.8500 | 0.8758 | 채점 중 |

val 은 판독문 유래 타깃 기준(904건), gold 는 전문의 레이블(12건)
LB 가 val 보다 높은 것은 검증 타깃 자체에 노이즈가 있기 때문이다

## 실험 기록

날짜는 한국 시간

| # | 날짜 | 구성 | val | gold | LB | 비고 |
|---|---|---|---|---|---|---|
| 1 | 09-28 | 224px 8ep lr 1e-4 전체 unfreeze | 0.6622 | - | - | 마지막까지 상승 중 |
| 2 | 09-28 | 224px 24ep 동일 | 0.6933 | 0.6390 | - | **ep0 이 최고** |
| 3 | 09-28 | 224px 12ep lr 8e-6 증강 prior | 0.7037 | 0.6390 | - | **ep0 이 최고** |
| 4 | 09-28 | 224px 10ep head 단순화 + 손실 정렬 | 0.8260 | 0.8487 | - | 패턴 깨짐 |
| 5 | 09-28 | 224px 30ep | 0.8420 | 0.8928 | **0.873** | ep18 최고 |
| 6 | 09-29 | 336px 30ep | 0.8426 | 0.8826 | - | ep16 최고 |
| 7 | 09-29 | 224 + 336 rank 평균 | 0.8500 | 0.8758 | 채점 중 | OOF 기준 +0.008 |

## Kaggle 자산

| 데이터셋 / 커널 출력 | 내용 |
|---|---|
| `hyejinkwak/knee-cache-build` | 224px 캐시 8칸 9슬 14.8 GiB |
| `hyejinkwak/knee-cache-336` | 336px 캐시 6칸 6슬 16.7 GiB |
| `hyejinkwak/knee-src` | 코드 모듈 10개 |
| `hyejinkwak/knee-dicom-wheels` | 오프라인 DICOM 코덱 |
| `hyejinkwak/knee-dinov2-weights` | 224/336 재샘플링 가중치 |
| `hyejinkwak/knee-weights` | `fold0_224.pt`, `fold0_336.pt` |
| `stevenleehans/rsna-knee-llm-report-labels` | 판독문 레이블 (공개 CC0) |

| 커널 | 역할 | GPU |
|---|---|---|
| `knee-cache-build` / `knee-cache-336` | 픽셀 캐시 빌드 | 없음 |
| `knee-cache-verify` | 캐시 마운트 검증 | 없음 |
| `knee-train-fold0` | 224px 학습 | T4 |
| `knee-train-336` | 336px 학습 | T4 |
| `knee-submit` | 추론 및 제출 | T4 |

## 확정된 설정

학습 (루트 노트북 `pilkwang` 0.891 과 대조해 맞춘 값)

    인코더 lr 8e-6, head lr 1e-3, unfreeze 마지막 6블록
    weight decay 0.02, OneCycleLR pct_start 0.15, 30 epoch
    batch 8 x accum 2 (336 은 4 x 4)
    침묵 셀 가중 0.05, 손실은 (BCE*w).mean()
    9슬라이스 중 한 그룹만 학습 (검증/추론은 전부 평균)
    SlotHead, slot_prior 고정 buffer
    증강 회전 ±8도 / 스케일 ±8% / 이동 ±5% / 밝기 ±10%, 좌우 뒤집기 없음

데이터

    plan C (평면 통합) - T1_1 = COR_T1 or SAG_T1 or T1_EXTRA
    타깃 stevenleehans v4_blend, 0.25 가 "판독문 미언급" 표시
    fold 는 1개만 쓴다. 5-fold 는 마감 직전 최종 제출용

## 1. 대회 구조 파악

디스커션 133개와 공개 커널 40개를 전부 수집해 읽었다
(`scripts/collect_discussions.py`, `scripts/collect_kernels.py`)

핵심 발견

- **레이블이 병목이다.** 4,407 스터디 중 전문의 레이블은 58개뿐이고
  나머지는 판독문 텍스트만 있다. `test.csv` 에는 판독문이 없다
  즉 텍스트는 타깃 생성용이지 모델 입력이 아니다
- 공개 상위 노트북이 전부 `pilkwang/rsna-knee-baseline-v1`(0.891) 한 뿌리에서 갈라졌다
  0.94대는 그 위에 공개 가중치 14~17개를 앙상블한 것이다
- 인코더를 키우는 것은 의미가 없다 (DINOv2 S->Base 가 노이즈 바닥 대비 null)

## 2. 데이터 EDA

`notebooks/eda.ipynb`, `docs/eda.md`

- 레이블 보유 스터디 58/4,407 (1.32%), 부분 레이블은 없다
- 판독문은 다국어다. 영어 40%, 네덜란드어 13%, 터키어 12%, 그리스 7%, 키릴 5%
- **`Fluid_Sensitive` 와 `Fat_Suppression` 이 24,371개 전부 동일하다**
  이름은 독립인 두 속성인데 주어진 컬럼은 한 축만 담는다
  DICOM 헤더의 TR/TE 로 다시 판별해야 한다
- 모든 스터디가 Sagittal/Coronal/Axial 을 갖는다
- 촬영 프로토콜이 81가지다. 기관마다 어느 시퀀스를 찍는지 다르다

## 3. 레이블 확보

`docs/labels.md`, `scripts/bench_labels.py`

공개 판독문 레이블 테이블 8종을 gold 58건에 대고 재측정했다
공개 보고(디스커션 737454)와 소수점 3자리까지 재현된다

| 테이블 | macro AUC |
|---|---|
| stevenleehans v4_blend | **0.8927** |
| stevenleehans v2 | 0.8873 |
| pilkwang v2 | 0.8700 |

`barun2104` 테이블은 gold 58건에 정답이 복사돼 있어 제외했다 (AUC 1.0)
누수 자동 탐지를 스크립트에 넣었다

**`v4_blend` 에서 `0.25` 가 "판독문 미언급" 표시라는 것을 확인했다**
`llm_labels_full` 의 `0.5` 와 99.9% 일치한다. 그 셀에 손실 가중 0.05 를 준다

## 4. 픽셀 캐시

DICOM 500GB 중 모델이 보는 것은 일부뿐이다. 한 번 디코드해 저장하면
학습마다 드는 55분이 사라진다

| 캐시 | 구성 | 용량 | mm/px |
|---|---|---|---|
| `knee-cache-build` | 224px 8칸 9슬 | 14.8 GiB | 0.5804 |
| `knee-cache-336` | 336px 6칸 6슬 | 16.7 GiB | 0.3869 |

조용히 틀리는 지점 세 가지를 코드로 막았다 (`src/dicom_io.py`)

- **파일명 정렬은 물리 순서와 무관하다** (순위상관 0.01)
  `ImagePositionPatient · n̂` 로 복원한다
  실측: 기하 순서 27.4 vs 파일명 순서 58.5 (인접 슬라이스 차이)
- **물리 크롭이 원본보다 크면 조용히 무시된다.** `applied=False` 로 알린다
- 시퀀스 판별을 CSV 가 아니라 DICOM 헤더에서 한다 (UNK 7.5% -> 0.0%)

마운트 검증에서 **인덱스와 배열 정렬 불일치 0건**을 확인했다
어긋나면 이미지와 레이블이 뒤바뀐 채로 학습된다

## 5. 학습 - 루트 노트북과의 대조가 전환점

처음 세 번은 **epoch 0 이 최고**였다. 학습이 모델을 망가뜨리고 있었다

| 실행 | 설정 | best |
|---|---|---|
| 1 | 8ep, lr 1e-4, 전체 unfreeze | 0.6622 |
| 2 | 24ep, 동일 | 0.6933 (ep0) |
| 3 | 12ep, lr 8e-6, 증강, prior | 0.7037 (ep0) |
| 4 | **head 단순화 + 손실 4종 정렬** | **0.8260** |
| 5 | 30 epoch | **0.8420** |

루트 노트북 구현을 읽고 5개 차이를 맞춘 것이 4번이다

1. **head 를 SlotHead 로 교체** (0.7M -> 0.107M)
   `nn.MultiheadAttention` 으로 슬롯 6개를 보고 있었다
   > With a study-level label there is no signal telling the model which part
   > of a study matters, so extra attention parameters below the slot level
   > would have nothing to learn from and would spend their capacity fitting noise
2. **침묵 셀 가중 0.3 -> 0.05** (전체 셀의 19%)
3. **손실을 `(BCE*w).mean()` 으로** (가중평균이 아니다)
4. **9슬라이스 중 한 그룹만 학습에 쓴다** (인코더 입력 18장 -> 6장, 3배 빠름)
5. **OneCycleLR** (cosine 은 0 까지만 내려간다)

## 6. 해상도 실험

224 는 0.5804 mm/px 로 1mm 구조의 Nyquist 한계(0.5)를 넘는다
336 은 0.3869 로 통과한다

전체 성능은 동률이었다 (0.8420 vs 0.8426) 그러나 레이블별로 갈렸다

| 336 이 나음 | | 224 가 나음 | |
|---|---|---|---|
| ACL | +0.027 | MCL | -0.033 |
| Contusion | +0.024 | Baker's | -0.017 |
| Lateral Meniscus | +0.011 | Effusion | -0.011 |

MCL 이 떨어진 것은 슬라이스가 9->6 으로 줄어든 영향으로 보인다
(출력 한도 20GB 때문에 336 에서 슬라이스를 줄였다)

**서로 다른 것을 보므로 앙상블에 유리하다**
OOF 로 미리 측정한 결과 rank 평균이 val 0.8500 (+0.008)
모델 간 상관은 0.83~0.93 이다

## 7. 겪은 사고와 대응

에러가 날 때마다 그 지점을 구조적으로 막았다

| 사고 | 원인 | 대응 |
|---|---|---|
| 6칸 캐시로 2시간 빌드 | `src/` 를 고치고 Kaggle 에 안 올림 | `check_src_sync.py` 가 푸시를 막는다 |
| `NameError: CACHE` | 노트북 셀을 문자열로 자르다 정의 삭제 | `check_notebook.py` 가 미정의 이름을 잡는다 |
| 24분 경로 탐색 | `glob(recursive=True)` 가 DICOM 80만개를 훑음 | `paths.find_input()` 깊이 제한 |
| `IndexError: group 2` | `predict()` 가 그룹 수를 3으로 하드코딩 | 모델에서 읽는다 |
| 예측이 상수 | 슬롯 배정 로직이 두 군데 복사됨 | `assign_slots()` 하나로 합침 |

마지막 것은 `assert spread > 1e-6` 이 잡아서 **잘못된 제출을 막았다**

## 8. 남은 것

1. **공개 가중치 앙상블** - 우리 모델끼리는 상관이 0.83~0.93 으로 높다
   구조가 다른 멤버(CoAtNet, RadImageNet, Raptor)가 필요하다
   `pjmathematician/rsna-knee-d4-lite` 의 추론 코드에 우리 체크포인트를 추가하는 방식
   각 체크포인트의 전처리 계약을 그대로 재현해야 한다
2. **5-fold** - 마감 직전 최종 제출용
3. **MCL 0.7481 / Lateral Meniscus 0.7598** 개선

## 작업 규칙

- `src/` 를 고치면 `bash scripts/push_src.sh` 를 먼저 돌린다
  `push_kernel.sh` 가 동기화를 검사해 막는다
- 노트북을 스크립트로 수정했으면 `scripts/check_notebook.py` 로 확인한다
- 커널 출력 한도 20GB. 336px 로 9슬라이스를 담으면 25GB 로 초과한다
- Kaggle 큐는 한국 시간 밤에 길고 오전에 짧다 (실측: 22시 43분 대기, 10시 즉시)
- GPU 할당량 주당 30시간, 매주 금요일 갱신

## 참고

- `docs/competition.md` 대회 규칙과 데이터 구조
- `docs/discussions.md` 디스커션 133개 정리
- `docs/baseline.md` 설계 근거
- `docs/eda.md` 데이터 분석
- `docs/labels.md` 레이블 테이블 벤치마크
- `docs/status.md` 현재 상태와 Kaggle 자산
