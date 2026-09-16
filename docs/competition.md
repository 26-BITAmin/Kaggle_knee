# RSNA Knee Abnormality Detection — 대회 요약

- 대회 페이지: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
- slug: `rsna-knee-abnormality-detection`

## 과제

무릎 영상(DICOM)에서 **12가지 이상(abnormality)** 을 각각 예측하는 멀티라벨 분류.
테스트셋 각 행마다 12개 타깃의 confidence score를 제출한다.

## 평가

- **Macro-averaged AUC ROC** — 12개 타깃 각각의 ROC AUC를 평균.
  → 타깃별로 양성 빈도가 크게 다를 수 있으므로, 희귀 타깃 하나가 점수를 크게 흔든다.
    fold 분할 시 타깃별 양성 수를 반드시 확인할 것.
- **Efficiency Prize**: 제출 평가에 걸린 `RuntimeSeconds` 를 최소화하는 별도 상.
  → 추론 속도(이미지 크기, TTA 횟수, 모델 크기)가 점수 외에도 의미가 있다.

## 형식

- 코드 대회. 학습·추론 모두 Kaggle 커널에서 수행.
- `enable_internet: false` (항상). 외부 가중치는 Kaggle Dataset으로 올려서 붙인다.
- 영상은 DICOM. 로컬에 내려받지 않는다 — 전체 데이터셋은 수백 GB.

## 레이블 순서 (12개)

`src/config.py`의 `LABELS`가 단일 진실 공급원.
`sample_submission.csv`를 받은 뒤 `python scripts/sync_labels.py`로 확정한다.

> 상태: **미확정** — kaggle.json 인증 후 채울 것.

## 확인 필요 (TBD)

대회 페이지가 JS 렌더링이라 로컬에서 자동 수집이 안 된다. 인증 후 확인할 것:

- [ ] 제출 마감일 / 팀 병합 마감일
- [ ] `train.csv` 컬럼 (patient_id, study_id, series_id 등 그룹 분할용 키)
- [ ] 이미지 단위 vs 스터디 단위 예측
- [ ] 커널 런타임 제한 (GPU 시간)
- [ ] Efficiency Prize의 구체적 산식
