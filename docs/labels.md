# 판독문 레이블 테이블 벤치마크

`python scripts/bench_labels.py` 로 재생성한다

gold 58건에 대고 각 공개 테이블의 AUC 를 쟀다
gold 는 `train.csv` 에서 레이블이 채워진 스터디다

> 58건뿐이라 신뢰구간이 매우 넓다
> 레이블당 양성이 9~35개이므로 0.02 수준 차이는 해석하지 않는다

## 평가에서 제외한 테이블

아래 테이블은 gold 58건에 **정답을 그대로 복사**해 두었다
AUC 가 1.0 으로 나오지만 판독문 추출 품질과는 무관하다

- `barun2104 folds`
- `barun2104 pseudo`

학습에는 쓸 수 있다 (나머지 4,349건은 판독문 유래)
다만 **gold 로 검증할 수 없으므로 품질을 알 수 없다**
`barun2104` 의 `fold` 컬럼은 별도로 유용하다

## 테이블별 macro AUC

| 테이블 | 행 | gold 매칭 | macro AUC |
|---|---|---|---|
| stevenleehans v4_blend | 4,407 | 58 | **0.8927** |
| stevenleehans v2 | 4,407 | 58 | **0.8873** |
| stevenleehans full | 4,407 | 58 | **0.8780** |
| pilkwang v2 | 4,406 | 57 | **0.8700** |
| lixin73 gpt56sol | 4,407 | 58 | **0.8352** |
| lixin73 labels | 4,407 | 58 | **0.8352** |

## 레이블별 AUC

| 테이블 | ACL | MCL | Medial Meniscus | Lateral Meniscus | Medial OA | Lateral OA | PF OA | Effusion | Synovitis | Baker's | Contusion | Fracture |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stevenleehans v4_blend | 0.987 | 0.968 | 0.948 | 0.879 | 0.932 | 0.833 | 0.902 | 0.877 | 0.790 | 0.944 | 0.860 | 0.793 |
| stevenleehans v2 | 0.993 | 0.964 | 0.954 | 0.862 | 0.931 | 0.830 | 0.901 | 0.853 | 0.790 | 0.946 | 0.831 | 0.793 |
| stevenleehans full | 0.993 | 0.964 | 0.954 | 0.862 | 0.931 | 0.830 | 0.901 | 0.853 | 0.678 | 0.946 | 0.831 | 0.793 |
| pilkwang v2 | 0.997 | 0.976 | 0.943 | 0.841 | 0.908 | 0.789 | 0.891 | 0.830 | 0.694 | 0.932 | 0.768 | 0.871 |
| lixin73 gpt56sol | 0.931 | 0.921 | 0.890 | 0.866 | 0.891 | 0.824 | 0.808 | 0.696 | 0.676 | 0.903 | 0.814 | 0.801 |
| lixin73 labels | 0.931 | 0.921 | 0.890 | 0.866 | 0.891 | 0.824 | 0.808 | 0.696 | 0.676 | 0.903 | 0.814 | 0.801 |

## 쌍별 rank 블렌드 (상위 10)

독립적인 두 테이블을 평균하면 각각보다 낫다는 보고가 있어 확인했다 (디스커션 737454)

| 조합 | macro AUC |
|---|---|
| stevenleehans v4_blend  +  pilkwang v2 | **0.8940** |
| stevenleehans v4_blend  +  lixin73 gpt56sol | **0.8928** |
| stevenleehans v4_blend  +  lixin73 labels | **0.8928** |
| stevenleehans v2  +  lixin73 gpt56sol | **0.8917** |
| stevenleehans v2  +  lixin73 labels | **0.8917** |
| stevenleehans v2  +  pilkwang v2 | **0.8905** |
| stevenleehans v4_blend  +  stevenleehans v2 | **0.8904** |
| stevenleehans v4_blend  +  stevenleehans full | **0.8904** |
| stevenleehans v2  +  stevenleehans full | **0.8873** |
| stevenleehans full  +  lixin73 gpt56sol | **0.8824** |

## 결론

- 단일 최고: **stevenleehans v4_blend** (0.8927)
- 블렌드 최고: **stevenleehans v4_blend  +  pilkwang v2** (0.8940)
- 차이: +0.0013

### 출처

| 데이터셋 | 라이선스 |
|---|---|
| `stevenleehans/rsna-knee-llm-report-labels` | CC0-1.0 |
| `pilkwang/rsna-knee-llm-labels` | CC0-1.0 |
| `lixin73/rsna-knee-llm-report-labels-sol56` | CC0-1.0 |
| `barun2104/rsna-knee-stratified-folds-and-llm-soft-labels` | other |