# RSNA Knee Abnormality Detection

Kaggle 대회 저장소. **학습·추론은 전부 Kaggle GPU 커널에서** 돌린다.
로컬은 코드 작성 / CSV EDA / 커널 푸시 전용이다.

과제: 무릎 DICOM의 12개 이상 소견 멀티라벨 분류.
평가: **macro-averaged AUC ROC** (+ RuntimeSeconds 기반 Efficiency Prize).

## 🚫 금지 규칙 (반드시 지킬 것)

- 절대 `kaggle competitions download -c ...` 를 `-f` 없이 실행하지 말 것.
  전체 데이터셋은 수백 GB이고, 로컬 디스크를 날린다.
  개별 파일만: `kaggle competitions download -c <slug> -f train.csv -p data/`
- 로컬에는 **CSV와 코드만** 둔다. DICOM은 Kaggle에만 존재한다.
- 푸시 전 항상 **subset 스모크 테스트** 통과 확인 (`DEBUG=True`, 소수 샘플).
- `kernel-metadata.json`의 `enable_internet`은 **항상 false**.
  (코드 대회 규칙 + 인터넷 켜면 제출 불가)
- API 토큰은 `~/.kaggle/access_token` 에만 두고 절대 커밋하지 않는다.
  (Kaggle 신방식: `KGAT_`로 시작하는 문자열. 구 `kaggle.json`은 더 이상 발급되지 않는다)

## 레이블 순서

제출 컬럼 순서가 어긋나면 AUC가 0.5 근처로 나오고 원인 찾는 데 시간이 오래 간다.
**`src/config.py`의 `LABELS`가 유일한 진실 공급원(single source of truth)** 이며,
`sample_submission.csv` 헤더와 일치해야 한다. 제출 직전 `config.verify_labels()`로 검증한다.

## 경로

로컬/Kaggle 경로 분기는 `src/paths.py`가 전부 처리한다.
노트북에서 하드코딩된 `/kaggle/input/...` 을 쓰지 말 것.

## 워크플로

1. 로컬에서 `kernels/train_fold0/train.ipynb` 수정
2. 스모크 테스트 (DEBUG 모드)
3. `bash scripts/push_kernel.sh train_fold0` 로 Kaggle에 푸시
4. Kaggle에서 GPU 실행 → 로그 확인 → 제출

`src/`를 고쳤으면 `bash scripts/push_src.sh`를 먼저 돌려야 커널에 반영된다.

## 스크립트

| 명령 | 역할 |
|---|---|
| `bash scripts/fetch_csv.sh` | 대회 CSV만 내려받기 (`-f` 사용) |
| `python scripts/sync_labels.py` | sample_submission.csv → `config.LABELS` 확정 |
| `bash scripts/push_src.sh` | `src/`를 `knee-src` 데이터셋으로 업로드 |
| `bash scripts/push_kernel.sh <name>` | 커널 푸시 (`enable_internet` 검사 포함) |

자세한 대회 정보는 `docs/competition.md` 참고.
