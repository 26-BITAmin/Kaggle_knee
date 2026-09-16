# Kaggle - RSNA Knee Abnormality Detection

무릎 DICOM 영상의 12개 이상 소견 멀티라벨 분류

평가는 macro AUC ROC

**학습/추론은 Kaggle GPU 커널에서 수행하고, 이 저장소는 코드와 CSV만 관리한다.**

## 셋업

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Kaggle 토큰: kaggle.com/settings/api 에서 발급받은 KGAT_ 문자열
mkdir -p ~/.kaggle && echo <KGAT_토큰> > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token

bash scripts/fetch_csv.sh      # CSV만 받는다 (-f 사용)
python scripts/sync_labels.py  # 레이블 순서 확정
```

## 구조

```
CLAUDE.md                     작업 규칙 (Claude Code용)
docs/competition.md           대회 요약
data/                         CSV만. DICOM 금지
src/config.py                 레이블 순서 + 학습 설정 + 제출 파일 생성
src/paths.py                  로컬/Kaggle 경로 자동 분기
kernels/train_fold0/          Kaggle 커널 (notebook + metadata)
scripts/                      CSV 받기 / 레이블 동기화 / 커널·코드 푸시
```

## 커널 푸시

```bash
bash scripts/push_src.sh          # src/ 를 knee-src 데이터셋으로 업로드
bash scripts/push_kernel.sh train_fold0
```

`src/` 를 고칠 때마다 `push_src.sh` 를 다시 돌려야 커널에 반영된다.
