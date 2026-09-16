# data/

**CSV만 둔다. DICOM은 여기 두지 않는다** (Kaggle 커널에서만 접근).

개별 파일만 받을 것 — `-f` 없이 실행하면 전체 데이터셋(수백 GB)을 받는다:

```bash
C=rsna-knee-abnormality-detection
kaggle competitions download -c $C -f train.csv              -p data/
kaggle competitions download -c $C -f sample_submission.csv  -p data/
unzip -o 'data/*.zip' -d data/ && rm -f data/*.zip   # 압축되어 받아진 경우
```

받은 뒤 레이블 순서 확정:

```bash
python scripts/sync_labels.py
```
