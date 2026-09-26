"""로컬 / Kaggle 커널 경로 자동 분기.

노트북에서는 하드코딩된 경로 대신 항상 이 모듈을 쓴다.

    from paths import TRAIN_CSV, OUTPUT_DIR, on_kaggle
"""

from __future__ import annotations

import os
from pathlib import Path

COMP_SLUG = "rsna-knee-abnormality-detection"


def on_kaggle() -> bool:
    """Kaggle 커널 안에서 실행 중인지."""
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle/input").exists()


IS_KAGGLE = on_kaggle()

if IS_KAGGLE:
    ROOT = Path("/kaggle")
    INPUT_ROOT = ROOT / "input"
    DATA_DIR = INPUT_ROOT / COMP_SLUG
    OUTPUT_DIR = ROOT / "working"
else:
    ROOT = Path(__file__).resolve().parents[1]
    INPUT_ROOT = ROOT / "data"
    DATA_DIR = ROOT / "data"
    OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if not IS_KAGGLE:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- 대회 파일 ---
TRAIN_CSV = DATA_DIR / "train.csv"                    # 스터디 레이블 + Report
TEST_CSV = DATA_DIR / "test.csv"                      # StudyInstanceUID 만
TRAIN_SERIES_CSV = DATA_DIR / "train_series.csv"      # 시리즈 메타데이터
TEST_SERIES_CSV = DATA_DIR / "test_series.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"

# DICOM: Kaggle에만 존재. <study_uid>/<series_uid>/<instance_uid>.dcm
TRAIN_SERIES_DIR = DATA_DIR / "train_series"
TEST_SERIES_DIR = DATA_DIR / "test_series"


def series_dir(study_uid: str, series_uid: str, train: bool = True) -> Path:
    """한 시리즈의 DICOM 디렉터리."""
    root = TRAIN_SERIES_DIR if train else TEST_SERIES_DIR
    return root / study_uid / series_uid

# --- 산출물 ---
SUBMISSION_CSV = OUTPUT_DIR / "submission.csv"
OOF_CSV = OUTPUT_DIR / "oof.csv"


def kaggle_input(dataset_name: str) -> Path:
    """추가로 붙인 Kaggle 데이터셋(사전학습 가중치 등) 경로."""
    return INPUT_ROOT / dataset_name


def describe() -> str:
    env = "Kaggle" if IS_KAGGLE else "Local"
    return (
        f"[{env}] DATA_DIR={DATA_DIR} (exists={DATA_DIR.exists()})\n"
        f"        OUTPUT_DIR={OUTPUT_DIR}"
    )


if __name__ == "__main__":
    print(describe())
