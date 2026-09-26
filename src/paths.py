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

def _find_competition_dir(input_root: Path) -> Path:
    """대회 데이터 디렉터리를 런타임에 찾는다.

    경로를 하드코딩하면 안 된다. 커널에서 대회 데이터가
    /kaggle/input/competitions/<slug>/ 아래에 놓이는 경우가 보고됐다
    (디스커션 735854). 후보를 순서대로 보고 train.csv 가 있는 곳을 고른다.
    """
    candidates = [
        input_root / COMP_SLUG,
        input_root / "competitions" / COMP_SLUG,
    ]
    for c in candidates:
        if (c / "train.csv").exists():
            return c
    # 얕은 깊이에서만 탐색한다. train_series/ 를 훑으면 끝나지 않는다.
    for d in sorted(input_root.glob("*/")) + sorted(input_root.glob("*/*/")):
        if (d / "train.csv").exists() and (d / "train_series").exists():
            return d
    return candidates[0]


if IS_KAGGLE:
    ROOT = Path("/kaggle")
    INPUT_ROOT = ROOT / "input"
    DATA_DIR = _find_competition_dir(INPUT_ROOT)
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


def find_input(name: str, max_depth: int = 3) -> Path:
    """/kaggle/input 아래에서 이름이 name 인 디렉터리를 찾는다.

    glob("**", recursive=True) 를 쓰면 안 된다.
    대회 DICOM 이 819,640 개라 파일 하나 찾는 데 10분 이상 걸린다
    (실측: 커널에서 두 번 호출해 24분 소모).

    알려진 위치를 먼저 보고, 없으면 깊이를 제한해 훑는다.
    train_series / test_series 는 내려가지 않는다.
    """
    root = INPUT_ROOT
    for c in (root / name, root / "datasets" / name):
        if c.exists():
            return c

    skip = {"train_series", "test_series"}

    def walk(d: Path, depth: int) -> Path | None:
        if depth > max_depth:
            return None
        try:
            entries = list(d.iterdir())
        except OSError:
            return None
        for e in entries:
            if e.name == name:
                return e
        for e in entries:
            if e.is_dir() and e.name not in skip:
                hit = walk(e, depth + 1)
                if hit is not None:
                    return hit
        return None

    hit = walk(root, 1)
    if hit is None:
        raise FileNotFoundError(f"{name} 을 {root} 아래에서 못 찾음")
    return hit


def kaggle_input(dataset_name: str, owner: str | None = None) -> Path:
    """추가로 붙인 Kaggle 데이터셋(사전학습 가중치 등) 경로.

    평평한 /kaggle/input/<name> 과 /kaggle/input/datasets/<owner>/<name>
    둘 다 나타날 수 있다 (디스커션 735854). 존재하는 쪽을 돌려준다.
    """
    flat = INPUT_ROOT / dataset_name
    if flat.exists() or owner is None:
        nested = INPUT_ROOT / "datasets" / (owner or "") / dataset_name
        return flat if flat.exists() or owner is None else nested
    nested = INPUT_ROOT / "datasets" / owner / dataset_name
    return nested if nested.exists() else flat


# 손상된 DICOM - 캐시 빌드 시 건너뛰거나 try/except 로 감쌀 것 (디스커션 737163, 738708)
CORRUPT_STUDIES = (
    "1.2.826.0.1.3680043.8.498.34685905030370793639196564723935583035",
    "1.2.826.0.1.3680043.8.498.37833587429731221455928642963031995680",
)


def describe() -> str:
    env = "Kaggle" if IS_KAGGLE else "Local"
    return (
        f"[{env}] DATA_DIR={DATA_DIR} (exists={DATA_DIR.exists()})\n"
        f"        OUTPUT_DIR={OUTPUT_DIR}"
    )


if __name__ == "__main__":
    print(describe())
