"""대회 설정 - 레이블 순서의 단일 진실 공급원(single source of truth).

제출 컬럼 순서가 sample_submission.csv와 어긋나면 AUC가 0.5 근처로 나온다.
제출 파일은 반드시 `make_submission()`으로 만들고, 저장 전 `verify_labels()`를 돌린다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from paths import SAMPLE_SUBMISSION_CSV

# ---------------------------------------------------------------------------
# 레이블 정의
# ---------------------------------------------------------------------------
# sample_submission.csv 헤더에서 확정한 순서 (2026-09-26)
# `python scripts/sync_labels.py` 로 재생성되지만 이미 확정됐으므로 바꾸지 말 것
# 제출 직전 verify_labels() 가 이 순서를 sample_submission.csv 와 대조한다
ID_COL: str = "StudyInstanceUID"
LABELS: tuple[str, ...] = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)

N_CLASSES = len(LABELS)


# ---------------------------------------------------------------------------
# 학습 설정
# ---------------------------------------------------------------------------
@dataclass
class CFG:
    seed: int = 42
    n_folds: int = 5
    fold: int = 0

    # 스모크 테스트: 소수 샘플 + 1 epoch. 푸시 전 항상 True로 한 번 통과시킬 것.
    debug: bool = False
    debug_samples: int = 64

    model_name: str = "tf_efficientnet_b0"
    image_size: int = 384
    batch_size: int = 16
    epochs: int = 5
    lr: float = 1e-4
    weight_decay: float = 1e-2
    num_workers: int = 2
    amp: bool = True

    labels: tuple[str, ...] = field(default_factory=lambda: LABELS)

    def __post_init__(self) -> None:
        if self.debug:
            self.epochs = 1


# ---------------------------------------------------------------------------
# 레이블 검증 / 제출 파일 생성
# ---------------------------------------------------------------------------
def labels_from_sample_submission(path: Path = SAMPLE_SUBMISSION_CSV) -> tuple[str, tuple[str, ...]]:
    """sample_submission.csv에서 (id_col, labels)를 읽어온다."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 없음. 먼저 실행: "
            f"kaggle competitions download -c rsna-knee-abnormality-detection "
            f"-f sample_submission.csv -p data/"
        )
    cols = list(pd.read_csv(path, nrows=1).columns)
    return cols[0], tuple(cols[1:])


def verify_labels(path: Path = SAMPLE_SUBMISSION_CSV) -> None:
    """config의 레이블 순서가 sample_submission.csv와 정확히 일치하는지 확인."""
    if not LABELS:
        raise RuntimeError("config.LABELS가 비어 있다. `python scripts/sync_labels.py` 실행할 것.")
    sub_id, sub_labels = labels_from_sample_submission(path)
    if (sub_id, sub_labels) != (ID_COL, LABELS):
        raise AssertionError(
            "레이블 불일치!\n"
            f"  config     : {ID_COL} | {LABELS}\n"
            f"  submission : {sub_id} | {sub_labels}"
        )


def make_submission(ids, probs, path: Path | None = None) -> pd.DataFrame:
    """항상 이 함수로 제출 파일을 만든다. 컬럼 순서를 강제한다.

    ids   : (N,)      샘플 ID
    probs : (N, C)    LABELS와 같은 순서의 예측 확률
    """
    import numpy as np

    probs = np.asarray(probs)
    if probs.ndim != 2 or probs.shape[1] != len(LABELS):
        raise ValueError(f"probs shape {probs.shape} != (N, {len(LABELS)})")

    sub = pd.DataFrame(probs, columns=list(LABELS))
    sub.insert(0, ID_COL, list(ids))
    sub = sub[[ID_COL, *LABELS]]  # 순서 고정

    if path is not None:
        sub.to_csv(path, index=False)
    return sub
