"""학습 타깃 구성 - 판독문 유래 레이블과 손실 가중.

gold 레이블은 4,407개 중 58개뿐이라 학습에 못 쓴다 (검증 전용).
공개 LLM 테이블에서 타깃을 만들고, 판독문이 언급하지 않은 셀은 가중을 낮춘다.

측정 근거 (docs/labels.md, 디스커션 733932 / 737454)

- stevenleehans v4_blend 가 gold 58건 대비 macro AUC 0.8927 로 가장 높다
- v4_blend 는 미언급을 **0.25** 로 표시한다 (llm_labels_full 의 0.5 와 99.9% 일치)
- 미언급 셀을 빼도 AUC 가 크게 오르지 않는다 → 완전 무가치는 아니다
  그래서 마스킹(0)이 아니라 저가중으로 둔다
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import config
import paths

# v4_blend 가 "판독문이 언급하지 않음" 을 표시하는 값
UNADDRESSED = 0.25
TABLE = "llm_labels_v4_blend.csv"


@dataclass
class LabelConfig:
    table: str = TABLE
    # 미언급 셀의 손실 가중. 0 이면 마스킹, 1 이면 동등 취급
    w_unaddressed: float = 0.3
    # 타깃을 이진화하지 않고 soft 로 쓴다 (판독문은 등급이 있는 신호다)
    soft_targets: bool = True
    # soft 타깃을 이 범위로 눌러 과신을 막는다
    clip: tuple[float, float] = (0.02, 0.98)


def find_label_table(name: str = TABLE) -> Path:
    """레이블 CSV 위치. 로컬(refs/) 과 Kaggle(input/) 둘 다 지원한다."""
    local = paths.ROOT / "refs" / "labels" / "stevenleehans_rsna-knee-llm-report-labels" / name
    if local.exists():
        return local
    if paths.IS_KAGGLE:
        d = paths.find_input("rsna-knee-llm-report-labels")
        p = d / name
        if p.exists():
            return p
        hits = list(d.glob(f"**/{name}"))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"{name} 을 못 찾음. Kaggle 커널이면 dataset_sources 에 "
        "stevenleehans/rsna-knee-llm-report-labels 를 넣을 것"
    )


def load_targets(cfg: LabelConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(타깃, 가중) 을 돌려준다. 둘 다 index=StudyInstanceUID, columns=LABELS."""
    cfg = cfg or LabelConfig()
    L = list(config.LABELS)

    df = pd.read_csv(find_label_table(cfg.table))
    if not all(c in df.columns for c in L):
        raise ValueError(f"레이블 컬럼 누락: {set(L) - set(df.columns)}")
    df = df.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")[L]
    df = df.apply(pd.to_numeric, errors="coerce")

    unaddressed = np.isclose(df.values, UNADDRESSED)
    weight = np.where(unaddressed, cfg.w_unaddressed, 1.0)

    y = df.values.astype(np.float32)
    if cfg.soft_targets:
        y = np.clip(y, *cfg.clip)
    else:
        y = (y >= 0.5).astype(np.float32)

    return (pd.DataFrame(y, index=df.index, columns=L),
            pd.DataFrame(weight.astype(np.float32), index=df.index, columns=L))


def load_gold() -> pd.DataFrame:
    """전문가 레이블 58건. **학습에 쓰지 않는다.**

    레이블 추출기와 모델을 재는 development panel 이다.
    개발 과정에서 이미 참조했으므로 독립 테스트셋으로 취급하면 안 된다.
    """
    L = list(config.LABELS)
    tr = pd.read_csv(paths.TRAIN_CSV)
    g = tr[tr[L].notna().any(axis=1)].set_index("StudyInstanceUID")[L]
    return g.astype(int)


def make_folds(studies: pd.Index, y: pd.DataFrame, n_folds: int = 5,
               seed: int = 42, gold: pd.Index | None = None) -> pd.Series:
    """스터디 단위 다중 레이블 층화 분할.

    스터디가 곧 1인 1회 검사라 그룹 분할은 필요 없다.
    다만 희귀 레이블이 한쪽 fold 로 쏠리면 macro AUC 가 크게 흔들리므로
    양성 개수가 적은 레이블부터 순서대로 층화한다 (iterative stratification 의 단순형).
    """
    rng = np.random.default_rng(seed)
    L = list(config.LABELS)
    b = (y.loc[studies, L].values >= 0.5)

    order = np.argsort(b.sum(axis=0))          # 희귀한 레이블부터
    fold = np.full(len(studies), -1, dtype=int)
    counts = np.zeros((n_folds, len(L)), dtype=int)

    # gold 스터디를 먼저 배치해 fold 마다 고르게 퍼지게 한다
    # 58건이 한쪽으로 쏠리면 development panel 측정이 아예 안 된다
    order_idx = list(rng.permutation(len(studies)))
    if gold is not None:
        gset = set(gold)
        g = [i for i in order_idx if studies[i] in gset]
        rest = [i for i in order_idx if studies[i] not in gset]
        for j, i in enumerate(g):            # 라운드로빈
            fold[i] = j % n_folds
            counts[j % n_folds] += b[i]
        order_idx = rest

    idx = order_idx
    for i in idx:
        # 이 스터디가 가진 레이블 중 가장 희귀한 것을 기준으로 fold 를 고른다
        pos = [k for k in order if b[i, k]]
        key = pos[0] if pos else order[0]
        cand = np.flatnonzero(counts[:, key] == counts[:, key].min())
        f = int(cand[np.argmin(counts[cand].sum(axis=1))])
        fold[i] = f
        counts[f] += b[i]

    return pd.Series(fold, index=studies, name="fold")
