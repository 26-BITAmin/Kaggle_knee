"""공개 판독문 레이블 테이블을 gold 58건에 대고 평가한다.

    python scripts/bench_labels.py

각 테이블의 레이블별 / macro AUC 를 재고, 쌍별 rank 블렌드도 같이 본다.
결과는 docs/labels.md 로 쓴다.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
import paths  # noqa: E402

L = list(config.LABELS)
LAB_DIR = ROOT / "refs" / "labels"
OUT = ROOT / "docs" / "labels.md"

TABLES = {
    "stevenleehans v4_blend": "stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv",
    "stevenleehans v2": "stevenleehans_rsna-knee-llm-report-labels/llm_labels_v2.csv",
    "stevenleehans full": "stevenleehans_rsna-knee-llm-report-labels/llm_labels_full.csv",
    "pilkwang v2": "pilkwang_rsna-knee-llm-labels/report_labels_v2.csv",
    "lixin73 gpt56sol": "lixin73_rsna-knee-llm-report-labels-sol56/report_labels_gpt56sol.csv",
    "lixin73 labels": "lixin73_rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv",
    "barun2104 folds": "barun2104_rsna-knee-stratified-folds-and-llm-soft-labels/train_folds.csv",
    "barun2104 pseudo": "barun2104_rsna-knee-stratified-folds-and-llm-soft-labels/train_folds_with_pseudo.csv",
}


def load_gold() -> pd.DataFrame:
    tr = pd.read_csv(paths.TRAIN_CSV)
    g = tr[tr[L].notna().any(axis=1)].set_index("StudyInstanceUID")[L]
    return g.astype(int)


def load_table(rel: str) -> pd.DataFrame | None:
    p = LAB_DIR / rel
    if not p.exists():
        return None
    df = pd.read_csv(p)
    src = L
    if "pseudo" in p.name:
        pc = [f"pseudo_{c}" for c in L]
        if all(c in df.columns for c in pc):
            src = pc   # 원본 컬럼은 버리고 pseudo 쪽만 쓴다
    if not all(c in df.columns for c in src):
        return None
    df = df.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")[src]
    df.columns = L
    return df.apply(pd.to_numeric, errors="coerce")


def per_label_auc(gold: pd.DataFrame, tab: pd.DataFrame) -> pd.Series:
    common = gold.index.intersection(tab.index)
    out = {}
    for c in L:
        y = gold.loc[common, c].values
        s = tab.loc[common, c].values
        m = ~np.isnan(s)
        if len(np.unique(y[m])) < 2 or m.sum() < 5:
            out[c] = np.nan
        else:
            out[c] = roc_auc_score(y[m], s[m])
    return pd.Series(out)


def leaks_gold(gold: pd.DataFrame, tab: pd.DataFrame) -> bool:
    """테이블이 gold 정답을 그대로 담고 있는지 검사한다.

    일부 공개 테이블은 is_manual 스터디에 실제 레이블을 복사해 둔다.
    그러면 gold 대비 AUC 가 1.0 이 나오는데 판독문 추출 품질과는 무관하다.
    """
    common = gold.index.intersection(tab.index)
    if len(common) < 10:
        return False
    a = gold.loc[common, L].values.astype(float)
    b = tab.loc[common, L].values.astype(float)
    m = ~np.isnan(b)
    return bool(m.sum() > 0 and np.allclose(a[m], b[m]))


def rank_norm(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda s: pd.Series(rankdata(s.fillna(s.median())) / len(s), index=s.index))


def main() -> int:
    gold = load_gold()
    tables = {}
    for name, rel in TABLES.items():
        t = load_table(rel)
        if t is None:
            print(f"[!] 건너뜀: {name}", file=sys.stderr)
            continue
        tables[name] = t

    leaked = {n for n, t in tables.items() if leaks_gold(gold, t)}
    for n in leaked:
        print(f"[!] {n}: gold 정답이 그대로 들어 있어 평가에서 제외한다", file=sys.stderr)
        tables.pop(n)

    rows = {}
    coverage = {}
    for name, t in tables.items():
        rows[name] = per_label_auc(gold, t)
        common = gold.index.intersection(t.index)
        coverage[name] = (len(t), len(common))

    res = pd.DataFrame(rows).T
    res["macro"] = res[L].mean(axis=1)
    res = res.sort_values("macro", ascending=False)

    # 쌍별 rank 블렌드
    ranked = {n: rank_norm(t.reindex(gold.index)) for n, t in tables.items()}
    blends = {}
    for a, b in combinations(tables, 2):
        mix = (ranked[a] + ranked[b]) / 2
        blends[f"{a}  +  {b}"] = per_label_auc(gold, mix).mean()
    blend_s = pd.Series(blends).sort_values(ascending=False)

    P = []
    w = P.append
    w("# 판독문 레이블 테이블 벤치마크")
    w("")
    w("`python scripts/bench_labels.py` 로 재생성한다")
    w("")
    w(f"gold {len(gold)}건에 대고 각 공개 테이블의 AUC 를 쟀다")
    w("gold 는 `train.csv` 에서 레이블이 채워진 스터디다")
    w("")
    w("> 58건뿐이라 신뢰구간이 매우 넓다")
    w("> 레이블당 양성이 9~35개이므로 0.02 수준 차이는 해석하지 않는다")
    w("")
    if leaked:
        w("## 평가에서 제외한 테이블")
        w("")
        w("아래 테이블은 gold 58건에 **정답을 그대로 복사**해 두었다")
        w("AUC 가 1.0 으로 나오지만 판독문 추출 품질과는 무관하다")
        w("")
        for n in sorted(leaked):
            w(f"- `{n}`")
        w("")
        w("학습에는 쓸 수 있다 (나머지 4,349건은 판독문 유래)")
        w("다만 **gold 로 검증할 수 없으므로 품질을 알 수 없다**")
        w("`barun2104` 의 `fold` 컬럼은 별도로 유용하다")
        w("")
    w("## 테이블별 macro AUC")
    w("")
    w("| 테이블 | 행 | gold 매칭 | macro AUC |")
    w("|---|---|---|---|")
    for n in res.index:
        tot, com = coverage[n]
        w(f"| {n} | {tot:,} | {com} | **{res.loc[n, 'macro']:.4f}** |")
    w("")
    w("## 레이블별 AUC")
    w("")
    w("| 테이블 | " + " | ".join(L) + " |")
    w("|" + "---|" * (len(L) + 1))
    for n in res.index:
        w(f"| {n} | " + " | ".join(
            "-" if pd.isna(res.loc[n, c]) else f"{res.loc[n, c]:.3f}" for c in L) + " |")
    w("")
    w("## 쌍별 rank 블렌드 (상위 10)")
    w("")
    w("독립적인 두 테이블을 평균하면 각각보다 낫다는 보고가 있어 확인했다 (디스커션 737454)")
    w("")
    w("| 조합 | macro AUC |")
    w("|---|---|")
    for k, v in blend_s.head(10).items():
        w(f"| {k} | **{v:.4f}** |")
    w("")
    best_single = res.index[0]
    best_blend = blend_s.index[0]
    w("## 결론")
    w("")
    w(f"- 단일 최고: **{best_single}** ({res.loc[best_single, 'macro']:.4f})")
    w(f"- 블렌드 최고: **{best_blend}** ({blend_s.iloc[0]:.4f})")
    w(f"- 차이: {blend_s.iloc[0] - res.loc[best_single, 'macro']:+.4f}")
    w("")
    w("### 출처")
    w("")
    w("| 데이터셋 | 라이선스 |")
    w("|---|---|")
    w("| `stevenleehans/rsna-knee-llm-report-labels` | CC0-1.0 |")
    w("| `pilkwang/rsna-knee-llm-labels` | CC0-1.0 |")
    w("| `lixin73/rsna-knee-llm-report-labels-sol56` | CC0-1.0 |")
    w("| `barun2104/rsna-knee-stratified-folds-and-llm-soft-labels` | other |")

    OUT.write_text("\n".join(P), encoding="utf-8")
    print("\n".join(P[:40]))
    print(f"\n[+] {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
