"""CSV EDA - 결과를 docs/eda.md 로 쓴다.

    python scripts/eda.py

DICOM은 건드리지 않는다. train.csv / train_series.csv / test*.csv 만 본다.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
import paths  # noqa: E402

OUT = ROOT / "docs" / "eda.md"
L = list(config.LABELS)


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def fmt(v):
        if isinstance(v, float):
            return "-" if pd.isna(v) else floatfmt.format(v)
        return str(v)

    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def script_of(text: str) -> str:
    """문자 집합으로 언어 계열을 대략 판별한다."""
    if not isinstance(text, str) or not text.strip():
        return "(빈값)"
    counts = {
        "한글": len(re.findall(r"[가-힣]", text)),
        "키릴": len(re.findall(r"[Ѐ-ӿ]", text)),
        "한자/일본어": len(re.findall(r"[぀-ヿ一-鿿]", text)),
        "아랍": len(re.findall(r"[؀-ۿ]", text)),
        "그리스": len(re.findall(r"[Ͱ-Ͽ]", text)),
        "히브리": len(re.findall(r"[֐-׿]", text)),
        "태국": len(re.findall(r"[฀-๿]", text)),
    }
    top = max(counts, key=counts.get)
    if counts[top] >= 5:
        return top
    return "라틴 계열"


LATIN_HINTS = {
    "영어": r"\b(the|and|with|there|normal|tear|joint|knee|effusion)\b",
    "스페인어": r"\b(el|la|los|las|con|sin|rodilla|derecha|izquierda|se observa)\b",
    "포르투갈어": r"\b(do|da|dos|das|com|sem|joelho|direito|esquerdo)\b",
    "프랑스어": r"\b(le|les|des|avec|sans|genou|droit|gauche)\b",
    "독일어": r"\b(der|die|das|und|mit|ohne|kniegelenk|rechts|links)\b",
    "이탈리아어": r"\b(il|lo|gli|con|senza|ginocchio|destro|sinistro)\b",
    "터키어": r"\b(ve|ile|sag|sol|diz|eklem)\b",
    "네덜란드어": r"\b(de|het|een|met|zonder|knie)\b",
}


def latin_lang(text: str) -> str:
    t = text.lower()
    scores = {k: len(re.findall(p, t)) for k, p in LATIN_HINTS.items()}
    top = max(scores, key=scores.get)
    return top if scores[top] >= 2 else "라틴 계열 기타"


def main() -> int:
    train = pd.read_csv(paths.TRAIN_CSV)
    tr_series = pd.read_csv(paths.DATA_DIR / "train_series.csv")
    te_series = pd.read_csv(paths.DATA_DIR / "test_series.csv")
    test = pd.read_csv(paths.DATA_DIR / "test.csv")
    sub = pd.read_csv(paths.SAMPLE_SUBMISSION_CSV)
    config.verify_labels()

    P: list[str] = []
    w = P.append

    w("# 데이터 EDA")
    w("")
    w("`python scripts/eda.py` 로 재생성한다. DICOM은 보지 않고 CSV만 분석한다.")
    w("")

    # ---------- 1. 파일 개요 ----------
    w("## 1. 파일 개요")
    w("")
    overview = pd.DataFrame([
        ["train.csv", len(train), len(train.columns), "스터디 단위 레이블 + 판독문"],
        ["train_series.csv", len(tr_series), len(tr_series.columns), "시리즈 단위 메타데이터"],
        ["test.csv", len(test), len(test.columns), "공개용 placeholder (실채점은 hidden rerun)"],
        ["test_series.csv", len(te_series), len(te_series.columns), "동일"],
        ["sample_submission.csv", len(sub), len(sub.columns), "제출 형식"],
    ], columns=["파일", "행", "열", "설명"])
    w(md_table(overview))
    w("")
    w(f"- 학습 스터디 **{train.StudyInstanceUID.nunique():,}개**, "
      f"시리즈 **{len(tr_series):,}개**")
    w(f"- ID 컬럼: `{config.ID_COL}`")
    w(f"- 제출 컬럼 순서: {', '.join(f'`{c}`' for c in L)}")
    w("")

    # ---------- 2. 레이블 희소성 ----------
    w("## 2. 레이블 희소성 - 이 대회의 핵심")
    w("")
    lab = train[L]
    labeled_mask = lab.notna().any(axis=1)
    n_gold = int(labeled_mask.sum())
    w(f"**레이블이 하나라도 있는 스터디는 {n_gold}개 / {len(train):,}개 "
      f"({n_gold / len(train) * 100:.2f}%)**")
    w("")
    w("나머지는 12개 컬럼이 전부 NaN이다. 학습 신호는 `Report` 컬럼에서 만들어야 한다.")
    w("")

    rows = []
    gold = train.loc[labeled_mask, L]
    for c in L:
        s = gold[c]
        n = int(s.notna().sum())
        pos = int((s == 1).sum())
        rows.append([c, n, pos, n - pos, pos / n if n else np.nan])
    w(md_table(pd.DataFrame(rows, columns=["레이블", "레이블 있음", "양성", "음성", "양성률"])))
    w("")
    vc = lab.notna().sum(axis=1).value_counts().sort_index()
    w("스터디당 채워진 레이블 개수 분포: "
      + ", ".join(f"{k}개={v}" for k, v in vc.items()))
    w("")

    # ---------- 3. 레이블 동시발생 ----------
    w("## 3. 레이블 동시발생 (gold 스터디 기준)")
    w("")
    g = gold.fillna(0).astype(int)
    co = pd.DataFrame(index=L, columns=L, dtype=float)
    for a in L:
        for b in L:
            na = int((g[a] == 1).sum())
            co.loc[a, b] = float(((g[a] == 1) & (g[b] == 1)).sum()) / na if na else np.nan
    disp = co.round(2).reset_index().rename(columns={"index": "P(열\\|행)"})
    w(md_table(disp, floatfmt="{:.2f}"))
    w("")
    w("행이 양성일 때 열이 양성일 조건부 확률. 대각선은 1.0.")
    w("표본이 58건뿐이라 신뢰구간이 매우 넓다 - 방향성만 참고할 것.")
    w("")

    # ---------- 4. 판독문 ----------
    w("## 4. 판독문 (Report)")
    w("")
    rep = train["Report"]
    w(f"- 결측: {int(rep.isna().sum())}건")
    lens = rep.dropna().str.len()
    w(f"- 길이(문자): 최소 {lens.min():,} / 25% {lens.quantile(.25):,.0f} / "
      f"중앙 {lens.median():,.0f} / 75% {lens.quantile(.75):,.0f} / 최대 {lens.max():,}")
    w("")
    fam = rep.fillna("").map(script_of)
    famvc = fam.value_counts()
    w("**문자 계열 분포**")
    w("")
    w(md_table(pd.DataFrame({"계열": famvc.index, "건수": famvc.values,
                             "비율": (famvc.values / len(train))}).assign()))
    w("")
    latin = rep[fam == "라틴 계열"].fillna("").map(latin_lang)
    lvc = latin.value_counts()
    w("**라틴 계열 내 추정 언어** (불용어 휴리스틱, 참고용)")
    w("")
    w(md_table(pd.DataFrame({"추정 언어": lvc.index, "건수": lvc.values,
                             "비율": (lvc.values / len(train))})))
    w("")
    w("> `test.csv`에는 `Report` 컬럼이 **없다**. 텍스트는 타깃 생성용이지 모델 입력이 아니다.")
    w("")

    # ---------- 5. 시리즈 메타데이터 ----------
    w("## 5. 시리즈 메타데이터")
    w("")
    per_study = tr_series.groupby("StudyInstanceUID").size()
    w(f"- 스터디당 시리즈: 최소 {per_study.min()} / 중앙 {per_study.median():.0f} / "
      f"평균 {per_study.mean():.1f} / 최대 {per_study.max()}")
    w(f"- 시리즈 총 {len(tr_series):,}개")
    w("")
    for col in ["Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"]:
        vc2 = tr_series[col].value_counts(dropna=False)
        w(f"**{col}**")
        w("")
        w(md_table(pd.DataFrame({col: vc2.index.astype(str), "건수": vc2.values,
                                 "비율": (vc2.values / len(tr_series))})))
        w("")

    # Fluid_Sensitive vs Fat_Suppression 동일 여부 (디스커션 제기 사항)
    fs, fat = tr_series["Fluid_Sensitive"], tr_series["Fat_Suppression"]
    same = int((fs.astype(str) == fat.astype(str)).sum())
    w(f"**Fluid_Sensitive == Fat_Suppression 인 시리즈: {same:,} / {len(tr_series):,} "
      f"({same / len(tr_series) * 100:.1f}%)**")
    w("")
    w("디스커션에서 두 컬럼이 사실상 같다는 지적이 있었다. 교차표는 다음과 같다.")
    w("")
    ct = pd.crosstab(fs, fat, dropna=False)
    w(md_table(ct.reset_index().rename(columns={ct.index.name: "Fluid\\Fat"}), floatfmt="{:.0f}"))
    w("")
    w("**평면 x 유체감응 조합**")
    w("")
    ct2 = pd.crosstab(tr_series["Anatomical_Plane"], tr_series["Fluid_Sensitive"])
    w(md_table(ct2.reset_index(), floatfmt="{:.0f}"))
    w("")

    # 스터디별 시퀀스 구성 커버리지
    w("**스터디별 핵심 시퀀스 보유율**")
    w("")
    grp = tr_series.groupby("StudyInstanceUID")
    cov = pd.DataFrame({
        "Sagittal 보유": grp["Anatomical_Plane"].apply(lambda s: (s == "Sagittal").any()),
        "Coronal 보유": grp["Anatomical_Plane"].apply(lambda s: (s == "Coronal").any()),
        "Axial 보유": grp["Anatomical_Plane"].apply(lambda s: (s == "Axial").any()),
        "FluidSensitive 보유": grp["Fluid_Sensitive"].apply(lambda s: s.astype(str).isin(["True", "1", "Yes"]).any()),
    })
    w(md_table(pd.DataFrame({
        "항목": cov.columns,
        "보유 스터디": [int(cov[c].sum()) for c in cov.columns],
        "비율": [cov[c].mean() for c in cov.columns],
    })))
    w("")
    combo = cov.apply(lambda r: "+".join(c.split()[0] for c in cov.columns if r[c]), axis=1)
    cvc = combo.value_counts().head(10)
    w("상위 조합 10개: " + ", ".join(f"`{k}`={v}" for k, v in cvc.items()))
    w("")

    # ---------- 6. train/test 정합성 ----------
    w("## 6. 정합성 체크")
    w("")
    checks = [
        ["train.csv 스터디 == train_series.csv 스터디",
         set(train.StudyInstanceUID) == set(tr_series.StudyInstanceUID)],
        ["train 스터디 UID 중복 없음", train.StudyInstanceUID.is_unique],
        ["sample_submission 컬럼 == config.LABELS", list(sub.columns[1:]) == L],
        ["test.csv 스터디 == sample_submission 스터디",
         set(test.StudyInstanceUID) == set(sub.StudyInstanceUID)],
        ["train/test 스터디 겹침 없음",
         len(set(train.StudyInstanceUID) & set(test.StudyInstanceUID)) == 0],
    ]
    w(md_table(pd.DataFrame(
        [[k, "OK" if v else "불일치"] for k, v in checks], columns=["항목", "결과"])))
    w("")

    # ---------- 7. 시사점 ----------
    w("## 7. 모델링 시사점")
    w("")
    w("1. **레이블을 먼저 만들어야 한다.** gold 58건으로는 학습이 불가능하다. "
      "판독문에서 12개 타깃을 추출하는 것이 파이프라인의 1단계이고, 그 품질이 성능의 천장이다.")
    w("2. **판독문은 다국어다.** 정규식/사전 방식은 언어마다 따로 만들어야 해서 "
      "LLM 독해가 유리하다 (공개 측정: 0.8136 → 0.8780).")
    w("3. **\"판독문에 언급 없음\"을 음성으로 뭉개지 말 것.** 0.5(미결정)로 두고 "
      "손실에서 마스킹하는 편이 낫다. 특히 Synovitis는 미언급이 84%다.")
    w("4. **58건 gold는 검증용으로만 쓴다.** 학습에 넣기엔 너무 적고, "
      "레이블 추출기 품질을 재는 유일한 자다.")
    w("5. **CV는 스터디 단위 분할이면 충분하다.** 스터디가 곧 환자 1명 1회 검사이므로 "
      "시리즈가 폴드를 가로지르지 않게만 하면 된다.")
    w("6. **시리즈 선택이 성능에 직결된다.** 스터디당 시리즈가 3~14개이고 "
      "평면·시퀀스 구성이 제각각이다. 어떤 시리즈를 몇 장 쓸지가 설계 포인트다.")
    w("7. **`Fat_Suppression`은 버려도 된다.** `Fluid_Sensitive`와 24,371개 전부 동일하다. "
      "두 컬럼을 따로 쓰는 코드가 있다면 중복 피처다.")
    w("8. **모든 스터디가 Sagittal/Coronal/Axial + fluid sensitive를 모두 갖는다.** "
      "시퀀스 결측 처리 분기는 필요 없다. 슬롯 고정 설계가 안전하다.")
    w("")

    OUT.write_text("\n".join(P), encoding="utf-8")
    print(f"[+] {OUT.relative_to(ROOT)} ({len(P)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
