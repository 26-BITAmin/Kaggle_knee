"""sample_submission.csv를 읽어 src/config.py의 ID_COL / LABELS를 채운다.

    python scripts/sync_labels.py

sample_submission.csv가 없으면 먼저 개별 파일만 내려받는다 (-f 필수).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from paths import SAMPLE_SUBMISSION_CSV  # noqa: E402

CONFIG_PY = ROOT / "src" / "config.py"


def main() -> int:
    if not SAMPLE_SUBMISSION_CSV.exists():
        print(
            f"[!] {SAMPLE_SUBMISSION_CSV} 없음.\n"
            "    kaggle competitions download -c rsna-knee-abnormality-detection "
            "-f sample_submission.csv -p data/\n"
            "    (-f 없이 실행 금지: 전체 데이터셋을 받아버린다)",
            file=sys.stderr,
        )
        return 1

    cols = list(pd.read_csv(SAMPLE_SUBMISSION_CSV, nrows=1).columns)
    id_col, labels = cols[0], cols[1:]

    src = CONFIG_PY.read_text()
    src = re.sub(r'^ID_COL: str = .*$', f'ID_COL: str = "{id_col}"', src, count=1, flags=re.M)
    labels_block = "LABELS: tuple[str, ...] = (\n" + "".join(
        f'    "{c}",\n' for c in labels
    ) + ")"
    src = re.sub(r"^LABELS: tuple\[str, \.\.\.\] = \(\)$", labels_block, src, count=1, flags=re.M)
    CONFIG_PY.write_text(src)

    print(f"[+] ID_COL = {id_col}")
    print(f"[+] LABELS ({len(labels)}개):")
    for i, c in enumerate(labels):
        print(f"    {i:2d}. {c}")
    print(f"\n[+] {CONFIG_PY.relative_to(ROOT)} 갱신 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
