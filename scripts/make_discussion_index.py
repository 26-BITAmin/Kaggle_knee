"""refs/discussions/index.json → 마크다운 인덱스 표.

    python scripts/make_discussion_index.py

docs/discussions.md 의 `<!-- INDEX -->` 이후를 통째로 갈아끼운다.
원문은 refs/ 에 있고 gitignore 대상이므로, 저장소에는 목록과 링크만 남긴다.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMP = "rsna-knee-abnormality-detection"
SRC = ROOT / "refs" / "discussions" / "index.json"
DOC = ROOT / "docs" / "discussions.md"
MARK = "<!-- INDEX -->"


def main() -> int:
    idx = json.loads(SRC.read_text(encoding="utf-8"))
    idx.sort(key=lambda e: (-e["votes"], -e.get("comments", 0)))

    lines = [
        MARK,
        "",
        f"## 전체 목록 ({len(idx)}개)",
        "",
        "득표순. `python scripts/make_discussion_index.py` 로 갱신한다.",
        "",
        "| 표 | 댓글 | 제목 |",
        "|---|---|---|",
    ]
    for e in idx:
        title = str(e["title"]).replace("|", "\\|").strip()
        url = f"https://www.kaggle.com/competitions/{COMP}/discussion/{e['id']}"
        lines.append(f"| {e['votes']} | {e.get('comments', 0)} | [{title}]({url}) |")

    doc = DOC.read_text(encoding="utf-8")
    head = doc.split(MARK, 1)[0].rstrip()
    DOC.write_text(head + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"[+] {DOC.relative_to(ROOT)} 인덱스 {len(idx)}개 갱신")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
