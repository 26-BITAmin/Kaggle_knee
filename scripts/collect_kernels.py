"""공개 커널 상위 N개를 refs/kernels/ 로 내려받는다.

    python scripts/collect_kernels.py [N]     # 기본 40

득표순 상위 커널의 소스를 받아 둔다. 학습/추론 전략을 조사하기 위한 참고용이며,
refs/ 는 gitignore 대상이다 (타인 저작물을 우리 저장소에 재배포하지 않는다).
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMP = "rsna-knee-abnormality-detection"
OUT = ROOT / "refs" / "kernels"


def retry(fn, *args, tries: int = 6, **kwargs):
    delay = 5.0
    for attempt in range(1, tries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            if "429" not in str(e) or attempt == tries:
                raise
            print(f"    429 - {delay:.0f}초 대기 후 재시도 ({attempt}/{tries - 1})")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def main() -> int:
    from kaggle.api.kaggle_api_extended import KaggleApi

    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 40

    api = KaggleApi()
    api.authenticate()
    OUT.mkdir(parents=True, exist_ok=True)

    # 1) 득표순 목록
    kernels, page = [], 1
    while len(kernels) < top_n:
        batch = retry(api.kernels_list, competition=COMP, sort_by="voteCount",
                      page_size=100, page=page)
        if not batch:
            break
        kernels.extend(batch)
        page += 1
        time.sleep(1.0)
    kernels = kernels[:top_n]
    print(f"[+] 커널 {len(kernels)}개 대상\n")

    index = []
    for i, k in enumerate(kernels, 1):
        ref = k.ref
        votes = int(getattr(k, "total_votes", 0) or 0)
        title = getattr(k, "title", ref)
        dest = OUT / re.sub(r"[^\w.-]", "_", ref)

        if dest.exists() and any(dest.iterdir()):
            print(f"  [{i:>3}/{len(kernels)}] (이미 있음) {ref}")
        else:
            dest.mkdir(parents=True, exist_ok=True)
            try:
                retry(api.kernels_pull, ref, path=str(dest), metadata=True)
                print(f"  [{i:>3}/{len(kernels)}] {votes:>4}표  {ref}")
            except Exception as e:  # noqa: BLE001
                print(f"  [!] {ref} 실패: {e}", file=sys.stderr)
                continue
            time.sleep(1.5)

        index.append({
            "ref": ref, "title": title, "votes": votes,
            "author": getattr(k, "author", ""),
            "lastRunTime": str(getattr(k, "last_run_time", "")),
            "url": f"https://www.kaggle.com/code/{ref}",
            "dir": dest.name,
        })

    (OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[+] {OUT.relative_to(ROOT)} 에 {len(index)}개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
