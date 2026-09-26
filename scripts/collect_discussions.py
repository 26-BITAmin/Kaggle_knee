"""대회 디스커션 전체를 refs/discussions/ 에 수집한다.

    python scripts/collect_discussions.py

토픽 목록 전체 → 각 토픽의 본문 + 댓글(대댓글 포함)을 마크다운으로 저장.
refs/ 는 gitignore 대상 — 타인의 글이므로 우리 저장소에 재배포하지 않는다.
"""

from __future__ import annotations

import html
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMP = "rsna-knee-abnormality-detection"
OUT = ROOT / "refs" / "discussions"


def slugify(s: str, limit: int = 60) -> str:
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE).strip()
    s = re.sub(r"[\s_]+", "-", s)
    return s[:limit].strip("-") or "untitled"


def to_text(raw: str | None) -> str:
    """디스커션 본문은 HTML이다. 코드블록을 살리면서 평문으로 바꾼다."""
    if not raw:
        return ""
    s = raw
    s = re.sub(r"<pre[^>]*>\s*<code[^>]*>(.*?)</code>\s*</pre>", r"\n```\n\1\n```\n", s, flags=re.S)
    s = re.sub(r"<pre[^>]*>(.*?)</pre>", r"\n```\n\1\n```\n", s, flags=re.S)
    s = re.sub(r"<code[^>]*>(.*?)</code>", r"`\1`", s, flags=re.S)
    s = re.sub(r"<a[^>]*href=\"([^\"]+)\"[^>]*>(.*?)</a>", r"[\2](\1)", s, flags=re.S)
    s = re.sub(r"<br\s*/?>", "\n", s)
    s = re.sub(r"</p>", "\n\n", s)
    s = re.sub(r"<li[^>]*>", "\n- ", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def render(msg, depth: int = 0) -> list[str]:
    d = msg.to_dict() if hasattr(msg, "to_dict") else dict(msg)
    pad = "  " * depth
    out = [
        f"{pad}### {'↳ ' if depth else ''}{d.get('authorName', '?')} "
        f"({d.get('votes', 0)}표, {str(d.get('postDate', ''))[:10]})",
        "",
        to_text(d.get("content")),
        "",
    ]
    for r in getattr(msg, "replies", None) or []:
        out += render(r, depth + 1)
    return out


def retry(fn, *args, tries: int = 9, **kwargs):
    """429(레이트 리밋)는 지수 백오프로 재시도한다."""
    delay = 10.0
    for attempt in range(1, tries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            if "429" not in str(e) or attempt == tries:
                raise
            print(f"    429 - {delay:.0f}초 대기 후 재시도 ({attempt}/{tries - 1})")
            time.sleep(delay)
            delay = min(delay * 2, 180.0)
    raise RuntimeError("unreachable")


def list_all_topics(api) -> list[dict]:
    topics, page = [], 1
    while True:
        r = retry(api.competition_list_topics, COMP, page=page)
        batch = [t.to_dict() for t in (r.topics or [])]
        if not batch:
            break
        topics.extend(batch)
        print(f"  목록 {page}페이지 → 누적 {len(topics)}/{r.total_count}")
        if len(topics) >= int(r.total_count or 0):
            break
        page += 1
        time.sleep(1.0)
    return topics


def fetch_messages(api, topic_id: int) -> tuple[dict, list]:
    topic, msgs, token = retry(api.forums_topic_show, topic_id, page_size=200)
    msgs = list(msgs or [])
    while token:
        _, more, token = retry(api.forums_topic_show, topic_id, page_size=200, page_token=token)
        if not more:
            break
        msgs.extend(more)
        time.sleep(0.2)
    return topic.to_dict(), msgs


def main() -> int:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    OUT.mkdir(parents=True, exist_ok=True)

    print("[*] 토픽 목록 수집")
    topics = list_all_topics(api)
    print(f"[+] 토픽 {len(topics)}개\n")

    done = {m.group(1) for f in OUT.glob("*.md")
            if (m := re.match(r"\d+_(\d+)_", f.name))}
    if done:
        print(f"[*] 이미 받은 {len(done)}개는 건너뛴다")

    index = []
    for i, t in enumerate(topics, 1):
        tid = int(t["id"])
        title = t.get("title", f"topic-{tid}")
        votes = int(t.get("votes") or 0)
        if str(tid) in done:
            continue
        try:
            detail, msgs = fetch_messages(api, tid)
        except Exception as e:  # noqa: BLE001
            print(f"  [!] {tid} 실패: {e}", file=sys.stderr)
            continue

        lines = [
            f"# {title}",
            "",
            f"- topic_id: `{tid}`",
            f"- author: {detail.get('authorName', '')}",
            f"- votes: {votes} / comments: {t.get('commentCount', 0)}",
            f"- posted: {str(t.get('postDate', ''))[:10]}",
            f"- url: https://www.kaggle.com{t.get('topicUrl', '')}",
            "",
            "---",
            "",
            "## 본문",
            "",
            to_text(detail.get("content")),
            "",
            "---",
            "",
            "## 댓글",
            "",
        ]
        for m in msgs:
            lines += render(m)

        fp = OUT / f"{votes:04d}_{tid}_{slugify(title)}.md"
        fp.write_text("\n".join(lines), encoding="utf-8")
        index.append({
            "id": tid, "title": title, "votes": votes,
            "author": detail.get("authorName", ""),
            "posted": str(t.get("postDate", ""))[:10],
            "comments": int(t.get("commentCount") or 0),
            "file": fp.name,
        })
        print(f"  [{i:>3}/{len(topics)}] {votes:>4}표 {len(msgs):>3}댓글  {title[:58]}")
        time.sleep(2.5)

    # 이번에 건너뛴 것까지 포함해 전체 목록으로 index 재생성
    by_id = {str(e["id"]): e for e in index}
    for f in sorted(OUT.glob("*.md")):
        m = re.match(r"(\d+)_(\d+)_", f.name)
        if m and m.group(2) not in by_id:
            by_id[m.group(2)] = {"id": int(m.group(2)), "title": f.stem.split("_", 2)[-1],
                                 "votes": int(m.group(1)), "file": f.name}
    index = list(by_id.values())
    index.sort(key=lambda x: -x["votes"])
    (OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[+] {OUT.relative_to(ROOT)} 에 {len(index)}개 저장")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
