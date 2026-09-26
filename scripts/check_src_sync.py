"""로컬 src/ 와 Kaggle knee-src 데이터셋이 같은지 검사한다.

    python scripts/check_src_sync.py

다르면 종료 코드 1. push_kernel.sh 가 이걸 보고 푸시를 막는다.

커널은 Kaggle 에 올라간 knee-src 를 읽는다. 로컬에서 src/ 를 고치고
push_src.sh 를 안 돌리면 커널이 옛날 코드로 돈다.
에러가 안 나고 결과만 조용히 틀린다 (실제로 8칸 대신 6칸으로 2시간 빌드한 적 있음).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATASET = "knee-src"


def main() -> int:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    user = api.config_values.get("username")

    local = {p.name: p.stat().st_size for p in sorted(SRC.glob("*.py"))}

    try:
        files = api.dataset_list_files(f"{user}/{DATASET}").files
        remote = {f.name: int(f.total_bytes) for f in files}
    except Exception as e:  # noqa: BLE001
        print(f"[!] {user}/{DATASET} 조회 실패: {e}", file=sys.stderr)
        return 1

    diffs = []
    for name in sorted(set(local) | set(remote)):
        a, b = local.get(name), remote.get(name)
        if a != b:
            diffs.append((name, a, b))

    if not diffs:
        print(f"[+] src 동기화 OK ({len(local)}개 파일)")
        return 0

    print("[!] 로컬 src/ 와 Kaggle knee-src 가 다르다", file=sys.stderr)
    for name, a, b in diffs:
        print(f"      {name}: 로컬 {a} vs Kaggle {b}", file=sys.stderr)
    print("\n    먼저 실행할 것:  bash scripts/push_src.sh \"변경 메모\"", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
