#!/usr/bin/env bash
# 커널을 Kaggle에 푸시한다.  사용법: bash scripts/push_kernel.sh [train_fold0]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KERNEL="${1:-train_fold0}"
DIR="$ROOT/kernels/$KERNEL"
META="$DIR/kernel-metadata.json"
PY="${PYTHON:-$ROOT/.venv/bin/python}"

[ -f "$META" ] || { echo "[!] $META 없음"; exit 1; }

# 안전장치: 인터넷은 항상 꺼져 있어야 한다 (코드 대회 규칙)
if ! grep -q '"enable_internet": *false' "$META"; then
  echo "[!] enable_internet이 false가 아니다. 제출 불가 커널이 된다. 중단."
  exit 1
fi

# 안전장치: 커널은 Kaggle 의 knee-src 를 읽는다.
# 로컬 src/ 를 고치고 push_src.sh 를 안 돌리면 옛날 코드로 조용히 돈다.
if grep -q "knee-src" "$META"; then
  "$PY" "$ROOT/scripts/check_src_sync.py" || {
    echo "[!] src 가 동기화되지 않았다. 중단."
    exit 1
  }
fi

# 안전장치: 노트북에 미정의 이름이 없는지 본다
# 스크립트로 셀을 수정하다 내용을 잘라먹어 NameError 로 죽은 적이 있다
NB="$DIR/$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['code_file'])" "$META")"
if [ -f "$NB" ]; then
  "$PY" "$ROOT/scripts/check_notebook.py" "$NB" || {
    echo "[!] 노트북 검사 실패. 중단."
    exit 1
  }
fi

USER="$("$PY" "$ROOT/scripts/_kaggle_user.py")"

# 메타데이터의 placeholder username을 실제 계정으로 치환
"$PY" - "$META" "$USER" <<'PYEOF'
import json, sys
meta_path, user = sys.argv[1], sys.argv[2]
meta = json.load(open(meta_path))
meta["id"] = f"{user}/{meta['id'].split('/', 1)[1]}"

PLACEHOLDER = "kaggle-username"


def own(ref):
    """placeholder 소유자만 실제 계정으로 바꾼다. 남의 공개 데이터셋은 건드리지 않는다."""
    owner, _, name = ref.partition("/")
    return f"{user}/{name}" if owner == PLACEHOLDER else ref


for key in ("dataset_sources", "kernel_sources", "model_sources"):
    meta[key] = [own(d) for d in meta.get(key, [])]
    if meta[key]:
        print(f"[+] {key} = {meta[key]}")
json.dump(meta, open(meta_path, "w"), indent=2, ensure_ascii=False)
open(meta_path, "a").write("\n")
print(f"[+] id = {meta['id']}")
PYEOF

echo "[*] pushing $DIR ..."
"$PY" -m kaggle kernels push -p "$DIR"
SLUG="$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['id'].split('/')[1])" "$META")"
echo "[+] https://www.kaggle.com/code/$USER/$SLUG"
