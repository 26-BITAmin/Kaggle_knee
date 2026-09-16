#!/usr/bin/env bash
# src/ 를 Kaggle 데이터셋(knee-src)으로 올린다. 커널에서 import 하기 위함.
# 사용법: bash scripts/push_src.sh ["변경 메모"]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
MSG="${1:-update src}"
BUILD="$ROOT/build/knee-src"

USER="$("$PY" "$ROOT/scripts/_kaggle_user.py")"

rm -rf "$BUILD" && mkdir -p "$BUILD"
cp "$ROOT"/src/*.py "$BUILD"/

cat > "$BUILD/dataset-metadata.json" <<JSON
{
  "title": "knee-src",
  "id": "$USER/knee-src",
  "licenses": [{"name": "CC0-1.0"}]
}
JSON

if "$PY" -m kaggle datasets status "$USER/knee-src" >/dev/null 2>&1; then
  "$PY" -m kaggle datasets version -p "$BUILD" -m "$MSG" --dir-mode zip
else
  "$PY" -m kaggle datasets create -p "$BUILD" --dir-mode zip
fi
echo "[+] https://www.kaggle.com/datasets/$USER/knee-src"
