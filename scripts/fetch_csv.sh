#!/usr/bin/env bash
# 대회 CSV만 내려받는다. (-f 필수 — 없으면 전체 데이터셋을 받는다)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
C=rsna-knee-abnormality-detection

for f in train.csv sample_submission.csv; do
  echo "[*] $f"
  "$PY" -m kaggle competitions download -c "$C" -f "$f" -p "$ROOT/data" || echo "[!] $f 건너뜀"
done

shopt -s nullglob
for z in "$ROOT"/data/*.zip; do
  unzip -o "$z" -d "$ROOT/data" && rm -f "$z"
done

ls -la "$ROOT/data"
