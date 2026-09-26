"""노트북에서 코드 셀만 뽑는다 (출력 제외).

    python scripts/nbcode.py <ipynb> [--outline]
"""
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
outline = "--outline" in sys.argv
nb = json.load(open(p))
for i, c in enumerate(nb.get("cells", [])):
    src = "".join(c.get("source", []))
    if c.get("cell_type") == "markdown":
        if outline:
            first = next((l for l in src.splitlines() if l.strip()), "")
            print(f"--- [md {i}] {first[:100]}")
        continue
    if outline:
        lines = src.splitlines()
        head = next((l for l in lines if l.strip() and not l.strip().startswith("#")), "")
        print(f"=== [code {i}] {len(lines)}줄 | {head[:90]}")
    else:
        print(f"\n# ================= cell {i} =================")
        print(src)
