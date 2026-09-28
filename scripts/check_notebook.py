"""노트북 코드 셀에서 정의되지 않은 이름을 찾는다.

    python scripts/check_notebook.py kernels/train_fold0/train.ipynb

노트북을 스크립트로 수정하다 셀 내용을 잘라먹으면 이름이 사라진다.
실제로 EPOCHS 뒤를 통째로 덮어써서 CACHE / cfg / mcfg 정의가 날아갔고
커널이 NameError 로 죽었다. 푸시 전에 이걸 돌린다.

완벽한 검사는 아니다. 실행 순서대로 이름이 쌓이는지만 본다.
"""

from __future__ import annotations

import ast
import builtins
import json
import sys
from pathlib import Path


def check(path: Path) -> list[tuple[int, str]]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    defined: set[str] = set(dir(builtins))
    problems: list[tuple[int, str]] = []

    for i, c in enumerate(nb.get("cells", [])):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            problems.append((i, f"SyntaxError: {e}"))
            continue

        used = {n.id for n in ast.walk(tree)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}

        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                defined.add(n.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    defined.add((a.asname or a.name).split(".")[0])
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(n.name)
            elif isinstance(n, ast.arg):
                defined.add(n.arg)
            elif isinstance(n, ast.comprehension):
                for t in ast.walk(n.target):
                    if isinstance(t, ast.Name):
                        defined.add(t.id)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                defined.add(n.name)
            elif isinstance(n, ast.withitem) and n.optional_vars is not None:
                for t in ast.walk(n.optional_vars):
                    if isinstance(t, ast.Name):
                        defined.add(t.id)

        missing = sorted(u for u in used if u not in defined)
        if missing:
            problems.append((i, f"미정의 이름: {missing}"))

    return problems


def main() -> int:
    if len(sys.argv) < 2:
        print("사용법: python scripts/check_notebook.py <ipynb> [...]", file=sys.stderr)
        return 2
    bad = 0
    for arg in sys.argv[1:]:
        p = Path(arg)
        probs = check(p)
        if probs:
            bad += 1
            print(f"[!] {p}", file=sys.stderr)
            for i, m in probs:
                print(f"      cell {i}: {m}", file=sys.stderr)
        else:
            print(f"[+] {p} OK")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
