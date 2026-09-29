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


def _defs(tree: ast.AST) -> set[str]:
    """트리 안에서 정의되는 이름"""
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, ast.comprehension):
            for t in ast.walk(n.target):
                if isinstance(t, ast.Name):
                    out.add(t.id)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.withitem) and n.optional_vars is not None:
            for t in ast.walk(n.optional_vars):
                if isinstance(t, ast.Name):
                    out.add(t.id)
    return out


def _loads(tree: ast.AST) -> tuple[set[str], set[str]]:
    """(셀 실행 시 바로 읽는 이름, 함수 본문에서 호출 시점에 읽는 이름)"""
    now: set[str] = set()
    later: set[str] = set()

    def visit(n: ast.AST, deferred: bool) -> None:
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            (later if deferred else now).add(n.id)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            # 데코레이터와 기본값은 정의 시점에 평가된다
            for d in getattr(n, "decorator_list", []):
                visit(d, deferred)
            for d in n.args.defaults + [d for d in n.args.kw_defaults if d]:
                visit(d, deferred)
            for b in (n.body if isinstance(n.body, list) else [n.body]):
                visit(b, True)
            return
        for c in ast.iter_child_nodes(n):
            visit(c, deferred)

    visit(tree, False)
    return now, later


def check(path: Path) -> list[tuple[int, str]]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    # get_ipython 은 IPython 커널이 주입한다
    defined: set[str] = set(dir(builtins)) | {"get_ipython"}
    # 가져온 외부 노트북의 죽은 분기처럼 알고 넘기는 이름. 노트북 메타데이터에 명시한다
    defined |= set(nb.get("metadata", {}).get("check_notebook", {}).get("allow_undefined", []))
    problems: list[tuple[int, str]] = []

    trees: list[tuple[int, ast.AST]] = []
    for i, c in enumerate(nb.get("cells", [])):
        if c.get("cell_type") != "code":
            continue
        try:
            trees.append((i, ast.parse("".join(c.get("source", [])))))
        except SyntaxError as e:
            problems.append((i, f"SyntaxError: {e}"))

    # 함수 본문은 호출될 때 이름을 찾으므로 뒤 셀에서 정의돼도 된다
    anywhere = set(defined).union(*(_defs(t) for _, t in trees))

    for i, tree in trees:
        now, later = _loads(tree)
        defined |= _defs(tree)
        missing = sorted({u for u in now if u not in defined}
                         | {u for u in later if u not in anywhere})
        if missing:
            problems.append((i, f"미정의 이름: {missing}"))

    return sorted(problems)


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
