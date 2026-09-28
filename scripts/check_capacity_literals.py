#!/usr/bin/env python3
"""Reject byte-capacity expressions outside the governed limits module."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LIMITS_MODULE = REPO_ROOT / "packages/dinkster-values/src/dinkster_values/limits.py"
SUPERVISOR_LIMITS_MODULE = (
    REPO_ROOT / "packages/dinkster-supervisor/src/dinkster_supervisor/limits.py"
)
LIMITS_MODULES = frozenset((LIMITS_MODULE.resolve(), SUPERVISOR_LIMITS_MODULE.resolve()))


def _literal_int(node: ast.AST, value: int) -> bool:
    return isinstance(node, ast.Constant) and type(node.value) is int and node.value == value


def _factors(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return [*_factors(node.left), *_factors(node.right)]
    return [node]


def is_byte_capacity_expression(node: ast.AST) -> bool:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.LShift):
        return _literal_int(node.right, 20) or _literal_int(node.right, 30)
    factors = _factors(node)
    if sum(_literal_int(factor, 1024) for factor in factors) >= 2:
        return True
    return any(
        isinstance(factor, ast.BinOp)
        and isinstance(factor.op, ast.Pow)
        and _literal_int(factor.left, 1024)
        and (_literal_int(factor.right, 2) or _literal_int(factor.right, 3))
        for factor in factors
    )


def capacity_literals(path: Path) -> list[tuple[int, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return sorted(
        {
            (node.lineno, node.col_offset + 1)
            for node in ast.walk(tree)
            if is_byte_capacity_expression(node)
        }
    )


def governed_sources(root: Path = REPO_ROOT) -> list[Path]:
    sources = [*root.glob("src/**/*.py"), *root.glob("packages/*/src/**/*.py")]
    return sorted(
        path
        for path in sources
        if "tests" not in path.parts and path.resolve() not in LIMITS_MODULES
    )


def main() -> int:
    violations = [
        f"{path.relative_to(REPO_ROOT)}:{line}:{column}"
        for path in governed_sources()
        for line, column in capacity_literals(path)
    ]
    if violations:
        print("Byte-capacity literals must be declared in dinkster_values.limits:", file=sys.stderr)
        print("\n".join(violations), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
