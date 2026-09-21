from __future__ import annotations

import ast
from pathlib import Path

from dinkster_inference import builtin_families

EXTERNAL_PROOF_FAMILY_IDS = frozenset({"test.toy-image"})


def registered_family_ids() -> frozenset[str]:
    return frozenset(family.id for family in builtin_families()) | EXTERNAL_PROOF_FAMILY_IDS


def family_literal_gates(path: Path, root: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    findings: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Compare, ast.Dict, ast.IfExp, ast.Match, ast.Set)):
            continue
        literals = {
            child.value
            for child in ast.walk(node)
            if isinstance(child, ast.Constant)
            and isinstance(child.value, str)
            and child.value in registered_family_ids()
        }
        findings.update(
            f"{path.relative_to(root)}:{node.lineno}: {literal}" for literal in literals
        )
    return tuple(sorted(findings))
