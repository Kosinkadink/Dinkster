from __future__ import annotations

import ast
import subprocess
from pathlib import Path

from dinkster_inference import builtin_families

EXTERNAL_PROOF_FAMILY_IDS = frozenset({"fixture.toy-image", "test.toy-image"})
NEW_FAMILY_REGISTRATION_PATHS = frozenset(
    {
        "docs/new-model-family.md",
        "packages/dinkster-inference-torch/tests/test_new_family_checklist.py",
        "tests/family_gate_scanner.py",
        "tests/test_family_registration_gates.py",
        "tests/test_native_arm.py",
    }
)
NEW_FAMILY_PROOF_PATH = "packages/dinkster-inference-torch/tests/test_new_family_checklist.py"


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


def new_family_proof_commit_paths(root: Path) -> frozenset[str]:
    commits = subprocess.run(
        ("git", "log", "--format=%H", "--diff-filter=A", "--", NEW_FAMILY_PROOF_PATH),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return frozenset(
        path
        for commit in commits
        for path in subprocess.run(
            ("git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )


def unexpected_new_family_paths(changed_paths: frozenset[str]) -> tuple[str, ...]:
    if NEW_FAMILY_PROOF_PATH not in changed_paths:
        return ()
    return tuple(sorted(changed_paths - NEW_FAMILY_REGISTRATION_PATHS))
