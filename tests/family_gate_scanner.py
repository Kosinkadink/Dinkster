from __future__ import annotations

import ast
import subprocess
from pathlib import Path

from dinkster_inference import builtin_families

EXTERNAL_PROOF_FAMILY_IDS = frozenset({"fixture.toy-image", "test.toy-image"})
NEW_FAMILY_REGISTRATION_PATHS = frozenset(
    {
        ".github/workflows/ci.yml",
        ".github/workflows/full-validation.yml",
        "docs/extension-design.md",
        "docs/new-model-family.md",
        "docs/supported/pack-routes-events-and-frontend-modules.md",
        "packages/dinkster-api/src/dinkster_api/v1.py",
        "packages/dinkster-inference/src/dinkster_inference/__init__.py",
        "packages/dinkster-inference/src/dinkster_inference/extensions.py",
        "packages/dinkster-inference/src/dinkster_inference/families.py",
        "packages/dinkster-inference/src/dinkster_inference/registries.py",
        "packages/dinkster-inference-torch/tests/test_new_family_checklist.py",
        "packages/dinkster-workers/src/dinkster_workers/doctor.py",
        "scripts/extension-factory-allowlist.json",
        "src/dinkster/compose.py",
        "tests/family_gate_scanner.py",
        "tests/fixtures/extension-contract-pack/dinkster-pack.toml",
        "tests/fixtures/extension-contract-pack/extension_contract_pack.py",
        "tests/test_api_v1.py",
        "tests/test_doctor.py",
        "tests/test_extension_contract_pack.py",
        "tests/test_family_registration_gates.py",
        "tests/test_inference_contracts.py",
        "tests/test_inference_extensions.py",
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


def changed_paths_since_merge_base(root: Path, base_ref: str = "origin/main") -> frozenset[str]:
    merge_base = subprocess.run(
        ("git", "merge-base", "HEAD", base_ref),
        cwd=root,
        capture_output=True,
        text=True,
    )
    if merge_base.returncode != 0:
        raise AssertionError(
            f"cannot resolve {base_ref}; validation checkouts must retain full history"
        )
    changed = subprocess.run(
        ("git", "diff", "--name-only", f"{merge_base.stdout.strip()}...HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return frozenset(changed)


def unexpected_new_family_paths(changed_paths: frozenset[str]) -> tuple[str, ...]:
    if NEW_FAMILY_PROOF_PATH not in changed_paths:
        return ()
    return tuple(sorted(changed_paths - NEW_FAMILY_REGISTRATION_PATHS))
