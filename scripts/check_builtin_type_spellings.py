"""Require built-in socket identities to come from the spelling module.

Widget tags and external ComfyUI protocol labels are not socket identities.
The guard checks type constructors, registry calls, and runtime type tests;
it does not conflate those independent vocabularies just because they use
the same word today.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

SPELLINGS = Path("packages/dinkster-values/src/dinkster_values/builtin_types.py")
REGISTRY_CALLS = frozenset(
    {
        "register",
        "spec",
        "wrap",
        "coerce",
        "owner_of",
        "declare_types",
        "register_rendition",
        "register_asset_decoder",
        "asset_decoder_for",
        "register_batch_merge",
        "batch_merge_for",
        "register_type_equivalence",
    }
)


def type_test(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return not node.id.endswith("_IO_TYPE") and node.id.endswith(
            ("type_id", "type_ids", "registry", "_TYPE", "_TYPE_ID")
        )
    return isinstance(node, ast.Attribute) and node.attr in {"type_id", "type_ids"}


def violations(root: Path) -> list[str]:
    declarations = ast.parse((root / SPELLINGS).read_text(encoding="utf-8"))
    ids = {
        node.value.value
        for node in declarations.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }
    roots = [root / "src", root / "scripts", root / "tools"]
    roots.extend(
        path
        for path in (root / "packages").rglob("src")
        if path.is_dir() and not {"tests", "test", "fixtures"}.intersection(path.parts)
    )
    errors: set[str] = set()
    for directory in roots:
        for path in directory.rglob("*.py"):
            if path == root / SPELLINGS or "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                values: list[ast.AST] = []
                concrete = False
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    concrete = node.func.attr == "concrete"
                    if concrete or node.func.attr in REGISTRY_CALLS:
                        values = list(node.args)
                elif isinstance(node, ast.Compare):
                    operands = [node.left, *node.comparators]
                    if any(type_test(operand) for operand in operands):
                        values = operands
                elif isinstance(node, ast.Assign) and any(
                    type_test(target) for target in node.targets
                ):
                    values = [node.value]
                elif isinstance(node, ast.AnnAssign) and type_test(node.target):
                    values = [] if node.value is None else [node.value]
                for value in values:
                    literals = (value,) if concrete else ast.walk(value)
                    for literal in literals:
                        if not isinstance(literal, ast.Constant) or not isinstance(
                            literal.value, str
                        ):
                            continue
                        if concrete or literal.value in ids:
                            errors.add(
                                f"{path.relative_to(root)}:{literal.lineno}: "
                                f"import the type constant instead of {literal.value!r}"
                            )
    return sorted(errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    errors = violations(args.root)
    for error in errors:
        print(error)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
