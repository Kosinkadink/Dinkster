"""Reject concrete family-plan branching outside classified loader boundaries."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import TypedDict


class ScannedSite(TypedDict):
    path: str
    line: int
    column: int
    type: str
    classification: str


class Site(ScannedSite):
    reason: str


def source_files(root: Path) -> tuple[Path, ...]:
    packages = root / "packages"
    return tuple(
        sorted(
            file
            for package in packages.iterdir()
            if package.is_dir()
            and (
                package.name == "dinkster-inference"
                or package.name.startswith("dinkster-inference-")
                or package.name == "dinkster-engine"
                or package.name.startswith("dinkster-engine-")
            )
            for file in (package / "src").rglob("*.py")
            if (package / "src").is_dir()
        )
    ) + tuple(
        sorted(
            file
            for file in (packages / "dinkster-native/src/dinkster_native").glob("*.py")
            if file.name.startswith(("native_arm", "nodes_"))
        )
    )


def isinstance_type(call: ast.Call) -> str | None:
    if not isinstance(call.func, ast.Name) or call.func.id != "isinstance" or len(call.args) != 2:
        return None
    checked_type = call.args[1]
    if not isinstance(checked_type, ast.Name):
        return None
    name = checked_type.id
    if (
        name == "NativeAssemblyPlan"
        or name.endswith("SplitAssemblyPlan")
        or not name.endswith("AssemblyPlan")
    ):
        return None
    return name


def family_comparison(node: ast.Compare) -> str | None:
    if len(node.ops) != 1 or len(node.comparators) != 1:
        return None
    selector = node.left
    if not (
        isinstance(selector, ast.Name)
        and selector.id in ("family", "family_id")
        or isinstance(selector, ast.Attribute)
        and selector.attr == "family_id"
    ):
        return None
    operator = node.ops[0]
    if isinstance(operator, ast.Eq):
        return f"{ast.unparse(selector)} =="
    if isinstance(operator, ast.In):
        return f"{ast.unparse(selector)} in"
    return None


def scan(root: Path) -> list[ScannedSite]:
    sites: list[ScannedSite] = []
    for path in source_files(root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        scan_family_comparisons = "dinkster-native" in path.parts
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent
        for node in ast.walk(tree):
            if scan_family_comparisons and isinstance(node, ast.Compare):
                comparison = family_comparison(node)
                if comparison is not None:
                    sites.append(
                        {
                            "path": path.relative_to(root).as_posix(),
                            "line": node.lineno,
                            "column": node.col_offset + 1,
                            "type": comparison,
                            "classification": "family-comparison",
                        }
                    )
                continue
            if not isinstance(node, ast.Call):
                continue
            checked_type = isinstance_type(node)
            if checked_type is None:
                continue
            unary = parents.get(node)
            branch = parents.get(unary) if isinstance(unary, ast.UnaryOp) else None
            boundary = (
                isinstance(unary, ast.UnaryOp)
                and isinstance(unary.op, ast.Not)
                and isinstance(branch, ast.If)
                and branch.test is unary
                and not branch.orelse
                and len(branch.body) == 1
                and isinstance(branch.body[0], ast.Raise)
            )
            sites.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "line": node.lineno,
                    "column": node.col_offset + 1,
                    "type": checked_type,
                    "classification": "boundary" if boundary else "branch",
                }
            )
    return sorted(sites, key=lambda site: (site["path"], site["line"], site["column"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=Path("scripts/family-isinstance-allowlist.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    allowlist = args.allowlist if args.allowlist.is_absolute() else root / args.allowlist
    try:
        raw = json.loads(allowlist.read_text(encoding="utf-8"))
        ceiling = raw["ceiling"]
        allowed_raw = raw["sites"]
        if not isinstance(ceiling, int) or ceiling < 0 or not isinstance(allowed_raw, list):
            raise ValueError("invalid ceiling or sites")
        if any(
            not isinstance(site, dict)
            or set(site) != set(Site.__annotations__)
            or site.get("classification") not in ("boundary", "family-comparison")
            or not isinstance(site.get("reason"), str)
            or not site["reason"].strip()
            or "\n" in site["reason"]
            for site in allowed_raw
        ):
            raise ValueError(
                "each allowed site requires a recognized classification and one-line reason"
            )
        allowed: list[Site] = allowed_raw
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"Cannot read family isinstance allowlist: {error}", file=sys.stderr)
        return 1

    scanned = scan(root)
    expected = [{key: site[key] for key in ScannedSite.__annotations__} for site in allowed]
    expected_sites = {json.dumps(site, sort_keys=True) for site in expected}
    prohibited = [
        site
        for site in scanned
        if site["classification"] == "branch"
        or (
            site["classification"] == "family-comparison"
            and json.dumps(site, sort_keys=True) not in expected_sites
        )
    ]
    if prohibited or len(scanned) != ceiling or scanned != expected:
        print(
            "Family isinstance gates differ from scripts/family-isinstance-allowlist.json:",
            file=sys.stderr,
        )
        print(
            f"  current={len(scanned)}, allowlisted={len(allowed)}, ceiling={ceiling}",
            file=sys.stderr,
        )
        for site in prohibited:
            print(
                f"  prohibited family branching {json.dumps(site, sort_keys=True)}", file=sys.stderr
            )
        current = {json.dumps(site, sort_keys=True) for site in scanned}
        recorded = {json.dumps(site, sort_keys=True) for site in expected}
        for site in sorted(current - recorded):
            print(f"  unlisted {site}", file=sys.stderr)
        for site in sorted(recorded - current):
            print(f"  stale {site}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
