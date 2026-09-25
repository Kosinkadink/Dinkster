"""Reject concrete family-plan branching outside classified loader boundaries."""

from __future__ import annotations

import argparse
import ast
import hashlib
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


FAMILY_VALUE_SUFFIXES = ("Runtime", "Carrier", "Latent", "Conditioning")


def source_files(root: Path) -> tuple[Path, ...]:
    packages = root / "packages"
    worker_environment = packages / "dinkster-workers/src/dinkster_workers/backend_env.py"
    files = tuple(
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
    return files + ((worker_environment,) if worker_environment.is_file() else ())


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


def named_types(node: ast.AST) -> tuple[tuple[str, str], ...]:
    if isinstance(node, ast.Name):
        return ((node.id, node.id),)
    if isinstance(node, ast.Attribute):
        return ((ast.unparse(node), node.attr),)
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(item for element in node.elts for item in named_types(element))
    return ()


def family_value_isinstance_types(call: ast.Call) -> tuple[str, ...]:
    if not isinstance(call.func, ast.Name) or call.func.id != "isinstance" or len(call.args) != 2:
        return ()
    return tuple(
        display
        for display, class_name in named_types(call.args[1])
        if class_name.endswith(FAMILY_VALUE_SUFFIXES)
    )


def exact_type_comparison(node: ast.Compare) -> str | None:
    if len(node.ops) != 1 or len(node.comparators) != 1:
        return None
    if not isinstance(node.ops[0], (ast.Is, ast.IsNot)):
        return None
    call = node.left
    checked_types = named_types(node.comparators[0])
    if not (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "type"
        and len(call.args) == 1
        and len(checked_types) == 1
        and checked_types[0][1].endswith(FAMILY_VALUE_SUFFIXES)
    ):
        return None
    return checked_types[0][0]


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
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent
        for node in ast.walk(tree):
            parent = parents.get(node)
            while parent is not None and not isinstance(
                parent, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                parent = parents.get(parent)
            scan_family_comparisons = "dinkster-native" in path.parts or (
                "dinkster-workers" in path.parts
                and isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
                and parent.name == "_residency_problems"
            )
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
            if isinstance(node, ast.Compare):
                checked_type = exact_type_comparison(node)
                if checked_type is not None:
                    sites.append(
                        {
                            "path": path.relative_to(root).as_posix(),
                            "line": node.lineno,
                            "column": node.col_offset + 1,
                            "type": checked_type,
                            "classification": "value-type",
                        }
                    )
                continue
            if not isinstance(node, ast.Call):
                continue
            family_value_types = family_value_isinstance_types(node)
            if family_value_types:
                sites.extend(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "line": node.lineno,
                        "column": node.col_offset + 1,
                        "type": checked_type,
                        "classification": "value-type",
                    }
                    for checked_type in family_value_types
                )
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
    parser.add_argument(
        "--write",
        action="store_true",
        help="freeze the currently scanned sites after reviewing them",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    allowlist = args.allowlist if args.allowlist.is_absolute() else root / args.allowlist
    try:
        raw = json.loads(allowlist.read_text(encoding="utf-8"))
        ceiling = raw["ceiling"]
        allowed_raw = raw["sites"]
        value_type_allowlist = raw.get(
            "value_type_allowlist",
            {"ceiling": 0, "sha256": hashlib.sha256(b"[]").hexdigest()},
        )
        if not isinstance(ceiling, int) or ceiling < 0 or not isinstance(allowed_raw, list):
            raise ValueError("invalid ceiling or sites")
        if (
            not isinstance(value_type_allowlist, dict)
            or set(value_type_allowlist) != {"ceiling", "sha256"}
            or not isinstance(value_type_allowlist["ceiling"], int)
            or value_type_allowlist["ceiling"] < 0
            or not isinstance(value_type_allowlist["sha256"], str)
        ):
            raise ValueError("invalid value-type allowlist")
        if any(
            not isinstance(site, dict)
            or set(site) != set(Site.__annotations__)
            or site.get("classification") not in ("boundary", "family-comparison", "value-type")
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
    scanned_value_types = [site for site in scanned if site["classification"] == "value-type"]
    scanned_classic = [site for site in scanned if site["classification"] != "value-type"]
    value_type_digest = hashlib.sha256(
        json.dumps(scanned_value_types, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if args.write:
        allowlist.write_text(
            json.dumps(
                {
                    "ceiling": ceiling,
                    "sites": allowed,
                    "value_type_allowlist": {
                        "ceiling": len(scanned_value_types),
                        "sha256": value_type_digest,
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return 0
    expected = [{key: site[key] for key in ScannedSite.__annotations__} for site in allowed]
    expected_sites = {json.dumps(site, sort_keys=True) for site in expected}
    value_types_match = (
        len(scanned_value_types) == value_type_allowlist["ceiling"]
        and value_type_digest == value_type_allowlist["sha256"]
    )
    prohibited = ([] if value_types_match else scanned_value_types) + [
        site
        for site in scanned_classic
        if site["classification"] == "branch"
        or (
            site["classification"] == "family-comparison"
            and json.dumps(site, sort_keys=True) not in expected_sites
        )
    ]
    if prohibited or len(scanned_classic) != ceiling or scanned_classic != expected:
        print(
            "Family isinstance gates differ from scripts/family-isinstance-allowlist.json:",
            file=sys.stderr,
        )
        print(
            f"  current={len(scanned_classic)}, allowlisted={len(allowed)}, ceiling={ceiling};"
            f" value-types={len(scanned_value_types)}/{value_type_allowlist['ceiling']}",
            file=sys.stderr,
        )
        for site in prohibited:
            print(
                f"  prohibited family branching {json.dumps(site, sort_keys=True)}", file=sys.stderr
            )
        current = {json.dumps(site, sort_keys=True) for site in scanned_classic}
        recorded = {json.dumps(site, sort_keys=True) for site in expected}
        for site in sorted(current - recorded):
            print(f"  unlisted {site}", file=sys.stderr)
        for site in sorted(recorded - current):
            print(f"  stale {site}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
