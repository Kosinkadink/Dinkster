"""Reject inference registry construction and catalog reads outside exact sites."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import TypedDict

SPECIAL_CALL_KINDS = {
    "build_builtin_assembly_registry": "assemblyBuilder",
    "builtin_families": "descriptorCatalog",
}
SITE_KINDS = ("assemblyBuilder", "descriptorCatalog", "registryFactory")
REGISTRY_FACTORY = re.compile(r"^(?:builtin|default)_.+_registry$")


class Site(TypedDict):
    kind: str
    call: str
    path: str
    line: int
    column: int
    issue: int


def source_files(root: Path) -> tuple[Path, ...]:
    paths = [root / "src"]
    packages = root / "packages"
    if packages.is_dir():
        paths.extend(path / "src" for path in packages.iterdir() if path.is_dir())
    return tuple(
        sorted(
            file
            for path in paths
            if path.is_dir()
            for file in path.rglob("*.py")
            if "__pycache__" not in file.parts
        )
    )


def call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def call_kind(call: str) -> str | None:
    if REGISTRY_FACTORY.fullmatch(call):
        return "registryFactory"
    return SPECIAL_CALL_KINDS.get(call)


def owning_issue(root: Path, path: Path) -> int:
    relative = path.relative_to(root).as_posix()
    if relative == "packages/dinkster-inference-torch/src/dinkster_inference_torch/memory.py":
        return 179
    return 120


def scan(root: Path) -> list[Site]:
    sites: list[Site] = []
    for path in source_files(root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call = call_name(node)
            kind = call_kind(call) if call is not None else None
            if kind is None or call is None:
                continue
            sites.append(
                {
                    "kind": kind,
                    "call": call,
                    "path": path.relative_to(root).as_posix(),
                    "line": node.lineno,
                    "column": node.col_offset + 1,
                    "issue": owning_issue(root, path),
                }
            )
    return sorted(
        sites,
        key=lambda site: (site["kind"], site["path"], site["line"], site["column"]),
    )


def load_allowlist(path: Path) -> tuple[dict[str, int], list[Site]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    ceilings = raw.get("ceilings")
    sites = raw.get("sites")
    if (
        not isinstance(ceilings, dict)
        or not all(
            isinstance(kind, str) and isinstance(ceiling, int) and ceiling >= 0
            for kind, ceiling in ceilings.items()
        )
        or not isinstance(sites, list)
    ):
        raise ValueError("allowlist requires nonnegative ceilings and a sites array")
    return ceilings, sites


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=Path("scripts/extension-factory-allowlist.json"),
    )
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    allowlist = args.allowlist
    if not allowlist.is_absolute():
        allowlist = root / allowlist
    sites = scan(root)
    if args.write:
        ceilings = {kind: sum(site["kind"] == kind for site in sites) for kind in SITE_KINDS}
        allowlist.parent.mkdir(parents=True, exist_ok=True)
        allowlist.write_text(
            json.dumps({"ceilings": ceilings, "sites": sites}, indent=2) + "\n",
            encoding="utf-8",
        )
        return 0

    try:
        ceilings, allowed = load_allowlist(allowlist)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Cannot read extension factory allowlist: {error}", file=sys.stderr)
        return 1
    kinds = set(SITE_KINDS)
    counts = {kind: sum(site["kind"] == kind for site in sites) for kind in kinds}
    allowed_counts = {kind: sum(site.get("kind") == kind for site in allowed) for kind in kinds}
    ceiling_drift = set(ceilings) != kinds or any(
        ceilings.get(kind) != allowed_counts[kind] or counts[kind] > ceilings.get(kind, -1)
        for kind in kinds
    )
    if ceiling_drift or sites != allowed:
        print(
            "Inference registry and descriptor catalog sites differ from "
            "scripts/extension-factory-allowlist.json:",
            file=sys.stderr,
        )
        for kind in sorted(kinds | set(ceilings)):
            print(
                f"  {kind}: current={counts.get(kind, 0)}, "
                f"allowlisted={allowed_counts.get(kind, 0)}, ceiling={ceilings.get(kind)!r}",
                file=sys.stderr,
            )
        current = {json.dumps(site, sort_keys=True) for site in sites}
        expected = {json.dumps(site, sort_keys=True) for site in allowed}
        for site in sorted(current - expected):
            print(f"  unlisted {site}", file=sys.stderr)
        for site in sorted(expected - current):
            print(f"  stale {site}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
