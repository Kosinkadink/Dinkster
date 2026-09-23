"""Reject inference registry construction and catalog reads outside exact sites."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

SPECIAL_CALL_KINDS = {
    "AttentionRegistry": "attentionFactory",
    "AttentionQKVDescriptor": "attentionFactory",
    "AttentionWrapperDescriptor": "attentionFactory",
    "AttentionOutputDescriptor": "attentionFactory",
    "AttentionBackendDescriptor": "attentionFactory",
    "BlockInjectionDescriptor": "attentionFactory",
    "build_builtin_assembly_registry": "assemblyBuilder",
    "builtin_families": "descriptorCatalog",
    "builtin_sampler_snapshot": "descriptorCatalog",
    "builtin_samplers": "descriptorCatalog",
    "builtin_schedulers": "descriptorCatalog",
}
SITE_KINDS = ("assemblyBuilder", "attentionFactory", "descriptorCatalog", "registryFactory")
REGISTRY_FACTORY = re.compile(r"^(?:builtin|default)_.+_registry$")


class ScannedSite(TypedDict):
    kind: str
    call: str
    path: str
    line: int
    column: int


class Site(ScannedSite):
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


def scan(root: Path) -> list[ScannedSite]:
    sites: list[ScannedSite] = []
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
                }
            )
    return sorted(
        sites,
        key=lambda site: (site["kind"], site["path"], site["line"], site["column"]),
    )


def parse_allowlist(
    raw: object, *, require_slack: bool = True
) -> tuple[dict[str, int], dict[str, int], list[Site]]:
    if not isinstance(raw, dict):
        raise ValueError("allowlist must be an object")
    ceilings = raw.get("ceilings")
    slack = raw.get("slack")
    sites = raw.get("sites")
    if (
        not isinstance(ceilings, dict)
        or not all(
            isinstance(kind, str) and isinstance(ceiling, int) and ceiling >= 0
            for kind, ceiling in ceilings.items()
        )
        or (require_slack and not isinstance(slack, dict))
        or (
            isinstance(slack, dict)
            and not all(
                isinstance(kind, str) and isinstance(value, int) and value >= 0
                for kind, value in slack.items()
            )
        )
        or not isinstance(sites, list)
        or not all(isinstance(site, dict) for site in sites)
    ):
        raise ValueError("allowlist requires nonnegative ceilings, slack, and a sites array")
    return ceilings, slack if isinstance(slack, dict) else {}, sites


def load_allowlist(path: Path) -> tuple[dict[str, int], dict[str, int], list[Site]]:
    return parse_allowlist(json.loads(path.read_text(encoding="utf-8")))


def load_baseline(
    *, root: Path, baseline_ref: str | None, baseline_allowlist: Path | None
) -> dict[str, int] | None:
    if baseline_ref is not None and baseline_allowlist is not None:
        raise ValueError("choose either --baseline-ref or --baseline-allowlist")
    if baseline_allowlist is not None:
        ceilings, _, _ = load_allowlist(baseline_allowlist)
        return ceilings
    if baseline_ref is None:
        return None
    merge_base = subprocess.run(
        ["git", "merge-base", "HEAD", baseline_ref],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if merge_base.returncode != 0 or not merge_base.stdout.strip():
        raise ValueError(
            f"cannot resolve merge base for {baseline_ref}: {merge_base.stderr.strip()}"
        )
    result = subprocess.run(
        [
            "git",
            "show",
            f"{merge_base.stdout.strip()}:scripts/extension-factory-allowlist.json",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"cannot read baseline {baseline_ref}: {result.stderr.strip()}")
    ceilings, _, _ = parse_allowlist(json.loads(result.stdout), require_slack=False)
    return ceilings


def attach_owning_issues(
    scanned: list[ScannedSite], allowed: list[Site]
) -> tuple[list[Site], list[ScannedSite]]:
    remaining = list(allowed)
    resolved: list[Site | None] = [None] * len(scanned)

    def assign(exact: bool) -> None:
        for index, site in enumerate(scanned):
            if resolved[index] is not None:
                continue
            for allowed_index, candidate in enumerate(remaining):
                same_site = (
                    candidate.get("kind") == site["kind"]
                    and candidate.get("call") == site["call"]
                    and candidate.get("path") == site["path"]
                )
                if exact:
                    same_site = (
                        same_site
                        and candidate.get("line") == site["line"]
                        and candidate.get("column") == site["column"]
                    )
                issue = candidate.get("issue")
                if same_site and isinstance(issue, int) and issue > 0:
                    resolved[index] = {**site, "issue": issue}
                    remaining.pop(allowed_index)
                    break

    assign(exact=True)
    assign(exact=False)
    return (
        [site for site in resolved if site is not None],
        [site for index, site in enumerate(scanned) if resolved[index] is None],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=Path("scripts/extension-factory-allowlist.json"),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help=(
            "refresh owned sites while preserving or lowering committed ceilings; "
            "new sites require a manually added positive issue"
        ),
    )
    parser.add_argument("--baseline-ref")
    parser.add_argument("--baseline-allowlist", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    allowlist = args.allowlist
    if not allowlist.is_absolute():
        allowlist = root / allowlist
    scanned = scan(root)
    try:
        existing_ceilings, slack, allowed = load_allowlist(allowlist)
        baseline_allowlist = args.baseline_allowlist
        if baseline_allowlist is not None and not baseline_allowlist.is_absolute():
            baseline_allowlist = root / baseline_allowlist
        baseline_ceilings = load_baseline(
            root=root,
            baseline_ref=args.baseline_ref,
            baseline_allowlist=baseline_allowlist,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Cannot read extension factory allowlist: {error}", file=sys.stderr)
        return 1
    kinds = set(SITE_KINDS)
    if set(existing_ceilings) != kinds or set(slack) != kinds:
        print("Extension factory ceiling and slack kinds differ", file=sys.stderr)
        return 1
    if baseline_ceilings is not None:
        raised_from_baseline = [
            kind for kind in SITE_KINDS if existing_ceilings[kind] > baseline_ceilings.get(kind, -1)
        ]
        if raised_from_baseline:
            print("Extension factory ceilings must not rise from the merge base:", file=sys.stderr)
            for kind in raised_from_baseline:
                print(
                    f"  {kind}: baseline={baseline_ceilings.get(kind)!r}, "
                    f"proposed={existing_ceilings[kind]}",
                    file=sys.stderr,
                )
            return 1
    sites, unowned = attach_owning_issues(scanned, allowed)
    if unowned:
        print(
            "Inference registry and descriptor catalog sites require explicit owning issues:",
            file=sys.stderr,
        )
        for site in unowned:
            print(f"  unlisted {json.dumps(site, sort_keys=True)}", file=sys.stderr)
        print(
            "Add each site to the allowlist with a positive issue, then refresh with "
            "`uv run --locked python scripts/check_extension_factories.py --write`.",
            file=sys.stderr,
        )
        return 1
    if args.write:
        counts = {kind: sum(site["kind"] == kind for site in sites) for kind in SITE_KINDS}
        raised = [kind for kind in SITE_KINDS if counts[kind] > existing_ceilings[kind]]
        if raised:
            print(
                "Cannot refresh extension factory allowlist without raising ceilings:",
                file=sys.stderr,
            )
            for kind in raised:
                print(
                    f"  {kind}: current={counts[kind]}, ceiling={existing_ceilings[kind]}",
                    file=sys.stderr,
                )
            return 1
        ceilings = {kind: min(existing_ceilings[kind], counts[kind]) for kind in SITE_KINDS}
        allowlist.parent.mkdir(parents=True, exist_ok=True)
        allowlist.write_text(
            json.dumps({"ceilings": ceilings, "slack": slack, "sites": sites}, indent=2) + "\n",
            encoding="utf-8",
        )
        return 0

    ceilings = existing_ceilings
    counts = {kind: sum(site["kind"] == kind for site in sites) for kind in kinds}
    allowed_counts = {kind: sum(site.get("kind") == kind for site in allowed) for kind in kinds}
    ceiling_drift = any(
        counts[kind] > ceilings[kind] or ceilings[kind] - counts[kind] > slack[kind]
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
        print(
            "Refresh intentional site changes with "
            "`uv run --locked python scripts/check_extension_factories.py --write`, "
            "then review the diff and confirm no ceiling rose.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
