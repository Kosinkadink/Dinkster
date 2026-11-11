"""Select trusted interrupted publication inputs without rebuilding immutable versions."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZipFile

from dinkster_registry import Lockfile

from scripts.select_official_packs import select_packs


class SupersededPublication(ValueError):
    """Original publication inputs no longer match the continuing source tree."""


def _pages(path: str, key: str) -> list[dict]:
    pages = json.loads(subprocess.check_output(["gh", "api", "--paginate", "--slurp", path]))
    return [row for page in pages for row in page[key]]


def recovery_inputs(
    root: Path, current_run: int, *, packs: list[str] | None = None
) -> dict[str, str]:
    """Reuse only complete source/lock production from an earlier interrupted main run."""
    if packs == []:
        return {}
    repository = "repos/Kosinkadink/Dinkster"
    runs = _pages(
        f"{repository}/actions/workflows/official-packs.yml/runs?branch=main&per_page=100",
        "workflow_runs",
    )
    completed: set[int] = set()
    variants = json.loads((root / "scripts/base_bundles.json").read_bytes())
    for run in runs:
        if run["id"] >= current_run:
            continue
        if (
            run["head_branch"] != "main"
            or run["head_repository"]["full_name"] != "Kosinkadink/Dinkster"
            or run["path"] != ".github/workflows/official-packs.yml"
            or run["event"] != "workflow_run"
        ):
            continue
        artifacts = _pages(
            f"{repository}/actions/runs/{run['id']}/artifacts?per_page=100", "artifacts"
        )
        if run["conclusion"] == "success":
            completed.update(
                int(match[1])
                for artifact in artifacts
                if (match := re.fullmatch(r"publication-completed-([1-9][0-9]*)", artifact["name"]))
            )
        if run["id"] in completed or run["conclusion"] not in {"failure", "cancelled"}:
            continue
        jobs = _pages(f"{repository}/actions/runs/{run['id']}/jobs?per_page=100", "jobs")
        sources = [job for job in jobs if job["name"] == "sources"]
        locks = [job for job in jobs if job["name"].startswith("locks (")]
        # Runs interrupted before producing immutable inputs cannot be continued.
        if not sources or sources[0]["conclusion"] != "success":
            print(f"publication run {run['id']}: continuation refused; source job did not succeed")
            continue
        if (
            len(sources) != 1
            or len(locks) != len(variants)
            or {job["name"].split(" (", 1)[1].split(",", 1)[0] for job in locks} != variants.keys()
            or any(job["conclusion"] != "success" for job in locks)
        ):
            print(
                f"publication run {run['id']}: continuation refused; "
                "all six native lock jobs must succeed"
            )
            continue
        expected = {"official-publication-inputs", *(f"publication-locks-{v}" for v in variants)}
        selected = [
            artifact
            for artifact in artifacts
            if artifact["name"] == "official-publication-inputs"
            or artifact["name"].startswith("publication-locks-")
        ]
        if (
            len(selected) != len(expected)
            or {artifact["name"] for artifact in selected} != expected
            or any(artifact["expired"] for artifact in selected)
        ):
            raise ValueError(f"publication run {run['id']}: artifacts must be unexpired and unique")
        head = run["head_sha"]
        if (
            re.fullmatch(r"[0-9a-f]{40}", head) is None
            or subprocess.run(
                ["git", "-C", str(root), "merge-base", "--is-ancestor", head, "HEAD"],
                check=False,
            ).returncode
        ):
            raise ValueError(f"publication run {run['id']}: source commit must belong to main")
        if any(
            artifact["workflow_run"]["id"] != run["id"]
            or artifact["workflow_run"]["head_sha"] != head
            for artifact in selected
        ):
            raise ValueError(f"publication run {run['id']}: artifact run identity differs")
        source_artifact = next(
            a["id"] for a in selected if a["name"] == "official-publication-inputs"
        )
        archive = subprocess.check_output(
            ["gh", "api", f"{repository}/actions/artifacts/{source_artifact}/zip"]
        )
        with tempfile.TemporaryDirectory(prefix="publication-recovery-") as directory:
            work = Path(directory)
            with ZipFile(io.BytesIO(archive)) as inputs:
                for name in ("sources/official-packs.lock", "engine/release-manifest.json"):
                    matches = [entry for entry in inputs.infolist() if entry.filename == name]
                    if len(matches) != 1:
                        raise ValueError(f"publication run {run['id']}: unique {name} required")
                    target = work / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(inputs.read(matches[0]))
            engine_head = json.loads((work / "engine/release-manifest.json").read_bytes())["commit"]
            if (
                re.fullmatch(r"[0-9a-f]{40}", engine_head) is None
                or subprocess.run(
                    ["git", "-C", str(root), "merge-base", "--is-ancestor", engine_head, head],
                    check=False,
                ).returncode
            ):
                raise ValueError(
                    "recovered engine commit must belong to the publication run's main"
                )
            try:
                validate_recovered_sources(root, work, engine_head)
            except SupersededPublication as error:
                print(f"publication run {run['id']}: continuation skipped; {error}")
                continue
            if packs is not None:
                catalog = Lockfile.from_record_json(
                    (work / "sources/official-packs.lock").read_text()
                )
                if {pack.pack for pack in catalog.packs} != set(packs):
                    print(
                        f"publication run {run['id']}: continuation skipped; "
                        "recovered sources do not match missing public versions"
                    )
                    continue
        return {
            "recovery_run": str(run["id"]),
            "recovery_head": engine_head,
            "recovery_artifact": str(source_artifact),
            "recovery_locks": ",".join(
                str(a["id"]) for a in selected if a["name"].startswith("publication-locks-")
            ),
        }
    return {}


def validate_recovered_sources(root: Path, work: Path, head: str) -> None:
    """Check the original selected Git bytes, not newly built wheels or newer engine pins."""
    catalog = Lockfile.from_record_json((work / "sources/official-packs.lock").read_text())
    names = {pack.pack for pack in catalog.packs}
    if not names:
        raise ValueError("recovered source catalog must identify the original publication commit")
    if not names.issubset(select_packs(root, before="0" * 40)):
        raise SupersededPublication("recovered selected source identity removed at continuing head")
    changed = names.intersection(select_packs(root, before=head, exclude_tests=True))
    if changed:
        raise SupersededPublication(
            f"recovered selected source bytes changed: {', '.join(sorted(changed))}"
        )
    if subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--name-only",
            head,
            "HEAD",
            "--",
            "scripts/base_bundles.json",
        ]
    ).strip():
        raise SupersededPublication("recovered supported base selection changed")
    if any(pack.source != f"git:Kosinkadink/Dinkster@{head}" for pack in catalog.packs):
        raise ValueError("recovered source catalog must identify the original publication commit")
    engine = json.loads((work / "engine/release-manifest.json").read_bytes())
    if engine["repository"] != "Kosinkadink/Dinkster" or engine["commit"] != head:
        raise ValueError("recovered engine identity differs from the original publication commit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=int)
    parser.add_argument("--packs", type=json.loads)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--head")
    args = parser.parse_args()
    root = Path.cwd()
    if args.validate is not None:
        if args.head is None:
            parser.error("--validate requires --head")
        validate_recovered_sources(root, args.validate, args.head)
    elif args.run is not None:
        outputs = recovery_inputs(root, args.run, packs=args.packs)
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            for key, value in outputs.items():
                output.write(f"{key}={value}\n")
        print(json.dumps(outputs))
    else:
        parser.error("select --run or --validate")
