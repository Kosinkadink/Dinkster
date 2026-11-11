"""Select manifest-versioned official packs from a committed push range or pack tag."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import tomllib
from pathlib import Path, PurePosixPath

from dinkster_registry import validate_version
from dinkster_schema import validate_name
from dinkster_workers import load_manifest

from scripts.build_official_packs import is_official_pack_manifest, official_library


def publication_selection(root: Path, before: str) -> tuple[str, ...]:
    """Engine dependency and publication changes require checking every official pack."""
    if re.fullmatch(r"[0-9a-f]{40}", before) is None:
        raise ValueError("publication base must be a full Git commit identity")
    if before == "0" * 40:
        return select_packs(root, before=before)
    changed = subprocess.check_output(
        ["git", "-C", str(root), "diff", "--name-only", before, "HEAD"], text=True
    ).splitlines()
    engine_changed = any(
        path in {"pyproject.toml", "uv.lock", "scripts/base_bundles.json"}
        or path.startswith("packages/")
        and path.endswith("/pyproject.toml")
        for path in changed
    )
    publication_changed = any(
        path == ".github/workflows/official-packs.yml"
        or path
        in {
            "scripts/select_official_packs.py",
            "scripts/prepare_official_publication.py",
            "scripts/build_release_pack_locks.py",
            "scripts/build_official_packs.py",
            "scripts/publish_official_packs.py",
            "scripts/check_disposable_publication.py",
            "scripts/check_publication_controls.py",
            "scripts/stage_publication_probe_index.py",
        }
        for path in changed
    )
    return select_packs(root, before="0" * 40 if engine_changed or publication_changed else before)


def select_packs(
    root: Path,
    *,
    before: str | None = None,
    tag: str | None = None,
    unpublished_at: str | None = None,
    version_changes_only: bool = False,
    exclude_tests: bool = False,
) -> tuple[str, ...]:
    """Select committed identities by changed source, exact tag or absent public version."""
    if sum(value is not None for value in (before, tag, unpublished_at)) != 1:
        raise ValueError("select one push base, pack tag or public registry")
    if version_changes_only and before is None:
        raise ValueError("version changes require a push base")
    if before is not None and re.fullmatch(r"[0-9a-f]{40}", before) is None:
        raise ValueError("push base must be a full Git commit identity")
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    tracked = (
        subprocess.check_output(
            ["git", "-C", str(root), "ls-tree", "-r", "--name-only", "-z", head, "--", "packages"]
        )
        .decode()
        .split("\0")
    )
    vision = PurePosixPath("packages/dinkster-nodes-vision")
    packs = {}
    libraries = set()
    with tempfile.TemporaryDirectory(prefix="official-pack-selection-") as directory:
        manifest_path = Path(directory) / "dinkster-pack.toml"
        for name in tracked:
            path = PurePosixPath(name)
            if path.name == "pyproject.toml" and len(path.parts) == 3:
                project = tomllib.loads(
                    subprocess.check_output(
                        ["git", "-C", str(root), "show", f"{head}:{name}"]
                    ).decode("utf-8")
                )
                if identity := official_library(project):
                    library, version = identity
                    if f"{path.parent}/dinkster-pack.toml" in tracked or library in packs:
                        raise ValueError("official library duplicates a pack identity")
                    libraries.add(library)
                    packs[library] = (path.parent, path.parent, version)
                continue
            if not is_official_pack_manifest(path):
                continue
            manifest_path.write_bytes(
                subprocess.check_output(["git", "-C", str(root), "show", f"{head}:{name}"])
            )
            manifest = load_manifest(manifest_path)
            if manifest.compatibility is None or manifest.publisher != "dinkster":
                raise ValueError(f"official pack {manifest.name} lacks publication declarations")
            if manifest.name in packs:
                raise ValueError(f"duplicate official pack identity: {manifest.name}")
            source = vision / "src" / manifest.nodes_entry.partition(":")[0].replace(".", "/")
            packs[manifest.name] = (path.parent, source, manifest.compatibility.version)
    if unpublished_at is not None:
        published = published_versions(unpublished_at, set(packs))
        return tuple(
            sorted(
                name for name, (_, _, version) in packs.items() if (name, version) not in published
            )
        )
    if tag is not None:
        selected = [
            name for name, (_, _, version) in packs.items() if tag == f"pack/{name}/v{version}"
        ]
        if not selected:
            raise ValueError("pack tag must name an official id and its declared version")
        return tuple(selected)
    if before == "0" * 40:
        changed = tracked
    else:
        assert before is not None
        changed = (
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(root),
                    "diff",
                    "--name-only",
                    "--no-renames",
                    "-z",
                    before,
                    head,
                    "--",
                    "packages",
                ]
            )
            .decode()
            .split("\0")
        )
    selected = set()
    for name in filter(None, changed):
        path = PurePosixPath(name)
        # Official node wheels remove tests; library payloads come from their backend.
        if (
            exclude_tests
            and "tests" in path.parts[2:]
            and not any(path.is_relative_to(packs[library][0]) for library in libraries)
        ):
            continue
        matching = {
            pack
            for pack, (directory, source, _) in packs.items()
            if path.is_relative_to(directory)
            or directory.parent == vision
            and path.is_relative_to(source)
        }
        if path.is_relative_to(vision) and not matching:
            matching = {
                pack for pack, (directory, _, _) in packs.items() if directory.parent == vision
            }
        selected.update(matching)
    if version_changes_only and before != "0" * 40:
        assert before is not None
        previous_paths = (
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(root),
                    "ls-tree",
                    "-r",
                    "--name-only",
                    "-z",
                    before,
                    "--",
                    "packages",
                ]
            )
            .decode()
            .split("\0")
        )
        for pack in tuple(selected):
            directory, _, version = packs[pack]
            manifest = f"{directory}/" + (
                "pyproject.toml" if pack in libraries else "dinkster-pack.toml"
            )
            if manifest in previous_paths:
                previous = subprocess.check_output(
                    ["git", "-C", str(root), "show", f"{before}:{manifest}"]
                )
                old = tomllib.loads(previous.decode("utf-8"))
                if pack in libraries and official_library(old) is None:
                    continue
                if old["project" if pack in libraries else "pack"].get("version") == version:
                    selected.remove(pack)
    return tuple(sorted(selected))


def published_versions(registry: str, names: set[str]) -> set[tuple[str, str]]:
    """Read exact public identities without credentials or downloading wheel payloads."""
    from dinkster_supervisor.environment_downloads import read_metadata

    def document(path: str) -> dict:
        value = json.loads(read_metadata(registry, path))
        if (
            not isinstance(value, dict)
            or type(value.get("schemaVersion")) is not int
            or value["schemaVersion"] not in (1, 2)
            or not isinstance(value.get("items"), list)
            or any(not isinstance(row, dict) for row in value["items"])
        ):
            raise ValueError("publication requires a valid public static index")
        return value

    catalog = document("static/packs/index.json")
    releases = catalog["items"]
    if catalog["schemaVersion"] == 2:
        releases = []
        seen = set()
        for row in catalog["items"]:
            name, version = row.get("name"), row.get("latestVersion")
            if (
                not isinstance(name, str)
                or validate_name(name) is not None
                or not isinstance(version, str)
                or validate_version(version) is not None
                or name in seen
                or row.get("url") != f"packs/{name}/index.json"
            ):
                raise ValueError("publication static catalog has invalid pack identities")
            seen.add(name)
            if name not in names:
                continue
            pack = document("static/" + row["url"])
            if (
                pack["schemaVersion"] != 2
                or pack.get("name") != name
                or any(release.get("name") != name for release in pack["items"])
            ):
                raise ValueError("publication static per-pack index identity differs")
            releases.extend(pack["items"])
    published = set()
    for row in releases:
        name, version = row.get("name"), row.get("version")
        if (
            not isinstance(name, str)
            or validate_name(name) is not None
            or not isinstance(version, str)
            or validate_version(version) is not None
            or (name, version) in published
        ):
            raise ValueError("publication static index has invalid release identities")
        published.add((name, version))
    return published


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--before")
    selection.add_argument("--tag")
    args = parser.parse_args()
    print(
        json.dumps(
            select_packs(Path(__file__).resolve().parents[1], before=args.before, tag=args.tag)
        )
    )
