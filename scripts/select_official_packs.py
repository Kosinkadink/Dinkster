"""Select manifest-versioned official packs from a committed push range or pack tag."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

from dinkster_workers import load_manifest

from scripts.build_official_packs import OFFICIAL_PACK_FAMILIES


def select_packs(
    root: Path, *, before: str | None = None, tag: str | None = None
) -> tuple[str, ...]:
    """Read committed manifests only; an engine-only push selects no pack."""
    if (before is None) == (tag is None):
        raise ValueError("select one push base or pack tag")
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
    families = {f"dinkster-nodes-{family}" for family in OFFICIAL_PACK_FAMILIES}
    vision = PurePosixPath("packages/dinkster-nodes-vision")
    packs = {}
    with tempfile.TemporaryDirectory(prefix="official-pack-selection-") as directory:
        manifest_path = Path(directory) / "dinkster-pack.toml"
        for name in tracked:
            path = PurePosixPath(name)
            if path.name != "dinkster-pack.toml" or not (
                len(path.parts) == 3
                and path.parts[1] in families
                or len(path.parts) == 4
                and path.parent.parent == vision
                and path.parent.name.endswith("_pack")
            ):
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
    return tuple(sorted(selected))


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
