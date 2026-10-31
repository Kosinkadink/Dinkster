"""Resolve official pack dependencies on the selected base's native host."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
from dataclasses import replace
from pathlib import Path
from urllib.parse import quote, urlsplit

from dinkster_supervisor.environment_archives import extract_base
from dinkster_supervisor.environment_constraints import effective_constraints
from dinkster_supervisor.environment_downloads import fetch_artifact
from dinkster_workers.pack_lock import PackLock

from scripts.acquire_engine_bases import qualify
from scripts.build_official_packs import build
from scripts.build_release import sha256


def build_locks(root: Path, release: Path, manifest: Path, output: Path) -> tuple[Path, ...]:
    """Keep source payloads unchanged while adding native per-variant lock bytes."""
    raw = json.loads(manifest.read_bytes())
    variant = raw["variant"]
    base = qualify(root, manifest, f"base-{variant}-{raw['id'][7:]}")
    system = {"Windows": "windows", "Linux": "linux", "Darwin": "macos"}[platform.system()]
    if base["platform"]["os"] != system:
        raise ValueError("pack locks must be resolved on the selected base's native host")
    if output.exists():
        raise ValueError("pack lock output must be a new directory")
    engine = json.loads((release / "release-manifest.json").read_bytes())
    for row in engine["artifacts"]:
        path = release / row["name"]
        if path.stat().st_size != row["size"] or sha256(path) != row["sha256"]:
            raise ValueError(f"release artifact differs from manifest: {path.name}")
    if engine["workerProtocol"] != base["workerProtocol"]:
        raise ValueError("engine and selected base worker protocols differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="release-pack-locks-", dir=output.parent) as directory:
        work = Path(directory)
        archive = base["archive"]
        mirror = (
            "https://github.com/Kosinkadink/Dinkster-Environments/releases/download/"
            f"base-{variant}-{base['id'][7:]}"
        )
        parts = tuple(
            fetch_artifact(
                (mirror,),
                row["filename"],
                work / row["filename"],
                row["sha256"],
                row["size"],
            )
            for row in archive["parts"]
        )
        tree = work / "base"
        extract_base(
            parts,
            tree,
            platform=system,
            executable=base["interpreter"]["executable"],
            expected_parts=tuple(archive["parts"]),
        )
        constraints = work / "constraints.txt"
        constraints.write_text(
            effective_constraints((release / "constraints.txt").read_text(), base),
            encoding="ascii",
        )
        wheel_urls = {row["filename"]: row["url"] for row in base["wheels"]}
        wheel_urls.update(
            {
                row["name"]: (
                    "https://github.com/Kosinkadink/Dinkster/releases/download/"
                    f"{quote(engine['tag'], safe='')}/{quote(row['name'], safe='')}"
                )
                for row in engine["artifacts"]
                if row["name"].endswith(".whl")
            }
        )
        built = build(
            root,
            work / "packs",
            version=engine["version"].split("+", 1)[0],
            variant=variant,
            python=tree / base["interpreter"]["executable"],
            constraints=constraints,
            indexes=tuple(
                dict.fromkeys(["https://pypi.org/simple", *[w["index"] for w in base["wheels"]]])
            ),
            find_links=(release, tree / "wheels"),
            accelerator="cuda" if base["nvidia"] is not None else "cpu",
        )
        locks = [path for path in built if path.name.endswith(f"-{variant}.pack-lock.json")]
        staged = work / "output"
        staged.mkdir()
        for path in built:
            if path in locks:
                lock = PackLock.read(path)
                # Resolver-local find-links paths cannot escape into a published lock.
                replace(
                    lock,
                    packages=tuple(
                        replace(
                            package,
                            wheels=tuple(
                                replace(wheel, url=wheel_urls[wheel.filename])
                                if urlsplit(wheel.url).scheme == "file"
                                else wheel
                                for wheel in package.wheels
                            ),
                        )
                        for package in lock.packages
                    ),
                ).write(staged / path.name)
            elif (
                not (release / path.name).is_file()
                or path.read_bytes() != (release / path.name).read_bytes()
            ):
                raise ValueError(f"native pack build changed source artifact: {path.name}")
        if not locks:
            raise ValueError("native pack build produced no dependency locks")
        staged.rename(output)
    return tuple(sorted(output.iterdir()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    for path in build_locks(
        Path(__file__).resolve().parents[1],
        args.release.resolve(),
        args.base_manifest.resolve(),
        args.output.resolve(),
    ):
        print(path)
