"""Build the exact engine checkout and provision it on its named managed base."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from dinkster_supervisor.environment_artifacts import (
    EnvironmentCatalog,
    acquire,
    validate_manifest,
)
from dinkster_supervisor.environment_generations import prepare_engine_interpreter
from dinkster_supervisor.environment_state import Channel, writer_lock

from scripts.build_release import (
    PUBLISHED_PROJECTS,
    development_version,
    git_head,
    select_dependency_wheels,
    sha256,
    validate_engine_wheels,
    versioned_sources,
    write_constraints,
    write_manifest,
)


def prepare_engine(root: Path, output: Path, store: Path, base_manifest: Path) -> Path:
    """Use release wheel selection and the user's verified generation constructor."""
    raw = json.loads(base_manifest.read_bytes())
    base = validate_manifest(base_manifest.read_bytes(), raw["id"], raw["variant"])
    variant = raw["variant"]
    version = development_version(root)
    output.mkdir()
    release = output / "engine"
    release.mkdir()
    with versioned_sources(root, version) as staged:
        for project in PUBLISHED_PROJECTS:
            subprocess.run(
                ["uv", "build", "--wheel", "--package", project, "--out-dir", str(release)],
                cwd=staged,
                check=True,
            )
    wheels = sorted(release.glob("*.whl"))
    validate_engine_wheels(wheels)
    constraints = write_constraints(root, release, wheels, "uv")
    engine: dict[str, object] = {
        "format": "dinkster.engine/1",
        "repository": "Kosinkadink/Dinkster",
        "commit": git_head(root),
        "version": version,
        "workerProtocol": base["workerProtocol"],
        "workspaceWheels": [wheel.name for wheel in wheels],
        "dependencies": select_dependency_wheels(root, "uv", [base_manifest]),
    }
    write_manifest(release, engine, [*wheels, constraints])
    manifest = release / "release-manifest.json"
    identity = "sha256:" + sha256(manifest)
    document: dict[str, object] = {
        "format": "dinkster.channel/1",
        "engine": {
            "version": version,
            "release": identity,
            "locations": [
                {"type": "mirror", "path": f"engine/{identity[7:]}/release-manifest.json"}
            ],
        },
        "base": {
            variant: {
                "id": raw["id"],
                "locations": [
                    {"type": "mirror", "path": f"base/{variant}/{raw['id'][7:]}/manifest.json"}
                ],
            }
        },
        "minimumDesktopVersion": "0.0.0",
    }
    base_source = f"https://releases.dinkster.org/base/{variant}/{raw.get('label', raw['id'][7:])}"
    catalog = EnvironmentCatalog(
        "main",
        document,
        Channel.parse(document),
        engine,
        (str(release),),
        {variant: base},
        {variant: (base_source,)},
        manifest.read_bytes(),
    )
    store.mkdir(parents=True, exist_ok=True)
    with writer_lock(store):
        effective = acquire(catalog, variant, store)
        python = prepare_engine_interpreter(
            output / "generation", store=store, base=raw["id"], constraints=effective
        )
    (output / "runtime.json").write_text(
        json.dumps(
            {
                "commit": git_head(root),
                "engineVersion": version,
                "engineRelease": identity,
                "baseId": raw["id"],
                "variant": variant,
                "python": str(python),
                "constraints": str(output / "generation/constraints.txt"),
            }
        )
        + "\n",
        encoding="ascii",
    )
    return python


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    args = parser.parse_args()
    print(
        prepare_engine(
            Path(__file__).resolve().parents[1],
            args.output.resolve(),
            args.store.resolve(),
            args.base_manifest.resolve(),
        )
    )
