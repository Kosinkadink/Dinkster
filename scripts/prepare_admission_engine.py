"""Build the exact engine checkout and provision it on its named managed base."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from dinkster_supervisor.environment_artifacts import (
    EnvironmentCatalog,
    acquire,
    validate_manifest,
)
from dinkster_supervisor.environment_generations import prepare_engine_interpreter
from dinkster_supervisor.environment_state import Channel, interpreter, writer_lock
from dinkster_workers.pack_lock import GenerationPackages

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


def prepare_engine(
    root: Path,
    output: Path,
    store: Path,
    base_manifest: Path,
    *,
    engine_source: Path | None = None,
    version: str | None = None,
) -> Path:
    """Use release wheel selection and the user's verified generation constructor."""
    raw = json.loads(base_manifest.read_bytes())
    base = validate_manifest(base_manifest.read_bytes(), raw["id"], raw["variant"])
    variant = raw["variant"]
    pinned = json.loads((root / "scripts/base_bundles.json").read_text())[variant]
    if raw["id"] != "sha256:" + pinned:
        raise ValueError("admission base differs from the engine checkout's pinned base")
    version = version or development_version(root)
    output.mkdir()
    release = output / "engine"
    if engine_source is None:
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
    else:
        shutil.copytree(engine_source, release)
    manifest = release / "release-manifest.json"
    identity = "sha256:" + sha256(manifest)
    engine = validate_manifest(manifest.read_bytes(), identity, None)
    if engine["commit"] != git_head(root) or engine["version"] != version:
        raise ValueError("prepared engine does not match the checkout and version")
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
    shutil.copyfile(base_manifest, output / "base-manifest.json")
    facts = GenerationPackages.inspect(python)
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
                "constraintsSha256": sha256(output / "generation/constraints.txt"),
                "packages": dict(facts.versions),
                "pythonVersion": facts.python_version,
            }
        )
        + "\n",
        encoding="ascii",
    )
    return python


def read_runtime(output: Path, root: Path, version: str) -> tuple[Path, str, Path, str]:
    """Bind the checking interpreter to the verified engine, base and installed facts."""
    runtime = json.loads((output / "runtime.json").read_bytes())
    if runtime["commit"] != git_head(root) or runtime["engineVersion"] != version:
        raise ValueError("admission runtime does not match the checkout and version")
    variant = runtime["variant"]
    pinned = json.loads((root / "scripts/base_bundles.json").read_text())[variant]
    base_id = "sha256:" + pinned
    if runtime["baseId"] != base_id:
        raise ValueError("admission runtime does not match the pinned base")
    validate_manifest((output / "base-manifest.json").read_bytes(), base_id, variant)
    engine = validate_manifest(
        (output / "engine/release-manifest.json").read_bytes(), runtime["engineRelease"], None
    )
    if engine["commit"] != runtime["commit"] or engine["version"] != version:
        raise ValueError("admission runtime engine identity differs")
    python = interpreter(output / "generation")
    constraints = output / "generation/constraints.txt"
    if str(python) != runtime["python"] or str(constraints) != runtime["constraints"]:
        raise ValueError("admission runtime paths differ from its generation")
    if sha256(constraints) != runtime["constraintsSha256"]:
        raise ValueError("admission runtime constraints changed")
    facts = GenerationPackages.inspect(python)
    if (
        dict(facts.versions) != runtime["packages"]
        or facts.python_version != runtime["pythonVersion"]
    ):
        raise ValueError("admission runtime installed packages changed")
    return python, base_id, constraints, variant


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--engine-source", type=Path)
    parser.add_argument("--engine-version")
    args = parser.parse_args()
    print(
        prepare_engine(
            Path(__file__).resolve().parents[1],
            args.output.resolve(),
            args.store.resolve(),
            args.base_manifest.resolve(),
            engine_source=args.engine_source.resolve() if args.engine_source else None,
            version=args.engine_version,
        )
    )
