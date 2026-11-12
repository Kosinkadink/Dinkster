"""Exercise publisher ordering, dependency refusal and exact already-active replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from scripts.check_disposable_publication import check_publication

HELPER = "dinkster-dry-run-helpers"
CONSUMER = "dinkster-dry-run-consumer"


def prepare_fixture(
    bases: Path,
    output: Path,
    *,
    incompatible: bool = False,
    engine: Path | None = None,
    missing_export: bool = False,
) -> None:
    """Create ordinary wheels and complete locks, retaining the supplied base identities."""
    from dinkster_registry import LockedPack, Lockfile, artifact_digest
    from dinkster_workers.pack_lock import LockedDistribution, LockedWheel, PackLock
    from dinkster_workers.pack_wheel import build_pack_archive

    from scripts.build_release import read_base_manifest

    sources = output / "sources"
    sources.mkdir(parents=True)
    library = output / "library"
    module = library / "src/dinkster_dry_run_helpers"
    module.mkdir(parents=True)
    (module / "__init__.py").write_text("VALUE = 37\n", encoding="ascii")
    (library / "pyproject.toml").write_text(
        '[project]\nname = "dinkster-dry-run-helpers"\nversion = "0.0.1"\n'
        'requires-python = ">=3.12"\nlicense = "MIT"\n'
        '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n'
        '[tool.hatch.build.targets.wheel]\npackages = ["src/dinkster_dry_run_helpers"]\n',
        encoding="ascii",
    )
    subprocess.run(
        ["uv", "build", "--wheel", str(library), "--out-dir", str(sources)],
        check=True,
        timeout=120,
    )
    pack = output / "pack"
    module = pack / "dinkster_dry_run_consumer"
    module.mkdir(parents=True)
    api_import = ""
    if engine is not None:
        export = (
            "register_compositor_type_missing" if missing_export else "register_compositor_type"
        )
        api_import = f"from dinkster_api.v1 import {export}\n"
        destination = output / "engine"
        destination.mkdir()
        manifest = engine / "release-manifest.json"
        shutil.copyfile(manifest, destination / manifest.name)
        for artifact in json.loads(manifest.read_bytes())["artifacts"]:
            filename = artifact["name"]
            if Path(filename).name != filename:
                raise ValueError("fixture engine artifact escapes its directory")
            shutil.copyfile(engine / filename, destination / filename)
    (module / "__init__.py").write_text(
        api_import + "from dinkster_dry_run_helpers import VALUE\n"
        "from dinkster_api.v1 import Node, NodeSchema\n"
        "assert VALUE == 37\n"
        "class Example(Node):\n"
        "    @classmethod\n    def define_schema(cls):\n"
        "        return NodeSchema(node_type='dinkster.dry_run.example', "
        "display_name='Example', category='test', description='Helper import test.')\n"
        "    @classmethod\n    def execute(cls):\n        return {}\n"
        "NODES = [Example]\n",
        encoding="ascii",
    )
    requirement = HELPER + (">=2" if incompatible else "==0.0.1")
    (pack / "dinkster-pack.toml").write_text(
        '[pack]\nname = "dinkster-dry-run-consumer"\nversion = "0.0.1"\n'
        'publisher = "dinkster"\nengine = ">=0.0.2,<0.1"\n'
        f"requires = [{json.dumps(requirement)}]\n"
        'namespaces = ["dinkster.dry_run"]\nlicense = "MIT"\n'
        '[pack.entry]\nnodes = "dinkster_dry_run_consumer:NODES"\n'
        '[pack.contracts]\nhost = "dinkster-pack-host/1"\napi = "dinkster-api/v1"\n',
        encoding="ascii",
    )
    build_pack_archive(pack, sources / "dinkster_dry_run_consumer-0.0.1-py3-none-any.whl")
    entries = []
    for name in (CONSUMER, HELPER):
        archive = sources / f"{name.replace('-', '_')}-0.0.1-py3-none-any.whl"
        entries.append(
            LockedPack(name, "0.0.1", artifact_digest(archive.read_bytes()), "dinkster", ())
        )
    (sources / "official-packs.lock").write_text(
        Lockfile.of(entries).record_json(), encoding="ascii"
    )
    locks = output / "locks"
    locks.mkdir()
    helper = sources / "dinkster_dry_run_helpers-0.0.1-py3-none-any.whl"
    for source in sorted(bases.glob("*/manifest.json")):
        base = read_base_manifest(source)
        destination = output / "bases" / base["variant"]
        destination.mkdir(parents=True)
        shutil.copyfile(source, destination / "manifest.json")
        for entry in entries:
            packages = ()
            requirements = ()
            if entry.pack == CONSUMER:
                requirements = (requirement,)
                packages = (
                    LockedDistribution(
                        HELPER,
                        "0.0.1",
                        (
                            LockedWheel(
                                helper.name,
                                helper.stat().st_size,
                                hashlib.sha256(helper.read_bytes()).hexdigest(),
                                "http://wheels:8000/" + helper.name,
                            ),
                        ),
                    ),
                )
            PackLock(
                entry.pack,
                base["variant"],
                ">=3.12",
                requirements,
                ("http://wheels:8000/simple/",),
                packages,
            ).write(locks / f"{entry.pack}-{entry.version}-{base['variant']}.pack-lock.json")
    index = output / "index"
    simple = index / "simple" / HELPER
    simple.mkdir(parents=True)
    shutil.copyfile(helper, index / helper.name)
    (simple / "index.html").write_text(
        f'<a href="../../{helper.name}">{helper.name}</a>', encoding="ascii"
    )
    if engine is not None:
        from scripts.stage_publication_probe_index import stage_index

        stage_index(sources, index, engine=engine)


def check_controls(fixtures: Path, service_image: str, publisher_image: str) -> dict:
    root = Path(__file__).resolve().parents[1]
    runtime_mirror = fixtures / "runtime"
    good, broken = fixtures / "good", fixtures / "broken"
    results = []
    for reversed_order in (False, True):
        if reversed_order:
            path = good / "sources/official-packs.lock"
            catalog = json.loads(path.read_bytes())
            catalog["packs"].reverse()
            path.write_text(json.dumps(catalog), encoding="ascii")
        result = check_publication(
            root,
            good,
            good / "index",
            service_image,
            publisher_image,
            runtime_mirror=runtime_mirror,
        )
        first, replay = result["submissions"]
        if [row["pack"] for row in first] != [HELPER, CONSUMER]:
            raise RuntimeError("publisher did not submit the library before its consumer")
        if [row["state"] for row in replay] != ["published", "published"]:
            raise RuntimeError("unchanged replay did not recognize every active identity")
        results.append(result)
    try:
        check_publication(
            root,
            broken,
            broken / "index",
            service_image,
            publisher_image,
            runtime_mirror=runtime_mirror,
        )
    except RuntimeError as exc:
        expected = f"{CONSUMER}: selected {HELPER} excludes >=2"
        if expected not in str(exc):
            raise
    else:
        raise RuntimeError("publisher accepted an incompatible selected dependency")
    selected = fixtures / "matching-engine"
    matching = check_publication(
        root,
        selected,
        selected / "index",
        service_image,
        publisher_image,
        runtime_mirror=runtime_mirror,
    )
    missing = fixtures / "missing-export"
    try:
        check_publication(
            root,
            missing,
            missing / "index",
            service_image,
            publisher_image,
            runtime_mirror=runtime_mirror,
        )
    except RuntimeError as exc:
        if "entry.unresolvable" not in str(exc):
            raise
    else:
        raise RuntimeError("publisher accepted an export absent from the selected API")
    return {
        "orderingAndReplay": results,
        "incompatibleDependencyRefused": True,
        "matchingEngine": matching,
        "missingExportRefused": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--bases", required=True, type=Path)
    prepare.add_argument("--engine", required=True, type=Path)
    prepare.add_argument("--runtime-mirror", type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    check = commands.add_parser("check")
    check.add_argument("--fixtures", required=True, type=Path)
    check.add_argument("--service-image", required=True)
    check.add_argument("--publisher-image", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_fixture(
            args.bases.resolve(), args.output.resolve() / "good", engine=args.engine.resolve()
        )
        prepare_fixture(
            args.bases.resolve(),
            args.output.resolve() / "broken",
            incompatible=True,
            engine=args.engine.resolve(),
        )
        prepare_fixture(
            args.bases.resolve(),
            args.output.resolve() / "matching-engine",
            engine=args.engine.resolve(),
        )
        prepare_fixture(
            args.bases.resolve(),
            args.output.resolve() / "missing-export",
            engine=args.engine.resolve(),
            missing_export=True,
        )
        if args.runtime_mirror is not None:
            shutil.copytree(args.runtime_mirror.resolve(), args.output.resolve() / "runtime")
    else:
        print(
            json.dumps(
                check_controls(
                    args.fixtures.resolve(),
                    args.service_image,
                    args.publisher_image,
                )
            )
        )
