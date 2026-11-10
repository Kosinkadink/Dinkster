"""Exercise publisher ordering, dependency refusal and exact already-active replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from dinkster_registry import LockedPack, Lockfile, artifact_digest
from dinkster_workers.pack_lock import LockedDistribution, LockedWheel, PackLock
from dinkster_workers.pack_wheel import build_pack_archive

from scripts.build_release import read_base_manifest
from scripts.check_disposable_publication import check_publication

HELPER = "dinkster-dry-run-helpers"
CONSUMER = "dinkster-dry-run-consumer"


def prepare_fixture(bases: Path, output: Path, *, incompatible: bool = False) -> None:
    """Create ordinary wheels and complete locks, retaining the supplied base identities."""
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
    (module / "__init__.py").write_text(
        "from dinkster_dry_run_helpers import VALUE\n"
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


def check_controls(bases: Path, service_image: str, publisher_image: str) -> dict:
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="disposable-publication-controls-") as directory:
        work = Path(directory)
        good, broken = work / "good", work / "broken"
        prepare_fixture(bases, good)
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
            )
            first, replay = result["submissions"]
            if [row["pack"] for row in first] != [HELPER, CONSUMER]:
                raise RuntimeError("publisher did not submit the library before its consumer")
            if [row["state"] for row in replay] != ["published", "published"]:
                raise RuntimeError("unchanged replay did not recognize every active identity")
            results.append(result)
        prepare_fixture(bases, broken, incompatible=True)
        try:
            check_publication(root, broken, broken / "index", service_image, publisher_image)
        except RuntimeError as exc:
            expected = f"{CONSUMER}: selected {HELPER} excludes >=2"
            if expected not in str(exc):
                raise
        else:
            raise RuntimeError("publisher accepted an incompatible selected dependency")
        return {"orderingAndReplay": results, "incompatibleDependencyRefused": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bases", required=True, type=Path)
    parser.add_argument("--service-image", required=True)
    parser.add_argument("--publisher-image", required=True)
    args = parser.parse_args()
    print(
        json.dumps(check_controls(args.bases.resolve(), args.service_image, args.publisher_image))
    )
