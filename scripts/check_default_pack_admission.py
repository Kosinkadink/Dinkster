"""Resolve registry defaults and run the registry scanner against this engine."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
from datetime import UTC, datetime
from graphlib import TopologicalSorter
from pathlib import Path

from dinkster_registry import unpack_artifact
from dinkster_workers import load_manifest

from dinkster.registries import NamedRegistry, default_pack_minimums, prepare_registry_sources


def check_defaults(
    registry: NamedRegistry, scanner: Path, *, engine_version: str
) -> dict[str, object]:
    """Use disposable local admission state; never submit to the public registry."""
    source = scanner.resolve(strict=True) / "src"
    if not (source / "dinkster_registry_service/admission.py").is_file():
        raise ValueError("scanner must be a registry service source checkout")
    sys.path.insert(0, str(source))
    try:
        admission = importlib.import_module("dinkster_registry_service.admission")
        databases = importlib.import_module("dinkster_registry_service.database")
        objects = importlib.import_module("dinkster_registry_service.object_store")
        principals = importlib.import_module("dinkster_registry_service.principals")
        settings = importlib.import_module("dinkster_registry_service.settings")
        if admission.__file__ is None or not Path(admission.__file__).resolve().is_relative_to(
            source
        ):
            raise ValueError("loaded scanner differs from the requested source checkout")
    finally:
        sys.path.remove(str(source))
    with tempfile.TemporaryDirectory(prefix="default-pack-admission-") as directory:
        work = Path(directory)
        artifacts = work / "artifacts"
        catalog, _ = prepare_registry_sources(
            registry, default_pack_minimums(), artifacts, engine_version=engine_version
        )
        if not catalog.packs:
            raise ValueError("engine admission requires a nonempty default set")
        entries = {pack.pack: pack for pack in catalog.packs}
        graph: dict[str, set[str]] = {}
        archives = {}
        for pack in catalog.packs:
            archive = artifacts / f"{pack.artifact_digest.removeprefix('blake3:')}.zip"
            root = unpack_artifact(archive, work / "sources" / pack.pack, pack.artifact_digest)
            manifest = load_manifest(root / "dinkster-pack.toml")
            dependencies = {dependency.pack for dependency in manifest.dependencies}
            if missing := dependencies - entries.keys():
                raise ValueError(
                    f"default {pack.pack} has unresolved pack dependencies: {sorted(missing)}"
                )
            graph[pack.pack] = dependencies
            archives[pack.pack] = archive
        order = tuple(TopologicalSorter(graph).static_order())
        database = databases.Database("sqlite:///" + str(work / "registry.db"))
        try:
            identity = principals.PrincipalRepository(database)
            identity.register_user("engine-ci")
            identity.register_publisher("dinkster", "engine-ci", datetime.now(UTC).isoformat())
            service = admission.AdmissionService(
                database,
                objects.LocalFilesystemObjectStore(work / "objects"),
                settings=settings.ServerSettings(),
            )
            results = []
            for name in order:
                pack = entries[name]
                candidate = service.submit(
                    archives[name].read_bytes(),
                    pack_name=name,
                    version=pack.version,
                    publisher="dinkster",
                    submitted_by="engine-ci",
                    declared_digest=pack.artifact_digest,
                )
                for _ in range(16):
                    status = service.status(candidate)
                    if status.status in {"accepted", "rejected", "review"}:
                        break
                    if not service.run_once():
                        break
                status = service.status(candidate)
                if status.status != "accepted":
                    findings = "; ".join(
                        f"{finding.code}: {finding.message}" for finding in status.findings
                    )
                    raise ValueError(
                        f"default {name}@{pack.version} admission {status.status}: "
                        f"{findings or status.reason or 'scanner did not complete'}"
                    )
                results.append(
                    {"pack": name, "version": pack.version, "artifactDigest": pack.artifact_digest}
                )
            return {"engineVersion": engine_version, "packs": results}
        finally:
            database.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scanner", type=Path, required=True)
    parser.add_argument("--registry", default="https://registry.dinkster.org")
    parser.add_argument("--engine-version", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            check_defaults(
                NamedRegistry("", args.registry), args.scanner, engine_version=args.engine_version
            )
        )
    )
