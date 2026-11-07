"""Resolve registry defaults and run the registry scanner against this engine."""

from __future__ import annotations

import argparse
import json
import tempfile
from graphlib import TopologicalSorter
from pathlib import Path

from dinkster_registry import unpack_artifact
from dinkster_registry.archive_admission import (
    StageResult,
    run_doctor_static,
    run_executable_probe,
    run_fast_security,
    run_ingest_integrity,
    run_manifest_admission,
)
from dinkster_workers import load_manifest
from packaging.specifiers import SpecifierSet

from dinkster.registries import NamedRegistry, default_pack_minimums, prepare_registry_sources
from scripts.run_selected_pack_tests import run_selected_tests


def _require_accepted(result: StageResult, identity: str) -> None:
    if result.blocked or result.review:
        findings = "; ".join(f"{finding.code}: {finding.message}" for finding in result.findings)
        raise ValueError(f"default {identity} admission refused: {findings}")


def check_defaults(registry: NamedRegistry, *, engine_version: str) -> dict[str, object]:
    """Run the public archive stages without service state or publication credentials."""
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
        sources = {}
        for pack in catalog.packs:
            archive = artifacts / f"{pack.artifact_digest.removeprefix('blake3:')}.zip"
            data = archive.read_bytes()
            identity = f"{pack.pack}@{pack.version}"
            _require_accepted(run_ingest_integrity(data, len(data), pack.artifact_digest), identity)
            root = unpack_artifact(archive, work / "sources" / pack.pack, pack.artifact_digest)
            _require_accepted(
                run_manifest_admission(
                    root,
                    pack.pack,
                    pack.version,
                    publisher="dinkster",
                    engine_version=engine_version,
                ),
                identity,
            )
            manifest = load_manifest(root / "dinkster-pack.toml")
            dependencies = {dependency.pack for dependency in manifest.dependencies}
            if missing := dependencies - entries.keys():
                raise ValueError(
                    f"default {pack.pack} has unresolved pack dependencies: {sorted(missing)}"
                )
            for dependency in manifest.dependencies:
                selected = entries[dependency.pack]
                if not SpecifierSet(dependency.version).contains(
                    selected.version, prereleases=True
                ):
                    raise ValueError(
                        f"default {identity} dependency {dependency.pack}@{selected.version} "
                        f"does not satisfy {dependency.version!r}"
                    )
            graph[pack.pack] = dependencies
            sources[pack.pack] = (root, data)
        order = tuple(TopologicalSorter(graph).static_order())
        results = []
        for name in order:
            pack = entries[name]
            identity = f"{name}@{pack.version}"
            root, data = sources[name]
            _require_accepted(run_doctor_static(root), identity)
            _require_accepted(run_executable_probe(root), identity)
            _require_accepted(run_fast_security(data), identity)
            results.append(
                {"pack": name, "version": pack.version, "artifactDigest": pack.artifact_digest}
            )
        run_selected_tests(Path(__file__).resolve().parents[1], catalog, artifacts)
        return {"engineVersion": engine_version, "packs": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="https://registry.dinkster.org")
    parser.add_argument("--engine-version", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            check_defaults(NamedRegistry("", args.registry), engine_version=args.engine_version)
        )
    )
