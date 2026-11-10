"""Resolve registry defaults and run the registry scanner against this engine."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from graphlib import TopologicalSorter
from pathlib import Path

from dinkster_registry import Lockfile, unpack_artifact
from dinkster_registry.archive_admission import (
    StageResult,
    admission_timing,
    run_doctor_static,
    run_executable_probe,
    run_fast_security,
    run_ingest_integrity,
    run_manifest_admission,
)
from dinkster_store import pack_archive_path
from dinkster_workers import load_manifest
from dinkster_workers.pack_wheel import pack_manifest_path
from packaging.specifiers import SpecifierSet

from dinkster.installer import InstallError, http_registry_fetcher
from dinkster.registries import (
    NamedRegistry,
    default_pack_minimums,
    prepare_registry_sources,
    registry_source_lock,
    resolve_release,
)
from scripts.build_official_packs import build
from scripts.run_selected_pack_tests import run_selected_tests
from scripts.select_official_packs import select_packs


def _require_accepted(result: StageResult, identity: str) -> None:
    if result.blocked or result.review:
        findings = "; ".join(f"{finding.code}: {finding.message}" for finding in result.findings)
        raise ValueError(f"default {identity} admission refused: {findings}")


def check_defaults(
    registry: NamedRegistry,
    *,
    engine_version: str,
    candidate_base: str | None = None,
    published_sources: Path | None = None,
) -> dict[str, object]:
    """Run the public archive stages without service state or publication credentials."""
    if candidate_base is not None and published_sources is not None:
        raise ValueError("select candidate or published sources, not both")
    root = Path(__file__).resolve().parents[1]
    minimums = default_pack_minimums()
    with tempfile.TemporaryDirectory(prefix="default-pack-admission-") as directory:
        work = Path(directory)
        artifacts = work / "artifacts"
        artifacts.mkdir()
        selected = Lockfile.of(())
        if candidate_base is not None:
            candidates = tuple(
                pack
                for pack in select_packs(root, before=candidate_base, version_changes_only=True)
                if pack in minimums
            )
            if candidates:
                output = work / "candidate"
                with admission_timing("defaults", "candidate-build"):
                    build(root, output, packs=candidates)
                selected = Lockfile.from_record_json((output / "official-packs.lock").read_text())
                for pack in selected.packs:
                    shutil.copyfile(
                        output / f"{pack.pack.replace('-', '_')}-{pack.version}-py3-none-any.whl",
                        artifacts / f"{pack.artifact_digest.removeprefix('blake3:')}.zip",
                    )
        elif published_sources is not None:
            published = Lockfile.from_record_json(published_sources.read_text())
            head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip()
            if not published.packs or any(
                pack.source != f"git:Kosinkadink/Dinkster@{head}" for pack in published.packs
            ):
                raise ValueError("publication sources do not match the engine checkout")
            deadline = time.monotonic() + 600
            entries = []
            for pack in published.packs:
                if pack.pack not in minimums:
                    continue
                while True:
                    try:
                        with admission_timing(pack.pack, "published-resolution"):
                            release = resolve_release(
                                registry, pack.pack, pack.version, engine_version=engine_version
                            )
                        break
                    except InstallError:
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(min(10, max(0, deadline - time.monotonic())))
                if release.artifact_digest != pack.artifact_digest:
                    raise ValueError(f"published {pack.pack}@{pack.version} digest mismatch")
                archive = artifacts / f"{pack.artifact_digest.removeprefix('blake3:')}.zip"
                with admission_timing(pack.pack, "published-download"):
                    archive.write_bytes(http_registry_fetcher(registry.endpoint)(release))
                entries.append(
                    registry_source_lock(release, archive, engine_version=engine_version)
                )
            selected = Lockfile.of(entries)
        with admission_timing("defaults", "registry-resolution-download"):
            catalog, _ = prepare_registry_sources(
                registry,
                {name: version for name, version in minimums.items() if selected.get(name) is None},
                artifacts,
                engine_version=engine_version,
            )
        catalog = Lockfile.of((*catalog.packs, *selected.packs))
        if not catalog.packs:
            raise ValueError("engine admission requires a nonempty default set")
        entries = {pack.pack: pack for pack in catalog.packs}
        graph: dict[str, set[str]] = {}
        sources = {}
        for pack in catalog.packs:
            archive = pack_archive_path(artifacts, pack.artifact_digest)
            data = archive.read_bytes()
            identity = f"{pack.pack}@{pack.version}"
            print(f"selected {identity} {pack.artifact_digest} from {pack.source}", flush=True)
            with admission_timing(identity, "ingest-integrity"):
                _require_accepted(
                    run_ingest_integrity(data, len(data), pack.artifact_digest), identity
                )
            with admission_timing(identity, "unpack"):
                source = unpack_artifact(
                    archive, work / "sources" / pack.pack, pack.artifact_digest
                )
                source = pack_manifest_path(source).parent
            with admission_timing(identity, "manifest-admission"):
                _require_accepted(
                    run_manifest_admission(
                        source,
                        pack.pack,
                        pack.version,
                        publisher="dinkster",
                        engine_version=engine_version,
                    ),
                    identity,
                )
            with admission_timing(identity, "dependency-graph"):
                manifest = load_manifest(source / "dinkster-pack.toml")
                dependencies = {dependency.pack for dependency in manifest.dependencies}
                if missing := dependencies - entries.keys():
                    raise ValueError(
                        f"default {pack.pack} has unresolved pack dependencies: {sorted(missing)}"
                    )
                for dependency in manifest.dependencies:
                    dependency_pack = entries[dependency.pack]
                    if not SpecifierSet(dependency.version).contains(
                        dependency_pack.version, prereleases=True
                    ):
                        raise ValueError(
                            f"default {identity} dependency "
                            f"{dependency.pack}@{dependency_pack.version} "
                            f"does not satisfy {dependency.version!r}"
                        )
                graph[pack.pack] = dependencies
            sources[pack.pack] = (source, data)
        order = tuple(TopologicalSorter(graph).static_order())
        results = []
        for name in order:
            pack = entries[name]
            identity = f"{name}@{pack.version}"
            source, data = sources[name]
            with admission_timing(identity, "doctor-static"):
                _require_accepted(run_doctor_static(source), identity)
            with admission_timing(identity, "executable-probe"):
                _require_accepted(run_executable_probe(source), identity)
            with admission_timing(identity, "fast-security"):
                _require_accepted(run_fast_security(data), identity)
            results.append(
                {"pack": name, "version": pack.version, "artifactDigest": pack.artifact_digest}
            )
        with admission_timing("defaults", "selected-tests"):
            run_selected_tests(root, catalog, artifacts)
        return {"engineVersion": engine_version, "packs": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="https://registry.dinkster.org")
    parser.add_argument("--engine-version", required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--candidate-base")
    selection.add_argument("--published-sources", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            check_defaults(
                NamedRegistry("", args.registry),
                engine_version=args.engine_version,
                candidate_base=args.candidate_base,
                published_sources=args.published_sources,
            )
        )
    )
