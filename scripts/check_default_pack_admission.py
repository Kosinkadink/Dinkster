"""Resolve registry defaults and run the registry scanner against this engine."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import threading
import time
import tomllib
from contextlib import contextmanager
from graphlib import TopologicalSorter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dinkster_registry import Lockfile, unpack_artifact
from dinkster_registry.archive_admission import (
    ProbeSettings,
    StageResult,
    run_doctor_static,
    run_executable_probe,
    run_fast_security,
    run_ingest_integrity,
    run_manifest_admission,
)
from dinkster_store import pack_archive_path
from dinkster_workers import load_manifest
from dinkster_workers.pack_wheel import pack_manifest_path
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

from dinkster.installer import InstallError, http_registry_fetcher
from dinkster.registries import (
    NamedRegistry,
    StaticPackIndex,
    default_pack_minimums,
    prepare_registry_sources,
    registry_source_lock,
    resolve_release,
)
from scripts.build_official_packs import build, library_metadata, official_library
from scripts.run_selected_pack_tests import run_selected_tests, selected_library_wheels
from scripts.select_official_packs import select_packs


@contextmanager
def library_probe_index(libraries: dict[str, Path]):
    """Serve the selected wheels to the unchanged probe dependency installer."""
    if not libraries:
        yield None
        return
    wheels = {}
    for name, archive in libraries.items():
        _, version, _ = library_metadata(archive)
        filename = f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
        wheels[f"/{filename}"] = archive

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            name = self.path.strip("/")
            if archive := wheels.get(self.path):
                payload = archive.read_bytes()
                content_type = "application/octet-stream"
            elif name in libraries:
                _, version, _ = library_metadata(libraries[name])
                filename = f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
                payload = f'<a href="/{filename}">{filename}</a>'.encode()
                content_type = "text/html"
            else:
                self.send_response(302)
                self.send_header("Location", f"https://pypi.org/simple/{name}/")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)

        do_HEAD = do_GET

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield ProbeSettings(probe_package_index_url=f"http://127.0.0.1:{server.server_port}")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _require_accepted(result: StageResult, identity: str) -> None:
    if result.blocked or result.review:
        if result.operator_details:
            print(
                json.dumps({"pack": identity, "operatorDetails": result.operator_details}),
                flush=True,
            )
        findings = "; ".join(f"{finding.code}: {finding.message}" for finding in result.findings)
        raise ValueError(f"default {identity} admission refused: {findings}")


def prepare_library_sources(
    registry: NamedRegistry, catalog: Lockfile, artifacts: Path, root: Path
) -> Lockfile:
    """Close registry library requirements over verified selected archive bytes."""
    entries = {pack.pack: pack for pack in catalog.packs}
    libraries = selected_library_wheels(catalog, artifacts)
    known_libraries = {
        identity[0]
        for path in (root / "packages").glob("*/pyproject.toml")
        if (identity := official_library(tomllib.loads(path.read_text())))
    }
    index = StaticPackIndex()
    library_names: set[str] | None = None
    requirements: dict[str, list[Requirement]] = {}
    pending = list(catalog.packs)
    with tempfile.TemporaryDirectory(prefix="admission-library-sources-") as directory:
        for pack in pending:
            archive = pack_archive_path(artifacts, pack.artifact_digest)
            if pack.pack in libraries:
                _, _, requires = library_metadata(archive)
            else:
                source = unpack_artifact(archive, Path(directory) / pack.pack, pack.artifact_digest)
                requires = load_manifest(pack_manifest_path(source)).requires_for("cpu")
            for value in requires:
                requirement = Requirement(value)
                if requirement.marker and not requirement.marker.evaluate():
                    continue
                name = canonicalize_name(requirement.name)
                if name in entries and name not in libraries:
                    continue
                if name not in libraries:
                    if library_names is None:
                        library_names = known_libraries | {
                            str(row["name"])
                            for row in index.catalog(registry, include_libraries=True)
                            if row.get("library") is True
                        }
                    if name not in library_names:
                        continue
                requirements.setdefault(name, []).append(requirement)
                ranges = ",".join(str(item.specifier) for item in requirements[name])
                if requirement.url is not None:
                    raise ValueError(f"registry library {name} must use a version range")
                if name in entries:
                    if not SpecifierSet(ranges).contains(entries[name].version, prereleases=True):
                        raise ValueError(
                            f"registry library {name}@{entries[name].version} "
                            f"does not satisfy {ranges}"
                        )
                    continue
                versions = [
                    str(row["version"])
                    for row in index.releases(registry, name)
                    if row.get("library") is True
                    and SpecifierSet(ranges).contains(str(row["version"]), prereleases=True)
                ]
                if not versions:
                    raise ValueError(
                        f"registry library {name} has no compatible release for {ranges}"
                    )
                selected = resolve_release(registry, name, max(versions, key=Version), index=index)
                archive = pack_archive_path(artifacts, selected.artifact_digest)
                archive.parent.mkdir(parents=True, exist_ok=True)
                archive.write_bytes(
                    http_registry_fetcher(registry.endpoint, token=registry.token(), index=index)(
                        selected
                    )
                )
                verified = selected_library_wheels(Lockfile.of([selected]), artifacts)
                if name not in verified:
                    raise ValueError(f"registry library {name} contains a node manifest")
                libraries.update(verified)
                entries[name] = selected
                pending.append(selected)
    return Lockfile.of(entries.values())


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
                build(root, output, packs=candidates)
                selected = Lockfile.from_record_json((output / "official-packs.lock").read_text())
                for pack in selected.packs:
                    shutil.copyfile(
                        output / f"{pack.pack.replace('-', '_')}-{pack.version}-py3-none-any.whl",
                        artifacts / f"{pack.artifact_digest.removeprefix('blake3:')}.zip",
                    )
        elif published_sources is not None:
            published = Lockfile.from_record_json(published_sources.read_text())
            library_names = {
                identity[0]
                for path in (root / "packages").glob("*/pyproject.toml")
                if (identity := official_library(tomllib.loads(path.read_text())))
            }
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
                if pack.pack not in minimums and pack.pack not in library_names:
                    continue
                while True:
                    try:
                        release = resolve_release(
                            registry,
                            pack.pack,
                            pack.version,
                            engine_version=None if pack.pack in library_names else engine_version,
                        )
                        break
                    except InstallError:
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(min(10, max(0, deadline - time.monotonic())))
                if release.artifact_digest != pack.artifact_digest:
                    raise ValueError(f"published {pack.pack}@{pack.version} digest mismatch")
                archive = artifacts / f"{pack.artifact_digest.removeprefix('blake3:')}.zip"
                archive.write_bytes(http_registry_fetcher(registry.endpoint)(release))
                entries.append(
                    release
                    if pack.pack in library_names
                    else registry_source_lock(release, archive, engine_version=engine_version)
                )
            selected = Lockfile.of(entries)
        catalog, _ = prepare_registry_sources(
            registry,
            {name: version for name, version in minimums.items() if selected.get(name) is None},
            artifacts,
            engine_version=engine_version,
        )
        catalog = Lockfile.of((*catalog.packs, *selected.packs))
        catalog = prepare_library_sources(registry, catalog, artifacts, root)
        if not catalog.packs:
            raise ValueError("engine admission requires a nonempty default set")
        entries = {pack.pack: pack for pack in catalog.packs}
        libraries = selected_library_wheels(catalog, artifacts)
        graph: dict[str, set[str]] = {}
        sources = {}
        for pack in catalog.packs:
            archive = pack_archive_path(artifacts, pack.artifact_digest)
            data = archive.read_bytes()
            identity = f"{pack.pack}@{pack.version}"
            print(f"selected {identity} {pack.artifact_digest} from {pack.source}", flush=True)
            if pack.pack in libraries:
                _, _, requires = library_metadata(archive)
                graph[pack.pack] = {
                    canonicalize_name(Requirement(value).name)
                    for value in requires
                    if canonicalize_name(Requirement(value).name) in libraries
                }
                continue
            _require_accepted(run_ingest_integrity(data, len(data), pack.artifact_digest), identity)
            source = unpack_artifact(archive, work / "sources" / pack.pack, pack.artifact_digest)
            source = pack_manifest_path(source).parent
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
            for value in manifest.requires_for("cpu"):
                requirement = Requirement(value)
                name = canonicalize_name(requirement.name)
                if name in libraries and (not requirement.marker or requirement.marker.evaluate()):
                    if not requirement.specifier.contains(entries[name].version, prereleases=True):
                        raise ValueError(f"default {identity} library does not satisfy {value!r}")
                    dependencies.add(name)
            graph[pack.pack] = dependencies
            sources[pack.pack] = (source, data)
        order = tuple(TopologicalSorter(graph).static_order())
        results = []
        with library_probe_index(libraries) as settings:
            for name in order:
                pack = entries[name]
                identity = f"{name}@{pack.version}"
                if name not in libraries:
                    source, data = sources[name]
                    _require_accepted(run_doctor_static(source), identity)
                    probe = (
                        run_executable_probe(source, settings=settings)
                        if settings
                        else run_executable_probe(source)
                    )
                    _require_accepted(probe, identity)
                    _require_accepted(run_fast_security(data), identity)
                results.append(
                    {"pack": name, "version": pack.version, "artifactDigest": pack.artifact_digest}
                )
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
