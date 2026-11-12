"""Resolve registry defaults and run the registry scanner against this engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import threading
import time
import tomllib
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from graphlib import TopologicalSorter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dinkster_registry import LockedPack, Lockfile, unpack_artifact
from dinkster_registry.archive_admission import (
    ProbeSettings,
    StageResult,
    admission_timing,
    run_doctor_static,
    run_executable_probe,
    run_fast_security,
    run_ingest_integrity,
    run_locked_probe,
    run_manifest_admission,
)
from dinkster_store import pack_archive_path
from dinkster_workers import load_manifest
from dinkster_workers.pack_lock import LockedDistribution, LockedWheel, PackLock
from dinkster_workers.pack_wheel import pack_manifest_path
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

from dinkster.installer import InstallError, http_registry_fetcher
from dinkster.registries import (
    NamedRegistry,
    RegistryConfig,
    StaticPackIndex,
    default_pack_minimums,
    prepare_registry_sources,
    read_registry_pack_locks,
    registry_source_lock,
    resolve_release,
)
from scripts.build_official_packs import build, library_metadata, official_library
from scripts.prepare_admission_engine import read_runtime
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
    root_requirements: list[str] = []
    metadata: dict[str, tuple[str, ...]] = {}
    with tempfile.TemporaryDirectory(prefix="admission-library-sources-") as directory:
        for pack in catalog.packs:
            archive = pack_archive_path(artifacts, pack.artifact_digest)
            if pack.pack in libraries:
                metadata[pack.artifact_digest] = library_metadata(archive)[2]
            else:
                source = unpack_artifact(archive, Path(directory) / pack.pack, pack.artifact_digest)
                root_requirements.extend(
                    load_manifest(pack_manifest_path(source)).requires_for("cpu")
                )

    failure = ""

    def resolve(selected: dict[str, LockedPack]) -> dict[str, LockedPack] | None:
        nonlocal library_names, failure
        requirements: dict[str, set[str]] = {}
        extras = {name: set() for name in libraries}
        pending = list(libraries)
        examined: dict[str, frozenset[str]] = {}

        def require(values: tuple[str, ...] | list[str], active_extras: set[str]) -> None:
            nonlocal library_names
            for value in values:
                requirement = Requirement(value)
                if requirement.marker and not any(
                    requirement.marker.evaluate({"extra": extra}) for extra in {"", *active_extras}
                ):
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
                if requirement.url is not None:
                    raise ValueError(f"registry library {name} must use a version range")
                requirements.setdefault(name, set()).add(str(requirement.specifier))
                extras.setdefault(name, set()).update(requirement.extras)
                if name in selected and examined.get(name) != frozenset(extras[name]):
                    pending.append(name)

        require(root_requirements, set())
        for name in pending:
            if examined.get(name) == frozenset(extras[name]):
                continue
            examined[name] = frozenset(extras[name])
            require(metadata[selected[name].artifact_digest], extras[name])

        for name, specifiers in sorted(requirements.items()):
            ranges = ",".join(sorted(specifiers))
            if name in selected and not SpecifierSet(ranges).contains(
                selected[name].version, prereleases=True
            ):
                failure = f"registry library {name} has no compatible release for {ranges}"
                return None
        unresolved = sorted(requirements.keys() - selected.keys())
        if not unresolved:
            return selected
        name = unresolved[0]
        ranges = ",".join(sorted(requirements[name]))
        versions = sorted(
            (
                str(row["version"])
                for row in index.releases(registry, name, include_libraries=True)
                if row.get("library") is True
                and SpecifierSet(ranges).contains(str(row["version"]), prereleases=True)
            ),
            key=Version,
            reverse=True,
        )
        if not versions:
            failure = f"registry library {name} has no compatible release for {ranges}"
        for version in versions:
            entry = resolve_release(registry, name, version, index=index)
            if entry.artifact_digest not in metadata:
                archive = pack_archive_path(artifacts, entry.artifact_digest)
                archive.parent.mkdir(parents=True, exist_ok=True)
                archive.write_bytes(
                    http_registry_fetcher(registry.endpoint, token=registry.token(), index=index)(
                        entry
                    )
                )
                verified = selected_library_wheels(Lockfile.of([entry]), artifacts)
                if name not in verified:
                    raise ValueError(f"registry library {name} contains a node manifest")
                metadata[entry.artifact_digest] = library_metadata(archive)[2]
            if resolved := resolve({**selected, name: entry}):
                return resolved
        return None

    resolved = resolve(entries)
    if resolved is None:
        raise ValueError(failure)
    return Lockfile.of(tuple(resolved.values()))


def check_defaults(
    registry: NamedRegistry,
    *,
    engine_version: str,
    candidate_base: str | None = None,
    published_sources: Path | None = None,
    prepare_output: Path | None = None,
    prepared_sources: Path | None = None,
    test_packs: tuple[str, ...] = (),
    generation_python: Path | None = None,
    base_id: str | None = None,
    constraints: Path | None = None,
    variant: str = "linux-cpu",
    runtime: Path | None = None,
) -> dict[str, object]:
    """Run the public archive stages without service state or publication credentials."""
    root = Path(__file__).resolve().parents[1]
    if runtime is not None:
        if any(value is not None for value in (generation_python, base_id, constraints)):
            raise ValueError("select a verified runtime, not individual interpreter inputs")
        generation_python, base_id, constraints, variant = read_runtime(
            runtime.resolve(), root, engine_version
        )
    if (
        sum(value is not None for value in (candidate_base, published_sources, prepared_sources))
        > 1
    ):
        raise ValueError("select candidate or published sources, not both")
    if prepare_output is not None and prepared_sources is not None:
        raise ValueError("cannot prepare sources from a prepared selection")
    if any(value is not None for value in (generation_python, base_id, constraints)):
        if generation_python is None or base_id is None or constraints is None:
            raise ValueError("locked admission requires generation_python, base_id and constraints")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", base_id) is None:
            raise ValueError("locked admission requires an exact sha256 base identity")
        if not generation_python.is_file() or not constraints.is_file():
            raise ValueError("locked admission interpreter or constraints are missing")
    minimums = default_pack_minimums()
    with tempfile.TemporaryDirectory(prefix="default-pack-admission-") as directory:
        work = Path(directory)
        artifacts = work / "artifacts"
        artifacts.mkdir()
        selected = Lockfile.of(())
        locks: dict[str, PackLock] = {}
        engine_context: dict[str, str] = {"engineVersion": engine_version}
        if base_id is not None:
            engine_context.update(baseId=base_id, variant=variant)
        if prepared_sources is not None:
            metadata = json.loads((prepared_sources / "engine.json").read_text())
            head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip()
            if metadata != {"commit": head, **engine_context}:
                raise ValueError("prepared sources do not match the engine checkout and version")
            selected = Lockfile.from_record_json((prepared_sources / "catalog.lock").read_text())
            shutil.copytree(prepared_sources / "artifacts", artifacts, dirs_exist_ok=True)
            if generation_python is not None:
                locks = {
                    pack.pack: PackLock.read(prepared_sources / "pack-locks" / f"{pack.pack}.json")
                    for pack in selected.packs
                }
        elif candidate_base is not None:
            candidates = tuple(
                pack
                for pack in select_packs(root, before=candidate_base, version_changes_only=True)
                if pack in minimums
            )
            if candidates:
                output = work / "candidate"
                with admission_timing("defaults", "candidate-build"):
                    if generation_python is None:
                        build(root, output, packs=candidates)
                    else:
                        build(
                            root,
                            output,
                            packs=candidates,
                            variant=variant,
                            python=generation_python,
                            constraints=constraints,
                            indexes=("https://pypi.org/simple",),
                            find_links=(runtime.resolve() / "wheelhouse",)
                            if runtime is not None
                            else (),
                            accelerator="cpu",
                        )
                selected = Lockfile.from_record_json((output / "official-packs.lock").read_text())
                for pack in selected.packs:
                    shutil.copyfile(
                        output / f"{pack.pack.replace('-', '_')}-{pack.version}-py3-none-any.whl",
                        artifacts / f"{pack.artifact_digest.removeprefix('blake3:')}.zip",
                    )
                    if generation_python is not None:
                        locks[pack.pack] = PackLock.read(
                            output / f"{pack.pack}-{pack.version}-{variant}.pack-lock.json"
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
                        with admission_timing(pack.pack, "published-resolution"):
                            release = resolve_release(
                                registry,
                                pack.pack,
                                pack.version,
                                engine_version=None
                                if pack.pack in library_names
                                else engine_version,
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
                    release
                    if pack.pack in library_names
                    else registry_source_lock(release, archive, engine_version=engine_version)
                )
            selected = Lockfile.of(entries)
        if prepared_sources is not None:
            if missing := set(minimums) - {pack.pack for pack in selected.packs}:
                raise ValueError(f"prepared sources omit defaults: {sorted(missing)}")
            catalog = selected
        else:
            with admission_timing("defaults", "registry-resolution-download"):
                catalog, _ = prepare_registry_sources(
                    registry,
                    {
                        name: version
                        for name, version in minimums.items()
                        if selected.get(name) is None
                    },
                    artifacts,
                    engine_version=engine_version,
                )
            catalog = Lockfile.of((*catalog.packs, *selected.packs))
            catalog = prepare_library_sources(registry, catalog, artifacts, root)
        if not catalog.packs:
            raise ValueError("engine admission requires a nonempty default set")
        if generation_python is not None:
            assert base_id is not None
            if prepared_sources is None:
                with admission_timing("defaults", "published-dependency-locks"):
                    published_locks = read_registry_pack_locks(
                        RegistryConfig((registry,)),
                        Lockfile.of([pack for pack in catalog.packs if pack.pack not in locks]),
                        variant=variant,
                        base_id=base_id,
                        selected=registry,
                    )
                locks.update((lock.pack, lock) for lock in published_locks)
            for pack in catalog.packs:
                lock = locks[pack.pack]
                if lock.pack != pack.pack or lock.variant != variant:
                    raise ValueError(f"{pack.pack}: dependency lock identity differs")
                archive = pack_archive_path(artifacts, pack.artifact_digest)
                payload = archive.read_bytes()
                outer = next((item for item in lock.packages if item.name == pack.pack), None)
                if outer is None and prepared_sources is not None:
                    raise ValueError(f"{pack.pack}: prepared dependency lock omits the outer wheel")
                wheel = LockedWheel(
                    f"{pack.pack.replace('-', '_')}-{pack.version}-py3-none-any.whl",
                    len(payload),
                    hashlib.sha256(payload).hexdigest(),
                    archive.resolve().as_uri(),
                )
                if outer is not None and (
                    outer.version != pack.version
                    or not any(
                        (item.filename, item.size, item.sha256)
                        == (wheel.filename, wheel.size, wheel.sha256)
                        for item in outer.wheels
                    )
                ):
                    raise ValueError(f"{pack.pack}: locked outer wheel differs from the archive")
                locks[pack.pack] = replace(
                    lock,
                    packages=(
                        *(item for item in lock.packages if item.name != pack.pack),
                        LockedDistribution(pack.pack, pack.version, (wheel,)),
                    ),
                )
        if prepare_output is not None:
            prepare_output.mkdir()
            shutil.copytree(artifacts, prepare_output / "artifacts")
            (prepare_output / "catalog.lock").write_text(catalog.record_json())
            head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip()
            (prepare_output / "engine.json").write_text(
                json.dumps({"commit": head, **engine_context})
            )
            if generation_python is not None:
                lock_output = prepare_output / "pack-locks"
                lock_output.mkdir()
                for lock in locks.values():
                    lock.write(lock_output / f"{lock.pack}.json")
            libraries = selected_library_wheels(catalog, artifacts)
            names = [pack.pack for pack in catalog.packs if pack.pack not in libraries]
            heavy = ("dinkster-3d", "dinkster-image", "dinkster-video", "dinkster-audio")
            groups = [[name] for name in heavy if name in names]
            remaining = [name for name in names if name not in heavy]
            groups.extend(remaining[index::9] for index in range(9) if remaining[index::9])
            return {"engineVersion": engine_version, "groups": groups}
        if missing := set(test_packs) - {pack.pack for pack in catalog.packs}:
            raise ValueError(f"requested admission packs are not in the catalog: {sorted(missing)}")
        entries = {pack.pack: pack for pack in catalog.packs}
        libraries = selected_library_wheels(catalog, artifacts)
        if generation_python is not None and libraries:
            wheel_store = work / "wheels"
            wheel_store.mkdir()
            # Selected libraries may not be public yet. The provisioner still
            # verifies these bytes against each consumer's immutable lock.
            for name, archive in libraries.items():
                _, version, _ = library_metadata(archive)
                filename = f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
                shutil.copyfile(archive, wheel_store / filename)
        graph: dict[str, set[str]] = {}
        sources = {}
        import_roots: dict[str, Path] = {}
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
            with admission_timing(identity, "ingest-integrity"):
                _require_accepted(
                    run_ingest_integrity(data, len(data), pack.artifact_digest), identity
                )
            with admission_timing(identity, "unpack"):
                source = unpack_artifact(
                    archive, work / "sources" / pack.pack, pack.artifact_digest
                )
                import_roots[pack.pack] = source
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
                for value in manifest.requires_for("cpu"):
                    requirement = Requirement(value)
                    name = canonicalize_name(requirement.name)
                    if name in libraries and (
                        not requirement.marker or requirement.marker.evaluate()
                    ):
                        if not requirement.specifier.contains(
                            entries[name].version, prereleases=True
                        ):
                            raise ValueError(
                                f"default {identity} library does not satisfy {value!r}"
                            )
                        dependencies.add(name)
                graph[pack.pack] = dependencies
            sources[pack.pack] = (source, data)
        order = tuple(TopologicalSorter(graph).static_order())
        results = []
        with (
            library_probe_index(libraries) if generation_python is None else nullcontext()
        ) as settings:
            for name in order:
                if test_packs and name not in test_packs:
                    continue
                pack = entries[name]
                identity = f"{name}@{pack.version}"
                if name not in libraries:
                    source, data = sources[name]
                    with admission_timing(identity, "doctor-static"):
                        _require_accepted(run_doctor_static(source), identity)
                    with admission_timing(identity, "executable-probe"):
                        if generation_python is None:
                            probe = (
                                run_executable_probe(source, settings=settings)
                                if settings
                                else run_executable_probe(source)
                            )
                        else:
                            reachable = [name]
                            for dependency_name in reachable:
                                if dependency_name in sources:
                                    for dependency in load_manifest(
                                        sources[dependency_name][0] / "dinkster-pack.toml"
                                    ).dependencies:
                                        if dependency.pack not in reachable:
                                            reachable.append(dependency.pack)
                            probe = run_locked_probe(
                                source,
                                locks[name],
                                generation_python=generation_python,
                                packs=work / "packs",
                                wheel_store=work / "wheels",
                                uv_cache=work / "uv-cache",
                                variant=variant,
                                source_roots=tuple(import_roots[item] for item in reachable),
                            )
                        _require_accepted(probe, identity)
                    with admission_timing(identity, "fast-security"):
                        _require_accepted(run_fast_security(data), identity)
                results.append(
                    {"pack": name, "version": pack.version, "artifactDigest": pack.artifact_digest}
                )
        with admission_timing("defaults", "selected-tests"):
            if generation_python is None:
                raise ValueError("selected tests require the managed engine and dependency locks")
            run_selected_tests(
                root,
                catalog,
                artifacts,
                generation_python=generation_python,
                locks=locks,
                variant=variant,
                test_packs=test_packs,
            )
        return {"engineVersion": engine_version, "packs": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="https://registry.dinkster.org")
    parser.add_argument("--engine-version", required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--candidate-base")
    selection.add_argument("--published-sources", type=Path)
    selection.add_argument("--prepared-sources", type=Path)
    parser.add_argument("--prepare-output", type=Path)
    parser.add_argument("--pack", action="append", default=[])
    parser.add_argument("--generation-python", type=Path)
    parser.add_argument("--base-id")
    parser.add_argument("--constraints", type=Path)
    parser.add_argument("--variant", default="linux-cpu")
    parser.add_argument("--runtime", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            check_defaults(
                NamedRegistry("", args.registry),
                engine_version=args.engine_version,
                candidate_base=args.candidate_base,
                published_sources=args.published_sources,
                prepare_output=args.prepare_output,
                prepared_sources=args.prepared_sources,
                test_packs=tuple(args.pack),
                generation_python=args.generation_python,
                base_id=args.base_id,
                constraints=args.constraints,
                variant=args.variant,
                runtime=args.runtime,
            )
        )
    )
