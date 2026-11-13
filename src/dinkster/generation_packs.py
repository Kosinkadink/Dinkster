"""Admit and select pack sources belonging to one explicit engine generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from dinkster_registry import (
    ArtifactError,
    InstallError,
    LockedPack,
    Lockfile,
    build_artifact,
    unpack_artifact,
)
from dinkster_schema import canonical_claim, canonical_name
from dinkster_security.documents import Release as SecurityRelease
from dinkster_security.host import SecurityHost
from dinkster_security.inventory import generation_dependencies
from dinkster_security.policy import Artifact as SecurityArtifact
from dinkster_security.policy import Dependency as SecurityDependency
from dinkster_store import pack_archive_path
from dinkster_workers import ManifestError, PackManifest, load_manifest
from dinkster_workers.accelerator import AcceleratorError, resolve_accelerator
from dinkster_workers.pack_lock import resolve_pack_lock
from dinkster_workers.pack_wheel import pack_manifest_path
from dinkster_workers.provision import ProvisionError

from . import __version__ as engine_version
from .compose import CompositionError, PackSpec
from .packs import pack_info_from_manifest
from .registries import RegistryConfig


def _manifest(root: Path, entry: LockedPack) -> PackManifest:
    manifest = load_manifest(pack_manifest_path(root))
    if canonical_name(manifest.name) != entry.pack:
        raise CompositionError(f"generation pack {entry.pack!r} manifest names {manifest.name!r}")
    if tuple(canonical_claim(claim) for claim in manifest.namespaces) != entry.claims:
        raise CompositionError(f"generation pack {entry.pack!r} manifest claims differ from lock")
    if (
        entry.publisher
        and manifest.publisher
        and canonical_name(manifest.publisher) != entry.publisher
    ):
        raise CompositionError(f"generation pack {entry.pack!r} publisher differs from lock")
    if manifest.compatibility is not None:
        try:
            manifest.compatibility.require_engine(entry.pack, engine_version)
        except ValueError as exc:
            raise CompositionError(str(exc)) from exc
    return manifest


def stage_generation_packs(generation: Path, lock: Lockfile, artifacts: Mapping[str, Path]) -> None:
    """Admit verified sources after dependency rebuild, before the owner seals the target."""
    if (generation / "generation.json").exists() or (generation / "dinkster.lock").exists():
        raise CompositionError("pack sources require a fresh, unsealed generation")
    sources = generation / "pack-sources"
    sources.mkdir(exist_ok=False)
    for entry in lock.packs:
        root = sources / entry.pack
        unpack_artifact(artifacts[entry.pack], root, entry.artifact_digest)
        manifest = _manifest(root, entry)
        if manifest.root != root:
            shutil.copyfile(artifacts[entry.pack], sources / f"{entry.pack}.whl")
    (generation / "dinkster.lock").write_text(
        lock.record_json() + "\n", encoding="utf-8", newline="\n"
    )


def prepare_pack_locks(
    generation: Path,
    lock: Lockfile,
    artifacts: Mapping[str, Path],
    output: Path,
    *,
    variant: str,
    indexes: Sequence[str] = (),
    find_links: Sequence[Path] = (),
    accelerator: str | None = None,
    uv: str = "uv",
) -> tuple[Path, ...]:
    """Resolve selected archives against an unsealed custom generation's constraints."""
    if (generation / "generation.json").exists() or (generation / "dinkster.lock").exists():
        raise CompositionError("pack locks require a fresh, unsealed generation")
    constraints = generation / "constraints.txt"
    if not constraints.is_file():
        raise CompositionError("custom pack locks require effective generation constraints")
    output.mkdir(exist_ok=False)
    paths: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="dinkster-custom-pack-sources-") as directory:
        manifests = []
        for entry in lock.packs:
            source = unpack_artifact(
                artifacts[entry.pack], Path(directory) / entry.pack, entry.artifact_digest
            )
            manifests.append(_manifest(source, entry))
        for entry, manifest in zip(lock.packs, manifests, strict=True):
            pack_indexes = tuple(indexes)
            if entry.source == "registry" or entry.source.startswith("registry:"):
                registry = RegistryConfig().registry_for(entry)
                pack_indexes += (registry.endpoint + f"/static/{variant}/simple/",)
            dependency_lock = resolve_pack_lock(
                pack=manifest.name,
                variant=variant,
                requirements=(
                    manifest.requires_for(accelerator) if accelerator else manifest.requires
                ),
                python=Path(sys.executable),
                constraints=constraints,
                indexes=pack_indexes,
                find_links=find_links,
                engine=manifest.compatibility.engine if manifest.compatibility else None,
                uv=uv,
            )
            path = output / f"{manifest.name}.pack-lock.json"
            dependency_lock.write(path)
            paths.append(path)
    return tuple(paths)


def load_generation_packs(generation: Path) -> tuple[str, tuple[PackSpec, ...]]:
    """Read the selected variant and admitted sources from this generation only."""
    if not generation.is_absolute():
        raise CompositionError("--generation requires an absolute path")
    generation = generation.resolve()
    record = json.loads((generation / "generation.json").read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("format") != "dinkster.generation/1":
        raise CompositionError("unsupported generation format")
    variant = record.get("variant")
    if not isinstance(variant, str) or not variant.strip():
        raise CompositionError("generation variant must be a non-empty string")
    if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", variant) is None:
        raise CompositionError("generation variant must be a portable variant name")
    hosting = record.get("inProcessPacks", [])
    if (
        not isinstance(hosting, list)
        or any(not isinstance(pack, str) for pack in hosting)
        or len(set(hosting)) != len(hosting)
    ):
        raise CompositionError("generation inProcessPacks must contain distinct pack ids")
    specs = generation_pack_specs(
        generation, tuple(hosting), local_build=record.get("source") == "local"
    )
    return variant, specs


def generation_pack_specs(
    generation: Path, in_process_packs: tuple[str, ...] = (), *, local_build: bool = False
) -> tuple[PackSpec, ...]:
    """Select verified admitted sources and their already-provisioned interpreters."""
    if not generation.is_absolute():
        raise CompositionError("--generation requires an absolute path")
    generation = generation.resolve()
    lock = Lockfile.from_record_json((generation / "dinkster.lock").read_text(encoding="utf-8"))
    if len(set(in_process_packs)) != len(in_process_packs):
        raise CompositionError("generation inProcessPacks must contain distinct pack ids")
    if set(in_process_packs) - {entry.pack for entry in lock.packs}:
        raise CompositionError("generation hosting names packs absent from dinkster.lock")
    development = generation / "development-sources.json"
    selected_sources: set[str] = set()
    live_sources: dict[str, Path] = {}
    if development.is_file():
        capture = json.loads(development.read_text(encoding="utf-8"))
        if capture.get("format") != "dinkster.development-sources/1":
            raise CompositionError("unsupported development source capture")
        selected_sources = {
            "local-build:" + row["path"] for row in capture["sources"] if row["kind"] == "pack"
        }
        live_sources = {
            "local-build:" + row["path"]: Path(row["path"])
            for row in capture["sources"]
            if row["kind"] == "pack" and row.get("live")
        }
    specs: list[PackSpec] = []
    manifests: dict[str, PackManifest] = {}
    for entry in lock.packs:
        root = generation / "pack-sources" / entry.pack
        if not root.resolve().is_relative_to(generation / "pack-sources"):
            raise CompositionError(f"generation pack {entry.pack!r} source escapes generation")
        manifest = _manifest(root, entry)
        with tempfile.TemporaryDirectory(prefix="dinkster-generation-verify-") as directory:
            if manifest.root == root:
                matches = (
                    build_artifact(root, Path(directory) / "pack.zip") == entry.artifact_digest
                )
            else:
                # Repacking changes compression and ZIP metadata, not the admitted content.
                admitted = unpack_artifact(
                    root.parent / f"{entry.pack}.whl",
                    Path(directory) / "admitted",
                    entry.artifact_digest,
                )
                files = {}
                for tree in (root, admitted):
                    paths = tuple(tree.rglob("*"))
                    if any(path.is_symlink() for path in paths):
                        raise CompositionError(f"generation pack {entry.pack!r} contains a symlink")
                    files[tree] = {
                        path.relative_to(tree)
                        for path in paths
                        if path.is_file() and "__pycache__" not in path.relative_to(tree).parts
                    }
                matches = files[root] == files[admitted] and all(
                    (root / path).read_bytes() == (admitted / path).read_bytes()
                    for path in files[admitted]
                )
        if not matches:
            raise CompositionError(
                f"generation pack {entry.pack!r} artifact digest differs from lock"
            )
        live_root = live_sources.get(entry.source)
        if live_root is not None:
            if entry.pack in in_process_packs:
                raise CompositionError("live packs require isolated hosting")
            if not live_root.is_absolute() or not live_root.is_dir() or live_root.is_symlink():
                raise CompositionError(f"live pack folder is unavailable: {live_root}")
            root = live_root
            manifest = _manifest(root, entry)
        manifests[entry.pack] = manifest
        in_process = entry.pack in in_process_packs
        python = (
            generation
            / "packs"
            / entry.pack
            / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        )
        if not in_process and not python.is_file():
            raise CompositionError(
                f"generation pack {entry.pack!r} interpreter is missing: {python}"
            )
        info = replace(
            pack_info_from_manifest(manifest),
            version=entry.version,
            artifact_digest=entry.artifact_digest,
            source=entry.source,
            publisher=entry.publisher,
        )
        specs.append(
            PackSpec(
                manifest=manifest.path,
                python=None if in_process else str(python),
                env={"PYTHONPATH": str(root)},
                packs={entry.pack: info},
                in_process=in_process,
                trust_reserved=entry.publisher == "dinkster"
                and (
                    entry.source == "registry"
                    or entry.source.startswith("registry:")
                    or entry.source in selected_sources
                    or (
                        local_build
                        and re.fullmatch(
                            r"local-build:git:Kosinkadink/Dinkster@[0-9a-f]{40}", entry.source
                        )
                        is not None
                    )
                ),
                require_catalog=live_root is None,
                asset_vault_write=(
                    entry.pack == "dinkster-nodes-remote"
                    and entry.publisher == "dinkster"
                    and not in_process
                ),
            )
        )
    entries = {entry.pack: entry for entry in lock.packs}
    for index, entry in enumerate(lock.packs):
        reachable = [entry.pack]
        for name in reachable:
            for dependency in manifests[name].dependencies:
                provider = entries.get(dependency.pack)
                if provider is None or not dependency.accepts(provider.version):
                    raise CompositionError(
                        f"generation pack {name!r} requires {dependency.pack!r} "
                        f"{dependency.version}, but the generation has no matching pack"
                    )
                if dependency.pack not in reachable:
                    reachable.append(dependency.pack)
        specs[index] = replace(
            specs[index],
            env={
                "PYTHONPATH": os.pathsep.join(
                    str(live_sources.get(entries[name].source, generation / "pack-sources" / name))
                    for name in reachable
                )
            },
        )
    project = generation.parent.parent
    if (project / "security.sqlite3").is_file() or (project / "security.toml").is_file():
        security = SecurityHost(project)
        security.refresh()
        record = generation / "generation.json"
        if record.is_file():
            files = json.loads(record.read_text(encoding="utf-8"))["files"]
        else:
            # Catalog preparation precedes sealing. Inspect the provisioner's
            # current receipts; sealed generations must retain their recorded hashes.
            receipts = [
                "installed-wheels.json",
                *(f"packs/{entry.pack}/installed-wheels.json" for entry in lock.packs),
            ]
            files = {
                name: "sha256:" + hashlib.sha256((generation / name).read_bytes()).hexdigest()
                for name in receipts
            }
        libraries = generation_dependencies(generation, files, tuple(entries), in_process_packs)

        def issuing_registry(entry: LockedPack) -> str:
            if entry.source.startswith(("registry:http://", "registry:https://")):
                return entry.source.removeprefix("registry:")
            if entry.source == "registry" or entry.source.startswith("registry:"):
                raise CompositionError("security inventory requires a resolved issuing registry")
            return entry.source

        artifacts: list[SecurityArtifact] = []
        for index, (entry, spec) in enumerate(zip(lock.packs, specs, strict=True)):
            dependencies = {
                SecurityDependency(
                    issuing_registry(provider),
                    provider.pack,
                    provider.version,
                    provider.artifact_digest,
                )
                for path in spec.env["PYTHONPATH"].split(os.pathsep)
                if Path(path).name != entry.pack
                for provider in (entries[Path(path).name],)
            }
            artifacts.append(
                SecurityArtifact(
                    issuing_registry(entry),
                    SecurityRelease.model_validate(
                        {
                            "pack": entry.pack,
                            "version": entry.version,
                            "artifactDigest": entry.artifact_digest,
                        }
                    ),
                    tuple(
                        sorted(
                            dependencies
                            | {
                                dependency
                                for path in spec.env["PYTHONPATH"].split(os.pathsep)
                                for dependency in libraries[Path(path).name]
                            },
                            key=lambda item: (item.registry, item.name, item.version, item.digest),
                        )
                    ),
                )
            )
            specs[index] = replace(spec, security_artifacts=(artifacts[-1],))
        result = security.assess(tuple(artifacts))
        if not result.allowed:
            if result.stale_registries:
                raise CompositionError("security feed exceeds configured maximum age")
            finding = next(item for item in result.findings if not item.allowed)
            raise CompositionError(
                f"security {finding.state}: {finding.artifact.release.pack}, "
                f"notice {finding.notice.id} revision {finding.notice.revision}, "
                f"decision {finding.decision}"
            )
    return tuple(specs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--catalog-lock", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument(
        "--prepare-locks", type=Path, help="new scratch output for custom pack locks"
    )
    parser.add_argument("--variant")
    parser.add_argument("--index", action="append", default=[])
    parser.add_argument("--find-links", type=Path, action="append", default=[])
    parser.add_argument("--accelerator")
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    packs: list[str] = []
    locks: tuple[Path, ...] = ()
    errors: list[str] = []
    try:
        if not all(
            path.is_absolute() for path in (args.generation, args.catalog_lock, args.artifacts)
        ):
            raise CompositionError(
                "--generation, --catalog-lock and --artifacts require absolute paths"
            )
        lock = Lockfile.from_record_json(args.catalog_lock.read_text(encoding="utf-8"))
        artifacts = {
            entry.pack: pack_archive_path(args.artifacts, entry.artifact_digest)
            for entry in lock.packs
        }
        if args.prepare_locks is not None:
            if not args.prepare_locks.is_absolute() or not args.variant:
                raise CompositionError("custom pack locks require absolute output and --variant")
            locks = prepare_pack_locks(
                args.generation,
                lock,
                artifacts,
                args.prepare_locks,
                variant=args.variant,
                indexes=args.index,
                find_links=args.find_links,
                accelerator=resolve_accelerator(args.accelerator) if args.accelerator else None,
                uv=args.uv,
            )
        else:
            stage_generation_packs(args.generation, lock, artifacts)
        packs = [entry.pack for entry in lock.packs]
    except (
        AcceleratorError,
        ArtifactError,
        CompositionError,
        InstallError,
        ManifestError,
        OSError,
        ProvisionError,
        ValueError,
        zipfile.BadZipFile,
    ) as exc:
        errors.append(str(exc))
        print(str(exc), file=sys.stderr)
    if args.json:
        result: dict[str, object] = {"packs": packs, "errors": errors}
        if args.prepare_locks is not None:
            result["locks"] = [str(path) for path in locks]
        print(json.dumps(result))
    elif packs:
        print("\n".join(packs))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
