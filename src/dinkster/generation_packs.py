"""Admit and select pack sources belonging to one explicit engine snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from dinkster_registry import (
    ArtifactError,
    InstallError,
    LockedPack,
    Lockfile,
    build_artifact,
    unpack_artifact,
)
from dinkster_schema import TypeExpr, canonical_claim, canonical_name
from dinkster_security.documents import Release as SecurityRelease
from dinkster_security.host import SecurityHost
from dinkster_security.inventory import generation_dependencies
from dinkster_security.policy import Artifact as SecurityArtifact
from dinkster_security.policy import Dependency as SecurityDependency
from dinkster_store import pack_archive_path
from dinkster_workers import ManifestError, PackManifest, load_manifest
from dinkster_workers.accelerator import AcceleratorError, resolve_accelerator
from dinkster_workers.catalog import read_catalog
from dinkster_workers.pack_environment import provision_pack, venv_python
from dinkster_workers.pack_lock import (
    GenerationPackages,
    PackLock,
    plan_inference,
    plan_pack,
    resolve_pack_lock,
)
from dinkster_workers.pack_wheel import pack_manifest_path
from dinkster_workers.provision import ProvisionError
from packaging.version import Version

from . import __version__ as engine_version
from .composition_error import CompositionError
from .registries import RegistryConfig

if TYPE_CHECKING:
    from .pack_spec import PackSpec


def _manifest(root: Path, entry: LockedPack) -> PackManifest:
    manifest = load_manifest(pack_manifest_path(root))
    if canonical_name(manifest.name) != entry.pack:
        raise CompositionError(f"snapshot pack {entry.pack!r} manifest names {manifest.name!r}")
    if {canonical_claim(claim) for claim in manifest.namespaces} != set(entry.claims):
        raise CompositionError(f"snapshot pack {entry.pack!r} manifest claims differ from lock")
    if (
        entry.publisher
        and manifest.publisher
        and canonical_name(manifest.publisher) != entry.publisher
    ):
        raise CompositionError(f"snapshot pack {entry.pack!r} publisher differs from lock")
    if manifest.compatibility is not None:
        try:
            manifest.compatibility.require_engine(entry.pack, engine_version)
        except ValueError as exc:
            raise CompositionError(str(exc)) from exc
    return manifest


def _validate_execution_locks(generation: Path, manifest: PackManifest) -> None:
    environments = generation / "packs"
    pack = canonical_name(manifest.name)
    paths = (
        sorted(
            path / "pack-lock.json"
            for path in environments.iterdir()
            if path.name == pack or path.name.startswith(pack + "~")
        )
        if environments.is_dir()
        else []
    )
    if not manifest.parts and not any(path.parent.name != pack for path in paths):
        return
    expected = {pack, *(f"{pack}~{part.part}" for part in manifest.parts)}
    actual = set()
    for path in paths:
        try:
            dependency = PackLock.read(path)
        except (OSError, ValueError, ProvisionError) as exc:
            raise CompositionError(
                f"{manifest.name}: cannot read execution lock {path}: {exc}"
            ) from exc
        if dependency.pack != pack or dependency.execution_id != path.parent.name:
            raise CompositionError(f"{manifest.name}: execution lock identity differs: {path}")
        actual.add(dependency.execution_id)
    if actual != expected:
        raise CompositionError(
            f"{manifest.name}: execution locks differ from declared archive parts; "
            f"missing {sorted(expected - actual)!r}; undeclared {sorted(actual - expected)!r}"
        )


def generation_pack_metadata(generation: Path) -> list[dict[str, object]]:
    """Read installed requests without importing pack code or provisioning workers."""
    from .packs import pack_info_from_manifest

    generation = generation.resolve()
    lock = Lockfile.from_record_json((generation / "dinkster.lock").read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for entry in lock.packs:
        row: dict[str, object] = {
            "pack": entry.pack,
            "version": entry.version,
            "displayName": entry.pack,
            "available": False,
            "networkOrigins": None,
            "graphicsCard": {"requested": None, "granted": None},
        }
        try:
            root = generation / "pack-sources" / entry.pack
            if not root.resolve().is_relative_to(generation / "pack-sources"):
                raise CompositionError(f"snapshot pack {entry.pack!r} source escapes snapshot")
            manifest = _manifest(root, entry)
            info = pack_info_from_manifest(manifest)
            row.update(
                displayName=info.display_name,
                available=True,
                networkOrigins=list(info.network_origins),
                graphicsCard={
                    "requested": any(part.sandbox.gpu for part in (manifest, *manifest.parts)),
                    "granted": False,
                },
            )
        except (OSError, ValueError, ManifestError, CompositionError) as exc:
            row["message"] = str(exc)
        rows.append(row)
    return rows


def stage_generation_packs(generation: Path, lock: Lockfile, artifacts: Mapping[str, Path]) -> None:
    """Admit verified sources after dependency rebuild, before the owner seals the target."""
    if any(
        (generation / name).exists() for name in ("build.json", "generation.json", "dinkster.lock")
    ):
        raise CompositionError("pack sources require a fresh, unsealed snapshot")
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
    if any(
        (generation / name).exists() for name in ("build.json", "generation.json", "dinkster.lock")
    ):
        raise CompositionError("pack locks require a fresh, unsealed snapshot")
    constraints = generation / "constraints.txt"
    if not constraints.is_file():
        raise CompositionError("custom pack locks require effective snapshot constraints")
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
            if entry.source.startswith("registry:"):
                registry = RegistryConfig().registry_for(entry)
                pack_indexes += (registry.endpoint + f"/static/{variant}/simple/",)
            for execution in (manifest, *manifest.parts):
                dependency_lock = resolve_pack_lock(
                    pack=manifest.name,
                    part=execution.part,
                    variant=variant,
                    requirements=(
                        execution.requires_for(accelerator) if accelerator else execution.requires
                    ),
                    python=Path(sys.executable),
                    constraints=constraints,
                    indexes=pack_indexes,
                    find_links=find_links,
                    engine=manifest.compatibility.engine if manifest.compatibility else None,
                    uv=uv,
                )
                path = output / f"{dependency_lock.execution_id}.pack-lock.json"
                dependency_lock.write(path)
                paths.append(path)
    return tuple(paths)


def load_generation_packs(
    generation: Path,
    *,
    selection: Mapping[str, object] | None = None,
    staging_root: Path | None = None,
) -> tuple[str, tuple[PackSpec, ...]]:
    """Read the selected variant and admitted sources from this generation only."""
    if not generation.is_absolute():
        raise CompositionError("--generation requires an absolute path")
    generation = generation.resolve()
    record = json.loads((generation / "build.json").read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("format") != "dinkster.generation/1":
        raise CompositionError("unsupported snapshot format")
    variant = record.get("variant")
    if not isinstance(variant, str) or not variant.strip():
        raise CompositionError("snapshot variant must be a non-empty string")
    if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", variant) is None:
        raise CompositionError("snapshot variant must be a portable variant name")
    hosting = record.get("inProcessPacks", [])
    if (
        not isinstance(hosting, list)
        or any(not isinstance(pack, str) for pack in hosting)
        or len(set(hosting)) != len(hosting)
    ):
        raise CompositionError("snapshot inProcessPacks must contain distinct pack ids")
    specs = generation_pack_specs(
        generation,
        tuple(hosting),
        local_build=record.get("source") == "local",
        selection=selection,
        staging_root=staging_root,
    )
    return variant, specs


def generation_pack_specs(
    generation: Path,
    in_process_packs: tuple[str, ...] = (),
    *,
    local_build: bool = False,
    selection: Mapping[str, object] | None = None,
    staging_root: Path | None = None,
) -> tuple[PackSpec, ...]:
    """Select verified admitted sources and their already-provisioned interpreters."""
    if not generation.is_absolute():
        raise CompositionError("--generation requires an absolute path")
    generation = generation.resolve()
    lock = Lockfile.from_record_json((generation / "dinkster.lock").read_text(encoding="utf-8"))
    if len(set(in_process_packs)) != len(in_process_packs):
        raise CompositionError("snapshot inProcessPacks must contain distinct pack ids")
    if set(in_process_packs) - {entry.pack for entry in lock.packs}:
        raise CompositionError("snapshot hosting names packs absent from dinkster.lock")
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
    provenance = generation / "artifact-sources.json"
    if provenance.is_file():
        selected_sources.update(
            row["source"]
            for name, row in json.loads(provenance.read_text()).get("packs", {}).items()
            if any(
                entry.pack == name
                and entry.source == row["source"]
                and entry.artifact_digest == row["artifactDigest"]
                for entry in lock.packs
            )
        )
    specs: list[PackSpec] = []
    manifests: dict[str, PackManifest] = {}
    for entry in lock.packs:
        from .pack_spec import PackSpec
        from .packs import pack_info_from_manifest

        root = generation / "pack-sources" / entry.pack
        if not root.resolve().is_relative_to(generation / "pack-sources"):
            raise CompositionError(f"snapshot pack {entry.pack!r} source escapes snapshot")
        manifest = _manifest(root, entry)
        with tempfile.TemporaryDirectory(
            prefix="dinkster-generation-verify-", dir=staging_root
        ) as directory:
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
                        raise CompositionError(f"snapshot pack {entry.pack!r} contains a symlink")
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
                f"snapshot pack {entry.pack!r} artifact digest differs from lock"
            )
        _validate_execution_locks(generation, manifest)
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
            raise CompositionError(f"snapshot pack {entry.pack!r} interpreter is missing: {python}")
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
                        f"snapshot pack {name!r} requires {dependency.pack!r} "
                        f"{dependency.version}, but the snapshot has no matching pack"
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
        record = generation / "build.json"
        if record.is_file():
            files = json.loads(record.read_text(encoding="utf-8"))["files"]
        else:
            # Catalog preparation precedes sealing. Inspect the provisioner's
            # current receipts; sealed generations must retain their recorded hashes.
            receipts = [
                "installed-wheels.json",
                *(
                    f"packs/{entry.pack}"
                    f"{'~' + part.part if part.part is not None else ''}/installed-wheels.json"
                    for entry in lock.packs
                    for part in (manifests[entry.pack], *manifests[entry.pack].parts)
                ),
            ]
            process = generation / "inference-process.json"
            if process.is_file():
                receipts.append(process.name)
                if json.loads(process.read_bytes()).get("format") == "dinkster.inference-process/2":
                    receipts.append("inference/shared-inference/installed-wheels.json")
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
    expanded: list[PackSpec] = []
    for spec in specs:
        manifest = load_manifest(spec.manifest, part=spec.part)
        for execution in (manifest, *manifest.parts):
            python = venv_python(generation / "packs" / execution.execution_id)
            if not spec.in_process and not python.is_file():
                raise CompositionError(
                    f"snapshot part {execution.execution_id!r} interpreter is missing: {python}"
                )
            expanded.append(
                replace(
                    spec,
                    part=execution.part,
                    python=None if spec.in_process else str(python),
                )
            )
    grouped = _group_first_party_packs(generation, expanded.copy())
    if selection is None:
        return grouped
    from dinkster_supervisor.environment_packs import read_pack_selection

    policy = read_pack_selection(dict(selection))
    allowed = policy["allow"]
    selected = set(entries) if allowed is None else set(allowed)
    unknown = selected - set(entries)
    if unknown:
        raise CompositionError(
            f"launch selection names packs absent from snapshot: {sorted(unknown)}"
        )
    if policy["excludePartners"]:
        selected.difference_update(
            name
            for name, manifest in manifests.items()
            if manifest.node_source == "remote-gateway-catalog"
        )
    # Validate the retained topology in full, then project it without leaving
    # excluded manifests or their import paths in a shared worker's launch.
    retained = [
        replace(spec, node_unavailability=grouped[index].node_unavailability)
        for index, spec in enumerate(expanded)
        if load_manifest(spec.manifest, part=spec.part).name in selected
    ]
    members = tuple(
        index
        for index, spec in enumerate(retained)
        if any(
            row.worker_group is not None and row.manifest == spec.manifest and row.part == spec.part
            for row in grouped
        )
    )
    if members:
        shared = next(row for row in grouped if row.worker_group is not None)
        return _group_members(retained, members, shared.python)
    return tuple(retained)


def _resident_nodes(specs: Sequence[PackSpec]) -> dict[str, tuple[str, ...]]:
    catalogs = {
        manifest.execution_id: catalog
        for spec in specs
        if (catalog := read_catalog(manifest := load_manifest(spec.manifest, part=spec.part)))
        is not None
    }
    resident_types = frozenset(
        tid for catalog in catalogs.values() for tid in catalog.resident_types
    )

    def resident(expr: TypeExpr) -> bool:
        if expr.kind == "asset":
            return False
        return (
            bool(resident_types.intersection(expr.types))
            or (expr.element is not None and resident(expr.element))
            or any(resident(element) for element in expr.elements)
        )

    return {
        name: tuple(
            node_type
            for node_type, schema in catalog.schemas.items()
            if any(resident(output.type) for output in schema.outputs)
            or any(not item.lazy and resident(item.type) for item in schema.inputs)
        )
        for name, catalog in catalogs.items()
    }


def prepare_inference_process(
    generation: Path,
    *,
    in_process_packs: tuple[str, ...] = (),
    isolated_inference_packs: tuple[str, ...] = (),
    local_build: bool = False,
    source_generation: Path | None = None,
    wheel_store: Path | None = None,
    uv_cache: Path | None = None,
    uv: str = "uv",
    offline: bool = True,
) -> None:
    """Record process membership while the target is inactive and unsealed."""
    if any((generation / name).exists() for name in ("build.json", "generation.json")):
        raise CompositionError("inference preparation requires an unsealed snapshot")
    specs = list(generation_pack_specs(generation, in_process_packs, local_build=local_build))
    for spec in specs:
        manifest = load_manifest(spec.manifest, part=spec.part)
        if not spec.trust_reserved and read_catalog(manifest) is None:
            if spec.require_catalog:
                raise CompositionError(
                    f"{manifest.execution_id}: admission requires a current schema catalog"
                )
            from dinkster_workers.doctor import prepare_catalog

            report = prepare_catalog(
                spec.manifest, part=spec.part, interpreter=spec.python, environment=spec.env
            )
            if not report.ok:
                raise CompositionError(
                    f"{manifest.execution_id}: admission catalog preparation failed: "
                    f"{report.findings}"
                )
    resident_nodes = _resident_nodes(specs)
    grouped = _group_first_party_packs(
        generation,
        list(specs),
        isolated=isolated_inference_packs,
        resolve_dependencies=wheel_store is not None,
    )
    members = [
        load_manifest(spec.manifest, part=spec.part).execution_id
        for spec in grouped
        if spec.worker_group is not None
    ]
    refusals: dict[str, str] = {}
    selected_plan = None
    if wheel_store is not None:
        parent_python = generation / (
            "python/python.exe" if os.name == "nt" else "python/bin/python3"
        )
        candidates: dict[str, list[tuple[PackSpec, PackManifest]]] = {}
        for spec in specs:
            manifest = load_manifest(spec.manifest, part=spec.part)
            if manifest.execution_id not in members and resident_nodes.get(manifest.execution_id):
                candidates.setdefault(manifest.name, []).append((spec, manifest))
        for pack, selections in candidates.items():
            reason = None
            names = [manifest.execution_id for _, manifest in selections]
            for spec, manifest in selections:
                if pack in isolated_inference_packs:
                    reason = "kept isolated by the per-pack inference switch"
                elif spec.in_process:
                    reason = "selected for in-process hosting, not the shared inference process"
                elif spec.asset_vault_write:
                    reason = "requests asset-vault write authority"
                elif manifest.sandbox.network:
                    reason = "requests pack.sandbox.network"
                elif manifest.sandbox.writable_mounts:
                    reason = "requests pack.sandbox.writable-mounts"
                if reason is not None:
                    reason = f"{manifest.execution_id!r}: {reason}"
                    break
            if reason is None:
                locks = tuple(
                    PackLock.read(generation / "packs" / member / "pack-lock.json")
                    for member in (*members, *names)
                )
                try:
                    candidate_plan = plan_inference(
                        locks,
                        python=parent_python,
                        variant=locks[0].variant,
                        find_links=tuple(
                            sorted({path.parent for path in wheel_store.rglob("*.whl")})
                        ),
                        wheel_store=wheel_store,
                        offline=offline,
                        uv=uv,
                    )
                    candidate_plan.require_layering()
                except ProvisionError as exc:
                    requirements = "; ".join(
                        f"{lock.execution_id!r}: {', '.join(lock.requirements) or '(none)'}"
                        for lock in locks
                        if lock.pack == pack
                    )
                    reason = f"{requirements}; {exc}"
                else:
                    selected_plan = candidate_plan
                    members.extend(names)
            if reason is not None:
                for name in names:
                    refusals[name] = f"Pack {name!r} is not admitted to shared inference: {reason}"
    python = next((spec.python for spec in grouped if spec.worker_group is not None), None)
    environment = next(
        (
            load_manifest(spec.manifest, part=spec.part).execution_id
            for spec in specs
            if python is not None and spec.python == python
        ),
        None,
    )
    format = "dinkster.inference-process/1"
    parent_versions: dict[str, str] = {}
    if members and wheel_store is not None:
        if uv_cache is None:
            raise CompositionError("shared environment preparation requires a uv cache")
        python = generation / ("python/python.exe" if os.name == "nt" else "python/bin/python3")
        facts = GenerationPackages.inspect(python)
        parent_versions = {
            name: facts.versions[name]
            for name in ("torch", "dinkster-inference", "comfy-aimdo", "dinkster-api")
            if name in facts.versions
        }
        locks = tuple(
            PackLock.read(generation / "packs" / name / "pack-lock.json") for name in members
        )
        variant = locks[0].variant
        previous = source_generation / "inference-process.json" if source_generation else None
        previous_record = (
            json.loads(previous.read_text(encoding="ascii"))
            if previous is not None and previous.is_file()
            else {}
        )
        retained = None
        if (
            source_generation is not None
            and previous_record.get("format") == "dinkster.inference-process/2"
            and previous_record.get("members") == members
            and all(
                (source_generation / "packs" / name / "pack-lock.json").read_bytes()
                == (generation / "packs" / name / "pack-lock.json").read_bytes()
                for name in members
            )
        ):
            retained = PackLock.read(
                source_generation / "inference/shared-inference/pack-lock.json"
            )
            if retained.pack != "shared-inference":
                raise CompositionError("shared dependency lock must name shared-inference")
            api = facts.versions.get("dinkster-api")
            if retained.engine != (f"=={api}" if api is not None else None) or any(
                f"{name}=={facts.versions.get(name)}" not in retained.requirements
                for name in ("torch", "dinkster-inference", "comfy-aimdo")
            ):
                retained = None
        if retained is not None:
            plan = plan_pack(
                retained,
                facts,
                variant=variant,
                own_environment_size=0,
                wheel_store=wheel_store,
                offline=offline,
                verify_dependencies=True,
                overrides=tuple(
                    package.name
                    for package in retained.packages
                    if package.name in facts.versions
                    and Version(package.version) != Version(facts.versions[package.name])
                ),
            )
            available = replace(
                facts,
                versions={**facts.versions, **{p.name: p.version for p in retained.packages}},
            )
            for lock in locks:
                plan_pack(
                    lock, available, variant=variant, own_environment_size=0
                ).require_layering()
        elif selected_plan is not None:
            plan = selected_plan
        else:
            plan = plan_inference(
                locks,
                python=python,
                variant=variant,
                find_links=tuple(sorted({path.parent for path in wheel_store.rglob("*.whl")})),
                wheel_store=wheel_store,
                offline=offline,
                uv=uv,
            )
        origins = {
            wheel.sha256: wheel.registry
            for lock in locks
            for package in lock.packages
            for wheel in package.wheels
            if wheel.registry is not None
        }
        receipt = generation / "installed-wheels.json"
        if receipt.is_file():
            origins.update(
                {
                    row["sha256"]: row["registry"]
                    for row in json.loads(receipt.read_text())["wheels"]
                }
            )
        updated = replace(
            plan.lock,
            packages=tuple(
                replace(
                    package,
                    wheels=tuple(
                        replace(wheel, registry=origins.get(wheel.sha256, wheel.registry))
                        for wheel in package.wheels
                    ),
                )
                for package in plan.lock.packages
            ),
        )
        plan = replace(
            plan,
            lock=updated,
            wheels=tuple(
                replace(wheel, registry=origins.get(wheel.sha256, wheel.registry))
                for wheel in plan.wheels
            ),
        )
        provision_pack(
            plan,
            generation_python=python,
            packs=generation / "inference",
            wheel_store=wheel_store,
            uv_cache=uv_cache,
            variant=variant,
            offline=offline,
            uv=uv,
        )
        environment = "inference/shared-inference"
        format = "dinkster.inference-process/2"
    (generation / "inference-process.json").write_text(
        json.dumps(
            {
                "format": format,
                "members": members,
                "environment": environment,
                **({"parentVersions": parent_versions} if parent_versions else {}),
                **(
                    {
                        "refusals": {
                            name: {"reason": reason, "nodes": resident_nodes[name]}
                            for name, reason in refusals.items()
                        }
                    }
                    if refusals
                    else {}
                ),
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )


def _group_first_party_packs(
    generation: Path,
    specs: list[PackSpec],
    *,
    isolated: tuple[str, ...] | None = None,
    resolve_dependencies: bool = False,
) -> tuple[PackSpec, ...]:
    recorded = generation / "inference-process.json"
    record = generation / "build.json"
    replay = isolated is None and recorded.is_file()
    if isolated is None:
        if not record.is_file():
            return tuple(specs)
        isolated = json.loads(record.read_text(encoding="utf-8")).get("isolatedInferencePacks", [])
    if (
        not isinstance(isolated, (list, tuple))
        or any(
            not isinstance(pack, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]*", pack) is None
            for pack in isolated
        )
        or len(set(isolated)) != len(isolated)
    ):
        raise CompositionError("snapshot isolatedInferencePacks must contain distinct pack ids")
    excluded = set(isolated)
    if replay:
        data = json.loads(recorded.read_text(encoding="utf-8"))
        names = {
            load_manifest(spec.manifest, part=spec.part).execution_id: index
            for index, spec in enumerate(specs)
        }
        members = data.get("members") if isinstance(data, dict) else None
        environment = data.get("environment") if isinstance(data, dict) else None
        if (
            not isinstance(data, dict)
            or data.get("format")
            not in {"dinkster.inference-process/1", "dinkster.inference-process/2"}
            or not isinstance(members, list)
            or any(not isinstance(name, str) or name not in names for name in members)
            or len(set(members)) != len(members)
            or (not members and environment is not None)
        ):
            raise CompositionError("invalid recorded inference process membership")
        refusals = data.get("refusals", {})
        if not isinstance(refusals, dict):
            raise CompositionError("invalid recorded inference refusals")
        for name, refusal in refusals.items():
            if (
                name not in names
                or name in members
                or not isinstance(refusal, dict)
                or not isinstance(refusal.get("reason"), str)
                or not refusal["reason"]
                or not isinstance(refusal.get("nodes"), list)
                or any(not isinstance(node, str) for node in refusal["nodes"])
            ):
                raise CompositionError("invalid recorded inference refusal")
            index = names[name]
            specs[index] = replace(
                specs[index],
                node_unavailability=dict.fromkeys(refusal["nodes"], refusal["reason"]),
            )
        admitted_packs = {name.partition("~")[0] for name in members}
        if admitted_packs.intersection(name.partition("~")[0] for name in refusals):
            raise CompositionError("recorded inference membership partially admits a refused pack")
        resident_nodes = _resident_nodes(specs)
        for name, index in names.items():
            manifest = load_manifest(specs[index].manifest, part=specs[index].part)
            if manifest.name in admitted_packs and resident_nodes.get(name) and name not in members:
                raise CompositionError(
                    f"recorded inference membership partially admits pack {manifest.name!r}: "
                    f"missing part {name!r}"
                )
        if not members:
            return tuple(specs)
        dedicated = data["format"] == "dinkster.inference-process/2"
        if dedicated:
            if environment != "inference/shared-inference":
                raise CompositionError("invalid recorded shared inference environment")
            interpreter = venv_python(generation / environment)
            python = str(interpreter)
            if not interpreter.is_file():
                raise CompositionError("recorded shared inference interpreter is missing")
        elif not isinstance(environment, str) or environment not in members:
            raise CompositionError("recorded inference environment must name a member")
        else:
            python = specs[names[environment]].python
        indices = tuple(names[name] for name in members)
        for index in indices:
            spec = specs[index]
            manifest = load_manifest(spec.manifest, part=spec.part)
            if (
                spec.in_process
                or manifest.name in excluded
                or spec.asset_vault_write
                or manifest.sandbox.network
                or manifest.sandbox.writable_mounts
            ):
                raise CompositionError(
                    f"recorded inference member {manifest.execution_id!r} requires isolated hosting"
                )
        return _group_members(specs, indices, python)
    # A shared worker must use one real, already-provisioned environment,
    # not a PYTHONPATH overlay of incompatible private site-packages.
    candidates = [
        index
        for index, spec in enumerate(specs)
        if spec.trust_reserved
        and spec.part is None
        and not load_manifest(spec.manifest).parts
        and not spec.in_process
        and not spec.asset_vault_write
        and load_manifest(spec.manifest, part=spec.part).name not in excluded
        and not load_manifest(spec.manifest, part=spec.part).sandbox.network
        and not load_manifest(spec.manifest, part=spec.part).sandbox.writable_mounts
    ]
    if resolve_dependencies and candidates:
        return _group_members(specs, tuple(candidates), specs[candidates[0]].python)
    if len(candidates) < 2:
        return tuple(specs)
    dependencies = {
        index: {
            package.name: (package.version, frozenset(wheel.sha256 for wheel in package.wheels))
            for package in PackLock.read(
                generation / "packs" / load_manifest(specs[index].manifest).name / "pack-lock.json"
            ).packages
        }
        for index in candidates
    }
    compatible = {
        index: tuple(
            member
            for member in candidates
            if dependencies[member].items() <= dependencies[index].items()
        )
        for index in candidates
    }
    primary = max(candidates, key=lambda index: len(compatible[index]))
    members = compatible[primary]
    for index in candidates:
        if index not in members:
            logging.getLogger(__name__).warning(
                "First-party pack %s remains isolated: its locked dependencies are not "
                "provided by the shared worker environment %s",
                load_manifest(specs[index].manifest).name,
                specs[primary].python,
            )
    if len(members) < 2:
        return tuple(specs)
    return _group_members(specs, members, specs[primary].python)


def _group_members(
    specs: list[PackSpec], members: tuple[int, ...], python: str | None
) -> tuple[PackSpec, ...]:
    manifests = tuple(specs[index].manifest for index in members)
    require_catalog = all(specs[index].require_catalog for index in members)
    if not require_catalog:
        # Live members need a live group announcement, but captured members
        # still require their admitted catalogs before any group code runs.
        for index in members:
            if (
                specs[index].require_catalog
                and read_catalog(load_manifest(specs[index].manifest, part=specs[index].part))
                is None
            ):
                raise CompositionError(
                    f"{load_manifest(specs[index].manifest, part=specs[index].part).execution_id}: "
                    "schema catalog is missing "
                    "or stale; run dinkster-pack prepare-catalogs before serving"
                )
    paths = tuple(
        dict.fromkeys(
            path for index in members for path in specs[index].env["PYTHONPATH"].split(os.pathsep)
        )
    )
    artifacts = tuple(artifact for index in members for artifact in specs[index].security_artifacts)
    for index in members:
        specs[index] = replace(
            specs[index],
            python=python,
            env={"PYTHONPATH": os.pathsep.join(paths)},
            worker_group="dinkster-first-party",
            group_manifests=manifests,
            group_parts=tuple(specs[member].part for member in members),
            require_catalog=require_catalog,
            security_artifacts=artifacts,
        )
    return tuple(specs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, required=True, metavar="SNAPSHOT")
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
