"""Compute inert, compatible security repair offers from retained and published metadata."""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

from dinkster_registry import InstallError, LockedPack, Lockfile
from dinkster_security.documents import Release
from dinkster_security.host import SecurityHost
from dinkster_security.inventory import generation_dependencies
from dinkster_security.policy import Artifact, Dependency
from dinkster_workers import load_manifest
from dinkster_workers.manifest import PackDependency
from dinkster_workers.pack_lock import (
    GenerationPackages,
    PackLock,
    normalize_requirement,
    plan_pack,
)
from dinkster_workers.pack_wheel import pack_manifest_path
from dinkster_workers.provision import ProvisionError
from packaging.specifiers import SpecifierSet
from packaging.tags import Tag
from packaging.utils import parse_wheel_filename
from packaging.version import Version

from .registries import RegistryConfig, StaticPackIndex, plan_registry_changes


def plan_security_repair(
    generation: Path,
    pack: str,
    config: RegistryConfig,
    *,
    index: StaticPackIndex,
    security: SecurityHost,
) -> dict[str, object]:
    """Offer one validated change, never download payloads, import packs or apply it.

    The offer is input to a host-owned approval plan, not authorization to apply.
    Root interpreter dependencies cannot be repaired by changing a pack lock.
    """
    state = importlib.import_module("dinkster_supervisor.environment_state")
    wheel_compatibility = importlib.import_module(
        "dinkster_supervisor.environment_constraints"
    ).wheel_compatibility
    record = state.Generation.read(generation / "generation.json")
    software = "sha256:" + hashlib.sha256((generation / "generation.json").read_bytes()).hexdigest()
    files = dict(record.files)
    if record.drift(generation):
        raise InstallError("security repair requires intact sealed generation inputs")
    for name in files:
        path = generation / name
        if not path.resolve().is_relative_to(generation.resolve()) or any(
            parent.is_symlink() or parent.is_junction() for parent in (path, *path.parents)
        ):
            raise InstallError("security repair requires intact sealed generation inputs")
    installed = Lockfile.from_record_json((generation / "dinkster.lock").read_text())
    entries = {entry.pack: entry for entry in installed.packs}
    if pack not in entries:
        raise InstallError("security repair requires an installed pack")
    hosting = record.in_process_packs
    libraries = generation_dependencies(generation, files, tuple(entries), hosting)
    manifests = {
        name: load_manifest(pack_manifest_path(generation / "pack-sources" / name))
        for name in entries
    }
    constraints = {name: manifest.dependencies for name, manifest in manifests.items()}
    locks = {
        name: PackLock.read(generation / "packs" / name / "pack-lock.json") for name in entries
    }
    required = {"dinkster.lock", "installed-wheels.json"} | {
        path.relative_to(generation).as_posix()
        for name, manifest in manifests.items()
        for path in (manifest.path, generation / "packs" / name / "pack-lock.json")
    }
    if required - files.keys():
        raise InstallError("security repair requires sealed source and dependency constraints")
    root_rows = json.loads((generation / "installed-wheels.json").read_bytes())["wheels"]
    root_dependencies = tuple(
        Dependency(
            row["registry"],
            parse_wheel_filename(row["filename"])[0],
            str(parse_wheel_filename(row["filename"])[1]),
            "sha256:" + row["sha256"],
        )
        for row in root_rows
    )
    compatibility = wheel_compatibility(state.interpreter(generation))
    markers = cast("dict[str, str]", compatibility["markers"])
    parent = GenerationPackages(
        {item.name: item.version for item in root_dependencies},
        markers["python_full_version"],
        frozenset(Tag(*tag.split("-")) for tag in cast("list[str]", compatibility["tags"])),
        markers,
    )

    def issuer(entry: LockedPack) -> str:
        if entry.source.startswith(("registry:http://", "registry:https://")):
            return entry.source.removeprefix("registry:")
        if entry.source == "registry" or entry.source.startswith("registry:"):
            raise InstallError("security repair requires immutable registry origins")
        return entry.source

    def artifacts(
        sources: Lockfile,
        dependencies: dict[str, tuple[Dependency, ...]],
        requirements: dict[str, tuple[PackDependency, ...]],
    ) -> tuple[Artifact, ...]:
        selected = {entry.pack: entry for entry in sources.packs}
        result = []
        for name, entry in selected.items():
            reachable = [name]
            for consumer in reachable:
                for dependency in requirements[consumer]:
                    provider = selected.get(dependency.pack)
                    if provider is None or not dependency.accepts(provider.version):
                        raise InstallError("security repair violates a retained pack dependency")
                    if provider.pack not in reachable:
                        reachable.append(provider.pack)
            result.append(
                Artifact(
                    issuer(entry),
                    Release.model_validate(
                        {
                            "pack": name,
                            "version": entry.version,
                            "artifactDigest": entry.artifact_digest,
                        }
                    ),
                    dependencies[name]
                    + tuple(
                        Dependency(
                            issuer(selected[item]),
                            item,
                            selected[item].version,
                            selected[item].artifact_digest,
                        )
                        for item in reachable
                        if item != name
                    ),
                )
            )
        return tuple(result)

    before = security.assess(artifacts(installed, libraries, constraints))
    affected = {item.artifact.release.pack for item in before.findings if not item.allowed}
    if pack not in affected:
        return {"status": "not-required", "pack": pack, "offer": None}
    refused_files = {
        item.notice.release.artifact_digest
        for item in before.findings
        if item.notice.state == "malicious"
    } | {digest for item in before.findings for digest in item.notice.causes}
    candidates_to_change = [pack]
    for name in candidates_to_change:
        for dependency in constraints[name]:
            if dependency.pack not in candidates_to_change:
                candidates_to_change.append(dependency.pack)
    candidates_to_change.extend(name for name in hosting if name not in candidates_to_change)
    failures: list[str] = []
    for name in candidates_to_change:
        entry = entries[name]
        if not entry.source.startswith(("registry:http://", "registry:https://")):
            continue
        try:
            registry = config.registry_for(entry)
            candidates = sorted(
                {cast("str", row["version"]) for row in index.releases(registry, name)},
                key=Version,
                reverse=True,
            )
        except (InstallError, OSError, ValueError) as exc:
            failures.append(f"{name}: {exc}")
            continue
        for version in candidates:
            if Version(version) < Version(entry.version):
                continue
            try:
                sources, replacements = plan_registry_changes(
                    config,
                    installed,
                    selections=(f"{name}@{version}",),
                    engine_version=record.engine_version,
                    variant=record.variant,
                    base_id=record.base,
                    index=index,
                )
                candidate = cast("LockedPack", sources.get(name))
                requirements = dict(constraints)
                python_requirements = {
                    key: manifest.requires for key, manifest in manifests.items()
                }
                if (candidate.version, candidate.artifact_digest) != (
                    entry.version,
                    entry.artifact_digest,
                ):
                    manifest = index.release_manifest(registry, candidate)
                    engine_range = manifest.get("dinkster")
                    if not isinstance(engine_range, str) or not SpecifierSet(engine_range).contains(
                        record.engine_version, prereleases=True
                    ):
                        raise InstallError("candidate manifest has incompatible engine constraints")
                    raw = manifest.get("dependencies")
                    if not isinstance(raw, list):
                        raise InstallError("candidate release has no pack dependency constraints")
                    requirements[name] = tuple(
                        PackDependency(item["packName"], item["versionRange"]) for item in raw
                    )
                    raw_python = manifest.get("pythonRequirements")
                    if not isinstance(raw_python, list) or any(
                        not isinstance(value, str) for value in raw_python
                    ):
                        raise InstallError("candidate release has no Python dependency constraints")
                    python_requirements[name] = tuple(
                        normalize_requirement(value) for value in raw_python
                    )
                else:
                    candidate = replace(candidate, claims=entry.claims)
                    sources = Lockfile.of(
                        [candidate if item.pack == name else item for item in sources.packs]
                    )
                replacement = replacements[0]
                if any(
                    "sha256:" + wheel.sha256 in refused_files
                    for package in replacement.packages
                    for wheel in package.wheels
                ):
                    raise InstallError("candidate lock still includes a refused file")
                if replacement == locks[name] and candidate == entry:
                    continue
                proposed = {**locks, name: replacement}
                facts = parent
                selected_libraries: dict[str, tuple[Dependency, ...]] = {}
                shared = list(root_dependencies)
                for key in sorted(proposed, key=lambda key: key not in hosting):
                    constrained = replace(
                        proposed[key],
                        requirements=tuple(
                            sorted(set(proposed[key].requirements) | set(python_requirements[key]))
                        ),
                    )
                    plan = plan_pack(
                        constrained, facts, variant=record.variant, own_environment_size=0
                    )
                    plan.require_layering()
                    selected_libraries[key] = tuple(
                        Dependency(
                            wheel.registry or "",
                            parse_wheel_filename(wheel.filename)[0],
                            str(parse_wheel_filename(wheel.filename)[1]),
                            "sha256:" + wheel.sha256,
                        )
                        for wheel in plan.wheels
                    )
                    if any(not item.registry for item in selected_libraries[key]):
                        raise InstallError("candidate dependency has no issuing origin")
                    if key in hosting:
                        shared.extend(selected_libraries[key])
                        facts = replace(
                            facts,
                            versions={
                                **facts.versions,
                                **{
                                    package.name: package.version
                                    for package in proposed[key].packages
                                },
                            },
                        )
                dependencies = {
                    key: tuple(set(shared) | set(items))
                    for key, items in selected_libraries.items()
                }
                proposed_artifacts = artifacts(sources, dependencies, requirements)
                result = security.assess(proposed_artifacts)
                blocked = {
                    item.artifact.release.pack for item in result.findings if not item.allowed
                }
                changed = tuple(item for item in proposed_artifacts if item.release.pack == name)
                if (
                    result.stale_registries
                    or pack in blocked
                    or blocked - affected
                    or not security.assess(changed, installing=True).allowed
                ):
                    raise InstallError("candidate still contains blocked or unapproved files")
                if "sha256:" + hashlib.sha256(
                    (generation / "generation.json").read_bytes()
                ).hexdigest() != software or record.drift(generation):
                    raise InstallError("generation changed during security repair planning")
                return {
                    "status": "approval-required",
                    "pack": pack,
                    "offer": {
                        "sources": json.loads(sources.record_json()),
                        "dependencyLocks": {name: json.loads(replacement.to_json())},
                        "changedPack": name,
                        "approvalRequired": True,
                        "generation": record.number,
                        "software": software,
                    },
                }
            except (InstallError, ProvisionError, OSError, ValueError, KeyError, TypeError) as exc:
                failures.append(f"{name}@{version}: {exc}")
    return {"status": "no-compatible-fix", "pack": pack, "offer": None, "reasons": failures}
