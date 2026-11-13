"""Compute inert, compatible security repair offers from retained and published metadata."""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass, replace
from itertools import combinations, product
from pathlib import Path
from typing import Any, cast

from dinkster_registry import InstallError, LockedPack, Lockfile
from dinkster_security.documents import Release
from dinkster_security.host import SecurityHost
from dinkster_security.inventory import generation_dependencies
from dinkster_security.policy import Artifact, Dependency
from dinkster_workers import PackManifest, load_manifest
from dinkster_workers.accelerator import resolve_accelerator
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


@dataclass(frozen=True)
class _GenerationSecurityInventory:
    """Sealed software inputs shared by offline inspection and repair planning."""

    record: Any
    software: str
    installed: Lockfile
    libraries: dict[str, tuple[Dependency, ...]]
    manifests: dict[str, PackManifest]
    locks: dict[str, PackLock]

    @classmethod
    def read(cls, generation: Path) -> _GenerationSecurityInventory:
        state = importlib.import_module("dinkster_supervisor.environment_state")
        software = (
            "sha256:" + hashlib.sha256((generation / "generation.json").read_bytes()).hexdigest()
        )
        record = state.Generation.read(generation / "generation.json")
        files = dict(record.files)
        if record.drift(generation):
            raise InstallError("security inventory requires intact sealed generation inputs")
        for name in (*files, "generation.json"):
            path = generation / name
            if not path.resolve().is_relative_to(generation.resolve()) or any(
                parent.is_symlink() or parent.is_junction() for parent in (path, *path.parents)
            ):
                raise InstallError("security inventory requires intact sealed generation inputs")
        installed = Lockfile.from_record_json((generation / "dinkster.lock").read_text())
        entries = {entry.pack: entry for entry in installed.packs}
        libraries = generation_dependencies(
            generation, files, tuple(entries), record.in_process_packs
        )
        manifests = {
            name: load_manifest(pack_manifest_path(generation / "pack-sources" / name))
            for name in entries
        }
        locks = {
            name: PackLock.read(generation / "packs" / name / "pack-lock.json") for name in entries
        }
        required = {"dinkster.lock", "installed-wheels.json"} | {
            path.relative_to(generation).as_posix()
            for name, manifest in manifests.items()
            for path in (manifest.path, generation / "packs" / name / "pack-lock.json")
        }
        if required - files.keys():
            raise InstallError(
                "security inventory requires sealed source and dependency constraints"
            )
        return cls(record, software, installed, libraries, manifests, locks)

    def artifacts(self) -> tuple[Artifact, ...]:
        return _artifacts(
            self.installed,
            self.libraries,
            {name: manifest.dependencies for name, manifest in self.manifests.items()},
        )


def _artifacts(
    sources: Lockfile,
    dependencies: dict[str, tuple[Dependency, ...]],
    requirements: dict[str, tuple[PackDependency, ...]],
) -> tuple[Artifact, ...]:
    def issuer(entry: LockedPack) -> str:
        if entry.source.startswith(("registry:http://", "registry:https://")):
            return entry.source.removeprefix("registry:")
        if entry.source == "registry" or entry.source.startswith("registry:"):
            raise InstallError("security inventory requires immutable registry origins")
        return entry.source

    selected = {entry.pack: entry for entry in sources.packs}
    result = []
    for name, entry in selected.items():
        reachable = [name]
        for consumer in reachable:
            for dependency in requirements[consumer]:
                provider = selected.get(dependency.pack)
                if provider is None or not dependency.accepts(provider.version):
                    raise InstallError("security inventory violates a retained pack dependency")
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
                tuple(dependency for item in reachable for dependency in dependencies[item])
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


def generation_security_status(generation: Path, pack: str | None = None) -> dict[str, object]:
    """Inspect accepted evidence for sealed software, without transport or execution."""
    if not generation.is_absolute() or generation.parent.name != "generations":
        raise InstallError("security status requires an absolute managed generation")
    root = generation.parent.parent
    if not (root / "security.sqlite3").is_file():
        raise InstallError("security status requires independently provisioned trust")
    inventory = _GenerationSecurityInventory.read(generation)
    if generation.name != str(inventory.record.number):
        raise InstallError("security generation directory and recorded number disagree")
    artifacts = inventory.artifacts()
    if pack is not None:
        artifacts = tuple(item for item in artifacts if item.release.pack == pack)
        if not artifacts:
            raise InstallError("security details require an installed pack")
    result = SecurityHost(root).status(artifacts)
    if inventory.record.drift(generation) or inventory.software != (
        "sha256:" + hashlib.sha256((generation / "generation.json").read_bytes()).hexdigest()
    ):
        raise InstallError("generation changed during security inspection")
    return {
        **result,
        "generation": inventory.record.number,
        "software": inventory.software,
        "packs": [
            {"registry": item.registry, **item.release.model_dump(mode="json", by_alias=True)}
            for item in artifacts
        ],
        "workInventory": {
            "status": "unavailable",
            "reason": "Offline generation inspection does not observe running jobs or workers.",
        },
        "automaticResume": False,
    }


def plan_security_repair(
    generation: Path,
    pack: str | None,
    config: RegistryConfig,
    *,
    index: StaticPackIndex,
    security: SecurityHost,
    accelerator: str,
) -> dict[str, object]:
    """Offer one validated change, never download payloads, import packs or apply it.

    The offer is input to a host-owned approval plan, not authorization to apply.
    Root interpreter dependencies cannot be repaired by changing a pack lock.
    """
    accelerator = resolve_accelerator(accelerator)
    inventory = _GenerationSecurityInventory.read(generation)
    record, software, installed = inventory.record, inventory.software, inventory.installed
    libraries, manifests, locks = inventory.libraries, inventory.manifests, inventory.locks
    entries = {entry.pack: entry for entry in installed.packs}
    if pack is not None and pack not in entries:
        raise InstallError("security repair requires an installed pack")
    hosting = record.in_process_packs
    constraints = {name: manifest.dependencies for name, manifest in manifests.items()}
    state = importlib.import_module("dinkster_supervisor.environment_state")
    wheel_compatibility = importlib.import_module(
        "dinkster_supervisor.environment_constraints"
    ).wheel_compatibility
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
    before = security.assess(_artifacts(installed, libraries, constraints))
    affected = {item.artifact.release.pack for item in before.findings}
    targets = (pack,) if pack is not None else tuple(sorted(affected))
    if not targets or (pack is not None and pack not in affected):
        return {"status": "not-required", "pack": pack, "packs": list(targets), "offer": None}
    refused_files = {
        item.notice.release.artifact_digest
        for item in before.findings
        if item.notice.state == "malicious"
    } | {digest for item in before.findings for digest in item.notice.causes}
    candidates_to_change = list(targets)
    for name in candidates_to_change:
        for dependency in constraints[name]:
            if dependency.pack not in candidates_to_change:
                candidates_to_change.append(dependency.pack)
    # A replacement can introduce constraints on any already installed pack.
    candidates_to_change.extend(name for name in entries if name not in candidates_to_change)
    failures: list[str] = []
    attempts = 0
    options: dict[str, list[str]] = {}
    for name in candidates_to_change:
        entry = entries[name]
        if not entry.source.startswith(("registry:http://", "registry:https://")):
            continue
        try:
            registry = config.registry_for(entry)
            options[name] = sorted(
                {
                    cast("str", row["version"])
                    for row in index.releases(registry, name)
                    if Version(cast("str", row["version"])) >= Version(entry.version)
                },
                key=Version,
                reverse=True,
            )
        except (InstallError, OSError, ValueError) as exc:
            failures.append(f"{name}: {exc}")
    for count in range(1, len(options) + 1):
        selections = (
            tuple(f"{name}@{version}" for name, version in zip(names, versions, strict=True))
            for names in combinations(options, count)
            for versions in product(*(options[name] for name in names))
        )
        for selection in selections:
            # Registry metadata must not turn an interactive preview into an
            # unbounded Cartesian search. Exhaustion is not proof of no fix.
            if attempts >= 512:
                return {
                    "status": "search-incomplete",
                    "pack": pack,
                    "packs": list(targets),
                    "offer": None,
                    "reasons": ["Repair search exceeded 512 combinations; no change was made."],
                }
            attempts += 1
            try:
                sources, replacements = plan_registry_changes(
                    config,
                    installed,
                    selections=selection,
                    engine_version=record.engine_version,
                    variant=record.variant,
                    base_id=record.base,
                    index=index,
                )
                requirements = dict(constraints)
                python_requirements = {
                    key: tuple(
                        normalize_requirement(value) for value in manifest.requires_for(accelerator)
                    )
                    for key, manifest in manifests.items()
                }
                changed_releases: set[str] = set()
                changed_locks = {lock.pack: lock for lock in replacements}
                for name in changed_locks:
                    entry = entries[name]
                    candidate = cast("LockedPack", sources.get(name))
                    if (candidate.version, candidate.artifact_digest) != (
                        entry.version,
                        entry.artifact_digest,
                    ):
                        changed_releases.add(name)
                        manifest = index.release_manifest(config.registry_for(candidate), candidate)
                        engine_range = manifest.get("dinkster")
                        if not isinstance(engine_range, str) or not SpecifierSet(
                            engine_range
                        ).contains(record.engine_version, prereleases=True):
                            raise InstallError(
                                "candidate manifest has incompatible engine constraints"
                            )
                        raw = manifest.get("dependencies")
                        if not isinstance(raw, list):
                            raise InstallError(
                                "candidate release has no pack dependency constraints"
                            )
                        requirements[name] = tuple(
                            PackDependency(item["packName"], item["versionRange"]) for item in raw
                        )
                        raw_python = manifest.get("pythonRequirements")
                        if not isinstance(raw_python, list) or any(
                            not isinstance(value, str) for value in raw_python
                        ):
                            raise InstallError(
                                "candidate release has no Python dependency constraints"
                            )
                        extras = manifest.get("pythonExtraRequirements")
                        if not isinstance(extras, dict) or any(
                            not isinstance(values, list)
                            or any(not isinstance(value, str) for value in values)
                            for values in extras.values()
                        ):
                            raise InstallError(
                                "candidate release has no accelerator dependency constraints"
                            )
                        python_requirements[name] = tuple(
                            normalize_requirement(value)
                            for value in [*raw_python, *extras.get(accelerator, [])]
                        )
                    else:
                        candidate = replace(candidate, claims=entry.claims)
                        sources = Lockfile.of(
                            [candidate if item.pack == name else item for item in sources.packs]
                        )
                for key, retained in manifests.items():
                    if key not in changed_releases and retained.compatibility is not None:
                        retained.compatibility.require_engine(key, record.engine_version)
                if any(
                    "sha256:" + wheel.sha256 in refused_files
                    for replacement in replacements
                    for package in replacement.packages
                    for wheel in package.wheels
                ):
                    raise InstallError("candidate lock still includes a refused file")
                if not changed_releases and all(
                    lock == locks[name] for name, lock in changed_locks.items()
                ):
                    continue
                proposed = {**locks, **changed_locks}
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
                proposed_artifacts = _artifacts(sources, dependencies, requirements)
                result = security.assess(proposed_artifacts)
                changed = tuple(
                    item for item in proposed_artifacts if item.release.pack in changed_locks
                )
                if (
                    not result.allowed
                    or any(item.artifact.release.pack in targets for item in result.findings)
                    or not security.assess(changed, installing=True).allowed
                ):
                    raise InstallError(
                        "candidate still contains flagged, blocked or unapproved files"
                    )
                if "sha256:" + hashlib.sha256(
                    (generation / "generation.json").read_bytes()
                ).hexdigest() != software or record.drift(generation):
                    raise InstallError("generation changed during security repair planning")
                return {
                    "status": "approval-required",
                    "pack": pack,
                    "packs": list(targets),
                    "offer": {
                        "sources": json.loads(sources.record_json()),
                        "previousSources": json.loads(installed.record_json()),
                        "dependencyLocks": {
                            name: json.loads(lock.to_json()) for name, lock in changed_locks.items()
                        },
                        "engineVersion": record.engine_version,
                        "engineRelease": record.engine_release,
                        "variant": record.variant,
                        "baseId": record.base,
                        "sourceWheels": {},
                        "securityRegistries": {
                            item.source: item.source.removeprefix("registry:")
                            for item in sources.packs
                        },
                        "plan": [
                            f"select {item.release.pack}@{item.release.version} "
                            f"({item.release.artifact_digest}) with the reviewed dependency lock"
                            for item in changed
                        ]
                        + ["retain engine, base, other pack locks and placement"],
                        "changedPacks": list(changed_locks),
                        "changedPack": next(iter(changed_locks))
                        if len(changed_locks) == 1
                        else None,
                        "approvalRequired": True,
                        "generation": record.number,
                        "software": software,
                        "accelerator": accelerator,
                    },
                }
            except (InstallError, ProvisionError, OSError, ValueError, KeyError, TypeError) as exc:
                failures.append(f"{', '.join(selection)}: {exc}")
    return {
        "status": "no-compatible-fix",
        "pack": pack,
        "packs": list(targets),
        "offer": None,
        "reasons": failures,
        "targets": [
            {
                "pack": name,
                "status": "no-compatible-fix",
                "reason": "No compatible joint repair was found; nothing was changed.",
            }
            for name in targets
        ],
    }
