"""Read default-pack payload bounds without an engine or payload acquisition."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

from blake3 import blake3
from dinkster_store import object_path
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

from .environment_artifacts import EnvironmentCatalog, engine_package_versions, parse_metadata
from .environment_downloads import read_metadata
from .environment_state import content_digest

REGISTRY = "https://registry.dinkster.org"


def target_markers(base: Mapping[str, object]) -> dict[str, str]:
    platform = cast("dict[str, str]", base["platform"])
    system, arch = platform["os"], platform["arch"]
    python = cast("str", base["python"])
    return {
        **cast("dict[str, str]", default_environment()),
        "python_full_version": python,
        "python_version": ".".join(python.split(".")[:2]),
        "implementation_version": python,
        "implementation_name": "cpython",
        "platform_python_implementation": "CPython",
        "sys_platform": {"windows": "win32", "macos": "darwin"}.get(system, "linux"),
        "os_name": "nt" if system == "windows" else "posix",
        "platform_system": {"windows": "Windows", "macos": "Darwin"}.get(system, "Linux"),
        "platform_machine": "AMD64" if system == "windows" and arch == "x86_64" else arch,
    }


def requirement_conflicts(
    requirements: list[str],
    engine_packages: Mapping[str, str],
    markers: Mapping[str, str] | None,
) -> list[str]:
    conflicts: list[str] = []
    for text in requirements:
        try:
            requirement = Requirement(text)
        except ValueError as exc:
            raise ValueError(f"registry release has invalid Python requirement {text!r}") from exc
        if requirement.marker:
            if markers is None:
                raise ValueError(
                    "target interpreter markers are required for conditional requirements"
                )
            if not requirement.marker.evaluate(dict(markers)):
                continue
        name = canonicalize_name(requirement.name)
        # Private dependencies belong to the lock; inference is always engine-owned.
        if name not in engine_packages and name != "dinkster-inference":
            continue
        installed = engine_packages.get(name)
        if (
            installed is None
            or requirement.url is not None
            or not requirement.specifier.contains(installed, prereleases=True)
        ):
            conflicts.append(f"requires {text}; engine provides {name}=={installed or 'absent'}")
    return conflicts


def _requirements(value: object) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("registry release must declare its Python requirements")
    if any(not isinstance(item, str) for item in cast("list[object]", value)):
        raise ValueError("registry release must declare its Python requirements")
    return cast("list[str]", value)


def default_pack_bound(
    catalog: EnvironmentCatalog,
    variant: str,
    store: Path,
    *,
    inference: dict[str, object] | None = None,
) -> dict[str, object]:
    """Bound all locked alternatives for the newest compatible default sources."""
    if os.environ.get("DINKSTER_REGISTRY") or os.environ.get("DINKSTER_REGISTRIES"):
        raise ValueError("Default-pack estimates are unavailable for custom registry configuration")
    defaults = catalog.engine.get("defaultPacks")
    if not isinstance(defaults, dict):
        raise ValueError("target engine has no default pack declaration")
    defaults = cast("dict[str, object]", defaults)
    catalog_document = parse_metadata(read_metadata(REGISTRY, "static/packs/index.json"))
    entries = catalog_document.get("items")
    if catalog_document.get("schemaVersion") != 2 or not isinstance(entries, list):
        raise ValueError("invalid pack catalog")
    available: set[str] = set()
    names: set[str] = set()
    for value in cast("list[object]", entries):
        if not isinstance(value, dict):
            raise ValueError("invalid pack catalog entry")
        entry = cast("dict[str, object]", value)
        name, version = entry.get("name"), entry.get("latestVersion")
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) is None
            or name in names
            or not isinstance(version, str)
            or entry.get("url") != f"packs/{name}/index.json"
            or type(entry.get("library")) is not bool
            or "publisher" not in entry
            or entry["publisher"] is not None
            and not isinstance(entry["publisher"], str)
        ):
            raise ValueError("invalid pack catalog identity")
        Version(version)
        names.add(name)
        if not entry["library"]:
            available.add(name)
    if not defaults.keys() <= available:
        raise ValueError("default packs are absent from the active pack catalog")
    base = catalog.bases[variant]
    base_id = catalog.channel.base_for(variant)
    markers = target_markers(base)
    packages = engine_package_versions(catalog.engine, variant, base, inference=inference)
    api = packages.get("dinkster-api", cast("str", catalog.engine["version"]))
    wheels: dict[tuple[str, str], int] = {}
    identities: list[object] = []

    def wheel(value: object, name: str, version: str) -> None:
        if not isinstance(value, dict):
            raise ValueError("invalid wheel metadata")
        row = cast("dict[str, object]", value)
        filename, digest, size = row.get("filename"), row.get("sha256"), row.get("size")
        if not isinstance(filename, str) or Path(filename).name != filename or "\\" in filename:
            raise ValueError("invalid wheel filename")
        wheel_name, wheel_version, _, _ = parse_wheel_filename(filename)
        if wheel_name != name or wheel_version != Version(version):
            raise ValueError("wheel identity differs from its distribution")
        if not isinstance(digest, str) or re.fullmatch(r"[a-f0-9]{64}", digest) is None:
            raise ValueError("invalid wheel identity")
        if type(size) is not int or size < 0:
            raise ValueError("invalid wheel size")
        url = row.get("url")
        if not isinstance(url, str) or urlparse(url).scheme not in {"http", "https", "file"}:
            raise ValueError("invalid wheel location")
        key = (digest, filename)
        if key in wheels and wheels[key] != size:
            raise ValueError("conflicting wheel sizes")
        wheels[key] = size

    for pack, minimum in sorted(defaults.items()):
        if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", pack) is None:
            raise ValueError("invalid default pack name")
        document = parse_metadata(read_metadata(REGISTRY, f"static/packs/{pack}/index.json"))
        rows = document.get("items")
        if (
            document.get("schemaVersion") != 2
            or document.get("name") != pack
            or not isinstance(rows, list)
        ):
            raise ValueError("invalid pack index")
        candidates: list[dict[str, object]] = []
        versions: set[str] = set()
        for value in cast("list[object]", rows):
            if not isinstance(value, dict):
                raise ValueError("invalid pack release")
            row = cast("dict[str, object]", value)
            if row.get("library") is True:
                continue
            if row.get("name") != pack:
                raise ValueError("invalid pack release")
            version = row.get("version")
            if not isinstance(version, str) or version in versions:
                raise ValueError("invalid or duplicate pack version")
            versions.add(version)
            if Version(version) >= Version(str(minimum)):
                candidates.append(row)
        for row in sorted(
            candidates, key=lambda row: Version(cast("str", row["version"])), reverse=True
        ):
            if not SpecifierSet(cast("str", row["engine"])).contains(
                api, prereleases=True
            ) or requirement_conflicts(_requirements(row.get("requires")), packages, markers):
                continue
            bindings = row.get("installLocks")
            if not isinstance(bindings, dict):
                raise ValueError("release has no dependency lock bindings")
            bindings = cast("dict[str, object]", bindings)
            binding = bindings.get(variant)
            if binding is None:
                continue
            if not isinstance(binding, dict):
                raise ValueError("invalid dependency lock binding")
            binding = cast("dict[str, object]", binding)
            outer = row.get("wheels")
            if not isinstance(outer, list):
                raise ValueError("release must declare one pack wheel")
            outer = cast("list[object]", outer)
            if len(outer) != 1 or not isinstance(outer[0], dict):
                raise ValueError("release must declare one pack wheel")
            outer_wheel = cast("dict[str, object]", outer[0])
            if (
                outer_wheel.get("url")
                != f"wheels/{outer_wheel.get('sha256')}/{outer_wheel.get('filename')}"
                or outer_wheel.get("tags") != ["py3-none-any"]
            ):
                raise ValueError("invalid pack wheel location or tags")
            outer_identity = outer_wheel.get("digest")
            if (
                not isinstance(outer_identity, str)
                or re.fullmatch(r"blake3:[a-f0-9]{64}", outer_identity) is None
            ):
                raise ValueError("invalid pack artifact identity")
            parts = binding.get("parts", {})
            if not isinstance(parts, dict):
                raise ValueError("invalid execution-part bindings")
            parts = cast("dict[str, object]", parts)
            if any(
                re.fullmatch(r"[a-z0-9][a-z0-9-]*", part) is None or part == "main"
                for part in parts
            ):
                raise ValueError("invalid execution-part name")
            selected: list[dict[str, object]] = []
            bound: list[str] = []
            for part, facts in [(None, binding), *sorted(parts.items())]:
                if not isinstance(facts, dict):
                    raise ValueError("invalid execution-part binding")
                facts = cast("dict[str, object]", facts)
                if facts.get("baseId") != base_id:
                    break
                digest, size = facts.get("digest"), facts.get("size")
                if (
                    not isinstance(digest, str)
                    or re.fullmatch(r"blake3:[a-f0-9]{64}", digest) is None
                ):
                    raise ValueError("invalid lock identity")
                path = "locks/" + digest.replace(":", "/") + ".json"
                if (
                    facts.get("url") != path
                    or facts.get("artifactDigest") != outer_identity
                    or type(size) is not int
                    or not 1 <= size <= 1024 * 1024
                ):
                    raise ValueError("invalid lock binding")
                payload = read_metadata(REGISTRY, "static/" + path)
                if len(payload) != size or "blake3:" + blake3(payload).hexdigest() != digest:
                    raise ValueError("dependency lock differs from its binding")
                lock = parse_metadata(payload)
                if (
                    lock.get("format")
                    not in {"dinkster.pack-lock/1", "dinkster.pack-lock/2", "dinkster.pack-lock/3"}
                    or lock.get("pack") != pack
                    or lock.get("variant") != variant
                    or lock.get("part") != part
                ):
                    raise ValueError("dependency lock identity differs")
                if (
                    lock["format"] == "dinkster.pack-lock/2"
                    and not isinstance(lock.get("engine"), str)
                ) or (lock["format"] == "dinkster.pack-lock/3" and not isinstance(part, str)):
                    raise ValueError("dependency lock format lacks its engine or part")
                if (
                    requirement_conflicts(
                        _requirements(lock.get("requirements")), packages, markers
                    )
                    or not SpecifierSet(cast("str", lock["requiresPython"])).contains(
                        markers["python_full_version"], prereleases=True
                    )
                    or (
                        lock.get("engine") is not None
                        and not SpecifierSet(cast("str", lock["engine"])).contains(
                            api, prereleases=True
                        )
                    )
                ):
                    break
                selected.append(lock)
                bound.append(digest)
            else:
                wheel(
                    {**outer_wheel, "url": REGISTRY + "/static/" + cast("str", outer_wheel["url"])},
                    pack,
                    cast("str", row["version"]),
                )
                for lock in selected:
                    distributions = lock.get("packages")
                    if not isinstance(distributions, list):
                        raise ValueError("invalid locked distributions")
                    names: set[str] = set()
                    for distribution in cast("list[object]", distributions):
                        if not isinstance(distribution, dict):
                            raise ValueError("invalid locked wheels")
                        distribution = cast("dict[str, object]", distribution)
                        name, version = distribution.get("name"), distribution.get("version")
                        if (
                            not isinstance(name, str)
                            or name != canonicalize_name(name)
                            or name in names
                            or name == pack
                            or (name == "dinkster-api" and lock.get("engine") is not None)
                            or not isinstance(version, str)
                        ):
                            raise ValueError("invalid locked distribution identity")
                        names.add(name)
                        locked_wheels = distribution.get("wheels")
                        if not isinstance(locked_wheels, list):
                            raise ValueError("invalid locked wheels")
                        for value in cast("list[object]", locked_wheels):
                            wheel(value, name, version)
                identities.append([pack, row["version"], outer, bound])
                break
        else:
            raise ValueError(f"no compatible default pack metadata for {pack}")

    missing = 0
    for (digest, filename), size in wheels.items():
        cached_paths = (
            object_path(store / "wheels", "sha256:" + digest) / filename,
            store / "wheels" / filename,
        )
        if not any(
            path.is_file()
            and path.stat().st_size == size
            and content_digest(path) == "sha256:" + digest
            for path in cached_paths
        ):
            missing += size
    identity = hashlib.sha256(
        json.dumps(
            [catalog.channel.engine_release, base_id, packages, identities], sort_keys=True
        ).encode()
    ).hexdigest()
    return {"bytes": missing, "identity": identity}
