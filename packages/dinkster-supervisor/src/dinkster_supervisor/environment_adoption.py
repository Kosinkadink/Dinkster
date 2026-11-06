"""Read standalone software inputs without opening an installer or user data."""

from __future__ import annotations

import hashlib
import re
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path
from typing import cast

from blake3 import blake3
from packaging.utils import canonicalize_name
from packaging.version import Version

from .environment_artifacts import parse_metadata
from .environment_state import EnvironmentError, content_digest
from .environment_stores import registry_path
from .limits import ENVIRONMENT_STREAM_CHUNK_BYTES


def _installed_dependencies(venv: Path, content: Path) -> dict[str, object]:
    """Inspect distribution headers only, without executing Python or .pth files."""
    for path in (venv, *venv.parents):
        if path == content:
            break
        if path.is_symlink() or path.is_junction():
            raise EnvironmentError(f"adoption refuses linked dependency metadata: {path}")
    if not venv.is_dir():
        return {"status": "not-recorded", "pins": [], "metadataDigests": {}}
    sites: list[Path] = []
    for library in (venv / "Lib", venv / "lib", venv / "lib64"):
        if library.name == "lib64" and library.is_symlink() and library.resolve() == venv / "lib":
            continue
        if library.is_symlink() or library.is_junction():
            raise EnvironmentError(f"adoption refuses linked dependency metadata: {library}")
        if not library.is_dir():
            continue
        if library.name == "Lib":
            sites.append(library / "site-packages")
        else:
            for python in sorted(library.iterdir()):
                if re.fullmatch(r"python[0-9]+\.[0-9]+", python.name):
                    if python.is_symlink() or python.is_junction():
                        raise EnvironmentError(
                            f"adoption refuses linked dependency metadata: {python}"
                        )
                    sites.append(python / "site-packages")
    versions: dict[str, str] = {}
    digests: dict[str, str] = {}
    inspected = False
    for site in sites:
        if site.is_symlink() or site.is_junction():
            raise EnvironmentError(f"adoption refuses linked dependency metadata: {site}")
        if not site.is_dir():
            continue
        inspected = True
        for distribution in sorted(site.glob("*.dist-info")):
            if distribution.is_symlink() or distribution.is_junction():
                raise EnvironmentError(
                    f"adoption refuses linked dependency metadata: {distribution}"
                )
            metadata = distribution / "METADATA"
            if metadata.is_symlink() or not metadata.is_file():
                raise EnvironmentError(
                    f"adoption requires intact distribution metadata: {metadata}"
                )
            raw = metadata.read_bytes()
            digests[str(metadata.relative_to(content))] = (
                "sha256:" + hashlib.sha256(raw).hexdigest()
            )
            headers = BytesParser().parsebytes(raw, headersonly=True)
            name, version = headers.get("Name"), headers.get("Version")
            if not isinstance(name, str) or not isinstance(version, str):
                raise EnvironmentError(f"adoption requires distribution name/version: {metadata}")
            name, version = canonicalize_name(name), str(Version(version))
            if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) is None:
                raise EnvironmentError(
                    f"adoption requires a canonical distribution name: {metadata}"
                )
            if name in versions and versions[name] != version:
                raise EnvironmentError(f"adoption found conflicting installed versions: {name}")
            versions[name] = version
    return {
        "status": "local-metadata" if inspected else "not-recorded",
        "pins": [f"{name}=={value}" for name, value in sorted(versions.items())],
        "metadataDigests": digests,
    }


def standalone_inputs(source: Path, target: Path, store: Path) -> dict[str, object]:
    """Describe only recorded sources and placement, never the ambient Python."""
    if not source.is_absolute() or source.is_symlink():
        raise EnvironmentError("--adopt-from must be an absolute, unlinked standalone root")
    source = source.resolve(strict=True)
    for path in (target, store, registry_path().resolve()):
        if path.is_relative_to(source) or source.is_relative_to(path):
            raise EnvironmentError("adoption destination and store must be disjoint from old root")

    def software(path: Path) -> bytes:
        if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
            raise EnvironmentError(f"adoption requires an intact software file: {path}")
        return path.read_bytes()

    current = software(source / "current").decode("utf-8").strip()
    if not current.isdecimal() or int(current) < 1:
        raise EnvironmentError("standalone root has an invalid current generation")
    lock_path = source / "generations" / f"{int(current)}.json"
    lock_text = software(lock_path).decode("utf-8")
    lock = parse_metadata(lock_text.encode("utf-8"))
    if lock.get("format") != "dinkster.lock/1" or not isinstance(lock.get("packs"), list):
        raise EnvironmentError("adoption requires a standalone pack source lock")
    packs: list[dict[str, object]] = []
    names: set[str] = set()
    for item in cast("list[object]", lock["packs"]):
        if not isinstance(item, dict):
            raise EnvironmentError("standalone source lock requires pack objects")
        row = cast("dict[str, object]", item)
        name, digest = row.get("pack"), row.get("artifactDigest")
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) is None
            or name in names
            or not isinstance(digest, str)
            or re.fullmatch(r"blake3:[0-9a-f]{64}", digest) is None
        ):
            raise EnvironmentError(
                "standalone source lock requires unique canonical pack identities"
            )
        names.add(name)
        packs.append(row)
    hosting_path = lock_path.with_suffix(".hosting.json")
    hosting_text = (
        software(hosting_path).decode("utf-8")
        if hosting_path.exists() or hosting_path.is_symlink()
        else None
    )
    hosting = parse_metadata(hosting_text.encode("utf-8")) if hosting_text is not None else {}
    if hosting.get("format") == "dinkster.hosting/1":
        groups, in_process = hosting.get("venvGroups", {}), hosting.get("inProcess", [])
    elif "format" not in hosting:
        groups, in_process = hosting, []
    else:
        raise EnvironmentError("unsupported standalone hosting format")
    if not isinstance(groups, dict) or not isinstance(in_process, list):
        raise EnvironmentError("invalid standalone hosting topology")
    in_process = cast("list[object]", in_process)
    if any(not isinstance(name, str) or name not in names for name in in_process) or len(
        set(cast("list[str]", in_process))
    ) != len(in_process):
        raise EnvironmentError("invalid standalone in-process membership")
    pointer = source / "shared-store"
    pointer_text = (
        software(pointer).decode("utf-8") if pointer.exists() or pointer.is_symlink() else None
    )
    content = Path(pointer_text.strip()) if pointer_text is not None else source
    if not content.is_absolute() or content.is_symlink():
        raise EnvironmentError("standalone shared store must be absolute and unlinked")
    content = content.resolve(strict=True)
    for path in (target, store, registry_path().resolve()):
        if path.is_relative_to(content) or content.is_relative_to(path):
            raise EnvironmentError("adoption must not write into the standalone content store")
    archives = content / "artifacts"
    if archives.is_symlink():
        raise EnvironmentError("standalone archives must not be linked")
    payloads: dict[str, str] = {}
    dependencies: dict[str, object] = {}
    for row in packs:
        identity = cast("str", row["artifactDigest"])
        path = archives / f"{identity[7:]}.zip"
        if path.is_symlink() or not path.is_file():
            raise EnvironmentError(
                f"standalone pack archive is missing: {row['pack']} ({identity})"
            )
        actual = blake3()
        with path.open("rb") as stream:
            while chunk := stream.read(ENVIRONMENT_STREAM_CHUNK_BYTES):
                actual.update(chunk)
        if "blake3:" + actual.hexdigest() != identity:
            raise EnvironmentError(f"standalone pack archive digest differs: {row['pack']}")
        payloads[path.name] = content_digest(path)
        if row["pack"] in in_process:
            dependencies[cast("str", row["pack"])] = {
                "status": "ambient-not-recorded",
                "scopes": {},
                "runtimePins": hosting.get("runtimePins", {}),
            }
            continue
        try:
            with zipfile.ZipFile(path) as archive:
                manifest = tomllib.loads(archive.read("dinkster-pack.toml").decode("utf-8"))
        except (zipfile.BadZipFile, KeyError, UnicodeError, tomllib.TOMLDecodeError) as exc:
            raise EnvironmentError("standalone archive requires an intact pack manifest") from exc
        pack = manifest.get("pack")
        manifest_name = (
            cast("dict[str, object]", pack).get("name") if isinstance(pack, dict) else None
        )
        if (
            not isinstance(manifest_name, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", manifest_name) is None
            or canonicalize_name(manifest_name) != row["pack"]
        ):
            raise EnvironmentError("standalone archive manifest differs from its pack identity")
        venvs = content / "venvs"
        if venvs.is_symlink() or venvs.is_junction():
            raise EnvironmentError("adoption refuses linked standalone venv storage")
        scopes = [venvs]
        if pointer_text is not None:
            scopes = sorted(venvs.iterdir()) if venvs.is_dir() else []
        dependencies[cast("str", row["pack"])] = {
            "scopes": {
                scope.name if pointer_text is not None else "private": _installed_dependencies(
                    scope / identity[7:] / manifest_name, content
                )
                for scope in scopes
            },
            "status": "inspect-scopes" if scopes else "not-recorded",
        }
    return {
        "source": str(source),
        "generation": int(current),
        "sourceLock": lock_text,
        "packs": packs,
        "hosting": hosting_text,
        "inProcessPacks": in_process,
        "venvGroups": groups,
        "storePointer": pointer_text,
        "artifacts": str(archives),
        "archiveDigests": payloads,
        "installedDependencies": dependencies,
        "previousEngine": "not recorded by standalone installer",
        "previousBase": "not recorded by standalone installer",
    }


def dependency_comparison(
    adoption: dict[str, object], locks: tuple[Path, ...]
) -> list[dict[str, object]]:
    """Compare observed versions to lock declarations, not an ambient base guess."""
    comparisons: list[dict[str, object]] = []
    installed = cast("dict[str, dict[str, object]]", adoption["installedDependencies"])
    for path in locks:
        lock = parse_metadata(path.read_bytes())
        name = cast("str", lock["pack"])
        packages = lock.get("packages")
        if not isinstance(packages, list):
            raise EnvironmentError("adoption dependency lock requires package records")
        target: dict[str, str] = {}
        for raw in cast("list[object]", packages):
            if not isinstance(raw, dict):
                raise EnvironmentError("adoption dependency lock requires package objects")
            package = cast("dict[str, object]", raw)
            distribution, version = package.get("name"), package.get("version")
            if not isinstance(distribution, str) or not isinstance(version, str):
                raise EnvironmentError("adoption dependency lock requires package name/version")
            target[canonicalize_name(distribution)] = str(Version(version))
        scopes = cast("dict[str, dict[str, object]]", installed[name]["scopes"])
        for scope, observed in (scopes or {"unrecorded": installed[name]}).items():
            source = {
                pin.split("==", 1)[0]: pin.split("==", 1)[1]
                for pin in cast("list[str]", observed.get("pins", []))
            }
            comparisons.append(
                {
                    "pack": name,
                    "scope": scope,
                    "sourceStatus": observed["status"],
                    "targetLock": str(path),
                    "changes": [
                        {"name": name, "source": source.get(name), "targetLock": target.get(name)}
                        for name in sorted(source.keys() | target.keys())
                        if source.get(name) != target.get(name)
                    ],
                }
            )
    return comparisons
