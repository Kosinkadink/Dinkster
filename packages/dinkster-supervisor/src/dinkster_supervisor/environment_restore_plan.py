"""Read-only restore availability and explicitly approved minimal alternatives."""

from __future__ import annotations

import copy
import email
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import tomllib
import zipfile
from pathlib import Path
from typing import cast
from urllib.parse import unquote, urlparse

from dinkster_store import object_path
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import parse_tag
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

from .environment_artifacts import (
    artifact_rows,
    dependency_source,
    dependency_wheels,
    discover,
    engine_constraints,
    parse_metadata,
    trusted_metadata,
    validate_dependency_base,
    validate_manifest,
)
from .environment_constraints import (
    effective_constraints,
    installed_wheel_origins,
    selected_wheel_rows,
)
from .environment_downloads import artifact_available, fetch_artifact, read_metadata
from .environment_frontend import frontend_filename, frontend_requirements, select_frontend
from .environment_generations import ACQUIRE_PACK_SOURCES, retained_pack_archive, verify_base_inputs
from .environment_history import history, validate_import
from .environment_state import (
    Channel,
    EnvironmentError,
    content_digest,
    digest,
    interpreter,
    read_current,
    relative_file,
)


def _pack_source_missing(
    snapshot: dict[str, object], store: Path, root: Path | None
) -> list[dict[str, object]]:
    text = _metadata(snapshot).get("dinkster.lock")
    if text is None:
        return []
    lock = parse_metadata(text.encode("utf-8"))
    rows = cast("list[dict[str, object]]", lock["packs"])
    needed = [
        row
        for row in rows
        if retained_pack_archive(store, cast("str", row["artifactDigest"])) is None
    ]
    if not needed:
        return []
    lock["packs"] = needed
    provenance = json.loads(_metadata(snapshot).get("artifact-sources.json", "{}"))
    mirrors = {
        name: row["gitMirror"]
        for name, row in provenance.get("packs", {}).items()
        if row.get("gitMirror")
    }
    unavailable = {cast("str", row["artifactDigest"]) for row in needed}
    current = read_current(root) if root is not None else None
    if current is not None and root is not None:
        try:
            with tempfile.TemporaryDirectory(prefix="dinkster-source-probe-") as directory:
                with subprocess.Popen(
                    [
                        str(interpreter(root / "builds" / str(current))),
                        "-I",
                        "-B",
                        "-c",
                        ACQUIRE_PACK_SOURCES,
                        str(root),
                        *(["--git-mirrors", json.dumps(mirrors)] if mirrors else []),
                    ],
                    cwd=directory,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "DINKSTER_SHARED_STORE": ""},
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    start_new_session=os.name != "nt",
                ) as process:
                    try:
                        output, _ = process.communicate(json.dumps(lock), timeout=120)
                    except subprocess.TimeoutExpired:
                        try:
                            if os.name == "nt":
                                subprocess.run(
                                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL,
                                    timeout=10,
                                )
                            else:
                                os.killpg(process.pid, signal.SIGKILL)
                        finally:
                            process.kill()
                        raise
            if process.returncode == 0:
                reported: object = json.loads(output)
                if isinstance(reported, dict) and all(
                    isinstance(item, str) and item in unavailable and isinstance(reason, str)
                    for item, reason in cast("dict[object, object]", reported).items()
                ):
                    unavailable = set(cast("dict[str, str]", reported))
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    return [
        {
            "kind": "pack-source",
            "identity": row["artifactDigest"],
            "artifact": f"{cast('str', row['artifactDigest'])[7:]}.zip",
            "pack": row["pack"],
            "source": row["source"],
        }
        for row in needed
        if row["artifactDigest"] in unavailable
    ]


def _identity(value: object) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
                "ascii"
            )
        ).hexdigest()
    )


def _metadata(snapshot: dict[str, object]) -> dict[str, str]:
    value = snapshot.get("metadata")
    if not isinstance(value, dict) or any(
        not isinstance(item, str) for item in cast("dict[str, object]", value).values()
    ):
        raise EnvironmentError("snapshot requires software metadata text")
    return cast("dict[str, str]", value)


def _sources(provenance: dict[str, object], key: str) -> tuple[str, ...]:
    value = provenance.get(key)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in cast("list[object]", value))
    ):
        raise EnvironmentError(f"snapshot requires recorded {key} locations")
    return tuple(cast("list[str]", value))


def missing_inputs(
    snapshot: dict[str, object], store: Path, *, offline: bool = False
) -> list[dict[str, object]]:
    """Prefer verified retained payloads; remote probes never populate the store."""
    metadata = _metadata(snapshot)
    source_engine = (
        parse_metadata(metadata.get("engine-manifest.json", "{}").encode("utf-8")).get("source")
        == "git"
    )
    if source_engine:
        from .environment_sources import source_missing_inputs

        return [
            *source_missing_inputs(snapshot, store, offline=offline),
            *_pack_missing_inputs(snapshot, store, offline=offline),
            *(_pack_source_missing(snapshot, store, None) if offline else []),
        ]
    if snapshot.get("source") == "local":
        return [
            *_local_missing_inputs(snapshot, store, offline=offline),
            *_pack_missing_inputs(snapshot, store, offline=offline),
            *(_pack_source_missing(snapshot, store, None) if offline else []),
        ]
    if snapshot.get("source") not in {"published", "source"}:
        raise EnvironmentError("restore planning requires published software inputs")
    metadata = _metadata(snapshot)
    for name in (
        "artifact-sources.json",
        "engine-manifest.json",
        "base-manifest.json",
        "constraints.txt",
    ):
        if name not in metadata:
            raise EnvironmentError(f"snapshot is missing recorded software input: {name}")
    provenance = parse_metadata(metadata["artifact-sources.json"].encode("utf-8"))
    raw_engine, raw_base = snapshot.get("engine"), snapshot.get("base")
    if not isinstance(raw_engine, dict) or not isinstance(raw_base, dict):
        raise EnvironmentError("snapshot requires engine and base identities")
    recorded_engine, recorded_base = (
        cast("dict[str, str]", raw_engine),
        cast("dict[str, str]", raw_base),
    )
    engine = validate_manifest(
        metadata["engine-manifest.json"].encode("utf-8"), recorded_engine["release"], None
    )
    base = validate_manifest(
        metadata["base-manifest.json"].encode("utf-8"),
        recorded_base["id"],
        recorded_base["variant"],
    )
    missing: list[dict[str, object]] = []

    def check(
        kind: str,
        identity: str,
        row: dict[str, object],
        key: str,
        roots: tuple[str, ...],
        cached: Path,
    ) -> None:
        name, sha, size = (
            cast("str", row[key]),
            cast("str", row["sha256"]),
            cast("int", row["size"]),
        )
        if not artifact_available(roots, name, cached, sha, size):
            missing.append(
                {
                    "kind": kind,
                    "identity": identity,
                    "artifact": name,
                    "sha256": sha,
                    "locations": list(roots),
                }
            )

    engine_roots = _sources(provenance, "engine")
    base_roots = _sources(provenance, "base")
    base_cache = object_path(store / "bases", recorded_base["id"])
    if not base_cache.is_dir() or base_cache.is_symlink():
        for row in artifact_rows(cast("dict[str, object]", base["archive"])["parts"], "filename"):
            check(
                "base",
                recorded_base["id"],
                row,
                "filename",
                base_roots,
                object_path(store / "downloads", recorded_base["id"])
                / cast("str", row["filename"]),
            )
    else:
        for row in artifact_rows(base["wheels"], "filename"):
            check(
                "base wheel",
                recorded_base["id"],
                row,
                "filename",
                (str(base_cache / "wheels"),),
                base_cache / "wheels" / cast("str", row["filename"]),
            )
    owned = set(cast("list[str]", base["replaces"]))
    release = object_path(store / "engines", recorded_engine["release"])
    source_payloads = any(
        row["name"] == "official-packs.lock" for row in artifact_rows(engine["artifacts"], "name")
    )
    for row in artifact_rows(engine["artifacts"], "name"):
        name = cast("str", row["name"])
        if name.endswith(".whl"):
            if parse_wheel_filename(name)[0] in owned:
                continue
            cached = object_path(store / "wheels", "sha256:" + cast("str", row["sha256"])) / name
        elif name == "constraints.txt" or (
            source_payloads and (name.endswith(".zip") or name == "official-packs.lock")
        ):
            cached = release / name
        else:
            continue
        check("engine", recorded_engine["release"], row, "name", engine_roots, cached)
    if "dependencies" in engine:
        variant = cast("str", provenance.get("dependencyVariant", recorded_base["variant"]))
        selection = provenance.get("inference") or cast(
            "dict[str, object]", engine["dependencies"]
        ).get(variant)
        validate_dependency_base(
            selection, cast("dict[str, object]", provenance.get("dependencyBase", base))
        )
        package_index = provenance.get("packageIndex")
        if package_index is not None and not isinstance(package_index, str):
            raise EnvironmentError("snapshot package index must be a string or null")
        for row in dependency_wheels(selection):
            if row["name"] in owned:
                continue
            cached = object_path(store / "wheels", "sha256:" + cast("str", row["sha256"])) / cast(
                "str", row["filename"]
            )
            if (
                cached.is_file()
                and not cached.is_symlink()
                and cached.stat().st_size == row["size"]
                and content_digest(cached) == "sha256:" + cast("str", row["sha256"])
            ):
                continue
            try:
                source = dependency_source(row, package_index)
            except (OSError, ValueError) as exc:
                missing.append(
                    {
                        "kind": "engine dependency",
                        "identity": recorded_engine["release"],
                        "artifact": row["filename"],
                        "sha256": row["sha256"],
                        "locations": [package_index or row["url"]],
                        "error": str(exc),
                    }
                )
                continue
            check(
                "engine dependency",
                recorded_engine["release"],
                row,
                "filename",
                (source,),
                cached,
            )
    frontend = select_frontend(engine, cast("dict[str, object]", provenance["channel"]))
    if frontend is not None:
        name = frontend_filename(frontend)
        cached = object_path(store / "wheels", "sha256:" + cast("str", frontend["sha256"])) / name
        check(
            "frontend",
            cast("str", frontend["version"]),
            {**frontend, "filename": name},
            "filename",
            (str(cached.parent) if offline else cast("str", frontend["url"]).rsplit("/", 1)[0],),
            cached,
        )
    return [*missing, *_pack_missing_inputs(snapshot, store, offline=offline)]


def _pack_wheel_requirements(
    row: dict[str, object], roots: tuple[str, ...], store: Path, *, offline: bool
) -> tuple[str, ...]:
    filename, sha, size = (
        cast("str", row["filename"]),
        cast("str", row["sha256"]),
        cast("int", row["size"]),
    )
    with tempfile.TemporaryDirectory(prefix="dinkster-restore-metadata-") as directory:
        source = None
        for candidate in (
            object_path(store / "wheels", "sha256:" + sha) / filename,
            store / "wheels" / filename,
        ):
            if not candidate.is_symlink() and candidate.is_file():
                with candidate.open("rb") as stream:
                    if (
                        candidate.stat().st_size == size
                        and hashlib.file_digest(stream, "sha256").hexdigest() == sha
                    ):
                        source = candidate
                        break
        if source is None:
            if offline:
                raise EnvironmentError(f"offline pack wheel missing or corrupt: {filename}")
            source = fetch_artifact(
                roots,
                filename,
                Path(directory) / filename,
                sha,
                size,
                source_url=cast("str", row["url"])
                if urlparse(cast("str", row["url"])).scheme != "file"
                else None,
            )
        with zipfile.ZipFile(source) as archive:
            names = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA") and len(name.split("/")) == 2
            ]
            if len(names) != 1 or archive.getinfo(names[0]).file_size > 8 * 1024 * 1024:
                raise EnvironmentError(f"wheel requires bounded distribution metadata: {filename}")
            metadata = email.message_from_bytes(archive.read(names[0]))
        return tuple(metadata.get_all("Requires-Dist", []))


def _pack_missing_inputs(
    snapshot: dict[str, object], store: Path, *, offline: bool = False
) -> list[dict[str, object]]:
    metadata = _metadata(snapshot)
    locks = [
        (name, parse_metadata(text.encode("utf-8")))
        for name, text in sorted(metadata.items())
        if name.startswith(("packs/", "inference/")) and name.endswith("/pack-lock.json")
    ]
    if not locks:
        return []
    host = snapshot.get("host")
    tags = cast("dict[str, object]", host).get("wheelTags") if isinstance(host, dict) else None
    if (
        not isinstance(tags, list)
        or not tags
        or any(not isinstance(tag, str) for tag in cast("list[object]", tags))
    ):
        raise EnvironmentError("snapshot requires recorded generation wheel compatibility tags")
    supported = {tag for text in cast("list[str]", tags) for tag in parse_tag(text)}
    base = parse_metadata(metadata["base-manifest.json"].encode("utf-8"))
    engine = parse_metadata(metadata["engine-manifest.json"].encode("utf-8"))
    versions: dict[str, str] = {}
    for document in (base, engine):
        key = "wheels" if "wheels" in document else "artifacts"
        filename_key = "filename" if key == "wheels" else "name"
        for row in artifact_rows(document[key], filename_key):
            filename = cast("str", row[filename_key])
            if filename.endswith(".whl"):
                name, version, _, _ = parse_wheel_filename(filename)
                if document is base or name not in cast("list[str]", base["replaces"]):
                    versions[name] = str(version)
    provenance = parse_metadata(metadata.get("artifact-sources.json", "{}").encode("utf-8"))
    frontend = select_frontend(engine, cast("dict[str, object]", provenance.get("channel", {})))
    if frontend is not None:
        versions["dinkster-frontend"] = cast("str", frontend["version"])
    if "dependencies" in engine:
        variant = cast("str", provenance.get("dependencyVariant", base["variant"]))
        dependency_rows = dependency_wheels(
            provenance.get("inference")
            or cast("dict[str, object]", engine["dependencies"]).get(variant)
        )
        for row in dependency_rows:
            if row["name"] not in cast("list[str]", base["replaces"]):
                versions[cast("str", row["name"])] = cast("str", row["version"])
    if "inference-process.json" in metadata:
        process = parse_metadata(metadata["inference-process.json"].encode("utf-8"))
        parent_versions = process.get("parentVersions", {})
        if not isinstance(parent_versions, dict):
            raise EnvironmentError("recorded inference parent versions must be an object")
        for name, version in cast("dict[str, object]", parent_versions).items():
            if name not in {"torch", "dinkster-inference", "comfy-aimdo", "dinkster-api"} or (
                not isinstance(version, str)
                or name in versions
                and Version(version) != Version(versions[name])
            ):
                raise EnvironmentError("recorded inference parent differs from its snapshot")
            versions[name] = str(Version(version))
    missing: list[dict[str, object]] = []
    in_process = cast("list[str]", snapshot["inProcessPacks"])
    for path, lock in sorted(locks, key=lambda item: item[1]["pack"] not in in_process):
        pack = lock.get("pack")
        shared_inference = path == "inference/shared-inference/pack-lock.json"
        formats = {"dinkster.pack-lock/1", "dinkster.pack-lock/2"}
        if not isinstance(pack, str) or lock.get("format") not in formats:
            raise EnvironmentError("snapshot requires named pack dependency locks")
        if lock.get("format") == "dinkster.pack-lock/2":
            engine_range = lock.get("engine")
            if not isinstance(engine_range, str) or not SpecifierSet(engine_range).contains(
                versions.get("dinkster-api", "0"), prereleases=True
            ):
                raise EnvironmentError("pack engine requirement differs from its snapshot")
        if lock.get("variant") != base["variant"] or not SpecifierSet(
            cast("str", lock["requiresPython"])
        ).contains(cast("str", base["python"])):
            raise EnvironmentError("pack dependency lock does not match the recorded base")
        packages = lock.get("packages")
        if not isinstance(packages, list):
            raise EnvironmentError("pack dependency packages must be an array")
        effective: dict[str, str] = {}
        for item in cast("list[object]", packages):
            if not isinstance(item, dict):
                raise EnvironmentError("pack dependency must be an object")
            package = cast("dict[str, object]", item)
            name, version = package.get("name"), package.get("version")
            if not isinstance(name, str) or not isinstance(version, str):
                raise EnvironmentError("pack dependency requires a name and version")
            effective[name] = version
        changed = any(
            name in versions and Version(versions[name]) != Version(version)
            for name, version in effective.items()
        )
        overrides = {
            name
            for name, version in effective.items()
            if shared_inference
            and name not in {"torch", "dinkster-inference", "comfy-aimdo", "dinkster-api"}
            and name in versions
            and Version(versions[name]) != Version(version)
        }
        effective.update(
            {name: version for name, version in versions.items() if name not in overrides}
        )
        selected: dict[str, tuple[dict[str, object], tuple[str, ...]]] = {}
        pack_missing = len(missing)
        requirements: dict[str, tuple[str, ...]] = {}
        for package in cast("list[dict[str, object]]", packages):
            name, version = cast("str", package["name"]), cast("str", package["version"])
            inherited = versions.get(name)
            if inherited is not None and name not in overrides:
                continue
            wheels = artifact_rows(package.get("wheels"), "filename")
            selected_wheel = next(
                (
                    row
                    for row in wheels
                    if parse_wheel_filename(cast("str", row["filename"]))[3] & supported
                ),
                None,
            )
            if selected_wheel is None:
                missing.append(
                    {
                        "kind": "pack dependency",
                        "identity": pack,
                        "artifact": f"{name}=={version}",
                        "locations": [],
                        "error": "recorded pack dependency is incompatible with the generation",
                    }
                )
                continue
            row = selected_wheel
            filename, sha = cast("str", row["filename"]), cast("str", row["sha256"])
            wheel_name, wheel_version, _, _ = parse_wheel_filename(filename)
            if wheel_name != name or wheel_version != Version(version):
                raise EnvironmentError("pack dependency wheel identity differs from its lock")
            url = row.get("url")
            if (
                not isinstance(url, str)
                or unquote(urlparse(url).path.rsplit("/", 1)[-1]) != filename
            ):
                raise EnvironmentError("pack wheel URL differs from its locked filename")
            parsed = urlparse(url)
            if parsed.scheme == "file" and parsed.netloc in {"", "localhost"}:
                source_path = unquote(parsed.path)
                if os.name == "nt":
                    source_path = source_path.removeprefix("/")
                roots = (str(Path(source_path).parent),)
            else:
                roots = (
                    parsed._replace(
                        path=parsed.path.rsplit("/", 1)[0], query="", fragment=""
                    ).geturl(),
                )
            selected[name] = (row, roots)
            wheel_store = store / "wheels"
            wheel_root = object_path(wheel_store, "sha256:" + sha)
            if wheel_store.is_symlink() or wheel_root.is_symlink():
                raise EnvironmentError("restore refuses linked pack wheel directories")
            if not any(
                artifact_available(
                    (str(cached.parent),) if offline else roots,
                    filename,
                    cached,
                    sha,
                    cast("int", row["size"]),
                    source_url=url if parsed.scheme != "file" else None,
                )
                for cached in (wheel_root / filename, wheel_store / filename)
            ):
                missing.append(
                    {
                        "kind": "pack dependency",
                        "identity": pack,
                        "artifact": filename,
                        "sha256": sha,
                        "locations": [url],
                    }
                )
        pending = [(name, "") for name in selected] if changed else []
        pending.extend((text, "") for text in cast("list[str]", lock["requirements"]))
        visited: set[tuple[str, str]] = set()
        while pending:
            text, extra = pending.pop()
            if (text, extra) in visited:
                continue
            visited.add((text, extra))
            try:
                requirement = Requirement(text)
                name = canonicalize_name(requirement.name)
                if name in {
                    "opencv-python",
                    "opencv-python-headless",
                    "opencv-contrib-python",
                    "opencv-contrib-python-headless",
                }:
                    name = "opencv-contrib-python-headless"
                    requirement.extras.discard("ffmpeg")
                if requirement.marker and not requirement.marker.evaluate(
                    {
                        **cast("dict[str, str]", cast("dict[str, object]", host)["markers"]),
                        "extra": extra,
                    }
                ):
                    continue
                version = effective.get(name)
                if (
                    version is None
                    or not requirement.specifier.contains(version, prereleases=True)
                    or requirement.url is not None
                    and name in versions
                    and name not in overrides
                ):
                    raise EnvironmentError(
                        f"{text}: {name}=={version or 'absent'} does not satisfy the requirement"
                    )
                # Only private wheels can add constraints to the installed parent's closure.
                if changed and len(missing) == pack_missing and name in selected:
                    row, roots = selected[name]
                    if name not in requirements:
                        requirements[name] = _pack_wheel_requirements(
                            row, roots, store, offline=offline
                        )
                    pending.extend(
                        (dependency, context)
                        for dependency in requirements[name]
                        for context in ("", *sorted(requirement.extras))
                    )
            except (OSError, ValueError, zipfile.BadZipFile) as exc:
                missing.append(
                    {
                        "kind": "pack dependency",
                        "identity": pack,
                        "artifact": text,
                        "locations": [],
                        "error": str(exc),
                    }
                )
        receipt = metadata.get(path.removesuffix("pack-lock.json") + "installed-wheels.json")
        if receipt is not None and len(missing) == pack_missing:
            expected = {
                "format": "dinkster.installed-wheels/1",
                "wheels": [
                    {key: row.get(key) for key in ("filename", "size", "sha256", "registry")}
                    for row, _ in selected.values()
                ],
                "inherited": sorted(
                    {
                        cast("str", row["name"])
                        for row in cast("list[dict[str, object]]", packages)
                        if row["name"] in versions and row["name"] not in overrides
                    }
                    | ({"dinkster-api"} if lock.get("engine") is not None else set())
                ),
            }
            if parse_metadata(receipt.encode("utf-8")) != expected:
                missing.append(
                    {
                        "kind": "pack dependency",
                        "identity": pack,
                        "artifact": "installed-wheels.json",
                        "locations": [],
                        "error": "recorded pack wheel receipt differs from the generation",
                    }
                )
        if pack in in_process and len(missing) == pack_missing:
            versions.update(effective)
    return missing


def _local_missing_inputs(
    snapshot: dict[str, object], store: Path, *, offline: bool = False
) -> list[dict[str, object]]:
    metadata = _metadata(snapshot)
    if not {"engine-manifest.json", "base-manifest.json", "constraints.txt"} <= metadata.keys():
        raise EnvironmentError("local snapshot is missing recorded software metadata")
    engine = parse_metadata(metadata["engine-manifest.json"].encode("utf-8"))
    base = parse_metadata(metadata["base-manifest.json"].encode("utf-8"))
    engine_record = cast("dict[str, object]", snapshot["engine"])
    base_record = cast("dict[str, object]", snapshot["base"])
    engine_id, base_id = digest(engine_record["release"]), digest(base_record["id"])
    for path in (
        store / "bases",
        object_path(store / "bases", base_id),
        store / "engines",
        object_path(store / "engines", engine_id),
        store / "wheels",
    ):
        if path.is_symlink():
            raise EnvironmentError("restore refuses linked local input directories")
    base_identity = dict(base)
    base_identity.pop("id", None)
    for document, expected in (
        (engine, engine_record["release"]),
        (base_identity, base_record["id"]),
    ):
        identity = (
            "sha256:" + hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()
        )
        if document.get("source") != "local" or identity != expected:
            raise EnvironmentError("local snapshot manifest identity mismatch")
    if (
        engine.get("version") != engine_record["version"]
        or base.get("id") != base_record["id"]
        or base.get("variant") != base_record["variant"]
        or engine.get("workerProtocol") != snapshot.get("workerProtocol")
        or base.get("workerProtocol") != snapshot.get("workerProtocol")
    ):
        raise EnvironmentError("local snapshot software identities disagree")
    missing: list[dict[str, object]] = []
    interpreter = base.get("interpreter")
    files = (
        cast("dict[str, object]", interpreter).get("files")
        if isinstance(interpreter, dict)
        else None
    )
    if not isinstance(files, dict) or not files:
        raise EnvironmentError("local snapshot requires recorded interpreter file hashes")
    expected_files: dict[str, str] = {}
    for name, value in cast("dict[str, object]", files).items():
        if not name.startswith("python/"):
            raise EnvironmentError("local snapshot has invalid interpreter paths")
        relative_file(name)
        expected_files[name] = digest(value)
    base_root = object_path(store / "bases", base_id)
    try:
        verify_base_inputs(base_root / "python", expected_files, operation="restore")
        for name, expected in expected_files.items():
            if content_digest(base_root / name) != expected:
                raise EnvironmentError(f"restore requires intact local base input: {name}")
    except (OSError, ValueError) as exc:
        missing.append(
            {
                "kind": "local base",
                "identity": base_record["id"],
                "artifact": "python",
                "locations": [],
                "error": str(exc),
            }
        )
    for document, identity in ((base, base_record["id"]), (engine, engine_record["release"])):
        for row in artifact_rows(document.get("wheels"), "filename"):
            filename, sha = cast("str", row["filename"]), cast("str", row["sha256"])
            wheel_root = object_path(store / "wheels", "sha256:" + sha)
            if wheel_root.is_symlink():
                raise EnvironmentError("restore refuses linked local wheel directories")
            url = row.get("url")
            roots = (
                (url.rsplit("/", 1)[0],)
                if isinstance(url, str) and url.startswith("https://")
                else (str(wheel_root),)
            )
            if not artifact_available(
                (str(wheel_root),) if offline else roots,
                filename,
                wheel_root / filename,
                sha,
                cast("int", row["size"]),
            ):
                missing.append(
                    {
                        "kind": "local wheel",
                        "identity": identity,
                        "artifact": filename,
                        "sha256": sha,
                        "locations": list(roots),
                    }
                )
    return missing


def _substitutions(before: dict[str, object], after: dict[str, object]) -> list[dict[str, object]]:
    fields = ["engine", "base", "constraints", "inProcessPacks", "isolatedInferencePacks", "source"]
    changes = [
        {"field": key, "before": before.get(key), "after": after.get(key)}
        for key in fields
        if before.get(key) != after.get(key)
    ]
    old, new = _metadata(before), _metadata(after)
    for key in sorted(old.keys() | new.keys()):
        if key == "dinkster.lock" or key.startswith("packs/"):
            if old.get(key) != new.get(key):
                changes.append(
                    {"field": f"metadata.{key}", "before": old.get(key), "after": new.get(key)}
                )
    return changes


def _replacement_wheel_receipt(
    snapshot: dict[str, object], engine: dict[str, object], base: dict[str, object], variant: str
) -> str:
    host = cast("dict[str, object]", snapshot["host"])
    tags, markers = host.get("wheelTags"), host.get("markers")
    if not isinstance(tags, list) or not isinstance(markers, dict):
        raise EnvironmentError("replacement preview requires recorded wheel compatibility")
    owned = cast("list[str]", base["replaces"])
    rows = [
        {"filename": row["filename"], "size": row["size"], "sha256": row["sha256"]}
        for row in artifact_rows(base["wheels"], "filename")
    ]
    engine_rows = [
        {"filename": row["name"], "size": row["size"], "sha256": row["sha256"]}
        for row in artifact_rows(engine["artifacts"], "name")
        if cast("str", row["name"]).endswith(".whl")
    ]
    metadata = _metadata(snapshot)
    provenance = parse_metadata(metadata["artifact-sources.json"].encode("utf-8"))
    inference = cast("dict[str, object] | None", provenance.get("inference"))
    if "dependencies" in engine:
        engine_rows.extend(
            dependency_wheels(
                inference or cast("dict[str, object]", engine["dependencies"]).get(variant)
            )
        )
    rows.extend(
        row
        for row in engine_rows
        if parse_wheel_filename(cast("str", row["filename"]))[0] not in owned
    )
    channel = cast("dict[str, object]", provenance["channel"])
    frontend = select_frontend(engine, channel)
    if frontend is not None:
        rows.append({**frontend, "filename": frontend_filename(frontend)})
    wheels = selected_wheel_rows(
        metadata["constraints.txt"], rows, {"tags": tags, "markers": markers}
    )
    wheels = installed_wheel_origins(wheels, base, engine, variant, channel, inference)
    return (
        json.dumps(
            {"format": "dinkster.installed-wheels/1", "wheels": wheels, "inherited": []},
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n"
    )


def _engine_alternatives(snapshot: dict[str, object]) -> list[dict[str, object]]:
    if snapshot.get("source") != "published":
        return []
    metadata = _metadata(snapshot)
    provenance = parse_metadata(metadata["artifact-sources.json"].encode("utf-8"))
    mirrors = _sources(provenance, "mirrors")
    base_record = cast("dict[str, str]", snapshot["base"])
    base = validate_manifest(
        metadata["base-manifest.json"].encode("utf-8"), base_record["id"], base_record["variant"]
    )
    variant = cast("str", provenance.get("dependencyVariant", base_record["variant"]))
    source_lock = parse_metadata(metadata.get("dinkster.lock", '{"packs":[]}').encode("utf-8"))
    packs = cast("list[dict[str, object]]", source_lock.get("packs", []))
    ranges: list[SpecifierSet] = []
    for pack in packs:
        text = metadata.get(f"pack-sources/{pack['pack']}/dinkster-pack.toml")
        if text is None:
            return []
        declaration = tomllib.loads(text).get("pack")
        if not isinstance(declaration, dict):
            return []
        fields = cast("dict[str, object]", declaration)
        name, version, engine_range = (
            fields.get("name"),
            fields.get("version"),
            fields.get("engine"),
        )
        if (
            not isinstance(name, str)
            or canonicalize_name(name) != pack["pack"]
            or version != pack["version"]
            or not isinstance(engine_range, str)
            or not str(SpecifierSet(engine_range))
        ):
            return []
        ranges.append(SpecifierSet(engine_range))
    if not packs and any(name.startswith("packs/") for name in metadata):
        return []
    publications: list[tuple[dict[str, object], tuple[str, ...], bytes]] = []
    for mirror in mirrors:
        try:
            trusted_metadata(mirror)
            index: object = json.loads(read_metadata(mirror, "engine/index.json"))
            if not isinstance(index, list):
                raise EnvironmentError("engine index must be an array")
        except (OSError, ValueError):
            index = []
        for item in cast("list[object]", index):
            if not isinstance(item, dict):
                continue
            row = cast("dict[str, object]", item)
            path, sha = row.get("path"), row.get("sha256")
            if not isinstance(path, str) or not isinstance(sha, str):
                continue
            try:
                relative_file(path)
                if len(path.split("/")) != 2 or not path.startswith("engine/"):
                    continue
                data = read_metadata(mirror, path + "/release-manifest.json")
                engine = validate_manifest(data, "sha256:" + sha, None)
                publications.append((engine, (mirror.rstrip("/") + "/" + path,), data))
            except (OSError, ValueError):
                continue
        for channel in ("nightly", "stable"):
            try:
                catalog = discover((mirror,), channel)
                publications.append((catalog.engine, catalog.engine_sources, catalog.engine_bytes))
            except (OSError, ValueError):
                continue
    alternatives: list[dict[str, object]] = []
    for engine, roots, data in publications:
        if engine["workerProtocol"] != snapshot.get("workerProtocol") or any(
            not allowed.contains(cast("str", engine["version"]), prereleases=True)
            for allowed in ranges
        ):
            continue
        try:
            if provenance.get("inference") is not None:
                original_engine = parse_metadata(metadata["engine-manifest.json"].encode("utf-8"))
                closures = [
                    {
                        row["name"]: row
                        for row in dependency_wheels(
                            cast("dict[str, object]", document.get("dependencies", {})).get(variant)
                        )
                        if row["name"] != "dinkster-inference"
                    }
                    for document in (original_engine, engine)
                ]
                # The retained override was resolved against the original engine's closure.
                if closures[0] != closures[1]:
                    continue
            if "dependencies" in engine:
                requirements = frontend_requirements(
                    engine_constraints(
                        engine,
                        variant,
                        inference=cast("dict[str, object] | None", provenance.get("inference")),
                    ),
                    engine,
                    cast("dict[str, object]", provenance["channel"]),
                )
            else:
                requirements = next(
                    read_metadata(root, "constraints.txt").decode("utf-8") for root in roots
                )
            candidate = copy.deepcopy(snapshot)
            candidate_metadata = dict(metadata)
            candidate_provenance = copy.deepcopy(provenance)
            document = cast("dict[str, object]", candidate_provenance["channel"])
            identity = "sha256:" + hashlib.sha256(data).hexdigest()
            document["engine"] = {
                "version": engine["version"],
                "release": identity,
                "locations": [
                    {"type": "mirror", "path": f"engine/{identity[7:]}/release-manifest.json"}
                ],
            }
            Channel.parse(document)
            candidate_provenance["engine"] = list(roots)
            if packs:
                candidate_provenance["packSourceRelease"] = provenance.get(
                    "packSourceRelease", cast("dict[str, object]", snapshot["engine"])["release"]
                )
            candidate_metadata["artifact-sources.json"] = (
                json.dumps(candidate_provenance, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
            )
            candidate_metadata["engine-manifest.json"] = data.decode("utf-8")
            candidate_metadata["constraints.txt"] = requirements
            candidate["engine"] = {"version": engine["version"], "release": identity}
            candidate["metadata"] = candidate_metadata
            # A replacement engine may require a base the historical engine cannot use.
            alternatives.extend(_base_alternatives(candidate))
            if "dependencies" in engine:
                selection = provenance.get("inference") or cast(
                    "dict[str, object]", engine["dependencies"]
                ).get(variant)
                validate_dependency_base(selection, base)
            constraints = effective_constraints(requirements, base, strict="dependencies" in engine)
            candidate_metadata["constraints.txt"] = constraints
            candidate["constraints"] = (
                "sha256:" + hashlib.sha256(constraints.encode("utf-8")).hexdigest()
            )
            candidate_metadata["installed-wheels.json"] = _replacement_wheel_receipt(
                candidate, engine, base, variant
            )
            alternatives.append(candidate)
        except (OSError, ValueError):
            continue
    return alternatives


def _base_alternatives(snapshot: dict[str, object]) -> list[dict[str, object]]:
    if snapshot.get("source") != "published":
        return []
    metadata = _metadata(snapshot)
    provenance = parse_metadata(metadata["artifact-sources.json"].encode("utf-8"))
    recorded_base = cast("dict[str, str]", snapshot["base"])
    variant = recorded_base["variant"]
    original = validate_manifest(
        metadata["base-manifest.json"].encode("utf-8"), recorded_base["id"], variant
    )
    engine_record = cast("dict[str, str]", snapshot["engine"])
    engine = validate_manifest(
        metadata["engine-manifest.json"].encode("utf-8"), engine_record["release"], None
    )
    publications: dict[str, tuple[dict[str, object], tuple[str, ...]]] = {}
    for mirror in _sources(provenance, "mirrors"):
        try:
            trusted_metadata(mirror)
            index: object = json.loads(read_metadata(mirror, f"base/{variant}/index.json"))
            if not isinstance(index, list):
                raise EnvironmentError("base index must be an array")
        except (OSError, ValueError):
            index = []
        for item in cast("list[object]", index):
            if not isinstance(item, dict):
                continue
            row = cast("dict[str, object]", item)
            path, identity = row.get("path"), row.get("id")
            if not isinstance(path, str) or not isinstance(identity, str):
                continue
            try:
                relative_file(path)
                if len(path.split("/")) != 3 or not path.startswith(f"base/{variant}/"):
                    continue
                data = read_metadata(mirror, path + "/manifest.json")
                base = validate_manifest(data, identity, variant)
                name = path.rsplit("/", 1)[-1]
                if name != identity[7:] and base.get("label") != name:
                    continue
                publications[identity] = (base, (mirror.rstrip("/") + "/" + path,))
            except (OSError, ValueError):
                continue
        for channel in ("nightly", "stable"):
            try:
                catalog = discover((mirror,), channel)
                base = catalog.bases[variant]
                identity = cast("str", base["id"])
                publications[identity] = (base, catalog.base_sources[variant])
            except (OSError, ValueError, KeyError):
                continue
    alternatives: list[dict[str, object]] = []
    for identity, (base, roots) in sorted(publications.items()):
        if (
            identity == recorded_base["id"]
            or base.get("platform") != original.get("platform")
            or base.get("python") != original.get("python")
            or base.get("workerProtocol") != snapshot.get("workerProtocol")
        ):
            continue
        try:
            inference = cast("dict[str, object] | None", provenance.get("inference"))
            if inference is not None:
                inference = {**inference, "base": identity}
            if "dependencies" in engine:
                selection = inference or cast("dict[str, object]", engine["dependencies"]).get(
                    variant
                )
                validate_dependency_base(selection, base)
                requirements = frontend_requirements(
                    engine_constraints(engine, variant, inference=inference),
                    engine,
                    cast("dict[str, object]", provenance["channel"]),
                )
            else:
                requirements = metadata["constraints.txt"]
            constraints = effective_constraints(requirements, base, strict="dependencies" in engine)
            candidate = copy.deepcopy(snapshot)
            candidate_metadata = dict(metadata)
            candidate_provenance = copy.deepcopy(provenance)
            channel = cast("dict[str, object]", candidate_provenance["channel"])
            bases = cast("dict[str, object]", channel["base"])
            bases[variant] = {
                "id": identity,
                "locations": [
                    {"type": "mirror", "path": f"base/{variant}/{identity[7:]}/manifest.json"}
                ],
            }
            Channel.parse(channel)
            candidate_provenance.update(
                base=list(roots), dependencyBase=base, dependencyVariant=variant
            )
            if inference is not None:
                candidate_provenance["inference"] = inference
            candidate_metadata["artifact-sources.json"] = (
                json.dumps(candidate_provenance, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
            )
            candidate_metadata["base-manifest.json"] = (
                json.dumps(base, sort_keys=True, ensure_ascii=True) + "\n"
            )
            candidate_metadata["constraints.txt"] = constraints
            candidate["base"] = {"id": identity, "variant": variant}
            candidate["constraints"] = (
                "sha256:" + hashlib.sha256(constraints.encode("utf-8")).hexdigest()
            )
            candidate["metadata"] = candidate_metadata
            candidate_metadata["installed-wheels.json"] = _replacement_wheel_receipt(
                candidate, engine, base, variant
            )
            alternatives.append(candidate)
        except (OSError, ValueError):
            continue
    return alternatives


def plan_restore(root: Path, entry_id: int) -> dict[str, object]:
    entries = history(root)
    selected = next((entry for entry in entries if entry["id"] == entry_id), None)
    if selected is None:
        raise EnvironmentError(f"installation history has no entry {entry_id}")
    validate_import(root, selected)
    snapshot = cast("dict[str, object]", selected["snapshot"])
    development = _metadata(snapshot).get("development-sources.json")
    if development is not None and any(
        row.get("live") for row in json.loads(development)["sources"]
    ):
        raise EnvironmentError(
            "live sources change with their folders and cannot be restored exactly; "
            "run develop without live marks to capture a restorable generation"
        )
    store = Path((root / "store").read_text(encoding="utf-8").strip())
    if not store.is_absolute() or store.is_symlink() or (root / "store").is_symlink():
        raise EnvironmentError("restore requires an absolute unlinked store")
    observed: dict[str, object] = {
        "current": read_current(root),
        "history": _identity(entries),
        "settings": content_digest(root / "environment.json"),
        "store": content_digest(root / "store"),
    }
    sources: dict[str | None, list[dict[str, object]]] = {}

    def unavailable_inputs(candidate: dict[str, object]) -> list[dict[str, object]]:
        source_lock = _metadata(candidate).get("dinkster.lock")
        if source_lock not in sources:
            sources[source_lock] = _pack_source_missing(candidate, store, root)
        return [*missing_inputs(candidate, store), *sources[source_lock]]

    missing = unavailable_inputs(snapshot)
    candidates: list[dict[str, object]] = [snapshot]
    if missing:
        candidates = []
        for entry in entries:
            try:
                validate_import(root, entry)
            except EnvironmentError:
                continue
            candidates.append(cast("dict[str, object]", entry["snapshot"]))
        candidates.extend(_engine_alternatives(snapshot))
        candidates.extend(_base_alternatives(snapshot))
    viable: dict[str, dict[str, object]] = {}
    for candidate in candidates:
        if candidate.get("source") not in {"published", "local", "source"}:
            continue
        try:
            development = _metadata(candidate).get("development-sources.json")
            if development is not None and any(
                row.get("live") for row in json.loads(development)["sources"]
            ):
                continue
            unavailable = unavailable_inputs(candidate)
        except (OSError, ValueError, KeyError):
            continue
        if unavailable:
            continue
        substitutions = _substitutions(snapshot, candidate)
        if missing and not substitutions:
            continue
        alternative: dict[str, object] = {"snapshot": candidate, "substitutions": substitutions}
        alternative["approval"] = _identity(
            {"observed": observed, "entry": entry_id, "missing": missing, **alternative}
        )
        viable.setdefault(_identity(substitutions), alternative)
    minimum = min(
        (len(cast("list[object]", candidate["substitutions"])) for candidate in viable.values()),
        default=0,
    )
    alternatives = [
        candidate
        for _, candidate in sorted(viable.items())
        if len(cast("list[object]", candidate["substitutions"])) == minimum
    ]
    return {
        "format": "dinkster.restore-plan/1",
        "entry": entry_id,
        "observed": observed,
        "missing": missing,
        "alternatives": alternatives,
        "exact": not missing,
    }
