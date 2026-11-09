"""Read-only restore availability and explicitly approved minimal alternatives."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import tomllib
from pathlib import Path
from typing import cast
from urllib.parse import unquote, urlparse

from dinkster_store import object_path
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
from .environment_constraints import effective_constraints
from .environment_downloads import artifact_available, read_metadata
from .environment_generations import retained_pack_archive, verify_base_inputs
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
    snapshot: dict[str, object], store: Path, root: Path
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
    # Only acquisition runs in the active engine: no pack imports, provisioning or activation.
    script = """
import json, os, sys
from pathlib import Path
from dinkster.installer import Installer
from dinkster.registries import (
    RegistryConfig, load_registries, routing_registry_fetcher, select_registry,
)
from dinkster_registry import Lockfile
project = Path(sys.argv[1])
config_path = project / 'registries.toml'
if config_path.is_file():
    config = load_registries(config_path)
elif os.environ.get('DINKSTER_REGISTRIES'):
    config = load_registries(Path(os.environ['DINKSTER_REGISTRIES']))
else:
    config = RegistryConfig()
selector = os.environ.get('DINKSTER_REGISTRY', '')
if not selector and not config_path.is_file() and not os.environ.get('DINKSTER_REGISTRIES'):
    selector = 'https://registry.dinkster.org'
selected = select_registry(config, selector) if selector else None
installer = Installer(
    Path.cwd(), accelerator='cpu', registry_fetch=routing_registry_fetcher(config, selected),
)
missing = []
for entry in Lockfile.from_record_json(sys.stdin.read()).packs:
    try:
        installer.acquire(entry)
    except Exception:
        missing.append(entry.artifact_digest)
print(json.dumps(missing))
"""
    unavailable = {cast("str", row["artifactDigest"]) for row in needed}
    current = read_current(root)
    if current is not None:
        try:
            with tempfile.TemporaryDirectory(prefix="dinkster-source-probe-") as directory:
                with subprocess.Popen(
                    [
                        str(interpreter(root / "generations" / str(current))),
                        "-I",
                        "-B",
                        "-c",
                        script,
                        str(root),
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
                if isinstance(reported, list) and all(
                    isinstance(item, str) and item in unavailable
                    for item in cast("list[object]", reported)
                ):
                    unavailable = set(cast("list[str]", reported))
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


def missing_inputs(snapshot: dict[str, object], store: Path) -> list[dict[str, object]]:
    """Prefer verified retained payloads; remote probes never populate the store."""
    if snapshot.get("source") == "local":
        return [*_local_missing_inputs(snapshot, store), *_pack_missing_inputs(snapshot, store)]
    if snapshot.get("source") != "published":
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
        selection = cast("dict[str, object]", engine["dependencies"]).get(variant)
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
    return [*missing, *_pack_missing_inputs(snapshot, store)]


def _pack_missing_inputs(snapshot: dict[str, object], store: Path) -> list[dict[str, object]]:
    metadata = _metadata(snapshot)
    locks = [
        parse_metadata(text.encode("utf-8"))
        for name, text in sorted(metadata.items())
        if name.startswith("packs/") and name.endswith("/pack-lock.json")
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
    if "dependencies" in engine:
        provenance = parse_metadata(metadata["artifact-sources.json"].encode("utf-8"))
        variant = cast("str", provenance.get("dependencyVariant", base["variant"]))
        selected = dependency_wheels(cast("dict[str, object]", engine["dependencies"]).get(variant))
        for row in selected:
            if row["name"] not in cast("list[str]", base["replaces"]):
                versions[cast("str", row["name"])] = cast("str", row["version"])
    missing: list[dict[str, object]] = []
    in_process = cast("list[str]", snapshot["inProcessPacks"])
    for lock in sorted(locks, key=lambda lock: lock["pack"] not in in_process):
        pack = lock.get("pack")
        if not isinstance(pack, str) or lock.get("format") != "dinkster.pack-lock/1":
            raise EnvironmentError("snapshot requires named pack dependency locks")
        if lock.get("variant") != base["variant"] or not SpecifierSet(
            cast("str", lock["requiresPython"])
        ).contains(cast("str", base["python"])):
            raise EnvironmentError("pack dependency lock does not match the recorded base")
        packages = lock.get("packages")
        if not isinstance(packages, list):
            raise EnvironmentError("pack dependency packages must be an array")
        for item in cast("list[object]", packages):
            if not isinstance(item, dict):
                raise EnvironmentError("pack dependency must be an object")
            package = cast("dict[str, object]", item)
            name, version = package.get("name"), package.get("version")
            if not isinstance(name, str) or not isinstance(version, str):
                raise EnvironmentError("pack dependency requires a name and version")
            inherited = versions.get(name)
            if inherited is not None and Version(inherited) == Version(version):
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
            if inherited is not None or selected_wheel is None:
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
                roots = (url.rsplit("/", 1)[0],)
            wheel_store = store / "wheels"
            wheel_root = object_path(wheel_store, "sha256:" + sha)
            if wheel_store.is_symlink() or wheel_root.is_symlink():
                raise EnvironmentError("restore refuses linked pack wheel directories")
            if not any(
                artifact_available(
                    roots,
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
        if pack in in_process:
            versions.update(
                {
                    cast("str", row["name"]): cast("str", row["version"])
                    for row in cast("list[dict[str, object]]", packages)
                }
            )
    return missing


def _local_missing_inputs(snapshot: dict[str, object], store: Path) -> list[dict[str, object]]:
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
                roots, filename, wheel_root / filename, sha, cast("int", row["size"])
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
    fields = ["engine", "base", "constraints", "inProcessPacks", "source"]
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
        trusted_metadata(mirror)
        try:
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
        for channel in ("main", "stable"):
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
            if "dependencies" in engine:
                requirements = engine_constraints(engine, variant)
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
                selection = cast("dict[str, object]", engine["dependencies"]).get(variant)
                validate_dependency_base(selection, base)
            constraints = effective_constraints(requirements, base, strict="dependencies" in engine)
            candidate_metadata["constraints.txt"] = constraints
            candidate["constraints"] = (
                "sha256:" + hashlib.sha256(constraints.encode("utf-8")).hexdigest()
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
        trusted_metadata(mirror)
        try:
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
        for channel in ("main", "stable"):
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
            if "dependencies" in engine:
                selection = cast("dict[str, object]", engine["dependencies"]).get(variant)
                validate_dependency_base(selection, base)
                requirements = engine_constraints(engine, variant)
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
    store = Path((root / "store").read_text(encoding="utf-8").strip())
    if not store.is_absolute() or store.is_symlink() or (root / "store").is_symlink():
        raise EnvironmentError("restore requires an absolute unlinked store")
    observed: dict[str, object] = {
        "current": read_current(root),
        "history": _identity(entries),
        "settings": content_digest(root / "project.json"),
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
        if candidate.get("source") not in {"published", "local"}:
            continue
        try:
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
