"""Read-only restore availability and explicitly approved minimal alternatives."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import cast

from packaging.utils import parse_wheel_filename

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
from .environment_generations import verify_base_inputs
from .environment_history import history
from .environment_state import (
    Channel,
    EnvironmentError,
    content_digest,
    digest,
    read_current,
    relative_file,
)


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
        return _local_missing_inputs(snapshot, store)
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
    base_cache = store / "bases" / recorded_base["id"][7:]
    if not base_cache.is_dir() or base_cache.is_symlink():
        for row in artifact_rows(cast("dict[str, object]", base["archive"])["parts"], "filename"):
            check(
                "base",
                recorded_base["id"],
                row,
                "filename",
                base_roots,
                store / "downloads" / recorded_base["id"][7:] / cast("str", row["filename"]),
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
    release = store / "engines" / recorded_engine["release"][7:]
    source_payloads = any(
        row["name"] == "official-packs.lock" for row in artifact_rows(engine["artifacts"], "name")
    )
    for row in artifact_rows(engine["artifacts"], "name"):
        name = cast("str", row["name"])
        if name.endswith(".whl"):
            if parse_wheel_filename(name)[0] in owned:
                continue
            cached = store / "wheels" / cast("str", row["sha256"]) / name
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
            cached = store / "wheels" / cast("str", row["sha256"]) / cast("str", row["filename"])
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
                store / "wheels" / cast("str", row["sha256"]) / cast("str", row["filename"]),
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
        store / "bases" / base_id[7:],
        store / "engines",
        store / "engines" / engine_id[7:],
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
    base_root = store / "bases" / cast("str", base_record["id"])[7:]
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
            if (store / "wheels" / sha).is_symlink():
                raise EnvironmentError("restore refuses linked local wheel directories")
            url = row.get("url")
            roots = (
                (url.rsplit("/", 1)[0],)
                if isinstance(url, str) and url.startswith("https://")
                else (str(store / "wheels" / sha),)
            )
            if not artifact_available(
                roots, filename, store / "wheels" / sha / filename, sha, cast("int", row["size"])
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
    # Nonempty packs need target-owned source and lock compatibility evidence.
    source_lock = parse_metadata(metadata.get("dinkster.lock", '{"packs":[]}').encode("utf-8"))
    if source_lock.get("packs") or any(name.startswith("packs/") for name in metadata):
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
        if engine["workerProtocol"] != snapshot.get("workerProtocol"):
            continue
        try:
            if "dependencies" in engine:
                selection = cast("dict[str, object]", engine["dependencies"]).get(variant)
                validate_dependency_base(selection, base)
                requirements = engine_constraints(engine, variant)
            else:
                requirements = next(
                    read_metadata(root, "constraints.txt").decode("utf-8") for root in roots
                )
            constraints = effective_constraints(requirements, base, strict="dependencies" in engine)
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
            candidate_metadata["artifact-sources.json"] = (
                json.dumps(candidate_provenance, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
            )
            candidate_metadata["engine-manifest.json"] = data.decode("utf-8")
            candidate_metadata["constraints.txt"] = constraints
            candidate["engine"] = {"version": engine["version"], "release": identity}
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
    missing = missing_inputs(snapshot, store)
    candidates = (
        [snapshot]
        if not missing
        else [
            *[cast("dict[str, object]", entry["snapshot"]) for entry in entries],
            *_engine_alternatives(snapshot),
        ]
    )
    viable: dict[str, dict[str, object]] = {}
    for candidate in candidates:
        if candidate.get("source") not in {"published", "local"}:
            continue
        try:
            unavailable = missing_inputs(candidate, store)
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
