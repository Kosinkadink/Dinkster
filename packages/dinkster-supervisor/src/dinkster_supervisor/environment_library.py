"""Library compatibility disclosure for software-only environment plans."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import cast

from dinkster_store.library_compatibility import (
    engine_readers,
    library_requirements,
    minimum_release,
)
from packaging.version import InvalidVersion, Version

from .environment_history import history
from .environment_state import EnvironmentError, interpreter, library_root


def library_list(
    root: Path,
    kind: str,
    *,
    project: str = "default",
    limit: int = 50,
    cursor: str | None = None,
) -> dict[str, object]:
    """Read one library page and reap the tools helper before returning."""
    from dinkster_library import LibraryClient, LibraryError

    operations = {"projects": "projects.list", "workflows": "records.list", "assets": "assets.list"}
    keys = {"projects": "items", "workflows": "records", "assets": "assets"}
    if kind not in operations or not 1 <= limit <= 200:
        raise EnvironmentError("invalid library list or page size")
    configured = library_root(root)
    if not configured:
        return {"kind": kind, keys[kind]: [], "cursor": None}
    arguments: dict[str, object] = {"limit": limit}
    if cursor is not None:
        arguments["cursor"] = cursor
    if kind != "projects":
        arguments.update(scope="local", project=project)
    if kind == "workflows":
        arguments["mediaType"] = "application/x-dinkster-workflow+json"
    try:
        try:
            (Path(configured) / "records.sqlite").stat()
        except FileNotFoundError:
            try:
                (Path(configured) / "library.sqlite").stat()
            except FileNotFoundError:
                return {"kind": kind, keys[kind]: [], "cursor": None}
        with LibraryClient(Path(configured)) as helper:
            return {"kind": kind, **helper.call(operations[kind], **arguments)}
    except (LibraryError, OSError, ValueError) as error:
        raise EnvironmentError(
            "By the way, these projects and files cannot be shown with the installed tools."
        ) from error


def library_api_impact(root: Path, low: int, high: int) -> list[str]:
    """List local restoration losses before an authorized helper API removal."""
    from dinkster_library import LibraryClient, LibraryError

    if not 0 < low <= high:
        raise EnvironmentError("supported API range must be positive and ordered")
    configured = library_root(root)
    if not configured or not (Path(configured) / "records.sqlite").is_file():
        return []
    messages: list[str] = []
    try:
        with LibraryClient(Path(configured)) as helper:
            cursor: object = None
            while True:
                page = helper.call("compatibility.affected", apiMin=low, apiMax=high, cursor=cursor)
                for item in cast("list[dict[str, object]]", page["items"]):
                    if item["kind"] == "project":
                        messages.append(
                            f"By the way, project {item['name']} uses snapshot {item['snapshot']} "
                            f"in {item['environment']}, which will no longer open "
                            "after this tools update."
                        )
                    else:
                        messages.append(
                            f"By the way, snapshot {item['snapshot']} in {item['environment']} "
                            "will no longer open after this tools update."
                        )
                cursor = page["cursor"]
                if cursor is None:
                    break
    except LibraryError as error:
        raise EnvironmentError(f"Cannot check snapshots before updating tools: {error}") from error
    return messages


def library_plan(
    root: Path,
    target_version: str | None = None,
    *,
    source: Path | None = None,
    python: Path | None = None,
) -> list[str]:
    library = library_root(root)
    if not library:
        return ["library persistence is disabled"]
    revisions, hidden = library_requirements(Path(library))
    if source is not None or python is not None:
        prospective, _ = engine_readers(source, python=python)
        revisions = tuple(sorted(set(revisions) | set(prospective)))
    release = minimum_release(revisions, hidden)
    if release is None:
        return [
            "library compatibility: earliest release is unknown for schema "
            + ", ".join(sorted(revisions))
            + "; retained snapshot compatibility cannot be established"
        ]
    minimum = Version(release)
    unknown_target = False
    if target_version is not None and source is None and python is None:
        try:
            target = Version(target_version)
        except InvalidVersion:
            target = None
        unknown_target = target is None or not Version("0.0.1") <= target <= Version("0.0.15")
        if not unknown_target and target is not None:
            # Released engines from 0.0.4 through 0.0.15 migrate to revision 0011.
            if target >= Version("0.0.4"):
                minimum = max(minimum, Version("0.0.4"))
                revisions = (*revisions, "0011_retention_clients")
            elif target >= Version("0.0.3"):
                minimum = max(minimum, Version("0.0.3"))
                revisions = (*revisions, "0010_result_renditions")
            elif target >= Version("0.0.2"):
                minimum = max(minimum, Version("0.0.2"))
                revisions = (*revisions, "0002_asset_observations")
    plan = [f"library after apply requires Dinkster {minimum} or newer"]
    if unknown_target:
        plan = [
            f"current library requires Dinkster {minimum} or newer; "
            f"earliest release after applying engine {target_version} is unknown"
        ]
    excluded: list[str] = []
    uncertain: list[str] = []
    try:
        entries = history(root)
    except (OSError, ValueError) as exc:
        return [*plan, f"retained snapshot compatibility cannot be established: {exc}"]
    for entry in entries:
        snapshot = cast("dict[str, object]", entry["snapshot"])
        raw_engine = snapshot.get("engine")
        engine = cast("dict[str, object]", raw_engine) if isinstance(raw_engine, dict) else {}
        if not isinstance(engine.get("version"), str):
            uncertain.append(f"history entry {entry['id']} (invalid engine metadata)")
            continue
        version = str(engine["version"])
        python = interpreter(root / "builds" / str(entry.get("generation", "")))
        if python.is_file():
            try:
                supported, supports_hidden = engine_readers(python=python)
            except (OSError, ValueError, subprocess.SubprocessError):
                uncertain.append(f"history entry {entry['id']} (Dinkster {version})")
                continue
            older = bool(set(revisions) - set(supported)) or (hidden and not supports_hidden)
        else:
            metadata = cast("dict[str, object]", snapshot.get("metadata", {}))
            try:
                parsed = Version(version)
            except InvalidVersion:
                parsed = None
            if (
                parsed is None
                or parsed.is_prerelease
                or parsed.local is not None
                or "development-sources.json" in metadata
                or snapshot.get("source") == "source"
            ):
                uncertain.append(f"history entry {entry['id']} (Dinkster {version})")
                continue
            older = parsed < minimum
        if older:
            excluded.append(f"history entry {entry['id']} (Dinkster {version})")
    plan.append(
        "library excludes retained snapshots: " + "; ".join(excluded)
        if excluded
        else "library excludes no known retained release snapshots"
    )
    if uncertain:
        plan.append(
            "library reader must be checked for retained snapshots: " + "; ".join(uncertain)
        )
    return plan
