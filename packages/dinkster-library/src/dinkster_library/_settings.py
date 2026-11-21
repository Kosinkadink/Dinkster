"""Revision-checked preferences, patched without replacing unseen fields."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ._records import object_value
from ._workflows import text
from .client import LibraryError

if TYPE_CHECKING:
    from ._records import Records

MAX_SETTINGS_REPLY = 1024 * 1024


def _merge(current: dict[str, object], patch: dict[str, object]) -> None:
    for name, value in patch.items():
        previous = current.get(name)
        if isinstance(value, dict) and isinstance(previous, dict):
            _merge(cast("dict[str, object]", previous), cast("dict[str, object]", value))
        else:
            current[name] = value


def model_preferences(operation: str, arguments: dict[str, object]) -> dict[str, object]:
    if operation not in ("folders.models.get", "folders.models.patch"):
        raise LibraryError("operation", f"unknown library operation: {operation}")
    home = Path(os.environ.get("DINKSTER_HOME", Path.home() / ".dinkster")).expanduser()
    if any(
        os.path.lexists(home / name)
        for name in ("model-settings.json", "model-file-selections.json")
    ):
        raise LibraryError(
            "migration", "Finish the complete account model settings migration first"
        )
    writing = operation.endswith(".patch")
    if set(arguments) - ({"revision", "patch", "remove"} if writing else set()):
        raise LibraryError("invalid", "unknown model preference field")
    patch, paths = _changes(arguments, writing=writing)
    home.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(home / "model-settings.sqlite", timeout=30)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1):
            raise LibraryError("format", f"Unsupported model settings format: {version}")
        db.execute(
            "CREATE TABLE IF NOT EXISTS preferences "
            "(id INTEGER PRIMARY KEY CHECK(id=1), "
            "revision INTEGER NOT NULL, document TEXT NOT NULL)"
        )
        db.execute("PRAGMA user_version=1")
        row = db.execute("SELECT revision,document FROM preferences WHERE id=1").fetchone()
        revision = row[0] if row else 0
        values = object_value(json.loads(row[1]), "stored model settings") if row else {}
        if writing:
            expected = arguments.get("revision")
            if type(expected) is not int or expected != revision:
                raise LibraryError("stale", "model settings changed; read them again before saving")
            _apply(values, patch, paths)
            revision += 1
            db.execute(
                "INSERT INTO preferences VALUES (1,?,?) ON CONFLICT(id) DO UPDATE SET "
                "revision=excluded.revision,document=excluded.document",
                (revision, json.dumps(values, allow_nan=False)),
            )
    return {"folders": {"revision": revision, "values": values}}


def _changes(
    arguments: dict[str, object], *, writing: bool
) -> tuple[dict[str, object], list[list[str]]]:
    patch = object_value(arguments.get("patch", {}), "settings patch") if writing else {}
    raw_remove = arguments.get("remove", [])
    if not isinstance(raw_remove, list):
        raise LibraryError("invalid", "removed settings paths must be a list")
    paths: list[list[str]] = []
    for raw_path in cast("list[object]", raw_remove):
        if not isinstance(raw_path, list) or not raw_path:
            raise LibraryError("invalid", "removed settings paths must be nonempty lists")
        paths.append(
            [text(part, "settings path component") for part in cast("list[object]", raw_path)]
        )
    return patch, paths


def _apply(values: dict[str, object], patch: dict[str, object], paths: list[list[str]]) -> None:
    # Null is a stored value. Removal is explicit, so unknown fields survive.
    _merge(values, patch)
    for path in paths:
        current = values
        for name in path[:-1]:
            child = current.get(name)
            if not isinstance(child, dict):
                raise LibraryError("invalid", "removed settings parent must be an object")
            current = cast("dict[str, object]", child)
        current.pop(path[-1], None)


def preferences(
    records: Records, operation: str, arguments: dict[str, object]
) -> dict[str, object]:
    paired = operation in ("folders.get", "folders.patch")
    writing = operation.endswith(".patch")
    if paired and set(arguments) != ({"grants", "projects"} if writing else set()):
        raise LibraryError("invalid", "folder patch requires grants and projects changes")
    with records.db:
        # Paired reads need one snapshot; paired writes must roll back both records.
        records.db.execute("BEGIN IMMEDIATE" if writing else "BEGIN")
        if paired:
            return {
                name: _preference(
                    records,
                    f"folders.{name}.{'patch' if writing else 'get'}",
                    object_value(arguments[name], f"{name} changes") if writing else {},
                )["folders"]
                for name in ("grants", "projects")
            }
        return _preference(records, operation, arguments)


def _preference(
    records: Records, operation: str, arguments: dict[str, object]
) -> dict[str, object]:
    prefix = operation.rsplit(".", 1)[0]
    if prefix not in ("settings", "folders.projects", "folders.grants") or operation not in (
        f"{prefix}.get",
        f"{prefix}.patch",
    ):
        raise LibraryError("operation", f"unknown library operation: {operation}")
    folders = prefix != "settings"
    legacy = {
        "settings": ("settings.json", "pack-settings"),
        "folders.projects": ("project-folders.json",),
        "folders.grants": ("mounts.toml",),
    }[prefix]
    if any((records.root / name).exists() for name in ("library.sqlite", *legacy)):
        raise LibraryError("migration", "Finish the complete library migration first")
    writing = operation.endswith(".patch")
    allowed: set[str] = set() if folders else {"environment", "pack"}
    if writing:
        allowed |= {"revision", "patch", "remove"}
    if set(arguments) - allowed:
        raise LibraryError("invalid", "unknown preference record field")
    identity: dict[str, object] = {}
    if folders:
        key = ("folders", prefix.split(".")[1])
    else:
        if ("environment" in arguments) == ("pack" in arguments):
            raise LibraryError("invalid", "settings require exactly one environment or pack")
        kind = "environment" if "environment" in arguments else "pack"
        identifier = text(arguments[kind], f"settings {kind}")
        key = (kind, identifier)
        identity[kind] = identifier
    result_key = "folders" if folders else "settings"
    patch, paths = _changes(arguments, writing=writing)
    row = records.db.execute(
        "SELECT revision,document FROM settings_records WHERE kind=? AND id=?", key
    ).fetchone()
    revision = row[0] if row else 0
    values = object_value(json.loads(row[1]), "stored settings") if row else {}
    if writing:
        expected = arguments.get("revision")
        if type(expected) is not int or expected != revision:
            raise LibraryError("stale", "settings changed; read them again before saving")
        _apply(values, patch, paths)
        revision += 1
        result = {result_key: {**identity, "revision": revision, "values": values}}
        # Folder records can exceed a line; negotiated framing carries them intact.
        if not folders and (
            len(json.dumps({"result": result}, allow_nan=False).encode("utf-8")) + 1
            > MAX_SETTINGS_REPLY
        ):
            raise LibraryError("invalid", "settings reply exceeds the 1 MiB limit")
        records.db.execute(
            "INSERT INTO settings_records VALUES (?,?,?,?) "
            "ON CONFLICT(kind,id) DO UPDATE SET "
            "revision=excluded.revision,document=excluded.document",
            (*key, revision, json.dumps(values, allow_nan=False)),
        )
        records.db.execute(
            "INSERT INTO changes(kind,id,revision) VALUES (?,?,?)",
            (result_key, json.dumps(key, separators=(",", ":")), revision),
        )
    return {result_key: {**identity, "revision": revision, "values": values}}
