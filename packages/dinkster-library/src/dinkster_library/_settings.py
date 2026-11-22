"""Revision-checked preferences, patched without replacing unseen fields."""

from __future__ import annotations

import json
import os
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


def preferences(
    records: Records, operation: str, arguments: dict[str, object]
) -> dict[str, object]:
    if operation not in ("settings.get", "settings.patch"):
        raise LibraryError("operation", f"unknown library operation: {operation}")
    if any(
        os.path.lexists(records.root / name)
        for name in ("library.sqlite", "settings.json", "pack-settings")
    ):
        raise LibraryError("migration", "Finish the complete library migration first")
    writing = operation == "settings.patch"
    allowed = {"environment", "pack"}
    if writing:
        allowed |= {"revision", "patch", "remove"}
    if set(arguments) - allowed:
        raise LibraryError("invalid", "unknown preference record field")
    if ("environment" in arguments) == ("pack" in arguments):
        raise LibraryError("invalid", "settings require exactly one environment or pack")
    kind = "environment" if "environment" in arguments else "pack"
    identifier = text(arguments[kind], f"settings {kind}")
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
    with records.db:
        records.db.execute("BEGIN IMMEDIATE" if writing else "BEGIN")
        row = records.db.execute(
            "SELECT revision,document FROM settings_records WHERE kind=? AND id=?",
            (kind, identifier),
        ).fetchone()
        revision = row[0] if row else 0
        values = object_value(json.loads(row[1]), "stored settings") if row else {}
        if writing:
            expected = arguments.get("revision")
            if type(expected) is not int or expected != revision:
                raise LibraryError("stale", "settings changed; read them again before saving")
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
            revision += 1
        result: dict[str, object] = {
            "settings": {kind: identifier, "revision": revision, "values": values}
        }
        if writing:
            if (
                len(json.dumps({"result": result}, allow_nan=False).encode("utf-8")) + 1
                > MAX_SETTINGS_REPLY
            ):
                raise LibraryError("invalid", "settings reply exceeds the 1 MiB limit")
            records.db.execute(
                "INSERT INTO settings_records VALUES (?,?,?,?) ON CONFLICT(kind,id) DO UPDATE SET "
                "revision=excluded.revision,document=excluded.document",
                (kind, identifier, revision, json.dumps(values, allow_nan=False)),
            )
            records.db.execute(
                "INSERT INTO changes(kind,id,revision) VALUES ('settings',?,?)",
                (json.dumps((kind, identifier), separators=(",", ":")), revision),
            )
    return result
