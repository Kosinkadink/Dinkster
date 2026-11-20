"""Environment and pack preferences, patched without replacing unseen fields."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from ._records import object_value
from ._workflows import text
from .client import LibraryError

if TYPE_CHECKING:
    from ._records import Records


def _merge(current: dict[str, object], patch: dict[str, object]) -> None:
    for name, value in patch.items():
        previous = current.get(name)
        if isinstance(value, dict) and isinstance(previous, dict):
            _merge(cast("dict[str, object]", previous), cast("dict[str, object]", value))
        else:
            current[name] = value


def settings(records: Records, operation: str, arguments: dict[str, object]) -> dict[str, object]:
    if any(
        (records.root / name).exists()
        for name in ("library.sqlite", "settings.json", "pack-settings")
    ):
        raise LibraryError("migration", "Finish the complete library migration first")
    if operation not in ("settings.get", "settings.patch"):
        raise LibraryError("operation", f"unknown library operation: {operation}")
    writing = operation == "settings.patch"
    allowed = {"environment", "pack"}
    if writing:
        allowed |= {"revision", "patch", "remove"}
    if set(arguments) - allowed:
        raise LibraryError("invalid", "unknown settings record field")
    if ("environment" in arguments) == ("pack" in arguments):
        raise LibraryError("invalid", "settings require exactly one environment or pack")
    kind = "environment" if "environment" in arguments else "pack"
    identifier = text(arguments[kind], f"settings {kind}")
    key = (kind, identifier)
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
        if writing:
            records.db.execute("BEGIN IMMEDIATE")
        row = records.db.execute(
            "SELECT revision,document FROM settings_records WHERE kind=? AND id=?", key
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
            records.db.execute(
                "INSERT INTO settings_records VALUES (?,?,?,?) "
                "ON CONFLICT(kind,id) DO UPDATE SET "
                "revision=excluded.revision,document=excluded.document",
                (*key, revision, json.dumps(values, allow_nan=False)),
            )
            records.db.execute(
                "INSERT INTO changes(kind,id,revision) VALUES ('settings',?,?)",
                (json.dumps(key, separators=(",", ":")), revision),
            )
    return {"settings": {kind: identifier, "revision": revision, "values": values}}
