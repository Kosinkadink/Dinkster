"""Helper-owned saved workflows and named asset records."""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from typing import TYPE_CHECKING, cast

from ._browse import record_cursor
from ._manifests import asset_digest
from ._records import object_value, project_id
from ._workflows import text
from .client import LibraryError

if TYPE_CHECKING:
    from ._records import Records


_FIELDS = {"name", "digest", "mediaType", "labels", "folder"}


def _fields(patch: dict[str, object]) -> dict[str, object]:
    cleaned = dict(patch)
    if "digest" in cleaned:
        asset_digest(cleaned["digest"])
    for field in ("name", "mediaType"):
        if field in cleaned:
            value = text(cleaned[field], field).strip()
            if field == "name" and len(value) > 512:
                raise LibraryError("invalid", "name exceeds 512 characters")
            cleaned[field] = value
    if "labels" in cleaned:
        raw = cleaned["labels"]
        if not isinstance(raw, list) or len(cast("list[object]", raw)) > 32:
            raise LibraryError("invalid", "labels must be a list of at most 32 strings")
        labels: list[str] = []
        for item in cast("list[object]", raw):
            label = text(item, "label").strip()
            if len(label) > 128:
                raise LibraryError("invalid", "label exceeds 128 characters")
            if label not in labels:
                labels.append(label)
        cleaned["labels"] = labels
    if "folder" in cleaned:
        folder = cleaned["folder"]
        if not isinstance(folder, str) or (
            folder
            and ("\\" in folder or any(part in ("", ".", "..") for part in folder.split("/")))
        ):
            raise LibraryError("invalid", "folder must be a safe virtual path")
    return cleaned


def saved(records: Records, operation: str, arguments: dict[str, object]) -> dict[str, object]:
    if os.path.lexists(records.root / "library.sqlite"):
        raise LibraryError("migration", "Finish the complete library migration first")
    allowed = {"scope", "record", "project"} if operation == "records.create" else {"scope", "id"}
    if operation in ("records.patch", "records.delete"):
        allowed.add("revision")
    if operation == "records.patch":
        allowed.add("patch")
    if set(arguments) - allowed:
        raise LibraryError("invalid", "unknown saved-record field")
    scope = text(arguments.get("scope"), "record scope")
    if operation == "records.create":
        document = object_value(arguments.get("record"), "saved record")
        if set(document) - _FIELDS:
            raise LibraryError("invalid", "unknown or immutable saved-record field")
        if not {"name", "digest", "mediaType"} <= document.keys():
            raise LibraryError("invalid", "name, digest and mediaType are required")
        document = _fields(document)
        document.setdefault("labels", [])
        identifier = uuid.uuid4().hex
    else:
        identifier = text(arguments.get("id"), "record id")
        if operation == "records.get":
            row = records.db.execute(
                "SELECT document FROM saved_records WHERE scope=? AND id=? AND deleted=0",
                (scope, identifier),
            ).fetchone()
            return {"record": json.loads(row[0]) if row else None}
        document = {}
    with records.db:
        records.db.execute("BEGIN IMMEDIATE")
        if operation == "records.create":
            project = project_id(arguments.get("project", "default"))
            owner = records.get_project(project)
            if owner is None or owner["deleted"]:
                raise LibraryError("missing", "project is unavailable")
            revision = 1
        else:
            row = records.db.execute(
                "SELECT project,revision,document FROM saved_records "
                "WHERE scope=? AND id=? AND deleted=0",
                (scope, identifier),
            ).fetchone()
            if row is None:
                return {"deleted": False} if operation == "records.delete" else {"record": None}
            expected = arguments.get("revision")
            if type(expected) is not int or expected != row[1]:
                raise LibraryError("stale", "saved record changed; read it again before saving")
            project, revision = row[0], row[1] + 1
            document = object_value(json.loads(row[2]), "saved record")
            if operation == "records.patch":
                patch = object_value(arguments.get("patch"), "record patch")
                if set(patch) - _FIELDS:
                    raise LibraryError("invalid", "unknown or immutable record patch field")
                document.update(_fields(patch))
        latest = records.db.execute("SELECT MAX(modified) FROM saved_records").fetchone()[0]
        modified = time.time()
        if latest is not None and modified <= latest:
            modified = math.nextafter(float(latest), math.inf)
        if operation == "records.create":
            document.update(id=identifier, scope=scope, project=project, created=modified)
        document.update(revision=revision, modified=modified)
        if not document.get("folder"):
            document.pop("folder", None)
        deleted = operation == "records.delete"
        records.db.execute(
            "INSERT INTO saved_records(id,scope,project,modified,revision,document,deleted) "
            "VALUES (?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
            "modified=excluded.modified,revision=excluded.revision,"
            "document=excluded.document,deleted=excluded.deleted",
            (
                identifier,
                scope,
                project,
                modified,
                revision,
                "{}" if deleted else json.dumps(document, allow_nan=False),
                deleted,
            ),
        )
        records.db.execute(
            "INSERT INTO changes(kind,id,revision) VALUES ('record',?,?)",
            (json.dumps([scope, identifier]), revision),
        )
        return {"deleted": True} if deleted else {"record": document}


def list_saved(
    records: Records, bound: dict[str, object], limit: int, after: tuple[float, str] | None
) -> dict[str, object]:
    clauses = ["scope=?", "deleted=0"]
    parameters: list[object] = [bound["scope"]]
    if bound["project"] is not None:
        clauses.append("project=?")
        parameters.append(bound["project"])
    if bound["mediaType"]:
        clauses.append("json_extract(document,'$.mediaType')=?")
        parameters.append(bound["mediaType"])
    if bound["q"]:
        clauses.append("instr(lower(json_extract(document,'$.name')),lower(?))>0")
        parameters.append(bound["q"])
    if bound["label"]:
        clauses.append("EXISTS (SELECT 1 FROM json_each(document,'$.labels') WHERE value=?)")
        parameters.append(bound["label"])
    if after is not None:
        clauses.append("(modified,id)<(?,?)")
        parameters.extend(after)
    rows = records.db.execute(
        "SELECT modified,id,document FROM saved_records WHERE "
        + " AND ".join(clauses)
        + " ORDER BY modified DESC,id DESC LIMIT ?",
        (*parameters, limit + 1),
    ).fetchall()
    return {
        "records": [json.loads(row[2]) for row in rows[:limit]],
        "cursor": record_cursor(bound, rows[limit - 1][:2]) if len(rows) > limit else None,
    }
