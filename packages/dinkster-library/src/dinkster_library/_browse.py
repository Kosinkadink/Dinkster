"""Read-only browsing of engine records with helper-owned workflow placement."""

from __future__ import annotations

import base64
import binascii
import json
import math
import sqlite3
from typing import TYPE_CHECKING, cast

from ._records import object_value, project_id
from ._workflows import WORKFLOW_MEDIA_TYPE, text
from .client import LibraryError

if TYPE_CHECKING:
    from ._records import Records


def list_records(records: Records, arguments: dict[str, object]) -> dict[str, object]:
    allowed = {"scope", "project", "mediaType", "q", "label", "limit", "cursor"}
    if set(arguments) - allowed:
        raise LibraryError("invalid", "unknown record query field")
    scope = text(arguments.get("scope"), "record scope")
    bound: dict[str, object] = {"scope": scope, "project": arguments.get("project")}
    for key in ("mediaType", "q", "label"):
        value = arguments.get(key, "")
        if not isinstance(value, str):
            raise LibraryError("invalid", f"{key} must be a string")
        bound[key] = value
    project = bound["project"]
    if project is not None:
        project = project_id(project)
        owner = records.get_project(project)
        if owner is None or owner["deleted"]:
            raise LibraryError("missing", "project is unavailable")
    limit = arguments.get("limit", 50)
    if type(limit) is not int or not 1 <= limit <= 200:
        raise LibraryError("invalid", "record page size must be an integer from 1 to 200")
    after: tuple[float, str] | None = None
    cursor = arguments.get("cursor")
    if cursor is not None:
        try:
            if not isinstance(cursor, str) or len(cursor) > 16384:
                raise ValueError
            decoded = object_value(
                json.loads(
                    base64.b64decode(
                        cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True
                    )
                ),
                "record cursor",
            )
            if decoded.get("query") != bound:
                raise ValueError
            raw_position = decoded.get("after")
            if not isinstance(raw_position, list):
                raise ValueError
            position = cast("list[object]", raw_position)
            if (
                len(position) != 2
                or type(position[0]) not in (int, float)
                or not math.isfinite(cast(float, position[0]))
                or not isinstance(position[1], str)
            ):
                raise ValueError
            after = (float(cast(float, position[0])), position[1])
        except (ValueError, TypeError, OverflowError, RecursionError, binascii.Error) as error:
            raise LibraryError(
                "invalid", "record cursor is malformed or does not match this query"
            ) from error
    path = records.root / "library.sqlite"
    if not path.exists():
        return {"records": [], "cursor": None}
    clauses = ["r.scope=?"]
    parameters: list[object] = [scope]
    if project is not None:
        clauses.extend(["r.media_type=?", "COALESCE(w.project,'default')=?"])
        parameters.extend([WORKFLOW_MEDIA_TYPE, project])
    if bound["mediaType"]:
        clauses.append("r.media_type=?")
        parameters.append(bound["mediaType"])
    if bound["q"]:
        clauses.append("instr(lower(r.name),lower(?))>0")
        parameters.append(bound["q"])
    if bound["label"]:
        clauses.append("EXISTS (SELECT 1 FROM json_each(r.labels) WHERE value=?)")
        parameters.append(bound["label"])
    if after is not None:
        clauses.append("(r.modified,r.id)<(?,?)")
        parameters.extend(after)
    parameters.append(limit + 1)
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute(
            "ATTACH DATABASE ? AS placement",
            ((records.root / "records.sqlite").resolve().as_uri() + "?mode=ro",),
        )
        rows = connection.execute(
            "SELECT r.*,w.project AS workflow_owner FROM records r "
            "LEFT JOIN placement.workflow_facts w ON w.scope=r.scope AND w.id=r.id WHERE "
            + " AND ".join(clauses)
            + " ORDER BY r.modified DESC,r.id DESC LIMIT ?",
            parameters,
        ).fetchall()
    finally:
        connection.close()
    selected: list[dict[str, object]] = []
    for row in rows:
        wire = dict(row)
        owner = wire.pop("workflow_owner")
        wire["mediaType"] = wire.pop("media_type")
        wire["labels"] = json.loads(cast(str, wire["labels"]))
        if wire["mediaType"] == WORKFLOW_MEDIA_TYPE:
            wire["project"] = owner if owner is not None else "default"
        selected.append(wire)
    page = selected[:limit]
    next_cursor = None
    if len(selected) > limit:
        last = page[-1]
        payload = json.dumps({"query": bound, "after": [last["modified"], last["id"]]})
        next_cursor = base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")
    return {"records": page, "cursor": next_cursor}
