"""Read-only browsing of engine records with helper-owned workflow placement."""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
import sqlite3
from typing import TYPE_CHECKING, cast

from ._records import object_value, project_id
from ._workflows import WORKFLOW_MEDIA_TYPE, text
from .client import LibraryError

if TYPE_CHECKING:
    from ._records import Records


def record_page(
    arguments: dict[str, object], bound: dict[str, object]
) -> tuple[int, tuple[float, str] | None]:
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
    return limit, after


def record_cursor(bound: dict[str, object], position: tuple[object, object]) -> str:
    payload = json.dumps({"query": bound, "after": position})
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


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
    limit, after = record_page(arguments, bound)
    path = records.root / "library.sqlite"
    if not path.exists():
        from ._saved import list_saved

        return list_saved(records, bound, limit, after)
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
        if not wire["folder"]:
            del wire["folder"]
        if wire["mediaType"] == WORKFLOW_MEDIA_TYPE:
            wire["project"] = owner if owner is not None else "default"
        selected.append(wire)
    page = selected[:limit]
    next_cursor = None
    if len(selected) > limit:
        last = page[-1]
        next_cursor = record_cursor(bound, (last["modified"], last["id"]))
    return {"records": page, "cursor": next_cursor}


def list_assets(records: Records, arguments: dict[str, object]) -> dict[str, object]:
    from .catalog import query_asset_rows

    if set(arguments) - {
        "scope",
        "project",
        "q",
        "kind",
        "origin",
        "since",
        "until",
        "limit",
        "cursor",
    }:
        raise LibraryError("invalid", "unknown asset query field")
    scope = text(arguments.get("scope"), "asset scope")
    project = project_id(arguments.get("project", "default"))
    owner = records.get_project(project)
    if owner is None or owner["deleted"]:
        raise LibraryError("missing", "project is unavailable")
    bound: dict[str, object] = {"source": "assets", "scope": scope, "project": project}
    filters: dict[str, str] = {}
    for key in ("q", "kind", "origin"):
        value = arguments.get(key, "")
        if not isinstance(value, str):
            raise LibraryError("invalid", f"{key} must be a string")
        filters[key] = value
        bound[key] = value
    if (
        filters["kind"]
        and re.fullmatch(r"[a-z0-9][a-z0-9-]*(/[a-z0-9][a-z0-9-]*)+", filters["kind"]) is None
    ):
        raise LibraryError("invalid", "asset kind must be a lowercase namespaced kind")
    dates: dict[str, float | None] = {}
    for key in ("since", "until"):
        value = arguments.get(key)
        try:
            if value is not None and (
                type(value) not in (int, float) or not math.isfinite(cast(float, value))
            ):
                raise ValueError
            dates[key] = float(cast(float, value)) if value is not None else None
        except (ValueError, OverflowError) as error:
            raise LibraryError("invalid", "asset date range must be finite numbers") from error
        bound[key] = dates[key]
    since, until = dates["since"], dates["until"]
    if since is not None and until is not None and since > until:
        raise LibraryError("invalid", "asset date range is reversed")
    limit, after = record_page(arguments, bound)
    path = records.root / "library.sqlite"
    if not path.exists():
        path = records.root / "records.sqlite"
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("BEGIN")
        groups = query_asset_rows(
            connection,
            scope,
            project=project,
            text=filters["q"],
            kind=filters["kind"],
            origin=filters["origin"],
            since=since,
            until=until,
            limit=limit + 1,
            after=after,
        )
    finally:
        connection.close()
    assets: list[dict[str, object]] = []
    for rows in groups[:limit]:
        observations: list[dict[str, object]] = []
        aliases: dict[str, None] = {}
        for row in rows:
            asset: dict[str, object] = {
                "digest": row["digest"],
                "name": row["stored_name"],
                "size": row["byte_size"],
                "mediaType": row["media_type"],
                "virtualPath": row["virtual_path"],
            }
            observations.append(
                json.loads(row["document"])
                if "document" in row.keys()
                else {
                    "id": row["id"],
                    "asset": asset,
                    "originalName": row["original_name"],
                    "kind": row["kind"],
                    "origin": json.loads(row["origin"]),
                    "project": row["project"],
                    "created": row["created"],
                }
            )
            aliases[row["original_name"]] = None
            aliases[row["stored_name"]] = None
        assets.append(
            {
                "asset": observations[-1]["asset"],
                "aliases": list(aliases),
                "observations": observations,
            }
        )
    next_cursor = None
    if len(groups) > limit:
        last = groups[limit - 1][-1]
        next_cursor = record_cursor(bound, (last["created"], last["digest"]))
    return {"assets": assets, "cursor": next_cursor}
