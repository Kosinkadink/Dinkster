"""Read-only browsing of engine records with helper-owned workflow placement."""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
import sqlite3
from typing import TYPE_CHECKING, cast

from ._legacy_records import legacy_store_exists
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
    if cursor is not None or "after" in arguments:
        try:
            raw_position = arguments.get("after")
            if cursor is not None:
                if "after" in arguments or not isinstance(cursor, str) or len(cursor) > 16384:
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
                or not position[1].strip()
            ):
                raise ValueError
            after = (float(cast(float, position[0])), position[1])
        except (ValueError, TypeError, OverflowError, RecursionError, binascii.Error) as error:
            raise LibraryError(
                "invalid", "record position is malformed or does not match this query"
            ) from error
    return limit, after


def record_cursor(bound: dict[str, object], position: tuple[object, object]) -> str:
    payload = json.dumps({"query": bound, "after": position})
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


def list_records(records: Records, arguments: dict[str, object]) -> dict[str, object]:
    allowed = {"scope", "project", "mediaType", "q", "label", "limit", "cursor", "after"}
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
    if not legacy_store_exists(records.root):
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
    from ._manifests import asset_digest
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
        "after",
        "digest",
    }:
        raise LibraryError("invalid", "unknown asset query field")
    scope = text(arguments.get("scope"), "asset scope")
    project = project_id(arguments.get("project", "default"))
    owner = records.get_project(project)
    if owner is None or owner["deleted"]:
        raise LibraryError("missing", "project is unavailable")
    bound: dict[str, object] = {"source": "assets", "scope": scope, "project": project}
    digest = arguments.get("digest", "")
    if digest != "":
        digest = asset_digest(digest)
        bound["digest"] = digest
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
    connection = _asset_store(records)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("BEGIN")
        groups = query_asset_rows(
            connection,
            scope,
            project=project,
            digest=cast(str, digest),
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


def _asset_store(records: Records) -> sqlite3.Connection:
    path = records.root / "library.sqlite"
    if not legacy_store_exists(records.root):
        path = records.root / "records.sqlite"
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)


def _listed(records: Records, project: str) -> bool:
    document = records.get_project(project)
    return document is not None and not document["deleted"]


def project_usage(records: Records, identifier: str) -> dict[str, object]:
    """How many outputs one project's runs produced, and the stored size of
    the distinct contents its workflows, imports and outputs refer to. Each
    content counts once per project; one shared with another project counts in
    both. Models do not count."""
    from ._workflows import needs

    workflows = cast(
        "list[dict[str, str]]", needs(records, identifier, include_workflows=True)["workflows"]
    )
    connection = _asset_store(records)
    try:
        rows = connection.execute(
            "SELECT digest, MAX(byte_size), MAX(json_extract(origin, '$.kind') = 'produced') "
            "FROM asset_observations WHERE scope = 'local' AND project = ? GROUP BY digest",
            (identifier,),
        ).fetchall()
    finally:
        connection.close()
    sizes = {row[0]: row[1] for row in rows}
    for record in workflows:
        if record["digest"] not in sizes:
            held = records.root / "vault" / record["digest"][7:9] / record["digest"][7:]
            try:
                sizes[record["digest"]] = held.stat().st_size
            except OSError:
                sizes[record["digest"]] = 0
    return {"outputs": sum(1 for row in rows if row[2]), "bytes": sum(sizes.values())}


def served_asset(records: Records, digest: str) -> dict[str, object]:
    """What a digest is to a listed project: the current document of one of
    its saved workflows, or an output its runs produced. The always-on process
    serves only these with no engine open."""
    from ._workflows import workflow_records

    for row in workflow_records(records):
        if row["mediaType"] != WORKFLOW_MEDIA_TYPE or row["digest"] != digest:
            continue
        owner = records.db.execute(
            "SELECT project FROM workflow_facts WHERE scope=? AND id=?", (row["scope"], row["id"])
        ).fetchone()
        if _listed(records, owner[0] if owner else str(row.get("project", "default"))):
            return {"role": "workflow", "mediaType": WORKFLOW_MEDIA_TYPE}
    connection = _asset_store(records)
    try:
        rows = connection.execute(
            "SELECT project, media_type FROM asset_observations WHERE scope = 'local' "
            "AND digest = ? AND json_extract(origin, '$.kind') = 'produced' ORDER BY created",
            (digest,),
        ).fetchall()
    finally:
        connection.close()
    for project, media_type in rows:
        if _listed(records, project):
            return {"role": "output", "mediaType": media_type}
    raise LibraryError("missing", "asset is not a listed project's workflow or output")
