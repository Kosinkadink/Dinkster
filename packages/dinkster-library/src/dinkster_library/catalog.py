"""Scoped asset catalog queries shared by engines and engine-free tools."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit, urlunsplit

if TYPE_CHECKING:
    from ._records import Records


def record_asset(records: Records, arguments: dict[str, object]) -> dict[str, object]:
    """Record ingest history and retain its digest without granting read authority."""
    from ._manifests import asset_digest
    from ._records import object_value, project_id
    from ._workflows import text as required_text
    from .client import LibraryError

    if (records.root / "library.sqlite").exists():
        raise LibraryError("migration", "Finish the complete library migration first")
    if set(arguments) != {"scope", "observation"}:
        raise LibraryError("invalid", "scope and observation are required")
    scope = required_text(arguments["scope"], "asset scope")
    document = dict(object_value(arguments["observation"], "asset observation"))
    identifier = document.get("id")
    if not isinstance(identifier, str) or re.fullmatch(r"[0-9a-f]{32}", identifier) is None:
        raise LibraryError("invalid", "observation id must be 32 lowercase hex digits")
    project = project_id(document.get("project", "default"))
    document["project"] = project
    asset = dict(object_value(document.get("asset"), "observed asset"))
    digest = asset_digest(asset.get("digest"))
    original = required_text(document.get("originalName"), "original name").strip()
    name = required_text(asset.get("name"), "stored name")
    if any(value in (".", "..") or "/" in value or "\\" in value for value in (original, name)):
        raise LibraryError("invalid", "asset names must be basenames")
    document["originalName"] = original
    size = asset.get("size")
    if type(size) is not int or not 0 <= size <= 9223372036854775807:
        raise LibraryError("invalid", "asset size must be a nonnegative SQLite integer")
    media_type = required_text(asset.get("mediaType"), "asset media type")
    path = asset.get("virtualPath", "")
    if not isinstance(path, str) or (
        path and ("\\" in path or any(part in ("", ".", "..") for part in path.split("/")))
    ):
        raise LibraryError("invalid", "virtual path must be a safe relative path")
    asset["virtualPath"] = path
    document["asset"] = asset
    kind = required_text(document.get("kind"), "asset kind")
    if re.fullmatch(r"[a-z0-9][a-z0-9-]*(/[a-z0-9][a-z0-9-]*)+", kind) is None:
        raise LibraryError("invalid", "asset kind must be a lowercase namespaced kind")
    origin = dict(object_value(document.get("origin"), "asset origin"))
    fields = {
        "upload": set[str](),
        "download": {"url"},
        "transfer": {"machine"},
        "produced": {"job", "node"},
        "unknown": set[str](),
    }.get(str(origin.get("kind")))
    if fields is None or set(origin) != fields | {"kind"}:
        raise LibraryError("invalid", "invalid asset origin")
    for value in origin.values():
        required_text(value, "origin value")
    if origin["kind"] == "download":
        try:
            url = urlsplit(cast(str, origin["url"]))
            if url.scheme not in ("http", "https") or not url.hostname:
                raise ValueError
            origin["url"] = urlunsplit(
                (url.scheme, url.netloc.rsplit("@", 1)[-1], url.path, "", "")
            )
        except ValueError as error:
            raise LibraryError("invalid", "download origin requires an HTTP URL") from error
    document["origin"] = origin
    created = document.get("created")
    try:
        if (
            type(created) not in (int, float)
            or not math.isfinite(cast(float, created))
            or cast(float, created) < 0
        ):
            raise ValueError
    except (ValueError, OverflowError) as error:
        raise LibraryError(
            "invalid", "observation time must be a finite nonnegative number"
        ) from error
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with records.db:
        records.db.execute("BEGIN IMMEDIATE")
        owner = records.get_project(project)
        if owner is None or owner["deleted"]:
            raise LibraryError("missing", "project is unavailable")
        row = records.db.execute(
            "SELECT scope,document FROM asset_observations WHERE id=?", (identifier,)
        ).fetchone()
        if row is not None:
            if row[0] != scope or row[1] != encoded:
                raise LibraryError(
                    "conflict", "observation id already describes another publication"
                )
            return {"observation": document, "created": False}
        records.db.execute(
            "INSERT INTO asset_observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                identifier,
                scope,
                digest,
                original,
                name,
                kind,
                media_type,
                size,
                path,
                json.dumps(origin),
                project,
                float(cast(float, created)),
                encoded,
            ),
        )
        records.db.execute(
            "INSERT INTO changes(kind,id,revision) VALUES ('observation',?,1)",
            (json.dumps([scope, identifier]),),
        )
    return {"observation": document, "created": True}


def query_asset_rows(
    connection: sqlite3.Connection,
    scope: str,
    *,
    project: str = "default",
    digest: str = "",
    text: str = "",
    kind: str = "",
    origin: str = "",
    since: float | None = None,
    until: float | None = None,
    limit: int = 50,
    after: tuple[float, str] | None = None,
) -> list[list[sqlite3.Row]]:
    """Read one page in the caller's transaction, retaining every selected digest's aliases."""
    clauses = ["scope = ?", "project = ?"]
    parameters: list[object] = [scope, project]
    if digest:
        clauses.append("digest = ?")
        parameters.append(digest)
    matching = [
        "matched.scope = asset_observations.scope",
        "matched.project = asset_observations.project",
        "matched.digest = asset_observations.digest",
    ]
    if text:
        matching.append(
            "(instr(lower(matched.original_name), lower(?)) > 0 OR "
            "instr(lower(matched.stored_name), lower(?)) > 0 OR "
            "EXISTS (SELECT 1 FROM json_each(matched.origin) "
            "WHERE instr(lower(json_each.value), lower(?)) > 0))"
        )
        parameters.extend((text, text, text))
    if kind:
        matching.append("matched.kind = ?")
        parameters.append(kind)
    if origin:
        matching.append("json_extract(matched.origin, '$.kind') = ?")
        parameters.append(origin)
    if since is not None:
        matching.append("matched.created >= ?")
        parameters.append(since)
    if until is not None:
        matching.append("matched.created <= ?")
        parameters.append(until)
    clauses.append(
        "EXISTS (SELECT 1 FROM asset_observations matched WHERE " + " AND ".join(matching) + ")"
    )
    having = ""
    if after is not None:
        having = " HAVING (MAX(created) < ? OR (MAX(created) = ? AND digest < ?))"
        parameters.extend((after[0], after[0], after[1]))
    parameters.append(max(1, limit))
    selected = connection.execute(
        "SELECT digest, MAX(created) AS latest FROM asset_observations WHERE "
        + " AND ".join(clauses)
        + " GROUP BY digest"
        + having
        + " ORDER BY latest DESC, digest DESC LIMIT ?",
        parameters,
    ).fetchall()
    if not selected:
        return []
    digests = [row["digest"] for row in selected]
    rows = connection.execute(
        "SELECT * FROM asset_observations WHERE scope = ? AND project = ? AND digest IN ("
        + ",".join("?" for _ in digests)
        + ") ORDER BY created, id",
        [scope, project, *digests],
    ).fetchall()
    grouped: dict[str, list[sqlite3.Row]] = {digest: [] for digest in digests}
    for row in rows:
        grouped[row["digest"]].append(row)
    return [grouped[digest] for digest in digests]
