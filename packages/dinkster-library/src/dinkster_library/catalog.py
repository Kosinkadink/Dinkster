"""Scoped asset catalog queries shared by engines and engine-free tools."""

from __future__ import annotations

import sqlite3


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
