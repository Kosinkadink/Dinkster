"""Local restoration impact computed from recorded engines and project bindings."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from .client import LibraryError

if TYPE_CHECKING:
    from ._records import Records


def affected(records: Records, arguments: dict[str, object]) -> dict[str, object]:
    low, high = arguments.get("apiMin"), arguments.get("apiMax")
    if type(low) is not int or type(high) is not int or not 0 < low <= high:
        raise LibraryError("invalid", "supported API range must be positive and ordered")
    cursor = arguments.get("cursor")
    if cursor is None:
        cursor = ["", ""]
    limit = arguments.get("limit", 100)
    if (
        not isinstance(cursor, list)
        or len(cast("list[object]", cursor)) != 2
        or any(not isinstance(value, str) for value in cast("list[object]", cursor))
        or type(limit) is not int
        or not 1 <= limit <= 1000
    ):
        raise LibraryError("invalid", "invalid compatibility cursor or page size")
    rows = records.db.execute(
        "WITH affected AS ("
        " SELECT environment,snapshot,document FROM engines"
        " WHERE json_extract(document,'$.apiVersion') NOT BETWEEN ? AND ?"
        "), entries AS ("
        " SELECT 'snapshot' kind,json_array(environment,snapshot) key,"
        " json_set(document,'$.kind','snapshot') document FROM affected"
        " UNION ALL"
        " SELECT 'project',p.id,"
        " json_set(p.document,'$.kind','project','$.environment',a.environment,"
        " '$.snapshot',a.snapshot) FROM projects p JOIN affected a"
        " ON json_extract(p.document,'$.environment.id')=a.environment"
        " AND json_extract(p.document,'$.environment.lastSnapshot')=a.snapshot"
        " WHERE json_extract(p.document,'$.deleted')=0"
        ") SELECT kind,key,document FROM entries WHERE (kind,key)>(?,?)"
        " ORDER BY kind,key LIMIT ?",
        (low, high, *cast("list[str]", cursor), limit + 1),
    ).fetchall()
    return {
        "items": [json.loads(row[2]) for row in rows[:limit]],
        "cursor": list(rows[limit - 1][:2]) if len(rows) > limit else None,
    }
