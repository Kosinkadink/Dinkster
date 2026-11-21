"""Immutable finished-run records owned by the library helper."""

from __future__ import annotations

import json
import math
import re
from typing import TYPE_CHECKING, cast

from ._browse import record_cursor, record_page
from ._records import object_value, project_id
from ._workflows import text
from .client import LibraryError

if TYPE_CHECKING:
    from ._records import Records


def _number(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise LibraryError("invalid", f"{name} must be a finite number")
    try:
        number = float(cast(float, value))
    except OverflowError as error:
        raise LibraryError("invalid", f"{name} must be a finite number") from error
    if not math.isfinite(number):
        raise LibraryError("invalid", f"{name} must be a finite number")
    return number


def _validate(document: dict[str, object]) -> None:
    for key in ("runId", "scope", "clientId", "jobId", "principalId", "principalKind"):
        text(document.get(key), key)
    if document.get("jobRef") != document["runId"]:
        raise LibraryError("invalid", "history jobRef must match runId")
    if document.get("state") not in ("completed", "failed", "cancelled", "interrupted"):
        raise LibraryError("invalid", "history requires a terminal state")
    for key in ("submittedAt", "finishedAt"):
        _number(document.get(key), key)
    if document.get("startedAt") is not None:
        _number(document["startedAt"], "startedAt")
    for key in ("priority", "attemptId", "executed", "cached", "skipped"):
        value = document.get(key)
        if type(value) is not int or (
            key != "priority" and value < (1 if key == "attemptId" else 0)
        ):
            raise LibraryError("invalid", f"history {key} is invalid")
    source = document.get("sourceDocument", "")
    if not isinstance(source, str) or (
        source and re.fullmatch(r"blake3:[0-9a-f]{64}", source) is None
    ):
        raise LibraryError("invalid", "history sourceDocument must be a digest")
    if "assetDigests" not in document:
        raise LibraryError("invalid", "history must state whether asset references are known")
    digests = document["assetDigests"]
    if digests is not None and (
        not isinstance(digests, list)
        or any(
            not isinstance(digest, str) or re.fullmatch(r"blake3:[0-9a-f]{64}", digest) is None
            for digest in cast("list[object]", digests)
        )
    ):
        raise LibraryError("invalid", "history assetDigests must be digests or null")
    if document.get("error") is not None:
        object_value(document["error"], "history error")
    receipts = document.get("nodeReceipts")
    if not isinstance(receipts, list) or any(
        not isinstance(receipt, dict) for receipt in cast("list[object]", receipts)
    ):
        raise LibraryError("invalid", "history nodeReceipts must be objects")


def history(records: Records, operation: str, arguments: dict[str, object]) -> dict[str, object]:
    if (records.root / "history.sqlite").exists():
        raise LibraryError(
            "migration", "Finish the complete library migration before using history"
        )
    scope = text(arguments.get("scope"), "history scope")
    if operation == "history.append":
        document = object_value(arguments.get("record"), "history record")
        _validate(document)
        if document["scope"] != scope:
            raise LibraryError("invalid", "history scope must match the record")
        project = project_id(arguments.get("project", "default"))
        if records.get_project(project) is None:
            raise LibraryError("missing", "history project is unavailable")
        with records.db:
            records.db.execute("BEGIN IMMEDIATE")
            inserted = records.db.execute(
                "INSERT INTO finished_history(run_id,scope,project,finished,document) "
                "VALUES (?,?,?,?,?) ON CONFLICT(run_id) DO NOTHING",
                (
                    document["runId"],
                    scope,
                    project,
                    _number(document["finishedAt"], "finishedAt"),
                    json.dumps(document, allow_nan=False),
                ),
            ).rowcount
            row = records.db.execute(
                "SELECT scope,project,document,deleted FROM finished_history WHERE run_id=?",
                (document["runId"],),
            ).fetchone()
            assert row is not None
            if row[0] != scope or row[1] != project:
                raise LibraryError(
                    "invalid", "history run already belongs to another scope or project"
                )
            if inserted:
                records.db.execute(
                    "INSERT INTO changes(kind,id,revision) VALUES ('history',?,1)",
                    (json.dumps([scope, document["runId"]]),),
                )
            return {
                "record": json.loads(row[2]) if not row[3] else None,
                "project": row[1],
                "created": bool(inserted),
            }
    if operation in ("history.get", "history.delete"):
        identifier = text(arguments.get("id"), "history run id")
        if operation == "history.get":
            row = records.db.execute(
                "SELECT project,document FROM finished_history "
                "WHERE scope=? AND run_id=? AND deleted=0",
                (scope, identifier),
            ).fetchone()
            return {"record": json.loads(row[1]), "project": row[0]} if row else {"record": None}
        with records.db:
            deleted = records.db.execute(
                "UPDATE finished_history SET deleted=1,document='{}' "
                "WHERE scope=? AND run_id=? AND deleted=0",
                (scope, identifier),
            ).rowcount
            if deleted:
                records.db.execute(
                    "INSERT INTO changes(kind,id,revision) VALUES ('history',?,2)",
                    (json.dumps([scope, identifier]),),
                )
        return {"deleted": bool(deleted)}
    if operation not in ("history.list", "history.clear"):
        raise LibraryError("operation", f"unknown library operation: {operation}")
    allowed = {"scope", "project", "clientId", "sourceDocument", "state"}
    allowed |= {"limit", "cursor", "after"} if operation == "history.list" else {"before"}
    if set(arguments) - allowed:
        raise LibraryError("invalid", "unknown history query field")
    bound: dict[str, object] = {"scope": scope, "project": arguments.get("project")}
    clauses = ["scope=?", "deleted=0"]
    parameters: list[object] = [scope]
    if bound["project"] is not None:
        clauses.append("project=?")
        parameters.append(project_id(bound["project"]))
    for key in ("clientId", "sourceDocument", "state"):
        value = arguments.get(key, "")
        if not isinstance(value, str):
            raise LibraryError("invalid", f"history {key} filter must be a string")
        bound[key] = value
        if value:
            clauses.append(f"json_extract(document,'$.{key}')=?")
            parameters.append(value)
    if operation == "history.clear":
        if arguments.get("before") is not None:
            clauses.append("finished < ?")
            parameters.append(_number(arguments["before"], "history before"))
        with records.db:
            records.db.execute("BEGIN IMMEDIATE")
            rows = records.db.execute(
                "UPDATE finished_history SET deleted=1,document='{}' WHERE "
                + " AND ".join(clauses)
                + " RETURNING run_id",
                parameters,
            ).fetchall()
            records.db.executemany(
                "INSERT INTO changes(kind,id,revision) VALUES ('history',?,2)",
                [(json.dumps([scope, row[0]]),) for row in rows],
            )
        return {"deleted": len(rows)}
    limit, after = record_page(arguments, bound)
    if "after" in arguments:
        position = arguments["after"]
        if arguments.get("cursor") is not None or not isinstance(position, list):
            raise LibraryError("invalid", "history after requires a position without a cursor")
        values = cast("list[object]", position)
        if len(values) != 2 or not isinstance(values[1], str):
            raise LibraryError("invalid", "history after requires a timestamp and run id")
        after = (_number(values[0], "history after"), text(values[1], "history after run id"))
    if after is not None:
        clauses.append("(finished < ? OR (finished = ? AND run_id < ?))")
        parameters.extend((after[0], after[0], after[1]))
    rows = records.db.execute(
        "SELECT finished,run_id,document FROM finished_history WHERE "
        + " AND ".join(clauses)
        + " ORDER BY finished DESC,run_id DESC LIMIT ?",
        (*parameters, limit + 1),
    ).fetchall()
    return {
        "items": [json.loads(row[2]) for row in rows[:limit]],
        "cursor": record_cursor(bound, rows[limit - 1][:2]) if len(rows) > limit else None,
    }
