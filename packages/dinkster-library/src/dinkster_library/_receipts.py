"""Immutable public-acquisition receipts attested by the owning engine."""

from __future__ import annotations

import json
import math
import os
import re
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit

from ._browse import record_cursor, record_page
from ._records import object_value
from ._workflows import text
from .client import LibraryError

if TYPE_CHECKING:
    from ._records import Records


def _digest(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"blake3:[0-9a-f]{64}", value) is None:
        raise LibraryError("invalid", "asset identity must be a canonical blake3 digest")
    return value


def _receipt_id(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{32}", value) is None:
        raise LibraryError("invalid", "receiptId must be 32 lowercase hexadecimal characters")
    return value


def _validate(document: dict[str, object]) -> float:
    if type(document.get("version")) is not int or document["version"] != 1:
        raise LibraryError("invalid", "public acquisition receipt version must be 1")
    _receipt_id(document.get("receiptId"))
    _digest(document.get("digest"))
    size = document.get("sizeBytes")
    if type(size) is not int or size < 0:
        raise LibraryError("invalid", "receipt sizeBytes must be a nonnegative integer")
    if document.get("sourceType") not in (
        "official-provider",
        "declarative-resolver",
        "code-resolver",
        "manual",
    ):
        raise LibraryError("invalid", "unknown public acquisition source type")
    for field in ("sourceId", "sourceRevision"):
        value = text(document.get(field), field)
        if (
            value != value.strip()
            or len(value) > 512
            or any(ord(char) < 32 or ord(char) == 127 for char in value)
        ):
            raise LibraryError(
                "invalid", f"{field} must be trimmed, bounded and contain no controls"
            )
    for field in ("listedUrl", "finalUrl"):
        value = text(document.get(field), field)
        if len(value) > 4096 or any(char.isspace() or ord(char) < 32 for char in value):
            raise LibraryError("invalid", f"{field} must be a bounded public HTTPS URL")
        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError as error:
            raise LibraryError("invalid", f"{field} is malformed") from error
        if (
            parsed.scheme != "https"
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.netloc.endswith(":")
            or (port is not None and not 1 <= port <= 65535)
        ):
            raise LibraryError(
                "invalid", f"{field} must use HTTPS without userinfo, query or fragment"
            )
    fetched = document.get("fetchedAt")
    if type(fetched) not in (int, float):
        raise LibraryError("invalid", "fetchedAt must be a finite nonnegative timestamp")
    try:
        timestamp = float(cast("float", fetched))
    except OverflowError as error:
        raise LibraryError("invalid", "fetchedAt must be a finite nonnegative timestamp") from error
    if not math.isfinite(timestamp) or timestamp < 0:
        raise LibraryError("invalid", "fetchedAt must be a finite nonnegative timestamp")
    return timestamp


def receipts(records: Records, operation: str, arguments: dict[str, object]) -> dict[str, object]:
    if any(
        os.path.lexists(records.root / name)
        for name in (
            "library.sqlite",
            "provenance.json",
            "resolver-indexes.json",
            "public-acquisition-receipts.json",
        )
    ):
        raise LibraryError("migration", "Finish the complete library migration first")
    allowed = {
        "sources.receipts.put": {"receipt"},
        "sources.receipts.get": {"receiptId"},
        "sources.receipts.list": {"digest", "limit", "cursor"},
    }
    if operation not in allowed:
        raise LibraryError("operation", f"unknown library operation: {operation}")
    if set(arguments) - allowed[operation]:
        raise LibraryError("invalid", "unknown receipt field")
    if operation == "sources.receipts.put":
        document = object_value(arguments.get("receipt"), "public acquisition receipt")
        timestamp = _validate(document)
        encoded = json.dumps(document, allow_nan=False, sort_keys=True)
        # A page of 100 bounded records stays below the private pipe's message limit.
        if len(encoded.encode("utf-8")) > 128 * 1024:
            raise LibraryError("invalid", "public acquisition receipt exceeds the 128 KiB limit")
        with records.db:
            records.db.execute("BEGIN IMMEDIATE")
            existing = records.db.execute(
                "SELECT document FROM acquisition_receipts WHERE id=?", (document["receiptId"],)
            ).fetchone()
            if existing:
                if existing[0] != encoded:
                    raise LibraryError("conflict", "public acquisition receipt is immutable")
                return {"created": False}
            total = records.db.execute("SELECT count(*) FROM acquisition_receipts").fetchone()[0]
            if total >= 100_000:
                raise LibraryError("limit", "public acquisition receipt store is full")
            records.db.execute(
                "INSERT INTO acquisition_receipts VALUES (?,?,?,?)",
                (document["receiptId"], document["digest"], timestamp, encoded),
            )
            records.db.execute(
                "INSERT INTO changes(kind,id,revision) VALUES ('acquisition-receipt',?,1)",
                (document["receiptId"],),
            )
        return {"created": True}
    if operation == "sources.receipts.get":
        identifier = _receipt_id(arguments.get("receiptId"))
        row = records.db.execute(
            "SELECT document FROM acquisition_receipts WHERE id=?", (identifier,)
        ).fetchone()
        return {"receipt": json.loads(row[0]) if row else None}
    digest = arguments.get("digest")
    if digest is not None:
        digest = _digest(digest)
    bound: dict[str, object] = {"acquisitionDigest": digest}
    limit, after = record_page(arguments, bound)
    if limit > 100:
        raise LibraryError("invalid", "receipt page size must be from 1 to 100")
    clauses: list[str] = []
    parameters: list[object] = []
    if digest is not None:
        clauses.append("digest=?")
        parameters.append(digest)
    if after is not None:
        clauses.append("(fetched,id)>(?,?)")
        parameters.extend(after)
    parameters.append(limit + 1)
    rows = records.db.execute(
        "SELECT fetched,id,document FROM acquisition_receipts "
        + ("WHERE " + " AND ".join(clauses) if clauses else "")
        + " ORDER BY fetched,id LIMIT ?",
        parameters,
    ).fetchall()
    return {
        "items": [json.loads(row[2]) for row in rows[:limit]],
        "cursor": record_cursor(bound, rows[limit - 1][:2]) if len(rows) > limit else None,
    }
