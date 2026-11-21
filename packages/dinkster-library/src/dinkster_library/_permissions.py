"""Durable permission overrides and hash-only delegation revocation records."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from typing import TYPE_CHECKING, cast

from ._browse import record_cursor, record_page
from ._records import object_value
from ._workflows import text
from .client import LibraryError

if TYPE_CHECKING:
    from ._records import Records


def _timestamp(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise LibraryError("invalid", f"{name} must be a finite nonnegative number")
    try:
        number = float(cast("float", value))
    except OverflowError as error:
        raise LibraryError("invalid", f"{name} must be a finite nonnegative number") from error
    if not math.isfinite(number) or number < 0:
        raise LibraryError("invalid", f"{name} must be a finite nonnegative number")
    return number


def _hash(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise LibraryError("invalid", "credentialHash must be a canonical SHA-256 hash")
    return value


def permissions(
    records: Records, operation: str, arguments: dict[str, object]
) -> dict[str, object]:
    if any((records.root / name).exists() for name in ("principals.sqlite", "library.sqlite")):
        raise LibraryError("migration", "Finish the complete library migration first")
    if operation.startswith("delegations."):
        return _delegations(records, operation, arguments)
    if operation not in ("permissions.get", "permissions.patch"):
        raise LibraryError("operation", f"unknown library operation: {operation}")
    writing = operation == "permissions.patch"
    allowed = {"principalId", "revision", "patch"} if writing else {"principalId"}
    if set(arguments) - allowed:
        raise LibraryError("invalid", "unknown permission field")
    principal = text(arguments.get("principalId"), "principalId")
    with records.db:
        if writing:
            records.db.execute("BEGIN IMMEDIATE")
        row = records.db.execute(
            "SELECT revision,document FROM permission_records WHERE principal=?", (principal,)
        ).fetchone()
        revision = row[0] if row else 0
        values = object_value(json.loads(row[1]), "stored permissions") if row else {}
        if writing:
            expected = arguments.get("revision")
            if type(expected) is not int or expected != revision:
                raise LibraryError("stale", "permissions changed; read them again before saving")
            patch = object_value(arguments.get("patch"), "permission patch")
            if any(not key.strip() or type(value) is not bool for key, value in patch.items()):
                raise LibraryError(
                    "invalid", "permission categories must be nonempty boolean fields"
                )
            values.update(patch)
            revision += 1
            result = {
                "permissions": {"principalId": principal, "revision": revision, "values": values}
            }
            if len(json.dumps({"result": result}).encode("utf-8")) + 1 > 1024 * 1024:
                raise LibraryError("invalid", "permission reply exceeds the 1 MiB limit")
            records.db.execute(
                "INSERT INTO permission_records VALUES (?,?,?) ON CONFLICT(principal) "
                "DO UPDATE SET revision=excluded.revision,document=excluded.document",
                (principal, revision, json.dumps(values, allow_nan=False)),
            )
            records.db.execute(
                "INSERT INTO changes(kind,id,revision) VALUES ('permissions',?,?)",
                (principal, revision),
            )
    return {"permissions": {"principalId": principal, "revision": revision, "values": values}}


def _delegations(
    records: Records, operation: str, arguments: dict[str, object]
) -> dict[str, object]:
    allowed: dict[str, set[str]] = {
        "delegations.create": {"record"},
        "delegations.get": {"id"},
        "delegations.lookup": {"credentialHash"},
        "delegations.list": {"principalId", "limit", "cursor"},
        "delegations.revoke": {"principalId", "id"},
    }
    if operation not in allowed:
        raise LibraryError("operation", f"unknown library operation: {operation}")
    if set(arguments) - allowed[operation]:
        raise LibraryError("invalid", "unknown delegation field")
    if operation == "delegations.create":
        document = object_value(arguments.get("record"), "delegation record")
        for field in ("id", "principalId", "agentPrincipalId", "scope", "displayName"):
            value = text(document.get(field), field)
            if field == "id" and len(value) > 512:
                raise LibraryError("invalid", "delegation IDs must fit 512 characters")
        credential = _hash(document.get("credentialHash"))
        if document.get("sessionId") is not None:
            text(document["sessionId"], "sessionId")
        created = _timestamp(document.get("createdAt"), "createdAt")
        if document.get("expiresAt") is not None:
            if _timestamp(document["expiresAt"], "expiresAt") <= created:
                raise LibraryError("invalid", "expiresAt must follow createdAt")
        if "revokedAt" in document or "token" in document:
            raise LibraryError("invalid", "delegations cannot supply tokens or revocation state")
        encoded = json.dumps(document, allow_nan=False, sort_keys=True)
        if len(encoded.encode("utf-8")) > 64 * 1024:
            raise LibraryError("invalid", "delegation record exceeds the 64 KiB limit")
        with records.db:
            records.db.execute("BEGIN IMMEDIATE")
            existing = records.db.execute(
                "SELECT document FROM delegation_records WHERE id=?", (document["id"],)
            ).fetchone()
            if existing:
                if existing[0] != encoded:
                    raise LibraryError("conflict", "delegation identity is immutable")
                return {"created": False}
            total, owned = records.db.execute(
                "SELECT count(*),coalesce(sum(principal=?),0) FROM delegation_records "
                "WHERE revoked IS NULL",
                (document["principalId"],),
            ).fetchone()
            if total >= 4096 or owned >= 32:
                raise LibraryError("limit", "delegation limit reached")
            conflict = records.db.execute(
                "SELECT 1 FROM delegation_records WHERE credential=? OR agent=?",
                (credential, document["agentPrincipalId"]),
            ).fetchone()
            if conflict:
                raise LibraryError("conflict", "delegation credential and agent must be unique")
            records.db.execute(
                "INSERT INTO delegation_records VALUES (?,?,?,?,?,?,NULL,?)",
                (
                    document["id"],
                    document["principalId"],
                    document["agentPrincipalId"],
                    credential,
                    created,
                    encoded,
                    1,
                ),
            )
            records.db.execute(
                "INSERT INTO changes(kind,id,revision) VALUES ('delegation',?,1)",
                (document["id"],),
            )
        return {"created": True}
    if operation == "delegations.list":
        principal = text(arguments.get("principalId"), "principalId")
        bound: dict[str, object] = {
            "delegationPrincipal": hashlib.sha256(principal.encode("utf-8")).hexdigest()
        }
        limit, after = record_page(arguments, bound)
        rows = records.db.execute(
            "SELECT created,id,document FROM delegation_records WHERE principal=? "
            "AND revoked IS NULL "
            + ("AND (created,id)>(?,?) " if after else "")
            + "ORDER BY created,id LIMIT ?",
            (principal, *(after or ()), limit + 1),
        ).fetchall()
        items: list[dict[str, object]] = []
        for row in rows[:limit]:
            item = object_value(json.loads(row[2]), "stored delegation")
            item.pop("credentialHash", None)
            items.append(item)
        return {
            "items": items,
            "cursor": record_cursor(bound, rows[limit - 1][:2]) if len(rows) > limit else None,
        }
    if operation == "delegations.revoke":
        principal = text(arguments.get("principalId"), "principalId")
        identifier = text(arguments.get("id"), "delegation id")
        with records.db:
            records.db.execute("BEGIN IMMEDIATE")
            changed = records.db.execute(
                "UPDATE delegation_records SET revoked=?,revision=revision+1 "
                "WHERE principal=? AND id=? AND revoked IS NULL",
                (time.time(), principal, identifier),
            ).rowcount
            if changed:
                records.db.execute(
                    "INSERT INTO changes(kind,id,revision) SELECT 'delegation',id,revision "
                    "FROM delegation_records WHERE id=?",
                    (identifier,),
                )
        return {"revoked": bool(changed)}
    lookup = operation == "delegations.lookup"
    key = (
        _hash(arguments.get("credentialHash"))
        if lookup
        else text(arguments.get("id"), "delegation id")
    )
    row = records.db.execute(
        "SELECT document,revoked FROM delegation_records WHERE "
        + ("credential=?" if lookup else "id=?"),
        (key,),
    ).fetchone()
    if row is None:
        return {"delegation": None}
    document = object_value(json.loads(row[0]), "stored delegation")
    document.pop("credentialHash", None)
    document["revokedAt"] = row[1]
    return {"delegation": document}
