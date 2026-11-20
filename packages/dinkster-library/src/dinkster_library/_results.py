"""Engine-owned immutable result publications and their derived references."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, cast

from ._manifests import asset_digest, asset_digests
from ._records import object_value
from ._workflows import text
from .client import LibraryError

if TYPE_CHECKING:
    from ._records import Records


def _token(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{32}", value) is None:
        raise LibraryError("invalid", "result token must be 32 lowercase hex digits")
    return value


def _offer(raw: object, scope: str, engine: str) -> dict[str, object]:
    document = dict(object_value(raw, "result offer"))
    _token(document.get("token"))
    for field in ("scope", "engine"):
        expected = scope if field == "scope" else engine
        if field in document and document[field] != expected:
            raise LibraryError("invalid", f"result offer {field} does not match its publisher")
        document[field] = expected
    for field in ("jobRef", "nodeId"):
        text(document.get(field), field)
    for field in ("attemptId", "publication"):
        value = document.get(field)
        if type(value) is not int or value < 1:
            raise LibraryError("invalid", f"{field} must be a positive integer")
    if type(document.get("complete")) is not bool:
        raise LibraryError("invalid", "result completeness must be a boolean")
    roots = asset_digests(document.get("digests"))
    document["digests"] = roots
    document["complete"] = document["complete"] and len(roots) <= 4096
    if "artifact" in document:
        if "outputId" in document:
            raise LibraryError("invalid", "result offer cannot name both an artifact and an output")
        # The engine verifies the descriptor's bytes and shape before publication.
        # References are explicit; the helper never infers them from opaque metadata.
        object_value(document["artifact"], "result artifact")
    else:
        text(document.get("outputId"), "outputId")
    return document


def results(
    records: Records, engine: str, operation: str, arguments: dict[str, object]
) -> dict[str, object]:
    if (records.root / "library.sqlite").exists():
        raise LibraryError("migration", "Finish the complete library migration first")
    if operation not in (
        "results.offers.put",
        "results.offers.get",
        "results.renditions.put",
        "results.renditions.get",
    ):
        raise LibraryError("operation", f"unknown library operation: {operation}")
    offer = operation.startswith("results.offers.")
    writing = operation.endswith(".put")
    allowed = {"scope", "offer"} if offer and writing else {"scope", "token"}
    if not offer:
        allowed.add("cacheKey")
        if writing:
            allowed.add("rendition")
    if set(arguments) - allowed:
        raise LibraryError("invalid", "unknown result publication field")
    scope = text(arguments.get("scope"), "result scope").strip()
    key = "offer" if offer else "rendition"
    document: dict[str, object] = {}
    cache_key = ""
    if offer and writing:
        document = _offer(arguments.get("offer"), scope, engine)
        token = cast(str, document["token"])
    else:
        token = _token(arguments.get("token"))
    if not offer:
        cache_key = text(arguments.get("cacheKey"), "result rendition cache identity")
        if writing:
            document = dict(object_value(arguments.get("rendition"), "result rendition"))
            asset_digest(document.get("documentDigest"))
            document["digests"] = asset_digests(document.get("digests"))
    table = "result_offers" if offer else "result_renditions"
    predicate = "engine=? AND scope=? AND token=?"
    identity: tuple[str, ...] = (engine, scope, token)
    if not offer:
        predicate += " AND cache_key=?"
        identity += (cache_key,)
    if not writing:
        row = records.db.execute(
            f"SELECT document FROM {table} WHERE {predicate}", identity
        ).fetchone()
        return {key: json.loads(row[0]) if row else None}
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with records.db:
        records.db.execute("BEGIN IMMEDIATE")
        if not offer:
            source = records.db.execute(
                "SELECT 1 FROM result_offers WHERE engine=? AND scope=? AND token=?",
                (engine, scope, token),
            ).fetchone()
            if source is None:
                return {"rendition": None, "created": False}
        if offer:
            inserted = records.db.execute(
                "INSERT INTO result_offers VALUES (?,?,?,?) ON CONFLICT(engine,token) DO NOTHING",
                (engine, scope, token, encoded),
            ).rowcount
        else:
            inserted = records.db.execute(
                "INSERT INTO result_renditions VALUES (?,?,?,?,?) "
                "ON CONFLICT(engine,token,cache_key) DO NOTHING",
                (engine, scope, token, cache_key, encoded),
            ).rowcount
        row = records.db.execute(
            f"SELECT document FROM {table} WHERE {predicate}", identity
        ).fetchone()
        if (
            row is None
            or json.dumps(json.loads(row[0]), sort_keys=True, separators=(",", ":")) != encoded
        ):
            raise LibraryError("conflict", "result publication is immutable")
    return {key: document, "created": bool(inserted)}
