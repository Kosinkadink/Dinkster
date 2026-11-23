"""Engine-owned result publications and claims."""

from __future__ import annotations

import json
import re
import sqlite3
from typing import TYPE_CHECKING, cast

from ._legacy_records import legacy_store_exists
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


def result_roots(db: sqlite3.Connection) -> set[str]:
    roots: set[str] = set()
    for (raw,) in db.execute("SELECT document FROM result_offers"):
        offer = object_value(json.loads(raw), "result offer")
        roots.update(asset_digests(offer.get("digests")))
    for (raw,) in db.execute("SELECT document FROM result_renditions"):
        rendition = object_value(json.loads(raw), "result rendition")
        roots.update(asset_digests(rendition.get("digests")))
    return roots


def _terminal(
    records: Records, engine: str, operation: str, arguments: dict[str, object]
) -> dict[str, object]:
    writing = operation == "results.terminal.put"
    allowed = {"scope", "jobRef"}
    if writing:
        allowed |= {"attemptId", "tokens", "state", "identity", "artifactsComplete"}
    if set(arguments) - allowed:
        raise LibraryError("invalid", "unknown terminal result field")
    scope = text(arguments.get("scope"), "terminal result scope").strip()
    job_ref = text(arguments.get("jobRef"), "terminal result job")
    key = (engine, scope, job_ref)
    if not writing:
        row = records.db.execute(
            "SELECT document FROM terminal_results WHERE engine=? AND scope=? AND job_ref=?", key
        ).fetchone()
        return {"result": json.loads(row[0]) if row else None}
    attempt, state = arguments.get("attemptId"), arguments.get("state")
    complete_artifacts = arguments.get("artifactsComplete")
    if type(attempt) is not int or attempt < 1:
        raise LibraryError("invalid", "terminal result attempt must be a positive integer")
    if state not in ("completed", "failed", "cancelled"):
        raise LibraryError("invalid", "terminal result state is invalid")
    if type(complete_artifacts) is not bool:
        raise LibraryError("invalid", "terminal artifact completeness must be a boolean")
    identity = object_value(arguments.get("identity"), "terminal result identity")
    raw_tokens = arguments.get("tokens")
    if not isinstance(raw_tokens, list):
        raise LibraryError("invalid", "terminal result tokens must be a list")
    tokens = [_token(value) for value in cast("list[object]", raw_tokens)]
    with records.db:
        records.db.execute("BEGIN IMMEDIATE")
        outputs: dict[str, dict[str, dict[str, object]]] = {}
        artifacts: list[dict[str, object]] = []
        complete = state == "completed" and complete_artifacts
        for token in tokens:
            row = records.db.execute(
                "SELECT document FROM result_offers WHERE engine=? AND scope=? AND token=?",
                (engine, scope, token),
            ).fetchone()
            if row is None:
                raise LibraryError("invalid", "terminal result offer is unavailable")
            offer = object_value(json.loads(row[0]), "terminal result offer")
            if offer["jobRef"] != job_ref or offer["attemptId"] != attempt:
                raise LibraryError("invalid", "terminal result offer belongs to another execution")
            if "artifact" in offer:
                artifact = object_value(offer["artifact"], "terminal result artifact")
                index = artifact.get("index")
                if not complete_artifacts or type(index) is not int or index != len(artifacts):
                    raise LibraryError("invalid", "terminal artifact publication is ambiguous")
                artifacts.append(offer)
            else:
                node = outputs.setdefault(cast(str, offer["nodeId"]), {})
                output = cast(str, offer["outputId"])
                if output in node:
                    raise LibraryError("invalid", "terminal result publication is ambiguous")
                node[output] = offer
            complete = complete and offer["complete"] is True
        result: dict[str, object] = {
            "engine": engine,
            "scope": scope,
            "jobRef": job_ref,
            "attemptId": attempt,
            "state": state,
            "identity": identity,
            "resultOffers": outputs,
            "artifacts": artifacts if complete_artifacts else None,
            "complete": complete,
        }
        encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        inserted = records.db.execute(
            "INSERT INTO terminal_results VALUES (?,?,?,?) "
            "ON CONFLICT(engine,scope,job_ref) DO NOTHING",
            (*key, encoded),
        ).rowcount
        row = records.db.execute(
            "SELECT document FROM terminal_results WHERE engine=? AND scope=? AND job_ref=?", key
        ).fetchone()
        if row[0] != encoded:
            raise LibraryError("conflict", "terminal result publication is immutable")
    return {"result": result, "created": bool(inserted)}


def results(
    records: Records, engine: str, operation: str, arguments: dict[str, object]
) -> dict[str, object]:
    if legacy_store_exists(records.root):
        raise LibraryError("migration", "Finish the complete library migration first")
    if operation in ("results.terminal.put", "results.terminal.get"):
        return _terminal(records, engine, operation, arguments)
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
                "SELECT document FROM result_offers WHERE engine=? AND scope=? AND token=?",
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
