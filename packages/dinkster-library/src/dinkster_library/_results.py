"""Engine-owned result publications, claims and their retained references."""

from __future__ import annotations

import json
import re
import sqlite3
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


def _available(db: sqlite3.Connection, engine: str, offer: dict[str, object]) -> bool:
    if (
        db.execute(
            "SELECT 1 FROM claimed_result_offers WHERE engine=? AND token=?",
            (engine, offer["token"]),
        ).fetchone()
        is None
    ):
        return True
    for (raw,) in db.execute(
        "SELECT document FROM retained_owners WHERE engine=? AND scope=? "
        "AND json_extract(document,'$.sourceToken')=?",
        (engine, offer["scope"], offer["token"]),
    ):
        owner = object_value(json.loads(raw), "retained owner")
        if owner["digests"] == offer["digests"] and owner["complete"] is offer["complete"]:
            return True
    return False


def _claim(records: Records, engine: str, arguments: dict[str, object]) -> dict[str, object]:
    from ._retention import retained_owner

    if set(arguments) - {"scope", "token", "ownerKind", "ownerId", "client"}:
        raise LibraryError("invalid", "unknown result claim field")
    scope = text(arguments.get("scope"), "result scope").strip()
    token = _token(arguments.get("token"))
    if arguments.get("ownerKind") not in ("shown-result", "preview"):
        raise LibraryError("invalid", "result claims require a shown-result or preview owner")
    with records.db:
        records.db.execute("BEGIN IMMEDIATE")
        row = records.db.execute(
            "SELECT document FROM result_offers WHERE engine=? AND scope=? AND token=?",
            (engine, scope, token),
        ).fetchone()
        if row is None:
            return {"owner": None}
        offer = object_value(json.loads(row[0]), "result offer")
        if not _available(records.db, engine, offer):
            return {"owner": None}
        owner = retained_owner(
            records,
            engine,
            "retention.owners.put",
            {
                "scope": scope,
                "ownerKind": arguments.get("ownerKind"),
                "ownerId": arguments.get("ownerId"),
                "client": arguments.get("client"),
                "revision": 0,
                "patch": {"digests": offer["digests"], "complete": offer["complete"]},
            },
            source_token=token,
        )
        records.db.execute(
            "INSERT INTO claimed_result_offers VALUES (?,?) ON CONFLICT DO NOTHING", (engine, token)
        )
    return owner


def result_roots(db: sqlite3.Connection) -> set[str]:
    active: set[tuple[str, str, str]] = set()
    roots: set[str] = set()
    for engine, scope, token in db.execute(
        "SELECT engine,scope,json_extract(document,'$.sourceToken') FROM retained_owners "
        "WHERE json_extract(document,'$.sourceToken') IS NOT NULL"
    ):
        active.add((engine, scope, token))
    for engine, scope, token, raw in db.execute(
        "SELECT o.engine,o.scope,o.token,o.document FROM result_offers o "
        "LEFT JOIN claimed_result_offers c ON c.engine=o.engine AND c.token=o.token "
        "WHERE c.token IS NULL"
    ):
        offer = object_value(json.loads(raw), "result offer")
        if offer.get("complete") is not True:
            raise LibraryError("references", "result offer references are incomplete")
        roots.update(asset_digests(offer.get("digests")))
        active.add((engine, scope, token))
    for engine, scope, token, raw in db.execute(
        "SELECT engine,scope,token,document FROM result_renditions"
    ):
        if (engine, scope, token) in active:
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
    if (records.root / "library.sqlite").exists():
        raise LibraryError("migration", "Finish the complete library migration first")
    if operation == "results.claim":
        return _claim(records, engine, arguments)
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
            if source is None or not _available(
                records.db, engine, object_value(json.loads(source[0]), "result offer")
            ):
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
