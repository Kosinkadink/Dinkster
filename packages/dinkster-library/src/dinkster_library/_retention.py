"""Engine-owned retention generations, holder census and revisioned roots."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from typing import TYPE_CHECKING, cast

from ._manifests import asset_digests
from ._records import object_value
from ._workflows import text
from .client import LibraryError

if TYPE_CHECKING:
    from ._records import Records


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _client(db: sqlite3.Connection, engine: str, arguments: dict[str, object]) -> dict[str, object]:
    principal = text(arguments.get("principalId"), "client principal")
    generation = text(arguments.get("generation"), "client generation")
    row = db.execute(
        "SELECT document FROM retention_clients WHERE engine=? AND principal=? AND generation=?",
        (engine, principal, generation),
    ).fetchone()
    if row is None:
        raise LibraryError("stale", "retention client generation is unknown")
    return object_value(json.loads(row[0]), "retention client")


def _save_client(db: sqlite3.Connection, client: dict[str, object]) -> None:
    db.execute(
        "INSERT INTO retention_clients VALUES (?,?,?,?,?) "
        "ON CONFLICT(engine,id) DO UPDATE SET "
        "generation=excluded.generation,document=excluded.document",
        (
            client["engine"],
            client["clientId"],
            client["principalId"],
            client["generation"],
            _json(client),
        ),
    )


def _invalidate(client: dict[str, object]) -> None:
    client["acknowledged"] = False
    client["revision"] = cast(int, client["revision"]) + 1


def _touch(client: dict[str, object]) -> None:
    client.update(lastActivity=time.time(), departed=False)


def quarantine_status(
    db: sqlite3.Connection, engine: str | None = None, client_id: str | None = None
) -> dict[str, int]:
    clients = (
        "SELECT engine,id FROM retention_clients c WHERE "
        "(? IS NULL OR c.engine=?) AND (? IS NULL OR c.id=?) AND "
        "(json_extract(c.document,'$.departed')=1 OR EXISTS "
        "(SELECT 1 FROM retention_quarantine q WHERE q.engine=c.engine AND q.client_id=c.id))"
    )
    parameters = (engine, engine, client_id, client_id)
    return {
        "departedClients": db.execute(f"SELECT COUNT(*) FROM ({clients})", parameters).fetchone()[
            0
        ],
        "owners": db.execute(
            f"SELECT COUNT(*) FROM retained_owners r JOIN ({clients}) c "
            "ON r.engine=c.engine AND r.client_id=c.id",
            parameters,
        ).fetchone()[0],
        "bytes": db.execute(
            "SELECT COALESCE(SUM(size),0) FROM (SELECT MAX(q.size) size "
            f"FROM retention_quarantine q JOIN ({clients}) c "
            "ON q.engine=c.engine AND q.client_id=c.id GROUP BY q.digest)",
            parameters,
        ).fetchone()[0],
    }


def depart_clients(db: sqlite3.Connection, vaults: object) -> None:
    """Snapshot unreported produced roots before exempting idle clients."""
    clients: list[dict[str, object]] = []
    for (raw,) in db.execute("SELECT document FROM retention_clients"):
        client = object_value(json.loads(raw), "retention client")
        if (
            client.get("departed") is not True
            and (client["connected"] is not True or client["acknowledged"] is not True)
            and client["pendingRequests"] == 0
            and client["activeStreams"] == 0
            and isinstance(last := client.get("lastActivity"), (int, float))
            and not isinstance(last, bool)
            and last <= time.time() - 300
        ):
            clients.append(client)
    from .produced_assets import ProducedAssetInventory
    from .produced_collection import claim_produced_vaults

    # The reference write barrier excludes admissions and owner changes. Claims
    # keep the inventories stable without holding SQLite across filesystem work.
    with claim_produced_vaults(vaults) as roots:
        if not clients:
            return
        inventory = [row for root in roots for row in ProducedAssetInventory(root).entries()]
        with db:
            db.execute("BEGIN IMMEDIATE")
            for client in clients:
                db.executemany(
                    "INSERT INTO retention_quarantine VALUES (?,?,?,?) "
                    "ON CONFLICT(engine,client_id,digest) DO UPDATE SET "
                    "size=MAX(size,excluded.size)",
                    [
                        (client["engine"], client["clientId"], row.digest, row.size)
                        for row in inventory
                    ],
                )
                client["departed"] = True
                _save_client(db, client)


def reset_clients(db: sqlite3.Connection, engine: str) -> None:
    """A restarted publisher forgets observations, not roots or other engines."""
    for (raw,) in db.execute(
        "SELECT document FROM retention_clients WHERE engine=?", (engine,)
    ).fetchall():
        client = object_value(json.loads(raw), "retention client")
        _invalidate(client)
        client.update(connected=False, pendingRequests=0, activeStreams=0)
        if client.get("departed") is not True:
            _touch(client)
        _save_client(db, client)


def _status(db: sqlite3.Connection, client: dict[str, object]) -> dict[str, object]:
    holders: dict[str, list[dict[str, object]]] = {str(n): [] for n in range(1, 9)}
    complete = client["acknowledged"] is True and client["connected"] is True
    complete = complete and client["pendingRequests"] == 0 and client["activeStreams"] == 0
    for raw, classes in db.execute(
        "SELECT document,classes FROM retained_owners WHERE engine=? AND client_id=? "
        "ORDER BY scope,kind,id",
        (client["engine"], client["clientId"]),
    ):
        owner = object_value(json.loads(raw), "retained owner")
        complete = complete and owner.get("complete") is True
        for kind in cast("list[int]", json.loads(classes)):
            holders[str(kind)].append(owner)
    result = dict(client)
    for field in ("acknowledged", "lastActivity", "departed"):
        result.pop(field, None)
    result.update(
        holders=holders,
        complete=complete,
        quarantine=quarantine_status(
            db, cast(str, client["engine"]), cast(str, client["clientId"])
        ),
    )
    return result


def _universe(db: sqlite3.Connection, engine: str) -> dict[str, object]:
    row = db.execute(
        "SELECT document FROM retention_universes WHERE engine=?", (engine,)
    ).fetchone()
    if row is None:
        raise LibraryError("invalid", "initialize this engine's retention census first")
    return object_value(json.loads(row[0]), "retention universe")


def _legacy(db: sqlite3.Connection, engine: str) -> None:
    universe = _universe(db, engine)
    universe["legacy"] = True
    db.execute(
        "UPDATE retention_universes SET document=? WHERE engine=?", (_json(universe), engine)
    )


def _clients(
    records: Records, engine: str, operation: str, arguments: dict[str, object]
) -> dict[str, object]:
    db = records.db
    if operation == "retention.clients.initialize":
        if set(arguments) != {"pristine"} or type(arguments["pristine"]) is not bool:
            raise LibraryError("invalid", "pristine must be a boolean")
        row = db.execute(
            "SELECT document FROM retention_universes WHERE engine=?", (engine,)
        ).fetchone()
        universe = (
            object_value(json.loads(row[0]), "retention universe")
            if row
            else {"closed": False, "legacy": False}
        )
        if not universe["closed"] and not universe["legacy"]:
            existing = (
                db.execute(
                    "SELECT 1 FROM retained_owners WHERE engine=? LIMIT 1", (engine,)
                ).fetchone()
                is not None
            )
            universe.update(
                closed=arguments["pristine"] and not existing,
                legacy=not arguments["pristine"] or existing,
            )
        db.execute(
            "INSERT INTO retention_universes VALUES (?,?) "
            "ON CONFLICT(engine) DO UPDATE SET document=excluded.document",
            (engine, _json(universe)),
        )
        reset_clients(db, engine)
        return {"closed": universe["closed"], "legacy": universe["legacy"]}
    _universe(db, engine)
    allowed = {"principalId", "generation"}
    if operation == "retention.clients.open":
        allowed.add("revision")
    elif operation in ("retention.clients.admit", "retention.clients.settle"):
        allowed.add("stream")
    elif operation == "retention.clients.acknowledge":
        allowed |= {"revision", "holders"}
    elif operation not in ("retention.clients.get", "retention.clients.disconnect"):
        raise LibraryError("operation", f"unknown library operation: {operation}")
    if set(arguments) - allowed:
        raise LibraryError("invalid", "unknown retention client field")
    principal = text(arguments.get("principalId"), "client principal")
    stream = arguments.get("stream", False)
    if (
        operation in ("retention.clients.admit", "retention.clients.settle")
        and type(stream) is not bool
    ):
        raise LibraryError("invalid", "stream must be a boolean")
    generation = arguments.get("generation")
    if operation == "retention.clients.admit" and generation is None:
        _legacy(db, engine)
        return {}
    if operation == "retention.clients.open":
        revision = arguments.get("revision", 0)
        if type(revision) is not int or revision < 0:
            raise LibraryError("invalid", "client revision must be a nonnegative integer")
        if generation is None:
            if revision != 0:
                raise LibraryError("stale", "new clients require revision zero")
            client: dict[str, object] = {
                "clientId": uuid.uuid4().hex,
                "principalId": principal,
                "engine": engine,
            }
        else:
            client = _client(db, engine, arguments)
            if (
                client["revision"] != revision
                or client["pendingRequests"]
                or client["activeStreams"]
            ):
                raise LibraryError("stale", "retention client changed or has pending reads")
        client.update(
            generation=uuid.uuid4().hex,
            revision=revision + 1,
            connected=True,
            acknowledged=False,
            pendingRequests=0,
            activeStreams=0,
        )
        _touch(client)
        _save_client(db, client)
        return _status(db, client)
    client = _client(db, engine, arguments)
    if operation == "retention.clients.get":
        _touch(client)
        _save_client(db, client)
        return _status(db, client)
    if operation == "retention.clients.disconnect":
        _invalidate(client)
        client["connected"] = False
    elif operation == "retention.clients.acknowledge":
        status = _status(db, client)
        revision = arguments.get("revision")
        if (
            client["connected"] is not True
            or client["pendingRequests"]
            or client["activeStreams"]
            or type(revision) is not int
            or revision != client["revision"]
            or _json(arguments.get("holders")) != _json(status["holders"])
        ):
            raise LibraryError("stale", "retention client or holder census changed")
        holders = cast("dict[str, list[dict[str, object]]]", status["holders"])
        if any(
            owner.get("complete") is not True for owners in holders.values() for owner in owners
        ):
            raise LibraryError("invalid", "retention holder is incomplete")
        client["acknowledged"] = True
        db.execute(
            "DELETE FROM retention_quarantine WHERE engine=? AND client_id=?",
            (engine, client["clientId"]),
        )
    else:
        field = "activeStreams" if stream else "pendingRequests"
        if operation == "retention.clients.admit":
            if client["connected"] is not True:
                raise LibraryError("stale", "retention client is disconnected")
            _invalidate(client)
            client[field] = cast(int, client[field]) + 1
        elif cast(int, client[field]) > 0:
            client[field] = cast(int, client[field]) - 1
            if stream:
                _invalidate(client)
    _touch(client)
    _save_client(db, client)
    return _status(db, client) if operation == "retention.clients.acknowledge" else {}


def retained_owner(
    records: Records,
    engine: str,
    operation: str,
    arguments: dict[str, object],
    *,
    source_token: str | None = None,
) -> dict[str, object]:
    db = records.db
    if operation not in (
        "retention.owners.get",
        "retention.owners.put",
        "retention.owners.release",
    ):
        raise LibraryError("operation", f"unknown library operation: {operation}")
    allowed = {"scope", "ownerKind", "ownerId"}
    if operation != "retention.owners.get":
        allowed |= {"revision", "client"}
    if operation == "retention.owners.put":
        allowed.add("patch")
    if set(arguments) - allowed:
        raise LibraryError("invalid", "unknown retained owner field")
    scope = text(arguments.get("scope"), "retention scope").strip()
    kind, identifier = arguments.get("ownerKind"), text(arguments.get("ownerId"), "retention owner")
    if (
        kind not in ("document", "history", "shown-result", "preview")
        or identifier != identifier.strip()
        or len(identifier) > 512
    ):
        raise LibraryError("invalid", "invalid retention owner identity")
    identity = (engine, scope, kind, identifier)
    row = db.execute(
        "SELECT document,client_id,classes FROM retained_owners "
        "WHERE engine=? AND scope=? AND kind=? AND id=?",
        identity,
    ).fetchone()
    document: dict[str, object] | None = (
        object_value(json.loads(row[0]), "retained owner") if row else None
    )
    if operation == "retention.owners.get":
        return {"owner": document}
    _universe(db, engine)
    revision = arguments.get("revision")
    if type(revision) is not int or revision < (1 if operation.endswith(".release") else 0):
        raise LibraryError("invalid", "retention revision is invalid")
    if document is None and operation.endswith(".release"):
        return {"owner": None}
    if revision != (document["revision"] if document is not None else 0):
        raise LibraryError("stale", "retained asset values changed")
    client_id = None
    classes: list[int] = []
    raw_client = arguments.get("client")
    if raw_client is None:
        if row is not None and row[1] is not None:
            raise LibraryError("stale", "retained owner requires its client generation")
        _legacy(db, engine)
    else:
        binding = object_value(raw_client, "retention client binding")
        if set(binding) != {"principalId", "generation", "holderClasses"}:
            raise LibraryError(
                "invalid", "client binding requires principal, generation and classes"
            )
        raw_classes = binding["holderClasses"]
        if (
            not isinstance(raw_classes, list)
            or not raw_classes
            or any(
                type(value) is not int or not 1 <= value <= 8
                for value in cast("list[object]", raw_classes)
            )
        ):
            raise LibraryError("invalid", "holder classes must be distinct integers from 1 to 8")
        classes = sorted(cast("list[int]", raw_classes))
        if len(set(classes)) != len(classes):
            raise LibraryError("invalid", "holder classes must be distinct integers from 1 to 8")
        client = _client(db, engine, binding)
        if client["connected"] is not True:
            raise LibraryError("stale", "retention client is disconnected")
        client_id = client["clientId"]
        if row is not None and (row[1] != client_id or json.loads(row[2]) != classes):
            raise LibraryError("stale", "retained owner belongs to another client or class set")
        _invalidate(client)
        _touch(client)
        _save_client(db, client)
    if document is None:
        document = {
            "engine": engine,
            "scope": scope,
            "ownerKind": kind,
            "ownerId": identifier,
            "digests": [],
            "complete": False,
        }
    if operation.endswith(".release"):
        document.update(digests=[], complete=True)
    else:
        patch = object_value(arguments.get("patch"), "retention patch")
        if set(patch) - {"digests", "complete"}:
            raise LibraryError("invalid", "unknown retained owner patch field")
        if "digests" in patch:
            roots = patch["digests"]
            if not isinstance(roots, list) or len(cast("list[object]", roots)) > 4096:
                raise LibraryError(
                    "invalid", "retained digests must contain at most 4096 identities"
                )
            document["digests"] = asset_digests(cast("list[object]", roots))
        if "complete" in patch:
            if type(patch["complete"]) is not bool:
                raise LibraryError("invalid", "retention completeness must be a boolean")
            document["complete"] = patch["complete"]
    if source_token is None:
        document.pop("sourceToken", None)
    else:
        document["sourceToken"] = source_token
    document["revision"] = revision + 1
    db.execute(
        "INSERT INTO retained_owners VALUES (?,?,?,?,?,?,?) "
        "ON CONFLICT(engine,scope,kind,id) DO UPDATE SET document=excluded.document,"
        "client_id=excluded.client_id,classes=excluded.classes",
        (*identity, _json(document), client_id, _json(classes)),
    )
    return {"owner": document}


def retention(
    records: Records, engine: str, operation: str, arguments: dict[str, object]
) -> dict[str, object]:
    if (records.root / "library.sqlite").exists():
        raise LibraryError("migration", "Finish the complete library migration first")
    if operation == "retention.census":
        if set(arguments) != {"vaults"}:
            raise LibraryError("invalid", "retention census needs produced vault directories")
        from .asset_identity import AssetError

        try:
            depart_clients(records.db, arguments["vaults"])
        except (AssetError, OSError) as error:
            raise LibraryError("references", str(error)) from error
        return {"quarantine": quarantine_status(records.db)}
    handler = _clients if operation.startswith("retention.clients.") else retained_owner
    if operation == "retention.owners.get":
        return handler(records, engine, operation, arguments)
    with records.db:
        records.db.execute("BEGIN IMMEDIATE")
        return handler(records, engine, operation, arguments)


def retained_roots(db: sqlite3.Connection) -> set[str]:
    for (raw,) in db.execute("SELECT document FROM retention_universes"):
        universe = object_value(json.loads(raw), "retention universe")
        if universe.get("closed") is not True or universe.get("legacy") is not False:
            raise LibraryError("references", "retention owner completeness is unknown")
    for (raw,) in db.execute("SELECT document FROM retention_clients"):
        client = object_value(json.loads(raw), "retention client")
        if (
            client["pendingRequests"] != 0
            or client["activeStreams"] != 0
            or (
                client.get("departed") is not True
                and (client["connected"] is not True or client["acknowledged"] is not True)
            )
        ):
            raise LibraryError("references", "client retention census is incomplete")
    roots = set(
        asset_digests([row[0] for row in db.execute("SELECT digest FROM retention_quarantine")])
    )
    for engine, raw, client_id in db.execute(
        "SELECT engine,document,client_id FROM retained_owners"
    ):
        owner = object_value(json.loads(raw), "retained owner")
        if (
            owner.get("complete") is not True
            or client_id is None
            or db.execute(
                "SELECT 1 FROM retention_clients WHERE engine=? AND id=?", (engine, client_id)
            ).fetchone()
            is None
        ):
            raise LibraryError("references", "retained owner completeness is unknown")
        roots.update(asset_digests(owner.get("digests")))
    return roots
