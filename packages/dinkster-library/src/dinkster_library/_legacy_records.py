"""Convert frozen direct-engine databases without importing engine store code."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

from .client import LibraryError


def legacy_store_exists(root: Path, name: str = "library.sqlite") -> bool:
    # Publication retires every legacy store. An old engine may recreate side
    # stores before the library revision fence refuses it; those are not live records.
    # Helper admission refuses an interrupted publication before any consumer runs.
    if (root / "library-format.json").exists():
        return False
    return os.path.lexists(root / name)


def _json(value: object) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _rows(source: sqlite3.Connection, table: str) -> Iterator[sqlite3.Row]:
    # Only constant table names from the legacy schema reach this function.
    if source.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone():
        yield from source.execute(f'SELECT * FROM "{table}"')


def convert_databases(backup: Path, target: sqlite3.Connection, engine: str) -> None:
    """Populate an unpublished records database from a verified source snapshot.

    The caller rebuilds the target from the backed-up records.sqlite on retry.
    No live record is replaced here, and all legacy identity and revision fields
    are copied rather than allocated again through create operations.
    """
    # Create any conversion tables absent from the backed-up helper schema.
    # This connection is unpublished; conversion never populates live records.
    tables = {
        "saved_records": "id TEXT PRIMARY KEY,scope TEXT NOT NULL,project TEXT NOT NULL,"
        "modified REAL NOT NULL,revision INTEGER NOT NULL,document TEXT NOT NULL,"
        "deleted INTEGER NOT NULL DEFAULT 0",
        "media_grants": "scope TEXT NOT NULL,digest TEXT NOT NULL,kind TEXT NOT NULL,"
        "document TEXT NOT NULL,PRIMARY KEY(scope,digest,kind)",
        "asset_observations": "id TEXT PRIMARY KEY,scope TEXT NOT NULL,digest TEXT NOT NULL,"
        "original_name TEXT NOT NULL,stored_name TEXT NOT NULL,kind TEXT NOT NULL,"
        "media_type TEXT NOT NULL,byte_size INTEGER NOT NULL,virtual_path TEXT NOT NULL,"
        "origin TEXT NOT NULL,project TEXT NOT NULL,created REAL NOT NULL,document TEXT NOT NULL",
        "dependency_manifests": "digest TEXT PRIMARY KEY,document TEXT NOT NULL",
        "derivation_manifests": "digest TEXT PRIMARY KEY,document TEXT NOT NULL,asset_digests TEXT",
        "settings_records": "kind TEXT NOT NULL,id TEXT NOT NULL,revision INTEGER NOT NULL,"
        "document TEXT NOT NULL,PRIMARY KEY(kind,id)",
        "result_offers": "engine TEXT NOT NULL,scope TEXT NOT NULL,token TEXT NOT NULL,"
        "document TEXT NOT NULL,PRIMARY KEY(engine,token)",
        "claimed_result_offers": "engine TEXT NOT NULL,token TEXT NOT NULL,"
        "PRIMARY KEY(engine,token)",
        "result_renditions": "engine TEXT NOT NULL,scope TEXT NOT NULL,token TEXT NOT NULL,"
        "cache_key TEXT NOT NULL,document TEXT NOT NULL,PRIMARY KEY(engine,token,cache_key)",
        "terminal_results": "engine TEXT NOT NULL,scope TEXT NOT NULL,job_ref TEXT NOT NULL,"
        "document TEXT NOT NULL,PRIMARY KEY(engine,scope,job_ref)",
        "finished_history": "run_id TEXT PRIMARY KEY,scope TEXT NOT NULL,project TEXT NOT NULL,"
        "finished REAL NOT NULL,document TEXT NOT NULL,deleted INTEGER NOT NULL DEFAULT 0",
        "permission_records": "principal TEXT PRIMARY KEY,revision INTEGER NOT NULL,"
        "document TEXT NOT NULL",
        "delegation_records": "id TEXT PRIMARY KEY,principal TEXT NOT NULL,"
        "agent TEXT NOT NULL UNIQUE,"
        "credential TEXT NOT NULL UNIQUE,created REAL NOT NULL,document TEXT NOT NULL,"
        "revoked REAL,revision INTEGER NOT NULL",
    }
    with target:
        for table, columns in tables.items():
            target.execute(f"CREATE TABLE IF NOT EXISTS {table} ({columns})")
    for name, convert in (
        ("library.sqlite", _assets),
        ("history.sqlite", _history),
        ("principals.sqlite", _permissions),
    ):
        path = backup / "library" / name
        if not path.exists():
            continue
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)) as source:
            source.row_factory = sqlite3.Row
            with target:
                target.execute("BEGIN IMMEDIATE")
                convert(source, target, engine)


def _assets(source: sqlite3.Connection, target: sqlite3.Connection, engine: str) -> None:
    schemas = {
        "0001_asset_records": (
            "records",
            "media_grants",
            "asset_dependencies",
            "asset_derivations",
        ),
        "0002_asset_observations": ("asset_observations",),
        "0003_asset_retentions": ("asset_retentions",),
        "0004_derivation_references": (),
        "0005_retention_completeness": (),
        "0006_cleanup_policy": ("asset_cleanup_policy",),
        "0007_result_offers": ("asset_result_offers",),
        "0008_terminal_results": ("asset_terminal_results",),
        "0009_artifact_offers": (),
        "0010_result_renditions": ("asset_result_renditions",),
        "0011_retention_clients": (
            "asset_retention_universe",
            "asset_retention_clients",
            "asset_retention_members",
        ),
        "0012_retention_departure": ("asset_retention_quarantine",),
        "0013_remove_retention_lists": (),
    }
    versions = [row[0] for row in _rows(source, "alembic_version")]
    if len(versions) != 1 or versions[0] not in schemas:
        raise LibraryError("migration", "Unsupported direct-engine library revision")
    expected = {"alembic_version"}
    for revision, tables in schemas.items():
        expected.update(tables)
        if revision == versions[0]:
            break
    if versions[0] == "0013_remove_retention_lists":
        expected.difference_update(
            {
                "asset_retentions",
                "asset_retention_universe",
                "asset_retention_clients",
                "asset_retention_members",
                "asset_retention_quarantine",
            }
        )
    present = {
        row[0]
        for row in source.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    if present != expected:
        raise LibraryError("migration", "Direct-engine library tables do not match its revision")
    for row in _rows(source, "records"):
        document = dict(row)
        document["mediaType"] = document.pop("media_type")
        document["labels"] = json.loads(row["labels"])
        if not document["folder"]:
            document.pop("folder")
        placement = target.execute(
            "SELECT project FROM workflow_facts WHERE scope=? AND id=?", (row["scope"], row["id"])
        ).fetchone()
        project = placement[0] if placement else "default"
        document["project"] = project
        target.execute(
            "INSERT INTO saved_records VALUES (?,?,?,?,?,?,0)",
            (row["id"], row["scope"], project, row["modified"], row["revision"], _json(document)),
        )
    for row in _rows(source, "media_grants"):
        document = {
            "scope": row["scope"],
            "digest": row["digest"],
            "kind": row["kind"],
            "mediaType": row["media_type"],
            "extension": row["extension"],
            "byteSize": row["byte_size"],
        }
        target.execute(
            "INSERT INTO media_grants VALUES (?,?,?,?)",
            (row["scope"], row["digest"], row["kind"], _json(document)),
        )
    for row in _rows(source, "asset_observations"):
        document = {
            "id": row["id"],
            "asset": {
                "digest": row["digest"],
                "name": row["stored_name"],
                "size": row["byte_size"],
                "mediaType": row["media_type"],
                "virtualPath": row["virtual_path"],
            },
            "originalName": row["original_name"],
            "kind": row["kind"],
            "origin": json.loads(row["origin"]),
            "project": row["project"],
            "created": row["created"],
        }
        target.execute(
            "INSERT INTO asset_observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (*tuple(row), _json(document)),
        )
    for row in _rows(source, "asset_dependencies"):
        target.execute("INSERT INTO dependency_manifests VALUES (?,?)", tuple(row))
    for row in _rows(source, "asset_derivations"):
        target.execute(
            "INSERT INTO derivation_manifests VALUES (?,?,?)",
            (
                row["cache_key"],
                row["manifest"],
                row["asset_digests"] if "asset_digests" in row.keys() else None,
            ),
        )
    for row in _rows(source, "asset_cleanup_policy"):
        target.execute(
            "INSERT INTO settings_records VALUES ('cleanup','policy',1,?)",
            (_json({"limitBytes": row["limit_bytes"]}),),
        )
    for row in _rows(source, "asset_result_offers"):
        document = {
            "engine": engine,
            "token": row["token"],
            "scope": row["scope"],
            "jobRef": row["job_ref"],
            "attemptId": row["attempt"],
            "nodeId": row["node_id"],
            "publication": row["publication"],
            "digests": json.loads(row["digests"]),
            "complete": bool(row["complete"]),
        }
        if "artifact" in row.keys() and row["artifact"] is not None:
            document["artifact"] = json.loads(row["artifact"])
        else:
            document["outputId"] = row["output_id"]
        target.execute(
            "INSERT INTO result_offers VALUES (?,?,?,?)",
            (engine, row["scope"], row["token"], _json(document)),
        )
        if row["claimed"]:
            target.execute("INSERT INTO claimed_result_offers VALUES (?,?)", (engine, row["token"]))
    for row in _rows(source, "asset_result_renditions"):
        document = {
            "documentDigest": row["document_digest"],
            "digests": json.loads(row["digests"]),
        }
        target.execute(
            "INSERT INTO result_renditions VALUES (?,?,?,?,?)",
            (engine, row["scope"], row["source_token"], row["cache_key"], _json(document)),
        )
    for row in _rows(source, "asset_terminal_results"):
        target.execute(
            "INSERT INTO terminal_results VALUES (?,?,?,?)",
            (engine, row["scope"], row["job_ref"], row["manifest"]),
        )


def _history(source: sqlite3.Connection, target: sqlite3.Connection, engine: str) -> None:
    if (
        source.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='history'"
        ).fetchone()
        is None
    ):
        raise LibraryError("migration", "Direct-engine history table is missing")
    for row in _rows(source, "history"):
        document: dict[str, object] = {
            "runId": row["run_id"],
            "jobRef": row["run_id"],
            "scope": row["scope"],
            "principalId": row["principal"] if "principal" in row.keys() else "local",
            "principalKind": row["principal_kind"] if "principal_kind" in row.keys() else "human",
            "clientId": row["client_id"],
            "jobId": row["job_id"],
            "state": row["state"],
            "priority": row["priority"],
            "submittedAt": row["submitted"],
            "finishedAt": row["finished"],
            "attemptId": row["attempt"] if "attempt" in row.keys() else 1,
            "executed": row["executed"],
            "cached": row["cached"],
            "skipped": row["skipped"],
            "nodeReceipts": json.loads(row["node_receipts"])
            if "node_receipts" in row.keys()
            else [],
            "assetDigests": json.loads(row["asset_digests"])
            if "asset_digests" in row.keys() and row["asset_digests"] is not None
            else None,
        }
        if "thumbnail" in row.keys() and row["thumbnail"] is not None:
            document["thumbnail"] = json.loads(row["thumbnail"])
        for column, field in (
            ("started", "startedAt"),
            ("source_document", "sourceDocument"),
            ("error", "error"),
        ):
            if row[column] is not None and row[column] != "":
                document[field] = json.loads(row[column]) if column == "error" else row[column]
        target.execute(
            "INSERT INTO finished_history VALUES (?,?,?,?,?,0)",
            (row["run_id"], row["scope"], "default", row["finished"], _json(document)),
        )


def _permissions(source: sqlite3.Connection, target: sqlite3.Connection, engine: str) -> None:
    principals: dict[str, dict[str, bool]] = {}
    for row in _rows(source, "principal_permissions"):
        principals.setdefault(row["principal_id"], {})[row["category"]] = bool(row["enabled"])
    for principal, values in principals.items():
        target.execute("INSERT INTO permission_records VALUES (?,1,?)", (principal, _json(values)))
    for row in _rows(source, "delegations"):
        document = {
            "id": row["id"],
            "principalId": row["principal_id"],
            "agentPrincipalId": row["agent_principal_id"],
            "credentialHash": row["credential_hash"],
            "scope": row["scope"],
            "sessionId": row["session_id"],
            "displayName": row["display_name"],
            "createdAt": row["created_at"],
            "expiresAt": row["expires_at"],
        }
        if row["revoked_at"] is not None:
            document["revokedAt"] = row["revoked_at"]
        target.execute(
            "INSERT INTO delegation_records VALUES (?,?,?,?,?,?,?,1)",
            (
                row["id"],
                row["principal_id"],
                row["agent_principal_id"],
                row["credential_hash"],
                row["created_at"],
                _json(document),
                row["revoked_at"],
            ),
        )
