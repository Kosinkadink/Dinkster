"""Helper-owned project transactions. Engines never import this module."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from filelock import FileLock, Timeout

from ._references import References
from .client import LibraryError

FORMAT = 2
CODE_VERSION = 2


def object_value(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise LibraryError("invalid", f"{name} must be an object")
    return cast("dict[str, object]", value)


def project_id(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", value) is None:
        raise LibraryError("invalid", "project id must match [a-z0-9][a-z0-9-]{0,63}")
    return value


class Records:
    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self._references = References(root)
        from ._resolvers import Resolvers

        self._resolvers = Resolvers(self)
        self._attachment = FileLock(root / f"helper-{uuid.uuid4()}.lock")
        self._connection: sqlite3.Connection | None = None
        try:
            # Admission and upgrades share this gate, so nobody can attach in the
            # interval between checking the live holders and changing the format.
            with FileLock(root / "records-format.lock"):
                marker = root / "library-format.json"
                if marker.exists():
                    required = object_value(json.loads(marker.read_text("utf-8")), "library format")
                    if required != {
                        "format": "dinkster.library-format/1",
                        "requiredCapabilities": ["library-helper/1"],
                        "libraryApi": 1,
                    }:
                        raise LibraryError("format", "This library requires newer library code")
                journal = root / "record-migration.json"
                if journal.exists():
                    migration = object_value(
                        json.loads(journal.read_text("utf-8")), "record migration journal"
                    )
                    if "publication" in migration and migration.get("complete") is not True:
                        raise LibraryError(
                            "migration", "Finish the interrupted library migration first"
                        )
                self._attachment.acquire()
                self._connection = sqlite3.connect(root / "records.sqlite", timeout=30)
                db = self._connection
                current = db.execute("PRAGMA user_version").fetchone()[0]
                if current not in (0, 1, FORMAT):
                    raise LibraryError(
                        "format",
                        f"This library needs format {current}; this helper carries {FORMAT}.",
                    )
                if current != FORMAT:
                    others = self.attachments(root)
                    if others:
                        raise LibraryError(
                            "attached",
                            "Stop other library helpers before upgrading: " + ", ".join(others),
                        )
                if current == 0:
                    db.executescript(
                        "BEGIN IMMEDIATE;"
                        "CREATE TABLE metadata (minimum_code INTEGER NOT NULL);"
                        "INSERT INTO metadata VALUES (1);"
                        "CREATE TABLE projects (id TEXT PRIMARY KEY, document TEXT NOT NULL);"
                        "CREATE TABLE changes (sequence INTEGER PRIMARY KEY AUTOINCREMENT, "
                        "kind TEXT NOT NULL, id TEXT NOT NULL, revision INTEGER NOT NULL);"
                        "CREATE TABLE workflow_facts (scope TEXT NOT NULL, id TEXT NOT NULL, "
                        "project TEXT NOT NULL, document TEXT NOT NULL, PRIMARY KEY(scope,id));"
                        "CREATE TABLE engines (environment TEXT NOT NULL, snapshot TEXT NOT NULL, "
                        "document TEXT NOT NULL, PRIMARY KEY(environment,snapshot));"
                        "PRAGMA user_version=1; COMMIT;"
                    )
                minimum = db.execute("SELECT minimum_code FROM metadata").fetchone()[0]
                if minimum > CODE_VERSION:
                    raise LibraryError(
                        "format",
                        f"This library needs library code {minimum} or later. "
                        f"This helper carries {CODE_VERSION}.",
                    )
                if current != FORMAT:
                    if current != 0:
                        backup_root = (
                            root
                            / "backups"
                            / (
                                "before-format-2-"
                                + datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
                                + "-"
                                + uuid.uuid4().hex
                            )
                        )
                        backup_root.mkdir(parents=True)
                        backup = sqlite3.connect(backup_root / "records.sqlite")
                        try:
                            db.backup(backup)
                        finally:
                            backup.close()
                    db.executescript(
                        "BEGIN IMMEDIATE;"
                        "CREATE UNIQUE INDEX projects_uid "
                        "ON projects(json_extract(document,'$.uid'));"
                        "CREATE INDEX workflow_project ON workflow_facts(project);"
                        "UPDATE metadata SET minimum_code=2;"
                        "PRAGMA user_version=2; COMMIT;"
                    )
                # Backup readers can prevent Windows deletion. Keep the library
                # usable and retry maintenance on the next attachment.
                try:
                    backups = sorted(
                        (root / "backups").glob("before-format-*/records.sqlite"),
                        key=lambda path: path.stat().st_mtime_ns,
                    )
                    for obsolete in backups[:-3]:
                        obsolete.unlink()
                        obsolete.parent.rmdir()
                except OSError:
                    pass
                db.execute("PRAGMA journal_mode=WAL")
                with db:
                    db.execute("BEGIN IMMEDIATE")
                    # Additive tables leave older helpers' project records readable.
                    db.execute(
                        "CREATE TABLE IF NOT EXISTS resolver_configuration ("
                        "id INTEGER PRIMARY KEY CHECK(id=1),document TEXT NOT NULL)"
                    )
                    db.execute(
                        "CREATE TABLE IF NOT EXISTS resolver_subscriptions ("
                        "id TEXT PRIMARY KEY,source TEXT NOT NULL,"
                        "revision INTEGER NOT NULL,payload BLOB)"
                    )
                    db.execute(
                        "CREATE UNIQUE INDEX IF NOT EXISTS resolver_source "
                        "ON resolver_subscriptions(source) WHERE payload IS NOT NULL"
                    )
                    db.execute(
                        "CREATE TABLE IF NOT EXISTS provenance_layers ("
                        "position INTEGER PRIMARY KEY AUTOINCREMENT,source TEXT NOT NULL UNIQUE)"
                    )
                    db.execute(
                        "CREATE TABLE IF NOT EXISTS provenance_records ("
                        "source TEXT NOT NULL,digest TEXT NOT NULL,document TEXT NOT NULL,"
                        "PRIMARY KEY(source,digest))"
                    )
                    db.execute(
                        "CREATE INDEX IF NOT EXISTS provenance_digest ON provenance_records(digest)"
                    )
                    db.execute(
                        "CREATE TABLE IF NOT EXISTS acquisition_receipts ("
                        "id TEXT PRIMARY KEY,digest TEXT NOT NULL,fetched REAL NOT NULL,"
                        "document TEXT NOT NULL)"
                    )
                    db.execute(
                        "CREATE INDEX IF NOT EXISTS receipts_order "
                        "ON acquisition_receipts(fetched,id)"
                    )
                    db.execute(
                        "CREATE INDEX IF NOT EXISTS receipts_digest "
                        "ON acquisition_receipts(digest,fetched,id)"
                    )
                    db.execute(
                        "CREATE TABLE IF NOT EXISTS permission_records ("
                        "principal TEXT PRIMARY KEY,revision INTEGER NOT NULL,"
                        "document TEXT NOT NULL)"
                    )
                    db.execute(
                        "CREATE TABLE IF NOT EXISTS delegation_records ("
                        "id TEXT PRIMARY KEY,principal TEXT NOT NULL,agent TEXT NOT NULL UNIQUE,"
                        "credential TEXT NOT NULL UNIQUE,created REAL NOT NULL,"
                        "document TEXT NOT NULL,"
                        "revoked REAL,revision INTEGER NOT NULL)"
                    )
                    db.execute(
                        "CREATE INDEX IF NOT EXISTS delegations_principal "
                        "ON delegation_records(principal,created,id)"
                    )
                    db.execute(
                        "CREATE TABLE IF NOT EXISTS settings_records ("
                        "kind TEXT NOT NULL,id TEXT NOT NULL,revision INTEGER NOT NULL,"
                        "document TEXT NOT NULL,PRIMARY KEY(kind,id))"
                    )
                    for table in (
                        "retention_universes",
                        "retention_clients",
                        "retention_quarantine",
                        "retained_owners",
                    ):
                        db.execute(f"DROP TABLE IF EXISTS {table}")
                    db.execute(
                        "CREATE TABLE IF NOT EXISTS terminal_results ("
                        "engine TEXT NOT NULL,scope TEXT NOT NULL,job_ref TEXT NOT NULL,"
                        "document TEXT NOT NULL,PRIMARY KEY(engine,scope,job_ref))"
                    )
                    db.execute(
                        "CREATE TABLE IF NOT EXISTS claimed_result_offers ("
                        "engine TEXT NOT NULL,token TEXT NOT NULL,PRIMARY KEY(engine,token))"
                    )
                    db.execute(
                        "CREATE TABLE IF NOT EXISTS result_offers ("
                        "engine TEXT NOT NULL,scope TEXT NOT NULL,token TEXT NOT NULL,"
                        "document TEXT NOT NULL,PRIMARY KEY(engine,token))"
                    )
                    db.execute(
                        "CREATE TABLE IF NOT EXISTS result_renditions ("
                        "engine TEXT NOT NULL,scope TEXT NOT NULL,token TEXT NOT NULL,"
                        "cache_key TEXT NOT NULL,document TEXT NOT NULL,"
                        "PRIMARY KEY(engine,token,cache_key))"
                    )
                    db.execute(
                        "CREATE TABLE IF NOT EXISTS media_grants ("
                        "scope TEXT NOT NULL,digest TEXT NOT NULL,kind TEXT NOT NULL,"
                        "document TEXT NOT NULL,PRIMARY KEY(scope,digest,kind))"
                    )
                    db.execute(
                        "CREATE TABLE IF NOT EXISTS asset_observations ("
                        "id TEXT PRIMARY KEY,scope TEXT NOT NULL,digest TEXT NOT NULL,"
                        "original_name TEXT NOT NULL,stored_name TEXT NOT NULL,kind TEXT NOT NULL,"
                        "media_type TEXT NOT NULL,byte_size INTEGER NOT NULL,"
                        "virtual_path TEXT NOT NULL,origin TEXT NOT NULL,project TEXT NOT NULL,"
                        "created REAL NOT NULL,document TEXT NOT NULL)"
                    )
                    db.execute(
                        "CREATE INDEX IF NOT EXISTS asset_observations_scope_digest "
                        "ON asset_observations(scope,project,digest,created)"
                    )
                    db.execute(
                        "CREATE TABLE IF NOT EXISTS saved_records ("
                        "id TEXT PRIMARY KEY,scope TEXT NOT NULL,project TEXT NOT NULL,"
                        "modified REAL NOT NULL,revision INTEGER NOT NULL,document TEXT NOT NULL,"
                        "deleted INTEGER NOT NULL DEFAULT 0)"
                    )
                    db.execute(
                        "CREATE INDEX IF NOT EXISTS saved_records_scope "
                        "ON saved_records(scope,modified DESC,id DESC)"
                    )
                    db.execute(
                        "CREATE TABLE IF NOT EXISTS dependency_manifests ("
                        "digest TEXT PRIMARY KEY,document TEXT NOT NULL)"
                    )
                    db.execute(
                        "CREATE TABLE IF NOT EXISTS derivation_manifests ("
                        "digest TEXT PRIMARY KEY,document TEXT NOT NULL,asset_digests TEXT)"
                    )
                    db.execute(
                        "CREATE TABLE IF NOT EXISTS engine_reference_manifests ("
                        "engine TEXT PRIMARY KEY,document TEXT NOT NULL)"
                    )
                    db.execute(
                        "CREATE TABLE IF NOT EXISTS finished_history ("
                        "run_id TEXT PRIMARY KEY,scope TEXT NOT NULL,project TEXT NOT NULL,"
                        "finished REAL NOT NULL,document TEXT NOT NULL,"
                        "deleted INTEGER NOT NULL DEFAULT 0)"
                    )
                    db.execute(
                        "CREATE INDEX IF NOT EXISTS finished_history_scope "
                        "ON finished_history(scope,finished DESC,run_id DESC)"
                    )
                    if self.get_project("default") is None:
                        self._register("default", "Default")
        except BaseException:
            self.close()
            raise

    @property
    def db(self) -> sqlite3.Connection:
        assert self._connection is not None
        return self._connection

    def attachments(self, root: Path) -> list[str]:
        holders: list[str] = []
        for path in root.glob("helper-*.lock"):
            if path == Path(self._attachment.lock_file):
                continue
            lock = FileLock(path)
            try:
                lock.acquire(timeout=0)
            except Timeout:
                holders.append(path.stem)
            else:
                lock.release()
        return holders

    def close(self) -> None:
        self._references.close()
        if self._connection is not None:
            self._connection.close()
        self._attachment.release()

    def get_project(self, identifier: str) -> dict[str, object] | None:
        row = self.db.execute("SELECT document FROM projects WHERE id=?", (identifier,)).fetchone()
        return object_value(json.loads(row[0]), "project") if row else None

    def _save(self, document: dict[str, object]) -> dict[str, object]:
        self.db.execute(
            "INSERT INTO projects VALUES (?,?) ON CONFLICT(id) "
            "DO UPDATE SET document=excluded.document",
            (document["id"], json.dumps(document, allow_nan=False)),
        )
        self.db.execute(
            "INSERT INTO changes(kind,id,revision) VALUES ('project',?,?)",
            (document["id"], document["revision"]),
        )
        return document

    def _register(self, identifier: str, name: object) -> dict[str, object]:
        if not isinstance(name, str) or not name.strip():
            raise LibraryError("invalid", "project name must be nonempty")
        now = datetime.now(UTC).isoformat()
        return self._save(
            {
                "id": identifier,
                "uid": str(uuid.uuid4()),
                "name": name,
                "created": now,
                "modified": now,
                "revision": 1,
                "packs": {"extra": []},
                "environment": None,
                "accountProject": None,
                "deleted": False,
            }
        )

    def engines(self, operation: str, arguments: dict[str, object]) -> dict[str, object]:
        if operation == "engines.seen":
            from ._workflows import text

            identity = {
                key: text(arguments.get(key), key)
                for key in ("environment", "snapshot", "engineVersion")
            }
            api = arguments.get("apiVersion")
            if type(api) is not int or api < 1:
                raise LibraryError("invalid", "apiVersion must be a positive integer")
            with self.db:
                self.db.execute("BEGIN IMMEDIATE")
                previous = self.db.execute(
                    "SELECT document FROM engines WHERE environment=? AND snapshot=?",
                    (identity["environment"], identity["snapshot"]),
                ).fetchone()
                document = object_value(json.loads(previous[0]), "engine") if previous else {}
                document.update(identity)
                document.update(apiVersion=api, lastSeen=datetime.now(UTC).isoformat())
                self.db.execute(
                    "INSERT INTO engines VALUES (?,?,?) ON CONFLICT(environment,snapshot) "
                    "DO UPDATE SET document=excluded.document",
                    (identity["environment"], identity["snapshot"], json.dumps(document)),
                )
            return {"engine": document}
        cursor = arguments.get("cursor", ["", ""])
        limit = arguments.get("limit", 100)
        if (
            not isinstance(cursor, list)
            or len(cast("list[object]", cursor)) != 2
            or any(not isinstance(value, str) for value in cast("list[object]", cursor))
            or type(limit) is not int
            or not 1 <= limit <= 1000
        ):
            raise LibraryError("invalid", "invalid engine cursor or page size")
        rows = self.db.execute(
            "SELECT environment,snapshot,document FROM engines WHERE (environment,snapshot)>(?,?) "
            "ORDER BY environment,snapshot LIMIT ?",
            (*cast("list[str]", cursor), limit + 1),
        ).fetchall()
        return {
            "items": [json.loads(row[2]) for row in rows[:limit]],
            "cursor": list(rows[limit - 1][:2]) if len(rows) > limit else None,
        }

    def dispatch(
        self,
        operation: str,
        arguments: dict[str, object],
        *,
        progress: Callable[[], None] | None = None,
    ) -> dict[str, object]:
        if operation.startswith("sources.resolvers."):
            return self._resolvers.dispatch(operation, arguments)
        if operation.startswith("sources.provenance."):
            from ._provenance import provenance

            return provenance(self, operation, arguments)
        if operation.startswith("sources.receipts."):
            from ._receipts import receipts

            return receipts(self, operation, arguments)
        if operation.startswith(("permissions.", "delegations.")):
            from ._permissions import permissions

            return permissions(self, operation, arguments)
        if operation.startswith("folders.models."):
            from ._settings import model_preferences

            return model_preferences(operation, arguments)
        if operation.startswith(("settings.", "folders.", "cleanup.policy.")):
            from ._settings import preferences

            return preferences(self, operation, arguments)
        if operation.startswith("references."):
            return self._references.dispatch(self.db, operation, arguments, progress=progress)
        if operation.startswith("results."):
            from ._results import results

            engine = self._references.engine_id
            if operation.endswith(".get"):
                return results(self, engine, operation, arguments)
            with self._references.writing():
                return results(self, engine, operation, arguments)
        if operation.startswith(("dependencies.", "derivations.")):
            from ._manifests import manifests

            if operation.endswith(".put"):
                with self._references.writing():
                    return manifests(self, operation, arguments)
            return manifests(self, operation, arguments)
        if operation.startswith("history."):
            from ._history import history

            if operation in ("history.append", "history.delete", "history.clear"):
                with self._references.writing():
                    return history(self, operation, arguments)
            return history(self, operation, arguments)
        if operation in ("grants.get", "grants.put"):
            from .media import grants

            if operation == "grants.get":
                return grants(self, operation, arguments)
            with self._references.writing():
                return grants(self, operation, arguments)
        if operation == "assets.record":
            from .catalog import record_asset

            with self._references.writing():
                return record_asset(self, arguments)
        if operation == "assets.recorded":
            from .catalog import recorded_assets

            return recorded_assets(self, arguments)
        if operation == "assets.list":
            from ._browse import list_assets

            return list_assets(self, arguments)
        if operation == "records.list":
            from ._browse import list_records

            return list_records(self, arguments)
        if operation in ("records.create", "records.get", "records.patch", "records.delete"):
            from ._saved import saved

            if operation == "records.get":
                return saved(self, operation, arguments)
            with self._references.writing():
                return saved(self, operation, arguments)
        if operation == "compatibility.affected":
            from ._compatibility import affected

            return affected(self, arguments)
        if operation.startswith("workflows."):
            from ._workflows import workflow

            return workflow(self, operation, arguments)
        if operation == "projects.needs":
            from ._workflows import needs

            return needs(
                self,
                project_id(arguments.get("id")),
                include_workflows=arguments.get("includeWorkflows", False),
            )
        if operation in ("engines.seen", "engines.list"):
            return self.engines(operation, arguments)
        if operation == "projects.list":
            cursor = arguments.get("cursor", "")
            limit = arguments.get("limit", 100)
            if not isinstance(cursor, str) or type(limit) is not int or not 1 <= limit <= 1000:
                raise LibraryError("invalid", "invalid project cursor or page size")
            environment = arguments.get("environment")
            include_deleted = arguments.get("includeDeleted", False)
            if environment is not None and not isinstance(environment, str):
                raise LibraryError("invalid", "environment filter must be a string")
            if type(include_deleted) is not bool:
                raise LibraryError("invalid", "includeDeleted must be a boolean")
            rows = self.db.execute(
                "SELECT document FROM projects WHERE id>? "
                "AND (? OR json_extract(document,'$.deleted')=0) "
                "AND (? IS NULL OR json_extract(document,'$.environment.id')=?) "
                "ORDER BY id LIMIT ?",
                (cursor, include_deleted, environment, environment, limit + 1),
            ).fetchall()
            documents = [object_value(json.loads(row[0]), "project") for row in rows]
            page = documents[:limit]
            return {"items": page, "cursor": page[-1]["id"] if len(documents) > limit else None}
        if operation == "projects.get":
            return {"project": self.get_project(project_id(arguments.get("id")))}
        if operation == "changes.head":
            return {
                "cursor": self.db.execute(
                    "SELECT COALESCE(MAX(sequence),0) FROM changes"
                ).fetchone()[0]
            }
        if operation == "changes":
            after = arguments.get("cursor", 0)
            limit = arguments.get("limit", 100)
            if (
                type(after) is not int
                or after < 0
                or type(limit) is not int
                or not 1 <= limit <= 1000
            ):
                raise LibraryError("invalid", "invalid change cursor or page size")
            rows = self.db.execute(
                "SELECT sequence,kind,id,revision FROM changes WHERE sequence>? "
                "ORDER BY sequence LIMIT ?",
                (after, limit),
            ).fetchall()
            return {
                "items": [
                    dict(zip(("cursor", "kind", "id", "revision"), row, strict=True))
                    for row in rows
                ],
                "cursor": rows[-1][0] if rows else after,
            }
        if operation not in ("projects.register", "projects.patch", "projects.delete"):
            raise LibraryError("operation", f"unknown library operation: {operation}")
        identifier = project_id(arguments.get("id"))
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            document = self.get_project(identifier)
            if operation == "projects.register":
                return {
                    "project": document
                    if document is not None
                    else self._register(identifier, arguments.get("name", identifier))
                }
            if document is None:
                raise LibraryError("missing", "project does not exist")
            revision = arguments.get("revision")
            if type(revision) is not int or revision != document["revision"]:
                raise LibraryError("stale", "project changed; read it again before saving")
            if operation == "projects.delete":
                if identifier == "default":
                    raise LibraryError("invalid", "the default project cannot be deleted")
                patch: dict[str, object] = {"deleted": True}
            else:
                patch = object_value(arguments.get("patch"), "project patch")
                if set(patch) - {"name", "packs", "environment", "accountProject"}:
                    raise LibraryError("invalid", "unknown or immutable project patch field")
                if "name" in patch and (
                    not isinstance(patch["name"], str) or not patch["name"].strip()
                ):
                    raise LibraryError("invalid", "project name must be nonempty")
                if "packs" in patch:
                    packs = object_value(patch["packs"], "packs patch")
                    extra = packs.get("extra")
                    if (
                        set(packs) != {"extra"}
                        or not isinstance(extra, list)
                        or any(
                            not isinstance(pack, str) or not pack
                            for pack in cast("list[object]", extra)
                        )
                    ):
                        raise LibraryError("invalid", "packs patch must contain an extra list")
                    patch["packs"] = {**object_value(document["packs"], "packs"), **packs}
                for field, required in (
                    ("environment", {"id", "lastSnapshot", "lastRunAt"}),
                    ("accountProject", {"issuer", "id"}),
                ):
                    if field not in patch or patch[field] is None:
                        continue
                    nested = object_value(patch[field], field)
                    if set(nested) - required:
                        raise LibraryError("invalid", f"invalid {field} patch")
                    previous = document.get(field)
                    base = object_value(previous, field) if previous else {}
                    if field == "environment" and nested.get("id", base.get("id")) != base.get(
                        "id"
                    ):
                        base = {"lastSnapshot": None, "lastRunAt": None}
                    merged = {**base, **nested}
                    if not required <= merged.keys():
                        raise LibraryError("invalid", f"{field} requires {sorted(required)}")
                    if field == "environment":
                        identity = merged["id"]
                        run = (merged["lastSnapshot"], merged["lastRunAt"])
                        valid = (
                            isinstance(identity, str)
                            and bool(identity)
                            and (
                                run == (None, None)
                                or all(isinstance(value, str) and value for value in run)
                            )
                        )
                    else:
                        valid = all(isinstance(value, str) and value for value in nested.values())
                    if not valid:
                        raise LibraryError("invalid", f"invalid {field} patch")
                    patch[field] = merged
            document.update(patch)
            document["revision"] = revision + 1
            document["modified"] = datetime.now(UTC).isoformat()
            return {"project": self._save(document)}
