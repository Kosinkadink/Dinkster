"""Chunked, revision-checked publication of resolver snapshots and their leads."""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
import sqlite3
import uuid
from typing import TYPE_CHECKING, cast

from ._manifests import asset_digest
from ._provenance import merge_provenance, provenance_document, replace_layer
from ._records import object_value
from .client import LibraryError

if TYPE_CHECKING:
    from ._records import Records

CHUNK_BYTES = 1024 * 1024


def _identifier(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{32}", value) is None:
        raise LibraryError("invalid", "resolver id must be 32 lowercase hexadecimal characters")
    return value


def _revision(value: object) -> int:
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        raise LibraryError("invalid", "resolver revision must be a nonnegative SQLite integer")
    return value


class Resolvers:
    def __init__(self, records: Records) -> None:
        self.records = records
        # Incomplete uploads belong only to this helper and disappear with it.
        self.stage: tuple[str, str, int, bytearray] | None = None

    def dispatch(self, operation: str, arguments: dict[str, object]) -> dict[str, object]:
        records = self.records
        if any(
            (records.root / name).exists()
            for name in ("library.sqlite", "provenance.json", "resolver-indexes.json")
        ):
            raise LibraryError("migration", "Finish the complete library migration first")
        allowed: dict[str, set[str]] = {
            "sources.resolvers.begin": {"id", "revision"},
            "sources.resolvers.append": {"token", "offset", "data"},
            "sources.resolvers.commit": {"token", "officialBootstrap"},
            "sources.resolvers.bootstrap.get": set(),
            "sources.resolvers.bootstrap.put": {"configuration"},
            "sources.resolvers.cancel": {"token"},
            "sources.resolvers.get": {"id"},
            "sources.resolvers.list": set(),
            "sources.resolvers.read": {"id", "revision", "offset", "limit"},
            "sources.resolvers.remove": {"id", "revision"},
        }
        if operation not in allowed:
            raise LibraryError("operation", f"unknown library operation: {operation}")
        if set(arguments) - allowed[operation]:
            raise LibraryError("invalid", "unknown resolver field")
        if operation == "sources.resolvers.bootstrap.get":
            return {"configuration": self._bootstrap()}
        if operation == "sources.resolvers.bootstrap.put":
            configuration = self._configuration(arguments.get("configuration"))
            with records.db:
                records.db.execute("BEGIN IMMEDIATE")
                row = records.db.execute(
                    "SELECT payload FROM resolver_subscriptions WHERE id=?",
                    (configuration["subscriptionId"],),
                ).fetchone()
                if row is not None and row[0] is not None:
                    envelope = object_value(json.loads(row[0]), "stored resolver snapshot")
                    self._check_identity(
                        configuration, object_value(envelope["subscription"], "subscription")
                    )
                created = self._put_bootstrap(configuration)
            return {"configuration": self._bootstrap(), "created": created}
        if operation == "sources.resolvers.begin":
            identifier = _identifier(arguments.get("id"))
            revision = _revision(arguments.get("revision"))
            if self.stage is not None:
                raise LibraryError("busy", "cancel or commit this helper's current resolver upload")
            token = uuid.uuid4().hex
            self.stage = token, identifier, revision, bytearray()
            return {"token": token, "offset": 0}
        if operation in (
            "sources.resolvers.append",
            "sources.resolvers.commit",
            "sources.resolvers.cancel",
        ):
            if self.stage is None or arguments.get("token") != self.stage[0]:
                raise LibraryError("stale", "resolver upload belongs to another or exited helper")
            token, identifier, revision, data = self.stage
            if operation == "sources.resolvers.cancel":
                self.stage = None
                return {}
            if operation == "sources.resolvers.append":
                offset = arguments.get("offset")
                encoded = arguments.get("data")
                if type(offset) is not int or offset != len(data) or not isinstance(encoded, str):
                    raise LibraryError("invalid", "resolver chunk must start at its current offset")
                try:
                    chunk = base64.b64decode(encoded, validate=True)
                except (ValueError, binascii.Error) as error:
                    raise LibraryError("invalid", "resolver chunk must be base64") from error
                if not 1 <= len(chunk) <= CHUNK_BYTES:
                    raise LibraryError("invalid", "resolver chunk must contain 1 byte to 1 MiB")
                data.extend(chunk)
                return {"token": token, "offset": len(data)}
            bootstrap = (
                self._configuration(arguments["officialBootstrap"])
                if "officialBootstrap" in arguments
                else None
            )
            result = self._commit(identifier, revision, data, bootstrap)
            self.stage = None
            return result
        if operation == "sources.resolvers.list":
            rows = records.db.execute(
                "SELECT id,source,revision,length(payload) FROM resolver_subscriptions "
                "WHERE payload IS NOT NULL ORDER BY id"
            ).fetchall()
            return {"items": [self._header(row) for row in rows]}
        identifier = _identifier(arguments.get("id"))
        if operation == "sources.resolvers.remove":
            revision = _revision(arguments.get("revision"))
            with records.db:
                records.db.execute("BEGIN IMMEDIATE")
                row = records.db.execute(
                    "SELECT revision,payload IS NOT NULL FROM resolver_subscriptions WHERE id=?",
                    (identifier,),
                ).fetchone()
                if row is None or row[0] != revision or not row[1]:
                    raise LibraryError("stale", "resolver changed; read it again before removing")
                replace_layer(records, "resolver-index:" + identifier, {})
                records.db.execute(
                    "UPDATE resolver_subscriptions SET revision=revision+1,payload=NULL WHERE id=?",
                    (identifier,),
                )
                self._change(identifier, revision + 1)
            return {"revision": revision + 1}
        if operation == "sources.resolvers.get":
            row = records.db.execute(
                "SELECT id,source,revision,length(payload) FROM resolver_subscriptions WHERE id=?",
                (identifier,),
            ).fetchone()
            return {"subscription": self._header(row) if row else None}
        revision = _revision(arguments.get("revision"))
        offset, limit = arguments.get("offset", 0), arguments.get("limit", CHUNK_BYTES)
        if (
            type(offset) is not int
            or not 0 <= offset < 2**63 - 1
            or type(limit) is not int
            or not 1 <= limit <= CHUNK_BYTES
        ):
            raise LibraryError("invalid", "invalid resolver read offset or chunk size")
        # The revision predicate and slice are one read, so chunks cannot mix publications.
        row = records.db.execute(
            "SELECT substr(payload,?,?),length(payload) FROM resolver_subscriptions "
            "WHERE id=? AND revision=? AND payload IS NOT NULL",
            (offset + 1, limit, identifier, revision),
        ).fetchone()
        if row is None:
            raise LibraryError("stale", "resolver snapshot changed; start reading again")
        if offset > row[1]:
            raise LibraryError("invalid", "resolver read starts beyond the snapshot")
        return {
            "data": base64.b64encode(row[0]).decode(),
            "offset": offset + len(row[0]),
            "complete": offset + len(row[0]) == row[1],
        }

    @staticmethod
    def _header(row: tuple[object, ...]) -> dict[str, object]:
        return {
            "id": row[0],
            "source": row[1],
            "revision": row[2],
            "bytes": row[3],
            "deleted": row[3] is None,
        }

    def _change(self, identifier: str, revision: int) -> None:
        self.records.db.execute(
            "INSERT INTO changes(kind,id,revision) VALUES ('resolver',?,?)", (identifier, revision)
        )
        self.records.db.execute(
            "INSERT INTO changes(kind,id,revision) VALUES ('provenance',?,1)",
            ("resolver-index:" + identifier,),
        )

    def _bootstrap(self) -> dict[str, object] | None:
        row = self.records.db.execute(
            "SELECT document FROM resolver_configuration WHERE id=1"
        ).fetchone()
        return object_value(json.loads(row[0]), "official bootstrap") if row else None

    @staticmethod
    def _configuration(value: object) -> dict[str, object]:
        configuration = object_value(value, "official resolver bootstrap")
        _identifier(configuration.get("subscriptionId"))
        for field, bound in (("source", 4096), ("providerId", 512)):
            item = configuration.get(field)
            if (
                not isinstance(item, str)
                or not item
                or item != item.strip()
                or len(item) > bound
                or any(ord(char) < 32 or ord(char) == 127 for char in item)
            ):
                raise LibraryError("invalid", f"official resolver {field} must be a bounded string")
        if len(json.dumps(configuration, allow_nan=False).encode("utf-8")) > 128 * 1024:
            raise LibraryError("invalid", "official resolver bootstrap exceeds 128 KiB")
        return configuration

    @staticmethod
    def _check_identity(configuration: dict[str, object], document: dict[str, object]) -> None:
        if document.get("id") == configuration["subscriptionId"] and (
            document.get("source") != configuration["source"]
            or object_value(document.get("index"), "resolver index").get("name")
            != configuration["providerId"]
        ):
            raise LibraryError("conflict", "official resolver identity differs from its bootstrap")

    def _put_bootstrap(self, configuration: dict[str, object]) -> bool:
        previous = self._bootstrap()
        if previous is not None:
            # An older caller may omit extensions, but cannot reset the saved choice.
            if any(
                key not in previous
                or json.dumps(previous[key], sort_keys=True, allow_nan=False)
                != json.dumps(value, sort_keys=True, allow_nan=False)
                for key, value in configuration.items()
            ):
                raise LibraryError("conflict", "official resolver bootstrap is already recorded")
            return False
        self.records.db.execute(
            "INSERT INTO resolver_configuration VALUES (1,?)",
            (json.dumps(configuration, allow_nan=False),),
        )
        self.records.db.execute(
            "INSERT INTO changes(kind,id,revision) VALUES ('resolver-configuration','official',1)"
        )
        return True

    def _commit(
        self,
        identifier: str,
        revision: int,
        data: bytearray,
        bootstrap: dict[str, object] | None,
    ) -> dict[str, object]:
        def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate resolver JSON field")
                result[key] = value
            return result

        def reject_constant(value: str) -> object:
            raise ValueError(f"invalid resolver JSON number {value}")

        def finite_number(value: str) -> float:
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("resolver JSON number is not finite")
            return number

        try:
            envelope = object_value(
                json.loads(
                    data.decode("utf-8"),
                    object_pairs_hook=object_pairs,
                    parse_constant=reject_constant,
                    parse_float=finite_number,
                ),
                "resolver snapshot",
            )
        except (ValueError, UnicodeError, RecursionError) as error:
            raise LibraryError(
                "invalid", "resolver snapshot must be a UTF-8 JSON object"
            ) from error
        document = object_value(envelope.get("subscription"), "resolver subscription")
        if document.get("id") != identifier:
            raise LibraryError("invalid", "resolver snapshot id differs from upload id")
        source = document.get("source")
        if not isinstance(source, str) or not source.strip() or len(source) > 4096:
            raise LibraryError("invalid", "resolver source must be a nonempty bounded string")
        if document.get("sourceType") not in ("local", "url"):
            raise LibraryError("invalid", "resolver sourceType must be local or url")
        for field in ("etag", "error"):
            if not isinstance(document.get(field, ""), str):
                raise LibraryError("invalid", f"resolver {field} must be a string")
        if type(document.get("completeSnapshot", False)) is not bool:
            raise LibraryError("invalid", "resolver completeSnapshot must be a boolean")
        times: list[float] = []
        for field in ("checkedAt", "refreshedAt"):
            value = document.get(field, 0.0)
            try:
                if type(value) not in (int, float):
                    raise ValueError
                timestamp = float(cast(float, value))
                if not math.isfinite(timestamp) or timestamp < 0:
                    raise ValueError
            except (ValueError, OverflowError) as error:
                raise LibraryError(
                    "invalid", f"resolver {field} must be finite and nonnegative"
                ) from error
            times.append(timestamp)
        if times[1] > times[0]:
            raise LibraryError("invalid", "resolver refreshedAt cannot follow checkedAt")
        index = object_value(document.get("index"), "resolver index")
        if type(index.get("dinksterResolver")) is not int or index["dinksterResolver"] != 1:
            raise LibraryError("invalid", "resolver index version must be 1")
        entries = index.get("entries")
        if not isinstance(entries, list) or len(cast("list[object]", entries)) > 100_000:
            raise LibraryError("invalid", "resolver index requires at most 100,000 entries")
        raw_records = envelope.get("records")
        if not isinstance(raw_records, list):
            raise LibraryError("invalid", "resolver snapshot requires its provenance records")
        incoming: dict[str, dict[str, object]] = {}
        for raw in cast("list[object]", raw_records):
            record = provenance_document(raw)
            digest = cast(str, record["digest"])
            incoming[digest] = (
                merge_provenance(incoming[digest], record) if digest in incoming else record
            )
        # The engine validates resolver semantics and supplies its regional leads.
        entry_digests = {
            asset_digest(object_value(entry, "resolver entry").get("digest"))
            for entry in cast("list[object]", entries)
        }
        if entry_digests != set(incoming):
            raise LibraryError("invalid", "resolver entries and provenance digests differ")
        records = self.records
        try:
            with records.db:
                records.db.execute("BEGIN IMMEDIATE")
                previous = records.db.execute(
                    "SELECT source,revision,payload FROM resolver_subscriptions WHERE id=?",
                    (identifier,),
                ).fetchone()
                if (previous[1] if previous else 0) != revision:
                    raise LibraryError("stale", "resolver changed during upload; read it again")
                if previous is not None and previous[0] != source:
                    raise LibraryError("conflict", "resolver source identity is immutable")
                if bootstrap is not None:
                    if bootstrap["subscriptionId"] != identifier:
                        raise LibraryError(
                            "invalid", "bootstrap must name the uploaded subscription"
                        )
                    self._put_bootstrap(bootstrap)
                configuration = self._bootstrap()
                if configuration is not None:
                    self._check_identity(configuration, document)
                if previous is not None and previous[2] is not None:
                    old = object_value(json.loads(previous[2]), "stored resolver snapshot")
                    old_document = object_value(old["subscription"], "stored subscription")
                    known = {
                        "id",
                        "source",
                        "sourceType",
                        "index",
                        "etag",
                        "error",
                        "checkedAt",
                        "refreshedAt",
                        "completeSnapshot",
                    }
                    unseen = {key: value for key, value in old_document.items() if key not in known}
                    envelope = {**old, **envelope, "subscription": {**unseen, **document}}
                payload = json.dumps(envelope, allow_nan=False).encode("utf-8")
                replace_layer(records, "resolver-index:" + identifier, incoming)
                records.db.execute(
                    "INSERT INTO resolver_subscriptions VALUES (?,?,?,?) "
                    "ON CONFLICT(id) DO UPDATE SET "
                    "revision=excluded.revision,payload=excluded.payload",
                    (identifier, source, revision + 1, payload),
                )
                if (
                    records.db.execute(
                        "SELECT count(*) FROM resolver_subscriptions WHERE payload IS NOT NULL"
                    ).fetchone()[0]
                    > 64
                ):
                    raise LibraryError("limit", "at most 64 resolver subscriptions may be active")
                self._change(identifier, revision + 1)
        except sqlite3.IntegrityError as error:
            raise LibraryError("conflict", "resolver source is already subscribed") from error
        return {"revision": revision + 1, "bytes": len(payload)}
