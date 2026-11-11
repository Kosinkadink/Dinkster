"""Atomic verified feeds, independent trust, and exact-notice decisions."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

from .documents import (
    Document,
    SecurityError,
    Trust,
    signed_payload,
    verify_document,
    verify_trust,
)


class SecurityStore:
    """Persist one install's security checkpoints without owning installed software."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS registries (
                    registry TEXT PRIMARY KEY,
                    trust BLOB NOT NULL,
                    envelope BLOB,
                    verified_at TEXT
                );
                CREATE TABLE IF NOT EXISTS decisions (
                    registry TEXT NOT NULL,
                    notice TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    approved INTEGER NOT NULL CHECK (approved IN (0, 1)),
                    PRIMARY KEY (registry, notice, revision)
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def provision_trust(self, data: bytes, *, recovery: bool = False) -> None:
        """Install keys authenticated by the caller's independent provisioning channel.

        Recovery replaces only trust. Cached blocks, checkpoints and decisions survive.
        Network responses must use rotate_trust, never this operation.
        """
        trust = Trust.model_validate_json(data)
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            old = connection.execute(
                "SELECT trust FROM registries WHERE registry = ?", (trust.registry,)
            ).fetchone()
            if old is not None and not recovery:
                raise SecurityError(
                    "registry trust already provisioned; use authenticated rotation"
                )
            connection.execute(
                "INSERT INTO registries (registry, trust) VALUES (?, ?) "
                "ON CONFLICT (registry) DO UPDATE SET trust = excluded.trust",
                (trust.registry, data),
            )

    def rotate_trust(self, registry: str, data: bytes) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            trust, _, _ = self._registry(connection, registry)
            replacement = verify_trust(data, trust)
            connection.execute(
                "UPDATE registries SET trust = ? WHERE registry = ?",
                (replacement.model_dump_json(by_alias=True).encode(), registry),
            )

    @staticmethod
    def _registry(
        connection: sqlite3.Connection, registry: str
    ) -> tuple[Trust, bytes | None, str | None]:
        row = connection.execute(
            "SELECT trust, envelope, verified_at FROM registries WHERE registry = ?",
            (registry,),
        ).fetchone()
        if row is None:
            raise SecurityError("registry has no independently provisioned security trust")
        return Trust.model_validate_json(row[0]), row[1], row[2]

    def import_feed(self, registry: str, data: bytes, *, now: datetime | None = None) -> Document:
        """Authenticate network or removable-media bytes before changing any checkpoint."""
        instant = datetime.now(UTC) if now is None else now
        if instant.utcoffset() is None:
            raise ValueError("verification time must include a timezone")
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            trust, previous, _ = self._registry(connection, registry)
            document = verify_document(data, trust, previous=previous)
            connection.execute(
                "UPDATE registries SET envelope = ?, verified_at = ? WHERE registry = ?",
                (data, instant.astimezone(UTC).isoformat(), registry),
            )
        return document

    def cached(self, registry: str) -> tuple[Document | None, datetime | None]:
        """Return the accepted checkpoint even when the network or current key is gone."""
        with closing(self._connect()) as connection:
            _, envelope, verified_at = self._registry(connection, registry)
        document = (
            None if envelope is None else Document.model_validate_json(signed_payload(envelope))
        )
        return document, None if verified_at is None else datetime.fromisoformat(verified_at)

    def registries(self) -> tuple[str, ...]:
        with closing(self._connect()) as connection:
            return tuple(row[0] for row in connection.execute("SELECT registry FROM registries"))

    def answer(self, registry: str, notice: UUID, revision: int, *, approved: bool) -> None:
        if type(approved) is not bool:
            raise ValueError("approval must be an explicit boolean")
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            _, envelope, _ = self._registry(connection, registry)
            if envelope is None:
                raise SecurityError("no verified notice to answer")
            document = Document.model_validate_json(signed_payload(envelope))
            current = next((item for item in document.notices if item.id == notice), None)
            if current is None or current.revision != revision:
                raise SecurityError("notice changed; review its current details before answering")
            if current.state != "vulnerable" or current.status != "active":
                raise SecurityError("only active vulnerable notices accept continuation decisions")
            connection.execute(
                "INSERT INTO decisions VALUES (?, ?, ?, ?) "
                "ON CONFLICT (registry, notice, revision) "
                "DO UPDATE SET approved = excluded.approved",
                (registry, str(notice), revision, int(approved)),
            )

    def decision(
        self, registry: str, notice: UUID, revision: int
    ) -> Literal["pending", "approved", "denied"]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT approved FROM decisions WHERE registry = ? AND notice = ? AND revision = ?",
                (registry, str(notice), revision),
            ).fetchone()
        return "pending" if row is None else "approved" if row[0] else "denied"
