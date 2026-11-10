"""Durability tests for collaborative document sessions."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from dinkster_collab import (
    ActorPrincipalMismatchError,
    ResyncRequiredError,
    SessionService,
    SessionStore,
    StaleBaseError,
    UnknownSessionError,
)

PATCH = [{"op": "replace", "path": ["value"], "value": 1}]


def append_ops(service: SessionService, session_id: str, count: int) -> None:
    for revision in range(count):
        service.append(
            session_id,
            op_id=f"op-{revision}",
            actor_id="actor",
            base_revision=revision,
            patch=PATCH,
        )


def test_sessions_survive_restart(tmp_path: Path) -> None:
    path = tmp_path / "sessions.sqlite"
    first_store = SessionStore(path)
    first = SessionService(store=first_store)
    surviving = first.create(
        scope="team",
        document_id="document",
        document_kind="image",
        snapshot={"value": 0},
    )
    closed = first.create(scope="team", document_id="closed", snapshot=None)
    append_ops(first, surviving.session_id, 4)
    append_ops(first, closed.session_id, 2)
    first.checkpoint(surviving.session_id, revision=2, document={"value": 2})
    expected = first.get(surviving.session_id)
    expected_ops = first.ops_after(surviving.session_id, 2)
    first.close(closed.session_id)
    first_store.close()

    second_store = SessionStore(path)
    second = SessionService(store=second_store)
    try:
        actual = second.get(surviving.session_id)
        assert (
            actual.session_id,
            actual.scope,
            actual.document_id,
            actual.document_kind,
            actual.snapshot,
            actual.snapshot_revision,
            actual.created_at,
        ) == (
            expected.session_id,
            expected.scope,
            expected.document_id,
            expected.document_kind,
            expected.snapshot,
            expected.snapshot_revision,
            expected.created_at,
        )
        assert actual.revision == expected.revision
        assert second.ops_after(surviving.session_id, 2) == expected_ops
        replayed_op, replayed = second.append(
            surviving.session_id,
            op_id=expected_ops[0].op_id,
            actor_id=expected_ops[0].actor_id,
            base_revision=99,
            patch=PATCH,
        )
        assert replayed
        assert replayed_op == expected_ops[0]
        with pytest.raises(UnknownSessionError):
            second.get(closed.session_id)
    finally:
        second_store.close()


def test_existing_database_adds_workflow_document_kind(tmp_path: Path) -> None:
    path = tmp_path / "sessions.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                scope TEXT NOT NULL,
                document_id TEXT NOT NULL,
                snapshot TEXT NOT NULL,
                snapshot_revision INTEGER NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?)",
            ("legacy", "local", "document", "null", 0, 1.0),
        )

    store = SessionStore(path)
    try:
        session = SessionService(store=store).get("legacy")
        assert session.document_kind == "dinkster.workflow"
        with sqlite3.connect(path) as connection:
            column = next(
                row
                for row in connection.execute("PRAGMA table_info(sessions)")
                if row[1] == "document_kind"
            )
        assert column[3:] == (1, "'workflow'", 0)
    finally:
        store.close()


def test_checkpoint_pruning_is_durable(tmp_path: Path) -> None:
    path = tmp_path / "sessions.sqlite"
    store = SessionStore(path)
    service = SessionService(store=store)
    session = service.create(scope="local", document_id="document", snapshot={"value": 0})
    append_ops(service, session.session_id, 4)
    service.checkpoint(session.session_id, revision=2, document={"value": 2})

    with sqlite3.connect(path) as conn:
        revisions = [
            row[0]
            for row in conn.execute(
                "SELECT revision FROM ops WHERE session_id = ? ORDER BY revision",
                (session.session_id,),
            )
        ]
    assert revisions == [3, 4]
    retained = service.ops_after(session.session_id, 2)
    store.close()

    reloaded_store = SessionStore(path)
    reloaded = SessionService(store=reloaded_store)
    try:
        assert reloaded.ops_after(session.session_id, 2) == retained
        with pytest.raises(ResyncRequiredError):
            reloaded.ops_after(session.session_id, 1)
    finally:
        reloaded_store.close()


def test_refused_stale_append_does_not_reach_store(tmp_path: Path) -> None:
    path = tmp_path / "sessions.sqlite"
    store = SessionStore(path)
    service = SessionService(store=store)
    session = service.create(scope="local", document_id="document", snapshot=None)
    append_ops(service, session.session_id, 1)
    with sqlite3.connect(path) as conn:
        before = conn.execute("SELECT count(*) FROM ops").fetchone()[0]

    with pytest.raises(StaleBaseError):
        service.append(
            session.session_id,
            op_id="refused",
            actor_id="actor",
            base_revision=0,
            patch=PATCH,
        )

    with sqlite3.connect(path) as conn:
        after = conn.execute("SELECT count(*) FROM ops").fetchone()[0]
    assert after == before
    store.close()


def test_storeless_service_creates_no_file(tmp_path: Path) -> None:
    service = SessionService()
    session = service.create(scope="local", document_id="document", snapshot=None)
    op, replayed = service.append(
        session.session_id,
        op_id="op",
        actor_id="actor",
        base_revision=0,
        patch=PATCH,
    )
    assert not replayed
    assert service.get(session.session_id).revision == op.revision
    assert list(tmp_path.iterdir()) == []


def test_actor_bindings_survive_restart_and_are_deleted_on_close(tmp_path: Path) -> None:
    path = tmp_path / "sessions.sqlite"
    first_store = SessionStore(path)
    first = SessionService(store=first_store)
    session = first.create(scope="team", document_id="document", snapshot=None)
    first.append(
        session.session_id,
        op_id="first",
        actor_id="actor",
        base_revision=0,
        patch=PATCH,
        principal_id="alice",
    )
    first_store.close()

    second_store = SessionStore(path)
    second = SessionService(store=second_store)
    with pytest.raises(ActorPrincipalMismatchError):
        second.append(
            session.session_id,
            op_id="spoofed",
            actor_id="actor",
            base_revision=1,
            patch=PATCH,
            principal_id="bob",
        )
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM ops").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM actors").fetchone()[0] == 1
    second.close(session.session_id)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM actors").fetchone()[0] == 0
    second_store.close()


def test_presence_binding_is_durable_without_an_op(tmp_path: Path) -> None:
    path = tmp_path / "sessions.sqlite"
    store = SessionStore(path)
    service = SessionService(store=store)
    session = service.create(scope="team", document_id="document", snapshot=None)
    service.bind_actor(session.session_id, "actor", "alice")
    store.close()

    reloaded_store = SessionStore(path)
    reloaded = SessionService(store=reloaded_store)
    try:
        with pytest.raises(ActorPrincipalMismatchError):
            reloaded.bind_actor(session.session_id, "actor", "bob")
    finally:
        reloaded_store.close()


def test_acl_survives_restart_and_is_deleted_on_close(tmp_path: Path) -> None:
    path = tmp_path / "sessions.sqlite"
    first_store = SessionStore(path)
    first = SessionService(store=first_store)
    session = first.create(
        scope="team", document_id="document", snapshot=None, principal_id="alice"
    )
    first.replace_acl(
        session.session_id,
        principal_id="alice",
        default_role="viewer",
        entries={"alice": "owner", "bob": "editor", "eve": "banned"},
    )
    first_store.close()

    second_store = SessionStore(path)
    second = SessionService(store=second_store)
    actual = second.get(session.session_id)
    assert actual.default_role == "viewer"
    assert actual.acl == {"alice": "owner", "bob": "editor", "eve": "banned"}
    second.close(session.session_id, principal_id="alice")
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM session_acl").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM session_policy").fetchone()[0] == 0
    second_store.close()


def test_storeless_acl_has_the_same_role_semantics() -> None:
    service = SessionService()
    session = service.create(
        scope="team", document_id="document", snapshot=None, principal_id="alice"
    )
    service.replace_acl(
        session.session_id,
        principal_id="alice",
        default_role="viewer",
        entries={"alice": "owner", "bob": "editor"},
    )
    assert service.role_for(session.session_id, "alice") == "owner"
    assert service.role_for(session.session_id, "bob") == "editor"
    assert service.role_for(session.session_id, "unlisted") == "viewer"
