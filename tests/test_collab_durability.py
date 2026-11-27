"""Durability tests for collaborative document sessions."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest
from dinkster_collab import (
    ResyncRequiredError,
    SessionService,
    SessionStore,
    StaleBaseError,
    UnknownSessionError,
    WriteRefusedError,
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


class AtMost:
    """Refuses a ``value`` above ``limit`` unless the session held it before."""

    def __init__(self, limit: int) -> None:
        self.limit = limit

    def remembers(self, document_kind: str, path: tuple[str | int, ...], value: object) -> bool:
        return path == ("value",)

    def refusal(
        self,
        document_kind: str,
        before: object,
        after: object,
        held: Callable[[tuple[str | int, ...], object], bool],
    ) -> dict[str, object] | None:
        value = after["value"] if isinstance(after, dict) else None
        if isinstance(value, int) and value > self.limit and not held(("value",), value):
            return {"error": "too-big"}
        return None


def test_held_values_survive_checkpoint_and_restart(tmp_path: Path) -> None:
    """A value is accepted back after a restart when the session held it,
    whether in the document it opened with or in an op a checkpoint pruned."""
    path = tmp_path / "sessions.sqlite"

    def write(service: SessionService, session_id: str, value: int) -> None:
        revision = service.get(session_id).revision
        service.append(
            session_id,
            op_id=f"op-{value}-{revision}",
            actor_id="actor",
            base_revision=revision,
            patch=[{"op": "replace", "path": ["value"], "value": value}],
        )

    store = SessionStore(path)
    first = SessionService(store=store, write_check=AtMost(10))
    session_id = first.create(
        scope="local", document_id="d", snapshot={"value": 50}, title="Initial name"
    ).session_id
    write(first, session_id, 8)
    first.rename(session_id, "Renamed session")
    store.close()

    store = SessionStore(path)
    second = SessionService(store=store, write_check=AtMost(10))
    second.checkpoint(session_id, revision=1, document={"value": 8})
    store.close()

    store = SessionStore(path)
    third = SessionService(store=store, write_check=AtMost(5))
    try:
        assert third.get(session_id).title == "Renamed session"
        write(third, session_id, 8)
        write(third, session_id, 50)
        with pytest.raises(WriteRefusedError):
            write(third, session_id, 9)
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
