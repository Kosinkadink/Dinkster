"""Open document kinds keep the v1 wire and snapshot validation boundaries."""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from dinkster_collab import (
    InvalidSnapshotError,
    SessionService,
    SessionStore,
    SnapshotValidatorRegistry,
    normalize_document_kind,
)

from dinkster.serve import _collaboration_snapshot_validators
from tests.test_collab import REPLACE, make_client, op_body
from tests.test_image_document_adoption import _blank_document


@pytest.mark.parametrize(
    ("kind", "canonical", "wire"),
    [
        (None, "dinkster.workflow", "workflow"),
        ("workflow", "dinkster.workflow", "workflow"),
        ("dinkster.workflow", "dinkster.workflow", "workflow"),
        ("image", "dinkster.image", "image"),
        ("dinkster.image", "dinkster.image", "image"),
        ("extension.type", "extension.type", "extension.type"),
    ],
)
def test_kind_normalization_and_v1_descriptors(kind: str | None, canonical: str, wire: str) -> None:
    async def scenario() -> None:
        service = SessionService(snapshot_validator=_collaboration_snapshot_validators())
        client = await make_client(service)
        document = _blank_document() if canonical == "dinkster.image" else {"opaque": [1, None]}
        body: dict[str, object] = {"scope": "local", "documentId": "blank", "snapshot": document}
        if kind is not None:
            body["documentKind"] = kind
        try:
            response = await client.post("/api/sessions", json=body)
            assert response.status == 201
            created = await response.json()
            sid = created["sessionId"]
            assert service.get(sid).document_kind == canonical
            assert created["documentKind"] == wire
            assert created["protocolVersion"] == 1
            assert await (await client.get(f"/api/sessions/{sid}")).json() == created
            listed = await (await client.get("/api/sessions?scope=local")).json()
            assert listed["sessions"] == [created]
            snapshot = await (await client.get(f"/api/sessions/{sid}/snapshot")).json()
            assert snapshot == {"revision": 0, "document": document, "documentKind": wire}
            ws = await client.ws_connect(f"/api/sessions/{sid}/events")
            try:
                hello = await ws.receive_json()
                assert hello == {"type": "session", **created}
                # An old client can submit its unchanged v1 envelope to a canonical session.
                appended = await client.post(f"/api/sessions/{sid}/ops", json=op_body("a", 0))
                assert appended.status == 200
                event = await ws.receive_json()
                assert event == {"type": "op", **await appended.json()}
                assert event["protocolVersion"] == 1
                checkpoint = await client.put(
                    f"/api/sessions/{sid}/snapshot", json={"revision": 1, "document": document}
                )
                assert checkpoint.status == 200
                assert await checkpoint.json() == {"snapshotRevision": 1, "revision": 1}
                descriptor = await (await client.get(f"/api/sessions/{sid}")).json()
                assert descriptor == {**created, "snapshotRevision": 1, "revision": 1}
                snapshot = await (await client.get(f"/api/sessions/{sid}/snapshot")).json()
                assert snapshot == {"revision": 1, "document": document, "documentKind": wire}
            finally:
                await ws.close()
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", [None, 1, [], {}, "", "other", ".type", "ext.", "a..b", "a. b"])
def test_invalid_kinds_are_rejected(kind: object) -> None:
    with pytest.raises(ValueError):
        normalize_document_kind(kind)
    assert normalize_document_kind() == "dinkster.workflow"


@pytest.mark.parametrize("kind", ["image", "dinkster.image"])
@pytest.mark.parametrize("invalid", [{}, {**_blank_document(), "lineage": "different"}])
def test_image_validation_and_lineage_at_create_and_checkpoint(kind: str, invalid: object) -> None:
    async def scenario() -> None:
        service = SessionService(snapshot_validator=_collaboration_snapshot_validators())
        client = await make_client(service)
        body = {"scope": "local", "documentId": "blank", "documentKind": kind}
        try:
            rejected = await client.post("/api/sessions", json={**body, "snapshot": invalid})
            assert rejected.status == 400
            assert service.sessions("local") == []
            created = await client.post(
                "/api/sessions", json={**body, "snapshot": _blank_document()}
            )
            sid = (await created.json())["sessionId"]
            await client.post(f"/api/sessions/{sid}/ops", json=op_body("a", 0))
            before = service.get(sid)
            rejected = await client.put(
                f"/api/sessions/{sid}/snapshot", json={"revision": 1, "document": invalid}
            )
            assert rejected.status == 400
            assert (await rejected.json())["error"] == "document-invalid"
            assert service.get(sid) == before
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf"), {1: "x"}, (1,)])
def test_unknown_kind_retains_json_validation_and_ownership(invalid: object) -> None:
    service = SessionService(snapshot_validator=_collaboration_snapshot_validators())
    with pytest.raises(ValueError):
        service.create(
            scope="local", document_id="doc", document_kind="extension.type", snapshot=[invalid]
        )
    snapshot = {"custom": [1]}
    session = service.create(
        scope="local", document_id="doc", document_kind="extension.type", snapshot=snapshot
    )
    snapshot["custom"].append(2)
    assert service.get(session.session_id).snapshot == {"custom": [1]}
    service.append(session.session_id, op_id="a", actor_id="u", base_revision=0, patch=REPLACE)
    before = service.get(session.session_id)
    with pytest.raises(ValueError):
        service.checkpoint(session.session_id, revision=1, document={"custom": invalid})
    assert service.get(session.session_id) == before
    service.checkpoint(session.session_id, revision=1, document=snapshot)
    snapshot["custom"].append(3)
    assert service.get(session.session_id).snapshot == {"custom": [1, 2]}


def test_registry_dispatch_and_duplicate_alias_registration() -> None:
    validators = SnapshotValidatorRegistry()
    validators.register("image", lambda _id, _snapshot: "image refused")
    with pytest.raises(ValueError, match="already registered"):
        validators.register("dinkster.image", lambda _id, _snapshot: None)
    assert validators("dinkster.image", "doc", {}) == "image refused"
    validators.register("extension.type", lambda doc_id, value: None if value == doc_id else "bad")
    service = SessionService(snapshot_validator=validators)
    with pytest.raises(InvalidSnapshotError, match="bad"):
        service.create(scope="local", document_id="doc", document_kind="extension.type", snapshot=0)
    session = service.create(
        scope="local", document_id="doc", document_kind="extension.type", snapshot="doc"
    )
    service.append(session.session_id, op_id="a", actor_id="u", base_revision=0, patch=REPLACE)
    with pytest.raises(InvalidSnapshotError, match="bad"):
        service.checkpoint(session.session_id, revision=1, document=0)
    assert SnapshotValidatorRegistry()("image", "doc", {}) is None


def test_persisted_legacy_image_kind_gets_canonical_checkpoint_validation(tmp_path: Path) -> None:
    path = tmp_path / "sessions.sqlite"
    with closing(SessionStore(path)) as store:
        service = SessionService(store=store)
        session = service.create(
            scope="local", document_id="blank", document_kind="image", snapshot=_blank_document()
        )
        service.append(session.session_id, op_id="a", actor_id="u", base_revision=0, patch=REPLACE)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE sessions SET document_kind = 'image'")
    with closing(SessionStore(path)) as store:
        service = SessionService(
            store=store, snapshot_validator=_collaboration_snapshot_validators()
        )
        assert service.get(session.session_id).document_kind == "dinkster.image"
        with pytest.raises(InvalidSnapshotError, match="lineage"):
            service.checkpoint(
                session.session_id,
                revision=1,
                document={**_blank_document(), "lineage": "different"},
            )
