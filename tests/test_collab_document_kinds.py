"""Open collaboration document kinds and built-in snapshot validators."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import dinkster_collab as collab
import pytest
from dinkster_collab import (
    InvalidSnapshotError,
    SessionService,
    SessionStore,
    SnapshotValidatorRegistry,
    document_kind_wire,
    normalize_document_kind,
)
from dinkster_video.document import make as make_video_document

from dinkster.serve import _collaboration_snapshot_validators
from tests.test_collab import make_client
from tests.test_image_document_adoption import _blank_document


@pytest.mark.parametrize(
    ("kind", "canonical", "wire"),
    [
        (None, "dinkster.workflow", "workflow"),
        ("workflow", "dinkster.workflow", "workflow"),
        ("dinkster.workflow", "dinkster.workflow", "workflow"),
        ("image", "dinkster.image", "image"),
        ("dinkster.image", "dinkster.image", "image"),
        ("video", "dinkster.video", "video"),
        ("dinkster.video", "dinkster.video", "video"),
        ("extension.type", "extension.type", "extension.type"),
    ],
)
def test_kind_normalization_and_v1_descriptors(kind: str | None, canonical: str, wire: str) -> None:
    async def scenario() -> None:
        service = SessionService(snapshot_validator=_collaboration_snapshot_validators(collab))
        client = await make_client(service)
        if canonical == "dinkster.image":
            document: object = _blank_document()
        elif canonical == "dinkster.video":
            document = make_video_document()
        else:
            document = {"opaque": [1, None]}
        body: dict[str, object] = {
            "scope": "local",
            "documentId": "blank",
            "snapshot": document,
        }
        if kind is not None:
            body["documentKind"] = kind
        try:
            response = await client.post("/api/sessions", json=body)
            assert response.status == 201
            created = await response.json()
            session_id = created["sessionId"]
            assert service.get(session_id).document_kind == canonical
            assert created["documentKind"] == wire
            assert await (await client.get(f"/api/sessions/{session_id}")).json() == created
            snapshot = await (await client.get(f"/api/sessions/{session_id}/snapshot")).json()
            assert snapshot == {
                "revision": 0,
                "document": document,
                "documentKind": wire,
            }
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", [None, 1, [], {}, "", "other", ".type", "ext.", "a..b", "a. b"])
def test_invalid_kinds_are_rejected(kind: object) -> None:
    with pytest.raises(ValueError):
        normalize_document_kind(kind)
    assert normalize_document_kind() == "dinkster.workflow"


def test_registry_dispatches_canonical_aliases_and_refuses_duplicates() -> None:
    validators = SnapshotValidatorRegistry()
    validators.register("image", lambda _document_id, _snapshot: "image refused")
    with pytest.raises(ValueError, match="already registered"):
        validators.register("dinkster.image", lambda _document_id, _snapshot: None)
    assert validators("dinkster.image", "doc", {}) == "image refused"
    assert document_kind_wire("dinkster.video") == "video"
    assert validators("extension.type", "doc", {}) is None


def test_image_and_video_validators_run_at_create_and_checkpoint() -> None:
    validators = _collaboration_snapshot_validators(collab)
    service = SessionService(snapshot_validator=validators)
    with pytest.raises(InvalidSnapshotError, match="ImageDocument"):
        service.create(scope="local", document_id="image", document_kind="image", snapshot={})
    with pytest.raises(InvalidSnapshotError, match="video document"):
        service.create(scope="local", document_id="video", document_kind="video", snapshot={})

    image = service.create(
        scope="local",
        document_id="blank",
        document_kind="image",
        snapshot=_blank_document(),
    )
    video = service.create(
        scope="local",
        document_id="video",
        document_kind="video",
        snapshot=make_video_document(),
    )
    assert image.document_kind == "dinkster.image"
    assert video.document_kind == "dinkster.video"


def test_checkpoint_rejects_invalid_video_snapshot() -> None:
    validators = _collaboration_snapshot_validators(collab)
    service = SessionService(snapshot_validator=validators)
    session = service.create(
        scope="local",
        document_id="video",
        document_kind="video",
        snapshot=make_video_document(),
    )
    service.append(
        session.session_id,
        op_id="op-1",
        actor_id="actor",
        base_revision=0,
        patch=[{"op": "replace", "path": ["name"], "value": "Edited"}],
    )
    with pytest.raises(InvalidSnapshotError, match="video document"):
        service.checkpoint(session.session_id, revision=1, document={"timeline": "bogus"})
    checkpointed = service.checkpoint(
        session.session_id, revision=1, document=make_video_document(name="Edited")
    )
    assert checkpointed.snapshot_revision == 1


def test_sqlite_restore_rewrites_legacy_and_namespaced_kinds(tmp_path: Path) -> None:
    path = tmp_path / "sessions.sqlite"
    store = SessionStore(path)
    service = SessionService(store=store)
    image = service.create(
        scope="local", document_id="blank", document_kind="image", snapshot=_blank_document()
    )
    extension = service.create(
        scope="local",
        document_id="ext-doc",
        document_kind="extension.type",
        snapshot={"opaque": True},
    )
    store.close()

    # Simulate a database written by the release that stored the legacy
    # spelling: rewrite the image row's kind back to "image".
    connection = sqlite3.connect(path)
    with connection:
        connection.execute(
            "UPDATE sessions SET document_kind = 'image' WHERE session_id = ?",
            (image.session_id,),
        )
    connection.close()

    second_store = SessionStore(path)
    restored = SessionService(store=second_store)
    try:
        assert restored.get(image.session_id).document_kind == "dinkster.image"
        assert restored.get(extension.session_id).document_kind == "extension.type"
        assert document_kind_wire(restored.get(image.session_id).document_kind) == "image"
        assert document_kind_wire(restored.get(extension.session_id).document_kind) == (
            "extension.type"
        )
    finally:
        second_store.close()


def test_malformed_stored_kind_fails_startup(tmp_path: Path) -> None:
    path = tmp_path / "sessions.sqlite"
    store = SessionStore(path)
    service = SessionService(store=store)
    session = service.create(scope="local", document_id="doc", snapshot={"value": 0})
    store.close()

    connection = sqlite3.connect(path)
    with connection:
        connection.execute(
            "UPDATE sessions SET document_kind = 'not namespaced' WHERE session_id = ?",
            (session.session_id,),
        )
    connection.close()

    # A kind that cannot be normalized must fail the restore loudly rather
    # than silently rewriting or dropping the stored session.
    with pytest.raises(ValueError, match="namespaced"):
        SessionService(store=SessionStore(path))
