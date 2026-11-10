"""dinkster-collab: server-ordered document sessions (platform plan v1).

The service tests pin the ordering/idempotency/retention model; the
HTTP tests pin the wire contract - envelope shape, stale-base rebase
loop, resync, protocol version refusal - and run against a BARE aiohttp
app (no engine, no dinkster-server) because package independence is part
of the design: the surface must be extractable to its own process.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from aiohttp import WSMsgType, WSServerHandshakeError, web
from aiohttp.test_utils import TestClient, TestServer
from dinkster_collab import (
    PROTOCOL_VERSION,
    ActorPrincipalMismatchError,
    InvalidSnapshotError,
    ResyncRequiredError,
    SessionOp,
    SessionRoleError,
    SessionService,
    SessionStore,
    SnapshotRequiredError,
    StaleBaseError,
    UnknownSessionError,
    add_session_routes,
    validate_patch,
)

REPLACE = [{"op": "replace", "path": ["meta", "title"], "value": "new"}]


# -- patch shape validation ----------------------------------------------------


def test_validate_patch_accepts_segment_path_shapes() -> None:
    assert validate_patch(REPLACE) is None
    assert (
        validate_patch(
            [
                {"op": "add", "path": ["a"], "value": 1},
                {"op": "add", "path": ["graphs", "g0", "nodes", 3], "value": {}},
                {"op": "add", "path": [], "value": {"whole": "doc"}},
                {"op": "remove", "path": ["b", 0]},
                {"op": "replace", "path": ["c"], "value": None},
            ]
        )
        is None
    )


def test_validate_patch_rejects_malformed() -> None:
    assert validate_patch({}) is not None  # not a list
    assert validate_patch([]) is not None  # empty
    assert validate_patch(["x"]) is not None  # entry not an object
    assert validate_patch([{"op": "explode", "path": ["x"]}]) is not None
    # RFC 6902 verbs the frontend never produces are rejected outright
    assert validate_patch([{"op": "move", "path": ["c"], "from": "/d"}]) is not None
    assert validate_patch([{"op": "copy", "path": ["e"], "from": "/f"}]) is not None
    assert validate_patch([{"op": "test", "path": ["g"], "value": 1}]) is not None
    # path must be a segment array, not a JSON Pointer string or scalar
    assert validate_patch([{"op": "add", "path": "/x", "value": 1}]) is not None
    assert validate_patch([{"op": "add", "path": 3, "value": 1}]) is not None
    # segments must be string|int - no bools, floats, or nested values
    assert validate_patch([{"op": "remove", "path": [True]}]) is not None
    assert validate_patch([{"op": "remove", "path": [1.5]}]) is not None
    assert validate_patch([{"op": "remove", "path": [["x"]]}]) is not None
    assert validate_patch([{"op": "add", "path": ["x"]}]) is not None  # no value


# -- service model -------------------------------------------------------------


def test_service_orders_appends_and_replays_idempotently() -> None:
    service = SessionService()
    session = service.create(scope="local", document_id="doc1", snapshot={"v": 0})
    assert session.revision == 0
    op1, replayed = service.append(
        session.session_id, op_id="a", actor_id="u1", base_revision=0, patch=REPLACE
    )
    assert (op1.revision, op1.base_revision, replayed) == (1, 0, False)
    # Idempotent resubmission: same opId answers the recorded envelope.
    again, replayed = service.append(
        session.session_id, op_id="a", actor_id="u1", base_revision=99, patch=REPLACE
    )
    assert again == op1 and replayed
    # Stale base: the client must rebase; the error names the head.
    with pytest.raises(StaleBaseError) as exc:
        service.append(session.session_id, op_id="b", actor_id="u2", base_revision=0, patch=REPLACE)
    assert exc.value.revision == 1
    op2, _ = service.append(
        session.session_id, op_id="b", actor_id="u2", base_revision=1, patch=REPLACE
    )
    assert op2.revision == 2
    assert [op.op_id for op in service.ops_after(session.session_id, 0)] == ["a", "b"]
    assert [op.op_id for op in service.ops_after(session.session_id, 1)] == ["b"]


def test_service_checkpoint_prunes_and_forces_resync() -> None:
    service = SessionService()
    session = service.create(scope="local", document_id="doc", snapshot={"v": 0})
    for index in range(3):
        service.append(
            session.session_id,
            op_id=f"op{index}",
            actor_id="u",
            base_revision=index,
            patch=REPLACE,
        )
    # A checkpoint must advance and must not outrun the session.
    with pytest.raises(ValueError):
        service.checkpoint(session.session_id, revision=0, document={})
    with pytest.raises(ValueError):
        service.checkpoint(session.session_id, revision=4, document={})
    service.checkpoint(session.session_id, revision=2, document={"v": 2})
    session = service.get(session.session_id)
    assert session.snapshot_revision == 2
    assert [op.op_id for op in session.ops] == ["op2"]
    # Pruned opIds are forgotten: idempotency window == retention window.
    assert "op0" not in session.ops_by_id
    # A cursor older than the snapshot cannot be replayed.
    with pytest.raises(ResyncRequiredError) as exc:
        service.ops_after(session.session_id, 1)
    assert exc.value.snapshot_revision == 2
    assert [op.op_id for op in service.ops_after(session.session_id, 2)] == ["op2"]


def test_service_validates_kind_specific_snapshots_before_mutation() -> None:
    def validator(document_kind: str, document_id: str, snapshot: object) -> str | None:
        if document_kind == "dinkster.image" and snapshot != {
            "lineage": document_id,
            "valid": True,
        }:
            return "invalid image snapshot"
        return None

    service = SessionService(snapshot_validator=validator)
    workflow = service.create(scope="local", document_id="workflow", snapshot=None)
    assert workflow.document_kind == "dinkster.workflow"
    with pytest.raises(InvalidSnapshotError):
        service.create(
            scope="local", document_id="image", document_kind="image", snapshot={"valid": False}
        )
    image = service.create(
        scope="local",
        document_id="image",
        document_kind="image",
        snapshot={"lineage": "image", "valid": True},
    )
    service.append(image.session_id, op_id="a", actor_id="u", base_revision=0, patch=REPLACE)
    with pytest.raises(InvalidSnapshotError):
        service.checkpoint(image.session_id, revision=1, document={"valid": False})
    unchanged = service.get(image.session_id)
    assert unchanged.snapshot_revision == 0
    assert unchanged.snapshot == {"lineage": "image", "valid": True}


def test_service_retention_cap_demands_snapshot() -> None:
    service = SessionService(max_retained_ops=2)
    session = service.create(scope="local", document_id="doc", snapshot=None)
    for index in range(2):
        service.append(
            session.session_id,
            op_id=f"op{index}",
            actor_id="u",
            base_revision=index,
            patch=REPLACE,
        )
    with pytest.raises(SnapshotRequiredError):
        service.append(
            session.session_id, op_id="op2", actor_id="u", base_revision=2, patch=REPLACE
        )
    # A checkpoint reopens the log.
    service.checkpoint(session.session_id, revision=2, document={"v": 2})
    op, _ = service.append(
        session.session_id, op_id="op2", actor_id="u", base_revision=2, patch=REPLACE
    )
    assert op.revision == 3


def test_service_binds_actors_to_principals() -> None:
    service = SessionService()
    session = service.create(scope="team", document_id="doc", snapshot=None)
    first, _ = service.append(
        session.session_id,
        op_id="first",
        actor_id="shared",
        base_revision=0,
        patch=REPLACE,
        principal_id="alice",
    )
    with pytest.raises(ActorPrincipalMismatchError):
        service.append(
            session.session_id,
            op_id="spoofed",
            actor_id="shared",
            base_revision=1,
            patch=REPLACE,
            principal_id="bob",
        )
    assert service.get(session.session_id).revision == first.revision
    second, _ = service.append(
        session.session_id,
        op_id="second",
        actor_id="shared",
        base_revision=1,
        patch=REPLACE,
        principal_id="alice",
    )
    third, _ = service.append(
        session.session_id,
        op_id="third",
        actor_id="bob-actor",
        base_revision=2,
        patch=REPLACE,
        principal_id="bob",
    )
    assert (second.revision, third.revision) == (2, 3)


def test_storeless_local_principal_accepts_every_actor() -> None:
    service = SessionService()
    session = service.create(scope="local", document_id="doc", snapshot=None)
    for revision, actor_id in enumerate(("first", "second", "first")):
        service.append(
            session.session_id,
            op_id=f"op-{revision}",
            actor_id=actor_id,
            base_revision=revision,
            patch=REPLACE,
        )
    assert service.get(session.session_id).revision == 3


def test_service_enforces_session_roles() -> None:
    service = SessionService()
    session = service.create(scope="team", document_id="doc", snapshot=None, principal_id="owner")
    assert service.role_for(session.session_id, "owner") == "owner"
    assert service.role_for(session.session_id, "editor") == "editor"
    service.append(
        session.session_id,
        op_id="default-editor",
        actor_id="editor",
        base_revision=0,
        patch=REPLACE,
        principal_id="editor",
    )
    service.replace_acl(
        session.session_id,
        principal_id="owner",
        default_role="editor",
        entries={"owner": "owner", "viewer": "viewer", "banned": "banned"},
    )
    assert service.ops_after(session.session_id, 0, principal_id="viewer")
    with pytest.raises(SessionRoleError):
        service.append(
            session.session_id,
            op_id="viewer",
            actor_id="viewer",
            base_revision=1,
            patch=REPLACE,
            principal_id="viewer",
        )
    for action in (
        lambda: service.ops_after(session.session_id, 0, principal_id="banned"),
        lambda: service.bind_actor(session.session_id, "banned", "banned"),
        lambda: service.close(session.session_id, principal_id="banned"),
    ):
        with pytest.raises(SessionRoleError):
            action()
    with pytest.raises(SessionRoleError):
        service.close(session.session_id, principal_id="editor")
    assert service.close(session.session_id, principal_id="owner").session_id == session.session_id


# -- HTTP/WS surface -----------------------------------------------------------


async def make_client(service: SessionService | None = None, **route_options: Any) -> TestClient:
    app = web.Application()  # bare host: no engine, no dinkster-server
    add_session_routes(app, service if service is not None else SessionService(), **route_options)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


class StubPrincipal:
    def __init__(self, principal_id: str) -> None:
        self.principal_id = principal_id

    def allows_in(self, scope: str, capability: str) -> bool:
        return True

    def scopes_for(self, capability: str) -> frozenset[str]:
        return frozenset({"team"})


async def make_authenticated_client(
    service: SessionService | None = None, **route_options: Any
) -> TestClient:
    app = web.Application()
    add_session_routes(
        app,
        service if service is not None else SessionService(),
        principal_for=lambda request: StubPrincipal(request.headers["X-Principal"]),
        resolve_scope=lambda _principal, _capability, explicit: str(explicit or "team"),
        **route_options,
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


def test_route_auth_callbacks_must_be_configured_together() -> None:
    app = web.Application()
    with pytest.raises(ValueError, match="must be provided together"):
        add_session_routes(
            app,
            SessionService(),
            principal_for=lambda _request: None,  # type: ignore[arg-type,return-value]
        )


def op_body(op_id: str, base: int, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "protocolVersion": PROTOCOL_VERSION,
        "opId": op_id,
        "actorId": "u1",
        "baseRevision": base,
        "patch": REPLACE,
    }
    body.update(overrides)
    return body


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_session_lifecycle_over_http() -> None:
    async def scenario() -> None:
        client = await make_client()
        try:
            resp = await client.post(
                "/api/sessions",
                json={"scope": "local", "documentId": "d1", "snapshot": {"v": 0}},
            )
            assert resp.status == 201
            created = await resp.json()
            assert created["protocolVersion"] == PROTOCOL_VERSION
            assert created["documentKind"] == "workflow"
            assert created["revision"] == 0 and created["snapshotRevision"] == 0
            sid = created["sessionId"]

            # Append: the full pinned envelope comes back, server-stamped.
            resp = await client.post(f"/api/sessions/{sid}/ops", json=op_body("a", 0))
            assert resp.status == 200
            envelope = await resp.json()
            assert envelope["sessionId"] == sid
            assert envelope["opId"] == "a" and envelope["actorId"] == "u1"
            assert envelope["baseRevision"] == 0 and envelope["revision"] == 1
            assert envelope["patch"] == REPLACE
            assert envelope["timestamp"] > 0 and "replayed" not in envelope

            # Idempotent replay is flagged, not re-ordered.
            resp = await client.post(f"/api/sessions/{sid}/ops", json=op_body("a", 5))
            assert (await resp.json())["replayed"] is True

            # Lost race: 409 stale-base names the head; rebase succeeds.
            resp = await client.post(f"/api/sessions/{sid}/ops", json=op_body("b", 0))
            assert resp.status == 409
            refusal = await resp.json()
            assert refusal == {"error": "stale-base", "revision": 1}
            resp = await client.post(f"/api/sessions/{sid}/ops", json=op_body("b", 1))
            assert (await resp.json())["revision"] == 2

            # Catch-up reads replay in order.
            page = await (await client.get(f"/api/sessions/{sid}/ops?after=0")).json()
            assert [op["opId"] for op in page["ops"]] == ["a", "b"]
            assert page["revision"] == 2

            # Checkpoint prunes; stale cursors are told to resync.
            resp = await client.put(
                f"/api/sessions/{sid}/snapshot", json={"revision": 2, "document": {"v": 2}}
            )
            assert (await resp.json())["snapshotRevision"] == 2
            resp = await client.get(f"/api/sessions/{sid}/ops?after=0")
            assert resp.status == 410
            assert (await resp.json()) == {
                "error": "resync-required",
                "snapshotRevision": 2,
            }
            snap = await (await client.get(f"/api/sessions/{sid}/snapshot")).json()
            assert snap == {
                "revision": 2,
                "document": {"v": 2},
                "documentKind": "workflow",
            }

            # Close: descriptor 404s afterwards.
            assert (await client.delete(f"/api/sessions/{sid}")).status == 200
            assert (await client.get(f"/api/sessions/{sid}")).status == 404
        finally:
            await client.close()

    asyncio.run(scenario())


def test_image_session_kind_and_snapshot_validation_over_http() -> None:
    def validator(document_kind: str, document_id: str, snapshot: object) -> str | None:
        if document_kind == "dinkster.image" and snapshot != {"lineage": document_id}:
            return "invalid image snapshot"
        return None

    async def scenario() -> None:
        service = SessionService(snapshot_validator=validator)
        client = await make_client(service)
        try:
            invalid_kind = await client.post(
                "/api/sessions",
                json={
                    "scope": "local",
                    "documentId": "image",
                    "documentKind": "other",
                    "snapshot": {},
                },
            )
            assert invalid_kind.status == 400
            invalid = await client.post(
                "/api/sessions",
                json={
                    "scope": "local",
                    "documentId": "image",
                    "documentKind": "image",
                    "snapshot": {},
                },
            )
            assert invalid.status == 400
            created = await client.post(
                "/api/sessions",
                json={
                    "scope": "local",
                    "documentId": "image",
                    "documentKind": "image",
                    "snapshot": {"lineage": "image"},
                },
            )
            assert created.status == 201
            descriptor = await created.json()
            assert descriptor["documentKind"] == "image"
            session_id = descriptor["sessionId"]
            assert (
                await client.post(f"/api/sessions/{session_id}/ops", json=op_body("a", 0))
            ).status == 200
            rejected = await client.put(
                f"/api/sessions/{session_id}/snapshot",
                json={"revision": 1, "document": {}},
            )
            assert rejected.status == 400
            assert (await rejected.json())["error"] == "document-invalid"
            snapshot = await client.get(f"/api/sessions/{session_id}/snapshot")
            assert await snapshot.json() == {
                "revision": 0,
                "document": {"lineage": "image"},
                "documentKind": "image",
            }
        finally:
            await client.close()

    asyncio.run(scenario())


def test_protocol_version_and_validation_refusals() -> None:
    async def scenario() -> None:
        client = await make_client()
        try:
            resp = await client.post(
                "/api/sessions",
                json={"scope": "local", "documentId": "d", "snapshot": None},
            )
            sid = (await resp.json())["sessionId"]
            # Unsupported protocolVersion is a loud 406 refusal.
            resp = await client.post(
                f"/api/sessions/{sid}/ops", json=op_body("a", 0, protocolVersion=2)
            )
            assert resp.status == 406
            assert (await resp.json()) == {
                "error": "protocol-version-unsupported",
                "requested": 2,
                "supported": [PROTOCOL_VERSION],
            }
            # Malformed envelopes are 400.
            for broken in (
                op_body("a", 0, opId=""),
                op_body("a", 0, actorId=""),
                # actorId is a joint frontend pin: [A-Za-z0-9_-]+ and never
                # __proto__ - actor ids are embedded in document node ids
                # and cursor-map keys, where "." is reserved for graph
                # addressing and "__proto__" is a JS prototype-pollution
                # hazard as an object key.
                op_body("a", 0, actorId="a.b"),
                op_body("a", 0, actorId="a b"),
                op_body("a", 0, actorId="u1\n"),
                op_body("a", 0, actorId="\u00fc1"),
                op_body("a", 0, actorId="__proto__"),
                op_body("a", 0, baseRevision="x"),
                op_body("a", 0, baseRevision=True),
                op_body("a", 0, patch=[{"op": "explode", "path": "/x"}]),
            ):
                resp = await client.post(f"/api/sessions/{sid}/ops", json=broken)
                assert resp.status == 400
            # The full pinned alphabet is accepted (UUIDs fit).
            ok = op_body("charset-ok", 0, actorId="AZaz09_-")
            resp = await client.post(f"/api/sessions/{sid}/ops", json=ok)
            assert resp.status == 200
            # Omitted scope lists the union of readable scopes (local here).
            assert (await client.get("/api/sessions")).status == 200
            listed = await (await client.get("/api/sessions?scope=local")).json()
            assert [s["sessionId"] for s in listed["sessions"]] == [sid]
        finally:
            await client.close()

    asyncio.run(scenario())


def test_http_ops_refuse_actor_reuse_by_another_principal() -> None:
    async def scenario() -> None:
        service = SessionService()
        client = await make_authenticated_client(service)
        alice = {"X-Principal": "alice"}
        bob = {"X-Principal": "bob"}
        try:
            created = await client.post(
                "/api/sessions",
                json={"scope": "team", "documentId": "d", "snapshot": None},
                headers=alice,
            )
            sid = (await created.json())["sessionId"]
            accepted = await client.post(
                f"/api/sessions/{sid}/ops", json=op_body("first", 0), headers=alice
            )
            assert accepted.status == 200
            refused = await client.post(
                f"/api/sessions/{sid}/ops", json=op_body("spoofed", 1), headers=bob
            )
            assert refused.status == 409
            assert await refused.json() == {"error": "actor-principal-mismatch"}
            assert service.get(sid).revision == 1
            continued = await client.post(
                f"/api/sessions/{sid}/ops", json=op_body("continued", 1), headers=alice
            )
            assert continued.status == 200
            other = await client.post(
                f"/api/sessions/{sid}/ops",
                json=op_body("other", 2, actorId="bob-actor"),
                headers=bob,
            )
            assert other.status == 200
        finally:
            await client.close()

    asyncio.run(scenario())


def test_acl_routes_enforce_roles_and_guard_the_last_owner() -> None:
    async def scenario() -> None:
        service = SessionService()
        client = await make_authenticated_client(service)
        alice = {"X-Principal": "alice"}
        bob = {"X-Principal": "bob"}
        banned = {"X-Principal": "banned"}
        try:
            created = await client.post(
                "/api/sessions",
                json={"scope": "team", "documentId": "d", "snapshot": None},
                headers=alice,
            )
            sid = (await created.json())["sessionId"]
            initial = await (await client.get(f"/api/sessions/{sid}/acl", headers=alice)).json()
            assert initial == {"defaultRole": "editor", "entries": {"alice": "owner"}}

            policy = {
                "defaultRole": "viewer",
                "entries": {"alice": "owner", "bob": "viewer", "banned": "banned"},
            }
            response = await client.put(f"/api/sessions/{sid}/acl", json=policy, headers=alice)
            assert response.status == 200
            assert await response.json() == policy
            assert (
                await (await client.get(f"/api/sessions/{sid}/acl", headers=alice)).json() == policy
            )

            refusals = (
                await client.post(f"/api/sessions/{sid}/ops", json=op_body("no", 0), headers=bob),
                await client.put(
                    f"/api/sessions/{sid}/snapshot",
                    json={"revision": 0, "document": None},
                    headers=bob,
                ),
                await client.delete(f"/api/sessions/{sid}", headers=bob),
                await client.get(f"/api/sessions/{sid}/acl", headers=bob),
                await client.put(f"/api/sessions/{sid}/acl", json=policy, headers=bob),
            )
            for refusal in refusals:
                assert refusal.status == 403
                assert await refusal.json() == {"error": "session-role"}
            assert (await client.get(f"/api/sessions/{sid}/snapshot", headers=bob)).status == 200

            for path in (
                f"/api/sessions/{sid}",
                f"/api/sessions/{sid}/ops?after=0",
                f"/api/sessions/{sid}/snapshot",
            ):
                refusal = await client.get(path, headers=banned)
                assert refusal.status == 403
                assert await refusal.json() == {"error": "session-role"}
            with pytest.raises(WSServerHandshakeError) as exc:
                await client.ws_connect(f"/api/sessions/{sid}/events", headers=banned)
            assert exc.value.status == 403

            no_owner = await client.put(
                f"/api/sessions/{sid}/acl",
                json={"defaultRole": "editor", "entries": {"alice": "editor"}},
                headers=alice,
            )
            assert no_owner.status == 409
            assert (await no_owner.json())["error"] == "acl-invalid"
            assert service.get(sid).acl == policy["entries"]
        finally:
            await client.close()

    asyncio.run(scenario())


def test_acl_demotion_closes_live_event_socket() -> None:
    async def scenario() -> None:
        client = await make_authenticated_client()
        alice = {"X-Principal": "alice"}
        bob = {"X-Principal": "bob"}
        try:
            created = await client.post(
                "/api/sessions",
                json={"scope": "team", "documentId": "d", "snapshot": None},
                headers=alice,
            )
            sid = (await created.json())["sessionId"]
            bob_ws = await client.ws_connect(f"/api/sessions/{sid}/events", headers=bob)
            assert json.loads((await bob_ws.receive()).data)["type"] == "session"
            response = await client.put(
                f"/api/sessions/{sid}/acl",
                json={
                    "defaultRole": "editor",
                    "entries": {"alice": "owner", "bob": "banned"},
                },
                headers=alice,
            )
            assert response.status == 200
            assert (await bob_ws.receive()).type in (
                WSMsgType.CLOSE,
                WSMsgType.CLOSING,
                WSMsgType.CLOSED,
            )
        finally:
            await client.close()

    asyncio.run(scenario())


def test_legacy_session_acl_is_visible_but_cannot_be_claimed(tmp_path: Path) -> None:
    async def scenario() -> None:
        path = tmp_path / "legacy.sqlite"
        original_store = SessionStore(path)
        original = SessionService(store=original_store)
        session = original.create(scope="team", document_id="d", snapshot=None)
        original_store.close()
        with sqlite3.connect(path) as conn:
            conn.execute("DELETE FROM session_acl")
            conn.execute("DELETE FROM session_policy")
        store = SessionStore(path)
        client = await make_authenticated_client(SessionService(store=store))
        alice = {"X-Principal": "alice"}
        try:
            response = await client.get(f"/api/sessions/{session.session_id}/acl", headers=alice)
            assert response.status == 200
            assert await response.json() == {"defaultRole": "editor", "entries": {}}
            response = await client.put(
                f"/api/sessions/{session.session_id}/acl",
                json={"defaultRole": "editor", "entries": {"alice": "owner"}},
                headers=alice,
            )
            assert response.status == 409
            assert await response.json() == {"error": "no-owner"}
            appended = await client.post(
                f"/api/sessions/{session.session_id}/ops",
                json=op_body("legacy-op", 0, actorId="legacy-actor"),
                headers=alice,
            )
            assert appended.status == 200
            closed = await client.delete(f"/api/sessions/{session.session_id}", headers=alice)
            assert closed.status == 200
            with sqlite3.connect(path) as conn:
                for table in ("sessions", "ops", "actors", "session_acl", "session_policy"):
                    assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
        finally:
            await client.close()
            store.close()

    asyncio.run(scenario())


def test_op_rate_limit_refills_and_rotating_actors_share_principal_budget(tmp_path: Path) -> None:
    async def scenario() -> None:
        clock = FakeClock()
        path = tmp_path / "sessions.sqlite"
        store = SessionStore(path)
        service = SessionService(store=store)
        client = await make_authenticated_client(
            service, op_rate=1.0, op_burst=2, monotonic_clock=clock
        )
        alice = {"X-Principal": "alice"}
        bob = {"X-Principal": "bob"}
        try:
            created = await client.post(
                "/api/sessions",
                json={"scope": "team", "documentId": "d", "snapshot": None},
                headers=alice,
            )
            sid = (await created.json())["sessionId"]
            await client.put(
                f"/api/sessions/{sid}/acl",
                json={
                    "defaultRole": "editor",
                    "entries": {"alice": "owner", "bob": "viewer"},
                },
                headers=alice,
            )
            refused = await client.post(
                f"/api/sessions/{sid}/ops",
                json=op_body("role-refused", 0, actorId="actor-c"),
                headers=bob,
            )
            assert refused.status == 403
            await client.put(
                f"/api/sessions/{sid}/acl",
                json={
                    "defaultRole": "editor",
                    "entries": {"alice": "owner", "bob": "editor"},
                },
                headers=alice,
            )
            for revision in range(2):
                response = await client.post(
                    f"/api/sessions/{sid}/ops",
                    json=op_body(f"c-{revision}", revision, actorId="actor-c"),
                    headers=bob,
                )
                assert response.status == 200
            for revision in range(2, 4):
                response = await client.post(
                    f"/api/sessions/{sid}/ops",
                    json=op_body(f"a-{revision}", revision, actorId="actor-a"),
                    headers=alice,
                )
                assert response.status == 200
            limited = await client.post(
                f"/api/sessions/{sid}/ops",
                json=op_body("limited", 4, actorId="actor-a"),
                headers=alice,
            )
            assert limited.status == 429
            assert await limited.json() == {"error": "rate-limited", "retryAfterMs": 1000}
            assert service.get(sid).revision == 4
            with sqlite3.connect(path) as conn:
                assert conn.execute("SELECT count(*) FROM ops").fetchone()[0] == 4

            independent = await client.post(
                f"/api/sessions/{sid}/ops",
                json=op_body("actor-b", 4, actorId="actor-b"),
                headers=alice,
            )
            assert independent.status == 429
            assert service.get(sid).revision == 4
            clock.advance(1.0)
            refilled = await client.post(
                f"/api/sessions/{sid}/ops",
                json=op_body("refilled", 4, actorId="actor-a"),
                headers=alice,
            )
            assert refilled.status == 200
        finally:
            await client.close()
            store.close()

    asyncio.run(scenario())


def test_presence_rate_limit_drops_excess_frames_only() -> None:
    async def scenario() -> None:
        clock = FakeClock()
        client = await make_authenticated_client(
            presence_rate=1.0, presence_burst=1, monotonic_clock=clock
        )
        alice = {"X-Principal": "alice"}
        bob = {"X-Principal": "bob"}
        try:
            created = await client.post(
                "/api/sessions",
                json={"scope": "team", "documentId": "d", "snapshot": None},
                headers=alice,
            )
            sid = (await created.json())["sessionId"]
            sender = await client.ws_connect(f"/api/sessions/{sid}/events", headers=alice)
            receiver = await client.ws_connect(f"/api/sessions/{sid}/events", headers=bob)
            await sender.receive()
            await receiver.receive()

            await sender.send_json({"type": "presence", "actorId": "actor-a", "payload": 1})
            assert json.loads((await receiver.receive()).data)["payload"] == 1
            await sender.send_json({"type": "presence", "actorId": "actor-a", "payload": 2})
            await sender.send_json({"type": "presence", "actorId": "actor-b", "payload": 3})
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(receiver.receive(), timeout=0.05)
            clock.advance(1.0)
            await sender.send_json({"type": "presence", "actorId": "actor-a", "payload": 4})
            assert json.loads((await receiver.receive()).data)["payload"] == 4
            await sender.close()
            await receiver.close()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_ws_presence_binds_actors_and_drops_spoofed_frames() -> None:
    async def scenario() -> None:
        client = await make_authenticated_client()
        alice = {"X-Principal": "alice"}
        bob = {"X-Principal": "bob"}
        try:
            created = await client.post(
                "/api/sessions",
                json={"scope": "team", "documentId": "d", "snapshot": None},
                headers=alice,
            )
            sid = (await created.json())["sessionId"]
            alice_ws = await client.ws_connect(f"/api/sessions/{sid}/events", headers=alice)
            bob_ws = await client.ws_connect(f"/api/sessions/{sid}/events", headers=bob)
            await alice_ws.receive()
            await bob_ws.receive()

            await alice_ws.send_json({"type": "presence", "actorId": "alice-actor"})
            assert json.loads((await bob_ws.receive()).data)["actorId"] == "alice-actor"
            await bob_ws.send_json(
                {"type": "presence", "actorId": "alice-actor", "payload": {"spoofed": True}}
            )
            await bob_ws.send_json(
                {"type": "presence", "actorId": "bob-actor", "payload": {"valid": True}}
            )
            assert json.loads((await alice_ws.receive()).data) == {
                "type": "presence",
                "actorId": "bob-actor",
                "payload": {"valid": True},
            }
            await alice_ws.close()
            await bob_ws.close()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_ws_delivers_ops_and_relays_presence() -> None:
    async def scenario() -> None:
        client = await make_client()
        try:
            resp = await client.post(
                "/api/sessions",
                json={"scope": "local", "documentId": "d", "snapshot": None},
            )
            sid = (await resp.json())["sessionId"]
            ws1 = await client.ws_connect(f"/api/sessions/{sid}/events")
            ws2 = await client.ws_connect(f"/api/sessions/{sid}/events")
            # Connect handshake: the session descriptor.
            for ws in (ws1, ws2):
                hello = json.loads((await ws.receive()).data)
                assert hello["type"] == "session" and hello["sessionId"] == sid

            # An HTTP-appended op fans out to every subscriber.
            await client.post(f"/api/sessions/{sid}/ops", json=op_body("a", 0))
            for ws in (ws1, ws2):
                frame = json.loads((await ws.receive()).data)
                assert frame["type"] == "op" and frame["opId"] == "a"
                assert frame["revision"] == 1 and frame["patch"] == REPLACE

            # Presence relays to OTHER subscribers only, never echoed.
            # A pin-violating actorId drops like any malformed frame
            # (joint pin: [A-Za-z0-9_-]+ and never __proto__): ws2 sees
            # only the valid frame sent after them, proving the drops.
            await ws1.send_str(
                json.dumps({"type": "presence", "actorId": "u.1", "payload": {"x": 2}})
            )
            await ws1.send_str(
                json.dumps({"type": "presence", "actorId": "__proto__", "payload": {"x": 2}})
            )
            await ws1.send_str(
                json.dumps({"type": "presence", "actorId": "u1", "payload": {"x": 3}})
            )
            frame = json.loads((await ws2.receive()).data)
            assert frame == {"type": "presence", "actorId": "u1", "payload": {"x": 3}}

            # Closing the session notifies and closes subscribers.
            await client.delete(f"/api/sessions/{sid}")
            frame = json.loads((await ws1.receive()).data)
            assert frame == {"type": "session_closed"}
            assert (await ws1.receive()).type in (
                WSMsgType.CLOSE,
                WSMsgType.CLOSING,
                WSMsgType.CLOSED,
            )
            await ws1.close()
            await ws2.close()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_ws_synthesizes_session_closed_when_delete_wins_the_handshake() -> None:
    """The lossy-DELETE race (frontend live report, 2026-07-26): a DELETE
    landing between the WS handler's session read and its subscriber
    registration broadcast session_closed to nobody and popped the set;
    the late subscriber's replay then died on UnknownSessionError -
    socket closed, terminal frame never delivered. The handler now
    synthesizes the frame itself when the replay finds the session gone.
    Simulated deterministically: the service reports the session unknown
    at exactly the handler's replay read - the state that interleaving
    produces."""

    class DeleteWinsService(SessionService):
        rigged = False

        def ops_after(
            self, session_id: str, after: int, *, principal_id: str = "local"
        ) -> list[SessionOp]:
            if self.rigged:
                self.rigged = False
                raise UnknownSessionError(session_id)
            return super().ops_after(session_id, after, principal_id=principal_id)

    async def scenario() -> None:
        service = DeleteWinsService()
        client = await make_client(service)
        try:
            resp = await client.post(
                "/api/sessions",
                json={"scope": "local", "documentId": "d", "snapshot": None},
            )
            sid = (await resp.json())["sessionId"]
            service.rigged = True
            ws = await client.ws_connect(f"/api/sessions/{sid}/events")
            hello = json.loads((await ws.receive()).data)
            assert hello["type"] == "session" and hello["sessionId"] == sid
            frame = json.loads((await ws.receive()).data)
            assert frame == {"type": "session_closed"}
            assert (await ws.receive()).type in (
                WSMsgType.CLOSE,
                WSMsgType.CLOSING,
                WSMsgType.CLOSED,
            )
            await ws.close()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_delete_always_delivers_session_closed_to_registered_subscribers() -> None:
    """Regression canary for the same race under real scheduling: the
    descriptor now rides the sender queue AFTER registration, so a
    client that has received it is provably registered, and a subsequent
    DELETE's broadcast cannot miss it. The old order (descriptor sent
    before registration) failed rounds like these ~1 in 5 live."""

    async def scenario() -> None:
        client = await make_client()
        try:
            for _ in range(10):
                resp = await client.post(
                    "/api/sessions",
                    json={"scope": "local", "documentId": "d", "snapshot": None},
                )
                sid = (await resp.json())["sessionId"]
                ws1 = await client.ws_connect(f"/api/sessions/{sid}/events")
                ws2 = await client.ws_connect(f"/api/sessions/{sid}/events")
                for ws in (ws1, ws2):
                    hello = json.loads((await ws.receive()).data)
                    assert hello["type"] == "session"
                await client.delete(f"/api/sessions/{sid}")
                for ws in (ws1, ws2):
                    frame = json.loads((await ws.receive()).data)
                    assert frame == {"type": "session_closed"}
                    await ws.close()
        finally:
            await client.close()

    asyncio.run(scenario())
