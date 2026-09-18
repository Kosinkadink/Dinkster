import asyncio
import sqlite3
import time
from pathlib import Path

import pytest
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestClient, TestServer
from dinkster_collab import SessionService, SessionStore, add_session_routes
from dinkster_collab.routes import _TokenBucketLimiter
from dinkster_collab.sessions import ActorLimitError
from dinkster_server.auth import (
    CAPABILITIES,
    DelegationStore,
    Principal,
    PrincipalPermissionStore,
    StaticBearerAuthenticator,
    TokenAuthenticator,
    add_principal_routes,
    handle_ws_ticket,
    install_auth,
    principal_for,
    resolve_scope,
)
from joserfc.jwk import OKPKey
from test_token_authenticator import (
    AUDIENCE,
    ISSUER,
    PRINCIPAL_ID,
    SCOPE,
    _claims,
    _JwksService,
    _sign,
)


@pytest.mark.parametrize("jwt_auth", [False, True])
def test_delegate_ceiling_ownership_kind_revocation_and_durable_attribution(
    tmp_path: Path, jwt_auth: bool
) -> None:
    async def scenario() -> None:
        key = OKPKey.generate_key("Ed25519")
        jwks = _JwksService(key)
        url = await jwks.start()
        user = Principal(PRINCIPAL_ID, {SCOPE: CAPABILITIES - {"principals:manage"}})
        authenticator = (
            TokenAuthenticator(url, ISSUER, AUDIENCE)
            if jwt_auth
            else StaticBearerAuthenticator(
                {
                    "user-credential": user,
                    "readonly-user": Principal(PRINCIPAL_ID, {SCOPE: frozenset({"sessions:read"})}),
                }
            )
        )
        human_token = (
            _sign(key, _claims(grants={SCOPE: sorted(user.grants[SCOPE])}))
            if jwt_auth
            else "user-credential"
        )
        human = {"Authorization": f"Bearer {human_token}"}
        permissions = PrincipalPermissionStore(tmp_path / "permissions.sqlite")
        store = SessionStore(tmp_path / "sessions.sqlite")
        service = SessionService(store=store)
        app = web.Application()
        install_auth(app, authenticator, permission_store=permissions)
        add_principal_routes(app, authenticator, permissions)
        add_session_routes(app, service, principal_for=principal_for, resolve_scope=resolve_scope)
        app.router.add_post("/api/auth/ws-ticket", handle_ws_ticket)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.get("/api/principals", headers=human)
            assert response.status == 200
            assert (await response.json())[0]["principalId"] == PRINCIPAL_ID
            response = await client.post(
                "/api/sessions",
                headers=human,
                json={
                    "scope": SCOPE,
                    "documentId": "d",
                    "snapshot": {},
                },
            )
            sid = (await response.json())["sessionId"]
            path = f"/api/sessions/{sid}"
            assert (await client.get(path)).status == 401
            readonly_token = (
                _sign(key, _claims(grants={SCOPE: ["sessions:read"]}))
                if jwt_auth
                else "readonly-user"
            )
            readonly = {"Authorization": f"Bearer {readonly_token}"}
            assert (await client.get(path + "/acl", headers=readonly)).status == 200
            assert (await client.put(path + "/acl", headers=readonly, json={})).status == 403
            response = await client.post(
                "/api/auth/delegations",
                headers=human,
                json={
                    "scope": SCOPE,
                    "sessionId": sid,
                    "displayName": "Delegate",
                },
            )
            assert response.status == 201
            issued = await response.json()
            assert issued["token"] != human_token
            agent = {"Authorization": f"Bearer {issued['token']}"}
            assert (await client.get(path + "/snapshot", headers=agent)).status == 200
            assert (await client.get("/api/sessions/another", headers=agent)).status == 403
            assert (
                await client.post("/api/auth/delegations", headers=agent, json={})
            ).status == 403
            assert (
                await client.put(
                    f"/api/principals/{PRINCIPAL_ID}/permissions",
                    headers=agent,
                    json={"edit": True},
                )
            ).status == 403
            service.bind_actor(sid, "owned", "another-user")
            refused = await client.post(path + "/actors", headers=human, json={"actorId": "owned"})
            assert refused.status == 409
            assert await refused.json() == {"error": "actor-principal-mismatch"}
            assert (
                await client.post(path + "/actors", headers=agent, json={"actorId": "agent-one"})
            ).status == 200
            body = {
                "protocolVersion": 1,
                "actorId": "agent-one",
                "opId": "op1",
                "baseRevision": 0,
                "patch": [{"op": "add", "path": ["value"], "value": 1}],
            }
            response = await client.post(path + "/ops", headers=agent, json=body)
            assert response.status == 200
            assert set(await response.json()) == {
                "protocolVersion",
                "sessionId",
                "actorId",
                "opId",
                "baseRevision",
                "revision",
                "patch",
                "timestamp",
            }
            recorded = service.get(sid).ops[0]
            assert (recorded.principal_id, recorded.actor_id, recorded.actor_kind) == (
                PRINCIPAL_ID,
                "agent-one",
                "agent",
            )
            with sqlite3.connect(tmp_path / "sessions.sqlite") as db:
                assert db.execute("SELECT principal_id, actor_kind FROM ops").fetchone() == (
                    PRINCIPAL_ID,
                    "agent",
                )
            ticket_response = await client.post("/api/auth/ws-ticket", headers=agent)
            ticket = (await ticket_response.json())["ticket"]
            receiver = await client.ws_connect(path + "/events", headers=human)
            sender = await client.ws_connect(path + "/events?ticket=" + ticket)
            await receiver.receive_json()
            await sender.receive_json()
            await sender.send_json(
                {
                    "type": "presence",
                    "actorId": "agent-one",
                    "payload": {
                        "v": 1,
                        "graph": "g0",
                        "selection": [],
                        "identity": {"kind": "human"},
                        "activity": {
                            "v": 1,
                            "type": "agent_tool_call",
                            "tool": "node.add",
                            "status": "running",
                            "pendingAsks": [],
                        },
                    },
                }
            )
            presence = await asyncio.wait_for(receiver.receive_json(), timeout=2)
            assert presence["payload"]["identity"] == {
                "kind": "agent",
                "owner": PRINCIPAL_ID,
                "displayName": "Delegate",
            }
            assert presence["payload"]["activity"]["status"] == "running"
            response = await client.put(
                f"/api/principals/{PRINCIPAL_ID}/permissions", headers=human, json={"edit": False}
            )
            assert response.status == 200
            body.update(opId="op2", baseRevision=1)
            assert (await client.post(path + "/ops", headers=agent, json=body)).status == 403
            permissions.update(PRINCIPAL_ID, {"edit": True})
            service.replace_acl(
                sid,
                principal_id=PRINCIPAL_ID,
                default_role="viewer",
                entries={"other-owner": "owner", PRINCIPAL_ID: "viewer"},
            )
            response = await client.post(path + "/ops", headers=agent, json=body)
            assert response.status == 403
            assert (await response.json())["error"] == "session-role"
            assert (
                await client.delete("/api/auth/delegations/" + issued["id"], headers=human)
            ).status == 200
            assert (await client.get(path, headers=agent)).status == 401
            assert (await asyncio.wait_for(sender.receive(), timeout=2)).type in {
                WSMsgType.CLOSE,
                WSMsgType.CLOSED,
            }
            await receiver.close()
        finally:
            await client.close()
            await jwks.close()
            permissions.close()
            store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("withdrawal", ["revoke", "read-toggle", "expiry"])
@pytest.mark.parametrize("event_delay", [0, 1.1], ids=["immediate-event", "idle-close"])
def test_job_event_stream_closes_when_delegate_authority_is_withdrawn(
    withdrawal: str, event_delay: float
) -> None:
    from dinkster_server.app import create_app
    from test_server import SCHEMAS, echo_graph, make_engine, submit_body

    async def scenario() -> None:
        key = OKPKey.generate_key("Ed25519")
        jwks = _JwksService(key)
        url = await jwks.start()
        human = {
            "Authorization": "Bearer " + _sign(key, _claims(grants={SCOPE: sorted(CAPABILITIES)}))
        }
        client = TestClient(
            TestServer(
                create_app(
                    make_engine, SCHEMAS, authenticator=TokenAuthenticator(url, ISSUER, AUDIENCE)
                )
            )
        )
        await client.start_server()
        try:
            minted = await client.post(
                "/api/auth/delegations",
                headers=human,
                json={
                    "scope": SCOPE,
                    "displayName": "Job observer",
                    "expiresInSeconds": 3 if withdrawal == "expiry" else 600,
                },
            )
            assert minted.status == 201
            delegation = await minted.json()
            agent = {"Authorization": "Bearer " + delegation["token"]}
            response = await client.post("/api/auth/ws-ticket", headers=agent)
            assert response.status == 200
            ticket = (await response.json())["ticket"]
            observer = await client.ws_connect("/api/events?ticket=" + ticket)
            human_observer = await client.ws_connect("/api/events", headers=human)
            submitted = await client.post(
                "/api/jobs",
                headers=human,
                json=submit_body(echo_graph(), ["s"], scope=SCOPE, jobId="before"),
            )
            assert submitted.status == 202
            for websocket in (observer, human_observer):
                async with asyncio.timeout(5):
                    while True:
                        event = await websocket.receive_json()
                        if event["type"] == "job_state" and event["state"] == "completed":
                            assert event["jobId"] == "before"
                            break

            if withdrawal == "revoke":
                response = await client.delete(
                    "/api/auth/delegations/" + delegation["id"], headers=human
                )
                assert response.status == 200
            elif withdrawal == "read-toggle":
                response = await client.put(
                    f"/api/principals/{PRINCIPAL_ID}/permissions",
                    headers=human,
                    json={"read": False},
                )
                assert response.status == 200
            else:
                await asyncio.sleep(max(0, delegation["expiresAt"] - time.time()))
            withdrawn_at = time.monotonic()
            expected_status = 403 if withdrawal == "read-toggle" else 401
            if withdrawal == "expiry":
                async with asyncio.timeout(3):
                    response = await client.get("/api/jobs", headers=agent)
                    while response.status == 200:
                        await asyncio.sleep(0.01)
                        response = await client.get("/api/jobs", headers=agent)
                    assert response.status == expected_status
            else:
                assert (await client.get("/api/jobs", headers=agent)).status == expected_status
            async with asyncio.timeout(5):
                closing = asyncio.create_task(observer.receive())
                try:
                    closed = None
                    if event_delay:
                        closed = await closing
                    submitted = await client.post(
                        "/api/jobs",
                        headers=human,
                        json=submit_body(echo_graph(), ["s"], scope=SCOPE, jobId="after"),
                    )
                    assert submitted.status == 202
                    event = await human_observer.receive_json()
                    assert event["type"] == "job_state"
                    assert event["jobId"] == "after"
                    assert {"clientId", "jobRef", "runId", "seq"} <= event.keys()
                    if closed is None:
                        closed = await closing
                    assert closed.type == WSMsgType.CLOSE
                    assert closed.data == 1008
                    assert closed.extra == "authorization-expired"
                    assert time.monotonic() - withdrawn_at < 2
                finally:
                    closing.cancel()
                    await asyncio.gather(closing, return_exceptions=True)
            await human_observer.close()
        finally:
            await client.close()
            await jwks.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("binary", [False, True], ids=["json", "binary"])
def test_job_event_stream_rechecks_authority_after_dequeue(
    monkeypatch: pytest.MonkeyPatch, binary: bool
) -> None:
    from dinkster_server.app import STATE_KEY, create_app
    from dinkster_server.events import BINARY_BLOB_KEY, Subscription
    from test_server import SCHEMAS, make_engine

    async def scenario() -> None:
        permissions = PrincipalPermissionStore()
        authenticator = StaticBearerAuthenticator(
            {"observer": Principal("agent", {"shared": CAPABILITIES}, kind="agent")}
        )
        app = create_app(
            make_engine, SCHEMAS, authenticator=authenticator, principal_permissions=permissions
        )
        original_get = Subscription.get

        async def withdraw_after_dequeue(sub: Subscription) -> dict[str, object] | None:
            event = await original_get(sub)
            permissions.update("agent", {"read": False})
            return event

        monkeypatch.setattr(Subscription, "get", withdraw_after_dequeue)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            headers = {"Authorization": "Bearer observer"}
            ticket = await (await client.post("/api/auth/ws-ticket", headers=headers)).json()
            observer = await client.ws_connect("/api/events?ticket=" + ticket["ticket"])
            event: dict[str, object] = {"type": "queued-probe"}
            if binary:
                event[BINARY_BLOB_KEY] = b"private-preview"
            app[STATE_KEY].hub.publish(event, client_id=None, droppable=False)
            async with asyncio.timeout(2):
                closed = await observer.receive()
            assert closed.type == WSMsgType.CLOSE
            assert closed.data == 1008
            assert closed.extra == "authorization-expired"
        finally:
            await client.close()

    asyncio.run(scenario())


def test_delegation_expiry_and_capacity() -> None:
    store = DelegationStore()
    user = Principal("u", {"scope": CAPABILITIES}, expires_at=time.time() + 2)
    issued = store.mint(user, "scope", "agent", None, 600)
    assert issued["expiresAt"] == user.expires_at
    principal = store.authenticate(str(issued["token"]))
    assert principal is not None and principal.kind == "agent"
    for _ in range(31):
        store.mint(user, "scope", "agent", None, 600)
    with pytest.raises(web.HTTPTooManyRequests):
        store.mint(user, "scope", "agent", None, 600)
    expired = store.mint(
        Principal("expired", {"scope": CAPABILITIES}, expires_at=1), "scope", "agent", None, 600
    )
    assert store.authenticate(str(expired["token"])) is None


def test_principal_kind_budgets_and_actor_storage_are_bounded() -> None:
    now = [0.0]
    limiter = _TokenBucketLimiter(2, 2, lambda: now[0])
    assert limiter.consume("user", "agent") is None
    assert limiter.consume("user", "agent") == 1000
    assert limiter.consume("user", "human") is None
    assert limiter.consume("user", "human") is None
    assert limiter.consume("user", "human") == 500
    now[0] = 10
    assert limiter.consume("another", "agent") is None
    assert len(limiter._buckets) == 1
    service = SessionService()
    sid = service.create(scope="local", document_id="d", snapshot={}).session_id
    for index in range(256):
        service.bind_actor(sid, f"actor-{index}", "local")
    with pytest.raises(ActorLimitError):
        service.bind_actor(sid, "overflow", "local")
    service.bind_actor(sid, "actor-0", "local")
    assert len(service.get(sid).actor_bindings) == 256


def test_store_failure_does_not_advance_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SessionStore(tmp_path / "sessions.sqlite")
    service = SessionService(store=store)
    sid = service.create(scope="local", document_id="d", snapshot={}).session_id

    def fail(*args: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(store, "append_op", fail)
    with pytest.raises(OSError):
        service.append(
            sid,
            op_id="op",
            actor_id="agent",
            base_revision=0,
            patch=[{"op": "add", "path": ["x"], "value": 1}],
        )
    assert service.get(sid).revision == 0
    assert service.get(sid).actor_bindings == {}
    store.close()


@pytest.mark.parametrize("auth_enabled", [False, True])
def test_agent_run_attribution_and_local_mode(auth_enabled: bool, tmp_path: Path) -> None:
    from dinkster_server.app import STATE_KEY, create_app
    from dinkster_server.history import HistoryStore, record_from_job
    from test_server import SCHEMAS, echo_graph, make_engine, submit_body

    async def scenario() -> None:
        authenticator = (
            StaticBearerAuthenticator(
                {"fixture-user": Principal("alice", {"shared": CAPABILITIES})}
            )
            if auth_enabled
            else None
        )
        client = TestClient(
            TestServer(create_app(make_engine, SCHEMAS, authenticator=authenticator))
        )
        await client.start_server()
        try:
            listed = await client.get(
                "/api/principals",
                headers={"Authorization": "Bearer fixture-user"} if auth_enabled else {},
            )
            assert listed.status == 200
            assert (await listed.json())[0]["local"] is not auth_enabled
            headers = {"X-Dinkster-Actor-Kind": "agent"}
            if auth_enabled:
                response = await client.post(
                    "/api/auth/delegations",
                    headers={"Authorization": "Bearer fixture-user"},
                    json={"scope": "shared", "displayName": "Run agent"},
                )
                assert response.status == 201
                headers["Authorization"] = "Bearer " + (await response.json())["token"]
            response = await client.post(
                "/api/jobs",
                headers=headers,
                json=submit_body(echo_graph(), ["s"], clientId="agent-run", scope="shared"),
            )
            assert response.status == 202
            wire = await response.json()
            principal_id = "alice" if auth_enabled else "local"
            assert wire["submittedBy"] == {"principalId": principal_id, "kind": "agent"}
            assert wire["clientId"] == "agent-run"
            job = client.app[STATE_KEY].queue.get("agent-run", "j1")
            assert job is not None
            assert job.scope == ("shared" if auth_enabled else "local")
            for _ in range(200):
                if job.state == "completed":
                    break
                await asyncio.sleep(0.01)
            assert job.state == "completed"
            history = HistoryStore(tmp_path / "run-history.sqlite")
            try:
                history.put(record_from_job(job))
                record = history.get(job.scope, job.run_id)
                assert record is not None
                assert (record.principal_id, record.principal_kind, record.client_id) == (
                    principal_id,
                    "agent",
                    "agent-run",
                )
            finally:
                history.close()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_local_agents_keep_explicit_history_journal_and_training_scopes() -> None:
    from dataclasses import replace

    from aiohttp.test_utils import make_mocked_request
    from dinkster_server.auth import LOCAL_PRINCIPAL, PRINCIPAL_KEY
    from dinkster_server.execution_journal import _run_scope
    from dinkster_server.history import _history_scope
    from dinkster_server.training_sessions import _training_scope

    for resolve in (_history_scope, _run_scope, _training_scope):
        request = make_mocked_request("GET", "/")
        request[PRINCIPAL_KEY] = replace(LOCAL_PRINCIPAL, kind="agent")
        with pytest.raises(web.HTTPBadRequest, match="Bad Request"):
            resolve(request)
        scoped = make_mocked_request("GET", "/?scope=local")
        scoped[PRINCIPAL_KEY] = request[PRINCIPAL_KEY]
        assert resolve(scoped) == "local"
