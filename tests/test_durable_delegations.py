import asyncio
import hashlib
import sqlite3
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

from aiohttp import WSMsgType
from aiohttp.test_utils import TestClient, TestServer
from dinkster_collab import SessionService, SessionStore, add_session_routes
from dinkster_server.app import create_app
from dinkster_server.auth import (
    CAPABILITIES,
    USER_SESSIONS_KEY,
    Principal,
    PrincipalPermissionStore,
    TokenAuthenticator,
    principal_for,
    resolve_scope,
)
from joserfc.jwk import OKPKey
from test_server import SCHEMAS, echo_graph, make_engine, submit_body
from test_token_authenticator import (
    AUDIENCE,
    ISSUER,
    PRINCIPAL_ID,
    SCOPE,
    _claims,
    _JwksService,
    _sign,
)


@asynccontextmanager
async def identity() -> AsyncIterator[tuple[OKPKey, str]]:
    key = OKPKey.generate_key("Ed25519")
    jwks = _JwksService(key)
    try:
        yield key, await jwks.start()
    finally:
        await jwks.close()


def human_headers(key: OKPKey, **claims: object) -> dict[str, str]:
    return {
        "Authorization": "Bearer "
        + _sign(key, _claims(grants={SCOPE: sorted(CAPABILITIES)}, **claims))
    }


@asynccontextmanager
async def server(path: Path, url: str | None, freshness: float = 600) -> AsyncIterator[TestClient]:
    permissions = PrincipalPermissionStore(path / "principals.sqlite")
    sessions = SessionStore(path / "sessions.sqlite")
    app = create_app(
        make_engine,
        SCHEMAS,
        authenticator=TokenAuthenticator(url, ISSUER, AUDIENCE) if url else None,
        principal_permissions=permissions,
        user_session_freshness_seconds=freshness,
    )
    add_session_routes(
        app,
        SessionService(store=sessions),
        principal_for=principal_for,
        resolve_scope=resolve_scope,
    )
    client = TestClient(TestServer(app))
    try:
        await client.start_server()
        yield client
    finally:
        await client.close()
        sessions.close()


async def mint(client: TestClient, human: dict[str, str], **options: object) -> dict[str, object]:
    response = await client.post(
        "/api/auth/delegations",
        headers=human,
        json={"scope": SCOPE, "displayName": "Test agent", **options},
    )
    assert response.status == 201
    return await response.json()


def test_delegation_and_revocation_survive_restart(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with identity() as (key, url):
            human = human_headers(key)
            async with server(tmp_path, url) as client:
                issued = await mint(client, human)
                assert "expiresAt" not in issued
                agent = {"Authorization": "Bearer " + str(issued["token"])}
                assert (await client.get("/api/jobs", headers=agent)).status == 200
                expiring = await mint(client, human, expiresInSeconds=86400)
                assert float(str(expiring["expiresAt"])) > time.time() + 86000
                await client.put(
                    f"/api/principals/{PRINCIPAL_ID}/permissions",
                    headers=human,
                    json={"execute": False},
                )
            with sqlite3.connect(tmp_path / "principals.sqlite") as db:
                row = db.execute(
                    "SELECT principal_id, agent_principal_id, credential_hash, created_at,"
                    " expires_at, revoked_at FROM delegations WHERE id = ?",
                    (issued["id"],),
                ).fetchone()
                assert row[0] == PRINCIPAL_ID
                assert row[1].startswith("agent:")
                assert row[2] == hashlib.sha256(str(issued["token"]).encode()).hexdigest()
                assert row[3] > 0 and row[4:] == (None, None)
            async with server(tmp_path, url) as client:
                response = await client.get("/api/jobs", headers=agent)
                assert response.status == 403
                assert await response.json() == {"error": "user-session-required"}
                assert (await client.get("/api/jobs", headers=human)).status == 200
                assert (await client.get("/api/jobs", headers=agent)).status == 200
                response = await client.post(
                    "/api/jobs", headers=agent, json=submit_body(echo_graph(), ["s"], scope=SCOPE)
                )
                assert response.status == 403
                assert (await response.json())["error"] == "capability-required"
                assert (
                    await client.delete("/api/auth/delegations/" + str(issued["id"]), headers=human)
                ).status == 200
            async with server(tmp_path, url) as client:
                assert (await client.get("/api/jobs", headers=human)).status == 200
                assert (await client.get("/api/jobs", headers=agent)).status == 401
                listed = await (await client.get("/api/auth/delegations", headers=human)).json()
                assert [item["id"] for item in listed] == [expiring["id"]]
            with sqlite3.connect(tmp_path / "principals.sqlite") as db:
                assert (
                    db.execute(
                        "SELECT revoked_at FROM delegations WHERE id = ?", (issued["id"],)
                    ).fetchone()[0]
                    > 0
                )

    asyncio.run(scenario())


def test_freshness_suspends_http_and_both_sockets_then_resumes(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with identity() as (key, url), server(tmp_path, url, freshness=2) as client:
            human = human_headers(key)
            issued = await mint(client, human)
            agent = {"Authorization": "Bearer " + str(issued["token"])}
            response = await client.post(
                "/api/sessions",
                headers=human,
                json={"scope": SCOPE, "documentId": "d", "snapshot": {}},
            )
            path = "/api/sessions/" + (await response.json())["sessionId"]
            sockets = []
            for route in ("/api/events", path + "/events"):
                ticket = await (await client.post("/api/auth/ws-ticket", headers=agent)).json()
                sockets.append(await client.ws_connect(route + "?ticket=" + ticket["ticket"]))
            await sockets[1].receive_json()
            await asyncio.sleep(2.05)
            for route in ("/api/jobs", path, path + "/ops?after=0", path + "/snapshot"):
                response = await client.get(route, headers=agent)
                assert response.status == 403
                assert await response.json() == {"error": "user-session-required"}
            response = await client.post("/api/auth/ws-ticket", headers=agent)
            assert response.status == 403
            assert await response.json() == {"error": "user-session-required"}
            async with asyncio.timeout(2):
                for socket in sockets:
                    closed = await socket.receive()
                    assert closed.type == WSMsgType.CLOSE
                    assert closed.data == 1008 and closed.extra == "user-session-required"
            assert (
                await client.get("/api/jobs", headers={"Authorization": "Bearer invalid"})
            ).status == 401
            expired = human_headers(key, iat=int(time.time()) - 100, exp=int(time.time()) - 1)
            assert (await client.get("/api/jobs", headers=expired)).status in (401, 403)
            response = await client.get("/api/jobs", headers=agent)
            assert response.status == 403
            assert await response.json() == {"error": "user-session-required"}
            client.app[USER_SESSIONS_KEY].verified(
                Principal(
                    PRINCIPAL_ID,
                    {SCOPE: CAPABILITIES},
                    kind="agent",
                    expires_at=time.time() + 60,
                    verified_jwt=True,
                )
            )
            response = await client.get("/api/jobs", headers=agent)
            assert response.status == 403
            assert await response.json() == {"error": "user-session-required"}
            assert (await client.get("/api/jobs", headers=agent)).status == 403
            assert (await client.post("/api/auth/ws-ticket", headers=human)).status == 200
            assert (await client.get("/api/jobs", headers=agent)).status == 200
            async with client.ws_connect(path + "/events", headers=agent) as socket:
                assert (await socket.receive_json())["type"] == "session"
            await asyncio.sleep(2.05)
            assert (await client.get("/api/jobs", headers=agent)).status == 403
            async with client.ws_connect("/api/events", headers=human):
                assert (await client.get("/api/jobs", headers=agent)).status == 200
            listed = await (await client.get("/api/auth/delegations", headers=human)).json()
            assert [item["id"] for item in listed] == [issued["id"]]

    asyncio.run(scenario())


def test_fresher_user_jwt_replaces_original_expiry_and_role_ceiling(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with identity() as (key, url), server(tmp_path, url, freshness=2) as client:
            expires = int(time.time()) + 2
            original = human_headers(key, exp=expires)
            issued = await mint(client, original)
            agent = {"Authorization": "Bearer " + str(issued["token"])}
            human = human_headers(key)
            while time.time() <= expires:
                assert (await client.get("/api/jobs", headers=human)).status == 200
                await asyncio.sleep(0.1)
            assert (await client.get("/api/jobs", headers=agent)).status == 200
            reduced = {
                "Authorization": "Bearer " + _sign(key, _claims(grants={SCOPE: ["jobs:read"]}))
            }
            assert (await client.get("/api/jobs", headers=reduced)).status == 200
            assert (await client.get("/api/jobs", headers=agent)).status == 200
            response = await client.post(
                "/api/jobs", headers=agent, json=submit_body(echo_graph(), ["s"], scope=SCOPE)
            )
            assert (
                response.status == 403 and (await response.json())["error"] == "capability-required"
            )
            async with client.ws_connect("/api/events", headers=agent) as socket:
                withdrawn = {"Authorization": "Bearer " + _sign(key, _claims(grants={SCOPE: []}))}
                assert (await client.post("/api/auth/ws-ticket", headers=withdrawn)).status == 200
                response = await client.get("/api/jobs", headers=agent)
                assert (
                    response.status == 403
                    and (await response.json())["error"] == "capability-required"
                )
                async with asyncio.timeout(2):
                    closed = await socket.receive()
                    assert closed.type == WSMsgType.CLOSE
                    assert closed.data == 1008 and closed.extra == "authorization-expired"

    asyncio.run(scenario())


def test_persisted_unrevoked_delegation_caps(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with identity() as (key, url):
            first = human_headers(key)
            async with server(tmp_path, url) as client:
                issued = [await mint(client, first) for _ in range(32)]
            async with server(tmp_path, url) as client:
                response = await client.post(
                    "/api/auth/delegations",
                    headers=first,
                    json={"scope": SCOPE, "displayName": "overflow"},
                )
                assert (
                    response.status == 429
                    and (await response.json())["error"] == "delegation-limit"
                )
                for user in range(127):
                    human = human_headers(key, sub="u_" + str(UUID(int=user + 1000)))
                    for _ in range(32):
                        await mint(client, human)
            async with server(tmp_path, url) as client:
                other = human_headers(key, sub="u_" + str(UUID(int=9999)))
                response = await client.post(
                    "/api/auth/delegations",
                    headers=other,
                    json={"scope": SCOPE, "displayName": "overflow"},
                )
                assert (
                    response.status == 429
                    and (await response.json())["error"] == "delegation-limit"
                )
                assert (
                    await client.delete(
                        "/api/auth/delegations/" + str(issued[0]["id"]), headers=first
                    )
                ).status == 200
                await mint(client, other)
                with sqlite3.connect(tmp_path / "principals.sqlite") as db:
                    assert (
                        db.execute(
                            "SELECT count(*) FROM delegations WHERE revoked_at IS NULL"
                        ).fetchone()[0]
                        == 4096
                    )

    asyncio.run(scenario())


def test_auth_off_agents_do_not_need_a_fresh_jwt(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with server(tmp_path, None, freshness=0.01) as client:
            await asyncio.sleep(0.02)
            agent = {"X-Dinkster-Actor-Kind": "agent"}
            assert (await client.get("/api/jobs", headers=agent)).status == 200
            response = await client.post("/api/auth/ws-ticket", headers=agent)
            assert response.status == 200
            ticket = (await response.json())["ticket"]
            async with client.ws_connect("/api/events?ticket=" + ticket) as socket:
                await asyncio.sleep(1.1)
                assert not socket.closed

    asyncio.run(scenario())
