from __future__ import annotations

import asyncio
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from dinkster_server import (
    AuthError,
    Principal,
    PrincipalPermissionStore,
    StaticBearerAuthenticator,
    load_authenticator,
)
from dinkster_server.auth import (
    AGENT_PERMISSION_DEFAULTS,
    add_principal_routes,
    handle_ws_ticket,
    install_auth,
)


def _authenticator(*, include_limited: bool = False) -> StaticBearerAuthenticator:
    principals = {
        "agent-token": Principal(
            "worker",
            {
                "workspace": frozenset(
                    {
                        "settings:read",
                        "settings:write",
                        "queue:control",
                        "jobs:submit",
                        "jobs:read",
                    }
                )
            },
            kind="agent",
        ),
        "operator-token": Principal(
            "operator",
            {"workspace": frozenset({"principals:manage"})},
        ),
        "reader-token": Principal(
            "reader",
            {"workspace": frozenset({"settings:read"})},
        ),
    }
    if include_limited:
        principals["limited-token"] = Principal(
            "limited",
            {"workspace": frozenset({"settings:read"})},
            kind="agent",
        )
    return StaticBearerAuthenticator(principals)


def _write_auth(path: Path, *, kind: str | None = None, extra: str = "") -> None:
    kind_line = "" if kind is None else f'kind = "{kind}"\n'
    path.write_text(
        "version = 1\n"
        "[[tokens]]\n"
        'token = "secret"\n'
        'principalId = "worker"\n'
        f"{kind_line}{extra}"
        "[tokens.grants]\n"
        'workspace = ["jobs:read"]\n',
        encoding="utf-8",
    )


def test_token_kind_parsing_defaults_to_human_and_remains_strict(tmp_path: Path) -> None:
    path = tmp_path / "auth.toml"
    _write_auth(path)
    human_authenticator = load_authenticator(path)
    _write_auth(path, kind="agent")
    agent_authenticator = load_authenticator(path)

    async def principals() -> tuple[Principal | None, Principal | None]:
        return (
            await human_authenticator.authenticate("secret"),
            await agent_authenticator.authenticate("secret"),
        )

    human, agent = asyncio.run(principals())
    assert human is not None and human.kind == "human"
    assert agent is not None and agent.kind == "agent"

    _write_auth(path, kind="service")
    with pytest.raises(AuthError, match="kind"):
        load_authenticator(path)
    _write_auth(path, extra="unexpected = true\n")
    with pytest.raises(AuthError, match="optional 'kind'"):
        load_authenticator(path)

    path.write_text(
        """version = 1
[[tokens]]
token = "first"
principalId = "worker"
kind = "human"
[tokens.grants]
workspace = ["jobs:read"]
[[tokens]]
token = "second"
principalId = "worker"
kind = "agent"
[tokens.grants]
workspace = ["jobs:read"]
""",
        encoding="utf-8",
    )
    with pytest.raises(AuthError, match="must match other tokens"):
        load_authenticator(path)


def test_agent_mask_defaults_intersect_token_grants_and_humans_are_unmasked() -> None:
    store = PrincipalPermissionStore()
    agent = Principal(
        "worker",
        {
            "workspace": frozenset(
                {"jobs:submit", "settings:write", "queue:control", "principals:manage"}
            )
        },
        kind="agent",
    )
    masked = store.apply(agent)
    assert masked.allows("jobs:submit")
    assert masked.allows_in("workspace", "jobs:submit")
    assert masked.scopes_for("jobs:submit") == frozenset({"workspace"})
    assert not masked.allows("settings:write")
    assert not masked.allows_in("workspace", "settings:write")
    assert masked.scopes_for("settings:write") == frozenset()
    assert not masked.allows("queue:control")
    assert not masked.allows("principals:manage")

    store.update("worker", {"settings": True, "queue": True})
    enabled = store.apply(agent)
    assert enabled.allows("settings:write")
    assert enabled.allows("queue:control")
    assert not enabled.allows("assets:write")

    human = Principal("human", agent.grants)
    assert store.apply(human) is human
    assert human.allows("settings:write")
    assert human.allows("queue:control")


def test_permission_management_and_runtime_toggle_flow() -> None:
    async def scenario() -> None:
        authenticator = _authenticator()
        store = PrincipalPermissionStore()
        app = web.Application()
        install_auth(app, authenticator, permission_store=store)
        add_principal_routes(app, authenticator, store)

        async def write_setting(_: web.Request) -> web.Response:
            return web.json_response({"updated": True})

        app.router.add_put("/api/settings/probe", write_setting)
        client = TestClient(TestServer(app))
        await client.start_server()
        agent_headers = {"Authorization": "Bearer agent-token"}
        operator_headers = {"Authorization": "Bearer operator-token"}
        reader_headers = {"Authorization": "Bearer reader-token"}
        try:
            assert (await client.put("/api/settings/probe", headers=agent_headers)).status == 403
            denied = await client.get("/api/principals", headers=reader_headers)
            assert denied.status == 200
            assert [p["principalId"] for p in await denied.json()] == ["reader"]

            listed_response = await client.get("/api/principals", headers=operator_headers)
            assert listed_response.status == 200
            listed = await listed_response.json()
            assert listed == [
                {
                    "principalId": "operator",
                    "kind": "human",
                    "local": False,
                    "scopes": ["workspace"],
                    "self": True,
                    "categories": dict(AGENT_PERMISSION_DEFAULTS),
                },
                {
                    "principalId": "reader",
                    "kind": "human",
                    "local": False,
                    "scopes": ["workspace"],
                    "self": False,
                    "categories": dict(AGENT_PERMISSION_DEFAULTS),
                },
                {
                    "principalId": "worker",
                    "kind": "agent",
                    "local": False,
                    "scopes": ["workspace"],
                    "self": False,
                    "categories": dict(AGENT_PERMISSION_DEFAULTS),
                },
            ]
            assert "token" not in str(listed).lower()

            unknown_category = await client.put(
                "/api/principals/worker/permissions",
                json={"shell": True},
                headers=operator_headers,
            )
            assert unknown_category.status == 400
            assert (
                await client.put(
                    "/api/principals/operator/permissions",
                    json={"settings": True},
                    headers=operator_headers,
                )
            ).status == 200
            assert (
                await client.put(
                    "/api/principals/missing/permissions",
                    json={"settings": True},
                    headers=operator_headers,
                )
            ).status == 404

            enabled = await client.put(
                "/api/principals/worker/permissions",
                json={"settings": True},
                headers=operator_headers,
            )
            assert enabled.status == 200
            assert (await client.put("/api/settings/probe", headers=agent_headers)).status == 200
            disabled = await client.put(
                "/api/principals/worker/permissions",
                json={"settings": False},
                headers=operator_headers,
            )
            assert disabled.status == 200
            assert (await client.put("/api/settings/probe", headers=agent_headers)).status == 403
        finally:
            await client.close()
            store.close()

    asyncio.run(scenario())


def test_permission_store_persists_explicit_values_and_memory_matches(tmp_path: Path) -> None:
    path = tmp_path / "principals.sqlite"
    first = PrincipalPermissionStore(path)
    first.update("worker", {"settings": True, "read": False})
    first.close()

    reopened = PrincipalPermissionStore(path)
    expected = dict(AGENT_PERMISSION_DEFAULTS)
    expected.update({"settings": True, "read": False})
    assert reopened.categories("worker") == expected
    reopened.close()

    memory = PrincipalPermissionStore()
    memory.update("worker", {"settings": True, "read": False})
    assert memory.categories("worker") == expected
    memory.close()


def test_websocket_ticket_applies_current_mask_to_original_token_grants() -> None:
    async def scenario() -> None:
        authenticator = _authenticator(include_limited=True)
        store = PrincipalPermissionStore()
        app = web.Application()
        install_auth(app, authenticator, permission_store=store)

        async def websocket(request: web.Request) -> web.WebSocketResponse:
            response = web.WebSocketResponse()
            await response.prepare(request)
            await response.close()
            return response

        app.router.add_post("/api/auth/ws-ticket", handle_ws_ticket)
        app.router.add_get("/api/events", websocket)
        client = TestClient(TestServer(app))
        await client.start_server()
        agent_headers = {"Authorization": "Bearer agent-token"}
        limited_headers = {"Authorization": "Bearer limited-token"}
        try:
            enabled_ticket = await (
                await client.post("/api/auth/ws-ticket", headers=agent_headers)
            ).json()
            store.update("worker", {"read": False})
            with pytest.raises(aiohttp.WSServerHandshakeError) as disabled:
                await client.ws_connect(f"/api/events?ticket={enabled_ticket['ticket']}")
            assert disabled.value.status == 403

            disabled_ticket = await (
                await client.post("/api/auth/ws-ticket", headers=agent_headers)
            ).json()
            store.update("worker", {"read": True})
            websocket_response = await client.ws_connect(
                f"/api/events?ticket={disabled_ticket['ticket']}"
            )
            await websocket_response.close()

            limited_ticket = await (
                await client.post("/api/auth/ws-ticket", headers=limited_headers)
            ).json()
            store.update("limited", {"read": True})
            with pytest.raises(aiohttp.WSServerHandshakeError) as absent_grant:
                await client.ws_connect(f"/api/events?ticket={limited_ticket['ticket']}")
            assert absent_grant.value.status == 403
        finally:
            await client.close()
            store.close()

    asyncio.run(scenario())
