"""dinkster-server: queue policy above the engine plus the native HTTP/WS
protocol (DESIGN 3.5, 3.10). Job identity is (clientId, jobId); events are
typed; node progress is a state map, never a single cursor."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from dinkster_caches import MemoryLRUCache
from dinkster_collab import SessionService, add_session_routes
from dinkster_engine import (
    CompiledGraph,
    Engine,
    EngineEvent,
    EventListener,
    ExecutionRuntime,
)
from dinkster_graph import Graph, GraphNode, Link, graph_to_wire
from dinkster_memory import (
    ConsumerItem,
    MemoryGovernor,
    PageMap,
    PressureSignal,
    Shedder,
)
from dinkster_protocol import (
    GRAPH_COMPILERS_SURFACE,
    GraphCompilerRegistrySnapshot,
    KeyedContribution,
    PackSettingsSchema,
)
from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
    report_preview,
    report_progress,
)
from dinkster_server import (
    BINARY_BLOB_KEY,
    STATE_KEY,
    Job,
    PackFrontendAsset,
    PackInfo,
    Principal,
    create_app,
    principal_for,
    resolve_scope,
)
from dinkster_values import (
    CURVE_TYPE,
    PNG_CONTAINER_VERSION,
    Curve,
    TypeRegistry,
    register_core_types,
    register_curve_type,
)
from dinkster_values.model import PyObjPayload, Value, ValueMeta
from dinkster_workers import InProcessWorker
from scaffold_nodes import (
    SCAFFOLD_NODES,
    register_scaffold_types,
    scaffold_choices,
    scaffold_lazy_choices,
)

from tests.async_scenarios import run_scenario

STRING = TypeExpr.concrete("STRING")
FLOAT = TypeExpr.concrete("FLOAT")


class Echo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.echo",
            inputs=(InputSpec("text", STRING),),
            outputs=(OutputSpec("out", STRING),),
        )

    @classmethod
    async def execute(cls, *, text: str) -> Mapping[str, object]:
        return cls.outputs(out=text)


class Shout(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.shout",
            inputs=(InputSpec("text", STRING),),
            outputs=(OutputSpec("out", STRING),),
        )

    @classmethod
    async def execute(cls, *, text: str) -> Mapping[str, object]:
        return cls.outputs(out=text.upper())


class Sleeper(Node):
    """Waits on a class-level gate so tests control when jobs finish."""

    gate: asyncio.Event
    entered: asyncio.Event

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.sleeper",
            inputs=(InputSpec("tag", STRING),),
            outputs=(OutputSpec("out", STRING),),
            idempotent=False,  # never coalesced/cached across tests
        )

    @classmethod
    async def execute(cls, *, tag: str) -> Mapping[str, object]:
        Sleeper.entered.set()
        await Sleeper.gate.wait()
        return cls.outputs(out=tag)


class Boom(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.boom",
            inputs=(InputSpec("tag", STRING),),
            outputs=(OutputSpec("out", STRING),),
        )

    @classmethod
    async def execute(cls, *, tag: str) -> Mapping[str, object]:
        raise ValueError(f"boom: {tag}")


class ShapeBoom(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.shape_boom",
            inputs=(InputSpec("layout", STRING),),
            outputs=(OutputSpec("out", STRING),),
        )

    @classmethod
    async def execute(cls, *, layout: str) -> Mapping[str, object]:
        if layout == "permutation":
            raise RuntimeError(
                "size mismatch for input: got torch.Size([1, 64, 64, 4]) "
                "and expected torch.Size([1, 4, 64, 64])"
            )
        raise RuntimeError(
            "The size of tensor a (64) must match the size of tensor b (32) "
            "at non-singleton dimension 2"
        )


class Previewer(Node):
    """Reports progress and a binary preview while running (DESIGN 3.5)."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.previewer",
            inputs=(InputSpec("text", STRING),),
            outputs=(OutputSpec("out", STRING),),
            idempotent=False,  # never cached: events must fire every run
        )

    @classmethod
    async def execute(cls, *, text: str) -> Mapping[str, object]:
        report_progress(1, 2, text="rendering")
        report_preview(b"\x89png-bytes", mime="image/png", width=2, height=2)
        return cls.outputs(out=text)


class Splitter(Node):
    """List output: descriptors must carry runtime length (DESIGN 3.13)."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.splitter",
            inputs=(InputSpec("text", STRING),),
            outputs=(OutputSpec("words", TypeExpr.list_of(STRING)),),
        )

    @classmethod
    async def execute(cls, *, text: str) -> Mapping[str, object]:
        return cls.outputs(words=text.split())


NODES: list[type[Node]] = [Echo, Shout, Sleeper, Boom, ShapeBoom, Previewer, Splitter]
SCHEMAS = build_schemas(NODES)


def make_engine(on_event: EventListener | None = None) -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    return Engine(
        schemas=SCHEMAS,
        registry=registry,
        worker=InProcessWorker(build_node_types(NODES), registry),
        cache=MemoryLRUCache(),
        on_event=on_event,
    )


def compiler_runtime(engine: Engine) -> ExecutionRuntime:
    async def unused_transport(*_args: object) -> Mapping[str, object]:
        raise AssertionError("tests replace compile_for_execution before transport")

    contribution = KeyedContribution(
        surface_id=GRAPH_COMPILERS_SURFACE,
        id="test.compiler",
        behavior_metadata=(("contractVersion", 1), ("order", 0)),
    )
    return replace(
        engine.pin_execution(),
        graph_compiler_registry=GraphCompilerRegistrySnapshot((contribution,)),
        graph_compile_transport=unused_transport,
    )


def generated_echo_compiled(runtime: ExecutionRuntime, text: str = "compiled") -> CompiledGraph:
    node_id = "$gen-test-compiled"
    return CompiledGraph(
        Graph({node_id: GraphNode("test.echo", {"text": text})}),
        (node_id,),
        runtime.extension_snapshot_digest,
        {
            node_id: {
                "nodeId": node_id,
                "compilerId": "test.compiler",
                "passIndex": 0,
                "sources": ("e",),
                "localKey": "compiled",
            }
        },
    )


def echo_graph(text: str = "hello") -> Graph:
    return Graph(
        nodes={
            "e": GraphNode("test.echo", {"text": text}),
            "s": GraphNode("test.shout", {"text": Link("e", "out")}),
        }
    )


def reset_sleeper() -> None:
    Sleeper.gate = asyncio.Event()
    Sleeper.entered = asyncio.Event()


async def wait_for_state(job: Job, *states: str, timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while job.state not in states:
            await asyncio.sleep(0.005)


# -- queue policy (no HTTP) ---------------------------------------------------


# -- HTTP/WS protocol ---------------------------------------------------------


async def make_client() -> TestClient:
    app = create_app(make_engine, SCHEMAS)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


class StubAuthenticator:
    def __init__(self, principal: Principal) -> None:
        self.principal = principal
        self.tokens: list[str] = []

    async def authenticate(self, token: str) -> Principal | None:
        self.tokens.append(token)
        return self.principal if token == "accepted" else None


class MappingAuthenticator:
    def __init__(self, principals: Mapping[str, Principal]) -> None:
        self.principals = dict(principals)

    async def authenticate(self, token: str) -> Principal | None:
        return self.principals.get(token)


def test_authentication_failures_are_structured_and_never_echo_tokens() -> None:
    async def scenario() -> None:
        authenticator = StubAuthenticator(Principal("alice", {"scope": frozenset()}))
        app = create_app(make_engine, SCHEMAS, authenticator=authenticator)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            headers_cases = (
                {},
                {"Authorization": "Basic opaque"},
                {"Authorization": "Bearer unknown-secret"},
                {"Authorization": f"Bearer nonascii-{chr(233)}"},
            )
            for headers in headers_cases:
                response = await client.get("/api/nodes", headers=headers)
                assert response.status == 401
                assert response.headers["WWW-Authenticate"] == "Bearer"
                body = await response.json()
                assert body == {
                    "error": "authentication-required",
                    "message": "a valid Bearer credential is required",
                }
                assert "unknown-secret" not in await response.text()
        finally:
            await client.close()

    run_scenario(scenario())


def test_missing_capability_is_403_for_every_route_family() -> None:
    routes = (
        ("POST", "/api/jobs", "jobs:submit"),
        ("GET", "/api/jobs", "jobs:read"),
        ("DELETE", "/api/jobs/client/job", "jobs:cancel"),
        ("GET", "/api/history", "history:read"),
        ("GET", "/api/assets/digest", "assets:read"),
        ("POST", "/api/assets", "assets:write"),
        ("GET", "/api/settings", "settings:read"),
        ("PUT", "/api/settings/jobs", "settings:write"),
        ("PUT", "/api/packs/missing/settings", "settings:write"),
        ("GET", "/api/sessions", "sessions:read"),
        ("POST", "/api/sessions", "sessions:write"),
        ("GET", "/api/queue", "queue:control"),
        ("GET", "/memory/status", "memory:read"),
        ("GET", "/cache/entry/key", "cache:read"),
    )

    async def scenario() -> None:
        authenticator = StubAuthenticator(Principal("reader", {"scope": frozenset()}))
        app = create_app(make_engine, SCHEMAS, authenticator=authenticator)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            for method, path, capability in routes:
                response = await client.request(
                    method, path, headers={"Authorization": "Bearer accepted"}
                )
                assert response.status == 403, (method, path, await response.text())
                assert await response.json() == {
                    "error": "capability-required",
                    "capability": capability,
                    "message": f"this route requires {capability}",
                }
        finally:
            await client.close()

    run_scenario(scenario())


def test_generation_routes_require_their_declared_capabilities() -> None:
    async def scenario() -> None:
        authenticator = StubAuthenticator(Principal("reader", {"scope": frozenset()}))
        app = create_app(make_engine, SCHEMAS, authenticator=authenticator)

        async def ok(_: web.Request) -> web.Response:
            return web.Response(status=204)

        app.router.add_get("/api/generation/models", ok)
        app.router.add_get("/v1/models", ok)
        protected = (
            ("POST", "/api/generation", "jobs:submit"),
            ("POST", "/api/generation/models/load", "settings:write"),
            ("POST", "/api/generation/models/unload", "settings:write"),
            ("DELETE", "/api/generation/sessions/session", "sessions:write"),
            ("POST", "/v1/completions", "jobs:submit"),
            ("POST", "/v1/chat/completions", "jobs:submit"),
            ("POST", "/v1/responses", "jobs:submit"),
        )
        for method, path, _ in protected:
            app.router.add_route(method, path, ok)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            headers = {"Authorization": "Bearer accepted"}
            assert (await client.get("/api/generation/models", headers=headers)).status == 204
            assert (await client.get("/v1/models", headers=headers)).status == 204
            assert (await client.get("/v1/models")).status == 401
            for method, path, capability in protected:
                response = await client.request(method, path, headers=headers)
                assert response.status == 403
                assert (await response.json())["capability"] == capability
        finally:
            await client.close()

    run_scenario(scenario())


def test_late_mounted_session_routes_use_read_and_write_capabilities() -> None:
    async def scenario() -> None:
        authenticator = StubAuthenticator(
            Principal("reader", {"scope": frozenset({"sessions:read"})})
        )
        app = create_app(make_engine, SCHEMAS, authenticator=authenticator)
        add_session_routes(
            app,
            SessionService(),
            principal_for=principal_for,
            resolve_scope=resolve_scope,
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            headers = {"Authorization": "Bearer accepted"}
            assert (await client.get("/api/sessions?scope=local", headers=headers)).status == 200
            assert (await client.head("/api/sessions?scope=local", headers=headers)).status == 200
            response = await client.post("/api/sessions", json={}, headers=headers)
            assert response.status == 403
            assert (await response.json())["capability"] == "sessions:write"
        finally:
            await client.close()

    run_scenario(scenario())


def test_websocket_authentication_is_rejected_before_upgrade() -> None:
    async def scenario() -> None:
        authenticator = StubAuthenticator(Principal("reader", {"scope": frozenset({"jobs:read"})}))
        app = create_app(make_engine, SCHEMAS, authenticator=authenticator)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            with pytest.raises(aiohttp.WSServerHandshakeError) as missing:
                await client.ws_connect("/api/events")
            assert missing.value.status == 401
            ws = await client.ws_connect(
                "/api/events", headers={"Authorization": "Bearer accepted"}
            )
            assert not ws.closed
            await ws.close()
        finally:
            await client.close()

    run_scenario(scenario())


def test_ws_ticket_is_single_use_expiring_and_never_authenticates_plain_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        principal = Principal("alice", {"a": frozenset({"jobs:read"})})
        authenticator = MappingAuthenticator({"alice-token": principal})
        app = create_app(make_engine, SCHEMAS, authenticator=authenticator)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            headers = {"Authorization": "Bearer alice-token"}
            minted = await client.post("/api/auth/ws-ticket", headers=headers)
            assert minted.status == 200
            payload = await minted.json()
            assert payload["expiresInSeconds"] == 30
            ticket = payload["ticket"]

            # Tickets are ignored by plain routes and are not consumed there.
            assert (await client.get(f"/api/nodes?ticket={ticket}")).status == 401
            ws = await client.ws_connect(f"/api/events?ticket={ticket}")
            await ws.close()
            with pytest.raises(aiohttp.WSServerHandshakeError) as reused:
                await client.ws_connect(f"/api/events?ticket={ticket}")
            assert reused.value.status == 401
            with pytest.raises(aiohttp.WSServerHandshakeError) as unknown:
                await client.ws_connect("/api/events?ticket=unknown")
            assert unknown.value.status == 401

            near_miss_ticket = await (
                await client.post("/api/auth/ws-ticket", headers=headers)
            ).json()
            with pytest.raises(aiohttp.WSServerHandshakeError) as near_miss:
                await client.ws_connect(
                    f"/api/sessions/a/extra/events?ticket={near_miss_ticket['ticket']}"
                )
            assert near_miss.value.status == 401
            ws = await client.ws_connect(f"/api/events?ticket={near_miss_ticket['ticket']}")
            await ws.close()

            monkeypatch.setattr("dinkster_server.auth.time.monotonic", lambda: 100.0)
            expiring = await (await client.post("/api/auth/ws-ticket", headers=headers)).json()
            monkeypatch.setattr("dinkster_server.auth.time.monotonic", lambda: 131.0)
            with pytest.raises(aiohttp.WSServerHandshakeError) as expired:
                await client.ws_connect(f"/api/events?ticket={expiring['ticket']}")
            assert expired.value.status == 401
        finally:
            await client.close()

    run_scenario(scenario())


def test_submit_scope_resolution_stamps_jobs_and_lists_readable_scope_union() -> None:
    async def scenario() -> None:
        submitter = Principal(
            "alice",
            {
                "a": frozenset({"jobs:submit", "jobs:read"}),
                "b": frozenset({"jobs:submit"}),
                "c": frozenset({"jobs:read"}),
            },
            kind="agent",
        )
        unique = Principal("unique", {"a": frozenset({"jobs:submit", "jobs:read"})})
        app = create_app(
            make_engine,
            SCHEMAS,
            authenticator=MappingAuthenticator({"token": submitter, "unique": unique}),
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            headers = {"Authorization": "Bearer token"}
            ambiguous = await client.post(
                "/api/jobs", json=submit_body(echo_graph(), ["s"]), headers=headers
            )
            assert ambiguous.status == 400
            assert (await ambiguous.json())["error"] == "scope-required"

            denied = await client.post(
                "/api/jobs",
                json=submit_body(echo_graph(), ["s"], scope="c"),
                headers=headers,
            )
            assert denied.status == 403
            assert await denied.json() == {
                "error": "capability-required",
                "capability": "jobs:submit",
                "scope": "c",
                "message": "scope c requires jobs:submit",
            }

            accepted = await client.post(
                "/api/jobs",
                json=submit_body(echo_graph(), ["s"], scope="a"),
                headers=headers,
            )
            wire = await accepted.json()
            assert accepted.status == 202
            assert wire["scope"] == "a"
            assert wire["submittedBy"] == {"principalId": "alice", "kind": "agent"}
            state = app[STATE_KEY]
            job = state.queue.get("c1", "j1")
            assert job is not None
            assert (job.scope, job.principal_id, job.principal_kind) == ("a", "alice", "agent")
            unique_response = await client.post(
                "/api/jobs",
                json=submit_body(echo_graph("unique"), ["s"], clientId="unique", jobId="j2"),
                headers={"Authorization": "Bearer unique"},
            )
            assert unique_response.status == 202
            assert (await unique_response.json())["scope"] == "a"
            state.queue.submit("c2", "hidden", echo_graph("hidden"), ["s"], scope="b")
            state.queue.submit("c3", "visible", echo_graph("visible"), ["s"], scope="c")
            listed = await (await client.get("/api/jobs", headers=headers)).json()
            assert {(job["clientId"], job["scope"]) for job in listed["jobs"]} == {
                ("c1", "a"),
                ("c3", "c"),
                ("unique", "a"),
            }
        finally:
            await client.close()

    run_scenario(scenario())


def test_cross_scope_job_resources_are_all_hidden_as_existing_404_shapes() -> None:
    async def scenario() -> None:
        principal = Principal("alice", {"a": frozenset({"jobs:read", "jobs:cancel"})})
        app = create_app(
            make_engine,
            SCHEMAS,
            authenticator=MappingAuthenticator({"token": principal}),
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            hidden = app[STATE_KEY].queue.submit(
                "other", "job", echo_graph(), ["s"], scope="b", principal_id="bob"
            )
            headers = {"Authorization": "Bearer token"}
            assert (await client.get("/api/jobs/other/job", headers=headers)).status == 404
            assert (
                await client.get(f"/api/jobs/by-ref/{hidden.run_id}", headers=headers)
            ).status == 404
            assert (
                await client.get(f"/api/jobs/by-ref/{hidden.run_id}/events", headers=headers)
            ).status == 404
            value = await client.get(
                "/api/values?clientId=other&jobId=job&nodeId=s&outputId=out",
                headers=headers,
            )
            assert value.status == 404
            assert (await value.json())["reason"] == "unknown-job"
            assert (await client.delete("/api/jobs/other/job", headers=headers)).status == 404
            assert (
                await client.delete(f"/api/jobs/by-ref/{hidden.run_id}", headers=headers)
            ).status == 404
        finally:
            await client.close()

    run_scenario(scenario())


def test_websocket_events_filter_job_scopes_but_broadcast_non_job_frames() -> None:
    async def receive_probes(
        websocket: aiohttp.ClientWebSocketResponse, own: str, hidden: str
    ) -> list[str]:
        received: list[str] = []
        while len(received) < 2:
            event_type = (await websocket.receive_json())["type"]
            assert event_type != hidden
            if event_type in {own, "global-probe"}:
                received.append(event_type)
        return received

    async def scenario() -> None:
        alice = Principal("alice", {"a": frozenset({"jobs:read"})})
        bob = Principal("bob", {"b": frozenset({"jobs:read"})})
        app = create_app(
            make_engine,
            SCHEMAS,
            authenticator=MappingAuthenticator({"alice": alice, "bob": bob}),
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            job_a = app[STATE_KEY].queue.submit("ca", "ja", echo_graph("a"), ["s"], scope="a")
            job_b = app[STATE_KEY].queue.submit("cb", "jb", echo_graph("b"), ["s"], scope="b")
            ws_a = await client.ws_connect("/api/events", headers={"Authorization": "Bearer alice"})
            ws_b = await client.ws_connect("/api/events", headers={"Authorization": "Bearer bob"})
            app[STATE_KEY].hub.publish(
                {"type": "probe-a", "jobRef": job_a.run_id},
                client_id="ca",
                droppable=False,
            )
            app[STATE_KEY].hub.publish(
                {"type": "probe-b", "jobRef": job_b.run_id},
                client_id="cb",
                droppable=False,
            )
            app[STATE_KEY].hub.publish({"type": "global-probe"}, client_id=None, droppable=False)
            async with asyncio.timeout(2):
                assert await receive_probes(ws_a, "probe-a", "probe-b") == [
                    "probe-a",
                    "global-probe",
                ]
                assert await receive_probes(ws_b, "probe-b", "probe-a") == [
                    "probe-b",
                    "global-probe",
                ]
            await ws_a.close()
            await ws_b.close()
        finally:
            await client.close()

    run_scenario(scenario())


def test_browser_boundary_accepts_loopback_and_rejects_foreign_host() -> None:
    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS, allow_hosts=["ui.example", "::1"])
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            accepted = await client.get("/api/health", headers={"Host": "127.0.0.2:3639"})
            assert accepted.status == 200
            configured = await client.get("/api/health", headers={"Host": "ui.example:3639"})
            assert configured.status == 200

            rejected = await client.get("/api/health", headers={"Host": "attacker.example"})
            assert rejected.status == 421
            assert await rejected.json() == {
                "error": "host-not-allowed",
                "message": "the request Host is not allowed by this server",
            }
        finally:
            await client.close()

    run_scenario(scenario())


def test_browser_boundary_rejects_foreign_origin_on_state_change() -> None:
    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS, allow_origins=["https://ui.example"])
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            rejected = await client.post(
                "/api/jobs",
                json={},
                headers={"Origin": "https://attacker.example"},
            )
            assert rejected.status == 403
            assert await rejected.json() == {
                "error": "origin-not-allowed",
                "message": "the request Origin is not allowed for this operation",
            }

            allowed = await client.post(
                "/api/jobs",
                json={},
                headers={"Origin": "https://ui.example"},
            )
            assert allowed.status == 400
        finally:
            await client.close()

    run_scenario(scenario())


def test_browser_boundary_rejects_websocket_upgrade_with_foreign_host() -> None:
    async def scenario() -> None:
        client = await make_client()
        try:
            with pytest.raises(aiohttp.WSServerHandshakeError) as rejected:
                await client.ws_connect("/api/events", headers={"Host": "attacker.example"})
            assert rejected.value.status == 421
        finally:
            await client.close()

    run_scenario(scenario())


def test_cors_exact_origin() -> None:
    """A configured exact origin is echoed with Vary: Origin and ETag
    exposed; reads from other origins get no CORS response headers."""

    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS, allow_origins=["https://ui.example"])
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            resp = await client.get("/api/health", headers={"Origin": "https://ui.example"})
            assert resp.headers["Access-Control-Allow-Origin"] == "https://ui.example"
            assert resp.headers["Access-Control-Expose-Headers"] == "ETag"
            assert "Origin" in resp.headers.getall("Vary", [])

            resp = await client.get("/api/health", headers={"Origin": "https://other.example"})
            assert resp.status == 200
            assert "Access-Control-Allow-Origin" not in resp.headers

            # No Origin header -> no CORS headers (same-origin/cli traffic).
            resp = await client.get("/api/health")
            assert "Access-Control-Allow-Origin" not in resp.headers

            # Preflight: answered locally, 204, permissions echoed.
            resp = await client.options(
                "/api/jobs",
                headers={
                    "Origin": "https://ui.example",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                },
            )
            assert resp.status == 204
            assert resp.headers["Access-Control-Allow-Origin"] == "https://ui.example"
            assert resp.headers["Access-Control-Allow-Methods"] == "POST"
            assert resp.headers["Access-Control-Allow-Headers"] == "content-type"

            # Preflight from an unconfigured origin is not granted.
            resp = await client.options(
                "/api/jobs",
                headers={
                    "Origin": "https://other.example",
                    "Access-Control-Request-Method": "POST",
                },
            )
            assert "Access-Control-Allow-Origin" not in resp.headers
        finally:
            await client.close()

    run_scenario(scenario())


def submit_body(graph: Graph, targets: list[str], **overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "clientId": "c1",
        "jobId": "j1",
        "graph": graph_to_wire(graph),
        "targets": targets,
    }
    body.update(overrides)
    return body


def test_pack_static_assets_and_settings_end_to_end(tmp_path: Path) -> None:
    schema = PackSettingsSchema.from_wire(
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "enabled": {"type": "boolean", "title": "Enabled", "default": True},
                "strength": {
                    "type": "number",
                    "title": "Strength",
                    "description": "Rendering strength",
                    "default": 0.5,
                    "minimum": 0,
                    "maximum": 1,
                },
            },
            "required": ["enabled", "strength"],
        }
    )
    other_schema = PackSettingsSchema.from_wire(
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {"label": {"type": "string", "title": "Label", "default": "other"}},
            "required": ["label"],
        }
    )
    packs = {
        "visual-pack": PackInfo(
            "Visual Pack",
            frontend_assets=(
                PackFrontendAsset("badge.png", "image/png", b"\x89PNG\r\nfixture"),
                PackFrontendAsset("styles/theme.css", "text/css", b".pack { color: #123; }\n"),
            ),
            settings_schema=schema,
        ),
        "other-pack": PackInfo("Other Pack", settings_schema=other_schema),
    }
    settings_root = tmp_path / "pack-settings"

    async def scenario() -> None:
        app = create_app(
            make_engine,
            SCHEMAS,
            packs=packs,
            pack_settings_root=settings_root,
        )
        async with TestClient(TestServer(app)) as client:
            image = await client.get("/packs/visual-pack/static/badge.png")
            assert image.status == 200
            assert image.headers["Content-Type"] == "image/png"
            assert await image.read() == b"\x89PNG\r\nfixture"
            css = await client.get("/packs/visual-pack/static/styles/theme.css")
            assert css.status == 200
            assert css.headers["Content-Type"].startswith("text/css")
            assert await css.read() == b".pack { color: #123; }\n"
            for path in (
                "/packs/visual-pack/static/missing.css",
                "/packs/visual-pack/static/styles/",
                "/packs/visual-pack/static/%2e%2e/secret",
                "/packs/visual-pack/static/styles/%2e%2e/%2e%2e/secret",
                "/packs/visual-pack/static/styles%5C..%5Csecret",
                "/packs/other-pack/static/badge.png",
            ):
                assert (await client.get(path)).status == 404

            initial = await client.get("/api/packs/visual-pack/settings")
            assert initial.status == 200
            initial_wire = await initial.json()
            assert initial_wire["packId"] == "visual-pack"
            assert initial_wire["displayName"] == "Visual Pack"
            assert initial_wire["schema"] == schema.to_wire()
            assert initial_wire["values"] == {"enabled": True, "strength": 0.5}
            nodes = await (await client.get("/api/nodes")).json()
            assert nodes["packs"]["visual-pack"]["settings"] is True
            assert "settings" not in nodes["packs"]["core"]

            updated = await client.put(
                "/api/packs/visual-pack/settings",
                json={"enabled": False, "strength": 0.75},
            )
            assert updated.status == 200
            assert (await updated.json())["values"] == {"enabled": False, "strength": 0.75}
            invalid = await client.put(
                "/api/packs/visual-pack/settings",
                json={"enabled": False, "strength": 2},
            )
            assert invalid.status == 400
            assert (await (await client.get("/api/packs/visual-pack/settings")).json())[
                "values"
            ] == {"enabled": False, "strength": 0.75}
            assert (await client.get("/api/packs/missing/settings")).status == 404
            assert (await (await client.get("/api/packs/other-pack/settings")).json())[
                "values"
            ] == {"label": "other"}

        restarted = create_app(
            make_engine,
            SCHEMAS,
            packs=packs,
            pack_settings_root=settings_root,
        )
        async with TestClient(TestServer(restarted)) as client:
            persisted = await client.get("/api/packs/visual-pack/settings")
            assert (await persisted.json())["values"] == {
                "enabled": False,
                "strength": 0.75,
            }
            assert (await (await client.get("/api/packs/other-pack/settings")).json())[
                "values"
            ] == {"label": "other"}

    run_scenario(scenario())


def test_job_lifecycle_over_http() -> None:
    async def scenario() -> None:
        client = await make_client()
        try:
            resp = await client.post("/api/jobs", json=submit_body(echo_graph(), ["s"]))
            assert resp.status == 202
            accepted = await resp.json()
            assert accepted["state"] == "queued"
            assert accepted["clientId"] == "c1"
            assert accepted["jobId"] == "j1"
            assert accepted["attemptId"] == 1
            assert accepted["latestSeq"] == 1
            nodes = await (await client.get("/api/nodes")).json()
            assert accepted["extensionSnapshotDigest"] == nodes["extensionSnapshotDigest"]
            # jobRef: server-assigned global job reference (platform plan),
            # same identity as the legacy runId alias.
            first_ref = accepted["jobRef"]
            assert first_ref and first_ref == accepted["runId"]

            async with asyncio.timeout(5):
                while True:
                    status = await (await client.get("/api/jobs/c1/j1")).json()
                    if status["state"] in ("completed", "failed", "cancelled"):
                        break
                    await asyncio.sleep(0.01)
            assert status["state"] == "completed"
            assert status["extensionSnapshotDigest"] == accepted["extensionSnapshotDigest"]
            # Results are value descriptors: envelope facts, not raw payloads.
            out = status["outputs"]["s"]["out"]
            assert out["typeId"] == "STRING"
            assert out["fingerprint"]
            # Node progress is a state map.
            assert status["nodeStates"] == {"e": "completed", "s": "completed"}

            # Duplicate active/terminal identity handling: resubmitting a
            # finished job key is allowed (a rerun), an active one conflicts.
            resp = await client.post("/api/jobs", json=submit_body(echo_graph(), ["s"]))
            assert resp.status == 202
            # A rerun is a new execution: same (clientId, jobId), new jobRef.
            assert (await resp.json())["jobRef"] != first_ref
        finally:
            await client.close()

    run_scenario(scenario())


def test_no_cache_submission_reexecutes_with_warm_runtime_and_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS)
        client = TestClient(TestServer(app))
        await client.start_server()
        engine = app[STATE_KEY].engine
        worker = engine._worker
        invocations: list[tuple[int, int]] = []
        invoke = worker.invoke

        async def recording_invoke(invocation, on_event=None):
            invocations.append((os.getpid(), id(worker)))
            return await invoke(invocation, on_event)

        monkeypatch.setattr(worker, "invoke", recording_invoke)
        try:
            results = []
            for job_id, cache_enabled in (
                ("warmup", True),
                ("no-cache-1", False),
                ("no-cache-2", False),
                ("warm", True),
            ):
                response = await client.post(
                    "/api/jobs",
                    json=submit_body(
                        echo_graph("fixed-seed-output"),
                        ["s"],
                        jobId=job_id,
                        cacheEnabled=cache_enabled,
                    ),
                )
                assert response.status == 202
                submitted = await response.json()
                assert submitted["cacheEnabled"] is cache_enabled
                job = app[STATE_KEY].queue.get("c1", job_id)
                assert job is not None
                await wait_for_state(job, "completed")
                assert job.result is not None
                results.append(job.result)

            warmup, first, second, warm = results
            assert set(warmup.executed) == {"e", "s"}
            assert set(first.executed) == {"e", "s"}
            assert set(second.executed) == {"e", "s"}
            assert first.cached == second.cached == ()
            assert warm.executed == ()
            assert set(warm.cached) == {"e", "s"}
            assert [result.outputs["s"]["out"].resolve() for result in results] == [
                "FIXED-SEED-OUTPUT"
            ] * 4
            assert len(invocations) == 6
            assert {pid for pid, _worker_id in invocations} == {os.getpid()}
            assert {worker_id for _pid, worker_id in invocations} == {id(worker)}
            assert app[STATE_KEY].engine is engine
            assert engine._worker is worker
        finally:
            await client.close()

    run_scenario(scenario())


def test_active_submit_idempotency_over_http() -> None:
    async def scenario() -> None:
        reset_sleeper()
        client = await make_client()
        try:
            assert (await client.post("/api/queue/pause")).status == 200
            graph = Graph(nodes={"n": GraphNode("test.sleeper", {"tag": "x"})})
            first_response = await client.post("/api/jobs", json=submit_body(graph, ["n"]))
            assert first_response.status == 202
            first = await first_response.json()
            duplicate_response = await client.post("/api/jobs", json=submit_body(graph, ["n"]))
            assert duplicate_response.status == 202
            duplicate = await duplicate_response.json()
            assert duplicate["duplicate"] is True
            assert duplicate["jobRef"] == first["jobRef"]
            assert duplicate["latestSeq"] == first["latestSeq"] == 1
            replay = await (await client.get(f"/api/jobs/by-ref/{first['jobRef']}/events")).json()
            assert [event["state"] for event in replay["events"]] == ["queued"]

            different_submissions = (
                submit_body(graph, ["n"], priority=1),
                submit_body(graph, ["other"]),
                submit_body(
                    Graph(nodes={"n": GraphNode("test.sleeper", {"tag": "y"})}),
                    ["n"],
                ),
                submit_body(
                    graph,
                    ["n"],
                    sourceDocument="blake3:" + "ab" * 32,
                ),
            )
            for different in different_submissions:
                conflict_response = await client.post("/api/jobs", json=different)
                assert conflict_response.status == 409
                conflict = await conflict_response.json()
                assert conflict == {
                    "error": "job-key-in-use",
                    "message": ("job key c1/j1 is in use by queued job with different content"),
                    "clientId": "c1",
                    "jobId": "j1",
                }
            # Cancel over HTTP; job reaches a terminal state.
            resp = await client.delete("/api/jobs/c1/j1")
            assert resp.status == 200
            async with asyncio.timeout(5):
                while True:
                    status = await (await client.get("/api/jobs/c1/j1")).json()
                    if status["state"] == "cancelled":
                        break
                    await asyncio.sleep(0.01)
        finally:
            await client.close()

    run_scenario(scenario())


def test_job_event_sequences_replay_and_by_ref_routes() -> None:
    async def scenario() -> None:
        client = await make_client()
        try:
            first_response = await client.post(
                "/api/jobs", json=submit_body(echo_graph("one"), ["s"])
            )
            second_response = await client.post(
                "/api/jobs",
                json=submit_body(echo_graph("two"), ["s"], clientId="c2", jobId="j2"),
            )
            first = await first_response.json()
            second = await second_response.json()
            refs = (first["jobRef"], second["jobRef"])

            for job_ref in refs:
                async with asyncio.timeout(5):
                    while True:
                        response = await client.get(f"/api/jobs/by-ref/{job_ref}")
                        assert response.status == 200
                        status = await response.json()
                        if status["state"] == "completed":
                            break
                        await asyncio.sleep(0.01)
                keyed = await (
                    await client.get(f"/api/jobs/{status['clientId']}/{status['jobId']}")
                ).json()
                assert status == keyed
                assert status["attemptId"] == 1
                replay_response = await client.get(f"/api/jobs/by-ref/{job_ref}/events")
                assert replay_response.status == 200
                replay = await replay_response.json()
                sequences = [event["seq"] for event in replay["events"]]
                assert sequences == list(range(1, replay["latestSeq"] + 1))
                assert replay["latestSeq"] == status["latestSeq"]
                assert all(event["jobRef"] == job_ref for event in replay["events"])
                job_states = [event for event in replay["events"] if event["type"] == "job_state"]
                assert all(event["attemptId"] == 1 for event in job_states)
                after = sequences[len(sequences) // 2]
                resumed = await (
                    await client.get(f"/api/jobs/by-ref/{job_ref}/events?after={after}")
                ).json()
                assert [event["seq"] for event in resumed["events"]] == [
                    seq for seq in sequences if seq > after
                ]
                assert resumed["latestSeq"] == replay["latestSeq"]

            for bad in ("-1", "nope", "1.0", "+1"):
                response = await client.get(f"/api/jobs/by-ref/{refs[0]}/events?after={bad}")
                assert response.status == 400
            assert (await client.get("/api/jobs/by-ref/missing")).status == 404
            assert (await client.get("/api/jobs/by-ref/missing/events")).status == 404
            assert (await client.delete("/api/jobs/by-ref/missing")).status == 404
        finally:
            await client.close()

    run_scenario(scenario())


def test_cancel_by_ref_matches_keyed_cancel() -> None:
    async def scenario() -> None:
        reset_sleeper()
        client = await make_client()
        try:
            assert (await client.post("/api/queue/pause")).status == 200
            graph = Graph(nodes={"n": GraphNode("test.sleeper", {"tag": "x"})})
            accepted = await (await client.post("/api/jobs", json=submit_body(graph, ["n"]))).json()
            response = await client.delete(f"/api/jobs/by-ref/{accepted['jobRef']}")
            assert response.status == 200
            by_ref = await response.json()
            keyed = await (await client.get("/api/jobs/c1/j1")).json()
            assert by_ref == keyed
            assert by_ref["state"] == "cancelled"
        finally:
            await client.close()

    run_scenario(scenario())


def test_progress_events_throttled_at_ingestion_final_survives_in_replay() -> None:
    # Feeds synthetic progress node_events straight into the engine-event
    # listener: the throttle must collapse the flood before the replay
    # buffer sees it, while the first and final updates always survive.
    # (Replay-buffer pressure itself is covered by the _JobReplay units.)
    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            state = app[STATE_KEY]
            accepted = await (
                await client.post("/api/jobs", json=submit_body(echo_graph(), ["s"]))
            ).json()
            run_ref = accepted["jobRef"]
            async with asyncio.timeout(5):
                while (await (await client.get("/api/jobs/c1/j1")).json())["state"] != "completed":
                    await asyncio.sleep(0.01)
            for step in range(1, 2001):
                state._on_engine_event(
                    EngineEvent(
                        "node_event",
                        run_ref,
                        "s",
                        {"name": "progress", "data": {"step": step, "total": 2000}},
                    )
                )
            response = await client.get(f"/api/jobs/by-ref/{run_ref}/events?after=0")
            assert response.status == 200
            events = (await response.json())["events"]
            progress = [event for event in events if event.get("event") == "progress"]
            # The tight flood collapses to a handful of kept updates, and the
            # final one survives in the replay buffer.
            assert progress
            assert progress[0]["data"]["step"] == 1
            assert progress[-1]["data"]["step"] == 2000
            assert len(progress) < 50
        finally:
            await client.close()

    run_scenario(scenario())


def test_queue_endpoints_over_http(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        reset_sleeper()
        client = await make_client()
        queue = client.app[STATE_KEY].queue
        dispatch_checked = asyncio.Event()
        original_clear = queue._wakeup.clear

        def clear_wakeup() -> None:
            original_clear()
            dispatch_checked.set()

        monkeypatch.setattr(queue._wakeup, "clear", clear_wakeup)
        try:
            # Baseline shape: nothing queued, nothing running, not paused.
            status = await (await client.get("/api/queue")).json()
            assert status == {
                "queued": [],
                "running": [],
                "maxRunningJobs": 1,
                "paused": False,
            }

            assert (await client.post("/api/queue/pause")).status == 200
            dispatch_checked.clear()
            graph = Graph(nodes={"n": GraphNode("test.sleeper", {"tag": "x"})})
            assert (await client.post("/api/jobs", json=submit_body(graph, ["n"]))).status == 202
            await asyncio.wait_for(dispatch_checked.wait(), 5)
            status = await (await client.get("/api/queue")).json()
            assert status["paused"] is True
            assert [j["jobId"] for j in status["queued"]] == ["j1"]
            assert status["running"] == []

            assert (await client.post("/api/queue/resume")).status == 200
            await Sleeper.entered.wait()
            status = await (await client.get("/api/queue")).json()
            assert status["paused"] is False
            assert status["queued"] == []
            assert [j["jobId"] for j in status["running"]] == ["j1"]

            Sleeper.gate.set()
            async with asyncio.timeout(5):
                while (await (await client.get("/api/jobs/c1/j1")).json())["state"] != "completed":
                    await asyncio.sleep(0.01)

            # Job listing: every client's, then filtered to one.
            body = submit_body(echo_graph(), ["s"], clientId="c2", jobId="other")
            assert (await client.post("/api/jobs", json=body)).status == 202
            jobs = (await (await client.get("/api/jobs")).json())["jobs"]
            assert {(j["clientId"], j["jobId"]) for j in jobs} == {("c1", "j1"), ("c2", "other")}
            jobs = (await (await client.get("/api/jobs?clientId=c2")).json())["jobs"]
            assert [(j["clientId"], j["jobId"]) for j in jobs] == [("c2", "other")]
        finally:
            await client.close()

    run_scenario(scenario())


def test_queue_clear_over_http_and_queue_state_broadcast() -> None:
    async def scenario() -> None:
        reset_sleeper()
        client = await make_client()
        try:
            async with client.ws_connect("/api/events?clientId=someone-else") as ws:
                # Occupy the single slot, then queue two more jobs.
                graph = Graph(nodes={"n": GraphNode("test.sleeper", {"tag": "x"})})
                assert (
                    await client.post("/api/jobs", json=submit_body(graph, ["n"]))
                ).status == 202
                await Sleeper.entered.wait()
                for job_id in ("q1", "q2"):
                    body = submit_body(echo_graph(), ["s"], jobId=job_id)
                    assert (await client.post("/api/jobs", json=body)).status == 202

                resp = await client.post("/api/queue/clear", json={"clientId": "c1"})
                assert resp.status == 200
                cleared = (await resp.json())["cleared"]
                assert {c["jobId"] for c in cleared} == {"q1", "q2"}
                # The running job survived the clear.
                status = await (await client.get("/api/queue")).json()
                assert [j["jobId"] for j in status["running"]] == ["j1"]

                # Pause/resume state changes are instance-wide broadcasts:
                # even a subscriber filtered to an unrelated client sees them.
                assert (await client.post("/api/queue/pause")).status == 200
                async with asyncio.timeout(5):
                    while True:
                        event = json.loads((await ws.receive()).data)
                        if event["type"] == "queue_state":
                            break
                assert event["paused"] is True
                assert (await client.post("/api/queue/resume")).status == 200

                # Validation: clientId must be a non-empty string when given.
                resp = await client.post("/api/queue/clear", json={"clientId": 42})
                assert resp.status == 400

                Sleeper.gate.set()
        finally:
            await client.close()

    run_scenario(scenario())


@pytest.mark.parametrize("with_library", (False, True))
def test_node_events_over_websocket_with_binary_preview(tmp_path: Path, with_library: bool) -> None:
    """A reporting node's progress arrives as a typed JSON node_event; its
    preview arrives as one self-describing binary frame (length-prefixed
    JSON header + raw payload), never base64 in JSON."""
    from dinkster_assets import AssetVault, LibraryStore
    from dinkster_server import ServerLibrary

    async def scenario() -> None:
        library = (
            ServerLibrary(AssetVault(tmp_path / "vault"), LibraryStore(tmp_path / "library.sqlite"))
            if with_library
            else None
        )
        client = TestClient(TestServer(create_app(make_engine, SCHEMAS, library=library)))
        await client.start_server()
        try:
            ws = await client.ws_connect("/api/events?clientId=c1")
            graph = Graph(nodes={"p": GraphNode("test.previewer", {"text": "go"})})
            resp = await client.post("/api/jobs", json=submit_body(graph, ["p"]))
            assert resp.status == 202

            progress: dict[str, object] | None = None
            preview_header: dict[str, object] | None = None
            preview_payload: bytes | None = None
            async with asyncio.timeout(5):
                while True:
                    msg = await ws.receive()
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        raw = msg.data
                        header_len = int.from_bytes(raw[:4], "big")
                        preview_header = json.loads(raw[4 : 4 + header_len])
                        preview_payload = raw[4 + header_len :]
                        continue
                    event = json.loads(msg.data)
                    if event["type"] == "node_event" and event["event"] == "progress":
                        progress = event
                    if event["type"] == "job_state" and event["state"] == "completed":
                        break

            assert progress is not None
            assert progress["nodeId"] == "p"
            assert progress["clientId"] == "c1"
            assert progress["data"] == {"step": 1, "total": 2, "text": "rendering"}
            assert "worker" not in progress
            assert "executionArm" not in progress

            assert preview_header is not None and preview_payload is not None
            assert preview_header["type"] == "node_event"
            assert preview_header["event"] == "preview"
            assert preview_header["nodeId"] == "p"
            assert preview_header["data"] == {
                "mime": "image/png",
                "width": 2,
                "height": 2,
            }
            assert BINARY_BLOB_KEY not in preview_header
            assert "worker" not in preview_header
            assert "executionArm" not in preview_header
            assert preview_payload == b"\x89png-bytes"
            assert preview_header["jobRef"] == event["jobRef"]
            assert preview_header["seq"] < event["seq"]
            replay = await (
                await client.get(f"/api/jobs/by-ref/{event['jobRef']}/events?after=0")
            ).json()
            replayed_preview = next(
                item for item in replay["events"] if item.get("event") == "preview"
            )
            assert replayed_preview == {**preview_header, "blobOmitted": True}
            if library is not None:
                assert library.vault.digests() == []
                response = await client.get(
                    f"/api/library/terminal-results/{event['jobRef']}?scope=local"
                )
                assert response.status == 200
                terminal = await response.json()
                assert terminal["complete"] is True
                assert terminal["artifacts"] == []
                assert set(terminal["resultOffers"]) == {"p"}
                assert set(terminal["resultOffers"]["p"]) == {"out"}
                output_offer = terminal["resultOffers"]["p"]["out"]
                assert output_offer["digests"] == [] and output_offer["complete"] is True
                with library.store.hold_asset_references() as roots:
                    assert roots == set()
            await ws.close()
        finally:
            await client.close()
        assert preview_payload == b"\x89png-bytes"

    run_scenario(scenario())


# -- cross-instance memory coordination (DESIGN 3.10) --------------------------


class TrimmableCache(Shedder):
    """A governed consumer holding sheddable bytes on one device."""

    def __init__(self, device: str, holding: int) -> None:
        self.device = device
        self.holding = holding

    def footprint(self, device: str) -> int:
        return self.holding if device == self.device else 0

    async def shed(self, pressure: PressureSignal) -> int:
        freed = min(self.holding, pressure.bytes_needed)
        self.holding -= freed
        return freed


async def make_governed_client(
    governor: MemoryGovernor,
) -> TestClient:
    app = create_app(make_engine, SCHEMAS, governor=governor)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


def test_reserve_status_renew_release_round_trip() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"vram:cuda:0": 100})
        client = await make_governed_client(governor)
        try:
            resp = await client.post(
                "/memory/reserve",
                json={"device": "vram:cuda:0", "bytes": 60, "ttlSeconds": 30},
            )
            assert resp.status == 201
            lease = await resp.json()
            assert lease["device"] == "vram:cuda:0"
            assert lease["bytes"] == 60
            assert 29 < lease["expiresInSeconds"] <= 30
            lease_id = lease["reservationId"]

            # The lease is visible in status: peers see what is pinned.
            data = await (await client.get("/memory/status")).json()
            assert data["memoryGovernor"]["vram:cuda:0"]["reservedBytes"] == 60
            assert [entry["reservationId"] for entry in data["leases"]] == [lease_id]

            resp = await client.post(f"/memory/reserve/{lease_id}/renew", json={"ttlSeconds": 45})
            assert resp.status == 200
            renewed = await resp.json()
            assert 44 < renewed["expiresInSeconds"] <= 45

            resp = await client.delete(f"/memory/reserve/{lease_id}")
            assert resp.status == 204
            assert governor.reserved("vram:cuda:0") == 0
            # Gone means gone: release and renew are now misses.
            assert (await client.delete(f"/memory/reserve/{lease_id}")).status == 404
            resp = await client.post(f"/memory/reserve/{lease_id}/renew", json={"ttlSeconds": 5})
            assert resp.status == 404
        finally:
            await client.close()

    run_scenario(scenario())


def test_reserve_maps_governor_verdicts_to_507_and_503() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        client = await make_governed_client(governor)
        try:
            # It can never fit: 507 Insufficient Storage.
            resp = await client.post("/memory/reserve", json={"device": "ram", "bytes": 200})
            assert resp.status == 507

            # It did not fit in time: 503, try later.
            first = await (
                await client.post("/memory/reserve", json={"device": "ram", "bytes": 100})
            ).json()
            resp = await client.post(
                "/memory/reserve",
                json={"device": "ram", "bytes": 50, "timeoutSeconds": 0.05},
            )
            assert resp.status == 503
            assert governor.reserved("ram") == 100  # no residue from failures
            await client.delete(f"/memory/reserve/{first['reservationId']}")
        finally:
            await client.close()

    run_scenario(scenario())


def test_lease_ttl_expiry_frees_reservation_over_http() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        client = await make_governed_client(governor)
        try:
            resp = await client.post(
                "/memory/reserve",
                json={"device": "ram", "bytes": 60, "ttlSeconds": 0.05},
            )
            assert resp.status == 201
            # The holder goes silent - a crashed peer. TTL is the cleanup.
            async with asyncio.timeout(2):
                while governor.reserved("ram") > 0:
                    await asyncio.sleep(0.01)
            data = await (await client.get("/memory/status")).json()
            assert data["leases"] == []
        finally:
            await client.close()

    run_scenario(scenario())


def test_memory_shed_and_cache_trim() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        cache = TrimmableCache("ram", 40)
        pool = TrimmableCache("ram", 30)
        governor.register_shedder(cache, name="cache", priority=0)
        governor.register_shedder(pool, name="pool", priority=1)
        client = await make_governed_client(governor)
        try:
            # Consumer names are discoverable through status.
            data = await (await client.get("/memory/status")).json()
            assert data["memoryGovernor"]["ram"]["consumers"] == {
                "cache": 40,
                "pool": 30,
            }

            # Shed asks everyone in priority order; freed is honest.
            resp = await client.post("/memory/shed", json={"device": "ram", "bytes": 50})
            assert resp.status == 200
            body = await resp.json()
            assert body["freedBytes"] == 50
            assert (cache.holding, pool.holding) == (0, 20)

            # Trim targets named consumers only; the pool is untouched.
            cache.holding = 40
            resp = await client.post(
                "/cache/trim",
                json={"device": "ram", "consumers": ["cache"]},
            )
            body = await resp.json()
            assert body["freedBytes"] == 40
            assert (cache.holding, pool.holding) == (0, 20)

            # Bytes omitted means full maintenance: everything sheddable goes.
            cache.holding = 10
            resp = await client.post("/cache/trim", json={"device": "ram"})
            body = await resp.json()
            assert body["freedBytes"] == 30
            assert (cache.holding, pool.holding) == (0, 0)
        finally:
            await client.close()

    run_scenario(scenario())


def test_memory_free_requires_and_preserves_paused_idle_queue() -> None:
    async def scenario() -> None:
        entered = asyncio.Event()
        finish = asyncio.Event()

        async def full_free(request_id: str) -> list[dict[str, object]]:
            assert request_id == "maintenance-1"
            entered.set()
            await finish.wait()
            return [
                {
                    "worker": "models-a",
                    "workerInstance": "instance-a",
                    "deviceMap": {"mapping": {"cuda:0": "cuda:1"}, "qualifier": None},
                    "status": "complete",
                    "consumers": [
                        {"consumer": "models", "status": "complete"},
                        {"consumer": "zero-cost", "status": "complete"},
                    ],
                },
                {
                    "worker": "models-b",
                    "workerInstance": "instance-b",
                    "deviceMap": {"mapping": {}, "qualifier": "cpu"},
                    "status": "complete",
                    "consumers": [
                        {"consumer": "models", "status": "complete"},
                        {"consumer": "zero-cost", "status": "complete"},
                    ],
                },
            ]

        app = create_app(make_engine, SCHEMAS, full_free=full_free)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.post("/memory/free", json={"requestId": "maintenance-1"})
            assert response.status == 409
            assert (await response.json())["queuePaused"] is False

            assert (await client.post("/api/queue/pause")).status == 200
            operation = asyncio.create_task(
                client.post("/memory/free", json={"requestId": "maintenance-1"})
            )
            await entered.wait()
            assert (await client.post("/api/queue/resume")).status == 409
            duplicate = await client.post(
                "/memory/free", json={"requestId": "maintenance-duplicate"}
            )
            assert duplicate.status == 409
            finish.set()

            response = await operation
            assert response.status == 200
            body = await response.json()
            assert body["requestId"] == "maintenance-1"
            assert body["completed"] is True
            assert body["queuePaused"] is True
            assert len(body["workers"]) == 2
            assert body["workers"][0]["deviceMap"]["mapping"] == {"cuda:0": "cuda:1"}
            assert app[STATE_KEY].queue.paused
            assert (await client.post("/api/queue/resume")).status == 200
        finally:
            await client.close()

    run_scenario(scenario())


def test_memory_free_reports_incomplete_participant_and_validates_nonce() -> None:
    async def scenario() -> None:
        async def full_free(_request_id: str) -> list[dict[str, object]]:
            return [
                {
                    "worker": "models",
                    "workerInstance": "instance-a",
                    "deviceMap": {"mapping": {}, "qualifier": None},
                    "status": "unsupported",
                    "consumers": [{"consumer": "legacy-cache", "status": "unsupported"}],
                }
            ]

        app = create_app(make_engine, SCHEMAS, full_free=full_free)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            assert (await client.post("/api/queue/pause")).status == 200
            for body in ({}, {"requestId": ""}, {"requestId": 1}):
                assert (await client.post("/memory/free", json=body)).status == 400
            response = await client.post("/memory/free", json={"requestId": "maintenance-2"})
            assert response.status == 200
            body = await response.json()
            assert body["completed"] is False
            assert body["queuePaused"] is True
            assert body["workers"][0]["status"] == "unsupported"
        finally:
            await client.close()

    run_scenario(scenario())


def test_memory_free_refuses_running_queue_without_cancelling_job() -> None:
    async def scenario() -> None:
        reset_sleeper()
        called = False

        async def full_free(_request_id: str) -> list[dict[str, object]]:
            nonlocal called
            called = True
            return []

        app = create_app(make_engine, SCHEMAS, full_free=full_free)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            graph = Graph(nodes={"n": GraphNode("test.sleeper", {"tag": "x"})})
            assert (await client.post("/api/jobs", json=submit_body(graph, ["n"]))).status == 202
            await Sleeper.entered.wait()
            assert (await client.post("/api/queue/pause")).status == 200
            response = await client.post("/memory/free", json={"requestId": "while-running"})
            assert response.status == 409
            assert called is False
            assert (await response.json())["queuePaused"] is True
            assert (await (await client.get("/api/jobs/c1/j1")).json())["state"] == "running"
            Sleeper.gate.set()
            async with asyncio.timeout(5):
                while (await (await client.get("/api/jobs/c1/j1")).json())["state"] != "completed":
                    await asyncio.sleep(0.01)
            assert app[STATE_KEY].queue.paused
        finally:
            Sleeper.gate.set()
            await client.close()

    run_scenario(scenario())


# -- consumer detail contract over the wire (DESIGN 3.10 observability) --------


class DetailedPool(TrimmableCache):
    """A consumer that also names what it holds: items with stable IDs."""

    def __init__(self, device: str, models: dict[str, int]) -> None:
        super().__init__(device, sum(models.values()))
        self.models = dict(models)

    async def shed(self, pressure: PressureSignal) -> int:
        if pressure.items is None:
            return await super().shed(pressure)
        freed = 0
        for item_id in pressure.items:
            nbytes = self.models.pop(item_id, 0)
            freed += nbytes
            self.holding -= nbytes
        return freed

    def details(self) -> list[ConsumerItem]:
        return [
            ConsumerItem(
                item_id=item_id,
                display_name=f"checkpoint {item_id}.safetensors",
                bytes_by_residency={self.device: nbytes},
                pages=PageMap(page_bytes=32, flags=(1, 1, 0)),
            )
            for item_id, nbytes in sorted(self.models.items())
        ]


def test_trim_items_unloads_one_model_by_stable_id() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"vram:cuda:0": 100})
        pool = DetailedPool("vram:cuda:0", {"ckpt-a": 40, "ckpt-b": 30})
        governor.register_shedder(pool, name="models")
        client = await make_governed_client(governor)
        try:
            # "Unload ckpt-a", as a frontend button would express it.
            resp = await client.post(
                "/cache/trim",
                json={
                    "device": "vram:cuda:0",
                    "consumers": ["models"],
                    "items": ["ckpt-a"],
                },
            )
            assert resp.status == 200
            assert (await resp.json())["freedBytes"] == 40
            assert set(pool.models) == {"ckpt-b"}
        finally:
            await client.close()

    run_scenario(scenario())


# -- frozen-run value peeking (/api/values, DESIGN 3.5) -------------------------


class CurveOutput(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.curve_output",
            outputs=(OutputSpec("curve", TypeExpr.concrete(CURVE_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(
            curve=Curve(((0.0, 0.125), (1.25, 0.875), (4.0, 0.25)), "monotone_cubic")
        )


VALUES_NODES: list[type[Node]] = NODES + SCAFFOLD_NODES + [CurveOutput]
VALUES_SCHEMAS = build_schemas(VALUES_NODES)


async def make_values_client(cache_entries: int = 1024) -> TestClient:
    def factory(on_event: EventListener) -> Engine:
        registry = TypeRegistry()
        register_core_types(registry)
        register_scaffold_types(registry)
        register_curve_type(registry)
        return Engine(
            schemas=VALUES_SCHEMAS,
            registry=registry,
            worker=InProcessWorker(build_node_types(VALUES_NODES), registry),
            cache=MemoryLRUCache(max_entries=cache_entries),
            on_event=on_event,
        )

    app = create_app(
        factory,
        VALUES_SCHEMAS,
        choices=scaffold_choices(),
        lazy_choices=scaffold_lazy_choices(),
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


async def finish_job(
    client: TestClient, graph: Graph, targets: list[str], job_id: str = "j1"
) -> Any:
    resp = await client.post("/api/jobs", json=submit_body(graph, targets, jobId=job_id))
    assert resp.status == 202
    async with asyncio.timeout(5):
        while True:
            status = await (await client.get(f"/api/jobs/c1/{job_id}")).json()
            if status["state"] in ("completed", "failed", "cancelled"):
                assert status["state"] == "completed"
                return status
            await asyncio.sleep(0.01)


def values_url(node_id: str, output_id: str = "out", job_id: str = "j1", **extra: str) -> str:
    params = {
        "clientId": "c1",
        "jobId": job_id,
        "nodeId": node_id,
        "outputId": output_id,
        **extra,
    }
    return "/api/values?" + "&".join(f"{k}={v}" for k, v in params.items())


def test_values_endpoint_serves_target_and_intermediate() -> None:
    """Lookup authority is execution-scoped identity: targets resolve from
    the retained job result, intermediates through the exact cache key the
    run's own events reported."""

    async def scenario() -> None:
        client = await make_values_client()
        try:
            await finish_job(client, echo_graph(), ["s"])
            resp = await client.get(values_url("s"))
            assert resp.status == 200
            data = await resp.json()
            assert data["available"] is True
            assert data["descriptor"]["typeId"] == "STRING"
            assert data["descriptor"]["value"] == "HELLO"
            assert data["descriptor"]["fingerprint"]
            assert data["renditions"] == [
                {"kind": "text", "mime": "text/plain", "default": True, "cacheKey": "text"}
            ]
            text = await client.get(values_url("s", rendition="text"))
            assert text.status == 200
            assert text.content_type == "text/plain"
            assert await text.text() == "HELLO"

            resp = await client.get(values_url("e"))  # intermediate, retained
            data = await resp.json()
            assert (resp.status, data["available"]) == (200, True)
            assert data["descriptor"]["value"] == "hello"
            text = await client.get(values_url("e", rendition="default"))
            assert text.status == 200
            assert await text.text() == "hello"
        finally:
            await client.close()

    run_scenario(scenario())


def test_values_endpoint_reports_eviction_never_substitutes() -> None:
    """An evicted intermediate is a structured refusal (410), never a
    silently different value - peek's whole safety story."""

    async def scenario() -> None:
        client = await make_values_client(cache_entries=1)
        try:
            await finish_job(client, echo_graph(), ["s"])
            # "s" evicted "e" from the single-entry cache; the target output
            # itself is retained on the job result regardless.
            resp = await client.get(values_url("e"))
            data = await resp.json()
            assert (resp.status, data["available"]) == (410, False)
            assert data["reason"] == "evicted"

            resp = await client.get(values_url("s"))
            assert (await resp.json())["descriptor"]["value"] == "HELLO"
        finally:
            await client.close()

    run_scenario(scenario())


def test_values_endpoint_identity_refusals() -> None:
    async def scenario() -> None:
        client = await make_values_client()
        try:
            await finish_job(client, echo_graph(), ["s"])

            resp = await client.get(values_url("s", job_id="nope"))
            assert (resp.status, (await resp.json())["reason"]) == (404, "unknown-job")

            resp = await client.get(values_url("s", output_id="nope"))
            assert (resp.status, (await resp.json())["reason"]) == (
                404,
                "unknown-output",
            )

            resp = await client.get(values_url("ghost"))
            assert (resp.status, (await resp.json())["reason"]) == (404, "not-retained")

            resp = await client.get("/api/values?clientId=c1&jobId=j1")
            assert resp.status == 400  # missing nodeId/outputId

            resp = await client.get(values_url("s", element="0"))
            assert (resp.status, (await resp.json())["reason"]) == (404, "bad-element")

            resp = await client.get(values_url("s", element="zero"))
            assert resp.status == 400
        finally:
            await client.close()

    run_scenario(scenario())


def test_values_endpoint_refuses_running_job() -> None:
    """Frozen-view contract: values are served once the job is terminal;
    live values ride the event stream."""

    async def scenario() -> None:
        reset_sleeper()
        client = await make_values_client()
        try:
            graph = Graph(nodes={"n": GraphNode("test.sleeper", {"tag": "x"})})
            resp = await client.post("/api/jobs", json=submit_body(graph, ["n"]))
            assert resp.status == 202
            await Sleeper.entered.wait()
            resp = await client.get(values_url("n"))
            data = await resp.json()
            assert (resp.status, data["reason"]) == (409, "not-complete")
            Sleeper.gate.set()
        finally:
            await client.close()

    run_scenario(scenario())


def test_values_endpoint_element_traversal() -> None:
    async def scenario() -> None:
        client = await make_values_client()
        try:
            graph = Graph(nodes={"sp": GraphNode("test.splitter", {"text": "a b c"})})
            await finish_job(client, graph, ["sp"])

            resp = await client.get(values_url("sp", output_id="words", element="1"))
            data = await resp.json()
            assert data["descriptor"]["typeId"] == "STRING"
            assert data["descriptor"]["value"] == "b"

            resp = await client.get(values_url("sp", output_id="words", element="5"))
            data = await resp.json()
            assert (resp.status, data["reason"]) == (404, "bad-element")
            assert "length 3" in data["error"]
        finally:
            await client.close()

    run_scenario(scenario())


def test_values_endpoint_file3d_manifest_preserves_named_asset_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_assets import AssetRef, AssetVault, digest_bytes, register_file3d_types

    vault = AssetVault(tmp_path / "vault")
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(vault.root))
    refs = []
    for data, name in ((b'{"asset":{"version":"2.0"}}', "scene.gltf"), (b"\x01\xfe", "a.bin")):
        digest = digest_bytes(data)
        with vault.writer(digest) as writer:
            writer.write(data)
            writer.commit()
        refs.append(AssetRef(digest, name, len(data), resolver=vault))
    expected = {
        "format": "gltf",
        "source": refs[0].to_wire(),
        "resources": {"nested/a.bin": refs[1].to_wire()},
    }

    class Model(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.model",
                outputs=(OutputSpec("file", TypeExpr.concrete("FILE_3D_GLTF")),),
            )

        @classmethod
        def execute(cls) -> Mapping[str, object]:
            return cls.outputs(file=expected)

    schemas = build_schemas([Model])

    def factory(on_event: EventListener) -> Engine:
        registry = TypeRegistry()
        register_core_types(registry)
        register_file3d_types(registry, vault)
        return Engine(
            schemas=schemas,
            registry=registry,
            worker=InProcessWorker(build_node_types([Model]), registry),
            cache=MemoryLRUCache(),
            on_event=on_event,
        )

    async def scenario() -> None:
        client = TestClient(TestServer(create_app(factory, schemas)))
        await client.start_server()
        try:
            await finish_job(client, Graph(nodes={"model": GraphNode("test.model", {})}), ["model"])
            url = values_url("model", output_id="file")
            response = await client.get(url)
            discovery = await response.json()
            assert response.status == 200
            assert "value" not in discovery["descriptor"]
            assert discovery["renditions"] == [
                {
                    "kind": "manifest",
                    "mime": "application/json",
                    "default": True,
                    "cacheKey": "manifest",
                }
            ]
            response = await client.get(url + "&rendition=manifest")
            assert response.status == 200
            assert response.content_type == "application/json"
            assert await response.json() == expected
            response = await client.get(
                url.replace("jobId=j1", "jobId=absent") + "&rendition=manifest"
            )
            assert response.status == 404
        finally:
            await client.close()

    run_scenario(scenario())


def test_values_endpoint_curve_points_rendition_is_authoritative_and_bounded() -> None:
    async def scenario() -> None:
        client = await make_values_client()
        try:
            graph = Graph(nodes={"curve": GraphNode("test.curve_output", {})})
            status = await finish_job(client, graph, ["curve"])
            descriptor = status["outputs"]["curve"]["curve"]
            assert set(descriptor) == {"typeId", "fingerprint", "meta"}
            assert descriptor["typeId"] == CURVE_TYPE
            assert descriptor["meta"] == {"pointCount": 3, "start": 0.0, "end": 4.0}

            url = values_url("curve", output_id="curve")
            response = await client.get(url)
            discovery = await response.json()
            assert discovery["descriptor"] == descriptor
            assert discovery["renditions"] == [
                {
                    "kind": "curve-points",
                    "mime": "application/json",
                    "default": True,
                    "cacheKey": "curve-points",
                    "limits": {"points": 4096},
                }
            ]

            response = await client.get(url + "&rendition=curve-points")
            assert response.status == 200
            assert response.headers["Content-Type"] == "application/json"
            assert await response.json() == {
                "interpolation": "monotone_cubic",
                "points": [
                    {"position": 0.0, "value": 0.125},
                    {"position": 1.25, "value": 0.875},
                    {"position": 4.0, "value": 0.25},
                ],
            }

            job = cast(TestServer, client.server).app[STATE_KEY].queue.get("c1", "j1")
            assert job is not None and job.result is not None
            outputs = cast(dict[str, dict[str, Value]], job.result.outputs)
            oversized = {"points": [{"position": index, "value": index} for index in range(4097)]}
            outputs["curve"]["curve"] = Value(
                type_id=CURVE_TYPE,
                fingerprint="oversized-curve",
                meta=ValueMeta({"pointCount": 4097, "start": 0.0, "end": 4096.0}),
                payload=PyObjPayload(oversized),
            )
            response = await client.get(url + "&rendition=curve-points")
            refusal = await response.json()
            assert (response.status, refusal["available"], refusal["reason"]) == (
                404,
                False,
                "rendition_unavailable",
            )
            assert "preview exceeds 4096" in refusal["error"]
        finally:
            await client.close()

    run_scenario(scenario())


def test_values_endpoint_image_png_rendition() -> None:
    """Rich types return negotiated browser-renderable bytes: renditions are
    declared in the descriptor response, fetched by kind, and cached by the
    fingerprint-derived ETag (identity looks up, fingerprint validates)."""

    async def scenario() -> None:
        client = await make_values_client()
        try:
            graph = Graph(nodes={"g": GraphNode("dev.image.gradient", {"width": 4, "height": 2})})
            await finish_job(client, graph, ["g"])
            url = values_url("g", output_id="image")

            resp = await client.get(url)
            data = await resp.json()
            assert data["descriptor"]["typeId"] == "dev.image"
            assert "value" not in data["descriptor"]  # tensors never inline
            assert data["renditions"] == [
                {
                    "kind": "png",
                    "mime": "image/png",
                    "default": True,
                    "cacheKey": f"png/{PNG_CONTAINER_VERSION}",
                }
            ]

            legacy_url = url + "&rendition=png"
            immutable_url = url + f"&rendition=png/{PNG_CONTAINER_VERSION}"
            assert immutable_url != legacy_url
            resp = await client.get(immutable_url)
            assert resp.status == 200
            assert resp.headers["Content-Type"] == "image/png"
            assert resp.headers["Cache-Control"] == "private, max-age=31536000, immutable"
            assert resp.headers["X-Dinkster-Type-Id"] == "dev.image"
            assert resp.headers["X-Dinkster-Rendition"] == "png"
            etag = resp.headers["ETag"]
            assert resp.headers["X-Dinkster-Fingerprint"] in etag
            legacy_etag = f'"{resp.headers["X-Dinkster-Fingerprint"]}/png"'
            assert etag != legacy_etag
            body = await resp.read()
            assert body[:8] == b"\x89PNG\r\n\x1a\n"

            # The unversioned compatibility URL always revalidates.
            resp = await client.get(legacy_url, headers={"If-None-Match": legacy_etag})
            assert resp.status == 200
            assert resp.headers["Cache-Control"] == "private, no-cache"
            assert resp.headers["ETag"] == etag
            assert await resp.read() == body
            resp = await client.get(legacy_url, headers={"If-None-Match": etag})
            assert resp.status == 304
            assert resp.headers["Cache-Control"] == "private, no-cache"

            # "default" remains a revalidating compatibility alias.
            resp = await client.get(url + "&rendition=default")
            assert resp.headers["X-Dinkster-Rendition"] == "png"
            assert resp.headers["Cache-Control"] == "private, no-cache"
            resp = await client.get(immutable_url, headers={"If-None-Match": etag})
            assert resp.status == 304

            resp = await client.get(url + "&rendition=png/unknown")
            assert resp.status == 406

            resp = await client.get(url + "&rendition=bogus")
            data = await resp.json()
            assert (resp.status, data["reason"]) == (406, "no-rendition")
            assert data["renditions"] == ["png"]
        finally:
            await client.close()

    run_scenario(scenario())


def test_values_endpoint_av_renditions_share_discovery_and_response_mime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AV discovery and bytes resolve MIME from the same value metadata."""
    import struct

    import numpy as np
    from dinkster_api.v1 import (
        AUDIO_TYPE,
        VIDEO_TYPE,
        register_audio_value_type,
        register_video_value_type,
    )

    from tests.test_video_probe import _encode

    mp4 = _encode("mp4", "libx264", "yuv420p")
    webm = _encode("webm", "libvpx-vp9", "yuv420p")
    waveform = np.array([[[-1.0, 0.0, 1.0]]], dtype=np.float32)

    class AVOutputs(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.av_outputs",
                outputs=(
                    OutputSpec("audio", TypeExpr.concrete(AUDIO_TYPE), preview=True),
                    OutputSpec("mp4", TypeExpr.concrete(VIDEO_TYPE), preview=True),
                    OutputSpec("webm", TypeExpr.concrete(VIDEO_TYPE), preview=True),
                ),
            )

        @classmethod
        def execute(cls) -> Mapping[str, object]:
            return cls.outputs(
                audio={"waveform": waveform, "sample_rate": 8_000},
                mp4={"container": "mp4", "bytes": mp4},
                webm={"container": "webm", "bytes": webm},
            )

    nodes: list[type[Node]] = [AVOutputs]
    schemas = build_schemas(nodes)
    expected_registry = TypeRegistry()
    register_core_types(expected_registry)
    register_audio_value_type(expected_registry, AUDIO_TYPE)
    register_video_value_type(expected_registry, VIDEO_TYPE)
    expected_fingerprints = {
        "audio": expected_registry.wrap(
            AUDIO_TYPE, {"waveform": waveform, "sample_rate": 8_000}
        ).fingerprint,
        "mp4": expected_registry.wrap(VIDEO_TYPE, {"container": "mp4", "bytes": mp4}).fingerprint,
        "webm": expected_registry.wrap(
            VIDEO_TYPE, {"container": "webm", "bytes": webm}
        ).fingerprint,
    }

    def factory(on_event: EventListener) -> Engine:
        registry = TypeRegistry()
        register_core_types(registry)
        register_audio_value_type(registry, AUDIO_TYPE)
        register_video_value_type(registry, VIDEO_TYPE)
        return Engine(
            schemas=schemas,
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
            on_event=on_event,
        )

    async def scenario() -> None:
        client = TestClient(TestServer(create_app(factory, schemas)))
        await client.start_server()
        try:
            await finish_job(client, Graph(nodes={"av": GraphNode("test.av_outputs", {})}), ["av"])
            cases = (
                ("audio", "wav", "audio/wav", None),
                ("mp4", "original", "video/mp4", mp4),
                ("webm", "original", "video/webm", webm),
            )
            for output_id, kind, mime, expected_body in cases:
                url = values_url("av", output_id=output_id)
                discovery = await (await client.get(url)).json()
                assert discovery["descriptor"]["fingerprint"] == expected_fingerprints[output_id]
                expected_renditions = [
                    {"kind": kind, "mime": mime, "default": True, "cacheKey": kind}
                ]
                if output_id == "audio":
                    expected_renditions.extend(
                        [
                            {
                                "kind": "waveform",
                                "mime": "image/png",
                                "default": False,
                                "cacheKey": "waveform/rgba-v1",
                                "version": "rgba-v1",
                                "parameters": ["batch", "waveform"],
                                "defaults": {"batch": "0"},
                                "limits": {"width": 2048, "height": 512, "pixels": 262144},
                            },
                            {
                                "kind": "window",
                                "mime": "audio/wav",
                                "default": False,
                                "cacheKey": "window/pcm16-v1",
                                "version": "pcm16-v1",
                                "parameters": ["batch", "window"],
                                "defaults": {"batch": "0"},
                                "limits": {"durationSeconds": 30, "sampleValues": 8388608},
                            },
                        ]
                    )
                assert discovery["renditions"] == expected_renditions

                response = await client.get(url + "&rendition=default")
                assert response.status == 200
                assert response.headers["Content-Type"] == mime
                assert (
                    response.headers["X-Dinkster-Fingerprint"] == expected_fingerprints[output_id]
                )
                assert response.headers["ETag"] == (f'"{expected_fingerprints[output_id]}/{kind}"')
                body = await response.read()
                if expected_body is None:
                    assert body[:12] == b"RIFF" + struct.pack("<I", 42) + b"WAVE"
                    assert struct.unpack("<3h", body[44:]) == (-32768, 0, 32767)
                else:
                    assert body == expected_body

            audio_url = values_url("av", output_id="audio")
            response = await client.get(
                audio_url + "&rendition=waveform/rgba-v1&batch=0&waveform=3x5"
            )
            assert response.status == 200
            assert response.headers["Content-Type"] == "image/png"
            assert response.headers["Cache-Control"] == "private, max-age=31536000, immutable"
            waveform_etag = response.headers["ETag"]
            assert waveform_etag == (
                f'"{expected_fingerprints["audio"]}/waveform/rgba-v1?batch=0&waveform=3x5"'
            )
            assert (await response.read()).startswith(b"\x89PNG\r\n\x1a\n")

            response = await client.get(audio_url + "&rendition=waveform&waveform=3x5")
            assert response.status == 200
            assert response.headers["Cache-Control"] == "private, no-cache"
            assert response.headers["ETag"] == waveform_etag

            response = await client.get(
                audio_url + "&rendition=waveform/rgba-v1&batch=00&waveform=3x5",
                headers={"If-None-Match": waveform_etag},
            )
            assert response.status == 304
            assert response.headers["ETag"] == waveform_etag

            response = await client.get(
                audio_url + "&rendition=window/pcm16-v1&batch=0&window=0,0.00025"
            )
            assert response.status == 200
            assert response.headers["Content-Type"] == "audio/wav"
            assert response.headers["Cache-Control"] == "private, max-age=31536000, immutable"
            assert response.headers["ETag"] == (
                f'"{expected_fingerprints["audio"]}/window/pcm16-v1?batch=0&window=0%2C0.00025"'
            )
            assert struct.unpack("<2h", (await response.read())[44:]) == (-32768, 0)

            for query in (
                "&rendition=waveform&waveform=3x5&waveform=4x5",
                "&rendition=waveform&window=0,1&waveform=3x5",
                "&rendition=waveform&batch=1&waveform=3x5",
                "&rendition=window&batch=0",
                f"&rendition=waveform&waveform={'1' * 5000}x1",
                f"&rendition=waveform&batch={'1' * 5000}&waveform=3x5",
                f"&rendition=window&window=0,{'1' * 5000}",
            ):
                response = await client.get(audio_url + query)
                data = await response.json()
                assert response.status == 400
                assert response.headers["Content-Type"].startswith("application/problem+json")
                assert data["reason"] == "invalid_rendition_request"

            import dinkster_values.audio_codec as codec
            from dinkster_values import RenditionUnavailable

            class UnavailableReader:
                def __init__(self, _obj: object) -> None:
                    pass

                def __enter__(self) -> UnavailableReader:
                    raise RenditionUnavailable("audio source is unavailable")

                def __exit__(self, *args: object) -> None:
                    pass

            monkeypatch.setattr(codec, "AudioWindowReader", UnavailableReader)
            response = await client.get(audio_url + "&rendition=waveform&waveform=3x5")
            data = await response.json()
            assert response.status == 404
            assert response.headers["Content-Type"].startswith("application/problem+json")
            assert data["reason"] == "rendition_unavailable"
        finally:
            await client.close()

    run_scenario(scenario())


def test_compat_save_target_reaches_execution_with_default_and_explicit_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The asset registry must wrap the SAVE_TARGET literal before a
    job reaches its worker. Both its advertised default and an
    explicit graph value arrive as validated SaveTarget values and produce
    digest-addressed asset receipts."""
    from dinkster_assets import (
        ASSET_TYPE,
        SAVE_TARGET_TYPE,
        AssetError,
        AssetWriter,
        MountSnapshotWriter,
        SaveTarget,
        digest_bytes,
        register_asset_type,
        register_save_target_type,
        resolver_from_env,
    )

    output_root = tmp_path / "output"
    output_root.mkdir()
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps(
            {
                "mounts": [
                    {
                        "id": "comfy-output",
                        "root": str(output_root),
                        "mode": "readwrite",
                    }
                ]
            }
        ),
        "utf-8",
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))

    save_target = TypeExpr.concrete(SAVE_TARGET_TYPE)

    class StringPrimitive(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="dinkster.string",
                inputs=(InputSpec("value", TypeExpr.concrete("STRING")),),
                outputs=(OutputSpec("value", TypeExpr.concrete("STRING")),),
            )

        @classmethod
        def execute(cls, *, value: str) -> Mapping[str, object]:
            return cls.outputs(value=value)

    class SetSaveTargetPrefix(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="dinkster.set_save_target_prefix",
                inputs=(
                    InputSpec("target", save_target),
                    InputSpec("prefix", TypeExpr.concrete("STRING")),
                ),
                outputs=(OutputSpec("save_target", save_target),),
            )

        @classmethod
        def execute(cls, *, target: SaveTarget, prefix: str) -> Mapping[str, object]:
            return cls.outputs(save_target=SaveTarget(mount=target.mount, prefix=prefix))

    class CompatSave(Node):
        seen: list[SaveTarget] = []

        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="dinkster.save_image",
                inputs=(
                    InputSpec(
                        "target",
                        save_target,
                        required=False,
                        default={"mount": "comfy-output", "prefix": "ComfyUI"},
                    ),
                ),
                outputs=(OutputSpec("receipt", TypeExpr.concrete(ASSET_TYPE)),),
                idempotent=False,
                output_node=True,
            )

        @classmethod
        async def execute(cls, *, target: SaveTarget) -> Mapping[str, object]:
            if not isinstance(target, SaveTarget):
                raise AssetError("save target input must be a SaveTarget")
            receipt = AssetWriter(MountSnapshotWriter(snapshot)).save_bytes(
                target,
                f"saved:{target.prefix}".encode(),
                suffix=".bin",
                media_type="application/octet-stream",
            )
            cls.seen.append(target)
            return cls.outputs(receipt=receipt)

    nodes: list[type[Node]] = [StringPrimitive, SetSaveTargetPrefix, CompatSave]
    schemas = build_schemas(nodes)

    def factory(on_event: EventListener) -> Engine:
        registry = TypeRegistry()
        register_core_types(registry)
        register_asset_type(registry, resolver_from_env())
        register_save_target_type(registry)
        return Engine(
            schemas=schemas,
            registry=registry,
            worker=InProcessWorker(
                build_node_types(nodes),
                registry,
            ),
            cache=MemoryLRUCache(),
            on_event=on_event,
        )

    async def scenario() -> None:
        app = create_app(factory, schemas)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            default_graph = Graph(nodes={"s": GraphNode("dinkster.save_image", {})})
            default_status = await finish_job(client, default_graph, ["s"])
            default_meta = default_status["outputs"]["s"]["receipt"]["meta"]
            default_payload = b"saved:ComfyUI"
            assert default_meta["digest"] == digest_bytes(default_payload)
            assert default_meta["virtualPath"].endswith("/ComfyUI_00001.bin")
            assert (output_root / "ComfyUI_00001.bin").read_bytes() == default_payload

            explicit = {"mount": "comfy-output", "prefix": "renders/scene"}
            explicit_graph = Graph(
                nodes={"s": GraphNode("dinkster.save_image", {"target": explicit})}
            )
            explicit_status = await finish_job(client, explicit_graph, ["s"], job_id="j2")
            explicit_meta = explicit_status["outputs"]["s"]["receipt"]["meta"]
            explicit_payload = b"saved:renders/scene"
            assert explicit_meta["digest"] == digest_bytes(explicit_payload)
            assert explicit_meta["virtualPath"].endswith("/renders/scene_00001.bin")
            assert (output_root / "renders" / "scene_00001.bin").read_bytes() == (explicit_payload)

            computed_graph = Graph(
                nodes={
                    "p": GraphNode("dinkster.string", {"value": "computed/live-scene"}),
                    "t": GraphNode(
                        "dinkster.set_save_target_prefix",
                        {
                            "target": {
                                "mount": "comfy-output",
                                "prefix": "manual/fallback",
                            },
                            "prefix": Link("p", "value"),
                        },
                    ),
                    "s": GraphNode(
                        "dinkster.save_image",
                        {"target": Link("t", "save_target")},
                    ),
                }
            )
            computed_status = await finish_job(client, computed_graph, ["s"], job_id="j3")
            computed_meta = computed_status["outputs"]["s"]["receipt"]["meta"]
            computed_payload = b"saved:computed/live-scene"
            assert computed_meta["digest"] == digest_bytes(computed_payload)
            assert computed_meta["virtualPath"].endswith("/computed/live-scene_00001.bin")
            assert (
                output_root / "computed" / "live-scene_00001.bin"
            ).read_bytes() == computed_payload

            raw_string_graph = Graph(
                nodes={
                    "p": GraphNode("dinkster.string", {"value": "raw/string"}),
                    "s": GraphNode("dinkster.save_image", {"target": Link("p", "value")}),
                }
            )
            response = await client.post(
                "/api/jobs",
                json=submit_body(raw_string_graph, ["s"], jobId="j4"),
            )
            assert response.status == 202
            raw_status = await poll_job(client, "j4")
            assert raw_status["state"] == "failed"
            assert "must be a SaveTarget" in json.dumps(raw_status)

            snapshot.write_text(
                json.dumps(
                    {
                        "mounts": [
                            {
                                "id": "comfy-output",
                                "root": str(output_root),
                                "mode": "read",
                            }
                        ]
                    }
                ),
                "utf-8",
            )
            response = await client.post(
                "/api/jobs",
                json=submit_body(computed_graph, ["s"], jobId="j5"),
            )
            assert response.status == 202
            read_only_status = await poll_job(client, "j5")
            assert read_only_status["state"] == "failed"
            assert "read-only" in json.dumps(read_only_status)

            unknown_mount_graph = Graph(
                nodes={
                    "p": GraphNode("dinkster.string", {"value": "computed/unknown"}),
                    "t": GraphNode(
                        "dinkster.set_save_target_prefix",
                        {
                            "target": {
                                "mount": "missing-output",
                                "prefix": "manual/fallback",
                            },
                            "prefix": Link("p", "value"),
                        },
                    ),
                    "s": GraphNode(
                        "dinkster.save_image",
                        {"target": Link("t", "save_target")},
                    ),
                }
            )
            response = await client.post(
                "/api/jobs",
                json=submit_body(unknown_mount_graph, ["s"], jobId="j6"),
            )
            assert response.status == 202
            unknown_status = await poll_job(client, "j6")
            assert unknown_status["state"] == "failed"
            assert "no ready mount 'missing-output'" in json.dumps(unknown_status)

            snapshot.write_text(json.dumps({"mounts": []}), "utf-8")
            response = await client.post(
                "/api/jobs",
                json=submit_body(computed_graph, ["s"], jobId="j7"),
            )
            assert response.status == 202
            revoked_status = await poll_job(client, "j7")
            assert revoked_status["state"] == "failed"
            assert "no ready mount 'comfy-output'" in json.dumps(revoked_status)

            assert CompatSave.seen == [
                SaveTarget(mount="comfy-output", prefix="ComfyUI"),
                SaveTarget(**explicit),
                SaveTarget(mount="comfy-output", prefix="computed/live-scene"),
            ]
        finally:
            await client.close()

    run_scenario(scenario())


# -- progressive schema announcement -------------------------------------------


async def poll_job(client: TestClient, job_id: str) -> dict[str, object]:
    async with asyncio.timeout(5):
        while True:
            status = await (await client.get(f"/api/jobs/c1/{job_id}")).json()
            if status["state"] in ("completed", "failed"):
                return status
            await asyncio.sleep(0.01)
