"""dinkster-supervisor: the layer-0 process that owns the public port.

The supervisor binds instantly and narrates engine startup at
/supervisor/status; every other route 503s with a pointer until the
engine is healthy, then proxies transparently (HTTP + WebSocket). The
HTTP surface is tested against an in-process engine by driving the
EngineLink directly; process management (spawn, health probe, exit
detection) is tested with a real stub engine subprocess.
"""

from __future__ import annotations

import asyncio
import sys
import textwrap
import time
from pathlib import Path

from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestClient, TestServer
from dinkster_supervisor import (
    PROTOCOL_VERSION,
    STATUS_PATH,
    EngineLink,
    EngineProcess,
    create_supervisor_app,
)


def make_engine_app() -> web.Application:
    """A stand-in engine: enough surface to prove the proxy is transparent
    (JSON, echo with query/body/headers, streaming, 404s, WebSocket)."""
    app = web.Application()

    async def health(request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def nodes(request: web.Request) -> web.Response:
        return web.json_response({"nodes": {"std.add": {}}, "packs": {}})

    async def echo(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "method": request.method,
                "query": dict(request.query),
                "body": (await request.read()).decode(),
                "content_type": request.headers.get("Content-Type", ""),
            },
            headers={"X-Engine": "yes"},
        )

    async def events(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str("hello")
        async for message in ws:
            if message.type == WSMsgType.TEXT:
                await ws.send_str(f"echo:{message.data}")
            elif message.type == WSMsgType.BINARY:
                await ws.send_bytes(bytes(reversed(message.data)))
        return ws

    async def cors_confused(request: web.Request) -> web.Response:
        # An engine wrongly configured with its own CORS: the supervisor
        # must strip this so its own stamp is the single answer.
        return web.json_response(
            {"ok": True},
            headers={
                "Access-Control-Allow-Origin": "https://sneaky.example",
                "Access-Control-Expose-Headers": "X-Sneaky",
            },
        )

    app.router.add_get("/api/health", health)
    app.router.add_get("/api/nodes", nodes)
    app.router.add_post("/api/echo", echo)
    app.router.add_get("/api/events", events)
    app.router.add_get("/api/cors-confused", cors_confused)
    return app


def test_status_and_gating_before_ready() -> None:
    """The status surface is up from the first request; everything else is
    a 503 pointing at it - never a connection refusal, never a hang. The
    restart route without a process manager is an explicit 409."""

    async def scenario() -> None:
        link = EngineLink()
        client = TestClient(TestServer(create_supervisor_app(link)))
        await client.start_server()
        try:
            resp = await client.get(STATUS_PATH)
            assert resp.status == 200
            assert await resp.json() == {
                "protocol": PROTOCOL_VERSION,
                "state": "starting",
            }

            for path in ("/api/nodes", "/api/jobs", "/anything"):
                resp = await client.get(path)
                assert resp.status == 503
                body = await resp.json()
                assert body == {
                    "error": "engine-not-ready",
                    "state": "starting",
                    "status": STATUS_PATH,
                }

            assert (await client.post("/supervisor/engine/restart")).status == 409

            # Failure is narrated, not hidden: state + detail + exit code.
            link.state = "failed"
            link.detail = "engine exited during startup (exit code 3)"
            link.pid = 12345
            link.exit_code = 3
            status = await (await client.get(STATUS_PATH)).json()
            assert status["state"] == "failed"
            assert status["engine"] == {"pid": 12345, "exitCode": 3}
            assert "exit code 3" in status["detail"]
            assert (await client.get("/api/nodes")).status == 503
        finally:
            await client.close()

    asyncio.run(scenario())


def test_status_echoes_launcher_instance() -> None:
    async def scenario() -> None:
        app = create_supervisor_app(EngineLink(), instance="desktop-launch-123")
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            status = await (await client.get(STATUS_PATH)).json()
            assert status["instance"] == "desktop-launch-123"
        finally:
            await client.close()

    asyncio.run(scenario())


def test_proxy_is_transparent_when_ready() -> None:
    """Once the link is ready, HTTP crosses verbatim in both directions:
    method, path, query, body, content type, response headers, status
    codes (404 included). The supervisor's own routes stay its own."""

    async def scenario() -> None:
        engine_server = TestServer(make_engine_app())
        await engine_server.start_server()
        link = EngineLink(state="ready", base_url=str(engine_server.make_url("")).rstrip("/"))
        client = TestClient(TestServer(create_supervisor_app(link)))
        await client.start_server()
        try:
            resp = await client.get("/api/nodes")
            assert resp.status == 200
            assert await resp.json() == {"nodes": {"std.add": {}}, "packs": {}}

            resp = await client.post(
                "/api/echo?a=1&b=two",
                data=b'{"k": "v"}',
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 200
            assert resp.headers["X-Engine"] == "yes"
            assert await resp.json() == {
                "method": "POST",
                "query": {"a": "1", "b": "two"},
                "body": '{"k": "v"}',
                "content_type": "application/json",
            }

            # Engine 404s pass through as the engine's own answer.
            assert (await client.get("/api/nope")).status == 404
            # The supervisor's status route is never proxied.
            status = await (await client.get(STATUS_PATH)).json()
            assert status["state"] == "ready"
        finally:
            await client.close()
            await engine_server.close()

    asyncio.run(scenario())


def test_supervisor_namespace_is_reserved() -> None:
    """The whole /supervisor/ prefix answers locally: unknown management
    paths 404 with a pointer even when the engine is ready and would have
    answered. Future supervisor routes can never collide with (or leak
    to) engine routes."""

    async def scenario() -> None:
        engine_server = TestServer(make_engine_app())
        await engine_server.start_server()
        link = EngineLink(state="ready", base_url=str(engine_server.make_url("")).rstrip("/"))
        client = TestClient(TestServer(create_supervisor_app(link)))
        await client.start_server()
        try:
            for method, path in (
                ("GET", "/supervisor/nope"),
                ("POST", "/supervisor/engine/nope"),
                ("DELETE", "/supervisor/installs/x"),
            ):
                resp = await client.request(method, path)
                assert resp.status == 404
                assert await resp.json() == {
                    "error": "unknown-supervisor-path",
                    "status": STATUS_PATH,
                }
            # ...while the known management routes still answer.
            assert (await client.get(STATUS_PATH)).status == 200
        finally:
            await client.close()
            await engine_server.close()

    asyncio.run(scenario())


def test_supervisor_cors() -> None:
    """CORS is owned by the public-port owner: opt-in via allow_origins,
    stamped on local AND proxied responses, preflights answered locally
    (before readiness gating), and upstream Access-Control-* headers
    stripped so a CORS-configured child cannot double-stamp."""

    async def scenario() -> None:
        engine_server = TestServer(make_engine_app())
        await engine_server.start_server()
        base_url = str(engine_server.make_url("")).rstrip("/")
        ui = "https://ui.example"

        # Default: no CORS headers on any route, even for a ready engine.
        link = EngineLink(state="ready", base_url=base_url)
        client = TestClient(TestServer(create_supervisor_app(link)))
        await client.start_server()
        resp = await client.get(STATUS_PATH, headers={"Origin": ui})
        assert "Access-Control-Allow-Origin" not in resp.headers
        resp = await client.get("/api/nodes", headers={"Origin": ui})
        assert "Access-Control-Allow-Origin" not in resp.headers
        # The confused engine's own CORS headers cross verbatim when the
        # supervisor has no CORS policy (transparent proxy).
        resp = await client.get("/api/cors-confused", headers={"Origin": ui})
        assert resp.headers["Access-Control-Allow-Origin"] == "https://sneaky.example"
        await client.close()

        # Configured: local + proxied responses stamped, upstream stripped.
        link = EngineLink(state="ready", base_url=base_url)
        client = TestClient(TestServer(create_supervisor_app(link, allow_origins=[ui])))
        await client.start_server()
        try:
            resp = await client.get(STATUS_PATH, headers={"Origin": ui})
            assert resp.headers["Access-Control-Allow-Origin"] == ui
            assert "Origin" in resp.headers.getall("Vary", [])

            resp = await client.get("/api/nodes", headers={"Origin": ui})
            assert resp.status == 200
            assert resp.headers["Access-Control-Allow-Origin"] == ui

            # Upstream CORS stripped, supervisor stamp is the only writer.
            resp = await client.get("/api/cors-confused", headers={"Origin": ui})
            assert resp.headers.getall("Access-Control-Allow-Origin") == [ui]
            assert resp.headers["Access-Control-Expose-Headers"] == "ETag"

            # Unconfigured origin: normal response, no CORS headers.
            resp = await client.get("/api/nodes", headers={"Origin": "https://other.example"})
            assert resp.status == 200
            assert "Access-Control-Allow-Origin" not in resp.headers

            # Preflight for a proxied path is answered locally (204, never
            # reaches the engine - which has no OPTIONS route anyway).
            resp = await client.options(
                "/api/echo",
                headers={
                    "Origin": ui,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                },
            )
            assert resp.status == 204
            assert resp.headers["Access-Control-Allow-Origin"] == ui
            assert resp.headers["Access-Control-Allow-Methods"] == "POST"
            assert resp.headers["Access-Control-Allow-Headers"] == "content-type"

            # Preflights work before engine readiness too (a UI must be
            # able to poll /supervisor/status cross-origin during startup).
            link.state = "starting"
            resp = await client.options(
                STATUS_PATH,
                headers={"Origin": ui, "Access-Control-Request-Method": "GET"},
            )
            assert resp.status == 204
            resp = await client.get(STATUS_PATH, headers={"Origin": ui})
            assert resp.headers["Access-Control-Allow-Origin"] == ui
            # The 503 gate is stamped as well: the pointer must be readable.
            resp = await client.get("/api/nodes", headers={"Origin": ui})
            assert resp.status == 503
            assert resp.headers["Access-Control-Allow-Origin"] == ui
        finally:
            await client.close()
            await engine_server.close()

    asyncio.run(scenario())


def test_proxy_websocket_events() -> None:
    """The /api/events WebSocket pumps both directions, text and binary -
    the one non-plain-HTTP channel the frontend consumes."""

    async def scenario() -> None:
        engine_server = TestServer(make_engine_app())
        await engine_server.start_server()
        link = EngineLink(state="ready", base_url=str(engine_server.make_url("")).rstrip("/"))
        client = TestClient(TestServer(create_supervisor_app(link)))
        await client.start_server()
        try:
            ws = await client.ws_connect("/api/events")
            message = await ws.receive(timeout=5)
            assert message.type == WSMsgType.TEXT and message.data == "hello"
            await ws.send_str("ping")
            message = await ws.receive(timeout=5)
            assert message.type == WSMsgType.TEXT and message.data == "echo:ping"
            await ws.send_bytes(b"\x01\x02\x03")
            message = await ws.receive(timeout=5)
            assert message.type == WSMsgType.BINARY and message.data == b"\x03\x02\x01"
            await ws.close()
        finally:
            await client.close()
            await engine_server.close()

    asyncio.run(scenario())


STUB_ENGINE = textwrap.dedent(
    """
    import argparse
    from aiohttp import web

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    async def health(request):
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_get("/api/health", health)
    web.run_app(app, host=args.host, port=args.port, print=None)
    """
)


async def wait_for(predicate, timeout: float = 30.0) -> None:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        await asyncio.sleep(0.05)


def test_engine_process_lifecycle(tmp_path: Path) -> None:
    """EngineProcess speaks protocol v1 against a real subprocess: injected
    --host/--port, health-probe readiness, terminate-on-close."""
    script = tmp_path / "stub_engine.py"
    script.write_text(STUB_ENGINE)

    async def scenario() -> None:
        link = EngineLink()
        engine = EngineProcess([sys.executable, str(script)], link, probe_interval=0.05)
        await engine.start()
        assert link.pid is not None and link.base_url.startswith("http://127.0.0.1:")
        await wait_for(lambda: link.state == "ready")
        await engine.close()
        assert link.state == "stopped" and link.pid is None

    asyncio.run(scenario())


def test_progress_from_health_shapes() -> None:
    """The health-body composition narration is mirrored to clients
    verbatim, so the supervisor is strict about its shape: anything
    malformed degrades to "no progress", never to a malformed status
    wire, and only the contracted keys cross."""
    from dinkster_supervisor.supervisor import _progress_from_health

    assert _progress_from_health({"ok": True}) is None
    assert _progress_from_health("nonsense") is None
    assert _progress_from_health({"composition": "packs"}) is None
    assert _progress_from_health({"composition": {"done": 1}}) is None
    assert _progress_from_health({"composition": {"done": True, "total": 2}}) is None
    assert _progress_from_health({"composition": {"done": "1", "total": 2}}) is None

    assert _progress_from_health({"composition": {"done": 1, "total": 2}}) == {
        "done": 1,
        "total": 2,
    }
    narrated = {"composition": {"done": 0, "total": 3, "phase": "packs"}}
    assert _progress_from_health(narrated) == {"done": 0, "total": 3, "phase": "packs"}
    # Empty phase is dropped; uncontracted keys never leak through.
    assert _progress_from_health(
        {"composition": {"done": 1, "total": 2, "phase": "", "extra": "x"}}
    ) == {"done": 1, "total": 2}


def test_status_narrates_composition_progress() -> None:
    """/supervisor/status carries "progress" exactly while the link holds
    mirrored composition narration - the settled frontend shape - and
    omits it (never null) otherwise."""

    async def scenario() -> None:
        link = EngineLink()
        link.state = "ready"
        link.progress = {"done": 1, "total": 3, "phase": "packs"}
        client = TestClient(TestServer(create_supervisor_app(link)))
        await client.start_server()
        try:
            status = await (await client.get(STATUS_PATH)).json()
            assert status["state"] == "ready"
            assert status["progress"] == {"done": 1, "total": 3, "phase": "packs"}
            link.progress = None
            status = await (await client.get(STATUS_PATH)).json()
            assert "progress" not in status
        finally:
            await client.close()

    asyncio.run(scenario())


# A stub engine that narrates in-flight composition until a release file
# appears next to it - deterministic transitions for the mirror test.
STUB_COMPOSING_ENGINE = textwrap.dedent(
    """
    import argparse
    from pathlib import Path

    from aiohttp import web

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    release = Path(__file__).parent / "release"

    async def health(request):
        if not release.exists():
            return web.json_response(
                {
                    "ok": True,
                    "composition": {"done": 1, "total": 2, "phase": "packs"},
                }
            )
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_get("/api/health", health)
    web.run_app(app, host=args.host, port=args.port, print=None)
    """
)


def test_engine_process_mirrors_composition_progress(tmp_path: Path) -> None:
    """Progressive startup through the supervisor's eyes: the engine is
    READY at its first healthy answer (the diagnostic host serves while packs
    compose - clients must not wait), composition narration mirrors onto
    the link while the health body carries it, and clears once composed."""
    script = tmp_path / "stub_composing.py"
    script.write_text(STUB_COMPOSING_ENGINE)

    async def scenario() -> None:
        link = EngineLink()
        engine = EngineProcess([sys.executable, str(script)], link, probe_interval=0.05)
        await engine.start()
        # Ready arrives WITH the narration, not after it.
        await wait_for(lambda: link.state == "ready")
        await wait_for(lambda: link.progress == {"done": 1, "total": 2, "phase": "packs"})
        (tmp_path / "release").touch()
        await wait_for(lambda: link.progress is None)
        assert link.state == "ready"
        await engine.close()
        assert link.state == "stopped" and link.progress is None

    asyncio.run(scenario())


def test_engine_process_startup_failure(tmp_path: Path) -> None:
    """An engine that dies before becoming healthy flips the link to
    "failed" with the exit code - the status surface narrates the crash
    instead of the port silently never opening (the ComfyUI failure)."""
    script = tmp_path / "dies.py"
    script.write_text("import sys; sys.exit(3)\n")

    async def scenario() -> None:
        link = EngineLink()
        engine = EngineProcess([sys.executable, str(script)], link, probe_interval=0.05)
        await engine.start()
        await wait_for(lambda: link.state == "failed")
        assert link.exit_code == 3
        assert "exit code 3" in link.detail
        await engine.close()

    asyncio.run(scenario())
