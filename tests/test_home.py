"""dinkster-home: the always-on process serves the frontend with no engine and
passes everything else to the open installation's engine.

Engines are stand-in apps answering like a managed supervisor; the last test
opens a real managed snapshot through ``dinkster-env start``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path

import aiohttp
import pytest
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestClient, TestServer
from dinkster_supervisor.home import ENGINE_CHANGED, Home, OpenTarget, create_home_app

from tests.async_scenarios import run_scenario


def make_frontend(root: Path) -> Path:
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text("<!doctype html><title>Dinkster</title>")
    (root / "assets" / "index-abc.js").write_text("export const built = true\n")
    return root


def make_engine(
    name: str, seen: list[dict[str, str]] | None = None, supervisor: dict | None = None
) -> web.Application:
    """A stand-in for an engine's managed supervisor and the engine behind it."""
    app = web.Application()
    state = supervisor if supervisor is not None else {"state": "ready", "restarts": 0}

    async def status(request: web.Request) -> web.Response:
        body = {"protocol": 1, "state": state["state"]}
        if "detail" in state:
            body["detail"] = state["detail"]
        return web.json_response(body)

    async def restart(request: web.Request) -> web.Response:
        state.update(state="ready", restarts=state["restarts"] + 1)
        state.pop("detail", None)
        return web.json_response({"protocol": 1, "state": "ready"})

    async def nodes(request: web.Request) -> web.Response:
        return web.json_response({"engine": name})

    async def asset(request: web.Request) -> web.Response:
        return web.json_response({"engine": name, "asset": request.match_info["digest"]})

    async def change(request: web.Request) -> web.Response:
        if seen is not None:
            seen.append({"origin": request.headers.get("Origin", "")})
        return web.json_response({"engine": name, "body": await request.json()})

    async def events(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str(f"hello from {name}")
        async for _message in ws:
            pass
        return ws

    app.router.add_get("/supervisor/status", status)
    app.router.add_post("/supervisor/engine/restart", restart)
    app.router.add_get("/api/nodes", nodes)
    app.router.add_get("/assets/{digest}", asset)
    app.router.add_post("/api/change", change)
    app.router.add_get("/api/events", events)
    return app


async def wait_for(client: TestClient, predicate: Callable[[dict[str, object]], bool]) -> dict:
    status: dict = {}
    for _ in range(200):
        status = await (await client.get("/supervisor/status")).json()
        if predicate(status):
            return status
        await asyncio.sleep(0.02)
    raise AssertionError(f"status never matched: {status}")


def test_frontend_and_status_answer_with_no_engine(tmp_path: Path) -> None:
    async def scenario() -> None:
        frontend = make_frontend(tmp_path / "frontend")
        (tmp_path / "secret.txt").write_text("outside")
        home = Home(installations=frozenset)
        client = TestClient(TestServer(create_home_app(home, frontend=frontend)))
        await client.start_server()
        try:
            page = await client.get("/")
            assert page.status == 200 and "<title>Dinkster</title>" in await page.text()
            script = await client.get("/assets/index-abc.js")
            assert script.status == 200
            assert script.headers["Content-Type"].startswith("text/javascript")
            assert await script.text() == "export const built = true\n"
            assert (await client.get("/assets/..%2F..%2Fsecret.txt")).status == 503

            status = await (await client.get("/supervisor/status")).json()
            assert status["protocol"] == 1 and status["role"] == "home"
            assert status["state"] == "stopped" and "open" not in status
            assert status["installations"] == 0

            refused = await client.get("/api/nodes")
            assert refused.status == 503
            assert (await refused.json())["error"] == "engine-not-ready"
            with pytest.raises(aiohttp.WSServerHandshakeError) as handshake:
                await client.ws_connect("/api/events")
            assert handshake.value.status == 503
        finally:
            await client.close()

    run_scenario(scenario())


def test_open_engine_receives_the_api_and_events_until_another_opens(tmp_path: Path) -> None:
    async def scenario() -> None:
        frontend = make_frontend(tmp_path / "frontend")
        first_root, second_root = tmp_path / "first", tmp_path / "second"
        seen: list[dict[str, str]] = []
        engines = {
            first_root: TestServer(make_engine("first", seen)),
            second_root: TestServer(make_engine("second")),
        }
        for server in engines.values():
            await server.start_server()

        async def start(target: OpenTarget, found: Callable[[str], None]) -> str:
            return str(engines[target.root].make_url("")).rstrip("/")

        home = Home(
            installations=lambda: frozenset(path.resolve() for path in engines),
            start=start,
            watch_interval=0.02,
        )
        client = TestClient(TestServer(create_home_app(home, frontend=frontend)))
        await client.start_server()
        try:
            unknown = await client.post("/supervisor/engine", json={"root": str(tmp_path / "x")})
            assert unknown.status == 404
            assert (await (await client.get("/supervisor/status")).json())["state"] == "stopped"

            opened = await client.post("/supervisor/engine", json={"root": str(first_root)})
            assert opened.status == 200
            status = await wait_for(client, lambda s: s["state"] == "ready")
            assert status["open"] == {"root": str(first_root.resolve()), "engine": "main"}
            assert await (await client.get("/api/nodes")).json() == {"engine": "first"}
            # A built frontend file wins over an engine path of the same shape.
            assert await (await client.get("/assets/index-abc.js")).text() == (
                "export const built = true\n"
            )
            assert await (await client.get("/assets/0123abcd")).json() == {
                "engine": "first",
                "asset": "0123abcd",
            }

            own = str(client.make_url("")).rstrip("/")
            changed = await client.post("/api/change", json={"a": 1}, headers={"Origin": own})
            assert await changed.json() == {"engine": "first", "body": {"a": 1}}
            assert seen == [{"origin": str(engines[first_root].make_url("")).rstrip("/")}]
            hostile = await client.post(
                "/api/change", json={}, headers={"Origin": "https://evil.example"}
            )
            assert hostile.status == 403 and len(seen) == 1

            socket = await client.ws_connect("/api/events")
            assert (await socket.receive(timeout=5)).data == "hello from first"

            await client.post("/supervisor/engine", json={"root": str(second_root)})
            closed = await socket.receive(timeout=5)
            assert closed.type == WSMsgType.CLOSE and closed.data == ENGINE_CHANGED
            status = await wait_for(client, lambda s: s["state"] == "ready")
            assert status["open"]["root"] == str(second_root.resolve())
            assert await (await client.get("/api/nodes")).json() == {"engine": "second"}
            socket = await client.ws_connect("/api/events")
            assert (await socket.receive(timeout=5)).data == "hello from second"
            await socket.close()
        finally:
            await client.close()
            for server in engines.values():
                await server.close()

    run_scenario(scenario())


def test_failed_start_is_reported_and_restart_opens_it_again(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "installation"
        supervisor: dict = {"state": "ready", "restarts": 0}
        engine = TestServer(make_engine("main", supervisor=supervisor))
        await engine.start_server()
        attempts: list[OpenTarget] = []

        async def start(target: OpenTarget, found: Callable[[str], None]) -> str:
            attempts.append(target)
            if len(attempts) == 1:
                raise RuntimeError("engine main failed before becoming ready")
            return str(engine.make_url("")).rstrip("/")

        home = Home(
            installations=lambda: frozenset({root.resolve()}), start=start, watch_interval=0.02
        )
        client = TestClient(TestServer(create_home_app(home, frontend=None)))
        await client.start_server()
        try:
            await client.post("/supervisor/engine", json={"root": str(root), "engine": "main"})
            status = await wait_for(client, lambda s: s["state"] == "failed")
            assert status["detail"] == "engine main failed before becoming ready"
            assert status["open"]["root"] == str(root.resolve())
            assert (await (await client.get("/api/nodes")).json())["state"] == "failed"

            assert (await client.post("/supervisor/engine/restart")).status == 200
            await wait_for(client, lambda s: s["state"] == "ready")
            assert await (await client.get("/api/nodes")).json() == {"engine": "main"}

            # An engine that fails under its live supervisor restarts there, not anew.
            supervisor.update(state="failed", detail="engine exited (exit code 9)")
            status = await wait_for(client, lambda s: s["state"] == "failed")
            assert status["detail"] == "engine exited (exit code 9)"
            assert (await client.post("/supervisor/engine/restart")).status == 200
            await wait_for(client, lambda s: s["state"] == "ready")
            assert supervisor["restarts"] == 1 and len(attempts) == 2

            await engine.close()
            status = await wait_for(client, lambda s: s["state"] == "failed")
            assert status["detail"] == "the engine stopped answering"
            assert (await client.get("/api/nodes")).status == 503
        finally:
            await client.close()
            await engine.close()

    run_scenario(scenario())


def test_start_with_nothing_open_opens_the_default_installation_or_says_why(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        default, other, third = (tmp_path / name for name in ("project", "other", "third"))
        known: set[Path] = set()
        engine = TestServer(make_engine("main"))
        await engine.start_server()
        opened: list[Path] = []

        async def start(target: OpenTarget, found: Callable[[str], None]) -> str:
            opened.append(target.root)
            return str(engine.make_url("")).rstrip("/")

        home = Home(
            installations=lambda: frozenset(known),
            start=start,
            watch_interval=0.02,
            default_root=default,
        )
        client = TestClient(TestServer(create_home_app(home, frontend=None)))
        await client.start_server()

        async def refusal() -> str:
            response = await client.post("/supervisor/engine/restart")
            assert response.status == 409
            return (await response.json())["message"]

        try:
            assert "Run dinkster setup" in await refusal()
            known.update({other.resolve(), third.resolve()})
            assert (await (await client.get("/supervisor/status")).json())["installations"] == 2
            assert "none is the default" in await refusal()
            assert opened == []

            known.add(default.resolve())
            assert (await client.post("/supervisor/engine/restart")).status == 200
            status = await wait_for(client, lambda s: s["state"] == "ready")
            assert status["open"] == {"root": str(default.resolve()), "engine": "main"}
            assert status["installations"] == 3

            # With the default gone and one installation left, that one opens.
            fresh = Home(
                installations=lambda: frozenset({third.resolve()}),
                start=start,
                default_root=default,
            )
            await fresh.restart()
            assert fresh.target == OpenTarget(third.resolve(), "main")
            await fresh.close()
        finally:
            await client.close()
            await engine.close()

    run_scenario(scenario())


def test_open_starts_a_managed_snapshot_registered_in_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_supervisor.environment_runtime import stop_engine
    from test_environment_runtime import free_port, make_generation

    root = tmp_path / "installation"
    make_generation(root)
    (root / "current").write_text("1\n")
    (root / "environment.json").write_text(
        json.dumps(
            {
                "format": "dinkster.project/1",
                "channel": "nightly",
                "variant": "linux-cpu",
                "port": free_port(),
                "mirrors": [str(root / "mirror")],
            }
        )
    )
    store = tmp_path / "store"
    store.mkdir()
    (store / "projects.json").write_text(
        json.dumps({"format": "dinkster.store/1", "projects": [str(root)]})
    )
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "user-data"))
    (tmp_path / "user-data/dinkster").mkdir(parents=True)
    (tmp_path / "user-data/dinkster/stores.json").write_text(
        json.dumps({"format": "dinkster.stores/1", "volumes": {"1": str(store)}})
    )

    async def scenario() -> None:
        client = TestClient(TestServer(create_home_app(Home(watch_interval=0.05), frontend=None)))
        await client.start_server()
        try:
            await client.post("/supervisor/engine", json={"root": str(root)})
            status: dict = {}
            for _ in range(600):
                status = await (await client.get("/supervisor/status")).json()
                if status["state"] != "starting":
                    break
                await asyncio.sleep(0.05)
            assert status["state"] == "ready", status
            generation = await (await client.get("/api/generation")).json()
            assert generation == {"generation": str(root / "builds/1")}
        finally:
            await client.close()

    try:
        run_scenario(scenario())
    finally:
        stop_engine(root, "main")
