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
from typing import Any, cast

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


def test_an_engine_stopped_elsewhere_reads_stopped_once_its_record_says_so(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "installation"
        engine = TestServer(make_engine("main"))
        await engine.start_server()

        async def start(target: OpenTarget, found: Callable[[str], None]) -> str:
            return str(engine.make_url("")).rstrip("/")

        # The engine's listener closes before its record says stopped.
        checks: list[bool] = []

        def stopped(target: OpenTarget) -> bool:
            checks.append(len(checks) >= 2)
            return checks[-1]

        home = Home(
            installations=lambda: frozenset({root.resolve()}),
            start=start,
            stopped=stopped,
            watch_interval=0.02,
        )
        client = TestClient(TestServer(create_home_app(home, frontend=None)))
        await client.start_server()
        try:
            await client.post("/supervisor/engine", json={"root": str(root)})
            await wait_for(client, lambda s: s["state"] == "ready")
            await engine.close()
            status = await wait_for(client, lambda s: s["state"] == "stopped")
            assert "detail" not in status and checks == [False, False, True]
        finally:
            await client.close()

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
            assert await refusal() == "No installation is set up on this machine."
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


def test_shutdown_stops_every_engine_it_opened_then_exits(tmp_path: Path) -> None:
    async def scenario() -> None:
        first, second = (tmp_path / name for name in ("first", "second"))
        engines = {first.resolve(): TestServer(make_engine("first"))}
        engines[second.resolve()] = TestServer(make_engine("second"))
        for engine in engines.values():
            await engine.start_server()

        async def start(target: OpenTarget, found: Callable[[str], None]) -> str:
            return str(engines[target.root].make_url("")).rstrip("/")

        stops: list[OpenTarget] = []

        def stop(target: OpenTarget) -> None:
            stops.append(target)
            if target.root == second.resolve():
                raise RuntimeError("engine second did not stop")

        exits: list[str] = []
        home = Home(
            installations=lambda: frozenset(engines), start=start, stop=stop, watch_interval=0.02
        )
        client = TestClient(
            TestServer(create_home_app(home, frontend=None, exit=lambda: exits.append("exit")))
        )
        await client.start_server()
        try:
            for root in (first, second, first):
                await client.post("/supervisor/engine", json={"root": str(root)})
                await wait_for(client, lambda s: s["state"] == "ready")
            assert exits == []
            async with client.ws_connect("/api/events") as ws:
                assert (await ws.receive()).data == "hello from first"
                response = await client.post("/supervisor/shutdown")
                assert (await ws.receive()).type in {WSMsgType.CLOSE, WSMsgType.CLOSED}
            assert response.status == 200
            assert await response.json() == {
                "status": "partial",
                "stopped": [{"root": str(first.resolve()), "engine": "main"}],
                "failed": [
                    {
                        "root": str(second.resolve()),
                        "engine": "main",
                        "message": "engine second did not stop",
                    }
                ],
            }
            # Each opened engine is stopped once, however often it was opened.
            assert [target.root for target in stops] == [first.resolve(), second.resolve()]
            await asyncio.sleep(0)
            assert exits == ["exit"]
            assert (await client.get("/api/nodes")).status == 503
        finally:
            await client.close()
            for engine in engines.values():
                await engine.close()

    run_scenario(scenario())


def helper_holds(lock: Path) -> bool:
    """Whether a running library helper holds this helper lock."""
    from filelock import FileLock, Timeout

    probe = FileLock(lock)
    try:
        probe.acquire(timeout=0)
    except Timeout:
        return True
    probe.release()
    return False


def observation(
    project: str, digest: str, created: float, origin: str = "produced"
) -> dict[str, object]:
    """A library asset observation of one image in a project."""
    return {
        "id": digest[-32:],
        "project": project,
        "originalName": f"{project}-{created:g}.png",
        "kind": "image/output",
        "origin": {"kind": "produced", "job": "j", "node": "n"}
        if origin == "produced"
        else {"kind": origin},
        "created": created,
        "asset": {
            "digest": digest,
            "name": f"{project}-{created:g}.png",
            "size": 3,
            "mediaType": "image/png",
        },
    }


def test_projects_and_outputs_answer_from_the_library_until_an_engine_is_ready(
    tmp_path: Path,
) -> None:
    from dinkster_library import LibraryClient

    library = tmp_path / "library"
    produced = observation

    async def scenario() -> None:
        root = tmp_path / "installation"
        seen: list[dict[str, str]] = []
        engine = TestServer(make_engine("main", seen))
        await engine.start_server()

        held: list[bool] = []

        async def start(target: OpenTarget, found: Callable[[str], None]) -> str:
            # An engine's start refuses a library another helper holds.
            held.append(any(helper_holds(lock) for lock in library.glob("helper-*.lock")))
            return str(engine.make_url("")).rstrip("/")

        home = Home(installations=lambda: frozenset({root.resolve()}), start=start)
        client = TestClient(TestServer(create_home_app(home, frontend=None, library=library)))
        await client.start_server()
        try:
            created = await client.post("/api/projects", json={"id": "film", "name": "Film"})
            assert created.status == 200
            film = (await created.json())["project"]
            assert film["id"] == "film" and film["name"] == "Film" and film["revision"] == 1
            await client.post("/api/projects", json={"id": "music"})

            listed = await (await client.get("/api/projects")).json()
            assert [item["id"] for item in listed["items"]] == ["default", "film", "music"]

            # Rename, with the engine's revision check and status codes.
            renamed = await client.patch(
                "/api/projects/film", json={"revision": 1, "patch": {"name": "Short film"}}
            )
            assert renamed.status == 200
            assert (await renamed.json())["project"]["name"] == "Short film"
            stale = await client.patch(
                "/api/projects/film", json={"revision": 1, "patch": {"name": "Again"}}
            )
            assert stale.status == 409
            assert (await client.get("/api/projects/absent/outputs")).status == 404
            needs = await client.get("/api/projects/film/needs")
            assert needs.status == 200 and "installed" not in await needs.json()
            deleted = await client.delete("/api/projects/music", json={"revision": 1})
            assert deleted.status == 200 and (await deleted.json())["project"]["deleted"]
            listed = await (await client.get("/api/projects")).json()
            assert [item["id"] for item in listed["items"]] == ["default", "film"]

            # Latest outputs of one project, newest first, and not another's.
            with LibraryClient(library) as helper:
                helper.call("projects.register", id="other")
                for project, digest, at in (
                    ("film", "blake3:" + "1" * 64, 10.0),
                    ("film", "blake3:" + "2" * 64, 30.0),
                    ("other", "blake3:" + "3" * 64, 20.0),
                ):
                    helper.call(
                        "assets.record", scope="local", observation=produced(project, digest, at)
                    )
            outputs = await (await client.get("/api/projects/film/outputs")).json()
            assert [row["asset"]["digest"] for row in outputs["assets"]] == [
                "blake3:" + "2" * 64,
                "blake3:" + "1" * 64,
            ]
            workflows = await client.get(
                "/api/library",
                params={
                    "scope": "local",
                    "project": "film",
                    "mediaType": "application/x-dinkster-workflow+json",
                },
            )
            assert workflows.status == 200 and (await workflows.json())["records"] == []

            # Once an engine is ready the same routes are the engine's.
            await client.post("/supervisor/engine", json={"root": str(root)})
            await wait_for(client, lambda s: s["state"] == "ready")
            changed = await client.post("/api/change", json={"x": 1})
            assert (await changed.json())["engine"] == "main"
            assert (await client.get("/api/projects")).status == 404
            assert held == [False]
        finally:
            await client.close()
            await engine.close()

    run_scenario(scenario())


def test_the_engine_and_the_process_answer_a_projects_outputs_alike(tmp_path: Path) -> None:
    from dinkster_library import LibraryClient
    from dinkster_server.projects import add_project_routes

    library = tmp_path / "library"
    library.mkdir()
    with LibraryClient(library) as helper:
        helper.call("projects.register", id="film")
        helper.call("projects.register", id="other")
        for project, digest, at, origin in (
            ("film", "blake3:" + "1" * 64, 10.0, "produced"),
            ("film", "blake3:" + "2" * 64, 40.0, "upload"),
            ("film", "blake3:" + "3" * 64, 30.0, "produced"),
            ("film", "blake3:" + "4" * 64, 20.0, "produced"),
            ("other", "blake3:" + "5" * 64, 50.0, "produced"),
        ):
            helper.call(
                "assets.record", scope="local", observation=observation(project, digest, at, origin)
            )

    async def scenario() -> None:
        engine_app = web.Application()
        add_project_routes(engine_app, library, engine_version="0.0.17")
        home = Home(installations=frozenset)
        home_app = create_home_app(home, frontend=None, library=library)
        async with (
            TestClient(TestServer(engine_app)) as engine,
            TestClient(TestServer(home_app)) as process,
        ):
            answers = []
            for client in (engine, process):
                answer = {}
                for path in (
                    "/api/projects/film/outputs",
                    "/api/projects/film/outputs?limit=1",
                    "/api/projects/film/outputs?limit=x",
                    "/api/projects/absent/outputs",
                ):
                    response = await client.get(path)
                    answer[path] = (response.status, await response.json())
                answers.append(answer)
            engine_answer, process_answer = answers
            assert process_answer == engine_answer
            status, outputs = engine_answer["/api/projects/film/outputs"]
            assert status == 200
            # What the project's runs produced, newest first: not the upload, not another project's.
            assert [row["asset"]["digest"] for row in outputs["assets"]] == [
                "blake3:" + "3" * 64,
                "blake3:" + "4" * 64,
                "blake3:" + "1" * 64,
            ]
            status, first = engine_answer["/api/projects/film/outputs?limit=1"]
            assert [row["asset"]["digest"] for row in first["assets"]] == ["blake3:" + "3" * 64]
            assert engine_answer["/api/projects/film/outputs?limit=x"][0] == 400
            assert engine_answer["/api/projects/absent/outputs"][0] == 404

    run_scenario(scenario())


def test_open_starts_a_managed_snapshot_registered_in_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_supervisor.environment_runtime import engine_status, stop_engine
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
        exits: list[None] = []
        app = create_home_app(
            Home(watch_interval=0.05), frontend=None, exit=lambda: exits.append(None)
        )
        client = TestClient(TestServer(app))
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

            # A deliberate stop of the open engine reads as stopped, not as a failure.
            await asyncio.to_thread(stop_engine, root, "main")
            status = await wait_for(client, lambda s: s["state"] == "stopped")
            assert "detail" not in status, status
            assert status["open"] == {"root": str(root.resolve()), "engine": "main"}

            # Shutdown stops what this process opened, even an engine it restarted.
            assert (await client.post("/supervisor/engine/restart")).status == 200
            await wait_for(client, lambda s: s["state"] == "ready")
            answer = await (await client.post("/supervisor/shutdown")).json()
            assert answer == {
                "status": "stopped",
                "stopped": [{"root": str(root.resolve()), "engine": "main"}],
                "failed": [],
            }
            await asyncio.sleep(0)
            assert exits == [None]
            rows = await asyncio.to_thread(engine_status, root)
            assert [row["state"] for row in rows if row["name"] == "main"] == ["stopped"]
        finally:
            await client.close()

    try:
        run_scenario(scenario())
    finally:
        stop_engine(root, "main")


def test_projects_page_data_answers_from_the_library_with_no_engine(tmp_path: Path) -> None:
    import io
    from types import SimpleNamespace

    from dinkster_library import LibraryClient
    from dinkster_library.asset_identity import digest_bytes
    from dinkster_server.library import LIBRARY_KEY, handle_asset_preview
    from dinkster_server.projects import add_project_routes
    from PIL import Image

    library = tmp_path / "library"
    library.mkdir()

    def store(data: bytes) -> str:
        digest = digest_bytes(data)
        held = library / "vault" / digest[7:9] / digest[7:]
        held.parent.mkdir(parents=True, exist_ok=True)
        held.write_bytes(data)
        return digest

    def image(size: tuple[int, int], frames: tuple[str, ...] = ("red",)) -> bytes:
        pictures = [Image.new("RGB", size, color) for color in frames]
        out = io.BytesIO()
        pictures[0].save(
            out, "GIF" if len(frames) > 1 else "PNG", save_all=True, append_images=pictures[1:]
        )
        return out.getvalue()

    def size_of(digest: str) -> int:
        return (library / "vault" / digest[7:9] / digest[7:]).stat().st_size

    wide = store(image((1000, 400)))
    small = store(image((100, 60)))
    animated = store(image((64, 32), ("red", "blue")))
    clip = store(b"\x00\x00\x00\x18ftypmp42 not an image")
    upload = store(image((40, 40)))
    elsewhere = store(image((30, 30), ("green",)))
    known = store(b'{"nodes": {"a": 1}}')
    changed = store(b'{"nodes": {"b": 2}}')
    model = store(b"model weights")
    pack = {"pack": "dinkster.example", "version": "1.0.0"}
    with LibraryClient(library) as helper:
        for project in ("film", "other", "gone"):
            helper.call("projects.register", id=project)
        for project, digest, at, origin in (
            ("film", wide, 10.0, "produced"),
            ("film", small, 20.0, "produced"),
            ("film", animated, 30.0, "produced"),
            ("film", clip, 40.0, "produced"),
            ("film", upload, 50.0, "upload"),
            ("film", small, 60.0, "upload"),
            ("gone", elsewhere, 70.0, "produced"),
        ):
            seen = observation(project, digest, at, origin)
            seen["id"] = f"{int(at):032x}"
            seen["asset"] = {**cast(dict, seen["asset"]), "size": size_of(digest)}
            helper.call("assets.record", scope="local", observation=seen)
        for project, digest, facts in (("film", known, True), ("film", changed, False)):
            record = helper.call(
                "records.create",
                scope="local",
                project=project,
                record={
                    "name": digest[-6:],
                    "digest": digest,
                    "mediaType": "application/x-dinkster-workflow+json",
                },
            )["record"]
            helper.call(
                "workflows.record",
                scope="local",
                id=record["id"],
                revision=0,
                project=project,
                # A workflow saved with no facts for its document is unknown.
                **(
                    {
                        "facts": {
                            "digest": digest,
                            "packs": [pack],
                            "models": [],
                            "machinePaths": False,
                        }
                    }
                    if facts
                    else {}
                ),
            )
        gone = helper.call("projects.get", id="gone")["project"]
        helper.call("projects.delete", id="gone", revision=gone["revision"])
    sizes = {
        digest: size_of(digest) for digest in (wide, small, animated, clip, upload, known, changed)
    }

    async def scenario() -> None:
        engine_app = web.Application()
        add_project_routes(engine_app, library, engine_version="0.0.18")
        held = {digest: library / "vault" / digest[7:9] / digest[7:] for digest in sizes}
        engine_app[LIBRARY_KEY] = cast(Any, SimpleNamespace(locate=held.get))
        engine_app.router.add_get("/api/assets/{digest}/preview", handle_asset_preview)
        home_app = create_home_app(Home(installations=frozenset), frontend=None, library=library)
        async with (
            TestClient(TestServer(engine_app)) as engine,
            TestClient(TestServer(home_app)) as process,
        ):
            # Each content film's outputs, imports and workflows refer to counts once;
            # four outputs, not its uploads, another project's output or a model.
            for client in (engine, process):
                usage = await client.get("/api/projects/film/usage")
                assert await usage.json() == {
                    "outputs": 4,
                    "bytes": sum(sizes.values()),
                }
                assert (await client.get("/api/projects/gone/usage")).status == 404

            previews: dict[str, tuple[int, bytes]] = {}
            for client in (engine, process):
                for digest in (wide, small, animated, clip):
                    response = await client.get(f"/api/assets/{digest}/preview")
                    body = await response.read() if response.status == 200 else b""
                    if client is engine:
                        previews[digest] = (response.status, body)
                    else:
                        assert (response.status, body) == previews[digest]
            for digest, size in ((wide, (512, 205)), (small, (100, 60)), (animated, (64, 32))):
                status, body = previews[digest]
                assert status == 200
                with Image.open(io.BytesIO(body)) as picture:
                    assert (picture.format, picture.size) == ("WEBP", size)
                    assert picture.convert("RGB").getpixel((5, 5))[0] > 200  # red first frame
            assert previews[clip][0] == 404

            # The process serves only a listed project's outputs and saved workflows.
            for digest in (upload, elsewhere, model):
                assert (await process.get(f"/api/assets/{digest}/preview")).status == 404
            document = await process.get(f"/api/assets/{known}")
            assert document.status == 200
            assert document.content_type == "application/x-dinkster-workflow+json"
            assert await document.read() == b'{"nodes": {"a": 1}}'
            for digest in (wide, model):
                assert (await process.get(f"/api/assets/{digest}")).status == 404
            cached = await process.get(
                f"/api/assets/{known}", headers={"If-None-Match": f'"{known}"'}
            )
            assert cached.status == 304

            needs = await process.get("/api/projects/film/needs")
            answer = await needs.json()
            assert answer["packs"] == [pack] and "installed" not in answer
            assert len(answer["unknown"]) == 1
            assert (await process.get("/api/projects/gone/needs")).status == 404

    run_scenario(scenario())
