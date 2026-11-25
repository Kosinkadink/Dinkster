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
from dinkster_server import create_app
from dinkster_supervisor.home import ENGINE_CHANGED, Home, OpenTarget, create_home_app
from test_server import SCHEMAS
from test_server import make_engine as make_execution_engine

from tests.async_scenarios import run_scenario
from tests.platform_support import symlink_or_skip


def make_frontend(root: Path) -> Path:
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text("<!doctype html><title>Dinkster</title>")
    (root / "assets" / "index-abc.js").write_text("export const built = true\n", newline="\n")
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


async def wait_for(client: TestClient, predicate: Callable[[dict], bool]) -> dict:
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


def test_a_deleted_project_is_listed_and_restored_with_its_workflows_and_outputs(
    tmp_path: Path,
) -> None:
    from dinkster_library import LibraryClient

    library = tmp_path / "library"
    workflow = "application/x-dinkster-workflow+json"

    async def scenario() -> None:
        home = Home(installations=lambda: frozenset())
        client = TestClient(TestServer(create_home_app(home, frontend=None, library=library)))
        await client.start_server()
        try:
            assert (await client.post("/api/projects", json={"id": "film"})).status == 200
            with LibraryClient(library) as helper:
                saved = cast(
                    "dict[str, object]",
                    helper.call(
                        "records.create",
                        scope="local",
                        project="film",
                        record={
                            "name": "Grain",
                            "digest": "blake3:" + "4" * 64,
                            "mediaType": workflow,
                        },
                    )["record"],
                )
                helper.call(
                    "assets.record",
                    scope="local",
                    observation=observation("film", "blake3:" + "5" * 64, 10.0),
                )
            query = {"scope": "local", "project": "film", "mediaType": workflow}

            async def contents() -> tuple[int, object, int, object]:
                records = await client.get("/api/library", params=query)
                outputs = await client.get("/api/projects/film/outputs")
                return (
                    records.status,
                    await records.json(),
                    outputs.status,
                    await outputs.json(),
                )

            before = await contents()
            assert [
                row["id"]
                for row in cast("dict[str, list[dict[str, object]]]", before[1])["records"]
            ] == [saved["id"]]
            assert before[0] == before[2] == 200

            restore = "/api/projects/film/restore"
            refused = await client.post(restore, json={"revision": 1})
            assert refused.status == 400, await refused.text()
            deleted = await client.delete("/api/projects/film", json={"revision": 1})
            assert (await deleted.json())["project"]["deleted"] is True
            gone = await contents()
            assert gone[0] == 400 and gone[2] == 404
            listed = await (await client.get("/api/projects")).json()
            assert [item["id"] for item in listed["items"]] == ["default"]
            every = await (await client.get("/api/projects?includeDeleted=true")).json()
            assert {item["id"]: item["deleted"] for item in every["items"]} == {
                "default": False,
                "film": True,
            }

            assert (await client.post(restore, json={"revision": 1})).status == 409
            assert (
                await client.post("/api/projects/absent/restore", json={"revision": 1})
            ).status == 404
            restored = await client.post(restore, json={"revision": 2})
            assert restored.status == 200, await restored.text()
            project = (await restored.json())["project"]
            assert project["deleted"] is False and project["revision"] == 3
            assert await contents() == before
            assert (await client.post(restore, json={"revision": 3})).status == 400
        finally:
            await client.close()

    run_scenario(scenario())


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
        engine_app = create_app(make_execution_engine, SCHEMAS)
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


@pytest.mark.parametrize("assigned", [False, True])
def test_open_starts_a_managed_snapshot_registered_in_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, assigned: bool
) -> None:
    from dinkster_supervisor.environment_runtime import engine_status, start_engine, stop_engine
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
    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
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
            if assigned:
                await asyncio.to_thread(start_engine, root, 1, "main", free_port(), ("GPU-aB12",))
            opened = await client.post("/supervisor/engine", json={"root": str(root)})
            assert opened.status == 200, await opened.text()
            status: dict = {}
            for _ in range(600):
                status = await (await client.get("/supervisor/status")).json()
                if status["state"] != "starting":
                    break
                await asyncio.sleep(0.05)
            assert status["state"] == "ready", status
            generation = await (await client.get("/api/generation")).json()
            assert generation == {"generation": str(root / "builds/1")}
            rows = await asyncio.to_thread(engine_status, root)
            assert rows[0]["gpus"] == (["GPU-aB12"] if assigned else [])

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

    from dinkster_assets import AssetVault, LibraryStore
    from dinkster_library import LibraryClient
    from dinkster_library.asset_identity import digest_bytes
    from dinkster_server import ServerLibrary
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
            record = cast(
                dict[str, Any],
                helper.call(
                    "records.create",
                    scope="local",
                    project=project,
                    record={
                        "name": digest[-6:],
                        "digest": digest,
                        "mediaType": "application/x-dinkster-workflow+json",
                    },
                )["record"],
            )
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
        gone = cast(dict[str, Any], helper.call("projects.get", id="gone")["project"])
        helper.call("projects.delete", id="gone", revision=gone["revision"])
    sizes = {
        digest: size_of(digest) for digest in (wide, small, animated, clip, upload, known, changed)
    }

    async def scenario() -> None:
        engine_app = create_app(
            make_execution_engine,
            SCHEMAS,
            library=ServerLibrary(
                AssetVault(library / "vault"), LibraryStore(tmp_path / "engine-library.sqlite")
            ),
        )
        add_project_routes(engine_app, library, engine_version="0.0.18")
        held = {digest: library / "vault" / digest[7:9] / digest[7:] for digest in sizes}
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
                    red = cast(tuple[int, int, int], picture.convert("RGB").getpixel((5, 5)))
                    assert red[0] > 200  # the first frame
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

            # Recording an opening keeps the project's revision, so a rename read
            # before it still applies; only when it was last opened changes.
            before = (await (await process.get("/api/projects/film")).json())["project"]
            assert before["lastOpenedAt"] is None
            openings = []
            for client in (process, engine):
                opened = await client.post("/api/projects/film/opened")
                assert opened.status == 200
                openings.append((await opened.json())["project"])
            assert [item["revision"] for item in openings] == [before["revision"]] * 2
            assert openings[0]["lastOpenedAt"] <= openings[1]["lastOpenedAt"]
            assert {key: value for key, value in openings[1].items() if key != "lastOpenedAt"} == {
                key: value for key, value in before.items() if key != "lastOpenedAt"
            }
            renamed = await process.patch(
                "/api/projects/film",
                json={"revision": before["revision"], "patch": {"name": "Film, cut"}},
            )
            assert renamed.status == 200
            assert (await renamed.json())["project"]["lastOpenedAt"] == openings[1]["lastOpenedAt"]
            for client in (engine, process):
                listed = {
                    item["id"]: item["lastOpenedAt"]
                    for item in (await (await client.get("/api/projects")).json())["items"]
                }
                assert listed == {
                    "default": None,
                    "film": openings[1]["lastOpenedAt"],
                    "other": None,
                }
                assert (await client.post("/api/projects/gone/opened")).status == 404

            needs = await process.get("/api/projects/film/needs")
            answer = await needs.json()
            assert answer["packs"] == [pack] and "installed" not in answer
            assert len(answer["unknown"]) == 1
            assert (await process.get("/api/projects/gone/needs")).status == 404

            # A missing or unreadable preview leaves the output list available.
            held[wide].unlink()
            held[small].write_bytes(b"unreadable image")
            for client in (engine, process):
                for digest in (wide, small):
                    assert (await client.get(f"/api/assets/{digest}/preview")).status == 404
                outputs = await client.get("/api/projects/film/outputs?limit=6")
                assert outputs.status == 200
                assert len((await outputs.json())["assets"]) == 4

            # Action names are also valid project ids.
            for identifier in ("opened", "usage"):
                assert (await process.post("/api/projects", json={"id": identifier})).status == 200
                for client in (engine, process):
                    project = (await (await client.get(f"/api/projects/{identifier}")).json())[
                        "project"
                    ]
                    assert project["id"] == identifier and project["lastOpenedAt"] is None
                removed = await engine.delete(f"/api/projects/{identifier}", json={"revision": 1})
                assert removed.status == 200
                assert (await removed.json())["project"]["deleted"] is True

    run_scenario(scenario())


def test_a_projects_output_folder_is_listed_and_opened_only_on_this_machine(
    tmp_path: Path,
) -> None:
    from dinkster_library import LibraryClient
    from dinkster_server.projects import add_project_routes

    library = tmp_path / "library"
    library.mkdir()
    default, renders = tmp_path / "Dinkster output", tmp_path / "renders"
    default.mkdir()
    renders.mkdir()
    with LibraryClient(library) as helper:
        for project in ("film", "plain", "skips", "none"):
            helper.call("projects.register", id=project)

    def grant(path: Path) -> dict[str, object]:
        return {"path": str(path), "mode": "readwrite", "priority": 0, "kind": "image/output"}

    async def folders(client: TestClient) -> dict[str, object]:
        items = (await (await client.get("/api/projects")).json())["items"]
        return {item["id"]: item["folder"] for item in items if item["id"] != "default"}

    async def scenario() -> None:
        opened: list[Path] = []

        async def open_folder(folder: Path) -> None:
            opened.append(folder)

        @web.middleware
        async def peer(request: web.Request, handler: Any) -> web.StreamResponse:
            # A test-only stand-in for a request arriving from another machine.
            remote = request.headers.get("X-Test-Peer")
            return await handler(request.clone(remote=remote) if remote else request)

        async def status(request: web.Request) -> web.Response:
            return web.json_response({"protocol": 1, "state": "ready"})

        root = tmp_path / "installation"
        engine_app = create_app(make_execution_engine, SCHEMAS)
        engine_app.router.add_get("/supervisor/status", status)
        add_project_routes(engine_app, library, engine_version="0.0.18")
        engine_server = TestServer(engine_app)

        async def start(target: OpenTarget, found: Callable[[str], None]) -> str:
            return str(engine_server.make_url("")).rstrip("/")

        home = Home(installations=lambda: frozenset({root.resolve()}), start=start)
        home_app = create_home_app(home, frontend=None, library=library, open_folder=open_folder)
        home_app.middlewares.insert(0, peer)
        elsewhere = create_home_app(Home(installations=frozenset), frontend=None, library=library)
        async with (
            TestClient(engine_server) as engine,
            TestClient(TestServer(home_app)) as process,
            TestClient(TestServer(elsewhere)) as headless,
            engine.ws_connect("/api/events") as events,
        ):
            assert (await wait_for(process, lambda s: True))["openFolder"] is True
            assert (await wait_for(headless, lambda s: True))["openFolder"] is False
            remote = await process.get("/supervisor/status", headers={"X-Test-Peer": "192.0.2.7"})
            assert (await remote.json())["openFolder"] is False

            # With no grants a project's outputs have nowhere to go.
            for client in (engine, process):
                assert await folders(client) == dict.fromkeys(("film", "plain", "skips", "none"))
            refused = await process.post("/supervisor/projects/film/folder")
            assert (refused.status, (await refused.json())["error"]) == (404, "no-folder")

            # Before any project decides, every project writes to the writable
            # grant named "output", which the engine shares on start.
            with LibraryClient(library) as helper:
                helper.call(
                    "folders.grants.patch",
                    revision=0,
                    patch={"mounts": {"output": grant(default), "renders": grant(renders)}},
                )
            for client in (engine, process):
                assert await folders(client) == dict.fromkeys(
                    ("film", "plain", "skips", "none"), str(default)
                )

            # A project's own choice wins over the default output grant; a project
            # that excludes the default, or chose none, has no output folder.
            with LibraryClient(library) as helper:
                helper.call(
                    "folders.grants.patch",
                    revision=1,
                    patch={"settings": {"output-mount": "renders"}},
                )
                helper.call(
                    "folders.projects.patch",
                    revision=0,
                    patch={
                        "version": 1,
                        "shared": ["output", "renders"],
                        "projects": {},
                        "excluded": {"skips": ["renders"]},
                        "outputs": {"film": "output", "none": None},
                    },
                )
            expected = {"film": str(default), "plain": str(renders), "skips": None, "none": None}
            for client in (engine, process):
                assert await folders(client) == expected
                film = (await (await client.get("/api/projects/film")).json())["project"]
                assert film["folder"] == str(default)

            # The process opens the folder its record names, never one the request names.
            shown = await process.post(
                "/supervisor/projects/plain/folder", json={"folder": str(default)}
            )
            assert (shown.status, await shown.json()) == (200, {"folder": str(renders)})
            assert opened == [renders]
            for project, error in (("skips", "no-folder"), ("absent", "no-folder")):
                answer = await process.post(f"/supervisor/projects/{project}/folder")
                assert (answer.status, (await answer.json())["error"]) == (404, error)
            for client, headers in ((headless, {}), (process, {"X-Test-Peer": "192.0.2.7"})):
                answer = await client.post("/supervisor/projects/plain/folder", headers=headers)
                assert (answer.status, (await answer.json())["error"]) == (404, "no-file-manager")
            renders.rmdir()
            gone = await process.post("/supervisor/projects/plain/folder")
            assert (gone.status, (await gone.json())["error"]) == (404, "no-folder")
            with LibraryClient(library) as helper:
                film = cast(dict[str, Any], helper.call("projects.get", id="film")["project"])
                helper.call("projects.delete", id="film", revision=film["revision"])
            deleted = await process.post("/supervisor/projects/film/folder")
            assert (deleted.status, (await deleted.json())["error"]) == (404, "no-folder")
            assert opened == [renders]

            # With an engine open the record comes from the engine.
            renders.mkdir()
            await process.post("/supervisor/engine", json={"root": str(root)})
            await wait_for(process, lambda s: s["state"] == "ready")
            shown = await process.post("/supervisor/projects/plain/folder")
            assert (shown.status, await shown.json()) == (200, {"folder": str(renders)})
            assert opened == [renders, renders]
            async with asyncio.timeout(5):
                while (await events.receive_json())["type"] != "library_changed":
                    pass

    run_scenario(scenario())


def make_frontend_installation(
    root: Path, *, wire: int, bundle: Path | None, development: dict | None = None
) -> Path:
    """A snapshot whose own interpreter reports ``wire`` and serves ``bundle``."""
    from test_environment_runtime import make_generation

    make_generation(root)
    # The snapshot's path entries put these ahead of the checkout's own packages.
    (root / "dinkster_schema").mkdir()
    (root / "dinkster_schema/__init__.py").write_text("")
    (root / "dinkster_schema/wire.py").write_text(f"SCHEMA_WIRE_VERSION = {wire}\n")
    found = "None" if bundle is None else f"Path({str(bundle)!r})"
    (root / "dinkster/frontend.py").write_text(
        f"from pathlib import Path\ndef discover_frontend_bundle():\n    return {found}\n"
    )
    if development is not None:
        (root / "builds/1/development-sources.json").write_text(json.dumps(development))
    return root.resolve()


def test_installations_report_their_wire_version_and_own_frontend(tmp_path: Path) -> None:
    async def scenario() -> None:
        bundle = make_frontend(tmp_path / "bundle")
        folder = make_frontend(tmp_path / "folder")
        same = make_frontend_installation(tmp_path / "same", wire=1, bundle=bundle)
        newer = make_frontend_installation(tmp_path / "newer", wire=2, bundle=bundle)
        developed = make_frontend_installation(
            tmp_path / "developed", wire=1, bundle=bundle, development={"frontendRoot": str(folder)}
        )
        bare = make_frontend_installation(tmp_path / "bare", wire=1, bundle=None)
        unset = tmp_path / "unset"
        unset.mkdir()
        roots = frozenset({same, newer, developed, bare, unset.resolve()})
        client = TestClient(
            TestServer(create_home_app(Home(installations=lambda: roots), frontend=None))
        )
        await client.start_server()

        async def facts() -> dict[str, dict]:
            answer = await (await client.get("/supervisor/installations")).json()
            return {Path(row["root"]).name: row["frontend"] for row in answer["installations"]}

        try:
            assert await facts() == {
                "same": {"wire": 1, "selected": None, "available": True},
                "newer": {"wire": 2, "selected": None, "available": True},
                "developed": {"wire": 1, "selected": "folder", "available": True},
                "bare": {"wire": 1, "selected": None, "available": False},
                "unset": {"wire": None, "selected": None, "available": False},
            }
            # A published frontend chosen beside the engine, then a Git frontend.
            (same / "builds/1/artifact-sources.json").write_text(
                json.dumps({"channel": {"frontend": {"version": "0.1.1.dev9"}}})
            )
            assert (await facts())["same"]["selected"] == "version"
            manifest = json.loads((same / "builds/1/engine-manifest.json").read_text())
            manifest["sources"] = {"frontend": {"url": "https://github.com/x/y"}}
            (same / "builds/1/engine-manifest.json").write_text(json.dumps(manifest))
            assert (await facts())["same"]["selected"] == "git"
        finally:
            await client.close()

    run_scenario(scenario())


def test_installation_frontend_address_serves_the_open_installations_own_frontend(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        from dinkster_supervisor.home import Frontends
        from dinkster_supervisor.home_frontend import FrontendSource

        shared = make_frontend(tmp_path / "shared")
        (shared / "index.html").write_text("<title>shared</title>")
        bundle = make_frontend(tmp_path / "bundle")
        (bundle / "index.html").write_text("<title>bundle</title>")
        folder = make_frontend(tmp_path / "folder")
        (folder / "index.html").write_text("<title>folder</title>")
        # An installed bundle can be reached through a link, as a site-packages folder can.
        linked = tmp_path / "bundle-link"
        symlink_or_skip(linked, bundle, target_is_directory=True)
        newer = make_frontend_installation(tmp_path / "newer", wire=2, bundle=linked)
        developed = make_frontend_installation(
            tmp_path / "developed", wire=1, bundle=bundle, development={"frontendRoot": str(folder)}
        )
        engine = TestServer(make_engine("engine"))
        await engine.start_server()

        async def start(target: OpenTarget, found: Callable[[str], None]) -> str:
            return str(engine.make_url("")).rstrip("/")

        home = Home(
            installations=lambda: frozenset({newer, developed}), start=start, watch_interval=0.02
        )
        frontends = Frontends(FrontendSource("folder", str(shared)), "http://127.0.0.1:1")
        main = TestClient(TestServer(create_home_app(home, frontend=frontends)))
        own = TestClient(TestServer(create_home_app(home, frontend=frontends, installation=True)))
        await main.start_server()
        await own.start_server()
        try:
            assert (await own.get("/")).status == 503
            status = await (await own.get("/supervisor/status")).json()
            assert status["installationFrontend"] == "http://127.0.0.1:1"
            assert status["frontend"] == str(shared) and status["frontendKind"] == "folder"

            await main.post("/supervisor/engine", json={"root": str(developed)})
            await wait_for(main, lambda s: s["state"] == "ready")
            assert "<title>folder</title>" in await (await own.get("/")).text()
            assert "<title>shared</title>" in await (await main.get("/")).text()
            assert await (await own.get("/api/nodes")).json() == {"engine": "engine"}

            await main.post("/supervisor/engine", json={"root": str(newer)})
            await wait_for(main, lambda s: s["open"]["root"] == str(newer))
            assert "<title>bundle</title>" in await (await own.get("/")).text()
            script = await own.get("/assets/index-abc.js")
            assert script.headers["Content-Type"].startswith("text/javascript")
        finally:
            await main.close()
            await own.close()
            await engine.close()

    run_scenario(scenario())


def test_an_installation_frontend_port_that_is_an_engine_port_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from dinkster_supervisor import home

    monkeypatch.setattr(home, "engine_ports", lambda: {3639: Path("/installations/nightly")})
    with pytest.raises(SystemExit) as refused:
        home.main(["--port", "3700", "--installation-frontend-port", "3639"])
    assert refused.value.code == 2
    assert f"3639 is the engine port of {Path('/installations/nightly')}" in capsys.readouterr().err


def test_development_frontend_serves_everything_but_the_engine(tmp_path: Path) -> None:
    from dinkster_supervisor.home import Frontends, shared_frontend

    with pytest.raises(RuntimeError, match="allow-frontend-dev-remote"):
        shared_frontend(None, "http://192.0.2.7:5199")
    assert shared_frontend(None, "http://192.0.2.7:5199/", allow_remote=True) is not None

    async def scenario() -> None:
        dev = web.Application()

        async def page(request: web.Request) -> web.StreamResponse:
            if request.headers.get("Upgrade", "").lower() == "websocket":
                ws = web.WebSocketResponse(protocols=("vite-hmr",))
                await ws.prepare(request)
                await ws.send_str("connected")
                async for message in ws:
                    await ws.send_str("echo " + message.data)
                return ws
            return web.Response(text=f"dev {request.path_qs}", content_type="text/html")

        dev.router.add_route("*", "/{tail:.*}", page)
        server = TestServer(dev)
        await server.start_server()
        engine = TestServer(make_engine("engine"))
        await engine.start_server()

        async def start(target: OpenTarget, found: Callable[[str], None]) -> str:
            return str(engine.make_url("")).rstrip("/")

        root = (tmp_path / "installation").resolve()
        home = Home(installations=lambda: frozenset({root}), start=start, watch_interval=0.02)
        source = shared_frontend(None, str(server.make_url("")))
        client = TestClient(TestServer(create_home_app(home, frontend=Frontends(source))))
        await client.start_server()
        try:
            assert await (await client.get("/")).text() == "dev /"
            assert await (await client.get("/src/main.tsx?t=1")).text() == "dev /src/main.tsx?t=1"
            # Engine paths never reach the development server.
            assert (await client.get("/api/nodes")).status == 503
            await client.post("/supervisor/engine", json={"root": str(root)})
            await wait_for(client, lambda s: s["state"] == "ready")
            assert await (await client.get("/api/nodes")).json() == {"engine": "engine"}
            status = await (await client.get("/supervisor/status")).json()
            assert status["frontendKind"] == "dev"

            reload = await client.ws_connect("/", protocols=("vite-hmr",))
            assert reload.protocol == "vite-hmr"
            assert (await reload.receive(timeout=5)).data == "connected"
            await reload.send_str("ping")
            assert (await reload.receive(timeout=5)).data == "echo ping"
            await reload.close()
        finally:
            await client.close()
            await engine.close()
            await server.close()

    run_scenario(scenario())


def recorded_table(snapshot: str) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "dinkster": {"version": "1.2.3", "schemaWire": 2},
        "packs": {
            "core": {"displayName": "Core"},
            "kept": {"displayName": "Kept", "artifactDigest": "sha256:aa"},
            "removed": {"displayName": "Removed", "artifactDigest": "sha256:bb"},
            "live": {"displayName": "Live", "source": "local-build:/work/live"},
        },
        "nodes": {
            "core.Note": {"pack": "core", "signature": "n"},
            "kept.Load": {"pack": "kept", "signature": "old"},
            "removed.Gone": {"pack": "removed", "signature": "g"},
            "live.Edit": {"pack": "live", "signature": "l"},
        },
        "typeOwners": {},
        "recordedAt": 1_700_000_000_000,
        "snapshot": snapshot,
    }


def make_installation(root: Path, current: int, record: dict[str, object] | None) -> Path:
    (root / "node-tables").mkdir(parents=True)
    (root / "current").write_text(f"{current}\n")
    if record is not None:
        (root / "node-tables" / "main.json").write_text(json.dumps(record))
    return root.resolve()


def test_provisional_nodes_serve_the_last_recorded_table_whatever_the_engine_state(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        recorded = make_installation(tmp_path / "recorded", 2, recorded_table("2"))
        empty = make_installation(tmp_path / "empty", 1, None)
        known = {"roots": frozenset({recorded})}

        async def start(target: OpenTarget, found: Callable[[str], None]) -> str:
            raise RuntimeError("the engine does not start")

        home = Home(installations=lambda: known["roots"], default_root=recorded, start=start)
        client = TestClient(TestServer(create_home_app(home, frontend=None)))
        await client.start_server()
        try:
            # Nothing is open: the installation restart would open answers.
            answer = await client.get("/api/nodes/provisional")
            assert answer.status == 200
            table = await answer.json()
            assert table == {**recorded_table("2"), "provisional": True}
            assert "epoch" not in table
            # /api/nodes stays the engine's own.
            assert (await client.get("/api/nodes")).status == 503

            # An open engine that failed to start still has its last record served.
            await client.post("/supervisor/engine", json={"root": str(recorded)})
            await wait_for(client, lambda s: s["state"] == "failed")
            assert (await (await client.get("/api/nodes/provisional")).json())["provisional"]

            # An installation that never recorded a table answers 404.
            known["roots"] = frozenset({recorded, empty})
            await client.post("/supervisor/engine", json={"root": str(empty)})
            missing = await client.get("/api/nodes/provisional")
            assert missing.status == 404
            assert (await missing.json())["error"] == "no-node-table"
        finally:
            await client.close()

    run_scenario(scenario())


def test_provisional_nodes_from_an_older_snapshot_take_the_current_catalogs(
    tmp_path: Path,
) -> None:
    root = make_installation(tmp_path / "installation", 3, recorded_table("2"))
    # The current snapshot's sealed catalogs, as its interpreter reads them.
    (root / "node-tables" / "catalogs-3.json").write_text(
        json.dumps(
            {
                "packs": ["added", "kept"],
                "nodes": {
                    "kept.Load": {"pack": "kept", "signature": "new"},
                    "added.Make": {"pack": "added", "signature": "m"},
                },
            }
        )
    )

    async def scenario() -> None:
        home = Home(installations=lambda: frozenset({root}), default_root=root)
        client = TestClient(TestServer(create_home_app(home, frontend=None)))
        await client.start_server()
        try:
            table = await (await client.get("/api/nodes/provisional")).json()
        finally:
            await client.close()
        assert table["provisional"] is True
        assert table["nodes"] == {
            "core.Note": {"pack": "core", "signature": "n"},
            "live.Edit": {"pack": "live", "signature": "l"},
            "kept.Load": {"pack": "kept", "signature": "new"},
            "added.Make": {"pack": "added", "signature": "m"},
        }
        assert set(table["packs"]) == {"core", "kept", "live", "added"}

    run_scenario(scenario())


def test_provisional_nodes_read_the_current_snapshots_catalogs_in_its_interpreter(
    tmp_path: Path,
) -> None:
    from test_environment_runtime import make_generation

    root = make_installation(tmp_path / "installation", 2, recorded_table("1"))
    make_generation(root, 2, fake_serve=False)
    (root / "builds/2/dinkster.lock").write_text('{"format":"dinkster.lock/1","packs":[]}\n')

    async def scenario() -> dict[str, object]:
        home = Home(installations=lambda: frozenset({root}), default_root=root)
        client = TestClient(TestServer(create_home_app(home, frontend=None)))
        await client.start_server()
        try:
            return await (await client.get("/api/nodes/provisional")).json()
        finally:
            await client.close()

    table = run_scenario(scenario())
    # The new snapshot installs no artifact packs: theirs leave, the rest stays.
    assert table["nodes"] == {
        "core.Note": {"pack": "core", "signature": "n"},
        "live.Edit": {"pack": "live", "signature": "l"},
    }
    kept = json.loads((root / "node-tables/catalogs-2.json").read_text())
    assert kept == {"packs": [], "nodes": {}}


def test_provisional_nodes_for_a_project_come_from_its_installation(tmp_path: Path) -> None:
    from dinkster_library import LibraryClient

    library = tmp_path / "library"
    library.mkdir()
    roots = {}
    for name, environment in (("default", "a" * 32), ("film", "b" * 32)):
        root = make_installation(tmp_path / name, 1, {**recorded_table("1"), "nodes": {name: {}}})
        (root / "environment-id").write_text(environment + "\n")
        (root / "environment.json").write_text(json.dumps({"libraryRoot": str(library)}))
        roots[name] = root
    with LibraryClient(library) as helper:
        helper.call("projects.register", id="film")
        helper.call(
            "projects.patch", id="film", revision=1, patch={"environment": {"id": "b" * 32}}
        )

    async def scenario() -> list[object]:
        home = Home(installations=lambda: frozenset(roots.values()), default_root=roots["default"])
        client = TestClient(TestServer(create_home_app(home, frontend=None)))
        await client.start_server()
        try:
            return [
                (await (await client.get(path)).json())["nodes"]
                for path in (
                    "/api/nodes/provisional?project=film",
                    "/api/nodes/provisional?project=unknown",
                    "/api/nodes/provisional",
                )
            ]
        finally:
            await client.close()

    # A project not in the library falls back to the default installation.
    assert run_scenario(scenario()) == [{"film": {}}, {"default": {}}, {"default": {}}]
