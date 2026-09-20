from __future__ import annotations

import asyncio
import urllib.error
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from dinkster import launch, setup
from dinkster.cli import main as cli_main
from dinkster.frontend import install_frontend


def test_setup_creates_the_default_launch_roots(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "state"))

    assert setup.main([]) == 0

    library, packs = setup.default_roots()
    assert library.is_dir()
    assert (packs / "generations").is_dir()
    assert "Run `dinkster`" in capsys.readouterr().out


def test_bare_cli_launches_one_origin_and_opens_browser(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "state"))
    setup.main([])
    bundle = tmp_path / "frontend"
    bundle.mkdir()
    (bundle / "index.html").write_text("editor")
    serve_calls: list[list[str]] = []
    browser_calls: list[str] = []

    monkeypatch.setattr(launch, "discover_frontend_bundle", lambda: bundle)
    monkeypatch.setattr(launch.serve, "main", lambda argv: serve_calls.append(argv))
    monkeypatch.setattr(launch, "_open_browser_when_ready", lambda url: browser_calls.append(url))

    class ImmediateThread:
        def __init__(self, *, target, args: tuple[str], daemon: bool) -> None:
            assert daemon is True
            self.callback = target
            self.args = args

        def start(self) -> None:
            self.callback(*self.args)

    monkeypatch.setattr(launch.threading, "Thread", ImmediateThread)

    assert cli_main(["--port", "4640"]) == 0

    expected_url = "http://127.0.0.1:4640"
    assert browser_calls == [expected_url]
    assert f"Dinkster is available at {expected_url}" in capsys.readouterr().out
    args = serve_calls[0]
    assert args[args.index("--port") + 1] == "4640"
    assert args[args.index("--frontend-root") + 1] == str(bundle)
    assert "--prepare-stale-catalogs" in args
    assert "--disable-p2p" in args
    assert "--comfy-python" not in args


def test_browser_waits_for_health_before_opening(monkeypatch) -> None:
    attempts = 0
    events: list[str] = []

    class HealthyResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

    def urlopen(_url: str, *, timeout: float):
        nonlocal attempts
        assert timeout == 0.5
        attempts += 1
        if attempts == 1:
            raise urllib.error.URLError("not ready")
        return HealthyResponse()

    monkeypatch.setattr(launch.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(launch.time, "sleep", lambda _seconds: events.append("sleep"))
    monkeypatch.setattr(launch.webbrowser, "open", lambda url: events.append(url))

    launch._open_browser_when_ready("http://127.0.0.1:4640")

    assert attempts == 2
    assert events == ["sleep", "http://127.0.0.1:4640"]


def test_no_browser_and_vite_proxy_are_the_only_alternate_launch_controls(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "state"))
    setup.main([])
    calls: list[list[str]] = []
    monkeypatch.setattr(launch.serve, "main", lambda argv: calls.append(argv))
    monkeypatch.setattr(
        launch.webbrowser,
        "open",
        lambda _url: (_ for _ in ()).throw(AssertionError("browser must stay closed")),
    )

    assert launch.main(["--no-browser", "--frontend-dev", "http://127.0.0.1:5199"]) == 0
    args = calls[0]
    assert args[args.index("--frontend-dev") + 1] == "http://127.0.0.1:5199"
    assert "--frontend-root" not in args


def test_static_frontend_preserves_api_routes_and_spa_fallback(tmp_path: Path) -> None:
    async def scenario() -> None:
        bundle = tmp_path / "dist"
        (bundle / "assets").mkdir(parents=True)
        (bundle / "index.html").write_text("<main>editor</main>")
        (bundle / "assets" / "app.js").write_text("console.log('app')")
        app = web.Application()

        async def health(_request: web.Request) -> web.Response:
            return web.json_response({"ok": True})

        app.router.add_get("/api/health", health)
        install_frontend(app, bundle=bundle)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            assert await (await client.get("/api/health")).json() == {"ok": True}
            assert await (await client.get("/workflow/one")).text() == "<main>editor</main>"
            asset = await client.get("/assets/app.js")
            assert asset.content_type == "application/javascript"
            assert await asset.text() == "console.log('app')"
            assert (await client.post("/unknown")).status == 404
        finally:
            await client.close()

    asyncio.run(scenario())


def test_frontend_dev_proxies_http_and_websockets() -> None:
    async def scenario() -> None:
        vite = web.Application()

        async def http(request: web.Request) -> web.Response:
            return web.Response(text=f"{request.method} {request.path_qs} {await request.text()}")

        async def websocket(request: web.Request) -> web.WebSocketResponse:
            socket = web.WebSocketResponse()
            await socket.prepare(request)
            async for message in socket:
                await socket.send_str(f"vite:{message.data}")
            return socket

        vite.router.add_get("/hmr", websocket)
        vite.router.add_route("*", "/{path:.*}", http)
        vite_server = TestServer(vite)
        await vite_server.start_server()
        app = web.Application()
        install_frontend(app, development_url=str(vite_server.make_url("/")))
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.post("/asset?version=2", data="body")
            assert await response.text() == "POST /asset?version=2 body"
            socket = await client.ws_connect("/hmr")
            await socket.send_str("change")
            assert (await socket.receive()).data == "vite:change"
            await socket.close()
        finally:
            await client.close()
            await vite_server.close()

    asyncio.run(scenario())
