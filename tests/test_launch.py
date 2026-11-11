from __future__ import annotations

import builtins
import json
import urllib.error
from pathlib import Path
from unittest.mock import Mock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import MountDef, MountTable, dump_mounts, load_mounts, load_output_mount

from dinkster import frontend, launch, serve, setup
from dinkster.cli import main as cli_main
from dinkster.frontend import install_frontend
from dinkster.mounts_api import MountService
from dinkster.project_folders import ProjectFolders
from tests.async_scenarios import run_scenario


def test_frontend_discovery_finds_a_sibling_development_build(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(frontend, "__file__", str(tmp_path / "Dinkster/src/dinkster/frontend.py"))
    monkeypatch.setattr(
        frontend.importlib.resources, "files", lambda _name: tmp_path / "Dinkster/src/dinkster"
    )
    bundle = tmp_path / "Dinkster-Frontend/packages/app/dist"
    bundle.mkdir(parents=True)
    (bundle / "index.html").write_text("<main>development</main>")
    assert frontend.discover_frontend_bundle() == bundle


def test_setup_creates_the_default_launch_roots(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "state"))

    assert setup.main([]) == 0

    library, packs = setup.default_roots()
    assert library.is_dir()
    assert (library / "output").is_dir()
    assert (library / "imports").is_dir()
    assert load_mounts(library / "mounts.toml") == (
        MountDef(id="imports", path=library / "imports", mode="readwrite"),
        MountDef(id="output", path=library / "output", mode="readwrite"),
    )
    assert load_output_mount(library / "mounts.toml") == "output"
    assert (packs / "generations").is_dir()
    assert "Run `dinkster`" in capsys.readouterr().out
    table = MountTable(output_mount="output")
    for mount in load_mounts(library / "mounts.toml"):
        table.add(mount)
    table.add(MountDef("derived", tmp_path / "derived"), source="derived")
    service = MountService(table, library / "mounts.toml")
    assert set(service.project_context("default").mounts) == {"derived", "imports", "output"}


@pytest.mark.parametrize(
    "mode", ["json", "text", "missing-locks", "no-defaults", "offline", "corrupt-cache"]
)
def test_setup_plan_uses_target_defaults_and_bound_locks_without_acquiring_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, mode: str
) -> None:
    import hashlib

    from dinkster_registry import LockedPack, artifact_digest
    from dinkster_store import object_path
    from dinkster_supervisor.environment_artifacts import acquire, discover
    from dinkster_workers.pack_lock import PackLock
    from test_environment_artifacts import _json, _mirror
    from test_installer import _serve_artifacts, _static_release

    from dinkster.registries import (
        NamedRegistry,
        RegistryConfig,
        StaticPackIndex,
        plan_registry_sources,
    )

    mirror, store, root = tmp_path / "mirror", tmp_path / "store", tmp_path / "project"
    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA", "DINKSTER_HOME"):
        monkeypatch.setenv(variable, str(tmp_path / variable))
    monkeypatch.delenv("DINKSTER_REGISTRY", raising=False)
    monkeypatch.delenv("DINKSTER_REGISTRIES", raising=False)
    base, engine, channel = _mirror(mirror)
    defaults = {"example-pack": "1.2.0", "other-pack": "1.2.3"}
    if mode != "no-defaults":
        engine["defaultPacks"] = defaults
    previous = mirror / "engine" / channel["engine"]["release"][7:]
    payload = _json(engine)
    release = hashlib.sha256(payload).hexdigest()
    previous.rename(previous.with_name(release))
    (previous.with_name(release) / "release-manifest.json").write_bytes(payload)
    channel["engine"]["release"] = "sha256:" + release
    channel["engine"]["locations"] = [
        {"type": "mirror", "path": f"engine/{release}/release-manifest.json"}
    ]
    (mirror / "channels/main.json").write_bytes(_json(channel))
    files: dict[str, bytes] = {}
    releases = []
    for name in defaults:
        entry = LockedPack(name, "1.2.3", artifact_digest(name.encode()), "acme", ())
        row = _static_release(entry, name.encode())
        lock = PackLock(name, "fixture-cpu", ">=3.12", (), (), ())
        lock_bytes = lock.to_json().encode()
        identity = artifact_digest(lock_bytes)
        path = "locks/" + identity.replace(":", "/") + ".json"
        row["installLocks"] = (
            {}
            if mode == "missing-locks"
            else {
                "fixture-cpu": {
                    "baseId": base["id"],
                    "artifactDigest": entry.artifact_digest,
                    "digest": identity,
                    "size": len(lock_bytes),
                    "url": path,
                }
            }
        )
        files["/static/" + path] = lock_bytes
        releases.append(row)
    files["/static/packs/index.json"] = _json({"schemaVersion": 1, "items": releases})
    requests: list[tuple[str, str | None]] = []
    server = _serve_artifacts(files, requests)
    endpoint = f"http://127.0.0.1:{server.server_port}"
    registry_file = tmp_path / "registries.toml"
    registry_file.write_text(f'[registries.private]\nendpoint = "{endpoint}"\ndefault = true\n')
    try:
        if mode == "offline":
            acquire(discover((str(mirror),), "main"), "fixture-cpu", store)
            registry = NamedRegistry("private", endpoint, default=True)
            plan_registry_sources(
                RegistryConfig((registry,)),
                registry,
                defaults,
                engine_version=engine["version"],
                variant="fixture-cpu",
                base_id=base["id"],
                index=StaticPackIndex(store / "metadata" / "registry"),
            )
            files.clear()
            mirror.rename(tmp_path / "disconnected")
            requests.clear()
        cached = None
        if mode in {"offline", "corrupt-cache"}:
            cached = (
                object_path(
                    store / "wheels", "sha256:" + hashlib.sha256(b"example-pack").hexdigest()
                )
                / "example_pack-1.2.3-py3-none-any.whl"
            )
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(b"example-pack" if mode == "offline" else b"X" * 12)
        before = {p: p.read_bytes() if p.is_file() else None for p in tmp_path.rglob("*")}
        arguments = [
            "setup",
            "--plan",
            "--root",
            str(root),
            "--data",
            str(store),
            "--mirror",
            str(mirror),
            "--channel",
            "main",
            "--variant",
            "fixture-cpu",
            "--registries",
            str(registry_file),
        ]
        if mode != "text":
            arguments.append("--json")
        if mode == "offline":
            arguments.append("--offline")
        code = cli_main(arguments)
        captured = capsys.readouterr()
        if mode == "no-defaults":
            assert code == 1
            assert "target engine has no default pack declaration" in captured.err
            assert requests == []
        elif mode == "missing-locks":
            assert code == 1
            assert "example-pack@1.2.3" in captured.err and "other-pack@1.2.3" in captured.err
        else:
            assert code == 0, captured.err
            if mode == "text":
                assert f"Engine: {engine['version']} (sha256:{release})" in captured.out
                assert "Runtime: Python 3.14.10; fixture-cpu" in captured.out
                assert "Pack: example-pack@1.2.3; registry:private" in captured.out
                assert "Dependency lock: other-pack; sha256:" in captured.out
                assert "Pack payload upper bound: 22 bytes" in captured.out
            else:
                result = json.loads(captured.out)
                assert result["engineRelease"] == "sha256:" + release
                assert result["baseId"] == base["id"]
                assert set(result["dependencyLocks"]) == set(defaults)
                assert result["maximumPackPayloadBytes"] == 22
                assert all(
                    row["source"] == "registry:private" for row in result["sources"]["packs"]
                )
                if mode == "offline":
                    assert all(row["cached"] for row in result["runtimeInputs"])
                    assert set(result["cachedPackPayloads"].values()) == {None, str(cached)}
                else:
                    assert all(value is None for value in result["cachedPackPayloads"].values())
        assert all(path.endswith(".json") for path, _ in requests)
        if mode == "offline":
            assert requests == []
        assert {p: p.read_bytes() if p.is_file() else None for p in tmp_path.rglob("*")} == before
        assert not root.exists()
    finally:
        server.shutdown()
        server.server_close()


def test_setup_preserves_a_selected_alternate_output_mount(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "state"))
    library, _ = setup.default_roots()
    library.mkdir(parents=True)
    alternate = library / "renders"
    alternate.mkdir()
    mounts = (MountDef("renders", alternate, "readwrite"),)
    (library / "mounts.toml").write_text(dump_mounts(mounts, output_mount="renders"), "utf-8")

    setup.main([])

    assert load_output_mount(library / "mounts.toml") == "renders"
    assert load_mounts(library / "mounts.toml") == (
        MountDef("imports", library / "imports", "readwrite"),
        *mounts,
    )
    assert not (library / "output").exists()


def test_setup_reuses_an_existing_mount_for_the_default_output_path(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "state"))
    library, _ = setup.default_roots()
    library.mkdir(parents=True)
    existing = MountDef("existing-output", library / "output", "read")
    (library / "mounts.toml").write_text(dump_mounts((existing,)), "utf-8")

    setup.main([])

    assert load_mounts(library / "mounts.toml") == (
        existing,
        MountDef("imports", library / "imports", "readwrite"),
    )
    assert load_output_mount(library / "mounts.toml") is None
    assert not (library / "output").exists()


@pytest.mark.parametrize("custom_id", [False, True])
def test_setup_preserves_existing_offline_output_grants(
    tmp_path: Path, monkeypatch, custom_id: bool
) -> None:
    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "state"))
    library, _ = setup.default_roots()
    library.mkdir(parents=True)
    root = library / "output" if custom_id else tmp_path / "offline-output"
    existing = MountDef("renders" if custom_id else "output", root, "read", priority=7)
    config = library / "mounts.toml"
    config.write_text(dump_mounts((existing,)), "utf-8")

    setup.main([])
    saved = config.read_bytes()
    setup.main([])

    assert config.read_bytes() == saved
    assert existing in load_mounts(config)
    assert load_output_mount(config) is None
    assert not root.exists()
    assert not (library / "output").exists()


def test_setup_does_not_restore_a_removed_output_grant(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "state"))
    setup.main([])
    library, _ = setup.default_roots()
    config = library / "mounts.toml"
    remaining = tuple(mount for mount in load_mounts(config) if mount.id != "output")
    config.write_text(dump_mounts(remaining), "utf-8")
    (library / "output").rmdir()

    setup.main([])

    assert load_mounts(config) == remaining
    assert load_output_mount(config) is None
    assert not (library / "output").exists()


@pytest.mark.parametrize("custom_id", [False, True])
def test_setup_preserves_existing_import_grants(
    tmp_path: Path, monkeypatch, custom_id: bool
) -> None:
    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "state"))
    library, _ = setup.default_roots()
    library.mkdir(parents=True)
    root = library / "imports" if custom_id else tmp_path / "offline-network-imports"
    existing = MountDef("plates" if custom_id else "imports", root, "read", priority=7)
    config = library / "mounts.toml"
    config.write_text(dump_mounts((existing,)), "utf-8")
    project_path = library / "project-folders.json"
    projects = ProjectFolders(project_path)
    projects.initialize((existing.id,))
    projects.register(existing.id, project="film", shared=False)
    projects.update("other", [], available=(existing.id,), output=None)
    preferences = json.loads(project_path.read_text("utf-8"))

    setup.main([])
    saved = config.read_bytes()
    setup.main([])

    assert config.read_bytes() == saved
    assert existing in load_mounts(config)
    assert len(load_mounts(config)) == 2
    assert not root.exists()
    assert not (library / "imports").exists()
    assert load_output_mount(config) == "output"
    assert json.loads(project_path.read_text("utf-8")) == {**preferences, "shared": ["output"]}


def test_setup_does_not_restore_a_removed_import_grant(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "state"))
    setup.main([])
    library, _ = setup.default_roots()
    config = library / "mounts.toml"
    remaining = tuple(mount for mount in load_mounts(config) if mount.id != "imports")
    config.write_text(dump_mounts(remaining, output_mount="output"), "utf-8")
    (library / "imports").rmdir()

    setup.main([])

    assert load_mounts(config) == remaining
    assert not (library / "imports").exists()


@pytest.mark.parametrize("interrupt_settings_write", (False, True))
def test_setup_adds_imports_to_existing_projects_without_resetting_preferences(
    tmp_path: Path, monkeypatch, interrupt_settings_write: bool
) -> None:
    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "state"))
    library, _ = setup.default_roots()
    library.mkdir(parents=True)
    config = library / "mounts.toml"
    mounts = (
        MountDef("output", library / "output", "readwrite"),
        MountDef("models", tmp_path / "models"),
    )
    config.write_text(dump_mounts(mounts, output_mount="output"), "utf-8")
    project_path = library / "project-folders.json"
    projects = ProjectFolders(project_path)
    projects.initialize(("output", "models"))
    projects.update("film", ["models"], available=("output", "models"), output=None)
    projects.update("music", ["output", "models"], available=("output", "models"), output="output")
    before = json.loads(project_path.read_text("utf-8"))

    if interrupt_settings_write:
        replace = setup.os.replace

        def interrupt(source: str | Path, destination: str | Path) -> None:
            if Path(destination) == project_path:
                raise OSError("project settings could not be published")
            replace(source, destination)

        with monkeypatch.context() as patch:
            patch.setattr(setup.os, "replace", interrupt)
            with pytest.raises(OSError, match="project settings could not be published"):
                setup.main([])
        assert any(mount.id == "imports" for mount in load_mounts(config))
        assert json.loads(project_path.read_text("utf-8")) == before
        assert not (library / ".imports-initialized").exists()

    setup.main([])

    after = json.loads(project_path.read_text("utf-8"))
    assert after == {**before, "shared": ["output", "models", "imports"]}
    table = MountTable(output_mount="output")
    for mount in load_mounts(config):
        table.add(mount)
    service = MountService(table, config)
    assert service.project_context("film").mounts == ("models", "imports")
    assert service.project_context("film").output_mount is None
    assert service.project_context("music").mounts == ("output", "models", "imports")
    assert service.project_context("music").output_mount == "output"
    service.projects.update("film", ["models"], available=service.mount_ids(), output=None)
    saved = project_path.read_bytes()

    setup.main([])

    assert project_path.read_bytes() == saved
    restarted = MountService(table, config)
    assert restarted.project_context("film").mounts == ("models",)
    assert restarted.project_context("music").mounts == ("output", "models", "imports")


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
    monkeypatch.setattr(serve, "main", lambda argv: serve_calls.append(argv))
    monkeypatch.setattr(launch, "_open_browser_when_ready", lambda url: browser_calls.append(url))

    class ImmediateThread:
        def __init__(self, *, target, args: tuple[str], daemon: bool) -> None:
            assert daemon is True
            self.callback = target
            self.args = args

        def start(self) -> None:
            self.callback(*self.args)

    monkeypatch.setattr(launch.threading, "Thread", ImmediateThread)
    output = Mock(wraps=builtins.print)
    monkeypatch.setattr(builtins, "print", output)

    assert cli_main(["--port", "4640"]) == 0

    expected_url = "http://127.0.0.1:4640"
    assert browser_calls == [expected_url]
    assert f"Dinkster is available at {expected_url}" in capsys.readouterr().out
    output.assert_any_call(f"Dinkster is available at {expected_url}", flush=True)
    output.assert_any_call("Checking pack catalogs...", flush=True)
    args = serve_calls[0]
    assert args[args.index("--port") + 1] == "4640"
    assert args[args.index("--frontend-root") + 1] == str(bundle)
    assert "--prepare-stale-catalogs" in args
    assert "--allow-mount-changes" in args
    assert "--execution-python" not in args


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


def test_explicit_frontend_root_overrides_discovery(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "state"))
    setup.main([])
    bundle = tmp_path / "managed frontend"
    bundle.mkdir()
    (bundle / "index.html").write_text("editor")
    monkeypatch.setattr(launch, "discover_frontend_bundle", lambda: None)
    calls = []
    monkeypatch.setattr(serve, "main", lambda argv: calls.append(argv))
    assert launch.main(["--no-browser", "--frontend-root", str(bundle)]) == 0
    assert calls[0][calls[0].index("--frontend-root") + 1] == str(bundle)


def test_no_browser_and_vite_proxy_are_the_only_alternate_launch_controls(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "state"))
    setup.main([])
    calls: list[list[str]] = []
    monkeypatch.setattr(serve, "main", lambda argv: calls.append(argv))
    monkeypatch.setattr(
        launch.webbrowser,
        "open",
        lambda _url: (_ for _ in ()).throw(AssertionError("browser must stay closed")),
    )

    assert launch.main(["--no-browser", "--frontend-dev", "http://127.0.0.1:5199"]) == 0
    args = calls[0]
    assert args[args.index("--frontend-dev") + 1] == "http://127.0.0.1:5199"
    assert "--frontend-root" not in args


def test_static_frontend_preserves_api_routes_and_spa_fallback(tmp_path: Path, monkeypatch) -> None:
    from dinkster_server.app import install_browser_request_security, install_cors

    system_guess_type = frontend.mimetypes.guess_type

    def windows_guess_type(name: str) -> tuple[str | None, str | None]:
        if name.endswith(".js"):
            return "application/javascript", None
        return system_guess_type(name)

    monkeypatch.setattr(frontend.mimetypes, "guess_type", windows_guess_type)

    async def scenario() -> None:
        bundle = tmp_path / "dist"
        (bundle / "assets").mkdir(parents=True)
        (bundle / "index.html").write_text("<main>editor</main>")
        (bundle / "assets" / "app.js").write_text("console.log('app')")
        app = web.Application()
        origin = "https://editor.example"
        install_browser_request_security(app, allow_hosts=(), allow_origins=(origin,))
        install_cors(app, (origin,))

        async def health(_request: web.Request) -> web.Response:
            return web.json_response({"ok": True})

        app.router.add_get("/api/health", health)
        install_frontend(app, bundle=bundle)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            assert await (await client.get("/api/health")).json() == {"ok": True}
            assert (await client.head("/api/health")).status == 200
            for path in ("/workflow/one", "/project/one", "/apiary"):
                page = await client.get(path)
                assert page.status == 200
                assert await page.text() == "<main>editor</main>"
            for path in ("/api", "/api/", "/api/compat/comfy/schemas?version=1"):
                for method in ("GET", "HEAD", "POST"):
                    response = await client.request(method, path)
                    assert response.status == 404
                    assert response.content_type != "text/html"
                    if method == "HEAD":
                        assert await response.read() == b""
            asset = await client.get("/assets/app.js", headers={"Origin": origin})
            assert asset.content_type == "text/javascript"
            assert await asset.text() == "console.log('app')"
            assert asset.headers["Access-Control-Allow-Origin"] == origin
            assert (await client.head("/assets/app.js")).content_type == "text/javascript"
            refused = await client.get("/assets/app.js", headers={"Host": "untrusted.example"})
            assert refused.status == 421
            assert (await client.post("/unknown")).status == 404
        finally:
            await client.close()

    run_scenario(scenario())


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

        async def health(_request: web.Request) -> web.Response:
            return web.json_response({"ok": True})

        app.router.add_get("/api/health", health)
        install_frontend(app, development_url=str(vite_server.make_url("/")))
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            assert await (await client.get("/api/health")).json() == {"ok": True}
            assert (await client.head("/api/health")).status == 200
            for path in ("/api", "/api/", "/api/compat/comfy/schemas"):
                for method in ("GET", "HEAD", "POST"):
                    response = await client.request(method, path)
                    assert response.status == 404
                    if method == "HEAD":
                        assert await response.read() == b""
            for path in ("/project/one", "/apiary"):
                response = await client.get(path)
                assert response.status == 200
                assert await response.text() == f"GET {path} "
            response = await client.post("/asset?version=2", data="body")
            assert await response.text() == "POST /asset?version=2 body"
            socket = await client.ws_connect("/hmr")
            await socket.send_str("change")
            assert (await socket.receive()).data == "vite:change"
            await socket.close()
        finally:
            await client.close()
            await vite_server.close()

    run_scenario(scenario())
