"""Managed HTTP previews never grant installation authority."""

from __future__ import annotations

import asyncio
import os
import subprocess
import threading
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer, make_mocked_request
from dinkster_registry import InstallError, Lockfile
from dinkster_server import PathRedactor, create_app
from dinkster_server.auth import PRINCIPAL_KEY, Principal, StaticBearerAuthenticator

import dinkster.pack_install_api as routes
from dinkster.compose import ServingComposer
from dinkster.installer import Installer, lock_local_pack
from tests.async_scenarios import run_scenario


@pytest.mark.parametrize("authenticated", [False, True])
def test_managed_preview_and_terminal_commands_never_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, authenticated: bool
) -> None:
    project = tmp_path / "project with ' quotes"
    project.mkdir()
    (project / "current").write_text("7")
    monkeypatch.setattr(routes.tempfile, "gettempdir", lambda: str(tmp_path))

    def plan(root: Path, selections: tuple[str, ...], output: Path) -> dict[str, object]:
        assert root == project
        assert selections == ("demo@1.0.0",)
        output.mkdir()
        return {"packPlan": {"current": 7}, "plan": ["add demo@1.0.0"], "expiresInSeconds": 900}

    monkeypatch.setattr(routes, "plan_pack_install", plan)

    async def scenario() -> list[str]:
        composer = ServingComposer()
        composition = composer.composition
        principals = {
            "operator": Principal("operator", grants={"local": frozenset({"settings:write"})}),
            "reader": Principal("reader", grants={"local": frozenset({"nodes:read"})}),
            "agent": Principal(
                "agent", kind="agent", grants={"local": frozenset({"settings:write"})}
            ),
            "delegate": Principal(
                "delegate",
                delegation_id="delegated",
                grants={"local": frozenset({"settings:write"})},
            ),
        }
        app = create_app(
            composition.make_engine,
            composition.schemas,
            authenticator=StaticBearerAuthenticator(principals) if authenticated else None,
        )
        routes.add_pack_install_routes(app, project, redactor=PathRedactor())
        client = TestClient(
            TestServer(app), headers={"Authorization": "Bearer operator"} if authenticated else {}
        )
        await client.start_server()
        try:
            selection = {"packs": [{"pack": "demo", "version": "1.0.0"}]}
            if authenticated:
                for token in ("reader", "agent", "delegate"):
                    for endpoint, body in (("plan", selection), ("apply", {"confirmation": "x"})):
                        response = await client.post(
                            f"/api/packs/install/{endpoint}",
                            json=body,
                            headers={"Authorization": f"Bearer {token}"},
                        )
                        assert response.status == 403
            for invalid in (
                {"packs": []},
                {"packs": [{"pack": "./demo", "version": "1.0.0"}]},
                {"packs": [{"pack": "demo", "version": "latest"}]},
                {"packs": [{"pack": "demo", "version": "1.0.0", "source": "file:///x"}]},
                {**selection, "root": "/other"},
            ):
                assert (await client.post("/api/packs/install/plan", json=invalid)).status == 400
            assert (
                await client.post(
                    "/api/packs/install/plan",
                    json=selection,
                    headers={"X-Dinkster-Actor-Kind": "agent"},
                )
            ).status == 403
            response = await client.post("/api/packs/install/plan", json=selection)
            assert response.status == 200, await response.text()
            result = await response.json()
            assert result["plan"] == {"current": 7}
            assert result["review"] == "add demo@1.0.0"
            assert result["expiresInSeconds"] == 900
            assert "confirmation" not in result
            response = await client.post(
                "/api/packs/install/apply", json={"confirmation": "anything"}
            )
            assert response.status == 409
            refusal = await response.json()
            assert refusal["error"] == "registry.managed-generation-required"
            assert refusal["commands"] == result["commands"]
            assert (project / "current").read_text() == "7"
            return result["commands"]
        finally:
            await client.close()
            await composer.close()

    commands = run_scenario(scenario())
    if os.name != "nt":
        # Execute the copyable commands against a CLI double to check shell quoting
        # and that apply uses the newly generated approval, not the HTTP preview.
        cli = tmp_path / "dinkster-env"
        cli.write_text(
            "#!/usr/bin/env python3\nimport json,sys\nfrom pathlib import Path\n"
            "args=sys.argv[1:]\nassert args[:2]==['--root', " + repr(str(project)) + "]\n"
            "if args[2]=='packs-plan':\n"
            " assert args[3]=='demo@1.0.0'\n"
            " Path(args[args.index('--output')+1]).mkdir()\n"
            " print(json.dumps({'approval':'sha256:new-cli-plan'}))\n"
            "else:\n"
            " assert args[2]=='packs-apply'\n"
            " assert args[args.index('--approval')+1]=='sha256:new-cli-plan'\n"
        )
        cli.chmod(0o755)
        env = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]}
        for command in commands:
            subprocess.run(["sh", "-ec", command], env=env, check=True, capture_output=True)


@pytest.mark.parametrize("cancel", [1, 2, "shutdown"])
def test_cancelled_preview_keeps_scratch_alive_until_worker_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel: int | str
) -> None:
    started, finish = threading.Event(), threading.Event()
    scratch: list[Path] = []

    def plan(_root: Path, _selections: tuple[str, ...], output: Path) -> dict[str, object]:
        output.mkdir()
        scratch.append(output)
        started.set()
        assert finish.wait(20)
        assert output.is_dir()
        return {"packPlan": {}, "plan": ["add demo@1.0.0"], "expiresInSeconds": 900}

    monkeypatch.setattr(routes, "plan_pack_install", plan)

    async def scenario() -> None:
        composer = ServingComposer()
        composition = composer.composition
        app = create_app(composition.make_engine, composition.schemas)
        request_task: asyncio.Task[Any] | None = None

        @web.middleware
        async def capture(request: web.Request, handler: Any) -> web.StreamResponse:
            nonlocal request_task
            if request.path.endswith("/plan") and request_task is None:
                request_task = asyncio.current_task()
            return await handler(request)

        app.middlewares.append(capture)
        routes.add_pack_install_routes(app, tmp_path, redactor=PathRedactor())
        cleanup_started = asyncio.Event()

        async def begin_cleanup(_: web.Application) -> None:
            cleanup_started.set()

        app.on_cleanup.insert(0, begin_cleanup)
        server = TestServer(app)
        await server.start_server(shutdown_timeout=0.001)
        client = TestClient(server)
        await client.start_server()
        selection = {"packs": [{"pack": "demo", "version": "1.0.0"}]}
        requesting = asyncio.ensure_future(client.post("/api/packs/install/plan", json=selection))
        try:
            assert await asyncio.to_thread(started.wait, 10)
            assert request_task is not None
            if cancel == "shutdown":
                cleanup = asyncio.create_task(server.close())
                await asyncio.wait_for(cleanup_started.wait(), timeout=10)
                assert not cleanup.done()
            else:
                request_task.cancel()
                if cancel == 2:
                    asyncio.get_running_loop().call_soon(request_task.cancel)
                response = await client.post("/api/packs/install/plan", json=selection)
                assert response.status == 409
                assert (await response.json())["error"] == "install-in-progress"
                cleanup = None
            assert scratch[0].is_dir()
            finish.set()
            if cleanup is not None:
                await cleanup
            await asyncio.gather(requesting, return_exceptions=True)
            assert not scratch[0].exists()
        finally:
            finish.set()
            await asyncio.gather(requesting, return_exceptions=True)
            await client.close()
            await composer.close()

    run_scenario(scenario())


@pytest.mark.parametrize("peer", ["203.0.113.12", "::ffff:203.0.113.12"])
def test_pack_install_refuses_nonlocal_transport_even_with_operator_and_forwarded_headers(
    tmp_path: Path, peer: str
) -> None:
    async def scenario() -> None:
        composer = ServingComposer()
        composition = composer.composition
        app = create_app(composition.make_engine, composition.schemas)
        routes.add_pack_install_routes(app, tmp_path, redactor=PathRedactor())
        transport = Mock()
        transport.get_extra_info.return_value = (peer, 12345)
        try:
            for action in ("plan", "apply"):
                request = make_mocked_request(
                    "POST",
                    f"/api/packs/install/{action}",
                    app=app,
                    transport=transport,
                    headers={"X-Forwarded-For": "127.0.0.1"},
                )
                request[PRINCIPAL_KEY] = Principal(
                    "operator", local=True, grants={"local": frozenset({"settings:write"})}
                )
                resolved = await app.router.resolve(request)
                with pytest.raises(web.HTTPForbidden):
                    await resolved.handler(request)
        finally:
            await composer.close()

    run_scenario(scenario())


def test_reviewed_base_rechecked_after_writer_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import contextmanager

    installer = Installer(tmp_path / "root", accelerator="cpu")
    source = tmp_path / "source"
    (source / "demo").mkdir(parents=True)
    (source / "demo" / "__init__.py").write_text("NODES = []\n")
    (source / "dinkster-pack.toml").write_text(
        '[pack]\nname = "demo"\n[pack.entry]\nnodes = "demo:NODES"\n'
    )
    entry, _ = lock_local_pack(source, installer.artifacts_dir)
    original = installer._locked
    competitor = Installer(installer.root, accelerator="cpu")

    @contextmanager
    def changed_before_lock():
        competitor.apply(Lockfile.of([entry]), venvs=False)
        with original():
            yield

    monkeypatch.setattr(installer, "_locked", changed_before_lock)
    with pytest.raises(InstallError, match="plan is stale"):
        installer.apply(Lockfile(), venvs=False, expected_base=None)
    assert installer.current_lockfile() == Lockfile.of([entry])
