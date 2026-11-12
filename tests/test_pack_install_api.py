"""Managed HTTP previews never grant installation authority."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
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
import dinkster.pack_preparation as preparation_state
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

    monkeypatch.setattr("dinkster_supervisor.environment_pack_install.plan_pack_install", plan)

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
    # The native shell must surface planning failure, even with parseable output.
    script = tmp_path / "cli.py"
    script.write_text(
        "import json,os,sys\nfrom pathlib import Path\n"
        "args=sys.argv[1:]\nassert args[:2]==['--root', " + repr(str(project)) + "]\n"
        "if args[2]=='packs-plan':\n"
        " assert args[3]=='demo@1.0.0'\n"
        " print(json.dumps({'approval':'sha256:new-cli-plan'}))\n"
        " if os.environ.get('PLAN_EXIT'): sys.exit(int(os.environ['PLAN_EXIT']))\n"
        " Path(args[args.index('--output')+1]).mkdir()\n"
        "else:\n"
        " assert args[2]=='packs-apply'\n"
        " assert args[args.index('--approval')+1]=='sha256:new-cli-plan'\n"
    )
    if os.name == "nt":
        (tmp_path / "dinkster-env.cmd").write_text(
            f'@echo off\n"{sys.executable}" "%~dp0cli.py" %*\n'
        )
        shell = shutil.which("pwsh") or shutil.which("powershell")
        assert shell is not None, "Windows command coverage requires PowerShell"
        arguments = [shell, "-NoProfile", "-NonInteractive", "-Command"]
    else:
        cli = tmp_path / "dinkster-env"
        cli.write_text("#!/usr/bin/env python3\n" + script.read_text())
        cli.chmod(0o755)
        arguments = ["sh", "-c"]
    env = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]}
    for command in commands:
        subprocess.run([*arguments, command], env=env, check=True, capture_output=True)
    failed = subprocess.run(
        [*arguments, commands[0]], env={**env, "PLAN_EXIT": "17"}, capture_output=True
    )
    assert failed.returncode != 0
    assert failed.stdout == b""


@pytest.mark.parametrize("outcome", ["ready", "cancelled", "stalled", "expired", "failed"])
def test_preparation_status_outlives_metadata_deadline_without_resurrecting_cancelled_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    from dinkster_supervisor.environment_progress import preparation_progress

    started, finish = threading.Event(), threading.Event()
    scratch: list[Path] = []
    clock = [0.0]
    monkeypatch.setattr(preparation_state, "monotonic", lambda: clock[0])

    def plan(_root: Path, _selections: tuple[str, ...], output: Path) -> dict[str, object]:
        output.mkdir()
        scratch.append(output)
        preparation_progress("checking-software", 128, 256)
        started.set()
        assert finish.wait(10)
        if outcome == "failed":
            raise OSError("private host path must not leak")
        return {"packPlan": {"current": 7}, "plan": ["add demo@1.0.0"], "expiresInSeconds": 900}

    monkeypatch.setattr("dinkster_supervisor.environment_pack_install.plan_pack_install", plan)

    async def scenario() -> None:
        composer = ServingComposer()
        app = create_app(
            composer.composition.make_engine,
            composer.composition.schemas,
            authenticator=StaticBearerAuthenticator(
                {
                    name: Principal(name, grants={"local": frozenset({"settings:write"})})
                    for name in ("operator", "other")
                }
            ),
        )
        routes.add_pack_install_routes(app, tmp_path, redactor=PathRedactor())
        client = TestClient(TestServer(app), headers={"Authorization": "Bearer operator"})
        await client.start_server()
        try:
            selection = {"packs": [{"pack": "demo", "version": "1.0.0"}]}
            response = await client.post("/api/packs/install/prepare", json=selection)
            assert response.status == 202
            identity = {"preparation": (await response.json())["preparation"]}
            assert await asyncio.to_thread(started.wait, 10)
            assert (await client.post("/api/packs/install/prepare", json=selection)).status == 409
            denied = await client.post(
                "/api/packs/install/preparation/status",
                json=identity,
                headers={"Authorization": "Bearer other"},
            )
            assert denied.status == 404
            clock[0] = 45
            status = await client.post("/api/packs/install/preparation/status", json=identity)
            snapshot = await status.json()
            assert snapshot["state"] == "preparing"
            assert snapshot["progress"] == {
                "sequence": 1,
                "stage": "checking-software",
                "completed": 128,
                "total": 256,
            }
            assert "result" not in snapshot
            if outcome == "cancelled":
                status = await client.post("/api/packs/install/preparation/cancel", json=identity)
                assert (await status.json())["state"] == "cancelled"
                assert (
                    await client.post("/api/packs/install/prepare", json=selection)
                ).status == 409
            elif outcome in {"stalled", "expired"}:
                clock[0] = 120 if outcome == "stalled" else 1800
                status = await client.post("/api/packs/install/preparation/status", json=identity)
                assert (await status.json())["error"] == f"preparation-{outcome}"
            finish.set()
            async with asyncio.timeout(10):
                while True:
                    status = await client.post(
                        "/api/packs/install/preparation/status", json=identity
                    )
                    snapshot = await status.json()
                    if snapshot["state"] != "preparing" and not scratch[0].exists():
                        break
                    await asyncio.sleep(0)
            if outcome == "ready":
                assert snapshot["result"]["plan"] == {"current": 7}
                assert "confirmation" not in snapshot["result"]
                assert len(snapshot["result"]["commands"]) == 2
            else:
                assert snapshot["state"] == ("cancelled" if outcome == "cancelled" else "failed")
                assert "result" not in snapshot
                assert "private host path" not in str(snapshot)
        finally:
            finish.set()
            await client.close()
            await composer.close()

    run_scenario(scenario())


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

    monkeypatch.setattr("dinkster_supervisor.environment_pack_install.plan_pack_install", plan)

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
