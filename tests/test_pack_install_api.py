"""HTTP confirmation and drift guards with real install and activation state."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp import ServerDisconnectedError, web
from aiohttp.test_utils import TestClient, TestServer, make_mocked_request
from dinkster_registry import InstallError, Lockfile, PlanRecord
from dinkster_server import STATE_KEY, PathRedactor, create_app
from dinkster_server.auth import PRINCIPAL_KEY, Principal, StaticBearerAuthenticator

import dinkster.pack_install_api as routes
from dinkster.compose import ServingComposer
from dinkster.installer import Installer, lock_local_pack
from tests.async_scenarios import run_scenario
from tests.test_installer import write_pack


@pytest.mark.parametrize("authenticated", [False, True])
@pytest.mark.parametrize("cancel_apply", [0, 1, 2])
def test_pack_install_confirmations_and_live_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, authenticated: bool, cancel_apply: int
) -> None:
    async def scenario() -> None:
        installer = Installer(tmp_path / "root", accelerator="cpu")
        source = write_pack(tmp_path / "source", "demo")
        entry, _ = lock_local_pack(source, installer.artifacts_dir)
        target = Lockfile.of([entry])
        spawn = asyncio.create_subprocess_exec
        published = asyncio.Event()
        finish_manager = asyncio.Event()
        request_task: asyncio.Task[Any] | None = None

        # Substitute only manager I/O and venv provisioning. Real Installer
        # publication and whole-generation activation still run below.
        async def command(*arguments: str, **_kwargs: Any) -> object:
            if "dinkster.manager" not in arguments:
                return await spawn(*arguments, **_kwargs)
            action = arguments[arguments.index("--root") + 2]
            path = Path(arguments[-1])
            if action == "install":
                current = installer.current_lockfile()
                record = PlanRecord(
                    target=target,
                    base=current.record_digest() if current else None,
                    venvs=False,
                    accelerator="cpu",
                )
                path.write_text(record.record_json())
            else:
                record = PlanRecord.from_record_json(path.read_text())
                installer.apply(record.target, expected_base=record.base, venvs=False)

            async def communicate() -> tuple[bytes, bytes]:
                if action == "apply" and cancel_apply:
                    published.set()
                    await finish_manager.wait()
                return b"add: demo 0.0.0 digest [registry] venv: new\n", b""

            return type(
                "Process",
                (),
                {
                    "returncode": 0,
                    "communicate": AsyncMock(side_effect=communicate),
                },
            )()

        monkeypatch.setattr(routes.asyncio, "create_subprocess_exec", command)
        composer = ServingComposer(worker_env={"PYTHONPATH": str(source)})
        composition = composer.composition
        auth = (
            StaticBearerAuthenticator(
                {
                    "test-operator": Principal(
                        "operator", grants={"local": frozenset({"settings:write"})}
                    ),
                    "test-reader": Principal("reader", grants={"local": frozenset({"nodes:read"})}),
                    "test-agent": Principal(
                        "agent", kind="agent", grants={"local": frozenset({"settings:write"})}
                    ),
                    "test-delegate": Principal(
                        "delegate",
                        delegation_id="untrusted-delegation",
                        grants={"local": frozenset({"settings:write"})},
                    ),
                }
            )
            if authenticated
            else None
        )
        app = create_app(composition.make_engine, composition.schemas, authenticator=auth)

        @web.middleware
        async def capture_request(request: web.Request, handler: Any) -> web.StreamResponse:
            nonlocal request_task
            if request.path == "/api/packs/install/apply":
                request_task = asyncio.current_task()
            return await handler(request)

        app.middlewares.append(capture_request)
        busy = asyncio.Lock()
        routes.add_pack_install_routes(app, composer, installer, busy=busy, redactor=PathRedactor())
        client = TestClient(
            TestServer(app),
            headers={"Authorization": "Bearer test-operator"} if authenticated else {},
        )
        await client.start_server()
        try:
            selection = {"packs": [{"pack": "demo", "version": "1.0.0"}]}
            if authenticated:
                for token in ("test-reader", "test-agent", "test-delegate"):
                    for endpoint, body in (
                        ("plan", selection),
                        ("apply", {"confirmation": "untrusted"}),
                    ):
                        denied = await client.post(
                            f"/api/packs/install/{endpoint}",
                            json=body,
                            headers={"Authorization": f"Bearer {token}"},
                        )
                        assert denied.status == 403
                assert installer.current_number() is None
            for invalid in (
                {"packs": []},
                {"packs": [{"pack": "./demo", "version": "1.0.0"}]},
                {"packs": [{"pack": "demo", "version": "latest"}]},
                {"packs": [{"pack": "demo", "version": "1.0.0", "source": "file:///x"}]},
                {**selection, "root": "/other"},
            ):
                response = await client.post("/api/packs/install/plan", json=invalid)
                assert response.status == 400
            response = await client.post(
                "/api/packs/install/plan",
                json=selection,
                headers={"X-Dinkster-Actor-Kind": "agent"},
            )
            assert response.status == 403
            first = await (await client.post("/api/packs/install/plan", json=selection)).json()
            assert installer.current_number() is None
            assert first["plan"]["base"] is None
            assert first["expiresInSeconds"] == 900
            assert "venv: new" in first["review"]
            second = await (await client.post("/api/packs/install/plan", json=selection)).json()
            response = await client.post(
                "/api/packs/install/apply", json={"confirmation": first["confirmation"]}
            )
            assert response.status == 409
            assert (await response.json())["error"] == "invalid-confirmation"
            await busy.acquire()
            response = await client.post(
                "/api/packs/install/apply", json={"confirmation": second["confirmation"]}
            )
            assert (await response.json())["error"] == "install-in-progress"
            busy.release()
            if cancel_apply:
                applying = asyncio.ensure_future(
                    client.post(
                        "/api/packs/install/apply", json={"confirmation": second["confirmation"]}
                    )
                )
                await asyncio.wait_for(published.wait(), timeout=10)
                assert request_task is not None
                request_task.cancel()
                if cancel_apply == 2:
                    # The handler resumes from the first cancel before this callback runs.
                    asyncio.get_running_loop().call_soon(request_task.cancel)
                assert busy.locked()
                finish_manager.set()
                with pytest.raises(ServerDisconnectedError):
                    await applying
                assert installer.current_number() == 1
                assert "demo" in composer.pack_specs()
                assert app[STATE_KEY].schema_epoch > 0
                assert not busy.locked()
                return
            response = await client.post(
                "/api/packs/install/apply", json={"confirmation": second["confirmation"]}
            )
            assert response.status == 200, await response.text()
            applied = await response.json()
            assert applied["generation"] == 1
            assert applied["activation"]["added"] == ["demo"]
            assert applied["activation"]["failed"] == {}
            assert "demo" in composer.pack_specs()
            assert app[STATE_KEY].schema_epoch == applied["activation"]["epoch"]
            response = await client.post(
                "/api/packs/install/apply", json={"confirmation": second["confirmation"]}
            )
            assert (await response.json())["error"] == "invalid-confirmation"
            third = await (await client.post("/api/packs/install/plan", json=selection)).json()
            installer.apply(Lockfile(), venvs=False)
            response = await client.post(
                "/api/packs/install/apply", json={"confirmation": third["confirmation"]}
            )
            assert (await response.json())["error"] == "plan-stale"
            assert installer.current_lockfile() == Lockfile()
            assert "demo" in composer.pack_specs()
        finally:
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
        installer = Installer(tmp_path / "root", accelerator="cpu")
        routes.add_pack_install_routes(
            app, composer, installer, busy=asyncio.Lock(), redactor=PathRedactor()
        )
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
                with pytest.raises(web.HTTPForbidden) as denied:
                    await resolved.handler(request)
                assert denied.value.text == '{"error":"local-operator-required"}'
            assert installer.current_number() is None
        finally:
            await composer.close()

    run_scenario(scenario())


def test_reviewed_base_rechecked_after_writer_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import contextmanager

    installer = Installer(tmp_path / "root", accelerator="cpu")
    entry, _ = lock_local_pack(write_pack(tmp_path / "source", "demo"), installer.artifacts_dir)
    original = installer._locked
    # The concurrent writer uses its own instance, as a separate manager does.
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
