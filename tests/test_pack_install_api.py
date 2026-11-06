"""HTTP confirmation and drift guards with real install and activation state."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_registry import InstallError, Lockfile, PlanRecord
from dinkster_server import STATE_KEY, PathRedactor, create_app

import dinkster.pack_install_api as routes
from dinkster.compose import ServingComposer
from dinkster.installer import Installer, lock_local_pack
from tests.async_scenarios import run_scenario
from tests.test_installer import write_pack


def test_pack_install_confirmations_and_live_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        installer = Installer(tmp_path / "root", accelerator="cpu")
        source = write_pack(tmp_path / "source", "demo")
        entry, _ = lock_local_pack(source, installer.artifacts_dir)
        target = Lockfile.of([entry])
        spawn = asyncio.create_subprocess_exec

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
            return type(
                "Process",
                (),
                {
                    "returncode": 0,
                    "communicate": AsyncMock(
                        return_value=(b"add: demo 0.0.0 digest [registry] venv: new\n", b"")
                    ),
                },
            )()

        monkeypatch.setattr(routes.asyncio, "create_subprocess_exec", command)
        composer = ServingComposer(worker_env={"PYTHONPATH": str(source)})
        composition = composer.composition
        app = create_app(composition.make_engine, composition.schemas)
        busy = asyncio.Lock()
        routes.add_pack_install_routes(app, composer, installer, busy=busy, redactor=PathRedactor())
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            selection = {"packs": [{"pack": "demo", "version": "1.0.0"}]}
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
