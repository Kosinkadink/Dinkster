from __future__ import annotations

import asyncio
import json
import os
import stat
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets.p2p_global import provider_declarations
from dinkster_p2p import LanNetworkPolicy, P2PManagerError
from dinkster_p2p import runtime as runtime_module
from dinkster_p2p.runtime import SidecarError, SidecarRuntime

from dinkster import lan_p2p, seed
from dinkster.seed import SeedService, parser
from tests.p2p_global_fixtures import build_provider_fixture


@pytest.fixture
def native_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    # Keep authority, verification, manager operations and native handles real; replace IPC only.
    monkeypatch.setattr(runtime_module, "current_lan_policy", lambda: LanNetworkPolicy(()))
    monkeypatch.setattr(lan_p2p, "current_lan_policy", lambda: LanNetworkPolicy(()))
    monkeypatch.setattr(SidecarRuntime, "_set_global_network_plan", lambda *_args: None)
    models = tmp_path / "models"
    models.mkdir()
    service = SeedService(
        parser().parse_args(
            [
                "--store",
                str(models),
                "--state-dir",
                str(tmp_path / "state"),
                "--provider-url",
                "https://provider.invalid/export?private=value",
                "--provider-id",
                "fixture",
                "--network-cost",
                "unmetered",
            ]
        )
    )
    runtime_args = {
        "state_root": service.vault.root / ".p2p",
        "vault_root": service.vault.root,
        "installation_root": None,
        "settings": service.settings,
    }
    harness = SimpleNamespace(
        service=service, runtime=SidecarRuntime(**runtime_args), rows=(), requests=[]
    )
    controller = service.controller
    manager = controller._manager
    controller._settings = dict(service.settings)
    controller._discovery = AsyncMock()
    manager._settings = dict(service.settings)
    manager._started = True
    monkeypatch.setattr(manager, "_process", object())

    async def request(operation: str, body: Any) -> Any:
        harness.requests.append((operation, body))
        try:
            return harness.runtime.operate(operation, body)
        except SidecarError as error:
            raise P2PManagerError(str(error)) from error

    def authority() -> Any:
        declarations = tuple(
            decision.declaration
            for fixture in harness.rows
            for decision in provider_declarations(
                fixture.snapshot,
                trusted_provider_ids=frozenset({fixture.snapshot.provider_id}),
                now=fixture.observed_at,
            )
            if decision.declaration is not None
        )
        return declarations, (), tuple(row.snapshot for row in harness.rows), True

    def publish(*rows: Any) -> None:
        harness.rows = rows
        service.mapped = service.store.refresh(tuple(row.snapshot for row in rows))

    def restart() -> None:
        harness.runtime.close()
        harness.runtime = SidecarRuntime(**{**runtime_args, "settings": service.settings})
        manager._global_authorizations.clear()
        controller._seed_leases.clear()
        controller._authority_signature = None

    harness.publish = publish
    harness.restart = restart
    monkeypatch.setattr(manager, "_request_locked", request)
    monkeypatch.setattr(manager, "_request_with_recovery_locked", request)
    monkeypatch.setattr(controller, "_authority_input", authority)
    monkeypatch.setattr(service, "start", AsyncMock())
    monkeypatch.setattr(service, "close", AsyncMock())
    try:
        yield harness
    finally:
        harness.runtime.close()


@pytest.mark.parametrize("scope,cap", [("lan-only", 1), ("lan-and-internet", 2)])
@pytest.mark.parametrize("replacement_count", [1, 3])
def test_replacement_retires_both_scopes_before_admission(
    native_service: Any, tmp_path: Path, scope: str, cap: int, replacement_count: int
) -> None:
    h = native_service
    service = h.service
    old = build_provider_fixture(tmp_path / "models" / "old", payload_size=32)
    replacements = [
        build_provider_fixture(tmp_path / "models" / str(index), payload_size=33 + index)
        for index in range(replacement_count)
    ]

    async def scenario() -> None:
        h.publish(old)
        service.settings.update(scope=scope, maxActiveSeeds=cap)
        await service.activity.update(service.settings)
        assert len(h.runtime._leases) == cap
        old_ids = set(h.runtime._leases)
        h.requests.clear()
        h.publish(*replacements)
        if replacement_count == 1:
            await service.controller.reconcile(local_files_changed=True)
            assert len(h.runtime._leases) == cap
        else:
            with pytest.raises(P2PManagerError, match="limit reached"):
                await service.controller.reconcile(local_files_changed=True)
        assert old_ids.isdisjoint(h.runtime._leases)
        assert old_ids.isdisjoint(service.controller._manager._global_authorizations)
        assert old.descriptor.asset_digest not in service.controller._seed_leases
        assert old.descriptor.asset_digest not in service.controller._seed_mappings
        first_grant = next(i for i, (op, _) in enumerate(h.requests) if op.startswith("grant"))
        assert old_ids <= {
            body["leaseId"] for op, body in h.requests[:first_grant] if op == "revoke"
        }
        assert len(h.runtime._leases) <= cap
        assert h.runtime.status()["listenPort"] is None

    asyncio.run(scenario())


@pytest.mark.parametrize("scope,cap", [("lan-only", 1), ("lan-and-internet", 2)])
def test_changed_authority_retires_same_digest_leases_before_regrant(
    native_service: Any, tmp_path: Path, scope: str, cap: int
) -> None:
    h = native_service
    old = build_provider_fixture(tmp_path / "models" / "old", payload_size=32)
    replacement = replace(
        old,
        snapshot=replace(old.snapshot, source_revision="fixture-r2"),
    )

    async def scenario() -> None:
        h.publish(old)
        h.service.settings.update(scope=scope, maxActiveSeeds=cap)
        await h.service.activity.update(h.service.settings)
        old_leases = dict(h.runtime._leases)
        assert len(old_leases) == cap
        h.requests.clear()
        h.publish(replacement)
        await h.service.controller.reconcile(local_files_changed=True)
        first_grant = next(i for i, (op, _) in enumerate(h.requests) if op.startswith("grant"))
        assert set(old_leases) <= {
            body["leaseId"] for op, body in h.requests[:first_grant] if op == "revoke"
        }
        assert all(lease != old_leases.get(key) for key, lease in h.runtime._leases.items())
        assert len(h.runtime._leases) == cap

    asyncio.run(scenario())


def test_refreshed_expiry_renews_seed_leases_without_revocation(
    native_service: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = native_service
    current = build_provider_fixture(tmp_path / "models" / "current", payload_size=32)
    refreshed = replace(
        current,
        observed_at=current.observed_at + 60,
        snapshot=replace(
            current.snapshot,
            refreshed_at=current.snapshot.refreshed_at + 60,
        ),
    )

    async def scenario() -> None:
        h.publish(current)
        await h.service.activity.update(h.service.settings)
        assert len(h.runtime._leases) == 2
        h.requests.clear()

        h.publish(refreshed)
        h.service.controller._clock = lambda: refreshed.observed_at
        h.service.controller._grants._clock = lambda: refreshed.observed_at
        monkeypatch.setattr("dinkster_p2p.global_leases.time.time", lambda: refreshed.observed_at)
        await h.service.controller.reconcile(local_files_changed=True)

        assert not any(operation == "revoke" for operation, _body in h.requests), h.requests
        assert [operation for operation, _body in h.requests].count("grant-seed") == 1
        assert [operation for operation, _body in h.requests].count("grant-global") == 1
        assert len(h.runtime._leases) == 2
        assert h.service.store.local_path_for(current.descriptor.asset_digest) is not None

    asyncio.run(scenario())


@pytest.mark.parametrize("still_metered", [False, True])
def test_explicit_enable_recovers_only_current_authorized_seeds_after_restart(
    native_service: Any, tmp_path: Path, still_metered: bool
) -> None:
    h = native_service
    service = h.service
    current = build_provider_fixture(tmp_path / "models" / "current", payload_size=32)
    omitted = build_provider_fixture(tmp_path / "models" / "omitted", payload_size=33)

    async def scenario() -> None:
        h.publish(current, omitted)
        service.settings["networkCostOverride"] = "metered"
        await service.activity.update(service.settings)
        assert h.runtime._activity[current.descriptor.asset_digest].global_resume_required
        service.settings["networkCostOverride"] = "metered" if still_metered else "unmetered"
        h.restart()
        await service.activity.update(service.settings)
        await service.activity.reconcile_network_policy()
        await service.controller.reconcile(local_files_changed=True)
        assert h.runtime._activity[current.descriptor.asset_digest].global_resume_required
        assert not any(op == "resume-transfer" for op, _ in h.requests)
        h.publish(current)
        # Supply even the removed digest: authority reconciliation must exclude it.
        service.mapped = (current.descriptor.asset_digest, omitted.descriptor.asset_digest)
        async with TestClient(TestServer(service.application())) as client:
            response = await client.post("/enabled", json=True)
            assert response.status == (409 if still_metered else 200), await response.text()
        resumed = [body["digest"] for op, body in h.requests if op == "resume-transfer"]
        assert resumed == ([] if still_metered else [current.descriptor.asset_digest])
        assert (
            h.runtime._activity[current.descriptor.asset_digest].global_resume_required
            == still_metered
        )
        assert h.runtime._activity[omitted.descriptor.asset_digest].global_resume_required
        assert all(
            lease.digest != omitted.descriptor.asset_digest for lease in h.runtime._leases.values()
        )
        assert h.runtime.status()["listenPort"] is None

    asyncio.run(scenario())


@pytest.mark.parametrize("fail_directory_sync", [False, True])
def test_disable_fsyncs_parent_before_success(
    native_service: Any, monkeypatch: pytest.MonkeyPatch, fail_directory_sync: bool
) -> None:
    service = native_service.service
    # No native leases exist in this persistence test; leave the native manager alive for teardown.
    monkeypatch.setattr(service.activity, "update", AsyncMock())
    actual_fsync = os.fsync
    synced: list[bool] = []

    def fsync(fd: int) -> None:
        directory = stat.S_ISDIR(os.fstat(fd).st_mode)
        synced.append(directory)
        if directory:
            assert json.loads((service.state / "enabled.json").read_text()) is False
            if fail_directory_sync:
                raise OSError("private persistence exception")
        actual_fsync(fd)

    async def scenario() -> None:
        async with TestClient(TestServer(service.application())) as client:
            with monkeypatch.context() as patch:
                patch.setattr(os, "fsync", fsync)
                response = await client.post("/enabled", json=False)
            assert response.status == (503 if fail_directory_sync and os.name != "nt" else 200)
            assert synced == ([False, True] if os.name != "nt" else [False])
            assert "private persistence exception" not in await response.text()
        assert SeedService(service.args).settings["seedingEnabled"] is False

    asyncio.run(scenario())


def test_bootstrap_and_nested_status_errors_hide_provider_url(native_service: Any) -> None:
    service = native_service.service

    async def scenario() -> None:
        async def unavailable(_request: web.Request) -> web.Response:
            raise web.HTTPServiceUnavailable()

        app = web.Application()
        app.router.add_get("/private-path", unavailable)
        async with TestServer(app) as provider:
            service.args.provider_url = str(provider.make_url("/private-path?private=query"))
            await service.refresh()
            status = await service.status()
            assert status["error"] == "provider-refresh-failed"
            assert service.args.provider_url not in json.dumps(status)
            assert "private-path" not in json.dumps(status)
            assert "private=query" not in json.dumps(status)
        service.controller._manager._last_error = service.args.provider_url
        assert (await service.status())["p2p"]["lastError"] == "operation-failed"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "loop_name,operation,code",
    [
        ("_refresh_loop", "refresh", "seed-reconcile-failed"),
        ("_network_loop", "reconcile_network_policy", "network-policy-failed"),
    ],
)
def test_background_errors_are_categories(
    native_service: Any, monkeypatch: pytest.MonkeyPatch, loop_name: str, operation: str, code: str
) -> None:
    service = native_service.service
    target = service if operation == "refresh" else service.activity
    monkeypatch.setattr(
        target, operation, AsyncMock(side_effect=RuntimeError("private arbitrary text"))
    )
    monkeypatch.setattr(asyncio, "sleep", AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(getattr(service, loop_name)())
    assert service.error == code


def test_http_and_cli_errors_are_sanitized(
    native_service: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    service = native_service.service

    async def scenario() -> None:
        async with TestClient(TestServer(service.application())) as client:
            monkeypatch.setattr(
                service.controller, "status", AsyncMock(side_effect=RuntimeError("private"))
            )
            response = await client.get("/status")
            assert response.status == 503 and await response.json() == {
                "error": "status-unavailable"
            }
            monkeypatch.setattr(
                service.activity, "update", AsyncMock(side_effect=RuntimeError("private"))
            )
            response = await client.post("/enabled", json=False)
            assert response.status == 503 and await response.json() == {"error": "control-failed"}

        async def status(_request: web.Request) -> web.Response:
            return web.json_response({"error": "private", "p2p": {"lastError": "private"}})

        async def refused(_request: web.Request) -> web.Response:
            return web.json_response({"error": "resume-refused"}, status=409)

        app = web.Application()
        app.router.add_get("/status", status)
        app.router.add_post("/enabled", refused)
        async with TestServer(app) as server:
            (service.state / "status-port").write_text(str(server.port))
            args = parser().parse_args(["status", "--state-dir", str(service.state)])
            await seed.run(args)
            output = json.loads(capsys.readouterr().out)
            assert output == {"error": "operation-failed", "p2p": {"lastError": "operation-failed"}}
            args.command = "enable"
            with pytest.raises(SystemExit) as failed:
                await seed.run(args)
            assert failed.value.code == 1
            assert json.loads(capsys.readouterr().out) == {"error": "resume-refused"}

    asyncio.run(scenario())
    monkeypatch.setattr(seed, "run", AsyncMock(side_effect=RuntimeError("private CLI text")))
    monkeypatch.setattr(seed.sys, "argv", ["dinkster-seed", "status"])
    with pytest.raises(SystemExit) as exit_info:
        seed.main()
    assert exit_info.value.code == 1
    assert capsys.readouterr().err == '{"error": "command-failed"}\n'
    monkeypatch.setattr(seed.sys, "argv", ["dinkster-seed", "--listen-port", "private CLI text"])
    with pytest.raises(SystemExit):
        seed.main()
    assert "private CLI text" not in capsys.readouterr().err
