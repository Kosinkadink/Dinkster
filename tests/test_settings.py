"""Runtime settings permissions, validation, application, and persistence."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import P2PPluginRegistration
from dinkster_assets.p2p_settings import (
    default_p2p_settings as shared_default_p2p_settings,
)
from dinkster_assets.p2p_settings import (
    normalize_p2p_settings as shared_normalize_p2p_settings,
)
from dinkster_memory import MemoryGovernor
from dinkster_schema import LOG_LEVEL_ENV, LOG_OVERRIDES_ENV
from dinkster_server import (
    ComfyArgumentError,
    P2PSettingsError,
    Principal,
    RuntimeSettings,
    SettingsError,
    comfy_cpu_args,
    comfy_dtype_args,
    create_app,
    load_settings,
    validate_comfy_args,
)
from dinkster_server import (
    default_p2p_settings as server_default_p2p_settings,
)
from dinkster_server import (
    normalize_p2p_settings as server_normalize_p2p_settings,
)
from dinkster_server.p2p_plugin import default_p2p_settings
from test_server import SCHEMAS, StubAuthenticator, make_engine


def settings(
    *,
    granted: frozenset[str] = frozenset(),
    path: Path | None = None,
    persisted: Mapping[str, object] | None = None,
    features: object = None,
) -> RuntimeSettings:
    return RuntimeSettings(
        {
            "memory-budgets": {"ram": 1024},
            "memory-headroom": 256 * 1024**2,
            "aimdo-policy": "auto",
            "dtype-policy": {
                "diffusion": "auto",
                "textEncoder": "auto",
                "vae": "auto",
            },
            "fp8-matmul": False,
            "worker-comfy-args": (),
            "jobs": {"maxRunningJobs": 1},
            "logging": {"level": "info", "overrides": {}},
            "p2p": default_p2p_settings(),
        },
        {
            "memory-budgets": "config",
            "memory-headroom": "default",
            "aimdo-policy": "default",
            "dtype-policy": "default",
            "fp8-matmul": "default",
            "worker-comfy-args": "default",
            "jobs": "default",
            "logging": "default",
            "p2p": "default",
        },
        granted=granted,
        path=path,
        persisted_values=persisted,
        features=features,
    )


def test_server_p2p_settings_use_shared_validation_with_and_without_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration_module = importlib.import_module("dinkster_assets.p2p_plugin")
    plugin_defaults = shared_default_p2p_settings()

    monkeypatch.setattr(registration_module, "_registration", None)
    assert server_default_p2p_settings() == plugin_defaults
    assert server_normalize_p2p_settings(plugin_defaults) == shared_normalize_p2p_settings(
        plugin_defaults
    )

    registration = P2PPluginRegistration(
        default_settings=shared_default_p2p_settings,
        normalize_settings=shared_normalize_p2p_settings,
        lan_interfaces=lambda: (),
    )
    monkeypatch.setattr(registration_module, "_registration", registration)
    assert server_default_p2p_settings() == plugin_defaults
    assert server_normalize_p2p_settings(plugin_defaults) == shared_normalize_p2p_settings(
        plugin_defaults
    )
    runtime = settings(granted=frozenset({"p2p"}))

    def reject_settings(_value: object) -> dict[str, object]:
        raise ValueError("plugin rejected settings")

    monkeypatch.setattr(
        registration_module,
        "_registration",
        P2PPluginRegistration(
            default_settings=shared_default_p2p_settings,
            normalize_settings=reject_settings,
            lan_interfaces=lambda: (),
        ),
    )
    with pytest.raises(P2PSettingsError, match="plugin rejected settings"):
        server_normalize_p2p_settings(plugin_defaults)

    async def scenario() -> None:
        app = create_app(
            make_engine,
            SCHEMAS,
            settings=runtime,
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.put("/api/settings/p2p", json=plugin_defaults)
            assert response.status == 400
            assert await response.json() == {
                "error": "invalid-settings",
                "category": "p2p",
                "message": "plugin rejected settings",
            }
        finally:
            await client.close()

    asyncio.run(scenario())


def test_server_p2p_settings_match_the_plugin_with_and_without_registration(
    monkeypatch: pytest.MonkeyPatch,
    pytestconfig: pytest.Config,
) -> None:
    if not pytestconfig.getoption("--p2p"):
        pytest.skip("plugin comparison requires --p2p")
    plugin = pytest.importorskip("dinkster_p2p")
    registration_module = importlib.import_module("dinkster_assets.p2p_plugin")
    defaults = plugin.default_p2p_settings()
    for registration in (
        None,
        P2PPluginRegistration(
            default_settings=plugin.default_p2p_settings,
            normalize_settings=plugin.normalize_p2p_settings,
            lan_interfaces=lambda: (),
        ),
    ):
        monkeypatch.setattr(registration_module, "_registration", registration)
        assert server_default_p2p_settings() == defaults
        assert server_normalize_p2p_settings(defaults) == plugin.normalize_p2p_settings(defaults)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("downloadsEnabled", 1),
        ("scope", "internet"),
        ("internetUploadBytesPerSecond", True),
        ("internetDownloadBytesPerSecond", -1),
        ("lanUploadBytesPerSecond", 2**31),
        ("stagingBudgetBytes", 2**53),
        ("stagingBudgetBytes", 1.5),
        ("maxActiveSeeds", 0),
        ("maxActiveSeeds", 4097),
        ("listenPort", -1),
        ("listenPort", 65536),
        ("internetSeedRatio", True),
        ("internetSeedRatio", float("inf")),
        ("internetSeedRatio", float("nan")),
        ("internetSeedRatio", -0.5),
        ("internetSeedRatio", 10**400),
    ],
)
def test_shared_p2p_validation_rejects_invalid_boundaries(field: str, invalid: object) -> None:
    with pytest.raises(P2PSettingsError, match=field):
        shared_normalize_p2p_settings({**shared_default_p2p_settings(), field: invalid})


def test_shared_p2p_validation_preserves_legacy_scope_and_budget_boundaries() -> None:
    legacy = {
        key: value
        for key, value in shared_default_p2p_settings().items()
        if key not in {"stagingBudgetBytes", "maxActiveSeeds", "listenPort"}
    }
    legacy.update(downloadsEnabled=True, scope="lan-only", internetSeedRatio=2)
    assert shared_normalize_p2p_settings(legacy) == {
        **legacy,
        "stagingBudgetBytes": 64 * 1024**3,
        "maxActiveSeeds": 64,
        "listenPort": 0,
    }
    assert "stagingBudgetBytes" not in legacy
    limits = {
        **legacy,
        "stagingBudgetBytes": 2**53 - 1,
        "maxActiveSeeds": 4096,
        "listenPort": 65535,
        "internetUploadBytesPerSecond": 2**31 - 1,
    }
    normalized = shared_normalize_p2p_settings(limits)
    assert normalized == limits
    assert type(normalized["internetSeedRatio"]) is float
    with pytest.raises(P2PSettingsError, match="exactly"):
        shared_normalize_p2p_settings({**limits, "dhtEnabled": True})
    with pytest.raises(P2PSettingsError, match="object"):
        shared_normalize_p2p_settings([])


def test_settings_get_is_ungated_and_reports_mutability() -> None:
    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS, settings=settings())
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.get("/api/settings")
            assert response.status == 200
            body = await response.json()
            assert body["categories"]["granted"] == []
            assert body["settings"]["memory-headroom"]["mutability"] == "live"
            assert body["settings"]["aimdo-policy"]["mutability"] == ("on-worker-restart")
            assert body["settings"]["fp8-matmul"] == {
                "value": False,
                "source": "default",
                "mutability": "on-worker-restart",
                "writable": False,
                "persistence": {"available": False, "persisted": False},
            }
            assert body["settings"]["worker-comfy-args"]["mutability"] == ("on-worker-restart")
            assert body["settings"]["worker-comfy-args"]["value"] == []
            assert body["settings"]["jobs"]["mutability"] == "live"
            assert body["settings"]["p2p"]["value"] == default_p2p_settings()
            assert body["settings"]["p2p"]["mutability"] == "live"
        finally:
            await client.close()

    asyncio.run(scenario())


def test_dtype_policy_maps_to_comfy_flags_without_auto_overrides() -> None:
    args = comfy_dtype_args({"diffusion": "bfloat16", "textEncoder": "auto", "vae": "float32"})
    assert args == (
        "--bf16-unet",
        "--fp32-vae",
    )
    assert validate_comfy_args(args, require_tuple=True) == args
    with pytest.raises(ComfyArgumentError, match="Dinkster dtype policy"):
        validate_comfy_args(("--bf16-unet",), require_tuple=True)


def test_cpu_policy_maps_to_owned_comfy_flag() -> None:
    args = comfy_cpu_args()
    assert args == ("--cpu",)
    assert validate_comfy_args(args, require_tuple=True) == args
    with pytest.raises(ComfyArgumentError, match="Dinkster memory/aimdo policy"):
        validate_comfy_args(("--cpu",), require_tuple=True)


def test_memory_headroom_put_pushes_live_persists_and_updates_reload_value(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        path = tmp_path / "settings.json"
        pushed: list[int] = []
        runtime = settings(
            granted=frozenset({"memory-headroom"}),
            path=path,
        )
        app = create_app(
            make_engine,
            SCHEMAS,
            settings=runtime,
            memory_headroom_changed=pushed.append,
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.put("/api/settings/memory-headroom", json="192M")
            assert response.status == 200
            assert (await response.json())["value"] == 192 * 1024**2
            assert pushed == [192 * 1024**2]
            assert load_settings(path)["memory-headroom"] == 192 * 1024**2
            assert runtime.memory_headroom == 192 * 1024**2
        finally:
            await client.close()

    asyncio.run(scenario())


def test_settings_put_permission_validation_and_happy_path_matrix() -> None:
    categories = {
        "memory-budgets": ({"ram": "2K"}, {"ram": -1}),
        "memory-headroom": ("128M", -1),
        "aimdo-policy": ("on", "sticky"),
        "dtype-policy": (
            {"diffusion": "bfloat16", "textEncoder": "float16", "vae": "float32"},
            {"diffusion": "bf16", "textEncoder": "float16", "vae": "float32"},
        ),
        "fp8-matmul": (True, "true"),
        "worker-comfy-args": (["--preview-size", "256"], ["--port=8188"]),
        "jobs": ({"maxRunningJobs": 2}, {"maxRunningJobs": 0}),
        "logging": (
            {"level": "debug", "overrides": {"dinkster.server": "warning"}},
            {"level": "loud", "overrides": {}},
        ),
        "p2p": (
            {**default_p2p_settings(), "downloadsEnabled": True},
            {**default_p2p_settings(), "scope": "internet"},
        ),
    }

    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 1024})
        denied_app = create_app(make_engine, SCHEMAS, governor=governor, settings=settings())
        denied = TestClient(TestServer(denied_app))
        await denied.start_server()
        try:
            for category, (valid, _) in categories.items():
                response = await denied.put(f"/api/settings/{category}", json=valid)
                assert response.status == 403
                assert await response.json() == {
                    "error": "settings-changes-disabled",
                    "category": category,
                    "granted": [],
                }
        finally:
            await denied.close()

        allowed_app = create_app(
            make_engine,
            SCHEMAS,
            governor=governor,
            settings=settings(granted=frozenset(categories)),
        )
        allowed = TestClient(TestServer(allowed_app))
        await allowed.start_server()
        try:
            for category, (valid, invalid) in categories.items():
                response = await allowed.put(f"/api/settings/{category}", json=invalid)
                assert response.status == 400, (category, await response.text())
                error = await response.json()
                assert error["error"] == "invalid-settings"
                assert error["category"] == category

                response = await allowed.put(f"/api/settings/{category}", json=valid)
                assert response.status == 200, (category, await response.text())
                section = await response.json()
                assert section["source"] == "runtime"
                assert section["writable"] is True
            assert governor.status()["ram"]["budgetBytes"] == 2048
        finally:
            await allowed.close()

    asyncio.run(scenario())


def test_settings_write_rbac_composes_with_operator_category_grants() -> None:
    async def scenario() -> None:
        granted_runtime = settings(granted=frozenset({"jobs"}))
        no_capability = TestClient(
            TestServer(
                create_app(
                    make_engine,
                    SCHEMAS,
                    settings=granted_runtime,
                    authenticator=StubAuthenticator(Principal("reader", {"scope": frozenset()})),
                )
            )
        )
        await no_capability.start_server()
        try:
            response = await no_capability.put(
                "/api/settings/jobs",
                json={"maxRunningJobs": 2},
                headers={"Authorization": "Bearer accepted"},
            )
            assert response.status == 403
            assert (await response.json())["error"] == "capability-required"
            assert granted_runtime.section("jobs")["value"] == {"maxRunningJobs": 1}
        finally:
            await no_capability.close()

        ungranted_runtime = settings()
        has_capability = TestClient(
            TestServer(
                create_app(
                    make_engine,
                    SCHEMAS,
                    settings=ungranted_runtime,
                    authenticator=StubAuthenticator(
                        Principal(
                            "operator",
                            {"scope": frozenset({"settings:write"})},
                        )
                    ),
                )
            )
        )
        await has_capability.start_server()
        try:
            response = await has_capability.put(
                "/api/settings/jobs",
                json={"maxRunningJobs": 2},
                headers={"Authorization": "Bearer accepted"},
            )
            assert response.status == 403
            assert await response.json() == {
                "error": "settings-changes-disabled",
                "category": "jobs",
                "granted": [],
            }
            assert ungranted_runtime.section("jobs")["value"] == {"maxRunningJobs": 1}
        finally:
            await has_capability.close()

    asyncio.run(scenario())


def test_settings_persistence_round_trip_is_atomic_and_preserves_prior_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "settings.json"
    initial = {"aimdo-policy": "on"}
    runtime = settings(granted=frozenset({"memory-headroom"}), path=path, persisted=initial)
    replaced: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def recording_replace(source: str | Path, destination: str | Path) -> None:
        source_path, destination_path = Path(source), Path(destination)
        assert source_path.name == "settings.json.tmp"
        assert source_path.is_file()
        replaced.append((source_path, destination_path))
        real_replace(source_path, destination_path)

    monkeypatch.setattr("dinkster_server.settings.os.replace", recording_replace)
    runtime.update("memory-headroom", "128M")

    assert replaced == [(path.with_name("settings.json.tmp"), path)]
    assert load_settings(path) == {
        "aimdo-policy": "on",
        "memory-headroom": 128 * 1024**2,
    }
    assert not path.with_name("settings.json.tmp").exists()


def test_settings_without_library_root_visibly_reports_unpersisted() -> None:
    runtime = settings(granted=frozenset({"aimdo-policy"}))
    section = runtime.update("aimdo-policy", "off")
    assert section["value"] == "off"
    assert section["persistence"] == {"available": False, "persisted": False}


def test_settings_logging_reapplies_and_reexports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, str]]] = []
    monkeypatch.setattr(
        "dinkster_server.settings.configure_logging",
        lambda level, *, overrides: calls.append((level, dict(overrides))),
    )
    monkeypatch.setenv(LOG_OVERRIDES_ENV, "dinkster.old=debug")
    runtime = settings(granted=frozenset({"logging"}))
    runtime.update("logging", {"level": "error", "overrides": {"dinkster.pack.a": "debug"}})
    assert calls == [("error", {"dinkster.pack.a": "debug"})]
    assert os.environ[LOG_LEVEL_ENV] == "error"
    assert os.environ[LOG_OVERRIDES_ENV] == "dinkster.pack.a=debug"

    runtime.update("logging", {"level": "info", "overrides": {}})
    assert LOG_OVERRIDES_ENV not in os.environ


@pytest.mark.parametrize(
    ("category", "value"),
    [
        ("memory-budgets", {"ram": True}),
        ("memory-headroom", "1.5G"),
        ("aimdo-policy", 1),
        ("fp8-matmul", 1),
        ("worker-comfy-args", "--preview-size"),
        ("jobs", {"maxRunningJobs": True}),
        ("logging", {"level": "info", "overrides": {"urllib3": "debug"}}),
        ("p2p", {**default_p2p_settings(), "seedingEnabled": "yes"}),
    ],
)
def test_settings_file_rejects_each_invalid_value_kind(
    tmp_path: Path, category: str, value: object
) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({category: value}), "utf-8")
    with pytest.raises(SettingsError):
        load_settings(path)


@pytest.mark.parametrize(
    ("argument", "flag", "owner"),
    [
        ("--listen", "--listen", "Dinkster server"),
        ("--front-end-version=x", "--front-end-version", "Dinkster server"),
        ("--reserve-vram=1", "--reserve-vram", "Dinkster memory/aimdo policy"),
        ("--lowvram", "--lowvram", "Dinkster memory/aimdo policy"),
        ("--front-end-v=x", "--front-end-v", "Dinkster server"),
        ("--disable-dynamic-vram", "--disable-dynamic-vram", "Dinkster memory/aimdo policy"),
    ],
)
def test_worker_comfy_args_put_deny_list_is_structured_and_precedes_mutation(
    tmp_path: Path, argument: str, flag: str, owner: str
) -> None:
    async def scenario() -> None:
        path = tmp_path / "settings.json"
        runtime = settings(
            granted=frozenset({"worker-comfy-args"}),
            path=path,
        )
        app = create_app(make_engine, SCHEMAS, settings=runtime)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.put("/api/settings/worker-comfy-args", json=[argument])
            assert response.status == 400
            body = await response.json()
            assert body["offendingFlag"] == flag
            assert body["owner"] == owner
            assert flag in body["message"]
            assert owner in body["message"]
            assert runtime.worker_comfy_args == ()
            assert not path.exists()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_worker_comfy_args_put_round_trip_persists_unknown_passthrough(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        path = tmp_path / "settings.json"
        runtime = settings(
            granted=frozenset({"worker-comfy-args"}),
            path=path,
        )
        app = create_app(make_engine, SCHEMAS, settings=runtime)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            value = ["--future-comfy-flag", "value"]
            response = await client.put("/api/settings/worker-comfy-args", json=value)
            assert response.status == 200
            assert (await response.json())["value"] == value
            assert runtime.worker_comfy_args == tuple(value)
            assert load_settings(path)["worker-comfy-args"] == tuple(value)
        finally:
            await client.close()

    asyncio.run(scenario())
