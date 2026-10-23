"""Only shipped entry points and explicitly enabled template routes exist."""

from __future__ import annotations

import asyncio
from importlib.metadata import distribution
from pathlib import Path
from unittest.mock import patch

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import ProvenanceStore, ResolverSubscriptionStore
from dinkster_server import create_app
from dinkster_server.settings import RuntimeSettings, normalize_features
from test_server import SCHEMAS, make_engine

from dinkster.resolver_api import RESOLVER_INDEXES_KEY, add_resolver_index_routes


def _settings(*, resolver_enabled: bool) -> RuntimeSettings:
    return RuntimeSettings(
        {
            "memory-budgets": {},
            "memory-headroom": 256 * 1024**2,
            "aimdo-policy": "auto",
            "dtype-policy": {"diffusion": "auto", "textEncoder": "auto", "vae": "auto"},
            "fp8-matmul": False,
            "worker-comfy-args": (),
            "jobs": {"maxRunningJobs": 1},
            "logging": {"level": "info", "overrides": {}},
        },
        {
            "memory-budgets": "default",
            "memory-headroom": "default",
            "aimdo-policy": "default",
            "dtype-policy": "default",
            "fp8-matmul": "default",
            "worker-comfy-args": "default",
            "jobs": "default",
            "logging": "default",
        },
        features={"resolver": {"enabled": resolver_enabled}},
    )


def test_server_exposes_only_supported_features_and_keeps_templates_disabled() -> None:
    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS)
        namespaces: set[str] = set()
        for route in app.router.routes():
            resource = route.resource
            assert resource is not None
            namespaces.add("/".join(resource.canonical.split("/")[:3]))
        assert namespaces == {
            "/api/auth",
            "/api/choices",
            "/api/composition",
            "/api/diagnostics",
            "/api/docs",
            "/api/events",
            "/api/extension-assets",
            "/api/extensions",
            "/api/health",
            "/api/jobs",
            "/api/nodes",
            "/api/packs",
            "/api/principals",
            "/api/queue",
            "/api/settings",
            "/api/values",
            "/api/workers",
            "/cache/trim",
            "/memory/free",
            "/memory/reserve",
            "/memory/reset-peak",
            "/memory/shed",
            "/memory/status",
            "/packs/{pack_id}",
        }
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            for path in (
                "/api/templates",
                "/api/packs/foundation/templates/loop",
                "/api/packs/foundation/templates/loop/thumbnail",
            ):
                assert (await client.get(path)).status == 404, path
            payload = await (await client.get("/api/settings")).json()
            assert payload["features"] == {
                "resolver": {"enabled": False},
                "templates": {"enabled": False},
            }
            assert payload["categories"]["available"] == [
                "memory-budgets",
                "memory-headroom",
                "aimdo-policy",
                "dtype-policy",
                "fp8-matmul",
                "worker-comfy-args",
                "jobs",
                "logging",
            ]
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("resolver_enabled", [False, True])
def test_resolver_index_routes_require_explicit_opt_in(
    tmp_path: Path, resolver_enabled: bool
) -> None:
    async def scenario() -> None:
        store = ResolverSubscriptionStore(
            tmp_path / "resolver-indexes.json",
            ProvenanceStore(tmp_path / "provenance.json"),
        )
        app = create_app(
            make_engine, SCHEMAS, settings=_settings(resolver_enabled=resolver_enabled)
        )
        add_resolver_index_routes(app, store)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.get("/api/assets/resolver-indexes")
            assert response.status == (200 if resolver_enabled else 404)
            assert (RESOLVER_INDEXES_KEY in app) is resolver_enabled
        finally:
            await client.close()

    asyncio.run(scenario())


def test_disabled_resolver_routes_do_not_bootstrap_or_refresh(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = ResolverSubscriptionStore(
            tmp_path / "resolver-indexes.json",
            ProvenanceStore(tmp_path / "provenance.json"),
        )
        app = create_app(make_engine, SCHEMAS, settings=_settings(resolver_enabled=False))
        with (
            patch.object(store, "bootstrap_official") as bootstrap,
            patch.object(store, "refresh") as refresh,
        ):
            add_resolver_index_routes(app, store)
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                assert (await client.get("/api/assets/resolver-indexes")).status == 404
                bootstrap.assert_not_called()
                refresh.assert_not_called()
            finally:
                await client.close()

    asyncio.run(scenario())


def test_installed_entry_points_match_the_shipped_commands() -> None:
    assert {
        entry.name
        for entry in distribution("dinkster").entry_points
        if entry.group == "console_scripts"
    } == {"dinkster", "dinkster-serve", "dinkster-pack", "dinkster-isolated", "dinkster-port"}


def test_feature_flags_require_boolean_opt_in() -> None:
    assert normalize_features({}) == {
        "resolver": {"enabled": False},
        "templates": {"enabled": False},
    }
    assert normalize_features({"templates": {"enabled": True}})["templates"]["enabled"]
    assert normalize_features({"resolver": {"enabled": True}})["resolver"]["enabled"]
    for feature in ("resolver", "templates"):
        with pytest.raises(ValueError, match="boolean"):
            normalize_features({feature: {"enabled": "false"}})
