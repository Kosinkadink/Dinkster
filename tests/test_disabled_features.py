"""Only shipped entry points and explicitly enabled template routes exist."""

from __future__ import annotations

import asyncio
from importlib.metadata import distribution

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_server import create_app
from dinkster_server.settings import normalize_features
from test_server import SCHEMAS, make_engine


def test_server_exposes_only_supported_features_and_keeps_templates_disabled() -> None:
    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS)
        namespaces = {
            "/".join(route.resource.canonical.split("/")[:3]) for route in app.router.routes()
        }
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


def test_installed_entry_points_match_the_shipped_commands() -> None:
    assert {
        entry.name
        for entry in distribution("dinkster").entry_points
        if entry.group == "console_scripts"
    } == {"dinkster", "dinkster-serve", "dinkster-pack", "dinkster-isolated", "dinkster-port"}


def test_feature_flags_require_boolean_opt_in() -> None:
    assert normalize_features({}) == {"templates": {"enabled": False}}
    assert normalize_features({"templates": {"enabled": True}})["templates"]["enabled"]
    with pytest.raises(ValueError, match="boolean"):
        normalize_features({"templates": {"enabled": "false"}})
