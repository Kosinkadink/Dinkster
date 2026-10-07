"""Disabled feature boundaries remain closed even with the P2P extra installed."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_server import create_app
from dinkster_server.settings import normalize_features
from test_server import SCHEMAS, make_engine

from dinkster.p2p_plugin import disabled_main, load_p2p_plugin


def test_disabled_plugin_never_imports_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_import(name: str) -> None:
        raise AssertionError(f"unexpected optional import: {name}")

    monkeypatch.setattr("dinkster.p2p_plugin.importlib.import_module", reject_import)
    assert load_p2p_plugin() is None
    assert load_p2p_plugin(enabled=False) is None
    with pytest.raises(SystemExit, match="P2P is disabled"):
        disabled_main()


def test_server_import_and_disabled_routes_without_optional_imports() -> None:
    code = """
import sys
class RejectP2P:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'dinkster_p2p' or fullname.startswith('dinkster_p2p.'):
            raise AssertionError('disabled server imported P2P')
sys.meta_path.insert(0, RejectP2P())
import dinkster.serve
from dinkster.p2p_plugin import load_p2p_plugin
assert load_p2p_plugin() is None
"""
    subprocess.run([sys.executable, "-c", code], check=True)

    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS)
        assert not any("p2p" in str(route.resource) for route in app.router.routes())
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            for path in (
                "/api/p2p/status",
                "/api/p2p/transfers/anything",
                "/dinkster-p2p/v1/mappings/anything",
                "/dinkster-p2p/v2/mappings/anything",
                "/api/templates",
                "/api/packs/foundation/templates/loop",
                "/api/packs/foundation/templates/loop/thumbnail",
            ):
                assert (await client.get(path)).status == 404, path
            payload = await (await client.get("/api/settings")).json()
            assert payload["features"] == {
                "p2p": {"enabled": False},
                "templates": {"enabled": False},
            }
        finally:
            await client.close()

    asyncio.run(scenario())


def test_cli_entry_points_do_not_import_optional_runtime() -> None:
    root = Path(__file__).parents[1]
    for entry in ("dinkster-seed", "dinkster-p2p-diagnostics"):
        result = subprocess.run(
            [str(root / ".venv" / "bin" / entry)], capture_output=True, text=True, check=False
        )
        assert result.returncode != 0
        assert result.stderr.strip() == "P2P is disabled"


def test_feature_flags_require_boolean_opt_in() -> None:
    assert normalize_features({}) == {"p2p": {"enabled": False}, "templates": {"enabled": False}}
    assert normalize_features({"templates": {"enabled": True}})["templates"]["enabled"]
    with pytest.raises(ValueError, match="boolean"):
        normalize_features({"p2p": {"enabled": "false"}})
