"""The dev widget/socket gallery: dev.gallery.* nodes, choice routes, and
the shipped template exercise every construct the native schema wire (v11)
can express, so the frontend can verify rendering/decode against a real
server instead of only its local fixtures.

What this proves: the gallery covers each native widget descriptor and
socket shape on the wire, its choice lists ride /api/choices/* (one
populated, one legally empty), the template document ships through the
development pack table, references only real gallery node types and ports,
and the whole surface is absent unless its manifest is composed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer
from dinkster_nodes_dev.gallery import (
    EMPTY_CHOICE_ID,
    GALLERY_TEMPLATE_ID,
    GALLERY_TEMPLATE_NAME,
    SAMPLERS,
    SAMPLERS_CHOICE_ID,
    gallery_template_bytes,
)
from dinkster_server import create_app

from dinkster.compose import PackSpec, compose_serving
from tests.async_scenarios import run_scenario

REPO_ROOT = Path(__file__).parent.parent
DEV_PACK_MANIFEST = REPO_ROOT / "packages" / "dinkster-nodes-dev" / "dinkster-pack.toml"


def _interface(wire: dict[str, object]) -> list[dict[str, object]]:
    entries = wire["interface"]
    assert isinstance(entries, list)
    return entries


def _entry(wire: dict[str, object], entry_id: str) -> dict[str, object]:
    for item in _interface(wire):
        if item.get("id") == entry_id:
            return item
    raise AssertionError(f"{wire['nodeType']}: no interface entry {entry_id!r}")


def test_gallery_choice_and_template_endpoints() -> None:
    """End to end over HTTP: /api/choices/* answers the populated and the
    legally-empty list, /api/templates indexes the gallery under the core
    pack, and the body endpoint serves the exact declared bytes."""

    async def scenario() -> None:
        from test_settings import settings

        composition = await compose_serving(
            [PackSpec(DEV_PACK_MANIFEST, trust_reserved=True)], include_default_packs=False
        )
        client = None
        try:
            app = create_app(
                composition.make_engine,
                composition.schemas,
                packs=composition.packs,
                node_packs=composition.node_packs,
                choices=composition.choices,
                lazy_choices=composition.lazy_choices,
                settings=settings(features={"templates": {"enabled": True}}),
            )
            client = TestClient(TestServer(app))
            await client.start_server()

            samplers = await (await client.get(f"/api/choices/{SAMPLERS_CHOICE_ID}")).json()
            assert samplers == list(SAMPLERS)
            empty = await (await client.get(f"/api/choices/{EMPTY_CHOICE_ID}")).json()
            assert empty == []

            listing = await (await client.get("/api/templates")).json()
            rows = {(row["pack"], row["id"]): row for row in listing["templates"]}
            row = rows[("dinkster-nodes-dev", GALLERY_TEMPLATE_ID)]
            assert row["name"] == GALLERY_TEMPLATE_NAME

            body = await client.get(
                f"/api/packs/dinkster-nodes-dev/templates/{GALLERY_TEMPLATE_ID}"
            )
            assert body.status == 200
            data = await body.read()
            assert data == gallery_template_bytes()
            assert row["digest"] == "sha256:" + hashlib.sha256(data).hexdigest()
        finally:
            if client is not None:
                await client.close()
            await composition.close()

    run_scenario(scenario())
