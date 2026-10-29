from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer
from dinkster_server import create_app
from dinkster_workers import load_manifest
from dinkster_workers.doctor import diagnose

from dinkster.compose import compose_serving, default_pack_spec

REPO_ROOT = Path(__file__).parents[1]
FOUNDATION_ROOT = REPO_ROOT / "packages" / "dinkster-nodes-foundation"
FOUNDATION_MANIFEST = FOUNDATION_ROOT / "dinkster-pack.toml"
GUIDE_PATH = FOUNDATION_ROOT / "docs/guides/loops/en.md"
TEMPLATE_IDS = (
    "loop-map-images",
    "loop-gather-image-batch",
    "loop-fold-scan-images",
    "loop-while-until",
    "loop-per-item-image-spawn",
)


def test_foundation_manifest_ships_executable_loop_guide_and_templates() -> None:
    manifest = load_manifest(FOUNDATION_MANIFEST)
    assert tuple(template.id for template in manifest.templates) == TEMPLATE_IDS
    for template in manifest.templates:
        document = json.loads(template.data)
        assert document["format"] == "dinkster-workflow"
        assert document["formatVersion"] == 1
        assert document["graphs"][document["root"]]["nodes"]
        assert template.digest == "sha256:" + hashlib.sha256(template.data).hexdigest()

    assert manifest.docs is not None
    assert manifest.docs.validation_problems == ()
    guide = next(page for page in manifest.docs.pages if page.kind == "guide")
    assert (guide.id, guide.locale, guide.title) == ("loops", "en", "Loops with real images")
    assert guide.data in GUIDE_PATH.read_bytes()
    for template_id in TEMPLATE_IDS:
        assert f'template = "{template_id}"'.encode() in guide.data

    report = diagnose(FOUNDATION_MANIFEST)
    assert report.ok

    spec = default_pack_spec("dinkster-nodes-foundation")
    assert spec.packs is not None
    info = spec.packs["dinkster-nodes-foundation"]
    assert tuple(template.id for template in info.templates) == TEMPLATE_IDS


def test_loop_templates_are_discoverable_and_load_byte_exactly() -> None:
    async def scenario() -> None:
        composition = await compose_serving()
        client = TestClient(
            TestServer(
                create_app(
                    composition.make_engine,
                    composition.schemas,
                    packs=composition.packs,
                    node_packs=composition.node_packs,
                    choices=composition.choices,
                    lazy_choices=composition.lazy_choices,
                )
            )
        )
        try:
            await client.start_server()
            catalog = await (await client.get("/api/templates")).json()
            rows = {
                row["id"]: row
                for row in catalog["templates"]
                if row["pack"] == "dinkster-nodes-foundation"
            }
            listed = tuple(template_id for template_id in TEMPLATE_IDS if template_id in rows)
            assert listed == TEMPLATE_IDS
            manifest = load_manifest(FOUNDATION_MANIFEST)
            for template in manifest.templates:
                response = await client.get(
                    f"/api/packs/dinkster-nodes-foundation/templates/{template.id}"
                )
                assert response.status == 200
                assert await response.read() == template.data
                assert rows[template.id]["digest"] == template.digest
        finally:
            await client.close()
            await composition.close()

    asyncio.run(scenario())
