from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import cast

from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import digest_bytes
from dinkster_graph import (
    PORTS_NODE_ID,
    Graph,
    GraphNode,
    Link,
    RegionNode,
    RegionOutput,
    graph_to_wire,
)
from dinkster_schema import TypeExpr
from dinkster_server import HistoryStore, create_app
from dinkster_values import list_children
from dinkster_workers import load_manifest
from dinkster_workers.doctor import diagnose

from dinkster.compose import compose_serving, default_pack_spec

REPO_ROOT = Path(__file__).parents[1]
FOUNDATION_ROOT = REPO_ROOT / "packages" / "dinkster-nodes-foundation"
FOUNDATION_MANIFEST = FOUNDATION_ROOT / "dinkster-pack.toml"
TEMPLATE_PATH = FOUNDATION_ROOT / "src/dinkster_nodes_foundation/map_and_gather_template.json"
GUIDE_PATH = FOUNDATION_ROOT / "docs/guides/map-and-gather/en.md"
TEMPLATE_ID = "map-and-gather"
TEMPLATE_SHA256 = "04d97593e00f9b7e376a9ad37c255084e9361db5c83dd762c892b879f28e2275"
INT = TypeExpr.concrete("core.int")


def map_and_gather_graph(document: dict[str, object]) -> Graph:
    graphs = cast(dict[str, dict[str, object]], document["graphs"])
    root = graphs[cast(str, document["root"])]
    region = cast(dict[str, dict[str, object]], root["nodes"])["region"]
    body = graphs[cast(str, region["type"])[1:]]
    add = cast(dict[str, dict[str, object]], body["nodes"])["add"]
    values = cast(dict[str, object], region["values"])
    region_config = cast(dict[str, object], region["region"])
    assert region_config["kind"] == "map"
    return Graph(
        nodes={
            "region": RegionNode(
                kind="map",
                body=Graph(
                    nodes={
                        "add": GraphNode(
                            cast(str, add["type"]),
                            {
                                "a": Link(PORTS_NODE_ID, "left"),
                                "b": Link(PORTS_NODE_ID, "right"),
                            },
                        )
                    }
                ),
                ports={"left": INT, "right": INT},
                inputs={"left": values["left"], "right": values["right"]},
                element_ports=tuple(cast(list[str], region_config["elementPorts"])),
                outputs={"sum": RegionOutput(Link("add", "sum"))},
            )
        }
    )


def test_foundation_manifest_ships_map_and_gather_guide_and_template() -> None:
    template_data = TEMPLATE_PATH.read_bytes()
    manifest = load_manifest(FOUNDATION_MANIFEST)

    assert len(manifest.templates) == 1
    template = manifest.templates[0]
    assert template.id == TEMPLATE_ID
    assert template.name == "Map and Gather"
    assert template.tags == ("map", "gather", "subgraph")
    assert template.assets == ()
    assert template.data == template_data
    assert template.digest == f"sha256:{TEMPLATE_SHA256}"
    assert hashlib.sha256(template_data).hexdigest() == TEMPLATE_SHA256

    assert manifest.docs is not None
    assert manifest.docs.validation_problems == ()
    guide = next(page for page in manifest.docs.pages if page.kind == "guide")
    assert (guide.id, guide.locale, guide.title) == (TEMPLATE_ID, "en", "Map and Gather")
    assert guide.data in GUIDE_PATH.read_bytes()
    assert guide.digest == "sha256:" + hashlib.sha256(guide.data).hexdigest()
    assert b'template = "map-and-gather"' in guide.data

    report = diagnose(FOUNDATION_MANIFEST)
    assert report.ok

    spec = default_pack_spec("dinkster-nodes-foundation")
    assert spec.packs is not None
    info = spec.packs["dinkster-nodes-foundation"]
    assert info.templates[0].data == template_data
    assert info.docs is not None
    assert any(page.kind == "guide" and page.id == TEMPLATE_ID for page in info.docs.pages)


def test_map_and_gather_template_routes_execute_and_replay(tmp_path: Path) -> None:
    template_data = TEMPLATE_PATH.read_bytes()
    document = cast(dict[str, object], json.loads(template_data))
    graph = map_and_gather_graph(document)
    source_document = digest_bytes(template_data)

    async def scenario() -> None:
        composition = await compose_serving()
        history = HistoryStore(tmp_path / "history.sqlite")
        client: TestClient | None = None
        try:
            direct = await composition.make_engine(lambda _event: None).run(graph, ["region"])
            output = list_children(direct.outputs["region"]["sum"])
            assert output is not None
            assert [item.resolve() for item in output] == [12, 14, 18]

            app = create_app(
                composition.make_engine,
                composition.schemas,
                packs=composition.packs,
                node_packs=composition.node_packs,
                choices=composition.choices,
                lazy_choices=composition.lazy_choices,
                history=history,
            )
            client = TestClient(TestServer(app))
            await client.start_server()

            templates = await (await client.get("/api/templates")).json()
            descriptor = next(
                row
                for row in templates["templates"]
                if row["pack"] == "dinkster-nodes-foundation" and row["id"] == TEMPLATE_ID
            )
            assert descriptor["digest"] == f"sha256:{TEMPLATE_SHA256}"
            template_response = await client.get(
                f"/api/packs/dinkster-nodes-foundation/templates/{TEMPLATE_ID}"
            )
            assert template_response.status == 200
            assert await template_response.read() == template_data

            docs = await (await client.get("/api/docs?kind=guide&id=map-and-gather")).json()
            assert len(docs["docs"]) == 1
            guide_descriptor = docs["docs"][0]
            guide_locale = guide_descriptor["locales"]["en"]
            guide_response = await client.get(
                "/api/packs/dinkster-nodes-foundation/docs/pages/" + guide_locale["digest"]
            )
            assert guide_response.status == 200
            guide_data = await guide_response.read()
            assert guide_data in GUIDE_PATH.read_bytes()
            assert hashlib.sha256(guide_data).hexdigest() == guide_locale["digest"].removeprefix(
                "sha256:"
            )

            response = await client.post(
                "/api/jobs",
                json={
                    "clientId": "guide",
                    "jobId": TEMPLATE_ID,
                    "graph": graph_to_wire(graph),
                    "targets": ["region"],
                    "sourceDocument": source_document,
                },
            )
            assert response.status == 202, await response.text()
            accepted = await response.json()
            async with asyncio.timeout(5):
                while True:
                    status = await (
                        await client.get(f"/api/jobs/by-ref/{accepted['jobRef']}")
                    ).json()
                    if status["state"] == "completed":
                        break
                    assert status["state"] not in ("failed", "cancelled")
                    await asyncio.sleep(0.01)
            assert status["sourceDocument"] == source_document
            assert status["outputs"]["region"]["sum"]["length"] == 3

            replay = await (
                await client.get(f"/api/jobs/by-ref/{accepted['jobRef']}/events")
            ).json()
            assert [event["seq"] for event in replay["events"]] == list(
                range(1, replay["latestSeq"] + 1)
            )
            assert replay["events"][-1]["type"] == "job_state"
            assert replay["events"][-1]["state"] == "completed"

            async with asyncio.timeout(5):
                while True:
                    records = await (
                        await client.get("/api/history", params={"scope": "local"})
                    ).json()
                    if records["records"]:
                        break
                    await asyncio.sleep(0.01)
            record = records["records"][0]
            assert record["jobId"] == TEMPLATE_ID
            assert record["sourceDocument"] == source_document
            assert record["state"] == "completed"
            assert record["nodeReceipts"]
            assert all(receipt["executionArm"] == "native" for receipt in record["nodeReceipts"])
        finally:
            if client is not None:
                await client.close()
            history.close()
            await composition.close()

    asyncio.run(scenario())
