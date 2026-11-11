from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from aiohttp.test_utils import TestClient, TestServer
from dinkster_graph import Graph, GraphNode, Link
from dinkster_server import RuntimeSettings, create_app

from dinkster.compose import PackSpec, compose_serving
from tests.async_scenarios import run_scenario

DEV_PACK = Path(__file__).parents[1] / "packages/dinkster-nodes-dev/dinkster-pack.toml"


def test_dev_features_catalog_and_content() -> None:
    async def scenario() -> None:
        composition = await compose_serving(
            [PackSpec(DEV_PACK, trust_reserved=True)], include_default_packs=False
        )
        try:
            values = {
                "memory-budgets": {},
                "memory-headroom": 256 * 1024**2,
                "aimdo-policy": "auto",
                "dtype-policy": {"diffusion": "auto", "textEncoder": "auto", "vae": "auto"},
                "fp8-matmul": False,
                "worker-comfy-args": (),
                "jobs": {"maxRunningJobs": 1},
                "logging": {"level": "info", "overrides": {}},
            }
            app = create_app(
                composition.make_engine,
                composition.schemas,
                packs=composition.packs,
                node_packs=composition.node_packs,
                choices=composition.choices,
                lazy_choices=composition.lazy_choices,
                settings=RuntimeSettings(
                    values,
                    {key: "default" for key in values},
                    features={"templates": {"enabled": True}},
                ),
            )
            async with TestClient(TestServer(app)) as client:
                catalog = await (await client.get("/api/nodes")).json()
                nodes = catalog["nodes"]
                slot = nodes["dev.features.slot"]
                assert any(item["role"] == "dynamicSlot" for item in slot["interface"])
                count = nodes["dev.features.output_count"]
                family = next(item for item in count["interface"] if item["role"] == "outputFamily")
                assert family["count"] == {"input": "count", "suffix": "index"}
                assert nodes["dev.features.conditional"]["widgetGroups"][0]["members"] == ["note"]
                widgets = nodes["dev.features.widget_kinds"]
                kinds = {
                    item["id"]: item["widget"]["type"]
                    for item in widgets["interface"]
                    if "widget" in item
                }
                assert kinds == {
                    "selections": "MULTI_COMBO",
                    "markdown": "MARKDOWN",
                    "colors": "COLORS",
                    "button": "BUTTON",
                    "items": "STRING",
                }
                representations = nodes["dev.features.representations"]["interface"][0]["widget"]
                assert representations["type"] == "REPRESENTATIONS"
                assert representations["userSwitchable"] is True
                assert any(
                    item["role"] == "outputDescriptors"
                    for item in nodes["dev.features.output_descriptors"]["interface"]
                )
                mirror = nodes["dev.features.mirror"]["mirror"]
                assert mirror["kind"] == "glsl"
                assert "uniform sampler2D u_image" in mirror["source"]
                pack = catalog["packs"]["dinkster-nodes-dev"]
                assert pack["settings"] is True
                assert "zh" in pack["locales"]
                for kind in ("blueprints", "templates"):
                    response = await client.get(
                        f"/api/packs/dinkster-nodes-dev/{kind}/node-features"
                    )
                    assert response.status == 200
                    document = await response.json()
                    assert (
                        document["graphs"]["root"]["nodes"]["mirror"]["type"]
                        == "dev.features.mirror"
                    )
                    examples = document["graphs"]["root"]["nodes"]
                    assert json.loads(examples["descriptors"]["values"]["entries"]) == {
                        "entries": [{"id": "answer", "name": "Answer", "type": "integer"}]
                    }
                    assert all(
                        nodes[example["type"]]["outputNode"]
                        for key, example in examples.items()
                        if key != "source"
                    )
                locale = await client.get(
                    f"/api/packs/dinkster-nodes-dev/locales/{pack['locales']['zh']}"
                )
                assert locale.status == 200
                translated = await locale.json()
                assert (
                    translated["nodes"]["dev.features.widget_kinds"]["displayName"]
                    != "Feature: Widget Kinds"
                )
                info = composition.packs["dinkster-nodes-dev"]
                assert info.docs is not None
                assert {(page.kind, page.id) for page in info.docs.pages} == {
                    ("node", "dev.features.widget_kinds"),
                    ("guide", "node-features"),
                }
        finally:
            await composition.close()

    run_scenario(scenario())


def test_dev_features_execute_through_pack_worker() -> None:
    async def scenario() -> None:
        composition = await compose_serving(
            [PackSpec(DEV_PACK, trust_reserved=True)], include_default_packs=False
        )
        try:
            entries = json.dumps(
                {
                    "entries": [
                        {"id": "caption", "name": "A caption", "type": "text"},
                        {"id": "number", "name": "Row two", "type": "integer"},
                    ]
                }
            )
            nodes = {
                "text_slot": GraphNode(
                    "dev.features.slot", {"source": "hello"}, slot_variants={"source": "text"}
                ),
                "int_slot": GraphNode(
                    "dev.features.slot", {"source": -7}, slot_variants={"source": "integer"}
                ),
                "count": GraphNode(
                    "dev.features.output_count",
                    {"count": 3, "start": -2},
                    output_members={"numbers": ("0", "1", "2")},
                ),
                "conditional": GraphNode(
                    "dev.features.conditional", {"details": False, "note": "retained"}
                ),
                "widgets": GraphNode(
                    "dev.features.widget_kinds",
                    {
                        "selections": ["blue", "red", "blue"],
                        "markdown": "**Changed**",
                        "colors": ["#112233"],
                        "items": ["z", "a"],
                        "button": "",
                    },
                ),
                "representations": GraphNode(
                    "dev.features.representations", {"text": "line 1\nline 2"}
                ),
                "descriptors": GraphNode("dev.features.output_descriptors", {"entries": entries}),
                "source": GraphNode("dev.features.image_source"),
                "mirror": GraphNode(
                    "dev.features.mirror", {"image": Link("source", "image"), "gain": 1.5}
                ),
                "alpha": GraphNode(
                    "dev.features.mirror",
                    {
                        "image": [[[[0.8, 0.2, 0.4, 0.3]]]],
                        "gain": 2.0,
                    },
                ),
            }
            result = await composition.make_engine(lambda event: None).run(
                Graph(nodes=nodes), list(nodes)
            )
            outputs = result.outputs
            assert outputs["text_slot"]["text"].resolve() == "text: hello"
            assert outputs["int_slot"]["text"].resolve() == "integer: -7"
            assert {key: value.resolve() for key, value in outputs["count"].items()} == {
                "numbers.0": -2,
                "numbers.1": -1,
                "numbers.2": 0,
            }
            assert outputs["conditional"]["text"].resolve() == "details=False; note=retained"
            widget_text = outputs["widgets"]["text"].resolve()
            assert isinstance(widget_text, str)
            assert json.loads(widget_text) == {
                "selections": ["blue", "red", "blue"],
                "markdown": "**Changed**",
                "colors": ["#112233"],
                "items": ["z", "a"],
                "button": "",
            }
            assert outputs["representations"]["text"].resolve() == "line 1\nline 2"
            assert {key: value.resolve() for key, value in outputs["descriptors"].items()} == {
                "caption": "A caption",
                "number": 2,
            }
            image = outputs["mirror"]["image"].resolve()
            assert isinstance(image, np.ndarray)
            np.testing.assert_allclose(image[0, 0, 0], [0.0, 0.0, 0.375])
            np.testing.assert_allclose(image[0, -1, -1], [1.0, 1.0, 0.375])
            alpha = outputs["alpha"]["image"].resolve()
            assert isinstance(alpha, np.ndarray)
            np.testing.assert_allclose(alpha, [[[[1.0, 0.4, 0.8, 0.3]]]])
        finally:
            await composition.close()

    run_scenario(scenario())
