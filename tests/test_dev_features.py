from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from dinkster_engine.automatic import resolve_automatic
from dinkster_graph import Graph, GraphNode, Link

from dinkster.compose import PackSpec, compose_serving
from tests.async_scenarios import run_scenario

DEV_PACK = Path(__file__).parents[1] / "packages/dinkster-nodes-dev/dinkster-pack.toml"


def test_dev_features_execute_through_pack_worker() -> None:
    async def scenario() -> None:
        composition = await compose_serving(
            [PackSpec(DEV_PACK, trust_reserved=True)], include_default_packs=False
        )
        try:
            entries = json.dumps(
                {
                    "entries": [
                        {"id": "caption", "label": "A caption", "kind": "integer"},
                        {"id": "number", "label": "Row two", "kind": "integer"},
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
                    output_members={"numbers": ("1", "2", "3")},
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
                "numbers.1": -1,
                "numbers.2": 0,
                "numbers.3": 1,
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
                "entries.caption": 1,
                "entries.number": 2,
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


def test_dev_automatic_values_load_and_run_through_pack_worker() -> None:
    async def scenario() -> None:
        composition = await compose_serving(
            [PackSpec(DEV_PACK, trust_reserved=True)], include_default_packs=False
        )
        try:
            nodes = {
                "follows": GraphNode("dev.automatic.frame_rate", {"rate": 60.0, "divisor": 4}),
                "fails": GraphNode("dev.automatic.frame_rate", {"rate": 60.0, "divisor": 0}),
                "fixed": GraphNode(
                    "dev.automatic.frame_rate", {"rate": 60.0, "divisor": 4, "frame_rate": 50.0}
                ),
            }
            graph, records = await resolve_automatic(
                Graph(nodes=nodes), list(nodes), composition.schemas
            )
            assert {r.node_id: (r.value, r.worked_out) for r in records} == {
                "follows": (15.0, True),
                "fails": (24.0, False),
            }
            result = await composition.make_engine(lambda event: None).run(graph, list(nodes))
            fps = {node: result.outputs[node]["fps"].resolve() for node in nodes}
            assert fps == {"follows": 15.0, "fails": 24.0, "fixed": 50.0}
            assert {"dev.automatic.lookup_rate", "dev.automatic.lora_stack"} <= set(
                composition.schemas
            )
        finally:
            await composition.close()

    run_scenario(scenario())
