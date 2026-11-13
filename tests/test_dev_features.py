from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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
