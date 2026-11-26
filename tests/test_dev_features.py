from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from dinkster_engine import EngineEvent
from dinkster_engine.automatic import resolve_automatic
from dinkster_graph import AbsentInput, Graph, GraphNode, Link
from dinkster_values import is_absent

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
                # The colors arrive as a list in stored member order: color2 first.
                "widgets": GraphNode(
                    "dev.features.widget_kinds",
                    {
                        "selections": ["blue", "red", "blue"],
                        "markdown": "**Changed**",
                        "colors.color2": "#112233",
                        "colors.color0": "#445566",
                        "items": ["z", "a"],
                        "button": "",
                    },
                ),
                # No members: the colors arrive as an empty list.
                "no_colors": GraphNode(
                    "dev.features.widget_kinds",
                    {
                        "selections": [],
                        "markdown": "",
                        "items": [],
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
                "colors": ["#112233", "#445566"],
                "items": ["z", "a"],
                "button": "",
            }
            no_colors = outputs["no_colors"]["text"].resolve()
            assert isinstance(no_colors, str)
            assert json.loads(no_colors)["colors"] == []
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
            assert {
                "dev.automatic.lookup_rate",
                "dev.automatic.video_rate",
                "dev.automatic.lora_stack",
            } <= set(composition.schemas)
        finally:
            await composition.close()

    run_scenario(scenario())


def test_dev_task_nodes_execute_through_pack_worker() -> None:
    async def scenario() -> None:
        composition = await compose_serving(
            [PackSpec(DEV_PACK, trust_reserved=True)], include_default_packs=False
        )
        try:
            nodes = {
                # Stored member order is the run's order: 3 before 1.
                "layers": GraphNode(
                    "dev.features.layers",
                    {
                        "title": "Stack",
                        "layers.3.name": "top",
                        "layers.3.strength": 0.5,
                        "layers.1.name": "base",
                        "layers.1.strength": 1.25,
                    },
                ),
                "fixed": GraphNode(
                    "dev.features.option_fields",
                    {"mode.amount": 0.25, "source": 2.0, "source.scale": 3.0},
                    slot_variants={"mode": "fixed", "source": "number"},
                ),
                "none": GraphNode(
                    "dev.features.option_fields",
                    {"source": "hi"},
                    slot_variants={"mode": "none", "source": "text"},
                ),
                "given": GraphNode("dev.features.required_value", {"value": 0.75}),
                "nothing": GraphNode(
                    "dev.features.required_value", {"value": AbsentInput("the option is off")}
                ),
            }
            result = await composition.make_engine(lambda event: None).run(
                Graph(nodes=nodes), list(nodes)
            )
            outputs = result.outputs
            assert outputs["layers"]["text"].resolve() == "Stack: top 0.50, base 1.25"
            assert (
                outputs["fixed"]["text"].resolve()
                == "mode=fixed; amount=0.25; source=number:2.0; scale=3.00"
            )
            assert outputs["none"]["text"].resolve() == "mode=none; source=text:hi"
            assert outputs["given"]["text"].resolve() == "value=0.75"
            assert "nothing" in result.skipped
            assert is_absent(outputs["nothing"]["text"])
        finally:
            await composition.close()

    run_scenario(scenario())


def test_each_curve_gets_the_histogram_at_its_own_input() -> None:
    async def scenario() -> None:
        composition = await compose_serving(
            [PackSpec(DEV_PACK, trust_reserved=True)], include_default_packs=False
        )
        try:
            nodes = {
                "rising": GraphNode("dev.features.histogram_shape", {"shape": "rising"}),
                "falling": GraphNode("dev.features.histogram_shape", {"shape": "falling"}),
                "curves": GraphNode(
                    "dev.features.curve_histograms",
                    {
                        "curves.1.curve": {"points": [[0, 0], [1, 1]]},
                        "curves.1.histogram": Link("rising", "histogram"),
                        "curves.2.curve": {"points": [[0, 1], [0.5, 0.2], [1, 0]]},
                        "curves.2.histogram": Link("falling", "histogram"),
                        "curves.3.curve": {"points": [[0, 0], [1, 1]]},
                    },
                ),
            }
            rising = [index * 4 for index in range(256)]
            falling = [(255 - index) * 4 for index in range(256)]
            events: list[EngineEvent] = []
            engine = composition.make_engine(events.append)
            # The second run takes the node from the cache and still sends both.
            for cached in (False, True):
                events.clear()
                result = await engine.run(Graph(nodes=nodes), ["curves"])
                assert ("curves" in result.cached) == cached
                assert result.outputs["curves"]["text"].resolve() == (
                    "1: 2 points, 256 bins; 2: 3 points, 256 bins; 3: 2 points, no histogram"
                )
                sent = sorted(
                    (
                        (event.node_id, event.detail["data"])
                        for event in events
                        if event.kind == "node_event"
                        and event.detail.get("name") == "dinkster.curve.histogram"
                    ),
                    key=lambda item: str(item[1]),
                )
                assert sent == [
                    ("curves", {"curve": "curves.1.curve", "histogram": rising}),
                    ("curves", {"curve": "curves.2.curve", "histogram": falling}),
                ]
        finally:
            await composition.close()

    run_scenario(scenario())


def test_no_dev_group_heads_a_member_with_a_bare_number() -> None:
    # A development node exercises what a real pack declares, and a real pack
    # names its members: a group numbered without a prefix needs a member_label.
    from dataclasses import fields, is_dataclass

    from dinkster_nodes_dev import PACK_NODES
    from dinkster_schema.body import Group, Numbered

    bare: list[str] = []

    def walk(value: object, node_type: str) -> None:
        if isinstance(value, Group):
            keys = value.keys
            if isinstance(keys, Numbered) and keys.prefix == "" and not value.member_label:
                bare.append(f"{node_type}: {value.id}")
        if isinstance(value, (tuple, list)):
            for item in value:  # pyright: ignore[reportUnknownVariableType]
                walk(item, node_type)  # pyright: ignore[reportUnknownArgumentType]
        elif is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                walk(getattr(value, field.name), node_type)

    for node in PACK_NODES:
        schema = node.define_schema()
        walk(schema.body, schema.node_type)
    assert bare == []
