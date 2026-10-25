"""Native logic preserves truthiness, short circuit, and lazy branch demand."""

from __future__ import annotations

import asyncio

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.logic import ComfyAndNode, ComfyNotNode, ComfyOrNode
from dinkster_nodes_comfy.primitive import PrimitiveInt
from dinkster_nodes_comfy.switch import ComfySwitchNode
from dinkster_schema import build_node_types, build_schemas
from dinkster_values import TypeRegistry, is_absent, register_core_types
from dinkster_workers import InProcessWorker


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, True), (0, True), ([], True), ({}, True), ("False", False), (-3, False), ([0], False)],
)
def test_not_uses_truthiness(value: object, expected: bool) -> None:
    assert ComfyNotNode.execute(value=value) == {"BOOLEAN": expected}


@pytest.mark.parametrize(
    ("values", "conjunction", "disjunction"),
    [
        ({}, True, False),
        ({"value0": "False", "value1": -3}, True, True),
        ({"value0": [0], "value1": []}, False, True),
        ({"value0": {}, "value1": 0}, False, False),
        ({"value0": False, "value1": True, "value2": False}, False, True),
    ],
)
def test_mapping_values_determine_truth(
    values: dict[str, object], conjunction: bool, disjunction: bool
) -> None:
    assert ComfyAndNode.execute(values=values) == {"BOOLEAN": conjunction}
    assert ComfyOrNode.execute(values=values) == {"BOOLEAN": disjunction}


class InvalidTruth:
    def __bool__(self) -> bool:
        raise ValueError("truth requested")


def test_mapping_iteration_short_circuits_in_insertion_order() -> None:
    invalid = InvalidTruth()
    assert ComfyAndNode.execute(values={"value9": 0, "value0": invalid}) == {"BOOLEAN": False}
    assert ComfyOrNode.execute(values={"value9": [0], "value0": invalid}) == {"BOOLEAN": True}
    with pytest.raises(ValueError, match="truth requested"):
        ComfyAndNode.execute(values={"value9": invalid, "value0": 0})
    with pytest.raises(ValueError, match="truth requested"):
        ComfyOrNode.execute(values={"value9": invalid, "value0": [0]})
    with pytest.raises(ValueError, match="truth requested"):
        ComfyNotNode.execute(value=invalid)


@pytest.mark.parametrize("switch", [False, True])
@pytest.mark.parametrize("connected", [False, True])
@pytest.mark.parametrize("cache_enabled", [False, True])
def test_switch_only_demands_the_selected_connected_arm(
    switch: bool, connected: bool, cache_enabled: bool
) -> None:
    async def scenario() -> None:
        nodes = [ComfySwitchNode, PrimitiveInt]
        registry = TypeRegistry()
        register_core_types(registry)
        engine = Engine(
            schemas=build_schemas(nodes),
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
        )
        selected = "on_true" if switch else "on_false"
        unselected = "on_false" if switch else "on_true"
        inputs: dict[str, object] = {"switch": switch, unselected: Link("unused", "INT")}
        if connected:
            inputs[selected] = Link("selected", "INT")
        graph = Graph(
            {
                "unused": GraphNode("PrimitiveInt", {"value": 29}),
                "selected": GraphNode("PrimitiveInt", {"value": -17}),
                "switch": GraphNode("ComfySwitchNode", inputs),
            }
        )
        for _ in range(2):
            result = await engine.run(graph, ["switch"], cache_enabled=cache_enabled)
            value = result.outputs["switch"]["output"]
            if connected:
                assert value.resolve() == -17
            else:
                assert is_absent(value)
            assert "unused" not in result.executed
            assert "selected" not in result.executed or connected

    asyncio.run(scenario())
