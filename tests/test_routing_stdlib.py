"""Stable routing member IDs in graph wire values."""

from __future__ import annotations

import pytest
from dinkster_graph import Graph, GraphNode, Link, graph_from_wire, graph_to_wire
from dinkster_values import TypeRegistry, register_core_types


def test_named_route_graph_wire_persists_only_stable_member_ids() -> None:
    graph = Graph(
        {
            "named": GraphNode(
                "dinkster.route.switch_by_name",
                {
                    "choice": "stable_right",
                    "values.stable_left": Link("left", "value"),
                    "values.stable_right": Link("right", "value"),
                },
            ),
        }
    )
    wire = graph_to_wire(graph)
    assert wire == {
        "nodes": {
            "named": {
                "nodeType": "dinkster.route.switch_by_name",
                "inputs": {
                    "choice": "stable_right",
                    "values.stable_left": {"$link": {"node": "left", "output": "value"}},
                    "values.stable_right": {"$link": {"node": "right", "output": "value"}},
                },
            },
        },
    }
    assert graph_from_wire(wire) == graph

    registry = TypeRegistry()
    register_core_types(registry)
    with pytest.raises(TypeError, match="COMBO expects a JSON scalar"):
        registry.wrap(
            "COMBO",
            {"value": "stable_right", "label": "Right branch"},
        )
