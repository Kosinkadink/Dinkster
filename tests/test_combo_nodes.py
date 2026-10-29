"""Typed combo edge nodes: dinkster.string_to_combo and dinkster.combo_to_string.

What this proves: both nodes are always-composed foundation nodes and are the
explicit core.string/core.combo bridges. Execution remains string identity,
no aliases are claimed, and neither node carries a widget of its own.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_foundation import COMBO_EDGE_NODES, FOUNDATION_NODES
from dinkster_schema import build_node_types, build_schemas, schema_to_wire
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker

STRING_TYPE = {"kind": "concrete", "types": ["core.string"]}
COMBO_TYPE = {"kind": "concrete", "types": ["core.combo"]}


def make_engine() -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    return Engine(
        schemas=build_schemas(FOUNDATION_NODES),
        registry=registry,
        worker=InProcessWorker(build_node_types(FOUNDATION_NODES), registry),
        cache=MemoryLRUCache(),
    )


def test_combo_edge_nodes_are_typed_foundation_nodes() -> None:
    """Both bridges compose with the foundation pack, claim no legacy alias, carry
    no widget, and expose the inverse concrete socket types."""
    schemas = build_schemas(FOUNDATION_NODES)
    assert {node.schema().node_type for node in COMBO_EDGE_NODES} == {
        "dinkster.string_to_combo",
        "dinkster.combo_to_string",
    }

    to_combo = schemas["dinkster.string_to_combo"]
    assert to_combo.aliases == ()
    assert "text to combo" in to_combo.search_terms
    wire = schema_to_wire(to_combo)
    interface = cast("list[dict[str, Any]]", wire["interface"])
    entries = {entry["id"]: entry for entry in interface}
    assert entries["string"]["type"] == STRING_TYPE
    assert "widget" not in entries["string"]
    assert entries["choice"]["type"] == COMBO_TYPE
    assert "comboSource" not in entries["choice"]

    to_string = schemas["dinkster.combo_to_string"]
    assert to_string.aliases == ()
    assert "combo to string" in to_string.search_terms
    wire = schema_to_wire(to_string)
    interface = cast("list[dict[str, Any]]", wire["interface"])
    entries = {entry["id"]: entry for entry in interface}
    assert entries["choice"]["type"] == COMBO_TYPE
    assert "widget" not in entries["choice"]
    assert entries["text"]["type"] == STRING_TYPE
    assert "comboSource" not in entries["text"]


def test_combo_edge_nodes_execute_as_identity() -> None:
    """A value survives a string -> combo -> string round trip verbatim,
    including values that would name no choice anywhere - vocabulary
    membership is the frontend's diagnostic and what happens at execution
    is the consuming node's own behavior, never these bridges'."""

    async def scenario() -> None:
        engine = make_engine()
        graph = Graph(
            nodes={
                "s": GraphNode("dinkster.string", {"value": "not-a-choice.txt"}),
                "c": GraphNode("dinkster.string_to_combo", {"string": Link("s", "value")}),
                "t": GraphNode("dinkster.combo_to_string", {"choice": Link("c", "choice")}),
            }
        )
        result = await engine.run(graph, ["c", "t"])
        assert result.outputs["c"]["choice"].resolve() == "not-a-choice.txt"
        assert result.outputs["c"]["choice"].type_id == "core.combo"
        assert result.outputs["t"]["text"].resolve() == "not-a-choice.txt"
        assert result.outputs["t"]["text"].type_id == "core.string"

    asyncio.run(scenario())
