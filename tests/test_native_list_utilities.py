"""Public list-node ordering, indexing, and schema contracts."""

from __future__ import annotations

from dinkster_api.v1 import (
    TypeRegistry,
)
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link, TypedLiteral, validate
from dinkster_schema import build_node_types, build_schemas
from dinkster_utilities.list_utilities import CreateList, GetItemFromList
from dinkster_utilities.primitive import PrimitiveInt
from dinkster_values import register_core_types
from dinkster_workers import InProcessWorker

from tests.async_scenarios import run_scenario


def test_native_list_nodes_execute_scalar_and_list_envelopes_without_wrappers() -> None:
    async def run() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        nodes = (CreateList, GetItemFromList, PrimitiveInt)
        schemas = build_schemas(nodes)
        engine = Engine(
            schemas=schemas,
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
        )
        graph = Graph(
            nodes={
                "items": GraphNode(
                    "dinkster.utilities.create_list",
                    {
                        "inputs.input2": TypedLiteral("STRING", "first"),
                        "inputs.input0": TypedLiteral("list<INT>", [7, 13]),
                        "inputs.input1": TypedLiteral("list<STRING>", []),
                    },
                ),
                "selected": GraphNode(
                    "dinkster.utilities.get_item_from_list",
                    {
                        "list": Link("items", "list"),
                        "index": TypedLiteral("list<INT>", [-1, 0]),
                    },
                ),
                "indices": GraphNode(
                    "dinkster.utilities.create_list",
                    {"inputs.input0": TypedLiteral("list<INT>", [-2, 0])},
                ),
                "linked": GraphNode(
                    "dinkster.utilities.get_item_from_list",
                    {"list": Link("items", "list"), "index": Link("indices", "list")},
                ),
                "scalar": GraphNode(
                    "dinkster.utilities.get_item_from_list",
                    {
                        "list": TypedLiteral("STRING", "singleton"),
                        "index": 0,
                    },
                ),
            }
        )
        result = await engine.run(graph, ["items", "selected", "linked", "scalar"])
        assert result.outputs["items"]["list"].resolve() == ["first", 7, 13]
        assert result.outputs["selected"]["item"].type_id == "INT"
        assert result.outputs["selected"]["item"].resolve() == 13
        assert result.outputs["linked"]["item"].resolve() == 7
        assert result.outputs["scalar"]["item"].resolve() == "singleton"
        rejected = Graph(
            nodes={
                "plain": GraphNode(
                    "dinkster.utilities.primitive_int", {"value": TypedLiteral("list<INT>", [2])}
                ),
            }
        )
        assert any(d.severity == "error" for d in validate(rejected, schemas, ["plain"]))

    run_scenario(run())
