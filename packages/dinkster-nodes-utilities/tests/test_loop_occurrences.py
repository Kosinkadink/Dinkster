"""Native loop expansion executes body occurrences through the engine."""

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link, RegionNode, RegionOutput, TypedLiteral
from dinkster_memory import GovernorReservationService, MemoryGovernor, ReservationRequest
from dinkster_schema import (
    ABSENT,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
    schema_to_wire,
)
from dinkster_utilities import NODES as UTILITY_NODES
from dinkster_utilities.loop_lists import ListElement, MakeList
from dinkster_utilities.math import ComfyMathExpression
from dinkster_utilities.primitive import PrimitiveBoolean, PrimitiveInt
from dinkster_utilities.switch import ComfySwitchNode
from dinkster_values import TypeRegistry, ValueMeta, is_absent, list_children, register_core_types
from dinkster_workers import GovernedWorker, InProcessWorker, IsolatedWorker

from tests.async_scenarios import run_scenario

LOOP_NODES = tuple(n for n in UTILITY_NODES if n.__module__ == "dinkster_utilities.loops")

NODES = (
    *LOOP_NODES,
    ComfyMathExpression,
    PrimitiveInt,
    PrimitiveBoolean,
    ComfySwitchNode,
    MakeList,
    ListElement,
)


@pytest.mark.parametrize(
    ("node_type", "expected"),
    [
        (
            "dinkster.utilities.end_loop",
            [("output_value", False), ("next_iteration_value", False), ("accumulate", True)],
        ),
        (
            "dinkster.utilities.loop_iteration",
            [
                ("iteration_index", True),
                ("is_first", True),
                ("is_last", True),
                ("list_item", False),
                ("current_iteration_value", False),
                ("reuse_cache", True),
            ],
        ),
    ],
)
def test_loop_wire_inputs_preserve_v3_declaration_order(
    node_type: str, expected: list[tuple[str, bool]]
) -> None:
    schema = build_schemas(LOOP_NODES)[node_type]
    interface = cast(list[dict[str, object]], schema_to_wire(schema)["interface"])
    inputs = [(item["id"], item["required"]) for item in interface if item["role"] == "input"]
    assert inputs == expected
    if node_type == "dinkster.utilities.end_loop":
        family = next(item for item in interface if item["role"] == "inputFamily")
        assert family["id"] == "terminations"
        assert family["required"] is False
        assert family["minMembers"] == 0
        assert family["maxMembers"] == 50
        assert family["memberPrefix"] == "termination"
        assert family["template"] == [
            {"role": "input", "id": "termination", "required": False, "type": {"kind": "wildcard"}}
        ]


def make_engine() -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    return Engine(
        schemas=build_schemas(NODES),
        registry=registry,
        worker=InProcessWorker(build_node_types(NODES), registry),
        cache=MemoryLRUCache(),
    )


@pytest.mark.parametrize("cache_enabled", [False, True])
@pytest.mark.parametrize(
    ("count", "accumulate", "expected"),
    [
        (0, False, []),
        (4, False, [10]),
        (4, True, [1, 3, 6, 10]),
    ],
)
def test_carried_value_and_empty_loop(
    cache_enabled: bool, count: int, accumulate: bool, expected: list[int]
) -> None:
    graph = Graph(
        {
            "start": GraphNode(
                "dinkster.utilities.start_loop",
                {
                    "mode.num_iterations": count,
                    "initial_iteration_value": TypedLiteral("INT", 0),
                },
                slot_variants={"mode": "simple"},
            ),
            "body": GraphNode(
                "dinkster.utilities.comfy_math_expression",
                {
                    "expression": "a + b + 1",
                    "values.a": Link("start", "current_iteration_value"),
                    "values.b": Link("start", "iteration_index"),
                },
            ),
            "end": GraphNode(
                "dinkster.utilities.end_loop",
                {
                    "output_value": Link("body", "INT"),
                    "next_iteration_value": Link("body", "INT"),
                    "accumulate": accumulate,
                },
            ),
        }
    )

    async def scenario() -> None:
        engine = make_engine()
        for _ in range(2):
            result = await asyncio.wait_for(
                engine.run(graph, ["end"], cache_enabled=cache_enabled), 10
            )
            assert result.outputs["end"]["outputs"].resolve() == expected

    run_scenario(scenario())


@pytest.mark.parametrize("mode", ["For", "List"])
def test_loop_modes(mode: str) -> None:
    options = (
        {"mode.start_iteration_index": 3, "mode.max_iteration": 10, "mode.step": 3}
        if mode == "For"
        else {"mode.list": TypedLiteral("list<INT>", [9, 2, 5])}
    )
    output = "iteration_index" if mode == "For" else "list_item"
    graph = Graph(
        {
            "start": GraphNode(
                "dinkster.utilities.start_loop", options, slot_variants={"mode": mode}
            ),
            "body": GraphNode(
                "dinkster.utilities.comfy_math_expression",
                {
                    "expression": "a * 2",
                    "values.a": Link("start", output),
                },
            ),
            "end": GraphNode(
                "dinkster.utilities.end_loop",
                {"output_value": Link("body", "INT"), "accumulate": True},
            ),
        }
    )
    result = run_scenario(asyncio.wait_for(make_engine().run(graph, ["end"]), 10))
    assert result.outputs["end"]["outputs"].resolve() == (
        [6, 12, 18] if mode == "For" else [18, 4, 10]
    )


@pytest.mark.parametrize("inner_cache", [False, True])
def test_nested_loop_keeps_each_occurrences_results(inner_cache: bool) -> None:
    graph = Graph(
        {
            "outer": GraphNode(
                "dinkster.utilities.start_loop",
                {"mode.num_iterations": 2},
                slot_variants={"mode": "simple"},
            ),
            "inner": GraphNode(
                "dinkster.utilities.start_loop",
                {
                    "mode.num_iterations": 3,
                    "parent_iteration": Link("outer", "iteration_index"),
                    "cache_iterations": inner_cache,
                },
                slot_variants={"mode": "simple"},
            ),
            "body": GraphNode(
                "dinkster.utilities.comfy_math_expression",
                {
                    "expression": "a * 10 + b",
                    "values.a": Link("outer", "iteration_index"),
                    "values.b": Link("inner", "iteration_index"),
                },
            ),
            "inner_end": GraphNode(
                "dinkster.utilities.end_loop",
                {"output_value": Link("body", "INT"), "accumulate": True},
            ),
            "outer_end": GraphNode(
                "dinkster.utilities.end_loop",
                {"output_value": Link("inner_end", "outputs"), "accumulate": True},
            ),
        }
    )

    async def scenario() -> None:
        engine = make_engine()
        for run in range(2):
            result = await asyncio.wait_for(engine.run(graph, ["outer_end"]), 10)
            assert result.outputs["outer_end"]["outputs"].resolve() == [[0, 1, 2], [10, 11, 12]]
            executed_body = [node for node in result.executed if node.endswith("_body")]
            assert len(executed_body) == (0 if run and inner_cache else 6)

    run_scenario(scenario())


@pytest.mark.parametrize("worker_kind", ["isolated", "governed"])
def test_loop_expansion_and_publication_across_worker_boundary(
    tmp_path: Path, worker_kind: str
) -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        boundary = IsolatedWorker(
            manifest,
            registry,
            use_shm=True,
            shm_threshold=1,
            extra_env={"PYTHONPATH": str(Path(__file__).resolve().parent.parent)},
        )
        worker = (
            GovernedWorker(
                boundary,
                planner=lambda _: (ReservationRequest("ram", 1),),
                reservations=GovernorReservationService(MemoryGovernor({"ram": 1})),
            )
            if worker_kind == "governed"
            else boundary
        )
        try:
            await boundary.start()
            engine = Engine(
                schemas=build_schemas(NODES),
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
            )
            graph = Graph(
                {
                    "start": GraphNode(
                        "dinkster.utilities.start_loop",
                        {"mode.num_iterations": 3},
                        slot_variants={"mode": "simple"},
                    ),
                    "body": GraphNode(
                        "dinkster.utilities.comfy_math_expression",
                        {"expression": "a + 7", "values.a": Link("start", "iteration_index")},
                    ),
                    "end": GraphNode(
                        "dinkster.utilities.end_loop",
                        {"output_value": Link("body", "INT"), "accumulate": True},
                    ),
                }
            )
            result = await asyncio.wait_for(engine.run(graph, ["end"]), 20)
            assert result.outputs["end"]["outputs"].resolve() == [7, 8, 9]
        finally:
            await boundary.close()

    run_scenario(scenario())


@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("linked", [False, True])
def test_accumulation_control_reads_its_value(accumulate: bool, linked: bool) -> None:
    graph = Graph(
        {
            "control": GraphNode("dinkster.utilities.primitive_boolean", {"value": accumulate}),
            "start": GraphNode(
                "dinkster.utilities.start_loop",
                {"mode.num_iterations": 3},
                slot_variants={"mode": "simple"},
            ),
            "end": GraphNode(
                "dinkster.utilities.end_loop",
                {
                    "output_value": Link("start", "iteration_index"),
                    "accumulate": Link("control", "BOOLEAN")
                    if linked
                    else TypedLiteral("BOOLEAN", accumulate),
                },
            ),
        }
    )
    result = run_scenario(asyncio.wait_for(make_engine().run(graph, ["end"]), 10))
    assert result.outputs["end"]["outputs"].resolve() == ([0, 1, 2] if accumulate else [2])


def test_loop_collection_preserves_child_envelopes() -> None:
    registry = TypeRegistry()
    register_core_types(registry)
    value = replace(
        registry.wrap("INT", 17),
        meta=ValueMeta({"resourceProducerArm": "producer", "provenance": "original"}),
    )

    class Source(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                "test.stamped", outputs=(OutputSpec("value", TypeExpr.concrete("INT")),)
            )

        @classmethod
        def execute(cls):
            return cls.outputs(value=value)

    nodes = (*NODES, Source)
    engine = Engine(
        schemas=build_schemas(nodes),
        registry=registry,
        worker=InProcessWorker(build_node_types(nodes), registry),
        cache=MemoryLRUCache(),
    )
    graph = Graph(
        {
            "source": GraphNode("test.stamped", {}),
            "items": GraphNode(
                "dinkster.utilities.make_list", {"items.0": Link("source", "value")}
            ),
            "start": GraphNode(
                "dinkster.utilities.start_loop",
                {"mode.list": Link("items", "list")},
                slot_variants={"mode": "List"},
            ),
            "end": GraphNode(
                "dinkster.utilities.end_loop",
                {"output_value": Link("start", "list_item"), "accumulate": True},
            ),
        }
    )

    async def scenario() -> None:
        for replay in (False, True):
            result = await asyncio.wait_for(engine.run(graph, ["end"]), 10)
            children = list_children(result.outputs["end"]["outputs"])
            assert children is not None and len(children) == 1
            assert children[0].meta == value.meta
            assert children[0].fingerprint == value.fingerprint
            if replay:
                assert "items" in result.cached

    run_scenario(scenario())


@pytest.mark.parametrize("cache_enabled", [False, True])
@pytest.mark.parametrize("absent_input", ["output_value", "terminations.0"])
def test_absent_loop_dependency_completes(cache_enabled: bool, absent_input: str) -> None:
    class Empty(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            integer = TypeExpr.concrete("INT")
            return NodeSchema(
                "test.empty",
                inputs=(InputSpec("index", integer),),
                outputs=(OutputSpec("value", integer, optional=True),),
            )

        @classmethod
        def execute(cls, index: int):
            return cls.outputs(value=ABSENT)

    nodes = (*NODES, Empty)
    registry = TypeRegistry()
    register_core_types(registry)
    engine = Engine(
        schemas=build_schemas(nodes),
        registry=registry,
        worker=InProcessWorker(build_node_types(nodes), registry),
        cache=MemoryLRUCache(),
    )
    graph = Graph(
        {
            "start": GraphNode(
                "dinkster.utilities.start_loop",
                {"mode.num_iterations": 1},
                slot_variants={"mode": "simple"},
            ),
            "empty": GraphNode("test.empty", {"index": Link("start", "iteration_index")}),
            "end": GraphNode(
                "dinkster.utilities.end_loop",
                {
                    "output_value": Link("start", "iteration_index"),
                    absent_input: Link("empty", "value"),
                },
            ),
        }
    )

    async def scenario() -> None:
        for _ in range(2):
            result = await asyncio.wait_for(
                engine.run(graph, ["end"], cache_enabled=cache_enabled), 3
            )
            output = result.outputs["end"]["outputs"]
            if absent_input == "output_value":
                assert is_absent(output)
            else:
                assert output.resolve() == [0]

    run_scenario(scenario())


def test_loop_inside_map_can_override_its_end() -> None:
    body = Graph(
        {
            "start": GraphNode(
                "dinkster.utilities.start_loop",
                {"mode.num_iterations": Link("$region", "count")},
                slot_variants={"mode": "simple"},
            ),
            "end": GraphNode(
                "dinkster.utilities.end_loop",
                {"output_value": Link("start", "iteration_index"), "accumulate": True},
            ),
        }
    )
    graph = Graph(
        {
            "region": RegionNode(
                kind="map",
                ports={"count": TypeExpr.concrete("INT")},
                inputs={"count": [2]},
                element_ports=("count",),
                body=body,
                outputs={"values": RegionOutput(Link("end", "outputs"))},
            )
        }
    )
    result = run_scenario(asyncio.wait_for(make_engine().run(graph, ["region"]), 3))
    assert result.outputs["region"]["values"].resolve() == [[0, 1]]
