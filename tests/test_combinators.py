"""Core list combinators + runtime type-variable solving (DESIGN 3.13).

Combinators are dumb nodes producing list values - zero engine involvement.
Their generic element types are template variables solved per invocation at
the worker boundary from input envelope type ids (dinkster_schema.solve).
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, ExecutionError, RunResult
from dinkster_graph import (
    PORTS_NODE_ID,
    Graph,
    GraphNode,
    Link,
    RegionNode,
    RegionOutput,
    graph_from_wire,
    graph_to_wire,
)
from dinkster_schema import (
    InputSpec,
    TypeExpr,
    TypeSolveError,
    bind_type_variables,
    build_node_types,
    build_schemas,
    resolved_type_id,
)
from dinkster_values import TypeRegistry, list_children, register_core_types
from dinkster_workers import InProcessWorker
from scaffold_nodes import SCAFFOLD_NODES, register_scaffold_types

INT = TypeExpr.concrete("core.int")
T = TypeExpr.variable("T")


def make_engine() -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    register_scaffold_types(registry)
    return Engine(
        schemas=build_schemas(SCAFFOLD_NODES),
        registry=registry,
        worker=InProcessWorker(build_node_types(SCAFFOLD_NODES), registry),
        cache=MemoryLRUCache(),
    )


def run(graph: Graph, targets: list[str]) -> RunResult:
    engine = make_engine()
    return asyncio.run(engine.run(graph, targets))


def resolved_list(result: RunResult, node_id: str, output_id: str) -> list[object]:
    value = result.outputs[node_id][output_id]
    children = list_children(value)
    assert children is not None, f"{value.type_id} is not a list"
    return [c.resolve() for c in children]


def int_source(value: int) -> GraphNode:
    return GraphNode("std.math.add_ints", {"a": value, "b": 0})


# -- solver unit behavior -----------------------------------------------------


def test_bind_flat_variable() -> None:
    specs = (InputSpec("a", T), InputSpec("b", T))
    bindings = bind_type_variables(specs, {"a": "core.int", "b": "core.int"})
    assert bindings == {"T": "core.int"}


def test_bind_through_list_constructor() -> None:
    specs = (InputSpec("list", TypeExpr.list_of(T)),)
    bindings = bind_type_variables(specs, {"list": "list<list<core.int>>"})
    assert bindings == {"T": "list<core.int>"}


def test_bind_conflict_is_loud() -> None:
    specs = (InputSpec("a", T), InputSpec("b", T))
    with pytest.raises(TypeSolveError, match="already bound"):
        bind_type_variables(specs, {"a": "core.int", "b": "core.string"})


def test_bind_respects_allowlist() -> None:
    constrained = TypeExpr.variable("N", allowed=("core.int", "core.float"))
    with pytest.raises(TypeSolveError, match="allowlist"):
        bind_type_variables((InputSpec("a", constrained),), {"a": "core.string"})


def test_bind_allowlist_requires_exact_atom() -> None:
    constrained = TypeExpr.variable("T", allowed=("dinkster.image", "dinkster.mask"))
    assert bind_type_variables((InputSpec("a", constrained),), {"a": "dinkster.image"}) == {
        "T": "dinkster.image"
    }
    with pytest.raises(TypeSolveError, match="allowlist"):
        bind_type_variables((InputSpec("a", constrained),), {"a": "comfy.IMAGE"})


def test_bind_agreement_requires_exact_atom() -> None:
    specs = (InputSpec("a", T), InputSpec("b", T))
    with pytest.raises(TypeSolveError, match="already bound"):
        bind_type_variables(specs, {"a": "dinkster.image", "b": "comfy.IMAGE"})


def test_bind_does_not_conflate_image_and_mask() -> None:
    constrained = TypeExpr.variable("N", allowed=("dinkster.mask",))
    with pytest.raises(TypeSolveError, match="allowlist"):
        bind_type_variables((InputSpec("a", constrained),), {"a": "dinkster.image"})
    specs = (InputSpec("a", T), InputSpec("b", T))
    with pytest.raises(TypeSolveError, match="already bound"):
        bind_type_variables(specs, {"a": "dinkster.image", "b": "dinkster.mask"})


def test_bind_scalar_into_list_expression_is_loud() -> None:
    specs = (InputSpec("list", TypeExpr.list_of(T)),)
    with pytest.raises(TypeSolveError, match="expected a list"):
        bind_type_variables(specs, {"list": "core.int"})


def test_bind_ignores_unfed_and_nonvariable_inputs() -> None:
    specs = (InputSpec("a", T), InputSpec("b", INT))
    assert bind_type_variables(specs, {"b": "core.int"}) == {}


def test_resolved_type_id_substitutes_recursively() -> None:
    bindings = {"T": "core.int"}
    assert resolved_type_id(T, bindings) == "core.int"
    assert resolved_type_id(TypeExpr.list_of(T), bindings) == "list<core.int>"
    assert (
        resolved_type_id(TypeExpr.list_of(TypeExpr.list_of(T)), bindings) == "list<list<core.int>>"
    )
    assert resolved_type_id(TypeExpr.variable("U"), bindings) is None
    assert resolved_type_id(TypeExpr.wildcard(), bindings) is None


# -- generic combinators through the engine -----------------------------------


def test_make_list_collects_in_document_order() -> None:
    graph = Graph(
        nodes={
            "x": int_source(5),
            "y": int_source(7),
            "m": GraphNode(
                "std.list.make",
                {"items.first": Link("x", "sum"), "items.second": Link("y", "sum")},
            ),
        }
    )
    result = run(graph, ["m"])
    assert result.outputs["m"]["list"].type_id == "list<core.int>"
    assert resolved_list(result, "m", "list") == [5, 7]


def test_make_list_element_type_follows_inputs() -> None:
    graph = Graph(
        nodes={
            "s": GraphNode("std.string.concat", {"a": "a", "b": "b"}),
            "m": GraphNode("std.list.make", {"items.only": Link("s", "text")}),
        }
    )
    result = run(graph, ["m"])
    assert result.outputs["m"]["list"].type_id == "list<core.string>"
    assert resolved_list(result, "m", "list") == ["a b"]


def test_nested_lists_end_to_end() -> None:
    """list<list<core.int>> flows through make/element/length unchanged:
    the constructor nests arbitrarily and the solver binds T to a list type."""
    graph = Graph(
        nodes={
            "a": int_source(1),
            "b": int_source(2),
            "c": int_source(3),
            "inner1": GraphNode(
                "std.list.make",
                {"items.a": Link("a", "sum"), "items.b": Link("b", "sum")},
            ),
            "inner2": GraphNode("std.list.make", {"items.c": Link("c", "sum")}),
            "outer": GraphNode(
                "std.list.make",
                {"items.x": Link("inner1", "list"), "items.y": Link("inner2", "list")},
            ),
            "pick": GraphNode("std.list.element", {"list": Link("outer", "list"), "index": 0}),
            "n": GraphNode("std.list.length", {"list": Link("pick", "item")}),
        }
    )
    result = run(graph, ["outer", "pick", "n"])
    outer = result.outputs["outer"]["list"]
    assert outer.type_id == "list<list<core.int>>"
    assert outer.meta.get("length") == 2  # outer length only
    inner = list_children(outer)
    assert inner is not None and inner[0].meta.get("length") == 2  # depth via children
    assert result.outputs["pick"]["item"].type_id == "list<core.int>"
    assert resolved_list(result, "pick", "item") == [1, 2]
    assert result.outputs["n"]["length"].resolve() == 2


def test_make_list_mixed_types_is_contract_error() -> None:
    graph = Graph(
        nodes={
            "x": int_source(1),
            "s": GraphNode("std.string.concat", {"a": "a", "b": "b"}),
            "m": GraphNode(
                "std.list.make",
                {"items.i": Link("x", "sum"), "items.s": Link("s", "text")},
            ),
        }
    )
    with pytest.raises(ExecutionError, match="already bound"):
        run(graph, ["m"])


def test_append_to_list_preserves_document_order_and_serialization() -> None:
    graph = Graph(
        nodes={
            "a": int_source(1),
            "b": int_source(2),
            "c": int_source(3),
            "d": int_source(4),
            "base": GraphNode(
                "std.list.make",
                {"items.a": Link("a", "sum"), "items.b": Link("b", "sum")},
            ),
            "append": GraphNode(
                "std.list.append",
                {
                    "list": Link("base", "list"),
                    "items.first": Link("c", "sum"),
                    "items.second": Link("d", "sum"),
                },
            ),
        }
    )
    restored = graph_from_wire(graph_to_wire(graph))
    assert restored == graph

    result = run(restored, ["append"])
    assert result.outputs["append"]["list"].type_id == "list<core.int>"
    assert resolved_list(result, "append", "list") == [1, 2, 3, 4]


def test_append_to_list_keeps_a_list_item_nested() -> None:
    graph = Graph(
        nodes={
            "a": int_source(1),
            "b": int_source(2),
            "first": GraphNode("std.list.make", {"items.a": Link("a", "sum")}),
            "second": GraphNode("std.list.make", {"items.b": Link("b", "sum")}),
            "base": GraphNode("std.list.make", {"items.first": Link("first", "list")}),
            "append": GraphNode(
                "std.list.append",
                {"list": Link("base", "list"), "items.second": Link("second", "list")},
            ),
        }
    )
    result = run(graph, ["append"])
    assert result.outputs["append"]["list"].type_id == "list<list<core.int>>"
    assert resolved_list(result, "append", "list") == [[1], [2]]


def test_append_to_list_mixed_types_is_contract_error() -> None:
    graph = Graph(
        nodes={
            "x": int_source(1),
            "base": GraphNode("std.list.make", {"items.x": Link("x", "sum")}),
            "s": GraphNode("std.string.concat", {"a": "a", "b": "b"}),
            "append": GraphNode(
                "std.list.append",
                {"list": Link("base", "list"), "items.s": Link("s", "text")},
            ),
        }
    )
    with pytest.raises(ExecutionError, match="already bound"):
        run(graph, ["append"])


def test_length_element_reverse_concat_repeat() -> None:
    graph = Graph(
        nodes={
            "x": int_source(5),
            "y": int_source(6),
            "m": GraphNode(
                "std.list.make",
                {"items.a": Link("x", "sum"), "items.b": Link("y", "sum")},
            ),
            "len": GraphNode("std.list.length", {"list": Link("m", "list")}),
            "first": GraphNode("std.list.element", {"list": Link("m", "list")}),
            "last": GraphNode("std.list.element", {"list": Link("m", "list"), "index": -1}),
            "rev": GraphNode("std.list.reverse", {"list": Link("m", "list")}),
            "cat": GraphNode(
                "std.list.concat",
                {"lists.one": Link("m", "list"), "lists.two": Link("rev", "list")},
            ),
            "rep": GraphNode("std.list.repeat", {"item": Link("x", "sum"), "count": 3}),
        }
    )
    result = run(graph, ["len", "first", "last", "rev", "cat", "rep"])
    assert result.outputs["len"]["length"].resolve() == 2
    assert result.outputs["first"]["item"].type_id == "core.int"
    assert result.outputs["first"]["item"].resolve() == 5
    assert result.outputs["last"]["item"].resolve() == 6
    assert resolved_list(result, "rev", "list") == [6, 5]
    assert resolved_list(result, "cat", "list") == [5, 6, 6, 5]
    assert result.outputs["rep"]["list"].type_id == "list<core.int>"
    assert resolved_list(result, "rep", "list") == [5, 5, 5]


def test_list_slice_uses_python_bounds_and_direction() -> None:
    graph = Graph(
        nodes={
            "source": GraphNode("std.list.range", {"start": -2, "stop": 8, "step": 2}),
            "clamped": GraphNode(
                "std.list.slice",
                {"list": Link("source", "list"), "start": -99, "stop": 99, "step": 2},
            ),
            "reversed": GraphNode(
                "std.list.slice",
                {"list": Link("source", "list"), "step": -1},
            ),
            "first": GraphNode("std.string.concat", {"a": "alpha", "b": "", "separator": ""}),
            "second": GraphNode("std.string.concat", {"a": "beta", "b": "", "separator": ""}),
            "words": GraphNode(
                "std.list.make",
                {"items.first": Link("first", "text"), "items.second": Link("second", "text")},
            ),
            "tail": GraphNode(
                "std.list.slice",
                {"list": Link("words", "list"), "start": 1, "stop": 99},
            ),
        }
    )

    result = run(graph, ["clamped", "reversed", "tail"])
    assert result.outputs["clamped"]["list"].type_id == "list<core.int>"
    assert resolved_list(result, "clamped", "list") == [-2, 2, 6]
    assert resolved_list(result, "reversed", "list") == [6, 4, 2, 0, -2]
    assert result.outputs["tail"]["list"].type_id == "list<core.string>"
    assert resolved_list(result, "tail", "list") == ["beta"]


def test_integer_range_uses_python_ordering_and_empty_behavior() -> None:
    graph = Graph(
        nodes={
            "ascending": GraphNode("std.list.range", {"stop": 5}),
            "descending": GraphNode("std.list.range", {"start": 5, "stop": -2, "step": -2}),
            "empty": GraphNode("std.list.range", {"start": 5, "stop": -1, "step": 2}),
        }
    )

    result = run(graph, ["ascending", "descending", "empty"])
    assert resolved_list(result, "ascending", "list") == [0, 1, 2, 3, 4]
    assert resolved_list(result, "descending", "list") == [5, 3, 1, -1]
    assert result.outputs["empty"]["list"].type_id == "list<core.int>"
    assert resolved_list(result, "empty", "list") == []


@pytest.mark.parametrize(
    ("node_type", "inputs"),
    [
        ("std.list.range", {"stop": 4, "step": 0}),
        ("std.list.slice", {"step": 0}),
    ],
)
def test_list_slice_and_range_refuse_zero_step(node_type: str, inputs: dict[str, object]) -> None:
    nodes = {
        "source": GraphNode("std.list.range", {"stop": 4}),
        "target": GraphNode(node_type, inputs),
    }
    if node_type == "std.list.slice":
        nodes["target"] = GraphNode(node_type, {"list": Link("source", "list"), **inputs})

    with pytest.raises(ExecutionError, match="step must not be zero"):
        run(Graph(nodes=nodes), ["target"])


def test_generic_value_nodes_accept_lists_as_whole_values() -> None:
    graph = Graph(
        nodes={
            "x": int_source(5),
            "base": GraphNode("std.list.make", {"items.x": Link("x", "sum")}),
            "condition": GraphNode("dinkster.boolean", {"value": True}),
            "repeat": GraphNode(
                "std.list.repeat",
                {"item": Link("base", "list"), "count": 2},
            ),
            "select": GraphNode(
                "dinkster.value.select",
                {
                    "condition": Link("condition", "value"),
                    "on_true": Link("base", "list"),
                    "on_false": Link("base", "list"),
                },
            ),
            "switch": GraphNode(
                "dinkster.route.switch",
                {"index": 0, "values.only": Link("base", "list")},
            ),
            "gate": GraphNode(
                "dinkster.route.gate",
                {
                    "condition": Link("condition", "value"),
                    "value": Link("base", "list"),
                },
            ),
        }
    )
    result = run(graph, ["repeat", "select", "switch", "gate"])

    assert result.outputs["repeat"]["list"].type_id == "list<list<core.int>>"
    assert resolved_list(result, "repeat", "list") == [[5], [5]]
    for node in ("select", "switch", "gate"):
        assert result.outputs[node]["value"].type_id == "list<core.int>"
        assert resolved_list(result, node, "value") == [5]


def test_element_out_of_range_is_loud() -> None:
    graph = Graph(
        nodes={
            "x": int_source(1),
            "m": GraphNode("std.list.make", {"items.a": Link("x", "sum")}),
            "e": GraphNode("std.list.element", {"list": Link("m", "list"), "index": 5}),
        }
    )
    with pytest.raises(ExecutionError, match="out of range"):
        run(graph, ["e"])


def test_repeat_zero_is_typed_empty_and_negative_is_loud() -> None:
    graph = Graph(
        nodes={
            "x": int_source(9),
            "rep": GraphNode("std.list.repeat", {"item": Link("x", "sum"), "count": 0}),
        }
    )
    result = run(graph, ["rep"])
    assert result.outputs["rep"]["list"].type_id == "list<core.int>"
    assert resolved_list(result, "rep", "list") == []

    bad = Graph(
        nodes={
            "x": int_source(9),
            "rep": GraphNode("std.list.repeat", {"item": Link("x", "sum"), "count": -1}),
        }
    )
    with pytest.raises(ExecutionError, match=">= 0"):
        run(bad, ["rep"])


def test_cross_product_output_types_follow_each_input() -> None:
    graph = Graph(
        nodes={
            "x": int_source(1),
            "y": int_source(2),
            "s": GraphNode("std.string.concat", {"a": "p", "b": "q", "separator": ""}),
            "nums": GraphNode(
                "std.list.make",
                {"items.a": Link("x", "sum"), "items.b": Link("y", "sum")},
            ),
            "strs": GraphNode("std.list.make", {"items.only": Link("s", "text")}),
            "cross": GraphNode(
                "std.list.cross_product",
                {"a": Link("nums", "list"), "b": Link("strs", "list")},
            ),
        }
    )
    result = run(graph, ["cross"])
    assert result.outputs["cross"]["a"].type_id == "list<core.int>"
    assert result.outputs["cross"]["b"].type_id == "list<core.string>"
    assert resolved_list(result, "cross", "a") == [1, 2]
    assert resolved_list(result, "cross", "b") == ["pq", "pq"]


def test_cross_product_feeds_zip_region() -> None:
    """The documented sweep composition: CrossProduct -> zip-binding map,
    last input varying fastest (the exact ordering of a region's cross
    binding, so the two spellings are interchangeable)."""
    graph = Graph(
        nodes={
            "a1": int_source(1),
            "a2": int_source(2),
            "b1": int_source(10),
            "b2": int_source(20),
            "b3": int_source(30),
            "la": GraphNode(
                "std.list.make",
                {"items.1": Link("a1", "sum"), "items.2": Link("a2", "sum")},
            ),
            "lb": GraphNode(
                "std.list.make",
                {
                    "items.1": Link("b1", "sum"),
                    "items.2": Link("b2", "sum"),
                    "items.3": Link("b3", "sum"),
                },
            ),
            "cross": GraphNode(
                "std.list.cross_product",
                {"a": Link("la", "list"), "b": Link("lb", "list")},
            ),
            "sweep": RegionNode(
                kind="map",
                body=Graph(
                    nodes={
                        "add": GraphNode(
                            "std.math.add_ints",
                            {
                                "a": Link(PORTS_NODE_ID, "a"),
                                "b": Link(PORTS_NODE_ID, "b"),
                            },
                        )
                    }
                ),
                ports={"a": INT, "b": INT},
                inputs={"a": Link("cross", "a"), "b": Link("cross", "b")},
                element_ports=("a", "b"),
                outputs={"sums": RegionOutput(Link("add", "sum"))},
            ),
        }
    )
    result = run(graph, ["sweep"])
    # 2x3=6 combos, last input varying fastest.
    assert resolved_list(result, "sweep", "sums") == [11, 21, 31, 12, 22, 32]


# -- image batch <-> list ------------------------------------------------------


def test_image_batch_list_roundtrip() -> None:
    graph = Graph(
        nodes={
            "g1": GraphNode("dev.image.gradient", {"width": 4, "height": 2}),
            "g2": GraphNode("dev.image.invert", {"image": Link("g1", "image")}),
            "m": GraphNode(
                "std.list.make",
                {"items.a": Link("g1", "image"), "items.b": Link("g2", "image")},
            ),
            "batch": GraphNode("dev.image.list_to_batch", {"images": Link("m", "list")}),
            "back": GraphNode("dev.image.batch_to_list", {"batch": Link("batch", "batch")}),
        }
    )
    result = run(graph, ["batch", "back"])
    assert result.outputs["batch"]["batch"].type_id == "dev.image"
    stacked = result.outputs["batch"]["batch"].resolve()
    assert isinstance(stacked, np.ndarray)
    assert stacked.shape == (2, 2, 4)
    out = result.outputs["back"]["images"]
    assert out.type_id == "list<dev.image>"
    children = list_children(out)
    assert children is not None and len(children) == 2
    first = children[0].resolve()
    assert isinstance(first, np.ndarray)
    assert first.shape == (2, 4)


def test_list_to_batch_shape_mismatch_and_empty_are_loud() -> None:
    mismatch = Graph(
        nodes={
            "g1": GraphNode("dev.image.gradient", {"width": 4, "height": 2}),
            "g2": GraphNode("dev.image.gradient", {"width": 3, "height": 2}),
            "m": GraphNode(
                "std.list.make",
                {"items.a": Link("g1", "image"), "items.b": Link("g2", "image")},
            ),
            "batch": GraphNode("dev.image.list_to_batch", {"images": Link("m", "list")}),
        }
    )
    with pytest.raises(ExecutionError, match="share one shape"):
        run(mismatch, ["batch"])

    empty = Graph(
        nodes={
            "g": GraphNode("dev.image.gradient", {"width": 4, "height": 2}),
            "none": GraphNode("std.list.repeat", {"item": Link("g", "image"), "count": 0}),
            "batch": GraphNode("dev.image.list_to_batch", {"images": Link("none", "list")}),
        }
    )
    with pytest.raises(ExecutionError, match="empty list"):
        run(empty, ["batch"])


# -- caching -------------------------------------------------------------------


def test_generic_node_result_is_cached_with_solved_type() -> None:
    async def scenario() -> None:
        engine = make_engine()
        graph = Graph(
            nodes={
                "x": int_source(1),
                "y": int_source(2),
                "m": GraphNode("std.list.make", {"items.a": Link("x", "sum")}),
                "append": GraphNode(
                    "std.list.append",
                    {"list": Link("m", "list"), "items.b": Link("y", "sum")},
                ),
            }
        )
        first = await engine.run(graph, ["append"])
        assert "append" in first.executed
        second = await engine.run(graph, ["append"])
        assert "append" in second.cached
        assert second.outputs["append"]["list"].type_id == "list<core.int>"
        assert resolved_list(second, "append", "list") == [1, 2]

    asyncio.run(scenario())
