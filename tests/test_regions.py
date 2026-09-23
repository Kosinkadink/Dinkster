"""Explicit graph regions (DESIGN 3.13): the single repetition primitive.

Map/fold/while as validation profiles over one engine expansion;
zip/cross/broadcast bindings; gather/compact/state/flatten outputs; per-iteration
caching through ordinary input fingerprints; whole-region absence; the wire
shape frontends submit.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Literal, cast

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EngineEvent, ExecutionError
from dinkster_graph import (
    PORTS_NODE_ID,
    REGION_INDEX_PORT_ID,
    Graph,
    GraphNode,
    GraphWireError,
    Link,
    RegionNode,
    RegionOutput,
    graph_from_wire,
    graph_to_wire,
    has_errors,
    region_interface,
    snapshot_graph,
    validate,
)
from dinkster_graph.model import BindingMode, RegionKind
from dinkster_nodes_foundation import (
    ListElement,
    RouteGate,
    RouteSwitch,
    RouteSwitchByName,
    ValueSelect,
)
from dinkster_schema import (
    AbsentOutput,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
)
from dinkster_values import (
    ABSENT_ORIGIN_META_KEY,
    ABSENT_STANDS_FOR_META_KEY,
    TypeRegistry,
    is_absent,
    list_children,
    register_core_types,
)
from dinkster_workers import InProcessWorker

INT = TypeExpr.concrete("core.int")
BOOL = TypeExpr.concrete("core.boolean")
COMBO = TypeExpr.concrete("core.combo")
STRING = TypeExpr.concrete("core.string")
GENERIC = TypeExpr.variable("T")


class AddOne(Node):
    ran: list[int] = []

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.add_one",
            inputs=(InputSpec("value", INT),),
            outputs=(OutputSpec("out", INT),),
        )

    @classmethod
    def execute(cls, value: int) -> Mapping[str, object]:
        AddOne.ran.append(value)
        return cls.outputs(out=value + 1)


class AddPair(Node):
    calls: list[tuple[int, int]] = []

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.add_pair",
            inputs=(InputSpec("a", INT), InputSpec("b", INT)),
            outputs=(OutputSpec("out", INT),),
        )

    @classmethod
    def execute(cls, a: int, b: int) -> Mapping[str, object]:
        cls.calls.append((a, b))
        return cls.outputs(out=a + b)


class Mul(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.mul",
            inputs=(InputSpec("a", INT), InputSpec("b", INT)),
            outputs=(OutputSpec("out", INT),),
        )

    @classmethod
    def execute(cls, a: int, b: int) -> Mapping[str, object]:
        return cls.outputs(out=a * b)


class IsEven(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.is_even",
            inputs=(InputSpec("value", INT),),
            outputs=(OutputSpec("result", BOOL),),
        )

    @classmethod
    def execute(cls, value: int) -> Mapping[str, object]:
        return cls.outputs(result=value % 2 == 0)


class BelowLimit(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.below_limit",
            inputs=(InputSpec("value", INT), InputSpec("limit", INT)),
            outputs=(OutputSpec("result", BOOL),),
        )

    @classmethod
    def execute(cls, value: int, limit: int) -> Mapping[str, object]:
        return cls.outputs(result=value < limit)


class ListIdentity(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.list_identity",
            inputs=(InputSpec("value", TypeExpr.list_of(INT)),),
            outputs=(OutputSpec("value", TypeExpr.list_of(INT)),),
        )

    @classmethod
    def execute(cls, value: list[int]) -> Mapping[str, object]:
        return cls.outputs(value=value)


class CounterStep(Node):
    """While-body: increments count and reports whether to keep going."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.counter_step",
            inputs=(InputSpec("count", INT), InputSpec("limit", INT)),
            outputs=(OutputSpec("next", INT), OutputSpec("keep_going", BOOL)),
        )

    @classmethod
    def execute(cls, count: int, limit: int) -> Mapping[str, object]:
        return cls.outputs(next=count + 1, keep_going=count + 1 < limit)


class MaybeDouble(Node):
    """Optional output: absent for odd inputs."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.maybe_double",
            inputs=(InputSpec("value", INT),),
            outputs=(OutputSpec("out", INT, optional=True),),
        )

    @classmethod
    def execute(cls, value: int) -> Mapping[str, object]:
        if value % 2 != 0:
            return cls.outputs(out=AbsentOutput("odd input"))
        return cls.outputs(out=value * 2)


class MaybeList(Node):
    """Optional list output: the region-input absence source."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.maybe_list",
            inputs=(InputSpec("produce", BOOL),),
            outputs=(OutputSpec("values", TypeExpr.list_of(INT), optional=True),),
        )

    @classmethod
    def execute(cls, produce: bool) -> Mapping[str, object]:
        if not produce:
            return cls.outputs(values=AbsentOutput("nothing to iterate"))
        return cls.outputs(values=[1, 2, 3])


class FailOnOdd(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.fail_on_odd",
            inputs=(InputSpec("value", INT),),
            outputs=(OutputSpec("out", INT),),
        )

    @classmethod
    def execute(cls, value: int) -> Mapping[str, object]:
        if value % 2 != 0:
            raise ValueError("odd input failed")
        return cls.outputs(out=value)


class Sleeper(Node):
    """Async body node that records concurrent executions."""

    active = 0
    peak = 0

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.region_sleeper",
            inputs=(InputSpec("value", INT),),
            outputs=(OutputSpec("out", INT),),
            io_bound=True,  # a wait, not compute: overlaps freely
        )

    @classmethod
    async def execute(cls, value: int) -> Mapping[str, object]:
        Sleeper.active += 1
        Sleeper.peak = max(Sleeper.peak, Sleeper.active)
        await asyncio.sleep(0.02)
        Sleeper.active -= 1
        return cls.outputs(out=value)


class GenericIdentity(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.generic_identity",
            inputs=(InputSpec("value", GENERIC),),
            outputs=(OutputSpec("value", GENERIC),),
        )

    @classmethod
    def execute(cls, value: object) -> Mapping[str, object]:
        return cls.outputs(value=value)


NODES: tuple[type[Node], ...] = (
    AddOne,
    AddPair,
    Mul,
    IsEven,
    BelowLimit,
    ListIdentity,
    CounterStep,
    MaybeDouble,
    MaybeList,
    FailOnOdd,
    Sleeper,
    GenericIdentity,
)


def make_engine(
    events: list[EngineEvent] | None = None,
    extra_nodes: Sequence[type[Node]] = (),
    extra_types: Sequence[str] = (),
) -> Engine:
    node_types = (*NODES, *extra_nodes)
    registry = TypeRegistry()
    register_core_types(registry)
    for type_id in extra_types:
        registry.register(type_id)
    return Engine(
        schemas=build_schemas(node_types),
        registry=registry,
        worker=InProcessWorker(build_node_types(node_types), registry),
        cache=MemoryLRUCache(),
        on_event=None if events is None else events.append,
    )


def port(port_id: str) -> Link:
    return Link(PORTS_NODE_ID, port_id)


def map_region(**overrides: object) -> RegionNode:
    """A canonical map region: add one to every element."""
    fields: dict[str, object] = {
        "kind": "map",
        "body": Graph(nodes={"add": GraphNode("test.add_one", {"value": port("item")})}),
        "ports": {"item": INT},
        "inputs": {"item": [10, 20, 30]},
        "element_ports": ("item",),
        "outputs": {"results": RegionOutput(Link("add", "out"))},
    }
    fields.update(overrides)
    return RegionNode(**fields)  # type: ignore[arg-type]


def broadcast_region(**overrides: object) -> RegionNode:
    """A canonical broadcast map region: add aligned list elements."""
    fields: dict[str, object] = {
        "kind": "map",
        "binding": "broadcast",
        "body": Graph(
            nodes={"add": GraphNode("test.add_pair", {"a": port("left"), "b": port("right")})}
        ),
        "ports": {"left": INT, "right": INT},
        "inputs": {"left": [1, 2], "right": [10, 20, 30, 40, 50]},
        "element_ports": ("left", "right"),
        "outputs": {"results": RegionOutput(Link("add", "out"))},
    }
    fields.update(overrides)
    return RegionNode(**fields)  # type: ignore[arg-type]


def flatten_region(**overrides: object) -> RegionNode:
    fields: dict[str, object] = {
        "kind": "map",
        "body": Graph(nodes={}),
        "ports": {"items": TypeExpr.list_of(INT)},
        "inputs": {"items": [[1], [2, 3], [4, 5, 6]]},
        "element_ports": ("items",),
        "outputs": {"items": RegionOutput(port("items"), mode="flatten")},
    }
    fields.update(overrides)
    return RegionNode(**fields)  # type: ignore[arg-type]


def fold_region(**overrides: object) -> RegionNode:
    """A canonical fold region: sum elements through a state chain."""
    fields: dict[str, object] = {
        "kind": "fold",
        "body": Graph(
            nodes={"acc": GraphNode("test.add_pair", {"a": port("total"), "b": port("item")})}
        ),
        "ports": {"item": INT, "total": INT},
        "inputs": {"item": [1, 2, 3, 4], "total": 0},
        "element_ports": ("item",),
        "state_ports": ("total",),
        "outputs": {"total": RegionOutput(Link("acc", "out"), mode="state")},
    }
    fields.update(overrides)
    return RegionNode(**fields)  # type: ignore[arg-type]


def while_region(**overrides: object) -> RegionNode:
    """A canonical while region: count up to a limit."""
    fields: dict[str, object] = {
        "kind": "while",
        "body": Graph(
            nodes={
                "step": GraphNode(
                    "test.counter_step",
                    {"count": port("count"), "limit": port("limit")},
                )
            }
        ),
        "ports": {"count": INT, "limit": INT},
        "inputs": {"count": 0, "limit": 5},
        "state_ports": ("count",),
        "outputs": {"count": RegionOutput(Link("step", "next"), mode="state")},
        "continue_source": Link("step", "keep_going"),
        "max_iterations": 100,
    }
    fields.update(overrides)
    return RegionNode(**fields)  # type: ignore[arg-type]


# -- wire roundtrips ---------------------------------------------------------


def test_map_region_wire_roundtrip() -> None:
    graph = Graph(nodes={"m": map_region()})
    wire = graph_to_wire(graph)
    assert graph_from_wire(wire) == graph
    region = wire["nodes"]["m"]["region"]
    assert region["kind"] == "map"
    assert region["elementPorts"] == ["item"]
    assert "binding" not in region  # zip is the default, omitted on the wire
    assert region["ports"]["item"] == {"kind": "concrete", "types": ["core.int"]}
    assert region["outputs"]["results"] == {
        "source": {"node": "add", "output": "out"},
        "mode": "gather",
    }
    body_input = region["body"]["nodes"]["add"]["inputs"]["value"]
    assert body_input == {"$link": {"node": "$region", "output": "item"}}


def test_implicit_region_index_wire_roundtrip() -> None:
    region = map_region(
        body=Graph(nodes={}),
        outputs={"indexes": RegionOutput(port(REGION_INDEX_PORT_ID))},
    )
    graph = Graph(nodes={"m": region})
    wire = graph_to_wire(graph)
    region_wire = wire["nodes"]["m"]["region"]

    assert REGION_INDEX_PORT_ID not in region_wire["ports"]
    assert region_wire["outputs"]["indexes"]["source"] == {
        "node": PORTS_NODE_ID,
        "output": REGION_INDEX_PORT_ID,
    }
    assert graph_from_wire(wire) == graph


def test_stream_region_port_wire_roundtrip() -> None:
    stream = TypeExpr.stream_of(TypeExpr.concrete("comfy.IMAGE"))
    graph = Graph(nodes={"m": map_region(ports={"item": stream})})
    wire = graph_to_wire(graph)
    assert wire["nodes"]["m"]["region"]["ports"]["item"] == {
        "kind": "stream",
        "element": {"kind": "concrete", "types": ["comfy.IMAGE"]},
    }
    assert graph_from_wire(wire) == graph


def test_fold_and_while_region_wire_roundtrip() -> None:
    graph = Graph(nodes={"f": fold_region(), "w": while_region()})
    wire = graph_to_wire(graph)
    assert graph_from_wire(wire) == graph
    w = wire["nodes"]["w"]["region"]
    assert w["statePorts"] == ["count"]
    assert w["maxIterations"] == 100
    assert w["continueSource"] == {"node": "step", "output": "keep_going"}


def test_nested_region_wire_roundtrip() -> None:
    inner = map_region(inputs={"item": port("row")})
    outer = RegionNode(
        kind="map",
        body=Graph(nodes={"inner": inner}),
        ports={"row": TypeExpr.list_of(INT)},
        inputs={"row": [[1, 2], [3, 4]]},
        element_ports=("row",),
        outputs={"rows": RegionOutput(Link("inner", "results"))},
    )
    graph = Graph(nodes={"o": outer})
    assert graph_from_wire(graph_to_wire(graph)) == graph


def test_cross_binding_wire_roundtrip() -> None:
    graph = Graph(nodes={"m": map_region(binding="cross")})
    wire = graph_to_wire(graph)
    assert wire["nodes"]["m"]["region"]["binding"] == "cross"
    assert graph_from_wire(wire) == graph


def test_broadcast_binding_wire_roundtrip() -> None:
    graph = Graph(nodes={"m": broadcast_region()})
    wire = graph_to_wire(graph)
    assert wire["nodes"]["m"]["region"]["binding"] == "broadcast"
    assert graph_from_wire(wire) == graph


def test_flatten_output_wire_roundtrip() -> None:
    graph = Graph(nodes={"m": flatten_region()})
    wire = graph_to_wire(graph)
    assert wire["nodes"]["m"]["region"]["outputs"]["items"]["mode"] == "flatten"
    assert graph_from_wire(wire) == graph


def test_compact_output_wire_roundtrip() -> None:
    region = map_region(outputs={"results": RegionOutput(Link("add", "out"), mode="compact")})
    graph = Graph(nodes={"m": region})
    wire = graph_to_wire(graph)
    assert wire["nodes"]["m"]["region"]["outputs"]["results"]["mode"] == "compact"
    assert graph_from_wire(wire) == graph


def test_last_output_and_rerun_cache_policy_wire_roundtrip() -> None:
    region = map_region(
        outputs={"result": RegionOutput(Link("add", "out"), mode="last")},
        cache_policy="rerun",
    )
    graph = Graph(nodes={"m": region})
    wire = graph_to_wire(graph)
    region_wire = wire["nodes"]["m"]["region"]
    assert region_wire["outputs"]["result"]["mode"] == "last"
    assert region_wire["cachePolicy"] == "rerun"
    assert graph_from_wire(wire) == graph


def test_reuse_cache_policy_is_omitted_on_wire() -> None:
    wire = graph_to_wire(Graph(nodes={"m": map_region()}))
    assert "cachePolicy" not in wire["nodes"]["m"]["region"]
    assert graph_from_wire(wire).nodes["m"].cache_policy == "reuse"  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        ({"kind": "loop"}, "unknown kind"),
        ({"binding": "diagonal"}, "unknown binding"),
        ({"cachePolicy": "sometimes"}, "cache policy"),
        ({"maxIterations": True}, "maxIterations"),
        ({"maxIterations": "5"}, "maxIterations"),
        ({"elementPorts": "item"}, "elementPorts"),
        ({"ports": {"item": {"kind": "nope", "types": []}}}, "malformed port type"),
        ({"outputs": {"r": {"source": {"node": "add"}, "mode": "gather"}}}, "source"),
        ({"outputs": {"r": {"source": {"node": "a", "output": "o"}, "mode": "x"}}}, "mode"),
        ({"body": []}, "graph wire"),
    ],
)
def test_malformed_region_wire_is_rejected(mutate: dict[str, object], match: str) -> None:
    wire = graph_to_wire(Graph(nodes={"m": map_region()}))
    wire["nodes"]["m"]["region"].update(mutate)
    with pytest.raises(GraphWireError, match=match):
        graph_from_wire(wire)


def test_snapshot_graph_deep_copies_regions() -> None:
    inputs: dict[str, object] = {"item": [1, 2]}
    body_inputs: dict[str, object] = {"value": port("item")}
    region = RegionNode(
        kind="map",
        body=Graph(nodes={"add": GraphNode("test.add_one", body_inputs)}),
        ports={"item": INT},
        inputs=inputs,
        element_ports=("item",),
        outputs={"results": RegionOutput(Link("add", "out"))},
    )
    snap = snapshot_graph(Graph(nodes={"m": region}))
    inputs["item"] = "MUTATED"
    body_inputs["value"] = "MUTATED"
    snapped = snap.nodes["m"]
    assert isinstance(snapped, RegionNode)
    assert snapped.inputs["item"] == [1, 2]
    body_node = snapped.body.nodes["add"]
    assert isinstance(body_node, GraphNode)
    assert body_node.inputs["value"] == port("item")
    with pytest.raises(TypeError):
        snapped.inputs["item"] = []  # type: ignore[index]


def test_snapshot_graph_copies_nested_literals_and_typed_literal_values() -> None:
    from dinkster_graph import TypedLiteral

    plain: dict[str, object] = {"items": [1, {"name": "before"}]}
    typed_value: list[object] = [{"nested": [2]}]
    graph = Graph(
        nodes={
            "n": GraphNode(
                "test.add_one",
                {"plain": plain, "typed": TypedLiteral("core.any", typed_value)},
            )
        }
    )
    snap = snapshot_graph(graph)
    plain["items"] = ["changed"]
    cast(dict[str, object], typed_value[0])["nested"] = [99]

    node = cast(GraphNode, snap.nodes["n"])
    assert node.inputs["plain"] == {"items": [1, {"name": "before"}]}
    typed = cast(TypedLiteral, node.inputs["typed"])
    assert typed.value == [{"nested": [2]}]


def test_invalid_direct_region_modes_have_named_diagnostics() -> None:
    body = Graph(nodes={"step": GraphNode("test.add_one", {"value": 1})})
    region = RegionNode(
        kind=cast(RegionKind, "repeat"),
        body=body,
        outputs={
            "out": RegionOutput(
                Link("step", "out"),
                mode=cast(Literal["gather", "state"], "collect"),
            )
        },
        binding=cast(BindingMode, "pairwise"),
        cache_policy=cast(Literal["reuse", "rerun"], "sometimes"),
    )
    found = codes(Graph(nodes={"bad": region}), ["bad"])
    assert {
        "unknown-region-kind",
        "unknown-binding-mode",
        "unknown-cache-policy",
        "unknown-output-mode",
    } <= found.keys()


# -- validation --------------------------------------------------------------


def schemas() -> dict[str, NodeSchema]:
    return build_schemas(NODES)


def codes(graph: Graph, targets: list[str]) -> dict[str, list[str]]:
    by_code: dict[str, list[str]] = {}
    for d in validate(graph, schemas(), targets):
        by_code.setdefault(d.code, []).append(d.node_id or "")
    return by_code


def test_valid_regions_pass_validation() -> None:
    graph = Graph(nodes={"m": map_region(), "f": fold_region(), "w": while_region()})
    diags = validate(graph, schemas(), ["m", "f", "w"])
    assert not has_errors(diags), [d.message for d in diags]


def test_reserved_region_id_is_rejected() -> None:
    graph = Graph(nodes={PORTS_NODE_ID: GraphNode("test.add_one", {"value": 1})})
    assert "reserved-node-id" in codes(graph, [])


def test_node_ids_with_path_characters_rejected_by_validation() -> None:
    # Directly constructed graphs (not just wire-decoded ones) get a
    # structured diagnostic: '/', '[' and ']' would break the closed
    # iteration-id/diagnostic-path grammar.
    for bad_id in ("a/b", "a[0]", "a]b"):
        graph = Graph(nodes={bad_id: GraphNode("test.add_one", {"value": 1})})
        assert "invalid-node-id" in codes(graph, [])


def test_nested_body_node_ids_with_path_characters_rejected() -> None:
    for bad_id in ("in/ner", "n[0]", "n]"):
        region = map_region(
            body=Graph(nodes={bad_id: GraphNode("test.add_one", {"value": port("item")})}),
            outputs={"results": RegionOutput(Link(bad_id, "out"))},
        )
        by_code = codes(Graph(nodes={"m": region}), ["m"])
        assert "invalid-node-id" in by_code
        # The diagnostic anchors to the full nested path.
        assert by_code["invalid-node-id"] == [f"m/{bad_id}"]


def test_undeclared_port_and_dangling_port() -> None:
    region = map_region(
        inputs={"item": [1], "mystery": 5},
        body=Graph(nodes={"add": GraphNode("test.add_one", {"value": port("nonexistent")})}),
    )
    by_code = codes(Graph(nodes={"m": region}), ["m"])
    assert by_code["undeclared-port"] == ["m"]
    assert by_code["dangling-port"] == ["m/add"]  # body diags carry region/node ids


def test_element_port_requires_declaration_and_input() -> None:
    region = map_region(ports={}, inputs={})
    by_code = codes(Graph(nodes={"m": region}), ["m"])
    assert "undeclared-port" in by_code
    assert "missing-input" in by_code


def test_kind_profiles_are_enforced() -> None:
    bad_map = map_region(
        ports={"item": INT, "s": INT},
        inputs={"item": [1], "s": 0},
        state_ports=("s",),
        outputs={
            "results": RegionOutput(Link("add", "out")),
            "s": RegionOutput(Link("add", "out"), mode="state"),
        },
    )
    assert "region-shape" in codes(Graph(nodes={"m": bad_map}), ["m"])

    bad_while = while_region(continue_source=None, max_iterations=None)
    shape = codes(Graph(nodes={"w": bad_while}), ["w"])["region-shape"]
    assert len(shape) == 2  # missing continue source AND missing cap

    for binding in ("cross", "broadcast"):
        bound_while = while_region(binding=cast("BindingMode", binding))
        assert "region-shape" in codes(Graph(nodes={"w": bound_while}), ["w"])


def test_stateless_fold_is_valid() -> None:
    region = fold_region(
        body=Graph(nodes={"add": GraphNode("test.add_one", {"value": port("item")})}),
        ports={"item": INT},
        inputs={"item": [1, 2, 3]},
        state_ports=(),
        outputs={"results": RegionOutput(Link("add", "out"))},
    )
    assert not has_errors(validate(Graph(nodes={"f": region}), schemas(), ["f"]))


def test_state_chain_must_close() -> None:
    # State port with no state-mode output naming its next value.
    open_chain = fold_region(outputs={"other": RegionOutput(Link("acc", "out"))})
    assert "state-chain" in codes(Graph(nodes={"f": open_chain}), ["f"])
    # State-mode output that is not a declared state port.
    rogue = fold_region(
        outputs={
            "total": RegionOutput(Link("acc", "out"), mode="state"),
            "rogue": RegionOutput(Link("acc", "out"), mode="state"),
        }
    )
    assert "state-chain" in codes(Graph(nodes={"f": rogue}), ["f"])


def test_region_output_sources_must_exist() -> None:
    region = map_region(outputs={"results": RegionOutput(Link("nope", "out"))})
    assert "dangling-output" in codes(Graph(nodes={"m": region}), ["m"])


def test_gather_requires_runtime_resolvable_type() -> None:
    region = map_region(
        ports={"item": TypeExpr.wildcard()},
        outputs={"results": RegionOutput(port("item"))},
        body=Graph(nodes={}),
        inputs={"item": Link("src", "values")},
    )
    graph = Graph(
        nodes={
            "src": GraphNode("test.maybe_list", {"produce": True}),
            "m": region,
        }
    )
    assert "gather-nonconcrete" in codes(graph, ["m"])


def test_implicit_region_index_is_a_concrete_body_output() -> None:
    region = map_region(
        body=Graph(nodes={}),
        outputs={"indexes": RegionOutput(port(REGION_INDEX_PORT_ID))},
    )
    node_schemas = schemas()

    diags = validate(Graph(nodes={"m": region}), node_schemas, ["m"])

    assert not has_errors(diags), [diag.message for diag in diags]
    assert region_interface(region, node_schemas)["indexes"] == TypeExpr.list_of(INT)


def test_generic_body_output_resolves_from_implicit_region_index() -> None:
    region = map_region(
        body=Graph(
            nodes={
                "route": GraphNode(
                    "dinkster.route.switch",
                    {
                        "index": 0,
                        "values.only": port(REGION_INDEX_PORT_ID),
                    },
                )
            }
        ),
        outputs={"indexes": RegionOutput(Link("route", "value"))},
    )
    node_schemas = build_schemas((*NODES, RouteSwitch))

    diags = validate(Graph(nodes={"m": region}), node_schemas, ["m"])

    assert not has_errors(diags), [diag.message for diag in diags]
    assert region_interface(region, node_schemas)["indexes"] == TypeExpr.list_of(INT)


def test_gather_resolves_route_switch_output_from_linked_inputs() -> None:
    region = map_region(
        body=Graph(
            nodes={
                "route": GraphNode(
                    "dinkster.route.switch",
                    {
                        "index": 0,
                        "values.first": port("item"),
                        "values.second": port("item"),
                    },
                )
            }
        ),
        outputs={"results": RegionOutput(Link("route", "value"))},
    )
    node_schemas = build_schemas((*NODES, RouteSwitch))

    diags = validate(Graph(nodes={"m": region}), node_schemas, ["m"])

    assert "gather-nonconcrete" not in {diag.code for diag in diags}
    assert region_interface(region, node_schemas)["results"] == TypeExpr.list_of(INT)


def test_gather_resolves_nested_list_element_output_from_linked_input() -> None:
    nested_ints = TypeExpr.list_of(TypeExpr.list_of(INT))
    region = map_region(
        ports={"rows": nested_ints},
        inputs={"rows": [[[1], [2, 3]]]},
        element_ports=("rows",),
        body=Graph(
            nodes={"pick": GraphNode("std.list.element", {"list": port("rows"), "index": 0})}
        ),
        outputs={"results": RegionOutput(Link("pick", "item"))},
    )
    node_schemas = build_schemas((*NODES, ListElement))

    diags = validate(Graph(nodes={"m": region}), node_schemas, ["m"])

    assert "gather-nonconcrete" not in {diag.code for diag in diags}
    assert region_interface(region, node_schemas)["results"] == nested_ints


def test_gather_resolves_generic_consumer_of_nested_region_output() -> None:
    inner = map_region(inputs={"item": port("row")})
    outer = RegionNode(
        kind="map",
        body=Graph(
            nodes={
                "inner": inner,
                "pick": GraphNode(
                    "std.list.element",
                    {"list": Link("inner", "results"), "index": 0},
                ),
            }
        ),
        ports={"row": TypeExpr.list_of(INT)},
        inputs={"row": [[1, 2], [3]]},
        element_ports=("row",),
        outputs={"results": RegionOutput(Link("pick", "item"))},
    )
    node_schemas = build_schemas((*NODES, ListElement))

    diags = validate(Graph(nodes={"outer": outer}), node_schemas, ["outer"])

    assert "gather-nonconcrete" not in {diag.code for diag in diags}
    assert region_interface(outer, node_schemas)["results"] == TypeExpr.list_of(INT)


def test_gather_resolves_reverse_ordered_generic_chain_to_fixed_point() -> None:
    nested_ints = TypeExpr.list_of(TypeExpr.list_of(INT))
    region = map_region(
        ports={"rows": nested_ints},
        inputs={"rows": [[[1], [2, 3]]]},
        element_ports=("rows",),
        body=Graph(
            nodes={
                "second": GraphNode(
                    "std.list.element",
                    {"list": Link("first", "item"), "index": 0},
                ),
                "first": GraphNode(
                    "std.list.element",
                    {"list": port("rows"), "index": 0},
                ),
            }
        ),
        outputs={"results": RegionOutput(Link("second", "item"))},
    )
    node_schemas = build_schemas((*NODES, ListElement))

    diags = validate(Graph(nodes={"m": region}), node_schemas, ["m"])

    assert "gather-nonconcrete" not in {diag.code for diag in diags}
    assert region_interface(region, node_schemas)["results"] == TypeExpr.list_of(INT)


def test_gather_refuses_typed_literal_only_and_conflicting_generic_outputs() -> None:
    from dinkster_graph import TypedLiteral

    stamped = map_region(
        body=Graph(
            nodes={
                "route": GraphNode(
                    "dinkster.route.switch",
                    {"index": 0, "values.only": TypedLiteral("core.int", 1)},
                )
            }
        ),
        outputs={"results": RegionOutput(Link("route", "value"))},
    )
    conflicting = map_region(
        ports={"item": INT, "label": STRING},
        inputs={"item": [1], "label": ["one"]},
        element_ports=("item", "label"),
        body=Graph(
            nodes={
                "select": GraphNode(
                    "dinkster.value.select",
                    {
                        "condition": True,
                        "on_false": port("item"),
                        "on_true": port("label"),
                    },
                )
            }
        ),
        outputs={"results": RegionOutput(Link("select", "value"))},
    )
    node_schemas = build_schemas((*NODES, RouteSwitch, ValueSelect))

    stamped_codes = {
        diag.code for diag in validate(Graph(nodes={"m": stamped}), node_schemas, ["m"])
    }
    conflicting_codes = {
        diag.code for diag in validate(Graph(nodes={"m": conflicting}), node_schemas, ["m"])
    }

    assert "gather-nonconcrete" in stamped_codes
    assert "gather-nonconcrete" in conflicting_codes


def test_flatten_refuses_scalar_body_output_with_named_diagnostic() -> None:
    region = map_region(outputs={"results": RegionOutput(Link("add", "out"), mode="flatten")})
    by_code = codes(Graph(nodes={"m": region}), ["m"])
    assert by_code["flatten-non-list"] == ["m"]


def test_region_edges_check_cardinality_both_ways() -> None:
    # A scalar into an element port (which expects list<T>) is an error...
    scalar_in = map_region(inputs={"item": Link("one", "out")})
    graph = Graph(nodes={"one": GraphNode("test.add_one", {"value": 1}), "m": scalar_in})
    assert "scalar-into-list" in codes(graph, ["m"])
    # ...and a region's gather (list<T>) into a scalar consumer input too.
    graph2 = Graph(
        nodes={
            "m": map_region(),
            "c": GraphNode("test.add_one", {"value": Link("m", "results")}),
        }
    )
    by_code = codes(graph2, ["c"])
    assert "list-into-scalar" in by_code
    diag = next(d for d in validate(graph2, schemas(), ["c"]) if d.code == "list-into-scalar")
    assert diag.node_id == "c"
    assert diag.input_id == "value"  # structured anchoring for frontends


def test_sibling_links_resolve_region_outputs() -> None:
    graph = Graph(
        nodes={
            "f": fold_region(),
            "c": GraphNode("test.add_one", {"value": Link("f", "total")}),
            "bad": GraphNode("test.add_one", {"value": Link("f", "nope")}),
        }
    )
    by_code = codes(graph, ["c", "bad"])
    assert by_code["dangling-output"] == ["bad"]


def test_region_interface_types() -> None:
    interface = region_interface(map_region(), schemas())
    assert interface["results"] == TypeExpr.list_of(INT)
    interface = region_interface(fold_region(), schemas())
    assert interface["total"] == INT
    compact = map_region(outputs={"results": RegionOutput(Link("add", "out"), mode="compact")})
    assert region_interface(compact, schemas())["results"] == TypeExpr.list_of(INT)


def test_nested_region_bodies_validate_recursively() -> None:
    inner = map_region(inputs={"item": port("row")})
    broken_inner = map_region(
        inputs={"item": port("row")},
        body=Graph(nodes={"add": GraphNode("test.add_one", {"value": port("ghost")})}),
    )
    outer = RegionNode(
        kind="map",
        body=Graph(nodes={"inner": inner, "broken": broken_inner}),
        ports={"row": TypeExpr.list_of(INT)},
        inputs={"row": [[1]]},
        element_ports=("row",),
        outputs={"rows": RegionOutput(Link("inner", "results"))},
    )
    diags = validate(Graph(nodes={"o": outer}), schemas(), ["o"])
    dangling = next(d for d in diags if d.code == "dangling-port")
    assert dangling.node_id == "o/broken/add"  # hierarchical ids all the way down


# -- engine: map -------------------------------------------------------------


def test_map_region_executes_per_element() -> None:
    async def scenario() -> None:
        AddOne.ran.clear()
        events: list[EngineEvent] = []
        engine = make_engine(events)
        result = await engine.run(Graph(nodes={"m": map_region()}), ["m"])
        out = result.outputs["m"]["results"]
        children = list_children(out)
        assert children is not None
        assert [c.resolve() for c in children] == [11, 21, 31]
        assert sorted(AddOne.ran) == [10, 20, 30]
        assert sorted(result.executed) == ["m[0]/add", "m[1]/add", "m[2]/add"]
        expanded = next(e for e in events if e.kind == "region_expanded")
        assert expanded.node_id == "m"
        assert expanded.detail["iterations"] == 3
        started = [
            cast(int, e.detail["iteration"]) for e in events if e.kind == "region_iteration_started"
        ]
        completed = [
            cast(int, e.detail["iteration"])
            for e in events
            if e.kind == "region_iteration_finished"
        ]
        assert sorted(started) == [0, 1, 2]
        assert sorted(completed) == [0, 1, 2]
        for iteration in range(3):
            assert next(
                i
                for i, event in enumerate(events)
                if event.kind == "region_iteration_started"
                and event.detail["iteration"] == iteration
            ) < next(
                i
                for i, event in enumerate(events)
                if event.kind == "region_iteration_finished"
                and event.detail["iteration"] == iteration
            )
        finished = next(e for e in events if e.kind == "region_finished")
        assert finished.detail["iterations"] == 3

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("type_id", "payloads"),
    [
        ("comfy.IMAGE", [[[[0.0, 0.25, 0.5], [0.75, 1.0, 0.125]]], [[[1.0, 0.5, 0.0]]]]),
        ("comfy.LATENT", [{"samples": [[[[1.5, -2.0], [3.25, 4.5]]]]}, {"samples": [[[[9.0]]]]}]),
        ("dinkster.conditioning", [{"tokens": [1, 4], "weight": 0.75}, {"tokens": [9]}]),
        ("comfy.MASK", [[[0.0, 1.0], [0.25, 0.75]], [[1.0]]]),
        (
            "comfy.AUDIO",
            [
                {"waveform": [[0.0, -0.5, 0.5]], "sample_rate": 48000},
                {"waveform": [[1.0]], "sample_rate": 44100},
            ],
        ),
        ("comfy.VIDEO", [{"frames": ["a", "b"], "fps": 24.0}, {"frames": ["c"], "fps": 30.0}]),
        (
            "dinkster.asset",
            [
                {"digest": "blake3:" + "1" * 64, "name": "one.png"},
                {"digest": "blake3:" + "2" * 64, "name": "two.wav"},
            ],
        ),
        ("core.string", ["alpha", "beta"]),
        ("core.int", [7, -3]),
        ("core.float", [1.25, -8.5]),
    ],
)
def test_map_gather_preserves_generic_payload_type_and_exact_values(
    type_id: str, payloads: list[object]
) -> None:
    async def scenario() -> None:
        value_type = TypeExpr.concrete(type_id)
        region = RegionNode(
            kind="map",
            body=Graph(
                nodes={"identity": GraphNode("test.generic_identity", {"value": port("item")})}
            ),
            ports={"item": value_type},
            inputs={"item": payloads},
            element_ports=("item",),
            outputs={"results": RegionOutput(Link("identity", "value"))},
        )
        core_types = {"core.string", "core.int", "core.float"}
        engine = make_engine(extra_types=() if type_id in core_types else (type_id,))
        result = await engine.run(Graph(nodes={"m": region}), ["m"])
        gathered = result.outputs["m"]["results"]
        assert gathered.type_id == f"list<{type_id}>"
        children = list_children(gathered)
        assert children is not None
        assert [child.type_id for child in children] == [type_id, type_id]
        assert [child.resolve() for child in children] == payloads

    asyncio.run(scenario())


def test_fold_state_preserves_heterogeneous_list_backed_value() -> None:
    async def scenario() -> None:
        heterogeneous_type_id = "test.heterogeneous"
        heterogeneous = TypeExpr.concrete(heterogeneous_type_id)
        region = RegionNode(
            kind="fold",
            body=Graph(
                nodes={"identity": GraphNode("test.generic_identity", {"value": port("carry")})}
            ),
            ports={"item": INT, "carry": heterogeneous},
            inputs={"item": [0, 1], "carry": [7, "7"]},
            element_ports=("item",),
            state_ports=("carry",),
            outputs={
                "carry": RegionOutput(Link("identity", "value"), mode="state"),
                "observed": RegionOutput(Link("identity", "value")),
            },
        )
        result = await make_engine(extra_types=(heterogeneous_type_id,)).run(
            Graph(nodes={"f": region}), ["f"]
        )

        assert result.outputs["f"]["carry"].type_id == heterogeneous_type_id
        assert result.outputs["f"]["carry"].resolve() == [7, "7"]
        observed = list_children(result.outputs["f"]["observed"])
        assert observed is not None
        assert [value.type_id for value in observed] == [
            heterogeneous_type_id,
            heterogeneous_type_id,
        ]
        assert [value.resolve() for value in observed] == [[7, "7"], [7, "7"]]

    asyncio.run(scenario())


def test_map_region_selects_lazy_branch_per_iteration() -> None:
    async def scenario() -> None:
        AddOne.ran.clear()
        region = map_region(
            body=Graph(
                nodes={
                    "false": GraphNode("test.add_one", {"value": port("item")}),
                    "true": GraphNode("test.mul", {"a": port("item"), "b": 2}),
                    "select": GraphNode(
                        "dinkster.value.select",
                        {
                            "condition": port("choose"),
                            "on_false": Link("false", "out"),
                            "on_true": Link("true", "out"),
                        },
                    ),
                }
            ),
            ports={"item": INT, "choose": BOOL},
            inputs={"item": [1, 2, 3], "choose": [False, True, False]},
            element_ports=("item", "choose"),
            outputs={"results": RegionOutput(Link("select", "value"))},
        )

        result = await make_engine(extra_nodes=(ValueSelect,)).run(
            Graph(nodes={"m": region}), ["m"]
        )

        assert result.outputs["m"]["results"].resolve() == [2, 4, 4]
        assert AddOne.ran == [1, 3]
        assert "m[0]/true" not in result.executed
        assert "m[1]/false" not in result.executed
        assert "m[2]/true" not in result.executed

    asyncio.run(scenario())


def test_region_selector_can_demand_direct_body_port() -> None:
    async def scenario() -> None:
        region = map_region(
            body=Graph(
                nodes={
                    "select": GraphNode(
                        "dinkster.value.select",
                        {
                            "condition": port("choose"),
                            "on_false": port("left"),
                            "on_true": port("right"),
                        },
                    )
                }
            ),
            ports={"left": INT, "right": INT, "choose": BOOL},
            inputs={
                "left": [1, 2, 3],
                "right": [10, 20, 30],
                "choose": [False, True, False],
            },
            element_ports=("left", "right", "choose"),
            outputs={"results": RegionOutput(Link("select", "value"))},
        )

        result = await make_engine(extra_nodes=(ValueSelect,)).run(
            Graph(nodes={"m": region}), ["m"]
        )

        assert result.outputs["m"]["results"].resolve() == [1, 20, 3]

    asyncio.run(scenario())


def test_map_region_lazy_worker_replays_distinct_iteration_demands() -> None:
    async def scenario() -> None:
        events: list[EngineEvent] = []
        engine = make_engine(events, (RouteSwitch,))
        region = map_region(
            body=Graph(
                nodes={
                    "false": GraphNode("test.add_one", {"value": port("item")}),
                    "true": GraphNode("test.mul", {"a": port("item"), "b": 2}),
                    "route": GraphNode(
                        "dinkster.route.switch",
                        {
                            "index": port("choice"),
                            "values.false": Link("false", "out"),
                            "values.true": Link("true", "out"),
                        },
                    ),
                }
            ),
            ports={"item": INT, "choice": INT},
            inputs={"item": [7, 7], "choice": [0, 1]},
            element_ports=("item", "choice"),
            outputs={"results": RegionOutput(Link("route", "value"))},
        )
        graph = Graph(nodes={"m": region})

        first = await engine.run(graph, ["m"])
        assert first.outputs["m"]["results"].resolve() == [8, 14]
        assert sorted(first.executed) == [
            "m[0]/false",
            "m[0]/route",
            "m[1]/route",
            "m[1]/true",
        ]
        demand_paths = {
            event.node_id
            for event in events
            if event.kind == "node_event" and event.detail.get("name") == "lazy_demand"
        }
        assert demand_paths == {"m[0]/route", "m[1]/route"}

        second = await engine.run(graph, ["m"])
        assert second.outputs["m"]["results"].resolve() == [8, 14]
        assert second.executed == ()
        assert sorted(second.cached) == [
            "m[0]/false",
            "m[0]/route",
            "m[1]/route",
            "m[1]/true",
        ]

    asyncio.run(scenario())


def test_named_route_selects_stable_member_ids_per_iteration() -> None:
    async def scenario() -> None:
        region = map_region(
            body=Graph(
                nodes={
                    "left": GraphNode("test.add_one", {"value": port("item")}),
                    "right": GraphNode("test.mul", {"a": port("item"), "b": 2}),
                    "route": GraphNode(
                        "dinkster.route.switch_by_name",
                        {
                            "choice": port("choice"),
                            "values.stable_left": Link("left", "out"),
                            "values.stable_right": Link("right", "out"),
                        },
                    ),
                }
            ),
            ports={"item": INT, "choice": COMBO},
            inputs={
                "item": [7, 7, 7],
                "choice": ["stable_left", "stable_right", "stable_left"],
            },
            element_ports=("item", "choice"),
            outputs={"results": RegionOutput(Link("route", "value"))},
        )

        result = await make_engine(extra_nodes=(RouteSwitchByName,)).run(
            Graph(nodes={"m": region}), ["m"]
        )

        assert result.outputs["m"]["results"].resolve() == [8, 14, 8]
        assert sorted(result.executed) == [
            "m[0]/left",
            "m[0]/route",
            "m[1]/right",
            "m[1]/route",
        ]
        assert {"m[2]/left", "m[2]/route"} <= set(result.cached)

    asyncio.run(scenario())


def test_lazy_gate_preserves_gather_absence_and_explicit_compaction() -> None:
    async def scenario() -> None:
        body = Graph(
            nodes={
                "add": GraphNode("test.add_one", {"value": port("item")}),
                "gate": GraphNode(
                    "dinkster.route.gate",
                    {"condition": port("keep"), "value": Link("add", "out")},
                ),
            }
        )
        fields = {
            "body": body,
            "ports": {"item": INT, "keep": BOOL},
            "inputs": {"item": [1, 2], "keep": [True, False]},
            "element_ports": ("item", "keep"),
        }
        gathered = map_region(
            **fields,
            outputs={"results": RegionOutput(Link("gate", "value"))},
        )
        compact = map_region(
            **fields,
            outputs={"results": RegionOutput(Link("gate", "value"), mode="compact")},
        )
        engine = make_engine(extra_nodes=(RouteGate,))

        gathered_result = await engine.run(Graph(nodes={"m": gathered}), ["m"])
        compact_result = await engine.run(Graph(nodes={"m": compact}), ["m"])

        assert is_absent(gathered_result.outputs["m"]["results"])
        assert compact_result.outputs["m"]["results"].resolve() == [2]
        assert "m[1]/add" not in gathered_result.executed
        assert "m[1]/add" not in compact_result.executed

    asyncio.run(scenario())


def test_fold_region_selects_lazy_branch_from_prior_state() -> None:
    async def scenario() -> None:
        region = fold_region(
            body=Graph(
                nodes={
                    "even": GraphNode("test.is_even", {"value": port("total")}),
                    "add": GraphNode("test.add_pair", {"a": port("total"), "b": port("item")}),
                    "multiply": GraphNode("test.mul", {"a": port("total"), "b": port("item")}),
                    "select": GraphNode(
                        "dinkster.value.select",
                        {
                            "condition": Link("even", "result"),
                            "on_false": Link("multiply", "out"),
                            "on_true": Link("add", "out"),
                        },
                    ),
                }
            ),
            outputs={"total": RegionOutput(Link("select", "value"), mode="state")},
        )

        result = await make_engine(extra_nodes=(ValueSelect,)).run(
            Graph(nodes={"f": region}), ["f"]
        )

        assert result.outputs["f"]["total"].resolve() == 20
        assert "f[0]/multiply" not in result.executed
        assert "f[1]/add" not in result.executed
        assert "f[2]/multiply" not in result.executed
        assert "f[3]/add" not in result.executed

    asyncio.run(scenario())


def test_while_region_selects_lazy_continuation_branch() -> None:
    async def scenario() -> None:
        region = while_region(
            body=Graph(
                nodes={
                    "next": GraphNode("test.add_one", {"value": port("count")}),
                    "even": GraphNode("test.is_even", {"value": Link("next", "out")}),
                    "short": GraphNode(
                        "test.below_limit", {"value": Link("next", "out"), "limit": 3}
                    ),
                    "long": GraphNode(
                        "test.below_limit", {"value": Link("next", "out"), "limit": 4}
                    ),
                    "select": GraphNode(
                        "dinkster.value.select",
                        {
                            "condition": Link("even", "result"),
                            "on_false": Link("short", "result"),
                            "on_true": Link("long", "result"),
                        },
                    ),
                }
            ),
            outputs={"count": RegionOutput(Link("next", "out"), mode="state")},
            continue_source=Link("select", "value"),
        )

        result = await make_engine(extra_nodes=(ValueSelect,)).run(
            Graph(nodes={"w": region}), ["w"]
        )

        assert result.outputs["w"]["count"].resolve() == 3
        assert "w[0]/long" not in result.executed
        assert "w[1]/short" not in result.executed
        assert "w[2]/long" not in result.executed

    asyncio.run(scenario())


def test_unselected_lazy_branch_does_not_prepare_nested_region() -> None:
    class TrackingWorker(InProcessWorker):
        def __init__(
            self,
            node_types: Mapping[str, type[Node]],
            registry: TypeRegistry,
        ) -> None:
            super().__init__(node_types, registry)
            self.prepared: list[str] = []

        async def prepare(self, node_types: Sequence[str]) -> None:
            self.prepared.extend(node_types)
            await super().prepare(node_types)

    async def scenario() -> None:
        node_types = (*NODES, ValueSelect)
        registry = TypeRegistry()
        register_core_types(registry)
        worker = TrackingWorker(build_node_types(node_types), registry)
        engine = Engine(
            schemas=build_schemas(node_types),
            registry=registry,
            worker=worker,
            cache=MemoryLRUCache(),
        )
        inner = map_region(inputs={"item": port("row")})
        outer = map_region(
            body=Graph(
                nodes={
                    "identity": GraphNode("test.list_identity", {"value": port("row")}),
                    "inner": inner,
                    "select": GraphNode(
                        "dinkster.value.select",
                        {
                            "condition": False,
                            "on_false": Link("identity", "value"),
                            "on_true": Link("inner", "results"),
                        },
                    ),
                }
            ),
            ports={"row": TypeExpr.list_of(INT)},
            inputs={"row": [[5, 6], [7, 8]]},
            element_ports=("row",),
            outputs={"rows": RegionOutput(Link("select", "value"))},
        )

        result = await engine.run(Graph(nodes={"outer": outer}), ["outer"])

        assert result.outputs["outer"]["rows"].resolve() == [[5, 6], [7, 8]]
        assert "test.add_one" not in worker.prepared
        assert worker.prepared.count("dinkster.value.select") == 1
        assert worker.prepared.count("test.list_identity") == 1
        assert all("/inner" not in node_id for node_id in result.executed)

    asyncio.run(scenario())


def test_lazy_map_failure_cancels_selected_sibling_iteration() -> None:
    class CoordinatedBranch(Node):
        started: asyncio.Event
        cancelled: asyncio.Event

        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.coordinated_branch",
                inputs=(InputSpec("fail", BOOL),),
                outputs=(OutputSpec("value", INT),),
                io_bound=True,
            )

        @classmethod
        async def execute(cls, fail: bool) -> Mapping[str, object]:
            if fail:
                await cls.started.wait()
                raise ValueError("selected branch failed")
            cls.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cls.cancelled.set()
                raise
            raise AssertionError("blocking branch unexpectedly resumed")

    async def scenario() -> None:
        CoordinatedBranch.started = asyncio.Event()
        CoordinatedBranch.cancelled = asyncio.Event()
        region = map_region(
            body=Graph(
                nodes={
                    "fail": GraphNode("test.coordinated_branch", {"fail": True}),
                    "block": GraphNode("test.coordinated_branch", {"fail": False}),
                    "select": GraphNode(
                        "dinkster.value.select",
                        {
                            "condition": port("choose"),
                            "on_false": Link("fail", "value"),
                            "on_true": Link("block", "value"),
                        },
                    ),
                }
            ),
            ports={"choose": BOOL},
            inputs={"choose": [False, True]},
            element_ports=("choose",),
            outputs={"results": RegionOutput(Link("select", "value"))},
        )

        with pytest.raises(ExecutionError, match="selected branch failed") as caught:
            await make_engine(extra_nodes=(ValueSelect, CoordinatedBranch)).run(
                Graph(nodes={"m": region}), ["m"]
            )

        assert caught.value.error.node_id == "m[0]/fail"
        assert CoordinatedBranch.started.is_set()
        assert CoordinatedBranch.cancelled.is_set()

    asyncio.run(scenario())


def test_lazy_demanded_region_prepares_its_body_types() -> None:
    class LazyIntConsumer(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.lazy-region-consumer",
                inputs=(InputSpec("value", INT, lazy=True),),
                outputs=(OutputSpec("value", INT),),
            )

        @classmethod
        def check_lazy_status(cls, value: int | None) -> tuple[str, ...]:
            return ("value",) if value is None else ()

        @classmethod
        def execute(cls, value: int | None) -> Mapping[str, object]:
            return cls.outputs(value=value)

    class TrackingWorker(InProcessWorker):
        def __init__(
            self,
            node_types: Mapping[str, type[Node]],
            registry: TypeRegistry,
        ) -> None:
            super().__init__(node_types, registry)
            self.prepared: list[tuple[str, ...]] = []

        async def prepare(self, node_types: Sequence[str]) -> None:
            self.prepared.append(tuple(node_types))
            await super().prepare(node_types)

    async def scenario() -> None:
        node_types = (*NODES, LazyIntConsumer)
        registry = TypeRegistry()
        register_core_types(registry)
        worker = TrackingWorker(build_node_types(node_types), registry)
        engine = Engine(
            schemas=build_schemas(node_types),
            registry=registry,
            worker=worker,
            cache=MemoryLRUCache(),
        )
        graph = Graph(
            {
                "region": fold_region(),
                "consumer": GraphNode(
                    "test.lazy-region-consumer",
                    {"value": Link("region", "total")},
                ),
            }
        )

        result = await engine.run(graph, ["consumer"])
        assert result.outputs["consumer"]["value"].resolve() == 10
        assert ("test.add_pair",) in worker.prepared

    asyncio.run(scenario())


def test_lazy_route_does_not_execute_an_unselected_region() -> None:
    async def scenario() -> None:
        AddPair.calls.clear()
        node_types = (*NODES, RouteSwitch)
        registry = TypeRegistry()
        register_core_types(registry)
        engine = Engine(
            schemas=build_schemas(node_types),
            registry=registry,
            worker=InProcessWorker(build_node_types(node_types), registry),
            cache=MemoryLRUCache(),
        )
        graph = Graph(
            {
                "unselected": fold_region(
                    inputs={"item": [1, 2], "total": 0},
                ),
                "selected": fold_region(
                    inputs={"item": [10, 20], "total": 0},
                ),
                "route": GraphNode(
                    "dinkster.route.switch",
                    {
                        "index": 1,
                        "values.unselected": Link("unselected", "total"),
                        "values.selected": Link("selected", "total"),
                    },
                ),
            }
        )

        result = await engine.run(graph, ["route"])
        assert result.outputs["route"]["value"].resolve() == 30
        assert AddPair.calls == [(0, 10), (10, 20)]
        assert "route" in result.executed
        assert all(not node_id.startswith("unselected") for node_id in result.executed)

    asyncio.run(scenario())


def test_map_region_accepts_typed_literal_input() -> None:
    """Region inputs share the typed-literal wrap path: the stamp names the
    runtime type instead of the declared port type (redundant on this
    concrete port - a warning, not an error - but the wrap must follow the
    stamp exactly as on node inputs)."""
    from dinkster_graph import TypedLiteral

    async def scenario() -> None:
        engine = make_engine()
        region = map_region(inputs={"item": TypedLiteral("list<core.int>", [10, 20, 30])})
        result = await engine.run(Graph(nodes={"m": region}), ["m"])
        children = list_children(result.outputs["m"]["results"])
        assert children is not None
        assert [c.resolve() for c in children] == [11, 21, 31]

    asyncio.run(scenario())


def test_map_region_zero_items_yields_typed_empty_list() -> None:
    async def scenario() -> None:
        AddOne.ran.clear()
        engine = make_engine()
        result = await engine.run(Graph(nodes={"m": map_region(inputs={"item": []})}), ["m"])
        out = result.outputs["m"]["results"]
        assert out.type_id == "list<core.int>"
        assert list_children(out) == ()
        assert AddOne.ran == []  # the body never ran

    asyncio.run(scenario())


def test_region_index_counts_map_fold_and_while_iterations() -> None:
    async def scenario() -> None:
        map_loop = map_region(
            body=Graph(nodes={}),
            outputs={"indexes": RegionOutput(port(REGION_INDEX_PORT_ID))},
        )
        fold_loop = fold_region(
            outputs={
                "total": RegionOutput(Link("acc", "out"), mode="state"),
                "indexes": RegionOutput(port(REGION_INDEX_PORT_ID)),
            }
        )
        while_loop = while_region(
            outputs={
                "count": RegionOutput(Link("step", "next"), mode="state"),
                "indexes": RegionOutput(port(REGION_INDEX_PORT_ID)),
            }
        )
        graph = Graph(nodes={"map": map_loop, "fold": fold_loop, "while": while_loop})

        result = await make_engine().run(graph, ["map", "fold", "while"])

        assert result.outputs["map"]["indexes"].resolve() == [0, 1, 2]
        assert result.outputs["fold"]["indexes"].resolve() == [0, 1, 2, 3]
        assert result.outputs["while"]["indexes"].resolve() == [0, 1, 2, 3, 4]

    asyncio.run(scenario())


def test_nested_region_index_belongs_to_the_immediate_region() -> None:
    async def scenario() -> None:
        inner = RegionNode(
            kind="map",
            body=Graph(nodes={}),
            ports={"item": INT},
            inputs={"item": port("row")},
            element_ports=("item",),
            outputs={"indexes": RegionOutput(port(REGION_INDEX_PORT_ID))},
        )
        outer = RegionNode(
            kind="map",
            body=Graph(nodes={"inner": inner}),
            ports={"row": TypeExpr.list_of(INT)},
            inputs={"row": [[10, 20], [30, 40, 50]]},
            element_ports=("row",),
            outputs={"indexes": RegionOutput(Link("inner", "indexes"))},
        )

        result = await make_engine().run(Graph(nodes={"outer": outer}), ["outer"])

        assert result.outputs["outer"]["indexes"].resolve() == [[0, 1], [0, 1, 2]]

    asyncio.run(scenario())


def test_declared_index_state_shadows_the_implicit_region_index() -> None:
    async def scenario() -> None:
        region = RegionNode(
            kind="while",
            body=Graph(
                nodes={
                    "step": GraphNode(
                        "test.counter_step",
                        {"count": port("index"), "limit": port("limit")},
                    )
                }
            ),
            ports={"index": INT, "limit": INT},
            inputs={"index": 10, "limit": 13},
            state_ports=("index",),
            outputs={"index": RegionOutput(Link("step", "next"), mode="state")},
            continue_source=Link("step", "keep_going"),
            max_iterations=5,
        )

        result = await make_engine().run(Graph(nodes={"while": region}), ["while"])

        assert result.outputs["while"]["index"].resolve() == 13

    asyncio.run(scenario())


def test_declared_index_port_preserves_its_type_and_value() -> None:
    async def scenario() -> None:
        region = map_region(
            body=Graph(nodes={}),
            ports={"item": INT, "index": STRING},
            inputs={"item": [1, 2], "index": "declared"},
            outputs={"indexes": RegionOutput(port(REGION_INDEX_PORT_ID))},
        )
        node_schemas = schemas()

        assert region_interface(region, node_schemas)["indexes"] == TypeExpr.list_of(STRING)
        result = await make_engine().run(Graph(nodes={"m": region}), ["m"])
        assert result.outputs["m"]["indexes"].resolve() == ["declared", "declared"]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("region", "expected"),
    [
        (flatten_region(), [1, 2, 3, 4, 5, 6]),
        (
            flatten_region(
                binding="broadcast",
                ports={"items": TypeExpr.list_of(INT), "ticks": INT},
                inputs={"items": [[1], [2, 3]], "ticks": [10, 20, 30]},
                element_ports=("items", "ticks"),
            ),
            [1, 2, 3, 2, 3],
        ),
    ],
)
def test_flatten_concatenates_in_iteration_order_under_zip_and_broadcast(
    region: RegionNode, expected: list[int]
) -> None:
    async def scenario() -> None:
        result = await make_engine().run(Graph(nodes={"m": region}), ["m"])
        output = result.outputs["m"]["items"]
        assert output.type_id == "list<core.int>"
        assert output.resolve() == expected

    asyncio.run(scenario())


def test_flatten_zero_iterations_yields_typed_empty_list() -> None:
    async def scenario() -> None:
        region = flatten_region(inputs={"items": []})
        result = await make_engine().run(Graph(nodes={"m": region}), ["m"])
        output = result.outputs["m"]["items"]
        assert output.type_id == "list<core.int>"
        assert list_children(output) == ()

    asyncio.run(scenario())


def test_map_iterations_run_concurrently() -> None:
    async def scenario() -> None:
        Sleeper.active = 0
        Sleeper.peak = 0
        engine = make_engine()
        region = map_region(
            body=Graph(nodes={"s": GraphNode("test.region_sleeper", {"value": port("item")})}),
            inputs={"item": [1, 2, 3, 4]},
            outputs={"results": RegionOutput(Link("s", "out"))},
        )
        await engine.run(Graph(nodes={"m": region}), ["m"])
        assert Sleeper.peak >= 2  # independent iterations overlapped

    asyncio.run(scenario())


def test_zip_binding_requires_equal_lengths() -> None:
    async def scenario() -> None:
        engine = make_engine()
        region = RegionNode(
            kind="map",
            body=Graph(
                nodes={"add": GraphNode("test.add_pair", {"a": port("xs"), "b": port("ys")})}
            ),
            ports={"xs": INT, "ys": INT},
            inputs={"xs": [1, 2, 3], "ys": [10, 20]},
            element_ports=("xs", "ys"),
            outputs={"sums": RegionOutput(Link("add", "out"))},
        )
        with pytest.raises(ExecutionError, match="equal-length"):
            await engine.run(Graph(nodes={"m": region}), ["m"])

    asyncio.run(scenario())


def test_broadcast_binding_repeats_final_elements_to_longest_list() -> None:
    async def scenario() -> None:
        engine = make_engine()
        result = await engine.run(Graph(nodes={"m": broadcast_region()}), ["m"])
        children = list_children(result.outputs["m"]["results"])
        assert children is not None
        assert [child.resolve() for child in children] == [11, 22, 32, 42, 52]

    asyncio.run(scenario())


def test_broadcast_binding_promotes_singleton_across_all_iterations() -> None:
    async def scenario() -> None:
        engine = make_engine()
        region = broadcast_region(inputs={"left": [1], "right": [10, 20, 30]})
        result = await engine.run(Graph(nodes={"m": region}), ["m"])
        children = list_children(result.outputs["m"]["results"])
        assert children is not None
        assert [child.resolve() for child in children] == [11, 21, 31]

    asyncio.run(scenario())


def test_broadcast_equal_lengths_match_zip_result() -> None:
    async def scenario() -> None:
        engine = make_engine()
        inputs = {"left": [1, 2, 3], "right": [10, 20, 30]}
        broadcast = broadcast_region(inputs=inputs)
        zipped = broadcast_region(inputs=inputs, binding="zip")
        broadcast_result = await engine.run(Graph(nodes={"m": broadcast}), ["m"])
        zip_result = await engine.run(Graph(nodes={"m": zipped}), ["m"])
        assert (
            broadcast_result.outputs["m"]["results"].fingerprint
            == zip_result.outputs["m"]["results"].fingerprint
        )

    asyncio.run(scenario())


def test_broadcast_mixed_empty_lists_fail_naming_empty_ports() -> None:
    async def scenario() -> None:
        AddPair.calls.clear()
        engine = make_engine()
        region = broadcast_region(inputs={"left": [], "right": [10]})
        with pytest.raises(ExecutionError, match="empty element ports: left"):
            await engine.run(Graph(nodes={"m": region}), ["m"])
        assert AddPair.calls == []

    asyncio.run(scenario())


def test_broadcast_all_empty_lists_produce_zero_iterations() -> None:
    async def scenario() -> None:
        AddPair.calls.clear()
        engine = make_engine()
        region = broadcast_region(inputs={"left": [], "right": []})
        result = await engine.run(Graph(nodes={"m": region}), ["m"])
        children = list_children(result.outputs["m"]["results"])
        assert children == ()
        assert AddPair.calls == []
        assert result.executed == ()

    asyncio.run(scenario())


def test_cross_binding_is_deterministic_cartesian_product() -> None:
    async def scenario() -> None:
        engine = make_engine()
        region = RegionNode(
            kind="map",
            body=Graph(nodes={"mul": GraphNode("test.mul", {"a": port("xs"), "b": port("ys")})}),
            ports={"xs": INT, "ys": INT},
            inputs={"xs": [1, 2, 3], "ys": [10, 100, 1000, 10000]},
            element_ports=("xs", "ys"),
            binding="cross",
            outputs={"products": RegionOutput(Link("mul", "out"))},
        )
        result = await engine.run(Graph(nodes={"m": region}), ["m"])
        children = list_children(result.outputs["m"]["products"])
        assert children is not None
        values = [c.resolve() for c in children]
        assert len(values) == 12  # 3 x 4, last port varying fastest
        assert values[:4] == [10, 100, 1000, 10000]
        assert values[4:8] == [20, 200, 2000, 20000]

    asyncio.run(scenario())


def test_map_binding_count_respects_max_iterations() -> None:
    async def scenario() -> None:
        engine = make_engine()
        region = map_region(inputs={"item": [1, 2, 3]}, max_iterations=2)
        with pytest.raises(ExecutionError, match="max_iterations"):
            await engine.run(Graph(nodes={"m": region}), ["m"])

    asyncio.run(scenario())


# -- engine: fold and while --------------------------------------------------


def test_fold_region_chains_state() -> None:
    async def scenario() -> None:
        engine = make_engine()
        result = await engine.run(Graph(nodes={"f": fold_region()}), ["f"])
        assert result.outputs["f"]["total"].resolve() == 10  # 0+1+2+3+4

    asyncio.run(scenario())


def test_fold_with_zero_elements_returns_initial_state() -> None:
    async def scenario() -> None:
        engine = make_engine()
        region = fold_region(inputs={"item": [], "total": 42})
        result = await engine.run(Graph(nodes={"f": region}), ["f"])
        assert result.outputs["f"]["total"].resolve() == 42

    asyncio.run(scenario())


def test_while_region_iterates_until_continue_false() -> None:
    async def scenario() -> None:
        events: list[EngineEvent] = []
        engine = make_engine(events)
        result = await engine.run(Graph(nodes={"w": while_region()}), ["w"])
        assert result.outputs["w"]["count"].resolve() == 5
        assert [e.detail["iteration"] for e in events if e.kind == "region_iteration_started"] == [
            0,
            1,
            2,
            3,
            4,
        ]
        assert [e.detail["iteration"] for e in events if e.kind == "region_iteration_finished"] == [
            0,
            1,
            2,
            3,
            4,
        ]
        finished = next(e for e in events if e.kind == "region_finished")
        assert finished.detail["iterations"] == 5

    asyncio.run(scenario())


def test_while_region_cap_with_continue_true_is_loud() -> None:
    async def scenario() -> None:
        engine = make_engine()
        region = while_region(max_iterations=3)  # limit 5 is unreachable
        with pytest.raises(ExecutionError, match="max_iterations=3"):
            await engine.run(Graph(nodes={"w": region}), ["w"])

    asyncio.run(scenario())


def test_conditional_is_zero_or_one_expansion() -> None:
    async def scenario() -> None:
        AddOne.ran.clear()
        engine = make_engine()
        # then-branch: one element -> the body runs once
        result = await engine.run(Graph(nodes={"m": map_region(inputs={"item": [7]})}), ["m"])
        children = list_children(result.outputs["m"]["results"])
        assert children is not None and children[0].resolve() == 8
        # else-branch: zero elements -> the body never runs
        AddOne.ran.clear()
        await engine.run(Graph(nodes={"m": map_region(inputs={"item": []})}), ["m"])
        assert AddOne.ran == []

    asyncio.run(scenario())


# -- engine: caching and coalescing ------------------------------------------


def test_per_item_caching_reexecutes_only_changed_iterations() -> None:
    async def scenario() -> None:
        AddOne.ran.clear()
        engine = make_engine()
        await engine.run(Graph(nodes={"m": map_region(inputs={"item": [1, 2, 3]})}), ["m"])
        assert sorted(AddOne.ran) == [1, 2, 3]
        AddOne.ran.clear()
        # Change ONE element: only that iteration's body re-executes.
        result = await engine.run(
            Graph(nodes={"m": map_region(inputs={"item": [1, 99, 3]})}), ["m"]
        )
        assert AddOne.ran == [99]
        assert result.executed == ("m[1]/add",)
        assert sorted(result.cached) == ["m[0]/add", "m[2]/add"]

    asyncio.run(scenario())


def test_duplicate_items_coalesce_through_single_flight() -> None:
    async def scenario() -> None:
        AddOne.ran.clear()
        engine = make_engine()
        result = await engine.run(Graph(nodes={"m": map_region(inputs={"item": [5, 5, 5]})}), ["m"])
        assert AddOne.ran == [5]  # one real execution
        children = list_children(result.outputs["m"]["results"])
        assert children is not None
        assert [c.resolve() for c in children] == [6, 6, 6]
        assert len(result.executed) == 1
        assert len(result.cached) == 2  # coalesced or cache-hit iterations

    asyncio.run(scenario())


def test_region_index_participates_in_cache_identity_when_consumed() -> None:
    async def scenario() -> None:
        AddPair.calls.clear()
        engine = make_engine()
        region = map_region(
            body=Graph(
                nodes={
                    "add": GraphNode(
                        "test.add_pair",
                        {"a": port("item"), "b": port(REGION_INDEX_PORT_ID)},
                    )
                }
            ),
            inputs={"item": [7, 7, 7]},
        )
        graph = Graph(nodes={"m": region})

        first = await engine.run(graph, ["m"])
        assert first.outputs["m"]["results"].resolve() == [7, 8, 9]
        assert sorted(AddPair.calls) == [(7, 0), (7, 1), (7, 2)]
        assert len(first.executed) == 3

        AddPair.calls.clear()
        second = await engine.run(graph, ["m"])
        assert AddPair.calls == []
        assert second.executed == ()
        assert sorted(second.cached) == ["m[0]/add", "m[1]/add", "m[2]/add"]

    asyncio.run(scenario())


def test_stateless_fold_executes_in_iteration_order() -> None:
    async def scenario() -> None:
        AddPair.calls.clear()
        region = RegionNode(
            kind="fold",
            body=Graph(
                nodes={
                    "add": GraphNode(
                        "test.add_pair",
                        {"a": port("item"), "b": port(REGION_INDEX_PORT_ID)},
                    )
                }
            ),
            ports={"item": INT},
            inputs={"item": [30, 10, 20]},
            element_ports=("item",),
            outputs={"results": RegionOutput(Link("add", "out"))},
        )
        result = await make_engine().run(Graph(nodes={"f": region}), ["f"])
        assert AddPair.calls == [(30, 0), (10, 1), (20, 2)]
        assert result.outputs["f"]["results"].resolve() == [30, 11, 22]

    asyncio.run(scenario())


@pytest.mark.parametrize("items", [[4, 8, 15], []])
def test_last_output_returns_final_value_or_typed_absence(items: list[int]) -> None:
    async def scenario() -> None:
        region = map_region(
            inputs={"item": items},
            outputs={"result": RegionOutput(Link("add", "out"), mode="last")},
        )
        result = await make_engine().run(Graph(nodes={"m": region}), ["m"])
        output = result.outputs["m"]["result"]
        if items:
            assert output.resolve() == 16
            assert output.type_id == "core.int"
        else:
            assert is_absent(output)
            assert output.meta.get(ABSENT_ORIGIN_META_KEY) == "m/result"
            assert output.meta.get(ABSENT_STANDS_FOR_META_KEY) == "core.int"

    asyncio.run(scenario())


@pytest.mark.parametrize(("cache_policy", "calls_per_run"), [("reuse", 1), ("rerun", 3)])
def test_region_cache_policy_applies_to_each_occurrence_across_runs(
    cache_policy: Literal["reuse", "rerun"], calls_per_run: int
) -> None:
    async def scenario() -> None:
        AddOne.ran.clear()
        engine = make_engine()
        region = map_region(inputs={"item": [7, 7, 7]}, cache_policy=cache_policy)
        graph = Graph(nodes={"m": region})

        first = await engine.run(graph, ["m"])
        assert first.outputs["m"]["results"].resolve() == [8, 8, 8]
        assert len(AddOne.ran) == calls_per_run

        second = await engine.run(graph, ["m"])
        assert second.outputs["m"]["results"].resolve() == [8, 8, 8]
        assert len(AddOne.ran) == calls_per_run * (1 if cache_policy == "reuse" else 2)
        if cache_policy == "reuse":
            assert len(second.cached) == 3
        else:
            assert second.cached == ()
            assert len(second.executed) == 3

    asyncio.run(scenario())


def test_inner_reuse_overrides_outer_rerun() -> None:
    async def scenario() -> None:
        AddOne.ran.clear()
        AddPair.calls.clear()
        engine = make_engine()
        inner = map_region(inputs={"item": port("row")}, cache_policy="reuse")
        outer = RegionNode(
            kind="map",
            body=Graph(
                nodes={
                    "inner": inner,
                    "outer_probe": GraphNode(
                        "test.add_pair",
                        {"a": port(REGION_INDEX_PORT_ID), "b": 100},
                    ),
                }
            ),
            ports={"row": TypeExpr.list_of(INT)},
            inputs={"row": [[0, 1], [0, 1]]},
            element_ports=("row",),
            outputs={
                "rows": RegionOutput(Link("inner", "results")),
                "outer_probes": RegionOutput(Link("outer_probe", "out")),
            },
            cache_policy="rerun",
        )
        graph = Graph(nodes={"outer": outer})

        first = await engine.run(graph, ["outer"])
        second = await engine.run(graph, ["outer"])

        assert first.outputs["outer"]["rows"].resolve() == [[1, 2], [1, 2]]
        assert second.outputs["outer"]["rows"].resolve() == [[1, 2], [1, 2]]
        assert first.outputs["outer"]["outer_probes"].resolve() == [100, 101]
        assert second.outputs["outer"]["outer_probes"].resolve() == [100, 101]
        assert sorted(AddOne.ran) == [0, 0, 1, 1]
        assert sorted(AddPair.calls) == [(0, 100), (0, 100), (1, 100), (1, 100)]
        assert len(first.executed) == 6
        assert len(second.executed) == 2
        assert first.cached == ()
        assert len(second.cached) == 4

    asyncio.run(scenario())


# -- engine: absence ---------------------------------------------------------


def test_absent_region_input_skips_the_whole_region() -> None:
    async def scenario() -> None:
        AddOne.ran.clear()
        events: list[EngineEvent] = []
        engine = make_engine(events)
        graph = Graph(
            nodes={
                "src": GraphNode("test.maybe_list", {"produce": False}),
                "m": map_region(inputs={"item": Link("src", "values")}),
            }
        )
        result = await engine.run(graph, ["m"])
        out = result.outputs["m"]["results"]
        assert is_absent(out)
        assert out.meta.get(ABSENT_ORIGIN_META_KEY) == "src/values"  # root provenance
        assert AddOne.ran == []
        assert "m" in result.skipped
        skip_event = next(e for e in events if e.kind == "node_skipped")
        assert skip_event.node_id == "m"

    asyncio.run(scenario())


def test_absent_body_output_makes_whole_gather_absent() -> None:
    async def scenario() -> None:
        engine = make_engine()
        region = RegionNode(
            kind="map",
            body=Graph(nodes={"maybe": GraphNode("test.maybe_double", {"value": port("item")})}),
            ports={"item": INT},
            inputs={"item": [2, 3, 4]},  # 3 is odd: its output is absent
            element_ports=("item",),
            outputs={"results": RegionOutput(Link("maybe", "out"))},
        )
        result = await engine.run(Graph(nodes={"m": region}), ["m"])
        out = result.outputs["m"]["results"]
        assert is_absent(out)  # no holes, no compaction: the whole gather
        origin = out.meta.get(ABSENT_ORIGIN_META_KEY)
        assert origin == "m[1]/maybe/out"  # names the iteration that decided

    asyncio.run(scenario())


def test_compact_gather_omits_absent_iterations_in_order() -> None:
    async def scenario() -> None:
        region = RegionNode(
            kind="map",
            body=Graph(nodes={"maybe": GraphNode("test.maybe_double", {"value": port("item")})}),
            ports={"item": INT},
            inputs={"item": [6, 3, 2, 5, 4]},
            element_ports=("item",),
            outputs={"results": RegionOutput(Link("maybe", "out"), mode="compact")},
        )
        result = await make_engine().run(Graph(nodes={"m": region}), ["m"])
        out = result.outputs["m"]["results"]
        assert out.type_id == "list<core.int>"
        assert out.resolve() == [12, 4, 8]

    asyncio.run(scenario())


def test_compact_gather_does_not_turn_failed_iterations_into_absence() -> None:
    async def scenario() -> None:
        events: list[EngineEvent] = []
        region = RegionNode(
            kind="map",
            body=Graph(nodes={"fail": GraphNode("test.fail_on_odd", {"value": port("item")})}),
            ports={"item": INT},
            inputs={"item": [2, 3, 4]},
            element_ports=("item",),
            outputs={"results": RegionOutput(Link("fail", "out"), mode="compact")},
        )
        with pytest.raises(ExecutionError, match="odd input failed"):
            await make_engine(events).run(Graph(nodes={"m": region}), ["m"])
        failed_iteration_started = any(
            event.kind == "region_iteration_started" and event.detail["iteration"] == 1
            for event in events
        )
        failed_iteration_finished = any(
            event.kind == "region_iteration_finished" and event.detail["iteration"] == 1
            for event in events
        )
        assert failed_iteration_started
        assert not failed_iteration_finished

    asyncio.run(scenario())


@pytest.mark.parametrize("items", [[1, 3, 5], []])
def test_compact_gather_all_absent_or_zero_iterations_is_typed_empty(items: list[int]) -> None:
    async def scenario() -> None:
        region = RegionNode(
            kind="map",
            body=Graph(nodes={"maybe": GraphNode("test.maybe_double", {"value": port("item")})}),
            ports={"item": INT},
            inputs={"item": items},
            element_ports=("item",),
            outputs={"results": RegionOutput(Link("maybe", "out"), mode="compact")},
        )
        result = await make_engine().run(Graph(nodes={"m": region}), ["m"])
        out = result.outputs["m"]["results"]
        assert out.type_id == "list<core.int>"
        assert list_children(out) == ()

    asyncio.run(scenario())


def test_compact_gather_is_legal_for_fold_and_while() -> None:
    async def scenario() -> None:
        fold = fold_region(
            body=Graph(
                nodes={
                    "acc": GraphNode("test.add_pair", {"a": port("total"), "b": port("item")}),
                    "maybe": GraphNode("test.maybe_double", {"value": port("item")}),
                }
            ),
            inputs={"item": [2, 3, 4], "total": 0},
            outputs={
                "total": RegionOutput(Link("acc", "out"), mode="state"),
                "present": RegionOutput(Link("maybe", "out"), mode="compact"),
            },
        )
        fold_result = await make_engine().run(Graph(nodes={"f": fold}), ["f"])
        assert fold_result.outputs["f"]["present"].resolve() == [4, 8]

        while_loop = while_region(
            body=Graph(
                nodes={
                    "step": GraphNode(
                        "test.counter_step",
                        {"count": port("count"), "limit": port("limit")},
                    ),
                    "maybe": GraphNode("test.maybe_double", {"value": Link("step", "next")}),
                }
            ),
            outputs={
                "count": RegionOutput(Link("step", "next"), mode="state"),
                "present": RegionOutput(Link("maybe", "out"), mode="compact"),
            },
        )
        while_result = await make_engine().run(Graph(nodes={"w": while_loop}), ["w"])
        assert while_result.outputs["w"]["present"].resolve() == [4, 8]

    asyncio.run(scenario())


def test_nested_compact_gathers_preserve_region_boundaries() -> None:
    async def scenario() -> None:
        inner = RegionNode(
            kind="map",
            body=Graph(nodes={"maybe": GraphNode("test.maybe_double", {"value": port("item")})}),
            ports={"item": INT},
            inputs={"item": port("row")},
            element_ports=("item",),
            outputs={"present": RegionOutput(Link("maybe", "out"), mode="compact")},
        )
        outer = RegionNode(
            kind="map",
            body=Graph(nodes={"inner": inner}),
            ports={"row": TypeExpr.list_of(INT)},
            inputs={"row": [[3, 2, 6], [1, 5], [8, 7]]},
            element_ports=("row",),
            outputs={"rows": RegionOutput(Link("inner", "present"), mode="compact")},
        )
        result = await make_engine().run(Graph(nodes={"o": outer}), ["o"])
        assert result.outputs["o"]["rows"].resolve() == [[4, 12], [], [16]]

    asyncio.run(scenario())


def test_absent_body_output_makes_whole_flatten_absent() -> None:
    async def scenario() -> None:
        inner = RegionNode(
            kind="map",
            body=Graph(nodes={"maybe": GraphNode("test.maybe_double", {"value": port("item")})}),
            ports={"item": INT},
            inputs={"item": port("row")},
            element_ports=("item",),
            outputs={"values": RegionOutput(Link("maybe", "out"))},
        )
        outer = RegionNode(
            kind="map",
            body=Graph(nodes={"inner": inner}),
            ports={"row": TypeExpr.list_of(INT)},
            inputs={"row": [[2], [3], [4]]},
            element_ports=("row",),
            outputs={"values": RegionOutput(Link("inner", "values"), mode="flatten")},
        )
        result = await make_engine().run(Graph(nodes={"o": outer}), ["o"])
        out = result.outputs["o"]["values"]
        assert is_absent(out)
        assert out.meta.get(ABSENT_ORIGIN_META_KEY) == "o[1]/inner[0]/maybe/out"

    asyncio.run(scenario())


# -- engine: composition -----------------------------------------------------


def test_region_output_feeds_sibling_nodes() -> None:
    async def scenario() -> None:
        engine = make_engine()
        graph = Graph(
            nodes={
                "f": fold_region(),
                "after": GraphNode("test.add_one", {"value": Link("f", "total")}),
            }
        )
        result = await engine.run(graph, ["after"])
        assert result.outputs["after"]["out"].resolve() == 11

    asyncio.run(scenario())


def test_broadcast_ports_pass_unchanged_into_every_iteration() -> None:
    async def scenario() -> None:
        engine = make_engine()
        region = RegionNode(
            kind="map",
            body=Graph(
                nodes={"add": GraphNode("test.add_pair", {"a": port("item"), "b": port("offset")})}
            ),
            ports={"item": INT, "offset": INT},
            inputs={"item": [1, 2, 3], "offset": 100},
            element_ports=("item",),
            outputs={"results": RegionOutput(Link("add", "out"))},
        )
        result = await engine.run(Graph(nodes={"m": region}), ["m"])
        children = list_children(result.outputs["m"]["results"])
        assert children is not None
        assert [c.resolve() for c in children] == [101, 102, 103]

    asyncio.run(scenario())


def test_nested_regions_execute() -> None:
    async def scenario() -> None:
        engine = make_engine()
        inner = map_region(inputs={"item": port("row")})
        outer = RegionNode(
            kind="map",
            body=Graph(nodes={"inner": inner}),
            ports={"row": TypeExpr.list_of(INT)},
            inputs={"row": [[1, 2], [3, 4, 5]]},
            element_ports=("row",),
            outputs={"rows": RegionOutput(Link("inner", "results"))},
        )
        result = await engine.run(Graph(nodes={"o": outer}), ["o"])
        rows = list_children(result.outputs["o"]["rows"])
        assert rows is not None
        assert [[c.resolve() for c in (list_children(r) or ())] for r in rows] == [
            [2, 3],
            [4, 5, 6],
        ]

    asyncio.run(scenario())


def test_occurrence_ids_compose_and_terminate_exactly_once() -> None:
    """Hazard H17: everything that multiplies execution mints new occurrence
    ids in the one closed grammar (node[i], '/'-nested), and each runtime id
    is monotonic - one start, one terminal event, nothing after terminal."""

    async def scenario() -> None:
        events: list[EngineEvent] = []
        engine = make_engine(events)
        inner = map_region(inputs={"item": port("row")})
        outer = RegionNode(
            kind="map",
            body=Graph(nodes={"inner": inner}),
            ports={"row": TypeExpr.list_of(INT)},
            inputs={"row": [[1, 2], [3, 4, 5]]},
            element_ports=("row",),
            outputs={"rows": RegionOutput(Link("inner", "results"))},
        )
        result = await engine.run(Graph(nodes={"o": outer}), ["o"])

        # Nested occurrences compose mechanically: outer[i]/inner[j]/node.
        assert sorted(result.executed) == [
            "o[0]/inner[0]/add",
            "o[0]/inner[1]/add",
            "o[1]/inner[0]/add",
            "o[1]/inner[1]/add",
            "o[1]/inner[2]/add",
        ]

        terminal = {"node_finished", "node_cached", "node_failed", "node_skipped"}
        started: dict[str, int] = {}
        terminated: dict[str, int] = {}
        for event in events:
            if event.node_id is None:
                continue
            if event.kind == "node_started":
                assert event.node_id not in terminated, (
                    f"{event.node_id} restarted after its terminal event"
                )
                started[event.node_id] = started.get(event.node_id, 0) + 1
            elif event.kind in terminal:
                terminated[event.node_id] = terminated.get(event.node_id, 0) + 1
        assert all(count == 1 for count in started.values()), started
        assert all(count == 1 for count in terminated.values()), terminated
        # Every started occurrence reached a terminal state.
        assert set(started) <= set(terminated), set(started) - set(terminated)
        for node_id in result.executed:
            assert terminated.get(node_id) == 1, f"{node_id} never terminated"

    asyncio.run(scenario())


def test_gather_of_a_port_is_the_identity_region() -> None:
    """Degenerate but legal: a region whose gather reads an element port
    directly, with an empty body - the engine runs zero body nodes."""

    async def scenario() -> None:
        engine = make_engine()
        region = RegionNode(
            kind="map",
            body=Graph(nodes={}),
            ports={"item": INT},
            inputs={"item": [4, 5]},
            element_ports=("item",),
            outputs={"echo": RegionOutput(port("item"))},
        )
        result = await engine.run(Graph(nodes={"m": region}), ["m"])
        children = list_children(result.outputs["m"]["echo"])
        assert children is not None
        assert [c.resolve() for c in children] == [4, 5]

    asyncio.run(scenario())
