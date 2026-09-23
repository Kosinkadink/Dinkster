from dataclasses import replace

from dinkster_graph import (
    Graph,
    GraphNode,
    Link,
    RegionNode,
    TypedLiteral,
    lower_selectors,
    validate,
)
from dinkster_schema import (
    InputSpec,
    NodeSchema,
    OutputSpec,
    SelectorSpec,
    TypeExpr,
    schema_from_wire,
    schema_signature,
    schema_to_wire,
)
from dinkster_server.preflight import graph_asset_names

T = TypeExpr.concrete("core.int")
SWITCH = NodeSchema(
    "test.switch",
    inputs=(
        InputSpec("switch", TypeExpr.concrete("core.boolean")),
        InputSpec("off", T),
        InputSpec("on", T),
    ),
    outputs=(OutputSpec("out", T),),
    selector=SelectorSpec("switch", {"false": "off", "true": "on"}),
)
SOURCE = NodeSchema("test.source", outputs=(OutputSpec("out", T),))
SINK = NodeSchema("test.sink", inputs=(InputSpec("value", T),))
ANY = NodeSchema("test.any", inputs=(InputSpec("value", TypeExpr.wildcard()),))
SCHEMAS = {s.node_type: s for s in (SWITCH, SOURCE, SINK, ANY)}


def test_selector_lowering_matrix() -> None:
    graph = Graph(
        {
            "off": GraphNode(SOURCE.node_type),
            "on": GraphNode(SOURCE.node_type),
            "orphan": GraphNode(SOURCE.node_type),
            "switch": GraphNode(
                SWITCH.node_type,
                {"switch": True, "off": Link("off", "out"), "on": Link("on", "out")},
            ),
            "sink": GraphNode(SINK.node_type, {"value": Link("switch", "out")}),
        }
    )
    result = lower_selectors(graph, ["sink"], SCHEMAS)
    assert set(result.graph.nodes) == {"on", "orphan", "sink"}
    assert result.graph.nodes["sink"].inputs["value"] == Link("on", "out")

    for value in (7, TypedLiteral("core.int", 8)):
        graph = Graph(
            {
                "switch": GraphNode(SWITCH.node_type, {"switch": True, "on": value}),
                "sink": GraphNode(SINK.node_type, {"value": Link("switch", "out")}),
            }
        )
        lowered = lower_selectors(graph, ["sink"], SCHEMAS)
        assert lowered.graph.nodes["sink"].inputs["value"] is value

    bad_inline = Graph(
        {
            "switch": GraphNode(SWITCH.node_type, {"switch": True, "on": 7}),
            "sink": GraphNode(ANY.node_type, {"value": Link("switch", "out")}),
        }
    )
    lowered = lower_selectors(bad_inline, ["sink"], SCHEMAS)
    assert any(
        d.code == "literal-on-nonconcrete"
        for d in validate(lowered.graph, SCHEMAS, lowered.targets)
    )


def test_selector_chains_regions_gc_and_refusals() -> None:
    chain = Graph(
        {
            "source": GraphNode(SOURCE.node_type),
            "a": GraphNode(SWITCH.node_type, {"switch": True, "on": Link("source", "out")}),
            "b": GraphNode(SWITCH.node_type, {"switch": False, "off": Link("a", "out")}),
            "sink": GraphNode(SINK.node_type, {"value": Link("b", "out")}),
        }
    )
    result = lower_selectors(chain, ["sink"], SCHEMAS)
    assert set(result.graph.nodes) == {"source", "sink"}

    temporary = Graph(
        {
            "candidate": GraphNode(SOURCE.node_type),
            "a": GraphNode(
                SWITCH.node_type,
                {"switch": True, "off": Link("candidate", "out"), "on": 1},
            ),
            "b": GraphNode(
                SWITCH.node_type,
                {"switch": True, "off": Link("candidate", "out"), "on": 2},
            ),
        }
    )
    assert lower_selectors(temporary, [], SCHEMAS).graph.nodes == {}

    shared = Graph(
        {
            "source": GraphNode(SOURCE.node_type),
            "switch": GraphNode(
                SWITCH.node_type,
                {"switch": True, "off": Link("source", "out"), "on": 1},
            ),
            "other": GraphNode(SINK.node_type, {"value": Link("source", "out")}),
            "sink": GraphNode(SINK.node_type, {"value": Link("switch", "out")}),
        }
    )
    assert "source" in lower_selectors(shared, ["sink"], SCHEMAS).graph.nodes

    cases = [
        (
            Graph({"s": GraphNode(SWITCH.node_type, {})}),
            ["s"],
            "prompt.selector_is_target",
        ),
        (
            Graph({"s": GraphNode(SWITCH.node_type, {"switch": 1})}),
            [],
            "prompt.bad_selector_value",
        ),
        (
            Graph({"s": GraphNode(SWITCH.node_type, {"switch": True})}),
            [],
            "prompt.missing_branch",
        ),
    ]
    for graph, targets, code in cases:
        assert lower_selectors(graph, targets, SCHEMAS).problems[0].code == code

    optional_switch = replace(
        SWITCH,
        inputs=tuple(
            replace(item, required=False) if item.id == "on" else item
            for item in SWITCH.inputs
        ),
    )
    missing_optional = Graph(
        {
            "s": GraphNode(optional_switch.node_type, {"switch": True}),
            "sink": GraphNode(SINK.node_type, {"value": Link("s", "out")}),
        }
    )
    optional_result = lower_selectors(
        missing_optional,
        ["sink"],
        {**SCHEMAS, optional_switch.node_type: optional_switch},
    )
    assert optional_result.problems == ()
    assert optional_result.graph == missing_optional

    computed = Graph(
        {
            "x": GraphNode(SOURCE.node_type),
            "s": GraphNode(
                SWITCH.node_type,
                {"switch": Link("x", "out"), "off": 0, "on": 1},
            ),
        }
    )
    runtime = lower_selectors(computed, ["s"], SCHEMAS)
    assert runtime.problems == ()
    assert runtime.graph == computed

    body = Graph({"s": GraphNode(SWITCH.node_type, {"switch": True, "on": 1})})
    region_graph = Graph({"r": RegionNode(kind="map", body=body)})
    region_result = lower_selectors(region_graph, [], SCHEMAS)
    assert region_result.problems == ()
    assert region_result.graph == region_graph
    region = RegionNode(kind="map", body=Graph({}), inputs={"p": Link("s", "out")})
    top = Graph({"s": GraphNode(SWITCH.node_type, {"switch": True, "on": 1}), "r": region})
    assert lower_selectors(top, ["r"], SCHEMAS).graph.nodes["r"].inputs["p"] == 1


def test_selector_model_validation() -> None:
    plain = replace(SWITCH, selector=None)
    assert "selector" not in schema_to_wire(plain)
    assert schema_to_wire(replace(plain, selector=None)) == schema_to_wire(plain)
    assert schema_signature(replace(plain, selector=None)) == schema_signature(plain)
    assert schema_from_wire(schema_to_wire(SWITCH)) == SWITCH
    assert schema_signature(SWITCH) != schema_signature(plain)
    assert replace(SWITCH, selector=None).selector is None
    for selector in (
        SelectorSpec("switch", {"false": "off", "true": "switch"}),
        SelectorSpec("switch", {"false": "off", "true": "missing"}),
    ):
        try:
            replace(SWITCH, selector=selector)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid selector accepted")
    for invalid in (
        (1, {"false": "off", "true": "on"}),
        ("switch", {"false": 1, "true": "on"}),
    ):
        try:
            SelectorSpec(invalid[0], invalid[1])  # type: ignore[arg-type]
        except ValueError:
            pass
        else:
            raise AssertionError("non-string selector id accepted")


def test_inactive_branch_assets_do_not_reach_submission_preflight() -> None:
    asset = {"digest": "blake3:" + "1" * 64, "name": "inactive.bin"}
    graph = Graph(
        {
            "inactive": GraphNode(SINK.node_type, {"value": asset}),
            "switch": GraphNode(
                SWITCH.node_type,
                {"switch": True, "off": Link("inactive", "out"), "on": 1},
            ),
            "sink": GraphNode(SINK.node_type, {"value": Link("switch", "out")}),
        }
    )
    lowered = lower_selectors(graph, ["sink"], SCHEMAS)
    assert graph_asset_names(graph)
    assert graph_asset_names(lowered.graph) == {}
