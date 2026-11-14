from dinkster_graph import (
    Graph,
    GraphNode,
    Link,
    lower_selectors,
)
from dinkster_schema import (
    InputSpec,
    NodeSchema,
    OutputSpec,
    SelectorSpec,
    TypeExpr,
)
from dinkster_server.preflight import graph_asset_names

T = TypeExpr.concrete("INT")
SWITCH = NodeSchema(
    "test.switch",
    inputs=(
        InputSpec("switch", TypeExpr.concrete("BOOLEAN")),
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
