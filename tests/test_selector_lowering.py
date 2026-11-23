from collections.abc import Mapping
from dataclasses import replace

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, ExecutionError, GraphValidationError
from dinkster_graph import (
    Graph,
    GraphNode,
    Link,
    TypedLiteral,
    lower_selectors,
)
from dinkster_schema import (
    Group,
    Input,
    Node,
    NodeSchema,
    Numbered,
    Output,
    SelectorSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_schema.nest import Item, Members
from dinkster_server.preflight import graph_asset_names
from dinkster_values import TypeRegistry, is_absent, register_core_types
from dinkster_workers import InProcessWorker

from tests.async_scenarios import run_scenario

T = TypeExpr.concrete("INT")
PICK = TypeExpr.union("INT", "BOOLEAN")


class Switch(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "test.switch",
            body=(
                Input("pick", PICK, required=False),
                Group(
                    "branch",
                    value=Input("branch", T, required=False, lazy=True),
                    keys=Numbered("branch"),
                    grows="button",
                    min_members=1,
                    reorderable=True,
                ),
                Output("out", T),
            ),
            selector=SelectorSpec("pick", "branch"),
        )

    @classmethod
    def execute(cls, *, pick: object, branch: Members[Item]) -> Mapping[str, object]:
        raise AssertionError("the engine resolves a selector without invoking it")


def test_a_switch_whose_branch_group_may_be_emptied_is_refused() -> None:
    schema = Switch.define_schema()
    body = tuple(
        replace(row, min_members=0) if isinstance(row, Group) else row for row in schema.body
    )
    with pytest.raises(ValueError, match="a selector group keeps at least one branch"):
        replace(schema, body=body)


class Source(Node):
    ran: list[int] = []

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema("test.source", body=(Input("value", T), Output("out", T)))

    @classmethod
    def execute(cls, *, value: int) -> Mapping[str, object]:
        cls.ran.append(value)
        return cls.outputs(out=value)


class Position(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema("test.position", body=(Input("value", T), Output("out", T)))

    @classmethod
    def execute(cls, *, value: int) -> Mapping[str, object]:
        return cls.outputs(out=value)


class Flag(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        flag = TypeExpr.concrete("BOOLEAN")
        return NodeSchema("test.flag", body=(Input("value", flag), Output("out", flag)))

    @classmethod
    def execute(cls, *, value: bool) -> Mapping[str, object]:
        return cls.outputs(out=value)


TWO = ("branch2", "branch1")
SINK = NodeSchema("test.sink", body=(Input("value", T),))
NODES = (Switch, Source, Position, Flag)
SCHEMAS = {**build_schemas(NODES), SINK.node_type: SINK}


def switch_graph(
    pick: object, members: tuple[str, ...] = ("branch2", "branch1"), **values: object
) -> Graph:
    """Two sources, 11 and 37, on members stored in the order ``members``:
    branch1 is wired from 37 and branch2 from 11, so a pick read by position
    and one read by key give different answers."""
    sources = {"branch1": 37, "branch2": 11}
    inputs: dict[str, object] = {
        f"branch.{key}": Link(f"source{sources[key]}", "out")
        for key in members
        if key in sources and key not in values
    }
    inputs |= {f"branch.{key}": value for key, value in values.items()}
    if pick is not None:
        inputs["pick"] = pick
    return Graph(
        {
            "source11": GraphNode("test.source", {"value": 11}),
            "source37": GraphNode("test.source", {"value": 37}),
            "switch": GraphNode("test.switch", inputs, output_members={"branch": members}),
            "sink": GraphNode("test.sink", {"value": Link("switch", "out")}),
        }
    )


@pytest.mark.parametrize(
    ("pick", "source", "pruned"),
    [
        ("branch1", "source37", "source11"),  # a stored key names a member, not a position
        ("branch3", "source11", "source37"),  # a removed member: the first in stored order
        (None, "source11", "source37"),
        (1, "source37", "source11"),  # a known position counts the stored order
        (False, "source11", "source37"),
    ],
)
def test_stored_pick_rewires_to_the_picked_member_and_prunes_the_rest(
    pick: object, source: str, pruned: str
) -> None:
    lowered = lower_selectors(switch_graph(pick), ["sink"], SCHEMAS)
    assert lowered.problems == ()
    assert set(lowered.graph.nodes) == {source, "sink"}
    assert lowered.graph.nodes["sink"].inputs == {"value": Link(source, "out")}
    assert pruned not in lowered.graph.nodes


@pytest.mark.parametrize(
    ("pick", "members", "message"),
    [
        (2, TWO, "Switch received position 2. There are 2 branches; use 0 or 1."),
        (-1, TWO, "Switch received position -1. There are 2 branches; use 0 or 1."),
        (
            True,
            ("branch1", "branch2", "branch3"),
            "Switch received true, which picks between exactly two branches. "
            "There are 3 branches; wire a position from 0 to 2 instead.",
        ),
    ],
)
def test_stored_pick_outside_the_members_is_refused(
    pick: object, members: tuple[str, ...], message: str
) -> None:
    lowered = lower_selectors(switch_graph(pick, members), ["sink"], SCHEMAS)
    assert [(p.code, p.message, p.input_id) for p in lowered.problems] == [
        ("prompt.bad_selector_value", message, "pick")
    ]


def test_picked_member_with_nothing_plugged_in_keeps_an_empty_switch() -> None:
    graph = switch_graph("branch3", ("branch1", "branch3", "branch2"))
    lowered = lower_selectors(graph, ["sink"], SCHEMAS)
    assert lowered.problems == ()
    assert set(lowered.graph.nodes) == {"switch", "sink"}
    assert lowered.graph.nodes["switch"] == GraphNode(
        "test.switch", {"pick": TypedLiteral("INT", 0)}, output_members={"branch": ("branch3",)}
    )
    # Queue admission lowers again and refuses any rewrite.
    assert lower_selectors(lowered.graph, ["sink"], SCHEMAS).graph == lowered.graph


def test_wired_pick_keeps_every_member_for_the_engine() -> None:
    graph = switch_graph(Link("position", "out"))
    position = GraphNode("test.position", {"value": 1})
    graph = Graph({**graph.nodes, "position": position})
    assert lower_selectors(graph, ["sink"], SCHEMAS).graph == graph


def test_inactive_branch_assets_do_not_reach_submission_preflight() -> None:
    asset = {"digest": "blake3:" + "1" * 64, "name": "inactive.bin"}
    graph = Graph(
        {
            "inactive": GraphNode(SINK.node_type, {"value": asset}),
            "switch": GraphNode(
                "test.switch",
                {"pick": "branch2", "branch.branch1": Link("inactive", "out"), "branch.branch2": 1},
            ),
            "sink": GraphNode(SINK.node_type, {"value": Link("switch", "out")}),
        }
    )
    lowered = lower_selectors(graph, ["sink"], SCHEMAS)
    assert graph_asset_names(graph)
    assert graph_asset_names(lowered.graph) == {}


def test_selector_survives_the_wire() -> None:
    schema = SCHEMAS["test.switch"]
    assert schema_from_wire(schema_to_wire(schema)).selector == SelectorSpec("pick", "branch")


def run_switch(pick: object, members: tuple[str, ...], **values: object):
    registry = TypeRegistry()
    register_core_types(registry)
    engine = Engine(
        schemas=build_schemas(NODES),
        registry=registry,
        worker=InProcessWorker(build_node_types(NODES), registry),
        cache=MemoryLRUCache(),
    )
    graph = switch_graph(Link("position", "out"), members, **values)
    nodes = {k: v for k, v in graph.nodes.items() if k != "sink"}
    nodes["position"] = GraphNode(
        "test.flag" if type(pick) is bool else "test.position", {"value": pick}
    )
    Source.ran.clear()
    return run_scenario(engine.run(Graph(nodes), ["switch"]))


@pytest.mark.parametrize(("pick", "value"), [(0, 11), (1, 37), (False, 11), (True, 37)])
def test_wired_pick_runs_only_the_member_at_its_position(pick: object, value: int) -> None:
    result = run_switch(pick, ("branch2", "branch1"))
    assert result.outputs["switch"]["out"].resolve() == value
    assert Source.ran == [value]


@pytest.mark.parametrize(
    ("pick", "members", "message"),
    [
        (2, TWO, "Switch received position 2. There are 2 branches; use 0 or 1."),
        (-1, TWO, "Switch received position -1. There are 2 branches; use 0 or 1."),
        (True, ("branch1", "branch2", "branch3"), "exactly two branches. There are 3 branches"),
    ],
)
def test_wired_pick_outside_the_members_fails_the_switch(
    pick: object, members: tuple[str, ...], message: str
) -> None:
    with pytest.raises(ExecutionError, match=message.replace(".", r"\.")):
        run_switch(pick, members)
    assert Source.ran == []


def test_wired_pick_of_an_empty_member_passes_nothing() -> None:
    result = run_switch(2, ("branch1", "branch2", "branch3"))
    assert is_absent(result.outputs["switch"]["out"])
    assert Source.ran == []


TEXT_PICK = TypeExpr.union("INT", "BOOLEAN", "STRING", "COMBO")
NAMES = TypeExpr.list_of(TypeExpr.concrete("STRING"))


def named_switch_schema(pick: TypeExpr = TEXT_PICK, names: Input | None = None) -> NodeSchema:
    return NodeSchema(
        "test.named_switch",
        body=(
            Input("pick", pick, required=False),
            names or Input("names", NAMES, required=False, hidden=True),
            Group(
                "branch",
                value=Input("branch", T, required=False, lazy=True),
                keys=Numbered("branch"),
                grows="button",
                min_members=1,
            ),
            Output("out", T),
        ),
        selector=SelectorSpec("pick", "branch", names="names"),
    )


class NamedSwitch(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return named_switch_schema()

    @classmethod
    def execute(cls, **_inputs: object) -> Mapping[str, object]:
        raise AssertionError("the engine resolves a selector without invoking it")


class AnySwitch(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        generic = TypeExpr.variable("T")
        return NodeSchema(
            "test.any_switch",
            body=(
                Input("pick", PICK, required=False),
                Group(
                    "branch",
                    value=Input("branch", generic, required=False, lazy=True),
                    keys=Numbered("branch"),
                    grows="button",
                    min_members=1,
                ),
                Output("out", generic),
            ),
            selector=SelectorSpec("pick", "branch"),
        )

    @classmethod
    def execute(cls, *, pick: object, branch: Members[Item]) -> Mapping[str, object]:
        raise AssertionError("the engine resolves a selector without invoking it")


class Text(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        text = TypeExpr.concrete("STRING")
        return NodeSchema("test.text", body=(Input("value", text), Output("out", text)))

    @classmethod
    def execute(cls, *, value: str) -> Mapping[str, object]:
        return cls.outputs(out=value)


NAMED_NODES = (NamedSwitch, Source, Text)


def run_named_switch(text: str, names: list[str]):
    """Members stored as branch2 (source 11) then branch1 (source 37)."""
    registry = TypeRegistry()
    register_core_types(registry)
    engine = Engine(
        schemas=build_schemas(NAMED_NODES),
        registry=registry,
        worker=InProcessWorker(build_node_types(NAMED_NODES), registry),
        cache=MemoryLRUCache(),
    )
    switch = GraphNode(
        "test.named_switch",
        {
            "pick": Link("text", "out"),
            "names": names,
            "branch.branch2": Link("source11", "out"),
            "branch.branch1": Link("source37", "out"),
        },
        output_members={"branch": ("branch2", "branch1")},
    )
    graph = Graph(
        {
            "source11": GraphNode("test.source", {"value": 11}),
            "source37": GraphNode("test.source", {"value": 37}),
            "text": GraphNode("test.text", {"value": text}),
            "switch": switch,
        }
    )
    Source.ran.clear()
    return run_scenario(engine.run(graph, ["switch"]))


@pytest.mark.parametrize(
    ("text", "names", "value"),
    [
        ("Detailed", ["Quick", "Detailed"], 37),  # a name picks by the names' member order
        ("branch2", ["Quick", "Detailed"], 11),  # a member key when no name matches
        ("branch1", ["branch1", "Quick"], 11),  # a name wins over a member key
    ],
)
def test_wired_text_picks_by_name_then_key(text: str, names: list[str], value: int) -> None:
    result = run_named_switch(text, names)
    assert result.outputs["switch"]["out"].resolve() == value
    assert Source.ran == [value]


@pytest.mark.parametrize(
    ("text", "names", "message"),
    [
        (
            "Fast",
            ["Quick", "Detailed"],
            'Switch received "Fast". The branches are Quick and Detailed.',
        ),
        ("quick", ["Quick", "Detailed"], 'Switch received "quick".'),
        ("Quick", ["Quick"], "Switch received 1 branch names for 2 branches."),
    ],
)
def test_wired_text_naming_no_branch_fails_the_switch(
    text: str, names: list[str], message: str
) -> None:
    with pytest.raises(ExecutionError, match=message.replace(".", r"\.")):
        run_named_switch(text, names)
    assert Source.ran == []


def test_stored_text_keeps_the_first_branch_rule() -> None:
    schemas = {**build_schemas(NAMED_NODES), SINK.node_type: SINK}
    graph = Graph(
        {
            "source11": GraphNode("test.source", {"value": 11}),
            "source37": GraphNode("test.source", {"value": 37}),
            "switch": GraphNode(
                "test.named_switch",
                {
                    "pick": "Detailed",
                    "branch.branch2": Link("source11", "out"),
                    "branch.branch1": Link("source37", "out"),
                },
                output_members={"branch": ("branch2", "branch1")},
            ),
            "sink": GraphNode("test.sink", {"value": Link("switch", "out")}),
        }
    )
    lowered = lower_selectors(graph, ["sink"], schemas)
    assert lowered.problems == ()
    assert lowered.graph.nodes["sink"].inputs == {"value": Link("source11", "out")}


def test_names_survive_the_wire() -> None:
    schema = named_switch_schema()
    assert schema_from_wire(schema_to_wire(schema)).selector == SelectorSpec(
        "pick", "branch", names="names"
    )
    assert schema_to_wire(SCHEMAS["test.switch"])["selector"] == {
        "input": "pick",
        "group": "branch",
    }


@pytest.mark.parametrize(
    ("pick", "names", "message"),
    [
        (PICK, None, "selector input must name a BOOLEAN | COMBO | INT | STRING union input"),
        (TEXT_PICK, Input("names", NAMES, required=False), "selector names 'names' must name"),
        (TEXT_PICK, Input("names", NAMES, hidden=True), "selector names 'names' must name"),
        (
            TEXT_PICK,
            Input("names", TypeExpr.concrete("STRING"), required=False, hidden=True),
            "selector names 'names' must name",
        ),
    ],
)
def test_a_names_selector_declares_its_pick_and_names_input(
    pick: TypeExpr, names: Input | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message.replace("|", r"\|")):
        named_switch_schema(pick, names)


def test_a_text_pick_needs_names() -> None:
    schema = Switch.define_schema()
    body = tuple(
        replace(row, type=TEXT_PICK) if isinstance(row, Input) and row.id == "pick" else row
        for row in schema.body
    )
    with pytest.raises(ValueError, match=r"selector input must name a BOOLEAN \| INT union"):
        replace(schema, body=body)


@pytest.mark.parametrize("wired", [True, False])
def test_a_switch_with_branches_of_two_types_runs_only_with_a_stored_pick(wired: bool) -> None:
    nodes = (AnySwitch, Source, Text, Position)
    schemas = build_schemas(nodes)
    registry = TypeRegistry()
    register_core_types(registry)
    engine = Engine(
        schemas=schemas,
        registry=registry,
        worker=InProcessWorker(build_node_types(nodes), registry),
        cache=MemoryLRUCache(),
    )
    graph = Graph(
        {
            "number": GraphNode("test.source", {"value": 11}),
            "text": GraphNode("test.text", {"value": "a"}),
            "position": GraphNode("test.position", {"value": 1}),
            "switch": GraphNode(
                "test.any_switch",
                {
                    "pick": Link("position", "out") if wired else "branch2",
                    "branch.branch1": Link("number", "out"),
                    "branch.branch2": Link("text", "out"),
                },
                output_members={"branch": ("branch1", "branch2")},
            ),
            "shown": GraphNode("test.text", {"value": Link("switch", "out")}),
        }
    )
    lowered = lower_selectors(graph, ["shown"], schemas)
    assert lowered.problems == ()
    Source.ran.clear()
    if not wired:
        result = run_scenario(engine.run(lowered.graph, ["shown"]))
        assert result.outputs["shown"]["out"].resolve() == "a"
        return
    with pytest.raises(GraphValidationError) as error:
        run_scenario(engine.run(lowered.graph, ["shown"]))
    assert [
        (d.node_id, d.message)
        for d in error.value.diagnostics
        if d.code == "type-variable-conflict"
    ] == [("switch", "These inputs must be the same type: branch1 is INT, branch2 is STRING.")]
    assert Source.ran == []
