from collections.abc import Mapping

import pytest
from dinkster_graph import (
    Graph,
    GraphCycleError,
    GraphNode,
    Link,
    RegionNode,
    RegionOutput,
    TypedLiteral,
    graph_from_wire,
    graph_to_wire,
    has_errors,
    migrate_node_replacements,
    plan,
    validate,
)
from dinkster_graph.residency import ResidentRoute, ResidentTopology
from dinkster_schema import (
    ComboWidget,
    Group,
    Input,
    InputFamilyMapping,
    MappingSource,
    Named,
    Node,
    NodeSchema,
    Numbered,
    Output,
    OutputFamilyMapping,
    ReplacementCase,
    ReplacementLink,
    ReplacementNode,
    ReplacementPredicate,
    ReplacementRule,
    TypeExpr,
    build_schemas,
)
from scaffold_nodes import SCAFFOLD_NODES

SCHEMAS = build_schemas(SCAFFOLD_NODES)
SCHEMAS["test.list_input"] = NodeSchema(
    node_type="test.list_input",
    body=(Input("list", TypeExpr.list_of(TypeExpr.variable("T"))),),
)


@pytest.mark.parametrize("shared", [True, False])
def test_resident_chain_requires_one_common_process(shared: bool) -> None:
    model = TypeExpr.concrete("custom.RESIDENT")
    schemas = {
        name: NodeSchema(name, body=(Input("value", model), Output("out", model)))
        for name in ("source", "middle", "sink")
    }
    topology = ResidentTopology(
        frozenset({"custom.RESIDENT"}),
        {
            "source": (ResidentRoute("left", "producer"),),
            "middle": (ResidentRoute("left", "middle"), ResidentRoute("right", "middle")),
            "sink": (ResidentRoute("left" if shared else "right", "consumer"),),
        },
    )
    graph = Graph(
        {
            "a": GraphNode("source"),
            "b": GraphNode("middle", {"value": Link("a", "out")}),
            "c": GraphNode("sink", {"value": Link("b", "out")}),
        }
    )
    problems = topology.validate(graph, schemas)
    assert bool(problems) is not shared
    if not shared:
        assert problems[0].node_id == "c"
        assert problems[0].input_id == "value"


@pytest.mark.parametrize(
    ("provider", "worker", "refused"),
    [(None, None, True), ("alternate", None, False), (None, "remote", False)],
)
def test_resident_routes_follow_default_provider_and_worker_placement(
    provider: str | None, worker: str | None, refused: bool
) -> None:
    model = TypeExpr.concrete("custom.RESIDENT")
    schemas = {
        "source": NodeSchema("source", body=(Output("out", model),)),
        "sink": NodeSchema("sink", body=(Input("value", model),)),
    }
    topology = ResidentTopology(
        frozenset({"custom.RESIDENT"}),
        {
            "source": (
                ResidentRoute("isolated", "source", provider="builtin"),
                ResidentRoute("shared", "source", provider="alternate"),
                ResidentRoute("shared", "source", "remote", False, "builtin"),
            ),
            "sink": (ResidentRoute("shared", "sink"),),
        },
    )
    graph = Graph(
        {
            "a": GraphNode(
                "source", {} if provider is None else {"provider": TypedLiteral("STRING", provider)}
            ),
            "b": GraphNode("sink", {"value": Link("a", "out")}),
        }
    )
    problems = topology.validate(graph, schemas, lambda path: worker if path == "a" else None)
    assert bool(problems) is refused
    if refused:
        assert problems[0].node_id == "b"
        assert "pack 'source'" in problems[0].message
        assert "Pack 'sink'" in problems[0].message


@pytest.mark.parametrize("lost", ["incoming", "outgoing", "region-input"])
def test_retired_node_refuses_to_silently_drop_live_connections(lost: str) -> None:
    integer = TypeExpr.concrete("INT")
    schemas = {
        "current": NodeSchema(
            "current",
            body=(Input("value", integer, default=99), Output("value", integer)),
            replacements=(
                ReplacementRule(
                    "retired", (ReplacementCase.build("current", outputs={"value": "result"}),)
                ),
            ),
        )
    }
    nodes: dict[str, GraphNode | RegionNode] = {"old": GraphNode("retired")}
    if lost == "incoming":
        nodes["source"] = GraphNode("current", {"value": 37})
        nodes["old"] = GraphNode("retired", {"operand": Link("source", "value")})
    elif lost == "outgoing":
        nodes["sink"] = GraphNode("current", {"value": Link("old", "unmapped")})
    else:
        nodes["region"] = RegionNode("map", Graph({}), inputs={"input": Link("old", "unmapped")})
    original = Graph(nodes)
    with pytest.raises(ValueError, match="connection"):
        migrate_node_replacements(original, schemas)
    assert original == Graph(nodes)


def test_replacement_guards_and_values_read_stored_slot_choices_alongside_connections() -> None:
    string = TypeExpr.concrete("STRING")
    case = ReplacementCase.build(
        "current",
        when=ReplacementPredicate.all_of(
            ReplacementPredicate.value_present("payload"),
            ReplacementPredicate.value_equals("payload", "image"),
            ReplacementPredicate.input_connected("payload"),
        ),
        inputs={
            "kind": MappingSource.from_value("payload"),
            "payload": MappingSource.link("payload"),
        },
        input_families={
            "kinds": InputFamilyMapping.copy(
                "oldKinds",
                inputs={"value": MappingSource.from_value("value")},
                source_template=("value",),
            ),
        },
    )
    schemas = {
        "current": NodeSchema(
            "current",
            body=(
                Input("kind", string),
                Input("payload", string),
                Group("kinds", value=Input("value", string), keys=Named(("a", "z"))),
            ),
            replacements=(ReplacementRule("retired", (case, ReplacementCase.build("wrong"))),),
        ),
        "wrong": NodeSchema("wrong"),
        "source": NodeSchema("source", body=(Output("value", string),)),
    }
    graph = Graph(
        {
            "old": GraphNode(
                "retired",
                {"payload": Link("source", "value")},
                slot_variants={"payload": "image", "oldKinds.z": "text", "oldKinds.a": "audio"},
            ),
            "source": GraphNode("source"),
        }
    )
    migrated = migrate_node_replacements(graph, schemas)
    node = migrated.nodes["old"]
    assert isinstance(node, GraphNode) and node.node_type == "current"
    assert node.inputs == {
        "kind": "image",
        "payload": Link("source", "value"),
        "kinds.z": "text",
        "kinds.a": "audio",
    }
    assert list(node.inputs) == ["kind", "payload", "kinds.z", "kinds.a"]


def test_retired_family_node_keeps_order_links_helpers_and_region_exports_on_reopen() -> None:
    integer = TypeExpr.concrete("INT")
    case = ReplacementCase.build(
        "current",
        when=ReplacementPredicate.value_equals("metadata", [1, {"revision": 2}]),
        nodes={"relay": ReplacementNode("relay")},
        input_families={
            "items": InputFamilyMapping.copy(
                "oldItems",
                inputs={"value": MappingSource.copy("value")},
                source_template=("value",),
                member_ids="targetPositions",
            )
        },
        output_families={"results": OutputFamilyMapping.copy("oldResults")},
        links=(ReplacementLink("results.1", "relay:input"),),
        outputs={"relay:value": "final"},
    )
    schemas = {
        "current": NodeSchema(
            "current",
            body=(
                Group("items", value=Input("item", integer), keys=Numbered("n", start=0)),
                Group("results", value=Output("result", integer), keys=Numbered("", start=0)),
            ),
            replacements=(ReplacementRule("retired", (case, ReplacementCase.build("wrong"))),),
        ),
        "wrong": NodeSchema("wrong"),
        "relay": NodeSchema("relay", body=(Input("input", integer), Output("value", integer))),
    }
    body = Graph(
        {
            "old": GraphNode(
                "retired",
                {"oldItems.second": 91, "oldItems.first": 37, "metadata": [1.0, {"revision": 2.0}]},
                output_members={"oldResults": ("1", "0")},
            ),
            "sink": GraphNode("relay", {"input": Link("old", "oldResults.0")}),
        }
    )
    graph = Graph(
        {
            "region": RegionNode(
                "map",
                body,
                outputs={"result": RegionOutput(Link("old", "final"))},
            )
        }
    )
    migrated = migrate_node_replacements(graph, schemas)
    region = migrated.nodes["region"]
    assert isinstance(region, RegionNode)
    current = region.body.nodes["old"]
    assert isinstance(current, GraphNode) and current.node_type == "current"
    assert list(current.inputs.items()) == [("items.n0", 91), ("items.n1", 37)]
    assert current.output_members == {"results": ("1", "0")}
    assert region.body.nodes["old:relay"].inputs == {"input": Link("old", "results.1")}
    assert region.body.nodes["sink"].inputs == {"input": Link("old", "results.0")}
    assert region.outputs["result"].source == Link("old:relay", "value")
    reopened = graph_from_wire(graph_to_wire(migrated))
    assert migrate_node_replacements(reopened, schemas) == migrated
    assert graph.nodes["region"] == RegionNode(
        "map",
        body,
        outputs={"result": RegionOutput(Link("old", "final"))},
    )


STRING = TypeExpr.concrete("STRING")
COMBO = TypeExpr.concrete("COMBO")


class _StringSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(node_type="test.string_source", body=(Output("value", STRING),))

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(value="text")


class _ComboRelayA(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.combo_a",
            body=(
                Input("value", COMBO, widget=ComboWidget(options=("a", "b"))),
                Output("value", COMBO),
            ),
        )

    @classmethod
    def execute(cls, *, value: str) -> Mapping[str, object]:
        return cls.outputs(value=value)


class _ComboRelayB(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.combo_b",
            body=(
                Input("value", COMBO, widget=ComboWidget(options=("x", "y"))),
                Output("value", COMBO),
            ),
        )

    @classmethod
    def execute(cls, *, value: str) -> Mapping[str, object]:
        return cls.outputs(value=value)


class _StringSink(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(node_type="test.string_sink", body=(Input("value", STRING),))


class _AcceptsComboOrString(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.combo_union",
            body=(Input("value", TypeExpr.union("COMBO", "STRING")),),
        )


class _AcceptsAnything(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.combo_wildcard",
            body=(Input("value", TypeExpr.wildcard()),),
        )


COMBO_SCHEMAS = build_schemas(
    (
        _StringSource,
        _ComboRelayA,
        _ComboRelayB,
        _StringSink,
        _AcceptsComboOrString,
        _AcceptsAnything,
    )
)


def valid_graph() -> Graph:
    return Graph(
        nodes={
            "g": GraphNode("dev.image.gradient", {"width": 8, "height": 8}),
            "i": GraphNode("dev.image.invert", {"image": Link("g", "image")}),
            "s": GraphNode("dev.image.stats", {"image": Link("i", "image")}),
        }
    )


def test_valid_graph_has_no_errors() -> None:
    assert not has_errors(validate(valid_graph(), SCHEMAS, ["s"]))


def test_plan_is_topological_and_output_driven() -> None:
    graph = Graph(
        nodes={
            **valid_graph().nodes,
            "orphan": GraphNode("dev.image.gradient", {}),
        }
    )
    order = plan(graph, ["s"])
    assert order.index("g") < order.index("i") < order.index("s")
    assert "orphan" not in order


def test_validation_codes() -> None:
    graph = Graph(
        nodes={
            "bad": GraphNode("no.such.type", {}),
            "missing": GraphNode("dev.image.invert", {}),
            "dangling": GraphNode("dev.image.invert", {"image": Link("ghost", "image")}),
            "wrongout": GraphNode("dev.image.invert", {"image": Link("missing", "nope")}),
        }
    )
    codes = {d.code for d in validate(graph, SCHEMAS, ["nowhere"])}
    assert {
        "unknown-target",
        "unknown-node-type",
        "missing-input",
        "dangling-link",
        "dangling-output",
    } <= codes


def test_type_mismatch_is_warning_not_error() -> None:
    graph = Graph(
        nodes={
            "a": GraphNode("dev.execution.value", {"value": 3}),
            "i": GraphNode("dev.image.invert", {"image": Link("a", "value")}),
        }
    )
    diags = validate(graph, SCHEMAS, ["i"])
    mismatches = [d for d in diags if d.code == "type-mismatch"]
    assert mismatches and all(d.severity == "warning" for d in mismatches)
    assert not has_errors(diags)


def test_combo_boundary_is_hard_both_directions_and_inside_lists() -> None:
    direct = Graph(
        nodes={
            "s": GraphNode("test.string_source", {}),
            "c": GraphNode("test.combo_a", {"value": Link("s", "value")}),
            "from_combo": GraphNode("test.combo_a", {"value": "a"}),
            "text": GraphNode("test.string_sink", {"value": Link("from_combo", "value")}),
        }
    )
    diags = validate(direct, COMBO_SCHEMAS, ["c", "text"])
    mismatches = [diag for diag in diags if diag.code == "type-mismatch"]
    assert len(mismatches) == 2
    assert all(diag.severity == "error" for diag in mismatches)
    assert has_errors(diags)

    class StringListSource(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.string_list_source",
                body=(Output("values", TypeExpr.list_of(STRING)),),
            )

    class ComboListSink(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.combo_list_sink",
                body=(Input("values", TypeExpr.list_of(COMBO)),),
            )

    class ComboListSource(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.combo_list_source",
                body=(Output("values", TypeExpr.list_of(COMBO)),),
            )

    class StringListSink(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.string_list_sink",
                body=(Input("values", TypeExpr.list_of(STRING)),),
            )

    class ComboScalarSink(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.combo_scalar_sink",
                body=(Input("value", COMBO),),
            )

    list_schemas = build_schemas(
        (
            StringListSource,
            ComboListSink,
            ComboListSource,
            StringListSink,
            ComboScalarSink,
            _ComboRelayA,
        )
    )
    list_graph = Graph(
        nodes={
            "strings": GraphNode("test.string_list_source", {}),
            "combos": GraphNode("test.combo_list_source", {}),
            "combo_sink": GraphNode("test.combo_list_sink", {"values": Link("strings", "values")}),
            "string_sink": GraphNode("test.string_list_sink", {"values": Link("combos", "values")}),
            "list_to_scalar": GraphNode(
                "test.combo_scalar_sink", {"value": Link("combos", "values")}
            ),
            "scalar": GraphNode("test.combo_a", {"value": "a"}),
            "scalar_to_list": GraphNode(
                "test.combo_list_sink", {"values": Link("scalar", "value")}
            ),
        }
    )
    list_mismatches = [
        diag
        for diag in validate(list_graph, list_schemas, ["combo_sink", "string_sink"])
        if diag.code == "type-mismatch"
    ]
    assert len(list_mismatches) == 2
    assert all(diag.severity == "error" for diag in list_mismatches)
    cardinality = validate(list_graph, list_schemas, ["list_to_scalar", "scalar_to_list"])
    assert {diag.code for diag in cardinality if "-into-" in diag.code} == {
        "list-into-scalar",
        "scalar-into-list",
    }


@pytest.mark.parametrize(
    ("type_id", "value", "valid"),
    [
        ("INT", True, False),
        ("INT", 2.5, False),
        ("INT", -3, True),
        ("FLOAT", "normal", False),
        ("FLOAT", False, False),
        ("FLOAT", 3, True),
        ("FLOAT", 0.375, True),
        ("STRING", 3, False),
        ("STRING", "", True),
        ("BOOLEAN", 1, False),
        ("BOOLEAN", False, True),
        ("COMBO", {"value": "a"}, False),
        ("COMBO", True, True),
        ("FLOAT", None, False),
    ],
)
@pytest.mark.parametrize("stamped", [False, True])
def test_primitive_literal_payload_matches_declared_type(
    type_id: str, value: object, valid: bool, stamped: bool
) -> None:
    schema = NodeSchema(
        node_type="test.literal", body=(Input("value", TypeExpr.concrete(type_id)),)
    )
    graph = Graph(
        nodes={
            "n": GraphNode(
                "test.literal", {"value": TypedLiteral(type_id, value) if stamped else value}
            )
        }
    )
    diagnostics = validate(graph, {"test.literal": schema}, ["n"])
    assert has_errors(diagnostics) is not valid
    if not valid:
        assert any(
            d.code == "literal-type" and d.node_id == "n" and d.input_id == "value"
            for d in diagnostics
        )


def test_nested_list_and_stamp_cannot_hide_invalid_primitive_literal() -> None:
    schema = NodeSchema(
        node_type="test.literal",
        body=(Input("value", TypeExpr.list_of(TypeExpr.list_of(TypeExpr.concrete("INT")))),),
    )
    for value in ([[2, True]], TypedLiteral("list<list<INT>>", [[2, True]])):
        graph = Graph(nodes={"n": GraphNode("test.literal", {"value": value})})
        assert has_errors(validate(graph, {"test.literal": schema}, ["n"]))
    assert not has_errors(
        validate(
            Graph(nodes={"n": GraphNode("test.literal", {"value": [[2, -3], []]})}),
            {"test.literal": schema},
            ["n"],
        )
    )
    scalar = NodeSchema(node_type="test.scalar", body=(Input("value", TypeExpr.concrete("FLOAT")),))
    assert has_errors(
        validate(
            Graph(
                nodes={"n": GraphNode("test.scalar", {"value": TypedLiteral("STRING", "normal")})}
            ),
            {"test.scalar": scalar},
            ["n"],
        )
    )


@pytest.mark.parametrize(
    ("extra", "expected_code"),
    (
        (TypedLiteral("INT", 7), None),
        (TypedLiteral("STRING", "seven"), None),
        (7, "literal-on-nonconcrete"),
        (TypedLiteral("missing.type", 7), "typed-literal-unknown-type"),
        (TypedLiteral("list<INT>", 7), "literal-shape"),
    ),
)
@pytest.mark.parametrize("in_body", (False, True))
def test_authored_inputs_use_ordinary_typed_literal_admission(
    extra: object, expected_code: str | None, in_body: bool
) -> None:
    declared = (Input("choice", STRING),)
    wildcard = TypeExpr.wildcard()
    schemas = {
        "test.authored": NodeSchema("test.authored", body=declared, additional_inputs=wildcard)
        if in_body
        else NodeSchema("test.authored", body=(*declared,), additional_inputs=wildcard)
    }
    graph = Graph(
        nodes={
            "authored": GraphNode(
                "test.authored",
                {"choice": "outside-menu", "2": extra, "9": TypedLiteral("STRING", "nine")},
            )
        }
    )
    diagnostics = validate(graph, schemas, ["authored"], known_types={"INT", "STRING"})
    if expected_code is None:
        assert diagnostics == []
    else:
        assert any(d.code == expected_code and d.severity == "error" for d in diagnostics)


def test_typed_literal_cardinality_is_structural() -> None:
    # A list stamp into a definitely-scalar input (union) and a scalar stamp
    # into a definitely-list input are errors, mirroring the link path.
    graph = Graph(
        nodes={
            "s": GraphNode(
                "dev.gallery.sockets",
                {"union2": TypedLiteral("list<INT>", [1])},
            ),
            "l": GraphNode("test.list_input", {"list": TypedLiteral("INT", 7)}),
        }
    )
    codes = {d.code for d in validate(graph, SCHEMAS, ["s", "l"], known_types={"INT"})}
    assert "list-into-scalar" in codes
    assert "scalar-into-list" in codes


def test_cycle_detected() -> None:
    graph = Graph(
        nodes={
            "x": GraphNode("dev.image.invert", {"image": Link("y", "image")}),
            "y": GraphNode("dev.image.invert", {"image": Link("x", "image")}),
        }
    )
    assert any(d.code == "cycle" for d in validate(graph, SCHEMAS, ["x"]))
    with pytest.raises(GraphCycleError):
        plan(graph, ["x"])


@pytest.mark.parametrize(
    ("nodes", "targets", "expected"),
    [
        ({"end": GraphNode("dinkster.utilities.end_loop")}, ["end"], "loop.unmatchedEnd"),
        ({"start": GraphNode("dinkster.utilities.start_loop")}, ["start"], "loop.unmatchedStart"),
        (
            {
                "a": GraphNode("dinkster.utilities.start_loop"),
                "b": GraphNode("dinkster.utilities.start_loop"),
                "end": GraphNode(
                    "dinkster.utilities.end_loop",
                    {"x": Link("a", "value"), "y": Link("b", "value")},
                ),
            },
            ["end"],
            "loop.ambiguousNesting",
        ),
        (
            {
                "start": GraphNode("dinkster.utilities.start_loop"),
                "end": GraphNode("dinkster.utilities.end_loop", {"x": Link("start", "value")}),
                "escape": GraphNode("relay", {"x": Link("start", "value")}),
            },
            ["end", "escape"],
            "loop.bodyEscape",
        ),
        (
            {
                "start": GraphNode("dinkster.utilities.start_loop"),
                "body": GraphNode("relay", {"x": Link("start", "value")}),
                "end": GraphNode(
                    "dinkster.utilities.end_loop", {"accumulate": Link("body", "value")}
                ),
            },
            ["end"],
            "loop.accumulateFromBody",
        ),
    ],
)
@pytest.mark.parametrize("inside_region", [False, True])
def test_loop_boundary_diagnostics_are_structural(nodes, targets, expected, inside_region) -> None:
    schemas = {
        kind: NodeSchema(
            kind,
            body=(
                *tuple(
                    Input(name, TypeExpr.wildcard(), required=False)
                    for name in ("x", "y", "accumulate")
                ),
                Output("value", TypeExpr.concrete("INT")),
            ),
        )
        for kind in ("dinkster.utilities.start_loop", "dinkster.utilities.end_loop", "relay")
    }
    graph = Graph(nodes)
    if inside_region:
        graph = Graph(
            {
                "region": RegionNode(
                    "fold",
                    graph,
                    ports={"item": TypeExpr.concrete("INT")},
                    inputs={"item": [1]},
                    element_ports=("item",),
                    outputs={
                        target: RegionOutput(Link(target, "value"), "last") for target in targets
                    },
                )
            }
        )
        targets = ["region"]
    diagnostics = validate(graph, schemas, targets)
    assert [(item.code, item.severity) for item in diagnostics] == [(expected, "error")]
    if inside_region:
        assert diagnostics[0].node_id is not None
        assert diagnostics[0].node_id.startswith("region/")


def test_nested_loop_pairing_consumes_inner_ends_before_outer_ends() -> None:
    from dinkster_graph.validate import LoopPair, validate_loops

    graph = Graph(
        {
            "outer": GraphNode("dinkster.utilities.start_loop"),
            "first": GraphNode("dinkster.utilities.start_loop", {"x": Link("outer", "value")}),
            "first_end": GraphNode("dinkster.utilities.end_loop", {"x": Link("first", "value")}),
            "second": GraphNode("dinkster.utilities.start_loop", {"x": Link("first_end", "value")}),
            "second_end": GraphNode("dinkster.utilities.end_loop", {"x": Link("second", "value")}),
            "outer_end": GraphNode(
                "dinkster.utilities.end_loop", {"x": Link("second_end", "value")}
            ),
        }
    )
    pairs, diagnostics = validate_loops(graph, ["outer_end"])
    assert diagnostics == []
    assert pairs == {
        "first": LoopPair("first_end", ()),
        "second": LoopPair("second_end", ()),
        "outer": LoopPair("outer_end", ("first", "first_end", "second", "second_end")),
    }
    rewritten = Graph(
        {
            **graph.nodes,
            "outer_end": GraphNode("dinkster.utilities.end_loop"),
            "first_end": GraphNode("dinkster.utilities.end_loop"),
            "second_end": GraphNode("dinkster.utilities.end_loop"),
            "new_start": GraphNode("dinkster.utilities.start_loop"),
            "new_end": GraphNode("dinkster.utilities.end_loop", {"x": Link("new_start", "value")}),
        }
    )
    new_pairs, diagnostics = validate_loops(rewritten, ["outer_end", "new_end"], pairs)
    assert diagnostics == []
    assert new_pairs == {"new_start": LoopPair("new_end", ())}
    _, diagnostics = validate_loops(
        Graph({**rewritten.nodes, "orphan": GraphNode("dinkster.utilities.end_loop")}),
        ["orphan"],
        pairs,
    )
    assert [diag.code for diag in diagnostics] == ["loop.unmatchedEnd"]
