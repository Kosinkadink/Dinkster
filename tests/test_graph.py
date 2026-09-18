from collections.abc import Mapping

import pytest
from dinkster_graph import (
    Graph,
    GraphCycleError,
    GraphNode,
    Link,
    TypedLiteral,
    has_errors,
    plan,
    top_level_node_id,
    validate,
)
from dinkster_schema import (
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_schemas,
)
from scaffold_nodes import SCAFFOLD_NODES

SCHEMAS = build_schemas(SCAFFOLD_NODES)

STRING = TypeExpr.concrete("core.string")
COMBO = TypeExpr.concrete("core.combo")


@pytest.mark.parametrize(
    ("runtime_node_id", "expected"),
    (
        ("node", "node"),
        ("region[2]/body", "region"),
        ("outer[1]/inner[3]/body", "outer"),
    ),
)
def test_runtime_node_paths_recover_top_level_document_id(
    runtime_node_id: str, expected: str
) -> None:
    assert top_level_node_id(runtime_node_id) == expected


class _StringSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(node_type="test.string_source", outputs=(OutputSpec("value", STRING),))

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(value="text")


class _ComboRelayA(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.combo_a",
            inputs=(InputSpec("value", COMBO, widget=ComboWidget(options=("a", "b"))),),
            outputs=(OutputSpec("value", COMBO),),
        )

    @classmethod
    def execute(cls, *, value: str) -> Mapping[str, object]:
        return cls.outputs(value=value)


class _ComboRelayB(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.combo_b",
            inputs=(InputSpec("value", COMBO, widget=ComboWidget(options=("x", "y"))),),
            outputs=(OutputSpec("value", COMBO),),
        )

    @classmethod
    def execute(cls, *, value: str) -> Mapping[str, object]:
        return cls.outputs(value=value)


class _StringSink(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(node_type="test.string_sink", inputs=(InputSpec("value", STRING),))


class _AcceptsComboOrString(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.combo_union",
            inputs=(InputSpec("value", TypeExpr.union("core.combo", "core.string")),),
        )


class _AcceptsAnything(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.combo_wildcard",
            inputs=(InputSpec("value", TypeExpr.wildcard()),),
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
            "a": GraphNode("std.math.add_ints", {"a": 1, "b": 2}),
            "i": GraphNode("dev.image.invert", {"image": Link("a", "sum")}),
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
                outputs=(OutputSpec("values", TypeExpr.list_of(STRING)),),
            )

    class ComboListSink(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.combo_list_sink",
                inputs=(InputSpec("values", TypeExpr.list_of(COMBO)),),
            )

    class ComboListSource(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.combo_list_source",
                outputs=(OutputSpec("values", TypeExpr.list_of(COMBO)),),
            )

    class StringListSink(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.string_list_sink",
                inputs=(InputSpec("values", TypeExpr.list_of(STRING)),),
            )

    class ComboScalarSink(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.combo_scalar_sink",
                inputs=(InputSpec("value", COMBO),),
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


def test_combo_literals_wildcards_unions_and_other_combos_are_legal() -> None:
    graph = Graph(
        nodes={
            # A combo literal is an ordinary plain literal, never $typed.
            "a": GraphNode("test.combo_a", {"value": "a"}),
            # Different choice lists do not refine core.combo identity.
            "b": GraphNode("test.combo_b", {"value": Link("a", "value")}),
            "wild": GraphNode("test.combo_wildcard", {"value": Link("a", "value")}),
            "union_combo": GraphNode("test.combo_union", {"value": Link("a", "value")}),
            "s": GraphNode("test.string_source", {}),
            "union_string": GraphNode("test.combo_union", {"value": Link("s", "value")}),
        }
    )
    assert validate(graph, COMBO_SCHEMAS, ["b", "wild", "union_combo", "union_string"]) == []


def test_plain_literal_on_nonconcrete_input_is_error() -> None:
    # M0 rule regression: a plain literal cannot say what type it is, so a
    # variable/wildcard input refuses it (the typed literal is the fix).
    graph = Graph(nodes={"m": GraphNode("dev.gallery.match", {"var_in": 7})})
    diags = validate(graph, SCHEMAS, ["m"])
    assert any(d.code == "literal-on-nonconcrete" and d.severity == "error" for d in diags)


def test_typed_literal_on_nonconcrete_input_is_valid() -> None:
    # The joint-contract case: an explicit stamp makes a literal legal on a
    # variable input (and a list stamp on a list<T> input) - clean document.
    graph = Graph(
        nodes={
            "m": GraphNode("dev.gallery.match", {"var_in": TypedLiteral("core.int", 7)}),
            "l": GraphNode(
                "std.list.length",
                {"list": TypedLiteral("list<core.int>", [1, 2, 3])},
            ),
        }
    )
    diags = validate(graph, SCHEMAS, ["m", "l"], known_types={"core.int"})
    assert diags == []


def test_typed_literal_malformed_type_id_is_error() -> None:
    for bad in ("", "list<>", "list<core.int", "core<int>", "list<list<>>"):
        graph = Graph(nodes={"m": GraphNode("dev.gallery.match", {"var_in": TypedLiteral(bad, 7)})})
        diags = validate(graph, SCHEMAS, ["m"])
        assert any(d.code == "typed-literal-malformed" for d in diags), bad


def test_typed_literal_unknown_type_needs_registry() -> None:
    graph = Graph(
        nodes={"m": GraphNode("dev.gallery.match", {"var_in": TypedLiteral("no.such.type", 7)})}
    )
    # Without a known-type vocabulary the check stays syntactic: clean.
    assert validate(graph, SCHEMAS, ["m"]) == []
    # With one (the engine passes its registry), an unregistered atom is an
    # error - including the atom inside a list stamp.
    diags = validate(graph, SCHEMAS, ["m"], known_types={"core.int"})
    assert any(d.code == "typed-literal-unknown-type" and d.severity == "error" for d in diags)
    listy = Graph(
        nodes={
            "m": GraphNode(
                "dev.gallery.match",
                {"var_in": TypedLiteral("list<no.such.type>", [1])},
            )
        }
    )
    diags = validate(listy, SCHEMAS, ["m"], known_types={"core.int"})
    assert any(d.code == "typed-literal-unknown-type" for d in diags)


def test_typed_literal_list_stamp_requires_list_value() -> None:
    graph = Graph(
        nodes={"m": GraphNode("dev.gallery.match", {"var_in": TypedLiteral("list<core.int>", 7)})}
    )
    diags = validate(graph, SCHEMAS, ["m"], known_types={"core.int"})
    assert any(d.code == "literal-shape" and d.severity == "error" for d in diags)


def test_typed_literal_on_concrete_input_is_redundant_warning() -> None:
    # Emission rule (joint contract): plain literals stay canonical on
    # concrete inputs; a stamp there is legal but warns.
    graph = Graph(
        nodes={
            "g": GraphNode(
                "dev.image.gradient",
                {"width": TypedLiteral("core.int", 8), "height": 8},
            )
        }
    )
    diags = validate(graph, SCHEMAS, ["g"], known_types={"core.int"})
    warnings = [d for d in diags if d.code == "typed-literal-on-concrete"]
    assert warnings and all(d.severity == "warning" for d in warnings)
    assert not has_errors(diags)


def test_typed_literal_cardinality_is_structural() -> None:
    # A list stamp into a definitely-scalar input (union) and a scalar stamp
    # into a definitely-list input are errors, mirroring the link path.
    graph = Graph(
        nodes={
            "s": GraphNode(
                "dev.gallery.sockets",
                {"union2": TypedLiteral("list<core.int>", [1])},
            ),
            "l": GraphNode("std.list.length", {"list": TypedLiteral("core.int", 7)}),
        }
    )
    codes = {d.code for d in validate(graph, SCHEMAS, ["s", "l"], known_types={"core.int"})}
    assert "list-into-scalar" in codes
    assert "scalar-into-list" in codes


def test_typed_literal_type_mismatch_is_warning() -> None:
    # Advisory compat, exactly as for links: a core.int stamp into a union
    # of image types warns but never errors.
    graph = Graph(
        nodes={"s": GraphNode("dev.gallery.sockets", {"union2": TypedLiteral("core.int", 7)})}
    )
    diags = validate(graph, SCHEMAS, ["s"], known_types={"core.int"})
    mismatches = [d for d in diags if d.code == "type-mismatch"]
    assert mismatches and all(d.severity == "warning" for d in mismatches)


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
