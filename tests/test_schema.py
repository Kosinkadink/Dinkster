import dataclasses
import json
from typing import Any, cast

import pytest
from dinkster_schema import (
    SCHEMA_WIRE_VERSION,
    AssetWidget,
    BooleanWidget,
    ColorWidget,
    ComboWidget,
    CompositorWidget,
    CurveWidget,
    CustomWidgetDescriptor,
    Deprecation,
    DynamicComboOption,
    DynamicComboSpec,
    DynamicSlotSpec,
    InputFamilySpec,
    InputSpec,
    MultiComboWidget,
    Node,
    NodeOutputError,
    NodeSchema,
    NumberWidget,
    OutputRepresents,
    OutputSpec,
    SaveTargetWidget,
    StringWidget,
    TypeExpr,
    WidgetRepresentation,
    WidgetRepresentations,
    elaborate,
    schema_from_wire,
    schema_signature,
    schema_to_wire,
    type_expr_from_wire,
    type_expr_to_wire,
)
from dinkster_schema.solve import TypeSolveError, bind_type_variables, resolved_type_id
from dinkster_values import CustomWidgetDescriptor as ValuesCustomWidgetDescriptor

SCHEMA = NodeSchema(
    node_type="test.node",
    display_name="Test Node",
    category="test",
    inputs=(
        InputSpec("a", TypeExpr.concrete("INT")),
        InputSpec("b", TypeExpr.union("INT", "FLOAT"), required=False, default=1),
        InputSpec("anything", TypeExpr.wildcard(), required=False),
    ),
    outputs=(OutputSpec("out", TypeExpr.concrete("INT")),),
)


def test_schema_reexports_custom_widget_descriptor() -> None:
    assert CustomWidgetDescriptor is ValuesCustomWidgetDescriptor


def test_wire_roundtrip_uses_only_version_one() -> None:
    wire = schema_to_wire(SCHEMA)
    assert SCHEMA_WIRE_VERSION == 1
    assert wire["schemaVersion"] == 1
    assert schema_from_wire(wire) == SCHEMA

    for unsupported in (0, 2, 44, 1.0, "1", None):
        mislabeled = dict(wire)
        mislabeled["schemaVersion"] = unsupported
        with pytest.raises(ValueError, match="unsupported schemaVersion"):
            schema_from_wire(mislabeled)
        with pytest.raises(ValueError, match="unsupported schemaVersion"):
            schema_to_wire(SCHEMA, wire_version=cast("Any", unsupported))


def test_current_schema_features_roundtrip() -> None:
    schema = dataclasses.replace(
        SCHEMA,
        dispatch_affinity="native",
        deprecation=Deprecation("Use test.other", replacement="test.other"),
        search_visibility="deprecated",
        aliases=("OldNode",),
        search_terms=("legacy",),
        output_node=True,
        emits_previews=True,
        may_expand_graph=True,
    )
    wire = schema_to_wire(schema)
    assert wire["dispatchAffinity"] == "native"
    assert wire["deprecation"] == {"message": "Use test.other", "replacement": "test.other"}
    assert wire["searchVisibility"] == "deprecated"
    assert wire["aliases"] == ["OldNode"]
    assert wire["searchTerms"] == ["legacy"]
    assert wire["outputNode"] is True
    assert wire["emitsPreviews"] is True
    assert wire["mayExpandGraph"] is True
    assert schema_from_wire(wire) == schema
    assert schema_signature(schema) == schema_signature(SCHEMA)
    # Default schemas stay unflagged on the wire: ordinary nodes never expand.
    assert "mayExpandGraph" not in schema_to_wire(SCHEMA)


@pytest.mark.parametrize(
    ("type_id", "widget", "expected"),
    [
        ("ASSET", AssetWidget(("image/png",)), {"type": "ASSET", "accept": ["image/png"]}),
        (
            "SAVE_TARGET",
            SaveTargetWidget(".png"),
            {"type": "SAVE_TARGET", "suffix": ".png"},
        ),
        ("COMBO", ComboWidget(options=("a", "b")), {"type": "COMBO", "options": ["a", "b"]}),
        (
            "BOOLEAN",
            BooleanWidget("yes", "no"),
            {"type": "BOOLEAN", "labelOn": "yes", "labelOff": "no"},
        ),
        (
            "INT",
            NumberWidget(min=0, max=10, step=1),
            {"type": "NUMBER", "min": 0, "max": 10, "step": 1},
        ),
        ("FLOAT", NumberWidget(round=False), {"type": "NUMBER", "round": False}),
        ("STRING", StringWidget(multiline=True), {"type": "STRING", "multiline": True}),
        ("STRING", ColorWidget(), {"type": "COLOR"}),
        ("CURVE", CurveWidget(), {"type": "CURVE"}),
        ("COMPOSITOR", CompositorWidget(), {"type": "COMPOSITOR"}),
    ],
)
def test_widget_roundtrip(type_id: str, widget: object, expected: dict[str, object]) -> None:
    schema = NodeSchema(
        "test.widget",
        inputs=(InputSpec("value", TypeExpr.concrete(type_id), widget=cast("Any", widget)),),
    )
    wire = schema_to_wire(schema)
    assert cast("list[dict[str, object]]", wire["interface"])[0]["widget"] == expected
    assert schema_from_wire(wire) == schema
    assert schema_signature(schema) == schema_signature(
        dataclasses.replace(schema, inputs=(dataclasses.replace(schema.inputs[0], widget=None),))
    )


@pytest.mark.parametrize("members", [("FLOAT", "INT"), ("INT", "FLOAT")])
@pytest.mark.parametrize("representations", [False, True])
@pytest.mark.parametrize("default", [25, 29.97])
def test_numeric_union_number_widget_preserves_socket_and_default(
    members: tuple[str, str], representations: bool, default: int | float
) -> None:
    widget = NumberWidget(min=1.0, max=1000.0, step=0.01, display="number", round=False)
    presentation = (
        WidgetRepresentations(
            (
                WidgetRepresentation("number", widget),
                WidgetRepresentation("slider", NumberWidget(display="slider", round=0.01)),
            ),
            default="number",
            user_switchable=True,
        )
        if representations
        else widget
    )
    schema = NodeSchema(
        "test.numeric-union",
        inputs=(
            InputSpec("frame_rate", TypeExpr.union(*members), default=default, widget=presentation),
        ),
    )
    wire = schema_to_wire(schema)
    item = cast("list[dict[str, Any]]", wire["interface"])[0]
    assert item["type"] == {"kind": "union", "types": list(members)}
    assert item["default"] == default and type(item["default"]) is type(default)
    restored = schema_from_wire(json.loads(json.dumps(wire)))
    assert restored == schema
    assert type(restored.inputs[0].default) is type(default)


@pytest.mark.parametrize("list_socket", [False, True])
@pytest.mark.parametrize("precision", [False, 0.01, 1, None])
def test_float_round_precision_preserves_false_and_absence(
    list_socket: bool, precision: Any
) -> None:
    socket = TypeExpr.concrete("FLOAT")
    if list_socket:
        socket = TypeExpr.list_of(socket)
    schema = NodeSchema(
        "test.float-round",
        inputs=(
            InputSpec("value", socket, widget=NumberWidget(display="number", round=precision)),
        ),
    )
    wire = schema_to_wire(schema)
    widget = cast("list[dict[str, Any]]", wire["interface"])[0]["widget"]
    if precision is None:
        assert "round" not in widget
        widget["round"] = None
        with pytest.raises(ValueError, match="round must be"):
            schema_from_wire(wire)
        del widget["round"]
    else:
        assert widget["round"] == precision
        assert type(widget["round"]) is type(precision)
    restored = schema_from_wire(json.loads(json.dumps(wire)))
    assert restored == schema
    assert isinstance(restored.inputs[0].widget, NumberWidget)
    assert type(restored.inputs[0].widget.round) is type(precision)
    wire_widget = cast("list[dict[str, Any]]", wire["interface"])[0]
    wire_widget["type"] = (
        {"kind": "list", "element": {"kind": "concrete", "types": ["INT"]}}
        if list_socket
        else {"kind": "concrete", "types": ["INT"]}
    )
    if precision is not None:
        integer = TypeExpr.concrete("INT")
        if list_socket:
            integer = TypeExpr.list_of(integer)
        with pytest.raises(ValueError, match="round requires"):
            InputSpec("value", integer, widget=NumberWidget(round=precision))
        with pytest.raises(ValueError, match="round requires"):
            schema_from_wire(wire)


@pytest.mark.parametrize("precision", [True, 0, 0.0, -1, float("nan"), float("inf"), -float("inf")])
def test_float_round_rejects_nonpositive_and_nonfinite_precision(precision: Any) -> None:
    with pytest.raises(ValueError, match="round must be"):
        NumberWidget(round=precision)
    schema = NodeSchema(
        "test.float-round",
        inputs=(InputSpec("value", TypeExpr.concrete("FLOAT"), widget=NumberWidget(round=0.1)),),
    )
    wire = schema_to_wire(schema)
    cast("list[dict[str, Any]]", wire["interface"])[0]["widget"]["round"] = precision
    with pytest.raises(ValueError, match="round must be"):
        schema_from_wire(wire)


def test_float_widget_preserves_large_float_bounds_on_the_wire() -> None:
    schema = NodeSchema(
        "test.float-bounds",
        inputs=(
            InputSpec(
                "value",
                TypeExpr.concrete("FLOAT"),
                widget=NumberWidget(min=-float(2**63), max=float(2**63), step=0.25),
            ),
        ),
    )

    wire = schema_to_wire(schema)
    widget = cast("list[dict[str, object]]", wire["interface"])[0]["widget"]
    assert widget == {
        "type": "NUMBER",
        "min": -9.223372036854776e18,
        "max": 9.223372036854776e18,
        "step": 0.25,
    }
    assert isinstance(widget, dict)
    assert type(widget["min"]) is float
    assert type(widget["max"]) is float
    assert schema_from_wire(wire) == schema


@pytest.mark.parametrize(
    "field,value",
    [("min", -(2**53)), ("min", 2**53), ("max", -(2**53)), ("max", 2**53), ("step", 2**53)],
)
@pytest.mark.parametrize(
    "socket",
    [
        TypeExpr.concrete("FLOAT"),
        TypeExpr.list_of(TypeExpr.concrete("FLOAT")),
        TypeExpr.union("FLOAT", "INT"),
        TypeExpr.union("INT", "FLOAT"),
    ],
)
def test_float_widget_rejects_unsafe_integer_constraints(
    field: str, value: int, socket: TypeExpr
) -> None:
    with pytest.raises(ValueError, match="FLOAT input"):
        InputSpec(
            "value",
            socket,
            widget=NumberWidget(**{field: value}),  # type: ignore[arg-type]
        )
    InputSpec(
        "value",
        socket,
        widget=NumberWidget(**{field: value - (1 if value > 0 else -1)}),  # type: ignore[arg-type]
    )


def test_multi_combo_roundtrip() -> None:
    schema = NodeSchema(
        "test.multi",
        inputs=(
            InputSpec(
                "values",
                TypeExpr.list_of(TypeExpr.concrete("COMBO")),
                default=["a"],
                widget=MultiComboWidget(options=("a", "b")),
            ),
        ),
    )
    assert schema_from_wire(schema_to_wire(schema)) == schema


def test_type_expressions_roundtrip_and_remain_closed() -> None:
    expressions = (
        TypeExpr.concrete("INT"),
        TypeExpr.union("INT", "FLOAT"),
        TypeExpr.wildcard(),
        TypeExpr.variable("T", ("INT",)),
        TypeExpr.list_of(TypeExpr.concrete("INT")),
        TypeExpr.asset_of(TypeExpr.concrete("IMAGE")),
    )
    for expression in expressions:
        assert type_expr_from_wire(type_expr_to_wire(expression)) == expression
    with pytest.raises(ValueError):
        type_expr_from_wire({"kind": "future"})


def test_malformed_widget_wire_is_rejected() -> None:
    wire = schema_to_wire(SCHEMA)
    entry = cast("list[dict[str, Any]]", wire["interface"])[0]
    entry["widget"] = {"type": ""}
    with pytest.raises(ValueError):
        schema_from_wire(wire)


def test_output_represents_wire_decode_rejects_unknown_fields_and_bad_shapes() -> None:
    schema = NodeSchema(
        node_type="test.represented-output",
        inputs=(
            InputSpec(
                "image",
                TypeExpr.asset_of(TypeExpr.concrete("IMAGE")),
                widget=AssetWidget(kind="media/image"),
            ),
        ),
        outputs=(
            OutputSpec(
                "image",
                TypeExpr.concrete("IMAGE"),
                represents=OutputRepresents("image", "decoded-image"),
            ),
        ),
    )

    def malformed(represents: object) -> dict[str, object]:
        wire = schema_to_wire(schema)
        cast("list[dict[str, Any]]", wire["interface"])[1]["represents"] = represents
        return wire

    with pytest.raises(ValueError, match="output.represents must be an object"):
        schema_from_wire(malformed(1))
    with pytest.raises(ValueError, match="output.represents has unknown fields"):
        schema_from_wire(
            malformed({"input": "image", "rendition": "decoded-image", "surprise": True})
        )
    with pytest.raises(ValueError, match="output.represents.input must be a string"):
        schema_from_wire(malformed({"input": 1, "rendition": "decoded-image"}))
    with pytest.raises(ValueError, match="output.represents.applies must be an object"):
        schema_from_wire(malformed({"input": "image", "rendition": "decoded-image", "applies": []}))
    with pytest.raises(ValueError, match="must be an array of strings"):
        schema_from_wire(
            malformed(
                {
                    "input": "image",
                    "rendition": "decoded-image",
                    "applies": {"mode": [1]},
                }
            )
        )


class TwoOutputs(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "test.two_outputs",
            outputs=(
                OutputSpec("first", TypeExpr.concrete("INT")),
                OutputSpec("second", TypeExpr.concrete("INT")),
            ),
        )


def test_outputs_helper_requires_exact_output_ids() -> None:
    assert TwoOutputs.outputs(first=1, second=2) == {"first": 1, "second": 2}
    with pytest.raises(NodeOutputError, match="missing outputs: second"):
        TwoOutputs.outputs(first=1, secondd=2)


def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(ValueError):
        NodeSchema(
            "test.duplicate",
            inputs=(
                InputSpec("x", TypeExpr.concrete("INT")),
                InputSpec("x", TypeExpr.concrete("INT")),
            ),
        )


def test_mixed_input_order_survives_wire_and_elaboration() -> None:
    integer = TypeExpr.concrete("INT")
    schema = NodeSchema(
        "test.ordered",
        inputs=(InputSpec("before", integer), InputSpec("after", integer)),
        input_families=(InputFamilySpec("values", integer, min_members=0),),
        combos=(
            DynamicComboSpec(
                "mode",
                (DynamicComboOption("a", (InputSpec("value", integer),)),),
                default="a",
            ),
        ),
        slots=(DynamicSlotSpec("slot", slot_type=integer, required=False),),
        input_order=("values", "before", "slot", "mode", "after"),
    )
    wire = schema_to_wire(schema)
    assert [item["id"] for item in cast("list[dict[str, Any]]", wire["interface"])] == [
        "values",
        "before",
        "slot",
        "mode",
        "after",
    ]
    assert schema_from_wire(wire) == schema
    effective = elaborate(
        schema_from_wire(wire),
        ("values.b", "values.a", "slot"),
        slot_variants={"mode": "a"},
    )
    assert [spec.id for spec in effective.inputs] == [
        "values.b",
        "values.a",
        "before",
        "slot",
        "mode.value",
        "after",
    ]
    assert schema_from_wire(schema_to_wire(effective)) == effective
    assert schema_signature(schema) != schema_signature(dataclasses.replace(schema, input_order=()))


@pytest.mark.parametrize(
    "order",
    [("family", "a"), ("family", "a", "a"), ("unknown", "a", "b"), ("family", "b", "a")],
)
def test_input_order_rejects_missing_duplicate_unknown_or_reversed_entries(order) -> None:
    integer = TypeExpr.concrete("INT")
    with pytest.raises(ValueError, match="input_order"):
        NodeSchema(
            "test.invalid_order",
            inputs=(InputSpec("a", integer), InputSpec("b", integer)),
            input_families=(InputFamilySpec("family", integer, min_members=0),),
            input_order=order,
        )


def test_positional_tuple_schema_roundtrip_and_independent_family_bindings() -> None:
    variable = TypeExpr.variable("T")
    schema = NodeSchema(
        "test.collect_product",
        input_families=(InputFamilySpec("items", variable, independent_types=True),),
        outputs=(OutputSpec("tuple", TypeExpr.tuple_from_family("items")),),
    )
    assert schema_from_wire(schema_to_wire(schema)) == schema
    stored = {"items.image": object(), "items.count": object()}
    effective = elaborate(schema, stored)
    bindings = bind_type_variables(effective.inputs, {"items.image": "IMAGE", "items.count": "INT"})
    assert resolved_type_id(effective.outputs[0].type, bindings) == "tuple<IMAGE,INT>"
    assert schema_from_wire(schema_to_wire(effective)) == effective
    reversed_effective = elaborate(schema, dict(reversed(tuple(stored.items()))))
    reversed_bindings = bind_type_variables(
        reversed_effective.inputs, {"items.image": "IMAGE", "items.count": "INT"}
    )
    assert (
        resolved_type_id(reversed_effective.outputs[0].type, reversed_bindings)
        == "tuple<INT,IMAGE>"
    )
    shared = dataclasses.replace(
        schema,
        input_families=(InputFamilySpec("items", variable),),
        outputs=(OutputSpec("list", TypeExpr.list_of(variable)),),
    )
    with pytest.raises(TypeSolveError, match="already bound"):
        bind_type_variables(
            elaborate(shared, stored).inputs, {"items.image": "IMAGE", "items.count": "INT"}
        )
    nested = TypeExpr.tuple_of(
        TypeExpr.concrete("IMAGE"), TypeExpr.list_of(TypeExpr.tuple_of(TypeExpr.concrete("INT")))
    )
    assert type_expr_from_wire(type_expr_to_wire(nested)) == nested
    assert nested.runtime_type_id() == "tuple<IMAGE,list<tuple<INT>>>"
    assert nested.cardinality() == "scalar"
    assert not nested.accepts_concrete("tuple<INT,list<tuple<INT>>>")


def test_tuple_element_projects_only_a_stored_integer_selector() -> None:
    schema = NodeSchema(
        "test.project_product",
        inputs=(
            InputSpec("tuple", TypeExpr.variable("Tuple")),
            InputSpec("index", TypeExpr.concrete("INT"), default=0),
        ),
        outputs=(OutputSpec("item", TypeExpr.tuple_element("tuple", "index")),),
    )
    assert schema_from_wire(schema_to_wire(schema)) == schema
    for stored, expected in (({}, "IMAGE"), ({"index": 1}, "INT"), ({"index": -1}, "INT")):
        effective = elaborate(schema, stored)
        bindings = bind_type_variables(effective.inputs, {"tuple": "tuple<IMAGE,INT>"})
        assert resolved_type_id(effective.outputs[0].type, bindings) == expected
        assert schema_from_wire(schema_to_wire(effective)) == effective
    for value in (True, 1.0, "1", object()):
        with pytest.raises(ValueError, match="constant INT.*list for same-typed"):
            elaborate(schema, {"index": value})
    effective = elaborate(schema, {"index": 2})
    with pytest.raises(TypeSolveError, match="out of range"):
        resolved_type_id(effective.outputs[0].type, {"Tuple": "tuple<IMAGE,INT>"})
    with pytest.raises(TypeSolveError, match="requires a tuple"):
        resolved_type_id(effective.outputs[0].type, {"Tuple": "list<INT>"})


def test_tuple_schema_rejects_missing_or_shared_declaration_sources() -> None:
    with pytest.raises(ValueError, match="independent"):
        NodeSchema(
            "test.bad_family", outputs=(OutputSpec("tuple", TypeExpr.tuple_from_family("missing")),)
        )
    with pytest.raises(ValueError, match="unknown tuple input"):
        NodeSchema(
            "test.bad_projection",
            outputs=(OutputSpec("item", TypeExpr.tuple_element("missing", "index")),),
        )
    with pytest.raises(ValueError, match="tuple_of"):
        TypeExpr.concrete("tuple<IMAGE,INT>")
