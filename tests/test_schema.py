import dataclasses
import json
from typing import Any, cast

import pytest
from dinkster_schema import (
    SCHEMA_WIRE_VERSION,
    AssetWidget,
    BooleanWidget,
    ColorWidget,
    ComboOption,
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
    MountWidget,
    MultiComboWidget,
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
    with pytest.raises(ValueError, match="MOUNT.*reserved"):
        CustomWidgetDescriptor("MOUNT")


def test_scalar_combo_choices_preserve_wire_types_and_metadata() -> None:
    options = ("auto", 8, 10, True, 1, 1.5, "8", ComboOption(False, label="Disabled"))
    schema = dataclasses.replace(
        SCHEMA,
        inputs=(
            InputSpec(
                "depth", TypeExpr.concrete("COMBO"), default=8, widget=ComboWidget(options=options)
            ),
        ),
    )
    wire = schema_to_wire(schema)
    interface = cast("list[dict[str, Any]]", wire["interface"])
    assert interface[0]["widget"]["options"] == [
        "auto",
        8,
        10,
        True,
        1,
        1.5,
        "8",
        {"value": False, "label": "Disabled"},
    ]
    decoded = schema_from_wire(json.loads(json.dumps(wire)))
    assert decoded == schema
    widget = decoded.inputs[0].widget
    assert isinstance(widget, ComboWidget)
    assert tuple(type(option) for option in widget.options) == tuple(type(x) for x in options)
    assert type(decoded.inputs[0].default) is int
    assert schema_to_wire(decoded) == wire


@pytest.mark.parametrize("value", [-(2**53 - 1), 2**53 - 1])
def test_static_combo_safe_integer_choices_roundtrip(value: int) -> None:
    schema = NodeSchema(
        node_type="test.safe-combo",
        inputs=(
            InputSpec(
                "choice",
                TypeExpr.concrete("COMBO"),
                default=value,
                widget=ComboWidget(options=(value, ComboOption(value, label="Boundary"))),
            ),
        ),
    )
    assert schema_from_wire(json.loads(json.dumps(schema_to_wire(schema)))) == schema


@pytest.mark.parametrize("value", [-(2**53 + 1), -(2**53), 2**53, 2**53 + 1])
@pytest.mark.parametrize("presented", [False, True])
def test_static_combo_wire_rejects_unsafe_integer_choices(value: int, presented: bool) -> None:
    wire = schema_to_wire(
        NodeSchema(
            node_type="test.unsafe-combo",
            inputs=(
                InputSpec("choice", TypeExpr.concrete("COMBO"), widget=ComboWidget(options=(1,))),
            ),
        )
    )
    cast("list[dict[str, Any]]", wire["interface"])[0]["widget"]["options"] = [
        {"value": value, "label": "Unsafe"} if presented else value
    ]
    with pytest.raises(ValueError):
        schema_from_wire(wire)


@pytest.mark.parametrize("value", [None, [], {}, "", float("nan"), float("inf"), -(2**53), 2**53])
def test_static_combo_rejects_non_choices(value: Any) -> None:
    with pytest.raises(ValueError):
        ComboWidget(options=(value,))
    with pytest.raises(ValueError):
        ComboOption(value)


@pytest.mark.parametrize("value", [8, True, 1.5, ComboOption(8)])
def test_multi_combo_retains_string_choices(value: Any) -> None:
    with pytest.raises(ValueError):
        MultiComboWidget(options=(value,))


@pytest.mark.parametrize("value", [-(2**53), 2**53, float("inf"), float("nan"), [], {}])
def test_scalar_combo_rejects_invalid_defaults_in_model_and_wire(value: Any) -> None:
    with pytest.raises(ValueError, match="combo default"):
        InputSpec("choice", TypeExpr.concrete("COMBO"), default=value)
    wire = schema_to_wire(
        NodeSchema(
            node_type="test.combo-default",
            inputs=(InputSpec("choice", TypeExpr.concrete("COMBO")),),
        )
    )
    cast("list[dict[str, Any]]", wire["interface"])[0]["default"] = value
    with pytest.raises(ValueError, match="combo default"):
        schema_from_wire(wire)


@pytest.mark.parametrize("input_values", [False, True])
def test_wire_roundtrip_uses_only_version_one(input_values: bool) -> None:
    schema = dataclasses.replace(SCHEMA, input_values=input_values)
    wire = schema_to_wire(schema)
    assert SCHEMA_WIRE_VERSION == 1
    assert wire["schemaVersion"] == 1
    assert schema_from_wire(wire) == schema

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
        ("STRING", MountWidget(), {"type": "MOUNT"}),
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


@pytest.mark.parametrize("representations", [False, True])
def test_forced_floats_numeric_presentation_preserves_socket(representations: bool) -> None:
    widget = NumberWidget(min=-1, step=0.001)
    presentation = (
        WidgetRepresentations(
            (WidgetRepresentation("number", widget),), default="number", user_switchable=False
        )
        if representations
        else widget
    )
    schema = NodeSchema(
        "test.floats-presentation",
        inputs=(
            InputSpec(
                "floats_strength",
                TypeExpr.concrete("FLOATS"),
                default=-1,
                force_input=True,
                widget=presentation,
            ),
        ),
    )
    wire = schema_to_wire(schema)
    item = cast("list[dict[str, Any]]", wire["interface"])[0]
    assert item["type"] == {"kind": "concrete", "types": ["FLOATS"]}
    assert item["default"] == -1
    assert item["forceInput"] is True
    assert schema_from_wire(wire) == schema
    assert schema.inputs[0].widget == presentation
    assert widget.min == -1 and widget.step == 0.001
    for wrong_type, forced in (
        (TypeExpr.concrete("FLOATS"), False),
        (TypeExpr.concrete("OPAQUE"), True),
        (TypeExpr.list_of(TypeExpr.concrete("FLOATS")), True),
        (TypeExpr.union("FLOATS", "FLOAT"), True),
    ):
        with pytest.raises(ValueError, match="number widget requires"):
            InputSpec("value", wrong_type, force_input=forced, widget=presentation)


@pytest.mark.parametrize("scope", ["direct", "combo", "family"])
def test_socketless_color_preserves_nested_literals_and_execution_identity(scope: str) -> None:
    def declaration(socketless: bool) -> NodeSchema:
        color = InputSpec(
            "color",
            TypeExpr.concrete("COLOR"),
            default="#123456",
            widget=ColorWidget(),
            socketless=socketless,
        )
        if scope == "combo":
            return NodeSchema(
                "ColorExample",
                combos=(
                    DynamicComboSpec(
                        "padding_style",
                        (DynamicComboOption("color", (color,)),),
                        default="color",
                    ),
                ),
            )
        if scope == "family":
            return NodeSchema("ColorExample", input_families=(InputFamilySpec("colors", (color,)),))
        return NodeSchema("ColorExample", inputs=(color,))

    schema = declaration(True)
    assert schema_from_wire(schema_to_wire(schema)) == schema
    assert schema_signature(schema) == schema_signature(declaration(False))
    stored = {"colors.first.color": "#abcdef"} if scope == "family" else {}
    choices = {"padding_style": "color"} if scope == "combo" else None
    effective = elaborate(schema, stored, slot_variants=choices)
    color = next(spec for spec in effective.inputs if spec.widget == ColorWidget())
    assert color.type == TypeExpr.concrete("COLOR")
    assert color.default == "#123456"
    assert color.socketless is True
    wire = schema_to_wire(effective)
    entries = cast("list[dict[str, Any]]", wire["interface"])
    entry = next(item for item in entries if item["id"] == color.id)
    assert entry["socketless"] is True
    assert entry["widget"] == {"type": "COLOR"}
    assert "forceInput" not in entry
    assert schema_from_wire(wire) == effective


def test_socketless_input_defaults_and_invalid_declarations() -> None:
    color = InputSpec("color", TypeExpr.concrete("COLOR"), widget=ColorWidget())
    wire = schema_to_wire(NodeSchema("ColorExample", inputs=(color,)))
    entry = cast("list[dict[str, Any]]", wire["interface"])[0]
    assert color.socketless is False
    assert "socketless" not in entry
    assert schema_from_wire(wire).inputs[0].socketless is False
    for bad in (None, 0, 1, "true"):
        with pytest.raises(ValueError, match="socketless"):
            dataclasses.replace(color, socketless=bad)
        entry["socketless"] = bad
        with pytest.raises(ValueError, match="socketless"):
            schema_from_wire(wire)
    entry["socketless"] = True
    entry["forceInput"] = True
    with pytest.raises(ValueError, match="socketless.*force_input"):
        schema_from_wire(wire)


def test_save_mount_adjacency_preserves_wire_order_and_execution_identity() -> None:
    from dinkster_api.v1 import MountWidget as PublicMountWidget

    assert PublicMountWidget is MountWidget
    schema = NodeSchema(
        "SaveExample",
        inputs=(
            InputSpec("filename_prefix", TypeExpr.concrete("STRING"), default="Example"),
            InputSpec("quality", TypeExpr.concrete("INT"), default=90),
            InputSpec(
                "mount",
                TypeExpr.concrete("STRING"),
                required=False,
                widget=MountWidget(),
                adjacent_to="filename_prefix",
            ),
        ),
    )
    wire = schema_to_wire(schema)
    entries = cast("list[dict[str, Any]]", wire["interface"])
    assert [entry["id"] for entry in entries] == ["filename_prefix", "quality", "mount"]
    assert "adjacentTo" not in entries[0]
    assert entries[-1]["adjacentTo"] == "filename_prefix"
    assert entries[-1]["widget"] == {"type": "MOUNT"}
    assert entries[-1]["required"] is False
    assert "default" not in entries[-1]
    assert schema_from_wire(wire) == schema
    without_hint = dataclasses.replace(
        schema,
        inputs=(*schema.inputs[:-1], dataclasses.replace(schema.inputs[-1], adjacent_to=None)),
    )
    assert schema_signature(schema) == schema_signature(without_hint)
    for bad in (None, "", 7, "missing", "mount"):
        entries[-1]["adjacentTo"] = bad
        with pytest.raises(ValueError, match="adjacent"):
            schema_from_wire(wire)
    entries[-1]["adjacentTo"] = "filename_prefix"
    entries[-1]["widget"] = {"type": "MOUNT", "path": "/tmp"}
    with pytest.raises(ValueError, match="MOUNT"):
        schema_from_wire(wire)


def test_input_family_adjacency_is_scoped_to_its_siblings() -> None:
    prefix = InputSpec("prefix", TypeExpr.concrete("STRING"))
    mount = InputSpec("mount", TypeExpr.concrete("STRING"), adjacent_to="prefix")
    family = InputFamilySpec("targets", (prefix, mount))
    schema = NodeSchema("SaveExamples", input_families=(family,))
    assert schema_from_wire(schema_to_wire(schema)) == schema
    without_hint = NodeSchema(
        "SaveExamples",
        input_families=(
            InputFamilySpec("targets", (prefix, dataclasses.replace(mount, adjacent_to=None))),
        ),
    )
    assert schema_signature(schema) == schema_signature(without_hint)
    for target in ("missing", "mount"):
        with pytest.raises(ValueError, match="adjacent_to"):
            InputFamilySpec("targets", (prefix, dataclasses.replace(mount, adjacent_to=target)))


@pytest.mark.parametrize(
    "input_type",
    [TypeExpr.concrete("INT"), TypeExpr.list_of(TypeExpr.concrete("STRING")), TypeExpr.wildcard()],
)
def test_mount_widget_requires_scalar_string(input_type: TypeExpr) -> None:
    with pytest.raises(ValueError, match="mount widget requires a concrete STRING"):
        InputSpec("mount", input_type, widget=MountWidget())


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


@pytest.mark.parametrize("hidden", [False, True])
def test_dynamic_combo_visibility_round_trips_without_changing_choice_contract(
    hidden: bool,
) -> None:
    combo = DynamicComboSpec(
        "codec",
        (
            DynamicComboOption("auto"),
            DynamicComboOption("custom", (InputSpec("quality", TypeExpr.concrete("INT")),)),
        ),
        required=False,
        hidden=hidden,
    )
    schema = NodeSchema("test.hidden-choice", combos=(combo,))
    wire = schema_to_wire(schema)
    entry = cast("list[dict[str, Any]]", wire["interface"])[0]
    assert (entry.get("hidden") is True) is hidden
    if not hidden:
        assert "hidden" not in entry
    assert "default" not in entry
    assert entry["required"] is False
    assert schema_from_wire(wire) == schema
    effective = elaborate(schema_from_wire(wire), (), slot_variants={"codec": "custom"})
    assert [item.id for item in effective.inputs] == ["codec.quality"]
    assert effective.inputs[0].type == TypeExpr.concrete("INT")
    assert elaborate(schema_from_wire(wire), ()).inputs == ()


@pytest.mark.parametrize("hidden", [None, 0, 1, "true"])
def test_dynamic_combo_visibility_rejects_nonboolean_flags(hidden: object) -> None:
    with pytest.raises(ValueError, match="hidden must be a bool"):
        DynamicComboSpec("codec", (DynamicComboOption("auto"),), hidden=hidden)  # type: ignore[arg-type]
    wire = schema_to_wire(
        NodeSchema(
            "test.hidden-choice", combos=(DynamicComboSpec("codec", (DynamicComboOption("auto"),)),)
        )
    )
    cast("list[dict[str, Any]]", wire["interface"])[0]["hidden"] = hidden
    with pytest.raises(ValueError, match="dynamicCombo.hidden"):
        schema_from_wire(wire)


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
