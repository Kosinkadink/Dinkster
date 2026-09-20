import dataclasses
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
    InputSpec,
    MultiComboWidget,
    Node,
    NodeOutputError,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    SaveTargetWidget,
    StringWidget,
    TypeExpr,
    schema_from_wire,
    schema_signature,
    schema_to_wire,
    type_expr_from_wire,
    type_expr_to_wire,
)
from dinkster_values import CustomWidgetDescriptor as ValuesCustomWidgetDescriptor

SCHEMA = NodeSchema(
    node_type="test.node",
    display_name="Test Node",
    category="test",
    inputs=(
        InputSpec("a", TypeExpr.concrete("core.int")),
        InputSpec("b", TypeExpr.union("core.int", "core.float"), required=False, default=1),
        InputSpec("anything", TypeExpr.wildcard(), required=False),
    ),
    outputs=(OutputSpec("out", TypeExpr.concrete("core.int")),),
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
    )
    wire = schema_to_wire(schema)
    assert wire["dispatchAffinity"] == "native"
    assert wire["deprecation"] == {"message": "Use test.other", "replacement": "test.other"}
    assert wire["searchVisibility"] == "deprecated"
    assert wire["aliases"] == ["OldNode"]
    assert wire["searchTerms"] == ["legacy"]
    assert wire["outputNode"] is True
    assert wire["emitsPreviews"] is True
    assert schema_from_wire(wire) == schema
    assert schema_signature(schema) == schema_signature(SCHEMA)


@pytest.mark.parametrize(
    ("type_id", "widget", "expected"),
    [
        ("dinkster.asset", AssetWidget(("image/png",)), {"type": "ASSET", "accept": ["image/png"]}),
        (
            "dinkster.save_target",
            SaveTargetWidget(".png"),
            {"type": "SAVE_TARGET", "suffix": ".png"},
        ),
        ("core.combo", ComboWidget(options=("a", "b")), {"type": "COMBO", "options": ["a", "b"]}),
        (
            "core.boolean",
            BooleanWidget("yes", "no"),
            {"type": "BOOLEAN", "labelOn": "yes", "labelOff": "no"},
        ),
        (
            "core.int",
            NumberWidget(min=0, max=10, step=1),
            {"type": "NUMBER", "min": 0, "max": 10, "step": 1},
        ),
        ("core.string", StringWidget(multiline=True), {"type": "STRING", "multiline": True}),
        ("core.string", ColorWidget(), {"type": "COLOR"}),
        ("dinkster.curve", CurveWidget(), {"type": "CURVE"}),
        ("dinkster.compositor", CompositorWidget(), {"type": "COMPOSITOR"}),
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


def test_multi_combo_roundtrip() -> None:
    schema = NodeSchema(
        "test.multi",
        inputs=(
            InputSpec(
                "values",
                TypeExpr.list_of(TypeExpr.concrete("core.combo")),
                default=["a"],
                widget=MultiComboWidget(options=("a", "b")),
            ),
        ),
    )
    assert schema_from_wire(schema_to_wire(schema)) == schema


def test_type_expressions_roundtrip_and_remain_closed() -> None:
    expressions = (
        TypeExpr.concrete("core.int"),
        TypeExpr.union("core.int", "core.float"),
        TypeExpr.wildcard(),
        TypeExpr.variable("T", ("core.int",)),
        TypeExpr.list_of(TypeExpr.concrete("core.int")),
        TypeExpr.asset_of(TypeExpr.concrete("dinkster.image")),
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


class TwoOutputs(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "test.two_outputs",
            outputs=(
                OutputSpec("first", TypeExpr.concrete("core.int")),
                OutputSpec("second", TypeExpr.concrete("core.int")),
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
                InputSpec("x", TypeExpr.concrete("core.int")),
                InputSpec("x", TypeExpr.concrete("core.int")),
            ),
        )
