import dataclasses
import json
from typing import Any, cast

import pytest
from dinkster_schema import (
    ComboWidget,
    DynamicComboOption,
    DynamicComboSpec,
    DynamicSlotSpec,
    InputFamilySpec,
    InputSpec,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    elaborate,
    schema_from_wire,
    schema_signature,
    schema_to_wire,
    type_expr_from_wire,
    type_expr_to_wire,
)
from dinkster_schema.solve import TypeSolveError, bind_type_variables, resolved_type_id

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


def test_editable_combo_and_additional_inputs_preserve_declared_types() -> None:
    schema = NodeSchema(
        "test.authored",
        inputs=(
            InputSpec(
                "choice", TypeExpr.concrete("COMBO"), widget=ComboWidget(editable_options=True)
            ),
        ),
        additional_inputs=TypeExpr.wildcard(),
    )
    assert schema_from_wire(json.loads(json.dumps(schema_to_wire(schema)))) == schema
    effective = elaborate(
        schema, {"choice": "not in menu", "option117": "duplicate", "sparse-name": 7}
    )
    assert [(spec.id, spec.type, spec.required) for spec in effective.inputs] == [
        ("choice", TypeExpr.concrete("COMBO"), True),
        ("option117", TypeExpr.wildcard(), False),
        ("sparse-name", TypeExpr.wildcard(), False),
    ]
    assert effective.is_static
    assert schema_signature(schema) != schema_signature(
        dataclasses.replace(schema, additional_inputs=None)
    )
    assert len(elaborate(schema, ["choice", *(f"option{i * 2}" for i in range(150))]).inputs) == 151
    with pytest.raises(ValueError, match="budget"):
        elaborate(schema, ["choice", *(f"option{i}" for i in range(1024))])
    with pytest.raises(ValueError, match="needs static options"):
        ComboWidget()


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
