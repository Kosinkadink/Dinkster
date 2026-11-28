from dataclasses import replace
from typing import cast

import pytest
from dinkster_schema import (
    ComboWidget,
    Formula,
    Group,
    Input,
    Node,
    NodeOutputError,
    NodeSchema,
    Numbered,
    Output,
    OutputKnownValue,
    OutputListCount,
    StringWidget,
    TypeExpr,
    schema_from_wire,
    schema_to_wire,
)


def test_list_count_advice_does_not_invalidate_cached_computation() -> None:
    from dinkster_schema import elaborate, schema_signature

    integer = TypeExpr.concrete("INT")
    items = TypeExpr.list_of(integer)
    plain = NodeSchema(
        node_type="test.counted",
        body=(
            Input("items", items),
            Input("count", integer, pin=False),
            Output("out", items),
            Input("video", TypeExpr.asset_of(TypeExpr.concrete("IMAGE"))),
        ),
    )
    for count in (0, 7, OutputListCount("items"), Formula("frames(video)")):
        advised = replace(
            plain,
            body=(
                plain.inputs[0],
                replace(plain.inputs[1], list_count_of="items"),
                replace(plain.outputs[0], list_count=count),
                plain.inputs[2],
            ),
        )
        restored = schema_from_wire(schema_to_wire(advised))
        assert restored.outputs[0].list_count == count
        assert schema_signature(restored) == schema_signature(plain)
        assert schema_signature(
            elaborate(restored, {"items": [1, 2], "count": 2})
        ) == schema_signature(elaborate(plain, {"items": [1, 2], "count": 2}))

    with pytest.raises(ValueError, match=r"frames\(\) takes the name of an input"):
        replace(
            plain, body=(*plain.inputs, Output("out", items, list_count=Formula("frames(missing)")))
        )


@pytest.mark.parametrize("source", ["items", "missing"])
@pytest.mark.parametrize("output", [False, True])
def test_count_sources_do_not_escape_their_declared_scope(source: str, output: bool) -> None:
    integer = TypeExpr.concrete("INT")
    items = TypeExpr.list_of(integer)
    plain = Output("out", items) if output else Input("count", integer, pin=False)
    counted = (
        replace(plain, list_count=OutputListCount(source))
        if isinstance(plain, Output)
        else replace(plain, list_count_of=source)
    )
    group = Group("parts", keys=Numbered("p"), member=(Input("items", items), plain))
    with pytest.raises(ValueError, match="count.*top-level"):
        NodeSchema(
            node_type="test.count_scope",
            body=(replace(group, member=(Input("items", items), counted)),),
        )

    wire = schema_to_wire(NodeSchema(node_type="test.count_scope", body=(group,)))
    member = cast(
        "list[dict[str, object]]", cast("list[dict[str, object]]", wire["body"])[0]["member"]
    )
    member[1]["listCount" if output else "listCountOf"] = {"input": source} if output else source
    with pytest.raises(ValueError, match="count.*top-level"):
        schema_from_wire(wire)

    if not output:
        with pytest.raises(ValueError, match="list_count_of must name a list input"):
            NodeSchema(
                node_type="test.unsent_count",
                body=(replace(cast("Input", counted), list_count_of="missing", sent=False),),
            )


class TwoOutputs(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.two_outputs",
            body=(
                Output("first", TypeExpr.concrete("INT")),
                Group("rest", value=Output("item", TypeExpr.concrete("INT")), keys=Numbered("")),
                Output("second", TypeExpr.concrete("INT")),
            ),
        )


def test_outputs_helper_rejects_typos_at_return_site() -> None:
    assert TwoOutputs.outputs(first=1, second=2, rest={"1": 3}) == {
        "first": 1,
        "second": 2,
        "rest": {"1": 3},
    }
    with pytest.raises(NodeOutputError) as excinfo:
        TwoOutputs.outputs(first=1, secondd=2, rest={"1": 3})
    message = str(excinfo.value)
    assert "test.two_outputs" in message
    assert "missing outputs: second" in message
    assert "unknown outputs: ['secondd']" in message


def _options_schema(options: Input | Group) -> NodeSchema:
    return NodeSchema(
        node_type="test.authored_options",
        body=(
            Input(
                "choice",
                TypeExpr.concrete("COMBO"),
                widget=ComboWidget(editable_options=True, options_input="options"),
            ),
            options,
            Output("INDEX", TypeExpr.concrete("INT")),
        ),
    )


_OPTIONS_TYPE = TypeExpr.list_of(TypeExpr.concrete("STRING"))


def test_editable_choice_names_the_input_that_receives_its_options() -> None:
    schema = _options_schema(Input("options", _OPTIONS_TYPE, required=False, hidden=True))
    wire = schema_to_wire(schema)
    choice = cast("list[dict[str, object]]", wire["body"])[0]
    assert choice["widget"] == {"type": "COMBO", "editableOptions": True, "optionsInput": "options"}
    assert schema_from_wire(wire) == schema
    plain = schema_to_wire(
        NodeSchema(
            node_type="test.plain_options",
            body=(
                Input(
                    "choice",
                    TypeExpr.concrete("COMBO"),
                    widget=ComboWidget(editable_options=True),
                ),
            ),
        )
    )
    plain_choice = cast("list[dict[str, dict[str, object]]]", plain["body"])[0]
    assert "optionsInput" not in plain_choice["widget"]


@pytest.mark.parametrize(
    "options",
    [
        Input("options", _OPTIONS_TYPE, required=False),
        Input("options", _OPTIONS_TYPE, hidden=True),
        Input("options", TypeExpr.concrete("STRING"), required=False, hidden=True),
        Input(
            "options",
            _OPTIONS_TYPE,
            required=False,
            hidden=True,
            widget=StringWidget(multiline=False),
        ),
        Input("other", _OPTIONS_TYPE, required=False, hidden=True),
        Group(
            "extra",
            value=Input("options", _OPTIONS_TYPE, required=False, hidden=True),
            keys=Numbered(""),
        ),
    ],
)
def test_options_input_must_be_a_hidden_optional_text_list_without_widget(
    options: Input | Group,
) -> None:
    with pytest.raises(
        ValueError, match="test.authored_options: input 'choice' options_input 'options' must"
    ):
        _options_schema(options)


def test_options_input_requires_editable_options() -> None:
    with pytest.raises(ValueError, match="options_input requires editable options"):
        ComboWidget(options=("a",), options_input="options")


def _known_value_schema(input_type: str, output_type: str) -> NodeSchema:
    widget = ComboWidget(options=("a", "b")) if input_type == "COMBO" else None
    return NodeSchema(
        node_type="test.known_value",
        body=(
            Input("value", TypeExpr.concrete(input_type), widget=widget),
            Output("out", TypeExpr.concrete(output_type), known_value=OutputKnownValue("value")),
        ),
    )


@pytest.mark.parametrize(("input_type", "output_type"), [("COMBO", "COMBO"), ("COMBO", "STRING")])
def test_known_value_reads_a_choice_for_a_choice_or_text_output(
    input_type: str, output_type: str
) -> None:
    schema = _known_value_schema(input_type, output_type)
    assert schema.outputs[0].known_value == OutputKnownValue("value")


@pytest.mark.parametrize(
    ("input_type", "output_type"), [("COMBO", "INT"), ("STRING", "COMBO"), ("INT", "FLOAT")]
)
def test_known_value_refuses_other_type_pairs(input_type: str, output_type: str) -> None:
    with pytest.raises(ValueError, match="known value must reference"):
        _known_value_schema(input_type, output_type)


def test_known_value_reads_an_input_typed_by_the_same_variable() -> None:
    value_type = TypeExpr.variable("value")
    schema = NodeSchema(
        node_type="test.known_variable",
        body=(
            Input("value", value_type),
            Output("out", value_type, known_value=OutputKnownValue("value")),
        ),
    )
    assert schema.outputs[0].known_value == OutputKnownValue("value")
    with pytest.raises(ValueError, match="known value must reference"):
        NodeSchema(
            node_type="test.known_other_variable",
            body=(
                Input("value", value_type),
                Output("out", TypeExpr.variable("other"), known_value=OutputKnownValue("value")),
            ),
        )


def _limited(
    limits: dict[str, str], value_type: TypeExpr | None = None, *rows: Input
) -> NodeSchema:
    value_type = value_type or TypeExpr.concrete("INT")
    limit_rows = rows or tuple(
        Input(row, value_type, required=False) for row in ("low", "high", "fallback")
    )
    return NodeSchema(
        node_type="test.limited",
        body=(
            Input("value", value_type),
            *limit_rows,
            Output("out", value_type, known_value=OutputKnownValue("value", limits=limits)),
        ),
    )


def test_known_value_names_the_inputs_that_hold_its_limits_outside_both_signatures() -> None:
    from dinkster_schema.wire import schema_signature

    schema = _limited({"default": "fallback", "max": "high", "min": "low"})
    wire = schema_to_wire(schema)

    def known_value(wire: dict[str, object]) -> dict[str, object]:
        return cast("list[dict[str, dict[str, object]]]", wire["body"])[-1]["knownValue"]

    known = known_value(wire)
    # The wire is canonical: limits in the order min, max, step, default.
    assert known == {
        "input": "value",
        "limits": {"min": "low", "max": "high", "default": "fallback"},
    }
    assert list(cast("dict[str, str]", known["limits"])) == ["min", "max", "default"]
    assert schema_from_wire(wire).body == schema.body
    assert "limits" not in known_value(schema_to_wire(_limited({})))
    hash(schema.outputs[0].known_value)

    def signatures(limits: dict[str, str]) -> tuple[str, str]:
        declared = _limited(limits)
        flat = NodeSchema(
            "test.limited", inputs=declared.inputs, outputs=declared.outputs, effective=True
        )
        return schema_signature(declared), schema_signature(flat)

    assert signatures({}) == signatures({"min": "low"}) == signatures({"min": "high"})

    # A type-variable value may name only a default, on an input of the same variable.
    variable = TypeExpr.variable("value")
    assert _limited({"default": "fallback"}, variable).outputs[0].known_value is not None
    assert _limited({"default": "fallback"}, TypeExpr.concrete("STRING")).outputs[0].known_value

    int_type = TypeExpr.concrete("INT")
    for limits, value_type, rows, message in (
        ({"floor": "low"}, None, (), "unknown output known value limit"),
        ({"min": ""}, None, (), "must name an input"),
        ({"min": "missing"}, None, (), "another optional input"),
        ({"min": "value"}, None, (), "another optional input"),
        ({"min": "low", "max": "low"}, None, (), "another optional input"),
        ({"min": "low"}, None, (Input("low", int_type),), "another optional input"),
        (
            {"min": "low"},
            None,
            (Input("low", TypeExpr.concrete("FLOAT"), required=False),),
            "another optional input",
        ),
        ({"min": "low"}, variable, (), "needs an INT or FLOAT value"),
        ({"step": "low"}, TypeExpr.concrete("STRING"), (), "needs an INT or FLOAT value"),
        (
            {"default": "fallback"},
            variable,
            (Input("fallback", TypeExpr.variable("other"), required=False),),
            "another optional input",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            _limited(limits, value_type, *rows)

    for bad, message in (
        ({}, "at least one limit"),
        ({"floor": "low"}, "unknown fields"),
        ({"min": 1}, "must be a string"),
        ({"min": ""}, "non-empty string"),
    ):
        known["limits"] = bad
        with pytest.raises(ValueError, match=message):
            schema_from_wire(wire)
