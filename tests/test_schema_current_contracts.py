from typing import cast

import pytest
from dinkster_schema import (
    ComboWidget,
    Group,
    Input,
    Node,
    NodeOutputError,
    NodeSchema,
    Numbered,
    Output,
    OutputKnownValue,
    StringWidget,
    TypeExpr,
    schema_from_wire,
    schema_to_wire,
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


def _options_schema(options: Input) -> NodeSchema:
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
