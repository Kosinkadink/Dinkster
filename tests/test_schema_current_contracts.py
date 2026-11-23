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
    TypeExpr,
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
