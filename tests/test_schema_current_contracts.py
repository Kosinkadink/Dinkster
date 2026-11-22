import pytest
from dinkster_schema import Group, Node, NodeOutputError, NodeSchema, Numbered, Output, TypeExpr


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
