import dataclasses

import pytest
from dinkster_schema import (
    OPTION_KEY_PATTERN,
    DynamicComboOption,
    DynamicComboSpec,
    InputSpec,
    MirrorSpec,
    MirrorTolerance,
    Node,
    NodeOutputError,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)

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


EXPRESSION_MIRROR = MirrorSpec(
    kind="expression",
    precision="bounded",
    tolerance=MirrorTolerance(relative=1e-12),
    grammar_version=1,
)


GLSL_MIRROR = MirrorSpec(
    kind="glsl",
    precision="bounded",
    tolerance=MirrorTolerance(per_channel=1.0 / 255.0),
    source="void main() {}",
)


SCOPED_GLSL_MIRROR = dataclasses.replace(
    GLSL_MIRROR, applies={"operation": ("gaussian_blur", "sharpen")}
)


OPERATION_COMBO = DynamicComboSpec(
    "operation",
    options=(
        DynamicComboOption("gaussian_blur"),
        DynamicComboOption("sharpen"),
        DynamicComboOption("noise"),
    ),
    default="gaussian_blur",
)


SCOPED_SCHEMA = dataclasses.replace(SCHEMA, combos=(OPERATION_COMBO,), mirror=SCOPED_GLSL_MIRROR)


class TwoOutputs(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.two_outputs",
            outputs=(
                OutputSpec("first", TypeExpr.concrete("INT")),
                OutputSpec("second", TypeExpr.concrete("INT")),
            ),
        )


@pytest.mark.parametrize(
    "key",
    ["Flux.2 [pro]", "Flux.2 [max]", "scale dimensions", "center"],
)
def test_dynamic_combo_option_key_grammar_accepts_printable_ascii_tokens(
    key: str,
) -> None:
    assert OPTION_KEY_PATTERN.fullmatch(key)
    assert DynamicComboOption(key).key == key


def test_outputs_helper_rejects_typos_at_return_site() -> None:
    with pytest.raises(NodeOutputError) as excinfo:
        TwoOutputs.outputs(first=1, secondd=2)
    message = str(excinfo.value)
    assert "test.two_outputs" in message
    assert "missing outputs: second" in message
    assert "undeclared outputs: secondd" in message


def test_schema_cached_per_subclass() -> None:
    assert TwoOutputs.schema() is TwoOutputs.schema()

    class Child(TwoOutputs):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(node_type="test.child")

    # Subclass gets its own schema, not the parent's cached one.
    assert Child.schema().node_type == "test.child"
    assert TwoOutputs.schema().node_type == "test.two_outputs"
