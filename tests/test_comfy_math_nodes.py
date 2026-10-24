"""Math expressions retain ordered values, numeric precision and safety bounds."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import cast

import pytest
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.math import ComfyMathExpression
from simpleeval import FeatureNotAvailable

from dinkster.compose import PackSpec, ServingComposer


@pytest.mark.parametrize(
    ("expression", "values", "float_value", "int_value", "bool_value"),
    [
        ("a - b * c", {"a": 9, "b": 2, "c": 3}, 3.0, 3, True),
        ("sum(values)", {"b": -8, "a": 3}, -5.0, -5, True),
        ("sum(a, b, c)", {"a": 7, "b": -9, "c": 2}, 0.0, 0, False),
        ("values[0] - values[1]", {"z": 3, "a": 11}, -8.0, -8, True),
        ("a", {"a": 9007199254740993}, 9007199254740992.0, 9007199254740993, True),
        ("a / b", {"a": -7, "b": 2}, -3.5, -3, True),
        ("a // b", {"a": -7, "b": 2}, -4.0, -4, True),
        ("a > b", {"a": 2, "b": 8}, 0.0, 0, False),
        ("a", {"a": -0.0}, -0.0, 0, False),
        ("pow(a, 4000)", {"a": -1}, 1.0, 1, True),
        ("pow(a, -4000)", {"a": 1}, 1.0, 1, True),
        ("sin(a) + cos(a)", {"a": 0}, 1.0, 1, True),
        ("round(a) + ceil(b) + floor(c)", {"a": 2.5, "b": -1.2, "c": -1.2}, -1.0, -1, True),
        ("log(a) + log2(b) + log10(c)", {"a": 1, "b": 8, "c": 100}, 5.0, 5, True),
        ("min(values) + max(values)", {"a": 3, "b": -9, "c": 7}, -2.0, -2, True),
        ("int(a) + float(b) + abs(c)", {"a": -1.9, "b": 2, "c": -3}, 4.0, 4, True),
    ],
)
def test_expression_results(
    expression: str,
    values: dict[str, object],
    float_value: float,
    int_value: int,
    bool_value: bool,
) -> None:
    result = ComfyMathExpression.execute(expression=expression, values=values)
    assert result == {"FLOAT": float_value, "INT": int_value, "BOOL": bool_value}
    assert type(result["FLOAT"]) is float
    assert type(result["INT"]) is int
    assert type(result["BOOL"]) is bool
    assert math.copysign(1, cast(float, result["FLOAT"])) == math.copysign(1, float_value)


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        (" \t", "Expression cannot be empty."),
        ("pow(a, 4001)", "Exponent 4001 exceeds maximum allowed (4000)"),
        ("pow(a, -4001)", "Exponent -4001 exceeds maximum allowed (4000)"),
        (
            "'text'",
            "Math Expression ''text'' must evaluate to a numeric result, got str: 'text'",
        ),
        ("1e309", "Math Expression '1e309' produced a non-finite result: inf"),
    ],
)
def test_expression_rejects_invalid_results_and_exponent_overflow(
    expression: str, message: str
) -> None:
    with pytest.raises(ValueError) as raised:
        ComfyMathExpression.execute(expression=expression, values={"a": 0})
    assert str(raised.value) == message


def test_float_overflow_does_not_truncate_or_accept_large_result() -> None:
    with pytest.raises(ValueError) as raised:
        ComfyMathExpression.execute(expression="a", values={"a": 10**400})
    assert str(raised.value) == (
        "Math Expression 'a' produced a result too large to represent as a float: " + str(10**400)
    )


def test_value_named_values_is_replaced_by_ordered_input_list() -> None:
    result = ComfyMathExpression.execute(expression="sum(values)", values={"values": 7, "a": -3})
    assert result == {"FLOAT": 4.0, "INT": 4, "BOOL": True}


def test_expression_keeps_simpleeval_attribute_access_restrictions() -> None:
    with pytest.raises(FeatureNotAvailable):
        ComfyMathExpression.execute(expression="a.__class__", values={"a": 1})


def test_isolated_manifest_evaluates_ordered_numeric_union_family() -> None:
    async def scenario() -> None:
        composer = ServingComposer()
        try:
            manifest = (
                Path(__file__).resolve().parents[1]
                / "packages/dinkster-nodes-comfy/dinkster-pack.toml"
            )
            await composer.add_pack(PackSpec(manifest=manifest))
            result = await composer.composition.make_engine(lambda _event: None).run(
                Graph(
                    nodes={
                        "int": GraphNode("PrimitiveInt", {"value": 9}),
                        "float": GraphNode("PrimitiveFloat", {"value": -2.5}),
                        "bool": GraphNode("PrimitiveBoolean", {"value": True}),
                        "math": GraphNode(
                            "ComfyMathExpression",
                            {
                                "expression": "a - b * c",
                                "values.a": Link("int", "INT"),
                                "values.b": Link("float", "FLOAT"),
                                "values.c": Link("bool", "BOOLEAN"),
                            },
                        ),
                    }
                ),
                ["math"],
            )
            assert result.outputs["math"]["FLOAT"].resolve() == 11.5
            assert result.outputs["math"]["INT"].resolve() == 11
            assert result.outputs["math"]["BOOL"].resolve() is True
        finally:
            await composer.close()

    asyncio.run(scenario())
