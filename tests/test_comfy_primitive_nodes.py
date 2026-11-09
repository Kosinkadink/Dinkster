"""Primitive nodes preserve the supplied value, without numeric or text coercion."""

from __future__ import annotations

import math
import sys
from typing import cast

import pytest
from dinkster_api.v1 import schema_to_wire
from dinkster_nodes_comfy.primitive import (
    PrimitiveBoolean,
    PrimitiveFloat,
    PrimitiveInt,
    PrimitiveString,
    PrimitiveStringMultiline,
)


@pytest.mark.parametrize("value", ["", "  text\r\nnext\t", "\U0001f600e\u0301"])
def test_text_preserves_whitespace_and_codepoints(value: str) -> None:
    assert PrimitiveString.execute(value=value) == {"STRING": value}
    assert PrimitiveStringMultiline.execute(value=value) == {"STRING": value}
    for node, multiline in ((PrimitiveString, False), (PrimitiveStringMultiline, True)):
        interface = cast(list[dict[str, object]], schema_to_wire(node.schema())["interface"])
        assert interface[0]["widget"] == {"type": "STRING", "multiline": multiline}


@pytest.mark.parametrize("value", [-sys.maxsize, -17, 0, 2**53 + 1, sys.maxsize])
def test_integer_preserves_large_and_negative_values(value: int) -> None:
    result = PrimitiveInt.execute(value=value)
    assert result == {"INT": value}
    assert type(result["INT"]) is int


@pytest.mark.parametrize("value", [-0.0, -0.125, 0.1, 1e100])
def test_float_preserves_fraction_and_sign(value: float) -> None:
    result = PrimitiveFloat.execute(value=value)
    assert result == {"FLOAT": value}
    assert type(result["FLOAT"]) is float
    assert math.copysign(1.0, result["FLOAT"]) == math.copysign(1.0, value)


@pytest.mark.parametrize("value", [True, False])
def test_boolean_remains_boolean(value: bool) -> None:
    result = PrimitiveBoolean.execute(value=value)
    assert result == {"BOOLEAN": value}
    assert type(result["BOOLEAN"]) is bool


@pytest.mark.parametrize("node", [PrimitiveInt, PrimitiveFloat])
def test_numeric_controller_and_bounds_survive_serialization(
    node: type[PrimitiveInt] | type[PrimitiveFloat],
) -> None:
    wire = schema_to_wire(node.schema())
    interface = cast(list[dict[str, object]], wire["interface"])
    assert interface[0]["widget"] == {
        "type": "NUMBER",
        "min": "-9223372036854775807" if node is PrimitiveInt else -9.223372036854776e18,
        "max": "9223372036854775807" if node is PrimitiveInt else 9.223372036854776e18,
        **({"controlAfterGenerate": "fixed"} if node is PrimitiveInt else {"step": 0.1}),
    }
