"""Numeric conversions preserve integer precision and reference failures."""

from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path
from typing import cast

import pytest
from dinkster_api.v1 import schema_to_wire
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.utilities import ComfyNumberConvert, SeedNode

from dinkster.compose import PackSpec, ServingComposer


@pytest.mark.parametrize(
    ("value", "float_value", "int_value"),
    [
        (True, 1.0, 1),
        (False, 0.0, 0),
        (-7, -7.0, -7),
        (2**53 + 1, 9007199254740992.0, 9007199254740993),
        (" 9007199254740993 ", 9007199254740992.0, 9007199254740993),
        ("9007199254740993.0", 9007199254740992.0, 9007199254740992),
        (" -1.9 ", -1.9, -1),
        (-1.9, -1.9, -1),
        ("2e3", 2000.0, 2000),
        (-0.0, -0.0, 0),
        ("-0.0", -0.0, 0),
    ],
)
def test_conversion_preserves_integer_text_and_float_semantics(
    value: object, float_value: float, int_value: int
) -> None:
    result = ComfyNumberConvert.execute(value=value)
    assert result == {"FLOAT": float_value, "INT": int_value}
    assert type(result["FLOAT"]) is float
    assert type(result["INT"]) is int
    assert math.copysign(1, cast(float, result["FLOAT"])) == math.copysign(1, float_value)


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        (" \t", ValueError, "Cannot convert empty string to number."),
        ("broken", ValueError, "Cannot convert string to number: 'broken'"),
        ("NaN", ValueError, "Cannot convert non-finite value to number: nan"),
        ("-inf", ValueError, "Cannot convert non-finite value to number: -inf"),
        (None, TypeError, "Unsupported input type: NoneType"),
        ([], TypeError, "Unsupported input type: list"),
        (float("inf"), OverflowError, "cannot convert float infinity to integer"),
        (float("nan"), ValueError, "cannot convert float NaN to integer"),
        (10**400, OverflowError, "int too large to convert to float"),
    ],
)
def test_invalid_conversion_preserves_error_kind_and_message(
    value: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error) as raised:
        ComfyNumberConvert.execute(value=value)
    assert str(raised.value) == message


def test_seed_preserves_large_integer_and_positional_controller_metadata() -> None:
    assert SeedNode.execute(seed=sys.maxsize) == {"seed": 9223372036854775807}
    wire = schema_to_wire(SeedNode.schema())
    interface = cast(list[dict[str, object]], wire["interface"])
    assert interface[0]["id"] == "seed"
    assert "default" not in interface[0]
    assert interface[0]["widget"] == {
        "type": "NUMBER",
        "min": 0,
        "max": "9223372036854775807",
        "controlAfterGenerate": "fixed",
    }
    assert interface[1]["id"] == "seed"


def test_isolated_manifest_converts_text_and_linked_seed_without_rounding_integer() -> None:
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
                        "seed": GraphNode("SeedNode", {"seed": 9007199254740993}),
                        "linked": GraphNode("ComfyNumberConvert", {"value": Link("seed", "seed")}),
                        "source": GraphNode("PrimitiveString", {"value": " 9007199254740993 "}),
                        "text": GraphNode(
                            "ComfyNumberConvert", {"value": Link("source", "STRING")}
                        ),
                    }
                ),
                ["linked", "text"],
            )
            for name in ("linked", "text"):
                assert result.outputs[name]["FLOAT"].resolve() == 9007199254740992.0
                assert result.outputs[name]["INT"].resolve() == 9007199254740993
        finally:
            await composer.close()

    asyncio.run(scenario())
