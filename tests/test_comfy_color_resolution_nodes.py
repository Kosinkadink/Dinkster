"""Public color and resolution contracts without a Torch dependency."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest
from dinkster_api.v1 import schema_from_wire, schema_to_wire
from dinkster_graph import Graph, GraphNode
from dinkster_nodes_comfy.color_resolution import ColorToRGBInt, ResolutionSelector

from dinkster.compose import PackSpec, ServingComposer


def test_color_and_resolution_schemas_preserve_public_types_and_positional_inputs() -> None:
    color = cast(dict[str, Any], schema_to_wire(ColorToRGBInt.schema()))
    assert color["interface"][0]["type"] == {"kind": "concrete", "types": ["COLOR"]}
    assert color["interface"][0]["default"] == "#ffffff"
    assert color["interface"][0]["widget"] == {"type": "COLOR"}
    resolution = cast(dict[str, Any], schema_to_wire(ResolutionSelector.schema()))
    inputs = [item for item in resolution["interface"] if item["role"] == "input"]
    assert [item["id"] for item in inputs] == ["aspect_ratio", "megapixels", "preview", "multiple"]
    assert inputs[2]["required"] is False
    assert inputs[2]["type"] == {"kind": "concrete", "types": ["RESOLUTION_PREVIEW"]}
    assert inputs[2]["widget"] == {
        "type": "RESOLUTION_PREVIEW",
        "ratio_widget": "aspect_ratio",
        "megapixels_widget": "megapixels",
        "multiple_widget": "multiple",
    }
    for node in (ColorToRGBInt, ResolutionSelector):
        wire = schema_to_wire(node.schema())
        assert schema_to_wire(schema_from_wire(wire)) == wire


def test_color_values_preserve_case_alpha_and_reference_rejection() -> None:
    assert ColorToRGBInt.execute(color="#aB12CD80") == {
        "rgb_int": 11211469,
        "hex": "#aB12CD",
        "alpha": 128 / 255,
    }
    assert ColorToRGBInt.execute(color="#01020f") == {
        "rgb_int": 66063,
        "hex": "#01020f",
        "alpha": 1.0,
    }
    assert ColorToRGBInt.execute(color="#12_345") == {
        "rgb_int": 16777215,
        "hex": "#12_345",
        "alpha": 1.0,
    }
    assert ColorToRGBInt.execute(color="#00000000")["alpha"] == 0.0
    for invalid in ("", "red", "#abc", "001122", "#12345g", "#1122334455"):
        with pytest.raises(ValueError, match="Color must be in format #RRGGBB or #RRGGBBAA"):
            ColorToRGBInt.execute(color=invalid)


def test_resolution_rounds_to_requested_multiple_and_ignores_preview_payload() -> None:
    assert ResolutionSelector.execute(aspect_ratio="1:1 (Square)", megapixels=1, multiple=20) == {
        "width": 1020,
        "height": 1020,
    }
    assert ResolutionSelector.execute(
        aspect_ratio="1:1 (Square)", megapixels=0.1, multiple=8, preview={"width": 1, "height": 2}
    ) == {"width": 320, "height": 320}
    assert ResolutionSelector.execute(
        aspect_ratio="21:9 (Ultrawide)", megapixels=1, multiple=8
    ) == {"width": 1568, "height": 672}
    with pytest.raises(KeyError):
        ResolutionSelector.execute(aspect_ratio="unrecognized", megapixels=1, multiple=8)


@pytest.mark.parametrize("in_process", (True, False))
def test_color_and_resolution_execute_twice_through_the_native_pack(in_process: bool) -> None:
    async def scenario() -> None:
        manifest = (
            Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-comfy/dinkster-pack.toml"
        )
        composer = ServingComposer()
        try:
            await composer.add_pack(PackSpec(manifest=manifest, in_process=in_process))
            engine = composer.composition.make_engine(lambda _event: None)
            graph = Graph(
                nodes={
                    "color": GraphNode("ColorToRGBInt", {"color": "#aB12CD80"}),
                    "resolution": GraphNode(
                        "ResolutionSelector",
                        {"aspect_ratio": "21:9 (Ultrawide)", "megapixels": 1.0, "multiple": 8},
                    ),
                }
            )
            for _ in range(2):
                result = await engine.run(graph, ["color", "resolution"])
                assert result.outputs["color"]["rgb_int"].resolve() == 11211469
                assert result.outputs["color"]["hex"].resolve() == "#aB12CD"
                assert result.outputs["color"]["alpha"].resolve() == 128 / 255
                assert result.outputs["resolution"]["width"].resolve() == 1568
                assert result.outputs["resolution"]["height"].resolve() == 672
        finally:
            await composer.close()

    asyncio.run(scenario())
