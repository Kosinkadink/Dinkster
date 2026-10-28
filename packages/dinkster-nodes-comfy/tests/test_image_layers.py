"""Layer document values, sparse fields, and serialized native schema."""

from __future__ import annotations

import math

import numpy as np
import pytest
from dinkster_api.v1 import schema_to_wire
from dinkster_nodes_comfy.image_layers import AddLayer


def test_add_layer_appends_without_reordering_or_changing_media() -> None:
    image = np.arange(24, dtype=np.float16).reshape(2, 1, 3, 4)
    mask = np.array([[[0.0, 0.25, 1.0]]], dtype=np.float32)
    previous_item = {"image": image, "z_index": 9, "custom": ["retained"]}
    previous = {"version": 7, "layers": [previous_item], "canvas": [13.9, 7.2], "extra": 42}
    result = AddLayer.execute(
        image=image,
        mask=mask,
        layers=previous,
        name="Upper",
        x=-3,
        y=5,
        opacity=0.375,
        blend_mode="multiply",
        rotation=-90,
        width=17,
        height=11,
        z_index=-4,
        flip_h=True,
    )["LAYERS"]
    assert result is not previous
    assert result.keys() == {"version", "layers", "canvas"}
    assert result["version"] == 1 and result["canvas"] == (13, 7)
    assert result["layers"][0] is previous_item
    item = result["layers"][1]
    assert item["image"] is image and item["mask"] is mask
    assert {key: value for key, value in item.items() if key not in ("image", "mask")} == {
        "type": "raster",
        "name": "Upper",
        "x": -3,
        "y": 5,
        "z_index": -4,
        "opacity": 0.375,
        "blend_mode": "multiply",
        "rotation": -math.pi / 2,
        "w": 17,
        "h": 11,
        "flip_h": True,
    }
    assert len(previous["layers"]) == 1 and previous["canvas"] == [13.9, 7.2]
    np.testing.assert_array_equal(image, np.arange(24, dtype=np.float16).reshape(2, 1, 3, 4))
    np.testing.assert_array_equal(mask, [[[0, 0.25, 1]]])


@pytest.mark.parametrize("canvas", ([True, 7], ["13", 7], [13, 0], [13, 7, 2]))
def test_add_layer_omits_invalid_canvas_and_default_fields(canvas: object) -> None:
    image = np.array([[[[0.125, 0.25, 0.75]]]], dtype=np.float32)
    result = AddLayer.execute(image=image, layers={"layers": None, "canvas": canvas})["LAYERS"]
    assert result.keys() == {"version", "layers"}
    assert len(result["layers"]) == 1
    item = result["layers"][0]
    assert item.keys() == {"image", "type", "x", "y", "z_index"}
    assert item["image"] is image
    assert (item["x"], item["y"], item["z_index"], item["type"]) == (0, 0, 0, "raster")


def test_add_layer_serialized_schema_preserves_optional_controls() -> None:
    wire = schema_to_wire(AddLayer.schema())
    inputs = [item for item in wire["interface"] if item["role"] == "input"]
    assert [item["id"] for item in inputs] == [
        "layers",
        "image",
        "mask",
        "name",
        "x",
        "y",
        "opacity",
        "blend_mode",
        "rotation",
        "width",
        "height",
        "z_index",
        "flip_h",
        "flip_v",
    ]
    assert [item["id"] for item in inputs if item["required"]] == ["image"]
    assert inputs[0]["type"] == {"kind": "concrete", "types": ["LAYERS"]}
    assert inputs[3]["widget"] == {"type": "STRING", "multiline": False}
    assert inputs[4]["widget"]["min"] == -16_384
    assert inputs[8]["widget"]["step"] == 1.0
    assert inputs[7]["default"] == "normal"
    assert len(inputs[7]["widget"]["options"]) == 26
    assert all(item["doc"] for item in inputs)
    assert [item["id"] for item in wire["interface"] if item["role"] == "output"] == ["LAYERS"]
