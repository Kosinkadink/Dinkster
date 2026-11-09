"""Per-frame placement, cropping and inherited canvas behavior."""

import copy
from typing import Any

import numpy as np
import pytest
from dinkster_image.layers_from_bounding_boxes import LayersFromBoundingBoxes


def test_layer_frames_crop_content_preserve_mask_and_append_without_mutation():
    image = np.arange(3 * 5 * 7 * 3).reshape(3, 5, 7, 3)
    mask = np.arange(35).reshape(1, 5, 7)
    previous = {"version": 1, "canvas": [320, 180], "layers": [{"z_index": 7}]}
    before = copy.deepcopy(previous)
    result: Any = LayersFromBoundingBoxes.execute(
        image=image,
        mask=mask,
        layers=previous,
        bboxes=[
            {
                "x": 13,
                "y": 29,
                "width": 100,
                "height": 80,
                "metadata": {"content_rect": [2, 1, 3, 2], "name": "Crop", "z_index": 4},
            },
            {"x": 61, "y": 17, "width": 33, "height": 19, "metadata": {"desc": "Scaled"}},
        ],
    )["LAYERS"]
    assert result["canvas"] == (320, 180)
    assert previous == before
    cropped, scaled, missing = result["layers"][1:]
    assert {key: value for key, value in cropped.items() if key not in ("image", "mask")} == {
        "type": "raster",
        "x": 15,
        "y": 30,
        "z_index": 4,
        "name": "Crop",
    }
    np.testing.assert_array_equal(cropped["image"], image[:1, 1:3, 2:5])
    np.testing.assert_array_equal(cropped["mask"], mask[:, 1:3, 2:5])
    assert {key: value for key, value in scaled.items() if key not in ("image", "mask")} == {
        "type": "raster",
        "x": 61,
        "y": 17,
        "z_index": 9,
        "w": 33,
        "h": 19,
        "name": "Scaled",
    }
    np.testing.assert_array_equal(scaled["image"], image[1:2])
    np.testing.assert_array_equal(missing["image"], image[2:3])
    np.testing.assert_array_equal(missing["mask"], mask)
    assert (missing["x"], missing["y"], missing["z_index"]) == (0, 0, 10)


def test_layer_normalized_elements_need_canvas_and_resolve_asymmetric_dimensions():
    image = np.zeros((1, 3, 5, 3))
    element = {"bbox": [100, 250, 800, 750], "desc": "Element"}
    with pytest.raises(ValueError, match="normalized element boxes need canvas_width"):
        LayersFromBoundingBoxes.execute(image=image, bboxes=[element])
    document: Any = LayersFromBoundingBoxes.execute(
        image=image, bboxes=[element], canvas_width=400, canvas_height=200
    )["LAYERS"]
    assert document["canvas"] == (400, 200)
    layer = document["layers"][0]
    assert (layer["x"], layer["y"], layer["w"], layer["h"], layer["name"]) == (
        100,
        20,
        200,
        140,
        "Element",
    )


def test_layer_crop_disabled_preserves_scaling_and_short_mask_does_not_repeat():
    image = np.arange(3 * 4 * 6 * 3).reshape(3, 4, 6, 3)
    mask = np.arange(2 * 4 * 6).reshape(2, 4, 6)
    result: Any = LayersFromBoundingBoxes.execute(
        image=image,
        mask=mask,
        crop_to_content=False,
        bboxes={
            "x": 8,
            "y": 9,
            "width": 15,
            "height": 17,
            "metadata": {"content_rect": [1, 2, 2, 1]},
        },
    )["LAYERS"]
    layers = result["layers"]
    np.testing.assert_array_equal(layers[0]["image"], image[:1])
    assert (layers[0]["x"], layers[0]["y"], layers[0]["w"], layers[0]["h"]) == (8, 9, 15, 17)
    np.testing.assert_array_equal(layers[1]["mask"], mask[1:2])
    assert "mask" not in layers[2]
