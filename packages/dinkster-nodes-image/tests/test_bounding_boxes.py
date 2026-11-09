"""Bounding-box canvas execution with asymmetric coordinates and saved edits."""

import json
from pathlib import Path

import numpy as np
import pytest
from dinkster_api.v1 import Node, schema_to_wire
from dinkster_image.bounding_boxes import CreateBoundingBoxes
from dinkster_image.layers_from_bounding_boxes import LayersFromBoundingBoxes

from tools.compare_nodes import compare


@pytest.mark.parametrize(
    ("node", "node_id"),
    (
        (CreateBoundingBoxes, "dinkster.image.create_bounding_boxes"),
        (LayersFromBoundingBoxes, "dinkster.image.layers_from_bounding_boxes"),
    ),
)
def test_box_schemas_match_reference_except_frozen_id(node: type[Node], node_id: str) -> None:
    reference = json.loads(
        (Path(__file__).parents[3] / "tools/data/comfy_core_reference.json").read_text()
    )
    row = next(item for item in reference["nodes"] if item["class_name"] == node.__name__)
    schema = node.define_schema()
    assert schema.node_type == node_id
    assert node.__name__ in schema.search_terms
    assert schema.aliases == ()
    wire = schema_to_wire(schema)
    wire["nodeType"] = node.__name__
    assert compare(row, wire) == []


def test_canvas_initialization_retains_edits_until_upstream_changes():
    incoming = {"x": 17, "y": 29, "width": 120, "height": 35}
    edited = {"x": 71, "y": 13, "width": -45, "height": 77}
    first = CreateBoundingBoxes.execute(
        width=400, height=200, bboxes=json.dumps(incoming), editor_state=[edited]
    )
    assert first["bboxes"] == [[incoming]]
    assert first["elements"] == [{"type": "obj", "bbox": [145, 42, 320, 342], "desc": ""}]

    replay = CreateBoundingBoxes.execute(
        width=400,
        height=200,
        bboxes=[incoming],
        last_incoming=[incoming],
        editor_state=[edited],
    )
    assert replay["bboxes"] == [[{"x": 26, "y": 13, "width": 45, "height": 77}]]
    assert replay["elements"] == [{"type": "obj", "bbox": [65, 65, 450, 178], "desc": ""}]
    changed = {"x": 33, "y": 12, "width": 50, "height": 25}
    assert CreateBoundingBoxes.execute(
        width=400,
        height=200,
        bboxes=[[changed]],
        last_incoming=[incoming],
        editor_state=[edited],
    )["bboxes"] == [[changed]]


def test_canvas_elements_preserve_text_palette_and_reverse_coordinates():
    result = CreateBoundingBoxes.execute(
        width=400,
        height=200,
        bboxes=[
            {
                "type": "text",
                "bbox": [800, 750, 100, 250],
                "text": "Hello",
                "desc": "Sign",
                "color_palette": ["#ff0000", "#aabbcc"],
            }
        ],
    )
    assert result["bboxes"] == [[{"x": 100, "y": 20, "width": 200, "height": 140}]]
    assert result["elements"] == [
        {
            "type": "text",
            "bbox": [100, 250, 800, 750],
            "text": "Hello",
            "desc": "Sign",
            "color_palette": ["#FF0000", "#AABBCC"],
        }
    ]
    preview = np.asarray(result["preview"])
    np.testing.assert_array_equal(preview[0, 20, 150], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(
        preview[0, 20, 250], np.asarray([170, 187, 204], dtype=np.float32) / 255
    )
    np.testing.assert_array_equal(preview[0, 180, 350], np.full(3, np.float32(32 / 255)))


def test_canvas_empty_input_keeps_saved_state_and_background_preview():
    background = np.ones((1, 30, 70, 3), dtype=np.float32)
    result = CreateBoundingBoxes.execute(
        width=140, height=60, background=background, editor_state=[], bboxes=" "
    )
    assert result["bboxes"] == []
    assert result["elements"] == []
    np.testing.assert_array_equal(result["preview"], np.full((1, 30, 70, 3), np.float32(63 / 255)))


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("not JSON", "bboxes string input is not valid JSON"),
        ({"unknown": 1}, "bboxes dict must be a bounding box"),
        ([123], "bboxes list must contain bounding boxes or elements"),
        (4, "bboxes input must be bounding boxes, elements, or a JSON string"),
        ([{"bbox": [0, 0, "bad", 1]}], "bboxes element 'bbox' must contain four numbers"),
    ],
)
def test_canvas_reports_reference_input_errors(value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CreateBoundingBoxes.execute(width=400, height=200, bboxes=value)
