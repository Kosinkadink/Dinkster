"""Dataset image transform schemas preserve the ComfyUI workflow contract."""

from typing import Any, cast

from dinkster_api.v1 import schema_to_wire
from dinkster_nodes_comfy.dataset_images import DATASET_IMAGE_NODES


def test_dataset_image_nodes_have_exact_core_schemas() -> None:
    expected = (
        (
            "ResizeImagesByShorterEdge",
            "Resize Images by Shorter Edge (DEPRECATED)",
            "image/transform",
            ("images", "shorter_edge"),
        ),
        (
            "ResizeImagesByLongerEdge",
            "Resize Images by Longer Edge (DEPRECATED)",
            "image/transform",
            ("images", "longer_edge"),
        ),
        (
            "CenterCropImages",
            "Crop Image (Center)",
            "image/transform",
            ("images", "width", "height"),
        ),
        (
            "RandomCropImages",
            "Crop Image (Random)",
            "image/transform",
            ("images", "width", "height", "seed"),
        ),
        ("NormalizeImages", "Normalize Image Colors", "image/color", ("images", "mean", "std")),
        ("AdjustBrightness", "Adjust Brightness", "image/adjustments", ("images", "factor")),
        ("AdjustContrast", "Adjust Contrast", "image/adjustments", ("images", "factor")),
    )
    assert len(DATASET_IMAGE_NODES) == len(expected)
    for node, expected_schema in zip(DATASET_IMAGE_NODES, expected, strict=True):
        schema = node.schema()
        assert (schema.node_type, schema.display_name, schema.category) == expected_schema[:3]
        assert tuple(input_spec.id for input_spec in schema.inputs) == expected_schema[3]
        assert tuple(output_spec.id for output_spec in schema.outputs) == ("images",)
        assert tuple(output_spec.type.types for output_spec in schema.outputs) == (("IMAGE",),)

    random_crop = schema_to_wire(DATASET_IMAGE_NODES[3].schema())
    interface = cast(list[dict[str, Any]], random_crop["interface"])
    seed = next(item for item in interface if item["id"] == "seed")
    assert seed["default"] == 0
    assert seed["widget"] == {
        "type": "NUMBER",
        "min": 0,
        "max": str(0xFFFFFFFFFFFFFFFF),
    }
