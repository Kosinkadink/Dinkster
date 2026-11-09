"""Schema contracts for the native image crop primitives."""

from __future__ import annotations

from typing import Any, cast

from dinkster_api.v1 import (
    BOUNDING_BOX_TYPE,
    COLOR_TYPE,
    CORE_INT,
    IMAGE_TYPE,
    MASK_TYPE,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_nodes_comfy.image_crop import ImageCropToMask, ImageCropV2, PrimitiveBoundingBox


def _entry(wire: dict[str, object], entry_id: str) -> dict[str, Any]:
    for entry in cast("list[dict[str, Any]]", wire["interface"]):
        if entry["id"] == entry_id:
            return entry
    raise AssertionError(f"missing {entry_id}")


def test_mask_crop_schema_uses_portable_media_and_color() -> None:
    schema = ImageCropToMask.schema()
    wire = schema_to_wire(schema)
    assert schema_from_wire(wire) == schema
    for name, type_id in (("images", IMAGE_TYPE), ("masks", MASK_TYPE), ("background", COLOR_TYPE)):
        assert _entry(wire, name)["type"] == {"kind": "concrete", "types": [type_id]}
    assert _entry(wire, "background")["default"] == "#000000"
    assert _entry(wire, "width")["widget"] == {"type": "NUMBER", "min": 64, "max": 4096, "step": 8}
    assert _entry(wire, "grow_mask")["widget"] == {
        "type": "NUMBER",
        "min": -32,
        "max": 32,
        "step": 1,
    }


def test_image_crop_v2_schema_uses_canonical_image_and_bounding_box_contracts() -> None:
    schema = ImageCropV2.schema()
    wire = schema_to_wire(schema)

    assert schema.node_type == "ImageCropV2"
    assert schema_from_wire(wire) == schema
    assert schema.outputs[0].preview is True
    assert schema.emits_previews is True
    assert _entry(wire, "image")["type"] == {"kind": "concrete", "types": [IMAGE_TYPE]}
    crop_region = _entry(wire, "crop_region")
    assert crop_region["type"] == {"kind": "concrete", "types": [BOUNDING_BOX_TYPE]}
    assert crop_region["required"] is True
    assert crop_region["default"] == {"x": 0, "y": 0, "width": 512, "height": 512}
    assert crop_region["widget"] == {"type": "BOUNDING_BOX", "component": "ImageCrop"}


def test_bounding_box_schema_uses_canonical_type_and_reference_defaults() -> None:
    schema = PrimitiveBoundingBox.schema()
    wire = schema_to_wire(schema)

    assert schema.node_type == "PrimitiveBoundingBox"
    assert schema_from_wire(wire) == schema
    for name, default, minimum in (("x", 0, 0), ("y", 0, 0), ("width", 512, 1), ("height", 512, 1)):
        entry = _entry(wire, name)
        assert entry["type"] == {"kind": "concrete", "types": [CORE_INT]}
        assert entry["required"] is True
        assert entry["default"] == default
        assert entry["widget"] == {"type": "NUMBER", "min": minimum, "max": 16_384}
    assert _entry(wire, "BOUNDING_BOX")["type"] == {
        "kind": "concrete",
        "types": [BOUNDING_BOX_TYPE],
    }
