"""Schema contracts for native tile and mask preview nodes."""

from typing import Any, cast

from dinkster_api.v1 import IMAGE_TYPE, MASK_TYPE, schema_to_wire
from dinkster_nodes_comfy.image_tiles import ImageMergeTileList, SplitImageToTileList
from dinkster_nodes_comfy.mask_preview import MaskPreview


def test_tile_node_schemas_preserve_types_defaults_and_aliases() -> None:
    split = cast("dict[str, Any]", schema_to_wire(SplitImageToTileList.schema()))
    merge = cast("dict[str, Any]", schema_to_wire(ImageMergeTileList.schema()))
    assert split["searchTerms"] == ["split image", "tile image", "slice image"]
    assert merge["searchTerms"] == split["searchTerms"]
    assert split["interface"][0]["type"]["types"] == [IMAGE_TYPE]
    assert split["interface"][-1]["type"] == {
        "kind": "list",
        "element": {"kind": "concrete", "types": [IMAGE_TYPE]},
    }
    assert [item["default"] for item in split["interface"][1:4]] == [1024, 1024, 128]
    assert [item["default"] for item in merge["interface"][1:4]] == [1024, 1024, 128]


def test_mask_preview_schema_declares_preview_output() -> None:
    wire = cast("dict[str, Any]", schema_to_wire(MaskPreview.schema()))
    assert wire["nodeType"] == "MaskPreview"
    assert wire["emitsPreviews"] is True
    assert wire["interface"] == [
        {
            "id": "mask",
            "required": True,
            "role": "input",
            "type": {"kind": "concrete", "types": [MASK_TYPE]},
        },
        {
            "id": "mask",
            "preview": True,
            "role": "output",
            "type": {"kind": "concrete", "types": [MASK_TYPE]},
        },
    ]
