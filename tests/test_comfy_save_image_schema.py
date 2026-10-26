"""Schema contract for the native ComfyUI SaveImage node."""

from typing import cast

from dinkster_api.v1 import ASSET_TYPE, MountWidget, TypeExpr
from dinkster_nodes_comfy.image_file_tools import (
    LoadImage,
    LoadImageMask,
    LoadImageOutput,
    PreviewImage,
    SaveImage,
)
from dinkster_schema import schema_to_wire


def test_save_image_preserves_fixed_inputs_and_appends_mount_widget() -> None:
    schema = SaveImage.schema()

    assert schema.node_type == "SaveImage"
    assert schema.display_name == "Save Image"
    assert schema.category == "image"
    assert tuple(spec.id for spec in schema.inputs) == ("images", "filename_prefix", "mount")
    assert schema.inputs[1].required
    assert schema.inputs[1].default == "ComfyUI"
    assert schema.inputs[1].widget is None
    assert schema.inputs[0].doc == "The images to save."
    assert schema.inputs[1].doc.startswith("The prefix for the file to save.")
    assert not schema.inputs[2].required
    assert schema.inputs[2].default is None
    assert schema.inputs[2].widget == MountWidget()
    assert schema.inputs[2].adjacent_to == "filename_prefix"
    assert tuple(spec.id for spec in schema.outputs) == ("images",)
    assert schema.output_node
    assert not schema.idempotent
    assert schema.search_terms == (
        "save",
        "save image",
        "export image",
        "output image",
        "write image",
        "download",
    )

    wire = schema_to_wire(schema)
    interface = cast(list[dict[str, object]], wire["interface"])
    filename_prefix = next(entry for entry in interface if entry["id"] == "filename_prefix")
    assert "widget" not in filename_prefix
    mount = next(entry for entry in interface if entry["id"] == "mount")
    assert mount["widget"] == {"type": "MOUNT"}
    assert mount["adjacentTo"] == "filename_prefix"


def test_preview_image_uses_the_public_image_preview_output() -> None:
    schema = PreviewImage.schema()

    assert tuple(spec.id for spec in schema.inputs) == ("images",)
    assert tuple(spec.id for spec in schema.outputs) == ("images",)
    assert schema.outputs[0].preview
    assert schema.output_node
    assert schema.search_terms == (
        "preview",
        "preview image",
        "show image",
        "view image",
        "display image",
        "image viewer",
    )


def test_load_image_output_preserves_fixed_description() -> None:
    assert LoadImageOutput.schema().description == (
        "Load an image from the output folder. When the refresh button is clicked, the node will "
        "update the image list and automatically select the first image, allowing for easy "
        "iteration."
    )


def test_image_file_inputs_use_the_bound_asset_contract() -> None:
    for node in (LoadImage, LoadImageMask, LoadImageOutput):
        image = node.schema().inputs[0]
        assert image.type == TypeExpr.concrete(ASSET_TYPE)
        assert image.source_filename is not None
