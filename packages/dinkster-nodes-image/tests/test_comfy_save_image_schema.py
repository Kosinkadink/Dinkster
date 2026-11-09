"""Schema contract for the native ComfyUI SaveImage node."""

from typing import cast

from dinkster_api.v1 import (
    ASSET_TYPE,
    AssetWidget,
    ComboWidget,
    MountWidget,
    NumberWidget,
    SourceFilenameSpec,
    StringWidget,
    TypeExpr,
)
from dinkster_image.ported.image_file_tools import (
    LoadImage,
    LoadImageMask,
    LoadImageOutput,
    Painter,
    PreviewImage,
    SaveAnimatedPNG,
    SaveAnimatedWEBP,
    SaveImage,
)
from dinkster_image.ported.save_image_advanced import SaveImageAdvanced
from dinkster_schema import schema_to_wire


def test_save_image_preserves_fixed_inputs_and_appends_mount_widget() -> None:
    schema = SaveImage.schema()

    assert schema.node_type == "dinkster.image.save_image"
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
        "SaveImage",
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
        "PreviewImage",
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


def test_painter_retains_order_with_an_optional_granted_mask() -> None:
    from dinkster_image import NODES as COMFY_NODES
    from dinkster_schema import schema_from_wire

    schema = Painter.schema()
    assert Painter in COMFY_NODES
    assert schema.node_type == "dinkster.image.painter" and not schema.aliases
    assert tuple(spec.id for spec in schema.inputs) == (
        "image",
        "mask",
        "width",
        "height",
        "bg_color",
    )
    assert tuple(spec.id for spec in schema.outputs) == ("IMAGE", "MASK")
    fields = {spec.id: spec for spec in schema.inputs}
    mask = fields["mask"]
    assert not mask.required and mask.default is None and mask.socketless
    assert mask.type == TypeExpr.concrete(ASSET_TYPE)
    assert isinstance(mask.widget, AssetWidget) and mask.widget.allow_upload
    assert mask.widget.kind == "media/image"
    assert mask.source_filename == SourceFilenameSpec("media/image", "input")
    assert not fields["image"].required
    for name in ("width", "height"):
        spec = fields[name]
        assert spec.default == 512 and spec.socketless and spec.hidden
        assert spec.widget == NumberWidget(min=64, max=4096, step=64)
    restored = schema_from_wire(schema_to_wire(schema))
    assert restored.input("mask") == mask
    assert restored.emits_previews


def test_animated_savers_preserve_source_controls_and_append_mount() -> None:
    for node, label, controls in (
        (SaveAnimatedPNG, "Save Animated PNG", ("compress_level",)),
        (SaveAnimatedWEBP, "Save Animated WEBP", ("lossless", "quality", "method")),
    ):
        schema = node.schema()
        assert schema.display_name == label
        assert tuple(item.id for item in schema.inputs) == (
            "images",
            "filename_prefix",
            "fps",
            *controls,
            "mount",
        )
        fields = {item.id: item for item in schema.inputs}
        assert fields["filename_prefix"].default == "ComfyUI"
        assert fields["filename_prefix"].widget == StringWidget(multiline=False)
        assert fields["fps"].default == 6.0
        assert fields["fps"].widget == NumberWidget(min=0.01, max=1000.0, step=0.01)
        assert fields["mount"].widget == MountWidget()
        assert fields["mount"].adjacent_to == "filename_prefix"
        assert not fields["mount"].required and fields["mount"].default is None
        assert schema.output_node and not schema.idempotent
        assert tuple(item.id for item in schema.outputs) == ("images",)
        if node is SaveAnimatedPNG:
            assert fields["compress_level"].default == 4 and fields["compress_level"].advanced
            assert fields["compress_level"].widget == NumberWidget(min=0, max=9)
        else:
            assert fields["lossless"].default is True
            assert fields["quality"].default == 80
            assert fields["quality"].widget == NumberWidget(min=0, max=100)
            assert fields["method"].widget == ComboWidget(options=("default", "fastest", "slowest"))


def test_advanced_image_save_retains_format_branches_and_adjacent_mount() -> None:
    schema = SaveImageAdvanced.schema()
    assert schema.input_order == ("images", "filename_prefix", "format", "mount")
    assert schema.inputs[1].widget == StringWidget(multiline=False)
    assert schema.inputs[-1].widget == MountWidget()
    assert schema.inputs[-1].adjacent_to == "filename_prefix"
    assert not schema.inputs[-1].required and schema.inputs[-1].default is None
    assert schema.output_node and not schema.idempotent
    assert tuple(item.id for item in schema.outputs) == ("images",)
    formats = schema.combos[0]
    assert formats.id == "format"
    assert tuple(option.key for option in formats.options) == ("png", "exr", "avif")
    assert schema_to_wire(schema)["nodeType"] == "dinkster.image.save_image_advanced"
