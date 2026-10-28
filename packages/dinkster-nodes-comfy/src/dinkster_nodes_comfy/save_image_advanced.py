"""High-depth ComfyUI image exports through mounted output targets."""

from __future__ import annotations

import importlib
import tempfile
from collections.abc import Generator, Mapping
from contextlib import closing
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO

from dinkster_api.v1 import (
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    ENCODED_MEDIA_LIMIT_BYTES,
    IMAGE_TYPE,
    ComboWidget,
    DynamicComboOption,
    DynamicComboSpec,
    InputSpec,
    MountWidget,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
)

from . import advanced_image_encoding as encoding
from .image_file_tools import image_batch, image_writer

_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_STRING = TypeExpr.concrete(CORE_STRING)
_COMBO = TypeExpr.concrete(CORE_COMBO)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_INT = TypeExpr.concrete(CORE_INT)


class SaveImageAdvanced(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "SaveImageAdvanced",
            display_name="Save Image (Advanced)",
            description="Saves the input images to your ComfyUI output directory.",
            category="image",
            search_terms=("save", "save image", "export image", "output image", "write image"),
            inputs=(
                InputSpec("images", _IMAGE, doc="The images to save."),
                InputSpec(
                    "filename_prefix",
                    _STRING,
                    default="ComfyUI",
                    widget=StringWidget(multiline=False),
                    doc=(
                        "The prefix for the file to save. May include formatting tokens such as "
                        "%date:yyyy-MM-dd% or %Empty Latent Image.width%."
                    ),
                ),
                InputSpec(
                    "mount",
                    _STRING,
                    required=False,
                    default=None,
                    widget=MountWidget(),
                    adjacent_to="filename_prefix",
                ),
            ),
            combos=(
                DynamicComboSpec(
                    "format",
                    (
                        DynamicComboOption(
                            "png",
                            (
                                InputSpec(
                                    "bit_depth",
                                    _COMBO,
                                    default="8-bit",
                                    advanced=True,
                                    widget=ComboWidget(options=("8-bit", "16-bit")),
                                ),
                                InputSpec(
                                    "input_color_space",
                                    _COMBO,
                                    default="sRGB",
                                    advanced=True,
                                    widget=ComboWidget(options=("sRGB",)),
                                ),
                            ),
                        ),
                        DynamicComboOption(
                            "exr",
                            (
                                InputSpec(
                                    "bit_depth",
                                    _COMBO,
                                    default="16-bit float",
                                    advanced=True,
                                    widget=ComboWidget(options=("32-bit float", "16-bit float")),
                                ),
                                InputSpec(
                                    "input_color_space",
                                    _COMBO,
                                    default="sRGB",
                                    advanced=True,
                                    widget=ComboWidget(
                                        options=("sRGB", "HDR", "linear", "linear ACEScg")
                                    ),
                                    doc=(
                                        "Colorspace of the input tensor. The EXR is always written "
                                        "as scene-linear in the matching gamut.\n"
                                        "sRGB - input is sRGB-encoded Rec.709; the inverse sRGB "
                                        "EOTF is applied.\n"
                                        "HDR - input is HLG-encoded Rec.2020 (BT.2100); the "
                                        "inverse HLG OETF is applied to get scene-linear light.\n"
                                        "linear - input is already scene-linear (Rec.709 "
                                        "primaries); written through unchanged. Use this for "
                                        "renderer/compositor output.\n"
                                        "linear ACEScg - input is already scene-linear ACEScg "
                                        "(AP1 / D60); written through unchanged with ACEScg "
                                        "chromaticities."
                                    ),
                                ),
                            ),
                        ),
                        DynamicComboOption(
                            "avif",
                            (
                                InputSpec(
                                    "bit_depth",
                                    _COMBO,
                                    default="auto",
                                    advanced=True,
                                    widget=ComboWidget(
                                        options=("auto", "8-bit YUV420", "10-bit YUV420")
                                    ),
                                    doc=(
                                        "Auto uses 8-bit YUV420 for sRGB and 10-bit YUV420 for HDR."
                                    ),
                                ),
                                InputSpec(
                                    "input_color_space",
                                    _COMBO,
                                    default="sRGB",
                                    advanced=True,
                                    widget=ComboWidget(options=("sRGB", "HDR", "HDR PQ")),
                                    doc=(
                                        "Colorspace of the input images. HDR selects BT.2020/HLG "
                                        "and HDR PQ selects BT.2020/PQ."
                                    ),
                                ),
                                InputSpec(
                                    "crf",
                                    _INT,
                                    default=18,
                                    advanced=True,
                                    widget=NumberWidget(min=1, max=63),
                                    doc="Lower values produce higher quality and larger files.",
                                ),
                                DynamicComboSpec(
                                    "save_mode",
                                    (
                                        DynamicComboOption("still images"),
                                        DynamicComboOption(
                                            "animated",
                                            (
                                                InputSpec(
                                                    "fps",
                                                    _FLOAT,
                                                    default=6.0,
                                                    widget=NumberWidget(
                                                        min=0.01, max=1000.0, step=0.01
                                                    ),
                                                ),
                                                InputSpec(
                                                    "loop_count",
                                                    _INT,
                                                    default=0,
                                                    advanced=True,
                                                    widget=NumberWidget(min=0, max=1000),
                                                    doc=(
                                                        "Number of times to loop the animation. "
                                                        "0 loops forever."
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                    display_name="save mode",
                                ),
                            ),
                        ),
                    ),
                    doc="The file format in which to save the image.",
                ),
            ),
            input_order=("images", "filename_prefix", "format", "mount"),
            outputs=(OutputSpec("images", _IMAGE),),
            idempotent=False,
            output_node=True,
        )

    @classmethod
    def execute(
        cls,
        *,
        images: Any,
        filename_prefix: str = "ComfyUI",
        format: str | Mapping[str, Any] = "png",
        mount: str | None = None,
        **values: Any,
    ) -> Mapping[str, object]:
        if isinstance(format, str):
            format = {
                "format": format,
                **{key.removeprefix("format."): value for key, value in values.items()},
            }
        file_format = format["format"]
        defaults = {"png": "8-bit", "exr": "16-bit float", "avif": "auto"}
        if file_format not in defaults:
            raise ValueError(f"Unsupported image format: {file_format!r}")
        bit_depth = format.get("bit_depth", defaults[file_format])
        colorspace = format.get("input_color_space", "sRGB")
        initialize_inference_process()
        write_metadata = not importlib.import_module(
            "dinkster_inference.cli_args"
        ).args.disable_metadata
        batch = materialize_image(images)
        if batch.ndim == 3:
            # Source grayscale batches omit channels; width four is not RGBA.
            batch = batch.unsqueeze(-1)
        batch = image_batch(batch)
        writer = image_writer()
        target = writer.output_target(
            filename_prefix,
            mount=mount,
            image_width=int(batch.shape[2]),
            image_height=int(batch.shape[1]),
        )

        def sources() -> Generator[BinaryIO]:
            if file_format == "avif":
                selected_mode = format.get("save_mode", {"save_mode": "still images"})
                if isinstance(selected_mode, str):
                    save_mode: Mapping[str, Any] = {
                        "save_mode": selected_mode,
                        **{
                            key.removeprefix("save_mode."): value
                            for key, value in format.items()
                            if key.startswith("save_mode.")
                        },
                    }
                else:
                    save_mode = selected_mode
                animated = save_mode["save_mode"] == "animated"
                batches = (batch,) if animated else batch.unsqueeze(1)
                for frames in batches:
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "image.avif"
                        encoding.save_avif(
                            frames,
                            str(path),
                            bit_depth,
                            colorspace,
                            format.get("crf", 18),
                            fps=save_mode.get("fps", 1.0),
                            loop_count=save_mode.get("loop_count") if animated else None,
                        )
                        with path.open("rb") as stream:
                            yield stream
            else:
                for image in batch:
                    if file_format == "exr":
                        image = encoding.to_scene_linear(image, colorspace)
                    encoded = encoding.encode_image(
                        image,
                        file_format,
                        bit_depth,
                        "linear" if file_format == "exr" else colorspace,
                    )
                    if file_format == "exr" and write_metadata:
                        encoded = encoding.inject_exr_metadata(encoded, None, None, colorspace)
                    if len(encoded) > ENCODED_MEDIA_LIMIT_BYTES:
                        raise ValueError("encoded image exceeds the output byte limit")
                    with BytesIO(encoded) as stream:
                        yield stream

        with closing(sources()) as streams:
            writer.save_stream_batch(
                target,
                streams,
                suffix=f".{file_format}",
                media_type={"png": "image/png", "exr": "image/x-exr", "avif": "image/avif"}[
                    file_format
                ],
                limit=ENCODED_MEDIA_LIMIT_BYTES,
            )
        return cls.outputs(images=images)


__all__ = ["SaveImageAdvanced"]
