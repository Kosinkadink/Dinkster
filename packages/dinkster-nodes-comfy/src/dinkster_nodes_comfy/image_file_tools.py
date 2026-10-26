"""ComfyUI-compatible image file nodes backed by mounted assets."""

from __future__ import annotations

import importlib
import io
import os
from collections.abc import Mapping
from typing import Any, Literal

from dinkster_api.v1 import (
    ASSET_TYPE,
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    IMAGE_TYPE,
    MASK_TYPE,
    AssetError,
    AssetRef,
    AssetWidget,
    AssetWriter,
    ComboWidget,
    InputSpec,
    MountSnapshotWriter,
    MountWidget,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    SourceFilenameSpec,
    StringWidget,
    TypeExpr,
    materialize_image,
    render_image_png,
)

_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_ASSET = TypeExpr.concrete(ASSET_TYPE)
_MASK = TypeExpr.concrete(MASK_TYPE)
_COMBO = TypeExpr.concrete(CORE_COMBO)
_BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_INT = TypeExpr.concrete(CORE_INT)
_STRING = TypeExpr.concrete(CORE_STRING)
_IMAGE_MEDIA_TYPES = ("image/png", "image/jpeg", "image/webp", "image/gif", "image/tiff")


def _writer() -> AssetWriter:
    snapshot = os.environ.get("DINKSTER_MOUNTS_SNAPSHOT", "")
    if not snapshot:
        raise AssetError("saving requires DINKSTER_MOUNTS_SNAPSHOT")
    return AssetWriter(MountSnapshotWriter(snapshot))


def _load_image(asset: AssetRef) -> tuple[Any, Any]:
    image_module = importlib.import_module("PIL.Image")
    image_ops = importlib.import_module("PIL.ImageOps")
    image_sequence = importlib.import_module("PIL.ImageSequence")
    numpy = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    images: list[Any] = []
    masks: list[Any] = []
    with asset.open() as handle:
        source = image_module.open(handle)
        width = height = None
        for frame in image_sequence.Iterator(source):
            frame = image_ops.exif_transpose(frame)
            rgb = frame.convert("RGB")
            if width is None:
                width, height = rgb.size
            if rgb.size != (width, height):
                continue
            images.append(torch.from_numpy(numpy.asarray(rgb, dtype=numpy.float32) / 255.0))
            if "A" in frame.getbands():
                alpha = numpy.asarray(frame.getchannel("A"), dtype=numpy.float32) / 255.0
                masks.append(1.0 - torch.from_numpy(alpha))
            else:
                masks.append(torch.zeros((64, 64), dtype=torch.float32))
    if not images:
        raise ValueError("image contains no readable frames")
    return torch.stack(images), torch.stack(masks)


def _asset_input(category: Literal["input", "output"]) -> InputSpec:
    return InputSpec(
        "image",
        _ASSET,
        widget=AssetWidget(_IMAGE_MEDIA_TYPES, kind="media/image", allow_upload=True),
        source_filename=SourceFilenameSpec("media/image", category),
    )


def _save_animation(
    images: Any,
    filename_prefix: str,
    mount: str | None,
    format_name: str,
    options: dict[str, object],
) -> None:
    torch = importlib.import_module("torch")
    image_module = importlib.import_module("PIL.Image")
    batch = materialize_image(images)
    if batch.ndim != 4 or batch.shape[0] < 1:
        raise ValueError("images must be a nonempty [B,H,W,C] batch")
    pixels = torch.floor(torch.clamp(batch, 0, 1) * 255).to(torch.uint8).cpu().numpy()
    frames = [image_module.fromarray(frame) for frame in pixels]
    buffer = io.BytesIO()
    if format_name == "WEBP":
        options = {**options, "exif": frames[0].getexif()}
    frames[0].save(buffer, format=format_name, save_all=True, append_images=frames[1:], **options)
    writer = _writer()
    suffix, media_type = (".webp", "image/webp") if format_name == "WEBP" else (".png", "image/png")
    writer.save_bytes(
        writer.output_target(
            filename_prefix,
            mount=mount,
            image_width=int(batch.shape[2]),
            image_height=int(batch.shape[1]),
        ),
        buffer.getvalue(),
        suffix=suffix,
        media_type=media_type,
    )


class SaveImage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SaveImage",
            display_name="Save Image",
            category="image",
            description="Saves the input images to your ComfyUI output directory.",
            inputs=(
                InputSpec("images", _IMAGE, doc="The images to save."),
                InputSpec(
                    "filename_prefix",
                    _STRING,
                    default="ComfyUI",
                    doc=(
                        "The prefix for the file to save. This may include formatting information "
                        "such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values "
                        "from nodes."
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
            outputs=(OutputSpec("images", _IMAGE),),
            idempotent=False,
            output_node=True,
            search_terms=(
                "save",
                "save image",
                "export image",
                "output image",
                "write image",
                "download",
            ),
        )

    @classmethod
    def execute(
        cls, *, images: Any, filename_prefix: str = "ComfyUI", mount: str | None = None
    ) -> Mapping[str, object]:
        batch = materialize_image(images)
        if batch.ndim != 4 or batch.shape[0] < 1:
            raise ValueError("images must be a nonempty [B,H,W,C] batch")
        writer = _writer()
        torch = importlib.import_module("torch")
        for index in range(batch.shape[0]):
            target = writer.output_target(
                filename_prefix.replace("%batch_num%", str(index)),
                mount=mount,
                image_width=int(batch.shape[2]),
                image_height=int(batch.shape[1]),
            )
            image = torch.floor(torch.clamp(batch[index : index + 1], 0, 1) * 255.0) / 255.0
            writer.save_bytes(
                target,
                render_image_png(image),
                suffix=".png",
                media_type="image/png",
            )
        return cls.outputs(images=images)


class PreviewImage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="PreviewImage",
            display_name="Preview Image",
            category="image",
            description="Preview the images without saving them to the ComfyUI output directory.",
            inputs=(InputSpec("images", _IMAGE),),
            outputs=(OutputSpec("images", _IMAGE, preview=True),),
            output_node=True,
            search_terms=(
                "preview",
                "preview image",
                "show image",
                "view image",
                "display image",
                "image viewer",
            ),
        )

    @classmethod
    def execute(cls, *, images: Any) -> Mapping[str, object]:
        materialize_image(images)
        return cls.outputs(images=images)


class SaveAnimatedWEBP(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "SaveAnimatedWEBP",
            display_name="Save Animated WEBP",
            category="image",
            inputs=(
                InputSpec("images", _IMAGE),
                InputSpec(
                    "filename_prefix",
                    _STRING,
                    default="ComfyUI",
                    widget=StringWidget(multiline=False),
                ),
                InputSpec(
                    "fps", _FLOAT, default=6.0, widget=NumberWidget(min=0.01, max=1000.0, step=0.01)
                ),
                InputSpec("lossless", _BOOLEAN, default=True),
                InputSpec("quality", _INT, default=80, widget=NumberWidget(min=0, max=100)),
                InputSpec(
                    "method", _COMBO, widget=ComboWidget(options=("default", "fastest", "slowest"))
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
            outputs=(OutputSpec("images", _IMAGE),),
            idempotent=False,
            output_node=True,
        )

    @classmethod
    def execute(
        cls,
        *,
        images: Any,
        filename_prefix: str,
        fps: float,
        lossless: bool,
        quality: int,
        method: str,
        mount: str | None = None,
    ) -> Mapping[str, object]:
        methods = {"default": 4, "fastest": 0, "slowest": 6}
        if method not in methods:
            raise ValueError("unknown WebP compression method")
        _save_animation(
            images,
            filename_prefix,
            mount,
            "WEBP",
            {
                "duration": int(1000 / fps),
                "lossless": lossless,
                "quality": quality,
                "method": methods[method],
            },
        )
        return cls.outputs(images=images)


class SaveAnimatedPNG(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "SaveAnimatedPNG",
            display_name="Save Animated PNG",
            category="image",
            inputs=(
                InputSpec("images", _IMAGE),
                InputSpec(
                    "filename_prefix",
                    _STRING,
                    default="ComfyUI",
                    widget=StringWidget(multiline=False),
                ),
                InputSpec(
                    "fps", _FLOAT, default=6.0, widget=NumberWidget(min=0.01, max=1000.0, step=0.01)
                ),
                InputSpec(
                    "compress_level",
                    _INT,
                    default=4,
                    advanced=True,
                    widget=NumberWidget(min=0, max=9),
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
        fps: float = 6.0,
        compress_level: int = 4,
        mount: str | None = None,
    ) -> Mapping[str, object]:
        _save_animation(
            images,
            filename_prefix,
            mount,
            "PNG",
            {"duration": int(1000 / fps), "compress_level": compress_level},
        )
        return cls.outputs(images=images)


class LoadImage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LoadImage",
            display_name="Load Image",
            category="image",
            inputs=(_asset_input("input"),),
            outputs=(OutputSpec("IMAGE", _IMAGE), OutputSpec("MASK", _MASK)),
            search_terms=(
                "load image",
                "open image",
                "import image",
                "image input",
                "upload image",
                "read image",
                "image loader",
            ),
        )

    @classmethod
    def execute(cls, *, image: AssetRef) -> Mapping[str, object]:
        images, masks = _load_image(image)
        return cls.outputs(IMAGE=images, MASK=masks)


class LoadImageMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LoadImageMask",
            display_name="Load Image (as Mask)",
            category="image",
            inputs=(
                _asset_input("input"),
                InputSpec(
                    "channel",
                    _COMBO,
                    widget=ComboWidget(options=("alpha", "red", "green", "blue")),
                ),
            ),
            outputs=(OutputSpec("MASK", _MASK),),
            search_terms=("import mask", "alpha mask", "channel mask"),
        )

    @classmethod
    def execute(cls, *, image: AssetRef, channel: str) -> Mapping[str, object]:
        images, alpha = _load_image(image)
        if channel == "alpha":
            return cls.outputs(MASK=alpha)
        channels = {"red": 0, "green": 1, "blue": 2}
        if channel not in channels:
            raise ValueError("channel must be alpha, red, green or blue")
        return cls.outputs(MASK=images[..., channels[channel]].clone())


class LoadImageOutput(LoadImage):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LoadImageOutput",
            display_name="Load Image (from Outputs)",
            category="image",
            description=(
                "Load an image from the output folder. When the refresh button is clicked, the "
                "node will update the image list and automatically select the first image, "
                "allowing for easy iteration."
            ),
            inputs=(_asset_input("output"),),
            outputs=(OutputSpec("IMAGE", _IMAGE), OutputSpec("MASK", _MASK)),
            search_terms=("output image", "previous generation"),
        )


__all__ = [
    "LoadImage",
    "LoadImageMask",
    "LoadImageOutput",
    "PreviewImage",
    "SaveAnimatedPNG",
    "SaveAnimatedWEBP",
    "SaveImage",
]
