"""ComfyUI-compatible image file nodes backed by mounted assets."""

from __future__ import annotations

import importlib
import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from io import BytesIO
from typing import Any, Literal

from dinkster_api.v1 import (
    ASSET_TYPE,
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    ENCODED_MEDIA_LIMIT_BYTES,
    IMAGE_TYPE,
    IMAGE_WORKING_SET_LIMIT_BYTES,
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
    current_export_snapshot,
    initialize_inference_process,
    materialize_image,
    open_image_components,
    render_image_png,
)

from .advanced_image_encoding import inject_png_metadata

_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_ASSET = TypeExpr.concrete(ASSET_TYPE)
_MASK = TypeExpr.concrete(MASK_TYPE)
_COMBO = TypeExpr.concrete(CORE_COMBO)
_BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_INT = TypeExpr.concrete(CORE_INT)
_STRING = TypeExpr.concrete(CORE_STRING)
_IMAGE_MEDIA_TYPES = ("image/png", "image/jpeg", "image/webp", "image/gif", "image/tiff")
_MAX_IMAGE_DIMENSION = 16_384
_MAX_IMAGE_FRAMES = 4096


def image_writer() -> AssetWriter:
    snapshot = os.environ.get("DINKSTER_MOUNTS_SNAPSHOT", "")
    if not snapshot:
        raise AssetError("saving requires DINKSTER_MOUNTS_SNAPSHOT")
    return AssetWriter(MountSnapshotWriter(snapshot))


def _check_dimensions(width: int, height: int) -> None:
    if min(width, height) < 1 or max(width, height) > _MAX_IMAGE_DIMENSION:
        raise ValueError(f"image dimensions must be between 1 and {_MAX_IMAGE_DIMENSION}")
    # Float RGB/mask lists and stack outputs coexist with Pillow and conversion scratch.
    if width * height * 96 > IMAGE_WORKING_SET_LIMIT_BYTES:
        raise ValueError("decoded image exceeds the image working-set limit")


def image_batch(images: Any) -> Any:
    batch = materialize_image(images)
    if batch.ndim != 4 or batch.shape[0] < 1 or batch.shape[3] not in (1, 3, 4):
        raise ValueError("images must be a nonempty [B,H,W,1|3|4] batch")
    if batch.shape[0] > _MAX_IMAGE_FRAMES:
        raise ValueError("image batch exceeds the frame limit")
    _check_dimensions(int(batch.shape[2]), int(batch.shape[1]))
    # Source, clamp/quantization tensors, byte frames and codec scratch can coexist.
    peak_bytes = batch.numel() * (4 * max(4, batch.element_size()) + 8)
    if peak_bytes > IMAGE_WORKING_SET_LIMIT_BYTES:
        raise ValueError("image batch exceeds the image working-set limit")
    return batch


def _load_image(asset: AssetRef) -> tuple[Any, Any]:
    if asset.size > IMAGE_WORKING_SET_LIMIT_BYTES:
        raise ValueError("image asset exceeds the input byte limit")
    with asset.open() as handle:
        if os.fstat(handle.fileno()).st_size > IMAGE_WORKING_SET_LIMIT_BYTES:
            raise ValueError("image asset exceeds the input byte limit")
        header = handle.read(12)
        animated_webp = False
        if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
            image_module = importlib.import_module("PIL.Image")
            handle.seek(0)
            with image_module.open(handle) as source:
                _check_dimensions(*source.size)
                animated_webp = getattr(source, "n_frames", 1) > 1
    # FFmpeg exposes animated WebP as a zero-dimension stream. Select the
    # fixed-source fallback before decoder admission, without catching refusals.
    if animated_webp:
        return _load_webp(asset)
    torch = importlib.import_module("torch")
    images: list[Any] = []
    masks: list[Any] = []
    decoded_bytes = 0
    # Retained outputs use at most half the budget, leaving the other half
    # for the next shared decoder frame and its conversion scratch.
    with open_image_components(
        asset,
        max_dimension=_MAX_IMAGE_DIMENSION,
        max_source_bytes=IMAGE_WORKING_SET_LIMIT_BYTES,
        max_working_set_bytes=IMAGE_WORKING_SET_LIMIT_BYTES // 2,
    ) as frames:
        for components in frames:
            if len(images) >= _MAX_IMAGE_FRAMES:
                raise ValueError("image asset exceeds the frame limit")
            height, width = components.pixels.shape[:2]
            frame_pixels = width * height
            # RGB may retain its RGBA backing array; include that alpha storage
            # as well as the separately inverted output mask without copying RGB.
            frame_bytes = components.pixels.nbytes + (
                components.alpha.nbytes * 2 if components.alpha is not None else 64 * 64 * 4
            )
            peak_bytes = 2 * (decoded_bytes + frame_bytes) + frame_pixels * 64 + 32_768
            if peak_bytes > IMAGE_WORKING_SET_LIMIT_BYTES:
                raise ValueError("decoded image batch exceeds the image working-set limit")
            decoded_bytes += frame_bytes
            images.append(torch.from_numpy(components.pixels))
            masks.append(
                1.0 - torch.from_numpy(components.alpha[..., 0])
                if components.alpha is not None
                else torch.zeros((64, 64), dtype=torch.float32)
            )
    return torch.stack(images), torch.stack(masks)


def _load_webp(asset: AssetRef) -> tuple[Any, Any]:
    image_module = importlib.import_module("PIL.Image")
    image_ops = importlib.import_module("PIL.ImageOps")
    image_sequence = importlib.import_module("PIL.ImageSequence")
    numpy = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    images: list[Any] = []
    masks: list[Any] = []
    decoded_bytes = 0
    with asset.open() as handle:
        if os.fstat(handle.fileno()).st_size > IMAGE_WORKING_SET_LIMIT_BYTES:
            raise ValueError("image asset exceeds the input byte limit")
        with image_module.open(handle) as source:
            _check_dimensions(*source.size)
            if getattr(source, "n_frames", 1) > _MAX_IMAGE_FRAMES:
                raise ValueError("image asset exceeds the frame limit")
            width = height = None
            for frame in image_sequence.Iterator(source):
                _check_dimensions(*frame.size)
                has_alpha = "A" in frame.getbands()
                frame_pixels = frame.width * frame.height
                frame_bytes = frame_pixels * 12 + (frame_pixels if has_alpha else 64 * 64) * 4
                peak_bytes = 2 * (decoded_bytes + frame_bytes) + frame_pixels * 64
                if peak_bytes > IMAGE_WORKING_SET_LIMIT_BYTES:
                    raise ValueError("decoded image batch exceeds the image working-set limit")
                frame = image_ops.exif_transpose(frame)
                if width is None:
                    width, height = frame.size
                if frame.size != (width, height):
                    continue
                decoded_bytes += frame_bytes
                rgb = frame.convert("RGB")
                images.append(torch.from_numpy(numpy.asarray(rgb, dtype=numpy.float32) / 255.0))
                if has_alpha:
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
    batch = image_batch(images)
    pixels = torch.floor(torch.clamp(batch, 0, 1) * 255).to(torch.uint8).cpu().numpy()
    frames = [image_module.fromarray(frame) for frame in pixels]
    initialize_inference_process()
    submitted = (
        None
        if importlib.import_module("dinkster_inference.cli_args").args.disable_metadata
        else current_export_snapshot()
    )
    if format_name == "WEBP":
        exif = frames[0].getexif()
        if submitted is not None:
            if submitted.prompt is not None:
                exif[0x0110] = f"prompt:{json.dumps(dict(submitted.prompt))}"
            for index, (key, value) in enumerate((submitted.extra_pnginfo or {}).items()):
                exif[0x010F - index] = f"{key}:{json.dumps(value)}"
        options = {**options, "exif": exif}
    elif submitted is not None:
        metadata = importlib.import_module("PIL.PngImagePlugin").PngInfo()
        if submitted.prompt:
            metadata.add(
                b"comf",
                b"prompt\0" + json.dumps(dict(submitted.prompt)).encode("latin-1", "strict"),
                after_idat=True,
            )
        for key, value in (submitted.extra_pnginfo or {}).items():
            metadata.add(
                b"comf",
                key.encode("latin-1", "strict")
                + b"\0"
                + json.dumps(value).encode("latin-1", "strict"),
                after_idat=True,
            )
        options = {**options, "pnginfo": metadata}
    writer = image_writer()
    suffix, media_type = (".webp", "image/webp") if format_name == "WEBP" else (".png", "image/png")
    with tempfile.TemporaryFile() as buffer:
        frames[0].save(
            buffer, format=format_name, save_all=True, append_images=frames[1:], **options
        )
        writer.save_stream(
            writer.output_target(
                filename_prefix,
                mount=mount,
                image_width=int(batch.shape[2]),
                image_height=int(batch.shape[1]),
            ),
            buffer,
            suffix=suffix,
            media_type=media_type,
            limit=ENCODED_MEDIA_LIMIT_BYTES,
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
        batch = image_batch(images)
        initialize_inference_process()
        submitted = (
            None
            if importlib.import_module("dinkster_inference.cli_args").args.disable_metadata
            else current_export_snapshot()
        )
        writer = image_writer()
        torch = importlib.import_module("torch")
        target = writer.output_target(
            filename_prefix,
            mount=mount,
            image_width=int(batch.shape[2]),
            image_height=int(batch.shape[1]),
        )

        def sources() -> Iterator[BytesIO]:
            for index in range(batch.shape[0]):
                image = torch.floor(torch.clamp(batch[index : index + 1], 0, 1) * 255.0) / 255.0
                encoded = render_image_png(image)
                if submitted is not None:
                    encoded = inject_png_metadata(
                        encoded,
                        dict(submitted.prompt) if submitted.prompt is not None else None,
                        dict(submitted.extra_pnginfo)
                        if submitted.extra_pnginfo is not None
                        else None,
                    )
                if len(encoded) > ENCODED_MEDIA_LIMIT_BYTES:
                    raise ValueError("encoded image exceeds the output byte limit")
                with BytesIO(encoded) as stream:
                    yield stream

        writer.save_stream_batch(
            target,
            sources(),
            suffix="_.png",
            media_type="image/png",
            limit=ENCODED_MEDIA_LIMIT_BYTES,
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
