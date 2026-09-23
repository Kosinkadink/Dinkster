"""Asset-backed still image, mask, preview, and animation nodes."""
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import io
import json
import math
import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, BinaryIO, cast

import numpy as np
from dinkster_api.v1 import (
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    SAVE_TARGET_TYPE,
    AssetError,
    AssetRef,
    AssetWidget,
    AssetWriter,
    ComboWidget,
    InputSpec,
    MappingSource,
    MountSnapshotWriter,
    Node,
    NodeSchema,
    NumberWidget,
    OutputRepresents,
    OutputSpec,
    ReplacementCase,
    ReplacementMigration,
    ReplacementPredicate,
    ReplacementRule,
    SaveTargetWidget,
    SourceFilenameSpec,
    StringWidget,
    TypeExpr,
    ValueTransform,
    annotate_mask,
    copy_media_semantics,
    is_digest,
    media_semantics,
)

from .image_metadata import metadata_document, parse_image_metadata_json, png_metadata

IMAGE_TYPE = "dinkster.image"
MASK_TYPE = "dinkster.mask"

IMAGE = TypeExpr.concrete(IMAGE_TYPE)
MASK = TypeExpr.concrete(MASK_TYPE)
IMAGE_ASSET = TypeExpr.asset_of(IMAGE)
MASK_ASSET = TypeExpr.asset_of(MASK)
IMAGE_ASSETS = TypeExpr.list_of(IMAGE_ASSET)
MASK_ASSETS = TypeExpr.list_of(MASK_ASSET)
SAVE_TARGET = TypeExpr.concrete(SAVE_TARGET_TYPE)
STRING = TypeExpr.concrete(CORE_STRING)
INT = TypeExpr.concrete(CORE_INT)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
COMBO = TypeExpr.concrete(CORE_COMBO)

IMAGE_ACCEPT = ("image/png", "image/jpeg", "image/webp", "image/gif", "image/tiff")
MAX_IMAGE_FILE_BYTES = 512 * 1024 * 1024
MAX_IMAGE_ARRAY_BYTES = 512 * 1024 * 1024
MAX_ENCODED_IMAGE_BYTES = 1024 * 1024 * 1024
MAX_ANIMATION_FRAMES = 4096
MAX_IMAGE_DIMENSION = 16384
IMAGE_FILE_DECODER_ID = "dinkster.media-image-file@2"
MASK_FILE_DECODER_ID = "dinkster.mask-file@2"

MAX_MASK_PAINT_JSON_BYTES = 4 * 1024 * 1024
MAX_MASK_PAINT_COMMANDS = 2048
MAX_MASK_PAINT_STROKE_POINTS = 8192
MAX_MASK_PAINT_TOTAL_POINTS = 32768

_FORMATS = {
    "png": ("PNG", ".png", "image/png"),
    "jpeg": ("JPEG", ".jpg", "image/jpeg"),
    "webp": ("WEBP", ".webp", "image/webp"),
}
_WEBP_METHODS = {"default": 4, "fastest": 0, "slowest": 6}
_MASK_POLARITIES = {
    "source": "coverage",
    "direct": "coverage",
    "inverted": "transparency",
    "opacity": "coverage",
    "coverage": "coverage",
    "transparency": "transparency",
    "mask_is_opacity": "coverage",
    "mask_is_transparency": "transparency",
}


@dataclass(frozen=True)
class _DecodedImage:
    image: np.ndarray
    alpha: np.ndarray | None
    metadata_json: str


def _mount_writer() -> AssetWriter:
    snapshot = os.environ.get("DINKSTER_MOUNTS_SNAPSHOT", "")
    if not snapshot:
        raise AssetError(
            "saving requires filesystem mounts, but this process has no "
            "DINKSTER_MOUNTS_SNAPSHOT configured"
        )
    return AssetWriter(MountSnapshotWriter(snapshot))


def _validate_asset_size(asset: AssetRef) -> None:
    if asset.size > MAX_IMAGE_FILE_BYTES:
        raise ValueError(f"image asset exceeds the {MAX_IMAGE_FILE_BYTES}-byte input limit")


def _validate_dimensions(
    width: int, height: int, channels: int = 4, bytes_per_channel: int = 4
) -> None:
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be nonzero")
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise ValueError(f"image dimensions exceed {MAX_IMAGE_DIMENSION} pixels")
    if width * height * channels * bytes_per_channel > MAX_IMAGE_ARRAY_BYTES:
        raise ValueError(f"decoded image exceeds the {MAX_IMAGE_ARRAY_BYTES}-byte array limit")


def _open_still(asset: AssetRef, *, allow_batch: bool = False) -> tuple[BinaryIO, Any, int]:
    from PIL import Image

    _validate_asset_size(asset)
    handle = asset.open()
    try:
        actual_size = os.fstat(handle.fileno()).st_size
        if actual_size > MAX_IMAGE_FILE_BYTES:
            raise ValueError(f"image asset exceeds the {MAX_IMAGE_FILE_BYTES}-byte input limit")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            source = Image.open(handle)
        if not allow_batch and int(getattr(source, "n_frames", 1)) != 1:
            source.close()
            raise ValueError("still image loaders reject animated and multipage assets")
        _validate_dimensions(
            *source.size, bytes_per_channel=2 if source.mode.startswith("I;16") else 1
        )
        return handle, source, actual_size
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        handle.close()
        raise ValueError(f"image exceeds the decompression safety limit: {exc}") from exc
    except BaseException:
        handle.close()
        raise


def _srgb(source: Any, icc_profile: object) -> Any:
    from PIL import ImageCms

    rgb = source.convert("RGB")
    if not isinstance(icc_profile, bytes):
        return rgb
    try:
        transformed = ImageCms.profileToProfile(
            source if source.mode in {"RGB", "RGBA", "RGBX", "CMYK", "LAB"} else rgb,
            ImageCms.ImageCmsProfile(io.BytesIO(icc_profile)),
            ImageCms.createProfile("sRGB"),
            renderingIntent=ImageCms.Intent.PERCEPTUAL,
            outputMode="RGB",
        )
    except (ImageCms.PyCMSError, OSError, TypeError, ValueError):
        return rgb
    return rgb if transformed is None else transformed


def _decode_still(asset: AssetRef, *, allow_batch: bool = False) -> _DecodedImage:
    from PIL import ImageOps, ImageSequence

    try:
        handle, source, actual_size = _open_still(asset, allow_batch=allow_batch)
        with handle, source:
            info = dict(source.info)
            images: list[np.ndarray] = []
            alphas: list[np.ndarray] = []
            has_alpha = source.format == "GIF"
            size = None
            for index, frame in enumerate(ImageSequence.Iterator(source)):
                if index >= MAX_ANIMATION_FRAMES:
                    raise ValueError(f"image batch exceeds {MAX_ANIMATION_FRAMES} frames")
                oriented = ImageOps.exif_transpose(frame)
                if size is None:
                    size = oriented.size
                if oriented.size != size:
                    continue
                sixteen_bit = oriented.mode.startswith("I;16")
                _validate_dimensions(*oriented.size, bytes_per_channel=2 if sixteen_bit else 1)
                if (len(images) + 1) * size[0] * size[1] * (
                    8 if sixteen_bit else 4
                ) > MAX_IMAGE_ARRAY_BYTES:
                    raise ValueError(
                        f"decoded image exceeds the {MAX_IMAGE_ARRAY_BYTES}-byte limit"
                    )
                has_alpha |= "A" in oriented.getbands() or "transparency" in oriented.info
                alpha_frame = np.asarray(oriented.convert("RGBA").getchannel("A"), dtype=np.uint8)
                alphas.append(alpha_frame)
                if sixteen_bit:
                    plane = np.asarray(oriented, dtype=np.uint16)
                    rgb = np.repeat(plane[..., None], 3, axis=-1)
                else:
                    rgb = np.asarray(_srgb(oriented, info.get("icc_profile")), dtype=np.uint8)
                images.append(rgb)
            image = np.stack(images)
            alpha = np.stack(alphas) if has_alpha else None
            document = metadata_document(
                info,
                name=asset.name,
                digest=asset.digest,
                media_type=asset.media_type,
                size=actual_size,
            )
            metadata_json = json.dumps(
                document, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            )
            return _DecodedImage(image=image, alpha=alpha, metadata_json=metadata_json)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot decode {asset.name!r} as a still image: {exc}") from exc


def decode_image_file(asset: object) -> object:
    if not isinstance(asset, AssetRef):
        raise TypeError(f"decode_image_file expects AssetRef, got {type(asset).__name__}")
    decoded = _decode_still(asset)
    if decoded.alpha is None:
        return decoded.image
    return np.concatenate((decoded.image, decoded.alpha[..., None]), axis=-1)


def _load_mask_values(asset: AssetRef, channel: str, mask_polarity: str) -> np.ndarray:
    from PIL import ImageOps

    if channel not in ("alpha", "luminance", "red", "green", "blue"):
        raise ValueError(f"unknown mask channel: {channel!r}")
    mask_polarity = _MASK_POLARITIES.get(mask_polarity, mask_polarity)
    if mask_polarity not in ("coverage", "transparency"):
        raise ValueError(f"unknown mask polarity: {mask_polarity!r}")
    try:
        handle, source, _ = _open_still(asset)
        with handle, source:
            info = dict(source.info)
            source.load()
            oriented = ImageOps.exif_transpose(source)
            if channel == "alpha":
                if "A" in oriented.getbands() or "transparency" in info:
                    mask = np.asarray(oriented.convert("RGBA").getchannel("A"), dtype=np.uint8)
                else:
                    mask = np.full((64, 64), 255, dtype=np.uint8)
            elif oriented.mode.startswith("I;16"):
                mask = np.asarray(oriented, dtype=np.uint16)
            elif channel == "luminance":
                mask = np.asarray(oriented.convert("L"), dtype=np.uint8)
            else:
                index = {"red": 0, "green": 1, "blue": 2}[channel]
                rgb = np.asarray(_srgb(oriented, info.get("icc_profile")), dtype=np.uint8)
                mask = rgb[..., index]
            if mask_polarity == "transparency":
                mask = np.iinfo(mask.dtype).max - mask
            return annotate_mask(
                np.ascontiguousarray(mask)[None, :, :],
                polarity=mask_polarity,
                semantic="alpha" if channel == "alpha" else "selection",
            )
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot decode {asset.name!r} as a mask: {exc}") from exc


def decode_mask_file(asset: object) -> object:
    if not isinstance(asset, AssetRef):
        raise TypeError(f"decode_mask_file expects AssetRef, got {type(asset).__name__}")
    return _load_mask_values(asset, "luminance", "coverage")


def _image_batch(images: object) -> np.ndarray:
    if not isinstance(images, np.ndarray):
        raise ValueError("images must be a numpy array")
    array = np.asarray(images, dtype=np.float32)
    if array.ndim != 4 or array.shape[0] < 1 or array.shape[3] not in (1, 3, 4):
        raise ValueError("images must be a nonempty float32 [B,H,W,1|3|4] batch")
    if array.shape[0] > MAX_ANIMATION_FRAMES:
        raise ValueError(f"image batch exceeds {MAX_ANIMATION_FRAMES} frames")
    _validate_dimensions(int(array.shape[2]), int(array.shape[1]), int(array.shape[3]))
    if array.nbytes > MAX_IMAGE_ARRAY_BYTES:
        raise ValueError(f"image batch exceeds the {MAX_IMAGE_ARRAY_BYTES}-byte array limit")
    if not np.isfinite(array).all():
        raise ValueError("images contain non-finite values")
    return copy_media_semantics(images, np.ascontiguousarray(array, dtype=np.float32))


def _mask_batch(masks: object) -> np.ndarray:
    if not isinstance(masks, np.ndarray):
        raise ValueError("masks must be a numpy array")
    array = np.asarray(masks, dtype=np.float32)
    if array.ndim == 2:
        array = array[None, :, :]
    if array.ndim != 3 or array.shape[0] < 1:
        raise ValueError("masks must be a nonempty float32 [B,H,W] batch")
    if array.shape[0] > MAX_ANIMATION_FRAMES:
        raise ValueError(f"mask batch exceeds {MAX_ANIMATION_FRAMES} elements")
    _validate_dimensions(int(array.shape[2]), int(array.shape[1]), 1)
    if array.nbytes > MAX_IMAGE_ARRAY_BYTES:
        raise ValueError(f"mask batch exceeds the {MAX_IMAGE_ARRAY_BYTES}-byte array limit")
    if not np.isfinite(array).all():
        raise ValueError("masks contain non-finite values")
    return copy_media_semantics(masks, np.ascontiguousarray(array, dtype=np.float32))


def _pillow_frame(frame: np.ndarray) -> Any:
    from PIL import Image

    if media_semantics(frame).get("alpha") == "premultiplied":
        frame = np.asarray(frame).copy()
        alpha = frame[..., -1:]
        frame[..., :-1] = np.divide(
            frame[..., :-1], alpha, out=np.zeros_like(frame[..., :-1]), where=alpha != 0
        )
    pixels = np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
    mode = {1: "L", 3: "RGB", 4: "RGBA"}[int(frame.shape[2])]
    if mode == "L":
        pixels = pixels[..., 0]
    return Image.fromarray(pixels, mode=mode)


def _encoded_still(
    frame: np.ndarray,
    *,
    format_name: str,
    quality: int,
    compression: int,
    lossless: bool,
    metadata_json: str,
) -> tuple[bytes, str, str]:
    if format_name not in _FORMATS:
        raise ValueError(f"format must be one of {tuple(_FORMATS)}, got {format_name!r}")
    if type(quality) is not int or not 0 <= quality <= 100:
        raise ValueError("quality must be an integer in 0..100")
    if type(compression) is not int or not 0 <= compression <= 9:
        raise ValueError("compression must be an integer in 0..9")
    if format_name == "jpeg" and int(frame.shape[2]) == 4:
        raise ValueError("JPEG does not support alpha; remove or composite the alpha channel")
    if metadata_json and format_name != "png":
        raise ValueError("explicit image metadata is supported only for PNG")
    pillow_format, suffix, media_type = _FORMATS[format_name]
    options: dict[str, object] = {}
    if format_name == "png":
        options.update(compress_level=compression, pnginfo=png_metadata(metadata_json))
    elif format_name == "jpeg":
        options.update(quality=max(1, quality), optimize=False)
    else:
        options.update(quality=quality, lossless=bool(lossless), method=4)
    buffer = io.BytesIO()
    _pillow_frame(frame).save(buffer, format=pillow_format, **options)
    data = buffer.getvalue()
    if len(data) > MAX_ENCODED_IMAGE_BYTES:
        raise ValueError(f"encoded image exceeds the {MAX_ENCODED_IMAGE_BYTES}-byte output limit")
    return data, suffix, media_type


def _mask_polarity_input(default: str) -> InputSpec:
    return InputSpec(
        "mask_polarity",
        COMBO,
        required=False,
        default=default,
        widget=ComboWidget(options=("coverage", "transparency")),
    )


def _mask_polarity_migration(
    node_type: str, inputs: tuple[str, ...], outputs: tuple[str, ...]
) -> ReplacementRule:
    cases: list[ReplacementCase] = []
    for source in ("mask_mode", "mask_polarity", "polarity"):
        for connected in (True, False):
            cases.append(
                ReplacementCase.build(
                    node_type,
                    when=(
                        ReplacementPredicate.input_connected(source)
                        if connected
                        else ReplacementPredicate.value_present(source)
                        if source != "polarity"
                        else None
                    ),
                    inputs={
                        **{name: MappingSource.copy(name) for name in inputs},
                        "mask_polarity": (
                            MappingSource.copy(source)
                            if connected
                            else MappingSource.from_value(
                                source, ValueTransform.enum_rename(_MASK_POLARITIES)
                            )
                        ),
                    },
                    outputs={name: name for name in outputs},
                )
            )
    return ReplacementRule(
        from_type=node_type,
        migration=ReplacementMigration(("polarity", "mask_mode", "mask_polarity")),
        cases=tuple(cases),
    )


class LoadImage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_image",
            editor_role="image-source",
            version=2,
            display_name="Load Image",
            category="image/io",
            inputs=(
                InputSpec(
                    "image",
                    IMAGE_ASSET,
                    widget=AssetWidget(IMAGE_ACCEPT, kind="media/image", allow_upload=True),
                    source_filename=SourceFilenameSpec("media/image", "input"),
                ),
                _mask_polarity_input("transparency"),
            ),
            outputs=(
                OutputSpec(
                    "image",
                    IMAGE,
                    preview=True,
                    alpha_policy="drop",
                    represents=OutputRepresents(input="image", rendition="decoded-image"),
                ),
                OutputSpec("mask", MASK, preview=True, mask_semantic="alpha"),
                OutputSpec("metadata", STRING),
            ),
            aliases=("LoadImage",),
            search_terms=("image loader", "upload image", "open image"),
        )

    @classmethod
    def execute(
        cls, *, image: AssetRef, mask_polarity: str = "transparency"
    ) -> Mapping[str, object]:
        if mask_polarity not in ("coverage", "transparency"):
            raise ValueError(f"unknown mask polarity: {mask_polarity}")
        decoded = _decode_still(image, allow_batch=True)
        mask = (
            np.zeros((decoded.image.shape[0], 64, 64), dtype=np.uint8)
            if decoded.alpha is None
            else np.ascontiguousarray(255 - decoded.alpha)
        )
        if mask_polarity == "coverage":
            mask = np.full_like(mask, 255) if decoded.alpha is None else decoded.alpha
        return cls.outputs(
            image=decoded.image,
            mask=annotate_mask(mask, polarity=mask_polarity, semantic="alpha"),
            metadata=decoded.metadata_json,
        )


def _paint_number(value: object, field: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"mask paint {field} must be a number")
    try:
        result = float(cast("int | float", value))
    except OverflowError as exc:
        raise ValueError(f"mask paint {field} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"mask paint {field} must be finite")
    return result


def _paint_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate field {key}")
        result[key] = value
    return result


def _reject_paint_json_constant(constant: str) -> object:
    raise ValueError(f"invalid numeric constant {constant}")


def _paint_operations(value: str) -> tuple[str, int, int, list[dict[str, object]]]:
    if type(value) is not str:
        raise TypeError("mask paint operations must be a string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("mask paint operations must be valid UTF-8") from exc
    if len(encoded) > MAX_MASK_PAINT_JSON_BYTES:
        raise ValueError(f"mask paint operations exceed {MAX_MASK_PAINT_JSON_BYTES} UTF-8 bytes")
    try:
        raw = json.loads(
            value,
            object_pairs_hook=_paint_json_object,
            parse_constant=_reject_paint_json_constant,
        )
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ValueError(f"invalid mask paint operations JSON: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != {
        "version",
        "sourceDigest",
        "width",
        "height",
        "commands",
    }:
        raise ValueError(
            "mask paint operations require version, sourceDigest, width, height and commands"
        )
    if type(raw["version"]) is not int or raw["version"] != 1:
        raise ValueError("mask paint operations version must be 1")
    source_digest = raw["sourceDigest"]
    if (
        not isinstance(source_digest, str)
        or not source_digest.startswith("blake3:")
        or not is_digest(source_digest)
    ):
        raise ValueError("mask paint sourceDigest must be a canonical BLAKE3 digest")
    width = raw["width"]
    height = raw["height"]
    if type(width) is not int or type(height) is not int or width < 1 or height < 1:
        raise ValueError("mask paint width and height must be positive integers")
    commands = raw["commands"]
    if not isinstance(commands, list):
        raise ValueError("mask paint commands must be an array")
    if len(commands) > MAX_MASK_PAINT_COMMANDS:
        raise ValueError(f"mask paint commands exceed {MAX_MASK_PAINT_COMMANDS}")

    result: list[dict[str, object]] = []
    total_points = 0
    for command_index, command in enumerate(commands):
        if not isinstance(command, dict) or "op" not in command:
            raise ValueError(f"mask paint command {command_index} must be an object with op")
        operation = command["op"]
        if operation in ("clear", "invert"):
            if set(command) != {"op"}:
                raise ValueError(f"mask paint {operation} command has unknown fields")
            result.append({"op": operation})
            continue
        if operation != "stroke" or set(command) != {
            "op",
            "mode",
            "size",
            "hardness",
            "points",
        }:
            raise ValueError(f"mask paint command {command_index} is unsupported or malformed")
        mode = command["mode"]
        if mode not in ("paint", "erase"):
            raise ValueError("mask paint stroke mode must be paint or erase")
        size = _paint_number(command["size"], "stroke size")
        hardness = _paint_number(command["hardness"], "stroke hardness")
        if not 1 <= size <= 256:
            raise ValueError("mask paint stroke size must be between 1 and 256")
        if not 0 <= hardness <= 1:
            raise ValueError("mask paint stroke hardness must be between 0 and 1")
        points = command["points"]
        if not isinstance(points, list) or not 1 <= len(points) <= MAX_MASK_PAINT_STROKE_POINTS:
            raise ValueError(
                f"mask paint stroke points must contain 1 to {MAX_MASK_PAINT_STROKE_POINTS} items"
            )
        total_points += len(points)
        if total_points > MAX_MASK_PAINT_TOTAL_POINTS:
            raise ValueError(f"mask paint operations exceed {MAX_MASK_PAINT_TOTAL_POINTS} points")
        checked_points: list[dict[str, float]] = []
        for point_index, point in enumerate(points):
            if not isinstance(point, dict) or set(point) != {"x", "y", "pressure"}:
                raise ValueError(
                    f"mask paint stroke point {point_index} must contain x, y and pressure"
                )
            x = _paint_number(point["x"], "point x")
            y = _paint_number(point["y"], "point y")
            pressure = _paint_number(point["pressure"], "point pressure")
            if not 0 <= pressure <= 1:
                raise ValueError("mask paint point pressure must be between 0 and 1")
            radius = max(0.5, size * (0.25 + pressure * 0.75) / 2)
            if not 0.5 - radius <= x <= width - 0.5 + radius:
                raise ValueError("mask paint point x cannot lie beyond the canvas brush reach")
            if not 0.5 - radius <= y <= height - 0.5 + radius:
                raise ValueError("mask paint point y cannot lie beyond the canvas brush reach")
            checked_points.append({"x": x, "y": y, "pressure": pressure})
        result.append(
            {
                "op": "stroke",
                "mode": mode,
                "size": size,
                "hardness": hardness,
                "points": checked_points,
            }
        )
    return source_digest, width, height, result


def _paint_dab(
    mask: np.ndarray,
    *,
    mode: str,
    size: float,
    hardness: float,
    point: Mapping[str, object],
) -> None:
    x = cast(float, point["x"])
    y = cast(float, point["y"])
    pressure = cast(float, point["pressure"])
    if pressure == 0:
        return
    radius = max(0.5, size * (0.25 + pressure * 0.75) / 2)
    inner = radius * hardness
    height, width = mask.shape
    x_start = max(0, math.floor(x - radius))
    x_stop = min(width - 1, math.ceil(x + radius))
    y_start = max(0, math.floor(y - radius))
    y_stop = min(height - 1, math.ceil(y + radius))
    target = 255 if mode == "paint" else 0
    columns = np.arange(x_start, x_stop + 1, dtype=np.float64) + 0.5
    rows = np.arange(y_start, y_stop + 1, dtype=np.float64) + 0.5
    distance = np.hypot(columns[None, :] - x, rows[:, None] - y)
    inside = distance <= radius
    if not np.any(inside):
        return
    if inner == radius:
        edge_strength = np.ones_like(distance)
    else:
        edge_strength = np.where(
            distance <= inner,
            1.0,
            1.0 - (distance - inner) / (radius - inner),
        )
    region = mask[y_start : y_stop + 1, x_start : x_stop + 1]
    current = region.astype(np.float64)
    value = current + (target - current) * edge_strength * pressure
    region[inside] = np.floor(value[inside] + 0.5).astype(np.uint8)


def _replay_mask_paint(base: np.ndarray, commands: list[dict[str, object]]) -> np.ndarray:
    mask = base.copy()
    for command in commands:
        operation = command["op"]
        if operation == "clear":
            mask.fill(0)
        elif operation == "invert":
            np.subtract(255, mask, out=mask)
        else:
            for point in cast("list[dict[str, float]]", command["points"]):
                _paint_dab(
                    mask,
                    mode=cast(str, command["mode"]),
                    size=cast(float, command["size"]),
                    hardness=cast(float, command["hardness"]),
                    point=point,
                )
    return mask


class PaintMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.mask.paint",
            editor_role="mask-paint",
            version=1,
            display_name="Paint Mask",
            category="mask/edit",
            inputs=(
                InputSpec(
                    "source",
                    IMAGE_ASSET,
                    widget=AssetWidget(IMAGE_ACCEPT, kind="media/image", allow_upload=False),
                ),
                InputSpec("operations", STRING, widget=StringWidget(multiline=True)),
            ),
            outputs=(
                OutputSpec(
                    "mask",
                    MASK,
                    preview=True,
                    mask_polarity="transparency",
                    mask_semantic="alpha",
                ),
            ),
            search_terms=("mask editor", "brush", "paint mask", "erase mask"),
        )

    @classmethod
    def execute(cls, *, source: AssetRef, operations: str) -> Mapping[str, object]:
        source_digest, width, height, commands = _paint_operations(operations)
        if source.digest != source_digest:
            raise ValueError("mask paint sourceDigest does not match the source asset")
        decoded = _decode_still(source)
        if decoded.image.shape != (1, height, width, 3):
            raise ValueError("mask paint dimensions do not match the decoded source")
        if decoded.alpha is None:
            base = np.zeros((height, width), dtype=np.uint8)
        else:
            base = 255 - decoded.alpha[0]
        painted = _replay_mask_paint(base, commands)
        return cls.outputs(
            mask=annotate_mask(
                painted.astype(np.float32)[None, :, :] / 255,
                polarity="transparency",
                semantic="alpha",
            )
        )


class LoadImageOutput(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_image_output",
            version=2,
            display_name="Load Output Image",
            category="image/io",
            inputs=(
                InputSpec(
                    "image",
                    IMAGE_ASSET,
                    widget=AssetWidget(IMAGE_ACCEPT, kind="media/image", allow_upload=True),
                    source_filename=SourceFilenameSpec("media/image", "output"),
                ),
                _mask_polarity_input("transparency"),
            ),
            outputs=(
                OutputSpec(
                    "image",
                    IMAGE,
                    preview=True,
                    alpha_policy="drop",
                    represents=OutputRepresents(input="image", rendition="decoded-image"),
                ),
                OutputSpec("mask", MASK, preview=True, mask_semantic="alpha"),
                OutputSpec("metadata", STRING),
            ),
            search_terms=("previous image", "output image"),
        )

    @classmethod
    def execute(
        cls, *, image: AssetRef, mask_polarity: str = "transparency"
    ) -> Mapping[str, object]:
        return LoadImage.execute(image=image, mask_polarity=mask_polarity)


class LoadMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_mask",
            version=2,
            display_name="Load Mask",
            category="mask/io",
            inputs=(
                InputSpec(
                    "mask",
                    MASK_ASSET,
                    widget=AssetWidget(IMAGE_ACCEPT, kind="media/image", allow_upload=True),
                    source_filename=SourceFilenameSpec("media/image", "input"),
                ),
                InputSpec(
                    "channel",
                    COMBO,
                    required=False,
                    default="alpha",
                    widget=ComboWidget(options=("alpha", "luminance", "red", "green", "blue")),
                ),
                _mask_polarity_input("transparency"),
            ),
            outputs=(OutputSpec("mask", MASK, preview=True),),
            search_terms=("mask loader", "alpha mask", "channel mask"),
            replacements=(
                _mask_polarity_migration("dinkster.load_mask", ("mask", "channel"), ("mask",)),
            ),
        )

    @classmethod
    def execute(
        cls, *, mask: AssetRef, channel: str = "alpha", mask_polarity: str = "transparency"
    ) -> Mapping[str, object]:
        return cls.outputs(mask=_load_mask_values(mask, channel, mask_polarity))


class ReadImageMetadata(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.read_image_metadata",
            display_name="Read Image Metadata",
            category="image/io",
            inputs=(
                InputSpec(
                    "image",
                    IMAGE_ASSET,
                    widget=AssetWidget(IMAGE_ACCEPT, kind="media/image", allow_upload=True),
                    source_filename=SourceFilenameSpec("media/image", "input"),
                ),
            ),
            outputs=(OutputSpec("metadata", STRING),),
            search_terms=("ComfyUI workflow metadata", "A1111 parameters", "PNG info"),
        )

    @classmethod
    def execute(cls, *, image: AssetRef) -> Mapping[str, object]:
        return cls.outputs(metadata=parse_image_metadata_json(image))


class SaveImage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.save_image",
            editor_role="image-save",
            display_name="Save Image",
            category="image/io",
            inputs=(
                InputSpec("images", IMAGE),
                InputSpec(
                    "target",
                    SAVE_TARGET,
                    required=False,
                    default=None,
                    widget=SaveTargetWidget(),
                ),
                InputSpec(
                    "format",
                    COMBO,
                    required=False,
                    default="png",
                    widget=ComboWidget(options=tuple(_FORMATS)),
                ),
                InputSpec(
                    "quality",
                    INT,
                    required=False,
                    default=90,
                    widget=NumberWidget(min=0, max=100, step=1),
                ),
                InputSpec(
                    "compression",
                    INT,
                    required=False,
                    default=4,
                    widget=NumberWidget(min=0, max=9, step=1),
                ),
                InputSpec(
                    "lossless", TypeExpr.concrete("core.boolean"), required=False, default=True
                ),
                InputSpec(
                    "metadata_json",
                    STRING,
                    required=False,
                    default="",
                    widget=StringWidget(multiline=True),
                    advanced=True,
                ),
            ),
            outputs=(OutputSpec("images", IMAGE), OutputSpec("assets", IMAGE_ASSETS)),
            idempotent=False,
            output_node=True,
            aliases=("SaveImage",),
            search_terms=("image saver", "PNG", "JPEG", "WebP"),
        )

    @classmethod
    def execute(
        cls,
        *,
        images: object,
        target: object = None,
        format: str = "png",
        quality: int = 90,
        compression: int = 4,
        lossless: bool = True,
        metadata_json: str = "",
    ) -> Mapping[str, object]:
        batch = _image_batch(images)
        encoded = [
            _encoded_still(
                copy_media_semantics(batch, frame),
                format_name=format,
                quality=quality,
                compression=compression,
                lossless=lossless,
                metadata_json=metadata_json,
            )
            for frame in batch
        ]
        writer = _mount_writer()
        destination = target if target is not None else writer.output_target("ComfyUI")
        assets = [
            writer.save_bytes(destination, data, suffix=suffix, media_type=media_type)
            for data, suffix, media_type in encoded
        ]
        return cls.outputs(images=batch, assets=assets)


class SaveMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.save_mask",
            version=2,
            display_name="Save Mask",
            category="mask/io",
            inputs=(
                InputSpec("masks", MASK),
                InputSpec(
                    "target",
                    SAVE_TARGET,
                    required=False,
                    default={"mount": "comfy-output", "prefix": "masks/ComfyUI"},
                    widget=SaveTargetWidget(".png"),
                ),
                InputSpec(
                    "bit_depth",
                    COMBO,
                    required=False,
                    default="8",
                    widget=ComboWidget(options=("8", "16")),
                ),
                _mask_polarity_input("coverage"),
                InputSpec(
                    "compression",
                    INT,
                    required=False,
                    default=4,
                    widget=NumberWidget(min=0, max=9, step=1),
                ),
            ),
            outputs=(OutputSpec("masks", MASK), OutputSpec("assets", MASK_ASSETS)),
            idempotent=False,
            output_node=True,
            search_terms=("mask saver", "16-bit mask", "PNG mask"),
            replacements=(
                _mask_polarity_migration(
                    "dinkster.save_mask",
                    ("masks", "target", "bit_depth", "compression"),
                    ("masks", "assets"),
                ),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        masks: object,
        target: object = None,
        bit_depth: str = "8",
        mask_polarity: str = "coverage",
        compression: int = 4,
    ) -> Mapping[str, object]:
        from PIL import Image

        batch = _mask_batch(masks)
        if bit_depth not in ("8", "16"):
            raise ValueError("bit_depth must be '8' or '16'")
        mask_polarity = _MASK_POLARITIES.get(mask_polarity, mask_polarity)
        if mask_polarity not in ("coverage", "transparency"):
            raise ValueError("mask_polarity must be 'coverage' or 'transparency'")
        if type(compression) is not int or not 0 <= compression <= 9:
            raise ValueError("compression must be an integer in 0..9")
        encoded: list[bytes] = []
        for mask in batch:
            values = 1.0 - mask if mask_polarity == "transparency" else mask
            if bit_depth == "16":
                pixels = np.rint(np.clip(values, 0.0, 1.0) * 65535.0).astype(np.uint16)
                image = Image.fromarray(pixels)
            else:
                pixels = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
                image = Image.fromarray(pixels, mode="L")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", compress_level=compression)
            data = buffer.getvalue()
            if len(data) > MAX_ENCODED_IMAGE_BYTES:
                raise ValueError(
                    f"encoded mask exceeds the {MAX_ENCODED_IMAGE_BYTES}-byte output limit"
                )
            encoded.append(data)
        destination = (
            target if target is not None else {"mount": "comfy-output", "prefix": "masks/ComfyUI"}
        )
        writer = _mount_writer()
        assets = [
            writer.save_bytes(destination, data, suffix=".png", media_type="image/png")
            for data in encoded
        ]
        return cls.outputs(masks=batch, assets=assets)


class PreviewImage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.preview_image",
            display_name="Preview Image",
            category="image/io",
            inputs=(InputSpec("images", IMAGE),),
            outputs=(OutputSpec("images", IMAGE, preview=True),),
            output_node=True,
            search_terms=("show image", "view image", "image viewer"),
        )

    @classmethod
    def execute(cls, *, images: object) -> Mapping[str, object]:
        return cls.outputs(images=_image_batch(images))


class SaveAnimatedImage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.save_animated_image",
            display_name="Save Animated Image",
            category="image/io",
            inputs=(
                InputSpec("images", IMAGE),
                InputSpec(
                    "target",
                    SAVE_TARGET,
                    required=False,
                    default={"mount": "comfy-output", "prefix": "animations/ComfyUI"},
                    widget=SaveTargetWidget(),
                ),
                InputSpec(
                    "format",
                    COMBO,
                    required=False,
                    default="webp",
                    widget=ComboWidget(options=("png", "webp")),
                ),
                InputSpec(
                    "fps",
                    FLOAT,
                    required=False,
                    default=6.0,
                    widget=NumberWidget(min=0.01, max=1000.0, step=0.01),
                ),
                InputSpec(
                    "loop",
                    INT,
                    required=False,
                    default=0,
                    widget=NumberWidget(min=0, max=65535, step=1),
                ),
                InputSpec(
                    "lossless", TypeExpr.concrete("core.boolean"), required=False, default=True
                ),
                InputSpec(
                    "quality",
                    INT,
                    required=False,
                    default=80,
                    widget=NumberWidget(min=0, max=100, step=1),
                ),
                InputSpec(
                    "method",
                    COMBO,
                    required=False,
                    default="default",
                    widget=ComboWidget(options=tuple(_WEBP_METHODS)),
                ),
                InputSpec(
                    "compression",
                    INT,
                    required=False,
                    default=4,
                    widget=NumberWidget(min=0, max=9, step=1),
                ),
                InputSpec(
                    "metadata_json",
                    STRING,
                    required=False,
                    default="",
                    widget=StringWidget(multiline=True),
                    advanced=True,
                ),
            ),
            outputs=(OutputSpec("images", IMAGE), OutputSpec("asset", IMAGE_ASSET, preview=True)),
            idempotent=False,
            output_node=True,
            search_terms=("animated PNG", "animated WebP", "animation saver"),
        )

    @classmethod
    def execute(
        cls,
        *,
        images: object,
        target: object = None,
        format: str = "webp",
        fps: float = 6.0,
        loop: int = 0,
        lossless: bool = True,
        quality: int = 80,
        method: str = "default",
        compression: int = 4,
        metadata_json: str = "",
    ) -> Mapping[str, object]:
        batch = _image_batch(images)
        rate = float(fps)
        if not math.isfinite(rate) or not 0.01 <= rate <= 1000.0:
            raise ValueError("fps must be finite and in 0.01..1000")
        if type(loop) is not int or not 0 <= loop <= 65535:
            raise ValueError("loop must be an integer in 0..65535")
        if type(quality) is not int or not 0 <= quality <= 100:
            raise ValueError("quality must be an integer in 0..100")
        if type(compression) is not int or not 0 <= compression <= 9:
            raise ValueError("compression must be an integer in 0..9")
        if method not in _WEBP_METHODS:
            raise ValueError(f"method must be one of {tuple(_WEBP_METHODS)}")
        if format not in ("png", "webp"):
            raise ValueError("format must be 'png' or 'webp'")
        if metadata_json and format != "png":
            raise ValueError("explicit animation metadata is supported only for PNG")
        frames = [_pillow_frame(copy_media_semantics(batch, frame)) for frame in batch]
        duration_ms = max(1, int(1000.0 / rate))
        options: dict[str, object] = {
            "save_all": True,
            "append_images": frames[1:],
            "duration": duration_ms,
            "loop": loop,
        }
        if format == "png":
            options.update(compress_level=compression, pnginfo=png_metadata(metadata_json))
            pillow_format, suffix, media_type = "PNG", ".png", "image/png"
        else:
            options.update(
                lossless=bool(lossless), quality=quality, method=_WEBP_METHODS[method], exact=True
            )
            pillow_format, suffix, media_type = "WEBP", ".webp", "image/webp"
        buffer = io.BytesIO()
        frames[0].save(buffer, format=pillow_format, **options)
        if buffer.tell() > MAX_ENCODED_IMAGE_BYTES:
            raise ValueError(
                f"encoded animation exceeds the {MAX_ENCODED_IMAGE_BYTES}-byte output limit"
            )
        destination = (
            target
            if target is not None
            else {"mount": "comfy-output", "prefix": "animations/ComfyUI"}
        )
        asset = _mount_writer().save_stream(
            destination,
            cast(BinaryIO, buffer),
            suffix=suffix,
            media_type=media_type,
            limit=MAX_ENCODED_IMAGE_BYTES,
        )
        return cls.outputs(images=batch, asset=asset)


IMAGE_IO_NODES: tuple[type[Node], ...] = (
    LoadImage,
    PaintMask,
    LoadImageOutput,
    LoadMask,
    ReadImageMetadata,
    SaveImage,
    SaveMask,
    PreviewImage,
    SaveAnimatedImage,
)

__all__ = [
    "IMAGE_ACCEPT",
    "IMAGE_ASSET",
    "IMAGE_FILE_DECODER_ID",
    "IMAGE_IO_NODES",
    "IMAGE_TYPE",
    "MASK_ASSET",
    "MASK_FILE_DECODER_ID",
    "MASK_TYPE",
    "LoadImage",
    "LoadImageOutput",
    "LoadMask",
    "PaintMask",
    "PreviewImage",
    "ReadImageMetadata",
    "SaveAnimatedImage",
    "SaveImage",
    "SaveMask",
    "decode_image_file",
    "decode_mask_file",
]
