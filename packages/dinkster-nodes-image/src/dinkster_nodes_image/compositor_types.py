"""Runtime layer values and the durable compositor recipe codec."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from dinkster_api.v1 import decode_image_array, encode_image_array
from dinkster_image_document.format import BLEND_MODES

from .support import check_output_size

LAYERS_TYPE = "dinkster.layers"
COMPOSITOR_TYPE = "dinkster.compositor"
MAX_COMPOSITOR_LAYERS = 50
MAX_COMPOSITOR_DIMENSION = 16_384

_LAYERS_MAGIC = b"DINKSTER-LAYERS\x00\x01"
_LAYERS_HEADER_LIMIT = 256 * 1024
_FINGERPRINT_PATTERN = re.compile(r"[0-9a-f]{64}")
_LAYER_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_COLOR_PATTERN = re.compile(r"#[0-9a-fA-F]{6}")


def _finite_number(value: object, subject: str) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{subject} must be a number")
    try:
        result = float(cast("int | float", value))
    except OverflowError as exc:
        raise ValueError(f"{subject} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{subject} must be finite")
    return result


def _strict_record(obj: object, fields: set[str], subject: str) -> Mapping[str, object]:
    if not isinstance(obj, Mapping):
        raise TypeError(f"{subject} must be an object")
    record = cast("Mapping[str, object]", obj)
    if set(record) != fields:
        raise ValueError(f"{subject} requires exactly {', '.join(sorted(fields))}")
    return record


def _validate_name(value: object, subject: str) -> None:
    if type(value) is not str or not 1 <= len(value) <= 256:
        raise ValueError(f"{subject} must contain 1 to 256 characters")
    if any("\ud800" <= character <= "\udfff" for character in value):
        raise ValueError(f"{subject} must not contain surrogate code points")


def _image_array(obj: object, subject: str) -> np.ndarray:
    array = np.asarray(obj, dtype=np.float32)
    if array.ndim != 4 or array.shape[0] < 1 or array.shape[3] not in (1, 3, 4):
        raise ValueError(f"{subject} must have non-empty BHWC shape with 1, 3, or 4 channels")
    if array.shape[1] < 1 or array.shape[2] < 1:
        raise ValueError(f"{subject} dimensions must be non-zero")
    check_output_size(tuple(int(size) for size in array.shape))
    return np.frombuffer(array.tobytes(order="C"), dtype=np.float32).reshape(array.shape)


def _mask_array(obj: object, subject: str) -> np.ndarray:
    array = np.asarray(obj, dtype=np.float32)
    if array.ndim != 3 or any(size < 1 for size in array.shape):
        raise ValueError(f"{subject} must have non-empty BHW shape")
    check_output_size((*tuple(int(size) for size in array.shape), 1))
    return np.frombuffer(array.tobytes(order="C"), dtype=np.float32).reshape(array.shape)


@dataclass(frozen=True, eq=False)
class CompositorSourceLayer:
    """One runtime layer source with its native placement."""

    image: np.ndarray
    mask: np.ndarray | None
    name: str
    x: float
    y: float
    width: float
    height: float
    rotation: float = 0.0
    opacity: float = 1.0
    blend_mode: str = "normal"
    visible: bool = True
    flip_horizontal: bool = False
    flip_vertical: bool = False

    def __post_init__(self) -> None:
        image = _image_array(self.image, "layer image")
        mask = None if self.mask is None else _mask_array(self.mask, "layer mask")
        if mask is not None:
            if mask.shape[0] not in (1, image.shape[0]):
                raise ValueError("layer mask batch must be one or match the image batch")
            if mask.shape[1:3] != image.shape[1:3]:
                raise ValueError("layer mask dimensions must match the image dimensions")
        _validate_name(self.name, "layer name")
        values = {
            "layer x": self.x,
            "layer y": self.y,
            "layer width": self.width,
            "layer height": self.height,
            "layer rotation": self.rotation,
            "layer opacity": self.opacity,
        }
        normalized = {name: _finite_number(value, name) for name, value in values.items()}
        if not 1.0 <= normalized["layer width"] <= MAX_COMPOSITOR_DIMENSION:
            raise ValueError(f"layer width must be between 1 and {MAX_COMPOSITOR_DIMENSION}")
        if not 1.0 <= normalized["layer height"] <= MAX_COMPOSITOR_DIMENSION:
            raise ValueError(f"layer height must be between 1 and {MAX_COMPOSITOR_DIMENSION}")
        if not 0.0 <= normalized["layer opacity"] <= 1.0:
            raise ValueError("layer opacity must be between 0 and 1")
        if self.blend_mode not in BLEND_MODES:
            raise ValueError(f"unknown compositor blend mode: {self.blend_mode}")
        for field in ("visible", "flip_horizontal", "flip_vertical"):
            if type(getattr(self, field)) is not bool:
                raise TypeError(f"layer {field} must be a boolean")
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "x", normalized["layer x"])
        object.__setattr__(self, "y", normalized["layer y"])
        object.__setattr__(self, "width", normalized["layer width"])
        object.__setattr__(self, "height", normalized["layer height"])
        object.__setattr__(self, "rotation", normalized["layer rotation"])
        object.__setattr__(self, "opacity", normalized["layer opacity"])

    @property
    def frame_count(self) -> int:
        return int(self.image.shape[0])

    def frame(self, index: int) -> CompositorSourceLayer:
        mask_index = 0 if self.mask is not None and self.mask.shape[0] == 1 else index
        return CompositorSourceLayer(
            image=self.image[index : index + 1],
            mask=None if self.mask is None else self.mask[mask_index : mask_index + 1],
            name=self.name if self.frame_count == 1 else f"{self.name} {index + 1}",
            x=self.x,
            y=self.y,
            width=self.width,
            height=self.height,
            rotation=self.rotation,
            opacity=self.opacity,
            blend_mode=self.blend_mode,
            visible=self.visible,
            flip_horizontal=self.flip_horizontal,
            flip_vertical=self.flip_vertical,
        )


@dataclass(frozen=True)
class LayerStack:
    """Ordered runtime layers; pixel arrays never enter the document recipe."""

    layers: tuple[CompositorSourceLayer, ...]
    canvas_width: int | None = None
    canvas_height: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(cast("object", self.layers), tuple) or not self.layers:
            raise ValueError("layer stack must contain an immutable non-empty tuple")
        if any(type(layer) is not CompositorSourceLayer for layer in self.layers):
            raise TypeError("layer stack members must be CompositorSourceLayer values")
        if sum(layer.frame_count for layer in self.layers) > MAX_COMPOSITOR_LAYERS:
            raise ValueError(f"layer stack cannot exceed {MAX_COMPOSITOR_LAYERS} expanded layers")
        if (self.canvas_width is None) != (self.canvas_height is None):
            raise ValueError("layer stack canvas width and height must be declared together")
        if self.canvas_width is not None:
            for name, value in (
                ("canvas width", self.canvas_width),
                ("canvas height", self.canvas_height),
            ):
                if type(value) is not int or not 1 <= value <= MAX_COMPOSITOR_DIMENSION:
                    raise ValueError(f"{name} must be between 1 and {MAX_COMPOSITOR_DIMENSION}")

    def expanded(self) -> tuple[CompositorSourceLayer, ...]:
        return tuple(
            layer.frame(index) for layer in self.layers for index in range(layer.frame_count)
        )


@dataclass(frozen=True)
class CompositorTransform:
    x: float
    y: float
    width: float
    height: float
    rotation: float

    def __post_init__(self) -> None:
        values = tuple(
            _finite_number(value, f"compositor transform {name}")
            for name, value in (
                ("x", self.x),
                ("y", self.y),
                ("width", self.width),
                ("height", self.height),
                ("rotation", self.rotation),
            )
        )
        if not 1.0 <= values[2] <= MAX_COMPOSITOR_DIMENSION:
            raise ValueError(
                f"compositor transform width must be between 1 and {MAX_COMPOSITOR_DIMENSION}"
            )
        if not 1.0 <= values[3] <= MAX_COMPOSITOR_DIMENSION:
            raise ValueError(
                f"compositor transform height must be between 1 and {MAX_COMPOSITOR_DIMENSION}"
            )
        for field, value in zip(("x", "y", "width", "height", "rotation"), values, strict=True):
            object.__setattr__(self, field, value)

    def to_record(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "rotation": self.rotation,
        }

    @classmethod
    def from_record(cls, obj: object, subject: str) -> CompositorTransform:
        record = _strict_record(obj, {"x", "y", "width", "height", "rotation"}, subject)
        return cls(
            x=cast("float", record["x"]),
            y=cast("float", record["y"]),
            width=cast("float", record["width"]),
            height=cast("float", record["height"]),
            rotation=cast("float", record["rotation"]),
        )


@dataclass(frozen=True)
class CompositorLayer:
    id: str
    source_index: int
    name: str
    visible: bool
    opacity: float
    blend_mode: str
    transform: CompositorTransform
    flip_horizontal: bool
    flip_vertical: bool

    def __post_init__(self) -> None:
        if type(self.id) is not str or not _LAYER_ID_PATTERN.fullmatch(self.id):
            raise ValueError("compositor layer id has invalid syntax")
        if type(self.source_index) is not int or self.source_index < 0:
            raise ValueError("compositor layer source must be a non-negative integer")
        _validate_name(self.name, "compositor layer name")
        if type(self.visible) is not bool:
            raise TypeError("compositor layer visible must be a boolean")
        opacity = _finite_number(self.opacity, "compositor layer opacity")
        if not 0.0 <= opacity <= 1.0:
            raise ValueError("compositor layer opacity must be between 0 and 1")
        if self.blend_mode not in BLEND_MODES:
            raise ValueError(f"unknown compositor blend mode: {self.blend_mode}")
        if type(self.transform) is not CompositorTransform:
            raise TypeError("compositor layer transform must be a CompositorTransform")
        if type(self.flip_horizontal) is not bool or type(self.flip_vertical) is not bool:
            raise TypeError("compositor layer flips must be booleans")
        object.__setattr__(self, "opacity", opacity)

    def to_record(self) -> dict[str, object]:
        return {
            "id": self.id,
            "source": self.source_index,
            "name": self.name,
            "visible": self.visible,
            "opacity": self.opacity,
            "blend": self.blend_mode,
            "transform": self.transform.to_record(),
            "flipH": self.flip_horizontal,
            "flipV": self.flip_vertical,
        }

    @classmethod
    def from_record(cls, obj: object, index: int) -> CompositorLayer:
        subject = f"compositor layer {index}"
        fields = {
            "id",
            "source",
            "name",
            "visible",
            "opacity",
            "blend",
            "transform",
            "flipH",
            "flipV",
        }
        record = _strict_record(obj, fields, subject)
        return cls(
            id=cast("str", record["id"]),
            source_index=cast("int", record["source"]),
            name=cast("str", record["name"]),
            visible=cast("bool", record["visible"]),
            opacity=cast("float", record["opacity"]),
            blend_mode=cast("str", record["blend"]),
            transform=CompositorTransform.from_record(record["transform"], f"{subject} transform"),
            flip_horizontal=cast("bool", record["flipH"]),
            flip_vertical=cast("bool", record["flipV"]),
        )


@dataclass(frozen=True)
class CompositorRecipe:
    """Versioned node-local edit recipe replayed only against exact sources."""

    version: int
    input_fingerprints: tuple[str, ...]
    canvas_width: int
    canvas_height: int
    background_color: str
    background_opacity: float
    background_visible: bool
    layers: tuple[CompositorLayer, ...]

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise ValueError("compositor recipe version must be 1")
        if not isinstance(cast("object", self.input_fingerprints), tuple):
            raise TypeError("compositor inputs must be an immutable tuple")
        if len(self.input_fingerprints) > MAX_COMPOSITOR_LAYERS:
            raise ValueError(f"compositor recipe cannot exceed {MAX_COMPOSITOR_LAYERS} sources")
        if any(
            type(fingerprint) is not str or not _FINGERPRINT_PATTERN.fullmatch(fingerprint)
            for fingerprint in self.input_fingerprints
        ):
            raise ValueError("compositor input fingerprints must be lowercase SHA-256 hex strings")
        if not isinstance(cast("object", self.layers), tuple):
            raise TypeError("compositor layers must be an immutable tuple")
        if not self.input_fingerprints:
            if self.layers:
                raise ValueError("empty compositor recipe must not contain layers")
            if (
                self.canvas_width != 1
                or self.canvas_height != 1
                or self.background_color != "#000000"
                or self.background_opacity != 0.0
                or self.background_visible is not False
            ):
                raise ValueError("empty compositor recipe must use canonical internal defaults")
            return
        for name, value in (("width", self.canvas_width), ("height", self.canvas_height)):
            if type(value) is not int or not 1 <= value <= MAX_COMPOSITOR_DIMENSION:
                raise ValueError(
                    f"compositor canvas {name} must be between 1 and {MAX_COMPOSITOR_DIMENSION}"
                )
        if type(self.background_color) is not str or not _COLOR_PATTERN.fullmatch(
            self.background_color
        ):
            raise ValueError("compositor background color must be #RRGGBB")
        opacity = _finite_number(self.background_opacity, "compositor background opacity")
        if not 0.0 <= opacity <= 1.0:
            raise ValueError("compositor background opacity must be between 0 and 1")
        if type(self.background_visible) is not bool:
            raise TypeError("compositor background visible must be a boolean")
        if len(self.layers) != len(self.input_fingerprints):
            raise ValueError("compositor recipe must contain exactly one layer per source")
        if any(type(layer) is not CompositorLayer for layer in self.layers):
            raise TypeError("compositor recipe layers must be CompositorLayer values")
        if len({layer.id for layer in self.layers}) != len(self.layers):
            raise ValueError("compositor layer ids must be unique")
        if sorted(layer.source_index for layer in self.layers) != list(range(len(self.layers))):
            raise ValueError("compositor source values must be a permutation of the inputs")
        object.__setattr__(self, "background_color", self.background_color.lower())
        object.__setattr__(self, "background_opacity", opacity)

    def to_record(self) -> dict[str, object]:
        if not self.input_fingerprints:
            return {"version": self.version, "inputs": [], "layers": []}
        return {
            "version": self.version,
            "inputs": list(self.input_fingerprints),
            "canvas": {"width": self.canvas_width, "height": self.canvas_height},
            "background": {
                "color": self.background_color,
                "opacity": self.background_opacity,
                "visible": self.background_visible,
            },
            "layers": [layer.to_record() for layer in self.layers],
        }

    @classmethod
    def from_record(cls, obj: object) -> CompositorRecipe:
        if isinstance(obj, Mapping):
            candidate = cast("Mapping[object, object]", obj)
            if set(candidate) == {"version", "inputs", "layers"}:
                record = dict(candidate)
                if record["inputs"] != [] or record["layers"] != []:
                    raise ValueError("short compositor recipe form must be exactly empty")
                return cls(
                    version=cast("int", record["version"]),
                    input_fingerprints=(),
                    canvas_width=1,
                    canvas_height=1,
                    background_color="#000000",
                    background_opacity=0.0,
                    background_visible=False,
                    layers=(),
                )
        record = _strict_record(
            cast("object", obj),
            {"version", "inputs", "canvas", "background", "layers"},
            COMPOSITOR_TYPE,
        )
        raw_fingerprints = record["inputs"]
        raw_layers = record["layers"]
        if not isinstance(raw_fingerprints, Sequence) or isinstance(raw_fingerprints, (str, bytes)):
            raise TypeError("compositor inputs must be an array")
        if not isinstance(raw_layers, Sequence) or isinstance(raw_layers, (str, bytes)):
            raise TypeError("compositor layers must be an array")
        canvas = _strict_record(record["canvas"], {"width", "height"}, "compositor canvas")
        background = _strict_record(
            record["background"],
            {"color", "opacity", "visible"},
            "compositor background",
        )
        return cls(
            version=cast("int", record["version"]),
            input_fingerprints=tuple(cast("Sequence[str]", raw_fingerprints)),
            canvas_width=cast("int", canvas["width"]),
            canvas_height=cast("int", canvas["height"]),
            background_color=cast("str", background["color"]),
            background_opacity=cast("float", background["opacity"]),
            background_visible=cast("bool", background["visible"]),
            layers=tuple(
                CompositorLayer.from_record(layer, index)
                for index, layer in enumerate(cast("Sequence[object]", raw_layers))
            ),
        )


def coerce_layer_stack(obj: object) -> LayerStack:
    if not isinstance(obj, LayerStack):
        raise TypeError(f"{LAYERS_TYPE} expects a LayerStack")
    return obj


def coerce_compositor_recipe(obj: object) -> CompositorRecipe:
    if isinstance(obj, CompositorRecipe):
        return obj
    return CompositorRecipe.from_record(obj)


def encode_compositor_recipe(obj: object) -> bytes:
    return json.dumps(
        coerce_compositor_recipe(obj).to_record(),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def decode_compositor_recipe(data: bytes) -> object:
    try:
        record = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(f"invalid {COMPOSITOR_TYPE} payload") from exc
    return coerce_compositor_recipe(record)


def compositor_recipe_meta(obj: object) -> Mapping[str, object]:
    recipe = coerce_compositor_recipe(obj)
    return {
        "version": recipe.version,
        "inputs": len(recipe.input_fingerprints),
        "canvas": (recipe.canvas_width, recipe.canvas_height),
    }


def _source_layer_record(
    layer: CompositorSourceLayer,
    image_size: int,
    mask_size: int,
) -> dict[str, object]:
    return {
        "imageBytes": image_size,
        "maskBytes": mask_size,
        "name": layer.name,
        "x": layer.x,
        "y": layer.y,
        "width": layer.width,
        "height": layer.height,
        "rotation": layer.rotation,
        "opacity": layer.opacity,
        "blendMode": layer.blend_mode,
        "visible": layer.visible,
        "flipHorizontal": layer.flip_horizontal,
        "flipVertical": layer.flip_vertical,
    }


def encode_layer_stack(obj: object) -> bytes:
    stack = coerce_layer_stack(obj)
    chunks: list[bytes] = []
    records: list[dict[str, object]] = []
    for layer in stack.layers:
        image_bytes = encode_image_array(layer.image)
        mask_bytes = b"" if layer.mask is None else encode_image_array(layer.mask)
        chunks.extend((image_bytes, mask_bytes))
        records.append(_source_layer_record(layer, len(image_bytes), len(mask_bytes)))
    header = json.dumps(
        {
            "canvasWidth": stack.canvas_width,
            "canvasHeight": stack.canvas_height,
            "layers": records,
        },
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    if len(header) > _LAYERS_HEADER_LIMIT:
        raise ValueError(f"{LAYERS_TYPE} header exceeds {_LAYERS_HEADER_LIMIT} bytes")
    return b"".join((_LAYERS_MAGIC, len(header).to_bytes(4, "little"), header, *chunks))


def decode_layer_stack(data: bytes) -> object:
    prefix = len(_LAYERS_MAGIC) + 4
    if len(data) < prefix or data[: len(_LAYERS_MAGIC)] != _LAYERS_MAGIC:
        raise ValueError(f"invalid {LAYERS_TYPE} payload")
    header_size = int.from_bytes(data[len(_LAYERS_MAGIC) : prefix], "little")
    header_end = prefix + header_size
    if header_size > _LAYERS_HEADER_LIMIT or header_end > len(data):
        raise ValueError(f"invalid {LAYERS_TYPE} header")
    try:
        header = json.loads(data[prefix:header_end])
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(f"invalid {LAYERS_TYPE} header") from exc
    record = _strict_record(
        header,
        {"canvasWidth", "canvasHeight", "layers"},
        LAYERS_TYPE,
    )
    raw_layers = record["layers"]
    if not isinstance(raw_layers, list):
        raise ValueError(f"{LAYERS_TYPE} layers must be an array")
    layer_records = cast("list[object]", raw_layers)
    if not 1 <= len(layer_records) <= MAX_COMPOSITOR_LAYERS:
        raise ValueError(f"{LAYERS_TYPE} layers must contain 1 to {MAX_COMPOSITOR_LAYERS} entries")
    cursor = header_end
    layers: list[CompositorSourceLayer] = []
    fields = {
        "imageBytes",
        "maskBytes",
        "name",
        "x",
        "y",
        "width",
        "height",
        "rotation",
        "opacity",
        "blendMode",
        "visible",
        "flipHorizontal",
        "flipVertical",
    }
    for index, raw_layer in enumerate(layer_records):
        layer_record = _strict_record(raw_layer, fields, f"{LAYERS_TYPE} layer {index}")
        image_size = layer_record["imageBytes"]
        mask_size = layer_record["maskBytes"]
        if (
            type(image_size) is not int
            or image_size < 1
            or type(mask_size) is not int
            or mask_size < 0
        ):
            raise ValueError(f"{LAYERS_TYPE} layer {index} has invalid payload sizes")
        image_end = cursor + image_size
        mask_end = image_end + mask_size
        if mask_end > len(data):
            raise ValueError(f"{LAYERS_TYPE} layer {index} payload is truncated")
        try:
            image = decode_image_array(data[cursor:image_end])
            mask = None if mask_size == 0 else decode_image_array(data[image_end:mask_end])
        except Exception as exc:
            raise ValueError(f"invalid {LAYERS_TYPE} layer {index} array payload") from exc
        layers.append(
            CompositorSourceLayer(
                image=cast("np.ndarray", image),
                mask=cast("np.ndarray | None", mask),
                name=cast("str", layer_record["name"]),
                x=cast("float", layer_record["x"]),
                y=cast("float", layer_record["y"]),
                width=cast("float", layer_record["width"]),
                height=cast("float", layer_record["height"]),
                rotation=cast("float", layer_record["rotation"]),
                opacity=cast("float", layer_record["opacity"]),
                blend_mode=cast("str", layer_record["blendMode"]),
                visible=cast("bool", layer_record["visible"]),
                flip_horizontal=cast("bool", layer_record["flipHorizontal"]),
                flip_vertical=cast("bool", layer_record["flipVertical"]),
            )
        )
        cursor = mask_end
    if cursor != len(data):
        raise ValueError(f"invalid {LAYERS_TYPE} payload trailing bytes")
    return LayerStack(
        tuple(layers),
        canvas_width=cast("int | None", record["canvasWidth"]),
        canvas_height=cast("int | None", record["canvasHeight"]),
    )


def layer_stack_meta(obj: object) -> Mapping[str, object]:
    stack = coerce_layer_stack(obj)
    return {
        "layers": sum(layer.frame_count for layer in stack.layers),
        "canvas": (
            None
            if stack.canvas_width is None
            else (stack.canvas_width, cast("int", stack.canvas_height))
        ),
    }


def source_layer_fingerprint(layer: CompositorSourceLayer) -> str:
    """Content and native placement fingerprint for recipe replay."""

    digest = hashlib.sha256()
    for label, array in ((b"image", layer.image), (b"mask", layer.mask)):
        digest.update(len(label).to_bytes(8, "big"))
        digest.update(label)
        if array is None:
            part = b"none"
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
        else:
            contiguous = np.ascontiguousarray(array, dtype="<f4")
            shape = json.dumps(contiguous.shape, separators=(",", ":")).encode("ascii")
            pixels = memoryview(contiguous).cast("B")
            for part in (shape, pixels):
                digest.update(len(part).to_bytes(8, "big"))
                digest.update(part)
    placement = json.dumps(
        _source_layer_record(layer, 0, 0),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest.update(len(placement).to_bytes(8, "big"))
    digest.update(placement)
    return digest.hexdigest()


__all__ = [
    "BLEND_MODES",
    "COMPOSITOR_TYPE",
    "LAYERS_TYPE",
    "MAX_COMPOSITOR_DIMENSION",
    "MAX_COMPOSITOR_LAYERS",
    "CompositorLayer",
    "CompositorRecipe",
    "CompositorSourceLayer",
    "CompositorTransform",
    "LayerStack",
    "coerce_compositor_recipe",
    "coerce_layer_stack",
    "compositor_recipe_meta",
    "decode_compositor_recipe",
    "decode_layer_stack",
    "encode_compositor_recipe",
    "encode_layer_stack",
    "layer_stack_meta",
    "source_layer_fingerprint",
]
