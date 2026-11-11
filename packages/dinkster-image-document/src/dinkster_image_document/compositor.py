"""Compositor widget values shared by the serving host and image worker."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from dinkster_assets import is_digest
from dinkster_values import COMPOSITOR_TYPE, TypeRegistry

from .format import BLEND_MODES, canonical_json

MAX_COMPOSITOR_LAYERS = 50
MAX_COMPOSITOR_DIMENSION = 16_384
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


def coerce_compositor_recipe(obj: object) -> CompositorRecipe:
    if isinstance(obj, CompositorRecipe):
        return obj
    return CompositorRecipe.from_record(obj)


def coerce_delta(obj: object) -> dict[str, Any]:
    if obj == {}:
        return {"version": 2, "documentDigest": None, "commands": []}
    if isinstance(obj, Mapping) and cast(Mapping[str, Any], obj).get("version") == 2:
        value = cast(dict[str, Any], obj)
        if set(value) != {"version", "documentDigest", "commands"}:
            raise ValueError("compositor delta requires version, documentDigest and commands")
        if value["documentDigest"] is not None and not is_digest(value["documentDigest"]):
            raise ValueError("invalid compositor document digest")
        if not isinstance(value["commands"], list):
            raise ValueError("compositor commands must be an array")
        return json.loads(canonical_json(value))
    if isinstance(obj, Mapping):
        value = cast(dict[str, Any], obj)
        if "canvas" in value and "w" in value["canvas"]:
            return json.loads(canonical_json(value))
    legacy = coerce_compositor_recipe(cast(object, obj))
    if not legacy.layers:
        return {"version": 2, "documentDigest": None, "commands": []}
    return legacy.to_record()


def encode_delta(obj: object) -> bytes:
    return canonical_json(coerce_delta(obj)).encode("utf-8")


def decode_delta(data: bytes) -> object:
    return coerce_delta(json.loads(data))


def validate_delta_encoded(data: bytes, _metadata: Mapping[str, object]) -> None:
    decode_delta(data)


def register_compositor_type(registry: TypeRegistry) -> None:
    if COMPOSITOR_TYPE not in registry:
        registry.register(
            COMPOSITOR_TYPE,
            encode=encode_delta,
            decode=decode_delta,
            coerce=coerce_delta,
            meta=lambda obj: {"version": coerce_delta(obj)["version"]},
            validate_encoded=validate_delta_encoded,
            asset_references_complete=True,
        )
