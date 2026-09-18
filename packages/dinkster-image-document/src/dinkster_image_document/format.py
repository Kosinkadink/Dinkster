"""Closed asset-backed ImageDocument format and canonical decoding."""

from __future__ import annotations

import base64
import copy
import json
import math
import re
from dataclasses import dataclass
from typing import Any, cast

from dinkster_assets import digest_bytes, is_digest
from dinkster_values.image_codec import image_color

IMAGE_DOCUMENT_MEDIA_TYPE = "application/vnd.dinkster.image-document+json"
MAX_DOCUMENT_BYTES = 16 * 1024 * 1024
MAX_DIMENSION = 16_384
MAX_PIXELS = 100_000_000
MAX_JSON_NODES = 1_000_000
MAX_SAFE_INTEGER = 9_007_199_254_740_991
FIXED_POINT_SCALE = 1_000_000
OPACITY_MAX = 65_535
INLINE_RESOURCE_LIMIT = 4096
BLEND_MODES = (
    "normal",
    "dissolve",
    "multiply",
    "screen",
    "overlay",
    "darken",
    "lighten",
    "color_dodge",
    "color_burn",
    "hard_light",
    "soft_light",
    "difference",
    "exclusion",
    "linear_dodge",
    "linear_burn",
    "vivid_light",
    "pin_light",
    "linear_light",
    "hard_mix",
    "subtract",
    "divide",
    "grain_extract",
    "grain_merge",
    "hue",
    "saturation",
    "color",
    "luminosity",
)

_ID = re.compile(r"^(l|m|r)(0|[1-9][0-9]*)(?:-([A-Za-z0-9_-]+))?$")
_ACTOR = re.compile(r"^[A-Za-z0-9_-]+$")
_BLENDS = frozenset(BLEND_MODES)
_TRANSFORM_KEYS = frozenset({"a", "b", "c", "d", "tx", "ty"})
_RECT_KEYS = frozenset({"x", "y", "width", "height"})
_LAYER_KEYS = frozenset(
    {"id", "kind", "name", "visible", "opacity", "transform", "blendMode", "clipping", "maskIds"}
)


class InvalidDocument(ValueError):
    pass


@dataclass(frozen=True)
class ParsedDocument:
    value: dict[str, Any]
    dependencies: tuple[dict[str, object], ...]


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidDocument("duplicate JSON object key")
        result[key] = value
    return result


def _check_json(value: object, depth: int = 0, visited: list[int] | None = None) -> None:
    counter = [0] if visited is None else visited
    counter[0] += 1
    if counter[0] > MAX_JSON_NODES:
        raise InvalidDocument("JSON node count exceeds the supported limit")
    if isinstance(value, dict):
        if depth >= 64:
            raise InvalidDocument("JSON nesting exceeds the supported limit")
        children = cast(dict[object, object], value)
        if not all(isinstance(key, str) for key in children) or "__proto__" in children:
            raise InvalidDocument("JSON contains an invalid object key")
        for child in children.values():
            _check_json(child, depth + 1, counter)
    elif isinstance(value, list):
        if depth >= 64:
            raise InvalidDocument("JSON nesting exceeds the supported limit")
        for child in cast(list[object], value):
            _check_json(child, depth + 1, counter)
    elif isinstance(value, float) and not math.isfinite(value):
        raise InvalidDocument("JSON number is not finite")
    elif value is not None and not isinstance(value, (str, bool, int, float)):
        raise InvalidDocument("document contains a non-JSON value")


def _object(value: object, required: set[str], optional: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InvalidDocument("document has an invalid or open object shape")
    result = cast(dict[str, Any], value)
    optional = optional or set()
    if set(result) - required - optional or required - set(result):
        raise InvalidDocument("document has an invalid or open object shape")
    return result


def _integer(value: object, low: int, high: int = MAX_SAFE_INTEGER) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise InvalidDocument("integer is outside its allowed bounds")
    return value


def _string(value: object, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or "\0" in value or (nonempty and not value):
        raise InvalidDocument("string is invalid")
    return value


def _transform(value: object) -> dict[str, int]:
    item = _object(value, set(_TRANSFORM_KEYS), {"components"})
    if "components" in item:
        components = _object(
            item["components"],
            {
                "x",
                "y",
                "width",
                "height",
                "rotation",
                "flipHorizontal",
                "flipVertical",
                "sourceWidth",
                "sourceHeight",
            },
        )
        if affine_from_components(components) != {key: item[key] for key in _TRANSFORM_KEYS}:
            raise InvalidDocument("transform components disagree with fixed-point affine")
    return {
        key: _integer(
            item[key],
            -1_048_576_000_000 if key in {"tx", "ty"} else -1_000_000_000,
            1_048_576_000_000 if key in {"tx", "ty"} else 1_000_000_000,
        )
        for key in ("a", "b", "c", "d", "tx", "ty")
    }


def ordered_layer_ids(layers: dict[str, Any], identifiers: list[str]) -> list[str]:
    positions = {identifier: index for index, identifier in enumerate(identifiers)}
    return sorted(identifiers, key=lambda key: layers[key].get("z_index", positions[key]))


def affine_from_components(item: dict[str, Any]) -> dict[str, int]:
    for key in ("x", "y", "width", "height", "rotation", "sourceWidth", "sourceHeight"):
        if type(item[key]) not in (int, float) or not math.isfinite(item[key]):
            raise InvalidDocument("transform components must be finite numbers")
    for key in ("width", "height", "sourceWidth", "sourceHeight"):
        if not 0 < item[key] <= MAX_DIMENSION:
            raise InvalidDocument("transform dimensions are outside bounds")
    for key in ("flipHorizontal", "flipVertical"):
        if not isinstance(item[key], bool):
            raise InvalidDocument("transform flips must be booleans")
    cosine, sine = math.cos(item["rotation"]), math.sin(item["rotation"])
    sx = item["width"] / item["sourceWidth"] * (-1 if item["flipHorizontal"] else 1)
    sy = item["height"] / item["sourceHeight"] * (-1 if item["flipVertical"] else 1)
    a, b, c, d = cosine * sx, sine * sx, -sine * sy, cosine * sy
    tx = item["x"] + item["width"] / 2 - (a * item["sourceWidth"] + c * item["sourceHeight"]) / 2
    ty = item["y"] + item["height"] / 2 - (b * item["sourceWidth"] + d * item["sourceHeight"]) / 2
    return {
        key: round(value * FIXED_POINT_SCALE)
        for key, value in zip(("a", "b", "c", "d", "tx", "ty"), (a, b, c, d, tx, ty), strict=True)
    }


def _rect(value: object) -> dict[str, int]:
    item = _object(value, set(_RECT_KEYS))
    return {
        "x": _integer(item["x"], 0, MAX_DIMENSION),
        "y": _integer(item["y"], 0, MAX_DIMENSION),
        "width": _integer(item["width"], 1, MAX_DIMENSION),
        "height": _integer(item["height"], 1, MAX_DIMENSION),
    }


def _ids(value: object, maximum: int) -> list[str]:
    if not isinstance(value, list):
        raise InvalidDocument("identifier list exceeds its allowed shape")
    items = cast(list[object], value)
    if len(items) > maximum:
        raise InvalidDocument("identifier list exceeds its allowed shape")
    return [_string(item, nonempty=True) for item in items]


def validate_document(value: object) -> list[dict[str, object]]:
    _check_json(value)
    doc = _object(
        value,
        {
            "format",
            "formatVersion",
            "lineage",
            "canvas",
            "allocation",
            "rootLayerIds",
            "layers",
            "masks",
            "resources",
        },
        {"extensions"},
    )
    if doc["format"] != "dinkster-image" or _integer(doc["formatVersion"], 1, 2) not in (1, 2):
        raise InvalidDocument("unsupported image document format or version")
    _string(doc["lineage"], nonempty=True)
    if "extensions" in doc and not isinstance(doc["extensions"], dict):
        raise InvalidDocument("extensions must be an object")
    canvas = _object(
        doc["canvas"],
        {"width", "height", "colorSpace", "channelDepth", "compositing"},
        {"color", "background"} if doc["formatVersion"] == 2 else set(),
    )
    if "background" in canvas:
        if not isinstance(canvas["background"], list):
            raise InvalidDocument("canvas background must be straight RGBA16")
        background = cast(list[object], canvas["background"])
        if len(background) != 4:
            raise InvalidDocument("canvas background must be straight RGBA16")
        for component in background:
            _integer(component, 0, OPACITY_MAX)
    width = _integer(canvas["width"], 1, MAX_DIMENSION)
    height = _integer(canvas["height"], 1, MAX_DIMENSION)
    if (
        width * height > MAX_PIXELS
        or (canvas["colorSpace"], canvas["channelDepth"]) != ("srgb", 8)
        or canvas["compositing"] not in {"premultiplied-alpha", "linear-premultiplied-alpha"}
    ):
        raise InvalidDocument("canvas facts are unsupported")
    if "color" in canvas:
        try:
            image_color(canvas["color"])
        except ValueError as exc:
            raise InvalidDocument(str(exc)) from exc

    allocation = _object(doc["allocation"], {"nextOrdinal"}, {"actorCursors"})
    next_ordinal = _integer(allocation["nextOrdinal"], 0)
    cursors: dict[str, int] = {}
    if "actorCursors" in allocation:
        raw_cursors = allocation["actorCursors"]
        if not isinstance(raw_cursors, dict):
            raise InvalidDocument("actor cursors must be an object")
        for actor, cursor in cast(dict[str, object], raw_cursors).items():
            if actor == "__proto__" or _ACTOR.fullmatch(actor) is None:
                raise InvalidDocument("actor id is invalid")
            cursors[actor] = _integer(cursor, 0)

    roots = _ids(doc["rootLayerIds"], 4096)
    if not all(isinstance(doc[key], dict) for key in ("layers", "masks", "resources")):
        raise InvalidDocument("document collection exceeds its allowed shape")
    layers = cast(dict[str, object], doc["layers"])
    masks = cast(dict[str, object], doc["masks"])
    resources = cast(dict[str, object], doc["resources"])
    if len(layers) > 4096 or len(masks) > 131072 or len(resources) > 8192:
        raise InvalidDocument("document collection exceeds its allowed shape")

    coordinates: set[tuple[str | None, int]] = set()

    def allocated(key: str, expected: str) -> None:
        match = _ID.fullmatch(key)
        if match is None or match.group(1) != expected:
            raise InvalidDocument("allocated id is not canonical")
        ordinal, actor = int(match.group(2)), match.group(3)
        cursor = next_ordinal if actor is None else cursors.get(actor, 0)
        if ordinal > MAX_SAFE_INTEGER or ordinal >= cursor or (actor, ordinal) in coordinates:
            raise InvalidDocument("allocated id conflicts with its cursor")
        coordinates.add((actor, ordinal))

    parents = {key: 0 for key in layers}
    children_by_layer: dict[str, list[str]] = {}
    masks_by_layer: dict[str, list[str]] = {}
    raster_refs: list[tuple[str, dict[str, int]]] = []
    for key, raw in layers.items():
        allocated(key, "l")
        if not isinstance(raw, dict):
            raise InvalidDocument("layer identity or kind is invalid")
        item = cast(dict[str, object], raw)
        kind = item.get("kind")
        if item.get("id") != key or kind not in {"group", "raster"}:
            raise InvalidDocument("layer identity or kind is invalid")
        layer = _object(
            item,
            set(
                _LAYER_KEYS
                | ({"childLayerIds"} if kind == "group" else {"resourceId", "sourceRect"})
            ),
            ({"z_index"} | ({"isolation"} if kind == "group" else set()))
            if doc["formatVersion"] == 2
            else set(),
        )
        if "z_index" in layer:
            _integer(layer["z_index"], -MAX_SAFE_INTEGER)
        if layer.get("isolation", "isolated") not in {"isolated", "pass-through"}:
            raise InvalidDocument("invalid group isolation")
        _string(layer["name"])
        if not isinstance(layer["visible"], bool):
            raise InvalidDocument("layer property is invalid")
        _integer(layer["opacity"], 0, OPACITY_MAX)
        _transform(layer["transform"])
        if layer["blendMode"] not in _BLENDS or layer["clipping"] not in {
            "none",
            "clip-to-previous",
        }:
            raise InvalidDocument("layer property is invalid")
        masks_by_layer[key] = _ids(layer["maskIds"], 32)
        if kind == "group":
            children_by_layer[key] = _ids(layer["childLayerIds"], 4096)
        else:
            raster_refs.append(
                (_string(layer["resourceId"], nonempty=True), _rect(layer["sourceRect"]))
            )

    for children in [roots, *children_by_layer.values()]:
        if len(children) != len(set(children)):
            raise InvalidDocument("layer child is duplicated")
        for child in children:
            if child not in layers:
                raise InvalidDocument("layer reference is dangling")
            parents[child] += 1
        if children:
            first = ordered_layer_ids(layers, children)[0]
            if cast(dict[str, object], layers[first])["clipping"] == "clip-to-previous":
                raise InvalidDocument("clipped layer has no previous sibling")
    if any(count != 1 for count in parents.values()):
        raise InvalidDocument("each layer must have exactly one parent")
    visited: set[str] = set()

    def visit(layer_id: str, depth: int) -> None:
        if depth > 64 or layer_id in visited:
            raise InvalidDocument("layer tree is cyclic or too deep")
        visited.add(layer_id)
        for child_id in children_by_layer.get(layer_id, []):
            visit(child_id, depth + 1)

    for root in roots:
        visit(root, 1)
    if visited != set(layers):
        raise InvalidDocument("layer is not rooted")

    owners: dict[str, int] = {}
    for layer_id, mask_ids in masks_by_layer.items():
        if len(mask_ids) != len(set(mask_ids)):
            raise InvalidDocument("mask is duplicated")
        for mask_id in mask_ids:
            candidate = masks.get(mask_id)
            if (
                not isinstance(candidate, dict)
                or cast(dict[str, object], candidate).get("ownerLayerId") != layer_id
            ):
                raise InvalidDocument("mask owner is invalid")
            owners[mask_id] = owners.get(mask_id, 0) + 1
    for key, raw in masks.items():
        allocated(key, "m")
        mask = _object(
            raw,
            {
                "id",
                "kind",
                "ownerLayerId",
                "enabled",
                "invert",
                "opacity",
                "transform",
                "combineMode",
                "channel",
                "resourceId",
                "sourceRect",
            },
        )
        if (
            mask["id"] != key
            or mask["kind"] != "raster"
            or owners.get(key) != 1
            or mask["ownerLayerId"] not in layers
            or not isinstance(mask["enabled"], bool)
            or not isinstance(mask["invert"], bool)
        ):
            raise InvalidDocument("mask identity or ownership is invalid")
        _integer(mask["opacity"], 0, OPACITY_MAX)
        _transform(mask["transform"])
        if mask["combineMode"] not in {"multiply", "add", "subtract", "intersect"} or mask[
            "channel"
        ] not in {"alpha", "luminance"}:
            raise InvalidDocument("mask property is invalid")
        raster_refs.append((_string(mask["resourceId"], nonempty=True), _rect(mask["sourceRect"])))

    dependencies: list[dict[str, object]] = []
    dimensions: dict[str, tuple[int, int]] = {}
    for key, raw in resources.items():
        allocated(key, "r")
        resource = _object(
            raw,
            {
                "id",
                "kind",
                "digest",
                "byteSize",
                "mediaType",
                "width",
                "height",
                "colorSpace",
                "channelDepth",
                "alphaMode",
            },
            {"inline"} if doc["formatVersion"] == 2 else set(),
        )
        resource_width = _integer(resource["width"], 1, MAX_DIMENSION)
        resource_height = _integer(resource["height"], 1, MAX_DIMENSION)
        digest = resource["digest"]
        if (
            resource["id"] != key
            or resource["kind"] != "raster"
            or not isinstance(digest, str)
            or not is_digest(digest)
            or resource_width * resource_height > MAX_PIXELS
            or resource["mediaType"] not in {"image/png", "image/jpeg", "image/webp"}
            or (resource["colorSpace"], resource["channelDepth"]) != ("srgb", 8)
            or resource["alphaMode"] not in {"straight", "premultiplied", "opaque"}
            or (resource["mediaType"] == "image/jpeg" and resource["alphaMode"] != "opaque")
        ):
            raise InvalidDocument("resource descriptor is invalid")
        byte_size = _integer(resource["byteSize"], 1)
        if "inline" in resource:
            encoded = _string(resource["inline"])
            if len(encoded) > (INLINE_RESOURCE_LIMIT + 2) // 3 * 4:
                raise InvalidDocument("inline resource exceeds threshold")
            try:
                data = base64.b64decode(encoded, validate=True)
            except ValueError as error:
                raise InvalidDocument("invalid inline resource") from error
            if (
                len(data) > INLINE_RESOURCE_LIMIT
                or len(data) != byte_size
                or digest_bytes(data) != digest
            ):
                raise InvalidDocument("inline resource does not match descriptor")
        dimensions[key] = (resource_width, resource_height)
        if "inline" in resource:
            continue
        dependencies.append(
            {
                "resourceId": key,
                "digest": digest,
                "byteSize": byte_size,
                "mediaType": resource["mediaType"],
                "width": resource_width,
                "height": resource_height,
                "colorSpace": "srgb",
                "channelDepth": 8,
                "alphaMode": resource["alphaMode"],
            }
        )
    for resource_id, rect in raster_refs:
        if (
            resource_id not in dimensions
            or rect["x"] + rect["width"] > dimensions[resource_id][0]
            or rect["y"] + rect["height"] > dimensions[resource_id][1]
        ):
            raise InvalidDocument("source rectangle exceeds its resource")
    dependencies.sort(key=lambda dependency: cast(str, dependency["resourceId"]))
    return dependencies


def _json_string(value: str) -> str:
    encoded = json.dumps(value, ensure_ascii=False)
    return "".join(
        f"\\u{ord(character):04x}" if 0xD800 <= ord(character) <= 0xDFFF else character
        for character in encoded
    )


def _json_number(value: float) -> str:
    if not math.isfinite(value):
        raise InvalidDocument("JSON number is not finite")
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    rendered = repr(abs(value)).lower()
    if "e" not in rendered:
        return sign + (rendered[:-2] if rendered.endswith(".0") else rendered)
    coefficient, exponent_text = rendered.split("e")
    exponent = int(exponent_text)
    whole, _, fraction = coefficient.partition(".")
    digits = whole + fraction
    decimal_position = len(whole) + exponent
    if 0 < decimal_position <= 21:
        if decimal_position >= len(digits):
            return sign + digits + "0" * (decimal_position - len(digits))
        return sign + digits[:decimal_position] + "." + digits[decimal_position:]
    if -6 < decimal_position <= 0:
        return sign + "0." + "0" * (-decimal_position) + digits
    mantissa = digits[0] + ("." + digits[1:] if len(digits) > 1 else "")
    scientific_exponent = decimal_position - 1
    exponent_sign = "+" if scientific_exponent >= 0 else ""
    return f"{sign}{mantissa}e{exponent_sign}{scientific_exponent}"


def canonical_json(value: object) -> str:
    if value is None or isinstance(value, bool):
        return json.dumps(value)
    if isinstance(value, str):
        return _json_string(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _json_number(value)
    if isinstance(value, list):
        return "[" + ",".join(canonical_json(item) for item in cast(list[object], value)) + "]"
    if isinstance(value, dict):
        entries = cast(dict[str, object], value).items()
        ordered = sorted(
            entries, key=lambda entry: entry[0].encode("utf-16-be", errors="surrogatepass")
        )
        return (
            "{"
            + ",".join(f"{_json_string(key)}:{canonical_json(item)}" for key, item in ordered)
            + "}"
        )
    raise InvalidDocument("JSON value cannot be canonicalized")


def _parse_integer(value: str) -> int | float:
    parsed = int(value)
    return parsed if -MAX_SAFE_INTEGER <= parsed <= MAX_SAFE_INTEGER else float(value)


def decode_document(data: bytes) -> ParsedDocument:
    if not data or len(data) > MAX_DOCUMENT_BYTES:
        raise InvalidDocument("image document exceeds its byte limit")
    try:
        text = data.decode("utf-8")
        loaded: object = json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_int=_parse_integer,
            parse_constant=lambda _: (_ for _ in ()).throw(
                InvalidDocument("non-finite JSON number")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise InvalidDocument("image document is not valid JSON") from error
    if not isinstance(loaded, dict):
        raise InvalidDocument("document root must be an object")
    value = cast(dict[str, object], loaded)
    dependencies = validate_document(value)
    if text != canonical_json(value):
        raise InvalidDocument("image document bytes are not canonical JSON")
    migrated = copy.deepcopy(value)
    migrated["formatVersion"] = 2
    return ParsedDocument(migrated, tuple(dependencies))
