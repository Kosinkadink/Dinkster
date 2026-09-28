"""Deterministic integer ImageDocument CPU reference renderer."""

from __future__ import annotations

import base64
import io
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from dinkster_assets import digest_bytes
from dinkster_values import IMAGE_DOCUMENT_RESOURCE_LIMIT_BYTES, encode_canonical_png
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError, features
from PIL import __version__ as pillow_version

from .blend import blend, dissolve_alpha, linear_to_srgb, srgb_to_linear
from .format import (
    FIXED_POINT_SCALE,
    OPACITY_MAX,
    InvalidDocument,
    ParsedDocument,
    ordered_layer_ids,
)

RENDER_PROFILE = "dinkster-image-document-v2-cpu-reference"
RENDERER_CONTRACT = ";".join(
    (
        "2",
        f"pillow={pillow_version}",
        f"jpeg={getattr(Image.core, 'jpeglib_version', 'none')}",
        f"webp={features.version('webp') or 'none'}",
    )
)
OUTPUT_ENCODING = "image/png;dinkster-canonical=1"
MAX_RENDER_PIXELS = 4_194_304
MAX_RENDER_WORK_PIXELS = 67_108_864
MAX_RESOURCE_MEMORY = IMAGE_DOCUMENT_RESOURCE_LIMIT_BYTES
ONE = 65_535
MAX_INT64 = 9_223_372_036_854_775_807

Pixels = NDArray[np.uint16]
ResourceReader = Callable[[str], bytes]


@dataclass(frozen=True)
class RenderSelector:
    kind: Literal["composite", "layer", "mask"]
    id: str | None = None


@dataclass(frozen=True)
class RenderResult:
    pixels: Pixels
    png: bytes
    width: int
    height: int


def parse_selector(value: str) -> RenderSelector:
    if value == "composite":
        return RenderSelector("composite")
    kind, separator, identifier = value.partition(":")
    if separator and identifier and kind in {"layer", "mask"}:
        return RenderSelector(cast(Literal["layer", "mask"], kind), identifier)
    raise InvalidDocument("render selector must be composite, layer:<id>, or mask:<id>")


def _divide(value: NDArray[np.int64], divisor: int | NDArray[np.int64] = ONE) -> NDArray[np.int64]:
    return (value + divisor // 2) // divisor


def _multiply(left: NDArray[np.int64], right: NDArray[np.int64]) -> NDArray[np.int64]:
    return _divide(left * right)


def _decode_resource(descriptor: dict[str, Any], data: bytes) -> Pixels:
    if len(data) != descriptor["byteSize"] or digest_bytes(data) != descriptor["digest"]:
        raise InvalidDocument("resource bytes do not match their descriptor")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise InvalidDocument("animated image resources are unsupported")
            if image.size != (descriptor["width"], descriptor["height"]):
                raise InvalidDocument("decoded resource dimensions do not match their descriptor")
            expected_format = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}[
                cast(str, descriptor["mediaType"])
            ]
            if image.format != expected_format:
                raise InvalidDocument("decoded resource media type does not match its descriptor")
            image.load()
            rgba = np.asarray(image.convert("RGBA"), dtype=np.int64) * 257
            if descriptor["alphaMode"] == "opaque":
                rgba[:, :, 3] = ONE
            if descriptor["alphaMode"] != "premultiplied":
                rgba[:, :, :3] = _divide(rgba[:, :, :3] * rgba[:, :, 3:4])
            else:
                rgba[:, :, :3] = np.minimum(rgba[:, :, :3], rgba[:, :, 3:4])
            return rgba.astype(np.uint16)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise InvalidDocument("resource is not a supported raster image") from error


def _sample(
    source: Pixels,
    rect: dict[str, Any],
    transform: dict[str, Any],
    width: int,
    height: int,
) -> Pixels:
    a, b, c, d = (int(transform[key]) for key in ("a", "b", "c", "d"))
    tx, ty = int(transform["tx"]), int(transform["ty"])
    determinant = a * d - b * c
    output = np.zeros((height, width, 4), dtype=np.uint16)
    if determinant == 0:
        return output
    coordinate_bound = max(
        (width * 2 + 1) * FIXED_POINT_SCALE + 2 * abs(tx),
        (height * 2 + 1) * FIXED_POINT_SCALE + 2 * abs(ty),
    )
    if (
        abs(determinant) > MAX_INT64 // 2
        or max(abs(a) + abs(b), abs(c) + abs(d)) * coordinate_bound > MAX_INT64
    ):
        raise InvalidDocument("transform exceeds the reference renderer integer range")
    destination_y = np.arange(height, dtype=np.int64)[:, None]
    destination_x = np.arange(width, dtype=np.int64)[None, :]
    right_x = (destination_x * 2 + 1) * FIXED_POINT_SCALE - 2 * tx
    right_y = (destination_y * 2 + 1) * FIXED_POINT_SCALE - 2 * ty
    local_x = np.floor_divide(d * right_x - c * right_y, determinant * 2)
    local_y = np.floor_divide(a * right_y - b * right_x, determinant * 2)
    rect_width, rect_height = int(rect["width"]), int(rect["height"])
    inside = (local_x >= 0) & (local_x < rect_width) & (local_y >= 0) & (local_y < rect_height)
    if not np.any(inside):
        return output
    output[inside] = source[local_y[inside] + int(rect["y"]), local_x[inside] + int(rect["x"])]
    return output


def _blend_function(
    source: NDArray[np.int64], backdrop: NDArray[np.int64], mode: str
) -> NDArray[np.int64]:
    if mode == "normal":
        return source
    if mode == "multiply":
        return _multiply(source, backdrop)
    if mode == "screen":
        return source + backdrop - _multiply(source, backdrop)
    if mode == "darken":
        return np.minimum(source, backdrop)
    if mode == "lighten":
        return np.maximum(source, backdrop)
    if mode != "overlay":
        return np.rint(blend(backdrop / ONE, source / ONE, mode, seed=0) * ONE).astype(np.int64)
    low = _divide(2 * source * backdrop)
    high = ONE - _divide(2 * (ONE - source) * (ONE - backdrop))
    return np.where(backdrop <= ONE // 2, low, high)


def _composite(
    backdrop: Pixels, source: Pixels, mode: str, opacity: int, *, linear: bool = False
) -> Pixels:
    destination = backdrop.astype(np.int64)
    incoming = source.astype(np.int64)
    if opacity != OPACITY_MAX:
        incoming = _divide(incoming * opacity, OPACITY_MAX)
    if mode == "dissolve":
        alpha = incoming[:, :, 3:4]
        coverage = dissolve_alpha(alpha / ONE, seed=0).astype(np.int64)
        incoming[:, :, :3] = _divide(incoming[:, :, :3] * ONE, np.maximum(alpha, 1)) * coverage
        incoming[:, :, 3:4] = coverage * ONE
    source_alpha = incoming[:, :, 3:4]
    backdrop_alpha = destination[:, :, 3:4]
    source_straight = np.where(
        source_alpha == 0,
        0,
        _divide(incoming[:, :, :3] * ONE, np.maximum(source_alpha, 1)),
    )
    backdrop_straight = np.where(
        backdrop_alpha == 0,
        0,
        _divide(destination[:, :, :3] * ONE, np.maximum(backdrop_alpha, 1)),
    )
    if linear:
        top = srgb_to_linear(source_straight / ONE)
        bottom = srgb_to_linear(backdrop_straight / ONE)
        sa, ba = source_alpha / ONE, backdrop_alpha / ONE
        alpha = sa + ba * (1 - sa)
        color = (
            (1 - sa) * ba * bottom
            + (1 - ba) * sa * top
            + sa * ba * blend(bottom, top, mode, seed=0)
        )
        straight = np.divide(color, alpha, out=np.zeros_like(color), where=alpha > 0)
        return np.rint(
            np.concatenate((linear_to_srgb(straight) * alpha, alpha), axis=-1) * ONE
        ).astype(np.uint16)
    blended = _blend_function(source_straight, backdrop_straight, mode)
    overlap = _divide(source_alpha * backdrop_alpha)
    color = (
        _divide(incoming[:, :, :3] * (ONE - backdrop_alpha))
        + _divide(destination[:, :, :3] * (ONE - source_alpha))
        + _divide(blended * overlap)
    )
    alpha = source_alpha + backdrop_alpha - overlap
    return np.concatenate((np.clip(color, 0, alpha), np.clip(alpha, 0, ONE)), axis=2).astype(
        np.uint16
    )


def _transform_surface(source: Pixels, transform: dict[str, Any]) -> Pixels:
    height, width = source.shape[:2]
    rect: dict[str, Any] = {"x": 0, "y": 0, "width": width, "height": height}
    return _sample(source, rect, transform, width, height)


def _mask_value(mask: dict[str, Any], sampled: Pixels) -> NDArray[np.int64]:
    alpha = sampled[:, :, 3].astype(np.int64)
    if mask["channel"] == "alpha":
        channel = alpha
    else:
        straight = np.zeros((*alpha.shape, 3), dtype=np.int64)
        for index in range(3):
            straight[:, :, index] = np.where(
                alpha == 0,
                0,
                _divide(sampled[:, :, index].astype(np.int64) * ONE, np.maximum(alpha, 1)),
            )
        channel = _divide(
            straight[:, :, 0] * 13_933 + straight[:, :, 1] * 46_871 + straight[:, :, 2] * 4_731
        )
    if mask["invert"]:
        channel = ONE - channel
    opacity = int(mask["opacity"])
    return ONE - opacity + _divide(channel * opacity)


class _Renderer:
    def __init__(self, parsed: ParsedDocument, reader: ResourceReader) -> None:
        self.document = parsed.value
        canvas = cast(dict[str, Any], self.document["canvas"])
        self.width, self.height = int(canvas["width"]), int(canvas["height"])
        self.layers = cast(dict[str, dict[str, Any]], self.document["layers"])
        self.masks = cast(dict[str, dict[str, Any]], self.document["masks"])
        self.resources = cast(dict[str, dict[str, Any]], self.document["resources"])
        self.reader = reader
        self.decoded: dict[str, Pixels] = {}

    def resource(self, resource_id: str) -> Pixels:
        if resource_id not in self.decoded:
            descriptor = self.resources[resource_id]
            digest = cast(str, descriptor["digest"])
            data = (
                base64.b64decode(descriptor["inline"])
                if "inline" in descriptor
                else self.reader(digest)
            )
            self.decoded[resource_id] = _decode_resource(descriptor, data)
        return self.decoded[resource_id]

    def mask_surface(self, mask: dict[str, Any]) -> Pixels:
        return _sample(
            self.resource(cast(str, mask["resourceId"])),
            cast(dict[str, Any], mask["sourceRect"]),
            cast(dict[str, Any], mask["transform"]),
            self.width,
            self.height,
        )

    def apply_masks(self, layer: dict[str, Any], surface: Pixels) -> Pixels:
        combined: NDArray[np.int64] | None = None
        for mask_id in cast(list[str], layer["maskIds"]):
            mask = self.masks[mask_id]
            if not mask["enabled"]:
                continue
            value = _mask_value(mask, self.mask_surface(mask))
            if combined is None:
                combined = value
            elif mask["combineMode"] == "multiply":
                combined = _divide(combined * value)
            elif mask["combineMode"] == "add":
                combined = np.minimum(ONE, combined + value)
            elif mask["combineMode"] == "subtract":
                combined = np.maximum(0, combined - value)
            else:
                combined = np.minimum(combined, value)
        if combined is None:
            return surface
        return _divide(surface.astype(np.int64) * combined[:, :, None]).astype(np.uint16)

    def layer(self, layer_id: str) -> Pixels:
        layer = self.layers[layer_id]
        if not layer["visible"] or layer["opacity"] == 0:
            return np.zeros((self.height, self.width, 4), dtype=np.uint16)
        if layer["kind"] == "raster":
            surface = _sample(
                self.resource(cast(str, layer["resourceId"])),
                cast(dict[str, Any], layer["sourceRect"]),
                cast(dict[str, Any], layer["transform"]),
                self.width,
                self.height,
            )
        else:
            surface, _ = self.siblings(cast(list[str], layer["childLayerIds"]))
        surface = self.apply_masks(layer, surface)
        if layer["kind"] == "group":
            surface = _transform_surface(surface, cast(dict[str, Any], layer["transform"]))
        return surface

    def siblings(
        self,
        layer_ids: list[str],
        background: list[int] | None = None,
        *,
        backdrop: Pixels | None = None,
        transforms: tuple[dict[str, Any], ...] = (),
    ) -> tuple[Pixels, NDArray[np.uint16]]:
        output = (
            np.zeros((self.height, self.width, 4), dtype=np.uint16)
            if backdrop is None
            else backdrop.copy()
        )
        if background is not None:
            color = np.asarray(background, dtype=np.int64)
            color[:3] = _divide(color[:3] * color[3])
            output[:] = color
        coverage = np.zeros((self.height, self.width), dtype=np.int64)
        previous_alpha: NDArray[np.uint16] | None = None
        for layer_id in ordered_layer_ids(self.layers, layer_ids):
            layer = self.layers[layer_id]
            if layer.get("isolation") == "pass-through" and layer["visible"] and layer["opacity"]:
                group_transforms = (cast(dict[str, Any], layer["transform"]), *transforms)
                composed, child_coverage = self.siblings(
                    layer["childLayerIds"], backdrop=output, transforms=group_transforms
                )
                weight = self.apply_masks(layer, np.full_like(output, ONE))
                for transform in group_transforms:
                    weight = _transform_surface(weight, transform)
                amount = _divide(weight[:, :, 3:4].astype(np.int64) * int(layer["opacity"]))
                if layer["clipping"] == "clip-to-previous":
                    if previous_alpha is None:
                        raise InvalidDocument("clipped layer has no previous sibling")
                    amount = _divide(amount * previous_alpha[:, :, None].astype(np.int64))
                if self.document["canvas"]["compositing"] == "linear-premultiplied-alpha":
                    backdrop_alpha = output[:, :, 3:4] / ONE
                    composed_alpha = composed[:, :, 3:4] / ONE
                    bottom = srgb_to_linear(output[:, :, :3] / np.maximum(output[:, :, 3:4], 1))
                    top = srgb_to_linear(composed[:, :, :3] / np.maximum(composed[:, :, 3:4], 1))
                    fraction = amount / ONE
                    alpha = backdrop_alpha * (1 - fraction) + composed_alpha * fraction
                    color = (
                        bottom * backdrop_alpha * (1 - fraction) + top * composed_alpha * fraction
                    )
                    straight = np.divide(color, alpha, out=np.zeros_like(color), where=alpha > 0)
                    output = np.rint(
                        np.concatenate((linear_to_srgb(straight) * alpha, alpha), axis=-1) * ONE
                    ).astype(np.uint16)
                else:
                    output = _divide(
                        output.astype(np.int64) * (ONE - amount)
                        + composed.astype(np.int64) * amount
                    ).astype(np.uint16)
                previous_alpha = _divide(child_coverage.astype(np.int64) * amount[:, :, 0]).astype(
                    np.uint16
                )
            else:
                surface = self.layer(layer_id)
                for transform in transforms:
                    surface = _transform_surface(surface, transform)
                if layer["clipping"] == "clip-to-previous":
                    if previous_alpha is None:
                        raise InvalidDocument("clipped layer has no previous sibling")
                    surface = _divide(
                        surface.astype(np.int64) * previous_alpha[:, :, None].astype(np.int64)
                    ).astype(np.uint16)
                output = _composite(
                    output,
                    surface,
                    cast(str, layer["blendMode"]),
                    int(layer["opacity"]),
                    linear=self.document["canvas"]["compositing"] == "linear-premultiplied-alpha",
                )
                previous_alpha = _divide(
                    surface[:, :, 3].astype(np.int64) * int(layer["opacity"]), OPACITY_MAX
                ).astype(np.uint16)
            alpha = previous_alpha.astype(np.int64)
            coverage = alpha + _divide(coverage * (ONE - alpha))
        return output, coverage.astype(np.uint16)

    def render(self, selector: RenderSelector) -> Pixels:
        if selector.kind == "composite":
            return self.siblings(
                cast(list[str], self.document["rootLayerIds"]),
                self.document["canvas"].get("background"),
            )[0]
        if selector.kind == "layer":
            if selector.id not in self.layers:
                raise InvalidDocument("selected layer does not exist")
            layer = self.layers[selector.id]
            return _composite(
                np.zeros((self.height, self.width, 4), dtype=np.uint16),
                self.layer(selector.id),
                "normal",
                int(layer["opacity"]),
            )
        if selector.id not in self.masks:
            raise InvalidDocument("selected mask does not exist")
        mask = self.masks[selector.id]
        value = _mask_value(mask, self.mask_surface(mask)).astype(np.uint16)
        result = np.empty((self.height, self.width, 4), dtype=np.uint16)
        result[:, :, :3] = value[:, :, None]
        result[:, :, 3] = ONE
        return result


def encode_png(pixels: Pixels) -> bytes:
    height, width = pixels.shape[:2]
    alpha = pixels[:, :, 3].astype(np.int64)
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[:, :, 3] = _divide(alpha, 257).astype(np.uint8)
    for channel in range(3):
        straight = np.where(
            alpha == 0,
            0,
            _divide(pixels[:, :, channel].astype(np.int64) * ONE, np.maximum(alpha, 1)),
        )
        rgba[:, :, channel] = _divide(straight, 257).astype(np.uint8)
    return encode_canonical_png(rgba.tobytes(), width=width, height=height, color_type=6)


def render_document(
    parsed: ParsedDocument,
    reader: ResourceReader,
    selector: RenderSelector | None = None,
) -> RenderResult:
    canvas = cast(dict[str, Any], parsed.value["canvas"])
    width, height = int(canvas["width"]), int(canvas["height"])
    if width * height > MAX_RENDER_PIXELS:
        raise InvalidDocument("canvas exceeds the reference renderer pixel limit")
    layers = cast(dict[str, dict[str, Any]], parsed.value["layers"])
    masks = cast(dict[str, dict[str, Any]], parsed.value["masks"])
    work_surfaces = len(layers) + sum(layer["kind"] == "group" for layer in layers.values())
    work_surfaces += sum(bool(mask["enabled"]) for mask in masks.values())
    if width * height * max(1, work_surfaces) > MAX_RENDER_WORK_PIXELS:
        raise InvalidDocument("document exceeds the reference renderer work limit")
    resources = cast(dict[str, dict[str, Any]], parsed.value["resources"])
    resource_memory = sum(
        int(resource["byteSize"]) + int(resource["width"]) * int(resource["height"]) * 8
        for resource in resources.values()
    )
    if resource_memory > MAX_RESOURCE_MEMORY:
        raise InvalidDocument("resources exceed the reference renderer memory limit")
    pixels = _Renderer(parsed, reader).render(selector or RenderSelector("composite"))
    return RenderResult(pixels, encode_png(pixels), width, height)
