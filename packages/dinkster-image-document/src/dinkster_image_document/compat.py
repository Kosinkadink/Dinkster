"""Comfy LAYERS boundaries use the same portable document bytes."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
from dinkster_assets import AssetRef, resolver_from_env
from dinkster_values import TypeRegistry, copy_media_semantics, media_semantics

from .document import ImageDocument, append_raster, document_meta, empty_document, raster_png
from .format import IMAGE_DOCUMENT_MEDIA_TYPE, canonical_json


def from_comfy_layers(value: Mapping[str, Any]) -> ImageDocument:
    if value.get("format") == "dinkster-image":
        return ImageDocument.from_record(dict(value))
    if value.get("version") != 1:
        raise ValueError("unsupported Comfy LAYERS version")
    layers = sorted(
        cast(list[dict[str, Any]], value["layers"]), key=lambda layer: layer.get("z_index", 0)
    )
    layers = [dict(layer) for layer in layers]
    for layer in layers:
        for key in ("image", "mask"):
            pixels = layer.get(key)
            if pixels is not None:
                source = pixels
                if hasattr(pixels, "detach"):
                    pixels = pixels.detach().cpu().numpy()
                layer[key] = copy_media_semantics(source, np.asarray(pixels, dtype=np.float32))
    canvas = value.get("canvas")
    width = (
        canvas[0]
        if canvas
        else max(
            1,
            math.ceil(
                max(
                    (
                        item.get("x", 0) + (item.get("w", 0) or item["image"].shape[2])
                        for item in layers
                    ),
                    default=1,
                )
            ),
        )
    )
    height = (
        canvas[1]
        if canvas
        else max(
            1,
            math.ceil(
                max(
                    (
                        item.get("y", 0) + (item.get("h", 0) or item["image"].shape[1])
                        for item in layers
                    ),
                    default=1,
                )
            ),
        )
    )
    result = empty_document(
        width,
        height,
        color=cast(Mapping[str, object] | None, media_semantics(layers[0]["image"]).get("color"))
        if layers
        else None,
    )
    fingerprints: list[str] = []
    for index, layer in enumerate(layers):
        image = layer["image"]
        mask = layer.get("mask")
        for frame_index, frame in enumerate(image):
            alpha = frame[..., 3] if frame.shape[-1] == 4 else None
            if mask is not None:
                coverage = 1 - np.asarray(mask)[0 if mask.shape[0] == 1 else frame_index]
                alpha = coverage if alpha is None else alpha * coverage
            fingerprint = hashlib.sha256()
            fingerprint.update(repr((1, *frame.shape)).encode())
            fingerprint.update(
                np.clip(np.rint(frame[..., :3] * 255), 0, 255).astype(np.uint8).tobytes()
            )
            if alpha is not None:
                fingerprint.update(np.clip(np.rint(alpha * 255), 0, 255).astype(np.uint8).tobytes())
            fingerprint.update(
                repr(
                    (
                        int(layer.get("x", 0)),
                        int(layer.get("y", 0)),
                        int(layer.get("w", 0)) or frame.shape[1],
                        int(layer.get("h", 0)) or frame.shape[0],
                        float(layer.get("rotation", 0)),
                        layer.get("opacity", 1.0),
                        layer.get("blend_mode", "normal"),
                        bool(layer.get("visible", True)),
                        bool(layer.get("flip_h", False)),
                        bool(layer.get("flip_v", False)),
                    )
                ).encode()
            )
            fingerprints.append(fingerprint.hexdigest()[:16])
            if frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)
            result = append_raster(
                result,
                raster_png(copy_media_semantics(image, frame)),
                mask_data=None
                if mask is None
                else raster_png(
                    np.asarray(mask)[0 if mask.shape[0] == 1 else frame_index][..., None]
                ),
                name=layer.get("name", f"Layer {index + 1}"),
                x=layer.get("x", 0),
                y=layer.get("y", 0),
                width=layer.get("w", 0),
                height=layer.get("h", 0),
                rotation=layer.get("rotation", 0),
                opacity=layer.get("opacity", 1),
                blend_mode=layer.get("blend_mode", "normal"),
                visible=layer.get("visible", True),
                flip_horizontal=layer.get("flip_h", False),
                flip_vertical=layer.get("flip_v", False),
                z_index=layer.get("z_index", index),
            )
    record = result.to_record()
    record["canvas"]["compositing"] = "linear-premultiplied-alpha"
    record["extensions"] = {"comfy": {"inputs": fingerprints}}
    return ImageDocument.from_record(record, result.bind)


def register_comfy_layers(registry: TypeRegistry) -> None:
    def coerce(obj: object) -> ImageDocument:
        if isinstance(obj, ImageDocument):
            return obj
        if not isinstance(obj, Mapping):
            raise TypeError("Comfy LAYERS must be a document")
        return from_comfy_layers(cast(Mapping[str, Any], obj))

    def decode(data: bytes) -> object:
        return ImageDocument(data, lambda wire: AssetRef.from_wire(wire, resolver_from_env()))

    if "comfy.LAYERS" not in registry:
        registry.register(
            "comfy.LAYERS",
            coerce=coerce,
            encode=lambda obj: coerce(obj).data,
            decode=decode,
            meta=lambda obj: document_meta(coerce(obj)),
        )
        registry.register_rendition(
            "comfy.LAYERS",
            "document",
            mime=IMAGE_DOCUMENT_MEDIA_TYPE,
            render=lambda obj: coerce(obj).data,
        )
    if "dinkster.layers" in registry:
        registry.register_type_equivalence(
            "comfy.LAYERS", "dinkster.layers", provider_id="dinkster.layer-document-equivalence@2"
        )


def register_comfy_compositor(registry: TypeRegistry) -> None:
    def coerce(obj: object) -> dict[str, Any]:
        if not isinstance(obj, Mapping):
            raise TypeError("COMPOSITOR must be a JSON object")
        value = cast(dict[str, Any], obj)
        if set(value) - {
            "version",
            "canvas",
            "background",
            "inputs",
            "order",
            "layers",
            "documentDigest",
            "commands",
        }:
            raise ValueError("unknown COMPOSITOR fields")
        return json.loads(canonical_json(value))

    if "comfy.COMPOSITOR" not in registry:
        registry.register(
            "comfy.COMPOSITOR",
            coerce=coerce,
            encode=lambda obj: canonical_json(coerce(obj)).encode("utf-8"),
            decode=lambda data: coerce(json.loads(data)),
        )
    if "dinkster.compositor" in registry:
        registry.register_type_equivalence(
            "comfy.COMPOSITOR",
            "dinkster.compositor",
            provider_id="dinkster.compositor-equivalence@2",
        )
