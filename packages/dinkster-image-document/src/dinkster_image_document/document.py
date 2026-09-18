"""Portable document values, asset publication, and structural edit commands."""

from __future__ import annotations

import base64
import copy
import io
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from dinkster_assets import AssetRef, AssetVault, digest_bytes, resolver_from_env
from dinkster_values import annotate_image, annotate_mask, media_semantics, render_image_png
from dinkster_values.image_codec import image_color
from PIL import Image

from .format import (
    INLINE_RESOURCE_LIMIT,
    InvalidDocument,
    affine_from_components,
    canonical_json,
    decode_document,
    ordered_layer_ids,
    validate_document,
)
from .render import parse_selector, render_document

IDENTITY = {"a": 1_000_000, "b": 0, "c": 0, "d": 1_000_000, "tx": 0, "ty": 0}
AssetBinder = Callable[[dict[str, object]], AssetRef]


@dataclass(frozen=True)
class ImageDocument:
    data: bytes
    bind: AssetBinder | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        parsed = decode_document(self.data)
        object.__setattr__(self, "data", canonical_json(parsed.value).encode("utf-8"))

    @classmethod
    def from_record(cls, value: object, bind: AssetBinder | None = None) -> ImageDocument:
        validate_document(value)
        return cls(canonical_json(value).encode("utf-8"), bind)

    def to_record(self) -> dict[str, Any]:
        return decode_document(self.data).value

    def read_resource(self, descriptor: dict[str, Any]) -> bytes:
        if "inline" in descriptor:
            return base64.b64decode(descriptor["inline"], validate=True)
        wire: dict[str, object] = {
            "digest": descriptor["digest"],
            "size": descriptor["byteSize"],
            "mediaType": descriptor["mediaType"],
            "name": descriptor["id"],
        }
        root = os.environ.get("DINKSTER_ASSET_VAULT")
        if root and AssetVault(root).resolve(descriptor["digest"]) is not None:
            return AssetRef.from_wire(wire, AssetVault(root)).read_bytes()
        if self.bind is not None:
            return self.bind(wire).read_bytes()
        return AssetRef.from_wire(wire, resolver_from_env()).read_bytes()

    def render(self, selector: str = "composite"):
        parsed = decode_document(self.data)
        resources = {item["digest"]: item for item in parsed.value["resources"].values()}
        return render_document(
            parsed, lambda digest: self.read_resource(resources[digest]), parse_selector(selector)
        )


def empty_document(
    width: int, height: int, *, color: Mapping[str, object] | None = None
) -> ImageDocument:
    return ImageDocument.from_record(
        {
            "format": "dinkster-image",
            "formatVersion": 2,
            "lineage": "node-document",
            "canvas": {
                "width": width,
                "height": height,
                "colorSpace": "srgb",
                "channelDepth": 8,
                "compositing": "premultiplied-alpha",
                "color": image_color(color),
            },
            "allocation": {"nextOrdinal": 0},
            "rootLayerIds": [],
            "layers": {},
            "masks": {},
            "resources": {},
        }
    )


def allocate(record: dict[str, Any], prefix: str) -> str:
    ordinal = record["allocation"]["nextOrdinal"]
    record["allocation"]["nextOrdinal"] += 1
    return f"{prefix}{ordinal}"


def admit_raster(record: dict[str, Any], data: bytes) -> str:
    with Image.open(io.BytesIO(data)) as image:
        media_type = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}[
            cast(str, image.format)
        ]
        width, height = image.size
        alpha = "straight" if "A" in image.getbands() or "transparency" in image.info else "opaque"
    digest = digest_bytes(data)
    for identifier, descriptor in record["resources"].items():
        if descriptor["digest"] == digest:
            return identifier
    identifier = allocate(record, "r")
    descriptor = {
        "id": identifier,
        "kind": "raster",
        "digest": digest,
        "byteSize": len(data),
        "mediaType": media_type,
        "width": width,
        "height": height,
        "colorSpace": "srgb",
        "channelDepth": 8,
        "alphaMode": alpha,
    }
    if len(data) <= INLINE_RESOURCE_LIMIT:
        descriptor["inline"] = base64.b64encode(data).decode("ascii")
    else:
        root = os.environ.get("DINKSTER_ASSET_VAULT")
        if not root:
            raise InvalidDocument("asset-backed raster publication requires DINKSTER_ASSET_VAULT")
        with AssetVault(root).writer(digest) as writer:
            writer.write(data)
            writer.commit()
    record["resources"][identifier] = descriptor
    return identifier


def raster_png(image: np.ndarray) -> bytes:
    if media_semantics(image).get("alpha") == "premultiplied":
        return render_image_png(image)
    pixels = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    if pixels.shape[-1] == 1:
        pixels = pixels[..., 0]
    elif pixels.shape[-1] == 4 and np.all(pixels[..., 3] == 255):
        pixels = pixels[..., :3]
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, "PNG")
    return output.getvalue()


def append_raster(
    document: ImageDocument,
    data: bytes,
    *,
    mask_data: bytes | None = None,
    name: str = "Layer",
    x: float = 0,
    y: float = 0,
    width: float = 0,
    height: float = 0,
    rotation: float = 0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    opacity: float = 1,
    blend_mode: str = "normal",
    visible: bool = True,
    z_index: int | None = None,
) -> ImageDocument:
    record = document.to_record()
    resource_id = admit_raster(record, data)
    resource = record["resources"][resource_id]
    components = {
        "x": x,
        "y": y,
        "width": width or resource["width"],
        "height": height or resource["height"],
        "rotation": rotation,
        "flipHorizontal": flip_horizontal,
        "flipVertical": flip_vertical,
        "sourceWidth": resource["width"],
        "sourceHeight": resource["height"],
    }
    layer_id = allocate(record, "l")
    layer: dict[str, Any] = {
        "id": layer_id,
        "kind": "raster",
        "name": name,
        "visible": visible,
        "opacity": round(opacity * 65535),
        "blendMode": blend_mode.replace("-", "_"),
        "clipping": "none",
        "maskIds": [],
        "resourceId": resource_id,
        "sourceRect": {"x": 0, "y": 0, "width": resource["width"], "height": resource["height"]},
        "transform": {**affine_from_components(components), "components": components},
    }
    if z_index is not None:
        layer["z_index"] = z_index
    record["layers"][layer_id] = layer
    record["rootLayerIds"].append(layer_id)
    if mask_data is not None:
        mask_resource = admit_raster(record, mask_data)
        if (
            record["resources"][mask_resource]["width"],
            record["resources"][mask_resource]["height"],
        ) != (resource["width"], resource["height"]):
            raise InvalidDocument("mask dimensions must match the source raster")
        mask_id = allocate(record, "m")
        record["masks"][mask_id] = {
            "id": mask_id,
            "kind": "raster",
            "ownerLayerId": layer_id,
            "resourceId": mask_resource,
            "sourceRect": dict(layer["sourceRect"]),
            "transform": copy.deepcopy(layer["transform"]),
            "enabled": True,
            "invert": True,
            "opacity": 65535,
            "combineMode": "multiply",
            "channel": "luminance",
        }
        layer["maskIds"].append(mask_id)
    return ImageDocument.from_record(record, document.bind)


def document_meta(document: ImageDocument) -> Mapping[str, object]:
    record = document.to_record()
    return {
        "format": "dinkster-image",
        "formatVersion": 2,
        "layers": len(record["layers"]),
        "canvas": record["canvas"],
        "asset_refs": [
            {
                "digest": resource["digest"],
                "size": resource["byteSize"],
                "mediaType": resource["mediaType"],
                "name": identifier,
            }
            for identifier, resource in record["resources"].items()
            if "inline" not in resource
        ],
        "cost": {"ram": len(document.data)},
    }


def flatten(document: ImageDocument, selector: str = "composite") -> tuple[np.ndarray, np.ndarray]:
    rendered = document.render(selector)
    # Use the canonical rendition's quantization at both node and Render API boundaries.
    with Image.open(io.BytesIO(rendered.png)) as image:
        rgba = np.asarray(image, dtype=np.float32)[None, ...] / 255
    transparency = np.ascontiguousarray(1 - rgba[..., 3])
    result = rgba if np.any(transparency) else rgba[..., :3]
    return (
        annotate_image(
            np.ascontiguousarray(result), color=document.to_record()["canvas"].get("color")
        ),
        annotate_mask(transparency, polarity="transparency", semantic="alpha"),
    )


def apply_commands(document: ImageDocument, commands: Sequence[Mapping[str, Any]]) -> ImageDocument:
    record = document.to_record()

    def siblings(identifier: str) -> list[str]:
        for ids in [
            record["rootLayerIds"],
            *(
                layer["childLayerIds"]
                for layer in record["layers"].values()
                if layer["kind"] == "group"
            ),
        ]:
            if identifier in ids:
                return ids
        raise InvalidDocument("layer does not exist")

    def remove(identifier: str) -> None:
        layer = record["layers"].pop(identifier)
        for child in layer.get("childLayerIds", []):
            remove(child)
        for mask in layer["maskIds"]:
            del record["masks"][mask]

    for raw in commands:
        command = copy.deepcopy(dict(raw))
        operation = command.pop("op")
        if operation == "canvas":
            record["canvas"].update(command["changes"])
        elif operation == "remove":
            identifier = command["id"]
            siblings(identifier).remove(identifier)
            remove(identifier)
        elif operation == "reorder":
            parent = command.get("parent")
            ids = (
                record["rootLayerIds"]
                if parent is None
                else record["layers"][parent]["childLayerIds"]
            )
            if set(ids) != set(command["ids"]) or len(ids) != len(command["ids"]):
                raise InvalidDocument("reorder must be a sibling permutation")
            ids[:] = command["ids"]
            for index, identifier in enumerate(ids):
                record["layers"][identifier]["z_index"] = index
        elif operation == "group":
            ids = command["ids"]
            if not ids or len(set(ids)) != len(ids):
                raise InvalidDocument("group needs distinct siblings")
            parent = siblings(ids[0])
            if any(identifier not in parent for identifier in ids):
                raise InvalidDocument("group members must be siblings")
            parent[:] = ordered_layer_ids(record["layers"], parent)
            position = min(parent.index(identifier) for identifier in ids)
            parent[:] = [identifier for identifier in parent if identifier not in ids]
            identifier = allocate(record, "l")
            record["layers"][identifier] = {
                "id": identifier,
                "kind": "group",
                "name": command.get("name", "Group"),
                "visible": True,
                "opacity": 65535,
                "transform": dict(IDENTITY),
                "blendMode": "normal",
                "clipping": "none",
                "maskIds": [],
                "childLayerIds": ids,
            }
            parent.insert(position, identifier)
            for index, sibling in enumerate(parent):
                record["layers"][sibling]["z_index"] = index
        elif operation in {"layer", "mask"}:
            collection = record["layers" if operation == "layer" else "masks"]
            identifier = command["id"]
            changes = command["changes"]
            forbidden = {"id", "kind", "childLayerIds", "maskIds", "ownerLayerId"}
            if forbidden.intersection(changes):
                raise InvalidDocument("use structural commands to change ownership")
            collection[identifier].update(changes)
        elif operation == "transform":
            target = record["masks"] if command.get("kind") == "mask" else record["layers"]
            components = command["components"]
            target[command["id"]]["transform"] = {
                **affine_from_components(components),
                "components": components,
            }
        elif operation == "add_mask":
            owner = command["ownerLayerId"]
            identifier = allocate(record, "m")
            mask = {
                "id": identifier,
                "kind": "raster",
                "ownerLayerId": owner,
                "enabled": True,
                "invert": False,
                "opacity": 65535,
                "transform": dict(IDENTITY),
                "combineMode": "multiply",
                "channel": "alpha",
                **command["mask"],
            }
            record["masks"][identifier] = mask
            record["layers"][owner]["maskIds"].append(identifier)
        elif operation == "remove_mask":
            mask = record["masks"].pop(command["id"])
            record["layers"][mask["ownerLayerId"]]["maskIds"].remove(command["id"])
        else:
            raise InvalidDocument(f"unknown document command: {operation}")
    return ImageDocument.from_record(record, document.bind)
