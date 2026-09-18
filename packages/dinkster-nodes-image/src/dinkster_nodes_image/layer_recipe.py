"""Compositor delta codec and legacy widget migration."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

from dinkster_api.v1 import digest_bytes, is_digest
from dinkster_image_document.document import ImageDocument, apply_commands
from dinkster_image_document.format import canonical_json

from .compositor_types import (
    CompositorRecipe,
    LayerStack,
    coerce_compositor_recipe,
    source_layer_fingerprint,
)


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


def prepare_compositor(source: object, obj: object) -> tuple[ImageDocument, str, bool]:
    from .layer_document import migrate_layers

    native = source
    if obj is not None and isinstance(source, LayerStack):
        value = coerce_delta(obj)
        if value["version"] == 1 and "width" in value.get("canvas", {}):
            legacy = coerce_compositor_recipe(value)
            if legacy.input_fingerprints == tuple(
                source_layer_fingerprint(layer) for layer in source.expanded()
            ):
                native = replace(
                    source, canvas_width=legacy.canvas_width, canvas_height=legacy.canvas_height
                )
    input_document = migrate_layers(native)
    document, stale = replay_compositor(input_document, obj, source)
    return document, digest_bytes(input_document.data), stale


def replay_compositor(
    document: ImageDocument, obj: object, source: object
) -> tuple[ImageDocument, bool]:
    if obj is None:
        return document, False
    delta = coerce_delta(obj)
    if delta.get("version") == 2:
        if delta["documentDigest"] not in (None, digest_bytes(document.data)):
            return document, True
        return apply_commands(document, delta["commands"]), False
    if "w" in delta.get("canvas", {}):
        record = document.to_record()
        ids = record["rootLayerIds"]
        expected = record.get("extensions", {}).get("comfy", {}).get("inputs")
        if delta.get("inputs") != expected or len(delta.get("layers", [])) != len(ids):
            return document, True
        commands: list[dict[str, Any]] = []
        for identifier, edit in zip(ids, delta["layers"], strict=True):
            base = record["layers"][identifier]["transform"]["components"]
            transform = edit.get("transform", {})
            components = {
                **base,
                **{key: transform.get(key, base[key]) for key in ("x", "y", "rotation")},
                "width": transform.get("w", base["width"]),
                "height": transform.get("h", base["height"]),
                "flipHorizontal": edit.get("flipH", False),
                "flipVertical": edit.get("flipV", False),
            }
            commands.extend(
                [
                    *_placement_commands(record, identifier, components),
                    {
                        "op": "layer",
                        "id": identifier,
                        "changes": {
                            "visible": edit.get("visible", True),
                            "opacity": round(edit.get("opacity", 1) * 65535),
                            "blendMode": edit.get("blend", "normal").replace("-", "_"),
                            "name": edit.get("name", record["layers"][identifier]["name"]),
                        },
                    },
                ]
            )
        order = delta.get("order", list(range(len(ids))))
        if sorted(order) != list(range(len(ids))):
            order = list(range(len(ids)))
        commands.append({"op": "reorder", "ids": [ids[index] for index in order]})
        record["canvas"].update(width=delta["canvas"]["w"], height=delta["canvas"]["h"])
        background = delta.get("background", {})
        record["canvas"]["background"] = _background(
            background.get("color", "#000000"),
            background.get("opacity", 1) if background.get("visible", False) else 0,
        )
        return apply_commands(ImageDocument.from_record(record, document.bind), commands), False
    legacy: CompositorRecipe = coerce_compositor_recipe(delta)
    record = document.to_record()
    expected = (
        tuple(source_layer_fingerprint(layer) for layer in source.expanded())
        if isinstance(source, LayerStack)
        else tuple(record.get("extensions", {}).get("legacyLayerInputs", []))
    )
    if legacy.input_fingerprints != expected:
        return document, True
    identifiers = record["rootLayerIds"]
    if len(identifiers) != len(legacy.layers):
        return document, True
    commands: list[dict[str, Any]] = []
    for edit in legacy.layers:
        identifier = identifiers[edit.source_index]
        rectangle = record["layers"][identifier]["sourceRect"]
        components = {
            **edit.transform.to_record(),
            "sourceWidth": rectangle["width"],
            "sourceHeight": rectangle["height"],
            "flipHorizontal": edit.flip_horizontal,
            "flipVertical": edit.flip_vertical,
        }
        commands.extend(
            [
                *_placement_commands(record, identifier, components),
                {
                    "op": "layer",
                    "id": identifier,
                    "changes": {
                        "name": edit.name,
                        "visible": edit.visible,
                        "opacity": round(edit.opacity * 65535),
                        "blendMode": edit.blend_mode,
                    },
                },
            ]
        )
    commands.append(
        {"op": "reorder", "ids": [identifiers[edit.source_index] for edit in legacy.layers]}
    )
    record["canvas"].update(width=legacy.canvas_width, height=legacy.canvas_height)
    record["canvas"]["background"] = _background(
        legacy.background_color, legacy.background_opacity if legacy.background_visible else 0
    )
    return apply_commands(ImageDocument.from_record(record, document.bind), commands), False


def _placement_commands(
    record: dict[str, Any], identifier: str, components: dict[str, Any]
) -> list[dict[str, Any]]:
    # Legacy source masks share the source layer's placement.
    return [
        {"op": "transform", "id": identifier, "components": components},
        *(
            {"op": "transform", "kind": "mask", "id": mask_id, "components": components}
            for mask_id in record["layers"][identifier]["maskIds"]
        ),
    ]


def _background(color: str, opacity: float) -> list[int]:
    return [int(color[index : index + 2], 16) * 257 for index in (1, 3, 5)] + [
        round(opacity * 65535)
    ]
