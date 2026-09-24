"""Bounded, data-only image metadata parsing."""

from __future__ import annotations
from dinkster_api.v1 import MEBIBYTE

import csv
import json
import os
import warnings
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import AssetRef

METADATA_FORMAT = "dinkster.image-metadata/1"
MAX_METADATA_FIELD_BYTES = MEBIBYTE
MAX_METADATA_TOTAL_BYTES = 2 * MEBIBYTE
MAX_METADATA_FIELDS = 128
MAX_METADATA_KEY_BYTES = 256
MAX_METADATA_ASSET_BYTES = 512 * MEBIBYTE
MAX_METADATA_IMAGE_DIMENSION = 16384


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number {value!r}")


def _text_fields(info: Mapping[str, object]) -> dict[str, str]:
    fields: dict[str, str] = {}
    total = 0
    for key, value in info.items():
        if not isinstance(value, str):
            continue
        key_size = len(key.encode("utf-8"))
        value_size = len(value.encode("utf-8"))
        if key_size > MAX_METADATA_KEY_BYTES:
            raise ValueError(f"image metadata key exceeds {MAX_METADATA_KEY_BYTES} bytes")
        if value_size > MAX_METADATA_FIELD_BYTES:
            raise ValueError(
                f"image metadata field {key!r} exceeds {MAX_METADATA_FIELD_BYTES} bytes"
            )
        total += key_size + value_size
        if total > MAX_METADATA_TOTAL_BYTES:
            raise ValueError(f"image text metadata exceeds {MAX_METADATA_TOTAL_BYTES} bytes")
        fields[key] = value
        if len(fields) > MAX_METADATA_FIELDS:
            raise ValueError(f"image text metadata exceeds {MAX_METADATA_FIELDS} fields")
    return fields


def _json_field(raw: str, key: str, errors: list[str]) -> object | None:
    try:
        value = cast(object, json.loads(raw, parse_constant=_reject_json_constant))
    except (ValueError, RecursionError):
        errors.append(f"{key}: invalid JSON")
        return None
    if not isinstance(value, dict | list):
        errors.append(f"{key}: expected a JSON object or array")
        return None
    return cast(object, value)


def _a1111_parameters(raw: str) -> dict[str, object]:
    lines = raw.splitlines()
    settings_index: int | None = None
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].lstrip().startswith("Steps:"):
            settings_index = index
            break
    text_lines = lines if settings_index is None else lines[:settings_index]
    settings_line = "" if settings_index is None else lines[settings_index]
    text = "\n".join(text_lines)
    marker = "\nNegative prompt: "
    if marker in text:
        prompt, negative_prompt = text.split(marker, 1)
    elif text.startswith("Negative prompt: "):
        prompt, negative_prompt = "", text[len("Negative prompt: ") :]
    else:
        prompt, negative_prompt = text, ""

    settings: dict[str, str] = {}
    if settings_line:
        for entry in next(csv.reader([settings_line], skipinitialspace=True)):
            name, separator, value = entry.partition(":")
            if separator and name.strip():
                settings[name.strip()] = value.strip()
    return {
        "negativePrompt": negative_prompt,
        "prompt": prompt,
        "settings": settings,
    }


def metadata_document(
    info: Mapping[str, object],
    *,
    name: str,
    digest: str,
    media_type: str,
    size: int,
) -> dict[str, object]:
    """Normalize Pillow metadata. Duplicate PNG text keys use Pillow's last value."""
    raw = _text_fields(info)
    errors: list[str] = []
    comfy: dict[str, object] = {}
    for key in ("prompt", "workflow"):
        if key in raw:
            decoded = _json_field(raw[key], key, errors)
            if decoded is not None:
                comfy[key] = decoded
    document: dict[str, object] = {
        "format": METADATA_FORMAT,
        "provenance": {
            "digest": digest,
            "mediaType": media_type,
            "name": name,
            "size": size,
        },
        "raw": raw,
    }
    if comfy:
        document["comfy"] = comfy
    if "parameters" in raw:
        document["a1111"] = _a1111_parameters(raw["parameters"])
    if errors:
        document["errors"] = errors
    return document


def parse_image_metadata(image: AssetRef) -> dict[str, object]:
    """Read normalized metadata without importing or executing embedded content."""
    from PIL import Image

    if image.size > MAX_METADATA_ASSET_BYTES:
        raise ValueError(f"metadata source exceeds the {MAX_METADATA_ASSET_BYTES}-byte input limit")
    try:
        with image.open() as handle, warnings.catch_warnings():
            actual_size = os.fstat(handle.fileno()).st_size
            if actual_size > MAX_METADATA_ASSET_BYTES:
                raise ValueError(
                    f"metadata source exceeds the {MAX_METADATA_ASSET_BYTES}-byte input limit"
                )
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(handle) as source:
                width, height = source.size
                if width < 1 or height < 1:
                    raise ValueError("metadata source dimensions must be nonzero")
                if width > MAX_METADATA_IMAGE_DIMENSION or height > MAX_METADATA_IMAGE_DIMENSION:
                    raise ValueError(
                        f"metadata source dimensions exceed {MAX_METADATA_IMAGE_DIMENSION} pixels"
                    )
                return metadata_document(
                    cast("Mapping[str, object]", source.info),
                    name=image.name,
                    digest=image.digest,
                    media_type=image.media_type,
                    size=actual_size,
                )
    except (
        OSError,
        ValueError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        raise ValueError(f"cannot read metadata from {image.name!r}: {exc}") from exc


def parse_image_metadata_json(image: AssetRef) -> str:
    return _canonical_json(parse_image_metadata(image))


def png_metadata(metadata_json: str) -> Any | None:
    """Convert an explicit JSON object to bounded PNG text chunks."""
    if not metadata_json:
        return None
    encoded = metadata_json.encode("utf-8")
    if len(encoded) > MAX_METADATA_TOTAL_BYTES:
        raise ValueError(f"metadata_json exceeds {MAX_METADATA_TOTAL_BYTES} bytes")
    try:
        value = cast(object, json.loads(metadata_json, parse_constant=_reject_json_constant))
    except (ValueError, RecursionError) as exc:
        raise ValueError(f"metadata_json is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("metadata_json must contain an object")
    values = cast("dict[str, object]", value)
    info = _text_fields(
        {
            str(key): item if isinstance(item, str) else _canonical_json(item)
            for key, item in values.items()
        }
    )
    from PIL.PngImagePlugin import PngInfo

    pnginfo = PngInfo()
    for key, item in info.items():
        pnginfo.add_text(key, item)
    return pnginfo


__all__ = [
    "MAX_METADATA_FIELD_BYTES",
    "MAX_METADATA_FIELDS",
    "MAX_METADATA_ASSET_BYTES",
    "MAX_METADATA_IMAGE_DIMENSION",
    "MAX_METADATA_TOTAL_BYTES",
    "METADATA_FORMAT",
    "metadata_document",
    "parse_image_metadata",
    "parse_image_metadata_json",
    "png_metadata",
]
