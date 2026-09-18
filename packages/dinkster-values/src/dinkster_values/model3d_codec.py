"""The dinkster.model3d codec: raw GLB (binary glTF 2.0) container bytes.

The byte contract is the GLB bytes as-is. The torch-free runtime form is
``{"format": "glb", "bytes": <bytes>}``; there is no ``kind`` field because
type identity belongs to the schema type id. The format id is repeated in
envelope metadata so clients can interrogate a value without loading its
payload. GLB is the only transport format; all other identifiers and
unrecognized byte streams fail loudly.

Format identity is derived from the 12-byte GLB header (magic, version 2,
declared total length matching the actual byte count). This is the cheap
structural check appropriate at a codec boundary; the deep chunk-level GLB
validation lives in the media upload authority (dinkster-assets), which every
uploaded asset already passed. Fingerprints hash the raw bytes under the
schema type id, making cache identity independent of the runtime wrapper
(hazard H4).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

from .model import stable_hash
from .registry import TypeRegistry

__all__ = [
    "MODEL3D_FILE_DECODER_ID",
    "MODEL3D_FORMATS",
    "decode_model3d",
    "decode_model3d_file",
    "encode_model3d",
    "model3d_fingerprint",
    "model3d_format",
    "model3d_meta",
    "register_model3d_type",
    "render_model3d_original",
    "validate_model3d_encoded",
]

MODEL3D_FORMATS = frozenset({"glb"})


def model3d_format(data: bytes | memoryview) -> str:
    """Identify the closed transport format from its bytes."""
    if (
        len(data) >= 12
        and data[:4] == b"glTF"
        and int.from_bytes(data[4:8], "little") == 2
        and int.from_bytes(data[8:12], "little") == len(data)
    ):
        return "glb"
    raise ValueError("model3d bytes are not a GLB (binary glTF 2.0) container")


def _parts(obj: object) -> tuple[str, bytes]:
    if not isinstance(obj, Mapping):
        raise TypeError("model3d value must be a mapping with format and bytes")
    value = cast("Mapping[str, object]", obj)
    try:
        format_id = value["format"]
        data = value["bytes"]
    except KeyError as exc:
        raise ValueError(f"model3d value is missing {exc.args[0]}") from exc
    if not isinstance(format_id, str) or format_id not in MODEL3D_FORMATS:
        raise ValueError("model3d format must be one of: glb")
    if not isinstance(data, bytes):
        raise TypeError("model3d bytes must be bytes")
    detected = model3d_format(data)
    if detected != format_id:
        raise ValueError(f"model3d format id {format_id!r} does not match {detected!r} bytes")
    return format_id, data


def encode_model3d(obj: object) -> bytes:
    """Return validated GLB bytes without rewriting the container."""
    return _parts(obj)[1]


def decode_model3d(data: bytes) -> object:
    """Wrap raw GLB bytes in the torch-free runtime form."""
    format_id = model3d_format(data)
    return {"format": format_id, "bytes": data}


def model3d_fingerprint(type_id: str) -> Callable[[object], str]:
    """A form-independent fingerprint over the raw container bytes."""

    def fingerprint(obj: object) -> str:
        return stable_hash([type_id.encode("utf-8"), encode_model3d(obj)])

    return fingerprint


def model3d_meta(obj: object) -> Mapping[str, object]:
    """Report the format id and raw byte size."""
    format_id, data = _parts(obj)
    return {"format": format_id, "byte_size": len(data)}


def render_model3d_original(obj: object) -> bytes:
    """Return the original bytes after revalidating their container wrapper."""
    return encode_model3d(obj)


def validate_model3d_encoded(data: bytes | memoryview, metadata: Mapping[str, object]) -> None:
    """Require envelope format metadata to describe the canonical bytes."""
    detected = model3d_format(data)
    if metadata.get("format") != detected:
        raise ValueError("model3d metadata format does not match encoded bytes")
    byte_size = metadata.get("byte_size")
    if isinstance(byte_size, bool) or not isinstance(byte_size, int):
        raise ValueError("model3d metadata byte_size must be an int")
    if byte_size != len(data):
        raise ValueError("model3d metadata byte_size does not match encoded bytes")


MODEL3D_FILE_DECODER_ID = "dinkster.model3d-file@1"
"""Stable identity of :func:`decode_model3d_file` for coerced-input cache
fingerprints (typed assets: identity = asset digest + provider identity).
Bump the ``@N`` suffix whenever the decode SEMANTICS change - same file
bytes producing a different runtime value is a new provider identity."""


def decode_model3d_file(asset: object) -> object:
    """One GLB asset -> the model3d runtime form.

    The ``asset<dinkster.model3d>`` decode provider. ``asset`` is the base
    asset runtime object, duck-typed to its ``open()`` protocol so this
    module never imports dinkster-assets (which depends on this package)."""
    opener = getattr(asset, "open", None)
    if opener is None:
        raise TypeError(
            f"decode_model3d_file expects an asset with open(), got {type(asset).__name__}"
        )
    with opener() as handle:
        data = handle.read()
    try:
        return decode_model3d(data)
    except ValueError as exc:
        name = getattr(asset, "name", "") or "asset"
        raise ValueError(f"cannot decode '{name}' as a 3D model: {exc}") from exc


def register_model3d_type(registry: TypeRegistry, type_id: str) -> None:
    """Register the GLB codec, rendition, and typed-asset decoder."""
    if type_id in registry:
        return
    registry.register(
        type_id,
        encode=encode_model3d,
        decode=decode_model3d,
        fingerprint=model3d_fingerprint(type_id),
        meta=model3d_meta,
        validate_encoded_buffer=validate_model3d_encoded,
    )
    registry.register_rendition(
        type_id,
        "glb",
        mime="model/gltf-binary",
        render=render_model3d_original,
    )
    registry.register_asset_decoder(
        type_id,
        provider_id=MODEL3D_FILE_DECODER_ID,
        decode=decode_model3d_file,
    )
