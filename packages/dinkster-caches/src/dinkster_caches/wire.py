"""Cache entry wire format: one manifest shape for disk and HTTP.

A persisted cache entry is a small JSON manifest naming, per output, the
envelope facts (type id, fingerprint, meta bytes) and the CAS digest of the
payload bytes. The same manifest is the on-disk file *and* the peer
endpoint's response body, so "rehydrate from my own disk" and "rehydrate
from a peer's cache" are one function with a different blob fetcher.

Trust posture matches the boundary's (DESIGN 3.2): meta bytes use the
default codec (JSON first, pickle fallback), so manifests must only be
accepted from the same trust domain as workers - peers you would hand a
worker token to. This is documented honesty, not a new hole: every payload
codec already has the same property.

Conservative on every failure: a manifest that does not parse, a blob that
is missing or corrupt, meta that does not decode to a mapping - all of it
is a cache miss, never a malformed Value.
"""

from __future__ import annotations

import base64
import pickle
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, cast

from dinkster_assets import AssetError
from dinkster_assets.value import metadata_asset_references
from dinkster_protocol import CachedResult, ExpandedReturn, SavedArtifact
from dinkster_protocol.result_document import return_unit_from_document, return_unit_to_document
from dinkster_values import (
    SAVED_ARTIFACT_CLAIM_LIMIT,
    EncodedPayload,
    ListPayload,
    TuplePayload,
    TypeRegistry,
    Value,
    ValueMeta,
    default_decode,
    default_encode,
    parse_list_type_id,
    parse_tuple_type_id,
    value_children,
    value_resource_ids,
)
from dinkster_values.storage import encoded_storage_meta

ENTRY_WIRE_VERSION = 3
"""v3 adds an atomic expansion document beside the present output values.
v1 (scalar) and v2 (recursive lists) remain readable as direct results.
Older readers reject v3 rather than returning an incomplete expansion.
Entries with saved-file associations use v4 so older readers miss instead
of silently losing the association. Entries without artifacts remain v3."""

_ACCEPTED_VERSIONS = (1, 2, 3, 4)

BlobFetch = Callable[[str, int], Awaitable[bytes | None]]
"""Fetch verified blob bytes by (digest, expected size); None on miss."""

BlobStore = Callable[[bytes], str]
"""Store blob bytes, returning the CAS digest they live under."""


@dataclass(frozen=True)
class EncodedValue:
    """One value ready to persist: leaf codec bytes, or list children.

    Exactly one of ``data`` (leaf) / ``children`` (list) is meaningful -
    lists persist as structure, their leaves as blobs, so two entries
    sharing an element share its bytes in the CAS."""

    value: Value
    data: bytes | None
    children: tuple[EncodedValue, ...] = ()


@dataclass(frozen=True)
class EncodedEntry:
    outputs: Mapping[str, EncodedValue]
    expansion: dict[str, object] | None
    artifacts: tuple[SavedArtifact, ...] = ()


def _encode_value(value: Value, registry: TypeRegistry) -> EncodedValue | None:
    children = value_children(value)
    if children is not None:
        encoded_children: list[EncodedValue] = []
        for child in children:
            encoded = _encode_value(child, registry)
            if encoded is None:
                return None
            encoded_children.append(encoded)
        return EncodedValue(value, None, tuple(encoded_children))
    payload = value.payload
    if isinstance(payload, EncodedPayload) and payload.type_id == value.type_id:
        data = payload.data
    elif value.type_id in registry:
        try:
            data = registry.spec(value.type_id).encode(payload.load())
        except Exception:
            return None
    else:
        return None
    return EncodedValue(value, data)


def encode_entry(outputs: Mapping[str, Value], registry: TypeRegistry) -> EncodedEntry | None:
    """Each output encoded for persistence, or None if the entry cannot be.

    Persistence is refused when:

    - A value referencing live process state anywhere in its tree
      (RESOURCE_ID_META_KEY on itself or a list child) would dangle past
      that process's life. The whole entry is refused - a partial entry
      would fail the engine's output-contract check anyway.
    - A value whose type is unregistered here and whose payload is not
      already encoded has no bytes to store.
    - A registered codec cannot encode the value's object payload.

    Values that arrived encoded (EncodedPayload) reuse their bytes as-is -
    persisting is a relay, not a re-encode (same rule as the boundary).
    """
    encoded: dict[str, EncodedValue] = {}
    for output_id, value in outputs.items():
        if value_resource_ids(value):
            return None
        encoded_value = _encode_value(value, registry)
        if encoded_value is None:
            return None
        encoded[output_id] = encoded_value
    expansion = outputs.expansion if isinstance(outputs, CachedResult) else None
    document = None
    if expansion is not None:
        index = 0

        def encode_present(_value: Value) -> int:
            nonlocal index
            result = index
            index += 1
            return result

        document = return_unit_to_document(expansion, encode_present)
    return EncodedEntry(
        encoded, document, outputs.artifacts if isinstance(outputs, CachedResult) else ()
    )


def _value_to_wire(encoded: EncodedValue, store_blob: BlobStore) -> dict[str, Any]:
    wire: dict[str, Any] = {
        "typeId": encoded.value.type_id,
        "fingerprint": encoded.value.fingerprint,
        "metaB64": base64.b64encode(default_encode(dict(encoded.value.meta.entries))).decode(
            "ascii"
        ),
    }
    if encoded.data is None:
        wire["elements"] = [_value_to_wire(child, store_blob) for child in encoded.children]
    else:
        wire["digest"] = store_blob(encoded.data)
        wire["size"] = len(encoded.data)
    return wire


def entry_to_wire(key: str, encoded: EncodedEntry, store_blob: BlobStore) -> dict[str, Any]:
    """The manifest for an encoded entry; ``store_blob`` persists each leaf's
    bytes and names the CAS digest the manifest references."""
    outputs: dict[str, Any] = {
        output_id: _value_to_wire(encoded_value, store_blob)
        for output_id, encoded_value in encoded.outputs.items()
    }
    wire: dict[str, Any] = {
        "version": ENTRY_WIRE_VERSION,
        "key": key,
        "outputs": outputs,
        "expansion": encoded.expansion,
    }
    if encoded.artifacts:
        wire["version"] = 4
        wire["artifacts"] = [asdict(item) for item in encoded.artifacts]
    return wire


def iter_manifest_payloads(manifest: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    """Every leaf payload descriptor (``digest``/``size``) in a manifest,
    however deeply nested in list ``elements``. Blob GC and footprint
    accounting walk manifests only through this."""
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        return
    stack: list[Any] = list(cast("Mapping[Any, Any]", outputs).values())
    while stack:
        entry = stack.pop()
        if not isinstance(entry, Mapping):
            continue
        entry_map = cast("Mapping[str, Any]", entry)
        elements = entry_map.get("elements")
        if isinstance(elements, Sequence) and not isinstance(elements, str | bytes):
            stack.extend(cast("Sequence[Any]", elements))
        elif "digest" in entry_map:
            yield entry_map


def manifest_asset_references(manifest: Mapping[str, object]) -> frozenset[str]:
    """External asset roots in persisted values; malformed state refuses collection.

    Payload digests belong to the cache's own CAS, not the asset vault. Read
    value envelopes and explicit saved-file/preview associations only.
    """
    if type(manifest.get("version")) is not int or manifest["version"] not in _ACCEPTED_VERSIONS:
        raise AssetError("cache reference manifest version is unavailable")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise AssetError("cache reference outputs are unavailable")
    stack = list(cast("Mapping[str, object]", outputs).values())
    references: dict[str, int] = {}
    while stack:
        raw = stack.pop()
        if not isinstance(raw, Mapping):
            raise AssetError("cache reference value is invalid")
        value = cast("Mapping[str, object]", raw)
        type_id, metadata = value.get("typeId"), value.get("metaB64")
        if not isinstance(type_id, str) or not isinstance(metadata, str):
            raise AssetError("cache reference metadata is unavailable")
        try:
            decoded = default_decode(base64.b64decode(metadata, validate=True))
        except Exception as exc:
            raise AssetError("cache reference metadata is unreadable") from exc
        if not isinstance(decoded, Mapping):
            raise AssetError("cache reference metadata must be a mapping")
        for ref in metadata_asset_references(type_id, cast("Mapping[str, object]", decoded)):
            previous = references.setdefault(ref.digest, ref.size)
            if previous != ref.size:
                raise AssetError("source references disagree about byte size")
        if "elements" in value:
            elements = value["elements"]
            if not isinstance(elements, list):
                raise AssetError("cache reference elements must be a list")
            stack.extend(cast("list[object]", elements))
    try:
        artifacts = _artifacts_from_wire(manifest)
    except (TypeError, ValueError, KeyError) as exc:
        raise AssetError("cache artifact references are unavailable") from exc
    for artifact in artifacts:
        for item in (artifact, artifact.preview):
            if item is not None:
                previous = references.setdefault(item.digest, item.size)
                if previous != item.size:
                    raise AssetError("source references disagree about byte size")
    return frozenset(references)


async def _value_from_wire(
    out_map: Mapping[str, Any], fetch_blob: BlobFetch, registry: TypeRegistry
) -> Value | None:
    try:
        type_id = str(out_map["typeId"])
        fingerprint = str(out_map["fingerprint"])
        meta_raw = default_decode(base64.b64decode(str(out_map["metaB64"])))
    except (
        AttributeError,
        EOFError,
        ImportError,
        IndexError,
        KeyError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
        pickle.UnpicklingError,
    ):
        return None
    if not isinstance(meta_raw, Mapping):
        return None
    meta = ValueMeta({str(k): v for k, v in cast("Mapping[object, object]", meta_raw).items()})
    elements = out_map.get("elements")
    if elements is not None:
        positions = parse_tuple_type_id(type_id)
        if parse_list_type_id(type_id) is None and positions is None:
            return None
        if not isinstance(elements, Sequence) or isinstance(elements, str | bytes):
            return None
        children: list[Value] = []
        for element in cast("Sequence[Any]", elements):
            if not isinstance(element, Mapping):
                return None
            child = await _value_from_wire(cast("Mapping[str, Any]", element), fetch_blob, registry)
            if child is None:
                return None
            children.append(child)
        if positions is not None and positions != tuple(child.type_id for child in children):
            return None
        return Value(
            type_id=type_id,
            fingerprint=fingerprint,
            meta=meta,
            payload=ListPayload(tuple(children))
            if positions is None
            else TuplePayload(tuple(children)),
        )
    try:
        digest = str(out_map["digest"])
        size = int(out_map["size"])
    except (KeyError, TypeError, ValueError):
        return None
    data = await fetch_blob(digest, size)
    if data is None or len(data) != size:
        return None
    meta = ValueMeta(encoded_storage_meta(meta.entries, size))
    spec = registry.spec(type_id) if type_id in registry else None
    if spec is not None and (
        spec.validate_encoded is not None or spec.validate_encoded_buffer is not None
    ):
        try:
            if spec.validate_encoded_buffer is not None:
                spec.validate_encoded_buffer(memoryview(data), meta.entries)
            else:
                assert spec.validate_encoded is not None
                spec.validate_encoded(data, meta.entries)
        except Exception:
            return None
    decoder = spec.decode if spec is not None else None
    return Value(
        type_id=type_id,
        fingerprint=fingerprint,
        meta=meta,
        payload=EncodedPayload(
            type_id,
            data,
            decoder,
            transport="cas",
        ),
    )


def _artifacts_from_wire(wire: Mapping[str, Any]) -> tuple[SavedArtifact, ...]:
    raw_artifacts = wire["artifacts"] if wire.get("version") == 4 else wire.get("artifacts", [])
    if type(raw_artifacts) is not list:
        raise ValueError("cache artifacts must be a list")
    rows = cast("list[Any]", raw_artifacts)
    if len(rows) > SAVED_ARTIFACT_CLAIM_LIMIT:
        raise ValueError("cache artifact count exceeds the limit")
    artifacts: list[SavedArtifact] = []
    for raw in rows:
        item: dict[str, Any] = dict(raw)
        preview = item.pop("preview", None)
        artifacts.append(
            SavedArtifact(**item, preview=SavedArtifact(**preview) if preview is not None else None)
        )
    return tuple(artifacts)


async def entry_from_wire(
    wire: Mapping[str, Any], fetch_blob: BlobFetch, registry: TypeRegistry
) -> CachedResult | None:
    """Rehydrate an entry from its manifest, or None (a miss) if anything
    about it cannot be trusted. Types unregistered here still rehydrate -
    as encoded envelopes that relay but do not load (hazard H2)."""
    if wire.get("version") not in _ACCEPTED_VERSIONS:
        return None
    if wire.get("version") in (3, 4):
        fields = {"version", "key", "outputs", "expansion"}
        if wire.get("version") == 4:
            fields.add("artifacts")
        if set(wire) != fields:
            return None
    elif "expansion" in wire:
        return None
    outputs_wire = wire.get("outputs")
    if not isinstance(outputs_wire, Mapping):
        return None
    entry: dict[str, Value] = {}
    for output_id, out in cast("Mapping[Any, Any]", outputs_wire).items():
        if not isinstance(out, Mapping):
            return None
        value = await _value_from_wire(cast("Mapping[str, Any]", out), fetch_blob, registry)
        if value is None or value_resource_ids(value):
            return None
        entry[str(output_id)] = value
    try:
        artifacts = _artifacts_from_wire(wire)
    except (TypeError, ValueError, KeyError):
        return None
    document = wire.get("expansion")
    if document is None:
        return CachedResult(entry, artifacts=artifacts)
    used: set[str] = set()

    def decode_present(index: object, output_id: str) -> Value:
        if type(index) is not int or index != len(used) or output_id in used:
            raise ValueError("invalid cached present binding index")
        used.add(output_id)
        return entry[output_id]

    try:
        expansion = return_unit_from_document(document, decode_present)
        if not isinstance(expansion, ExpandedReturn) or used != set(entry):
            return None
        return CachedResult(entry, expansion, artifacts)
    except (KeyError, TypeError, ValueError, RecursionError, OverflowError):
        return None
