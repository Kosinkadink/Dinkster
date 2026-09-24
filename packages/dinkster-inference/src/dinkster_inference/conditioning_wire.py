"""Canonical torch-free wire carrier for conditioning records."""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

from dinkster_values import (
    CONDITIONING_HEADER_LIMIT_BYTES,
    ResidencyTable,
    ResidentCodec,
    ResidentLookupError,
    TypeRegistry,
    TypeSpec,
    stable_hash,
)

from .conditioning import (
    EMPTY_RANGE,
    AreaDescriptor,
    AreaUnits,
    ConditioningChannel,
    ConditioningRecord,
    ConditioningSet,
    ExtensionInputValue,
    ExtensionValue,
    MaskDescriptor,
    PayloadDescriptor,
    PayloadReference,
    PercentRange,
    TokenLayoutDescriptor,
    TokenSegmentDescriptor,
    canonical_conditioning_set,
)
from .guidance import ConditionScaleVector

CONDITIONING_TYPE_ID = "dinkster.conditioning"
CONDITIONING_CARRIER_FORMAT = "dinkster-conditioning-carrier-v1"
RESIDENT_CONDITIONING_FORMAT = "dinkster-resident-conditioning-v1"

_MAGIC = b"DMFC"
_RESIDENT_MAGIC = b"DMFR\x01"
_VERSION = 1
_PREFIX_SIZE = len(_MAGIC) + 1 + 8
_MAX_HEADER_BYTES = CONDITIONING_HEADER_LIMIT_BYTES
_MAX_PAYLOADS = 4096
_DTYPE_WIDTHS = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U64": 8,
    "U32": 4,
    "U16": 2,
    "U8": 1,
    "BOOL": 1,
}


class ConditioningWireError(ValueError):
    """A deterministic carrier refusal with a stable machine-readable code."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        message = f"conditioning-wire:{code}"
        super().__init__(f"{message}: {detail}" if detail else message)


@dataclass(frozen=True)
class PayloadBinding:
    """One byte-aligned payload; multi-byte values use little-endian order."""

    reference_id: str
    shape: tuple[int, ...]
    dtype: str
    space: str
    data: bytes

    def __post_init__(self) -> None:
        reference_id = cast("object", self.reference_id)
        shape = cast("object", self.shape)
        dtype = cast("object", self.dtype)
        space = cast("object", self.space)
        data = cast("object", self.data)
        if not isinstance(reference_id, str) or not reference_id:
            raise ConditioningWireError("invalid-binding", "reference_id must be non-empty")
        try:
            reference_id.encode("utf-8")
        except UnicodeEncodeError:
            raise ConditioningWireError(
                "invalid-binding", "reference_id must be valid Unicode"
            ) from None
        if (
            not isinstance(shape, tuple)
            or not shape
            or any(type(dim) is not int or dim < 0 for dim in cast("tuple[object, ...]", shape))
        ):
            raise ConditioningWireError("invalid-binding", "shape must contain non-negative ints")
        if not isinstance(dtype, str) or dtype not in _DTYPE_WIDTHS:
            raise ConditioningWireError("unknown-dtype", repr(dtype))
        if not isinstance(space, str) or not space:
            raise ConditioningWireError("invalid-binding", "space must be non-empty")
        try:
            space.encode("utf-8")
        except UnicodeEncodeError:
            raise ConditioningWireError("invalid-binding", "space must be valid Unicode") from None
        if not isinstance(data, bytes):
            raise ConditioningWireError("invalid-binding", "data must be bytes")
        expected = math.prod(self.shape) * _DTYPE_WIDTHS[self.dtype]
        if len(self.data) != expected:
            raise ConditioningWireError(
                "payload-byte-length",
                f"{self.reference_id!r} has {len(self.data)} bytes, expected {expected}",
            )


@dataclass(frozen=True)
class ConditioningCarrier:
    """Canonical records plus their deduplicated, content-addressed payloads."""

    conditioning: ConditioningSet
    bindings: tuple[PayloadBinding, ...]
    canonical_bytes: bytes | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ResidentConditioningCarrier:
    """Typed conditioning whose opaque payload remains in its worker process."""

    payload: object

    def __post_init__(self) -> None:
        if self.payload is None:
            raise TypeError("resident conditioning payload must not be None")

    @property
    def _dinkster_resident_owner(self) -> object:
        return getattr(self.payload, "_dinkster_resident_owner", self.payload)

    @property
    def _dinkster_resident_refs(self) -> tuple[object, ...]:
        refs = getattr(self.payload, "_dinkster_resident_refs", ())
        if not isinstance(refs, tuple):
            raise TypeError("resident conditioning payload references must be a tuple")
        return cast("tuple[object, ...]", refs)

    @property
    def _dinkster_resident_fingerprint(self) -> str:
        fingerprint = getattr(self.payload, "_dinkster_resident_fingerprint", None)
        if not isinstance(fingerprint, str) or not fingerprint:
            raise TypeError("resident conditioning payload requires a stable fingerprint")
        return fingerprint


def conditioning(value: object, input_id: str) -> ConditioningCarrier:
    """Admit the one native conditioning value form at a consumer boundary."""

    if type(value) is not ConditioningCarrier:
        raise TypeError(f"{input_id} must be a ConditioningCarrier")
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _content_id(binding: PayloadBinding) -> str:
    return "pcid:" + stable_hash(
        [
            _canonical_json(list(binding.shape)),
            binding.dtype.encode("utf-8"),
            binding.space.encode("utf-8"),
            binding.data,
        ]
    )


def _extension_references(value: ExtensionValue) -> Iterable[PayloadReference]:
    if isinstance(value, PayloadReference):
        yield value
    elif isinstance(value, tuple):
        for item in value:
            yield from _extension_references(item)
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _extension_references(item)


def _reachable_references(
    value: ConditioningSet,
) -> tuple[dict[str, list[tuple[tuple[int, ...], str, str]]], set[str]]:
    described: dict[str, list[tuple[tuple[int, ...], str, str]]] = {}
    reachable: set[str] = set()

    def descriptor(payload: PayloadDescriptor) -> None:
        reference_id = payload.reference.id
        reachable.add(reference_id)
        described.setdefault(reference_id, []).append((payload.shape, payload.dtype, payload.space))

    for record in value.records:
        for _, payload in record.channels:
            descriptor(payload)
        if record.mask is not None:
            reachable.add(record.mask.payload.id)
        if record.scale_vector is not None:
            descriptor(record.scale_vector.values)
        for _, metadata in record.extension_metadata:
            for reference in _extension_references(cast("ExtensionValue", metadata)):
                reachable.add(reference.id)
    return described, reachable


def _rewrite_extension(value: ExtensionValue, ids: Mapping[str, str]) -> ExtensionInputValue:
    if isinstance(value, PayloadReference):
        return PayloadReference(ids[value.id])
    if isinstance(value, tuple):
        return tuple(_rewrite_extension(item, ids) for item in value)
    if isinstance(value, Mapping):
        return {key: _rewrite_extension(item, ids) for key, item in value.items()}
    return value


def _rewrite_payload(value: PayloadDescriptor, ids: Mapping[str, str]) -> PayloadDescriptor:
    return PayloadDescriptor(
        PayloadReference(ids[value.reference.id]), value.shape, value.dtype, value.space
    )


def _rewrite_conditioning(value: ConditioningSet, ids: Mapping[str, str]) -> ConditioningSet:
    records: list[ConditioningRecord] = []
    for record in value.records:
        mask = record.mask
        if mask is not None:
            mask = MaskDescriptor(
                PayloadReference(ids[mask.payload.id]),
                mask.strength,
                mask.set_area_to_bounds,
            )
        scale = record.scale_vector
        if scale is not None:
            scale = ConditionScaleVector(_rewrite_payload(scale.values, ids))
        records.append(
            ConditioningRecord(
                channels=tuple(
                    (channel, _rewrite_payload(payload, ids))
                    for channel, payload in record.channels
                ),
                area=record.area,
                mask=mask,
                schedule=record.schedule,
                scale_vector=scale,
                token_layout=record.token_layout,
                extension_metadata=tuple(
                    (key, _rewrite_extension(cast("ExtensionValue", metadata), ids))
                    for key, metadata in sorted(record.extension_metadata)
                ),
            )
        )
    return ConditioningSet(tuple(records))


def make_conditioning_carrier(
    conditioning: ConditioningSet, bindings: Iterable[PayloadBinding]
) -> ConditioningCarrier:
    """Bind every reference, rewrite it to content identity, and deduplicate."""

    if not isinstance(cast("object", conditioning), ConditioningSet):
        raise ConditioningWireError("invalid-conditioning", "expected ConditioningSet")
    described, reachable = _reachable_references(conditioning)
    if len(reachable) > _MAX_PAYLOADS:
        raise ConditioningWireError("payload-count-limit")
    by_reference: dict[str, PayloadBinding] = {}
    for binding in bindings:
        if not isinstance(cast("object", binding), PayloadBinding):
            raise ConditioningWireError("invalid-binding", "expected PayloadBinding")
        if binding.reference_id not in reachable:
            raise ConditioningWireError("unknown-binding", repr(binding.reference_id))
        previous = by_reference.get(binding.reference_id)
        if previous is not None and previous != binding:
            raise ConditioningWireError("conflicting-binding", repr(binding.reference_id))
        by_reference[binding.reference_id] = binding
    missing = sorted(reachable - by_reference.keys())
    if missing:
        raise ConditioningWireError("unbound-reference", repr(missing[0]))

    ids: dict[str, str] = {}
    canonical_bindings: dict[str, PayloadBinding] = {}
    for reference_id in sorted(reachable):
        binding = by_reference[reference_id]
        for expected in described.get(reference_id, ()):
            if (binding.shape, binding.dtype, binding.space) != expected:
                raise ConditioningWireError("descriptor-mismatch", repr(reference_id))
        content_id = _content_id(binding)
        ids[reference_id] = content_id
        canonical = PayloadBinding(
            content_id, binding.shape, binding.dtype, binding.space, binding.data
        )
        previous = canonical_bindings.get(content_id)
        if previous is not None and previous != canonical:
            raise ConditioningWireError("content-id-collision", content_id)
        canonical_bindings[content_id] = canonical

    rewritten = _rewrite_conditioning(conditioning, ids)
    return ConditioningCarrier(
        rewritten, tuple(canonical_bindings[key] for key in sorted(canonical_bindings))
    )


def _manifest(binding: PayloadBinding) -> dict[str, object]:
    return {
        "content_id": binding.reference_id,
        "shape": list(binding.shape),
        "dtype": binding.dtype,
        "space": binding.space,
        "byte_length": len(binding.data),
    }


def _validate_canonical_carrier(carrier: ConditioningCarrier) -> None:
    if len(carrier.bindings) > _MAX_PAYLOADS:
        raise ConditioningWireError("payload-count-limit")
    ids = [binding.reference_id for binding in carrier.bindings]
    if len(set(ids)) != len(ids):
        raise ConditioningWireError("duplicate-content-id")
    if ids != sorted(ids):
        raise ConditioningWireError("unsorted-content-ids")
    described, reachable = _reachable_references(carrier.conditioning)
    available = set(ids)
    missing = sorted(reachable - available)
    if missing:
        raise ConditioningWireError("unbound-reference", repr(missing[0]))
    unknown = sorted(available - reachable)
    if unknown:
        raise ConditioningWireError("unknown-binding", repr(unknown[0]))
    for binding in carrier.bindings:
        if binding.reference_id != _content_id(binding):
            raise ConditioningWireError("content-id-mismatch", binding.reference_id)
        if any(
            (binding.shape, binding.dtype, binding.space) != requirement
            for requirement in described.get(binding.reference_id, ())
        ):
            raise ConditioningWireError("descriptor-mismatch", repr(binding.reference_id))


def _build_carrier_bytes(carrier: ConditioningCarrier) -> bytes:
    if not isinstance(cast("object", carrier), ConditioningCarrier):
        raise ConditioningWireError("invalid-carrier", "expected ConditioningCarrier")
    _validate_canonical_carrier(carrier)
    header = {
        "format": CONDITIONING_CARRIER_FORMAT,
        "conditioning": json.loads(canonical_conditioning_set(carrier.conditioning)),
        "payload_manifest": [_manifest(binding) for binding in carrier.bindings],
    }
    header_bytes = _canonical_json(header)
    if len(header_bytes) > _MAX_HEADER_BYTES:
        raise ConditioningWireError("header-too-large")
    return b"".join(
        (
            _MAGIC,
            bytes((_VERSION,)),
            struct.pack("<Q", len(header_bytes)),
            header_bytes,
            *(binding.data for binding in carrier.bindings),
        )
    )


def encode_conditioning_carrier(carrier: ConditioningCarrier) -> bytes:
    """Fully encode a carrier and retain the exact bytes for codec replay."""

    encoded = _build_carrier_bytes(carrier)
    object.__setattr__(carrier, "canonical_bytes", encoded)
    return encoded


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ConditioningWireError("noncanonical-header", f"duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(constant: str) -> object:
    raise ConditioningWireError("malformed-header-json", constant)


def _parse_finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ConditioningWireError("malformed-header-json", "non-finite number")
    return number


def _contains_negative_zero(value: object) -> bool:
    if isinstance(value, float):
        return value == 0.0 and math.copysign(1.0, value) < 0.0
    if isinstance(value, list):
        return any(_contains_negative_zero(item) for item in cast("Sequence[object]", value))
    if isinstance(value, dict):
        return any(
            _contains_negative_zero(item)
            for item in cast("Mapping[object, object]", value).values()
        )
    return False


def _contains_invalid_unicode(value: object) -> bool:
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            return True
        return False
    if isinstance(value, list):
        return any(_contains_invalid_unicode(item) for item in cast("Sequence[object]", value))
    if isinstance(value, dict):
        mapping = cast("Mapping[object, object]", value)
        return any(
            _contains_invalid_unicode(key) or _contains_invalid_unicode(item)
            for key, item in mapping.items()
        )
    return False


def _mapping(value: object, code: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ConditioningWireError(code, "expected object")
    return cast("Mapping[str, object]", value)


def _sequence(value: object, code: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise ConditioningWireError(code, "expected array")
    return cast("Sequence[object]", value)


def _exact_keys(value: Mapping[str, object], expected: set[str], code: str) -> None:
    if set(value) != expected:
        raise ConditioningWireError(code, f"keys are {sorted(value)!r}")


def _decode_payload(value: object) -> PayloadDescriptor:
    payload = _mapping(value, "invalid-conditioning")
    _exact_keys(payload, {"ref", "shape", "dtype", "space"}, "invalid-conditioning")
    reference = payload["ref"]
    shape = payload["shape"]
    dtype = payload["dtype"]
    space = payload["space"]
    if not isinstance(reference, str) or not isinstance(dtype, str) or not isinstance(space, str):
        raise ConditioningWireError("invalid-conditioning", "invalid payload strings")
    dims = _sequence(shape, "invalid-conditioning")
    return PayloadDescriptor(
        PayloadReference(reference), tuple(cast("int", dim) for dim in dims), dtype, space
    )


def _decode_extension(value: object) -> ExtensionInputValue:
    item = _mapping(value, "invalid-conditioning")
    kind = item.get("type")
    if kind == "none":
        _exact_keys(item, {"type"}, "invalid-conditioning")
        return None
    if kind in {"bool", "int", "float", "str"}:
        _exact_keys(item, {"type", "value"}, "invalid-conditioning")
        raw = item["value"]
        expected = {"bool": bool, "int": int, "float": float, "str": str}[cast("str", kind)]
        if type(raw) is not expected:
            raise ConditioningWireError("invalid-conditioning", f"invalid {kind} value")
        return raw
    if kind == "payload-reference":
        _exact_keys(item, {"type", "id"}, "invalid-conditioning")
        if not isinstance(item["id"], str):
            raise ConditioningWireError("invalid-conditioning", "invalid payload reference")
        return PayloadReference(item["id"])
    if kind == "sequence":
        _exact_keys(item, {"type", "items"}, "invalid-conditioning")
        return tuple(
            _decode_extension(child) for child in _sequence(item["items"], "invalid-conditioning")
        )
    if kind == "mapping":
        _exact_keys(item, {"type", "entries"}, "invalid-conditioning")
        result: dict[str, ExtensionInputValue] = {}
        for raw_entry in _sequence(item["entries"], "invalid-conditioning"):
            entry = _sequence(raw_entry, "invalid-conditioning")
            if len(entry) != 2 or not isinstance(entry[0], str) or entry[0] in result:
                raise ConditioningWireError("invalid-conditioning", "invalid mapping entry")
            result[entry[0]] = _decode_extension(entry[1])
        return result
    raise ConditioningWireError("invalid-conditioning", f"unknown extension type {kind!r}")


def _decode_conditioning(value: object) -> ConditioningSet:
    root = _mapping(value, "invalid-conditioning")
    _exact_keys(root, {"format", "records"}, "invalid-conditioning")
    if root["format"] != "dinkster-conditioning-set-v1":
        raise ConditioningWireError("invalid-conditioning", "unknown conditioning format")
    records: list[ConditioningRecord] = []
    for raw_record in _sequence(root["records"], "invalid-conditioning"):
        record = _mapping(raw_record, "invalid-conditioning")
        _exact_keys(
            record,
            {
                "channels",
                "area",
                "mask",
                "schedule",
                "scale_vector",
                "token_layout",
                "extension_metadata",
            },
            "invalid-conditioning",
        )
        channels: list[tuple[ConditioningChannel, PayloadDescriptor]] = []
        for raw_channel in _sequence(record["channels"], "invalid-conditioning"):
            channel = _mapping(raw_channel, "invalid-conditioning")
            _exact_keys(channel, {"id", "payload"}, "invalid-conditioning")
            if not isinstance(channel["id"], str):
                raise ConditioningWireError("invalid-conditioning", "invalid channel id")
            channels.append(
                (ConditioningChannel(channel["id"]), _decode_payload(channel["payload"]))
            )

        area = None
        if record["area"] is not None:
            raw_area = _mapping(record["area"], "invalid-conditioning")
            area_keys = {"height", "width", "y", "x", "units", "strength"}
            if set(raw_area) not in (area_keys, area_keys | {"temporal", "z"}):
                raise ConditioningWireError(
                    "invalid-conditioning", f"keys are {sorted(raw_area)!r}"
                )
            area = AreaDescriptor(
                cast("int | float", raw_area["height"]),
                cast("int | float", raw_area["width"]),
                cast("int | float", raw_area["y"]),
                cast("int | float", raw_area["x"]),
                AreaUnits(cast("str", raw_area["units"])),
                cast("float", raw_area["strength"]),
                cast("float | None", raw_area.get("temporal")),
                cast("float | None", raw_area.get("z")),
            )

        mask = None
        if record["mask"] is not None:
            raw_mask = _mapping(record["mask"], "invalid-conditioning")
            _exact_keys(
                raw_mask,
                {"payload_ref", "strength", "set_area_to_bounds"},
                "invalid-conditioning",
            )
            mask = MaskDescriptor(
                PayloadReference(cast("str", raw_mask["payload_ref"])),
                cast("float", raw_mask["strength"]),
                cast("bool", raw_mask["set_area_to_bounds"]),
            )

        raw_range = _mapping(record["schedule"], "invalid-conditioning")
        if raw_range.get("kind") == "empty":
            _exact_keys(raw_range, {"kind"}, "invalid-conditioning")
            schedule = EMPTY_RANGE
        elif raw_range.get("kind") == "percent":
            _exact_keys(raw_range, {"kind", "start", "end"}, "invalid-conditioning")
            schedule = PercentRange(
                cast("float", raw_range["start"]), cast("float", raw_range["end"])
            )
        else:
            raise ConditioningWireError("invalid-conditioning", "invalid schedule kind")

        scale = None
        if record["scale_vector"] is not None:
            scale = ConditionScaleVector(_decode_payload(record["scale_vector"]))

        layout = None
        if record["token_layout"] is not None:
            raw_layout = _mapping(record["token_layout"], "invalid-conditioning")
            _exact_keys(
                raw_layout,
                {"family_id", "version", "text_streams", "segments"},
                "invalid-conditioning",
            )
            segments: list[TokenSegmentDescriptor] = []
            for raw_segment in _sequence(raw_layout["segments"], "invalid-conditioning"):
                segment = _mapping(raw_segment, "invalid-conditioning")
                _exact_keys(
                    segment,
                    {"name", "stream", "start_token", "token_count"},
                    "invalid-conditioning",
                )
                segments.append(
                    TokenSegmentDescriptor(
                        cast("str", segment["name"]),
                        cast("str", segment["stream"]),
                        cast("int", segment["start_token"]),
                        cast("int | None", segment["token_count"]),
                    )
                )
            layout = TokenLayoutDescriptor(
                cast("str", raw_layout["family_id"]),
                cast("int", raw_layout["version"]),
                tuple(
                    cast("str", stream)
                    for stream in _sequence(raw_layout["text_streams"], "invalid-conditioning")
                ),
                tuple(segments),
            )

        metadata = _mapping(record["extension_metadata"], "invalid-conditioning")
        records.append(
            ConditioningRecord(
                channels=tuple(channels),
                area=area,
                mask=mask,
                schedule=schedule,
                scale_vector=scale,
                token_layout=layout,
                extension_metadata=tuple(
                    (key, _decode_extension(raw)) for key, raw in metadata.items()
                ),
            )
        )
    result = ConditioningSet(tuple(records))
    if json.loads(canonical_conditioning_set(result)) != value:
        raise ConditioningWireError("noncanonical-conditioning")
    return result


def decode_conditioning_carrier(
    data: bytes, envelope_metadata: Mapping[str, object] | None = None
) -> ConditioningCarrier:
    """Parse, validate, and reconstruct a canonical conditioning carrier."""

    if (
        envelope_metadata is not None
        and envelope_metadata.get("format") != CONDITIONING_CARRIER_FORMAT
    ):
        raise ConditioningWireError("metadata-format-mismatch")
    if len(data) < _PREFIX_SIZE:
        raise ConditioningWireError("truncated-prefix")
    if data[:4] != _MAGIC:
        raise ConditioningWireError("wrong-magic")
    if data[4] != _VERSION:
        raise ConditioningWireError("unknown-version", str(data[4]))
    header_length = struct.unpack("<Q", data[5:13])[0]
    if header_length > _MAX_HEADER_BYTES:
        raise ConditioningWireError("header-too-large")
    if header_length > len(data) - _PREFIX_SIZE:
        raise ConditioningWireError("truncated-header")
    header_bytes = data[_PREFIX_SIZE : _PREFIX_SIZE + header_length]
    try:
        header: object = json.loads(
            header_bytes,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
            parse_float=_parse_finite_float,
        )
        if _contains_invalid_unicode(header):
            raise ConditioningWireError("malformed-header-json", "invalid Unicode scalar")
        if _contains_negative_zero(header) or _canonical_json(header) != header_bytes:
            raise ConditioningWireError("noncanonical-header")
    except ConditioningWireError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise ConditioningWireError("malformed-header-json", str(error)) from None
    root = _mapping(header, "malformed-header-json")
    _exact_keys(root, {"format", "conditioning", "payload_manifest"}, "malformed-header-json")
    if root["format"] != CONDITIONING_CARRIER_FORMAT:
        raise ConditioningWireError("unknown-format-tag", repr(root["format"]))
    raw_manifest = _sequence(root["payload_manifest"], "malformed-manifest")
    if len(raw_manifest) > _MAX_PAYLOADS:
        raise ConditioningWireError("payload-count-limit")

    manifest: list[PayloadBinding] = []
    offset = _PREFIX_SIZE + header_length
    ids: list[str] = []
    for raw_entry in raw_manifest:
        entry = _mapping(raw_entry, "malformed-manifest")
        _exact_keys(
            entry,
            {"content_id", "shape", "dtype", "space", "byte_length"},
            "malformed-manifest",
        )
        content_id = entry["content_id"]
        dtype = entry["dtype"]
        space = entry["space"]
        byte_length = entry["byte_length"]
        if (
            not isinstance(content_id, str)
            or not isinstance(dtype, str)
            or not isinstance(space, str)
        ):
            raise ConditioningWireError("malformed-manifest", "invalid string field")
        if dtype not in _DTYPE_WIDTHS:
            raise ConditioningWireError("unknown-dtype", repr(dtype))
        if type(byte_length) is not int or byte_length < 0:
            raise ConditioningWireError("malformed-manifest", "invalid byte_length")
        dims = _sequence(entry["shape"], "malformed-manifest")
        shape = tuple(cast("int", dim) for dim in dims)
        if not shape or any(type(dim) is not int or dim < 0 for dim in shape):
            raise ConditioningWireError("malformed-manifest", "invalid shape")
        if byte_length != math.prod(shape) * _DTYPE_WIDTHS[dtype]:
            raise ConditioningWireError("segment-length-mismatch", content_id)
        end = offset + byte_length
        if end > len(data):
            raise ConditioningWireError("segment-length-mismatch", content_id)
        binding = PayloadBinding(content_id, shape, dtype, space, data[offset:end])
        ids.append(content_id)
        manifest.append(binding)
        offset = end
    if len(set(ids)) != len(ids):
        raise ConditioningWireError("duplicate-content-id")
    if ids != sorted(ids):
        raise ConditioningWireError("unsorted-content-ids")
    if offset != len(data):
        raise ConditioningWireError("trailing-bytes")
    for binding in manifest:
        if binding.reference_id != _content_id(binding):
            raise ConditioningWireError("content-id-mismatch", binding.reference_id)

    try:
        conditioning = _decode_conditioning(root["conditioning"])
    except ConditioningWireError:
        raise
    except (TypeError, ValueError) as error:
        raise ConditioningWireError("invalid-conditioning", str(error)) from None
    described, reachable = _reachable_references(conditioning)
    manifest_by_id = {binding.reference_id: binding for binding in manifest}
    missing = sorted(reachable - manifest_by_id.keys())
    if missing:
        raise ConditioningWireError("unbound-reference", repr(missing[0]))
    unknown = sorted(manifest_by_id.keys() - reachable)
    if unknown:
        raise ConditioningWireError("unknown-binding", repr(unknown[0]))
    for reference_id, requirements in described.items():
        binding = manifest_by_id[reference_id]
        if any((binding.shape, binding.dtype, binding.space) != item for item in requirements):
            raise ConditioningWireError("descriptor-mismatch", repr(reference_id))
    carrier = ConditioningCarrier(conditioning, tuple(manifest), data)
    if _build_carrier_bytes(carrier) != data:
        raise ConditioningWireError("noncanonical-carrier")
    return carrier


def register_conditioning_type(
    registry: TypeRegistry,
    *,
    resident_table: ResidencyTable | None = None,
    resident_meta: Callable[[object], Mapping[str, object]] | None = None,
) -> TypeSpec:
    """Register canonical and process-resident conditioning carriers."""

    resident_codec = ResidentCodec(resident_table, resident_meta)

    def coerce(obj: object) -> object:
        if (
            not isinstance(obj, ConditioningCarrier)
            and type(obj) is not ResidentConditioningCarrier
        ):
            raise ConditioningWireError(
                "invalid-carrier", "expected canonical or resident conditioning carrier"
            )
        return obj

    def fingerprint(obj: object) -> str:
        if type(obj) is ResidentConditioningCarrier:
            return resident_codec.fingerprint(obj)
        carrier = cast("ConditioningCarrier", obj)
        encoded = encode_conditioning_carrier(carrier)
        return stable_hash([encoded])

    def encode(obj: object) -> bytes:
        if type(obj) is ResidentConditioningCarrier:
            return _RESIDENT_MAGIC + resident_codec.encode(obj)
        carrier = cast("ConditioningCarrier", obj)
        if carrier.canonical_bytes is None:
            raise ConditioningWireError("carrier-not-wrapped")
        return carrier.canonical_bytes

    def decode(data: bytes) -> object:
        if data.startswith(_RESIDENT_MAGIC):
            value = resident_codec.decode(data[len(_RESIDENT_MAGIC) :])
            if type(value) is not ResidentConditioningCarrier:
                raise ConditioningWireError("invalid-resident-carrier")
            return value
        return decode_conditioning_carrier(data)

    def validate_encoded(data: bytes, metadata: Mapping[str, object]) -> None:
        if data.startswith(_RESIDENT_MAGIC):
            if metadata.get("format") != RESIDENT_CONDITIONING_FORMAT:
                raise ConditioningWireError("metadata-format")
            try:
                resident_codec.validate_encoded(data[len(_RESIDENT_MAGIC) :])
            except ResidentLookupError as error:
                raise ConditioningWireError("malformed-resident-stub", str(error)) from None
            return
        decode_conditioning_carrier(data, metadata)

    def metadata(obj: object) -> Mapping[str, object]:
        if type(obj) is ResidentConditioningCarrier:
            return {
                **resident_codec.metadata(obj),
                "format": RESIDENT_CONDITIONING_FORMAT,
            }
        return {"format": CONDITIONING_CARRIER_FORMAT}

    return registry.register(
        CONDITIONING_TYPE_ID,
        encode=encode,
        decode=decode,
        fingerprint=fingerprint,
        meta=metadata,
        coerce=coerce,
        validate_encoded=validate_encoded,
    )
