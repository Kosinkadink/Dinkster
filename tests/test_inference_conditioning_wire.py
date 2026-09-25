"""Canonical conditioning carrier proofs."""

from __future__ import annotations

import asyncio
import inspect
import json
import struct
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from dinkster_caches import DiskCacheStore
from dinkster_inference import (
    CONDITIONING_CARRIER_FORMAT,
    CONDITIONING_TYPE_ID,
    EMPTY_RANGE,
    RESIDENT_CONDITIONING_FORMAT,
    AreaDescriptor,
    AreaUnits,
    ConditioningCarrier,
    ConditioningChannel,
    ConditioningRecord,
    ConditioningSet,
    ConditioningWireError,
    MaskDescriptor,
    PayloadBinding,
    PayloadDescriptor,
    PayloadReference,
    PercentRange,
    ResidentConditioningCarrier,
    TokenLayoutDescriptor,
    TokenSegmentDescriptor,
    canonical_conditioning_set,
    conditioning,
    decode_conditioning_carrier,
    encode_conditioning_carrier,
    make_conditioning_carrier,
    register_conditioning_type,
)
from dinkster_inference.guidance import ConditionScaleVector
from dinkster_values import (
    RESOURCE_ID_META_KEY,
    RESOURCE_OWNER_META_KEY,
    RESOURCE_REFS_META_KEY,
    EncodedPayload,
    ResidencyTable,
    ResidentLookupError,
    TypeRegistry,
    Value,
    ValueMeta,
    process_instance_token,
    stable_hash,
)
from dinkster_workers import ValueCodec
from dinkster_workers.boundary import EncodedPayloadValidationError


def _payload(
    reference_id: str, shape: tuple[int, ...], dtype: str, space: str
) -> PayloadDescriptor:
    return PayloadDescriptor(PayloadReference(reference_id), shape, dtype, space)


def _complete_source(prefix: str = "source", *, negative_zero: bool = False) -> ConditioningSet:
    zero = -0.0 if negative_zero else 0.0
    text = _payload(f"{prefix}-text", (1, 2), "F32", "text")
    scale = _payload(f"{prefix}-scale", (1,), "F32", "scale")
    mask = PayloadReference(f"{prefix}-mask")
    record = ConditioningRecord(
        channels=((ConditioningChannel.TEXT, text),),
        area=AreaDescriptor(zero, 0.5, zero, 0.5, AreaUnits.PERCENT, zero),
        mask=MaskDescriptor(mask, zero, True),
        schedule=PercentRange(zero, 1.0),
        scale_vector=ConditionScaleVector(scale),
        token_layout=TokenLayoutDescriptor(
            "dinkster.test",
            1,
            ("text",),
            (TokenSegmentDescriptor("prompt", "text", 0, 2),),
        ),
        extension_metadata=(
            ("pack/payload", mask),
            ("pack/nested", {"zero": zero, "refs": [mask]}),
        ),
    )
    empty = ConditioningRecord(channels=((ConditioningChannel.POOLED, text),), schedule=EMPTY_RANGE)
    return ConditioningSet((record, empty))


def _bindings(prefix: str = "source") -> tuple[PayloadBinding, ...]:
    return (
        PayloadBinding(f"{prefix}-text", (1, 2), "F32", "text", bytes(range(8))),
        PayloadBinding(f"{prefix}-scale", (1,), "F32", "scale", b"scale"[:4]),
        PayloadBinding(f"{prefix}-mask", (2, 2), "U8", "mask", b"mask"),
    )


def _carrier(prefix: str = "source", *, negative_zero: bool = False) -> ConditioningCarrier:
    return make_conditioning_carrier(
        _complete_source(prefix, negative_zero=negative_zero), _bindings(prefix)
    )


def _split(encoded: bytes) -> tuple[dict[str, object], bytes]:
    length = struct.unpack("<Q", encoded[5:13])[0]
    return json.loads(encoded[13 : 13 + length]), encoded[13 + length :]


def _frame(header: object, payloads: bytes = b"", *, canonical: bool = True) -> bytes:
    header_bytes = json.dumps(
        header,
        ensure_ascii=True,
        separators=(",", ":") if canonical else (", ", ": "),
        sort_keys=True,
    ).encode()
    return b"DMFC\x01" + struct.pack("<Q", len(header_bytes)) + header_bytes + payloads


def _assert_code(code: str, action: Callable[[], object]) -> None:
    with pytest.raises(ConditioningWireError) as caught:
        action()
    assert caught.value.code == code
    assert str(caught.value).startswith(f"conditioning-wire:{code}")


def test_conditioning_accepts_only_the_canonical_carrier_form() -> None:
    carrier = _carrier()

    assert conditioning(carrier, "positive") is carrier
    with pytest.raises(TypeError, match="positive must come from a Dinkster conditioning node"):
        conditioning(ResidentConditioningCarrier(object()), "positive")
    with pytest.raises(TypeError, match="negative must come from a Dinkster conditioning node"):
        conditioning([[object(), {}]], "negative")


def test_wrap_fingerprints_full_canonical_bytes_and_codec_replays_exactly() -> None:
    registry = TypeRegistry()
    spec = register_conditioning_type(registry)
    carrier = _carrier()
    assert carrier.canonical_bytes is None
    value = registry.wrap(CONDITIONING_TYPE_ID, carrier)
    assert carrier.canonical_bytes is not None
    assert value.fingerprint == stable_hash([carrier.canonical_bytes])
    assert spec.encode(value.resolve()) is carrier.canonical_bytes
    assert value.meta.entries == {"format": CONDITIONING_CARRIER_FORMAT}


def test_resident_conditioning_uses_owner_routed_stub_without_serializing_payload() -> None:
    class Payload:
        def __init__(self, owner: object, reference: object) -> None:
            self.owner = owner
            self.reference = reference

        @property
        def _dinkster_resident_owner(self) -> object:
            return self.owner

        @property
        def _dinkster_resident_refs(self) -> tuple[object, ...]:
            return (self.reference,)

        @property
        def _dinkster_resident_fingerprint(self) -> str:
            return "conditioning:stable"

    table = ResidencyTable()
    registry = TypeRegistry()
    spec = register_conditioning_type(registry, resident_table=table)
    owner = object()
    reference = object()
    carrier = ResidentConditioningCarrier(Payload(owner, reference))

    value = registry.wrap(CONDITIONING_TYPE_ID, carrier)
    encoded = spec.encode(value.resolve())

    assert encoded.startswith(b"DMFR\x01")
    assert len(encoded) < 128
    assert value.fingerprint == "conditioning:stable"
    assert value.meta.get("format") == RESIDENT_CONDITIONING_FORMAT
    assert value.meta.get(RESOURCE_ID_META_KEY) == "resident:" + table.rid_for(owner)
    assert value.meta.get(RESOURCE_REFS_META_KEY) == ("resident:" + table.rid_for(reference),)
    assert value.meta.get(RESOURCE_OWNER_META_KEY) == process_instance_token()
    assert spec.decode(encoded) is carrier
    assert spec.validate_encoded is not None
    spec.validate_encoded(encoded, value.meta.entries)

    other = TypeRegistry()
    other_spec = register_conditioning_type(other, resident_table=ResidencyTable())
    with pytest.raises(ResidentLookupError, match="another worker"):
        other_spec.decode(encoded)


def test_order_independent_fingerprint_content_dedup_and_signed_zero() -> None:
    registry = TypeRegistry()
    register_conditioning_type(registry)
    first = make_conditioning_carrier(_complete_source("a"), reversed(_bindings("a")))
    second = _carrier("b", negative_zero=True)
    first_value = registry.wrap(CONDITIONING_TYPE_ID, first)
    second_value = registry.wrap(CONDITIONING_TYPE_ID, second)
    assert first_value.fingerprint == second_value.fingerprint
    assert first.canonical_bytes == second.canonical_bytes
    assert len(first.bindings) == 3
    text_refs = [
        payload.reference.id
        for record in first.conditioning.records
        for _, payload in record.channels
    ]
    assert text_refs[0] == text_refs[1]


def test_round_trip_reconstructs_records_payloads_and_empty_singleton() -> None:
    carrier = _carrier()
    encoded = encode_conditioning_carrier(carrier)
    header, _ = _split(encoded)
    conditioning_header = header["conditioning"]
    assert isinstance(conditioning_header, dict)
    records = conditioning_header["records"]
    assert isinstance(records, list)
    area_header = records[0]["area"]
    assert "temporal" not in area_header
    assert "z" not in area_header
    decoded = decode_conditioning_carrier(encoded, {"format": CONDITIONING_CARRIER_FORMAT})
    assert canonical_conditioning_set(decoded.conditioning) == canonical_conditioning_set(
        carrier.conditioning
    )
    assert [(item.reference_id, item.data) for item in decoded.bindings] == [
        (item.reference_id, item.data) for item in carrier.bindings
    ]
    assert decoded.conditioning.records[1].schedule is EMPTY_RANGE
    first = decoded.conditioning.records[0]
    assert first.extension_metadata == carrier.conditioning.records[0].extension_metadata


def test_round_trip_preserves_temporal_percent_area() -> None:
    source = _complete_source()
    area = AreaDescriptor(
        0.25,
        0.5,
        0.0,
        0.125,
        AreaUnits.PERCENT,
        0.6,
        temporal=0.75,
        z=0.25,
    )
    conditioning = ConditioningSet((replace(source.records[0], area=area), source.records[1]))
    carrier = make_conditioning_carrier(conditioning, _bindings())

    encoded = encode_conditioning_carrier(carrier)
    header, _ = _split(encoded)
    conditioning_header = header["conditioning"]
    assert isinstance(conditioning_header, dict)
    records = conditioning_header["records"]
    assert isinstance(records, list)
    area_header = records[0]["area"]
    assert area_header["temporal"] == 0.75
    assert area_header["z"] == 0.25
    decoded = decode_conditioning_carrier(encoded)

    assert decoded.conditioning.records[0].area == area
    assert encode_conditioning_carrier(decoded) == encoded


def test_decoded_carrier_rewrap_is_byte_identical_and_fingerprint_stable() -> None:
    registry = TypeRegistry()
    spec = register_conditioning_type(registry)
    original = registry.wrap(CONDITIONING_TYPE_ID, _carrier())
    original_bytes = spec.encode(original.resolve())
    decoded = decode_conditioning_carrier(original_bytes)
    rebuilt = make_conditioning_carrier(decoded.conditioning, decoded.bindings)
    rewrapped = registry.wrap(CONDITIONING_TYPE_ID, rebuilt)
    assert rewrapped.fingerprint == original.fingerprint
    assert spec.encode(rewrapped.resolve()) == original_bytes


def test_constructor_refuses_unbound_unknown_conflicting_and_mismatched_bindings() -> None:
    source = _complete_source()
    bindings = _bindings()
    _assert_code("unbound-reference", lambda: make_conditioning_carrier(source, bindings[:-1]))
    _assert_code(
        "unknown-binding",
        lambda: make_conditioning_carrier(
            source, bindings + (PayloadBinding("other", (1,), "U8", "x", b"x"),)
        ),
    )
    _assert_code(
        "conflicting-binding",
        lambda: make_conditioning_carrier(
            source,
            bindings + (PayloadBinding("source-mask", (2, 2), "U8", "mask", b"MASK"),),
        ),
    )
    _assert_code(
        "descriptor-mismatch",
        lambda: make_conditioning_carrier(
            source,
            (
                PayloadBinding("source-text", (2, 1), "F32", "text", bytes(range(8))),
                *bindings[1:],
            ),
        ),
    )
    _assert_code(
        "payload-byte-length",
        lambda: PayloadBinding("bad", (2,), "F32", "x", b"short"),
    )


def test_identical_payload_content_deduplicates_across_distinct_references() -> None:
    left = _payload("left", (1,), "F32", "text")
    right = _payload("right", (1,), "F32", "text")
    source = ConditioningSet(
        (
            ConditioningRecord(((ConditioningChannel.TEXT, left),)),
            ConditioningRecord(((ConditioningChannel.TEXT, right),)),
        )
    )
    carrier = make_conditioning_carrier(
        source,
        (
            PayloadBinding("left", (1,), "F32", "text", b"same"),
            PayloadBinding("right", (1,), "F32", "text", b"same"),
        ),
    )
    assert len(carrier.bindings) == 1
    assert carrier.conditioning.records[0].channels[0][1].reference == (
        carrier.conditioning.records[1].channels[0][1].reference
    )


def test_opaque_relay_and_cache_persistence_reuse_canonical_bytes_without_decode(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        encoded = encode_conditioning_carrier(_carrier())
        fingerprint = stable_hash([encoded])
        opaque = Value(
            type_id=CONDITIONING_TYPE_ID,
            fingerprint=fingerprint,
            meta=ValueMeta({"format": CONDITIONING_CARRIER_FORMAT}),
            payload=EncodedPayload(CONDITIONING_TYPE_ID, encoded, None),
        )
        registry = TypeRegistry()
        codec = ValueCodec(registry, use_shm=False)
        blobs: list[bytes] = []
        wire, _ = codec.encode(opaque, blobs, [])
        relayed, _ = codec.decode(wire, blobs, [])
        assert relayed.fingerprint == fingerprint
        assert isinstance(relayed.payload, EncodedPayload)
        assert relayed.payload.data == encoded

        cache = DiskCacheStore(tmp_path, registry)
        await cache.put("opaque-conditioning", {"out": relayed})
        hit = await cache.get("opaque-conditioning")
        assert hit is not None
        persisted = hit["out"]
        assert persisted.fingerprint == fingerprint
        assert isinstance(persisted.payload, EncodedPayload)
        assert persisted.payload.data == encoded

    asyncio.run(scenario())


def test_metadata_format_mismatch_refuses() -> None:
    encoded = encode_conditioning_carrier(_carrier())
    _assert_code(
        "metadata-format-mismatch",
        lambda: decode_conditioning_carrier(encoded, {"format": "other"}),
    )


def test_boundary_refuses_malformed_carrier_before_trusting_fingerprint() -> None:
    registry = TypeRegistry()
    register_conditioning_type(registry)
    malformed = bytearray(encode_conditioning_carrier(_carrier()))
    malformed[4] = 2
    value = Value(
        type_id=CONDITIONING_TYPE_ID,
        fingerprint="untrusted-fingerprint",
        meta=ValueMeta({"format": CONDITIONING_CARRIER_FORMAT}),
        payload=EncodedPayload(CONDITIONING_TYPE_ID, bytes(malformed), None),
    )
    codec = ValueCodec(registry, use_shm=False)
    blobs: list[bytes] = []
    wire, _ = codec.encode(value, blobs, [])
    with pytest.raises(EncodedPayloadValidationError) as caught:
        codec.decode(wire, blobs, [])
    assert isinstance(caught.value.__cause__, ConditioningWireError)
    assert caught.value.__cause__.code == "unknown-version"


def test_malformed_cached_carrier_is_a_counted_miss(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_conditioning_type(registry)
        malformed = encode_conditioning_carrier(_carrier())[:-1]
        value = Value(
            type_id=CONDITIONING_TYPE_ID,
            fingerprint="untrusted-fingerprint",
            meta=ValueMeta({"format": CONDITIONING_CARRIER_FORMAT}),
            payload=EncodedPayload(CONDITIONING_TYPE_ID, malformed, None),
        )
        cache = DiskCacheStore(tmp_path, registry)
        await cache.put("malformed-conditioning", {"out": value})
        assert await cache.get("malformed-conditioning") is None
        assert cache.hits == 0
        assert cache.misses == 1

    asyncio.run(scenario())


def test_wrong_magic_refuses() -> None:
    encoded = encode_conditioning_carrier(_carrier())
    _assert_code("wrong-magic", lambda: decode_conditioning_carrier(b"NOPE" + encoded[4:]))


def test_unknown_version_refuses_deterministically() -> None:
    encoded = bytearray(encode_conditioning_carrier(_carrier()))
    encoded[4] = 2
    for _ in range(2):
        _assert_code("unknown-version", lambda: decode_conditioning_carrier(bytes(encoded)))


def test_header_length_exceeding_remaining_bytes_refuses() -> None:
    encoded = b"DMFC\x01" + struct.pack("<Q", 10) + b"{}"
    _assert_code("truncated-header", lambda: decode_conditioning_carrier(encoded))


def test_header_over_64_mib_refuses_before_reading_body() -> None:
    encoded = b"DMFC\x01" + struct.pack("<Q", 64 * 1024 * 1024 + 1)
    _assert_code("header-too-large", lambda: decode_conditioning_carrier(encoded))


def test_noncanonical_header_json_refuses() -> None:
    encoded = encode_conditioning_carrier(_carrier())
    header, payloads = _split(encoded)
    _assert_code(
        "noncanonical-header",
        lambda: decode_conditioning_carrier(_frame(header, payloads, canonical=False)),
    )


def test_malformed_header_json_refuses() -> None:
    encoded = b"DMFC\x01" + struct.pack("<Q", 1) + b"{"
    _assert_code("malformed-header-json", lambda: decode_conditioning_carrier(encoded))


@pytest.mark.parametrize(
    "header_bytes",
    (
        b'{"conditioning":1e400,"format":"dinkster-conditioning-carrier-v1","payload_manifest":[]}',
        b'{"conditioning":{},"format":"dinkster-conditioning-carrier-v1",'
        b'"payload_manifest":[{"byte_length":0,"content_id":"pcid:'
        + b"0" * 40
        + b'","dtype":"U8","shape":[0],"space":"\\ud800"}]}',
    ),
)
def test_malformed_json_values_refuse_deterministically(header_bytes: bytes) -> None:
    encoded = b"DMFC\x01" + struct.pack("<Q", len(header_bytes)) + header_bytes
    for _ in range(2):
        _assert_code("malformed-header-json", lambda: decode_conditioning_carrier(encoded))


def test_unknown_format_tag_refuses() -> None:
    header, payloads = _split(encode_conditioning_carrier(_carrier()))
    header["format"] = "other"
    _assert_code(
        "unknown-format-tag", lambda: decode_conditioning_carrier(_frame(header, payloads))
    )


def test_manifest_segment_count_over_4096_refuses() -> None:
    header, _ = _split(encode_conditioning_carrier(_carrier()))
    manifest = header["payload_manifest"]
    assert isinstance(manifest, list)
    header["payload_manifest"] = [manifest[0]] * 4097
    _assert_code("payload-count-limit", lambda: decode_conditioning_carrier(_frame(header)))


def test_unsorted_content_ids_refuse() -> None:
    header, payloads = _split(encode_conditioning_carrier(_carrier()))
    manifest = header["payload_manifest"]
    assert isinstance(manifest, list)
    lengths = [entry["byte_length"] for entry in manifest]
    segments: list[bytes] = []
    offset = 0
    for length in lengths:
        assert isinstance(length, int)
        segments.append(payloads[offset : offset + length])
        offset += length
    header["payload_manifest"] = list(reversed(manifest))
    _assert_code(
        "unsorted-content-ids",
        lambda: decode_conditioning_carrier(_frame(header, b"".join(reversed(segments)))),
    )


def test_duplicate_content_ids_refuse() -> None:
    header, payloads = _split(encode_conditioning_carrier(_carrier()))
    manifest = header["payload_manifest"]
    assert isinstance(manifest, list)
    first = manifest[0]
    length = first["byte_length"]
    assert isinstance(length, int)
    header["payload_manifest"] = [first, first]
    _assert_code(
        "duplicate-content-id",
        lambda: decode_conditioning_carrier(_frame(header, payloads[:length] * 2)),
    )


def test_content_id_not_matching_segment_refuses() -> None:
    header, payloads = _split(encode_conditioning_carrier(_carrier()))
    manifest = header["payload_manifest"]
    assert isinstance(manifest, list)
    manifest[0]["content_id"] = "pcid:" + "0" * 40
    _assert_code(
        "content-id-mismatch", lambda: decode_conditioning_carrier(_frame(header, payloads))
    )


def test_segment_length_mismatch_refuses() -> None:
    header, payloads = _split(encode_conditioning_carrier(_carrier()))
    manifest = header["payload_manifest"]
    assert isinstance(manifest, list)
    manifest[0]["byte_length"] += 1
    _assert_code(
        "segment-length-mismatch", lambda: decode_conditioning_carrier(_frame(header, payloads))
    )


def test_truncated_segment_refuses_deterministically() -> None:
    encoded = encode_conditioning_carrier(_carrier())[:-1]
    for _ in range(2):
        _assert_code("segment-length-mismatch", lambda: decode_conditioning_carrier(encoded))


def test_trailing_bytes_refuse() -> None:
    encoded = encode_conditioning_carrier(_carrier()) + b"trailer"
    _assert_code("trailing-bytes", lambda: decode_conditioning_carrier(encoded))


def test_unknown_dtype_refuses() -> None:
    header, payloads = _split(encode_conditioning_carrier(_carrier()))
    manifest = header["payload_manifest"]
    assert isinstance(manifest, list)
    manifest[0]["dtype"] = "F8"
    _assert_code("unknown-dtype", lambda: decode_conditioning_carrier(_frame(header, payloads)))


def test_carrier_module_is_torch_free() -> None:
    import dinkster_inference.conditioning_wire as module

    assert "import torch" not in inspect.getsource(module)
