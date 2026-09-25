"""One-carrier payload residency and transport contracts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest
from dinkster_caches import DiskCacheStore
from dinkster_inference import (
    CONDITIONING_CARRIER_FORMAT,
    CONDITIONING_TYPE_ID,
    RESIDENT_CONDITIONING_FORMAT,
    ConditioningCarrier,
    ConditioningChannel,
    ConditioningRecord,
    ConditioningSet,
    LivePayloadBinding,
    PayloadDescriptor,
    PayloadReference,
    ResidentPayloadBinding,
    TokenLayoutDescriptor,
    TokenSegmentDescriptor,
    make_conditioning_carrier,
    register_conditioning_type,
)
from dinkster_values import RESOURCE_ID_META_KEY, ResidencyTable, ResidentLookupError, TypeRegistry
from dinkster_workers import ValueCodec


def _conditioning() -> ConditioningSet:
    payload = PayloadDescriptor(PayloadReference("text"), (1, 2), "F32", "text")
    layout = TokenLayoutDescriptor(
        "dinkster.test",
        1,
        ("text",),
        (TokenSegmentDescriptor("prompt", "text", 0, 2),),
    )
    return ConditioningSet(
        (
            ConditioningRecord(
                channels=((ConditioningChannel.TEXT, payload),),
                token_layout=layout,
            ),
        )
    )


def test_deferred_payload_materializes_at_transport_boundary(tmp_path: Path) -> None:
    payload = bytes(range(8))
    materialized: list[object] = []

    def materialize(value: object) -> bytes:
        assert isinstance(value, bytes)
        materialized.append(value)
        return value

    carrier = make_conditioning_carrier(
        _conditioning(),
        (
            LivePayloadBinding(
                "text",
                (1, 2),
                "F32",
                "text",
                payload,
                "producer:text-v1",
                materialize,
            ),
        ),
    )
    registry = TypeRegistry()
    spec = register_conditioning_type(registry)

    value = registry.wrap(CONDITIONING_TYPE_ID, carrier)
    assert materialized == []
    assert value.resolve() is carrier
    assert value.meta.get("format") == CONDITIONING_CARRIER_FORMAT

    codec = ValueCodec(registry, use_shm=False)
    blobs: list[bytes] = []
    wire, _ = codec.encode(value, blobs, [])
    assert materialized == [payload]
    transported, _ = codec.decode(wire, blobs, [])
    transported_carrier = transported.resolve()
    assert isinstance(transported_carrier, ConditioningCarrier)
    assert all(binding.kind == "materialized" for binding in transported_carrier.bindings)

    async def persist() -> None:
        cache = DiskCacheStore(tmp_path, registry)
        await cache.put("conditioning", {"value": value})
        hit = await DiskCacheStore(tmp_path, registry).get("conditioning")
        assert hit is not None
        persisted = hit["value"].resolve()
        assert isinstance(persisted, ConditioningCarrier)
        assert all(binding.kind == "materialized" for binding in persisted.bindings)

    asyncio.run(persist())
    assert spec.encode(carrier).startswith(b"DMFC\x01")


def test_resident_payload_uses_a_stub_inside_the_one_carrier() -> None:
    @dataclass(frozen=True)
    class Payload:
        owner: object

        @property
        def _dinkster_resident_owner(self) -> object:
            return self.owner

    table = ResidencyTable()
    registry = TypeRegistry()
    spec = register_conditioning_type(registry, resident_table=table)
    owner = object()
    carrier = make_conditioning_carrier(
        _conditioning(),
        (ResidentPayloadBinding("text", (1, 2), "F32", "text", Payload(owner), "producer:text-a"),),
    )

    value = registry.wrap(CONDITIONING_TYPE_ID, carrier)
    encoded = spec.encode(value.resolve())

    assert value.resolve() is carrier
    assert encoded.startswith(b"DMFR\x01")
    assert value.meta.get("format") == RESIDENT_CONDITIONING_FORMAT
    assert value.meta.get(RESOURCE_ID_META_KEY) == "resident:" + table.rid_for(owner)
    assert spec.decode(encoded) is carrier

    other = TypeRegistry()
    other_spec = register_conditioning_type(other, resident_table=ResidencyTable())
    with pytest.raises(ResidentLookupError, match="another worker"):
        other_spec.decode(encoded)


def test_resident_fingerprint_includes_stable_producer_inputs() -> None:
    @dataclass(frozen=True)
    class Payload:
        owner: object
        value: tuple[int, ...]

        @property
        def _dinkster_resident_owner(self) -> object:
            return self.owner

    registry = TypeRegistry()
    register_conditioning_type(registry, resident_table=ResidencyTable())
    owner = object()

    def fingerprint(producer_fingerprint: str) -> str:
        carrier = make_conditioning_carrier(
            _conditioning(),
            (
                ResidentPayloadBinding(
                    "text",
                    (1, 2),
                    "F32",
                    "text",
                    Payload(owner, (1, 2)),
                    producer_fingerprint,
                ),
            ),
        )
        return registry.wrap(CONDITIONING_TYPE_ID, carrier).fingerprint

    assert fingerprint("producer:bytes-a") != fingerprint("producer:bytes-b")
