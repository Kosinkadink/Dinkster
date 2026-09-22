"""Byte carriers charge retained codec bytes, not the content they describe."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path
from typing import cast

import dinkster_api.v1 as api
import pytest
from dinkster_caches import DiskCacheStore, MemoryLRUCache
from dinkster_caches.memory import entry_cost
from dinkster_values import (
    COST_META_KEY,
    RESOURCE_ID_META_KEY,
    RESOURCE_REFS_META_KEY,
    TypeRegistry,
    Value,
    ValueMeta,
    byte_storage_meta,
    stable_hash,
)
from dinkster_values.model import PyObjPayload
from dinkster_values.storage import encoded_storage_meta
from dinkster_workers import ValueCodec

CANONICAL_JSON = (
    b'{"assets":[{"bytes":1000000000,"id":"asset"}],'
    b'"layers":[{"height":100000,"raster":"AAECAw==","width":100000}],"version":2}'
)


def test_byte_storage_meta_api_is_the_values_helper() -> None:
    assert api.byte_storage_meta is byte_storage_meta


@pytest.mark.parametrize("data", [b"", b"{}", CANONICAL_JSON])
def test_byte_storage_meta_charges_retained_bytes_once(data: bytes) -> None:
    metadata = byte_storage_meta(data)
    assert metadata == {"storage_bytes": len(data), COST_META_KEY: {"ram": len(data)}}
    value = Value("custom.bytes", "identity", ValueMeta(metadata), PyObjPayload(data))
    alias = Value(value.type_id, value.fingerprint, value.meta, value.payload)
    assert value.resolve() is data
    assert entry_cost({"out": value, "alias": alias}, "ram") == len(data)
    assert byte_storage_meta(data) is not metadata
    assert byte_storage_meta(data)[COST_META_KEY] is not metadata[COST_META_KEY]


@pytest.mark.parametrize("size", [0, len(CANONICAL_JSON)])
def test_received_byte_storage_replaces_residency_without_mutating_producer(size: int) -> None:
    metadata = {
        "storage_bytes": 999,
        "semantics": {"version": 2, "alpha": "premultiplied", "layers": ["raster"]},
        RESOURCE_REFS_META_KEY: [{"resource_id": "asset"}],
        COST_META_KEY: {"ram:producer": 123, "vram:producer/cuda:0": 456},
    }
    before = deepcopy(metadata)
    local = encoded_storage_meta(metadata, size)
    assert local == {**metadata, "storage_bytes": size, COST_META_KEY: {"ram": size}}
    assert local is not metadata
    assert local["semantics"] is metadata["semantics"]
    assert local[RESOURCE_REFS_META_KEY] is metadata[RESOURCE_REFS_META_KEY]
    assert metadata == before
    assert "storage_dtype" not in local


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"unknown": {"ram": 55}},
        {COST_META_KEY: {"vram:producer/cuda:0": 123}},
        {
            RESOURCE_ID_META_KEY: "resident",
            "storage_bytes": 999,
            "storage_dtype": "fp16",
            COST_META_KEY: {"vram:producer/cuda:0": 123},
        },
    ],
)
def test_resource_and_untagged_metadata_are_unchanged(metadata: dict[str, object]) -> None:
    before = deepcopy(metadata)
    local = encoded_storage_meta(metadata, 7)
    assert local == metadata == before
    assert local is not metadata
    for key in metadata:
        assert local[key] is metadata[key]


@pytest.mark.parametrize("with_byte_marker", [False, True])
def test_array_storage_dtype_is_unchanged(with_byte_marker: bool) -> None:
    metadata: dict[str, object] = {
        "storage_dtype": "fp16",
        "shape": [2, 3],
        COST_META_KEY: {"vram:cuda:0": 12},
    }
    if with_byte_marker:
        metadata["storage_bytes"] = 999
    before = deepcopy(metadata)
    local = encoded_storage_meta(metadata, 48)
    expected = {**metadata, COST_META_KEY: {"ram": 48}}
    if with_byte_marker:
        expected["storage_bytes"] = 48
    assert local == expected
    assert metadata == before


@pytest.mark.parametrize("boundary", ["codec", "disk"])
@pytest.mark.parametrize("data", [b"", CANONICAL_JSON])
def test_declared_byte_codec_replay_preserves_identity_and_normalizes_metadata(
    tmp_path: Path, boundary: str, data: bytes
) -> None:
    registry = TypeRegistry()
    fingerprint_calls: list[object] = []

    def fingerprint(obj: object) -> str:
        fingerprint_calls.append(obj)
        return stable_hash([b"custom-byte-carrier", cast(bytes, obj)])

    registry.register(
        "custom.bytes",
        encode=lambda obj: cast(bytes, obj),
        decode=lambda encoded: encoded,
        fingerprint=fingerprint,
        meta=lambda obj: byte_storage_meta(cast(bytes, obj)),
    )
    wrapped = registry.wrap("custom.bytes", data)
    assert wrapped.meta.get("storage_bytes") == len(data)
    producer = Value(
        wrapped.type_id,
        wrapped.fingerprint,
        ValueMeta(
            {
                **wrapped.meta.entries,
                "storage_bytes": 999,
                "semantics": {"version": 2, "alpha": "premultiplied"},
                COST_META_KEY: {"ram:producer": 123, "vram:producer/cuda:0": 456},
            }
        ),
        wrapped.payload,
    )
    before = deepcopy(dict(producer.meta.entries))

    async def scenario() -> None:
        if boundary == "codec":
            sender = ValueCodec(registry, shm_threshold=1000000)
            receiver = ValueCodec(registry, shm_threshold=1000000)
            blobs: list[bytes] = []
            wire, sent = sender.encode(producer, blobs, [])
            received, stat = receiver.decode(wire, blobs, [])
            assert sent.declared_codec and stat.declared_codec
            assert sent.size_bytes == stat.size_bytes == len(data)
        else:
            await DiskCacheStore(tmp_path, registry).put("key", {"out": producer})
            replay = await DiskCacheStore(tmp_path, registry).get("key")
            assert replay is not None
            received = replay["out"]
        assert received.type_id == producer.type_id
        assert received.fingerprint == producer.fingerprint
        assert received.resolve() == data
        assert received.meta.entries == {
            **before,
            "storage_bytes": len(data),
            COST_META_KEY: {"ram": len(data)},
        }
        cache = MemoryLRUCache(max_bytes=len(data))
        await cache.put("key", {"out": received})
        assert await cache.get("key") is not None
        assert cache.footprint("ram") == len(data)
        assert cache.footprint("vram:producer/cuda:0") == 0

    asyncio.run(scenario())
    assert fingerprint_calls == [data]
    assert producer.meta.entries == before
    assert producer.payload is wrapped.payload
    assert producer.resolve() is data
