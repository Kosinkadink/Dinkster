from __future__ import annotations

import gc
import threading
import weakref
from typing import cast

import pytest
from dinkster_values import MediaChunk, MediaStream, TypeRegistry


def stream(
    *,
    total: int = 7,
    chunk_size: int = 3,
    reads: list[tuple[int, int]] | None = None,
    closed: list[str] | None = None,
) -> MediaStream:
    return MediaStream(
        "comfy.IMAGE",
        total,
        chunk_size,
        lambda start, stop: (
            (reads.append((start, stop)) if reads is not None else None)
            or tuple(range(start, stop))
        ),
        identity="source-digest",
        close=None if closed is None else lambda: closed.append("source"),
    )


@pytest.mark.parametrize(
    ("start", "stop", "message"),
    [
        (-1, 1, "within"),
        (0, 0, "nonempty"),
        (2, 1, "nonempty"),
        (0, 8, "within"),
        (0, 4, "chunk_size"),
        (False, 1, "integers"),
    ],
)
def test_invalid_ranges_refuse_before_source_read(start: int, stop: int, message: str) -> None:
    reads: list[tuple[int, int]] = []
    value = stream(reads=reads)
    try:
        with pytest.raises(ValueError, match=message):
            value.read(start, stop)
        assert reads == []
    finally:
        value.close()


@pytest.mark.parametrize(
    ("total", "chunk_size", "message"),
    [(-1, 1, "total"), (True, 1, "total"), (1, 0, "chunk_size"), (1, False, "chunk_size")],
)
def test_constructor_rejects_invalid_bounds(total: int, chunk_size: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        stream(total=total, chunk_size=chunk_size)


def test_invalid_iteration_range_refuses_before_source_read() -> None:
    reads: list[tuple[int, int]] = []
    value = stream(reads=reads)
    try:
        chunks = value.iter_chunks(3, 2)
        with pytest.raises(ValueError, match="outside"):
            next(chunks)
        assert reads == []
    finally:
        value.close()


@pytest.mark.parametrize("chunk_size", range(1, 8))
def test_iteration_is_ordered_exact_and_demand_driven(chunk_size: int) -> None:
    reads: list[tuple[int, int]] = []
    value = stream(chunk_size=chunk_size, reads=reads)
    try:
        chunks = value.iter_chunks()
        assert reads == []
        result = list(chunks)
        assert [(chunk.start, chunk.stop) for chunk in result] == reads
        assert tuple(
            item for chunk in result for item in cast("tuple[int, ...]", chunk.value)
        ) == tuple(range(7))
    finally:
        value.close()


def test_retain_closes_source_and_callbacks_after_final_lease() -> None:
    events: list[str] = []
    value = stream(closed=events)
    retained = value.retain()
    value.on_close(lambda: events.append("callback"))
    value.close()
    value.close()
    assert events == []
    assert retained.read(0, 1).value == (0,)
    retained.close()
    retained.close()
    assert events == ["source", "callback"]
    value.on_close(lambda: events.append("late"))
    assert events == ["source", "callback", "late"]
    with pytest.raises(ValueError, match="closed"):
        value.retain()


def test_collection_closes_source_when_callback_captures_stream() -> None:
    events: list[str] = []

    def make_cyclic_stream() -> weakref.ReferenceType[MediaStream]:
        value: MediaStream

        def read(_start: int, _stop: int) -> str:
            return value.identity

        def close() -> None:
            events.extend(("source", value.identity))

        value = MediaStream("comfy.IMAGE", 1, 1, read, identity="source-digest", close=close)
        value.on_close(lambda: events.append(value.identity))
        return weakref.ref(value)

    value_ref = make_cyclic_stream()
    gc.collect()
    assert value_ref() is None
    assert events == ["source", "source-digest", "source-digest"]


def test_final_close_waits_for_an_active_read() -> None:
    reading = threading.Event()
    release = threading.Event()
    events: list[str] = []

    def read(start: int, stop: int) -> tuple[int, ...]:
        reading.set()
        assert release.wait(timeout=5)
        return tuple(range(start, stop))

    value = MediaStream(
        "comfy.IMAGE", 1, 1, read, identity="serialized", close=lambda: events.append("close")
    )
    reader = threading.Thread(target=lambda: value.read(0, 1))
    reader.start()
    assert reading.wait(timeout=5)
    closer = threading.Thread(target=value.close)
    closer.start()
    assert events == []
    release.set()
    reader.join(timeout=5)
    closer.join(timeout=5)
    assert not reader.is_alive() and not closer.is_alive()
    assert events == ["close"]


def test_cleanup_runs_every_callback_when_source_close_fails() -> None:
    events: list[str] = []

    def fail() -> None:
        events.append("source")
        raise RuntimeError("source failed")

    value = MediaStream("comfy.IMAGE", 1, 1, lambda _a, _b: (), identity="errors", close=fail)
    value.on_close(lambda: events.append("first"))
    value.on_close(lambda: events.append("second"))
    with pytest.raises(RuntimeError, match="source failed"):
        value.close()
    assert events == ["source", "first", "second"]


def test_registry_wraps_matching_stream_with_stable_metadata() -> None:
    registry = TypeRegistry()
    registry.register("comfy.IMAGE")
    ref: dict[str, object] = {"digest": "sha256:" + "a" * 64, "size": 12}
    first = MediaStream(
        "comfy.IMAGE",
        7,
        3,
        lambda start, stop: tuple(range(start, stop)),
        identity="source-digest",
        asset_refs=[ref],
    )
    second = MediaStream(
        "comfy.IMAGE",
        7,
        5,
        lambda start, stop: tuple(range(start, stop)),
        identity="source-digest",
        asset_refs=[dict(ref)],
    )
    try:
        wrapped = registry.wrap("stream<comfy.IMAGE>", first)
        same_content = registry.wrap("stream<comfy.IMAGE>", second)
        ref["size"] = 99
        assert wrapped.type_id == "stream<comfy.IMAGE>"
        assert wrapped.resolve() is first
        assert wrapped.fingerprint == same_content.fingerprint
        assert wrapped.meta.entries == {
            "total": 7,
            "chunk_size": 3,
            "sample_rate": None,
            "identity": "source-digest",
            "asset_refs": [{"digest": "sha256:" + "a" * 64, "size": 12}],
        }
        with pytest.raises(AttributeError):
            first.identity = "replacement"  # type: ignore[misc]
        with pytest.raises(TypeError, match="does not accept"):
            registry.wrap("comfy.IMAGE", first)
        with pytest.raises(TypeError, match="matching MediaStream"):
            registry.wrap("stream<comfy.IMAGE>", object())
        with pytest.raises(KeyError, match="unregistered"):
            registry.wrap("stream<comfy.AUDIO>", first)
    finally:
        first.close()
        second.close()


def test_registry_retypes_equivalent_stream_with_an_independent_lease() -> None:
    events: list[str] = []
    registry = TypeRegistry()
    registry.register("comfy.IMAGE")
    registry.register("dinkster.image")
    registry.register_type_equivalence(
        "comfy.IMAGE", "dinkster.image", provider_id="test.image-equivalence@1"
    )
    original = stream(closed=events)
    source_fingerprint = registry.wrap("stream<comfy.IMAGE>", original).fingerprint
    wrapped = registry.wrap("stream<dinkster.image>", original)
    retained = wrapped.resolve()
    assert isinstance(retained, MediaStream)
    assert retained is not original
    assert retained.type_id == "dinkster.image"
    assert wrapped.fingerprint != source_fingerprint
    original.close()
    assert events == []
    retained.close()
    assert events == ["source"]


def test_registry_refuses_different_registered_stream_type() -> None:
    registry = TypeRegistry()
    registry.register("comfy.IMAGE")
    registry.register("comfy.AUDIO")
    value = stream()
    try:
        with pytest.raises(TypeError, match="matching MediaStream"):
            registry.wrap("stream<comfy.AUDIO>", value)
    finally:
        value.close()


def test_registry_refuses_closed_stream() -> None:
    registry = TypeRegistry()
    registry.register("comfy.IMAGE")
    value = stream()
    value.close()
    with pytest.raises(ValueError, match="closed"):
        registry.wrap("stream<comfy.IMAGE>", value)


def test_media_chunk_rejects_noncanonical_ranges() -> None:
    with pytest.raises(ValueError, match="integers"):
        MediaChunk(0, 1, True, object())
