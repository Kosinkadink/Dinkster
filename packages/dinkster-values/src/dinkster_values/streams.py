"""Replayable bounded media ranges with shared source ownership."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import suppress
from copy import copy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .model import PyObjPayload, Value, ValueMeta, stable_hash


def stream_type_id(element: str) -> str:
    return f"stream<{element}>"


def parse_stream_type_id(type_id: str) -> str | None:
    if type_id.startswith("stream<") and type_id.endswith(">") and len(type_id) > 8:
        return type_id[7:-1]
    return None


@dataclass(frozen=True)
class MediaChunk:
    start: int
    stop: int
    total: int
    value: object

    def __post_init__(self) -> None:
        if any(type(position) is not int for position in (self.start, self.stop, self.total)):
            raise ValueError("chunk ranges must be integers")
        if not 0 <= self.start < self.stop <= self.total:
            raise ValueError("chunk range must be nonempty and within the declared total")


@dataclass
class _StreamSource:
    read: Callable[[int, int], object] | None
    close: Callable[[], None] | None
    owners: int = 1
    lock: Any = field(default_factory=threading.RLock)
    read_lock: Any = field(default_factory=threading.RLock)
    callbacks: list[Callable[[], None]] = field(default_factory=list[Callable[[], None]])
    released: bool = False


def _release_source(source: _StreamSource) -> None:
    with source.lock:
        source.owners -= 1
        if source.owners:
            return
    errors: list[BaseException] = []
    try:
        with source.read_lock:
            source.read = None
            close, source.close = source.close, None
            if close is not None:
                close()
    except BaseException as exc:
        errors.append(exc)
    with source.lock:
        source.released = True
        callbacks, source.callbacks = source.callbacks, []
    for callback in callbacks:
        try:
            callback()
        except BaseException as exc:
            errors.append(exc)
    if errors:
        raise errors[0]


class MediaStream:
    """A lease over exact, replayable half-open frame or sample ranges.

    Reads and final source cleanup are serialized. Retained streams own
    independent leases; the source and close callbacks run after the final
    lease closes.
    """

    def __init__(
        self,
        element_type: str,
        total: int,
        chunk_size: int,
        read_range: Callable[[int, int], object],
        *,
        identity: str,
        sample_rate: int | None = None,
        asset_refs: Sequence[Mapping[str, object]] = (),
        close: Callable[[], None] | None = None,
    ) -> None:
        if type(total) is not int or total < 0:
            raise ValueError("stream total must be a nonnegative integer")
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("stream chunk_size must be a positive integer")
        if (
            type(element_type) is not str
            or not element_type
            or "<" in element_type
            or ">" in element_type
            or type(identity) is not str
            or not identity
        ):
            raise ValueError("stream requires an atom type and source identity")
        if not callable(read_range) or (close is not None and not callable(close)):
            raise ValueError("stream source operations must be callable")
        if sample_rate is not None and (type(sample_rate) is not int or sample_rate < 1):
            raise ValueError("sample_rate must be a positive integer")
        self._element_type = element_type
        self._total = total
        self._chunk_size = chunk_size
        self._identity = identity
        self._sample_rate = sample_rate
        self._asset_refs = tuple(MappingProxyType(dict(ref)) for ref in asset_refs)
        self._source = _StreamSource(read_range, close)
        self._closed = False

    @property
    def type_id(self) -> str:
        return self._element_type

    @property
    def total(self) -> int:
        return self._total

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def sample_rate(self) -> int | None:
        return self._sample_rate

    @property
    def asset_refs(self) -> tuple[Mapping[str, object], ...]:
        return self._asset_refs

    @property
    def closed(self) -> bool:
        return self._closed

    def retain(self, *, element_type: str | None = None) -> MediaStream:
        """Acquire an independent consumer lease.

        ``element_type`` stamps a type equivalence already checked by the
        caller's registry.
        """
        retained_type = self.type_id if element_type is None else element_type
        if (
            type(retained_type) is not str
            or not retained_type
            or "<" in retained_type
            or ">" in retained_type
        ):
            raise ValueError("retained stream requires an atom type")
        with self._source.lock:
            if self._closed:
                raise ValueError("stream is closed")
            retained = copy(self)
            retained._element_type = retained_type
            self._source.owners += 1
            retained._closed = False
            return retained

    def read(self, start: int, stop: int) -> MediaChunk:
        MediaChunk(start, stop, self.total, None)
        if stop - start > self.chunk_size:
            raise ValueError("range exceeds stream chunk_size")
        with self._source.read_lock:
            with self._source.lock:
                if self._closed:
                    raise ValueError("stream is closed")
            read = self._source.read
            if read is None:
                raise ValueError("stream is closed")
            return MediaChunk(start, stop, self.total, read(start, stop))

    def iter_chunks(self, start: int = 0, stop: int | None = None) -> Iterator[MediaChunk]:
        end = self.total if stop is None else stop
        if type(start) is not int or type(end) is not int or not 0 <= start <= end <= self.total:
            raise ValueError("iteration range is outside the declared total")
        for offset in range(start, end, self.chunk_size):
            yield self.read(offset, min(offset + self.chunk_size, end))

    def on_close(self, callback: Callable[[], None]) -> None:
        """Run a callback after the final retained consumer releases the source."""
        with self._source.lock:
            if not self._source.released:
                self._source.callbacks.append(callback)
                return
        callback()

    def close(self) -> None:
        with self._source.lock:
            if self._closed:
                return
            self._closed = True
        _release_source(self._source)

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


def make_stream_value(stream: MediaStream, element_type: str | None = None) -> Value:
    value_stream = stream if element_type is None else stream.retain(element_type=element_type)
    try:
        metadata: dict[str, object] = {
            "total": value_stream.total,
            "chunk_size": value_stream.chunk_size,
            "sample_rate": value_stream.sample_rate,
            "identity": value_stream.identity,
        }
        if value_stream.asset_refs:
            metadata["asset_refs"] = [dict(ref) for ref in value_stream.asset_refs]
        return Value(
            type_id=stream_type_id(value_stream.type_id),
            fingerprint=stable_hash(
                [
                    value_stream.type_id.encode(),
                    value_stream.identity.encode(),
                    str(value_stream.total).encode(),
                    str(value_stream.sample_rate).encode(),
                ]
            ),
            meta=ValueMeta(metadata),
            payload=PyObjPayload(value_stream),
        )
    except BaseException:
        if value_stream is not stream:
            value_stream.close()
        raise
