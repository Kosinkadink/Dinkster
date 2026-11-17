"""Asyncio streams over indexed inherited pipe frames, without local sockets."""

from __future__ import annotations

import asyncio
import contextlib
import os
import struct
import threading
from collections import deque
from collections.abc import Callable

_HEADER = struct.Struct("!II")
_CHUNK = 64 * 1024


class _PipeStream(asyncio.Transport):
    def __init__(self, owner: PipeStreams, index: int) -> None:
        super().__init__()
        self.owner, self.index = owner, index
        self.reader = asyncio.StreamReader()
        self.protocol = asyncio.StreamReaderProtocol(self.reader)
        self.protocol.connection_made(self)
        self.writer = asyncio.StreamWriter(self, self.protocol, self.reader, owner.loop)
        self.readable = threading.Event()
        self.readable.set()
        self.closed = False
        self.finished = False
        self.size = 0
        self.paused = False
        self.high, self.low = _CHUNK * 2, _CHUNK

    def is_closing(self) -> bool:
        return self.closed

    def get_write_buffer_size(self) -> int:
        return self.size

    def get_write_buffer_limits(self) -> tuple[int, int]:
        return self.low, self.high

    def set_write_buffer_limits(self, high: int | None = None, low: int | None = None) -> None:
        high = _CHUNK * 2 if high is None else high
        low = high // 4 if low is None else low
        if not 0 <= low <= high:
            raise ValueError("invalid pipe write buffer limits")
        self.high, self.low = high, low

    def write(self, data: bytes | bytearray | memoryview) -> None:
        if self.closed:
            raise ConnectionError("worker pipe is closed")
        for offset in range(0, len(data), _CHUNK):
            chunk = bytes(data[offset : offset + _CHUNK])
            self.size += len(chunk)
            self.owner.send(self, chunk)
        if self.size > self.high and not self.paused:
            self.paused = True
            self.protocol.pause_writing()

    def sent(self, size: int) -> None:
        self.size -= size
        if self.paused and self.size <= self.low:
            self.paused = False
            self.protocol.resume_writing()

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.readable.set()
            self.owner.send(self, b"")

    def abort(self) -> None:
        self.close()

    def finish(self, error: Exception | None = None) -> None:
        if not self.finished:
            self.finished = self.closed = True
            self.readable.set()
            self.protocol.connection_lost(error)

    def pause_reading(self) -> None:
        self.readable.clear()

    def resume_reading(self) -> None:
        self.readable.set()

    def is_reading(self) -> bool:
        return not self.closed and self.readable.is_set()


class PipeStreams:
    """One bounded reader pump and one ordered writer pump for all members."""

    def __init__(self, source: int, destination: int, count: int) -> None:
        self.loop = asyncio.get_running_loop()
        self.source, self.destination = source, destination
        self.pending: deque[tuple[_PipeStream, bytes]] = deque()
        self.active: _PipeStream | None = None
        self.condition = threading.Condition()
        self.stopped = False
        self.streams: list[_PipeStream] = [_PipeStream(self, index) for index in range(count)]
        self.available = dict(enumerate(self.streams))
        threading.Thread(target=self.receive, daemon=True).start()
        threading.Thread(target=self.transmit, daemon=True).start()

    def connect(self, index: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        stream = self.available.pop(index)
        return stream.reader, stream.writer

    def send(self, stream: _PipeStream, data: bytes) -> None:
        with self.condition:
            if self.stopped:
                raise ConnectionError("worker pipe disconnected")
            self.pending.append((stream, data))
            self.condition.notify()

    def dispatch(self, callback: Callable[[], None]) -> bool:
        try:
            self.loop.call_soon_threadsafe(callback)
        except RuntimeError:
            return False
        return True

    def fail(self, error: Exception | None) -> None:
        with self.condition:
            self.stopped = True
            abandoned = {stream for stream, data in self.pending if data}
            active = self.active
            self.pending.clear()
            self.condition.notify()
            for stream in self.streams:
                stream.readable.set()
                failure = error
                if failure is None and stream in abandoned:
                    failure = ConnectionResetError("worker pipe disconnected with pending output")
                if failure is None and stream is active:
                    # Only the writer can settle an in-flight OS write. Its loop-side
                    # flow-control notification may lag behind actual completion.
                    continue
                self.dispatch(lambda stream=stream, failure=failure: stream.finish(failure))

    def transmit(self) -> None:
        remaining = len(self.streams)
        try:
            while remaining:
                with self.condition:
                    self.condition.wait_for(lambda: self.stopped or self.pending)
                    if self.stopped:
                        return
                    stream, data = self.pending.popleft()
                    self.active = stream
                frame = memoryview(_HEADER.pack(stream.index, len(data)) + data)
                while frame:
                    written = os.write(self.destination, frame)
                    if written == 0:
                        raise BrokenPipeError("worker pipe closed")
                    frame = frame[written:]
                with self.condition:
                    self.active = None
                    if data:
                        size = len(data)
                        self.dispatch(lambda stream=stream, size=size: stream.sent(size))
                    else:
                        remaining -= 1
                        self.dispatch(stream.finish)
                    if self.stopped:
                        self.dispatch(stream.finish)
                        return
        except OSError as exc:
            self.fail(exc)
        finally:
            os.close(self.destination)

    def read_exactly(self, size: int) -> bytes:
        data = bytearray()
        while len(data) < size:
            chunk = os.read(self.source, size - len(data))
            if not chunk:
                raise EOFError
            data.extend(chunk)
        return bytes(data)

    def receive(self) -> None:
        try:
            while True:
                raw_index, raw_size = _HEADER.unpack(self.read_exactly(_HEADER.size))
                index, size = int(raw_index), int(raw_size)
                if index >= len(self.streams) or size > _CHUNK:
                    raise ValueError("invalid worker pipe frame")
                stream: _PipeStream = self.streams[index]
                data = self.read_exactly(size)
                stream.readable.wait()
                delivered = threading.Event()

                def deliver(
                    stream: _PipeStream = stream,
                    data: bytes = data,
                    delivered: threading.Event = delivered,
                ) -> None:
                    try:
                        if not stream.closed:
                            if data:
                                stream.reader.feed_data(data)
                            else:
                                stream.reader.feed_eof()
                    finally:
                        delivered.set()

                if not self.dispatch(deliver):
                    return
                # At most one chunk is scheduled ahead of reader backpressure.
                delivered.wait()
                if self.stopped:
                    return
        except EOFError:
            self.fail(None)
        except (OSError, ValueError) as exc:
            self.fail(exc)
        finally:
            with contextlib.suppress(OSError):
                os.close(self.source)
