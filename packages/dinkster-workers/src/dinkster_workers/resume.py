"""Process-lifetime transport state for resumable remote invocations."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from dinkster_values import RESUMABLE_EVENT_RETENTION_BYTES, RESUMABLE_RESULT_RETENTION_BYTES

from .boundary import BoundaryError, read_frame, write_frame

DEFAULT_RESUME_GRACE_S = 120.0
MAX_RESUMABLE_INVOCATIONS = 1024
MAX_RETAINED_RESULT_BYTES = RESUMABLE_RESULT_RETENTION_BYTES
MAX_RETAINED_EVENT_BYTES = RESUMABLE_EVENT_RETENTION_BYTES


def _retained_frame_bytes(header: Mapping[str, object], blobs: Sequence[bytes]) -> int:
    try:
        header_bytes = len(
            json.dumps(
                {**header, "blobs": [len(blob) for blob in blobs]},
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
    except (ValueError, TypeError, RecursionError, UnicodeError) as exc:
        raise BoundaryError("frame header is not strict JSON") from exc
    return header_bytes + sum(len(blob) for blob in blobs)


@dataclass(frozen=True, slots=True)
class InvocationKey:
    job_ref: str
    attempt_id: int
    invocation_id: str

    @classmethod
    def from_header(cls, header: Mapping[str, object]) -> InvocationKey:
        job_ref = header.get("jobRef")
        attempt_id = header.get("attemptId")
        invocation_id = header.get("invocationId")
        if type(job_ref) is not str or not job_ref:
            raise BoundaryError("invocation identity jobRef must be a non-empty string")
        if type(attempt_id) is not int or attempt_id < 1:
            raise BoundaryError("invocation identity attemptId must be a positive integer")
        if type(invocation_id) is not str or not invocation_id:
            raise BoundaryError("invocation identity invocationId must be a non-empty string")
        return cls(job_ref, attempt_id, invocation_id)

    def to_wire(self) -> dict[str, object]:
        return {
            "jobRef": self.job_ref,
            "attemptId": self.attempt_id,
            "invocationId": self.invocation_id,
        }


@dataclass(slots=True)
class _Record:
    key: InvocationKey
    state: str = "running"
    event_seq: int = 0
    latest_event: tuple[dict[str, object], list[bytes]] | None = None
    reserve: tuple[dict[str, object], list[bytes]] | None = None
    result: tuple[dict[str, object], list[bytes]] | None = None
    release: tuple[dict[str, object], list[bytes]] | None = None
    result_bytes: int = 0
    event_bytes: int = 0
    had_reservation: bool = False


class ResumableConversation:
    """Keep one daemon conversation alive while its physical stream changes."""

    def __init__(
        self,
        engine_instance_id: str,
        label: str,
        worker_instance: str,
        *,
        grace: float = DEFAULT_RESUME_GRACE_S,
        on_close: Callable[[ResumableConversation], Awaitable[None]] | None = None,
    ) -> None:
        self.engine_instance_id = engine_instance_id
        self.label = label
        self.worker_instance = worker_instance
        self.grace = grace
        self.last_activity = time.monotonic()
        self._on_close = on_close
        self._owner_epoch = 0
        self._incoming: asyncio.Queue[tuple[int, tuple[dict[str, Any], list[bytes]]] | None] = (
            asyncio.Queue()
        )
        self._physical_writer: asyncio.StreamWriter | None = None
        self._pump_task: asyncio.Task[None] | None = None
        self._expiry_task: asyncio.Task[None] | None = None
        self._send_lock = asyncio.Lock()
        self._ready = asyncio.Event()
        self._transport_lost = asyncio.Event()
        self._hello: tuple[dict[str, object], list[bytes]] | None = None
        self._latest_memory_report: tuple[dict[str, object], list[bytes]] | None = None
        self._records: dict[InvocationKey, _Record] = {}
        self._keys_by_invocation: dict[str, InvocationKey] = {}
        self._retained_result_bytes = 0
        self._retained_event_bytes = 0
        self._rebind_ready = False
        self._host_started = False
        self._closed = False

    @property
    def owner_epoch(self) -> int:
        return self._owner_epoch

    @property
    def connected(self) -> bool:
        return self._physical_writer is not None and not self._closed

    @property
    def has_records(self) -> bool:
        return bool(self._records)

    def start_host(self) -> None:
        self._host_started = True
        self._rebind_ready = True
        self._ready.set()

    def transport_loss_event(self) -> asyncio.Event:
        return self._transport_lost

    async def wait_ready(self) -> None:
        await self._ready.wait()
        if self._closed:
            raise ConnectionError("resumable conversation is closed")

    async def attach(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> bool:
        if self._closed:
            raise RuntimeError("resumable conversation is closed")
        resumed = self._host_started
        async with self._send_lock:
            if self._closed:
                raise RuntimeError("resumable conversation is closed")
            self.last_activity = time.monotonic()
            self._owner_epoch += 1
            epoch = self._owner_epoch
            if self._expiry_task is not None:
                self._expiry_task.cancel()
                self._expiry_task = None
            old_writer = self._physical_writer
            old_pump = self._pump_task
            self._transport_lost.set()
            self._transport_lost = asyncio.Event()
            self._ready.clear()
            self._physical_writer = writer
            self._rebind_ready = not resumed
            if old_pump is not None and old_pump is not asyncio.current_task():
                old_pump.cancel()
            if old_writer is not None and old_writer is not writer:
                old_writer.transport.abort()
            await write_frame(
                writer,
                {
                    "type": "resumeAccepted",
                    "workerInstance": self.worker_instance,
                    "ownerEpoch": epoch,
                    "resumed": resumed,
                    "resumeGrace": self.grace,
                },
                [],
            )
            if resumed:
                if self._hello is None:
                    raise RuntimeError("resumable conversation has no cached hello")
                await write_frame(writer, self._hello[0], self._hello[1])
            self._pump_task = asyncio.create_task(self._pump(reader, epoch))
        return resumed

    async def _pump(self, reader: asyncio.StreamReader, epoch: int) -> None:
        try:
            while True:
                frame = await read_frame(reader)
                if frame is None:
                    return
                self.last_activity = time.monotonic()
                if epoch != self._owner_epoch or self._closed:
                    return
                if frame[0].get("type") == "invoke":
                    self.register(frame[0])
                await self._incoming.put((epoch, frame))
        except BoundaryError:
            return
        finally:
            await self.detach(epoch)

    async def detach(self, epoch: int) -> None:
        async with self._send_lock:
            if epoch != self._owner_epoch or self._closed:
                return
            writer = self._physical_writer
            self._physical_writer = None
            self._rebind_ready = False
            self._ready.clear()
            self._transport_lost.set()
            if writer is not None:
                writer.close()
            close_now = not self._records or self.grace == 0
        if close_now:
            await self.close()
            return
        if self._expiry_task is None:
            self._expiry_task = asyncio.create_task(self._expire(epoch))

    async def _expire(self, epoch: int) -> None:
        try:
            await asyncio.sleep(self.grace)
            if epoch == self._owner_epoch and self._physical_writer is None:
                await self.close()
        except asyncio.CancelledError:
            pass

    async def read(self) -> tuple[dict[str, Any], list[bytes]] | None:
        while True:
            item = await self._incoming.get()
            if item is None:
                return None
            epoch, frame = item
            if epoch == self._owner_epoch:
                return frame

    def register(self, header: Mapping[str, object]) -> InvocationKey:
        key = InvocationKey.from_header(header)
        if key in self._records or key.invocation_id in self._keys_by_invocation:
            raise BoundaryError("duplicate resumable invocation identity")
        if len(self._records) >= MAX_RESUMABLE_INVOCATIONS:
            raise BoundaryError("resumable invocation capacity exhausted")
        self._records[key] = _Record(key)
        self._keys_by_invocation[key.invocation_id] = key
        return key

    def key_for_invocation(self, invocation_id: object) -> InvocationKey | None:
        return self._keys_by_invocation.get(str(invocation_id))

    def _record_for_frame(self, header: Mapping[str, object]) -> _Record | None:
        raw_id = header.get("invocationId", header.get("requestId"))
        key = self.key_for_invocation(raw_id)
        return None if key is None else self._records.get(key)

    @staticmethod
    def _with_key(header: Mapping[str, object], key: InvocationKey) -> dict[str, object]:
        return {**header, **key.to_wire()}

    async def send(self, header: dict[str, object], blobs: Sequence[bytes] = ()) -> None:
        if self._closed:
            raise ConnectionError("resumable conversation is closed")
        kind = header.get("type")
        body = dict(header)
        payloads = list(blobs)
        record = self._record_for_frame(body)
        if kind == "hello":
            self._hello = (body, payloads)
        elif kind == "memoryReport":
            self._latest_memory_report = (body, payloads)
        elif record is not None and kind == "invocationEvent":
            record.event_seq += 1
            body = self._with_key(body, record.key)
            body["eventSeq"] = record.event_seq
            if not payloads:
                retained = _retained_frame_bytes(body, payloads)
                replacing = record.event_bytes
                if self._retained_event_bytes - replacing + retained <= MAX_RETAINED_EVENT_BYTES:
                    record.latest_event = (body, payloads)
                    record.event_bytes = retained
                    self._retained_event_bytes += retained - replacing
                else:
                    record.latest_event = None
                    record.event_bytes = 0
                    self._retained_event_bytes -= replacing
        elif record is not None and kind == "memoryReserve":
            body = self._with_key(body, record.key)
            record.had_reservation = True
            record.reserve = (body, payloads)
        elif record is not None and kind == "memoryRelease":
            body = self._with_key(body, record.key)
            record.release = (body, payloads)
        elif record is not None and kind == "result":
            body = self._with_key(body, record.key)
            retained = _retained_frame_bytes(body, payloads)
            replacing = record.result_bytes
            if self._retained_result_bytes - replacing + retained > MAX_RETAINED_RESULT_BYTES:
                await self.close()
                raise BoundaryError("resumable result retention capacity exhausted")
            record.state = "completed"
            record.result = (body, payloads)
            record.result_bytes = retained
            self._retained_result_bytes += retained - replacing
        lost_epoch: int | None = None
        async with self._send_lock:
            if self._closed:
                raise ConnectionError("resumable conversation is closed")
            writer = self._physical_writer
            epoch = self._owner_epoch
            if writer is not None and (self._rebind_ready or kind == "hello"):
                try:
                    await write_frame(writer, body, payloads)
                except ConnectionError:
                    lost_epoch = epoch
        if lost_epoch is not None:
            await self.detach(lost_epoch)

    def grant_received(self, invocation_id: object) -> None:
        record = self._record_for_frame({"requestId": invocation_id})
        if record is not None:
            record.reserve = None

    def acknowledge_result(self, header: Mapping[str, object]) -> None:
        key = InvocationKey.from_header(header)
        record = self._records.get(key)
        if record is None:
            raise BoundaryError("result acknowledgement names no resumable invocation")
        if record.result is None:
            raise BoundaryError("result acknowledgement precedes the result")
        if record.had_reservation and record.release is None:
            raise BoundaryError("result acknowledgement precedes memory release")
        self._records.pop(key)
        self._keys_by_invocation.pop(key.invocation_id, None)
        self._retained_result_bytes -= record.result_bytes
        self._retained_event_bytes -= record.event_bytes

    def finish_cancelled(self, invocation_id: object) -> None:
        key = self.key_for_invocation(invocation_id)
        if key is None:
            return
        record = self._records.pop(key, None)
        self._keys_by_invocation.pop(key.invocation_id, None)
        if record is not None:
            self._retained_result_bytes -= record.result_bytes
            self._retained_event_bytes -= record.event_bytes

    def plan_rebind(
        self, header: Mapping[str, object]
    ) -> tuple[dict[str, object], tuple[InvocationKey, ...], tuple[InvocationKey, ...]]:
        owner_epoch = header.get("ownerEpoch")
        if type(owner_epoch) is not int or owner_epoch != self._owner_epoch:
            raise BoundaryError("rebind owner epoch is stale")
        raw_entries = header.get("invocations")
        if not isinstance(raw_entries, list):
            raise BoundaryError("rebind invocations must be a bounded list")
        entries = cast("list[object]", raw_entries)
        if len(entries) > MAX_RESUMABLE_INVOCATIONS:
            raise BoundaryError("rebind invocations must be a bounded list")
        requested: list[InvocationKey] = []
        cancellations: list[InvocationKey] = []
        statuses: list[dict[str, object]] = []
        seen: set[InvocationKey] = set()
        for raw in entries:
            if not isinstance(raw, Mapping):
                raise BoundaryError("rebind invocation entry must be an object")
            entry = cast("Mapping[str, object]", raw)
            if set(entry) != {
                "jobRef",
                "attemptId",
                "invocationId",
                "lastEventSeq",
                "cancelled",
            }:
                raise BoundaryError("rebind invocation entry has invalid fields")
            key = InvocationKey.from_header(entry)
            if key in seen:
                raise BoundaryError("rebind invocation identities must be unique")
            seen.add(key)
            last_seq = entry.get("lastEventSeq")
            cancelled = entry.get("cancelled")
            if type(last_seq) is not int or last_seq < 0 or type(cancelled) is not bool:
                raise BoundaryError("rebind invocation state is malformed")
            record = self._records.get(key)
            status = (
                "cancelled"
                if record is None and cancelled
                else ("missing" if record is None else record.state)
            )
            if record is not None:
                requested.append(key)
                if cancelled:
                    status = "cancelled"
                    cancellations.append(key)
            statuses.append({**key.to_wire(), "status": status})
        if not set(self._records).issubset(seen):
            raise BoundaryError("rebind omitted daemon invocation state")
        response: dict[str, object] = {
            "type": "rebindResult",
            "ownerEpoch": self._owner_epoch,
            "invocations": statuses,
        }
        return response, tuple(requested), tuple(cancellations)

    async def complete_rebind(
        self,
        response: dict[str, object],
        requested: Sequence[InvocationKey],
        last_event_sequences: Mapping[InvocationKey, int],
    ) -> None:
        lost_epoch: int | None = None
        async with self._send_lock:
            if self._closed:
                raise ConnectionError("resumable conversation is closed")
            writer = self._physical_writer
            if writer is None:
                return
            if response.get("ownerEpoch") != self._owner_epoch:
                return
            epoch = self._owner_epoch
            try:
                await write_frame(writer, response, [])
                if self._latest_memory_report is not None:
                    await write_frame(
                        writer,
                        self._latest_memory_report[0],
                        self._latest_memory_report[1],
                    )
                for key in requested:
                    record = self._records.get(key)
                    if record is None:
                        continue
                    if record.reserve is not None:
                        await write_frame(writer, record.reserve[0], record.reserve[1])
                    if record.latest_event is not None and record.event_seq > (
                        last_event_sequences.get(key, 0)
                    ):
                        await write_frame(writer, record.latest_event[0], record.latest_event[1])
                    if record.result is not None:
                        await write_frame(writer, record.result[0], record.result[1])
                    if record.release is not None:
                        await write_frame(writer, record.release[0], record.release[1])
            except ConnectionError:
                lost_epoch = epoch
            else:
                self._rebind_ready = True
                self._ready.set()
        if lost_epoch is not None:
            await self.detach(lost_epoch)

    async def close(self) -> None:
        async with self._send_lock:
            if self._closed:
                return
            self._closed = True
            self._transport_lost.set()
            self._ready.set()
        expiry = self._expiry_task
        self._expiry_task = None
        stopping: list[asyncio.Task[None]] = []
        if expiry is not None and expiry is not asyncio.current_task():
            expiry.cancel()
            stopping.append(expiry)
        pump = self._pump_task
        self._pump_task = None
        if pump is not None and pump is not asyncio.current_task():
            pump.cancel()
            stopping.append(pump)
        writer = self._physical_writer
        self._physical_writer = None
        if writer is not None:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
        if stopping:
            await asyncio.gather(*stopping, return_exceptions=True)
        await self._incoming.put(None)
        if self._on_close is not None:
            await self._on_close(self)
