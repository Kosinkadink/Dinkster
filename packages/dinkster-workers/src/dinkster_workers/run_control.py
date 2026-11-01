"""Run-scoped control relay for boundary workers.

The relay carries requests, not callbacks.  A request is always tied to the
boundary invocation that owns it, so a completed or cancelled invocation
cannot address a later run which happens to reuse a node id.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from multiprocessing.shared_memory import SharedMemory
from types import MappingProxyType
from typing import Any, cast

from dinkster_protocol import InvocationResult
from dinkster_schema import RunExecutionContext
from dinkster_values import Value

from .boundary import BoundaryError, ValueCodec, encode_result, release_segment

Send = Callable[[dict[str, object], Sequence[bytes], Sequence[SharedMemory]], Awaitable[None]]


def _document(value: Mapping[str, object]) -> bytes:
    return json.dumps(value, default=dict, separators=(",", ":")).encode("utf-8")


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze(item) for key, item in cast("dict[str, object]", value).items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in cast("list[object]", value))
    return value


class PreparedRunContext:
    """Complete worker-side value preparation before publishing to the engine."""

    def __init__(
        self,
        context: RunExecutionContext,
        prepare_values: Callable[[Mapping[str, Value]], Awaitable[Mapping[str, Value]]],
    ) -> None:
        self._context = context
        self._prepare_values = prepare_values
        self._active = True

    @property
    def node_id(self) -> str:
        return self._context.node_id

    @property
    def inputs(self) -> Mapping[str, Value]:
        return self._context.inputs

    def close(self) -> None:
        self._active = False

    def _check_active(self) -> None:
        if not self._active:
            raise RuntimeError("execution context is no longer active")

    async def get_graph(self) -> Mapping[str, object]:
        self._check_active()
        return await self._context.get_graph()

    async def await_run_value(self, node_id: str) -> Mapping[str, Value]:
        self._check_active()
        return await self._context.await_run_value(node_id)

    async def publish_run_value(self, node_id: str, value: Mapping[str, Value]) -> None:
        self._check_active()
        value = await self._prepare_values(value)
        self._check_active()
        await self._context.publish_run_value(node_id, value)

    async def inhibit(self, node_ids: Sequence[str]) -> None:
        self._check_active()
        await self._context.inhibit(node_ids)

    async def override_node(self, node_id: str, inputs: Mapping[str, object]) -> None:
        self._check_active()
        await self._context.override_node(node_id, inputs)


class BoundaryRunContext:
    """Worker-local proxy for the parent invocation's live run context."""

    def __init__(
        self,
        invocation_id: str,
        node_id: str,
        inputs: Mapping[str, Value],
        codec: ValueCodec,
        send: Send,
        suspend: Callable[[], Awaitable[None]],
        resume: Callable[[], Awaitable[None]],
        prepare_values: Callable[[Mapping[str, Value]], Awaitable[Mapping[str, Value]]],
    ) -> None:
        self._invocation_id = invocation_id
        self._node_id = node_id
        self._inputs = inputs
        self._codec = codec
        self._send = send
        self._suspend = suspend
        self._resume = resume
        self._prepare_values = prepare_values
        self._wait_lock = asyncio.Lock()
        self._pending: dict[str, asyncio.Future[tuple[dict[str, Any], list[bytes]]]] = {}
        self._completed: set[str] = set()
        self._index = 0
        self._closed = False

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def inputs(self) -> Mapping[str, Value]:
        return self._inputs

    def close(self) -> None:
        self._closed = True
        for future in self._pending.values():
            if not future.done():
                future.cancel()
        self._pending.clear()
        self._completed.clear()

    def accept(self, header: dict[str, Any], blobs: list[bytes]) -> bool:
        request_id = header.get("requestId")
        if type(request_id) is not str or header.get("invocationId") != self._invocation_id:
            return False
        future = self._pending.pop(request_id, None)
        if future is None:
            return request_id in self._completed
        if future.done():
            return False
        future.set_result((header, blobs))
        self._completed.add(request_id)
        return True

    async def _request(
        self,
        operation: str,
        fields: Mapping[str, object] = {},
        blobs: Sequence[bytes] = (),
        segments: Sequence[SharedMemory] = (),
    ) -> tuple[dict[str, Any], list[bytes]]:
        if self._closed:
            for segment in segments:
                release_segment(segment)
            raise RuntimeError("execution context is no longer active")
        request_id = f"{self._invocation_id}:control:{self._index}"
        self._index += 1
        future: asyncio.Future[tuple[dict[str, Any], list[bytes]]] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending[request_id] = future
        try:
            await self._send(
                {
                    "type": "runControl",
                    "invocationId": self._invocation_id,
                    "requestId": request_id,
                    "operation": operation,
                    **fields,
                },
                blobs,
                segments,
            )
            header, reply_blobs = await future
        finally:
            self._pending.pop(request_id, None)
        error = header.get("error")
        if type(error) is str:
            if header.get("errorType") == "ValueError":
                raise ValueError(error)
            raise RuntimeError(error)
        return header, reply_blobs

    async def get_graph(self) -> Mapping[str, object]:
        _, blobs = await self._request("getGraph")
        if len(blobs) != 1:
            raise BoundaryError("run control graph response is malformed")
        graph = json.loads(blobs[0])
        if not isinstance(graph, Mapping):
            raise BoundaryError("run control graph response is not a mapping")
        return cast("Mapping[str, object]", _freeze(cast("Mapping[str, object]", graph)))

    async def await_run_value(self, node_id: str) -> Mapping[str, Value]:
        async with self._wait_lock:
            await self._suspend()
            try:
                header, blobs = await self._request("awaitRunValue", {"nodeId": node_id})
            except asyncio.CancelledError:
                raise
            except BaseException:
                if not self._closed:
                    await self._resume()
                raise
            if self._closed:
                raise RuntimeError("execution context is no longer active")
            await self._resume()
        values = header.get("values")
        if not isinstance(values, Mapping):
            raise BoundaryError("run control value response is malformed")
        typed_values = cast("Mapping[object, object]", values)
        consumed: list[str] = []
        result: dict[str, Value] = {
            name: self._codec.decode(cast("Mapping[str, Any]", wire), blobs, consumed)[0]
            for name, wire in typed_values.items()
            if type(name) is str and isinstance(wire, Mapping)
        }
        if len(result) != len(typed_values):
            raise BoundaryError("run control value response is malformed")
        if consumed:
            await self._send({"type": "shmAck", "segments": consumed}, [], [])
        return result

    async def publish_run_value(self, node_id: str, value: Mapping[str, Value]) -> None:
        if self._closed:
            raise RuntimeError("execution context is no longer active")
        value = await self._prepare_values(value)
        with self._codec.suspend_persistent_cas():
            encoded, blobs, segments = encode_result(
                self._codec,
                InvocationResult(outputs=value),
                self._invocation_id,
                0,
            )
        await self._request(
            "publishRunValue", {"nodeId": node_id, "values": encoded["outputs"]}, blobs, segments
        )

    async def inhibit(self, node_ids: Sequence[str]) -> None:
        await self._request("inhibit", {"nodeIds": list(node_ids)})

    async def override_node(self, node_id: str, inputs: Mapping[str, object]) -> None:
        await self._request("overrideNode", {"nodeId": node_id}, [_document(inputs)])


async def dispatch_run_control(
    context: RunExecutionContext,
    codec: ValueCodec,
    header: Mapping[str, Any],
    blobs: list[bytes],
    send: Send,
    publish: Callable[[str, Mapping[str, Value]], Awaitable[None]],
    prepare_values: Callable[[Mapping[str, Value]], Awaitable[None]],
) -> None:
    """Run one parent-side request and send a single reply."""
    request_id = header.get("requestId")
    invocation_id = header.get("invocationId")
    if type(request_id) is not str or type(invocation_id) is not str:
        return
    reply: dict[str, object] = {
        "type": "runControlResult",
        "invocationId": invocation_id,
        "requestId": request_id,
    }
    reply_blobs: list[bytes] = []
    segments: list[SharedMemory] = []
    consumed: list[str] = []
    try:
        operation = header.get("operation")
        if operation == "getGraph":
            reply_blobs.append(_document(await context.get_graph()))
        elif operation == "awaitRunValue":
            node_id = header.get("nodeId")
            if type(node_id) is not str:
                raise BoundaryError("run control node id is malformed")
            published = await context.await_run_value(node_id)
            await prepare_values(published)
            with codec.suspend_persistent_cas():
                encoded, reply_blobs, segments = encode_result(
                    codec,
                    InvocationResult(outputs=published),
                    invocation_id,
                    0,
                )
            reply["values"] = encoded["outputs"]
        elif operation == "publishRunValue":
            raw_values = header.get("values")
            node_id = header.get("nodeId")
            if type(node_id) is not str or not isinstance(raw_values, Mapping):
                raise BoundaryError("run control publish request is malformed")
            typed_values = cast("Mapping[object, object]", raw_values)
            values: dict[str, Value] = {
                name: codec.decode(cast("Mapping[str, Any]", wire), blobs, consumed)[0]
                for name, wire in typed_values.items()
                if type(name) is str and isinstance(wire, Mapping)
            }
            if len(values) != len(typed_values):
                raise BoundaryError("run control publish request is malformed")
            await publish(node_id, values)
        elif operation == "inhibit":
            node_ids_raw = header.get("nodeIds")
            if not isinstance(node_ids_raw, list):
                raise BoundaryError("run control inhibit request is malformed")
            node_id_items = cast("list[object]", node_ids_raw)
            if not all(type(item) is str for item in node_id_items):
                raise BoundaryError("run control inhibit request is malformed")
            node_ids = cast("list[str]", cast("object", node_id_items))
            await context.inhibit(node_ids)
        elif operation == "overrideNode":
            node_id = header.get("nodeId")
            if type(node_id) is not str or len(blobs) != 1:
                raise BoundaryError("run control override request is malformed")
            inputs = json.loads(blobs[0])
            if not isinstance(inputs, Mapping):
                raise BoundaryError("run control override inputs are malformed")
            await context.override_node(node_id, cast("Mapping[str, object]", inputs))
        else:
            raise BoundaryError("unknown run control operation")
    except Exception as exc:  # node-facing failures retain their normal messages
        reply["error"] = str(exc)
        reply["errorType"] = "ValueError" if isinstance(exc, ValueError) else "RuntimeError"
    if consumed:
        await send({"type": "shmAck", "segments": consumed}, [], [])
    await send(reply, reply_blobs, segments)
