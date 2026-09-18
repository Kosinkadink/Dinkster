"""Private BoundarySession transport adapter for MultiDevice workgroups."""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TypeAlias

from dinkster_protocol import (
    WORKGROUP_CAPABILITY,
    AbortWorkGroup,
    BeginWorkGroup,
    CancelWorkGroup,
    CommitWorkGroup,
    DeviceResourceId,
    PrepareReplica,
    ReleaseWorkGroup,
    ReplicaId,
    ReplicaReady,
    ReplicaRefused,
    RunWorkUnit,
    WorkerInstanceId,
    WorkGroupAttempt,
    WorkGroupCancelled,
    WorkGroupDefinition,
    WorkGroupId,
    WorkGroupMessage,
    WorkGroupPrepared,
    WorkGroupProtocolError,
    WorkGroupRefused,
    WorkGroupReleased,
    WorkUnitFailed,
    WorkUnitProgress,
    WorkUnitResult,
    negotiate_workgroup_capabilities,
    workgroup_message_from_wire,
    workgroup_message_to_wire,
)

from .boundary import BoundaryError
from .workgroup import ReplicaEndpoint

WORKGROUP_FRAME_TYPE = "workgroupMessage"
WORKGROUP_HELLO_FIELD = "workgroupCapabilities"

WorkGroupCommandHandler: TypeAlias = Callable[
    [WorkGroupMessage], Awaitable[tuple[WorkGroupMessage, ...]]
]
_SendFrame: TypeAlias = Callable[[Mapping[str, object], Sequence[bytes]], Awaitable[None]]
_Correlation: TypeAlias = tuple[
    WorkGroupId,
    WorkGroupAttempt,
    ReplicaId,
    WorkerInstanceId,
    DeviceResourceId,
]

_COMMAND_TYPES = (
    PrepareReplica,
    BeginWorkGroup,
    CommitWorkGroup,
    AbortWorkGroup,
    RunWorkUnit,
    CancelWorkGroup,
    ReleaseWorkGroup,
)
_REPLY_TYPES = (
    ReplicaReady,
    ReplicaRefused,
    WorkGroupPrepared,
    WorkGroupRefused,
    WorkUnitProgress,
    WorkUnitResult,
    WorkUnitFailed,
    WorkGroupCancelled,
    WorkGroupReleased,
)


class _ReplyChannel:
    def __init__(self, definition: WorkGroupDefinition) -> None:
        self.definition = definition
        self.messages: deque[WorkGroupMessage] = deque()
        self.waiters: deque[asyncio.Future[None]] = deque()
        self.sent = False
        self.terminal_queued = False
        self.settled = False


class WorkGroupTransportClosed(RuntimeError):
    """The boundary transport ended before another reply arrived."""


def workgroup_frame(message: WorkGroupMessage) -> dict[str, object]:
    try:
        body = workgroup_message_to_wire(message)
    except (TypeError, WorkGroupProtocolError) as exc:
        raise BoundaryError(f"invalid workgroup message: {exc}") from exc
    return {"type": WORKGROUP_FRAME_TYPE, "message": body}


def workgroup_message_from_frame(
    header: Mapping[str, object], blobs: Sequence[bytes]
) -> WorkGroupMessage:
    if blobs:
        raise BoundaryError("workgroup frames must not contain blobs")
    fields = header
    expected = {"type", "message"}
    if "blobs" in fields:
        if fields["blobs"] != []:
            raise BoundaryError("workgroup frames must not contain blobs")
        expected.add("blobs")
    if type(header) is not dict or set(fields) != expected:
        raise BoundaryError("workgroup frame has unknown or missing fields")
    if fields.get("type") != WORKGROUP_FRAME_TYPE:
        raise BoundaryError("workgroup frame has the wrong type")
    try:
        return workgroup_message_from_wire(fields["message"])
    except (TypeError, WorkGroupProtocolError) as exc:
        raise BoundaryError(f"invalid workgroup message: {exc}") from exc


class WorkGroupSession:
    """One negotiated, task-free transport shared by bound replica endpoints."""

    def __init__(self, send_frame: _SendFrame) -> None:
        self._send_frame = send_frame
        self._capabilities: frozenset[str] | None = None
        self._channels: dict[_Correlation, _ReplyChannel] = {}
        self._failure: BaseException | None = None

    @property
    def capabilities(self) -> frozenset[str]:
        if self._capabilities is None:
            raise RuntimeError("BoundarySession.begin() has not completed")
        return self._capabilities

    def negotiate(self, hello: Mapping[str, object]) -> None:
        raw = hello.get(WORKGROUP_HELLO_FIELD, [])
        try:
            self._capabilities = negotiate_workgroup_capabilities(raw)
        except WorkGroupProtocolError as exc:
            raise RuntimeError(
                f"worker hello carried malformed {WORKGROUP_HELLO_FIELD}: {exc}"
            ) from exc

    def bind(
        self,
        definition: WorkGroupDefinition,
        replica: ReplicaId,
        *,
        worker_instance: str | None,
    ) -> ReplicaEndpoint:
        if self._capabilities is None:
            raise RuntimeError("BoundarySession.begin() has not completed")
        if WORKGROUP_CAPABILITY not in self._capabilities:
            raise RuntimeError("worker did not negotiate the workgroup capability")
        if self._failure is not None:
            raise WorkGroupTransportClosed("workgroup boundary transport is closed")
        if type(definition) is not WorkGroupDefinition or type(replica) is not ReplicaId:
            raise ValueError("workgroup binding requires exact definition and replica identities")
        binding = next((member for member in definition.members if member.replica == replica), None)
        if binding is None:
            raise ValueError("endpoint replica is absent from the definition")
        if worker_instance is None or binding.worker != WorkerInstanceId(worker_instance):
            raise ValueError("endpoint worker differs from the authenticated session")
        correlation = (
            definition.group,
            definition.attempt,
            binding.replica,
            binding.worker,
            binding.device,
        )
        if correlation in self._channels:
            raise ValueError("endpoint is already bound to this group attempt")
        self._channels = {
            bound: channel for bound, channel in self._channels.items() if not channel.settled
        }
        channel = _ReplyChannel(definition)
        self._channels[correlation] = channel
        return ReplicaEndpoint.bind(
            definition,
            replica,
            send=lambda message: self._send(correlation, message, channel),
            receive=lambda: self.receive(correlation, channel),
        )

    def unbind(self, definition: WorkGroupDefinition, replica: ReplicaId) -> None:
        if type(definition) is not WorkGroupDefinition or type(replica) is not ReplicaId:
            raise ValueError("workgroup unbinding requires exact definition and replica identities")
        binding = next((member for member in definition.members if member.replica == replica), None)
        if binding is None:
            raise ValueError("endpoint replica is absent from the definition")
        correlation = (
            definition.group,
            definition.attempt,
            binding.replica,
            binding.worker,
            binding.device,
        )
        channel = self._channels.get(correlation)
        if channel is None:
            raise ValueError("endpoint is not bound to this group attempt")
        if channel.definition != definition:
            raise ValueError("endpoint is bound to a different workgroup definition")
        pristine = not (
            channel.sent
            or channel.messages
            or channel.waiters
            or channel.terminal_queued
            or channel.settled
        )
        if not pristine and not channel.settled and self._failure is None:
            raise RuntimeError("workgroup endpoint is still active")
        del self._channels[correlation]

    async def send(self, message: WorkGroupMessage) -> None:
        await self._send(None, message)

    async def _send(
        self,
        correlation: _Correlation | None,
        message: WorkGroupMessage,
        expected_channel: _ReplyChannel | None = None,
    ) -> None:
        if self._failure is not None:
            raise WorkGroupTransportClosed(
                "workgroup boundary transport is closed"
            ) from self._failure
        if correlation is not None:
            channel = self._channels.get(correlation)
            if (
                channel is None
                or (expected_channel is not None and channel is not expected_channel)
                or channel.settled
            ):
                raise WorkGroupTransportClosed("workgroup reply channel is closed")
            channel.sent = True
        await self._send_frame(workgroup_frame(message), ())

    async def receive(
        self, correlation: _Correlation, expected_channel: _ReplyChannel | None = None
    ) -> WorkGroupMessage:
        channel = self._channels.get(correlation)
        if (
            channel is None
            or (expected_channel is not None and channel is not expected_channel)
            or channel.settled
        ):
            raise WorkGroupTransportClosed("workgroup reply channel is closed")
        while True:
            if channel.settled:
                raise WorkGroupTransportClosed("workgroup reply channel is closed")
            if channel.messages:
                message = channel.messages.popleft()
                if type(message) is WorkGroupReleased and channel.terminal_queued:
                    channel.settled = True
                    while channel.waiters:
                        waiter = channel.waiters.popleft()
                        if not waiter.done():
                            waiter.set_result(None)
                return message
            if self._failure is not None:
                raise WorkGroupTransportClosed(
                    "workgroup boundary transport is closed"
                ) from self._failure
            future = asyncio.get_running_loop().create_future()
            channel.waiters.append(future)
            try:
                await future
            finally:
                if not future.done():
                    future.cancel()
                with contextlib.suppress(ValueError):
                    channel.waiters.remove(future)

    def accept(self, header: Mapping[str, object], blobs: Sequence[bytes]) -> None:
        message = workgroup_message_from_frame(header, blobs)
        correlation = (
            message.group,
            message.attempt,
            message.replica,
            message.worker,
            message.device,
        )
        channel = self._channels.get(correlation)
        exact_channel = channel is not None
        if channel is not None and (channel.settled or channel.terminal_queued):
            raise BoundaryError("workgroup reply targets a retired group attempt")
        if channel is None:
            if len(self._channels) != 1:
                raise BoundaryError("workgroup reply has no unambiguous bound endpoint")
            bound, channel = next(iter(self._channels.items()))
            same_endpoint = correlation[:1] + correlation[2:] == bound[:1] + bound[2:]
            if same_endpoint and correlation[1] != bound[1]:
                raise BoundaryError("workgroup reply targets a retired group attempt")
        channel.messages.append(message)
        if exact_channel and type(message) is WorkGroupReleased:
            channel.terminal_queued = True
        while channel.waiters:
            waiter = channel.waiters.popleft()
            if not waiter.done():
                waiter.set_result(None)
                return

    def fail(self, cause: BaseException | None = None) -> None:
        if self._failure is not None:
            return
        self._failure = cause or WorkGroupTransportClosed("workgroup boundary transport is closed")
        for channel in self._channels.values():
            while channel.waiters:
                waiter = channel.waiters.popleft()
                if not waiter.done():
                    waiter.set_exception(
                        WorkGroupTransportClosed("workgroup boundary transport is closed")
                    )


def workgroup_command_from_frame(
    header: Mapping[str, object],
    blobs: Sequence[bytes],
) -> WorkGroupMessage:
    command = workgroup_message_from_frame(header, blobs)
    if type(command) not in _COMMAND_TYPES:
        raise BoundaryError("workgroup handler received a non-exact or illegal command")
    return command


async def handle_workgroup_command(
    handler: WorkGroupCommandHandler,
    command: WorkGroupMessage,
) -> tuple[dict[str, object], ...]:
    replies = await handler(command)
    if type(replies) is not tuple:
        raise BoundaryError("workgroup handler must return an exact tuple of replies")
    if any(type(reply) not in _REPLY_TYPES for reply in replies):
        raise BoundaryError("workgroup handler returned a non-exact or illegal reply")
    return tuple(workgroup_frame(reply) for reply in replies)
