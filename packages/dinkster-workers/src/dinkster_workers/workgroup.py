"""Parent-owned runtime coordination for fenced MultiDevice workgroups."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import cast

from dinkster_memory import ReservationRequest, ReservationService
from dinkster_protocol import (
    AbortWorkGroup,
    BeginAdmission,
    BeginWorkGroup,
    CancelWorkGroup,
    CommitWorkGroup,
    DispatchWork,
    GatherFailed,
    GatherSucceeded,
    PrepareReplica,
    ReleaseWorkGroup,
    ReplicaBinding,
    ReplicaId,
    ReplicaReady,
    ReplicaRefused,
    RequestCancellation,
    RunWorkUnit,
    SettleFailure,
    WorkGroupCancelled,
    WorkGroupDefinition,
    WorkGroupEvent,
    WorkGroupLifecycle,
    WorkGroupMessage,
    WorkGroupPrepared,
    WorkGroupProtocolError,
    WorkGroupRefused,
    WorkGroupReleased,
    WorkGroupState,
    WorkUnitFailed,
    WorkUnitProgress,
    WorkUnitResult,
    reduce_workgroup,
)

__all__ = [
    "ReplicaEndpoint",
    "WorkGroupCoordinator",
    "WorkGroupPreparationRefused",
    "WorkGroupRuntimeError",
]

_Send = Callable[[WorkGroupMessage], Awaitable[None]]
_Receive = Callable[[], Awaitable[WorkGroupMessage]]
_Gather = Callable[[WorkGroupLifecycle], Awaitable[str | None]]

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
_UNIT_TYPES = (RunWorkUnit, WorkUnitProgress, WorkUnitResult, WorkUnitFailed)


class WorkGroupRuntimeError(RuntimeError):
    """A runtime failure with the last honestly observed lifecycle."""

    def __init__(self, message: str, lifecycle: WorkGroupLifecycle) -> None:
        super().__init__(message)
        self.lifecycle = lifecycle


class WorkGroupPreparationRefused(WorkGroupRuntimeError):
    """Replica preparation refused before reducer-owned admission."""

    def __init__(
        self,
        refusals: tuple[ReplicaRefused, ...],
        lifecycle: WorkGroupLifecycle,
    ) -> None:
        reasons = ", ".join(refusal.reason for refusal in refusals)
        super().__init__(f"preparation refused: {reasons}", lifecycle)
        self.refusals = refusals


@dataclass(frozen=True, slots=True, init=False)
class ReplicaEndpoint:
    """One transport-independent endpoint bound to one replica attempt.

    A peer answers each ``BeginWorkGroup`` before acknowledging a later
    abort/release so the coordinator can close reducer-owned admission.
    """

    definition: WorkGroupDefinition
    binding: ReplicaBinding
    _sender: _Send
    _receiver: _Receive

    @classmethod
    def bind(
        cls,
        definition: WorkGroupDefinition,
        replica: ReplicaId,
        *,
        send: _Send,
        receive: _Receive,
    ) -> ReplicaEndpoint:
        if type(definition) is not WorkGroupDefinition or type(replica) is not ReplicaId:
            raise ValueError("endpoint requires exact definition and replica identities")
        binding = next((member for member in definition.members if member.replica == replica), None)
        if binding is None:
            raise ValueError("endpoint replica is absent from the definition")
        endpoint = object.__new__(cls)
        object.__setattr__(endpoint, "definition", definition)
        object.__setattr__(endpoint, "binding", binding)
        object.__setattr__(endpoint, "_sender", send)
        object.__setattr__(endpoint, "_receiver", receive)
        return endpoint

    async def send(self, message: WorkGroupMessage) -> None:
        self._validate(message, _COMMAND_TYPES, "command")
        await self._sender(message)

    async def receive(self) -> WorkGroupMessage:
        message = await self._receiver()
        self._validate(message, _REPLY_TYPES, "reply")
        return message

    def _validate(
        self,
        message: WorkGroupMessage,
        allowed: tuple[type[object], ...],
        direction: str,
    ) -> None:
        if type(message) not in allowed:
            raise ValueError(f"endpoint {direction} has a non-exact or illegal type")
        binding = self.binding
        if (
            message.group,
            message.attempt,
            message.replica,
            message.worker,
            message.device,
        ) != (
            self.definition.group,
            self.definition.attempt,
            binding.replica,
            binding.worker,
            binding.device,
        ):
            raise ValueError(f"endpoint {direction} is not bound to this group attempt")
        if type(message) is PrepareReplica and message.recipe != binding.recipe:
            raise ValueError("endpoint command has a foreign replica recipe")
        if type(message) in _UNIT_TYPES:
            unit_message = cast(
                "RunWorkUnit | WorkUnitProgress | WorkUnitResult | WorkUnitFailed",
                message,
            )
            unit = next(
                (
                    candidate
                    for candidate in self.definition.units
                    if candidate.unit == unit_message.unit
                ),
                None,
            )
            if unit is None or (unit.replica, unit.slot) != (
                unit_message.replica,
                unit_message.slot,
            ):
                raise ValueError(f"endpoint {direction} has a foreign unit or semantic slot")


@dataclass(frozen=True, slots=True)
class WorkGroupCoordinator:
    """Execute one immutable workgroup while retaining parent reservations."""

    reservations: ReservationService
    cleanup_timeout: float = 1.0

    def __post_init__(self) -> None:
        if self.cleanup_timeout <= 0:
            raise ValueError("cleanup_timeout must be positive")

    def validate(
        self,
        definition: WorkGroupDefinition,
        endpoints: Sequence[ReplicaEndpoint],
        reservation_requests: Sequence[ReservationRequest] = (),
    ) -> tuple[ReplicaEndpoint, ...]:
        if type(definition) is not WorkGroupDefinition:
            raise ValueError("coordinator requires an exact workgroup definition")
        endpoint_tuple = tuple(endpoints)
        if any(type(endpoint) is not ReplicaEndpoint for endpoint in endpoint_tuple):
            raise ValueError("coordinator requires exact replica endpoints")
        by_replica = {endpoint.binding.replica: endpoint for endpoint in endpoint_tuple}
        if len(by_replica) != len(endpoint_tuple) or set(by_replica) != {
            member.replica for member in definition.members
        }:
            raise ValueError("endpoints must cover every definition member exactly once")
        ordered = tuple(by_replica[member.replica] for member in definition.members)
        if any(endpoint.definition != definition for endpoint in ordered):
            raise ValueError("endpoint is bound to a different definition or attempt")
        if any(
            endpoint.binding != member
            for endpoint, member in zip(ordered, definition.members, strict=True)
        ):
            raise ValueError("endpoint binding differs from the definition")
        if any(type(request) is not ReservationRequest for request in reservation_requests):
            raise ValueError("reservation requests must have exact public types")
        return ordered

    async def execute(
        self,
        definition: WorkGroupDefinition,
        endpoints: Sequence[ReplicaEndpoint],
        reservation_requests: Sequence[ReservationRequest] = (),
        *,
        gather: _Gather | None = None,
        started: asyncio.Event | None = None,
        on_failure: Callable[[], None] | None = None,
    ) -> WorkGroupLifecycle:
        def notify_failure() -> None:
            if on_failure is None:
                return
            try:
                on_failure()
            except BaseException:
                # Notification cannot replace or interrupt owned cleanup.
                pass

        ordered = self.validate(definition, endpoints, reservation_requests)
        runtime = _Runtime(definition, ordered, self.cleanup_timeout, gather, started)
        requests = tuple(reservation_requests)
        async with self.reservations.reserve(requests):
            runtime.start_pumps()
            try:
                lifecycle = await runtime.run()
                if lifecycle.state is WorkGroupState.FAILED:
                    notify_failure()
                return lifecycle
            except asyncio.CancelledError:
                cleanup = asyncio.create_task(runtime.emergency_cleanup("caller cancelled"))
                await _await_owned(cleanup)
                raise
            except _EndpointFailure as exc:
                notify_failure()
                cleanup = asyncio.create_task(runtime.emergency_cleanup(str(exc)))
                await _await_owned(cleanup)
                message = (
                    "endpoint reply rejected"
                    if isinstance(exc.cause, (ValueError, WorkGroupProtocolError))
                    else "transport lost"
                )
                raise WorkGroupRuntimeError(message, runtime.lifecycle) from exc.cause
            except WorkGroupRuntimeError as exc:
                notify_failure()
                cleanup = asyncio.create_task(runtime.emergency_cleanup("runtime failed"))
                await _await_owned(cleanup)
                exc.lifecycle = runtime.lifecycle
                raise
            except Exception as exc:
                notify_failure()
                cleanup = asyncio.create_task(
                    runtime.emergency_cleanup(str(exc) or "runtime failed")
                )
                await _await_owned(cleanup)
                raise WorkGroupRuntimeError("workgroup runtime failed", runtime.lifecycle) from exc
            finally:
                await runtime.stop_pumps()


async def _await_owned(task: asyncio.Task[None]) -> None:
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError as cancelled:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        await task
        raise cancelled


@dataclass(frozen=True, slots=True)
class _Reply:
    endpoint: ReplicaEndpoint
    message: WorkGroupMessage | None = None
    error: BaseException | None = None


class _EndpointFailure(RuntimeError):
    def __init__(self, endpoint: ReplicaEndpoint, cause: BaseException) -> None:
        super().__init__(f"endpoint {endpoint.binding.replica.value} failed: {cause}")
        self.endpoint = endpoint
        self.cause = cause


class _Runtime:
    def __init__(
        self,
        definition: WorkGroupDefinition,
        endpoints: tuple[ReplicaEndpoint, ...],
        cleanup_timeout: float,
        gather: _Gather | None,
        started: asyncio.Event | None,
    ) -> None:
        self.definition = definition
        self.endpoints = endpoints
        self.by_replica = {endpoint.binding.replica: endpoint for endpoint in endpoints}
        self.cleanup_timeout = cleanup_timeout
        self.gather = gather
        self.started = started
        self.lifecycle = WorkGroupLifecycle(definition)
        self.queues = {
            endpoint.binding.replica: asyncio.Queue[_Reply](maxsize=1) for endpoint in endpoints
        }
        self.pumps: tuple[asyncio.Task[None], ...] = ()
        self.getters: dict[ReplicaId, asyncio.Task[_Reply]] = {}
        self.contacted: set[ReplicaId] = set()
        self.lost: set[ReplicaId] = set()
        self.abort_sent: set[ReplicaId] = set()
        self.cancel_sent: set[ReplicaId] = set()
        self.release_sent: set[ReplicaId] = set()
        self.bridge_released: set[ReplicaId] = set()

    def start_pumps(self) -> None:
        self.pumps = tuple(asyncio.create_task(self._pump(endpoint)) for endpoint in self.endpoints)
        self.getters = {
            replica: asyncio.create_task(queue.get()) for replica, queue in self.queues.items()
        }

    async def stop_pumps(self) -> None:
        getters = tuple(self.getters.values())
        for task in (*self.pumps, *getters):
            task.cancel()
        await asyncio.gather(*self.pumps, return_exceptions=True)
        await asyncio.gather(*getters, return_exceptions=True)

    async def _pump(self, endpoint: ReplicaEndpoint) -> None:
        queue = self.queues[endpoint.binding.replica]
        try:
            while True:
                await queue.put(_Reply(endpoint, message=await endpoint.receive()))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await queue.put(_Reply(endpoint, error=exc))

    async def _next(
        self,
        allowed: set[ReplicaId] | None = None,
        timeout: float | None = None,
    ) -> tuple[ReplicaEndpoint, WorkGroupMessage]:
        replicas = allowed if allowed is not None else set(self.queues)
        if not replicas:
            raise RuntimeError("no replica replies are eligible")
        tasks = {self.getters[replica]: replica for replica in replicas}
        done, _ = await asyncio.wait(
            tasks,
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            raise TimeoutError
        task = min(done, key=lambda candidate: tasks[candidate].value)
        replica = tasks[task]
        reply = task.result()
        self.getters[replica] = asyncio.create_task(self.queues[replica].get())
        if reply.error is not None:
            self.lost.add(reply.endpoint.binding.replica)
            raise _EndpointFailure(reply.endpoint, reply.error)
        return reply.endpoint, cast("WorkGroupMessage", reply.message)

    async def _send(self, message: WorkGroupMessage) -> None:
        endpoint = self.by_replica[message.replica]
        if type(message) is PrepareReplica:
            self.contacted.add(message.replica)
        elif type(message) is AbortWorkGroup:
            if message.replica in self.abort_sent:
                return
            self.abort_sent.add(message.replica)
        elif type(message) is CancelWorkGroup:
            if message.replica in self.cancel_sent:
                return
            self.cancel_sent.add(message.replica)
        elif type(message) is ReleaseWorkGroup:
            if message.replica in self.release_sent:
                return
            self.release_sent.add(message.replica)
        try:
            await endpoint.send(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.lost.add(message.replica)
            raise _EndpointFailure(endpoint, exc) from exc

    async def _send_all(self, messages: Sequence[WorkGroupMessage]) -> None:
        for message in messages:
            await self._send(message)

    async def _send_best_effort(
        self,
        messages: Sequence[WorkGroupMessage],
        deadline: float,
    ) -> None:
        pending = tuple(messages)
        for index, message in enumerate(pending):
            remaining = self._remaining(deadline)
            if remaining <= 0:
                return
            budget = remaining / (len(pending) - index)
            try:
                await asyncio.wait_for(self._send(message), budget)
            except TimeoutError:
                self.lost.add(message.replica)
            except _EndpointFailure:
                continue

    def _reduce(self, event: WorkGroupEvent) -> tuple[WorkGroupMessage, ...]:
        self.lifecycle, commands = reduce_workgroup(self.lifecycle, event)
        return commands

    async def run(self) -> WorkGroupLifecycle:
        await self._send_all(
            tuple(
                PrepareReplica(
                    worker=member.worker,
                    replica=member.replica,
                    group=self.definition.group,
                    attempt=self.definition.attempt,
                    device=member.device,
                    recipe=member.recipe,
                )
                for member in self.definition.members
            )
        )
        preparation: dict[ReplicaId, ReplicaReady | ReplicaRefused] = {}
        while len(preparation) < len(self.endpoints):
            endpoint, reply = await self._next(set(self.by_replica) - set(preparation))
            if type(reply) not in (ReplicaReady, ReplicaRefused):
                raise WorkGroupProtocolError("preparation received an illegal or duplicate reply")
            replica = endpoint.binding.replica
            if replica in preparation:
                raise WorkGroupProtocolError("preparation received a duplicate reply")
            preparation[replica] = cast("ReplicaReady | ReplicaRefused", reply)
        refusals = tuple(reply for reply in preparation.values() if type(reply) is ReplicaRefused)
        if refusals:
            await self._bridge_rollback()
            raise WorkGroupPreparationRefused(refusals, self.lifecycle)

        await self._send_all(self._reduce(BeginAdmission()))
        admission_replies: set[ReplicaId] = set()
        refused = False
        while len(admission_replies) < len(self.endpoints):
            endpoint, reply = await self._next(set(self.by_replica) - admission_replies)
            replica = endpoint.binding.replica
            if replica in admission_replies or type(reply) not in (
                WorkGroupPrepared,
                WorkGroupRefused,
            ):
                raise WorkGroupProtocolError("admission received an illegal or duplicate reply")
            admission_replies.add(replica)
            if type(reply) is WorkGroupRefused:
                await self._send_all(self._reduce(reply))
                refused = True
            else:
                await self._send_all(self._reduce(cast("WorkGroupPrepared", reply)))
        if refused:
            await self._bridge_release()
            return self.lifecycle

        dispatch = self._reduce(DispatchWork())
        if self.started is not None:
            self.started.set()
        await self._send_all(dispatch)
        while self.lifecycle.state is WorkGroupState.RUNNING:
            _, reply = await self._next()
            if type(reply) not in (WorkUnitProgress, WorkUnitResult, WorkUnitFailed):
                raise WorkGroupProtocolError("running workgroup received an illegal reply")
            commands = self._reduce(cast("WorkGroupEvent", reply))
            await self._send_all(commands)
            if self._is_state(WorkGroupState.FAILING):
                return await self._finish_failure()

        if not self._is_state(WorkGroupState.GATHERING):
            raise WorkGroupProtocolError("workgroup left running without gathering")
        reason = await self.gather(self.lifecycle) if self.gather is not None else None
        commands = self._reduce(GatherSucceeded() if reason is None else GatherFailed(reason))
        await self._send_all(commands)
        if self._is_state(WorkGroupState.FAILING):
            return await self._finish_failure()
        await self._drain_releases(reducer_owned=True)
        return self.lifecycle

    async def _finish_failure(self) -> WorkGroupLifecycle:
        while len(self.lifecycle.cancelled) < len(self.definition.members):
            remaining = set(self.by_replica) - set(self.lifecycle.cancelled)
            _, reply = await self._next(remaining)
            if type(reply) is WorkUnitFailed:
                self._reduce(reply)
                continue
            if type(reply) in (WorkUnitProgress, WorkUnitResult):
                continue
            if type(reply) is not WorkGroupCancelled:
                raise WorkGroupProtocolError("failure cleanup received an illegal reply")
            await self._send_all(self._reduce(reply))
        await self._drain_releases(reducer_owned=True)
        self._reduce(SettleFailure())
        return self.lifecycle

    async def _bridge_rollback(self) -> None:
        await self._send_all(
            tuple(self._child(AbortWorkGroup, endpoint.binding) for endpoint in self.endpoints)
        )
        await self._bridge_release()

    async def _bridge_release(self) -> None:
        await self._send_all(
            tuple(self._child(ReleaseWorkGroup, endpoint.binding) for endpoint in self.endpoints)
        )
        await self._drain_releases(reducer_owned=False)

    async def _drain_releases(self, *, reducer_owned: bool) -> None:
        expected = {endpoint.binding.replica for endpoint in self.endpoints}
        while (set(self.lifecycle.released) if reducer_owned else self.bridge_released) != expected:
            observed = set(self.lifecycle.released) if reducer_owned else self.bridge_released
            endpoint, reply = await self._next(expected - observed)
            if reducer_owned and type(reply) is WorkUnitFailed:
                if self._is_state(WorkGroupState.FAILING):
                    self._reduce(reply)
                continue
            if reducer_owned and type(reply) in (WorkUnitProgress, WorkUnitResult):
                continue
            if type(reply) is not WorkGroupReleased:
                raise WorkGroupProtocolError("release drain received an illegal reply")
            if reducer_owned:
                self._reduce(reply)
            else:
                replica = endpoint.binding.replica
                if replica in self.bridge_released:
                    raise WorkGroupProtocolError("release drain received a duplicate reply")
                self.bridge_released.add(replica)

    async def emergency_cleanup(self, reason: str) -> None:
        del reason
        deadline = asyncio.get_running_loop().time() + self.cleanup_timeout

        if self._is_state(WorkGroupState.DEFINED):
            contacted = tuple(
                endpoint
                for endpoint in self.endpoints
                if endpoint.binding.replica in self.contacted
            )
            await self._send_best_effort(
                tuple(self._child(AbortWorkGroup, endpoint.binding) for endpoint in contacted),
                deadline,
            )
            await self._send_best_effort(
                tuple(self._child(ReleaseWorkGroup, endpoint.binding) for endpoint in contacted),
                deadline,
            )
            await self._emergency_release_drain(
                {endpoint.binding.replica for endpoint in contacted}, deadline
            )
            return

        if self.lifecycle.state in (
            WorkGroupState.ADMITTING,
            WorkGroupState.DISPATCHING,
            WorkGroupState.RUNNING,
            WorkGroupState.GATHERING,
        ):
            try:
                commands = self._reduce(RequestCancellation("runtime cleanup"))
            except WorkGroupProtocolError:
                commands = ()
            await self._send_best_effort(commands, deadline)

        if self._is_state(WorkGroupState.REFUSED):
            await self._send_best_effort(
                tuple(self._child(AbortWorkGroup, endpoint.binding) for endpoint in self.endpoints),
                deadline,
            )
            await self._send_best_effort(
                tuple(
                    self._child(ReleaseWorkGroup, endpoint.binding) for endpoint in self.endpoints
                ),
                deadline,
            )
            await self._emergency_release_drain(set(self.by_replica), deadline)
            return

        if self.lifecycle.state in (WorkGroupState.CANCELLING, WorkGroupState.FAILING):
            cancel_targets = tuple(
                endpoint
                for endpoint in self.endpoints
                if endpoint.binding.replica not in self.lifecycle.cancelled
            )
            await self._send_best_effort(
                tuple(
                    self._cancel(endpoint.binding, "runtime cleanup") for endpoint in cancel_targets
                ),
                deadline,
            )
            while self._remaining(deadline) > 0:
                awaiting = {
                    endpoint.binding.replica
                    for endpoint in cancel_targets
                    if endpoint.binding.replica not in self.lifecycle.cancelled
                    and endpoint.binding.replica not in self.lost
                }
                if not awaiting:
                    break
                try:
                    _, reply = await self._next(awaiting, self._remaining(deadline))
                except TimeoutError:
                    break
                except _EndpointFailure:
                    continue
                if type(reply) is WorkGroupPrepared:
                    try:
                        self._reduce(reply)
                    except WorkGroupProtocolError:
                        pass
                elif type(reply) in (WorkUnitProgress, WorkUnitResult):
                    if self._is_state(WorkGroupState.CANCELLING):
                        try:
                            self._reduce(cast("WorkGroupEvent", reply))
                        except WorkGroupProtocolError:
                            pass
                elif type(reply) is WorkGroupCancelled:
                    try:
                        await self._send_best_effort(self._reduce(reply), deadline)
                    except WorkGroupProtocolError:
                        pass

        if self.lifecycle.state in (WorkGroupState.SUCCEEDED, WorkGroupState.CANCELLED):
            release_targets = tuple(self.endpoints)
        else:
            release_targets = tuple(
                endpoint
                for endpoint in self.endpoints
                if endpoint.binding.replica in self.lifecycle.cancelled
                and endpoint.binding.replica not in self.lost
            )
        await self._send_best_effort(
            tuple(self._child(ReleaseWorkGroup, endpoint.binding) for endpoint in release_targets),
            deadline,
        )
        await self._emergency_release_drain(
            {endpoint.binding.replica for endpoint in release_targets}, deadline
        )

    async def _emergency_release_drain(
        self,
        targets: set[ReplicaId],
        deadline: float,
    ) -> None:
        while self._remaining(deadline) > 0:
            observed = set(self.lifecycle.released) | self.bridge_released
            awaiting = {replica for replica in targets - observed if replica not in self.lost}
            if not awaiting:
                return
            try:
                endpoint, reply = await self._next(awaiting, self._remaining(deadline))
            except TimeoutError:
                return
            except _EndpointFailure:
                continue
            if type(reply) is not WorkGroupReleased:
                continue
            if self.lifecycle.state in (
                WorkGroupState.SUCCEEDED,
                WorkGroupState.CANCELLED,
            ) or (
                self._is_state(WorkGroupState.FAILING)
                and len(self.lifecycle.cancelled) == len(self.definition.members)
            ):
                try:
                    self._reduce(reply)
                    continue
                except WorkGroupProtocolError:
                    pass
            self.bridge_released.add(endpoint.binding.replica)

    @staticmethod
    def _remaining(deadline: float) -> float:
        return max(0.0, deadline - asyncio.get_running_loop().time())

    def _is_state(self, state: WorkGroupState) -> bool:
        return self.lifecycle.state is state

    def _child(
        self, kind: type[AbortWorkGroup] | type[ReleaseWorkGroup], binding: ReplicaBinding
    ) -> AbortWorkGroup | ReleaseWorkGroup:
        return kind(
            worker=binding.worker,
            replica=binding.replica,
            group=self.definition.group,
            attempt=self.definition.attempt,
            device=binding.device,
        )

    def _cancel(self, binding: ReplicaBinding, reason: str) -> CancelWorkGroup:
        return CancelWorkGroup(
            worker=binding.worker,
            replica=binding.replica,
            group=self.definition.group,
            attempt=self.definition.attempt,
            device=binding.device,
            reason=reason,
        )
