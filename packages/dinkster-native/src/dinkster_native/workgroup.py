"""Workgroup lifecycle gate for process-isolated native invocations."""

from __future__ import annotations

import asyncio
import importlib
import os
from dataclasses import dataclass, field
from typing import Any

from dinkster_protocol import (
    MAX_REASON_BYTES,
    WORKGROUP_CAPABILITY,
    WORKGROUP_DATA_PLANE_CAPABILITY,
    AbortWorkGroup,
    BeginWorkGroup,
    CancelWorkGroup,
    CommitWorkGroup,
    PrepareReplica,
    ReleaseWorkGroup,
    ReplicaReady,
    RunWorkUnit,
    WorkGroupCancelled,
    WorkGroupMessage,
    WorkGroupPrepared,
    WorkGroupReleased,
    WorkUnitFailed,
    WorkUnitResult,
)


def _bounded_reason(reason: str) -> str:
    encoded = reason.encode("utf-8", errors="replace")
    if len(encoded) <= MAX_REASON_BYTES:
        return encoded.decode("utf-8")
    return encoded[:MAX_REASON_BYTES].decode("utf-8", errors="ignore")


def _distributed_lifecycle(name: str) -> Any:
    try:
        module = importlib.import_module("dinkster_inference_torch.distributed")
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        return _torch_free_lifecycle
    return getattr(module, name)


def _torch_free_lifecycle(*_args: object) -> None:
    return None


def _log_pinned_storage(phase: str) -> None:
    if os.environ.get("DINKSTER_PINNED_STORAGE_DEBUG") != "1":
        return
    module = importlib.import_module("dinkster_inference_torch.pinned_host")
    module.log_storage_ledger(phase)


@dataclass
class _Attempt:
    command: PrepareReplica
    prepared: bool = False
    active: bool = False
    activated: asyncio.Event = field(default_factory=asyncio.Event)
    run_allowed: asyncio.Event = field(default_factory=asyncio.Event)
    completion: asyncio.Future[str | None] | None = None
    invocation_id: str | None = None
    cancelled: bool = False

    def common(self) -> dict[str, object]:
        return {
            "worker": self.command.worker,
            "replica": self.command.replica,
            "group": self.command.group,
            "attempt": self.command.attempt,
            "device": self.command.device,
        }


class SingleJobWorkGroupHandler:
    workgroup_capabilities = (
        WORKGROUP_CAPABILITY,
        WORKGROUP_DATA_PLANE_CAPABILITY,
    )

    def __init__(self) -> None:
        self._attempts: dict[tuple[object, object], _Attempt] = {}

    @staticmethod
    def _key(message: WorkGroupMessage) -> tuple[object, object]:
        return message.group, message.attempt

    def _attempt(self, message: WorkGroupMessage) -> _Attempt:
        attempt = self._attempts.get(self._key(message))
        if attempt is None:
            raise RuntimeError("workgroup command has no prepared attempt")
        prepared = attempt.command
        if (
            message.worker,
            message.replica,
            message.device,
        ) != (
            prepared.worker,
            prepared.replica,
            prepared.device,
        ):
            raise RuntimeError("workgroup command differs from the prepared rank")
        return attempt

    async def __call__(self, message: WorkGroupMessage) -> tuple[WorkGroupMessage, ...]:
        if type(message) is PrepareReplica:
            key = self._key(message)
            if key in self._attempts:
                raise RuntimeError("workgroup attempt was prepared twice")
            attempt = _Attempt(message)
            self._attempts[key] = attempt
            return (ReplicaReady(**attempt.common()),)  # type: ignore[arg-type]

        attempt = self._attempt(message)
        common = attempt.common()
        if type(message) is BeginWorkGroup:
            if attempt.prepared:
                raise RuntimeError("workgroup admission was requested twice")
            attempt.prepared = True
            return (WorkGroupPrepared(**common),)  # type: ignore[arg-type]
        if type(message) is CommitWorkGroup:
            if not attempt.prepared or attempt.active:
                raise RuntimeError("workgroup commit is out of order")
            attempt.active = True
            _distributed_lifecycle("activate_distributed_sampling_attempt")(
                message.group.value,
                message.attempt.value,
            )
            attempt.activated.set()
            return ()
        if type(message) is RunWorkUnit:
            if not attempt.active or attempt.completion is not None:
                raise RuntimeError("workgroup run is out of order")
            attempt.completion = asyncio.get_running_loop().create_future()
            attempt.run_allowed.set()
            failure = await attempt.completion
            unit = {"unit": message.unit, "slot": message.slot}
            if failure is None:
                return (WorkUnitResult(**common, **unit),)  # type: ignore[arg-type]
            failure_reply = WorkUnitFailed(
                **common,  # type: ignore[arg-type]
                **unit,
                reason=_bounded_reason(failure),
            )
            return (failure_reply,)
        if type(message) is CancelWorkGroup:
            attempt.cancelled = True
            attempt.activated.set()
            attempt.run_allowed.set()
            if attempt.completion is not None and not attempt.completion.done():
                attempt.completion.set_result(message.reason)
            return (WorkGroupCancelled(**common),)  # type: ignore[arg-type]
        if type(message) is AbortWorkGroup:
            attempt.activated.set()
            attempt.run_allowed.set()
            return ()
        if type(message) is ReleaseWorkGroup:
            if attempt.active:
                _distributed_lifecycle("release_distributed_sampling_attempt")(
                    message.group.value,
                    message.attempt.value,
                )
            self._attempts.pop(self._key(message), None)
            return (WorkGroupReleased(**common),)  # type: ignore[arg-type]
        raise RuntimeError(f"unsupported workgroup command {type(message).__name__}")

    async def before_invocation(self, invocation_id: str) -> None:
        attempts = tuple(self._attempts.values())
        if not attempts:
            return
        if len(attempts) != 1:
            raise RuntimeError("multiple workgroup attempts are active")
        attempt = attempts[0]
        await attempt.activated.wait()
        if not attempt.active:
            raise RuntimeError("workgroup attempt ended before invocation dispatch")
        await attempt.run_allowed.wait()
        if attempt.invocation_id is not None:
            raise RuntimeError("workgroup attempt received multiple invocations")
        attempt.invocation_id = invocation_id

    async def after_invocation(self, invocation_id: str, failure: str | None) -> None:
        _log_pinned_storage(invocation_id)
        attempt = next(
            (
                candidate
                for candidate in self._attempts.values()
                if candidate.invocation_id == invocation_id
            ),
            None,
        )
        if attempt is None or attempt.completion is None:
            return
        if not attempt.completion.done():
            attempt.completion.set_result(failure)

    def invocation_cancelled(self, invocation_id: str) -> bool:
        return any(
            attempt.invocation_id == invocation_id and attempt.cancelled
            for attempt in self._attempts.values()
        )

    def parent_manages_invocation_reservations(self) -> bool:
        return True


def create_single_job_workgroup_handler() -> SingleJobWorkGroupHandler:
    return SingleJobWorkGroupHandler()
