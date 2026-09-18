"""Reservation contracts: how workers ask for memory without seeing the
governor (DESIGN 3.10, hazards H9/H13).

A worker about to materialize something large asks *before* allocating.
It speaks in ReservationRequests - bytes on a residency class - to a
ReservationService, and never holds the MemoryGovernor itself: in-process
composition hands it GovernorReservationService directly, the isolated
boundary relays the same requests as lease frames, and future remote
endpoints will speak the same shape over the network. Node code sees none
of this; planning happens in worker wrappers and pack-declared policies.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

from dinkster_values import Value

from .governor import MemoryGovernor


@dataclass(frozen=True)
class ReservationRequest:
    """Bytes to hold on one residency class for one invocation's duration.

    residency uses the same keys budgets and COST_META_KEY use:
    ``"ram"``, ``"disk"``, ``"vram:cuda:0"``.
    """

    residency: str
    nbytes: int

    def __post_init__(self) -> None:
        if not self.residency:
            raise ValueError("residency must be a non-empty string")
        if self.nbytes < 0:
            raise ValueError("nbytes must be >= 0")


@runtime_checkable
class InvocationView(Protocol):
    """What a reservation planner may observe: node type and envelopes.

    Deliberately narrower than the engine's Invocation (which satisfies it
    structurally): planning is pure observation of *what* will run and what
    metadata rides its inputs - never node code, schemas, or devices. Packs
    write planners against this view without depending on the engine.
    """

    @property
    def node_type(self) -> str: ...

    @property
    def inputs(self) -> Mapping[str, Value]: ...


ReservationPlanner: TypeAlias = Callable[[InvocationView], Sequence[ReservationRequest]]
"""Pack policy: what an invocation is about to materialize.

Returning () means "nothing worth reserving"; raising means the plan itself
cannot be trusted (for example an asset that declares no byte size), which
fails the invocation honestly instead of silently reserving zero.
"""


@runtime_checkable
class ReservationService(Protocol):
    """Admission for a batch of reservation requests, held as one lease."""

    def reserve(
        self, requests: Sequence[ReservationRequest]
    ) -> AbstractAsyncContextManager[None]: ...


class GovernorReservationService:
    """The direct implementation: requests become governor reservations.

    Duplicate residency classes are merged (one governor reservation per
    class), and classes are acquired in sorted order - the same total-order
    discipline the engine's admission lanes use, so two multi-class
    reservations can never deadlock each other.
    """

    def __init__(self, governor: MemoryGovernor, *, timeout: float | None = None) -> None:
        self._governor = governor
        self._timeout = timeout

    @asynccontextmanager
    async def reserve(self, requests: Sequence[ReservationRequest]) -> AsyncGenerator[None]:
        merged: dict[str, int] = {}
        for request in requests:
            merged[request.residency] = merged.get(request.residency, 0) + request.nbytes
        async with AsyncExitStack() as stack:
            for residency in sorted(merged):
                await stack.enter_async_context(
                    self._governor.reserve(residency, merged[residency], timeout=self._timeout)
                )
            yield
