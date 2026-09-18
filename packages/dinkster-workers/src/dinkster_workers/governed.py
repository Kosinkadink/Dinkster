"""GovernedWorker: reservation-before-allocation at the worker boundary
(DESIGN 3.10).

The engine's admission lanes bound *execution* (who occupies a device);
memory reservations bound *allocation* (whether bytes fit). This wrapper is
where the second one happens: a pack-declared planner inspects the
invocation's envelopes and names what the execution is about to
materialize; the reservation is held across execute() so two workflows
cannot both believe the same bytes are free.

Node code never sees any of this (hazard H9): the planner is pack policy
keyed on node types and envelope metadata, and denial surfaces as an
ordinary NodeError - the engine reports it like any node failure.
"""

from __future__ import annotations

import traceback
from collections.abc import Sequence
from typing import cast

from dinkster_memory import (
    BudgetExceeded,
    ReservationPlanner,
    ReservationService,
    ReservationTimeout,
)
from dinkster_protocol import (
    Invocation,
    InvocationResult,
    LazyStatusInvocation,
    LazyStatusResult,
    LazyStatusWorker,
    NodeError,
    OnInvocationEvent,
    Worker,
)

__all__ = ["GovernedWorker", "ReservationPlanner"]


class GovernedWorker:
    """Wraps any Worker with memory admission. Composition, not inheritance:
    the inner worker's unwrapping/validation behavior is untouched."""

    def __init__(
        self,
        inner: Worker,
        *,
        planner: ReservationPlanner,
        reservations: ReservationService,
    ) -> None:
        self._inner = inner
        self._planner = planner
        self._reservations = reservations

    async def prepare(self, node_types: Sequence[str]) -> None:
        await self._inner.prepare(node_types)

    async def invoke(
        self,
        invocation: Invocation,
        on_event: OnInvocationEvent | None = None,
    ) -> InvocationResult:
        try:
            requests = tuple(self._planner(invocation))
        except Exception as exc:  # noqa: BLE001 - planner is pack code
            return InvocationResult(
                error=NodeError(
                    node_id=invocation.node_id,
                    node_type=invocation.node_type,
                    message=f"reservation planning failed: {exc}",
                    traceback=traceback.format_exc(),
                )
            )
        if not requests:
            return await self._inner.invoke(invocation, on_event=on_event)
        try:
            async with self._reservations.reserve(requests):
                return await self._inner.invoke(invocation, on_event=on_event)
        except (BudgetExceeded, ReservationTimeout) as exc:
            return InvocationResult(
                error=NodeError(
                    node_id=invocation.node_id,
                    node_type=invocation.node_type,
                    message=f"memory admission failed: {exc}",
                )
            )

    async def check_lazy_status(
        self,
        invocation: LazyStatusInvocation,
        on_event: OnInvocationEvent | None = None,
    ) -> LazyStatusResult:
        if not callable(getattr(self._inner, "check_lazy_status", None)):
            return LazyStatusResult(
                error=NodeError(
                    invocation.node_id,
                    invocation.node_type,
                    "lazy-protocol-skew: governed worker does not support lazy status",
                )
            )
        return await cast("LazyStatusWorker", self._inner).check_lazy_status(
            invocation, on_event=on_event
        )
