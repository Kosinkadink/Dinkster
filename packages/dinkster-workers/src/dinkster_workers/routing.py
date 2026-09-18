"""RoutingWorker: placement policy, first cut (DESIGN 3.3).

The engine takes one Worker; placement is a Worker that routes by node
type. In-process nodes, an isolated pack, and (later) a remote machine
compose behind one protocol - the engine stays unchanged (hazard H3), and
where a node runs is configuration, never node code (hazard H9).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

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


class RoutingWorker:
    def __init__(self, routes: Mapping[str, Worker], *, default: Worker | None = None) -> None:
        """``routes`` maps node type -> Worker; ``default`` catches the rest."""
        self._routes = dict(routes)
        self._default = default

    def add_routes(self, routes: Mapping[str, Worker]) -> None:
        """Additively route new node types (progressive pack announcement).

        Strictly additive, mirroring Engine.announce_schemas: re-routing an
        already-routed type is refused - where a node runs is configuration,
        and configuration conflicts fail loudly, never last-write-wins. The
        merged mapping lands in one reference swap so a concurrent
        invocation resolves against a consistent view.
        """
        for node_type in routes:
            if node_type in self._routes:
                raise ValueError(
                    f"add_routes would re-route node type {node_type!r}; "
                    "routes grow additively, never in place"
                )
        self._routes = {**self._routes, **routes}

    def has_route(self, node_type: str) -> bool:
        """Return whether a node type has an explicit worker route."""
        return node_type in self._routes

    def swap_routes(self, remove: Sequence[str], add: Mapping[str, Worker]) -> None:
        """Atomically replace one pack's routes (hot reload, DESIGN 3.9).

        The reload seam's routing half: the removed types and the added
        types land in ONE reference swap, so a concurrent invocation
        resolves against either the old view or the new one, never a
        half-swapped mapping. Validation mirrors add_routes: every removed
        type must be routed, and an added type may collide only with a
        type being removed in the same swap (the reloaded pack's own).
        Removal without re-add is legal - a reloaded pack may drop node
        types; invocations of a dropped type fail loudly at _worker_for.
        """
        removed = set(remove)
        for node_type in removed:
            if node_type not in self._routes:
                raise ValueError(f"swap_routes would remove unrouted node type {node_type!r}")
        for node_type in add:
            if node_type in self._routes and node_type not in removed:
                raise ValueError(
                    f"swap_routes would re-route node type {node_type!r}, "
                    "which is not being removed in this swap"
                )
        merged = {t: w for t, w in self._routes.items() if t not in removed}
        merged.update(add)
        self._routes = merged

    def _worker_for(self, node_type: str) -> Worker:
        worker = self._routes.get(node_type, self._default)
        if worker is None:
            raise KeyError(f"no worker routes node type: {node_type}")
        return worker

    async def prepare(self, node_types: Sequence[str]) -> None:
        groups: dict[int, tuple[Worker, list[str]]] = {}
        for node_type in node_types:
            worker = self._worker_for(node_type)
            groups.setdefault(id(worker), (worker, []))[1].append(node_type)
        for worker, types in groups.values():
            await worker.prepare(types)

    async def invoke(
        self,
        invocation: Invocation,
        on_event: OnInvocationEvent | None = None,
    ) -> InvocationResult:
        return await self._worker_for(invocation.node_type).invoke(invocation, on_event=on_event)

    async def check_lazy_status(
        self,
        invocation: LazyStatusInvocation,
        on_event: OnInvocationEvent | None = None,
    ) -> LazyStatusResult:
        worker = self._worker_for(invocation.node_type)
        hook = getattr(worker, "check_lazy_status", None)
        if not callable(hook):
            return LazyStatusResult(
                error=NodeError(
                    invocation.node_id,
                    invocation.node_type,
                    "lazy-protocol-skew: routed worker does not support lazy status",
                )
            )
        return await cast("LazyStatusWorker", worker).check_lazy_status(
            invocation, on_event=on_event
        )
