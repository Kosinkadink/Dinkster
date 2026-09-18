"""Worker memory relay: the parent-side face of consumers living in a child
process (DESIGN 3.10, open-questions follow-up (a)).

A MemoryGovernor in the engine process cannot see a ResidentPool inside an
isolated worker. This module presents the child's governed consumers to the
parent's governor as ordinary Shedders:

- **Snapshot, not round trips, for the synchronous surface.** ``footprint()``
  runs on every admission and ``details()`` serves status queries - both are
  synchronous and must never await a boundary crossing. The child pushes
  ``memoryReport`` frames (footprints plus detail items) at state-changing
  moments: right after hello, after every invocation, and inside every shed
  reply. The parent serves both methods from that snapshot; staleness bounded
  by push points is the honest contract, and measured free memory remains
  ground truth beside it.
- **shed() is the live surface.** ``vram:*`` pressure crosses as one
  advisory round trip: unloading preserves resident identity, so the worst
  transport race is a model reloading later. ``ram`` pressure crosses as a
  *gated two-phase release* (DESIGN 3.10) - and only when the composing host
  supplied a ReleaseGuard; without one it is refused (returns 0), because
  release drops the child's strong reference while parent caches and live
  runs may still hold stubs. The gate, per candidate the child proposes:
  every guard invalidator drops its cache entries referencing the resource
  (a failing invalidator disqualifies the candidate); then the resource is
  atomically *condemned* in the guard's pins - refused if a live run has
  it pinned, and once condemned no new pin can succeed, so a reference in
  flight across a suspension point is detected (its pin returns False) and
  recomputed rather than dangled. Only then does the commit cross; the
  child re-checks its use-clock token and its own in-flight result holds,
  and the reply names what was actually released - refusals are absolved
  (references stay valid), confirmations stay condemned forever (resource
  ids are never reused). Every reply carries a fresh snapshot applied
  *before* the proxy returns, so the governor's admission rescore sees the
  freed bytes immediately.
- **A dead worker holds nothing and frees nothing.** Once the relay closes,
  footprints read 0, details are empty, and shed returns 0 - never a claim
  of bytes the transport cannot vouch for.

Device facts translate at the boundary like everything else (devices.py):
report keys arrive in the child's namespace and are stored in the parent's;
outgoing shed pressure inverts the map, and a parent device with no child
counterpart sheds nothing rather than the wrong silicon.
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from dinkster_memory import ConsumerItem, PageMap, PressureSignal, Shedder
from dinkster_values import ResourcePins

from .devices import DeviceMap

SendHeader = Callable[[dict[str, object]], Awaitable[None]]
"""How the relay reaches the child: send one frame header (no blobs)."""

_RELEASE_TIMEOUT_S = 60.0
"""Backstop for release round trips - generous (releasing drops references,
it does not copy bytes), but finite: an unanswerable request must resolve
as 'nothing provably happened', never wedge the governor."""

InvalidateFn = Callable[[str], object]
"""Drop every cache entry referencing a resource id; raising disqualifies
that resource's release (the return value is ignored). The same shape the
in-process ResidentPool takes - ``MemoryLRUCache.drop_referencing`` fits."""


@dataclass(frozen=True)
class ReleaseGuard:
    """The parent-side half of the cross-process ram-release gate.

    ``pins`` must be the same registry the Engine pins run-held envelopes
    into, and ``invalidators`` must cover every parent-side cache that can
    hand out stubs (each one's ``drop_referencing``). Wiring only part of
    it re-opens the dangling-stub hole - compose all three (engine pins,
    cache invalidators, this guard) from one place.
    """

    pins: ResourcePins
    invalidators: tuple[InvalidateFn, ...]


@dataclass(frozen=True)
class WorkerFullReleaseResult:
    status: Literal["complete", "busy", "unsupported", "error"]
    consumers: tuple[dict[str, object], ...]
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status == "error":
            if not self.error:
                raise ValueError("an error worker result requires an error message")
        elif self.error is not None:
            raise ValueError("only an error worker result may carry an error message")


def _wire_status(wire: Mapping[str, Any], allowed: set[str]) -> tuple[str, str | None] | None:
    status = wire.get("status")
    error = wire.get("error")
    if not isinstance(status, str) or status not in allowed:
        return None
    if status == "error":
        if not isinstance(error, str) or not error:
            return None
        return status, error
    if error is not None:
        return None
    return status, None


def consumer_item_to_wire(item: ConsumerItem) -> dict[str, object]:
    wire: dict[str, object] = {
        "itemId": item.item_id,
        "displayName": item.display_name,
        "bytesByResidency": dict(item.bytes_by_residency),
    }
    if item.pages is not None:
        wire["pages"] = {
            "pageBytes": item.pages.page_bytes,
            "flags": list(item.pages.flags),
        }
    return wire


def consumer_item_from_wire(
    wire: Mapping[str, Any], residency: Callable[[str], str]
) -> ConsumerItem:
    """Decode one item, translating residency-class keys into the parent's
    device namespace as they cross (colliding keys accumulate - honesty
    over precision if a map ever folds two child devices together)."""
    bytes_by_residency: dict[str, int] = {}
    for key, nbytes in cast("Mapping[str, object]", wire.get("bytesByResidency", {})).items():
        parent_key = residency(str(key))
        bytes_by_residency[parent_key] = bytes_by_residency.get(parent_key, 0) + int(
            cast("int", nbytes)
        )
    pages_wire = wire.get("pages")
    pages = None
    if isinstance(pages_wire, Mapping):
        pages_map = cast("Mapping[str, Any]", pages_wire)
        pages = PageMap(
            page_bytes=int(pages_map["pageBytes"]),
            flags=tuple(int(flag) for flag in pages_map["flags"]),
        )
    return ConsumerItem(
        item_id=str(wire["itemId"]),
        display_name=str(wire["displayName"]),
        bytes_by_residency=bytes_by_residency,
        pages=pages,
    )


@dataclass
class _ConsumerSnapshot:
    footprints: dict[str, int] = field(default_factory=dict[str, int])
    items: tuple[ConsumerItem, ...] = ()


class MemoryRelay:
    """Snapshot store and shed transport for one worker's consumers.

    Owned by IsolatedWorker; the proxies it hands out are what actually
    register with the governor. ``close()`` makes every proxy inert - it is
    called both on orderly shutdown and when the read loop discovers the
    child died, so a stale snapshot can never outlive its process.
    """

    def __init__(
        self,
        send: SendHeader,
        device_map: DeviceMap | None,
        release_guard: ReleaseGuard | None = None,
    ) -> None:
        self._send = send
        self._device_map = device_map
        self._release_guard = release_guard
        self._snapshots: dict[str, _ConsumerSnapshot] = {}
        self._pending: dict[str, asyncio.Future[int]] = {}
        # Release round trips (query, commit): requestId -> reply header.
        # None resolution means the transport died - never a claim of bytes.
        self._replies: dict[str, asyncio.Future[Mapping[str, Any] | None]] = {}
        self._maintenance_replies: dict[str, asyncio.Future[Mapping[str, Any] | None]] = {}
        self._ids = itertools.count(1)
        self._alive = True

    # -- namespace translation ------------------------------------------

    def _parent_residency(self, residency: str) -> str:
        if self._device_map is None:
            return residency
        return self._device_map.residency(residency)

    def to_child_residency(self, residency: str) -> str | None:
        """Invert the device map for outgoing pressure. None means the
        parent device has no counterpart in the child's namespace - a
        pinned worker cannot be asked to shed silicon it cannot see, and
        passing the string through would target the *wrong* device."""
        if self._device_map is None:
            return residency
        return self._device_map.to_child_residency(residency)

    # -- snapshot --------------------------------------------------------

    def apply_report(self, consumers: Mapping[str, Any]) -> None:
        for name, body_raw in consumers.items():
            if not isinstance(body_raw, Mapping):
                continue
            body = cast("Mapping[str, Any]", body_raw)
            footprints: dict[str, int] = {}
            for key, nbytes in cast("Mapping[str, object]", body.get("footprints", {})).items():
                parent_key = self._parent_residency(str(key))
                footprints[parent_key] = footprints.get(parent_key, 0) + int(cast("int", nbytes))
            items = tuple(
                consumer_item_from_wire(cast("Mapping[str, Any]", wire), self._parent_residency)
                for wire in cast("Sequence[object]", body.get("items", ()))
                if isinstance(wire, Mapping)
            )
            self._snapshots[str(name)] = _ConsumerSnapshot(footprints, items)

    def footprint(self, consumer: str, device: str) -> int:
        snapshot = self._snapshots.get(consumer)
        if snapshot is None:
            return 0
        return snapshot.footprints.get(device, 0)

    def details(self, consumer: str) -> tuple[ConsumerItem, ...]:
        snapshot = self._snapshots.get(consumer)
        if snapshot is None:
            return ()
        return snapshot.items

    # -- shedding ----------------------------------------------------------

    async def shed(self, consumer: str, pressure: PressureSignal) -> int:
        if not self._alive:
            return 0
        if not pressure.device.startswith("vram:"):
            # ram release drops the child's strong reference: it may only
            # cross through the two-phase gate, and only when the composing
            # host wired one (DESIGN 3.10).
            return await self._shed_release(consumer, pressure)
        child_device = self.to_child_residency(pressure.device)
        if child_device is None:
            return 0
        request_id = f"shed{next(self._ids)}"
        future: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        header: dict[str, object] = {
            "type": "memoryShed",
            "requestId": request_id,
            "consumer": consumer,
            "device": child_device,
            "bytesNeeded": pressure.bytes_needed,
        }
        if pressure.items is not None:
            header["items"] = list(pressure.items)
        try:
            await self._send(header)
            return await future
        except Exception:  # noqa: BLE001 - transport failure must not claim bytes
            self._pending.pop(request_id, None)
            return 0

    def on_shed_result(self, header: Mapping[str, Any]) -> None:
        # Fresh snapshot first: the governor rescores admission the moment
        # the proxy returns, and it must see the freed bytes.
        consumers = header.get("consumers")
        if isinstance(consumers, Mapping):
            self.apply_report(cast("Mapping[str, Any]", consumers))
        future = self._pending.pop(str(header.get("requestId")), None)
        if future is not None and not future.done():
            future.set_result(int(cast("int", header.get("freedBytes", 0))))

    # -- ram release (the two-phase gate; DESIGN 3.10) ----------------------

    async def _shed_release(self, consumer: str, pressure: PressureSignal) -> int:
        guard = self._release_guard
        if guard is None:
            return 0  # no gate wired: refusing beats a dangling stub
        child_device = self.to_child_residency(pressure.device)
        if child_device is None:
            return 0
        query: dict[str, object] = {
            "type": "memoryReleaseQuery",
            "consumer": consumer,
            "device": child_device,
            "bytesNeeded": pressure.bytes_needed,
        }
        if pressure.items is not None:
            query["items"] = list(pressure.items)
        reply = await self._round_trip(query)
        if reply is None:
            return 0
        # The gate, per candidate: invalidate every parent cache that could
        # hand out a stub, then atomically condemn - which fails if a live
        # run has the resource pinned, and once taken makes every later
        # pin() return False, so nothing can slip in between this check and
        # the child's drop.
        safe: list[dict[str, object]] = []
        condemned: list[str] = []
        for wire_raw in cast("Sequence[object]", reply.get("candidates", ())):
            if not isinstance(wire_raw, Mapping):
                continue
            wire = cast("Mapping[str, Any]", wire_raw)
            resource_id = str(wire.get("resourceId", ""))
            if not resource_id:
                continue
            disqualified = False
            for invalidate in guard.invalidators:
                try:
                    invalidate(resource_id)
                except Exception:  # noqa: BLE001 - a cache that may still
                    disqualified = True  # hold a stub beats reclaimed bytes
                    break
            if disqualified or not guard.pins.condemn(resource_id):
                continue
            condemned.append(resource_id)
            safe.append(
                {
                    "itemId": str(wire.get("itemId", "")),
                    "resourceId": resource_id,
                    "nbytes": int(wire.get("nbytes", 0)),
                    "token": str(wire.get("token", "")),
                }
            )
        if not safe:
            return 0
        reply = await self._round_trip(
            {
                "type": "memoryReleaseCommit",
                "consumer": consumer,
                "device": child_device,
                "candidates": safe,
            }
        )
        if reply is None:
            # Transport died (or timed out) mid-commit: the child's state
            # is unknown, so the condemnations stand - a spurious recompute
            # beats a dangling stub - and no bytes are claimed. Cancellation
            # raising out of the round trip lands in the same conservative
            # place: condemned-forever, zero claimed.
            return 0
        released = {str(rid) for rid in cast("Sequence[object]", reply.get("released", ()))}
        for resource_id in condemned:
            if resource_id not in released:
                # The child refused (used since proposal): references stay
                # valid, so the tombstone is rolled back.
                guard.pins.absolve(resource_id)
        return int(cast("int", reply.get("freedBytes", 0)))

    async def _round_trip(self, header: dict[str, object]) -> Mapping[str, Any] | None:
        """One release-protocol round trip; None on any transport failure
        (the caller must treat that as 'nothing provably happened')."""
        if not self._alive:
            return None
        request_id = f"rel{next(self._ids)}"
        header["requestId"] = request_id
        future: asyncio.Future[Mapping[str, Any] | None] = (
            asyncio.get_running_loop().create_future()
        )
        self._replies[request_id] = future
        try:
            await self._send(header)
            # The timeout is a liveness backstop: a live transport whose
            # reply was lost (crashed release task in the child) must not
            # hang shedding - and the governor behind it - forever.
            return await asyncio.wait_for(future, timeout=_RELEASE_TIMEOUT_S)
        except Exception:  # noqa: BLE001 - transport failure must not claim bytes
            return None
        finally:
            # Also on cancellation: a stranded future entry would be
            # resolved-but-orphaned at close, and the dict would grow.
            self._replies.pop(request_id, None)

    def on_release_reply(self, header: Mapping[str, Any]) -> None:
        """Route a memoryReleaseCandidates or memoryReleaseResult frame to
        its round trip. A result frame carries a fresh snapshot; apply it
        before resolving, so the governor's rescore sees freed bytes."""
        consumers = header.get("consumers")
        if isinstance(consumers, Mapping):
            self.apply_report(cast("Mapping[str, Any]", consumers))
        request_id = str(header.get("requestId"))
        future = self._replies.pop(request_id, None)
        if future is None:
            future = self._maintenance_replies.pop(request_id, None)
        if future is not None and not future.done():
            future.set_result(header)

    async def full_release(
        self,
        request_id: str,
        worker_instance: str,
        *,
        release_guard: ReleaseGuard | None = None,
    ) -> WorkerFullReleaseResult | None:
        """Release every declared consumer through the remote reference gate."""
        guard = release_guard or self._release_guard
        if not self._alive or guard is None:
            return None
        try:
            query, query_timed_out = await self._maintenance_round_trip(
                {
                    "type": "memoryFreeQuery",
                    "operationRequestId": request_id,
                    "workerInstance": worker_instance,
                }
            )
        except asyncio.CancelledError:
            await self._abort_full_release(request_id, worker_instance)
            raise
        if query is None:
            return None
        if query.get("type") != "memoryFreeCandidates":
            await self._abort_full_release(request_id, worker_instance)
            return None

        async def abort_query() -> None:
            await self._abort_full_release(request_id, worker_instance)

        query_worker = _wire_status(query, {"complete", "busy", "unsupported", "error"})
        if query_worker is None:
            await abort_query()
            return None
        query_worker_status, query_worker_error = query_worker
        if (
            query.get("operationRequestId") != request_id
            or query.get("workerInstance") != worker_instance
        ):
            await abort_query()
            return None
        if query_worker_status in {"unsupported", "error"}:
            return WorkerFullReleaseResult(
                cast("Literal['unsupported', 'error']", query_worker_status),
                (),
                query_worker_error,
            )
        commit_consumers: list[dict[str, object]] = []
        gate_results: dict[str, dict[str, object]] = {}
        condemned: dict[str, set[str]] = {}
        raw_consumers = query.get("consumers", ())
        if not isinstance(raw_consumers, Sequence) or isinstance(raw_consumers, (str, bytes)):
            await abort_query()
            return None
        parsed: list[tuple[str, str, tuple[dict[str, object], ...], str | None]] = []
        names: set[str] = set()
        for raw in cast("Sequence[object]", raw_consumers):
            if not isinstance(raw, Mapping):
                await abort_query()
                return None
            wire = cast("Mapping[str, Any]", raw)
            name = wire.get("consumer")
            parsed_status = _wire_status(wire, {"ready", "busy", "unsupported", "error"})
            if not isinstance(name, str) or not name or name in names or parsed_status is None:
                await abort_query()
                return None
            status, query_error = parsed_status
            names.add(name)
            candidates_raw = wire.get("candidates", ())
            if not isinstance(candidates_raw, Sequence) or isinstance(candidates_raw, (str, bytes)):
                await abort_query()
                return None
            candidates: list[dict[str, object]] = []
            for candidate_raw in cast("Sequence[object]", candidates_raw):
                if not isinstance(candidate_raw, Mapping):
                    await abort_query()
                    return None
                candidate = cast("Mapping[str, Any]", candidate_raw)
                item_id = candidate.get("itemId")
                resource_id = candidate.get("resourceId")
                nbytes = candidate.get("nbytes", 0)
                token = candidate.get("token")
                if (
                    not isinstance(item_id, str)
                    or not item_id
                    or not isinstance(resource_id, str)
                    or not resource_id
                    or type(nbytes) is not int
                    or nbytes < 0
                    or not isinstance(token, str)
                    or not token
                ):
                    await abort_query()
                    return None
                candidates.append(
                    {
                        "itemId": item_id,
                        "resourceId": resource_id,
                        "nbytes": nbytes,
                        "token": token,
                    }
                )
            if status in {"unsupported", "error"} and candidates:
                await abort_query()
                return None
            parsed.append((name, status, tuple(candidates), query_error))
        for name, status, candidate_wires, query_error in parsed:
            if status in {"busy", "unsupported", "error"}:
                result: dict[str, object] = {
                    "consumer": name,
                    "status": status,
                }
                if status == "error":
                    result["error"] = query_error or "consumer query failed"
                gate_results[name] = result
                approved: list[dict[str, object]] = []
            else:
                approved = []
                gate_status = "complete"
                gate_error: str | None = None
                for candidate in candidate_wires:
                    resource_id = str(candidate["resourceId"])
                    try:
                        for invalidate in guard.invalidators:
                            invalidate(resource_id)
                    except Exception as exc:  # noqa: BLE001 - incomplete beats unsafe release
                        gate_status = "error"
                        gate_error = f"cache invalidation failed: {exc}"
                        continue
                    if not guard.pins.condemn(resource_id):
                        if gate_status != "error":
                            gate_status = "busy"
                        continue
                    approved.append(candidate)
                    condemned.setdefault(name, set()).add(resource_id)
                gate_result: dict[str, object] = {
                    "consumer": name,
                    "status": gate_status,
                }
                if gate_error is not None:
                    gate_result["error"] = gate_error
                gate_results[name] = gate_result
            commit_consumers.append(
                {"consumer": name, "queryStatus": status, "candidates": approved}
            )

        commit, commit_timed_out = await self._maintenance_round_trip(
            {
                "type": "memoryFreeCommit",
                "operationRequestId": request_id,
                "workerInstance": worker_instance,
                "workerStatus": query_worker_status,
                "consumers": commit_consumers,
            }
        )
        if commit is None or commit.get("type") != "memoryFreeResult":
            return None
        if (
            commit.get("operationRequestId") != request_id
            or commit.get("workerInstance") != worker_instance
        ):
            return None
        commit_worker = _wire_status(commit, {"complete", "busy", "unsupported", "error"})
        if commit_worker is None:
            return None
        worker_status, worker_error = commit_worker
        raw_results = commit.get("consumers", ())
        if not isinstance(raw_results, Sequence) or isinstance(raw_results, (str, bytes)):
            return None
        child_results: dict[str, tuple[str, set[str], str | None]] = {}
        for raw in cast("Sequence[object]", raw_results):
            if not isinstance(raw, Mapping):
                return None
            child_wire = cast("Mapping[str, Any]", raw)
            name = child_wire.get("consumer")
            if not isinstance(name, str) or not name or name in child_results:
                return None
            parsed_status = _wire_status(child_wire, {"complete", "busy", "unsupported", "error"})
            released_raw = child_wire.get("released", ())
            if (
                parsed_status is None
                or not isinstance(released_raw, Sequence)
                or isinstance(released_raw, (str, bytes))
            ):
                return None
            status, child_error = parsed_status
            released_items = tuple(cast("Sequence[object]", released_raw))
            if any(not isinstance(item, str) or not item for item in released_items):
                return None
            released = set(cast("tuple[str, ...]", released_items))
            if len(released) != len(released_items) or not released <= condemned.get(name, set()):
                return None
            child_results[name] = (
                status,
                released,
                child_error,
            )
        if set(child_results) != set(gate_results):
            return None
        results: list[dict[str, object]] = []
        for name, gate in gate_results.items():
            child = child_results.get(name)
            assert child is not None
            child_status, released, child_error = child
            # An error may follow a partial release without a complete list
            # of freed resources. Only a validated refusal proves survival.
            if child_status in {"busy", "unsupported"}:
                for resource_id in condemned.get(name, set()) - released:
                    guard.pins.absolve(resource_id)
            if gate["status"] in {"busy", "error", "unsupported"}:
                results.append(gate)
                continue
            result = {"consumer": name, "status": child_status}
            if child_status == "error":
                result["error"] = child_error or "consumer release failed"
            results.append(result)
        if query_timed_out or commit_timed_out:
            return WorkerFullReleaseResult("error", tuple(results), "worker full release timed out")
        return WorkerFullReleaseResult(
            cast(
                "Literal['complete', 'busy', 'unsupported', 'error']",
                worker_status,
            ),
            tuple(results),
            worker_error,
        )

    async def _maintenance_round_trip(
        self, header: dict[str, object]
    ) -> tuple[Mapping[str, Any] | None, bool]:
        if not self._alive:
            return None, False
        request_id = f"free{next(self._ids)}"
        header["requestId"] = request_id
        future: asyncio.Future[Mapping[str, Any] | None] = (
            asyncio.get_running_loop().create_future()
        )
        self._maintenance_replies[request_id] = future
        try:
            send_cancelled = False
            send_failed = False
            try:
                await self._send(header)
            except asyncio.CancelledError:
                send_cancelled = True
            except Exception:  # noqa: BLE001 - delivery may have preceded drain failure
                send_failed = True
            if send_cancelled or send_failed:
                result, settlement_cancelled = await self._wait_until_done(future)
                if send_cancelled or settlement_cancelled:
                    raise asyncio.CancelledError from None
                return result, False
            try:
                return (
                    await asyncio.wait_for(asyncio.shield(future), timeout=_RELEASE_TIMEOUT_S),
                    False,
                )
            except TimeoutError:
                result, cancelled = await self._wait_until_done(future)
                if cancelled:
                    raise asyncio.CancelledError from None
                return result, True
            except asyncio.CancelledError:
                await self._wait_until_done(future)
                raise
        except Exception:  # noqa: BLE001 - incomplete beats an unproven release
            return None, False
        finally:
            self._maintenance_replies.pop(request_id, None)

    @staticmethod
    async def _wait_until_done(
        future: asyncio.Future[Mapping[str, Any] | None],
    ) -> tuple[Mapping[str, Any] | None, bool]:
        cancelled = False
        while not future.done():
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError:
                cancelled = True
        return future.result(), cancelled

    async def _abort_full_release(self, operation_id: str, worker_instance: str) -> bool:
        reply, _ = await self._maintenance_round_trip(
            {
                "type": "memoryFreeAbort",
                "operationRequestId": operation_id,
                "workerInstance": worker_instance,
            }
        )
        parsed = None if reply is None else _wire_status(reply, {"complete", "error"})
        return bool(
            reply is not None
            and reply.get("type") == "memoryFreeAborted"
            and reply.get("operationRequestId") == operation_id
            and reply.get("workerInstance") == worker_instance
            and parsed == ("complete", None)
        )

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """The child is gone (or going): footprints to zero, pending sheds
        resolve to zero freed. Idempotent - the read loop and close() both
        call it."""
        self._alive = False
        self._snapshots.clear()
        for future in self._pending.values():
            if not future.done():
                future.set_result(0)
        self._pending.clear()
        for reply in self._replies.values():
            if not reply.done():
                reply.set_result(None)
        self._replies.clear()
        # A dropped transport does not prove that process-local cleanup has
        # stopped. Maintenance replies remain pending until the owning
        # process reports a terminal outcome or is known dead.

    async def drain_releases(self) -> None:
        pending = tuple(reply for reply in self._maintenance_replies.values() if not reply.done())
        if not pending:
            return
        settled = asyncio.gather(*pending)
        cancelled = False
        while not settled.done():
            try:
                await asyncio.shield(settled)
            except asyncio.CancelledError:
                cancelled = True
        settled.result()
        if cancelled:
            raise asyncio.CancelledError

    def process_died(self) -> None:
        """Settle exchanges only after the resource-owning process exited."""
        self.close()
        for reply in self._maintenance_replies.values():
            if not reply.done():
                reply.set_result(None)
        self._maintenance_replies.clear()

    # -- proxies -----------------------------------------------------------

    def proxy(self, consumer: str, *, detailed: bool) -> Shedder:
        """One governed face per child consumer. Detail-contract membership
        mirrors the child's: a proxy for a consumer without details must
        not satisfy DetailedConsumer, or the governor would route
        item-targeted pressure it can never resolve."""
        if detailed:
            return _DetailedRelayedConsumer(self, consumer)
        return _RelayedConsumer(self, consumer)


class _RelayedConsumer:
    def __init__(self, relay: MemoryRelay, consumer: str) -> None:
        self._relay = relay
        self._consumer = consumer

    def footprint(self, device: str) -> int:
        return self._relay.footprint(self._consumer, device)

    async def shed(self, pressure: PressureSignal) -> int:
        return await self._relay.shed(self._consumer, pressure)


class _DetailedRelayedConsumer(_RelayedConsumer):
    def details(self) -> Sequence[ConsumerItem]:
        return self._relay.details(self._consumer)
