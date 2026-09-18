"""MemoryGovernor: budgets are declared, costs are accounted, eviction has
one arbiter (hazard H13).

ComfyUI's model_management fails because memory decisions are distributed
guesses - every caller frees-and-hopes. The governor inverts that:

- **Devices are residency classes**: ``"ram"``, ``"disk"``,
  ``"vram:cuda:0"`` - the same keys values carry in their COST_META_KEY
  metadata, so accounting is observation of envelopes, not instrumentation
  in nodes.
- **Reservation before allocation.** A worker about to materialize
  something large asks first; the governor sheds registered consumers
  (caches, idle resource pools) or delays until the reservation fits.
  This is what makes *parallel* workflows safe on one GPU: two runs
  cannot both believe the same 4 GB is free.
- **Shedding flows through one place.** Consumers register as Shedders
  with a priority; the governor is the only thing that tells a cache to
  evict.

An unbudgeted device is tracked but never blocks: declared budgets act as caps
only where the operator declares them. A fresh effective-RAM read may trigger
one best-effort shedding pass before an unbudgeted RAM grant, but measured
availability never enters the grant/refusal math. Cross-instance coordination
(the /memory endpoints) is a server-layer client of exactly this object.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from collections.abc import (
    AsyncGenerator,
    Callable,
    Collection,
    Iterable,
    Mapping,
    Sequence,
)
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

from .details import ConsumerItem, DetailedConsumer
from .system import SystemMemorySnapshot, system_memory_snapshot

log = logging.getLogger("dinkster.memory.governor")


class BudgetExceeded(Exception):
    """The request can never fit: it exceeds the device budget outright."""


class ReservationTimeout(Exception):
    """The reservation did not fit before the caller's deadline."""


@dataclass(frozen=True)
class PressureSignal:
    """A request to free memory: which device, and how many bytes short.

    ``items`` narrows the request to specific things the consumer holds
    (stable IDs from its detail contract) - how "unload this model" is
    expressed. None means anything sheddable qualifies.
    """

    device: str
    bytes_needed: int
    items: tuple[str, ...] | None = None


@dataclass(frozen=True)
class MeasuredMemory:
    """What the device itself reports: ground truth beside declared budgets.

    The two disagreeing is signal, not error - a non-Dinkster process hogging
    the GPU shows up here and nowhere in the ledger. Optional accelerator
    components preserve attribution without changing the measured-free
    admission rule.
    """

    free_bytes: int
    total_bytes: int
    driver_free_bytes: int | None = None
    allocator_reclaimable_bytes: int | None = None
    dynamic_evictable_bytes: int | None = None
    dynamic_pinned_bytes: int | None = None

    _COMPONENT_FIELDS = (
        ("driver_free_bytes", "driverFreeBytes"),
        ("allocator_reclaimable_bytes", "allocatorReclaimableBytes"),
        ("dynamic_evictable_bytes", "dynamicEvictableBytes"),
        ("dynamic_pinned_bytes", "dynamicPinnedBytes"),
    )

    def to_wire(self) -> dict[str, int] | None:
        """Validate and encode one additive measured-memory wire value."""
        if (
            type(self.free_bytes) is not int
            or type(self.total_bytes) is not int
            or self.free_bytes < 0
            or self.total_bytes <= 0
            or self.free_bytes > self.total_bytes
        ):
            return None
        wire = {"freeBytes": self.free_bytes, "totalBytes": self.total_bytes}
        for attribute, name in self._COMPONENT_FIELDS:
            value = getattr(self, attribute)
            if value is None:
                continue
            if type(value) is not int or value < 0:
                return None
            wire[name] = value
        return wire

    @classmethod
    def from_wire(cls, value: object) -> MeasuredMemory | None:
        """Decode known fields and ignore additive fields from newer workers."""
        if not isinstance(value, Mapping):
            return None
        fields = cast("Mapping[object, object]", value)
        free = fields.get("freeBytes")
        total = fields.get("totalBytes")
        if (
            type(free) is not int
            or type(total) is not int
            or free < 0
            or total <= 0
            or free > total
        ):
            return None
        components: dict[str, int] = {}
        for attribute, name in cls._COMPONENT_FIELDS:
            if name not in fields:
                continue
            component = fields[name]
            if type(component) is not int or component < 0:
                return None
            components[attribute] = component
        return cls(free_bytes=free, total_bytes=total, **components)


TelemetryProbe = Callable[[str], "MeasuredMemory | None"]
"""Measure one device (residency-class key), None when unmeasurable."""


@dataclass(frozen=True)
class Reservation:
    reservation_id: str
    device: str
    nbytes: int


@runtime_checkable
class Shedder(Protocol):
    """A governed consumer of memory: something that holds bytes it could
    give back (a cache store, an idle-model pool).

    Shedders must never call ``MemoryGovernor.reserve`` from ``shed`` -
    shedding runs *inside* the admission path, and a shedder that reserves
    would wait on itself.
    """

    def footprint(self, device: str) -> int:
        """Bytes currently held on a device. Cheap; called on every admission."""
        ...

    async def shed(self, pressure: PressureSignal) -> int:
        """Free up to ``pressure.bytes_needed`` bytes; return bytes actually freed."""
        ...


@dataclass(frozen=True)
class _RegisteredShedder:
    order: int
    priority: int
    name: str
    shedder: Shedder


class MemoryGovernor:
    """Per-instance arbiter of memory budgets, reservations, and shedding."""

    def __init__(
        self,
        budgets: Mapping[str, int] | None = None,
        *,
        telemetry: TelemetryProbe | None = None,
        telemetry_devices: Callable[[], Iterable[str]] | None = None,
        system_memory: Callable[[], SystemMemorySnapshot] = system_memory_snapshot,
    ) -> None:
        self._budgets: dict[str, int] = dict(budgets or {})
        for device, nbytes in self._budgets.items():
            if nbytes < 0:
                raise ValueError(f"budget for {device!r} must be >= 0")
        self._telemetry = telemetry
        # The measured device universe (e.g. ReportedTelemetry.devices):
        # status() unions these in so a measured but unbudgeted device is
        # visible without inventing a budget. Worker telemetry is enumeration
        # and status only. The separate local system snapshot can trigger
        # best-effort RAM shedding, but measured free stays out of _fits.
        self._telemetry_devices = telemetry_devices
        self._system_memory = system_memory
        self._reserved: dict[str, int] = {}
        self._reservations: dict[str, Reservation] = {}
        self._shedders: list[_RegisteredShedder] = []
        self._reserved_listeners: list[Callable[[str, int], None]] = []
        self._cond = asyncio.Condition()
        self._ids = itertools.count(1)
        self._order = itertools.count()

    # -- configuration -------------------------------------------------

    def set_budget(self, device: str, nbytes: int) -> None:
        if nbytes < 0:
            raise ValueError(f"budget for {device!r} must be >= 0")
        self._budgets[device] = nbytes

    def register_shedder(self, shedder: Shedder, *, priority: int = 0, name: str = "") -> None:
        """Lower priority sheds first: caches before idle resource pools.

        Ties shed in registration order.
        """
        self._shedders.append(
            _RegisteredShedder(
                order=next(self._order),
                priority=priority,
                name=name or type(shedder).__name__,
                shedder=shedder,
            )
        )
        self._shedders.sort(key=lambda r: (r.priority, r.order))

    def unregister_shedder(self, shedder: Shedder) -> None:
        """Remove a consumer whose bytes are gone (a closed worker's relay
        proxies). Matches by object identity, and unknown shedders are
        ignored - unregistration runs from cleanup paths that must not
        raise over already-dead state."""
        self._shedders = [r for r in self._shedders if r.shedder is not shedder]

    def subscribe_reserved(self, callback: Callable[[str, int], None]) -> None:
        """Observe committed reserved-total changes outside the governor lock."""
        self._reserved_listeners.append(callback)

    def _notify_reserved(self, device: str, total: int) -> None:
        for callback in self._reserved_listeners:
            try:
                callback(device, total)
            except Exception:  # noqa: BLE001 - observers cannot break admission
                name = getattr(callback, "__qualname__", repr(callback))
                log.warning("reserved-total listener %s raised", name, exc_info=True)

    # -- accounting ----------------------------------------------------

    def reserved(self, device: str) -> int:
        return self._reserved.get(device, 0)

    def footprint(self, device: str) -> int:
        return sum(r.shedder.footprint(device) for r in self._shedders)

    def available(self, device: str) -> int | None:
        """Bytes admittable right now, or None when the device is unbudgeted."""
        budget = self._budgets.get(device)
        if budget is None:
            return None
        return budget - self.reserved(device) - self.footprint(device)

    def status(self) -> dict[str, dict[str, object]]:
        """One shape for observability and the cross-instance endpoints.

        Per-device ``consumers`` names each registered shedder and its
        footprint - the names /cache/trim targets are discoverable, never
        out-of-band knowledge. ``measured`` is the device's own report
        (free/total) beside the declared view; None when no probe is
        wired or the device is unmeasurable - honest absence, never zero.
        """
        devices = set(self._budgets) | set(self._reserved)
        if self._telemetry_devices is not None:
            try:
                devices.update(self._telemetry_devices())
            except Exception:  # noqa: BLE001 - enumeration is best-effort observability
                pass
        report: dict[str, dict[str, object]] = {}
        for device in sorted(devices):
            report[device] = {
                "budgetBytes": self._budgets.get(device),
                "reservedBytes": self.reserved(device),
                "consumerFootprintBytes": self.footprint(device),
                "availableBytes": self.available(device),
                "measured": self._measure(device),
                "consumers": {r.name: r.shedder.footprint(device) for r in self._shedders},
            }
        return report

    def _measure(self, device: str) -> dict[str, int] | None:
        if self._telemetry is None:
            return None
        try:
            measured = self._telemetry(device)
        except Exception:  # noqa: BLE001 - probes touch drivers; absence beats a crash
            return None
        if not isinstance(measured, MeasuredMemory):
            return None
        return measured.to_wire()

    def _live_ram_shortfall(self, device: str, nbytes: int) -> int:
        """Return immediate RAM pressure without turning it into a cap."""
        if device != "ram" or nbytes <= 0 or not self._shedders:
            return 0
        if self.footprint(device) <= 0:
            return 0
        try:
            available = self._system_memory().effective_available_bytes
        except (AttributeError, OSError, RuntimeError, ValueError):
            return 0
        return max(0, nbytes - available)

    def details(self) -> dict[str, Sequence[ConsumerItem]]:
        """Item-level contents per consumer, for those that expose them.

        Aggregation without interpretation: the governor never reads the
        items, it only serves them. Consumers without the detail contract
        are simply absent - their footprint number remains in status().
        """
        report: dict[str, Sequence[ConsumerItem]] = {}
        for registered in self._shedders:
            if isinstance(registered.shedder, DetailedConsumer):
                report[registered.name] = tuple(registered.shedder.details())
        return report

    # -- shedding ------------------------------------------------------

    async def shed(
        self,
        device: str,
        nbytes: int,
        *,
        consumers: Collection[str] | None = None,
        items: Collection[str] | None = None,
    ) -> int:
        """Ask consumers, in priority order, to free ``nbytes`` on ``device``.

        ``consumers`` restricts shedding to the named shedders (how
        /cache/trim targets caches without unloading idle resource pools);
        None means everyone. ``items`` narrows further to stable item IDs
        and requires naming exactly one consumer - item IDs are one
        consumer's namespace, and a signal a consumer cannot resolve must
        free nothing, so item pressure is only ever sent to a consumer
        exposing the detail contract.

        Returns bytes actually freed, which may fall short: the caller
        (a reservation, a peer instance's /memory/shed) decides what
        falling short means.
        """
        item_tuple: tuple[str, ...] | None = None
        if items is not None:
            if consumers is None or len(consumers) != 1:
                raise ValueError(
                    "item-targeted shedding requires exactly one consumer: "
                    "item IDs are meaningful only within one consumer's "
                    "namespace"
                )
            item_tuple = tuple(items)
        remaining = nbytes
        freed_total = 0
        for registered in self._shedders:
            if remaining <= 0:
                break
            if consumers is not None and registered.name not in consumers:
                continue
            if item_tuple is not None and not isinstance(registered.shedder, DetailedConsumer):
                continue  # cannot resolve item IDs; must not shed instead
            freed = await registered.shedder.shed(
                PressureSignal(device=device, bytes_needed=remaining, items=item_tuple)
            )
            freed_total += freed
            remaining -= freed
        return freed_total

    # -- reservations --------------------------------------------------

    @asynccontextmanager
    async def reserve(
        self, device: str, nbytes: int, *, timeout: float | None = None
    ) -> AsyncGenerator[Reservation]:
        """Admission control for a large allocation.

        A RAM request first makes one best-effort live-pressure shed when useful.
        It then grants when the declared budget fits (or no budget exists).
        Otherwise it sheds consumers and, if that is still not enough, waits for
        other reservations to release. Raises BudgetExceeded when the request
        can never fit and ReservationTimeout when a declared-budget admission
        reaches ``timeout``.
        """
        if nbytes < 0:
            raise ValueError("reservation size must be >= 0")
        reservation = await self._admit(device, nbytes, timeout)
        try:
            yield reservation
        finally:
            await self._release(reservation)

    async def _admit(self, device: str, nbytes: int, timeout: float | None) -> Reservation:
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        budget = self._budgets.get(device)
        live_shortfall = (
            0
            if budget is not None and nbytes > budget
            else self._live_ram_shortfall(device, nbytes)
        )
        shed_attempted = False
        while True:
            granted: Reservation | None = None
            granted_total = 0
            shortfall = 0
            budget_pressure = False
            async with self._cond:
                budget = self._budgets.get(device)
                fits = budget is None or self._fits(device, nbytes, budget)
                if budget is not None and nbytes > budget:
                    raise BudgetExceeded(
                        f"{nbytes} bytes exceeds the {budget}-byte budget for {device!r}"
                    )
                if not shed_attempted and (live_shortfall > 0 or not fits):
                    if budget is not None and not fits:
                        shortfall = self.reserved(device) + self.footprint(device) + nbytes - budget
                        budget_pressure = True
                    shortfall = max(shortfall, live_shortfall)
                elif fits:
                    granted = self._record(device, nbytes)
                    granted_total = self._reserved[device]
                elif shed_attempted:
                    if self.reserved(device) == 0:
                        # Nothing is reserved, so nothing will ever release:
                        # waiting would hang. Consumers already shed what they
                        # could; the request honestly does not fit.
                        raise ReservationTimeout(
                            f"could not reserve {nbytes} bytes on {device!r}: "
                            "consumers shed what they could and no "
                            "reservations are outstanding"
                        )
                    # Wait for a release, then rescore.
                    remaining = None if deadline is None else deadline - loop.time()
                    if remaining is not None and remaining <= 0:
                        raise ReservationTimeout(f"could not reserve {nbytes} bytes on {device!r}")
                    try:
                        await asyncio.wait_for(self._cond.wait(), remaining)
                    except TimeoutError:
                        raise ReservationTimeout(
                            f"could not reserve {nbytes} bytes on {device!r}"
                        ) from None
                    shed_attempted = False  # released memory changes the score
                    continue
            if granted is not None:
                self._notify_reserved(device, granted_total)
                return granted
            # Shed outside the lock: shedders run arbitrary consumer code.
            remaining = None if deadline is None else deadline - loop.time()
            if remaining is not None and remaining <= 0:
                if budget_pressure:
                    raise ReservationTimeout(f"could not reserve {nbytes} bytes on {device!r}")
                live_shortfall = 0
                continue
            try:
                # A timed-out shed is cancelled. Shedder implementations must
                # therefore be cancellation-safe: only report/free bytes for
                # mutations completed before an await, and leave their state
                # internally consistent when CancelledError interrupts them.
                await asyncio.wait_for(self.shed(device, shortfall), remaining)
            except TimeoutError:
                if budget_pressure:
                    raise ReservationTimeout(
                        f"could not reserve {nbytes} bytes on {device!r}"
                    ) from None
            except Exception:
                if budget_pressure:
                    raise
            live_shortfall = 0
            shed_attempted = budget_pressure

    def _fits(self, device: str, nbytes: int, budget: int) -> bool:
        return self.reserved(device) + self.footprint(device) + nbytes <= budget

    def _record(self, device: str, nbytes: int) -> Reservation:
        reservation = Reservation(
            reservation_id=f"r{next(self._ids)}", device=device, nbytes=nbytes
        )
        self._reservations[reservation.reservation_id] = reservation
        self._reserved[device] = self.reserved(device) + nbytes
        return reservation

    async def _release(self, reservation: Reservation) -> None:
        released = False
        async with self._cond:
            if self._reservations.pop(reservation.reservation_id, None) is None:
                return
            self._reserved[reservation.device] -= reservation.nbytes
            self._cond.notify_all()
            released = True
            total = self._reserved[reservation.device]
        if released:
            self._notify_reserved(reservation.device, total)
