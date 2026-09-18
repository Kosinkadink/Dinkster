"""LeaseBroker: governor reservations with handles and TTLs (DESIGN 3.10).

``MemoryGovernor.reserve`` is a context manager - the right shape when the
holder is a stack frame that cannot leak. A peer instance on the other end
of ``POST /memory/reserve`` is not a stack frame: it can crash, hang, or
walk away mid-lease. The broker converts reservations into *leases* -
id-addressed, TTL-bounded - so cross-instance coordination stays
cooperative and advisory by design: a dead peer's lease expires and the
memory flows back; nothing can deadlock waiting on a ghost.

Renewal is the heartbeat: a live holder extends its lease; silence is
release. TTLs are capped (``max_ttl``) so no single request can pin memory
indefinitely - a peer needing longer simply keeps renewing, which is
exactly the liveness proof the cap exists to demand.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass

from .governor import MemoryGovernor


@dataclass(frozen=True)
class Lease:
    """One TTL-bounded reservation, addressable by id across the wire."""

    lease_id: str
    device: str
    nbytes: int
    expires_at: float  # event-loop time; ttl_remaining() is the wire shape

    def ttl_remaining(self) -> float:
        return max(0.0, self.expires_at - asyncio.get_running_loop().time())


class _Held:
    """Bookkeeping for one live lease: its holder task and release gate."""

    def __init__(self, lease: Lease) -> None:
        self.lease = lease
        self.released = asyncio.Event()
        self.task: asyncio.Task[None] | None = None


class LeaseBroker:
    """TTL leases over one instance's MemoryGovernor.

    Admission semantics are the governor's own (shed-then-wait, budget
    checks); the broker adds identity and mortality, never policy.
    """

    def __init__(self, governor: MemoryGovernor, *, max_ttl: float = 300.0) -> None:
        if max_ttl <= 0:
            raise ValueError("max_ttl must be > 0")
        self._governor = governor
        self._max_ttl = max_ttl
        self._held: dict[str, _Held] = {}
        self._holder_tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    def clamp_ttl(self, ttl: float) -> float:
        if ttl <= 0:
            raise ValueError("ttl must be > 0")
        return min(ttl, self._max_ttl)

    def leases(self) -> list[Lease]:
        return [held.lease for held in self._held.values()]

    async def acquire(
        self, device: str, nbytes: int, *, ttl: float, timeout: float | None = None
    ) -> Lease:
        """Reserve through the governor; hold until release or TTL expiry.

        Raises what the governor raises (BudgetExceeded, ReservationTimeout)
        - denial is the governor's verdict, relayed, never rewritten.
        """
        if self._closed:
            raise RuntimeError("LeaseBroker is closed")
        ttl = self.clamp_ttl(ttl)
        loop = asyncio.get_running_loop()
        lease = Lease(
            lease_id=uuid.uuid4().hex,
            device=device,
            nbytes=nbytes,
            expires_at=loop.time() + ttl,
        )
        held = _Held(lease)
        granted: asyncio.Future[None] = loop.create_future()

        async def hold() -> None:
            try:
                async with self._governor.reserve(device, nbytes, timeout=timeout):
                    if self._closed:
                        # The broker closed while we sat in admission; the
                        # context exit hands the grant straight back.
                        granted.set_exception(RuntimeError("LeaseBroker is closed"))
                        return
                    # The TTL clock starts at grant, not at request: a lease
                    # that waited in admission still gets its full lifetime.
                    held.lease = Lease(
                        lease_id=lease.lease_id,
                        device=device,
                        nbytes=nbytes,
                        expires_at=loop.time() + ttl,
                    )
                    self._held[lease.lease_id] = held
                    granted.set_result(None)
                    # Wait for release; re-arm on timeout because renewal
                    # may have pushed expiry out while we slept.
                    while not held.released.is_set():
                        remaining = held.lease.expires_at - loop.time()
                        if remaining <= 0:
                            break  # expired: the context exit releases
                        try:
                            await asyncio.wait_for(held.released.wait(), remaining)
                        except TimeoutError:
                            continue
            except BaseException as exc:
                if not granted.done():
                    granted.set_exception(exc)
                    return
                raise
            finally:
                self._held.pop(lease.lease_id, None)

        held.task = asyncio.create_task(hold())
        self._holder_tasks.add(held.task)
        held.task.add_done_callback(self._holder_tasks.discard)
        await granted
        return held.lease

    async def release(self, lease_id: str) -> bool:
        """Release a lease; False when unknown (already expired/released)."""
        held = self._held.get(lease_id)
        if held is None:
            return False
        held.released.set()
        if held.task is not None:
            await held.task
        return True

    def renew(self, lease_id: str, *, ttl: float) -> Lease | None:
        """Extend a live lease's TTL from now; None when it no longer exists.

        Expiry is checked eagerly: renewing a lease whose clock ran out is a
        miss even if the holder task has not woken to release it yet.
        """
        ttl = self.clamp_ttl(ttl)
        held = self._held.get(lease_id)
        if held is None:
            return None
        loop = asyncio.get_running_loop()
        if held.lease.expires_at <= loop.time():
            return None
        held.lease = Lease(
            lease_id=held.lease.lease_id,
            device=held.lease.device,
            nbytes=held.lease.nbytes,
            expires_at=loop.time() + ttl,
        )
        return held.lease

    async def close(self) -> None:
        """Release everything; the broker refuses new leases afterwards."""
        self._closed = True
        for held in list(self._held.values()):
            held.released.set()
        tasks = list(self._holder_tasks)
        for task in tasks:
            if not any(held.task is task for held in self._held.values()):
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
