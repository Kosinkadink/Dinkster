"""LeaseBroker (DESIGN 3.10): reservations with handles and TTLs, so
cross-instance holders can crash without pinning memory. Admission
semantics stay the governor's; the broker adds identity and mortality."""

from __future__ import annotations

import asyncio

import pytest
from dinkster_memory import (
    BudgetExceeded,
    LeaseBroker,
    MemoryGovernor,
    PressureSignal,
    ReservationTimeout,
    Shedder,
)

VRAM0 = "vram:cuda:0"


def reserved(governor: MemoryGovernor) -> int:
    return governor.reserved(VRAM0)


def test_acquire_reserves_and_release_frees() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 100})
        broker = LeaseBroker(governor)
        lease = await broker.acquire(VRAM0, 60, ttl=30.0)
        assert reserved(governor) == 60
        assert lease.ttl_remaining() > 29.0
        assert [held.lease_id for held in broker.leases()] == [lease.lease_id]

        assert await broker.release(lease.lease_id) is True
        assert reserved(governor) == 0
        assert broker.leases() == []
        # Releasing again is a miss, not an error: the lease is gone.
        assert await broker.release(lease.lease_id) is False

    asyncio.run(scenario())


def test_ttl_expiry_frees_the_reservation() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 100})
        broker = LeaseBroker(governor)
        lease = await broker.acquire(VRAM0, 60, ttl=0.05)
        assert reserved(governor) == 60
        # The holder never calls back - a crashed peer. TTL is the cleanup.
        async with asyncio.timeout(2):
            while reserved(governor) > 0:
                await asyncio.sleep(0.01)
        assert broker.leases() == []
        assert await broker.release(lease.lease_id) is False

    asyncio.run(scenario())


def test_renew_extends_and_expired_lease_cannot_renew() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 100})
        broker = LeaseBroker(governor)
        lease = await broker.acquire(VRAM0, 10, ttl=0.1)
        original_expiry = lease.expires_at
        renewed = broker.renew(lease.lease_id, ttl=2.0)
        assert renewed is not None
        assert renewed.expires_at > original_expiry
        # The renewal keeps the lease alive beyond its original lifetime.
        await asyncio.sleep(0.25)
        assert reserved(governor) == 10
        await asyncio.sleep(2.0)  # silence: the renewed lease dies
        assert broker.renew(lease.lease_id, ttl=1.0) is None
        async with asyncio.timeout(2):
            while reserved(governor) > 0:
                await asyncio.sleep(0.01)

    asyncio.run(scenario())


def test_ttl_is_capped_by_max_ttl() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 100})
        broker = LeaseBroker(governor, max_ttl=0.5)
        lease = await broker.acquire(VRAM0, 10, ttl=3600.0)
        assert lease.ttl_remaining() <= 0.5
        await broker.release(lease.lease_id)
        with pytest.raises(ValueError, match="ttl must be > 0"):
            await broker.acquire(VRAM0, 10, ttl=0.0)

    asyncio.run(scenario())


def test_governor_verdicts_are_relayed_not_rewritten() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 100})
        broker = LeaseBroker(governor)
        with pytest.raises(BudgetExceeded):
            await broker.acquire(VRAM0, 200, ttl=1.0)
        assert reserved(governor) == 0

        lease = await broker.acquire(VRAM0, 100, ttl=30.0)
        with pytest.raises(ReservationTimeout):
            await broker.acquire(VRAM0, 50, ttl=1.0, timeout=0.05)
        assert reserved(governor) == 100  # the failed acquire left no residue
        await broker.release(lease.lease_id)

    asyncio.run(scenario())


def test_waiting_acquire_gets_full_ttl_at_grant() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 100})
        broker = LeaseBroker(governor)
        first = await broker.acquire(VRAM0, 100, ttl=30.0)

        async def blocked() -> float:
            lease = await broker.acquire(VRAM0, 50, ttl=0.5, timeout=5.0)
            return lease.ttl_remaining()

        task = asyncio.create_task(blocked())
        await asyncio.sleep(0.2)  # let it sit in admission for a while
        await broker.release(first.lease_id)
        remaining = await task
        # The clock started at grant: near-full TTL despite the 0.2s wait.
        assert remaining > 0.4

    asyncio.run(scenario())


class FakeCache(Shedder):
    def __init__(self, device: str, holding: int) -> None:
        self.device = device
        self.holding = holding

    def footprint(self, device: str) -> int:
        return self.holding if device == self.device else 0

    async def shed(self, pressure: PressureSignal) -> int:
        freed = min(self.holding, pressure.bytes_needed)
        self.holding -= freed
        return freed


def test_acquire_sheds_consumers_like_any_reservation() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 100})
        cache = FakeCache(VRAM0, 80)
        governor.register_shedder(cache, name="cache")
        broker = LeaseBroker(governor)
        lease = await broker.acquire(VRAM0, 60, ttl=1.0)
        assert cache.holding <= 40  # the governor shed the cache to fit
        await broker.release(lease.lease_id)

    asyncio.run(scenario())


def test_close_releases_everything_and_refuses_new_leases() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 100, "ram": 100})
        broker = LeaseBroker(governor)
        await broker.acquire(VRAM0, 50, ttl=30.0)
        await broker.acquire("ram", 25, ttl=30.0)
        await broker.close()
        assert governor.reserved(VRAM0) == 0
        assert governor.reserved("ram") == 0
        with pytest.raises(RuntimeError, match="closed"):
            await broker.acquire(VRAM0, 1, ttl=1.0)

    asyncio.run(scenario())
