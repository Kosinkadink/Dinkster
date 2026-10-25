from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, BrokenBarrierError

import dinkster_api.v1 as api
import dinkster_memory.resident as resident
import pytest
from dinkster_memory import (
    ConsumerItem,
    LeaseBroker,
    MemoryGovernor,
    PressureSignal,
    ReservationTimeout,
)
from dinkster_memory.resident import ResidentPool


class HungShedder:
    def footprint(self, device: str) -> int:
        return 100

    async def shed(self, pressure: PressureSignal) -> int:
        await asyncio.Event().wait()
        return 0


def test_reservation_timeout_bounds_hung_shedder() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        governor.register_shedder(HungShedder(), name="hung")
        started = asyncio.get_running_loop().time()
        with pytest.raises(ReservationTimeout):
            async with governor.reserve("ram", 1, timeout=0.05):
                pass
        assert asyncio.get_running_loop().time() - started < 0.5

    asyncio.run(scenario())


def test_close_cancels_acquire_pending_in_admission() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        governor.register_shedder(HungShedder(), name="hung")
        broker = LeaseBroker(governor)
        acquire = asyncio.create_task(broker.acquire("ram", 1, ttl=10))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await broker.close()
        with pytest.raises(asyncio.CancelledError):
            await acquire
        assert not broker._holder_tasks

    asyncio.run(scenario())


def test_consumer_item_snapshots_residency_mapping() -> None:
    sizes = {"ram": 10}
    item = ConsumerItem("item", "Item", sizes)
    sizes["ram"] = 20
    assert item.bytes_by_residency["ram"] == 10


def test_keyed_residents_share_alternating_loads_and_reload_after_pressure() -> None:
    pool = ResidentPool()
    loads: list[str] = []

    def obtain(key: str) -> tuple[dict[str, bytes], dict[str, bytes]]:
        def load() -> tuple[dict[str, bytes], dict[str, bytes]]:
            loads.append(key)
            return ({"model": key.encode()}, {"clip": b"xyz"})

        return pool.obtain(
            key,
            load,
            ram_bytes=lambda value: len(value[0]["model"]) + len(value[1]["clip"]),
        )

    first = obtain("checkpoint-a")
    second = obtain("checkpoint-bb")
    assert first != second
    assert obtain("checkpoint-a") is first
    assert obtain("checkpoint-bb") is second
    assert loads == ["checkpoint-a", "checkpoint-bb"]
    assert pool.footprint("ram") == 31

    async def release() -> None:
        candidates = pool.propose_release(PressureSignal("ram", 1))
        assert len(candidates) == 1
        assert await pool.release("ram", candidates) == {candidates[0].item_id: 15}

    asyncio.run(release())
    assert pool.footprint("ram") == 16
    assert obtain("checkpoint-bb") is second
    assert obtain("checkpoint-a") == first
    assert loads == ["checkpoint-a", "checkpoint-bb", "checkpoint-a"]


def test_keyed_resident_failed_load_is_retryable() -> None:
    pool = ResidentPool()

    def fail() -> bytes:
        raise ValueError("bad checkpoint")

    with pytest.raises(ValueError, match="bad checkpoint"):
        pool.obtain("checkpoint", fail, ram_bytes=len)
    assert pool.footprint("ram") == 0
    assert pool.obtain("checkpoint", lambda: b"weights", ram_bytes=len) == b"weights"
    assert pool.footprint("ram") == 7


def test_public_keyed_residents_share_first_concurrent_load(monkeypatch: pytest.MonkeyPatch) -> None:
    constructors = Barrier(2)

    def slow_pool() -> ResidentPool:
        try:
            constructors.wait(timeout=1)
        except BrokenBarrierError:
            pass
        return ResidentPool()

    monkeypatch.setattr(resident, "_pool", None)
    monkeypatch.setattr(resident, "ResidentPool", slow_pool)
    callers = Barrier(2)
    loads: list[str] = []

    def obtain() -> bytearray:
        callers.wait(timeout=5)

        def load() -> bytearray:
            loads.append("checkpoint")
            return bytearray(b"weights")

        return api.obtain_resident("checkpoint", load, ram_bytes=len)

    with ThreadPoolExecutor(2) as threads:
        first = threads.submit(obtain)
        second = threads.submit(obtain)
        assert first.result(timeout=5) is second.result(timeout=5)
    assert loads == ["checkpoint"]
