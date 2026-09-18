from __future__ import annotations

import asyncio

import pytest
from dinkster_memory import (
    ConsumerItem,
    LeaseBroker,
    MemoryGovernor,
    PressureSignal,
    ReservationTimeout,
)


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
