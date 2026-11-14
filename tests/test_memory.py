"""M3 memory governance (DESIGN 3.6, hazard H13): budgets are declared,
costs ride the envelope, reservation precedes allocation, and eviction has
exactly one arbiter. ResourceHandles carry identity + residency + cost;
crossing a wire drops the object, never the facts."""

from __future__ import annotations

import asyncio

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_memory import (
    BudgetExceeded,
    ConsumerItem,
    MemoryGovernor,
    PressureSignal,
    ReservationTimeout,
    parse_size,
)
from dinkster_values import (
    COST_META_KEY,
    TypeRegistry,
    Value,
)

from tests.async_scenarios import run_scenario

VRAM0 = "vram:cuda:0"


# -- accelerator reserve policy -----------------------------------------


# -- persisted budgets ---------------------------------------------------


def test_parse_budget_size_uses_binary_suffixes() -> None:
    assert parse_size("1") == 1
    assert parse_size(" 8K ") == 8 * 1024
    assert parse_size("2m") == 2 * 1024**2
    assert parse_size("3G") == 3 * 1024**3
    assert parse_size("4t") == 4 * 1024**4
    # Zero is the "deny every reservation on this device" knob: the
    # governor accepts budgets >= 0 and the pre-config CLI accepted it.
    assert parse_size("0") == 0
    assert parse_size("0k") == 0
    for bad in ("", "-1", "1.5G", "5X", "G"):
        with pytest.raises(ValueError):
            parse_size(bad)


# -- ResourceHandle values ----------------------------------------------


# -- MemoryGovernor: budgets and reservations ---------------------------


def test_reserve_over_budget_raises_immediately() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 100})
        with pytest.raises(BudgetExceeded):
            async with governor.reserve(VRAM0, 101):
                pass

    run_scenario(scenario())


def test_parallel_workflows_cannot_both_believe_memory_is_free() -> None:
    """The core parallel-safety property: the second reservation waits for
    the first to release instead of double-booking the device."""

    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 100})
        order: list[str] = []
        first_entered = asyncio.Event()
        second_requested = asyncio.Event()

        async def first() -> None:
            async with governor.reserve(VRAM0, 80):
                order.append("first-in")
                first_entered.set()
                await second_requested.wait()
                assert order == ["first-in"]
                assert governor.reserved(VRAM0) == 80
            order.append("first-out")

        async def second() -> None:
            await first_entered.wait()
            second_requested.set()
            async with governor.reserve(VRAM0, 80):
                order.append("second-in")

        await asyncio.wait_for(asyncio.gather(first(), second()), 2)
        assert order == ["first-in", "first-out", "second-in"]

    run_scenario(scenario())


def test_reserve_timeout_when_holder_never_releases() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 100})
        async with governor.reserve(VRAM0, 80):
            with pytest.raises(ReservationTimeout):
                async with governor.reserve(VRAM0, 80, timeout=0.05):
                    pass

    run_scenario(scenario())


# -- MemoryGovernor: shedding -------------------------------------------


class FakePool:
    """A governed consumer holding a fixed footprint it can give back."""

    def __init__(self, device: str, holding: int) -> None:
        self.device = device
        self.holding = holding
        self.pressures: list[PressureSignal] = []

    def footprint(self, device: str) -> int:
        return self.holding if device == self.device else 0

    async def shed(self, pressure: PressureSignal) -> int:
        self.pressures.append(pressure)
        freed = min(self.holding, pressure.bytes_needed)
        self.holding -= freed
        return freed


def test_reservation_sheds_consumers_to_make_room() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 100})
        pool = FakePool(VRAM0, 70)
        governor.register_shedder(pool)
        assert governor.available(VRAM0) == 30
        async with governor.reserve(VRAM0, 60):
            # The pool was asked for exactly the shortfall, once.
            assert pool.pressures == [PressureSignal(VRAM0, 30)]
            assert pool.holding == 40

    run_scenario(scenario())


class StubbornPool:
    """Holds bytes it refuses to give back (a pinned model, in real life)."""

    def __init__(self, device: str, holding: int) -> None:
        self.device = device
        self.holding = holding

    def footprint(self, device: str) -> int:
        return self.holding if device == self.device else 0

    async def shed(self, pressure: PressureSignal) -> int:
        return 0


def test_reservation_fails_honestly_when_shedding_is_not_enough() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 100})
        governor.register_shedder(FakePool(VRAM0, 10))
        governor.register_shedder(StubbornPool(VRAM0, 60))
        # 100 budget - 60 stubborn = 40 usable; 50 cannot fit and there is
        # no reservation to wait for, so this must raise, not hang.
        with pytest.raises(ReservationTimeout):
            async with governor.reserve(VRAM0, 50):
                pass

    run_scenario(scenario())


# -- MemoryLRUCache as a governed consumer ------------------------------


def costed_registry() -> TypeRegistry:
    """A type whose values *own* their bytes (a big tensor, not a handle):
    cost rides the envelope without RESOURCE_ID_META_KEY, so the cache is
    the owner and accounts it."""
    registry = TypeRegistry()
    registry.register("test.blob", meta=lambda obj: {COST_META_KEY: {VRAM0: obj}})
    return registry


def costed_entry(registry: TypeRegistry, nbytes: int) -> dict[str, Value]:
    return {"out": registry.wrap("test.blob", nbytes)}


def test_governor_drives_cache_eviction_end_to_end() -> None:
    """The full H13 loop: a reservation that does not fit causes the
    governor (and only the governor) to evict cache entries."""

    async def scenario() -> None:
        registry = costed_registry()
        governor = MemoryGovernor({VRAM0: 100})
        cache = MemoryLRUCache()
        governor.register_shedder(cache)
        await cache.put("a", costed_entry(registry, 40))  # type: ignore[arg-type]
        await cache.put("b", costed_entry(registry, 40))  # type: ignore[arg-type]
        assert governor.available(VRAM0) == 20
        async with governor.reserve(VRAM0, 50):
            assert cache.footprint(VRAM0) <= 50
            assert await cache.get("a") is None  # LRU went first
            assert await cache.get("b") is not None

    run_scenario(scenario())


# -- consumer detail contract (DESIGN 3.10: memory observability) --------


class ModelPool:
    """A detail-contract consumer: items with stable IDs, sheddable by ID."""

    def __init__(self, device: str, models: dict[str, int]) -> None:
        self.device = device
        self.models = dict(models)  # item id -> bytes on device
        self.pressures: list[PressureSignal] = []

    def footprint(self, device: str) -> int:
        return sum(self.models.values()) if device == self.device else 0

    async def shed(self, pressure: PressureSignal) -> int:
        self.pressures.append(pressure)
        if pressure.device != self.device:
            return 0
        candidates = (
            [i for i in pressure.items if i in self.models]
            if pressure.items is not None
            else list(self.models)
        )
        freed = 0
        for item_id in candidates:
            if pressure.items is None and freed >= pressure.bytes_needed:
                break
            freed += self.models.pop(item_id)
        return freed

    def details(self) -> list[ConsumerItem]:
        return [
            ConsumerItem(
                item_id=item_id,
                display_name=f"model {item_id}",
                bytes_by_residency={self.device: nbytes},
            )
            for item_id, nbytes in sorted(self.models.items())
        ]


def test_item_targeted_shed_unloads_named_items_only() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 100})
        pool = ModelPool(VRAM0, {"ckpt-a": 40, "ckpt-b": 30})
        governor.register_shedder(pool, name="models")
        # "Unload ckpt-a": more pressure than the item holds must still
        # free only the named item, never its neighbors.
        freed = await governor.shed(VRAM0, 100, consumers=["models"], items=["ckpt-a"])
        assert freed == 40
        assert set(pool.models) == {"ckpt-b"}
        assert pool.pressures[-1].items == ("ckpt-a",)

    run_scenario(scenario())


# -- measured telemetry ---------------------------------------------------
