"""M3 memory governance (DESIGN 3.6, hazard H13): budgets are declared,
costs ride the envelope, reservation precedes allocation, and eviction has
exactly one arbiter. ResourceHandles carry identity + residency + cost;
crossing a wire drops the object, never the facts."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_memory import (
    AcceleratorMemoryPolicy,
    BudgetExceeded,
    BudgetsError,
    ConsumerItem,
    MeasuredMemory,
    MemoryGovernor,
    PressureSignal,
    ReportedTelemetry,
    ReservationTimeout,
    SystemMemorySnapshot,
    load_budgets,
    parse_budgets,
    parse_size,
)
from dinkster_values import (
    COST_META_KEY,
    RESOURCE_HANDLE_TYPE,
    ResourceHandle,
    TypeRegistry,
    Value,
    register_resource_handle_type,
)

from tests.async_scenarios import run_scenario

VRAM0 = "vram:cuda:0"


# -- accelerator reserve policy -----------------------------------------


def test_accelerator_policy_resolves_hard_budget_without_conflating_reserves() -> None:
    policy = AcceleratorMemoryPolicy(
        physical_headroom_bytes=10,
        inference_reserve_bytes=20,
    )
    resolved = policy.resolve(100, 70)
    assert resolved.effective_budget_bytes == 70
    assert resolved.budget_headroom_bytes == 30
    assert resolved.minimum_free_bytes == 30
    assert resolved.residency_capacity_bytes == 40
    assert not resolved.insufficient_total
    assert not resolved.budget_exceeds_total

    disabled = policy.resolve(100, 0)
    assert disabled.effective_budget_bytes == 0
    assert disabled.budget_headroom_bytes == 100
    assert disabled.residency_capacity_bytes == 0

    below_reserves = policy.resolve(100, 20)
    assert below_reserves.residency_capacity_bytes == 0
    assert not below_reserves.insufficient_total


def test_accelerator_policy_accepts_wddm_budget_above_reported_total() -> None:
    policy = AcceleratorMemoryPolicy(
        physical_headroom_bytes=10,
        inference_reserve_bytes=20,
    )
    resolved = policy.resolve(100, 150)
    assert resolved.effective_budget_bytes == 100
    assert resolved.budget_headroom_bytes == 0
    assert resolved.residency_capacity_bytes == 70
    assert resolved.budget_exceeds_total


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


def test_parse_budgets_accepts_string_sizes_and_integer_bytes() -> None:
    assert parse_budgets({"budgets": {"ram": "24G", VRAM0: 20 * 1024**3}}, "memory.toml") == {
        "ram": 24 * 1024**3,
        VRAM0: 20 * 1024**3,
    }
    assert parse_budgets({}, "memory.toml") == {}
    assert parse_budgets({"budgets": {}}, "memory.toml") == {}
    # A zero budget persists like any other: the governor treats it as
    # a device where nothing may be admitted, not as "unbudgeted".
    assert parse_budgets({"budgets": {"ram": 0}}, "memory.toml") == {"ram": 0}
    assert parse_budgets({"budgets": {"ram": "0"}}, "memory.toml") == {"ram": 0}


def test_parse_budgets_rejects_unknown_top_level_keys_with_source() -> None:
    with pytest.raises(BudgetsError, match="operator-memory.toml.*unknown top-level"):
        parse_budgets({"budget": {"ram": "1G"}}, "operator-memory.toml")


def test_load_budgets_handles_empty_and_missing_files(tmp_path: Path) -> None:
    path = tmp_path / "memory.toml"
    assert load_budgets(path) == {}
    path.write_text("", "utf-8")
    assert load_budgets(path) == {}


def make_handle(**overrides: object) -> ResourceHandle:
    fields: dict[str, object] = {
        "resource_id": "blake3:ab12?dtype=fp16",
        "kind": "model",
        "residency": {"gpu": ("cuda:0", "cuda:1")},
        "cost": {VRAM0: 4_000_000_000, "ram": 500_000_000},
        "obj": object(),
    }
    fields.update(overrides)
    return ResourceHandle(**fields)  # type: ignore[arg-type]


# -- ResourceHandle values ----------------------------------------------


def test_handle_fingerprint_is_resource_id_never_residency() -> None:
    registry = TypeRegistry()
    register_resource_handle_type(registry)
    on_gpu0 = registry.wrap(RESOURCE_HANDLE_TYPE, make_handle(residency={"gpu": "cuda:0"}))
    on_gpu1 = registry.wrap(RESOURCE_HANDLE_TYPE, make_handle(residency={"gpu": "cuda:1"}))
    # Same model on a different device is the same computation (hazard H4).
    assert on_gpu0.fingerprint == on_gpu1.fingerprint == "blake3:ab12?dtype=fp16"


# -- MemoryGovernor: budgets and reservations ---------------------------


def test_reserve_unbudgeted_device_tracks_but_never_blocks() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor()
        async with governor.reserve("vram:cuda:9", 10**15):
            assert governor.reserved("vram:cuda:9") == 10**15
            assert governor.available("vram:cuda:9") is None

    run_scenario(scenario())


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


def test_ram_pressure_sheds_shortfall_without_default_cap() -> None:
    async def scenario() -> None:
        queries = 0

        def system_memory() -> SystemMemorySnapshot:
            nonlocal queries
            queries += 1
            return SystemMemorySnapshot(1000, 10, 1000, 10, ("injected",))

        governor = MemoryGovernor(system_memory=system_memory)
        pool = FakePool("ram", 30)
        governor.register_shedder(pool)

        async with governor.reserve("ram", 60):
            assert governor.reserved("ram") == 60
            assert governor.available("ram") is None

        assert queries == 1
        assert pool.pressures == [PressureSignal("ram", 50)]
        assert pool.holding == 0

    run_scenario(scenario())


def test_ram_snapshot_failure_keeps_unbudgeted_grant() -> None:
    async def scenario() -> None:
        def unavailable() -> SystemMemorySnapshot:
            raise RuntimeError("memory provider unavailable")

        governor = MemoryGovernor(system_memory=unavailable)
        pool = FakePool("ram", 10)
        governor.register_shedder(pool)
        async with governor.reserve("ram", 100):
            pass
        assert pool.pressures == []

    run_scenario(scenario())


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


def test_shed_priority_order_caches_before_pools() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 100})
        pool = FakePool(VRAM0, 50)
        cache = FakePool(VRAM0, 50)
        governor.register_shedder(pool, priority=10, name="pool")
        governor.register_shedder(cache, priority=0, name="cache")
        freed = await governor.shed(VRAM0, 60)
        assert freed == 60
        assert cache.holding == 0  # lower priority shed first, fully
        assert pool.holding == 40  # then only the remainder

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


def test_status_reports_one_shape_for_endpoints() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 100, "ram": 1000})
        governor.register_shedder(FakePool(VRAM0, 25))
        async with governor.reserve(VRAM0, 50):
            status = governor.status()
            assert status[VRAM0] == {
                "budgetBytes": 100,
                "reservedBytes": 50,
                "consumerFootprintBytes": 25,
                "availableBytes": 25,
                "measured": None,  # no probe wired: honest absence
                "peakUsedBytes": None,
                # Names are discoverable so /cache/trim can target them.
                "consumers": {"FakePool": 25},
            }
            assert status["ram"]["reservedBytes"] == 0

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


def test_cache_drops_entries_referencing_a_resource() -> None:
    async def scenario() -> None:
        registry = costed_registry()
        register_resource_handle_type(registry)
        cache = MemoryLRUCache()
        handle = make_handle(resource_id="blake3:aa", residency={})
        await cache.put("model", {"out": registry.wrap(RESOURCE_HANDLE_TYPE, handle)})  # type: ignore[arg-type]
        await cache.put("blob", costed_entry(registry, 30))  # type: ignore[arg-type]
        assert cache.drop_referencing("blake3:aa") == 1
        assert await cache.get("model") is None  # reference gone
        assert await cache.get("blob") is not None  # bystander survives
        assert cache.drop_referencing("blake3:aa") == 0  # idempotent

    run_scenario(scenario())


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


def test_status_reports_measured_beside_declared() -> None:
    def probe(device: str) -> MeasuredMemory | None:
        if device == VRAM0:
            return MeasuredMemory(free_bytes=7_000, total_bytes=24_000)
        if device == "ram":
            raise OSError("driver hiccup")  # absence, never a crash
        return None

    governor = MemoryGovernor({VRAM0: 100, "ram": 1000, "disk": 10}, telemetry=probe)
    status = governor.status()
    assert status[VRAM0]["measured"] == {"freeBytes": 7_000, "totalBytes": 24_000}
    assert status["ram"]["measured"] is None  # probe raised
    assert status["disk"]["measured"] is None  # probe declined


def test_reported_usage_sums_distinct_processes_but_keeps_hardware_freshest() -> None:
    first, duplicate, second = object(), object(), object()
    reported = ReportedTelemetry()
    reported.update(
        first,
        {
            VRAM0: MeasuredMemory(
                600,
                1000,
                process_id=1,
                torch_allocated_bytes=50,
                torch_reserved_bytes=100,
                process_rss_bytes=700,
                pinned_host_bytes=10,
            )
        },
    )
    reported.update(
        duplicate,
        {
            VRAM0: MeasuredMemory(
                500,
                1000,
                process_id=1,
                torch_allocated_bytes=70,
                torch_reserved_bytes=110,
                process_rss_bytes=800,
                pinned_host_bytes=20,
            )
        },
    )
    reported.update(
        second,
        {
            VRAM0: MeasuredMemory(
                400,
                1000,
                process_id=2,
                torch_allocated_bytes=90,
                torch_reserved_bytes=130,
                process_rss_bytes=900,
                pinned_host_bytes=30,
            )
        },
    )
    measured = reported.probe(VRAM0)
    assert measured is not None
    assert (measured.free_bytes, measured.total_bytes) == (400, 1000)
    assert (measured.torch_allocated_bytes, measured.torch_reserved_bytes) == (160, 240)
    assert (measured.process_rss_bytes, measured.pinned_host_bytes) == (1700, 50)
    assert measured.process_id is None and measured.process_count == 2
    reported.clear(second)
    assert reported.probe(VRAM0) == MeasuredMemory(
        500,
        1000,
        process_id=1,
        torch_allocated_bytes=70,
        torch_reserved_bytes=110,
        process_rss_bytes=800,
        pinned_host_bytes=20,
    )


def test_peak_tracks_raw_driver_usage_and_reset_keeps_current_value() -> None:
    measured = MeasuredMemory(800, 1000, driver_free_bytes=600)
    governor = MemoryGovernor({VRAM0: 1200}, telemetry=lambda _: measured)
    assert governor.status()[VRAM0]["peakUsedBytes"] == 400
    measured = MeasuredMemory(900, 1000, driver_free_bytes=750)
    assert governor.status()[VRAM0]["peakUsedBytes"] == 400
    assert governor.reset_peak(VRAM0) == 250
    assert governor.status()[VRAM0]["peakUsedBytes"] == 250
    measured = MeasuredMemory(500, 1000, driver_free_bytes=300)
    assert governor.status()[VRAM0]["peakUsedBytes"] == 700
