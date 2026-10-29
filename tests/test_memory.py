"""M3 memory governance (DESIGN 3.6, hazard H13): budgets are declared,
costs ride the envelope, reservation precedes allocation, and eviction has
exactly one arbiter. ResourceHandles carry identity + residency + cost;
crossing a wire drops the object, never the facts."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_memory import (
    AcceleratorMemoryPolicy,
    AcceleratorMemoryPolicyError,
    BudgetExceeded,
    BudgetsError,
    ConsumerItem,
    DetailedConsumer,
    MeasuredMemory,
    MemoryGovernor,
    PageMap,
    PressureSignal,
    ReservationTimeout,
    Shedder,
    SystemMemorySnapshot,
    load_budgets,
    parse_budgets,
    parse_size,
)
from dinkster_values import (
    COST_META_KEY,
    RESOURCE_HANDLE_TYPE,
    RESOURCES_META_KEY,
    ResourceError,
    ResourceHandle,
    TypeRegistry,
    Value,
    register_resource_handle_type,
)

VRAM0 = "vram:cuda:0"


# -- accelerator reserve policy -----------------------------------------


def test_accelerator_policy_default_and_zero_headroom() -> None:
    default = AcceleratorMemoryPolicy()
    assert default.physical_headroom_bytes == 256 * 1024**2
    assert default.inference_reserve_bytes == int(0.8 * 1024**3)
    assert default.minimum_free_bytes == default.physical_headroom_bytes + int(0.8 * 1024**3)

    zero = AcceleratorMemoryPolicy(physical_headroom_bytes=0, inference_reserve_bytes=0)
    assert zero.resolve(100).residency_capacity_bytes == 100


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


def test_accelerator_policy_rejects_malformed_inputs_and_resolves_small_devices() -> None:
    for kwargs in (
        {"physical_headroom_bytes": -1},
        {"physical_headroom_bytes": True},
        {"inference_reserve_bytes": -1},
    ):
        with pytest.raises(AcceleratorMemoryPolicyError):
            AcceleratorMemoryPolicy(**kwargs)  # type: ignore[arg-type]

    policy = AcceleratorMemoryPolicy(
        physical_headroom_bytes=40,
        inference_reserve_bytes=61,
    )
    constrained = policy.resolve(100)
    assert constrained.insufficient_total
    assert constrained.residency_capacity_bytes == 0
    zero_total = policy.resolve(0)
    assert zero_total.insufficient_total
    assert zero_total.residency_capacity_bytes == 0
    with pytest.raises(AcceleratorMemoryPolicyError, match="hard accelerator budget"):
        policy.resolve(100, -1)


# -- persisted budgets ---------------------------------------------------


def test_reserved_subscribers_observe_grant_and_release_outside_lock(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 100})
        seen: list[tuple[str, int, int, bool]] = []

        def observer(device: str, total: int) -> None:
            seen.append((device, total, governor.reserved(device), governor._cond.locked()))

        def broken_observer(_device: str, _total: int) -> None:
            raise RuntimeError("listener failed")

        second: list[int] = []
        governor.subscribe_reserved(observer)
        governor.subscribe_reserved(broken_observer)
        governor.subscribe_reserved(lambda _device, total: second.append(total))
        caplog.set_level(logging.WARNING, logger="dinkster.memory.governor")
        async with governor.reserve(VRAM0, 40):
            assert governor.reserved(VRAM0) == 40
        assert seen == [
            (VRAM0, 40, 40, False),
            (VRAM0, 0, 0, False),
        ]
        assert second == [40, 0]
        assert caplog.text.count("broken_observer raised") == 2

    asyncio.run(scenario())


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


@pytest.mark.parametrize(
    "data",
    [
        [],
        {"budgets": "24G"},
        {"budgets": {"": "1G"}},
        {"budgets": {"bad device": "1G"}},
        {"budgets": {"ram": -1}},
        {"budgets": {"ram": True}},
        {"budgets": {"ram": 1.5}},
        {"budgets": {"ram": "garbage"}},
    ],
)
def test_parse_budgets_rejects_malformed_shapes_with_source(data: object) -> None:
    with pytest.raises(BudgetsError, match="operator-memory.toml"):
        parse_budgets(data, "operator-memory.toml")


def test_parse_budgets_rejects_unknown_top_level_keys_with_source() -> None:
    with pytest.raises(BudgetsError, match="operator-memory.toml.*unknown top-level"):
        parse_budgets({"budget": {"ram": "1G"}}, "operator-memory.toml")


def test_load_budgets_handles_empty_and_missing_files(tmp_path: Path) -> None:
    path = tmp_path / "memory.toml"
    assert load_budgets(path) == {}
    path.write_text("", "utf-8")
    assert load_budgets(path) == {}


def test_load_budgets_names_malformed_toml_source(tmp_path: Path) -> None:
    path = tmp_path / "memory.toml"
    path.write_text("[budgets\n", "utf-8")
    with pytest.raises(BudgetsError, match="memory.toml.*invalid TOML"):
        load_budgets(path)


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


def test_handle_meta_carries_residency_and_cost() -> None:
    registry = TypeRegistry()
    register_resource_handle_type(registry)
    value = registry.wrap(RESOURCE_HANDLE_TYPE, make_handle())
    assert value.meta.get(RESOURCES_META_KEY) == {"gpu": ("cuda:0", "cuda:1")}
    cost = value.meta.get(COST_META_KEY)
    assert isinstance(cost, dict) and cost[VRAM0] == 4_000_000_000


def test_handle_wire_roundtrip_drops_object_keeps_facts() -> None:
    registry = TypeRegistry()
    register_resource_handle_type(registry)
    spec = registry.spec(RESOURCE_HANDLE_TYPE)
    local = make_handle()
    remote = spec.decode(spec.encode(local))
    assert isinstance(remote, ResourceHandle)
    assert not remote.is_local
    assert remote.resource_id == local.resource_id
    assert remote.residency == {"gpu": ("cuda:0", "cuda:1")}
    assert remote.cost == dict(local.cost)
    with pytest.raises(ResourceError, match="not local"):
        remote.require_obj()
    assert local.require_obj() is local.obj


# -- MemoryGovernor: budgets and reservations ---------------------------


def test_reserve_grants_within_budget_and_releases() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 100})
        async with governor.reserve(VRAM0, 60):
            assert governor.reserved(VRAM0) == 60
            assert governor.available(VRAM0) == 40
        assert governor.reserved(VRAM0) == 0

    asyncio.run(scenario())


def test_reserve_unbudgeted_device_tracks_but_never_blocks() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor()
        async with governor.reserve("vram:cuda:9", 10**15):
            assert governor.reserved("vram:cuda:9") == 10**15
            assert governor.available("vram:cuda:9") is None

    asyncio.run(scenario())


def test_reserve_over_budget_raises_immediately() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 100})
        with pytest.raises(BudgetExceeded):
            async with governor.reserve(VRAM0, 101):
                pass

    asyncio.run(scenario())


def test_parallel_workflows_cannot_both_believe_memory_is_free() -> None:
    """The core parallel-safety property: the second reservation waits for
    the first to release instead of double-booking the device."""

    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 100})
        order: list[str] = []

        async def first() -> None:
            async with governor.reserve(VRAM0, 80):
                order.append("first-in")
                await asyncio.sleep(0.05)
            order.append("first-out")

        async def second() -> None:
            await asyncio.sleep(0.01)  # let first win the device
            async with governor.reserve(VRAM0, 80):
                order.append("second-in")

        await asyncio.gather(first(), second())
        assert order == ["first-in", "first-out", "second-in"]

    asyncio.run(scenario())


def test_reserve_timeout_when_holder_never_releases() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 100})
        async with governor.reserve(VRAM0, 80):
            with pytest.raises(ReservationTimeout):
                async with governor.reserve(VRAM0, 80, timeout=0.05):
                    pass

    asyncio.run(scenario())


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

    asyncio.run(scenario())


def test_ram_pressure_reads_each_reservation() -> None:
    async def scenario() -> None:
        available = iter((100, 20))

        def system_memory() -> SystemMemorySnapshot:
            current = next(available)
            return SystemMemorySnapshot(1000, current, 1000, current, ("injected",))

        governor = MemoryGovernor(system_memory=system_memory)
        pool = FakePool("ram", 100)
        governor.register_shedder(pool)

        async with governor.reserve("ram", 50):
            pass
        async with governor.reserve("ram", 50):
            pass

        assert pool.pressures == [PressureSignal("ram", 30)]
        assert pool.holding == 70

    asyncio.run(scenario())


def test_ram_admission_without_footprint_skips_snapshot() -> None:
    async def scenario() -> None:
        def unexpected_query() -> SystemMemorySnapshot:
            raise AssertionError("RAM without consumers must keep the direct admission path")

        governor = MemoryGovernor(system_memory=unexpected_query)
        async with governor.reserve("ram", 10**15):
            assert governor.reserved("ram") == 10**15

    asyncio.run(scenario())


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

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("available_bytes", "expected_shortfall"),
    ((50, 30), (10, 50)),
)
def test_ram_pressure_combines_shortfalls(
    available_bytes: int,
    expected_shortfall: int,
) -> None:
    async def scenario() -> None:
        snapshot = SystemMemorySnapshot(
            1000,
            available_bytes,
            1000,
            available_bytes,
            ("injected",),
        )
        governor = MemoryGovernor({"ram": 100}, system_memory=lambda: snapshot)
        pool = FakePool("ram", 70)
        governor.register_shedder(pool)

        async with governor.reserve("ram", 60):
            assert pool.pressures == [PressureSignal("ram", expected_shortfall)]
            assert pool.holding == 70 - expected_shortfall

    asyncio.run(scenario())


def test_ram_over_budget_refuses_without_live_pressure() -> None:
    async def scenario() -> None:
        def unexpected_query() -> SystemMemorySnapshot:
            raise AssertionError("an impossible request must not query live memory")

        governor = MemoryGovernor({"ram": 100}, system_memory=unexpected_query)
        pool = FakePool("ram", 70)
        governor.register_shedder(pool)

        with pytest.raises(BudgetExceeded):
            async with governor.reserve("ram", 101):
                pass
        assert pool.pressures == []

    asyncio.run(scenario())


@pytest.mark.parametrize("expire_before_shed", [False, True])
def test_ram_pressure_timeout_keeps_unbudgeted_grant(
    monkeypatch: pytest.MonkeyPatch,
    expire_before_shed: bool,
) -> None:
    now = 0.0

    class SlowPool(FakePool):
        cancelled = False

        async def shed(self, pressure: PressureSignal) -> int:
            nonlocal now
            self.pressures.append(pressure)
            now = 1.0
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            return 0

    async def scenario() -> None:
        monkeypatch.setattr(asyncio.get_running_loop(), "time", lambda: now)

        def snapshot() -> SystemMemorySnapshot:
            nonlocal now
            if expire_before_shed:
                now = 1.0
            return SystemMemorySnapshot(1000, 10, 1000, 10, ("injected",))

        governor = MemoryGovernor(system_memory=snapshot)
        pool = SlowPool("ram", 30)
        governor.register_shedder(pool)

        async with governor.reserve("ram", 60, timeout=0.01):
            assert governor.reserved("ram") == 60
        assert governor.reserved("ram") == 0
        assert pool.holding == 30
        assert pool.pressures == ([] if expire_before_shed else [PressureSignal("ram", 50)])
        assert pool.cancelled is not expire_before_shed

    asyncio.run(scenario())


def test_shedders_are_protocol_instances() -> None:
    assert isinstance(FakePool(VRAM0, 1), Shedder)
    assert isinstance(MemoryLRUCache(), Shedder)


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

    asyncio.run(scenario())


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

    asyncio.run(scenario())


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

    asyncio.run(scenario())


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
                # Names are discoverable so /cache/trim can target them.
                "consumers": {"FakePool": 25},
            }
            assert status["ram"]["reservedBytes"] == 0

    asyncio.run(scenario())


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


def test_cache_excludes_owner_accounted_references_from_footprint() -> None:
    """A ResourceHandle's cost is the owner's footprint: the cache holding
    a reference must not count it, or the governor sees the same bytes in
    two consumers."""

    async def scenario() -> None:
        registry = TypeRegistry()
        register_resource_handle_type(registry)
        cache = MemoryLRUCache()
        handle = make_handle(cost={VRAM0: 4_000_000}, residency={})
        await cache.put("model", {"out": registry.wrap(RESOURCE_HANDLE_TYPE, handle)})  # type: ignore[arg-type]
        assert cache.footprint(VRAM0) == 0

    asyncio.run(scenario())


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

    asyncio.run(scenario())


def test_cache_footprint_reads_cost_metadata() -> None:
    async def scenario() -> None:
        registry = costed_registry()
        cache = MemoryLRUCache()
        await cache.put("a", costed_entry(registry, 30))  # type: ignore[arg-type]
        await cache.put("b", costed_entry(registry, 20))  # type: ignore[arg-type]
        assert cache.footprint(VRAM0) == 50
        assert cache.footprint("vram:cuda:1") == 0

    asyncio.run(scenario())


def test_cache_sheds_lru_first_and_skips_costless_entries() -> None:
    async def scenario() -> None:
        registry = costed_registry()
        cache = MemoryLRUCache()
        await cache.put("old", costed_entry(registry, 30))  # type: ignore[arg-type]
        await cache.put("costless", {})
        await cache.put("new", costed_entry(registry, 30))  # type: ignore[arg-type]
        await cache.get("old")  # touch: old becomes most recent
        freed = await cache.shed(PressureSignal(VRAM0, 30))
        assert freed == 30
        # "new" was least-recently-used among costed entries; "costless"
        # survives because evicting it frees nothing.
        assert await cache.get("new") is None
        assert await cache.get("old") is not None
        assert await cache.get("costless") is not None

    asyncio.run(scenario())


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

    asyncio.run(scenario())


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


def test_detail_contract_shapes_validate() -> None:
    with pytest.raises(ValueError, match="item_id"):
        ConsumerItem(item_id="", display_name="x")
    with pytest.raises(ValueError, match="display_name"):
        ConsumerItem(item_id="x", display_name="")
    with pytest.raises(ValueError, match=">= 0"):
        ConsumerItem(item_id="x", display_name="x", bytes_by_residency={VRAM0: -1})
    with pytest.raises(ValueError, match="page_bytes"):
        PageMap(page_bytes=0, flags=(1,))
    pages = PageMap(page_bytes=32, flags=(1, 0, 1))
    assert pages.page_count == 3  # derived from flags, never stored twice


def test_details_serves_only_detail_consumers() -> None:
    governor = MemoryGovernor({VRAM0: 100})
    pool = ModelPool(VRAM0, {"ckpt-a": 40})
    plain = FakePool(VRAM0, 10)
    governor.register_shedder(pool, name="models")
    governor.register_shedder(plain, name="plain")
    assert isinstance(pool, DetailedConsumer)
    assert not isinstance(plain, DetailedConsumer)
    details = governor.details()
    # The plain consumer is absent from details but present in status():
    # its footprint number is still accounted, just not itemized.
    assert set(details) == {"models"}
    (item,) = details["models"]
    assert item.item_id == "ckpt-a"
    assert item.display_name == "model ckpt-a"
    assert item.bytes_by_residency == {VRAM0: 40}
    assert governor.status()[VRAM0]["consumers"] == {"models": 40, "plain": 10}


def test_item_targeted_shed_requires_exactly_one_consumer() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 100})
        governor.register_shedder(ModelPool(VRAM0, {"a": 10}), name="models")
        with pytest.raises(ValueError, match="exactly one consumer"):
            await governor.shed(VRAM0, 10, items=["a"])
        with pytest.raises(ValueError, match="exactly one consumer"):
            await governor.shed(VRAM0, 10, consumers=["models", "other"], items=["a"])

    asyncio.run(scenario())


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

    asyncio.run(scenario())


def test_item_pressure_never_reaches_a_consumer_that_cannot_resolve_it() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 100})
        plain = FakePool(VRAM0, 50)
        governor.register_shedder(plain, name="plain")
        # A consumer without the detail contract cannot resolve item IDs;
        # sending it the signal would shed *something*, which is worse
        # than shedding nothing.
        freed = await governor.shed(VRAM0, 10, consumers=["plain"], items=["mystery"])
        assert freed == 0
        assert plain.pressures == []

    asyncio.run(scenario())


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


def test_status_measured_is_null_without_a_probe() -> None:
    governor = MemoryGovernor({VRAM0: 100})
    assert governor.status()[VRAM0]["measured"] is None
