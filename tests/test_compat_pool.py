"""ResidentPool: the compat residency table as a governed consumer.

The safety property under test throughout: shedding governs *device
residency*, never the references - a stub for a shed resident still
resolves, so memory pressure can never turn into ResidentLookupError.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from dinkster_assets import AssetError
from dinkster_caches import MemoryLRUCache
from dinkster_compat_comfy import (
    NativeComponentHandle,
    ResidentLookupError,
    ResidentPool,
    comfy_resident_meta,
    register_resident_type,
)
from dinkster_compat_comfy.native_residency import NativeResidencyBusyError
from dinkster_memory import MemoryGovernor, PressureSignal
from dinkster_values import ResourcePins, TypeRegistry

VRAM0 = "vram:cuda:0"
VRAM1 = "vram:cuda:1"


class FakePatcher:
    """ModelPatcher-shaped: load_device + model_size."""

    def __init__(self, device: str, size: int) -> None:
        self.load_device = device
        self._size = size

    def model_size(self) -> int:
        return self._size


def make_pool(**kwargs: object) -> ResidentPool:
    return ResidentPool(cost_of=comfy_resident_meta, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_compat_unload_preserves_manager_and_pool_accounting(
    monkeypatch: pytest.MonkeyPatch, wrapped: bool, fails: bool
) -> None:
    from dinkster_compat_comfy.pool import resident_advisory_unload

    patcher = FakePatcher("cuda:0", 1000)
    resident = SimpleNamespace(patcher=patcher) if wrapped else patcher
    calls: list[object] = []

    def unload() -> None:
        calls.append(patcher)
        if fails:
            raise RuntimeError("device eviction failed")

    target = SimpleNamespace(model=patcher, model_unload=unload)
    other = SimpleNamespace(
        model=object(), model_unload=lambda: pytest.fail("unrelated model was unloaded")
    )
    loaded = [other, target]
    monkeypatch.setitem(
        sys.modules, "comfy.model_management", SimpleNamespace(current_loaded_models=loaded)
    )
    pool = make_pool(unload=resident_advisory_unload)
    rid = pool.rid_for(resident)

    freed = asyncio.run(pool.shed(PressureSignal(device=VRAM0, bytes_needed=1000)))

    assert calls == [patcher]
    assert loaded == ([other, target] if fails else [other])
    assert freed == (0 if fails else 1000)
    assert pool.footprint(VRAM0) == (1000 if fails else 0)
    assert pool.get(rid) is resident


def test_compat_unload_without_comfy_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    from dinkster_compat_comfy.pool import resident_advisory_unload

    monkeypatch.setitem(sys.modules, "comfy.model_management", None)
    resident_advisory_unload(FakePatcher("cuda:0", 1000))


# --- admission, identity, labels -----------------------------------------


def test_admission_reads_declared_cost_and_keeps_stable_ids() -> None:
    pool = make_pool()
    model = FakePatcher("cuda:0", 1000)
    rid = pool.rid_for(model)
    assert pool.rid_for(model) == rid  # table idempotence survives
    assert pool.footprint(VRAM0) == 1000
    assert pool.footprint(VRAM1) == 0
    assert pool.footprint("ram") == 1000  # the offload copy
    (item,) = pool.details()
    assert item.item_id == rid  # stable ID: the rid itself
    assert item.bytes_by_residency == {VRAM0: 1000, "ram": 1000}


def test_labels_come_from_the_loader_never_a_class_name() -> None:
    pool = make_pool()
    model = FakePatcher("cuda:0", 10)
    rid = pool.label(model, "sd15.safetensors")
    assert rid == pool.rid_for(model)  # label() admits, same identity
    (item,) = pool.details()
    assert item.display_name == "sd15.safetensors"
    assert "FakePatcher" not in item.display_name


def test_vae_source_is_observational_canonical_and_released_with_resident() -> None:
    pool = make_pool()
    vae = FakePatcher("cuda:0", 10)
    assert pool.source_for(vae) is None
    assert len(pool) == 0
    pool.label(vae, "vae.safetensors")
    digest = "blake3:" + "a" * 64
    pool.label_source(
        vae,
        digest=digest,
        name="vae.safetensors",
        latent_space="dinkster.sd15",
    )
    source = pool.source_for(vae)
    assert source is not None
    assert json.loads(source.hint()) == {
        "latentSpace": "dinkster.sd15",
        "sourceDigest": digest,
        "sourceName": "vae.safetensors",
        "version": 1,
    }
    with pytest.raises(AssetError, match="conflicting VAE source digest"):
        pool.label_source(vae, digest="blake3:" + "b" * 64)
    asyncio.run(pool.shed(PressureSignal(device="ram", bytes_needed=10)))
    assert pool.source_for(vae) is None


def test_different_checkpoint_digests_can_share_one_family_latent_space() -> None:
    pool = make_pool()
    first = FakePatcher("cuda:0", 10)
    second = FakePatcher("cuda:0", 10)
    pool.label(first, "first.safetensors")
    pool.label(second, "second.safetensors")
    pool.label_source(first, digest="blake3:" + "a" * 64, latent_space="dinkster.sdxl")
    pool.label_source(second, digest="blake3:" + "b" * 64, latent_space="dinkster.sdxl")
    first_source = pool.source_for(first)
    second_source = pool.source_for(second)
    assert first_source is not None and second_source is not None
    assert first_source.digest != second_source.digest
    assert first_source.latent_space == second_source.latent_space == "dinkster.sdxl"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": "data:text/plain,vae"}, "path-free"),
        ({"name": "file:vae.safetensors"}, "path-free"),
        ({"name": "http://example.test/vae"}, "path-free"),
        ({"logical_id": "../builtin-vae"}, "path-free"),
        ({"logical_id": "vae\x00id"}, "path-free"),
        ({"latent_space": "sdxl"}, "family or codec ID"),
        ({"latent_space": "dinkster.SDXL"}, "family or codec ID"),
    ],
)
def test_vae_source_rejects_url_path_control_and_arbitrary_latent_space(
    kwargs: dict[str, str], message: str
) -> None:
    pool = make_pool()
    vae = FakePatcher("cuda:0", 10)
    pool.rid_for(vae)
    with pytest.raises(AssetError, match=message):
        pool.label_source(vae, digest="blake3:" + "a" * 64, **kwargs)
    assert pool.source_for(vae) is None


def test_source_lookup_serializes_with_release_and_does_not_touch_use_order() -> None:
    pool = make_pool()
    vae = FakePatcher("cuda:0", 10)
    rid = pool.rid_for(vae)
    pool.label_source(vae, digest="blake3:" + "a" * 64, latent_space="dinkster.sd15")
    before = pool.propose_release(PressureSignal(device="ram", bytes_needed=10))[0].token
    assert pool.source_for(vae) is not None
    after = pool.propose_release(PressureSignal(device="ram", bytes_needed=10))[0].token
    assert after == before

    entered = threading.Event()
    finish = threading.Event()

    def block_release(_resource_id: str) -> None:
        entered.set()
        assert finish.wait(timeout=5)

    pool.register_invalidator(block_release)
    candidates = pool.propose_release(PressureSignal(device="ram", bytes_needed=10))
    with ThreadPoolExecutor(max_workers=2) as executor:
        releasing = executor.submit(lambda: asyncio.run(pool.release("ram", candidates)))
        assert entered.wait(timeout=5)
        lookup = executor.submit(pool.source_for, vae)
        assert not lookup.done()
        finish.set()
        assert releasing.result(timeout=5) == {rid: 10}
        assert lookup.result(timeout=5) is None
    assert pool.source_for(vae) is None


def test_removed_resident_lineage_cannot_reappear_on_a_new_admission() -> None:
    pool = make_pool()
    first = FakePatcher("cuda:0", 10)
    first_rid = pool.rid_for(first)
    pool.label_source(first, digest="blake3:" + "a" * 64, latent_space="dinkster.sd15")
    assert pool.remove(first_rid) is first
    assert pool.source_for(first) is None

    second = FakePatcher("cuda:0", 10)
    second_rid = pool.rid_for(second)
    assert second_rid != first_rid
    assert pool.source_for(second) is None


def test_unlabeled_residents_fall_back_to_rid_not_class_name() -> None:
    pool = make_pool()
    rid = pool.rid_for(FakePatcher("cuda:0", 10))
    (item,) = pool.details()
    assert rid[:12] in item.display_name
    assert "FakePatcher" not in item.display_name


def test_unreadable_cost_admits_with_no_footprint() -> None:
    def explode(obj: object) -> dict[str, object]:
        raise RuntimeError("foreign object")

    pool = ResidentPool(cost_of=explode)
    pool.rid_for(FakePatcher("cuda:0", 10))
    assert pool.footprint(VRAM0) == 0  # no cost beats a crash
    (item,) = pool.details()
    assert item.bytes_by_residency == {}


def test_loaded_state_query_distinguishes_loaded_unloaded_and_unknown() -> None:
    pool = make_pool()
    rid = pool.rid_for(FakePatcher("cuda:0", 500))
    resource_id = f"resident:{rid}"
    token_before = pool.propose_release(PressureSignal(device="ram", bytes_needed=500))[0].token
    assert pool.is_loaded(resource_id) is True
    token_after = pool.propose_release(PressureSignal(device="ram", bytes_needed=500))[0].token
    assert token_after == token_before  # observational: no last_used touch
    asyncio.run(pool.shed(PressureSignal(device=VRAM0, bytes_needed=500)))
    assert pool.is_loaded(resource_id) is False
    assert pool.is_loaded("resident:unknown") is None


# --- shedding: LRU, device-scoped, identity-preserving --------------------


def test_untargeted_pressure_unloads_least_recently_used_first() -> None:
    unloaded: list[object] = []
    pool = make_pool(unload=unloaded.append)
    old = FakePatcher("cuda:0", 400)
    fresh = FakePatcher("cuda:0", 400)
    rid_old = pool.rid_for(old)
    pool.rid_for(fresh)
    pool.get(rid_old)  # a stub resolving is a use: old is now the fresher
    freed = asyncio.run(pool.shed(PressureSignal(device=VRAM0, bytes_needed=100)))
    assert freed == 400
    assert unloaded == [fresh]  # LRU went first, and only as much as needed
    assert pool.footprint(VRAM0) == 400


def test_shedding_skips_residents_on_other_devices() -> None:
    unloaded: list[object] = []
    pool = make_pool(unload=unloaded.append)
    other = FakePatcher("cuda:1", 400)
    here = FakePatcher("cuda:0", 400)
    pool.rid_for(other)
    pool.rid_for(here)
    freed = asyncio.run(pool.shed(PressureSignal(device=VRAM0, bytes_needed=10_000)))
    assert freed == 400  # all it holds on cuda:0; never lies upward
    assert unloaded == [here]  # unloading cuda:1 frees no cuda:0 pressure


def test_stub_still_resolves_after_shedding() -> None:
    """The safety property: pressure never dangles a stub."""
    pool = make_pool()
    registry = TypeRegistry()
    register_resident_type(registry, "comfy.MODEL", pool)
    spec = registry.spec("comfy.MODEL")
    model = FakePatcher("cuda:0", 500)
    stub = spec.encode(model)
    asyncio.run(pool.shed(PressureSignal(device=VRAM0, bytes_needed=500)))
    assert spec.decode(stub) is model  # identity survived the shed
    rid = json.loads(stub)["residentId"]
    assert pool.get(rid) is model


def test_resolving_a_shed_resident_reloads_its_accounting() -> None:
    pool = make_pool()
    rid = pool.rid_for(FakePatcher("cuda:0", 500))
    asyncio.run(pool.shed(PressureSignal(device=VRAM0, bytes_needed=500)))
    assert pool.footprint(VRAM0) == 0
    assert pool.footprint("ram") == 500  # advisory unload keeps the ram copy
    (item,) = pool.details()
    assert item.bytes_by_residency == {"ram": 500}  # weights sit in host RAM
    pool.get(rid)  # resolution means imminent use: comfy reloads it
    assert pool.footprint(VRAM0) == 500
    (item,) = pool.details()
    assert item.bytes_by_residency == {VRAM0: 500, "ram": 500}


def test_already_unloaded_residents_free_nothing() -> None:
    calls: list[object] = []
    pool = make_pool(unload=calls.append)
    pool.rid_for(FakePatcher("cuda:0", 500))
    assert asyncio.run(pool.shed(PressureSignal(device=VRAM0, bytes_needed=500))) == 500
    assert (
        asyncio.run(pool.shed(PressureSignal(device=VRAM0, bytes_needed=500))) == 0
    )  # nothing loaded, nothing freed - never double-counted
    assert len(calls) == 1


def test_failing_unload_hook_preserves_accounting(caplog: pytest.LogCaptureFixture) -> None:
    def explode(obj: object) -> None:
        raise RuntimeError("driver said no")

    pool = make_pool(unload=explode)
    pool.rid_for(FakePatcher("cuda:0", 500))
    freed = asyncio.run(pool.shed(PressureSignal(device=VRAM0, bytes_needed=500)))
    assert freed == 0
    assert pool.footprint(VRAM0) == 500
    assert "DINKSTER_COMPAT_UNLOAD_FAILED" in caplog.text


# --- item-targeted pressure ------------------------------------------------


def test_item_pressure_unloads_exactly_the_named_residents() -> None:
    unloaded: list[object] = []
    pool = make_pool(unload=unloaded.append)
    keep = FakePatcher("cuda:0", 400)
    drop = FakePatcher("cuda:0", 400)
    pool.rid_for(keep)
    rid_drop = pool.rid_for(drop)
    freed = asyncio.run(pool.shed(PressureSignal(device=VRAM0, bytes_needed=0, items=(rid_drop,))))
    assert freed == 400
    assert unloaded == [drop]  # "unload this model" means this model
    assert pool.footprint(VRAM0) == 400


def test_unresolved_item_frees_nothing() -> None:
    pool = make_pool()
    pool.rid_for(FakePatcher("cuda:0", 400))
    freed = asyncio.run(
        pool.shed(PressureSignal(device=VRAM0, bytes_needed=400, items=("no-such",)))
    )
    assert freed == 0  # never evict a neighbor to satisfy a bad ID
    assert pool.footprint(VRAM0) == 400


# --- ram pressure: invalidate-then-release ---------------------------------


def test_ram_pressure_invalidates_references_then_releases() -> None:
    """The --cache-ram loop: freeing host RAM drops cache references
    first, the strong reference second - in that order, always."""
    order: list[str] = []
    pool = make_pool(unload=lambda obj: order.append("unload"))
    rid = pool.rid_for(FakePatcher("cuda:0", 500))
    pool.register_invalidator(lambda resource_id: order.append(f"drop {resource_id}"))
    freed = asyncio.run(pool.shed(PressureSignal(device="ram", bytes_needed=500)))
    assert freed == 500
    assert order == [f"drop resident:{rid}", "unload"]  # references first
    assert pool.footprint("ram") == 0
    assert len(pool) == 0  # the reference is gone
    with pytest.raises(ResidentLookupError):
        pool.get(rid)  # honest: released, invalidation made this unreachable


def test_release_refuses_a_candidate_used_since_proposal() -> None:
    """The use-clock token: a resident touched between propose_release and
    release was decided on stale information - compare-and-drop refuses,
    fresh state survives."""
    pool = make_pool()
    patcher = FakePatcher("cuda:0", 500)
    rid = pool.rid_for(patcher)
    candidates = pool.propose_release(PressureSignal(device="ram", bytes_needed=500))
    assert [c.item_id for c in candidates] == [rid]
    pool.get(rid)  # a use: the token the candidate carries is now stale
    outcome = asyncio.run(pool.release("ram", candidates))
    assert outcome == {}  # refused, nothing claimed
    assert pool.get(rid) is patcher  # the resident survived


def test_failing_invalidator_aborts_the_release() -> None:
    """A cache that may still hold a stub beats reclaimed bytes."""

    def explode(resource_id: str) -> None:
        raise RuntimeError("cache is mid-mutation")

    pool = make_pool()
    rid = pool.rid_for(FakePatcher("cuda:0", 500))
    pool.register_invalidator(explode)
    freed = asyncio.run(pool.shed(PressureSignal(device="ram", bytes_needed=500)))
    assert freed == 0  # refused: releasing could dangle a stub
    assert pool.get(rid) is not None  # the resident survived


def test_ram_pressure_refuses_a_resource_pinned_by_a_live_run() -> None:
    pool = make_pool()
    model = FakePatcher("cuda:0", 500)
    rid = pool.rid_for(model)
    resource_id = f"resident:{rid}"
    pins = ResourcePins()
    pool.register_pins(pins)
    assert pins.pin(resource_id)

    freed = asyncio.run(pool.shed(PressureSignal(device="ram", bytes_needed=500)))

    assert freed == 0
    assert pool.get(rid) is model
    pins.unpin(resource_id)
    assert asyncio.run(pool.shed(PressureSignal(device="ram", bytes_needed=500))) == 500


def test_ram_pressure_preserves_component_referenced_by_model_overlay() -> None:
    invalidated: list[str] = []
    released: list[object] = []

    class Component(NativeComponentHandle):
        def __init__(self) -> None:
            self.load_device = "cuda:0"
            self.live = True

        def model_size(self) -> int:
            return 500

        @contextmanager
        def terminal_release_guard(self):  # type: ignore[no-untyped-def]
            if self.live:
                raise NativeResidencyBusyError("component has a live dependent")
            yield

    component = Component()

    def terminal_release(obj: object) -> bool:
        released.append(obj)
        return True

    pool = make_pool(terminal_release=terminal_release)
    rid = pool.rid_for(component)
    pool.register_invalidator(invalidated.append)

    freed = asyncio.run(pool.shed(PressureSignal(device="ram", bytes_needed=500)))

    assert freed == 0
    assert invalidated == []
    assert released == []
    assert pool.get(rid) is component

    component.live = False
    freed = asyncio.run(pool.shed(PressureSignal(device="ram", bytes_needed=500)))
    assert freed == 500
    assert invalidated == [f"resident:{rid}"]
    assert released == [component]


def test_ram_pressure_releases_lru_first_and_only_as_needed() -> None:
    pool = make_pool()
    old = FakePatcher("cuda:0", 400)
    fresh = FakePatcher("cuda:0", 400)
    rid_old = pool.rid_for(old)
    rid_fresh = pool.rid_for(fresh)
    pool.get(rid_old)  # old becomes the fresher of the two
    freed = asyncio.run(pool.shed(PressureSignal(device="ram", bytes_needed=100)))
    assert freed == 400
    assert pool.get(rid_old) is old  # survivor
    with pytest.raises(ResidentLookupError):
        pool.get(rid_fresh)  # LRU went first


def test_item_targeted_ram_pressure_releases_exactly_the_named() -> None:
    pool = make_pool()
    keep = FakePatcher("cuda:0", 400)
    drop = FakePatcher("cuda:0", 400)
    rid_keep = pool.rid_for(keep)
    rid_drop = pool.rid_for(drop)
    freed = asyncio.run(
        pool.shed(PressureSignal(device="ram", bytes_needed=400, items=(rid_drop,)))
    )
    assert freed == 400
    assert pool.get(rid_keep) is keep
    with pytest.raises(ResidentLookupError):
        pool.get(rid_drop)


def test_full_release_includes_zero_cost_and_refuses_pins() -> None:
    async def scenario() -> None:
        pool = make_pool()
        zero = FakePatcher("cuda:0", 0)
        pinned = FakePatcher("cuda:0", 400)
        zero_id = pool.rid_for(zero)
        pinned_id = pool.rid_for(pinned)
        pins = ResourcePins()
        pool.register_pins(pins)
        assert pins.pin(f"resident:{pinned_id}")

        result = await pool.full_release()
        assert result.status == "busy"
        with pytest.raises(ResidentLookupError):
            pool.get(zero_id)
        assert pool.get(pinned_id) is pinned

        pins.unpin(f"resident:{pinned_id}")
        assert (await pool.full_release()).status == "complete"
        assert (await pool.full_release()).status == "complete"

    asyncio.run(scenario())


def test_full_release_ignores_duplicate_pin_registration_but_refuses_distinct_active_pins() -> None:
    async def scenario() -> None:
        pool = make_pool()
        zero = FakePatcher("cuda:0", 0)
        zero_id = pool.rid_for(zero)
        pins = ResourcePins()
        pool.register_pins(pins)
        pool.register_pins(pins)

        assert (await pool.full_release()).status == "complete"
        with pytest.raises(ResidentLookupError):
            pool.get(zero_id)

        pinned = FakePatcher("cuda:0", 400)
        pinned_id = pool.rid_for(pinned)
        active_pins = ResourcePins()
        pool.register_pins(active_pins)
        assert active_pins.pin(f"resident:{pinned_id}")
        result = await pool.full_release()
        assert result.status == "busy"
        assert pool.get(pinned_id) is pinned

        active_pins.unpin(f"resident:{pinned_id}")
        assert (await pool.full_release()).status == "complete"
        assert (await pool.full_release()).status == "complete"

    asyncio.run(scenario())


def test_full_release_reports_cleanup_failure() -> None:
    def fail_release(_resident: object) -> bool:
        raise RuntimeError("cleanup failed")

    pool = make_pool(terminal_release=fail_release)
    model = FakePatcher("cuda:0", 400)
    rid = pool.rid_for(model)

    result = asyncio.run(pool.full_release())

    assert result.status == "error"
    assert result.error is not None
    assert "terminal release failed" in result.error
    assert pool.get(rid) is model


def test_released_resident_recomputes_through_cache_miss() -> None:
    """The recompute path needs no engine machinery: invalidation makes
    the next request a cache miss, and the loader runs again."""

    async def scenario() -> None:
        pool = make_pool()
        registry = TypeRegistry()
        register_resident_type(registry, "comfy.MODEL", pool)
        cache = MemoryLRUCache()
        pool.register_invalidator(cache.drop_referencing)

        loads = 0

        async def load_or_cached() -> object:
            nonlocal loads
            entry = await cache.get("loader-key")
            if entry is not None:
                return entry["out"].resolve()
            loads += 1
            model = FakePatcher("cuda:0", 500)
            value = registry.wrap("comfy.MODEL", model)
            await cache.put("loader-key", {"out": value})
            return model

        first = await load_or_cached()
        assert await load_or_cached() is first  # cache hit, no reload
        assert loads == 1

        await pool.shed(PressureSignal(device="ram", bytes_needed=500))
        second = await load_or_cached()  # miss -> reload, never a dangle
        assert loads == 2
        assert second is not first

    asyncio.run(scenario())


# --- governor integration ---------------------------------------------------


def test_pool_is_a_detail_contract_consumer_under_the_governor() -> None:
    async def scenario() -> None:
        pool = make_pool()
        governor = MemoryGovernor({VRAM0: 1000})
        governor.register_shedder(pool, name="comfy-models")
        model = FakePatcher("cuda:0", 800)
        rid = pool.label(model, "sd15.safetensors")

        status = governor.status()
        assert status[VRAM0]["consumers"] == {"comfy-models": 800}
        details = governor.details()
        (item,) = details["comfy-models"]
        assert item.item_id == rid
        assert item.display_name == "sd15.safetensors"

        # Item-targeted trim through the governor's own routing.
        freed = await governor.shed(VRAM0, 800, consumers=["comfy-models"], items=[rid])
        assert freed == 800
        assert pool.footprint(VRAM0) == 0
        assert pool.get(rid) is model  # identity survived, panel-driven too

    asyncio.run(scenario())


def test_ram_budget_end_to_end_cache_then_pool() -> None:
    """The whole --cache-ram loop under one governor: a RAM reservation
    that does not fit sheds the result cache first (priority order), then
    the pool releases the model - cache references invalidated before the
    reference drops, so nothing ever dangles."""

    async def scenario() -> None:
        pool = make_pool()
        registry = TypeRegistry()
        register_resident_type(registry, "comfy.MODEL", pool)
        cache = MemoryLRUCache()
        pool.register_invalidator(cache.drop_referencing)
        governor = MemoryGovernor({"ram": 1000})
        governor.register_shedder(cache, name="results")  # caches shed first
        governor.register_shedder(pool, name="comfy-models", priority=1)

        model = FakePatcher("cuda:0", 800)
        await cache.put("loader-key", {"out": registry.wrap("comfy.MODEL", model)})  # type: ignore[arg-type]
        assert pool.footprint("ram") == 800  # the pool owns the bytes
        assert cache.footprint("ram") == 0  # the cache holds a reference

        async with governor.reserve("ram", 600):
            assert pool.footprint("ram") == 0  # model released
            assert await cache.get("loader-key") is None  # reference gone first

    asyncio.run(scenario())


def test_governor_pressure_makes_room_for_a_reservation() -> None:
    async def scenario() -> None:
        pool = make_pool()
        governor = MemoryGovernor({VRAM0: 1000})
        governor.register_shedder(pool, name="comfy-models")
        rid = pool.rid_for(FakePatcher("cuda:0", 800))
        async with governor.reserve(VRAM0, 600):
            # 800 held + 600 wanted > 1000 budget: the pool got pressure.
            assert pool.footprint(VRAM0) == 0
        assert pool.get(rid) is not None  # and the resident survived it

    asyncio.run(scenario())


# --- telemetry probe ---------------------------------------------------------


def test_torch_vram_telemetry_is_honest_about_absence() -> None:
    from dinkster_compat_comfy import torch_vram_telemetry

    assert torch_vram_telemetry("ram") is None  # not a vram class
    assert torch_vram_telemetry("vram:npu:0") is None  # not a probed family
    for key in (VRAM0, "vram:xpu:0"):
        measured = torch_vram_telemetry(key)
        # Environment-dependent: with the backend, real numbers; without,
        # None - never a fake zero either way.
        if measured is not None:
            assert measured.free_bytes >= 0
            assert measured.total_bytes >= measured.free_bytes
