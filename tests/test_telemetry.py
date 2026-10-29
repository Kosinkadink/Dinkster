"""Worker memory telemetry (interprocess memory, slice 2): a pack's
measured device memory crosses the boundary in hello/memoryReport/
memoryShedResult frames, lands DeviceMap-translated in a parent-side
ReportedTelemetry store, and surfaces through governor.status() beside
declared budgets. INFORMATIONAL ONLY: measured values never touch
admission - physical free is global while allocator-reclaimable bytes
are process-local, so measured-free admission math would double-count
(Oracle review 2026-07-26; ROADMAP "Memory governance (interprocess)").
"""

from __future__ import annotations

import asyncio
import importlib
import sys
import types
from pathlib import Path
from typing import Any, cast

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_compat_comfy import vram_telemetry_snapshot
from dinkster_engine import Engine, ExecutionError
from dinkster_graph import Graph, GraphNode
from dinkster_memory import MeasuredMemory, MemoryGovernor, ReportedTelemetry
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import DeviceMap, IsolatedWorker, ManifestError, load_manifest

TESTS_DIR = Path(__file__).parent

VRAM0 = "vram:cuda:0"
INITIAL = MeasuredMemory(free_bytes=8_000, total_bytes=10_000)


# -- ReportedTelemetry store ------------------------------------------------


def test_reported_telemetry_store() -> None:
    store = ReportedTelemetry()
    assert store.probe(VRAM0) is None
    assert store.devices() == frozenset()

    a, b = object(), object()
    store.update(a, {VRAM0: MeasuredMemory(free_bytes=100, total_bytes=200)})
    store.update(b, {"ram": MeasuredMemory(free_bytes=5, total_bytes=10)})
    assert store.probe(VRAM0) == MeasuredMemory(free_bytes=100, total_bytes=200)
    assert store.devices() == frozenset({VRAM0, "ram"})

    # Freshest source wins when two sources report the same device.
    store.update(b, {VRAM0: MeasuredMemory(free_bytes=50, total_bytes=200)})
    assert store.probe(VRAM0) == MeasuredMemory(free_bytes=50, total_bytes=200)

    # Snapshots replace whole: a device absent from the update is gone.
    store.update(b, {"ram": MeasuredMemory(free_bytes=5, total_bytes=10)})
    assert store.probe(VRAM0) == MeasuredMemory(free_bytes=100, total_bytes=200)

    store.clear(a)
    assert store.probe(VRAM0) is None
    store.clear(a)  # unknown/already-cleared sources are ignored
    assert store.devices() == frozenset({"ram"})


def test_measured_memory_wire_is_additive_in_both_compatibility_directions() -> None:
    # A new parent accepts an old worker's two-field value.
    assert MeasuredMemory.from_wire({"freeBytes": 80, "totalBytes": 100}) == MeasuredMemory(
        free_bytes=80,
        total_bytes=100,
    )

    structured = MeasuredMemory(
        free_bytes=80,
        total_bytes=100,
        driver_free_bytes=50,
        allocator_reclaimable_bytes=10,
        dynamic_evictable_bytes=20,
        dynamic_pinned_bytes=30,
    )
    wire = structured.to_wire()
    assert wire is not None

    # An old parent reads its known fields and ignores a new worker's additions.
    assert (wire["freeBytes"], wire["totalBytes"]) == (80, 100)
    assert MeasuredMemory.from_wire({**wire, "futureComponentBytes": 40}) == structured


@pytest.mark.parametrize(
    "wire",
    [
        {"freeBytes": -1, "totalBytes": 100},
        {"freeBytes": 101, "totalBytes": 100},
        {"freeBytes": 80, "totalBytes": 100, "driverFreeBytes": -1},
        {"freeBytes": 80, "totalBytes": 100, "allocatorReclaimableBytes": True},
        {"freeBytes": 80, "totalBytes": 100, "dynamicEvictableBytes": "20"},
        {"freeBytes": 80, "totalBytes": 100, "dynamicPinnedBytes": None},
        {"freeBytes": 80, "totalBytes": 100, "dynamicPinnedBytes": -1},
    ],
)
def test_measured_memory_wire_drops_malformed_readings(wire: dict[str, object]) -> None:
    assert MeasuredMemory.from_wire(wire) is None


def test_measured_memory_encoder_rejects_invalid_component() -> None:
    assert (
        MeasuredMemory(
            free_bytes=80,
            total_bytes=100,
            dynamic_pinned_bytes=-1,
        ).to_wire()
        is None
    )


# -- manifest entry ----------------------------------------------------------


def test_manifest_telemetry_entry_is_optional_and_validated(tmp_path: Path) -> None:
    plain = tmp_path / "plain.toml"
    plain.write_text('[pack]\nname = "p"\n\n[pack.entry]\nnodes = "m:N"\n')
    assert load_manifest(plain).telemetry_entry is None

    with_probe = tmp_path / "probe.toml"
    with_probe.write_text(
        '[pack]\nname = "p"\n\n[pack.entry]\nnodes = "m:N"\ntelemetry = "m:measure"\n'
    )
    assert load_manifest(with_probe).telemetry_entry == "m:measure"

    malformed = tmp_path / "bad.toml"
    malformed.write_text(
        '[pack]\nname = "p"\n\n[pack.entry]\nnodes = "m:N"\ntelemetry = "noattr"\n'
    )
    with pytest.raises(ManifestError, match="telemetry"):
        load_manifest(malformed)


# -- boundary crossing --------------------------------------------------------


def write_manifest(
    tmp_path: Path, telemetry: str | None = "memorypack_nodes:memory_telemetry"
) -> Path:
    manifest = tmp_path / "dinkster-pack.toml"
    text = (
        '[pack]\nname = "memorypack"\n\n[pack.entry]\n'
        'nodes = "memorypack_nodes:NODES"\ntypes = "memorypack_nodes:register_types"\n'
        'consumers = "memorypack_nodes:memory_consumers"\n'
    )
    if telemetry is not None:
        text += f'telemetry = "{telemetry}"\n'
    manifest.write_text(text)
    return manifest


def telemetry_worker(
    tmp_path: Path,
    registry: TypeRegistry,
    store: ReportedTelemetry,
    *,
    telemetry: str | None = "memorypack_nodes:memory_telemetry",
    **kwargs: object,
) -> IsolatedWorker:
    return IsolatedWorker(
        write_manifest(tmp_path, telemetry),
        registry,
        extra_env={"PYTHONPATH": str(TESTS_DIR)},
        telemetry=store,
        **kwargs,  # type: ignore[arg-type]
    )


def core_registry() -> TypeRegistry:
    registry = TypeRegistry()
    register_core_types(registry)
    return registry


def make_engine(registry: TypeRegistry, worker: IsolatedWorker) -> Engine:
    return Engine(
        schemas=dict(worker.schemas),
        registry=registry,
        worker=worker,
        cache=MemoryLRUCache(),
    )


def load_graph(vram: int = 400) -> Graph:
    return Graph(
        nodes={
            "load": GraphNode("mem.load", {"name": "sd15.safetensors", "vram": vram, "ram": 300})
        }
    )


async def eventually(predicate: Any, timeout: float = 8.0) -> bool:
    """Snapshot pushes race the invocation result; poll for the frame."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            return False
        await asyncio.sleep(0.02)
    return True


def test_hello_lands_initial_measurements(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = ReportedTelemetry()
        worker = telemetry_worker(tmp_path, core_registry(), store)
        await worker.start()
        try:
            # No invocation needed: measurements rode the handshake.
            assert store.probe(VRAM0) == INITIAL
            assert store.devices() == frozenset({VRAM0})
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_hello_lands_structured_cuda_measurement(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = ReportedTelemetry()
        worker = telemetry_worker(
            tmp_path,
            core_registry(),
            store,
            telemetry="memorypack_nodes:structured_memory_telemetry",
        )
        await worker.start()
        try:
            assert store.probe(VRAM0) == MeasuredMemory(
                free_bytes=8_000,
                total_bytes=10_000,
                driver_free_bytes=6_000,
                allocator_reclaimable_bytes=1_000,
                dynamic_evictable_bytes=1_000,
                dynamic_pinned_bytes=2_000,
            )
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_reports_refresh_measurements(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = ReportedTelemetry()
        registry = core_registry()
        worker = telemetry_worker(tmp_path, registry, store)
        await worker.start()
        try:
            engine = make_engine(registry, worker)
            await engine.run(load_graph(vram=400), ["load"])
            # The load consumed fake device memory; the post-invocation
            # memoryReport carries the decremented free.
            assert await eventually(
                lambda: (
                    (m := store.probe(VRAM0)) is not None
                    and m.free_bytes == INITIAL.free_bytes - 400
                )
            )
            measured = store.probe(VRAM0)
            assert measured is not None
            assert measured.total_bytes == INITIAL.total_bytes
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_device_map_translates_measurements(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = ReportedTelemetry()
        worker = telemetry_worker(
            tmp_path,
            core_registry(),
            store,
            device_map=DeviceMap({"cuda:0": "cuda:1"}),
        )
        await worker.start()
        try:
            # The child honestly reports its own cuda:0; the parent's map
            # says that silicon is cuda:1 here - measurements must land on
            # the translated key, never collide with the local cuda:0.
            assert store.probe("vram:cuda:1") == INITIAL
            assert store.probe(VRAM0) is None
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_malformed_telemetry_entries_are_dropped(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = ReportedTelemetry()
        registry = core_registry()
        worker = telemetry_worker(
            tmp_path, registry, store, telemetry="memorypack_nodes:malformed_telemetry"
        )
        await worker.start()
        try:
            # The valid sibling survives; every garbage entry is gone.
            assert store.probe(VRAM0) == INITIAL
            assert store.devices() == frozenset({VRAM0})
            # The worker is alive and invokable despite the bad probe.
            engine = make_engine(registry, worker)
            result = await engine.run(load_graph(), ["load"])
            assert "load" in result.outputs
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_raising_probe_measures_nothing_and_worker_survives(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = ReportedTelemetry()
        registry = core_registry()
        worker = telemetry_worker(
            tmp_path, registry, store, telemetry="memorypack_nodes:raising_telemetry"
        )
        await worker.start()
        try:
            assert store.devices() == frozenset()
            engine = make_engine(registry, worker)
            result = await engine.run(load_graph(), ["load"])
            assert "load" in result.outputs
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_pack_without_telemetry_reports_nothing(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = ReportedTelemetry()
        registry = core_registry()
        governor = MemoryGovernor()
        worker = telemetry_worker(tmp_path, registry, store, telemetry=None, governor=governor)
        await worker.start()
        try:
            engine = make_engine(registry, worker)
            result = await engine.run(load_graph(vram=400), ["load"])
            assert "load" in result.outputs
            # The consumer report still lands (old-worker shape) while the
            # telemetry store stays empty: the "measured" field is additive
            # and its absence is compatibility, not an error.
            assert await eventually(lambda: governor.footprint(VRAM0) == 400)
            assert store.devices() == frozenset()
            assert store.probe(VRAM0) is None
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_close_clears_measurements(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = ReportedTelemetry()
        worker = telemetry_worker(tmp_path, core_registry(), store)
        await worker.start()
        assert store.probe(VRAM0) == INITIAL
        await worker.close()
        # A closed worker's measurements are gone, not stale.
        assert store.probe(VRAM0) is None
        assert store.devices() == frozenset()

    asyncio.run(scenario())


def test_worker_death_clears_measurements(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = ReportedTelemetry()
        registry = core_registry()
        worker = telemetry_worker(tmp_path, registry, store)
        await worker.start()
        try:
            assert store.probe(VRAM0) == INITIAL
            engine = make_engine(registry, worker)
            crash = Graph(nodes={"n": GraphNode("mem.exit", {"code": 3})})
            with pytest.raises(ExecutionError):
                await engine.run(crash, ["n"])
            # Read-loop death (not orderly close) must drop the snapshot
            # immediately: a dead worker's "free" is a lie.
            assert await eventually(lambda: store.probe(VRAM0) is None)
            assert store.devices() == frozenset()
        finally:
            await worker.close()

    asyncio.run(scenario())


# -- governor: status enumeration + admission unchanged ----------------------


def test_status_enumerates_telemetry_devices() -> None:
    store = ReportedTelemetry()
    governor = MemoryGovernor({"ram": 1000}, telemetry=store.probe, telemetry_devices=store.devices)
    assert VRAM0 not in governor.status()

    store.update(object(), {VRAM0: MeasuredMemory(free_bytes=7, total_bytes=9)})
    status = governor.status()
    assert set(status) == {"ram", VRAM0}
    # Measured but unbudgeted: visible, honest, no invented budget.
    assert status[VRAM0]["budgetBytes"] is None
    assert status[VRAM0]["measured"] == {"freeBytes": 7, "totalBytes": 9}
    assert status[VRAM0]["availableBytes"] is None
    assert status["ram"]["measured"] is None  # nobody measures ram


def test_status_survives_a_raising_device_enumeration() -> None:
    def exploding() -> frozenset[str]:
        raise RuntimeError("boom")

    governor = MemoryGovernor({"ram": 1000}, telemetry_devices=exploding)
    assert set(governor.status()) == {"ram"}


def test_measured_values_never_change_admission() -> None:
    async def scenario() -> None:
        store = ReportedTelemetry()
        governor = MemoryGovernor(
            {"ram": 1000}, telemetry=store.probe, telemetry_devices=store.devices
        )
        # The device itself claims almost nothing is free - declared math
        # must not care (measured free is global, allocator-reclaimable
        # bytes are process-local; mixing them double-counts).
        store.update(
            object(),
            {
                "ram": MeasuredMemory(free_bytes=1, total_bytes=1000),
                VRAM0: MeasuredMemory(free_bytes=1, total_bytes=10),
            },
        )
        assert governor.available("ram") == 1000
        async with governor.reserve("ram", 900):
            assert governor.reserved("ram") == 900
        # Unbudgeted devices still admit everything, measured or not.
        assert governor.available(VRAM0) is None
        async with governor.reserve(VRAM0, 10**12):
            pass

    asyncio.run(scenario())


# -- compat pack probe --------------------------------------------------------


def test_vram_telemetry_snapshot_without_torch_is_empty() -> None:
    # The root venv is deliberately torch-free: calling the real probe
    # here proves the defensive import path returns honest absence.
    assert "torch" not in sys.modules
    assert vram_telemetry_snapshot() == {}


def test_vram_telemetry_snapshot_with_fake_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = {"cuda:0": (6_000, 24_000), "cuda:1": (12_000, 24_000)}

    fake = types.ModuleType("torch")
    fake_any = cast("Any", fake)
    fake_any.device = lambda spec, index=None: spec if index is None else f"{spec}:{index}"
    fake_any.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        mem_get_info=lambda device: info[str(device)],
    )
    monkeypatch.setitem(sys.modules, "torch", fake)

    assert vram_telemetry_snapshot() == {
        "vram:cuda:0": MeasuredMemory(free_bytes=6_000, total_bytes=24_000),
        "vram:cuda:1": MeasuredMemory(free_bytes=12_000, total_bytes=24_000),
    }


@pytest.mark.parametrize("failure", [None, ImportError("missing"), RuntimeError("failed")])
def test_vram_telemetry_snapshot_uses_structured_capacity_or_driver_fallback(
    failure: Exception | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = types.ModuleType("torch")
    fake_any = cast("Any", fake)
    fake_any.device = lambda spec, index=None: spec if index is None else f"{spec}:{index}"
    fake_any.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        mem_get_info=lambda _device: (6_000, 24_000),
    )

    def dynamic_snapshot(device: object) -> object:
        assert device == "cuda:0"
        if failure is not None:
            raise failure
        return types.SimpleNamespace(
            free_bytes=7_277,
            total_bytes=24_000,
            driver_free_bytes=6_000,
            allocator_reclaimable_bytes=500,
            dynamic_evictable_bytes=777,
            dynamic_pinned_bytes=1_234,
        )

    inference = types.SimpleNamespace(dynamic_cuda_memory_snapshot=dynamic_snapshot)

    def fake_import(name: str) -> object:
        if name == "torch":
            return fake
        if name == "dinkster_inference_torch":
            if isinstance(failure, ImportError):
                raise failure
            return inference
        raise AssertionError(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    expected = (
        MeasuredMemory(
            free_bytes=7_277,
            total_bytes=24_000,
            driver_free_bytes=6_000,
            allocator_reclaimable_bytes=500,
            dynamic_evictable_bytes=777,
            dynamic_pinned_bytes=1_234,
        )
        if failure is None
        else MeasuredMemory(free_bytes=6_000, total_bytes=24_000)
    )
    assert vram_telemetry_snapshot() == {"vram:cuda:0": expected}


def test_vram_telemetry_snapshot_without_cuda_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = types.ModuleType("torch")
    cast("Any", fake).cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert vram_telemetry_snapshot() == {}


def test_vram_telemetry_snapshot_with_fake_xpu_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A host with only XPU devices (no CUDA namespace at all) still reports.
    info = {"xpu:0": (5_000, 16_000)}

    fake = types.ModuleType("torch")
    fake_any = cast("Any", fake)
    fake_any.device = lambda spec, index=None: spec if index is None else f"{spec}:{index}"
    fake_any.xpu = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        mem_get_info=lambda device: info[str(device)],
    )
    monkeypatch.setitem(sys.modules, "torch", fake)

    assert vram_telemetry_snapshot() == {
        "vram:xpu:0": MeasuredMemory(free_bytes=5_000, total_bytes=16_000),
    }


def test_vram_telemetry_snapshot_reports_cuda_and_xpu_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda_info = {"cuda:0": (6_000, 24_000)}
    xpu_info = {"xpu:0": (5_000, 16_000)}

    fake = types.ModuleType("torch")
    fake_any = cast("Any", fake)
    fake_any.device = lambda spec, index=None: spec if index is None else f"{spec}:{index}"
    fake_any.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        mem_get_info=lambda device: cuda_info[str(device)],
    )
    fake_any.xpu = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        mem_get_info=lambda device: xpu_info[str(device)],
    )
    monkeypatch.setitem(sys.modules, "torch", fake)

    assert vram_telemetry_snapshot() == {
        "vram:cuda:0": MeasuredMemory(free_bytes=6_000, total_bytes=24_000),
        "vram:xpu:0": MeasuredMemory(free_bytes=5_000, total_bytes=16_000),
    }


@pytest.mark.parametrize("failure", [None, ImportError("missing"), RuntimeError("failed")])
def test_vram_telemetry_snapshot_uses_structured_xpu_capacity_or_driver_fallback(
    failure: Exception | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = types.ModuleType("torch")
    fake_any = cast("Any", fake)
    fake_any.device = lambda spec, index=None: spec if index is None else f"{spec}:{index}"
    fake_any.xpu = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        mem_get_info=lambda _device: (5_000, 16_000),
    )

    def structured_snapshot(device: object) -> object:
        assert device == "xpu:0"
        if failure is not None:
            raise failure
        return types.SimpleNamespace(
            free_bytes=5_400,
            total_bytes=16_000,
            driver_free_bytes=5_000,
            allocator_reclaimable_bytes=400,
            driver_reported=True,
        )

    inference = types.SimpleNamespace(xpu_memory_snapshot=structured_snapshot)

    def fake_import(name: str) -> object:
        if name == "torch":
            return fake
        if name == "dinkster_inference_torch":
            if isinstance(failure, ImportError):
                raise failure
            return inference
        raise AssertionError(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    expected = (
        MeasuredMemory(
            free_bytes=5_400,
            total_bytes=16_000,
            driver_free_bytes=5_000,
            allocator_reclaimable_bytes=400,
        )
        if failure is None
        else MeasuredMemory(free_bytes=5_000, total_bytes=16_000)
    )
    assert vram_telemetry_snapshot() == {"vram:xpu:0": expected}


def test_vram_telemetry_snapshot_omits_xpu_without_mem_get_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A torch whose XPU namespace lacks mem_get_info measures nothing:
    # honest absence, never a fake zero.
    fake = types.ModuleType("torch")
    fake_any = cast("Any", fake)
    fake_any.device = lambda spec, index=None: spec if index is None else f"{spec}:{index}"
    fake_any.xpu = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
    )
    monkeypatch.setitem(sys.modules, "torch", fake)

    assert vram_telemetry_snapshot() == {}


def test_vram_telemetry_snapshot_rejects_structured_xpu_free_without_driver_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A structured snapshot whose free bytes were derived from allocator
    # reserve (not the driver's own report) is not measured telemetry.
    # Without mem_get_info the driver fallback measures nothing either.
    fake = types.ModuleType("torch")
    fake_any = cast("Any", fake)
    fake_any.device = lambda spec, index=None: spec if index is None else f"{spec}:{index}"
    fake_any.xpu = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
    )

    inference = types.SimpleNamespace(
        xpu_memory_snapshot=lambda _device: types.SimpleNamespace(
            free_bytes=16_000,
            total_bytes=16_000,
            driver_free_bytes=15_600,
            allocator_reclaimable_bytes=400,
            driver_reported=False,
        )
    )

    def fake_import(name: str) -> object:
        if name == "torch":
            return fake
        if name == "dinkster_inference_torch":
            return inference
        raise AssertionError(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    assert vram_telemetry_snapshot() == {}


def test_vram_telemetry_snapshot_survives_a_malformed_driver_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One family's driver returning garbage must not break the other.
    fake = types.ModuleType("torch")
    fake_any = cast("Any", fake)
    fake_any.device = lambda spec, index=None: spec if index is None else f"{spec}:{index}"
    fake_any.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        mem_get_info=lambda _device: (6_000, 24_000),
    )
    fake_any.xpu = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        mem_get_info=lambda _device: ("garbage", None),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)

    assert vram_telemetry_snapshot() == {
        "vram:cuda:0": MeasuredMemory(free_bytes=6_000, total_bytes=24_000),
    }
