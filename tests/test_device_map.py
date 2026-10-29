"""Worker-qualified device namespacing (DESIGN 3.10).

A worker pinned with CUDA_VISIBLE_DEVICES=1 honestly reports cuda:0 - in
*its* namespace. The parent launched it with that env, so the parent owns
the translation: DeviceMap rewrites device facts as they cross into the
parent - residency meta (admission lanes), cost meta (governor budgets),
and reservation lease requests. Without it, two pinned workers would
collide on one lane while occupying different silicon.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dinkster_engine import Invocation
from dinkster_memory import GovernorReservationService, MemoryGovernor
from dinkster_values import (
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    COST_META_KEY,
    RESOURCES_META_KEY,
    TypeRegistry,
    Value,
    ValueMeta,
    register_core_types,
)
from dinkster_workers import DeviceMap, IsolatedWorker

TESTS_DIR = Path(__file__).parent

CUDA1_MAP = DeviceMap({"cuda:0": "cuda:1"})


# -- pure mapping ---------------------------------------------------------


def test_device_and_residency_mapping() -> None:
    assert CUDA1_MAP.device("cuda:0") == "cuda:1"
    assert CUDA1_MAP.device("cuda:7") == "cuda:7"  # unmapped passes through
    assert CUDA1_MAP.residency("vram:cuda:0") == "vram:cuda:1"
    assert CUDA1_MAP.residency("ram") == "ram"
    assert CUDA1_MAP.residency("disk") == "disk"


def test_qualifier_suffixes_every_unmapped_fact() -> None:
    """The remote posture: another machine's devices - and its ram - must
    never land on local budgets or lanes by default."""
    remote = DeviceMap({}, qualifier="box1")
    assert remote.device("cuda:0") == "cuda:0@box1"
    assert remote.residency("vram:cuda:0") == "vram:cuda:0@box1"
    assert remote.residency("ram") == "ram@box1"
    assert remote.residency("disk") == "disk@box1"
    # An explicit mapping entry still wins: deliberate unification.
    unified = DeviceMap({"cuda:0": "cuda:1"}, qualifier="box1")
    assert unified.device("cuda:0") == "cuda:1"
    assert unified.device("cuda:9") == "cuda:9@box1"


def test_qualifier_inversion_refuses_local_facts() -> None:
    """Pressure aimed at a local device must never be forwarded to a
    remote worker as if it named the remote's silicon."""
    remote = DeviceMap({}, qualifier="box1")
    assert remote.to_child_residency("vram:cuda:0@box1") == "vram:cuda:0"
    assert remote.to_child_residency("ram@box1") == "ram"
    assert remote.to_child_residency("vram:cuda:0") is None  # local GPU
    assert remote.to_child_residency("ram") is None  # local ram
    assert remote.to_child_device("cuda:1@box1") == "cuda:1"
    assert remote.to_child_device("cuda:1") is None


def test_pinned_map_inversion_matches_relay_semantics() -> None:
    """The un-qualified inversion the relay always used, now on DeviceMap:
    mapped devices invert, a parent device whose name means different
    silicon inside the child is refused, everything else passes."""
    assert CUDA1_MAP.to_child_residency("vram:cuda:1") == "vram:cuda:0"
    assert CUDA1_MAP.to_child_residency("vram:cuda:0") is None  # child's name
    assert CUDA1_MAP.to_child_residency("vram:cuda:7") == "vram:cuda:7"
    assert CUDA1_MAP.to_child_residency("ram") == "ram"


def wrapped_gpu_value(resources: object, cost: object) -> Value:
    registry = TypeRegistry()
    registry.register(
        "t.gpu",
        meta=lambda obj: {RESOURCES_META_KEY: resources, COST_META_KEY: cost},
    )
    return registry.wrap("t.gpu", object())


def test_value_meta_mapping_rewrites_devices_only() -> None:
    value = wrapped_gpu_value({"gpu": "cuda:0"}, {"vram:cuda:0": 128, "ram": 16})
    mapped = CUDA1_MAP.value(value)
    assert mapped.meta.get(RESOURCES_META_KEY) == {"gpu": "cuda:1"}
    assert mapped.meta.get(COST_META_KEY) == {"vram:cuda:1": 128, "ram": 16}
    # Identity is device-free (hazard H4): fingerprint and payload untouched.
    assert mapped.fingerprint == value.fingerprint
    assert mapped.payload is value.payload
    # The original value is not mutated.
    assert value.meta.get(RESOURCES_META_KEY) == {"gpu": "cuda:0"}


def test_value_meta_mapping_handles_multi_device_residency() -> None:
    value = wrapped_gpu_value(
        {"gpu": ("cuda:0", "cuda:2")},
        {"vram:cuda:0": 64, "vram:cuda:2": 64},
    )
    mapped = DeviceMap({"cuda:0": "cuda:1", "cuda:2": "cuda:3"}).value(value)
    assert mapped.meta.get(RESOURCES_META_KEY) == {"gpu": ("cuda:1", "cuda:3")}
    assert mapped.meta.get(COST_META_KEY) == {"vram:cuda:1": 64, "vram:cuda:3": 64}


def test_identity_map_returns_value_unchanged() -> None:
    value = wrapped_gpu_value({"gpu": "cuda:0"}, {"vram:cuda:0": 128})
    assert DeviceMap({}).value(value) is value
    plain = Value(type_id="t.plain", fingerprint="f", meta=ValueMeta({}), payload=value.payload)
    assert CUDA1_MAP.value(plain) is plain


# -- across the isolated boundary ------------------------------------------


def write_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "isopack"\n\n[pack.entry]\n'
        'nodes = "isopack_nodes:NODES"\ntypes = "isopack_nodes:register_types"\n'
        'reservations = "isopack_nodes:plan_reservations"\n'
    )
    return manifest


def core_registry() -> TypeRegistry:
    registry = TypeRegistry()
    register_core_types(registry)
    return registry


def test_isolated_output_meta_lands_in_parent_namespace(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = core_registry()
        worker = IsolatedWorker(
            write_manifest(tmp_path),
            registry,
            extra_env={"PYTHONPATH": str(TESTS_DIR)},
            device_map=CUDA1_MAP,
        )
        await worker.start()
        try:
            result = await worker.invoke(
                Invocation(
                    invocation_id="i1",
                    node_id="g",
                    node_type="iso.gpu_blob_out",
                    inputs={},
                    effective_schema=worker.schemas["iso.gpu_blob_out"],
                )
            )
            assert result.error is None
            assert result.outputs is not None
            blob = result.outputs["blob"]
            # The child said cuda:0 (its namespace); the parent sees cuda:1,
            # so admission lanes and budgets key on the physical device.
            assert blob.meta.get(RESOURCES_META_KEY) == {"gpu": "cuda:1"}
            assert blob.meta.get(COST_META_KEY) == {"vram:cuda:1": 128, "ram": 16}
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_isolated_lease_residency_is_mapped_before_the_governor(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        # The governor budgets only the *parent-namespace* device. A child
        # asking for vram:cuda:0 must be accounted on vram:cuda:1 - if the
        # mapping were skipped, cuda:0 is unbudgeted and reserved() would
        # never register on cuda:1.
        governor = MemoryGovernor({"vram:cuda:1": 100})
        registry = core_registry()
        worker = IsolatedWorker(
            write_manifest(tmp_path),
            registry,
            extra_env={"PYTHONPATH": str(TESTS_DIR)},
            reservations=GovernorReservationService(governor),
            device_map=CUDA1_MAP,
        )
        await worker.start()
        try:
            task = asyncio.create_task(
                worker.invoke(
                    Invocation(
                        invocation_id="i1",
                        node_id="hog",
                        node_type="iso.hog",
                        inputs={
                            "nbytes": registry.wrap(CORE_INT, 60),
                            "seconds": registry.wrap(CORE_FLOAT, 0.2),
                            "residency": registry.wrap(CORE_STRING, "vram:cuda:0"),
                        },
                        effective_schema=worker.schemas["iso.hog"],
                    )
                )
            )
            async with asyncio.timeout(5):
                while governor.reserved("vram:cuda:1") != 60:
                    await asyncio.sleep(0.005)
            assert governor.reserved("vram:cuda:0") == 0
            result = await task
            assert result.error is None
            async with asyncio.timeout(5):
                while governor.reserved("vram:cuda:1") != 0:
                    await asyncio.sleep(0.005)
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_isolated_lease_denial_respects_mapped_budget(tmp_path: Path) -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"vram:cuda:1": 100})
        registry = core_registry()
        worker = IsolatedWorker(
            write_manifest(tmp_path),
            registry,
            extra_env={"PYTHONPATH": str(TESTS_DIR)},
            reservations=GovernorReservationService(governor),
            device_map=CUDA1_MAP,
        )
        await worker.start()
        try:
            result = await worker.invoke(
                Invocation(
                    invocation_id="i1",
                    node_id="hog",
                    node_type="iso.hog",
                    inputs={
                        "nbytes": registry.wrap(CORE_INT, 200),
                        "seconds": registry.wrap(CORE_FLOAT, 0.0),
                        "residency": registry.wrap(CORE_STRING, "vram:cuda:0"),
                    },
                    effective_schema=worker.schemas["iso.hog"],
                )
            )
            assert result.error is not None
            assert "memory admission failed" in result.error.message
        finally:
            await worker.close()

    asyncio.run(scenario())
