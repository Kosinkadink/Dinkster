"""Torch-free proofs for the engine-side native model-tenant registry."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest
from dinkster_compat_comfy.native_residency import NativeResidencyCoordinator
from dinkster_compat_comfy.tenant_registry import (
    NativeModelTenantRegistry,
    TenantDefectiveError,
    TenantRegistryError,
)
from dinkster_memory import TenantRegistration


@dataclass(frozen=True)
class _Device:
    name: str
    type: str = "cuda"

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class _Memory:
    free_total: int
    free_torch: int = 0


@dataclass(frozen=True)
class _Policy:
    inference: int = 0

    def minimum_inference_memory(self) -> int:
        return self.inference


class _Manager:
    def __init__(self, free: int = 1_000, policy: _Policy | None = None) -> None:
        self.available = free
        self.policy = policy or _Policy()
        self._registry: list[Any] = []
        self.free_calls: list[tuple[int, object, tuple[object, ...]]] = []
        self.load_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.remove_calls: list[tuple[tuple[object, ...], bool]] = []
        self.empty_cache = lambda _device: None
        self.block_remove = False
        self.remove_started = threading.Event()
        self.remove_release = threading.Event()

    def free_memory(self, _device: object) -> _Memory:
        return _Memory(self.available)

    def current_policy(self) -> _Policy:
        return self.policy

    def policy_memory(self, device: object) -> _Memory:
        return self.free_memory(device)

    def free(
        self,
        memory_required: int,
        device: object,
        keep: tuple[Any, ...] = (),
        *,
        skip_demand_paged: bool = False,
    ) -> None:
        del skip_demand_paged
        self.free_calls.append((memory_required, device, keep))

    def remove(self, mechanisms: tuple[Any, ...], *, unload: bool = True) -> None:
        self.remove_calls.append((mechanisms, unload))
        if self.block_remove:
            self.remove_started.set()
            self.remove_release.wait()
        ids = {id(item) for item in mechanisms}
        self._registry[:] = [item for item in self._registry if id(item) not in ids]

    def registered(self) -> tuple[Any, ...]:
        return tuple(self._registry)

    def _touch(self, mechanism: object) -> None:
        self.remove((mechanism,), unload=False)
        self._registry.insert(0, mechanism)

    def load(self, mechanisms: tuple[Any, ...], **_kwargs: object) -> None:
        self.load_calls.append((tuple(mechanisms), dict(_kwargs)))
        for mechanism in mechanisms:
            self._touch(mechanism)


@dataclass
class _Handle:
    pack: str = "pack"
    model_id: str = "model"
    size_estimate_bytes: int = 100
    device_preference: str | None = None
    offloads: int = 0
    evictions: int = 0
    failure: BaseException | None = None
    started: threading.Event | None = None
    release: threading.Event | None = None
    cancelled: threading.Event | None = None
    registry: NativeModelTenantRegistry | None = None
    registration: Any | None = None

    async def offload(self) -> None:
        self.offloads += 1
        if self.started is not None:
            self.started.set()
        try:
            while self.release is not None and not self.release.is_set():
                await asyncio.sleep(0.001)
            if self.registry is not None:
                await self.registry.notify_resident(cast("TenantRegistration", self.registration))
            if self.failure is not None:
                raise self.failure
        except asyncio.CancelledError:
            if self.cancelled is not None:
                self.cancelled.set()
            raise

    async def evict(self) -> None:
        self.evictions += 1
        if self.failure is not None:
            raise self.failure


@dataclass
class _Clock:
    now: float = 0.0

    def __call__(self) -> float:
        return self.now


@dataclass
class _NativeMechanism:
    load_device: _Device = _Device("cuda:0")
    loaded: int = 1
    demand_paged: bool = False

    def loaded_bytes(self) -> int:
        return self.loaded

    def release_working_buffers(self) -> bool:
        return False

    def execution_context(self):  # noqa: ANN201
        return nullcontext()

    def unload(self) -> None:
        self.loaded = 0


@dataclass(eq=False)
class _NativeHandle:
    mechanisms: tuple[_NativeMechanism, ...]
    phases: list[str] = field(default_factory=list)

    def require_active(self) -> None:
        pass

    def run_lifecycle(self, phase: str) -> None:
        self.phases.append(phase)

    def reconcile_pool(self) -> None:
        pass


def _registry(
    *,
    manager: _Manager | None = None,
    clock: _Clock | None = None,
    callback_timeout: float = 1.0,
    placement_timeout: float = 10.0,
) -> tuple[NativeModelTenantRegistry, NativeResidencyCoordinator, _Manager]:
    resolved = manager or _Manager()
    coordinator = NativeResidencyCoordinator(cast("Any", resolved))
    registry = NativeModelTenantRegistry(
        cast("Any", coordinator),
        cast("Any", _Device("cuda:0")),
        total_memory=lambda _device: 1_000,
        device_exists=lambda preference, _device: preference in {"cpu", "cuda:0", "cuda:1"},
        callback_timeout=callback_timeout,
        placement_timeout=placement_timeout,
        clock=clock or _Clock(),
    )
    return registry, coordinator, resolved


def test_tenant_registry_import_is_torch_free() -> None:
    root = Path(__file__).parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "packages/dinkster-compat-comfy/src")}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dinkster_compat_comfy.tenant_registry; "
            "assert 'torch' not in sys.modules",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_coordinator_epoch_is_top_level_and_hook_runs_under_lock() -> None:
    manager = _Manager()
    coordinator = NativeResidencyCoordinator(cast("Any", manager))
    epochs: list[int] = []
    competing_lock = threading.Event()

    def hook(epoch: int) -> None:
        epochs.append(epoch)
        competing_lock.clear()

        def compete() -> None:
            with coordinator.locked():
                competing_lock.set()

        thread = threading.Thread(target=compete)
        thread.start()
        assert not competing_lock.wait(0.02)

    coordinator.add_pass_epoch_hook(hook)
    with coordinator.placement_pass() as first:
        with coordinator.placement_pass() as nested:
            assert nested == first
        assert not competing_lock.is_set()
    assert competing_lock.wait(1)
    with coordinator.placement_pass() as second:
        assert second == first + 1
    assert epochs == [first, second]


def test_stage_and_advisory_unload_each_advance_one_epoch() -> None:
    manager = _Manager()
    coordinator = NativeResidencyCoordinator(cast("Any", manager))
    epochs: list[int] = []
    coordinator.add_pass_epoch_hook(epochs.append)
    mechanism = _NativeMechanism()
    handle = _NativeHandle((mechanism,))

    with coordinator.stage(cast("Any", handle), cast("Any", handle.mechanisms)):
        pass
    mechanism.loaded = 1
    coordinator.advisory_unload(cast("Any", handle))

    assert epochs == [1, 2]
    assert handle.phases == ["pre-run", "inject", "eject", "unload"]


def test_nested_stage_keeps_outer_mechanism_in_placement_batch() -> None:
    manager = _Manager()
    coordinator = NativeResidencyCoordinator(cast("Any", manager))
    outer_mechanism = _NativeMechanism()
    inner_mechanism = _NativeMechanism()
    outer = _NativeHandle((outer_mechanism,))
    inner = _NativeHandle((inner_mechanism,))

    with coordinator.stage(
        cast("Any", outer),
        cast("Any", outer.mechanisms),
        memory_required=10,
        minimum_memory=20,
    ):
        with coordinator.stage(
            cast("Any", inner),
            cast("Any", inner.mechanisms),
            memory_required=30,
            minimum_memory=15,
        ):
            pass

    assert manager.load_calls == [
        ((outer_mechanism,), {"memory_required": 10, "minimum_memory": 20}),
        (
            (outer_mechanism, inner_mechanism),
            {"memory_required": 30, "minimum_memory": 20},
        ),
    ]


def test_register_validates_identity_capacity_device_and_frozen_size() -> None:
    async def scenario() -> None:
        registry, _coordinator, manager = _registry()
        handle = _Handle(device_preference="cuda:1")
        registration = await registry.register(handle)
        assert registration.assigned_device == "cuda:0"
        assert manager.registered()[0].loaded_bytes() == 0
        handle.size_estimate_bytes = 999
        assert manager.registered()[0].total_bytes() == 100
        with pytest.raises(TenantRegistryError, match="duplicate live"):
            await registry.register(handle)
        with pytest.raises(TenantRegistryError, match="does not exist"):
            await registry.register(_Handle(model_id="other", device_preference="cuda:9"))
        with pytest.raises(TenantRegistryError, match="governed device capacity"):
            await registry.register(_Handle(model_id="large", size_estimate_bytes=1_001))
        await registry.unregister(registration)
        assert manager.registered() == ()
        assert manager.remove_calls[-1][1] is False
        assert handle.evictions == 0

    asyncio.run(scenario())


def test_notify_resident_admits_touches_and_can_refuse() -> None:
    async def scenario() -> None:
        registry, _coordinator, manager = _registry()
        registration = await registry.register(_Handle())
        mechanism = manager.registered()[0]
        assert mechanism.loaded_bytes() == 0
        await registry.notify_resident(registration)
        assert mechanism.loaded_bytes() == 100
        manager.available = 50
        with pytest.raises(TenantRegistryError, match="budget refused"):
            await registry.notify_resident(registration)

    asyncio.run(scenario())


def test_mechanism_maps_offload_then_evict_and_prevents_phantom_bytes() -> None:
    async def scenario() -> None:
        registry, coordinator, manager = _registry()
        handle = _Handle()
        registration = await registry.register(handle)
        mechanism = manager.registered()[0]
        assert mechanism.loaded_bytes() == 0
        assert mechanism.partially_unload(1) == 0
        assert mechanism.loaded_bytes() == 0
        await registry.notify_resident(registration)
        with coordinator.placement_pass():
            assert mechanism.partially_unload(50) == 100
        assert handle.offloads == 1
        assert mechanism.loaded_bytes() == 0
        await registry.notify_resident(registration)
        with coordinator.placement_pass():
            mechanism.unload()
        assert handle.evictions == 1
        assert mechanism.loaded_bytes() == 0

    asyncio.run(scenario())


def test_callback_failure_poisoning_is_structured_and_non_reentrant() -> None:
    async def scenario() -> None:
        registry, coordinator, manager = _registry()
        handle = _Handle(failure=ValueError("broken"))
        registration = await registry.register(handle)
        await registry.notify_resident(registration)
        mechanism = manager.registered()[0]
        with coordinator.placement_pass():
            assert mechanism.partially_unload(1) == 0
        assert mechanism.loaded_bytes() == 0
        assert registry.defective_tenants[0].registration == registration
        assert "ValueError: broken" in registry.defective_tenants[0].reason
        with pytest.raises(TenantDefectiveError, match="DEFECTIVE"):
            await registry.notify_resident(registration)
        await registry.unregister(registration)
        assert registry.defective_tenants == ()

        reentrant, coordinator2, manager2 = _registry()
        reentrant_handle = _Handle(registry=reentrant)
        reentrant_registration = await reentrant.register(reentrant_handle)
        reentrant_handle.registration = reentrant_registration
        await reentrant.notify_resident(reentrant_registration)
        with coordinator2.placement_pass():
            assert manager2.registered()[0].partially_unload(1) == 0
        assert "non-reentrant" in reentrant.defective_tenants[0].reason

    asyncio.run(scenario())


def test_callback_timeout_cancels_and_unregister_serializes() -> None:
    async def scenario() -> None:
        registry, coordinator, manager = _registry(callback_timeout=0.02)
        started = threading.Event()
        release = threading.Event()
        cancelled = threading.Event()
        handle = _Handle(started=started, release=release, cancelled=cancelled)
        registration = await registry.register(handle)
        await registry.notify_resident(registration)
        mechanism = manager.registered()[0]
        with coordinator.placement_pass():
            assert mechanism.partially_unload(1) == 0
        assert started.wait(1)
        assert cancelled.wait(1)
        assert "timed out" in registry.defective_tenants[0].reason
        await registry.unregister(registration)

        registry2, coordinator2, manager2 = _registry(callback_timeout=1)
        started2 = threading.Event()
        release2 = threading.Event()
        handle2 = _Handle(started=started2, release=release2)
        registration2 = await registry2.register(handle2)
        await registry2.notify_resident(registration2)
        mechanism2 = manager2.registered()[0]

        callback = asyncio.create_task(asyncio.to_thread(lambda: mechanism2.partially_unload(1)))
        assert await asyncio.to_thread(started2.wait, 1)
        unregister = asyncio.create_task(registry2.unregister(registration2))
        await asyncio.sleep(0.01)
        assert not unregister.done()
        release2.set()
        assert await callback == 100
        await unregister
        assert manager2.registered() == ()

    asyncio.run(scenario())


def test_unregister_takes_coordinator_before_callback_lock() -> None:
    async def scenario() -> None:
        registry, coordinator, manager = _registry()
        registration = await registry.register(_Handle())
        await registry.notify_resident(registration)
        manager.block_remove = True

        unregister = asyncio.create_task(registry.unregister(registration))
        assert await asyncio.to_thread(manager.remove_started.wait, 1)

        placement_entered = threading.Event()

        def placement() -> None:
            with coordinator.placement_pass():
                placement_entered.set()

        placement_task = asyncio.create_task(asyncio.to_thread(placement))
        await asyncio.sleep(0.01)
        assert not placement_entered.is_set()
        manager.remove_release.set()
        await unregister
        await placement_task
        assert registry.defective_tenants == ()

    asyncio.run(scenario())


def test_cancelled_register_and_notify_roll_back_hidden_residency() -> None:
    async def cancel_behind_lock(
        coordinator: NativeResidencyCoordinator,
        operation: Any,
    ) -> None:
        entered = threading.Event()
        release = threading.Event()

        def hold() -> None:
            with coordinator.locked():
                entered.set()
                release.wait()

        holder = threading.Thread(target=hold)
        holder.start()
        assert await asyncio.to_thread(entered.wait, 1)
        task = asyncio.create_task(operation)
        await asyncio.sleep(0.01)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        holder.join()

    async def scenario() -> None:
        registry, coordinator, manager = _registry()
        await cancel_behind_lock(coordinator, registry.register(_Handle()))
        assert manager.registered() == ()
        registration = await registry.register(_Handle())

        await cancel_behind_lock(
            coordinator,
            registry.notify_resident(registration),
        )
        assert manager.registered() == ()
        await registry.notify_resident(registration)
        assert manager.registered()[0].loaded_bytes() == 100

    asyncio.run(scenario())


def test_pass_budget_skip_detaches_then_reenrolls_and_resets() -> None:
    async def scenario() -> None:
        clock = _Clock()
        registry, coordinator, manager = _registry(clock=clock, placement_timeout=5)
        handle = _Handle()
        registration = await registry.register(handle)
        await registry.notify_resident(registration)
        mechanism = manager.registered()[0]

        with coordinator.placement_pass():
            clock.now += 6
            assert mechanism.partially_unload(1) == 0
            mechanism.unload()
            manager.remove((mechanism,), unload=False)
        assert handle.offloads == handle.evictions == 0
        assert mechanism not in manager.registered()

        with coordinator.placement_pass():
            assert mechanism in manager.registered()
            assert mechanism.partially_unload(1) == 100
        assert handle.offloads == 1

        manager.remove((mechanism,), unload=False)
        with coordinator.placement_pass():
            assert mechanism not in manager.registered()

    asyncio.run(scenario())


def test_defective_and_shed_tenants_are_not_pass_reenrolled() -> None:
    async def scenario() -> None:
        registry, coordinator, manager = _registry()
        failed = _Handle(failure=RuntimeError("fail"))
        failed_registration = await registry.register(failed)
        await registry.notify_resident(failed_registration)
        failed_mechanism = manager.registered()[0]
        with coordinator.placement_pass():
            failed_mechanism.unload()
            manager.remove((failed_mechanism,), unload=False)
        with coordinator.placement_pass():
            assert failed_mechanism not in manager.registered()

        shed = _Handle(model_id="shed")
        shed_registration = await registry.register(shed)
        await registry.notify_resident(shed_registration)
        shed_mechanism = manager.registered()[0]
        with coordinator.placement_pass():
            assert shed_mechanism.partially_unload(1) == 100
            manager.remove((shed_mechanism,), unload=False)
        with coordinator.placement_pass():
            assert shed_mechanism not in manager.registered()
        await registry.notify_resident(shed_registration)
        assert shed_mechanism in manager.registered()

    asyncio.run(scenario())
