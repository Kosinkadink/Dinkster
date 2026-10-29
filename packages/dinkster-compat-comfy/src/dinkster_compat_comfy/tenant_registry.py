"""Engine-side model-tenant residency over the native manager fleet.

Imports stay torch-free so the root environment can inspect the compat pack.
Device probing and total-memory lookup are resolved lazily, or injected by
tests and hosts.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import AbstractContextManager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Protocol, cast
from uuid import uuid4

from dinkster_memory import ModelTenantHandle, TenantRegistration

logger = logging.getLogger(__name__)


class _Device(Protocol):
    type: str


class _DeviceMemory(Protocol):
    free_total: int


class _MemoryPolicy(Protocol):
    def minimum_inference_memory(self) -> int: ...


class _ResidencyMechanism(Protocol):
    @property
    def load_device(self) -> _Device: ...

    @property
    def demand_paged(self) -> bool: ...

    def total_bytes(self) -> int: ...

    def loaded_bytes(self) -> int: ...

    def automatically_reclaimable_bytes(self) -> int: ...

    def offloaded_bytes(self) -> int: ...

    def partially_load(self, extra_memory: int | None) -> int: ...

    def partially_unload(self, memory_to_free: int) -> int: ...

    def unload(self) -> None: ...


class _ResidencyManager(Protocol):
    def current_policy(self) -> _MemoryPolicy: ...

    def policy_memory(self, device: _Device) -> _DeviceMemory: ...

    def free(
        self,
        memory_required: int,
        device: _Device,
        keep: Sequence[_ResidencyMechanism] = (),
        *,
        skip_demand_paged: bool = False,
    ) -> None: ...

    def remove(
        self,
        mechanisms: Sequence[_ResidencyMechanism],
        *,
        unload: bool = True,
    ) -> None: ...

    def registered(self) -> tuple[object, ...]: ...

    def _touch(self, mechanism: _ResidencyMechanism) -> None: ...


class _Coordinator(Protocol):
    manager: _ResidencyManager

    def add_pass_epoch_hook(self, hook: Callable[[int], None]) -> None: ...

    def locked(self) -> AbstractContextManager[None]: ...

    def placement_pass(self) -> AbstractContextManager[int]: ...


class TenantRegistryError(RuntimeError):
    """A model-tenant operation violated residency or admission policy."""


class TenantDefectiveError(TenantRegistryError):
    """A poisoned tenant refused operations until unregister."""


@dataclass(frozen=True)
class DefectiveTenant:
    """Structured health record for one fail-poisoned tenant."""

    registration: TenantRegistration
    pack: str
    model_id: str
    reason: str


@dataclass
class _TenantState:
    handle: ModelTenantHandle
    registration: TenantRegistration
    size: int
    callback_lock: asyncio.Lock
    status: str = "nonresident"
    skipped: bool = False
    revision: int = 0
    mechanism: _TenantMechanism = field(init=False)


_active_callback_registry: ContextVar[NativeModelTenantRegistry | None] = ContextVar(
    "dinkster_active_tenant_callback_registry", default=None
)


class _TenantMechanism:
    """ResidencyMechanism adapter whose allocations remain pack-owned."""

    def __init__(
        self,
        registry: NativeModelTenantRegistry,
        state: _TenantState,
        load_device: _Device,
    ) -> None:
        self._registry = registry
        self._state = state
        self._load_device = load_device

    @property
    def load_device(self) -> _Device:
        return self._load_device

    @property
    def demand_paged(self) -> bool:
        return False

    def total_bytes(self) -> int:
        return self._state.size

    def loaded_bytes(self) -> int:
        return self._state.size if self._state.status == "resident" else 0

    def automatically_reclaimable_bytes(self) -> int:
        return 0

    def offloaded_bytes(self) -> int:
        return self.total_bytes() - self.loaded_bytes()

    def partially_load(self, extra_memory: int | None) -> int:
        del extra_memory
        return 0

    def partially_unload(self, memory_to_free: int) -> int:
        del memory_to_free
        if self._state.status != "resident":
            return 0
        if self._registry._run_callback(  # pyright: ignore[reportPrivateUsage]
            self._state, "offload"
        ):
            return self._state.size
        return 0

    def unload(self) -> None:
        self._registry._run_callback(  # pyright: ignore[reportPrivateUsage]
            self._state, "evict"
        )


def _default_total_memory(device: _Device) -> int:
    memory = importlib.import_module("dinkster_inference_torch.memory")
    return int(memory.get_total_memory(device))


def _default_device_exists(preference: str, assigned: _Device) -> bool:
    try:
        torch = importlib.import_module("torch")
        device = torch.device(preference)
    except (ImportError, RuntimeError, TypeError, ValueError):
        return preference == str(assigned)
    if device.type == "cpu":
        return True
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    return 0 <= int(index) < int(torch.cuda.device_count())


class NativeModelTenantRegistry:
    """ModelTenantRegistry sharing a NativeResidencyCoordinator fleet."""

    def __init__(
        self,
        coordinator: _Coordinator,
        load_device: _Device,
        *,
        total_memory: Callable[[_Device], int] = _default_total_memory,
        device_exists: Callable[[str, _Device], bool] = _default_device_exists,
        callback_timeout: float = 15.0,
        placement_timeout: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if callback_timeout <= 0 or placement_timeout <= 0:
            raise ValueError("tenant callback and placement timeouts must be positive")
        self._coordinator = coordinator
        self._load_device = load_device
        self._total_memory = total_memory
        self._device_exists = device_exists
        self._callback_timeout = callback_timeout
        self._placement_timeout = placement_timeout
        self._clock = clock
        self._by_key: dict[tuple[str, str], _TenantState] = {}
        self._by_registration: dict[str, _TenantState] = {}
        self._defective: dict[str, DefectiveTenant] = {}
        self._pass_epoch = 0
        self._pass_started = clock()

        self._callback_loop = asyncio.new_event_loop()
        self._callback_ready = threading.Event()
        self._callback_thread = threading.Thread(
            target=self._run_callback_loop,
            name="dinkster-tenant-callbacks",
            daemon=True,
        )
        self._callback_thread.start()
        self._callback_ready.wait()
        coordinator.add_pass_epoch_hook(self._begin_pass)

    @property
    def defective_tenants(self) -> tuple[DefectiveTenant, ...]:
        with self._coordinator.locked():
            return tuple(self._defective.values())

    def _run_callback_loop(self) -> None:
        asyncio.set_event_loop(self._callback_loop)
        self._callback_ready.set()
        self._callback_loop.run_forever()

    def _begin_pass(self, epoch: int) -> None:
        """Reset D2 and repair prior skip detaches under coordinator lock."""
        self._pass_epoch = epoch
        self._pass_started = self._clock()
        self._repair_registry()

    def _repair_registry(self) -> None:
        """Repair skipped residents without creating a residency channel."""
        manager = self._coordinator.manager
        registered = {id(item) for item in manager.registered()}
        for state in tuple(self._by_registration.values()):
            mechanism = state.mechanism
            if state.skipped and state.status == "resident":
                if id(mechanism) not in registered:
                    manager._touch(mechanism)  # pyright: ignore[reportPrivateUsage]
                state.skipped = False
            elif state.status != "resident" and id(mechanism) in registered:
                manager.remove((mechanism,), unload=False)

    def _require_not_reentrant(self) -> None:
        if _active_callback_registry.get() is self:
            raise TenantRegistryError("tenant callbacks are non-reentrant")

    @staticmethod
    def _validate_handle(handle: ModelTenantHandle) -> tuple[str, str, int, str | None]:
        pack = cast("object", handle.pack)
        model_id = cast("object", handle.model_id)
        size = cast("object", handle.size_estimate_bytes)
        preference = cast("object", handle.device_preference)
        if not isinstance(pack, str) or not pack:
            raise TenantRegistryError("tenant pack must be a non-empty string")
        if not isinstance(model_id, str) or not model_id:
            raise TenantRegistryError("tenant model_id must be a non-empty string")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise TenantRegistryError("tenant size_estimate_bytes must be a positive integer")
        if preference is not None and (not isinstance(preference, str) or not preference):
            raise TenantRegistryError("tenant device_preference must be a non-empty string")
        if not callable(handle.offload) or not callable(handle.evict):
            raise TenantRegistryError("tenant offload and evict callbacks must be callable")
        return pack, model_id, size, preference

    def _reserve_bytes(self) -> int:
        return self._coordinator.manager.current_policy().minimum_inference_memory()

    def _make_room(self, state: _TenantState) -> None:
        manager = self._coordinator.manager
        required = state.size + self._reserve_bytes()
        manager.free(required, self._load_device, keep=(state.mechanism,))
        if manager.policy_memory(self._load_device).free_total < required:
            raise TenantRegistryError(
                f"tenant budget refused for {state.handle.pack!r}/{state.handle.model_id!r}"
            )

    async def register(self, handle: ModelTenantHandle) -> TenantRegistration:
        self._require_not_reentrant()
        pack, model_id, size, preference = self._validate_handle(handle)
        if preference is not None and not self._device_exists(preference, self._load_device):
            raise TenantRegistryError(f"tenant device preference {preference!r} does not exist")
        registration = TenantRegistration(str(uuid4()), str(self._load_device))

        def register_sync() -> _TenantState:
            with self._coordinator.placement_pass():
                key = (pack, model_id)
                if key in self._by_key:
                    raise TenantRegistryError(
                        f"duplicate live tenant identity {pack!r}/{model_id!r}"
                    )
                reserve = self._reserve_bytes()
                capacity = self._total_memory(self._load_device) - reserve
                if size > capacity:
                    raise TenantRegistryError(
                        f"tenant size {size} exceeds governed device capacity {capacity}"
                    )
                state = _TenantState(
                    handle=handle,
                    registration=registration,
                    size=size,
                    callback_lock=asyncio.Lock(),
                )
                state.mechanism = _TenantMechanism(self, state, self._load_device)
                self._make_room(state)
                self._coordinator.manager._touch(  # pyright: ignore[reportPrivateUsage]
                    state.mechanism
                )
                self._by_key[key] = state
                self._by_registration[registration.registration_id] = state
                return state

        worker = asyncio.create_task(asyncio.to_thread(register_sync))
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            state: _TenantState | None = None
            try:
                state = await asyncio.shield(worker)
            except BaseException:
                pass
            if state is not None:

                def rollback_sync() -> None:
                    with self._coordinator.locked():
                        if self._by_registration.get(registration.registration_id) is state:
                            self._coordinator.manager.remove((state.mechanism,), unload=False)
                            self._by_registration.pop(registration.registration_id)
                            self._by_key.pop((state.handle.pack, state.handle.model_id))
                            state.status = "unregistered"
                            state.revision += 1

                rollback = asyncio.create_task(asyncio.to_thread(rollback_sync))
                await asyncio.shield(rollback)
            raise
        return registration

    def _state_for(self, registration: TenantRegistration) -> _TenantState:
        state = self._by_registration.get(registration.registration_id)
        if state is None or state.registration != registration:
            raise TenantRegistryError(
                f"unknown tenant registration {registration.registration_id!r}"
            )
        return state

    async def notify_resident(self, registration: TenantRegistration) -> None:
        self._require_not_reentrant()

        def notify_sync() -> tuple[_TenantState, str, bool, int]:
            with self._coordinator.placement_pass():
                state = self._state_for(registration)
                if state.status == "defective":
                    raise TenantDefectiveError(
                        f"tenant {registration.registration_id!r} is DEFECTIVE"
                    )
                previous_status = state.status
                previous_skipped = state.skipped
                self._make_room(state)
                state.status = "resident"
                state.skipped = False
                state.revision += 1
                self._coordinator.manager._touch(  # pyright: ignore[reportPrivateUsage]
                    state.mechanism
                )
                return state, previous_status, previous_skipped, state.revision

        worker = asyncio.create_task(asyncio.to_thread(notify_sync))
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            result: tuple[_TenantState, str, bool, int] | None = None
            try:
                result = await asyncio.shield(worker)
            except BaseException:
                pass
            if result is not None:
                state, previous_status, previous_skipped, revision = result

                def rollback_sync() -> None:
                    with self._coordinator.locked():
                        if (
                            self._by_registration.get(registration.registration_id) is state
                            and state.revision == revision
                            and state.status == "resident"
                        ):
                            state.status = previous_status
                            state.skipped = previous_skipped
                            state.revision += 1
                            if previous_status != "resident":
                                self._coordinator.manager.remove((state.mechanism,), unload=False)

                rollback = asyncio.create_task(asyncio.to_thread(rollback_sync))
                await asyncio.shield(rollback)
            raise

    async def _hold_callback_lock(
        self,
        state: _TenantState,
        acquired: Future[None],
        release: threading.Event,
    ) -> None:
        await state.callback_lock.acquire()
        try:
            if not acquired.done():
                acquired.set_result(None)
            while not release.is_set():
                await asyncio.sleep(0.001)
        finally:
            state.callback_lock.release()

    async def unregister(self, registration: TenantRegistration) -> None:
        self._require_not_reentrant()

        def unregister_sync() -> None:
            with self._coordinator.locked():
                self._repair_registry()
                current = self._state_for(registration)
                acquired: Future[None] = Future()
                release = threading.Event()
                hold = asyncio.run_coroutine_threadsafe(
                    self._hold_callback_lock(current, acquired, release),
                    self._callback_loop,
                )
                acquired.result()
                try:
                    self._coordinator.manager.remove((current.mechanism,), unload=False)
                    self._by_registration.pop(registration.registration_id)
                    self._by_key.pop((current.handle.pack, current.handle.model_id))
                    self._defective.pop(registration.registration_id, None)
                    current.status = "unregistered"
                    current.revision += 1
                finally:
                    release.set()
                    hold.result()

        worker = asyncio.create_task(asyncio.to_thread(unregister_sync))
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            try:
                await asyncio.shield(worker)
            except BaseException:
                pass
            raise

    async def terminal_release(self, registration: TenantRegistration) -> None:
        """Evict one tenant through its callback, then retire its registration."""
        self._require_not_reentrant()
        state = self._state_for(registration)
        await self._invoke_callback(state, "evict")
        await self.unregister(registration)

    async def _invoke_callback(
        self,
        state: _TenantState,
        callback_name: str,
    ) -> None:
        async with state.callback_lock:
            token = _active_callback_registry.set(self)
            try:
                callback = (
                    state.handle.offload if callback_name == "offload" else state.handle.evict
                )
                await callback()
            finally:
                _active_callback_registry.reset(token)

    def _poison(self, state: _TenantState, reason: str) -> None:
        state.status = "defective"
        state.skipped = False
        state.revision += 1
        record = DefectiveTenant(
            registration=state.registration,
            pack=state.handle.pack,
            model_id=state.handle.model_id,
            reason=reason,
        )
        self._defective[state.registration.registration_id] = record
        logger.error(
            "model tenant %s/%s (%s) is DEFECTIVE: %s",
            state.handle.pack,
            state.handle.model_id,
            state.registration.registration_id,
            reason,
        )

    def _run_callback(self, state: _TenantState, callback_name: str) -> bool:
        if state.status == "defective":
            return False
        remaining = self._placement_timeout - (self._clock() - self._pass_started)
        if remaining <= 0:
            state.skipped = state.status == "resident"
            return False
        timeout = min(self._callback_timeout, remaining)
        future = asyncio.run_coroutine_threadsafe(
            self._invoke_callback(state, callback_name), self._callback_loop
        )
        try:
            future.result(timeout=timeout)
        except FutureTimeoutError:
            future.cancel()
            self._poison(state, f"{callback_name} callback timed out after {timeout:g}s")
            return False
        except BaseException as exc:
            self._poison(
                state,
                f"{callback_name} callback failed: {type(exc).__name__}: {exc}",
            )
            return False
        state.status = "nonresident"
        state.skipped = False
        state.revision += 1
        return True


__all__ = [
    "DefectiveTenant",
    "NativeModelTenantRegistry",
    "TenantDefectiveError",
    "TenantRegistryError",
]
