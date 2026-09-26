"""Native runtime residency owned by the Comfy compatibility worker.

The memory governor owns coarse process admission outside the worker. This
module owns per-unit placement inside it: each assembled component enrolls in
the inference stack's weight-store residency mechanism, and one process-local
coordinator serializes placement with the complete forward that consumes it.
Load-device selection and admission accept CPU, CUDA (NVIDIA, and AMD
through ROCm torch builds), MPS, and XPU devices.

Imports stay torch-free. The compat pack's schemas are inspected by the root
environment, which deliberately has no torch; torch and the inference manager
are resolved only when a native runtime is constructed in its worker.
"""

from __future__ import annotations

import copy
import importlib
import os
import threading
import weakref
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Protocol, cast

from dinkster_inference import (
    AttachmentDeclaration,
    DependencyPlan,
    ExecutionObserver,
    ExecutionObserverAttachment,
    ExecutionSpan,
    ExecutionSpanEvent,
    ExecutionStage,
    InferenceRuntimeHandle,
    PatchOverlay,
    ReconstructionRecipe,
)
from dinkster_protocol import (
    AttentionPolicyConfig,
    derive_attention_route_token,
    validate_attention_policy,
)
from dinkster_workers import KNOWN_ACCELERATORS, AcceleratorError, current_execution_context
from dinkster_workers.accelerator import ACCELERATOR_ENV

from .memory_policy import native_memory_policy

NativeStageEvent = ExecutionSpanEvent
NativeStageObserver = ExecutionObserver


@dataclass(frozen=True)
class ResidencyRouteFacts:
    """How a native runtime handle's components were enrolled for residency.

    A diagnostic record only: it never enters runtime identity or the
    materialization key, so the same recipe deduplicates identically
    whichever mechanism serves it.

    ``requested`` is the arm value the worker was launched with, ``mechanism``
    the residency mechanism the platform gates actually selected ("aimdo"
    or "eager"), and ``fallback_reason`` the failed capability gate when a
    requested mechanism could not be selected. A mixed handle
    reports its dynamic mechanism only when at least one component uses it.
    The component tuples record, in enrollment order, which components
    enrolled on the dynamic mechanism, which stayed eager because the
    runtime's policy declares them resident, and which fell back to eager
    at capability admission. Selected-Aimdo construction errors propagate
    without publishing a successful handle. Aimdo's temporary
    per-operation weight materialization remains Aimdo, not component fallback.
    """

    requested: str
    mechanism: str
    fallback_reason: str | None = None
    dynamic_components: tuple[str, ...] = ()
    resident_components: tuple[str, ...] = ()
    fallback_components: tuple[str, ...] = ()


_stage_observer: ContextVar[ExecutionObserverAttachment | None] = ContextVar(
    "dinkster_native_stage_observer_attachment", default=None
)


@contextmanager
def observe_native_stages(
    observer: NativeStageObserver, *, invocation_id: str | None = None
) -> Generator[ExecutionObserverAttachment, None, None]:
    """Install an invocation-local observer around native residency stages."""

    attachment = ExecutionObserverAttachment(observer, invocation_id)
    token = _stage_observer.set(attachment)
    try:
        yield attachment
    finally:
        _stage_observer.reset(token)


def current_native_observer() -> ExecutionObserverAttachment | None:
    return _stage_observer.get()


@contextmanager
def native_execution_span(
    stage: ExecutionStage,
    operation: str,
    *,
    parent_span_id: int | None = None,
    component_role: str | None = None,
    device: str | None = None,
) -> Generator[ExecutionSpan | None, None, None]:
    attachment = _stage_observer.get()
    if attachment is None:
        yield None
        return
    span = attachment.begin(
        stage,
        operation,
        parent_span_id=parent_span_id,
        component_role=component_role,
        device=device,
    )
    try:
        yield span
    finally:
        attachment.end(span)


class _PoolPlacement(Protocol):
    def set_loaded(self, obj: object, loaded: bool) -> None: ...


class _ResidencyMechanism(Protocol):
    @property
    def load_device(self) -> Any: ...

    @property
    def demand_paged(self) -> bool: ...

    def total_bytes(self) -> int: ...

    def loaded_bytes(self) -> int: ...

    def automatically_reclaimable_bytes(self) -> int: ...

    def offloaded_bytes(self) -> int: ...

    def partially_load(self, extra_memory: int | None) -> int: ...

    def partially_unload(self, memory_to_free: int) -> int: ...

    def unload(self) -> None: ...

    def retain_offload_storage(self) -> None: ...

    def execution_context(self) -> AbstractContextManager[None]: ...

    def working_set_reservation_bytes(self) -> int: ...

    def reserve_working_set(self) -> AbstractContextManager[None]: ...

    def release_working_buffers(self) -> bool: ...


def _cuda_namespace_available(torch: Any) -> bool:
    cuda = getattr(torch, "cuda", None)
    return (
        cuda is not None
        and callable(getattr(cuda, "is_available", None))
        and bool(cuda.is_available())
    )


def _xpu_available(torch: Any) -> bool:
    xpu = getattr(torch, "xpu", None)
    return (
        xpu is not None
        and callable(getattr(xpu, "is_available", None))
        and bool(xpu.is_available())
    )


def _mps_available(torch: Any) -> bool:
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    return (
        mps is not None
        and callable(getattr(mps, "is_available", None))
        and bool(mps.is_available())
    )


def _hip_version(torch: Any) -> str | None:
    hip = getattr(getattr(torch, "version", None), "hip", None)
    return str(hip) if hip else None


def select_load_device(
    torch: Any,
    *,
    selection: str = "auto",
    environ: Mapping[str, str] | None = None,
) -> Any:
    """The device weights load to.

    ``auto`` probes the torch build's own capability, in order: the CUDA
    namespace (NVIDIA, and AMD on ROCm builds, which deliberately expose
    HIP through ``torch.cuda``), then ``torch.xpu`` (Intel), then the MPS
    backend, then cpu. Host driver detection never decides here - only
    what this process's torch can actually use.

    An explicit selection - the ``selection`` argument, or
    ``$DINKSTER_ACCELERATOR`` when the argument is ``auto`` - is
    authoritative and refuses loudly when its backend is unavailable:
    silently computing on the cpu fallback when the user pinned an
    accelerator is the failure mode this guards against. ``rocm`` and
    ``cuda`` both map to ``cuda:0`` but are distinct families:
    ``torch.version.hip`` is the backend identity, so each refuses the
    other's build (see :func:`native_accelerator_family`).
    """
    env: Mapping[str, str] = os.environ if environ is None else environ
    if selection == "auto":
        selection = env.get(ACCELERATOR_ENV, "").strip() or "auto"
    if selection == "auto":
        if _cuda_namespace_available(torch):
            return torch.device("cuda:0")
        if _xpu_available(torch):
            return torch.device("xpu:0")
        if _mps_available(torch):
            return torch.device("mps")
        return torch.device("cpu")
    if selection not in KNOWN_ACCELERATORS:
        raise AcceleratorError(
            f"unknown accelerator {selection!r}; expected auto or one of "
            f"{', '.join(KNOWN_ACCELERATORS)}"
        )
    if selection == "cpu":
        return torch.device("cpu")
    if selection in ("cuda", "rocm"):
        if not _cuda_namespace_available(torch):
            raise ValueError(
                f"accelerator {selection!r} was selected but this torch build "
                "reports no available CUDA-namespace device"
            )
        hip = _hip_version(torch)
        if selection == "cuda" and hip is not None:
            raise ValueError(
                "accelerator 'cuda' was selected but this torch build is ROCm "
                f"(torch.version.hip={hip!r}); select 'rocm' instead"
            )
        if selection == "rocm" and hip is None:
            raise ValueError(
                "accelerator 'rocm' was selected but this torch build has no "
                "HIP runtime (torch.version.hip is unset); select 'cuda' for "
                "NVIDIA builds"
            )
        return torch.device("cuda:0")
    if selection == "xpu":
        if not _xpu_available(torch):
            raise ValueError(
                "accelerator 'xpu' was selected but this torch build reports "
                "no available XPU device"
            )
        return torch.device("xpu:0")
    if not _mps_available(torch):
        raise ValueError(
            "accelerator 'mps' was selected but this torch build reports no available MPS backend"
        )
    return torch.device("mps")


def select_intermediate_device(
    torch: Any,
    *,
    selection: str = "cpu",
    environ: Mapping[str, str] | None = None,
) -> Any:
    """Select where tensors exchanged between nodes are stored.

    CPU is the default, matching ComfyUI without ``--gpu-only``. Passing
    ``selection="auto"`` or an explicit accelerator opts into accelerator
    placement and uses the same validated policy as model loading.
    """
    if selection == "cpu":
        return torch.device("cpu")
    return select_load_device(torch, selection=selection, environ=environ)


def select_current_device(
    torch: Any,
    *,
    selection: str = "auto",
    environ: Mapping[str, str] | None = None,
) -> Any:
    """Select the active device in the process's accelerator namespace."""
    device = select_load_device(torch, selection=selection, environ=environ)
    if device.type == "cuda":
        return torch.device("cuda", torch.cuda.current_device())
    if device.type == "xpu":
        return torch.device("xpu", torch.xpu.current_device())
    return device


def native_accelerator_family(torch: Any, device: Any) -> str:
    """The accelerator family behind a torch device. ``device.type``
    alone cannot distinguish ROCm from CUDA - ROCm torch builds expose
    AMD GPUs as ``cuda`` devices - so the CUDA device type is split on
    ``torch.version.hip``."""
    kind = str(getattr(device, "type", None) or device).split(":")[0]
    if kind == "cuda":
        return "rocm" if _hip_version(torch) is not None else "cuda"
    return kind


def _mechanism_device(mechanisms: Sequence[_ResidencyMechanism]) -> str | None:
    return None if not mechanisms else str(mechanisms[0].load_device)


class _EnrollAssembled(Protocol):
    def __call__(
        self,
        assembled: Any,
        *,
        load_device: object,
        offload_device: object,
        patch_sets: Mapping[str, object] | None = ...,
        patch_weight_dtype: object | None = ...,
        patch_key_prefixes: Mapping[str, str] | None = ...,
        storage_dtypes: Mapping[str, object] | None = ...,
        mechanism_factory: Any = ...,
        memory_policy: Any | None = ...,
        mps_snapshot: Callable[[Any], Any] | None = ...,
    ) -> Mapping[str, _ResidencyMechanism]: ...


class _ResidencyManager(Protocol):
    empty_cache: Any
    mps_snapshot: Callable[[Any], Any]

    def current_policy(self) -> Any: ...

    def policy_memory(self, device: Any, *, policy: Any | None = None) -> Any: ...

    def load(
        self,
        mechanisms: Sequence[_ResidencyMechanism],
        *,
        memory_required: int = 0,
        minimum_memory: int | None = None,
        force_full_load: bool = False,
    ) -> None: ...

    def free(
        self,
        memory_required: int,
        device: Any,
        keep: Sequence[_ResidencyMechanism] = (),
        *,
        skip_demand_paged: bool = False,
        _policy: Any | None = None,
    ) -> None: ...

    def remove(
        self,
        mechanisms: Sequence[_ResidencyMechanism],
        *,
        unload: bool = True,
        discard: bool = False,
    ) -> None: ...

    def registered(self) -> tuple[_ResidencyMechanism, ...]: ...

    def reserve_working_sets(
        self, mechanisms: Sequence[_ResidencyMechanism]
    ) -> AbstractContextManager[None]: ...


PassEpochHook = Callable[[int], None]


class NativeResidencyBusyError(RuntimeError):
    """Advisory unloading refused because a native stage is active."""


class NativeRuntimeReleasedError(RuntimeError):
    """A terminally released native checkpoint handle was used again."""


class NativeRebuildUnavailableError(RuntimeError):
    """A handle has no worker-local recipe materializer bound."""


class AttachmentCloneError(RuntimeError):
    """A declared attachment cannot satisfy clone/rebuild semantics."""

    def __init__(self, attachment: str, reason: str) -> None:
        self.attachment = attachment
        self.reason = reason
        super().__init__(f"attachment {attachment!r} cannot be cloned/rebuilt: {reason}")


@dataclass(frozen=True)
class NativeLifecycleEvent:
    """Worker-local callback context; never part of a recipe or RPC."""

    phase: str
    handle: NativeRuntimeHandle
    source: NativeRuntimeHandle | None = None


LifecycleCallback = Callable[[NativeLifecycleEvent], None]
LIFECYCLE_PHASES = (
    "clone",
    "load",
    "unload",
    "pre-run",
    "inject",
    "eject",
    "cleanup",
)


class NativeLifecycleCallbackError(RuntimeError):
    """Every callback ran, but one or more failed."""

    def __init__(self, phase: str, failures: tuple[BaseException, ...]) -> None:
        self.phase = phase
        self.failures = failures
        details = "; ".join(f"{type(failure).__name__}: {failure}" for failure in failures)
        super().__init__(f"native lifecycle phase {phase!r} failed: {details}")


HandleMaterializer = Callable[[ReconstructionRecipe, Mapping[str, object]], "NativeRuntimeHandle"]
ComponentHandleMaterializer = Callable[
    [ReconstructionRecipe, Mapping[str, object]], "NativeComponentHandle"
]
DependencyPath = tuple[str, ...]
ConditionalDependencyMaterializer = Callable[
    [frozenset[DependencyPath]], Mapping[DependencyPath, "NativeRuntimeHandle"]
]


@dataclass(frozen=True)
class NativeResidencyToken:
    """One unique lifetime claim in the process-local dependency ledger."""

    serial: int
    materialization_key: str
    account_identity: str
    estimated_cost: int


class NativeResidencyPool:
    """Exact lifetime and aggregate-cost ledger for native dependency handles."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_serial = 1
        self._active: dict[int, NativeResidencyToken] = {}

    def acquire(
        self,
        materialization_key: str,
        account_identity: str,
        estimated_cost: int,
    ) -> NativeResidencyToken:
        if not materialization_key:
            raise ValueError("native materialization key must be non-empty")
        if not account_identity:
            raise ValueError("native residency account identity must be non-empty")
        if type(estimated_cost) is not int or estimated_cost < 0:
            raise ValueError("native residency estimated cost must be non-negative")
        with self._lock:
            token = NativeResidencyToken(
                self._next_serial,
                materialization_key,
                account_identity,
                estimated_cost,
            )
            self._next_serial += 1
            self._active[token.serial] = token
            return token

    def release(self, token: NativeResidencyToken) -> None:
        with self._lock:
            if self._active.get(token.serial) is not token:
                raise RuntimeError("native residency token was not active")
            del self._active[token.serial]

    def active_tokens(self) -> tuple[NativeResidencyToken, ...]:
        with self._lock:
            return tuple(self._active.values())

    def account_costs(self) -> Mapping[str, int]:
        totals: dict[str, int] = {}
        with self._lock:
            for token in self._active.values():
                totals[token.account_identity] = (
                    totals.get(token.account_identity, 0) + token.estimated_cost
                )
        return totals


class NativeResidencyCoordinator:
    """One manager and exclusion boundary for all native runtime stages.

    The RLock is held across manager placement, the caller's full forward,
    and pool-accounting reconciliation. Advisory pressure uses a non-blocking
    acquisition so the worker event loop never waits behind sampling running
    in ``asyncio.to_thread``.
    """

    def __init__(
        self,
        manager: _ResidencyManager | None = None,
        *,
        free_memory: Any | None = None,
    ) -> None:
        if manager is not None and free_memory is not None:
            raise ValueError("manager and free_memory are mutually exclusive")
        if manager is None:
            residency = importlib.import_module("dinkster_inference_torch.residency")
            kwargs = {"policy_provider": native_memory_policy}
            if free_memory is None:
                manager = cast("_ResidencyManager", residency.ResidencyManager(**kwargs))
            else:
                kwargs["free_memory"] = free_memory
                manager = cast(
                    "_ResidencyManager",
                    residency.ResidencyManager(**kwargs),
                )
        self.manager = manager
        self._lock = threading.RLock()
        self._handles: weakref.WeakSet[NativeRuntimeHandle] = weakref.WeakSet()
        self._placement_depth = 0
        self._placement_epoch = 0
        self._placement_hooks: list[PassEpochHook] = []
        self._stage_depth = 0
        self._stage_mechanisms: dict[int, _ResidencyMechanism] = {}
        self._active_stages: list[tuple[tuple[_ResidencyMechanism, ...], int, int | None]] = []

    def add_pass_epoch_hook(self, hook: PassEpochHook) -> None:
        """Run ``hook`` before each top-level placement operation.

        Hooks execute under the coordinator lock, before a manager can
        snapshot candidates. Nested manager frees remain in the same epoch.
        """
        with self._lock:
            self._placement_hooks.append(hook)

    @contextmanager
    def locked(self) -> Generator[None, None, None]:
        """Serialize non-placement registry maintenance with native stages."""
        with self._lock:
            yield

    @contextmanager
    def try_locked(self) -> Generator[bool, None, None]:
        """Try registry maintenance without waiting behind an active stage."""
        acquired = self._lock.acquire(blocking=False)
        try:
            yield acquired
        finally:
            if acquired:
                self._lock.release()

    @contextmanager
    def placement_pass(self) -> Generator[int, None, None]:
        """Bracket one top-level lock-held placement operation."""
        with self._lock:
            top_level = self._placement_depth == 0
            if top_level:
                self._placement_epoch += 1
            self._placement_depth += 1
            try:
                if top_level:
                    for hook in tuple(self._placement_hooks):
                        hook(self._placement_epoch)
                yield self._placement_epoch
            finally:
                self._placement_depth -= 1

    def enroll(self, handle: NativeRuntimeHandle) -> None:
        with self._lock:
            self._handles.add(handle)

    def advisory_unload_components(self, resource_identity: str) -> None:
        """Advisory-unload live component handles matching one identity."""
        if not resource_identity:
            raise ValueError("native component resource identity must be non-empty")
        if not self._lock.acquire(blocking=False):
            raise NativeResidencyBusyError("native stage is active; advisory unload refused")
        try:
            for handle in tuple(self._handles):
                if (
                    isinstance(handle, NativeComponentHandle)
                    and not handle.released
                    and handle.resource_identity == resource_identity
                ):
                    self.advisory_unload(cast("Any", handle))
        finally:
            self._lock.release()

    def enroll_component(
        self,
        module: object,
        *,
        load_device: Any,
        offload_device: Any,
        enroller: Callable[..., _ResidencyMechanism] | None = None,
        mechanism_factory: Any | None = None,
        **enrollment_kwargs: Any,
    ) -> _ResidencyMechanism:
        """Enroll one module with this coordinator's active memory inputs."""
        if enroller is None:
            inference_torch = importlib.import_module("dinkster_inference_torch")
            enroller = cast("Callable[..., _ResidencyMechanism]", inference_torch.enroll_component)
        kwargs = dict(enrollment_kwargs)
        device_type = str(getattr(load_device, "type", load_device)).split(":", 1)[0]
        if device_type == "mps":
            kwargs.update(
                memory_policy=self.manager.current_policy(),
                mps_snapshot=self.manager.mps_snapshot,
            )
        if mechanism_factory is not None:
            kwargs["mechanism_factory"] = mechanism_factory
        return enroller(
            module,
            load_device=load_device,
            offload_device=offload_device,
            **kwargs,
        )

    @contextmanager
    def stage(
        self,
        handle: NativeRuntimeHandle,
        mechanisms: Sequence[_ResidencyMechanism],
        *,
        memory_required: int = 0,
        minimum_memory: int | None = None,
        reserve_working_set: bool = False,
        clear_cache_after: bool = False,
        observer_stage: ExecutionStage = "load",
        parent_span_id: int | None = None,
    ) -> Generator[None, None, None]:
        if type(memory_required) is not int:
            raise TypeError("native stage memory_required must be an exact integer")
        if memory_required < 0:
            raise ValueError("native stage memory_required must be nonnegative")
        with self.placement_pass():
            with ExitStack() as reservations:
                stage = (tuple(mechanisms), memory_required, minimum_memory)
                self._active_stages.append(stage)
                primary: BaseException | None = None
                self._stage_depth += 1
                for mechanism in mechanisms:
                    self._stage_mechanisms[id(mechanism)] = mechanism
                try:
                    handle.require_active()
                    was_loaded = any(mechanism.loaded_bytes() > 0 for mechanism in mechanisms)
                    active_mechanisms: list[_ResidencyMechanism] = []
                    active_ids: set[int] = set()
                    for staged_mechanisms, _required, _minimum in self._active_stages:
                        for mechanism in staged_mechanisms:
                            if id(mechanism) not in active_ids:
                                active_ids.add(id(mechanism))
                                active_mechanisms.append(mechanism)
                    active_memory_required = max(
                        required for _mechanisms, required, _minimum in self._active_stages
                    )
                    active_minimums = [
                        minimum
                        for _mechanisms, _required, minimum in self._active_stages
                        if minimum is not None
                    ]
                    with native_execution_span(
                        observer_stage,
                        "prefetch",
                        parent_span_id=parent_span_id,
                        device=_mechanism_device(mechanisms),
                    ):
                        self.manager.load(
                            active_mechanisms,
                            memory_required=active_memory_required,
                            minimum_memory=max(active_minimums) if active_minimums else None,
                        )
                        self._reconcile()
                        if reserve_working_set:
                            reservations.enter_context(
                                self.manager.reserve_working_sets(mechanisms)
                            )
                        for mechanism in mechanisms:
                            reservations.enter_context(mechanism.execution_context())
                        if not was_loaded and any(
                            mechanism.loaded_bytes() > 0 for mechanism in mechanisms
                        ):
                            handle.run_lifecycle("load")
                    handle.run_lifecycle("pre-run")
                    handle.run_lifecycle("inject")
                    yield
                except BaseException as exc:
                    primary = exc
                finally:
                    try:
                        handle.run_lifecycle("eject")
                    except BaseException as exc:
                        if primary is None:
                            primary = exc
                        else:
                            primary.add_note(f"native eject also failed: {exc!r}")
                    try:
                        self._stage_depth -= 1
                        released_working_buffers = False
                        if self._stage_depth == 0:
                            staged = tuple(self._stage_mechanisms.values())
                            self._stage_mechanisms.clear()
                            for mechanism in staged:
                                released_working_buffers = (
                                    mechanism.release_working_buffers() or released_working_buffers
                                )
                        if (
                            clear_cache_after or released_working_buffers
                        ) and handle.load_device.type != "cpu":
                            self.manager.empty_cache(handle.load_device)
                    except BaseException as exc:
                        if primary is None:
                            primary = exc
                        else:
                            primary.add_note(f"native cache cleanup also failed: {exc!r}")
                    try:
                        self._reconcile()
                    finally:
                        if self._active_stages.pop() is not stage:
                            raise RuntimeError("native residency stages exited out of order")
                if primary is not None:
                    raise primary

    def advisory_unload(self, handle: NativeRuntimeHandle) -> None:
        if not self._lock.acquire(blocking=False):
            raise NativeResidencyBusyError("native stage is active; advisory unload refused")
        try:
            with native_execution_span(
                "load", "offload", device=_mechanism_device(handle.mechanisms)
            ):
                with self.placement_pass():
                    handle.require_active()
                    devices: list[Any] = []
                    was_loaded = False
                    for mechanism in handle.mechanisms:
                        mechanism_was_loaded = mechanism.loaded_bytes() > 0
                        was_loaded = was_loaded or mechanism_was_loaded
                        mechanism.unload()
                        if (
                            mechanism_was_loaded
                            and mechanism.load_device.type != "cpu"
                            and mechanism.load_device not in devices
                        ):
                            devices.append(mechanism.load_device)
                    for device in devices:
                        self.manager.empty_cache(device)
                    if was_loaded:
                        handle.run_lifecycle("unload")
                    self._reconcile()
        finally:
            self._lock.release()

    def unload_stage(
        self,
        handle: NativeRuntimeHandle,
        mechanisms: Sequence[_ResidencyMechanism],
        *,
        observer_stage: ExecutionStage,
        parent_span_id: int | None,
        component_role: str,
    ) -> None:
        with self._lock:
            with native_execution_span(
                observer_stage,
                "offload",
                parent_span_id=parent_span_id,
                component_role=component_role,
                device=_mechanism_device(mechanisms),
            ):
                with self.placement_pass():
                    handle.require_active()
                    devices: list[Any] = []
                    primary: BaseException | None = None
                    for mechanism in mechanisms:
                        if (
                            mechanism.loaded_bytes() > 0
                            and mechanism.load_device.type != "cpu"
                            and mechanism.load_device not in devices
                        ):
                            devices.append(mechanism.load_device)
                        try:
                            mechanism.unload()
                        except BaseException as exc:
                            if primary is None:
                                primary = exc
                            else:
                                primary.add_note(
                                    f"native stage mechanism offload also failed: {exc!r}"
                                )
                    for device in devices:
                        self.manager.empty_cache(device)
                    self._reconcile()
                    if primary is not None:
                        raise primary

    def terminal_release(self, handle: NativeRuntimeHandle) -> None:
        with self._lock:
            with native_execution_span(
                "load", "release", device=_mechanism_device(handle.mechanisms)
            ):
                handle.require_active()
                was_loaded = any(mechanism.loaded_bytes() > 0 for mechanism in handle.mechanisms)
                primary: BaseException | None = None
                try:
                    if isinstance(handle, NativeComponentHandle) and handle.discard_on_release:
                        self.manager.remove(handle.mechanisms, discard=True)
                    else:
                        self.manager.remove(handle.mechanisms)
                except BaseException as exc:
                    primary = exc
                else:
                    if isinstance(handle, NativeComponentHandle):
                        try:
                            handle.detach_residency_enrollment()
                        except BaseException as exc:
                            primary = exc
                if was_loaded:
                    try:
                        handle.run_lifecycle("unload")
                    except BaseException as exc:
                        if primary is None:
                            primary = exc
                        else:
                            primary.add_note(f"native unload also failed: {exc!r}")
                try:
                    handle.run_lifecycle("cleanup")
                except BaseException as exc:
                    if primary is None:
                        primary = exc
                    else:
                        primary.add_note(f"native cleanup also failed: {exc!r}")
                finally:
                    handle.mark_released()
                    self._handles.discard(handle)
                    handle.drop_materialized()
                    handle.reconcile_pool()
                if primary is not None:
                    raise primary

    def _reconcile(self) -> None:
        for handle in tuple(self._handles):
            handle.reconcile_pool()


class NativeComponentHandle:
    """Residency owner for one standalone module used by a runtime."""

    _dinkster_native_residency = True

    def __init__(
        self,
        module: object,
        mechanism: _ResidencyMechanism,
        load_device: Any,
        *,
        resource_identity: str,
        coordinator: NativeResidencyCoordinator,
        recipe: ReconstructionRecipe | None = None,
        materializer: ComponentHandleMaterializer | None = None,
        source_resolvers: Mapping[str, object] | None = None,
        residency_route_facts: Callable[[], ResidencyRouteFacts] | None = None,
        runtime: object | None = None,
        discard_on_release: bool = False,
    ) -> None:
        identity = cast("object", resource_identity)
        if not isinstance(identity, str) or not identity:
            raise ValueError("native component resource identity must be a non-empty string")
        if recipe is not None and recipe.runtime_identity != resource_identity:
            raise ValueError("native component identity does not match its reconstruction recipe")
        self._module = module
        self._runtime = runtime
        self._resource_identity = resource_identity
        self._recipe = recipe
        self._materializer = materializer
        self._source_resolvers = dict(source_resolvers or {})
        self._residency_route_facts = residency_route_facts
        self._discard_on_release = discard_on_release
        self.mechanisms: tuple[_ResidencyMechanism, ...] = (mechanism,)
        self.load_device = load_device
        self.coordinator = coordinator
        self._pool: _PoolPlacement | None = None
        self._released = False
        self._dependents: weakref.WeakSet[object] = weakref.WeakSet()
        coordinator.enroll(cast("Any", self))

    @property
    def module(self) -> object:
        self.require_active()
        return self._module

    @property
    def component(self) -> object:
        self.require_active()
        return self._module

    @property
    def runtime(self) -> object | None:
        self.require_active()
        return self._runtime

    @property
    def discard_on_release(self) -> bool:
        """The invocation owner guarantees module destruction after release."""
        return self._discard_on_release

    @property
    def resource_identity(self) -> str:
        return self._resource_identity

    @property
    def recipe(self) -> ReconstructionRecipe | None:
        return self._recipe

    @property
    def released(self) -> bool:
        return self._released

    @property
    def residency_route(self) -> ResidencyRouteFacts | None:
        """Diagnostic record of this handle's enrollment; never identity."""
        if self._residency_route_facts is None:
            return None
        return self._residency_route_facts()

    def require_active(self) -> None:
        if self._released:
            raise NativeRuntimeReleasedError("native component handle was terminally released")

    def register_dependent(self, dependent: object) -> None:
        with self.coordinator.locked():
            self.require_active()
            self._dependents.add(dependent)

    @contextmanager
    def terminal_release_guard(self) -> Generator[None, None, None]:
        with self.coordinator.try_locked() as acquired:
            if not acquired:
                raise NativeResidencyBusyError("native component handle is in active use")
            if self._dependents:
                raise NativeResidencyBusyError(
                    "native component handle is still referenced by a model overlay"
                )
            yield

    def run_lifecycle(self, phase: str, *, source: object | None = None) -> None:
        del source
        if phase not in LIFECYCLE_PHASES:
            raise ValueError(f"unknown native lifecycle phase {phase!r}")

    def mark_released(self) -> None:
        self._released = True

    def detach_residency_enrollment(self) -> None:
        module_dict = getattr(self._module, "__dict__", {})
        if "_dinkster_residency_state_store" not in module_dict:
            return
        inference_torch = importlib.import_module("dinkster_inference_torch")
        inference_torch.detach_residency_enrollment(self._module, self.mechanisms[0])

    def drop_materialized(self) -> None:
        self._module = None
        self._runtime = None
        self.mechanisms = ()

    def model_size(self) -> int:
        return sum(mechanism.total_bytes() for mechanism in self.mechanisms)

    def attach_pool(self, pool: _PoolPlacement) -> None:
        if self._discard_on_release:
            raise RuntimeError("discard-owned component handle cannot attach to a resident pool")
        if self._pool is not None and self._pool is not pool:
            raise RuntimeError("native component handle cannot change resident pools")
        self._pool = pool
        self.reconcile_pool()

    def reconcile_pool(self) -> None:
        if self._pool is not None:
            self._pool.set_loaded(
                self, any(mechanism.loaded_bytes() > 0 for mechanism in self.mechanisms)
            )

    @contextmanager
    def stage(
        self,
        *,
        memory_required: int = 0,
        clear_cache_after: bool = False,
        observer_stage: ExecutionStage = "load",
        parent_span_id: int | None = None,
    ) -> Generator[None, None, None]:
        with self.coordinator.stage(
            cast("Any", self),
            self.mechanisms,
            memory_required=memory_required,
            clear_cache_after=clear_cache_after,
            observer_stage=observer_stage,
            parent_span_id=parent_span_id,
        ):
            yield

    @contextmanager
    def stage_with(
        self,
        runtime_handle: InferenceRuntimeHandle,
        role: str,
    ) -> Generator[None, None, None]:
        if not isinstance(runtime_handle, NativeRuntimeHandle):
            raise TypeError(
                "native component composite staging requires a NativeRuntimeHandle, "
                f"got {type(runtime_handle).__name__}"
            )
        if runtime_handle.coordinator is not self.coordinator:
            raise ValueError("component and runtime handles use different residency coordinators")
        with self.coordinator.locked():
            self.require_active()
            runtime_handle.require_active()
            with ExitStack() as stages:
                stages.enter_context(self.stage())
                stages.enter_context(runtime_handle.stage(role))
                yield

    def advisory_unload(self) -> None:
        self.coordinator.advisory_unload(cast("Any", self))

    def terminal_release(self) -> None:
        with self.coordinator.locked():
            self._terminal_release_locked()

    def rebuild(self) -> NativeComponentHandle:
        """Rematerialize this component from its retained recipe."""

        if self._recipe is None or self._materializer is None:
            raise RuntimeError("native component handle has no reconstruction recipe")
        return self._materializer(self._recipe, self._source_resolvers)

    def clone(
        self,
        overlays_delta: tuple[PatchOverlay, ...] = (),
        *,
        source_resolvers: Mapping[str, object] | None = None,
    ) -> NativeComponentHandle:
        """Freshly materialize this component with appended weight overlays."""

        self.require_active()
        if self._recipe is None or self._materializer is None:
            raise RuntimeError("native component handle has no reconstruction recipe")
        resolvers = dict(self._source_resolvers)
        resolvers.update(source_resolvers or {})
        return self._materializer(self._recipe.append_overlays(overlays_delta), resolvers)

    def _terminal_release_locked(self) -> None:
        if self._dependents:
            raise NativeResidencyBusyError(
                "native component handle is still referenced by a model overlay"
            )
        self.coordinator.terminal_release(cast("Any", self))


class NativeComponentPublisher:
    """Publish standalone modules into one pack-owned residency lifetime."""

    def __init__(
        self,
        *,
        coordinator: NativeResidencyCoordinator | None = None,
        _torch_module: Any | None = None,
        _enroll_component: Callable[..., _ResidencyMechanism] | None = None,
        _pool: _PoolPlacement | None = None,
    ) -> None:
        self._coordinator = coordinator or default_native_residency()
        self._torch_module = _torch_module
        self._enroll_component = _enroll_component
        self._pool = _pool
        self._handles: set[NativeComponentHandle] = set()
        self._lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = False

    def publish(self, module: object, *, resource_identity: str) -> NativeComponentHandle:
        with self._lock:
            if self._closed:
                raise RuntimeError("native component publisher is closed")
            identity = cast("object", resource_identity)
            if not isinstance(identity, str) or not identity:
                raise ValueError("component resource identity must be a non-empty string")
            torch = self._torch_module or importlib.import_module("torch")
            if not isinstance(module, torch.nn.Module):
                raise TypeError(f"component must be a torch.nn.Module, got {type(module).__name__}")
            enroll_component = self._enroll_component
            if enroll_component is None:
                inference_torch = importlib.import_module("dinkster_inference_torch")
                enroll_component = inference_torch.enroll_component
            load_device = select_load_device(torch)
            mechanism = self._coordinator.enroll_component(
                module,
                load_device=load_device,
                offload_device=torch.device("cpu"),
                enroller=enroll_component,
            )
            handle = NativeComponentHandle(
                module,
                mechanism,
                load_device,
                resource_identity=resource_identity,
                coordinator=self._coordinator,
            )
            try:
                pool = self._pool
                if pool is None:
                    pool = importlib.import_module("dinkster_native.pool").default_pool()
                    self._pool = pool
                label = getattr(pool, "label", None)
                if callable(label):
                    label(handle, resource_identity)
                handle.attach_pool(pool)
            except BaseException as exc:
                try:
                    handle.terminal_release()
                except BaseException as cleanup_exc:
                    exc.add_note(f"native component cleanup also failed: {cleanup_exc!r}")
                raise
            self._handles.add(handle)
            return handle

    def close(self) -> None:
        with self._close_lock:
            with self._lock:
                if self._closed and not self._handles:
                    return
                self._closed = True
                handles = tuple(self._handles)
            primary: BaseException | None = None
            for handle in handles:
                if handle.released:
                    with self._lock:
                        self._handles.discard(handle)
                    continue
                try:
                    release_resident = getattr(self._pool, "release_resident", None)
                    if callable(release_resident):
                        released = release_resident(handle, honor_pins=False)
                        if not released and not handle.released:
                            raise NativeResidencyBusyError(
                                "native component release was refused by the resident pool"
                            )
                    else:
                        handle.terminal_release()
                except BaseException as exc:
                    if primary is None:
                        primary = exc
                    else:
                        primary.add_note(f"native component cleanup also failed: {exc!r}")
                if handle.released:
                    with self._lock:
                        self._handles.discard(handle)
            if primary is not None:
                raise primary


class NativeRuntimeHandle:
    """One resident identity for an assembled native checkpoint bundle.

    MODEL, CLIP, and VAE outputs all publish this same object. Its
    components enroll as separate per-unit manager mechanisms, while its pool
    cost and terminal lifetime stay one conservative allocation group.
    """

    _dinkster_native_residency = True

    def __init__(
        self,
        runtime: Any,
        load_device: object,
        *,
        recipe: ReconstructionRecipe,
        patch_sets: Mapping[str, object] | None = None,
        patch_weight_dtype: object | None = None,
        patch_key_prefixes: Mapping[str, str] | None = None,
        storage_dtypes: Mapping[str, object] | None = None,
        materializer: HandleMaterializer | None = None,
        source_resolvers: Mapping[str, object] | None = None,
        attachments: Mapping[str, object] | None = None,
        lifecycle_callbacks: Mapping[str, Mapping[str, Sequence[LifecycleCallback]]] | None = None,
        coordinator: NativeResidencyCoordinator | None = None,
        mechanism_factory: Any | None = None,
        residency_route_facts: Callable[[], ResidencyRouteFacts] | None = None,
        _torch_module: Any | None = None,
        _enroll_assembled: _EnrollAssembled | None = None,
    ) -> None:
        torch = _torch_module if _torch_module is not None else importlib.import_module("torch")
        if runtime.runtime_identity != recipe.runtime_identity:
            raise ValueError("native runtime identity does not match its reconstruction recipe")
        self._runtime: Any | None = runtime
        self._recipe = recipe
        self._residency_route_facts = residency_route_facts
        self._materializer = materializer
        self._source_resolvers = dict(source_resolvers or {})
        self._attachments = dict(attachments or {})
        declarations = {attachment.name for attachment in recipe.attachments}
        unknown_attachments = set(self._attachments) - declarations
        if unknown_attachments:
            raise ValueError(
                "live attachments lack recipe declarations: "
                + ", ".join(sorted(unknown_attachments))
            )
        self._callbacks: dict[str, dict[str, list[LifecycleCallback]]] = {
            phase: {} for phase in LIFECYCLE_PHASES
        }
        for phase, keyed in (lifecycle_callbacks or {}).items():
            if phase not in self._callbacks:
                raise ValueError(f"unknown native lifecycle phase {phase!r}")
            for key, callbacks in keyed.items():
                if not key:
                    raise ValueError("native lifecycle callback key must be non-empty")
                callback_list = list(callbacks)
                if not all(callable(callback) for callback in callback_list):
                    raise TypeError("native lifecycle callbacks must be callable")
                self._callbacks[phase][key] = callback_list
        self.load_device = torch.device(load_device)
        if self.load_device.type not in ("cpu", "cuda", "mps", "xpu"):
            raise ValueError(
                "native runtime residency supports cpu, cuda, mps, and xpu "
                f"devices, got {self.load_device.type!r}"
            )
        self.offload_device = torch.device("cpu")
        self.coordinator = coordinator if coordinator is not None else default_native_residency()
        self._pool: _PoolPlacement | None = None
        self._released = False
        self._residency_pool: NativeResidencyPool | None = None
        self._residency_token: NativeResidencyToken | None = None
        self._dependency_plan: DependencyPlan | None = None
        self._dependency_handles: dict[DependencyPath, NativeRuntimeHandle] = {}
        self._conditional_dependency_materializer: ConditionalDependencyMaterializer | None = None

        policy = getattr(runtime, "residency_policy", None)
        inference_torch = None
        enroll_assembled = _enroll_assembled
        if policy is not None:
            inference_torch = importlib.import_module("dinkster_inference_torch")
            if not isinstance(policy, inference_torch.NativeResidencyPolicy):
                raise TypeError("runtime residency_policy must be a NativeResidencyPolicy or None")
        if enroll_assembled is None:
            if inference_torch is None:
                inference_torch = importlib.import_module("dinkster_inference_torch")
            enroll_assembled = cast("_EnrollAssembled", inference_torch.enroll_assembled)
        mps_admission: dict[str, Any] = {}
        if self.load_device.type == "mps":
            mps_admission = {
                "memory_policy": self.coordinator.manager.current_policy(),
                "mps_snapshot": self.coordinator.manager.mps_snapshot,
            }
        storage_kwargs = {} if storage_dtypes is None else {"storage_dtypes": storage_dtypes}
        if mechanism_factory is None:
            if patch_sets:
                enrolled = enroll_assembled(
                    runtime.assembled,
                    load_device=self.load_device,
                    offload_device=self.offload_device,
                    patch_sets=patch_sets,
                    patch_weight_dtype=patch_weight_dtype,
                    patch_key_prefixes=patch_key_prefixes,
                    **storage_kwargs,
                    **mps_admission,
                )
            else:
                enrolled = enroll_assembled(
                    runtime.assembled,
                    load_device=self.load_device,
                    offload_device=self.offload_device,
                    **storage_kwargs,
                    **mps_admission,
                )
        else:
            if patch_sets:
                enrolled = enroll_assembled(
                    runtime.assembled,
                    load_device=self.load_device,
                    offload_device=self.offload_device,
                    patch_sets=patch_sets,
                    patch_weight_dtype=patch_weight_dtype,
                    patch_key_prefixes=patch_key_prefixes,
                    mechanism_factory=mechanism_factory,
                    **storage_kwargs,
                    **mps_admission,
                )
            else:
                enrolled = enroll_assembled(
                    runtime.assembled,
                    load_device=self.load_device,
                    offload_device=self.offload_device,
                    mechanism_factory=mechanism_factory,
                    **storage_kwargs,
                    **mps_admission,
                )
        declared_components: Mapping[str, object] | None = getattr(
            runtime.assembled, "components", None
        )
        policy_enrollment = False
        if policy is not None:
            expected_components = tuple(
                component
                for component in policy.enrollment_components
                if (
                    component in declared_components
                    if declared_components is not None
                    else getattr(runtime.assembled, component, None) is not None
                )
            )
            policy_enrollment = any(
                component in enrolled for component in policy.enrollment_components
            )
            if policy_enrollment and (
                expected_components not in policy.enrollment_orders
                or tuple(enrolled) != expected_components
            ):
                raise ValueError(
                    "native runtime enrollment must return exactly "
                    + ", ".join(expected_components)
                    + " in declaration order"
                )
        self._unload_after_stage: frozenset[str] = (
            policy.unload_after_stage if policy is not None and policy_enrollment else frozenset()
        )
        standalone_model = tuple(enrolled) == ("diffusion",)
        role_by_component = {
            "diffusion": "diffusion",
            "clip_l": "text",
            "clip_g": "text",
            "t5xxl": "text",
            "umt5xxl": "text",
            "qwen3_4b": "text",
            "gemma2_2b": "text",
            "text_encoder": "text",
            "text": "text",
            "clip_vision": "vision",
            "vae": "vae",
        }
        if declared_components is not None:
            context = current_execution_context()
            registries = cast(
                "Any",
                context.inference_registries
                if context is not None and context.inference_registries is not None
                else importlib.import_module("dinkster_inference").builtin_registries(),
            )
            descriptor = registries.components.get(runtime.family.id)
            role_by_component = {
                component: role_by_component.get(component, component)
                for component in declared_components
            }
            if descriptor is not None:
                role_by_component.update({role: "text" for role in descriptor.text_encoder_roles})
                role_by_component.update({role: "vae" for role in descriptor.codec_roles})
                role_by_component[descriptor.model_role] = "diffusion"
        declared_roles: tuple[str, ...] = ()
        if policy is not None:
            collisions = set(policy.enrollment_components) & role_by_component.keys()
            if collisions and declared_components is None:
                raise ValueError(
                    "native residency policy components collide with classic components: "
                    + ", ".join(sorted(collisions))
                )
            role_by_component.update(policy.component_roles)
            declared_roles = tuple(policy.component_roles.values())
        role_lists: dict[str, list[_ResidencyMechanism]] = {
            role: []
            for role in dict.fromkeys(
                ("diffusion", "text", "vision", "vae", *role_by_component.values(), *declared_roles)
            )
        }
        streamed_declaration = cast(
            "object", getattr(runtime, "streamed_residency_components", frozenset[str]())
        )
        if not isinstance(streamed_declaration, frozenset):
            raise TypeError("streamed residency components must be a frozenset of strings")
        streamed_values = cast("frozenset[object]", streamed_declaration)
        if not all(isinstance(component, str) for component in streamed_values):
            raise TypeError("streamed residency components must be a frozenset of strings")
        streamed_components = cast("frozenset[str]", streamed_values)
        unknown_streamed = streamed_components - enrolled.keys()
        if unknown_streamed:
            raise ValueError(
                "native runtime streams unknown components: " + ", ".join(sorted(unknown_streamed))
            )
        retained_declaration = cast(
            "object",
            getattr(runtime, "retained_offload_storage_components", frozenset[str]()),
        )
        if not isinstance(retained_declaration, frozenset):
            raise TypeError("retained offload storage components must be a frozenset of strings")
        retained_values = cast("frozenset[object]", retained_declaration)
        if not all(isinstance(component, str) for component in retained_values):
            raise TypeError("retained offload storage components must be a frozenset of strings")
        retained_components = cast("frozenset[str]", retained_values)
        unknown_retained = retained_components - enrolled.keys()
        if unknown_retained:
            raise ValueError(
                "native runtime retains offload storage for unknown components: "
                + ", ".join(sorted(unknown_retained))
            )
        mechanisms: list[_ResidencyMechanism] = []
        eager_role_lists: dict[str, list[_ResidencyMechanism]] = {role: [] for role in role_lists}
        for component, mechanism in enrolled.items():
            try:
                role = role_by_component[component]
            except KeyError:
                raise ValueError(
                    f"native runtime enrollment returned unknown component {component!r}"
                ) from None
            if component in retained_components:
                mechanism.retain_offload_storage()
            role_lists[role].append(mechanism)
            if component not in streamed_components:
                eager_role_lists[role].append(mechanism)
            mechanisms.append(mechanism)
        by_role = {
            role: tuple(role_mechanisms)
            for role, role_mechanisms in role_lists.items()
            if role_mechanisms
        }
        has_diffusion = "diffusion" in by_role or (
            policy is not None
            and policy_enrollment
            and any(role in by_role for role in policy.diffusion_roles)
        )
        if (
            declared_components is None
            and not standalone_model
            and (not has_diffusion or "text" not in by_role or "vae" not in by_role)
        ):
            raise ValueError(
                "native runtime must provide diffusion, text encoder, and VAE components"
            )
        self._by_role = by_role
        self._eager_by_role = {
            role: tuple(role_mechanisms)
            for role, role_mechanisms in eager_role_lists.items()
            if role in by_role
        }
        self.mechanisms = tuple(mechanisms)
        self.coordinator.enroll(self)

    @property
    def runtime(self) -> Any:
        self.require_active()
        if self._runtime is None:
            raise NativeRuntimeReleasedError("native checkpoint materialization was dropped")
        return self._runtime

    @property
    def recipe(self) -> ReconstructionRecipe:
        return self._recipe

    def has_residency_stage(self, role: str) -> bool:
        """Return whether the live runtime enrolled the named stage."""
        self.require_active()
        return bool(self._by_role.get(role))

    @property
    def residency_route(self) -> ResidencyRouteFacts | None:
        """Diagnostic record of this handle's enrollment; never identity."""
        if self._residency_route_facts is None:
            return None
        return self._residency_route_facts()

    @property
    def attachments(self) -> Mapping[str, object]:
        return dict(self._attachments)

    @property
    def released(self) -> bool:
        return self._released

    @property
    def materialization_key(self) -> str | None:
        token = self._residency_token
        return None if token is None else token.materialization_key

    @property
    def owned_dependencies(self) -> Mapping[DependencyPath, NativeRuntimeHandle]:
        return dict(self._dependency_handles)

    def bind_residency_token(self, pool: NativeResidencyPool, token: NativeResidencyToken) -> None:
        if self._residency_token is not None:
            raise RuntimeError("native handle already owns a residency token")
        self._residency_pool = pool
        self._residency_token = token

    def bind_dependencies(
        self,
        plan: DependencyPlan,
        handles: Mapping[DependencyPath, NativeRuntimeHandle],
        conditional_materializer: ConditionalDependencyMaterializer,
    ) -> None:
        if self._dependency_plan is not None:
            raise RuntimeError("native handle dependencies are already bound")
        expected = set(plan.load_order) - {()}
        unknown = set(handles) - expected
        if unknown:
            raise ValueError("native dependency handles contain unknown paths")
        self._dependency_plan = plan
        self._dependency_handles = dict(handles)
        self._conditional_dependency_materializer = conditional_materializer

    def require_active(self) -> None:
        if self._released:
            raise NativeRuntimeReleasedError("native checkpoint handle was terminally released")

    def mark_released(self) -> None:
        self._released = True

    def drop_materialized(self) -> None:
        """Drop every live runtime/module reference while retaining identity."""
        self._runtime = None
        self._by_role = {"diffusion": (), "text": (), "vae": ()}
        self._eager_by_role = {"diffusion": (), "text": (), "vae": ()}
        self.mechanisms = ()

    def _release_residency_token(self) -> None:
        token = self._residency_token
        pool = self._residency_pool
        if token is None:
            return
        assert pool is not None
        pool.release(token)
        self._residency_token = None
        self._residency_pool = None

    def register_lifecycle_callback(
        self, phase: str, key: str, callback: LifecycleCallback
    ) -> None:
        self.require_active()
        if phase not in self._callbacks:
            raise ValueError(f"unknown native lifecycle phase {phase!r}")
        if not key:
            raise ValueError("native lifecycle callback key must be non-empty")
        if not callable(callback):
            raise TypeError("native lifecycle callback must be callable")
        self._callbacks[phase].setdefault(key, []).append(callback)

    def _callback_snapshot(
        self,
    ) -> dict[str, dict[str, tuple[LifecycleCallback, ...]]]:
        return {
            phase: {key: tuple(callbacks) for key, callbacks in keyed.items()}
            for phase, keyed in self._callbacks.items()
        }

    def run_lifecycle(self, phase: str, *, source: NativeRuntimeHandle | None = None) -> None:
        try:
            keyed = self._callbacks[phase]
        except KeyError:
            raise ValueError(f"unknown native lifecycle phase {phase!r}") from None
        failures: list[BaseException] = []
        event = NativeLifecycleEvent(phase=phase, handle=self, source=source)
        for key in sorted(keyed):
            for callback in tuple(keyed[key]):
                try:
                    callback(event)
                except BaseException as exc:
                    failures.append(exc)
        if failures:
            raise NativeLifecycleCallbackError(phase, tuple(failures))

    def attach(self, name: str, value: object) -> None:
        self.require_active()
        if name not in {item.name for item in self._recipe.attachments}:
            raise ValueError(f"attachment {name!r} has no recipe declaration")
        self._attachments[name] = value

    @staticmethod
    def _rebuild_attachment(declaration: AttachmentDeclaration) -> object:
        if not declaration.rebuildable:
            raise AttachmentCloneError(
                declaration.name,
                "no serializable rebuild data or worker-local rebuild callback was declared",
            )
        data = dict(declaration.rebuild_data or ())
        entry = declaration.rebuild_entry_point
        if entry is None:
            return data
        module_name, attr_name = entry.split(":", 1)
        rebuild = getattr(importlib.import_module(module_name), attr_name, None)
        if not callable(rebuild):
            raise AttachmentCloneError(
                declaration.name,
                f"rebuild callback {entry!r} is not callable",
            )
        return rebuild(data, declaration.device)

    def _attachments_for_clone(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for declaration in self._recipe.attachments:
            if not declaration.rebuildable:
                raise AttachmentCloneError(
                    declaration.name,
                    "no serializable rebuild data or worker-local rebuild callback was declared",
                )
            if declaration.clone == "refuse":
                raise AttachmentCloneError(
                    declaration.name, "clone semantics explicitly refuse cloning"
                )
            if declaration.clone == "rebuild":
                result[declaration.name] = self._rebuild_attachment(declaration)
                continue
            try:
                value = self._attachments[declaration.name]
            except KeyError:
                raise AttachmentCloneError(
                    declaration.name, "the declared live attachment is missing"
                ) from None
            if declaration.clone == "share":
                result[declaration.name] = value
                continue
            try:
                result[declaration.name] = copy.deepcopy(value)
            except Exception as exc:
                raise AttachmentCloneError(
                    declaration.name,
                    f"copy semantics failed: {type(exc).__name__}: {exc}",
                ) from exc
        return result

    def _materialize_recipe(
        self,
        recipe: ReconstructionRecipe,
        *,
        attachments: Mapping[str, object],
        source_resolvers: Mapping[str, object] | None = None,
    ) -> NativeRuntimeHandle:
        materializer = self._materializer
        if materializer is None:
            raise NativeRebuildUnavailableError(
                "native handle has no worker-local recipe materializer"
            )
        resolvers = dict(self._source_resolvers)
        resolvers.update(source_resolvers or {})
        rebuilt = materializer(recipe, resolvers)
        rebuilt._attachments.update(attachments)
        rebuilt._callbacks = {
            phase: {key: list(callbacks) for key, callbacks in keyed.items()}
            for phase, keyed in self._callback_snapshot().items()
        }
        if self._pool is not None:
            rebuilt.attach_pool(self._pool)
        return rebuilt

    def clone(
        self,
        overlays_delta: tuple[PatchOverlay, ...] = (),
        *,
        source_resolvers: Mapping[str, object] | None = None,
    ) -> NativeRuntimeHandle:
        """Freshly materialize an owned runtime for base recipe + overlays.

        V1 shares only immutable declarations, worker-local callback code,
        explicitly share-mode attachments, and the resident pool. Module
        state and patched stores are always newly materialized because the
        residency mechanisms mutate loaded storage while retaining backups.
        """
        return self._clone_recipe(
            self._recipe.append_overlays(overlays_delta),
            source_resolvers=source_resolvers,
        )

    def clone_with_attention_policy(
        self,
        attention_policy: object,
    ) -> NativeRuntimeHandle:
        """Rematerialize this runtime with one explicit attention route."""
        policy = validate_attention_policy(attention_policy)
        context = current_execution_context()
        if context is None or context.attention_capabilities is None:
            raise RuntimeError("an attention backend requires selected worker capability evidence")
        source_token = self._recipe.knobs.attention_route_token
        role_policies = () if source_token is None else source_token.requested_role_policies
        token = derive_attention_route_token(
            context.attention_capabilities,
            AttentionPolicyConfig(policy, role_policies),
        )
        if token.version == 3:
            raise RuntimeError(
                f"required attention policy {policy!r} is unavailable on the selected worker"
            )
        knobs = replace(
            self._recipe.knobs,
            attention_policy=policy,
            attention_route_token=token,
        )
        return self._clone_recipe(replace(self._recipe, knobs=knobs))

    def _clone_recipe(
        self,
        recipe: ReconstructionRecipe,
        *,
        source_resolvers: Mapping[str, object] | None = None,
    ) -> NativeRuntimeHandle:
        self.require_active()
        attachments = self._attachments_for_clone()
        clone = self._materialize_recipe(
            recipe,
            attachments=attachments,
            source_resolvers=source_resolvers,
        )
        try:
            clone.run_lifecycle("clone", source=self)
        except BaseException as exc:
            try:
                clone.terminal_release()
            except BaseException as cleanup:
                exc.add_note(f"failed clone cleanup also failed: {cleanup!r}")
            raise
        return clone

    def rebuild(self) -> NativeRuntimeHandle:
        """Rematerialize this recipe after all live state was dropped."""
        attachments = {
            declaration.name: self._rebuild_attachment(declaration)
            for declaration in self._recipe.attachments
        }
        return self._materialize_recipe(
            self._recipe,
            attachments=attachments,
        )

    def model_size(self) -> int:
        return sum(mechanism.total_bytes() for mechanism in self.mechanisms)

    def attach_pool(self, pool: _PoolPlacement) -> None:
        if self._pool is not None and self._pool is not pool:
            raise RuntimeError("native checkpoint handle cannot change resident pools")
        self._pool = pool
        self.reconcile_pool()

    def _active_dependency_handles(
        self, active_conditional: frozenset[DependencyPath]
    ) -> tuple[NativeRuntimeHandle, ...]:
        plan = self._dependency_plan
        if plan is None:
            if active_conditional:
                raise ValueError("native handle has no conditional dependencies")
            return ()
        active = plan.select_active(active_conditional)
        missing = {
            path for path in active.load_order if path and path not in self._dependency_handles
        }
        if missing:
            materializer = self._conditional_dependency_materializer
            assert materializer is not None
            built = materializer(active_conditional)
            if set(built) != missing:
                raise RuntimeError("conditional dependency materializer returned unexpected paths")
            self._dependency_handles.update(built)
        return tuple(self._dependency_handles[path] for path in active.load_order if path)

    @contextmanager
    def stage(
        self,
        role: str,
        *,
        active_conditional: frozenset[DependencyPath] = frozenset(),
        unload_before: tuple[str, ...] = (),
        memory_required: int = 0,
        minimum_memory: int | None = None,
        observer_stage: ExecutionStage = "load",
        parent_span_id: int | None = None,
    ) -> Generator[None, None, None]:
        with self.coordinator.locked():
            self.require_active()
            try:
                mechanisms = self._by_role[role]
                eager_mechanisms = self._eager_by_role[role]
            except KeyError:
                raise ValueError(f"unknown native runtime stage {role!r}") from None
            plan = self._dependency_plan
            if plan is None:
                if active_conditional:
                    raise ValueError("native handle has no conditional dependencies")
            else:
                plan.select_active(active_conditional)
            for unload_role in unload_before:
                if unload_role == role:
                    raise ValueError("native runtime stage cannot unload itself before execution")
                try:
                    unload_mechanisms = self._by_role[unload_role]
                except KeyError:
                    raise ValueError(f"unknown native runtime stage {unload_role!r}") from None
                self.coordinator.unload_stage(
                    self,
                    unload_mechanisms,
                    observer_stage=observer_stage,
                    parent_span_id=parent_span_id,
                    component_role=unload_role,
                )
            with native_execution_span(
                observer_stage,
                "lease",
                parent_span_id=parent_span_id,
                component_role=role,
                device=str(self.load_device),
            ) as lease_span:
                lease_parent = parent_span_id if lease_span is None else lease_span.span_id
                try:
                    dependencies = self._active_dependency_handles(active_conditional)
                    primary: BaseException | None = None
                    try:
                        with ExitStack() as stack:
                            for dependency in dependencies:
                                stack.enter_context(
                                    dependency.stage(
                                        role,
                                        observer_stage=observer_stage,
                                        parent_span_id=lease_parent,
                                    )
                                )
                            stack.enter_context(
                                self.coordinator.stage(
                                    self,
                                    eager_mechanisms,
                                    memory_required=memory_required,
                                    minimum_memory=minimum_memory,
                                    reserve_working_set=role == "vae",
                                    clear_cache_after=(
                                        role == "vae"
                                        or any(mechanism.demand_paged for mechanism in mechanisms)
                                    ),
                                    observer_stage=observer_stage,
                                    parent_span_id=lease_parent,
                                )
                            )
                            yield
                    except BaseException as exc:
                        primary = exc
                    if role in self._unload_after_stage and observer_stage == "condition":
                        try:
                            self.coordinator.unload_stage(
                                self,
                                mechanisms,
                                observer_stage=observer_stage,
                                parent_span_id=lease_parent,
                                component_role=role,
                            )
                        except BaseException as exc:
                            if primary is None:
                                primary = exc
                            else:
                                primary.add_note(f"native stage offload also failed: {exc!r}")
                    if primary is not None:
                        raise primary
                except BaseException as exc:
                    if self._dependency_plan is not None and not self.released:
                        try:
                            self.terminal_release()
                        except BaseException as cleanup:
                            exc.add_note(
                                f"failed dependency stage cleanup also failed: {cleanup!r}"
                            )
                    raise

    def advisory_unload(self) -> None:
        self.coordinator.advisory_unload(self)

    def terminal_release(self) -> None:
        with self.coordinator.locked():
            self._terminal_release_locked()

    def _terminal_release_locked(self) -> None:
        primary: BaseException | None = None
        try:
            self.coordinator.terminal_release(self)
        except BaseException as exc:
            primary = exc
        try:
            self._release_residency_token()
        except BaseException as exc:
            if primary is None:
                primary = exc
            else:
                primary.add_note(f"native residency token release also failed: {exc!r}")
        plan = self._dependency_plan
        release_order = () if plan is None else plan.release_order[1:]
        for path in release_order:
            dependency = self._dependency_handles.get(path)
            if dependency is None or dependency.released:
                continue
            try:
                dependency.terminal_release()
            except BaseException as exc:
                if primary is None:
                    primary = exc
                else:
                    primary.add_note(f"native dependency {path!r} release also failed: {exc!r}")
        self._dependency_handles.clear()
        self._conditional_dependency_materializer = None
        if primary is not None:
            raise primary

    def reconcile_pool(self) -> None:
        if self._pool is not None:
            self._pool.set_loaded(
                self,
                any(mechanism.loaded_bytes() > 0 for mechanism in self.mechanisms),
            )


_coordinator: NativeResidencyCoordinator | None = None
_dynamic_coordinator: NativeResidencyCoordinator | None = None
_coordinator_init_lock = threading.Lock()
_dependency_residency_pool = NativeResidencyPool()


def default_dependency_residency_pool() -> NativeResidencyPool:
    return _dependency_residency_pool


def default_native_residency(*, free_memory: Any | None = None) -> NativeResidencyCoordinator:
    """The native worker process's manager and stage lock for one arm.

    Production workers receive one process-local aimdo mode from validated
    child argv, so they construct either eager handles or dynamically armed
    handles, never both. Separate singletons preserve that invariant in tests
    that deliberately toggle the process-local mode to compare both paths.
    """
    global _coordinator, _dynamic_coordinator  # noqa: PLW0603
    coordinator = _coordinator if free_memory is None else _dynamic_coordinator
    if coordinator is None:
        with _coordinator_init_lock:
            coordinator = _coordinator if free_memory is None else _dynamic_coordinator
            if coordinator is None:
                coordinator = NativeResidencyCoordinator(free_memory=free_memory)
                if free_memory is None:
                    _coordinator = coordinator
                else:
                    _dynamic_coordinator = coordinator
    return coordinator


__all__ = [
    "AttachmentCloneError",
    "LIFECYCLE_PHASES",
    "LifecycleCallback",
    "NativeLifecycleCallbackError",
    "NativeLifecycleEvent",
    "NativeComponentHandle",
    "NativeComponentPublisher",
    "NativeResidencyBusyError",
    "NativeResidencyCoordinator",
    "NativeResidencyPool",
    "NativeResidencyToken",
    "NativeRuntimeHandle",
    "NativeRuntimeReleasedError",
    "NativeRebuildUnavailableError",
    "NativeStageEvent",
    "NativeStageObserver",
    "current_native_observer",
    "default_dependency_residency_pool",
    "default_native_residency",
    "native_accelerator_family",
    "native_execution_span",
    "observe_native_stages",
    "select_current_device",
    "select_intermediate_device",
    "select_load_device",
]
