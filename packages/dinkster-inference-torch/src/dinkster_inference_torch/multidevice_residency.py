"""Lane-local persistent residency on one explicit device."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import TypeAlias, cast

import torch
from dinkster_inference.replica_lane import (
    ReplicaAdmissionRefusal,
    ReplicaPreparationRefusal,
    ReplicaUnitFailure,
    WorkUnitDefinition,
)

from .attention import AttentionKernel
from .module_residency import EnrolledResidency, enroll_component


class DeviceReplicaResidencyError(RuntimeError):
    """The per-device host cannot perform the requested lifecycle action."""


class DeviceReplicaResidencyState(StrEnum):
    NEW = "NEW"
    PREPARED = "PREPARED"
    REFUSED = "REFUSED"
    ADMITTED = "ADMITTED"
    ACTIVE = "ACTIVE"
    CANCELLED = "CANCELLED"
    ABORTED = "ABORTED"
    FAILED = "FAILED"
    RELEASED = "RELEASED"


UnitRunner: TypeAlias = Callable[[WorkUnitDefinition], ReplicaUnitFailure | None]
ReplicaBuild: TypeAlias = tuple[torch.nn.Module, AttentionKernel, UnitRunner]
ReplicaBuilder: TypeAlias = Callable[[], ReplicaBuild | ReplicaPreparationRefusal]
AdmissionGate: TypeAlias = Callable[[], ReplicaAdmissionRefusal | None]


def _explicit_device(value: object, name: str) -> torch.device:
    if not isinstance(value, torch.device):
        raise TypeError(f"{name} must be a torch.device")
    if value.type not in ("cpu", "cuda") or value == torch.device(value.type):
        raise ValueError(f"{name} must be an explicitly indexed CPU or CUDA device")
    return value


def _as_object(value: object) -> object:
    return value


def _clear_enrollment(module: torch.nn.Module | None) -> None:
    if module is None:
        return
    module.__dict__.pop("_dinkster_resident_weights", None)
    for owner in module.modules():
        owner.__dict__.pop("_residency", None)


class DeviceReplicaResidencyHost:
    """Own one built module and its full residency for one lane attempt."""

    def __init__(
        self,
        device: object,
        offload_device: object,
        builder: object,
        *,
        admission_gate: object | None = None,
    ) -> None:
        self.device = _explicit_device(device, "device")
        if not isinstance(offload_device, torch.device):
            raise TypeError("offload_device must be a torch.device")
        if offload_device.type not in ("cpu", "cuda"):
            raise ValueError("offload_device must be a CPU or CUDA device")
        if not callable(builder):
            raise TypeError("builder must be callable")
        if admission_gate is not None and not callable(admission_gate):
            raise TypeError("admission_gate must be callable")
        self.offload_device = offload_device
        self._builder = cast(ReplicaBuilder, builder)
        self._admission_gate = cast(AdmissionGate | None, admission_gate)
        self.state = DeviceReplicaResidencyState.NEW
        self._module: torch.nn.Module | None = None
        self._mechanism: EnrolledResidency | None = None
        self._attention_kernel: AttentionKernel | None = None
        self._unit_runner: UnitRunner | None = None
        self._release_called = False

    def _active_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        if self.state is not DeviceReplicaResidencyState.ACTIVE:
            raise DeviceReplicaResidencyError("attention kernel is available only while active")
        kernel = self._attention_kernel
        if kernel is None:
            raise DeviceReplicaResidencyError("active host has no attention kernel")
        return kernel(
            q,
            k,
            v,
            mask=mask,
            causal=causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )

    @property
    def attention_kernel(self) -> AttentionKernel:
        if self.state is not DeviceReplicaResidencyState.ACTIVE:
            raise DeviceReplicaResidencyError("attention kernel is available only while active")
        return self._active_attention

    @property
    def resident_bytes(self) -> int:
        mechanism = self._mechanism
        return 0 if mechanism is None else mechanism.loaded_bytes()

    def _clear(self, primary: BaseException | None = None) -> None:
        module = self._module
        owns_enrollment = self._mechanism is not None
        self._module = None
        self._mechanism = None
        self._attention_kernel = None
        self._unit_runner = None
        if not owns_enrollment:
            return
        try:
            _clear_enrollment(module)
        except BaseException as cleanup:
            if primary is None:
                raise
            primary.add_note(f"residency metadata cleanup also failed: {cleanup!r}")

    def _unload_after_failure(self, primary: BaseException) -> None:
        mechanism = self._mechanism
        if mechanism is not None:
            try:
                mechanism.unload()
            except BaseException as cleanup:
                primary.add_note(f"residency cleanup also failed: {cleanup!r}")
        self._clear(primary)

    def prepare(self) -> ReplicaPreparationRefusal | None:
        if self.state is not DeviceReplicaResidencyState.NEW:
            raise DeviceReplicaResidencyError("prepare is illegal or duplicate")
        try:
            built = _as_object(self._builder())
            if type(built) is ReplicaPreparationRefusal:
                self.state = DeviceReplicaResidencyState.REFUSED
                return built
            if type(built) is not tuple or len(built) != 3:
                raise TypeError("builder must return an exact three-item tuple or refusal")
            module, kernel, runner = built
            if not isinstance(module, torch.nn.Module):
                raise TypeError("builder module must be a torch.nn.Module")
            if not isinstance(kernel, AttentionKernel):
                raise TypeError("builder kernel must satisfy AttentionKernel")
            if not callable(runner):
                raise TypeError("builder unit runner must be callable")
            self._module = module
            self._attention_kernel = kernel
            self._unit_runner = cast(UnitRunner, runner)
            mechanism: EnrolledResidency = enroll_component(
                module,
                load_device=self.device,
                offload_device=self.offload_device,
            )
            self._mechanism = mechanism
            mechanism.partially_load(None)
            if (
                mechanism.loaded_bytes() != mechanism.total_bytes()
                or mechanism.offloaded_bytes() != 0
            ):
                raise DeviceReplicaResidencyError("component did not become fully resident")
        except BaseException as primary:
            self.state = DeviceReplicaResidencyState.FAILED
            self._unload_after_failure(primary)
            raise
        self.state = DeviceReplicaResidencyState.PREPARED
        return None

    def admit(self) -> ReplicaAdmissionRefusal | None:
        if self.state is not DeviceReplicaResidencyState.PREPARED:
            raise DeviceReplicaResidencyError("admit is illegal or duplicate")
        outcome = self._admission_gate() if self._admission_gate is not None else None
        if outcome is not None and type(outcome) is not ReplicaAdmissionRefusal:
            raise TypeError("admission gate returned an invalid outcome")
        self.state = (
            DeviceReplicaResidencyState.REFUSED
            if outcome is not None
            else DeviceReplicaResidencyState.ADMITTED
        )
        return outcome

    def activate(self) -> None:
        if self.state is not DeviceReplicaResidencyState.ADMITTED:
            raise DeviceReplicaResidencyError("activate is illegal or duplicate")
        self.state = DeviceReplicaResidencyState.ACTIVE

    def run_unit(self, unit: WorkUnitDefinition) -> ReplicaUnitFailure | None:
        if self.state is not DeviceReplicaResidencyState.ACTIVE:
            raise DeviceReplicaResidencyError("unit execution is available only while active")
        if type(unit) is not WorkUnitDefinition:
            raise TypeError("unit must be an exact WorkUnitDefinition")
        runner = self._unit_runner
        if runner is None:
            raise DeviceReplicaResidencyError("active host has no unit runner")
        outcome = runner(unit)
        if outcome is not None and type(outcome) is not ReplicaUnitFailure:
            raise TypeError("unit runner returned an invalid outcome")
        return outcome

    def cancel(self, reason: str) -> None:
        if self.state not in (
            DeviceReplicaResidencyState.PREPARED,
            DeviceReplicaResidencyState.REFUSED,
            DeviceReplicaResidencyState.ADMITTED,
            DeviceReplicaResidencyState.ACTIVE,
        ):
            raise DeviceReplicaResidencyError("cancel is illegal or duplicate")
        if type(reason) is not str or not reason:
            raise ValueError("cancel reason must be a nonempty string")
        self.state = DeviceReplicaResidencyState.CANCELLED

    def abort(self) -> None:
        if self.state not in (
            DeviceReplicaResidencyState.PREPARED,
            DeviceReplicaResidencyState.REFUSED,
            DeviceReplicaResidencyState.ADMITTED,
        ):
            raise DeviceReplicaResidencyError("abort is illegal or duplicate")
        self.state = DeviceReplicaResidencyState.ABORTED

    def release(self) -> None:
        if self.state not in (
            DeviceReplicaResidencyState.PREPARED,
            DeviceReplicaResidencyState.REFUSED,
            DeviceReplicaResidencyState.ADMITTED,
            DeviceReplicaResidencyState.ACTIVE,
            DeviceReplicaResidencyState.CANCELLED,
            DeviceReplicaResidencyState.ABORTED,
            DeviceReplicaResidencyState.FAILED,
        ):
            raise DeviceReplicaResidencyError("release is illegal or duplicate")
        if self._release_called:
            raise DeviceReplicaResidencyError("release is duplicate")
        self._release_called = True
        self.state = DeviceReplicaResidencyState.RELEASED
        mechanism = self._mechanism
        try:
            if mechanism is not None:
                mechanism.unload()
        except BaseException as primary:
            self._clear(primary)
            raise
        finally:
            if self._mechanism is not None or self._module is not None:
                self._clear()


__all__ = [
    "AdmissionGate",
    "DeviceReplicaResidencyError",
    "DeviceReplicaResidencyHost",
    "DeviceReplicaResidencyState",
    "ReplicaBuild",
    "ReplicaBuilder",
    "UnitRunner",
]
