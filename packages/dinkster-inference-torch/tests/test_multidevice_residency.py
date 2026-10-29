from __future__ import annotations

import inspect
from typing import Any, cast

import pytest
import torch
from dinkster_inference.replica_lane import (
    ReplicaAdmissionRefusal,
    ReplicaPreparationRefusal,
    ReplicaUnitFailure,
)
from dinkster_inference_torch.module_residency import enroll_component
from dinkster_inference_torch.multidevice_attention import MultiDeviceAttentionKernel
from dinkster_inference_torch.multidevice_residency import (
    DeviceReplicaResidencyError,
    DeviceReplicaResidencyHost,
    DeviceReplicaResidencyState,
)
from dinkster_inference_torch.operations import CastOperations
from dinkster_inference_torch.residency import ResidentWeights
from dinkster_protocol import ReplicaId, SemanticSlot, WorkUnitDefinition, WorkUnitId

CPU0 = torch.device("cpu", 0)
CPU1 = torch.device("cpu", 1)
CPU = torch.device("cpu")


class RecordingKernel:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]] = []

    def __call__(
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
        self.calls.append(
            (
                q,
                k,
                v,
                {
                    "mask": mask,
                    "causal": causal,
                    "scale": scale,
                    "enable_gqa": enable_gqa,
                },
            )
        )
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)


def _unit() -> WorkUnitDefinition:
    return WorkUnitDefinition(WorkUnitId("unit"), ReplicaId("replica"), SemanticSlot.SINGLE)


def _no_op_runner(unit: WorkUnitDefinition) -> None:
    del unit


def _module() -> torch.nn.Module:
    module = CastOperations(torch.float32).linear(3, 3)
    with torch.no_grad():
        module.weight.fill_(1)
        assert module.bias is not None
        module.bias.zero_()
    return module


def _builder(
    *,
    runner_result: ReplicaUnitFailure | None = None,
) -> tuple[
    DeviceReplicaResidencyHost,
    list[torch.nn.Module],
    RecordingKernel,
    list[WorkUnitDefinition],
]:
    modules: list[torch.nn.Module] = []
    kernel = RecordingKernel()
    runs: list[WorkUnitDefinition] = []

    def build():  # noqa: ANN202 - exact builder inference is under test
        module = _module()
        modules.append(module)

        def run(unit: WorkUnitDefinition) -> ReplicaUnitFailure | None:
            runs.append(unit)
            return runner_result

        return module, kernel, run

    return DeviceReplicaResidencyHost(CPU0, CPU, build), modules, kernel, runs


def _activate(host: DeviceReplicaResidencyHost) -> None:
    assert host.prepare() is None
    assert host.admit() is None
    host.activate()


def test_explicit_device_and_type_validation_precedes_builder_work() -> None:
    calls = 0

    def builder() -> ReplicaPreparationRefusal:
        nonlocal calls
        calls += 1
        return ReplicaPreparationRefusal("unused")

    for device in (CPU, torch.device("meta", 0), "cpu:0"):
        with pytest.raises((TypeError, ValueError)):
            DeviceReplicaResidencyHost(cast(Any, device), CPU, builder)
    with pytest.raises(TypeError):
        DeviceReplicaResidencyHost(CPU0, cast(Any, "cpu"), builder)
    with pytest.raises(TypeError):
        DeviceReplicaResidencyHost(CPU0, CPU, cast(Any, object()))
    assert calls == 0


def test_import_and_construction_do_not_probe_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(*_args: object, **_kwargs: object) -> Any:
        raise AssertionError("GPU probing is forbidden")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected)
    monkeypatch.setattr(torch.cuda, "device_count", unexpected)
    import dinkster_inference_torch.multidevice_residency as module

    module.DeviceReplicaResidencyHost(CPU0, CPU, lambda: ReplicaPreparationRefusal("no"))
    assert "torch.cuda" not in inspect.getsource(module)


def test_full_residency_persists_across_run_and_cancel_until_release() -> None:
    host, modules, _kernel, runs = _builder()
    _activate(host)
    module = modules[0]
    mechanism = module.__dict__["_dinkster_resident_weights"]
    assert isinstance(mechanism, ResidentWeights)
    assert mechanism.loaded_bytes() == mechanism.total_bytes() > 0
    resident = host.resident_bytes
    assert host.run_unit(_unit()) is None
    assert runs == [_unit()]
    assert host.resident_bytes == resident
    host.cancel("stop")
    assert host.state is DeviceReplicaResidencyState.CANCELLED
    assert host.resident_bytes == resident
    host.release()
    assert host.state is DeviceReplicaResidencyState.RELEASED
    assert host.resident_bytes == 0
    assert "_dinkster_resident_weights" not in module.__dict__
    assert all("_residency" not in owner.__dict__ for owner in module.modules())
    with pytest.raises(DeviceReplicaResidencyError):
        host.release()


def test_typed_preparation_and_admission_refusals_retain_no_stale_access() -> None:
    refused = DeviceReplicaResidencyHost(
        CPU0, CPU, lambda: ReplicaPreparationRefusal("builder refused")
    )
    assert refused.prepare() == ReplicaPreparationRefusal("builder refused")
    assert refused.resident_bytes == 0
    with pytest.raises(DeviceReplicaResidencyError):
        _ = refused.attention_kernel
    refused.abort()
    refused.release()

    host, modules, _kernel, _runs = _builder()
    host = DeviceReplicaResidencyHost(
        CPU0,
        CPU,
        lambda: (_module(), RecordingKernel(), _no_op_runner),
        admission_gate=lambda: ReplicaAdmissionRefusal("admission refused"),
    )
    assert host.prepare() is None
    resident = host.resident_bytes
    assert host.admit() == ReplicaAdmissionRefusal("admission refused")
    assert host.resident_bytes == resident > 0
    host.release()
    assert host.resident_bytes == 0
    assert modules == []


def test_attention_is_active_only_and_preserves_exact_call_surface() -> None:
    host, _modules, kernel, _runs = _builder()
    with pytest.raises(DeviceReplicaResidencyError):
        _ = host.attention_kernel
    assert host.prepare() is None
    assert host.admit() is None
    with pytest.raises(DeviceReplicaResidencyError):
        _ = host.attention_kernel
    host.activate()
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 5, 4)
    v = torch.randn(1, 2, 5, 6)
    output = host.attention_kernel(q, k, v)
    cached_kernel = host.attention_kernel
    assert output.shape == (1, 2, 3, 6)
    assert kernel.calls[0][3] == {
        "mask": None,
        "causal": False,
        "scale": None,
        "enable_gqa": False,
    }
    host.cancel("stop")
    with pytest.raises(DeviceReplicaResidencyError):
        _ = host.attention_kernel
    with pytest.raises(DeviceReplicaResidencyError):
        cached_kernel(q, k, v)
    host.release()
    with pytest.raises(DeviceReplicaResidencyError):
        cached_kernel(q, k, v)


def test_structural_cpu_multidevice_attention_composition_uses_active_kernel() -> None:
    host, _modules, kernel, _runs = _builder()
    _activate(host)
    q = torch.randn(1, 4, 3, 4)
    k = torch.randn(1, 4, 5, 4)
    v = torch.randn(1, 4, 5, 6)
    with MultiDeviceAttentionKernel(host.attention_kernel, (CPU0, CPU1)) as multi:
        output = multi(q, k, v)
    assert output.shape == (1, 4, 3, 6)
    assert len(kernel.calls) == 2
    assert {call[0].device.type for call in kernel.calls} == {"cpu"}
    host.release()


def test_unit_failure_and_run_exception_preserve_residency_until_release() -> None:
    host, _modules, _kernel, _runs = _builder(runner_result=ReplicaUnitFailure("failed"))
    _activate(host)
    resident = host.resident_bytes
    assert host.run_unit(_unit()) == ReplicaUnitFailure("failed")
    assert host.resident_bytes == resident
    host.release()

    def build():  # noqa: ANN202 - exact builder inference is under test
        def fail(_unit: WorkUnitDefinition) -> None:
            raise RuntimeError("run failed")

        return _module(), RecordingKernel(), fail

    throwing = DeviceReplicaResidencyHost(CPU0, CPU, build)
    _activate(throwing)
    resident = throwing.resident_bytes
    with pytest.raises(RuntimeError, match="run failed"):
        throwing.run_unit(_unit())
    assert throwing.resident_bytes == resident
    throwing.release()


class _FailingResidentWeights(ResidentWeights):
    def __init__(
        self,
        *,
        load_error: BaseException | None,
        unload_error: BaseException | None,
    ) -> None:
        self.load_error = load_error
        self.unload_error = unload_error
        self.unload_calls = 0

    def partially_load(self, extra_memory: int | None) -> int:
        del extra_memory
        if self.load_error is not None:
            raise self.load_error
        return 1

    def loaded_bytes(self) -> int:
        return 1

    def total_bytes(self) -> int:
        return 1

    def offloaded_bytes(self) -> int:
        return 0

    def unload(self) -> None:
        self.unload_calls += 1
        if self.unload_error is not None:
            raise self.unload_error


class _CleanupFailModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fail_cleanup = False

    def modules(self, remove_duplicate: bool = True):  # noqa: ANN201 - torch's iterator annotation is inferred
        if self.fail_cleanup:
            raise RuntimeError("metadata cleanup failed")
        return super().modules(remove_duplicate=remove_duplicate)


def test_build_enrollment_and_load_failures_clear_every_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def build_failure():  # noqa: ANN202 - injected failure
        raise RuntimeError("build failed")

    host = DeviceReplicaResidencyHost(CPU0, CPU, build_failure)
    with pytest.raises(RuntimeError, match="build failed"):
        host.prepare()
    assert host.state is DeviceReplicaResidencyState.FAILED
    assert host.resident_bytes == 0

    import dinkster_inference_torch.multidevice_residency as module

    def enrollment_failure(*_args: object, **_kwargs: object) -> ResidentWeights:
        raise RuntimeError("enrollment failed")

    monkeypatch.setattr(module, "enroll_component", enrollment_failure)
    enrolled = DeviceReplicaResidencyHost(
        CPU0, CPU, lambda: (_module(), RecordingKernel(), _no_op_runner)
    )
    with pytest.raises(RuntimeError, match="enrollment failed"):
        enrolled.prepare()
    assert enrolled.resident_bytes == 0

    primary = RuntimeError("load failed")
    mechanism = _FailingResidentWeights(
        load_error=primary, unload_error=RuntimeError("cleanup failed")
    )

    def failed_load_enrollment(*args: object, **kwargs: object) -> ResidentWeights:
        del args, kwargs
        return mechanism

    monkeypatch.setattr(module, "enroll_component", failed_load_enrollment)
    loading = DeviceReplicaResidencyHost(
        CPU0, CPU, lambda: (_module(), RecordingKernel(), _no_op_runner)
    )
    with pytest.raises(RuntimeError, match="load failed") as caught:
        loading.prepare()
    assert caught.value is primary
    assert caught.value.__notes__ == [
        "residency cleanup also failed: RuntimeError('cleanup failed')"
    ]
    assert mechanism.unload_calls == 1
    assert loading.resident_bytes == 0


def test_failed_enrollment_preserves_foreign_residency_metadata() -> None:
    owned = _module()
    existing = enroll_component(owned, load_device=CPU0, offload_device=CPU)
    bindings = {
        owner: owner.__dict__["_residency"]
        for owner in owned.modules()
        if "_residency" in owner.__dict__
    }
    host = DeviceReplicaResidencyHost(CPU0, CPU, lambda: (owned, RecordingKernel(), _no_op_runner))

    with pytest.raises(RuntimeError, match="already enrolled"):
        host.prepare()

    assert owned.__dict__["_dinkster_resident_weights"] is existing
    assert all(owner.__dict__.get("_residency") is binding for owner, binding in bindings.items())
    existing.unload()
    owned.__dict__.pop("_dinkster_resident_weights")
    for owner in bindings:
        owner.__dict__.pop("_residency")


def test_release_failure_is_terminal_clears_references_and_never_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_inference_torch.multidevice_residency as module

    mechanism = _FailingResidentWeights(
        load_error=None, unload_error=RuntimeError("release failed")
    )

    def failed_release_enrollment(*args: object, **kwargs: object) -> ResidentWeights:
        del args, kwargs
        return mechanism

    monkeypatch.setattr(module, "enroll_component", failed_release_enrollment)
    host = DeviceReplicaResidencyHost(
        CPU0, CPU, lambda: (_module(), RecordingKernel(), _no_op_runner)
    )
    assert host.prepare() is None
    with pytest.raises(RuntimeError, match="release failed"):
        host.release()
    assert host.state is DeviceReplicaResidencyState.RELEASED
    assert host.resident_bytes == 0
    assert mechanism.unload_calls == 1
    with pytest.raises(DeviceReplicaResidencyError):
        host.release()
    assert mechanism.unload_calls == 1


def test_metadata_cleanup_failure_clears_references_and_preserves_unload_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_inference_torch.multidevice_residency as module

    mechanism = _FailingResidentWeights(
        load_error=None, unload_error=RuntimeError("unload primary")
    )

    def enrollment(*args: object, **kwargs: object) -> ResidentWeights:
        del args, kwargs
        return mechanism

    monkeypatch.setattr(module, "enroll_component", enrollment)
    owned = _CleanupFailModule()
    host = DeviceReplicaResidencyHost(CPU0, CPU, lambda: (owned, RecordingKernel(), _no_op_runner))
    assert host.prepare() is None
    owned.fail_cleanup = True
    with pytest.raises(RuntimeError, match="unload primary") as caught:
        host.release()
    assert caught.value.__notes__ == [
        "residency metadata cleanup also failed: RuntimeError('metadata cleanup failed')"
    ]
    assert host.state is DeviceReplicaResidencyState.RELEASED
    assert host.resident_bytes == 0
    assert mechanism.unload_calls == 1
    with pytest.raises(DeviceReplicaResidencyError):
        host.release()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two visible CUDA devices")
def test_cuda_two_replica_residencies_dispatch_attention_and_release() -> None:
    devices = (torch.device("cuda", 0), torch.device("cuda", 1))
    modules: list[torch.nn.Module] = []
    hosts: list[DeviceReplicaResidencyHost] = []

    for device in devices:
        kernel = RecordingKernel()

        def build(kernel: RecordingKernel = kernel):
            module = _module()
            modules.append(module)
            return module, kernel, _no_op_runner

        host = DeviceReplicaResidencyHost(device, CPU, build)
        _activate(host)
        hosts.append(host)

    resident = tuple(host.resident_bytes for host in hosts)
    assert all(size > 0 for size in resident)
    assert tuple(next(module.parameters()).device for module in modules) == devices
    by_device = dict(zip(devices, hosts, strict=True))

    def dispatch(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        return by_device[q.device].attention_kernel(
            q,
            k,
            v,
            mask=mask,
            causal=causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )

    q = torch.randn(1, 4, 3, 4, device=devices[0])
    k = torch.randn(1, 4, 5, 4, device=devices[0])
    v = torch.randn(1, 4, 5, 6, device=devices[0])
    with MultiDeviceAttentionKernel(dispatch, devices) as multi:
        output = multi(q, k, v)
    assert output.shape == (1, 4, 3, 6)
    assert tuple(host.run_unit(_unit()) for host in hosts) == (None, None)

    for host, size in zip(hosts, resident, strict=True):
        host.cancel("settled")
        assert host.resident_bytes == size
        host.release()
        assert host.resident_bytes == 0
    assert all("_dinkster_resident_weights" not in module.__dict__ for module in modules)
