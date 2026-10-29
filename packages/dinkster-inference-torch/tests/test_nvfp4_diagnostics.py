from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import dinkster_inference_torch
import dinkster_inference_torch.wiring as wiring
import pytest
import torch
from dinkster_inference_torch._nvfp4_diagnostics import (
    Nvfp4DiagnosticsRecorder,
    nvfp4_runtime_status,
)
from dinkster_inference_torch.module_residency import ModuleStateStore
from dinkster_inference_torch.quant_linear import Nvfp4Linear


def test_snapshots_are_immutable_and_bounded() -> None:
    recorder = Nvfp4DiagnosticsRecorder()
    for _ in range(20):
        with recorder.invocation():
            recorder.record("route_non_cuda")
    status = recorder.snapshot()
    assert len(status.completed) == 16
    assert status.lifetime == {"route_non_cuda": 20}
    with pytest.raises(TypeError):
        status.lifetime["route_non_cuda"] = 0  # type: ignore[index]


def test_nested_concurrent_and_unscoped_accounting() -> None:
    recorder = Nvfp4DiagnosticsRecorder()
    recorder.record("observed_loaded")

    def invoke() -> None:
        with recorder.invocation():
            with recorder.invocation():
                recorder.record("route_native")

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: invoke(), range(8)))  # pyright: ignore[reportUnknownLambdaType]
    status = recorder.snapshot()
    assert len(status.completed) == 8
    assert status.unscoped == {"observed_loaded": 1}
    assert status.lifetime["route_native"] == 8


def test_error_identity_and_context_reset() -> None:
    recorder = Nvfp4DiagnosticsRecorder()
    error = RuntimeError("must not be retained")
    with pytest.raises(RuntimeError) as caught:
        with recorder.invocation():
            raise error
    assert caught.value is error
    recorder.record("route_rank")
    status = recorder.snapshot()
    assert status.completed[-1].terminal == "error"
    assert status.unscoped == {"route_rank": 1}


def test_recorders_are_isolated_and_vocabulary_is_finite() -> None:
    first, second = Nvfp4DiagnosticsRecorder(), Nvfp4DiagnosticsRecorder()
    first.record("dequantize_success")
    assert second.snapshot().lifetime == {}
    with pytest.raises(ValueError, match="unknown"):
        first.record("a path or secret")


def test_state_changes_and_route_outcomes_are_counted_once() -> None:
    recorder = Nvfp4DiagnosticsRecorder()
    owner = object()
    recorder.observe_state(owner, "loaded")
    recorder.observe_state(owner, "offloaded")
    with recorder.invocation():
        for event in (
            "route_native",
            "route_no_quantize_backend",
            "route_no_scaled_mm_backend",
            "route_backend_fallback",
            "quantize_success",
            "quantize_error",
            "dequantize_success",
            "dequantize_error",
            "requantize_success",
            "requantize_error",
        ):
            recorder.record(event)
    status = recorder.snapshot()
    assert status.lifetime["observed_state_change"] == 1
    invocation = status.completed[-1].counters
    assert invocation["route_native"] == 1
    assert "route_full_precision" not in invocation
    assert all(
        invocation[event] == 1
        for event in (
            "route_no_quantize_backend",
            "route_no_scaled_mm_backend",
            "route_backend_fallback",
            "quantize_success",
            "quantize_error",
            "dequantize_success",
            "dequantize_error",
            "requantize_success",
            "requantize_error",
        )
    )


def test_recorder_survives_module_state_reconstruction_and_binding_isolated() -> None:
    layer = Nvfp4Linear(16, 16, bias=False, compute_dtype=torch.float32)
    recorder = Nvfp4DiagnosticsRecorder()
    layer._bind_diagnostics(recorder)  # pyright: ignore[reportPrivateUsage]
    stored = ModuleStateStore(layer)["weight"]
    assert stored.recorder is recorder  # type: ignore[union-attr]
    layer._bind_diagnostics(  # pyright: ignore[reportPrivateUsage]
        recorder
    )  # same-runtime binding is idempotent
    with pytest.raises(RuntimeError, match="another runtime"):
        layer._bind_diagnostics(  # pyright: ignore[reportPrivateUsage]
            Nvfp4DiagnosticsRecorder()
        )


def test_runtime_status_hook_remains_package_private() -> None:
    assert not hasattr(wiring, "nvfp4_runtime_status")
    assert "nvfp4_runtime_status" not in dinkster_inference_torch.__all__
    assert not hasattr(dinkster_inference_torch, "nvfp4_runtime_status")
    with pytest.raises(TypeError, match="no NVFP4"):
        nvfp4_runtime_status(object())
    recorder = Nvfp4DiagnosticsRecorder()
    runtime = SimpleNamespace(
        assembled=SimpleNamespace(diffusion=SimpleNamespace(_nvfp4_diagnostics=recorder))
    )
    assert nvfp4_runtime_status(runtime) == recorder.snapshot()
