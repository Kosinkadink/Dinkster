"""Device residency and cost meta for resident Comfy values.

Which GPU a sampler occupies is a fact of the *model value it receives*,
never of the node type (DESIGN: hazard H12). ComfyUI encodes that fact in
the ModelPatcher: ``load_device`` is where the model executes when used.
This module reads it - by duck typing, in the child process only - and
publishes it as envelope meta the engine's admission lanes and the
MemoryGovernor's budgets understand:

- ``resources``: ``{"gpu": "cuda:0"}`` (or a tuple of devices for a model
  spanning GPUs) - binds admission to concrete lanes ("gpu:cuda:0"), so
  two models on different GPUs sample in parallel while two on the same
  GPU serialize.
- ``cost``: ``{"vram:cuda:0": nbytes}`` - what *using* the model costs on
  its execution device, for reservation before allocation.

Everything is defensive: an object without patcher shape (or a CPU-only
one) simply contributes no gpu residency, and admission falls back to the
abstract "gpu" lane - conservative, never wrong.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, cast

from dinkster_memory import MeasuredMemory
from dinkster_values import COST_META_KEY, RESOURCES_META_KEY

_GPU_DEVICE_PREFIXES = ("cuda", "xpu", "mps", "npu")


def _patcher_of(obj: object) -> object | None:
    """The ModelPatcher governing this value: MODEL is one, CLIP and VAE
    carry one at ``.patcher``."""
    candidate = getattr(obj, "patcher", obj)
    if hasattr(candidate, "load_device"):
        return candidate
    return None


def _device_strings(patcher: object) -> tuple[str, ...]:
    """Concrete GPU device ids the patcher executes on. Single-device today
    (``load_device``); the tuple shape is the multigpu contract, so a
    patcher exposing multiple devices (``load_devices``) already fits."""
    raw = getattr(patcher, "load_devices", None) or [getattr(patcher, "load_device", None)]
    devices: list[str] = []
    for device in cast("list[object]", list(raw)):
        if device is None:
            continue
        text = str(device)
        if text.startswith(_GPU_DEVICE_PREFIXES):
            devices.append(text)
    return tuple(devices)


def _model_nbytes(patcher: object) -> int:
    size_fn = getattr(patcher, "model_size", None)
    if not callable(size_fn):
        return 0
    try:
        size = cast("Any", size_fn)()
    except Exception:  # noqa: BLE001 - foreign code; no meta beats a crash
        return 0
    return int(size) if isinstance(size, (int, float)) else 0


def comfy_resident_meta(obj: object) -> Mapping[str, object]:
    """ValueMeta for a resident Comfy value: residency + cost, or nothing
    when the object has no readable patcher shape."""
    resident_cost = getattr(obj, "_dinkster_resident_cost", None)
    if resident_cost is not None:
        if not isinstance(resident_cost, Mapping):
            raise TypeError("_dinkster_resident_cost must map resource names to nonnegative ints")
        raw_cost = cast("Mapping[object, object]", resident_cost)
        if any(
            type(resource) is not str or not resource or type(cost) is not int or cost < 0
            for resource, cost in raw_cost.items()
        ):
            raise TypeError("_dinkster_resident_cost must map resource names to nonnegative ints")
        return {COST_META_KEY: cast("dict[str, int]", dict(raw_cost))}
    patcher = _patcher_of(obj)
    if patcher is None:
        return {}
    devices = _device_strings(patcher)
    nbytes = _model_nbytes(patcher)
    if not devices:
        # Native CPU execution still owns the assembled weight bytes for the
        # resident's lifetime. Preserve the historical empty metadata for
        # ordinary CPU Comfy patchers; this marker belongs only to the native
        # checkpoint bundle.
        if getattr(patcher, "_dinkster_native_residency", False) and nbytes > 0:
            return {COST_META_KEY: {"ram": nbytes}}
        return {}
    meta: dict[str, object] = {
        RESOURCES_META_KEY: {"gpu": devices[0] if len(devices) == 1 else devices}
    }
    if nbytes > 0:
        # What using the model costs, split evenly across its devices -
        # the single-device case is exact, the spanning case is honest
        # accounting until patchers report per-device layouts. "ram" is
        # the offload copy: comfy keeps weights in host RAM for the
        # model's whole lifetime, GPU-loaded or not.
        share = nbytes // len(devices)
        cost: dict[str, int] = {f"vram:{device}": share for device in devices}
        cost["ram"] = nbytes
        meta[COST_META_KEY] = cost
    return meta


def torch_vram_telemetry(device: str) -> MeasuredMemory | None:
    """TelemetryProbe for ``vram:cuda:N`` and ``vram:xpu:N`` residency
    classes, backed by the device's own report (``mem_get_info`` on the
    matching torch namespace) - ground truth beside declared budgets, and
    how a non-Dinkster process hogging the GPU becomes visible. ROCm devices
    report through ``vram:cuda:N`` because the ROCm torch build exposes
    AMD GPUs as CUDA devices. None for anything unmeasurable (no torch,
    no backend, a torch without ``torch.xpu.mem_get_info``, a non-vram
    class): honest absence, never a fake zero.
    """
    if device.startswith("vram:cuda"):
        family = "cuda"
    elif device.startswith("vram:xpu"):
        family = "xpu"
    else:
        return None
    try:
        torch = cast("Any", importlib.import_module("torch"))
        namespace = getattr(torch, family, None)
        if namespace is None or not namespace.is_available():
            return None
        mem_get_info = getattr(namespace, "mem_get_info", None)
        if not callable(mem_get_info):
            return None
        free, total = cast(
            "tuple[int, int]", mem_get_info(torch.device(device.removeprefix("vram:")))
        )
        return MeasuredMemory(free_bytes=int(free), total_bytes=int(total))
    except Exception:  # noqa: BLE001 - a failing probe is an absent probe
        return None


def _structured_cuda_measurement(device: object) -> MeasuredMemory:
    inference_torch = cast("Any", importlib.import_module("dinkster_inference_torch"))
    memory = inference_torch.dynamic_cuda_memory_snapshot(device)
    return MeasuredMemory(
        free_bytes=int(memory.free_bytes),
        total_bytes=int(memory.total_bytes),
        driver_free_bytes=int(memory.driver_free_bytes),
        allocator_reclaimable_bytes=int(memory.allocator_reclaimable_bytes),
        dynamic_evictable_bytes=int(memory.dynamic_evictable_bytes),
        dynamic_pinned_bytes=int(memory.dynamic_pinned_bytes),
    )


def _structured_xpu_measurement(device: object) -> MeasuredMemory:
    inference_torch = cast("Any", importlib.import_module("dinkster_inference_torch"))
    memory = inference_torch.xpu_memory_snapshot(device)
    if not memory.driver_reported:
        # Allocator-derived free cannot see other processes' usage;
        # measured telemetry must come from the driver's own report.
        raise RuntimeError("XPU snapshot free bytes were not driver-reported")
    return MeasuredMemory(
        free_bytes=int(memory.free_bytes),
        total_bytes=int(memory.total_bytes),
        driver_free_bytes=int(memory.driver_free_bytes),
        allocator_reclaimable_bytes=int(memory.allocator_reclaimable_bytes),
    )


def _family_telemetry_snapshot(torch: Any, family: str) -> dict[str, MeasuredMemory]:
    try:
        namespace = getattr(torch, family, None)
        if namespace is None or not namespace.is_available():
            return {}
        count = int(namespace.device_count())
    except Exception:  # noqa: BLE001 - a failing probe is an absent probe
        return {}
    structured = _structured_cuda_measurement if family == "cuda" else _structured_xpu_measurement
    snapshot: dict[str, MeasuredMemory] = {}
    for index in range(count):
        measured: MeasuredMemory | None = None
        try:
            measured = structured(torch.device(family, index))
        except Exception:  # noqa: BLE001 - a failing structured probe falls back to the driver
            measured = torch_vram_telemetry(f"vram:{family}:{index}")
        if measured is not None:
            snapshot[f"vram:{family}:{index}"] = measured
    return snapshot


def vram_telemetry_snapshot() -> Mapping[str, MeasuredMemory]:
    """Measured accelerator capacity and attribution for every visible
    CUDA and XPU device.

    Structured CUDA snapshots add torch allocator reserve and only
    unpinned, evictable Dinkster VBAR pages to driver free, capped at
    physical total. Pinned pages remain visible as attribution and never
    become free. Structured XPU snapshots carry driver free and allocator
    reserve attribution. This probe is the only derivation point; parent
    telemetry passes it through without changing admission.

    ROCm devices appear under ``vram:cuda:N`` because the ROCm torch
    build exposes AMD GPUs through the CUDA namespace; their family
    identity lives in accelerator and route evidence, not here. CPU-only
    workers and MPS return an empty mapping - honest absence, never a
    fake zero. Defensive throughout: no torch, no backend, or a failing
    driver all measure nothing rather than fail the worker.
    """
    try:
        torch = cast("Any", importlib.import_module("torch"))
    except Exception:  # noqa: BLE001 - a failing probe is an absent probe
        return {}
    snapshot: dict[str, MeasuredMemory] = {}
    snapshot.update(_family_telemetry_snapshot(torch, "cuda"))
    snapshot.update(_family_telemetry_snapshot(torch, "xpu"))
    return snapshot
