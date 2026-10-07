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
import os
import sys
from collections.abc import Callable, Mapping
from functools import cache
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


def _optional_counter(read: Callable[[], object]) -> int | None:
    try:
        value = read()
        return (
            int(value)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
            else None
        )
    except Exception:  # noqa: BLE001 - one unavailable metric must not hide the others
        return None


@cache
def _nvml_device(uuid: str) -> tuple[Any, Any]:
    nvml = cast("Any", importlib.import_module("pynvml"))
    nvml.nvmlInit()
    return nvml, nvml.nvmlDeviceGetHandleByUUID(uuid)


def _nvml_telemetry(properties: Any) -> dict[str, int]:
    """Match the physical GPU UUID; CUDA_VISIBLE_DEVICES remaps torch indices."""
    try:
        uuid = str(properties.uuid)
        if not uuid.startswith("GPU-"):
            uuid = f"GPU-{uuid}"
        nvml, handle = _nvml_device(uuid)
    except Exception:  # noqa: BLE001 - optional vendor telemetry
        return {}
    readings = {
        "driver_free_bytes": lambda: nvml.nvmlDeviceGetMemoryInfo(handle).free,
        "gpu_utilization_percent": lambda: nvml.nvmlDeviceGetUtilizationRates(handle).gpu,
        "gpu_temperature_celsius": lambda: nvml.nvmlDeviceGetTemperature(
            handle, nvml.NVML_TEMPERATURE_GPU
        ),
        "gpu_power_milliwatts": lambda: nvml.nvmlDeviceGetPowerUsage(handle),
        "gpu_power_limit_milliwatts": lambda: nvml.nvmlDeviceGetPowerManagementLimit(handle),
    }
    return {
        name: value
        for name, read in readings.items()
        if (value := _optional_counter(read)) is not None
    }


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
        torch_device = torch.device(device.removeprefix("vram:"))
        free, total = cast("tuple[int, int]", mem_get_info(torch_device))
        hardware: dict[str, int] = {}
        name: str | None = None
        try:
            properties = namespace.get_device_properties(torch_device)
            name = str(properties.name)
            if family == "cuda" and not getattr(torch.version, "hip", None):
                hardware = _nvml_telemetry(properties)
        except Exception:  # noqa: BLE001 - basic capacity remains useful without device details
            pass
        manager_module = sys.modules.get("dinkster_inference.model_management")
        pinned = _optional_counter(
            lambda: cast("Any", manager_module).get_model_manager().total_pinned_memory
        )
        return MeasuredMemory(
            free_bytes=int(free),
            total_bytes=int(total),
            torch_allocated_bytes=_optional_counter(
                lambda: namespace.memory_allocated(torch_device)
            ),
            torch_reserved_bytes=_optional_counter(lambda: namespace.memory_reserved(torch_device)),
            process_id=os.getpid(),
            process_count=1,
            process_rss_bytes=_optional_counter(
                lambda: cast("Any", importlib.import_module("psutil")).Process().memory_info().rss
            ),
            pinned_host_bytes=pinned,
            gpu_name=name,
            **hardware,
        )
    except Exception:  # noqa: BLE001 - a failing probe is an absent probe
        return None


def _family_telemetry_snapshot(torch: Any, family: str) -> dict[str, MeasuredMemory]:
    try:
        namespace = getattr(torch, family, None)
        if namespace is None or not namespace.is_available():
            return {}
        count = int(namespace.device_count())
    except Exception:  # noqa: BLE001 - a failing probe is an absent probe
        return {}
    snapshot: dict[str, MeasuredMemory] = {}
    for index in range(count):
        measured = torch_vram_telemetry(f"vram:{family}:{index}")
        if measured is not None:
            snapshot[f"vram:{family}:{index}"] = measured
    return snapshot


def vram_telemetry_snapshot() -> Mapping[str, MeasuredMemory]:
    """Measured accelerator capacity and attribution for every visible
    CUDA and XPU device.

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
