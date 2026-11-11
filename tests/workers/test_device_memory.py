"""Worker memory telemetry follows physical GPU identity across remapping."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

VRAM0 = "vram:cuda:0"


def test_native_usage_probe_measures_worker_and_matches_nvml_by_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_workers.device_memory import _nvml_device, torch_vram_telemetry

    _nvml_device.cache_clear()
    handles: list[str] = []
    namespace = SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=lambda _: (7000, 10000),
        memory_allocated=lambda _: 3000,
        memory_reserved=lambda _: 5000,
        get_device_properties=lambda _: SimpleNamespace(
            name="Remapped GPU", uuid="physical-second"
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=namespace, device=lambda value: value, version=SimpleNamespace(hip=None)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "pynvml",
        SimpleNamespace(
            nvmlInit=lambda: None,
            nvmlDeviceGetHandleByUUID=lambda uuid: handles.append(uuid) or "second",
            nvmlDeviceGetMemoryInfo=lambda _: SimpleNamespace(free=6000),
            nvmlDeviceGetUtilizationRates=lambda _: SimpleNamespace(gpu=73),
            NVML_TEMPERATURE_GPU=0,
            nvmlDeviceGetTemperature=lambda *_: 61,
            nvmlDeviceGetPowerUsage=lambda _: 121000,
            nvmlDeviceGetPowerManagementLimit=lambda _: 180000,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(
            Process=lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=12345))
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "dinkster_inference.model_management",
        SimpleNamespace(TOTAL_PINNED_MEMORY=2000),
    )
    measured = torch_vram_telemetry(VRAM0)
    assert measured is not None
    assert handles == ["GPU-physical-second"]
    assert measured.to_wire() == {
        "freeBytes": 7000,
        "totalBytes": 10000,
        "driverFreeBytes": 6000,
        "torchAllocatedBytes": 3000,
        "torchReservedBytes": 5000,
        "processId": os.getpid(),
        "processCount": 1,
        "processRssBytes": 12345,
        "pinnedHostBytes": 2000,
        "gpuName": "Remapped GPU",
        "gpuUtilizationPercent": 73,
        "gpuTemperatureCelsius": 61,
        "gpuPowerMilliwatts": 121000,
        "gpuPowerLimitMilliwatts": 180000,
    }
    assert torch_vram_telemetry(VRAM0) == measured
    assert handles == ["GPU-physical-second"]
    _nvml_device.cache_clear()
    monkeypatch.delitem(sys.modules, "pynvml")
    without_vendor = torch_vram_telemetry(VRAM0)
    assert without_vendor is not None and without_vendor.torch_allocated_bytes == 3000
    assert without_vendor.gpu_utilization_percent is None
