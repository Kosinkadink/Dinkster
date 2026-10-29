"""Real ResidencyManager integration proofs for pack-owned tenants."""

from __future__ import annotations

import asyncio
import importlib
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_inference import (
    SD15,
    ReconstructionRecipe,
    RuntimeKnobs,
    WeightSourceBinding,
    WeightSourceRef,
)

ROOT = Path(__file__).parents[3]
for package_source in (ROOT / "packages").glob("*/src"):
    sys.path.insert(0, str(package_source))
compat_package = types.ModuleType("dinkster_compat_comfy")
compat_package.__path__ = [str(ROOT / "packages/dinkster-compat-comfy/src/dinkster_compat_comfy")]
sys.modules.setdefault("dinkster_compat_comfy", compat_package)

native_residency = importlib.import_module("dinkster_compat_comfy.native_residency")
tenant_registry = importlib.import_module("dinkster_compat_comfy.tenant_registry")
inference_torch = importlib.import_module("dinkster_inference_torch")
NativeResidencyCoordinator = native_residency.NativeResidencyCoordinator
NativeModelTenantRegistry = tenant_registry.NativeModelTenantRegistry
DeviceMemory = inference_torch.DeviceMemory
MemoryPolicy = inference_torch.MemoryPolicy
ResidencyManager = inference_torch.ResidencyManager

CPU = torch.device("cpu")


@dataclass
class _MemoryBox:
    free: int = 1_000

    def measure(self, _device: torch.device) -> Any:
        return DeviceMemory(self.free, 0)


@dataclass
class _Handle:
    memory: _MemoryBox
    pack: str = "pack"
    model_id: str = "model"
    size_estimate_bytes: int = 100
    device_preference: str | None = None
    fail: bool = False
    calls: list[str] | None = None

    async def offload(self) -> None:
        assert self.calls is not None
        self.calls.append("offload")
        if self.fail:
            raise RuntimeError("offload failed")
        self.memory.free += self.size_estimate_bytes

    async def evict(self) -> None:
        assert self.calls is not None
        self.calls.append("evict")
        if self.fail:
            raise RuntimeError("evict failed")
        self.memory.free += self.size_estimate_bytes


@dataclass
class _DemandPaged:
    loaded: int

    load_device: torch.device = CPU
    demand_paged: bool = True
    calls: int = 0

    def total_bytes(self) -> int:
        return self.loaded

    def loaded_bytes(self) -> int:
        return self.loaded

    def automatically_reclaimable_bytes(self) -> int:
        return self.loaded

    def offloaded_bytes(self) -> int:
        return 0

    def partially_load(self, extra_memory: int | None) -> int:
        del extra_memory
        return 0

    def partially_unload(self, memory_to_free: int) -> int:
        del memory_to_free
        self.calls += 1
        return 0

    def unload(self) -> None:
        self.calls += 1


@dataclass
class _Clock:
    now: float = 0.0

    def __call__(self) -> float:
        return self.now


def _empty_cache(_device: object) -> None:
    pass


def _total_memory(_device: object) -> int:
    return 1_000


def _device_exists(preference: str, _device: object) -> bool:
    return preference == "cpu"


def _registry(
    memory: _MemoryBox,
    *,
    clock: _Clock | None = None,
    placement_timeout: float = 60.0,
) -> tuple[Any, Any, Any]:
    manager = ResidencyManager(
        policy=MemoryPolicy(
            inference_reserve=0,
            physical_headroom=0,
            min_weight_memory_ratio=0,
            load_inflation=1,
        ),
        free_memory=memory.measure,
        empty_cache=_empty_cache,
    )
    coordinator = NativeResidencyCoordinator(manager)
    registry = NativeModelTenantRegistry(
        coordinator,
        CPU,
        total_memory=_total_memory,
        device_exists=_device_exists,
        callback_timeout=1,
        placement_timeout=placement_timeout,
        clock=clock or _Clock(),
    )
    return registry, coordinator, manager


def _scaled_fp8_component() -> tuple[torch.nn.Sequential, Any]:
    layer = inference_torch.Fp8Linear(
        4,
        3,
        bias=True,
        fp8_dtype=torch.float8_e4m3fn,
        compute_dtype=torch.float32,
    )
    layer.load_state_dict(
        {
            "weight": torch.randn(3, 4).to(torch.float8_e4m3fn),
            "weight_scale": torch.tensor(0.625),
            "input_scale": torch.tensor(1.25),
            "bias": torch.randn(3),
        },
        strict=True,
        assign=True,
    )
    return torch.nn.Sequential(layer), layer


def test_native_handle_uses_active_manager_policy_for_mps_fp8_admission(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    active_policy = MemoryPolicy(
        inference_reserve=500_000_000,
        physical_headroom=1_500_000_000,
    )
    inactive_policy = MemoryPolicy(inference_reserve=0, physical_headroom=0)
    available = active_policy.minimum_inference_memory() + 47

    def snapshot(_device: torch.device) -> Any:
        return inference_torch.MpsMemorySnapshot(
            recommended_max_bytes=4_000_000_000,
            driver_allocated_bytes=1_000_000_000,
            current_allocated_bytes=750_000_000,
            system_total_bytes=8_000_000_000,
            system_available_bytes=available,
        )

    def unexpected_snapshot(_device: torch.device) -> Any:
        raise AssertionError("coordinator enrollment must use the manager snapshot provider")

    module_residency = importlib.import_module("dinkster_inference_torch.module_residency")
    monkeypatch.setattr(module_residency, "mps_memory_snapshot", unexpected_snapshot)
    clip_l, layer = _scaled_fp8_component()
    assembled = inference_torch.AssembledSD(
        family=SD15,
        diffusion=inference_torch.INITLESS.linear(4, 4),
        clip_l=clip_l,
        clip_g=None,
        vae=inference_torch.INITLESS.linear(4, 4),
        _storage_dtype_follows_compute=False,
        _component_compute_dtypes={
            "diffusion": torch.float16,
            "clip_l": torch.float16,
            "vae": torch.float16,
        },
    )
    recipe = ReconstructionRecipe(
        sources=(
            WeightSourceBinding(
                "checkpoint",
                WeightSourceRef("blake3:" + "0" * 64, "test.safetensors", 0),
            ),
        ),
        family_id=SD15.id,
        component_identity=(f"family={SD15.id}",),
        knobs=RuntimeKnobs(
            diffusion_dtype="float16",
            text_dtype="float16",
            vae_dtype="float16",
            fp8_matmul=False,
        ),
    )
    runtime = SimpleNamespace(assembled=assembled, runtime_identity=recipe.runtime_identity)
    manager = ResidencyManager(
        policy=inactive_policy,
        policy_provider=lambda: active_policy,
        mps_snapshot=snapshot,
    )
    coordinator = NativeResidencyCoordinator(manager)

    native_residency.NativeRuntimeHandle(
        runtime,
        "mps",
        recipe=recipe,
        coordinator=coordinator,
    )

    assert "FP8 upcast needs 48 bytes" in caplog.text
    assert "2000000000-byte inference reserve" in caplog.text
    assert clip_l[0] is not layer
    assert clip_l[0].weight.dtype == torch.float32


def test_native_component_publisher_uses_active_manager_policy_for_mps_fp8_admission(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("DINKSTER_ACCELERATOR", "mps")
    active_policy = MemoryPolicy(
        inference_reserve=500_000_000,
        physical_headroom=1_500_000_000,
    )
    available = active_policy.minimum_inference_memory() + 47

    def snapshot(_device: torch.device) -> Any:
        return inference_torch.MpsMemorySnapshot(
            recommended_max_bytes=4_000_000_000,
            driver_allocated_bytes=1_000_000_000,
            current_allocated_bytes=750_000_000,
            system_total_bytes=8_000_000_000,
            system_available_bytes=available,
        )

    def unexpected_snapshot(_device: torch.device) -> Any:
        raise AssertionError("component enrollment must use the manager snapshot provider")

    module_residency = importlib.import_module("dinkster_inference_torch.module_residency")
    monkeypatch.setattr(module_residency, "mps_memory_snapshot", unexpected_snapshot)
    component, layer = _scaled_fp8_component()
    manager = ResidencyManager(
        policy=MemoryPolicy(inference_reserve=0, physical_headroom=0),
        policy_provider=lambda: active_policy,
        mps_snapshot=snapshot,
    )
    coordinator = NativeResidencyCoordinator(manager)
    mps_torch = SimpleNamespace(
        nn=torch.nn,
        device=torch.device,
        cuda=SimpleNamespace(is_available=lambda: False),
        xpu=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )
    publisher = native_residency.NativeComponentPublisher(
        coordinator=coordinator,
        _torch_module=mps_torch,
        _enroll_component=inference_torch.enroll_component,
    )

    publisher.publish(component, resource_identity="native:dinkster.qwen_image:test")

    assert "FP8 upcast needs 48 bytes" in caplog.text
    assert "2000000000-byte inference reserve" in caplog.text
    assert component[0] is not layer
    assert component[0].weight.dtype == torch.float32


def test_real_manager_orders_tenant_offload_then_evict() -> None:
    async def scenario() -> None:
        memory = _MemoryBox()
        registry, coordinator, manager = _registry(memory)
        calls: list[str] = []
        handle = _Handle(memory, calls=calls)
        registration = await registry.register(handle)
        await registry.notify_resident(registration)
        mechanism = manager.registered()[0]

        memory.free = 0
        with coordinator.placement_pass():
            manager.free(50, CPU)
        assert calls == ["offload"]
        assert mechanism.loaded_bytes() == 0
        assert mechanism in manager.registered()

        await registry.notify_resident(registration)
        memory.free = 0
        with coordinator.placement_pass():
            manager.free(200, CPU)
        assert calls == ["offload", "evict"]
        assert mechanism.loaded_bytes() == 0
        assert mechanism not in manager.registered()

    asyncio.run(scenario())


def test_aimdo_credit_skips_demand_paged_before_shedding_tenant() -> None:
    async def scenario() -> None:
        memory = _MemoryBox()
        registry, coordinator, manager = _registry(memory)
        calls: list[str] = []
        registration = await registry.register(_Handle(memory, calls=calls))
        await registry.notify_resident(registration)
        demand = _DemandPaged(80)
        manager._touch(demand)  # pyright: ignore[reportPrivateUsage]

        memory.free = 0
        with coordinator.placement_pass():
            manager.free(100, CPU, skip_demand_paged=True)
        assert demand.calls == 0
        assert calls == ["offload"]

    asyncio.run(scenario())


def test_real_manager_pass_skip_reenroll_and_failure_poisoning() -> None:
    async def scenario() -> None:
        memory = _MemoryBox()
        clock = _Clock()
        registry, coordinator, manager = _registry(memory, clock=clock, placement_timeout=5)
        calls: list[str] = []
        registration = await registry.register(_Handle(memory, calls=calls))
        await registry.notify_resident(registration)
        mechanism = manager.registered()[0]

        memory.free = 0
        with coordinator.placement_pass():
            clock.now = 6
            manager.free(200, CPU)
        assert calls == []
        assert mechanism not in manager.registered()
        with coordinator.placement_pass():
            assert mechanism in manager.registered()

        failed_calls: list[str] = []
        failed = _Handle(memory, model_id="failed", fail=True, calls=failed_calls)
        failed_registration = await registry.register(failed)
        await registry.notify_resident(failed_registration)
        failed_mechanism = manager.registered()[0]
        memory.free = 0
        with coordinator.placement_pass():
            manager.free(200, CPU)
        assert failed_calls == ["evict"]
        assert failed_mechanism not in manager.registered()
        assert registry.defective_tenants[0].registration == failed_registration
        with coordinator.placement_pass():
            assert failed_mechanism not in manager.registered()

    asyncio.run(scenario())
