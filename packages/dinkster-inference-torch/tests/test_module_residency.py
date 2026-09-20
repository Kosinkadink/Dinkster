"""Post-assembly module-state residency and routed operator access."""

from __future__ import annotations

import asyncio
import gc
import importlib
import weakref
from collections.abc import Generator, MutableMapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import dinkster_inference_torch.residency as residency_mod
import pytest
import torch
from dinkster_inference import (
    QWEN_IMAGE,
    SD15,
    WAN21,
    LoRASpec,
    PatchTarget,
    decode_lora,
    load_safetensors_header,
    native_unet_key_map,
)
from dinkster_inference.patches import (
    AdapterPatch,
    DiffPatch,
    ModelAsLoraPatch,
    NestedPatch,
    PatchEntry,
    PatchOffset,
    PatchSet,
    SetPatch,
    patch_payloads,
)
from dinkster_inference_torch import (
    INITLESS,
    STORAGE_DTYPE_POLICY_TOKENS,
    AssembledMiniMaxH3Model,
    AssembledQwenImage,
    AssembledSD,
    AssembledWan21,
    AutoencoderKL,
    BOFTAdapter,
    CastOperations,
    ClipTextModel,
    Fp8Linear,
    Fp8ScaledWeight,
    GLoRAAdapter,
    Int8Embedding,
    Int8Linear,
    LoHaAdapter,
    LoKrAdapter,
    LoRAAdapter,
    MemoryPolicy,
    ModuleStateStore,
    MpsMemorySnapshot,
    OFTAdapter,
    Operations,
    PartialResidencyTiming,
    PatchApplyError,
    ResidencyRouted,
    ResidencyUnit,
    ResidentWeights,
    StorageDtypePolicyError,
    StoredWeight,
    T5LayerNorm,
    UNetModel,
    apply_patches,
    build_patch_set,
    cast_weight,
    collect_partial_residency_timing,
    declare_residency_materialization_ceilings,
    declare_residency_unit,
    detach_residency_enrollment,
    enroll_assembled,
    enroll_component,
    load_tensors,
)
from dinkster_inference_torch import module_residency as module_residency_mod
from dinkster_inference_torch import quant_linear as quant_linear_mod
from dinkster_inference_torch._nvfp4_diagnostics import Nvfp4DiagnosticsRecorder
from dinkster_inference_torch.module_residency import ResidencyBinding
from dinkster_inference_torch.operations import (
    bound_compute_device,
    materialized_conv2d_parameters,
    module_compute_device,
)
from dinkster_inference_torch.quant import Int8PackedWeight, Nvfp4PackedWeight, requantize_int8
from dinkster_inference_torch.quant_linear import Nvfp4Linear
from dinkster_inference_torch.residency import WeightLease
from dinkster_inference_torch.rounding import string_to_seed

CPU = torch.device("cpu")


def _double_patch_value(value: torch.Tensor, /, *, inplace: bool = False) -> torch.Tensor:
    return value.mul_(2) if inplace else value * 2


def _fill(module: torch.nn.Module, dtype: torch.dtype = torch.float32) -> None:
    generator = torch.Generator().manual_seed(73)
    state = {
        key: torch.randn(tuple(value.shape), generator=generator).to(dtype)
        for key, value in module.state_dict().items()
    }
    module.load_state_dict(state, strict=True, assign=True)


def _fp8_layer(
    *,
    bias: bool = True,
    compute_dtype: torch.dtype = torch.float32,
) -> Fp8Linear:
    layer = Fp8Linear(
        4,
        3,
        bias=bias,
        fp8_dtype=torch.float8_e4m3fn,
        compute_dtype=compute_dtype,
    )
    generator = torch.Generator().manual_seed(19)
    state = {
        "weight": torch.randn(3, 4, generator=generator).to(torch.float8_e4m3fn),
        "weight_scale": torch.tensor(0.625),
        "input_scale": torch.tensor(1.25),
    }
    if bias:
        state["bias"] = torch.randn(3, generator=generator)
    layer.load_state_dict(state, strict=True, assign=True)
    return layer


def _apply_mps_fp8_upcast(module: torch.nn.Module) -> None:
    plan = module_residency_mod._plan_fp8_upcast_for_mps(  # pyright: ignore[reportPrivateUsage]
        module
    )
    module_residency_mod._commit_fp8_upcasts(  # pyright: ignore[reportPrivateUsage]
        (plan,)
    )


def _nvfp4_layer() -> Nvfp4Linear:
    layer = Nvfp4Linear(
        16,
        16,
        bias=True,
        compute_dtype=torch.float32,
        pre_quant_scale=True,
    )
    kitchen = quant_linear_mod._require_nvfp4_kitchen()  # pyright: ignore[reportPrivateUsage]
    weight = torch.linspace(-1.0, 1.0, 256).reshape(16, 16)
    scale = torch.tensor(weight.abs().amax().item() / (448.0 * 6.0), dtype=torch.float32)
    qweight, block = kitchen.quantize(weight, scale)
    layer.load_state_dict(
        {
            "weight": qweight,
            "weight_scale": block,
            "weight_scale_2": scale,
            "input_scale": torch.tensor(0.25),
            "pre_quant_scale": torch.linspace(0.5, 1.5, 16),
            "bias": torch.linspace(-0.1, 0.1, 16),
        },
        strict=True,
        assign=True,
    )
    return layer


def _int8_layer() -> Int8Linear:
    layer = Int8Linear(
        256,
        7,
        bias=True,
        compute_dtype=torch.float32,
        convrot=True,
        convrot_groupsize=256,
    )
    generator = torch.Generator().manual_seed(23)
    layer.load_state_dict(
        {
            "weight": torch.randint(-100, 101, (7, 256), generator=generator, dtype=torch.int8),
            "weight_scale": torch.rand((7, 1), generator=generator) / 100,
            "bias": torch.randn(7, generator=generator),
        },
        strict=True,
        assign=True,
    )
    return layer


def _int8_embedding() -> Int8Embedding:
    layer = Int8Embedding(
        11,
        256,
        compute_dtype=torch.float32,
        convrot=True,
        convrot_groupsize=256,
    )
    generator = torch.Generator().manual_seed(29)
    layer.load_state_dict(
        {
            "weight": torch.randint(-100, 101, (11, 256), generator=generator, dtype=torch.int8),
            "weight_scale": torch.rand((11, 1), generator=generator) / 100,
        },
        strict=True,
        assign=True,
    )
    return layer


def test_store_reports_route_owned_materialization_ceilings() -> None:
    initless = INITLESS.linear(8, 4)
    _fill(initless, torch.bfloat16)
    cast_at_use = CastOperations(torch.float32).linear(8, 4)
    _fill(cast_at_use, torch.bfloat16)
    declare_residency_materialization_ceilings(initless, {"weight": 2, "bias": 2})
    declare_residency_materialization_ceilings(cast_at_use, {"weight": 4, "bias": 4})
    root = torch.nn.Sequential(_int8_layer(), torch.nn.Linear(8, 4), initless, cast_at_use)
    store = ModuleStateStore(root)

    assert store.max_materialized_itemsize("0.weight") == 4
    assert store.max_materialized_itemsize("0.bias") == 4
    assert store.max_materialized_itemsize("1.weight") == 4
    assert store.max_materialized_itemsize("1.bias") == 4
    assert store.max_materialized_itemsize("2.weight") == 2
    assert store.max_materialized_itemsize("2.bias") == 2
    assert store.max_materialized_itemsize("3.weight") == 4
    assert store.max_materialized_itemsize("3.bias") == 4
    assert store.uses_raw_residency("2.weight")
    assert store.uses_raw_residency("2.bias")
    assert not store.uses_raw_residency("3.weight")
    assert not store.uses_raw_residency("3.bias")
    with pytest.raises(KeyError, match="missing"):
        store.max_materialized_itemsize("missing")


def test_store_rejects_state_wider_than_declared_materialization_ceiling() -> None:
    module = INITLESS.linear(2, 2, bias=False)
    _fill(module, torch.bfloat16)
    declare_residency_materialization_ceilings(module, {"weight": 2})
    store = ModuleStateStore(module)

    with pytest.raises(ValueError, match="uses 4 bytes.*ceiling is 2"):
        store["weight"] = torch.ones(2, 2, dtype=torch.float32)


def test_store_reports_frozen_cast_embedding_as_raw_residency() -> None:
    embedding = CastOperations(torch.float32).embedding(8, 4)
    _fill(embedding, torch.float16)
    store = ModuleStateStore(embedding)

    assert not store.uses_raw_residency("weight")
    embedding.requires_grad_(False)
    assert store.uses_raw_residency("weight")


def _operation_cases(
    operations: Operations,
) -> list[tuple[torch.nn.Module, torch.Tensor]]:
    generator = torch.Generator().manual_seed(31)
    return [
        (operations.linear(6, 4), torch.randn(3, 6, generator=generator)),
        (
            operations.conv2d(2, 3, 3, padding=1),
            torch.randn(1, 2, 7, 7, generator=generator),
        ),
        (
            operations.conv1d(4, 6, 3, stride=2, padding=2, dilation=2, groups=2),
            torch.randn(2, 4, 13, generator=generator),
        ),
        (
            operations.conv_transpose1d(
                4,
                6,
                3,
                stride=2,
                padding=1,
                output_padding=1,
                groups=2,
                dilation=2,
            ),
            torch.randn(2, 4, 7, generator=generator),
        ),
        (
            operations.conv3d(
                4,
                6,
                (2, 3, 3),
                stride=(1, 2, 1),
                padding=(1, 1, 2),
                dilation=(1, 2, 1),
                groups=2,
            ),
            torch.randn(2, 4, 5, 9, 8, generator=generator),
        ),
        (
            operations.group_norm(8, num_groups=4),
            torch.randn(2, 8, 5, 5, generator=generator),
        ),
        (operations.layer_norm(16), torch.randn(4, 16, generator=generator)),
        (operations.embedding(10, 8), torch.arange(6).reshape(2, 3)),
        (
            operations.rms_norm(16, eps=1e-6),
            torch.randn(4, 16, generator=generator),
        ),
    ]


class _StateToy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)
        self.register_buffer("scale", torch.tensor(2.0))


class _UnitToy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = INITLESS.linear(4, 4)
        self.nested = torch.nn.Sequential(INITLESS.linear(4, 4, bias=False))
        _fill(self)


class _GroupedUnit(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = INITLESS.linear(4, 3, bias=False)
        self.up = INITLESS.linear(4, 3, bias=False)
        self.down = INITLESS.linear(3, 4, bias=False)
        declare_residency_unit(self)


class _GroupedUnitToy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = _GroupedUnit()
        self.second = _GroupedUnit()
        _fill(self)


class _TiedToy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = INITLESS.linear(4, 4, bias=False)
        self.second = INITLESS.linear(4, 4, bias=False)
        weight = torch.nn.Parameter(torch.randn(4, 4), requires_grad=False)
        self.first.weight = weight
        self.second.weight = weight


class _MixedComputeToy(ResidencyRouted, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.modulation = torch.nn.Parameter(torch.empty(4), requires_grad=False)
        self.patch = CastOperations(torch.float32).linear(4, 4, bias=False)
        self.body = CastOperations(torch.bfloat16).linear(4, 4, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is None:
            modulation = cast_weight(self.modulation, dtype=value.dtype)
        else:
            with binding.lease() as lease:
                modulation = lease.get("modulation", dtype=value.dtype)
        return self.body(self.patch(value.float()).to(value.dtype)) + modulation


def _mixed_compute_toy() -> _MixedComputeToy:
    module = _MixedComputeToy()
    generator = torch.Generator().manual_seed(79)
    state = {
        key: torch.randn(tuple(value.shape), generator=generator).to(
            torch.float32 if key.startswith("patch.") else torch.float16
        )
        for key, value in module.state_dict().items()
    }
    module.load_state_dict(state, strict=True, assign=True)
    return module


class _ConflictingTargetTiedToy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = CastOperations(torch.float16).linear(4, 4, bias=False)
        self.second = CastOperations(torch.bfloat16).linear(4, 4, bias=False)
        weight = torch.nn.Parameter(torch.randn(4, 4), requires_grad=False)
        self.first.weight = weight
        self.second.weight = weight


class _MixedT5(torch.nn.Module):
    def __init__(self, operations: Operations) -> None:
        super().__init__()
        self.before = operations.linear(4, 4, bias=False)
        self.norm = T5LayerNorm(4)
        self.after = operations.linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.after(self.norm(self.before(x)))


class _ExternalResidencyUnit(ResidencyRouted, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gain = torch.nn.Parameter(torch.empty(2, dtype=torch.float16), requires_grad=False)
        self.register_buffer("offset", torch.empty(2), persistent=True)
        self.projection = INITLESS.linear(2, 2, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        offset = cast(torch.Tensor, self.offset)
        binding = self._offloaded_residency()
        if binding is None:
            owned = value * self.gain + offset
        else:
            with binding.lease() as lease:
                owned = value * lease.get("gain", dtype=self.gain.dtype) + lease.get(
                    "offset", dtype=offset.dtype
                )
        return self.projection(owned)


def _filled_external_residency_unit() -> _ExternalResidencyUnit:
    module = _ExternalResidencyUnit()
    generator = torch.Generator().manual_seed(73)
    state = {
        key: torch.randn(tuple(value.shape), generator=generator).to(value.dtype)
        for key, value in module.state_dict().items()
    }
    module.load_state_dict(state, strict=True, assign=True)
    return module


def test_store_is_a_live_view_and_restores_exact_parameter() -> None:
    module = _StateToy()
    store = ModuleStateStore(module)
    original = module.linear.weight
    original_scale = cast(torch.Tensor, module.scale)
    assert store["linear.weight"] is original
    assert store["scale"] is original_scale

    replacement = torch.randn_like(original)
    store["linear.weight"] = replacement
    assert isinstance(module.linear.weight, torch.nn.Parameter)
    assert module.linear.weight is store["linear.weight"]
    assert not module.linear.weight.requires_grad
    assert torch.equal(module.linear.weight, replacement)

    replacement_scale = torch.tensor(3.0)
    store["scale"] = replacement_scale
    assert module.scale is replacement_scale
    store["scale"] = original_scale
    assert module.scale is original_scale

    store["linear.weight"] = original
    assert module.linear.weight is original
    with pytest.raises(TypeError, match="cannot be deleted"):
        del store["linear.weight"]


def test_store_folds_fp8_weight_and_scale() -> None:
    layer = _fp8_layer()
    store = ModuleStateStore(layer)
    assert set(store) == {"weight", "input_scale", "bias"}
    assert "weight_scale" not in store
    original_weight = layer.weight
    original_scale = layer.weight_scale
    stored = store["weight"]
    assert isinstance(stored, Fp8ScaledWeight)
    assert store["weight"] is stored
    assert stored.qdata is original_weight
    assert stored.scale is original_scale

    replacement = Fp8ScaledWeight(
        stored.qdata.clone(),
        torch.tensor(0.25),
        stored.orig_dtype,
    )
    store["weight"] = replacement
    assert store["weight"] is store["weight"]
    assert store["weight"] is not stored
    assert isinstance(layer.weight, torch.nn.Parameter)
    assert torch.equal(layer.weight.view(torch.uint8), replacement.qdata.view(torch.uint8))
    assert layer.weight_scale is replacement.scale

    store["weight"] = stored
    assert layer.weight is original_weight
    assert layer.weight_scale is original_scale


@pytest.mark.parametrize("prefix", ["", "0"], ids=["empty", "nonempty"])
def test_layer_lease_joins_prefix_like_binding_key(prefix: str) -> None:
    root: torch.nn.Module
    if prefix:
        root = torch.nn.Sequential(INITLESS.linear(2, 2, bias=False))
    else:
        root = INITLESS.linear(2, 2, bias=False)
    _fill(root)
    store = ModuleStateStore(root)
    mechanism = ResidentWeights(store, load_device=CPU, offload_device=CPU)
    binding = ResidencyBinding(mechanism, prefix or "weight", prefix, store)
    expected_key = "weight" if not prefix else f"{prefix}.weight"
    assert binding.key("weight") == expected_key

    expected = mechanism.use(expected_key, dtype=torch.float32)
    with binding.lease() as lease:
        actual = lease.get("weight", dtype=torch.float32)
        stored = lease.get_stored("weight")
    assert torch.equal(actual, expected)
    assert stored is store[expected_key]


def test_lease_transfers_record_partial_residency_timing() -> None:
    """Transfer receipts live at the lease level, so every leased
    consumer's copies are counted, not just encoded GGUF layers."""
    layer = INITLESS.linear(2, 2, bias=False)
    _fill(layer)
    store = ModuleStateStore(layer)
    mechanism = ResidentWeights(store, load_device=CPU, offload_device=CPU)
    binding = ResidencyBinding(mechanism, "weight", "", store)
    expected = mechanism.use("weight", dtype=torch.float16)
    stored = store["weight"]
    assert isinstance(stored, torch.Tensor)
    stored_bytes = stored.nbytes

    with collect_partial_residency_timing() as timing:
        with binding.lease() as lease:
            assert lease.timing_collector() is timing
            actual = lease.get("weight", dtype=torch.float16)
            # The lease caches materialized keys: a repeated get is
            # not a second transfer.
            assert lease.get("weight", dtype=torch.float16) is actual
    report = timing.report()

    assert torch.equal(actual, expected)
    assert report.leased_transfers == 1
    # Transfer bytes count the stored representation that crossed
    # devices (float32 here), not the cast result the consumer sees.
    assert report.transfer_bytes == stored_bytes
    assert report.transfer_bytes != actual.nbytes
    assert report.transfer_ms > 0.0
    assert 0.0 < report.exposed_stall_ms <= report.transfer_ms
    # The cast-at-use work is the dequant phase, not transfer time.
    assert report.dequant_ms > 0.0
    assert report.leased_forwards == 0
    assert report.compute_ms == 0.0

    with binding.lease() as lease:
        assert lease.timing_collector() is None


def test_binding_lease_reads_the_timing_collector_on_entry_not_construction() -> None:
    """The collection window is defined by when the lease bracket is
    entered: a lease constructed before the window but entered inside
    it records into the active collector, and one constructed inside
    the window but entered after it records nothing."""
    layer = INITLESS.linear(2, 2, bias=False)
    _fill(layer)
    store = ModuleStateStore(layer)
    mechanism = ResidentWeights(store, load_device=CPU, offload_device=CPU)
    binding = ResidencyBinding(mechanism, "weight", "", store)

    constructed_outside = binding.lease()
    with collect_partial_residency_timing() as timing:
        with constructed_outside as lease:
            assert lease.timing_collector() is timing
            lease.get("weight", dtype=torch.float16)
    assert timing.report().leased_transfers == 1

    with collect_partial_residency_timing() as ended:
        constructed_inside = binding.lease()
    with constructed_inside as lease:
        assert lease.timing_collector() is None
        lease.get("weight", dtype=torch.float16)
    assert ended.report().leased_transfers == 0


def test_source_pins_absent_for_cpu_load_device() -> None:
    layer = INITLESS.linear(2, 2, bias=False)
    _fill(layer)
    store = ModuleStateStore(layer)
    mechanism = ResidentWeights(store, load_device=CPU, offload_device=CPU)
    assert mechanism._source_pins is None  # pyright: ignore[reportPrivateUsage]


def test_stored_source_pins_lifecycle_accounting(monkeypatch: pytest.MonkeyPatch) -> None:
    """In-place source pins register eligible component tensors once,
    account registered bytes, shed under registration pressure after a
    device synchronize, and empty the owner registry on release_all."""
    registered: dict[int, int] = {}
    syncs: list[object] = []

    def fake_register(ptr: int, size: int) -> bool:
        registered[ptr] = size
        return True

    def fake_unregister(ptr: int) -> bool:
        registered.pop(ptr)
        return True

    def fake_discard(device: torch.device) -> None:
        return None

    def fake_synchronize(device: torch.device | None = None) -> None:
        syncs.append(device)

    def fake_budget(size: int) -> bool:
        return True

    def fake_registerable(size: int, *, evict_active: bool = True) -> bool:
        return True

    monkeypatch.setattr(residency_mod, "_cuda_host_register", fake_register)
    monkeypatch.setattr(residency_mod, "_cuda_host_unregister", fake_unregister)
    monkeypatch.setattr(residency_mod, "_discard_cuda_async_error", fake_discard)
    monkeypatch.setattr(torch.cuda, "synchronize", fake_synchronize)
    monkeypatch.setattr(residency_mod.pinned_host, "ensure_pin_budget", fake_budget)
    monkeypatch.setattr(residency_mod.pinned_host, "ensure_pin_registerable", fake_registerable)
    pinned_host = residency_mod.pinned_host

    pins = residency_mod._StoredSourcePins(torch.device("cuda", 0))  # pyright: ignore[reportPrivateUsage]
    initial = pinned_host.TOTAL_PINNED_MEMORY

    plain = torch.randn(4, 4)
    pins.ensure(plain)
    assert registered == {plain.data_ptr(): plain.nbytes}
    assert pinned_host.TOTAL_PINNED_MEMORY - initial == plain.nbytes
    pins.ensure(plain)
    assert pinned_host.TOTAL_PINNED_MEMORY - initial == plain.nbytes

    packed = Fp8ScaledWeight(
        torch.zeros(4, 4, dtype=torch.float8_e4m3fn),
        torch.tensor(0.5),
        torch.float16,
    )
    pins.ensure(packed)
    pinned = plain.nbytes + packed.qdata.nbytes + packed.scale.nbytes
    assert pinned_host.TOTAL_PINNED_MEMORY - initial == pinned

    # Empty and non-contiguous tensors stay pageable.
    pins.ensure(torch.empty(0))
    pins.ensure(torch.randn(4, 6).T)
    assert pinned_host.TOTAL_PINNED_MEMORY - initial == pinned

    # Registration pressure sheds pins after one synchronize.
    assert not syncs
    freed = pins.free_registrations(1)
    assert freed == plain.nbytes
    assert len(syncs) == 1
    assert plain.data_ptr() not in registered
    assert pinned_host.TOTAL_PINNED_MEMORY - initial == pinned - freed

    # Releasing an already-shed pin is a no-op.
    pins.release(plain)
    assert len(syncs) == 1

    pins.release_all()
    assert registered == {}
    assert pinned_host.TOTAL_PINNED_MEMORY == initial
    assert pins not in pinned_host._owners  # pyright: ignore[reportPrivateUsage]


def _fake_stored_source_pins(
    monkeypatch: pytest.MonkeyPatch,
    tensor: torch.Tensor,
) -> tuple[Any, dict[int, int], list[object], int]:
    registered: dict[int, int] = {}
    syncs: list[object] = []

    def register(ptr: int, size: int) -> bool:
        registered[ptr] = size
        return True

    def unregister(ptr: int) -> bool:
        return registered.pop(ptr, None) is not None

    def synchronize(device: torch.device | None = None) -> None:
        syncs.append(device)

    def registerable(_size: int, *, evict_active: bool = True) -> bool:
        return True

    monkeypatch.setattr(residency_mod, "_cuda_host_register", register)
    monkeypatch.setattr(residency_mod, "_cuda_host_unregister", unregister)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(residency_mod.pinned_host, "ensure_pin_budget", registerable)
    monkeypatch.setattr(residency_mod.pinned_host, "ensure_pin_registerable", registerable)
    monkeypatch.setattr(residency_mod.pinned_host, "_owners", [])
    pins = residency_mod._StoredSourcePins(torch.device("cuda", 0))  # pyright: ignore[reportPrivateUsage]
    initial = residency_mod.pinned_host.TOTAL_PINNED_MEMORY
    pins.ensure(tensor)
    return pins, registered, syncs, initial


def test_stored_source_pins_refuse_explicit_active_eviction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = torch.randn(4, 4)
    pins, registered, syncs, initial = _fake_stored_source_pins(monkeypatch, source)
    pinned_host = residency_mod.pinned_host

    pins.acquire_active()
    try:
        assert not pinned_host.free_registrations(source.nbytes, evict_active=True)
        assert registered == {source.data_ptr(): source.nbytes}
        assert syncs == []
        assert pinned_host.TOTAL_PINNED_MEMORY == initial + source.nbytes
    finally:
        pins.release_active()

    assert pinned_host.free_registrations(source.nbytes, evict_active=True)
    assert registered == {}
    assert syncs == [torch.device("cuda", 0)]
    assert pinned_host.TOTAL_PINNED_MEMORY == initial
    pins.release_all()


def test_stored_source_pins_recheck_activity_after_inactive_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = torch.randn(4, 4)
    pins, registered, syncs, initial = _fake_stored_source_pins(monkeypatch, source)
    pinned_host = residency_mod.pinned_host
    free_registrations = pins.free_registrations

    def activate_after_selection(size: int) -> int:
        pins.acquire_active()
        return free_registrations(size)

    monkeypatch.setattr(pins, "free_registrations", activate_after_selection)
    assert not pinned_host.free_registrations(source.nbytes, evict_active=False)
    assert pins.pin_active
    assert registered == {source.data_ptr(): source.nbytes}
    assert syncs == []
    assert pinned_host.TOTAL_PINNED_MEMORY == initial + source.nbytes

    pins.release_active()
    assert free_registrations(source.nbytes) == source.nbytes
    pins.release_all()
    assert registered == {}
    assert syncs == [torch.device("cuda", 0)]
    assert pinned_host.TOTAL_PINNED_MEMORY == initial


def test_mechanism_routes_offloaded_reads_through_source_pins() -> None:
    """Offloaded consumer reads pin the stored source first, brackets
    and prefetch handles mark the pins active, loading a unit releases
    its pins, and unload releases everything."""

    class RecorderPins:
        def __init__(self) -> None:
            self.ensured: list[object] = []
            self.released: list[object] = []
            self.released_all = 0
            self.active = 0

        def ensure(self, stored: object) -> None:
            self.ensured.append(stored)

        def release(self, stored: object) -> None:
            self.released.append(stored)

        def release_all(self) -> None:
            self.released_all += 1

        def acquire_active(self) -> None:
            self.active += 1

        def release_active(self) -> None:
            self.active -= 1

    layer = INITLESS.linear(2, 2, bias=False)
    _fill(layer)
    store = ModuleStateStore(layer)
    mechanism = ResidentWeights(store, load_device=CPU, offload_device=CPU)
    recorder = RecorderPins()
    mechanism._source_pins = cast(Any, recorder)  # pyright: ignore[reportPrivateUsage]
    original = store["weight"]

    with mechanism.lease("weight") as lease:
        assert recorder.active == 1
        lease.get("weight", dtype=torch.float16)
    assert recorder.active == 0
    assert recorder.ensured == [original]

    handle = mechanism.prefetch((("weight", None),))
    assert handle is not None
    assert recorder.active == 1
    handle.close()
    assert recorder.active == 0
    assert recorder.ensured == [original, original]

    mechanism.partially_load(None)
    assert recorder.released == [original]

    # Loaded keys skip pinning.
    mechanism.use("weight", dtype=torch.float16)
    assert recorder.ensured == [original, original]

    mechanism.unload()
    assert recorder.released_all == 1


def test_prefetch_owns_source_pin_activity_during_staging_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PrefetchAbort(BaseException):
        pass

    class AbortingPrefetchCache(dict[object, object]):
        def update(self, *args: Any, **kwargs: Any) -> None:
            values = dict(*args, **kwargs)
            request, value = next(iter(values.items()))
            self[request] = value
            raise PrefetchAbort

    class RecorderPins:
        def __init__(self) -> None:
            self.active = 0
            self.ensure_calls = 0
            self.abort_at: int | None = None

        def ensure(self, _stored: object) -> None:
            assert self.active == 1
            self.ensure_calls += 1
            if self.ensure_calls == self.abort_at:
                raise PrefetchAbort

        def release(self, _stored: object) -> None:
            pass

        def release_all(self) -> None:
            pass

        def acquire_active(self) -> None:
            self.active += 1

        def release_active(self) -> None:
            self.active -= 1

    layer = INITLESS.linear(2, 2, bias=True)
    _fill(layer)
    store = ModuleStateStore(layer)
    mechanism = ResidentWeights(store, load_device=CPU, offload_device=CPU)
    recorder = RecorderPins()
    mechanism._source_pins = cast(Any, recorder)  # pyright: ignore[reportPrivateUsage]

    handle = mechanism.prefetch((("weight", None),))
    assert handle is not None
    assert recorder.active == 1
    handle.close()
    handle.close()
    assert recorder.active == 0
    assert mechanism._prefetched == {}  # pyright: ignore[reportPrivateUsage]

    recorder.ensure_calls = 0
    recorder.abort_at = 2
    with pytest.raises(PrefetchAbort):
        mechanism.prefetch((("weight", None), ("bias", None)))
    assert recorder.active == 0
    assert mechanism._prefetched == {}  # pyright: ignore[reportPrivateUsage]

    recorder.ensure_calls = 0
    recorder.abort_at = None
    mechanism._prefetched = cast(Any, AbortingPrefetchCache())  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(PrefetchAbort):
        mechanism.prefetch((("weight", None), ("bias", None)))
    assert recorder.active == 0
    assert mechanism._prefetched == {}  # pyright: ignore[reportPrivateUsage]

    mechanism._prefetched = {}  # pyright: ignore[reportPrivateUsage]
    handle = mechanism.prefetch((("weight", None),))
    assert handle is not None

    def abort_wait() -> None:
        raise PrefetchAbort

    monkeypatch.setattr(mechanism._transfer_hooks, "producer_wait_for_consumer", abort_wait)  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(PrefetchAbort):
        handle.close()
    assert recorder.active == 0
    assert mechanism._prefetched == {}  # pyright: ignore[reportPrivateUsage]
    handle.close()
    assert recorder.active == 0


def test_mechanism_prefetch_stages_offloaded_weights_for_leases() -> None:
    """A consuming lease adopts the staged value instead of starting a
    second copy: the receipt attributes the move to prefetch and the
    lease records only its wait."""
    layer = INITLESS.linear(2, 2, bias=False)
    _fill(layer)
    store = ModuleStateStore(layer)
    mechanism = ResidentWeights(store, load_device=CPU, offload_device=CPU)
    expected = mechanism.use("weight", dtype=torch.float16)
    stored = store["weight"]
    assert isinstance(stored, torch.Tensor)

    with collect_partial_residency_timing() as timing:
        handle = mechanism.prefetch((("weight", torch.float16),))
        assert handle is not None
        with mechanism.lease("weight") as lease:
            actual = lease.get("weight", dtype=torch.float16)
            # The adopted value is cached like a lease's own transfer:
            # a repeated get is the same tensor.
            assert lease.get("weight", dtype=torch.float16) is actual
        handle.close()
    report = timing.report()

    assert torch.equal(actual, expected)
    assert report.prefetched_transfers == 1
    assert report.prefetch_bytes == stored.nbytes
    assert report.transfer_bytes == stored.nbytes
    assert report.leased_transfers == 0
    assert report.transfer_ms > 0.0
    assert report.dequant_ms > 0.0
    request = ("weight", torch.float16)
    assert mechanism._peek_prefetched(request) is None  # pyright: ignore[reportPrivateUsage]


def test_mechanism_prefetch_values_match_unstaged_use() -> None:
    """Uncollected staging goes through the same fused use calls the
    mechanism serves directly, so staged values are bit-identical for
    both the cast and the stored-representation request kinds."""
    layer = INITLESS.linear(2, 2, bias=False)
    _fill(layer)
    store = ModuleStateStore(layer)
    mechanism = ResidentWeights(store, load_device=CPU, offload_device=CPU)
    expected_cast = mechanism.use("weight", dtype=torch.float16)
    expected_stored = mechanism.use_stored("weight")
    assert isinstance(expected_stored, torch.Tensor)

    handle = mechanism.prefetch((("weight", torch.float16), ("weight", None)))
    assert handle is not None
    with mechanism.lease("weight") as lease:
        assert torch.equal(lease.get("weight", dtype=torch.float16), expected_cast)
        stored = lease.get_stored("weight")
        assert isinstance(stored, torch.Tensor)
        assert torch.equal(stored, expected_stored)
    handle.close()


def test_mechanism_prefetch_skips_loaded_duplicate_and_staged_requests() -> None:
    layer = INITLESS.linear(2, 2, bias=False)
    _fill(layer)
    store = ModuleStateStore(layer)
    mechanism = ResidentWeights(store, load_device=CPU, offload_device=CPU)

    request = ("weight", None)
    # A duplicate within one call is staged once; a request already
    # staged by an open handle is not staged again.
    handle = mechanism.prefetch((request, request))
    assert handle is not None
    assert mechanism.prefetch((request,)) is None
    handle.close()
    assert mechanism._peek_prefetched(request) is None  # pyright: ignore[reportPrivateUsage]

    # A unit loaded between prefetch and lease is served from the
    # loaded weights; closing the handle still releases the stash.
    handle = mechanism.prefetch((request,))
    assert handle is not None
    mechanism.partially_load(None)
    with mechanism.lease("weight") as lease:
        stored = lease.get_stored("weight")
        assert isinstance(stored, torch.Tensor)
    handle.close()
    assert mechanism._peek_prefetched(request) is None  # pyright: ignore[reportPrivateUsage]

    # Loaded keys are never staged.
    assert mechanism.prefetch((request,)) is None


def test_units_follow_direct_state_owners() -> None:
    module = _UnitToy()
    mechanism = enroll_component(module, load_device=CPU, offload_device=CPU)
    mechanism.partially_load(None)
    assert mechanism.loaded_unit_names() == frozenset({"first", "nested.0"})


def test_declared_units_group_descendant_state_owners() -> None:
    module = _GroupedUnitToy()
    mechanism = enroll_component(module, load_device=CPU, offload_device=CPU)
    mechanism.partially_load(None)
    assert mechanism.loaded_unit_names() == frozenset({"first", "second"})
    for name, group in (("first", module.first), ("second", module.second)):
        bindings = [
            cast(ResidencyRouted, layer).residency_binding()
            for layer in (group.gate, group.up, group.down)
        ]
        assert all(binding is not None for binding in bindings)
        assert {binding.unit for binding in bindings if binding is not None} == {name}


def test_declared_expert_marker_reaches_the_residency_unit() -> None:
    module = _GroupedUnitToy()
    declare_residency_unit(module.first, expert=True)
    mechanism = enroll_component(module, load_device=CPU, offload_device=CPU)

    units = {unit.name: unit for unit in mechanism.residency_units()}
    assert units["first"].expert is True
    assert units["second"].expert is False
    assert mechanism.unit_bytes("first") == module.first.gate.weight.nbytes * 3
    with pytest.raises(KeyError):
        mechanism.unit_bytes("missing")


def test_declared_units_reject_overlapping_nested_groups() -> None:
    module = _GroupedUnitToy()
    declare_residency_unit(module)
    with pytest.raises(ValueError, match="overlap"):
        enroll_component(module, load_device=CPU, offload_device=CPU)


def test_declared_units_reject_tied_state_crossing_group_boundaries() -> None:
    module = _GroupedUnitToy()
    module.second.gate.weight = module.first.gate.weight
    with pytest.raises(ValueError, match="tied state outside"):
        enroll_component(module, load_device=CPU, offload_device=CPU)


def test_units_merge_tied_storage_across_modules() -> None:
    module = _TiedToy()
    mechanism = enroll_component(module, load_device=CPU, offload_device=CPU)
    mechanism.partially_load(None)
    assert mechanism.loaded_unit_names() == frozenset({"first"})


def test_double_enrollment_raises() -> None:
    module = INITLESS.linear(4, 4)
    enroll_component(module, load_device=CPU, offload_device=CPU)
    with pytest.raises(RuntimeError, match="already enrolled"):
        enroll_component(module, load_device=CPU, offload_device=CPU)


def test_residency_assignment_evidence_requires_current_authorized_tensor() -> None:
    module = INITLESS.linear(2, 2, bias=False)
    _fill(module)
    original = module.weight
    generation = module_residency_mod._residency_assignment_generation  # pyright: ignore[reportPrivateUsage]
    version = module_residency_mod._residency_assignment_version  # pyright: ignore[reportPrivateUsage]
    assert generation(module, "weight", original) is None
    assert version(module, "weight", original) is None

    mechanism = enroll_component(
        module,
        load_device=CPU,
        offload_device=CPU,
        patch_set=PatchSet({"weight": (PatchEntry(DiffPatch(torch.ones_like(module.weight))),)}),
    )
    assert generation(module, "weight", original) is None
    assert version(module, "weight", original) is None

    mechanism.partially_load(None)
    loaded = module.weight
    loaded_generation = generation(module, "weight", loaded)
    assert loaded is not original
    assert loaded_generation is not None
    assert version(module, "weight", loaded) == loaded._version

    mechanism.unload()
    restored = module.weight
    restored_generation = generation(module, "weight", restored)
    assert restored is original
    assert restored_generation is not None
    assert restored_generation > loaded_generation
    assert version(module, "weight", restored) == restored._version
    assert generation(module, "weight", loaded) is None
    assert version(module, "weight", loaded) is None

    direct = torch.nn.Parameter(restored.detach().clone())
    module.weight = direct
    assert generation(module, "weight", direct) is None
    assert version(module, "weight", direct) is None

    ordinary = torch.nn.Parameter(direct.detach().clone())
    ModuleStateStore(module)["weight"] = ordinary
    assert generation(module, "weight", ordinary) is None
    assert version(module, "weight", ordinary) is None


def test_residency_assignment_generation_rejects_changed_storage() -> None:
    module = INITLESS.linear(2, 2, bias=False)
    _fill(module)
    mechanism = enroll_component(module, load_device=CPU, offload_device=CPU)
    mechanism.partially_load(None)
    current = module.weight
    query = module_residency_mod._residency_assignment_generation  # pyright: ignore[reportPrivateUsage]
    assert query(module, "weight", current) is not None

    current.data = current.detach().clone()
    assert query(module, "weight", current) is None


def test_residency_assignment_accepts_inference_tensor_without_version_counter() -> None:
    module = INITLESS.linear(2, 2, bias=False)
    _fill(module)
    mechanism = enroll_component(
        module,
        load_device=CPU,
        offload_device=CPU,
        patch_set=PatchSet({"weight": (PatchEntry(DiffPatch(torch.ones_like(module.weight))),)}),
    )

    with torch.inference_mode():
        mechanism.partially_load(None)
    assigned = module.weight

    assert assigned.is_inference()
    assert (
        module_residency_mod._residency_assignment_generation(  # pyright: ignore[reportPrivateUsage]
            module, "weight", assigned
        )
        == 1
    )
    assert (
        module_residency_mod._residency_assignment_version(  # pyright: ignore[reportPrivateUsage]
            module, "weight", assigned
        )
        is None
    )


def test_bound_compute_device_is_none_for_unbound_factory_layer() -> None:
    assert bound_compute_device(INITLESS.embedding(8, 4)) is None


def test_bound_compute_device_is_load_device_for_loaded_unit() -> None:
    layer = INITLESS.embedding(8, 4)
    _fill(layer)
    mechanism = enroll_component(layer, load_device=CPU, offload_device=CPU)
    mechanism.partially_load(None)
    assert mechanism.loaded_bytes() == mechanism.total_bytes()
    assert bound_compute_device(layer) == CPU


def test_bound_compute_device_is_load_device_for_offloaded_unit() -> None:
    layer = INITLESS.embedding(8, 4)
    _fill(layer)
    mechanism = enroll_component(layer, load_device=CPU, offload_device=CPU)
    mechanism.partially_load(0)
    assert mechanism.loaded_bytes() == 0
    assert bound_compute_device(layer) == CPU


def test_module_compute_device_falls_back_to_parameter_device() -> None:
    module = torch.nn.Linear(2, 2, device="meta")
    assert module_compute_device(module) == torch.device("meta")


def test_module_compute_device_prefers_root_residency_load_device() -> None:
    module = torch.nn.Linear(2, 2, device="meta")
    module.__dict__["_dinkster_resident_weights"] = SimpleNamespace(load_device=CPU)
    assert module_compute_device(module) == CPU


def test_module_compute_device_prefers_bound_layer_over_root_residency() -> None:
    class BoundModule(ResidencyRouted, torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(1, device="meta"))

        def residency_binding(self) -> Any:
            return SimpleNamespace(mechanism=SimpleNamespace(load_device=CPU))

    module = BoundModule()
    module.__dict__["_dinkster_resident_weights"] = SimpleNamespace(
        load_device=torch.device("meta")
    )
    assert module_compute_device(module) == CPU


def test_enrollment_rejects_a_state_owner_without_routed_access() -> None:
    with pytest.raises(TypeError, match="owns state but has no residency route"):
        enroll_component(_StateToy(), load_device=CPU, offload_device=CPU)


def test_two_dimensional_operation_routes_preserve_unload_reload_forward() -> None:
    class Routed2d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = INITLESS.conv2d(
                4,
                4,
                (3, 2),
                padding=(1, 1),
                groups=2,
                padding_mode="replicate",
            )
            self.norm = INITLESS.batch_norm2d(4)
            self.up = INITLESS.conv_transpose2d(4, 2, 2, stride=2)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.up(self.norm(self.conv(value)))

    module = Routed2d().eval()
    generator = torch.Generator().manual_seed(121)
    state = {
        key: (
            torch.randint(0, 10, value.shape, generator=generator)
            if not value.is_floating_point()
            else torch.randn(value.shape, generator=generator)
        )
        for key, value in module.state_dict().items()
    }
    state["norm.running_var"] = state["norm.running_var"].abs().add(0.5)
    module.load_state_dict(state, assign=True)
    input_value = torch.randn(2, 4, 7, 9, generator=generator)
    expected = module(input_value)

    mechanism = enroll_component(module, load_device=CPU, offload_device=CPU)
    mechanism.partially_load(0)
    offloaded = module(input_value)
    mechanism.partially_load(None)
    loaded = module(input_value)
    mechanism.unload()
    unloaded = module(input_value)

    assert torch.equal(offloaded, expected)
    assert torch.equal(loaded, expected)
    assert torch.equal(unloaded, expected)


def test_enroll_assembled_uses_component_field_names_and_skips_none() -> None:
    assembled = AssembledSD(
        family=SD15,
        diffusion=cast("UNetModel", INITLESS.linear(2, 2)),
        clip_l=cast("ClipTextModel", INITLESS.linear(2, 2)),
        clip_g=None,
        vae=cast("AutoencoderKL", INITLESS.linear(2, 2)),
    )
    mechanisms = enroll_assembled(assembled, load_device=CPU, offload_device=CPU)
    assert set(mechanisms) == {"diffusion", "clip_l", "vae"}
    assert len({id(mechanism) for mechanism in mechanisms.values()}) == 3


def test_enroll_assembled_qwen_image_uses_runtime_component_names() -> None:
    assembled = AssembledQwenImage(
        family=QWEN_IMAGE,
        diffusion=cast("Any", INITLESS.linear(2, 2)),
        text=cast("Any", INITLESS.linear(2, 2)),
        vae=cast("Any", INITLESS.linear(2, 2)),
        _component_compute_dtypes={
            "diffusion": torch.bfloat16,
            "text": torch.bfloat16,
            "vae": torch.bfloat16,
        },
    )

    mechanisms = enroll_assembled(assembled, load_device=CPU, offload_device=CPU)

    assert set(mechanisms) == {"diffusion", "text", "vae"}


def test_enroll_assembled_wan21_uses_all_native_components() -> None:
    diffusion = INITLESS.linear(2, 2)
    cast("Any", diffusion).config = SimpleNamespace(model_type="i2v")
    assembled = AssembledWan21(
        family=WAN21,
        diffusion=cast("Any", diffusion),
        umt5xxl=cast("Any", INITLESS.linear(2, 2)),
        vae=cast("Any", INITLESS.linear(2, 2)),
        tokenizer_model=b"sentencepiece",
        clip_vision=cast("Any", INITLESS.linear(2, 2)),
    )
    mechanisms = enroll_assembled(assembled, load_device=CPU, offload_device=CPU)
    assert set(mechanisms) == {"diffusion", "umt5xxl", "clip_vision", "vae"}
    assert len({id(mechanism) for mechanism in mechanisms.values()}) == 4


@pytest.mark.parametrize(
    ("model_type", "clip_vision"),
    (("t2v", object()), ("i2v", None)),
)
def test_assembled_wan21_requires_exact_profile_vision_pairing(
    model_type: str, clip_vision: object | None
) -> None:
    with pytest.raises(ValueError, match="I2V requires CLIP vision and T2V must not carry it"):
        AssembledWan21(
            family=WAN21,
            diffusion=cast("Any", SimpleNamespace(config=SimpleNamespace(model_type=model_type))),
            umt5xxl=cast("Any", object()),
            vae=cast("Any", object()),
            tokenizer_model=b"sentencepiece",
            clip_vision=cast("Any", clip_vision),
        )


def test_enroll_assembled_binds_int8_component_residency() -> None:
    diffusion = _int8_layer()
    assembled = AssembledSD(
        family=SD15,
        diffusion=cast("UNetModel", diffusion),
        clip_l=cast("ClipTextModel", INITLESS.linear(2, 2)),
        clip_g=None,
        vae=cast("AutoencoderKL", INITLESS.linear(2, 2)),
    )
    mechanisms = enroll_assembled(assembled, load_device=CPU, offload_device=CPU)
    assert diffusion._residency is not None  # pyright: ignore[reportPrivateUsage]
    assert diffusion._residency.mechanism is mechanisms["diffusion"]  # pyright: ignore[reportPrivateUsage]


def test_terminal_detachment_releases_nested_module_without_cyclic_gc() -> None:
    class NestedModule(ResidencyRouted, torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.root_weight = torch.nn.Parameter(torch.ones(2))
            self.child = INITLESS.linear(2, 2)

    module = NestedModule()
    child = module.child
    root_tensor = module.root_weight
    child_tensor = child.weight
    module_ref = weakref.ref(module)
    child_ref = weakref.ref(child)
    root_tensor_ref = weakref.ref(root_tensor)
    child_tensor_ref = weakref.ref(child_tensor)
    mechanism = enroll_component(module, load_device=CPU, offload_device=CPU)

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        detach_residency_enrollment(module, mechanism)
        del mechanism, root_tensor, child_tensor, child, module
        assert module_ref() is None
        assert child_ref() is None
        assert root_tensor_ref() is None
        assert child_tensor_ref() is None
    finally:
        if was_enabled:
            gc.enable()


@pytest.mark.parametrize("discard_on_release", [False, True])
def test_component_terminal_release_preserves_base_and_dies_without_gc(
    discard_on_release: bool,
) -> None:
    native = importlib.import_module("dinkster_compat_comfy.native_residency")

    class NestedModule(ResidencyRouted, torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.root_weight = torch.nn.Parameter(torch.ones(2))
            self.child = INITLESS.linear(2, 2)

    base = NestedModule()
    _fill(base)
    base_parameters = dict(base.named_parameters())
    base_ptrs = {name: tensor.data_ptr() for name, tensor in base_parameters.items()}
    base_values = {name: tensor.detach().clone() for name, tensor in base_parameters.items()}
    manager = residency_mod.ResidencyManager()
    coordinator = native.NativeResidencyCoordinator(manager)

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        module = NestedModule()
        module.load_state_dict(base.state_dict(), assign=True)
        assert module.root_weight is not base.root_weight
        assert module.root_weight.data_ptr() == base.root_weight.data_ptr()
        assert module.child.weight.data_ptr() == base.child.weight.data_ptr()
        mechanism = enroll_component(
            module,
            load_device=CPU,
            offload_device=CPU,
            patch_set=PatchSet(
                {
                    "root_weight": (PatchEntry(DiffPatch(torch.ones(2))),),
                    "child.weight": (PatchEntry(DiffPatch(torch.ones(2, 2))),),
                }
            ),
        )
        handle = native.NativeComponentHandle(
            module,
            mechanism,
            CPU,
            resource_identity="native:test:" + "2" * 64,
            coordinator=coordinator,
            discard_on_release=discard_on_release,
        )
        manager.load([mechanism])
        module_ref = weakref.ref(module)
        child_ref = weakref.ref(module.child)
        tensor_refs = [weakref.ref(tensor) for tensor in module.parameters()]
        patched_root = module.root_weight
        patched_child = module.child.weight
        with torch.no_grad():
            result = module.child(torch.ones(1, 2))
            expected = torch.nn.functional.linear(
                torch.ones(1, 2), base.child.weight + 1, base.child.bias
            )
        assert torch.equal(result, expected)
        handle.terminal_release()
        assert handle.released
        assert manager.registered() == ()
        assert handle.mechanisms == ()
        assert mechanism.loaded_bytes() == 0
        assert module.residency_binding() is None
        assert "_dinkster_residency_state_store" not in module.__dict__
        if discard_on_release:
            assert module.root_weight is patched_root
            assert module.child.weight is patched_child
        else:
            assert module.root_weight.data_ptr() == base.root_weight.data_ptr()
            assert module.child.weight.data_ptr() == base.child.weight.data_ptr()
        for name, tensor in base.named_parameters():
            assert tensor is base_parameters[name]
            assert tensor.data_ptr() == base_ptrs[name]
            assert torch.equal(tensor, base_values[name])
        del module, mechanism, patched_root, patched_child
        assert module_ref() is None
        assert child_ref() is None
        assert all(ref() is None for ref in tensor_refs)
    finally:
        if was_enabled:
            gc.enable()


def test_terminal_detachment_preserves_state_and_allows_reenrollment() -> None:
    module = INITLESS.linear(2, 2)
    routed = cast(ResidencyRouted, module)
    weight = module.weight
    mechanism = enroll_component(module, load_device=CPU, offload_device=CPU)

    detach_residency_enrollment(module, mechanism)

    assert module.weight is weight
    assert routed.residency_binding() is None
    replacement = enroll_component(module, load_device=CPU, offload_device=CPU)
    binding = routed.residency_binding()
    assert binding is not None
    assert binding.mechanism is replacement


def test_terminal_detachment_preserves_shared_parameter_storage() -> None:
    first = INITLESS.linear(2, 2)
    second = INITLESS.linear(2, 2)
    second.weight = first.weight
    module = torch.nn.Sequential(first, second)
    mechanism = enroll_component(module, load_device=CPU, offload_device=CPU)

    detach_residency_enrollment(module, mechanism)

    assert first.weight is second.weight
    replacement = enroll_component(module, load_device=CPU, offload_device=CPU)
    assert first.weight is second.weight
    detach_residency_enrollment(module, replacement)


@pytest.mark.parametrize("cast_at_use", [False, True], ids=["initless", "cast"])
def test_every_operations_layer_routes_bitwise(cast_at_use: bool) -> None:
    operations: Operations = CastOperations(torch.float32) if cast_at_use else INITLESS
    storage_dtype = torch.float16 if cast_at_use else torch.float32
    for module, input in _operation_cases(operations):
        _fill(module, storage_dtype)
        mechanism = enroll_component(module, load_device=CPU, offload_device=CPU)
        mechanism.partially_load(None)
        resident = module(input)
        mechanism.partially_unload(mechanism.loaded_bytes())
        offloaded = module(input)
        assert torch.equal(offloaded, resident)
        if not cast_at_use:
            assert offloaded.dtype == storage_dtype


class _TrackingWeightLease:
    def __init__(
        self,
        lease: WeightLease,
        requests: list[tuple[str, torch.dtype]],
    ) -> None:
        self.lease = lease
        self.requests = requests

    def get(self, key: str, *, dtype: torch.dtype) -> torch.Tensor:
        self.requests.append((key, dtype))
        return self.lease.get(key, dtype=dtype)

    def get_stored(self, key: str) -> StoredWeight:
        return self.lease.get_stored(key)

    def timing_collector(self) -> PartialResidencyTiming | None:
        return self.lease.timing_collector()


class _TrackingResidentWeights(ResidentWeights):
    def __init__(
        self,
        weights: MutableMapping[str, StoredWeight],
        *,
        load_device: torch.device | str,
        offload_device: torch.device | str,
        patch_set: PatchSet[torch.Tensor] | None = None,
        units: Sequence[ResidencyUnit] | None = None,
        intermediate_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            weights,
            load_device=load_device,
            offload_device=offload_device,
            patch_set=patch_set,
            units=units,
            intermediate_dtype=intermediate_dtype,
        )
        self.opens = 0
        self.closes = 0
        self.requests: list[tuple[str, torch.dtype]] = []
        self.closed_leases: list[_TrackingWeightLease] = []

    def lease(self, unit: str) -> AbstractContextManager[WeightLease]:
        inner = super().lease(unit)

        @contextmanager
        def bracket() -> Generator[WeightLease]:
            self.opens += 1
            wrapped: _TrackingWeightLease | None = None
            try:
                with inner as lease:
                    wrapped = _TrackingWeightLease(lease, self.requests)
                    yield wrapped
            finally:
                self.closes += 1
                if wrapped is not None:
                    self.closed_leases.append(wrapped)

        return bracket()


def test_cast_operations_prefetch_matches_every_factory_lease_request() -> None:
    for module, input_value in _operation_cases(CastOperations(torch.float32)):
        _fill(module, torch.float16)
        mechanism = cast(
            _TrackingResidentWeights,
            enroll_component(
                module,
                load_device=CPU,
                offload_device=CPU,
                mechanism_factory=_TrackingResidentWeights,
            ),
        )
        prefetch = module.residency_prefetch()  # type: ignore[attr-defined]
        assert prefetch is not None
        assert prefetch[0] is mechanism
        assert prefetch[1]
        assert all(dtype == torch.float32 for _key, dtype in prefetch[1])

        module(input_value)

        assert mechanism.opens == mechanism.closes == 1
        assert len(mechanism.requests) == len(prefetch[1])
        assert dict(mechanism.requests) == dict(prefetch[1])
        assert len(mechanism.closed_leases) == 1
        for key, dtype in prefetch[1]:
            assert dtype is not None
            with pytest.raises(RuntimeError, match="closed"):
                mechanism.closed_leases[0].get(key, dtype=dtype)


def test_cast_embedding_offloaded_patch_uses_compute_dtype_before_gather() -> None:
    embedding = CastOperations(torch.float32).embedding(8, 4)
    _fill(embedding, torch.float16)
    original = embedding.weight.detach().clone()
    delta = torch.full_like(original, 0.125)
    mechanism = cast(
        _TrackingResidentWeights,
        enroll_component(
            embedding,
            load_device=CPU,
            offload_device=CPU,
            patch_set=PatchSet({"weight": (PatchEntry(DiffPatch(delta)),)}),
            mechanism_factory=_TrackingResidentWeights,
        ),
    )
    token_ids = torch.tensor([[1, 4, 1]])

    assert embedding.residency_prefetch() == (mechanism, (("weight", torch.float32),))  # type: ignore[attr-defined]
    output = embedding(token_ids)

    assert mechanism.requests == [("weight", torch.float32)]
    expected_weight = apply_patches(
        original.to(torch.float32),
        (PatchEntry(DiffPatch(delta)),),
        key="weight",
        original_weight=original,
    )
    assert torch.equal(output, torch.nn.functional.embedding(token_ids, expected_weight))


def test_cast_embedding_offloaded_inference_prefetches_storage_dtype() -> None:
    embedding = CastOperations(torch.float32).embedding(8, 4)
    _fill(embedding, torch.float16)
    mechanism = cast(
        _TrackingResidentWeights,
        enroll_component(
            embedding,
            load_device=CPU,
            offload_device=CPU,
            mechanism_factory=_TrackingResidentWeights,
        ),
    )

    with torch.no_grad():
        assert embedding.residency_prefetch() == (mechanism, (("weight", torch.float16),))  # type: ignore[attr-defined]
        embedding(torch.tensor([[1, 4, 1]]))

    assert mechanism.requests == [("weight", torch.float16)]


def test_public_residency_subclass_routes_whole_owned_state() -> None:
    module = _filled_external_residency_unit()
    assert tuple(module.state_dict()) == ("gain", "offset", "projection.weight")

    mechanism = cast(
        _TrackingResidentWeights,
        enroll_component(
            module,
            load_device=CPU,
            offload_device=CPU,
            mechanism_factory=_TrackingResidentWeights,
        ),
    )
    with pytest.raises(RuntimeError, match="already enrolled"):
        enroll_component(module, load_device=CPU, offload_device=CPU)

    mechanism.partially_load(None)
    input_value = torch.randn(3, 2)
    resident = module(input_value)
    assert module.residency_prefetch() is None
    assert module.projection.residency_prefetch() is None  # type: ignore[attr-defined]

    mechanism.unload()
    prefetch = module.residency_prefetch()
    assert prefetch == (
        mechanism,
        (("gain", torch.float16), ("offset", torch.float32)),
    )
    prefetch_again = module.residency_prefetch()
    assert prefetch is not None
    assert prefetch_again is not None
    assert prefetch[1] is prefetch_again[1]
    assert module.projection.residency_prefetch() == (  # type: ignore[attr-defined]
        mechanism,
        (("projection.weight", torch.float32),),
    )
    offloaded = module(input_value)

    assert torch.equal(offloaded, resident)
    assert mechanism.opens == mechanism.closes == 2
    assert mechanism.requests == [
        ("gain", torch.float16),
        ("offset", torch.float32),
        ("projection.weight", torch.float32),
    ]
    assert len(mechanism.closed_leases) == 2
    for lease in mechanism.closed_leases:
        with pytest.raises(RuntimeError, match="closed"):
            lease.get("gain", dtype=torch.float32)


class _ConstantBufferUnit(ResidencyRouted, torch.nn.Module):
    """One routed owner mixing persistent state with a derived constant."""

    def __init__(self) -> None:
        super().__init__()
        self.gain = torch.nn.Parameter(torch.empty(2), requires_grad=False)
        self.register_buffer("offset", torch.empty(2), persistent=True)
        self.register_buffer("anchor", torch.arange(2, dtype=torch.float32), persistent=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        anchor = cast(torch.Tensor, self.anchor)
        binding = self._offloaded_residency()
        if binding is None:
            return value * self.gain + cast(torch.Tensor, self.offset) + anchor
        with binding.lease() as lease:
            return (
                value * lease.get("gain", dtype=torch.float32)
                + lease.get("offset", dtype=torch.float32)
                + anchor
            )


def test_generic_prefetch_excludes_non_persistent_buffers() -> None:
    module = _ConstantBufferUnit()
    generator = torch.Generator().manual_seed(74)
    state = {
        key: torch.randn(tuple(value.shape), generator=generator).to(value.dtype)
        for key, value in module.state_dict().items()
    }
    module.load_state_dict(state, strict=True, assign=True)
    assert tuple(module.state_dict()) == ("gain", "offset")

    mechanism = enroll_component(module, load_device=CPU, offload_device=CPU)
    mechanism.partially_load(None)
    input_value = torch.randn(3, 2)
    resident = module(input_value)
    assert module.residency_prefetch() is None

    mechanism.unload()
    prefetch = module.residency_prefetch()
    assert prefetch is not None
    assert [key for key, _dtype in prefetch[1]] == ["gain", "offset"]
    assert torch.equal(module(input_value), resident)


def test_public_residency_subclass_failure_closes_every_owning_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _filled_external_residency_unit()
    mechanism = cast(
        _TrackingResidentWeights,
        enroll_component(
            module,
            load_device=CPU,
            offload_device=CPU,
            mechanism_factory=_TrackingResidentWeights,
        ),
    )
    mechanism.unload()

    def fail(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise RuntimeError("injected external subclass failure")

    monkeypatch.setattr(torch.nn.functional, "linear", fail)
    with pytest.raises(RuntimeError, match="injected external subclass failure"):
        module(torch.randn(3, 2))

    assert mechanism.opens == mechanism.closes == 2
    assert len(mechanism.closed_leases) == 2
    for lease in mechanism.closed_leases:
        with pytest.raises(RuntimeError, match="closed"):
            lease.get("gain", dtype=torch.float32)


@pytest.mark.parametrize("cast_at_use", [False, True], ids=["initless", "cast"])
@pytest.mark.parametrize("bias", [False, True], ids=["no_bias", "bias"])
def test_convolution_residency_uses_one_closed_lease_and_exact_dtypes(
    cast_at_use: bool,
    bias: bool,
) -> None:
    operations: Operations = CastOperations(torch.float32) if cast_at_use else INITLESS
    storage_dtype = torch.float16 if cast_at_use else torch.float32
    cases = (
        (
            operations.conv1d(4, 6, 3, stride=2, padding=2, dilation=2, groups=2, bias=bias),
            torch.randn(2, 4, 13),
        ),
        (
            operations.conv_transpose1d(
                4,
                6,
                3,
                stride=2,
                padding=1,
                output_padding=1,
                groups=2,
                bias=bias,
                dilation=2,
            ),
            torch.randn(2, 4, 7),
        ),
        (
            operations.conv3d(
                4,
                6,
                (2, 3, 3),
                stride=(1, 2, 1),
                padding=(1, 1, 2),
                dilation=(1, 2, 1),
                groups=2,
                bias=bias,
            ),
            torch.randn(2, 4, 5, 9, 8),
        ),
    )
    for module, input_value in cases:

        def forward(current: torch.nn.Module, value: torch.Tensor) -> torch.Tensor:
            if isinstance(current, torch.nn.ConvTranspose1d):
                return current(value, output_size=[15])
            return current(value)

        _fill(module, storage_dtype)
        original = dict(module.state_dict(keep_vars=True))
        mechanism = cast(
            _TrackingResidentWeights,
            enroll_component(
                module,
                load_device=CPU,
                offload_device=CPU,
                mechanism_factory=_TrackingResidentWeights,
            ),
        )
        mechanism.partially_load(None)
        expected = forward(module, input_value)
        assert mechanism.opens == 0
        mechanism.unload()
        first = forward(module, input_value)
        second = forward(module, input_value)
        assert torch.equal(first, expected)
        assert torch.equal(second, expected)
        assert mechanism.opens == mechanism.closes == 2
        expected_dtype = torch.float32 if cast_at_use else storage_dtype
        expected_requests = [("weight", expected_dtype)]
        if bias:
            expected_requests.insert(0, ("bias", expected_dtype))
        assert mechanism.requests == expected_requests * 2
        assert len(mechanism.closed_leases) == 2
        assert mechanism.closed_leases[0] is not mechanism.closed_leases[1]
        with pytest.raises(RuntimeError, match="closed"):
            mechanism.closed_leases[0].get("weight", dtype=expected_dtype)
        assert all(
            module.state_dict(keep_vars=True)[key] is value for key, value in original.items()
        )


def test_convolution_residency_failure_closes_the_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = INITLESS.conv_transpose1d(4, 6, 3, groups=2)
    _fill(module)
    mechanism = cast(
        _TrackingResidentWeights,
        enroll_component(
            module,
            load_device=CPU,
            offload_device=CPU,
            mechanism_factory=_TrackingResidentWeights,
        ),
    )
    mechanism.unload()

    def fail(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise RuntimeError("injected convolution failure")

    monkeypatch.setattr(torch.nn.functional, "conv_transpose1d", fail)
    with pytest.raises(RuntimeError, match="injected convolution failure"):
        module(torch.randn(2, 4, 7))
    assert mechanism.opens == mechanism.closes == 1
    assert len(mechanism.closed_leases) == 1
    with pytest.raises(RuntimeError, match="closed"):
        mechanism.closed_leases[0].get("weight", dtype=torch.float32)


@pytest.mark.parametrize("cast_at_use", [False, True], ids=["initless", "cast"])
def test_materialized_conv2d_parameters_route_loaded_and_offloaded_state(
    cast_at_use: bool,
) -> None:
    operations: Operations = CastOperations(torch.float32) if cast_at_use else INITLESS
    storage_dtype = torch.float16 if cast_at_use else torch.float32
    module = operations.conv2d(4, 6, 3, bias=True)
    _fill(module, storage_dtype)
    mechanism = cast(
        _TrackingResidentWeights,
        enroll_component(
            module,
            load_device=CPU,
            offload_device=CPU,
            mechanism_factory=_TrackingResidentWeights,
        ),
    )
    mechanism.partially_load(None)
    with materialized_conv2d_parameters(module) as loaded:
        loaded_parameters = tuple(
            parameter.clone() for parameter in loaded if parameter is not None
        )
    assert mechanism.opens == 0

    mechanism.unload()
    with materialized_conv2d_parameters(module) as offloaded:
        offloaded_parameters = tuple(
            parameter.clone() for parameter in offloaded if parameter is not None
        )

    assert len(loaded_parameters) == len(offloaded_parameters) == 2
    assert all(
        torch.equal(current, expected)
        for current, expected in zip(offloaded_parameters, loaded_parameters, strict=True)
    )
    expected_dtype = torch.float32 if cast_at_use else storage_dtype
    assert mechanism.requests == [("weight", expected_dtype), ("bias", expected_dtype)]
    assert mechanism.opens == mechanism.closes == 1


def test_t5_layer_norm_routes_bitwise_with_identical_cast_order() -> None:
    norm = T5LayerNorm(8)
    norm.weight = torch.nn.Parameter(torch.randn(8).to(torch.float16))
    input = torch.randn(3, 8)
    assert torch.equal(
        cast_weight(norm.weight, dtype=input.dtype, device=CPU),
        norm.weight.to(dtype=input.dtype),
    )
    mechanism = enroll_component(norm, load_device=CPU, offload_device=CPU)
    mechanism.partially_load(None)
    resident = norm(input)
    mechanism.partially_unload(mechanism.loaded_bytes())
    offloaded = norm(input)
    assert torch.equal(offloaded, resident)


@pytest.mark.parametrize("cast_at_use", [False, True], ids=["initless", "cast"])
def test_mixed_residency_routes_direct_t5_norm_inside_operations_tree(
    cast_at_use: bool,
) -> None:
    operations: Operations = CastOperations(torch.float32) if cast_at_use else INITLESS
    module = _MixedT5(operations)
    _fill(module, torch.float16 if cast_at_use else torch.float32)
    input = torch.randn(2, 4)
    mechanism = enroll_component(module, load_device=CPU, offload_device=CPU)
    mechanism.partially_load(None)
    resident = module(input)
    mechanism.partially_unload(1)
    assert not mechanism.is_loaded("norm")
    assert mechanism.loaded_bytes() < mechanism.total_bytes()
    assert torch.equal(module(input), resident)


def test_offloaded_patch_matches_resident_patch_and_restores_object() -> None:
    module = INITLESS.linear(4, 3, bias=False)
    _fill(module)
    original = module.weight
    delta = torch.full_like(original, 0.125)
    patch_set = PatchSet({"weight": (PatchEntry(DiffPatch(delta)),)})
    mechanism = enroll_component(
        module,
        load_device=CPU,
        offload_device=CPU,
        patch_set=patch_set,
    )
    input = torch.randn(2, 4)
    mechanism.partially_load(None)
    resident = module(input)
    mechanism.unload()
    assert module.weight is original
    assert torch.equal(module(input), resident)


def test_fp8_dequant_route_is_bitwise_and_layout_survives() -> None:
    layer = _fp8_layer()
    before = {key: tuple(value.shape) for key, value in layer.state_dict().items()}
    input = torch.randn(5, 4)
    mechanism = enroll_component(layer, load_device=CPU, offload_device=CPU)
    route = layer.residency_prefetch()
    assert route is not None
    assert route == (
        mechanism,
        (("weight", None), ("bias", torch.float32)),
    )
    route_again = layer.residency_prefetch()
    assert route_again is not None
    assert route[1] is route_again[1]
    mechanism.partially_load(None)
    resident = layer(input)
    mechanism.unload()
    assert torch.equal(layer(input), resident)
    mechanism.partially_load(None)
    after = {key: tuple(value.shape) for key, value in layer.state_dict().items()}
    assert after == before


def test_fp8_patch_route_is_cached_at_residency_binding() -> None:
    layer = _fp8_layer(bias=False)
    patch_set = PatchSet({"weight": (PatchEntry(DiffPatch(torch.ones(3, 4))),)})
    mechanism = enroll_component(
        layer,
        load_device=CPU,
        offload_device=CPU,
        patch_set=patch_set,
    )
    input = torch.randn(2, 4)
    mechanism.partially_load(None)
    layer.bind_fp8_matmul(True)
    mechanism.unload()

    route = layer.residency_prefetch()
    assert route is not None
    assert route == (mechanism, (("weight", torch.float32),))
    route_again = layer.residency_prefetch()
    assert route_again is not None
    assert route[1] is route_again[1]
    with mechanism.lease("weight") as lease:
        expected_weight = lease.get("weight", dtype=torch.float32)
    expected = torch.nn.functional.linear(input, expected_weight)
    assert torch.equal(layer(input), expected)


def test_mps_upcast_swaps_scaled_fp8_linear_bitwise() -> None:
    toy = torch.nn.Sequential(_fp8_layer())
    layer = cast(Fp8Linear, toy[0])
    generator = torch.Generator().manual_seed(29)
    input = torch.randn(5, 4, generator=generator)
    expected_weight = layer.weight.to(dtype=torch.float32) * layer.weight_scale.to(
        dtype=torch.float32
    )
    expected = toy(input)
    _apply_mps_fp8_upcast(toy)
    replaced = toy[0]
    assert not isinstance(replaced, Fp8Linear)
    assert isinstance(replaced, torch.nn.Linear)
    assert replaced.weight.dtype == torch.float32
    assert torch.equal(replaced.weight, expected_weight)
    assert torch.equal(toy(input), expected)
    assert set(toy.state_dict()) == {"0.weight", "0.bias"}


def test_mps_upcast_constructs_replacement_without_allocating_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = module_residency_mod.INITLESS.linear

    def linear(
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
    ) -> torch.nn.Linear:
        replacement = original(in_features, out_features, bias=bias)
        assert replacement.weight.device.type == "meta"
        assert replacement.bias.device.type == "meta"
        return replacement

    monkeypatch.setattr(module_residency_mod.INITLESS, "linear", linear)
    toy = torch.nn.Sequential(_fp8_layer())

    _apply_mps_fp8_upcast(toy)

    assert toy[0].weight.device.type == "cpu"
    assert toy[0].bias.device.type == "cpu"


def test_mps_upcast_casts_plain_fp8_cast_layer_bitwise() -> None:
    layer = CastOperations(torch.float32).linear(4, 3)
    generator = torch.Generator().manual_seed(23)
    layer.load_state_dict(
        {
            "weight": torch.randn(3, 4, generator=generator).to(torch.float8_e4m3fn),
            "bias": torch.randn(3, generator=generator),
        },
        strict=True,
        assign=True,
    )
    toy = torch.nn.Sequential(layer)
    input = torch.randn(5, 4, generator=generator)
    expected_weight = layer.weight.to(dtype=torch.float32)
    expected = toy(input)
    _apply_mps_fp8_upcast(toy)
    assert toy[0] is layer
    assert layer.weight.dtype == torch.float32
    assert torch.equal(layer.weight, expected_weight)
    assert torch.equal(toy(input), expected)


def _mps_snapshot(available: int) -> MpsMemorySnapshot:
    return MpsMemorySnapshot(
        recommended_max_bytes=4_000_000_000,
        driver_allocated_bytes=1_000_000_000,
        current_allocated_bytes=750_000_000,
        system_total_bytes=8_000_000_000,
        system_available_bytes=available,
    )


@pytest.fixture
def ample_mps_upcast_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    def snapshot(_device: torch.device) -> MpsMemorySnapshot:
        return _mps_snapshot(3_000_000_000)

    monkeypatch.setattr(
        module_residency_mod,
        "mps_memory_snapshot",
        snapshot,
    )


def test_mps_upcast_byte_plan_deduplicates_module_and_tensor_aliases() -> None:
    scaled = _fp8_layer()
    first = CastOperations(torch.float32).linear(4, 3, bias=False)
    second = CastOperations(torch.float32).linear(4, 3, bias=False)
    shared = torch.nn.Parameter(torch.randn(3, 4).to(torch.float8_e4m3fn))
    first.weight = shared
    second.weight = shared
    root = torch.nn.Module()
    root.scaled_first = scaled
    root.scaled_second = scaled
    root.plain_first = first
    root.plain_second = second

    plan = module_residency_mod._plan_fp8_upcast_for_mps(  # pyright: ignore[reportPrivateUsage]
        root
    )
    planned = module_residency_mod._fp8_upcast_nbytes(  # pyright: ignore[reportPrivateUsage]
        (plan,)
    )

    assert planned == 2 * (3 * 4 * torch.float32.itemsize)


def test_mps_upcast_byte_plan_includes_transient_scale_cast() -> None:
    scaled = _fp8_layer(bias=False, compute_dtype=torch.bfloat16)
    root = torch.nn.Sequential(scaled)

    plan = module_residency_mod._plan_fp8_upcast_for_mps(  # pyright: ignore[reportPrivateUsage]
        root
    )
    planned = module_residency_mod._fp8_upcast_nbytes(  # pyright: ignore[reportPrivateUsage]
        (plan,)
    )

    assert planned == (3 * 4 + 1) * torch.bfloat16.itemsize


def test_mps_fp8_upcast_attempts_allocation_and_cites_budget(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    reserve = MemoryPolicy().minimum_inference_memory()
    toy = torch.nn.Sequential(_fp8_layer())
    layer = toy[0]

    def snapshot(_device: torch.device) -> MpsMemorySnapshot:
        return _mps_snapshot(reserve + 47)

    monkeypatch.setattr(module_residency_mod, "mps_memory_snapshot", snapshot)

    enroll_component(toy, load_device="mps", offload_device=CPU)

    message = caplog.text
    assert "48 bytes of anonymous compute-dtype storage" in message
    assert f"{reserve}-byte inference reserve" in message
    assert f"{reserve + 47} bytes available" in message
    assert "1000000000 bytes driver-allocated" in message
    assert "attempting CPU dequantization" in message
    assert toy[0] is not layer
    assert toy[0].weight.dtype == torch.float32


def test_mps_fp8_upcast_preserves_actual_allocation_failure(
    ample_mps_upcast_memory: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = torch.OutOfMemoryError("CPU dequantization allocation failed")
    layer = _fp8_layer()
    toy = torch.nn.Sequential(layer)

    def allocation_failure(_layer: Fp8Linear) -> torch.nn.Linear:
        raise failure

    monkeypatch.setattr(module_residency_mod, "_upcast_fp8_linear", allocation_failure)
    with pytest.raises(torch.OutOfMemoryError) as raised:
        enroll_component(toy, load_device="mps", offload_device=CPU)
    assert raised.value is failure
    assert toy[0] is layer
    assert layer.weight.dtype == torch.float8_e4m3fn


def test_mps_fp8_upcast_admits_exact_available_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reserve = MemoryPolicy().minimum_inference_memory()

    def snapshot(_device: torch.device) -> MpsMemorySnapshot:
        return _mps_snapshot(reserve + 48)

    monkeypatch.setattr(module_residency_mod, "mps_memory_snapshot", snapshot)
    toy = torch.nn.Sequential(_fp8_layer())

    enroll_component(
        toy,
        load_device="mps",
        offload_device=CPU,
    )

    assert not isinstance(toy[0], Fp8Linear)


def test_enroll_assembled_mps_fp8_pressure_is_aggregate_and_nonblocking(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    reserve = MemoryPolicy().minimum_inference_memory()
    diffusion_layer = _fp8_layer()
    clip_layer = _fp8_layer()
    diffusion = torch.nn.Sequential(diffusion_layer)
    clip_l = torch.nn.Sequential(clip_layer)
    assembled = _policy_assembled(diffusion, clip_l, _policy_linear(), enabled=False)

    def snapshot(_device: torch.device) -> MpsMemorySnapshot:
        return _mps_snapshot(reserve + 95)

    monkeypatch.setattr(module_residency_mod, "mps_memory_snapshot", snapshot)

    enroll_assembled(assembled, load_device="mps", offload_device=CPU)

    assert "FP8 upcast needs 96 bytes" in caplog.text
    assert diffusion[0] is not diffusion_layer
    assert clip_l[0] is not clip_layer
    assert diffusion[0].weight.dtype == torch.float32
    assert clip_l[0].weight.dtype == torch.float32


def test_enroll_assembled_mps_fp8_upcast_deduplicates_across_components(
    ample_mps_upcast_memory: None,
) -> None:
    shared = _fp8_layer()
    diffusion = torch.nn.Sequential(shared)
    clip_l = torch.nn.Sequential(shared)
    assembled = _policy_assembled(diffusion, clip_l, _policy_linear(), enabled=False)

    enroll_assembled(
        assembled,
        load_device="mps",
        offload_device=CPU,
    )

    assert diffusion[0] is clip_l[0]
    assert not isinstance(diffusion[0], Fp8Linear)


def test_mps_enrollment_without_fp8_does_not_consult_upcast_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = INITLESS.linear(4, 3)
    _fill(module, torch.bfloat16)

    def unexpected_snapshot(_device: torch.device) -> MpsMemorySnapshot:
        raise AssertionError("non-FP8 enrollment must not consult the upcast budget")

    monkeypatch.setattr(module_residency_mod, "mps_memory_snapshot", unexpected_snapshot)
    enroll_component(module, load_device="mps", offload_device=CPU)


def test_mps_assembled_enrollment_without_fp8_does_not_consult_upcast_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assembled = _policy_assembled(
        _policy_linear(),
        _policy_linear(),
        _policy_linear(),
        enabled=False,
    )

    def unexpected_snapshot(_device: torch.device) -> MpsMemorySnapshot:
        raise AssertionError("non-FP8 assembly must not consult the upcast budget")

    monkeypatch.setattr(module_residency_mod, "mps_memory_snapshot", unexpected_snapshot)
    enroll_assembled(assembled, load_device="mps", offload_device=CPU)


def test_mps_enrollment_swaps_scaled_fp8_storage(
    ample_mps_upcast_memory: None,
) -> None:
    toy = torch.nn.Sequential(_fp8_layer())
    enroll_component(toy, load_device="mps", offload_device=CPU)
    assert not any(isinstance(module, Fp8Linear) for module in toy.modules())
    assert toy.state_dict()["0.weight"].dtype == torch.float32


def test_cpu_enrollment_keeps_scaled_fp8_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    toy = torch.nn.Sequential(_fp8_layer())
    layer = toy[0]

    def unexpected_snapshot(_device: torch.device) -> MpsMemorySnapshot:
        raise AssertionError("non-MPS enrollment must not consult MPS memory")

    monkeypatch.setattr(module_residency_mod, "mps_memory_snapshot", unexpected_snapshot)
    enroll_component(toy, load_device=CPU, offload_device=CPU)
    assert toy[0] is layer
    assert cast(Fp8Linear, toy[0]).weight.dtype == torch.float8_e4m3fn


def test_mps_enrollment_upcasts_fp8_hardware_matmul_layers(
    ample_mps_upcast_memory: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    layer = _fp8_layer()
    layer.fp8_matmul = True
    toy = torch.nn.Sequential(layer)
    expected = layer.weight.float() * layer.weight_scale.float()
    enroll_component(toy, load_device="mps", offload_device=CPU)
    assert not isinstance(toy[0], Fp8Linear)
    replacement = toy[0]
    assert isinstance(replacement, torch.nn.Linear)
    assert torch.equal(replacement.weight, expected)
    assert "using CPU dequantization" in caplog.text


def test_mps_enrollment_refuses_bare_scaled_fp8_linear() -> None:
    with pytest.raises(RuntimeError, match="owning parent module"):
        enroll_component(_fp8_layer(), load_device="mps", offload_device=CPU)


def test_enroll_assembled_upcasts_scaled_fp8_for_mps(
    ample_mps_upcast_memory: None,
) -> None:
    clip_l = torch.nn.Sequential(_fp8_layer())
    assembled = _policy_assembled(_policy_linear(), clip_l, _policy_linear(), enabled=False)
    enroll_assembled(assembled, load_device="mps", offload_device=CPU)
    assert not any(isinstance(module, Fp8Linear) for module in clip_l.modules())


def test_enroll_assembled_mps_fp8_matmul_fallback_names_component_and_family(
    ample_mps_upcast_memory: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    layer = _fp8_layer()
    layer.fp8_matmul = True
    clip_l = torch.nn.Sequential(layer)
    assembled = _policy_assembled(_policy_linear(), clip_l, _policy_linear(), enabled=False)
    enroll_assembled(assembled, load_device="mps", offload_device=CPU)
    assert "component 'clip_l' of family dinkster.sd15" in caplog.text
    assert not isinstance(clip_l[0], Fp8Linear)


def test_mps_upcast_replaces_fp8_linear_registered_under_two_names() -> None:
    layer = _fp8_layer()
    parent = torch.nn.Module()
    parent.first = layer
    parent.second = layer
    _apply_mps_fp8_upcast(parent)
    assert not isinstance(parent.first, Fp8Linear)
    assert not isinstance(parent.second, Fp8Linear)
    assert parent.first is parent.second


def test_mps_upcast_shares_replacement_across_parents() -> None:
    layer = _fp8_layer()
    left = torch.nn.Sequential(layer)
    right = torch.nn.Sequential(layer)
    root = torch.nn.ModuleList([left, right])
    _apply_mps_fp8_upcast(root)
    assert not isinstance(left[0], Fp8Linear)
    assert left[0] is right[0]


def test_mps_upcast_refusal_leaves_module_unmodified() -> None:
    good = _fp8_layer()
    plain = CastOperations(torch.float32).linear(4, 3)
    plain.load_state_dict(
        {
            "weight": torch.randn(3, 4).to(torch.float8_e4m3fn),
            "bias": torch.randn(3),
        },
        strict=True,
        assign=True,
    )
    bad = torch.nn.Linear(4, 3)
    _fill(bad, torch.float8_e4m3fn)
    # Missing compute dtype must invalidate all planned swaps and casts.
    toy = torch.nn.Sequential(good, plain, torch.nn.Sequential(bad))
    with pytest.raises(RuntimeError, match="no bound compute dtype"):
        _apply_mps_fp8_upcast(toy)
    assert toy[0] is good
    assert isinstance(toy[0], Fp8Linear)
    assert good.weight.dtype == torch.float8_e4m3fn
    assert plain.weight.dtype == torch.float8_e4m3fn


def test_mps_enrollment_of_enrolled_module_refuses_before_rewriting() -> None:
    toy = torch.nn.Sequential(_fp8_layer())
    layer = toy[0]
    enroll_component(toy, load_device=CPU, offload_device=CPU)
    with pytest.raises(RuntimeError, match="already enrolled"):
        enroll_component(toy, load_device="mps", offload_device=CPU)
    assert toy[0] is layer
    assert cast(Fp8Linear, toy[0]).weight.dtype == torch.float8_e4m3fn


def test_enroll_assembled_mps_refusal_leaves_all_components_unmodified() -> None:
    clip_layer = _fp8_layer()
    clip_l = torch.nn.Sequential(clip_layer)
    bad = torch.nn.Linear(4, 3)
    _fill(bad, torch.float8_e4m3fn)
    vae = torch.nn.Sequential(bad)
    assembled = _policy_assembled(_policy_linear(), clip_l, vae, enabled=False)
    with pytest.raises(RuntimeError, match=r"component 'vae' of family dinkster\.sd15"):
        enroll_assembled(assembled, load_device="mps", offload_device=CPU)
    assert clip_l[0] is clip_layer
    assert isinstance(clip_l[0], Fp8Linear)
    assert clip_layer.weight.dtype == torch.float8_e4m3fn


def test_enroll_assembled_mps_unknown_patch_component_rejected_before_rewriting() -> None:
    clip_layer = _fp8_layer()
    clip_l = torch.nn.Sequential(clip_layer)
    assembled = _policy_assembled(_policy_linear(), clip_l, _policy_linear(), enabled=False)
    with pytest.raises(ValueError, match="unknown assembled components"):
        enroll_assembled(
            assembled,
            load_device="mps",
            offload_device=CPU,
            patch_sets={"bogus": PatchSet({})},
        )
    assert clip_l[0] is clip_layer
    assert clip_layer.weight.dtype == torch.float8_e4m3fn


def test_enroll_assembled_mps_enrolled_component_rejected_before_rewriting() -> None:
    diffusion = _policy_linear()
    enroll_component(diffusion, load_device=CPU, offload_device=CPU)
    clip_layer = _fp8_layer()
    clip_l = torch.nn.Sequential(clip_layer)
    assembled = _policy_assembled(diffusion, clip_l, _policy_linear(), enabled=False)
    with pytest.raises(RuntimeError, match="already enrolled"):
        enroll_assembled(assembled, load_device="mps", offload_device=CPU)
    assert clip_l[0] is clip_layer
    assert clip_layer.weight.dtype == torch.float8_e4m3fn


def test_nvfp4_ordinary_state_residency_roundtrip_and_prefetch() -> None:
    layer = _nvfp4_layer()
    diagnostics = Nvfp4DiagnosticsRecorder()
    layer._bind_diagnostics(diagnostics)  # pyright: ignore[reportPrivateUsage]
    original = dict(layer.state_dict(keep_vars=True))
    store = ModuleStateStore(layer)
    assert set(store) == {
        "weight",
        "input_scale",
        "pre_quant_scale",
        "bias",
    }
    assert isinstance(store["weight"], Nvfp4PackedWeight)
    assert store.protected_quantization_keys() == {
        "weight_scale",
        "weight_scale_2",
        "input_scale",
        "pre_quant_scale",
    }
    input = torch.randn(3, 16)
    expected = layer(input)
    mechanism = enroll_component(layer, load_device=CPU, offload_device=CPU)
    prefetch = layer.residency_prefetch()
    assert prefetch is not None
    assert prefetch[0] is mechanism
    assert prefetch[1] == (
        ("weight", None),
        ("input_scale", torch.float32),
        ("pre_quant_scale", torch.float32),
        ("bias", torch.float32),
    )
    assert torch.equal(layer(input), expected)
    mechanism.partially_load(None)
    assert layer.residency_prefetch() is None
    assert torch.equal(layer(input), expected)
    before = mechanism.total_bytes()
    mechanism.unload()
    assert mechanism.total_bytes() == before
    assert torch.equal(layer(input), expected)
    assert all(layer.state_dict(keep_vars=True)[key] is value for key, value in original.items())
    status = diagnostics.snapshot()
    assert status.lifetime["observed_loaded"] >= 3
    assert status.lifetime["observed_offloaded"] >= 2
    assert status.lifetime["observed_state_change"] >= 4


def test_int8_convrot_state_residency_roundtrip_and_prefetch() -> None:
    layer = _int8_layer()
    original = dict(layer.state_dict(keep_vars=True))
    store = ModuleStateStore(layer)
    assert set(store) == {"weight", "bias"}
    assert store.protected_quantization_keys() == {"weight_scale"}
    stored = store["weight"]
    assert isinstance(stored, Int8PackedWeight)
    assert stored.qdata is original["weight"]
    assert stored.scale is original["weight_scale"]
    assert stored.convrot
    assert stored.convrot_groupsize == 256
    input = torch.randn(3, 256)
    expected = layer(input)
    mechanism = enroll_component(layer, load_device=CPU, offload_device=CPU)
    prefetch = layer.residency_prefetch()
    assert prefetch is not None
    assert prefetch[0] is mechanism
    assert prefetch[1] == (
        ("weight", None),
        ("bias", torch.float32),
    )
    assert torch.equal(layer(input), expected)
    mechanism.partially_load(None)
    assert layer.residency_prefetch() is None
    assert torch.equal(layer(input), expected)
    mechanism.unload()
    assert torch.equal(layer(input), expected)
    assert all(layer.state_dict(keep_vars=True)[key] is value for key, value in original.items())


def test_int8_embedding_state_residency_roundtrip_and_prefetch() -> None:
    layer = _int8_embedding()
    original = dict(layer.state_dict(keep_vars=True))
    store = ModuleStateStore(layer)
    assert set(store) == {"weight"}
    assert store.protected_quantization_keys() == {"weight_scale"}
    stored = store["weight"]
    assert isinstance(stored, Int8PackedWeight)
    assert stored.qdata is original["weight"]
    assert stored.scale is original["weight_scale"]
    indices = torch.tensor([[1, 7, 4], [10, 0, 3]])
    expected = layer(indices)
    mechanism = enroll_component(layer, load_device=CPU, offload_device=CPU)
    prefetch = layer.residency_prefetch()
    assert prefetch is not None
    assert prefetch[0] is mechanism
    assert prefetch[1] == (("weight", None),)
    assert torch.equal(layer(indices), expected)
    mechanism.partially_load(None)
    assert layer.residency_prefetch() is None
    assert torch.equal(layer(indices), expected)
    mechanism.unload()
    assert torch.equal(layer(indices), expected)
    assert all(layer.state_dict(keep_vars=True)[key] is value for key, value in original.items())


def test_int8_scale_patch_refuses_before_mechanism_mutation() -> None:
    layer = _int8_layer()
    original = {key: value for key, value in layer.state_dict(keep_vars=True).items()}
    called = False

    def factory(*_args: object, **_kwargs: object) -> ResidentWeights:
        nonlocal called
        called = True
        raise AssertionError("mechanism must not be constructed")

    patch = PatchSet({"weight_scale": (PatchEntry(DiffPatch(torch.tensor(0.0))),)})
    with pytest.raises(
        PatchApplyError,
        match="packed quantization-state patch/requantization",
    ):
        enroll_component(
            layer,
            load_device=CPU,
            offload_device=CPU,
            patch_set=patch,
            mechanism_factory=factory,
        )
    assert not called
    for key, value in original.items():
        assert layer.state_dict(keep_vars=True)[key] is value


def test_int8_logical_weight_patch_requantizes_and_restores_exact_state() -> None:
    layer = _int8_layer()
    original = dict(layer.state_dict(keep_vars=True))
    before = ModuleStateStore(layer)["weight"]
    assert isinstance(before, Int8PackedWeight)
    delta = torch.linspace(-0.01, 0.01, before.qdata.numel()).reshape(before.shape)
    patch_set = PatchSet({"weight": (PatchEntry(DiffPatch(delta)),)})
    mechanism = enroll_component(
        layer,
        load_device=CPU,
        offload_device=CPU,
        patch_set=patch_set,
    )

    mechanism.partially_load(None)
    actual = ModuleStateStore(layer)["weight"]
    assert isinstance(actual, Int8PackedWeight)
    expected_float = apply_patches(
        before.dequantize(torch.float32),
        patch_set.entries("weight"),
        key="weight",
        original_weight=before.dequantize(torch.float32),
    )
    expected = requantize_int8(before, expected_float, seed=string_to_seed("weight"))
    assert torch.equal(actual.qdata, expected.qdata)
    assert torch.equal(actual.scale, expected.scale)

    mechanism.unload()
    assert all(layer.state_dict(keep_vars=True)[key] is value for key, value in original.items())


def test_int8_convrot_accepts_lora_weight_and_bias_patches() -> None:
    layer = _int8_layer()
    generator = torch.Generator().manual_seed(20260822)
    before = ModuleStateStore(layer)["weight"]
    assert isinstance(before, Int8PackedWeight)
    up = torch.randn((7, 4), generator=generator, dtype=torch.bfloat16)
    down = torch.randn((4, 256), generator=generator, dtype=torch.bfloat16)
    diff_b = torch.randn((7,), generator=generator, dtype=torch.bfloat16)
    bias = layer.bias
    assert bias is not None
    patch_set = PatchSet(
        {
            "weight": (PatchEntry(AdapterPatch(LoRAAdapter(up, down))),),
            "bias": (PatchEntry(DiffPatch(diff_b)),),
        }
    )
    prefix = "diffusion_model.blocks.0.cross_attn.k."
    weight_key = f"{prefix}weight"
    bias_key = f"{prefix}bias"
    expected_float = apply_patches(
        before.dequantize(torch.float16),
        patch_set.entries("weight"),
        key=weight_key,
        intermediate_dtype=torch.float32,
        original_weight=before.dequantize(torch.float16),
    )
    expected = requantize_int8(before, expected_float, seed=string_to_seed(weight_key))
    expected_bias = apply_patches(
        bias.to(torch.float16, copy=True),
        patch_set.entries("bias"),
        key=bias_key,
        intermediate_dtype=torch.float32,
        original_weight=bias,
    ).to(bias.dtype)
    old_float = apply_patches(
        before.dequantize(torch.float32),
        patch_set.entries("weight"),
        key="weight",
        intermediate_dtype=torch.float32,
        original_weight=before.dequantize(torch.float32),
    )
    old = requantize_int8(before, old_float, seed=string_to_seed("weight"))
    assert not torch.equal(expected.qdata, old.qdata)

    mechanism = enroll_component(
        layer,
        load_device=CPU,
        offload_device=CPU,
        patch_set=patch_set,
        patch_weight_dtype=torch.float16,
        patch_key_prefix=prefix,
    )
    mechanism.partially_load(None)
    actual = ModuleStateStore(layer)["weight"]
    assert isinstance(actual, Int8PackedWeight)
    assert torch.equal(actual.qdata, expected.qdata)
    assert torch.equal(actual.scale, expected.scale)
    assert layer.bias is not None
    assert torch.equal(layer.bias, expected_bias)


def test_offloaded_patched_int8_requantizes_and_keeps_packed_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_kitchen  # pyright: ignore[reportMissingTypeStubs]

    layer = _int8_layer()
    before = ModuleStateStore(layer)["weight"]
    assert isinstance(before, Int8PackedWeight)
    delta = torch.linspace(-0.01, 0.01, layer.weight.numel()).reshape(layer.weight.shape)
    patch_set = PatchSet({"weight": (PatchEntry(DiffPatch(delta)),)})
    prefix = "diffusion_model.blocks.0.cross_attn.k."
    mechanism = enroll_component(
        layer,
        load_device=CPU,
        offload_device=CPU,
        patch_set=patch_set,
        patch_weight_dtype=torch.float16,
        patch_key_prefix=prefix,
    )
    prefetch = layer.residency_prefetch()
    assert prefetch is not None
    assert prefetch[1] == (
        ("weight", None),
        ("bias", torch.float32),
    )
    assert mechanism.weight_functions("weight") == ()
    with mechanism.lease("weight") as lease:
        actual = lease.get_stored("weight")
    assert isinstance(actual, Int8PackedWeight)
    expected_float = apply_patches(
        before.dequantize(torch.float16),
        patch_set.entries("weight"),
        key=f"{prefix}weight",
        intermediate_dtype=torch.float32,
        original_weight=before.dequantize(torch.float16),
    )
    expected = requantize_int8(
        before,
        expected_float,
        seed=string_to_seed(f"{prefix}weight"),
    )
    assert torch.equal(actual.qdata, expected.qdata)
    assert torch.equal(actual.scale, expected.scale)
    stored = ModuleStateStore(layer)["weight"]
    assert isinstance(stored, Int8PackedWeight)
    assert torch.equal(before.qdata, stored.qdata)
    input = torch.randn(5, 256)
    expected_output = dinkster_kitchen.int8_linear(
        input,
        actual.qdata,
        actual.scale,
        layer.bias,
        out_dtype=layer.compute_dtype,
        convrot=True,
        convrot_groupsize=256,
    )

    def reject_float_linear(*_args: object, **_kwargs: object) -> torch.Tensor:
        pytest.fail("patched INT8 took the float linear path")

    monkeypatch.setattr(torch.nn.functional, "linear", reject_float_linear)
    assert torch.equal(layer(input), expected_output)


@pytest.mark.parametrize(
    "patch_key",
    [
        "weight_scale",
        "weight_scale_2",
        "input_scale",
        "pre_quant_scale",
    ],
)
def test_nvfp4_quantization_patch_refuses_before_mechanism_mutation(
    patch_key: str,
) -> None:
    layer = _nvfp4_layer()
    original = {key: value for key, value in layer.state_dict(keep_vars=True).items()}
    called = False

    def factory(*_args: object, **_kwargs: object) -> ResidentWeights:
        nonlocal called
        called = True
        raise AssertionError("mechanism must not be constructed")

    patch = PatchSet({patch_key: (PatchEntry(DiffPatch(torch.tensor(0.0))),)})
    with pytest.raises(
        PatchApplyError,
        match="packed quantization-state patch/requantization",
    ):
        enroll_component(
            layer,
            load_device=CPU,
            offload_device=CPU,
            patch_set=patch,
            mechanism_factory=factory,
        )
    assert not called
    for key, value in original.items():
        assert layer.state_dict(keep_vars=True)[key] is value


def test_nvfp4_logical_weight_patch_reaches_mechanism() -> None:
    layer = _nvfp4_layer()
    diagnostics = Nvfp4DiagnosticsRecorder()
    layer._bind_diagnostics(diagnostics)  # pyright: ignore[reportPrivateUsage]
    original = dict(layer.state_dict(keep_vars=True))
    mechanism = enroll_component(
        layer,
        load_device=CPU,
        offload_device=CPU,
        patch_set=PatchSet({"weight": (PatchEntry(DiffPatch(torch.zeros(16, 16))),)}),
    )
    mechanism.partially_load(None)
    stored = ModuleStateStore(layer)["weight"]
    assert isinstance(stored, Nvfp4PackedWeight)
    assert stored.recorder is diagnostics
    assert diagnostics.snapshot().lifetime["requantize_success"] == 1
    mechanism.unload()
    assert all(layer.state_dict(keep_vars=True)[key] is value for key, value in original.items())


def test_nvfp4_bias_patch_remains_allowed() -> None:
    layer = _nvfp4_layer()
    bias_patch = PatchSet({"bias": (PatchEntry(DiffPatch(torch.full((16,), 0.125))),)})
    mechanism = enroll_component(
        layer,
        load_device=CPU,
        offload_device=CPU,
        patch_set=bias_patch,
    )
    mechanism.partially_load(None)
    assert mechanism.is_loaded("")


def test_assembled_nvfp4_scale_patch_refuses_before_any_mechanism_factory() -> None:
    factory = _RecordingFactory()
    assembled = _policy_assembled(
        _policy_linear(),
        _policy_linear(),
        _nvfp4_layer(),
        enabled=False,
    )
    patch = PatchSet({"weight_scale": (PatchEntry(DiffPatch(torch.zeros(16, 16))),)})
    with pytest.raises(
        PatchApplyError,
        match="packed quantization-state patch/requantization",
    ):
        enroll_assembled(
            assembled,
            load_device=CPU,
            offload_device=CPU,
            patch_sets={"vae": patch},
            mechanism_factory=factory,
        )
    assert factory.calls == 0


def test_offloaded_patched_fp8_forces_the_dequant_route() -> None:
    layer = _fp8_layer()
    delta = torch.full((3, 4), 0.125)
    mechanism = enroll_component(
        layer,
        load_device=CPU,
        offload_device=CPU,
        patch_set=PatchSet({"weight": (PatchEntry(DiffPatch(delta)),)}),
    )
    # Capability checks guard normal hardware enablement. Set the bound
    # policy directly here: an offloaded patched weight must bypass the
    # hardware branch before torch._scaled_mm can be reached on CPU.
    layer.fp8_matmul = True
    input = torch.randn(5, 4)
    expected = torch.nn.functional.linear(
        input,
        mechanism.use("weight", dtype=layer.compute_dtype),
        mechanism.use("bias", dtype=layer.compute_dtype),
    )
    assert torch.equal(layer(input), expected)


def test_state_dict_layout_survives_partial_unload_and_reload() -> None:
    original = torch.nn.Sequential(
        INITLESS.linear(4, 4),
        INITLESS.layer_norm(4),
        INITLESS.linear(4, 2),
    )
    enrolled = torch.nn.Sequential(
        INITLESS.linear(4, 4),
        INITLESS.layer_norm(4),
        INITLESS.linear(4, 2),
    )
    _fill(original)
    enrolled.load_state_dict(original.state_dict(), strict=True, assign=True)
    mechanism = enroll_component(enrolled, load_device=CPU, offload_device=CPU)
    mechanism.partially_load(None)
    mechanism.partially_unload(1)
    mechanism.partially_load(None)

    expected = original.state_dict()
    actual = enrolled.state_dict()
    assert set(actual) == set(expected)
    for key, value in actual.items():
        assert value.shape == expected[key].shape


def _policy_linear(
    storage_dtype: torch.dtype = torch.float32,
    *,
    compute_dtype: torch.dtype = torch.float16,
) -> torch.nn.Linear:
    layer = CastOperations(compute_dtype).linear(4, 4)
    _fill(layer, storage_dtype)
    return layer


def _policy_assembled(
    diffusion: torch.nn.Module,
    clip_l: torch.nn.Module,
    vae: torch.nn.Module,
    *,
    targets: dict[str, torch.dtype] | None = None,
    enabled: bool = True,
) -> AssembledSD:
    return AssembledSD(
        family=SD15,
        diffusion=cast("UNetModel", diffusion),
        clip_l=cast("ClipTextModel", clip_l),
        clip_g=None,
        vae=cast("AutoencoderKL", vae),
        _storage_dtype_follows_compute=enabled,
        _component_compute_dtypes=(
            {"diffusion": torch.float16, "clip_l": torch.float16, "vae": torch.float16}
            if targets is None
            else targets
        ),
    )


def _state_bytes(
    module: torch.nn.Module,
) -> dict[str, tuple[torch.Tensor, torch.dtype, torch.Tensor]]:
    return {
        key: (
            value,
            value.dtype,
            value.detach().reshape(-1).contiguous().view(torch.uint8).clone(),
        )
        for key, value in module.state_dict(keep_vars=True).items()
    }


def _assert_state_unchanged(
    module: torch.nn.Module,
    before: dict[str, tuple[torch.Tensor, torch.dtype, torch.Tensor]],
) -> None:
    after = module.state_dict(keep_vars=True)
    assert after.keys() == before.keys()
    for key, (original, dtype, data) in before.items():
        assert after[key] is original, key
        assert after[key].dtype == dtype, key
        assert torch.equal(after[key].reshape(-1).contiguous().view(torch.uint8), data), key


class _RecordingFactory:
    def __init__(self) -> None:
        self.calls = 0
        self.storage_dtypes: list[set[torch.dtype]] = []
        self.patch_sets: list[PatchSet[torch.Tensor] | None] = []
        self.modules: list[torch.nn.Module] = []

    def __call__(
        self,
        weights: MutableMapping[str, StoredWeight],
        *,
        load_device: torch.device | str,
        offload_device: torch.device | str,
        patch_set: PatchSet[torch.Tensor] | None = None,
        units: Sequence[ResidencyUnit] | None = None,
        intermediate_dtype: torch.dtype = torch.float32,
    ) -> ResidentWeights:
        self.calls += 1
        self.modules.append(cast(Any, weights).module)
        self.patch_sets.append(patch_set)
        self.storage_dtypes.append(
            {
                stored.qdata.dtype
                if isinstance(stored, Fp8ScaledWeight | Int8PackedWeight | Nvfp4PackedWeight)
                else stored.dtype
                for stored in weights.values()
            }
        )
        return ResidentWeights(
            weights,
            load_device=load_device,
            offload_device=offload_device,
            patch_set=patch_set,
            units=units,
            intermediate_dtype=intermediate_dtype,
        )


def test_storage_dtype_policy_disabled_report_and_state_are_exact() -> None:
    components = tuple(_policy_linear() for _ in range(3))
    before = tuple(_state_bytes(component) for component in components)
    enrolled = enroll_assembled(
        _policy_assembled(*components, enabled=False),
        load_device=CPU,
        offload_device=CPU,
    )

    assert not enrolled.storage_dtype_report.enabled
    assert dict(enrolled.storage_dtype_report.outcomes) == {}
    for component, snapshot in zip(components, before, strict=True):
        _assert_state_unchanged(component, snapshot)


@pytest.mark.parametrize("target", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_explicit_diffusion_storage_dtype_converts_only_diffusion(target: torch.dtype) -> None:
    diffusion = _policy_linear()
    clip = _policy_linear()
    vae = _policy_linear()
    factory = _RecordingFactory()

    enrolled = enroll_assembled(
        _policy_assembled(diffusion, clip, vae, enabled=False),
        load_device=CPU,
        offload_device=CPU,
        storage_dtypes={"diffusion": target},
        mechanism_factory=factory,
    )

    assert enrolled.storage_dtype_report.enabled
    assert dict(enrolled.storage_dtype_report.outcomes) == {"diffusion": "converted"}
    assert {parameter.dtype for parameter in diffusion.parameters()} == {target}
    assert {parameter.dtype for parameter in clip.parameters()} == {torch.float32}
    assert {parameter.dtype for parameter in vae.parameters()} == {torch.float32}
    assert factory.storage_dtypes == [{target}, {torch.float32}, {torch.float32}]


def test_storage_dtype_policy_token_set_is_closed_and_exact() -> None:
    assert STORAGE_DTYPE_POLICY_TOKENS == (
        "converted",
        "already_at_target",
        "no_floating_state",
        "quantized",
        "mixed_floating",
        "storage_widening",
        "unsupported_dtype",
        "unrouted_state",
        "unmanaged_buffer",
    )


def test_storage_dtype_policy_reports_no_floating_state() -> None:
    integer_only = _policy_linear()
    integer_only.register_parameter("weight", None)
    integer_only.register_parameter("bias", None)
    integer_only.register_buffer("sentinel", torch.tensor([11], dtype=torch.int64))
    enrolled = enroll_assembled(
        _policy_assembled(integer_only, _policy_linear(), _policy_linear()),
        load_device=CPU,
        offload_device=CPU,
    )
    assert enrolled.storage_dtype_report.outcomes["diffusion"] == "no_floating_state"
    assert cast(torch.Tensor, integer_only.sentinel).item() == 11


@pytest.mark.parametrize("target", [torch.float16, torch.bfloat16])
def test_storage_dtype_policy_reports_and_converts_every_eligible_component(
    target: torch.dtype,
) -> None:
    components = tuple(_policy_linear(compute_dtype=target) for _ in range(3))
    expected = tuple(
        {
            name: parameter.detach().to(target).clone()
            for name, parameter in component.named_parameters()
        }
        for component in components
    )
    targets = {"diffusion": target, "clip_l": target, "vae": target}
    factory = _RecordingFactory()
    enrolled = enroll_assembled(
        _policy_assembled(*components, targets=targets),
        load_device=CPU,
        offload_device=CPU,
        mechanism_factory=factory,
    )

    assert enrolled.storage_dtype_report.enabled
    assert dict(enrolled.storage_dtype_report.outcomes) == {
        "diffusion": "converted",
        "clip_l": "converted",
        "vae": "converted",
    }
    assert factory.storage_dtypes == [{target}, {target}, {target}]
    assert factory.modules == list(components)
    for component, component_expected in zip(components, expected, strict=True):
        parameters = dict(component.named_parameters())
        assert parameters.keys() == component_expected.keys()
        for name, value in component_expected.items():
            assert parameters[name].dtype == target
            assert torch.equal(parameters[name].view(torch.uint8), value.view(torch.uint8)), name


@pytest.mark.parametrize(
    ("storage", "target"),
    [
        (torch.float16, torch.bfloat16),
        (torch.bfloat16, torch.float16),
    ],
)
def test_storage_dtype_policy_converts_between_supported_floating_dtypes(
    storage: torch.dtype,
    target: torch.dtype,
) -> None:
    components = tuple(_policy_linear(storage, compute_dtype=target) for _ in range(3))
    expected = tuple(
        {
            name: parameter.detach().to(target).clone()
            for name, parameter in component.named_parameters()
        }
        for component in components
    )
    factory = _RecordingFactory()
    enrolled = enroll_assembled(
        _policy_assembled(
            *components,
            targets={"diffusion": target, "clip_l": target, "vae": target},
        ),
        load_device=CPU,
        offload_device=CPU,
        mechanism_factory=factory,
    )

    assert set(enrolled.storage_dtype_report.outcomes.values()) == {"converted"}
    assert factory.storage_dtypes == [{target}, {target}, {target}]
    for component, component_expected in zip(components, expected, strict=True):
        for name, value in component.named_parameters():
            assert value.dtype == target
            assert torch.equal(value.view(torch.uint8), component_expected[name].view(torch.uint8))


@pytest.mark.parametrize("storage", [torch.float16, torch.bfloat16])
def test_storage_dtype_policy_preserves_storage_for_wider_compute(
    storage: torch.dtype,
) -> None:
    components = tuple(_policy_linear(storage, compute_dtype=torch.float32) for _ in range(3))
    before = tuple(_state_bytes(component) for component in components)
    input = torch.randn((2, 4), generator=torch.Generator().manual_seed(89))
    expected = tuple(component(input) for component in components)
    factory = _RecordingFactory()

    enrolled = enroll_assembled(
        _policy_assembled(
            *components,
            targets={
                "diffusion": torch.float32,
                "clip_l": torch.float32,
                "vae": torch.float32,
            },
        ),
        load_device=CPU,
        offload_device=CPU,
        mechanism_factory=factory,
    )

    assert set(enrolled.storage_dtype_report.outcomes.values()) == {"storage_widening"}
    assert factory.storage_dtypes == [{storage}, {storage}, {storage}]
    for component, snapshot, output in zip(components, before, expected, strict=True):
        _assert_state_unchanged(component, snapshot)
        assert torch.equal(component(input), output)


def test_storage_dtype_policy_converts_each_route_to_its_compute_dtype() -> None:
    diffusion = _mixed_compute_toy()
    patch_weight = diffusion.patch.weight
    expected = diffusion(
        torch.randn((2, 4), generator=torch.Generator().manual_seed(83)).bfloat16()
    )
    factory = _RecordingFactory()
    enrolled = enroll_assembled(
        _policy_assembled(
            diffusion,
            _policy_linear(),
            _policy_linear(),
            targets={
                "diffusion": torch.bfloat16,
                "clip_l": torch.float16,
                "vae": torch.float16,
            },
        ),
        load_device=CPU,
        offload_device=CPU,
        mechanism_factory=factory,
    )

    assert enrolled.storage_dtype_report.outcomes["diffusion"] == "converted"
    assert diffusion.patch.weight is patch_weight
    assert diffusion.patch.weight.dtype is torch.float32
    assert diffusion.body.weight.dtype is torch.bfloat16
    assert diffusion.modulation.dtype is torch.bfloat16
    assert factory.storage_dtypes[0] == {torch.float32, torch.bfloat16}
    actual = diffusion(torch.randn((2, 4), generator=torch.Generator().manual_seed(83)).bfloat16())
    assert torch.equal(actual, expected)


def test_storage_dtype_policy_patches_mixed_compute_routes_before_final_cast() -> None:
    diffusion = _mixed_compute_toy()
    original_patch = diffusion.patch.weight.detach().clone()
    original_body = diffusion.body.weight.detach().clone()
    patch_delta = torch.full_like(original_patch, 0.125)
    body_delta = torch.full_like(original_body, 0.0625)
    patches = PatchSet(
        {
            "patch.weight": (PatchEntry(DiffPatch(patch_delta)),),
            "body.weight": (PatchEntry(DiffPatch(body_delta)),),
        }
    )
    enrolled = enroll_assembled(
        _policy_assembled(
            diffusion,
            _policy_linear(),
            _policy_linear(),
            targets={
                "diffusion": torch.bfloat16,
                "clip_l": torch.float16,
                "vae": torch.float16,
            },
        ),
        load_device=CPU,
        offload_device=CPU,
        patch_sets={"diffusion": patches},
    )

    assert enrolled.storage_dtype_report.outcomes["diffusion"] == "converted"
    assert diffusion.patch.weight.dtype is torch.float32
    assert diffusion.body.weight.dtype is torch.bfloat16
    assert torch.equal(diffusion.patch.weight, original_patch.float() + patch_delta.float())
    assert torch.equal(
        diffusion.body.weight,
        (original_body.float() + body_delta.float()).to(torch.bfloat16),
    )
    assert enrolled["diffusion"].weight_functions("patch.weight") == ()
    assert enrolled["diffusion"].weight_functions("body.weight") == ()


def test_h3_peft_lora_materializes_expected_scaled_delta(tmp_path: Path) -> None:
    from test_autoencoder_kl import write_safetensors

    class SmallH3DiT(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = INITLESS.linear(3, 2, bias=False)

    diffusion = SmallH3DiT()
    diffusion.layer.weight.data.copy_(torch.tensor([[0.5, -0.25, 0.75], [-0.5, 0.125, 0.25]]))
    original = diffusion.layer.weight.detach().clone()
    down = torch.tensor([[1.0, 2.0, -1.0], [0.5, -0.25, 1.5]])
    up = torch.tensor([[2.0, -1.0], [0.25, 3.0]])
    prefix = "diffusion_model.layer"
    path = write_safetensors(
        tmp_path / "h3-peft.safetensors",
        {
            f"{prefix}.lora_A.weight": down,
            f"{prefix}.lora_B.weight": up,
            f"{prefix}.alpha": torch.tensor(4.0),
        },
    )
    source = load_safetensors_header(path)
    geometries = {key: source.entry(key).geometry for key in source.keys()}
    tensors = load_tensors(path)
    assert set(tensors) == {
        f"{prefix}.lora_A.weight",
        f"{prefix}.lora_B.weight",
        f"{prefix}.alpha",
    }
    decoded = decode_lora(
        geometries,
        native_unet_key_map(("diffusion_model.layer.weight",)),
    )
    assert decoded.dialect == "none"
    specs = tuple(decoded.patches.values())
    assert len(specs) == 1 and isinstance(specs[0], LoRASpec)
    assert specs[0].variant == "peft"
    routed = {
        PatchTarget(target.key.removeprefix("diffusion_model."), target.offset): spec
        for target, spec in decoded.patches.items()
    }
    patch_set = build_patch_set(routed, tensors, strength=0.25)

    enrolled = enroll_assembled(
        AssembledMiniMaxH3Model(
            cast("Any", diffusion),
            _component_compute_dtypes={"diffusion": torch.float32},
        ),
        load_device=CPU,
        offload_device=CPU,
        patch_sets={"diffusion": patch_set},
    )
    enrolled["diffusion"].partially_load(None)

    expected = original + (up @ down) * (4.0 / down.shape[0]) * 0.25
    assert torch.equal(diffusion.layer.weight, expected)


def test_storage_dtype_policy_preserves_declared_runtime_constant_buffers() -> None:
    diffusion = _policy_linear(torch.bfloat16, compute_dtype=torch.float32)
    constant = torch.tensor([1.25], dtype=torch.float32)
    diffusion.register_buffer("constant", constant, persistent=False)
    diffusion.__dict__["_dinkster_residency_constant_buffers"] = frozenset({"constant"})

    enrolled = enroll_assembled(
        _policy_assembled(
            diffusion,
            _policy_linear(),
            _policy_linear(),
            targets={
                "diffusion": torch.float32,
                "clip_l": torch.float16,
                "vae": torch.float16,
            },
        ),
        load_device=CPU,
        offload_device=CPU,
    )

    assert enrolled.storage_dtype_report.outcomes["diffusion"] == "storage_widening"
    assert cast(Any, diffusion).constant is constant


def _ineligible_component(outcome: str) -> torch.nn.Module:
    if outcome == "mixed_floating":
        return _ConflictingTargetTiedToy()
    if outcome == "quantized":
        return _fp8_layer()
    if outcome == "unsupported_dtype":
        return _policy_linear(torch.float64)
    raise AssertionError(outcome)


@pytest.mark.parametrize("outcome", ["mixed_floating", "quantized", "unsupported_dtype"])
def test_storage_dtype_policy_skips_ineligible_component_and_converts_sibling(
    outcome: str,
) -> None:
    diffusion = _policy_linear()
    skipped = _ineligible_component(outcome)
    vae = _policy_linear(torch.float16)
    skipped_before = _state_bytes(skipped)
    enrolled = enroll_assembled(
        _policy_assembled(diffusion, skipped, vae),
        load_device=CPU,
        offload_device=CPU,
    )

    assert dict(enrolled.storage_dtype_report.outcomes) == {
        "diffusion": "converted",
        "clip_l": outcome,
        "vae": "already_at_target",
    }
    assert {parameter.dtype for parameter in diffusion.parameters()} == {torch.float16}
    _assert_state_unchanged(skipped, skipped_before)


@pytest.mark.parametrize("target", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["diff", "diff_pad", "set", "model", "nested", "ordered_offset"])
def test_storage_dtype_policy_patches_fp32_then_casts_once(target: torch.dtype, kind: str) -> None:
    components = tuple(_policy_linear(compute_dtype=target) for _ in range(3))
    if kind == "nested":
        base_value = 0.0010030090270812437
        delta_value = 0.008072653884964682 if target is torch.float16 else 0.022199798183652877
    else:
        base_value, delta_value = (
            (1.0001, 0.1)
            if target is torch.float16
            else (0.0010030090270812437, 0.033299697275479316)
        )
    components[0].weight.data.fill_(base_value)
    original = components[0].weight.detach().clone()
    delta = torch.full(
        (5, 4) if kind == "diff_pad" else original.shape,
        delta_value,
        dtype=original.dtype,
    )
    if kind == "diff":
        entry = PatchEntry(DiffPatch(delta))
        entries = (entry,)
    elif kind == "diff_pad":
        entry = PatchEntry(DiffPatch(delta, pad_weight=True))
        entries = (entry,)
    elif kind == "set":
        entry = PatchEntry(SetPatch(original + delta))
        entries = (entry,)
    elif kind == "model":
        entry = PatchEntry(ModelAsLoraPatch(original + delta))
        entries = (entry,)
    elif kind == "nested":
        entry = PatchEntry(
            NestedPatch(
                original,
                (PatchEntry(DiffPatch(delta)),),
                convert=_double_patch_value,
            )
        )
        entries = (entry,)
    else:
        entry = PatchEntry(
            DiffPatch(delta[:2]),
            strength=0.5,
            strength_model=0.75,
            offset=PatchOffset(0, 1, 2),
            function=torch.neg,
        )
        entries = (entry, PatchEntry(DiffPatch(torch.full_like(original, 0.01))))
    patch_set = PatchSet({"weight": entries})
    revision = patch_set.revision
    payload = delta.clone()
    oracle = apply_patches(
        original.to(torch.float32, copy=True),
        patch_set.entries("weight"),
        key="weight",
        original_weight=original,
    ).to(target)
    factory = _RecordingFactory()

    enrolled = enroll_assembled(
        _policy_assembled(
            *components,
            targets={"diffusion": target, "clip_l": target, "vae": target},
        ),
        load_device=CPU,
        offload_device=CPU,
        patch_sets={"diffusion": patch_set},
        mechanism_factory=factory,
    )

    assert dict(enrolled.storage_dtype_report.outcomes) == {
        "diffusion": "converted",
        "clip_l": "converted",
        "vae": "converted",
    }
    assert torch.equal(components[0].weight, oracle)
    if kind in {"diff", "model", "nested"}:
        cast_first = apply_patches(
            original.to(target).to(torch.float32),
            patch_set.entries("weight"),
            key="weight",
            original_weight=original,
        ).to(target)
        assert not torch.equal(oracle, cast_first)
    assert factory.patch_sets == [None, None, None]
    assert enrolled["diffusion"].weight_functions("weight") == ()
    assert patch_set.revision == revision
    assert torch.equal(delta, payload)


def _adapter_patch_cases() -> list[tuple[str, AdapterPatch[torch.Tensor]]]:
    generator = torch.Generator().manual_seed(811)

    def tensor(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator)

    return [
        ("lora", AdapterPatch(LoRAAdapter(tensor(4, 2), tensor(2, 4)))),
        (
            "loha",
            AdapterPatch(LoHaAdapter(tensor(4, 2), tensor(2, 4), tensor(4, 2), tensor(2, 4))),
        ),
        (
            "lokr",
            AdapterPatch(LoKrAdapter(w1=tensor(2, 2), w2=tensor(2, 2))),
        ),
        (
            "glora",
            AdapterPatch(GLoRAAdapter(tensor(2, 4), tensor(4, 2), tensor(2, 4), tensor(4, 2))),
        ),
        ("oft", AdapterPatch(OFTAdapter(tensor(1, 4, 4)))),
        ("boft", AdapterPatch(BOFTAdapter(tensor(1, 1, 4, 4)))),
    ]


@pytest.mark.parametrize("target", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("kind", "value"),
    _adapter_patch_cases(),
    ids=[case[0] for case in _adapter_patch_cases()],
)
def test_storage_dtype_policy_converts_every_adapter_kind_without_deferral(
    target: torch.dtype, kind: str, value: AdapterPatch[torch.Tensor]
) -> None:
    components = tuple(_policy_linear(compute_dtype=target) for _ in range(3))
    original = components[0].weight.detach().clone()
    entry = PatchEntry(value, strength=0.375, strength_model=0.875)
    patch_set = PatchSet({"weight": (entry,)}, structural_digest="a" * 64)
    revision = patch_set.revision
    payloads = tuple(payload.detach().clone() for payload in patch_payloads((entry,)))
    expected = apply_patches(
        original.to(torch.float32, copy=True),
        patch_set.entries("weight"),
        key="weight",
        intermediate_dtype=torch.float32,
        original_weight=original,
    ).to(target)
    factory = _RecordingFactory()

    enrolled = enroll_assembled(
        _policy_assembled(
            *components,
            targets={"diffusion": target, "clip_l": target, "vae": target},
        ),
        load_device=CPU,
        offload_device=CPU,
        patch_sets={"diffusion": patch_set},
        mechanism_factory=factory,
    )

    assert torch.equal(components[0].weight, expected), kind
    assert factory.patch_sets == [None, None, None]
    assert enrolled["diffusion"].weight_functions("weight") == ()
    enrolled["diffusion"].partially_load(None)
    enrolled["diffusion"].unload()
    assert torch.equal(components[0].weight, expected), kind
    assert patch_set.revision == revision
    assert patch_set.structural_digest == "a" * 64
    assert all(
        torch.equal(payload, before)
        for payload, before in zip(patch_payloads((entry,)), payloads, strict=True)
    )
    replay = cast(PatchSet[torch.Tensor], components[0].__dict__["_dinkster_base_patch_set"])
    replay_payloads = patch_payloads(replay.entries("weight"))
    assert all(payload.device.type == "cpu" for payload in replay_payloads)
    with torch.no_grad():
        for payload in patch_payloads((entry,)):
            payload.add_(1)
    assert all(
        torch.equal(payload, before)
        for payload, before in zip(replay_payloads, payloads, strict=True)
    )


def test_storage_dtype_policy_tied_patch_applies_once_and_preserves_identity() -> None:
    tied = _TiedToy()
    original = tied.first.weight.detach().clone()
    delta = torch.ones_like(original)
    patch_set = PatchSet({"first.weight": (PatchEntry(DiffPatch(delta)),)})
    enrolled = enroll_assembled(
        _policy_assembled(tied, _policy_linear(), _policy_linear()),
        load_device=CPU,
        offload_device=CPU,
        patch_sets={"diffusion": patch_set},
    )
    assert enrolled.storage_dtype_report.outcomes["diffusion"] == "converted"
    assert tied.first.weight is tied.second.weight
    assert torch.equal(tied.first.weight, (original + delta).to(torch.float16))


def test_storage_dtype_policy_tied_identity_survives_storage_moves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tied = _TiedToy()
    enrolled = enroll_assembled(
        _policy_assembled(tied, _policy_linear(), _policy_linear()),
        load_device=CPU,
        offload_device=CPU,
    )
    moves = 0

    def cloning_move(
        stored: StoredWeight,
        _device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> StoredWeight:
        nonlocal moves
        assert non_blocking is False
        moves += 1
        assert isinstance(stored, torch.Tensor)
        return stored.detach().clone()

    monkeypatch.setattr(residency_mod, "move_stored", cloning_move)
    mechanism = enrolled["diffusion"]
    mechanism.partially_load(None)
    assert tied.first.weight is tied.second.weight
    mechanism.unload()
    assert tied.first.weight is tied.second.weight
    assert moves > 0


def test_storage_dtype_policy_refuses_ambiguous_tied_and_overlapping_views() -> None:
    tied = _TiedToy()
    ambiguous = PatchSet(
        {
            "first.weight": (PatchEntry(DiffPatch(torch.ones(4, 4))),),
            "second.weight": (PatchEntry(DiffPatch(torch.ones(4, 4))),),
        }
    )
    with pytest.raises(PatchApplyError, match="ambiguous tied"):
        enroll_assembled(
            _policy_assembled(tied, _policy_linear(), _policy_linear()),
            load_device=CPU,
            offload_device=CPU,
            patch_sets={"diffusion": ambiguous},
        )

    overlapping = _policy_linear()
    base = torch.arange(20, dtype=torch.float32)
    overlapping.weight = torch.nn.Parameter(base[:16].reshape(4, 4), requires_grad=False)
    overlapping.bias = torch.nn.Parameter(base[4:8], requires_grad=False)
    with pytest.raises(PatchApplyError, match="overlapping storage"):
        enroll_assembled(
            _policy_assembled(overlapping, _policy_linear(), _policy_linear()),
            load_device=CPU,
            offload_device=CPU,
        )


def test_storage_dtype_policy_allows_disjoint_shared_storage_views() -> None:
    disjoint = _policy_linear()
    base = torch.arange(32, dtype=torch.float32)
    disjoint.weight = torch.nn.Parameter(base[::2].reshape(4, 4), requires_grad=False)
    disjoint.bias = torch.nn.Parameter(base[1:8:2], requires_grad=False)
    expected_weight = disjoint.weight.detach().to(torch.float16)
    expected_bias = disjoint.bias.detach().to(torch.float16)

    enrolled = enroll_assembled(
        _policy_assembled(disjoint, _policy_linear(), _policy_linear()),
        load_device=CPU,
        offload_device=CPU,
    )

    assert enrolled.storage_dtype_report.outcomes["diffusion"] == "converted"
    assert torch.equal(disjoint.weight, expected_weight)
    assert torch.equal(disjoint.bias, expected_bias)


def test_storage_overlap_preflight_is_bounded_for_large_expanded_views() -> None:
    backing = torch.ones(2)
    left = backing[:1].expand(1_000_000_000)
    disjoint = backing[1:].expand(1_000_000_000)
    overlapping = backing[:1].expand(1_000_000_000)

    left_region = module_residency_mod._occupied_bytes(left)  # pyright: ignore[reportPrivateUsage]
    disjoint_region = module_residency_mod._occupied_bytes(disjoint)  # pyright: ignore[reportPrivateUsage]
    overlapping_region = module_residency_mod._occupied_bytes(overlapping)  # pyright: ignore[reportPrivateUsage]

    assert not module_residency_mod._occupied_bytes_overlap(  # pyright: ignore[reportPrivateUsage]
        left_region, disjoint_region
    )
    assert module_residency_mod._occupied_bytes_overlap(  # pyright: ignore[reportPrivateUsage]
        left_region, overlapping_region
    )


def test_storage_dtype_policy_refuses_overlapping_offset_storages() -> None:
    overlapping = _policy_linear()
    backing = bytearray(80)
    weight = torch.frombuffer(backing, dtype=torch.float32, count=16, offset=0)
    bias = torch.frombuffer(backing, dtype=torch.float32, count=4, offset=48)
    assert weight.untyped_storage().data_ptr() != bias.untyped_storage().data_ptr()
    overlapping.weight = torch.nn.Parameter(weight.reshape(4, 4), requires_grad=False)
    overlapping.bias = torch.nn.Parameter(bias, requires_grad=False)

    with pytest.raises(PatchApplyError, match="overlapping storage"):
        enroll_assembled(
            _policy_assembled(overlapping, _policy_linear(), _policy_linear()),
            load_device=CPU,
            offload_device=CPU,
        )


def test_storage_dtype_policy_inert_patch_set_still_converts() -> None:
    components = tuple(_policy_linear() for _ in range(3))
    enrolled = enroll_assembled(
        _policy_assembled(*components),
        load_device=CPU,
        offload_device=CPU,
        patch_sets={"diffusion": PatchSet({"weight": ()})},
    )
    assert set(enrolled.storage_dtype_report.outcomes.values()) == {"converted"}


@pytest.mark.parametrize("reason", ["unrouted_state", "unmanaged_buffer"])
def test_storage_dtype_policy_refuses_anomalous_convertible_state_before_factory(
    reason: str,
) -> None:
    anomalous: torch.nn.Module
    if reason == "unrouted_state":
        anomalous = _StateToy()
    else:
        anomalous = _policy_linear()
        anomalous.register_buffer(
            "temporary", torch.tensor([5], dtype=torch.int64), persistent=False
        )
    siblings = (_policy_linear(), _policy_linear())
    before = (_state_bytes(anomalous), *(_state_bytes(item) for item in siblings))
    factory = _RecordingFactory()

    with pytest.raises(StorageDtypePolicyError) as caught:
        enroll_assembled(
            _policy_assembled(siblings[0], anomalous, siblings[1]),
            load_device=CPU,
            offload_device=CPU,
            mechanism_factory=factory,
        )

    assert caught.value.reason == reason
    assert factory.calls == 0
    for component, snapshot in zip((anomalous, *siblings), before, strict=True):
        _assert_state_unchanged(component, snapshot)


def test_storage_dtype_policy_conversion_failure_restores_all_registrations() -> None:
    first = _TiedToy()
    failing = _policy_linear()
    last = _policy_linear()
    last.register_buffer("scale", torch.tensor([1.25], dtype=torch.float32))
    components = (first, failing, last)
    before = tuple(_state_bytes(component) for component in components)
    factory = _RecordingFactory()

    class InjectedCancellation(BaseException):
        pass

    def cancel(_delta: torch.Tensor) -> torch.Tensor:
        raise InjectedCancellation("injected conversion cancellation")

    patch_set = PatchSet(
        {"weight": (PatchEntry(DiffPatch(torch.ones_like(failing.weight)), function=cancel),)}
    )

    with pytest.raises(InjectedCancellation, match="injected conversion cancellation"):
        enroll_assembled(
            _policy_assembled(*components),
            load_device=CPU,
            offload_device=CPU,
            patch_sets={"clip_l": patch_set},
            mechanism_factory=factory,
        )

    assert factory.calls == 0
    for component, snapshot in zip(components, before, strict=True):
        _assert_state_unchanged(component, snapshot)
    assert first.first.weight is first.second.weight


@pytest.mark.parametrize("enabled", [True, False])
def test_assembly_rejects_duplicate_component_before_enrollment(enabled: bool) -> None:
    shared = _policy_linear()
    factory = _RecordingFactory()
    with pytest.raises(ValueError, match="must not share a module instance"):
        enroll_assembled(
            _policy_assembled(shared, shared, _policy_linear(), enabled=enabled),
            load_device=CPU,
            offload_device=CPU,
            mechanism_factory=factory,
        )
    assert factory.calls == 0
    assert not hasattr(shared, "_dinkster_resident_weights")
    assert shared.residency_binding() is None  # type: ignore[attr-defined]


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("failure", [1, 2, 3])
@pytest.mark.parametrize("failure_error", [RuntimeError, asyncio.CancelledError, KeyboardInterrupt])
def test_storage_dtype_policy_factory_failure_rolls_back_whole_assembly(
    enabled: bool,
    failure: int,
    failure_error: type[BaseException],
) -> None:
    components = (
        _policy_linear(torch.float32),
        _policy_linear(torch.bfloat16),
        _policy_linear(torch.float16),
    )
    before = tuple(_state_bytes(component) for component in components)

    class TrackingResidentWeights(ResidentWeights):
        unload_calls = 0

        def unload(self) -> None:
            self.unload_calls += 1
            super().unload()

    constructed: list[TrackingResidentWeights] = []
    calls = 0

    def factory(
        weights: MutableMapping[str, StoredWeight],
        *,
        load_device: torch.device | str,
        offload_device: torch.device | str,
        patch_set: PatchSet[torch.Tensor] | None = None,
        units: Sequence[ResidencyUnit] | None = None,
        intermediate_dtype: torch.dtype = torch.float32,
    ) -> ResidentWeights:
        nonlocal calls
        calls += 1
        if calls == failure:
            raise failure_error("injected factory failure")
        mechanism = TrackingResidentWeights(
            weights,
            load_device=load_device,
            offload_device=offload_device,
            patch_set=patch_set,
            units=units,
            intermediate_dtype=intermediate_dtype,
        )
        constructed.append(mechanism)
        return mechanism

    with pytest.raises(failure_error, match="injected factory failure"):
        enroll_assembled(
            _policy_assembled(*components, enabled=enabled),
            load_device=CPU,
            offload_device=CPU,
            mechanism_factory=factory,
        )

    for component, snapshot in zip(components, before, strict=True):
        _assert_state_unchanged(component, snapshot)
        assert not hasattr(component, "_dinkster_resident_weights")
        assert component.residency_binding() is None  # type: ignore[attr-defined]
    assert all(mechanism.unload_calls == 1 for mechanism in constructed)

    retried = enroll_assembled(
        _policy_assembled(*components, enabled=enabled), load_device=CPU, offload_device=CPU
    )
    assert retried.storage_dtype_report.enabled is enabled
    assert bool(retried.storage_dtype_report.outcomes) is enabled


def test_storage_dtype_policy_rollback_preserves_external_storage_view() -> None:
    components = tuple(_policy_linear() for _ in range(3))
    original = components[0].weight
    external_view = original.detach().view(-1)
    storage_pointer = original.untyped_storage().data_ptr()
    data_pointer = original.data_ptr()
    expected = external_view.view(torch.uint8).clone()

    def fail_factory(
        weights: MutableMapping[str, StoredWeight],
        *,
        load_device: torch.device | str,
        offload_device: torch.device | str,
        patch_set: PatchSet[torch.Tensor] | None = None,
        units: Sequence[ResidencyUnit] | None = None,
        intermediate_dtype: torch.dtype = torch.float32,
    ) -> ResidentWeights:
        del weights
        raise RuntimeError("injected factory failure")

    with pytest.raises(RuntimeError, match="injected factory failure"):
        enroll_assembled(
            _policy_assembled(*components),
            load_device=CPU,
            offload_device=CPU,
            mechanism_factory=fail_factory,
        )

    assert components[0].weight is original
    assert original.untyped_storage().data_ptr() == storage_pointer
    assert original.data_ptr() == data_pointer
    assert external_view.untyped_storage().data_ptr() == storage_pointer
    assert external_view.data_ptr() == data_pointer
    assert torch.equal(external_view.view(torch.uint8), expected)


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("failure_error", [RuntimeError, asyncio.CancelledError, KeyboardInterrupt])
def test_storage_dtype_policy_bind_failure_rolls_back_whole_assembly(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    failure_error: type[BaseException],
) -> None:
    components = (
        _policy_linear(torch.float32),
        _policy_linear(torch.bfloat16),
        _policy_linear(torch.float16),
    )
    before = tuple(_state_bytes(component) for component in components)
    original = components[0].weight
    external_view = original.detach().view(-1)
    storage_pointer = original.untyped_storage().data_ptr()
    expected = external_view.view(torch.uint8).clone()
    previous = cast("ResidencyBinding", object())
    components[1].__dict__["_residency"] = previous
    original_bind = components[1].bind_residency  # type: ignore[attr-defined]

    def cancel_bind(binding: ResidencyBinding) -> None:
        components[1].__dict__["_residency"] = binding
        raise failure_error("injected bind failure")

    monkeypatch.setattr(components[1], "bind_residency", cancel_bind)
    with pytest.raises(failure_error, match="injected bind failure"):
        enroll_assembled(
            _policy_assembled(*components, enabled=enabled),
            load_device=CPU,
            offload_device=CPU,
        )

    for component, snapshot in zip(components, before, strict=True):
        _assert_state_unchanged(component, snapshot)
        assert not hasattr(component, "_dinkster_resident_weights")
    assert "_residency" not in components[0].__dict__
    assert components[1].__dict__.get("_residency") is previous
    assert "_residency" not in components[2].__dict__
    assert components[0].weight is original
    assert original.untyped_storage().data_ptr() == storage_pointer
    assert external_view.untyped_storage().data_ptr() == storage_pointer
    assert torch.equal(external_view.view(torch.uint8), expected)

    monkeypatch.setattr(components[1], "bind_residency", original_bind)
    retried = enroll_assembled(
        _policy_assembled(*components, enabled=enabled), load_device=CPU, offload_device=CPU
    )
    assert retried.storage_dtype_report.enabled is enabled


@pytest.mark.parametrize("failure_error", [RuntimeError, asyncio.CancelledError, KeyboardInterrupt])
def test_component_bind_failure_cleans_up_and_retries(
    monkeypatch: pytest.MonkeyPatch,
    failure_error: type[BaseException],
) -> None:
    component = _policy_linear(torch.bfloat16)
    before = _state_bytes(component)
    original_bind = component.bind_residency  # type: ignore[attr-defined]

    def fail_bind(binding: ResidencyBinding) -> None:
        component.__dict__["_residency"] = binding
        raise failure_error("injected bind failure")

    monkeypatch.setattr(component, "bind_residency", fail_bind)
    with pytest.raises(failure_error, match="injected bind failure"):
        enroll_component(component, load_device=CPU, offload_device=CPU)
    _assert_state_unchanged(component, before)
    assert component.residency_binding() is None  # type: ignore[attr-defined]
    assert not hasattr(component, "_dinkster_resident_weights")

    monkeypatch.setattr(component, "bind_residency", original_bind)
    enroll_component(component, load_device=CPU, offload_device=CPU)
    assert component.residency_binding() is not None  # type: ignore[attr-defined]


def test_component_bind_failure_preserves_primary_error_and_notes_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component = _policy_linear()

    class CleanupFailureResidentWeights(ResidentWeights):
        def unload(self) -> None:
            raise RuntimeError("injected cleanup failure")

    def factory(
        weights: MutableMapping[str, StoredWeight],
        *,
        load_device: torch.device | str,
        offload_device: torch.device | str,
        patch_set: PatchSet[torch.Tensor] | None = None,
        units: Sequence[ResidencyUnit] | None = None,
        intermediate_dtype: torch.dtype = torch.float32,
    ) -> ResidentWeights:
        return CleanupFailureResidentWeights(
            weights,
            load_device=load_device,
            offload_device=offload_device,
            patch_set=patch_set,
            units=units,
            intermediate_dtype=intermediate_dtype,
        )

    def fail_bind(_binding: ResidencyBinding) -> None:
        raise RuntimeError("primary bind failure")

    monkeypatch.setattr(component, "bind_residency", fail_bind)
    with pytest.raises(RuntimeError, match="primary bind failure") as caught:
        enroll_component(component, load_device=CPU, offload_device=CPU, mechanism_factory=factory)
    assert caught.value.__notes__ == [
        "residency enrollment cleanup also failed: RuntimeError('injected cleanup failure')"
    ]


@pytest.mark.parametrize("component", ["diffusion", "clip_l", "vae"])
def test_storage_dtype_policy_missing_patch_target_refuses_before_factory(
    component: str,
) -> None:
    components = tuple(_policy_linear() for _ in range(3))
    assert components[1].bias is not None
    components[1].bias = torch.nn.Parameter(components[1].bias.to(torch.float16))
    before = tuple(_state_bytes(item) for item in components)
    factory = _RecordingFactory()
    patch_set = PatchSet({"missing": (PatchEntry(DiffPatch(torch.ones(4, 4))),)})
    with pytest.raises(PatchApplyError, match="not in component"):
        enroll_assembled(
            _policy_assembled(*components),
            load_device=CPU,
            offload_device=CPU,
            patch_sets={component: patch_set},
            mechanism_factory=factory,
        )
    assert factory.calls == 0
    for item, snapshot in zip(components, before, strict=True):
        _assert_state_unchanged(item, snapshot)


def test_storage_dtype_policy_preserves_integer_buffer_bytes_and_execution() -> None:
    diffusion = _policy_linear()
    diffusion.register_buffer("sentinel", torch.tensor([7], dtype=torch.int64))
    sentinel = cast(torch.Tensor, diffusion.sentinel)
    storage_pointer = sentinel.untyped_storage().data_ptr()
    sentinel_bytes = sentinel.view(torch.uint8).clone()
    enrolled = enroll_assembled(
        _policy_assembled(diffusion, _policy_linear(), _policy_linear()),
        load_device=CPU,
        offload_device=CPU,
    )

    assert enrolled.storage_dtype_report.outcomes["diffusion"] == "converted"
    converted = cast(torch.Tensor, diffusion.sentinel)
    assert converted is sentinel
    assert converted.untyped_storage().data_ptr() == storage_pointer
    assert converted.dtype is torch.int64
    assert torch.equal(converted.view(torch.uint8), sentinel_bytes)
    assert (converted + 1).item() == 8


def test_storage_dtype_policy_preserves_tied_parameter_identity() -> None:
    tied = _TiedToy()
    enrolled = enroll_assembled(
        _policy_assembled(tied, _policy_linear(), _policy_linear()),
        load_device=CPU,
        offload_device=CPU,
    )
    assert enrolled.storage_dtype_report.outcomes["diffusion"] == "converted"
    assert tied.first.weight is tied.second.weight
    assert tied.first.weight.dtype is torch.float16


def test_storage_dtype_policy_offload_reload_roundtrip_is_byte_exact() -> None:
    diffusion = _policy_linear()
    enrolled = enroll_assembled(
        _policy_assembled(diffusion, _policy_linear(), _policy_linear()),
        load_device=CPU,
        offload_device=CPU,
    )
    expected = {
        name: parameter.detach().view(torch.uint8).clone()
        for name, parameter in diffusion.named_parameters()
    }
    mechanism = enrolled["diffusion"]
    for _ in range(2):
        mechanism.partially_load(None)
        mechanism.unload()
        for name, parameter in diffusion.named_parameters():
            assert parameter.dtype is torch.float16
            assert torch.equal(parameter.view(torch.uint8), expected[name])


def test_enroll_assembled_lumina2_uses_three_checkpoint_components() -> None:
    from dinkster_inference import LUMINA2
    from dinkster_inference_torch.assemble import AssembledLumina2

    components = ("diffusion", "gemma2_2b", "vae")
    assembled = AssembledLumina2(
        family=LUMINA2,
        diffusion=cast("Any", INITLESS.linear(2, 2)),
        gemma2_2b=cast("Any", INITLESS.linear(2, 2)),
        vae=cast("Any", INITLESS.linear(2, 2)),
        _component_compute_dtypes=dict.fromkeys(components, torch.float32),
    )
    mechanisms = enroll_assembled(assembled, load_device=CPU, offload_device=CPU)

    assert tuple(mechanisms) == components
    assert len({id(mechanism) for mechanism in mechanisms.values()}) == 3
    assert not mechanisms.storage_dtype_report.enabled
    for component, mechanism in mechanisms.items():
        assert assembled.compute_dtype(component) == torch.float32
        assert getattr(assembled, component)._dinkster_resident_weights is mechanism
        assert mechanism.total_bytes() == 24
        with mechanism.reserve_working_set():
            mechanism.partially_load(None)
        mechanism.unload()


def test_declared_component_map_is_authoritative_and_uses_public_compute_dtype() -> None:
    def compute_dtype(role: str) -> torch.dtype | None:
        return torch.float16 if role == "arbitrary-role" else None

    module = CastOperations(torch.float16).linear(2, 2, bias=False)
    module.load_state_dict({"weight": torch.ones(2, 2)}, assign=True)
    ignored = INITLESS.linear(2, 2)
    assembled = SimpleNamespace(
        components={"arbitrary-role": module},
        diffusion=ignored,
        _storage_dtype_follows_compute=True,
        compute_dtype=compute_dtype,
    )
    mechanisms = enroll_assembled(cast("Any", assembled), load_device=CPU, offload_device=CPU)
    assert tuple(mechanisms) == ("arbitrary-role",)
    assert module.weight.dtype == torch.float16
    assert not hasattr(ignored, "_dinkster_resident_weights")
    assert mechanisms.storage_dtype_report.outcomes["arbitrary-role"] == "converted"
    mechanisms["arbitrary-role"].unload()


@pytest.mark.parametrize("invalid", ["not-a-map", {"": None}, {"pixels": None}])
def test_declared_component_map_rejects_invalid_entries(invalid: object) -> None:
    with pytest.raises(TypeError, match="assembled component"):
        enroll_assembled(
            cast("Any", SimpleNamespace(components=invalid)),
            load_device=CPU,
            offload_device=CPU,
        )


def test_declared_component_map_rejects_duplicate_modules_before_enrollment() -> None:
    module = INITLESS.linear(2, 2)
    with pytest.raises(ValueError, match="must not share a module"):
        enroll_assembled(
            cast("Any", SimpleNamespace(components={"words": module, "pixels": module})),
            load_device=CPU,
            offload_device=CPU,
        )
    assert not hasattr(module, "_dinkster_resident_weights")
