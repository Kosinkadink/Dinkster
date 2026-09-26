"""Stage 4c slice 2: the residency mechanism + manager + aimdo seam.

ResidentWeights is pinned against the reference placement semantics
(comfy/model_patcher.py load/partially_load/partially_unload
@ 947c2749): largest-offload-estimate units claim residency first
(strict-< budget), unloading goes smallest-first, resident patched
keys hold patched storage with exact-original backups, offloaded
patched keys stay pristine, ordinary values patch at cast time, and
packed values patch and requantize on demand.

ResidencyManager is pinned against load_models_gpu/free_memory
@ 947c2749 with fake mechanisms and a fake device, so the policy math
(reserve arithmetic, the low-VRAM budget formula, eviction ordering,
partial-unload-before-detach) is tested without CUDA. Real-device
movement lives in test_gpu.py, capability-gated.

Everything here runs on CPU with load_device == offload_device ==
cpu: placement is then observable through accounting, patched-vs-
original storage values, and weight_functions - not tensor.device.

Run with the torch venv: .venv-torch/bin/python -m pytest -q
packages/dinkster-inference-torch/tests
"""

from __future__ import annotations

import platform
import weakref
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch
from dinkster_inference.patches import DiffPatch, PatchEntry, PatchSet
from dinkster_inference_torch import (
    DeviceMemory,
    Fp8ScaledWeight,
    MemoryPolicy,
    MpsMemorySnapshot,
    PatchApplyError,
    ResidencyManager,
    ResidencyUnit,
    ResidentWeights,
    StoredWeight,
    XpuMemorySnapshot,
    get_free_memory,
    get_total_memory,
    lora_compute_dtype,
    move_stored,
    mps_memory_snapshot,
    patch_stored_weight,
    probe_aimdo,
    quantize_fp8_scaled,
    soft_empty_cache,
    stored_nbytes,
    xpu_memory_snapshot,
)
from dinkster_inference_torch import residency as residency_mod
from dinkster_memory import SystemMemorySnapshot

CPU = torch.device("cpu")


class _FakeEagerTransferHooks:
    non_blocking = True

    def __init__(self) -> None:
        self.events: list[str] = []
        self.cleanup_check = lambda: None

    @contextmanager
    def loading_context(self) -> Generator[None]:
        self.events.append("loading-enter")
        try:
            yield
        finally:
            self.events.append("loading-exit")

    def consumer_wait_for_loading(self) -> None:
        self.events.append("loading-wait")

    @contextmanager
    def producer_context(self) -> Generator[None]:
        self.events.append("producer-enter")
        try:
            yield
        finally:
            self.events.append("producer-exit")

    def consumer_wait_for_producer(self) -> None:
        self.events.append("consumer-wait")

    def producer_wait_for_consumer(self) -> None:
        self.cleanup_check()
        self.events.append("cleanup")


class _InjectedCancellation(BaseException):
    pass


def diff_set(key: str, delta: torch.Tensor) -> PatchSet[torch.Tensor]:
    return PatchSet({key: (PatchEntry(DiffPatch(delta)),)})


def make_store(*sizes: tuple[str, int]) -> dict[str, StoredWeight]:
    """fp32 tensors so stochastic-rounding writeback is a plain cast
    and patched values are exactly comparable."""
    gen = torch.Generator().manual_seed(9)
    return {name: torch.randn(n, n, generator=gen) for name, n in sizes}


def as_tensor(stored: StoredWeight) -> torch.Tensor:
    """Narrow a store value to a plain tensor for comparisons."""
    assert isinstance(stored, torch.Tensor)
    return stored


# --------------------------------------------------- storage helpers


def test_stored_nbytes_plain_and_fp8() -> None:
    plain = torch.randn(4, 4)
    assert stored_nbytes(plain) == plain.nbytes
    fp8 = quantize_fp8_scaled(torch.randn(4, 4), torch.float8_e4m3fn)
    assert stored_nbytes(fp8) == fp8.qdata.nbytes + fp8.scale.nbytes


def test_move_stored_same_device_is_identity() -> None:
    plain = torch.randn(4, 4)
    assert move_stored(plain, CPU) is plain
    fp8 = quantize_fp8_scaled(torch.randn(4, 4), torch.float8_e4m3fn)
    moved = move_stored(fp8, CPU)
    assert isinstance(moved, Fp8ScaledWeight)
    assert moved.qdata is fp8.qdata and moved.scale is fp8.scale


def test_untied_resident_move_preserves_plain_move_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    store: dict[str, StoredWeight] = {"weight": original}
    resident = ResidentWeights(store, load_device=CPU, offload_device=CPU)
    moved = original.detach().clone()

    def cloning_move(
        _stored: StoredWeight, _device: torch.device, *, non_blocking: bool = False
    ) -> StoredWeight:
        assert non_blocking is False
        return moved

    monkeypatch.setattr(residency_mod, "move_stored", cloning_move)

    resident.partially_load(None)
    assert store["weight"] is moved


def test_resident_weights_reports_only_loaded_allocator_weights() -> None:
    resident = ResidentWeights(make_store(("a", 2), ("b", 3)), load_device=CPU, offload_device=CPU)

    assert resident.memory_accounting().weights == 0
    resident.partially_load(None)
    accounting = resident.memory_accounting()
    assert accounting.weights == resident.loaded_bytes() == 52
    assert accounting.allocator_weight_bytes == 52
    assert accounting.activation_runtime_workspace == 0
    assert accounting.other_reclaimable == 0
    assert accounting.memory_compiler == "unavailable"


@pytest.mark.parametrize("fully_loaded", [False, True])
def test_discard_orders_streams_releases_source_pins_and_preserves_store(
    monkeypatch: pytest.MonkeyPatch, fully_loaded: bool
) -> None:
    hooks = _FakeEagerTransferHooks()
    store = make_store(("a", 2), ("b", 2))
    original = as_tensor(store["a"])
    original_ref = weakref.ref(original)
    resident = ResidentWeights(
        store,
        load_device=CPU,
        offload_device=CPU,
        patch_set=diff_set("a", torch.ones_like(original)),
        transfer_hooks=hooks,
    )
    resident.partially_load(None)
    if not fully_loaded:
        resident.partially_unload(1)
    loaded = dict(store)
    states = [resident.unit_state(name) for name in ("a", "b")]
    assert any(state.loaded for state in states)

    pinned_host = residency_mod.pinned_host
    initial = pinned_host.TOTAL_PINNED_MEMORY
    storage_initial = pinned_host.TOTAL_PINNED_STORAGE
    registrations: dict[int, int] = {}

    def register(ptr: int, size: int) -> bool:
        registrations[ptr] = size
        return True

    def unregister(ptr: int) -> bool:
        hooks.events.append("unpin")
        del registrations[ptr]
        return True

    def synchronize(_device: torch.device) -> None:
        hooks.events.append("sync")

    def budget(_size: int) -> bool:
        return True

    def registerable(_size: int, *, evict_active: bool = True) -> bool:
        return True

    monkeypatch.setattr(residency_mod, "_cuda_host_register", register)
    monkeypatch.setattr(residency_mod, "_cuda_host_unregister", unregister)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(pinned_host, "ensure_pin_budget", budget)
    monkeypatch.setattr(pinned_host, "ensure_pin_registerable", registerable)
    pins = residency_mod._StoredSourcePins(torch.device("cuda", 0))  # pyright: ignore[reportPrivateUsage]
    resident._source_pins = pins  # pyright: ignore[reportPrivateUsage]
    pins.ensure(original)
    assert pinned_host.TOTAL_PINNED_MEMORY == initial + original.nbytes
    del original
    hooks.events.clear()

    def before_release() -> None:
        assert original_ref() is not None
        assert registrations
        assert pins in pinned_host._owners  # pyright: ignore[reportPrivateUsage]
        assert resident.loaded_bytes() > 0

    def unexpected_move(*_args: object, **_kwargs: object) -> StoredWeight:
        raise AssertionError("discard must not move stored tensors")

    hooks.cleanup_check = before_release
    monkeypatch.setattr(residency_mod, "move_stored", unexpected_move)
    resident.discard()
    assert hooks.events == ["cleanup", "sync", "unpin"]
    assert registrations == {}
    assert pinned_host.TOTAL_PINNED_MEMORY == initial
    assert pinned_host.TOTAL_PINNED_STORAGE == storage_initial
    assert pins not in pinned_host._owners  # pyright: ignore[reportPrivateUsage]
    assert original_ref() is None
    assert not resident._backup  # pyright: ignore[reportPrivateUsage]
    assert resident.loaded_unit_names() == frozenset()
    assert not any(state.loaded for state in states)
    assert resident.loaded_bytes() == 0
    assert all(store[key] is tensor and tensor.device == CPU for key, tensor in loaded.items())
    hooks.cleanup_check = lambda: None
    resident.discard()
    assert hooks.events == ["cleanup", "sync", "unpin", "cleanup"]
    assert pinned_host.TOTAL_PINNED_MEMORY == initial


@pytest.mark.parametrize("failure_at", ["stream", "pins"])
def test_discard_failure_keeps_real_loaded_state_and_manager_registration(
    monkeypatch: pytest.MonkeyPatch, failure_at: str
) -> None:
    store = make_store(("a", 2))
    hooks = _FakeEagerTransferHooks()
    resident = ResidentWeights(
        store,
        load_device=CPU,
        offload_device=CPU,
        patch_set=diff_set("a", torch.ones(2, 2)),
        transfer_hooks=hooks,
    )
    pins = residency_mod._StoredSourcePins(CPU)  # pyright: ignore[reportPrivateUsage]
    manager = ResidencyManager()
    manager.load([resident])
    resident._source_pins = pins  # pyright: ignore[reportPrivateUsage]
    backup = dict(resident._backup)  # pyright: ignore[reportPrivateUsage]
    loaded = store["a"]

    def fail() -> None:
        raise RuntimeError("discard cleanup failed")

    if failure_at == "stream":
        monkeypatch.setattr(hooks, "producer_wait_for_consumer", fail)
    else:
        monkeypatch.setattr(pins, "release_all", fail)
    with pytest.raises(RuntimeError, match="discard cleanup failed"):
        manager.remove([resident], discard=True)
    assert manager.registered() == (resident,)
    assert resident.loaded_bytes() == resident.total_bytes()
    assert resident.is_loaded("a")
    assert store["a"] is loaded
    assert resident._backup == backup  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("through_manager", [False, True])
def test_source_pin_release_failure_retains_ownership_and_retries(
    monkeypatch: pytest.MonkeyPatch, through_manager: bool
) -> None:
    pinned_host = residency_mod.pinned_host
    initial = pinned_host.TOTAL_PINNED_MEMORY
    storage_initial = pinned_host.TOTAL_PINNED_STORAGE
    successful = torch.ones(2)
    failed = torch.ones(3)
    successful_ptr, failed_ptr = successful.data_ptr(), failed.data_ptr()
    successful_bytes, failed_bytes = successful.nbytes, failed.nbytes
    successful_ref, failed_ref = weakref.ref(successful), weakref.ref(failed)
    failures = {failed_ptr}
    attempts: list[int] = []

    def register(_ptr: int, _size: int) -> bool:
        return True

    def unregister(ptr: int) -> bool:
        attempts.append(ptr)
        return ptr not in failures

    def ignore_device(_device: torch.device) -> None:
        pass

    def budget(_size: int, *, evict_active: bool = True) -> bool:
        return True

    monkeypatch.setattr(residency_mod, "_cuda_host_register", register)
    monkeypatch.setattr(residency_mod, "_cuda_host_unregister", unregister)
    monkeypatch.setattr(residency_mod, "_discard_cuda_async_error", ignore_device)
    monkeypatch.setattr(torch.cuda, "synchronize", ignore_device)
    monkeypatch.setattr(pinned_host, "ensure_pin_budget", budget)
    monkeypatch.setattr(pinned_host, "ensure_pin_registerable", budget)
    pins = residency_mod._StoredSourcePins(CPU)  # pyright: ignore[reportPrivateUsage]
    pins_ref = weakref.ref(pins)
    pins.ensure(successful)
    pins.ensure(failed)
    del successful, failed
    assert pinned_host.TOTAL_PINNED_MEMORY == initial + successful_bytes + failed_bytes
    resident = ResidentWeights(
        make_store(("a", 2)),
        load_device=CPU,
        offload_device=CPU,
        patch_set=diff_set("a", torch.ones(2, 2)),
    )
    manager = ResidencyManager()
    manager.load([resident])
    if through_manager:
        resident._source_pins = pins  # pyright: ignore[reportPrivateUsage]
    try:
        with pytest.raises(RuntimeError, match="failed source host unregistration"):
            if through_manager:
                manager.remove([resident], discard=True)
            else:
                pins.release_all()
        assert attempts == [successful_ptr, failed_ptr]
        assert successful_ref() is None
        assert failed_ref() is not None
        assert pins in pinned_host._owners  # pyright: ignore[reportPrivateUsage]
        assert pinned_host.TOTAL_PINNED_MEMORY == initial + failed_bytes
        assert manager.registered() == (resident,)
        assert resident.loaded_bytes() == resident.total_bytes()
        assert resident.is_loaded("a")
        assert resident._backup  # pyright: ignore[reportPrivateUsage]
        del pins
        pins = pins_ref()
        assert pins is not None
        assert pins.free_registrations(failed_bytes) == 0
        assert pinned_host.TOTAL_PINNED_MEMORY == initial + failed_bytes
        failures.clear()
        if through_manager:
            manager.remove([resident], discard=True)
            assert manager.registered() == ()
            assert resident.loaded_bytes() == 0
            assert not resident._backup  # pyright: ignore[reportPrivateUsage]
        else:
            pins.release_all()
        pins.release_all()
        assert attempts == [successful_ptr, failed_ptr, failed_ptr, failed_ptr]
        assert failed_ref() is None
        assert not pins._pinned  # pyright: ignore[reportPrivateUsage]
        assert pins not in pinned_host._owners  # pyright: ignore[reportPrivateUsage]
        assert pinned_host.TOTAL_PINNED_MEMORY == initial
        assert pinned_host.TOTAL_PINNED_STORAGE == storage_initial
    finally:
        failures.clear()
        owner = pins_ref()
        if owner is not None:
            owner.release_all()


def test_discard_leaves_prefetch_lease_closure_with_its_owner() -> None:
    resident = ResidentWeights(make_store(("a", 2)), load_device=CPU, offload_device=CPU)
    request = ("a", torch.float16)
    prefetch = resident.prefetch((request,))
    assert prefetch is not None
    with resident.lease("a") as lease:
        materialized = lease.get("a", dtype=torch.float16)
        materialized_ref = weakref.ref(materialized)
    del materialized
    assert materialized_ref() is not None
    prefetch.close()
    assert materialized_ref() is None
    resident.discard()
    prefetch.close()
    assert resident._peek_prefetched(request) is None  # pyright: ignore[reportPrivateUsage]


# ------------------------------------------------ consumption leases


def test_lease_get_matches_use_for_plain_and_fp8_storage() -> None:
    plain = torch.randn(4, 4, generator=torch.Generator().manual_seed(17))
    fp8 = quantize_fp8_scaled(
        torch.randn(4, 4, generator=torch.Generator().manual_seed(23)),
        torch.float8_e4m3fn,
    )
    store: dict[str, StoredWeight] = {"plain": plain, "q": fp8}
    resident = ResidentWeights(store, load_device=CPU, offload_device=CPU)

    expected_plain = resident.use("plain", dtype=torch.float16)
    with resident.lease("plain") as lease:
        actual_plain = lease.get("plain", dtype=torch.float16)
    assert torch.equal(actual_plain, expected_plain)

    expected_fp8 = resident.use("q", dtype=torch.float32)
    direct_stored = move_stored(store["q"], resident.load_device)
    with resident.lease("q") as lease:
        actual_fp8 = lease.get("q", dtype=torch.float32)
        leased_stored = lease.get_stored("q")
    assert torch.equal(actual_fp8, expected_fp8)
    assert isinstance(direct_stored, Fp8ScaledWeight)
    assert isinstance(leased_stored, Fp8ScaledWeight)
    assert torch.equal(
        leased_stored.qdata.view(torch.uint8),
        direct_stored.qdata.view(torch.uint8),
    )
    assert torch.equal(leased_stored.scale, direct_stored.scale)
    assert leased_stored.qdata.device == direct_stored.qdata.device
    assert leased_stored.scale.device == direct_stored.scale.device


def test_lease_closes_get_and_get_stored_with_unit_named() -> None:
    resident = ResidentWeights(make_store(("a", 2)), load_device=CPU, offload_device=CPU)
    with resident.lease("a") as lease:
        lease.get("a", dtype=torch.float32)
        lease.get_stored("a")

    with pytest.raises(RuntimeError, match="lease for unit 'a' is closed"):
        lease.get("a", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="lease for unit 'a' is closed"):
        lease.get_stored("a")


def test_lease_unknown_key_matches_use_key_error() -> None:
    resident = ResidentWeights(make_store(("a", 2)), load_device=CPU, offload_device=CPU)
    with pytest.raises(KeyError) as direct_error:
        resident.use("ghost", dtype=torch.float32)
    with resident.lease("unknown-unit-is-tolerated") as lease:
        with pytest.raises(KeyError) as lease_error:
            lease.get("ghost", dtype=torch.float32)
    assert lease_error.value.args == direct_error.value.args == ("ghost",)


def test_lease_of_loaded_unit_matches_use() -> None:
    resident = ResidentWeights(make_store(("a", 2)), load_device=CPU, offload_device=CPU)
    resident.partially_load(None)
    assert resident.is_loaded("a")
    expected = resident.use("a", dtype=torch.float16)
    with resident.lease("a") as lease:
        actual = lease.get("a", dtype=torch.float16)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("failure", [None, RuntimeError, _InjectedCancellation])
def test_offloaded_lease_retains_casts_through_consumer_and_cleans_once(
    failure: type[BaseException] | None,
) -> None:
    generator = torch.Generator().manual_seed(41)
    weight = torch.randn(3, 4, generator=generator)
    bias = torch.randn(3, generator=generator)
    store: dict[str, StoredWeight] = {"weight": weight, "bias": bias}
    patch_set = PatchSet(
        {
            "weight": (PatchEntry(DiffPatch(torch.zeros_like(weight))),),
            "bias": (PatchEntry(DiffPatch(torch.zeros_like(bias))),),
        }
    )
    hooks = _FakeEagerTransferHooks()
    resident = ResidentWeights(
        store,
        load_device=CPU,
        offload_device=CPU,
        patch_set=patch_set,
        units=(ResidencyUnit("linear", ("weight", "bias")),),
        transfer_hooks=hooks,
    )
    input = torch.randn(2, 4, generator=generator)
    expected = torch.nn.functional.linear(input, weight, bias)
    cast_refs: list[weakref.ReferenceType[torch.Tensor]] = []

    def consume() -> torch.Tensor:
        with resident.lease("linear") as lease:
            cast_weight_once = lease.get("weight", dtype=torch.float32)
            assert lease.get("weight", dtype=torch.float32) is cast_weight_once
            cast_bias_once = lease.get("bias", dtype=torch.float32)
            cast_refs.extend((weakref.ref(cast_weight_once), weakref.ref(cast_bias_once)))
            assert hooks.events == [
                "producer-enter",
                "producer-exit",
                "consumer-wait",
                "producer-enter",
                "producer-exit",
                "consumer-wait",
            ]
            result = torch.nn.functional.linear(input, cast_weight_once, cast_bias_once)
            hooks.events.append("consumer")
            del cast_weight_once, cast_bias_once
            if failure is not None:
                raise failure("injected operation exit")
            return result

    def cleanup_check() -> None:
        assert all(ref() is not None for ref in cast_refs), (
            "cast tensors were released before stream cleanup"
        )

    hooks.cleanup_check = cleanup_check
    if failure is None:
        assert torch.equal(consume(), expected)
    else:
        with pytest.raises(failure, match="injected operation exit"):
            consume()
    assert hooks.events[-2:] == ["consumer", "cleanup"]
    assert hooks.events.count("cleanup") == 1
    assert all(ref() is None for ref in cast_refs)


def test_loaded_lease_preserves_no_copy_path_without_transfer_hooks() -> None:
    stored = torch.randn(3, 4, generator=torch.Generator().manual_seed(43))
    store: dict[str, StoredWeight] = {"weight": stored}
    hooks = _FakeEagerTransferHooks()
    resident = ResidentWeights(
        store,
        load_device=CPU,
        offload_device=CPU,
        transfer_hooks=hooks,
    )
    resident.partially_load(None)
    hooks.events.clear()

    with resident.lease("weight") as lease:
        actual = lease.get("weight", dtype=stored.dtype)

    assert actual is stored
    assert hooks.events == []


# ------------------------------------------------ construction gates


def test_unit_key_not_in_store_raises() -> None:
    with pytest.raises(PatchApplyError, match="not in the weight store"):
        ResidentWeights(
            make_store(("a", 2)),
            load_device=CPU,
            offload_device=CPU,
            units=(ResidencyUnit("u", ("a", "ghost")),),
        )


def test_key_in_two_units_raises() -> None:
    with pytest.raises(PatchApplyError, match="appears in units"):
        ResidentWeights(
            make_store(("a", 2)),
            load_device=CPU,
            offload_device=CPU,
            units=(
                ResidencyUnit("u1", ("a",)),
                ResidencyUnit("u2", ("a",)),
            ),
        )


def test_duplicate_unit_name_raises() -> None:
    with pytest.raises(PatchApplyError, match="unit name 'duplicate' appears more than once"):
        ResidentWeights(
            make_store(("a", 2), ("b", 3)),
            load_device=CPU,
            offload_device=CPU,
            units=(
                ResidencyUnit("duplicate", ("a",)),
                ResidencyUnit("duplicate", ("b",)),
            ),
        )


def test_uncovered_store_key_raises() -> None:
    with pytest.raises(PatchApplyError, match="do not cover"):
        ResidentWeights(
            make_store(("a", 2), ("b", 2)),
            load_device=CPU,
            offload_device=CPU,
            units=(ResidencyUnit("u", ("a",)),),
        )


def test_patch_target_missing_from_store_raises() -> None:
    with pytest.raises(PatchApplyError, match="patch target"):
        ResidentWeights(
            make_store(("a", 2)),
            load_device=CPU,
            offload_device=CPU,
            patch_set=diff_set("ghost", torch.zeros(2, 2)),
        )


# ------------------------------------------- placement + accounting


def test_full_load_patches_storage_and_unload_restores_exactly() -> None:
    store = make_store(("a", 4), ("b", 4))
    original_a = as_tensor(store["a"]).clone()
    delta = torch.full((4, 4), 0.25)
    resident = ResidentWeights(
        store,
        load_device=CPU,
        offload_device=CPU,
        patch_set=diff_set("a", delta),
    )
    assert resident.loaded_bytes() == 0
    assert resident.offloaded_bytes() == resident.total_bytes()

    gained = resident.partially_load(None)
    assert gained == resident.total_bytes() == resident.loaded_bytes()
    assert resident.offloaded_bytes() == 0
    # resident patched key holds PATCHED storage...
    assert torch.allclose(as_tensor(store["a"]), original_a + delta)
    # ...and needs no cast-time functions
    assert resident.weight_functions("a") == ()
    assert resident.weight_functions("b") == ()

    resident.unload()
    assert resident.loaded_bytes() == 0
    # exact original back, bitwise
    assert torch.equal(as_tensor(store["a"]), original_a)


def test_retained_offload_storage_restores_without_copying_loaded_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = torch.nn.Parameter(torch.arange(4, dtype=torch.float32), requires_grad=False)
    loaded = torch.nn.Parameter(original.detach().clone(), requires_grad=False)
    store: dict[str, StoredWeight] = {"first": original, "second": original}
    resident = ResidentWeights(
        store,
        load_device=CPU,
        offload_device=CPU,
        units=(ResidencyUnit("tied", ("first", "second")),),
        transfer_hooks=_FakeEagerTransferHooks(),
    )
    moved_sources: list[StoredWeight] = []

    def observe_move(
        stored: StoredWeight, _device: torch.device, *, non_blocking: bool = False
    ) -> StoredWeight:
        moved_sources.append(stored)
        if len(moved_sources) == 1:
            assert non_blocking is False
            return loaded
        return stored

    monkeypatch.setattr(residency_mod, "move_stored", observe_move)
    resident.retain_offload_storage()
    resident.partially_load(None)
    assert store["first"] is store["second"] is loaded

    resident.unload()

    assert len(moved_sources) == 2
    assert moved_sources[0] is moved_sources[1] is original
    assert store["first"] is store["second"] is original
    assert resident.loaded_bytes() == 0


def test_offload_storage_retention_refuses_first_enable_after_loading() -> None:
    resident = ResidentWeights(make_store(("weight", 2)), load_device=CPU, offload_device=CPU)
    resident.partially_load(None)

    with pytest.raises(RuntimeError, match="must be enabled before loading"):
        resident.retain_offload_storage()


def test_accounting_cache_tracks_resizing_patches_without_rescanning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = torch.zeros(2, 2)
    store: dict[str, StoredWeight] = {"weight": original}
    resident = ResidentWeights(
        store,
        load_device=CPU,
        offload_device=CPU,
        patch_set=PatchSet({"weight": (PatchEntry(DiffPatch(torch.ones(3, 2), pad_weight=True)),)}),
    )
    original_bytes = original.nbytes
    assert resident.total_bytes() == original_bytes

    resident.partially_load(None)
    resized_bytes = as_tensor(store["weight"]).nbytes
    assert resized_bytes > original_bytes
    assert resident.total_bytes() == resized_bytes
    assert resident.loaded_bytes() == resized_bytes
    assert resident.offloaded_bytes() == 0

    def fail_scan(_stored: StoredWeight) -> int:
        pytest.fail("stable accounting rescanned the weight store")

    with monkeypatch.context() as accounting:
        accounting.setattr(residency_mod, "stored_nbytes", fail_scan)
        assert resident.total_bytes() == resized_bytes
        assert resident.loaded_bytes() == resized_bytes
        assert resident.offloaded_bytes() == 0

    resident.unload()
    assert resident.total_bytes() == original_bytes
    assert resident.loaded_bytes() == 0
    assert resident.offloaded_bytes() == original_bytes


def test_full_load_moves_before_patching_on_the_current_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = torch.tensor([1.0])
    store: dict[str, StoredWeight] = {"weight": original}
    hooks = _FakeEagerTransferHooks()
    actual_patch = residency_mod.patch_stored_weight

    def moved(
        stored: StoredWeight, _device: torch.device, *, non_blocking: bool = False
    ) -> StoredWeight:
        assert stored is original
        assert non_blocking is False
        hooks.events.append("move")
        return torch.tensor([3.0])

    def patched(
        stored: StoredWeight,
        entries: tuple[PatchEntry[torch.Tensor], ...],
        *,
        key: str,
        intermediate_dtype: torch.dtype,
        weight_dtype: torch.dtype | None = None,
    ) -> StoredWeight:
        assert torch.equal(as_tensor(stored), torch.tensor([3.0]))
        hooks.events.append("patch")
        return actual_patch(
            stored,
            entries,
            key=key,
            intermediate_dtype=intermediate_dtype,
            weight_dtype=weight_dtype,
        )

    monkeypatch.setattr(residency_mod, "move_stored", moved)
    monkeypatch.setattr(residency_mod, "patch_stored_weight", patched)
    resident = ResidentWeights(
        store,
        load_device=CPU,
        offload_device=CPU,
        patch_set=diff_set("weight", torch.tensor([2.0])),
        transfer_hooks=hooks,
    )

    resident.partially_load(None)

    assert hooks.events == ["loading-enter", "move", "patch", "loading-exit", "loading-wait"]
    assert torch.equal(as_tensor(store["weight"]), torch.tensor([5.0]))
    monkeypatch.undo()
    resident.unload()
    assert store["weight"] is original


def test_budget_prefers_largest_offload_estimate_strict() -> None:
    """load() @ 947c2749: units walk largest-first and fit under a
    STRICT < budget."""
    store = make_store(("big", 8), ("small", 2))
    resident = ResidentWeights(store, load_device=CPU, offload_device=CPU)
    big = stored_nbytes(store["big"])
    small = stored_nbytes(store["small"])

    # exactly big bytes: strict < means big does NOT fit; small does
    resident.partially_load(big)
    assert resident.loaded_unit_names() == frozenset({"small"})

    # one byte over: big fits, then small no longer does
    resident.unload()
    gained = resident.partially_load(big + 1)
    assert resident.loaded_unit_names() == frozenset({"big"})
    assert gained == big

    # room for both
    gained = resident.partially_load(small + 1)
    assert resident.loaded_unit_names() == frozenset({"big", "small"})
    assert gained == small


def test_patched_units_order_ahead_of_equal_sized_unpatched() -> None:
    """The LOWVRAM_PATCH_ESTIMATE factor: a patched unit is more
    expensive to offload, so it claims residency first among
    equals."""
    store = make_store(("plain", 4), ("patched", 4))
    resident = ResidentWeights(
        store,
        load_device=CPU,
        offload_device=CPU,
        patch_set=diff_set("patched", torch.zeros(4, 4)),
    )
    unit_bytes = stored_nbytes(store["plain"])
    resident.partially_load(unit_bytes + 1)
    assert resident.loaded_unit_names() == frozenset({"patched"})


def test_offloaded_patched_key_defers_and_matches_resident_patch() -> None:
    store = make_store(("a", 4))
    original = as_tensor(store["a"]).clone()
    delta = torch.full((4, 4), 0.5)
    patch_set = diff_set("a", delta)
    resident = ResidentWeights(store, load_device=CPU, offload_device=CPU, patch_set=patch_set)
    # never loaded: storage stays pristine, patches apply at cast time
    assert torch.equal(as_tensor(store["a"]), original)
    functions = resident.weight_functions("a")
    assert len(functions) == 1
    deferred = resident.use("a", dtype=torch.float32)
    expected = patch_stored_weight(original, patch_set.entries("a"), key="a")
    # fp32 storage: deferred (weight-dtype intermediate) and at-load
    # (fp32 intermediate) paths coincide exactly
    assert torch.equal(deferred, expected)

    resident.partially_load(None)
    assert torch.equal(resident.use("a", dtype=torch.float32), expected)


def test_partially_unload_goes_smallest_first() -> None:
    store = make_store(("big", 8), ("mid", 4), ("small", 2))
    resident = ResidentWeights(store, load_device=CPU, offload_device=CPU)
    resident.partially_load(None)

    freed = resident.partially_unload(1)
    assert freed == stored_nbytes(store["small"])
    assert resident.loaded_unit_names() == frozenset({"big", "mid"})

    freed = resident.partially_unload(stored_nbytes(store["mid"]))
    assert freed == stored_nbytes(store["mid"])
    assert resident.loaded_unit_names() == frozenset({"big"})


def test_negative_extra_memory_shrinks_residency() -> None:
    """partially_load @ 947c2749 routes a negative allowance into
    partially_unload (smallest-first), never a re-settle."""
    store = make_store(("big", 8), ("small", 2))
    resident = ResidentWeights(store, load_device=CPU, offload_device=CPU)
    resident.partially_load(None)
    change = resident.partially_load(-stored_nbytes(store["small"]))
    assert change == -stored_nbytes(store["small"])
    assert resident.loaded_unit_names() == frozenset({"big"})


def test_failed_unit_load_rolls_back_the_whole_unit() -> None:
    """A patch failure mid-unit must not leak moved/patched storage
    into the store: the unit rolls back and stays unloaded."""
    store = make_store(("w", 4), ("b", 4))
    original_w = store["w"]
    original_b = store["b"]
    resident = ResidentWeights(
        store,
        load_device=CPU,
        offload_device=CPU,
        # wrong-shape diff: patching "b" raises after "w" already moved
        patch_set=diff_set("b", torch.zeros(2, 2)),
        units=(ResidencyUnit("unit", ("w", "b")),),
    )
    with pytest.raises(PatchApplyError):
        resident.partially_load(None)
    assert resident.loaded_bytes() == 0
    assert resident.loaded_unit_names() == frozenset()
    assert store["w"] is original_w
    assert store["b"] is original_b
    # the mechanism stays usable: a fixed patch set means a new
    # instance (documented contract), but unload here must be a no-op
    resident.unload()
    assert store["b"] is original_b


@pytest.mark.parametrize("failure", [RuntimeError, _InjectedCancellation])
def test_failed_streamed_unit_load_waits_before_rollback(
    monkeypatch: pytest.MonkeyPatch,
    failure: type[BaseException],
) -> None:
    store = make_store(("w", 2), ("b", 2), ("other", 1))
    originals = dict(store)
    hooks = _FakeEagerTransferHooks()
    calls = 0

    def failing_patch(
        stored: StoredWeight,
        _entries: tuple[PatchEntry[torch.Tensor], ...],
        *,
        key: str,
        intermediate_dtype: torch.dtype,
        weight_dtype: torch.dtype | None = None,
    ) -> StoredWeight:
        nonlocal calls
        del key, intermediate_dtype, weight_dtype
        calls += 1
        if calls == 2:
            raise failure("injected patch failure")
        return stored

    monkeypatch.setattr(residency_mod, "patch_stored_weight", failing_patch)
    resident = ResidentWeights(
        store,
        load_device=CPU,
        offload_device=CPU,
        patch_set=PatchSet(
            {
                "w": (PatchEntry(DiffPatch(torch.zeros(2, 2))),),
                "b": (PatchEntry(DiffPatch(torch.zeros(2, 2))),),
            }
        ),
        units=(
            ResidencyUnit("unit", ("w", "b")),
            ResidencyUnit("other", ("other",)),
        ),
        transfer_hooks=hooks,
    )

    with pytest.raises(failure, match="injected patch failure"):
        resident.partially_load(stored_nbytes(store["w"]) + stored_nbytes(store["b"]) + 1)

    assert hooks.events == ["producer-enter", "producer-exit", "consumer-wait"]
    assert resident.loaded_unit_names() == frozenset()
    assert all(store[key] is original for key, original in originals.items())


def test_multi_key_unit_moves_together() -> None:
    store = make_store(("w", 4), ("b", 2), ("other", 2))
    resident = ResidentWeights(
        store,
        load_device=CPU,
        offload_device=CPU,
        units=(
            ResidencyUnit("layer", ("w", "b")),
            ResidencyUnit("other", ("other",)),
        ),
    )
    layer_bytes = stored_nbytes(store["w"]) + stored_nbytes(store["b"])
    gained = resident.partially_load(layer_bytes + 1)
    assert resident.loaded_unit_names() == frozenset({"layer"})
    assert gained == layer_bytes


def test_fp8_store_roundtrip_through_residency() -> None:
    source = torch.randn(8, 8, generator=torch.Generator().manual_seed(3))
    fp8 = quantize_fp8_scaled(source, torch.float8_e4m3fn)
    store: dict[str, StoredWeight] = {"q": fp8}
    delta = torch.full((8, 8), 0.125)
    resident = ResidentWeights(
        store,
        load_device=CPU,
        offload_device=CPU,
        patch_set=diff_set("q", delta),
    )
    resident.partially_load(None)
    loaded = store["q"]
    assert isinstance(loaded, Fp8ScaledWeight)
    assert loaded is not fp8  # requantized patched storage
    resident.unload()
    assert store["q"] is fp8  # exact original object restored


# ------------------------------------------------------ manager fakes


@dataclass
class FakeMechanism:
    name: str
    total: int
    device: torch.device = CPU
    loaded: int = 0
    cached: int = 0
    demand_paged: bool = False
    fully_offloadable: bool = False
    working_set_extra: int = 0
    automatic_reclaimable: int | None = None
    calls: list[tuple[str, int | None]] = field(default_factory=list)
    working_set_events: list[str] = field(default_factory=list)

    @property
    def load_device(self) -> torch.device:
        return self.device

    def total_bytes(self) -> int:
        return self.total

    def loaded_bytes(self) -> int:
        return self.loaded

    def automatically_reclaimable_bytes(self) -> int:
        return self.loaded if self.automatic_reclaimable is None else self.automatic_reclaimable

    def offloaded_bytes(self) -> int:
        return self.total - self.loaded

    def working_set_reservation_bytes(self) -> int:
        return self.working_set_extra

    @contextmanager
    def reserve_working_set(self):  # noqa: ANN201
        self.working_set_events.append("enter")
        try:
            yield
        finally:
            self.working_set_events.append("exit")

    def partially_load(self, extra_memory: int | None) -> int:
        self.calls.append(("partially_load", extra_memory))
        before = self.loaded
        if extra_memory is None or self.loaded + extra_memory > self.total:
            self.loaded = self.total
        else:
            self.loaded = max(0, self.loaded + extra_memory)
        return self.loaded - before

    def partially_unload(self, memory_to_free: int) -> int:
        self.calls.append(("partially_unload", memory_to_free))
        cached_freed = min(memory_to_free, self.cached)
        self.cached -= cached_freed
        loaded_freed = min(memory_to_free - cached_freed, self.loaded)
        self.loaded -= loaded_freed
        return cached_freed + loaded_freed

    def partial_unload_capacity(self) -> int:
        return self.loaded + self.cached

    def can_fully_offload(self) -> bool:
        return self.fully_offloadable

    def unload(self) -> None:
        self.calls.append(("unload", None))
        self.loaded = 0
        self.cached = 0

    def release_working_buffers(self) -> bool:
        return False


class FakeDevice:
    """A device whose free memory is capacity minus whatever the fake
    mechanisms currently hold - the coupling get_free_memory would
    observe for real."""

    def __init__(self, capacity: int, device: torch.device) -> None:
        self.capacity = capacity
        self.device = device
        self.mechanisms: list[FakeMechanism] = []
        self.empty_cache_calls = 0

    def free(self, device: torch.device) -> DeviceMemory:
        assert device == self.device
        used = sum(m.loaded + m.cached for m in self.mechanisms)
        return DeviceMemory(free_total=self.capacity - used, free_torch=0)

    def empty_cache(self, device: torch.device) -> None:
        self.empty_cache_calls += 1


VRAM = torch.device("cuda", 0)  # never touched: fakes only


def make_manager(capacity: int, **policy_overrides: object) -> tuple[ResidencyManager, FakeDevice]:
    fake = FakeDevice(capacity, VRAM)
    defaults: dict[str, object] = {
        "inference_reserve": 100,
        "physical_headroom": 0,
        "min_weight_memory_ratio": 0.0,
        "load_inflation": 1.0,
    }
    defaults.update(policy_overrides)
    policy = MemoryPolicy(**defaults)  # pyright: ignore[reportArgumentType]
    manager = ResidencyManager(policy=policy, free_memory=fake.free, empty_cache=fake.empty_cache)
    return manager, fake


def test_working_set_reservation_requires_capacity_beyond_inference_reserve() -> None:
    manager, fake = make_manager(1000)
    fits = FakeMechanism("fits", total=400, device=VRAM, working_set_extra=400)
    fake.mechanisms.append(fits)

    with manager.reserve_working_sets([fits]):
        assert fits.working_set_events == ["enter"]
    assert fits.working_set_events == ["enter", "exit"]

    too_large = FakeMechanism("too-large", total=901, device=VRAM, working_set_extra=901)
    with manager.reserve_working_sets([too_large]):
        assert too_large.working_set_events == []


def test_working_set_reservation_sums_same_device_and_releases_on_failure() -> None:
    manager, fake = make_manager(1000)
    first = FakeMechanism("first", total=500, device=VRAM, working_set_extra=500)
    second = FakeMechanism("second", total=401, device=VRAM, working_set_extra=401)
    fake.mechanisms.extend((first, second))

    with manager.reserve_working_sets([first, second]):
        assert first.working_set_events == []
        assert second.working_set_events == []

    second.working_set_extra = 400
    with pytest.raises(RuntimeError, match="stage failed"):
        with manager.reserve_working_sets([first, second]):
            raise RuntimeError("stage failed")
    assert first.working_set_events == ["enter", "exit"]
    assert second.working_set_events == ["enter", "exit"]


# --------------------------------------------------- manager: budgets


def test_load_gives_the_reference_lowvram_budget() -> None:
    """current_free + loaded -> max(0, free - minimum, min(free *
    ratio, free - inference)) - loaded, verbatim @ 947c2749."""
    manager, fake = make_manager(1000)
    m = FakeMechanism("m", total=500, device=VRAM)
    fake.mechanisms.append(m)
    manager.load([m])
    # free=1000: lowvram = max(0, 1000-100, min(0, 900)) = 900
    assert m.calls == [("partially_load", 900)]
    assert m.loaded == 500  # 900 > total: full load
    assert manager.registered() == (m,)


def test_load_budget_caps_an_oversized_model() -> None:
    manager, fake = make_manager(1000)
    m = FakeMechanism("m", total=2000, device=VRAM)
    fake.mechanisms.append(m)
    manager.load([m])
    assert m.calls == [("partially_load", 900)]
    assert m.loaded == 900  # partial residency, inference reserve kept
    assert fake.free(VRAM).free_total == 100


def test_load_evicts_older_models_partially_first() -> None:
    """free_memory @ 947c2749 partial-unloads before detaching; the
    incoming model is never evicted for itself."""
    manager, fake = make_manager(1000)
    old = FakeMechanism("old", total=2000, device=VRAM)
    fake.mechanisms.append(old)
    manager.load([old])
    assert old.loaded == 900

    new = FakeMechanism("new", total=600, device=VRAM)
    fake.mechanisms.append(new)
    manager.load([new])
    # pre-load free pass: required 600*1.0+100=700, free was 100 ->
    # shortfall 600 < old.loaded 900 -> partial unload, no detach
    assert ("partially_unload", 600) in old.calls
    assert ("unload", None) not in old.calls
    assert old.loaded == 300
    assert new.loaded == 600
    # MRU: newest first, no duplicate entries
    assert manager.registered() == (new, old)


def test_load_detaches_when_partial_unload_cannot_satisfy() -> None:
    manager, fake = make_manager(1000)
    old = FakeMechanism("old", total=300, device=VRAM)
    fake.mechanisms.append(old)
    manager.load([old])
    assert old.loaded == 300

    new = FakeMechanism("new", total=850, device=VRAM)
    fake.mechanisms.append(new)
    manager.load([new])
    # required 850+100=950, free 700 -> shortfall 250 < loaded 300?
    # no: 250 < 300 -> partial frees 250; recheck -> satisfied.
    # then minimum pass ok; budget: free 950 -> lowvram 850
    assert old.loaded == 50
    assert new.loaded == 850

    # now demand more than old can ever partially satisfy
    third = FakeMechanism("third", total=900, device=VRAM)
    fake.mechanisms.append(third)
    manager.load([third])
    # shortfall exceeds old's remaining 50 -> unload + registry pop
    assert ("unload", None) in old.calls or old.loaded == 0
    assert old not in manager.registered()
    assert fake.empty_cache_calls >= 1


def test_reload_moves_entry_to_front_without_duplicates() -> None:
    """Upstream duplicates re-loaded registry entries
    (docs/comfyui-issues/comfyui-load-models-gpu-duplicate-registry.md);
    Dinkster moves to front."""
    manager, fake = make_manager(10_000)
    a = FakeMechanism("a", total=100, device=VRAM)
    b = FakeMechanism("b", total=100, device=VRAM)
    fake.mechanisms.extend([a, b])
    manager.load([a])
    manager.load([b])
    assert manager.registered() == (b, a)
    manager.load([a])
    assert manager.registered() == (a, b)


def test_reload_does_not_resettle_a_fully_resident_mechanism() -> None:
    manager, fake = make_manager(10_000)
    mechanism = FakeMechanism("resident", total=100, device=VRAM)
    fake.mechanisms.append(mechanism)
    manager.load([mechanism])
    assert mechanism.calls == [("partially_load", 9900)]

    manager.load([mechanism])

    assert mechanism.calls == [("partially_load", 9900)]
    assert manager.registered() == (mechanism,)


def test_load_dedups_and_reverses_like_the_reference() -> None:
    manager, fake = make_manager(10_000)
    a = FakeMechanism("a", total=100, device=VRAM)
    b = FakeMechanism("b", total=100, device=VRAM)
    fake.mechanisms.extend([a, b])
    manager.load([a, b, a])
    # dedup [a, b]; reversed -> b settles first, a ends newest
    assert manager.registered() == (a, b)


def test_cpu_load_device_gets_full_load_and_no_eviction_pass() -> None:
    manager, _ = make_manager(1000)
    m = FakeMechanism("m", total=500, device=CPU)
    manager.load([m])
    assert m.calls == [("partially_load", None)]
    assert m.loaded == 500


def test_force_full_load_bypasses_budgeting() -> None:
    manager, fake = make_manager(1000)
    m = FakeMechanism("m", total=2000, device=VRAM)
    fake.mechanisms.append(m)
    manager.load([m], force_full_load=True)
    assert m.calls == [("partially_load", None)]
    assert m.loaded == 2000


def test_memory_required_inflates_the_free_pass() -> None:
    manager, fake = make_manager(1000, load_inflation=1.1)
    old = FakeMechanism("old", total=800, device=VRAM)
    fake.mechanisms.append(old)
    manager.load([old])
    assert old.loaded == 800

    new = FakeMechanism("new", total=100, device=VRAM)
    fake.mechanisms.append(new)
    manager.load([new], memory_required=300)
    # extra_mem = max(100, 300+0) = 300;
    # free pass wants 100*1.1 + 300 = 410; free was 200 -> shortfall
    assert ("partially_unload", 210) in old.calls
    # budget for new: free 410 -> lowvram = 410 - 300 = 110 -> full
    assert new.loaded == 100


def test_memory_required_evicts_around_an_already_resident_target() -> None:
    manager, fake = make_manager(1000)
    old = FakeMechanism("old", total=800, device=VRAM)
    target = FakeMechanism("target", total=100, device=VRAM)
    fake.mechanisms.extend([old, target])
    manager.load([old])
    manager.load([target])
    assert old.loaded == 800
    assert target.loaded == 100
    target_calls = list(target.calls)

    manager.load([target], memory_required=300)

    assert ("partially_unload", 200) in old.calls
    assert target.loaded == 100
    assert target.calls == target_calls
    assert manager.registered()[0] is target


def test_minimum_memory_floor_triggers_second_free_pass() -> None:
    manager, fake = make_manager(1000)
    old = FakeMechanism("old", total=900, device=VRAM)
    fake.mechanisms.append(old)
    manager.load([old])
    assert old.loaded == 900

    new = FakeMechanism("new", total=50, device=VRAM)
    fake.mechanisms.append(new)
    manager.load([new], minimum_memory=500)
    # minimum_required = max(100, 500+0) = 500; first pass frees
    # 50+100=150 (shortfall 50); second pass tops up to 500 free
    assert fake.free(VRAM).free_total >= 500 - new.loaded


def test_free_eviction_prefers_most_offloaded_then_smallest() -> None:
    manager, fake = make_manager(1000)
    mostly_offloaded = FakeMechanism("mo", total=400, loaded=100, device=VRAM)
    small = FakeMechanism("small", total=200, loaded=200, device=VRAM)
    big = FakeMechanism("big", total=700, loaded=700, device=VRAM)
    fake.mechanisms.extend([mostly_offloaded, small, big])
    for m in (mostly_offloaded, small, big):
        manager._touch(m)  # pyright: ignore[reportPrivateUsage]

    # free memory: 1000 - 1000 = 0; ask for 90
    manager.free(90, VRAM)
    # most-offloaded candidate goes first; 90 < 100 -> partial
    assert ("partially_unload", 90) in mostly_offloaded.calls
    assert small.calls == [] and big.calls == []


def test_free_reclaims_dynamic_cache_when_no_static_units_are_loaded() -> None:
    manager, fake = make_manager(1000)
    cached = FakeMechanism("cached", total=800, cached=400, device=VRAM)
    fake.mechanisms.append(cached)
    manager._touch(cached)  # pyright: ignore[reportPrivateUsage]

    manager.free(700, VRAM)

    assert ("partially_unload", 100) in cached.calls
    assert ("unload", None) not in cached.calls
    assert cached.cached == 300
    assert cached in manager.registered()


@pytest.mark.parametrize("memory_required", (1000, 1100))
def test_free_preserves_fully_offloadable_state_at_or_beyond_capacity(
    memory_required: int,
) -> None:
    manager, fake = make_manager(1000)
    mechanism = FakeMechanism(
        "stateful",
        total=200,
        loaded=200,
        device=VRAM,
        fully_offloadable=True,
    )
    fake.mechanisms.append(mechanism)
    manager._touch(mechanism)  # pyright: ignore[reportPrivateUsage]

    manager.free(memory_required, VRAM)

    assert mechanism.calls == [("partially_unload", 200)]
    assert mechanism.loaded == 0
    assert manager.registered() == (mechanism,)
    manager.remove((mechanism,))
    assert mechanism.calls[-1] == ("unload", None)
    assert manager.registered() == ()


def test_free_preserves_fully_offloaded_state_before_continuing_to_later_victims() -> None:
    manager, fake = make_manager(1000)
    stateful = FakeMechanism(
        "stateful",
        total=200,
        loaded=200,
        device=VRAM,
        fully_offloadable=True,
    )
    victim = FakeMechanism("victim", total=300, loaded=300, device=VRAM)
    fake.mechanisms.extend((stateful, victim))
    manager._touch(stateful)  # pyright: ignore[reportPrivateUsage]
    manager._touch(victim)  # pyright: ignore[reportPrivateUsage]

    manager.free(1000, VRAM)

    assert stateful.calls == [("partially_unload", 200)]
    assert stateful.loaded == 0
    assert victim.calls == [("unload", None)]
    assert manager.registered() == (stateful,)


def test_free_keep_protects_requested_models() -> None:
    manager, fake = make_manager(1000)
    keeper = FakeMechanism("keeper", total=800, loaded=800, device=VRAM)
    victim = FakeMechanism("victim", total=200, loaded=200, device=VRAM)
    fake.mechanisms.extend([keeper, victim])
    manager._touch(keeper)  # pyright: ignore[reportPrivateUsage]
    manager._touch(victim)  # pyright: ignore[reportPrivateUsage]

    manager.free(150, VRAM, keep=[keeper])
    assert keeper.calls == []
    assert victim.loaded == 50


def test_all_demand_paged_load_skips_demand_paged_victims() -> None:
    manager, fake = make_manager(1000)
    victim = FakeMechanism("victim", total=600, loaded=600, device=VRAM, demand_paged=True)
    incoming = FakeMechanism("incoming", total=500, device=VRAM, demand_paged=True)
    fake.mechanisms.extend([victim, incoming])
    manager._touch(victim)  # pyright: ignore[reportPrivateUsage]

    manager.load([incoming])

    assert victim.calls == []
    assert victim in manager.registered()


def test_demand_paged_load_demotes_fixed_residency_beyond_reusable_pages() -> None:
    manager, fake = make_manager(1000)
    victim = FakeMechanism(
        "victim",
        total=1000,
        loaded=200,
        cached=400,
        device=VRAM,
        demand_paged=True,
        automatic_reclaimable=400,
    )
    fake.mechanisms.append(victim)
    manager._touch(victim)  # pyright: ignore[reportPrivateUsage]

    manager.free(900, VRAM, skip_demand_paged=True)

    assert victim.calls == [("partially_unload", 500)]
    assert victim.cached == 0
    assert victim.loaded == 100
    assert victim in manager.registered()
    assert fake.free(VRAM).free_total == 900


def test_demand_paged_load_keeps_victim_and_continues_after_fixed_demotion() -> None:
    manager, fake = make_manager(1300)
    dynamic = FakeMechanism(
        "dynamic",
        total=1000,
        loaded=200,
        cached=400,
        device=VRAM,
        demand_paged=True,
        automatic_reclaimable=400,
    )
    eager = FakeMechanism("eager", total=300, loaded=300, device=VRAM)
    fake.mechanisms.extend((dynamic, eager))
    manager._touch(eager)  # pyright: ignore[reportPrivateUsage]
    manager._touch(dynamic)  # pyright: ignore[reportPrivateUsage]

    manager.free(1300, VRAM, skip_demand_paged=True)

    assert dynamic.calls == [("partially_unload", 600)]
    assert dynamic in manager.registered()
    assert eager.calls == [("unload", None)]
    assert eager not in manager.registered()
    assert fake.free(VRAM).free_total == 1300


def test_demand_paged_load_still_detaches_empty_eager_victim() -> None:
    manager, fake = make_manager(1000)
    empty = FakeMechanism("empty", total=100, device=VRAM)
    loaded = FakeMechanism("loaded", total=500, loaded=500, device=VRAM)
    fake.mechanisms.extend((empty, loaded))
    manager._touch(loaded)  # pyright: ignore[reportPrivateUsage]
    manager._touch(empty)  # pyright: ignore[reportPrivateUsage]

    manager.free(600, VRAM, skip_demand_paged=True)

    assert empty.calls == [("unload", None)]
    assert empty not in manager.registered()
    assert loaded.calls == [("partially_unload", 100)]
    assert fake.free(VRAM).free_total == 600


def test_eager_load_does_not_skip_a_demand_paged_victim() -> None:
    manager, fake = make_manager(1000)
    victim = FakeMechanism("victim", total=600, loaded=600, device=VRAM, demand_paged=True)
    incoming = FakeMechanism("incoming", total=500, device=VRAM)
    fake.mechanisms.extend([victim, incoming])
    manager._touch(victim)  # pyright: ignore[reportPrivateUsage]

    manager.load([incoming])

    assert victim.calls


def test_free_empties_cache_only_after_detach_or_reserve_ratio() -> None:
    manager, fake = make_manager(1000)
    m = FakeMechanism("m", total=200, loaded=200, device=VRAM)
    fake.mechanisms.append(m)
    manager._touch(m)  # pyright: ignore[reportPrivateUsage]

    manager.free(50, VRAM)  # partial: no cache empty (free_torch=0)
    assert fake.empty_cache_calls == 0
    manager.free(2000, VRAM)  # cannot satisfy: detach -> cache empty
    assert m.loaded == 0
    assert m not in manager.registered()
    assert fake.empty_cache_calls == 1


def test_free_empties_unquantifiable_mps_cache_on_shortfall() -> None:
    device = torch.device("mps")
    fake = FakeDevice(1000, device)
    manager = ResidencyManager(free_memory=fake.free, empty_cache=fake.empty_cache)

    manager.free(500, device)
    assert fake.empty_cache_calls == 0
    manager.free(1200, device)
    assert fake.empty_cache_calls == 1


# ------------------------------------------------- manager: MPS admission


MPS = torch.device("mps")  # never touched: fakes only

FAKE_MPS_SNAPSHOT = MpsMemorySnapshot(
    recommended_max_bytes=1000,
    driver_allocated_bytes=600,
    current_allocated_bytes=250,
    system_total_bytes=2000,
    system_available_bytes=700,
)


def make_mps_manager(capacity: int) -> tuple[ResidencyManager, FakeDevice]:
    fake = FakeDevice(capacity, MPS)
    policy = MemoryPolicy(
        inference_reserve=100,
        physical_headroom=0,
        min_weight_memory_ratio=0.0,
        load_inflation=1.0,
    )
    manager = ResidencyManager(
        policy=policy,
        free_memory=fake.free,
        empty_cache=fake.empty_cache,
        mps_snapshot=lambda device: FAKE_MPS_SNAPSHOT,
    )
    return manager, fake


def test_mps_load_attempts_partial_loading_when_eviction_cannot_clear_the_reserve(
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager, fake = make_mps_manager(50)
    m = FakeMechanism("m", total=200, device=MPS)
    fake.mechanisms.append(m)

    manager.load([m])

    message = caplog.text
    assert "loading 200 bytes of weights" in message
    assert "leaves 50 bytes free after eviction" in message
    assert "100-byte inference reserve" in message
    assert FAKE_MPS_SNAPSHOT.describe() in message
    assert "attempting memory-budgeted loading" in message
    assert m.calls == [("partially_load", 0)]
    assert manager.registered() == (m,)


def test_mps_load_admits_after_eviction_clears_the_reserve() -> None:
    manager, fake = make_mps_manager(1000)
    old = FakeMechanism("old", total=950, loaded=950, device=MPS)
    fake.mechanisms.append(old)
    manager._touch(old)  # pyright: ignore[reportPrivateUsage]

    new = FakeMechanism("new", total=500, device=MPS)
    fake.mechanisms.append(new)
    manager.load([new])

    assert new.loaded == 500
    assert manager.registered()[0] is new


def test_mps_retouch_of_resident_model_is_not_refused() -> None:
    manager, fake = make_mps_manager(200)
    m = FakeMechanism("m", total=200, loaded=200, device=MPS)
    fake.mechanisms.append(m)
    manager._touch(m)  # pyright: ignore[reportPrivateUsage]

    manager.load([m])  # free=0 < reserve, but nothing left to place
    assert manager.registered() == (m,)


def test_cuda_load_is_never_refused_by_mps_admission() -> None:
    manager, fake = make_manager(50)
    m = FakeMechanism("m", total=200, device=VRAM)
    fake.mechanisms.append(m)

    manager.load([m])  # same starved budget as the MPS case

    assert m.calls == [("partially_load", 0)]
    assert manager.registered() == (m,)


@dataclass
class OomMechanism(FakeMechanism):
    message: str = "MPS backend out of memory (tried to allocate 1.00 GiB)"

    def partially_load(self, extra_memory: int | None) -> int:
        raise RuntimeError(self.message)


@pytest.mark.parametrize("capacity", [50, 1000])
def test_mps_transfer_oom_carries_the_budget_note(capacity: int) -> None:
    manager, fake = make_mps_manager(capacity)
    m = OomMechanism("m", total=500, device=MPS)
    fake.mechanisms.append(m)

    with pytest.raises(RuntimeError, match="MPS backend out of memory") as excinfo:
        manager.load([m])

    notes = getattr(excinfo.value, "__notes__", [])
    assert notes == [f"MPS unified-memory budget at failure: {FAKE_MPS_SNAPSHOT.describe()}"]


def test_non_mps_or_unrelated_errors_get_no_budget_note() -> None:
    manager, fake = make_manager(1000)
    cuda_oom = OomMechanism("cuda-oom", total=500, device=VRAM)
    fake.mechanisms.append(cuda_oom)
    with pytest.raises(RuntimeError) as excinfo:
        manager.load([cuda_oom])
    assert getattr(excinfo.value, "__notes__", []) == []

    mps_manager, mps_fake = make_mps_manager(1000)
    unrelated = OomMechanism("unrelated", total=500, device=MPS, message="settle failed")
    mps_fake.mechanisms.append(unrelated)
    with pytest.raises(RuntimeError, match="settle failed") as excinfo:
        mps_manager.load([unrelated])
    assert getattr(excinfo.value, "__notes__", []) == []


def test_mps_budget_note_survives_a_failing_snapshot_read() -> None:
    fake = FakeDevice(1000, MPS)

    def broken_snapshot(device: torch.device) -> MpsMemorySnapshot:
        raise RuntimeError("MPS allocator returned invalid memory values")

    manager = ResidencyManager(
        policy=MemoryPolicy(
            inference_reserve=100,
            physical_headroom=0,
            min_weight_memory_ratio=0.0,
            load_inflation=1.0,
        ),
        free_memory=fake.free,
        empty_cache=fake.empty_cache,
        mps_snapshot=broken_snapshot,
    )
    m = OomMechanism("m", total=500, device=MPS)
    fake.mechanisms.append(m)

    with pytest.raises(RuntimeError, match="MPS backend out of memory") as excinfo:
        manager.load([m])
    assert getattr(excinfo.value, "__notes__", []) == []


# ----------------------------------------------------- memory policy


def test_minimum_inference_memory_uses_visible_server_default() -> None:
    policy = MemoryPolicy()
    assert policy.inference_reserve == int(0.8 * 1024**3)
    assert policy.physical_headroom == 256 * 1024**2
    assert policy.minimum_inference_memory() == (
        policy.inference_reserve + policy.physical_headroom
    )


def test_budget_projection_and_aimdo_headroom_enforce_one_hard_budget() -> None:
    samples = iter((DeviceMemory(32, 30), DeviceMemory(12, 5)))
    policy = MemoryPolicy(
        inference_reserve=0,
        physical_headroom=0,
        hard_budgets={"cuda:0": 24},
    )
    manager = ResidencyManager(
        policy=policy,
        free_memory=lambda _device: next(samples),
        total_memory=lambda _device: 32,
    )

    resolved = policy.resolve(VRAM, 32)
    assert resolved.budget_headroom_bytes == 8
    assert manager.policy_memory(VRAM) == DeviceMemory(24, 24)
    assert manager.policy_memory(VRAM) == DeviceMemory(4, 4)


def test_hard_budgets_apply_per_family_and_reject_malformed_keys() -> None:
    policy = MemoryPolicy(hard_budgets={"cuda:0": 24, "xpu:0": 16})
    assert policy.hard_budget(torch.device("cuda", 0)) == 24
    assert policy.hard_budget(torch.device("xpu", 0)) == 16
    assert policy.hard_budget(torch.device("xpu", 1)) is None
    assert policy.hard_budget(CPU) is None

    for key in ("mps:0", "cuda:x", "xpu", "vram:cuda:0"):
        with pytest.raises(ValueError, match="hard budget device"):
            MemoryPolicy(hard_budgets={key: 1})


def test_budget_projection_enforces_one_xpu_hard_budget() -> None:
    samples = iter((DeviceMemory(32, 30), DeviceMemory(12, 5)))
    policy = MemoryPolicy(
        inference_reserve=0,
        physical_headroom=0,
        hard_budgets={"xpu:0": 24},
    )
    manager = ResidencyManager(
        policy=policy,
        free_memory=lambda _device: next(samples),
        total_memory=lambda _device: 32,
    )

    device = torch.device("xpu", 0)
    resolved = policy.resolve(device, 32)
    assert resolved.budget_headroom_bytes == 8
    assert manager.policy_memory(device) == DeviceMemory(24, 24)
    assert manager.policy_memory(device) == DeviceMemory(4, 4)


def test_cpu_memory_introspection() -> None:
    mem = get_free_memory(CPU)
    assert 0 < mem.free_total == mem.free_torch
    assert get_total_memory(CPU) >= mem.free_total


def test_cpu_memory_introspection_uses_effective_container_values() -> None:
    snapshot = SystemMemorySnapshot(
        host_total_bytes=2000,
        host_available_bytes=1500,
        effective_total_bytes=700,
        effective_available_bytes=500,
        provenance=("injected", "cgroup-v2"),
    )

    def query() -> SystemMemorySnapshot:
        return snapshot

    assert get_free_memory(CPU, system_memory=query) == DeviceMemory(500, 500)
    assert get_total_memory(CPU, system_memory=query) == 700


def test_device_memory_projection_preserves_reference_reads_and_arithmetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, torch.device]] = []

    def memory_stats_as_nested_dict(device: torch.device) -> dict[str, object]:
        calls.append(("stats", device))
        return {
            "active_bytes": {"all": {"current": 100}},
            "reserved_bytes": {"all": {"current": 130}},
        }

    def mem_get_info(device: torch.device) -> tuple[int, int]:
        calls.append(("driver", device))
        return 80, 100

    monkeypatch.setattr(torch.cuda, "memory_stats_as_nested_dict", memory_stats_as_nested_dict)
    monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)

    # The raw compatibility projection is intentionally not capped.
    assert get_free_memory(VRAM) == DeviceMemory(110, 30)
    assert calls == [("stats", VRAM), ("driver", VRAM)]


def test_device_memory_projection_preserves_missing_and_invalid_stats_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stats: dict[str, object] = {
        "active_bytes": object(),
        "reserved_bytes": {"all": {}},
    }

    def memory_stats_as_nested_dict(_device: torch.device) -> dict[str, object]:
        return stats

    def mem_get_info(_device: torch.device) -> tuple[int, int]:
        return 80, 100

    monkeypatch.setattr(torch.cuda, "memory_stats_as_nested_dict", memory_stats_as_nested_dict)
    monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)

    assert get_free_memory(VRAM) == DeviceMemory(80, 0)

    stats["active_bytes"] = {"all": {"current": object()}}
    with pytest.raises(TypeError):
        get_free_memory(VRAM)


def test_unsupported_device_type_raises() -> None:
    with pytest.raises(ValueError, match="no memory introspection"):
        get_free_memory(torch.device("meta"))
    with pytest.raises(ValueError, match="no memory introspection"):
        get_total_memory(torch.device("meta"))


def test_mps_memory_introspection_intersects_system_and_metal_headroom() -> None:
    mps = torch.device("mps")
    snapshots = iter(
        (
            SystemMemorySnapshot(2000, 800, 1500, 700, ("injected",)),
            SystemMemorySnapshot(2000, 800, 1500, 700, ("injected",)),
            SystemMemorySnapshot(2000, 800, 1500, 300, ("injected",)),
        )
    )
    allocator = iter(((1000, 600, 250), (1000, 600, 250), (1000, 100, 50)))

    assert get_free_memory(
        mps, system_memory=lambda: next(snapshots), mps_memory=lambda: next(allocator)
    ) == DeviceMemory(
        free_total=400,
        free_torch=0,
    )
    assert (
        get_total_memory(
            mps, system_memory=lambda: next(snapshots), mps_memory=lambda: next(allocator)
        )
        == 1000
    )
    assert get_free_memory(
        mps, system_memory=lambda: next(snapshots), mps_memory=lambda: next(allocator)
    ) == DeviceMemory(
        free_total=300,
        free_torch=0,
    )


def test_mps_memory_snapshot_reads_and_derives_the_unified_budget() -> None:
    system = SystemMemorySnapshot(2000, 800, 1500, 700, ("injected",))
    snapshot = mps_memory_snapshot(
        torch.device("mps"),
        system_memory=lambda: system,
        mps_memory=lambda: (1000, 600, 250),
    )

    assert snapshot == MpsMemorySnapshot(
        recommended_max_bytes=1000,
        driver_allocated_bytes=600,
        current_allocated_bytes=250,
        system_total_bytes=1500,
        system_available_bytes=700,
    )
    assert snapshot.metal_headroom_bytes == 400
    assert snapshot.free_bytes == 400
    assert snapshot.total_bytes == 1000
    assert snapshot.device_memory() == DeviceMemory(free_total=400, free_torch=0)
    for number in ("1000", "600", "250", "400", "700", "1500"):
        assert f"{number} bytes" in snapshot.describe()


def test_mps_memory_snapshot_requires_an_mps_device() -> None:
    with pytest.raises(ValueError, match="requires an MPS device"):
        mps_memory_snapshot(torch.device("cpu"))


def test_mps_memory_introspection_clamps_exhausted_metal_headroom() -> None:
    snapshot = SystemMemorySnapshot(2000, 1500, 2000, 1500, ("injected",))

    assert get_free_memory(
        torch.device("mps"),
        system_memory=lambda: snapshot,
        mps_memory=lambda: (1000, 1200, 900),
    ) == DeviceMemory(free_total=0, free_torch=0)


@pytest.mark.parametrize("allocator", [(0, 0, 0), (-1, 0, 0), (1000, -1, 0), (1000, 0, -1)])
def test_mps_memory_introspection_rejects_invalid_allocator_values(
    allocator: tuple[int, int, int],
) -> None:
    snapshot = SystemMemorySnapshot(2000, 1500, 2000, 1500, ("injected",))

    with pytest.raises(RuntimeError, match="invalid memory values"):
        get_free_memory(
            torch.device("mps"),
            system_memory=lambda: snapshot,
            mps_memory=lambda: allocator,
        )
    with pytest.raises(RuntimeError, match="invalid memory values"):
        get_total_memory(
            torch.device("mps"),
            system_memory=lambda: snapshot,
            mps_memory=lambda: allocator,
        )


def test_soft_empty_cache_empties_the_mps_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append("mps"))
    soft_empty_cache(torch.device("mps"))
    assert calls == ["mps"]
    soft_empty_cache(CPU)
    assert calls == ["mps"]


# ------------------------------------------------ xpu memory


XPU = torch.device("xpu", 0)


def test_xpu_memory_introspection_uses_injected_snapshot() -> None:
    snapshot = XpuMemorySnapshot(
        total_bytes=1000,
        driver_free_bytes=600,
        allocator_reclaimable_bytes=50,
        driver_reported=True,
    )

    assert get_free_memory(XPU, xpu_memory=lambda _device: snapshot) == DeviceMemory(
        free_total=650,
        free_torch=50,
    )
    assert get_total_memory(XPU, xpu_memory=lambda _device: snapshot) == 1000


def test_xpu_snapshot_free_bytes_is_capped_at_total() -> None:
    snapshot = XpuMemorySnapshot(
        total_bytes=1000,
        driver_free_bytes=980,
        allocator_reclaimable_bytes=100,
        driver_reported=False,
    )

    assert snapshot.free_bytes == 1000
    # The compatibility projection is deliberately uncapped, like CUDA's.
    assert snapshot.device_memory() == DeviceMemory(free_total=1080, free_torch=100)


def test_xpu_memory_snapshot_prefers_the_driver_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, torch.device]] = []

    def memory_stats(device: torch.device) -> dict[str, int]:
        calls.append(("stats", device))
        return {
            "active_bytes.all.current": 100,
            "reserved_bytes.all.current": 130,
        }

    def mem_get_info(device: torch.device) -> tuple[int, int]:
        calls.append(("driver", device))
        # A deliberately wrong second element: total must come from the
        # device property, matching the reference's XPU arithmetic.
        return 80, 999

    def get_device_properties(_device: torch.device) -> SimpleNamespace:
        return SimpleNamespace(total_memory=200)

    monkeypatch.setattr(torch.xpu, "memory_stats", memory_stats)
    monkeypatch.setattr(torch.xpu, "mem_get_info", mem_get_info)
    monkeypatch.setattr(torch.xpu, "get_device_properties", get_device_properties)

    snapshot = xpu_memory_snapshot(XPU)
    assert snapshot == XpuMemorySnapshot(
        total_bytes=200,
        driver_free_bytes=80,
        allocator_reclaimable_bytes=30,
        driver_reported=True,
        allocator_reserved_bytes=130,
    )
    assert calls == [("stats", XPU), ("driver", XPU)]
    assert get_free_memory(XPU) == DeviceMemory(110, 30)
    assert get_total_memory(XPU) == 200


def test_xpu_memory_snapshot_rejects_missing_allocator_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def memory_stats(_device: torch.device) -> dict[str, int]:
        return {}

    monkeypatch.setattr(torch.xpu, "memory_stats", memory_stats)
    with pytest.raises(KeyError):
        xpu_memory_snapshot(XPU)


def _fake_xpu_allocator_without_mem_get_info(
    monkeypatch: pytest.MonkeyPatch, *, active: int, reserved: int, total: int
) -> None:
    def memory_stats(_device: torch.device) -> dict[str, int]:
        return {
            "active_bytes.all.current": active,
            "reserved_bytes.all.current": reserved,
        }

    def get_device_properties(_device: torch.device) -> SimpleNamespace:
        return SimpleNamespace(total_memory=total)

    monkeypatch.setattr(torch.xpu, "memory_stats", memory_stats)
    monkeypatch.setattr(torch.xpu, "mem_get_info", None)
    monkeypatch.setattr(torch.xpu, "get_device_properties", get_device_properties)


def test_xpu_memory_snapshot_derives_free_without_mem_get_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_xpu_allocator_without_mem_get_info(monkeypatch, active=100, reserved=130, total=200)

    snapshot = xpu_memory_snapshot(XPU)
    assert snapshot == XpuMemorySnapshot(
        total_bytes=200,
        driver_free_bytes=70,
        allocator_reclaimable_bytes=30,
        driver_reported=False,
        allocator_reserved_bytes=130,
    )


def test_xpu_memory_snapshot_clamps_derived_free_at_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_xpu_allocator_without_mem_get_info(monkeypatch, active=250, reserved=250, total=200)

    assert xpu_memory_snapshot(XPU).driver_free_bytes == 0


def test_xpu_memory_snapshot_requires_an_xpu_device() -> None:
    with pytest.raises(ValueError, match="requires an XPU device"):
        xpu_memory_snapshot(CPU)


def test_soft_empty_cache_synchronizes_and_empties_the_xpu_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def synchronize(_device: torch.device) -> None:
        calls.append("sync")

    monkeypatch.setattr(torch.xpu, "synchronize", synchronize)
    monkeypatch.setattr(torch.xpu, "empty_cache", lambda: calls.append("empty"))
    soft_empty_cache(XPU)
    assert calls == ["sync", "empty"]
    soft_empty_cache(CPU)
    assert calls == ["sync", "empty"]


# ------------------------------------------------ lora compute dtype


def _fake_cuda_props(monkeypatch: pytest.MonkeyPatch, *, major: int, minor: int, name: str) -> None:
    monkeypatch.setattr(torch.version, "hip", None)

    def fake_props(_device: torch.device) -> SimpleNamespace:
        return SimpleNamespace(major=major, minor=minor, name=name)

    monkeypatch.setattr(torch.cuda, "get_device_properties", fake_props)


def test_lora_compute_dtype_is_float32_on_cpu() -> None:
    assert lora_compute_dtype(CPU) is torch.float32


def test_lora_compute_dtype_is_float16_on_mps() -> None:
    assert lora_compute_dtype(torch.device("mps")) is torch.float16


def test_lora_compute_dtype_rejects_unsupported_device_types() -> None:
    with pytest.raises(ValueError, match="no fp16 dtype policy"):
        lora_compute_dtype(torch.device("meta"))


def test_lora_compute_dtype_is_float32_on_intel_accelerators() -> None:
    # Runtime fp16 LoRA patching has produced NaN output on XPU where a
    # CPU float32 merge works (ComfyUI issue #14720): patch in float32.
    assert lora_compute_dtype(torch.device("xpu")) is torch.float32


def test_lora_compute_dtype_matches_reference_capability_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda = torch.device("cuda", 0)

    _fake_cuda_props(monkeypatch, major=12, minor=0, name="NVIDIA RTX PRO 6000")
    assert lora_compute_dtype(cuda) is torch.float16

    _fake_cuda_props(monkeypatch, major=5, minor=2, name="GeForce GTX 970")
    assert lora_compute_dtype(cuda) is torch.float32

    _fake_cuda_props(monkeypatch, major=7, minor=0, name="Tesla V100-SXM2-16GB")
    assert lora_compute_dtype(cuda) is torch.float16

    # 16-series cards match case-sensitively and stay float32.
    _fake_cuda_props(monkeypatch, major=7, minor=5, name="NVIDIA GeForce GTX 1660")
    assert lora_compute_dtype(cuda) is torch.float32

    _fake_cuda_props(monkeypatch, major=7, minor=5, name="NVIDIA T500")
    assert lora_compute_dtype(cuda) is torch.float32

    _fake_cuda_props(monkeypatch, major=7, minor=5, name="nvidia t500")
    assert lora_compute_dtype(cuda) is torch.float16

    # 10-series cards get float16 only on Windows.
    on_windows = torch.float16 if any(platform.win32_ver()) else torch.float32

    _fake_cuda_props(monkeypatch, major=6, minor=0, name="Tesla P100-PCIE-16GB")
    assert lora_compute_dtype(cuda) is on_windows

    # TITAN Xp lands in the 10-series list through its "titan x" prefix.
    _fake_cuda_props(monkeypatch, major=6, minor=1, name="NVIDIA TITAN Xp")
    assert lora_compute_dtype(cuda) is on_windows


def test_lora_compute_dtype_is_float16_on_hip_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.version, "hip", "6.0.0")
    assert lora_compute_dtype(torch.device("cuda", 0)) is torch.float16


# -------------------------------------------------------- aimdo seam


def test_probe_aimdo_never_raises_and_is_consistent() -> None:
    status = probe_aimdo()
    if not status.importable:
        assert not status.initialized
        assert status.vendor is None
    # in-process init must not have happened: nothing in Dinkster calls
    # control.init(), and it would have had to run before torch import
    assert not status.initialized


# ---------------------------------------------------- manager: remove


def test_remove_drops_mechanism_from_registry() -> None:
    manager, fake = make_manager(1000)
    m = FakeMechanism("m", total=500, device=VRAM)
    fake.mechanisms.append(m)
    manager.load([m])
    assert manager.registered() == (m,)
    manager.remove([m])
    assert manager.registered() == ()
    assert m.loaded == 0


def test_remove_unregistered_mechanism_is_silently_ignored() -> None:
    """free() can autonomously detach-and-pop a mechanism during
    eviction, so a terminal-release path must tolerate already-removed
    entries without tracking manager internals."""
    manager, fake = make_manager(1000)
    registered = FakeMechanism("r", total=200, device=VRAM)
    fake.mechanisms.append(registered)
    manager.load([registered])
    stranger = FakeMechanism("s", total=100, device=VRAM)
    manager.remove([stranger])  # never registered: no raise
    assert manager.registered() == (registered,)
    # unload=True defensively unloads even the unregistered mechanism
    assert ("unload", None) in stranger.calls


def test_remove_unload_true_restores_real_store() -> None:
    """Through a real ResidentWeights: remove() unloads, restoring the
    exact original patched-storage objects, and deregisters."""
    store = make_store(("a", 4))
    original_a = store["a"]
    delta = torch.full((4, 4), 0.25)
    resident = ResidentWeights(
        store,
        load_device=CPU,
        offload_device=CPU,
        patch_set=diff_set("a", delta),
    )
    manager = ResidencyManager()  # cpu-only: defaults never touch cuda
    manager.load([resident])
    assert resident.loaded_bytes() == resident.total_bytes()
    assert store["a"] is not original_a  # patched storage resident
    manager.remove([resident])
    assert manager.registered() == ()
    assert resident.loaded_bytes() == 0
    assert store["a"] is original_a  # exact original object restored


def test_remove_unload_false_deregisters_only() -> None:
    manager, fake = make_manager(1000)
    m = FakeMechanism("m", total=500, device=VRAM)
    fake.mechanisms.append(m)
    manager.load([m])
    assert m.loaded == 500
    calls_before = fake.empty_cache_calls
    manager.remove([m], unload=False)
    assert manager.registered() == ()
    assert m.loaded == 500  # placement untouched
    assert ("unload", None) not in m.calls
    assert fake.empty_cache_calls == calls_before


def test_remove_empties_cache_once_per_affected_device() -> None:
    manager, fake = make_manager(1000)
    a = FakeMechanism("a", total=200, device=VRAM)
    b = FakeMechanism("b", total=200, device=VRAM)
    fake.mechanisms.extend([a, b])
    manager.load([a, b])
    calls_before = fake.empty_cache_calls
    manager.remove([a, b])
    # one shared non-cpu device: exactly one cache clear
    assert fake.empty_cache_calls == calls_before + 1
    # nothing loaded before the unload: no cache clear
    c = FakeMechanism("c", total=50, device=VRAM)
    manager.remove([c])
    assert fake.empty_cache_calls == calls_before + 1
    # cpu load device: never cleared even when loaded
    d = FakeMechanism("d", total=50, device=CPU, loaded=50)
    manager.remove([d])
    assert d.loaded == 0
    assert fake.empty_cache_calls == calls_before + 1


def test_remove_dedups_duplicate_mechanisms() -> None:
    manager, fake = make_manager(1000)
    m = FakeMechanism("m", total=300, device=VRAM)
    fake.mechanisms.append(m)
    manager.load([m])
    manager.remove([m, m])
    assert m.calls.count(("unload", None)) == 1
    assert manager.registered() == ()


class _DiscardMechanism(FakeMechanism):
    def discard(self) -> None:
        self.calls.append(("discard", None))
        self.loaded = 0


@pytest.mark.parametrize("registered", [False, True])
def test_remove_discard_capability_deduplicates_without_emptying_cache(registered: bool) -> None:
    manager, fake = make_manager(1000)
    mechanism = _DiscardMechanism("discard", total=200, device=VRAM, loaded=200)
    if registered:
        manager.load([mechanism])
    calls_before = fake.empty_cache_calls
    manager.remove([mechanism, mechanism], discard=True)
    assert mechanism.calls.count(("discard", None)) == 1
    assert ("unload", None) not in mechanism.calls
    assert mechanism.loaded_bytes() == 0
    assert fake.empty_cache_calls == calls_before
    assert manager.registered() == ()


def test_remove_discard_falls_back_to_full_unload_and_cache_cleanup() -> None:
    from dinkster_inference_torch.aimdo_residency import AimdoWeights

    assert not issubclass(AimdoWeights, residency_mod.DiscardableResidency)
    manager, fake = make_manager(1000)
    mechanism = FakeMechanism("fallback", total=200, device=VRAM, loaded=200)
    manager.load([mechanism])
    calls_before = fake.empty_cache_calls
    manager.remove([mechanism], discard=True)
    assert mechanism.calls.count(("unload", None)) == 1
    assert mechanism.loaded_bytes() == 0
    assert fake.empty_cache_calls == calls_before + 1
    assert manager.registered() == ()


def test_remove_discard_rejects_deregistration_only_before_mutation() -> None:
    manager, _fake = make_manager(1000)
    mechanism = _DiscardMechanism("discard", total=200, device=VRAM, loaded=200)
    manager.load([mechanism])
    with pytest.raises(ValueError, match="discard requires unload=True"):
        manager.remove([mechanism], unload=False, discard=True)
    assert manager.registered() == (mechanism,)
    assert mechanism.loaded_bytes() == 200
    assert mechanism.calls == []


@pytest.mark.parametrize("discard_capable", [False, True])
def test_remove_discard_failure_preserves_registry_and_skips_cache_cleanup(
    monkeypatch: pytest.MonkeyPatch, discard_capable: bool
) -> None:
    manager, fake = make_manager(1000)
    first = _DiscardMechanism("first", total=200, device=VRAM, loaded=200)
    kind = _DiscardMechanism if discard_capable else FakeMechanism
    failing = kind("failing", total=200, device=VRAM, loaded=200)
    manager.load([first, failing])
    before = manager.registered()
    cache_before = fake.empty_cache_calls

    def fail() -> None:
        raise RuntimeError("terminal cleanup failed")

    monkeypatch.setattr(failing, "discard" if discard_capable else "unload", fail)
    with pytest.raises(RuntimeError, match="terminal cleanup failed"):
        manager.remove([first, failing], discard=True)
    assert first.loaded_bytes() == 0
    assert failing.loaded_bytes() == 200
    assert manager.registered() == before
    assert fake.empty_cache_calls == cache_before


@pytest.mark.parametrize(
    ("device", "rocm", "expected"),
    [
        (torch.device("cuda", 0), False, 0.0),
        (torch.device("cuda", 0), True, 0.4),
        (torch.device("xpu", 0), False, 0.4),
        (torch.device("mps"), False, 0.4),
    ],
)
def test_default_weight_memory_ratio_is_selected_by_accelerator_family(
    device: torch.device, rocm: bool, expected: float
) -> None:
    assert MemoryPolicy().weight_memory_ratio(device, rocm=rocm) == expected


def test_explicit_weight_memory_ratio_overrides_accelerator_family() -> None:
    policy = MemoryPolicy(min_weight_memory_ratio=0.17)
    assert policy.weight_memory_ratio(torch.device("cuda", 0), rocm=False) == 0.17
    assert policy.weight_memory_ratio(torch.device("cuda", 0), rocm=True) == 0.17
    assert policy.weight_memory_ratio(torch.device("xpu", 0)) == 0.17
    assert policy.weight_memory_ratio(torch.device("mps")) == 0.17


@pytest.mark.parametrize(
    ("device", "hip_version", "expected_budget"),
    [
        (torch.device("cuda", 0), None, 200),
        (torch.device("cuda", 0), "6.0.0", 400),
        (torch.device("xpu", 0), None, 400),
        (torch.device("mps"), None, 400),
    ],
)
def test_classic_ratio_default_preserves_the_exact_budget_formula(
    monkeypatch: pytest.MonkeyPatch,
    device: torch.device,
    hip_version: str | None,
    expected_budget: int,
) -> None:
    monkeypatch.setattr(torch.version, "hip", hip_version)
    fake = FakeDevice(1000, device)
    manager = ResidencyManager(
        policy=MemoryPolicy(
            inference_reserve=100,
            physical_headroom=0,
            load_inflation=1.0,
        ),
        free_memory=fake.free,
        empty_cache=fake.empty_cache,
    )
    mechanism = FakeMechanism("oversized", total=2000, device=device)
    fake.mechanisms.append(mechanism)

    manager.load([mechanism], minimum_memory=800)

    assert mechanism.calls == [("partially_load", expected_budget)]


@pytest.mark.parametrize(
    ("reserved", "expected_available"),
    [(32, 1), (33, 0), (34, 0)],
)
def test_intel_hard_budget_caps_allocator_residency_without_double_counting_cache(
    reserved: int, expected_available: int
) -> None:
    device = torch.device("xpu", 0)
    policy = MemoryPolicy(
        inference_reserve=0,
        physical_headroom=0,
        hard_budgets={"xpu:0": 24},
    )
    measured = DeviceMemory(
        free_total=100,
        free_torch=9,
        allocator_reserved_bytes=reserved,
    )
    manager = ResidencyManager(
        policy=policy,
        free_memory=lambda _device: measured,
        total_memory=lambda _device: 100,
    )

    projected = manager.policy_memory(device)

    assert projected.free_total == expected_available
    assert projected.free_torch == expected_available


def test_intel_hard_budget_refuses_weight_residency_that_would_cross_allocator_cap() -> None:
    device = torch.device("xpu", 0)
    mechanism = FakeMechanism("oversized", total=10, device=device)

    def allocator_memory(_device: torch.device) -> DeviceMemory:
        active = 20 + mechanism.loaded
        reserved = max(24, active)
        reclaimable = reserved - active
        return DeviceMemory(
            free_total=100 - reserved + reclaimable,
            free_torch=reclaimable,
            allocator_reserved_bytes=reserved,
        )

    manager = ResidencyManager(
        policy=MemoryPolicy(
            inference_reserve=0,
            physical_headroom=0,
            hard_budgets={"xpu:0": 24},
            load_inflation=1.0,
        ),
        free_memory=allocator_memory,
        total_memory=lambda _device: 100,
        empty_cache=lambda _device: None,
    )

    manager.load([mechanism])

    assert mechanism.calls == [("partially_load", 4)]
    assert mechanism.loaded == 4
    assert manager.policy_memory(device).free_total == 0


def test_intel_hard_budget_unloads_existing_weights_that_exceed_allocator_cap() -> None:
    device = torch.device("xpu", 0)
    mechanism = FakeMechanism("over-budget", total=10, loaded=10, device=device)
    cache_releases: list[torch.device] = []

    def allocator_memory(_device: torch.device) -> DeviceMemory:
        active = 20 + mechanism.loaded
        return DeviceMemory(
            free_total=100 - active,
            free_torch=0,
            allocator_reserved_bytes=active,
        )

    manager = ResidencyManager(
        policy=MemoryPolicy(
            inference_reserve=0,
            physical_headroom=0,
            hard_budgets={"xpu:0": 24},
            load_inflation=1.0,
        ),
        free_memory=allocator_memory,
        total_memory=lambda _device: 100,
        empty_cache=cache_releases.append,
    )

    manager.load([mechanism])

    assert mechanism.calls == [("partially_load", -6)]
    assert mechanism.loaded == 4
    assert cache_releases == [device]
    assert manager.policy_memory(device).allocator_budget_debt_bytes == 0


def test_intel_allocator_residency_does_not_affect_uncapped_devices() -> None:
    measured = DeviceMemory(
        free_total=100,
        free_torch=9,
        allocator_reserved_bytes=80,
    )
    manager = ResidencyManager(free_memory=lambda _device: measured)

    assert manager.policy_memory(torch.device("xpu", 0)) is measured


def test_intel_memory_snapshot_uses_direct_counters_before_stats_are_initialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def memory_stats(_device: torch.device) -> dict[str, int]:
        return {}

    def zero_counter(_device: torch.device) -> int:
        return 0

    def mem_get_info(_device: torch.device) -> tuple[int, int]:
        return 180, 200

    def get_device_properties(_device: torch.device) -> SimpleNamespace:
        return SimpleNamespace(total_memory=200)

    monkeypatch.setattr(torch.xpu, "memory_stats", memory_stats)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(torch.xpu, "memory_allocated", zero_counter)
    monkeypatch.setattr(torch.xpu, "memory_reserved", zero_counter)
    monkeypatch.setattr(torch.xpu, "mem_get_info", mem_get_info)
    monkeypatch.setattr(torch.xpu, "get_device_properties", get_device_properties)

    assert xpu_memory_snapshot(XPU) == XpuMemorySnapshot(
        total_bytes=200,
        driver_free_bytes=180,
        allocator_reclaimable_bytes=0,
        driver_reported=True,
        allocator_reserved_bytes=0,
    )


def test_force_full_load_does_not_override_an_intel_hard_budget() -> None:
    device = torch.device("xpu", 0)
    mechanism = FakeMechanism("force-full", total=10, device=device)

    def allocator_memory(_device: torch.device) -> DeviceMemory:
        active = 20 + mechanism.loaded
        return DeviceMemory(
            free_total=100 - active,
            free_torch=0,
            allocator_reserved_bytes=active,
        )

    manager = ResidencyManager(
        policy=MemoryPolicy(
            inference_reserve=0,
            physical_headroom=0,
            hard_budgets={"xpu:0": 24},
            load_inflation=1.0,
        ),
        free_memory=allocator_memory,
        total_memory=lambda _device: 100,
        empty_cache=lambda _device: None,
    )

    manager.load([mechanism], force_full_load=True)

    assert mechanism.calls == [("partially_load", 4)]
    assert mechanism.loaded == 4
    assert manager.policy_memory(device).allocator_budget_debt_bytes == 0
