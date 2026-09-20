"""Encoded-resident GGUF linear execution and block decoders.

The residency modes' correctness rests on one equivalence: a
GgufEncodedLinear forward (decode blocks to float32, cast to the
compute dtype at use, F.linear) is bit-identical to the eager
loader's decode-at-load path (same decode, same cast, same linear).
These tests pin every vectorized block decoder against the pure
reference decoder and the forward against an eagerly materialized
weight, for every encoded-resident layout.
"""

from __future__ import annotations

import struct
from collections.abc import Callable

import pytest
import torch
from dinkster_inference import (
    Q4_0,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_0,
    GGMLType,
    builtin_gguf_storage_registry,
    decode_ggml_blocks,
)
from dinkster_inference.patches import DiffPatch, PatchEntry, PatchSet
from dinkster_inference_torch import (
    DeviceMemory,
    GgufDecodedCache,
    GgufDecodedCacheResidency,
    GgufEncodedLinear,
    MemoryPolicy,
    ModuleStateStore,
    PartialResidencyTiming,
    PatchApplyError,
    ResidencyManager,
    collect_partial_residency_timing,
    enroll_component,
)
from dinkster_inference_torch import gguf_linear as gguf_linear_mod
from dinkster_inference_torch import residency as residency_mod
from dinkster_inference_torch.gguf_linear import (
    FUSED_MATMUL_MAX_TOKENS,
    GGUF_BLOCK_DECODERS,
    GGUF_BLOCK_SHAPES,
    Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_ELEMENTS,
    decode_q8_0_blocks,
    synthetic_gguf_blocks,
)
from dinkster_inference_torch.model_prefetch import make_prefetch_queue, prefetch_queue_pop

#: float16 scale fields exercising zero, the smallest subnormal, the
#: smallest normal, ordinary magnitudes, both signs, and the extremes.
_FP16_EDGE_SCALES = (
    0.0,
    5.9604644775390625e-08,
    6.103515625e-05,
    0.5,
    1.0,
    -1.5,
    3.140625,
    -448.0,
    65504.0,
)

#: Byte offsets of every float16 scale field per encoded layout.
_FP16_FIELD_OFFSETS = {
    "Q4_0": (0,),
    "Q8_0": (0,),
    "Q4_K": (0, 2),
    "Q5_K": (0, 2),
    "Q6_K": (208,),
}


def reference_quant_blocks(ggml_type: GGMLType, count: int, seed: int) -> torch.Tensor:
    """Random encoded blocks whose float16 scale fields are finite.

    Quant, scale-pack, and plane bytes take every value; the float16
    fields cycle through edge scales so decoded values stay finite
    (torch.equal treats NaN as unequal) while covering subnormal,
    zero, negative, and near-maximum scales.
    """

    generator = torch.Generator().manual_seed(seed)
    blocks = torch.randint(
        0, 256, (count, ggml_type.block_bytes), dtype=torch.uint8, generator=generator
    )
    offsets = _FP16_FIELD_OFFSETS[ggml_type.name]
    for row in range(count):
        for position, offset in enumerate(offsets):
            value = _FP16_EDGE_SCALES[(row * len(offsets) + position) % len(_FP16_EDGE_SCALES)]
            blocks[row, offset : offset + 2] = torch.frombuffer(
                bytearray(struct.pack("<e", value)), dtype=torch.uint8
            )
    return blocks


def encode_q8_0(weight: torch.Tensor) -> torch.Tensor:
    """Reference Q8_0 encoder: per-32-block float16 scale = amax/127."""
    flat = weight.to(torch.float32).reshape(-1, Q8_0_BLOCK_ELEMENTS)
    scales = (flat.abs().amax(dim=1) / 127.0).to(torch.float16)
    wide = scales.to(torch.float32).reshape(-1, 1)
    quants = torch.where(wide == 0.0, torch.zeros_like(flat), flat / wide.clamp(min=1e-30))
    quants = quants.round().clamp(-127, 127).to(torch.int8)
    blocks = torch.empty((flat.shape[0], Q8_0_BLOCK_BYTES), dtype=torch.uint8)
    blocks[:, :2] = scales.view(torch.uint8).reshape(-1, 2)
    blocks[:, 2:] = quants.view(torch.uint8)
    return blocks


def test_block_decoder_registry_covers_every_encoded_layout() -> None:
    layouts = builtin_gguf_storage_registry()
    assert set(GGUF_BLOCK_DECODERS) == {layout.ggml_type.name for layout in layouts}
    assert set(GGUF_BLOCK_SHAPES) == set(GGUF_BLOCK_DECODERS)
    for layout in layouts:
        assert GGUF_BLOCK_SHAPES[layout.ggml_type.name] == (
            layout.ggml_type.block_elements,
            layout.ggml_type.block_bytes,
        )


@pytest.mark.parametrize(
    "ggml_type", (Q4_0, Q4_K, Q5_K, Q6_K, Q8_0), ids=lambda ggml_type: ggml_type.name
)
def test_vectorized_decoder_is_bit_identical_to_pure_reference(ggml_type: GGMLType) -> None:
    decoder: Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor]
    decoder = GGUF_BLOCK_DECODERS[ggml_type.name]
    count = 2 * len(_FP16_EDGE_SCALES)
    blocks = reference_quant_blocks(ggml_type, count, seed=ggml_type.code)
    payload = bytes(blocks.flatten().tolist())
    expected = torch.tensor(decode_ggml_blocks(ggml_type, payload), dtype=torch.float32)

    decoded = decoder(blocks, (count, ggml_type.block_elements))

    assert decoded.dtype == torch.float32
    assert torch.equal(
        decoded.view(torch.int32),
        expected.reshape(count, ggml_type.block_elements).view(torch.int32),
    )


def test_decode_matches_pure_reference_decoder() -> None:
    encoded = struct.pack("<H", 0x3800) + bytes((index * 9 - 127) & 0xFF for index in range(32))
    payload = encoded * 3
    blocks = torch.frombuffer(bytearray(payload), dtype=torch.uint8).reshape(-1, Q8_0_BLOCK_BYTES)
    expected = torch.tensor(decode_ggml_blocks(Q8_0, payload), dtype=torch.float32)

    decoded = decode_q8_0_blocks(blocks, (3, Q8_0_BLOCK_ELEMENTS))

    assert decoded.dtype == torch.float32
    assert torch.equal(decoded, expected.reshape(3, Q8_0_BLOCK_ELEMENTS))


def test_encoder_decode_round_trip_is_exact_for_representable_values() -> None:
    # Each block's largest magnitude is exactly 127 so the encoder's
    # amax/127 recovers the constructed float16 scale bit-exactly.
    torch.manual_seed(0)
    quants = torch.randint(-127, 128, (16, Q8_0_BLOCK_ELEMENTS), dtype=torch.int8)
    quants[:, 0] = 127
    scales = torch.rand(16, 1).to(torch.float16).to(torch.float32)
    weight = (quants.to(torch.float32) * scales).reshape(8, 64)

    assert torch.equal(decode_q8_0_blocks(encode_q8_0(weight), (8, 64)), weight)


@pytest.mark.parametrize("compute_dtype", (torch.float32, torch.bfloat16, torch.float16))
def test_forward_is_bit_identical_to_eagerly_decoded_linear(
    compute_dtype: torch.dtype,
) -> None:
    torch.manual_seed(1)
    blocks = encode_q8_0(torch.randn(8, 64))
    decoded = decode_q8_0_blocks(blocks, (8, 64))
    bias = torch.randn(8).to(compute_dtype)

    module = GgufEncodedLinear(64, 8, bias=True, compute_dtype=compute_dtype)
    module.load_state_dict({"weight_blocks": blocks, "bias": bias})
    x = torch.randn(3, 64).to(compute_dtype)

    expected = torch.nn.functional.linear(x, decoded.to(compute_dtype), bias)
    output = module(x)
    assert output.dtype == compute_dtype
    assert torch.equal(output, expected)

    # Module-wide dtype casts must not disturb the encoded payload.
    module.to(torch.bfloat16)
    assert module.weight_blocks.dtype == torch.uint8
    assert torch.equal(module.weight_blocks, blocks)


@pytest.mark.parametrize(
    "ggml_type", (Q4_0, Q4_K, Q5_K, Q6_K, Q8_0), ids=lambda ggml_type: ggml_type.name
)
def test_every_layout_forward_matches_the_eager_decode_path(ggml_type: GGMLType) -> None:
    # One superblock row keeps every layout's in_features aligned to
    # its block size; four rows exercise distinct scale fields.
    in_features = ggml_type.block_elements
    out_features = 4
    blocks = reference_quant_blocks(ggml_type, out_features, seed=ggml_type.code + 100)
    decoded = GGUF_BLOCK_DECODERS[ggml_type.name](blocks, (out_features, in_features))
    torch.manual_seed(3)
    bias = torch.randn(out_features).to(torch.bfloat16)
    x = torch.randn(3, in_features).to(torch.bfloat16)

    module = GgufEncodedLinear(
        in_features, out_features, ggml_type=ggml_type.name, bias=True, compute_dtype=torch.bfloat16
    )
    module.load_state_dict({"weight_blocks": blocks, "bias": bias})
    expected = torch.nn.functional.linear(x, decoded.to(torch.bfloat16), bias)
    assert torch.equal(module(x), expected)

    cache = GgufDecodedCache(1 << 20)
    cached = GgufEncodedLinear(
        in_features,
        out_features,
        ggml_type=ggml_type.name,
        bias=True,
        compute_dtype=torch.bfloat16,
        decoded_cache=cache,
        cache_key="layer",
    )
    cached.load_state_dict({"weight_blocks": blocks, "bias": bias})
    assert torch.equal(cached(x), expected)
    assert cache.get("layer") is not None


def test_bias_free_module_and_block_geometry_validation() -> None:
    module = GgufEncodedLinear(Q8_0_BLOCK_ELEMENTS, 1, bias=False, compute_dtype=torch.float32)
    assert module.bias is None
    module.load_state_dict({"weight_blocks": encode_q8_0(torch.ones(1, Q8_0_BLOCK_ELEMENTS))})
    assert module(torch.zeros(1, Q8_0_BLOCK_ELEMENTS)).shape == (1, 1)

    with pytest.raises(ValueError, match="does not split into 32-element blocks"):
        GgufEncodedLinear(3, 5, bias=False, compute_dtype=torch.float32)
    with pytest.raises(ValueError, match="does not split into 256-element blocks"):
        GgufEncodedLinear(32, 4, ggml_type="Q4_K", bias=False, compute_dtype=torch.float32)
    with pytest.raises(ValueError, match="no encoded-resident layout"):
        GgufEncodedLinear(32, 4, ggml_type="F16", bias=False, compute_dtype=torch.float32)


def test_decoded_cache_sticky_first_fill_and_accounting() -> None:
    cache = GgufDecodedCache(1024)
    assert cache.budget_bytes == 1024
    assert cache.used_bytes == 0

    first = torch.zeros(128, dtype=torch.float32)  # 512 bytes
    assert cache.offer("first", first) is first
    assert cache.get("first") is first
    assert cache.used_bytes == 512

    # A resident entry wins over a competing offer under the same key.
    competing = torch.ones(128, dtype=torch.float32)
    assert cache.offer("first", competing) is first
    assert cache.used_bytes == 512

    # An entry that does not fit is skipped without displacing anything,
    # and a later smaller entry still fills the remaining budget.
    too_big = torch.zeros(256, dtype=torch.float32)  # 1024 bytes
    assert cache.offer("big", too_big) is too_big
    assert cache.get("big") is None
    assert cache.used_bytes == 512
    smaller = torch.zeros(128, dtype=torch.float32)
    assert cache.offer("second", smaller) is smaller
    assert cache.used_bytes == 1024

    # Eviction walks reverse insertion order and reports freed bytes.
    assert cache.free_bytes(1) == 512
    assert cache.get("second") is None
    assert cache.get("first") is first
    assert cache.used_bytes == 512

    cache.clear()
    assert cache.used_bytes == 0
    assert cache.get("first") is None
    assert cache.budget_bytes == 1024


def test_decoded_cache_zero_budget_and_validation() -> None:
    cache = GgufDecodedCache(0)
    weight = torch.zeros(8, dtype=torch.float32)
    assert cache.offer("key", weight) is weight
    assert cache.get("key") is None
    assert cache.used_bytes == 0

    with pytest.raises(ValueError, match="non-negative byte count or None"):
        GgufDecodedCache(-1)
    with pytest.raises(ValueError, match="non-negative byte count or None"):
        GgufDecodedCache(True)  # type: ignore[arg-type]


def test_decoded_cache_auto_admission_tracks_live_free_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_inference_torch import memory as memory_module

    reserve = memory_module.MemoryPolicy().minimum_inference_memory()
    free = {"bytes": reserve + 1024}

    def fake_free(device: torch.device) -> memory_module.DeviceMemory:
        assert device.type == "cpu"
        return memory_module.DeviceMemory(free_total=free["bytes"], free_torch=free["bytes"])

    monkeypatch.setattr(memory_module, "get_free_memory", fake_free)

    cache = GgufDecodedCache(None)
    assert cache.budget_bytes is None

    weight = torch.zeros(128, dtype=torch.float32)  # 512 bytes
    assert cache.offer("key", weight) is weight
    assert cache.get("key") is weight
    assert cache.budget_bytes is None

    # A device squeezed below the working reserve refuses new offers
    # without touching resident entries.
    free["bytes"] = reserve - 1
    refused = torch.zeros(128, dtype=torch.float32)
    assert cache.offer("refused", refused) is refused
    assert cache.get("refused") is None
    assert cache.get("key") is weight

    # The refused key is admitted once the reserve is free again.
    free["bytes"] = reserve
    assert cache.offer("refused", refused) is refused
    assert cache.get("refused") is refused


def test_decoded_cache_auto_admission_shares_a_device_without_overcommit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two auto caches (two components or two models) admit against the
    # same live free-memory figure: once resident decoded weights push
    # the device to the working reserve, every auto cache on it stops
    # admitting, so combined usage never overcommits.
    from dinkster_inference_torch import memory as memory_module

    reserve = memory_module.MemoryPolicy().minimum_inference_memory()
    device = {"free": reserve + 1024}

    def fake_free(_device: torch.device) -> memory_module.DeviceMemory:
        return memory_module.DeviceMemory(free_total=device["free"], free_torch=device["free"])

    monkeypatch.setattr(memory_module, "get_free_memory", fake_free)

    first = GgufDecodedCache(None)
    second = GgufDecodedCache(None)

    def offer(cache: GgufDecodedCache, key: str) -> bool:
        weight = torch.zeros(128, dtype=torch.float32)  # 512 bytes
        device["free"] -= 512  # the decoded weight exists before the offer
        cache.offer(key, weight)
        admitted = cache.get(key) is weight
        if not admitted:
            device["free"] += 512  # a refused weight is dropped after use
        return admitted

    assert offer(first, "a")
    assert offer(second, "b")
    assert not offer(first, "c")
    assert not offer(second, "d")
    assert first.used_bytes + second.used_bytes == 1024
    assert device["free"] == reserve


def test_cached_forward_is_bit_identical_and_reuses_the_decoded_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(2)
    blocks = encode_q8_0(torch.randn(8, 64))
    bias = torch.randn(8).to(torch.bfloat16)
    state = {"weight_blocks": blocks, "bias": bias}
    x = torch.randn(3, 64).to(torch.bfloat16)

    uncached = GgufEncodedLinear(64, 8, bias=True, compute_dtype=torch.bfloat16)
    uncached.load_state_dict(state)

    cache = GgufDecodedCache(1 << 20)
    cached = GgufEncodedLinear(
        64, 8, bias=True, compute_dtype=torch.bfloat16, decoded_cache=cache, cache_key="layer"
    )
    cached.load_state_dict(state)

    expected = uncached(x)
    assert torch.equal(cached(x), expected)
    resident = cache.get("layer")
    assert resident is not None
    assert resident.dtype == torch.bfloat16
    assert cache.used_bytes == resident.numel() * resident.element_size()

    # The second forward consumes the cached weight without re-decoding.
    def poisoned_decode(blocks: torch.Tensor, logical_shape: tuple[int, ...]) -> torch.Tensor:
        raise AssertionError("cache hit must not re-decode")

    monkeypatch.setattr(cached, "_decode", poisoned_decode)
    assert torch.equal(cached(x), expected)
    assert cache.get("layer") is resident


def test_cached_forward_evicts_decoded_weights_and_retries_after_decode_oom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = GgufDecodedCache(1 << 20)
    resident = _cached_linear(cache, "resident", torch.eye(32))
    candidate_weight = torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32) / 1000
    candidate = _cached_linear(cache, "candidate", candidate_weight)
    input = torch.arange(32, dtype=torch.float32).reshape(1, 32) / 100

    resident(torch.zeros(1, 32))
    assert cache.get("resident") is not None

    attempts = 0

    def decode_after_eviction(blocks: torch.Tensor, logical_shape: tuple[int, ...]) -> torch.Tensor:
        nonlocal attempts
        attempts += 1
        if cache.get("resident") is not None:
            raise torch.OutOfMemoryError("decoded cache exhausted device memory")
        return decode_q8_0_blocks(blocks, logical_shape)

    monkeypatch.setattr(candidate, "_decode", decode_after_eviction)

    expected = torch.nn.functional.linear(
        input, decode_q8_0_blocks(candidate.weight_blocks, (32, 32))
    )
    assert torch.equal(candidate(input), expected)
    assert attempts == 2
    assert cache.get("resident") is None
    assert cache.get("candidate") is not None


def test_module_moves_clear_the_shared_cache_and_key_is_required() -> None:
    blocks = encode_q8_0(torch.randn(4, 32))
    cache = GgufDecodedCache(1 << 20)
    module = GgufEncodedLinear(
        32, 4, bias=False, compute_dtype=torch.float32, decoded_cache=cache, cache_key="layer"
    )
    module.load_state_dict({"weight_blocks": blocks})

    x = torch.randn(2, 32)
    expected = module(x)
    assert cache.get("layer") is not None

    moved = module.to(torch.bfloat16)
    assert cache.get("layer") is None
    assert cache.used_bytes == 0
    assert torch.equal(moved(x), expected)

    with pytest.raises(ValueError, match="requires a non-empty cache key"):
        GgufEncodedLinear(32, 4, bias=False, compute_dtype=torch.float32, decoded_cache=cache)


def test_synthetic_blocks_are_deterministic_and_decode_finite() -> None:
    for layout in sorted(GGUF_BLOCK_DECODERS):
        elements, block_bytes = GGUF_BLOCK_SHAPES[layout]
        first = synthetic_gguf_blocks(layout, 6, seed=591)
        second = synthetic_gguf_blocks(layout, 6, seed=591)

        assert first.dtype == torch.uint8
        assert first.shape == (6, block_bytes)
        assert torch.equal(first, second)
        assert not torch.equal(first, synthetic_gguf_blocks(layout, 6, seed=592))

        decoded = GGUF_BLOCK_DECODERS[layout](first, (6, elements))
        assert decoded.shape == (6, elements)
        assert bool(torch.isfinite(decoded).all())


def test_synthetic_blocks_reject_unknown_layouts() -> None:
    with pytest.raises(ValueError, match="no encoded GGUF layout"):
        synthetic_gguf_blocks("Q2_K", 1, seed=591)


def _cached_linear(cache: GgufDecodedCache, key: str, weight: torch.Tensor) -> GgufEncodedLinear:
    linear = GgufEncodedLinear(
        weight.shape[1],
        weight.shape[0],
        bias=False,
        compute_dtype=torch.float32,
        decoded_cache=cache,
        cache_key=key,
    )
    linear.load_state_dict({"weight_blocks": encode_q8_0(weight)})
    return linear


def test_decoded_cache_residency_accounting_and_eviction() -> None:
    torch.manual_seed(7)
    cache = GgufDecodedCache(1 << 20)
    module = torch.nn.ModuleList(
        _cached_linear(cache, key, torch.randn(8, 64)) for key in ("first", "second")
    )

    mechanism = GgufDecodedCacheResidency(module)
    assert mechanism.load_device == torch.device("cpu")
    assert mechanism.demand_paged is False
    assert mechanism.offloaded_bytes() == 0
    assert mechanism.working_set_reservation_bytes() == 0
    with mechanism.reserve_working_set():
        pass
    assert mechanism.total_bytes() == 0

    x = torch.randn(2, 64)
    outputs = [linear(x) for linear in module]
    entry = 8 * 64 * 4
    assert mechanism.total_bytes() == mechanism.loaded_bytes() == 2 * entry

    # Partial eviction drops the newest entry first (cache contract).
    assert mechanism.partially_unload(1) == entry
    assert cache.get("second") is None
    assert cache.get("first") is not None

    # The reference's negative-allowance shrink; None and positive
    # allowances never load because caches refill on forward.
    assert mechanism.partially_load(entry) == 0
    assert mechanism.partially_load(None) == 0
    assert mechanism.partially_load(-1) == -entry
    assert mechanism.total_bytes() == 0

    # Later forwards re-decode, re-offer, and stay bit-identical.
    refilled = [linear(x) for linear in module]
    for before, after in zip(outputs, refilled, strict=True):
        assert torch.equal(before, after)
    assert mechanism.total_bytes() == 2 * entry
    mechanism.unload()
    assert cache.used_bytes == 0
    assert mechanism.total_bytes() == 0

    # Device moves swap buffer tensors; load_device tracks the move.
    module.to("meta")
    assert mechanism.load_device == torch.device("meta")


def test_decoded_cache_residency_requires_a_cache_backed_linear() -> None:
    with pytest.raises(ValueError, match="no cache-backed encoded GGUF linears"):
        GgufDecodedCacheResidency(torch.nn.Linear(4, 4))
    uncached = GgufEncodedLinear(64, 8, bias=False, compute_dtype=torch.float32)
    with pytest.raises(ValueError, match="no cache-backed encoded GGUF linears"):
        GgufDecodedCacheResidency(uncached)


def test_decoded_cache_residency_spans_distinct_caches() -> None:
    torch.manual_seed(9)
    caches = (GgufDecodedCache(1 << 20), GgufDecodedCache(1 << 20))
    module = torch.nn.ModuleList(
        _cached_linear(cache, "layer", torch.randn(8, 64)) for cache in caches
    )
    x = torch.randn(1, 64)
    for linear in module:
        linear(x)

    entry = 8 * 64 * 4
    mechanism = GgufDecodedCacheResidency(module)
    assert mechanism.total_bytes() == 2 * entry

    # Eviction crosses cache boundaries until the request is met.
    assert mechanism.partially_unload(entry + 1) == 2 * entry
    assert caches[0].used_bytes == 0
    assert caches[1].used_bytes == 0

    for linear in module:
        linear(x)
    assert mechanism.total_bytes() == 2 * entry
    mechanism.unload()
    assert mechanism.total_bytes() == 0


def test_residency_manager_free_reclaims_decoded_cache_bytes() -> None:
    torch.manual_seed(11)
    cache = GgufDecodedCache(1 << 20)
    module = torch.nn.ModuleList(
        _cached_linear(cache, key, torch.randn(8, 64)) for key in ("first", "second")
    )
    x = torch.randn(2, 64)
    expected = [linear(x) for linear in module]
    entry = 8 * 64 * 4
    mechanism = GgufDecodedCacheResidency(module)
    capacity = 3 * entry
    cpu = torch.device("cpu")

    def free_memory(device: torch.device) -> DeviceMemory:
        assert device == cpu
        return DeviceMemory(free_total=capacity - mechanism.total_bytes(), free_torch=0)

    cleared: list[torch.device] = []
    manager = ResidencyManager(
        policy=MemoryPolicy(inference_reserve=0, physical_headroom=0, load_inflation=1.0),
        free_memory=free_memory,
        empty_cache=cleared.append,
    )
    manager.load([mechanism])
    assert manager.registered() == (mechanism,)
    assert cache.used_bytes == 2 * entry

    # A shortfall below the cached bytes evicts entries and keeps the
    # mechanism registered.
    manager.free(2 * entry, cpu)
    assert cache.used_bytes == entry
    assert manager.registered() == (mechanism,)
    assert cleared == []

    # Pressure beyond the cached bytes detaches the mechanism.
    manager.free(capacity, cpu)
    assert cache.used_bytes == 0
    assert manager.registered() == ()
    assert cleared == [cpu]

    refilled = [linear(x) for linear in module]
    for before, after in zip(expected, refilled, strict=True):
        assert torch.equal(before, after)


@pytest.mark.parametrize("ggml_type", (Q4_K, Q8_0), ids=lambda ggml_type: ggml_type.name)
def test_enrolled_forward_is_bit_identical_across_offload_states(ggml_type: GGMLType) -> None:
    in_features = ggml_type.block_elements
    out_features = 4
    blocks = reference_quant_blocks(ggml_type, out_features, seed=ggml_type.code + 200)
    decoded = GGUF_BLOCK_DECODERS[ggml_type.name](blocks, (out_features, in_features))
    torch.manual_seed(7)
    bias = torch.randn(out_features)
    x = torch.randn(3, in_features)
    module = GgufEncodedLinear(
        in_features, out_features, ggml_type=ggml_type.name, bias=True, compute_dtype=torch.float32
    )
    module.load_state_dict({"weight_blocks": blocks, "bias": bias})
    expected = torch.nn.functional.linear(x, decoded, bias)
    assert module.residency_prefetch() is None

    cpu = torch.device("cpu")
    mechanism = enroll_component(module, load_device=cpu, offload_device=cpu)
    assert mechanism.loaded_unit_names() == frozenset()
    assert torch.equal(module(x), expected)

    route = module.residency_prefetch()
    assert route is not None
    prefetch_mechanism, requests = route
    assert prefetch_mechanism is mechanism
    assert requests == (("weight_blocks", None), ("bias", torch.float32))

    mechanism.partially_load(None)
    assert module.residency_prefetch() is None
    assert torch.equal(module(x), expected)

    mechanism.unload()
    assert torch.equal(module(x), expected)


def test_prefetch_queue_streams_offloaded_gguf_weights_ahead_of_the_forward() -> None:
    """The block-loop prefetch queue stages an offloaded encoded
    linear's blocks and bias before its forward runs: the output stays
    bit-identical and the receipt attributes both copies to prefetch
    with no lease-started transfer."""
    in_features = Q8_0.block_elements
    out_features = 4
    blocks = reference_quant_blocks(Q8_0, out_features, seed=Q8_0.code + 300)
    decoded = GGUF_BLOCK_DECODERS[Q8_0.name](blocks, (out_features, in_features))
    torch.manual_seed(7)
    bias = torch.randn(out_features)
    x = torch.randn(3, in_features)
    module = GgufEncodedLinear(
        in_features, out_features, ggml_type=Q8_0.name, bias=True, compute_dtype=torch.float32
    )
    module.load_state_dict({"weight_blocks": blocks, "bias": bias})
    expected = torch.nn.functional.linear(x, decoded, bias)

    cpu = torch.device("cpu")
    mechanism = enroll_component(module, load_device=cpu, offload_device=cpu)
    assert mechanism.loaded_unit_names() == frozenset()

    with collect_partial_residency_timing() as timing:
        queue = make_prefetch_queue([module])
        assert queue is not None
        prefetch_queue_pop(queue, module)
        actual = module(x)
        prefetch_queue_pop(queue, None)
    report = timing.report()

    assert torch.equal(actual, expected)
    assert report.prefetched_transfers == 2
    assert report.prefetch_bytes == blocks.nbytes + bias.nbytes
    assert report.transfer_bytes == report.prefetch_bytes
    assert report.leased_transfers == 0
    assert report.leased_forwards == 1

    # Uncollected, the queued forward is the same bits.
    queue = make_prefetch_queue([module])
    assert queue is not None
    prefetch_queue_pop(queue, module)
    assert torch.equal(module(x), expected)
    prefetch_queue_pop(queue, None)


def test_offloaded_forward_bypasses_the_decoded_cache() -> None:
    torch.manual_seed(11)
    blocks = encode_q8_0(torch.randn(8, 64))
    decoded = decode_q8_0_blocks(blocks, (8, 64))
    cache = GgufDecodedCache(1 << 20)
    module = GgufEncodedLinear(
        64, 8, bias=False, compute_dtype=torch.float32, decoded_cache=cache, cache_key="layer"
    )
    module.load_state_dict({"weight_blocks": blocks})
    x = torch.randn(3, 64)
    expected = torch.nn.functional.linear(x, decoded, None)

    cpu = torch.device("cpu")
    mechanism = enroll_component(module, load_device=cpu, offload_device=cpu)
    poison = cache.offer("layer", torch.zeros(8, 64))
    assert torch.equal(module(x), expected)
    assert cache.get("layer") is poison

    cache.clear()
    assert torch.equal(module(x), expected)
    assert cache.used_bytes == 0

    mechanism.partially_load(None)
    assert torch.equal(module(x), expected)
    assert cache.get("layer") is not None


def test_enrollment_refuses_patches_on_encoded_blocks() -> None:
    torch.manual_seed(13)
    blocks = encode_q8_0(torch.randn(8, 64))
    module = GgufEncodedLinear(64, 8, bias=True, compute_dtype=torch.float32)
    module.load_state_dict({"weight_blocks": blocks, "bias": torch.randn(8)})
    assert ModuleStateStore(module).protected_quantization_keys() == {"weight_blocks"}

    cpu = torch.device("cpu")
    patch_set = PatchSet(
        {"weight_blocks": (PatchEntry(DiffPatch(torch.zeros_like(blocks, dtype=torch.float32))),)}
    )
    with pytest.raises(PatchApplyError, match="packed quantization-state patch/requantization"):
        enroll_component(module, load_device=cpu, offload_device=cpu, patch_set=patch_set)


def test_bias_patches_apply_through_the_leased_forward() -> None:
    torch.manual_seed(17)
    blocks = encode_q8_0(torch.randn(8, 64))
    decoded = decode_q8_0_blocks(blocks, (8, 64))
    bias = torch.randn(8)
    module = GgufEncodedLinear(64, 8, bias=True, compute_dtype=torch.float32)
    module.load_state_dict({"weight_blocks": blocks, "bias": bias})
    x = torch.randn(3, 64)

    cpu = torch.device("cpu")
    enroll_component(
        module,
        load_device=cpu,
        offload_device=cpu,
        patch_set=PatchSet({"bias": (PatchEntry(DiffPatch(torch.ones(8))),)}),
    )
    patched = torch.nn.functional.linear(x, decoded, bias + torch.ones(8))
    assert torch.equal(module(x), patched)


def test_timing_receipts_capture_leased_forward_phases() -> None:
    torch.manual_seed(19)
    blocks = encode_q8_0(torch.randn(8, 64))
    decoded = decode_q8_0_blocks(blocks, (8, 64))
    bias = torch.randn(8)
    module = GgufEncodedLinear(64, 8, bias=True, compute_dtype=torch.float32)
    module.load_state_dict({"weight_blocks": blocks, "bias": bias})
    x = torch.randn(3, 64)
    expected = torch.nn.functional.linear(x, decoded, bias)

    cpu = torch.device("cpu")
    enroll_component(module, load_device=cpu, offload_device=cpu)
    with collect_partial_residency_timing() as timing:
        offloaded = module(x)
    report = timing.report()

    assert torch.equal(offloaded, expected)
    assert report.leased_forwards == 1
    assert report.leased_transfers == 2
    assert report.transfer_bytes == blocks.nbytes + bias.nbytes
    assert report.transfer_ms > 0.0
    assert report.dequant_ms > 0.0
    assert report.compute_ms > 0.0
    # Synchronous CPU transfers are fully exposed: the stall phase
    # brackets the same copy, inside the transfer bracket.
    assert 0.0 < report.exposed_stall_ms <= report.transfer_ms


def test_timing_receipts_skip_loaded_forwards() -> None:
    torch.manual_seed(21)
    blocks = encode_q8_0(torch.randn(8, 64))
    decoded = decode_q8_0_blocks(blocks, (8, 64))
    module = GgufEncodedLinear(64, 8, bias=False, compute_dtype=torch.float32)
    module.load_state_dict({"weight_blocks": blocks})
    x = torch.randn(3, 64)
    expected = torch.nn.functional.linear(x, decoded, None)

    cpu = torch.device("cpu")
    mechanism = enroll_component(module, load_device=cpu, offload_device=cpu)
    mechanism.partially_load(None)
    with collect_partial_residency_timing() as timing:
        assert torch.equal(module(x), expected)
    report = timing.report()

    assert report.leased_forwards == 0
    assert report.leased_transfers == 0
    assert report.transfer_bytes == 0
    assert report.transfer_ms == 0.0
    assert report.exposed_stall_ms == 0.0
    assert report.dequant_ms == 0.0
    assert report.compute_ms == 0.0

    # Without an active collector the leased forward is untouched.
    mechanism.unload()
    assert torch.equal(module(x), expected)


def test_timing_receipts_record_into_the_innermost_collector() -> None:
    torch.manual_seed(23)
    blocks = encode_q8_0(torch.randn(8, 64))
    module = GgufEncodedLinear(64, 8, bias=False, compute_dtype=torch.float32)
    module.load_state_dict({"weight_blocks": blocks})
    x = torch.randn(3, 64)

    cpu = torch.device("cpu")
    enroll_component(module, load_device=cpu, offload_device=cpu)
    with collect_partial_residency_timing() as outer:
        with collect_partial_residency_timing() as inner:
            module(x)
        module(x)

    assert inner.report().leased_forwards == 1
    assert outer.report().leased_forwards == 1


def test_uncollected_leased_forward_reads_thread_local_state_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a collector, receipts cost one thread-local read when
    the lease opens; the forward itself takes the untimed path."""
    torch.manual_seed(27)
    blocks = encode_q8_0(torch.randn(8, 64))
    bias = torch.randn(8)
    module = GgufEncodedLinear(64, 8, bias=True, compute_dtype=torch.float32)
    module.load_state_dict({"weight_blocks": blocks, "bias": bias})
    x = torch.randn(3, 64)

    cpu = torch.device("cpu")
    enroll_component(module, load_device=cpu, offload_device=cpu)
    reads = 0
    real = residency_mod.active_partial_residency_timing

    def counting() -> PartialResidencyTiming | None:
        nonlocal reads
        reads += 1
        return real()

    monkeypatch.setattr(residency_mod, "active_partial_residency_timing", counting)
    module(x)
    assert reads == 1


def test_explicit_fused_matmul_unavailable_warns_and_matches_decode(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A stubbed probe isolates the eligibility checks from the host's
    # actual CUDA/triton capability.
    def stub_op(
        input: torch.Tensor,
        blocks: torch.Tensor,
        bias: torch.Tensor | None,
        out_features: int,
    ) -> torch.Tensor:
        raise AssertionError("bind-time checks must not execute the op")

    stub_ops = {name: stub_op for name in ("Q4_0", "Q4_K", "Q5_K", "Q6_K", "Q8_0")}
    monkeypatch.setattr(gguf_linear_mod, "_fused_linear_ops", stub_ops)

    # Every encoded layout has a fused op today, so the unsupported-
    # layout refusal is exercised against a reduced op-name mapping.
    with monkeypatch.context() as reduced:
        reduced.setattr(
            gguf_linear_mod,
            "_FUSED_OP_NAMES",
            {
                name: value
                for name, value in gguf_linear_mod._FUSED_OP_NAMES.items()  # pyright: ignore[reportPrivateUsage]
                if name != "Q6_K"
            },
        )
        q6k = GgufEncodedLinear(256, 4, ggml_type="Q6_K", bias=False, compute_dtype=torch.float16)
        q6k.bind_fused_matmul(True)
        assert q6k.fused_matmul is False

    misaligned = GgufEncodedLinear(16, 64, bias=False, compute_dtype=torch.float16)
    misaligned.bind_fused_matmul(True)
    assert misaligned.fused_matmul is False

    kquant_misaligned = GgufEncodedLinear(
        64, 8, ggml_type="Q4_K", bias=False, compute_dtype=torch.float16
    )
    kquant_misaligned.bind_fused_matmul(True)
    assert kquant_misaligned.fused_matmul is False

    fp32 = GgufEncodedLinear(64, 8, bias=False, compute_dtype=torch.float32)
    fp32.bind_fused_matmul(True)
    assert fp32.fused_matmul is False

    # Every fused-supported layout binds through its own op.
    for ggml_type, in_features in (("Q4_0", 64), ("Q4_K", 256), ("Q5_K", 256), ("Q6_K", 256)):
        module = GgufEncodedLinear(
            in_features, 8, ggml_type=ggml_type, bias=False, compute_dtype=torch.float16
        )
        module.bind_fused_matmul(True)
        assert module.fused_matmul is True
        assert module._fused_op is stub_op  # pyright: ignore[reportPrivateUsage]

    module = GgufEncodedLinear(64, 8, bias=False, compute_dtype=torch.float16)
    blocks = encode_q8_0(torch.randn(8, 64))
    module.load_state_dict({"weight_blocks": blocks})
    input = torch.randn(3, 64).to(torch.float16)
    expected = module(input)
    monkeypatch.setattr(gguf_linear_mod, "_fused_linear_ops", {"Q8_0": None})
    module.bind_fused_matmul(True)
    assert module.fused_matmul is False
    torch.testing.assert_close(module(input), expected, rtol=0, atol=0)
    # Disabling never consults the probe and always succeeds.
    module.bind_fused_matmul(False)
    assert module.fused_matmul is False


def test_fused_bind_over_cpu_blocks_falls_back_to_the_decode_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def stub_op(
        input: torch.Tensor,
        blocks: torch.Tensor,
        bias: torch.Tensor | None,
        out_features: int,
    ) -> torch.Tensor:
        nonlocal calls
        calls += 1
        raise AssertionError("the fused op must not run over CPU blocks")

    monkeypatch.setattr(gguf_linear_mod, "_fused_linear_ops", {"Q8_0": stub_op})

    torch.manual_seed(31)
    blocks = encode_q8_0(torch.randn(8, 64))
    bias = torch.randn(8).to(torch.float16)
    x = torch.randn(3, 64).to(torch.float16)

    plain = GgufEncodedLinear(64, 8, bias=True, compute_dtype=torch.float16)
    plain.load_state_dict({"weight_blocks": blocks, "bias": bias})
    expected = plain(x)

    fused = GgufEncodedLinear(64, 8, bias=True, compute_dtype=torch.float16)
    fused.load_state_dict({"weight_blocks": blocks, "bias": bias})
    fused.bind_fused_matmul(True)
    assert fused.fused_matmul is True
    assert torch.equal(fused(x), expected)

    # The offloaded lease path applies the same CUDA-blocks guard.
    cpu = torch.device("cpu")
    enroll_component(fused, load_device=cpu, offload_device=cpu)
    assert torch.equal(fused(x), expected)
    assert calls == 0


def test_bind_default_fused_matmul_binds_when_layer_and_host_support_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def stub_op(
        input: torch.Tensor,
        blocks: torch.Tensor,
        bias: torch.Tensor | None,
        out_features: int,
    ) -> torch.Tensor:
        raise AssertionError("bind-time checks must not execute the op")

    stub_ops = {name: stub_op for name in ("Q4_0", "Q4_K", "Q5_K", "Q6_K", "Q8_0")}
    monkeypatch.setattr(gguf_linear_mod, "_fused_linear_ops", stub_ops)

    for ggml_type, in_features in (
        ("Q4_0", 64),
        ("Q4_K", 256),
        ("Q5_K", 256),
        ("Q6_K", 256),
        ("Q8_0", 64),
    ):
        module = GgufEncodedLinear(
            in_features, 8, ggml_type=ggml_type, bias=False, compute_dtype=torch.float16
        )
        assert module.bind_default_fused_matmul() is True
        assert module.fused_matmul is True
        assert module._fused_op is stub_op  # pyright: ignore[reportPrivateUsage]


def test_bind_default_fused_matmul_keeps_the_decode_route_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every manual-bind refusal fails open on the default bind: the
    layer keeps the decode route and reports False."""

    def stub_op(
        input: torch.Tensor,
        blocks: torch.Tensor,
        bias: torch.Tensor | None,
        out_features: int,
    ) -> torch.Tensor:
        raise AssertionError("a refused default bind must not execute the op")

    monkeypatch.setattr(gguf_linear_mod, "_fused_linear_ops", {"Q8_0": stub_op})

    # An unsupported layout (exercised against a reduced op-name
    # mapping because every encoded layout has a fused op today).
    with monkeypatch.context() as reduced:
        reduced.setattr(
            gguf_linear_mod,
            "_FUSED_OP_NAMES",
            {
                name: value
                for name, value in gguf_linear_mod._FUSED_OP_NAMES.items()  # pyright: ignore[reportPrivateUsage]
                if name != "Q6_K"
            },
        )
        q6k = GgufEncodedLinear(256, 4, ggml_type="Q6_K", bias=False, compute_dtype=torch.float16)
        assert q6k.bind_default_fused_matmul() is False
        assert q6k.fused_matmul is False

    misaligned = GgufEncodedLinear(16, 64, bias=False, compute_dtype=torch.float16)
    assert misaligned.bind_default_fused_matmul() is False
    assert misaligned.fused_matmul is False

    kquant_misaligned = GgufEncodedLinear(
        64, 8, ggml_type="Q4_K", bias=False, compute_dtype=torch.float16
    )
    assert kquant_misaligned.bind_default_fused_matmul() is False
    assert kquant_misaligned.fused_matmul is False

    fp32 = GgufEncodedLinear(64, 8, bias=False, compute_dtype=torch.float32)
    assert fp32.bind_default_fused_matmul() is False
    assert fp32.fused_matmul is False

    probeless = GgufEncodedLinear(64, 8, bias=False, compute_dtype=torch.float16)
    monkeypatch.setattr(gguf_linear_mod, "_fused_linear_ops", {"Q8_0": None})
    assert probeless.bind_default_fused_matmul() is False
    assert probeless.fused_matmul is False


def test_fused_route_admits_flattened_token_counts_up_to_the_layout_threshold() -> None:
    # Every fused-supported layout carries a measured threshold.
    assert set(FUSED_MATMUL_MAX_TOKENS) == set(
        gguf_linear_mod._FUSED_OP_NAMES  # pyright: ignore[reportPrivateUsage]
    )

    module = GgufEncodedLinear(64, 8, bias=False, compute_dtype=torch.float16)
    threshold = FUSED_MATMUL_MAX_TOKENS["Q8_0"]
    admits = module._fused_route_admits  # pyright: ignore[reportPrivateUsage]
    assert admits(torch.empty(threshold, 64, dtype=torch.float16))
    assert not admits(torch.empty(threshold + 1, 64, dtype=torch.float16))
    # The threshold counts flattened tokens across leading batch dims.
    assert admits(torch.empty(2, threshold // 2, 64, dtype=torch.float16))
    assert not admits(torch.empty(2, threshold // 2 + 1, 64, dtype=torch.float16))
