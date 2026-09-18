"""Parity, cache, and contract tests for Wan Animate2."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from attention_spy import CallableModuleKernel, assert_kernel_is_not_model_state
from dinkster_inference import WAN21_ANIMATE2_14B, Wan21Config, wan21_layout
from dinkster_inference_torch import pinned_host, wan21_animate2
from dinkster_inference_torch.attention import select_attention
from dinkster_inference_torch.memory import MemoryPolicy
from dinkster_inference_torch.module_residency import enroll_component
from dinkster_inference_torch.operations import ResidencyRouted
from dinkster_inference_torch.wan21_animate2 import (
    PoseBranchCache,
    WanAnimate2Block,
    WanAnimate2Model,
)
from unet_fill import fill_state_dict, hashed_input

GOLDENS = json.loads(
    (Path(__file__).parent / "goldens" / "wan21_animate2_goldens.json").read_text()
)

# Convrot plus per-channel scaling reconstructs ordinary values within these
# bounds; sparse exact corrections keep production outliers under the same limits.
POSE_CACHE_RECONSTRUCTION_ATOL = {"int8": 0.02, "int4": 0.35}

# Fresh pinned ComfyUI outputs were bit-identical to Dinkster on the failing Linux
# host, isolating issue #564 to cross-host stored-golden drift. The three cases
# measured 7.16e-7, 5.97e-7, and 7.75e-7 max absolute drift locally; a second
# host measured about 9e-7 overall. These per-case limits leave at least 1.39x
# headroom over the widest observed drift.
ANIMATE2_GOLDEN_ATOL = {
    "no_pose": 1.25e-6,
    "pose_default": 1.25e-6,
    "pose_distinct": 1.25e-6,
}


def _decode(payload: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(payload["data"], dtype=getattr(torch, payload["dtype"])).reshape(
        payload["shape"]
    )


def _model(*, kernel: Any = None) -> WanAnimate2Model:
    kwargs = {} if kernel is None else {"attention_kernel": kernel}
    model = WanAnimate2Model(Wan21Config(**GOLDENS["config"]), **kwargs)
    model.load_state_dict(fill_state_dict(GOLDENS["state_dict"]), strict=True)
    return model


def _inputs() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return (
        hashed_input("animate2_reduced:x", GOLDENS["input_shape"]),
        torch.tensor(GOLDENS["timesteps"], dtype=torch.float32),
        hashed_input("animate2_reduced:context", GOLDENS["context_shape"]),
        hashed_input("animate2_reduced:vision", GOLDENS["vision_shape"]),
        hashed_input("animate2_reduced:pose", GOLDENS["pose_shape"]),
        hashed_input("animate2_reduced:pose-context", GOLDENS["pose_context_shape"]),
        hashed_input("animate2_reduced:pose-vision", GOLDENS["pose_vision_shape"]),
    )


def test_golden_provenance_is_exact_and_regeneration_is_documented() -> None:
    assert GOLDENS["_meta"] == {
        "attention": "attention_pytorch",
        "commit": "b78cec879b9460d5cb25228a83a942fb78d2cd24",
        "python": "3.12.11",
        "reference": "ComfyUI WanAnimate2Model",
        "rope": "pure torch (model_management.in_training=True)",
        "source": "git archive of exact commit object",
        "torch": "2.13.0+cpu",
    }


def test_official_geometry_uses_animate2_blocks_with_exact_i2v_state() -> None:
    with torch.device("meta"):
        model = WanAnimate2Model(WAN21_ANIMATE2_14B)
    state = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    owners = [
        module
        for module in model.modules()
        if tuple(module.parameters(recurse=False)) or tuple(module.buffers(recurse=False))
    ]

    assert state == wan21_layout(WAN21_ANIMATE2_14B)
    assert len(state) == 1303
    assert all(isinstance(block, WanAnimate2Block) for block in model.blocks)
    assert all(isinstance(module, ResidencyRouted) for module in owners)


@pytest.mark.parametrize(
    ("name", "kwargs"),
    (
        ("no_pose", {}),
        ("pose_default", {"pose": True}),
        (
            "pose_distinct",
            {
                "pose": True,
                "distinct": True,
                "pose_strength": 0.625,
                "reference_strength": 0.75,
            },
        ),
    ),
)
def test_reduced_forward_matches_stored_reference(name: str, kwargs: dict[str, object]) -> None:
    model = _model()
    x, timesteps, context, vision, pose, pose_context, pose_vision = _inputs()
    actual = model(
        x,
        timesteps,
        context,
        vision,
        pose_latents=pose if kwargs.get("pose") else None,
        pose_context=pose_context if kwargs.get("distinct") else None,
        pose_vision=pose_vision if kwargs.get("distinct") else None,
        pose_strength=cast(float, kwargs.get("pose_strength", 1.0)),
        reference_strength=cast(float, kwargs.get("reference_strength", 1.0)),
    )

    torch.testing.assert_close(
        actual,
        _decode(GOLDENS["outputs"][name]),
        rtol=0.0,
        atol=ANIMATE2_GOLDEN_ATOL[name],
    )


def test_pose_attention_uses_shifted_branch_and_preceding_frame_tail() -> None:
    spy = CallableModuleKernel(select_attention("flux").kernel)
    model = _model(kernel=spy)
    x, timesteps, context, vision, pose, pose_context, pose_vision = _inputs()

    output = model(
        x,
        timesteps,
        context,
        vision,
        pose_latents=pose,
        pose_context=pose_context,
        pose_vision=pose_vision,
    )

    assert output.shape == (1, 16, 3, 5, 6)
    assert len(spy.calls) == 16
    self_attention = [call for call in spy.calls if call["k_shape"][-2] >= call["q_shape"][-2]]
    assert [call["q_shape"][-2] for call in self_attention] == [18, 9, 9, 9] * 2
    assert [call["k_shape"][-2] for call in self_attention] == [18, 27, 36, 36] * 2
    assert_kernel_is_not_model_state(model, spy)


def test_default_pose_cache_reuses_block_inputs_without_changing_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    x, timesteps, context, vision, pose, pose_context, pose_vision = _inputs()
    uncached = model(
        x,
        timesteps,
        context,
        vision,
        pose_latents=pose,
        pose_context=pose_context,
        pose_vision=pose_vision,
    )
    cache = PoseBranchCache()
    kwargs = {
        "pose_latents": pose,
        "pose_context": pose_context,
        "pose_vision": pose_vision,
        "pose_cache": cache,
    }

    first = model(x, timesteps, context, vision, **kwargs)
    cached = model(x, timesteps, context, vision, **kwargs)

    assert cache.memory_bytes() == 18544
    assert torch.equal(first, uncached)
    assert torch.equal(cached, first)

    real_take = cache.take

    def lose_restore_reserve(
        index: int,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
    ) -> torch.Tensor:
        if index == 1:
            raise torch.OutOfMemoryError("injected post-preflight pressure")
        return real_take(index, device, dtype, batch_size)

    monkeypatch.setattr(cache, "take", lose_restore_reserve)
    late_fallback = model(x, timesteps, context, vision, **kwargs)

    assert torch.equal(late_fallback, uncached)
    assert cache.memory_bytes() == 0


def test_pose_cache_fallback_recomputes_outside_the_exception_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The OOM traceback holds the failing frames' tensors while the handler
    runs, so the pose recompute must execute with no active exception."""
    model = _model()
    x, timesteps, context, vision, pose, pose_context, pose_vision = _inputs()
    uncached = model(
        x,
        timesteps,
        context,
        vision,
        pose_latents=pose,
        pose_context=pose_context,
        pose_vision=pose_vision,
    )
    cache = PoseBranchCache()
    kwargs = {
        "pose_latents": pose,
        "pose_context": pose_context,
        "pose_vision": pose_vision,
        "pose_cache": cache,
    }
    model(x, timesteps, context, vision, **kwargs)

    real_take = cache.take

    def failing_take(
        index: int,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
    ) -> torch.Tensor:
        if index == 1:
            raise torch.OutOfMemoryError("injected pose cache restore pressure")
        return real_take(index, device, dtype, batch_size)

    monkeypatch.setattr(cache, "take", failing_take)

    active_exceptions: list[BaseException | None] = []
    original_forward_pose = WanAnimate2Block.forward_pose

    def recording_forward_pose(
        self: WanAnimate2Block, *args: Any, **forward_kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        active_exceptions.append(sys.exc_info()[1])
        return original_forward_pose(self, *args, **forward_kwargs)

    monkeypatch.setattr(WanAnimate2Block, "forward_pose", recording_forward_pose)

    fallback = model(x, timesteps, context, vision, **kwargs)

    assert torch.equal(fallback, uncached)
    assert active_exceptions
    assert all(active is None for active in active_exceptions)


@pytest.mark.parametrize(
    ("dtype", "expected_bytes"),
    (("default", 8704), ("int8", 2592), ("int4", 1568)),
)
def test_pose_cache_storage_modes_roundtrip_and_repeat_batch(
    dtype: str, expected_bytes: int
) -> None:
    cache = PoseBranchCache(dtype=dtype)  # type: ignore[arg-type]
    pose = hashed_input(f"animate2-cache:{dtype}:pose", (1, 16, 2, 2, 2))
    block_input = hashed_input(f"animate2-cache:{dtype}:block", (1, 8, 256))
    cache.select(pose)
    cache.put(0, block_input)

    restored = cache.take(0, torch.device("cpu"), torch.float32, 2)

    assert cache.memory_bytes() == expected_bytes
    assert restored.shape == (2, 8, 256)
    assert torch.equal(restored[0], restored[1])
    if dtype == "default":
        assert torch.equal(restored[:1], block_input)
    else:
        assert (
            torch.max(torch.abs(restored[:1] - block_input)).item()
            <= POSE_CACHE_RECONSTRUCTION_ATOL[dtype]
        )


@pytest.mark.parametrize("dtype", ("int8", "int4"))
def test_pose_cache_production_shape_corrects_quantization_outliers(dtype: str) -> None:
    cache = PoseBranchCache(dtype=dtype)  # type: ignore[arg-type]
    pose = torch.zeros((1, 16, 2, 2, 2), dtype=torch.float16)
    block_input = hashed_input(f"animate2-cache-production:{dtype}", (1, 1008, 5120)).to(
        torch.float16
    )
    block_input[0, ::251, ::257] = 393.0
    cache.select(pose)
    cache.put(0, block_input)

    restored = cache.take(0, torch.device("cpu"), torch.float16, 2)
    slot = cache._slot  # pyright: ignore[reportPrivateUsage]
    assert slot is not None
    entry = slot.blocks[0]
    corrections = entry.corrections

    assert entry.params is not None
    assert corrections is not None and corrections.indices.dtype == torch.int32
    assert corrections.values.dtype == torch.float16
    assert corrections.indices.max().item() <= torch.iinfo(torch.int32).max
    assert cache.memory_bytes() == (
        pose.nbytes
        + entry.tensor.nbytes
        + entry.params.scale.nbytes
        + corrections.indices.nbytes
        + corrections.values.nbytes
    )
    assert cache.memory_bytes() < pose.nbytes + block_input.nbytes
    assert torch.equal(restored[0], restored[1])
    assert (
        torch.max(torch.abs(restored[:1].float() - block_input.float())).item()
        <= (POSE_CACHE_RECONSTRUCTION_ATOL[dtype])
    )
    flat_restored = restored[0].reshape(-1)
    flat_source = block_input[0].reshape(-1)
    assert torch.equal(
        flat_restored.index_select(0, corrections.indices.long()),
        flat_source.index_select(0, corrections.indices.long()),
    )


def test_pose_cache_selects_distinct_lru_slots_by_pose_value() -> None:
    cache = PoseBranchCache()
    first_pose = torch.zeros((1, 16, 2, 2, 2))
    second_pose = torch.ones_like(first_pose)
    first_block = torch.ones((1, 2, 8))
    second_block = torch.full_like(first_block, 2.0)
    cache.select(first_pose)
    cache.put(0, first_block)
    cache.select(second_pose)
    cache.put(0, second_block)
    cache.select(first_pose)

    assert cache.filled(1)
    assert torch.equal(cache.take(0, torch.device("cpu"), torch.float32, 1), first_block)
    assert cache.memory_bytes() == 1152


def test_pose_cache_keys_pose_text_and_vision_inputs() -> None:
    cache = PoseBranchCache()
    pose = torch.zeros((1, 16, 2, 2, 2))
    first_text = torch.zeros((1, 2, 8))
    second_text = torch.ones_like(first_text)
    vision = torch.zeros((1, 2, 16))
    block = torch.ones((1, 2, 8))
    cache.select(pose, first_text, vision)
    cache.put(0, block)

    cache.select(pose, second_text, vision)
    assert not cache.filled(1)
    cache.put(0, block * 2)
    cache.select(pose, first_text, vision)

    assert cache.filled(1)
    assert torch.equal(cache.take(0, torch.device("cpu"), torch.float32, 1), block)


def test_pose_cache_is_bypassed_when_batch_pose_inputs_differ() -> None:
    model = _model()
    x, timesteps, context, vision, pose, pose_context, pose_vision = _inputs()
    cache = PoseBranchCache()
    batch = 2
    x = x.repeat(batch, 1, 1, 1, 1)
    timesteps = timesteps.repeat(batch)
    context = context.repeat(batch, 1, 1)
    vision = vision.repeat(batch, 1, 1)
    pose = pose.repeat(batch, 1, 1, 1, 1)
    pose_context = pose_context.repeat(batch, 1, 1)
    pose_context[1] += 1.0
    pose_vision = pose_vision.repeat(batch, 1, 1)
    kwargs = {
        "pose_latents": pose,
        "pose_context": pose_context,
        "pose_vision": pose_vision,
        "pose_cache": cache,
    }

    first = model(x, timesteps, context, vision, **kwargs)
    second = model(x, timesteps, context, vision, **kwargs)

    assert torch.equal(second, first)
    assert cache.memory_bytes() == 0


def test_pose_cache_limit_counts_keys_quantization_parameters_and_blocks() -> None:
    pose = torch.zeros((1, 16, 2, 2, 2))
    block = torch.zeros((1, 8, 256))
    exact_bytes = 512 + 2048 + 32

    exact = PoseBranchCache(dtype="int8", memory_limit_bytes=exact_bytes)
    exact.select(pose)
    exact.put(0, block)
    assert exact.filled(1)
    assert exact.memory_bytes() == exact_bytes

    limited = PoseBranchCache(dtype="int8", memory_limit_bytes=exact_bytes - 1)
    limited.select(pose)
    limited.put(0, block)
    assert not limited.filled(1)
    assert limited.memory_bytes() == 512

    below_key = PoseBranchCache(memory_limit_bytes=511)
    below_key.select(pose)
    below_key.put(0, block)
    assert not below_key.filled(1)
    assert below_key.memory_bytes() == 0

    replaces_smaller_key = PoseBranchCache(memory_limit_bytes=100)
    replaces_smaller_key.select(torch.zeros((1, 1)))
    replaces_smaller_key.select(torch.zeros((1, 25)))
    assert replaces_smaller_key.memory_bytes() == 100


@pytest.mark.parametrize(
    ("store", "reserve"),
    (
        ("cpu", pinned_host.AVAILABLE_RAM_FLOOR),
        ("cuda", MemoryPolicy().minimum_inference_memory()),
    ),
)
def test_pose_cache_refuses_new_key_below_store_reserve(
    monkeypatch: pytest.MonkeyPatch, store: str, reserve: int
) -> None:
    pose = torch.zeros((1, 16, 2, 2, 2))
    key_bytes = pose.numel() * pose.element_size()

    def free_memory(device: torch.device) -> SimpleNamespace:
        assert device == torch.device(store)
        return SimpleNamespace(free_total=reserve + key_bytes - 1)

    monkeypatch.setattr(wan21_animate2, "get_free_memory", free_memory)
    cache = PoseBranchCache(store)
    cache.select(pose)

    assert cache.memory_bytes() == 0


def test_pose_cache_retains_key_but_skips_block_below_host_reserve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pose = torch.zeros((1, 16, 2, 2, 2))
    block = torch.zeros((1, 8, 256))
    key_bytes = pose.numel() * pose.element_size()
    block_bytes = block.numel() * block.element_size()
    free_totals = iter(
        (
            pinned_host.AVAILABLE_RAM_FLOOR + key_bytes,
            pinned_host.AVAILABLE_RAM_FLOOR + block_bytes - 1,
        )
    )

    def free_memory(_device: torch.device) -> SimpleNamespace:
        return SimpleNamespace(free_total=next(free_totals))

    monkeypatch.setattr(wan21_animate2, "get_free_memory", free_memory)
    cache = PoseBranchCache()
    cache.select(pose)
    cache.put(0, block)

    assert cache.memory_bytes() == key_bytes
    assert not cache.filled(1)


def test_pose_cache_staging_respects_explicit_memory_limit() -> None:
    pose = torch.zeros((1, 16, 2, 2, 2))
    block = torch.zeros((1, 8, 256))
    retained_bytes = pose.nbytes + block.nbytes
    cache = PoseBranchCache(memory_limit_bytes=retained_bytes)
    cache.select(pose)
    cache.put(0, block)

    assert cache.filled(1)
    assert not cache.prepare_restore(1, torch.device("cuda"), torch.float32, 1)
    assert cache.memory_bytes() == retained_bytes


@pytest.mark.parametrize("constrained_device", ("cpu", "cuda"))
def test_pose_cache_staging_respects_host_and_accelerator_reserves(
    monkeypatch: pytest.MonkeyPatch,
    constrained_device: str,
) -> None:
    pose = torch.zeros((1, 16, 2, 2, 2))
    block = torch.zeros((1, 8, 256))
    cache = PoseBranchCache()
    cache.select(pose)
    cache.put(0, block)
    retained_bytes = cache.memory_bytes()

    def free_memory(device: torch.device) -> SimpleNamespace:
        if device.type == constrained_device:
            reserve = (
                pinned_host.AVAILABLE_RAM_FLOOR
                if device.type == "cpu"
                else MemoryPolicy().minimum_inference_memory()
            )
            return SimpleNamespace(free_total=reserve)
        return SimpleNamespace(free_total=1 << 60)

    monkeypatch.setattr(wan21_animate2, "get_free_memory", free_memory)

    assert not cache.prepare_restore(1, torch.device("cuda"), torch.float32, 1)
    assert cache.memory_bytes() == retained_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("constraint", ("limit", "host", "accelerator"))
def test_pose_cache_cuda_constraint_recomputes_without_changing_output(
    monkeypatch: pytest.MonkeyPatch,
    constraint: str,
) -> None:
    device = torch.device("cuda")
    model = _model().to(device)
    x, timesteps, context, vision, pose, pose_context, pose_vision = (
        value.to(device) for value in _inputs()
    )
    cache = PoseBranchCache()
    kwargs = {
        "pose_latents": pose,
        "pose_context": pose_context,
        "pose_vision": pose_vision,
        "pose_cache": cache,
    }
    first = model(x, timesteps, context, vision, **kwargs)
    assert cache.filled(len(model.blocks))

    if constraint == "limit":
        cache.memory_limit_bytes = cache.memory_bytes()
    else:
        constrained_device = "cpu" if constraint == "host" else "cuda"

        def free_memory(target: torch.device) -> SimpleNamespace:
            if target.type == constrained_device:
                reserve = (
                    pinned_host.AVAILABLE_RAM_FLOOR
                    if target.type == "cpu"
                    else MemoryPolicy().minimum_inference_memory()
                )
                return SimpleNamespace(free_total=reserve)
            return SimpleNamespace(free_total=1 << 60)

        monkeypatch.setattr(wan21_animate2, "get_free_memory", free_memory)

    fallback = model(x, timesteps, context, vision, **kwargs)

    assert torch.equal(fallback, first)
    assert not cache._staging  # pyright: ignore[reportPrivateUsage]
    cache.free()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("store", ("cpu", "cuda"))
@pytest.mark.parametrize(
    "dtype",
    ("default", "int8", "int4"),
)
def test_pose_cache_cuda_prefetch_roundtrip(store: str, dtype: str) -> None:
    device = torch.device("cuda")
    cache = PoseBranchCache(store, dtype=dtype)  # type: ignore[arg-type]
    pose = hashed_input(f"animate2-cuda:{store}:{dtype}:pose", (1, 16, 2, 2, 2)).to(device)
    block = hashed_input(f"animate2-cuda:{store}:{dtype}:block", (1, 8, 256)).to(device)
    if dtype != "default":
        block[0, 0, 0] = 393.0
    cache.select(pose)
    cache.put(0, block)
    retained_bytes = cache.memory_bytes()

    restored = cache.take(0, device, torch.float32, 1)
    torch.cuda.synchronize(device)

    assert restored.is_cuda
    if dtype == "default":
        assert torch.equal(restored, block)
    else:
        assert (
            torch.max(torch.abs(restored - block)).item() <= POSE_CACHE_RECONSTRUCTION_ATOL[dtype]
        )
    if store == "cpu":
        assert cache.memory_bytes() > retained_bytes
        assert (
            sum(
                pair.pinned_bytes
                for pair in cache._staging.values()  # pyright: ignore[reportPrivateUsage]
            )
            > 0
        )
    else:
        assert cache.memory_bytes() == retained_bytes
        assert not cache._staging  # pyright: ignore[reportPrivateUsage]
    cache.free()
    assert not cache._staging  # pyright: ignore[reportPrivateUsage]
    assert not cache.pin_active


def test_offloaded_animate2_forward_matches_resident_with_pose() -> None:
    model = _model()
    mechanism = enroll_component(model, load_device="cpu", offload_device="cpu")
    mechanism.partially_load(None)
    x, timesteps, context, vision, pose, pose_context, pose_vision = _inputs()
    kwargs = {
        "pose_latents": pose,
        "pose_context": pose_context,
        "pose_vision": pose_vision,
    }

    resident = model(x, timesteps, context, vision, **kwargs)
    mechanism.partially_unload(mechanism.loaded_bytes())
    offloaded = model(x, timesteps, context, vision, **kwargs)

    assert torch.equal(offloaded, resident)


@pytest.mark.parametrize(
    ("replacement", "message"),
    (
        ({"pose_latents": torch.empty((1, 16, 1, 5, 6))}, "must have shape"),
        ({"pose_context": torch.empty((1, 0, 12))}, "at least one row"),
        ({"pose_vision": torch.empty((1, 2, 1279))}, "1280"),
        ({"pose_strength": float("nan")}, "finite non-negative"),
        ({"reference_strength": -1.0}, "finite non-negative"),
    ),
)
def test_animate2_refuses_invalid_pose_inputs_before_execution(
    replacement: dict[str, object], message: str
) -> None:
    model = _model()
    x, timesteps, context, vision, pose, _pose_context, _pose_vision = _inputs()
    kwargs: dict[str, object] = {"pose_latents": pose}
    kwargs.update(replacement)

    with pytest.raises((TypeError, ValueError), match=message):
        model(x, timesteps, context, vision, **kwargs)  # type: ignore[arg-type]
