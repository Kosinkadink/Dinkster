"""Focused Wan 2.1 CausalAR model and cache contracts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from dinkster_inference import (
    WAN21_CAUSAL_AR_1_3B,
    Parameterization,
    SamplerInfo,
    Wan21Config,
    wan21_layout,
)
from dinkster_inference_torch.attention import select_attention
from dinkster_inference_torch.flux import EmbedND
from dinkster_inference_torch.operations import CastOperations
from dinkster_inference_torch.wan21_causal import (
    Wan21CausalModel,
    _CausalWanAttentionBlock,  # pyright: ignore[reportPrivateUsage]
    _CrossAttentionCache,  # pyright: ignore[reportPrivateUsage]
    _SelfAttentionCache,  # pyright: ignore[reportPrivateUsage]
)
from dinkster_inference_torch.wan21_model import WanAttentionBlock
from dinkster_inference_torch.wan21_runtime import (
    Wan21RuntimeError,
    _Wan21CausalDenoiser,  # pyright: ignore[reportPrivateUsage]
)
from unet_fill import fill_state_dict, hashed_input


def _tiny_config() -> Wan21Config:
    return Wan21Config(
        hidden_size=8,
        ffn_hidden_size=16,
        num_heads=1,
        num_layers=1,
        text_dim=4,
        time_freq_dim=4,
    )


def _freqs(rows: int) -> torch.Tensor:
    embedder = EmbedND(dim=8, theta=10000, axes_dim=(4, 2, 2))
    ids = torch.zeros((1, rows, 3), dtype=torch.float32)
    ids[..., 1] = torch.arange(rows)
    return embedder(ids).movedim(1, 2)


def test_causal_model_is_weight_compatible_with_the_admitted_header() -> None:
    with torch.device("meta"):
        model = Wan21CausalModel()

    actual = {key: tuple(value.shape) for key, value in model.state_dict().items()}

    assert actual == wan21_layout(WAN21_CAUSAL_AR_1_3B)
    with pytest.raises(ValueError, match="exact 1.3B profile"):
        Wan21CausalModel(cast("Any", _tiny_config()))


def test_causal_attention_reuses_allocations_and_projected_text_after_rewind() -> None:
    config = _tiny_config()
    block = _CausalWanAttentionBlock(
        config,
        operations=CastOperations(torch.float32),
        attention_kernel=select_attention("flux").kernel,
    )
    state = [(key, tuple(value.shape)) for key, value in sorted(block.state_dict().items())]
    block.load_state_dict(fill_state_dict(state), strict=True)
    x = hashed_input("causal:block:x", (1, 2, 8))
    time = torch.zeros((1, 1, 6, 8))
    context = hashed_input("causal:block:context", (1, 3, 8))
    self_cache = _SelfAttentionCache(
        torch.empty((1, 4, 1, 8)),
        torch.empty((1, 4, 1, 8)),
    )
    cross_cache = _CrossAttentionCache()
    key_storage = self_cache.key.untyped_storage().data_ptr()
    value_storage = self_cache.value.untyped_storage().data_ptr()

    with torch.no_grad():
        first = block.forward_causal(x, time, _freqs(2), context, self_cache, cross_cache)
    assert self_cache.end == 2
    assert cross_cache.key is not None and cross_cache.value is not None
    cross_key = cross_cache.key
    cross_value = cross_cache.value

    self_cache.rewind(2)
    with torch.no_grad():
        replay = block.forward_causal(
            x,
            time,
            _freqs(2),
            context + 100.0,
            self_cache,
            cross_cache,
        )

    assert torch.equal(replay, first)
    assert self_cache.end == 2
    assert self_cache.key.untyped_storage().data_ptr() == key_storage
    assert self_cache.value.untyped_storage().data_ptr() == value_storage
    assert cross_cache.key is cross_key
    assert cross_cache.value is cross_value

    with torch.no_grad():
        block.forward_causal(x, time, _freqs(2), context, self_cache, cross_cache)
    assert self_cache.end == 4
    with pytest.raises(ValueError, match="capacity"):
        with torch.no_grad():
            block.forward_causal(x[:, :1], time, _freqs(1), context, self_cache, cross_cache)


def test_first_causal_block_matches_ordinary_wan_without_history() -> None:
    config = _tiny_config()
    operations = CastOperations(torch.float32)
    attention = select_attention("flux").kernel
    ordinary = WanAttentionBlock(
        config,
        operations=operations,
        attention_kernel=attention,
    )
    causal = _CausalWanAttentionBlock(
        config,
        operations=operations,
        attention_kernel=attention,
    )
    state = [(key, tuple(value.shape)) for key, value in sorted(ordinary.state_dict().items())]
    weights = fill_state_dict(state)
    ordinary.load_state_dict(weights, strict=True)
    causal.load_state_dict(weights, strict=True)
    x = hashed_input("causal:parity:x", (1, 3, 8))
    time = hashed_input("causal:parity:time", (1, 1, 6, 8))
    context = hashed_input("causal:parity:context", (1, 2, 8))
    freqs = _freqs(3)
    self_cache = _SelfAttentionCache(
        torch.empty((1, 3, 1, 8)),
        torch.empty((1, 3, 1, 8)),
    )

    with torch.no_grad():
        expected = ordinary(x, time, freqs, context, None)
        inherited = WanAttentionBlock.forward(causal, x, time, freqs, context, None)
        actual = causal.forward_causal(
            x,
            time,
            freqs,
            context,
            self_cache,
            _CrossAttentionCache(),
        )

    assert torch.equal(inherited, expected)
    assert torch.equal(actual, expected)
    assert self_cache.end == 3


def test_causal_cache_lifecycle_and_rope_offsets_are_invocation_owned() -> None:
    with torch.device("meta"):
        model = Wan21CausalModel()
    caches = model.create_caches(
        batch_size=2,
        max_tokens=7,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert len(caches.self_attention) == len(caches.cross_attention) == 30
    assert all(cache.key.shape == (2, 7, 12, 128) for cache in caches.self_attention)
    assert all(cache.end == 0 for cache in caches.self_attention)
    caches.rewind(0)
    with pytest.raises(ValueError, match="rewind"):
        caches.rewind(1)

    x = torch.zeros((1,), dtype=torch.float32)
    first = model._causal_rope((2, 1, 1), 0, x)  # pyright: ignore[reportPrivateUsage]
    shifted = model._causal_rope((2, 1, 1), 3, x)  # pyright: ignore[reportPrivateUsage]
    assert first.shape == shifted.shape == (1, 2, 1, 64, 2, 2)
    assert not torch.equal(first, shifted)


def test_causal_model_refuses_the_ordinary_forward_path() -> None:
    model = Wan21CausalModel.__new__(Wan21CausalModel)
    torch.nn.Module.__init__(model)
    with pytest.raises(ValueError, match="autoregressive video sampler"):
        cast("Any", model)(torch.zeros(1))


@pytest.mark.parametrize("initial", (None, torch.zeros((1, 16, 1, 2, 2))))
@pytest.mark.parametrize("sigmas", ((1.0, 0.5, 0.25), (1.0, 0.5, 0.5, 0.0)))
def test_causal_denoiser_refuses_schedules_that_cannot_commit_clean_blocks(
    initial: torch.Tensor | None,
    sigmas: tuple[float, ...],
) -> None:
    model = SimpleNamespace(config=SimpleNamespace(out_channels=16))
    denoiser = _Wan21CausalDenoiser(
        cast("Any", model),
        torch.ones((1, 2, 4)),
        initial_latent=initial,
        compute_dtype=torch.float32,
    )

    with pytest.raises(Wan21RuntimeError, match="strictly decreasing sigmas ending at 0"):
        denoiser.sample_autoregressive(
            torch.zeros((1, 16, 2, 2, 2)),
            sigmas,
            SamplerInfo(Parameterization.FLOW, seed=0),
            num_frame_per_block=1,
        )


def test_causal_denoiser_commits_initial_and_generated_blocks_with_cache_rewinds() -> None:
    class Caches:
        def __init__(self) -> None:
            self.end = 0
            self.rewinds: list[int] = []

        def rewind(self, rows: int) -> None:
            assert rows <= self.end
            self.end -= rows
            self.rewinds.append(rows)

    class Model:
        config = SimpleNamespace(out_channels=16)

        def __init__(self) -> None:
            self.cache_request: tuple[int, int, torch.device, torch.dtype] | None = None
            self.calls: list[tuple[int, tuple[float, ...], int]] = []
            self.caches = Caches()

        def create_caches(
            self,
            *,
            batch_size: int,
            max_tokens: int,
            device: torch.device,
            dtype: torch.dtype,
        ) -> Caches:
            self.cache_request = (batch_size, max_tokens, device, dtype)
            return self.caches

        def forward_block(
            self,
            x: torch.Tensor,
            timesteps: torch.Tensor,
            _context: torch.Tensor,
            *,
            time_start: int,
            caches: Caches,
        ) -> torch.Tensor:
            rows = x.shape[2] * ((x.shape[3] + 1) // 2) * ((x.shape[4] + 1) // 2)
            caches.end += rows
            self.calls.append((time_start, tuple(timesteps.tolist()), rows))
            return torch.zeros_like(x)

    model = Model()
    initial = torch.full((1, 16, 1, 3, 3), -3.0)
    denoiser = _Wan21CausalDenoiser(
        cast("Any", model),
        torch.ones((1, 2, 4)),
        initial_latent=initial,
        compute_dtype=torch.float32,
    )
    x = torch.arange(1 * 16 * 4 * 3 * 3, dtype=torch.float32).reshape(1, 16, 4, 3, 3)
    events: list[tuple[int, int, float]] = []

    output = denoiser.sample_autoregressive(
        x,
        (1.0, 0.5, 0.0),
        SamplerInfo(Parameterization.FLOW, seed=23),
        num_frame_per_block=2,
        on_step=lambda event: events.append((event.step, event.total, event.sigma)),
    )

    first_noise = torch.randn((1, 16, 2, 3, 3), generator=torch.Generator().manual_seed(23))
    second_noise = torch.randn((1, 16, 1, 3, 3), generator=torch.Generator().manual_seed(1023))
    assert torch.equal(output[:, :, :1], initial)
    torch.testing.assert_close(output[:, :, 1:3], x[:, :, 1:3] * 0.5 + first_noise * 0.5)
    torch.testing.assert_close(output[:, :, 3:], x[:, :, 3:] * 0.5 + second_noise * 0.5)
    assert model.cache_request == (1, 16, torch.device("cpu"), torch.float32)
    assert model.caches.rewinds == [8, 8, 4, 4]
    assert model.caches.end == 16
    assert model.calls == [
        (0, (0.0,), 4),
        (1, (1000.0,), 8),
        (1, (500.0,), 8),
        (1, (0.0,), 8),
        (3, (1000.0,), 4),
        (3, (500.0,), 4),
        (3, (0.0,), 4),
    ]
    assert events == [(0, 2, 1.0), (0, 2, 0.5), (1, 2, 1.0), (1, 2, 0.5)]
