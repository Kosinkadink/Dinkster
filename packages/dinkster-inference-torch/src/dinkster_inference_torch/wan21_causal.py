"""Wan 2.1 CausalAR transformer with invocation-scoped attention caches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from dinkster_inference import WAN21_CAUSAL_AR_1_3B, Wan21Config

from .attention import AttentionKernel, attention_kernel_context, select_attention
from .flux import apply_rope
from .operations import INITLESS, Operations
from .ops import cast_weight
from .wan21_model import (
    Wan21Model,
    WanAttentionBlock,
    WanCrossAttention,
    WanSelfAttention,
    sinusoidal_embedding_1d,
)

_DEFAULT_ATTENTION = select_attention("flux").kernel


def _repeat_time_rows(value: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    repeats = 1
    if value.shape[1] > 1:
        repeats = tokens.shape[1] // value.shape[1]
    if repeats == 1:
        return value
    if repeats * value.shape[1] == tokens.shape[1]:
        return torch.repeat_interleave(value, repeats, dim=1)
    return torch.repeat_interleave(value, repeats + 1, dim=1)[:, : tokens.shape[1]]


@dataclass(slots=True)
class _SelfAttentionCache:
    key: torch.Tensor
    value: torch.Tensor
    end: int = 0

    def rewind(self, rows: int) -> None:
        if type(rows) is not int or rows < 0 or rows > self.end:
            raise ValueError("Wan CausalAR cache rewind exceeds the committed rows")
        self.end -= rows


@dataclass(slots=True)
class _CrossAttentionCache:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class _Wan21CausalCaches:
    """Preallocated self-attention and projected text state for one sample call."""

    self_attention: tuple[_SelfAttentionCache, ...]
    cross_attention: tuple[_CrossAttentionCache, ...]

    def __post_init__(self) -> None:
        if not self.self_attention or len(self.self_attention) != len(self.cross_attention):
            raise ValueError("Wan CausalAR requires one self/cross cache pair per layer")

    def rewind(self, rows: int) -> None:
        for cache in self.self_attention:
            cache.rewind(rows)


class _CausalWanSelfAttention(WanSelfAttention):
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        cache: _SelfAttentionCache | None = None,
    ) -> torch.Tensor:
        batch, sequence, heads, width = (
            x.shape[0],
            x.shape[1],
            self.num_heads,
            self.head_dim,
        )
        q = self.norm_q(self.q(x)).view(batch, sequence, heads, width)
        k = self.norm_k(self.k(x)).view(batch, sequence, heads, width)
        q, k = apply_rope(q, k, freqs)
        v = self.v(x).view(batch, sequence, heads, width)
        if cache is None:
            attended = self._attention_kernel(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
            )
            return self.o(attended.transpose(1, 2).reshape(batch, sequence, heads * width))
        if (
            cache.key.shape != cache.value.shape
            or cache.key.ndim != 4
            or cache.key.shape[0] != batch
            or cache.key.shape[2:] != (heads, width)
            or cache.key.device != k.device
            or cache.key.dtype != k.dtype
        ):
            raise ValueError("Wan CausalAR self-attention cache does not match the model input")
        new_end = cache.end + sequence
        if new_end > cache.key.shape[1]:
            raise ValueError("Wan CausalAR self-attention cache capacity was exceeded")
        cache.key[:, cache.end : new_end].copy_(k)
        cache.value[:, cache.end : new_end].copy_(v)
        cache.end = new_end
        attended = self._attention_kernel(
            q.transpose(1, 2),
            cache.key[:, :new_end].transpose(1, 2),
            cache.value[:, :new_end].transpose(1, 2),
        )
        return self.o(attended.transpose(1, 2).reshape(batch, sequence, heads * width))


class _CausalWanAttentionBlock(WanAttentionBlock):
    _causal_attention_kernel: AttentionKernel

    def __init__(
        self,
        config: Wan21Config,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__(
            config,
            operations=operations,
            attention_kernel=attention_kernel,
            _self_attention_type=_CausalWanSelfAttention,
        )
        object.__setattr__(self, "_causal_attention_kernel", attention_kernel)

    def _cross(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        cache: _CrossAttentionCache,
    ) -> torch.Tensor:
        attention = cast("WanCrossAttention", self.cross_attn)
        batch, query_rows = x.shape[:2]
        q = (
            attention.norm_q(attention.q(x))
            .view(batch, query_rows, attention.num_heads, attention.head_dim)
            .transpose(1, 2)
        )
        if cache.key is None or cache.value is None:
            key_rows = context.shape[1]
            cache.key = (
                attention.norm_k(attention.k(context))
                .view(batch, key_rows, attention.num_heads, attention.head_dim)
                .transpose(1, 2)
            )
            cache.value = (
                attention.v(context)
                .view(batch, key_rows, attention.num_heads, attention.head_dim)
                .transpose(1, 2)
            )
        elif (
            cache.key.shape != cache.value.shape
            or cache.key.ndim != 4
            or cache.key.shape[0] != batch
            or cache.key.shape[1] != attention.num_heads
            or cache.key.shape[3] != attention.head_dim
            or cache.key.device != q.device
            or cache.key.dtype != q.dtype
        ):
            raise ValueError("Wan CausalAR cross-attention cache does not match the model input")
        key = cache.key
        value = cache.value
        assert key is not None and value is not None
        attended = self._causal_attention_kernel(q, key, value)
        return attention.o(attended.transpose(1, 2).reshape(batch, query_rows, -1))

    def _forward_owned_causal(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
        self_cache: _SelfAttentionCache,
        cross_cache: _CrossAttentionCache,
        modulation_state: torch.Tensor,
    ) -> torch.Tensor:
        modulation = (modulation_state.unsqueeze(0) + time).unbind(2)
        x = x.contiguous()
        normalized = torch.addcmul(
            _repeat_time_rows(modulation[0], x),
            self.norm1(x),
            1 + _repeat_time_rows(modulation[1], x),
        )
        attention = cast("_CausalWanSelfAttention", self.self_attn)
        x = torch.addcmul(
            x,
            attention(normalized, freqs, self_cache),
            _repeat_time_rows(modulation[2], x),
        )
        x = x + self._cross(self.norm3(x), context, cross_cache)
        normalized = torch.addcmul(
            _repeat_time_rows(modulation[3], x),
            self.norm2(x),
            1 + _repeat_time_rows(modulation[4], x),
        )
        return torch.addcmul(
            x,
            self.ffn(normalized),
            _repeat_time_rows(modulation[5], x),
        )

    def forward_causal(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
        self_cache: _SelfAttentionCache,
        cross_cache: _CrossAttentionCache,
    ) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is None:
            modulation = cast_weight(self.modulation, device=x.device, dtype=x.dtype)
            return self._forward_owned_causal(
                x, time, freqs, context, self_cache, cross_cache, modulation
            )
        with binding.lease() as lease:
            return self._forward_owned_causal(
                x,
                time,
                freqs,
                context,
                self_cache,
                cross_cache,
                lease.get("modulation", dtype=x.dtype),
            )


class Wan21CausalModel(Wan21Model):
    """Weight-compatible Wan 2.1 backbone for blockwise autoregressive sampling."""

    def __init__(
        self,
        config: Wan21Config = WAN21_CAUSAL_AR_1_3B,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
    ) -> None:
        if config is not WAN21_CAUSAL_AR_1_3B:
            raise ValueError("Wan CausalAR model requires the exact 1.3B profile")
        super().__init__(
            WAN21_CAUSAL_AR_1_3B,
            operations=operations,
            attention_kernel=attention_kernel,
            _block_type=_CausalWanAttentionBlock,
        )

    def create_caches(
        self,
        *,
        batch_size: int,
        max_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> _Wan21CausalCaches:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Wan CausalAR cache batch size must be positive")
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("Wan CausalAR cache capacity must be positive")
        head_dim = self.config.hidden_size // self.config.num_heads
        shape = (batch_size, max_tokens, self.config.num_heads, head_dim)
        self_attention = tuple(
            _SelfAttentionCache(
                torch.empty(shape, device=device, dtype=dtype),
                torch.empty(shape, device=device, dtype=dtype),
            )
            for _ in self.blocks
        )
        return _Wan21CausalCaches(
            self_attention,
            tuple(_CrossAttentionCache() for _ in self.blocks),
        )

    def _causal_rope(
        self,
        shape: tuple[int, int, int],
        time_start: int,
        x: torch.Tensor,
    ) -> torch.Tensor:
        time, height, width = shape
        ids = torch.zeros((time, height, width, 3), device=x.device, dtype=x.dtype)
        ids[..., 0] += torch.arange(
            time_start,
            time_start + time,
            device=x.device,
            dtype=x.dtype,
        )[:, None, None]
        ids[..., 1] += torch.arange(height, device=x.device, dtype=x.dtype)[None, :, None]
        ids[..., 2] += torch.arange(width, device=x.device, dtype=x.dtype)[None, None, :]
        return self.rope_embedder(ids.reshape(1, -1, 3)).movedim(1, 2)

    def forward_block(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        *,
        time_start: int,
        caches: _Wan21CausalCaches,
    ) -> torch.Tensor:
        self._validate(
            x,
            timesteps,
            context,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        if type(time_start) is not int or time_start < 0:
            raise ValueError("Wan CausalAR time_start must be a non-negative integer")
        if len(caches.self_attention) != len(self.blocks):
            raise ValueError("Wan CausalAR cache count does not match the model layers")
        original_shape = x.shape[2:]
        pad_t = (-x.shape[2]) % self.config.patch_size[0]
        pad_h = (-x.shape[3]) % self.config.patch_size[1]
        pad_w = (-x.shape[4]) % self.config.patch_size[2]
        if pad_t or pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode="circular")
        x = self.patch_embedding(x.float()).to(x.dtype)
        grid = (x.shape[2], x.shape[3], x.shape[4])
        freqs = self._causal_rope(grid, time_start, x)
        x = x.flatten(2).transpose(1, 2)
        time = self.time_embedding(
            sinusoidal_embedding_1d(self.config.time_freq_dim, timesteps.flatten()).to(x.dtype)
        ).reshape(timesteps.shape[0], -1, self.config.hidden_size)
        projected_time = self.time_projection(time).unflatten(2, (6, self.config.hidden_size))
        context = self.text_embedding(context)
        with attention_kernel_context(self._attention_kernel, x.numel(), device=x.device):
            for index, block in enumerate(self.blocks):
                causal = cast("_CausalWanAttentionBlock", block)
                x = causal.forward_causal(
                    x,
                    projected_time,
                    freqs,
                    context,
                    caches.self_attention[index],
                    caches.cross_attention[index],
                )
        x = self.head(x, time)
        batch = x.shape[0]
        patch_t, patch_h, patch_w = self.config.patch_size
        x = x.view(batch, *grid, patch_t, patch_h, patch_w, self.config.out_channels)
        x = torch.einsum("bthwpqrc->bctphqwr", x)
        x = x.reshape(
            batch,
            self.config.out_channels,
            grid[0] * patch_t,
            grid[1] * patch_h,
            grid[2] * patch_w,
        )
        return x[:, :, : original_shape[0], : original_shape[1], : original_shape[2]]

    def forward(self, *_args: object, **_kwargs: object) -> torch.Tensor:
        raise ValueError("Wan CausalAR requires the autoregressive video sampler")


__all__ = ["Wan21CausalModel"]
