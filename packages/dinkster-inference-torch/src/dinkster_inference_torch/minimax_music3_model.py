"""Native MiniMax Music 3 diffusion transformer."""

from __future__ import annotations

import math
from collections.abc import Generator
from contextlib import contextmanager
from typing import cast

import torch
import torch.nn.functional as F
from dinkster_inference import (
    CONDITION_HOP_FRAMES,
    MAX_CONDITION_FRAMES,
    MINIMAX_MUSIC3_CONFIG,
    MiniMaxMusic3Config,
    minimax_music3_latent_length,
)

from .attention import AttentionKernel, attention_kernel_context, select_attention
from .operations import INITLESS, Operations, ResidencyRouted
from .ops import cast_weight

_DEFAULT_ATTENTION = select_attention("flux").kernel


@contextmanager
def _direct_state(
    module: ResidencyRouted, name: str, like: torch.Tensor
) -> Generator[torch.Tensor]:
    stored = cast(torch.Tensor, getattr(module, name))
    binding = module._offloaded_residency()  # pyright: ignore[reportPrivateUsage]
    if binding is None:
        yield cast_weight(stored, device=like.device, dtype=like.dtype)
        return
    with binding.lease() as lease:
        yield lease.get(name, dtype=like.dtype)


class MiniMaxMusic3FourierFeatures(ResidencyRouted, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(128, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        with _direct_state(self, "weight", value) as weight:
            features = 2.0 * math.pi * value @ weight.transpose(0, 1)
        return torch.cat((features.cos(), features.sin()), dim=-1)


class MiniMaxMusic3LayerNorm(ResidencyRouted, torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.empty(width))
        self.register_buffer("beta", torch.empty(width))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is None:
            gamma = cast_weight(self.gamma, device=hidden.device, dtype=hidden.dtype)
            beta = cast_weight(
                cast(torch.Tensor, self.beta), device=hidden.device, dtype=hidden.dtype
            )
            return F.layer_norm(hidden, (hidden.shape[-1],), gamma, beta)
        with binding.lease() as lease:
            return F.layer_norm(
                hidden,
                (hidden.shape[-1],),
                lease.get("gamma", dtype=hidden.dtype),
                lease.get("beta", dtype=hidden.dtype),
            )


class MiniMaxMusic3RotaryEmbedding(ResidencyRouted, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("inv_freq", torch.empty(16))

    def table(self, length: int, like: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(length, device=like.device, dtype=torch.float32)
        with _direct_state(self, "inv_freq", positions) as inverse:
            frequencies = torch.outer(positions, inverse).to(like.dtype)
        cosine, sine = frequencies.cos(), frequencies.sin()
        return torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(
            1, 1, length, frequencies.shape[-1], 2, 2
        )


def _apply_rope(hidden: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    original_dtype = hidden.dtype
    pairs = hidden.reshape(*hidden.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2)
    pairs = pairs.to(table.dtype)
    rotated = table[..., 0] * pairs[..., 0] + table[..., 1] * pairs[..., 1]
    return rotated.movedim(-1, -2).flatten(-2).to(original_dtype)


class MiniMaxMusic3Attention(torch.nn.Module):
    _attention_kernel: AttentionKernel

    def __init__(
        self,
        config: MiniMaxMusic3Config,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.heads = config.attention_heads
        self.head_dim = config.attention_head_dim
        self.to_qkv = operations.linear(config.hidden_width, 3 * config.hidden_width, bias=False)
        self.to_out = operations.linear(config.hidden_width, config.hidden_width, bias=False)
        object.__setattr__(self, "_attention_kernel", attention_kernel)

    def forward(self, hidden: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape
        query, key, value = self.to_qkv(hidden).chunk(3, dim=-1)
        shape = (batch, length, self.heads, self.head_dim)
        query = query.reshape(shape).transpose(1, 2)
        key = key.reshape(shape).transpose(1, 2)
        value = value.reshape(shape).transpose(1, 2)
        rotary_width = table.shape[-3] * 2
        if torch.is_grad_enabled():
            rotated_query = _apply_rope(query[..., :rotary_width], table)
            rotated_key = _apply_rope(key[..., :rotary_width], table)
            query = torch.cat((rotated_query, query[..., rotary_width:]), dim=-1)
            key = torch.cat((rotated_key, key[..., rotary_width:]), dim=-1)
        else:
            import dinkster_kitchen  # pyright: ignore[reportMissingTypeStubs]

            dinkster_kitchen.apply_rope_split_half1_(query[..., :rotary_width], table)
            dinkster_kitchen.apply_rope_split_half1_(key[..., :rotary_width], table)
        output = self._attention_kernel(query, key, value)
        return self.to_out(output.transpose(1, 2).reshape(batch, length, width))


class MiniMaxMusic3FeedForward(torch.nn.Module):
    def __init__(self, config: MiniMaxMusic3Config, *, operations: Operations) -> None:
        super().__init__()
        self.ff = torch.nn.Sequential(
            MiniMaxMusic3Glu(config, operations=operations),
            torch.nn.Identity(),
            operations.linear(config.feed_forward_width, config.hidden_width),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.ff(hidden)


class MiniMaxMusic3Glu(torch.nn.Module):
    def __init__(self, config: MiniMaxMusic3Config, *, operations: Operations) -> None:
        super().__init__()
        self.proj = operations.linear(config.hidden_width, 2 * config.feed_forward_width)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        value, gate = self.proj(hidden).chunk(2, dim=-1)
        activated = F.silu(gate)
        if torch.is_grad_enabled():
            return value * activated
        return activated.mul_(value)


class MiniMaxMusic3Block(torch.nn.Module):
    def __init__(
        self,
        config: MiniMaxMusic3Config,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.pre_norm = MiniMaxMusic3LayerNorm(config.hidden_width)
        self.self_attn = MiniMaxMusic3Attention(
            config, operations=operations, attention_kernel=attention_kernel
        )
        self.ff_norm = MiniMaxMusic3LayerNorm(config.hidden_width)
        self.ff = MiniMaxMusic3FeedForward(config, operations=operations)

    def forward(self, hidden: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        attention = self.self_attn(self.pre_norm(hidden), table)
        hidden = hidden + attention if torch.is_grad_enabled() else hidden.add_(attention)
        feed_forward = self.ff(self.ff_norm(hidden))
        return hidden + feed_forward if torch.is_grad_enabled() else hidden.add_(feed_forward)


class MiniMaxMusic3ContinuousTransformer(torch.nn.Module):
    def __init__(
        self,
        config: MiniMaxMusic3Config,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.project_in = operations.linear(2304, config.hidden_width, bias=False)
        self.project_out = operations.linear(
            config.hidden_width, config.latent_channels, bias=False
        )
        self.rotary_pos_emb = MiniMaxMusic3RotaryEmbedding()
        self.layers = torch.nn.ModuleList(
            MiniMaxMusic3Block(config, operations=operations, attention_kernel=attention_kernel)
            for _ in range(config.blocks)
        )

    def forward(
        self,
        hidden: torch.Tensor,
        timestep_embedding: torch.Tensor,
        rotary_table: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.project_in(hidden)
        hidden = torch.cat((timestep_embedding.unsqueeze(1), hidden), dim=1)
        table = rotary_table[:, :, : hidden.shape[1]]
        for layer in self.layers:
            hidden = layer(hidden, table)
        return self.project_out(hidden[:, 1:])


class MiniMaxMusic3DiffusionTransformer(torch.nn.Module):
    def __init__(
        self,
        config: MiniMaxMusic3Config,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.transformer = MiniMaxMusic3ContinuousTransformer(
            config, operations=operations, attention_kernel=attention_kernel
        )
        self.timestep_features = MiniMaxMusic3FourierFeatures()
        self.to_timestep_embed = torch.nn.Sequential(
            operations.linear(256, config.hidden_width),
            torch.nn.SiLU(),
            operations.linear(config.hidden_width, config.hidden_width),
        )
        self.preprocess_conv = operations.conv1d(2304, 2304, 1, bias=False)
        self.postprocess_conv = operations.conv1d(
            config.latent_channels, config.latent_channels, 1, bias=False
        )

    def prepare_timestep(
        self,
        timestep: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        features = self.timestep_features(timestep[:, None]).to(dtype=dtype)
        return self.to_timestep_embed(features)

    def forward(
        self,
        latent: torch.Tensor,
        condition: torch.Tensor,
        timestep_embedding: torch.Tensor,
        rotary_table: torch.Tensor,
    ) -> torch.Tensor:
        hidden = torch.cat((latent, torch.zeros_like(latent), condition), dim=1)
        processed = self.preprocess_conv(hidden)
        hidden = processed + hidden if torch.is_grad_enabled() else processed.add_(hidden)
        output = self.transformer(
            hidden.transpose(1, 2), timestep_embedding, rotary_table
        ).transpose(1, 2)
        processed = self.postprocess_conv(output)
        return processed + output if torch.is_grad_enabled() else processed.add_(output)


class MiniMaxMusic3DiT(ResidencyRouted, torch.nn.Module):
    def __init__(
        self,
        config: MiniMaxMusic3Config = MINIMAX_MUSIC3_CONFIG,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
    ) -> None:
        super().__init__()
        self.config = config
        self._attention_kernel = attention_kernel
        self.latent_conditioners = torch.nn.Sequential(
            operations.conv1d(config.context_width, config.hidden_width, 3, padding=1)
        )
        self.diffusion_transformer = MiniMaxMusic3DiffusionTransformer(
            config, operations=operations, attention_kernel=attention_kernel
        )
        self.cond_layer_logits = torch.nn.Parameter(torch.empty(config.context_layers))
        self.cond_layer_scale = torch.nn.Parameter(torch.empty(1))

    def aligned_condition(self, hidden: torch.Tensor) -> torch.Tensor:
        frames = hidden.shape[1]
        hidden = hidden.transpose(1, 2).reshape(
            hidden.shape[0], self.config.context_layers, self.config.context_width, frames
        )
        binding = self._offloaded_residency()
        if binding is None:
            logits = cast_weight(self.cond_layer_logits, device=hidden.device, dtype=hidden.dtype)
            scale = cast_weight(self.cond_layer_scale, device=hidden.device, dtype=hidden.dtype)
            hidden = torch.einsum("blht,l->bht", hidden, torch.softmax(logits, dim=0))
            hidden = scale * hidden
        else:
            with binding.lease() as lease:
                weights = torch.softmax(lease.get("cond_layer_logits", dtype=hidden.dtype), dim=0)
                hidden = torch.einsum("blht,l->bht", hidden, weights)
                hidden = lease.get("cond_layer_scale", dtype=hidden.dtype) * hidden
        condition = self.latent_conditioners(hidden)
        return F.interpolate(
            condition,
            size=minimax_music3_latent_length(frames),
            mode="nearest",
        )

    def forward(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        conditioning_scale: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_prepared(
            latent,
            timestep,
            self.prepare_condition(context, conditioning_scale),
            self.prepare_rotary(latent),
        )

    def prepare_condition(
        self,
        context: torch.Tensor,
        conditioning_scale: torch.Tensor,
    ) -> torch.Tensor:
        return self.aligned_condition(context) * conditioning_scale[:, :1, :1]

    def prepare_rotary(self, latent: torch.Tensor) -> torch.Tensor:
        length = min(latent.shape[-1], minimax_music3_latent_length(MAX_CONDITION_FRAMES))
        return self.diffusion_transformer.transformer.rotary_pos_emb.table(length + 1, latent)

    def forward_prepared(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
        rotary_table: torch.Tensor,
    ) -> torch.Tensor:
        if condition.shape[-1] < latent.shape[-1]:
            condition = F.pad(condition, (0, latent.shape[-1] - condition.shape[-1]))
        else:
            condition = condition[..., : latent.shape[-1]]
        timestep_embedding = self.diffusion_transformer.prepare_timestep(timestep, latent.dtype)
        window = minimax_music3_latent_length(MAX_CONDITION_FRAMES)
        with attention_kernel_context(
            self._attention_kernel,
            latent.shape[0] * self.config.hidden_width * min(latent.shape[-1], window),
            device=latent.device,
        ):
            if latent.shape[-1] <= window:
                return -self.diffusion_transformer(
                    latent, condition, timestep_embedding, rotary_table
                )

            output = torch.zeros_like(latent)
            count = torch.zeros((1, 1, latent.shape[-1]), device=latent.device, dtype=latent.dtype)
            hop = minimax_music3_latent_length(CONDITION_HOP_FRAMES)
            start = 0
            while start < latent.shape[-1]:
                end = min(start + window, latent.shape[-1])
                output[..., start:end] -= self.diffusion_transformer(
                    latent[..., start:end],
                    condition[..., start:end],
                    timestep_embedding,
                    rotary_table,
                )
                count[..., start:end] += 1
                if end == latent.shape[-1]:
                    break
                start += hop
            return output / count


__all__ = [
    "MiniMaxMusic3DiT",
]
