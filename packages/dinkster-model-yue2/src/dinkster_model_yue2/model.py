"""YuE2 acoustic flow transformer."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as functional
from dinkster_inference import QwenTextConfig
from dinkster_inference_torch.attention import AttentionKernel, select_attention
from dinkster_inference_torch.model_prefetch import (
    close_prefetch_queue,
    make_prefetch_queue,
    prefetch_queue_pop,
)
from dinkster_inference_torch.operations import INITLESS, Operations
from dinkster_inference_torch.qwen_text import (
    QwenBlock,
    _apply_rope,  # pyright: ignore[reportPrivateUsage]
    _rope,  # pyright: ignore[reportPrivateUsage]
)
from dinkster_inference_torch.unet import timestep_embedding

from .declarations import LATENT_CHANNELS


def yue2_qwen_config() -> QwenTextConfig:
    return QwenTextConfig(
        architecture="yue2_qwen3_3b",
        vocab_size=184_704,
        hidden_size=2_048,
        intermediate_size=6_144,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        max_position_embeddings=24_576,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        qkv_bias=False,
        qk_norm=True,
        prompt_template="{}",
        min_tokens=1,
        pad_token_id=151_643,
        merged_qkv=True,
        merged_mlp=True,
    )


class TimestepEmbedder(torch.nn.Module):
    def __init__(self, hidden: int, *, operations: Operations) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            operations.linear(256, hidden, bias=True),
            torch.nn.SiLU(),
            operations.linear(hidden, hidden, bias=True),
        )

    def forward(self, timesteps: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return self.mlp(timestep_embedding(timesteps, 256).to(dtype))


class AudioPositionEmbedding(torch.nn.Module):
    def __init__(self, frames: int, hidden: int) -> None:
        super().__init__()
        self.register_buffer("pe", torch.empty(frames, hidden))

    def forward(self, length: int, value: torch.Tensor) -> torch.Tensor:
        position = self.get_buffer("pe")
        return position[:length].to(device=value.device, dtype=value.dtype)


class _AcousticBackbone(torch.nn.Module):
    def __init__(
        self,
        config: QwenTextConfig,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            QwenBlock(config, operations=operations, attention_kernel=attention_kernel)
            for _ in range(config.num_hidden_layers)
        )
        self.norm = operations.rms_norm(config.hidden_size, eps=config.rms_norm_eps)


def _block_with_prefix(
    block: QwenBlock,
    hidden: torch.Tensor,
    frequencies: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    prefix: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    attention = block.self_attn
    normalized = block.input_layernorm(hidden)
    assert attention.qkv_proj is not None
    query_width, key_width, value_width = attention.qkv_widths
    query, key, value = attention.qkv_proj(normalized).split(
        (query_width, key_width, value_width), dim=-1
    )
    batch, length, _ = hidden.shape
    query = query.view(batch, length, attention.num_heads, attention.head_dim).transpose(1, 2)
    key = key.view(batch, length, attention.num_kv_heads, attention.head_dim).transpose(1, 2)
    value = value.view(batch, length, attention.num_kv_heads, attention.head_dim).transpose(1, 2)
    if attention.q_norm is not None:
        query = attention.q_norm(query)
    if attention.k_norm is not None:
        key = attention.k_norm(key)
    query, key = _apply_rope(query, key, frequencies)
    prefix_key, prefix_value = prefix
    key = torch.cat((prefix_key, key), dim=2)
    value = torch.cat((prefix_value, value), dim=2)
    prefix_length = prefix_key.shape[2]
    mask = torch.full(
        (length, prefix_length + length),
        torch.finfo(hidden.dtype).min / 4,
        device=hidden.device,
        dtype=hidden.dtype,
    ).triu_(prefix_length + 1)
    output = attention._attention_kernel(  # pyright: ignore[reportPrivateUsage]
        query,
        key,
        value,
        mask=mask,
        causal=False,
        enable_gqa=attention.num_heads != attention.num_kv_heads,
    )
    hidden = hidden + attention.o_proj(output.transpose(1, 2).reshape(batch, length, -1))
    return hidden + block.mlp(block.post_attention_layernorm(hidden))


class YuE2AcousticModel(torch.nn.Module):
    def __init__(
        self,
        _config: object = None,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel | None = None,
    ) -> None:
        super().__init__()
        self.config = yue2_qwen_config()
        kernel = select_attention("qwen").kernel if attention_kernel is None else attention_kernel
        self.model = _AcousticBackbone(
            self.config,
            operations=operations,
            attention_kernel=kernel,
        )
        self.vae2llm = operations.linear(LATENT_CHANNELS, self.config.hidden_size, bias=True)
        self.llm2vae = operations.linear(self.config.hidden_size, LATENT_CHANNELS, bias=True)
        self.time_embedder = TimestepEmbedder(self.config.hidden_size, operations=operations)
        self.latent_pos_embed = AudioPositionEmbedding(
            self.config.max_position_embeddings, self.config.hidden_size
        )

    def forward(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        chunks: tuple[tuple[int, int, int, int], ...],
    ) -> torch.Tensor:
        batch, channels, frames = latent.shape
        if channels != LATENT_CHANNELS or not chunks or frames != chunks[-1][1]:
            raise ValueError("YuE2 latent duration must match its generated music conditioning")
        time = self.time_embedder(timestep.to(latent.dtype), latent.dtype)[:, None]
        output = torch.empty_like(latent)
        config = self.config
        for start, end, kv_start, kv_end in chunks:
            prefix_length = kv_end - kv_start
            length = end - start + 2
            state = functional.pad(latent[..., start:end].transpose(1, 2), (0, 0, 1, 1))
            state = self.vae2llm(state) + time + self.latent_pos_embed(length, latent)[None]
            frequencies = _rope(
                config.head_dim,
                length,
                config.rope_theta,
                device=latent.device,
                start=prefix_length,
            )
            prefix = context[:, kv_start:kv_end].reshape(
                batch,
                prefix_length,
                config.num_hidden_layers,
                2,
                config.num_key_value_heads,
                config.head_dim,
            )
            prefix = prefix.permute(2, 3, 0, 4, 1, 5)
            prefetch = make_prefetch_queue(self.model.layers)
            try:
                for index, layer in enumerate(self.model.layers):
                    prefetch_queue_pop(prefetch, layer)
                    state = _block_with_prefix(
                        cast("QwenBlock", layer),
                        state,
                        frequencies,
                        (prefix[index, 0], prefix[index, 1]),
                    )
                prefetch_queue_pop(prefetch, None)
            finally:
                close_prefetch_queue(prefetch)
            output[..., start:end] = self.llm2vae(self.model.norm(state))[:, 1:-1].transpose(1, 2)
        return output


__all__ = ["YuE2AcousticModel", "yue2_qwen_config"]
