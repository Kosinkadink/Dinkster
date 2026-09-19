"""Native Ideogram 4 single-stream diffusion transformer."""

from __future__ import annotations

import math
from typing import Protocol

import torch
import torch.nn.functional as F
from dinkster_inference import IDEOGRAM4_CONFIG

from .attention import AttentionKernel, select_attention
from .model_prefetch import close_prefetch_queue, make_prefetch_queue, prefetch_queue_pop
from .operations import INITLESS, Operations, materialized_rms_norm_weight

_DEFAULT_ATTENTION = select_attention("flux").kernel
_OUTPUT_IMAGE = 2
_LLM_TOKEN = 3
_IMAGE_POSITION_OFFSET = 65536


class _Config(Protocol):
    @property
    def hidden_size(self) -> int: ...
    @property
    def layers(self) -> int: ...
    @property
    def attention_heads(self) -> int: ...
    @property
    def attention_head_dim(self) -> int: ...
    @property
    def intermediate_size(self) -> int: ...
    @property
    def adaln_dim(self) -> int: ...
    @property
    def latent_channels(self) -> int: ...
    @property
    def ae_channels(self) -> int: ...
    @property
    def patch(self) -> tuple[int, int]: ...
    @property
    def text_width(self) -> int: ...
    @property
    def rope_theta(self) -> float: ...
    @property
    def rope_dims(self) -> tuple[int, int, int]: ...
    @property
    def norm_eps(self) -> float: ...


def _rope_matrix(
    position_ids: torch.Tensor,
    *,
    head_dim: int,
    theta: float,
    rope_dims: tuple[int, int, int],
) -> torch.Tensor:
    positions = position_ids[0].transpose(0, 1).float()
    numerator = torch.arange(0, head_dim, 2, device=position_ids.device).float()
    inverse = 1.0 / (theta ** (numerator / head_dim))
    frequencies = (inverse[None, :, None] @ positions[:, None, :]).transpose(1, 2)
    interleaved = frequencies[0].clone()
    for axis, offset in ((1, 1), (2, 2)):
        end = rope_dims[axis] * 3
        interleaved[..., offset:end:3] = frequencies[axis, ..., offset:end:3]
    embedding = torch.cat((interleaved, interleaved), dim=-1)
    cosine = embedding.cos().unsqueeze(0)
    full_sine = embedding.sin().unsqueeze(0)
    half = full_sine.shape[-1] // 2
    sine = full_sine[..., :half]
    negative_sine = -full_sine[..., half:]
    matrix = torch.stack((cosine[..., :half], negative_sine, sine, cosine[..., half:]), dim=-1)
    return matrix.reshape(*matrix.shape[:-1], 2, 2).unsqueeze(2)


def _apply_rope(value: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    original_shape = value.shape
    pairs = value.reshape(*value.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2)
    pairs = pairs.to(matrix.dtype)
    output = matrix[..., 0] * pairs[..., 0] + matrix[..., 1] * pairs[..., 1]
    return output.movedim(-1, -2).reshape(original_shape).to(value.dtype)


def _norm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    matrix: torch.Tensor,
    query_norm: torch.nn.RMSNorm,
    key_norm: torch.nn.RMSNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.is_grad_enabled():
        return _apply_rope(query_norm(query), matrix), _apply_rope(key_norm(key), matrix)
    import dinkster_kitchen  # pyright: ignore[reportMissingTypeStubs]

    with (
        materialized_rms_norm_weight(query_norm) as query_weight,
        materialized_rms_norm_weight(key_norm) as key_weight,
    ):
        return dinkster_kitchen.rms_rope_split_half_(
            query,
            key,
            matrix,
            query_weight.detach(),
            key_weight.detach(),
            epsilon=float(query_norm.eps or 1e-5),
            rot_dim=query.shape[-1],
        )


class Ideogram4Attention(torch.nn.Module):
    _attention_kernel: AttentionKernel

    def __init__(
        self,
        config: _Config,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.heads = config.attention_heads
        self.head_dim = config.attention_head_dim
        self.qkv = operations.linear(hidden, hidden * 3, bias=False)
        self.norm_q = operations.rms_norm(self.head_dim, eps=1e-5)
        self.norm_k = operations.rms_norm(self.head_dim, eps=1e-5)
        self.o = operations.linear(hidden, hidden, bias=False)
        object.__setattr__(self, "_attention_kernel", attention_kernel)

    def forward(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor | None,
        rope: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = hidden.shape
        qkv = self.qkv(hidden).view(batch, length, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query, key = _norm_rope(query, key, rope, self.norm_q, self.norm_k)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        output = self._attention_kernel(query, key, value, mask=mask)
        return self.o(output.transpose(1, 2).reshape(batch, length, -1))


class Ideogram4FeedForward(torch.nn.Module):
    def __init__(self, config: _Config, *, operations: Operations) -> None:
        super().__init__()
        self.w1 = operations.linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = operations.linear(config.intermediate_size, config.hidden_size, bias=False)
        self.w3 = operations.linear(config.hidden_size, config.intermediate_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(hidden)) * self.w3(hidden))


class Ideogram4Block(torch.nn.Module):
    def __init__(
        self,
        config: _Config,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.attention = Ideogram4Attention(
            config, operations=operations, attention_kernel=attention_kernel
        )
        self.feed_forward = Ideogram4FeedForward(config, operations=operations)
        self.attention_norm1 = operations.rms_norm(hidden, eps=config.norm_eps)
        self.ffn_norm1 = operations.rms_norm(hidden, eps=config.norm_eps)
        self.attention_norm2 = operations.rms_norm(hidden, eps=config.norm_eps)
        self.ffn_norm2 = operations.rms_norm(hidden, eps=config.norm_eps)
        self.adaln_modulation = operations.linear(config.adaln_dim, hidden * 4)

    def forward(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor | None,
        rope: torch.Tensor,
        adaln: torch.Tensor,
    ) -> torch.Tensor:
        scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaln_modulation(adaln).chunk(4, dim=-1)
        attended = self.attention(self.attention_norm1(hidden) * (1.0 + scale_msa), mask, rope)
        hidden = hidden + gate_msa.tanh() * self.attention_norm2(attended)
        feed_forward = self.feed_forward(self.ffn_norm1(hidden) * (1.0 + scale_mlp))
        return hidden + gate_mlp.tanh() * self.ffn_norm2(feed_forward)


def _sinusoidal_embedding(timestep: torch.Tensor, width: int) -> torch.Tensor:
    timestep = timestep.float()
    half = width // 2
    frequency = math.log(1e4) / (half - 1)
    frequency = torch.exp(
        torch.arange(half, dtype=torch.float32, device=timestep.device) * -frequency
    )
    embedding = timestep.unsqueeze(-1) * frequency
    embedding = torch.cat((embedding.sin(), embedding.cos()), dim=-1)
    return F.pad(embedding, (0, width % 2))


class Ideogram4TimeEmbedding(torch.nn.Module):
    def __init__(self, width: int, *, operations: Operations) -> None:
        super().__init__()
        self.width = width
        self.mlp_in = operations.linear(width, width)
        self.mlp_out = operations.linear(width, width)

    def forward(self, timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        embedding = _sinusoidal_embedding(1e4 * timestep.float(), self.width).to(dtype)
        return self.mlp_out(F.silu(self.mlp_in(embedding)))


class Ideogram4FinalLayer(torch.nn.Module):
    def __init__(self, config: _Config, *, operations: Operations) -> None:
        super().__init__()
        self.norm_final = operations.layer_norm(
            config.hidden_size, eps=1e-6, elementwise_affine=False
        )
        self.linear = operations.linear(config.hidden_size, config.latent_channels)
        self.adaln_modulation = operations.linear(config.adaln_dim, config.hidden_size)

    def forward(self, hidden: torch.Tensor, adaln: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + self.adaln_modulation(F.silu(adaln))
        return self.linear(self.norm_final(hidden) * scale)


class Ideogram4DiT(torch.nn.Module):
    def __init__(
        self,
        config: _Config = IDEOGRAM4_CONFIG,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
    ) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        self.input_proj = operations.linear(config.latent_channels, hidden)
        self.llm_cond_norm = operations.rms_norm(config.text_width, eps=1e-6)
        self.llm_cond_proj = operations.linear(config.text_width, hidden)
        self.t_embedding = Ideogram4TimeEmbedding(hidden, operations=operations)
        self.adaln_proj = operations.linear(hidden, config.adaln_dim)
        self.embed_image_indicator = operations.embedding(2, hidden)
        self.layers = torch.nn.ModuleList(
            Ideogram4Block(config, operations=operations, attention_kernel=attention_kernel)
            for _ in range(config.layers)
        )
        self.final_layer = Ideogram4FinalLayer(config, operations=operations)

    def _image_tokens(self, latent: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = latent.shape
        patch_h, patch_w = self.config.patch
        tokens = latent.view(batch, self.config.ae_channels, patch_h, patch_w, height, width)
        return tokens.permute(0, 4, 5, 2, 3, 1).reshape(batch, height * width, channels)

    def _image(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, _, channels = tokens.shape
        patch_h, patch_w = self.config.patch
        image = tokens.reshape(batch, height, width, patch_h, patch_w, self.config.ae_channels)
        return image.permute(0, 5, 3, 4, 1, 2).reshape(batch, channels, height, width)

    @staticmethod
    def _image_positions(height: int, width: int, device: torch.device) -> torch.Tensor:
        row = torch.arange(height, device=device).view(-1, 1).expand(height, width).reshape(-1)
        column = torch.arange(width, device=device).view(1, -1).expand(height, width).reshape(-1)
        return torch.stack((torch.zeros_like(row), row, column), dim=1) + _IMAGE_POSITION_OFFSET

    def _backbone(
        self,
        context: torch.Tensor | None,
        tokens: torch.Tensor,
        timestep: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None,
        indicator: torch.Tensor,
    ) -> torch.Tensor:
        image_mask = (indicator == _OUTPUT_IMAGE).to(tokens.dtype).unsqueeze(-1)
        hidden = self.input_proj(tokens * image_mask) * image_mask
        time = self.t_embedding(timestep, tokens.dtype)
        if timestep.ndim == 1:
            time = time.unsqueeze(1)
        adaln = F.silu(self.adaln_proj(time))
        if context is not None:
            text_length = context.shape[1]
            text_mask = (indicator[:, :text_length] == _LLM_TOKEN).to(tokens.dtype).unsqueeze(-1)
            projected = self.llm_cond_proj(self.llm_cond_norm(context * text_mask)) * text_mask
            hidden[:, :text_length] = hidden[:, :text_length] + projected
        hidden = hidden + self.embed_image_indicator((indicator == _OUTPUT_IMAGE).long())
        rope = _rope_matrix(
            positions,
            head_dim=self.config.attention_head_dim,
            theta=self.config.rope_theta,
            rope_dims=self.config.rope_dims,
        )
        if mask is not None and mask.dtype == torch.bool:
            mask = torch.zeros_like(mask, dtype=hidden.dtype).masked_fill_(
                ~mask, -torch.finfo(hidden.dtype).max
            )
        queue = make_prefetch_queue(self.layers)
        try:
            for layer in self.layers:
                prefetch_queue_pop(queue, layer)
                hidden = layer(hidden, mask, rope, adaln)
            prefetch_queue_pop(queue, None)
        finally:
            close_prefetch_queue(queue)
        return self.final_layer(hidden, adaln)

    def forward(
        self,
        latent: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latent.ndim != 4 or latent.shape[1] != self.config.latent_channels:
            raise ValueError(
                "Ideogram 4 latent must have shape"
                f" [batch,{self.config.latent_channels},height,width]"
            )
        batch, _, height, width = latent.shape
        if timesteps.shape != (batch,) or timesteps.device != latent.device:
            raise ValueError("Ideogram 4 timesteps must match the latent batch and device")
        timesteps = 1.0 - timesteps
        image_tokens = self._image_tokens(latent)
        image_positions = self._image_positions(height, width, latent.device)
        if context is None:
            if attention_mask is not None:
                raise ValueError("image-only Ideogram 4 conditioning has no attention mask")
            length = image_tokens.shape[1]
            positions = image_positions.unsqueeze(0).expand(batch, length, 3)
            indicator = torch.full(
                (batch, length), _OUTPUT_IMAGE, dtype=torch.long, device=latent.device
            )
            output = self._backbone(None, image_tokens, timesteps, positions, None, indicator)
            return -self._image(output, height, width)
        if (
            context.ndim != 3
            or context.shape[0] != batch
            or context.shape[2] != self.config.text_width
            or context.device != latent.device
        ):
            raise ValueError(
                f"Ideogram 4 context must have shape [batch,tokens,{self.config.text_width}]"
            )
        text_length = context.shape[1]
        if attention_mask is not None and attention_mask.shape != (batch, text_length):
            raise ValueError("Ideogram 4 attention mask must match the text rows")
        length = text_length + image_tokens.shape[1]
        tokens = torch.zeros(
            batch, length, latent.shape[1], dtype=latent.dtype, device=latent.device
        )
        tokens[:, text_length:] = image_tokens
        text_positions = torch.arange(text_length, device=latent.device).view(-1, 1).expand(-1, 3)
        positions = torch.cat((text_positions, image_positions), dim=0).unsqueeze(0)
        positions = positions.expand(batch, length, 3)
        indicator = torch.empty(batch, length, dtype=torch.long, device=latent.device)
        indicator[:, :text_length] = _LLM_TOKEN
        indicator[:, text_length:] = _OUTPUT_IMAGE
        mask = None
        if attention_mask is not None:
            segments = torch.ones(batch, length, dtype=torch.long, device=latent.device)
            padding = attention_mask == 0
            segments[:, :text_length][padding] = -1
            indicator[:, :text_length][padding] = 0
            mask = (segments.unsqueeze(2) == segments.unsqueeze(1)).unsqueeze(1)
        output = self._backbone(context, tokens, timesteps, positions, mask, indicator)
        return -self._image(output[:, text_length:], height, width)


__all__ = [
    "Ideogram4Attention",
    "Ideogram4Block",
    "Ideogram4DiT",
    "Ideogram4FeedForward",
    "Ideogram4FinalLayer",
    "Ideogram4TimeEmbedding",
]
