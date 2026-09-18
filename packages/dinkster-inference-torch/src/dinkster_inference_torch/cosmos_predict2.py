"""Native torch Cosmos Predict2 diffusion transformer, image path.

Transcribed from comfy/ldm/cosmos/predict2.py and
comfy/ldm/cosmos/position_embedding.py @ 82f839f5e737d8bfce480872ba05
e5a430f2526f, restricted to the legs the Anima image checkpoint
exercises: rope3d positional embedding without fps modulation,
AdaLN-LoRA modulation always on, and a zero padding-mask channel
always concatenated to the latent. The video-only legs (fps-modulated
rope, learnable absolute positional embeddings, resized non-zero
padding masks) are not ported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F

from .attention import AttentionKernel, select_attention
from .model_prefetch import close_prefetch_queue, make_prefetch_queue, prefetch_queue_pop
from .operations import INITLESS, Operations, materialized_rms_norm_weight

_DEFAULT_ATTENTION = select_attention("flux").kernel
_NORM_EPS = 1e-6
_ROPE_THETA = 10000.0


@dataclass(frozen=True, slots=True)
class CosmosPredict2Geometry:
    """Construction geometry for production and reduced CPU proofs."""

    in_channels: int
    out_channels: int
    patch_spatial: int
    patch_temporal: int
    model_channels: int
    num_blocks: int
    num_heads: int
    crossattn_emb_channels: int
    adaln_lora_dim: int
    mlp_ratio: float = 4.0
    rope_h_extrapolation_ratio: float = 1.0
    rope_w_extrapolation_ratio: float = 1.0
    rope_t_extrapolation_ratio: float = 1.0

    def __post_init__(self) -> None:
        dimensions = (
            self.in_channels,
            self.out_channels,
            self.patch_spatial,
            self.patch_temporal,
            self.model_channels,
            self.num_blocks,
            self.num_heads,
            self.crossattn_emb_channels,
            self.adaln_lora_dim,
        )
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("geometry dimensions must be positive integers")
        ratios = (
            self.mlp_ratio,
            self.rope_h_extrapolation_ratio,
            self.rope_w_extrapolation_ratio,
            self.rope_t_extrapolation_ratio,
        )
        if any(type(value) is not float or value <= 0.0 for value in ratios):
            raise ValueError("geometry ratios must be positive floats")
        if self.model_channels % self.num_heads != 0:
            raise ValueError("model_channels must divide evenly into num_heads")
        head_dim = self.model_channels // self.num_heads
        if head_dim % 2 != 0:
            raise ValueError("attention head dimension must be even")
        # Three rope axes split as dim_h = dim_w = head_dim // 6 * 2 and
        # dim_t = the remainder; each axis needs at least two frequency
        # pairs so its NTK exponent dim / (dim - 2) is finite.
        dim_h = head_dim // 6 * 2
        if dim_h < 4 or head_dim - 2 * dim_h < 4:
            raise ValueError("attention head dimension is too small for three rope axes")

    @property
    def head_dim(self) -> int:
        return self.model_channels // self.num_heads

    @property
    def ffn_width(self) -> int:
        return int(self.model_channels * self.mlp_ratio)


def _rope_axis_table(
    length: int, extrapolation_ratio: float, axis_dim: int, device: torch.device | None
) -> torch.Tensor:
    """One rope axis as float32 [length, axis_dim / 2, 2, 2] rotation blocks."""
    theta = _ROPE_THETA * extrapolation_ratio ** (axis_dim / (axis_dim - 2))
    exponents = (
        torch.arange(0, axis_dim, 2, dtype=torch.float32, device=device)[: axis_dim // 2] / axis_dim
    )
    frequencies = 1.0 / theta**exponents
    angles = torch.outer(torch.arange(length, dtype=torch.float32, device=device), frequencies)
    cosine, sine = torch.cos(angles), torch.sin(angles)
    return torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(length, -1, 2, 2)


def cosmos_predict2_rope_table(
    geometry: CosmosPredict2Geometry,
    temporal: int,
    height: int,
    width: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Float32 rope table [temporal * height * width, head_dim / 2, 2, 2].

    Axes concatenate along the frequency dimension in t, h, w order and
    positions flatten row-major over (t, h, w), matching the reference
    VideoRopePosition3DEmb image path. The table is computed directly on
    ``device``, matching the reference's on-device trigonometry.
    """
    head_dim = geometry.head_dim
    dim_h = head_dim // 6 * 2
    dim_t = head_dim - 2 * dim_h
    t_table = _rope_axis_table(temporal, geometry.rope_t_extrapolation_ratio, dim_t, device)
    h_table = _rope_axis_table(height, geometry.rope_h_extrapolation_ratio, dim_h, device)
    w_table = _rope_axis_table(width, geometry.rope_w_extrapolation_ratio, dim_h, device)
    parts = (
        t_table.view(temporal, 1, 1, dim_t // 2, 2, 2).expand(temporal, height, width, -1, -1, -1),
        h_table.view(1, height, 1, dim_h // 2, 2, 2).expand(temporal, height, width, -1, -1, -1),
        w_table.view(1, 1, width, dim_h // 2, 2, 2).expand(temporal, height, width, -1, -1, -1),
    )
    return torch.cat(parts, dim=-3).reshape(temporal * height * width, head_dim // 2, 2, 2)


def _apply_split_half_rope(value: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    pairs = value.reshape(*value.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2)
    pairs = pairs.to(table.dtype)
    rotated = table[..., 0] * pairs[..., 0] + table[..., 1] * pairs[..., 1]
    return rotated.movedim(-1, -2).reshape(value.shape).type_as(value)


def _fused_norm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    table: torch.Tensor,
    query_norm: torch.nn.RMSNorm,
    key_norm: torch.nn.RMSNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.is_grad_enabled():
        return (
            _apply_split_half_rope(query_norm(query), table),
            _apply_split_half_rope(key_norm(key), table),
        )
    import comfy_kitchen  # pyright: ignore[reportMissingTypeStubs]

    with (
        materialized_rms_norm_weight(query_norm) as query_weight,
        materialized_rms_norm_weight(key_norm) as key_weight,
    ):
        return comfy_kitchen.rms_rope_split_half_(
            query,
            key,
            table,
            query_weight.detach(),
            key_weight.detach(),
            epsilon=_NORM_EPS,
        )


def cosmos_predict2_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Float32 sinusoidal embedding [B, T, dim], cosine rows before sine."""
    flat = timesteps.flatten().float()
    half = dim // 2
    exponent = (
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    angles = flat[:, None] * torch.exp(exponent)[None, :]
    embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
    return embedding.view(timesteps.shape[0], timesteps.shape[1], dim)


class _Timesteps(torch.nn.Module):
    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return cosmos_predict2_timestep_embedding(timesteps, self.num_channels)


class _TimestepEmbedding(torch.nn.Module):
    """AdaLN-LoRA timestep head: returns the raw sinusoidal sample as the
    block embedding and the two-linear MLP output as the shared LoRA rows."""

    def __init__(self, width: int, *, operations: Operations) -> None:
        super().__init__()
        self.linear_1 = operations.linear(width, width, bias=False)
        self.activation = torch.nn.SiLU()
        self.linear_2 = operations.linear(width, 3 * width, bias=False)

    def forward(self, sample: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        adaln_lora = self.linear_2(self.activation(self.linear_1(sample)))
        return sample, adaln_lora


class _PatchifyRearrange(torch.nn.Module):
    """Parameter-free b c (t r) (h m) (w n) -> b t h w (c r m n) rearrange."""

    def __init__(self, spatial: int, temporal: int) -> None:
        super().__init__()
        self.spatial = spatial
        self.temporal = temporal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = x.shape
        value = x.view(
            batch,
            channels,
            frames // self.temporal,
            self.temporal,
            height // self.spatial,
            self.spatial,
            width // self.spatial,
            self.spatial,
        )
        value = value.permute(0, 2, 4, 6, 1, 3, 5, 7)
        return value.reshape(
            batch,
            frames // self.temporal,
            height // self.spatial,
            width // self.spatial,
            channels * self.temporal * self.spatial * self.spatial,
        )


class _PatchEmbed(torch.nn.Module):
    def __init__(
        self,
        spatial: int,
        temporal: int,
        in_channels: int,
        out_channels: int,
        *,
        operations: Operations,
    ) -> None:
        super().__init__()
        self.proj = torch.nn.Sequential(
            _PatchifyRearrange(spatial, temporal),
            operations.linear(in_channels * spatial * spatial * temporal, out_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _GPT2FeedForward(torch.nn.Module):
    def __init__(self, width: int, ffn_width: int, *, operations: Operations) -> None:
        super().__init__()
        self.activation = torch.nn.GELU()
        self.layer1 = operations.linear(width, ffn_width, bias=False)
        self.layer2 = operations.linear(ffn_width, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.activation(self.layer1(x)))


class CosmosPredict2Attention(torch.nn.Module):
    """One Predict2 attention projection; rope marks the self-attention path."""

    _attention_kernel: AttentionKernel

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        heads: int,
        head_dim: int,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        inner = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim
        self.q_proj = operations.linear(query_dim, inner, bias=False)
        self.q_norm = operations.rms_norm(head_dim, eps=_NORM_EPS)
        self.k_proj = operations.linear(context_dim, inner, bias=False)
        self.k_norm = operations.rms_norm(head_dim, eps=_NORM_EPS)
        self.v_proj = operations.linear(context_dim, inner, bias=False)
        self.output_proj = operations.linear(inner, query_dim, bias=False)
        object.__setattr__(self, "_attention_kernel", attention_kernel)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        rope_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        selected = x if context is None else context
        batch, length = x.shape[0], x.shape[1]
        query_shape = (batch, -1, self.heads, self.head_dim)
        kv_shape = (selected.shape[0], -1, self.heads, self.head_dim)
        query = self.q_proj(x).view(query_shape)
        key = self.k_proj(selected).view(kv_shape)
        value = self.v_proj(selected).view(kv_shape)
        if rope_table is not None:
            query, key = _fused_norm_rope(query, key, rope_table, self.q_norm, self.k_norm)
        else:
            query = self.q_norm(query)
            key = self.k_norm(key)
        output = self._attention_kernel(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            mask=None,
            causal=False,
            scale=None,
            enable_gqa=False,
        )
        output = output.transpose(1, 2).reshape(batch, length, self.heads * self.head_dim)
        return self.output_proj(output)


class CosmosPredict2Block(torch.nn.Module):
    """Self-attention, cross-attention, and MLP with AdaLN-LoRA modulation.

    The residual stream keeps the incoming dtype while attention and MLP
    run at the timestep-embedding dtype; the reference promotes the fp16
    residual stream to fp32 before the block loop for this reason.
    """

    def __init__(
        self,
        geometry: CosmosPredict2Geometry,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        width = geometry.model_channels
        self.layer_norm_self_attn = operations.layer_norm(
            width, eps=_NORM_EPS, elementwise_affine=False
        )
        self.self_attn = CosmosPredict2Attention(
            width,
            width,
            geometry.num_heads,
            geometry.head_dim,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.layer_norm_cross_attn = operations.layer_norm(
            width, eps=_NORM_EPS, elementwise_affine=False
        )
        self.cross_attn = CosmosPredict2Attention(
            width,
            geometry.crossattn_emb_channels,
            geometry.num_heads,
            geometry.head_dim,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.layer_norm_mlp = operations.layer_norm(width, eps=_NORM_EPS, elementwise_affine=False)
        self.mlp = _GPT2FeedForward(width, geometry.ffn_width, operations=operations)
        self.adaln_modulation_self_attn = self._adaln_modulation(geometry, operations)
        self.adaln_modulation_cross_attn = self._adaln_modulation(geometry, operations)
        self.adaln_modulation_mlp = self._adaln_modulation(geometry, operations)

    @staticmethod
    def _adaln_modulation(
        geometry: CosmosPredict2Geometry, operations: Operations
    ) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.SiLU(),
            operations.linear(geometry.model_channels, geometry.adaln_lora_dim, bias=False),
            operations.linear(geometry.adaln_lora_dim, 3 * geometry.model_channels, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_table: torch.Tensor,
        adaln_lora: torch.Tensor,
    ) -> torch.Tensor:
        residual_dtype = x.dtype
        compute_dtype = emb.dtype
        batch, temporal, height, width, dim = x.shape

        def modulation(head: torch.nn.Sequential) -> tuple[torch.Tensor, ...]:
            rows = (head(emb) + adaln_lora).chunk(3, dim=-1)
            return tuple(row[:, :, None, None, :] for row in rows)

        shift, scale, gate = modulation(self.adaln_modulation_self_attn)
        normalized = self.layer_norm_self_attn(x) * (1 + scale) + shift
        result = self.self_attn(
            normalized.to(compute_dtype).reshape(batch, -1, dim),
            None,
            rope_table,
        ).view(batch, temporal, height, width, dim)
        x = torch.addcmul(x, gate.to(residual_dtype), result.to(residual_dtype))

        shift, scale, gate = modulation(self.adaln_modulation_cross_attn)
        normalized = self.layer_norm_cross_attn(x) * (1 + scale) + shift
        result = self.cross_attn(
            normalized.to(compute_dtype).reshape(batch, -1, dim),
            crossattn_emb,
        ).view(batch, temporal, height, width, dim)
        x = torch.addcmul(x, gate.to(residual_dtype), result.to(residual_dtype))

        shift, scale, gate = modulation(self.adaln_modulation_mlp)
        normalized = (self.layer_norm_mlp(x) * (1 + scale) + shift).to(compute_dtype)
        return torch.addcmul(x, gate.to(residual_dtype), self.mlp(normalized).to(residual_dtype))


class CosmosPredict2FinalLayer(torch.nn.Module):
    def __init__(
        self,
        geometry: CosmosPredict2Geometry,
        *,
        operations: Operations,
    ) -> None:
        super().__init__()
        width = geometry.model_channels
        patch_rows = (
            geometry.patch_spatial * geometry.patch_spatial * geometry.patch_temporal
        ) * geometry.out_channels
        self.hidden_size = width
        self.layer_norm = operations.layer_norm(width, eps=_NORM_EPS, elementwise_affine=False)
        self.linear = operations.linear(width, patch_rows, bias=False)
        self.adaln_modulation = torch.nn.Sequential(
            torch.nn.SiLU(),
            operations.linear(width, geometry.adaln_lora_dim, bias=False),
            operations.linear(geometry.adaln_lora_dim, 2 * width, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        adaln_lora: torch.Tensor,
    ) -> torch.Tensor:
        shift, scale = (
            self.adaln_modulation(emb) + adaln_lora[:, :, : 2 * self.hidden_size]
        ).chunk(2, dim=-1)
        shift = shift[:, :, None, None, :]
        scale = scale[:, :, None, None, :]
        return self.linear(self.layer_norm(x) * (1 + scale) + shift)


class CosmosPredict2Model(torch.nn.Module):
    """Unregistered Cosmos Predict2 MiniTrainDIT source, image path only."""

    def __init__(
        self,
        geometry: CosmosPredict2Geometry,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
    ) -> None:
        super().__init__()
        if type(geometry) is not CosmosPredict2Geometry:
            raise TypeError("geometry must be exact CosmosPredict2Geometry")
        self.geometry = geometry
        width = geometry.model_channels
        self.t_embedder = torch.nn.Sequential(
            _Timesteps(width),
            _TimestepEmbedding(width, operations=operations),
        )
        self.x_embedder = _PatchEmbed(
            geometry.patch_spatial,
            geometry.patch_temporal,
            geometry.in_channels + 1,
            width,
            operations=operations,
        )
        self.blocks = torch.nn.ModuleList(
            CosmosPredict2Block(geometry, operations=operations, attention_kernel=attention_kernel)
            for _ in range(geometry.num_blocks)
        )
        self.final_layer = CosmosPredict2FinalLayer(geometry, operations=operations)
        self.t_embedding_norm = operations.rms_norm(width, eps=_NORM_EPS)

    def _validate_inputs(
        self, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor
    ) -> None:
        geometry = self.geometry
        if x.ndim != 5 or x.shape[1] != geometry.in_channels:
            raise ValueError(f"x must be [B, {geometry.in_channels}, T, H, W]")
        if not x.is_floating_point():
            raise ValueError("x must use a floating-point dtype")
        if timesteps.ndim not in (1, 2) or timesteps.shape[0] != x.shape[0]:
            raise ValueError("timesteps must be [B] or [B, T] matching the batch")
        if (
            context.ndim != 3
            or context.shape[0] != x.shape[0]
            or context.shape[2] != geometry.crossattn_emb_channels
        ):
            raise ValueError(f"context must be [B, rows, {geometry.crossattn_emb_channels}]")
        if not context.is_floating_point() or context.device != x.device:
            raise ValueError("context must be floating-point on the x device")

    def _unpatchify(self, rows: torch.Tensor) -> torch.Tensor:
        geometry = self.geometry
        batch, temporal, height, width, _ = rows.shape
        spatial, patch_temporal = geometry.patch_spatial, geometry.patch_temporal
        value = rows.view(
            batch,
            temporal,
            height,
            width,
            spatial,
            spatial,
            patch_temporal,
            geometry.out_channels,
        )
        value = value.permute(0, 7, 1, 6, 2, 4, 3, 5)
        return value.reshape(
            batch,
            geometry.out_channels,
            temporal * patch_temporal,
            height * spatial,
            width * spatial,
        )

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        geometry = self.geometry
        self._validate_inputs(x, timesteps, context)
        original = x.shape
        pad_t = (-original[2]) % geometry.patch_temporal
        pad_h = (-original[3]) % geometry.patch_spatial
        pad_w = (-original[4]) % geometry.patch_spatial
        if pad_t or pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode="circular")

        padding_mask = torch.zeros(x.shape[0], 1, *x.shape[2:], dtype=x.dtype, device=x.device)
        hidden = self.x_embedder(torch.cat((x, padding_mask), dim=1))
        temporal, height, width = hidden.shape[1:4]
        rope_table = (
            cosmos_predict2_rope_table(geometry, temporal, height, width, hidden.device)
            .unsqueeze(1)
            .unsqueeze(0)
        )

        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)
        sinusoidal = cast(_Timesteps, self.t_embedder[0])(timesteps)
        emb, adaln_lora = cast(_TimestepEmbedding, self.t_embedder[1])(sinusoidal.to(hidden.dtype))
        emb = self.t_embedding_norm(emb)

        # The residual stream carries large values; fp16 compute keeps the
        # residual in fp32 while attention and MLP run at fp16.
        if hidden.dtype == torch.float16:
            hidden = hidden.float()
        prefetch = make_prefetch_queue(self.blocks)
        try:
            for block in self.blocks:
                prefetch_queue_pop(prefetch, block)
                hidden = block(hidden, emb, context, rope_table, adaln_lora)
            prefetch_queue_pop(prefetch, None)
        finally:
            close_prefetch_queue(prefetch)

        rows = self.final_layer(hidden.to(context.dtype), emb, adaln_lora)
        output = self._unpatchify(rows)
        return output[:, :, : original[2], : original[3], : original[4]]


__all__ = [
    "CosmosPredict2Attention",
    "CosmosPredict2Block",
    "CosmosPredict2FinalLayer",
    "CosmosPredict2Geometry",
    "CosmosPredict2Model",
    "cosmos_predict2_rope_table",
    "cosmos_predict2_timestep_embedding",
]
