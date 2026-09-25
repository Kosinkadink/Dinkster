"""Unregistered MiniMax H3 video VAE source component.

This is the causal 3D encoder and ViT3D decoder from
``comfy/ldm/minimax/vae.py`` at ComfyUI
``2a68ce33b4c9ea6ee4283e618a74560cefb32694``. The default configuration,
state-dict names, normalization constants, temporal chunk plan, spatial
tiling, and split-half RoPE layout match that source. Construction alone does
not register the codec or claim family support.
"""

from __future__ import annotations

import math
from collections.abc import Generator
from contextlib import contextmanager
from typing import cast

import dinkster_kitchen  # pyright: ignore[reportMissingTypeStubs]
import torch
import torch.nn.functional as F
from dinkster_inference.minimax_h3_codecs import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    LATENTS_MEAN,
    LATENTS_STD,
    MiniMaxH3VideoVAEConfig,
)

from .attention import AttentionKernel, select_attention
from .memory import get_free_memory
from .operations import (
    INITLESS,
    CastOperations,
    Operations,
    ResidencyRouted,
    materialized_rms_norm_weight,
)
from .ops import cast_weight
from .quant_linear import linear_input_act

__all__ = [
    "CausalConv3d",
    "MiniMaxH3VideoVAE",
    "MiniMaxH3VideoVAEConfig",
    "RotaryEmbeddingND",
    "TemporalIsolatedGroupNorm",
    "ViT3DDecoder",
    "apply_rope_split_half",
    "create_token_ids",
]


def _operations_compute_dtype(operations: Operations) -> torch.dtype | None:
    return operations.dtype if isinstance(operations, CastOperations) else None


def _cast_direct_state(stored: torch.Tensor, dtype: torch.dtype | None) -> torch.Tensor:
    return stored if dtype is None else cast_weight(stored, dtype=dtype)


def _kitchen_ndhwc(input: torch.Tensor) -> bool:
    return (
        input.is_cuda
        and torch.version.hip is None
        and input.dtype in (torch.float16, torch.bfloat16)
    )


def _fused_norm_pad(
    input: torch.Tensor,
    norm: TemporalIsolatedGroupNorm | None,
    spatial_pad: tuple[int, int, int, int],
    front: int,
) -> torch.Tensor | None:
    if not _kitchen_ndhwc(input) or input.shape[1] % 8:
        return None
    if norm is None:
        return dinkster_kitchen.group_norm_silu_pad3d(
            input, None, None, 1, 0.0, (*spatial_pad, front), silu=False
        )
    with norm._materialized_affine(input) as (  # pyright: ignore[reportPrivateUsage]
        weight,
        bias,
    ):
        return dinkster_kitchen.group_norm_silu_pad3d(
            input,
            weight,
            bias,
            norm.num_groups,
            norm.eps,
            (*spatial_pad, front),
            silu=True,
        )


def _fp16_accum_conv(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    stride: tuple[int, int, int],
) -> torch.Tensor | None:
    if not input.is_cuda or input.dtype != torch.float16:
        return None
    return dinkster_kitchen.fp16_conv3d(input, weight, bias, residual, stride)


class CausalConv3d(ResidencyRouted, torch.nn.Conv3d):
    """Reflect spatial padding and front-only zero temporal padding."""

    causal_padding: tuple[int, int, int]
    _compute_dtype: torch.dtype | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        layer = operations.conv3d(in_channels, out_channels, kernel_size, stride=stride)
        torch.nn.Module.__init__(self)
        self.in_channels = layer.in_channels
        self.out_channels = layer.out_channels
        self.kernel_size = layer.kernel_size
        self.stride = layer.stride
        self.padding = layer.padding
        self.dilation = layer.dilation
        self.transposed = layer.transposed
        self.output_padding = layer.output_padding
        self.groups = layer.groups
        self.padding_mode = layer.padding_mode
        self._reversed_padding_repeated_twice = layer._reversed_padding_repeated_twice
        self.weight = layer.weight
        self.bias = layer.bias
        self._compute_dtype = _operations_compute_dtype(operations)
        if not isinstance(padding, int) and len(padding) != 3:
            raise ValueError("causal padding must have three entries")
        self.causal_padding = (padding, padding, padding) if isinstance(padding, int) else padding

    def reset_parameters(self) -> None:
        return None

    def _causal_forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        pre_norm: TemporalIsolatedGroupNorm | None,
        spatial_pad: tuple[int, int, int, int] | None,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        temporal, height, width = self.causal_padding
        if spatial_pad is None:
            spatial_pad = (width, width, height, height)
        front = 0 if input.shape[2] == 1 else temporal * 2
        ndhwc = _kitchen_ndhwc(input)
        fused = (
            _fused_norm_pad(input, pre_norm, spatial_pad, front)
            if pre_norm is not None or front or any(spatial_pad)
            else None
        )
        if fused is not None:
            x = fused
        else:
            x = input if pre_norm is None else F.silu(pre_norm(input), inplace=True)
            if any(spatial_pad):
                x = F.pad(x, (*spatial_pad, 0, 0), mode="reflect")
            if front:
                x = F.pad(x, (0, 0, 0, 0, front, 0))
        if input.shape[2] == 1 and temporal:
            weight = weight[:, :, -1:, :, :]
            output = F.conv3d(
                x,
                weight,
                bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        elif ndhwc:
            weight = weight.contiguous(memory_format=torch.channels_last_3d)
            fused_conv = _fp16_accum_conv(
                x, weight, bias, residual, cast(tuple[int, int, int], self.stride)
            )
            if fused_conv is not None:
                return fused_conv
            output = F.conv3d(
                x, weight, bias, self.stride, self.padding, self.dilation, self.groups
            )
        else:
            output = self._conv_forward(x, weight, bias)
        return output if residual is None else output.add_(residual)

    def _prefetch_dtype(self, stored: torch.Tensor) -> torch.dtype:
        return stored.dtype if self._compute_dtype is None else self._compute_dtype

    def forward(
        self,
        input: torch.Tensor,
        pre_norm: TemporalIsolatedGroupNorm | None = None,
        spatial_pad: tuple[int, int, int, int] | None = None,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is None:
            return self._causal_forward(
                input,
                _cast_direct_state(self.weight, self._compute_dtype),
                None if self.bias is None else _cast_direct_state(self.bias, self._compute_dtype),
                pre_norm,
                spatial_pad,
                residual,
            )
        with binding.lease() as lease:
            dtype = self.weight.dtype if self._compute_dtype is None else self._compute_dtype
            bias = None if self.bias is None else lease.get("bias", dtype=dtype)
            return self._causal_forward(
                input,
                lease.get("weight", dtype=dtype),
                bias,
                pre_norm,
                spatial_pad,
                residual,
            )


class TemporalIsolatedGroupNorm(ResidencyRouted, torch.nn.GroupNorm):
    """Group normalization whose statistics are isolated per frame."""

    _compute_dtype: torch.dtype | None

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        operations: Operations | None = None,
    ) -> None:
        if operations is not None and (device is not None or dtype is not None):
            raise ValueError("device and dtype cannot be combined with operations")
        if operations is not None and affine:
            layer = operations.group_norm(num_channels, num_groups=num_groups, eps=eps)
            torch.nn.Module.__init__(self)
            self.num_groups = layer.num_groups
            self.num_channels = layer.num_channels
            self.eps = layer.eps
            self.affine = layer.affine
            self.weight = layer.weight
            self.bias = layer.bias
        else:
            torch.nn.GroupNorm.__init__(
                self,
                num_groups,
                num_channels,
                eps=eps,
                affine=affine,
                device=device,
                dtype=dtype,
            )
        self._compute_dtype = None if operations is None else _operations_compute_dtype(operations)

    def _isolated_forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        if input.dim() != 5:
            return F.group_norm(input, self.num_groups, weight, bias, self.eps)
        batch, channels, frames, height, width = input.shape
        isolated = (
            input.permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(batch * frames, channels, 1, height, width)
        )
        isolated = F.group_norm(isolated, self.num_groups, weight, bias, self.eps)
        return (
            isolated.view(batch, frames, channels, height, width)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )

    def _prefetch_dtype(self, stored: torch.Tensor) -> torch.dtype:
        return stored.dtype if self._compute_dtype is None else self._compute_dtype

    @contextmanager
    def _materialized_affine(
        self, input: torch.Tensor
    ) -> Generator[tuple[torch.Tensor | None, torch.Tensor | None]]:
        weight = cast(torch.Tensor | None, self.weight)
        bias = cast(torch.Tensor | None, self.bias)
        dtype = (
            self._compute_dtype
            if self._compute_dtype is not None
            else input.dtype
            if weight is None
            else weight.dtype
        )
        binding = self._offloaded_residency()
        if binding is None:
            yield (
                None if weight is None else _cast_direct_state(weight, self._compute_dtype),
                None if bias is None else _cast_direct_state(bias, self._compute_dtype),
            )
            return
        with binding.lease() as lease:
            yield (
                None if weight is None else lease.get("weight", dtype=dtype),
                None if bias is None else lease.get("bias", dtype=dtype),
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        with self._materialized_affine(input) as (weight, bias):
            return self._isolated_forward(input, weight, bias)


def _group_norm_3d(
    num_channels: int, operations: Operations = INITLESS
) -> TemporalIsolatedGroupNorm:
    return TemporalIsolatedGroupNorm(32, num_channels, eps=1e-6, affine=True, operations=operations)


class Downsample3D(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        time_stride: int = 1,
        space_stride: int = 2,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.space_stride = space_stride
        self.conv = CausalConv3d(
            in_channels,
            out_channels,
            3,
            padding=(1, 0, 0),
            stride=(time_stride, space_stride, space_stride),
            operations=operations,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.space_stride == 2:
            return self.conv(x, spatial_pad=(0, 1, 0, 1))
        return self.conv(x)


class ResnetBlock3D(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = _group_norm_3d(in_channels, operations)
        self.norm2 = _group_norm_3d(self.out_channels, operations)
        self.conv1 = CausalConv3d(
            in_channels, self.out_channels, 3, padding=1, operations=operations
        )
        self.conv2 = CausalConv3d(
            self.out_channels, self.out_channels, 3, padding=1, operations=operations
        )
        self.nin_shortcut: CausalConv3d | None = None
        if in_channels != self.out_channels:
            self.nin_shortcut = CausalConv3d(
                in_channels, self.out_channels, 1, operations=operations
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, pre_norm=self.norm1)
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return self.conv2(h, pre_norm=self.norm2, residual=x)


class _EncoderLevel(torch.nn.Module):
    downsample: Downsample3D

    def __init__(
        self,
        blocks: list[ResnetBlock3D],
        downsample: Downsample3D | None,
    ) -> None:
        super().__init__()
        self.block = torch.nn.ModuleList(blocks)
        if downsample is not None:
            self.downsample = downsample


class EncoderFCN3D(torch.nn.Module):
    def __init__(
        self, config: MiniMaxH3VideoVAEConfig, *, operations: Operations = INITLESS
    ) -> None:
        super().__init__()
        levels = len(config.ch_mult)
        counts = (
            (config.num_res_blocks,) * levels
            if isinstance(config.num_res_blocks, int)
            else config.num_res_blocks
        )
        block_mid = [config.ch * multiplier for multiplier in config.ch_mult]
        block_in = [block_mid[0], *block_mid[:-1]]
        self.conv_in = CausalConv3d(
            config.in_channels, block_in[0], 3, padding=1, operations=operations
        )
        self.down = torch.nn.ModuleList()
        for level in range(levels):
            blocks = [
                ResnetBlock3D(
                    block_in[level] if index == 0 else block_mid[level],
                    block_mid[level],
                    operations=operations,
                )
                for index in range(counts[level])
            ]
            downsample = None
            if config.space_down[level] * config.time_down[level] > 1:
                downsample = Downsample3D(
                    block_mid[level],
                    block_mid[level],
                    time_stride=config.time_down[level],
                    space_stride=config.space_down[level],
                    operations=operations,
                )
            self.down.append(_EncoderLevel(blocks, downsample))
        self.num_res_blocks = counts
        self.norm_out = _group_norm_3d(block_mid[-1], operations)
        self.conv_out = CausalConv3d(
            block_mid[-1], config.z_channels * 2, 3, padding=1, operations=operations
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for down in self.down:
            level = cast(_EncoderLevel, down)
            for block in level.block:
                h = block(h)
            if hasattr(level, "downsample"):
                h = level.downsample(h)
        return self.conv_out(h, pre_norm=self.norm_out)


def create_token_ids(
    patch_dims: tuple[int, int, int], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    coords = []
    for size in patch_dims:
        axis = torch.arange(0.5, size, dtype=dtype, device=device)
        coords.append(2.0 * axis / size - 1.0)
    grid = torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=-1)
    return grid.flatten(0, len(patch_dims) - 1).unsqueeze(0)


class RotaryEmbeddingND(torch.nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, dim: int, rotary_base: float = 100.0, n_dim: int = 3) -> None:
        super().__init__()
        self.angle_scale = 2.0 * math.pi
        inv_freq = 1 / rotary_base ** torch.arange(0, 1, 2 * n_dim / dim, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, image_ids: torch.Tensor) -> torch.Tensor:
        angles = (
            self.angle_scale
            * image_ids[:, :, :, None].float()
            * self.inv_freq.to(image_ids.device)[None, None, None, :]
        ).flatten(2, 3)
        cosine, sine = torch.cos(angles), torch.sin(angles)
        table = torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(
            *angles.shape[:2], 1, angles.shape[-1], 2, 2
        )
        return table.to(image_ids.dtype)


def _apply_rope_split_half_one(x: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    pairs = x.reshape(*x.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).to(table.dtype)
    output = table[..., 0] * pairs[..., 0] + table[..., 1] * pairs[..., 1]
    return output.movedim(-1, -2).reshape_as(x).type_as(x)


def apply_rope_split_half(
    query: torch.Tensor, key: torch.Tensor, table: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the dinkster-kitchen split-half RoPE layout to query and key."""
    return _apply_rope_split_half_one(query, table), _apply_rope_split_half_one(key, table)


class FeedForward(torch.nn.Module):
    def __init__(self, dim: int, *, mult: int = 4, operations: Operations = INITLESS) -> None:
        super().__init__()
        inner = dim * mult
        self.w1 = operations.linear(dim, inner * 2)
        self.w2 = operations.linear(inner, dim)

    def forward(
        self,
        x: torch.Tensor,
        pre_norm: torch.nn.RMSNorm,
        residual: torch.Tensor,
        residual_scale: torch.Tensor,
    ) -> torch.Tensor:
        with materialized_rms_norm_weight(pre_norm) as weight:
            eps = torch.finfo(x.dtype).eps if pre_norm.eps is None else pre_norm.eps
            hidden = linear_input_act(self.w1, x, "rms_norm", weight, eps)
        return linear_input_act(
            self.w2,
            hidden,
            "swiglu",
            residual=residual,
            residual_scale=residual_scale,
        )


_DEFAULT_VAE_ATTENTION = select_attention("vae").kernel


class Attention(torch.nn.Module):
    _attention_kernel: AttentionKernel
    qk_norm_scale: torch.Tensor

    def __init__(
        self,
        heads: int,
        dim_head: int,
        *,
        eps: float = 1e-5,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_VAE_ATTENTION,
    ) -> None:
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner = heads * dim_head
        self.norm_q = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.norm_k = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.register_buffer("qk_norm_scale", torch.ones(dim_head), persistent=False)
        self.to_qkv = operations.linear(inner, inner * 3)
        self.to_out = operations.linear(inner, inner)
        object.__setattr__(self, "_attention_kernel", attention_kernel)

    def forward(
        self,
        x: torch.Tensor,
        rotary: torch.Tensor | None,
        pre_norm: torch.nn.RMSNorm,
        residual: torch.Tensor,
        residual_scale: torch.Tensor,
    ) -> torch.Tensor:
        batch, sequence, _ = x.shape
        with materialized_rms_norm_weight(pre_norm) as weight:
            eps = torch.finfo(x.dtype).eps if pre_norm.eps is None else pre_norm.eps
            qkv = linear_input_act(self.to_qkv, x, "rms_norm", weight, eps).view(
                batch, sequence, -1, 3 * self.dim_head
            )
        query, key, value = qkv.chunk(3, dim=-1)
        if rotary is not None:
            rotated = rotary.shape[-3] * 2
            rms_rope = (
                dinkster_kitchen.rms_rope_split_half
                if torch.is_grad_enabled()
                else dinkster_kitchen.rms_rope_split_half_
            )
            query, key = rms_rope(
                query,
                key,
                rotary,
                self.qk_norm_scale.to(query.device),
                epsilon=(
                    torch.finfo(query.dtype).eps if self.norm_q.eps is None else self.norm_q.eps
                ),
                rot_dim=rotated,
            )
        else:
            query = F.rms_norm(query, (self.dim_head,), self.norm_q.weight, self.norm_q.eps)
            key = F.rms_norm(key, (self.dim_head,), self.norm_k.weight, self.norm_k.eps)
        output = self._attention_kernel(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        ).nan_to_num_(0.0)
        return linear_input_act(
            self.to_out,
            output.transpose(1, 2).reshape(batch, sequence, -1),
            None,
            residual=residual,
            residual_scale=residual_scale,
        )


class TransformerBlock(ResidencyRouted, torch.nn.Module):
    _compute_dtype: torch.dtype | None

    def __init__(
        self,
        heads: int,
        dim_head: int,
        *,
        eps: float = 1e-5,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_VAE_ATTENTION,
    ) -> None:
        super().__init__()
        dim = heads * dim_head
        self.norm1 = operations.rms_norm(dim, eps=eps)
        self.attn = Attention(
            heads,
            dim_head,
            eps=eps,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.scale1 = torch.nn.Parameter(torch.empty(dim))
        self.norm2 = operations.rms_norm(dim, eps=eps)
        self.ff = FeedForward(dim, operations=operations)
        self.scale2 = torch.nn.Parameter(torch.empty(dim))
        self._compute_dtype = _operations_compute_dtype(operations)

    def _forward_owned(
        self,
        x: torch.Tensor,
        rotary: torch.Tensor | None,
        scale1: torch.Tensor,
        scale2: torch.Tensor,
    ) -> torch.Tensor:
        x = self.attn(x, rotary, self.norm1, x, scale1)
        return self.ff(x, self.norm2, x, scale2)

    def _prefetch_dtype(self, stored: torch.Tensor) -> torch.dtype:
        return stored.dtype if self._compute_dtype is None else self._compute_dtype

    def forward(self, x: torch.Tensor, rotary: torch.Tensor | None = None) -> torch.Tensor:
        scale1_dtype = self._prefetch_dtype(self.scale1)
        scale2_dtype = self._prefetch_dtype(self.scale2)
        binding = self._offloaded_residency()
        if binding is None:
            scale1 = _cast_direct_state(self.scale1, self._compute_dtype).to(device=x.device)
            scale2 = _cast_direct_state(self.scale2, self._compute_dtype).to(device=x.device)
            return self._forward_owned(x, rotary, scale1, scale2)
        with binding.lease() as lease:
            return self._forward_owned(
                x,
                rotary,
                lease.get("scale1", dtype=scale1_dtype),
                lease.get("scale2", dtype=scale2_dtype),
            )


class ViT3DDecoder(ResidencyRouted, torch.nn.Module):
    mask_token: torch.Tensor
    _compute_dtype: torch.dtype | None

    def __init__(
        self,
        *,
        patch_size: int = 16,
        patch_size_t: int = 4,
        in_channels: int = 24,
        out_channels: int = 3,
        num_layers: int = 36,
        heads: int = 32,
        dim_head: int = 64,
        rope_theta: float = 100.0,
        rope_dim_ratio: float = 0.75,
        eps: float = 1e-5,
        num_register_tokens: int = 4,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_VAE_ATTENTION,
    ) -> None:
        super().__init__()
        dim = heads * dim_head
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens
        self.pos_embed = RotaryEmbeddingND(int(dim_head * rope_dim_ratio), rope_theta)
        self.x_embedder = operations.linear(in_channels, dim)
        self.register_tokens = torch.nn.Parameter(torch.empty(1, num_register_tokens, dim))
        self.register_buffer("mask_token", torch.empty(1, 1, dim))
        self.transformer_blocks = torch.nn.ModuleList(
            TransformerBlock(
                heads,
                dim_head,
                eps=eps,
                operations=operations,
                attention_kernel=attention_kernel,
            )
            for _ in range(num_layers)
        )
        self.norm_out = operations.layer_norm(dim, eps=eps)
        self.proj_out = operations.linear(
            dim, out_channels * patch_size_t * patch_size * patch_size
        )
        self._compute_dtype = _operations_compute_dtype(operations)

    def _prefetch_dtype(self, stored: torch.Tensor) -> torch.dtype:
        return stored.dtype if self._compute_dtype is None else self._compute_dtype

    def _forward_owned(self, x: torch.Tensor, register_tokens: torch.Tensor) -> torch.Tensor:
        batch, _, frames, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens.transpose(1, 2).contiguous().transpose(1, 2)
        hidden = self.x_embedder(tokens)
        patches = hidden.shape[1]
        suffix = 1 + self.num_register_tokens
        hidden = torch.cat(
            (
                hidden,
                register_tokens.to(device=hidden.device).expand(batch, -1, -1),
                torch.zeros_like(hidden[:, :1]),
            ),
            dim=1,
        )
        image_ids = create_token_ids((frames, height, width), x.device, x.dtype).expand(
            batch, -1, -1
        )
        suffix_ids = torch.zeros((batch, suffix, 3), device=x.device, dtype=image_ids.dtype)
        rotary = self.pos_embed(torch.cat((image_ids, suffix_ids), dim=1))
        for block in self.transformer_blocks:
            hidden = block(hidden, rotary)
        output = self.proj_out(self.norm_out(hidden))[:, :patches]
        output = output.view(
            batch,
            frames,
            height,
            width,
            self.out_channels,
            self.patch_size_t,
            self.patch_size,
            self.patch_size,
        )
        return (
            output.permute(0, 4, 1, 5, 2, 6, 3, 7)
            .contiguous()
            .reshape(
                batch,
                self.out_channels,
                frames * self.patch_size_t,
                height * self.patch_size,
                width * self.patch_size,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is None:
            register_tokens = _cast_direct_state(self.register_tokens, self._compute_dtype).to(
                device=x.device
            )
            return self._forward_owned(x, register_tokens)
        with binding.lease() as lease:
            register_tokens = lease.get(
                "register_tokens", dtype=self._prefetch_dtype(self.register_tokens)
            )
            return self._forward_owned(x, register_tokens)


class MiniMaxH3VideoVAE(ResidencyRouted, torch.nn.Module):
    """Exact H3 video VAE source with bounded temporal and spatial work."""

    comfy_has_chunked_io = True
    latents_mean: torch.Tensor
    latents_std: torch.Tensor
    pixel_mean: torch.Tensor
    pixel_std: torch.Tensor
    _compute_dtype: torch.dtype | None

    def __init__(
        self,
        config: MiniMaxH3VideoVAEConfig | None = None,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_VAE_ATTENTION,
    ) -> None:
        super().__init__()
        if config is None:
            config = MiniMaxH3VideoVAEConfig()
        self.config = config
        self.vae_ratio = math.prod(config.space_down)
        self.vae_ratio_t = math.prod(config.time_down)
        self.clip_length = config.clip_length
        self.token_drop = config.token_drop
        self.frame_pre_padding = (-self.clip_length) % self.vae_ratio_t
        self.tokens_chunk_size = math.ceil(self.clip_length / self.vae_ratio_t)
        self.token_overlap = (-self.token_drop) % self.tokens_chunk_size
        self.frame_overlap = max(self.token_overlap * self.vae_ratio_t - self.frame_pre_padding, 0)
        self.tiling = config.tiling
        self.tile_size = config.tile_size
        self.tile_overlap_min = config.tile_overlap_min
        self._compute_dtype = _operations_compute_dtype(operations)

        self.encoder = EncoderFCN3D(config, operations=operations)
        self.quant_conv = CausalConv3d(
            config.z_channels * 2, config.embed_dim * 2, 1, operations=operations
        )
        self.post_quant_conv = CausalConv3d(
            config.embed_dim, config.z_channels, 1, operations=operations
        )
        self.decoder = ViT3DDecoder(
            patch_size=self.vae_ratio,
            patch_size_t=self.vae_ratio_t,
            in_channels=config.z_channels,
            out_channels=config.out_channels,
            num_layers=config.decoder_num_layers,
            heads=config.decoder_heads,
            dim_head=config.decoder_dim_head,
            rope_theta=config.decoder_rope_theta,
            rope_dim_ratio=config.decoder_rope_dim_ratio,
            num_register_tokens=config.decoder_num_register_tokens,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.register_buffer("latents_mean", torch.tensor(LATENTS_MEAN[: config.embed_dim]))
        self.register_buffer("latents_std", torch.tensor(LATENTS_STD[: config.embed_dim]))
        self.register_buffer(
            "pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False
        )
        self.register_buffer(
            "pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False
        )

    def _prefetch_dtype(self, stored: torch.Tensor) -> torch.dtype:
        return stored.dtype if self._compute_dtype is None else self._compute_dtype

    def residency_prefetch(
        self,
    ) -> tuple[object, tuple[tuple[str, torch.dtype | None], ...]] | None:
        binding = self._offloaded_residency()
        if binding is None:
            return None
        return binding.mechanism, (
            (binding.key("latents_mean"), self._prefetch_dtype(self.latents_mean)),
            (binding.key("latents_std"), self._prefetch_dtype(self.latents_std)),
        )

    def _encode_moments(self, content: torch.Tensor) -> torch.Tensor:
        return self.quant_conv(self.encoder(content))

    def _decode_pixels(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant_conv(latent))

    def _normalize_pixels(self, content: torch.Tensor) -> torch.Tensor:
        return (
            content.add(1.0)
            .mul_(0.5)
            .sub_(self.pixel_mean.to(content))
            .div_(self.pixel_std.to(content))
        )

    def _finalize_pixels(self, content: torch.Tensor) -> torch.Tensor:
        content = content.float()
        content = content * self.pixel_std.to(device=content.device, dtype=torch.float32)
        return content.add_(self.pixel_mean.to(device=content.device, dtype=torch.float32)).clamp_(
            0.0, 1.0
        )

    def _normalize_latents(self, mean: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is not None:
            with binding.lease() as lease:
                center = lease.get("latents_mean", dtype=self._prefetch_dtype(self.latents_mean))
                scale = lease.get("latents_std", dtype=self._prefetch_dtype(self.latents_std))
                return (mean - center.view(1, -1, 1, 1, 1)) / scale.view(1, -1, 1, 1, 1)
        center = _cast_direct_state(self.latents_mean, self._compute_dtype).to(device=mean.device)
        scale = _cast_direct_state(self.latents_std, self._compute_dtype).to(device=mean.device)
        center = center.view(1, -1, 1, 1, 1)
        scale = scale.view(1, -1, 1, 1, 1)
        return (mean - center) / scale

    def _denormalize_latents(self, latent: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is not None:
            with binding.lease() as lease:
                center = lease.get("latents_mean", dtype=self._prefetch_dtype(self.latents_mean))
                scale = lease.get("latents_std", dtype=self._prefetch_dtype(self.latents_std))
                return latent * scale.view(1, -1, 1, 1, 1) + center.view(1, -1, 1, 1, 1)
        center = _cast_direct_state(self.latents_mean, self._compute_dtype).to(device=latent.device)
        scale = _cast_direct_state(self.latents_std, self._compute_dtype).to(device=latent.device)
        center = center.view(1, -1, 1, 1, 1)
        scale = scale.view(1, -1, 1, 1, 1)
        return latent * scale + center

    def encode_output_shape(self, input_shape: torch.Size | tuple[int, ...]) -> tuple[int, ...]:
        if len(input_shape) != 5:
            raise ValueError("MiniMax H3 content must have rank 5 [B,C,T,H,W]")
        batch, _, frames, height, width = input_shape
        if min(batch, frames, height, width) < 1:
            raise ValueError("MiniMax H3 content extents must be positive")
        encoded_frames = (
            1
            if frames == 1
            else math.ceil(frames / self.clip_length) * self.tokens_chunk_size - self.token_drop
        )
        return (
            batch,
            self.config.embed_dim,
            encoded_frames,
            math.ceil(height / self.vae_ratio),
            math.ceil(width / self.vae_ratio),
        )

    def decode_output_shape(self, input_shape: torch.Size | tuple[int, ...]) -> tuple[int, ...]:
        if len(input_shape) != 5:
            raise ValueError("MiniMax H3 latent must have rank 5 [B,C,T,H,W]")
        batch, _, frames, height, width = input_shape
        if frames == 1:
            output_frames = 1
        else:
            pad_tokens, chunks = self._decode_temporal_chunks(frames)
            output_frames = self._decode_temporal_frame_plan(
                frames + pad_tokens, chunks, pad_tokens
            )
        return (
            batch,
            self.config.out_channels,
            output_frames,
            height * self.vae_ratio,
            width * self.vae_ratio,
        )

    def _adaptive_encode(self, content: torch.Tensor) -> torch.Tensor:
        return self.tiled_encode(content) if self.tiling else self._encode_moments(content)

    def _adaptive_decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.tiled_decode(latent) if self.tiling else self._decode_pixels(latent)

    def split_tiles(self, input_len: int) -> tuple[list[int], list[int], list[int]]:
        if self.tile_size >= input_len:
            return [0], [input_len], []
        count = math.ceil(input_len / self.tile_size)
        while True:
            overlaps = [self.tile_overlap_min] * (count - 1)
            remaining = self.tile_size * count - sum(overlaps) - input_len
            if remaining >= 0:
                break
            count += 1
        for index in range(remaining // self.vae_ratio):
            overlaps[index % (count - 1)] += self.vae_ratio
        starts = [0]
        for index in range(count - 1):
            starts.append(starts[-1] + self.tile_size - overlaps[index])
        return starts, [self.tile_size] * count, overlaps

    @staticmethod
    def blend(a: torch.Tensor, b: torch.Tensor, extent: int, dim: int) -> torch.Tensor:
        extent = min(a.shape[dim], b.shape[dim], extent)
        if extent == 0:
            return b
        positions = torch.arange(extent, device=b.device, dtype=b.dtype)
        shape = [1] * a.ndim
        shape[dim] = extent
        weight_a = (1 - positions / extent).view(shape)
        weight_b = (positions / extent).view(shape)
        slice_a = [slice(None)] * a.ndim
        slice_a[dim] = slice(-extent, None)
        slice_b = [slice(None)] * b.ndim
        slice_b[dim] = slice(0, extent)
        blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b
        if extent == b.shape[dim]:
            return blended
        slice_b[dim] = slice(extent, None)
        return torch.cat((blended, b[tuple(slice_b)]), dim=dim)

    def tiled_encode(self, content: torch.Tensor) -> torch.Tensor:
        y_starts, y_lengths, y_overlaps = self.split_tiles(content.shape[-2])
        x_starts, x_lengths, x_overlaps = self.split_tiles(content.shape[-1])
        rows = [
            [
                self._encode_moments(content[..., y : y + y_length, x : x + x_length])
                for x, x_length in zip(x_starts, x_lengths, strict=True)
            ]
            for y, y_length in zip(y_starts, y_lengths, strict=True)
        ]
        latent_y = [overlap // self.vae_ratio for overlap in y_overlaps]
        latent_x = [overlap // self.vae_ratio for overlap in x_overlaps]
        result_rows = []
        for row_index, row in enumerate(rows):
            result_row = []
            for column, tile in enumerate(row):
                if row_index > 0:
                    tile = self.blend(
                        rows[row_index - 1][column], tile, latent_y[row_index - 1], -2
                    )
                if column > 0:
                    tile = self.blend(row[column - 1], tile, latent_x[column - 1], -1)
                if row_index < len(rows) - 1:
                    tile = tile[..., : -latent_y[row_index], :]
                if column < len(row) - 1:
                    tile = tile[..., :, : -latent_x[column]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def _decode_tile_row(
        self,
        latent_row: torch.Tensor,
        x_starts: list[int],
        x_lengths: list[int],
    ) -> Generator[torch.Tensor]:
        free = get_free_memory(latent_row.device).free_total
        group_size = max(1, min(4, free // (128 * 2**20 * latent_row.shape[0])))
        slices = [
            latent_row[
                ...,
                x // self.vae_ratio : (x + x_length) // self.vae_ratio,
            ]
            for x, x_length in zip(x_starts, x_lengths, strict=True)
        ]
        for start in range(0, len(slices), group_size):
            group = slices[start : start + group_size]
            yield from self._decode_pixels(torch.cat(group)).chunk(len(group))

    def tiled_decode(self, latent: torch.Tensor) -> torch.Tensor:
        height = latent.shape[-2] * self.vae_ratio
        width = latent.shape[-1] * self.vae_ratio
        y_starts, y_lengths, y_overlaps = self.split_tiles(height)
        x_starts, x_lengths, x_overlaps = self.split_tiles(width)
        canvas: torch.Tensor | None = None
        strip: torch.Tensor | None = None
        output_y = 0
        for row_index, (y, y_length) in enumerate(zip(y_starts, y_lengths, strict=True)):
            latent_y, latent_height = y // self.vae_ratio, y_length // self.vae_ratio
            tiles = self._decode_tile_row(
                latent[..., latent_y : latent_y + latent_height, :], x_starts, x_lengths
            )
            new_strip: torch.Tensor | None = None
            left_tail: torch.Tensor | None = None
            output_x = 0
            written_height = 0
            for column in range(len(x_starts)):
                tile = next(tiles)
                if row_index > 0:
                    assert strip is not None
                    x = x_starts[column]
                    tile = self.blend(
                        strip[..., :, x : x + x_lengths[column]],
                        tile,
                        y_overlaps[row_index - 1],
                        -2,
                    )
                if column > 0:
                    assert left_tail is not None
                    tile = self.blend(left_tail, tile, x_overlaps[column - 1], -1)
                left_tail = (
                    tile[..., :, -x_overlaps[column] :].clone()
                    if column < len(x_starts) - 1
                    else None
                )
                if column < len(x_starts) - 1:
                    tile = tile[..., :, : -x_overlaps[column]]
                if canvas is None:
                    canvas = torch.empty(
                        *tile.shape[:-2], height, width, dtype=tile.dtype, device=tile.device
                    )
                if row_index < len(y_starts) - 1:
                    if new_strip is None:
                        new_strip = torch.empty(
                            *tile.shape[:-2],
                            y_overlaps[row_index],
                            width,
                            dtype=tile.dtype,
                            device=tile.device,
                        )
                    new_strip[
                        ...,
                        :,
                        output_x : output_x + tile.shape[-1],
                    ].copy_(tile[..., -y_overlaps[row_index] :, :])
                    tile = tile[..., : -y_overlaps[row_index], :]
                canvas[
                    ...,
                    output_y : output_y + tile.shape[-2],
                    output_x : output_x + tile.shape[-1],
                ].copy_(tile)
                output_x += tile.shape[-1]
                written_height = tile.shape[-2]
                del tile
            strip = new_strip
            output_y += written_height
        assert canvas is not None
        return canvas

    def _validate_content(self, content: torch.Tensor) -> torch.Tensor:
        if content.ndim == 4:
            content = content.unsqueeze(2)
        if content.ndim != 5:
            raise ValueError("MiniMax H3 content must have rank 4 or 5 [B,C,(T),H,W]")
        if content.shape[1] != self.config.in_channels:
            raise ValueError(f"MiniMax H3 content must have {self.config.in_channels} channels")
        if min(content.shape[0], content.shape[2], content.shape[3], content.shape[4]) <= 0:
            raise ValueError(
                "MiniMax H3 content batch, frame, and spatial extents must be positive"
            )
        if content.shape[-2] < 2 or content.shape[-1] < 2:
            raise ValueError("MiniMax H3 content height and width must support reflect padding")
        if not content.is_floating_point() or content.layout != torch.strided:
            raise TypeError("MiniMax H3 content must be a strided floating-point tensor")
        return content

    def _validate_latent(self, latent: torch.Tensor) -> None:
        if latent.ndim != 5:
            raise ValueError("MiniMax H3 latent must have rank 5 [B,C,T,H,W]")
        if latent.shape[1] != self.config.embed_dim:
            raise ValueError(f"MiniMax H3 latent must have {self.config.embed_dim} channels")
        if min(latent.shape[0], latent.shape[2], latent.shape[3], latent.shape[4]) <= 0:
            raise ValueError("MiniMax H3 latent batch, frame, and spatial extents must be positive")
        if not latent.is_floating_point() or latent.layout != torch.strided:
            raise TypeError("MiniMax H3 latent must be a strided floating-point tensor")

    def _validate_output_buffer(self, buffer: torch.Tensor, shape: tuple[int, ...]) -> None:
        if tuple(buffer.shape) != shape:
            raise ValueError(f"output_buffer shape must be {shape}, got {tuple(buffer.shape)}")
        if buffer.dtype != torch.float32:
            raise TypeError("output_buffer must have float32 dtype")
        if buffer.layout != torch.strided:
            raise ValueError("output_buffer must use strided layout")
        if buffer.device.type == "meta":
            raise ValueError("output_buffer must have real storage")
        if not buffer.is_contiguous():
            raise ValueError("output_buffer must be contiguous")

    def encode_temporal(self, content: torch.Tensor, device: torch.device) -> torch.Tensor:
        parts = []
        for start in range(0, content.shape[2], self.clip_length):
            clip = content[:, :, start : start + self.clip_length].to(device)
            if clip.shape[2] < self.clip_length:
                clip = torch.cat(
                    (clip, clip[:, :, -1:].repeat(1, 1, self.clip_length - clip.shape[2], 1, 1)),
                    dim=2,
                )
            try:
                parts.append(self._adaptive_encode(self._normalize_pixels(clip)))
            finally:
                del clip
        latent = torch.cat(parts, dim=2)
        if self.token_drop:
            latent = latent[:, :, : -self.token_drop]
        return latent

    def _decode_temporal_pad_frames(self, latent_length: int, pad_tokens: int) -> int:
        if pad_tokens <= 0:
            return 0
        intra_tail = self.clip_length % self.vae_ratio_t
        if intra_tail == 0:
            return pad_tokens * self.vae_ratio_t
        before_pad = latent_length - pad_tokens
        return sum(
            intra_tail if (before_pad + offset) % self.tokens_chunk_size == 0 else self.vae_ratio_t
            for offset in range(pad_tokens)
        )

    def _decode_temporal_frame_plan(self, latent_length: int, chunks: int, pad_tokens: int) -> int:
        chunk_frames = self.tokens_chunk_size * self.vae_ratio_t
        splits = int(self.token_drop > 0) + 1
        total = 0
        final_overlap = 0
        for index in range(chunks):
            start = index * self.tokens_chunk_size
            end = start + self.tokens_chunk_size + self.token_overlap
            tokens = max(0, min(end, latent_length) - min(start, latent_length))
            decoded_frames = tokens * self.vae_ratio_t
            for split in range(splits):
                frame_start = split * chunk_frames
                frame_end = min(frame_start + chunk_frames, decoded_frames)
                count = max(0, frame_end - frame_start - self.frame_pre_padding)
                if split == 0:
                    total += count
                else:
                    final_overlap = count
        return total + final_overlap - self._decode_temporal_pad_frames(latent_length, pad_tokens)

    def _decode_temporal_chunks(self, latent_length: int) -> tuple[int, int]:
        pseudo_tokens = latent_length + self.token_drop
        pad_tokens = (-pseudo_tokens) % self.tokens_chunk_size
        pseudo_tokens += pad_tokens
        chunks = pseudo_tokens // self.tokens_chunk_size - int(self.token_drop > 0)
        if chunks < 1:
            pad_tokens += self.tokens_chunk_size
            chunks += 1
        return pad_tokens, chunks

    def decode_temporal(
        self, latent: torch.Tensor, output_buffer: torch.Tensor | None = None
    ) -> torch.Tensor:
        shape = self.decode_output_shape(latent.shape)
        if output_buffer is None:
            output_buffer = torch.empty(shape, dtype=torch.float32, device="cpu")
        self._validate_output_buffer(output_buffer, shape)
        pad_tokens, chunks = self._decode_temporal_chunks(latent.shape[2])
        if pad_tokens:
            latent = torch.cat((latent, latent[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)), dim=2)
        chunk_frames = self.tokens_chunk_size * self.vae_ratio_t
        splits = int(self.token_drop > 0) + 1
        overlap: torch.Tensor | None = None
        write_position = 0

        def write_part(part: torch.Tensor) -> None:
            nonlocal write_position
            if part.shape[2] <= 0:
                return
            finalized = self._finalize_pixels(part)
            count = min(finalized.shape[2], max(0, output_buffer.shape[2] - write_position))
            if count:
                output_buffer[:, :, write_position : write_position + count].copy_(
                    finalized[:, :, :count]
                )
                write_position += count

        try:
            for index in range(chunks):
                start = index * self.tokens_chunk_size
                end = start + self.tokens_chunk_size + self.token_overlap
                clip_latent = latent[:, :, start:end]
                clip_content: torch.Tensor | None = None
                try:
                    clip_content = self._adaptive_decode(clip_latent)
                    for split in range(splits):
                        frame_start = split * chunk_frames
                        frame_end = min(frame_start + chunk_frames, clip_content.shape[2])
                        part = clip_content[:, :, frame_start:frame_end]
                        part = part[:, :, self.frame_pre_padding :]
                        if split == 0:
                            if overlap is not None:
                                part = self.blend(overlap, part, self.frame_overlap, -3)
                                overlap = None
                            write_part(part)
                        else:
                            overlap = part.contiguous()
                    if index == chunks - 1 and overlap is not None:
                        write_part(overlap)
                        overlap = None
                finally:
                    del clip_content, clip_latent
        finally:
            overlap = None
        if write_position != output_buffer.shape[2]:
            raise RuntimeError(
                f"decoded {write_position} frames for a buffer expecting {output_buffer.shape[2]}"
            )
        return output_buffer

    def encode(
        self, content: torch.Tensor, device: torch.device | str | None = None
    ) -> torch.Tensor:
        content = self._validate_content(content)
        target = content.device if device is None else torch.device(device)
        if content.shape[2] == 1:
            moments = self._adaptive_encode(self._normalize_pixels(content.to(target)))
            moments = moments[:, :, -1:]
        else:
            moments = self.encode_temporal(content, target)
        mean = moments.float().chunk(2, dim=1)[0]
        return self._normalize_latents(mean)

    def decode(
        self, latent: torch.Tensor, output_buffer: torch.Tensor | None = None
    ) -> torch.Tensor:
        self._validate_latent(latent)
        shape = self.decode_output_shape(latent.shape)
        if output_buffer is not None:
            self._validate_output_buffer(output_buffer, shape)
        latent = self._denormalize_latents(latent)
        if latent.shape[2] != 1:
            return self.decode_temporal(latent, output_buffer)
        content: torch.Tensor | None = None
        try:
            content = self._finalize_pixels(self._adaptive_decode(latent)[:, :, -1:])
            if output_buffer is None:
                return content
            output_buffer.copy_(content)
            return output_buffer
        finally:
            del content
