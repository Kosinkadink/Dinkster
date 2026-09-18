"""Native causal Wan 2.1 video VAE.

The architecture, state names, causal cache plan, and chunked temporal paths
follow ``comfy/ldm/wan/vae.py`` at ComfyUI
``2a68ce33b4c9ea6ee4283e618a74560cefb32694``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import torch
import torch.nn.functional as F

from .attention import AttentionKernel, select_attention
from .operations import (
    INITLESS,
    CastOperations,
    Operations,
    ResidencyRouted,
    bound_compute_dtype,
)
from .ops import cast_weight

__all__ = [
    "AttentionBlock",
    "CausalConv3d",
    "Decoder3d",
    "Encoder3d",
    "LATENTS_MEAN",
    "LATENTS_STD",
    "RMSNorm",
    "Resample",
    "ResidualBlock",
    "WanVAE",
    "WanVAEConfig",
    "count_cache_layers",
]

CACHE_T = 2
LATENTS_MEAN = (
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
)
LATENTS_STD = (
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
)

_DEFAULT_VAE_ATTENTION = select_attention("vae").kernel
_FIRST_UPSAMPLE = object()

_ResampleMode = Literal["none", "upsample2d", "upsample3d", "downsample2d", "downsample3d"]
_FeatureCache = list[torch.Tensor | object | None]


def _operations_compute_dtype(operations: Operations) -> torch.dtype | None:
    return operations.dtype if isinstance(operations, CastOperations) else None


def _cast_direct_state(stored: torch.Tensor, dtype: torch.dtype | None) -> torch.Tensor:
    return stored if dtype is None else cast_weight(stored, dtype=dtype)


@dataclass(frozen=True, slots=True)
class WanVAEConfig:
    """Base Wan21 production geometry with reduced-size test knobs."""

    dim: int = 96
    z_dim: int = 16
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attn_scales: tuple[float, ...] = ()
    temporal_downsample: tuple[bool, ...] = (False, True, True)
    image_channels: int = 3
    conv_out_channels: int = 3
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.dim <= 0 or self.z_dim <= 0 or self.num_res_blocks <= 0:
            raise ValueError("dim, z_dim, and num_res_blocks must be positive")
        if len(self.dim_mult) < 2 or any(multiplier <= 0 for multiplier in self.dim_mult):
            raise ValueError("dim_mult must contain at least two positive entries")
        if len(self.temporal_downsample) != len(self.dim_mult) - 1:
            raise ValueError("temporal_downsample must have one entry per resample level")
        if sum(self.temporal_downsample) != 2:
            raise ValueError("Wan21 requires exactly two temporal downsample levels")
        if (self.image_channels, self.conv_out_channels) not in ((3, 3), (3, 1)):
            raise ValueError("Wan21 content channels must be RGB-to-RGB or RGB-to-mask")
        if not 0.0 <= self.dropout < 1.0 or not math.isfinite(self.dropout):
            raise ValueError("dropout must be finite in [0,1)")
        if any(not math.isfinite(scale) or scale <= 0.0 for scale in self.attn_scales):
            raise ValueError("attention scales must be finite and positive")

    @property
    def spatial_ratio(self) -> int:
        return 2 ** (len(self.dim_mult) - 1)


class CausalConv3d(ResidencyRouted, torch.nn.Conv3d):
    """Front-padded temporal convolution with the one-frame fast path."""

    _padding: int

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        *,
        bias: bool = True,
        operations: Operations = INITLESS,
    ) -> None:
        layer = operations.conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
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
        conv_padding = cast(tuple[int, int, int], self.padding)
        self._padding = 2 * conv_padding[0]
        self.padding = (0, conv_padding[1], conv_padding[2])

    def reset_parameters(self) -> None:
        return None

    def _causal_forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        cache_x: torch.Tensor | None = None,
        cache_list: _FeatureCache | None = None,
        cache_idx: int | None = None,
    ) -> torch.Tensor:
        x = input
        if cache_list is not None:
            if cache_idx is None:
                raise ValueError("cache_idx is required with cache_list")
            cached = cache_list[cache_idx]
            if cached is not None and not isinstance(cached, torch.Tensor):
                raise RuntimeError("causal convolution cache contains an invalid marker")
            cache_x = cached
            cache_list[cache_idx] = None
            del cached

        if cache_x is None and x.shape[2] == 1:
            return F.conv3d(
                x,
                weight[:, :, -1:],
                bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

        if self._padding > 0:
            padding_needed = self._padding
            parts: list[torch.Tensor] = []
            if cache_x is not None:
                cache_x = cache_x.to(device=x.device, dtype=x.dtype)
                padding_needed = max(0, padding_needed - cache_x.shape[2])
            if padding_needed:
                padding_shape = list(x.shape)
                padding_shape[2] = padding_needed
                parts.append(torch.zeros(padding_shape, device=x.device, dtype=x.dtype))
            if cache_x is not None:
                parts.append(cache_x)
            parts.append(x)
            x = torch.cat(parts, dim=2)
            del parts
        del cache_x
        return self._conv_forward(x, weight, bias)

    def _prefetch_dtype(self, stored: torch.Tensor) -> torch.dtype:
        return stored.dtype if self._compute_dtype is None else self._compute_dtype

    def forward(
        self,
        input: torch.Tensor,
        cache_x: torch.Tensor | None = None,
        cache_list: _FeatureCache | None = None,
        cache_idx: int | None = None,
    ) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is None:
            return self._causal_forward(
                input,
                _cast_direct_state(self.weight, self._compute_dtype),
                None if self.bias is None else _cast_direct_state(self.bias, self._compute_dtype),
                cache_x,
                cache_list,
                cache_idx,
            )
        with binding.lease() as lease:
            dtype = self.weight.dtype if self._compute_dtype is None else self._compute_dtype
            bias = None if self.bias is None else lease.get("bias", dtype=dtype)
            return self._causal_forward(
                input,
                lease.get("weight", dtype=dtype),
                bias,
                cache_x,
                cache_list,
                cache_idx,
            )


class RMSNorm(ResidencyRouted, torch.nn.Module):
    """Wan channel-axis L2 normalization with checkpoint gamma/bias state."""

    gamma: torch.nn.Parameter
    bias: torch.nn.Parameter | None

    def __init__(
        self,
        dim: int,
        *,
        channel_first: bool = True,
        images: bool = True,
        bias: bool = False,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        broadcastable_dims = (1, 1) if images else (1, 1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = torch.nn.Parameter(torch.empty(shape))
        self.bias = torch.nn.Parameter(torch.empty(shape)) if bias else None
        self._compute_dtype = _operations_compute_dtype(operations)

    def _prefetch_dtype(self, stored: torch.Tensor) -> torch.dtype:
        return stored.dtype if self._compute_dtype is None else self._compute_dtype

    def _normalize(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        result = F.normalize(x, dim=1 if self.channel_first else -1) * self.scale
        # Reuse the scaled buffer instead of allocating another full activation.
        result.mul_(gamma)
        return result if bias is None else result.add_(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is None:
            return self._normalize(
                x,
                _cast_direct_state(self.gamma, self._compute_dtype),
                None if self.bias is None else _cast_direct_state(self.bias, self._compute_dtype),
            )
        with binding.lease() as lease:
            dtype = self.gamma.dtype if self._compute_dtype is None else self._compute_dtype
            bias = None if self.bias is None else lease.get("bias", dtype=dtype)
            return self._normalize(x, lease.get("gamma", dtype=dtype), bias)


class Resample(torch.nn.Module):
    """Wan spatial resampling with optional causal temporal conversion."""

    def __init__(
        self,
        dim: int,
        mode: _ResampleMode,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode
        if mode in ("upsample2d", "upsample3d"):
            self.resample = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                operations.conv2d(dim, dim // 2, 3, padding=1),
            )
        elif mode in ("downsample2d", "downsample3d"):
            self.resample = torch.nn.Sequential(
                torch.nn.ZeroPad2d((0, 1, 0, 1)),
                operations.conv2d(dim, dim, 3, stride=2),
            )
        elif mode == "none":
            self.resample = torch.nn.Identity()
        else:
            raise ValueError(f"unsupported Wan21 resample mode {mode!r}")

        if mode == "upsample3d":
            self.time_conv = CausalConv3d(
                dim,
                dim * 2,
                (3, 1, 1),
                padding=(1, 0, 0),
                operations=operations,
            )
        elif mode == "downsample3d":
            self.time_conv = CausalConv3d(
                dim,
                dim,
                (3, 1, 1),
                stride=(2, 1, 1),
                padding=0,
                operations=operations,
            )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: _FeatureCache | None = None,
        feat_idx: list[int] | None = None,
        *,
        final: bool = False,
    ) -> torch.Tensor | None:
        if feat_idx is None:
            feat_idx = [0]
        batch, channels, frames, height, width = x.shape
        if self.mode == "upsample3d" and feat_cache is not None:
            index = feat_idx[0]
            if feat_cache[index] is None:
                feat_cache[index] = _FIRST_UPSAMPLE
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -CACHE_T:]
                cached = feat_cache[index]
                if cached is _FIRST_UPSAMPLE:
                    x = self.time_conv(x)
                elif isinstance(cached, torch.Tensor):
                    x = self.time_conv(x, cached)
                else:
                    raise RuntimeError("upsample cache contains an invalid value")
                feat_cache[index] = cache_x
                feat_idx[0] += 1
                x = x.reshape(batch, 2, channels, frames, height, width)
                x = torch.stack((x[:, 0], x[:, 1]), dim=3)
                x = x.reshape(batch, channels, frames * 2, height, width)

        frames = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, x.shape[1], height, width)
        x = self.resample(x)
        x = x.reshape(batch, frames, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)

        if self.mode == "downsample3d" and feat_cache is not None:
            index = feat_idx[0]
            cached = feat_cache[index]
            if cached is None:
                feat_cache[index] = x
            elif isinstance(cached, torch.Tensor):
                cache_x = x[:, :, -1:]
                x = self.time_conv(torch.cat((cached[:, :, -1:], x), dim=2))
                feat_cache[index] = cache_x
                deferred = feat_cache[index + 1]
                if deferred is not None:
                    if not isinstance(deferred, torch.Tensor):
                        raise RuntimeError("downsample cache contains an invalid value")
                    x = torch.cat((deferred, x), dim=2)
                    feat_cache[index + 1] = None
                if x.shape[2] == 1 and not final:
                    feat_cache[index + 1] = x
                    feat_idx[0] += 2
                    return None
            else:
                raise RuntimeError("downsample cache contains an invalid value")
            feat_idx[0] += 2
        return x


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual = torch.nn.Sequential(
            RMSNorm(in_dim, images=False, operations=operations),
            torch.nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1, operations=operations),
            RMSNorm(out_dim, images=False, operations=operations),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1, operations=operations),
        )
        self.shortcut = (
            CausalConv3d(in_dim, out_dim, 1, operations=operations)
            if in_dim != out_dim
            else torch.nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: _FeatureCache | None = None,
        feat_idx: list[int] | None = None,
        *,
        final: bool = False,
    ) -> torch.Tensor:
        del final
        if feat_idx is None:
            feat_idx = [0]
        old_x = x
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                index = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:]
                x = layer(x, cache_list=feat_cache, cache_idx=index)
                feat_cache[index] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + self.shortcut(old_x)


class AttentionBlock(torch.nn.Module):
    """Per-frame single-head VAE attention through the injected seam."""

    _attention_kernel: AttentionKernel

    def __init__(
        self,
        dim: int,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_VAE_ATTENTION,
    ) -> None:
        super().__init__()
        self.dim = dim
        object.__setattr__(self, "_attention_kernel", attention_kernel)
        self.norm = RMSNorm(dim, operations=operations)
        self.to_qkv = operations.conv2d(dim, dim * 3, 1)
        self.proj = operations.conv2d(dim, dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: _FeatureCache | None = None,
        feat_idx: list[int] | None = None,
        *,
        final: bool = False,
    ) -> torch.Tensor:
        del feat_cache, feat_idx, final
        identity = x
        batch, channels, frames, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        shape = q.shape
        q, k, v = (
            tensor.reshape(batch * frames, 1, channels, -1).transpose(2, 3).contiguous()
            for tensor in (q, k, v)
        )
        x = self._attention_kernel(q, k, v)
        x = x.transpose(2, 3).reshape(shape)
        x = self.proj(x)
        x = x.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)
        return x + identity


def _run_layer(
    layer: torch.nn.Module,
    x: torch.Tensor,
    feat_cache: _FeatureCache | None,
    feat_idx: list[int],
    *,
    final: bool = False,
) -> torch.Tensor | None:
    if feat_cache is None:
        return cast(torch.Tensor, layer(x))
    if isinstance(layer, (ResidualBlock, AttentionBlock, Resample)):
        return layer(x, feat_cache, feat_idx, final=final)
    return cast(torch.Tensor, layer(x))


class Encoder3d(torch.nn.Module):
    def __init__(
        self,
        config: WanVAEConfig,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_VAE_ATTENTION,
    ) -> None:
        super().__init__()
        self.config = config
        dims = [config.dim * multiplier for multiplier in (1, *config.dim_mult)]
        scale = 1.0
        self.conv1 = CausalConv3d(
            config.image_channels, dims[0], 3, padding=1, operations=operations
        )
        downsamples: list[torch.nn.Module] = []
        out_dim = dims[0]
        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            for _ in range(config.num_res_blocks):
                downsamples.append(
                    ResidualBlock(in_dim, out_dim, config.dropout, operations=operations)
                )
                if scale in config.attn_scales:
                    downsamples.append(
                        AttentionBlock(
                            out_dim,
                            operations=operations,
                            attention_kernel=attention_kernel,
                        )
                    )
                in_dim = out_dim
            if index != len(config.dim_mult) - 1:
                mode: _ResampleMode = (
                    "downsample3d" if config.temporal_downsample[index] else "downsample2d"
                )
                downsamples.append(Resample(out_dim, mode, operations=operations))
                scale /= 2.0
        self.downsamples = torch.nn.Sequential(*downsamples)
        self.middle = torch.nn.Sequential(
            ResidualBlock(out_dim, out_dim, config.dropout, operations=operations),
            AttentionBlock(
                out_dim,
                operations=operations,
                attention_kernel=attention_kernel,
            ),
            ResidualBlock(out_dim, out_dim, config.dropout, operations=operations),
        )
        self.head = torch.nn.Sequential(
            RMSNorm(out_dim, images=False, operations=operations),
            torch.nn.SiLU(),
            CausalConv3d(
                out_dim,
                config.z_dim * 2,
                3,
                padding=1,
                operations=operations,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: _FeatureCache | None = None,
        feat_idx: list[int] | None = None,
        *,
        final: bool = False,
    ) -> torch.Tensor | None:
        if feat_idx is None:
            feat_idx = [0]
        if feat_cache is not None:
            index = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:]
            cached = feat_cache[index]
            if cached is not None and not isinstance(cached, torch.Tensor):
                raise RuntimeError("encoder cache contains an invalid value")
            x = self.conv1(x, cached)
            feat_cache[index] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.downsamples:
            result = _run_layer(layer, x, feat_cache, feat_idx, final=final)
            if result is None:
                return None
            x = result
        for layer in self.middle:
            result = _run_layer(layer, x, feat_cache, feat_idx, final=final)
            assert result is not None
            x = result
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                index = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:]
                cached = feat_cache[index]
                if cached is not None and not isinstance(cached, torch.Tensor):
                    raise RuntimeError("encoder head cache contains an invalid value")
                x = layer(x, cached)
                feat_cache[index] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Decoder3d(torch.nn.Module):
    def __init__(
        self,
        config: WanVAEConfig,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_VAE_ATTENTION,
    ) -> None:
        super().__init__()
        dims = [
            config.dim * multiplier
            for multiplier in (config.dim_mult[-1], *reversed(config.dim_mult))
        ]
        scale = 1.0 / 2 ** (len(config.dim_mult) - 2)
        self.conv1 = CausalConv3d(config.z_dim, dims[0], 3, padding=1, operations=operations)
        self.middle = torch.nn.Sequential(
            ResidualBlock(dims[0], dims[0], config.dropout, operations=operations),
            AttentionBlock(
                dims[0],
                operations=operations,
                attention_kernel=attention_kernel,
            ),
            ResidualBlock(dims[0], dims[0], config.dropout, operations=operations),
        )
        upsamples: list[torch.nn.Module] = []
        out_dim = dims[-1]
        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            if index in (1, 2, 3):
                in_dim //= 2
            for _ in range(config.num_res_blocks + 1):
                upsamples.append(
                    ResidualBlock(in_dim, out_dim, config.dropout, operations=operations)
                )
                if scale in config.attn_scales:
                    upsamples.append(
                        AttentionBlock(
                            out_dim,
                            operations=operations,
                            attention_kernel=attention_kernel,
                        )
                    )
                in_dim = out_dim
            if index != len(config.dim_mult) - 1:
                mode: _ResampleMode = (
                    "upsample3d"
                    if tuple(reversed(config.temporal_downsample))[index]
                    else "upsample2d"
                )
                upsamples.append(Resample(out_dim, mode, operations=operations))
                scale *= 2.0
        self.upsamples = torch.nn.Sequential(*upsamples)
        self.head = torch.nn.Sequential(
            RMSNorm(out_dim, images=False, operations=operations),
            torch.nn.SiLU(),
            CausalConv3d(
                out_dim,
                config.conv_out_channels,
                3,
                padding=1,
                operations=operations,
            ),
        )

    def _run_up(
        self,
        layer_index: int,
        x_ref: list[torch.Tensor | None],
        feat_cache: _FeatureCache | None,
        feat_idx: list[int],
        out_chunks: list[torch.Tensor],
    ) -> None:
        x = x_ref[0]
        x_ref[0] = None
        if x is None:
            raise RuntimeError("decoder recursion received an empty tensor holder")
        if layer_index >= len(self.upsamples):
            for layer in self.head:
                if isinstance(layer, CausalConv3d) and feat_cache is not None:
                    index = feat_idx[0]
                    cache_x = x[:, :, -CACHE_T:]
                    cached = feat_cache[index]
                    if cached is not None and not isinstance(cached, torch.Tensor):
                        raise RuntimeError("decoder head cache contains an invalid value")
                    x = layer(x, cached)
                    feat_cache[index] = cache_x
                    feat_idx[0] += 1
                else:
                    x = layer(x)
            out_chunks.append(x)
            return

        layer = self.upsamples[layer_index]
        result = _run_layer(layer, x, feat_cache, feat_idx)
        if result is None:
            raise RuntimeError("decoder upsample unexpectedly deferred output")
        x = result
        del result
        if isinstance(layer, Resample) and layer.mode == "upsample3d" and x.shape[2] > 2:
            for frame_index in range(0, x.shape[2], 2):
                self._run_up(
                    layer_index + 1,
                    [x[:, :, frame_index : frame_index + 2]],
                    feat_cache,
                    feat_idx.copy(),
                    out_chunks,
                )
            return
        next_x_ref: list[torch.Tensor | None] = [x]
        del x
        self._run_up(layer_index + 1, next_x_ref, feat_cache, feat_idx, out_chunks)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: _FeatureCache | None = None,
        feat_idx: list[int] | None = None,
    ) -> list[torch.Tensor]:
        if feat_idx is None:
            feat_idx = [0]
        if feat_cache is not None:
            index = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:]
            cached = feat_cache[index]
            if cached is not None and not isinstance(cached, torch.Tensor):
                raise RuntimeError("decoder cache contains an invalid value")
            x = self.conv1(x, cached)
            feat_cache[index] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)
        for layer in self.middle:
            result = _run_layer(layer, x, feat_cache, feat_idx)
            assert result is not None
            x = result
        out_chunks: list[torch.Tensor] = []
        self._run_up(0, [x], feat_cache, feat_idx, out_chunks)
        return out_chunks


def count_cache_layers(model: torch.nn.Module) -> int:
    return sum(
        isinstance(module, CausalConv3d)
        or (isinstance(module, Resample) and module.mode == "downsample3d")
        for module in model.modules()
    )


class WanVAE(torch.nn.Module):
    """Wan21 VAE source with strict geometry and call-scoped caches."""

    _dinkster_residency_constant_buffers = frozenset({"latents_mean", "latents_std"})
    latents_mean: torch.Tensor
    latents_std: torch.Tensor

    def __init__(
        self,
        config: WanVAEConfig | None = None,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_VAE_ATTENTION,
    ) -> None:
        super().__init__()
        self.config = WanVAEConfig() if config is None else config
        self.encoder = Encoder3d(
            self.config,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.conv1 = CausalConv3d(
            self.config.z_dim * 2,
            self.config.z_dim * 2,
            1,
            operations=operations,
        )
        self.conv2 = CausalConv3d(
            self.config.z_dim,
            self.config.z_dim,
            1,
            operations=operations,
        )
        self.decoder = Decoder3d(
            self.config,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.register_buffer(
            "latents_mean", torch.tensor(LATENTS_MEAN, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "latents_std", torch.tensor(LATENTS_STD, dtype=torch.float32), persistent=False
        )

    def compute_dtype(self) -> torch.dtype:
        return bound_compute_dtype(self.conv1) or self.conv1.weight.dtype

    def _validate_content(self, content: torch.Tensor) -> None:
        if content.ndim != 5:
            raise ValueError("Wan21 content must have rank 5 [B,C,T,H,W]")
        if content.shape[1] != self.config.image_channels:
            raise ValueError(f"Wan21 content must have {self.config.image_channels} channels")
        if any(size <= 0 for size in content.shape):
            raise ValueError("Wan21 content dimensions must be positive")
        if (
            content.shape[3] % self.config.spatial_ratio
            or content.shape[4] % self.config.spatial_ratio
        ):
            raise ValueError(
                f"Wan21 content height and width must be divisible by {self.config.spatial_ratio}"
            )

    def _validate_latent(self, latent: torch.Tensor) -> None:
        if latent.ndim != 5:
            raise ValueError("Wan21 latent must have rank 5 [B,C,T,H,W]")
        if latent.shape[1] != self.config.z_dim:
            raise ValueError(f"Wan21 latent must have {self.config.z_dim} channels")
        if any(size <= 0 for size in latent.shape):
            raise ValueError("Wan21 latent dimensions must be positive")

    def process_in(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 5 or latent.shape[1] != len(LATENTS_MEAN):
            raise ValueError("Wan21 normalization requires rank-5 16-channel latents")
        mean = self.latents_mean.view(1, -1, 1, 1, 1).to(latent)
        std = self.latents_std.view(1, -1, 1, 1, 1).to(latent)
        return (latent - mean) / std

    def process_out(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 5 or latent.shape[1] != len(LATENTS_MEAN):
            raise ValueError("Wan21 normalization requires rank-5 16-channel latents")
        mean = self.latents_mean.view(1, -1, 1, 1, 1).to(latent)
        std = self.latents_std.view(1, -1, 1, 1, 1).to(latent)
        return latent * std + mean

    def encode(self, content: torch.Tensor) -> torch.Tensor:
        self._validate_content(content)
        usable_frames = 1 + ((content.shape[2] - 1) // 4) * 4
        iterations = 1 + (usable_frames - 1) // 2
        feature_cache: _FeatureCache | None = None
        if iterations > 1:
            feature_cache = cast(_FeatureCache, [None] * count_cache_layers(self.encoder))
        output: torch.Tensor | None = None
        try:
            for index in range(iterations):
                feat_idx = [0]
                if index == 0:
                    part = self.encoder(
                        content[:, :, :1],
                        feat_cache=feature_cache,
                        feat_idx=feat_idx,
                    )
                else:
                    part = self.encoder(
                        content[:, :, 1 + 2 * (index - 1) : 1 + 2 * index],
                        feat_cache=feature_cache,
                        feat_idx=feat_idx,
                        final=index == iterations - 1,
                    )
                if part is None:
                    continue
                output = part if output is None else torch.cat((output, part), dim=2)
            if output is None:
                raise RuntimeError("Wan21 encoder produced no latent frames")
            mean, _log_variance = self.conv1(output).chunk(2, dim=1)
            return mean
        finally:
            if feature_cache is not None:
                feature_cache.clear()

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        self._validate_latent(latent)
        iterations = 1 + latent.shape[2] // 2
        feature_cache: _FeatureCache | None = None
        if iterations > 1:
            feature_cache = cast(_FeatureCache, [None] * count_cache_layers(self.decoder))
        output_chunks: list[torch.Tensor] = []
        x = self.conv2(latent)
        try:
            for index in range(iterations):
                feat_idx = [0]
                if index == 0:
                    part = self.decoder(
                        x[:, :, :1],
                        feat_cache=feature_cache,
                        feat_idx=feat_idx,
                    )
                else:
                    part = self.decoder(
                        x[:, :, 1 + 2 * (index - 1) : 1 + 2 * index],
                        feat_cache=feature_cache,
                        feat_idx=feat_idx,
                    )
                output_chunks.extend(part)
            if not output_chunks:
                raise RuntimeError("Wan21 decoder produced no content frames")
            return torch.cat(output_chunks, dim=2)
        finally:
            output_chunks.clear()
            if feature_cache is not None:
                feature_cache.clear()
