"""Native Wan 2.2 16x spatial causal video VAE.

The architecture, official state names, patch layout, and temporal cache plan
follow ``comfy/ldm/wan/vae2_2.py`` at ComfyUI commit ``b78cec87``.
"""

from __future__ import annotations

from typing import Literal, cast

import torch
import torch.nn.functional as F
from dinkster_inference.wan22_vae import WAN22_VAE_CONFIG, Wan22VAEConfig

from .attention import AttentionKernel, select_attention
from .operations import INITLESS, Operations
from .wan21_vae import AttentionBlock, CausalConv3d, RMSNorm

__all__ = [
    "AvgDown3D",
    "DownResidualBlock",
    "DupUp3D",
    "Encoder3d",
    "Decoder3d",
    "Resample",
    "ResidualBlock",
    "UpResidualBlock",
    "Wan22VAE",
    "Wan22VAEConfig",
    "count_conv3d",
    "patchify",
    "unpatchify",
]

CACHE_T = 2

LATENTS_MEAN = (
    -0.2289,
    -0.0052,
    -0.1323,
    -0.2339,
    -0.2799,
    0.0174,
    0.1838,
    0.1557,
    -0.1382,
    0.0542,
    0.2813,
    0.0891,
    0.1570,
    -0.0098,
    0.0375,
    -0.1825,
    -0.2246,
    -0.1207,
    -0.0698,
    0.5109,
    0.2665,
    -0.2108,
    -0.2158,
    0.2502,
    -0.2055,
    -0.0322,
    0.1109,
    0.1567,
    -0.0729,
    0.0899,
    -0.2799,
    -0.1230,
    -0.0313,
    -0.1649,
    0.0117,
    0.0723,
    -0.2839,
    -0.2083,
    -0.0520,
    0.3748,
    0.0152,
    0.1957,
    0.1433,
    -0.2944,
    0.3573,
    -0.0548,
    -0.1681,
    -0.0667,
)
LATENTS_STD = (
    0.4765,
    1.0364,
    0.4514,
    1.1677,
    0.5313,
    0.4990,
    0.4818,
    0.5013,
    0.8158,
    1.0344,
    0.5894,
    1.0901,
    0.6885,
    0.6165,
    0.8454,
    0.4978,
    0.5759,
    0.3523,
    0.7135,
    0.6804,
    0.5833,
    1.4146,
    0.8986,
    0.5659,
    0.7069,
    0.5338,
    0.4889,
    0.4917,
    0.4069,
    0.4999,
    0.6866,
    0.4093,
    0.5709,
    0.6065,
    0.6415,
    0.4944,
    0.5726,
    1.2042,
    0.5458,
    1.6887,
    0.3971,
    1.0600,
    0.3943,
    0.5537,
    0.5444,
    0.4089,
    0.7468,
    0.7744,
)

_DEFAULT_VAE_ATTENTION = select_attention("vae").kernel
_FIRST_UPSAMPLE = object()

_ResampleMode = Literal["none", "upsample2d", "upsample3d", "downsample2d", "downsample3d"]
_FeatureCache = list[torch.Tensor | object | None]


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Pack square spatial patches into channels in ComfyUI's c-r-q order."""

    if patch_size == 1:
        return x
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if x.ndim == 4:
        batch, channels, height, width = x.shape
        if height % patch_size or width % patch_size:
            raise ValueError("patchify spatial dimensions must be divisible by patch_size")
        return (
            x.reshape(
                batch,
                channels,
                height // patch_size,
                patch_size,
                width // patch_size,
                patch_size,
            )
            .permute(0, 1, 5, 3, 2, 4)
            .reshape(
                batch,
                channels * patch_size * patch_size,
                height // patch_size,
                width // patch_size,
            )
        )
    if x.ndim == 5:
        batch, channels, frames, height, width = x.shape
        if height % patch_size or width % patch_size:
            raise ValueError("patchify spatial dimensions must be divisible by patch_size")
        return (
            x.reshape(
                batch,
                channels,
                frames,
                height // patch_size,
                patch_size,
                width // patch_size,
                patch_size,
            )
            .permute(0, 1, 6, 4, 2, 3, 5)
            .reshape(
                batch,
                channels * patch_size * patch_size,
                frames,
                height // patch_size,
                width // patch_size,
            )
        )
    raise ValueError(f"invalid patchify input shape {tuple(x.shape)}")


def unpatchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Restore square spatial patches packed in ComfyUI's c-r-q order."""

    if patch_size == 1:
        return x
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    factor = patch_size * patch_size
    if x.shape[1] % factor:
        raise ValueError("unpatchify channels must be divisible by patch_size squared")
    channels = x.shape[1] // factor
    if x.ndim == 4:
        batch, _, height, width = x.shape
        return (
            x.reshape(batch, channels, patch_size, patch_size, height, width)
            .permute(0, 1, 4, 3, 5, 2)
            .reshape(batch, channels, height * patch_size, width * patch_size)
        )
    if x.ndim == 5:
        batch, _, frames, height, width = x.shape
        return (
            x.reshape(batch, channels, patch_size, patch_size, frames, height, width)
            .permute(0, 1, 4, 5, 3, 6, 2)
            .reshape(
                batch,
                channels,
                frames,
                height * patch_size,
                width * patch_size,
            )
        )
    raise ValueError(f"invalid unpatchify input shape {tuple(x.shape)}")


class AvgDown3D(torch.nn.Module):
    """Parameter-free channel-aware average downsampling shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t: int,
        factor_s: int = 1,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, factor_t, factor_s) <= 0:
            raise ValueError("AvgDown3D channels and factors must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        if in_channels * self.factor % out_channels:
            raise ValueError("AvgDown3D input groups must divide the output channels")
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (-x.shape[2]) % self.factor_t
        x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        batch, channels, frames, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(f"AvgDown3D expected {self.in_channels} channels, got {channels}")
        if height % self.factor_s or width % self.factor_s:
            raise ValueError("AvgDown3D spatial dimensions must divide its factor")
        x = x.reshape(
            batch,
            channels,
            frames // self.factor_t,
            self.factor_t,
            height // self.factor_s,
            self.factor_s,
            width // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.reshape(
            batch,
            channels * self.factor,
            frames // self.factor_t,
            height // self.factor_s,
            width // self.factor_s,
        )
        return x.reshape(
            batch,
            self.out_channels,
            self.group_size,
            frames // self.factor_t,
            height // self.factor_s,
            width // self.factor_s,
        ).mean(dim=2)


class DupUp3D(torch.nn.Module):
    """Parameter-free channel-aware duplication shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t: int,
        factor_s: int = 1,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, factor_t, factor_s) <= 0:
            raise ValueError("DupUp3D channels and factors must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        if out_channels * self.factor % in_channels:
            raise ValueError("DupUp3D output groups must divide the input channels")
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: torch.Tensor, *, first_chunk: bool = False) -> torch.Tensor:
        if x.shape[1] != self.in_channels:
            raise ValueError(f"DupUp3D expected {self.in_channels} channels, got {x.shape[1]}")
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.reshape(
            x.shape[0],
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            x.shape[2],
            x.shape[3],
            x.shape[4],
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.reshape(
            x.shape[0],
            self.out_channels,
            x.shape[2] * self.factor_t,
            x.shape[4] * self.factor_s,
            x.shape[6] * self.factor_s,
        )
        return x[:, :, self.factor_t - 1 :] if first_chunk else x


class Resample(torch.nn.Module):
    """Wan 2.2 spatial resampling with optional causal temporal conversion."""

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
                operations.conv2d(dim, dim, 3, padding=1),
            )
        elif mode in ("downsample2d", "downsample3d"):
            self.resample = torch.nn.Sequential(
                torch.nn.ZeroPad2d((0, 1, 0, 1)),
                operations.conv2d(dim, dim, 3, stride=2),
            )
        elif mode == "none":
            self.resample = torch.nn.Identity()
        else:
            raise ValueError(f"unsupported Wan 2.2 resample mode {mode!r}")

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
                operations=operations,
            )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: _FeatureCache | None = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:
        if feat_idx is None:
            feat_idx = [0]
        batch, channels, frames, height, width = x.shape
        if self.mode == "upsample3d" and feat_cache is not None:
            index = feat_idx[0]
            cached = feat_cache[index]
            if cached is None:
                feat_cache[index] = _FIRST_UPSAMPLE
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -CACHE_T:].clone()
                if cache_x.shape[2] < CACHE_T:
                    if cached is _FIRST_UPSAMPLE:
                        previous = torch.zeros_like(cache_x)
                    elif isinstance(cached, torch.Tensor):
                        previous = cached[:, :, -1:].to(device=x.device, dtype=x.dtype)
                    else:
                        raise RuntimeError("upsample cache contains an invalid value")
                    cache_x = torch.cat((previous, cache_x), dim=2)
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
                feat_cache[index] = x.clone()
            elif isinstance(cached, torch.Tensor):
                cache_x = x[:, :, -1:].clone()
                x = self.time_conv(
                    torch.cat((cached[:, :, -1:].to(device=x.device, dtype=x.dtype), x), dim=2)
                )
                feat_cache[index] = cache_x
            else:
                raise RuntimeError("downsample cache contains an invalid value")
            feat_idx[0] += 1
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
    ) -> torch.Tensor:
        if feat_idx is None:
            feat_idx = [0]
        identity = x
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                index = feat_idx[0]
                cached = feat_cache[index]
                cache_x = x[:, :, -CACHE_T:].clone()
                if cache_x.shape[2] < CACHE_T and cached is not None:
                    if not isinstance(cached, torch.Tensor):
                        raise RuntimeError("residual cache contains an invalid value")
                    cache_x = torch.cat(
                        (cached[:, :, -1:].to(device=x.device, dtype=x.dtype), cache_x), dim=2
                    )
                del cached
                x = layer(x, cache_list=feat_cache, cache_idx=index)
                feat_cache[index] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + self.shortcut(identity)


class DownResidualBlock(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
        mult: int,
        *,
        temporal_downsample: bool = False,
        down: bool = False,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temporal_downsample else 1,
            factor_s=2 if down else 1,
        )
        downsamples: list[torch.nn.Module] = []
        for _ in range(mult):
            downsamples.append(ResidualBlock(in_dim, out_dim, dropout, operations=operations))
            in_dim = out_dim
        if down:
            mode: _ResampleMode = "downsample3d" if temporal_downsample else "downsample2d"
            downsamples.append(Resample(out_dim, mode, operations=operations))
        self.downsamples = torch.nn.Sequential(*downsamples)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: _FeatureCache | None = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:
        if feat_idx is None:
            feat_idx = [0]
        identity = x
        for layer in self.downsamples:
            if isinstance(layer, (ResidualBlock, Resample)) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        return x + self.avg_shortcut(identity)


class UpResidualBlock(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
        mult: int,
        *,
        temporal_upsample: bool = False,
        up: bool = False,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.avg_shortcut = (
            DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temporal_upsample else 1,
                factor_s=2,
            )
            if up
            else None
        )
        upsamples: list[torch.nn.Module] = []
        for _ in range(mult):
            upsamples.append(ResidualBlock(in_dim, out_dim, dropout, operations=operations))
            in_dim = out_dim
        if up:
            mode: _ResampleMode = "upsample3d" if temporal_upsample else "upsample2d"
            upsamples.append(Resample(out_dim, mode, operations=operations))
        self.upsamples = torch.nn.Sequential(*upsamples)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: _FeatureCache | None = None,
        feat_idx: list[int] | None = None,
        *,
        first_chunk: bool = False,
    ) -> torch.Tensor:
        if feat_idx is None:
            feat_idx = [0]
        main = x
        for layer in self.upsamples:
            if isinstance(layer, (ResidualBlock, Resample)) and feat_cache is not None:
                main = layer(main, feat_cache, feat_idx)
            else:
                main = layer(main)
        if self.avg_shortcut is None:
            return main
        return main + self.avg_shortcut(x, first_chunk=first_chunk)


class Encoder3d(torch.nn.Module):
    def __init__(
        self,
        config: Wan22VAEConfig,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_VAE_ATTENTION,
    ) -> None:
        super().__init__()
        dims = [config.dim * multiplier for multiplier in (1, *config.dim_mult)]
        self.conv1 = CausalConv3d(
            config.image_channels * config.patch_size**2,
            dims[0],
            3,
            padding=1,
            operations=operations,
        )
        downsamples: list[torch.nn.Module] = []
        out_dim = dims[0]
        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            downsamples.append(
                DownResidualBlock(
                    in_dim,
                    out_dim,
                    config.dropout,
                    config.num_res_blocks,
                    temporal_downsample=(
                        config.temporal_downsample[index]
                        if index < len(config.temporal_downsample)
                        else False
                    ),
                    down=index != len(config.dim_mult) - 1,
                    operations=operations,
                )
            )
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
    ) -> torch.Tensor:
        if feat_idx is None:
            feat_idx = [0]
        if feat_cache is not None:
            index = feat_idx[0]
            cached = feat_cache[index]
            if cached is not None and not isinstance(cached, torch.Tensor):
                raise RuntimeError("encoder cache contains an invalid value")
            cache_x = x[:, :, -CACHE_T:].clone()
            if cache_x.shape[2] < CACHE_T and cached is not None:
                cache_x = torch.cat(
                    (cached[:, :, -1:].to(device=x.device, dtype=x.dtype), cache_x), dim=2
                )
            x = self.conv1(x, cached)
            feat_cache[index] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.downsamples:
            if feat_cache is not None:
                x = cast(DownResidualBlock, layer)(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        for layer in self.middle:
            if isinstance(layer, (ResidualBlock, AttentionBlock)) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                index = feat_idx[0]
                cached = feat_cache[index]
                if cached is not None and not isinstance(cached, torch.Tensor):
                    raise RuntimeError("encoder head cache contains an invalid value")
                cache_x = x[:, :, -CACHE_T:].clone()
                if cache_x.shape[2] < CACHE_T and cached is not None:
                    cache_x = torch.cat(
                        (cached[:, :, -1:].to(device=x.device, dtype=x.dtype), cache_x), dim=2
                    )
                x = layer(x, cached)
                feat_cache[index] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Decoder3d(torch.nn.Module):
    def __init__(
        self,
        config: Wan22VAEConfig,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_VAE_ATTENTION,
    ) -> None:
        super().__init__()
        dims = [
            config.decoder_dim * multiplier
            for multiplier in (config.dim_mult[-1], *reversed(config.dim_mult))
        ]
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
        temporal_upsample = tuple(reversed(config.temporal_downsample))
        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            upsamples.append(
                UpResidualBlock(
                    in_dim,
                    out_dim,
                    config.dropout,
                    config.num_res_blocks + 1,
                    temporal_upsample=(
                        temporal_upsample[index] if index < len(temporal_upsample) else False
                    ),
                    up=index != len(config.dim_mult) - 1,
                    operations=operations,
                )
            )
        self.upsamples = torch.nn.Sequential(*upsamples)
        self.head = torch.nn.Sequential(
            RMSNorm(out_dim, images=False, operations=operations),
            torch.nn.SiLU(),
            CausalConv3d(
                out_dim,
                config.conv_out_channels * config.patch_size**2,
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
        first_chunk: bool = False,
    ) -> torch.Tensor:
        if feat_idx is None:
            feat_idx = [0]
        if feat_cache is not None:
            index = feat_idx[0]
            cached = feat_cache[index]
            if cached is not None and not isinstance(cached, torch.Tensor):
                raise RuntimeError("decoder cache contains an invalid value")
            cache_x = x[:, :, -CACHE_T:].clone()
            if cache_x.shape[2] < CACHE_T and cached is not None:
                cache_x = torch.cat(
                    (cached[:, :, -1:].to(device=x.device, dtype=x.dtype), cache_x), dim=2
                )
            x = self.conv1(x, cached)
            feat_cache[index] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)
        for layer in self.middle:
            if isinstance(layer, (ResidualBlock, AttentionBlock)) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        for layer in self.upsamples:
            if feat_cache is not None:
                x = cast(UpResidualBlock, layer)(
                    x,
                    feat_cache,
                    feat_idx,
                    first_chunk=first_chunk,
                )
            else:
                x = layer(x)
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                index = feat_idx[0]
                cached = feat_cache[index]
                if cached is not None and not isinstance(cached, torch.Tensor):
                    raise RuntimeError("decoder head cache contains an invalid value")
                cache_x = x[:, :, -CACHE_T:].clone()
                if cache_x.shape[2] < CACHE_T and cached is not None:
                    cache_x = torch.cat(
                        (cached[:, :, -1:].to(device=x.device, dtype=x.dtype), cache_x), dim=2
                    )
                x = layer(x, cached)
                feat_cache[index] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def count_conv3d(model: torch.nn.Module) -> int:
    return sum(isinstance(module, CausalConv3d) for module in model.modules())


class Wan22VAE(torch.nn.Module):
    """Wan 2.2 VAE with strict geometry and call-scoped causal caches."""

    _dinkster_residency_constant_buffers = frozenset({"latents_mean", "latents_std"})
    latents_mean: torch.Tensor
    latents_std: torch.Tensor

    def __init__(
        self,
        config: Wan22VAEConfig = WAN22_VAE_CONFIG,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_VAE_ATTENTION,
    ) -> None:
        super().__init__()
        if not isinstance(cast("object", config), Wan22VAEConfig):
            raise TypeError("config must be Wan22VAEConfig")
        self.config = config
        self.encoder = Encoder3d(
            config,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.conv1 = CausalConv3d(
            config.z_dim * 2,
            config.z_dim * 2,
            1,
            operations=operations,
        )
        self.conv2 = CausalConv3d(
            config.z_dim,
            config.z_dim,
            1,
            operations=operations,
        )
        self.decoder = Decoder3d(
            config,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.register_buffer(
            "latents_mean", torch.tensor(LATENTS_MEAN, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "latents_std", torch.tensor(LATENTS_STD, dtype=torch.float32), persistent=False
        )

    def _validate_content(self, content: torch.Tensor) -> None:
        if content.ndim != 5:
            raise ValueError("Wan 2.2 content must have rank 5 [B,C,T,H,W]")
        if content.shape[1] != self.config.image_channels:
            raise ValueError(f"Wan 2.2 content must have {self.config.image_channels} channels")
        if any(size <= 0 for size in content.shape):
            raise ValueError("Wan 2.2 content dimensions must be positive")
        if (
            content.shape[3] % self.config.spatial_ratio
            or content.shape[4] % self.config.spatial_ratio
        ):
            raise ValueError(
                f"Wan 2.2 content height and width must be divisible by {self.config.spatial_ratio}"
            )

    def _validate_latent(self, latent: torch.Tensor) -> None:
        if latent.ndim != 5:
            raise ValueError("Wan 2.2 latent must have rank 5 [B,C,T,H,W]")
        if latent.shape[1] != self.config.z_dim:
            raise ValueError(f"Wan 2.2 latent must have {self.config.z_dim} channels")
        if any(size <= 0 for size in latent.shape):
            raise ValueError("Wan 2.2 latent dimensions must be positive")

    def process_in(self, latent: torch.Tensor) -> torch.Tensor:
        self._validate_latent(latent)
        mean = self.latents_mean.view(1, -1, 1, 1, 1).to(latent)
        std = self.latents_std.view(1, -1, 1, 1, 1).to(latent)
        return (latent - mean) / std

    def process_out(self, latent: torch.Tensor) -> torch.Tensor:
        self._validate_latent(latent)
        mean = self.latents_mean.view(1, -1, 1, 1, 1).to(latent)
        std = self.latents_std.view(1, -1, 1, 1, 1).to(latent)
        return latent * std + mean

    def encode(self, content: torch.Tensor) -> torch.Tensor:
        self._validate_content(content)
        x = patchify(content, self.config.patch_size)
        iterations = 1 + (x.shape[2] - 1) // self.config.temporal_ratio
        feature_cache = cast(_FeatureCache, [None] * count_conv3d(self.encoder))
        output: torch.Tensor | None = None
        try:
            for index in range(iterations):
                feat_idx = [0]
                if index == 0:
                    part = self.encoder(x[:, :, :1], feature_cache, feat_idx)
                else:
                    start = 1 + self.config.temporal_ratio * (index - 1)
                    stop = 1 + self.config.temporal_ratio * index
                    part = self.encoder(x[:, :, start:stop], feature_cache, feat_idx)
                output = part if output is None else torch.cat((output, part), dim=2)
            if output is None:
                raise RuntimeError("Wan 2.2 encoder produced no latent frames")
            mean, _log_variance = self.conv1(output).chunk(2, dim=1)
            return mean
        finally:
            feature_cache.clear()

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        self._validate_latent(latent)
        feature_cache = cast(_FeatureCache, [None] * count_conv3d(self.decoder))
        output: torch.Tensor | None = None
        x = self.conv2(latent)
        try:
            for index in range(x.shape[2]):
                part = self.decoder(
                    x[:, :, index : index + 1],
                    feature_cache,
                    [0],
                    first_chunk=index == 0,
                )
                output = part if output is None else torch.cat((output, part), dim=2)
            if output is None:
                raise RuntimeError("Wan 2.2 decoder produced no content frames")
            return unpatchify(output, self.config.patch_size)
        finally:
            feature_cache.clear()
