"""Unregistered MiniMax H3 DAC/BigVGAN audio codec source.

The architecture and state-dict layout follow
``comfy/ldm/minimax/audio_vae.py`` at ComfyUI
``2a68ce33b4c9ea6ee4283e618a74560cefb32694``. The default profile maps
32 kHz stereo waveforms to normalized ``[B, 32, 2, T]`` latents at 40 Hz.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import torch
import torch.nn.functional as F

from .attention import AttentionKernel, select_attention
from .operations import INITLESS, CastOperations, Operations, ResidencyRouted
from .ops import cast_weight

_DEFAULT_AUDIO_ATTENTION = select_attention("vae").kernel


def _operations_compute_dtype(operations: Operations) -> torch.dtype | None:
    return operations.dtype if isinstance(operations, CastOperations) else None


def _direct_state_dtype(compute_dtype: torch.dtype | None, input_dtype: torch.dtype) -> torch.dtype:
    return input_dtype if compute_dtype is None else compute_dtype


def _cast_direct_state(
    stored: torch.Tensor, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    return cast_weight(stored, dtype=dtype, device=device)


def snake(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Apply the SnakeBeta function without mutating ``x``."""

    periodic = torch.sin(alpha * x)
    return periodic.square() * (beta + 1e-9).reciprocal() + x


class Snake1d(ResidencyRouted, torch.nn.Module):
    _compute_dtype: torch.dtype | None

    def __init__(self, channels: int, *, operations: Operations = INITLESS) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("Snake1d channels must be positive")
        self.alpha = torch.nn.Parameter(torch.empty(1, channels, 1))
        self._compute_dtype = _operations_compute_dtype(operations)

    def _prefetch_dtype(self, stored: torch.Tensor) -> torch.dtype:
        return stored.dtype if self._compute_dtype is None else self._compute_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = _direct_state_dtype(self._compute_dtype, x.dtype)
        binding = self._offloaded_residency()
        if binding is None:
            alpha = _cast_direct_state(self.alpha, dtype=dtype, device=x.device)
            return snake(x, alpha, alpha)
        with binding.lease() as lease:
            alpha = lease.get("alpha", dtype=dtype)
            return snake(x, alpha, alpha)


class SnakeBeta(ResidencyRouted, torch.nn.Module):
    _compute_dtype: torch.dtype | None

    def __init__(self, in_features: int, *, operations: Operations = INITLESS) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError("SnakeBeta features must be positive")
        self.alpha = torch.nn.Parameter(torch.empty(in_features))
        self.beta = torch.nn.Parameter(torch.empty(in_features))
        self._compute_dtype = _operations_compute_dtype(operations)

    def _prefetch_dtype(self, stored: torch.Tensor) -> torch.dtype:
        return stored.dtype if self._compute_dtype is None else self._compute_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = _direct_state_dtype(self._compute_dtype, x.dtype)
        binding = self._offloaded_residency()
        if binding is None:
            alpha = torch.exp(_cast_direct_state(self.alpha, dtype=dtype, device=x.device))
            beta = torch.exp(_cast_direct_state(self.beta, dtype=dtype, device=x.device))
            return snake(x, alpha.view(1, -1, 1), beta.view(1, -1, 1))
        with binding.lease() as lease:
            alpha = torch.exp(lease.get("alpha", dtype=dtype))
            beta = torch.exp(lease.get("beta", dtype=dtype))
            return snake(x, alpha.view(1, -1, 1), beta.view(1, -1, 1))


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    """Build the normalized alias-free resampling filter used by BigVGAN."""

    if not 0.0 < cutoff <= 0.5:
        raise ValueError("cutoff must be in (0, 0.5]")
    if half_width <= 0.0:
        raise ValueError("half_width must be positive")
    if kernel_size < 2:
        raise ValueError("kernel_size must be at least 2")
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    attenuation = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    return filter_.div_(filter_.sum()).view(1, 1, kernel_size)


class UpSample1d(ResidencyRouted, torch.nn.Module):
    filter: torch.Tensor
    _compute_dtype: torch.dtype | None

    def __init__(
        self,
        ratio: int = 2,
        kernel_size: int = 12,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        if ratio <= 0 or kernel_size < ratio or kernel_size % ratio:
            raise ValueError("upsample kernel_size must be a positive multiple of ratio")
        self.ratio = ratio
        self.stride = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * ratio + (kernel_size - ratio) // 2
        self.pad_right = self.pad * ratio + (kernel_size - ratio + 1) // 2
        self.register_buffer(
            "filter",
            kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, kernel_size),
        )
        self._compute_dtype = _operations_compute_dtype(operations)

    def _prefetch_dtype(self, stored: torch.Tensor) -> torch.dtype:
        return stored.dtype if self._compute_dtype is None else self._compute_dtype

    def _forward_filter(self, x: torch.Tensor, filter_: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        padded = F.pad(x, (self.pad, self.pad), mode="replicate")
        expanded = filter_.expand(channels, -1, -1)
        output = F.conv_transpose1d(padded, expanded, stride=self.stride, groups=channels).mul_(
            self.ratio
        )
        return output[..., self.pad_left : -self.pad_right]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"upsample input must be [B,C,T], got {tuple(x.shape)}")
        dtype = _direct_state_dtype(self._compute_dtype, x.dtype)
        binding = self._offloaded_residency()
        if binding is None:
            filter_ = _cast_direct_state(self.filter, dtype=dtype, device=x.device)
            return self._forward_filter(x, filter_)
        with binding.lease() as lease:
            return self._forward_filter(x, lease.get("filter", dtype=dtype))


class LowPassFilter1d(ResidencyRouted, torch.nn.Module):
    filter: torch.Tensor
    _compute_dtype: torch.dtype | None

    def __init__(
        self,
        cutoff: float = 0.5,
        half_width: float = 0.6,
        stride: int = 1,
        kernel_size: int = 12,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        if stride <= 0:
            raise ValueError("low-pass stride must be positive")
        self.pad_left = kernel_size // 2 - int(kernel_size % 2 == 0)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))
        self._compute_dtype = _operations_compute_dtype(operations)

    def _prefetch_dtype(self, stored: torch.Tensor) -> torch.dtype:
        return stored.dtype if self._compute_dtype is None else self._compute_dtype

    def _forward_filter(self, x: torch.Tensor, filter_: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        padded = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        return F.conv1d(
            padded, filter_.expand(channels, -1, -1), stride=self.stride, groups=channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"low-pass input must be [B,C,T], got {tuple(x.shape)}")
        dtype = _direct_state_dtype(self._compute_dtype, x.dtype)
        binding = self._offloaded_residency()
        if binding is None:
            filter_ = _cast_direct_state(self.filter, dtype=dtype, device=x.device)
            return self._forward_filter(x, filter_)
        with binding.lease() as lease:
            return self._forward_filter(x, lease.get("filter", dtype=dtype))


class DownSample1d(torch.nn.Module):
    def __init__(
        self,
        ratio: int = 2,
        kernel_size: int = 12,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        if ratio <= 0:
            raise ValueError("downsample ratio must be positive")
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=kernel_size,
            operations=operations,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(x)


class Activation1d(torch.nn.Module):
    def __init__(
        self,
        activation: torch.nn.Module,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size, operations=operations)
        self.downsample = DownSample1d(down_ratio, down_kernel_size, operations=operations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.act(self.upsample(x)))


class ResidualUnit(torch.nn.Module):
    def __init__(
        self,
        dim: int = 16,
        dilation: int = 1,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = torch.nn.Sequential(
            Snake1d(dim, operations=operations),
            operations.conv1d(dim, dim, 7, dilation=dilation, padding=pad),
            Snake1d(dim, operations=operations),
            operations.conv1d(dim, dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block(x)
        pad = (x.shape[-1] - output.shape[-1]) // 2
        residual = x[..., pad:-pad] if pad > 0 else x
        return output.add_(residual)


class EncoderBlock(torch.nn.Module):
    def __init__(
        self,
        dim: int = 16,
        stride: int = 1,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        if dim <= 0 or dim % 2:
            raise ValueError("encoder block width must be a positive even number")
        if stride <= 0:
            raise ValueError("encoder block stride must be positive")
        width = dim // 2
        self.block = torch.nn.Sequential(
            ResidualUnit(width, dilation=1, operations=operations),
            ResidualUnit(width, dilation=3, operations=operations),
            ResidualUnit(width, dilation=9, operations=operations),
            Snake1d(width, operations=operations),
            operations.conv1d(
                width,
                dim,
                2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DACEncoder(torch.nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: Sequence[int] = (2, 4, 4, 5, 5),
        d_latent: int = 2048,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        if d_model <= 0 or d_latent <= 0 or not strides or any(rate <= 0 for rate in strides):
            raise ValueError("DAC encoder dimensions and strides must be positive")
        layers: list[torch.nn.Module] = [operations.conv1d(1, d_model, 7, padding=3)]
        width = d_model
        for stride in strides:
            width *= 2
            layers.append(EncoderBlock(width, stride=stride, operations=operations))
        layers.extend(
            (
                Snake1d(width, operations=operations),
                operations.conv1d(width, d_latent, 3, padding=1),
            )
        )
        self.block = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GeGluMlp(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.norm = operations.layer_norm(in_features)
        self.act = torch.nn.GELU(approximate="tanh")
        self.w0 = operations.linear(in_features, hidden_features)
        self.w1 = operations.linear(in_features, hidden_features)
        self.w2 = operations.linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        return self.w2(self.act(self.w0(normalized)).mul_(self.w1(normalized)))


class CausalAttention(ResidencyRouted, torch.nn.Module):
    zero_k_bias: torch.Tensor
    _attention_kernel: AttentionKernel
    _compute_dtype: torch.dtype | None

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_AUDIO_ATTENTION,
    ) -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0 or num_heads <= 0 or in_dim % num_heads:
            raise ValueError("causal attention dimensions must be positive and head-divisible")
        self.head_dim = in_dim // num_heads
        self.num_heads = num_heads
        self.out_dim = out_dim
        object.__setattr__(self, "_attention_kernel", attention_kernel)
        self.qkv = operations.linear(in_dim, in_dim * 3, bias=False)
        self.q_bias = torch.nn.Parameter(torch.empty(in_dim))
        self.v_bias = torch.nn.Parameter(torch.empty(in_dim))
        self.register_buffer("zero_k_bias", torch.empty(in_dim))
        self.proj = operations.linear(out_dim, out_dim)
        self._compute_dtype = _operations_compute_dtype(operations)

    def _prefetch_dtype(self, stored: torch.Tensor) -> torch.dtype:
        return stored.dtype if self._compute_dtype is None else self._compute_dtype

    def _forward_biases(
        self,
        x: torch.Tensor,
        q_bias: torch.Tensor,
        zero_k_bias: torch.Tensor,
        v_bias: torch.Tensor,
    ) -> torch.Tensor:
        batch, tokens, _ = x.shape
        bias = torch.cat((q_bias, zero_k_bias, v_bias)).to(device=x.device)
        qkv = self.qkv(x).add_(bias)
        query, key, value = (
            qkv.reshape(batch, tokens, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        attended = self._attention_kernel(query, key, value, causal=True)
        pooled = F.adaptive_avg_pool1d(torch.mean(attended, dim=1), self.out_dim)
        return self.proj(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = _direct_state_dtype(self._compute_dtype, x.dtype)
        binding = self._offloaded_residency()
        if binding is None:
            return self._forward_biases(
                x,
                _cast_direct_state(self.q_bias, dtype=dtype, device=x.device),
                _cast_direct_state(self.zero_k_bias, dtype=dtype, device=x.device),
                _cast_direct_state(self.v_bias, dtype=dtype, device=x.device),
            )
        with binding.lease() as lease:
            return self._forward_biases(
                x,
                lease.get("q_bias", dtype=dtype),
                lease.get("zero_k_bias", dtype=dtype),
                lease.get("v_bias", dtype=dtype),
            )


class AttnProjection(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        mlp_ratio: float = 2,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_AUDIO_ATTENTION,
    ) -> None:
        super().__init__()
        self.norm1 = operations.layer_norm(in_dim)
        self.attn = CausalAttention(
            in_dim,
            out_dim,
            num_heads,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.proj = operations.linear(in_dim, out_dim)
        self.norm3 = operations.layer_norm(in_dim)
        self.norm2 = operations.layer_norm(out_dim)
        hidden_dim = int(out_dim * mlp_ratio)
        if hidden_dim <= 0:
            raise ValueError("attention projection MLP width must be positive")
        self.mlp = GeGluMlp(out_dim, hidden_dim, operations=operations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.proj(self.norm3(x)).add_(self.attn(self.norm1(x)))
        return projected.add_(self.mlp(self.norm2(projected)))


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


class AMPBlock1(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Sequence[int] = (1, 3, 5),
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        if channels <= 0 or kernel_size <= 0 or not dilation:
            raise ValueError("AMP block dimensions must be positive")
        self.convs1 = torch.nn.ModuleList(
            operations.conv1d(
                channels,
                channels,
                kernel_size,
                dilation=rate,
                padding=get_padding(kernel_size, rate),
            )
            for rate in dilation
        )
        self.convs2 = torch.nn.ModuleList(
            operations.conv1d(
                channels,
                channels,
                kernel_size,
                padding=get_padding(kernel_size),
            )
            for _ in dilation
        )
        self.activations = torch.nn.ModuleList(
            Activation1d(
                SnakeBeta(channels, operations=operations),
                operations=operations,
            )
            for _ in range(len(dilation) * 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acts1 = self.activations[::2]
        acts2 = self.activations[1::2]
        for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, acts1, acts2, strict=True):
            update = conv2(act2(conv1(act1(x))))
            x = update.add_(x)
        return x


class BigVGAN(torch.nn.Module):
    def __init__(
        self,
        num_mels: int = 2048,
        upsample_initial_channel: int = 1024,
        upsample_rates: Sequence[int] = (5, 5, 2, 2, 2, 2, 2),
        upsample_kernel_sizes: Sequence[int] = (9, 9, 4, 4, 4, 4, 4),
        resblock_kernel_sizes: Sequence[int] = (3, 7, 11),
        resblock_dilation_sizes: Sequence[Sequence[int]] = (
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        ),
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        if (
            num_mels <= 0
            or upsample_initial_channel <= 0
            or not upsample_rates
            or len(upsample_rates) != len(upsample_kernel_sizes)
            or not resblock_kernel_sizes
            or len(resblock_kernel_sizes) != len(resblock_dilation_sizes)
            or any(rate <= 0 for rate in upsample_rates)
        ):
            raise ValueError("BigVGAN configuration dimensions must be positive and aligned")
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = operations.conv1d(num_mels, upsample_initial_channel, 7, padding=3)
        self.ups = torch.nn.ModuleList()
        for index, (rate, kernel) in enumerate(
            zip(upsample_rates, upsample_kernel_sizes, strict=True)
        ):
            in_channels = upsample_initial_channel // (2**index)
            out_channels = upsample_initial_channel // (2 ** (index + 1))
            if out_channels <= 0 or kernel < rate:
                raise ValueError("BigVGAN upsample channels and kernels must stay positive")
            self.ups.append(
                torch.nn.ModuleList(
                    [
                        operations.conv_transpose1d(
                            in_channels,
                            out_channels,
                            kernel,
                            stride=rate,
                            padding=(kernel - rate) // 2,
                        )
                    ]
                )
            )
        self.resblocks = torch.nn.ModuleList()
        final_channels = upsample_initial_channel
        for index in range(len(self.ups)):
            final_channels = upsample_initial_channel // (2 ** (index + 1))
            for kernel, dilation in zip(
                resblock_kernel_sizes, resblock_dilation_sizes, strict=True
            ):
                self.resblocks.append(
                    AMPBlock1(
                        final_channels,
                        kernel,
                        dilation,
                        operations=operations,
                    )
                )
        self.activation_post = Activation1d(
            SnakeBeta(final_channels, operations=operations),
            operations=operations,
        )
        self.conv_post = operations.conv1d(final_channels, 1, 7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)
        for index in range(self.num_upsamples):
            upsample_group = cast(torch.nn.ModuleList, self.ups[index])
            for upsample in upsample_group:
                x = upsample(x)
            combined: torch.Tensor | None = None
            for offset in range(self.num_kernels):
                output = self.resblocks[index * self.num_kernels + offset](x)
                combined = output if combined is None else combined.add_(output)
            assert combined is not None
            x = combined.div_(self.num_kernels)
        return self.conv_post(self.activation_post(x)).clamp_(-1.0, 1.0)


class MiniMaxH3AudioVAE(ResidencyRouted, torch.nn.Module):
    """The unregistered H3 stereo audio VAE with checkpoint-compatible keys."""

    sample_rate = 32_000
    output_sample_rate = 32_000
    encoder: torch.nn.Module
    pre_block: torch.nn.Module
    mean_proj: torch.nn.Module
    logs_proj: torch.nn.Module
    dec_in_proj: torch.nn.Module
    decoder: torch.nn.Module
    latents_mean: torch.Tensor
    latents_std: torch.Tensor
    _compute_dtype: torch.dtype | None

    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: Sequence[int] = (2, 4, 4, 5, 5),
        latent_dim: int = 2048,
        decoder_dim: int = 1024,
        vae_latent_channels: int = 32,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_AUDIO_ATTENTION,
    ) -> None:
        super().__init__()
        if vae_latent_channels != 32:
            raise ValueError("MiniMax H3 audio latent width must be exactly 32")
        if tuple(encoder_rates) != (2, 4, 4, 5, 5):
            raise ValueError("MiniMax H3 audio encoder rates must be exactly (2, 4, 4, 5, 5)")
        self.vae_latent_channels = vae_latent_channels
        self.hop_length = math.prod(encoder_rates)
        self.samples_per_latent = self.hop_length
        self.latents_per_second = self.sample_rate // self.hop_length
        self._compute_dtype = _operations_compute_dtype(operations)
        self.encoder = DACEncoder(
            encoder_dim,
            encoder_rates,
            latent_dim,
            operations=operations,
        )
        self.pre_block = AttnProjection(
            latent_dim,
            vae_latent_channels,
            num_heads=8,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.mean_proj = operations.conv1d(vae_latent_channels, vae_latent_channels, 1)
        self.logs_proj = operations.conv1d(vae_latent_channels, vae_latent_channels, 1)
        self.dec_in_proj = operations.conv1d(vae_latent_channels, latent_dim, 1)
        self.decoder = BigVGAN(
            num_mels=latent_dim,
            upsample_initial_channel=decoder_dim,
            operations=operations,
        )
        self.register_buffer("latents_mean", torch.empty(vae_latent_channels))
        self.register_buffer("latents_std", torch.empty(vae_latent_channels))

    def encode_output_shape(self, input_shape: torch.Size | tuple[int, ...]) -> tuple[int, ...]:
        if len(input_shape) != 3:
            raise ValueError("MiniMax H3 audio waveform must have rank 3 [B,2,L]")
        batch, stereo, samples = input_shape
        if batch < 1 or stereo != 2 or samples < 1:
            raise ValueError("MiniMax H3 audio waveform must be nonempty stereo content")
        return (
            batch,
            self.vae_latent_channels,
            stereo,
            math.ceil(samples / self.hop_length),
        )

    def _prefetch_dtype(self, stored: torch.Tensor) -> torch.dtype:
        return stored.dtype if self._compute_dtype is None else self._compute_dtype

    def _validate_state(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        expected = (self.vae_latent_channels,)
        if (
            tuple(mean.shape) != expected
            or not mean.is_floating_point()
            or not torch.isfinite(mean).all().item()
        ):
            raise ValueError("latents_mean must be a finite floating 32-element vector")
        if (
            tuple(std.shape) != expected
            or not std.is_floating_point()
            or not torch.isfinite(std).all().item()
            or torch.any(std <= 0).item()
        ):
            raise ValueError("latents_std must be a finite positive floating 32-element vector")

    def _decode_owned(
        self,
        latent: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, stereo, frames = latent.shape
        mono_batch = latent.permute(0, 2, 1, 3).reshape(batch * stereo, channels, frames)
        normalized_mean = mean.to(device=latent.device).view(1, -1, 1)
        normalized_std = std.to(device=latent.device).view(1, -1, 1)
        decoded = self.decoder(
            self.dec_in_proj(mono_batch.mul(normalized_std).add_(normalized_mean))
        )
        return decoded.reshape(batch, stereo, -1)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if (
            latent.ndim != 4
            or latent.shape[0] <= 0
            or latent.shape[1] != self.vae_latent_channels
            or latent.shape[2] != 2
            or latent.shape[3] <= 0
        ):
            raise ValueError(
                f"MiniMax H3 audio latent must be nonempty [B,32,2,T], got {tuple(latent.shape)}"
            )
        if latent.layout != torch.strided or not latent.is_floating_point():
            raise ValueError("MiniMax H3 audio latent must be a strided floating tensor")
        dtype = _direct_state_dtype(self._compute_dtype, latent.dtype)
        binding = self._offloaded_residency()
        if binding is None:
            self._validate_state(self.latents_mean, self.latents_std)
            return self._decode_owned(
                latent,
                _cast_direct_state(self.latents_mean, dtype=dtype, device=latent.device),
                _cast_direct_state(self.latents_std, dtype=dtype, device=latent.device),
            )
        with binding.lease() as lease:
            stored_mean = lease.get_stored("latents_mean")
            stored_std = lease.get_stored("latents_std")
            if not isinstance(stored_mean, torch.Tensor) or not isinstance(
                stored_std, torch.Tensor
            ):
                raise ValueError("MiniMax H3 audio normalization state must be tensors")
            self._validate_state(stored_mean, stored_std)
            return self._decode_owned(
                latent,
                lease.get("latents_mean", dtype=dtype),
                lease.get("latents_std", dtype=dtype),
            )

    def _encode_owned(
        self,
        waveform: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        batch, stereo, length = waveform.shape
        right_pad = (-length) % self.hop_length
        padded = F.pad(waveform, (0, right_pad))
        encoded = self.encoder(padded.reshape(batch * stereo, 1, -1))
        projected = self.pre_block(encoded.transpose(1, 2)).transpose(1, 2)
        posterior_mean = self.mean_proj(projected)
        normalized_mean = mean.to(device=waveform.device).view(1, -1, 1)
        normalized_std = std.to(device=waveform.device).view(1, -1, 1)
        normalized = posterior_mean.sub(normalized_mean).div_(normalized_std)
        return normalized.reshape(batch, stereo, self.vae_latent_channels, -1).permute(0, 2, 1, 3)

    def encode(self, waveform: torch.Tensor, *, sample_rate: int = 32_000) -> torch.Tensor:
        if sample_rate != self.sample_rate:
            raise ValueError(f"MiniMax H3 audio encode requires 32000 Hz, got {sample_rate}")
        if (
            waveform.ndim != 3
            or waveform.shape[0] <= 0
            or waveform.shape[1] != 2
            or waveform.shape[2] <= 0
        ):
            raise ValueError(
                f"MiniMax H3 audio waveform must be nonempty [B,2,L], got {tuple(waveform.shape)}"
            )
        if waveform.layout != torch.strided or not waveform.is_floating_point():
            raise ValueError("MiniMax H3 audio waveform must be a strided floating tensor")
        dtype = _direct_state_dtype(self._compute_dtype, waveform.dtype)
        binding = self._offloaded_residency()
        if binding is None:
            self._validate_state(self.latents_mean, self.latents_std)
            return self._encode_owned(
                waveform,
                _cast_direct_state(self.latents_mean, dtype=dtype, device=waveform.device),
                _cast_direct_state(self.latents_std, dtype=dtype, device=waveform.device),
            )
        with binding.lease() as lease:
            stored_mean = lease.get_stored("latents_mean")
            stored_std = lease.get_stored("latents_std")
            if not isinstance(stored_mean, torch.Tensor) or not isinstance(
                stored_std, torch.Tensor
            ):
                raise ValueError("MiniMax H3 audio normalization state must be tensors")
            self._validate_state(stored_mean, stored_std)
            return self._encode_owned(
                waveform,
                lease.get("latents_mean", dtype=dtype),
                lease.get("latents_std", dtype=dtype),
            )


__all__ = [
    "AMPBlock1",
    "Activation1d",
    "AttnProjection",
    "BigVGAN",
    "CausalAttention",
    "DACEncoder",
    "DownSample1d",
    "EncoderBlock",
    "GeGluMlp",
    "LowPassFilter1d",
    "MiniMaxH3AudioVAE",
    "ResidualUnit",
    "Snake1d",
    "SnakeBeta",
    "UpSample1d",
    "get_padding",
    "kaiser_sinc_filter1d",
    "snake",
]
