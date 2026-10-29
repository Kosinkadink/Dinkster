"""Native Wav2Vec2 audio encoder for the exact supported profiles."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dinkster_inference import WAV2VEC2_CHINESE_BASE, WAV2VEC2_LARGE, Wav2Vec2Config

from .attention import AttentionKernel, attention_kernel_context, select_attention
from .operations import INITLESS, Operations, ResidencyRouted
from .ops import cast_weight

_DEFAULT_ATTENTION = select_attention("clip").kernel


class _LayerNormConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        *,
        bias: bool,
        operations: Operations,
    ) -> None:
        super().__init__()
        self.conv = operations.conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=bias,
        )
        self.layer_norm = operations.layer_norm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return F.gelu(self.layer_norm(x.transpose(-2, -1)).transpose(-2, -1))


class _GroupNormConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        *,
        bias: bool,
        operations: Operations,
    ) -> None:
        super().__init__()
        self.conv = operations.conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=bias,
        )
        self.layer_norm = operations.group_norm(
            out_channels,
            num_groups=out_channels,
            eps=1e-5,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.layer_norm(self.conv(x)))


class _ConvNoNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        *,
        bias: bool,
        operations: Operations,
    ) -> None:
        super().__init__()
        self.conv = operations.conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.conv(x))


class _ConvFeatureEncoder(torch.nn.Module):
    def __init__(self, config: Wav2Vec2Config, *, operations: Operations) -> None:
        super().__init__()
        dimensions = (
            ((1, config.conv_dim, 10, 5),)
            + ((config.conv_dim, config.conv_dim, 3, 2),) * 4
            + (
                (config.conv_dim, config.conv_dim, 2, 2),
                (config.conv_dim, config.conv_dim, 2, 2),
            )
        )
        if config.conv_norm:
            self.conv_layers = torch.nn.ModuleList(
                _LayerNormConv(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    bias=config.conv_bias,
                    operations=operations,
                )
                for in_channels, out_channels, kernel, stride in dimensions
            )
        else:
            first, *remaining = dimensions
            self.conv_layers = torch.nn.ModuleList(
                (
                    _GroupNormConv(
                        *first,
                        bias=config.conv_bias,
                        operations=operations,
                    ),
                    *(
                        _ConvNoNorm(
                            *dimension,
                            bias=config.conv_bias,
                            operations=operations,
                        )
                        for dimension in remaining
                    ),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        for layer in self.conv_layers:
            x = layer(x)
        return x.transpose(1, 2)


class _FeatureProjection(torch.nn.Module):
    def __init__(self, conv_dim: int, embed_dim: int, *, operations: Operations) -> None:
        super().__init__()
        self.layer_norm = operations.layer_norm(conv_dim)
        self.projection = operations.linear(conv_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.layer_norm(x))


class _WeightNormalizedConv1d(ResidencyRouted, torch.nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.weight_g = torch.nn.Parameter(torch.empty(1, 1, 128))
        self.weight_v = torch.nn.Parameter(torch.empty(embed_dim, embed_dim // 16, 128))
        self.bias = torch.nn.Parameter(torch.empty(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is None:
            weight_g = cast_weight(self.weight_g, device=x.device, dtype=x.dtype)
            weight_v = cast_weight(self.weight_v, device=x.device, dtype=x.dtype)
            bias = cast_weight(self.bias, device=x.device, dtype=x.dtype)
        else:
            with binding.lease() as lease:
                weight_g = lease.get("weight_g", dtype=x.dtype)
                weight_v = lease.get("weight_v", dtype=x.dtype)
                bias = lease.get("bias", dtype=x.dtype)
                weight = torch._weight_norm(  # pyright: ignore[reportPrivateUsage, reportPrivateImportUsage]
                    weight_v, weight_g, 2
                )
                return F.conv1d(x, weight, bias, padding=64, groups=16)
        weight = torch._weight_norm(  # pyright: ignore[reportPrivateUsage, reportPrivateImportUsage]
            weight_v, weight_g, 2
        )
        return F.conv1d(x, weight, bias, padding=64, groups=16)


class _PositionalConvEmbedding(torch.nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.conv = _WeightNormalizedConv1d(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x.transpose(1, 2))[:, :, :-1]
        return F.gelu(x).transpose(1, 2)


class _Attention(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.k_proj = operations.linear(embed_dim, embed_dim)
        self.v_proj = operations.linear(embed_dim, embed_dim)
        self.q_proj = operations.linear(embed_dim, embed_dim)
        self.out_proj = operations.linear(embed_dim, embed_dim)
        self._attention_kernel = attention_kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, _channels = x.shape

        def heads(value: torch.Tensor) -> torch.Tensor:
            return value.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)

        output = self._attention_kernel(
            heads(self.q_proj(x)),
            heads(self.k_proj(x)),
            heads(self.v_proj(x)),
        )
        output = output.transpose(1, 2).reshape(batch, sequence, -1)
        return self.out_proj(output)


class _FeedForward(torch.nn.Module):
    def __init__(self, embed_dim: int, *, operations: Operations) -> None:
        super().__init__()
        self.intermediate_dense = operations.linear(embed_dim, embed_dim * 4)
        self.output_dense = operations.linear(embed_dim * 4, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_dense(F.gelu(self.intermediate_dense(x)))


class _TransformerEncoderLayer(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        do_stable_layer_norm: bool,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.attention = _Attention(
            embed_dim,
            num_heads,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.layer_norm = operations.layer_norm(embed_dim)
        self.feed_forward = _FeedForward(embed_dim, operations=operations)
        self.final_layer_norm = operations.layer_norm(embed_dim)
        self.do_stable_layer_norm = do_stable_layer_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.do_stable_layer_norm:
            x = self.layer_norm(x)
        x = residual + self.attention(x)
        if self.do_stable_layer_norm:
            return x + self.feed_forward(self.final_layer_norm(x))
        x = self.layer_norm(x)
        return self.final_layer_norm(x + self.feed_forward(x))


class _TransformerEncoder(torch.nn.Module):
    def __init__(
        self,
        config: Wav2Vec2Config,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self._attention_kernel = attention_kernel
        self.pos_conv_embed = _PositionalConvEmbedding(config.embed_dim)
        self.layers = torch.nn.ModuleList(
            _TransformerEncoderLayer(
                config.embed_dim,
                config.num_heads,
                do_stable_layer_norm=config.do_stable_layer_norm,
                operations=operations,
                attention_kernel=attention_kernel,
            )
            for _ in range(config.num_layers)
        )
        self.layer_norm = operations.layer_norm(config.embed_dim)
        self.do_stable_layer_norm = config.do_stable_layer_norm

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        x = x + self.pos_conv_embed(x)
        outputs: list[torch.Tensor] = []
        if not self.do_stable_layer_norm:
            x = self.layer_norm(x)
        with attention_kernel_context(
            self._attention_kernel,
            x.numel(),
            device=x.device,
        ):
            for layer in self.layers:
                outputs.append(x)
                x = layer(x)
        if self.do_stable_layer_norm:
            x = self.layer_norm(x)
        outputs.append(x)
        return x, tuple(outputs)


class Wav2Vec2Model(ResidencyRouted, torch.nn.Module):
    """An exact supported Wav2Vec2 encoder used by native Wan runtimes."""

    def __init__(
        self,
        config: Wav2Vec2Config = WAV2VEC2_LARGE,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
    ) -> None:
        super().__init__()
        if config not in (WAV2VEC2_LARGE, WAV2VEC2_CHINESE_BASE):
            raise ValueError("Wav2Vec2Model supports only exact large and Chinese base profiles")
        self.config = config
        self.feature_extractor = _ConvFeatureEncoder(config, operations=operations)
        self.feature_projection = _FeatureProjection(
            config.conv_dim,
            config.embed_dim,
            operations=operations,
        )
        self.masked_spec_embed = torch.nn.Parameter(torch.empty(config.embed_dim))
        self.encoder = _TransformerEncoder(
            config,
            operations=operations,
            attention_kernel=attention_kernel,
        )

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if type(audio) is not torch.Tensor or not audio.is_floating_point():
            raise TypeError("Wav2Vec2 audio must be an exact floating torch.Tensor")
        if audio.ndim != 3 or any(size <= 0 for size in audio.shape):
            raise ValueError("Wav2Vec2 audio must be nonempty [batch,channels,samples]")
        audio = audio.mean(dim=1)
        if self.config.do_normalize:
            audio = (audio - audio.mean()) / torch.sqrt(audio.var() + 1e-7)
        features = self.feature_projection(self.feature_extractor(audio))
        return self.encoder(features)


__all__ = ["Wav2Vec2Model"]
