"""Native Whisper Large v3 audio encoder."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from dinkster_inference import WHISPER_LARGE_V3, WhisperLargeV3Config

from .attention import AttentionKernel, attention_kernel_context, select_attention
from .ltx_audio_vae import _mel_filterbank  # pyright: ignore[reportPrivateUsage]
from .operations import INITLESS, Operations

_DEFAULT_ATTENTION = select_attention("clip").kernel


class _WhisperFeatureExtractor(torch.nn.Module):
    def __init__(self, config: WhisperLargeV3Config) -> None:
        super().__init__()
        self.config = config
        self.register_buffer(
            "window",
            torch.hann_window(config.n_fft, device="cpu"),
            persistent=False,
        )
        with torch.device("cpu"):
            mel_filters = _mel_filterbank(
                config.n_fft // 2 + 1,
                config.n_mels,
                config.sample_rate,
            )
        self.register_buffer(
            "mel_filters",
            mel_filters,
            persistent=False,
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if type(audio) is not torch.Tensor or audio.dtype is not torch.float32:
            raise TypeError("Whisper audio must be an exact float32 torch.Tensor")
        if audio.ndim != 3 or any(size <= 0 for size in audio.shape):
            raise ValueError("Whisper audio must be nonempty [batch,channels,samples]")
        audio = audio.mean(dim=1)
        if audio.shape[-1] > self.config.chunk_samples:
            audio = audio[..., : self.config.chunk_samples]
        elif audio.shape[-1] < self.config.chunk_samples:
            audio = F.pad(audio, (0, self.config.chunk_samples - audio.shape[-1]))

        window = cast(torch.Tensor, self.window).to(device=audio.device)
        filters = cast(torch.Tensor, self.mel_filters).to(device=audio.device)
        spectrum = torch.stft(
            audio,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.n_fft,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        power = spectrum.abs().pow(2.0)
        mel = torch.matmul(power.transpose(-1, -2), filters).transpose(-1, -2)
        mel = mel[:, :, :-1]
        log_mel = torch.clamp(mel, min=1e-10).log10()
        log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
        return (log_mel + 4.0) / 4.0


class _Attention(torch.nn.Module):
    def __init__(
        self,
        state: int,
        heads: int,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = state // heads
        self.q_proj = operations.linear(state, state)
        self.k_proj = operations.linear(state, state, bias=False)
        self.v_proj = operations.linear(state, state)
        self.out_proj = operations.linear(state, state)
        self._attention_kernel = attention_kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, _state = x.shape

        def split_heads(value: torch.Tensor) -> torch.Tensor:
            return value.view(batch, sequence, self.heads, self.head_dim).transpose(1, 2)

        output = self._attention_kernel(
            split_heads(self.q_proj(x)),
            split_heads(self.k_proj(x)),
            split_heads(self.v_proj(x)),
        )
        return self.out_proj(output.transpose(1, 2).reshape(batch, sequence, -1))


class _EncoderLayer(torch.nn.Module):
    def __init__(
        self,
        state: int,
        heads: int,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.self_attn = _Attention(
            state,
            heads,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.self_attn_layer_norm = operations.layer_norm(state)
        self.fc1 = operations.linear(state, state * 4)
        self.fc2 = operations.linear(state * 4, state)
        self.final_layer_norm = operations.layer_norm(state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.self_attn_layer_norm(x))
        return x + self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))


class _AudioEncoder(torch.nn.Module):
    def __init__(
        self,
        n_mels: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.n_ctx = n_ctx
        self.conv1 = operations.conv1d(n_mels, n_state, 3, padding=1)
        self.conv2 = operations.conv1d(n_state, n_state, 3, stride=2, padding=1)
        self.embed_positions = operations.embedding(n_ctx, n_state)
        self.layers = torch.nn.ModuleList(
            _EncoderLayer(
                n_state,
                n_head,
                operations=operations,
                attention_kernel=attention_kernel,
            )
            for _ in range(n_layer)
        )
        self.layer_norm = operations.layer_norm(n_state)
        self._attention_kernel = attention_kernel

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if (
            type(features) is not torch.Tensor
            or features.ndim != 3
            or features.shape[1] != self.n_mels
            or features.shape[2] != self.n_ctx * 2
        ):
            raise ValueError(f"Whisper features must be [batch,{self.n_mels},{self.n_ctx * 2}]")
        x = F.gelu(self.conv1(features))
        x = F.gelu(self.conv2(x)).transpose(1, 2)
        positions = torch.arange(x.shape[1], device=x.device)
        x = x + self.embed_positions(positions)

        outputs: list[torch.Tensor] = []
        with attention_kernel_context(
            self._attention_kernel,
            x.numel(),
            device=x.device,
        ):
            for layer in self.layers:
                outputs.append(x)
                x = layer(x)
        x = self.layer_norm(x)
        outputs.append(x)
        return x, tuple(outputs)


class WhisperLargeV3Model(torch.nn.Module):
    """The exact 32-layer Whisper Large v3 audio encoder used by Wan HuMo."""

    def __init__(
        self,
        config: WhisperLargeV3Config = WHISPER_LARGE_V3,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
    ) -> None:
        super().__init__()
        if config != WHISPER_LARGE_V3:
            raise ValueError("WhisperLargeV3Model supports only the exact Large v3 profile")
        self.config = config
        self.feature_extractor = _WhisperFeatureExtractor(config)
        self.encoder = _AudioEncoder(
            config.n_mels,
            config.n_audio_ctx,
            config.n_audio_state,
            config.n_audio_head,
            config.n_audio_layer,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self._dinkster_residency_constant_buffers = frozenset(
            {"feature_extractor.window", "feature_extractor.mel_filters"}
        )

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        return self.encoder(self.feature_extractor(audio))


__all__ = ["WhisperLargeV3Model"]
