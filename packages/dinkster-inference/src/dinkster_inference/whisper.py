"""Exact Whisper Large v3 encoder configuration and weight layout."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WhisperLargeV3Config:
    """The supported Whisper Large v3 audio encoder geometry."""

    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    sample_rate: int
    n_fft: int
    hop_length: int
    chunk_samples: int

    def __post_init__(self) -> None:
        values = (
            self.n_mels,
            self.n_audio_ctx,
            self.n_audio_state,
            self.n_audio_head,
            self.n_audio_layer,
            self.sample_rate,
            self.n_fft,
            self.hop_length,
            self.chunk_samples,
        )
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("Whisper dimensions, rates, and lengths must be positive integers")
        if self.n_audio_state % self.n_audio_head:
            raise ValueError("Whisper state width must divide evenly across attention heads")
        if self.n_fft % 2:
            raise ValueError("Whisper FFT size must be even")
        if (
            self.chunk_samples % self.hop_length
            or self.chunk_samples // self.hop_length != self.n_audio_ctx * 2
        ):
            raise ValueError("Whisper chunk, hop, and context lengths are inconsistent")


WHISPER_LARGE_V3 = WhisperLargeV3Config(
    n_mels=128,
    n_audio_ctx=1500,
    n_audio_state=1280,
    n_audio_head=20,
    n_audio_layer=32,
    sample_rate=16_000,
    n_fft=400,
    hop_length=160,
    chunk_samples=480_000,
)


def whisper_large_v3_layout(
    config: WhisperLargeV3Config = WHISPER_LARGE_V3,
) -> dict[str, tuple[int, ...]]:
    """Canonical state for the supported Whisper Large v3 encoder."""

    if config != WHISPER_LARGE_V3:
        raise ValueError("only the exact Whisper Large v3 profile is supported")
    state = config.n_audio_state
    hidden = state * 4
    layout: dict[str, tuple[int, ...]] = {
        "encoder.conv1.weight": (state, config.n_mels, 3),
        "encoder.conv1.bias": (state,),
        "encoder.conv2.weight": (state, state, 3),
        "encoder.conv2.bias": (state,),
        "encoder.embed_positions.weight": (config.n_audio_ctx, state),
        "encoder.layer_norm.weight": (state,),
        "encoder.layer_norm.bias": (state,),
    }
    for index in range(config.n_audio_layer):
        prefix = f"encoder.layers.{index}"
        layout.update(
            {
                f"{prefix}.self_attn.q_proj.weight": (state, state),
                f"{prefix}.self_attn.q_proj.bias": (state,),
                f"{prefix}.self_attn.k_proj.weight": (state, state),
                f"{prefix}.self_attn.v_proj.weight": (state, state),
                f"{prefix}.self_attn.v_proj.bias": (state,),
                f"{prefix}.self_attn.out_proj.weight": (state, state),
                f"{prefix}.self_attn.out_proj.bias": (state,),
                f"{prefix}.self_attn_layer_norm.weight": (state,),
                f"{prefix}.self_attn_layer_norm.bias": (state,),
                f"{prefix}.fc1.weight": (hidden, state),
                f"{prefix}.fc1.bias": (hidden,),
                f"{prefix}.fc2.weight": (state, hidden),
                f"{prefix}.fc2.bias": (state,),
                f"{prefix}.final_layer_norm.weight": (state,),
                f"{prefix}.final_layer_norm.bias": (state,),
            }
        )
    return layout


__all__ = ["WHISPER_LARGE_V3", "WhisperLargeV3Config", "whisper_large_v3_layout"]
