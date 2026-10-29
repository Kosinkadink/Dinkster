"""LTX-2 prompt duration prediction."""

from __future__ import annotations

import math
from typing import Any, cast

import torch
import torch.nn.functional as F
from dinkster_inference import LTXDurationHeadConfig

from .operations import Operations, ResidencyRouted


class _LTXDurationAttention(ResidencyRouted, torch.nn.MultiheadAttention):
    pass


class LTXDurationAttentionPooler(ResidencyRouted, torch.nn.Module):
    def __init__(self, config: LTXDurationHeadConfig, *, operations: Operations) -> None:
        super().__init__()
        self.query_tokens = torch.nn.Parameter(
            torch.empty(config.num_queries, config.hidden_dim, device="meta")
        )
        self.cross_attn = _LTXDurationAttention(
            config.hidden_dim,
            config.num_heads,
            batch_first=True,
            device="meta",
        )
        cast("Any", self.cross_attn).out_proj = operations.linear(
            config.hidden_dim, config.hidden_dim
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        queries = self.query_tokens.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        pooled, _ = self.cross_attn(queries, tokens, tokens, need_weights=False)
        return pooled


class LTXDurationHead(ResidencyRouted, torch.nn.Module):
    """Predict natural shot duration in seconds from LTX connector outputs."""

    def __init__(self, config: LTXDurationHeadConfig, *, operations: Operations) -> None:
        super().__init__()
        self.config = config
        self.video_input_proj = operations.linear(
            config.video_input_dim,
            config.hidden_dim,
        )
        self.video_modality_emb = torch.nn.Parameter(torch.empty(config.hidden_dim, device="meta"))
        self.audio_input_proj = operations.linear(
            config.audio_input_dim,
            config.hidden_dim,
        )
        self.audio_modality_emb = torch.nn.Parameter(torch.empty(config.hidden_dim, device="meta"))
        self.attention_pooler = LTXDurationAttentionPooler(config, operations=operations)
        self.mlp_hidden = operations.linear(
            config.hidden_dim * config.num_queries,
            config.mlp_hidden_dim,
        )
        self.mlp_out = operations.linear(config.mlp_hidden_dim, 1)

    def forward(
        self,
        video_tokens: torch.Tensor | None = None,
        audio_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_groups: list[torch.Tensor] = []
        if video_tokens is not None:
            token_groups.append(self.video_input_proj(video_tokens) + self.video_modality_emb)
        if audio_tokens is not None:
            token_groups.append(self.audio_input_proj(audio_tokens) + self.audio_modality_emb)
        if not token_groups:
            raise ValueError("LTX duration head requires video or audio tokens")
        pooled = self.attention_pooler(torch.cat(token_groups, dim=1))
        hidden = F.gelu(
            self.mlp_hidden(pooled.reshape(pooled.shape[0], -1)),
            approximate="tanh",
        )
        return self.mlp_out(hidden).squeeze(-1).exp()


def ltx_duration_frames(
    seconds: float,
    frame_rate: float,
    min_seconds: float,
    max_seconds: float,
    *,
    time_scale: int = 8,
) -> int:
    """Clamp seconds and snap the result to the causal ``8k + 1`` grid."""
    for name, value in (
        ("seconds", seconds),
        ("frame_rate", frame_rate),
        ("min_seconds", min_seconds),
        ("max_seconds", max_seconds),
    ):
        if type(value) is not float or not math.isfinite(value):
            raise TypeError(f"{name} must be a finite float")
    if frame_rate <= 0.0 or min_seconds < 0.0 or max_seconds < min_seconds:
        raise ValueError("duration bounds and frame rate must be ordered and nonnegative")
    if type(time_scale) is not int or time_scale <= 0:
        raise ValueError("duration time scale must be a positive integer")
    min_frames = max(1, round(min_seconds * frame_rate))
    max_frames = round(max_seconds * frame_rate)
    raw_frames = max(min_frames, min(round(seconds * frame_rate), max_frames))
    frames = (raw_frames - 1) // time_scale * time_scale + 1
    if frames < min_frames:
        frames = min(-(-(min_frames - 1) // time_scale) * time_scale + 1, max_frames)
    return frames


__all__ = ["LTXDurationHead", "ltx_duration_frames"]
