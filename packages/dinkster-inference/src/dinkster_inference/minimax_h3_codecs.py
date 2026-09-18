"""Torch-free MiniMax H3 codec contracts: configuration and layouts.

The video VAE configuration, latent normalization constants, and the
exact state-dict layouts of both production codecs live here so
split-checkpoint planning can validate codec sources without
constructing torch modules. The layout tables are generated from the
torch modules by ``tools/gen_minimax_h3_vae_layouts.py``; the torch
suite cross-checks them against the live modules.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType

from .vendored import read_vendored

MINIMAX_H3_VIDEO_VAE_LAYOUT_SHA256 = (
    "a47094dd7f42317d57a3aa7639dcc38379cb894bfe7b9734429f8dc6751e6b4f"
)
MINIMAX_H3_AUDIO_VAE_LAYOUT_SHA256 = (
    "946291ca98a76206e3dd99f66ae1209f7cef07d86b6332767b7efee371080c05"
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LATENTS_MEAN = (
    0.858090341091156,
    -0.9606591463088989,
    1.0661640167236328,
    -0.5090325474739075,
    -0.2727581858634949,
    -1.3675414323806763,
    -0.2553254961967468,
    -0.26907554268836975,
    -0.5376840829849243,
    -0.0464097298681736,
    0.6657370328903198,
    0.19690127670764923,
    -0.5460608005523682,
    -0.4035342037677765,
    -0.23683024942874908,
    0.25928452610969543,
    -0.30133944749832153,
    0.211341992020607,
    -1.1206848621368408,
    0.3581933379173279,
    -0.04225143790245056,
    0.2604829967021942,
    0.22864092886447906,
    0.7056031823158264,
)

LATENTS_STD = (
    1.2223774194717407,
    1.2767263650894165,
    1.68317747116088865,
    1.7549455165863037,
    1.5636216402053833,
    2.194143533706665,
    0.96531379222869875,
    1.05698859691619875,
    0.841948926448822,
    0.7729952931404114,
    1.8955937623977661,
    0.946841835975647,
    0.7996809482574463,
    0.44988900423049925,
    0.7197399735450745,
    0.69362932443618775,
    2.961095094680786,
    2.7694199085235595,
    3.0496184825897215,
    2.1088054180145265,
    3.276226282119751,
    3.1627357006073,
    2.28168129920959475,
    2.6127843856811525,
)


@dataclass(frozen=True, slots=True)
class MiniMaxH3VideoVAEConfig:
    """Exact production defaults plus reduced-size construction knobs."""

    in_channels: int = 3
    out_channels: int = 3
    ch: int = 128
    embed_dim: int = 24
    z_channels: int = 24
    ch_mult: tuple[int, ...] = (1, 2, 2, 4, 4, 8)
    num_res_blocks: int | tuple[int, ...] = 2
    space_down: tuple[int, ...] = (2, 2, 2, 2, 1, 1)
    time_down: tuple[int, ...] = (1, 2, 2, 1, 1, 1)
    clip_length: int = 17
    token_drop: int = 3
    tile_size: int = 256
    tile_overlap_min: int = 64
    tiling: bool = True
    decoder_num_layers: int = 36
    decoder_heads: int = 32
    decoder_dim_head: int = 64
    decoder_rope_theta: float = 100.0
    decoder_rope_dim_ratio: float = 0.75
    decoder_num_register_tokens: int = 4

    def __post_init__(self) -> None:
        levels = len(self.ch_mult)
        if levels == 0 or len(self.space_down) != levels or len(self.time_down) != levels:
            raise ValueError("ch_mult, space_down, and time_down must have equal nonzero lengths")
        if any(value not in (1, 2) for value in self.space_down + self.time_down):
            raise ValueError("space_down and time_down entries must be 1 or 2")
        if isinstance(self.num_res_blocks, tuple) and len(self.num_res_blocks) != levels:
            raise ValueError("num_res_blocks tuple must have one entry per level")
        positive = (
            self.in_channels,
            self.out_channels,
            self.ch,
            self.embed_dim,
            self.z_channels,
            self.clip_length,
            self.tile_size,
            self.decoder_num_layers,
            self.decoder_heads,
            self.decoder_dim_head,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("channel, chunk, tile, and decoder dimensions must be positive")
        if self.token_drop < 0 or self.tile_overlap_min < 0:
            raise ValueError("token_drop and tile_overlap_min must be nonnegative")
        if self.token_drop >= math.ceil(self.clip_length / math.prod(self.time_down)):
            raise ValueError("token_drop must be smaller than the encoded chunk length")
        if self.in_channels != 3 or self.out_channels != 3:
            raise ValueError("MiniMax H3 video content must have exactly three RGB channels")
        if self.embed_dim > len(LATENTS_MEAN):
            raise ValueError("embed_dim exceeds the pinned latent transform")
        rotary_dim = int(self.decoder_dim_head * self.decoder_rope_dim_ratio)
        if rotary_dim <= 0 or rotary_dim > self.decoder_dim_head or rotary_dim % 6 != 0:
            raise ValueError(
                "decoder rotary dimension must be positive, <= head size, and divisible by 6"
            )


def _load_layout(name: str, expected_sha256: str) -> Mapping[str, tuple[int, ...]]:
    data = json.loads(read_vendored(name, expected_sha256))
    return MappingProxyType({key: tuple(shape) for key, shape in data.items()})


@cache
def minimax_h3_video_vae_layout() -> Mapping[str, tuple[int, ...]]:
    """Key -> shape table of the production-default video VAE."""

    return _load_layout("minimax_h3_video_vae_layout.json.gz", MINIMAX_H3_VIDEO_VAE_LAYOUT_SHA256)


@cache
def minimax_h3_audio_vae_layout() -> Mapping[str, tuple[int, ...]]:
    """Key -> shape table of the production-default audio VAE."""

    return _load_layout("minimax_h3_audio_vae_layout.json.gz", MINIMAX_H3_AUDIO_VAE_LAYOUT_SHA256)


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "LATENTS_MEAN",
    "LATENTS_STD",
    "MINIMAX_H3_AUDIO_VAE_LAYOUT_SHA256",
    "MINIMAX_H3_VIDEO_VAE_LAYOUT_SHA256",
    "MiniMaxH3VideoVAEConfig",
    "minimax_h3_audio_vae_layout",
    "minimax_h3_video_vae_layout",
]
