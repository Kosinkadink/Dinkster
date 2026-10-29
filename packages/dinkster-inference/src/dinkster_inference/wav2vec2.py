"""Exact supported Wav2Vec2 inference profiles and weight layouts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Wav2Vec2Config:
    """One supported Wav2Vec2 encoder geometry."""

    embed_dim: int
    num_heads: int
    num_layers: int
    conv_dim: int = 512
    sample_rate: int = 16_000
    conv_norm: bool = field(default=True, repr=False)
    conv_bias: bool = field(default=True, repr=False)
    do_normalize: bool = field(default=True, repr=False)
    do_stable_layer_norm: bool = field(default=True, repr=False)

    def __post_init__(self) -> None:
        dimensions = (
            self.embed_dim,
            self.num_heads,
            self.num_layers,
            self.conv_dim,
            self.sample_rate,
        )
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("Wav2Vec2 dimensions and sample rate must be positive")
        if self.embed_dim % self.num_heads:
            raise ValueError("Wav2Vec2 embedding width must divide evenly across heads")
        behaviors = (
            self.conv_norm,
            self.conv_bias,
            self.do_normalize,
            self.do_stable_layer_norm,
        )
        if any(type(value) is not bool for value in behaviors):
            raise ValueError("Wav2Vec2 behavior flags must be exact bools")


WAV2VEC2_LARGE = Wav2Vec2Config(embed_dim=1024, num_heads=16, num_layers=24)
WAV2VEC2_CHINESE_BASE = Wav2Vec2Config(
    embed_dim=768,
    num_heads=12,
    num_layers=12,
    conv_norm=False,
    conv_bias=False,
    do_normalize=False,
    do_stable_layer_norm=False,
)
_SUPPORTED_CONFIGS = (WAV2VEC2_LARGE, WAV2VEC2_CHINESE_BASE)


def wav2vec2_layout(config: Wav2Vec2Config = WAV2VEC2_LARGE) -> dict[str, tuple[int, ...]]:
    """Canonical model state for one exact supported Wav2Vec2 profile."""

    if config not in _SUPPORTED_CONFIGS:
        raise ValueError("only the exact Wav2Vec2 large and Chinese base profiles are supported")
    layout: dict[str, tuple[int, ...]] = {"masked_spec_embed": (config.embed_dim,)}
    conv_specs = (
        (1, config.conv_dim, 10),
        (config.conv_dim, config.conv_dim, 3),
        (config.conv_dim, config.conv_dim, 3),
        (config.conv_dim, config.conv_dim, 3),
        (config.conv_dim, config.conv_dim, 3),
        (config.conv_dim, config.conv_dim, 2),
        (config.conv_dim, config.conv_dim, 2),
    )
    for index, (in_channels, out_channels, kernel) in enumerate(conv_specs):
        prefix = f"feature_extractor.conv_layers.{index}"
        layout[f"{prefix}.conv.weight"] = (out_channels, in_channels, kernel)
        if config.conv_bias:
            layout[f"{prefix}.conv.bias"] = (out_channels,)
        if config.conv_norm or index == 0:
            layout[f"{prefix}.layer_norm.weight"] = (out_channels,)
            layout[f"{prefix}.layer_norm.bias"] = (out_channels,)
    layout.update(
        {
            "feature_projection.layer_norm.weight": (config.conv_dim,),
            "feature_projection.layer_norm.bias": (config.conv_dim,),
            "feature_projection.projection.weight": (config.embed_dim, config.conv_dim),
            "feature_projection.projection.bias": (config.embed_dim,),
            "encoder.pos_conv_embed.conv.weight_g": (1, 1, 128),
            "encoder.pos_conv_embed.conv.weight_v": (
                config.embed_dim,
                config.embed_dim // 16,
                128,
            ),
            "encoder.pos_conv_embed.conv.bias": (config.embed_dim,),
            "encoder.layer_norm.weight": (config.embed_dim,),
            "encoder.layer_norm.bias": (config.embed_dim,),
        }
    )
    hidden = config.embed_dim * 4
    for index in range(config.num_layers):
        prefix = f"encoder.layers.{index}"
        for projection in ("k_proj", "v_proj", "q_proj", "out_proj"):
            layout[f"{prefix}.attention.{projection}.weight"] = (
                config.embed_dim,
                config.embed_dim,
            )
            layout[f"{prefix}.attention.{projection}.bias"] = (config.embed_dim,)
        layout[f"{prefix}.layer_norm.weight"] = (config.embed_dim,)
        layout[f"{prefix}.layer_norm.bias"] = (config.embed_dim,)
        layout[f"{prefix}.feed_forward.intermediate_dense.weight"] = (
            hidden,
            config.embed_dim,
        )
        layout[f"{prefix}.feed_forward.intermediate_dense.bias"] = (hidden,)
        layout[f"{prefix}.feed_forward.output_dense.weight"] = (
            config.embed_dim,
            hidden,
        )
        layout[f"{prefix}.feed_forward.output_dense.bias"] = (config.embed_dim,)
        layout[f"{prefix}.final_layer_norm.weight"] = (config.embed_dim,)
        layout[f"{prefix}.final_layer_norm.bias"] = (config.embed_dim,)
    return layout


__all__ = [
    "WAV2VEC2_CHINESE_BASE",
    "WAV2VEC2_LARGE",
    "Wav2Vec2Config",
    "wav2vec2_layout",
]
