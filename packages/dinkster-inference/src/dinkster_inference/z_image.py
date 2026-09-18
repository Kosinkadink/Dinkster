"""Exact native latent Z-Image architecture and fail-closed detection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .devices import BFLOAT16, FLOAT32, DType
from .families import EvidenceValue
from .spaces import FlowSigmas
from .weights import WeightSource

_PREFIXES = ("model.diffusion_model.", "")


@dataclass(frozen=True)
class ZImageConfig:
    family_id: str = "dinkster.z_image"
    hidden_width: int = 3840
    caption_width: int = 2560
    main_blocks: int = 30
    noise_refiner_blocks: int = 2
    context_refiner_blocks: int = 2
    attention_heads: int = 30
    kv_heads: int = 30
    attention_head_dim: int = 128
    ffn_width: int = 10240
    latent_channels: int = 16
    patch: tuple[int, int] = (2, 2)
    frame_patch: int = 1
    rope_axes: tuple[int, int, int] = (32, 48, 48)
    rope_theta: float = 256.0
    qk_norm_eps: float = 1e-5
    timestep_embedding_width: int = 256
    modulation_width: int = 256
    timestep_multiplier: float = 1000.0
    block_modulation_silu: bool = False
    pad_tokens_multiple: int = 32
    learned_padding: bool = True
    sampling_shift: float = 3.0
    inference_dtypes: tuple[DType, DType] = (BFLOAT16, FLOAT32)
    memory_factor: float = 2.8

    def __post_init__(self) -> None:
        actual = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        expected = (
            "dinkster.z_image",
            3840,
            2560,
            30,
            2,
            2,
            30,
            30,
            128,
            10240,
            16,
            (2, 2),
            1,
            (32, 48, 48),
            256.0,
            1e-5,
            256,
            256,
            1000.0,
            False,
            32,
            True,
            3.0,
            (BFLOAT16, FLOAT32),
            2.8,
        )
        if any(
            type(value) is not type(required) or value != required
            for value, required in zip(actual, expected, strict=True)
        ):
            raise ValueError("ZImageConfig only represents exact latent Z-Image")


Z_IMAGE_CONFIG = ZImageConfig()
Z_IMAGE_SIGMAS = FlowSigmas(shift=Z_IMAGE_CONFIG.sampling_shift)


@dataclass(frozen=True)
class ZImagePixelConfig(ZImageConfig):
    family_id: str = "dinkster.z_image_pixel_space"
    latent_channels: int = 3
    patch: tuple[int, int] = (32, 32)
    memory_factor: float = 0.03
    decoder_hidden_width: int = 3840
    decoder_blocks: int = 4
    decoder_max_frequencies: int = 8

    def __post_init__(self) -> None:
        expected = (
            "dinkster.z_image_pixel_space",
            3840,
            2560,
            30,
            2,
            2,
            30,
            30,
            128,
            10240,
            3,
            (32, 32),
            1,
            (32, 48, 48),
            256.0,
            1e-5,
            256,
            256,
            1000.0,
            False,
            32,
            True,
            3.0,
            (BFLOAT16, FLOAT32),
            0.03,
            3840,
            4,
            8,
        )
        actual = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        if any(
            type(value) is not type(required) or value != required
            for value, required in zip(actual, expected, strict=True)
        ):
            raise ValueError("ZImagePixelConfig only represents exact Zeta-Chroma")


Z_IMAGE_PIXEL_CONFIG = ZImagePixelConfig()


@dataclass(frozen=True)
class ZImageControlConfig:
    hidden_width: int = 3840
    control_layers: int = 6
    noise_refiner_blocks: int = 2
    latent_channels: int = 16
    injection_blocks: tuple[int, ...] = (0, 5, 10, 15, 20, 25)

    def __post_init__(self) -> None:
        expected = (3840, 6, 2, 16, (0, 5, 10, 15, 20, 25))
        actual = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        if actual != expected:
            raise ValueError("ZImageControlConfig only represents the exact Union checkpoint")


Z_IMAGE_CONTROL_CONFIG = ZImageControlConfig()
Z_IMAGE_CONTROL_RESIDUAL_SITES = tuple(
    f"z_image.dit.layer.{index:02d}.control_residual.v1"
    for index in Z_IMAGE_CONTROL_CONFIG.injection_blocks
)


def z_image_control_layout() -> Mapping[str, tuple[int, ...]]:
    """Return the immutable 136-entry Alibaba PAI Union checkpoint layout."""
    hidden = Z_IMAGE_CONTROL_CONFIG.hidden_width
    layout: dict[str, tuple[int, ...]] = {
        "control_all_x_embedder.2-1.weight": (hidden, 64),
        "control_all_x_embedder.2-1.bias": (hidden,),
    }
    common = {
        "adaLN_modulation.0.weight": (4 * hidden, 256),
        "adaLN_modulation.0.bias": (4 * hidden,),
        "attention.norm_q.weight": (128,),
        "attention.norm_k.weight": (128,),
        "attention.to_q.weight": (hidden, hidden),
        "attention.to_k.weight": (hidden, hidden),
        "attention.to_v.weight": (hidden, hidden),
        "attention.to_out.0.weight": (hidden, hidden),
        "attention_norm1.weight": (hidden,),
        "attention_norm2.weight": (hidden,),
        "feed_forward.w1.weight": (10240, hidden),
        "feed_forward.w2.weight": (hidden, 10240),
        "feed_forward.w3.weight": (10240, hidden),
        "ffn_norm1.weight": (hidden,),
        "ffn_norm2.weight": (hidden,),
    }
    for index in range(6):
        layout.update({f"control_layers.{index}.{key}": shape for key, shape in common.items()})
        layout[f"control_layers.{index}.after_proj.weight"] = (hidden, hidden)
        layout[f"control_layers.{index}.after_proj.bias"] = (hidden,)
    layout["control_layers.0.before_proj.weight"] = (hidden, hidden)
    layout["control_layers.0.before_proj.bias"] = (hidden,)
    for index in range(2):
        layout.update(
            {f"control_noise_refiner.{index}.{key}": shape for key, shape in common.items()}
        )
    return MappingProxyType(layout)


def detect_z_image_control(source: WeightSource) -> bool:
    """Fail closed unless every key, shape, and BF16 dtype matches the pinned artifact."""
    layout = z_image_control_layout()
    if frozenset(source.keys()) != frozenset(layout):
        return False
    return all(
        source.entry(key).geometry.shape == shape and source.entry(key).geometry.dtype == BFLOAT16
        for key, shape in layout.items()
    )


def _block_layout(
    config: ZImageConfig, *, modulated: bool, split_attention: bool = False
) -> dict[str, tuple[int, ...]]:
    hidden = config.hidden_width
    attention = (
        {
            "attention.to_q.weight": (hidden, hidden),
            "attention.to_k.weight": (hidden, hidden),
            "attention.to_v.weight": (hidden, hidden),
            "attention.to_out.0.weight": (hidden, hidden),
            "attention.norm_q.weight": (config.attention_head_dim,),
            "attention.norm_k.weight": (config.attention_head_dim,),
        }
        if split_attention
        else {
            "attention.qkv.weight": (3 * hidden, hidden),
            "attention.out.weight": (hidden, hidden),
            "attention.q_norm.weight": (config.attention_head_dim,),
            "attention.k_norm.weight": (config.attention_head_dim,),
        }
    )
    layout = {
        **attention,
        "attention_norm1.weight": (hidden,),
        "attention_norm2.weight": (hidden,),
        "ffn_norm1.weight": (hidden,),
        "ffn_norm2.weight": (hidden,),
        "feed_forward.w1.weight": (config.ffn_width, hidden),
        "feed_forward.w2.weight": (hidden, config.ffn_width),
        "feed_forward.w3.weight": (config.ffn_width, hidden),
    }
    if modulated:
        layout["adaLN_modulation.0.weight"] = (4 * hidden, config.modulation_width)
        layout["adaLN_modulation.0.bias"] = (4 * hidden,)
    return layout


def z_image_layout() -> Mapping[str, tuple[int, ...]]:
    """Return the immutable 453-entry official latent Z-Image layout."""
    config = Z_IMAGE_CONFIG
    hidden = config.hidden_width
    layout: dict[str, tuple[int, ...]] = {
        "cap_embedder.0.weight": (config.caption_width,),
        "cap_embedder.1.weight": (hidden, config.caption_width),
        "cap_embedder.1.bias": (hidden,),
        "cap_pad_token": (1, hidden),
        "x_embedder.weight": (hidden, config.latent_channels * 4),
        "x_embedder.bias": (hidden,),
        "x_pad_token": (1, hidden),
        "t_embedder.mlp.0.weight": (1024, config.modulation_width),
        "t_embedder.mlp.0.bias": (1024,),
        "t_embedder.mlp.2.weight": (config.modulation_width, 1024),
        "t_embedder.mlp.2.bias": (config.modulation_width,),
        "final_layer.adaLN_modulation.1.weight": (hidden, config.modulation_width),
        "final_layer.adaLN_modulation.1.bias": (hidden,),
        "final_layer.linear.weight": (config.latent_channels * 4, hidden),
        "final_layer.linear.bias": (config.latent_channels * 4,),
    }
    for root, count, modulated in (
        ("context_refiner", config.context_refiner_blocks, False),
        ("noise_refiner", config.noise_refiner_blocks, True),
        ("layers", config.main_blocks, True),
    ):
        block = _block_layout(config, modulated=modulated)
        for index in range(count):
            layout.update({f"{root}.{index}.{key}": shape for key, shape in block.items()})
    return MappingProxyType(layout)


def z_image_pixel_layout() -> Mapping[str, tuple[int, ...]]:
    """Return the immutable 557-entry Zeta-Chroma pixel-space layout."""
    config = Z_IMAGE_PIXEL_CONFIG
    hidden = config.hidden_width
    patch_values = config.latent_channels * config.patch[0] * config.patch[1]
    decoder = config.decoder_hidden_width
    layout: dict[str, tuple[int, ...]] = {
        "__sequential__": (0,),
        "__x0__": (0,),
        "cap_embedder.0.weight": (config.caption_width,),
        "cap_embedder.1.weight": (hidden, config.caption_width),
        "cap_embedder.1.bias": (hidden,),
        "cap_pad_token": (1, hidden),
        "x_embedder.weight": (hidden, patch_values),
        "x_embedder.bias": (hidden,),
        "x_pad_token": (1, hidden),
        "t_embedder.mlp.0.weight": (1024, config.modulation_width),
        "t_embedder.mlp.0.bias": (1024,),
        "t_embedder.mlp.2.weight": (config.modulation_width, 1024),
        "t_embedder.mlp.2.bias": (config.modulation_width,),
        "dec_net.cond_embed.weight": (decoder, hidden),
        "dec_net.cond_embed.bias": (decoder,),
        "dec_net.input_embedder.embedder.0.weight": (
            decoder,
            patch_values + config.decoder_max_frequencies**2,
        ),
        "dec_net.input_embedder.embedder.0.bias": (decoder,),
        "dec_net.final_layer.linear.weight": (patch_values, decoder),
        "dec_net.final_layer.linear.bias": (patch_values,),
    }
    for index in range(config.decoder_blocks):
        root = f"dec_net.res_blocks.{index}"
        layout.update(
            {
                f"{root}.in_ln.weight": (decoder,),
                f"{root}.in_ln.bias": (decoder,),
                f"{root}.mlp.0.weight": (decoder, decoder),
                f"{root}.mlp.0.bias": (decoder,),
                f"{root}.mlp.2.weight": (decoder, decoder),
                f"{root}.mlp.2.bias": (decoder,),
                f"{root}.adaLN_modulation.1.weight": (3 * decoder, decoder),
                f"{root}.adaLN_modulation.1.bias": (3 * decoder,),
            }
        )
    for root, count, modulated in (
        ("context_refiner", config.context_refiner_blocks, False),
        ("noise_refiner", config.noise_refiner_blocks, True),
        ("layers", config.main_blocks, True),
    ):
        block = _block_layout(config, modulated=modulated, split_attention=True)
        for index in range(count):
            layout.update({f"{root}.{index}.{key}": shape for key, shape in block.items()})
    return MappingProxyType(layout)


@dataclass(frozen=True)
class ZImageEvidence:
    config: ZImageConfig | ZImagePixelConfig
    key_prefix: str
    matched_keys: tuple[str, ...]
    fields: Mapping[str, EvidenceValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "matched_keys", tuple(self.matched_keys))
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))


def detect_z_image(source: WeightSource) -> ZImageEvidence | None:
    """Admit only exact latent Z-Image or Zeta-Chroma pixel-space layouts."""
    keys = frozenset(source.keys())
    for config, layout in (
        (Z_IMAGE_PIXEL_CONFIG, z_image_pixel_layout()),
        (Z_IMAGE_CONFIG, z_image_layout()),
    ):
        for prefix in _PREFIXES:
            expected_keys = frozenset(prefix + key for key in layout)
            candidate_keys = (
                keys if not prefix else frozenset(key for key in keys if key.startswith(prefix))
            )
            if candidate_keys != expected_keys:
                continue
            matched: list[str] = []
            for suffix, shape in layout.items():
                key = prefix + suffix
                try:
                    geometry = source.entry(key).geometry
                except KeyError:
                    break
                if geometry.shape != shape:
                    break
                matched.append(key)
            else:
                return ZImageEvidence(
                    config,
                    prefix,
                    tuple(sorted(matched)),
                    {
                        "caption_width": config.caption_width,
                        "hidden_width": config.hidden_width,
                        "key_prefix": prefix,
                        "main_blocks": config.main_blocks,
                        "patch": f"{config.patch[0]}x{config.patch[1]}",
                        "sampling_shift": config.sampling_shift,
                    },
                )
    return None


__all__ = [
    "Z_IMAGE_CONFIG",
    "Z_IMAGE_CONTROL_CONFIG",
    "Z_IMAGE_CONTROL_RESIDUAL_SITES",
    "Z_IMAGE_PIXEL_CONFIG",
    "Z_IMAGE_SIGMAS",
    "ZImageConfig",
    "ZImageControlConfig",
    "ZImageEvidence",
    "ZImagePixelConfig",
    "detect_z_image",
    "detect_z_image_control",
    "z_image_control_layout",
    "z_image_layout",
    "z_image_pixel_layout",
]
