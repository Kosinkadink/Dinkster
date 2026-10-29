"""Torch-free contract for the DINOv3 ViT-H/16+ image encoder.

Architecture and preprocessing facts come from
``comfy/image_encoders/dino3.py`` (DINOV3_VITH_CONFIG @ 36408117): a
32-layer SwiGLU ViT with RoPE over patch positions, one class token,
four register tokens, and ImageNet normalization at a 1024px input.
TripoSplat conditions on this encoder's full token sequence.

The reference admits the checkpoint from two marker keys
(``layer.0.mlp.gate_proj.weight`` + ``layer.31.norm1.weight``,
comfy/clip_vision.py @ 36408117) and loads non-strictly. Detection
here requires the ENTIRE key/shape listing exactly; the published
VAST-AI/TripoSplat ``dino_v3_vit_h.safetensors`` header matches it
(tests/goldens/triposplat_headers.json). Dtypes are ignored - the
published artifact ships bf16. The RoPE ``inv_freq`` buffer is
non-persistent and absent from checkpoints.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from .weights import TensorGeometry


class DINOv3DetectError(ValueError):
    """The header is not the exact DINOv3 ViT-H/16+ checkpoint."""


@dataclass(frozen=True, slots=True)
class DINOv3ViTConfig:
    hidden_size: int = 1280
    num_hidden_layers: int = 32
    num_attention_heads: int = 20
    num_register_tokens: int = 4
    intermediate_size: int = 5120
    image_size: int = 1024
    patch_size: int = 16
    num_channels: int = 3
    rope_theta: float = 100.0
    layer_norm_eps: float = 1e-5
    mlp_kind: Literal["swiglu", "gelu"] = "swiglu"
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __post_init__(self) -> None:
        dimensions = (
            self.hidden_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_register_tokens,
            self.intermediate_size,
            self.image_size,
            self.patch_size,
            self.num_channels,
        )
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("DINOv3 dimensions and layer counts must be positive integers")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.num_channels != 3:
            raise ValueError("DINOv3 requires three input channels")
        if self.mlp_kind not in ("swiglu", "gelu"):
            raise ValueError("DINOv3 MLP kind must be swiglu or gelu")
        for value in (self.rope_theta, self.layer_norm_eps):
            if not isinstance(value, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("rope_theta and layer_norm_eps must be finite positive floats")

    @property
    def tokens(self) -> int:
        """Sequence length at the square training resolution: class
        token + register tokens + patch tokens."""
        return 1 + self.num_register_tokens + (self.image_size // self.patch_size) ** 2


DINOV3_VITH = DINOv3ViTConfig()
DINOV3_VITL = DINOv3ViTConfig(
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    intermediate_size=4096,
    image_size=512,
    mlp_kind="gelu",
)


@dataclass(frozen=True, slots=True)
class NAFConfig:
    """Pixal3D's Neighborhood Attention Filtering feature upsampler."""

    channels: int = 256
    attention_heads: int = 4
    rope_heads: int = 4
    kernel_size: int = 9
    image_layers: int = 2
    rope_base: float = 100.0


PIXAL3D_NAF = NAFConfig()


def naf_layout(config: NAFConfig = PIXAL3D_NAF) -> dict[str, tuple[int, ...]]:
    """The exact learned NAF state carried under ``naf.`` in Pixal3D vision weights."""
    half = config.channels // 2
    layout: dict[str, tuple[int, ...]] = {
        "image_encoder.encoder.0.weight": (half, 3, 1, 1),
        "image_encoder.encoder.0.bias": (half,),
        "image_encoder.sem_encoder.0.weight": (half, 3, 3, 3),
        "image_encoder.sem_encoder.0.bias": (half,),
        "image_encoder.rope.periods": (config.channels // config.rope_heads // 4,),
    }
    for branch, kernel in (("encoder", 1), ("sem_encoder", 3)):
        for index in range(1, config.image_layers + 1):
            prefix = f"image_encoder.{branch}.{index}."
            for norm in ("norm1", "norm2"):
                layout[f"{prefix}{norm}.weight"] = (half,)
                layout[f"{prefix}{norm}.bias"] = (half,)
            for convolution in ("conv1", "conv2"):
                layout[f"{prefix}{convolution}.weight"] = (half, half, kernel, kernel)
                layout[f"{prefix}{convolution}.bias"] = (half,)
    return layout


def detect_naf(geometries: Mapping[str, TensorGeometry]) -> NAFConfig:
    """Admit only the exact NAF architecture shipped with Pixal3D."""
    expected = naf_layout()
    problems: list[str] = []
    for key, shape in expected.items():
        found = geometries.get(key)
        if found is None:
            problems.append(f"missing {key}")
        elif found.shape != shape:
            problems.append(f"{key}: expected shape {shape}, found {found.shape}")
    problems.extend(f"unexpected key {key}" for key in sorted(set(geometries) - set(expected)))
    if problems:
        shown = "; ".join(problems[:6])
        if len(problems) > 6:
            shown += f"; and {len(problems) - 6} more"
        raise DINOv3DetectError(f"not the Pixal3D NAF checkpoint: {shown}")
    return PIXAL3D_NAF


def dinov3_vith_layout(
    config: DINOv3ViTConfig = DINOV3_VITH,
) -> dict[str, tuple[int, ...]]:
    """The checkpoint's exact key-to-shape contract."""
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    layout: dict[str, tuple[int, ...]] = {
        "embeddings.cls_token": (1, 1, hidden),
        "embeddings.mask_token": (1, 1, hidden),
        "embeddings.register_tokens": (1, config.num_register_tokens, hidden),
        "embeddings.patch_embeddings.weight": (
            hidden,
            config.num_channels,
            config.patch_size,
            config.patch_size,
        ),
        "embeddings.patch_embeddings.bias": (hidden,),
        "norm.weight": (hidden,),
        "norm.bias": (hidden,),
    }
    for index in range(config.num_hidden_layers):
        prefix = f"layer.{index}."
        for norm in ("norm1", "norm2"):
            layout[f"{prefix}{norm}.weight"] = (hidden,)
            layout[f"{prefix}{norm}.bias"] = (hidden,)
        # The key projection is bias-free; query/value/output carry biases.
        layout[f"{prefix}attention.k_proj.weight"] = (hidden, hidden)
        for projection in ("q_proj", "v_proj", "o_proj"):
            layout[f"{prefix}attention.{projection}.weight"] = (hidden, hidden)
            layout[f"{prefix}attention.{projection}.bias"] = (hidden,)
        for scale in ("layer_scale1", "layer_scale2"):
            layout[f"{prefix}{scale}.lambda1"] = (hidden,)
        projections = ("gate_proj", "up_proj") if config.mlp_kind == "swiglu" else ("up_proj",)
        for projection in projections:
            layout[f"{prefix}mlp.{projection}.weight"] = (intermediate, hidden)
            layout[f"{prefix}mlp.{projection}.bias"] = (intermediate,)
        layout[f"{prefix}mlp.down_proj.weight"] = (hidden, intermediate)
        layout[f"{prefix}mlp.down_proj.bias"] = (hidden,)
    return layout


def detect_dinov3_vith(
    geometries: Mapping[str, TensorGeometry],
) -> DINOv3ViTConfig:
    """Admit only the exact ViT-H/16+ header, or refuse loudly."""
    if not geometries:
        raise DINOv3DetectError("empty state dict header")
    if "layer.0.mlp.gate_proj.weight" not in geometries or "layer.31.norm1.weight" not in (
        geometries
    ):
        raise DINOv3DetectError(
            "not a DINOv3 ViT-H/16+ checkpoint (no layer.0.mlp.gate_proj.weight +"
            " layer.31.norm1.weight marker pair)"
        )
    expected = dinov3_vith_layout()
    problems: list[str] = []
    for key, shape in expected.items():
        found = geometries.get(key)
        if found is None:
            problems.append(f"missing {key}")
        elif found.shape != shape:
            problems.append(f"{key}: expected shape {shape}, found {found.shape}")
    problems.extend(f"unexpected key {key}" for key in sorted(set(geometries) - set(expected)))
    if problems:
        shown = "; ".join(problems[:6])
        if len(problems) > 6:
            shown += f"; and {len(problems) - 6} more"
        raise DINOv3DetectError(f"not the DINOv3 ViT-H/16+ checkpoint: {shown}")
    return DINOV3_VITH


def detect_dinov3_vitl(
    geometries: Mapping[str, TensorGeometry],
) -> DINOv3ViTConfig:
    """Admit only the exact ViT-L/16 standard-MLP checkpoint."""
    if "layer.0.mlp.up_proj.weight" not in geometries or "layer.23.norm1.weight" not in geometries:
        raise DINOv3DetectError(
            "not a DINOv3 ViT-L/16 checkpoint (missing layer 0 MLP or layer 23 marker)"
        )
    expected = dinov3_vith_layout(DINOV3_VITL)
    problems: list[str] = []
    for key, shape in expected.items():
        found = geometries.get(key)
        if found is None:
            problems.append(f"missing {key}")
        elif found.shape != shape:
            problems.append(f"{key}: expected shape {shape}, found {found.shape}")
    problems.extend(f"unexpected key {key}" for key in sorted(set(geometries) - set(expected)))
    if problems:
        shown = "; ".join(problems[:6])
        if len(problems) > 6:
            shown += f"; and {len(problems) - 6} more"
        raise DINOv3DetectError(f"not the DINOv3 ViT-L/16 checkpoint: {shown}")
    return DINOV3_VITL


__all__ = [
    "DINOV3_VITH",
    "DINOV3_VITL",
    "DINOv3DetectError",
    "DINOv3ViTConfig",
    "NAFConfig",
    "PIXAL3D_NAF",
    "detect_dinov3_vith",
    "detect_dinov3_vitl",
    "detect_naf",
    "dinov3_vith_layout",
    "naf_layout",
]
