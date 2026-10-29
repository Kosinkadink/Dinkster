"""Torch-free contract for Wan 2.1's CLIP ViT-H/14 image encoder.

The architecture and preprocessing facts come from ``comfy/clip_model.py``
and ``comfy/clip_vision_config_h.json`` at ComfyUI commit b78cec87. Admission
is intentionally limited to Comfy-Org's Wan 2.1 ``clip_vision_h.safetensors``:
1264219396 bytes, publisher SHA256
``64a7ef761bfccbadbaa3da77366aac4185a6c58fa5de5f589b42a65bcc21f161``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from .devices import FLOAT16, INT64
from .weights import TensorGeometry


class ClipVisionDetectError(ValueError):
    """The header is not the exact Wan 2.1 CLIP ViT-H checkpoint."""


@dataclass(frozen=True, slots=True)
class ClipVisionConfig:
    hidden_size: int = 1280
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    intermediate_size: int = 5120
    image_size: int = 224
    patch_size: int = 14
    projection_dim: int = 1024
    num_channels: int = 3
    layer_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        dimensions = (
            self.hidden_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.intermediate_size,
            self.image_size,
            self.patch_size,
            self.projection_dim,
            self.num_channels,
        )
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("CLIP vision dimensions and layer counts must be positive integers")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.num_channels != 3:
            raise ValueError("CLIP vision requires three input channels")
        if (
            not isinstance(self.layer_norm_eps, float)
            or not math.isfinite(self.layer_norm_eps)
            or self.layer_norm_eps <= 0
        ):
            raise ValueError("layer_norm_eps must be a finite positive float")

    @property
    def tokens(self) -> int:
        return (self.image_size // self.patch_size) ** 2 + 1


WAN21_CLIP_VISION = ClipVisionConfig()


def clip_vision_layout(
    config: ClipVisionConfig = WAN21_CLIP_VISION,
) -> dict[str, tuple[int, ...]]:
    """Return the checkpoint's exact key-to-shape contract."""
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    layout: dict[str, tuple[int, ...]] = {
        "vision_model.embeddings.position_ids": (1, config.tokens),
        "vision_model.embeddings.class_embedding": (hidden,),
        "vision_model.embeddings.patch_embedding.weight": (
            hidden,
            config.num_channels,
            config.patch_size,
            config.patch_size,
        ),
        "vision_model.embeddings.position_embedding.weight": (config.tokens, hidden),
        "vision_model.pre_layrnorm.weight": (hidden,),
        "vision_model.pre_layrnorm.bias": (hidden,),
        "vision_model.post_layernorm.weight": (hidden,),
        "vision_model.post_layernorm.bias": (hidden,),
        "visual_projection.weight": (config.projection_dim, hidden),
    }
    for index in range(config.num_hidden_layers):
        prefix = f"vision_model.encoder.layers.{index}."
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            layout[f"{prefix}self_attn.{projection}.weight"] = (hidden, hidden)
            layout[f"{prefix}self_attn.{projection}.bias"] = (hidden,)
        for norm in ("layer_norm1", "layer_norm2"):
            layout[f"{prefix}{norm}.weight"] = (hidden,)
            layout[f"{prefix}{norm}.bias"] = (hidden,)
        layout[f"{prefix}mlp.fc1.weight"] = (intermediate, hidden)
        layout[f"{prefix}mlp.fc1.bias"] = (intermediate,)
        layout[f"{prefix}mlp.fc2.weight"] = (hidden, intermediate)
        layout[f"{prefix}mlp.fc2.bias"] = (hidden,)
    return layout


def detect_wan21_clip_vision(
    geometries: Mapping[str, TensorGeometry],
) -> ClipVisionConfig:
    """Admit only the exact official F16 ViT-H/14 header."""
    if not geometries:
        raise ClipVisionDetectError("empty state dict header")
    expected = clip_vision_layout()
    problems: list[str] = []
    for key, shape in expected.items():
        found = geometries.get(key)
        if found is None:
            problems.append(f"missing {key}")
            continue
        if found.shape != shape:
            problems.append(f"{key}: expected shape {shape}, found {found.shape}")
        expected_dtype = INT64 if key.endswith("position_ids") else FLOAT16
        if found.dtype != expected_dtype:
            problems.append(
                f"{key}: expected dtype {expected_dtype.name}, found {found.dtype.name}"
            )
    problems.extend(f"unexpected key {key}" for key in sorted(set(geometries) - set(expected)))
    if problems:
        shown = "; ".join(problems[:6])
        if len(problems) > 6:
            shown += f"; and {len(problems) - 6} more"
        raise ClipVisionDetectError(f"not the Wan 2.1 CLIP ViT-H/14 checkpoint: {shown}")
    return WAN21_CLIP_VISION
