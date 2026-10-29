"""Torch-free contract for the standard SD1.5 IP-Adapter.

The layout and attention placement are pinned to ComfyUI commit b78cec87,
ComfyUI_IPAdapter_plus commit a0f451a5, and h94/IP-Adapter's standard
``ip-adapter_sd15.safetensors`` artifact. Other IP-Adapter variants have
different projection and site contracts and are deliberately rejected.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from .clip_vision import ClipVisionConfig, clip_vision_layout
from .conditioning import PayloadReference, PercentRange
from .devices import FLOAT16, FLOAT32, INT64, DType
from .weights import TensorGeometry


class SD15IPAdapterDetectError(ValueError):
    """The header is not the standard SD1.5 IP-Adapter contract."""


@dataclass(frozen=True, slots=True)
class SD15AttentionSite:
    id: str
    width: int
    adapter_index: int

    def __post_init__(self) -> None:
        if type(self.id) is not str or not self.id.endswith(".attn2"):
            raise ValueError("IP-Adapter site id must be a canonical attn2 module path")
        if type(self.width) is not int or self.width <= 0:
            raise ValueError("IP-Adapter site width must be a positive integer")
        if type(self.adapter_index) is not int or self.adapter_index <= 0:
            raise ValueError("IP-Adapter source index must be a positive integer")


SD15_IPADAPTER_SITES: tuple[SD15AttentionSite, ...] = tuple(
    SD15AttentionSite(site_id, width, index)
    for site_id, width, index in (
        ("input_blocks.1.1.transformer_blocks.0.attn2", 320, 1),
        ("input_blocks.2.1.transformer_blocks.0.attn2", 320, 3),
        ("input_blocks.4.1.transformer_blocks.0.attn2", 640, 5),
        ("input_blocks.5.1.transformer_blocks.0.attn2", 640, 7),
        ("input_blocks.7.1.transformer_blocks.0.attn2", 1280, 9),
        ("input_blocks.8.1.transformer_blocks.0.attn2", 1280, 11),
        ("output_blocks.3.1.transformer_blocks.0.attn2", 1280, 13),
        ("output_blocks.4.1.transformer_blocks.0.attn2", 1280, 15),
        ("output_blocks.5.1.transformer_blocks.0.attn2", 1280, 17),
        ("output_blocks.6.1.transformer_blocks.0.attn2", 640, 19),
        ("output_blocks.7.1.transformer_blocks.0.attn2", 640, 21),
        ("output_blocks.8.1.transformer_blocks.0.attn2", 640, 23),
        ("output_blocks.9.1.transformer_blocks.0.attn2", 320, 25),
        ("output_blocks.10.1.transformer_blocks.0.attn2", 320, 27),
        ("output_blocks.11.1.transformer_blocks.0.attn2", 320, 29),
        ("middle_block.1.transformer_blocks.0.attn2", 1280, 31),
    )
)


@dataclass(frozen=True, slots=True)
class SD15IPAdapterConfig:
    family_id: str = field(init=False, default="dinkster.sd15")
    clip_embedding_dim: int = 1024
    token_count: int = 4
    token_dim: int = 768
    sites: tuple[SD15AttentionSite, ...] = SD15_IPADAPTER_SITES

    def __post_init__(self) -> None:
        if (
            self.clip_embedding_dim != 1024
            or self.token_count != 4
            or self.token_dim != 768
            or self.sites != SD15_IPADAPTER_SITES
        ):
            raise ValueError("standard SD1.5 IP-Adapter geometry is immutable")


SD15_IPADAPTER = SD15IPAdapterConfig()
SD15_IPADAPTER_CLIP_VISION = ClipVisionConfig()


def sd15_ipadapter_layout(
    config: SD15IPAdapterConfig = SD15_IPADAPTER,
) -> dict[str, tuple[int, ...]]:
    """Return the official adapter artifact's exact key-to-shape contract."""
    layout = {
        "image_proj.proj.weight": (
            config.token_count * config.token_dim,
            config.clip_embedding_dim,
        ),
        "image_proj.proj.bias": (config.token_count * config.token_dim,),
        "image_proj.norm.weight": (config.token_dim,),
        "image_proj.norm.bias": (config.token_dim,),
    }
    for site in config.sites:
        prefix = f"ip_adapter.{site.adapter_index}"
        layout[f"{prefix}.to_k_ip.weight"] = (site.width, config.token_dim)
        layout[f"{prefix}.to_v_ip.weight"] = (site.width, config.token_dim)
    return layout


def _validate_exact_layout(
    geometries: Mapping[str, TensorGeometry],
    expected: Mapping[str, tuple[int, ...]],
    *,
    expected_dtype: DType,
    description: str,
) -> None:
    problems: list[str] = []
    for key, shape in expected.items():
        found = geometries.get(key)
        if found is None:
            problems.append(f"missing {key}")
        elif found.shape != shape or found.dtype != expected_dtype:
            problems.append(
                f"{key}: expected {expected_dtype.name} {shape}, "
                f"found {found.dtype.name} {found.shape}"
            )
    problems.extend(f"unexpected key {key}" for key in sorted(set(geometries) - set(expected)))
    if problems:
        shown = "; ".join(problems[:6])
        if len(problems) > 6:
            shown += f"; and {len(problems) - 6} more"
        raise SD15IPAdapterDetectError(f"not the {description}: {shown}")


def detect_sd15_ipadapter(
    geometries: Mapping[str, TensorGeometry],
) -> SD15IPAdapterConfig:
    """Admit only h94's 36-tensor standard SD1.5 IP-Adapter artifact."""
    _validate_exact_layout(
        geometries,
        sd15_ipadapter_layout(),
        expected_dtype=FLOAT16,
        description="standard SD1.5 IP-Adapter checkpoint",
    )
    return SD15_IPADAPTER


def detect_sd15_ipadapter_clip_vision(
    geometries: Mapping[str, TensorGeometry],
) -> ClipVisionConfig:
    """Admit only the official F32 CLIP ViT-H/14 projection checkpoint."""
    expected = clip_vision_layout(SD15_IPADAPTER_CLIP_VISION)
    problems: list[str] = []
    for key, shape in expected.items():
        found = geometries.get(key)
        if found is None:
            problems.append(f"missing {key}")
            continue
        expected_dtype = INT64 if key.endswith("position_ids") else FLOAT32
        if found.shape != shape or found.dtype != expected_dtype:
            problems.append(
                f"{key}: expected {shape} {expected_dtype.name}, "
                f"found {found.shape} {found.dtype.name}"
            )
    problems.extend(f"unexpected key {key}" for key in sorted(set(geometries) - set(expected)))
    if problems:
        shown = "; ".join(problems[:6])
        if len(problems) > 6:
            shown += f"; and {len(problems) - 6} more"
        raise SD15IPAdapterDetectError(
            "not the official SD1.5 IP-Adapter CLIP ViT-H/14 checkpoint: " + shown
        )
    return SD15_IPADAPTER_CLIP_VISION


@dataclass(frozen=True, slots=True)
class SD15AttentionContribution:
    """Portable declaration for one immutable standard IP-Adapter addition."""

    model: PayloadReference
    projected_tokens: PayloadReference
    window: PercentRange
    scalar_gain: float
    mask: PayloadReference | None = None
    sites: tuple[SD15AttentionSite, ...] = SD15_IPADAPTER_SITES
    site_gains: tuple[float, ...] = field(default_factory=lambda: (1.0,) * 16)
    lane_gains: tuple[tuple[str, float], ...] = (
        ("positive", 1.0),
        ("negative", 1.0),
        ("empty", 1.0),
    )
    lane_sources: tuple[tuple[str, str], ...] = (
        ("positive", "conditional"),
        ("negative", "unconditional"),
        ("empty", "conditional"),
    )
    placement: str = "attn2-pre-to-out"

    def __post_init__(self) -> None:
        if (
            type(self.model) is not PayloadReference
            or type(self.projected_tokens) is not PayloadReference
        ):
            raise TypeError("IP-Adapter model and projected tokens must be payload references")
        if type(self.window) is not PercentRange:
            raise TypeError("IP-Adapter window must be an exact PercentRange")
        if type(self.scalar_gain) is not float or not math.isfinite(self.scalar_gain):
            raise ValueError("IP-Adapter scalar gain must be a finite exact float")
        if self.mask is not None and type(self.mask) is not PayloadReference:
            raise TypeError("IP-Adapter mask must be a PayloadReference or None")
        if self.sites != SD15_IPADAPTER_SITES:
            raise ValueError("IP-Adapter contribution must cover the canonical 16 SD1.5 sites")
        if len(self.site_gains) != len(self.sites) or any(
            type(gain) is not float or not math.isfinite(gain) for gain in self.site_gains
        ):
            raise ValueError("IP-Adapter site gains must cover every site with finite floats")
        expected_lanes = (("positive", 1.0), ("negative", 1.0), ("empty", 1.0))
        if self.lane_gains != expected_lanes:
            raise ValueError("standard IP-Adapter lane gains are immutable")
        expected_sources = (
            ("positive", "conditional"),
            ("negative", "unconditional"),
            ("empty", "conditional"),
        )
        if self.lane_sources != expected_sources:
            raise ValueError("standard IP-Adapter lane sources are immutable")
        if self.placement != "attn2-pre-to-out":
            raise ValueError("standard IP-Adapter placement is attn2 before to_out")


__all__ = [
    "SD15AttentionContribution",
    "SD15AttentionSite",
    "SD15IPAdapterConfig",
    "SD15IPAdapterDetectError",
    "SD15_IPADAPTER",
    "SD15_IPADAPTER_CLIP_VISION",
    "SD15_IPADAPTER_SITES",
    "detect_sd15_ipadapter",
    "detect_sd15_ipadapter_clip_vision",
    "sd15_ipadapter_layout",
]
