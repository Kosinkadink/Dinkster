"""Torch-free Chroma and Chroma Radiance architecture contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeVar

from .devices import BFLOAT16, FLOAT16, FLOAT32, DType
from .families import DetectionEvidence
from .quantization import QuantizationError, split_quantization
from .weights import TensorGeometry, WeightSource

_V = TypeVar("_V")

CHROMA_FAMILY_ID = "dinkster.chroma"
CHROMA_RADIANCE_FAMILY_ID = "dinkster.chroma_radiance"
CHROMA_AXES_DIM = (16, 56, 56)
CHROMA_THETA = 10000
CHROMA_MLP_RATIO = 4.0
CHROMA_CONTEXT_DIM = 4096
CHROMA_HIDDEN_SIZE = 3072
CHROMA_DEPTH = 19
CHROMA_SINGLE_DEPTH = 38
CHROMA_NUM_HEADS = 24
CHROMA_APPROXIMATOR_INPUT_DIM = 64
CHROMA_APPROXIMATOR_HIDDEN_DIM = 5120
CHROMA_APPROXIMATOR_LAYERS = 5
CHROMA_MODULATION_EMBED_DIM = 32
CHROMA_RADIANCE_CHANNELS = 3
CHROMA_RADIANCE_NERF_HIDDEN = 64
CHROMA_RADIANCE_NERF_RATIO = 4
CHROMA_RADIANCE_NERF_DEPTH = 4
CHROMA_RADIANCE_MAX_FREQS = 8


class ChromaDetectError(ValueError):
    """The source is not one exact supported Chroma family layout."""


@dataclass(frozen=True)
class ChromaConfig:
    hidden_size: int
    depth: int
    depth_single_blocks: int
    num_heads: int
    context_in_dim: int = CHROMA_CONTEXT_DIM
    latent_channels: int = 16
    patch_size: int = 2
    out_channels: int = 64
    axes_dim: tuple[int, ...] = CHROMA_AXES_DIM
    theta: int = CHROMA_THETA
    mlp_ratio: float = CHROMA_MLP_RATIO
    qkv_bias: bool = True
    approximator_input_dim: int = CHROMA_APPROXIMATOR_INPUT_DIM
    approximator_hidden_dim: int = CHROMA_APPROXIMATOR_HIDDEN_DIM
    approximator_layers: int = CHROMA_APPROXIMATOR_LAYERS

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "depth",
            "depth_single_blocks",
            "num_heads",
            "context_in_dim",
            "latent_channels",
            "patch_size",
            "out_channels",
            "theta",
            "approximator_input_dim",
            "approximator_hidden_dim",
            "approximator_layers",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if sum(self.axes_dim) != self.head_dim:
            raise ValueError("axes_dim must sum to the per-head dimension")
        if self.out_channels != self.latent_channels * self.patch_size**2:
            raise ValueError("Chroma output width must reconstruct its patchified latent")

    @property
    def family_id(self) -> str:
        return CHROMA_FAMILY_ID

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def mlp_hidden_dim(self) -> int:
        return int(self.hidden_size * self.mlp_ratio)

    @property
    def modulation_count(self) -> int:
        return 3 * self.depth_single_blocks + 12 * self.depth + 2


@dataclass(frozen=True)
class ChromaRadianceConfig:
    hidden_size: int
    depth: int
    depth_single_blocks: int
    num_heads: int
    patch_size: int
    context_in_dim: int = CHROMA_CONTEXT_DIM
    axes_dim: tuple[int, ...] = CHROMA_AXES_DIM
    theta: int = CHROMA_THETA
    mlp_ratio: float = CHROMA_MLP_RATIO
    qkv_bias: bool = True
    approximator_input_dim: int = CHROMA_APPROXIMATOR_INPUT_DIM
    approximator_hidden_dim: int = CHROMA_APPROXIMATOR_HIDDEN_DIM
    approximator_layers: int = CHROMA_APPROXIMATOR_LAYERS
    nerf_hidden_size: int = CHROMA_RADIANCE_NERF_HIDDEN
    nerf_mlp_ratio: int = CHROMA_RADIANCE_NERF_RATIO
    nerf_depth: int = CHROMA_RADIANCE_NERF_DEPTH
    nerf_max_freqs: int = CHROMA_RADIANCE_MAX_FREQS
    nerf_tile_size: int = 512
    nerf_final_head_type: str = "linear"
    use_x0: bool = False
    use_sequential_txt_ids: bool = False

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "depth",
            "depth_single_blocks",
            "num_heads",
            "patch_size",
            "context_in_dim",
            "theta",
            "approximator_input_dim",
            "approximator_hidden_dim",
            "approximator_layers",
            "nerf_hidden_size",
            "nerf_mlp_ratio",
            "nerf_depth",
            "nerf_max_freqs",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if sum(self.axes_dim) != self.head_dim:
            raise ValueError("axes_dim must sum to the per-head dimension")
        if self.nerf_tile_size < 0:
            raise ValueError("nerf_tile_size must be non-negative")
        if self.nerf_final_head_type not in ("linear", "conv"):
            raise ValueError("nerf_final_head_type must be 'linear' or 'conv'")

    @property
    def family_id(self) -> str:
        return CHROMA_RADIANCE_FAMILY_ID

    @property
    def latent_channels(self) -> int:
        return CHROMA_RADIANCE_CHANNELS

    @property
    def out_channels(self) -> int:
        return CHROMA_RADIANCE_CHANNELS

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def mlp_hidden_dim(self) -> int:
        return int(self.hidden_size * self.mlp_ratio)

    @property
    def modulation_count(self) -> int:
        return 3 * self.depth_single_blocks + 12 * self.depth + 2


ChromaFamilyConfig = ChromaConfig | ChromaRadianceConfig


def normalize_chroma_keys(mapping: Mapping[str, _V]) -> dict[str, _V]:
    """Normalize Chroma's legacy approximator wrapper and RMS scale spelling."""
    normalized: dict[str, _V] = {}
    sources: dict[str, str] = {}
    for source_key, value in mapping.items():
        key = source_key.replace("distilled_guidance_layer.0.", "distilled_guidance_layer.", 1)
        if key.endswith(".scale"):
            key = key[: -len(".scale")] + ".weight"
        if key in normalized:
            raise ValueError(
                f"Chroma keys {sources[key]!r} and {source_key!r} both normalize to {key!r}"
            )
        normalized[key] = value
        sources[key] = source_key
    return normalized


def _linear(layout: dict[str, tuple[int, ...]], prefix: str, out: int, inner: int) -> None:
    layout[f"{prefix}.weight"] = (out, inner)
    layout[f"{prefix}.bias"] = (out,)


def _flux_blocks(layout: dict[str, tuple[int, ...]], config: ChromaFamilyConfig) -> None:
    hidden = config.hidden_size
    mlp = config.mlp_hidden_dim
    head = config.head_dim
    for index in range(config.depth):
        for stream in ("img", "txt"):
            prefix = f"double_blocks.{index}.{stream}"
            _linear(layout, f"{prefix}_attn.qkv", 3 * hidden, hidden)
            layout[f"{prefix}_attn.norm.query_norm.weight"] = (head,)
            layout[f"{prefix}_attn.norm.key_norm.weight"] = (head,)
            _linear(layout, f"{prefix}_attn.proj", hidden, hidden)
            _linear(layout, f"{prefix}_mlp.0", mlp, hidden)
            _linear(layout, f"{prefix}_mlp.2", hidden, mlp)
    for index in range(config.depth_single_blocks):
        prefix = f"single_blocks.{index}"
        _linear(layout, f"{prefix}.linear1", 3 * hidden + mlp, hidden)
        _linear(layout, f"{prefix}.linear2", hidden, hidden + mlp)
        layout[f"{prefix}.norm.query_norm.weight"] = (head,)
        layout[f"{prefix}.norm.key_norm.weight"] = (head,)


def _approximator(layout: dict[str, tuple[int, ...]], config: ChromaFamilyConfig) -> None:
    prefix = "distilled_guidance_layer"
    hidden = config.approximator_hidden_dim
    _linear(layout, f"{prefix}.in_proj", hidden, config.approximator_input_dim)
    for index in range(config.approximator_layers):
        _linear(layout, f"{prefix}.layers.{index}.in_layer", hidden, hidden)
        _linear(layout, f"{prefix}.layers.{index}.out_layer", hidden, hidden)
        layout[f"{prefix}.norms.{index}.weight"] = (hidden,)
    _linear(layout, f"{prefix}.out_proj", config.hidden_size, hidden)


def chroma_layout(config: ChromaConfig) -> dict[str, tuple[int, ...]]:
    layout: dict[str, tuple[int, ...]] = {}
    _linear(layout, "img_in", config.hidden_size, config.latent_channels * config.patch_size**2)
    _linear(layout, "txt_in", config.hidden_size, config.context_in_dim)
    _approximator(layout, config)
    _flux_blocks(layout, config)
    _linear(layout, "final_layer.linear", config.out_channels, config.hidden_size)
    return layout


def chroma_radiance_layout(config: ChromaRadianceConfig) -> dict[str, tuple[int, ...]]:
    layout: dict[str, tuple[int, ...]] = {
        "img_in_patch.weight": (
            config.hidden_size,
            config.latent_channels,
            config.patch_size,
            config.patch_size,
        ),
        "img_in_patch.bias": (config.hidden_size,),
    }
    _linear(layout, "txt_in", config.hidden_size, config.context_in_dim)
    _approximator(layout, config)
    _flux_blocks(layout, config)
    _linear(
        layout,
        "nerf_image_embedder.embedder.0",
        config.nerf_hidden_size,
        config.latent_channels + config.nerf_max_freqs**2,
    )
    generated = 3 * config.nerf_hidden_size**2 * config.nerf_mlp_ratio
    for index in range(config.nerf_depth):
        _linear(layout, f"nerf_blocks.{index}.param_generator", generated, config.hidden_size)
        layout[f"nerf_blocks.{index}.norm.weight"] = (config.nerf_hidden_size,)
    final = "nerf_final_layer_conv" if config.nerf_final_head_type == "conv" else "nerf_final_layer"
    layout[f"{final}.norm.weight"] = (config.nerf_hidden_size,)
    if config.nerf_final_head_type == "conv":
        layout[f"{final}.conv.weight"] = (
            config.out_channels,
            config.nerf_hidden_size,
            3,
            3,
        )
        layout[f"{final}.conv.bias"] = (config.out_channels,)
    else:
        _linear(layout, f"{final}.linear", config.out_channels, config.nerf_hidden_size)
    if config.use_x0:
        layout["__x0__"] = (0,)
    if config.use_sequential_txt_ids:
        layout["__sequential__"] = (0,)
    return layout


def _count_blocks(keys: frozenset[str], prefix: str) -> int:
    count = 0
    while any(key.startswith(f"{prefix}{count}.") for key in keys):
        count += 1
    return count


def _required(geometries: Mapping[str, TensorGeometry], key: str, rank: int) -> TensorGeometry:
    geometry = geometries.get(key)
    if geometry is None:
        raise ChromaDetectError(f"missing {key}")
    if len(geometry.shape) != rank:
        raise ChromaDetectError(f"{key} has rank {len(geometry.shape)}, expected {rank}")
    return geometry


def detect_chroma_config(geometries: Mapping[str, TensorGeometry]) -> ChromaFamilyConfig:
    try:
        normalized = normalize_chroma_keys(geometries)
    except ValueError as error:
        raise ChromaDetectError(str(error)) from error
    keys = frozenset(normalized)
    marker = _required(normalized, "distilled_guidance_layer.norms.0.weight", 1)
    if marker.shape != (CHROMA_APPROXIMATOR_HIDDEN_DIM,):
        raise ChromaDetectError("distilled guidance marker has an unsupported width")
    txt = _required(normalized, "txt_in.weight", 2)
    hidden_size, context_in_dim = txt.shape
    if context_in_dim != CHROMA_CONTEXT_DIM:
        raise ChromaDetectError("Chroma requires 4096-wide T5 conditioning")
    if hidden_size != CHROMA_HIDDEN_SIZE:
        raise ChromaDetectError("Chroma requires the official 3072-wide transformer")
    depth = _count_blocks(keys, "double_blocks.")
    single_depth = _count_blocks(keys, "single_blocks.")
    if (depth, single_depth) != (CHROMA_DEPTH, CHROMA_SINGLE_DEPTH):
        raise ChromaDetectError("Chroma requires the official 19+38 block layout")
    num_heads = CHROMA_NUM_HEADS
    if "nerf_blocks.0.norm.weight" in keys:
        patch = _required(normalized, "img_in_patch.weight", 4)
        if patch.shape[1] != CHROMA_RADIANCE_CHANNELS or patch.shape[2] != patch.shape[3]:
            raise ChromaDetectError("Radiance patch embedding must consume square RGB patches")
        config: ChromaFamilyConfig = ChromaRadianceConfig(
            hidden_size=hidden_size,
            depth=depth,
            depth_single_blocks=single_depth,
            num_heads=num_heads,
            context_in_dim=context_in_dim,
            patch_size=patch.shape[3],
            nerf_final_head_type=(
                "conv" if "nerf_final_layer_conv.norm.weight" in keys else "linear"
            ),
            use_x0="__x0__" in keys,
            use_sequential_txt_ids="__sequential__" in keys,
        )
        expected = chroma_radiance_layout(config)
    else:
        img = _required(normalized, "img_in.weight", 2)
        if img.shape != (hidden_size, 64):
            raise ChromaDetectError("Chroma image projection must consume 64-wide patches")
        config = ChromaConfig(
            hidden_size=hidden_size,
            depth=depth,
            depth_single_blocks=single_depth,
            num_heads=num_heads,
            context_in_dim=context_in_dim,
        )
        expected = chroma_layout(config)
    if set(normalized) != set(expected):
        missing = sorted(set(expected) - set(normalized))
        extra = sorted(set(normalized) - set(expected))
        detail: list[str] = []
        if missing:
            detail.append("missing " + ", ".join(missing[:3]))
        if extra:
            detail.append("unexpected " + ", ".join(extra[:3]))
        raise ChromaDetectError("layout mismatch: " + "; ".join(detail))
    wrong = sorted(key for key, shape in expected.items() if normalized[key].shape != shape)
    if wrong:
        raise ChromaDetectError(f"layout shape mismatch at {wrong[0]}")
    if any(geometry.dtype.kind != "float" for geometry in normalized.values()):
        raise ChromaDetectError("Chroma weights require floating-point storage")
    return config


def detect_chroma(source: WeightSource) -> DetectionEvidence | None:
    source_geometries = {key: source.entry(key).geometry for key in source.keys()}
    for prefix in ("", "model.diffusion_model."):
        try:
            geometries = split_quantization(
                source_geometries,
                source.metadata(),
                prefix=prefix,
            ).architecture
        except QuantizationError:
            return None
        marker_keys = (
            "distilled_guidance_layer.norms.0.weight",
            "distilled_guidance_layer.norms.0.scale",
            "distilled_guidance_layer.0.norms.0.weight",
            "distilled_guidance_layer.0.norms.0.scale",
        )
        if not any(key in geometries for key in marker_keys):
            continue
        try:
            config = detect_chroma_config(geometries)
        except ChromaDetectError:
            return None
        radiance = isinstance(config, ChromaRadianceConfig)
        matched = next(key for key in marker_keys if key in geometries)
        if radiance:
            matched = next(
                key
                for key in ("nerf_blocks.0.norm.weight", "nerf_blocks.0.norm.scale")
                if key in geometries
            )
        return DetectionEvidence(
            family_id=config.family_id,
            matched_keys=(prefix + matched,),
            fields={
                "key_prefix": prefix,
                "hidden_size": config.hidden_size,
                "depth": config.depth,
                "depth_single_blocks": config.depth_single_blocks,
                "patch_size": config.patch_size,
                "radiance": radiance,
            },
        )
    return None


CHROMA_INFERENCE_DTYPES: frozenset[DType] = frozenset({FLOAT16, BFLOAT16, FLOAT32})


__all__ = [
    "CHROMA_FAMILY_ID",
    "CHROMA_INFERENCE_DTYPES",
    "CHROMA_RADIANCE_FAMILY_ID",
    "ChromaConfig",
    "ChromaDetectError",
    "ChromaFamilyConfig",
    "ChromaRadianceConfig",
    "chroma_layout",
    "chroma_radiance_layout",
    "detect_chroma",
    "detect_chroma_config",
    "normalize_chroma_keys",
]
