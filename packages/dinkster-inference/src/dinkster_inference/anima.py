"""Torch-free Anima profile and fail-closed header detection.

The profile follows ComfyUI 82f839f5e737d8bfce480872ba05e5a430f2526f
(comfy/ldm/anima/model.py, comfy/ldm/cosmos/predict2.py,
comfy/model_detection.py - the same baseline the executed goldens
pin): the official 2B checkpoint is a Cosmos Predict2 MiniTrainDIT
with a six-block LLM adapter mapping Qwen3-0.6B hidden states onto T5
vocabulary positions. Exactly that geometry is accepted; tunes with
extra blocks are refused until supported with evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .devices import BFLOAT16, FLOAT16, FLOAT32, DType
from .families import EvidenceValue
from .weights import TensorGeometry, WeightSource

_PREFIXES = ("model.diffusion_model.", "net.", "")
_BLOCK_INDICES = frozenset(range(28))
_ADAPTER_BLOCK_INDICES = frozenset(range(6))
# suffix -> exact required geometry, per the 2B MiniTrainDIT layout.
# 68 = (16 latent + 1 padding-mask channel) * 1 * 2 * 2 patch volume;
# 64 = 16 output latent channels * 1 * 2 * 2; 6144 = 3 * 2048 AdaLN
# shift/scale/gate rows; 4096 = 2 * 2048 final-layer shift/scale rows.
_TOP_LEVEL: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {
        "x_embedder.proj.1.weight": (2048, 68),
        "t_embedder.1.linear_1.weight": (2048, 2048),
        "t_embedder.1.linear_2.weight": (6144, 2048),
        "t_embedding_norm.weight": (2048,),
        "final_layer.linear.weight": (64, 2048),
        "final_layer.adaln_modulation.1.weight": (256, 2048),
        "final_layer.adaln_modulation.2.weight": (4096, 256),
        "llm_adapter.embed.weight": (32128, 1024),
        "llm_adapter.out_proj.weight": (1024, 1024),
        "llm_adapter.norm.weight": (1024,),
    }
)
_BLOCK_SUFFIXES: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {
        # ComfyUI's Predict2 discriminator key; 8192 = 4 * 2048 MLP rows.
        "mlp.layer1.weight": (8192, 2048),
        "self_attn.q_norm.weight": (128,),
        "cross_attn.k_proj.weight": (2048, 1024),
    }
)
_ADAPTER_BLOCK_SUFFIXES: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {
        "self_attn.q_norm.weight": (64,),
        "cross_attn.q_proj.weight": (1024, 1024),
    }
)


@dataclass(frozen=True)
class AnimaConfig:
    """The exact, non-runnable Anima 2B architecture profile."""

    family_id: str = "dinkster.anima"
    blocks: int = 28
    hidden_width: int = 2048
    attention_heads: int = 16
    attention_head_dim: int = 128
    context_width: int = 1024
    adaln_lora_dim: int = 256
    patchified_input_channels: int = 68
    output_latent_channels: int = 16
    patch: tuple[int, int, int] = (1, 2, 2)
    adapter_blocks: int = 6
    adapter_width: int = 1024
    adapter_heads: int = 16
    adapter_head_dim: int = 64
    adapter_source_width: int = 1024
    adapter_vocabulary: int = 32128
    latent_id: str = "Wan21"
    latent_channels: int = 16
    latent_dimensions: int = 3
    temporal_downscale: int = 4
    sampling_multiplier: float = 1.0
    sampling_shift: float = 3.0
    inference_dtypes: tuple[DType, DType, DType] = (BFLOAT16, FLOAT16, FLOAT32)
    memory_factor: float = 1.0
    text_encoder_id: str = "Qwen3-0.6B"

    def __post_init__(self) -> None:
        actual = (
            self.family_id,
            self.blocks,
            self.hidden_width,
            self.attention_heads,
            self.attention_head_dim,
            self.context_width,
            self.adaln_lora_dim,
            self.patchified_input_channels,
            self.output_latent_channels,
            self.patch,
            self.adapter_blocks,
            self.adapter_width,
            self.adapter_heads,
            self.adapter_head_dim,
            self.adapter_source_width,
            self.adapter_vocabulary,
            self.latent_id,
            self.latent_channels,
            self.latent_dimensions,
            self.temporal_downscale,
            self.sampling_multiplier,
            self.sampling_shift,
            self.inference_dtypes,
            self.memory_factor,
            self.text_encoder_id,
        )
        expected = (
            "dinkster.anima",
            28,
            2048,
            16,
            128,
            1024,
            256,
            68,
            16,
            (1, 2, 2),
            6,
            1024,
            16,
            64,
            1024,
            32128,
            "Wan21",
            16,
            3,
            4,
            1.0,
            3.0,
            (BFLOAT16, FLOAT16, FLOAT32),
            1.0,
            "Qwen3-0.6B",
        )
        if any(
            type(value) is not type(required) or value != required
            for value, required in zip(actual, expected, strict=True)
        ) or any(type(value) is not int for value in self.patch):
            raise ValueError("AnimaConfig only represents the exact supported profile")


ANIMA_CONFIG = AnimaConfig()


@dataclass(frozen=True)
class AnimaEvidence:
    """Immutable deterministic evidence for the exact Anima profile."""

    config: AnimaConfig
    key_prefix: str
    matched_keys: tuple[str, ...]
    fields: Mapping[str, EvidenceValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "matched_keys", tuple(self.matched_keys))
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))


def _entry_geometry(source: WeightSource, keys: frozenset[str], key: str) -> TensorGeometry | None:
    if key not in keys:
        return None
    try:
        return source.entry(key).geometry
    except KeyError:
        return None


def _block_indices(
    keys: frozenset[str], root: str, allowed: frozenset[int]
) -> frozenset[int] | None:
    allowed_text = frozenset(str(index) for index in allowed)
    indices: set[int] = set()
    for key in keys:
        if not key.startswith(root):
            continue
        text, separator, remainder = key[len(root) :].partition(".")
        if (
            not separator
            or not remainder
            or not text.isascii()
            or not text.isdecimal()
            or len(text) > 1
            and text.startswith("0")
            or text not in allowed_text
        ):
            return None
        indices.add(int(text))
    return frozenset(indices)


def _matches_prefix(
    source: WeightSource, keys: frozenset[str], prefix: str
) -> tuple[str, ...] | None:
    if _block_indices(keys, prefix + "blocks.", _BLOCK_INDICES) != _BLOCK_INDICES:
        return None
    if (
        _block_indices(keys, prefix + "llm_adapter.blocks.", _ADAPTER_BLOCK_INDICES)
        != _ADAPTER_BLOCK_INDICES
    ):
        return None

    matched: list[str] = []
    for suffix, shape in _TOP_LEVEL.items():
        key = prefix + suffix
        geometry = _entry_geometry(source, keys, key)
        if geometry is None or geometry.shape != shape:
            return None
        matched.append(key)

    for index in sorted(_BLOCK_INDICES):
        for suffix, shape in _BLOCK_SUFFIXES.items():
            key = prefix + f"blocks.{index}.{suffix}"
            geometry = _entry_geometry(source, keys, key)
            if geometry is None or geometry.shape != shape:
                return None
            matched.append(key)

    for index in sorted(_ADAPTER_BLOCK_INDICES):
        for suffix, shape in _ADAPTER_BLOCK_SUFFIXES.items():
            key = prefix + f"llm_adapter.blocks.{index}.{suffix}"
            geometry = _entry_geometry(source, keys, key)
            if geometry is None or geometry.shape != shape:
                return None
            matched.append(key)

    return tuple(sorted(matched))


def detect_anima(source: WeightSource) -> AnimaEvidence | None:
    """Detect only the exact Anima 2B header, or fail closed."""

    keys = frozenset(source.keys())
    for prefix in _PREFIXES:
        matched_keys = _matches_prefix(source, keys, prefix)
        if matched_keys is None:
            continue
        config = ANIMA_CONFIG
        return AnimaEvidence(
            config=config,
            key_prefix=prefix,
            matched_keys=matched_keys,
            fields={
                "adaln_lora_dim": config.adaln_lora_dim,
                "adapter_blocks": config.adapter_blocks,
                "adapter_head_dim": config.adapter_head_dim,
                "adapter_heads": config.adapter_heads,
                "adapter_vocabulary": config.adapter_vocabulary,
                "adapter_width": config.adapter_width,
                "attention_head_dim": config.attention_head_dim,
                "attention_heads": config.attention_heads,
                "blocks": config.blocks,
                "context_width": config.context_width,
                "hidden_width": config.hidden_width,
                "key_prefix": prefix,
                "output_latent_channels": config.output_latent_channels,
                "patch": "1x2x2",
                "patchified_input_channels": config.patchified_input_channels,
            },
        )
    return None


def anima_layout(config: AnimaConfig = ANIMA_CONFIG) -> dict[str, tuple[int, ...]]:
    """The exact key -> shape listing of one Anima state dict (the
    MiniTrainDIT backbone at the top level plus the ``llm_adapter.*``
    subtree, per comfy/ldm/cosmos/predict2.py and
    comfy/ldm/anima/model.py @ 82f839f5). Both MLP expansions are the
    fixed 4x profile ratio. Arithmetic consistency of the fields is
    checked so a reduced test geometry cannot silently list an
    impossible module."""
    patch_volume = 1
    for extent in config.patch:
        patch_volume *= extent
    width = config.hidden_width
    if width != config.attention_heads * config.attention_head_dim:
        raise ValueError("hidden_width must equal attention_heads * attention_head_dim")
    if config.patchified_input_channels != (config.latent_channels + 1) * patch_volume:
        raise ValueError(
            "patchified_input_channels must cover the latent channels plus one"
            " padding-mask channel across the patch volume"
        )
    adapter = config.adapter_width
    if adapter != config.adapter_heads * config.adapter_head_dim:
        raise ValueError("adapter_width must equal adapter_heads * adapter_head_dim")
    if config.context_width != adapter:
        raise ValueError(
            "context_width must equal adapter_width: the adapter output feeds"
            " cross-attention without an input projection"
        )

    head_dim = config.attention_head_dim
    lora = config.adaln_lora_dim
    context = config.context_width
    layout: dict[str, tuple[int, ...]] = {
        "x_embedder.proj.1.weight": (width, config.patchified_input_channels),
        "t_embedder.1.linear_1.weight": (width, width),
        "t_embedder.1.linear_2.weight": (3 * width, width),
        "t_embedding_norm.weight": (width,),
        "final_layer.linear.weight": (patch_volume * config.output_latent_channels, width),
        "final_layer.adaln_modulation.1.weight": (lora, width),
        "final_layer.adaln_modulation.2.weight": (2 * width, lora),
    }
    for index in range(config.blocks):
        root = f"blocks.{index}."
        for name, kv_width in (("self_attn", width), ("cross_attn", context)):
            layout[root + f"{name}.q_proj.weight"] = (width, width)
            layout[root + f"{name}.q_norm.weight"] = (head_dim,)
            layout[root + f"{name}.k_proj.weight"] = (width, kv_width)
            layout[root + f"{name}.k_norm.weight"] = (head_dim,)
            layout[root + f"{name}.v_proj.weight"] = (width, kv_width)
            layout[root + f"{name}.output_proj.weight"] = (width, width)
        layout[root + "mlp.layer1.weight"] = (4 * width, width)
        layout[root + "mlp.layer2.weight"] = (width, 4 * width)
        for name in ("self_attn", "cross_attn", "mlp"):
            layout[root + f"adaln_modulation_{name}.1.weight"] = (lora, width)
            layout[root + f"adaln_modulation_{name}.2.weight"] = (3 * width, lora)
    layout["llm_adapter.embed.weight"] = (config.adapter_vocabulary, adapter)
    for index in range(config.adapter_blocks):
        root = f"llm_adapter.blocks.{index}."
        for name, kv_width in (
            ("self_attn", adapter),
            ("cross_attn", config.adapter_source_width),
        ):
            layout[root + f"norm_{name}.weight"] = (adapter,)
            layout[root + f"{name}.q_proj.weight"] = (adapter, adapter)
            layout[root + f"{name}.q_norm.weight"] = (config.adapter_head_dim,)
            layout[root + f"{name}.k_proj.weight"] = (adapter, kv_width)
            layout[root + f"{name}.k_norm.weight"] = (config.adapter_head_dim,)
            layout[root + f"{name}.v_proj.weight"] = (adapter, kv_width)
            layout[root + f"{name}.o_proj.weight"] = (adapter, adapter)
        layout[root + "norm_mlp.weight"] = (adapter,)
        layout[root + "mlp.0.weight"] = (4 * adapter, adapter)
        layout[root + "mlp.0.bias"] = (4 * adapter,)
        layout[root + "mlp.2.weight"] = (adapter, 4 * adapter)
        layout[root + "mlp.2.bias"] = (adapter,)
    layout["llm_adapter.out_proj.weight"] = (context, adapter)
    layout["llm_adapter.out_proj.bias"] = (context,)
    layout["llm_adapter.norm.weight"] = (context,)
    return layout


__all__ = [
    "ANIMA_CONFIG",
    "AnimaConfig",
    "AnimaEvidence",
    "anima_layout",
    "detect_anima",
]
