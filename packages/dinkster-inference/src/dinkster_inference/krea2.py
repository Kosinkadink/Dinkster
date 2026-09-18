"""Exact Krea 2 (K2) single-stream DiT profile and fail-closed detection.

The profile follows ComfyUI b78cec879b9460d5cb25228a83a942fb78d2cd24
(``comfy/ldm/krea2/model.py`` SingleStreamDiT constructor defaults, the
``txtfusion.projector.weight`` leg of ``comfy/model_detection.py``, and
the ``Krea2`` entry in ``comfy/supported_models.py``): one shared
transformer stream over concatenated text and patchified image tokens,
AdaLN-single modulation (a shared per-step projection plus per-block
bias), GQA attention with per-head QK RMSNorm and a sigmoid output
gate, SwiGLU MLPs, 3-axis RoPE with theta 1000, and a 12-layer text
fusion adapter that collapses the stacked Qwen3-VL-4B taps
(:mod:`dinkster_inference.krea2_text`) into one 2560-wide sequence.

The reference detection derives its geometry from a handful of anchor
shapes; this module is stricter, the same fail-closed discipline as
:mod:`dinkster_inference.z_image` and :mod:`dinkster_inference.flux2`: only
the single published Krea 2 geometry is admitted (RAW and Turbo share
one tensor layout), required to match the ENTIRE 430-entry key/shape
listing exactly. Quantized repackages carrying extra scale tensors do
not match and are refused, not misread.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .devices import BFLOAT16, FLOAT16, FLOAT32, DType
from .families import EvidenceValue
from .spaces import FluxFlowSigmas
from .weights import TensorGeometry, WeightSource


class Krea2DetectError(ValueError):
    """The geometry mapping is not the published Krea 2 DiT; the
    message names what was found instead."""


# Combined-checkpoint prefix first, bare diffusion file second
# (comfy/model_detection.py any_suffix_in scan @ b78cec87).
_PREFIXES = ("model.diffusion_model.", "")


@dataclass(frozen=True)
class Krea2Config:
    """The exact published Krea 2 DiT profile.

    ``mlp_width`` and ``text_mlp_width`` are the reference's SwiGLU
    widths, ``int(2 * width / 3) * multiplier`` rounded up to a
    multiple of 128; ``rope_axes`` is the constructor's 3-axis split
    of the 128-wide head. Every field is pinned - the constructor
    refuses any other value.
    """

    family_id: str = "dinkster.krea2"
    features: int = 6144
    transformer_blocks: int = 28
    attention_heads: int = 48
    kv_heads: int = 12
    attention_head_dim: int = 128
    mlp_width: int = 16384
    time_width: int = 256
    text_width: int = 2560
    text_layers: int = 12
    text_heads: int = 20
    text_kv_heads: int = 20
    text_mlp_width: int = 6912
    text_fusion_layerwise_blocks: int = 2
    text_fusion_refiner_blocks: int = 2
    latent_channels: int = 16
    patch: tuple[int, int] = (2, 2)
    rope_axes: tuple[int, int, int] = (32, 48, 48)
    rope_theta: float = 1000.0
    rms_norm_eps: float = 1e-5
    latent_id: str = "Wan21"
    latent_dimensions: int = 3
    temporal_downscale: int = 4
    sampling_multiplier: float = 1.0
    sampling_shift: float = 1.15
    inference_dtypes: tuple[DType, ...] = (BFLOAT16, FLOAT16, FLOAT32)
    memory_factor: float = 2.2
    text_encoder_id: str = "Qwen3-VL-4B"

    def __post_init__(self) -> None:
        actual = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        expected = (
            "dinkster.krea2",
            6144,
            28,
            48,
            12,
            128,
            16384,
            256,
            2560,
            12,
            20,
            20,
            6912,
            2,
            2,
            16,
            (2, 2),
            (32, 48, 48),
            1000.0,
            1e-5,
            "Wan21",
            3,
            4,
            1.0,
            1.15,
            (BFLOAT16, FLOAT16, FLOAT32),
            2.2,
            "Qwen3-VL-4B",
        )
        if any(
            type(value) is not type(required) or value != required
            for value, required in zip(actual, expected, strict=True)
        ):
            raise ValueError("Krea2Config only represents the exact published Krea 2 DiT")


KREA2_CONFIG = Krea2Config()
# ModelType.FLUX: ModelSamplingFlux over 10000 timesteps with the
# family's sampling_settings shift (comfy/model_base.py Krea2,
# comfy/supported_models.py Krea2 @ b78cec87).
KREA2_SIGMAS = FluxFlowSigmas(shift=KREA2_CONFIG.sampling_shift)


def _attention_layout(width: int, head_dim: int, kv_width: int) -> dict[str, tuple[int, ...]]:
    return {
        "attn.wq.weight": (width, width),
        "attn.wk.weight": (kv_width, width),
        "attn.wv.weight": (kv_width, width),
        "attn.gate.weight": (width, width),
        "attn.qknorm.qnorm.scale": (head_dim,),
        "attn.qknorm.knorm.scale": (head_dim,),
        "attn.wo.weight": (width, width),
    }


def _swiglu_layout(width: int, mlp_width: int) -> dict[str, tuple[int, ...]]:
    return {
        "mlp.gate.weight": (mlp_width, width),
        "mlp.up.weight": (mlp_width, width),
        "mlp.down.weight": (width, mlp_width),
    }


def krea2_layout() -> Mapping[str, tuple[int, ...]]:
    """Return the immutable 430-entry published Krea 2 DiT layout."""
    config = KREA2_CONFIG
    features = config.features
    text = config.text_width
    patch_values = config.latent_channels * config.patch[0] * config.patch[1]
    layout: dict[str, tuple[int, ...]] = {
        "first.weight": (features, patch_values),
        "first.bias": (features,),
        "tmlp.0.weight": (features, config.time_width),
        "tmlp.0.bias": (features,),
        "tmlp.2.weight": (features, features),
        "tmlp.2.bias": (features,),
        "tproj.1.weight": (6 * features, features),
        "tproj.1.bias": (6 * features,),
        "txtmlp.0.scale": (text,),
        "txtmlp.1.weight": (features, text),
        "txtmlp.1.bias": (features,),
        "txtmlp.3.weight": (features, features),
        "txtmlp.3.bias": (features,),
        "txtfusion.projector.weight": (1, config.text_layers),
        "last.norm.scale": (features,),
        "last.linear.weight": (patch_values, features),
        "last.linear.bias": (patch_values,),
        "last.modulation.lin": (2, features),
    }
    block = {
        "mod.lin": (6 * features,),
        "prenorm.scale": (features,),
        "postnorm.scale": (features,),
        **_attention_layout(
            features, config.attention_head_dim, config.kv_heads * config.attention_head_dim
        ),
        **_swiglu_layout(features, config.mlp_width),
    }
    for index in range(config.transformer_blocks):
        layout.update({f"blocks.{index}.{key}": shape for key, shape in block.items()})
    fusion_block = {
        "prenorm.scale": (text,),
        "postnorm.scale": (text,),
        **_attention_layout(text, config.attention_head_dim, text),
        **_swiglu_layout(text, config.text_mlp_width),
    }
    for root, count in (
        ("layerwise_blocks", config.text_fusion_layerwise_blocks),
        ("refiner_blocks", config.text_fusion_refiner_blocks),
    ):
        for index in range(count):
            layout.update(
                {f"txtfusion.{root}.{index}.{key}": shape for key, shape in fusion_block.items()}
            )
    return MappingProxyType(layout)


def detect_krea2_config(geometries: Mapping[str, TensorGeometry]) -> Krea2Config:
    """Classify a diffusion-model-scoped header (any checkpoint prefix
    already stripped) as the published Krea 2 geometry, or refuse
    loudly. Dtypes are ignored - checkpoints legitimately ship
    bf16/fp16 - but the key set and shapes must match exactly."""
    if not geometries:
        raise Krea2DetectError("empty state dict header")
    if "txtfusion.projector.weight" not in geometries:
        raise Krea2DetectError(
            "not a Krea 2 DiT (no txtfusion.projector.weight);"
            " other single-stream families have their own detectors"
        )
    layout = krea2_layout()
    problems: list[str] = []
    for key, shape in layout.items():
        found = geometries.get(key)
        if found is None:
            problems.append(f"missing {key}")
        elif found.shape != shape:
            problems.append(f"{key}: expected shape {shape}, found {found.shape}")
    problems.extend(f"unexpected key {key}" for key in sorted(set(geometries) - set(layout)))
    if problems:
        shown = "; ".join(problems[:6])
        more = len(problems) - 6
        if more > 0:
            shown += f"; and {more} more"
        raise Krea2DetectError(f"geometry does not match the published Krea 2 layout: {shown}")
    return KREA2_CONFIG


@dataclass(frozen=True)
class Krea2Evidence:
    config: Krea2Config
    key_prefix: str
    matched_keys: tuple[str, ...]
    fields: Mapping[str, EvidenceValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "matched_keys", tuple(self.matched_keys))
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))


def detect_krea2(source: WeightSource) -> Krea2Evidence | None:
    """Family-detector seam over :func:`detect_krea2_config`: scan the
    known checkpoint prefixes and return evidence for the first exact
    Krea 2 match, or None (never raises for foreign checkpoints)."""
    all_keys = tuple(source.keys())
    for prefix in _PREFIXES:
        scoped = {
            key.removeprefix(prefix): source.entry(key).geometry
            for key in all_keys
            if key.startswith(prefix)
        }
        if not scoped:
            continue
        try:
            config = detect_krea2_config(scoped)
        except Krea2DetectError:
            continue
        matched = tuple(sorted(prefix + key for key in scoped))
        return Krea2Evidence(
            config=config,
            key_prefix=prefix,
            matched_keys=matched,
            fields={
                "features": config.features,
                "key_prefix": prefix,
                "kv_heads": config.kv_heads,
                "sampling_shift": config.sampling_shift,
                "text_layers": config.text_layers,
                "text_width": config.text_width,
                "transformer_blocks": config.transformer_blocks,
            },
        )
    return None


__all__ = [
    "KREA2_CONFIG",
    "KREA2_SIGMAS",
    "Krea2Config",
    "Krea2DetectError",
    "Krea2Evidence",
    "detect_krea2",
    "detect_krea2_config",
    "krea2_layout",
]
