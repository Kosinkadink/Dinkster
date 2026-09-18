"""Flux2 diffusion transformer: exact configs and fail-closed detection.

Flux2 is the reference's global-modulation Flux variant
(comfy/model_detection.py flux branch, the
``double_stream_modulation_img.lin.weight`` leg @ b78cec87): shared
modulation modules computed once per forward, SiLU-gated packed MLPs,
bias-free linears throughout, 4 RoPE axes of 32 with theta 2000, text
positions on axis 3, patch size 1 over the 128-channel packed latent
(the 2x2 pixel-shuffle packing lives in the Flux2 VAE, not the DiT).

Unlike classic Flux, detection here does not derive a config from a
geometry scan: only the three published Flux2 geometries are admitted
(dev, Klein 9B, Klein 4B - base and guidance-distilled Klein releases
share one tensor layout), each required to match the ENTIRE key/shape
listing exactly, the same fail-closed discipline as
:mod:`dinkster_inference.z_image`. RMSNorm ``.scale`` spelling is
normalized internally (:func:`~dinkster_inference.flux.normalize_flux_keys`).
Quantized repackages carrying extra scale tensors (the Comfy-Org
fp8-mixed dev export) do not match and are refused, not misread.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .families import EvidenceValue
from .flux import FluxConfig, flux_layout, normalize_flux_keys
from .weights import TensorGeometry, WeightSource


class Flux2DetectError(ValueError):
    """The geometry mapping is not a published Flux2 checkpoint; the
    message names what was found instead."""


#: Flux2 constants the reference detection pins rather than derives
#: (comfy/model_detection.py flux branch, the flux2 leg @ b78cec87).
FLUX2_AXES_DIM = (32, 32, 32, 32)
FLUX2_THETA = 2000
FLUX2_PATCH_SIZE = 1
FLUX2_MLP_RATIO = 3.0
FLUX2_LATENT_CHANNELS = 128
FLUX2_TXT_IDS_DIMS = (3,)


def _flux2_config(
    *,
    context_in_dim: int,
    hidden_size: int,
    depth: int,
    depth_single_blocks: int,
    guidance_embed: bool,
) -> FluxConfig:
    return FluxConfig(
        in_channels=FLUX2_LATENT_CHANNELS,
        out_channels=FLUX2_LATENT_CHANNELS,
        vec_in_dim=None,
        context_in_dim=context_in_dim,
        hidden_size=hidden_size,
        depth=depth,
        depth_single_blocks=depth_single_blocks,
        num_heads=hidden_size // sum(FLUX2_AXES_DIM),
        axes_dim=FLUX2_AXES_DIM,
        theta=FLUX2_THETA,
        patch_size=FLUX2_PATCH_SIZE,
        mlp_ratio=FLUX2_MLP_RATIO,
        qkv_bias=False,
        guidance_embed=guidance_embed,
        txt_ids_dims=FLUX2_TXT_IDS_DIMS,
        global_modulation=True,
        mlp_silu_act=True,
        ops_bias=False,
    )


#: Flux2 dev: guidance-distilled, Mistral3-Small conditioning
#: (3 x 5120 concatenated hidden states).
FLUX2_DEV_CONFIG = _flux2_config(
    context_in_dim=15360,
    hidden_size=6144,
    depth=8,
    depth_single_blocks=48,
    guidance_embed=True,
)

#: Flux2 Klein 9B: Qwen3-8B conditioning (3 x 4096); the base and
#: guidance-distilled releases share this layout.
FLUX2_KLEIN_9B_CONFIG = _flux2_config(
    context_in_dim=12288,
    hidden_size=4096,
    depth=8,
    depth_single_blocks=24,
    guidance_embed=False,
)

#: Flux2 Klein 4B: Qwen3-4B conditioning (3 x 2560); the base and
#: guidance-distilled releases share this layout.
FLUX2_KLEIN_4B_CONFIG = _flux2_config(
    context_in_dim=7680,
    hidden_size=3072,
    depth=5,
    depth_single_blocks=20,
    guidance_embed=False,
)

KNOWN_FLUX2_CONFIGS: Mapping[str, FluxConfig] = MappingProxyType(
    {
        "flux2_dev": FLUX2_DEV_CONFIG,
        "flux2_klein_9b": FLUX2_KLEIN_9B_CONFIG,
        "flux2_klein_4b": FLUX2_KLEIN_4B_CONFIG,
    }
)

# Combined-checkpoint prefix first, bare diffusion file second
# (comfy/model_detection.py any_suffix_in scan @ b78cec87).
_PREFIXES = ("model.diffusion_model.", "")


def flux2_layout(config: FluxConfig) -> dict[str, tuple[int, ...]]:
    """The exact key -> shape listing of one Flux2 geometry in
    canonical ``.weight`` RMSNorm spelling; refuses non-Flux2 configs
    so callers cannot silently list a classic layout through the
    Flux2 seam."""
    if not (config.global_modulation and config.mlp_silu_act) or config.ops_bias:
        raise ValueError("flux2_layout requires a Flux2 config (see KNOWN_FLUX2_CONFIGS)")
    return flux_layout(config)


def detect_flux2_config(
    geometries: Mapping[str, TensorGeometry],
) -> FluxConfig:
    """Classify a diffusion-model-scoped header (any checkpoint prefix
    already stripped) as one of the published Flux2 geometries, or
    refuse loudly. Dtypes are ignored - checkpoints legitimately ship
    bf16/fp16 - but key sets and shapes must match exactly."""
    if not geometries:
        raise Flux2DetectError("empty state dict header")
    try:
        geometries = normalize_flux_keys(geometries)
    except ValueError as error:
        raise Flux2DetectError(str(error)) from error
    keys = frozenset(geometries)

    if "double_stream_modulation_img.lin.weight" not in keys:
        raise Flux2DetectError(
            "not a Flux2 DiT (no double_stream_modulation_img.lin.weight);"
            " classic Flux detection lives in dinkster_inference.flux"
        )

    img_in = geometries.get("img_in.weight")
    for name, config in KNOWN_FLUX2_CONFIGS.items():
        if img_in is None or img_in.shape != (config.hidden_size, FLUX2_LATENT_CHANNELS):
            continue
        layout = flux2_layout(config)
        problems: list[str] = []
        for key, shape in layout.items():
            found = geometries.get(key)
            if found is None:
                problems.append(f"missing {key}")
            elif found.shape != shape:
                problems.append(f"{key}: expected shape {shape}, found {found.shape}")
        problems.extend(f"unexpected key {key}" for key in sorted(set(geometries) - set(layout)))
        if not problems:
            return config
        shown = "; ".join(problems[:6])
        more = len(problems) - 6
        if more > 0:
            shown += f"; and {more} more"
        raise Flux2DetectError(f"geometry does not match the {name} layout: {shown}")
    found_shape = None if img_in is None else img_in.shape
    raise Flux2DetectError(
        "global-modulation checkpoint with no published Flux2 geometry:"
        f" img_in.weight is {found_shape}, expected (hidden_size,"
        f" {FLUX2_LATENT_CHANNELS}) with hidden_size in"
        f" {sorted(config.hidden_size for config in KNOWN_FLUX2_CONFIGS.values())}"
    )


@dataclass(frozen=True)
class Flux2Evidence:
    config: FluxConfig
    key_prefix: str
    matched_keys: tuple[str, ...]
    fields: Mapping[str, EvidenceValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "matched_keys", tuple(self.matched_keys))
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))


def detect_flux2(source: WeightSource) -> Flux2Evidence | None:
    """Family-detector seam over :func:`detect_flux2_config`: scan the
    known checkpoint prefixes and return evidence for the first exact
    Flux2 match, or None (never raises for foreign checkpoints)."""
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
            config = detect_flux2_config(scoped)
        except Flux2DetectError:
            continue
        matched = tuple(sorted(prefix + key for key in scoped))
        return Flux2Evidence(
            config=config,
            key_prefix=prefix,
            matched_keys=matched,
            fields={
                "context_in_dim": config.context_in_dim,
                "depth": config.depth,
                "depth_single_blocks": config.depth_single_blocks,
                "guidance_embed": config.guidance_embed,
                "hidden_size": config.hidden_size,
                "key_prefix": prefix,
            },
        )
    return None


def flux2_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """The reference's empirically fitted Flux2 schedule exponent.

    Exact port of compute_empirical_mu (comfy_extras/nodes_flux.py
    @ b78cec87): a linear fit in the image token count above 4300
    tokens, otherwise a step-count interpolation between the 10-step
    and 200-step fit lines. ``image_seq_len`` counts packed latent
    cells (latent height times width at 16x spatial downscale). The
    result is the exponent :func:`flux_time_shift` consumes, the same
    convention as ``FluxFlowSigmas.shift``, so it plugs directly into
    a ``sampling_shift`` override.
    """
    if type(image_seq_len) is not int or image_seq_len < 1:
        raise ValueError("image token count must be a positive integer")
    if type(num_steps) is not int or num_steps < 1:
        raise ValueError("step count must be a positive integer")
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        return a2 * image_seq_len + b2

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return a * num_steps + b


__all__ = [
    "FLUX2_AXES_DIM",
    "FLUX2_DEV_CONFIG",
    "FLUX2_KLEIN_4B_CONFIG",
    "FLUX2_KLEIN_9B_CONFIG",
    "FLUX2_LATENT_CHANNELS",
    "FLUX2_MLP_RATIO",
    "FLUX2_PATCH_SIZE",
    "FLUX2_THETA",
    "FLUX2_TXT_IDS_DIMS",
    "Flux2DetectError",
    "Flux2Evidence",
    "KNOWN_FLUX2_CONFIGS",
    "detect_flux2",
    "detect_flux2_config",
    "flux2_empirical_mu",
    "flux2_layout",
]
