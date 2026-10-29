"""Flux diffusion transformer: torch-free config, layout, detection.

The reference infers this architecture from state-dict keys and
shapes (comfy/model_detection.py, the ``double_blocks.0.img_attn.
norm.key_norm`` branch @ b78cec87) and constructs it in
comfy/ldm/flux/model.py Flux. :class:`FluxConfig` carries exactly
the FluxParams fields that Flux construction consumes; the torch modules
(dinkster_inference_torch.flux) construct
from that config through the typed Operations seam with state-dict
keys IDENTICAL to the reference's bare BFL layout.

Unlike the SD UNet, most classic-Flux facts are pinned constants in
the reference's detection rather than derived from shapes: axes_dim
(16, 56, 56), theta 10000, patch_size 2, mlp_ratio 4.0, qkv_bias
True, and in_channels 16. Detection derives hidden_size /
context_in_dim / optional vec_in_dim / depth / depth_single_blocks /
guidance_embed from the header and pins the rest, exactly like the
reference; num_heads is hidden_size // sum(axes_dim).

Detection REJECTS rather than guesses: every family the reference's
flux branch routes to a variant config is refused with a
:class:`FluxDetectError` naming the marker it found - Flux2
(``double_stream_modulation_img``, detected separately in
:mod:`dinkster_inference.flux2`), Chroma / Chroma Radiance
(``distilled_guidance_layer``), LongCat-Image (no ``vector_in`` at the 3584
Qwen2.5-VL context width), and FluxInpaint (img_in widened to 384 columns).
The one admitted vector-free architecture is the exact Ovis row-28 geometry:
2048-wide normalized context, gated MLPs, 3072 hidden width, 6 double blocks,
27 single blocks, and neither vector nor guidance embedder. A classic-width
header missing vector_in is still reported as truncated, not as a lineage.
Deferred variants are ledgered in ROADMAP.md ("Native inference"), never
silently dropped. A config candidate that survives the scan is then required
to reproduce the ENTIRE key/shape listing (:func:`flux_layout`); any drift
refuses with the differences.

RMSNorm parameters ship as ``*_norm.scale`` in bare BFL exports and
``*_norm.weight`` in others; the reference renames scale to weight
before loading (comfy/supported_models.py Flux.process_unet_state_dict
@ b78cec87). :func:`normalize_flux_keys` is that rename as a pure
mapping transform; detection applies it internally, loaders apply it
to tensor mappings before ``load_state_dict``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TypeVar

from .weights import TensorGeometry

_V = TypeVar("_V")


class FluxDetectError(ValueError):
    """The geometry mapping is not a supported classic Flux
    (dev / schnell) checkpoint; the message names what was found
    instead."""


#: Classic-Flux constants the reference detection pins rather than
#: derives (comfy/model_detection.py flux branch @ b78cec87).
FLUX_AXES_DIM = (16, 56, 56)
FLUX_THETA = 10000
FLUX_PATCH_SIZE = 2
FLUX_MLP_RATIO = 4.0
FLUX_LATENT_CHANNELS = 16
#: The reference's LongCat-Image discriminator: context width 3584
#: (Qwen2.5-VL hidden size) with no vector_in
#: (comfy/model_detection.py flux branch @ b78cec87).
FLUX_LONGCAT_CONTEXT_DIM = 3584

#: Exact discriminator facts of the accepted Ovis row-28 artifact. These are
#: deliberately private: Ovis remains a structural variant of the existing
#: Flux matcher/family, not a new public family configuration.
_FLUX_OVIS_CONTEXT_DIM = 2048
_FLUX_OVIS_HIDDEN_SIZE = 3072
_FLUX_OVIS_DEPTH = 6
_FLUX_OVIS_SINGLE_DEPTH = 27
_FLUX_OVIS_TXT_IDS_DIMS = (1, 2)


@dataclass(frozen=True)
class FluxConfig:
    """The construction-relevant subset of the reference FluxParams
    for classic Flux (dev / schnell).

    ``txt_norm`` and ``yak_mlp`` describe the context-normalized and gated
    row-28 Flux variants. Header detection assigns ``vec_in_dim=None`` and
    ``txt_ids_dims=(1, 2)`` only to the exact accepted vector-free Ovis
    combination. ``global_modulation``, ``mlp_silu_act``, and ``ops_bias``
    describe Flux2 (comfy/ldm/flux/model.py FluxParams @ b78cec87): shared
    modulation modules computed once per forward instead of per block,
    SiLU-gated packed MLPs, and bias-free linears throughout. Only the exact
    published Flux2 geometries set them (:mod:`dinkster_inference.flux2`).
    Variant flags are excluded from the dataclass repr so the
    classic config identity stays byte-stable; each admitted variant's required
    model keys distinguish its structural identity.
    ``guidance_embed`` is the dev / schnell split: dev distills CFG into
    a guidance embedding, schnell has none."""

    in_channels: int
    out_channels: int
    vec_in_dim: int | None
    context_in_dim: int
    hidden_size: int
    depth: int
    depth_single_blocks: int
    num_heads: int
    axes_dim: tuple[int, ...] = FLUX_AXES_DIM
    theta: int = FLUX_THETA
    patch_size: int = FLUX_PATCH_SIZE
    mlp_ratio: float = FLUX_MLP_RATIO
    qkv_bias: bool = True
    guidance_embed: bool = True
    txt_norm: bool = field(default=False, repr=False)
    yak_mlp: bool = field(default=False, repr=False)
    txt_ids_dims: tuple[int, ...] = field(default=(), repr=False)
    global_modulation: bool = field(default=False, repr=False)
    mlp_silu_act: bool = field(default=False, repr=False)
    ops_bias: bool = field(default=True, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "txt_ids_dims", tuple(self.txt_ids_dims))
        for name in (
            "in_channels",
            "out_channels",
            "context_in_dim",
            "hidden_size",
            "depth",
            "depth_single_blocks",
            "num_heads",
            "theta",
            "patch_size",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.vec_in_dim is not None and self.vec_in_dim < 1:
            raise ValueError("vec_in_dim must be >= 1 or None")
        if self.hidden_size % self.num_heads:
            raise ValueError(
                f"hidden_size {self.hidden_size} is not divisible by num_heads {self.num_heads}"
            )
        if not self.axes_dim or any(d < 2 or d % 2 for d in self.axes_dim):
            raise ValueError(
                "axes_dim entries must be positive and even (RoPE"
                f" rotates pairs), got {self.axes_dim}"
            )
        if sum(self.axes_dim) != self.head_dim:
            raise ValueError(
                f"axes_dim {self.axes_dim} sums to {sum(self.axes_dim)},"
                f" expected the per-head dim"
                f" {self.hidden_size} // {self.num_heads}"
                f" = {self.head_dim}"
            )
        if len(set(self.txt_ids_dims)) != len(self.txt_ids_dims) or any(
            index < 0 or index >= len(self.axes_dim) for index in self.txt_ids_dims
        ):
            raise ValueError(
                "txt_ids_dims must contain unique axes_dim indices, got"
                f" {self.txt_ids_dims} for {len(self.axes_dim)} axes"
            )
        if self.mlp_hidden_dim < 1:
            raise ValueError(
                f"mlp_ratio {self.mlp_ratio} yields an empty MLP for hidden_size {self.hidden_size}"
            )
        if self.mlp_silu_act and self.yak_mlp:
            raise ValueError("mlp_silu_act and yak_mlp select contradictory MLP layouts")

    @property
    def head_dim(self) -> int:
        """Per-head width; also the positional-embedding dim."""
        return self.hidden_size // self.num_heads

    @property
    def mlp_hidden_dim(self) -> int:
        """``int(hidden_size * mlp_ratio)``, the reference's MLP
        widening."""
        return int(self.hidden_size * self.mlp_ratio)


#: Flux dev: guidance-distilled (comfy/model_detection.py derivations
#: plus comfy/supported_models.py Flux @ b78cec87).
FLUX_DEV_CONFIG = FluxConfig(
    in_channels=16,
    out_channels=16,
    vec_in_dim=768,
    context_in_dim=4096,
    hidden_size=3072,
    depth=19,
    depth_single_blocks=38,
    num_heads=24,
    guidance_embed=True,
)

#: Flux schnell: the timestep-distilled release, identical geometry
#: minus the guidance embedder (comfy/supported_models.py FluxSchnell
#: @ b78cec87).
FLUX_SCHNELL_CONFIG = FluxConfig(
    in_channels=16,
    out_channels=16,
    vec_in_dim=768,
    context_in_dim=4096,
    hidden_size=3072,
    depth=19,
    depth_single_blocks=38,
    num_heads=24,
    guidance_embed=False,
)

KNOWN_FLUX_CONFIGS: Mapping[str, FluxConfig] = {
    "flux_dev": FLUX_DEV_CONFIG,
    "flux_schnell": FLUX_SCHNELL_CONFIG,
}


def normalize_flux_keys(mapping: Mapping[str, _V]) -> dict[str, _V]:
    """Rename ``*_norm.scale`` keys to ``*_norm.weight`` - the
    reference's RMSNorm spelling normalization
    (comfy/supported_models.py Flux.process_unet_state_dict
    @ b78cec87). Pure and value-agnostic: works on tensor mappings
    and :class:`TensorGeometry` headers alike."""
    out: dict[str, _V] = {}
    sources: dict[str, str] = {}
    for source_key, value in mapping.items():
        key = source_key
        if source_key.endswith("_norm.scale"):
            key = source_key[: -len(".scale")] + ".weight"
        if key in out:
            raise ValueError(
                f"Flux keys {sources[key]!r} and {source_key!r} both normalize to {key!r}"
            )
        out[key] = value
        sources[key] = source_key
    return out


def _mlp_embedder(
    layout: dict[str, tuple[int, ...]],
    prefix: str,
    in_dim: int,
    hidden: int,
    *,
    bias: bool = True,
) -> None:
    layout[f"{prefix}.in_layer.weight"] = (hidden, in_dim)
    if bias:
        layout[f"{prefix}.in_layer.bias"] = (hidden,)
    layout[f"{prefix}.out_layer.weight"] = (hidden, hidden)
    if bias:
        layout[f"{prefix}.out_layer.bias"] = (hidden,)


def flux_layout(config: FluxConfig) -> dict[str, tuple[int, ...]]:
    """The exact key -> shape listing of the reference Flux model
    (comfy/ldm/flux/model.py @ b78cec87) in canonical ``.weight``
    RMSNorm spelling. ``load_state_dict(..., strict=True)`` on the
    native module accepts precisely these keys."""
    hidden = config.hidden_size
    head_dim = config.head_dim
    mlp = config.mlp_hidden_dim
    patch = config.patch_size * config.patch_size

    ops_bias = config.ops_bias

    layout: dict[str, tuple[int, ...]] = {}
    layout["img_in.weight"] = (hidden, config.in_channels * patch)
    if ops_bias:
        layout["img_in.bias"] = (hidden,)
    _mlp_embedder(layout, "time_in", 256, hidden, bias=ops_bias)
    if config.vec_in_dim is not None:
        # The reference constructs vector_in without the bias knob
        # (comfy/ldm/flux/model.py @ b78cec87), so it keeps biases even
        # in a bias-free config.
        _mlp_embedder(layout, "vector_in", config.vec_in_dim, hidden)
    if config.guidance_embed:
        _mlp_embedder(layout, "guidance_in", 256, hidden, bias=ops_bias)
    if config.txt_norm:
        layout["txt_norm.weight"] = (config.context_in_dim,)
    layout["txt_in.weight"] = (hidden, config.context_in_dim)
    if ops_bias:
        layout["txt_in.bias"] = (hidden,)
    if config.global_modulation:
        # The shared Modulation modules are constructed with bias=False
        # regardless of ops_bias (comfy/ldm/flux/model.py @ b78cec87).
        layout["double_stream_modulation_img.lin.weight"] = (6 * hidden, hidden)
        layout["double_stream_modulation_txt.lin.weight"] = (6 * hidden, hidden)
        layout["single_stream_modulation.lin.weight"] = (3 * hidden, hidden)

    for i in range(config.depth):
        for stream in ("img", "txt"):
            p = f"double_blocks.{i}.{stream}"
            if not config.global_modulation:
                layout[f"{p}_mod.lin.weight"] = (6 * hidden, hidden)
                layout[f"{p}_mod.lin.bias"] = (6 * hidden,)
            layout[f"{p}_attn.qkv.weight"] = (3 * hidden, hidden)
            if config.qkv_bias:
                layout[f"{p}_attn.qkv.bias"] = (3 * hidden,)
            layout[f"{p}_attn.norm.query_norm.weight"] = (head_dim,)
            layout[f"{p}_attn.norm.key_norm.weight"] = (head_dim,)
            layout[f"{p}_attn.proj.weight"] = (hidden, hidden)
            if ops_bias:
                layout[f"{p}_attn.proj.bias"] = (hidden,)
            if config.yak_mlp:
                for projection in ("gate_proj", "up_proj"):
                    layout[f"{p}_mlp.{projection}.weight"] = (mlp, hidden)
                    layout[f"{p}_mlp.{projection}.bias"] = (mlp,)
                layout[f"{p}_mlp.down_proj.weight"] = (hidden, mlp)
                layout[f"{p}_mlp.down_proj.bias"] = (hidden,)
            elif config.mlp_silu_act:
                # build_mlp's SiLU-gated branch hardcodes bias=False and
                # packs gate and value into one doubled first linear
                # (comfy/ldm/flux/layers.py @ b78cec87).
                layout[f"{p}_mlp.0.weight"] = (2 * mlp, hidden)
                layout[f"{p}_mlp.2.weight"] = (hidden, mlp)
            else:
                layout[f"{p}_mlp.0.weight"] = (mlp, hidden)
                layout[f"{p}_mlp.0.bias"] = (mlp,)
                layout[f"{p}_mlp.2.weight"] = (hidden, mlp)
                layout[f"{p}_mlp.2.bias"] = (hidden,)

    for i in range(config.depth_single_blocks):
        p = f"single_blocks.{i}"
        packed_mlp = 2 * mlp if config.yak_mlp or config.mlp_silu_act else mlp
        layout[f"{p}.linear1.weight"] = (3 * hidden + packed_mlp, hidden)
        if ops_bias:
            layout[f"{p}.linear1.bias"] = (3 * hidden + packed_mlp,)
        layout[f"{p}.linear2.weight"] = (hidden, hidden + mlp)
        if ops_bias:
            layout[f"{p}.linear2.bias"] = (hidden,)
        layout[f"{p}.norm.query_norm.weight"] = (head_dim,)
        layout[f"{p}.norm.key_norm.weight"] = (head_dim,)
        if not config.global_modulation:
            layout[f"{p}.modulation.lin.weight"] = (3 * hidden, hidden)
            layout[f"{p}.modulation.lin.bias"] = (3 * hidden,)

    layout["final_layer.linear.weight"] = (config.out_channels * patch, hidden)
    if ops_bias:
        layout["final_layer.linear.bias"] = (config.out_channels * patch,)
    layout["final_layer.adaLN_modulation.1.weight"] = (2 * hidden, hidden)
    if ops_bias:
        layout["final_layer.adaLN_modulation.1.bias"] = (2 * hidden,)
    return layout


def _count_blocks(keys: frozenset[str], template: str) -> int:
    """Contiguous block count, comfy/model_detection.py count_blocks
    @ b78cec87 over key names."""
    count = 0
    while any(key.startswith(template.format(count)) for key in keys):
        count += 1
    return count


def _linear(geometries: Mapping[str, TensorGeometry], key: str) -> TensorGeometry:
    found = geometries.get(key)
    if found is None:
        raise FluxDetectError(f"missing {key}")
    if len(found.shape) != 2:
        raise FluxDetectError(f"{key} has rank {len(found.shape)}, expected a linear (rank 2)")
    return found


def detect_flux_config(
    geometries: Mapping[str, TensorGeometry],
) -> FluxConfig:
    """Classify a diffusion-model-scoped header (bare BFL layout,
    any checkpoint prefix already stripped) as classic Flux dev or
    schnell, or refuse loudly. Dtypes are ignored - checkpoints
    legitimately ship fp16/bf16/fp8. The scan is the reference's flux
    branch (comfy/model_detection.py @ b78cec87) with every variant
    leg turned into a rejection, then the candidate must reproduce
    the full layout. RMSNorm ``.scale`` spelling is normalized
    internally (:func:`normalize_flux_keys`)."""
    if not geometries:
        raise FluxDetectError("empty state dict header")
    try:
        geometries = normalize_flux_keys(geometries)
    except ValueError as error:
        raise FluxDetectError(str(error)) from error
    keys = frozenset(geometries)

    if "double_blocks.0.img_attn.norm.key_norm.weight" not in keys:
        raise FluxDetectError(
            "not a Flux-lineage DiT (no double_blocks.0.img_attn.norm."
            "key_norm); other families route elsewhere"
        )
    if "double_stream_modulation_img.lin.weight" in keys:
        raise FluxDetectError(
            "global-modulation checkpoint (double_stream_modulation_img);"
            " Flux2 detection lives in dinkster_inference.flux2"
        )
    if any(key.startswith("distilled_guidance_layer.") for key in keys):
        # Prefix scan rather than the reference's two exact norm keys:
        # those norms are named ``norms.N`` so a bare-format ``.scale``
        # spelling would dodge normalize_flux_keys and misattribute
        # Chroma to the vector_in rejection below.
        raise FluxDetectError(
            "distilled_guidance_layer checkpoint; Chroma / Chroma"
            " Radiance are not ported (ROADMAP: Native inference)"
        )
    yak_mlp = "double_blocks.0.img_mlp.gate_proj.weight" in keys
    txt_norm = "txt_norm.weight" in keys
    img_in = _linear(geometries, "img_in.weight")
    patch = FLUX_PATCH_SIZE * FLUX_PATCH_SIZE
    in_channels = img_in.shape[1] // patch
    if in_channels != FLUX_LATENT_CHANNELS or img_in.shape[1] % patch:
        raise FluxDetectError(
            f"img_in takes {img_in.shape[1]} columns, expected"
            f" {FLUX_LATENT_CHANNELS * patch}"
            f" ({FLUX_LATENT_CHANNELS} latent channels x"
            f" {FLUX_PATCH_SIZE}x{FLUX_PATCH_SIZE} patch); widened"
            " inpainting checkpoints (FluxInpaint) are not ported"
            " (ROADMAP: Native inference)"
        )
    hidden_size = img_in.shape[0]

    txt_in = _linear(geometries, "txt_in.weight")
    if txt_in.shape[0] != hidden_size:
        raise FluxDetectError(
            f"txt_in projects to {txt_in.shape[0]} but img_in to"
            f" {hidden_size}; not a coherent Flux checkpoint"
        )
    context_in_dim = txt_in.shape[1]

    if hidden_size % sum(FLUX_AXES_DIM):
        raise FluxDetectError(
            f"hidden_size {hidden_size} is not a multiple of"
            f" sum(axes_dim) {sum(FLUX_AXES_DIM)}; attention heads are"
            " not derivable"
        )
    num_heads = hidden_size // sum(FLUX_AXES_DIM)

    depth = _count_blocks(keys, "double_blocks.{}.")
    depth_single_blocks = _count_blocks(keys, "single_blocks.{}.")
    guidance_embed = "guidance_in.in_layer.weight" in keys

    txt_ids_dims: tuple[int, ...] = ()
    if not any(key.startswith("vector_in.") for key in keys):
        # LongCat's independent 3584-wide route remains deferred. A classic
        # header that lost all four vector keys remains damage, never an Ovis
        # match. Both checks occur before the Ovis co-signal gate so mapping
        # insertion order cannot affect classification.
        if context_in_dim == FLUX_LONGCAT_CONTEXT_DIM:
            raise FluxDetectError(
                "no vector_in (CLIP-pooled embedder) and context width"
                f" {FLUX_LONGCAT_CONTEXT_DIM}; vector-free Flux lineages"
                " (LongCat-Image) are not ported (ROADMAP: Native"
                " inference)"
            )
        if context_in_dim == FLUX_DEV_CONFIG.context_in_dim:
            raise FluxDetectError(
                "missing vector_in.in_layer.weight (truncated classic Flux checkpoint)"
            )
        ovis_facts = (
            context_in_dim == _FLUX_OVIS_CONTEXT_DIM
            and hidden_size == _FLUX_OVIS_HIDDEN_SIZE
            and depth == _FLUX_OVIS_DEPTH
            and depth_single_blocks == _FLUX_OVIS_SINGLE_DEPTH
            and txt_norm
            and yak_mlp
            and not guidance_embed
        )
        if not ovis_facts:
            raise FluxDetectError(
                "unsupported vector-free Flux geometry; only the approved"
                " Ovis row-28 combination (context 2048, hidden 3072,"
                " txt_norm, gated MLP, 6 double/27 single blocks, no"
                " guidance_in) is admitted"
            )
        vec_in_dim = None
        txt_ids_dims = _FLUX_OVIS_TXT_IDS_DIMS
    else:
        if yak_mlp and txt_norm:
            raise FluxDetectError(
                "vector-present Flux combining txt_norm and gated MLP has"
                " unproven text-position semantics; only the exact"
                " vector-free Ovis composition is admitted (ROADMAP: Native"
                " inference)"
            )
        vec_in_dim = _linear(geometries, "vector_in.in_layer.weight").shape[1]

    try:
        config = FluxConfig(
            in_channels=in_channels,
            out_channels=in_channels,
            vec_in_dim=vec_in_dim,
            context_in_dim=context_in_dim,
            hidden_size=hidden_size,
            depth=depth,
            depth_single_blocks=depth_single_blocks,
            num_heads=num_heads,
            guidance_embed=guidance_embed,
            txt_norm=txt_norm,
            yak_mlp=yak_mlp,
            txt_ids_dims=txt_ids_dims,
        )
    except ValueError as error:
        raise FluxDetectError(
            f"scanned geometry does not form a valid Flux config: {error}"
        ) from error

    layout = flux_layout(config)
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
        raise FluxDetectError(f"geometry does not match the detected Flux layout: {shown}")
    return config


__all__ = [
    "FLUX_AXES_DIM",
    "FLUX_DEV_CONFIG",
    "FLUX_LATENT_CHANNELS",
    "FLUX_MLP_RATIO",
    "FLUX_PATCH_SIZE",
    "FLUX_SCHNELL_CONFIG",
    "FLUX_THETA",
    "FluxConfig",
    "FluxDetectError",
    "KNOWN_FLUX_CONFIGS",
    "detect_flux_config",
    "flux_layout",
    "normalize_flux_keys",
]
