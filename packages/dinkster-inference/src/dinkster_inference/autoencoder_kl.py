"""SD/SDXL AutoencoderKL: torch-free detection, config, and descriptor.

The reference detects this family inside comfy/sd.py VAE.__init__
(@ b78cec87, the "default SD1.x/SD2.x VAE parameters" branch): a
hardcoded ddconfig, two key-presence checks for the x4-upscaler
variant, and shape reads off a fully loaded state dict. Here the same
decisions run over ``TensorGeometry`` mappings (headers, not
payloads) and produce a frozen :class:`KLConfig`; the torch modules
(dinkster_inference_torch.autoencoder_kl) construct from that config.

Detection REJECTS rather than guesses: Diffusers AutoencoderKL keys
are canonicalized through the reference's finite map, while every
non-KL family the reference's elif chain would route elsewhere
(video/temporal decoders, TAESD, conv3d), plus variants Dinkster has not
ported yet (double_z=False, decoder-only checkpoints), raises
:class:`KLDetectError` with the reason. Deferred variants are
ledgered in ROADMAP.md, never silently dropped.

Two supported quant-conv forms: the classic SD/SDXL layout with
quant_conv/post_quant_conv 1x1 convs, and the regularizer-only
AutoencodingEngine layout (comfy/sd.py @ b78cec87 selects it when
post_quant_conv is absent - the classic Flux ae.safetensors), where
the diagonal Gaussian regularizer consumes encoder.conv_out's moments
directly and the decoder consumes the latent directly
(``KLConfig.quant_convs`` False, embed_dim == z_channels).

The batch-norm-latent variant (the Flux2 VAE; comfy/sd.py @ b78cec87
sets ddconfig["batch_norm_latent"] when ``bn.running_mean`` is
present) packs each 2x2 latent patch into channels after encode and
normalizes with frozen BatchNorm running statistics (eps 1e-4, no
affine), so the external latent is 4*embed_dim channels at twice the
convolutional downscale (128ch/16x for Flux2's 32-channel 8x core).
Both real layouts carry quant convs (the BFL ae nests them under the
coder prefixes; the repackaged Diffusers layout keeps them top
level); the reference silently drops bn for the regularizer-only
construction, so that pairing refuses here instead.

Divergences from the reference, deliberate:

- ``ch_mult`` is INFERRED from block geometry and then required to be
  one of the two known variants, instead of assumed from two key
  presence checks; inconsistent topologies fail loudly instead of
  misbuilding.
- The external latent channel count is ``embed_dim``
  (post_quant_conv input), not ``z_channels``; the reference
  bookkeeps z_channels, which only coincides because SD uses
  embed_dim == z_channels. Behavior is identical for every supported
  checkpoint.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeVar, cast

from .codecs import CodecDescriptor, CodecTiling
from .devices import BFLOAT16, FLOAT32
from .latents import LatentDescriptor
from .weights import TensorGeometry

V = TypeVar("V")

#: comfy/sd.py @ b78cec87: some KL checkpoints nest the quant convs
#: under the coder prefixes; the reference renames before loading.
KL_PREFIX_RENAMES: Mapping[str, str] = MappingProxyType(
    {
        "decoder.post_quant_conv.": "post_quant_conv.",
        "encoder.quant_conv.": "quant_conv.",
    }
)

#: Finite Diffusers AutoencoderKL key conversion, transcribed from
#: comfy/diffusers_convert.py @ f4b99bc6. Tuples are canonical SD
#: spelling first and Diffusers spelling second.
_DIFFUSERS_KL_RENAMES: list[tuple[str, str]] = [
    ("nin_shortcut", "conv_shortcut"),
    ("norm_out", "conv_norm_out"),
    ("mid.attn_1.", "mid_block.attentions.0."),
]
for _level in range(4):
    for _block in range(2):
        _DIFFUSERS_KL_RENAMES.append(
            (
                f"encoder.down.{_level}.block.{_block}.",
                f"encoder.down_blocks.{_level}.resnets.{_block}.",
            )
        )
    if _level < 3:
        _DIFFUSERS_KL_RENAMES.extend(
            (
                (
                    f"down.{_level}.downsample.",
                    f"down_blocks.{_level}.downsamplers.0.",
                ),
                (
                    f"up.{3 - _level}.upsample.",
                    f"up_blocks.{_level}.upsamplers.0.",
                ),
            )
        )
    for _block in range(3):
        _DIFFUSERS_KL_RENAMES.append(
            (
                f"decoder.up.{3 - _level}.block.{_block}.",
                f"decoder.up_blocks.{_level}.resnets.{_block}.",
            )
        )
for _block in range(2):
    _DIFFUSERS_KL_RENAMES.append((f"mid.block_{_block + 1}.", f"mid_block.resnets.{_block}."))

_DIFFUSERS_KL_ATTN_RENAMES = (
    ("norm.", "group_norm."),
    ("q.", "query."),
    ("k.", "key."),
    ("v.", "value."),
    ("q.", "to_q."),
    ("k.", "to_k."),
    ("v.", "to_v."),
    ("proj_out.", "to_out.0."),
    ("proj_out.", "proj_attn."),
)
DIFFUSERS_KL_MARKER = "decoder.up_blocks.0.resnets.0.norm1.weight"
_DIFFUSERS_NAMESPACE_PARTS = (
    "down_blocks.",
    "up_blocks.",
    "mid_block.",
    "conv_norm_out.",
)
_CANONICAL_KL_NAMESPACE_PARTS = (
    "encoder.down.",
    "decoder.up.",
    ".mid.block_",
    ".mid.attn_1.",
    ".nin_shortcut.",
    ".norm_out.",
)
_DIFFUSERS_ATTN_PREFIX = ".mid_block.attentions.0."
_DIFFUSERS_ATTN_PARAMETERS = frozenset(
    f"{layer}.{parameter}"
    for layer in (
        "group_norm",
        "query",
        "key",
        "value",
        "to_q",
        "to_k",
        "to_v",
        "proj_attn",
        "to_out.0",
    )
    for parameter in ("weight", "bias")
)

#: The two ch_mult layouts the reference builds (comfy/sd.py
#: @ b78cec87): standard SD/SDXL x8, and the Stable Diffusion
#: x4-upscaler VAE.
KL_STANDARD_CH_MULT = (1, 2, 4, 4)
KL_X4_CH_MULT = (1, 2, 4)

#: The batch-norm-latent stage (comfy/ldm/models/autoencoder.py
#: AutoencodingEngineLegacy @ b78cec87): 2x2 latent patch packed into
#: channels, frozen BatchNorm running stats with this eps, no affine.
KL_BATCH_NORM_EPS = 1e-4
KL_LATENT_PATCH = 2


class KLDetectError(ValueError):
    """The geometry mapping is not a supported SD/SDXL AutoencoderKL
    checkpoint; the message names what was found instead."""


@dataclass(frozen=True)
class KLConfig:
    """Everything needed to construct an SD/SDXL AutoencoderKL.

    ``ch``/``decoder_ch`` are the encoder/decoder base widths (the
    reference's ddconfig ``ch`` and the decoder_ddconfig override for
    checkpoints whose decoder was retrained wider or narrower);
    ``ch_mult`` is the per-level multiplier ladder; ``z_channels`` is
    the pre-quant latent width and ``embed_dim`` the external latent
    width (quant_conv maps 2*z -> 2*embed, post_quant_conv maps
    embed -> z). ``double_z`` is always true for this family - the
    encoder emits mean and logvar.

    ``quant_convs`` is False for the regularizer-only
    AutoencodingEngine variant (the classic Flux ae.safetensors;
    comfy/sd.py @ b78cec87 builds AutoencodingEngine +
    DiagonalGaussianRegularizer when post_quant_conv is absent): no
    quant_conv/post_quant_conv modules exist, the regularizer consumes
    encoder.conv_out's moments directly, and the decoder consumes the
    latent directly - so embed_dim must equal z_channels.

    ``batch_norm_latent`` is True for the Flux2 VAE
    (AutoencodingEngineLegacy @ b78cec87 with
    ddconfig["batch_norm_latent"]): after encode, each 2x2 latent
    patch is packed into channels and normalized with the frozen
    ``bn`` BatchNorm running statistics; decode inverts before
    post_quant_conv. The reference builds the norm over
    4*z_channels and applies it to the 4*embed_dim packed latent, so
    only embed_dim == z_channels is coherent; and since its
    regularizer-only construction silently drops bn, quant_convs is
    required."""

    in_channels: int
    out_channels: int
    ch: int
    decoder_ch: int
    ch_mult: tuple[int, ...]
    num_res_blocks: int
    z_channels: int
    embed_dim: int
    dropout: float = 0.0
    quant_convs: bool = True
    batch_norm_latent: bool = False

    def __post_init__(self) -> None:
        for name in (
            "in_channels",
            "out_channels",
            "ch",
            "decoder_ch",
            "num_res_blocks",
            "z_channels",
            "embed_dim",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if not self.ch_mult or any(m < 1 for m in self.ch_mult):
            raise ValueError(f"ch_mult entries must be >= 1, got {self.ch_mult}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not self.quant_convs and self.embed_dim != self.z_channels:
            raise ValueError(
                "quant_convs=False requires embed_dim == z_channels,"
                f" got embed_dim={self.embed_dim}"
                f" z_channels={self.z_channels}"
            )
        if self.batch_norm_latent:
            if not self.quant_convs:
                raise ValueError("batch_norm_latent requires quant_convs")
            if self.embed_dim != self.z_channels:
                raise ValueError(
                    "batch_norm_latent requires embed_dim == z_channels,"
                    f" got embed_dim={self.embed_dim}"
                    f" z_channels={self.z_channels}"
                )

    @property
    def spatial_downscale(self) -> int:
        """Content-to-latent spatial ratio: one 2x downsample per
        level transition (8 for standard SD/SDXL, 4 for x4), doubled
        by the 2x2 latent patch packing when batch_norm_latent."""
        downscale = 2 ** (len(self.ch_mult) - 1)
        if self.batch_norm_latent:
            downscale *= KL_LATENT_PATCH
        return downscale

    @property
    def latent_channels(self) -> int:
        """External latent width: embed_dim, times the packed 2x2
        patch when batch_norm_latent (128 for the Flux2 VAE)."""
        if self.batch_norm_latent:
            return self.embed_dim * KL_LATENT_PATCH * KL_LATENT_PATCH
        return self.embed_dim


def diffusers_kl_key(key: str) -> str:
    """Return the canonical KL model key for one Diffusers source key."""
    converted = key
    for canonical, diffusers in _DIFFUSERS_KL_RENAMES:
        converted = converted.replace(diffusers, canonical)
    if "attentions" in key:
        for canonical, diffusers in _DIFFUSERS_KL_ATTN_RENAMES:
            converted = converted.replace(diffusers, canonical)
    return converted


def is_diffusers_kl(geometries: Mapping[str, object]) -> bool:
    """The exact VAE layout signal used by comfy/sd.py @ f4b99bc6."""
    return DIFFUSERS_KL_MARKER in geometries


def normalize_kl_keys(sd: Mapping[str, V]) -> dict[str, V]:
    """Canonicalize supported KL source spellings.

    Generic values are preserved. At header time, Diffusers rank-2
    attention projection geometries become canonical 1x1 Conv2D
    geometries; payload reshaping is recorded by assembly planning.
    Mixed, incomplete, colliding, or unknown Diffusers namespaces
    refuse instead of being guessed.
    """
    diffusers = is_diffusers_kl(sd)
    has_diffusers_namespace = any(
        any(part in key for part in _DIFFUSERS_NAMESPACE_PARTS) for key in sd
    )
    if not diffusers:
        if has_diffusers_namespace:
            raise KLDetectError(
                "incomplete or unknown diffusers-format VAE state dict;"
                f" missing {DIFFUSERS_KL_MARKER}"
            )
    elif any(any(part in key for part in _CANONICAL_KL_NAMESPACE_PARTS) for key in sd):
        raise KLDetectError(
            "mixed canonical and diffusers-format VAE state dict; refusing ambiguous layout"
        )
    converted: dict[str, V] = {}
    for key, value in sd.items():
        normalized_key = key
        for old, new in KL_PREFIX_RENAMES.items():
            if normalized_key.startswith(old):
                normalized_key = new + normalized_key[len(old) :]
                break
        if diffusers and _DIFFUSERS_ATTN_PREFIX in normalized_key:
            parameter = normalized_key.split(_DIFFUSERS_ATTN_PREFIX, 1)[1]
            if parameter not in _DIFFUSERS_ATTN_PARAMETERS:
                raise KLDetectError(
                    f"unknown diffusers-format VAE attention key {key!r};"
                    " finite key map does not cover it"
                )
        model_key = diffusers_kl_key(normalized_key) if diffusers else normalized_key
        if diffusers and any(part in model_key for part in _DIFFUSERS_NAMESPACE_PARTS):
            raise KLDetectError(
                f"unknown diffusers-format VAE key {key!r}; finite key map does not cover it"
            )
        if model_key in converted:
            raise KLDetectError(f"ambiguous diffusers-format VAE keys collide at {model_key!r}")
        if (
            isinstance(value, TensorGeometry)
            and model_key.endswith((".q.weight", ".k.weight", ".v.weight", ".proj_out.weight"))
            and len(value.shape) == 2
        ):
            value = cast("V", TensorGeometry((*value.shape, 1, 1), value.dtype))
        converted[model_key] = value
    return converted


def _count_keys(sd: Mapping[str, TensorGeometry], template: str) -> int:
    n = 0
    while template.format(n) in sd:
        n += 1
    return n


def _conv(
    sd: Mapping[str, TensorGeometry],
    key: str,
    *,
    kernel: int,
    out_channels: int | None = None,
    in_channels: int | None = None,
) -> TensorGeometry:
    """Require ``key`` to be a rank-4 conv weight with the given
    kernel and (optionally) channel counts, or refuse."""
    geo = sd.get(key)
    if geo is None:
        raise KLDetectError(f"missing {key}; not a supported KL layout")
    shape = geo.shape
    if len(shape) != 4:
        raise KLDetectError(f"{key} has rank {len(shape)}, expected a rank-4 conv weight")
    if any(extent < 1 for extent in shape):
        raise KLDetectError(f"{key} has a zero-sized dimension: {shape}")
    if shape[2] != kernel or shape[3] != kernel:
        raise KLDetectError(f"{key} kernel is {shape[2]}x{shape[3]}, expected {kernel}x{kernel}")
    if out_channels is not None and shape[0] != out_channels:
        raise KLDetectError(f"{key} emits {shape[0]} channels, expected {out_channels}")
    if in_channels is not None and shape[1] != in_channels:
        raise KLDetectError(f"{key} takes {shape[1]} channels, expected {in_channels}")
    return geo


def _check_shortcut(
    sd: Mapping[str, TensorGeometry],
    prefix: str,
    in_channels: int,
    out_channels: int,
) -> None:
    """A ResnetBlock carries nin_shortcut exactly when its channel
    count changes (the reference's in != out condition)."""
    key = f"{prefix}.nin_shortcut.weight"
    if (key in sd) != (in_channels != out_channels):
        raise KLDetectError(
            f"{prefix} nin_shortcut presence contradicts its"
            f" {in_channels} -> {out_channels} channels"
        )
    if key in sd:
        _conv(sd, key, kernel=1, out_channels=out_channels, in_channels=in_channels)


def _check_block(
    sd: Mapping[str, TensorGeometry],
    prefix: str,
    in_channels: int,
    out_channels: int,
) -> None:
    _conv(
        sd,
        f"{prefix}.conv1.weight",
        kernel=3,
        out_channels=out_channels,
        in_channels=in_channels,
    )
    _conv(
        sd,
        f"{prefix}.conv2.weight",
        kernel=3,
        out_channels=out_channels,
        in_channels=out_channels,
    )
    _check_shortcut(sd, prefix, in_channels, out_channels)


def _check_mid(sd: Mapping[str, TensorGeometry], prefix: str, channels: int) -> None:
    _check_block(sd, f"{prefix}.block_1", channels, channels)
    _check_block(sd, f"{prefix}.block_2", channels, channels)
    for name in ("q", "k", "v", "proj_out"):
        _conv(
            sd,
            f"{prefix}.attn_1.{name}.weight",
            kernel=1,
            out_channels=channels,
            in_channels=channels,
        )


def _index(key: str, text: str, limit: int, what: str) -> int:
    """Parse a level/block index segment strictly (the constructed
    model's ModuleList keys are exactly str(i)) and require it inside
    the detected layout."""
    try:
        value = int(text)
    except ValueError:
        # int() refusals (unicode superscript digits, segments over
        # the integer conversion limit) must surface as KL errors,
        # never incidental ValueError
        raise KLDetectError(f"{key}: {what} index {text!r} is not an integer") from None
    if str(value) != text:
        # rejects zero-padded ("01"), signed ("+1"), and unicode
        # digit ("\u0663") spellings the model never emits
        raise KLDetectError(f"{key}: {what} index {text!r} is not an integer")
    if not 0 <= value < limit:
        raise KLDetectError(
            f"{key}: {what} {value} is outside the detected layout (expected < {limit})"
        )
    return value


def _check_indices(sd: Mapping[str, TensorGeometry], levels: int, num_res_blocks: int) -> None:
    """Every key under the coder topology namespaces must name a
    member the detected layout actually constructs. _count_keys stops
    at the first gap, so without this walk a sparse extra block or
    level (encoder.down.0.block.3.* after blocks 0..1) would survive
    detection and only surface after a wrong model was built."""
    members = {
        ("encoder", "down"): (num_res_blocks, "downsample"),
        ("decoder", "up"): (num_res_blocks + 1, "upsample"),
    }
    for key in sd:
        parts = key.split(".")
        if len(parts) < 4:
            continue
        spec = members.get((parts[0], parts[1]))
        if spec is None:
            continue
        blocks, resample = spec
        _index(key, parts[2], levels, f"{parts[0]}.{parts[1]} level")
        if parts[3] == "block":
            if len(parts) < 5:
                raise KLDetectError(f"{key}: truncated block key")
            _index(key, parts[4], blocks, f"{parts[0]}.{parts[1]} block")
        elif parts[3] != resample:
            raise KLDetectError(
                f"{key}: unsupported member {parts[3]!r}; expected block or {resample}"
            )


def detect_kl_config(geometries: Mapping[str, TensorGeometry]) -> KLConfig:
    """Classify a checkpoint header as SD/SDXL AutoencoderKL and read
    its :class:`KLConfig` from the geometry, or refuse loudly."""
    sd = normalize_kl_keys(geometries)

    if "decoder.mid.block_1.mix_factor" in sd:
        raise KLDetectError("temporal/video KL decoder (mix_factor keys); not this family")
    if "taesd_decoder.1.weight" in sd:
        raise KLDetectError("TAESD checkpoint; not this family")
    for key in sd:
        if ".conv_shortcut." in key:
            raise KLDetectError(
                f"conv_shortcut ResnetBlock variant ({key}); not a supported KL layout"
            )
        if ".temb_proj." in key:
            raise KLDetectError(f"time-conditioned ResnetBlock ({key}); not a supported KL layout")
        if key.startswith(("encoder.down.", "decoder.up.")) and ".attn." in key:
            raise KLDetectError(f"per-resolution attention ({key}); not a supported KL layout")

    dec_in = sd.get("decoder.conv_in.weight")
    enc_in = sd.get("encoder.conv_in.weight")
    if dec_in is None:
        raise KLDetectError("no decoder.conv_in.weight; not a KL autoencoder")
    if enc_in is None:
        raise KLDetectError(
            "no encoder.conv_in.weight; decoder-only KL checkpoints are"
            " not ported yet (ROADMAP: Native inference)"
        )
    if len(dec_in.shape) == 5 or len(enc_in.shape) == 5:
        raise KLDetectError("conv3d (video) KL autoencoder; not this family")
    dec_in = _conv(sd, "decoder.conv_in.weight", kernel=3)
    enc_in = _conv(sd, "encoder.conv_in.weight", kernel=3)

    quant = sd.get("quant_conv.weight")
    post_quant = sd.get("post_quant_conv.weight")
    if (quant is None) != (post_quant is None):
        raise KLDetectError(
            "quant_conv and post_quant_conv must be present together; found only one"
        )
    z_channels = dec_in.shape[1]
    quant_convs = quant is not None
    if quant_convs:
        quant = _conv(sd, "quant_conv.weight", kernel=1)
        post_quant = _conv(sd, "post_quant_conv.weight", kernel=1)
        embed_dim = post_quant.shape[1]
        if post_quant.shape[0] != z_channels:
            raise KLDetectError(
                f"post_quant_conv maps {post_quant.shape[1]} ->"
                f" {post_quant.shape[0]} channels but decoder.conv_in expects"
                f" {z_channels}"
            )
        if quant.shape[0] == embed_dim and quant.shape[1] == z_channels:
            raise KLDetectError(
                "double_z=False KL variant (quant_conv carries no logvar"
                " half); not ported yet (ROADMAP: Native inference)"
            )
        if quant.shape[0] != 2 * embed_dim or quant.shape[1] != 2 * z_channels:
            raise KLDetectError(
                f"quant_conv geometry {quant.shape} does not map"
                f" 2*z ({2 * z_channels}) -> 2*embed ({2 * embed_dim})"
            )
    else:
        # Regularizer-only AutoencodingEngine (comfy/sd.py @ b78cec87
        # builds AutoencodingEngine + DiagonalGaussianRegularizer when
        # post_quant_conv is absent - the classic Flux ae.safetensors):
        # the moments come straight off encoder.conv_out and the
        # decoder consumes the latent directly, so embed_dim IS
        # z_channels.
        embed_dim = z_channels

    bn_mean = sd.get("bn.running_mean")
    bn_var = sd.get("bn.running_var")
    if (bn_mean is None) != (bn_var is None):
        raise KLDetectError(
            "bn.running_mean and bn.running_var must be present together; found only one"
        )
    batch_norm_latent = bn_mean is not None
    if batch_norm_latent:
        assert bn_mean is not None and bn_var is not None
        if not quant_convs:
            raise KLDetectError(
                "batch-norm-latent KL without quant convs; the"
                " reference's regularizer-only construction silently"
                " drops bn, refusing instead"
            )
        if embed_dim != z_channels:
            raise KLDetectError(
                "batch-norm-latent KL requires embed_dim =="
                f" z_channels, got embed_dim={embed_dim}"
                f" z_channels={z_channels}"
            )
        packed = z_channels * KL_LATENT_PATCH * KL_LATENT_PATCH
        for name, geo in (("bn.running_mean", bn_mean), ("bn.running_var", bn_var)):
            if tuple(geo.shape) != (packed,):
                raise KLDetectError(
                    f"{name} geometry {geo.shape} does not cover the"
                    f" packed 2x2 latent ({packed} channels)"
                )
    enc_out = _conv(sd, "encoder.conv_out.weight", kernel=3)
    if enc_out.shape[0] != 2 * z_channels:
        raise KLDetectError(
            f"encoder.conv_out does not emit 2*z_channels ({2 * z_channels}) moment channels"
        )

    num_res_blocks = _count_keys(sd, "encoder.down.0.block.{}.conv1.weight")
    if num_res_blocks == 0:
        raise KLDetectError("no encoder.down.0 residual blocks found")
    decoder_blocks = _count_keys(sd, "decoder.up.0.block.{}.conv1.weight")
    if decoder_blocks != num_res_blocks + 1:
        raise KLDetectError(
            f"decoder has {decoder_blocks} blocks per level, expected"
            f" num_res_blocks + 1 = {num_res_blocks + 1}"
        )

    levels = _count_keys(sd, "encoder.down.{}.block.0.conv1.weight")
    if _count_keys(sd, "decoder.up.{}.block.0.conv1.weight") != levels:
        raise KLDetectError("encoder and decoder disagree on resolution level count")
    _check_indices(sd, levels, num_res_blocks)

    ch = enc_in.shape[0]
    ch_mult: list[int] = []
    for level in range(levels):
        width = _conv(sd, f"encoder.down.{level}.block.0.conv1.weight", kernel=3).shape[0]
        if width % ch != 0:
            raise KLDetectError(
                f"encoder level {level} width {width} is not a multiple of base ch {ch}"
            )
        ch_mult.append(width // ch)
    inferred = tuple(ch_mult)
    if inferred not in (KL_STANDARD_CH_MULT, KL_X4_CH_MULT):
        raise KLDetectError(
            f"ch_mult {inferred} is not a known SD/SDXL KL layout"
            f" ({KL_STANDARD_CH_MULT} or x4 {KL_X4_CH_MULT})"
        )
    if dec_in.shape[0] % inferred[-1] != 0:
        raise KLDetectError(
            f"decoder.conv_in width {dec_in.shape[0]} is not a multiple"
            f" of ch_mult[-1] ({inferred[-1]})"
        )
    decoder_ch = dec_in.shape[0] // inferred[-1]
    dec_out = sd.get("decoder.conv_out.weight")
    if dec_out is None:
        raise KLDetectError("no decoder.conv_out.weight")

    # Structural walk: every residual block, shortcut, resample conv,
    # and mid stack must match the layout the config would construct -
    # a topology detection cannot rebuild refuses here instead of
    # misbuilding and failing (or worse, not failing) at load time.
    in_mults = (1, *inferred)
    for level in range(levels):
        out_w = ch * inferred[level]
        blocks = _count_keys(sd, f"encoder.down.{level}.block.{{}}.conv1.weight")
        if blocks != num_res_blocks:
            raise KLDetectError(
                f"encoder level {level} has {blocks} residual blocks,"
                f" expected {num_res_blocks} like level 0"
            )
        for block in range(num_res_blocks):
            block_in = ch * in_mults[level] if block == 0 else out_w
            _check_block(sd, f"encoder.down.{level}.block.{block}", block_in, out_w)
        has_down = f"encoder.down.{level}.downsample.conv.weight" in sd
        if has_down != (level < levels - 1):
            raise KLDetectError(
                f"encoder level {level} downsample presence contradicts a {levels}-level layout"
            )
        if has_down:
            _conv(
                sd,
                f"encoder.down.{level}.downsample.conv.weight",
                kernel=3,
                out_channels=out_w,
                in_channels=out_w,
            )
    _check_mid(sd, "encoder.mid", ch * inferred[-1])
    _conv(
        sd,
        "encoder.conv_out.weight",
        kernel=3,
        out_channels=2 * z_channels,
        in_channels=ch * inferred[-1],
    )

    _check_mid(sd, "decoder.mid", decoder_ch * inferred[-1])
    running_in = decoder_ch * inferred[-1]
    for level in reversed(range(levels)):
        out_w = decoder_ch * inferred[level]
        blocks = _count_keys(sd, f"decoder.up.{level}.block.{{}}.conv1.weight")
        if blocks != num_res_blocks + 1:
            raise KLDetectError(
                f"decoder level {level} has {blocks} blocks, expected"
                f" num_res_blocks + 1 = {num_res_blocks + 1}"
            )
        for block in range(num_res_blocks + 1):
            block_in = running_in if block == 0 else out_w
            _check_block(sd, f"decoder.up.{level}.block.{block}", block_in, out_w)
        has_up = f"decoder.up.{level}.upsample.conv.weight" in sd
        if has_up != (level > 0):
            raise KLDetectError(
                f"decoder level {level} upsample presence contradicts a {levels}-level layout"
            )
        if has_up:
            _conv(
                sd,
                f"decoder.up.{level}.upsample.conv.weight",
                kernel=3,
                out_channels=out_w,
                in_channels=out_w,
            )
        running_in = out_w
    _conv(
        sd,
        "decoder.conv_out.weight",
        kernel=3,
        in_channels=decoder_ch * inferred[0],
    )

    return KLConfig(
        in_channels=enc_in.shape[1],
        out_channels=dec_out.shape[0],
        ch=ch,
        decoder_ch=decoder_ch,
        ch_mult=inferred,
        num_res_blocks=num_res_blocks,
        z_channels=z_channels,
        embed_dim=embed_dim,
        quant_convs=quant_convs,
        batch_norm_latent=batch_norm_latent,
    )


def kl_descriptor(config: KLConfig) -> CodecDescriptor:
    """The codec descriptor for a detected KL autoencoder. Tiling
    defaults are the reference's 2D geometry (comfy/sd.py @ b78cec87:
    decode tile 64 latents / overlap 16, encode tile 512 pixels /
    overlap 64); dtypes are the reference's SD KL working_dtypes."""
    return CodecDescriptor(
        id="dinkster.autoencoder_kl",
        display_name="AutoencoderKL (SD/SDXL)",
        kind="image",
        latent=LatentDescriptor(
            channels=config.latent_channels,
            dimensions=2,
            spatial_downscale=config.spatial_downscale,
        ),
        supported_dtypes=frozenset({BFLOAT16, FLOAT32}),
        content_channels=config.out_channels,
        supports_tiling=True,
        tiling=CodecTiling(
            decode_tile=(64, 64),
            decode_overlap=(16, 16),
            encode_tile=(512, 512),
            encode_overlap=(64, 64),
        ),
    )


@dataclass(frozen=True)
class KLMemoryEstimator:
    """The reference's AutoencoderKL memory formulas (comfy/sd.py
    @ b78cec87): bytes ~ constant * content-plane elements * dtype
    size. The decode formula bakes in the x8 upscale (the reference
    never adjusts it for the x4 variant; matched for fidelity).
    ``ratio`` is the reference's VAE_KL_MEM_RATIO device knob (2.73 on
    AMD); the device-policy layer constructs with it.
    ``decode_multiplier`` is the reference's 4.0 batch-norm-latent
    adjustment: the packed latent plane is 4x smaller, so the decode
    estimate is restored by the same factor."""

    ratio: float = 1.0
    decode_multiplier: float = 1.0

    def encode_bytes(self, content: TensorGeometry) -> int:
        plane = content.shape[-2] * content.shape[-1]
        return int(1767 * plane * _dtype_bytes(content) * self.ratio)

    def decode_bytes(self, latent: TensorGeometry) -> int:
        plane = latent.shape[-2] * latent.shape[-1]
        return int(2178 * plane * 64 * _dtype_bytes(latent) * self.ratio * self.decode_multiplier)


def _dtype_bytes(geometry: TensorGeometry) -> int:
    return (geometry.dtype.bits + 7) // 8


__all__ = [
    "KL_BATCH_NORM_EPS",
    "KL_LATENT_PATCH",
    "KL_PREFIX_RENAMES",
    "KL_STANDARD_CH_MULT",
    "KL_X4_CH_MULT",
    "KLConfig",
    "KLDetectError",
    "KLMemoryEstimator",
    "detect_kl_config",
    "kl_descriptor",
    "normalize_kl_keys",
]
