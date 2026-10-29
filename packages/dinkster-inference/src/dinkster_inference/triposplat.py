"""TripoSplat image-to-3D flow denoiser: exact facts and fail-closed detection.

TripoSplat (comfy/ldm/triposplat @ 36408117) jointly denoises a fixed
(B, 8192, 16) latent token sequence and a (B, 1, 5) camera token with a
shared-modulation transformer (LatentSeqMMFlowModel). Conditioning is a
DINOv3 ViT-H/16+ token sequence (cross-attention context) plus an
optional Flux2 VAE reference-image latent added through a second
embedder. The "VAE" is not a KL autoencoder: an octree probability
decoder samples point coordinates, then an elastic gaussian decoder
predicts 32 gaussians per point (OctreeGaussianDecoder).

The reference detects the DiT from two marker keys
(``cam_out_layer.weight`` + ``repo_layers.0.final_map.weight``,
comfy/model_detection.py @ 36408117) and the gaussian decoder from
``gs.base_offset_scale`` + ``octree.out_proj.weight`` (comfy/sd.py
@ 36408117), then instantiates fixed architectures. Only one geometry
is published, so detection here requires the ENTIRE key/shape listing
exactly (the :mod:`dinkster_inference.flux2` discipline); the marker keys
merely select the refusal message. Dtypes are ignored - the published
artifacts ship f16 - but key sets and shapes must match exactly. The
listings are pinned against the published artifact headers
(tests/goldens/triposplat_headers.json).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from .devices import BFLOAT16, FLOAT16, FLOAT32, DType
from .families import EvidenceValue
from .spaces import FlowSigmas
from .weights import TensorGeometry, WeightSource


class TripoSplatDetectError(ValueError):
    """The geometry mapping is not a published TripoSplat artifact; the
    message names what was found instead."""


@dataclass(frozen=True)
class TripoSplatConfig:
    """The exact TripoSplat flow-model architecture requirements."""

    family_id: str = "dinkster.triposplat"
    q_token_length: int = 8192
    latent_channels: int = 16
    model_channels: int = 1024
    cond_channels: int = 1280
    cond2_channels: int = 128
    num_blocks: int = 24
    num_refiner_blocks: int = 2
    attention_heads: int = 16
    attention_head_dim: int = 64
    cam_channels: int = 5
    mlp_ratio: int = 4
    repo_hidden_size: int = 128
    sampling_shift: float = 3.0
    memory_factor: float = 0.6
    inference_dtypes: tuple[DType, ...] = (FLOAT16, BFLOAT16, FLOAT32)

    def __post_init__(self) -> None:
        actual = (
            self.family_id,
            self.q_token_length,
            self.latent_channels,
            self.model_channels,
            self.cond_channels,
            self.cond2_channels,
            self.num_blocks,
            self.num_refiner_blocks,
            self.attention_heads,
            self.attention_head_dim,
            self.cam_channels,
            self.mlp_ratio,
            self.repo_hidden_size,
            self.sampling_shift,
            self.memory_factor,
            self.inference_dtypes,
        )
        expected = (
            "dinkster.triposplat",
            8192,
            16,
            1024,
            1280,
            128,
            24,
            2,
            16,
            64,
            5,
            4,
            128,
            3.0,
            0.6,
            (FLOAT16, BFLOAT16, FLOAT32),
        )
        if actual != expected:
            raise ValueError("TripoSplatConfig must carry the exact published architecture")


TRIPOSPLAT_CONFIG = TripoSplatConfig()

#: ModelSamplingDiscreteFlow with the family's shift
#: (comfy/supported_models.py TripoSplat sampling_settings @ 36408117).
TRIPOSPLAT_SIGMAS = FlowSigmas(shift=TRIPOSPLAT_CONFIG.sampling_shift)


@dataclass(frozen=True)
class TripoSplatGaussianDecoderConfig:
    """The exact octree gaussian decoder architecture requirements
    (comfy/ldm/triposplat/vae.py OctreeGaussianDecoder @ 36408117).

    ``latent_channels`` is the cross-attention context width: both
    decoders attend over the denoised 16-channel latent sequence.
    ``feature_channels`` is the elastic decoder's packed per-point
    output (32 gaussians x [xyz 3, color 3, scaling 3, rotation 4,
    opacity 1] + 32 offset scales = 480).
    """

    model_channels: int = 1024
    latent_channels: int = 16
    octree_blocks: int = 4
    gaussian_blocks: int = 16
    attention_heads: int = 16
    attention_head_dim: int = 64
    mlp_ratio: int = 4
    gaussians_per_point: int = 32
    feature_channels: int = 480
    max_voxel_level: int = 8

    def __post_init__(self) -> None:
        actual = (
            self.model_channels,
            self.latent_channels,
            self.octree_blocks,
            self.gaussian_blocks,
            self.attention_heads,
            self.attention_head_dim,
            self.mlp_ratio,
            self.gaussians_per_point,
            self.feature_channels,
            self.max_voxel_level,
        )
        expected = (1024, 16, 4, 16, 16, 64, 4, 32, 480, 8)
        if actual != expected:
            raise ValueError(
                "TripoSplatGaussianDecoderConfig must carry the exact published architecture"
            )


TRIPOSPLAT_GAUSSIAN_DECODER_CONFIG = TripoSplatGaussianDecoderConfig()

# Combined-checkpoint prefix first, bare diffusion file second
# (comfy/model_detection.py any_suffix_in scan @ 36408117).
_PREFIXES = ("model.diffusion_model.", "")


def triposplat_layout(
    config: TripoSplatConfig = TRIPOSPLAT_CONFIG,
) -> dict[str, tuple[int, ...]]:
    """The exact key -> shape listing of the TripoSplat DiT checkpoint
    (LatentSeqMMFlowModel @ 36408117; the Sobol ``pos_emb`` buffer is
    non-persistent and absent)."""
    hidden = config.model_channels
    heads = config.attention_heads
    head_dim = config.attention_head_dim
    repo_hidden = config.repo_hidden_size
    mlp_hidden = hidden * config.mlp_ratio
    layout: dict[str, tuple[int, ...]] = {
        "t_embedder.mlp.0.weight": (hidden, 256),
        "t_embedder.mlp.0.bias": (hidden,),
        "t_embedder.mlp.2.weight": (hidden, hidden),
        "t_embedder.mlp.2.bias": (hidden,),
        # share_mod: one modulation projection computed per forward.
        "adaLN_modulation.1.weight": (6 * hidden, hidden),
        "adaLN_modulation.1.bias": (6 * hidden,),
        "input_layer.weight": (hidden, config.latent_channels),
        "input_layer.bias": (hidden,),
        "cond_embedder.weight": (hidden, config.cond_channels),
        "cond_embedder.bias": (hidden,),
        "cond_embedder2.weight": (hidden, config.cond2_channels),
        "cond_embedder2.bias": (hidden,),
        "cam_refiner.mlp.0.weight": (hidden, config.cam_channels),
        "cam_refiner.mlp.0.bias": (hidden,),
        "cam_refiner.mlp.2.weight": (hidden, hidden),
        "cam_refiner.mlp.2.bias": (hidden,),
        "shift_table": (1, 2, hidden),
        "out_layer.weight": (config.latent_channels, hidden),
        "out_layer.bias": (config.latent_channels,),
        "cam_out_layer.weight": (config.cam_channels, hidden),
        "cam_out_layer.bias": (config.cam_channels,),
    }

    def repo(prefix: str) -> None:
        # RePo3DRotaryEmbedding: rotary angles are predicted per token;
        # head_dim splits 20/20/24 into three axes whose frequency
        # tables are trained parameters of half each axis' width.
        dim_0 = 2 * (head_dim // 6)
        dim_1 = 2 * (head_dim // 6)
        dim_2 = head_dim - dim_0 - dim_1
        layout[prefix + "norm.weight"] = (hidden,)
        layout[prefix + "norm.bias"] = (hidden,)
        layout[prefix + "gate_map.weight"] = (repo_hidden, hidden)
        layout[prefix + "content_map.weight"] = (repo_hidden, hidden)
        layout[prefix + "final_map.weight"] = (3 * heads, repo_hidden)
        layout[prefix + "freqs_0"] = (dim_0 // 2,)
        layout[prefix + "freqs_1"] = (dim_1 // 2,)
        layout[prefix + "freqs_2"] = (dim_2 // 2,)

    def attention(prefix: str) -> None:
        layout[prefix + "qkv.weight"] = (3 * hidden, hidden)
        layout[prefix + "qkv.bias"] = (3 * hidden,)
        layout[prefix + "q_norm.gamma"] = (heads, head_dim)
        layout[prefix + "k_norm.gamma"] = (heads, head_dim)
        layout[prefix + "out.weight"] = (hidden, hidden)
        layout[prefix + "out.bias"] = (hidden,)

    def mlp(prefix: str) -> None:
        layout[prefix + "mlp.0.weight"] = (mlp_hidden, hidden)
        layout[prefix + "mlp.0.bias"] = (mlp_hidden,)
        layout[prefix + "mlp.2.weight"] = (hidden, mlp_hidden)
        layout[prefix + "mlp.2.bias"] = (hidden,)

    for index in range(config.num_refiner_blocks):
        repo(f"noise_repo_layers.{index}.")
        repo(f"context_repo_layers.{index}.")
        # Noise refiner: modulated (norms carry no affine; the shared
        # modulation is offset by a per-block shift table).
        attention(f"noise_refiner.{index}.attn.")
        mlp(f"noise_refiner.{index}.mlp.")
        layout[f"noise_refiner.{index}.shift_table"] = (1, 6 * hidden)
        # Context refiner: unmodulated, so its norms are affine.
        attention(f"context_refiner.{index}.attn.")
        mlp(f"context_refiner.{index}.mlp.")
        for norm in ("norm1", "norm2"):
            layout[f"context_refiner.{index}.{norm}.weight"] = (hidden,)
            layout[f"context_refiner.{index}.{norm}.bias"] = (hidden,)
    for index in range(config.num_blocks):
        repo(f"repo_layers.{index}.")
        attention(f"blocks.{index}.attn.")
        mlp(f"blocks.{index}.mlp.")
        layout[f"blocks.{index}.shift_table"] = (1, 6 * hidden)
    return layout


def triposplat_gaussian_decoder_layout(
    config: TripoSplatGaussianDecoderConfig = TRIPOSPLAT_GAUSSIAN_DECODER_CONFIG,
) -> dict[str, tuple[int, ...]]:
    """The exact key -> shape listing of the octree gaussian decoder
    checkpoint (OctreeGaussianDecoder @ 36408117). The perturbation and
    offset-scale buffers are persistent and present."""
    hidden = config.model_channels
    context = config.latent_channels
    heads = config.attention_heads
    head_dim = config.attention_head_dim
    mlp_hidden = hidden * config.mlp_ratio
    layout: dict[str, tuple[int, ...]] = {
        "octree.input_layer.weight": (hidden, hidden),
        "octree.input_layer.bias": (hidden,),
        "octree.l_embedder.mlp.0.weight": (hidden, 256),
        "octree.l_embedder.mlp.0.bias": (hidden,),
        "octree.l_embedder.mlp.2.weight": (hidden, hidden),
        "octree.l_embedder.mlp.2.bias": (hidden,),
        "octree.adaLN_modulation.1.weight": (6 * hidden, hidden),
        "octree.adaLN_modulation.1.bias": (6 * hidden,),
        "octree.in_proj.weight": (hidden, 3),
        "octree.in_proj.bias": (hidden,),
        # 8-way child occupancy logits per octree node.
        "octree.out_proj.weight": (8, hidden),
        "octree.out_proj.bias": (8,),
        "gs.input_layer.weight": (hidden, hidden),
        "gs.input_layer.bias": (hidden,),
        "gs.in_proj.weight": (hidden, 3),
        "gs.in_proj.bias": (hidden,),
        "gs.out_proj.weight": (config.feature_channels, hidden),
        "gs.out_proj.bias": (config.feature_channels,),
        "gs.points_offset_perturbation": (config.gaussians_per_point, 3),
        "gs.base_offset_scale": (),
    }

    def cross_attention(prefix: str) -> None:
        layout[prefix + "to_q.weight"] = (hidden, hidden)
        layout[prefix + "to_q.bias"] = (hidden,)
        layout[prefix + "to_kv.weight"] = (2 * hidden, context)
        layout[prefix + "to_kv.bias"] = (2 * hidden,)
        layout[prefix + "q_rms_norm.gamma"] = (heads, head_dim)
        layout[prefix + "k_rms_norm.gamma"] = (heads, head_dim)
        layout[prefix + "to_out.weight"] = (hidden, hidden)
        layout[prefix + "to_out.bias"] = (hidden,)

    def mlp(prefix: str) -> None:
        layout[prefix + "mlp.0.weight"] = (mlp_hidden, hidden)
        layout[prefix + "mlp.0.bias"] = (mlp_hidden,)
        layout[prefix + "mlp.2.weight"] = (hidden, mlp_hidden)
        layout[prefix + "mlp.2.bias"] = (hidden,)

    for index in range(config.octree_blocks):
        # Cross-only modulated blocks; norms carry no affine and the
        # modulation projection is shared at the decoder root.
        cross_attention(f"octree.blocks.{index}.cross_attn.")
        mlp(f"octree.blocks.{index}.mlp.")
    for index in range(config.gaussian_blocks):
        # Self + cross blocks; only the cross-attention pre-norm is affine.
        prefix = f"gs.blocks.{index}."
        layout[prefix + "self_attn.to_qkv.weight"] = (3 * hidden, hidden)
        layout[prefix + "self_attn.to_qkv.bias"] = (3 * hidden,)
        layout[prefix + "self_attn.q_rms_norm.gamma"] = (heads, head_dim)
        layout[prefix + "self_attn.k_rms_norm.gamma"] = (heads, head_dim)
        layout[prefix + "self_attn.to_out.weight"] = (hidden, hidden)
        layout[prefix + "self_attn.to_out.bias"] = (hidden,)
        layout[prefix + "norm2.weight"] = (hidden,)
        layout[prefix + "norm2.bias"] = (hidden,)
        cross_attention(prefix + "cross_attn.")
        mlp(prefix + "mlp.")
    return layout


def _match_exact(
    geometries: Mapping[str, TensorGeometry],
    layout: Mapping[str, tuple[int, ...]],
    subject: str,
) -> None:
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
        raise TripoSplatDetectError(f"geometry does not match the {subject} layout: {shown}")


def detect_triposplat_config(
    geometries: Mapping[str, TensorGeometry],
) -> TripoSplatConfig:
    """Classify a diffusion-model-scoped header (any checkpoint prefix
    already stripped) as the published TripoSplat DiT, or refuse
    loudly. Dtypes are ignored; key sets and shapes must match
    exactly."""
    if not geometries:
        raise TripoSplatDetectError("empty state dict header")
    if "cam_out_layer.weight" not in geometries or "repo_layers.0.final_map.weight" not in (
        geometries
    ):
        raise TripoSplatDetectError(
            "not a TripoSplat DiT (no cam_out_layer.weight +"
            " repo_layers.0.final_map.weight marker pair)"
        )
    _match_exact(geometries, triposplat_layout(), "TripoSplat DiT")
    return TRIPOSPLAT_CONFIG


def detect_triposplat_gaussian_decoder(
    geometries: Mapping[str, TensorGeometry],
) -> TripoSplatGaussianDecoderConfig:
    """Classify a first-stage-scoped header as the published TripoSplat
    octree gaussian decoder, or refuse loudly."""
    if not geometries:
        raise TripoSplatDetectError("empty state dict header")
    if "gs.base_offset_scale" not in geometries or "octree.out_proj.weight" not in geometries:
        raise TripoSplatDetectError(
            "not a TripoSplat gaussian decoder (no gs.base_offset_scale +"
            " octree.out_proj.weight marker pair)"
        )
    _match_exact(
        geometries,
        triposplat_gaussian_decoder_layout(),
        "TripoSplat gaussian decoder",
    )
    return TRIPOSPLAT_GAUSSIAN_DECODER_CONFIG


@dataclass(frozen=True)
class TripoSplatEvidence:
    config: TripoSplatConfig
    key_prefix: str
    matched_keys: tuple[str, ...]
    fields: Mapping[str, EvidenceValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "matched_keys", tuple(self.matched_keys))
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))


def detect_triposplat(source: WeightSource) -> TripoSplatEvidence | None:
    """Family-detector seam over :func:`detect_triposplat_config`: scan
    the known checkpoint prefixes and return evidence for an exact
    match, or None (never raises for foreign checkpoints)."""
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
            config = detect_triposplat_config(scoped)
        except TripoSplatDetectError:
            continue
        matched = tuple(sorted(prefix + key for key in scoped))
        return TripoSplatEvidence(
            config=config,
            key_prefix=prefix,
            matched_keys=matched,
            fields={
                "cam_channels": config.cam_channels,
                "cond_channels": config.cond_channels,
                "cond2_channels": config.cond2_channels,
                "depth": config.num_blocks,
                "hidden_size": config.model_channels,
                "key_prefix": prefix,
                "latent_channels": config.latent_channels,
                "q_token_length": config.q_token_length,
            },
        )
    return None


@dataclass(frozen=True, slots=True)
class TripoSplatFamilyRegistration:
    """Exact split-artifact family facts: the DiT, the DINOv3 vision
    conditioner, the Flux2 VAE for reference-image latents, and the
    octree gaussian decoder ship as separate files."""

    id: str = field(default="dinkster.triposplat", init=False)
    aliases: tuple[str, ...] = field(default=(), init=False)
    display_name: str = field(default="TripoSplat", init=False)
    config: TripoSplatConfig = field(default=TRIPOSPLAT_CONFIG, init=False)
    sigmas: FlowSigmas = field(default=TRIPOSPLAT_SIGMAS, init=False)
    component_roles: tuple[str, ...] = field(
        default=(
            "dit",
            "dinov3-vision-conditioner",
            "reference-latent-vae",
            "gaussian-decoder",
        ),
        init=False,
    )

    def detect(self, source: WeightSource) -> TripoSplatEvidence | None:
        """Detect the registered exact profile without reading payloads."""

        return detect_triposplat(source)


TRIPOSPLAT_FAMILY = TripoSplatFamilyRegistration()


__all__ = [
    "TRIPOSPLAT_CONFIG",
    "TRIPOSPLAT_FAMILY",
    "TRIPOSPLAT_GAUSSIAN_DECODER_CONFIG",
    "TRIPOSPLAT_SIGMAS",
    "TripoSplatConfig",
    "TripoSplatDetectError",
    "TripoSplatEvidence",
    "TripoSplatFamilyRegistration",
    "TripoSplatGaussianDecoderConfig",
    "detect_triposplat",
    "detect_triposplat_config",
    "detect_triposplat_gaussian_decoder",
    "triposplat_gaussian_decoder_layout",
    "triposplat_layout",
]
