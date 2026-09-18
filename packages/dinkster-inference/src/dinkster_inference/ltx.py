"""Torch-free LTX-Video and LTX audio-video checkpoint facts.

Detection and runtime constants follow ``comfy/model_detection.py``,
``comfy/supported_models.py``, ``comfy/latent_formats.py``, and
``comfy/model_sampling.py`` at ComfyUI ``82f839f5``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Literal, cast

from .codecs import CodecDescriptor, CodecTiling
from .devices import BFLOAT16, FLOAT32
from .families import DetectionEvidence
from .gemma_text import (
    LTXAV_22B_V23_AUDIO_CONNECTOR_CONFIG,
    LTXAV_22B_V23_VIDEO_CONNECTOR_CONFIG,
    LtxConnectorConfig,
    ltx_connector_layout,
)
from .latents import LatentDescriptor, MultiStreamLatentDescriptor
from .sampling import Parameterization, SamplingDescriptor
from .spaces import FluxFlowSigmas
from .weights import WeightSource

_PREFIXES = ("model.diffusion_model.", "")
_MARKER = "adaln_single.emb.timestep_embedder.linear_1.bias"
_AUDIO_MARKER = "audio_adaln_single.linear.weight"


@dataclass(frozen=True, slots=True)
class _Profile:
    name: str
    family_id: Literal["dinkster.ltxv", "dinkster.ltxav"]
    layers: int
    hidden: int
    head_dim: int
    caption_channels: int
    ffn: int
    adaln_rows: int
    audio_hidden: int | None = None
    audio_adaln_rows: int | None = None
    prompt_adaln: bool = False
    ff_bias: bool = True


_PROFILES = (
    _Profile("2b-v0.9", "dinkster.ltxv", 28, 2048, 64, 4096, 8192, 12288),
    _Profile("2b-v0.9.5", "dinkster.ltxv", 28, 2048, 64, 4096, 8192, 12288),
    _Profile("19b", "dinkster.ltxav", 48, 4096, 128, 3840, 16384, 24576, 2048, 12288),
    _Profile("22b", "dinkster.ltxav", 48, 4096, 128, 3840, 16384, 36864, 2048, 18432, True),
    _Profile(
        "22b-v2.5",
        "dinkster.ltxav",
        48,
        4096,
        128,
        3840,
        16384,
        36864,
        2048,
        18432,
        True,
        False,
    ),
)


def _shape(source: WeightSource, keys: frozenset[str], key: str) -> tuple[int, ...] | None:
    if key not in keys:
        return None
    try:
        return source.entry(key).geometry.shape
    except KeyError:
        return None


def _linear_shape(
    source: WeightSource, keys: frozenset[str], key: str, rows: int, columns: int
) -> bool:
    shape = _shape(source, keys, key)
    if shape is None or len(shape) != 2 or shape[0] != rows:
        return False
    # Packed 4-bit weights halve the input axis; output geometry remains authoritative.
    return shape[1] in (columns, columns // 2)


def _layers(keys: frozenset[str], prefix: str) -> int | None:
    root = prefix + "transformer_blocks."
    indices: set[int] = set()
    for key in keys:
        if key.startswith(root):
            text = key[len(root) :].partition(".")[0]
            if text.isdigit():
                indices.add(int(text))
    if not indices or indices != set(range(max(indices) + 1)):
        return None
    return len(indices)


def _match(
    source: WeightSource, keys: frozenset[str], prefix: str, profile: _Profile
) -> DetectionEvidence | None:
    marker = prefix + _MARKER
    audio_marker = prefix + _AUDIO_MARKER
    av = audio_marker in keys
    if (
        marker not in keys
        or av != (profile.family_id == "dinkster.ltxav")
        or _layers(keys, prefix) != profile.layers
    ):
        return None
    prompt = prefix + "prompt_adaln_single.emb.timestep_embedder.linear_1.bias" in keys
    ff_bias = prefix + "transformer_blocks.0.ff.net.0.proj.bias" in keys
    if prompt != profile.prompt_adaln or ff_bias != profile.ff_bias:
        return None
    required = (
        _shape(source, keys, marker) == (profile.hidden,),
        _linear_shape(source, keys, prefix + "patchify_proj.weight", profile.hidden, 128),
        _linear_shape(source, keys, prefix + "proj_out.weight", 128, profile.hidden),
        _linear_shape(
            source, keys, prefix + "adaln_single.linear.weight", profile.adaln_rows, profile.hidden
        ),
        _linear_shape(
            source,
            keys,
            prefix + "transformer_blocks.0.ff.net.0.proj.weight",
            profile.ffn,
            profile.hidden,
        ),
        _linear_shape(
            source,
            keys,
            prefix + "transformer_blocks.0.attn2.to_k.weight",
            profile.hidden,
            profile.hidden,
        ),
    )
    matched = [marker]
    if not all(required):
        return None
    if av:
        assert profile.audio_hidden is not None and profile.audio_adaln_rows is not None
        if not all(
            (
                _linear_shape(
                    source, keys, prefix + "audio_patchify_proj.weight", profile.audio_hidden, 128
                ),
                _linear_shape(
                    source, keys, prefix + "audio_proj_out.weight", 128, profile.audio_hidden
                ),
                _linear_shape(
                    source, keys, audio_marker, profile.audio_adaln_rows, profile.audio_hidden
                ),
            )
        ):
            return None
        matched.append(audio_marker)
    matched.extend(
        (
            prefix + "patchify_proj.weight",
            prefix + "proj_out.weight",
            prefix + "adaln_single.linear.weight",
            prefix + "transformer_blocks.0.ff.net.0.proj.weight",
        )
    )
    return DetectionEvidence(
        profile.family_id,
        tuple(matched),
        {
            "profile": profile.name,
            "key_prefix": prefix,
            "layers": profile.layers,
            "hidden_width": profile.hidden,
            "attention_head_dim": profile.head_dim,
            "attention_heads": 32,
            "caption_channels": profile.caption_channels,
            "av": av,
            "prompt_adaln": prompt,
            "ff_bias": ff_bias,
        },
    )


def _metadata_causal_positioning(source: WeightSource) -> bool | Literal["refuse"]:
    """The transformer causal_temporal_positioning flag from the safetensors
    config metadata; False without metadata (the reference model default) and
    refuse on malformed metadata."""
    encoded = source.metadata().get("config")
    if encoded is None:
        return False
    try:
        decoded = cast("object", json.loads(encoded))
    except json.JSONDecodeError:
        return "refuse"
    if type(decoded) is not dict:
        return "refuse"
    transformer: object = cast("dict[object, object]", decoded).get("transformer", {})
    if type(transformer) is not dict:
        return "refuse"
    value: object = cast("dict[object, object]", transformer).get(
        "causal_temporal_positioning", False
    )
    if type(value) is not bool:
        return "refuse"
    return value


def _detect(source: WeightSource, family_id: str) -> DetectionEvidence | None:
    # The reference merges the config metadata into every LTX model
    # configuration (comfy/model_detection.py @ 82f839f5), so malformed
    # metadata refuses both families.
    causal = _metadata_causal_positioning(source)
    if causal == "refuse":
        return None
    keys = frozenset(source.keys())
    matches = [
        evidence
        for prefix in _PREFIXES
        for profile in _PROFILES
        if profile.family_id == family_id
        if (evidence := _match(source, keys, prefix, profile)) is not None
    ]
    # The two 2B releases have identical tensor geometry; the reference
    # separates them only by the causal_temporal_positioning metadata flag.
    if family_id == "dinkster.ltxv" and len(matches) == 2:
        wanted = "2b-v0.9.5" if causal else "2b-v0.9"
        matches = [item for item in matches if item.fields["profile"] == wanted]
    return matches[0] if len(matches) == 1 else None


def detect_ltxv(source: WeightSource) -> DetectionEvidence | None:
    return _detect(source, "dinkster.ltxv")


def detect_ltxav(source: WeightSource) -> DetectionEvidence | None:
    return _detect(source, "dinkster.ltxav")


@dataclass(frozen=True, slots=True)
class LTXVDetector:
    def detect(self, source: WeightSource) -> DetectionEvidence | None:
        return detect_ltxv(source)


@dataclass(frozen=True, slots=True)
class LTXAVDetector:
    def detect(self, source: WeightSource) -> DetectionEvidence | None:
        return detect_ltxav(source)


#: Constants the reference pins rather than derives for the LTX-Video 2B
#: line (comfy/ldm/lightricks/model.py LTXVModel defaults @ b78cec87).
LTXV_THETA = 10000.0
LTXV_MAX_POS = (20, 2048, 2048)
LTXV_VAE_SCALE_FACTORS = (8, 32, 32)
LTXV_TIMESTEP_MULTIPLIER = 1000.0
LTXV_TIME_PROJ_CHANNELS = 256


@dataclass(frozen=True, slots=True)
class LTXVConfig:
    """Exact geometry of one LTX-Video 2B diffusion transformer.

    Field names follow the reference constructor
    (comfy/ldm/lightricks/model.py LTXVModel @ b78cec87). The two
    published 2B releases share one tensor layout and differ only in
    ``causal_temporal_positioning`` (v0.9.5 anchors the first frame's
    temporal coordinate causally; v0.9 does not).
    """

    in_channels: int = 128
    cross_attention_dim: int = 2048
    attention_head_dim: int = 64
    num_attention_heads: int = 32
    caption_channels: int = 4096
    num_layers: int = 28
    causal_temporal_positioning: bool = False

    def __post_init__(self) -> None:
        if self.hidden_size != self.cross_attention_dim:
            raise ValueError(
                "LTXV context rows are viewed as hidden-size rows after caption"
                f" projection; cross_attention_dim {self.cross_attention_dim} must"
                f" equal hidden size {self.hidden_size}"
            )

    @property
    def hidden_size(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def ffn_dim(self) -> int:
        return 4 * self.hidden_size


LTXV_2B_V09_CONFIG = LTXVConfig(causal_temporal_positioning=False)
LTXV_2B_V095_CONFIG = LTXVConfig(causal_temporal_positioning=True)


def ltxv_layout(config: LTXVConfig) -> dict[str, tuple[int, ...]]:
    """Every state-dict key and shape of the LTXV diffusion transformer,
    without the checkpoint's ``model.diffusion_model.`` prefix."""
    hidden = config.hidden_size
    layout: dict[str, tuple[int, ...]] = {
        "patchify_proj.weight": (hidden, config.in_channels),
        "patchify_proj.bias": (hidden,),
        "adaln_single.emb.timestep_embedder.linear_1.weight": (
            hidden,
            LTXV_TIME_PROJ_CHANNELS,
        ),
        "adaln_single.emb.timestep_embedder.linear_1.bias": (hidden,),
        "adaln_single.emb.timestep_embedder.linear_2.weight": (hidden, hidden),
        "adaln_single.emb.timestep_embedder.linear_2.bias": (hidden,),
        "adaln_single.linear.weight": (6 * hidden, hidden),
        "adaln_single.linear.bias": (6 * hidden,),
        "caption_projection.linear_1.weight": (hidden, config.caption_channels),
        "caption_projection.linear_1.bias": (hidden,),
        "caption_projection.linear_2.weight": (hidden, hidden),
        "caption_projection.linear_2.bias": (hidden,),
        "scale_shift_table": (2, hidden),
        "proj_out.weight": (config.in_channels, hidden),
        "proj_out.bias": (config.in_channels,),
    }
    for index in range(config.num_layers):
        block = f"transformer_blocks.{index}."
        layout[block + "scale_shift_table"] = (6, hidden)
        for attn, context_dim in (("attn1.", hidden), ("attn2.", config.cross_attention_dim)):
            layout[block + attn + "q_norm.weight"] = (hidden,)
            layout[block + attn + "k_norm.weight"] = (hidden,)
            layout[block + attn + "to_q.weight"] = (hidden, hidden)
            layout[block + attn + "to_q.bias"] = (hidden,)
            layout[block + attn + "to_k.weight"] = (hidden, context_dim)
            layout[block + attn + "to_k.bias"] = (hidden,)
            layout[block + attn + "to_v.weight"] = (hidden, context_dim)
            layout[block + attn + "to_v.bias"] = (hidden,)
            layout[block + attn + "to_out.0.weight"] = (hidden, hidden)
            layout[block + attn + "to_out.0.bias"] = (hidden,)
        layout[block + "ff.net.0.proj.weight"] = (config.ffn_dim, hidden)
        layout[block + "ff.net.0.proj.bias"] = (config.ffn_dim,)
        layout[block + "ff.net.2.weight"] = (hidden, config.ffn_dim)
        layout[block + "ff.net.2.bias"] = (hidden,)
    return layout


#: Constants the reference pins rather than derives for the LTX-2
#: audio-video line (comfy/ldm/lightricks/av_model.py LTXAVModel and
#: symmetric_patchifier.py AudioPatchifier @ b78cec87). Audio latents are
#: ``[B, 8, T, 16]`` mel spectrogram patches; one latent frame spans
#: LTXAV_AUDIO_LATENT_DOWNSAMPLE mel hops of LTXAV_AUDIO_HOP_LENGTH
#: samples at LTXAV_AUDIO_SAMPLE_RATE Hz.
LTXAV_AUDIO_CHANNELS = 8
LTXAV_AUDIO_FREQUENCY_BINS = 16
LTXAV_AUDIO_MAX_POS = (20.0,)
LTXAV_AUDIO_SAMPLE_RATE = 16000
LTXAV_AUDIO_HOP_LENGTH = 160
LTXAV_AUDIO_LATENT_DOWNSAMPLE = 4

#: The embeddings-connector prefixes shared by LTXAV checkpoints. LTX-2
#: 19B routes them into the text stack; the 22B profiles keep them in the
#: diffusion model.
LTXAV_CONNECTOR_PREFIXES = (
    "audio_embeddings_connector.",
    "video_embeddings_connector.",
)


@dataclass(frozen=True, slots=True)
class LTXGeneratedKeyframes:
    """Latent-frame slots that carry LTX's learned keyframe marker."""

    tokens_per_frame: int
    first_latent_frame: int
    num_keyframes: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int
            for value in (self.tokens_per_frame, self.first_latent_frame, self.num_keyframes)
        ):
            raise TypeError("LTX generated-keyframe fields must be exact integers")
        if self.tokens_per_frame <= 0:
            raise ValueError("LTX generated-keyframe tokens_per_frame must be positive")
        if self.first_latent_frame < 0 or self.num_keyframes < 0:
            raise ValueError("LTX generated-keyframe frame values must be nonnegative")


@dataclass(frozen=True, slots=True)
class LTXAVConfig:
    """Exact geometry of one LTX-2 audio-video diffusion transformer.

    Field names follow the reference constructor
    (comfy/ldm/lightricks/av_model.py LTXAVModel @ b78cec87); defaults
    are the published 19B checkpoint's metadata config merged over the
    reference detection facts. LTX-2 19B routes its connector tensors
    through the text stack, while the 22B profiles own asymmetric
    connector towers in the diffusion model.
    """

    in_channels: int = 128
    cross_attention_dim: int = 4096
    attention_head_dim: int = 128
    num_attention_heads: int = 32
    audio_in_channels: int = 128
    audio_cross_attention_dim: int = 2048
    audio_attention_head_dim: int = 64
    audio_num_attention_heads: int = 32
    caption_channels: int = 3840
    num_layers: int = 48
    causal_temporal_positioning: bool = True
    use_middle_indices_grid: bool = True
    av_ca_timestep_scale_multiplier: float = 1000.0
    cross_attention_adaln: bool = False
    ff_bias: bool = True
    audio_ff_bias: bool = True
    caption_proj_before_connector: bool = field(default=False, repr=False)
    gated_attention: bool = field(default=False, repr=False)
    use_keyframes_abs_pos_embedding: bool = field(default=False, repr=False)
    video_connector: LtxConnectorConfig | None = field(default=None, repr=False)
    audio_connector: LtxConnectorConfig | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if any(
            type(value) is not bool
            for value in (
                self.ff_bias,
                self.audio_ff_bias,
                self.use_keyframes_abs_pos_embedding,
            )
        ):
            raise TypeError("LTXAV feature flags must be exact bools")
        if self.hidden_size != self.cross_attention_dim:
            raise ValueError(
                "LTXAV video context rows are viewed as hidden-size rows after"
                f" caption projection; cross_attention_dim {self.cross_attention_dim}"
                f" must equal hidden size {self.hidden_size}"
            )
        if self.audio_hidden_size != self.audio_cross_attention_dim:
            raise ValueError(
                "LTXAV audio context rows are viewed as audio-hidden rows after"
                " caption projection; audio_cross_attention_dim"
                f" {self.audio_cross_attention_dim} must equal audio hidden size"
                f" {self.audio_hidden_size}"
            )
        if self.audio_in_channels != LTXAV_AUDIO_CHANNELS * LTXAV_AUDIO_FREQUENCY_BINS:
            raise ValueError(
                "LTXAV audio tokens flatten 8 latent channels by 16 frequency"
                f" bins; audio_in_channels must be 128, got {self.audio_in_channels}"
            )
        if (self.video_connector is None) != (self.audio_connector is None):
            raise ValueError("LTXAV video and audio connectors must be configured together")
        if self.caption_proj_before_connector != (self.video_connector is not None):
            raise ValueError(
                "diffusion-owned LTXAV connectors require caption_proj_before_connector"
            )
        if self.video_connector is not None:
            assert self.audio_connector is not None
            if self.video_connector.inner_dim != self.cross_attention_dim:
                raise ValueError("LTXAV video connector width must equal cross_attention_dim")
            if self.audio_connector.inner_dim != self.audio_cross_attention_dim:
                raise ValueError("LTXAV audio connector width must equal audio_cross_attention_dim")

    @property
    def hidden_size(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def audio_hidden_size(self) -> int:
        return self.audio_num_attention_heads * self.audio_attention_head_dim

    @property
    def ffn_dim(self) -> int:
        return 4 * self.hidden_size

    @property
    def audio_ffn_dim(self) -> int:
        return 4 * self.audio_hidden_size

    @property
    def adaln_rows(self) -> int:
        """Modulation rows per stream table (three extra rows drive the
        text cross-attention when it is adaLN-modulated)."""
        return 9 if self.cross_attention_adaln else 6


LTXAV_19B_CONFIG = LTXAVConfig()
LTXAV_22B_V23_CONFIG = LTXAVConfig(
    cross_attention_adaln=True,
    caption_proj_before_connector=True,
    gated_attention=True,
    video_connector=LTXAV_22B_V23_VIDEO_CONNECTOR_CONFIG,
    audio_connector=LTXAV_22B_V23_AUDIO_CONNECTOR_CONFIG,
)
LTXAV_22B_V25_CONFIG = LTXAVConfig(
    cross_attention_adaln=True,
    ff_bias=False,
    caption_proj_before_connector=True,
    gated_attention=True,
    video_connector=LTXAV_22B_V23_VIDEO_CONNECTOR_CONFIG,
    audio_connector=LTXAV_22B_V23_AUDIO_CONNECTOR_CONFIG,
)


@dataclass(frozen=True, slots=True)
class LTXDurationHeadConfig:
    """Exact geometry of the LTX-2.4 prompt duration head."""

    video_input_dim: int = 4096
    audio_input_dim: int = 2048
    hidden_dim: int = 256
    num_queries: int = 1
    num_heads: int = 4
    mlp_hidden_dim: int = 256

    def __post_init__(self) -> None:
        if (
            min(
                self.video_input_dim,
                self.audio_input_dim,
                self.hidden_dim,
                self.num_queries,
                self.num_heads,
                self.mlp_hidden_dim,
            )
            <= 0
        ):
            raise ValueError("LTX duration head dimensions must be positive")
        if self.hidden_dim % self.num_heads:
            raise ValueError("LTX duration head width must divide into attention heads")


LTXAV_DURATION_HEAD_CONFIG = LTXDurationHeadConfig()


def ltxav_duration_head_layout(
    config: LTXDurationHeadConfig,
) -> dict[str, tuple[int, ...]]:
    """Every tensor in the standalone LTX-2.4 prompt duration head."""
    hidden = config.hidden_dim
    queries = config.num_queries
    return {
        "video_input_proj.weight": (hidden, config.video_input_dim),
        "video_input_proj.bias": (hidden,),
        "video_modality_emb": (hidden,),
        "audio_input_proj.weight": (hidden, config.audio_input_dim),
        "audio_input_proj.bias": (hidden,),
        "audio_modality_emb": (hidden,),
        "attention_pooler.query_tokens": (queries, hidden),
        "attention_pooler.cross_attn.in_proj_weight": (3 * hidden, hidden),
        "attention_pooler.cross_attn.in_proj_bias": (3 * hidden,),
        "attention_pooler.cross_attn.out_proj.weight": (hidden, hidden),
        "attention_pooler.cross_attn.out_proj.bias": (hidden,),
        "mlp_hidden.weight": (config.mlp_hidden_dim, hidden * queries),
        "mlp_hidden.bias": (config.mlp_hidden_dim,),
        "mlp_out.weight": (1, config.mlp_hidden_dim),
        "mlp_out.bias": (1,),
    }


def _ltxav_adaln_layout(key: str, hidden: int, rows: int) -> dict[str, tuple[int, ...]]:
    return {
        key + ".emb.timestep_embedder.linear_1.weight": (hidden, LTXV_TIME_PROJ_CHANNELS),
        key + ".emb.timestep_embedder.linear_1.bias": (hidden,),
        key + ".emb.timestep_embedder.linear_2.weight": (hidden, hidden),
        key + ".emb.timestep_embedder.linear_2.bias": (hidden,),
        key + ".linear.weight": (rows * hidden, hidden),
        key + ".linear.bias": (rows * hidden,),
    }


def _ltxav_attention_layout(
    prefix: str,
    query_dim: int,
    context_dim: int,
    inner: int,
    heads: int,
    gated: bool,
) -> dict[str, tuple[int, ...]]:
    layout = {
        prefix + "q_norm.weight": (inner,),
        prefix + "k_norm.weight": (inner,),
        prefix + "to_q.weight": (inner, query_dim),
        prefix + "to_q.bias": (inner,),
        prefix + "to_k.weight": (inner, context_dim),
        prefix + "to_k.bias": (inner,),
        prefix + "to_v.weight": (inner, context_dim),
        prefix + "to_v.bias": (inner,),
        prefix + "to_out.0.weight": (query_dim, inner),
        prefix + "to_out.0.bias": (query_dim,),
    }
    if gated:
        layout[prefix + "to_gate_logits.weight"] = (heads, query_dim)
        layout[prefix + "to_gate_logits.bias"] = (heads,)
    return layout


def ltxav_layout(config: LTXAVConfig) -> dict[str, tuple[int, ...]]:
    """Every state-dict key and shape of the LTXAV diffusion transformer,
    without the checkpoint's ``model.diffusion_model.`` prefix. The 19B
    layout excludes its text-side connector tensors; the 2.3 22B layout
    includes its diffusion-owned connectors."""
    hidden = config.hidden_size
    audio = config.audio_hidden_size
    rows = config.adaln_rows
    layout: dict[str, tuple[int, ...]] = {
        "patchify_proj.weight": (hidden, config.in_channels),
        "patchify_proj.bias": (hidden,),
        **_ltxav_adaln_layout("adaln_single", hidden, rows),
        "audio_patchify_proj.weight": (audio, config.audio_in_channels),
        "audio_patchify_proj.bias": (audio,),
        **_ltxav_adaln_layout("audio_adaln_single", audio, rows),
        **_ltxav_adaln_layout("av_ca_video_scale_shift_adaln_single", hidden, 4),
        **_ltxav_adaln_layout("av_ca_a2v_gate_adaln_single", hidden, 1),
        **_ltxav_adaln_layout("av_ca_audio_scale_shift_adaln_single", audio, 4),
        **_ltxav_adaln_layout("av_ca_v2a_gate_adaln_single", audio, 1),
        "scale_shift_table": (2, hidden),
        "proj_out.weight": (config.in_channels, hidden),
        "proj_out.bias": (config.in_channels,),
        "audio_scale_shift_table": (2, audio),
        "audio_proj_out.weight": (config.audio_in_channels, audio),
        "audio_proj_out.bias": (config.audio_in_channels,),
    }
    if config.use_keyframes_abs_pos_embedding:
        layout["keyframes_abs_pos_embedding"] = (1, hidden)
    if config.caption_proj_before_connector:
        assert config.video_connector is not None and config.audio_connector is not None
        for prefix, connector in (
            ("video_embeddings_connector.", config.video_connector),
            ("audio_embeddings_connector.", config.audio_connector),
        ):
            layout.update(
                {prefix + key: shape for key, shape in ltx_connector_layout(connector).items()}
            )
    else:
        layout.update(
            {
                "caption_projection.linear_1.weight": (hidden, config.caption_channels),
                "caption_projection.linear_1.bias": (hidden,),
                "caption_projection.linear_2.weight": (hidden, hidden),
                "caption_projection.linear_2.bias": (hidden,),
                "audio_caption_projection.linear_1.weight": (audio, config.caption_channels),
                "audio_caption_projection.linear_1.bias": (audio,),
                "audio_caption_projection.linear_2.weight": (audio, audio),
                "audio_caption_projection.linear_2.bias": (audio,),
            }
        )
    if config.cross_attention_adaln:
        layout.update(_ltxav_adaln_layout("prompt_adaln_single", hidden, 2))
        layout.update(_ltxav_adaln_layout("audio_prompt_adaln_single", audio, 2))
    for index in range(config.num_layers):
        block = f"transformer_blocks.{index}."
        layout[block + "scale_shift_table"] = (rows, hidden)
        layout[block + "audio_scale_shift_table"] = (rows, audio)
        if config.cross_attention_adaln:
            layout[block + "prompt_scale_shift_table"] = (2, hidden)
            layout[block + "audio_prompt_scale_shift_table"] = (2, audio)
        layout[block + "scale_shift_table_a2v_ca_audio"] = (5, audio)
        layout[block + "scale_shift_table_a2v_ca_video"] = (5, hidden)
        layout.update(
            _ltxav_attention_layout(
                block + "attn1.",
                hidden,
                hidden,
                hidden,
                config.num_attention_heads,
                config.gated_attention,
            )
        )
        layout.update(
            _ltxav_attention_layout(
                block + "attn2.",
                hidden,
                config.cross_attention_dim,
                hidden,
                config.num_attention_heads,
                config.gated_attention,
            )
        )
        layout.update(
            _ltxav_attention_layout(
                block + "audio_attn1.",
                audio,
                audio,
                audio,
                config.audio_num_attention_heads,
                config.gated_attention,
            )
        )
        layout.update(
            _ltxav_attention_layout(
                block + "audio_attn2.",
                audio,
                config.audio_cross_attention_dim,
                audio,
                config.audio_num_attention_heads,
                config.gated_attention,
            )
        )
        layout.update(
            _ltxav_attention_layout(
                block + "audio_to_video_attn.",
                hidden,
                audio,
                audio,
                config.audio_num_attention_heads,
                config.gated_attention,
            )
        )
        layout.update(
            _ltxav_attention_layout(
                block + "video_to_audio_attn.",
                audio,
                hidden,
                audio,
                config.audio_num_attention_heads,
                config.gated_attention,
            )
        )
        layout[block + "ff.net.0.proj.weight"] = (config.ffn_dim, hidden)
        layout[block + "ff.net.2.weight"] = (hidden, config.ffn_dim)
        if config.ff_bias:
            layout[block + "ff.net.0.proj.bias"] = (config.ffn_dim,)
            layout[block + "ff.net.2.bias"] = (hidden,)
        layout[block + "audio_ff.net.0.proj.weight"] = (config.audio_ffn_dim, audio)
        layout[block + "audio_ff.net.2.weight"] = (audio, config.audio_ffn_dim)
        if config.audio_ff_bias:
            layout[block + "audio_ff.net.0.proj.bias"] = (config.audio_ffn_dim,)
            layout[block + "audio_ff.net.2.bias"] = (audio,)
    return layout


#: Reference decode conditioning defaults
#: (comfy/ldm/lightricks/vae/causal_video_autoencoder.py VideoVAE @ b78cec87).
LTXV_VAE_DECODE_NOISE_SCALE = 0.025
LTXV_VAE_DECODE_TIMESTEP = 0.05

LTXVAEBlockKind = Literal[
    "res_x",
    "res_x_y",
    "compress_time",
    "compress_space",
    "compress_all",
    "compress_time_res",
    "compress_space_res",
    "compress_all_res",
]

_COMPRESS_STRIDES: dict[str, tuple[int, int, int]] = {
    "compress_time": (2, 1, 1),
    "compress_space": (1, 2, 2),
    "compress_all": (2, 2, 2),
    "compress_time_res": (2, 1, 1),
    "compress_space_res": (1, 2, 2),
    "compress_all_res": (2, 2, 2),
}


@dataclass(frozen=True, slots=True)
class LTXVAEBlock:
    """One block of the LTX causal video VAE, in the vocabulary of the
    reference block lists (causal_video_autoencoder.py @ b78cec87).

    ``multiplier`` multiplies the channel count in the encoder and divides
    it in the decoder (where compress blocks also use it as the
    ``out_channels_reduction_factor``); the reference's ``compress_all_x_y``
    is ``compress_all`` with a multiplier. ``layers`` counts res blocks
    inside ``res_x``. ``inject_noise`` and ``residual`` are decoder-only
    reference options.
    """

    kind: LTXVAEBlockKind
    layers: int = 1
    multiplier: int = 1
    inject_noise: bool = False
    residual: bool = False

    def __post_init__(self) -> None:
        if self.kind not in ("res_x", "res_x_y") and self.kind not in _COMPRESS_STRIDES:
            raise ValueError(f"unknown LTX VAE block kind {self.kind!r}")
        if self.layers < 1:
            raise ValueError("LTX VAE block layer count must be positive")
        if self.multiplier < 1:
            raise ValueError("LTX VAE block multiplier must be positive")
        if self.multiplier != 1 and self.kind == "res_x":
            raise ValueError("res_x LTX VAE blocks do not change channel width")
        if self.layers != 1 and self.kind != "res_x":
            raise ValueError("only res_x LTX VAE blocks stack layers")
        if self.inject_noise and self.kind not in ("res_x", "res_x_y"):
            raise ValueError("only res LTX VAE blocks inject noise")
        if self.inject_noise and self.multiplier != 1:
            raise ValueError("noise injection needs equal input and output channel widths")
        if self.residual and self.kind not in ("compress_time", "compress_space", "compress_all"):
            raise ValueError("only decoder compress LTX VAE blocks take the residual skip")
        if self.kind in _COMPRESS_STRIDES:
            packed = self.stride[0] * self.stride[1] * self.stride[2]
            if self.kind.endswith("_res") and packed % self.multiplier:
                raise ValueError(
                    f"{self.kind} averages groups of {packed} packed channels; its"
                    f" multiplier {self.multiplier} must divide that packing"
                )
            if self.residual and packed % self.multiplier:
                raise ValueError(
                    f"the {self.kind} residual skip repeats {packed} packed channels; its"
                    f" multiplier {self.multiplier} must divide that packing"
                )

    @property
    def stride(self) -> tuple[int, int, int]:
        stride = _COMPRESS_STRIDES.get(self.kind)
        if stride is None:
            raise ValueError(f"{self.kind} LTX VAE blocks do not resample")
        return stride


@dataclass(frozen=True, slots=True)
class LTXVideoVAEConfig:
    """Exact geometry of one LTX causal video VAE.

    Field names follow the reference configuration dictionaries
    (causal_video_autoencoder.py VideoVAE @ b78cec87). ``decoder_blocks``
    is kept in the reference configuration order; module construction
    walks it reversed, exactly like the reference. Aspects every
    supported checkpoint shares are pinned rather than configurable:
    pixel normalization, ``uniform`` latent log-variance, three image
    channels, and 3D convolutions.
    """

    encoder_blocks: tuple[LTXVAEBlock, ...]
    decoder_blocks: tuple[LTXVAEBlock, ...]
    latent_channels: int = 128
    base_channels: int = 128
    patch_size: int = 4
    causal_decoder: bool = False
    timestep_conditioning: bool = False
    decode_noise_scale: float = LTXV_VAE_DECODE_NOISE_SCALE
    decode_timestep: float = LTXV_VAE_DECODE_TIMESTEP
    encoder_spatial_padding_mode: Literal["zeros", "reflect"] = "zeros"
    decoder_spatial_padding_mode: Literal["zeros", "reflect"] = "reflect"

    def __post_init__(self) -> None:
        if not self.encoder_blocks or not self.decoder_blocks:
            raise ValueError("LTX VAE configurations must name encoder and decoder blocks")
        if self.latent_channels <= 0 or self.base_channels <= 0:
            raise ValueError("LTX VAE channel counts must be positive")
        if self.patch_size < 1 or self.patch_size & (self.patch_size - 1):
            raise ValueError("LTX VAE patch size must be a power of two")
        for block in self.encoder_blocks:
            if block.inject_noise or block.residual:
                raise ValueError("noise injection and residual skips are decoder options")
        for block in self.decoder_blocks:
            if block.kind.endswith("_res"):
                raise ValueError("space-to-depth compression blocks are encoder blocks")
        for mode in (self.encoder_spatial_padding_mode, self.decoder_spatial_padding_mode):
            if mode not in ("zeros", "reflect"):
                raise ValueError(f"LTX VAE spatial padding must be zeros or reflect, not {mode!r}")
        for block, (_, out) in zip(self.encoder_blocks, self.encoder_channels(), strict=True):
            if block.kind.endswith("_res"):
                packed = block.stride[0] * block.stride[1] * block.stride[2]
                if out % packed:
                    raise ValueError(
                        f"encoder {block.kind} output width {out} is not divisible by"
                        f" its space-to-depth packing {packed}"
                    )
        self.decoder_channels()  # refuses non-divisible channel walks

    def encoder_channels(self) -> tuple[tuple[int, int], ...]:
        """Per-block (in, out) channels, in ``encoder_blocks`` order."""
        walk: list[tuple[int, int]] = []
        channels = self.base_channels
        for block in self.encoder_blocks:
            walk.append((channels, channels * block.multiplier))
            channels *= block.multiplier
        return tuple(walk)

    def decoder_channels(self) -> tuple[tuple[int, int], ...]:
        """Per-block (in, out) channels, in module order (the reverse of
        ``decoder_blocks``)."""
        walk: list[tuple[int, int]] = []
        channels = self.decoder_input_channels
        for block in reversed(self.decoder_blocks):
            if channels % block.multiplier:
                raise ValueError(
                    f"decoder {block.kind} multiplier {block.multiplier} does not divide"
                    f" its {channels} input channels"
                )
            walk.append((channels, channels // block.multiplier))
            channels //= block.multiplier
        return tuple(walk)

    @property
    def decoder_input_channels(self) -> int:
        channels = self.base_channels
        for block in self.decoder_blocks:
            channels *= block.multiplier
        return channels

    @property
    def spatial_ratio(self) -> int:
        ratio = self.patch_size
        for block in self.encoder_blocks:
            if block.kind in _COMPRESS_STRIDES:
                ratio *= block.stride[1]
        return ratio

    @property
    def temporal_ratio(self) -> int:
        ratio = 1
        for block in self.encoder_blocks:
            if block.kind in _COMPRESS_STRIDES:
                ratio *= block.stride[0]
        return ratio


@dataclass(frozen=True, slots=True)
class LTXDiffusionVideoVAEConfig:
    """Exact LTX-2.5 causal encoder and neighborhood-attention decoder geometry."""

    encoder_blocks: tuple[LTXVAEBlock, ...]
    latent_channels: int = 128
    base_channels: int = 128
    patch_size: int = 4
    head_dim: int = 64
    stage_channels: tuple[int, ...] = (2048, 1024, 512, 512, 256)
    stage_depths: tuple[int, ...] = (4, 6, 4, 2, 8)
    stage_kernels: tuple[tuple[int, int, int], ...] = (
        (3, 7, 7),
        (3, 7, 7),
        (3, 5, 5),
        (3, 5, 5),
        (11, 11, 11),
    )
    upsamples: tuple[tuple[tuple[int, int, int], int], ...] = (
        ((1, 2, 2), 2),
        ((2, 1, 1), 2),
        ((2, 2, 2), 1),
        ((2, 2, 2), 2),
    )
    timestep_dim: int = 384
    output_channels: int = 3

    def __post_init__(self) -> None:
        if not self.encoder_blocks:
            raise ValueError("LTX diffusion VAE requires encoder blocks")
        if len(self.stage_channels) != 5 or len(self.stage_depths) != 5:
            raise ValueError("LTX diffusion VAE requires five decoder stages")
        if len(self.stage_kernels) != 5 or len(self.upsamples) != 4:
            raise ValueError("LTX diffusion VAE requires five kernels and four upsamplers")
        if any(value <= 0 for value in self.stage_channels + self.stage_depths):
            raise ValueError("LTX diffusion VAE stage dimensions must be positive")
        if any(channels % self.head_dim for channels in self.stage_channels):
            raise ValueError("LTX diffusion VAE stage widths must divide into attention heads")

    @property
    def encoder_config(self) -> LTXVideoVAEConfig:
        return LTXVideoVAEConfig(
            encoder_blocks=self.encoder_blocks,
            decoder_blocks=(LTXVAEBlock("res_x"),),
            latent_channels=self.latent_channels,
            base_channels=self.base_channels,
            patch_size=self.patch_size,
            encoder_spatial_padding_mode="zeros",
        )

    @property
    def spatial_ratio(self) -> int:
        return self.encoder_config.spatial_ratio

    @property
    def temporal_ratio(self) -> int:
        return self.encoder_config.temporal_ratio


def _vae_conv3d(key: str, out_channels: int, in_channels: int) -> dict[str, tuple[int, ...]]:
    return {
        key + ".weight": (out_channels, in_channels, 3, 3, 3),
        key + ".bias": (out_channels,),
    }


def _vae_time_embedder(key: str, dim: int) -> dict[str, tuple[int, ...]]:
    return {
        key + ".timestep_embedder.linear_1.weight": (dim, LTXV_TIME_PROJ_CHANNELS),
        key + ".timestep_embedder.linear_1.bias": (dim,),
        key + ".timestep_embedder.linear_2.weight": (dim, dim),
        key + ".timestep_embedder.linear_2.bias": (dim,),
    }


def _vae_res_block(
    key: str, in_channels: int, out_channels: int, *, inject_noise: bool, timestep: bool
) -> dict[str, tuple[int, ...]]:
    layout = _vae_conv3d(key + ".conv1.conv", out_channels, in_channels)
    layout.update(_vae_conv3d(key + ".conv2.conv", out_channels, out_channels))
    if in_channels != out_channels:
        layout[key + ".conv_shortcut.weight"] = (out_channels, in_channels, 1, 1, 1)
        layout[key + ".conv_shortcut.bias"] = (out_channels,)
        layout[key + ".norm3.norm.weight"] = (in_channels,)
        layout[key + ".norm3.norm.bias"] = (in_channels,)
    if inject_noise:
        layout[key + ".per_channel_scale1"] = (in_channels, 1, 1)
        layout[key + ".per_channel_scale2"] = (in_channels, 1, 1)
    if timestep:
        layout[key + ".scale_shift_table"] = (4, in_channels)
    return layout


def ltxv_vae_layout(config: LTXVideoVAEConfig) -> dict[str, tuple[int, ...]]:
    """Every state-dict key and shape of the LTX video VAE, without the
    checkpoint's ``vae.`` prefix. Checkpoints additionally carry
    unconsumed ``per_channel_statistics`` aggregates the reference never
    loads."""
    patched = 3 * config.patch_size**2
    layout = _vae_conv3d("encoder.conv_in.conv", config.base_channels, patched)
    for index, (block, (into, out)) in enumerate(
        zip(config.encoder_blocks, config.encoder_channels(), strict=True)
    ):
        key = f"encoder.down_blocks.{index}"
        if block.kind == "res_x":
            for layer in range(block.layers):
                layout.update(
                    _vae_res_block(
                        f"{key}.res_blocks.{layer}", out, out, inject_noise=False, timestep=False
                    )
                )
        elif block.kind == "res_x_y":
            layout.update(_vae_res_block(key, into, out, inject_noise=False, timestep=False))
        elif block.kind.endswith("_res"):
            reduced = out // (block.stride[0] * block.stride[1] * block.stride[2])
            layout.update(_vae_conv3d(key + ".conv.conv", reduced, into))
        else:
            layout.update(_vae_conv3d(key + ".conv", out, into))
    encoder_out = config.encoder_channels()[-1][1]
    layout.update(_vae_conv3d("encoder.conv_out.conv", config.latent_channels + 1, encoder_out))

    layout.update(
        _vae_conv3d("decoder.conv_in.conv", config.decoder_input_channels, config.latent_channels)
    )
    for index, (block, (into, out)) in enumerate(
        zip(tuple(reversed(config.decoder_blocks)), config.decoder_channels(), strict=True)
    ):
        key = f"decoder.up_blocks.{index}"
        if block.kind == "res_x":
            if config.timestep_conditioning:
                layout.update(_vae_time_embedder(key + ".time_embedder", into * 4))
            for layer in range(block.layers):
                layout.update(
                    _vae_res_block(
                        f"{key}.res_blocks.{layer}",
                        into,
                        into,
                        inject_noise=block.inject_noise,
                        timestep=config.timestep_conditioning,
                    )
                )
        elif block.kind == "res_x_y":
            layout.update(
                _vae_res_block(key, into, out, inject_noise=block.inject_noise, timestep=False)
            )
        else:
            stride = block.stride
            expanded = stride[0] * stride[1] * stride[2] * into // block.multiplier
            layout.update(_vae_conv3d(key + ".conv.conv", expanded, into))
    decoder_out = config.decoder_channels()[-1][1]
    layout.update(_vae_conv3d("decoder.conv_out.conv", patched, decoder_out))
    if config.timestep_conditioning:
        layout["decoder.timestep_scale_multiplier"] = ()
        layout.update(_vae_time_embedder("decoder.last_time_embedder", decoder_out * 2))
        layout["decoder.last_scale_shift_table"] = (2, decoder_out)

    layout["per_channel_statistics.std-of-means"] = (config.latent_channels,)
    layout["per_channel_statistics.mean-of-means"] = (config.latent_channels,)
    return layout


def _ltx_diffusion_attention_layout(
    key: str, channels: int, head_dim: int
) -> dict[str, tuple[int, ...]]:
    hidden = (channels * 4 + 15) // 16 * 16
    return {
        key + ".attn.qkv.weight": (channels * 3, channels),
        key + ".attn.qkv.bias": (channels * 3,),
        key + ".attn.proj.weight": (channels, channels),
        key + ".attn.proj.bias": (channels,),
        key + ".attn.q_norm.weight": (head_dim,),
        key + ".attn.k_norm.weight": (head_dim,),
        key + ".mlp.w_up.weight": (hidden, channels),
        key + ".mlp.w_gate.weight": (hidden, channels),
        key + ".mlp.w_down.weight": (channels, hidden),
        key + ".norm1.weight": (channels,),
        key + ".norm2.weight": (channels,),
    }


def ltx_diffusion_video_vae_layout(
    config: LTXDiffusionVideoVAEConfig,
) -> dict[str, tuple[int, ...]]:
    """Every tensor in the LTX-2.5 diffusion video VAE."""
    layout = {
        key: shape
        for key, shape in ltxv_vae_layout(config.encoder_config).items()
        if key.startswith("encoder.") or key.startswith("per_channel_statistics.")
    }
    first = config.stage_channels[0]
    last = config.stage_channels[-1]
    pixel_channels = config.output_channels * config.patch_size**2
    layout.update(
        {
            "decoder.conv_in.weight": (first, config.latent_channels),
            "decoder.conv_in.bias": (first,),
            "decoder.conv_in_x_t.weight": (last, pixel_channels),
            "decoder.conv_in_x_t.bias": (last,),
            "decoder.conv_out.weight": (pixel_channels, last),
            "decoder.conv_out.bias": (pixel_channels,),
            "decoder.norm_out.weight": (last,),
            "decoder.t_embedder.mlp.0.weight": (config.timestep_dim, 256),
            "decoder.t_embedder.mlp.0.bias": (config.timestep_dim,),
            "decoder.t_embedder.mlp.2.weight": (config.timestep_dim, config.timestep_dim),
            "decoder.t_embedder.mlp.2.bias": (config.timestep_dim,),
            "decoder.shared_adaln.proj.weight": (7 * last, config.timestep_dim),
            "decoder.shared_adaln.proj.bias": (7 * last,),
        }
    )
    for stage, (channels, depth) in enumerate(
        zip(config.stage_channels[:-1], config.stage_depths[:-1], strict=True)
    ):
        for block in range(depth):
            layout.update(
                _ltx_diffusion_attention_layout(
                    f"decoder.det_stages.{stage}.{block}", channels, config.head_dim
                )
            )
        stride, reduction = config.upsamples[stage]
        expanded = math.prod(stride) * channels // reduction
        layout[f"decoder.upsamples.{stage}.proj.weight"] = (expanded, channels)
        layout[f"decoder.upsamples.{stage}.proj.bias"] = (expanded,)
    for block in range(config.stage_depths[-1]):
        key = f"decoder.diff_blocks.{block}"
        layout.update(_ltx_diffusion_attention_layout(key, last, config.head_dim))
        layout[key + ".context_proj.weight"] = (last, last)
        layout[key + ".context_proj.bias"] = (last,)
        layout[key + ".scale_shift_table"] = (7, last)
    return layout


# Both 2B VAE geometries are transcribed from the config JSON embedded in
# the checkpoints' safetensors metadata, which the reference prefers over
# its built-in per-version defaults (comfy/sd.py VAE.__init__ @ b78cec87).
# The v0.9 checkpoint stores its shared block list in the reference's
# integer shorthand, where a bare res_x_y count means multiplier 2.
_LTXV_2B_V09_VAE_BLOCKS = (
    LTXVAEBlock("res_x", layers=4),
    LTXVAEBlock("compress_all"),
    LTXVAEBlock("res_x_y", multiplier=2),
    LTXVAEBlock("res_x", layers=3),
    LTXVAEBlock("compress_all"),
    LTXVAEBlock("res_x_y", multiplier=2),
    LTXVAEBlock("res_x", layers=3),
    LTXVAEBlock("compress_all"),
    LTXVAEBlock("res_x", layers=3),
    LTXVAEBlock("res_x", layers=4),
)

LTXV_2B_V09_VAE_CONFIG = LTXVideoVAEConfig(
    encoder_blocks=_LTXV_2B_V09_VAE_BLOCKS,
    decoder_blocks=_LTXV_2B_V09_VAE_BLOCKS,
)

# The v0.9.5 checkpoint's embedded config differs from the reference's
# built-in version-1 default: every decoder res group has five layers and
# noise injection stays off.
LTXV_2B_V095_VAE_CONFIG = LTXVideoVAEConfig(
    encoder_blocks=(
        LTXVAEBlock("res_x", layers=4),
        LTXVAEBlock("compress_space_res", multiplier=2),
        LTXVAEBlock("res_x", layers=6),
        LTXVAEBlock("compress_time_res", multiplier=2),
        LTXVAEBlock("res_x", layers=6),
        LTXVAEBlock("compress_all_res", multiplier=2),
        LTXVAEBlock("res_x", layers=2),
        LTXVAEBlock("compress_all_res", multiplier=2),
        LTXVAEBlock("res_x", layers=2),
    ),
    decoder_blocks=(
        LTXVAEBlock("res_x", layers=5),
        LTXVAEBlock("compress_all", multiplier=2, residual=True),
        LTXVAEBlock("res_x", layers=5),
        LTXVAEBlock("compress_all", multiplier=2, residual=True),
        LTXVAEBlock("res_x", layers=5),
        LTXVAEBlock("compress_all", multiplier=2, residual=True),
        LTXVAEBlock("res_x", layers=5),
    ),
    timestep_conditioning=True,
)

# The LTX-2 combined checkpoint ships the v0.9.5 block stack and declares
# timestep_conditioning false in its vae metadata (the reference builds the
# VAE from that metadata), so the video VAE loads with timestep conditioning
# off even though per-channel statistics tensors are present.
LTXAV_19B_VAE_CONFIG = LTXVideoVAEConfig(
    encoder_blocks=LTXV_2B_V095_VAE_CONFIG.encoder_blocks,
    decoder_blocks=LTXV_2B_V095_VAE_CONFIG.decoder_blocks,
)

LTXAV_22B_V23_VAE_CONFIG = LTXVideoVAEConfig(
    encoder_blocks=(
        LTXVAEBlock("res_x", layers=4),
        LTXVAEBlock("compress_space_res", multiplier=2),
        LTXVAEBlock("res_x", layers=6),
        LTXVAEBlock("compress_time_res", multiplier=2),
        LTXVAEBlock("res_x", layers=4),
        LTXVAEBlock("compress_all_res", multiplier=2),
        LTXVAEBlock("res_x", layers=2),
        LTXVAEBlock("compress_all_res"),
        LTXVAEBlock("res_x", layers=2),
    ),
    decoder_blocks=(
        LTXVAEBlock("res_x", layers=4),
        LTXVAEBlock("compress_space", multiplier=2),
        LTXVAEBlock("res_x", layers=6),
        LTXVAEBlock("compress_time", multiplier=2),
        LTXVAEBlock("res_x", layers=4),
        LTXVAEBlock("compress_all"),
        LTXVAEBlock("res_x", layers=2),
        LTXVAEBlock("compress_all", multiplier=2),
        LTXVAEBlock("res_x", layers=2),
    ),
    decoder_spatial_padding_mode="zeros",
)

LTXAV_22B_V25_VAE_CONFIG = LTXDiffusionVideoVAEConfig(
    encoder_blocks=(
        LTXVAEBlock("res_x", layers=4),
        LTXVAEBlock("compress_space_res", multiplier=2),
        LTXVAEBlock("res_x", layers=6),
        LTXVAEBlock("compress_time_res", multiplier=2),
        LTXVAEBlock("res_x", layers=4),
        LTXVAEBlock("compress_all_res", multiplier=2),
        LTXVAEBlock("res_x", layers=2),
        LTXVAEBlock("compress_all_res", multiplier=1),
        LTXVAEBlock("res_x", layers=2),
    )
)


# latent2rgb preview projection copied verbatim from
# comfy/latent_formats.py LTXV @ 82f839f5.
_LTXV_RGB_FACTORS = (
    (0.011202, -0.00063815, -0.010021),
    (0.086031, 0.065813, 0.00095409),
    (-0.012576, -0.0075734, -0.0040528),
    (0.0094063, -0.0021688, 0.0026093),
    (0.0037636, 0.012765, 0.0091548),
    (0.021024, -0.0052973, 0.0034373),
    (-0.0088896, -0.019703, -0.018761),
    (-0.01316, -0.010523, 0.0019709),
    (-0.0015152, -0.0069891, -0.007581),
    (-0.0017247, 0.0004656, -0.0033839),
    (0.013617, 0.0047077, -0.0020045),
    (0.010256, 0.0077318, 0.013948),
    (-0.016108, -0.0062151, 0.0011561),
    (0.0073407, 0.015628, 0.00044865),
    (0.00095357, -0.0029518, -0.01476),
    (0.019143, 0.010868, 0.012264),
    (0.0044575, 3.6682e-05, -0.0068508),
    (-0.00045681, 0.003257, 0.0077929),
    (0.033902, 0.033405, 0.037454),
    (-0.023001, -0.0024877, -0.0031033),
    (0.050265, 0.038841, 0.033539),
    (-0.0041018, -0.0011095, 0.0015859),
    (-0.12689, -0.13107, -0.21005),
    (0.026276, 0.014189, -0.0035963),
    (-0.0048679, 0.0088486, 0.0078029),
    (-0.001661, -0.0048597, -0.005206),
    (-0.002101, 0.002361, 0.0093796),
    (-0.022482, -0.021305, -0.015087),
    (-0.015753, -0.010646, -0.0065083),
    (-0.0046975, 0.0050288, -0.006739),
    (0.011951, 0.020712, 0.016191),
    (-0.0063704, -0.0084827, -0.0095483),
    (0.007261, -0.0099326, -0.022978),
    (-0.00091904, 0.0062882, 0.009572),
    (-0.037178, -0.037123, -0.056713),
    (-0.13373, -0.1072, -0.053801),
    (-0.0053702, 0.0081256, 0.0088397),
    (-0.15247, -0.21437, -0.21843),
    (0.031441, 0.0070335, -0.0097541),
    (0.0021528, -0.0089817, -0.021023),
    (0.0038461, -0.0058957, -0.015014),
    (-0.004347, -0.01294, -0.015972),
    (-0.0054781, -0.010842, -0.0030204),
    (-0.0065347, 0.0030806, -0.010163),
    (-0.0050414, -0.0071503, -0.00089686),
    (-0.0085851, -0.0024351, 0.0010674),
    (-0.0090016, -0.0096493, 0.0015692),
    (0.0050914, 0.012099, 0.019968),
    (0.013758, 0.011669, 0.0081958),
    (-0.010518, -0.011575, -0.0041307),
    (-0.02841, -0.031266, -0.022149),
    (0.0029336, 0.036511, 0.018717),
    (-0.016703, -0.016696, -0.0044529),
    (0.048818, 0.040063, 0.008741),
    (-0.015066, -0.00057328, 0.0029785),
    (-0.017613, -0.0081034, 0.013086),
    (-0.0092633, 0.010803, -0.0063489),
    (0.0030851, 0.0004775, 0.012347),
    (-0.022785, -0.023043, -0.026005),
    (-0.024787, -0.015389, -0.022104),
    (-0.023572, 0.0010544, 0.012361),
    (-0.0078915, -0.0012271, -0.0060968),
    (-0.011478, -0.0012543, 0.0062679),
    (-0.054229, 0.026644, 0.0063394),
    (0.0044216, -0.0073338, -0.010464),
    (-0.0045013, 0.0016082, 0.01442),
    (0.013673, 0.0088877, 0.0041253),
    (-0.010145, 0.0090072, 0.015695),
    (-0.0056234, 0.0011847, 0.0081261),
    (-0.0037171, -0.0053538, 0.001259),
    (0.029476, 0.021424, 0.030424),
    (-0.034925, -0.02434, -0.025316),
    (-0.034127, -0.022406, -0.010589),
    (-0.017342, -0.013249, -0.010719),
    (-0.0021478, -0.0086051, -0.0029878),
    (0.0012089, -0.0042391, -0.0068569),
    (0.00090411, -0.0066886, -6.7547e-05),
    (0.016048, -0.010057, -0.028929),
    (0.001229, 0.010163, 0.018861),
    (0.017264, 0.00027257, 0.013785),
    (-0.013482, -0.0036427, 0.00067481),
    (0.0046782, -0.0052423, 0.0024467),
    (-0.0059113, -0.0062244, -0.0018162),
    (0.015496, 0.014582, 0.0019514),
    (0.0074958, 0.0015886, -0.0082305),
    (0.019086, 0.001636, -0.0039674),
    (-0.0057021, -0.0027307, -0.0041066),
    (0.001745, 0.014602, 0.025794),
    (-0.00082788, 0.0022902, 0.0045161),
    (0.011632, 0.0089193, -0.0072813),
    (0.0075721, 0.0026784, 0.011393),
    (0.0051939, 0.0036903, 0.014049),
    (-0.018383, -0.022529, -0.024477),
    (0.00058842, -0.0057874, -0.01477),
    (-0.016125, -0.0086101, -0.014533),
    (0.02054, 0.020729, 0.0064338),
    (0.0033587, -0.011226, -0.016444),
    (-0.0014742, -0.010489, 0.0017097),
    (0.02813, 0.023546, 0.032791),
    (-0.018532, -0.012842, -0.0087756),
    (-0.0080533, -0.010771, -0.017536),
    (-0.0039009, 0.01615, 0.033359),
    (-0.0074554, -0.014154, -0.006191),
    (0.0034734, -0.01137, -0.010581),
    (0.011476, 0.0039281, 0.0028231),
    (0.0071639, -0.0014741, -0.0038066),
    (0.002225, -0.0087552, -0.0095719),
    (0.024146, 0.021696, 0.028056),
    (-0.0054365, -0.024291, -0.017802),
    (0.0074263, 0.01051, 0.012705),
    (0.0062669, 0.0062658, 0.019211),
    (0.016378, 0.0094933, 0.0066971),
    (0.017173, 0.023601, 0.023296),
    (-0.014568, -0.0098279, -0.011556),
    (0.014431, 0.01443, 0.0066362),
    (-0.006823, 0.018863, 0.014555),
    (0.0061156, 0.00347, -0.0026662),
    (-0.0026983, -0.0059402, -0.0092276),
    (0.010235, 0.0074173, -0.0076243),
    (-0.013255, 0.019322, -0.00092153),
    (0.0024222, -0.0048039, -0.015759),
    (0.026244, 0.025951, 0.020249),
    (0.015711, 0.018498, 0.0027407),
    (-0.0021714, 0.0047214, -0.022443),
    (-0.0074747, 0.0074166, 0.01443),
    (-0.0083906, -0.0079776, 0.0097927),
    (0.038321, 0.0096622, -0.019268),
    (-0.014605, -0.0067032, 0.0039675),
)
_LTXV_RGB_BIAS = (-0.0571, -0.1657, -0.2512)

# Video-stream preview projection for the packed AV latent, copied verbatim
# from comfy/latent_formats.py LTXAV @ 82f839f5.
_LTXAV_RGB_FACTORS = (
    (0.001135, -0.010555, -0.004925),
    (-0.008019, -0.006231, -0.005564),
    (0.012637, 0.005605, 0.012713),
    (0.023454, 0.020771, 0.017844),
    (-0.01194, -0.000932, 0.009292),
    (0.018602, 0.011018, 0.013969),
    (-0.036369, -0.046631, -0.057898),
    (-0.031919, 0.000131, 0.015214),
    (0.014519, 0.021041, 0.015325),
    (0.018889, 0.016149, -0.002836),
    (-0.003784, -0.006057, -0.008195),
    (0.013262, 0.030259, 0.029775),
    (0.050465, 0.050366, 0.025255),
    (0.018628, 0.007691, 0.002893),
    (-0.015698, -0.008451, -0.000676),
    (-0.0136, -0.012587, -0.004437),
    (0.012482, 0.021469, 0.027913),
    (-0.018241, -0.013488, -0.010975),
    (0.013828, 0.012568, 0.021984),
    (0.017911, 0.006552, 0.005567),
    (0.026769, 0.006803, -0.00936),
    (-0.006794, -0.008447, -0.013921),
    (0.029708, 0.018671, 0.022811),
    (-0.014732, -0.019169, 0.000903),
    (0.019607, 0.032595, 0.053409),
    (-0.003721, 0.003976, 0.010364),
    (-0.020193, -0.026076, -0.036068),
    (-0.002328, 0.006527, 0.013052),
    (0.017171, 0.009224, 0.006548),
    (0.001104, -0.000591, 0.000147),
    (-0.000217, 0.011834, 0.017945),
    (-0.015329, -0.012463, -0.006178),
    (-0.009478, -0.00868, -0.004107),
    (-0.005565, -0.006006, -0.001493),
    (0.009451, 0.008794, 0.013207),
    (-0.009989, -0.008027, -0.009568),
    (-0.001505, -0.008805, -0.006828),
    (0.001105, 0.008999, 0.009079),
    (0.025935, 0.016426, 0.008036),
    (0.006313, 0.000694, -0.006039),
    (-0.001893, -0.006951, -0.00956),
    (-0.007082, -0.002566, -0.007152),
    (-0.005231, 0.004829, 0.00822),
    (-0.004333, 0.001251, -0.004852),
    (-0.017024, -0.01273, -0.007457),
    (0.024988, 0.032963, 0.036556),
    (0.013697, 0.012278, 0.009979),
    (-0.013751, -0.008369, -0.015446),
    (-0.009348, -0.001047, 0.007622),
    (-0.003135, -0.00335, -0.003766),
    (0.007436, 0.004957, 0.01048),
    (0.018315, 0.022066, 0.021104),
    (-0.005621, -0.00677, -0.008219),
    (-0.007427, 0.001911, -0.001231),
    (-0.007413, 0.000486, -0.006039),
    (-0.014698, -0.00716, 0.006509),
    (0.013775, 0.014185, 0.008203),
    (0.060246, 0.069787, 0.072833),
    (0.009861, 0.00487, 0.001194),
    (-0.00366, 0.003251, 0.008015),
    (0.003696, -0.00368, -0.008851),
    (0.014924, 0.006196, 0.005282),
    (-0.00674, -0.004319, -0.006729),
    (0.020635, 0.015163, 0.012385),
    (-0.032623, -0.006105, 0.010436),
    (-0.058988, -0.030162, -0.037961),
    (-0.035614, -0.021929, -0.011062),
    (-0.023412, -0.011305, -0.005054),
    (-0.002716, -0.005184, -0.004084),
    (0.014591, 0.015294, 0.014045),
    (0.00831, 0.002466, -0.003225),
    (0.005176, 0.001119, 0.000695),
    (-0.021569, -0.030886, -0.044732),
    (0.007517, 0.003891, 0.000551),
    (-0.006793, 0.004059, 0.010184),
    (-0.086481, -0.082033, -0.083414),
    (0.004192, 0.000762, -0.008658),
    (0.01097, 0.009002, 0.007384),
    (0.004042, -0.006732, -0.011031),
    (0.012164, 0.006401, 0.007483),
    (0.029252, 0.01399, 0.011128),
    (0.048452, 0.034648, 0.016269),
    (0.024104, 0.012647, 0.011754),
    (-0.013216, -0.020192, -0.019752),
    (-0.010799, -0.008535, -0.005467),
    (0.005823, 0.001403, 0.00189),
    (0.052393, 0.044771, 0.032777),
    (0.007576, -0.00808, -0.012453),
    (0.00983, 0.004244, 0.001213),
    (-0.025867, -0.013169, -0.010636),
    (0.008494, 0.003135, 0.00079),
    (0.003969, -0.002625, -0.010204),
    (0.006509, 0.008272, 0.020819),
    (-0.004943, -0.013424, -0.015351),
    (0.005541, 0.009136, -0.003666),
    (-0.0143, -0.015864, -0.016853),
    (0.00265, 0.028393, 0.014125),
    (-0.027661, -0.045422, -0.064995),
    (0.00922, 0.015522, 0.010574),
    (-0.002236, 0.002915, 0.004557),
    (-0.020269, -0.008212, -0.000532),
    (0.019294, 0.003655, -0.002809),
    (0.007116, -0.002784, 1.7e-05),
    (0.057277, 0.07327, 0.074401),
    (-0.002616, -0.001696, -0.000498),
    (0.007248, 0.009793, 0.022829),
    (-0.00259, -0.005601, -0.000436),
    (-0.007681, 0.003893, -0.004119),
    (-0.057392, -0.045545, -0.02529),
    (0.045188, 0.047985, 0.054059),
    (0.000937, -0.008861, -0.038406),
    (-0.010192, -0.008036, -0.005385),
    (-0.030222, -0.027498, -0.030765),
    (-0.008359, 0.013247, 0.010918),
    (0.004102, 0.002093, 0.006934),
    (0.039461, 0.027339, 0.008284),
    (-0.075747, -0.07634, -0.071625),
    (0.002692, 0.005096, -0.002247),
    (-0.002453, -0.002785, -0.010483),
    (0.012265, 0.005481, 0.001729),
    (0.017755, 0.008655, 0.003532),
    (0.05556, 0.049128, 0.044137),
    (-0.025861, -0.023798, -0.018815),
    (-0.014876, -0.01077, -0.010713),
    (-0.017315, -0.012599, -0.008661),
    (-0.008461, -0.00621, -0.007744),
    (-0.040175, -0.042255, -0.048119),
    (-0.019355, -0.021055, -0.021919),
)
_LTXAV_RGB_BIAS = (-0.347892, -0.363814, -0.370287)

# The video VAE is temporally causal: t latent frames decode to t * 8 - 7
# content frames (comfy/sd.py lightricks upscale_ratio @ 82f839f5). No
# content_fps: the family conditions on a user-chosen frame rate instead of
# fixing one.
LTXV_LATENT = LatentDescriptor(
    channels=128,
    dimensions=3,
    spatial_downscale=32,
    temporal_downscale=8,
    temporal_causal=True,
    rgb_factors=_LTXV_RGB_FACTORS,
    rgb_bias=_LTXV_RGB_BIAS,
)
LTXAV_LATENT = MultiStreamLatentDescriptor(
    (
        (
            "video",
            LatentDescriptor(
                channels=128,
                dimensions=3,
                spatial_downscale=32,
                temporal_downscale=8,
                temporal_causal=True,
                rgb_factors=_LTXAV_RGB_FACTORS,
                rgb_bias=_LTXAV_RGB_BIAS,
            ),
        ),
        # Audio latents are (batch, z_channels=8, latent_frames, frequency_bins):
        # comfy_extras/nodes_lt_audio.py LTXVEmptyLatentAudio @ 82f839f5, with
        # z_channels 8 from the released audio VAE configuration.
        ("audio", LatentDescriptor(channels=8, dimensions=2)),
    )
)

# ModelSamplingFlux builds 10000 values from t=1/10000 through t=1 and applies
# flux_time_shift(2.37, 1, t), so the first shifted value is sigma_min.
LTX_SIGMAS = FluxFlowSigmas(shift=2.37, timesteps=10000)
LTX_SAMPLING = SamplingDescriptor(
    parameterization=Parameterization.FLOW,
    sigma_min=LTX_SIGMAS.sigma_min,
    sigma_max=LTX_SIGMAS.sigma_max,
    shift=2.37,
)

# The LTX video VAE streams its own memory-bounded chunks internally, so no
# overlap/feather tile fallback is wired.
LTXV_CODEC = CodecDescriptor(
    id="dinkster.ltxv_vae",
    display_name="LTX causal video VAE",
    kind="video",
    latent=LTXV_LATENT,
    supported_dtypes=frozenset({BFLOAT16, FLOAT32}),
    content_channels=3,
    supports_tiling=False,
)

LTXAV_VIDEO_CODEC = CodecDescriptor(
    id="dinkster.ltxav_vae",
    display_name="LTX-2 causal video VAE",
    kind="video",
    latent=dict(LTXAV_LATENT.streams)["video"],
    supported_dtypes=frozenset({BFLOAT16, FLOAT32}),
    content_channels=3,
    tiling=CodecTiling(
        decode_tile=(999, 32, 32),
        decode_overlap=(1, 8, 8),
        encode_tile=(65, 1024, 1024),
        encode_overlap=(9, 128, 128),
    ),
)


@dataclass(frozen=True, slots=True)
class LTXLatentUpsamplerConfig:
    """Architecture of the official LTX-2.x spatial latent upscaler."""

    in_channels: int = 128
    mid_channels: int = 1024
    num_blocks_per_stage: int = 4
    dims: int = 3
    spatial_upsample: bool = True
    temporal_upsample: bool = False
    spatial_scale: float = 2.0
    rational_resampler: bool = False


LTX_LATENT_UPSAMPLER_CONFIG = LTXLatentUpsamplerConfig()


def ltx_latent_upsampler_layout(
    config: LTXLatentUpsamplerConfig = LTX_LATENT_UPSAMPLER_CONFIG,
) -> dict[str, tuple[int, ...]]:
    """Exact state-dict geometry of the supported spatial upscaler."""

    channels = config.mid_channels
    layout: dict[str, tuple[int, ...]] = {
        "initial_conv.weight": (channels, config.in_channels, 3, 3, 3),
        "initial_conv.bias": (channels,),
        "initial_norm.weight": (channels,),
        "initial_norm.bias": (channels,),
        "upsampler.0.weight": (4 * channels, channels, 3, 3),
        "upsampler.0.bias": (4 * channels,),
        "final_conv.weight": (config.in_channels, channels, 3, 3, 3),
        "final_conv.bias": (config.in_channels,),
    }
    for stage in ("res_blocks", "post_upsample_res_blocks"):
        for index in range(config.num_blocks_per_stage):
            prefix = f"{stage}.{index}"
            for convolution in ("conv1", "conv2"):
                layout[f"{prefix}.{convolution}.weight"] = (channels, channels, 3, 3, 3)
                layout[f"{prefix}.{convolution}.bias"] = (channels,)
            for normalization in ("norm1", "norm2"):
                layout[f"{prefix}.{normalization}.weight"] = (channels,)
                layout[f"{prefix}.{normalization}.bias"] = (channels,)
    return layout


#: Resolution-dependent flow shift line (comfy_extras/nodes_lt.py
#: ModelSamplingLTXV and LTXVScheduler @ b78cec87): a linear ramp in the
#: latent token count from 0.95 at 1024 tokens to 2.05 at 4096 tokens,
#: unclamped outside that range.
LTXV_SHIFT_BASE = 0.95
LTXV_SHIFT_MAX = 2.05
LTXV_SHIFT_TOKENS_LOW = 1024
LTXV_SHIFT_TOKENS_HIGH = 4096


def ltxv_dynamic_shift(
    tokens: int,
    *,
    max_shift: float = LTXV_SHIFT_MAX,
    base_shift: float = LTXV_SHIFT_BASE,
) -> float:
    """The reference's dynamic flow shift for one latent geometry.

    ``tokens`` counts latent cells (frames * height * width of the latent).
    The result is the exponent :func:`flux_time_shift` consumes, the same
    convention as ``FluxFlowSigmas.shift``, so it plugs directly into a
    ``sampling_shift`` override.
    """
    if type(tokens) is not int or tokens < 1:
        raise ValueError("token count must be a positive integer")
    for name, value in (("max_shift", max_shift), ("base_shift", base_shift)):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be a finite nonnegative number")
    slope = (max_shift - base_shift) / (LTXV_SHIFT_TOKENS_HIGH - LTXV_SHIFT_TOKENS_LOW)
    return tokens * slope + (base_shift - slope * LTXV_SHIFT_TOKENS_LOW)


__all__ = [
    "LTXAVConfig",
    "LTXAVDetector",
    "LTXDurationHeadConfig",
    "LTXGeneratedKeyframes",
    "LTXAV_19B_CONFIG",
    "LTXAV_19B_VAE_CONFIG",
    "LTXAV_22B_V23_CONFIG",
    "LTXAV_22B_V25_CONFIG",
    "LTXAV_22B_V23_VAE_CONFIG",
    "LTXAV_22B_V25_VAE_CONFIG",
    "LTXAV_DURATION_HEAD_CONFIG",
    "LTXAV_AUDIO_CHANNELS",
    "LTXAV_AUDIO_FREQUENCY_BINS",
    "LTXAV_AUDIO_HOP_LENGTH",
    "LTXAV_AUDIO_LATENT_DOWNSAMPLE",
    "LTXAV_AUDIO_MAX_POS",
    "LTXAV_AUDIO_SAMPLE_RATE",
    "LTXAV_CONNECTOR_PREFIXES",
    "LTXAV_LATENT",
    "LTXAV_VIDEO_CODEC",
    "LTXLatentUpsamplerConfig",
    "LTX_LATENT_UPSAMPLER_CONFIG",
    "LTXVAEBlock",
    "LTXVAEBlockKind",
    "LTXVConfig",
    "LTXVDetector",
    "LTXDiffusionVideoVAEConfig",
    "LTXVideoVAEConfig",
    "LTXV_2B_V095_CONFIG",
    "LTXV_2B_V095_VAE_CONFIG",
    "LTXV_2B_V09_CONFIG",
    "LTXV_2B_V09_VAE_CONFIG",
    "LTXV_CODEC",
    "LTXV_LATENT",
    "LTXV_MAX_POS",
    "LTXV_SHIFT_BASE",
    "LTXV_SHIFT_MAX",
    "LTXV_SHIFT_TOKENS_HIGH",
    "LTXV_SHIFT_TOKENS_LOW",
    "LTXV_THETA",
    "LTXV_TIMESTEP_MULTIPLIER",
    "LTXV_TIME_PROJ_CHANNELS",
    "LTXV_VAE_DECODE_NOISE_SCALE",
    "LTXV_VAE_DECODE_TIMESTEP",
    "LTXV_VAE_SCALE_FACTORS",
    "LTX_SAMPLING",
    "LTX_SIGMAS",
    "detect_ltxav",
    "detect_ltxv",
    "ltxav_layout",
    "ltxav_duration_head_layout",
    "ltx_diffusion_video_vae_layout",
    "ltx_latent_upsampler_layout",
    "ltxv_dynamic_shift",
    "ltxv_layout",
    "ltxv_vae_layout",
]
