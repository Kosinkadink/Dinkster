"""Torch-free MiniMax H3 profile, paired registration, and task contracts.

The paired family registration is not a wired runtime or capability claim.
The detector follows ComfyUI's MiniMax H3 header geometry while requiring the
exact 5376-wide profile. Linear output dimensions carry architecture facts;
unavoidable input-dimension and table-width checks are accepted only for
truthful full-precision geometry, never ambiguous packed geometry.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from types import MappingProxyType
from typing import Generic, TypeAlias, TypeVar, cast

from .conditioning import PayloadDescriptor, PayloadReference
from .families import EvidenceValue
from .latent_masks import LatentMaskMapping
from .patches import SizedTensor
from .spaces import FlowSigmas
from .timeline_guides import CodecTemporalMapping
from .weights import TensorGeometry, WeightSource

_PREFIXES = ("model.diffusion_model.", "")
_SHAPES: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {
        "video_patch_proj.weight": (5376, 96),
        "audio_patch_proj.weight": (5376, 32),
        "condition_proj.weight": (5376, 5120),
        "blocks.0.attn.q_norm.weight": (128,),
        "blocks.0.attn.qkv_proj.weight": (21504, 5376),
        "blocks.0.mlp.fc1.weight": (28672, 5376),
        "final_layer.video_out.weight": (96, 5376),
        "final_layer.audio_out.weight": (32, 5376),
        "rope.inv_freq": (16,),
    }
)
_TIME_EMBEDDING_SHAPES: tuple[Mapping[str, tuple[int, ...]], ...] = (
    MappingProxyType({"adaln_t_table": (1025, 8)}),
    MappingProxyType(
        {
            "time_embedder.proj_in.weight": (5376, 256),
            "time_embedder.proj_out.weight": (2688, 5376),
        }
    ),
)
_FULL_SHAPE_KEYS = frozenset(
    {
        "video_patch_proj.weight",
        "audio_patch_proj.weight",
        "condition_proj.weight",
        "blocks.0.attn.qkv_proj.weight",
        "blocks.0.mlp.fc1.weight",
        "final_layer.video_out.weight",
        "final_layer.audio_out.weight",
        "adaln_t_table",
        "time_embedder.proj_in.weight",
        "time_embedder.proj_out.weight",
    }
)
_BLOCK_INDICES = frozenset(range(50))
_BLOCK_INDEX_TEXT = frozenset(str(index) for index in _BLOCK_INDICES)


@dataclass(frozen=True)
class MiniMaxH3Config:
    """The exact, non-runnable MiniMax H3 architecture requirements."""

    family_id: str = "dinkster.minimax_h3"
    video_latent_channels: int = 24
    audio_latent_channels: int = 32
    depth: int = 50
    hidden_width: int = 5376
    attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_width: int = 14336
    text_width: int = 5120
    patch: tuple[int, int, int] = (1, 2, 2)
    video_spatial_downscale: int = 16
    video_fps: int = 24
    audio_content_channels: int = 2
    audio_sample_rate_hz: int = 32_000
    audio_latent_rate_hz: int = 40
    batch_size: int = 1
    video_schedule_shift: float = 12.0
    audio_schedule_shift: float = 3.0
    conditioner_id: str = "Qwen3-VL-32B"
    conditioner_layer: int = 50
    video_codec_id: str = "MiniMaxH3VideoVAE"
    audio_codec_id: str = "MiniMaxH3AudioVAE"

    def __post_init__(self) -> None:
        actual = (
            self.family_id,
            self.video_latent_channels,
            self.audio_latent_channels,
            self.depth,
            self.hidden_width,
            self.attention_heads,
            self.attention_head_dim,
            self.ffn_width,
            self.text_width,
            self.patch,
            self.video_spatial_downscale,
            self.video_fps,
            self.audio_content_channels,
            self.audio_sample_rate_hz,
            self.audio_latent_rate_hz,
            self.batch_size,
            self.video_schedule_shift,
            self.audio_schedule_shift,
            self.conditioner_id,
            self.conditioner_layer,
            self.video_codec_id,
            self.audio_codec_id,
        )
        expected = (
            "dinkster.minimax_h3",
            24,
            32,
            50,
            5376,
            56,
            128,
            14336,
            5120,
            (1, 2, 2),
            16,
            24,
            2,
            32_000,
            40,
            1,
            12.0,
            3.0,
            "Qwen3-VL-32B",
            50,
            "MiniMaxH3VideoVAE",
            "MiniMaxH3AudioVAE",
        )
        if any(
            type(value) is not type(required) or value != required
            for value, required in zip(actual, expected, strict=True)
        ) or any(type(value) is not int for value in self.patch):
            raise ValueError("MiniMaxH3Config only represents the exact staged profile")


MINIMAX_H3_CONFIG = MiniMaxH3Config()


T = TypeVar("T", bound=SizedTensor)


@dataclass(frozen=True)
class MiniMaxH3Sigmas:
    """Paired H3 sigma math driven solely by the video flow schedule.

    At video sigma zero, ``audio_state_factor`` returns the continuous
    mathematical limit ``audio_shift / video.shift``. A later executable
    denoiser must terminate before evaluating a model at that endpoint.
    """

    video: FlowSigmas
    audio_shift: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.video.shift) or self.video.shift <= 0.0:
            raise ValueError("video shift must be finite and positive")
        if not math.isfinite(self.audio_shift) or self.audio_shift <= 0.0:
            raise ValueError("audio shift must be finite and positive")

    @property
    def sigma_min(self) -> float:
        return self.video.sigma_min

    @property
    def sigma_max(self) -> float:
        return self.video.sigma_max

    @property
    def table(self) -> tuple[float, ...] | None:
        return self.video.table

    def sigma(self, timestep: float) -> float:
        return self.video.sigma(timestep)

    def timestep(self, sigma: float) -> float:
        return self.video.timestep(sigma)

    def percent_to_sigma(self, percent: float) -> float:
        return self.video.percent_to_sigma(percent)

    @property
    def audio_scale(self) -> float:
        return self.video.shift / self.audio_shift

    def _checked_video_sigma(self, video_sigma: float) -> float:
        if not math.isfinite(video_sigma) or not 0.0 <= video_sigma <= 1.0:
            raise ValueError("video sigma must be finite and within [0, 1]")
        return video_sigma

    def audio_sigma(self, video_sigma: float) -> float:
        sigma = self._checked_video_sigma(video_sigma)
        return sigma / (self.audio_scale + (1.0 - self.audio_scale) * sigma)

    def audio_state_factor(self, video_sigma: float) -> float:
        sigma = self._checked_video_sigma(video_sigma)
        if sigma == 0.0:
            return self.audio_shift / self.video.shift
        return self.audio_sigma(sigma) / sigma

    def audio_velocity_factors(self, video_sigma: float) -> tuple[float, float]:
        sigma_audio = self.audio_sigma(video_sigma)
        return (
            1.0 - self.audio_scale,
            1.0 + (self.audio_scale - 1.0) * sigma_audio,
        )


MINIMAX_H3_SIGMAS = MiniMaxH3Sigmas(FlowSigmas(shift=12.0), audio_shift=3.0)
MINIMAX_H3_VIDEO_TEMPORAL_MAPPING = CodecTemporalMapping((1, 4, 4, 4, 4), 5, 3)
MINIMAX_H3_VIDEO_MASK_MAPPING = LatentMaskMapping(
    "video",
    MINIMAX_H3_VIDEO_TEMPORAL_MAPPING,
    MINIMAX_H3_CONFIG.video_fps,
    MINIMAX_H3_CONFIG.video_spatial_downscale,
)
MINIMAX_H3_AUDIO_MASK_MAPPING = LatentMaskMapping(
    "audio",
    CodecTemporalMapping((1,)),
    MINIMAX_H3_CONFIG.audio_latent_rate_hz,
)


@dataclass(frozen=True, slots=True)
class MiniMaxH3FamilyRegistration:
    """Exact paired-runtime family facts that cannot fit a singular latent."""

    id: str = field(default="dinkster.minimax_h3", init=False)
    aliases: tuple[str, ...] = field(default=(), init=False)
    display_name: str = field(default="MiniMax H3", init=False)
    config: MiniMaxH3Config = field(default=MINIMAX_H3_CONFIG, init=False)
    sigmas: MiniMaxH3Sigmas = field(default=MINIMAX_H3_SIGMAS, init=False)
    component_roles: tuple[str, ...] = field(
        default=(
            "fl2va-dit",
            "ref2va-dit",
            "qwen3vl-32b-conditioner",
            "video-vae",
            "audio-vae",
        ),
        init=False,
    )

    def detect(self, source: WeightSource) -> MiniMaxH3Evidence | None:
        """Detect the registered exact profile without reading payloads."""

        return detect_minimax_h3(source)


MINIMAX_H3_FAMILY = MiniMaxH3FamilyRegistration()


class MiniMaxH3Task(StrEnum):
    """The closed set of staged H3 conditioning tasks."""

    T2VA = "t2va"
    FL2VA = "fl2va"
    REF2VA = "ref2va"


class MiniMaxH3KeyframeRole(StrEnum):
    """A first- or last-frame input to FL2VA conditioning."""

    FIRST = "first"
    LAST = "last"


class MiniMaxH3DiTPayloadKind(StrEnum):
    """The closed reference payload kinds declared to a later DiT adapter."""

    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class MiniMaxH3PresentationKind(StrEnum):
    """Structural kinds in the raw Qwen presentation."""

    TEXT = "text"
    VISION_START = "vision_start"
    IMAGE_CONTENT = "image_content"
    VIDEO_CONTENT = "video_content"
    VISION_END = "vision_end"


class MiniMaxH3TokenTag(IntEnum):
    """MiniMax H3 adaLN modality tags."""

    VISION = 0
    TEXT = 1


def _validate_prompt(prompt: object) -> None:
    if type(prompt) is not str:
        raise TypeError("prompt must be a string")


def _validate_payload(payload: object) -> None:
    if type(payload) is not PayloadDescriptor:
        raise TypeError("payload must be a PayloadDescriptor")
    if type(payload.reference) is not PayloadReference:
        raise TypeError("payload reference must be a PayloadReference")
    if type(payload.reference.id) is not str:
        raise TypeError("payload reference id must be a string")
    if type(payload.shape) is not tuple or any(
        type(dimension) is not int or dimension <= 0 for dimension in payload.shape
    ):
        raise ValueError("MiniMax H3 payload dimensions must be positive")
    if type(payload.dtype) is not str or type(payload.space) is not str:
        raise TypeError("payload dtype and space must be strings")


@dataclass(frozen=True)
class MiniMaxH3Keyframe:
    """One typed FL2VA keyframe payload."""

    role: MiniMaxH3KeyframeRole
    payload: PayloadDescriptor

    def __post_init__(self) -> None:
        if type(self.role) is not MiniMaxH3KeyframeRole:
            raise TypeError("keyframe role must be MiniMaxH3KeyframeRole")
        _validate_payload(self.payload)


@dataclass(frozen=True)
class MiniMaxH3ImageReference:
    """One REF2VA image reference."""

    payload: PayloadDescriptor

    def __post_init__(self) -> None:
        _validate_payload(self.payload)


@dataclass(frozen=True)
class MiniMaxH3AudioReference:
    """One standalone REF2VA audio reference."""

    payload: PayloadDescriptor
    sample_rate: int

    def __post_init__(self) -> None:
        _validate_payload(self.payload)
        if type(self.sample_rate) is not int or self.sample_rate <= 0:
            raise ValueError("audio sample rate must be a positive integer")


@dataclass(frozen=True)
class MiniMaxH3VideoReference:
    """One REF2VA video and its optional soundtrack."""

    frames: tuple[PayloadDescriptor, ...]
    presentation_indices: tuple[int, ...]
    timestamps: tuple[float, ...]
    audio: MiniMaxH3AudioReference | None = None

    def __post_init__(self) -> None:
        if type(self.frames) is not tuple or not self.frames:
            raise TypeError("video frames must be a non-empty tuple")
        for frame in self.frames:
            _validate_payload(frame)
        if type(self.presentation_indices) is not tuple or not self.presentation_indices:
            raise TypeError("video presentation indices must be a non-empty tuple")
        if any(type(index) is not int for index in self.presentation_indices):
            raise TypeError("video presentation indices must be exact integers")
        presentation_stride = MINIMAX_H3_CONFIG.video_fps // 2
        expected_indices = tuple(range(0, len(self.frames), presentation_stride))
        if self.presentation_indices != expected_indices:
            raise ValueError("video presentation indices must select the exact 2-fps frames")
        if type(self.timestamps) is not tuple:
            raise TypeError("video timestamps must be a tuple")
        if len(self.timestamps) != len(self.presentation_indices):
            raise ValueError("video timestamps must match the presentation frame count")
        if any(
            type(timestamp) is not float or not math.isfinite(timestamp) or timestamp < 0.0
            for timestamp in self.timestamps
        ):
            raise ValueError("video timestamps must be finite nonnegative floats")
        if any(
            following <= preceding
            for preceding, following in zip(self.timestamps, self.timestamps[1:], strict=False)
        ):
            raise ValueError("video timestamps must be strictly increasing")
        expected_timestamps = tuple(
            index / MINIMAX_H3_CONFIG.video_fps for index in expected_indices
        )
        if self.timestamps != expected_timestamps:
            raise ValueError("video timestamps must match the exact 24-fps frame times")
        if self.audio is not None:
            if type(self.audio) is not MiniMaxH3AudioReference:
                raise TypeError("video audio must be a MiniMaxH3AudioReference")


MiniMaxH3Reference: TypeAlias = (
    MiniMaxH3ImageReference | MiniMaxH3AudioReference | MiniMaxH3VideoReference
)


@dataclass(frozen=True)
class MiniMaxH3T2VARequest:
    """Prompt-only T2VA conditioning request."""

    prompt: str
    task: MiniMaxH3Task = field(default=MiniMaxH3Task.T2VA, init=False)

    def __post_init__(self) -> None:
        _validate_prompt(self.prompt)


@dataclass(frozen=True)
class MiniMaxH3FL2VARequest:
    """FL2VA request with one or both unique endpoint keyframes."""

    prompt: str
    keyframes: tuple[MiniMaxH3Keyframe, ...]
    task: MiniMaxH3Task = field(default=MiniMaxH3Task.FL2VA, init=False)

    def __post_init__(self) -> None:
        _validate_prompt(self.prompt)
        if type(self.keyframes) is not tuple:
            raise TypeError("FL2VA keyframes must be a tuple")
        if not 1 <= len(self.keyframes) <= 2:
            raise ValueError("FL2VA requires one or two keyframes")
        if any(type(keyframe) is not MiniMaxH3Keyframe for keyframe in self.keyframes):
            raise TypeError("FL2VA keyframes must be MiniMaxH3Keyframe values")
        roles = tuple(keyframe.role for keyframe in self.keyframes)
        if len(frozenset(roles)) != len(roles):
            raise ValueError("FL2VA keyframe roles must be unique")


@dataclass(frozen=True)
class MiniMaxH3REF2VARequest:
    """REF2VA request whose typed references retain request order."""

    prompt: str
    references: tuple[MiniMaxH3Reference, ...]
    task: MiniMaxH3Task = field(default=MiniMaxH3Task.REF2VA, init=False)

    def __post_init__(self) -> None:
        _validate_prompt(self.prompt)
        if type(self.references) is not tuple:
            raise TypeError("REF2VA references must be a tuple")
        if not self.references:
            raise ValueError("REF2VA requires at least one reference")
        allowed = (
            MiniMaxH3ImageReference,
            MiniMaxH3AudioReference,
            MiniMaxH3VideoReference,
        )
        if any(type(reference) not in allowed for reference in self.references):
            raise TypeError("REF2VA references must be typed MiniMax H3 references")


MiniMaxH3ConditioningRequest: TypeAlias = (
    MiniMaxH3T2VARequest | MiniMaxH3FL2VARequest | MiniMaxH3REF2VARequest
)


@dataclass(frozen=True, slots=True)
class MiniMaxH3AudioContent(Generic[T]):
    """One stereo waveform and its explicit sample rate."""

    waveform: T
    sample_rate: int

    def __post_init__(self) -> None:
        if type(self.sample_rate) is not int or self.sample_rate <= 0:
            raise ValueError("audio sample rate must be a positive integer")


@dataclass(frozen=True)
class MiniMaxH3PresentationSegment:
    """One raw text or expanded vision segment and its modality tag."""

    kind: MiniMaxH3PresentationKind
    text: str | None = None
    payloads: tuple[PayloadDescriptor, ...] = ()
    token_tag: MiniMaxH3TokenTag = field(init=False)

    def __post_init__(self) -> None:
        if type(self.kind) is not MiniMaxH3PresentationKind:
            raise TypeError("presentation kind must be MiniMaxH3PresentationKind")
        if type(self.payloads) is not tuple:
            raise TypeError("presentation payloads must be a tuple")
        marker = {
            MiniMaxH3PresentationKind.VISION_START: "<|vision_start|>",
            MiniMaxH3PresentationKind.VISION_END: "<|vision_end|>",
        }.get(self.kind)
        if self.kind is MiniMaxH3PresentationKind.TEXT:
            if type(self.text) is not str or self.payloads:
                raise ValueError("text segments require only raw text")
            tag = MiniMaxH3TokenTag.TEXT
        elif marker is not None:
            if type(self.text) is not str or self.text != marker or self.payloads:
                raise ValueError("vision marker segments require their exact marker")
            tag = MiniMaxH3TokenTag.VISION
        else:
            expected_count = 1 if self.kind is MiniMaxH3PresentationKind.IMAGE_CONTENT else 2
            if self.text is not None or len(self.payloads) != expected_count:
                raise ValueError("vision content has the wrong payload count")
            for payload in self.payloads:
                _validate_payload(payload)
            tag = MiniMaxH3TokenTag.VISION
        object.__setattr__(self, "token_tag", tag)


@dataclass(frozen=True)
class MiniMaxH3DiTKeyframePayload:
    """One ordered keyframe payload declaration for a later DiT adapter."""

    role: MiniMaxH3KeyframeRole
    payload: PayloadDescriptor

    def __post_init__(self) -> None:
        MiniMaxH3Keyframe(self.role, self.payload)


@dataclass(frozen=True)
class MiniMaxH3DiTReferencePayload:
    """One ordered reference payload declaration for a later DiT adapter."""

    kind: MiniMaxH3DiTPayloadKind
    ordinal: int
    payloads: tuple[PayloadDescriptor, ...]

    def __post_init__(self) -> None:
        if type(self.kind) is not MiniMaxH3DiTPayloadKind:
            raise TypeError("DiT payload kind must be MiniMaxH3DiTPayloadKind")
        if type(self.ordinal) is not int or self.ordinal <= 0:
            raise ValueError("DiT payload ordinal must be a positive integer")
        if type(self.payloads) is not tuple or not self.payloads:
            raise TypeError("DiT payloads must be a non-empty tuple")
        for payload in self.payloads:
            _validate_payload(payload)
        if self.kind is not MiniMaxH3DiTPayloadKind.VIDEO and len(self.payloads) != 1:
            raise ValueError("image and audio declarations require exactly one payload")


@dataclass(frozen=True, init=False)
class MiniMaxH3ConditionerPlan:
    """Deterministic, torch-free H3 conditioner input plan."""

    task: MiniMaxH3Task
    presentation: tuple[MiniMaxH3PresentationSegment, ...]
    dit_keyframes: tuple[MiniMaxH3DiTKeyframePayload, ...]
    dit_references: tuple[MiniMaxH3DiTReferencePayload, ...]
    conditioner_id: str = field(default="Qwen3-VL-32B", init=False)
    conditioner_layer: int = field(default=50, init=False)
    chat_templated: bool = field(default=False, init=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("conditioner plans are produced only by normalization")


def _conditioner_plan(
    task: MiniMaxH3Task,
    presentation: tuple[MiniMaxH3PresentationSegment, ...],
    dit_keyframes: tuple[MiniMaxH3DiTKeyframePayload, ...],
    dit_references: tuple[MiniMaxH3DiTReferencePayload, ...],
) -> MiniMaxH3ConditionerPlan:
    plan = object.__new__(MiniMaxH3ConditionerPlan)
    object.__setattr__(plan, "task", task)
    object.__setattr__(plan, "presentation", presentation)
    object.__setattr__(plan, "dit_keyframes", dit_keyframes)
    object.__setattr__(plan, "dit_references", dit_references)
    object.__setattr__(plan, "conditioner_id", "Qwen3-VL-32B")
    object.__setattr__(plan, "conditioner_layer", 50)
    object.__setattr__(plan, "chat_templated", False)
    return plan


def _text_segment(text: str) -> MiniMaxH3PresentationSegment:
    return MiniMaxH3PresentationSegment(MiniMaxH3PresentationKind.TEXT, text)


def _vision_segments(
    payloads: tuple[PayloadDescriptor, ...], *, video: bool = False
) -> tuple[MiniMaxH3PresentationSegment, ...]:
    content_kind = (
        MiniMaxH3PresentationKind.VIDEO_CONTENT
        if video
        else MiniMaxH3PresentationKind.IMAGE_CONTENT
    )
    return (
        MiniMaxH3PresentationSegment(MiniMaxH3PresentationKind.VISION_START, "<|vision_start|>"),
        MiniMaxH3PresentationSegment(content_kind, payloads=payloads),
        MiniMaxH3PresentationSegment(MiniMaxH3PresentationKind.VISION_END, "<|vision_end|>"),
    )


def normalize_minimax_h3_conditioning(
    request: MiniMaxH3ConditioningRequest,
) -> MiniMaxH3ConditionerPlan:
    """Normalize one exact H3 request without tokenizing or touching payloads."""

    allowed = (MiniMaxH3T2VARequest, MiniMaxH3FL2VARequest, MiniMaxH3REF2VARequest)
    if type(request) not in allowed:
        raise TypeError("request must be an exact MiniMax H3 conditioning request")
    presentation: list[MiniMaxH3PresentationSegment] = []
    keyframe_payloads: list[MiniMaxH3DiTKeyframePayload] = []
    reference_payloads: list[MiniMaxH3DiTReferencePayload] = []
    if type(request) is MiniMaxH3FL2VARequest:
        by_role = {keyframe.role: keyframe for keyframe in request.keyframes}
        picture_ordinal = 0
        for role in (MiniMaxH3KeyframeRole.FIRST, MiniMaxH3KeyframeRole.LAST):
            keyframe = by_role.get(role)
            if keyframe is None:
                continue
            picture_ordinal += 1
            presentation.append(_text_segment(f"<Picture {picture_ordinal}>: "))
            presentation.extend(_vision_segments((keyframe.payload,)))
            keyframe_payloads.append(MiniMaxH3DiTKeyframePayload(role, keyframe.payload))
    elif type(request) is MiniMaxH3REF2VARequest:
        counters = {
            MiniMaxH3DiTPayloadKind.IMAGE: 0,
            MiniMaxH3DiTPayloadKind.AUDIO: 0,
            MiniMaxH3DiTPayloadKind.VIDEO: 0,
        }

        def next_ordinal(kind: MiniMaxH3DiTPayloadKind) -> int:
            counters[kind] += 1
            return counters[kind]

        for reference in request.references:
            if type(reference) is MiniMaxH3ImageReference:
                ordinal = next_ordinal(MiniMaxH3DiTPayloadKind.IMAGE)
                presentation.append(_text_segment(f"<Picture {ordinal}>: "))
                presentation.extend(_vision_segments((reference.payload,)))
                reference_payloads.append(
                    MiniMaxH3DiTReferencePayload(
                        MiniMaxH3DiTPayloadKind.IMAGE, ordinal, (reference.payload,)
                    )
                )
            elif type(reference) is MiniMaxH3AudioReference:
                ordinal = next_ordinal(MiniMaxH3DiTPayloadKind.AUDIO)
                presentation.append(_text_segment(f"<Audio {ordinal}>: "))
                reference_payloads.append(
                    MiniMaxH3DiTReferencePayload(
                        MiniMaxH3DiTPayloadKind.AUDIO, ordinal, (reference.payload,)
                    )
                )
            else:
                video = cast(MiniMaxH3VideoReference, reference)
                if video.audio is not None:
                    audio_ordinal = next_ordinal(MiniMaxH3DiTPayloadKind.AUDIO)
                    presentation.append(_text_segment(f"<Audio {audio_ordinal}>: "))
                    reference_payloads.append(
                        MiniMaxH3DiTReferencePayload(
                            MiniMaxH3DiTPayloadKind.AUDIO,
                            audio_ordinal,
                            (video.audio.payload,),
                        )
                    )
                video_ordinal = next_ordinal(MiniMaxH3DiTPayloadKind.VIDEO)
                presentation.append(_text_segment(f"<Video {video_ordinal}>: "))
                frames = tuple(video.frames[index] for index in video.presentation_indices)
                timestamps = video.timestamps
                if len(frames) % 2:
                    frames += (frames[-1],)
                    timestamps += (timestamps[-1],)
                for index in range(0, len(frames), 2):
                    start = timestamps[index]
                    midpoint = start + (timestamps[index + 1] - start) / 2.0
                    presentation.append(_text_segment(f"<{midpoint:.1f} seconds>"))
                    presentation.extend(
                        _vision_segments((frames[index], frames[index + 1]), video=True)
                    )
                reference_payloads.append(
                    MiniMaxH3DiTReferencePayload(
                        MiniMaxH3DiTPayloadKind.VIDEO, video_ordinal, video.frames
                    )
                )
    presentation.append(_text_segment(request.prompt))
    return _conditioner_plan(
        request.task,
        tuple(presentation),
        tuple(keyframe_payloads),
        tuple(reference_payloads),
    )


@dataclass(frozen=True)
class MiniMaxH3Evidence:
    """Deterministic header evidence for the exact staged H3 profile."""

    config: MiniMaxH3Config
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


def _matches_geometry(
    suffix: str,
    geometry: TensorGeometry,
    *,
    allow_int8_geometry: bool,
) -> bool:
    expected = _SHAPES.get(suffix)
    if expected is None:
        expected = next(shapes[suffix] for shapes in _TIME_EMBEDDING_SHAPES if suffix in shapes)
    if len(geometry.shape) != len(expected) or geometry.shape[0] != expected[0]:
        return False
    if suffix not in _FULL_SHAPE_KEYS:
        return True
    if (geometry.dtype.kind != "float" or geometry.dtype.bits < 16) and not (
        allow_int8_geometry and geometry.dtype.name == "int8"
    ):
        return False
    return geometry.shape[1] == expected[1]


def _block_indices(keys: frozenset[str], prefix: str) -> frozenset[int] | None:
    root = prefix + "blocks."
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
        ):
            return None
        if text not in _BLOCK_INDEX_TEXT:
            return None
        indices.add(int(text))
    return frozenset(indices)


def detect_minimax_h3(
    source: WeightSource, *, allow_int8_geometry: bool = False
) -> MiniMaxH3Evidence | None:
    """Detect the exact H3 profile from geometry alone, or fail closed.

    INT8 matrix geometry is accepted only when the caller has independently
    established immutable official-artifact authority.
    """

    keys = frozenset(source.keys())
    for prefix in _PREFIXES:
        if _block_indices(keys, prefix) != _BLOCK_INDICES:
            continue
        matched: list[str] = []
        for suffix in _SHAPES:
            key = prefix + suffix
            geometry = _entry_geometry(source, keys, key)
            if geometry is None or not _matches_geometry(
                suffix,
                geometry,
                allow_int8_geometry=allow_int8_geometry,
            ):
                break
            matched.append(key)
        else:
            for time_shapes in _TIME_EMBEDDING_SHAPES:
                time_matched: list[str] = []
                for suffix in time_shapes:
                    key = prefix + suffix
                    geometry = _entry_geometry(source, keys, key)
                    if geometry is None or not _matches_geometry(
                        suffix,
                        geometry,
                        allow_int8_geometry=allow_int8_geometry,
                    ):
                        break
                    time_matched.append(key)
                else:
                    matched.extend(time_matched)
                    break
            else:
                continue
            config = MINIMAX_H3_CONFIG
            return MiniMaxH3Evidence(
                config=config,
                key_prefix=prefix,
                matched_keys=tuple(matched),
                fields={
                    "audio_latent_channels": config.audio_latent_channels,
                    "attention_head_dim": config.attention_head_dim,
                    "attention_heads": config.attention_heads,
                    "depth": config.depth,
                    "ffn_width": config.ffn_width,
                    "hidden_width": config.hidden_width,
                    "key_prefix": prefix,
                    "patch": "1x2x2",
                    "text_width": config.text_width,
                    "video_latent_channels": config.video_latent_channels,
                },
            )
    return None


__all__ = [
    "MINIMAX_H3_AUDIO_MASK_MAPPING",
    "MINIMAX_H3_SIGMAS",
    "MINIMAX_H3_CONFIG",
    "MINIMAX_H3_FAMILY",
    "MINIMAX_H3_VIDEO_MASK_MAPPING",
    "MINIMAX_H3_VIDEO_TEMPORAL_MAPPING",
    "MiniMaxH3AudioContent",
    "MiniMaxH3AudioReference",
    "MiniMaxH3Sigmas",
    "MiniMaxH3ConditionerPlan",
    "MiniMaxH3ConditioningRequest",
    "MiniMaxH3Config",
    "MiniMaxH3DiTKeyframePayload",
    "MiniMaxH3DiTPayloadKind",
    "MiniMaxH3DiTReferencePayload",
    "MiniMaxH3Evidence",
    "MiniMaxH3FL2VARequest",
    "MiniMaxH3FamilyRegistration",
    "MiniMaxH3ImageReference",
    "MiniMaxH3Keyframe",
    "MiniMaxH3KeyframeRole",
    "MiniMaxH3PresentationKind",
    "MiniMaxH3PresentationSegment",
    "MiniMaxH3REF2VARequest",
    "MiniMaxH3Reference",
    "MiniMaxH3T2VARequest",
    "MiniMaxH3Task",
    "MiniMaxH3TokenTag",
    "MiniMaxH3VideoReference",
    "detect_minimax_h3",
    "normalize_minimax_h3_conditioning",
]
