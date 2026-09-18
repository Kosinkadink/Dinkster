"""LTX media conditioning carried through canonical payload references."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import cast

import torch
from dinkster_inference import (
    LTXAV_AUDIO_CHANNELS,
    LTXAV_AUDIO_FREQUENCY_BINS,
    LTXV_VAE_SCALE_FACTORS,
    ConditioningCarrier,
    ConditioningRecord,
    ConditioningSet,
    ExtensionInputValue,
    MultiStreamLatent,
    PayloadBinding,
    PayloadReference,
    make_conditioning_carrier,
)

from .latent_streams import normalize_latent_mask
from .payloads import TensorPayloadError, payload_binding_to_tensor, tensor_to_payload_binding

_GUIDES_KEY = "dinkster-model-ltx/guides"
_REFERENCE_AUDIO_KEY = "dinkster-model-ltx/reference-audio"
_GUIDE_COORDINATES_SPACE = "ltxv-guide-coordinates"
_GUIDE_ATTENTION_MASK_SPACE = "ltxv-guide-attention-mask"
_REFERENCE_AUDIO_SPACE = "ltxav-reference-audio"


class LTXMediaError(ValueError):
    """An LTX media-conditioning contract refusal."""


@dataclass(frozen=True, slots=True)
class LTXVGuideConditioning:
    """One appended classic LTX-Video guide and its target coordinates."""

    keyframe_indices: torch.Tensor
    latent_shape: tuple[int, int, int]
    strength: float
    attention_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        keyframes = self.keyframe_indices
        if (
            type(keyframes) is not torch.Tensor
            or keyframes.layout is not torch.strided
            or keyframes.dtype not in (torch.int32, torch.int64)
            or keyframes.ndim != 4
            or keyframes.shape[0] <= 0
            or keyframes.shape[1] != 3
            or keyframes.shape[3] != 2
        ):
            raise TypeError("LTX-Video guide coordinates must be nonempty integer [B,3,tokens,2]")
        if (
            type(self.latent_shape) is not tuple
            or len(self.latent_shape) != 3
            or any(type(size) is not int or size <= 0 for size in self.latent_shape)
        ):
            raise TypeError("LTX-Video guide latent shape must contain three positive ints")
        if keyframes.shape[2] != math.prod(self.latent_shape):
            raise LTXMediaError("LTX-Video guide coordinates must cover the full guide latent")
        if (
            type(self.strength) is not float
            or not math.isfinite(self.strength)
            or self.strength < 0.0
        ):
            raise LTXMediaError("LTX-Video guide strength must be a nonnegative finite float")
        mask = self.attention_mask
        if mask is not None:
            if (
                type(mask) is not torch.Tensor
                or mask.layout is not torch.strided
                or not mask.is_floating_point()
                or mask.ndim != 5
                or mask.shape[0] != 1
                or mask.shape[1] != 1
                or any(size <= 0 for size in mask.shape)
            ):
                raise TypeError(
                    "LTX-Video guide attention mask must be floating [1,1,frames,height,width]"
                )
            _validate_unit_values(mask, "LTX-Video guide attention mask")

    @property
    def pre_filter_count(self) -> int:
        return math.prod(self.latent_shape)


@dataclass(frozen=True, slots=True)
class LTXVMediaConditioning:
    positive: ConditioningCarrier
    negative: ConditioningCarrier
    latent: MultiStreamLatent[torch.Tensor]
    denoise_mask: MultiStreamLatent[torch.Tensor]


def _validate_unit_values(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise LTXMediaError(f"{name} values must be finite")
    if float(value.amin()) < 0.0 or float(value.amax()) > 1.0:
        raise LTXMediaError(f"{name} values must be within [0, 1]")


def _video_latent(value: object) -> tuple[MultiStreamLatent[torch.Tensor], torch.Tensor]:
    if type(value) is not MultiStreamLatent:
        raise TypeError("LTX-Video media latent must be an exact MultiStreamLatent")
    streams = cast("MultiStreamLatent[torch.Tensor]", value)
    if streams.roles != ("video",):
        raise LTXMediaError("LTX-Video media conditioning requires the exact 'video' role")
    video = streams.by_role("video")
    if (
        type(video) is not torch.Tensor
        or video.layout is not torch.strided
        or not video.is_floating_point()
        or video.ndim != 5
        or any(size <= 0 for size in video.shape)
    ):
        raise TypeError("LTX-Video media latent must be nonempty floating [B,C,T,H,W]")
    return streams, video


def _normalized_video_mask(
    value: torch.Tensor | MultiStreamLatent[torch.Tensor] | None,
    latent: MultiStreamLatent[torch.Tensor],
) -> torch.Tensor:
    video = latent.by_role("video")
    if value is None:
        mask = torch.ones(
            (video.shape[0], 1, video.shape[2], video.shape[3], video.shape[4]),
            device=video.device,
            dtype=torch.float32,
        )
    else:
        mask = normalize_latent_mask(value, latent).by_role("video")[:, :1]
    _validate_unit_values(mask, "LTX-Video denoise mask")
    return mask


def _metadata(record: ConditioningRecord, key: str) -> ExtensionInputValue | None:
    return dict(record.extension_metadata).get(key)


def _record(
    carrier: object, family_id: str | tuple[str, ...]
) -> tuple[ConditioningCarrier, ConditioningRecord]:
    if type(carrier) is not ConditioningCarrier:
        raise TypeError("LTX media conditioning requires exact ConditioningCarrier values")
    typed = carrier
    if len(typed.conditioning.records) != 1:
        raise LTXMediaError("LTX media conditioning requires exactly one conditioning record")
    record = typed.conditioning.records[0]
    layout = record.token_layout
    expected = (family_id,) if isinstance(family_id, str) else family_id
    if layout is None or layout.family_id not in expected:
        families = " or ".join(repr(value) for value in expected)
        raise LTXMediaError(f"LTX media conditioning requires family {families}")
    return typed, record


def _references(value: object) -> set[str]:
    if isinstance(value, PayloadReference):
        return {value.id}
    if isinstance(value, tuple):
        result: set[str] = set()
        for item in value:
            result.update(_references(item))
        return result
    if isinstance(value, Mapping):
        result = set()
        for item in value.values():
            result.update(_references(item))
        return result
    return set()


def _record_references(record: ConditioningRecord) -> set[str]:
    result = {descriptor.reference.id for _, descriptor in record.channels}
    if record.mask is not None:
        result.add(record.mask.payload.id)
    if record.scale_vector is not None:
        result.add(record.scale_vector.values.reference.id)
    for _, value in record.extension_metadata:
        result.update(_references(value))
    return result


def _replace_metadata(
    carrier: ConditioningCarrier,
    record: ConditioningRecord,
    metadata: tuple[tuple[str, object], ...],
    *new_bindings: PayloadBinding,
) -> ConditioningCarrier:
    updated = replace(record, extension_metadata=cast("object", metadata))
    bindings = {binding.reference_id: binding for binding in carrier.bindings}
    for binding in new_bindings:
        bindings[binding.reference_id] = binding
    referenced = _record_references(updated)
    return make_conditioning_carrier(
        ConditioningSet((updated,)),
        tuple(bindings[reference] for reference in sorted(referenced)),
    )


def _with_metadata(
    carrier: ConditioningCarrier,
    record: ConditioningRecord,
    key: str,
    value: object,
    *bindings: PayloadBinding,
) -> ConditioningCarrier:
    existing = tuple((name, item) for name, item in record.extension_metadata if name != key)
    return _replace_metadata(carrier, record, (*existing, (key, value)), *bindings)


def _without_metadata(
    carrier: ConditioningCarrier, record: ConditioningRecord, key: str
) -> ConditioningCarrier:
    metadata = tuple((name, value) for name, value in record.extension_metadata if name != key)
    return _replace_metadata(carrier, record, cast("tuple[tuple[str, object], ...]", metadata))


def _guide_metadata(carrier: object) -> tuple[Mapping[str, object], ...]:
    _, record = _record(carrier, "dinkster.ltxv")
    value = _metadata(record, _GUIDES_KEY)
    if value is None:
        return ()
    if not isinstance(value, tuple) or any(not isinstance(item, Mapping) for item in value):
        raise LTXMediaError("LTX-Video guide metadata is malformed")
    return cast("tuple[Mapping[str, object], ...]", value)


def _decode_ltxv_guides(
    carrier: ConditioningCarrier, entries: tuple[Mapping[str, object], ...]
) -> tuple[LTXVGuideConditioning, ...]:
    bindings = {binding.reference_id: binding for binding in carrier.bindings}
    guides: list[LTXVGuideConditioning] = []
    required = {
        "coordinates",
        "pre_filter_count",
        "frames",
        "height",
        "width",
        "strength",
        "attention_mask",
    }
    for entry in entries:
        if set(entry) != required:
            raise LTXMediaError("LTX-Video guide metadata is malformed")
        coordinates = entry["coordinates"]
        if not isinstance(coordinates, PayloadReference):
            raise LTXMediaError("LTX-Video guide coordinates reference is malformed")
        binding = bindings.get(coordinates.id)
        if binding is None or binding.space != _GUIDE_COORDINATES_SPACE:
            raise LTXMediaError("LTX-Video guide coordinates payload is missing")
        try:
            keyframes = payload_binding_to_tensor(binding)
        except TensorPayloadError as error:
            raise LTXMediaError(
                f"LTX-Video guide coordinates could not be decoded: {error}"
            ) from None
        shape_values = (entry["frames"], entry["height"], entry["width"])
        if any(type(value) is not int or value <= 0 for value in shape_values):
            raise LTXMediaError("LTX-Video guide latent shape is malformed")
        latent_shape = cast("tuple[int, int, int]", shape_values)
        pre_filter_count = entry["pre_filter_count"]
        if type(pre_filter_count) is not int or pre_filter_count != math.prod(latent_shape):
            raise LTXMediaError("LTX-Video guide token count does not match its latent shape")
        strength = entry["strength"]
        if type(strength) is not float:
            raise LTXMediaError("LTX-Video guide strength is malformed")
        raw_mask = entry["attention_mask"]
        attention_mask: torch.Tensor | None = None
        if raw_mask is not None:
            if not isinstance(raw_mask, PayloadReference):
                raise LTXMediaError("LTX-Video guide attention-mask reference is malformed")
            mask_binding = bindings.get(raw_mask.id)
            if mask_binding is None or mask_binding.space != _GUIDE_ATTENTION_MASK_SPACE:
                raise LTXMediaError("LTX-Video guide attention-mask payload is missing")
            try:
                attention_mask = payload_binding_to_tensor(mask_binding)
            except TensorPayloadError as error:
                raise LTXMediaError(
                    f"LTX-Video guide attention mask could not be decoded: {error}"
                ) from None
        guides.append(LTXVGuideConditioning(keyframes, latent_shape, strength, attention_mask))
    return tuple(guides)


def _guide_coordinates(
    guide: torch.Tensor,
    frame_index: int,
    scale_factors: tuple[int, int, int],
    *,
    causal_fix: bool,
) -> torch.Tensor:
    batch, _, frames, height, width = guide.shape
    grid = torch.meshgrid(
        torch.arange(frames, device=guide.device),
        torch.arange(height, device=guide.device),
        torch.arange(width, device=guide.device),
        indexing="ij",
    )
    start = torch.stack(grid, dim=0)
    end = start + 1
    coordinates = torch.stack((start, end), dim=-1).reshape(3, -1, 2)
    coordinates = coordinates.unsqueeze(0).repeat(batch, 1, 1, 1)
    scale = torch.tensor(scale_factors, device=guide.device).view(1, 3, 1, 1)
    coordinates = coordinates * scale
    if causal_fix:
        coordinates[:, 0] = (coordinates[:, 0] + 1 - scale_factors[0]).clamp(min=0)
    coordinates[:, 0] += frame_index
    return coordinates


def ltxv_guides_equal(
    left: tuple[LTXVGuideConditioning, ...],
    right: tuple[LTXVGuideConditioning, ...],
) -> bool:
    if len(left) != len(right):
        return False
    for first, second in zip(left, right, strict=True):
        if (
            first.latent_shape != second.latent_shape
            or first.strength != second.strength
            or not torch.equal(first.keyframe_indices, second.keyframe_indices)
            or (first.attention_mask is None) != (second.attention_mask is None)
        ):
            return False
        if first.attention_mask is not None and not torch.equal(
            first.attention_mask, cast("torch.Tensor", second.attention_mask)
        ):
            return False
    return True


def _validated_guides(
    carrier: ConditioningCarrier, video: torch.Tensor
) -> tuple[tuple[Mapping[str, object], ...], tuple[LTXVGuideConditioning, ...], int]:
    entries = _guide_metadata(carrier)
    guides = _decode_ltxv_guides(carrier, entries)
    if any(guide.latent_shape[1:] != tuple(video.shape[3:]) for guide in guides):
        raise LTXMediaError("LTX-Video guide metadata does not match the latent geometry")
    return entries, guides, sum(guide.latent_shape[0] for guide in guides)


def ltxv_condition_initial_frames(
    latent: MultiStreamLatent[torch.Tensor],
    encoded_frames: torch.Tensor,
    *,
    strength: float = 1.0,
    denoise_mask: torch.Tensor | MultiStreamLatent[torch.Tensor] | None = None,
) -> tuple[MultiStreamLatent[torch.Tensor], MultiStreamLatent[torch.Tensor]]:
    """Replace the initial latent frames and return their sampler mask."""
    streams, video = _video_latent(latent)
    if (
        type(encoded_frames) is not torch.Tensor
        or encoded_frames.layout is not torch.strided
        or not encoded_frames.is_floating_point()
        or encoded_frames.ndim != 5
        or encoded_frames.shape[0] != video.shape[0]
        or encoded_frames.shape[1] != video.shape[1]
        or encoded_frames.shape[3:] != video.shape[3:]
        or not 1 <= encoded_frames.shape[2] <= video.shape[2]
    ):
        raise TypeError("initial LTX-Video frames must match [B,C,frames,H,W] latent geometry")
    if type(strength) is not float or not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise LTXMediaError("initial LTX-Video strength must be a finite float within [0, 1]")
    mask = _normalized_video_mask(denoise_mask, streams)
    result = video.clone()
    result[:, :, : encoded_frames.shape[2]] = encoded_frames.to(
        device=video.device, dtype=video.dtype
    )
    mask = mask.clone()
    mask[:, :, : encoded_frames.shape[2]] = 1.0 - strength
    return streams.replace("video", result), MultiStreamLatent.from_pairs((("video", mask),))


def ltxv_add_guide(
    positive: ConditioningCarrier,
    negative: ConditioningCarrier,
    latent: MultiStreamLatent[torch.Tensor],
    guide_latent: torch.Tensor,
    *,
    frame_index: int,
    strength: float = 1.0,
    denoise_mask: torch.Tensor | MultiStreamLatent[torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    scale_factors: tuple[int, int, int] = LTXV_VAE_SCALE_FACTORS,
    causal_fix: bool | None = None,
) -> LTXVMediaConditioning:
    """Append one ordered guide and map its real-frame target coordinates."""
    positive_carrier, positive_record = _record(positive, "dinkster.ltxv")
    negative_carrier, negative_record = _record(negative, "dinkster.ltxv")
    streams, video = _video_latent(latent)
    if (
        type(guide_latent) is not torch.Tensor
        or guide_latent.layout is not torch.strided
        or not guide_latent.is_floating_point()
        or guide_latent.ndim != 5
        or guide_latent.shape[0] != video.shape[0]
        or guide_latent.shape[1] != video.shape[1]
        or guide_latent.shape[3:] != video.shape[3:]
        or guide_latent.shape[2] <= 0
    ):
        raise TypeError("LTX-Video guide latent must match [B,C,frames,H,W] latent geometry")
    if type(frame_index) is not int:
        raise TypeError("LTX-Video guide frame index must be an int")
    if type(strength) is not float or not math.isfinite(strength) or strength < 0.0:
        raise LTXMediaError("LTX-Video guide strength must be a nonnegative finite float")
    if (
        type(scale_factors) is not tuple
        or len(scale_factors) != 3
        or any(type(value) is not int or value <= 0 for value in scale_factors)
    ):
        raise TypeError("LTX-Video scale factors must contain three positive ints")
    if causal_fix is not None and type(causal_fix) is not bool:
        raise TypeError("LTX-Video causal fix must be a bool when provided")
    positive_entries, positive_guides, existing_guides = _validated_guides(positive_carrier, video)
    negative_entries, negative_guides, _ = _validated_guides(negative_carrier, video)
    if positive_entries != negative_entries or not ltxv_guides_equal(
        positive_guides, negative_guides
    ):
        raise LTXMediaError("positive and negative LTX-Video guides must match")
    generated_frames = video.shape[2] - existing_guides
    if generated_frames <= 0 or guide_latent.shape[2] > generated_frames:
        raise LTXMediaError("LTX-Video guide exceeds the generated latent span")

    time_scale = scale_factors[0]
    resolved_index = frame_index
    if resolved_index < 0:
        resolved_index = max((generated_frames - 1) * time_scale + 1 + resolved_index, 0)
    if (guide_latent.shape[2] > 1 or causal_fix is False) and resolved_index != 0:
        resolved_index = (resolved_index - 1) // time_scale * time_scale + 1
    latent_index = (resolved_index + time_scale - 1) // time_scale
    if latent_index + guide_latent.shape[2] > generated_frames:
        raise LTXMediaError("LTX-Video guide exceeds the generated latent span")
    if causal_fix is None:
        causal_fix = resolved_index == 0 or guide_latent.shape[2] == 1
    coordinates = _guide_coordinates(
        guide_latent, resolved_index, scale_factors, causal_fix=causal_fix
    )

    stored_mask: torch.Tensor | None = None
    if attention_mask is not None:
        if (
            type(attention_mask) is not torch.Tensor
            or attention_mask.layout is not torch.strided
            or not attention_mask.is_floating_point()
            or attention_mask.ndim != 3
            or any(size <= 0 for size in attention_mask.shape)
        ):
            raise TypeError(
                "LTX-Video guide attention mask must be nonempty floating [frames,height,width]"
            )
        _validate_unit_values(attention_mask, "LTX-Video guide attention mask")
        stored_mask = (
            attention_mask.detach().to(device="cpu", dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )

    coordinate_binding = tensor_to_payload_binding(
        f"ltxv-guide-coordinates-{len(positive_entries)}",
        coordinates,
        space=_GUIDE_COORDINATES_SPACE,
    )
    entry: dict[str, object] = {
        "coordinates": PayloadReference(coordinate_binding.reference_id),
        "pre_filter_count": int(coordinates.shape[2]),
        "frames": int(guide_latent.shape[2]),
        "height": int(guide_latent.shape[3]),
        "width": int(guide_latent.shape[4]),
        "strength": strength,
        "attention_mask": None,
    }
    bindings: list[PayloadBinding] = [coordinate_binding]
    if stored_mask is not None:
        mask_binding = tensor_to_payload_binding(
            f"ltxv-guide-attention-{len(positive_entries)}",
            stored_mask,
            space=_GUIDE_ATTENTION_MASK_SPACE,
        )
        entry["attention_mask"] = PayloadReference(mask_binding.reference_id)
        bindings.append(mask_binding)
    entries = (*positive_entries, entry)
    positive_carrier = _with_metadata(
        positive_carrier, positive_record, _GUIDES_KEY, entries, *bindings
    )
    negative_carrier = _with_metadata(
        negative_carrier, negative_record, _GUIDES_KEY, entries, *bindings
    )

    mask = _normalized_video_mask(denoise_mask, streams)
    guide_mask = torch.full(
        (video.shape[0], 1, guide_latent.shape[2], video.shape[3], video.shape[4]),
        max(0.0, 1.0 - strength),
        device=mask.device,
        dtype=mask.dtype,
    )
    result = torch.cat((video, guide_latent.to(device=video.device, dtype=video.dtype)), dim=2)
    result_mask = torch.cat((mask, guide_mask), dim=2)
    return LTXVMediaConditioning(
        positive_carrier,
        negative_carrier,
        streams.replace("video", result),
        MultiStreamLatent.from_pairs((("video", result_mask),)),
    )


def ltxv_crop_guides(
    positive: ConditioningCarrier,
    negative: ConditioningCarrier,
    latent: MultiStreamLatent[torch.Tensor],
    denoise_mask: torch.Tensor | MultiStreamLatent[torch.Tensor],
) -> LTXVMediaConditioning:
    """Remove appended guide frames and their conditioning metadata."""
    positive_carrier, positive_record = _record(positive, ("dinkster.ltxv", "dinkster.ltxav"))
    positive_layout = positive_record.token_layout
    assert positive_layout is not None
    positive_family = positive_layout.family_id
    negative_carrier, negative_record = _record(negative, positive_family)
    streams, video = _video_latent(latent)
    if positive_family == "dinkster.ltxav":
        if any(
            key == _GUIDES_KEY
            for record in (positive_record, negative_record)
            for key, _ in record.extension_metadata
        ):
            raise LTXMediaError("LTX-Video guide metadata requires family 'dinkster.ltxv'")
        entries = negative_entries = ()
        positive_guides = negative_guides = ()
        count = 0
    else:
        entries, positive_guides, count = _validated_guides(positive_carrier, video)
        negative_entries, negative_guides, _ = _validated_guides(negative_carrier, video)
    if entries != negative_entries or not ltxv_guides_equal(positive_guides, negative_guides):
        raise LTXMediaError("positive and negative LTX-Video guides must match")
    if count == 0:
        mask = _normalized_video_mask(denoise_mask, streams)
        return LTXVMediaConditioning(
            positive_carrier,
            negative_carrier,
            streams,
            MultiStreamLatent.from_pairs((("video", mask),)),
        )
    if count >= video.shape[2]:
        raise LTXMediaError("LTX-Video guide metadata exceeds the latent span")
    mask = _normalized_video_mask(denoise_mask, streams)
    positive_carrier = _without_metadata(positive_carrier, positive_record, _GUIDES_KEY)
    negative_carrier = _without_metadata(negative_carrier, negative_record, _GUIDES_KEY)
    return LTXVMediaConditioning(
        positive_carrier,
        negative_carrier,
        streams.replace("video", video[:, :, :-count].clone()),
        MultiStreamLatent.from_pairs((("video", mask[:, :, :-count].clone()),)),
    )


def ltxav_reference_audio_conditioning(
    positive: ConditioningCarrier,
    negative: ConditioningCarrier,
    audio_latent: torch.Tensor,
) -> tuple[ConditioningCarrier, ConditioningCarrier]:
    """Attach one content-owned LTX-2 reference-audio token payload."""
    positive_carrier, positive_record = _record(positive, "dinkster.ltxav")
    negative_carrier, negative_record = _record(negative, "dinkster.ltxav")
    if (
        _metadata(positive_record, _REFERENCE_AUDIO_KEY) is not None
        or _metadata(negative_record, _REFERENCE_AUDIO_KEY) is not None
    ):
        raise LTXMediaError("LTX-2 conditioning already has reference audio")
    if (
        type(audio_latent) is not torch.Tensor
        or audio_latent.layout is not torch.strided
        or not audio_latent.is_floating_point()
        or audio_latent.ndim != 4
        or audio_latent.shape[0] <= 0
        or audio_latent.shape[1] != LTXAV_AUDIO_CHANNELS
        or audio_latent.shape[2] <= 0
        or audio_latent.shape[3] != LTXAV_AUDIO_FREQUENCY_BINS
    ):
        raise TypeError("LTX-2 reference audio must be a nonempty floating [B,8,T,16] latent")
    tokens = audio_latent.permute(0, 2, 1, 3).reshape(
        audio_latent.shape[0], audio_latent.shape[2], audio_latent.shape[1] * audio_latent.shape[3]
    )
    binding = tensor_to_payload_binding(
        "ltxav-reference-audio", tokens, space=_REFERENCE_AUDIO_SPACE
    )
    reference = PayloadReference(binding.reference_id)
    return (
        _with_metadata(
            positive_carrier,
            positive_record,
            _REFERENCE_AUDIO_KEY,
            reference,
            binding,
        ),
        _with_metadata(
            negative_carrier,
            negative_record,
            _REFERENCE_AUDIO_KEY,
            reference,
            binding,
        ),
    )


def materialize_ltxv_guides(
    carrier: ConditioningCarrier,
) -> tuple[ConditioningCarrier, tuple[LTXVGuideConditioning, ...]]:
    """Strip and decode classic guide metadata before text materialization."""
    if type(carrier) is not ConditioningCarrier:
        raise TypeError("LTX-Video conditioning must be an exact ConditioningCarrier")
    typed = carrier
    if len(typed.conditioning.records) != 1:
        return typed, ()
    record = typed.conditioning.records[0]
    unknown = tuple(key for key, _ in record.extension_metadata if key != _GUIDES_KEY)
    if unknown:
        raise LTXMediaError("LTX-Video conditioning has unsupported extension metadata")
    value = _metadata(record, _GUIDES_KEY)
    if value is None:
        return typed, ()
    layout = record.token_layout
    if layout is None or layout.family_id != "dinkster.ltxv":
        raise LTXMediaError("LTX-Video guide metadata requires family 'dinkster.ltxv'")
    if not isinstance(value, tuple) or any(not isinstance(item, Mapping) for item in value):
        raise LTXMediaError("LTX-Video guide metadata is malformed")
    raw_entries = cast("tuple[Mapping[str, object], ...]", value)
    if not raw_entries:
        return typed, ()
    guides = _decode_ltxv_guides(typed, raw_entries)
    return _without_metadata(typed, record, _GUIDES_KEY), guides


def materialize_ltxav_reference_audio(
    carrier: ConditioningCarrier, *, token_width: int
) -> tuple[ConditioningCarrier, torch.Tensor | None]:
    """Strip and decode LTX-2 reference-audio tokens before text materialization."""
    if type(carrier) is not ConditioningCarrier:
        raise TypeError("LTX-2 conditioning must be an exact ConditioningCarrier")
    typed = carrier
    if len(typed.conditioning.records) != 1:
        return typed, None
    record = typed.conditioning.records[0]
    unknown = tuple(key for key, _ in record.extension_metadata if key != _REFERENCE_AUDIO_KEY)
    if unknown:
        raise LTXMediaError("LTX-2 conditioning has unsupported extension metadata")
    value = _metadata(record, _REFERENCE_AUDIO_KEY)
    if value is None:
        return typed, None
    layout = record.token_layout
    if layout is None or layout.family_id != "dinkster.ltxav":
        raise LTXMediaError("LTX-2 reference audio requires family 'dinkster.ltxav'")
    if not isinstance(value, PayloadReference):
        raise LTXMediaError("LTX-2 reference-audio metadata is malformed")
    bindings = {binding.reference_id: binding for binding in typed.bindings}
    binding = bindings.get(value.id)
    if binding is None or binding.space != _REFERENCE_AUDIO_SPACE:
        raise LTXMediaError("LTX-2 reference-audio payload is missing")
    try:
        tokens = payload_binding_to_tensor(binding)
    except TensorPayloadError as error:
        raise LTXMediaError(f"LTX-2 reference audio could not be decoded: {error}") from None
    if (
        not tokens.is_floating_point()
        or tokens.ndim != 3
        or tokens.shape[0] <= 0
        or tokens.shape[1] <= 0
        or tokens.shape[2] != token_width
    ):
        raise LTXMediaError(f"LTX-2 reference audio must be nonempty [B,tokens,{token_width}]")
    return _without_metadata(typed, record, _REFERENCE_AUDIO_KEY), tokens


__all__ = [
    "LTXMediaError",
    "LTXVGuideConditioning",
    "LTXVMediaConditioning",
    "ltxav_reference_audio_conditioning",
    "ltxv_add_guide",
    "ltxv_condition_initial_frames",
    "ltxv_crop_guides",
    "materialize_ltxav_reference_audio",
    "materialize_ltxv_guides",
]
