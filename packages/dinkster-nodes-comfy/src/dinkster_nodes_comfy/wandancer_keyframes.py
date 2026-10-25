"""WanDancer keyframe placement and matching audio segments."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    AUDIO_TYPE,
    CORE_INT,
    IMAGE_TYPE,
    MASK_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_audio,
    materialize_image,
)

_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_MASK = TypeExpr.concrete(MASK_TYPE)
_AUDIO = TypeExpr.concrete(AUDIO_TYPE)
_INT = TypeExpr.concrete(CORE_INT)


def _pad_keyframes(
    images: Any, segment_length: int, segment_index: int, audio: Mapping[str, Any]
) -> tuple[Any, Any, dict[str, Any]]:
    torch = importlib.import_module("torch")
    shape: tuple[int, int, int, int] = images.shape
    count, height, width, channels = shape
    audio_duration = audio["waveform"].shape[-1] / audio["sample_rate"]
    segment_duration = segment_length / 30
    num_segments = int((audio_duration - 0.2) / segment_duration) + 1 if audio_duration > 0.2 else 0
    total_frames = num_segments * segment_length
    mask = torch.zeros((segment_length, height, width), device=images.device, dtype=images.dtype)
    keyframes = torch.zeros(
        (segment_length, height, width, channels), device=images.device, dtype=images.dtype
    )
    if total_frames > 0 and count > 0:
        frame_interval = float(total_frames) / count
        segment_count = int(math.ceil(total_frames / segment_length))
        images_before = 0
        for previous in range(segment_index):
            end = (
                total_frames - segment_length * previous - 1
                if previous == segment_count - 1
                else segment_length - 1
            )
            consumed = 0
            while consumed * frame_interval < end - frame_interval:
                consumed += 1
            images_before += consumed
        end = (
            total_frames - segment_length * segment_index - 1
            if segment_index == segment_count - 1
            else segment_length - 1
        )
        positions: list[tuple[int, int]] = []
        index = 0
        while index * frame_interval < end - frame_interval:
            positions.append((int(math.ceil(frame_interval * index)), images_before + index))
            index += 1
        positions.append((end, images_before + index))
        valid_positions = [
            (position, index)
            for position, index in positions
            if index < count and position < segment_length
        ]
        if valid_positions:
            frame_positions, image_indices = zip(*valid_positions, strict=True)
            frame_positions = torch.tensor(frame_positions, dtype=torch.long, device=images.device)
            image_indices = torch.tensor(image_indices, dtype=torch.long, device=images.device)
            mask[frame_positions] = 1
            keyframes[frame_positions] = images[image_indices]
    start_time = segment_index * segment_duration
    end_time = min(start_time + segment_duration, audio_duration)
    rate = audio["sample_rate"]
    segment = {
        "waveform": audio["waveform"][:, :, int(start_time * rate) : int(end_time * rate)],
        "sample_rate": rate,
    }
    return keyframes, mask, segment


class WanDancerPadKeyframes(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="WanDancerPadKeyframes",
            category="image/video",
            inputs=(
                InputSpec("images", _IMAGE),
                InputSpec(
                    "segment_length",
                    _INT,
                    default=149,
                    widget=NumberWidget(min=1, max=10000),
                    doc="Length of this segment (usually 149 frames)",
                ),
                InputSpec(
                    "segment_index",
                    _INT,
                    default=0,
                    widget=NumberWidget(min=0, max=100),
                    doc="Which segment this is (0 for first, 1 for second, etc.)",
                ),
                InputSpec(
                    "audio",
                    _AUDIO,
                    doc="Audio to calculate total output frames from and extract segment audio.",
                ),
            ),
            outputs=(
                OutputSpec("keyframes_sequence", _IMAGE, doc="Padded keyframe sequence"),
                OutputSpec("keyframes_mask", _MASK, doc="Mask indicating valid frames"),
                OutputSpec("audio_segment", _AUDIO, doc="Audio segment for this video segment"),
            ),
        )

    @classmethod
    def execute(
        cls, *, images: object, segment_length: int, segment_index: int, audio: object
    ) -> Mapping[str, object]:
        initialize_inference_process()
        frames, mask, segment = _pad_keyframes(
            materialize_image(images), segment_length, segment_index, materialize_audio(audio)
        )
        return cls.outputs(keyframes_sequence=frames, keyframes_mask=mask, audio_segment=segment)


class WanDancerPadKeyframesList(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="WanDancerPadKeyframesList",
            category="image/video",
            inputs=(
                InputSpec("images", _IMAGE),
                InputSpec(
                    "segment_length",
                    _INT,
                    default=149,
                    widget=NumberWidget(min=1, max=10000),
                    doc="Length of each segment (usually 149 frames)",
                ),
                InputSpec(
                    "num_segments",
                    _INT,
                    default=1,
                    widget=NumberWidget(min=1, max=100),
                    doc="How many padded segments to emit as lists.",
                ),
                InputSpec("audio", _AUDIO, doc="Audio to slice for each emitted segment."),
            ),
            outputs=(
                OutputSpec(
                    "keyframes_sequence", TypeExpr.list_of(_IMAGE), doc="Padded keyframe sequences"
                ),
                OutputSpec(
                    "keyframes_mask", TypeExpr.list_of(_MASK), doc="Masks indicating valid frames"
                ),
                OutputSpec(
                    "audio_segment",
                    TypeExpr.list_of(_AUDIO),
                    doc="Audio segment for each video segment",
                ),
            ),
        )

    @classmethod
    def execute(
        cls, *, images: object, segment_length: int, num_segments: int, audio: object
    ) -> Mapping[str, object]:
        initialize_inference_process()
        pixels, sound = materialize_image(images), materialize_audio(audio)
        frames, masks, segments = zip(
            *(
                _pad_keyframes(pixels, segment_length, index, sound)
                for index in range(num_segments)
            ),
            strict=True,
        )
        return cls.outputs(
            keyframes_sequence=list(frames),
            keyframes_mask=list(masks),
            audio_segment=list(segments),
        )


WANDANCER_KEYFRAME_NODES = (WanDancerPadKeyframes, WanDancerPadKeyframesList)
