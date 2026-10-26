"""Native lazy VIDEO dataset transform nodes."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from dinkster_api.v1 import (
    CORE_INT,
    CORE_STRING,
    VIDEO_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    edit_video,
    effective_video_facts,
)

_INT = TypeExpr.concrete(CORE_INT)
_STRINGS = TypeExpr.list_of(TypeExpr.concrete(CORE_STRING))
_VIDEO = TypeExpr.concrete(VIDEO_TYPE)
_VIDEOS = TypeExpr.list_of(_VIDEO)
_SEEDS = TypeExpr.list_of(_INT)


def _seed(seed: list[int]) -> int:
    if not seed:
        raise ValueError("seed list must contain a value")
    return seed[0]


def _frame_facts(video: Mapping[str, object]) -> tuple[int, Fraction]:
    facts = effective_video_facts(video)
    frame_count = facts["frame_count"]
    fps = facts["fps"]
    if not isinstance(frame_count, int) or frame_count < 1:
        raise ValueError("VIDEO requires a known positive frame count")
    if not isinstance(fps, Fraction) or fps <= 0:
        raise ValueError("VIDEO requires a known positive frame rate")
    return frame_count, fps


def _trim(
    video: Mapping[str, object], start_frame: int, length: int, fps: Fraction
) -> dict[str, object]:
    return edit_video(
        video,
        {
            "trim": {
                "start_time": Fraction(start_frame, 1) / fps,
                "duration": Fraction(length, 1) / fps,
            },
            "strict_duration": False,
        },
    )


class VideoTemporalCrop(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="VideoTemporalCrop",
            display_name="Crop Video (Temporal)",
            category="video/transform",
            description="Crop a continuous range of frames from a video.",
            search_terms=("crop", "crop video", "temporal crop", "truncate video"),
            inputs=(
                InputSpec("video", _VIDEO, doc="Input video."),
                InputSpec(
                    "start_frame",
                    _INT,
                    default=0,
                    doc="Starting frame index.",
                    widget=NumberWidget(min=0, max=99999),
                ),
                InputSpec(
                    "length",
                    _INT,
                    default=16,
                    doc="Number of frames to keep.",
                    widget=NumberWidget(min=1, max=99999),
                ),
            ),
            outputs=(OutputSpec("video", _VIDEO, doc="Cropped video (lazy)."),),
        )

    @classmethod
    def execute(
        cls, *, video: Mapping[str, object], start_frame: int, length: int
    ) -> Mapping[str, object]:
        total_frames, fps = _frame_facts(video)
        start = min(start_frame, total_frames - 1)
        return cls.outputs(video=_trim(video, start, min(length, total_frames - start), fps))


class VideoRandomTemporalCrop(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="VideoRandomTemporalCrop",
            display_name="Crop Video (Temporal Random)",
            category="video/transform",
            description="Randomly crop a continuous range of frames from a video.",
            search_terms=("crop", "crop video", "temporal crop", "truncate video", "random crop"),
            inputs=(
                InputSpec("video", _VIDEO, doc="Input video."),
                InputSpec(
                    "length",
                    _INT,
                    default=16,
                    doc="Number of frames to keep.",
                    widget=NumberWidget(min=1, max=99999),
                ),
                InputSpec(
                    "seed",
                    _INT,
                    default=0,
                    doc="Random seed.",
                    widget=NumberWidget(min=0, max=0xFFFFFFFFFFFFFFFF),
                ),
            ),
            outputs=(OutputSpec("video", _VIDEO, doc="Cropped video (lazy)."),),
        )

    @classmethod
    def execute(
        cls, *, video: Mapping[str, object], length: int, seed: int
    ) -> Mapping[str, object]:
        total_frames, fps = _frame_facts(video)
        length = min(length, total_frames)
        max_start = total_frames - length
        numpy = importlib.import_module("numpy")
        start = (
            int(numpy.random.RandomState(seed % (2**32 - 1)).randint(0, max_start + 1))
            if max_start
            else 0
        )
        return cls.outputs(video=_trim(video, start, length, fps))


class ShuffleVideoDataset(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ShuffleVideoDataset",
            display_name="Shuffle Videos List",
            category="video/batch",
            description="Randomly shuffle the order of videos in a list.",
            search_terms=("shuffle", "randomize", "mix"),
            inputs=(
                InputSpec("videos", _VIDEOS, doc="List of videos to shuffle."),
                InputSpec(
                    "seed",
                    _SEEDS,
                    default=[0],
                    doc="Random seed.",
                    widget=NumberWidget(min=0, max=0xFFFFFFFFFFFFFFFF),
                ),
            ),
            outputs=(OutputSpec("videos", _VIDEOS, doc="Shuffled videos"),),
        )

    @classmethod
    def execute(cls, *, videos: list[Any], seed: list[int]) -> Mapping[str, object]:
        numpy = importlib.import_module("numpy")
        indices = numpy.random.RandomState(_seed(seed) % (2**32 - 1)).permutation(len(videos))
        return cls.outputs(videos=[videos[int(index)] for index in indices])


class ShuffleVideoTextDataset(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ShuffleVideoTextDataset",
            display_name="Shuffle Pairs of Video-Text",
            category="video/batch",
            description="Randomly shuffle the order of pairs of video-text in a list.",
            search_terms=("shuffle", "randomize", "mix"),
            inputs=(
                InputSpec("videos", _VIDEOS, doc="List of videos to shuffle."),
                InputSpec("texts", _STRINGS, doc="List of texts to shuffle."),
                InputSpec(
                    "seed",
                    _SEEDS,
                    default=[0],
                    doc="Random seed.",
                    widget=NumberWidget(min=0, max=0xFFFFFFFFFFFFFFFF),
                ),
            ),
            outputs=(
                OutputSpec("videos", _VIDEOS, doc="Shuffled videos"),
                OutputSpec("texts", _STRINGS, doc="Shuffled texts"),
            ),
        )

    @classmethod
    def execute(
        cls, *, videos: list[Any], texts: list[str], seed: list[int]
    ) -> Mapping[str, object]:
        numpy = importlib.import_module("numpy")
        indices = numpy.random.RandomState(_seed(seed) % (2**32 - 1)).permutation(len(videos))
        return cls.outputs(
            videos=[videos[int(index)] for index in indices],
            texts=[texts[int(index)] for index in indices],
        )


DATASET_VIDEO_NODES = (
    VideoTemporalCrop,
    VideoRandomTemporalCrop,
    ShuffleVideoDataset,
    ShuffleVideoTextDataset,
)
