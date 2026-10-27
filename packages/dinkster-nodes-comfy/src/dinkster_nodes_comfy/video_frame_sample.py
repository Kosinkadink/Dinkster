"""Native lazy and bounded VIDEO frame sampling."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from fractions import Fraction
from typing import Any, cast

from dinkster_api.v1 import (
    CORE_COMBO,
    CORE_INT,
    VIDEO_TYPE,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    annotate_image,
    assemble_video,
    copy_media_semantics,
    edit_video,
    effective_video_facts,
    iter_video_pixels,
)

_VIDEO = TypeExpr.concrete(VIDEO_TYPE)
_INT = TypeExpr.concrete(CORE_INT)


class VideoFrameSample(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="VideoFrameSample",
            display_name="Sample Video Frame",
            category="video",
            description="Sample a fixed number of frames from a video using various strategies.",
            search_terms=("sample frames", "extract frames"),
            inputs=(
                InputSpec("video", _VIDEO, doc="Input video."),
                InputSpec(
                    "num_frames",
                    _INT,
                    default=16,
                    widget=NumberWidget(min=1, max=9999),
                    doc="Number of frames to sample.",
                ),
                InputSpec(
                    "strategy",
                    TypeExpr.concrete(CORE_COMBO),
                    default="uniform",
                    widget=ComboWidget(options=("uniform", "head", "tail", "random")),
                    doc=(
                        "uniform: evenly spaced, head: first N, tail: last N, "
                        "random: random sorted."
                    ),
                ),
                InputSpec(
                    "seed",
                    _INT,
                    default=0,
                    widget=NumberWidget(min=0, max=0xFFFFFFFFFFFFFFFF),
                    doc="Random seed (only used with 'random' strategy).",
                ),
            ),
            outputs=(OutputSpec("video", _VIDEO, doc="Sampled video."),),
        )

    @classmethod
    def execute(
        cls, *, video: Mapping[str, object], num_frames: int, strategy: str, seed: int
    ) -> Mapping[str, object]:
        facts = effective_video_facts(video)
        total_frames = facts["frame_count"]
        fps = facts["fps"]
        if not isinstance(total_frames, int) or total_frames < 1:
            raise ValueError("VIDEO requires a known positive frame count")
        if not isinstance(fps, Fraction) or fps <= 0:
            raise ValueError("VIDEO requires a known positive frame rate")
        num_frames = min(num_frames, total_frames)
        if strategy in ("head", "tail"):
            start = 0 if strategy == "head" else total_frames - num_frames
            return cls.outputs(
                video=edit_video(
                    video,
                    {
                        "trim": {
                            "start_time": Fraction(start) / fps,
                            "duration": Fraction(num_frames) / fps,
                        },
                        "strict_duration": False,
                    },
                )
            )
        np = importlib.import_module("numpy")
        if strategy == "uniform":
            indices = (
                [total_frames // 2]
                if num_frames == 1
                else [
                    round(index * (total_frames - 1) / (num_frames - 1))
                    for index in range(num_frames)
                ]
            )
        elif strategy == "random":
            rng = np.random.RandomState(seed % (2**32 - 1))
            indices = sorted(rng.choice(total_frames, size=num_frames, replace=False).tolist())
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        wanted = set(indices)
        last = max(wanted)
        selected: dict[int, Any] = {}
        frames = iter_video_pixels(video)
        try:
            for index, (_timestamp, pixels) in enumerate(frames):
                if index in wanted:
                    selected[index] = pixels.copy()
                if index >= last:
                    break
        finally:
            frames.close()
        images = copy_media_semantics(
            selected[indices[0]], np.stack([selected[i] for i in indices])
        )
        probe = cast("Mapping[str, object]", video["probe"])
        color = {"primaries": probe["primaries"], "transfer": probe["transfer"], "range": 2}
        if probe["matrix"] not in (0, 2):
            color["matrix"] = probe["matrix"]
        if probe["bit_depth"] is not None:
            color["bit_depth"] = probe["bit_depth"]
        return cls.outputs(video=assemble_video(annotate_image(images, color=color), fps=fps))
