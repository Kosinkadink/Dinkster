"""Family-neutral temporal mappings and realized timeline guides."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from .latents import MultiStreamLatent

T = TypeVar("T")


class TimelineGuideError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CodecTemporalMapping:
    """Map repeating codec groups and content frames onto one timeline."""

    content_frames_per_latent: tuple[int, ...]
    timeline_numerator: int = 1
    timeline_denominator: int = 1

    def __post_init__(self) -> None:
        groups = self.content_frames_per_latent
        if type(groups) is not tuple or not groups:
            raise TimelineGuideError("codec temporal groups must be a nonempty tuple")
        if any(type(size) is not int or size < 1 for size in groups):
            raise TimelineGuideError("codec temporal groups must be exact positive integers")
        if type(self.timeline_numerator) is not int or self.timeline_numerator < 1:
            raise TimelineGuideError("timeline numerator must be an exact positive integer")
        if type(self.timeline_denominator) is not int or self.timeline_denominator < 1:
            raise TimelineGuideError("timeline denominator must be an exact positive integer")

    def content_extent(self, latent_extent: int) -> int:
        if type(latent_extent) is not int or latent_extent < 1:
            raise TimelineGuideError("latent temporal extent must be an exact positive integer")
        period = self.content_frames_per_latent
        cycles, remainder = divmod(latent_extent, len(period))
        return cycles * sum(period) + sum(period[:remainder])

    def timeline_position(self, content_frame_index: int) -> float:
        if type(content_frame_index) is not int or content_frame_index < 0:
            raise TimelineGuideError("content frame index must be an exact nonnegative integer")
        return (self.timeline_numerator / self.timeline_denominator) * content_frame_index


def resolve_timeline_frame_index(frame_index: int, target_frame_count: int) -> int:
    if type(frame_index) is not int:
        raise TypeError("timeline frame index must be an exact integer")
    if type(target_frame_count) is not int or target_frame_count < 1:
        raise TimelineGuideError("target frame count must be an exact positive integer")
    resolved = frame_index if frame_index >= 0 else target_frame_count + frame_index
    if not 0 <= resolved < target_frame_count:
        raise TimelineGuideError(
            f"frame index {frame_index} is outside the target's {target_frame_count} frames"
        )
    return resolved


@dataclass(frozen=True, slots=True)
class TimelineGuide(Generic[T]):
    """Role-labeled realized latents anchored to a content-frame range."""

    frame_index: int
    frame_count: int
    latent: MultiStreamLatent[T]

    def __post_init__(self) -> None:
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise TimelineGuideError("guide frame index must be an exact nonnegative integer")
        if type(self.frame_count) is not int or self.frame_count < 1:
            raise TimelineGuideError("guide frame count must be an exact positive integer")
        if type(self.latent) is not MultiStreamLatent:
            raise TypeError("guide latent must be an exact MultiStreamLatent")

    def validate_for_target(self, target_frame_count: int) -> None:
        if type(target_frame_count) is not int or target_frame_count < 1:
            raise TimelineGuideError("target frame count must be an exact positive integer")
        if self.frame_index + self.frame_count > target_frame_count:
            raise TimelineGuideError(
                f"guide frames [{self.frame_index}, {self.frame_index + self.frame_count}) "
                f"do not fit the target's {target_frame_count} frames"
            )


__all__ = [
    "CodecTemporalMapping",
    "TimelineGuide",
    "TimelineGuideError",
    "resolve_timeline_frame_index",
]
